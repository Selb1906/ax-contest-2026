"""안심구역 반입용 번들 빌드 — wheel 다운로드 + 코드 패킹.

사용법:
  python scripts/build_bundle.py              # wheel 다운로드 + zip 생성
  python scripts/build_bundle.py --skip-download  # wheel 이미 있으면 코드만 재패킹
  python scripts/build_bundle.py --check      # 누락 패키지만 확인

안심구역 환경:
  - Python 3.9 (cp39)
  - Windows 64bit + Linux x86_64 양쪽 대비
  - 인터넷 없음 → 모든 의존성을 wheel로 반입
"""
from __future__ import annotations

import argparse
import io as _stdio
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = _stdio.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "dsz_bundle"
WHEELS_DIR = BUNDLE_DIR / "wheels"
CODE_DIR = BUNDLE_DIR / "code"
ZIP_OUT = ROOT / "dsz_bundle_submit.zip"

# 안심구역 대상 Python 버전
TARGET_PYTHON = "3.9"

# 핵심 패키지 목록 (직접 사용하는 것만 — 의존성은 pip이 자동 해결)
CORE_PACKAGES = [
    "numpy>=1.26,<2.1",
    "pandas>=2.1,<2.4",
    "pyarrow>=15,<22",
    "scipy>=1.11,<1.14",
    "scikit-learn>=1.4,<1.7",
    "pyyaml>=6",
    "lightgbm>=3.3,<4.7",
    "shap>=0.42,<0.50",
    "openpyxl>=3.1",
    "optuna>=3.0,<4.0",
    "matplotlib>=3.8,<3.10",
    "tqdm>=4.60",
]

# 플랫폼별 다운로드 설정
PLATFORMS = [
    {"platform": "win_amd64", "label": "Windows x64"},
    {"platform": "manylinux2014_x86_64", "label": "Linux x64"},
]

# 코드에 포함할 디렉토리/파일
CODE_INCLUDES = [
    "src/",
    "scripts/",
    "configs/",
    "ui/",
    "run.py",
    "requirements.txt",
    ".gitignore",
]

# 외부 데이터 (이미 dsz_bundle/external_data/ 에 있는 것)
EXTERNAL_DATA_DIR = BUNDLE_DIR / "external_data"


def download_wheels(platforms: list[dict] | None = None):
    """pip download로 wheel 다운로드 (cp39, 지정 플랫폼)."""
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)

    if platforms is None:
        platforms = PLATFORMS

    for plat in platforms:
        print(f"\n{'='*60}")
        print(f"  다운로드: {plat['label']} (cp{TARGET_PYTHON.replace('.', '')})")
        print(f"{'='*60}")

        for pkg in CORE_PACKAGES:
            cmd = [
                sys.executable, "-m", "pip", "download",
                "--dest", str(WHEELS_DIR),
                "--python-version", TARGET_PYTHON,
                "--only-binary", ":all:",
                "--platform", plat["platform"],
                "--no-cache-dir",
                pkg,
            ]
            print(f"  → {pkg} ...", end=" ", flush=True)
            result = subprocess.run(
                cmd, capture_output=True, text=True
            )
            if result.returncode == 0:
                print("OK")
            else:
                # pure python wheel은 플랫폼 무관 — 이미 있을 수 있음
                if "already satisfied" in result.stdout.lower() or \
                   "already" in result.stderr.lower():
                    print("(이미 존재)")
                else:
                    print(f"WARN: {result.stderr.strip()[:100]}")

    # 순수 Python wheel (any 플랫폼) — 한번만
    print(f"\n{'='*60}")
    print(f"  다운로드: Pure Python wheels")
    print(f"{'='*60}")
    for pkg in CORE_PACKAGES:
        cmd = [
            sys.executable, "-m", "pip", "download",
            "--dest", str(WHEELS_DIR),
            "--python-version", TARGET_PYTHON,
            "--only-binary", ":all:",
            "--platform", "any",
            "--no-cache-dir",
            pkg,
        ]
        subprocess.run(cmd, capture_output=True, text=True)

    # 중복 제거 (같은 패키지의 다른 버전)
    _deduplicate_wheels()


def _deduplicate_wheels():
    """같은 패키지의 중복 버전 제거 (최신만 유지)."""
    from collections import defaultdict
    wheels = defaultdict(list)
    for whl in WHEELS_DIR.glob("*.whl"):
        # wheel filename: name-version-pyver-abi-platform.whl
        parts = whl.stem.split("-")
        if len(parts) >= 2:
            pkg_name = parts[0].lower().replace("_", "-")
            platform = parts[-1] if len(parts) >= 5 else "any"
            key = (pkg_name, platform)
            wheels[key].append(whl)

    removed = 0
    for key, files in wheels.items():
        if len(files) > 1:
            # 수정 시간 기준 최신 유지
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            for old in files[1:]:
                old.unlink()
                removed += 1
    if removed:
        print(f"  중복 wheel {removed}개 제거")


def copy_code():
    """최신 소스 코드��� 번들에 복사."""
    if CODE_DIR.exists():
        shutil.rmtree(CODE_DIR)
    CODE_DIR.mkdir(parents=True)

    for item in CODE_INCLUDES:
        src = ROOT / item
        dst = CODE_DIR / item
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True,
                           ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    print(f"  코드 복사 완료 → {CODE_DIR}")


def build_zip():
    """dsz_bundle/ 전체를 zip으로 패킹."""
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()

    print(f"\n  ZIP 생성 중: {ZIP_OUT.name}")
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(BUNDLE_DIR.rglob("*")):
            if f.is_file():
                arcname = f.relative_to(BUNDLE_DIR)
                zf.write(f, arcname)

    size_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
    print(f"  완료: {ZIP_OUT.name} ({size_mb:.1f} MB)")


def check_missing():
    """현재 wheel 디렉토리에서 누락된 패키지 확인."""
    existing = {whl.stem.split("-")[0].lower().replace("_", "-")
                for whl in WHEELS_DIR.glob("*.whl")} if WHEELS_DIR.exists() else set()

    # 패키지명 추출 (버전 제거)
    needed = set()
    for pkg in CORE_PACKAGES:
        name = pkg.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip()
        needed.add(name.lower().replace("_", "-"))

    # matplotlib 의존성 수동 체크
    matplotlib_deps = {
        "pillow", "kiwisolver", "fonttools", "cycler",
        "contourpy", "pyparsing", "packaging",
    }
    optuna_deps = {
        "sqlalchemy", "alembic", "cmaes", "colorlog",
    }
    needed |= matplotlib_deps | optuna_deps

    missing = needed - existing
    present = needed & existing

    print(f"\n  확인된 wheel: {len(existing)}개")
    print(f"  필요 패키지 (직접+주요 의존성): {len(needed)}개")
    print(f"  ✅ 있음: {len(present)}개")
    if missing:
        print(f"  ❌ 누락: {len(missing)}개")
        for m in sorted(missing):
            print(f"     - {m}")
    else:
        print(f"  ✅ 모두 존재!")
    return missing


def update_setup_md():
    """SETUP.md를 최신 패키지 목록으로 갱신."""
    wheels = sorted(WHEELS_DIR.glob("*.whl")) if WHEELS_DIR.exists() else []
    pkg_names = sorted(set(
        whl.stem.split("-")[0].replace("_", "-") for whl in wheels
    ))

    content = f"""# 안심구역 환경 설정 가이드

## 1. 패키지 설치 (오프라인)

```bash
# 전체 설치 (의존성 순서 자동 해결)
pip install --no-index --find-links=wheels/ --no-deps wheels/*.whl

# 또는 핵심 패키지만 (의존성 포함 설치)
pip install --no-index --find-links=wheels/ \\
    numpy pandas pyarrow scipy scikit-learn lightgbm \\
    shap matplotlib optuna pyyaml openpyxl tqdm
```

포함된 wheel ({len(wheels)}개): {', '.join(pkg_names)}

## 2. 환경 검증

```bash
# 필수 패키지 임포트 테스트
python -c "
import numpy, pandas, pyarrow, scipy, sklearn, lightgbm, shap
import matplotlib, optuna, yaml, openpyxl
print('All imports OK')
"
```

## 3. 코드 설치

```bash
# 번들에서 코드 복사 (또는 code/ 디렉토리를 작업 디렉토리로 사용)
cp -r code/* /path/to/workspace/
cd /path/to/workspace/
```

## 4. 데이터 소스 설정

```bash
# 1) LP 데이터 구조 탐색 (최초 1회)
python -m scripts.inspect_lp --path /path/to/lp/data

# 2) 자동 생성된 configs/source_dsz.yaml 확인·수정
```

## 5. 실행

```bash
# 대화형 실행기 (팀 작업 분배 포함)
python run.py

# 또는 원스톱 파이프라인 직접 실행
python -m scripts.run_full_analysis --source configs/source_dsz.yaml --train-end 2023-12
```

## 6. 반출 대상 파일

| 디렉토리 | 내용 | 반출 가능 |
|----------|------|-----------|
| `profile_stats/dsz/` | 집계 통계 (MANIFEST.json) | ✅ |
| `btm_results/btm_summary.json` | BTM 집계 | ✅ |
| `eval_results/` | 모델 평가 결과 (MAPE 등) | ✅ |
| `sliding_results/` | 검침일별 상세 평가 | ✅ |
| `explain_results/` | SHAP 피처 중요도 | ✅ |
| `weights/` | 학습된 모델 가중치 | ✅ |
| `btm_results/btm_flags.csv` | 고객별 BTM 플래그 | ❌ (고객ID 포함) |

## 7. 트러블슈팅

- **한글 깨짐**: `matplotlib` 폰트 → `Malgun Gothic` (Windows) / `NanumGothic` (Linux)
- **메모리 부족**: `--meter-days 1,15` 로 검침일 축소, `--skip-shap` 옵션
- **중단 후 재개**: `--resume` 플래그로 체크포인트에서 이어서 실행
"""
    setup_path = BUNDLE_DIR / "SETUP.md"
    setup_path.write_text(content, encoding="utf-8")
    print(f"  SETUP.md 갱신 완료")


def main():
    parser = argparse.ArgumentParser(description="안심구�� 반입 번들 빌드")
    parser.add_argument("--skip-download", action="store_true",
                        help="wheel 다운로드 건너뛰기")
    parser.add_argument("--check", action="store_true",
                        help="누락 패키지만 확인")
    parser.add_argument("--no-zip", action="store_true",
                        help="zip 생성 건너뛰기")
    args = parser.parse_args()

    if args.check:
        check_missing()
        return

    print("""
╔══════════════════════════════════════════════════════╗
║     안심구역 반입 번들 빌드                          ║
╚══════════════════════════════════════════════════════╝""")
    print(f"  대상 Python: {TARGET_PYTHON}")
    print(f"  플랫폼: {', '.join(p['label'] for p in PLATFORMS)}")
    print(f"  번들 디렉토리: {BUNDLE_DIR}")

    # 1. Wheel 다운로드
    if not args.skip_download:
        download_wheels()
    else:
        print("\n  [skip] wheel 다운로드")

    # 2. 누락 체크
    missing = check_missing()

    # 3. 코드 복사
    copy_code()

    # 4. SETUP.md — 수동 관리 (덮어쓰지 않음)
    setup_path = BUNDLE_DIR / "SETUP.md"
    if setup_path.exists():
        print(f"  SETUP.md 유지 (수동 관리)")
    else:
        print(f"  ⚠️  SETUP.md 없음 — dsz_bundle/SETUP.md를 직접 작성하세요")

    # 5. ZIP 생성
    if not args.no_zip:
        build_zip()

    print(f"\n{'='*60}")
    print(f"  번들 빌드 완료!")
    if missing:
        print(f"  ⚠️  누락 패키지 {len(missing)}개 — 수동 확��� 필요")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
