"""안심구역 첫 실행 — LP 데이터 구조 탐색.

실 LP 파일의 구조를 파악하고 source_dsz.yaml을 자동 생성.
이 스크립트를 가장 먼저 실행.

사용법:
  python -m scripts.inspect_lp --path /path/to/lp/data
  python -m scripts.inspect_lp --path /path/to/lp/file.csv
  python -m scripts.inspect_lp --scan          # LP 파일 자동 탐색
  python -m scripts.inspect_lp --scan --root D:/   # 특정 드라이브부터 탐색
"""
from __future__ import annotations

import argparse
import io as _stdio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import yaml


def try_read(path: Path, nrows: int = 5) -> tuple[pd.DataFrame | None, str]:
    """여러 인코딩·형식으로 읽기 시도."""
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path), "parquet"

    if suffix in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(path, nrows=nrows)
            if len(df.columns) > 2:
                return df, f"excel({suffix})"
        except Exception:
            pass

    # CSV 계열 — 여러 인코딩·구분자 시도
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        for sep in [",", "\t", "|", ";"]:
            try:
                df = pd.read_csv(path, nrows=nrows, encoding=enc, sep=sep)
                if len(df.columns) > 2:
                    return df, f"csv({enc}, sep='{sep}')"
            except Exception:
                continue
    return None, "failed"


def guess_column_map(columns: list[str]) -> dict[str, str]:
    """컬럼명에서 표준 매핑 추측."""
    patterns = {
        "customer_id": ["계약번호", "계기번호", "고객", "cust", "id", "번호"],
        "ts_date": ["검침년월일", "검침일자"],
        "ts_time": ["검침시분"],
        "ts": ["일시", "검침", "날짜", "date", "time", "dt", "metering"],
        "contract_type": ["계약종별", "계약", "종별", "용도", "contract"],
        "p_active_kwh": ["유효전력량계", "유효", "active", "전력량", "kwh", "사용량"],
        "p_reactive_lag": ["지상무효전력량계", "지상무효"],
        "p_reactive_lead": ["진상무효전력량계", "진상무효"],
        "p_reactive_kwh": ["무효전력량계", "무효", "reactive", "kvarh"],
        "p_apparent_kwh": ["피상전력량계", "피상", "apparent"],
        "max_demand_kw": ["최대수요", "max_demand", "피크", "peak"],
        "n_households": ["세대", "household", "가구"],
        "contract_power_kw": ["계약전력", "contract_power"],
        "region_code": ["본부", "지역", "region", "시도"],
        "region_sub": ["지사"],
        "supply_method": ["공급방식"],
        "usage_purpose": ["전기사용용도", "사용용도"],
        "industry_code": ["산업분류", "업종", "industry"],
        "meter_interval": ["검침주기"],
    }
    mapping = {}
    for std_name, keywords in patterns.items():
        for col in columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in keywords):
                if std_name not in mapping:
                    mapping[std_name] = col
                    break
    return mapping


def scan_for_lp(root: str | None = None) -> list[Path]:
    """파일 시스템에서 LP 데이터로 추정되는 파일/디렉토리를 탐색."""
    import os
    import platform

    candidates = []

    # 탐색 시작점 결정
    if root:
        roots = [Path(root)]
    elif platform.system() == "Windows":
        roots = [Path(f"{d}:/") for d in "DEFCGHIJ" if Path(f"{d}:/").exists()]
    else:
        roots = [Path("/data"), Path("/home"), Path("/mnt"), Path("/")]

    # LP 데이터 특성: CSV/parquet, 큰 용량, 한글 or LP 관련 이름
    lp_keywords = ["lp", "load", "전력", "검침", "사용량", "ami", "meter", "고객"]
    skip_dirs = {"Windows", "Program Files", "Program Files (x86)",
                 "$Recycle.Bin", "System Volume Information",
                 ".git", "__pycache__", "node_modules", "venv", ".venv"}

    print(f"  탐색 시작: {[str(r) for r in roots]}")
    print(f"  키워드: {lp_keywords}")
    print()

    for start in roots:
        if not start.exists():
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(start):
                # 시스템/불필요 디렉토리 건너뛰기
                dirnames[:] = [d for d in dirnames if d not in skip_dirs]

                dp = Path(dirpath)
                # 너무 깊으면 건너뛰기
                if len(dp.parts) > 8:
                    dirnames.clear()
                    continue

                for fn in filenames:
                    fl = fn.lower()
                    # CSV/parquet/xlsx 만 대상
                    if not any(fl.endswith(ext) for ext in (".csv", ".parquet", ".xlsx", ".xls")):
                        continue
                    # 키워드 매치 또는 큰 파일(>10MB)
                    fpath = dp / fn
                    name_match = any(kw in fl or kw in str(dp).lower() for kw in lp_keywords)
                    try:
                        size_mb = fpath.stat().st_size / (1024 * 1024)
                    except OSError:
                        continue
                    if name_match or size_mb > 10:
                        candidates.append((fpath, size_mb, name_match))
                        if len(candidates) >= 50:
                            break
                if len(candidates) >= 50:
                    break
        except PermissionError:
            continue

    # 정렬: 키워드 매치 우선, 그 다음 크기 순
    candidates.sort(key=lambda x: (-x[2], -x[1]))
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="LP 데이터 구조 탐색")
    parser.add_argument("--path", default=None, help="LP 파일 또는 디렉터리 경로")
    parser.add_argument("--scan", action="store_true",
                        help="파일 시스템에서 LP 데이터 자동 탐색")
    parser.add_argument("--root", default=None,
                        help="탐색 시작 경로 (--scan과 함께 사용)")
    args = parser.parse_args()

    if not args.path and not args.scan:
        print("사용법:")
        print("  python -m scripts.inspect_lp --path <경로>   # 경로 직접 지정")
        print("  python -m scripts.inspect_lp --scan          # 자동 탐색")
        print("  python -m scripts.inspect_lp --scan --root D:/")
        return 1

    # --scan 모드: LP 파일 탐색
    if args.scan:
        print(f"\n{'='*60}")
        print("LP 데이터 자동 탐색")
        print(f"{'='*60}\n")

        candidates = scan_for_lp(args.root)

        if not candidates:
            print("  LP 데이터로 추정되는 파일을 찾지 못했습니다.")
            print("  → 직원에게 데이터 경로를 문의하세요.")
            return 1

        print(f"\n  후보 {len(candidates)}건 발견:\n")
        print(f"  {'#':>3s}  {'크기':>8s}  {'키워드':>4s}  경로")
        print(f"  {'─'*3}  {'─'*8}  {'─'*4}  {'─'*50}")
        for i, (fpath, size_mb, matched) in enumerate(candidates[:20]):
            marker = "●" if matched else " "
            print(f"  {i+1:>3d}  {size_mb:>7.1f}M  {marker:>4s}  {fpath}")

        if not args.path:
            print(f"\n  → 위 목록에서 확인 후 다시 실행:")
            print(f"     python -m scripts.inspect_lp --path <선택한 경로>")
            return 0

    target = Path(args.path)

    # 파일 목록
    if target.is_dir():
        files = sorted(
            list(target.glob("*.csv")) + list(target.glob("*.CSV"))
            + list(target.glob("*.parquet"))
            + list(target.glob("*.xlsx")) + list(target.glob("*.xls"))
        )
        print(f"[dir] {target}")
        print(f"  파일 수: {len(files)}")
        for f in files[:10]:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name:40s} {size_mb:>8.1f} MB")
        if len(files) > 10:
            print(f"  ... 외 {len(files) - 10}개")
    else:
        files = [target]

    if not files:
        print("[error] 파일을 찾을 수 없습니다")
        return 1

    # 첫 파일 분석
    sample_file = files[0]
    print(f"\n[read] {sample_file.name}")
    df, fmt = try_read(sample_file, nrows=10)
    if df is None:
        print("[error] 읽기 실패 — 인코딩이나 형식을 확인하세요")
        return 1

    print(f"  포맷: {fmt}")
    print(f"  컬럼 수: {len(df.columns)}")
    print(f"\n  컬럼 목록:")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        sample = df[col].iloc[0] if len(df) > 0 else "N/A"
        null_rate = df[col].isna().mean()
        print(f"    {i:>3d}. {col:30s} {str(dtype):>10s}  sample={sample}  null={null_rate:.0%}")

    # 전체 파일 행 수 추정
    if fmt != "parquet":
        full_df = pd.read_csv(sample_file, encoding=fmt.split("(")[1].split(",")[0] if "(" in fmt else "utf-8")
        print(f"\n  전체 행 수 (이 파일): {len(full_df):,}")
        print(f"  고유값 수 (첫 컬럼): {full_df.iloc[:, 0].nunique():,}")

    # 컬럼 매핑 추측
    print(f"\n{'='*60}")
    print("자동 매핑 추측 결과:")
    print(f"{'='*60}")
    col_map = guess_column_map(df.columns.tolist())
    for std, orig in col_map.items():
        print(f"  {std:20s} ← {orig}")

    # 매핑 안 된 표준 컬럼
    all_std = ["customer_id", "ts", "contract_type", "p_active_kwh", "p_reactive_kwh", "max_demand_kw"]
    missing = [s for s in all_std if s not in col_map]
    if missing:
        print(f"\n  ⚠ 매핑 실패: {missing}")
        print("    → source_dsz.yaml에서 수동으로 매핑 필요")

    # 시간 형식 경고
    if "ts" in col_map:
        ts_col = col_map["ts"]
        ts_sample = str(df[ts_col].iloc[0]) if len(df) > 0 else ""
        print(f"\n  시간 형식 확인: '{ts_sample}'")
        if "24:00" in ts_sample or "2400" in ts_sample:
            print("  ⚠ 24시 형식 감지 — io_adapter에서 자동 보정됨")
        if any(str(df[ts_col].iloc[i]).find(" 01:") >= 0 for i in range(min(5, len(df)))):
            first_hours = [str(df[ts_col].iloc[i]) for i in range(min(24, len(df)))]
            if not any("00:" in h or " 0:" in h for h in first_hours):
                print("  ⚠ 01~24시 형식 의심 — io_adapter에서 자동 보정 시도")

    # source_dsz.yaml 자동 생성
    dsz_config = {
        "source": {
            "kind": "dsz_lp",
            "path": str(target),
            "column_map": col_map,
            "contract_type_default": None,
        }
    }
    out_path = Path("configs/source_dsz.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(dsz_config, f, allow_unicode=True, default_flow_style=False)
    print(f"\n[saved] {out_path}")
    print("  → 매핑을 확인하고 필요시 수정 후 파이프라인 실행")

    # 데이터 샘플 출력
    print(f"\n{'='*60}")
    print("데이터 샘플 (상위 5행):")
    print(f"{'='*60}")
    print(df.head().to_string())

    # 기본 통계
    if len(df.columns) > 2:
        print(f"\n{'='*60}")
        print("수치형 컬럼 기초 통계:")
        print(f"{'='*60}")
        print(df.describe().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
