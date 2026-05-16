"""안심구역 환경 점검 — 도착 직후 실행.

사용법:
  py -m scripts.check_env
  또는 venv 활성화 후: python -m scripts.check_env
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

REQUIRED = [
    ("numpy", "1.26"),
    ("pandas", "2.1"),
    ("pyarrow", "10.0"),
    ("lightgbm", "3.3"),
    ("sklearn", "1.4"),
    ("yaml", None),
    ("scipy", "1.10"),
]

OPTIONAL = [
    ("shap", "0.42"),
    ("matplotlib", "3.8"),
    ("optuna", "3.0"),
    ("numba", "0.58"),
]


def check_package(name, min_ver=None):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        ok = True
        if min_ver and ver != "?":
            from packaging.version import Version
            try:
                ok = Version(ver) >= Version(min_ver)
            except Exception:
                pass
        status = "OK" if ok else "LOW"
        return status, ver
    except ImportError:
        return "MISSING", None


def main():
    print("=" * 60)
    print("안심구역 환경 점검")
    print("=" * 60)

    # Python 버전
    v = sys.version
    py_ok = sys.version_info >= (3, 9)
    print(f"\n  Python: {v}  {'[OK]' if py_ok else '[WARN] 3.9+ 필요'}")
    print(f"  실행 경로: {sys.executable}")
    print(f"  작업 디렉토리: {os.getcwd()}")
    print(f"  프로젝트 루트: {_PROJECT_ROOT}")

    # venv 확인
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    print(f"  venv 활성화: {'예' if in_venv else '아니오 — venv 먼저 활성화 권장'}")

    # 필수 패키지
    print(f"\n  [필수 패키지]")
    all_ok = True
    for name, min_ver in REQUIRED:
        status, ver = check_package(name, min_ver)
        marker = "[OK]" if status == "OK" else "[!!]"
        ver_str = ver or "미설치"
        print(f"    {marker} {name:15s} {ver_str}")
        if status != "OK":
            all_ok = False

    # 선택 패키지
    print(f"\n  [선택 패키지]")
    for name, min_ver in OPTIONAL:
        status, ver = check_package(name, min_ver)
        marker = "[OK]" if status == "OK" else "[--]"
        ver_str = ver or "미설치"
        print(f"    {marker} {name:15s} {ver_str}")

    # parquet 읽기/쓰기 테스트
    print(f"\n  [PyArrow parquet 테스트]")
    try:
        import pandas as pd
        import numpy as np
        test_df = pd.DataFrame({"a": np.arange(100), "b": np.random.randn(100)})
        test_path = _PROJECT_ROOT / "_env_test.parquet"
        test_df.to_parquet(test_path, index=False, engine="pyarrow")
        reload = pd.read_parquet(test_path)
        assert len(reload) == 100
        test_path.unlink()
        print(f"    [OK] parquet 읽기/쓰기 정상")
    except Exception as e:
        print(f"    [!!] parquet 실패: {e}")
        all_ok = False

    # 디스크 용량
    print(f"\n  [디스크 용량]")
    import shutil
    for drive in set([_PROJECT_ROOT.drive, "E:", "C:"]):
        if not drive:
            continue
        try:
            usage = shutil.disk_usage(drive + "\\")
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            warn = " [경고: 50GB 미만]" if free_gb < 50 else ""
            print(f"    {drive}\\ {free_gb:.1f}GB 여유 / {total_gb:.0f}GB 전체{warn}")
        except Exception:
            pass

    # holidays_kr 확인
    print(f"\n  [외부 데이터]")
    for p in [
        _PROJECT_ROOT / "dsz_bundle" / "external_data" / "holidays_kr.csv",
        _PROJECT_ROOT / "external_data" / "holidays_kr.csv",
        _PROJECT_ROOT / "data" / "holidays_kr.csv",
    ]:
        if p.exists():
            print(f"    [OK] {p.relative_to(_PROJECT_ROOT)}")
            break
    else:
        print(f"    [--] holidays_kr.csv 미발견 (공휴일 미적용)")

    # LP 데이터 경로 탐색
    print(f"\n  [LP 데이터 탐색]")
    lp_candidates = [
        Path(r"E:\lpdata\data.csv"),
        Path(r"E:\LP데이터\data.csv"),
    ]
    for p in lp_candidates:
        if p.exists():
            size_gb = p.stat().st_size / (1024 ** 3)
            print(f"    [발견] {p} ({size_gb:.1f}GB)")

    # 메모리
    print(f"\n  [메모리]")
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        c_ulonglong = ctypes.c_ulonglong

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", c_ulonglong),
                ("ullAvailPhys", c_ulonglong),
                ("ullTotalPageFile", c_ulonglong),
                ("ullAvailPageFile", c_ulonglong),
                ("ullTotalVirtual", c_ulonglong),
                ("ullAvailVirtual", c_ulonglong),
                ("sullAvailExtendedVirtual", c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_gb = stat.ullTotalPhys / (1024 ** 3)
        avail_gb = stat.ullAvailPhys / (1024 ** 3)
        print(f"    총 RAM: {total_gb:.0f}GB, 사용 가능: {avail_gb:.0f}GB")
    except Exception:
        print(f"    (메모리 정보 조회 실패)")

    # 결과
    print(f"\n{'=' * 60}")
    if all_ok:
        print("  환경 점검 통과 — 파이프라인 실행 가능")
    else:
        print("  환경 점검 일부 실패 — 위 로그 확인")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
