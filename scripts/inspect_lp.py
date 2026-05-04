"""안심구역 첫 실행 — LP 데이터 구조 탐색.

실 LP 파일의 구조를 파악하고 source_dsz.yaml을 자동 생성.
이 스크립트를 가장 먼저 실행.

사용법:
  python -m scripts.inspect_lp --path /path/to/lp/data
  python -m scripts.inspect_lp --path /path/to/lp/file.csv
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
        "customer_id": ["고객", "cust", "id", "번호", "호수"],
        "ts": ["일시", "검침", "날짜", "시간", "date", "time", "dt", "metering"],
        "contract_type": ["계약", "종별", "용도", "contract", "type", "구분"],
        "p_active_kwh": ["유효", "active", "전력량", "kwh", "사용량", "소비"],
        "p_reactive_kwh": ["무효", "reactive", "kvarh", "무효전력"],
        "max_demand_kw": ["최대수요", "max_demand", "피크", "peak", "최대전력"],
        "n_households": ["세대", "household", "가구", "호수"],
        "contract_power_kw": ["계약전력", "contract_power", "계약용량"],
        "region_code": ["지역", "region", "시도", "행정", "area"],
        "industry_code": ["업종", "industry", "산업"],
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


def main() -> int:
    parser = argparse.ArgumentParser(description="LP 데이터 구조 탐색")
    parser.add_argument("--path", required=True, help="LP 파일 또는 디렉터리 경로")
    args = parser.parse_args()

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
