"""end-to-end 스모크 테스트.

1. synth → parquet 로드, 스키마 검증
2. 기본 통계 출력 (알림 조건 경계값 감각 확인용)
3. public_apt 소스도 로드 시도 (3년치 CSV 가 있을 때만)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io as _stdio

# Windows 콘솔에서 한글 출력 안 깨지게
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import pandas as pd

from src import io_adapter, schemas


def smoke_synth() -> pd.DataFrame:
    print("=" * 60)
    print("[1] synth 로드")
    df = io_adapter.load_from_yaml("configs/source_synth.yaml")
    rep = schemas.validate(df)
    print(f"  rows       : {rep.n_rows:,}")
    print(f"  customers  : {rep.n_customers}")
    print(f"  ts range   : {rep.ts_min} ~ {rep.ts_max}")
    print(f"  contracts  : {rep.contract_type_counts}")
    print(f"  optional   : {rep.present_optional}")
    print()
    print("[1-b] 기본 통계 (유효전력 kWh/15min)")
    stats = df.groupby(schemas.CONTRACT_TYPE)[schemas.P_ACTIVE_KWH].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    print(stats.to_string())
    print()

    # 월 집계로 알림 조건 부합 여부 감 잡기
    df = df.copy()
    df["month"] = df[schemas.TS].dt.to_period("M")
    monthly = (
        df.groupby([schemas.CUSTOMER_ID, schemas.CONTRACT_TYPE, "month"])[
            schemas.P_ACTIVE_KWH
        ].sum().reset_index(name="monthly_kwh")
    )
    monthly = monthly.sort_values([schemas.CUSTOMER_ID, "month"])
    monthly["prev_m"] = monthly.groupby(schemas.CUSTOMER_ID)["monthly_kwh"].shift(1)
    monthly["pct_vs_prev"] = (monthly["monthly_kwh"] / monthly["prev_m"] - 1.0) * 100
    trigger_rate = (monthly["pct_vs_prev"] > 30).mean()
    print(
        f"[1-c] 합성 데이터상 '전월 대비 +30% 초과' 월 발생률: "
        f"{trigger_rate:.2%} (고객×월 단위)"
    )
    return df


def smoke_public_apt() -> None:
    print("=" * 60)
    print("[2] public_apt 로드 (샘플)")
    try:
        df = io_adapter.load_from_yaml("configs/source_public_apt.yaml")
    except FileNotFoundError as e:
        print(f"  skip (파일 없음): {e}")
        return
    rep = schemas.validate(df)
    print(f"  rows       : {rep.n_rows:,}")
    print(f"  customers  : {rep.n_customers} (격자 단위)")
    print(f"  ts range   : {rep.ts_min} ~ {rep.ts_max}")
    print("  기온 분포 (°C) :")
    print(df["_temp_c"].describe().to_string())


if __name__ == "__main__":
    smoke_synth()
    smoke_public_apt()
    print("=" * 60)
    print("OK")
