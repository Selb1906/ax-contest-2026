"""배치 추론 — 학습된 모델로 신규 데이터 예측.

학습(train)과 분리된 추론(inference) 스크립트.
운영 환경에서는 이 스크립트만 주기적으로 실행.

사용법:
  python -m scripts.predict_batch --source configs/source_dsz.yaml --weights weights/dsz_lgbm/
"""
from __future__ import annotations

import argparse
import io as _stdio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import pandas as pd

from src import io_adapter, eval as ev
from src.preprocess import preprocess
from src.scale import batch_predict, build_features_chunked
from src.tariff import estimate_bill


def main() -> int:
    parser = argparse.ArgumentParser(description="배치 추론")
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", default="weights/dsz_lgbm/")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--output", default="predictions/")
    parser.add_argument("--meter-day", type=int, default=1,
                        help="검침일 (1~28, 기본: 1 = 캘린더 월)")
    args = parser.parse_args()

    t0 = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로딩 + 전처리
    print("[load]", args.source)
    df = io_adapter.load_smart(args.source, validate=False)
    df, prep_log = preprocess(df)
    print(f"  {len(df):,} rows, {df['customer_id'].nunique()} customers")

    # 2. 피처 빌드 (청크)
    print("[features] building...")
    features, spec = build_features_chunked(df, chunk_size=args.chunk_size)

    # 3. 배치 추론
    print("[predict]")
    preds = batch_predict(args.weights, features, chunk_size=10000)

    # 4. 알림 판정
    daily = ev.daily_by_customer(df)
    monthly = ev.monthly_by_customer(daily)
    horizon = ev.build_horizon_table(daily, horizons=(10, 20))
    ctx = ev.attach_alarm_context(horizon, monthly)

    alarm_merged = preds.merge(ctx, on=["customer_id", "year_month", "horizon_days"], how="left")
    alarm_merged["alarm_prev"] = alarm_merged["pred_monthly_kwh"] > alarm_merged["prev_month_kwh"] * 1.30
    alarm_merged["alarm_yoy"] = alarm_merged["pred_monthly_kwh"] > alarm_merged.get("yoy_month_kwh", pd.Series(dtype=float)) * 1.30
    alarm_merged["alarm_ma3"] = alarm_merged["pred_monthly_kwh"] > alarm_merged.get("ma3_kwh", pd.Series(dtype=float)) * 1.50
    alarm_merged["alarm_any"] = (
        alarm_merged["alarm_prev"].fillna(False)
        | alarm_merged["alarm_yoy"].fillna(False)
        | alarm_merged["alarm_ma3"].fillna(False)
    )

    # 5. 요금 환산
    print(f"[billing] meter_day={args.meter_day}", flush=True)
    bills = []
    for _, row in alarm_merged.iterrows():
        ct = row.get("contract_type", "일반용")
        month = row["year_month"].month if hasattr(row["year_month"], "month") else 1
        bill = estimate_bill(row["pred_monthly_kwh"], ct, month, meter_day=args.meter_day)
        bills.append(bill["total_won"])
    alarm_merged["pred_bill_won"] = bills

    # 6. 저장
    alarm_merged.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    alarm_customers = alarm_merged[alarm_merged["alarm_any"]]
    alarm_customers.to_csv(out / "alarm_customers.csv", index=False, encoding="utf-8-sig")

    elapsed = time.time() - t0
    n_alarm = alarm_merged["alarm_any"].sum()
    print(f"\n[done] {elapsed:.1f}s")
    print(f"  총 예측: {len(alarm_merged):,}")
    print(f"  알림 대상: {n_alarm:,} ({n_alarm/len(alarm_merged)*100:.1f}%)")
    print(f"  저장: {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
