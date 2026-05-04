"""LightGBM 학습 + 공공 베이스라인 대비 평가."""
from __future__ import annotations

import argparse
import io as _stdio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd

from src import baselines, eval as ev, io_adapter
from src.models import lgbm as lgbm_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--train-end", default=None, help="예: 2021-12 (포함, M 포맷)")
    ap.add_argument("--save", default="weights/pretrained_lgbm", help="모델 저장 디렉터리")
    ap.add_argument("--compare-baselines", action="store_true")
    args = ap.parse_args()

    print(f"[load] {args.source}")
    df = io_adapter.load_from_yaml(args.source, validate=False)
    if "_temp_c" in df.columns and "temp_c" not in df.columns:
        df = df.rename(columns={"_temp_c": "temp_c"})

    print("[feat] build features")
    features, spec = lgbm_model.build_features(df)
    print(f"       shape={features.shape}  has_weather={spec.has_weather}")

    train_end = pd.Period(args.train_end, freq="M") if args.train_end else None
    print("[train] lightgbm")
    booster, test_preds = lgbm_model.train(features, spec, train_end=train_end)

    print(f"[save] {args.save}")
    lgbm_model.save(
        booster,
        spec,
        args.save,
        notes=f"source={args.source}, train_end={train_end}",
    )

    # 동일 test 구간에서 베이스라인 재평가 (공정 비교)
    daily = ev.daily_by_customer(df)
    monthly = ev.monthly_by_customer(daily)
    horizon_tbl = ev.build_horizon_table(daily, horizons=(10, 20))
    ctx = ev.attach_alarm_context(horizon_tbl, monthly)
    # test 구간 마스크: test_preds 에 있는 (customer, year_month, horizon) 만
    key = test_preds[["customer_id", "year_month", "horizon_days"]].drop_duplicates()
    test_ctx = ctx.merge(key, on=["customer_id", "year_month", "horizon_days"])
    test_horizon = horizon_tbl.merge(
        key, on=["customer_id", "year_month", "horizon_days"]
    )

    results = []
    lgbm_metrics = ev.evaluate(test_preds, test_ctx)
    lgbm_metrics.insert(0, "model", "lightgbm")
    results.append(lgbm_metrics)

    if args.compare_baselines:
        for name, fn in baselines.BASELINES.items():
            preds = fn(monthly, test_horizon)
            m = ev.evaluate(preds, test_ctx)
            m.insert(0, "model", name)
            results.append(m)

    combined = pd.concat(results, ignore_index=True)
    display_cols = [
        "model",
        "horizon_days",
        "contract_type",
        "n",
        "mape_pct",
        "mae",
        "alarm_precision",
        "alarm_recall",
        "alarm_f1",
    ]
    show = [c for c in display_cols if c in combined.columns]
    print()
    print(combined[show].round(3).to_string(index=False))

    Path("eval_results").mkdir(exist_ok=True)
    combined.to_csv(
        "eval_results/lgbm_vs_baselines.csv", index=False, encoding="utf-8-sig"
    )
    print("\n[write] eval_results/lgbm_vs_baselines.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
