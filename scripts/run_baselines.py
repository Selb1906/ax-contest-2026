"""베이스라인 모델들을 한 소스에 대해 일괄 평가."""
from __future__ import annotations

import argparse
import io as _stdio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd

from src import baselines, eval as ev, io_adapter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--horizons", default="10,20", help="쉼표 구분 N일 목록")
    ap.add_argument("--out", default=None, help="결과 CSV 경로 (생략 시 출력만)")
    ap.add_argument("--models", default=",".join(baselines.BASELINES))
    args = ap.parse_args()

    print(f"[load] {args.source}")
    df = io_adapter.load_from_yaml(args.source, validate=False)
    if "_temp_c" in df.columns and "temp_c" not in df.columns:
        df = df.rename(columns={"_temp_c": "temp_c"})

    horizons = [int(x) for x in args.horizons.split(",")]
    print(f"[build] horizon table (horizons={horizons})")
    daily = ev.daily_by_customer(df)
    monthly = ev.monthly_by_customer(daily)
    horizon_tbl = ev.build_horizon_table(daily, horizons=horizons)
    ctx = ev.attach_alarm_context(horizon_tbl, monthly)
    print(
        f"       eval samples = {len(ctx):,}  (customers={ctx['customer_id'].nunique()}, "
        f"months={ctx['year_month'].nunique()})"
    )

    all_rows = []
    for name in args.models.split(","):
        name = name.strip()
        if name not in baselines.BASELINES:
            print(f"[skip] unknown model: {name}")
            continue
        print(f"[eval] {name}")
        preds = baselines.BASELINES[name](monthly, horizon_tbl)
        metrics = ev.evaluate(preds, ctx)
        metrics.insert(0, "model", name)
        all_rows.append(metrics)

    result = pd.concat(all_rows, ignore_index=True)

    # 콘솔 요약 출력 — 주요 컬럼만
    display_cols = [
        "model",
        "horizon_days",
        "contract_type",
        "n",
        "mape_pct",
        "mae",
        "bias",
        "alarm_precision",
        "alarm_recall",
        "alarm_f1",
        "alarm_alarm_rate_true",
    ]
    show = [c for c in display_cols if c in result.columns]
    print()
    print(result[show].round(3).to_string(index=False))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n[write] {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
