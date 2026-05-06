"""UI 용 예측 캐시 생성.

LP 데이터 → 베이스라인 예측 + 집계 결과를 data/ui/ 에 parquet 저장.
대시보드(generate_dashboard.py)가 여기서 생성된 파일을 읽는다.

사용법:
  python -m scripts.prepare_ui_data --source configs/source_dsz.yaml
  python -m scripts.prepare_ui_data --source configs/source_dsz.yaml --meter-day 15
"""
from __future__ import annotations

import argparse
import io as _stdio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd

from src import baselines, eval as ev, io_adapter


OUT_DIR = Path("data/ui")


def main() -> int:
    parser = argparse.ArgumentParser(description="대시보드용 데이터 생성")
    parser.add_argument("--source", default="configs/source_synth_v1.yaml",
                        help="데이터 소스 YAML")
    parser.add_argument("--meter-day", type=int, default=1,
                        help="검침일 (1~28, 기본: 1 = 캘린더 월)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = args.meter_day

    print(f"[load] {args.source}")
    print(f"[검침일] {md}일" + (" (캘린더 월)" if md == 1 else ""))
    df = io_adapter.load_from_yaml(args.source, validate=False)

    daily = ev.daily_by_customer(df)
    monthly = ev.monthly_by_customer(daily)
    horizon = ev.build_horizon_table(daily, horizons=(10, 20), meter_day=md)
    ctx = ev.attach_alarm_context(horizon, monthly)

    # 베이스라인 4종 예측
    preds_all = []
    for name, fn in baselines.BASELINES.items():
        p = fn(monthly, horizon)
        p["model"] = name
        preds_all.append(p)
    preds = pd.concat(preds_all, ignore_index=True)

    # 평가 메트릭
    rows = []
    for name, fn in baselines.BASELINES.items():
        pred = fn(monthly, horizon)
        m = ev.evaluate(pred, ctx)
        m.insert(0, "model", name)
        rows.append(m)
    metrics = pd.concat(rows, ignore_index=True)

    # 일자별 누적 kWh
    d = daily.copy()
    d["year_month"] = pd.to_datetime(d["date"]).dt.to_period("M")
    d["day_of_month"] = pd.to_datetime(d["date"]).dt.day
    d["cum_kwh"] = d.sort_values(["customer_id", "date"]).groupby(
        ["customer_id", "year_month"], observed=True
    )["day_kwh"].cumsum()

    # 파일 저장
    daily.to_parquet(OUT_DIR / "daily.parquet", index=False)
    d.to_parquet(OUT_DIR / "daily_cum.parquet", index=False)
    monthly.assign(year_month=monthly["year_month"].astype(str)).to_parquet(
        OUT_DIR / "monthly.parquet", index=False
    )
    preds_out = preds.copy()
    preds_out["year_month"] = preds_out["year_month"].astype(str)
    preds_out.to_parquet(OUT_DIR / "preds.parquet", index=False)
    ctx_out = ctx.copy()
    ctx_out["year_month"] = ctx_out["year_month"].astype(str)
    ctx_out.to_parquet(OUT_DIR / "ctx.parquet", index=False)
    metrics.to_parquet(OUT_DIR / "metrics.parquet", index=False)

    # 메타 정보 (대시보드에서 검침일 표시용)
    meta = {"meter_day": md, "source": args.source}
    pd.DataFrame([meta]).to_parquet(OUT_DIR / "meta.parquet", index=False)

    print(
        f"[ok] daily={len(daily):,}  monthly={len(monthly):,}  "
        f"preds={len(preds):,}  ctx={len(ctx):,}  metrics={len(metrics):,}"
    )
    print(f"[out] {OUT_DIR}/*.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
