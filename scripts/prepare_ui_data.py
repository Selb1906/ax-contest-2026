"""UI 용 예측 캐시 생성.

합성 v1 LP → 베이스라인(partial_linear) 예측 + 간단 집계 결과를
data/ui/ 밑에 parquet 으로 저장. Streamlit 앱은 여기만 읽어서 빠르게 렌더링.
"""
from __future__ import annotations

import io as _stdio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd

from src import baselines, eval as ev, io_adapter


SRC = "configs/source_synth_v1.yaml"
OUT_DIR = Path("data/ui")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] {SRC}")
    df = io_adapter.load_from_yaml(SRC, validate=False)

    daily = ev.daily_by_customer(df)
    monthly = ev.monthly_by_customer(daily)
    horizon = ev.build_horizon_table(daily, horizons=(10, 20))
    ctx = ev.attach_alarm_context(horizon, monthly)

    # 베이스라인 4종 예측
    preds_all = []
    for name, fn in baselines.BASELINES.items():
        p = fn(monthly, horizon)
        p["model"] = name
        preds_all.append(p)
    preds = pd.concat(preds_all, ignore_index=True)

    # 평가 메트릭(UI 에서 성능 패널 표시)
    rows = []
    for name, fn in baselines.BASELINES.items():
        pred = fn(monthly, horizon)
        m = ev.evaluate(pred, ctx)
        m.insert(0, "model", name)
        rows.append(m)
    metrics = pd.concat(rows, ignore_index=True)

    # 일자별 누적 kWh (고객·월 기준, UI 그래프용)
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

    print(
        f"[ok] daily={len(daily):,}  monthly={len(monthly):,}  "
        f"preds={len(preds):,}  ctx={len(ctx):,}  metrics={len(metrics):,}"
    )
    print(f"[out] {OUT_DIR}/*.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
