"""sliding_raw.csv → 규모별 × 월별 × horizon별 집계 CSV 생성.

Usage:
    py scripts/export_analysis.py [sliding_results_dir]
    기본값: sliding_results/
"""
import sys, pathlib
import pandas as pd, numpy as np

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "sliding_results")
raw = src / "sliding_raw.csv"
df = pd.read_csv(str(raw))

if "year_month" in df.columns:
    df["month"] = pd.to_datetime(df["year_month"]).dt.month
elif "month" not in df.columns:
    print("WARNING: month column not found, skipping monthly breakdown")
    df["month"] = 0

BINS = [
    (0, 100, "0-100"),
    (100, 1000, "100-1K"),
    (1000, 10000, "1K-10K"),
    (10000, 1e9, "10K+"),
]

rows = []
for lo, hi, lbl in BINS:
    for hz in [10, 20]:
        for m in sorted(df["month"].unique()):
            s = df[
                (df["full_month_kwh"] >= lo)
                & (df["full_month_kwh"] < hi)
                & (df["horizon_days"] == hz)
                & (df["month"] == m)
            ]
            if len(s) == 0:
                continue
            rmse = np.sqrt((s["error"] ** 2).mean())
            rows.append({
                "scale": lbl,
                "horizon": hz,
                "month": int(m),
                "n": len(s),
                "mape": round(s["pct_error"].mean(), 2),
                "rmse": round(rmse, 1),
                "cvrmse": round(rmse / s["full_month_kwh"].mean() * 100, 2),
            })

out = pd.DataFrame(rows)
out_path = src / "analysis_by_scale_month.csv"
out.to_csv(str(out_path), index=False)
print(out.to_string(index=False))
print(f"\nSaved to {out_path}")
