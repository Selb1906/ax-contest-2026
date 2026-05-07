"""BTM 처리 전략 3가지 비교 — 안심구역용.

A: BTM 고객 제외 (baseline)
B: is_btm (0/1 이진) 피처 투입
C: btm_score (0~1 연속) 피처 투입

사용법:
  python -m scripts.compare_btm_strategies --source configs/source_dsz.yaml --train-end 2023-12
"""
from __future__ import annotations

import argparse
import io as _stdio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import numpy as np
import pandas as pd

from src import io_adapter, eval as ev
from src.btm_detect import detect as btm_detect
from src.preprocess import preprocess
from src.models import lgbm


def train_and_eval(df, train_end, label):
    """피처 빌드 + 학습 + 평가."""
    features, spec = lgbm.build_features(df)
    booster, test_preds = lgbm.train(features, spec, train_end=train_end)

    daily = ev.daily_by_customer(df)
    monthly = ev.monthly_by_customer(daily)
    horizon = ev.build_horizon_table(daily, horizons=(10, 20))
    ctx = ev.attach_alarm_context(horizon, monthly)
    metrics = ev.evaluate(test_preds, ctx)
    metrics.insert(0, "strategy", label)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-end", default=None)
    args = parser.parse_args()

    train_end = pd.Period(args.train_end, freq="M") if args.train_end else None

    print("[load]")
    df = io_adapter.load_from_yaml(args.source, validate=False)
    df, _ = preprocess(df)

    print("[btm detect]")
    btm_result = btm_detect(df)
    n_btm = btm_result.summary["n_btm_detected"]
    print(f"  BTM: {n_btm}명")

    if n_btm == 0:
        print("  BTM 미발견 — 비교 불필요 (모든 전략 동일)")
        return 0

    btm_flags = btm_result.customer_flags[["customer_id", "is_btm", "btm_score"]]

    # 전략 A: BTM 제외
    print("\n[A] BTM 제외")
    btm_ids = set(btm_flags[btm_flags["is_btm"] == 1]["customer_id"])
    df_a = df[~df["customer_id"].isin(btm_ids)].copy()
    # is_btm, btm_score 컬럼 제거 (피처로 안 씀)
    df_a = df_a.drop(columns=["is_btm", "btm_score"], errors="ignore")
    metrics_a = train_and_eval(df_a, train_end, "A_exclude")

    # 전략 B: is_btm (0/1) 피처
    print("\n[B] is_btm (0/1) 피처")
    df_b = df.merge(btm_flags[["customer_id", "is_btm"]], on="customer_id", how="left",
                    suffixes=("_orig", ""))
    df_b["is_btm"] = df_b["is_btm"].fillna(0).astype(int)
    df_b = df_b.drop(columns=["btm_score", "is_btm_orig"], errors="ignore")
    metrics_b = train_and_eval(df_b, train_end, "B_binary_flag")

    # 전략 C: btm_score (0~1) 피처
    print("\n[C] btm_score (0~1) 피처")
    df_c = df.merge(btm_flags[["customer_id", "is_btm", "btm_score"]], on="customer_id", how="left",
                    suffixes=("_orig", ""))
    df_c["is_btm"] = df_c["is_btm"].fillna(0).astype(int)
    df_c["btm_score"] = df_c["btm_score"].fillna(0.0)
    df_c = df_c.drop(columns=["is_btm_orig", "btm_score_orig"], errors="ignore")
    metrics_c = train_and_eval(df_c, train_end, "C_continuous_score")

    # 비교
    all_metrics = pd.concat([metrics_a, metrics_b, metrics_c], ignore_index=True)
    Path("btm_results").mkdir(exist_ok=True)
    all_metrics.to_csv("btm_results/strategy_comparison.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("BTM 처리 전략 비교")
    print(f"{'='*60}")

    summary = all_metrics.groupby("strategy").agg(
        mape_mean=("mape_pct", "mean"),
        f1_mean=("alarm_f1", "mean"),
    ).sort_values("mape_mean")

    for _, row in summary.iterrows():
        marker = " <-- best" if row.name == summary.index[0] else ""
        print(f"  {row.name:25s}: MAPE {row['mape_mean']:.3f}%  F1 {row['f1_mean']:.3f}{marker}")

    # BTM 고객만의 성능 (전략 B, C에서)
    print(f"\n  BTM 고객 수: {n_btm}명")
    print(f"  전체 대비: {n_btm/df['customer_id'].nunique()*100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
