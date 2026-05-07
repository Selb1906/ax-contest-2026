"""Sliding 상세 평가 — 1년 전체를 매일 sliding하며 오차 분석.

사용법:
  python -m scripts.sliding_eval --source configs/source_dsz.yaml --train-end 2023-12 --weights weights/dsz_lgbm/
"""
from __future__ import annotations

import argparse
import io as _stdio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import numpy as np
import pandas as pd

from src import io_adapter
from src.preprocess import preprocess
from src.eval import daily_by_customer, monthly_by_customer, build_horizon_table, attach_alarm_context
from src.eval_detailed import sliding_evaluation, save_detailed_evaluation
from src.models.lgbm import load, build_features, predict
from src.checkpoint import CheckpointManager
from src.schemas import CUSTOMER_ID

OUT_DIR = Path("sliding_results")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sliding 상세 평가")
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--weights", default="weights/dsz_lgbm/")
    parser.add_argument("--meter-days", default=",".join(str(i) for i in range(1, 32)),
                        help="검침일 목록 (쉼표 구분, 기본: 1~31)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager("checkpoints/sliding")
    train_end = pd.Period(args.train_end, freq="M")
    meter_days = [int(x) for x in args.meter_days.split(",")]

    print(f"""
╔══════════════════════════════════════════════════════╗
║          Sliding 상세 평가                            ║
╚══════════════════════════════════════════════════════╝
  검침일: {meter_days}
  모델:   {args.weights}
""")

    # 데이터 + 모델 로드
    with ckpt.step("load", skip_if_done=False) as prog:
        df = io_adapter.load_from_yaml(args.source, validate=False)
        df, _ = preprocess(df)
        booster, spec = load(args.weights)
        prog.update(label=f"데이터: {len(df):,}행, 모델: {len(spec.all)} 피처")

    # sliding 평가
    daily = daily_by_customer(df)
    monthly = monthly_by_customer(daily)

    all_results = []
    with ckpt.step("sliding", total=len(meter_days)) as prog:
        for md in meter_days:
            t0 = time.time()
            horizon_tbl = build_horizon_table(daily, horizons=(10, 20), meter_day=md)
            ctx = attach_alarm_context(horizon_tbl, monthly)

            # 피처 빌드 + 예측
            features, feat_spec = build_features(df)
            features_clean = features.dropna(subset=["full_month_kwh", "partial_kwh"]).copy()
            for c in spec.categorical:
                if c in features_clean.columns:
                    features_clean[c] = features_clean[c].astype("category")

            test_mask = features_clean["year_month"] > train_end
            if test_mask.sum() == 0:
                prog.update(1, label=f"md={md}: 테스트 데이터 없음")
                continue

            test_features = features_clean[test_mask]
            pred = booster.predict(test_features[spec.all], num_iteration=booster.best_iteration)

            pred_df = test_features[[CUSTOMER_ID, "year_month", "horizon_days"]].copy()
            pred_df["pred_monthly_kwh"] = pred

            merged = ctx.merge(pred_df, on=[CUSTOMER_ID, "year_month", "horizon_days"], how="inner")
            merged["meter_day"] = md
            merged["error"] = merged["pred_monthly_kwh"] - merged["full_month_kwh"]
            merged["abs_error"] = merged["error"].abs()
            merged["pct_error"] = merged["abs_error"] / merged["full_month_kwh"].clip(lower=1e-9) * 100
            merged["month"] = merged["year_month"].apply(lambda x: x.month)

            mape = merged["pct_error"].mean()
            elapsed = time.time() - t0
            prog.update(1, label=f"md={md}: MAPE {mape:.3f}%, {len(merged)}건, {elapsed:.1f}s")

            all_results.append(merged)
            ckpt.save_intermediate(f"sliding_md{md:02d}", merged)

    if not all_results:
        print("  결과 없음")
        return 1

    results = pd.concat(all_results, ignore_index=True)

    # 상세 분석 + 저장
    with ckpt.step("analysis") as prog:
        prog.update(label="차원별 오차 집계 + 차트 생성...")
        dims = save_detailed_evaluation(results, OUT_DIR)

    print(f"\n  저장: {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
