"""피처 선택 + 선택된 피처로 모델 학습 비교.

사용법:
  python -m scripts.select_features --source configs/source_dsz.yaml --train-end 2023-12
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

from src import io_adapter
from src.models import lgbm
from src.feature_selection import auto_select_features, print_selection_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--corr-threshold", type=float, default=0.85)
    parser.add_argument("--vif-threshold", type=float, default=10.0)
    args = parser.parse_args()

    print("[load] data...")
    df = io_adapter.load_smart(args.source, validate=False)
    features, spec = lgbm.build_features(df)

    print(f"  rows={len(features):,}  candidate features={len(spec.all)}")
    print(f"  features: {spec.all}")

    # 피처 선택
    numeric_candidates = [c for c in spec.numeric if c in features.columns]
    result = auto_select_features(
        features, numeric_candidates,
        target_col="full_month_kwh",
        corr_threshold=args.corr_threshold,
        vif_threshold=args.vif_threshold,
    )
    print_selection_report(result)

    # 선택 전/후 모델 비교
    train_end = pd.Period(args.train_end, freq="M") if args.train_end else None

    print(f"\n{'='*60}")
    print("모델 비교: 전체 피처 vs 선택 피처")
    print(f"{'='*60}")

    # 전체 피처
    booster_full, preds_full = lgbm.train(features, spec, train_end=train_end)
    actual = features[features["year_month"] > train_end]["full_month_kwh"] if train_end else features["full_month_kwh"]

    # 선택 피처
    from dataclasses import replace
    selected_with_cat = result.selected + spec.categorical
    spec_selected = replace(spec)
    # 선택된 피처로만 학습
    features_selected = features.copy()
    # 선택되지 않은 numeric 피처를 NaN으로 (LightGBM이 무시)
    for c in spec.numeric:
        if c not in result.selected:
            features_selected[c] = np.nan

    booster_sel, preds_sel = lgbm.train(features_selected, spec, train_end=train_end)

    print(f"\n  전체 피처 ({len(spec.all)}개): 위 결과 참조")
    print(f"  선택 피처 ({len(result.selected) + len(spec.categorical)}개): 위 결과 참조")

    # 결과 저장
    Path("feature_selection").mkdir(exist_ok=True)
    result.ranking.to_csv("feature_selection/ranking.csv", index=False, encoding="utf-8-sig")
    result.inter_correlation.to_csv("feature_selection/inter_correlation.csv", encoding="utf-8-sig")
    pd.DataFrame({"selected": result.selected}).to_csv(
        "feature_selection/selected_features.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\n[saved] feature_selection/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
