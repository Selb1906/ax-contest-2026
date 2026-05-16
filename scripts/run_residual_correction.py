"""잔차 보정 독립 실행 — 튜닝된 LightGBM의 잔차를 Ridge로 보정.

Usage:
    python -m scripts.run_residual_correction --source configs/source_dsz.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
import os; os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.models import lgbm
from src.residual_correction import (
    build_residual_features,
    train_residual_model,
    analyze_residual_pattern,
)
from src.eval import regression_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--model-dir", default="weights/dsz_lgbm_tuned")
    args = parser.parse_args()

    # 필수 파일 검증
    model_path = Path(args.model_dir) / "model.txt"
    if not model_path.exists():
        print(f"[FATAL] 모델 없음: {model_path}"); return 1

    # 데이터 로드
    print("[1/4] 데이터 로드...", flush=True)
    fc = Path("checkpoints/features_cache.parquet")
    if fc.exists():
        features = pd.read_parquet(fc)
        features["year_month"] = features["year_month"].apply(lambda x: pd.Period(str(x), freq="M"))
    else:
        from src.io_adapter import load_smart
        df = load_smart(args.source)
        features, _ = lgbm.build_features(df)

    # AMI 피처 조인
    _ami_path = Path("data/preprocessed/ami_features.parquet")
    if _ami_path.exists():
        _ami = pd.read_parquet(_ami_path)
        _ami_new = [c for c in _ami.columns if c not in features.columns and c not in ("customer_id", "year_month")]
        if _ami_new:
            features["year_month_str"] = features["year_month"].astype(str)
            _ami["year_month_str"] = _ami["year_month"].astype(str)
            features = features.merge(
                _ami[["customer_id", "year_month_str"] + _ami_new],
                on=["customer_id", "year_month_str"], how="left")
            features = features.drop(columns=["year_month_str"], errors="ignore")
            print(f"  [AMI] {len(_ami_new)}개 피처 조인")

    # ablation 최적 config 적용
    from src.utils import load_best_config, apply_tmy_remainder
    _best_cfg = load_best_config()
    if _best_cfg:
        _wv = _best_cfg.get("best_per_factor", {}).get("weather_version", "regional")
        features = apply_tmy_remainder(features, _wv)

    # 모델 로드
    print("[2/4] 모델 로드...", flush=True)
    booster, spec = lgbm.load(args.model_dir)

    if args.train_end:
        train_end = pd.Period(args.train_end, freq="M")
    else:
        max_ym = features["year_month"].max()
        train_end = pd.Period(f"{max_ym.year - 1}-12", freq="M")

    features = features.dropna(subset=["full_month_kwh", "partial_kwh"]).copy()
    for c in spec.categorical:
        if c in features.columns:
            features[c] = features[c].astype("category")
    for c in spec.all:
        if c not in features.columns:
            features[c] = np.nan

    train_mask = features["year_month"] <= train_end
    test_mask = ~train_mask

    # Stage 1 예측
    print("[3/4] Stage 1 예측 + 잔차 보정...", flush=True)
    pred_train = booster.predict(features.loc[train_mask, spec.all], num_iteration=booster.best_iteration)
    actual_train = features.loc[train_mask, "full_month_kwh"].values

    pred_test = booster.predict(features.loc[test_mask, spec.all], num_iteration=booster.best_iteration)
    actual_test = features.loc[test_mask, "full_month_kwh"].values

    # 보정 전 지표
    before = regression_metrics(actual_test, pred_test)
    print(f"  보정 전: MAE={before['mae']:.1f}, RMSE={before['rmse']:.1f}, CVRMSE={before['cvrmse_pct']:.1f}%")

    # Stage 2: Ridge 잔차 보정
    res_feat_train, residuals_train = build_residual_features(features[train_mask], pred_train, actual_train)
    ridge_model, scaler, feature_cols = train_residual_model(res_feat_train, residuals_train)

    res_feat_test, _ = build_residual_features(features[test_mask], pred_test, actual_test)
    from src.residual_correction import predict_residual
    correction = predict_residual(ridge_model, scaler, feature_cols, res_feat_test)
    pred_corrected = pred_test + correction

    # 보정 후 지표
    after = regression_metrics(actual_test, pred_corrected)
    print(f"  보정 후: MAE={after['mae']:.1f}, RMSE={after['rmse']:.1f}, CVRMSE={after['cvrmse_pct']:.1f}%")
    print(f"  개선: MAE {before['mae']-after['mae']:+.1f}, RMSE {before['rmse']-after['rmse']:+.1f}")

    # 잔차 패턴 분석
    print("[4/4] 잔차 패턴 분석...", flush=True)
    residuals_test = actual_test - pred_test
    pattern = analyze_residual_pattern(features[test_mask], residuals_test, ridge_model, feature_cols)

    out_dir = Path("residual_correction")
    out_dir.mkdir(exist_ok=True)

    summary = {
        "before": {k: round(v, 4) if isinstance(v, float) else v for k, v in before.items()},
        "after": {k: round(v, 4) if isinstance(v, float) else v for k, v in after.items()},
        "ridge_coefficients": {col: float(c) for col, c in zip(feature_cols, ridge_model.coef_)},
        "ridge_intercept": float(ridge_model.intercept_),
    }
    with open(out_dir / "residual_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 결과: {out_dir}/residual_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
