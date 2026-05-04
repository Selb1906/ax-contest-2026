"""잔차 보정 모델 — 2-stage stacking.

Stage 1: LightGBM → pred_1 (메인 예측)
Stage 2: Ridge → pred_residual (체계적 편향 보정)
최종:    pred_final = pred_1 + pred_residual

왜 Ridge인가:
  - 잔차는 원래 작은 값 → 복잡한 모델은 과적합
  - L2 정규화로 안정적
  - 해석 가능 (어떤 조건에서 모델이 체계적으로 틀리는지)

반출 안전: 보정 모델 계수는 집계 수치이므로 반출 가능.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def build_residual_features(
    features: pd.DataFrame,
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """잔차 보정용 피처 구성.

    잔차의 체계적 패턴을 잡기 위한 피처:
    - month: 계절별 편향
    - horizon_days: +10일 vs +20일 차이
    - disruption_ratio: 특수일 보정
    - pred_1 자체: 예측값 크기에 비례하는 오차
    """
    residuals = actuals - predictions

    res_features = pd.DataFrame()
    if "month" in features.columns:
        res_features["month"] = features["month"].values
    if "horizon_days" in features.columns:
        res_features["horizon_days"] = features["horizon_days"].values
    if "disruption_ratio" in features.columns:
        res_features["disruption_ratio"] = features["disruption_ratio"].values
    if "contract_type" in features.columns:
        res_features["is_residential"] = (features["contract_type"] == "주택용").astype(int).values

    res_features["pred_1"] = predictions
    res_features["pred_1_log"] = np.log1p(np.abs(predictions))

    return res_features, residuals


def train_residual_model(
    res_features: pd.DataFrame,
    residuals: np.ndarray,
    alpha: float = 1.0,
) -> tuple[Ridge, StandardScaler, list[str]]:
    """잔차 보정 Ridge 모델 학습."""
    feature_cols = [c for c in res_features.columns if res_features[c].dtype != "category"]
    X = res_features[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=alpha)
    model.fit(X_scaled, residuals)

    return model, scaler, feature_cols


def predict_residual(
    model: Ridge,
    scaler: StandardScaler,
    feature_cols: list[str],
    res_features: pd.DataFrame,
) -> np.ndarray:
    """잔차 예측."""
    X = res_features[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def two_stage_predict(
    pred_stage1: np.ndarray,
    features: pd.DataFrame,
    residual_model: Ridge,
    residual_scaler: StandardScaler,
    residual_feature_cols: list[str],
) -> np.ndarray:
    """2-stage 최종 예측 = Stage1 + 잔차 보정."""
    res_features, _ = build_residual_features(features, pred_stage1, np.zeros_like(pred_stage1))
    correction = predict_residual(residual_model, residual_scaler, residual_feature_cols, res_features)
    return pred_stage1 + correction


def analyze_residual_pattern(
    features: pd.DataFrame,
    residuals: np.ndarray,
    model: Ridge,
    feature_cols: list[str],
    min_n: int = 10,
) -> dict:
    """잔차 패턴 분석 — 보고서용, 반출 안전.

    어떤 조건에서 모델이 체계적으로 틀리는지 분석.
    고객ID 없이 집계만 산출.
    """
    result = {}

    # Ridge 계수 (반출 안전)
    coef_dict = {col: float(c) for col, c in zip(feature_cols, model.coef_)}
    coef_dict["intercept"] = float(model.intercept_)
    result["ridge_coefficients"] = coef_dict

    # 월별 평균 잔차 (반출 안전 — 월별 집계)
    if "month" in features.columns:
        monthly_res = []
        for m in range(1, 13):
            mask = features["month"].values == m
            if mask.sum() >= min_n:
                monthly_res.append({
                    "month": m,
                    "mean_residual": float(residuals[mask].mean()),
                    "std_residual": float(residuals[mask].std()),
                    "n": int(mask.sum()),
                })
        result["by_month"] = monthly_res

    # horizon별 (반출 안전)
    if "horizon_days" in features.columns:
        horizon_res = []
        for h in features["horizon_days"].unique():
            mask = features["horizon_days"].values == h
            if mask.sum() >= min_n:
                horizon_res.append({
                    "horizon_days": int(h),
                    "mean_residual": float(residuals[mask].mean()),
                    "std_residual": float(residuals[mask].std()),
                    "n": int(mask.sum()),
                })
        result["by_horizon"] = horizon_res

    # 보정 전후 비교
    result["before_correction"] = {
        "mean_abs_residual": float(np.abs(residuals).mean()),
        "std_residual": float(residuals.std()),
        "bias": float(residuals.mean()),
    }

    return result


def run_residual_correction(
    features: pd.DataFrame,
    pred_stage1: np.ndarray,
    actuals: np.ndarray,
    train_mask: np.ndarray,
    out_dir: str | Path | None = None,
) -> dict:
    """잔차 보정 전체 파이프라인.

    train_mask 구간에서 잔차 모델 학습 → test에서 보정 → 전후 비교.
    """
    import json

    # 학습: train 구간의 잔차로 Ridge 학습
    train_res_features, train_residuals = build_residual_features(
        features.iloc[train_mask], pred_stage1[train_mask], actuals[train_mask]
    )
    model, scaler, feature_cols = train_residual_model(train_res_features, train_residuals)

    # 테스트: test 구간에서 보정
    test_mask = ~np.array(train_mask)
    test_res_features, test_residuals = build_residual_features(
        features.iloc[test_mask], pred_stage1[test_mask], actuals[test_mask]
    )
    correction = predict_residual(model, scaler, feature_cols, test_res_features)
    pred_corrected = pred_stage1[test_mask] + correction

    # 보정 전후 비교
    from .eval import regression_metrics
    before = regression_metrics(actuals[test_mask], pred_stage1[test_mask])
    after = regression_metrics(actuals[test_mask], pred_corrected)

    # 패턴 분석
    pattern = analyze_residual_pattern(features.iloc[test_mask], test_residuals, model, feature_cols)

    result = {
        "before_correction": before,
        "after_correction": after,
        "improvement": {
            "mape_delta": before.get("mape_pct", 0) - after.get("mape_pct", 0),
            "rmse_delta": before.get("rmse", 0) - after.get("rmse", 0),
        },
        "pattern_analysis": pattern,
        "model_alpha": model.alpha,
        "feature_cols": feature_cols,
    }

    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "residual_correction.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n  잔차 보정 결과:")
    print(f"    보정 전: MAPE {before.get('mape_pct', 0):.3f}%  RMSE {before.get('rmse', 0):.2f}")
    print(f"    보정 후: MAPE {after.get('mape_pct', 0):.3f}%  RMSE {after.get('rmse', 0):.2f}")
    print(f"    개선:    MAPE {result['improvement']['mape_delta']:+.3f}%p")

    return result
