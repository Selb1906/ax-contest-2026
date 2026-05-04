"""BTM 태양광 발전량 예측 모델 — 기상→발전량 보조 모델.

목적:
  BTM 고객의 자가발전량을 추정하여 "실제 소비 = 계량값 + 발전량"을 복원.
  운영 시에는 기상 예보로 발전량을 예측한 뒤 메인 모델에 투입.

데이터 요건:
  - 전력거래소 "자가용 태양광 출력 추계" (월별 or 시간별)
  - ASOS 기상 데이터 (일사량, 기온, 운량)

현재 상태:
  전력거래소 데이터 미확보 → 골격만 구현, 데이터 확보 시 바로 실행 가능.

2단계 예측 구조:
  Stage 1: 기상 → BTM 발전량 예측 (이 모듈)
  Stage 2: LP + 기상 + BTM발전량(예측) → 월 사용량 예측 (lgbm.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_solar_features(
    weather_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """기상 월 집계 → 태양광 발전 예측용 피처.

    Parameters
    ----------
    weather_monthly : ASOS 월별 집계 (solar_mj_mean, temp_mean, cloud_total 등)
    """
    df = weather_monthly.copy()
    features = pd.DataFrame()
    features["year_month"] = df["year_month"]

    if "solar_mj_mean" in df.columns:
        features["solar_mj"] = df["solar_mj_mean"]
    if "temp_mean" in df.columns:
        features["temp_mean"] = df["temp_mean"]
        # 태양광 효율은 고온에서 하락 (온도 계수 ~-0.4%/°C)
        features["temp_loss_factor"] = 1 - 0.004 * np.maximum(df["temp_mean"] - 25, 0)

    features["month"] = df["year_month"].apply(lambda x: x.month)
    # 일조 가능 시간 (월별 낮 길이 근사)
    features["daylight_hours"] = features["month"].map({
        1: 9.8, 2: 10.8, 3: 12.0, 4: 13.2, 5: 14.2, 6: 14.8,
        7: 14.5, 8: 13.6, 9: 12.4, 10: 11.2, 11: 10.1, 12: 9.5,
    })

    return features


def train_solar_model(
    solar_features: pd.DataFrame,
    btm_generation: pd.DataFrame,
    train_end: pd.Period | None = None,
) -> tuple:
    """BTM 발전량 예측 모델 학습.

    Parameters
    ----------
    solar_features : build_solar_features() 출력
    btm_generation : (year_month, btm_generation_kwh) — 전력거래소 데이터
    train_end : 학습 종료 기간
    """
    import lightgbm as lgb

    merged = solar_features.merge(btm_generation, on="year_month", how="inner")
    feature_cols = [c for c in merged.columns
                    if c not in ("year_month", "btm_generation_kwh")]

    if train_end is None:
        max_ym = merged["year_month"].max()
        train_end = pd.Period(f"{max_ym.year - 1}-12", freq="M")

    train = merged[merged["year_month"] <= train_end]
    test = merged[merged["year_month"] > train_end]

    X_tr, y_tr = train[feature_cols], train["btm_generation_kwh"]
    X_te, y_te = test[feature_cols], test["btm_generation_kwh"]

    dtr = lgb.Dataset(X_tr, label=y_tr)
    params = {
        "objective": "regression",
        "metric": "mape",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "verbose": -1,
    }
    booster = lgb.train(params, dtr, num_boost_round=200)

    if len(X_te) > 0:
        pred = booster.predict(X_te)
        mape = np.mean(np.abs(pred - y_te.values) / y_te.values) * 100
        print(f"[btm_solar] test MAPE: {mape:.2f}%")

    return booster, feature_cols


def predict_btm_generation(
    booster,
    feature_cols: list[str],
    solar_features: pd.DataFrame,
) -> pd.DataFrame:
    """기상 예보 → BTM 발전량 예측."""
    pred = booster.predict(solar_features[feature_cols])
    result = solar_features[["year_month"]].copy()
    result["predicted_btm_kwh"] = pred
    return result


def restore_actual_consumption(
    metered_kwh: float,
    predicted_btm_kwh: float,
) -> float:
    """계량 사용량 + 예측 발전량 → 실제 소비 추정.

    metered = actual_consumption - btm_generation
    → actual_consumption = metered + btm_generation
    """
    return metered_kwh + max(predicted_btm_kwh, 0)


def design_summary() -> str:
    """보고서용 설계 요약."""
    return """
## BTM 태양광 발전량 예측 모델 설계

### 배경
BTM(Behind-The-Meter) 태양광 설치 고객의 계량 사용량은 실제 소비에서
자가발전량이 차감된 값이다. 자가발전량을 모르면 예측 모델이 왜곡된다.
한전 고객 중 태양광 설치 비율은 지속 증가 중이며, 스케일 아웃 시
무시할 수 없는 규모가 된다.

### 2단계 예측 구조
```
Stage 1: 기상 데이터 (일사량, 기온, 운량)
         → LightGBM → 월별 BTM 발전량 예측

Stage 2: LP 사용량 + 기상 + BTM 발전량(Stage 1 예측값)
         → LightGBM → 월별 실제 소비 예측
         → 실제 소비 - BTM 발전량 = 계량 기준 예측
         → 알림 판정
```

### 필요 데이터
- 전력거래소 "자가용 태양광 출력 추계" (학습용)
- ASOS 기상 데이터 (입력)
- 기상 예보 (운영 시 잔여 기간)

### 기대 효과
- BTM 고객을 제외하지 않고 학습 가능 → 데이터 손실 방지
- 여름철 태양광 발전 증가 시 "사용량 감소"를 오탐하지 않음
- 스케일 아웃 시 BTM 비율 증가에 대응

### 한계
- 개별 고객의 패널 용량을 모름 → 지역 평균으로 대체
- 전력거래소 BTM 데이터 접근 필요 (별도 신청)
- 기상 예보 오차가 발전량 예측에 전파
"""
