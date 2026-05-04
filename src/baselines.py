"""월간 전력사용량 예측 베이스라인 4종.

공용 인터페이스: predict(history_monthly, horizon_tbl) → pred_df

- naive_last_month     : 직전 월의 전체 kWh
- seasonal_yoy         : 전년 동월 kWh
- moving_avg3          : 직전 3개월 평균
- partial_linear       : partial_kwh × (days_in_month / days_observed)
                         → 프로파일러가 확인한 97.6% 상관 직접 활용

partial_linear 가 이론적으로 가장 강한 베이스라인이고 DSZ 실 LP에서도 그럴 가능성 높다.
글로벌 모델(LightGBM 등)은 이 선을 얼마나 넘어서느냐로 가치가 판명된다.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .schemas import CONTRACT_TYPE, CUSTOMER_ID


PredictFn = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def _lookup_map(monthly: pd.DataFrame) -> pd.Series:
    return monthly.set_index([CUSTOMER_ID, "year_month"])["monthly_kwh"].sort_index()


def predict_naive_last_month(
    monthly: pd.DataFrame, horizon_tbl: pd.DataFrame
) -> pd.DataFrame:
    m = _lookup_map(monthly)
    preds = []
    for _, row in horizon_tbl.iterrows():
        key = (row[CUSTOMER_ID], row["year_month"] - 1)
        preds.append(m.get(key, np.nan))
    out = horizon_tbl[[CUSTOMER_ID, "year_month", "horizon_days"]].copy()
    out["pred_monthly_kwh"] = preds
    return out


def predict_seasonal_yoy(
    monthly: pd.DataFrame, horizon_tbl: pd.DataFrame
) -> pd.DataFrame:
    m = _lookup_map(monthly)
    preds = []
    for _, row in horizon_tbl.iterrows():
        key = (row[CUSTOMER_ID], row["year_month"] - 12)
        preds.append(m.get(key, np.nan))
    out = horizon_tbl[[CUSTOMER_ID, "year_month", "horizon_days"]].copy()
    out["pred_monthly_kwh"] = preds
    return out


def predict_moving_avg3(
    monthly: pd.DataFrame, horizon_tbl: pd.DataFrame
) -> pd.DataFrame:
    m = _lookup_map(monthly)
    preds = []
    for _, row in horizon_tbl.iterrows():
        vals = [
            m.get((row[CUSTOMER_ID], row["year_month"] - k), np.nan) for k in (1, 2, 3)
        ]
        arr = np.asarray(vals, dtype=float)
        preds.append(np.nan if np.all(np.isnan(arr)) else np.nanmean(arr))
    out = horizon_tbl[[CUSTOMER_ID, "year_month", "horizon_days"]].copy()
    out["pred_monthly_kwh"] = preds
    return out


def predict_partial_linear(
    monthly: pd.DataFrame, horizon_tbl: pd.DataFrame
) -> pd.DataFrame:
    """월초 관측분을 월 전체로 선형 스케일.

    pred = partial_kwh × (days_in_month / days_observed)
    """
    _ = monthly  # 미사용 (인터페이스 통일용)
    out = horizon_tbl[
        [CUSTOMER_ID, "year_month", "horizon_days", "partial_kwh",
         "days_observed", "days_in_month"]
    ].copy()
    out["pred_monthly_kwh"] = out["partial_kwh"] * out["days_in_month"] / out[
        "days_observed"
    ]
    return out[[CUSTOMER_ID, "year_month", "horizon_days", "pred_monthly_kwh"]]


def predict_partial_seasonal_yoy(
    monthly: pd.DataFrame, horizon_tbl: pd.DataFrame
) -> pd.DataFrame:
    """월초 관측분 + 전년 동월 비례 미관측분.

    pred = partial_kwh + prev_year_same_month × (1 - days_observed/days_in_month)

    부하가 월초~월말에 걸쳐 변화하는 경우(예: 냉방 집중 월), 단순 선형보다
    전년 패턴을 따르는 게 더 정확할 수 있음.
    """
    m = _lookup_map(monthly)
    rem_frac = 1 - horizon_tbl["days_observed"] / horizon_tbl["days_in_month"]
    yoy_full = np.array(
        [
            m.get((c, ym - 12), np.nan)
            for c, ym in zip(horizon_tbl[CUSTOMER_ID], horizon_tbl["year_month"])
        ]
    )
    out = horizon_tbl[[CUSTOMER_ID, "year_month", "horizon_days"]].copy()
    out["pred_monthly_kwh"] = horizon_tbl["partial_kwh"].to_numpy() + yoy_full * rem_frac.to_numpy()
    return out


BASELINES: dict[str, PredictFn] = {
    "naive_last_month": predict_naive_last_month,
    "seasonal_yoy": predict_seasonal_yoy,
    "moving_avg3": predict_moving_avg3,
    "partial_linear": predict_partial_linear,
    "partial_seasonal_yoy": predict_partial_seasonal_yoy,
}
