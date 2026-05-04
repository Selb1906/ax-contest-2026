"""평가 하네스 + 알림 판정.

과제 정의:
  - 기준일(검침일) +10일 / +20일 시점에 **기준월 전력사용량** 을 예측
  - 알림: (전월+30% 초과) OR (전년동월+30% 초과) OR (직전3개월평균+50% 초과) 중 1건 만족

본 하네스는 **캘린더 월 기준 시뮬레이션** 을 기본으로 한다 (공개 데이터엔 고객별 검침일이
없으므로). 안심구역에서 실제 검침일 컬럼이 있으면 `ref_date` 를 바꿔 동일 로직 재사용.

+10일 horizon  ≒ 월초 약 10일 관측 + 나머지 ~20일 예측
+20일 horizon  ≒ 월초 약 20일 관측 + 나머지 ~10일 예측
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .schemas import CONTRACT_TYPE, CUSTOMER_ID, P_ACTIVE_KWH, TS


# --------------------------------------------------------------
# 월 집계
# --------------------------------------------------------------


def daily_by_customer(df: pd.DataFrame) -> pd.DataFrame:
    """(customer, date, contract_type) → day_kwh 롱 테이블."""
    t = pd.to_datetime(df[TS])
    out = (
        df.assign(date=t.dt.normalize())
        .groupby([CUSTOMER_ID, CONTRACT_TYPE, "date"], observed=True)[P_ACTIVE_KWH]
        .sum()
        .reset_index(name="day_kwh")
    )
    return out


def monthly_by_customer(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["year_month"] = pd.to_datetime(d["date"]).dt.to_period("M")
    m = (
        d.groupby([CUSTOMER_ID, CONTRACT_TYPE, "year_month"], observed=True)["day_kwh"]
        .sum()
        .reset_index(name="monthly_kwh")
    )
    return m.sort_values([CUSTOMER_ID, "year_month"])


def build_horizon_table(
    daily: pd.DataFrame,
    *,
    horizons: Iterable[int] = (10, 20),
    meter_day: int | None = None,
    meter_day_col: str | None = None,
) -> pd.DataFrame:
    """각 (customer, target_month) × horizon 조합에 대한 평가 테이블.

    Parameters
    ----------
    meter_day : 고정 검침일 (예: 15면 매월 15일이 기준월 시작)
                None이면 월 1일 기준 (캘린더 월)
    meter_day_col : daily에 고객별 검침일 컬럼이 있을 때 컬럼명
                    (예: "meter_reading_day")

    과제 정의:
      기준월 = 검침일 ~ 다음 검침일 전날
      +10일 = 검침일로부터 10일 경과 시점에 예측
      +20일 = 검침일로부터 20일 경과 시점에 예측
    """
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])

    # 검침일 기준 기준월 산정
    if meter_day_col and meter_day_col in d.columns:
        d["_meter_day"] = d[meter_day_col]
    elif meter_day and meter_day > 1:
        d["_meter_day"] = meter_day
    else:
        d["_meter_day"] = 1

    # 기준월: 검침일 이전이면 전월, 이후면 당월
    d["day_of_month"] = d["date"].dt.day
    d["_cal_month"] = d["date"].dt.to_period("M")
    d["year_month"] = d["_cal_month"]
    if meter_day and meter_day > 1:
        d["year_month"] = d.apply(
            lambda r: r["_cal_month"] if r["day_of_month"] >= r["_meter_day"]
            else r["_cal_month"] - 1,
            axis=1,
        )
        d["day_since_meter"] = d.apply(
            lambda r: r["day_of_month"] - r["_meter_day"]
            if r["day_of_month"] >= r["_meter_day"]
            else r["day_of_month"] + (r["date"].days_in_month - r["_meter_day"]),
            axis=1,
        )
    else:
        d["year_month"] = d["date"].dt.to_period("M")
        d["day_since_meter"] = d["day_of_month"] - 1  # 0-based

    # 기준월 내 일수
    billing_days = (
        d.groupby([CUSTOMER_ID, "year_month"], observed=True)["date"]
        .count()
        .reset_index(name="days_in_month")
    )
    full = (
        d.groupby([CUSTOMER_ID, CONTRACT_TYPE, "year_month"], observed=True)["day_kwh"]
        .sum()
        .reset_index(name="full_month_kwh")
    )

    frames = []
    for h in horizons:
        partial = (
            d[d["day_since_meter"] < h]
            .groupby([CUSTOMER_ID, CONTRACT_TYPE, "year_month"], observed=True)[
                "day_kwh"
            ]
            .sum()
            .reset_index(name="partial_kwh")
        )
        tbl = partial.merge(full, on=[CUSTOMER_ID, CONTRACT_TYPE, "year_month"])
        tbl = tbl.merge(billing_days, on=[CUSTOMER_ID, "year_month"])
        tbl["horizon_days"] = h
        tbl["days_observed"] = np.minimum(h, tbl["days_in_month"])
        tbl["remainder_kwh"] = tbl["full_month_kwh"] - tbl["partial_kwh"]
        frames.append(tbl)
    return pd.concat(frames, ignore_index=True).sort_values(
        [CUSTOMER_ID, "year_month", "horizon_days"]
    )


# --------------------------------------------------------------
# 지표
# --------------------------------------------------------------


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    if not mask.any():
        return {"n": 0}
    yt, yp = y_true[mask], y_pred[mask]
    err = yp - yt
    abs_err = np.abs(err)
    rmse = float(np.sqrt(np.mean(err**2)))
    mean_true = float(np.mean(yt))
    mape = np.mean(abs_err / yt) * 100
    smape = np.mean(2 * abs_err / (np.abs(yt) + np.abs(yp))) * 100
    return {
        "n": int(mask.sum()),
        "mae": float(np.mean(abs_err)),
        "rmse": rmse,
        "nrmse_pct": rmse / mean_true * 100 if mean_true > 0 else 0,
        "cvrmse_pct": rmse / mean_true * 100 if mean_true > 0 else 0,
        "mape_pct": float(mape),
        "smape_pct": float(smape),
        "bias": float(np.mean(err)),
        "std_error": float(np.std(err)),
    }


# --------------------------------------------------------------
# 알림 판정
# --------------------------------------------------------------


@dataclass
class AlarmThresholds:
    prev_month: float = 1.30
    yoy: float = 1.30
    ma3: float = 1.50


def attach_alarm_context(
    horizon_tbl: pd.DataFrame, monthly: pd.DataFrame
) -> pd.DataFrame:
    """horizon_tbl 에 전월·전년동월·직전3개월평균 월간 사용량을 붙인다."""
    m = monthly.set_index([CUSTOMER_ID, "year_month"])["monthly_kwh"].sort_index()

    def _lookup(cust: str, ym: pd.Period) -> float:
        try:
            return float(m.loc[(cust, ym)])
        except KeyError:
            return np.nan

    out = horizon_tbl.copy()
    out["prev_month_kwh"] = [
        _lookup(c, ym - 1) for c, ym in zip(out[CUSTOMER_ID], out["year_month"])
    ]
    out["yoy_month_kwh"] = [
        _lookup(c, ym - 12) for c, ym in zip(out[CUSTOMER_ID], out["year_month"])
    ]
    ma_vals = []
    for c, ym in zip(out[CUSTOMER_ID], out["year_month"]):
        vals = [_lookup(c, ym - k) for k in (1, 2, 3)]
        ma_vals.append(np.nanmean(vals) if np.any(np.isfinite(vals)) else np.nan)
    out["ma3_kwh"] = ma_vals
    return out


def alarm_labels(
    pred_monthly: np.ndarray | pd.Series,
    prev_month: np.ndarray | pd.Series,
    yoy_month: np.ndarray | pd.Series,
    ma3: np.ndarray | pd.Series,
    th: AlarmThresholds | None = None,
) -> pd.DataFrame:
    th = th or AlarmThresholds()
    p = np.asarray(pred_monthly, dtype=float)
    cond_prev = p > th.prev_month * np.asarray(prev_month, dtype=float)
    cond_yoy = p > th.yoy * np.asarray(yoy_month, dtype=float)
    cond_ma = p > th.ma3 * np.asarray(ma3, dtype=float)
    any3 = np.where(
        np.isnan(cond_prev) | np.isnan(cond_yoy) | np.isnan(cond_ma),
        False,
        cond_prev.astype(bool) | cond_yoy.astype(bool) | cond_ma.astype(bool),
    )
    # 위의 isnan 체크는 bool 배열엔 동작 안 함 → 다시 계산
    cond_prev_ok = (p > th.prev_month * np.asarray(prev_month, dtype=float))
    cond_yoy_ok = (p > th.yoy * np.asarray(yoy_month, dtype=float))
    cond_ma_ok = (p > th.ma3 * np.asarray(ma3, dtype=float))
    valid_prev = np.isfinite(p) & np.isfinite(np.asarray(prev_month, dtype=float))
    valid_yoy = np.isfinite(p) & np.isfinite(np.asarray(yoy_month, dtype=float))
    valid_ma = np.isfinite(p) & np.isfinite(np.asarray(ma3, dtype=float))
    any_ok = (
        (cond_prev_ok & valid_prev)
        | (cond_yoy_ok & valid_yoy)
        | (cond_ma_ok & valid_ma)
    )
    return pd.DataFrame(
        {
            "cond_prev": cond_prev_ok & valid_prev,
            "cond_yoy": cond_yoy_ok & valid_yoy,
            "cond_ma": cond_ma_ok & valid_ma,
            "any_of_3": any_ok,
        }
    )


def alarm_classification_metrics(
    pred_any: np.ndarray, true_any: np.ndarray
) -> dict:
    pred = np.asarray(pred_any, dtype=bool)
    truth = np.asarray(true_any, dtype=bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "alarm_rate_pred": float(pred.mean()),
        "alarm_rate_true": float(truth.mean()),
    }


# --------------------------------------------------------------
# 통합 평가
# --------------------------------------------------------------


def evaluate(
    predictions: pd.DataFrame,
    horizon_tbl_with_alarm: pd.DataFrame,
    *,
    group_cols: Iterable[str] = ("horizon_days", CONTRACT_TYPE),
) -> pd.DataFrame:
    """predictions: columns = [customer_id, year_month, horizon_days, pred_monthly_kwh]

    horizon_tbl_with_alarm: attach_alarm_context 의 출력.

    반환: group_cols 별 지표(regression + alarm) DataFrame.
    """
    merged = horizon_tbl_with_alarm.merge(
        predictions,
        on=[CUSTOMER_ID, "year_month", "horizon_days"],
        how="inner",
    )
    th = AlarmThresholds()

    pred_labels = alarm_labels(
        merged["pred_monthly_kwh"],
        merged["prev_month_kwh"],
        merged["yoy_month_kwh"],
        merged["ma3_kwh"],
        th,
    )
    true_labels = alarm_labels(
        merged["full_month_kwh"],
        merged["prev_month_kwh"],
        merged["yoy_month_kwh"],
        merged["ma3_kwh"],
        th,
    )

    rows = []
    for keys, sub in merged.groupby(list(group_cols), observed=True):
        idx = sub.index
        reg = regression_metrics(
            sub["full_month_kwh"].to_numpy(), sub["pred_monthly_kwh"].to_numpy()
        )
        al = alarm_classification_metrics(
            pred_labels.loc[idx, "any_of_3"].to_numpy(),
            true_labels.loc[idx, "any_of_3"].to_numpy(),
        )
        if isinstance(keys, tuple):
            row = dict(zip(group_cols, keys))
        else:
            row = {list(group_cols)[0]: keys}
        row.update(reg)
        row.update({f"alarm_{k}": v for k, v in al.items()})
        rows.append(row)
    return pd.DataFrame(rows)
