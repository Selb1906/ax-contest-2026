"""TOU (Time-of-Use) 시간대 구분 — 한전 전기공급약관 별표3 기준.

적용 계약종별: 일반용전력(갑)II, 산업용전력(갑)II, 일반용전력(을),
              산업용전력(을), 교육용전력(을)

계절 구분:
  여름철: 6.1~8.31
  봄가을철: 3.1~5.31, 9.1~10.31
  겨울철: 11.1~익년2.말일

공휴일: 전체 경부하 / 토요일: 최대부하→중간부하로 계량
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def get_season(month: int) -> str:
    if month in (6, 7, 8):
        return "summer"
    elif month in (3, 4, 5, 9, 10):
        return "spring_fall"
    else:
        return "winter"


# (시작시, 종료시) 리스트 — 종료시는 exclusive
_TOU_TABLE = {
    "summer": {
        "off_peak":  [(23, 24), (0, 9)],
        "mid_peak":  [(9, 10), (12, 13), (17, 23)],
        "on_peak":   [(10, 12), (13, 17)],
    },
    "spring_fall": {
        "off_peak":  [(23, 24), (0, 9)],
        "mid_peak":  [(9, 10), (12, 13), (17, 23)],
        "on_peak":   [(10, 12), (13, 17)],
    },
    "winter": {
        "off_peak":  [(23, 24), (0, 9)],
        "mid_peak":  [(9, 10), (12, 17), (20, 22)],
        "on_peak":   [(10, 12), (17, 20), (22, 23)],
    },
}


def classify_tou(hour: int, season: str) -> str:
    for period, ranges in _TOU_TABLE[season].items():
        for start, end in ranges:
            if start <= hour < end:
                return period
    return "off_peak"


def add_tou_columns(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """DataFrame에 season, tou_period 컬럼 추가."""
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    months = ts.dt.month
    hours = ts.dt.hour

    out["season"] = months.map(get_season)
    out["tou_period"] = [
        classify_tou(h, s) for h, s in zip(hours, out["season"])
    ]
    return out


def tou_usage_ratios(df: pd.DataFrame, ts_col: str = "ts", value_col: str = "p_active_kwh") -> pd.DataFrame:
    """고객×월별 TOU 시간대 사용량 비율 산출 — 피처용."""
    d = add_tou_columns(df, ts_col)
    d["year_month"] = pd.to_datetime(d[ts_col]).dt.to_period("M")

    total = d.groupby(["customer_id", "year_month"])[value_col].sum().rename("total_kwh")
    by_period = d.groupby(["customer_id", "year_month", "tou_period"])[value_col].sum().rename("period_kwh")

    ratios = by_period.reset_index().merge(total.reset_index(), on=["customer_id", "year_month"])
    ratios["ratio"] = ratios["period_kwh"] / ratios["total_kwh"].clip(lower=1e-9)

    pivot = ratios.pivot_table(
        index=["customer_id", "year_month"],
        columns="tou_period",
        values="ratio",
        fill_value=0,
    ).reset_index()
    pivot.columns.name = None

    for col in ["off_peak", "mid_peak", "on_peak"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
        pivot = pivot.rename(columns={col: f"tou_ratio_{col}"})

    return pivot
