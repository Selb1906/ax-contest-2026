"""요금 산출용 피처 추정 — 역률·TOU 비율의 월 전체 추정.

관측 N일 데이터에서 월 전체의 역률·TOU 비율을 추정하는 여러 전략.
안심구역에서 stat_partial_month_stability 결과를 보고 최적 전략 선택.

전략:
  A. observed — 관측 N일 값 그대로 연장 (상관 높을 때)
  B. prev_month — 전월 값 사용 (래그)
  C. historical_mean — 고객 과거 평균 사용
  D. seasonal_mean — 고객의 해당 계절 평균
  E. weighted_blend — A와 B의 가중 평균
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import CUSTOMER_ID, P_ACTIVE_KWH, P_REACTIVE_KWH, TS


def compute_monthly_pf(df: pd.DataFrame) -> pd.DataFrame:
    """고객×월별 역률 산출."""
    if P_REACTIVE_KWH not in df.columns:
        return pd.DataFrame()

    d = df[[CUSTOMER_ID, TS, P_ACTIVE_KWH, P_REACTIVE_KWH]].copy()
    d["year_month"] = pd.to_datetime(d[TS]).dt.to_period("M")

    g = d.groupby([CUSTOMER_ID, "year_month"], observed=True).agg(
        p_sum=(P_ACTIVE_KWH, "sum"),
        q_sum=(P_REACTIVE_KWH, lambda x: x.abs().sum()),
    ).reset_index()

    apparent = np.sqrt(g["p_sum"]**2 + g["q_sum"]**2).clip(lower=1e-9)
    g["monthly_pf"] = g["p_sum"] / apparent
    g["monthly_reactive_ratio"] = g["q_sum"] / g["p_sum"].clip(lower=1e-9)
    return g[[CUSTOMER_ID, "year_month", "monthly_pf", "monthly_reactive_ratio"]]


def compute_monthly_tou_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """고객×월별 TOU 시간대 사용 비율 산출."""
    from .tou import get_season, classify_tou

    d = df[[CUSTOMER_ID, TS, P_ACTIVE_KWH]].copy()
    t = pd.to_datetime(d[TS])
    d["year_month"] = t.dt.to_period("M")
    d["hour"] = t.dt.hour
    d["month"] = t.dt.month
    d["season"] = d["month"].map(get_season)
    d["tou"] = [classify_tou(h, s) for h, s in zip(d["hour"], d["season"])]

    total = d.groupby([CUSTOMER_ID, "year_month"], observed=True)[P_ACTIVE_KWH].sum().rename("total")
    by_tou = d.groupby([CUSTOMER_ID, "year_month", "tou"], observed=True)[P_ACTIVE_KWH].sum().rename("tou_kwh")

    ratios = by_tou.reset_index().merge(total.reset_index(), on=[CUSTOMER_ID, "year_month"])
    ratios["ratio"] = ratios["tou_kwh"] / ratios["total"].clip(lower=1e-9)

    pivot = ratios.pivot_table(
        index=[CUSTOMER_ID, "year_month"], columns="tou", values="ratio", fill_value=0
    ).reset_index()
    pivot.columns.name = None
    for col in ["off_peak", "mid_peak", "on_peak"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    return pivot


class BillingFeatureEstimator:
    """요금 피처 추정기 — 전략별 역률·TOU 비율 추정."""

    def __init__(self, df: pd.DataFrame):
        self.monthly_pf = compute_monthly_pf(df)
        self.monthly_tou = compute_monthly_tou_ratios(df)
        self.has_pf = len(self.monthly_pf) > 0

    def estimate_pf(
        self,
        customer_id: str,
        year_month: pd.Period,
        observed_pf: float | None = None,
        strategy: str = "observed",
    ) -> float:
        """역률 추정."""
        cust_pf = self.monthly_pf[self.monthly_pf[CUSTOMER_ID] == customer_id]
        lookup = cust_pf.set_index("year_month")["monthly_pf"]

        if strategy == "observed" and observed_pf is not None:
            return observed_pf

        elif strategy == "prev_month":
            return lookup.get(year_month - 1, np.nan)

        elif strategy == "historical_mean":
            past = cust_pf[cust_pf["year_month"] < year_month]["monthly_pf"]
            return float(past.mean()) if len(past) > 0 else np.nan

        elif strategy == "seasonal_mean":
            target_month = year_month.month
            same_season = cust_pf[
                (cust_pf["year_month"].apply(lambda x: x.month) == target_month)
                & (cust_pf["year_month"] < year_month)
            ]["monthly_pf"]
            return float(same_season.mean()) if len(same_season) > 0 else np.nan

        elif strategy == "weighted_blend":
            obs = observed_pf if observed_pf is not None else np.nan
            prev = lookup.get(year_month - 1, np.nan)
            if np.isnan(obs):
                return prev
            if np.isnan(prev):
                return obs
            return 0.7 * obs + 0.3 * prev

        return np.nan

    def estimate_tou_ratios(
        self,
        customer_id: str,
        year_month: pd.Period,
        observed_ratios: dict | None = None,
        strategy: str = "observed",
    ) -> dict[str, float]:
        """TOU 비율 추정."""
        cust_tou = self.monthly_tou[self.monthly_tou[CUSTOMER_ID] == customer_id]
        default = {"off_peak": 0.3, "mid_peak": 0.4, "on_peak": 0.3}

        if strategy == "observed" and observed_ratios is not None:
            return observed_ratios

        elif strategy == "prev_month":
            prev = cust_tou[cust_tou["year_month"] == year_month - 1]
            if len(prev) > 0:
                r = prev.iloc[0]
                return {k: float(r.get(k, default[k])) for k in default}

        elif strategy == "historical_mean":
            past = cust_tou[cust_tou["year_month"] < year_month]
            if len(past) > 0:
                return {k: float(past[k].mean()) for k in default if k in past.columns}

        elif strategy == "seasonal_mean":
            target_month = year_month.month
            same = cust_tou[
                (cust_tou["year_month"].apply(lambda x: x.month) == target_month)
                & (cust_tou["year_month"] < year_month)
            ]
            if len(same) > 0:
                return {k: float(same[k].mean()) for k in default if k in same.columns}

        elif strategy == "weighted_blend":
            obs = observed_ratios or {}
            prev = self.estimate_tou_ratios(customer_id, year_month, strategy="prev_month")
            return {
                k: 0.7 * obs.get(k, prev.get(k, default[k]))
                   + 0.3 * prev.get(k, default[k])
                for k in default
            }

        return default

    def compare_strategies(
        self,
        df: pd.DataFrame,
        horizons: tuple[int, ...] = (10, 20),
    ) -> pd.DataFrame:
        """모든 전략의 역률 추정 정확도 비교 — 최적 전략 선택용."""
        if not self.has_pf:
            return pd.DataFrame()

        t = pd.to_datetime(df[TS])
        d = df[[CUSTOMER_ID, TS, P_ACTIVE_KWH, P_REACTIVE_KWH]].copy()
        d["year_month"] = t.dt.to_period("M")
        d["day"] = t.dt.day

        strategies = ["observed", "prev_month", "historical_mean", "seasonal_mean", "weighted_blend"]
        results = []

        for ym in self.monthly_pf["year_month"].unique():
            for cid in self.monthly_pf[self.monthly_pf["year_month"] == ym][CUSTOMER_ID].unique():
                actual_pf_row = self.monthly_pf[
                    (self.monthly_pf[CUSTOMER_ID] == cid) & (self.monthly_pf["year_month"] == ym)
                ]
                if actual_pf_row.empty:
                    continue
                actual_pf = float(actual_pf_row["monthly_pf"].iloc[0])

                for h in horizons:
                    obs_data = d[(d[CUSTOMER_ID] == cid) & (d["year_month"] == ym) & (d["day"] <= h)]
                    if obs_data.empty:
                        continue
                    p = obs_data[P_ACTIVE_KWH].sum()
                    q = obs_data[P_REACTIVE_KWH].abs().sum()
                    obs_pf = float(p / np.sqrt(p**2 + q**2)) if p > 0 else np.nan

                    for strat in strategies:
                        est = self.estimate_pf(cid, ym, obs_pf, strategy=strat)
                        if not np.isnan(est):
                            results.append({
                                "strategy": strat,
                                "horizon": h,
                                "actual_pf": actual_pf,
                                "estimated_pf": est,
                                "error": abs(est - actual_pf),
                            })

        if not results:
            return pd.DataFrame()

        rdf = pd.DataFrame(results)
        summary = rdf.groupby(["strategy", "horizon"]).agg(
            mae=("error", "mean"),
            n=("error", "count"),
        ).reset_index().sort_values(["horizon", "mae"])
        return summary
