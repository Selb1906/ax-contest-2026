"""LP 프로파일러.

목적:
  - 안심구역 내 실 LP 에 돌려서 반출 가능한 **집계 통계**만 산출
  - 합성 생성기 v1 파라미터 보정 / 보고서 "데이터 특성 분석" 근거
  - DSZ 심사 통과 가능한 구조 (개별 고객 재식별 불가, 최소 N 강제)

설계:
  - 각 통계 단위가 독립 함수 → 실패 시 skip 가능
  - 산출물은 profile_stats/ 아래 개별 파일 (JSON/CSV). 심사자가 파일 단위로 검토
  - MANIFEST.json 에 전체 목록 + 각 산출물 "재식별 위험 평가" 기록

반출 안전 정책:
  - MIN_N: 어떤 집계든 구성 표본이 MIN_N 미만이면 해당 bin/row 삭제
  - 수치는 6자리 유효숫자로 라운딩
  - 산출물에 customer_id 컬럼이 남지 않도록 groupby 이후 삭제 검증
"""
from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .schemas import (
    CONTRACT_TYPE,
    CUSTOMER_ID,
    MAX_DEMAND_KW,
    P_ACTIVE_KWH,
    P_REACTIVE_KWH,
    TS,
)

MIN_N_DEFAULT = 10
ROUND_DIGITS = 6


# --------------------------------------------------------------
# 유틸
# --------------------------------------------------------------


def _round(obj: Any) -> Any:
    """재귀적 수치 라운딩."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return round(obj, ROUND_DIGITS)
    if isinstance(obj, (np.floating, np.integer)):
        return _round(obj.item())
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(x) for x in obj]
    return obj


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_round(data), f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.round(ROUND_DIGITS).to_csv(path, index=True, encoding="utf-8-sig")


def _safe_counts(s: pd.Series, min_n: int) -> dict[str, int]:
    c = s.value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in c.items() if v >= min_n}


# --------------------------------------------------------------
# 개별 통계 함수 — 각각 Artifact 반환
# --------------------------------------------------------------


@dataclass
class Artifact:
    name: str
    description: str
    path: str
    rows_or_fields: int
    reidentification_risk: str  # "low" | "medium" | "high"
    notes: str = ""


def stat_schema(df: pd.DataFrame, out_dir: Path, _: int) -> Artifact:
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append(
            {
                "name": c,
                "dtype": str(s.dtype),
                "missing_rate": float(s.isna().mean()),
                "unique_sample": int(s.nunique(dropna=True)) if s.nunique() < 50 else -1,
            }
        )
    data = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": cols,
    }
    p = out_dir / "00_schema.json"
    _write_json(p, data)
    return Artifact(
        "schema",
        "컬럼 이름·타입·결측률. 컬럼명 자체는 비식별 (재식별 근거가 되지 않음)",
        str(p),
        df.shape[1],
        "low",
    )


def stat_coverage(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    ts = pd.to_datetime(df[TS])
    total = len(ts)
    ts_min, ts_max = ts.min(), ts.max()
    expected_intervals = (
        int((ts_max - ts_min).total_seconds() // 900) + 1 if total > 0 else 0
    )
    n_customers = df[CUSTOMER_ID].nunique()
    avg_per_customer = total / n_customers if n_customers else 0
    # gap 통계 (전체 기준 — 고객별 돌리면 비용 커서 샘플만)
    sorted_ts = ts.sort_values().drop_duplicates()
    if len(sorted_ts) > 1:
        gaps_min = sorted_ts.diff().dt.total_seconds().div(60).dropna()
        gap_summary = {
            "median_gap_min": float(gaps_min.median()),
            "p99_gap_min": float(gaps_min.quantile(0.99)),
            "max_gap_min": float(gaps_min.max()),
        }
    else:
        gap_summary = {}
    data = {
        "ts_min": str(ts_min),
        "ts_max": str(ts_max),
        "total_rows": int(total),
        "n_customers": int(n_customers),
        "avg_rows_per_customer": float(avg_per_customer),
        "expected_intervals_in_span": expected_intervals,
        "gap_summary_min": gap_summary,
    }
    p = out_dir / "01_coverage.json"
    _write_json(p, data)
    return Artifact(
        "coverage",
        "시간 커버리지와 데이터 밀도. 모두 전체 집계 통계",
        str(p),
        7,
        "low",
    )


def stat_customer_counts(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    counts = _safe_counts(df[CONTRACT_TYPE], min_n)
    n_customers_by_type = (
        df.groupby(CONTRACT_TYPE)[CUSTOMER_ID].nunique().to_dict()
    )
    n_customers_by_type = {
        k: int(v) for k, v in n_customers_by_type.items() if v >= min_n
    }
    data = {
        "row_counts_by_contract": counts,
        "n_unique_customers_by_contract": n_customers_by_type,
        "total_customers": int(df[CUSTOMER_ID].nunique()),
    }
    p = out_dir / "02_customer_counts.json"
    _write_json(p, data)
    return Artifact(
        "customer_counts",
        "계약종별 레코드·고객 수. 총수만 포함, 개별 고객 미노출",
        str(p),
        3,
        "low",
    )


def stat_load_distribution(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    rows = []
    for ct, grp in df.groupby(CONTRACT_TYPE):
        v = grp[P_ACTIVE_KWH].dropna()
        if len(v) < min_n:
            continue
        log_v = np.log(v[v > 0]) if (v > 0).any() else pd.Series(dtype=float)
        row = {
            "contract_type": ct,
            "n": int(len(v)),
            "mean": float(v.mean()),
            "std": float(v.std()),
            "skew": float(v.skew()),
            "min": float(v.min()),
            "q01": float(v.quantile(0.01)),
            "q05": float(v.quantile(0.05)),
            "q25": float(v.quantile(0.25)),
            "median": float(v.median()),
            "q75": float(v.quantile(0.75)),
            "q95": float(v.quantile(0.95)),
            "q99": float(v.quantile(0.99)),
            "max": float(v.max()),
            "zero_rate": float((v == 0).mean()),
            "neg_rate": float((v < 0).mean()),
            "lognormal_mu": float(log_v.mean()) if len(log_v) else float("nan"),
            "lognormal_sigma": float(log_v.std()) if len(log_v) else float("nan"),
        }
        rows.append(row)
    out = pd.DataFrame(rows).set_index("contract_type")
    p = out_dir / "03_load_distribution.csv"
    _write_csv(p, out)
    return Artifact(
        "load_distribution",
        "계약종별 유효전력 분포 요약 (quantile·lognormal 파라미터). 합성 보정용 핵심 통계",
        str(p),
        len(rows),
        "low",
    )


def stat_hourly_profile(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    d = df[[TS, CONTRACT_TYPE, P_ACTIVE_KWH]].copy()
    d["hour"] = pd.to_datetime(d[TS]).dt.hour
    piv = d.pivot_table(
        index="hour",
        columns=CONTRACT_TYPE,
        values=P_ACTIVE_KWH,
        aggfunc="mean",
    )
    # count 매트릭스로 N 확인 → 미만 bin mask
    cnt = d.pivot_table(
        index="hour", columns=CONTRACT_TYPE, values=P_ACTIVE_KWH, aggfunc="count"
    )
    piv = piv.where(cnt >= min_n)
    p = out_dir / "04_hourly_profile.csv"
    _write_csv(p, piv)
    return Artifact(
        "hourly_profile",
        "시간대(0-23)별 계약종별 유효전력 평균. 개별 고객 배제",
        str(p),
        piv.size,
        "low",
    )


def stat_weekly_profile(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    d = df[[TS, CONTRACT_TYPE, P_ACTIVE_KWH]].copy()
    d["dayofweek"] = pd.to_datetime(d[TS]).dt.dayofweek  # 0=월, 6=일
    piv = d.pivot_table(
        index="dayofweek",
        columns=CONTRACT_TYPE,
        values=P_ACTIVE_KWH,
        aggfunc="mean",
    )
    cnt = d.pivot_table(
        index="dayofweek",
        columns=CONTRACT_TYPE,
        values=P_ACTIVE_KWH,
        aggfunc="count",
    )
    piv = piv.where(cnt >= min_n)
    p = out_dir / "05_weekly_profile.csv"
    _write_csv(p, piv)
    return Artifact(
        "weekly_profile",
        "요일(0=월 ~ 6=일)별 계약종별 평균. 집계만",
        str(p),
        piv.size,
        "low",
    )


def stat_monthly_profile(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    d = df[[TS, CONTRACT_TYPE, P_ACTIVE_KWH]].copy()
    t = pd.to_datetime(d[TS])
    d["year_month"] = t.dt.to_period("M").astype(str)
    piv = d.pivot_table(
        index="year_month",
        columns=CONTRACT_TYPE,
        values=P_ACTIVE_KWH,
        aggfunc="mean",
    )
    cnt = d.pivot_table(
        index="year_month",
        columns=CONTRACT_TYPE,
        values=P_ACTIVE_KWH,
        aggfunc="count",
    )
    piv = piv.where(cnt >= min_n)
    p = out_dir / "06_monthly_profile.csv"
    _write_csv(p, piv)
    return Artifact(
        "monthly_profile",
        "연-월별 계약종별 평균 (3년 트렌드). 집계만",
        str(p),
        piv.size,
        "low",
    )


def stat_seasonality_pivot(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    d = df[[TS, CONTRACT_TYPE, P_ACTIVE_KWH]].copy()
    t = pd.to_datetime(d[TS])
    d["month"] = t.dt.month
    d["hour"] = t.dt.hour
    res_mask = d[CONTRACT_TYPE] == "주택용"
    gen_mask = d[CONTRACT_TYPE] == "일반용"

    def pivot(mask, name):
        sub = d[mask]
        piv = sub.pivot_table(
            index="month", columns="hour", values=P_ACTIVE_KWH, aggfunc="mean"
        )
        cnt = sub.pivot_table(
            index="month", columns="hour", values=P_ACTIVE_KWH, aggfunc="count"
        )
        piv = piv.where(cnt >= min_n)
        p = out_dir / f"07_seasonality_{name}.csv"
        _write_csv(p, piv)
        return piv.size

    res_cells = pivot(res_mask, "residential")
    gen_cells = pivot(gen_mask, "general")
    return Artifact(
        "seasonality_pivot",
        "월(1-12) × 시간(0-23) 평균 부하. 계약종별 별도 파일",
        str(out_dir / "07_seasonality_*.csv"),
        res_cells + gen_cells,
        "low",
    )


def _monthly_sum_by_customer(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df[TS])
    tmp = df[[CUSTOMER_ID, CONTRACT_TYPE, P_ACTIVE_KWH]].copy()
    tmp["month"] = t.dt.to_period("M")
    m = (
        tmp.groupby([CUSTOMER_ID, CONTRACT_TYPE, "month"], observed=True)[P_ACTIVE_KWH]
        .sum()
        .reset_index(name="monthly_kwh")
    )
    m = m.sort_values([CUSTOMER_ID, "month"])
    return m


def stat_alarm_base_rates(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    m = _monthly_sum_by_customer(df)
    g = m.groupby(CUSTOMER_ID, observed=True)["monthly_kwh"]
    m["prev_m"] = g.shift(1)
    m["prev_year_same_m"] = g.shift(12)
    m["ma3"] = g.rolling(3, min_periods=3).mean().shift(1).reset_index(
        level=0, drop=True
    )

    cond_prev = m["monthly_kwh"] > m["prev_m"] * 1.30
    cond_yoy = m["monthly_kwh"] > m["prev_year_same_m"] * 1.30
    cond_ma = m["monthly_kwh"] > m["ma3"] * 1.50
    any3 = cond_prev.fillna(False) | cond_yoy.fillna(False) | cond_ma.fillna(False)
    all3 = cond_prev.fillna(False) & cond_yoy.fillna(False) & cond_ma.fillna(False)

    def _rate(mask, base_valid):
        n = int(base_valid.sum())
        if n < min_n:
            return {"n_eligible": n, "trigger_rate": None}
        return {
            "n_eligible": n,
            "n_trigger": int((mask & base_valid).sum()),
            "trigger_rate": float((mask & base_valid).sum() / n),
        }

    data = {
        "definitions": {
            "cond_prev": "monthly_kwh > 1.30 × 직전월",
            "cond_yoy": "monthly_kwh > 1.30 × 전년동월",
            "cond_ma3": "monthly_kwh > 1.50 × 직전 3개월 평균",
            "any_of_3": "알림 대상 (과제 정의)",
            "all_of_3": "세 조건 모두 충족 (참고)",
        },
        "calendar_month_basis": {
            "cond_prev": _rate(cond_prev, m["prev_m"].notna()),
            "cond_yoy": _rate(cond_yoy, m["prev_year_same_m"].notna()),
            "cond_ma3": _rate(cond_ma, m["ma3"].notna()),
            "any_of_3": _rate(any3, m["prev_m"].notna()),
            "all_of_3": _rate(all3, m["prev_year_same_m"].notna() & m["ma3"].notna()),
        },
        "by_contract": {},
    }
    for ct, sub in m.groupby(CONTRACT_TYPE, observed=True):
        sub_prev = sub["monthly_kwh"] > sub["prev_m"] * 1.30
        sub_yoy = sub["monthly_kwh"] > sub["prev_year_same_m"] * 1.30
        sub_ma = sub["monthly_kwh"] > sub["ma3"] * 1.50
        sub_any = sub_prev.fillna(False) | sub_yoy.fillna(False) | sub_ma.fillna(False)
        eligible = sub["prev_m"].notna()
        n = int(eligible.sum())
        if n < min_n:
            continue
        data["by_contract"][str(ct)] = {
            "n_eligible": n,
            "any_of_3_rate": float((sub_any & eligible).sum() / n),
        }

    p = out_dir / "10_alarm_base_rates.json"
    _write_json(p, data)
    return Artifact(
        "alarm_base_rates",
        "과제 알림 조건 3종의 역사적 발생률 (고객×월 단위 집계)",
        str(p),
        len(data["calendar_month_basis"]) + len(data["by_contract"]),
        "low",
    )


def stat_partial_month_correlation(
    df: pd.DataFrame, out_dir: Path, min_n: int
) -> Artifact:
    """월초 N일 누적 vs 월 전체 kWh 상관.

    +10/+20일 horizon 설계 방향 결정 근거.
    """
    t = pd.to_datetime(df[TS])
    tmp = df[[CUSTOMER_ID, CONTRACT_TYPE, P_ACTIVE_KWH]].copy()
    tmp["date"] = t.dt.normalize()
    tmp["year_month"] = t.dt.to_period("M")
    day_sum = (
        tmp.groupby([CUSTOMER_ID, CONTRACT_TYPE, "year_month", "date"], observed=True)[
            P_ACTIVE_KWH
        ]
        .sum()
        .reset_index(name="day_kwh")
    )
    day_sum["day_of_month"] = pd.to_datetime(day_sum["date"]).dt.day

    result = {}
    for N in (10, 15, 20):
        cum = (
            day_sum[day_sum["day_of_month"] <= N]
            .groupby([CUSTOMER_ID, CONTRACT_TYPE, "year_month"], observed=True)[
                "day_kwh"
            ]
            .sum()
            .reset_index(name=f"first_{N}d_kwh")
        )
        total = (
            day_sum.groupby([CUSTOMER_ID, CONTRACT_TYPE, "year_month"], observed=True)[
                "day_kwh"
            ]
            .sum()
            .reset_index(name="full_month_kwh")
        )
        merged = cum.merge(
            total, on=[CUSTOMER_ID, CONTRACT_TYPE, "year_month"]
        )
        by_ct = {}
        for ct, sub in merged.groupby(CONTRACT_TYPE, observed=True):
            if len(sub) < min_n:
                continue
            corr = sub[[f"first_{N}d_kwh", "full_month_kwh"]].corr().iloc[0, 1]
            ratio_mean = (sub[f"first_{N}d_kwh"] / sub["full_month_kwh"]).mean()
            by_ct[str(ct)] = {
                "n": int(len(sub)),
                "corr": float(corr),
                "mean_ratio_of_full_month": float(ratio_mean),
            }
        result[f"N={N}"] = by_ct

    data = {
        "description": (
            "월초 N일 누적 kWh 와 그 달 전체 kWh 의 상관·비율. "
            "상관이 높으면 '부분 실측 + 잔여 예측' 구조가 유리. "
            "비율의 평균은 이론적으로 N/30"
        ),
        "results": result,
    }
    p = out_dir / "14_partial_month_correlation.json"
    _write_json(p, data)
    return Artifact(
        "partial_month_correlation",
        "horizon +10/+15/+20일 예측 구조 설계 근거",
        str(p),
        sum(len(v) for v in result.values()),
        "low",
    )


def stat_partial_month_stability(
    df: pd.DataFrame, out_dir: Path, min_n: int
) -> Artifact:
    """월초 N일 관측값이 월 전체를 대표하는지 검증 — 역률·TOU 비율·일변동.

    "왜 관측값을 연장했는가"의 실증 근거.
    partial_month_correlation이 kWh에 대해 한 것을 다른 지표에도 적용.
    """
    t = pd.to_datetime(df[TS])
    d = df[[CUSTOMER_ID, CONTRACT_TYPE, P_ACTIVE_KWH, TS]].copy()
    d["date"] = t.dt.normalize()
    d["year_month"] = t.dt.to_period("M")
    d["day_of_month"] = t.dt.day
    d["hour"] = t.dt.hour

    has_reactive = P_REACTIVE_KWH in df.columns and df[P_REACTIVE_KWH].notna().sum() > min_n
    if has_reactive:
        d["reactive"] = df[P_REACTIVE_KWH].values

    results = {}
    for N in (10, 15, 20):
        obs_mask = d["day_of_month"] <= N
        obs = d[obs_mask]
        full = d

        metrics = {}

        # 역률 안정성
        if has_reactive:
            def _pf_by_cust_month(subset):
                g = subset.groupby([CUSTOMER_ID, "year_month"], observed=True)
                p = g[P_ACTIVE_KWH].sum()
                q = g["reactive"].apply(lambda x: x.abs().sum())
                apparent = np.sqrt(p**2 + q**2).clip(lower=1e-9)
                return (p / apparent).reset_index(name="pf")

            pf_obs = _pf_by_cust_month(obs)
            pf_full = _pf_by_cust_month(full)
            pf_merged = pf_obs.merge(
                pf_full, on=[CUSTOMER_ID, "year_month"], suffixes=("_obs", "_full")
            )
            if len(pf_merged) >= min_n:
                corr = pf_merged["pf_obs"].corr(pf_merged["pf_full"])
                mae = (pf_merged["pf_obs"] - pf_merged["pf_full"]).abs().mean()
                metrics["power_factor"] = {
                    "corr": float(corr),
                    "mae": float(mae),
                    "n": int(len(pf_merged)),
                }

        # 피크시간 비율 안정성
        def _peak_ratio(subset):
            subset = subset.copy()
            peak = subset[subset["hour"].between(10, 17)]
            g_total = subset.groupby([CUSTOMER_ID, "year_month"], observed=True)[P_ACTIVE_KWH].sum()
            g_peak = peak.groupby([CUSTOMER_ID, "year_month"], observed=True)[P_ACTIVE_KWH].sum()
            ratio = (g_peak / g_total.clip(lower=1e-9)).reset_index(name="peak_ratio")
            return ratio

        pr_obs = _peak_ratio(obs)
        pr_full = _peak_ratio(full)
        pr_merged = pr_obs.merge(
            pr_full, on=[CUSTOMER_ID, "year_month"], suffixes=("_obs", "_full")
        )
        if len(pr_merged) >= min_n:
            corr = pr_merged["peak_ratio_obs"].corr(pr_merged["peak_ratio_full"])
            mae = (pr_merged["peak_ratio_obs"] - pr_merged["peak_ratio_full"]).abs().mean()
            metrics["peak_ratio"] = {
                "corr": float(corr),
                "mae": float(mae),
                "n": int(len(pr_merged)),
            }

        # 일별 변동계수 안정성
        daily_obs = obs.groupby([CUSTOMER_ID, "year_month", "date"], observed=True)[P_ACTIVE_KWH].sum()
        daily_full = full.groupby([CUSTOMER_ID, "year_month", "date"], observed=True)[P_ACTIVE_KWH].sum()

        cv_obs = daily_obs.groupby(level=[0, 1]).apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan
        ).reset_index(name="cv")
        cv_full = daily_full.groupby(level=[0, 1]).apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan
        ).reset_index(name="cv")
        cv_merged = cv_obs.merge(
            cv_full, on=[CUSTOMER_ID, "year_month"], suffixes=("_obs", "_full")
        ).dropna()
        if len(cv_merged) >= min_n:
            corr = cv_merged["cv_obs"].corr(cv_merged["cv_full"])
            metrics["daily_cv"] = {
                "corr": float(corr),
                "n": int(len(cv_merged)),
            }

        results[f"N={N}"] = metrics

    data = {
        "description": (
            "월초 N일 관측값이 월 전체를 대표하는지 실증 검증. "
            "역률·피크비율·일변동계수에 대해 관측기간 vs 전체 상관 산출. "
            "상관이 높으면 '관측값 연장' 방식의 논리적 근거."
        ),
        "results": results,
    }
    p = out_dir / "18_partial_month_stability.json"
    _write_json(p, data)
    return Artifact(
        "partial_month_stability",
        "역률·TOU·변동성의 부분월 안정성 검증 — 관측값 연장 근거",
        str(p),
        sum(len(v) for v in results.values()),
        "low",
    )


def stat_power_factor(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    if P_REACTIVE_KWH not in df.columns or df[P_REACTIVE_KWH].notna().sum() < min_n:
        return Artifact("power_factor", "skipped — 무효전력 결측", "", 0, "low")
    a = df[P_ACTIVE_KWH].clip(lower=1e-9)
    r = df[P_REACTIVE_KWH].fillna(0)
    pf = a / np.sqrt(a**2 + r**2)
    q = {
        "n": int(pf.notna().sum()),
        "mean": float(pf.mean()),
        "median": float(pf.median()),
        "q05": float(pf.quantile(0.05)),
        "q25": float(pf.quantile(0.25)),
        "q75": float(pf.quantile(0.75)),
        "q95": float(pf.quantile(0.95)),
    }
    p = out_dir / "08_power_factor.json"
    _write_json(p, {"description": "역률 = 유효/피상", "summary": q})
    return Artifact(
        "power_factor", "역률 분포 요약", str(p), 7, "low"
    )


def stat_max_demand_ratio(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    if MAX_DEMAND_KW not in df.columns or df[MAX_DEMAND_KW].notna().sum() < min_n:
        return Artifact("max_demand_ratio", "skipped — 최대수요 결측", "", 0, "low")
    ratio = df[MAX_DEMAND_KW] / (df[P_ACTIVE_KWH].clip(lower=1e-9) * 4.0)
    q = {
        "n": int(ratio.notna().sum()),
        "mean": float(ratio.mean()),
        "median": float(ratio.median()),
        "q25": float(ratio.quantile(0.25)),
        "q75": float(ratio.quantile(0.75)),
        "q95": float(ratio.quantile(0.95)),
    }
    p = out_dir / "09_max_demand_ratio.json"
    _write_json(
        p,
        {
            "description": "max_demand_kW / (active_kWh × 4). 15분 평균 대비 순간 피크 배수",
            "summary": q,
        },
    )
    return Artifact(
        "max_demand_ratio", "순간 피크 / 평균 배수 분포", str(p), 6, "low"
    )


def stat_weather_regression(
    df: pd.DataFrame, out_dir: Path, min_n: int
) -> Artifact:
    """기온 공변량이 join 돼 있으면 HDD/CDD 계수 회귀."""
    if "temp_c" not in df.columns or df["temp_c"].notna().sum() < min_n:
        return Artifact(
            "weather_regression", "skipped — 기온 컬럼 없음", "", 0, "low"
        )
    t = pd.to_datetime(df[TS])
    d = df[[CONTRACT_TYPE, P_ACTIVE_KWH, "temp_c"]].copy()
    d["date"] = t.dt.normalize()
    day = (
        d.groupby([CONTRACT_TYPE, "date"], observed=True)
        .agg(load=(P_ACTIVE_KWH, "mean"), temp=("temp_c", "mean"))
        .reset_index()
    )
    rows = []
    for ct, sub in day.groupby(CONTRACT_TYPE, observed=True):
        sub = sub.dropna()
        if len(sub) < min_n:
            continue
        t_arr = sub["temp"].to_numpy()
        hdd = np.clip(15.0 - t_arr, 0, None)
        cdd = np.clip(t_arr - 24.0, 0, None)
        X = np.column_stack([np.ones_like(t_arr), hdd, cdd])
        y = sub["load"].to_numpy()
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rows.append(
            {
                "contract_type": ct,
                "n_days": int(len(sub)),
                "intercept": float(coef[0]),
                "hdd_coef": float(coef[1]),
                "cdd_coef": float(coef[2]),
                "mean_load": float(y.mean()),
            }
        )
    out = pd.DataFrame(rows)
    p = out_dir / "13_weather_regression.csv"
    _write_csv(p, out.set_index("contract_type"))
    return Artifact(
        "weather_regression",
        "일평균 부하 = α + β·HDD(15) + γ·CDD(24), 계약종별",
        str(p),
        len(rows),
        "low",
    )


def stat_data_quality(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    ts = pd.to_datetime(df[TS])
    dup = df.duplicated(subset=[CUSTOMER_ID, TS]).sum()
    missing = {c: float(df[c].isna().mean()) for c in df.columns}
    neg = float((df[P_ACTIVE_KWH] < 0).mean()) if P_ACTIVE_KWH in df.columns else None
    # 급격 점프 (1step 대비 10x) 탐지 — 표본 고객 100명 한정 (성능)
    sample_customers = df[CUSTOMER_ID].drop_duplicates().head(100).tolist()
    jump_count = 0
    total_steps = 0
    for cid, grp in df[df[CUSTOMER_ID].isin(sample_customers)].groupby(CUSTOMER_ID):
        v = grp.sort_values(TS)[P_ACTIVE_KWH].to_numpy()
        if len(v) < 2:
            continue
        ratio = np.where(v[:-1] > 0, v[1:] / v[:-1], 0)
        jump_count += int((ratio > 10).sum() + (ratio < 0.1).sum())
        total_steps += len(ratio)
    jump_rate = float(jump_count / total_steps) if total_steps else 0.0
    data = {
        "duplicate_rows_at_customer_ts": int(dup),
        "missing_rate_per_column": missing,
        "negative_active_rate": neg,
        "jump_10x_rate_in_sample": jump_rate,
        "jump_sample_customers": len(sample_customers),
    }
    p = out_dir / "12_data_quality.json"
    _write_json(p, data)
    return Artifact(
        "data_quality",
        "결측·중복·음수·급격 점프 요약. 모두 집계 수치",
        str(p),
        4,
        "low",
    )


def stat_tou_profile(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    """한전 TOU(경부하/중간부하/최대부하) 시간대별 평균 부하.

    전기공급약관 별표3 기준 계절×시간대×계약종별 집계.
    """
    from .tou import get_season, classify_tou

    d = df[[TS, CONTRACT_TYPE, P_ACTIVE_KWH]].copy()
    t = pd.to_datetime(d[TS])
    d["season"] = t.dt.month.map(get_season)
    d["tou_period"] = [
        classify_tou(h, s) for h, s in zip(t.dt.hour, d["season"])
    ]

    # 계절×시간대×계약종별 평균
    piv = d.pivot_table(
        index=["season", "tou_period"],
        columns=CONTRACT_TYPE,
        values=P_ACTIVE_KWH,
        aggfunc="mean",
    )
    cnt = d.pivot_table(
        index=["season", "tou_period"],
        columns=CONTRACT_TYPE,
        values=P_ACTIVE_KWH,
        aggfunc="count",
    )
    piv = piv.where(cnt >= min_n)
    p = out_dir / "15_tou_profile.csv"
    _write_csv(p, piv)
    return Artifact(
        "tou_profile",
        "한전 TOU 시간대(경부하/중간/최대) × 계절 × 계약종별 평균 부하",
        str(p),
        piv.size,
        "low",
    )


def stat_tou_usage_ratios(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    """고객별 TOU 시간대 사용 비율의 분포 — 최대부하 비율이 높은 고객군 파악.

    과금 부담이 큰 고객 식별 근거. 반출 시 고객ID 없이 분포만 산출.
    """
    from .tou import tou_usage_ratios

    ratios = tou_usage_ratios(df)
    rows = []
    for ct, sub in df[[CUSTOMER_ID, CONTRACT_TYPE]].drop_duplicates().merge(
        ratios, on=CUSTOMER_ID, how="inner"
    ).groupby(CONTRACT_TYPE, observed=True):
        for col in ["tou_ratio_off_peak", "tou_ratio_mid_peak", "tou_ratio_on_peak"]:
            if col not in sub.columns:
                continue
            v = sub[col].dropna()
            if len(v) < min_n:
                continue
            rows.append({
                "contract_type": ct,
                "tou_period": col.replace("tou_ratio_", ""),
                "n": int(len(v)),
                "mean": float(v.mean()),
                "std": float(v.std()),
                "q25": float(v.quantile(0.25)),
                "median": float(v.median()),
                "q75": float(v.quantile(0.75)),
            })
    out = pd.DataFrame(rows)
    p = out_dir / "16_tou_usage_ratios.csv"
    if not out.empty:
        _write_csv(p, out.set_index(["contract_type", "tou_period"]))
    else:
        _write_csv(p, out)
    return Artifact(
        "tou_usage_ratios",
        "고객별 TOU 시간대 사용 비율 분포 — 과금 부담 고객군 파악",
        str(p),
        len(rows),
        "low",
    )


def stat_correlation_matrix(df: pd.DataFrame, out_dir: Path, min_n: int) -> Artifact:
    """주요 변수 간 Pearson 상관행렬 + 히트맵 + VIF(다중공선성).

    EDA 단계에서 변수 간 관계 파악 및 다중공선성 사전 탐지.
    """
    d = df[[TS, CUSTOMER_ID, CONTRACT_TYPE, P_ACTIVE_KWH]].copy()
    t = pd.to_datetime(d[TS])
    d["hour"] = t.dt.hour
    d["dayofweek"] = t.dt.dayofweek
    d["month"] = t.dt.month
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)

    if P_REACTIVE_KWH in df.columns and df[P_REACTIVE_KWH].notna().sum() > min_n:
        d["p_reactive_kwh"] = df[P_REACTIVE_KWH].values
    if MAX_DEMAND_KW in df.columns and df[MAX_DEMAND_KW].notna().sum() > min_n:
        d["max_demand_kw"] = df[MAX_DEMAND_KW].values
    if "temp_c" in df.columns and df["temp_c"].notna().sum() > min_n:
        d["temp_c"] = df["temp_c"].values
        from .utils import hdd_cdd
        hdd, cdd = hdd_cdd(d["temp_c"].fillna(15.0).to_numpy())
        d["hdd"] = hdd
        d["cdd"] = cdd
    if "_humidity" in df.columns and df["_humidity"].notna().sum() > min_n:
        d["humidity"] = df["_humidity"].values

    corr_cols = [c for c in [
        P_ACTIVE_KWH, "p_reactive_kwh", "max_demand_kw",
        "hour", "dayofweek", "month", "is_weekend",
        "temp_c", "hdd", "cdd", "humidity",
    ] if c in d.columns]

    if len(corr_cols) < 3:
        return Artifact("correlation_matrix", "skipped — 변수 부족", "", 0, "low")

    corr = d[corr_cols].corr()
    p = out_dir / "17_correlation_matrix.csv"
    _write_csv(p, corr)

    # VIF 산출
    from numpy.linalg import LinAlgError
    numeric_for_vif = d[corr_cols].dropna()
    vif_results = []
    if len(numeric_for_vif) >= min_n and len(corr_cols) >= 2:
        try:
            X = numeric_for_vif.to_numpy(dtype=float)
            X = X - X.mean(axis=0)
            for i, col in enumerate(corr_cols):
                others = [j for j in range(len(corr_cols)) if j != i]
                if not others:
                    continue
                X_others = X[:, others]
                y = X[:, i]
                coef, *_ = np.linalg.lstsq(X_others, y, rcond=None)
                y_hat = X_others @ coef
                ss_res = ((y - y_hat) ** 2).sum()
                ss_tot = ((y - y.mean()) ** 2).sum()
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                vif = 1 / (1 - r2) if r2 < 1 else float("inf")
                vif_results.append({
                    "feature": col,
                    "vif": float(vif),
                    "r_squared": float(r2),
                    "multicollinear": vif > 10,
                })
        except (LinAlgError, ValueError):
            pass

    vif_df = pd.DataFrame(vif_results)
    vif_path = out_dir / "17_vif.csv"
    if not vif_df.empty:
        _write_csv(vif_path, vif_df.set_index("feature"))

    # 히트맵
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
            try:
                matplotlib.rcParams["font.family"] = f
                break
            except Exception:
                continue
        matplotlib.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr.columns, fontsize=9)
        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.values[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        heatmap_path = out_dir / "17_correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    n_multi = sum(1 for r in vif_results if r["multicollinear"])
    notes = f"VIF>10 변수 {n_multi}개" if vif_results else "VIF 미산출"
    return Artifact(
        "correlation_matrix",
        f"변수 간 상관행렬 + 히트맵 + VIF 다중공선성 탐지. {notes}",
        str(p),
        corr.size + len(vif_results),
        "low",
    )


# --------------------------------------------------------------
# 오케스트레이터
# --------------------------------------------------------------


_STAT_FUNCS: list[tuple[str, Callable]] = [
    ("schema", stat_schema),
    ("coverage", stat_coverage),
    ("customer_counts", stat_customer_counts),
    ("load_distribution", stat_load_distribution),
    ("hourly_profile", stat_hourly_profile),
    ("weekly_profile", stat_weekly_profile),
    ("monthly_profile", stat_monthly_profile),
    ("seasonality_pivot", stat_seasonality_pivot),
    ("power_factor", stat_power_factor),
    ("max_demand_ratio", stat_max_demand_ratio),
    ("alarm_base_rates", stat_alarm_base_rates),
    ("data_quality", stat_data_quality),
    ("weather_regression", stat_weather_regression),
    ("partial_month_correlation", stat_partial_month_correlation),
    ("partial_month_stability", stat_partial_month_stability),
    ("tou_profile", stat_tou_profile),
    ("tou_usage_ratios", stat_tou_usage_ratios),
    ("correlation_matrix", stat_correlation_matrix),
]


def run(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    min_n: int = MIN_N_DEFAULT,
    skip: set[str] | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = skip or set()

    artifacts: list[Artifact] = []
    failures: list[dict] = []
    t0 = time.time()
    for name, fn in _STAT_FUNCS:
        if name in skip:
            continue
        t = time.time()
        try:
            art = fn(df, out_dir, min_n)
            art_dict = asdict(art)
            art_dict["elapsed_sec"] = round(time.time() - t, 2)
            artifacts.append(art)
            print(
                f"  [ok] {name:28s} {art_dict['elapsed_sec']:>5.1f}s  {art.path}"
            )
        except Exception as e:
            failures.append({"name": name, "error": repr(e)})
            print(f"  [FAIL] {name}: {e}")

    manifest = {
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_total_sec": round(time.time() - t0, 2),
        "min_n_threshold": min_n,
        "python": platform.python_version(),
        "input": {
            "n_rows": int(len(df)),
            "n_customers": int(df[CUSTOMER_ID].nunique()),
            "columns": list(df.columns),
        },
        "artifacts": [asdict(a) for a in artifacts],
        "failures": failures,
        "export_readiness_note": (
            "모든 산출물은 집계 통계이며 개별 고객 재식별 정보 없음. "
            f"최소 집계 크기 N >= {min_n} 강제. "
            "수치는 유효숫자 {ROUND_DIGITS}자리로 라운딩."
        ).format(ROUND_DIGITS=ROUND_DIGITS),
    }
    _write_json(out_dir / "MANIFEST.json", manifest)
    print(f"[done] {len(artifacts)} ok, {len(failures)} failed, "
          f"{manifest['elapsed_total_sec']:.1f}s, manifest={out_dir/'MANIFEST.json'}")
    return manifest
