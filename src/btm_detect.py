"""BTM (Behind-The-Meter) 고객 탐지.

안심구역 스카우팅 시 실 LP 에서 자가발전(태양광 등) 설치 고객을 식별.
계량 사용량 ≠ 실제 소비이므로 예측 모델에서 별도 처리가 필요한 고객군.

탐지 신호 4가지:
  1. 음수 유효전력 — 역송, 가장 확실한 증거
  2. 낮 시간 골짜기 — 12~15시 사용량이 일평균 대비 현저히 낮음
  3. 수준 급변 — 특정 시점 전후로 월 사용량이 30%+ 하락 (설치 시점 추정)
  4. 여름 역전 — 7~8월 사용량이 봄·가을(4~5, 9~10월) 대비 오히려 낮음

출력:
  - 고객별 BTM 플래그 DataFrame (DSZ 내부 전용, 반출 불가)
  - 집계 통계 dict (반출 안전)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schemas import CONTRACT_TYPE, CUSTOMER_ID, P_ACTIVE_KWH, TS


@dataclass
class BTMResult:
    customer_flags: pd.DataFrame  # customer_id, contract_type, signal_*, btm_score, is_btm
    summary: dict                 # 반출 안전 집계


# ── 신호 1: 음수 유효전력 (역송) ──


def _signal_negative(df: pd.DataFrame) -> pd.DataFrame:
    neg = df[df[P_ACTIVE_KWH] < 0].copy()
    if neg.empty:
        cust = df[[CUSTOMER_ID]].drop_duplicates()
        cust["neg_count"] = 0
        cust["neg_rate"] = 0.0
        return cust.set_index(CUSTOMER_ID)

    per_cust = neg.groupby(CUSTOMER_ID)[P_ACTIVE_KWH].agg(
        neg_count="count",
        neg_min_kwh="min",
    )
    total = df.groupby(CUSTOMER_ID)[P_ACTIVE_KWH].count().rename("total")
    per_cust = per_cust.join(total)
    per_cust["neg_rate"] = per_cust["neg_count"] / per_cust["total"]

    all_cust = pd.DataFrame(index=df[CUSTOMER_ID].unique())
    all_cust.index.name = CUSTOMER_ID
    all_cust = all_cust.join(per_cust[["neg_count", "neg_rate"]]).fillna(0)
    return all_cust


# ── 신호 2: 낮 시간 골짜기 (12~15시 vs 일평균) ──


def _signal_daytime_valley(df: pd.DataFrame) -> pd.DataFrame:
    """낮 시간 골짜기 — 계약종별 임계값 분리.

    주택용: 낮에 외출로 사용량 감소가 정상 (0.5~0.7도 일반적)
      → 임계값 0.3 (극단적으로 낮을 때만 BTM 의심)
    일반용: 주간 영업이 정상 → 낮이 오히려 높아야 함
      → 임계값 0.7 (낮이 일평균보다 낮으면 BTM 의심)
    """
    from .schemas import CONTRACT_TYPE

    d = df[[CUSTOMER_ID, TS, P_ACTIVE_KWH]].copy()
    if CONTRACT_TYPE in df.columns:
        d[CONTRACT_TYPE] = df[CONTRACT_TYPE].values
    d["hour"] = pd.to_datetime(d[TS]).dt.hour

    midday = d[d["hour"].between(12, 15)]
    midday_avg = midday.groupby(CUSTOMER_ID, observed=True)[P_ACTIVE_KWH].mean().rename("midday_avg")
    daily_avg = d.groupby(CUSTOMER_ID, observed=True)[P_ACTIVE_KWH].mean().rename("daily_avg")

    # 고객별 계약종별
    cust_ct = d.groupby(CUSTOMER_ID, observed=True)[CONTRACT_TYPE].first() if CONTRACT_TYPE in d.columns else None

    merged = pd.concat([midday_avg, daily_avg], axis=1).dropna()
    merged["midday_ratio"] = merged["midday_avg"] / merged["daily_avg"].clip(lower=1e-9)

    # 계약종별 임계값 분리
    if cust_ct is not None:
        merged = merged.join(cust_ct)
        threshold = merged[CONTRACT_TYPE].map({"주택용": 0.3, "일반용": 0.7}).fillna(0.5)
    else:
        threshold = 0.5

    merged["daytime_valley"] = (merged["midday_ratio"] < threshold).astype(int)
    return merged[["midday_ratio", "daytime_valley"]]


# ── 신호 3: 수준 급변 (월 사용량 30%+ 하락 감지) ──


def _signal_level_shift(df: pd.DataFrame) -> pd.DataFrame:
    d = df[[CUSTOMER_ID, TS, P_ACTIVE_KWH]].copy()
    d["year_month"] = pd.to_datetime(d[TS]).dt.to_period("M")
    monthly = (
        d.groupby([CUSTOMER_ID, "year_month"], observed=True)[P_ACTIVE_KWH]
        .sum()
        .reset_index(name="monthly_kwh")
        .sort_values([CUSTOMER_ID, "year_month"])
    )

    results = []
    for cid, grp in monthly.groupby(CUSTOMER_ID, observed=True):
        vals = grp["monthly_kwh"].values
        if len(vals) < 6:
            results.append({CUSTOMER_ID: cid, "level_shift": 0, "max_drop_pct": 0.0})
            continue
        # 6개월 롤링 평균, 전후 비교
        half = len(vals) // 2
        if half < 3:
            results.append({CUSTOMER_ID: cid, "level_shift": 0, "max_drop_pct": 0.0})
            continue
        best_drop = 0.0
        for i in range(3, len(vals) - 3):
            before = vals[max(0, i - 3):i].mean()
            after = vals[i:min(len(vals), i + 3)].mean()
            if before > 0:
                drop = (before - after) / before
                best_drop = max(best_drop, drop)
        results.append({
            CUSTOMER_ID: cid,
            "level_shift": int(best_drop > 0.30),
            "max_drop_pct": float(best_drop),
        })

    return pd.DataFrame(results).set_index(CUSTOMER_ID)


# ── 신호 4: 여름 역전 (7~8월 < 봄가을) ──


def _signal_summer_inversion(df: pd.DataFrame) -> pd.DataFrame:
    d = df[[CUSTOMER_ID, TS, P_ACTIVE_KWH]].copy()
    d["month"] = pd.to_datetime(d[TS]).dt.month

    summer = d[d["month"].isin([7, 8])]
    shoulder = d[d["month"].isin([4, 5, 9, 10])]

    summer_avg = summer.groupby(CUSTOMER_ID, observed=True)[P_ACTIVE_KWH].mean().rename("summer_avg")
    shoulder_avg = shoulder.groupby(CUSTOMER_ID, observed=True)[P_ACTIVE_KWH].mean().rename("shoulder_avg")

    merged = pd.concat([summer_avg, shoulder_avg], axis=1).dropna()
    merged["summer_ratio"] = merged["summer_avg"] / merged["shoulder_avg"].clip(lower=1e-9)
    # 여름이 봄가을보다 오히려 낮으면 태양광 의심
    merged["summer_inverted"] = (merged["summer_ratio"] < 0.85).astype(int)
    return merged[["summer_ratio", "summer_inverted"]]


# ── 종합 ──


def detect(
    df: pd.DataFrame,
    *,
    neg_threshold: float = 0.001,
    min_n_export: int = 10,
) -> BTMResult:
    """BTM 고객 탐지.

    Parameters
    ----------
    df : 표준 스키마 LP DataFrame (음수 값 허용 — validate(strict=False) 후 호출)
    neg_threshold : 음수 비율이 이 값 이상이면 신호 1 확정
    min_n_export : 반출 집계에서 최소 표본 크기

    Returns
    -------
    BTMResult : customer_flags (내부용) + summary (반출용)
    """
    cust_base = (
        df[[CUSTOMER_ID, CONTRACT_TYPE]]
        .drop_duplicates(CUSTOMER_ID)
        .set_index(CUSTOMER_ID)
    )

    s1 = _signal_negative(df)
    s2 = _signal_daytime_valley(df)
    s3 = _signal_level_shift(df)
    s4 = _signal_summer_inversion(df)

    flags = cust_base.copy()
    flags = flags.join(s1[["neg_count", "neg_rate"]], how="left")
    flags = flags.join(s2[["midday_ratio", "daytime_valley"]], how="left")
    flags = flags.join(s3[["max_drop_pct", "level_shift"]], how="left")
    flags = flags.join(s4[["summer_ratio", "summer_inverted"]], how="left")
    flags = flags.fillna(0)

    flags["signal_neg"] = (flags["neg_rate"] >= neg_threshold).astype(int)
    flags["signal_valley"] = flags["daytime_valley"].astype(int)
    flags["signal_shift"] = flags["level_shift"].astype(int)
    flags["signal_summer"] = flags["summer_inverted"].astype(int)

    signal_cols = ["signal_neg", "signal_valley", "signal_shift", "signal_summer"]
    flags["btm_score"] = flags[signal_cols].sum(axis=1) / len(signal_cols)
    # 2개 이상 신호 or 음수 확인 → BTM 확정
    flags["is_btm"] = (
        (flags[signal_cols].sum(axis=1) >= 2) | (flags["signal_neg"] == 1)
    ).astype(int)

    flags = flags.reset_index()

    # 반출 안전 집계
    n_total = len(flags)
    n_btm = int(flags["is_btm"].sum())
    summary = {
        "n_total_customers": n_total,
        "n_btm_detected": n_btm,
        "btm_rate": float(n_btm / n_total) if n_total else 0.0,
        "by_signal": {
            "negative_power": int(flags["signal_neg"].sum()),
            "daytime_valley": int(flags["signal_valley"].sum()),
            "level_shift": int(flags["signal_shift"].sum()),
            "summer_inversion": int(flags["signal_summer"].sum()),
        },
        "btm_score_distribution": {
            "mean": float(flags["btm_score"].mean()),
            "q25": float(flags["btm_score"].quantile(0.25)),
            "median": float(flags["btm_score"].quantile(0.5)),
            "q75": float(flags["btm_score"].quantile(0.75)),
            "max": float(flags["btm_score"].max()),
        },
    }

    # 계약종별 BTM 비율 (min_n 이상인 그룹만)
    by_ct = {}
    for ct, sub in flags.groupby(CONTRACT_TYPE):
        if len(sub) >= min_n_export:
            by_ct[str(ct)] = {
                "n": int(len(sub)),
                "n_btm": int(sub["is_btm"].sum()),
                "btm_rate": float(sub["is_btm"].mean()),
            }
    summary["by_contract"] = by_ct

    return BTMResult(customer_flags=flags, summary=summary)
