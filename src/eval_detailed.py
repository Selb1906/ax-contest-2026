"""상세 평가 — sliding window + 요일/월/특수일별 오차 분석.

1년 전체를 매일 sliding하며 예측 성능을 세밀하게 분석.
보고서 핵심 테이블 + 차트 근거.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .eval import (
    build_horizon_table, daily_by_customer, monthly_by_customer,
    attach_alarm_context, regression_metrics,
)
from .schemas import CUSTOMER_ID, CONTRACT_TYPE
from .special_days import load_holidays


def sliding_evaluation(
    df: pd.DataFrame,
    predict_fn,
    *,
    meter_days: list[int] | None = None,
    horizons: tuple[int, ...] = (10, 20),
) -> pd.DataFrame:
    """매일 sliding하며 예측 + 평가.

    Parameters
    ----------
    df : LP 데이터
    predict_fn : (features_df) → pred_df with pred_monthly_kwh
    meter_days : sliding할 검침일 목록 (기본: 1~28)
    horizons : 예측 horizon

    Returns
    -------
    모든 (meter_day, horizon, customer, month)에 대한 예측·실측·오차 DataFrame
    """
    if meter_days is None:
        meter_days = list(range(1, 29))

    daily = daily_by_customer(df)
    monthly = monthly_by_customer(daily)

    all_results = []
    for md in meter_days:
        horizon_tbl = build_horizon_table(daily, horizons=horizons, meter_day=md)
        ctx = attach_alarm_context(horizon_tbl, monthly)

        pred_df = predict_fn(df, horizon_tbl)
        if pred_df is None or len(pred_df) == 0:
            continue

        merged = ctx.merge(
            pred_df, on=[CUSTOMER_ID, "year_month", "horizon_days"], how="inner"
        )
        merged["meter_day"] = md
        merged["error"] = merged["pred_monthly_kwh"] - merged["full_month_kwh"]
        merged["abs_error"] = merged["error"].abs()
        merged["pct_error"] = (
            merged["abs_error"] / merged["full_month_kwh"].clip(lower=1e-9) * 100
        )
        merged["month"] = merged["year_month"].apply(lambda x: x.month)
        all_results.append(merged)

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


def error_by_dimension(
    results: pd.DataFrame,
    holidays_path: str | None = None,
) -> dict[str, pd.DataFrame]:
    """오차를 여러 차원으로 집계.

    Returns
    -------
    dict of DataFrames:
      "by_horizon"   — horizon별
      "by_month"     — 월별
      "by_meter_day" — 검침일별
      "by_contract"  — 계약종별
      "by_special"   — 특수일 여부별 (disruption_ratio 기반)
      "overall"      — 전체 요약
    """
    def _agg(group):
        yt = group["full_month_kwh"].values
        yp = group["pred_monthly_kwh"].values
        return regression_metrics(yt, yp)

    output = {}

    # 전체
    overall = _agg(results)
    output["overall"] = pd.DataFrame([overall])

    # horizon별
    rows = []
    for h, grp in results.groupby("horizon_days"):
        m = _agg(grp)
        m["horizon_days"] = h
        rows.append(m)
    output["by_horizon"] = pd.DataFrame(rows)

    # 월별
    rows = []
    for month, grp in results.groupby("month"):
        m = _agg(grp)
        m["month"] = month
        rows.append(m)
    output["by_month"] = pd.DataFrame(rows) if rows else pd.DataFrame()

    # 검침일별
    if "meter_day" in results.columns:
        rows = []
        for md, grp in results.groupby("meter_day"):
            m = _agg(grp)
            m["meter_day"] = md
            rows.append(m)
        output["by_meter_day"] = pd.DataFrame(rows) if rows else pd.DataFrame()

    # 계약종별
    if CONTRACT_TYPE in results.columns:
        rows = []
        for ct, grp in results.groupby(CONTRACT_TYPE):
            m = _agg(grp)
            m["contract_type"] = ct
            rows.append(m)
        output["by_contract"] = pd.DataFrame(rows) if rows else pd.DataFrame()

    # 특수일 기반 (disruption_ratio가 있으면)
    try:
        from .special_days import build_special_day_features
        yms = results["year_month"].unique()
        sd = build_special_day_features(yms, holidays_path)
        results_with_sd = results.merge(
            sd[["year_month", "disruption_ratio", "n_holiday"]],
            on="year_month", how="left",
        )
        # 특수일 많은 월 vs 적은 월
        median_dr = results_with_sd["disruption_ratio"].median()
        results_with_sd["is_disrupted"] = results_with_sd["disruption_ratio"] > median_dr
        rows = []
        for flag, grp in results_with_sd.groupby("is_disrupted"):
            m = _agg(grp)
            m["is_disrupted"] = bool(flag)
            m["avg_disruption_ratio"] = float(grp["disruption_ratio"].mean())
            rows.append(m)
        output["by_special"] = pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        pass

    # 날짜별 오차 (반출 안전 — 고객 수 min_n 이상 날짜만)
    if "year_month" in results.columns and "meter_day" in results.columns:
        min_n_date = 10
        date_rows = []
        for (ym, md), grp in results.groupby(["year_month", "meter_day"]):
            if len(grp) < min_n_date:
                continue
            m = _agg(grp)
            m["year_month"] = str(ym)
            m["meter_day"] = int(md)
            date_rows.append(m)
        if date_rows:
            date_df = pd.DataFrame(date_rows)
            # 오차 상위/하위
            if "mape_pct" in date_df.columns and len(date_df) > 0:
                worst = date_df.nlargest(10, "mape_pct")
                best = date_df.nsmallest(10, "mape_pct")
                output["worst_dates"] = worst
                output["best_dates"] = best
                output["by_date"] = date_df

    return output


def save_detailed_evaluation(
    results: pd.DataFrame,
    out_dir: str | Path,
    holidays_path: str | None = None,
) -> dict:
    """상세 평가 결과를 저장."""
    from .result_saver import save_dataframe, save_chart

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 원본 결과
    save_dataframe(results, out, "sliding_raw", "Sliding evaluation 전체 결과")

    # 차원별 집계
    dims = error_by_dimension(results, holidays_path)
    for name, df in dims.items():
        if len(df) > 0:
            save_dataframe(df, out, f"error_{name}", f"오차 분석: {name}")

    # 월별 MAPE 차트
    if "by_month" in dims and len(dims["by_month"]) > 0:
        by_month = dims["by_month"].sort_values("month")

        def plot_monthly(fig, ax):
            ax.bar(by_month["month"], by_month["mape_pct"], color="#0984e3", width=0.7)
            ax.set_xlabel("월")
            ax.set_ylabel("MAPE (%)")
            ax.set_xticks(range(1, 13))
            ax.grid(axis="y", alpha=0.3)
        save_chart(plot_monthly, out, "mape_by_month", by_month, "월별 MAPE")

    # 검침일별 MAPE 차트
    if "by_meter_day" in dims and len(dims["by_meter_day"]) > 0:
        by_md = dims["by_meter_day"].sort_values("meter_day")

        def plot_meter_day(fig, ax):
            ax.plot(by_md["meter_day"], by_md["mape_pct"], "-o", color="#e17055", markersize=3)
            ax.set_xlabel("검침일")
            ax.set_ylabel("MAPE (%)")
            ax.grid(alpha=0.3)
        save_chart(plot_meter_day, out, "mape_by_meter_day", by_md, "검침일별 MAPE")

    # 오차 큰/작은 날짜 (반출 안전 — customer_id 없음)
    if "worst_dates" in dims and len(dims["worst_dates"]) > 0:
        save_dataframe(dims["worst_dates"], out, "worst_dates", "오차 상위 10개 날짜")
        save_dataframe(dims["best_dates"], out, "best_dates", "오차 하위 10개 날짜")
        print(f"\n  오차 큰 날짜 (상위 5):")
        for _, row in dims["worst_dates"].head(5).iterrows():
            print(f"    {row.get('year_month','')} md={row.get('meter_day','')}: "
                  f"MAPE {row.get('mape_pct',0):.2f}%  n={row.get('n',0)}")
        print(f"  오차 작은 날짜 (하위 5):")
        for _, row in dims["best_dates"].head(5).iterrows():
            print(f"    {row.get('year_month','')} md={row.get('meter_day','')}: "
                  f"MAPE {row.get('mape_pct',0):.2f}%  n={row.get('n',0)}")

    # 요약 출력
    if "overall" in dims and len(dims["overall"]) > 0:
        ov = dims["overall"].iloc[0]
        print(f"\n  Sliding 평가 요약:")
        print(f"    MAPE:     {ov.get('mape_pct', 0):.3f}%")
        print(f"    RMSE:     {ov.get('rmse', 0):.2f}")
        print(f"    NRMSE:    {ov.get('nrmse_pct', 0):.3f}%")
        print(f"    CVRMSE:   {ov.get('cvrmse_pct', 0):.3f}%")
        print(f"    MAE:      {ov.get('mae', 0):.2f}")
        print(f"    Bias:     {ov.get('bias', 0):+.2f}")
        print(f"    Std(err): {ov.get('std_error', 0):.2f}")
        print(f"    N:        {ov.get('n', 0):,}")

    return dims
