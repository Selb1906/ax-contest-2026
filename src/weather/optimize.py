"""기상 가중치 최적화 — LP 데이터 기반 역산.

참고문헌(임종훈 외 2013)의 고정 가중치를 넘어서,
실 LP 데이터와의 상관에서 최적 가중치를 직접 추정.

방법 1: LP 기반 가중치 역산 (회귀)
방법 2: 계절별 가중치 분리
방법 3: 8개 스테이션 개별 피처 투입 (SHAP으로 암묵적 가중치 도출)
방법 4: 기준온도(HDD/CDD base) 최적화
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .asos import compute_degree_hours


# ── 방법 1: LP 기반 가중치 역산 ──


def optimize_weights_regression(
    monthly_load: pd.DataFrame,
    station_cdh: pd.DataFrame,
    station_ids: list[int] | None = None,
    mode: str = "cooling",
) -> dict:
    """LP 월 사용량과 관측소별 CDH/HDH의 회귀로 최적 가중치 추정.

    Parameters
    ----------
    monthly_load : (year_month, monthly_kwh) — 전체 고객 평균 or 개별
    station_cdh : (station_id, year_month, cdh_total or hdh_total)
    station_ids : 사용할 관측소 (기본: 8대 도시)
    mode : "cooling" → cdh_total, "heating" → hdh_total

    Returns
    -------
    {station_id: weight, ...} + regression stats
    """
    if station_ids is None:
        station_ids = [108, 112, 119, 114, 133, 143, 156, 159]

    col = "cdh_total" if mode == "cooling" else "hdh_total"

    pivot = station_cdh[station_cdh["station_id"].isin(station_ids)].pivot_table(
        index="year_month", columns="station_id", values=col, aggfunc="first"
    )

    merged = monthly_load.set_index("year_month").join(pivot, how="inner")
    merged = merged.dropna()

    if len(merged) < len(station_ids) + 1:
        return {"error": "insufficient data", "n": len(merged)}

    y = merged["monthly_kwh"].values
    X = merged[station_ids].values

    # 비음수 제약 회귀 (가중치는 0 이상이어야)
    from scipy.optimize import nnls
    coef, residual = nnls(X, y)

    # 정규화 (합=1)
    total = coef.sum()
    weights_norm = coef / total if total > 0 else coef

    # R² 산출
    y_pred = X @ coef
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    result = {
        "weights": {int(s): float(w) for s, w in zip(station_ids, weights_norm)},
        "raw_coef": {int(s): float(c) for s, c in zip(station_ids, coef)},
        "r_squared": float(r2),
        "residual_norm": float(residual),
        "n_months": len(merged),
        "mode": mode,
    }
    return result


def optimize_weights_seasonal(
    monthly_load: pd.DataFrame,
    station_cdh: pd.DataFrame,
    station_ids: list[int] | None = None,
) -> dict:
    """계절별로 가중치를 분리 추정.

    여름(6~8): CDH 기반
    겨울(11~2): HDH 기반
    봄가을(3~5, 9~10): CDH+HDH 혼합
    """
    if station_ids is None:
        station_ids = [108, 112, 119, 114, 133, 143, 156, 159]

    load = monthly_load.copy()
    load["month"] = load["year_month"].dt.month if hasattr(load["year_month"], "dt") else load["year_month"].apply(lambda x: x.month)

    seasons = {
        "summer": ([6, 7, 8], "cooling"),
        "winter": ([11, 12, 1, 2], "heating"),
        "spring_fall": ([3, 4, 5, 9, 10], "cooling"),
    }

    results = {}
    for season_name, (months, mode) in seasons.items():
        season_load = load[load["month"].isin(months)].drop(columns=["month"])
        if len(season_load) < len(station_ids):
            results[season_name] = {"error": "insufficient data"}
            continue
        results[season_name] = optimize_weights_regression(
            season_load, station_cdh, station_ids, mode
        )

    return results


# ── 방법 4: 기준온도 최적화 ──


def optimize_base_temperature(
    hourly_weather: pd.DataFrame,
    monthly_load: pd.DataFrame,
    hdd_range: tuple[float, float] = (10, 20),
    cdd_range: tuple[float, float] = (20, 30),
    step: float = 1.0,
    station_id: int | None = None,
    contract_type: str | None = None,
) -> dict:
    """HDD/CDD 기준온도를 그리드 서치로 최적화.

    Parameters
    ----------
    hourly_weather : ASOS 시간별 (station_id, ts, temp_c)
    monthly_load : (year_month, monthly_kwh, [contract_type])
    hdd_range, cdd_range : 탐색 범위
    step : 탐색 간격 (°C)

    Returns
    -------
    최적 기준온도 + 각 조합의 상관계수 테이블
    """
    w = hourly_weather.copy()
    if station_id is not None:
        w = w[w["station_id"] == station_id]

    w["year_month"] = pd.to_datetime(w["ts"]).dt.to_period("M")
    temp = w[["year_month", "temp_c"]].dropna()

    load = monthly_load.copy()
    if contract_type is not None and "contract_type" in load.columns:
        load = load[load["contract_type"] == contract_type]

    hdd_bases = np.arange(hdd_range[0], hdd_range[1] + step, step)
    cdd_bases = np.arange(cdd_range[0], cdd_range[1] + step, step)

    results = []
    for hb in hdd_bases:
        for cb in cdd_bases:
            if hb >= cb:
                continue

            monthly_dh = temp.copy()
            monthly_dh["hdh"] = compute_degree_hours(monthly_dh["temp_c"].values, hb, "heating")
            monthly_dh["cdh"] = compute_degree_hours(monthly_dh["temp_c"].values, cb, "cooling")

            agg = monthly_dh.groupby("year_month").agg(
                hdh_total=("hdh", "sum"),
                cdh_total=("cdh", "sum"),
            ).reset_index()

            merged = load.set_index("year_month").join(
                agg.set_index("year_month"), how="inner"
            ).dropna()

            if len(merged) < 6:
                continue

            y = merged["monthly_kwh"].values
            X = np.column_stack([
                np.ones(len(y)),
                merged["hdh_total"].values,
                merged["cdh_total"].values,
            ])

            try:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ coef
                ss_res = ((y - y_pred) ** 2).sum()
                ss_tot = ((y - y.mean()) ** 2).sum()
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                mape = np.mean(np.abs(y - y_pred) / y) * 100

                results.append({
                    "hdd_base": float(hb),
                    "cdd_base": float(cb),
                    "r_squared": float(r2),
                    "mape_pct": float(mape),
                    "intercept": float(coef[0]),
                    "hdd_coef": float(coef[1]),
                    "cdd_coef": float(coef[2]),
                })
            except Exception:
                continue

    if not results:
        return {"error": "no valid combinations"}

    df = pd.DataFrame(results).sort_values("r_squared", ascending=False)
    best = df.iloc[0]

    return {
        "best": {
            "hdd_base": float(best["hdd_base"]),
            "cdd_base": float(best["cdd_base"]),
            "r_squared": float(best["r_squared"]),
            "mape_pct": float(best["mape_pct"]),
            "hdd_coef": float(best["hdd_coef"]),
            "cdd_coef": float(best["cdd_coef"]),
        },
        "contract_type": contract_type,
        "search_grid": df.to_dict("records"),
        "n_combinations": len(df),
    }


# ── 통합 실행 ──


def run_weather_optimization(
    hourly_weather: pd.DataFrame,
    monthly_load: pd.DataFrame,
    station_features: pd.DataFrame,
    out_dir: str | None = None,
) -> dict:
    """전체 기상 최적화 파이프라인.

    Parameters
    ----------
    hourly_weather : ASOS 시간별 원본
    monthly_load : (customer_id, year_month, monthly_kwh, contract_type)
    station_features : monthly_weather_features() 출력

    Returns
    -------
    방법 1~4 결과 종합
    """
    import json
    from pathlib import Path

    # 전체 고객 월평균 부하
    avg_load = monthly_load.groupby("year_month")["monthly_kwh"].mean().reset_index()

    print("[optimize] 방법 1: LP 기반 가중치 역산...")
    w_cooling = optimize_weights_regression(avg_load, station_features, mode="cooling")
    w_heating = optimize_weights_regression(avg_load, station_features, mode="heating")
    print(f"  cooling R²={w_cooling.get('r_squared', 'N/A')}")
    print(f"  heating R²={w_heating.get('r_squared', 'N/A')}")

    print("[optimize] 방법 2: 계절별 가중치...")
    seasonal = optimize_weights_seasonal(avg_load, station_features)
    for s, r in seasonal.items():
        r2 = r.get("r_squared", "N/A")
        print(f"  {s}: R²={r2}")

    print("[optimize] 방법 4: 기준온도 최적화...")
    base_results = {}
    for ct in ["주택용", "일반용", None]:
        label = ct if ct else "전체"
        result = optimize_base_temperature(
            hourly_weather, monthly_load,
            contract_type=ct,
        )
        base_results[label] = result
        if "best" in result:
            b = result["best"]
            print(f"  {label}: HDD base={b['hdd_base']}°C, CDD base={b['cdd_base']}°C, R²={b['r_squared']:.4f}")

    all_results = {
        "method1_weight_regression": {
            "cooling": w_cooling,
            "heating": w_heating,
        },
        "method2_seasonal_weights": seasonal,
        "method4_base_temp_optimization": base_results,
    }

    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "weather_optimization.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[saved] {p / 'weather_optimization.json'}")

    # LP 역산 가중치로 대표기상 CSV 생성 → Ablation 비교 대상
    from .asos import weighted_national_average
    weather_dir = Path("data/weather")
    weather_dir.mkdir(parents=True, exist_ok=True)

    if "weights" in w_cooling and w_cooling.get("r_squared", 0) > 0:
        try:
            lp_weights = {int(k): v for k, v in w_cooling["weights"].items()}
            avg_lp = weighted_national_average(station_features, lp_weights)
            avg_lp.to_csv(weather_dir / "national_lp_regression.csv", index=False, encoding="utf-8-sig")
            print(f"[saved] data/weather/national_lp_regression.csv (LP역산 가중치)")
        except Exception as e:
            print(f"  LP역산 CSV 생성 실패: {e}")

    for season_key, season_result in seasonal.items():
        if "weights" in season_result and season_result.get("r_squared", 0) > 0:
            try:
                s_weights = {int(k): v for k, v in season_result["weights"].items()}
                avg_s = weighted_national_average(station_features, s_weights)
                fname = f"national_seasonal_{season_key}.csv"
                avg_s.to_csv(weather_dir / fname, index=False, encoding="utf-8-sig")
                print(f"[saved] data/weather/{fname} (계절별 {season_key})")
            except Exception:
                pass

    return all_results
