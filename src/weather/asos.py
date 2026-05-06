"""ASOS(종관기상관측) 시간별 데이터 파서 + CDH/HDH 산출.

기상청 기상자료개방포털에서 다운로드한 CSV (euc-kr) 파싱.
8대 관측소: 서울(108), 인천(112), 수원(119), 원주(114),
          대전(133), 대구(143), 광주(156), 부산(159)

CDH/HDH는 시간별로 산출 후 월 합산 — 일평균 기반 CDD보다 정확.
시간대별 분리(주간/야간/저녁)로 주택용/일반용 민감도 차이 포착.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ASOS 컬럼 매핑 (euc-kr 원본명 → 영문)
_COL_MAP = {
    "지점": "station_id",
    "지점명": "station_name",
    "일시": "ts",
    "기온(°C)": "temp_c",
    "강수량(mm)": "precip_mm",
    "풍속(m/s)": "wind_speed",
    "풍향(16방위)": "wind_dir",
    "습도(%)": "humidity",
    "증기압(hPa)": "vapor_pressure",
    "이슬점온도(°C)": "dewpoint_c",
    "현지기압(hPa)": "pressure_local",
    "해면기압(hPa)": "pressure_sea",
    "일조(hr)": "sunshine_hr",
    "일사(MJ/m2)": "solar_mj",
    "적설(cm)": "snow_cm",
    "전운량(10분위)": "cloud_total",
    "중하층운량(10분위)": "cloud_low",
    "운형(운형약어)": "cloud_type",
    "최저운고(100m )": "cloud_base",
    "지면온도(°C)": "ground_temp_c",
}


def read_asos_csv(path: str | Path) -> pd.DataFrame:
    """단일 ASOS CSV 파일 로드."""
    df = pd.read_csv(path, encoding="euc-kr")
    df = df.rename(columns=_COL_MAP)
    df["ts"] = pd.to_datetime(df["ts"])
    numeric_cols = ["temp_c", "humidity", "wind_speed", "solar_mj",
                    "precip_mm", "dewpoint_c", "ground_temp_c"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_asos_directory(directory: str | Path) -> pd.DataFrame:
    """ASOS 디렉터리 내 모든 CSV 로드 + 합치기."""
    directory = Path(directory)
    frames = []
    for f in sorted(directory.glob("*.csv")):
        if f.name.startswith("_"):
            continue
        frames.append(read_asos_csv(f))
    if not frames:
        raise FileNotFoundError(f"No CSV files in {directory}")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["station_id", "ts"]).reset_index(drop=True)
    return df


# ── CDH/HDH 산출 ──


def compute_degree_hours(
    temp: np.ndarray,
    base: float,
    mode: str = "cooling",
) -> np.ndarray:
    """시간별 degree-hour 산출.

    mode="cooling": CDH = max(temp - base, 0)
    mode="heating": HDH = max(base - temp, 0)
    """
    if mode == "cooling":
        return np.maximum(temp - base, 0)
    else:
        return np.maximum(base - temp, 0)


def monthly_degree_hours(
    df: pd.DataFrame,
    hdd_base: float = 15.0,
    cdd_base: float = 24.0,
) -> pd.DataFrame:
    """관측소×월별 CDH/HDH 산출 — 시간대별 분리 포함.

    시간대 구분:
      daytime: 09~18시 (일반용 영업시간)
      evening: 18~23시 (주택용 퇴근 후)
      night: 23~09시 (야간)
    """
    d = df[["station_id", "ts", "temp_c"]].copy()
    d = d.dropna(subset=["temp_c"])
    d["year_month"] = d["ts"].dt.to_period("M")
    d["hour"] = d["ts"].dt.hour

    d["cdh"] = compute_degree_hours(d["temp_c"].values, cdd_base, "cooling")
    d["hdh"] = compute_degree_hours(d["temp_c"].values, hdd_base, "heating")

    # 시간대 구분
    d["period"] = "night"
    d.loc[d["hour"].between(9, 17), "period"] = "daytime"
    d.loc[d["hour"].between(18, 22), "period"] = "evening"

    # 월별 전체 합산
    total = d.groupby(["station_id", "year_month"]).agg(
        cdh_total=("cdh", "sum"),
        hdh_total=("hdh", "sum"),
        temp_mean=("temp_c", "mean"),
        temp_max=("temp_c", "max"),
        temp_min=("temp_c", "min"),
    ).reset_index()

    # 시간대별 CDH/HDH
    for period in ["daytime", "evening", "night"]:
        sub = d[d["period"] == period].groupby(["station_id", "year_month"]).agg(
            cdh=("cdh", "sum"),
            hdh=("hdh", "sum"),
        ).reset_index()
        sub = sub.rename(columns={
            "cdh": f"cdh_{period}",
            "hdh": f"hdh_{period}",
        })
        total = total.merge(sub, on=["station_id", "year_month"], how="left")

    total = total.fillna(0)
    return total


def monthly_weather_features(
    df: pd.DataFrame,
    hdd_base: float = 15.0,
    cdd_base: float = 24.0,
) -> pd.DataFrame:
    """관측소×월별 기상 피처 종합 (CDH/HDH + 습도/풍속/일사 등)."""
    dh = monthly_degree_hours(df, hdd_base, cdd_base)

    d = df[["station_id", "ts"]].copy()
    d["year_month"] = d["ts"].dt.to_period("M")

    extra_cols = ["humidity", "wind_speed", "solar_mj"]
    for col in extra_cols:
        if col in df.columns:
            d[col] = pd.to_numeric(df[col], errors="coerce")

    extra = d.groupby(["station_id", "year_month"]).agg(
        **{f"{c}_mean": (c, "mean") for c in extra_cols if c in d.columns}
    ).reset_index()

    result = dh.merge(extra, on=["station_id", "year_month"], how="left")
    return result


# ── 가중평균 (참고문헌 기반 8대 도시) ──

# 출처: 임종훈 외 (2013), 표6 개선된 인구수 기준
EIGHT_CITY_WEIGHTS = {
    108: 0.4376,  # 서울+경기북부
    112: 0.0558,  # 인천
    119: 0.0448,  # 수원(경기남부)
    114: 0.0306,  # 원주(강원)
    133: 0.0633,  # 대전+충남+충북
    143: 0.0838,  # 대구+경북
    156: 0.0647,  # 광주+전남+전북
    159: 0.2194,  # 부산+경남+울산
}


def weighted_national_average(
    monthly_features: pd.DataFrame,
    weights: dict[int, float] | None = None,
) -> pd.DataFrame:
    """8대 도시 가중평균으로 전국 대표 기상 시계열 생성."""
    if weights is None:
        weights = EIGHT_CITY_WEIGHTS

    w_total = sum(weights.values())
    weights_norm = {k: v / w_total for k, v in weights.items()}

    numeric_cols = [c for c in monthly_features.columns
                    if c not in ("station_id", "year_month", "station_name")]

    rows = []
    for ym, grp in monthly_features.groupby("year_month"):
        row = {"year_month": ym}
        for col in numeric_cols:
            weighted_sum = 0.0
            weight_sum = 0.0
            for _, r in grp.iterrows():
                stn = r["station_id"]
                w = weights_norm.get(stn, 0)
                if w > 0 and pd.notna(r[col]):
                    weighted_sum += r[col] * w
                    weight_sum += w
            row[col] = weighted_sum / weight_sum if weight_sum > 0 else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ── 부분월 분리 (관측/잔여 구간) ──


def split_observed_remainder(
    df: pd.DataFrame,
    year_month: pd.Period,
    horizon_days: int,
    station_id: int | None = None,
    hdd_base: float = 15.0,
    cdd_base: float = 24.0,
) -> dict:
    """특정 월·horizon에 대해 관측/잔여 구간 기상 분리."""
    mask = df["ts"].dt.to_period("M") == year_month
    if station_id is not None:
        mask &= df["station_id"] == station_id
    sub = df[mask].copy()
    if sub.empty:
        return {}

    sub["day"] = sub["ts"].dt.day
    obs = sub[sub["day"] <= horizon_days]
    rem = sub[sub["day"] > horizon_days]

    def _agg(part):
        if part.empty:
            return {}
        t = part["temp_c"].dropna()
        cdh = compute_degree_hours(t.values, cdd_base, "cooling").sum()
        hdh = compute_degree_hours(t.values, hdd_base, "heating").sum()
        return {
            "temp_mean": float(t.mean()),
            "cdh": float(cdh),
            "hdh": float(hdh),
        }

    obs_agg = _agg(obs)
    rem_agg = _agg(rem)
    full_agg = _agg(sub)

    return {
        "observed": {f"obs_{k}": v for k, v in obs_agg.items()},
        "remainder": {f"rem_{k}": v for k, v in rem_agg.items()},
        "full": {f"full_{k}": v for k, v in full_agg.items()},
    }


# ── 기상 민감도 분석 ──


def weather_sensitivity_analysis(
    model_predict_fn,
    features: pd.DataFrame,
    weather_cols: list[str],
    perturbations: list[float] = [-3, -2, -1, 0, 1, 2, 3],
) -> pd.DataFrame:
    """기상 변수를 교란했을 때 예측 MAPE 변화 측정.

    Parameters
    ----------
    model_predict_fn : features → pred_kwh array
    features : 피처 DataFrame (full_month_kwh 포함)
    weather_cols : 교란할 기상 컬럼 목록
    perturbations : 기온 교란 값 (°C)
    """
    base_pred = model_predict_fn(features)
    actual = features["full_month_kwh"].values
    base_mape = np.mean(np.abs(base_pred - actual) / actual) * 100

    rows = [{"perturbation": 0, "source": "actual", "mape_pct": base_mape}]

    for delta in perturbations:
        if delta == 0:
            continue
        perturbed = features.copy()
        for col in weather_cols:
            if col in perturbed.columns:
                perturbed[col] = perturbed[col] + delta
        pred = model_predict_fn(perturbed)
        mape = np.mean(np.abs(pred - actual) / actual) * 100
        rows.append({
            "perturbation": delta,
            "source": f"temp{delta:+.0f}°C",
            "mape_pct": mape,
        })

    # 평년값 시나리오: 기상 피처를 월별 평균으로 대체
    climatology = features.copy()
    for col in weather_cols:
        if col in climatology.columns:
            monthly_mean = climatology.groupby("month")[col].transform("mean")
            climatology[col] = monthly_mean
    clim_pred = model_predict_fn(climatology)
    clim_mape = np.mean(np.abs(clim_pred - actual) / actual) * 100
    rows.append({"perturbation": None, "source": "climatology", "mape_pct": clim_mape})

    return pd.DataFrame(rows)
