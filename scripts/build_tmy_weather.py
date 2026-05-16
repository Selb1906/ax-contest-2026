"""TMY 기반 월간 기상 피처 생성 — ASOS 실측과 동일 형태.

출력:
  data/weather/tmy_raw/station_*.csv         — TMY 순정
  data/weather/tmy_bias/station_*.csv        — TMY + 최근3년 ASOS 편차 보정
  data/weather/tmy_forecast/station_*.csv    — TMY + ASOS min/max 스케일링 (예보 시뮬레이션)

사용법:
  python -m scripts.build_tmy_weather
"""
from __future__ import annotations

import io as _stdio
import os
import sys
from pathlib import Path

sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.weather.asos import compute_degree_hours

TMY_PATH = Path("dsz_bundle/external_data/epw_hourly.csv")
ASOS_PATH = Path("ASOS/asos_all.parquet")
OUT_BASE = Path("data/weather")

# ASOS station → TMY station 매핑
ASOS_TO_TMY = {
    108: 108,  # 서울
    112: 112,  # 인천
    114: 101,  # 원주 → 춘천
    119: 119,  # 수원
    133: 133,  # 대전
    143: 143,  # 대구
    156: 156,  # 광주
    159: 159,  # 부산
    184: 184,  # 제주
}


def load_tmy() -> pd.DataFrame:
    df = pd.read_csv(TMY_PATH)
    df["ts"] = pd.to_datetime(df["ts"])
    df["month"] = df["ts"].dt.month
    df["hour"] = df["ts"].dt.hour
    df["day_of_year"] = df["ts"].dt.dayofyear
    # GHI Wh/m2 → MJ/m2 (1시간 기준)
    df["solar_mj"] = df["ghi_whm2"] / 1e6 * 3.6
    df["humidity"] = df["rh_pct"]
    df["wind_speed"] = df["wind_speed_ms"]
    return df


def monthly_features_from_hourly(
    hourly: pd.DataFrame,
    station_id: int,
    hdd_base: float = 15.0,
    cdd_base: float = 24.0,
) -> pd.DataFrame:
    """시간별 데이터 → 월간 집계 (ASOS station CSV와 동일 형태)."""
    d = hourly.copy()
    d["cdh"] = compute_degree_hours(d["temp_c"].values, cdd_base, "cooling")
    d["hdh"] = compute_degree_hours(d["temp_c"].values, hdd_base, "heating")

    d["period"] = "night"
    d.loc[d["hour"].between(9, 17), "period"] = "daytime"
    d.loc[d["hour"].between(18, 22), "period"] = "evening"

    total = d.groupby("month").agg(
        cdh_total=("cdh", "sum"),
        hdh_total=("hdh", "sum"),
        temp_mean=("temp_c", "mean"),
        temp_max=("temp_c", "max"),
        temp_min=("temp_c", "min"),
        humidity_mean=("humidity", "mean"),
        wind_speed_mean=("wind_speed", "mean"),
        solar_mj_mean=("solar_mj", "mean"),
    ).reset_index()

    # 시간대별
    for period in ["daytime", "evening", "night"]:
        sub = d[d["period"] == period].groupby("month").agg(
            cdh=("cdh", "sum"), hdh=("hdh", "sum")
        ).reset_index()
        sub = sub.rename(columns={"cdh": f"cdh_{period}", "hdh": f"hdh_{period}"})
        total = total.merge(sub, on="month", how="left")

    # 강수: TMY에 없음 → 0으로 채움 (ASOS 평년값으로 나중에 대체)
    total["precip_sum"] = 0.0
    total["rainy_hours"] = 0

    total["station_id"] = station_id
    # month → year_month (TMY는 대표년이라 연도 무의미, "TMY-MM" 형태)
    total["year_month"] = total["month"].apply(lambda m: f"TMY-{m:02d}")

    return total


def apply_bias_correction(
    tmy_monthly: pd.DataFrame,
    asos_monthly: pd.DataFrame,
    station_id: int,
    recent_years: int = 3,
) -> pd.DataFrame:
    """TMY 월간 피처에 최근 N년 ASOS 월평균과의 편차를 보정."""
    out = tmy_monthly.copy()

    # ASOS에서 해당 관측소의 최근 N년 월평균
    asos_st = asos_monthly[asos_monthly["station_id"] == station_id].copy()
    if len(asos_st) == 0:
        return out

    asos_st["_ym"] = pd.to_datetime(asos_st["year_month"].astype(str))
    max_date = asos_st["_ym"].max()
    cutoff = max_date - pd.DateOffset(years=recent_years)
    recent = asos_st[asos_st["_ym"] >= cutoff]
    if len(recent) == 0:
        return out

    recent["month"] = recent["_ym"].dt.month
    asos_avg = recent.groupby("month").mean(numeric_only=True)

    # TMY 월별 보정
    numeric_cols = ["cdh_total", "hdh_total", "temp_mean", "temp_max", "temp_min",
                    "cdh_daytime", "hdh_daytime", "cdh_evening", "hdh_evening",
                    "cdh_night", "hdh_night", "humidity_mean", "wind_speed_mean",
                    "solar_mj_mean"]

    for col in numeric_cols:
        if col in asos_avg.columns and col in out.columns:
            for _, row in out.iterrows():
                m = row["month"]
                if m in asos_avg.index:
                    tmy_val = row[col]
                    asos_val = asos_avg.loc[m, col]
                    if pd.notna(tmy_val) and pd.notna(asos_val) and tmy_val != 0:
                        bias = asos_val - tmy_val
                        out.loc[out["month"] == m, col] = tmy_val + bias

    # 강수는 ASOS 평년값으로 대체
    for m in range(1, 13):
        if m in asos_avg.index:
            if "precip_sum" in asos_avg.columns:
                out.loc[out["month"] == m, "precip_sum"] = asos_avg.loc[m, "precip_sum"]
            if "rainy_hours" in asos_avg.columns:
                out.loc[out["month"] == m, "rainy_hours"] = asos_avg.loc[m, "rainy_hours"]

    return out


def apply_forecast_correction(
    tmy_hourly: pd.DataFrame,
    asos_hourly: pd.DataFrame,
    station_id: int,
    hdd_base: float = 15.0,
    cdd_base: float = 24.0,
) -> pd.DataFrame:
    """TMY 시간별 → ASOS 일별 min/max로 스케일링 후 월간 집계.

    중기예보 시뮬레이션: ASOS 일별 최고/최저를 "예보"로 간주하고
    TMY 시간별 패턴을 해당 범위에 맞춤.
    """
    tmy = tmy_hourly.copy()
    asos = asos_hourly[asos_hourly["station_id"] == station_id].copy()
    if len(asos) == 0:
        return monthly_features_from_hourly(tmy, station_id, hdd_base, cdd_base)

    asos["date"] = asos["ts"].dt.normalize()
    asos["month"] = asos["ts"].dt.month
    asos["day_of_year"] = asos["ts"].dt.dayofyear

    # ASOS 일별 min/max (각 날의 "예보" 역할)
    asos_daily = asos.groupby(["month", "day_of_year"]).agg(
        asos_min=("temp_c", "min"),
        asos_max=("temp_c", "max"),
    ).reset_index()

    # 다년 평균: 같은 day_of_year의 평균 min/max
    asos_doy = asos_daily.groupby("day_of_year").agg(
        asos_min=("asos_min", "mean"),
        asos_max=("asos_max", "mean"),
    ).reset_index()

    # TMY 일별 min/max
    tmy_daily = tmy.groupby("day_of_year").agg(
        tmy_min=("temp_c", "min"),
        tmy_max=("temp_c", "max"),
    ).reset_index()

    merged = tmy_daily.merge(asos_doy, on="day_of_year", how="left")

    # 스케일링: TMY 시간별 → ASOS 범위로
    corrected = tmy.copy()
    for _, row in merged.iterrows():
        doy = row["day_of_year"]
        tmy_min, tmy_max = row["tmy_min"], row["tmy_max"]
        asos_min, asos_max = row["asos_min"], row["asos_max"]

        if pd.isna(asos_min) or pd.isna(asos_max):
            continue
        if tmy_max == tmy_min:
            corrected.loc[corrected["day_of_year"] == doy, "temp_c"] = (asos_min + asos_max) / 2
        else:
            mask = corrected["day_of_year"] == doy
            vals = corrected.loc[mask, "temp_c"].values
            scaled = (vals - tmy_min) / (tmy_max - tmy_min) * (asos_max - asos_min) + asos_min
            corrected.loc[mask, "temp_c"] = scaled

    return monthly_features_from_hourly(corrected, station_id, hdd_base, cdd_base)


def main():
    print("[TMY 기상 피처 생성]")

    # 로드
    tmy_all = load_tmy()
    print(f"  TMY: {len(tmy_all):,}행, {tmy_all['station_id'].nunique()}개 관측소")

    asos_hourly = None
    asos_monthly = None
    if ASOS_PATH.exists():
        asos_hourly = pd.read_parquet(ASOS_PATH)
        asos_hourly["ts"] = pd.to_datetime(asos_hourly["ts"])
        print(f"  ASOS: {len(asos_hourly):,}행")

        # ASOS 월간 집계 로드 (이미 생성된 station CSV)
        asos_monthly_frames = []
        for asos_sid in ASOS_TO_TMY.keys():
            p = OUT_BASE / f"station_{asos_sid}.csv"
            if p.exists():
                asos_monthly_frames.append(pd.read_csv(p))
        if asos_monthly_frames:
            asos_monthly = pd.concat(asos_monthly_frames, ignore_index=True)
            print(f"  ASOS 월간: {len(asos_monthly)}행")

    # 1. TMY Raw
    out_raw = OUT_BASE / "tmy_raw"
    out_raw.mkdir(parents=True, exist_ok=True)
    for asos_sid, tmy_sid in ASOS_TO_TMY.items():
        tmy_st = tmy_all[tmy_all["station_id"] == tmy_sid]
        monthly = monthly_features_from_hourly(tmy_st, asos_sid)
        monthly.to_csv(out_raw / f"station_{asos_sid}.csv", index=False, encoding="utf-8-sig")
    print(f"  [1/3] tmy_raw: {out_raw}/")

    # 2. TMY Bias
    out_bias = OUT_BASE / "tmy_bias"
    out_bias.mkdir(parents=True, exist_ok=True)
    if asos_monthly is not None:
        for asos_sid, tmy_sid in ASOS_TO_TMY.items():
            tmy_st = tmy_all[tmy_all["station_id"] == tmy_sid]
            monthly = monthly_features_from_hourly(tmy_st, asos_sid)
            corrected = apply_bias_correction(monthly, asos_monthly, asos_sid)
            corrected.to_csv(out_bias / f"station_{asos_sid}.csv", index=False, encoding="utf-8-sig")
        print(f"  [2/3] tmy_bias: {out_bias}/")
    else:
        print(f"  [2/3] tmy_bias: ASOS 없음 → 스킵")

    # 3. TMY Forecast (ASOS min/max 스케일링)
    out_fcst = OUT_BASE / "tmy_forecast"
    out_fcst.mkdir(parents=True, exist_ok=True)
    if asos_hourly is not None:
        for asos_sid, tmy_sid in ASOS_TO_TMY.items():
            tmy_st = tmy_all[tmy_all["station_id"] == tmy_sid]
            corrected = apply_forecast_correction(tmy_st, asos_hourly, asos_sid)
            # 강수: ASOS 평년값 대체
            if asos_monthly is not None:
                asos_st = asos_monthly[asos_monthly["station_id"] == asos_sid]
                if len(asos_st) > 0:
                    asos_st["_month"] = pd.to_datetime(asos_st["year_month"].astype(str)).dt.month
                    avg_precip = asos_st.groupby("_month")[["precip_sum", "rainy_hours"]].mean()
                    for m in range(1, 13):
                        if m in avg_precip.index:
                            corrected.loc[corrected["month"] == m, "precip_sum"] = avg_precip.loc[m, "precip_sum"]
                            corrected.loc[corrected["month"] == m, "rainy_hours"] = avg_precip.loc[m, "rainy_hours"]
            corrected.to_csv(out_fcst / f"station_{asos_sid}.csv", index=False, encoding="utf-8-sig")
        print(f"  [3/3] tmy_forecast: {out_fcst}/")
    else:
        print(f"  [3/3] tmy_forecast: ASOS 없음 → 스킵")

    print(f"\n[완료]")
    print(f"  data/weather/tmy_raw/      — TMY 순정")
    print(f"  data/weather/tmy_bias/     — TMY + 최근3년 편차 보정")
    print(f"  data/weather/tmy_forecast/ — TMY + ASOS min/max 스케일링")


if __name__ == "__main__":
    main()
