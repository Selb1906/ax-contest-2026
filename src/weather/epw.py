"""EPW (EnergyPlus Weather) AMY 파일 파서.

EPW 포맷:
- 첫 8줄: 헤더 (LOCATION, DESIGN CONDITIONS, ..., DATA PERIODS)
- 그 이후 8760 행: 시간 단위 기상 데이터

필드 순서 (0-indexed, 데이터행 기준):
 0 Year, 1 Month, 2 Day, 3 Hour(1-24), 4 Minute,
 5 Data Source Flags (문자열, 무시),
 6 Dry Bulb Temperature (°C),
 7 Dew Point Temperature (°C),
 8 Relative Humidity (%),
 9 Atmospheric Pressure (Pa),
10 Extraterrestrial Horizontal Radiation (Wh/m²),
11 Extraterrestrial Direct Normal Radiation (Wh/m²),
12 Horizontal Infrared Radiation from Sky (Wh/m²),
13 Global Horizontal Radiation (Wh/m²),
14 Direct Normal Radiation (Wh/m²),
15 Diffuse Horizontal Radiation (Wh/m²),
16 Global Horizontal Illuminance, 17 Direct Normal Illum, 18 Diffuse Illum, 19 Zenith Luminance,
20 Wind Direction (°), 21 Wind Speed (m/s),
22 Total Sky Cover (0-10), 23 Opaque Sky Cover (0-10),
24 Visibility (km), 25 Ceiling Height (m),
26 Present Weather Obs, 27 Present Weather Codes,
28 Precipitable Water (mm), 29 Aerosol Optical Depth,
30 Snow Depth (cm), 31 Days Since Last Snowfall,
32 Albedo, 33 Liquid Precipitation Depth (mm), 34 Liquid Precipitation Quantity (hr)

결측 sentinel: 9999 (일부 필드는 999 — 필드별로 다름). 아래 파서는 보편적 sentinel
9999 / 99999 / 999 을 NaN 으로 변환한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# 레이아웃별 필드 위치. 한국 AMY EPW는 조명·천장고 필드가 생략된 29 컬럼 축약형이
# 많아 풍향·풍속·운량 인덱스가 좌측으로 시프트됨. 첫 데이터 행의 컬럼 수로 자동 판별.
# 형식: (col_idx, name, missing_sentinel)  sentinel 이상이면 NaN 처리.
_FIELDS_29: tuple[tuple[int, str, float], ...] = (
    (6, "temp_c", 99.9),
    (7, "dewpoint_c", 99.9),
    (8, "rh_pct", 999.0),
    (9, "pressure_pa", 999999.0),
    (13, "ghi_whm2", 9999.0),
    (14, "dni_whm2", 9999.0),
    (15, "dhi_whm2", 9999.0),
    (18, "wind_dir_deg", 999.0),
    (19, "wind_speed_ms", 999.0),
    (20, "total_sky_cover", 99.0),
    (21, "opaque_sky_cover", 99.0),
)

_FIELDS_35: tuple[tuple[int, str, float], ...] = (
    (6, "temp_c", 99.9),
    (7, "dewpoint_c", 99.9),
    (8, "rh_pct", 999.0),
    (9, "pressure_pa", 999999.0),
    (13, "ghi_whm2", 9999.0),
    (14, "dni_whm2", 9999.0),
    (15, "dhi_whm2", 9999.0),
    (20, "wind_dir_deg", 999.0),
    (21, "wind_speed_ms", 999.0),
    (22, "total_sky_cover", 99.0),
    (23, "opaque_sky_cover", 99.0),
    (33, "precip_mm", 999.0),
)


def _select_fields(n_cols: int) -> tuple[tuple[int, str, float], ...]:
    if n_cols >= 34:
        return _FIELDS_35
    return _FIELDS_29


@dataclass(frozen=True)
class EPWLocation:
    station_id: str
    station_name: str
    country: str
    source: str
    lat: float
    lon: float
    tz_hours: float
    elevation_m: float


def _parse_location_line(line: str) -> EPWLocation:
    # LOCATION,SEOUL,KOR,ASOS,108,37.57,126.97,9.0,86.0
    parts = [p.strip() for p in line.split(",")]
    if parts[0] != "LOCATION":
        raise ValueError(f"expected LOCATION line, got: {line[:80]!r}")
    return EPWLocation(
        station_id=parts[4],
        station_name=parts[1],
        country=parts[2],
        source=parts[3],
        lat=float(parts[5]),
        lon=float(parts[6]),
        tz_hours=float(parts[7]),
        elevation_m=float(parts[8]),
    )


def read_epw(path: str | Path) -> tuple[EPWLocation, pd.DataFrame]:
    """단일 EPW 파일 → (위치정보, 시간단위 DataFrame).

    반환 DataFrame 컬럼:
      ts (tz-naive KST, hour-beginning), station_id, temp_c, dewpoint_c, rh_pct,
      pressure_pa, ghi_whm2, dni_whm2, dhi_whm2, wind_dir_deg, wind_speed_ms,
      total_sky_cover, opaque_sky_cover, precip_mm
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        header_lines = [f.readline() for _ in range(8)]
        data_lines = f.readlines()
    loc = _parse_location_line(header_lines[0])

    # 첫 유효 행으로 레이아웃 판별
    first = next((ln for ln in data_lines if ln.strip()), "")
    fields_layout = _select_fields(len(first.split(",")))

    rows: list[list[float | int]] = []
    for ln in data_lines:
        if not ln.strip():
            continue
        fields = ln.split(",")
        if len(fields) < 22:
            continue
        y, m, d, h = int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])
        vals: list[float | int] = [y, m, d, h]
        for idx, _name, _miss in fields_layout:
            try:
                vals.append(float(fields[idx]))
            except (ValueError, IndexError):
                vals.append(np.nan)
        rows.append(vals)

    cols = ["_year", "_month", "_day", "_hour"] + [name for _, name, _ in fields_layout]
    df = pd.DataFrame(rows, columns=cols)

    # 결측 sentinel 처리
    for _idx, name, sentinel in fields_layout:
        df.loc[df[name] >= sentinel, name] = np.nan

    # 타임스탬프 (hour-beginning: EPW Hour=1 → 00:00)
    df["ts"] = pd.to_datetime(
        dict(year=df["_year"], month=df["_month"], day=df["_day"], hour=df["_hour"] - 1),
        errors="coerce",
    )
    df = df.drop(columns=["_year", "_month", "_day", "_hour"])
    df["station_id"] = loc.station_id
    df["station_name"] = loc.station_name
    df["lat"] = loc.lat
    df["lon"] = loc.lon

    cols_ordered = ["station_id", "station_name", "lat", "lon", "ts"] + [
        n for _, n, _ in fields_layout
    ]
    df = df[cols_ordered].sort_values("ts").reset_index(drop=True)
    return loc, df


def read_epw_directory(
    directory: str | Path,
    *,
    pattern: str = "*.epw",
    stations: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """디렉터리 내 EPW 일괄 파싱 → (관측소 메타, 합쳐진 시계열).

    stations 로 관측소 ID 필터 (예: {"108","159"}).
    """
    d = Path(directory)
    files = sorted(d.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no EPW files in {d} matching {pattern}")

    metas: list[dict] = []
    series: list[pd.DataFrame] = []
    for fp in files:
        loc, df = read_epw(fp)
        if stations is not None and loc.station_id not in set(stations):
            continue
        metas.append(
            {
                "station_id": loc.station_id,
                "station_name": loc.station_name,
                "country": loc.country,
                "source": loc.source,
                "lat": loc.lat,
                "lon": loc.lon,
                "tz_hours": loc.tz_hours,
                "elevation_m": loc.elevation_m,
                "file": str(fp.name),
            }
        )
        series.append(df)

    meta_df = pd.DataFrame(metas).sort_values("station_id").reset_index(drop=True)
    ts_df = pd.concat(series, ignore_index=True).sort_values(
        ["station_id", "ts"]
    ).reset_index(drop=True)
    return meta_df, ts_df


def upsample_to_15min(
    weather: pd.DataFrame,
    *,
    station_id: str | None = None,
    method: str = "ffill",
) -> pd.DataFrame:
    """시간단위 기상 → 15분 단위로 broadcast/보간.

    method:
      - "ffill": 같은 시간 내 15분 4개 모두 동일값 (기본)
      - "linear": 인접 시간 사이 선형 보간 (기온 등에 적합)
    """
    if station_id is not None:
        w = weather[weather["station_id"] == station_id].copy()
    else:
        # 복수 관측소면 장기 실행 — 필터 권장
        w = weather.copy()

    idx = pd.date_range(w["ts"].min(), w["ts"].max(), freq="15min")
    out_frames: list[pd.DataFrame] = []
    numeric = [c for c in w.columns if c not in ("station_id", "station_name", "lat", "lon", "ts")]
    for sid, grp in w.groupby("station_id"):
        g = grp.set_index("ts").sort_index()
        reindexed = g[numeric].reindex(idx)
        if method == "ffill":
            reindexed = reindexed.ffill()
        elif method == "linear":
            reindexed = reindexed.interpolate(method="time")
        else:
            raise ValueError(f"unknown method: {method}")
        reindexed["station_id"] = sid
        reindexed["station_name"] = grp["station_name"].iloc[0]
        reindexed["lat"] = grp["lat"].iloc[0]
        reindexed["lon"] = grp["lon"].iloc[0]
        reindexed = reindexed.rename_axis("ts").reset_index()
        out_frames.append(reindexed)
    return pd.concat(out_frames, ignore_index=True)
