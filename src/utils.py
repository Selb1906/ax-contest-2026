"""15분 시간인덱스·달력·기상(HDD/CDD) 유틸."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .schemas import INTERVAL_MINUTES


def make_ts_index(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    interval_minutes: int = INTERVAL_MINUTES,
    tz: str | None = None,
) -> pd.DatetimeIndex:
    """[start, end] 범위의 15분(기본) 타임스탬프 인덱스.

    end 는 포함. 마지막 값이 interval 경계에 맞지 않으면 그대로 절삭.
    """
    freq = f"{interval_minutes}min"
    idx = pd.date_range(start=start, end=end, freq=freq, tz=tz, inclusive="both")
    return idx


def hdd_cdd(
    temperature: pd.Series | np.ndarray,
    *,
    hdd_base: float = 15.0,
    cdd_base: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Heating/Cooling degree 값 (시간 단위 인스턴스).

    반환은 (hdd, cdd) — 음수는 0으로 클리핑.
    """
    t = np.asarray(temperature, dtype=float)
    hdd = np.clip(hdd_base - t, 0.0, None)
    cdd = np.clip(t - cdd_base, 0.0, None)
    return hdd, cdd


def load_holidays(path: str | Path | None) -> set[pd.Timestamp]:
    """공휴일 CSV → date 집합.

    CSV 형식: 첫 컬럼이 YYYY-MM-DD (헤더 포함). 경로 None 이면 빈 집합.
    """
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    df = pd.read_csv(p)
    col = df.columns[0]
    dates = pd.to_datetime(df[col], errors="coerce").dropna()
    return set(dates.dt.normalize().tolist())


def is_holiday(
    ts: pd.DatetimeIndex | pd.Series, holidays: set[pd.Timestamp]
) -> np.ndarray:
    if not holidays:
        return np.zeros(len(ts), dtype=bool)
    normalized = pd.DatetimeIndex(ts).normalize()
    return normalized.isin(holidays)


def synthetic_temperature(
    ts: pd.DatetimeIndex,
    *,
    mean_annual: float = 12.5,
    amplitude_annual: float = 15.0,
    daily_amplitude: float = 5.0,
    noise_sigma: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """한반도 연평균 12.5°C, 여름 피크 ~27.5°C, 겨울 저점 ~-2.5°C 근사.

    일교차는 오후 14시 피크, 새벽 4시 저점. 15분 단위 노이즈 추가.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    day_of_year = ts.dayofyear.to_numpy()
    # 한반도: 7월 중순(약 196일차) 최고, 1월 중순 최저. cos 피크가 196일차에 오게 위상 맞춤.
    seasonal = mean_annual + (amplitude_annual / 2.0) * np.cos(
        2 * np.pi * (day_of_year - 196) / 365.25
    )

    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    diurnal = (daily_amplitude / 2.0) * np.cos(2 * np.pi * (hour - 14) / 24.0)

    noise = rng.normal(0.0, noise_sigma, size=len(ts))
    return seasonal + diurnal + noise


def daily_shape(
    hours: np.ndarray | pd.Series,
    peaks: Iterable[dict],
    trough: dict | None,
) -> np.ndarray:
    """일중 가중치 곡선.

    peaks: [{hour, multiplier}, ...] 에서 가우시안 커널로 피크를 조립.
    trough: {hour, multiplier} — multiplier < 1 인 저점 가우시안.
    """
    h = np.asarray(hours, dtype=float)
    base = np.ones_like(h)
    sigma = 2.5

    def gauss(x, center):
        d = np.minimum(np.abs(x - center), 24 - np.abs(x - center))  # wrap
        return np.exp(-0.5 * (d / sigma) ** 2)

    for p in peaks:
        center = float(p["hour"])
        mult = float(p["multiplier"])
        base += (mult - 1.0) * gauss(h, center)

    if trough is not None:
        center = float(trough["hour"])
        mult = float(trough["multiplier"])
        base += (mult - 1.0) * gauss(h, center)

    return np.clip(base, 0.05, None)


def align_to_interval(df: pd.DataFrame, ts_col: str, interval_minutes: int) -> pd.DataFrame:
    """ts 컬럼을 interval 경계로 floor."""
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col]).dt.floor(f"{interval_minutes}min")
    return out
