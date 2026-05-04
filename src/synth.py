"""합성 LP 생성기 v0.

YAML 설정을 받아 주택용·일반용 고객 집합의 15분 단위 시계열을 만든다.
스카우팅 방문 이후 실 LP 프로파일 통계로 파라미터 재보정 예정 (v1).

디자인 원칙:
- 고객 축으로 벡터화된 numpy 연산 (향후 Dask/Polars 로 병렬화 확장 용이)
- 계산식은 단순하지만 재현 가능 (seed 고정)
- 출력은 customer_id 파티션 parquet → 하류 로더에서 스트리밍 가능
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .schemas import (
    CONTRACT_TYPE,
    CUSTOMER_ID,
    MAX_DEMAND_KW,
    P_ACTIVE_KWH,
    P_APPARENT_KWH,
    P_REACTIVE_KWH,
    TS,
)
from .utils import (
    daily_shape,
    hdd_cdd,
    is_holiday,
    load_holidays,
    make_ts_index,
    synthetic_temperature,
)
from .weather.epw import read_epw_directory


@dataclass
class GeneratorConfig:
    seed: int
    n_customers: int
    residential_ratio: float
    start_date: str
    end_date: str
    interval_min: int
    output_path: Path
    partition_by: list[str]
    residential: dict[str, Any]
    general: dict[str, Any]
    power_factor: dict[str, float]
    max_demand: dict[str, float]
    weather: dict[str, Any]
    calendar: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GeneratorConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        g = raw["generator"]
        return cls(
            seed=int(g["seed"]),
            n_customers=int(g["n_customers"]),
            residential_ratio=float(g["residential_ratio"]),
            start_date=str(g["start_date"]),
            end_date=str(g["end_date"]),
            interval_min=int(g["interval_min"]),
            output_path=Path(g["output"]["path"]),
            partition_by=list(g["output"].get("partition_by", [])),
            residential=raw["residential"],
            general=raw["general"],
            power_factor=raw["power_factor"],
            max_demand=raw["max_demand"],
            weather=raw["weather"],
            calendar=raw["calendar"],
        )


def _sample_base_load(
    params: dict[str, Any], size: int, rng: np.random.Generator
) -> np.ndarray:
    d = params["base_kwh_per_15min"]
    if d["dist"] == "lognormal":
        return rng.lognormal(mean=float(d["mu"]), sigma=float(d["sigma"]), size=size)
    raise ValueError(f"unknown base load dist: {d['dist']}")


def _simulate_segment(
    *,
    ts: pd.DatetimeIndex,
    contract: str,
    params: dict[str, Any],
    customer_ids: np.ndarray,
    base_loads: np.ndarray,
    temperature: np.ndarray,
    holidays: set[pd.Timestamp],
    pf_cfg: dict[str, float],
    md_cfg: dict[str, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """하나의 계약종별 cohort 에 대해 (고객×시간) 매트릭스로 벡터화 생성."""
    n_cust = len(customer_ids)
    n_t = len(ts)

    # 시간 축 피처
    hour = (ts.hour + ts.minute / 60.0).to_numpy()
    dow = ts.dayofweek.to_numpy()
    weekend = dow >= 5
    holi = is_holiday(ts, holidays)

    # 일중 형상
    daily = daily_shape(hour, params.get("daily_peaks", []), params.get("night_trough"))
    # 주말 배수
    weekend_mult = float(params.get("weekend_multiplier", 1.0))
    week_factor = np.where(weekend, weekend_mult, 1.0)
    holiday_mult = float(params.get("holiday_multiplier", 1.0))
    holi_factor = np.where(holi, holiday_mult, 1.0)

    # 기상 반응
    weather = params.get("weather", {})
    hdd, cdd = hdd_cdd(
        temperature,
        hdd_base=float(weather.get("hdd_base", 15.0)),
        cdd_base=float(weather.get("cdd_base", 24.0)),
    )
    cdd_coef = float(weather.get("cdd_coef", 0.0))
    hdd_coef = float(weather.get("hdd_coef", 0.0))
    weather_factor = 1.0 + cdd_coef * cdd + hdd_coef * hdd  # (n_t,)

    # 시간 축 종합 멀티플라이어 (n_t,)
    time_factor = daily * week_factor * holi_factor * weather_factor

    # 매트릭스 (n_cust, n_t) 생성 — 메모리 주의: 필요 시 청크 단위
    active = base_loads[:, None] * time_factor[None, :]

    # 잡음 (multiplicative lognormal)
    noise_sigma = float(params.get("noise_sigma", 0.15))
    noise = rng.lognormal(mean=0.0, sigma=noise_sigma, size=active.shape)
    active = active * noise

    # 역률 → 무효, 피상
    pf_samples = np.clip(
        rng.normal(loc=pf_cfg["mean"], scale=pf_cfg["std"], size=n_cust),
        pf_cfg.get("min", 0.7),
        0.999,
    )
    tan_phi = np.sqrt(1 - pf_samples**2) / pf_samples  # tan(acos(pf))
    reactive = active * tan_phi[:, None]
    apparent = np.sqrt(active**2 + reactive**2)

    # 최대수요 (15min kWh → kW 환산 후 multiplier)
    md_mult = rng.uniform(md_cfg["multiplier_min"], md_cfg["multiplier_max"], size=active.shape)
    max_demand = (4.0 * active) * md_mult

    # long format 으로 풀기
    cust_col = np.repeat(customer_ids, n_t)
    ts_col = np.tile(ts.to_numpy(), n_cust)
    df = pd.DataFrame(
        {
            CUSTOMER_ID: cust_col,
            TS: ts_col,
            CONTRACT_TYPE: contract,
            P_ACTIVE_KWH: active.reshape(-1),
            P_REACTIVE_KWH: reactive.reshape(-1),
            P_APPARENT_KWH: apparent.reshape(-1),
            MAX_DEMAND_KW: max_demand.reshape(-1),
        }
    )
    return df


def generate(cfg: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    ts = make_ts_index(cfg.start_date, cfg.end_date, interval_minutes=cfg.interval_min)

    # 기상: 설정에 따라 합성 or EPW AMY 실측 로드
    w = cfg.weather
    src = w.get("source", "synthetic")
    if src == "synthetic":
        temperature = synthetic_temperature(
            ts,
            mean_annual=float(w.get("mean_annual", 12.5)),
            amplitude_annual=float(w.get("amplitude_annual", 25.0)),
            daily_amplitude=float(w.get("daily_amplitude", 5.0)),
            noise_sigma=float(w.get("noise_sigma", 2.0)),
            rng=rng,
        )
    elif src == "epw":
        epw_dir = w["epw_directory"]
        station_id = str(w["station_id"])
        _meta, ts_df = read_epw_directory(epw_dir, stations={station_id})
        if ts_df.empty:
            raise ValueError(f"EPW station {station_id} not found in {epw_dir}")
        # 시간단위 → 15분 인덱스로 forward-fill 브로드캐스트
        hourly = ts_df.set_index("ts")["temp_c"].sort_index()
        temperature = hourly.reindex(ts, method="ffill").to_numpy()
        # 시작 시점이 EPW 범위 밖이면 NaN → 합성으로 채움 (엣지 케이스)
        if np.isnan(temperature).any():
            fill = synthetic_temperature(
                ts,
                mean_annual=float(w.get("mean_annual", 12.5)),
                amplitude_annual=float(w.get("amplitude_annual", 25.0)),
                daily_amplitude=float(w.get("daily_amplitude", 5.0)),
                noise_sigma=0.0,
                rng=rng,
            )
            temperature = np.where(np.isnan(temperature), fill, temperature)
    else:
        raise NotImplementedError(f"weather source '{src}' not supported")

    holidays = load_holidays(cfg.calendar.get("holidays_path"))

    # 고객 배정
    n_res = int(round(cfg.n_customers * cfg.residential_ratio))
    n_gen = cfg.n_customers - n_res
    res_ids = np.array([f"RES_{i:06d}" for i in range(n_res)])
    gen_ids = np.array([f"GEN_{i:06d}" for i in range(n_gen)])

    res_base = _sample_base_load(cfg.residential, n_res, rng)
    gen_base = _sample_base_load(cfg.general, n_gen, rng)

    parts: list[pd.DataFrame] = []
    if n_res > 0:
        parts.append(
            _simulate_segment(
                ts=ts,
                contract="주택용",
                params=cfg.residential,
                customer_ids=res_ids,
                base_loads=res_base,
                temperature=temperature,
                holidays=holidays,
                pf_cfg=cfg.power_factor,
                md_cfg=cfg.max_demand,
                rng=rng,
            )
        )
    if n_gen > 0:
        parts.append(
            _simulate_segment(
                ts=ts,
                contract="일반용",
                params=cfg.general,
                customer_ids=gen_ids,
                base_loads=gen_base,
                temperature=temperature,
                holidays=holidays,
                pf_cfg=cfg.power_factor,
                md_cfg=cfg.max_demand,
                rng=rng,
            )
        )
    return pd.concat(parts, ignore_index=True)


def write_parquet(df: pd.DataFrame, cfg: GeneratorConfig) -> Path:
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.partition_by:
        df.to_parquet(
            cfg.output_path,
            engine="pyarrow",
            partition_cols=cfg.partition_by,
            index=False,
        )
    else:
        df.to_parquet(cfg.output_path, engine="pyarrow", index=False)
    return cfg.output_path
