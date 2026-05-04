"""표준 LP 데이터 스키마.

안심구역 내 실 LP, 공공 공동주택 융합데이터, 로컬 합성 데이터 — 세 소스가
모두 이 스키마로 정규화되어 파이프라인 하류에 전달된다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

CUSTOMER_ID = "customer_id"
TS = "ts"
CONTRACT_TYPE = "contract_type"
P_ACTIVE_KWH = "p_active_kwh"
P_REACTIVE_KWH = "p_reactive_kwh"
P_APPARENT_KWH = "p_apparent_kwh"
MAX_DEMAND_KW = "max_demand_kw"

REQUIRED_COLUMNS: tuple[str, ...] = (
    CUSTOMER_ID,
    TS,
    CONTRACT_TYPE,
    P_ACTIVE_KWH,
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    P_REACTIVE_KWH,
    P_APPARENT_KWH,
    MAX_DEMAND_KW,
)

CONTRACT_TYPES: tuple[str, ...] = ("주택용", "일반용")

INTERVAL_MINUTES = 15


@dataclass(frozen=True)
class SchemaReport:
    n_rows: int
    n_customers: int
    ts_min: pd.Timestamp
    ts_max: pd.Timestamp
    missing_required: tuple[str, ...]
    present_optional: tuple[str, ...]
    contract_type_counts: dict[str, int]


def validate(df: pd.DataFrame, *, strict: bool = True) -> SchemaReport:
    """표준 스키마 검증.

    strict=True 면 필수 컬럼 결여·잘못된 타입에서 즉시 ValueError.
    """
    missing = tuple(c for c in REQUIRED_COLUMNS if c not in df.columns)
    if strict and missing:
        raise ValueError(f"missing required columns: {missing}")

    if TS in df.columns and not pd.api.types.is_datetime64_any_dtype(df[TS]):
        raise ValueError(f"{TS} must be datetime64 dtype, got {df[TS].dtype}")

    if CONTRACT_TYPE in df.columns:
        bad = set(df[CONTRACT_TYPE].dropna().unique()) - set(CONTRACT_TYPES)
        if strict and bad:
            raise ValueError(f"unexpected contract_type values: {bad}")

    if P_ACTIVE_KWH in df.columns and (df[P_ACTIVE_KWH] < 0).any():
        if strict:
            raise ValueError(f"{P_ACTIVE_KWH} must be non-negative (BTM 역송 의심 — btm_detect 먼저 실행)")
        import warnings
        n_neg = int((df[P_ACTIVE_KWH] < 0).sum())
        warnings.warn(f"{P_ACTIVE_KWH}: {n_neg}건 음수값 발견 — BTM(역송) 가능성")

    present_opt = tuple(c for c in OPTIONAL_COLUMNS if c in df.columns)

    return SchemaReport(
        n_rows=len(df),
        n_customers=df[CUSTOMER_ID].nunique() if CUSTOMER_ID in df.columns else 0,
        ts_min=df[TS].min() if TS in df.columns else pd.NaT,
        ts_max=df[TS].max() if TS in df.columns else pd.NaT,
        missing_required=missing,
        present_optional=present_opt,
        contract_type_counts=(
            df[CONTRACT_TYPE].value_counts().to_dict()
            if CONTRACT_TYPE in df.columns
            else {}
        ),
    )


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """선택 컬럼 중 없는 것은 NaN 으로 채워 존재하게 만든다."""
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    return out
