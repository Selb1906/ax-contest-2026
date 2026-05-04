"""LP 데이터 전처리 — 결측·이상값 처리.

프로파일러가 탐지한 문제를 실제로 처리하는 모듈.
처리 전후 통계를 기록하여 보고서에 "어떤 전처리를 했는지" 문서화.

원칙:
  - 원본은 절대 수정하지 않음 (copy 후 처리)
  - 모든 처리는 로그 남김 (before/after 통계)
  - 처리 방법은 설정으로 선택 가능
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schemas import CUSTOMER_ID, P_ACTIVE_KWH, P_REACTIVE_KWH, MAX_DEMAND_KW, TS


@dataclass
class PreprocessLog:
    original_rows: int = 0
    final_rows: int = 0
    steps: list[dict] = field(default_factory=list)

    def add(self, name: str, removed: int, detail: str = ""):
        self.steps.append({"step": name, "removed": removed, "detail": detail})

    def summary(self) -> dict:
        return {
            "original_rows": self.original_rows,
            "final_rows": self.final_rows,
            "total_removed": self.original_rows - self.final_rows,
            "removal_rate": (self.original_rows - self.final_rows) / self.original_rows
            if self.original_rows > 0 else 0,
            "steps": self.steps,
        }


def remove_duplicates(df: pd.DataFrame, log: PreprocessLog) -> pd.DataFrame:
    """고객×시간 중복 행 제거 (마지막 값 유지)."""
    before = len(df)
    df = df.drop_duplicates(subset=[CUSTOMER_ID, TS], keep="last")
    removed = before - len(df)
    if removed > 0:
        log.add("remove_duplicates", removed, f"{removed}건 중복 제거")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    log: PreprocessLog,
    method: str = "interpolate",
    max_gap_hours: int = 6,
) -> pd.DataFrame:
    """결측값 처리.

    method:
      "interpolate" — 선형 보간 (gap이 max_gap_hours 이하일 때만)
      "ffill" — 전방 채움
      "drop" — 결측 행 제거
      "zero" — 0으로 채움
    """
    missing_before = df[P_ACTIVE_KWH].isna().sum()
    if missing_before == 0:
        return df

    if method == "interpolate":
        max_gap = max_gap_hours * 4  # 15분 단위
        df = df.sort_values([CUSTOMER_ID, TS])
        df[P_ACTIVE_KWH] = df.groupby(CUSTOMER_ID, observed=True)[P_ACTIVE_KWH].transform(
            lambda x: x.interpolate(method="linear", limit=max_gap)
        )
        if P_REACTIVE_KWH in df.columns:
            df[P_REACTIVE_KWH] = df.groupby(CUSTOMER_ID, observed=True)[P_REACTIVE_KWH].transform(
                lambda x: x.interpolate(method="linear", limit=max_gap)
            )

    elif method == "ffill":
        df = df.sort_values([CUSTOMER_ID, TS])
        df[P_ACTIVE_KWH] = df.groupby(CUSTOMER_ID, observed=True)[P_ACTIVE_KWH].ffill()

    elif method == "drop":
        df = df.dropna(subset=[P_ACTIVE_KWH])

    elif method == "zero":
        df[P_ACTIVE_KWH] = df[P_ACTIVE_KWH].fillna(0)

    missing_after = df[P_ACTIVE_KWH].isna().sum()
    filled = missing_before - missing_after
    log.add("handle_missing", int(filled),
            f"method={method}, before={missing_before}, after={missing_after}")
    return df


def remove_outliers(
    df: pd.DataFrame,
    log: PreprocessLog,
    method: str = "iqr",
    iqr_factor: float = 3.0,
    zscore_threshold: float = 4.0,
) -> pd.DataFrame:
    """이상값 제거.

    method:
      "iqr" — IQR × factor 범위 밖 제거 (고객별)
      "zscore" — Z-score 기반 (고객별)
      "jump" — 전후 대비 10x 이상 점프 제거
      "quantile" — 상하위 0.1% 제거
    """
    before = len(df)

    if method == "iqr":
        def _iqr_filter(group):
            q1 = group[P_ACTIVE_KWH].quantile(0.25)
            q3 = group[P_ACTIVE_KWH].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            return group[(group[P_ACTIVE_KWH] >= lower) & (group[P_ACTIVE_KWH] <= upper)]
        df = df.groupby(CUSTOMER_ID, observed=True, group_keys=False).apply(_iqr_filter)

    elif method == "zscore":
        def _zscore_filter(group):
            mean = group[P_ACTIVE_KWH].mean()
            std = group[P_ACTIVE_KWH].std()
            if std == 0:
                return group
            z = (group[P_ACTIVE_KWH] - mean).abs() / std
            return group[z <= zscore_threshold]
        df = df.groupby(CUSTOMER_ID, observed=True, group_keys=False).apply(_zscore_filter)

    elif method == "jump":
        df = df.sort_values([CUSTOMER_ID, TS])
        prev = df.groupby(CUSTOMER_ID, observed=True)[P_ACTIVE_KWH].shift(1)
        ratio = df[P_ACTIVE_KWH] / prev.clip(lower=1e-9)
        normal = (ratio < 10) & (ratio > 0.1) | prev.isna()
        df = df[normal]

    elif method == "quantile":
        def _quantile_filter(group):
            lower = group[P_ACTIVE_KWH].quantile(0.001)
            upper = group[P_ACTIVE_KWH].quantile(0.999)
            return group[(group[P_ACTIVE_KWH] >= lower) & (group[P_ACTIVE_KWH] <= upper)]
        df = df.groupby(CUSTOMER_ID, observed=True, group_keys=False).apply(_quantile_filter)

    removed = before - len(df)
    log.add("remove_outliers", removed, f"method={method}, removed={removed}")
    return df


def fill_time_gaps(
    df: pd.DataFrame,
    log: PreprocessLog,
    interval_min: int = 15,
) -> pd.DataFrame:
    """시간 갭 채움 — 누락된 타임스탬프를 NaN 행으로 생성."""
    before = len(df)
    frames = []
    for cid, grp in df.groupby(CUSTOMER_ID, observed=True):
        ts_min, ts_max = grp[TS].min(), grp[TS].max()
        full_idx = pd.date_range(ts_min, ts_max, freq=f"{interval_min}min")
        full = pd.DataFrame({TS: full_idx, CUSTOMER_ID: cid})
        merged = full.merge(grp, on=[CUSTOMER_ID, TS], how="left")
        frames.append(merged)

    df = pd.concat(frames, ignore_index=True) if frames else df
    added = len(df) - before
    if added > 0:
        log.add("fill_time_gaps", -added, f"{added}건 빈 타임스탬프 생성 (NaN)")
    return df


def preprocess(
    df: pd.DataFrame,
    *,
    dedup: bool = True,
    fill_gaps: bool = False,
    missing_method: str = "interpolate",
    outlier_method: str = "iqr",
    max_gap_hours: int = 6,
    iqr_factor: float = 3.0,
) -> tuple[pd.DataFrame, PreprocessLog]:
    """전처리 파이프라인.

    Returns
    -------
    (처리된 DataFrame, 처리 로그)
    """
    log = PreprocessLog(original_rows=len(df))
    df = df.copy()

    if dedup:
        df = remove_duplicates(df, log)

    if fill_gaps:
        df = fill_time_gaps(df, log)

    df = handle_missing_values(df, log, method=missing_method, max_gap_hours=max_gap_hours)

    df = remove_outliers(df, log, method=outlier_method, iqr_factor=iqr_factor)

    # 최종 결측 제거 (보간 후 남은 것)
    remaining_na = df[P_ACTIVE_KWH].isna().sum()
    if remaining_na > 0:
        df = df.dropna(subset=[P_ACTIVE_KWH])
        log.add("final_drop_na", int(remaining_na), "보간 후 잔여 결측 제거")

    log.final_rows = len(df)
    return df, log
