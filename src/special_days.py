"""특수일 피처 엔지니어링.

전력거래소 방식을 참고: 일반일과 특수일을 분리하여 월 구성 비율을 피처화.

일 유형 분류:
  - workday: 평일 (비공휴일, 비토요일)
  - saturday: 토요일 (비공휴일) — TOU에서 최대부하→중간부하 전환
  - sunday: 일요일 (비공휴일)
  - holiday: 공휴일 — TOU에서 전체 경부하 적용
  - bridge: 징검다리 (공휴일과 주말 사이 낀 평일)
  - monday: 월요일 (비공휴일) — 설비 가동 시작, 최저부하 오전 이동

월 단위 예측에서 활용:
  - 해당 월의 일 유형별 일수를 피처로 → 같은 "7월"이라도
    공휴일 3일인 해 vs 1일인 해의 사용량이 다름
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_holidays(path: str | Path | None = None) -> pd.DataFrame:
    """공휴일 로드. parquet 캐시 우선, 없으면 CSV fallback + 자동 캐시 저장.

    category 컬럼 포함.
    """
    if path is None:
        # parquet 후보를 먼저 탐색
        parquet_candidates = [
            Path("dsz_bundle/external_data/holidays_kr.parquet"),
            Path("external_data/holidays_kr.parquet"),
            Path("data/holidays_kr.parquet"),
        ]
        for p in parquet_candidates:
            if p.exists():
                print(f"  [holidays] parquet 캐시 로드: {p}", flush=True)
                return pd.read_parquet(p)

        # parquet 없으면 CSV 후보 탐색
        csv_candidates = [
            Path("dsz_bundle/external_data/holidays_kr.csv"),
            Path("external_data/holidays_kr.csv"),
            Path("data/holidays_kr.csv"),
        ]
        for p in csv_candidates:
            if p.exists():
                path = p
                break

    if path is None:
        return pd.DataFrame(columns=["date", "name", "type", "category"])

    path = Path(path)
    parquet_path = path.with_suffix(".parquet")

    # parquet 캐시가 있으면 우선 사용
    if parquet_path.exists():
        print(f"  [holidays] parquet 캐시 로드: {parquet_path}", flush=True)
        return pd.read_parquet(parquet_path)

    # CSV 읽기
    df = pd.read_csv(path, parse_dates=["date"])

    # parquet 캐시 자동 저장
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"  [holidays] parquet 캐시 저장: {parquet_path}", flush=True)
    except Exception:
        pass  # 쓰기 권한 없는 경우 무시

    return df


def classify_day(date: pd.Timestamp, holiday_dates: set) -> str:
    """단일 날짜의 일 유형 분류."""
    is_holiday = date.normalize() in holiday_dates
    dow = date.dayofweek  # 0=월 ~ 6=일

    if is_holiday:
        return "holiday"
    if dow == 6:
        return "sunday"
    if dow == 5:
        return "saturday"
    if dow == 0:
        return "monday"
    return "workday"


def detect_bridge_days(dates: pd.DatetimeIndex, holiday_dates: set) -> set:
    """징검다리 연휴 탐지: 공휴일과 주말 사이에 낀 평일."""
    bridge = set()
    for d in dates:
        if d in holiday_dates or d.dayofweek >= 5:
            continue
        prev = d - pd.Timedelta(days=1)
        nxt = d + pd.Timedelta(days=1)
        prev_special = prev in holiday_dates or prev.dayofweek >= 5
        next_special = nxt in holiday_dates or nxt.dayofweek >= 5
        if prev_special and next_special:
            bridge.add(d)
    return bridge


def count_holiday_adjacent(dates: pd.DatetimeIndex, holiday_dates: set) -> int:
    """공휴일 인접일 수: 공휴일 전날 또는 다음날인 평일 (bridge 제외).

    연휴 전후로 사용 패턴이 달라지는 효과를 월 단위로 집약.
    전력거래소는 3일까지 보지만, 3년치 월 단위 모델에서는 ±1일이 현실적.
    """
    bridge = detect_bridge_days(dates, holiday_dates)
    adjacent = set()
    for d in dates:
        if d in holiday_dates or d.dayofweek >= 5 or d in bridge:
            continue
        prev = d - pd.Timedelta(days=1)
        nxt = d + pd.Timedelta(days=1)
        if prev in holiday_dates or nxt in holiday_dates:
            adjacent.add(d)
    return len(adjacent)


def count_consecutive_holidays(dates: pd.DatetimeIndex, holiday_dates: set) -> int:
    """3일 이상 연속 휴일(주말+공휴일) 블록의 총 일수.

    설·추석·징검다리 연휴의 규모를 포착.
    """
    all_off = set()
    for d in dates:
        if d in holiday_dates or d.dayofweek >= 5:
            all_off.add(d)

    sorted_dates = sorted(dates)
    total_long = 0
    streak = 0
    for d in sorted_dates:
        if d in all_off:
            streak += 1
        else:
            if streak >= 3:
                total_long += streak
            streak = 0
    if streak >= 3:
        total_long += streak
    return total_long


def monthly_day_composition(
    year_month: pd.Period,
    holidays_df: pd.DataFrame | None = None,
    holidays_path: str | Path | None = None,
) -> dict:
    """특정 연-월의 일 유형별 일수 산출."""
    if holidays_df is None:
        holidays_df = load_holidays(holidays_path)

    start = year_month.start_time
    end = year_month.end_time
    dates = pd.date_range(start, end, freq="D")

    holiday_dates = set(holidays_df["date"].dt.normalize())
    bridge_dates = detect_bridge_days(dates, holiday_dates)

    counts = {
        "n_days": len(dates),
        "n_workday": 0,
        "n_monday": 0,
        "n_saturday": 0,
        "n_sunday": 0,
        "n_holiday": 0,
        "n_bridge": 0,
    }
    # 공휴일 카테고리별
    holiday_cats = holidays_df[
        (holidays_df["date"] >= start) & (holidays_df["date"] <= end)
    ]["category"].value_counts().to_dict() if "category" in holidays_df.columns else {}
    counts["n_election"] = int(holiday_cats.get("election", 0))
    counts["n_substitute"] = int(holiday_cats.get("substitute", 0))
    counts["n_new_holiday"] = int(holiday_cats.get("new_2023", 0))

    for d in dates:
        if d in bridge_dates:
            counts["n_bridge"] += 1
        elif d.normalize() in holiday_dates:
            counts["n_holiday"] += 1
        elif d.dayofweek == 6:
            counts["n_sunday"] += 1
        elif d.dayofweek == 5:
            counts["n_saturday"] += 1
        elif d.dayofweek == 0:
            counts["n_monday"] += 1
        else:
            counts["n_workday"] += 1

    counts["n_holiday_adjacent"] = count_holiday_adjacent(dates, holiday_dates)
    counts["n_long_weekend_days"] = count_consecutive_holidays(dates, holiday_dates)

    counts["workday_ratio"] = (counts["n_workday"] + counts["n_monday"]) / counts["n_days"]
    counts["holiday_ratio"] = (counts["n_holiday"] + counts["n_bridge"]) / counts["n_days"]
    counts["weekend_ratio"] = (counts["n_saturday"] + counts["n_sunday"]) / counts["n_days"]
    counts["disruption_ratio"] = (
        counts["n_holiday"] + counts["n_bridge"] + counts["n_holiday_adjacent"]
    ) / counts["n_days"]

    return counts


def build_special_day_features(
    year_months: list[pd.Period] | pd.PeriodIndex,
    holidays_path: str | Path | None = None,
) -> pd.DataFrame:
    """여러 연-월에 대해 일 유형 피처 일괄 산출."""
    holidays_df = load_holidays(holidays_path)
    rows = []
    for ym in year_months:
        comp = monthly_day_composition(ym, holidays_df)
        comp["year_month"] = ym
        rows.append(comp)
    return pd.DataFrame(rows)
