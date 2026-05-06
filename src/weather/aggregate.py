"""기상 데이터 지역 가중평균 — 전력거래소 방식 참고.

고객 지역 정보가 없거나 불완전할 때, 8대 권역의 전력판매량 비중으로
기상 데이터를 가중평균하여 대표 기상 시계열 생성.

가중치 기준:
  - 1차: 전력판매량 비중 (한전 통계 기반)
  - 2차: 인구 비중 (fallback)

관측소 → 권역 매핑:
  수도권: 서울(108), 인천(112), 수원(119)
  강원: 춘천(101), 강릉(105)
  충북: 청주(131)
  충남: 대전(133)
  전북: 전주(146)
  전남: 광주(156), 목포(165)
  경북: 대구(143)
  경남: 부산(159)
  제주: 제주(184)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# 관측소 → 권역 매핑
# ── 참고문헌 기반 8대 도시 가중치 ──
# 출처: 임종훈 외, "단기 전력수요예측 정확도 개선을 위한 대표기온 산정방안",
#       조명·전기설비학회논문지 27(6), 2013, pp.39-43
# 표6: 개선된 인구수 기준 가중치 (최종 채택안)

EIGHT_CITY_WEIGHTS = {
    108: 0.4376,   # 서울 — 서울+경기북부
    112: 0.0558,   # 인천
    119: 0.0448,   # 수원 — 경기남부
    # 114: 0.0306, # 원주 — 강원 (EPW 미보유, 춘천 101로 대체)
    101: 0.0306,   # 춘천 → 강원 대체
    133: 0.0633,   # 대전 — 대전+충남+충북
    143: 0.0838,   # 대구 — 대구+경북
    156: 0.0647,   # 광주 — 광주+전남+전북
    159: 0.2194,   # 부산 — 부산+경남+울산
}

STATION_TO_REGION = {
    108: "서울", 112: "인천", 119: "수원",
    101: "강원", 105: "강원",
    131: "대전", 133: "대전",
    146: "광주", 156: "광주", 165: "광주",
    143: "대구",
    159: "부산",
    184: "제주",
}

# 권역별 가중치 (참고문헌 표6 기반, 제주는 참고문헌에 없으므로 잔여분 배분)
REGION_WEIGHTS = {
    "서울": 0.4376,
    "인천": 0.0558,
    "수원": 0.0448,
    "강원": 0.0306,
    "대전": 0.0633,
    "대구": 0.0838,
    "광주": 0.0647,
    "부산": 0.2194,
    "제주": 0.0000,  # 참고문헌: 제주 미포함 (제주 제외 전국 수요예측)
}


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def weighted_average_weather(
    ts_df: pd.DataFrame,
    weight_type: str = "power",
    target_cols: list[str] | None = None,
) -> pd.DataFrame:
    """관측소별 시계열 → 가중평균 단일 시계열.

    Parameters
    ----------
    ts_df : epw_hourly.csv 형태 (station_id, timestamp, temp_c, ...)
    weight_type : "power" (전력판매량) 또는 "population" (인구)
    target_cols : 평균할 컬럼 (기본: temp_c, humidity, wind_speed)

    Returns
    -------
    timestamp 인덱스의 가중평균 시계열 DataFrame
    """
    weights = REGION_POWER_WEIGHT if weight_type == "power" else REGION_POP_WEIGHT
    weights = _normalize_weights(weights)

    if target_cols is None:
        target_cols = [c for c in ["temp_c", "rh_pct", "wind_speed_ms", "ghi_whm2"]
                       if c in ts_df.columns]

    if "station_id" not in ts_df.columns:
        return ts_df

    # 관측소 → 권역 → 가중치
    ts = ts_df.copy()
    ts["region"] = ts["station_id"].map(STATION_TO_REGION)
    ts = ts.dropna(subset=["region"])

    # 권역 내 여러 관측소는 먼저 단순 평균
    region_avg = (
        ts.groupby(["region", "ts"], observed=True)[target_cols]
        .mean()
        .reset_index()
    )

    # 권역별 가중 합산
    result_rows = []
    for timestamp, grp in region_avg.groupby("ts"):
        row = {"ts": timestamp}
        for col in target_cols:
            weighted_sum = 0.0
            weight_sum = 0.0
            for _, r in grp.iterrows():
                w = weights.get(r["region"], 0)
                if pd.notna(r[col]):
                    weighted_sum += r[col] * w
                    weight_sum += w
            row[col] = weighted_sum / weight_sum if weight_sum > 0 else np.nan
        result_rows.append(row)

    result = pd.DataFrame(result_rows)
    if not result.empty:
        result["ts"] = pd.to_datetime(result["ts"])
        result = result.sort_values("ts").reset_index(drop=True)
    return result


def match_weather_to_customer(
    ts_df: pd.DataFrame,
    customer_region: str | None = None,
    customer_station: int | None = None,
    weight_type: str = "power",
) -> pd.DataFrame:
    """고객 지역 정보에 따라 기상 데이터 선택.

    - 관측소 ID 있음 → 해당 관측소 데이터
    - 권역만 있음 → 해당 권역 관측소 평균
    - 지역 정보 없음 → 전국 가중평균
    """
    if customer_station is not None:
        subset = ts_df[ts_df["station_id"] == customer_station]
        if len(subset) > 0:
            return subset

    if customer_region is not None:
        region_stations = [s for s, r in STATION_TO_REGION.items() if r == customer_region]
        subset = ts_df[ts_df["station_id"].isin(region_stations)]
        if len(subset) > 0:
            target_cols = [c for c in ["temp_c", "rh_pct", "wind_speed_ms", "ghi_whm2"]
                           if c in subset.columns]
            return subset.groupby("ts")[target_cols].mean().reset_index()

    return weighted_average_weather(ts_df, weight_type)
