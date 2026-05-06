"""본부/지사명 → ASOS 관측소 매칭.

한전 LP 데이터의 "본부", "지사" 컬럼에서 지역을 파악하고
가장 가까운 ASOS 관측소를 매칭.

다양한 표기 대응: "서울본부", "서울", "서울지역본부", "SEOUL" 등
부분 매칭으로 키워드가 포함되면 매칭.
"""
from __future__ import annotations


# 한전 본부/지사 키워드 → ASOS 관측소 ID
# 여러 표기에 대응하기 위해 키워드 기반 부분 매칭
_REGION_TO_STATION = {
    # 수도권
    "서울": 108,
    "남서울": 108,
    "인천": 112,
    "경기": 119,    # 수원
    "수원": 119,

    # 강원
    "강원": 114,    # 원주
    "원주": 114,
    "춘천": 114,
    "강릉": 114,

    # 충청
    "대전": 133,
    "충남": 133,
    "대전충남": 133,
    "충북": 133,    # 대전으로 대체 (청주가 가까우나 8대 도시에 없음)
    "청주": 133,
    "세종": 133,

    # 전라
    "전북": 156,    # 광주
    "전주": 156,
    "광주": 156,
    "전남": 156,
    "광주전남": 156,
    "목포": 156,

    # 경상
    "대구": 143,
    "경북": 143,
    "대구경북": 143,
    "부산": 159,
    "경남": 159,
    "울산": 159,

    # 제주
    "제주": 159,    # 8대 도시에 제주 없음, 부산으로 대체 (가장 가까운 남쪽)
}


def match_station(region_name: str) -> int | None:
    """본부/지사명 → ASOS 관측소 ID.

    부분 매칭: "서울본부" → "서울" 키워드 포함 → 108
    """
    if not region_name or not isinstance(region_name, str):
        return None

    name = region_name.strip().upper().replace(" ", "")

    # 정확히 일치
    for keyword, station in _REGION_TO_STATION.items():
        if keyword.upper() == name:
            return station

    # 부분 매칭 (긴 키워드부터 시도 — "대전충남"이 "대전"보다 먼저)
    sorted_keywords = sorted(_REGION_TO_STATION.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if keyword.upper() in name or name in keyword.upper():
            return _REGION_TO_STATION[keyword]

    return None


def match_stations_bulk(region_series) -> dict:
    """고객별 본부/지사 → 관측소 매칭 (시리즈 전체).

    Returns: {region_name: station_id} 매핑 딕셔너리 + 매칭 통계
    """
    import pandas as pd

    unique_regions = region_series.dropna().unique()
    mapping = {}
    unmatched = []

    for region in unique_regions:
        station = match_station(str(region))
        if station:
            mapping[region] = station
        else:
            unmatched.append(region)

    n_total = len(unique_regions)
    n_matched = len(mapping)
    print(f"  본부/지사 매칭: {n_matched}/{n_total} ({n_matched/n_total*100:.0f}%)")
    if unmatched:
        print(f"  미매칭: {unmatched}")

    return mapping


def attach_station_weather(
    df,
    weather_hourly,
    region_col: str = "region_code",
    customer_id_col: str = "customer_id",
):
    """고객별 지역 기반으로 해당 ASOS 관측소 기상을 LP에 조인.

    가중평균 대신 고객 지역의 관측소 기상을 직접 매칭.
    """
    import pandas as pd

    if region_col not in df.columns:
        print("  [skip] region_code 컬럼 없음 — 가중평균 사용")
        return df

    # 고객별 지역 → 관측소 매핑
    cust_region = df[[customer_id_col, region_col]].drop_duplicates(customer_id_col)
    region_mapping = match_stations_bulk(cust_region[region_col])
    cust_region["matched_station"] = cust_region[region_col].map(region_mapping)

    # 관측소별 기상 (시간별)
    if "station_id" not in weather_hourly.columns:
        print("  [skip] 기상 데이터에 station_id 없음")
        return df

    # 고객에 관측소 ID 부여
    df = df.merge(
        cust_region[[customer_id_col, "matched_station"]],
        on=customer_id_col, how="left"
    )

    # 시간 기준 조인
    df["_ts_h"] = pd.to_datetime(df["ts"]).dt.floor("h")
    weather_hourly = weather_hourly.copy()
    weather_hourly["_ts_h"] = pd.to_datetime(weather_hourly["ts"]).dt.floor("h")

    # 고객의 관측소 기상 매칭
    weather_cols = ["temp_c", "humidity", "wind_speed", "solar_mj"]
    available = [c for c in weather_cols if c in weather_hourly.columns]

    merged = df.merge(
        weather_hourly[["station_id", "_ts_h"] + available],
        left_on=["matched_station", "_ts_h"],
        right_on=["station_id", "_ts_h"],
        how="left",
    )
    merged = merged.drop(columns=["_ts_h", "station_id", "matched_station"], errors="ignore")

    n_matched = merged[available[0]].notna().sum() if available else 0
    print(f"  기상 매칭: {n_matched:,}/{len(merged):,} ({n_matched/len(merged)*100:.1f}%)")

    return merged
