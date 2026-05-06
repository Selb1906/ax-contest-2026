"""전국 대표기상 CSV 생성 — 여러 가중치 버전.

안심구역에서 실 LP와 매칭하여 어떤 가중치가 최적인지 비교 평가용.
각 버전은 동일 포맷의 CSV로 출력 → 모델에 교체 투입만 하면 됨.

사용법:
  python -m scripts.build_representative_weather

출력:
  data/weather/
    national_v1_paper.csv     — 논문 표6 개선 인구수 가중치
    national_v2_power.csv     — 논문 표5 전력판매량 가중치
    national_v3_population.csv — 논문 표3 인구수 가중치
    national_v4_equal.csv     — 8개 도시 균등 가중치
    national_v5_seoul.csv     — 서울 단일 (수도권 집중 시나리오)
    station_*.csv             — 개별 관측소별 (지역 매칭용)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.weather.asos import (
    read_asos_directory,
    monthly_weather_features,
    weighted_national_average,
)

ROOT = Path(__file__).resolve().parents[1]
ASOS_DIR = ROOT / "ASOS"
OUT_DIR = ROOT / "data" / "weather"

# 논문 표6: 개선된 인구수 기준 (최종 채택)
W_V1_PAPER = {
    108: 0.4376, 112: 0.0558, 119: 0.0448, 114: 0.0306,
    133: 0.0633, 143: 0.0838, 156: 0.0647, 159: 0.2194,
}

# 논문 표5: 전력판매량 기준
W_V2_POWER = {
    108: 0.1613, 112: 0.0696, 119: 0.1441, 114: 0.0383,
    133: 0.1547, 143: 0.1274, 156: 0.1223, 159: 0.1823,
}

# 논문 표3: 인구수 기준
W_V3_POP = {
    108: 0.3702, 112: 0.0558, 119: 0.0721, 114: 0.0306,
    133: 0.1033, 143: 0.1038, 156: 0.1047, 159: 0.1594,
}

# 균등 가중치
W_V4_EQUAL = {s: 1.0 / 8 for s in [108, 112, 119, 114, 133, 143, 156, 159]}

# 서울 단일
W_V5_SEOUL = {108: 1.0}

VERSIONS = {
    "national_v1_paper": W_V1_PAPER,
    "national_v2_power": W_V2_POWER,
    "national_v3_population": W_V3_POP,
    "national_v4_equal": W_V4_EQUAL,
    "national_v5_seoul": W_V5_SEOUL,
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] ASOS data...")
    df = read_asos_directory(ASOS_DIR)
    sids = sorted(df["station_id"].unique())
    print(f"  rows={len(df):,}  stations={sids}")
    print(f"  period: {df['ts'].min()} ~ {df['ts'].max()}")

    print("\n[compute] monthly weather features per station...")
    mwf = monthly_weather_features(df)
    print(f"  shape: {mwf.shape}")

    # 개별 관측소 저장
    for sid in sids:
        sub = mwf[mwf["station_id"] == sid].copy()
        p = OUT_DIR / f"station_{sid}.csv"
        sub.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"  saved {len(sids)} station files")

    # 가중평균 버전별 저장
    print("\n[build] weighted averages...")
    for name, weights in VERSIONS.items():
        avg = weighted_national_average(mwf, weights)
        p = OUT_DIR / f"{name}.csv"
        avg.to_csv(p, index=False, encoding="utf-8-sig")

        # 간단 요약
        jan = avg[avg["year_month"].astype(str).str.endswith("-01")]
        jul = avg[avg["year_month"].astype(str).str.endswith("-07")]
        jan_t = jan["temp_mean"].mean() if len(jan) else 0
        jul_t = jul["temp_mean"].mean() if len(jul) else 0
        jan_hdh = jan["hdh_total"].mean() if len(jan) else 0
        jul_cdh = jul["cdh_total"].mean() if len(jul) else 0
        print(f"  {name:30s} Jan={jan_t:+5.1f}°C HDH={jan_hdh:>6.0f}  "
              f"Jul={jul_t:5.1f}°C CDH={jul_cdh:>6.0f}")

    print(f"\n[done] {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
