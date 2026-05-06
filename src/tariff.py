"""전기요금 환산 모듈 — kWh → 원 추정.

2026년 4월 16일 시행 한전 전기공급약관 기준.
실제 청구 금액과 차이: 기후환경요금, 연료비조정요금, 부가세, 기반기금 미반영.

출처: E:\AX_Contest\2026년도+4월+16일+시행+전기요금표(종합)_1.pdf
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProgressiveTier:
    upper_kwh: float
    base_won: float
    rate: float


# ════════════════════════════════════════
# 주택용 저압 (2026.04.16)
# 단가는 연중 동일, 계절에 따라 누진 구간 경계만 변경
# 출처: 전기요금표(주택용 저압) 2024.07.01 시행 → 2026.04.16 동일 유지 확인
# ════════════════════════════════════════

RESIDENTIAL_LOW = {
    "normal": [  # 기타계절 (1~6, 9~12) — 200/400 구간
        ProgressiveTier(200,  910,  120.0),
        ProgressiveTier(400,  1600, 214.6),
        ProgressiveTier(float("inf"), 7300, 307.3),
    ],
    "summer": [  # 하계 (7~8) — 300/450 구간 (확대)
        ProgressiveTier(300,  910,  120.0),
        ProgressiveTier(450,  1600, 214.6),
        ProgressiveTier(float("inf"), 7300, 307.3),
    ],
}

# 주택용 고압 (2026.04.16)
# 단가는 연중 동일 (저압과 다른 단가), 구간 경계는 저압과 동일하게 변경
RESIDENTIAL_HIGH = {
    "normal": [  # 기타계절 — 200/400 구간
        ProgressiveTier(200,  730,  105.0),
        ProgressiveTier(400,  1260, 174.0),
        ProgressiveTier(float("inf"), 6060, 242.3),
    ],
    "summer": [  # 하계 — 300/450 구간 (확대)
        ProgressiveTier(300,  730,  105.0),
        ProgressiveTier(450,  1260, 174.0),
        ProgressiveTier(float("inf"), 6060, 242.3),
    ],
}


SUPER_USER_THRESHOLD = 1000  # kWh
SUPER_USER_RATE = 736.2      # 원/kWh (하계 7~8, 동계 12~2)
SUPER_USER_RATE_HIGH = 601.3 # 원/kWh (주택용 고압)


def _progressive_bill(
    monthly_kwh: float,
    tiers: list[ProgressiveTier],
    *,
    super_user: bool = False,
    super_rate: float = SUPER_USER_RATE,
) -> dict:
    remaining = monthly_kwh
    energy = 0.0
    base = 0.0
    tier_idx = 0
    prev_upper = 0.0
    for i, t in enumerate(tiers):
        band = t.upper_kwh - prev_upper
        usage = min(remaining, band)
        if usage > 0:
            energy += usage * t.rate
            base = t.base_won
            tier_idx = i + 1
        remaining -= usage
        prev_upper = t.upper_kwh
        if remaining <= 0:
            break

    super_user_extra = 0
    if super_user and monthly_kwh > SUPER_USER_THRESHOLD:
        excess = monthly_kwh - SUPER_USER_THRESHOLD
        normal_rate_for_excess = tiers[-1].rate
        super_user_extra = excess * (super_rate - normal_rate_for_excess)
        energy += super_user_extra

    total = base + energy
    eff = total / monthly_kwh if monthly_kwh > 0 else 0
    return {"base_won": base, "energy_won": round(energy), "total_won": round(total),
            "tier": tier_idx, "effective_rate": round(eff, 1), "tariff_type": "progressive",
            "super_user_extra": round(super_user_extra)}


# ════════════════════════════════════════
# 일반용(갑) I — 계약전력 1,000kW 미만, 계절별 단일 단가
# ════════════════════════════════════════

GENERAL_GAP_I = {
    "low": {"base_per_kw": 5230, "summer": 123.6, "spring_fall": 86.4, "winter": 110.8},
    "high_a_1": {"base_per_kw": 5550, "summer": 123.3, "spring_fall": 86.5, "winter": 109.3},
    "high_a_2": {"base_per_kw": 6370, "summer": 118.8, "spring_fall": 82.1, "winter": 104.8},
    "high_b_1": {"base_per_kw": 5550, "summer": 122.6, "spring_fall": 86.1, "winter": 108.5},
    "high_b_2": {"base_per_kw": 6370, "summer": 118.1, "spring_fall": 81.6, "winter": 104.0},
}

# ════════════════════════════════════════
# 일반용(갑) II — 계약전력 1,000kW 이상, TOU
# ════════════════════════════════════════

GENERAL_GAP_II_TOU = {
    "high_a_1": {
        "base_per_kw": 6090,
        "summer":      {"off_peak": 76.5, "mid_peak": 121.2, "on_peak": 187.1},
        "spring_fall": {"off_peak": 76.5, "mid_peak": 90.9,  "on_peak": 111.4},
        "winter":      {"off_peak": 80.5, "mid_peak": 119.7, "on_peak": 158.4},
    },
    "high_a_2": {
        "base_per_kw": 6980,
        "summer":      {"off_peak": 72.0, "mid_peak": 116.7, "on_peak": 182.6},
        "spring_fall": {"off_peak": 72.0, "mid_peak": 86.4,  "on_peak": 106.9},
        "winter":      {"off_peak": 76.0, "mid_peak": 115.2, "on_peak": 153.9},
    },
    "high_b_1": {
        "base_per_kw": 6090,
        "summer":      {"off_peak": 75.0, "mid_peak": 118.5, "on_peak": 181.4},
        "spring_fall": {"off_peak": 75.0, "mid_peak": 89.2,  "on_peak": 109.0},
        "winter":      {"off_peak": 78.8, "mid_peak": 116.8, "on_peak": 154.1},
    },
    "high_b_2": {
        "base_per_kw": 6980,
        "summer":      {"off_peak": 70.5, "mid_peak": 114.0, "on_peak": 176.9},
        "spring_fall": {"off_peak": 70.5, "mid_peak": 84.7,  "on_peak": 104.5},
        "winter":      {"off_peak": 74.3, "mid_peak": 112.3, "on_peak": 149.6},
    },
}

# ════════════════════════════════════════
# 일반용(을) — 계약전력 300kW 미만, 계절별 단일 단가
# ════════════════════════════════════════

GENERAL_EUL_SMALL = {
    "low":      {"base_per_kw": 6160, "summer": 132.4, "spring_fall": 91.9, "winter": 119.0},
    "high_a_1": {"base_per_kw": 7170, "summer": 142.6, "spring_fall": 98.6, "winter": 130.3},
    "high_a_2": {"base_per_kw": 8230, "summer": 138.6, "spring_fall": 94.3, "winter": 125.0},
    "high_b_1": {"base_per_kw": 7170, "summer": 140.5, "spring_fall": 97.5, "winter": 127.3},
    "high_b_2": {"base_per_kw": 8230, "summer": 135.2, "spring_fall": 92.2, "winter": 122.0},
}

# ════════════════════════════════════════
# 일반용(을) — 계약전력 300kW 미만, TOU (시간대별 계량기 설치)
# ════════════════════════════════════════

GENERAL_EUL_SMALL_TOU = {
    "high_a_1": {
        "base_per_kw": 7170,
        "summer":      {"off_peak": 89.4, "mid_peak": 140.6, "on_peak": 163.1},
        "spring_fall": {"off_peak": 89.4, "mid_peak": 96.8,  "on_peak": 108.1},
        "winter":      {"off_peak": 98.1, "mid_peak": 128.5, "on_peak": 143.3},
    },
    "high_a_2": {
        "base_per_kw": 8230,
        "summer":      {"off_peak": 84.1, "mid_peak": 135.3, "on_peak": 157.8},
        "spring_fall": {"off_peak": 84.1, "mid_peak": 91.5,  "on_peak": 102.8},
        "winter":      {"off_peak": 92.8, "mid_peak": 123.2, "on_peak": 138.0},
    },
    "high_b_1": {
        "base_per_kw": 7170,
        "summer":      {"off_peak": 88.8, "mid_peak": 137.4, "on_peak": 153.8},
        "spring_fall": {"off_peak": 88.8, "mid_peak": 94.7,  "on_peak": 100.1},
        "winter":      {"off_peak": 97.8, "mid_peak": 125.1, "on_peak": 139.3},
    },
    "high_b_2": {
        "base_per_kw": 8230,
        "summer":      {"off_peak": 83.5, "mid_peak": 132.1, "on_peak": 148.5},
        "spring_fall": {"off_peak": 83.5, "mid_peak": 89.4,  "on_peak": 94.8},
        "winter":      {"off_peak": 92.5, "mid_peak": 119.8, "on_peak": 134.0},
    },
}

# ════════════════════════════════════════
# 일반용(을) — 계약전력 300kW 이상, TOU
# ════════════════════════════════════════

GENERAL_EUL_LARGE_TOU = {
    "high_a_1": {
        "base_per_kw": 7220,
        "summer":      {"off_peak": 92.8, "mid_peak": 145.7, "on_peak": 227.8},
        "spring_fall": {"off_peak": 92.8, "mid_peak": 115.3, "on_peak": 146.0},
        "winter":      {"off_peak": 99.8, "mid_peak": 145.9, "on_peak": 203.4},
    },
    "high_a_2": {
        "base_per_kw": 8320,
        "summer":      {"off_peak": 87.3, "mid_peak": 140.2, "on_peak": 222.3},
        "spring_fall": {"off_peak": 87.3, "mid_peak": 109.8, "on_peak": 140.5},
        "winter":      {"off_peak": 94.3, "mid_peak": 140.4, "on_peak": 197.9},
    },
    "high_a_3": {
        "base_per_kw": 9810,
        "summer":      {"off_peak": 86.4, "mid_peak": 139.6, "on_peak": 209.9},
        "spring_fall": {"off_peak": 86.4, "mid_peak": 108.5, "on_peak": 132.2},
        "winter":      {"off_peak": 93.7, "mid_peak": 139.8, "on_peak": 186.7},
    },
    "high_b_1": {
        "base_per_kw": 6630,
        "summer":      {"off_peak": 95.9, "mid_peak": 148.2, "on_peak": 229.4},
        "spring_fall": {"off_peak": 95.9, "mid_peak": 118.2, "on_peak": 148.5},
        "winter":      {"off_peak": 102.9, "mid_peak": 148.2, "on_peak": 204.4},
    },
    "high_b_2": {
        "base_per_kw": 7380,
        "summer":      {"off_peak": 92.1, "mid_peak": 144.4, "on_peak": 225.6},
        "spring_fall": {"off_peak": 92.1, "mid_peak": 114.4, "on_peak": 144.7},
        "winter":      {"off_peak": 99.1, "mid_peak": 144.4, "on_peak": 200.6},
    },
}


# ════════════════════════════════════════
# 역률 할인/할증 (일반용 전용)
# 전기공급약관 제77조: 지상역률 90% 기준
# ════════════════════════════════════════

POWER_FACTOR_BASE = 0.90  # 기준 역률

def calc_power_factor_adjustment(elec_won: float, power_factor: float | None) -> int:
    """역률 할인(>90%)/할증(<90%) 금액. 전기요금(기본+전력량)에 대해 적용.

    역률 90% 초과: 초과 1%당 기본요금의 0.2% 할인
    역률 90% 미만: 미달 1%당 기본요금의 0.2% 할증
    """
    if power_factor is None or power_factor <= 0:
        return 0
    pf = min(power_factor, 1.0)
    diff_pct = round((pf - POWER_FACTOR_BASE) * 100)
    if diff_pct == 0:
        return 0
    adjustment = elec_won * 0.002 * diff_pct
    return round(adjustment)


# ════════════════════════════════════════
# 월 최저요금
# ════════════════════════════════════════

MIN_BILL_PER_KW = {
    "gap_i": 0,
    "gap_ii": 0,
    "eul_small": 0,
    "eul_large": 0,
}


# ════════════════════════════════════════
# TOU / 단일단가 계산
# ════════════════════════════════════════

def _tou_bill(monthly_kwh, rates, season="spring_fall", tou_ratios=None, max_demand_kw=0):
    if tou_ratios is None:
        tou_ratios = {"off_peak": 0.3, "mid_peak": 0.4, "on_peak": 0.3}
    tou_rates = rates[season]
    energy = sum(monthly_kwh * tou_ratios.get(p, 0) * r for p, r in tou_rates.items())
    base = max_demand_kw * rates["base_per_kw"]
    total = base + energy
    eff = total / monthly_kwh if monthly_kwh > 0 else 0
    return {"base_won": round(base), "energy_won": round(energy), "total_won": round(total),
            "effective_rate": round(eff, 1), "season": season, "tariff_type": "tou"}


def _seasonal_flat_bill(monthly_kwh, rates, season="spring_fall", max_demand_kw=0):
    rate = rates[season]
    energy = monthly_kwh * rate
    base = max_demand_kw * rates["base_per_kw"]
    total = base + energy
    eff = total / monthly_kwh if monthly_kwh > 0 else 0
    return {"base_won": round(base), "energy_won": round(energy), "total_won": round(total),
            "effective_rate": round(eff, 1), "season": season, "tariff_type": "seasonal_flat"}


# ════════════════════════════════════════
# 계약종별 자동 매칭
# ════════════════════════════════════════

_CONTRACT_ALIAS = {
    "주택용": "res_low",
    "주택용(저압)": "res_low",
    "주택용(고압)": "res_high",
    "주택용저압": "res_low",
    "주택용고압": "res_high",
    "일반용": "eul_small_low",
    "일반용(갑)": "gap_i_low",
    "일반용(갑)I": "gap_i_low",
    "일반용(갑)II": "gap_ii_ha1",
    "일반용(갑)Ⅰ": "gap_i_low",
    "일반용(갑)Ⅱ": "gap_ii_ha1",
    "일반용(을)": "eul_small_low",
    "일반용(을)저압": "eul_small_low",
    "일반용(을)고압A": "eul_large_ha1",
    "일반용(을)고압B": "eul_large_hb1",
    "일반용(을)고압C": "eul_large_hc1",
}


def _billing_period_seasons(month: int, meter_day: int) -> list[tuple[str, float]]:
    """검침일 기반 기준 기간의 계절별 일수 비율 산출.

    기준 기간 = 전월 meter_day ~ 당월 meter_day-1.
    예: month=7, meter_day=15 → 기간 6/15~7/14
        6/15~6/30 (16일): 여름(6~8)
        7/1~7/14 (14일): 여름(6~8)
        → [("summer", 1.0)]

    예: month=6, meter_day=20 → 기간 5/20~6/19
        5/20~5/31 (12일): 봄가을(3~5)
        6/1~6/19 (19일): 여름(6~8)
        → [("spring_fall", 12/31), ("summer", 19/31)]
    """
    from .tou import get_season
    import calendar

    if meter_day <= 1:
        return [(get_season(month), 1.0)]

    # 전월
    prev_month = month - 1 if month > 1 else 12
    prev_year_offset = -1 if month == 1 else 0  # 연도 보정용 (계절 판정엔 불필요)
    days_in_prev = calendar.monthrange(2024, prev_month)[1]  # 윤년 대비 2024

    # 전월 구간: meter_day ~ 말일
    days_prev_part = days_in_prev - meter_day + 1
    # 당월 구간: 1일 ~ meter_day-1
    days_curr_part = meter_day - 1
    total_days = days_prev_part + days_curr_part

    if total_days <= 0:
        return [(get_season(month), 1.0)]

    season_prev = get_season(prev_month)
    season_curr = get_season(month)

    if season_prev == season_curr:
        return [(season_prev, 1.0)]

    return [
        (season_prev, days_prev_part / total_days),
        (season_curr, days_curr_part / total_days),
    ]


def _residential_summer_flag(month: int, meter_day: int) -> str | list[tuple[str, float]]:
    """주택용 하계/기타계절 판정 (검침일 고려)."""
    if meter_day <= 1:
        return "summer" if month in (7, 8) else "normal"

    seasons = _billing_period_seasons(month, meter_day)
    flags = []
    for season, ratio in seasons:
        flag = "summer" if season == "summer" else "normal"
        flags.append((flag, ratio))

    if len(flags) == 1:
        return flags[0][0]
    if flags[0][0] == flags[1][0]:
        return flags[0][0]
    return flags


def estimate_bill(
    monthly_kwh: float,
    contract_type: str,
    month: int = 1,
    tou_ratios: dict[str, float] | None = None,
    max_demand_kw: float = 0,
    meter_day: int = 1,
) -> dict:
    """통합 요금 추정 — 계약종별 자동 매칭, 검침일 기반 일할 계산.

    meter_day > 1이면 기준 기간이 전월~당월에 걸치므로,
    두 계절의 단가를 일수 비례로 가중 적용.
    2026.04.16 시행 전기요금표 기준.
    """
    from .tou import get_season

    ct_clean = contract_type.replace(" ", "").strip()
    key = _CONTRACT_ALIAS.get(ct_clean)
    if key is None:
        for alias, k in _CONTRACT_ALIAS.items():
            if alias in ct_clean or ct_clean in alias:
                key = k
                break
    if key is None:
        key = "eul_small_low"

    # 주택용: 누진제 (하계/기타 구분) + 슈퍼유저
    if key in ("res_low", "res_high"):
        tiers_map = RESIDENTIAL_LOW if key == "res_low" else RESIDENTIAL_HIGH
        su_rate = SUPER_USER_RATE if key == "res_low" else SUPER_USER_RATE_HIGH
        # 슈퍼유저: 하계(7~8) 또는 동계(12~2)에 1000kWh 초과 시 적용
        is_super_season = month in (7, 8, 12, 1, 2)
        sf = _residential_summer_flag(month, meter_day)
        if isinstance(sf, str):
            return _progressive_bill(monthly_kwh, tiers_map[sf],
                                     super_user=is_super_season, super_rate=su_rate)
        total_base = 0
        total_energy = 0
        tier_max = 0
        su_extra = 0
        for flag, ratio in sf:
            r = _progressive_bill(monthly_kwh * ratio, tiers_map[flag],
                                  super_user=is_super_season, super_rate=su_rate)
            total_base = max(total_base, r["base_won"])
            total_energy += r["energy_won"]
            tier_max = max(tier_max, r["tier"])
            su_extra += r.get("super_user_extra", 0)
        total = total_base + total_energy
        eff = total / monthly_kwh if monthly_kwh > 0 else 0
        return {"base_won": round(total_base), "energy_won": round(total_energy),
                "total_won": round(total), "tier": tier_max,
                "effective_rate": round(eff, 1), "tariff_type": "progressive",
                "super_user_extra": round(su_extra)}

    # 일반용: 계절별 단가 (단일 or TOU)
    seasons = _billing_period_seasons(month, meter_day)

    if key.startswith("gap_i_"):
        sub = key.replace("gap_i_", "")
        rates = GENERAL_GAP_I.get(sub, GENERAL_GAP_I["low"])
        if len(seasons) == 1:
            return _seasonal_flat_bill(monthly_kwh, rates, seasons[0][0], max_demand_kw)
        return _blended_seasonal_flat(monthly_kwh, rates, seasons, max_demand_kw)

    elif key.startswith("gap_ii_"):
        sub = key.replace("gap_ii_", "")
        rates = GENERAL_GAP_II_TOU.get(sub, GENERAL_GAP_II_TOU["high_a_1"])
        if len(seasons) == 1:
            return _tou_bill(monthly_kwh, rates, seasons[0][0], tou_ratios, max_demand_kw)
        return _blended_tou(monthly_kwh, rates, seasons, tou_ratios, max_demand_kw)

    elif key.startswith("eul_small_"):
        sub = key.replace("eul_small_", "")
        rates = GENERAL_EUL_SMALL.get(sub, GENERAL_EUL_SMALL["low"])
        if len(seasons) == 1:
            return _seasonal_flat_bill(monthly_kwh, rates, seasons[0][0], max_demand_kw)
        return _blended_seasonal_flat(monthly_kwh, rates, seasons, max_demand_kw)

    elif key.startswith("eul_large_"):
        sub = key.replace("eul_large_", "")
        rates = GENERAL_EUL_LARGE_TOU.get(sub, GENERAL_EUL_LARGE_TOU["high_a_1"])
        if len(seasons) == 1:
            return _tou_bill(monthly_kwh, rates, seasons[0][0], tou_ratios, max_demand_kw)
        return _blended_tou(monthly_kwh, rates, seasons, tou_ratios, max_demand_kw)

    else:
        return _seasonal_flat_bill(monthly_kwh, GENERAL_EUL_SMALL["low"],
                                   seasons[0][0], max_demand_kw)


def _blended_seasonal_flat(monthly_kwh, rates, seasons, max_demand_kw=0):
    """두 계절에 걸친 기간의 단일단가 일할 계산."""
    blended_rate = sum(rates[s] * w for s, w in seasons)
    energy = monthly_kwh * blended_rate
    base = max_demand_kw * rates["base_per_kw"]
    total = base + energy
    eff = total / monthly_kwh if monthly_kwh > 0 else 0
    season_desc = " + ".join(f"{s}({w*100:.0f}%)" for s, w in seasons)
    return {"base_won": round(base), "energy_won": round(energy), "total_won": round(total),
            "effective_rate": round(eff, 1), "season": season_desc, "tariff_type": "seasonal_flat_blended"}


def _blended_tou(monthly_kwh, rates, seasons, tou_ratios=None, max_demand_kw=0):
    """두 계절에 걸친 기간의 TOU 일할 계산."""
    if tou_ratios is None:
        tou_ratios = {"off_peak": 0.3, "mid_peak": 0.4, "on_peak": 0.3}
    energy = 0
    for season, weight in seasons:
        tou_rates = rates[season]
        energy += sum(monthly_kwh * weight * tou_ratios.get(p, 0) * r
                      for p, r in tou_rates.items())
    base = max_demand_kw * rates["base_per_kw"]
    total = base + energy
    eff = total / monthly_kwh if monthly_kwh > 0 else 0
    season_desc = " + ".join(f"{s}({w*100:.0f}%)" for s, w in seasons)
    return {"base_won": round(base), "energy_won": round(energy), "total_won": round(total),
            "effective_rate": round(eff, 1), "season": season_desc, "tariff_type": "tou_blended"}


# ════════════════════════════════════════
# 부가 요금 (2026년 2분기 기준, 동결 유지)
# ════════════════════════════════════════

CLIMATE_ENV_RATE = 9.0       # 기후환경요금 (원/kWh) — 2023~ 동결
FUEL_ADJ_RATE = 5.0          # 연료비조정요금 (원/kWh) — 상한 +5원 동결
VAT_RATE = 0.10              # 부가가치세 10%
FUND_RATE = 0.027            # 전력산업기반기금 2.7% (2025.7.1~ 인하)


# ════════════════════════════════════════
# 복지 할인 (주택용 전용, 한전 ON 기준)
# 출처: 한전 전기공급약관 제69조~제75조
# ════════════════════════════════════════

# 약관 제67조 제6항 분류
# group="2": 제2호 정액할인, group="4": 제4호 정률할인
# duplex: True면 2호 마목/사목으로 4호와 중복 가능
WELFARE_DISCOUNTS = {
    "없음":             {"rate": 0,   "cap": 0,     "cap_summer": 0,     "group": None, "duplex": False,
                         "label": "할인 없음"},
    "기초생활(생계/의료)": {"rate": 1.0, "cap": 16000, "cap_summer": 20000, "group": "2",  "duplex": True,
                         "label": "기초생활수급자 생계·의료 (평시 16,000 / 여름 20,000원)"},
    "기초생활(주거/교육)": {"rate": 1.0, "cap": 10000, "cap_summer": 12000, "group": "2",  "duplex": True,
                         "label": "기초생활수급자 주거·교육 (평시 10,000 / 여름 12,000원)"},
    "차상위계층":        {"rate": 1.0, "cap": 8000,  "cap_summer": 10000, "group": "2",  "duplex": True,
                         "label": "차상위계층 (평시 8,000 / 여름 10,000원)"},
    "장애인(심한)":      {"rate": 1.0, "cap": 16000, "cap_summer": 20000, "group": "2",  "duplex": False,
                         "label": "장애의 정도가 심한 장애인 (평시 16,000 / 여름 20,000원)"},
    "국가유공자":        {"rate": 1.0, "cap": 16000, "cap_summer": 20000, "group": "2",  "duplex": False,
                         "label": "국가유공자 1~3급 (평시 16,000 / 여름 20,000원)"},
    "독립유공자":        {"rate": 1.0, "cap": 16000, "cap_summer": 20000, "group": "2",  "duplex": False,
                         "label": "독립유공자 (평시 16,000 / 여름 20,000원)"},
    "사회복지시설":      {"rate": 0.3, "cap": 0,     "cap_summer": 0,     "group": "3",  "duplex": False,
                         "label": "사회복지시설 (30%, 한도 없음)"},
    "다자녀(3자녀+)":    {"rate": 0.3, "cap": 16000, "cap_summer": 16000, "group": "4",  "duplex": False,
                         "label": "다자녀 (30%, 월 16,000원 한도)"},
    "대가족(5인+)":      {"rate": 0.3, "cap": 16000, "cap_summer": 16000, "group": "4",  "duplex": False,
                         "label": "대가족 (30%, 월 16,000원 한도)"},
    "출산가구":          {"rate": 0.3, "cap": 16000, "cap_summer": 16000, "group": "4",  "duplex": False,
                         "label": "출산가구 (30%, 월 16,000원 한도, 3년)"},
    "생명유지장치":      {"rate": 0.3, "cap": 0,     "cap_summer": 0,     "group": "4",  "duplex": False,
                         "label": "생명유지장치 (30%, 한도 없음)"},
}


def calc_welfare_discount(
    elec_won: float,
    welfare_types: str | list[str],
    is_summer: bool = False,
) -> tuple[int, str]:
    """복지할인 금액 계산 — 중복 적용 규정 반영.

    약관 제67조 제6항:
      - 원칙: 가장 큰 할인 하나만 적용
      - 예외: 제2호 마목/사목(기초수급·차상위) + 제4호(다자녀 등) 중복 가능
              → 정액(2호) 먼저 감액 후 잔액에 정률(4호) 적용

    Returns: (할인 금액, 적용 설명)
    """
    if isinstance(welfare_types, str):
        welfare_types = [welfare_types]

    welfare_types = [w for w in welfare_types if w != "없음"]
    if not welfare_types:
        return 0, "없음"

    entries = [(w, WELFARE_DISCOUNTS.get(w)) for w in welfare_types if WELFARE_DISCOUNTS.get(w)]
    if not entries:
        return 0, "없음"

    # 각 항목별 할인 금액 계산
    def _calc_one(info, base_won):
        cap = info["cap_summer"] if is_summer else info["cap"]
        discount = base_won * info["rate"]
        if cap > 0:
            discount = min(discount, cap)
        return round(discount)

    # 2호(duplex=True) + 4호 중복 가능 여부 판별
    group2_duplex = [(w, i) for w, i in entries if i["group"] == "2" and i["duplex"]]
    group4 = [(w, i) for w, i in entries if i["group"] == "4"]

    if group2_duplex and group4:
        # 중복 적용: 정액(2호) 먼저 → 잔액에 정률(4호)
        best2 = max(group2_duplex, key=lambda x: _calc_one(x[1], elec_won))
        discount2 = _calc_one(best2[1], elec_won)
        remaining = elec_won - discount2
        best4 = max(group4, key=lambda x: _calc_one(x[1], remaining))
        discount4 = _calc_one(best4[1], remaining)
        total = discount2 + discount4
        desc = f"{best2[0]} + {best4[0]} (중복)"
        return total, desc

    # 원칙: 가장 큰 것 하나만
    best_discount = 0
    best_name = ""
    for w, info in entries:
        d = _calc_one(info, elec_won)
        if d > best_discount:
            best_discount = d
            best_name = w
    return best_discount, best_name


def final_bill(
    monthly_kwh: float,
    contract_type: str,
    month: int = 1,
    *,
    max_demand_kw: float = 0,
    climate_rate: float = CLIMATE_ENV_RATE,
    fuel_rate: float = FUEL_ADJ_RATE,
    welfare_type: str | list[str] = "없음",
    meter_day: int = 1,
    power_factor: float | None = None,
) -> dict:
    """최종 청구금액 산출 — 전체 단계 포함.

    1) 전기요금 = 기본요금 + 전력량요금 (검침일 일할, 슈퍼유저 포함)
    2) 역률 조정 (일반용: 90% 기준, 초과 할인 / 미달 할증)
    3) 기후환경요금 = 단가 x kWh
    4) 연료비조정요금 = 단가 x kWh
    5) 복지할인 (주택용, 해당 시)
    6) 소계 = 1 + 2 + 3 + 4 - 5
    7) 부가가치세 = 소계 x 10% (원 미만 반올림)
    8) 전력산업기반기금 = 소계 x 2.7% (10원 미만 절사)
    9) 최종 청구 = 6 + 7 + 8

    출처: 2026.04.16 시행 전기요금표(종합)
    """
    base_bill = estimate_bill(monthly_kwh, contract_type, month,
                              max_demand_kw=max_demand_kw, meter_day=meter_day)
    elec_won = base_bill["total_won"]

    # 역률 할인/할증 (일반용만, 주택용은 해당 없음)
    is_residential = base_bill.get("tariff_type") == "progressive"
    pf_adj = 0
    if not is_residential and power_factor is not None:
        pf_adj = calc_power_factor_adjustment(base_bill["base_won"], power_factor)

    climate_won = round(climate_rate * monthly_kwh)
    fuel_won = round(fuel_rate * monthly_kwh)

    is_summer = month in (7, 8)
    welfare_discount, welfare_desc = calc_welfare_discount(elec_won, welfare_type, is_summer)

    subtotal = elec_won - pf_adj + climate_won + fuel_won - welfare_discount
    subtotal = max(subtotal, 0)
    vat = round(subtotal * VAT_RATE)
    fund = int(subtotal * FUND_RATE) // 10 * 10

    total = subtotal + vat + fund

    return {
        **base_bill,
        "elec_won": elec_won,
        "pf_adjustment": pf_adj,
        "climate_won": climate_won,
        "fuel_won": fuel_won,
        "welfare_type": welfare_type,
        "welfare_desc": welfare_desc,
        "welfare_discount": welfare_discount,
        "subtotal": subtotal,
        "vat": vat,
        "fund": fund,
        "final_won": total,
    }


def bill_increase_analysis(
    pred_kwh: float, prev_kwh: float, contract_type: str, month: int = 1,
    *, welfare_type: str = "없음",
) -> dict:
    bill_pred = final_bill(pred_kwh, contract_type, month, welfare_type=welfare_type)
    bill_prev = final_bill(prev_kwh, contract_type, month, welfare_type=welfare_type)
    kwh_chg = (pred_kwh / prev_kwh - 1) * 100 if prev_kwh > 0 else 0
    won_chg = (bill_pred["final_won"] / bill_prev["final_won"] - 1) * 100 if bill_prev["final_won"] > 0 else 0
    return {
        "pred_kwh": pred_kwh, "prev_kwh": prev_kwh,
        "pred_bill": bill_pred["final_won"], "prev_bill": bill_prev["final_won"],
        "kwh_change_pct": round(kwh_chg, 1), "won_change_pct": round(won_chg, 1),
        "amplification": round(won_chg / kwh_chg, 2) if kwh_chg != 0 else 0,
        "tier_prev": bill_prev.get("tier"), "tier_pred": bill_pred.get("tier"),
        "tariff_type": bill_pred.get("tariff_type"),
        "breakdown_pred": bill_pred,
        "breakdown_prev": bill_prev,
    }


def list_supported_contracts() -> list[str]:
    return list(_CONTRACT_ALIAS.keys())
