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
# ════════════════════════════════════════

RESIDENTIAL_LOW = {
    "normal": [  # 기타계절 (1~6, 9~12)
        ProgressiveTier(300,  910,  120.0),
        ProgressiveTier(450,  1600, 214.6),
        ProgressiveTier(float("inf"), 7300, 307.3),
    ],
    "summer": [  # 하계 (7~8)
        ProgressiveTier(200,  730,  105.0),
        ProgressiveTier(400,  1260, 174.0),
        ProgressiveTier(float("inf"), 6060, 242.3),
    ],
}

# 주택용 고압 (2026.04.16)
RESIDENTIAL_HIGH = {
    "normal": [  # 기타계절
        ProgressiveTier(200,  910,  120.0),
        ProgressiveTier(400,  1600, 214.6),
        ProgressiveTier(float("inf"), 7300, 307.3),
    ],
    "summer": [  # 하계
        ProgressiveTier(300,  730,  105.0),
        ProgressiveTier(450,  1260, 174.0),
        ProgressiveTier(float("inf"), 6060, 242.3),
    ],
}


def _progressive_bill(monthly_kwh: float, tiers: list[ProgressiveTier]) -> dict:
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
    total = base + energy
    eff = total / monthly_kwh if monthly_kwh > 0 else 0
    return {"base_won": base, "energy_won": round(energy), "total_won": round(total),
            "tier": tier_idx, "effective_rate": round(eff, 1), "tariff_type": "progressive"}


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


def estimate_bill(
    monthly_kwh: float,
    contract_type: str,
    month: int = 1,
    tou_ratios: dict[str, float] | None = None,
    max_demand_kw: float = 0,
) -> dict:
    """통합 요금 추정 — 계약종별 자동 매칭. 2026.04.16 시행."""
    from .tou import get_season
    season = get_season(month)
    summer_flag = "summer" if month in (7, 8) else "normal"

    ct_clean = contract_type.replace(" ", "").strip()
    key = _CONTRACT_ALIAS.get(ct_clean)
    if key is None:
        for alias, k in _CONTRACT_ALIAS.items():
            if alias in ct_clean or ct_clean in alias:
                key = k
                break
    if key is None:
        key = "eul_small_low"

    if key == "res_low":
        return _progressive_bill(monthly_kwh, RESIDENTIAL_LOW[summer_flag])
    elif key == "res_high":
        return _progressive_bill(monthly_kwh, RESIDENTIAL_HIGH[summer_flag])
    elif key.startswith("gap_i_"):
        sub = key.replace("gap_i_", "")
        rates = GENERAL_GAP_I.get(sub, GENERAL_GAP_I["low"])
        return _seasonal_flat_bill(monthly_kwh, rates, season, max_demand_kw)
    elif key.startswith("gap_ii_"):
        sub = key.replace("gap_ii_", "")
        rates = GENERAL_GAP_II_TOU.get(sub, GENERAL_GAP_II_TOU["high_a_1"])
        return _tou_bill(monthly_kwh, rates, season, tou_ratios, max_demand_kw)
    elif key.startswith("eul_small_"):
        sub = key.replace("eul_small_", "")
        rates = GENERAL_EUL_SMALL.get(sub, GENERAL_EUL_SMALL["low"])
        return _seasonal_flat_bill(monthly_kwh, rates, season, max_demand_kw)
    elif key.startswith("eul_large_"):
        sub = key.replace("eul_large_", "")
        rates = GENERAL_EUL_LARGE_TOU.get(sub, GENERAL_EUL_LARGE_TOU["high_a_1"])
        return _tou_bill(monthly_kwh, rates, season, tou_ratios, max_demand_kw)
    else:
        return _seasonal_flat_bill(monthly_kwh, GENERAL_EUL_SMALL["low"], season, max_demand_kw)


def bill_increase_analysis(
    pred_kwh: float, prev_kwh: float, contract_type: str, month: int = 1,
) -> dict:
    bill_pred = estimate_bill(pred_kwh, contract_type, month)
    bill_prev = estimate_bill(prev_kwh, contract_type, month)
    kwh_chg = (pred_kwh / prev_kwh - 1) * 100 if prev_kwh > 0 else 0
    won_chg = (bill_pred["total_won"] / bill_prev["total_won"] - 1) * 100 if bill_prev["total_won"] > 0 else 0
    return {
        "pred_kwh": pred_kwh, "prev_kwh": prev_kwh,
        "pred_bill": bill_pred["total_won"], "prev_bill": bill_prev["total_won"],
        "kwh_change_pct": round(kwh_chg, 1), "won_change_pct": round(won_chg, 1),
        "amplification": round(won_chg / kwh_chg, 2) if kwh_chg != 0 else 0,
        "tier_prev": bill_prev.get("tier"), "tier_pred": bill_pred.get("tier"),
        "tariff_type": bill_pred.get("tariff_type"),
    }


def list_supported_contracts() -> list[str]:
    return list(_CONTRACT_ALIAS.keys())
