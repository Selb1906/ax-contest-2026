"""1차 분석 결과 분포 기반 합성 LP CSV 생성.

Usage:
    python -m scripts.gen_realistic_synth
    → data/synth/realistic_lp.csv (~1GB)
"""
from __future__ import annotations
import sys, os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd

np.random.seed(42)

# DSZ와 동일 기간
START = pd.Timestamp("2024-02-16")
END = pd.Timestamp("2025-11-01")
INTERVAL = pd.Timedelta(minutes=15)

# 1차 분석 규모 분포
SCALE_DIST = {
    "0-100": 0.104,      # 월 합 0~100 kWh
    "100-1K": 0.511,
    "1K-10K": 0.376,
    "10K+": 0.009,
}

# 월 사용량 범위 (kWh)
SCALE_RANGE = {
    "0-100": (5, 100),
    "100-1K": (100, 1000),
    "1K-10K": (1000, 10000),
    "10K+": (10000, 50000),
}

# 15분 간격 수
ts_range = pd.date_range(START, END, freq="15min")
N_INTERVALS = len(ts_range)
DAYS = (END - START).days

# 1GB 목표: ~10M rows → 고객 수 = 10M / N_INTERVALS
TARGET_ROWS = 10_000_000
N_CUSTOMERS = max(50, TARGET_ROWS // N_INTERVALS)
print(f"기간: {START.date()} ~ {END.date()} ({DAYS}일)")
print(f"15분 간격 수: {N_INTERVALS:,}")
print(f"고객 수: {N_CUSTOMERS}")
print(f"예상 행 수: {N_CUSTOMERS * N_INTERVALS:,}")

# 고객 분배
customers = []
cust_id = 0

# 규모별 고객
for scale, ratio in SCALE_DIST.items():
    n = max(1, int(N_CUSTOMERS * ratio))
    lo, hi = SCALE_RANGE[scale]
    for _ in range(n):
        customers.append({
            "customer_id": f"CNTR_{cust_id:06d}",
            "scale": scale,
            "monthly_kwh": np.random.uniform(lo, hi),
            "contract_type": "일반용(갑)저압",
            "contract_power_kw": np.random.choice([3, 5, 10, 50, 100, 500]),
            "region_code": "서울본부",
            "supply_method": "저압",
            "usage_purpose": np.random.choice(["일반", "산업", "교육", "의료"]),
            "industry_code": np.random.choice(["서비스업", "제조업", "도소매", "기타"]),
            "start_date": START,  # 전체 기간
        })
        cust_id += 1

# 0 사용량 고객 (빈 건물)
N_ZERO = max(2, N_CUSTOMERS // 20)
for _ in range(N_ZERO):
    customers.append({
        "customer_id": f"CNTR_{cust_id:06d}",
        "scale": "zero",
        "monthly_kwh": 0,
        "contract_type": "일반용(갑)저압",
        "contract_power_kw": 3,
        "region_code": "서울본부",
        "supply_method": "저압",
        "usage_purpose": "일반",
        "industry_code": "기타",
        "start_date": START,
    })
    cust_id += 1

# cold start 고객 (2025-06 이후에만 데이터)
N_COLD = max(3, N_CUSTOMERS // 15)
for _ in range(N_COLD):
    customers.append({
        "customer_id": f"CNTR_{cust_id:06d}",
        "scale": np.random.choice(["100-1K", "1K-10K"]),
        "monthly_kwh": np.random.uniform(200, 5000),
        "contract_type": "일반용(갑)저압",
        "contract_power_kw": np.random.choice([5, 10, 50]),
        "region_code": "서울본부",
        "supply_method": "저압",
        "usage_purpose": "일반",
        "industry_code": "서비스업",
        "start_date": pd.Timestamp("2025-06-01"),  # test 기간에만
    })
    cust_id += 1

print(f"총 고객: {len(customers)} (zero={N_ZERO}, cold_start={N_COLD})")

# CSV 생성
OUT = Path("data/synth/realistic_lp.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# 한글 컬럼명 (DSZ 형식)
COL_NAMES = {
    "contract_type": "계약종별",
    "p_active_kwh": "유효전력량계",
    "p_reactive_lag": "지상무효전력량",
    "p_apparent_kwh": "피상전력량",
    "customer_id": "계약번호",
    "meter_date": "검침년월일",
    "meter_time": "검침시분",
    "p_reactive_lead": "진상무효전력량",
    "contract_power_kw": "계약전력",
    "region_code": "본부명",
    "region_sub": "지사명",
    "supply_method": "공급방식",
    "usage_purpose": "전기사용용도",
    "industry_code": "산업분류",
    "meter_interval": "검침주기",
}

print("CSV 생성 중...", flush=True)

chunk_size = 500_000
total_rows = 0

with open(OUT, "w", encoding="utf-8-sig") as f:
    header = ",".join(COL_NAMES.values())
    f.write(header + "\n")

    for ci, cust in enumerate(customers):
        cust_ts = ts_range[ts_range >= cust["start_date"]]
        if len(cust_ts) == 0:
            continue

        n = len(cust_ts)
        monthly = cust["monthly_kwh"]
        # 15분당 kWh = 월합 / (30일 × 96구간)
        base_kwh = monthly / (30 * 96)

        # 시간대 패턴 + 잡음
        hours = cust_ts.hour
        # 주간(9-18) 높고 야간 낮음
        hour_factor = np.where((hours >= 9) & (hours < 18), 1.3, 0.7)
        # 월별 계절성
        months = cust_ts.month
        season_factor = 1.0 + 0.2 * np.sin((months - 7) * np.pi / 6)  # 여름 피크
        # 랜덤 잡음
        noise = np.random.lognormal(0, 0.3, n)

        p_active = np.maximum(0, base_kwh * hour_factor * season_factor * noise)
        if monthly == 0:
            p_active = np.zeros(n)

        p_reactive_lag = p_active * np.random.uniform(0.05, 0.3, n)
        p_reactive_lead = p_active * np.random.uniform(0, 0.05, n)
        p_apparent = np.sqrt(p_active**2 + (p_reactive_lag - p_reactive_lead)**2)

        dates = cust_ts.strftime("%Y%m%d")
        time_int = cust_ts.hour * 100 + cust_ts.minute

        rows_data = pd.DataFrame({
            COL_NAMES["contract_type"]: cust["contract_type"],
            COL_NAMES["p_active_kwh"]: np.round(p_active, 4),
            COL_NAMES["p_reactive_lag"]: np.round(p_reactive_lag, 4),
            COL_NAMES["p_apparent_kwh"]: np.round(p_apparent, 4),
            COL_NAMES["customer_id"]: cust["customer_id"],
            COL_NAMES["meter_date"]: dates,
            COL_NAMES["meter_time"]: time_int,
            COL_NAMES["p_reactive_lead"]: np.round(p_reactive_lead, 4),
            COL_NAMES["contract_power_kw"]: cust["contract_power_kw"],
            COL_NAMES["region_code"]: cust["region_code"],
            COL_NAMES["region_sub"]: cust["region_code"] + "지점",
            COL_NAMES["supply_method"]: cust["supply_method"],
            COL_NAMES["usage_purpose"]: cust["usage_purpose"],
            COL_NAMES["industry_code"]: cust["industry_code"],
            COL_NAMES["meter_interval"]: 15,
        })
        rows_data.to_csv(f, header=False, index=False)
        total_rows += n

        if (ci + 1) % 20 == 0:
            size_mb = OUT.stat().st_size / 1024 / 1024
            print(f"  {ci+1}/{len(customers)} 고객, {total_rows:,}행, {size_mb:.0f}MB", flush=True)

size_mb = OUT.stat().st_size / 1024 / 1024
print(f"\n완료: {OUT} ({total_rows:,}행, {size_mb:.0f}MB)")
print(f"고객: {len(customers)} (zero={N_ZERO}, cold_start={N_COLD})")
