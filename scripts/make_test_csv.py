"""안심구역 환경을 모사한 테스트 CSV 생성.

합성 parquet → 한전 스키마 CSV 변환 + 인코딩 문제 삽입.
preprocess_lp.py → run_full_analysis.py 전체 흐름 테스트용.

안심구역에서 겪은 문제를 재현:
  - UTF-8 BOM + 일부 깨진 바이트
  - 탭/쉼표 구분자
  - 검침시분 H:MM 형식
  - 한글 컬럼명
  - 대용량 (합성 3년치)
"""
from __future__ import annotations

import io as _stdio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import numpy as np
import pandas as pd


def main():
    src = Path("data/synth/lp_1gb.parquet")
    dst = Path("data/synth/test_dsz.csv")

    if not src.exists():
        print(f"[ERROR] {src} 없음. 먼저 gen_synth 실행:")
        print(f"  python -m scripts.gen_synth --config configs/synth_1gb.yaml")
        return

    print(f"[1/4] parquet 로딩...", end=" ", flush=True)
    t0 = time.time()
    df = pd.read_parquet(src)
    print(f"{len(df):,}행 ({time.time()-t0:.1f}초)")

    print(f"[2/4] 한전 스키마로 변환...", flush=True)
    # 컬럼 변환
    out = pd.DataFrame()
    out["계기번호"] = "MTR_" + df["customer_id"].str.extract(r"(\d+)")[0]
    out["계약번호"] = "CNTR_" + df["customer_id"].str.extract(r"(\d+)")[0]
    ts = pd.to_datetime(df["ts"])
    out["검침년월일"] = ts.dt.strftime("%Y-%m-%d").astype(str)
    out["검침시분"] = ts.dt.hour.astype(str) + ":" + ts.dt.minute.astype(str).str.zfill(2)
    out["검침주기"] = "15분"
    out["유효전력량계"] = df["p_active_kwh"].round(1)
    out["지상무효전력량계"] = df["p_reactive_kwh"].round(1)
    out["진상무효전력량계"] = (df["p_reactive_kwh"] * 0.1).round(1)
    out["피상전력량계"] = df["p_apparent_kwh"].round(1)

    # 고객별 고정 정보
    regions = ["서울본부", "경기본부", "인천본부", "대전세종충남본부",
               "광주전남본부", "부산울산본부", "대구본부", "충북본부", "강원본부"]
    branches = ["강남지사", "수원지사", "남인천지사", "대전지사",
                "광주지사", "부산지사", "대구지사", "청주지사", "춘천지사"]
    ct_map = {"일반용": "일반용(갑)", "주택용": "주택용"}
    supply = ["저압", "고압A", "고압B"]
    usage = ["업무시설", "상가", "공장", "학교", "음식점", "아파트", "농업시설"]
    industry = ["서비스업", "도매 및 소매업", "제조업", "교육서비스업",
                "숙박 및 음식점업", "가구 내 고용활동", "농업, 임업 및 어업"]

    np.random.seed(42)
    cust_ids = df["customer_id"].unique()
    cust_info = {}
    for i, cid in enumerate(cust_ids):
        cust_info[cid] = {
            "본부": regions[i % len(regions)],
            "지사": branches[i % len(branches)],
            "공급방식": supply[i % len(supply)],
            "계약전력": np.random.choice([50, 100, 200, 500, 1000]),
            "전기사용용도": usage[i % len(usage)],
            "산업분류": industry[i % len(industry)],
        }

    out["본부"] = df["customer_id"].map(lambda x: cust_info[x]["본부"])
    out["지사"] = df["customer_id"].map(lambda x: cust_info[x]["지사"])
    out["계약종별"] = df["contract_type"].map(ct_map).fillna(df["contract_type"])
    out["공급방식"] = df["customer_id"].map(lambda x: cust_info[x]["공급방식"])
    out["계약전력"] = df["customer_id"].map(lambda x: cust_info[x]["계약전력"])
    out["전기사용용도"] = df["customer_id"].map(lambda x: cust_info[x]["전기사용용도"])
    out["산업분류"] = df["customer_id"].map(lambda x: cust_info[x]["산업분류"])

    print(f"  컬럼: {out.columns.tolist()}")
    print(f"  행: {len(out):,}")

    print(f"[3/4] CSV 저장 (UTF-8 BOM + 깨진 바이트 삽입)...", flush=True)
    t1 = time.time()

    # 먼저 정상 CSV로 저장
    out.to_csv(dst, index=False, encoding="utf-8-sig", sep="\t")

    # 깨진 바이트 삽입 (일부 행에 0xED 등 잘못된 UTF-8 바이트)
    print(f"  정상 CSV 저장 완료 ({time.time()-t1:.1f}초)", flush=True)
    print(f"  깨진 바이트 삽입 중...", end=" ", flush=True)

    with open(dst, "rb") as f:
        data = f.read()

    # 파일 중간중간에 깨진 바이트 삽입 (안심구역에서 겪은 문제 재현)
    bad_bytes = b"\xed\xa0\x80"  # 잘못된 UTF-8 서로게이트
    insert_positions = [
        len(data) // 4,
        len(data) // 2,
        len(data) * 3 // 4,
    ]
    result = bytearray(data)
    for pos in reversed(insert_positions):
        # 줄바꿈 위치를 찾아서 그 직전에 삽입
        newline = result.rfind(b"\n", 0, pos)
        if newline > 0:
            result[newline-3:newline] = bad_bytes

    with open(dst, "wb") as f:
        f.write(result)

    print(f"OK", flush=True)

    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"\n[4/4] 완료!")
    print(f"  파일: {dst}")
    print(f"  크기: {size_mb:.0f} MB")
    print(f"  행: {len(out):,}")
    print(f"  형식: UTF-8 BOM, 탭 구분, H:MM 시간, 깨진 바이트 3곳")
    print(f"\n테스트 실행:")
    print(f"  python -m scripts.preprocess_lp \"{dst}\"")
    print(f"  python -m scripts.run_full_analysis --source configs/source_dsz.yaml")


if __name__ == "__main__":
    main()
