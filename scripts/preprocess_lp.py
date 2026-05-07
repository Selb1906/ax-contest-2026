"""대용량 LP CSV 청크 단위 전처리 → 소규모 Parquet 출력.

34GB / 1.3억 행 CSV를 통째로 읽지 않고, 100만 행씩 읽으면서
일별 집계(daily), 고객x월 AMI 피처(ami_features), 고객 정보(customer_info)를
누적 계산하여 Parquet으로 저장.

출력물은 기존 io_adapter.py / lgbm.py 표준 스키마와 호환.

사용법 (안심구역 PC):
  python scripts/preprocess_lp.py "E:\\lpdata\\data.csv"
  python scripts/preprocess_lp.py "E:\\lpdata\\data.csv" --outdir "E:\\smp\\preprocessed"
"""
from __future__ import annotations

import argparse
import io as _stdio
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# stdout line buffering (진행 표시 즉시 출력)
sys.stdout = _stdio.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", line_buffering=True
)

# 프로젝트 루트를 sys.path에 추가 (src 모듈 임포트용)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────
# 컬럼 매핑 (AX 경진대회 한전 제공 데이터 기준 — 하드코딩)
# ──────────────────────────────────────────────────────────
COL_MAP = {
    "계약번호": "customer_id",
    "검침년월일": "ts_date",
    "검침시분": "ts_time",
    "계약종별": "contract_type",
    "유효전력량계": "p_active_kwh",
    "지상무효전력량계": "p_reactive_lag",
    "진상무효전력량계": "p_reactive_lead",
    "피상전력량계": "p_apparent_kwh",
    "계약전력": "contract_power_kw",
    "본부": "region_code",
    "지사": "region_sub",
    "공급방식": "supply_method",
    "전기사용용도": "usage_purpose",
    "산업분류": "industry_code",
}

# 필수 원본 컬럼 (이것만 있으면 동작)
REQUIRED_RAW_COLS = {"계약번호", "검침년월일", "검침시분", "유효전력량계"}

CHUNK_SIZE = 1_000_000  # 100만 행

# ──────────────────────────────────────────────────────────
# 인코딩 감지
# ──────────────────────────────────────────────────────────

def detect_encoding(csv_path: str) -> str:
    """UTF-8 BOM → UTF-8 → CP949 → EUC-KR 순서로 첫 5행 읽기 시도."""
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            test = pd.read_csv(csv_path, encoding=enc, nrows=5, on_bad_lines="skip")
            if len(test.columns) >= 2:
                print(f"[인코딩 감지] {enc} — 컬럼 {len(test.columns)}개")
                print(f"  컬럼: {test.columns.tolist()}")
                return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            # C 엔진 파싱 에러 → python 엔진으로 재시도
            try:
                test = pd.read_csv(
                    csv_path, encoding=enc, nrows=5,
                    on_bad_lines="skip", engine="python",
                )
                if len(test.columns) >= 2:
                    print(f"[인코딩 감지] {enc} (python engine) — 컬럼 {len(test.columns)}개")
                    print(f"  컬럼: {test.columns.tolist()}")
                    return enc
            except Exception:
                continue

    # 모든 인코딩 실패 → 깨진 바이트 제거한 임시 파일 생성
    print("[인코딩 감지] 모든 인코딩 실패 → 깨진 바이트 제거 후 재시도")
    return _create_clean_file(csv_path)


def _create_clean_file(csv_path: str) -> str:
    """깨진 바이트를 제거한 임시 CSV를 생성하고, 특수 인코딩 문자열을 반환."""
    tmp_dir = Path(csv_path).parent
    tmp_path = tmp_dir / "_clean_temp.csv"
    print(f"  임시 파일 생성 중: {tmp_path}")

    line_count = 0
    with open(csv_path, "rb") as fin, open(tmp_path, "w", encoding="utf-8", newline="") as fout:
        for raw_line in fin:
            # 디코드 불가능한 바이트 제거
            decoded = raw_line.decode("utf-8", errors="ignore")
            fout.write(decoded)
            line_count += 1
            if line_count % 5_000_000 == 0:
                print(f"    {line_count:,} lines...")

    print(f"  임시 파일 완료: {line_count:,} lines → {tmp_path}")
    # 특수 마커: 호출 측에서 이 경로를 사용
    return f"__CLEAN__{tmp_path}"


# ──────────────────────────────────────────────────────────
# 시간 파싱
# ──────────────────────────────────────────────────────────

def parse_timestamps(chunk: pd.DataFrame) -> pd.DataFrame:
    """ts_date + ts_time → datetime, hour, date, year_month, dayofweek 컬럼 생성.

    검침시분 "2400" → 다음날 00:00 보정.
    """
    date_str = chunk["ts_date"].astype(str).str.strip()
    time_str = chunk["ts_time"].astype(str).str.strip()

    # 형식 자동 감지: "0:00" (H:MM) vs "0000" (HHMM)
    if time_str.str.contains(":").any():
        is_2400 = time_str.isin(["24:00", "2400"])
        time_str_fixed = time_str.replace("24:00", "00:00")
        combined = date_str + " " + time_str_fixed
    else:
        time_str = time_str.str.zfill(4)
        is_2400 = time_str == "2400"
        time_str_fixed = time_str.where(~is_2400, "0000")
        combined = date_str + " " + time_str_fixed.str[:2] + ":" + time_str_fixed.str[2:]

    dt = pd.to_datetime(combined, errors="coerce", format="mixed")

    # 2400이었던 행 → +1일
    if is_2400.any():
        dt = dt.copy()
        dt.loc[is_2400] = dt.loc[is_2400] + pd.Timedelta(days=1)

    chunk["_dt"] = dt
    chunk["hour"] = dt.dt.hour
    chunk["date"] = dt.dt.normalize()
    chunk["year_month"] = dt.dt.to_period("M").astype(str)
    chunk["dayofweek"] = dt.dt.dayofweek  # 0=월, 6=일

    return chunk


# ──────────────────────────────────────────────────────────
# 누적 집계 클래스
# ──────────────────────────────────────────────────────────

class DailyAccumulator:
    """customer_id x date x contract_type 일별 집계를 청크마다 누적."""

    def __init__(self):
        self._acc: pd.DataFrame | None = None

    def update(self, chunk: pd.DataFrame) -> None:
        agg = (
            chunk.groupby(["customer_id", "date", "contract_type"], observed=True)
            .agg(
                day_kwh=("p_active_kwh", "sum"),
                day_reactive_kwh=("_reactive_total", "sum"),
                day_apparent_kwh=("p_apparent_kwh", "sum"),
                n_intervals=("p_active_kwh", "count"),
            )
            .reset_index()
        )
        if self._acc is None:
            self._acc = agg
        else:
            merged = pd.concat([self._acc, agg], ignore_index=True)
            self._acc = (
                merged.groupby(["customer_id", "date", "contract_type"], observed=True)
                .agg(
                    day_kwh=("day_kwh", "sum"),
                    day_reactive_kwh=("day_reactive_kwh", "sum"),
                    day_apparent_kwh=("day_apparent_kwh", "sum"),
                    n_intervals=("n_intervals", "sum"),
                )
                .reset_index()
            )

    def finalize(self) -> pd.DataFrame:
        if self._acc is None:
            return pd.DataFrame()
        df = self._acc.copy()
        # io_adapter 호환 컬럼 추가
        df["ts"] = df["date"]
        df["p_active_kwh"] = df["day_kwh"]
        df["p_reactive_kwh"] = df["day_reactive_kwh"]
        return df


class AMIAccumulator:
    """customer_id x year_month 별 AMI 피처 누적 계산."""

    def __init__(self):
        # key = (customer_id, year_month)
        self._data: dict[tuple, dict] = defaultdict(lambda: {
            "total_kwh": 0.0,
            "peak_kwh": 0.0,       # 10~17시
            "night_kwh": 0.0,      # 23시 or 0~8시
            "weekend_kwh": 0.0,
            "weekday_kwh": 0.0,
            "weekend_intervals": 0,
            "weekday_intervals": 0,
            "p_active_sum": 0.0,
            "p_apparent_sum": 0.0,
            "p_reactive_sum": 0.0,
            "max_kwh": 0.0,
            "interval_count": 0,
            "daily_sums": defaultdict(float),
        })

    def update(self, chunk: pd.DataFrame) -> None:
        # 필요 컬럼만 추출하여 numpy 처리로 속도 향상
        cid = chunk["customer_id"].values
        ym = chunk["year_month"].values
        hour = chunk["hour"].values
        dow = chunk["dayofweek"].values
        date_vals = chunk["date"].values
        kwh = chunk["p_active_kwh"].values
        apparent = chunk["p_apparent_kwh"].values
        reactive = chunk["_reactive_total"].values

        for i in range(len(chunk)):
            if pd.isna(kwh[i]):
                continue
            key = (cid[i], ym[i])
            d = self._data[key]
            val = float(kwh[i])

            d["total_kwh"] += val
            d["interval_count"] += 1

            # 피크: 10 <= hour < 17
            h = int(hour[i])
            if 10 <= h < 17:
                d["peak_kwh"] += val

            # 야간: hour >= 23 or hour < 9
            if h >= 23 or h < 9:
                d["night_kwh"] += val

            # 주말/평일
            dw = int(dow[i])
            if dw >= 5:  # 토(5), 일(6)
                d["weekend_kwh"] += val
                d["weekend_intervals"] += 1
            else:
                d["weekday_kwh"] += val
                d["weekday_intervals"] += 1

            # 최대 15분 값
            if val > d["max_kwh"]:
                d["max_kwh"] = val

            # 역률 계산용
            d["p_active_sum"] += val
            if not np.isnan(apparent[i]):
                d["p_apparent_sum"] += float(apparent[i])
            if not np.isnan(reactive[i]):
                d["p_reactive_sum"] += float(reactive[i])

            # 일별 합계 (CV 계산용) — date를 문자열 키로
            date_key = str(date_vals[i])[:10]
            d["daily_sums"][date_key] += val

    def finalize(self) -> pd.DataFrame:
        rows = []
        for (cid, ym), d in self._data.items():
            total = d["total_kwh"]
            cnt = d["interval_count"]
            mx = d["max_kwh"]
            pa = d["p_active_sum"]
            pr = d["p_reactive_sum"]

            # 비율 계산 (0 나누기 방지)
            peak_ratio = d["peak_kwh"] / total if total > 0 else np.nan
            night_ratio = d["night_kwh"] / total if total > 0 else np.nan

            # 부하율: (평균 15분 사용량) / (최대 15분 사용량)
            load_factor = (total / cnt) / mx if cnt > 0 and mx > 0 else np.nan

            # 주말/평일 비율
            we_avg = d["weekend_kwh"] / d["weekend_intervals"] if d["weekend_intervals"] > 0 else np.nan
            wd_avg = d["weekday_kwh"] / d["weekday_intervals"] if d["weekday_intervals"] > 0 else np.nan
            ww_ratio = we_avg / wd_avg if wd_avg and wd_avg > 0 else np.nan

            # 일별 CV
            daily_vals = list(d["daily_sums"].values())
            if len(daily_vals) >= 2:
                arr = np.array(daily_vals, dtype=np.float64)
                mean_d = arr.mean()
                std_d = arr.std(ddof=1)
                daily_cv = std_d / mean_d if mean_d > 0 else np.nan
            else:
                daily_cv = np.nan

            # 역률: p_active / sqrt(p_active^2 + p_reactive^2)
            if pa > 0 and pr >= 0:
                denom = math.sqrt(pa ** 2 + pr ** 2)
                power_factor = pa / denom if denom > 0 else np.nan
            else:
                power_factor = np.nan

            # 무효/유효 비율
            reactive_ratio = pr / pa if pa > 0 else np.nan

            rows.append({
                "customer_id": cid,
                "year_month": ym,
                "observed_peak_ratio": peak_ratio,
                "observed_night_ratio": night_ratio,
                "observed_load_factor": load_factor,
                "observed_weekend_weekday_ratio": ww_ratio,
                "observed_daily_cv": daily_cv,
                "observed_power_factor": power_factor,
                "observed_reactive_ratio": reactive_ratio,
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()


class CustomerInfoAccumulator:
    """customer_id별 고객 정보 1행 (마지막 관측값 유지)."""

    INFO_COLS = [
        "contract_type", "contract_power_kw", "region_code",
        "region_sub", "supply_method", "usage_purpose", "industry_code",
    ]

    def __init__(self):
        self._info: dict[str, dict] = {}

    def update(self, chunk: pd.DataFrame) -> None:
        available = [c for c in self.INFO_COLS if c in chunk.columns]
        if not available:
            return
        # 고객별 마지막 행의 정보 사용
        sub = chunk[["customer_id"] + available].drop_duplicates(
            subset=["customer_id"], keep="last"
        )
        for _, row in sub.iterrows():
            cid = row["customer_id"]
            info = self._info.setdefault(cid, {})
            for col in available:
                val = row[col]
                if pd.notna(val):
                    info[col] = val

    def finalize(self) -> pd.DataFrame:
        if not self._info:
            return pd.DataFrame()
        rows = []
        for cid, info in self._info.items():
            info["customer_id"] = cid
            rows.append(info)
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────
# 메인 처리
# ──────────────────────────────────────────────────────────

def run(csv_path: str, outdir: str) -> dict:
    """CSV를 청크 단위로 읽으며 집계 → Parquet 출력."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 인코딩 감지 ──
    enc_result = detect_encoding(csv_path)
    actual_path = csv_path
    if enc_result.startswith("__CLEAN__"):
        actual_path = enc_result.replace("__CLEAN__", "")
        encoding = "utf-8"
        print(f"[인코딩] 깨진 바이트 제거 파일 사용: {actual_path}")
    else:
        encoding = enc_result

    # ── 구분자 자동 감지 ──
    with open(actual_path, "r", encoding=encoding, errors="replace") as _f:
        first_line = _f.readline()
    sep = "\t" if "\t" in first_line else ","
    print(f"[구분자] {'탭' if sep == chr(9) else '쉼표'}")

    # ── 원본 컬럼 확인 ──
    test_df = pd.read_csv(actual_path, encoding=encoding, sep=sep, nrows=3, on_bad_lines="skip")
    raw_cols = set(test_df.columns)
    missing_required = REQUIRED_RAW_COLS - raw_cols
    if missing_required:
        print(f"[ERROR] 필수 컬럼 누락: {missing_required}")
        print(f"  존재하는 컬럼: {sorted(raw_cols)}")
        sys.exit(1)

    # 존재하는 컬럼만 rename 대상으로 사용
    active_col_map = {k: v for k, v in COL_MAP.items() if k in raw_cols}
    print(f"[컬럼 매핑] {len(active_col_map)}/{len(COL_MAP)} 컬럼 활성")
    for orig, std in active_col_map.items():
        print(f"  {orig:20s} → {std}")

    # ── 누적기 초기화 ──
    daily_acc = DailyAccumulator()
    ami_acc = AMIAccumulator()
    cust_acc = CustomerInfoAccumulator()

    # ── 청크 읽기 ──
    t0 = time.time()
    total_rows = 0
    chunk_idx = 0
    bad_rows = 0

    reader = pd.read_csv(
        actual_path,
        encoding=encoding,
        sep=sep,
        on_bad_lines="skip",
        chunksize=CHUNK_SIZE,
        dtype=str,
    )

    print(f"\n{'='*60}")
    print(f"청크 처리 시작 (CHUNK_SIZE={CHUNK_SIZE:,})")
    print(f"{'='*60}")

    for chunk in reader:
        chunk_t0 = time.time()
        n_raw = len(chunk)

        # 컬럼 rename
        chunk = chunk.rename(columns=active_col_map)

        # 숫자 컬럼 변환
        for num_col in ["p_active_kwh", "p_reactive_lag", "p_reactive_lead",
                        "p_apparent_kwh", "contract_power_kw"]:
            if num_col in chunk.columns:
                chunk[num_col] = pd.to_numeric(chunk[num_col], errors="coerce")

        # 무효전력 합산 (지상 + 진상)
        lag = chunk.get("p_reactive_lag", pd.Series(0.0, index=chunk.index))
        lead = chunk.get("p_reactive_lead", pd.Series(0.0, index=chunk.index))
        lag = pd.to_numeric(lag, errors="coerce").fillna(0.0)
        lead = pd.to_numeric(lead, errors="coerce").fillna(0.0)
        chunk["_reactive_total"] = lag + lead

        # p_apparent_kwh 없으면 0으로
        if "p_apparent_kwh" not in chunk.columns:
            chunk["p_apparent_kwh"] = 0.0
        else:
            chunk["p_apparent_kwh"] = chunk["p_apparent_kwh"].fillna(0.0)

        # 시간 파싱
        chunk = parse_timestamps(chunk)

        # 유효한 행만 (datetime 파싱 성공 + p_active_kwh 존재)
        valid_mask = chunk["_dt"].notna() & chunk["p_active_kwh"].notna()
        n_invalid = (~valid_mask).sum()
        bad_rows += n_invalid
        chunk = chunk[valid_mask].copy()

        if len(chunk) == 0:
            chunk_idx += 1
            total_rows += n_raw
            continue

        # customer_id를 문자열로 확정
        chunk["customer_id"] = chunk["customer_id"].astype(str)

        # contract_type이 없으면 "unknown"
        if "contract_type" not in chunk.columns:
            chunk["contract_type"] = "unknown"
        else:
            chunk["contract_type"] = chunk["contract_type"].fillna("unknown").astype(str)

        # ── 누적 집계 ──
        daily_acc.update(chunk)
        ami_acc.update(chunk)
        cust_acc.update(chunk)

        total_rows += n_raw
        chunk_idx += 1
        elapsed = time.time() - t0
        rate = total_rows / elapsed if elapsed > 0 else 0
        chunk_sec = time.time() - chunk_t0

        print(
            f"  chunk {chunk_idx:>4d}: "
            f"누적 {total_rows:>12,}행 | "
            f"이번 {n_raw:>8,}행 (유효 {len(chunk):,}, 무효 {n_invalid:,}) | "
            f"경과 {elapsed:>7.1f}s | "
            f"{rate:>10,.0f} 행/초 | "
            f"청크 {chunk_sec:.1f}s"
        )

    total_time = time.time() - t0

    # ── 최종 출력 ──
    print(f"\n{'='*60}")
    print("최종 집계 및 Parquet 출력")
    print(f"{'='*60}")

    # 1) daily
    daily_df = daily_acc.finalize()
    daily_path = out / "daily.parquet"
    if len(daily_df) > 0:
        daily_df.to_parquet(daily_path, index=False, engine="pyarrow", compression="snappy")
        print(f"  daily.parquet: {len(daily_df):,}행, {daily_path.stat().st_size / 1e6:.1f} MB")
    else:
        print("  [WARN] daily 집계 결과 0행")

    # 2) ami_features
    ami_df = ami_acc.finalize()
    ami_path = out / "ami_features.parquet"
    if len(ami_df) > 0:
        ami_df.to_parquet(ami_path, index=False, engine="pyarrow", compression="snappy")
        print(f"  ami_features.parquet: {len(ami_df):,}행, {ami_path.stat().st_size / 1e6:.1f} MB")
    else:
        print("  [WARN] ami_features 집계 결과 0행")

    # 3) customer_info
    cust_df = cust_acc.finalize()
    cust_path = out / "customer_info.parquet"
    if len(cust_df) > 0:
        cust_df.to_parquet(cust_path, index=False, engine="pyarrow", compression="snappy")
        print(f"  customer_info.parquet: {len(cust_df):,}행, {cust_path.stat().st_size / 1e6:.1f} MB")
    else:
        print("  [WARN] customer_info 집계 결과 0행")

    # 4) meta.json
    meta = {
        "input_file": str(csv_path),
        "encoding": encoding,
        "total_rows_read": total_rows,
        "bad_rows_skipped": bad_rows,
        "chunk_size": CHUNK_SIZE,
        "n_chunks": chunk_idx,
        "elapsed_seconds": round(total_time, 1),
        "rows_per_second": round(total_rows / total_time) if total_time > 0 else 0,
        "outputs": {
            "daily": {
                "rows": len(daily_df),
                "columns": daily_df.columns.tolist() if len(daily_df) > 0 else [],
            },
            "ami_features": {
                "rows": len(ami_df),
                "columns": ami_df.columns.tolist() if len(ami_df) > 0 else [],
            },
            "customer_info": {
                "rows": len(cust_df),
                "columns": cust_df.columns.tolist() if len(cust_df) > 0 else [],
            },
        },
    }
    meta_path = out / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print(f"  meta.json: {meta_path}")

    # ── 요약 ──
    print(f"\n{'='*60}")
    print("처리 완료 요약")
    print(f"{'='*60}")
    src_size_gb = Path(csv_path).stat().st_size / (1024 ** 3)
    out_files = list(out.glob("*.parquet"))
    out_size_mb = sum(f.stat().st_size for f in out_files) / (1024 ** 2)
    print(f"  입력: {src_size_gb:.1f} GB ({total_rows:,} 행)")
    print(f"  출력: {out_size_mb:.1f} MB ({len(out_files)} parquet 파일)")
    print(f"  무효행: {bad_rows:,}")
    print(f"  소요: {total_time:.0f}초 ({total_time / 60:.1f}분)")
    print(f"  속도: {total_rows / total_time:,.0f} 행/초" if total_time > 0 else "")

    # ── 셀프 테스트 ──
    print(f"\n{'='*60}")
    print("셀프 테스트")
    print(f"{'='*60}")
    _self_test(out, daily_df, ami_df, cust_df)

    # 임시 파일 정리
    if enc_result.startswith("__CLEAN__"):
        clean_path = Path(actual_path)
        if clean_path.exists():
            try:
                clean_path.unlink()
                print(f"\n[정리] 임시 파일 삭제: {clean_path}")
            except OSError:
                print(f"\n[정리] 임시 파일 삭제 실패 (수동 삭제 필요): {clean_path}")

    return meta


def _self_test(
    outdir: Path,
    daily_df: pd.DataFrame,
    ami_df: pd.DataFrame,
    cust_df: pd.DataFrame,
) -> None:
    """출력 파일 존재·기본 품질 확인."""
    ok = True

    # 파일 존재 확인
    for name in ["daily.parquet", "ami_features.parquet", "customer_info.parquet", "meta.json"]:
        fpath = outdir / name
        if fpath.exists():
            print(f"  [OK] {name} 존재 ({fpath.stat().st_size / 1e6:.2f} MB)")
        else:
            print(f"  [FAIL] {name} 없음!")
            ok = False

    # daily 검증
    if len(daily_df) > 0:
        n_cust = daily_df["customer_id"].nunique()
        date_min = daily_df["date"].min()
        date_max = daily_df["date"].max()
        print(f"  [daily] 고객 수: {n_cust:,}")
        print(f"  [daily] 날짜 범위: {date_min} ~ {date_max}")
        print(f"  [daily] 행 수: {len(daily_df):,}")
        neg = (daily_df["day_kwh"] < 0).sum()
        if neg > 0:
            print(f"  [daily] 경고: day_kwh 음수 {neg:,}건 (BTM 역송 가능성)")
    else:
        print("  [WARN] daily 0행 — 데이터 확인 필요")

    # ami_features NaN 비율
    if len(ami_df) > 0:
        print(f"  [ami] 행 수: {len(ami_df):,}")
        for col in ami_df.columns:
            if col in ("customer_id", "year_month"):
                continue
            nan_rate = ami_df[col].isna().mean()
            status = "OK" if nan_rate < 0.1 else "WARN" if nan_rate < 0.5 else "HIGH"
            print(f"  [ami] {col}: NaN {nan_rate:.1%} [{status}]")
    else:
        print("  [WARN] ami_features 0행")

    # customer_info
    if len(cust_df) > 0:
        print(f"  [cust] 고객 수: {len(cust_df):,}")
        print(f"  [cust] 컬럼: {cust_df.columns.tolist()}")
    else:
        print("  [WARN] customer_info 0행")

    if ok:
        print("\n  === 셀프 테스트 통과 ===")
    else:
        print("\n  === 셀프 테스트 일부 실패 — 위 로그 확인 ===")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="대용량 LP CSV → 집계 Parquet 전처리"
    )
    parser.add_argument(
        "csv_path",
        help="입력 CSV 파일 경로 (예: E:\\lpdata\\data.csv)",
    )
    parser.add_argument(
        "--outdir",
        default="data/preprocessed",
        help="출력 디렉토리 (default: data/preprocessed)",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    if not Path(csv_path).exists():
        print(f"[ERROR] 파일이 존재하지 않습니다: {csv_path}")
        sys.exit(1)

    print(f"[입력] {csv_path}")
    print(f"[출력] {args.outdir}")
    print(f"[크기] {Path(csv_path).stat().st_size / (1024**3):.1f} GB")
    print()

    run(csv_path, args.outdir)


if __name__ == "__main__":
    main()
