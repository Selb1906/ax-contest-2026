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

# 프로젝트 루트를 sys.path에 추가 + CWD 고정
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

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
            print(f"  {enc} 실패 (UnicodeError)", flush=True)
            continue
        except Exception as e:
            print(f"  {enc} C엔진 실패 ({type(e).__name__}) → python 엔진 시도", flush=True)
            try:
                test = pd.read_csv(
                    csv_path, encoding=enc, nrows=5,
                    on_bad_lines="skip", engine="python",
                )
                if len(test.columns) >= 2:
                    print(f"[인코딩 감지] {enc} (python engine) — 컬럼 {len(test.columns)}개")
                    print(f"  컬럼: {test.columns.tolist()}")
                    return enc
            except Exception as e2:
                print(f"  {enc} python 엔진도 실패 ({type(e2).__name__})", flush=True)
                continue

    # 모든 인코딩 실패 → 깨진 바이트 제거한 임시 파일 생성
    print("[인코딩 감지] 모든 인코딩 실패 → 깨진 바이트 제거 후 재시도")
    return _create_clean_file(csv_path)


def _create_clean_file(csv_path: str) -> str:
    """깨진 바이트를 제거한 임시 CSV를 생성하고, 특수 인코딩 문자열을 반환."""
    tmp_dir = Path(csv_path).parent
    tmp_path = tmp_dir / "_clean_temp.csv"

    # 디스크 용량 체크: 입력 파일 크기 + 10% 여유
    import shutil as _shutil
    src_size = Path(csv_path).stat().st_size
    free = _shutil.disk_usage(str(tmp_dir)).free
    needed = int(src_size * 1.1)
    if free < needed:
        free_gb = free / (1024 ** 3)
        need_gb = needed / (1024 ** 3)
        print(f"  [경고] 디스크 여유 {free_gb:.1f}GB < 필요 {need_gb:.1f}GB", flush=True)
        print(f"  임시 파일 생성을 건너뛰고 errors=replace 모드로 진행합니다", flush=True)
        return f"__REPLACE__{csv_path}"

    print(f"  임시 파일 생성 중: {tmp_path} (여유 {free/(1024**3):.1f}GB)")

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

_cached_dt_format = None  # 첫 청크에서 감지 후 재사용

def parse_timestamps(chunk: pd.DataFrame) -> pd.DataFrame:
    """ts_date + ts_time → datetime, hour, date, year_month, dayofweek 컬럼 생성.

    검침시분 "2400" → 다음날 00:00 보정.
    """
    date_str = chunk["ts_date"].astype(str).str.strip().str[:10]  # "2022-01-01 00:00:00" → "2022-01-01"
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

    # 포맷 자동 감지 (첫 청크에서 판별 → 캐시 → 이후 재사용)
    global _cached_dt_format
    if _cached_dt_format is None:
        sample = combined.dropna().iloc[0] if len(combined.dropna()) > 0 else ""
        for candidate in ["%Y-%m-%d %H:%M", "%Y%m%d %H:%M", "%Y-%m-%d %H%M",
                           "%Y%m%d %H%M", "%Y/%m/%d %H:%M",
                           "%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S"]:
            try:
                pd.to_datetime(sample, format=candidate)
                _cached_dt_format = candidate
                print(f"    [시간 포맷 감지] {candidate} (샘플: {sample})", flush=True)
                break
            except Exception:
                continue
        if _cached_dt_format is None:
            _cached_dt_format = "mixed"
            print(f"    [시간 포맷] 자동 추론 (느림, 샘플: {sample})", flush=True)

    if _cached_dt_format == "mixed":
        dt = pd.to_datetime(combined, errors="coerce")
    else:
        dt = pd.to_datetime(combined, errors="coerce", format=_cached_dt_format)

    # 그래도 object면 강제 변환
    if not hasattr(dt.dtype, 'tz') and dt.dtype == object:
        dt = pd.to_datetime(dt, errors="coerce")

    # 2400이었던 행 → +1일
    if is_2400.any():
        dt = dt.copy()
        dt.loc[is_2400] = dt.loc[is_2400] + pd.Timedelta(days=1)

    # NaT 행 제거
    nat_count = dt.isna().sum()
    if nat_count > 0:
        print(f"    ts 파싱 실패 {nat_count}건 제거", flush=True)

    chunk["_dt"] = dt
    chunk["hour"] = dt.dt.hour.fillna(0).astype(int)
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
                day_reactive_lag=("_reactive_lag", "sum"),
                day_reactive_lead=("_reactive_lead", "sum"),
                day_apparent_kwh=("p_apparent_kwh", "sum"),
                n_intervals=("p_active_kwh", "count"),
                **( {"temp_c": ("temp_c", "mean")} if "temp_c" in chunk.columns else {} ),
            )
            .reset_index()
        )
        if self._acc is None:
            self._acc = agg
        else:
            merged = pd.concat([self._acc, agg], ignore_index=True)
            agg_dict = {
                "day_kwh": ("day_kwh", "sum"),
                "day_reactive_lag": ("day_reactive_lag", "sum"),
                "day_reactive_lead": ("day_reactive_lead", "sum"),
                "day_apparent_kwh": ("day_apparent_kwh", "sum"),
                "n_intervals": ("n_intervals", "sum"),
            }
            if "temp_c" in merged.columns:
                agg_dict["temp_c"] = ("temp_c", "mean")
            self._acc = (
                merged.groupby(["customer_id", "date", "contract_type"], observed=True)
                .agg(**agg_dict)
                .reset_index()
            )

    def finalize(self) -> pd.DataFrame:
        if self._acc is None:
            return pd.DataFrame()
        df = self._acc.copy()
        # io_adapter 호환 컬럼 추가
        df["ts"] = df["date"]
        df["p_active_kwh"] = df["day_kwh"]
        df["p_reactive_kwh"] = df["day_reactive_lag"] + df["day_reactive_lead"]
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
            "p_reactive_lag_sum": 0.0,
            "p_reactive_lead_sum": 0.0,
            "pf_interval_sum": 0.0,
            "pf_interval_count": 0,
            "max_kwh": 0.0,
            "interval_count": 0,
            "daily_sums": defaultdict(float),
            "tou_off_kwh": 0.0,
            "tou_mid_kwh": 0.0,
            "tou_on_kwh": 0.0,
        })

    @staticmethod
    def _classify_tou(h: int, month: int, dow: int) -> str:
        """TOU 시간대 분류 (인라인, 한전 약관 별표3)."""
        is_holiday_or_sunday = (dow == 6)
        if is_holiday_or_sunday:
            return "off"
        if month in (11, 12, 1, 2):  # 겨울
            if h >= 23 or h < 9:
                return "off"
            if 10 <= h < 12 or 17 <= h < 20 or 22 <= h < 23:
                return "on"
            return "mid"
        else:  # 여름 + 봄가을 (TOU 시간대 동일)
            if h >= 23 or h < 9:
                return "off"
            if 10 <= h < 12 or 13 <= h < 17:
                if dow == 5:  # 토요일: 최대→중간
                    return "mid"
                return "on"
            return "mid"

    def update(self, chunk: pd.DataFrame) -> None:
        cid = chunk["customer_id"].values
        ym = chunk["year_month"].values
        hour = chunk["hour"].values
        dow = chunk["dayofweek"].values
        month_vals = chunk["month"].values if "month" in chunk.columns else pd.to_datetime(chunk["date"]).dt.month.values
        date_vals = chunk["date"].values
        kwh = chunk["p_active_kwh"].values
        apparent = chunk["p_apparent_kwh"].values
        reactive_lag = chunk["_reactive_lag"].values
        reactive_lead = chunk["_reactive_lead"].values

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

            # TOU 시간대별 kWh 누적
            tou = self._classify_tou(h, int(month_vals[i]), dw)
            d[f"tou_{tou}_kwh"] += val

            # 최대 15분 값
            if val > d["max_kwh"]:
                d["max_kwh"] = val

            # 역률 계산용 — lag/lead 분리 누적
            d["p_active_sum"] += val
            if not np.isnan(apparent[i]):
                d["p_apparent_sum"] += float(apparent[i])
            r_lag = float(reactive_lag[i]) if not np.isnan(reactive_lag[i]) else 0.0
            r_lead = float(reactive_lead[i]) if not np.isnan(reactive_lead[i]) else 0.0
            d["p_reactive_lag_sum"] += r_lag
            d["p_reactive_lead_sum"] += r_lead
            # 15분 구간별 역률 (한전 방식: 구간별 독립 계산 후 평균)
            r_interval = max(r_lag, r_lead)
            if val > 0 and r_interval >= 0:
                pf_i = val / math.sqrt(val ** 2 + r_interval ** 2)
                d["pf_interval_sum"] += pf_i
                d["pf_interval_count"] += 1

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
            pr_lag = d["p_reactive_lag_sum"]
            pr_lead = d["p_reactive_lead_sum"]

            # 비율 계산 (0 나누기 방지)
            peak_ratio = d["peak_kwh"] / total if total > 0 else np.nan
            night_ratio = d["night_kwh"] / total if total > 0 else np.nan

            # 부하율: (평균 구간 사용량) / (최대 구간 사용량)
            # 15분/60분 구간 무관하게 비율이므로 보정 불필요
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

            # 역률: 15분 구간별 역률의 월 평균 (한전 방식)
            pf_cnt = d["pf_interval_count"]
            power_factor = d["pf_interval_sum"] / pf_cnt if pf_cnt > 0 else np.nan

            # 무효/유효 비율 (지상 + 진상 합계 기준)
            reactive_ratio = (pr_lag + pr_lead) / pa if pa > 0 else np.nan

            tou_off = d["tou_off_kwh"]
            tou_mid = d["tou_mid_kwh"]
            tou_on = d["tou_on_kwh"]
            tou_total = tou_off + tou_mid + tou_on
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
                "tou_off_kwh": tou_off,
                "tou_mid_kwh": tou_mid,
                "tou_on_kwh": tou_on,
                "tou_off_ratio": tou_off / tou_total if tou_total > 0 else np.nan,
                "tou_mid_ratio": tou_mid / tou_total if tou_total > 0 else np.nan,
                "tou_on_ratio": tou_on / tou_total if tou_total > 0 else np.nan,
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()


class BTMAccumulator:
    """고객별 BTM 신호를 15분 데이터에서 누적 계산."""

    def __init__(self):
        self._data: dict[str, dict] = defaultdict(lambda: {
            "total_count": 0,
            "neg_count": 0,
            "midday_kwh_sum": 0.0,
            "midday_count": 0,
            "other_kwh_sum": 0.0,
            "other_count": 0,
        })

    def update(self, chunk: pd.DataFrame) -> None:
        cid = chunk["customer_id"].values
        kwh = chunk["p_active_kwh"].values
        hour = chunk["hour"].values

        for i in range(len(chunk)):
            if pd.isna(kwh[i]):
                continue
            d = self._data[cid[i]]
            val = float(kwh[i])
            d["total_count"] += 1

            if val < 0:
                d["neg_count"] += 1

            h = int(hour[i])
            if 10 <= h < 15:
                d["midday_kwh_sum"] += val
                d["midday_count"] += 1
            else:
                d["other_kwh_sum"] += val
                d["other_count"] += 1

    def finalize(self) -> pd.DataFrame:
        rows = []
        for cid, d in self._data.items():
            total = d["total_count"]
            neg_rate = d["neg_count"] / total if total > 0 else 0
            midday_avg = d["midday_kwh_sum"] / d["midday_count"] if d["midday_count"] > 0 else 0
            other_avg = d["other_kwh_sum"] / d["other_count"] if d["other_count"] > 0 else 0
            midday_ratio = midday_avg / other_avg if other_avg > 0 else 1.0

            rows.append({
                "customer_id": cid,
                "btm_neg_count": d["neg_count"],
                "btm_neg_rate": neg_rate,
                "btm_midday_ratio": midday_ratio,
                "btm_midday_avg": midday_avg,
                "btm_other_avg": other_avg,
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
    use_errors_replace = False
    if enc_result.startswith("__CLEAN__"):
        actual_path = enc_result.replace("__CLEAN__", "")
        encoding = "utf-8"
        print(f"[인코딩] 깨진 바이트 제거 파일 사용: {actual_path}")
    elif enc_result.startswith("__REPLACE__"):
        actual_path = enc_result.replace("__REPLACE__", "")
        encoding = "utf-8"
        use_errors_replace = True
        print(f"[인코딩] 디스크 부족 → errors=replace 모드: {actual_path}")
    else:
        encoding = enc_result

    # ── 구분자 자동 감지 ──
    with open(actual_path, "r", encoding=encoding, errors="replace") as _f:
        first_line = _f.readline()
    sep = "\t" if "\t" in first_line else ","
    print(f"[구분자] {'탭' if sep == chr(9) else '쉼표'}")

    # ── 원본 컬럼 확인 ──
    _csv_kw = {"encoding": encoding, "sep": sep, "on_bad_lines": "skip"}
    if use_errors_replace:
        _csv_kw["encoding_errors"] = "replace"
    test_df = pd.read_csv(actual_path, nrows=3, **_csv_kw)
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

    # ── ASOS 시간별 기온 로드 (observed/remainder 분리용) ──
    asos_temp_df = None
    asos_parquet = _PROJECT_ROOT / "ASOS" / "asos_all.parquet"
    if asos_parquet.exists():
        try:
            from src.weather.region_match import match_station
            _asos = pd.read_parquet(asos_parquet, columns=["station_id", "ts", "temp_c"])
            _asos["ts"] = pd.to_datetime(_asos["ts"])
            _asos["_ts_h"] = _asos["ts"].dt.floor("h")
            # 관측소×시간별 → 중복 제거 (같은 시간 여러 행 있을 수 있음)
            asos_temp_df = _asos.groupby(["station_id", "_ts_h"], observed=True)["temp_c"].mean().reset_index()
            print(f"[ASOS 기온] {len(asos_temp_df):,}행 로드 (observed/remainder 분리용)")
        except Exception as e:
            print(f"[ASOS 기온] 로드 실패: {e} → 기온 없이 진행")

    # ── 누적기 초기화 ──
    daily_acc = DailyAccumulator()
    ami_acc = AMIAccumulator()
    btm_acc = BTMAccumulator()
    cust_acc = CustomerInfoAccumulator()

    # ── 청크 읽기 ──
    t0 = time.time()
    total_rows = 0
    chunk_idx = 0
    bad_rows = 0

    # 필요한 컬럼만 읽기 (I/O 최적화 — 불필요 컬럼 파싱 건너뜀)
    use_cols = list(active_col_map.keys())
    _chunk_kw = {"encoding": encoding, "sep": sep, "on_bad_lines": "skip",
                 "chunksize": CHUNK_SIZE, "dtype": str, "usecols": use_cols,
                 "encoding_errors": "replace"}
    reader = pd.read_csv(actual_path, **_chunk_kw)
    print(f"[읽기 최적화] {len(use_cols)}개 컬럼만 읽기 (전체 {len(raw_cols)}개 중)")

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
        chunk["_reactive_lag"] = lag
        chunk["_reactive_lead"] = lead

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

        # ── ASOS 기온 조인 (DataFrame merge — 벡터화) ──
        if asos_temp_df is not None:
            chunk["_ts_h"] = chunk["_dt"].dt.floor("h")
            if "region_code" in chunk.columns:
                if not hasattr(run, "_station_cache"):
                    run._station_cache = {}
                for rc in chunk["region_code"].dropna().unique():
                    if rc not in run._station_cache:
                        run._station_cache[rc] = match_station(str(rc)) or 108
                chunk["_sid"] = chunk["region_code"].map(run._station_cache).fillna(108).astype(int)
            else:
                chunk["_sid"] = 108
            chunk = chunk.merge(
                asos_temp_df, left_on=["_sid", "_ts_h"],
                right_on=["station_id", "_ts_h"], how="left"
            )
            chunk = chunk.drop(columns=["_ts_h", "_sid", "station_id"], errors="ignore")

        # ── 누적 집계 ──
        daily_acc.update(chunk)
        ami_acc.update(chunk)
        btm_acc.update(chunk)
        cust_acc.update(chunk)

        total_rows += n_raw
        chunk_idx += 1

        # 매 청크마다 현재 누적 상태 저장
        try:
            if daily_acc._acc is not None and len(daily_acc._acc) > 0:
                _save = daily_acc._acc.copy()
                for c in _save.columns:
                    if _save[c].dtype == object:
                        _save[c] = _save[c].astype(str)
                _new = out / f"daily_chunk{chunk_idx:03d}.parquet"
                _save.to_parquet(_new, index=False)
                # 이전 청크 파일 삭제 (최신 1개만 유지)
                _prev = out / f"daily_chunk{chunk_idx-1:03d}.parquet"
                if _prev.exists():
                    _prev.unlink()
        except Exception:
            pass

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

    # ── 최종 집계 ──
    print(f"\n{'='*60}")
    print("최종 집계 및 Parquet 출력")
    print(f"{'='*60}")

    daily_df = daily_acc.finalize()
    ami_df = ami_acc.finalize()
    btm_df = btm_acc.finalize()
    cust_df = cust_acc.finalize()

    # 청크 중간 저장 파일 정리
    for _old in out.glob("daily_chunk*.parquet"):
        try:
            _old.unlink()
        except Exception:
            pass

    # ── 핵심 검증 (parquet 쓰기 전) ──
    REQUIRED_DAILY_COLS = {"customer_id", "date", "day_kwh", "ts"}
    validation_errors = []

    if len(daily_df) == 0:
        validation_errors.append("daily 집계 결과 0행 — CSV 파싱 완전 실패")
    else:
        missing = REQUIRED_DAILY_COLS - set(daily_df.columns)
        if missing:
            validation_errors.append(f"daily 필수 컬럼 누락: {missing}")
        if daily_df["customer_id"].nunique() == 0:
            validation_errors.append("daily에 고객이 0명")
        neg_pct = (daily_df["day_kwh"] < -1000).mean() if "day_kwh" in daily_df.columns else 0
        if neg_pct > 0.5:
            validation_errors.append(f"day_kwh 50% 이상이 -1000 미만 ({neg_pct:.1%})")
        all_nan_cols = [c for c in daily_df.columns if daily_df[c].isna().all()]
        if len(all_nan_cols) > 3:
            validation_errors.append(f"전체 NaN 컬럼 {len(all_nan_cols)}개: {all_nan_cols[:5]}")

    if validation_errors:
        print(f"\n  [검증 실패] parquet 생성을 중단합니다!")
        for err in validation_errors:
            print(f"    - {err}")
        print(f"\n  원본 CSV로 직접 로딩을 시도하세요.")
        # 실패 manifest 기록 (load_smart가 이걸 보고 CSV fallback)
        fail_meta = {"status": "FAILED", "errors": validation_errors,
                     "input_file": str(csv_path), "total_rows_read": total_rows}
        fail_path = out / "meta.json"
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(fail_meta, f, ensure_ascii=False, indent=2)
        return fail_meta

    # ── Atomic write: 임시 디렉토리 → 검증 → 최종 위치 ──
    import shutil
    import hashlib

    tmp_out = out / "_tmp_write"
    tmp_out.mkdir(parents=True, exist_ok=True)

    def _safe_to_parquet(df_write, path, label):
        """parquet 저장. 실패 시 비압축 재시도, 그래도 실패하면 CSV."""
        for c in df_write.columns:
            if df_write[c].dtype == object:
                df_write[c] = df_write[c].astype(str)
        # 시도 1: pyarrow + snappy
        try:
            df_write.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
            print(f"  {label}.parquet: {len(df_write):,}행, {path.stat().st_size / 1e6:.1f} MB")
            return True
        except Exception as e1:
            print(f"  [시도1 실패] pyarrow+snappy: {e1}", flush=True)

        # 시도 2: 비압축 + Period 타입 문자열 변환
        try:
            df_safe = df_write.copy()
            for c in df_safe.columns:
                if str(df_safe[c].dtype).startswith("period"):
                    df_safe[c] = df_safe[c].astype(str)
            df_safe.to_parquet(path, index=False, engine="pyarrow", compression=None)
            print(f"  {label}.parquet (비압축): {len(df_safe):,}행, {path.stat().st_size / 1e6:.1f} MB")
            return True
        except Exception as e2:
            print(f"  [시도2 실패] pyarrow 비압축: {e2}", flush=True)

        # 최후 수단: CSV
        csv_path = path.with_suffix(".csv")
        print(f"  [최후 수단] parquet 실패 → CSV 저장", flush=True)
        df_write.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  {label}.csv: {len(df_write):,}행, {csv_path.stat().st_size / 1e6:.1f} MB")
        return False

    # 1) daily
    daily_tmp = tmp_out / "daily.parquet"
    daily_ok = _safe_to_parquet(daily_df, daily_tmp, "daily")

    # 2) ami_features
    ami_tmp = tmp_out / "ami_features.parquet"
    if len(ami_df) > 0:
        _safe_to_parquet(ami_df, ami_tmp, "ami_features")
    else:
        print("  [WARN] ami_features 집계 결과 0행 (일별 데이터일 수 있음)")

    # 3) customer_info
    cust_tmp = tmp_out / "customer_info.parquet"
    if len(cust_df) > 0:
        _safe_to_parquet(cust_df, cust_tmp, "customer_info")
    else:
        print("  [WARN] customer_info 집계 결과 0행")

    # 4) btm_signals
    btm_tmp = tmp_out / "btm_signals.parquet"
    if len(btm_df) > 0:
        _safe_to_parquet(btm_df, btm_tmp, "btm_signals")
        n_neg = (btm_df["btm_neg_count"] > 0).sum()
        n_valley = (btm_df["btm_midday_ratio"] < 0.8).sum()
        print(f"    역송 고객: {n_neg}명, 낮시간 저사용: {n_valley}명")
    else:
        print("  [WARN] btm_signals 집계 결과 0행")

    # ── 재검증: daily 파일 재로드 비교 ──
    # parquet 또는 fallback(pkl/csv)로 저장된 실제 파일을 찾아서 검증
    daily_actual = None
    for ext in [".parquet", ".pkl", ".csv"]:
        candidate = tmp_out / f"daily{ext}"
        if candidate.exists():
            daily_actual = candidate
            break

    if daily_actual is None:
        validation_errors.append("daily 파일이 생성되지 않음")
    else:
        print(f"\n  [재검증] {daily_actual.name} 재로드 비교...")
        try:
            if daily_actual.suffix == ".parquet":
                reload_daily = pd.read_parquet(daily_actual)
            elif daily_actual.suffix == ".pkl":
                reload_daily = pd.read_pickle(daily_actual)
            else:
                reload_daily = pd.read_csv(daily_actual)
            if len(reload_daily) != len(daily_df):
                validation_errors.append(f"daily 재로드 행 불일치: {len(daily_df)} → {len(reload_daily)}")
            if set(reload_daily.columns) != set(daily_df.columns):
                validation_errors.append(f"daily 재로드 컬럼 불일치")
            reload_kwh_sum = reload_daily["day_kwh"].sum()
            orig_kwh_sum = daily_df["day_kwh"].sum()
            if abs(reload_kwh_sum - orig_kwh_sum) > 1.0:
                validation_errors.append(f"daily day_kwh 합계 불일치: {orig_kwh_sum:.1f} → {reload_kwh_sum:.1f}")
            else:
                print(f"    daily: {len(reload_daily):,}행, kwh합계={reload_kwh_sum:,.0f} OK")
        except Exception as e:
            validation_errors.append(f"daily 재로드 실패: {e}")

    if validation_errors:
        print(f"\n  [재검증 실패] parquet이 손상되었습니다!")
        for err in validation_errors:
            print(f"    - {err}")
        shutil.rmtree(tmp_out, ignore_errors=True)
        fail_meta = {"status": "FAILED", "errors": validation_errors,
                     "input_file": str(csv_path), "total_rows_read": total_rows}
        with open(out / "meta.json", "w", encoding="utf-8") as f:
            json.dump(fail_meta, f, ensure_ascii=False, indent=2)
        return fail_meta

    # ── 재검증 통과 → 임시 → 최종 위치로 이동 ──
    for tmp_file in tmp_out.iterdir():
        final_file = out / tmp_file.name
        if final_file.exists():
            final_file.unlink()
        shutil.move(str(tmp_file), str(final_file))
    shutil.rmtree(tmp_out, ignore_errors=True)

    # ── 체크섬 계산 (parquet/pkl/csv 어떤 형식이든) ──
    checksums = {}
    for base in ["daily", "ami_features", "customer_info"]:
        for ext in [".parquet", ".pkl", ".csv"]:
            fpath = out / f"{base}{ext}"
            if fpath.exists():
                h = hashlib.md5()
                with open(fpath, "rb") as f:
                    for block in iter(lambda: f.read(65536), b""):
                        h.update(block)
                checksums[fpath.name] = h.hexdigest()
                break

    # 4) meta.json (검증 통과 후에만 status=OK)
    n_cust_daily = daily_df["customer_id"].nunique()
    date_min = str(daily_df["date"].min())
    date_max = str(daily_df["date"].max())
    meta = {
        "status": "OK",
        "input_file": str(csv_path),
        "encoding": encoding,
        "total_rows_read": total_rows,
        "bad_rows_skipped": bad_rows,
        "chunk_size": CHUNK_SIZE,
        "n_chunks": chunk_idx,
        "elapsed_seconds": round(total_time, 1),
        "rows_per_second": round(total_rows / total_time) if total_time > 0 else 0,
        "checksums": checksums,
        "validation": {
            "n_customers": n_cust_daily,
            "date_range": [date_min, date_max],
            "daily_rows": len(daily_df),
            "daily_kwh_sum": round(float(daily_df["day_kwh"].sum()), 1),
            "daily_columns": sorted(daily_df.columns.tolist()),
        },
        "outputs": {
            "daily": {
                "rows": len(daily_df),
                "columns": daily_df.columns.tolist(),
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
    print(f"  고객 수: {n_cust_daily:,}")
    print(f"  날짜 범위: {date_min} ~ {date_max}")
    print(f"  무효행: {bad_rows:,}")
    print(f"  소요: {total_time:.0f}초 ({total_time / 60:.1f}분)")
    print(f"  속도: {total_rows / total_time:,.0f} 행/초" if total_time > 0 else "")
    print(f"  체크섬: {checksums}")

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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="청크 크기 (default: RAM 기반 자동 결정)",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    if not Path(csv_path).exists():
        print(f"[ERROR] 파일이 존재하지 않습니다: {csv_path}")
        sys.exit(1)

    # 청크 크기 결정: 지정 없으면 RAM 기반 자동
    global CHUNK_SIZE
    if args.chunk_size:
        CHUNK_SIZE = args.chunk_size
    else:
        try:
            import shutil as _sh
            total_ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
        except Exception:
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                class _MS(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                                ("ullTotalPhys", ctypes.c_ulonglong)] + [("_pad" + str(i), ctypes.c_ulonglong) for i in range(6)]
                ms = _MS(); ms.dwLength = ctypes.sizeof(ms)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
                total_ram_gb = ms.ullTotalPhys / (1024**3)
            except Exception:
                total_ram_gb = 16
        if total_ram_gb >= 64:
            CHUNK_SIZE = 5_000_000
        elif total_ram_gb >= 32:
            CHUNK_SIZE = 3_000_000
        else:
            CHUNK_SIZE = 1_000_000

    print(f"[입력] {csv_path}")
    print(f"[출력] {args.outdir}")
    print(f"[크기] {Path(csv_path).stat().st_size / (1024**3):.1f} GB")
    print(f"[청크] {CHUNK_SIZE:,} 행 (RAM ~{total_ram_gb:.0f}GB 기반)")
    print()

    run(csv_path, args.outdir)


if __name__ == "__main__":
    main()
