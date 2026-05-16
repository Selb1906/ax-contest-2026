"""안심구역 원스톱 분석 — 체크포인트 + 진행률 + 이중 저장.

중단해도 이어서 실행 가능. 모든 결과는 CSV+PNG+JSON 이중 저장.

사용법:
  python -m scripts.run_full_analysis --source configs/source_dsz.yaml --train-end 2023-12
  python -m scripts.run_full_analysis --source configs/source_dsz.yaml --resume  (이어서 실행)
"""
from __future__ import annotations

import argparse
import gc
import io as _stdio
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# stdout을 화면 + 파일 동시 출력
class _TeeWriter:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

_log_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = open(_PROJECT_ROOT / f"pipeline_log_{_log_ts}.txt", "w", encoding="utf-8")
sys.stdout = _TeeWriter(sys.stdout, _log_file)
sys.stderr = _TeeWriter(sys.stderr, _log_file)

print("[준비] 라이브러리 로딩 중...")
t_import = time.time()

print("  numpy...", end=" ", flush=True)
import numpy as np
print("OK")

print("  pandas...", end=" ", flush=True)
import pandas as pd
print("OK")

print("  src.io_adapter...", end=" ", flush=True)
from src import baselines, eval as ev, io_adapter, profiler
print("OK", flush=True)
print("  src.btm_detect...", end=" ", flush=True)
from src.btm_detect import detect as btm_detect
print("OK", flush=True)
print("  src.models.lgbm...", end=" ", flush=True)
from src.checkpoint import CheckpointManager, SkipStep
from src.models import lgbm
print("OK", flush=True)
print("  src.models.explain...", end=" ", flush=True)
from src.models.explain import run_full_explanation
print("OK", flush=True)
print("  src.preprocess...", end=" ", flush=True)
from src.preprocess import preprocess
from src.result_saver import save_dataframe, save_chart, save_full_results
print(f"OK ({time.time() - t_import:.1f}초)", flush=True)
from src.schemas import CUSTOMER_ID


def main() -> int:
    parser = argparse.ArgumentParser(description="안심구역 전체 분석 파이프라인")
    parser.add_argument("--source", required=True, help="데이터 소스 YAML")
    parser.add_argument("--train-end", default=None, help="학습 종료 기간 (예: 2023-12)")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--resume", action="store_true", help="체크포인트에서 이어서 실행")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--meter-days", default=None,
                        help="검침일 목록 (쉼표 구분, 예: 1,15). 미지정 시 1~31 전수")
    args = parser.parse_args()

    ckpt = CheckpointManager(args.checkpoint_dir)
    skip_done = args.resume

    # 검침일 파싱 (한전 검침일: 1차 1~5, 2차 8~12, 3차 15~17, 4차 18~19, 5차 22~24, 6차 25~26, 7차 말일)
    if args.meter_days:
        METER_DAYS = [int(x) for x in args.meter_days.split(",")]
    else:
        METER_DAYS = [1, 15]

    # 무인 실행을 위한 Step별 결과 기록 + 전체 타이밍
    step_results = {}
    pipeline_start_time = datetime.now()
    pipeline_start_ts = time.time()
    error_log_path = Path("error_log.txt")

    print("""
╔══════════════════════════════════════════════════════╗
║         전체 분석 파이프라인 시작                      ║
╚══════════════════════════════════════════════════════╝""")
    print(f"  소스: {args.source}")
    print(f"  학습 종료: {args.train_end}")
    print(f"  검침일: {METER_DAYS}")
    print(f"  체크포인트: {args.checkpoint_dir}/ {'(이어서 실행)' if skip_done else '(새로 실행)'}")

    # ────────────────────────────────────────────
    # STEP 0: 전처리 parquet 확인/생성
    # ────────────────────────────────────────────
    preprocessed_dir = Path("data/preprocessed")
    daily_parquet = preprocessed_dir / "daily.parquet"
    ami_parquet = preprocessed_dir / "ami_features.parquet"
    cust_parquet = preprocessed_dir / "customer_info.parquet"

    use_preprocessed = daily_parquet.exists()
    if use_preprocessed:
        print(f"\n[전처리 결과 감지] {preprocessed_dir}/ → 즉시 로드", flush=True)
    else:
        # 소스 yaml에서 CSV 경로 추출하여 자동 전처리
        import yaml as _yaml
        with open(args.source, "r", encoding="utf-8") as _f:
            _src_cfg = _yaml.safe_load(_f)
        csv_path = _src_cfg.get("source", {}).get("path", "")
        if csv_path and Path(csv_path).exists() and Path(csv_path).suffix.lower() == ".csv":
            csv_size_gb = Path(csv_path).stat().st_size / (1024**3)
            if csv_size_gb > 1.0:
                print(f"\n[대용량 CSV 감지] {csv_path} ({csv_size_gb:.1f}GB)")
                print(f"  preprocess_lp.py로 청크 변환 시작...")
                import subprocess
                ret = subprocess.run(
                    [sys.executable, "-m", "scripts.preprocess_lp", csv_path,
                     "--outdir", str(preprocessed_dir)],
                    cwd=str(_PROJECT_ROOT),
                )
                if ret.returncode != 0:
                    print(f"  [경고] 전처리 실패 — 원본 CSV로 로딩 시도")

    # ────────────────────────────────────────────
    # STEP 1: 데이터 로딩
    # ────────────────────────────────────────────
    with ckpt.step("step1_load", skip_if_done=False) as prog:
        # meta.json 검증 → parquet 로드 (실패 시 CSV fallback)
        meta_ok = False
        meta_path = preprocessed_dir / "meta.json"
        if daily_parquet.exists() and meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as _mf:
                    _meta = json.load(_mf)
                if _meta.get("status") == "OK":
                    meta_ok = True
                else:
                    print(f"  [경고] meta.json status={_meta.get('status')} → CSV fallback", flush=True)
                    for err in _meta.get("errors", []):
                        print(f"    - {err}", flush=True)
            except Exception as e:
                print(f"  [경고] meta.json 파싱 실패: {e}", flush=True)
        elif daily_parquet.exists():
            print(f"  [경고] meta.json 없음 — parquet 검증 불가, 그래도 로드 시도", flush=True)
            meta_ok = True

        if meta_ok and daily_parquet.exists():
            t_load = time.time()
            print("  daily.parquet 로딩...", end=" ", flush=True)
            df = pd.read_parquet(daily_parquet)
            print(f"OK ({time.time()-t_load:.1f}초, {len(df):,}행)")

            # 필수 컬럼 검증
            required_cols = {"customer_id", "date", "day_kwh"}
            missing_cols = required_cols - set(df.columns)
            if missing_cols or len(df) == 0:
                print(f"  [경고] parquet 필수 컬럼 누락 {missing_cols} 또는 0행 → CSV fallback", flush=True)
                df = io_adapter.load_from_yaml(args.source, validate=False)
                use_preprocessed = False
            else:
                # AMI 피처, 고객 정보도 로드
                if ami_parquet.exists():
                    print("  ami_features.parquet 로딩...", end=" ", flush=True)
                    ami_df = pd.read_parquet(ami_parquet)
                    print(f"OK ({len(ami_df):,}행)")
                if cust_parquet.exists():
                    print("  customer_info.parquet 로딩...", end=" ", flush=True)
                    cust_df = pd.read_parquet(cust_parquet)
                    df = df.merge(cust_df, on="customer_id", how="left", suffixes=("", "_cust"))
                    print(f"OK ({len(cust_df):,}행)")
                prog.update(label=f"전처리 parquet 로드 완료 (검증 통과)")
        else:
            print("  CSV/parquet 원본 로딩...", flush=True)
            df = io_adapter.load_from_yaml(args.source, validate=False)
        # 공공데이터 side-car 컬럼 리네임 (_temp_c → temp_c 등)
        if "_temp_c" in df.columns:
            df = df.rename(columns={"_temp_c": "temp_c", "_humidity": "humidity", "_wind": "wind_speed"})
            print("  [리네임] _temp_c → temp_c, _humidity → humidity, _wind → wind_speed", flush=True)

        n_cust = df["customer_id"].nunique()
        prog.update(label=f"rows={len(df):,}  customers={n_cust}")
        prog.update(label=f"columns: {list(df.columns)}")
        prog.update(label=f"ts: {df['ts'].min()} ~ {df['ts'].max()}")
        ckpt.save_intermediate("step1_load_summary", {
            "rows": len(df), "customers": n_cust,
            "columns": list(df.columns),
            "ts_min": str(df["ts"].min()), "ts_max": str(df["ts"].max()),
        })

    # ────────────────────────────────────────────
    # STEP 1.5: BTM 탐지
    # ────────────────────────────────────────────
    with ckpt.step("step2_btm", skip_if_done=skip_done) as prog:
        # preprocess_lp에서 15분 데이터 기반 BTM 신호가 있으면 우선 사용
        btm_sig_path = preprocessed_dir / "btm_signals.parquet"
        if btm_sig_path.exists():
            btm_sig = pd.read_parquet(btm_sig_path)
            n_neg = (btm_sig["btm_neg_count"] > 0).sum()
            n_valley = (btm_sig["btm_midday_ratio"] < 0.8).sum()
            print(f"  [BTM 신호] 15분 데이터 기반 — 역송 {n_neg}명, 낮저사용 {n_valley}명", flush=True)

            # BTM 판정: 역송 있거나 midday_ratio 이상치 (2σ)
            r_mean = btm_sig["btm_midday_ratio"].mean()
            r_std = btm_sig["btm_midday_ratio"].std()
            threshold = r_mean - 2 * r_std if r_std > 0 else 0
            btm_sig["is_btm"] = (
                (btm_sig["btm_neg_count"] > 0) |
                (btm_sig["btm_midday_ratio"] < threshold)
            ).astype(int)
            btm_sig["btm_score"] = 0.0
            btm_sig.loc[btm_sig["btm_neg_count"] > 0, "btm_score"] += 0.5
            btm_sig.loc[btm_sig["btm_midday_ratio"] < threshold, "btm_score"] += 0.5

            n_btm = btm_sig["is_btm"].sum()
            prog.update(label=f"BTM: {n_btm}명 (역송 {n_neg} + 낮저사용 {n_valley})")

            Path("btm_results").mkdir(exist_ok=True)
            save_dataframe(btm_sig, "btm_results", "btm_flags", "고객별 BTM 플래그 (반출불가)")
            summary = {"n_total": len(btm_sig), "n_btm_detected": int(n_btm),
                       "btm_rate": float(n_btm / len(btm_sig)) if len(btm_sig) > 0 else 0,
                       "n_neg_customers": int(n_neg), "n_valley_customers": int(n_valley)}
            with open("btm_results/btm_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            ckpt.save_intermediate("btm_summary", summary)

            btm_flags = btm_sig[["customer_id", "is_btm", "btm_score"]].copy()
        else:
            # fallback: 일별 데이터로 BTM 탐지 (정확도 낮음)
            print("  [BTM] btm_signals.parquet 없음 — 일별 데이터로 탐지", flush=True)
            btm_result = btm_detect(df)
            n_btm = btm_result.summary["n_btm_detected"]
            prog.update(label=f"BTM: {n_btm}명 ({btm_result.summary['btm_rate']*100:.1f}%)")

            Path("btm_results").mkdir(exist_ok=True)
            save_dataframe(btm_result.customer_flags, "btm_results", "btm_flags", "고객별 BTM 플래그 (반출불가)")
            with open("btm_results/btm_summary.json", "w", encoding="utf-8") as f:
                json.dump(btm_result.summary, f, ensure_ascii=False, indent=2)
            ckpt.save_intermediate("btm_summary", btm_result.summary)
            btm_flags = btm_result.customer_flags[["customer_id", "is_btm", "btm_score"]].copy()

    # BTM 피처 병합
    df = df.merge(btm_flags, on="customer_id", how="left")
    df["is_btm"] = df["is_btm"].fillna(0).astype(int)
    df["btm_score"] = df["btm_score"].fillna(0.0)
    print(f"  BTM 피처 병합 완료 (is_btm={df['is_btm'].sum()}명)")

    # ────────────────────────────────────────────
    # STEP 2: 전처리 (BTM 탐지 후 — 음수 제거해도 안전)
    # ────────────────────────────────────────────
    with ckpt.step("step1_5_preprocess", skip_if_done=skip_done) as prog:
        df, prep_log = preprocess(df)
        for s in prep_log.steps:
            prog.update(label=f"{s['step']}: {s['detail']}")
        summary = prep_log.summary()
        prog.update(label=f"최종: {summary['original_rows']:,} → {summary['final_rows']:,} "
                          f"({summary['removal_rate']*100:.2f}% 제거)")
        ckpt.save_intermediate("preprocess_log", summary)
        Path("preprocess_log").mkdir(exist_ok=True)
        with open("preprocess_log/log.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # ────────────────────────────────────────────
    # STEP 3: 프로파일러
    # ────────────────────────────────────────────
    if not args.skip_profile:
        with ckpt.step("step3_profiler", total=19, skip_if_done=skip_done) as prog:
            profiler.run(df, "profile_stats/dsz")
    else:
        print("\n[skip] 프로파일러")

    # ────────────────────────────────────────────
    # STEP 4: 베이스라인
    # ────────────────────────────────────────────
    # daily/monthly는 이후 모든 Step에서 필요 → 항상 계산
    if 'daily' not in locals():
        daily = ev.daily_by_customer(df)
    if 'monthly' not in locals():
        monthly = ev.monthly_by_customer(daily)

    if 'horizon' not in locals():
        horizon = ev.build_horizon_table(daily, horizons=(10, 20))
    if 'ctx' not in locals():
        ctx = ev.attach_alarm_context(horizon, monthly)

    _step_name = "step4_baselines"
    _t_step = time.time()
    try:
        with ckpt.step("step4_baselines", total=5, skip_if_done=skip_done) as prog:

            baseline_rows = []
            for name, fn in baselines.BASELINES.items():
                try:
                    pred = fn(monthly, horizon)
                    m = ev.evaluate(pred, ctx)
                    m.insert(0, "model", name)
                    baseline_rows.append(m)
                    if "mape_pct" in m.columns and len(m) > 0:
                        mape_avg = m["mape_pct"].mean()
                        prog.update(1, label=f"{name}: MAPE {mape_avg:.2f}%")
                    else:
                        prog.update(1, label=f"{name}: 평가 데이터 부족")
                except Exception as e:
                    prog.update(1, label=f"{name}: 실패 ({e})")

            baseline_metrics = pd.concat(baseline_rows, ignore_index=True)
            save_dataframe(baseline_metrics, "eval_results", "baselines", "베이스라인 5종 평가")
            ckpt.save_intermediate("baselines", baseline_metrics)
        step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(_err_msg, flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
        # 후속 Step에서 필요한 변수 초기화
        if 'daily' not in locals():
            daily = ev.daily_by_customer(df)
        if 'monthly' not in locals():
            monthly = ev.monthly_by_customer(daily)
        if 'horizon' not in locals():
            horizon = ev.build_horizon_table(daily, horizons=(10, 20))
        if 'ctx' not in locals():
            ctx = ev.attach_alarm_context(horizon, monthly)
        baseline_metrics = pd.DataFrame()
    gc.collect()

    # ────────────────────────────────────────────
    # STEP 4.5: 데이터 분할 + 누수 체크
    # ────────────────────────────────────────────
    _step_name = "step4_5_split"
    _t_step = time.time()
    try:
        with ckpt.step("step4_5_split", skip_if_done=skip_done) as prog:
            from src.split import time_split, suggest_split, check_leakage

            features_pre, spec = lgbm.build_features(df)

            # 전처리 모드: preprocess_lp에서 미리 계산한 AMI 피처 조인
            if use_preprocessed and ami_parquet.exists() and 'ami_df' in locals():
                ami_cols_to_join = [c for c in ami_df.columns if c not in features_pre.columns
                                    or c in ("customer_id", "year_month")]
                if len(ami_cols_to_join) > 2:  # customer_id, year_month 외에 피처가 있으면
                    features_pre["year_month_str"] = features_pre["year_month"].astype(str)
                    ami_join = ami_df.copy()
                    ami_join["year_month_str"] = ami_join["year_month"].astype(str)
                    features_pre = features_pre.merge(
                        ami_join.drop(columns=["year_month"], errors="ignore"),
                        left_on=["customer_id", "year_month_str"],
                        right_on=["customer_id", "year_month_str"],
                        how="left"
                    )
                    features_pre = features_pre.drop(columns=["year_month_str"], errors="ignore")
                    print(f"  [AMI 피처 조인] {len(ami_cols_to_join)-2}개 피처 추가", flush=True)

            # 기상 피처 조인 — 고객별 본부→관측소 매핑 우선, 없으면 전국 가중평균
            weather_dir = Path("data/weather")
            weather_joined = False

            # 방법 1: 고객별 지역 매핑 (본부 → ASOS 관측소)
            station_csvs = list(weather_dir.glob("station_*.csv")) if weather_dir.exists() else []
            if station_csvs:
                try:
                    from src.weather.region_match import match_station
                    # YAML 오버라이드 로드
                    _region_override = {}
                    try:
                        import yaml as _yaml
                        with open(args.source, "r", encoding="utf-8") as _f:
                            _src_cfg = _yaml.safe_load(_f)
                        _region_override = _src_cfg.get("source", {}).get("region_station_override", {}) or {}
                        if _region_override:
                            print(f"  [기상] YAML 오버라이드: {_region_override}", flush=True)
                    except Exception:
                        pass

                    def _match_with_override(name):
                        if not name or not isinstance(name, str):
                            return None
                        if name in _region_override:
                            return _region_override[name]
                        return match_station(name)

                    # 고객별 관측소 매핑
                    cust_region = df[["customer_id"]].drop_duplicates("customer_id")
                    if "region_code" in df.columns:
                        cust_region = cust_region.merge(df[["customer_id", "region_code"]].drop_duplicates("customer_id"), on="customer_id", how="left")
                        cust_region["_station_id"] = cust_region["region_code"].apply(
                            lambda x: _match_with_override(str(x)) if pd.notna(x) else None
                        )
                    else:
                        cust_region["_station_id"] = 108  # region_code 없으면 서울(108) 기본
                        print(f"  [기상] region_code 없음 → 전체 서울(108) 매핑", flush=True)
                    n_matched = cust_region["_station_id"].notna().sum()
                    print(f"  [기상] 본부→관측소 매핑: {n_matched}/{len(cust_region)}고객", flush=True)

                    if n_matched > 0:
                        # 전체 관측소 기상 로드
                        all_weather = pd.concat([pd.read_csv(p) for p in station_csvs], ignore_index=True)
                        all_weather["year_month"] = all_weather["year_month"].apply(
                            lambda x: pd.Period(str(x), freq="M"))
                        wcols = [c for c in all_weather.columns if c not in ("year_month", "station_id")]

                        # features에 station_id 부여
                        features_pre = features_pre.merge(
                            cust_region[["customer_id", "_station_id"]], on="customer_id", how="left")
                        # 매칭 안 된 고객은 서울(108) 기본값
                        features_pre["_station_id"] = features_pre["_station_id"].fillna(108).astype(int)

                        # station_id + year_month로 기상 조인
                        features_pre = features_pre.merge(
                            all_weather, left_on=["_station_id", "year_month"],
                            right_on=["station_id", "year_month"], how="left"
                        )
                        features_pre = features_pre.drop(columns=["_station_id", "station_id"], errors="ignore")

                        new_cols = [c for c in wcols if c in features_pre.columns and c not in spec.all]
                        spec.extra_numeric.extend(new_cols)
                        print(f"  [기상 피처 조인] 고객별 관측소 매핑 → {len(new_cols)}개 컬럼 추가", flush=True)
                        weather_joined = True
                except Exception as e:
                    print(f"  [기상] 지역 매핑 실패 ({e}) → 가중평균 시도", flush=True)

            # 방법 2: 전국 가중평균 (fallback)
            if not weather_joined:
                for wname in ["national_v2_power", "national_v1_paper", "national_v3_population"]:
                    wp = weather_dir / f"{wname}.csv"
                    if wp.exists():
                        w = pd.read_csv(wp)
                        w["year_month"] = w["year_month"].apply(lambda x: pd.Period(str(x), freq="M"))
                        wcols = [c for c in w.columns if c not in ("year_month", "station_id")]
                        features_pre = features_pre.merge(w, on="year_month", how="left")
                        new_cols = [c for c in wcols if c in features_pre.columns and c not in spec.all]
                        spec.extra_numeric.extend(new_cols)
                        print(f"  [기상 피처 조인] {wname} (전국 가중평균) → {len(new_cols)}개 컬럼 추가", flush=True)
                        weather_joined = True
                        break
            if not weather_joined:
                print(f"  [기상 피처] data/weather/ 없음 — 기상 없이 진행", flush=True)

            # HDD/CDD: daily.parquet에 기온이 없어도 월별 temp_mean에서 생성
            if weather_joined and "temp_mean" in features_pre.columns:
                from src.utils import hdd_cdd as _hdd_cdd
                _h, _c = _hdd_cdd(features_pre["temp_mean"].to_numpy())
                features_pre["hdd"] = _h
                features_pre["cdd"] = _c
                if "hdd" not in spec.extra_numeric:
                    spec.extra_numeric.extend(["hdd", "cdd"])
                print(f"  [HDD/CDD] temp_mean에서 생성 완료", flush=True)

            prog.update(label=f"피처: {len(spec.all)}개, 행: {len(features_pre):,}")

            if args.train_end:
                train_end_p = pd.Period(args.train_end, freq="M")
                # val_end 자동 설정: train_end + 6개월
                val_end_p = train_end_p + 6
                max_ym = features_pre["year_month"].max()
                if val_end_p >= max_ym:
                    val_end_p = None  # val 없이 train/test만
            else:
                train_end_p, val_end_p = suggest_split(features_pre)

            splits = time_split(features_pre, train_end_p, val_end_p)
            leakage_warnings = check_leakage(features_pre, splits)
            if leakage_warnings:
                prog.update(label=f"누수 경고 {len(leakage_warnings)}건!")
            else:
                prog.update(label="누수 체크 통과")

            ckpt.save_intermediate("split_info", {
                "train_end": str(train_end_p),
                "val_end": str(val_end_p) if val_end_p else None,
                "train_n": int(splits["train"].sum()),
                "val_n": int(splits["val"].sum()),
                "test_n": int(splits["test"].sum()),
                "leakage_warnings": leakage_warnings,
            })
        step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(_err_msg, flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    gc.collect()

    # ────────────────────────────────────────────
    # STEP 5: LightGBM 학습
    # ────────────────────────────────────────────
    # features 확보 — Step 4.5에서 빌드 또는 캐시에서 복원
    _features_cache = Path("checkpoints/features_cache.parquet")
    if 'features_pre' in locals():
        features = features_pre
    elif _features_cache.exists():
        features = pd.read_parquet(_features_cache)
        features["year_month"] = features["year_month"].apply(lambda x: pd.Period(str(x), freq="M"))
        print("[features] 캐시에서 복원", flush=True)
    # 캐시 저장/갱신
    try:
        _feat_save = features.copy()
        for c in _feat_save.columns:
            if str(_feat_save[c].dtype).startswith("period"):
                _feat_save[c] = _feat_save[c].astype(str)
        _feat_save.to_parquet(_features_cache, index=False)
    except Exception:
        pass

    _step_name = "step5_lgbm"
    _t_step = time.time()
    try:
        with ckpt.step("step5_lgbm", skip_if_done=skip_done) as prog:
            train_end = train_end_p  # 분할에서 확정된 값
            n_train = (features["year_month"] <= train_end_p).sum()
            n_test = (features["year_month"] > train_end_p).sum()
            prog.update(label=f"train={n_train:,}  test={n_test:,}  train_end={train_end_p}")
            if n_train == 0:
                prog.update(label="학습 데이터 0건 — train_end를 확인하세요")
                raise ValueError(f"train=0: 데이터가 {features['year_month'].min()}~{features['year_month'].max()} 인데 train_end={train_end_p}")
            booster, test_preds = lgbm.train(features, spec, train_end=train_end)

            lgbm_metrics = ev.evaluate(test_preds, ctx)
            lgbm_metrics.insert(0, "model", "lightgbm")
            mape_str = ", ".join(
                f"+{int(r['horizon_days'])}d {r['mape_pct']:.2f}%"
                for _, r in lgbm_metrics.iterrows()
            )
            prog.update(label=f"lightgbm: {mape_str}")

            all_metrics = pd.concat([baseline_metrics, lgbm_metrics], ignore_index=True)
            save_full_results(
                "eval_results", all_metrics,
                params={"features": spec.all, "train_end": str(train_end)},
                description="전체 모델 평가",
            )
            lgbm.save(booster, spec, "weights/dsz_lgbm", notes="안심구역 실 LP 학습")
            ckpt.save_intermediate("all_metrics", all_metrics)
            ckpt.save_intermediate("lgbm_preds", test_preds)
        step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(_err_msg, flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
        # 후속 Step에서 필요한 변수 초기화
        if 'booster' not in locals():
            booster = None
        if 'test_preds' not in locals():
            test_preds = pd.DataFrame()
        if 'all_metrics' not in locals():
            all_metrics = baseline_metrics.copy() if len(baseline_metrics) > 0 else pd.DataFrame()
    gc.collect()

    # ────────────────────────────────────────────
    # STEP 5.5: Sliding 상세 평가 (기본)
    # ────────────────────────────────────────────
    _step_name = "step5_5_sliding"
    _t_step = time.time()
    try:
        with ckpt.step("step5_5_sliding", total=6, skip_if_done=skip_done) as prog:
            from src.eval_detailed import error_by_dimension, save_detailed_evaluation
            from src.schemas import CUSTOMER_ID as _CID

            train_end_p = pd.Period(args.train_end, freq="M") if args.train_end else None
            if train_end_p is None:
                max_ym = features["year_month"].max()
                train_end_p = pd.Period(f"{max_ym.year - 1}-12", freq="M")

            test_features = features[features["year_month"] > train_end_p].copy()
            for c in spec.categorical:
                if c in test_features.columns:
                    test_features[c] = test_features[c].astype("category")

            sliding_meter_days = [1, 5, 10, 15, 20, 25]
            sliding_results = []

            for md in sliding_meter_days:
                try:
                    horizon_md = ev.build_horizon_table(daily, horizons=(10, 20), meter_day=md)
                    ctx_md = ev.attach_alarm_context(horizon_md, monthly)

                    pred_vals = booster.predict(
                        test_features[spec.all], num_iteration=booster.best_iteration
                    )
                    pred_df = test_features[[_CID, "year_month", "horizon_days"]].copy()
                    pred_df["pred_monthly_kwh"] = pred_vals

                    merged = ctx_md.merge(
                        pred_df, on=[_CID, "year_month", "horizon_days"], how="inner"
                    )
                    merged["meter_day"] = md
                    merged["error"] = merged["pred_monthly_kwh"] - merged["full_month_kwh"]
                    merged["abs_error"] = merged["error"].abs()
                    merged["pct_error"] = merged["abs_error"] / merged["full_month_kwh"].clip(lower=1e-9) * 100
                    merged["month"] = merged["year_month"].apply(lambda x: x.month)

                    mape = merged["pct_error"].mean()
                    prog.update(1, label=f"검침일={md}: MAPE {mape:.3f}%, {len(merged)}건")
                    sliding_results.append(merged)
                except Exception as e:
                    prog.update(1, label=f"검침일={md}: 실패 ({e})")

            if sliding_results:
                all_sliding = pd.concat(sliding_results, ignore_index=True)
                dims = save_detailed_evaluation(all_sliding, "sliding_results")
                ckpt.save_intermediate("sliding_overall", dims.get("overall", pd.DataFrame()))
        step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(_err_msg, flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    gc.collect()

    # ────────────────────────────────────────────
    # STEP 5.6: Partial Linear Sliding 평가 (베이스라인 규모별 비교용)
    # ────────────────────────────────────────────
    _step_name = "step5_6_partial_linear_sliding"
    _t_step = time.time()
    try:
        with ckpt.step("step5_6_pl_sliding", skip_if_done=skip_done) as prog:
            from src.eval_detailed import save_detailed_evaluation as _save_det
            from src.schemas import CUSTOMER_ID as _CID2

            train_end_pl = pd.Period(args.train_end, freq="M") if args.train_end else None
            if train_end_pl is None:
                max_ym_pl = monthly["year_month"].max()
                train_end_pl = pd.Period(f"{max_ym_pl.year - 1}-12", freq="M")

            pl_sliding_meter_days = [1, 5, 10, 15, 20, 25]
            pl_sliding_results = []

            for md in pl_sliding_meter_days:
                try:
                    h_md = ev.build_horizon_table(daily, horizons=(10, 20), meter_day=md)
                    ctx_md = ev.attach_alarm_context(h_md, monthly)
                    test_ctx = ctx_md[ctx_md["year_month"] > train_end_pl].copy()
                    if len(test_ctx) == 0:
                        continue
                    test_ctx["pred_monthly_kwh"] = (
                        test_ctx["partial_kwh"]
                        / test_ctx["days_observed"].clip(lower=1)
                        * test_ctx["days_in_month"]
                    )
                    test_ctx["meter_day"] = md
                    test_ctx["error"] = test_ctx["pred_monthly_kwh"] - test_ctx["full_month_kwh"]
                    test_ctx["abs_error"] = test_ctx["error"].abs()
                    test_ctx["pct_error"] = test_ctx["abs_error"] / test_ctx["full_month_kwh"].clip(lower=1e-9) * 100
                    test_ctx["month"] = test_ctx["year_month"].apply(lambda x: x.month)
                    pl_sliding_results.append(test_ctx)
                    prog.update(label=f"partial_linear md={md}: {len(test_ctx)}건")
                except Exception as _e_pl:
                    prog.update(label=f"partial_linear md={md}: 실패")

            if pl_sliding_results:
                all_pl = pd.concat(pl_sliding_results, ignore_index=True)
                _save_det(all_pl, "sliding_results_partial_linear")
                print(f"  [partial_linear] sliding 평가 완료: {len(all_pl)}건", flush=True)
        step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(_err_msg, flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    gc.collect()

    # ────────────────────────────────────────────
    # STEP 6: SHAP 분석
    # ────────────────────────────────────────────
    _step_name = "step6_shap"
    _t_step = time.time()
    _run_shap = not args.skip_shap
    if _run_shap:
        try:
            import shap as _shap_test  # noqa: F401
        except ImportError:
            print("[skip] SHAP 라이브러리 미설치 — 자동 스킵", flush=True)
            _run_shap = False
            step_results[_step_name] = {"status": "SKIP", "elapsed": 0, "error": "shap 미설치"}
    if _run_shap:
        try:
            with ckpt.step("step6_shap", skip_if_done=skip_done) as prog:
                _train_end_ref = train_end if 'train_end' in locals() else (train_end_p if 'train_end_p' in locals() else None)
                test_mask = features["year_month"] > (_train_end_ref if _train_end_ref else pd.Period("2099-12", freq="M"))
                test_features = features[test_mask] if test_mask.any() else features.tail(500)

                alarm_df = None
                if len(test_preds) > 0:
                    merged = test_preds.merge(ctx, on=["customer_id", "year_month", "horizon_days"], how="left")
                    merged["is_alarm"] = False
                    if "prev_month_kwh" in merged.columns:
                        c1 = merged["pred_monthly_kwh"] > merged["prev_month_kwh"] * 1.30
                        c2 = merged["pred_monthly_kwh"] > merged.get("yoy_month_kwh", pd.Series(dtype=float)) * 1.30
                        c3 = merged["pred_monthly_kwh"] > merged.get("ma3_kwh", pd.Series(dtype=float)) * 1.50
                        merged["is_alarm"] = c1.fillna(False) | c2.fillna(False) | c3.fillna(False)
                    alarm_df = merged[["is_alarm"]].reset_index(drop=True)

                prog.update(label="SHAP 산출 중... (시간 소요)")
                results = run_full_explanation(
                    booster, test_features, spec, "explain_results", alarm_labels=alarm_df
                )
                prog.update(label=f"산출물: {list(results.keys())}")
                ckpt.save_intermediate("shap_results", results)
            step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
        except Exception as e:
            _elapsed = time.time() - _t_step
            _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
            print(_err_msg, flush=True)
            step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    elif args.skip_shap:
        print("\n[skip] SHAP 분석", flush=True)
        step_results[_step_name] = {"status": "SKIP", "elapsed": 0, "error": "사용자 스킵"}
    gc.collect()

    # ────────────────────────────────────────────
    # STEP 7: 기상 최적화
    # ────────────────────────────────────────────
    _step_name = "step7_weather_opt"
    _t_step = time.time()
    try:
        with ckpt.step("step7_weather_opt", skip_if_done=skip_done) as prog:
            try:
                from src.weather.asos import read_asos_directory, monthly_weather_features
                from src.weather.optimize import run_weather_optimization

                asos_dir = Path("data/weather/asos")
                if not asos_dir.exists():
                    asos_dir = Path("ASOS")
                if asos_dir.exists():
                    prog.update(label="ASOS 로드 중...")
                    asos_df = read_asos_directory(asos_dir)
                    station_feats = monthly_weather_features(asos_df)
                    monthly_with_ct = monthly.copy()
                    if "contract_type" not in monthly_with_ct.columns:
                        cust_ct = df[["customer_id", "contract_type"]].drop_duplicates()
                        monthly_with_ct = monthly_with_ct.merge(cust_ct, on="customer_id", how="left")
                    # 누수 방지: test 기간 제외
                    test_cutoff = val_end_p if val_end_p else train_end_p
                    monthly_trainval = monthly_with_ct[monthly_with_ct["year_month"] <= test_cutoff]
                    prog.update(label=f"최적화 실행 중 (test 제외, ~{test_cutoff})...")
                    opt_results = run_weather_optimization(
                        asos_df, monthly_trainval, station_feats, out_dir="weather_opt"
                    )
                    ckpt.save_intermediate("weather_opt", opt_results)
                else:
                    prog.update(label="ASOS 디렉터리 미발견 — 스킵")
            except Exception as e:
                prog.update(label=f"기상 최적화 실패: {e}")
        step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(_err_msg, flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    gc.collect()

    # ────────────────────────────────────────────
    # STEP 8: 기상 민감도 (기온/강수/일사 × 방향성 편향 × 월별 분석)
    # ────────────────────────────────────────────
    _step_name = "step8_sensitivity"
    _t_step = time.time()
    try:
        with ckpt.step("step8_sensitivity", skip_if_done=skip_done) as prog:
            if 'test_features' not in locals() or len(test_features) == 0:
                prog.update(label="테스트 피처 없음 — 스킵")
            else:
                from src.schemas import CUSTOMER_ID as _CID
                sens_out = Path("sensitivity_results")
                sens_out.mkdir(exist_ok=True)

                def _predict(X_mod):
                    X_c = X_mod.copy()
                    for c in spec.categorical:
                        if c in X_c.columns:
                            X_c[c] = X_c[c].astype("category")
                    return booster.predict(X_c[spec.all], num_iteration=booster.best_iteration)

                def _eval_metrics(X_mod):
                    pred = _predict(X_mod)
                    actual = X_mod["full_month_kwh"].values
                    mask = np.isfinite(pred) & np.isfinite(actual) & (actual > 0)
                    err = pred[mask] - actual[mask]
                    ae = np.abs(err)
                    mape = float(np.mean(ae / actual[mask]) * 100)
                    mae = float(np.mean(ae))
                    rmse = float(np.sqrt(np.mean(err ** 2)))
                    cvrmse = float(rmse / np.mean(actual[mask]) * 100) if np.mean(actual[mask]) > 0 else np.nan
                    return {"mape": mape, "mae": mae, "rmse": rmse, "cvrmse": cvrmse}

                def _eval_monthly_mape(X_mod):
                    pred = _predict(X_mod)
                    X_tmp = X_mod.copy()
                    X_tmp["_pred"] = pred
                    X_tmp["_month"] = X_tmp["year_month"].apply(lambda x: x.month)
                    X_tmp["_pct_err"] = (X_tmp["_pred"] - X_tmp["full_month_kwh"]).abs() / X_tmp["full_month_kwh"].clip(lower=1e-9) * 100
                    return X_tmp.groupby("_month")["_pct_err"].mean().to_dict()

                all_sensitivity = []

                # === 기온 민감도 ===
                temp_cols = [c for c in spec.all if any(k in c for k in ["cdh", "hdh", "temp", "hdd", "cdd"])]
                if temp_cols:
                    prog.update(label="기온 민감도...")
                    base_m = _eval_metrics(test_features)
                    base_monthly = _eval_monthly_mape(test_features)
                    all_sensitivity.append({"type": "temp", "bias": 0, **base_m, **{f"m{k}": v for k, v in base_monthly.items()}})

                    for bias in [-5, -3, -2, -1, 1, 2, 3, 5]:
                        X_mod = test_features.copy()
                        for c in temp_cols:
                            if "temp" in c:
                                X_mod[c] = X_mod[c] + bias
                            elif "cdh" in c or "cdd" in c:
                                X_mod[c] = (X_mod[c] + abs(bias) * 24).clip(lower=0) if bias > 0 else (X_mod[c] - abs(bias) * 24).clip(lower=0)
                            elif "hdh" in c or "hdd" in c:
                                X_mod[c] = (X_mod[c] - abs(bias) * 24).clip(lower=0) if bias > 0 else (X_mod[c] + abs(bias) * 24).clip(lower=0)
                        m = _eval_metrics(X_mod)
                        monthly_mape = _eval_monthly_mape(X_mod)
                        prog.update(label=f"  temp {bias:+d}°C: MAPE {m['mape']:.3f}%")
                        all_sensitivity.append({"type": "temp", "bias": bias, **m, **{f"m{k}": v for k, v in monthly_mape.items()}})

                # === 강수 민감도 ===
                rain_cols = [c for c in spec.all if "precip" in c or "rainy" in c]
                if rain_cols:
                    prog.update(label="강수 민감도...")
                    for scale in [0.55, 0.8, 1.2, 1.45]:
                        X_mod = test_features.copy()
                        for c in rain_cols:
                            X_mod[c] = X_mod[c] * scale
                        m = _eval_metrics(X_mod)
                        monthly_mape = _eval_monthly_mape(X_mod)
                        label_str = f"rain ×{scale:.2f}"
                        prog.update(label=f"  {label_str}: MAPE {m['mape']:.3f}%")
                        all_sensitivity.append({"type": "rain", "bias": scale, **m, **{f"m{k}": v for k, v in monthly_mape.items()}})

                # === 일사 민감도 ===
                solar_cols = [c for c in spec.all if "solar" in c]
                if solar_cols:
                    prog.update(label="일사 민감도...")
                    for scale in [0.5, 0.7, 1.3, 1.5]:
                        X_mod = test_features.copy()
                        for c in solar_cols:
                            X_mod[c] = X_mod[c] * scale
                        m = _eval_metrics(X_mod)
                        monthly_mape = _eval_monthly_mape(X_mod)
                        label_str = f"solar ×{scale:.1f}"
                        prog.update(label=f"  {label_str}: MAPE {m['mape']:.3f}%")
                        all_sensitivity.append({"type": "solar", "bias": scale, **m, **{f"m{k}": v for k, v in monthly_mape.items()}})

                # 저장
                if all_sensitivity:
                    sens_df = pd.DataFrame(all_sensitivity)
                    sens_df.to_csv(sens_out / "sensitivity_all.csv", index=False, encoding="utf-8-sig")
                    sens_df.to_parquet(sens_out / "sensitivity_all.parquet", index=False)
                    prog.update(label=f"민감도 분석 완료: {len(all_sensitivity)}개 실험")

                    # 월별 히트맵 데이터 분리 저장
                    for stype in ["temp", "rain", "solar"]:
                        sub = sens_df[sens_df["type"] == stype]
                        if len(sub) > 0:
                            sub.to_csv(sens_out / f"sensitivity_{stype}.csv", index=False, encoding="utf-8-sig")
                else:
                    prog.update(label="기상 피처 없음 — 스킵")

        step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(_err_msg, flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    gc.collect()

    # ────────────────────────────────────────────
    # STEP 10: Ablation Study — 최적 조합 탐색
    # ────────────────────────────────────────────
    _step_name = "step10_ablation"
    _t_step = time.time()
    optimal_config = None  # 후속 Step에서 참조
    try:
        with ckpt.step("step10_ablation", skip_if_done=skip_done) as prog:
            try:
                import subprocess
                prog.update(label="Ablation Study 실행 중...")
                # Ablation에는 val_end를 전달 — Ablation 내부에서 train은 ~val_end, test는 val_end 이후
                abl_end = str(val_end_p) if 'val_end_p' in locals() and val_end_p else (args.train_end or f"{features['year_month'].max().year - 1}-12")
                result = subprocess.run(
                    [sys.executable, "-m", "scripts.ablation_study",
                     "--source", args.source, "--train-end", abl_end],
                    capture_output=False, cwd=str(_PROJECT_ROOT),
                )
                if result.returncode == 0:
                    opt_path = Path("ablation_results/optimal_config.json")
                    if opt_path.exists():
                        with open(opt_path, encoding="utf-8") as f:
                            optimal_config = json.load(f)
                        prog.update(label=f"최적 설정: {optimal_config}")
                    else:
                        prog.update(label="최적 설정 파일 없음 — 기본값 유지")
                else:
                    prog.update(label="Ablation 실패 — 기본값 유지")
            except Exception as e:
                prog.update(label=f"Ablation 실패: {e}")
        _abl_ok = result.returncode == 0 if 'result' in locals() else False
        step_results[_step_name] = {"status": "OK" if _abl_ok else "FAIL",
                                    "elapsed": time.time() - _t_step,
                                    **({"error": "subprocess 실패"} if not _abl_ok else {})}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(_err_msg, flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    gc.collect()

    # ────────────────────────────────────────────
    # STEP 11: 최적 조합으로 재학습 + 피처 자동 선정
    # ────────────────────────────────────────────
    _step_name = "step11_retrain_optimal"
    _t_step = time.time()
    try:
        with ckpt.step("step11_retrain_optimal", skip_if_done=skip_done) as prog:
            if optimal_config and optimal_config != {}:
                prog.update(label="최적 설정으로 재학습 중...")

                # 피처 선택이 최적이면 auto_select 적용
                if optimal_config.get("feature_selection", False):
                    from src.feature_selection import auto_select_features, print_selection_report
                    prog.update(label="피처 자동 선정 중 (train+val만, 누수 방지)...")
                    # 누수 방지: test 제외
                    _val_end_ref = val_end_p if 'val_end_p' in locals() and val_end_p else train_end_p
                    trainval_features = features[features["year_month"] <= _val_end_ref]
                    sel_result = auto_select_features(
                        trainval_features, spec.numeric, "full_month_kwh",
                        corr_threshold=0.85, vif_threshold=10.0,
                    )
                    print_selection_report(sel_result)
                    save_dataframe(sel_result.ranking, "feature_selection", "ranking", "피처 순위")

                    # 선택되지 않은 피처 NaN
                    features_opt = features.copy()
                    for c in spec.numeric:
                        if c not in sel_result.selected:
                            features_opt[c] = np.nan
                    prog.update(label=f"선정 피처: {len(sel_result.selected)}개 / {len(spec.numeric)}개")
                else:
                    features_opt = features
                    prog.update(label="피처 선택 OFF — 전체 피처 유지")

                # 재학습
                if args.train_end:
                    train_end_p = pd.Period(args.train_end, freq="M")
                elif 'train_end_p' not in locals() or train_end_p is None:
                    max_ym = features["year_month"].max()
                    train_end_p = pd.Period(f"{max_ym.year - 1}-12", freq="M")
                booster_opt, preds_opt = lgbm.train(features_opt, spec, train_end=train_end_p)
                lgbm.save(booster_opt, spec, "weights/dsz_lgbm_optimal",
                          notes=f"Ablation 최적: {optimal_config}")

                # 재평가 (sliding)
                prog.update(label="최적 모델 Sliding 재평가 중...")
                test_opt = features_opt[features_opt["year_month"] > train_end_p].copy()
                for c in spec.categorical:
                    if c in test_opt.columns:
                        test_opt[c] = test_opt[c].astype("category")

                opt_sliding = []
                for md in METER_DAYS:
                    try:
                        h_md = ev.build_horizon_table(daily, horizons=(10, 20), meter_day=md)
                        ctx_md = ev.attach_alarm_context(h_md, monthly)
                        pred_vals = booster_opt.predict(
                            test_opt[spec.all], num_iteration=booster_opt.best_iteration
                        )
                        pred_df = test_opt[[CUSTOMER_ID, "year_month", "horizon_days"]].copy()
                        pred_df["pred_monthly_kwh"] = pred_vals
                        merged = ctx_md.merge(pred_df, on=[CUSTOMER_ID, "year_month", "horizon_days"], how="inner")
                        merged["meter_day"] = md
                        merged["error"] = merged["pred_monthly_kwh"] - merged["full_month_kwh"]
                        merged["pct_error"] = merged["error"].abs() / merged["full_month_kwh"].clip(lower=1e-9) * 100
                        merged["month"] = merged["year_month"].apply(lambda x: x.month)
                        opt_sliding.append(merged)
                    except Exception:
                        pass

                if opt_sliding:
                    from src.eval_detailed import save_detailed_evaluation
                    all_opt = pd.concat(opt_sliding, ignore_index=True)
                    dims_opt = save_detailed_evaluation(all_opt, "sliding_results_optimal")
                    ckpt.save_intermediate("optimal_sliding", dims_opt.get("overall", pd.DataFrame()))
            else:
                prog.update(label="Ablation 미실행 — 기본 모델 유지")
        step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(_err_msg, flush=True)
        print(traceback.format_exc(), flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    gc.collect()

    # ────────────────────────────────────────────
    # STEP 11.5: 잔차 보정 (2-stage Ridge)
    # ────────────────────────────────────────────
    _step_name = "step11_5_residual"
    _t_step = time.time()
    try:
        with ckpt.step("step11_5_residual", skip_if_done=skip_done) as prog:
            try:
                from src.residual_correction import run_residual_correction
                prog.update(label="잔차 보정 모델 학습 중...")

                # 최적 모델(또는 기본 모델)의 예측값과 실측값
                best_booster = booster
                best_spec = spec
                opt_model_path = Path("weights/dsz_lgbm_optimal/model.txt")
                if opt_model_path.exists():
                    best_booster, best_spec = lgbm.load("weights/dsz_lgbm_optimal")
                    prog.update(label="최적 모델 기반 잔차 보정")

                feat_clean = features.dropna(subset=["full_month_kwh", "partial_kwh"]).copy()
                for c in best_spec.categorical:
                    if c in feat_clean.columns:
                        feat_clean[c] = feat_clean[c].astype("category")

                # 모델이 기대하는 피처만 사용, 없는 컬럼은 NaN
                for c in best_spec.all:
                    if c not in feat_clean.columns:
                        feat_clean[c] = np.nan

                pred_all = best_booster.predict(
                    feat_clean[best_spec.all], num_iteration=best_booster.best_iteration
                )
                actual_all = feat_clean["full_month_kwh"].values
                train_mask_arr = (feat_clean["year_month"] <= train_end_p).values

                res_result = run_residual_correction(
                    feat_clean, pred_all, actual_all, train_mask_arr,
                    out_dir="residual_correction"
                )
                ckpt.save_intermediate("residual_correction", res_result)
            except Exception as e:
                prog.update(label=f"잔차 보정 실패: {e}")
        step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(_err_msg, flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    gc.collect()

    # ────────────────────────────────────────────
    # STEP 12: 하이퍼파라미터 튜닝
    # ────────────────────────────────────────────
    _step_name = "step12_tuning"
    _t_step = time.time()
    try:
        with ckpt.step("step12_tuning", skip_if_done=skip_done) as prog:
            try:
                prog.update(label="Optuna 튜닝 시작 (10분 타임아웃)...")
                import subprocess as _sp2
                train_end_str = args.train_end or f"{features['year_month'].max().year - 1}-12"
                result = _sp2.run(
                    [sys.executable, "-m", "scripts.tune_lgbm",
                     "--source", args.source, "--train-end", train_end_str,
                     "--timeout", "600", "--n-trials", "50"],
                    capture_output=False, cwd=str(_PROJECT_ROOT),
                )
                if result.returncode == 0:
                    prog.update(label="튜닝 완료 — weights/dsz_lgbm_tuned/")
                else:
                    prog.update(label="튜닝 실패 또는 시간 초과")
            except Exception as e:
                prog.update(label=f"튜닝 실패: {e}")
        _tune_ok = result.returncode == 0 if 'result' in locals() else False
        step_results[_step_name] = {"status": "OK" if _tune_ok else "FAIL",
                                    "elapsed": time.time() - _t_step,
                                    **({"error": "subprocess 실패"} if not _tune_ok else {})}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(_err_msg, flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    gc.collect()

    # ────────────────────────────────────────────
    # STEP 12.5: 튜닝 모델 Sliding 재평가
    # ────────────────────────────────────────────
    _step_name = "step12_5_tuned_sliding"
    _t_step = time.time()
    try:
        with ckpt.step("step12_5_tuned_sliding", skip_if_done=skip_done) as prog:
            tuned_path = Path("weights/dsz_lgbm_tuned/model.txt")
            if tuned_path.exists():
                try:
                    booster_tuned, spec_tuned = lgbm.load("weights/dsz_lgbm_tuned")
                    prog.update(label="튜닝 모델 Sliding 재평가 중...")
                    print(f"    [DEBUG] tuned spec: {len(spec_tuned.all)}개, booster: {booster_tuned.num_feature()}", flush=True)

                    # features_cache에서 로드 (튜닝 모델과 동일 피처 보장)
                    _fc = Path("checkpoints/features_cache.parquet")
                    if _fc.exists():
                        test_tuned = pd.read_parquet(_fc)
                        test_tuned["year_month"] = test_tuned["year_month"].apply(
                            lambda x: pd.Period(str(x), freq="M"))
                        print(f"    [DEBUG] features_cache 로드: {len(test_tuned)}행, {len(test_tuned.columns)}컬럼", flush=True)
                    else:
                        test_tuned = features.copy()
                        print(f"    [DEBUG] features 변수 사용: {len(test_tuned)}행, {len(test_tuned.columns)}컬럼", flush=True)
                    train_end_p = pd.Period(args.train_end, freq="M") if args.train_end else None
                    if train_end_p is None:
                        max_ym = test_tuned["year_month"].max()
                        train_end_p = pd.Period(f"{max_ym.year - 1}-12", freq="M")
                    test_tuned = test_tuned[test_tuned["year_month"] > train_end_p]
                    print(f"    [DEBUG] test rows: {len(test_tuned)}, train_end: {train_end_p}", flush=True)
                    for c in spec_tuned.categorical:
                        if c in test_tuned.columns:
                            test_tuned[c] = test_tuned[c].astype("category")
                    nan_filled = []
                    for c in spec_tuned.all:
                        if c not in test_tuned.columns:
                            test_tuned[c] = np.nan
                            nan_filled.append(c)
                    if nan_filled:
                        print(f"    [DEBUG] NaN 채운 컬럼 {len(nan_filled)}개: {nan_filled[:5]}...", flush=True)

                    # predict 테스트
                    try:
                        _test_pred = booster_tuned.predict(
                            test_tuned[spec_tuned.all], num_iteration=booster_tuned.best_iteration)
                        print(f"    [DEBUG] predict 성공: {len(_test_pred)}건", flush=True)
                    except Exception as _pe:
                        print(f"    [DEBUG] predict 실패: {_pe}", flush=True)

                    tuned_sliding = []
                    for md in METER_DAYS:
                        try:
                            h_md = ev.build_horizon_table(daily, horizons=(10, 20), meter_day=md)
                            ctx_md = ev.attach_alarm_context(h_md, monthly)
                            pred_vals = booster_tuned.predict(
                                test_tuned[spec_tuned.all], num_iteration=booster_tuned.best_iteration
                            )
                            pred_df = test_tuned[[CUSTOMER_ID, "year_month", "horizon_days"]].copy()
                            pred_df["pred_monthly_kwh"] = pred_vals
                            merged = ctx_md.merge(pred_df, on=[CUSTOMER_ID, "year_month", "horizon_days"], how="inner")
                            if md == METER_DAYS[0]:
                                print(f"    [DEBUG] md={md}: ctx {len(ctx_md)}행, pred {len(pred_df)}행, merged {len(merged)}행", flush=True)
                                print(f"    [DEBUG] ctx ym: {ctx_md['year_month'].dtype} {ctx_md['year_month'].unique()[:3]}", flush=True)
                                print(f"    [DEBUG] pred ym: {pred_df['year_month'].dtype} {pred_df['year_month'].unique()[:3]}", flush=True)
                            merged["meter_day"] = md
                            merged["error"] = merged["pred_monthly_kwh"] - merged["full_month_kwh"]
                            merged["pct_error"] = merged["error"].abs() / merged["full_month_kwh"].clip(lower=1e-9) * 100
                            merged["month"] = merged["year_month"].apply(lambda x: x.month)
                            tuned_sliding.append(merged)
                        except Exception as _e_tuned:
                            if md == METER_DAYS[0]:
                                print(f"    [tuned sliding 에러] md={md}: {_e_tuned}", flush=True)
                                import traceback as _tb
                                _tb.print_exc()

                    print(f"    [DEBUG] tuned_sliding 결과: {len(tuned_sliding)}개 검침일", flush=True)
                    if tuned_sliding:
                        from src.eval_detailed import save_detailed_evaluation
                        all_tuned = pd.concat(tuned_sliding, ignore_index=True)
                        dims_tuned = save_detailed_evaluation(all_tuned, "sliding_results_tuned")
                        ckpt.save_intermediate("tuned_sliding", dims_tuned.get("overall", pd.DataFrame()))
                except Exception as e:
                    prog.update(label=f"튜닝 모델 평가 실패: {e}")
            else:
                prog.update(label="튜닝 모델 없음 — 스킵")
        step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(_err_msg, flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    gc.collect()

    # ────────────────────────────────────────────
    # STEP 13: 스케일 아웃 지표 (최종 모델 기준)
    # ────────────────────────────────────────────
    _step_name = "step13_scale"
    _t_step = time.time()
    try:
        with ckpt.step("step13_scale", skip_if_done=skip_done) as prog:
            try:
                from src.scale import estimate_scale_metrics
                scale_metrics = estimate_scale_metrics(df)
                cur = scale_metrics["current"]
                prog.update(label=f"현재: {cur['n_customers']:,}호, {cur['memory_mb']:.0f}MB")
                for target, proj in scale_metrics["projections"].items():
                    prog.update(label=f"{target}: ~{proj['estimated_memory_gb']}GB, ~{proj['estimated_time_min']:.0f}분")
                Path("scale_report").mkdir(exist_ok=True)
                with open("scale_report/metrics.json", "w", encoding="utf-8") as f:
                    json.dump(scale_metrics, f, ensure_ascii=False, indent=2, default=str)
                ckpt.save_intermediate("scale_metrics", scale_metrics)
            except Exception as e:
                prog.update(label=f"스케일 지표 실패: {e}")
        step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(_err_msg, flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    gc.collect()

    # ────────────────────────────────────────────
    # figures용 UI 데이터 준비
    # ────────────────────────────────────────────
    try:
        ui_dir = Path("data/ui")
        ui_dir.mkdir(parents=True, exist_ok=True)

        def _period_safe(df_ui):
            """Period 컬럼을 문자열로 변환 (parquet 직렬화 호환)."""
            period_cols = [c for c in df_ui.columns
                          if str(df_ui[c].dtype).startswith("period")]
            if not period_cols:
                return df_ui
            df_out = df_ui.copy()
            for c in period_cols:
                df_out[c] = df_out[c].astype(str)
            return df_out

        if 'ctx' in locals() and isinstance(ctx, pd.DataFrame) and len(ctx) > 0:
            _period_safe(ctx).to_parquet(ui_dir / "ctx.parquet", index=False)
        if 'test_preds' in locals() and isinstance(test_preds, pd.DataFrame) and len(test_preds) > 0:
            _period_safe(test_preds).to_parquet(ui_dir / "preds.parquet", index=False)
        if 'daily' in locals() and isinstance(daily, pd.DataFrame) and len(daily) > 0:
            _period_safe(daily).to_parquet(ui_dir / "daily.parquet", index=False)
        if 'monthly' in locals() and isinstance(monthly, pd.DataFrame) and len(monthly) > 0:
            _period_safe(monthly).to_parquet(ui_dir / "monthly.parquet", index=False)
        if 'all_metrics' in locals() and isinstance(all_metrics, pd.DataFrame) and len(all_metrics) > 0:
            _period_safe(all_metrics).to_parquet(ui_dir / "metrics.parquet", index=False)
        if 'ami_df' in locals() and isinstance(ami_df, pd.DataFrame) and len(ami_df) > 0:
            tou_cols = [c for c in ami_df.columns if c.startswith("tou_")]
            if tou_cols:
                _period_safe(ami_df[["customer_id", "year_month"] + tou_cols]).to_parquet(
                    ui_dir / "tou_breakdown.parquet", index=False)
        print("[UI 데이터 저장] data/ui/ 갱신 완료", flush=True)
    except Exception as e:
        print(f"[경고] UI 데이터 저장 실패: {e}", flush=True)

    # ────────────────────────────────────────────
    # 보고서용 이미지 자동 생성
    # ────────────────────────────────────────────
    _step_name = "figures_generate"
    _t_step = time.time()
    print("\n[보고서용 이미지 생성]", flush=True)
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "scripts.generate_figures"], check=False,
                       cwd=str(_PROJECT_ROOT))
        print("  [OK] 이미지 생성 완료", flush=True)
        step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(f"  [경고] 이미지 생성 실패: {e}", flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    gc.collect()

    # ────────────────────────────────────────────
    # 반출물 수집 (export/)
    # ────────────────────────────────────────────
    _step_name = "export_collect"
    _t_step = time.time()
    print("\n[반출물 수집] export/ 폴더로 복사", flush=True)
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "scripts.export_summary", "--dashboard"], check=False,
                       cwd=str(_PROJECT_ROOT))
        # 내부용 전체 대시보드 (고객 데이터 포함)
        subprocess.run([sys.executable, "-m", "scripts.generate_dashboard",
                        "--out", "dashboard.html"], check=False, cwd=str(_PROJECT_ROOT))
        print("  [OK] 반출물 수집 완료 → export/", flush=True)
        print("  [OK] 대시보드: dashboard.html (내부용), export/dashboard_safe.html (반출용)", flush=True)
        step_results[_step_name] = {"status": "OK", "elapsed": time.time() - _t_step}
    except Exception as e:
        _elapsed = time.time() - _t_step
        _err_msg = f"[에러] {_step_name} — {type(e).__name__}: {e}"
        print(f"  [경고] 반출물 수집 실패: {e}", flush=True)
        step_results[_step_name] = {"status": "FAIL", "elapsed": _elapsed, "error": str(e)}
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write(f"{_err_msg}\n{traceback.format_exc()}\n")
    gc.collect()

    # ────────────────────────────────────────────
    # 완료 요약
    # ────────────────────────────────────────────
    pipeline_end_time = datetime.now()
    pipeline_elapsed = time.time() - pipeline_start_ts
    elapsed = time.time() - ckpt.progress.start_time if ckpt.progress.start_time else pipeline_elapsed

    n_ok = sum(1 for r in step_results.values() if r["status"] == "OK")
    n_fail = sum(1 for r in step_results.values() if r["status"] == "FAIL")
    n_skip = sum(1 for r in step_results.values() if r["status"] == "SKIP")

    print(f"""
╔══════════════════════════════════════════════════════╗
║                    전체 완료                          ║
╚══════════════════════════════════════════════════════╝

  소요 시간: {elapsed/60:.1f}분
  체크포인트: {args.checkpoint_dir}/
  결과: OK {n_ok} / FAIL {n_fail} / SKIP {n_skip}
""", flush=True)

    # Step별 소요 시간 요약
    print("=" * 60, flush=True)
    print("파이프라인 완료 요약", flush=True)
    print("=" * 60, flush=True)
    for name, result in step_results.items():
        status = result["status"]
        step_elapsed = result["elapsed"]
        if status == "OK":
            marker = "[OK]"
        elif status == "SKIP":
            marker = "[--]"
        else:
            marker = "[XX]"
        err_suffix = f"  ({result.get('error', '')})" if status == "FAIL" else ""
        print(f"  {marker} {name:30s} {step_elapsed:>7.0f}초 {status}{err_suffix}", flush=True)
    print("=" * 60, flush=True)
    print(f"  총 소요: {pipeline_elapsed:.0f}초 ({pipeline_elapsed/60:.1f}분)", flush=True)

    print(f"""
  반출 대상:
    profile_stats/dsz/       — 프로파일러 19개 통계
    preprocess_log/          — 전처리 로그
    btm_results/             — BTM 탐지 + 전략 비교
    eval_results/            — 모델 평가 (기본값)
    sliding_results/         — Sliding 상세 평가 (기본값)
    sliding_results_optimal/ — Sliding 상세 평가 (최적 조합)
    sliding_results_tuned/   — Sliding 상세 평가 (튜닝 모델)
    residual_correction/     — 잔차 보정 결과 + 패턴 분석
    ablation_results/        — Ablation Study
    feature_selection/       — 피처 선정 근거
    weights/dsz_lgbm/        — 기본 모델
    weights/dsz_lgbm_optimal/ — 최적 조합 모델
    weights/dsz_lgbm_tuned/  — 튜닝 모델
    tuning_results/          — 튜닝 이력
    explain_results/         — SHAP (PNG + CSV)
    weather_opt/             — 기상 최적화 + 민감도
    scale_report/            — 스케일 아웃 지표

  반출 불가:
    btm_results/btm_flags.csv (고객별)
    checkpoints/ (중간 결과)
""", flush=True)

    # ────────────────────────────────────────────
    # pipeline_done.txt 생성
    # ────────────────────────────────────────────
    try:
        with open("pipeline_done.txt", "w", encoding="utf-8") as f:
            f.write(f"시작: {pipeline_start_time}\n")
            f.write(f"종료: {pipeline_end_time}\n")
            f.write(f"총 소요: {pipeline_elapsed:.0f}초 ({pipeline_elapsed/60:.1f}분)\n")
            f.write(f"결과: OK {n_ok} / FAIL {n_fail} / SKIP {n_skip}\n")
            f.write("\n")
            for name, result in step_results.items():
                err_info = f" | {result.get('error', '')}" if result["status"] == "FAIL" else ""
                f.write(f"{result['status']:4s} {name} ({result['elapsed']:.0f}초){err_info}\n")
        print("[완료 알림] pipeline_done.txt 생성됨", flush=True)
    except Exception as e:
        print(f"[경고] pipeline_done.txt 생성 실패: {e}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
