"""Ablation Study — 기본값에서 하나씩 바꿔가며 최적 조합 탐색.

Phase 1: 기본값 베이스라인
Phase 2: 개별 요소 탐색 (하나만 변경)
Phase 3: 최적값 조합
Phase 4: 하이퍼파라미터 튜닝 (선택)

사용법:
  python -m scripts.ablation_study --source configs/source_dsz.yaml --train-end 2023-12
"""
from __future__ import annotations

import argparse
import io as _stdio
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import os
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import io_adapter, eval as ev
from src.preprocess import preprocess
from src.btm_detect import detect as btm_detect
from src.checkpoint import CheckpointManager
from src.result_saver import save_dataframe, save_comparison_table

OUT_DIR = Path("ablation_results")


@dataclass
class Config:
    """실험 설정 — 하나의 조합을 표현."""
    name: str = "default"
    btm_mode: str = "score"         # "exclude" | "binary" | "score"
    weather_version: str = "regional"  # "regional" | "tmy_raw" | "tmy_bias" | "tmy_forecast" | "none"
    hdd_base: float = 15.0
    cdd_base: float = 24.0
    feature_selection: bool = False  # True면 auto_select 적용
    corr_threshold: float = 0.85
    vif_threshold: float = 10.0

    def describe(self) -> str:
        return f"btm={self.btm_mode} weather={self.weather_version} base={self.hdd_base}/{self.cdd_base} feat_sel={self.feature_selection}"


def prepare_data(df_raw, btm_flags, config):
    """Config에 따라 데이터 준비."""
    df = df_raw.copy()

    # BTM 처리
    if config.btm_mode == "exclude":
        btm_ids = set(btm_flags[btm_flags["is_btm"] == 1]["customer_id"])
        df = df[~df["customer_id"].isin(btm_ids)]
    elif config.btm_mode == "binary":
        df = df.merge(btm_flags[["customer_id", "is_btm"]], on="customer_id", how="left")
        df["is_btm"] = df["is_btm"].fillna(0).astype(int)
    elif config.btm_mode == "score":
        df = df.merge(btm_flags[["customer_id", "is_btm", "btm_score"]], on="customer_id", how="left")
        df["is_btm"] = df["is_btm"].fillna(0).astype(int)
        df["btm_score"] = df["btm_score"].fillna(0.0)

    return df


def _load_weather_dir(w_dir: Path) -> pd.DataFrame | None:
    """디렉토리의 station_*.csv를 합쳐서 반환."""
    csvs = list(w_dir.glob("station_*.csv"))
    if not csvs:
        return None
    all_w = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    return all_w


def _get_cust_station_map(df_raw: pd.DataFrame) -> pd.DataFrame:
    """고객별 본부 → ASOS 관측소 매핑."""
    from src.weather.region_match import match_station
    if "region_code" in df_raw.columns:
        cust = df_raw[["customer_id", "region_code"]].drop_duplicates("customer_id")
        cust["_station_id"] = cust["region_code"].apply(
            lambda x: match_station(str(x)) if pd.notna(x) else 108)
    else:
        cust = df_raw[["customer_id"]].drop_duplicates()
        cust["_station_id"] = 108
    return cust[["customer_id", "_station_id"]]


def _join_station_weather(features, cust_map, all_weather, is_tmy=False):
    """고객별 관측소 기상을 features에 조인."""
    features = features.merge(cust_map, on="customer_id", how="left")
    features["_station_id"] = features["_station_id"].fillna(108).astype(int)

    if is_tmy:
        all_weather["month"] = all_weather["month"].astype(int)
        features["_month"] = features["year_month"].apply(lambda x: x.month)
        wcols = [c for c in all_weather.columns if c not in ("year_month", "station_id", "month")]
        # 기존 기상 컬럼 제거 (TMY로 대체하기 위해)
        existing_weather = [c for c in wcols if c in features.columns]
        if existing_weather:
            features = features.drop(columns=existing_weather, errors="ignore")
        tmy_join = all_weather.drop(columns=["year_month"], errors="ignore").rename(columns={"month": "_tmy_month"})
        features = features.merge(
            tmy_join,
            left_on=["_station_id", "_month"], right_on=["station_id", "_tmy_month"], how="left")
        features = features.drop(columns=["_station_id", "_month", "station_id", "_tmy_month"], errors="ignore")
    else:
        all_weather["year_month"] = all_weather["year_month"].apply(
            lambda x: pd.Period(str(x), freq="M"))
        wcols = [c for c in all_weather.columns if c not in ("year_month", "station_id")]
        features = features.merge(
            all_weather, left_on=["_station_id", "year_month"],
            right_on=["station_id", "year_month"], how="left")
        features = features.drop(columns=["_station_id", "station_id"], errors="ignore")
    return features


def _replace_remainder_with_tmy(features, cust_map, tmy_weather):
    """remainder 피처(hdd_remainder, cdd_remainder)만 TMY 값으로 교체.
    observed 피처는 ASOS 실측 유지.
    """
    from src.utils import hdd_cdd
    tmy_weather["month"] = tmy_weather["month"].astype(int)
    features = features.merge(cust_map, on="customer_id", how="left", suffixes=("", "_map"))
    features["_station_id"] = features["_station_id"].fillna(108).astype(int)
    features["_month"] = features["year_month"].apply(lambda x: x.month)

    # TMY의 temp_mean으로 remainder HDD/CDD 대체 (벡터화)
    tmy_lookup = tmy_weather.groupby(["station_id", "month"])["temp_mean"].first().reset_index()
    tmy_lookup.columns = ["_station_id", "_month", "_tmy_temp"]
    features = features.merge(tmy_lookup, on=["_station_id", "_month"], how="left")

    if "_tmy_temp" not in features.columns:
        raise RuntimeError("[FATAL] TMY remainder 교체 실패: _tmy_temp 컬럼 생성 안 됨")
    mask = features["_tmy_temp"].notna()
    if not mask.any():
        raise RuntimeError("[FATAL] TMY remainder 교체 실패: _tmy_temp 전부 NaN (station_id 매핑 실패?)")
    tmy_temps = features["_tmy_temp"].values
    h_arr, c_arr = hdd_cdd(tmy_temps)
    if "hdd_remainder" not in features.columns:
        raise RuntimeError("[FATAL] TMY remainder 교체 실패: hdd_remainder 컬럼 없음")
    features.loc[mask, "hdd_remainder"] = h_arr[mask]
    features.loc[mask, "cdd_remainder"] = c_arr[mask]
    print(f"  [TMY remainder] {mask.sum()}행 교체 완료", flush=True)

    features = features.drop(columns=["_station_id", "_month", "_tmy_temp"], errors="ignore")
    return features


def attach_weather(features, df_raw, config):
    """기상 피처 조인 — observed는 항상 ASOS, remainder만 방식에 따라 변경."""
    if config.weather_version == "none":
        return features

    weather_dir = Path("data/weather")
    cust_map = _get_cust_station_map(df_raw)

    if config.weather_version == "regional":
        # 전부 ASOS 실측 (observed + remainder 모두)
        asos_w = _load_weather_dir(weather_dir)
        if asos_w is not None:
            features = _join_station_weather(features, cust_map, asos_w, is_tmy=False)

    elif config.weather_version in ("tmy_raw", "tmy_bias", "tmy_forecast"):
        # 1) station 기상은 ASOS 실측 조인 (observed 피처 확보)
        asos_w = _load_weather_dir(weather_dir)
        if asos_w is not None:
            features = _join_station_weather(features, cust_map, asos_w, is_tmy=False)
        # 2) remainder HDD/CDD만 TMY 기온으로 교체
        tmy_dir = weather_dir / config.weather_version
        tmy_w = _load_weather_dir(tmy_dir)
        if tmy_w is not None:
            features = _replace_remainder_with_tmy(features, cust_map, tmy_w)
            print(f"  [TMY] remainder HDD/CDD를 {config.weather_version}로 교체", flush=True)

    elif config.weather_version == "tmy_hybrid":
        # ASOS 실측 조인 후 remainder를 forecast TMY로 교체
        asos_w = _load_weather_dir(weather_dir)
        if asos_w is not None:
            features = _join_station_weather(features, cust_map, asos_w, is_tmy=False)
        fcst_w = _load_weather_dir(weather_dir / "tmy_forecast")
        if fcst_w is not None:
            features = _replace_remainder_with_tmy(features, cust_map, fcst_w)
            print(f"  [TMY] remainder HDD/CDD를 tmy_forecast로 교체 (hybrid)", flush=True)

    return features


SLIDING_METER_DAYS = [1, 15]  # 대표 검침일만 사용 (속도 vs 정확도 트레이드오프)


def run_single_experiment(df, config, train_end):
    """단일 설정으로 LightGBM 학습 + Sliding 평가."""
    import time as _time
    _t0 = _time.time()
    from src.models.lgbm import build_features, train
    from src.eval import regression_metrics

    print(f"    [{config.name}] 피처 생성...", end=" ", flush=True)
    features, spec = build_features(df)
    print(f"[DEBUG] build_features: {len(features)}행, spec={len(spec.all)}개, has_weather={spec.has_weather}", flush=True)

    # AMI 피처 조인 (preprocess_lp에서 15분 기반 계산된 것)
    ami_path = Path("data/preprocessed/ami_features.parquet")
    if ami_path.exists():
        ami_df = pd.read_parquet(ami_path)
        ami_new = [c for c in ami_df.columns if c not in features.columns and c not in ("customer_id", "year_month")]
        if ami_new:
            features["year_month_str"] = features["year_month"].astype(str)
            ami_df["year_month_str"] = ami_df["year_month"].astype(str)
            features = features.merge(
                ami_df[["customer_id", "year_month_str"] + ami_new],
                on=["customer_id", "year_month_str"], how="left"
            )
            features = features.drop(columns=["year_month_str"], errors="ignore")
            spec.extra_numeric.extend([c for c in ami_new if c in features.columns])
            print(f"[AMI 조인] {len(ami_new)}개 피처 추가", flush=True)
        else:
            print(f"[AMI] 새 컬럼 없음 (이미 포함)", flush=True)
    else:
        print(f"[AMI] {ami_path} 없음", flush=True)

    print(f"학습...", end=" ", flush=True)

    # 기상 피처 조인
    n_before = len(features.columns)
    features = attach_weather(features, df, config)
    n_after = len(features.columns)
    hdd_rem_ok = "hdd_remainder" in features.columns and features["hdd_remainder"].notna().any()
    print(f"[DEBUG] attach_weather: {n_after-n_before}개 컬럼 추가, hdd_remainder={'OK' if hdd_rem_ok else 'MISSING'}", flush=True)

    # 기상 컬럼을 spec에 추가
    weather_cols = [c for c in features.columns if c not in spec.all
                    and c not in ("customer_id", "year_month", "horizon_days",
                                  "full_month_kwh", "partial_kwh", "remainder_kwh",
                                  "days_in_month", "days_observed", "date")
                    and features[c].dtype in ("float64", "float32", "int64")]
    if weather_cols:
        spec.extra_numeric = list(set(spec.extra_numeric + weather_cols))
        print(f"[DEBUG] spec에 기상 {len(weather_cols)}개 추가: {weather_cols[:5]}...", flush=True)

    # 피처 선택
    if config.feature_selection:
        from src.feature_selection import auto_select_features
        # 누수 방지: train+val에서만 피처 선택 (test 미포함)
        # Ablation의 train_end는 val 포함 기간이므로 그대로 사용
        trainval = features[features["year_month"] <= train_end]  # Ablation에서는 train_end가 val 끝
        result = auto_select_features(
            trainval, spec.numeric, "full_month_kwh",
            corr_threshold=config.corr_threshold,
            vif_threshold=config.vif_threshold,
        )
        for c in spec.numeric:
            if c not in result.selected:
                features[c] = np.nan

    # 학습 (1번)
    features_clean = features.dropna(subset=["full_month_kwh", "partial_kwh"])
    n_train = (features_clean["year_month"] <= train_end).sum()
    if n_train == 0:
        return None

    booster, _ = train(features_clean, spec, train_end=train_end)

    # Sliding 평가 (학습 안 함, 예측만 검침일별 반복)
    daily = ev.daily_by_customer(df)
    monthly = ev.monthly_by_customer(daily)

    test_features = features_clean[features_clean["year_month"] > train_end].copy()
    for c in spec.categorical:
        if c in test_features.columns:
            test_features[c] = test_features[c].astype("category")

    if len(test_features) == 0:
        return None

    print(f"sliding 평가...", end=" ", flush=True)
    # Sliding: 검침일별 × horizon별 분리 수집 (y_true, y_pred, customer_id)
    records_10 = []
    records_20 = []

    for md in SLIDING_METER_DAYS:
        try:
            horizon_tbl = ev.build_horizon_table(daily, horizons=(10, 20), meter_day=md)
            ctx = ev.attach_alarm_context(horizon_tbl, monthly)
            pred_vals = booster.predict(test_features[spec.all], num_iteration=booster.best_iteration)
            pred_df = test_features[["customer_id", "year_month", "horizon_days"]].copy()
            pred_df["pred_monthly_kwh"] = pred_vals

            merged = ctx.merge(pred_df, on=["customer_id", "year_month", "horizon_days"], how="inner")
            if len(merged) == 0:
                continue

            for h in [10, 20]:
                sub = merged[merged["horizon_days"] == h]
                if len(sub) == 0:
                    continue
                triples = list(zip(sub["full_month_kwh"].tolist(), sub["pred_monthly_kwh"].tolist(), sub["customer_id"].tolist()))
                if h == 10:
                    records_10.extend(triples)
                else:
                    records_20.extend(triples)
        except Exception:
            pass

    _elapsed = _time.time() - _t0
    if not records_10 and not records_20:
        print(f"데이터 부족 ({_elapsed:.0f}초)", flush=True)
        return None

    def _metrics(triples):
        if not triples:
            return {}
        yt = np.array([t[0] for t in triples])
        yp = np.array([t[1] for t in triples])
        cids = np.array([t[2] for t in triples])
        return regression_metrics(yt, yp, customer_ids=cids)

    reg_10 = _metrics(records_10)
    reg_20 = _metrics(records_20)
    reg_all = _metrics(records_10 + records_20)

    print(f"완료 ({_elapsed:.0f}초)", flush=True)
    return {
        "name": config.name,
        "config": config.describe(),
        # +10일 (메인)
        "mape_10d": float(reg_10.get("mape_pct", np.nan)),
        "nrmse_10d": float(reg_10.get("nrmse_pct", np.nan)),
        "cvrmse_med_10d": float(reg_10.get("cvrmse_median_pct", np.nan)),
        "rmse_10d": float(reg_10.get("rmse", np.nan)),
        "n_10d": int(reg_10.get("n", 0)),
        # +20일 (메인)
        "mape_20d": float(reg_20.get("mape_pct", np.nan)),
        "nrmse_20d": float(reg_20.get("nrmse_pct", np.nan)),
        "cvrmse_med_20d": float(reg_20.get("cvrmse_median_pct", np.nan)),
        "rmse_20d": float(reg_20.get("rmse", np.nan)),
        "n_20d": int(reg_20.get("n", 0)),
        # 종합 (참고)
        "mape_all": float(reg_all.get("mape_pct", np.nan)),
        "nrmse_all": float(reg_all.get("nrmse_pct", np.nan)),
        "bias_all": float(reg_all.get("bias", np.nan)),
        "std_error_all": float(reg_all.get("std_error", np.nan)),
        "n_all": int(reg_all.get("n", 0)),
        "n_sliding_days": len(SLIDING_METER_DAYS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--skip-phase4", action="store_true", help="튜닝 스킵")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager("checkpoints/ablation")
    if args.train_end:
        train_end = pd.Period(args.train_end, freq="M")
    else:
        train_end = None  # 데이터 로드 후 자동 감지

    # 필수 파일 검증
    if not Path(args.source).exists():
        print(f"[FATAL] source 파일 없음: {args.source}"); return 1

    t_total = time.time()

    print("""
╔══════════════════════════════════════════════════════╗
║              Ablation Study 시작                      ║
╚══════════════════════════════════════════════════════╝""")

    # 데이터 준비 — preprocessed parquet 우선
    print("\n[준비] 데이터 로딩 + 전처리 + BTM 탐지...")
    df_raw = io_adapter.load_smart(args.source, validate=False)
    df_raw, _ = preprocess(df_raw)
    # BTM: preprocess_lp의 15분 기반 신호 우선
    btm_sig_path = Path("data/preprocessed/btm_signals.parquet")
    if btm_sig_path.exists():
        btm_sig = pd.read_parquet(btm_sig_path)
        r_mean = btm_sig["btm_midday_ratio"].mean()
        r_std = btm_sig["btm_midday_ratio"].std()
        threshold = r_mean - 2 * r_std if r_std > 0 else 0
        btm_sig["is_btm"] = (
            (btm_sig["btm_neg_count"] > 0) | (btm_sig["btm_midday_ratio"] < threshold)
        ).astype(int)
        btm_sig["btm_score"] = 0.0
        btm_sig.loc[btm_sig["btm_neg_count"] > 0, "btm_score"] += 0.5
        btm_sig.loc[btm_sig["btm_midday_ratio"] < threshold, "btm_score"] += 0.5
        btm_flags = btm_sig[["customer_id", "is_btm", "btm_score"]]
        n_btm = int(btm_sig["is_btm"].sum())
        print(f"  [BTM] 15분 기반 신호: {n_btm}명 탐지")
    else:
        btm_result = btm_detect(df_raw)
        btm_flags = btm_result.customer_flags[["customer_id", "is_btm", "btm_score"]]
        n_btm = btm_result.summary["n_btm_detected"]
    print(f"  데이터: {len(df_raw):,}행, {df_raw['customer_id'].nunique()}호, BTM {n_btm}명")

    # train_end 자동 감지
    if train_end is None:
        if "date" in df_raw.columns:
            max_date = pd.to_datetime(df_raw["date"]).max()
        elif "ts" in df_raw.columns:
            max_date = pd.to_datetime(df_raw["ts"]).max()
        else:
            max_date = pd.Timestamp.now()
        train_end = pd.Period(f"{max_date.year - 1}-12", freq="M")
        print(f"  [자동 감지] train_end={train_end}")

    default = Config()
    all_results = []

    # ──────────────────────────────────────
    # Phase 1: 기본값 베이스라인
    # ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 1: 기본값 베이스라인")
    print(f"{'='*60}")
    df_default = prepare_data(df_raw, btm_flags, default)
    r = run_single_experiment(df_default, default, train_end)
    if r:
        all_results.append(r)
        print(f"  기준선 MAPE: {r['mape_10d']:.3f}%")
        baseline_mape = r["cvrmse_med_10d"]
    else:
        print("  기본값 실험 실패")
        return 1

    # ──────────────────────────────────────
    # Phase 2: 개별 요소 탐색
    # ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 2: 개별 요소 탐색 (하나만 변경)")
    print(f"{'='*60}")

    best_per_factor = {}

    # 2a. BTM 전략
    if n_btm > 0:
        print("\n  --- BTM 전략 ---")
        btm_best = ("score", baseline_mape)
        for mode in ["exclude", "binary"]:
            cfg = Config(name=f"btm_{mode}", btm_mode=mode)
            df_exp = prepare_data(df_raw, btm_flags, cfg)
            r = run_single_experiment(df_exp, cfg, train_end)
            if r:
                all_results.append(r)
                delta = r["mape_10d"] - baseline_mape
                print(f"    btm={mode}: +10d {r['mape_10d']:.3f}% +20d {r['mape_20d']:.3f}% (delta {delta:+.3f}%)")
                if r["cvrmse_med_10d"] < btm_best[1]:
                    btm_best = (mode, r["cvrmse_med_10d"])
        best_per_factor["btm_mode"] = btm_best[0]
        print(f"    최적: btm={btm_best[0]}")
    else:
        best_per_factor["btm_mode"] = "score"
        print("\n  --- BTM: 미발견, 스킵 ---")

    # 2b. 기상 방식 — ASOS 실측 vs TMY (전체 월 기상 대체)
    weather_dir = Path("data/weather")
    weather_versions = ["regional", "tmy_forecast", "tmy_bias", "tmy_raw"]
    available_weather = []
    for wv in weather_versions:
        if wv == "regional":
            if list(weather_dir.glob("station_*.csv")):
                available_weather.append(wv)
        else:
            if (weather_dir / wv).exists():
                available_weather.append(wv)

    tmy_candidates = [wv for wv in available_weather if wv != "regional"]
    if tmy_candidates:
        print("\n  --- 기상 방식 (TMY 중 최적 선택 — regional은 실 운영 불가) ---")
        weather_best = (tmy_candidates[0], float("inf"))
        for wv in tmy_candidates:
            cfg = Config(name=f"weather_{wv}", weather_version=wv)
            df_exp = prepare_data(df_raw, btm_flags, cfg)
            r = run_single_experiment(df_exp, cfg, train_end)
            if r:
                all_results.append(r)
                delta = r["cvrmse_med_10d"] - baseline_mape
                label = {"tmy_raw": "TMY순정", "tmy_bias": "TMY+편차보정",
                         "tmy_forecast": "TMY+예보보정"}.get(wv, wv)
                print(f"    {label}: CVRMSE_med {r['cvrmse_med_10d']:.3f}% ({delta:+.3f}%)")
                if r["cvrmse_med_10d"] < weather_best[1]:
                    weather_best = (wv, r["cvrmse_med_10d"])
        best_per_factor["weather_version"] = weather_best[0]
        print(f"    최적: weather={weather_best[0]}")
    else:
        best_per_factor["weather_version"] = "tmy_raw"
        print("\n  --- [경고] TMY 데이터 없음 — tmy_raw 기본값 설정 (DSZ에서 TMY 파일 확인 필요) ---")

    # 2c. 기준온도 — 스킵 (daily 기온 없이 HDD/CDD 기준온도 변경 효과 제한적)
    best_per_factor["hdd_base"] = 15.0
    best_per_factor["cdd_base"] = 24.0
    print("\n  --- 기준온도: 스킵 (15/24 유지) ---")

    # 2d. 피처 선택 (auto_select on/off)
    print("\n  --- 피처 선택 ---")
    cfg = Config(name="feat_selection", feature_selection=True)
    df_exp = prepare_data(df_raw, btm_flags, cfg)
    r = run_single_experiment(df_exp, cfg, train_end)
    if r:
        all_results.append(r)
        delta = r["mape_10d"] - baseline_mape
        print(f"    feat_sel=True: MAPE {r['mape_10d']:.3f}% ({delta:+.3f}%)")
        best_per_factor["feature_selection"] = r["cvrmse_med_10d"] < baseline_mape
    else:
        best_per_factor["feature_selection"] = False

    # 2e. 피처 그룹별 기여도 (그룹 하나씩 제거)
    print("\n  --- 피처 그룹별 기여도 ---")
    from src.models.lgbm import (
        _AMI_FEATURES, _SPECIAL_DAY_FEATURES, _WEATHER_FEATURES,
        _NUMERIC_FEATURES_BASE, _CALENDAR_FEATURES,
    )
    # 피처 그룹 제거를 위한 직접 실험
    from src.models.lgbm import build_features as _bf, train as _tr
    df_base = prepare_data(df_raw, btm_flags, default)
    try:
        features_base, spec_base = _bf(df_base)
        # AMI 피처 조인 (run_single_experiment과 동일하게)
        _ami_p = Path("data/preprocessed/ami_features.parquet")
        if _ami_p.exists():
            _ami_d = pd.read_parquet(_ami_p)
            _ami_n = [c for c in _ami_d.columns if c not in features_base.columns and c not in ("customer_id", "year_month")]
            if _ami_n:
                features_base["year_month_str"] = features_base["year_month"].astype(str)
                _ami_d["year_month_str"] = _ami_d["year_month"].astype(str)
                features_base = features_base.merge(
                    _ami_d[["customer_id", "year_month_str"] + _ami_n],
                    on=["customer_id", "year_month_str"], how="left")
                features_base = features_base.drop(columns=["year_month_str"], errors="ignore")
                spec_base.extra_numeric.extend([c for c in _ami_n if c in features_base.columns])
                print(f"    [AMI drop실험] {len(_ami_n)}개 피처 조인", flush=True)
        # 기상 조인
        features_base = attach_weather(features_base, df_base, default)
        _wc = [c for c in features_base.columns if c not in spec_base.all
               and c not in ("customer_id", "year_month", "horizon_days",
                             "full_month_kwh", "partial_kwh", "remainder_kwh",
                             "days_in_month", "days_observed", "date")
               and features_base[c].dtype in ("float64", "float32", "int64")]
        if _wc:
            spec_base.extra_numeric = list(set(spec_base.extra_numeric + _wc))
        features_base = features_base.dropna(subset=["full_month_kwh", "partial_kwh"])
        for c in spec_base.categorical:
            features_base[c] = features_base[c].astype("category")

        # 실제 features에 있는 컬럼 기준으로 그룹 구성
        _ami_actual = [c for c in features_base.columns
                       if c.startswith("observed_") or c == "pf_vs_prev_month"]
        _tou_actual = [c for c in features_base.columns if c.startswith("tou_")]
        feature_groups = {
            # "partial" 제외 — 핵심 피처, 제거 시 모델 무의미
            "calendar": list(_CALENDAR_FEATURES),
            "lag": [f for f in _NUMERIC_FEATURES_BASE if f not in ("partial_kwh", "partial_rate")],
            "AMI": _ami_actual,
            "weather": list(_WEATHER_FEATURES),
            "special_day": list(_SPECIAL_DAY_FEATURES),
            "TOU": _tou_actual,
            "categorical": spec_base.categorical,
        }
        group_results = []

        for group_name, group_cols in feature_groups.items():
            present = [c for c in group_cols if c in features_base.columns]
            if not present:
                continue
            features_drop = features_base.copy()

            # 카테고리 피처는 NaN 불가 → 컬럼 제거 + spec 수정
            is_cat_group = any(c in spec_base.categorical for c in present)
            if is_cat_group:
                features_drop = features_drop.drop(columns=present, errors="ignore")
                import copy
                spec_drop = copy.deepcopy(spec_base)
                spec_drop.categorical = [c for c in spec_drop.categorical if c not in present]
                spec_drop.extra_numeric = [c for c in spec_drop.extra_numeric if c not in present]
            else:
                for c in present:
                    features_drop[c] = np.nan
                spec_drop = spec_base

            n_train = (features_drop["year_month"] <= train_end).sum()
            if n_train == 0:
                continue
            try:
                booster_drop, _ = _tr(features_drop, spec_drop, train_end=train_end)

                # Sliding 평가 (Phase 1과 동일 방식)
                daily_b = ev.daily_by_customer(df_base)
                monthly_b = ev.monthly_by_customer(daily_b)

                test_drop = features_drop[features_drop["year_month"] > train_end].copy()
                for tc in spec_drop.categorical:
                    if tc in test_drop.columns:
                        test_drop[tc] = test_drop[tc].astype("category")
                if len(test_drop) == 0:
                    continue

                from src.eval import regression_metrics as _rm
                rec_10, rec_20 = [], []
                for md in SLIDING_METER_DAYS:
                    try:
                        ht = ev.build_horizon_table(daily_b, horizons=(10, 20), meter_day=md)
                        ctx_b = ev.attach_alarm_context(ht, monthly_b)
                        pv = booster_drop.predict(test_drop[spec_drop.all],
                                                  num_iteration=booster_drop.best_iteration)
                        pdf = test_drop[["customer_id", "year_month", "horizon_days"]].copy()
                        pdf["pred_monthly_kwh"] = pv
                        mg = ctx_b.merge(pdf, on=["customer_id", "year_month", "horizon_days"], how="inner")
                        if len(mg) == 0:
                            continue
                        for h in [10, 20]:
                            sub = mg[mg["horizon_days"] == h]
                            if len(sub) == 0:
                                continue
                            triples = list(zip(sub["full_month_kwh"].tolist(), sub["pred_monthly_kwh"].tolist(), sub["customer_id"].tolist()))
                            if h == 10:
                                rec_10.extend(triples)
                            else:
                                rec_20.extend(triples)
                    except Exception:
                        pass

                def _drop_metrics(triples):
                    if not triples:
                        return {}
                    yt = np.array([t[0] for t in triples])
                    yp = np.array([t[1] for t in triples])
                    cids = np.array([t[2] for t in triples])
                    return _rm(yt, yp, customer_ids=cids)

                reg_10_d = _drop_metrics(rec_10)
                reg_20_d = _drop_metrics(rec_20)
                if reg_10_d:
                    mape_drop = float(reg_10_d.get("mape_pct", np.nan))
                    cvmed_drop = float(reg_10_d.get("cvrmse_median_pct", np.nan))
                    contribution = cvmed_drop - baseline_mape
                    print(f"    {group_name:12s} 제거: MAPE {mape_drop:.3f}% CVRMSE_med {cvmed_drop:.3f}% (기여: {contribution:+.3f}%)")
                    group_results.append({
                        "group": group_name, "features_removed": present,
                        "mape_without": mape_drop, "cvrmse_med_without": cvmed_drop,
                        "contribution": contribution,
                    })
                    all_results.append({
                        "name": f"drop_{group_name}", "config": f"remove {group_name}",
                        "mape_10d": mape_drop,
                        "nrmse_10d": float(reg_10_d.get("nrmse_pct", np.nan)),
                        "cvrmse_med_10d": cvmed_drop,
                        "mape_20d": float(reg_20_d.get("mape_pct", np.nan)),
                        "nrmse_20d": float(reg_20_d.get("nrmse_pct", np.nan)),
                        "cvrmse_med_20d": float(reg_20_d.get("cvrmse_median_pct", np.nan)),
                        "n_10d": int(reg_10_d.get("n", 0)),
                        "n_20d": int(reg_20_d.get("n", 0)),
                    })
            except Exception as e:
                print(f"    {group_name:12s}: 실패 ({e})")
    except Exception as e:
        print(f"    피처 그룹 실험 실패: {e}")

    exclude_features = []
    if group_results:
        gr_df = pd.DataFrame(group_results).sort_values("contribution", ascending=False)
        Path("ablation_results").mkdir(exist_ok=True)
        gr_df.to_csv("ablation_results/feature_group_contribution.csv", index=False, encoding="utf-8-sig")
        print(f"\n    피처 그룹 기여도 순위:")
        for _, row in gr_df.iterrows():
            marker = "제거" if row["contribution"] <= 0 else ""
            print(f"      {row['group']:12s}: {row['contribution']:+.3f}%p {marker}")
            if row["contribution"] <= 0:
                exclude_features.extend(row["features_removed"])
        if exclude_features:
            print(f"\n    기여도 ≤ 0 제거 대상: {exclude_features}")

    # 2f. SHAP 0% 피처 정리 — lgbm.py 피처 리스트에서 이미 제거됨
    print("\n  --- SHAP 0% 피처 ---")
    try:
        _total_feats = len(spec_base.all)  # type: ignore[possibly-undefined]
        print(f"    현재 피처 수: {_total_feats}개 (SHAP 0% 피처는 lgbm.py에서 사전 제거됨)")
    except NameError:
        print("    피처 수 확인 불가 (spec_base 미생성)")

    # ──────────────────────────────────────
    # Phase 3: 최적값 조합
    # ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 3: 최적값 조합")
    print(f"{'='*60}")
    optimal = Config(
        name="optimal_combined",
        btm_mode=best_per_factor["btm_mode"],
        weather_version=best_per_factor["weather_version"],
        hdd_base=best_per_factor["hdd_base"],
        cdd_base=best_per_factor["cdd_base"],
        feature_selection=best_per_factor["feature_selection"],
    )
    print(f"  설정: {optimal.describe()}")
    df_opt = prepare_data(df_raw, btm_flags, optimal)
    r_opt = run_single_experiment(df_opt, optimal, train_end)
    if r_opt:
        all_results.append(r_opt)
        improvement = baseline_mape - r_opt["cvrmse_med_10d"]
        print(f"  최적 조합 MAPE: {r_opt['mape_10d']:.3f}%")
        print(f"  기본 대비 개선: {improvement:+.3f}% ({improvement/baseline_mape*100:.1f}%)")

    # 결과 저장
    results_df = pd.DataFrame(all_results).sort_values("cvrmse_med_10d")
    save_dataframe(results_df, OUT_DIR, "ablation_results", "Ablation Study 전체 결과")

    with open(OUT_DIR / "best_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "baseline_mape": baseline_mape,
            "optimal_mape": r_opt["cvrmse_med_10d"] if r_opt else None,
            "best_per_factor": best_per_factor,
            "optimal_config": optimal.describe(),
            "exclude_features": exclude_features,
        }, f, ensure_ascii=False, indent=2, default=str)

    # 요약 출력
    elapsed = time.time() - t_total
    print(f"""
╔══════════════════════════════════════════════════════╗
║              Ablation Study 완료                      ║
╚══════════════════════════════════════════════════════╝

  소요: {elapsed/60:.1f}분, 실험 {len(all_results)}회

  요소별 최적:
    BTM:      {best_per_factor['btm_mode']}
    기상:     {best_per_factor['weather_version']}
    기준온도: {best_per_factor['hdd_base']}/{best_per_factor['cdd_base']}
    피처선택: {best_per_factor['feature_selection']}

  기본값 MAPE:  {baseline_mape:.3f}%
  최적조합 MAPE: {r_opt['mape_10d']:.3f}% ({baseline_mape - r_opt['mape_10d']:+.3f}%)

  다음: python -m scripts.tune_lgbm (Phase 4 튜닝)

  결과: {OUT_DIR}/
""")

    # Phase 3 최적 설정을 YAML로 저장 (튜닝 스크립트에서 사용)
    with open(OUT_DIR / "optimal_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "btm_mode": optimal.btm_mode,
            "weather_version": optimal.weather_version,
            "hdd_base": optimal.hdd_base,
            "cdd_base": optimal.cdd_base,
            "feature_selection": optimal.feature_selection,
        }, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
