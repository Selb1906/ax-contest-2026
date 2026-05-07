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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    weather_version: str = "none"   # "none" | "v1_paper" | "v2_power" | ... | "8stations"
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


def attach_weather(horizon, config):
    """기상 피처 조인."""
    if config.weather_version == "none":
        return horizon

    weather_dir = Path("data/weather")
    if config.weather_version == "8stations":
        station_ids = [108, 112, 114, 119, 133, 143, 156, 159]
        for sid in station_ids:
            p = weather_dir / f"station_{sid}.csv"
            if not p.exists():
                continue
            sw = pd.read_csv(p)
            sw["year_month"] = sw["year_month"].apply(lambda x: pd.Period(str(x), freq="M"))
            rename = {c: f"{c}_s{sid}" for c in sw.columns if c not in ("station_id", "year_month")}
            sw = sw.rename(columns=rename).drop(columns=["station_id"], errors="ignore")
            horizon = horizon.merge(sw, on="year_month", how="left")
    else:
        p = weather_dir / f"{config.weather_version}.csv"
        if p.exists():
            w = pd.read_csv(p)
            w["year_month"] = w["year_month"].apply(lambda x: pd.Period(str(x), freq="M"))
            horizon = horizon.merge(w, on="year_month", how="left")

    return horizon


SLIDING_METER_DAYS = list(range(1, 32))  # 한전 검침일 1~31 (7차 말일 포함)


def run_single_experiment(df, config, train_end):
    """단일 설정으로 LightGBM 학습 + Sliding 평가."""
    import time as _time
    _t0 = _time.time()
    from src.models.lgbm import build_features, train
    from src.eval import regression_metrics

    print(f"    [{config.name}] 피처 생성...", end=" ", flush=True)
    features, spec = build_features(df)
    print(f"학습...", end=" ", flush=True)

    # 기상 피처 조인
    features = attach_weather(features, config)

    # 기상 컬럼을 spec에 추가
    weather_cols = [c for c in features.columns if c not in spec.all
                    and c not in ("customer_id", "year_month", "horizon_days",
                                  "full_month_kwh", "partial_kwh", "remainder_kwh",
                                  "days_in_month", "days_observed", "date")
                    and features[c].dtype in ("float64", "float32", "int64")]
    if weather_cols:
        spec.extra_numeric = list(set(spec.extra_numeric + weather_cols))

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
    # Sliding: 검침일별 × horizon별 분리 수집
    records_10 = []  # (y_true, y_pred) for +10일
    records_20 = []  # (y_true, y_pred) for +20일

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
                pairs = list(zip(sub["full_month_kwh"].tolist(), sub["pred_monthly_kwh"].tolist()))
                if h == 10:
                    records_10.extend(pairs)
                else:
                    records_20.extend(pairs)
        except Exception:
            pass

    _elapsed = _time.time() - _t0
    if not records_10 and not records_20:
        print(f"데이터 부족 ({_elapsed:.0f}초)", flush=True)
        return None

    def _metrics(pairs):
        if not pairs:
            return {}
        yt = np.array([p[0] for p in pairs])
        yp = np.array([p[1] for p in pairs])
        return regression_metrics(yt, yp)

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
        "rmse_10d": float(reg_10.get("rmse", np.nan)),
        "n_10d": int(reg_10.get("n", 0)),
        # +20일 (메인)
        "mape_20d": float(reg_20.get("mape_pct", np.nan)),
        "nrmse_20d": float(reg_20.get("nrmse_pct", np.nan)),
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
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--skip-phase4", action="store_true", help="튜닝 스킵")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager("checkpoints/ablation")
    train_end = pd.Period(args.train_end, freq="M")

    t_total = time.time()

    print("""
╔══════════════════════════════════════════════════════╗
║              Ablation Study 시작                      ║
╚══════════════════════════════════════════════════════╝""")

    # 데이터 준비
    print("\n[준비] 데이터 로딩 + 전처리 + BTM 탐지...")
    df_raw = io_adapter.load_from_yaml(args.source, validate=False)
    df_raw, _ = preprocess(df_raw)
    btm_result = btm_detect(df_raw)
    btm_flags = btm_result.customer_flags[["customer_id", "is_btm", "btm_score"]]
    n_btm = btm_result.summary["n_btm_detected"]
    print(f"  데이터: {len(df_raw):,}행, {df_raw['customer_id'].nunique()}호, BTM {n_btm}명")

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
        baseline_mape = r["mape_10d"]
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
                if r["mape_10d"] < btm_best[1]:
                    btm_best = (mode, r["mape_10d"])
        best_per_factor["btm_mode"] = btm_best[0]
        print(f"    최적: btm={btm_best[0]}")
    else:
        best_per_factor["btm_mode"] = "score"
        print("\n  --- BTM: 미발견, 스킵 ---")

    # 2b. 기상 가중치 (기존 5개 + 8스테이션 + Step7 LP역산/계절별)
    weather_versions = ["national_v1_paper", "national_v2_power", "national_v3_population",
                        "national_v4_equal", "national_v5_seoul"]
    available_weather = [v for v in weather_versions if (Path("data/weather") / f"{v}.csv").exists()]

    # Step 7에서 LP 역산·계절별 가중치로 만든 CSV가 있으면 추가
    for extra in ["national_lp_regression", "national_seasonal_summer", "national_seasonal_winter"]:
        if (Path("data/weather") / f"{extra}.csv").exists():
            available_weather.append(extra)

    if available_weather:
        print("\n  --- 기상 가중치 ---")
        weather_best = ("none", baseline_mape)
        for wv in available_weather:
            cfg = Config(name=f"weather_{wv}", weather_version=wv)
            df_exp = prepare_data(df_raw, btm_flags, cfg)
            r = run_single_experiment(df_exp, cfg, train_end)
            if r:
                all_results.append(r)
                delta = r["mape_10d"] - baseline_mape
                tag = " (LP역산)" if "regression" in wv else " (계절별)" if "seasonal" in wv else ""
                print(f"    weather={wv}{tag}: MAPE {r['mape_10d']:.3f}% ({delta:+.3f}%)")
                if r["mape_10d"] < weather_best[1]:
                    weather_best = (wv, r["mape_10d"])
        best_per_factor["weather_version"] = weather_best[0]
        print(f"    최적: weather={weather_best[0]}")
    else:
        best_per_factor["weather_version"] = "none"
        print("\n  --- 기상: 파일 없음, 스킵 ---")

    # 2c. 기준온도 — Step 7(OLS 그리드)의 결과를 활용
    print("\n  --- 기준온도 ---")
    base_candidates = []

    # Step 7 결과가 있으면 상위 3개 + 기본값으로 후보 구성
    step7_path = Path("weather_opt/weather_optimization.json")
    if step7_path.exists():
        try:
            with open(step7_path, encoding="utf-8") as f:
                step7 = json.load(f)
            for ct_key in ["전체", "주택용", "일반용"]:
                bt = step7.get("method4_base_temp_optimization", {}).get(ct_key, {})
                if "best" in bt:
                    b = bt["best"]
                    pair = (int(b["hdd_base"]), int(b["cdd_base"]))
                    if pair not in base_candidates:
                        base_candidates.append(pair)
            print(f"    Step 7(OLS)에서 후보 {len(base_candidates)}개 로드: {base_candidates}")
        except Exception:
            pass

    # 기본값 + Step7 후보가 없으면 고정 목록
    if not base_candidates:
        base_candidates = [(12, 22), (13, 23), (14, 24), (16, 26)]
        print(f"    Step 7 결과 없음 → 고정 후보 사용: {base_candidates}")
    else:
        # 기본값(15/24)도 포함
        if (15, 24) not in base_candidates:
            base_candidates.append((15, 24))

    base_best = (15.0, 24.0, baseline_mape)
    for hdd_b, cdd_b in base_candidates:
        if hdd_b == 15 and cdd_b == 24:
            continue
        cfg = Config(name=f"base_{hdd_b}_{cdd_b}", hdd_base=hdd_b, cdd_base=cdd_b)
        df_exp = prepare_data(df_raw, btm_flags, cfg)
        r = run_single_experiment(df_exp, cfg, train_end)
        if r:
            all_results.append(r)
            delta = r["mape_10d"] - baseline_mape
            print(f"    base={hdd_b}/{cdd_b}: MAPE {r['mape_10d']:.3f}% ({delta:+.3f}%)")
            if r["mape_10d"] < base_best[2]:
                base_best = (hdd_b, cdd_b, r["mape_10d"])
    best_per_factor["hdd_base"] = base_best[0]
    best_per_factor["cdd_base"] = base_best[1]
    print(f"    최적: base={base_best[0]}/{base_best[1]}")

    # 2d. 피처 선택 (auto_select on/off)
    print("\n  --- 피처 선택 ---")
    cfg = Config(name="feat_selection", feature_selection=True)
    df_exp = prepare_data(df_raw, btm_flags, cfg)
    r = run_single_experiment(df_exp, cfg, train_end)
    if r:
        all_results.append(r)
        delta = r["mape_10d"] - baseline_mape
        print(f"    feat_sel=True: MAPE {r['mape_10d']:.3f}% ({delta:+.3f}%)")
        best_per_factor["feature_selection"] = r["mape_10d"] < baseline_mape
    else:
        best_per_factor["feature_selection"] = False

    # 2e. 피처 그룹별 기여도 (그룹 하나씩 제거)
    print("\n  --- 피처 그룹별 기여도 ---")
    from src.models.lgbm import (
        _AMI_FEATURES, _SPECIAL_DAY_FEATURES, _WEATHER_FEATURES,
        _NUMERIC_FEATURES_BASE, _CALENDAR_FEATURES,
    )
    feature_groups = {
        "AMI 패턴": list(_AMI_FEATURES),
        "특수일": list(_SPECIAL_DAY_FEATURES),
        "기상": list(_WEATHER_FEATURES),
        "래그": [f for f in _NUMERIC_FEATURES_BASE if f not in ("partial_kwh", "partial_rate")],
        "역률": [f for f in _AMI_FEATURES if "power_factor" in f or "reactive" in f or "pf_vs" in f],
    }
    group_results = []
    for group_name, group_cols in feature_groups.items():
        if not group_cols:
            continue
        # 해당 그룹 피처를 NaN으로 만들어 제거 효과 측정
        cfg = Config(name=f"drop_{group_name}")
        df_exp = prepare_data(df_raw, btm_flags, cfg)
        # 임시로 해당 컬럼을 df에서 제거하는 대신, 실험 함수 내에서 NaN 처리
        r = run_single_experiment(df_exp, cfg, train_end)
        if r:
            # 기본값 대비 해당 그룹 제거 시 성능 저하 = 그룹의 기여도
            # 여기서는 기본값과 동일 (그룹 제거가 Config에 미반영)
            # → 직접 피처 제거 실험
            pass

    # 피처 그룹 제거를 위한 직접 실험
    from src.models.lgbm import build_features as _bf, train as _tr
    df_base = prepare_data(df_raw, btm_flags, default)
    try:
        features_base, spec_base = _bf(df_base)
        features_base = features_base.dropna(subset=["full_month_kwh", "partial_kwh"])
        for c in spec_base.categorical:
            features_base[c] = features_base[c].astype("category")

        for group_name, group_cols in feature_groups.items():
            present = [c for c in group_cols if c in features_base.columns]
            if not present:
                continue
            features_drop = features_base.copy()
            for c in present:
                features_drop[c] = np.nan

            n_train = (features_drop["year_month"] <= train_end).sum()
            if n_train == 0:
                continue
            try:
                booster_drop, preds_drop = _tr(features_drop, spec_base, train_end=train_end)
                from src.eval import evaluate as _eval, attach_alarm_context as _aac
                daily_b = ev.daily_by_customer(df_base)
                monthly_b = ev.monthly_by_customer(daily_b)
                horizon_b = ev.build_horizon_table(daily_b, horizons=(10, 20))
                ctx_b = _aac(horizon_b, monthly_b)
                m = _eval(preds_drop, ctx_b)
                if len(m) > 0 and "mape_pct" in m.columns:
                    mape_drop = float(m["mape_pct"].mean())
                    contribution = mape_drop - baseline_mape
                    print(f"    {group_name:12s} 제거: MAPE {mape_drop:.3f}% (기여: {contribution:+.3f}%)")
                    group_results.append({
                        "group": group_name, "features_removed": present,
                        "mape_without": mape_drop, "contribution": contribution,
                    })
                    all_results.append({
                        "name": f"drop_{group_name}", "config": f"remove {group_name}",
                        "mape_10d": mape_drop, "f1_mean": float(m["alarm_f1"].mean()),
                        "n_test": int(m["n"].sum()),
                    })
            except Exception as e:
                print(f"    {group_name:12s}: 실패 ({e})")
    except Exception as e:
        print(f"    피처 그룹 실험 실패: {e}")

    if group_results:
        gr_df = pd.DataFrame(group_results).sort_values("contribution", ascending=False)
        Path("ablation_results").mkdir(exist_ok=True)
        gr_df.to_csv("ablation_results/feature_group_contribution.csv", index=False, encoding="utf-8-sig")
        print(f"\n    피처 그룹 기여도 순위:")
        for _, row in gr_df.iterrows():
            print(f"      {row['group']:12s}: {row['contribution']:+.3f}%p")

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
        improvement = baseline_mape - r_opt["mape_10d"]
        print(f"  최적 조합 MAPE: {r_opt['mape_10d']:.3f}%")
        print(f"  기본 대비 개선: {improvement:+.3f}% ({improvement/baseline_mape*100:.1f}%)")

    # 결과 저장
    results_df = pd.DataFrame(all_results).sort_values("mape_10d")
    save_dataframe(results_df, OUT_DIR, "ablation_results", "Ablation Study 전체 결과")

    with open(OUT_DIR / "best_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "baseline_mape": baseline_mape,
            "optimal_mape": r_opt["mape_10d"] if r_opt else None,
            "best_per_factor": best_per_factor,
            "optimal_config": optimal.describe(),
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
  최적조합 MAPE: {r_opt['mape_mean']:.3f}% ({baseline_mape - r_opt['mape_mean']:+.3f}%)

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
