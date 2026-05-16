"""LightGBM 하이퍼파라미터 튜닝 — Optuna 기반.

최적 모델(BTM 전략 + 피처 선택 확정 후)에 대해 실행.
중단해도 그 시점까지의 최적 파라미터를 사용 가능.

사용법:
  python -m scripts.tune_lgbm --source configs/source_dsz.yaml --train-end 2023-12 --timeout 600
  python -m scripts.tune_lgbm --source configs/source_dsz.yaml --train-end 2023-12 --n-trials 50

출력:
  tuning_results/
    best_params.json     — 최적 파라미터
    study_history.csv    — 전체 탐색 이력
    importance.csv       — 파라미터 중요도
"""
from __future__ import annotations

import argparse
import io as _stdio
import json
import sys
import time
from pathlib import Path

import os
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import io_adapter
from src.preprocess import preprocess
from src.models.lgbm import build_features, FeatureSpec

OUT_DIR = Path("tuning_results")


def objective(trial, X_tr, y_tr, X_val, y_val, spec, cid_val=None, pl_val=None, actual_val=None):
    """Optuna objective — 고객별 CVRMSE median 최소화 (잔차 학습)."""
    params = {
        "objective": "regression",
        "metric": "mape",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
    }

    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=spec.categorical)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr, categorical_feature=spec.categorical)

    booster = lgb.train(
        params, dtr,
        num_boost_round=500,
        valid_sets=[dval],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    residual_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    # 잔차 → 최종 예측 = partial_linear + 잔차
    if pl_val is not None and actual_val is not None:
        final_pred = pl_val + residual_pred
        actual = actual_val
    else:
        final_pred = residual_pred
        actual = y_val.values
    mask = actual > 100
    if mask.sum() < 10:
        return float("inf")
    # 고객별 CVRMSE median
    if cid_val is not None:
        cids = cid_val[mask]
        unique_cids = np.unique(cids)
        cust_cvrmses = []
        for cid in unique_cids:
            cmask = cids == cid
            c_actual = actual[mask][cmask]
            c_pred = final_pred[mask][cmask]
            if len(c_actual) >= 2 and np.mean(c_actual) > 0:
                c_rmse = np.sqrt(np.mean((c_pred - c_actual)**2))
                cust_cvrmses.append(c_rmse / np.mean(c_actual) * 100)
        cvrmse = float(np.median(cust_cvrmses)) if cust_cvrmses else float("inf")
    else:
        err = final_pred[mask] - actual[mask]
        cvrmse = np.sqrt(np.mean(err**2)) / np.mean(actual[mask]) * 100

    # best_iteration 저장
    trial.set_user_attr("best_iteration", booster.best_iteration)
    trial.set_user_attr("num_trees", booster.num_trees())

    # 진행 상황 출력
    n = trial.number + 1
    try:
        best_so_far = min(cvrmse, trial.study.best_value)
    except ValueError:
        best_so_far = cvrmse
    print(f"  #{n:>3d}  CVRMSE {cvrmse:.3f}%  trees {booster.best_iteration}  best {best_so_far:.3f}%")

    return cvrmse


def main() -> int:
    parser = argparse.ArgumentParser(description="LightGBM 하이퍼파라미터 튜닝")
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--timeout", type=int, default=600, help="최대 시간 (초, 기본 600=10분)")
    parser.add_argument("--n-trials", type=int, default=150, help="최대 시도 횟수")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 피처 로드 — run_full_analysis에서 저장한 캐시 우선
    features_cache = Path("checkpoints/features_cache.parquet")
    if features_cache.exists():
        print("[load] features_cache.parquet → 즉시 로드")
        features = pd.read_parquet(features_cache)
        features["year_month"] = features["year_month"].apply(lambda x: pd.Period(str(x), freq="M"))
        # spec 복원: 피처 컬럼에서 직접 구성
        non_feature = {"customer_id", "year_month", "horizon_days", "full_month_kwh",
                       "partial_kwh", "remainder_kwh", "days_in_month", "days_observed", "date",
                       "n_households", "kwh_per_household"}
        all_cols = [c for c in features.columns if c not in non_feature]
        cat_cols = [c for c in all_cols if features[c].dtype == "object" or features[c].dtype.name == "category"]
        num_cols = [c for c in all_cols if c not in cat_cols]
        from src.models.lgbm import FeatureSpec
        # 중복 방지: base spec의 numeric과 겹치는 컬럼 제외
        _base_spec = FeatureSpec(has_weather=False, has_ami=False, has_special_days=False)
        _base_cols = set(_base_spec.numeric + _base_spec.categorical)
        extra_only = [c for c in num_cols if c not in _base_cols]
        spec = FeatureSpec(has_weather=False, has_ami=False, has_special_days=False,
                          extra_numeric=extra_only, categorical=cat_cols if cat_cols else ["contract_type"])
        print(f"  features: {len(features)}행, {len(spec.all)}개 피처")
    else:
        print("[load] 캐시 없음 → 원본에서 빌드")
        df = io_adapter.load_smart(args.source, validate=False)
        df, _ = preprocess(df)
        print("[features]")
        features, spec = build_features(df)

    features = features.dropna(subset=["full_month_kwh", "partial_kwh"]).copy()

    # AMI 피처 조인
    ami_path = Path("data/preprocessed/ami_features.parquet")
    if ami_path.exists():
        ami_df = pd.read_parquet(ami_path)
        ami_new = [c for c in ami_df.columns if c not in features.columns and c not in ("customer_id", "year_month")]
        if ami_new:
            features["year_month_str"] = features["year_month"].astype(str)
            ami_df["year_month_str"] = ami_df["year_month"].astype(str)
            features = features.merge(
                ami_df[["customer_id", "year_month_str"] + ami_new],
                on=["customer_id", "year_month_str"], how="left")
            features = features.drop(columns=["year_month_str"], errors="ignore")
            spec.extra_numeric.extend([c for c in ami_new if c in features.columns])
            print(f"  [AMI] {len(ami_new)}개 피처 조인")

    for c in spec.categorical:
        features[c] = features[c].astype("category")

    # ablation 결과에서 최적 config 로드
    from src.utils import load_best_config, apply_tmy_remainder
    _best_cfg = load_best_config()
    if _best_cfg:
        _exclude = _best_cfg.get("exclude_features", [])
        if _exclude:
            _excluded_actual = [c for c in _exclude if c in spec.extra_numeric]
            spec.extra_numeric = [c for c in spec.extra_numeric if c not in _exclude]
            print(f"  [피처 제거] ablation 기여도 ≤ 0: {_excluded_actual}")
        _wv = _best_cfg.get("best_per_factor", {}).get("weather_version", "regional")
        features = apply_tmy_remainder(features, _wv)

    # 디버그: 핵심 피처 존재 확인
    _debug_cols = ["hdd_remainder", "cdd_remainder", "hdd_observed", "cdd_observed", "observed_power_factor"]
    _present = [c for c in _debug_cols if c in spec.all and c in features.columns and features[c].notna().any()]
    _missing = [c for c in _debug_cols if c not in spec.all]
    print(f"  [DEBUG] 핵심 피처: 있음={_present}, 없음={_missing}")
    print(f"  [DEBUG] spec.all: {len(spec.all)}개")

    train_end = pd.Period(args.train_end, freq="M") if args.train_end else None
    if train_end is None:
        max_ym = features["year_month"].max()
        train_end = pd.Period(f"{max_ym.year - 1}-12", freq="M")

    train_mask = features["year_month"] <= train_end
    val_mask = ~train_mask

    # partial_linear 베이스라인 → 잔차 학습
    pl_pred_all = (
        features["partial_kwh"] / features["days_observed"].clip(lower=1) * features["days_in_month"]
    )
    y_residual = features["full_month_kwh"] - pl_pred_all

    X_tr = features.loc[train_mask, spec.all]
    y_tr = y_residual.loc[train_mask]
    X_val = features.loc[val_mask, spec.all]
    y_val = y_residual.loc[val_mask]
    cid_val = features.loc[val_mask, "customer_id"].values
    pl_val = pl_pred_all.loc[val_mask].values
    actual_val = features.loc[val_mask, "full_month_kwh"].values

    print(f"  train={len(X_tr):,}  val={len(X_val):,}  features={len(spec.all)}  (잔차 학습)")

    # 기본값 CVRMSE 측정
    print("\n[baseline]")
    default_params = {
        "objective": "regression", "metric": "mae",
        "learning_rate": 0.05, "num_leaves": 31,
        "min_data_in_leaf": 40, "verbose": -1,
    }
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=spec.categorical)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr, categorical_feature=spec.categorical)
    base_booster = lgb.train(
        default_params, dtr, num_boost_round=400,
        valid_sets=[dval], valid_names=["valid"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    base_residual = base_booster.predict(X_val, num_iteration=base_booster.best_iteration)
    base_final = pl_val + base_residual
    base_actual = actual_val
    base_mask = base_actual > 100
    if base_mask.sum() > 0 and cid_val is not None:
        cids = cid_val[base_mask]
        cust_cv = []
        for cid in np.unique(cids):
            cm = cids == cid
            ca = base_actual[base_mask][cm]
            cp = base_final[base_mask][cm]
            if len(ca) >= 2 and np.mean(ca) > 0:
                cust_cv.append(np.sqrt(np.mean((cp - ca)**2)) / np.mean(ca) * 100)
        base_cvrmse = float(np.median(cust_cv)) if cust_cv else float("inf")
    elif base_mask.sum() > 0:
        base_err = base_pred[base_mask] - base_actual[base_mask]
        base_cvrmse = np.sqrt(np.mean(base_err**2)) / np.mean(base_actual[base_mask]) * 100
    else:
        base_cvrmse = float("inf")
    print(f"  기본값 CVRMSE median: {base_cvrmse:.4f}%")

    # Optuna 튜닝
    print(f"\n[tune] timeout={args.timeout}s, max_trials={args.n_trials}")
    print(f"  중단(Ctrl+C)해도 그 시점 최적값 저장됨\n")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize", study_name="lgbm_tune",
        storage=None,  # in-memory (SQLite 잠금 방지)
    )
    t0 = time.time()

    def _early_stop_callback(study, trial):
        if trial.number >= 30 and trial.number - study.best_trial.number >= 25:
            print(f"\n  [조기 종료] 25 trial 연속 best 미갱신 → 중단")
            study.stop()

    try:
        study.optimize(
            lambda trial: objective(trial, X_tr, y_tr, X_val, y_val, spec,
                                   cid_val=cid_val, pl_val=pl_val, actual_val=actual_val),
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True,
            callbacks=[_early_stop_callback],
        )
    except KeyboardInterrupt:
        print("\n  [중단됨] 현재까지 최적값 저장 중...")

    elapsed = time.time() - t0
    n_completed = len(study.trials)
    best = study.best_trial

    print(f"\n{'='*60}")
    print(f"튜닝 완료: {n_completed}회 시도, {elapsed:.0f}초")
    print(f"{'='*60}")
    print(f"  기본값 CVRMSE:  {base_cvrmse:.4f}%")
    print(f"  최적값 CVRMSE:  {best.value:.4f}%")
    print(f"  개선:           {base_cvrmse - best.value:.4f}% ({(base_cvrmse - best.value)/base_cvrmse*100:.1f}%)")
    print(f"\n  최적 파라미터:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # 저장
    result = {
        "baseline_cvrmse": round(base_cvrmse, 4),
        "best_cvrmse": round(best.value, 4),
        "improvement_pct": round((base_cvrmse - best.value) / base_cvrmse * 100, 2),
        "n_trials": n_completed,
        "elapsed_sec": round(elapsed, 1),
        "best_params": best.params,
        "best_iteration": best.user_attrs.get("best_iteration"),
    }
    with open(OUT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 탐색 이력
    history = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = {"trial": t.number, "cvrmse": t.value}
            row.update(t.params)
            history.append(row)
    pd.DataFrame(history).to_csv(OUT_DIR / "study_history.csv", index=False, encoding="utf-8-sig")

    # 파라미터 중요도
    try:
        importance = optuna.importance.get_param_importances(study)
        imp_df = pd.DataFrame([{"param": k, "importance": v} for k, v in importance.items()])
        imp_df.to_csv(OUT_DIR / "param_importance.csv", index=False, encoding="utf-8-sig")
        print(f"\n  파라미터 중요도:")
        for k, v in importance.items():
            print(f"    {k}: {v:.4f}")
    except Exception:
        pass

    # 최적 파라미터로 최종 모델 학습 + 저장
    print(f"\n[retrain] 최적 파라미터로 최종 모델 학습...")
    final_params = {**default_params, **best.params}
    final_params["feature_pre_filter"] = False
    dtr2 = lgb.Dataset(X_tr, label=y_tr, categorical_feature=spec.categorical,
                       free_raw_data=False, params={"feature_pre_filter": False})
    dval2 = lgb.Dataset(X_val, label=y_val, reference=dtr2, categorical_feature=spec.categorical,
                        free_raw_data=False)
    final_booster = lgb.train(
        final_params, dtr2,
        num_boost_round=500,
        valid_sets=[dval2], valid_names=["valid"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    from src.models.lgbm import save
    save(final_booster, spec, "weights/dsz_lgbm_tuned",
         notes=f"tuned: {n_completed} trials, CVRMSE {best.value:.4f}%")
    print(f"  저장: weights/dsz_lgbm_tuned/")

    print(f"\n[done] {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
