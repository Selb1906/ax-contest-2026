"""안심구역 원스톱 분석 — 체크포인트 + 진행률 + 이중 저장.

중단해도 이어서 실행 가능. 모든 결과는 CSV+PNG+JSON 이중 저장.

사용법:
  python -m scripts.run_full_analysis --source configs/source_dsz.yaml --train-end 2023-12
  python -m scripts.run_full_analysis --source configs/source_dsz.yaml --resume  (이어서 실행)
"""
from __future__ import annotations

import argparse
import io as _stdio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

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
from src.checkpoint import CheckpointManager
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
        METER_DAYS = list(range(1, 32))

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
                )
                if ret.returncode != 0:
                    print(f"  [경고] 전처리 실패 — 원본 CSV로 로딩 시도")

    # ────────────────────────────────────────────
    # STEP 1: 데이터 로딩
    # ────────────────────────────────────────────
    with ckpt.step("step1_load", skip_if_done=False) as prog:
        if daily_parquet.exists():
            t_load = time.time()
            print("  daily.parquet 로딩...", end=" ", flush=True)
            df = pd.read_parquet(daily_parquet)
            print(f"OK ({time.time()-t_load:.1f}초, {len(df):,}행)")
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
            prog.update(label=f"전처리 parquet 로드 완료")
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
    # STEP 1.5: 전처리
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
    # STEP 2: BTM 탐지
    # ────────────────────────────────────────────
    with ckpt.step("step2_btm", skip_if_done=skip_done) as prog:
        btm_result = btm_detect(df)
        n_btm = btm_result.summary["n_btm_detected"]
        prog.update(label=f"BTM: {n_btm}명 ({btm_result.summary['btm_rate']*100:.1f}%)")

        Path("btm_results").mkdir(exist_ok=True)
        save_dataframe(btm_result.customer_flags, "btm_results", "btm_flags", "고객별 BTM 플래그 (반출불가)")
        with open("btm_results/btm_summary.json", "w", encoding="utf-8") as f:
            json.dump(btm_result.summary, f, ensure_ascii=False, indent=2)
        ckpt.save_intermediate("btm_summary", btm_result.summary)

    # BTM 피처 병합
    btm_flags = btm_result.customer_flags[["customer_id", "is_btm", "btm_score"]].copy()
    df = df.merge(btm_flags, on="customer_id", how="left")
    df["is_btm"] = df["is_btm"].fillna(0).astype(int)
    df["btm_score"] = df["btm_score"].fillna(0.0)
    print(f"  BTM 피처 병합 완료 (is_btm + btm_score)")

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
    with ckpt.step("step4_baselines", total=5, skip_if_done=skip_done) as prog:
        daily = ev.daily_by_customer(df)
        monthly = ev.monthly_by_customer(daily)
        horizon = ev.build_horizon_table(daily, horizons=(10, 20))
        ctx = ev.attach_alarm_context(horizon, monthly)

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

    # ────────────────────────────────────────────
    # STEP 4.5: 데이터 분할 + 누수 체크
    # ────────────────────────────────────────────
    with ckpt.step("step4_5_split", skip_if_done=skip_done) as prog:
        from src.split import time_split, suggest_split, check_leakage

        features_pre, spec = lgbm.build_features(df)

        # 전처리 모드: preprocess_lp에서 미리 계산한 AMI 피처 조인
        if use_preprocessed and ami_parquet.exists() and 'ami_df' in dir():
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

    # ────────────────────────────────────────────
    # STEP 5: LightGBM 학습
    # ────────────────────────────────────────────
    features = features_pre  # Step 4.5에서 빌드한 것 재사용

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

    # ────────────────────────────────────────────
    # STEP 5.5: Sliding 상세 평가 (기본)
    # ────────────────────────────────────────────
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

    # ────────────────────────────────────────────
    # STEP 6: SHAP 분석
    # ────────────────────────────────────────────
    if not args.skip_shap:
        with ckpt.step("step6_shap", skip_if_done=skip_done) as prog:
            test_mask = features["year_month"] > (train_end if train_end else pd.Period("2099-12", freq="M"))
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
    else:
        print("\n[skip] SHAP 분석")

    # ────────────────────────────────────────────
    # STEP 7: 기상 최적화
    # ────────────────────────────────────────────
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

    # ────────────────────────────────────────────
    # STEP 8: 기상 민감도
    # ────────────────────────────────────────────
    if not args.skip_shap:
        with ckpt.step("step8_sensitivity", skip_if_done=skip_done) as prog:
            try:
                from src.weather.asos import weather_sensitivity_analysis
                weather_cols = [c for c in spec.all if any(
                    k in c for k in ["cdh", "hdh", "temp", "hdd", "cdd"]
                )]
                if weather_cols and len(test_features) > 0:
                    def predict_fn(X):
                        X_copy = X.copy()
                        for c in spec.categorical:
                            X_copy[c] = X_copy[c].astype("category")
                        return booster.predict(X_copy[spec.all], num_iteration=booster.best_iteration)

                    sensitivity = weather_sensitivity_analysis(
                        predict_fn, test_features, weather_cols
                    )
                    save_dataframe(sensitivity, "weather_opt", "sensitivity", "기상 민감도 분석")
                    for _, row in sensitivity.iterrows():
                        prog.update(label=f"{row['source']:>15s}: MAPE {row['mape_pct']:.3f}%")
                else:
                    prog.update(label="기상 피처 없음 — 스킵")
            except Exception as e:
                prog.update(label=f"민감도 실패: {e}")

    # ────────────────────────────────────────────
    # STEP 10: Ablation Study — 최적 조합 탐색
    # ────────────────────────────────────────────
    with ckpt.step("step10_ablation", skip_if_done=skip_done) as prog:
        try:
            import subprocess
            prog.update(label="Ablation Study 실행 중...")
            # Ablation에는 val_end를 전달 — Ablation 내부에서 train은 ~val_end, test는 val_end 이후
            abl_end = str(val_end_p) if val_end_p else (args.train_end or f"{features['year_month'].max().year - 1}-12")
            result = subprocess.run(
                [sys.executable, "-m", "scripts.ablation_study",
                 "--source", args.source, "--train-end", abl_end],
                capture_output=False,
            )
            if result.returncode == 0:
                # 최적 설정 로드
                opt_path = Path("ablation_results/optimal_config.json")
                if opt_path.exists():
                    with open(opt_path, encoding="utf-8") as f:
                        optimal_config = json.load(f)
                    prog.update(label=f"최적 설정: {optimal_config}")
                else:
                    optimal_config = None
                    prog.update(label="최적 설정 파일 없음 — 기본값 유지")
            else:
                optimal_config = None
                prog.update(label="Ablation 실패 — 기본값 유지")
        except Exception as e:
            optimal_config = None
            prog.update(label=f"Ablation 실패: {e}")

    # ────────────────────────────────────────────
    # STEP 11: 최적 조합으로 재학습 + 피처 자동 선정
    # ────────────────────────────────────────────
    with ckpt.step("step11_retrain_optimal", skip_if_done=skip_done) as prog:
        if optimal_config and optimal_config != {}:
            prog.update(label="최적 설정으로 재학습 중...")

            # 피처 선택이 최적이면 auto_select 적용
            if optimal_config.get("feature_selection", False):
                from src.feature_selection import auto_select_features, print_selection_report
                prog.update(label="피처 자동 선정 중 (train+val만, 누수 방지)...")
                # 누수 방지: test 제외
                trainval_features = features[features["year_month"] <= (val_end_p if val_end_p else train_end_p)]
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
            train_end_p = pd.Period(args.train_end, freq="M") if args.train_end else None
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

    # ────────────────────────────────────────────
    # STEP 11.5: 잔차 보정 (2-stage Ridge)
    # ────────────────────────────────────────────
    with ckpt.step("step11_5_residual", skip_if_done=skip_done) as prog:
        try:
            from src.residual_correction import run_residual_correction
            prog.update(label="잔차 보정 모델 학습 중...")

            # 최적 모델(또는 기본 모델)의 예측값과 실측값
            best_booster = booster  # 기본 모델
            opt_model_path = Path("weights/dsz_lgbm_optimal/model.txt")
            if opt_model_path.exists():
                best_booster, _ = lgbm.load("weights/dsz_lgbm_optimal")
                prog.update(label="최적 모델 기반 잔차 보정")

            feat_clean = features.dropna(subset=["full_month_kwh", "partial_kwh"]).copy()
            for c in spec.categorical:
                if c in feat_clean.columns:
                    feat_clean[c] = feat_clean[c].astype("category")

            pred_all = best_booster.predict(
                feat_clean[spec.all], num_iteration=best_booster.best_iteration
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

    # ────────────────────────────────────────────
    # STEP 12: 하이퍼파라미터 튜닝
    # ────────────────────────────────────────────
    with ckpt.step("step12_tuning", skip_if_done=skip_done) as prog:
        try:
            prog.update(label="Optuna 튜닝 시작 (10분 타임아웃)...")
            import subprocess as _sp2
            train_end_str = args.train_end or f"{features['year_month'].max().year - 1}-12"
            result = _sp2.run(
                [sys.executable, "-m", "scripts.tune_lgbm",
                 "--source", args.source, "--train-end", train_end_str,
                 "--timeout", "600", "--n-trials", "50"],
                capture_output=False,
            )
            if result.returncode == 0:
                prog.update(label="튜닝 완료 — weights/dsz_lgbm_tuned/")
            else:
                prog.update(label="튜닝 실패 또는 시간 초과")
        except Exception as e:
            prog.update(label=f"튜닝 실패: {e}")

    # ────────────────────────────────────────────
    # STEP 12.5: 튜닝 모델 Sliding 재평가
    # ────────────────────────────────────────────
    with ckpt.step("step12_5_tuned_sliding", skip_if_done=skip_done) as prog:
        tuned_path = Path("weights/dsz_lgbm_tuned/model.txt")
        if tuned_path.exists():
            try:
                booster_tuned, spec_tuned = lgbm.load("weights/dsz_lgbm_tuned")
                prog.update(label="튜닝 모델 Sliding 재평가 중...")

                test_tuned = features.copy()
                train_end_p = pd.Period(args.train_end, freq="M") if args.train_end else None
                if train_end_p is None:
                    max_ym = features["year_month"].max()
                    train_end_p = pd.Period(f"{max_ym.year - 1}-12", freq="M")
                test_tuned = test_tuned[test_tuned["year_month"] > train_end_p]
                for c in spec_tuned.categorical:
                    if c in test_tuned.columns:
                        test_tuned[c] = test_tuned[c].astype("category")

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
                        merged["meter_day"] = md
                        merged["error"] = merged["pred_monthly_kwh"] - merged["full_month_kwh"]
                        merged["pct_error"] = merged["error"].abs() / merged["full_month_kwh"].clip(lower=1e-9) * 100
                        tuned_sliding.append(merged)
                    except Exception:
                        pass

                if tuned_sliding:
                    from src.eval_detailed import save_detailed_evaluation
                    all_tuned = pd.concat(tuned_sliding, ignore_index=True)
                    dims_tuned = save_detailed_evaluation(all_tuned, "sliding_results_tuned")
                    ckpt.save_intermediate("tuned_sliding", dims_tuned.get("overall", pd.DataFrame()))
            except Exception as e:
                prog.update(label=f"튜닝 모델 평가 실패: {e}")
        else:
            prog.update(label="튜닝 모델 없음 — 스킵")

    # ────────────────────────────────────────────
    # STEP 13: 스케일 아웃 지표 (최종 모델 기준)
    # ────────────────────────────────────────────
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

    # ────────────────────────────────────────────
    # 완료
    # ────────────────────────────────────────────
    elapsed = time.time() - ckpt.progress.start_time if ckpt.progress.start_time else 0

    # ────────────────────────────────────────────
    # 보고서용 이미지 자동 생성
    # ────────────────────────────────────────────
    print("\n[보고서용 이미지 생성]", flush=True)
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "scripts.generate_figures"], check=False)
        print("  [OK] 이미지 생성 완료", flush=True)
    except Exception as e:
        print(f"  [경고] 이미지 생성 실패: {e}", flush=True)

    print(f"""
╔══════════════════════════════════════════════════════╗
║                    전체 완료                          ║
╚══════════════════════════════════════════════════════╝

  소요 시간: {elapsed/60:.1f}분
  체크포인트: {args.checkpoint_dir}/

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
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
