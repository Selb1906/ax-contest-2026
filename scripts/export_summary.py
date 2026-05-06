"""반출용 통합 성능 요약 CSV + 반출 안전 대시보드 생성.

사용법:
  python -m scripts.export_summary                    # 통합 CSV 생성
  python -m scripts.export_summary --dashboard        # + 반출 안전 대시보드
"""
from __future__ import annotations

import io as _stdio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "export"


def load_model_metrics():
    """모델별 성능 (베이스라인 + 제안모델)."""
    for p in [ROOT / "eval_results" / "metrics.csv",
              ROOT / "eval_results" / "lgbm_vs_baselines.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            df["category"] = "모델 비교"
            return df
    return None


def load_ablation():
    """Ablation 설정별 성능."""
    p = ROOT / "ablation_results" / "ablation_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "category": "Ablation",
            "model": r.get("name", ""),
            "config": r.get("config", ""),
            "mape_pct": r.get("mape_10d"),
            "mape_20d": r.get("mape_20d"),
        })
    return pd.DataFrame(rows)


def load_tuning():
    """Optuna 튜닝 전후."""
    best_path = ROOT / "tuning_results" / "best_params.json"
    history_path = ROOT / "tuning_results" / "study_history.csv"
    rows = []
    if best_path.exists():
        with open(best_path, encoding="utf-8") as f:
            bp = json.load(f)
        rows.append({
            "category": "Optuna 튜닝",
            "model": "튜닝 후 최적",
            "config": json.dumps(bp.get("params", bp), ensure_ascii=False)[:100],
            "mape_pct": bp.get("best_mape") or bp.get("mape"),
        })
    if history_path.exists():
        hist = pd.read_csv(history_path)
        if len(hist) > 0 and "value" in hist.columns:
            rows.append({
                "category": "Optuna 튜닝",
                "model": f"탐색 이력 ({len(hist)}회)",
                "config": f"최저 {hist['value'].min():.3f}%, 최고 {hist['value'].max():.3f}%",
                "mape_pct": hist["value"].min(),
            })
    return pd.DataFrame(rows) if rows else None


def load_residual():
    """잔차 보정 전후."""
    p = ROOT / "residual_correction" / "residual_correction.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        rc = json.load(f)
    b = rc.get("before_correction", {})
    a = rc.get("after_correction", {})
    rows = [
        {"category": "잔차 보정", "model": "보정 전", "mae": b.get("mae"), "rmse": b.get("rmse"),
         "mape_pct": b.get("mape_pct"), "cvrmse_pct": b.get("cvrmse_pct")},
        {"category": "잔차 보정", "model": "보정 후", "mae": a.get("mae"), "rmse": a.get("rmse"),
         "mape_pct": a.get("mape_pct"), "cvrmse_pct": a.get("cvrmse_pct")},
    ]
    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="반출용 통합 요약")
    parser.add_argument("--dashboard", action="store_true", help="반출 안전 대시보드도 생성")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # 통합 성능 요약
    frames = []
    for name, loader in [
        ("모델 비교", load_model_metrics),
        ("Ablation", load_ablation),
        ("Optuna", load_tuning),
        ("잔차 보정", load_residual),
    ]:
        df = loader()
        if df is not None:
            frames.append(df)
            print(f"  {name}: {len(df)}행")
        else:
            print(f"  {name}: 없음")

    if frames:
        # 통합 CSV (concat) — 빈 셀 포함
        summary = pd.concat(frames, ignore_index=True)

        # 카테고리별 개별 CSV — 빈 셀 없이 깔끔
        for cat_df in frames:
            if cat_df is None or len(cat_df) == 0:
                continue
            cat = cat_df["category"].iloc[0].replace(" ", "_")
            clean = cat_df.dropna(axis=1, how="all")
            clean.to_csv(OUT / f"perf_{cat}.csv", index=False, encoding="utf-8-sig")

        # 핵심 지표만 추린 요약 테이블
        key_cols = ["category", "model", "mae", "rmse", "cvrmse_pct", "mape_pct", "alarm_f1", "config"]
        available = [c for c in key_cols if c in summary.columns]
        compact = summary[available].copy()
        compact = compact.dropna(subset=[c for c in ["mape_pct", "mae"] if c in compact.columns], how="all")
        compact.to_csv(OUT / "performance_summary.csv", index=False, encoding="utf-8-sig")
        print(f"\n[저장] export/performance_summary.csv ({len(compact)}행, 핵심 지표만)")

        for f in OUT.glob("perf_*.csv"):
            n = len(pd.read_csv(f))
            print(f"  {f.name} ({n}행)")
    else:
        print("\n  성능 데이터 없음")

    # sliding 상세 분석 복사 (반출 안전 — 집계만)
    sliding_dir = ROOT / "sliding_results"
    safe_sliding = [
        "error_overall.csv", "error_by_month.csv", "error_by_horizon.csv",
        "error_by_contract.csv", "error_by_meter_day.csv", "error_by_date.csv",
        "error_by_special.csv", "best_dates.csv", "worst_dates.csv",
    ]
    copied = 0
    for f in safe_sliding:
        src = sliding_dir / f
        if src.exists():
            import shutil
            shutil.copy2(src, OUT / f"sliding_{f}")
            copied += 1
    if copied:
        print(f"\n[저장] sliding 상세 분석 {copied}개 → export/sliding_*.csv")

    # 논문용 이미지 복사
    fig_dir = ROOT / "figures"
    if fig_dir.exists():
        (OUT / "figures").mkdir(exist_ok=True)
        fig_count = 0
        for f in fig_dir.glob("*.png"):
            import shutil
            shutil.copy2(f, OUT / "figures" / f.name)
            fig_count += 1
        if fig_count:
            print(f"\n[저장] 논문용 이미지 {fig_count}개 → export/figures/")

    # SHAP 이미지 복사
    explain_dir = ROOT / "explain_results"
    if explain_dir.exists():
        (OUT / "shap").mkdir(exist_ok=True)
        shap_count = 0
        for f in explain_dir.glob("*.png"):
            import shutil
            shutil.copy2(f, OUT / "shap" / f.name)
            shap_count += 1
        for f in explain_dir.glob("*.csv"):
            import shutil
            shutil.copy2(f, OUT / "shap" / f.name)
            shap_count += 1
        if shap_count:
            print(f"[저장] SHAP 결과 {shap_count}개 → export/shap/")

    # 모델 가중치 복사
    weights_dir = ROOT / "weights"
    if weights_dir.exists():
        (OUT / "weights").mkdir(exist_ok=True)
        import shutil
        for sub in weights_dir.iterdir():
            if sub.is_dir():
                dst = OUT / "weights" / sub.name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(sub, dst)
        print(f"[저장] 모델 가중치 → export/weights/")

    # 기타 반출 안전 파일
    safe_files = {
        "btm_results/btm_summary.json": "btm_summary.json",
        "preprocess_log/log.json": "preprocess_log.json",
        "weather_opt/weather_optimization.json": "weather_optimization.json",
        "residual_correction/residual_correction.json": "residual_correction.json",
        "tuning_results/best_params.json": "tuning_best_params.json",
        "tuning_results/study_history.csv": "tuning_history.csv",
        "tuning_results/param_importance.csv": "tuning_param_importance.csv",
        "ablation_results/best_config.json": "ablation_best_config.json",
    }
    misc_count = 0
    for src_rel, dst_name in safe_files.items():
        src = ROOT / src_rel
        if src.exists():
            import shutil
            shutil.copy2(src, OUT / dst_name)
            misc_count += 1
    if misc_count:
        print(f"[저장] 기타 반출 파일 {misc_count}개 → export/")

    # 반출 안전 대시보드
    if args.dashboard:
        print("\n[대시보드] 반출 안전 버전 생성 중...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "scripts.generate_dashboard",
             "--out", str(OUT / "dashboard_safe.html"),
             "--export-safe"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  export/dashboard_safe.html 생성 완료")
        else:
            print(f"  실패: {result.stderr[-200:]}")

    print(f"\n[완료] 반출 대상 파일: export/")


if __name__ == "__main__":
    main()
