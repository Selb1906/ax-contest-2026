"""결과 CSV 자동 생성 — 1차/2차 분석 결과를 하나의 CSV로 통합.

Usage:
    python -m scripts.generate_results_csv
    → results_combined.csv 생성
"""
from __future__ import annotations
import json, sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
import os; os.chdir(_PROJECT_ROOT)

import pandas as pd
import numpy as np

OUT = _PROJECT_ROOT / "results_combined.csv"

rows = []

def _add(section, model, scale, horizon, source, **metrics):
    rows.append({
        "Section": section, "Model": model, "Scale": scale, "Horizon": horizon,
        "N": metrics.get("n"), "MAPE": metrics.get("mape_pct"), "RMSE": metrics.get("rmse"),
        "CVRMSE": metrics.get("cvrmse_pct"), "CVRMSE_med": metrics.get("cvrmse_median_pct"),
        "MAE": metrics.get("mae"), "Bias": metrics.get("bias"),
        "Precision": metrics.get("alarm_precision"), "Recall": metrics.get("alarm_recall"),
        "F1": metrics.get("alarm_f1"),
    })

def _load_overall(name, source_dir):
    p = Path(source_dir) / "error_overall.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if len(df) == 0:
        return
    r = df.iloc[0].to_dict()
    _add("Overall", name, "all", "all", str(p), **r)
    # 조건별 알림 성능
    for cond, label in [("cond_prev", "prev+30%"), ("cond_yoy", "yoy+30%"), ("cond_ma", "ma3+50%")]:
        p_key = f"{cond}_precision"
        if p_key in r:
            _add("AlarmCond", f"{name}_{label}", "", "", str(p),
                 alarm_precision=r.get(f"{cond}_precision"),
                 alarm_recall=r.get(f"{cond}_recall"),
                 alarm_f1=r.get(f"{cond}_f1"))

def _load_by_horizon(name, source_dir):
    p = Path(source_dir) / "error_by_horizon.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        hz = r.get("horizon_days", "all")
        rd = {k: v for k, v in r.to_dict().items() if k != "horizon_days"}
        _add("ByHorizon", name, "all", f"+{int(hz)}d" if hz != "all" else "all", str(p), **rd)

def _load_cold_start(name, source_dir):
    p = Path(source_dir) / "error_by_cold_start.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        cs = r.get("cold_start", "")
        rd = {k: v for k, v in r.to_dict().items() if k not in ("cold_start", "n_customers")}
        rd["n"] = r.get("n_customers", r.get("n"))
        _add("ColdStart", f"{name}_{cs}", "", "", str(p), **rd)

def _load_by_month(name, source_dir):
    p = Path(source_dir) / "error_by_month.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        m = r.get("month", "")
        rd = {k: v for k, v in r.to_dict().items() if k != "month"}
        _add("ByMonth", name, "", f"M{int(m)}", str(p), **rd)

def _load_scale_horizon(name, source_dir):
    p = Path(source_dir) / "error_by_scale_horizon.csv"
    if not p.exists():
        p = Path(source_dir) / "error_by_scale.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        scale = r.get("scale", "all")
        hz = r.get("horizon_days", "all")
        rd = {k: v for k, v in r.to_dict().items() if k not in ("scale", "horizon_days")}
        _add("ScaleHz", name, scale, f"+{int(hz)}d" if hz != "all" else "all", str(p), **rd)

def _load_ablation():
    p = Path("ablation_results/ablation_results.csv")
    if not p.exists():
        return
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        _add("Ablation", r["name"], "", "+10d", str(p),
             mape_pct=r.get("mape_10d"), cvrmse_median_pct=r.get("cvrmse_med_10d"),
             rmse=r.get("rmse_10d"), n=r.get("n_10d"))
        if "mape_20d" in r and pd.notna(r.get("mape_20d")):
            _add("Ablation", r["name"], "", "+20d", str(p),
                 mape_pct=r.get("mape_20d"), cvrmse_median_pct=r.get("cvrmse_med_20d"),
                 rmse=r.get("rmse_20d"), n=r.get("n_20d"))

def _load_optimal_config():
    p = Path("ablation_results/best_config.json")
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    bpf = d.get("best_per_factor", {})
    for k, v in bpf.items():
        _add("OptConfig", k, "", "", str(p), mape_pct=v if isinstance(v, (int, float)) else None,
             n=None, bias=None)
        # 문자열 값은 Model 필드에 넣기
        if isinstance(v, str):
            rows[-1]["MAPE"] = v  # 임시로 MAPE 컬럼에 문자열 저장

def _load_ablation_contrib():
    p = Path("ablation_results/feature_group_contribution.csv")
    if not p.exists():
        return
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        _add("AblContrib", r["group"], "", "", str(p),
             cvrmse_median_pct=r.get("cvrmse_med_without"),
             bias=r.get("contribution"))

def _load_tuning():
    p = Path("tuning_results/best_params.json")
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    _add("Tuning", "baseline", "", "", str(p), cvrmse_median_pct=d.get("baseline_cvrmse"))
    _add("Tuning", "best", "", "", str(p), cvrmse_median_pct=d.get("best_cvrmse"))
    _add("Tuning", "n_trials", "", "", str(p), n=d.get("n_trials"))
    bp = d.get("best_params", {})
    for k, v in bp.items():
        _add("TuneParam", k, "", "", str(p), mape_pct=v)

def _load_residual():
    p = Path("residual_correction/residual_summary.json")
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    if "before" in d:
        _add("Residual", "before", "", "", str(p), **d["before"])
    if "after" in d:
        _add("Residual", "after", "", "", str(p), **d["after"])
    if "ridge_intercept" in d:
        _add("Residual", "ridge_alpha", "", "", str(p),
             bias=d.get("ridge_intercept"))

def _load_1st_analysis():
    """1차 반출 데이터에서 가져오기."""
    export_dir = Path("SC-2026-00020 반출")
    if not export_dir.exists():
        return

    # performance_summary.csv
    ps = export_dir / "performance_summary.csv"
    if ps.exists():
        df = pd.read_csv(ps)
        for _, r in df.iterrows():
            cat = r.get("category", "")
            model = r.get("model", "")
            if cat == "모델 비교":
                _add("1st_ModelCompare", model, "", "", str(ps),
                     mae=r.get("mae"), rmse=r.get("rmse"),
                     cvrmse_pct=r.get("cvrmse_pct"), mape_pct=r.get("mape_pct"),
                     alarm_f1=r.get("alarm_f1"))

    # analysis_filtered.csv (규모별)
    af = export_dir / "analysis_filtered.csv"
    if af.exists():
        with open(af, encoding="utf-8") as f:
            lines = f.readlines()
        current_section = ""
        for line in lines:
            line = line.strip()
            if line.startswith("scale("):
                current_section = line.split(",")[0]
                continue
            if not line or line.startswith("scale"):
                continue
            parts = line.split(",")
            if len(parts) >= 6:
                try:
                    _add(f"1st_{current_section}", "", parts[0], f"+{parts[1]}d", str(af),
                         n=float(parts[2]), mape_pct=float(parts[3]),
                         rmse=float(parts[4]), cvrmse_pct=float(parts[5]))
                except (ValueError, IndexError):
                    pass

    # shap_global.csv
    sg = export_dir / "shap_global.csv"
    if sg.exists():
        df = pd.read_csv(sg)
        for _, r in df.head(10).iterrows():
            _add("1st_SHAP", r["feature"], "", "", str(sg),
                 mape_pct=r.get("pct_of_total"), n=r.get("rank"))

    # 10_alarm_base_rates.json
    ab = export_dir / "10_alarm_base_rates.json"
    if ab.exists():
        with open(ab, encoding="utf-8") as f:
            d = json.load(f)
        cal = d.get("calendar_month_basis", {})
        for cond in ["cond_prev", "cond_yoy", "cond_ma3", "any_of_3"]:
            if cond in cal:
                _add("1st_AlarmBase", cond, "", "", str(ab),
                     mape_pct=cal[cond].get("trigger_rate"),
                     n=cal[cond].get("n_trigger"))

    # metrics.csv (모델별 상세)
    mc = export_dir / "metrics.csv"
    if mc.exists():
        df = pd.read_csv(mc)
        for _, r in df.iterrows():
            _add("1st_Metrics", r["model"], "", f"+{int(r['horizon_days'])}d", str(mc),
                 n=r.get("n"), mae=r.get("mae"), rmse=r.get("rmse"),
                 cvrmse_pct=r.get("cvrmse_pct"), mape_pct=r.get("mape_pct"),
                 alarm_precision=r.get("alarm_precision"),
                 alarm_recall=r.get("alarm_recall"),
                 alarm_f1=r.get("alarm_f1"))


def main():
    print("[1/3] 1차 분석 데이터 로드...", flush=True)
    _load_1st_analysis()
    print(f"  1차: {len(rows)}행")

    print("[2/3] 2차 분석 데이터 로드...", flush=True)
    n_before = len(rows)
    for name in ["partial_linear", "lgbm_base", "lgbm_tuned", "lgbm_corrected"]:
        # suffix 자동 탐색: 없으면 _2d, _6d, _12d 순으로
        base_dir = f"sliding_results_{name}"
        if not Path(base_dir).exists():
            for sfx in ["_2d", "_6d", "_12d"]:
                if Path(f"{base_dir}{sfx}").exists():
                    base_dir = f"{base_dir}{sfx}"
                    break
        _load_overall(name, base_dir)
        _load_by_horizon(name, base_dir)
        _load_scale_horizon(name, base_dir)
        _load_by_month(name, base_dir)
        _load_cold_start(name, base_dir)
        # suffix별 추가 로드 (수렴 분석용)
        for sfx in ["_2d", "_6d", "_12d"]:
            sfx_dir = f"sliding_results_{name}{sfx}"
            if Path(sfx_dir).exists() and sfx_dir != base_dir:
                _load_overall(f"{name}{sfx}", sfx_dir)
    _load_ablation()
    _load_ablation_contrib()
    _load_optimal_config()
    _load_tuning()
    _load_residual()
    print(f"  2차: {len(rows) - n_before}행")

    print("[3/3] CSV 저장...", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"  → {OUT} ({len(df)}행)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
