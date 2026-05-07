"""파이프라인 결과 -> 자체 완결 HTML 대시보드 생성.

사용법:
  python -m scripts.generate_dashboard --results eval_results/ --out dashboard.html
  python -m scripts.generate_dashboard  # 기본 경로 사용

생성된 dashboard.html 은 브라우저에서 열면 바로 동작.
추가 패키지/서버 불필요. CDN 없음 (안심구역 오프라인 호환).
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════
# 1. 데이터 로딩
# ═══════════════════════════════════════════════════════

def _safe_read_parquet(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _load_image_base64(path: Path) -> str | None:
    if path.exists():
        data = path.read_bytes()
        ext = path.suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "svg": "image/svg+xml"}.get(ext.lstrip("."), "image/png")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    return None


def load_pipeline_data(
    ui_dir: Path,
    eval_dir: Path,
    sliding_dir: Path,
    explain_dir: Path,
) -> dict:
    """모든 데이터를 dict 로 수집. 없는 항목은 None."""
    data = {}

    # -- UI parquets --
    for name in ("daily", "daily_cum", "monthly", "preds", "ctx", "metrics"):
        df = _safe_read_parquet(ui_dir / f"{name}.parquet")
        if df is not None:
            # Period 타입 직렬화 불가 -> str
            for c in df.columns:
                if hasattr(df[c], "dt") and hasattr(df[c].dt, "to_timestamp"):
                    try:
                        df[c] = df[c].astype(str)
                    except Exception:
                        pass
            data[name] = df
        else:
            data[name] = None

    # -- eval results --
    for name in ("baselines", "lgbm_vs_baselines"):
        csv_path = eval_dir / f"{name}.csv"
        parquet_path = eval_dir / f"{name}.parquet"
        df = _safe_read_csv(csv_path) if csv_path.exists() else _safe_read_parquet(parquet_path)
        data[f"eval_{name}"] = df

    # eval_results/metrics.parquet 가 있으면 ui/metrics.parquet 대체 (LightGBM 포함)
    eval_metrics = _safe_read_parquet(eval_dir / "metrics.parquet")
    if eval_metrics is not None and "model" in eval_metrics.columns:
        for c in eval_metrics.columns:
            if hasattr(eval_metrics[c], "dt"):
                try:
                    eval_metrics[c] = eval_metrics[c].astype(str)
                except Exception:
                    pass
        data["metrics"] = eval_metrics

    # -- sliding results --
    sliding_files = list(sliding_dir.glob("*.csv")) + list(sliding_dir.glob("*.parquet"))
    if sliding_files:
        frames = []
        for f in sliding_files:
            if f.suffix == ".csv":
                frames.append(pd.read_csv(f))
            else:
                frames.append(pd.read_parquet(f))
        data["sliding"] = pd.concat(frames, ignore_index=True) if frames else None
    else:
        data["sliding"] = None

    # -- SHAP images --
    shap_images = {}
    for ext in ("png", "jpg", "jpeg", "svg"):
        for img in explain_dir.glob(f"*.{ext}"):
            b64 = _load_image_base64(img)
            if b64:
                shap_images[img.stem] = b64
    data["shap_images"] = shap_images if shap_images else None

    # -- ablation results --
    ablation_csv = ROOT / "ablation_results" / "feature_group_contribution.csv"
    data["ablation"] = _safe_read_csv(ablation_csv)

    ablation_full = _safe_read_csv(ROOT / "ablation_results" / "ablation_results.csv")
    data["ablation_full"] = ablation_full

    best_config_path = ROOT / "ablation_results" / "best_config.json"
    if best_config_path.exists():
        import json as _json
        with open(best_config_path, encoding="utf-8") as f:
            data["ablation_best"] = _json.load(f)
    else:
        data["ablation_best"] = None

    # -- residual correction --
    res_json = ROOT / "residual_correction" / "residual_correction.json"
    if res_json.exists():
        import json as _json
        with open(res_json, encoding="utf-8") as f:
            data["residual_correction"] = _json.load(f)
    else:
        data["residual_correction"] = None

    return data


# ═══════════════════════════════════════════════════════
# 2. DataFrame -> JSON-safe dict
# ═══════════════════════════════════════════════════════

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Period):
            return str(obj)
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return super().default(obj)


def _df_to_records(df: pd.DataFrame | None) -> list[dict] | None:
    if df is None:
        return None
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype(str)
    records = df.where(df.notna(), None).to_dict(orient="records")
    return records


def _df_to_columnar(df: pd.DataFrame | None) -> dict | None:
    """Columnar format: {columns:[...], data:{col:[values]}} -- far smaller than records for wide data."""
    if df is None:
        return None
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype(str)
    cols = list(df.columns)
    data = {}
    for c in cols:
        vals = df[c].where(df[c].notna(), None).tolist()
        # round floats to reduce JSON size
        if df[c].dtype.kind == "f":
            vals = [round(v, 2) if v is not None else None for v in vals]
        data[c] = vals
    return {"columns": cols, "data": data, "length": len(df)}


def build_json_payload(data: dict) -> str:
    payload = {}
    # Large tables -> columnar format for smaller JSON
    for key in ("daily_cum",):
        payload[key] = _df_to_columnar(data.get(key))
    # Regular tables -> records
    for key in ("monthly", "preds", "ctx", "metrics",
                "eval_baselines", "eval_lgbm_vs_baselines", "sliding"):
        payload[key] = _df_to_records(data.get(key))
    # Skip raw daily (not needed - daily_cum has all we need, fleet uses monthly)
    payload["daily"] = None
    payload["shap_images"] = data.get("shap_images")
    payload["ablation"] = _df_to_records(data.get("ablation"))
    payload["ablation_full"] = _df_to_records(data.get("ablation_full"))
    payload["ablation_best"] = data.get("ablation_best")
    payload["residual_correction"] = data.get("residual_correction")
    return json.dumps(payload, cls=_NpEncoder, ensure_ascii=False)


# ═══════════════════════════════════════════════════════
# 3. 정적 파일 인라인 삽입
# ═══════════════════════════════════════════════════════

def read_static(name: str) -> str:
    p = ROOT / "ui" / "static" / name
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    print(f"[warn] static file not found: {p}")
    return ""


# ═══════════════════════════════════════════════════════
# 4. HTML 템플릿
# ═══════════════════════════════════════════════════════

def build_html(json_payload: str) -> str:
    bootstrap_css = read_static("bootstrap.min.css")
    bootstrap_js = read_static("bootstrap.min.js")
    echarts_js = read_static("echarts.min.js")

    return _HTML_TEMPLATE.replace("/* __BOOTSTRAP_CSS__ */", bootstrap_css) \
                         .replace("/* __BOOTSTRAP_JS__ */", bootstrap_js) \
                         .replace("/* __ECHARTS_JS__ */", echarts_js) \
                         .replace("__JSON_PAYLOAD__", json_payload)


# ═══════════════════════════════════════════════════════
# 5. 메인 HTML (큰 문자열)
# ═══════════════════════════════════════════════════════

_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="ko" data-bs-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>전기요금 과다발생 사전 예측 대시보드</title>
<style>/* __BOOTSTRAP_CSS__ */</style>
<style>
/* ── Theme Variables ── */
:root, [data-theme="dark"] {
  --bg: #212529;
  --card-bg: #1f2937;
  --card-border: #374151;
  --accent: #3b82f6;
  --accent-green: #10b981;
  --accent-red: #ef4444;
  --accent-yellow: #f59e0b;
  --text-primary: #f3f4f6;
  --text-secondary: #9ca3af;
  --navbar-bg: #111827;
  --table-stripe: rgba(255,255,255,0.03);
  --input-bg: #111827;
  --input-border: #374151;
}
[data-theme="light"] {
  --bg: #f5f5f5;
  --card-bg: #ffffff;
  --card-border: #e0e0e0;
  --accent: #2563eb;
  --accent-green: #059669;
  --accent-red: #dc2626;
  --accent-yellow: #d97706;
  --text-primary: #1f2937;
  --text-secondary: #6b7280;
  --navbar-bg: #ffffff;
  --table-stripe: rgba(0,0,0,0.02);
  --input-bg: #ffffff;
  --input-border: #d1d5db;
}
body {
  background: var(--bg);
  color: var(--text-primary);
  font-family: 'Segoe UI', 'Malgun Gothic', system-ui, sans-serif;
}
.navbar { background: var(--navbar-bg) !important; border-bottom: 1px solid var(--card-border); }
[data-theme="light"] .navbar { box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
[data-theme="light"] .table { color: var(--text-primary); }
[data-theme="light"] .form-control, [data-theme="light"] .form-select {
  background: var(--input-bg); color: var(--text-primary); border-color: var(--input-border);
}
#theme-toggle { cursor: pointer; background: var(--card-bg); border: 1px solid var(--card-border);
  color: var(--text-primary); padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; }
.navbar-brand { font-weight: 700; font-size: 1.25rem; }
.nav-tabs { border-bottom: 2px solid var(--card-border); }
.nav-tabs .nav-link {
  color: var(--text-secondary);
  border: none;
  padding: 0.75rem 1.25rem;
  font-weight: 600;
  font-size: 0.9rem;
  border-radius: 0;
  transition: color 0.2s;
}
.nav-tabs .nav-link:hover { color: var(--text-primary); background: transparent; }
.nav-tabs .nav-link.active {
  color: var(--accent);
  background: transparent;
  border-bottom: 3px solid var(--accent);
}
.card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 12px;
}
.card-header { background: transparent; border-bottom: 1px solid var(--card-border); font-weight: 600; }
.card-body { overflow: visible; }
.card-body [id^="chart-"] { margin: 4px; }
.kpi-card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 12px;
  padding: 20px 24px;
  text-align: center;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.kpi-card .kpi-label {
  color: var(--text-secondary);
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}
.kpi-card .kpi-value {
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--text-primary);
}
.kpi-card .kpi-delta {
  font-size: 0.82rem;
  margin-top: 4px;
  min-height: 1.2em;
}
.delta-up { color: var(--accent-red); }
.delta-down { color: var(--accent-green); }
.delta-neutral { color: var(--text-secondary); }

.alarm-badge {
  display: inline-block;
  padding: 5px 14px;
  border-radius: 20px;
  font-size: 0.82rem;
  font-weight: 600;
}
.alarm-danger {
  background: rgba(239,68,68,0.15);
  color: var(--accent-red);
  border: 1px solid rgba(239,68,68,0.3);
}
.alarm-safe {
  background: rgba(16,185,129,0.15);
  color: var(--accent-green);
  border: 1px solid rgba(16,185,129,0.3);
}

.table { color: var(--text-primary); }
.table thead th {
  background: #111827;
  color: var(--text-secondary);
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border-bottom: 2px solid var(--card-border);
}
.table td { border-color: var(--card-border); vertical-align: middle; }
.table-hover tbody tr:hover { background: rgba(59,130,246,0.06); }

.form-select, .form-control {
  background: #111827;
  border-color: var(--card-border);
  color: var(--text-primary);
}
.form-select:focus, .form-control:focus {
  background: #111827;
  border-color: var(--accent);
  color: var(--text-primary);
  box-shadow: 0 0 0 0.2rem rgba(59,130,246,0.25);
}

.report-card {
  background: linear-gradient(135deg, #1e293b 0%, #1f2937 100%);
  border-radius: 16px;
  padding: 28px 32px;
  border: 1px solid var(--card-border);
  margin: 16px 0;
}
.report-card h4 { color: var(--text-primary); margin-bottom: 16px; }

.tip-box {
  background: rgba(16,185,129,0.08);
  border-left: 4px solid var(--accent-green);
  border-radius: 0 8px 8px 0;
  padding: 14px 18px;
  margin: 12px 0;
  color: var(--text-secondary);
}
.warn-box {
  background: rgba(239,68,68,0.08);
  border-left: 4px solid var(--accent-red);
  border-radius: 0 8px 8px 0;
  padding: 14px 18px;
  margin: 12px 0;
  color: var(--text-secondary);
}

.section-title {
  font-size: 1.1rem;
  font-weight: 700;
  margin-top: 1.5rem;
  margin-bottom: 1rem;
  color: var(--text-primary);
}

.no-data-msg {
  text-align: center;
  padding: 60px 20px;
  color: var(--text-secondary);
}
.no-data-msg .icon { font-size: 2.5rem; margin-bottom: 12px; }

.tab-content > .tab-pane { padding-top: 1.5rem; }

/* Scrollable container */
.container-xxl { max-width: 1440px; }

/* Override bootstrap dark form elements */
.form-label { color: var(--text-secondary); font-size: 0.82rem; font-weight: 600; }
</style>
</head>
<body>

<!-- ═══ Navbar ═══ -->
<nav class="navbar px-3 py-2 sticky-top">
  <span class="navbar-brand">전기요금 과다발생 사전 예측</span>
  <div class="d-flex align-items-center gap-3">
    <span class="text-secondary" style="font-size:0.8rem;">한국전력공사 지정과제 &middot; 2026 AX</span>
    <button id="theme-toggle" onclick="toggleTheme()">Light</button>
  </div>
</nav>

<div class="container-xxl px-3 py-2">

<!-- ═══ Tab Navigation ═══ -->
<ul class="nav nav-tabs mb-0" id="mainTabs" role="tablist">
  <li class="nav-item"><a class="nav-link active" data-bs-toggle="tab" href="#tab-customer" role="tab">고객 뷰</a></li>
  <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#tab-fleet" role="tab">모델 성능</a></li>
  <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#tab-tariff" role="tab">요금 분석</a></li>
  <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#tab-model" role="tab">모델 상세</a></li>
</ul>

<div class="tab-content" id="mainTabContent">

<!-- ═══════════════════════════════════════ -->
<!-- TAB 1: 고객 뷰                         -->
<!-- ═══════════════════════════════════════ -->
<div class="tab-pane fade show active" id="tab-customer" role="tabpanel">
  <div id="customer-no-data" class="no-data-msg" style="display:none;">
    <div class="icon">—</div>
    <div>데이터 없음 &mdash; 파이프라인 실행 후 재생성하세요.</div>
  </div>
  <div id="customer-content" style="display:none;">
    <div class="row g-3 mb-3">
      <div class="col-md-4">
        <label class="form-label">고객 선택</label>
        <select id="sel-customer" class="form-select form-select-sm"></select>
      </div>
      <div class="col-md-4">
        <label class="form-label">기준월</label>
        <select id="sel-month" class="form-select form-select-sm"></select>
      </div>
      <div class="col-md-4">
        <label class="form-label">검침일 기준</label>
        <div class="btn-group btn-group-sm w-100" role="group">
          <input type="radio" class="btn-check" name="horizon" id="hz10" value="10" checked>
          <label class="btn btn-outline-primary" for="hz10">검침일+10일</label>
          <input type="radio" class="btn-check" name="horizon" id="hz20" value="20">
          <label class="btn btn-outline-primary" for="hz20">검침일+20일</label>
        </div>
      </div>
    </div>
    <!-- KPI row -->
    <div class="row g-3 mb-3" id="customer-kpis"></div>
    <!-- Alarm row -->
    <div class="section-title">알림 판정</div>
    <div class="row g-3 mb-2" id="customer-alarms"></div>
    <div id="customer-alarm-msg"></div>
    <!-- Charts -->
    <div class="section-title">최근 12개월 실측 + 예측</div>
    <div id="chart-monthly" style="height:300px;"></div>
  </div>
</div>

<!-- ═══════════════════════════════════════ -->
<!-- TAB 2: 모델 성능                    -->
<!-- ═══════════════════════════════════════ -->
<div class="tab-pane fade" id="tab-fleet" role="tabpanel">
  <div id="fleet-no-data" class="no-data-msg" style="display:none;">
    <div class="icon">—</div>
    <div>데이터 없음 &mdash; 파이프라인 실행 후 재생성하세요.</div>
  </div>
  <div id="fleet-content" style="display:none;">
    <div class="row g-3 mb-3">
      <div class="col-md-3">
        <label class="form-label">검침일 기준</label>
        <div class="btn-group w-100" role="group">
          <input type="radio" class="btn-check" name="perf-hz" id="perf-hz10" value="10" checked>
          <label class="btn btn-outline-primary" for="perf-hz10">검침일+10일</label>
          <input type="radio" class="btn-check" name="perf-hz" id="perf-hz20" value="20">
          <label class="btn btn-outline-primary" for="perf-hz20">검침일+20일</label>
        </div>
      </div>
      <div class="col-md-3">
        <label class="form-label">계약종별</label>
        <select id="perf-ct" class="form-select form-select-sm">
          <option value="">전체</option>
        </select>
      </div>
    </div>
    <div class="table-responsive" id="perf-table"></div>
  </div>
</div>

<!-- ═══════════════════════════════════════ -->
<!-- TAB 3: 요금 분석                        -->
<!-- ═══════════════════════════════════════ -->
<div class="tab-pane fade" id="tab-tariff" role="tabpanel">
  <div class="row g-3 mb-3">
    <div class="col-md-3">
      <label class="form-label">계약종별</label>
      <select id="tariff-ct" class="form-select form-select-sm">
        <option value="res_low">주택용 (저압)</option>
        <option value="res_high">주택용 (고압)</option>
        <option value="gap_i_low">일반용(갑)I 저압</option>
        <option value="gap_i_high_a_1">일반용(갑)I 고압A선택I</option>
        <option value="gap_i_high_a_2">일반용(갑)I 고압A선택II</option>
        <option value="gap_i_high_b_1">일반용(갑)I 고압B선택I</option>
        <option value="gap_i_high_b_2">일반용(갑)I 고압B선택II</option>
        <option value="gap_ii_high_a_1">일반용(갑)II 고압A선택I</option>
        <option value="gap_ii_high_a_2">일반용(갑)II 고압A선택II</option>
        <option value="gap_ii_high_b_1">일반용(갑)II 고압B선택I</option>
        <option value="gap_ii_high_b_2">일반용(갑)II 고압B선택II</option>
        <option value="eul_small_low">일반용(을) 저압</option>
        <option value="eul_small_high_a_1">일반용(을) &lt;300kW 고압A-I</option>
        <option value="eul_small_high_a_2">일반용(을) &lt;300kW 고압A-II</option>
        <option value="eul_small_high_b_1">일반용(을) &lt;300kW 고압B-I</option>
        <option value="eul_small_high_b_2">일반용(을) &lt;300kW 고압B-II</option>
        <option value="eul_large_high_a_1">일반용(을) 300kW+ 고압A-I</option>
        <option value="eul_large_high_a_2">일반용(을) 300kW+ 고압A-II</option>
        <option value="eul_large_high_a_3">일반용(을) 300kW+ 고압A-III</option>
        <option value="eul_large_high_b_1">일반용(을) 300kW+ 고압B-I</option>
        <option value="eul_large_high_b_2">일반용(을) 300kW+ 고압B-II</option>
      </select>
    </div>
    <div class="col-md-2">
      <label class="form-label">월 사용량 (kWh)</label>
      <input type="number" id="tariff-kwh" class="form-control form-control-sm" value="1000" min="0" step="50">
    </div>
    <div class="col-md-1">
      <label class="form-label">월</label>
      <select id="tariff-month" class="form-select form-select-sm">
        <option value="1">1월</option><option value="2">2월</option><option value="3">3월</option>
        <option value="4">4월</option><option value="5">5월</option><option value="6">6월</option>
        <option value="7" selected>7월</option><option value="8">8월</option><option value="9">9월</option>
        <option value="10">10월</option><option value="11">11월</option><option value="12">12월</option>
      </select>
    </div>
    <div class="col-md-2">
      <label class="form-label">계약전력 (kW) / 역률 (%)</label>
      <div class="input-group input-group-sm">
        <input type="number" id="tariff-demand" class="form-control" value="0" min="0" step="10" placeholder="kW">
        <input type="number" id="tariff-pf" class="form-control" value="" min="60" max="100" step="1" placeholder="역률%">
      </div>
    </div>
    <div class="col-md-2">
      <label class="form-label">기후환경 / 연료비 (원/kWh)</label>
      <div class="input-group input-group-sm">
        <input type="number" id="tariff-climate" class="form-control" value="9.0" step="0.1" min="0">
        <input type="number" id="tariff-fuel" class="form-control" value="5.0" step="0.1" min="-5" max="5">
      </div>
    </div>
    <div class="col-md-2">
      <label class="form-label">복지할인</label>
      <select id="tariff-welfare" class="form-select form-select-sm">
        <option value="없음">없음</option>
        <option value="기초생활수급자">기초생활수급자</option>
        <option value="차상위계층">차상위계층</option>
        <option value="장애인">장애인</option>
        <option value="독립유공자">독립유공자</option>
        <option value="다자녀(3자녀+)">다자녀(3자녀+)</option>
        <option value="대가족(5인+)">대가족(5인+)</option>
        <option value="출산가구">출산가구</option>
        <option value="사회복지시설">사회복지시설</option>
      </select>
    </div>
  </div>

  <div class="alert alert-secondary py-1 px-3 mb-2" style="font-size:0.82rem; opacity:0.8;">
    ※ 검침일 1일 기준 계산 (검침일에 따른 계절 혼합 미적용 — 향후 개선 예정)
  </div>
  <!-- Tariff KPI -->
  <div class="row g-3 mb-3" id="tariff-kpis"></div>

  <div class="row g-3">
    <div class="col-md-6">
      <div class="section-title">단계별 청구 내역</div>
      <div class="table-responsive" id="tariff-breakdown"></div>
      <div class="section-title mt-4">사용량 증가별 요금 변동</div>
      <div class="table-responsive" id="tariff-increase"></div>
    </div>
    <div class="col-md-6">
      <div class="section-title">계약종별 비교 (동일 사용량)</div>
      <div class="table-responsive" id="tariff-compare"></div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════ -->
<!-- TAB 4: 모델 상세                        -->
<!-- ═══════════════════════════════════════ -->
<div class="tab-pane fade" id="tab-model" role="tabpanel">
  <div class="row g-3">
    <div class="col-md-6">
      <div class="section-title">SHAP 피처 중요도</div>
      <div id="model-shap"></div>
    </div>
    <div class="col-md-6">
      <div class="section-title">피처 그룹 기여도 (제거 실험)</div>
      <div class="table-responsive" id="model-ablation"></div>
    </div>
  </div>
  <div class="section-title mt-4">Ablation 전체 결과 (BTM / 기상 / 기준온도 / 피처선택)</div>
  <div class="table-responsive" id="model-ablation-full"></div>
  <div class="section-title mt-4">잔차 보정 (2-Stage Ridge) 전후 비교</div>
  <div class="table-responsive" id="model-residual"></div>
</div>

</div><!-- /tab-content -->
</div><!-- /container -->

<!-- ═══ Inline JS Libraries ═══ -->
<script>/* __ECHARTS_JS__ */</script>
<script>/* __BOOTSTRAP_JS__ */</script>

<!-- ═══ Embedded Data ═══ -->
<script>
const DATA = __JSON_PAYLOAD__;
</script>

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- Application JS                                             -->
<!-- ═══════════════════════════════════════════════════════════ -->
<script>
"use strict";

// ──────────────────────────────────────
// Tariff Engine (JS port of tariff.py)
// ──────────────────────────────────────

const TARIFF = (function() {

  // -- 주택용 저압 --
  // 단가는 연중 동일, 하계(7~8)에 구간만 확대 (300/450)
  const RES_LOW = {
    normal: [{upper:200, base:910, rate:120.0},{upper:400, base:1600, rate:214.6},{upper:Infinity, base:7300, rate:307.3}],
    summer: [{upper:300, base:910, rate:120.0},{upper:450, base:1600, rate:214.6},{upper:Infinity, base:7300, rate:307.3}]
  };
  const RES_HIGH = {
    normal: [{upper:200, base:730, rate:105.0},{upper:400, base:1260, rate:174.0},{upper:Infinity, base:6060, rate:242.3}],
    summer: [{upper:300, base:730, rate:105.0},{upper:450, base:1260, rate:174.0},{upper:Infinity, base:6060, rate:242.3}]
  };

  // -- 일반용(갑)I --
  const GAP_I = {
    low:      {base_kw:5230, summer:123.6, spring_fall:86.4, winter:110.8},
    high_a_1: {base_kw:5550, summer:123.3, spring_fall:86.5, winter:109.3},
    high_a_2: {base_kw:6370, summer:118.8, spring_fall:82.1, winter:104.8},
    high_b_1: {base_kw:5550, summer:122.6, spring_fall:86.1, winter:108.5},
    high_b_2: {base_kw:6370, summer:118.1, spring_fall:81.6, winter:104.0}
  };

  // -- 일반용(갑)II TOU --
  const GAP_II = {
    high_a_1: {base_kw:6090, summer:{off:76.5,mid:121.2,on:187.1}, spring_fall:{off:76.5,mid:90.9,on:111.4}, winter:{off:80.5,mid:119.7,on:158.4}},
    high_a_2: {base_kw:6980, summer:{off:72.0,mid:116.7,on:182.6}, spring_fall:{off:72.0,mid:86.4,on:106.9}, winter:{off:76.0,mid:115.2,on:153.9}},
    high_b_1: {base_kw:6090, summer:{off:75.0,mid:118.5,on:181.4}, spring_fall:{off:75.0,mid:89.2,on:109.0}, winter:{off:78.8,mid:116.8,on:154.1}},
    high_b_2: {base_kw:6980, summer:{off:70.5,mid:114.0,on:176.9}, spring_fall:{off:70.5,mid:84.7,on:104.5}, winter:{off:74.3,mid:112.3,on:149.6}}
  };

  // -- 일반용(을) <300kW 단일단가 --
  const EUL_SMALL = {
    low:      {base_kw:6160, summer:132.4, spring_fall:91.9, winter:119.0},
    high_a_1: {base_kw:7170, summer:142.6, spring_fall:98.6, winter:130.3},
    high_a_2: {base_kw:8230, summer:138.6, spring_fall:94.3, winter:125.0},
    high_b_1: {base_kw:7170, summer:140.5, spring_fall:97.5, winter:127.3},
    high_b_2: {base_kw:8230, summer:135.2, spring_fall:92.2, winter:122.0}
  };

  // -- 일반용(을) <300kW TOU --
  const EUL_SMALL_TOU = {
    high_a_1: {base_kw:7170, summer:{off:89.4,mid:140.6,on:163.1}, spring_fall:{off:89.4,mid:96.8,on:108.1}, winter:{off:98.1,mid:128.5,on:143.3}},
    high_a_2: {base_kw:8230, summer:{off:84.1,mid:135.3,on:157.8}, spring_fall:{off:84.1,mid:91.5,on:102.8}, winter:{off:92.8,mid:123.2,on:138.0}},
    high_b_1: {base_kw:7170, summer:{off:88.8,mid:137.4,on:153.8}, spring_fall:{off:88.8,mid:94.7,on:100.1}, winter:{off:97.8,mid:125.1,on:139.3}},
    high_b_2: {base_kw:8230, summer:{off:83.5,mid:132.1,on:148.5}, spring_fall:{off:83.5,mid:89.4,on:94.8}, winter:{off:92.5,mid:119.8,on:134.0}}
  };

  // -- 일반용(을) 300kW+ TOU --
  const EUL_LARGE = {
    high_a_1: {base_kw:7220, summer:{off:92.8,mid:145.7,on:227.8}, spring_fall:{off:92.8,mid:115.3,on:146.0}, winter:{off:99.8,mid:145.9,on:203.4}},
    high_a_2: {base_kw:8320, summer:{off:87.3,mid:140.2,on:222.3}, spring_fall:{off:87.3,mid:109.8,on:140.5}, winter:{off:94.3,mid:140.4,on:197.9}},
    high_a_3: {base_kw:9810, summer:{off:86.4,mid:139.6,on:209.9}, spring_fall:{off:86.4,mid:108.5,on:132.2}, winter:{off:93.7,mid:139.8,on:186.7}},
    high_b_1: {base_kw:6630, summer:{off:95.9,mid:148.2,on:229.4}, spring_fall:{off:95.9,mid:118.2,on:148.5}, winter:{off:102.9,mid:148.2,on:204.4}},
    high_b_2: {base_kw:7380, summer:{off:92.1,mid:144.4,on:225.6}, spring_fall:{off:92.1,mid:114.4,on:144.7}, winter:{off:99.1,mid:144.4,on:200.6}}
  };

  // -- 복지할인 --
  const WELFARE = {
    "없음":               {rate:0,   cap:0,     capS:0,     grp:null, dup:false},
    "기초생활(생계/의료)": {rate:1.0, cap:16000, capS:20000, grp:"2",  dup:true},
    "기초생활(주거/교육)": {rate:1.0, cap:10000, capS:12000, grp:"2",  dup:true},
    "차상위계층":          {rate:1.0, cap:8000,  capS:10000, grp:"2",  dup:true},
    "장애인(심한)":        {rate:1.0, cap:16000, capS:20000, grp:"2",  dup:false},
    "국가유공자":          {rate:1.0, cap:16000, capS:20000, grp:"2",  dup:false},
    "독립유공자":          {rate:1.0, cap:16000, capS:20000, grp:"2",  dup:false},
    "사회복지시설":        {rate:0.3, cap:0,     capS:0,     grp:"3",  dup:false},
    "다자녀(3자녀+)":      {rate:0.3, cap:16000, capS:16000, grp:"4",  dup:false},
    "대가족(5인+)":        {rate:0.3, cap:16000, capS:16000, grp:"4",  dup:false},
    "출산가구":            {rate:0.3, cap:16000, capS:16000, grp:"4",  dup:false},
    "생명유지장치":        {rate:0.3, cap:0,     capS:0,     grp:"4",  dup:false}
  };

  const VAT_RATE = 0.10;
  const FUND_RATE = 0.027;

  function getSeason(month) {
    if (month >= 6 && month <= 8) return "summer";
    if ((month >= 3 && month <= 5) || (month >= 9 && month <= 10)) return "spring_fall";
    return "winter";
  }

  function progressiveBill(kwh, tiers, month, ctKey) {
    let remaining = kwh, energy = 0, base = 0, tierIdx = 0, prevUpper = 0;
    for (let i = 0; i < tiers.length; i++) {
      const t = tiers[i];
      const band = t.upper - prevUpper;
      const usage = Math.min(remaining, band);
      if (usage > 0) { energy += usage * t.rate; base = t.base; tierIdx = i + 1; }
      remaining -= usage;
      prevUpper = t.upper;
      if (remaining <= 0) break;
    }
    // 슈퍼유저: 하계(7,8) 또는 동계(12,1,2)에 1000kWh 초과 시 추가 단가
    let superExtra = 0;
    const isSuperSeason = [7, 8, 12, 1, 2].includes(month);
    if (isSuperSeason && kwh > 1000) {
      const superRate = (ctKey === "res_high") ? 601.3 : 736.2;
      const lastTierRate = tiers[tiers.length - 1].rate;
      superExtra = (kwh - 1000) * (superRate - lastTierRate);
      energy += superExtra;
    }
    const total = base + energy;
    return {base_won: base, energy_won: Math.round(energy), elec_won: Math.round(total),
            effective_rate: kwh > 0 ? +(total/kwh).toFixed(1) : 0, tier: tierIdx, tariff_type: "progressive",
            super_user_extra: Math.round(superExtra)};
  }

  function seasonalFlatBill(kwh, rates, season, demandKw) {
    const rate = rates[season];
    const energy = kwh * rate;
    const base = demandKw * rates.base_kw;
    const total = base + energy;
    return {base_won: Math.round(base), energy_won: Math.round(energy), elec_won: Math.round(total),
            effective_rate: kwh > 0 ? +(total/kwh).toFixed(1) : 0, tariff_type: "seasonal_flat"};
  }

  function touBill(kwh, rates, season, demandKw) {
    const sr = rates[season];
    // default TOU ratios: off 30%, mid 40%, on 30%
    const energy = kwh * (0.3 * sr.off + 0.4 * sr.mid + 0.3 * sr.on);
    const base = demandKw * rates.base_kw;
    const total = base + energy;
    return {base_won: Math.round(base), energy_won: Math.round(energy), elec_won: Math.round(total),
            effective_rate: kwh > 0 ? +(total/kwh).toFixed(1) : 0, tariff_type: "tou"};
  }

  function calcElec(kwh, ctKey, month, demandKw) {
    const season = getSeason(month);
    const summerFlag = (month === 7 || month === 8) ? "summer" : "normal";

    if (ctKey === "res_low")  return progressiveBill(kwh, RES_LOW[summerFlag], month, ctKey);
    if (ctKey === "res_high") return progressiveBill(kwh, RES_HIGH[summerFlag], month, ctKey);

    // 일반용(갑)I
    if (ctKey.startsWith("gap_i_")) {
      const sub = ctKey.replace("gap_i_","");
      return seasonalFlatBill(kwh, GAP_I[sub] || GAP_I.low, season, demandKw);
    }
    // 일반용(갑)II TOU
    if (ctKey.startsWith("gap_ii_")) {
      const sub = ctKey.replace("gap_ii_","");
      return touBill(kwh, GAP_II[sub] || GAP_II.high_a_1, season, demandKw);
    }
    // 일반용(을) <300kW
    if (ctKey.startsWith("eul_small_")) {
      const sub = ctKey.replace("eul_small_","");
      if (sub.startsWith("high_") && EUL_SMALL_TOU[sub]) {
        return touBill(kwh, EUL_SMALL_TOU[sub], season, demandKw);
      }
      return seasonalFlatBill(kwh, EUL_SMALL[sub] || EUL_SMALL.low, season, demandKw);
    }
    // 일반용(을) 300kW+
    if (ctKey.startsWith("eul_large_")) {
      const sub = ctKey.replace("eul_large_","");
      return touBill(kwh, EUL_LARGE[sub] || EUL_LARGE.high_a_1, season, demandKw);
    }
    // fallback
    return seasonalFlatBill(kwh, EUL_SMALL.low, season, demandKw);
  }

  function calcWelfare(elecWon, welfareType, isSummer) {
    const w = WELFARE[welfareType];
    if (!w || w.rate === 0) return 0;
    const cap = isSummer ? w.capS : w.cap;
    let disc = elecWon * w.rate;
    if (cap > 0) disc = Math.min(disc, cap);
    return Math.round(disc);
  }

  function calcPfAdjustment(baseWon, pf) {
    if (pf == null || pf <= 0) return 0;
    pf = Math.min(pf, 1.0);
    const diff = Math.round((pf - 0.90) * 100);
    if (diff === 0) return 0;
    return Math.round(baseWon * 0.002 * diff);
  }

  function finalBill(kwh, ctKey, month, demandKw, climateRate, fuelRate, welfareType, pf) {
    const elec = calcElec(kwh, ctKey, month, demandKw || 0);
    const isResidential = ctKey.startsWith("res");
    const pfAdj = (!isResidential && pf) ? calcPfAdjustment(elec.base_won, pf) : 0;
    const climateWon = Math.round(climateRate * kwh);
    const fuelWon = Math.round(fuelRate * kwh);
    const isSummer = (month === 7 || month === 8);
    const welfareDiscount = calcWelfare(elec.elec_won, welfareType || "없음", isSummer);
    let subtotal = elec.elec_won - pfAdj + climateWon + fuelWon - welfareDiscount;
    subtotal = Math.max(subtotal, 0);
    const vat = Math.round(subtotal * VAT_RATE);
    const fund = Math.floor(Math.floor(subtotal * FUND_RATE) / 10) * 10;
    const finalWon = subtotal + vat + fund;
    return {
      ...elec, pf_adjustment: pfAdj, climate_won: climateWon, fuel_won: fuelWon,
      welfare_discount: welfareDiscount, subtotal: subtotal,
      vat: vat, fund: fund, final_won: finalWon
    };
  }

  // 계약종별 비교용 키-레이블 매핑
  const CT_LABELS = {
    res_low: "주택용(저압)", res_high: "주택용(고압)",
    gap_i_low: "일반용(갑)I 저압",
    gap_i_high_a_1: "일반용(갑)I 고A-I", gap_i_high_a_2: "일반용(갑)I 고A-II",
    gap_i_high_b_1: "일반용(갑)I 고B-I", gap_i_high_b_2: "일반용(갑)I 고B-II",
    gap_ii_high_a_1: "일반용(갑)II 고A-I", gap_ii_high_a_2: "일반용(갑)II 고A-II",
    gap_ii_high_b_1: "일반용(갑)II 고B-I", gap_ii_high_b_2: "일반용(갑)II 고B-II",
    eul_small_low: "일반용(을) 저압",
    eul_small_high_a_1: "일반용(을)<300 고A-I", eul_small_high_a_2: "일반용(을)<300 고A-II",
    eul_small_high_b_1: "일반용(을)<300 고B-I", eul_small_high_b_2: "일반용(을)<300 고B-II",
    eul_large_high_a_1: "일반용(을)300+ 고A-I", eul_large_high_a_2: "일반용(을)300+ 고A-II",
    eul_large_high_a_3: "일반용(을)300+ 고A-III",
    eul_large_high_b_1: "일반용(을)300+ 고B-I", eul_large_high_b_2: "일반용(을)300+ 고B-II"
  };

  const COMPARE_KEYS = Object.keys(CT_LABELS);

  return { finalBill, getSeason, CT_LABELS, COMPARE_KEYS, WELFARE, VAT_RATE, FUND_RATE };
})();


// ──────────────────────────────────────
// Utility helpers
// ──────────────────────────────────────

// Theme toggle — destroy and rebuild charts so colors update
function toggleTheme() {
  const html = document.documentElement;
  const cur = html.getAttribute("data-theme") || "dark";
  const next = cur === "dark" ? "light" : "dark";
  html.setAttribute("data-theme", next);
  html.setAttribute("data-bs-theme", next);
  el("theme-toggle").textContent = next === "dark" ? "Light" : "Dark";
  // re-init the currently visible tab's charts with new theme colors
  const activeHref = document.querySelector(".nav-tabs .nav-link.active");
  if (!activeHref) return;
  const target = activeHref.getAttribute("href");
  const reinit = {
    "#tab-customer": CustomerView.init,
    "#tab-fleet": FleetView.init,
    "#tab-tariff": TariffView.init,
    "#tab-model": ModelView.init,
  };
  if (reinit[target]) reinit[target]();
}

function isVal(v) { return v != null && !isNaN(v); }
const MODEL_KR = {
  "lightgbm": "제안 모델", "naive_last_month": "전월 동일",
  "seasonal_yoy": "전년 동월", "moving_avg3": "3개월 이동평균",
  "partial_linear": "부분 선형 보간", "partial_seasonal_yoy": "부분 전년 동월"
};
function modelKr(name) { return MODEL_KR[name] || name; }
function fmt(n, dec) { return !isVal(n) ? "—" : Number(n).toLocaleString("ko-KR", {maximumFractionDigits: dec||0}); }
function fmtPct(n) { return !isVal(n) ? "—" : (n > 0 ? "+" : "") + n.toFixed(1) + "%"; }
function el(id) { return document.getElementById(id); }
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function makeKpi(label, value, delta, deltaDir) {
  const dc = deltaDir === "up" ? "delta-up" : deltaDir === "down" ? "delta-down" : "delta-neutral";
  return '<div class="col-md-3 col-6"><div class="kpi-card">' +
    '<div class="kpi-label">' + label + '</div>' +
    '<div class="kpi-value">' + value + '</div>' +
    (delta ? '<div class="kpi-delta ' + dc + '">' + delta + '</div>' : '') +
    '</div></div>';
}

// Echarts theme — reads CSS variables so it adapts to dark/light
function isDark() { return (document.documentElement.getAttribute("data-theme") || "dark") === "dark"; }
function echartsTheme() {
  const textCol = cssVar("--text-secondary");
  const borderCol = cssVar("--card-border");
  const splitCol = isDark() ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.06)";
  return {
    backgroundColor: "transparent",
    textStyle: { color: textCol },
    legend: { textStyle: { color: textCol } },
    xAxis: { axisLine: { lineStyle: { color: borderCol } }, axisLabel: { color: textCol, fontSize: 11 }, splitLine: { lineStyle: { color: splitCol } } },
    yAxis: { axisLine: { lineStyle: { color: borderCol } }, axisLabel: { color: textCol, fontSize: 11 }, splitLine: { lineStyle: { color: splitCol } } }
  };
}
function echartsColors() {
  return isDark()
    ? { main: "#3b82f6", secondary: "#4b5563", red: "#ef4444", green: "#10b981", muted: "#6b7280",
        palette: ["#3b82f6","#6366f1","#10b981","#f59e0b","#ef4444"], pie_label: "#9ca3af" }
    : { main: "#2563eb", secondary: "#94a3b8", red: "#dc2626", green: "#059669", muted: "#9ca3af",
        palette: ["#2563eb","#7c3aed","#059669","#d97706","#dc2626"], pie_label: "#4b5563" };
}

const TH_PREV = 1.30, TH_YOY = 1.30, TH_MA3 = 1.50;

// Convert columnar format to filtered records on-demand
function getCumRows(custId, yearMonth) {
  const dc = DATA.daily_cum;
  if (!dc || !dc.data) return [];
  const d = dc.data;
  const rows = [];
  for (let i = 0; i < dc.length; i++) {
    if (d.customer_id[i] === custId && d.year_month[i] === yearMonth) {
      rows.push({day_of_month: d.day_of_month[i], cum_kwh: d.cum_kwh[i], day_kwh: d.day_kwh[i]});
    }
  }
  return rows.sort((a,b) => a.day_of_month - b.day_of_month);
}

// ──────────────────────────────────────
// TAB 1: 고객 뷰
// ──────────────────────────────────────

const CustomerView = (function() {
  let chartMonthly = null;

  function init() {
    if (!DATA.monthly || !DATA.ctx || !DATA.preds) {
      el("customer-no-data").style.display = "";
      return;
    }
    el("customer-content").style.display = "";

    // populate customer dropdown
    const customers = [...new Set(DATA.monthly.map(r => r.customer_id))].sort();
    const selCust = el("sel-customer");
    customers.forEach(c => { const o = document.createElement("option"); o.value = c; o.textContent = c; selCust.appendChild(o); });

    selCust.addEventListener("change", () => { populateMonths(); render(); });
    el("sel-month").addEventListener("change", render);
    document.querySelectorAll('input[name="horizon"]').forEach(r => r.addEventListener("change", render));

    populateMonths();
    render();
  }

  function populateMonths() {
    const cust = el("sel-customer").value;
    const months = [...new Set(DATA.monthly.filter(r => r.customer_id === cust).map(r => r.year_month))].sort();
    const sel = el("sel-month");
    sel.innerHTML = "";
    months.forEach(m => { const o = document.createElement("option"); o.value = m; o.textContent = m; sel.appendChild(o); });
    if (months.length > 1) sel.selectedIndex = Math.floor(months.length / 2);
  }

  function render() {
    const cust = el("sel-customer").value;
    const ym = el("sel-month").value;
    const hz = parseInt(document.querySelector('input[name="horizon"]:checked').value);
    if (!cust || !ym) return;

    // find ctx row
    const ctxRow = DATA.ctx.find(r => r.customer_id === cust && r.year_month === ym && r.horizon_days === hz);
    if (!ctxRow) { el("customer-kpis").innerHTML = '<div class="col-12 text-secondary">선택 조합에 데이터 없음</div>'; return; }

    // find prediction
    const predRow = DATA.preds.find(r => r.customer_id === cust && r.year_month === ym && r.horizon_days === hz && r.model === "partial_linear");
    const predKwh = predRow ? predRow.pred_monthly_kwh : null;

    const prev = ctxRow.prev_month_kwh;
    const yoy = ctxRow.yoy_month_kwh;
    const ma3 = ctxRow.ma3_kwh;

    // 고객 뷰 선택 → 전역 저장 (요금 탭 연동)
    window._customerContext = {
      predKwh: predKwh,
      month: ym ? parseInt(ym.split("-")[1]) : 1,
      contractType: ctxRow.contract_type || null,
      contractPower: ctxRow.contract_power_kw || 0,
      powerFactor: ctxRow.observed_power_factor || null,
    };

    // 실측값 + 오차 계산
    const actual = ctxRow.full_month_kwh;
    let mapeVal = null;
    let errKwh = null;
    if (isVal(predKwh) && isVal(actual) && actual > 0) {
      errKwh = predKwh - actual;
      mapeVal = Math.abs(errKwh) / actual * 100;
    }

    // KPIs (5개 균등: 예측, 실측, 오차, 전월, 3개월평균)
    function kpi5(label, value, delta, deltaDir) {
      const dc = deltaDir === "up" ? "delta-up" : deltaDir === "down" ? "delta-down" : "delta-neutral";
      return '<div class="col"><div class="kpi-card">' +
        '<div class="kpi-label">' + label + '</div>' +
        '<div class="kpi-value">' + value + '</div>' +
        (delta ? '<div class="kpi-delta ' + dc + '">' + delta + '</div>' : '<div class="kpi-delta">&nbsp;</div>') +
        '</div></div>';
    }
    let kpiHtml = kpi5("예측 사용량", isVal(predKwh) ? fmt(predKwh) + " kWh" : "—");
    kpiHtml += kpi5("실측 사용량", isVal(actual) ? fmt(actual) + " kWh" : "—");
    if (isVal(mapeVal)) {
      const errDir = errKwh > 0 ? "up" : "down";
      kpiHtml += kpi5("예측 오차 (MAPE)", mapeVal.toFixed(1) + "%",
        (errKwh > 0 ? "+" : "") + fmt(Math.round(errKwh)) + " kWh", errDir);
    } else {
      kpiHtml += kpi5("예측 오차 (MAPE)", "—");
    }
    if (isVal(prev) && prev > 0 && isVal(predKwh)) {
      const d = ((predKwh/prev)-1)*100;
      kpiHtml += kpi5("전월", fmt(prev) + " kWh", fmtPct(d), d > 0 ? "up" : "down");
    } else {
      kpiHtml += kpi5("전월", isVal(prev) ? fmt(prev) + " kWh" : "—");
    }
    el("customer-kpis").innerHTML = kpiHtml;

    // Alarms
    const condPrev = (isVal(predKwh) && isVal(prev)) ? predKwh > TH_PREV * prev : false;
    const condYoy  = (isVal(predKwh) && isVal(yoy))  ? predKwh > TH_YOY * yoy : false;
    const condMa   = (isVal(predKwh) && isVal(ma3))  ? predKwh > TH_MA3 * ma3 : false;
    const any3 = condPrev || condYoy || condMa;

    const alarms = [
      {label:"전월 대비 +30%", cond:condPrev, rhs: isVal(prev) ? (prev*TH_PREV) : null},
      {label:"전년 동월 대비 +30%", cond:condYoy, rhs: isVal(yoy) ? (yoy*TH_YOY) : null},
      {label:"3개월 평균 대비 +50%", cond:condMa, rhs: isVal(ma3) ? (ma3*TH_MA3) : null}
    ];
    let alarmHtml = "";
    alarms.forEach(a => {
      const cls = a.cond ? "alarm-danger" : "alarm-safe";
      const icon = a.cond ? "ALERT" : "OK";
      const detail = (isVal(a.rhs) && isVal(predKwh))
        ? "예측 " + fmt(predKwh) + " vs 임계 " + fmt(a.rhs)
        : "기준 데이터 부족";
      alarmHtml += '<div class="col-md-4"><div class="kpi-card text-start">' +
        '<div class="kpi-label">' + a.label + '</div>' +
        '<span class="alarm-badge ' + cls + '">' + icon + '</span>' +
        '<div style="color:var(--text-secondary);font-size:0.78rem;margin-top:8px;">' + detail + '</div>' +
        '</div></div>';
    });
    el("customer-alarms").innerHTML = alarmHtml;

    if (any3) {
      el("customer-alarm-msg").innerHTML = '<div class="warn-box"><strong>알림 발송 대상</strong> &mdash; 예측 사용량이 3조건 중 1건 이상 초과</div>';
    } else {
      el("customer-alarm-msg").innerHTML = '<div class="tip-box">현재 평균적 소비 범위 내</div>';
    }

    renderMonthlyChart(cust, ym, predKwh, any3);
  }

  function renderMonthlyChart(cust, ym, predKwh, isAlert) {
    if (!DATA.monthly) return;
    const hz = parseInt(document.querySelector('input[name="horizon"]:checked').value);

    // 이 고객의 전체 월별 데이터
    const allMonthly = DATA.monthly.filter(r => r.customer_id === cust)
      .sort((a,b) => a.year_month.localeCompare(b.year_month));
    if (!allMonthly.length) return;

    // 이 고객의 전체 예측 데이터 (partial_linear, 선택된 horizon)
    const allPreds = DATA.preds.filter(r => r.customer_id === cust && r.model === "partial_linear" && r.horizon_days === hz);

    // 최근 12개월 실측 + 예측이 있는 월만 표시
    const recent = allMonthly.slice(-12);
    const labels = recent.map(r => r.year_month);
    const actualData = recent.map(r => r.monthly_kwh);
    const predData = labels.map(m => {
      const p = allPreds.find(pp => pp.year_month === m);
      return p ? p.pred_monthly_kwh : null;
    });

    // 오차(%) 라벨 데이터
    const mapeLabels = actualData.map((a, i) => {
      const p = predData[i];
      if (!isVal(a) || !isVal(p) || a <= 0) return "";
      const pct = ((p - a) / a * 100);
      return (pct >= 0 ? "+" : "") + pct.toFixed(1) + "%";
    });

    const C2 = echartsColors();
    const T2 = echartsTheme();
    if (chartMonthly) { chartMonthly.dispose(); } chartMonthly = echarts.init(el("chart-monthly"));
    chartMonthly.setOption({
      ...T2,
      tooltip: { trigger: "axis" },
      legend: { ...T2.legend, top: 4, right: 10 },
      grid: { left: 70, right: 35, top: 40, bottom: 70, containLabel: false },
      xAxis: { ...T2.xAxis, type: "category", data: labels, axisLabel: { ...T2.xAxis.axisLabel, rotate: 45, fontSize: 10 } },
      yAxis: { ...T2.yAxis, type: "value", name: "kWh", nameLocation: "center", nameGap: 50 },
      series: [
        { name: "실측", type: "bar", data: actualData, itemStyle: { color: C2.secondary }, barGap: "10%" },
        { name: "예측", type: "bar", data: predData, itemStyle: { color: isAlert ? C2.red : C2.main }, barGap: "10%",
          label: {
            show: true, position: "top", fontSize: 10,
            color: cssVar("--text-secondary"),
            formatter: function(params) { return mapeLabels[params.dataIndex] || ""; }
          }
        }
      ]
    }, true);
  }

  function _chartReset() { chartMonthly = null; }
  return { init, _chartReset };
})();


// ──────────────────────────────────────
// TAB 2: 모델 성능
// ──────────────────────────────────────

const FleetView = (function() {
  let _listenersAttached = false;

  function buildTable(perfData, horizon, ctFilter) {
    let rows = perfData.filter(r => r.horizon_days === horizon);
    if (ctFilter) rows = rows.filter(r => r.contract_type === ctFilter);
    if (!rows.length) return '<div class="text-secondary">해당 조건의 데이터 없음</div>';

    let tbl = '<table class="table table-sm table-hover mb-0"><thead><tr>' +
      '<th>모델</th>' + (ctFilter ? '' : '<th>계약종</th>') +
      '<th>MAE (kWh)</th><th>RMSE (kWh)</th><th>CVRMSE (%)</th><th>MAPE (%)</th><th>알림 F1</th>' +
      '</tr></thead><tbody>';
    rows.forEach(r => {
      const isBest = (r.model || "").includes("lightgbm");
      const style = isBest ? ' style="font-weight:700;background:rgba(59,130,246,0.08);"' : '';
      tbl += '<tr' + style + '>' +
        '<td>' + modelKr(r.model) + '</td>' +
        (ctFilter ? '' : '<td>' + (r.contract_type || "전체") + '</td>') +
        '<td>' + (r.mae != null ? r.mae.toFixed(1) : "—") + '</td>' +
        '<td>' + (r.rmse != null ? r.rmse.toFixed(1) : "—") + '</td>' +
        '<td>' + (r.cvrmse_pct != null ? r.cvrmse_pct.toFixed(2) : "—") + '</td>' +
        '<td>' + (r.mape_pct != null ? r.mape_pct.toFixed(2) : "—") + '</td>' +
        '<td>' + (r.alarm_f1 != null ? r.alarm_f1.toFixed(3) : "—") + '</td>' +
        '</tr>';
    });
    tbl += '</tbody></table>';
    return tbl;
  }

  function _getPerfData() {
    // lgbm_vs_baselines에 F1이 있으므로 우선, metrics에서 cvrmse 보충
    const base = DATA.eval_lgbm_vs_baselines || DATA.metrics;
    if (!base) return null;
    if (DATA.metrics && base !== DATA.metrics) {
      return base.map(r => {
        if (!isVal(r.cvrmse_pct)) {
          const m = DATA.metrics.find(mm => mm.model === r.model && mm.horizon_days === r.horizon_days && mm.contract_type === r.contract_type);
          if (m && isVal(m.cvrmse_pct)) return {...r, cvrmse_pct: m.cvrmse_pct};
        }
        return r;
      });
    }
    return base;
  }

  function render() {
    const perfData = _getPerfData();
    if (!perfData) return;
    const hz = parseInt(document.querySelector('input[name="perf-hz"]:checked').value);
    const ct = el("perf-ct").value;
    el("perf-table").innerHTML = buildTable(perfData, hz, ct || null);
  }

  function init() {
    if (!DATA.metrics && !DATA.eval_lgbm_vs_baselines) {
      el("fleet-no-data").style.display = "";
      return;
    }
    el("fleet-content").style.display = "";

    // 계약종 드롭다운 채우기
    const perfData = _getPerfData();
    const ctSet = new Set();
    perfData.forEach(r => { if (r.contract_type) ctSet.add(r.contract_type); });
    const sel = el("perf-ct");
    if (sel.options.length <= 1) {
      [...ctSet].sort().forEach(ct => {
        const o = document.createElement("option");
        o.value = ct; o.textContent = ct;
        sel.appendChild(o);
      });
    }

    if (!_listenersAttached) {
      document.querySelectorAll('input[name="perf-hz"]').forEach(r => r.addEventListener("change", render));
      sel.addEventListener("change", render);
      _listenersAttached = true;
    }

    render();
  }

  return { init };
})();


// ──────────────────────────────────────
// TAB 3: 요금 분석
// ──────────────────────────────────────

const TariffView = (function() {

  let _listenersAttached = false;

  function init() {
    if (!_listenersAttached) {
      ["tariff-ct","tariff-kwh","tariff-month","tariff-demand","tariff-pf","tariff-climate","tariff-fuel","tariff-welfare"].forEach(id => {
        el(id).addEventListener("change", render);
        el(id).addEventListener("input", render);
      });
      _listenersAttached = true;
    }

    // 고객 뷰에서 넘어온 값이 있으면 자동 채우기
    const ctx = window._customerContext;
    if (ctx) {
      if (isVal(ctx.predKwh) && ctx.predKwh > 0) {
        el("tariff-kwh").value = Math.round(ctx.predKwh);
      }
      if (ctx.month >= 1 && ctx.month <= 12) {
        el("tariff-month").value = ctx.month;
      }
      if (isVal(ctx.contractPower) && ctx.contractPower > 0) {
        el("tariff-demand").value = Math.round(ctx.contractPower);
      }
      if (isVal(ctx.powerFactor) && ctx.powerFactor > 0) {
        el("tariff-pf").value = Math.round(ctx.powerFactor * 100);
      }
    }

    render();
  }

  function getInputs() {
    return {
      ct: el("tariff-ct").value,
      kwh: parseFloat(el("tariff-kwh").value) || 0,
      month: parseInt(el("tariff-month").value),
      demand: parseFloat(el("tariff-demand").value) || 0,
      pf: el("tariff-pf").value ? parseFloat(el("tariff-pf").value) / 100 : null,
      climate: parseFloat(el("tariff-climate").value) || 9.0,
      fuel: parseFloat(el("tariff-fuel").value) || 5.0,
      welfare: el("tariff-welfare").value
    };
  }

  function render() {
    const inp = getInputs();
    const bill = TARIFF.finalBill(inp.kwh, inp.ct, inp.month, inp.demand, inp.climate, inp.fuel, inp.welfare, inp.pf);

    // KPIs
    const surcharge = bill.climate_won + bill.fuel_won;
    const effRate = inp.kwh > 0 ? (bill.final_won / inp.kwh).toFixed(1) : 0;
    let kpi = makeKpi("전기요금", fmt(bill.elec_won) + "원", "기본 " + fmt(bill.base_won) + " + 전력량 " + fmt(bill.energy_won), "neutral");
    kpi += makeKpi("부과금", fmt(surcharge) + "원", "기후환경 " + fmt(bill.climate_won) + " + 연료비 " + fmt(bill.fuel_won), "neutral");
    if (bill.welfare_discount > 0) {
      kpi += makeKpi("복지할인", "-" + fmt(bill.welfare_discount) + "원", inp.welfare, "down");
    } else {
      kpi += makeKpi("복지할인", "해당 없음");
    }
    kpi += makeKpi("최종 청구금액", fmt(bill.final_won) + "원", "실효단가 " + effRate + "원/kWh", "neutral");
    el("tariff-kpis").innerHTML = kpi;

    // Breakdown table
    const ctLabel = el("tariff-ct").selectedOptions[0].text;
    let rows = [
      ["&#9312; 기본요금", fmt(bill.base_won) + "원", ctLabel],
      ["&#9313; 전력량요금", fmt(bill.energy_won) + "원", bill.effective_rate ? "실효 " + bill.effective_rate + "원/kWh" : ""],
    ];
    if (bill.pf_adjustment && bill.pf_adjustment !== 0) {
      const pfSign = bill.pf_adjustment > 0 ? "할인 -" : "할증 +";
      rows.push(["&#9314; 역률 조정", pfSign + fmt(Math.abs(bill.pf_adjustment)) + "원",
        "역률 " + (inp.pf ? (inp.pf * 100).toFixed(0) + "%" : "—")]);
    }
    rows.push(
      ["&#9315; 기후환경요금", fmt(bill.climate_won) + "원", inp.climate + "원 &times; " + fmt(inp.kwh) + "kWh"],
      ["&#9316; 연료비조정요금", fmt(bill.fuel_won) + "원", inp.fuel + "원 &times; " + fmt(inp.kwh) + "kWh"],
    );
    if (bill.welfare_discount > 0) {
      rows.push(["&#9317; 복지할인", "-" + fmt(bill.welfare_discount) + "원", inp.welfare]);
    }
    rows.push(["소계", fmt(bill.subtotal) + "원", ""]);
    rows.push(["&#9317; 부가가치세", fmt(bill.vat) + "원", "10%"]);
    rows.push(["&#9318; 전력산업기반기금", fmt(bill.fund) + "원", "2.7%"]);
    rows.push(["<strong>최종 청구금액</strong>", "<strong>" + fmt(bill.final_won) + "원</strong>", ""]);

    let tbl = '<table class="table table-sm table-hover mb-0"><thead><tr><th>단계</th><th>금액</th><th>비고</th></tr></thead><tbody>';
    rows.forEach(r => { tbl += '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td><td>' + r[2] + '</td></tr>'; });
    tbl += '</tbody></table>';
    el("tariff-breakdown").innerHTML = tbl;

    // Increase table
    const pcts = [10, 20, 30, 50, 80];
    let incTbl = '<table class="table table-sm table-hover mb-0"><thead><tr><th>증가</th><th>사용량</th><th>청구금액</th><th>차액</th><th>요금증가</th><th>증폭</th></tr></thead><tbody>';
    pcts.forEach(pct => {
      const pKwh = inp.kwh * (1 + pct / 100);
      const pBill = TARIFF.finalBill(pKwh, inp.ct, inp.month, inp.demand, inp.climate, inp.fuel, inp.welfare, inp.pf);
      const wonChg = bill.final_won > 0 ? ((pBill.final_won / bill.final_won - 1) * 100) : 0;
      const amp = pct > 0 ? (wonChg / pct) : 0;
      incTbl += '<tr><td>+' + pct + '%</td><td>' + fmt(pKwh) + '</td><td>' + fmt(pBill.final_won) + '원</td>' +
        '<td>+' + fmt(pBill.final_won - bill.final_won) + '원</td><td>+' + wonChg.toFixed(1) + '%</td>' +
        '<td>' + amp.toFixed(2) + 'x</td></tr>';
    });
    incTbl += '</tbody></table>';
    el("tariff-increase").innerHTML = incTbl;

    // Compare table — full list with color-coded diff and bold for current
    let cmpTbl = '<table class="table table-sm table-hover mb-0"><thead><tr><th>계약종별</th><th>전기요금</th><th>최종청구</th><th>실효단가</th><th>차액</th></tr></thead><tbody>';
    TARIFF.COMPARE_KEYS.forEach(ck => {
      const b = TARIFF.finalBill(inp.kwh, ck, inp.month, inp.demand, inp.climate, inp.fuel, inp.welfare, inp.pf);
      const diff = b.final_won - bill.final_won;
      const isCurrent = (ck === inp.ct);
      const eff = inp.kwh > 0 ? (b.final_won / inp.kwh).toFixed(1) : "—";
      let diffCell;
      if (isCurrent) {
        diffCell = '<span style="color:var(--text-secondary);">기준</span>';
      } else if (diff > 0) {
        diffCell = '<span style="color:var(--accent-red);font-weight:600;">+' + fmt(diff) + '</span>';
      } else if (diff < 0) {
        diffCell = '<span style="color:var(--accent-green);font-weight:600;">' + fmt(diff) + '</span>';
      } else {
        diffCell = '<span style="color:var(--text-secondary);">0</span>';
      }
      const rowStyle = isCurrent ? ' style="font-weight:700;background:rgba(59,130,246,0.08);"' : '';
      cmpTbl += '<tr' + rowStyle + '><td>' + (TARIFF.CT_LABELS[ck] || ck) + '</td><td>' + fmt(b.elec_won) +
        '</td><td>' + fmt(b.final_won) + '</td><td>' + eff + '</td><td>' + diffCell + '</td></tr>';
    });
    cmpTbl += '</tbody></table>';
    el("tariff-compare").innerHTML = cmpTbl;
  }

  return { init };
})();


const ReportView = { init: function(){} };
/* ReportView disabled
function _disabled() {
    const cust = "";

    const custMonthly = DATA.monthly.filter(r => r.customer_id === cust).sort((a,b) => a.year_month.localeCompare(b.year_month));
    if (custMonthly.length < 2) { el("report-cards").innerHTML = '<div class="text-secondary">데이터 부족</div>'; return; }

    const ct = custMonthly[0].contract_type;

    // find latest ym with prediction
    let predRow = null, latestYm = null;
    for (let i = custMonthly.length - 1; i >= 0; i--) {
      const ym = custMonthly[i].year_month;
      predRow = DATA.preds.find(r => r.customer_id === cust && r.year_month === ym && r.model === "partial_linear" && r.horizon_days === 20);
      if (predRow && predRow.pred_monthly_kwh != null) { latestYm = ym; break; }
    }
    if (!predRow || !latestYm) { el("report-cards").innerHTML = '<div class="text-secondary">예측 데이터 없음</div>'; return; }

    const predKwh = predRow.pred_monthly_kwh;
    const monthNum = parseInt(latestYm.split("-")[1]);

    const ctxRow = DATA.ctx.find(r => r.customer_id === cust && r.year_month === latestYm && r.horizon_days === 20);
    if (!ctxRow) { el("report-cards").innerHTML = '<div class="text-secondary">컨텍스트 데이터 없음</div>'; return; }

    const prev = ctxRow.prev_month_kwh;
    const yoy = ctxRow.yoy_month_kwh;
    const ma3 = ctxRow.ma3_kwh;

    const condPrev = (isVal(predKwh) && isVal(prev)) ? predKwh > TH_PREV * prev : false;
    const condYoy  = (isVal(predKwh) && isVal(yoy))  ? predKwh > TH_YOY * yoy : false;
    const condMa   = (isVal(predKwh) && isVal(ma3))  ? predKwh > TH_MA3 * ma3 : false;
    const any3 = condPrev || condYoy || condMa;

    const prevDelta = (isVal(prev) && prev > 0 && isVal(predKwh)) ? fmtPct((predKwh/prev-1)*100) : "—";

    let html = "";

    // Report card 1: alert or normal
    if (any3) {
      let alertRows = "";
      if (condPrev) alertRows += '<tr><td>전월 대비</td><td style="color:var(--accent-red);font-weight:700;">' + prevDelta + ' [초과]</td></tr>';
      else alertRows += '<tr><td>전월 대비</td><td>' + prevDelta + ' [정상]</td></tr>';
      if (isVal(yoy)) {
        const yoyD = fmtPct((predKwh/yoy-1)*100);
        alertRows += '<tr><td>전년 동월 대비</td><td' + (condYoy ? ' style="color:var(--accent-red);font-weight:700;"' : '') + '>' + yoyD + (condYoy ? ' [초과]' : ' [정상]') + '</td></tr>';
      }
      if (isVal(ma3)) {
        const ma3D = fmtPct((predKwh/ma3-1)*100);
        alertRows += '<tr><td>3개월 평균 대비</td><td' + (condMa ? ' style="color:var(--accent-red);font-weight:700;"' : '') + '>' + ma3D + (condMa ? ' [초과]' : ' [정상]') + '</td></tr>';
      }
      html += '<div class="report-card"><h4>전력사용량 과다발생 사전 알림</h4>' +
        '<p style="color:var(--text-secondary);"><strong style="color:white;">' + cust + '</strong> 고객님, ' + latestYm + ' 전력사용량이 평소보다 높을 것으로 예측됩니다.</p>' +
        '<table class="table table-sm mb-0"><tr><th>항목</th><th>값</th></tr>' +
        '<tr><td>예측 사용량</td><td style="color:var(--accent-green);font-weight:700;">' + fmt(predKwh) + ' kWh</td></tr>' +
        '<tr><td>전월 사용량</td><td>' + (isVal(prev) ? fmt(prev) + ' kWh' : '—') + '</td></tr>' +
        alertRows + '</table></div>';
    } else {
      html += '<div class="report-card"><h4>전력사용량 분석 리포트</h4>' +
        '<p style="color:var(--text-secondary);"><strong style="color:white;">' + cust + '</strong> 고객님, ' + latestYm + ' 전력사용량은 평균적인 소비 범위 내로 예측됩니다.</p>' +
        '<table class="table table-sm mb-0"><tr><th>항목</th><th>값</th></tr>' +
        '<tr><td>예측 사용량</td><td style="color:var(--accent-green);font-weight:700;">' + fmt(predKwh) + ' kWh</td></tr>' +
        '<tr><td>전월 사용량</td><td>' + (isVal(prev) ? fmt(prev) + ' kWh' : '—') + '</td></tr>' +
        '<tr><td>전월 대비</td><td>' + prevDelta + ' [정상]</td></tr></table></div>';
    }

    // Report card 2: bill impact
    if (isVal(prev) && prev > 0) {
      // simple tariff mapping for ct
      let ctKey = "eul_small_low";
      if (ct && (ct.includes("주택") || ct.includes("res"))) ctKey = "res_low";
      const billPred = TARIFF.finalBill(predKwh, ctKey, monthNum, 0, 9.0, 5.0, "없음");
      const billPrev = TARIFF.finalBill(prev, ctKey, monthNum, 0, 9.0, 5.0, "없음");
      const kwhChg = ((predKwh / prev) - 1) * 100;
      const wonChg = billPrev.final_won > 0 ? ((billPred.final_won / billPrev.final_won) - 1) * 100 : 0;
      const amp = kwhChg !== 0 ? (wonChg / kwhChg) : 0;
      const ampColor = amp > 1.2 ? 'color:var(--accent-red);font-weight:700;' : 'color:var(--accent-green);font-weight:700;';

      html += '<div class="report-card"><h4>예상 요금 영향</h4>' +
        '<table class="table table-sm mb-0"><tr><th>항목</th><th>값</th></tr>' +
        '<tr><td>전월 예상 요금</td><td>' + fmt(billPrev.final_won) + '원</td></tr>' +
        '<tr><td>이번 달 예상 요금</td><td style="color:var(--accent-green);font-weight:700;">' + fmt(billPred.final_won) + '원</td></tr>' +
        '<tr><td>사용량 변동</td><td>' + fmtPct(kwhChg) + '</td></tr>' +
        '<tr><td>요금 변동</td><td style="' + ampColor + '">' + fmtPct(wonChg) + '</td></tr>' +
        '<tr><td>증폭률</td><td style="' + ampColor + '">' + amp.toFixed(2) + '배</td></tr>' +
        '</table></div>';

      if (amp > 1.2) {
        const reason = ctKey.startsWith("res") ? "누진 구간 상승" : "TOU 최대부하 시간대 집중";
        html += '<div class="warn-box">사용량이 <strong>' + fmtPct(kwhChg) + '</strong> 변동하지만 요금은 <strong>' + fmtPct(wonChg) + '</strong> 변동합니다. ' + reason + '에 의한 증폭 효과입니다.</div>';
      }
    }

    // Savings tips
    if (any3) {
      html += '<div class="tip-box"><strong>절약 제안</strong><br>' +
        '&bull; 대기전력 차단 (멀티탭 스위치 활용)<br>' +
        '&bull; 냉난방 설정온도 1~2&deg;C 조정<br>' +
        '&bull; 사용 시간대 분산 (경부하 시간대 활용: 23시~09시)</div>';
    }

    el("report-cards").innerHTML = html;
  }

}
end ReportView disabled */


// ──────────────────────────────────────
// TAB 4: 모델 상세
// ──────────────────────────────────────

const ModelView = (function() {

  function init() {
    // SHAP images
    if (DATA.shap_images && Object.keys(DATA.shap_images).length > 0) {
      let html = '';
      Object.entries(DATA.shap_images).forEach(([name, src]) => {
        html += '<div class="card mb-3"><div class="card-header">' + name + '</div><div class="card-body p-2"><img src="' + src + '" class="img-fluid" alt="' + name + '"></div></div>';
      });
      el("model-shap").innerHTML = html;
    } else {
      el("model-shap").innerHTML = '<div class="no-data-msg" style="min-height:200px;display:flex;align-items:center;justify-content:center;"><div style="text-align:center;color:var(--text-secondary);">SHAP 이미지 없음<br><span style="font-size:0.8rem;">explain_results/ 에 이미지 생성 후 재빌드</span></div></div>';
    }

    // Ablation table
    if (DATA.ablation && DATA.ablation.length > 0) {
      let tbl = '<table class="table table-sm table-hover mb-0"><thead><tr>' +
        '<th>피처 그룹</th><th>제거 시 MAPE (%)</th><th>기여도</th>' +
        '</tr></thead><tbody>';
      DATA.ablation.forEach(r => {
        const contrib = isVal(r.contribution) ? r.contribution.toFixed(1) : "—";
        const contribColor = r.contribution < 0 ? 'color:var(--accent-red);font-weight:600;' : 'color:var(--accent-green);font-weight:600;';
        tbl += '<tr>' +
          '<td>' + (r.group || "") + '</td>' +
          '<td>' + (isVal(r.mape_without) ? r.mape_without.toFixed(2) : "—") + '</td>' +
          '<td style="' + contribColor + '">' + contrib + '</td>' +
          '</tr>';
      });
      tbl += '</tbody></table>';
      tbl += '<div style="color:var(--text-secondary);font-size:0.75rem;margin-top:8px;">기여도: 해당 그룹 제거 시 MAPE 변화 (음수 = 제거하면 성능 악화 = 중요)</div>';
      el("model-ablation").innerHTML = tbl;
    } else {
      el("model-ablation").innerHTML = '<div class="no-data-msg" style="min-height:200px;display:flex;align-items:center;justify-content:center;"><div style="text-align:center;color:var(--text-secondary);">Ablation 데이터 없음<br><span style="font-size:0.8rem;">ablation_study 실행 후 재빌드</span></div></div>';
    }

    // Ablation full results (BTM/기상/기준온도/피처선택 실험)
    if (DATA.ablation_full && DATA.ablation_full.length > 0) {
      let tbl = '<table class="table table-sm table-hover mb-0"><thead><tr>' +
        '<th>실험</th><th>설정</th><th>검침일+10 MAPE (%)</th><th>검침일+20 MAPE (%)</th>' +
        '</tr></thead><tbody>';

      const baseline = DATA.ablation_full.find(r => r.name === "default");
      const baseMape = baseline ? baseline.mape_10d : null;

      DATA.ablation_full.forEach(r => {
        const isDefault = r.name === "default";
        const isOptimal = (r.name || "").includes("optimal");
        const style = isOptimal ? ' style="font-weight:700;background:rgba(59,130,246,0.08);"'
                     : isDefault ? ' style="font-weight:600;"' : '';

        let nameKr = r.name || "";
        if (isDefault) nameKr = "기본값 (베이스라인)";
        else if (isOptimal) nameKr = "최적 조합";
        else if (nameKr.startsWith("btm_")) nameKr = "BTM: " + nameKr.replace("btm_", "");
        else if (nameKr.startsWith("weather_")) nameKr = "기상: " + nameKr.replace("weather_", "");
        else if (nameKr.startsWith("base_")) nameKr = "기준온도: " + nameKr.replace("base_", "");
        else if (nameKr === "feat_selection") nameKr = "피처 자동 선택";
        else if (nameKr.startsWith("drop_")) nameKr = "제거: " + nameKr.replace("drop_", "");

        const m10 = isVal(r.mape_10d) ? r.mape_10d.toFixed(2) : "—";
        const m20 = isVal(r.mape_20d) ? r.mape_20d.toFixed(2) : "—";

        tbl += '<tr' + style + '><td>' + nameKr + '</td><td style="font-size:0.78rem;color:var(--text-secondary);">' +
          (r.config || "") + '</td><td>' + m10 + '</td><td>' + m20 + '</td></tr>';
      });
      tbl += '</tbody></table>';

      if (DATA.ablation_best) {
        const ab = DATA.ablation_best;
        tbl += '<div style="margin-top:12px;color:var(--text-secondary);font-size:0.82rem;">' +
          '<strong>최적 설정:</strong> ' + (ab.optimal_config || "—") +
          ' | 기본 MAPE: ' + (ab.baseline_mape != null ? ab.baseline_mape.toFixed(2) + '%' : '—') +
          ' &rarr; 최적: ' + (ab.optimal_mape != null ? ab.optimal_mape.toFixed(2) + '%' : '—') +
          '</div>';
      }

      el("model-ablation-full").innerHTML = tbl;
    } else {
      el("model-ablation-full").innerHTML = '<div class="text-secondary">Ablation 전체 결과 없음 — ablation_study 실행 후 재빌드</div>';
    }

    // Residual correction
    const rc = DATA.residual_correction;
    if (rc && rc.before_correction && rc.after_correction) {
      const b = rc.before_correction;
      const a = rc.after_correction;
      const improved = (a.mape_pct || 0) < (b.mape_pct || 0);
      let tbl = '<table class="table table-sm mb-0"><thead><tr>' +
        '<th>지표</th><th>보정 전</th><th>보정 후</th><th>변화</th>' +
        '</tr></thead><tbody>';

      [["MAE (kWh)", "mae"], ["RMSE (kWh)", "rmse"], ["CVRMSE (%)", "cvrmse_pct"], ["MAPE (%)", "mape_pct"], ["편향 (kWh)", "bias"]].forEach(([label, key]) => {
        const bv = isVal(b[key]) ? b[key] : null;
        const av = isVal(a[key]) ? a[key] : null;
        const delta = (bv != null && av != null) ? av - bv : null;
        const deltaStr = delta != null ? ((delta > 0 ? "+" : "") + delta.toFixed(2)) : "—";
        const deltaColor = delta != null ? (delta < 0 ? 'color:var(--accent-green);font-weight:600;' : 'color:var(--accent-red);font-weight:600;') : '';
        tbl += '<tr><td>' + label + '</td>' +
          '<td>' + (bv != null ? bv.toFixed(2) : "—") + '</td>' +
          '<td>' + (av != null ? av.toFixed(2) : "—") + '</td>' +
          '<td style="' + deltaColor + '">' + deltaStr + '</td></tr>';
      });
      tbl += '</tbody></table>';

      if (!improved) {
        tbl += '<div class="warn-box" style="margin-top:12px;">보정 후 성능이 개선되지 않았습니다. 데이터 기간이 짧아 잔차 패턴의 통계적 유의성이 부족할 수 있습니다.</div>';
      }

      // Ridge 계수 표시
      if (rc.pattern_analysis && rc.pattern_analysis.ridge_coefficients) {
        const coefs = rc.pattern_analysis.ridge_coefficients;
        tbl += '<div style="margin-top:16px;color:var(--text-secondary);font-size:0.82rem;"><strong>Ridge 보정 계수:</strong> ';
        const sorted = Object.entries(coefs).filter(([k]) => k !== "intercept").sort((a,b) => Math.abs(b[1]) - Math.abs(a[1]));
        tbl += sorted.map(([k,v]) => k + '=' + v.toFixed(1)).join(', ');
        tbl += '</div>';
      }

      el("model-residual").innerHTML = tbl;
    } else {
      el("model-residual").innerHTML = '<div class="text-secondary">잔차 보정 데이터 없음 — run_full_analysis Step 11.5 실행 후 재빌드</div>';
    }
  }

  return { init };
})();


// ──────────────────────────────────────
// Bootstrap: init on tab shown + DOMContentLoaded
// ──────────────────────────────────────

document.addEventListener("DOMContentLoaded", function() {
  CustomerView.init();

  const tabHandlers = {
    "#tab-customer": function() { CustomerView.init(); },
    "#tab-fleet": function() { FleetView.init(); },
    "#tab-tariff": function() { TariffView.init(); },
    "#tab-model": function() { ModelView.init(); },
  };

  document.querySelectorAll('a[data-bs-toggle="tab"]').forEach(t => {
    t.addEventListener("shown.bs.tab", function(e) {
      const target = e.target.getAttribute("href");
      if (tabHandlers[target]) tabHandlers[target]();
      setTimeout(function() {
        document.querySelectorAll("[_echarts_instance_]").forEach(function(c) {
          var inst = echarts.getInstanceByDom(c);
          if (inst) inst.resize();
        });
      }, 50);
    });
  });

  window.addEventListener("resize", function() {
    document.querySelectorAll("[_echarts_instance_]").forEach(function(c) {
      var inst = echarts.getInstanceByDom(c);
      if (inst) inst.resize();
    });
  });
});
</script>

</body>
</html>
'''


# ═══════════════════════════════════════════════════════
# 6. CLI Entry Point
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="파이프라인 결과 -> 자체 완결 HTML 대시보드")
    parser.add_argument("--results", type=str, default="eval_results",
                        help="eval 결과 디렉토리 (default: eval_results/)")
    parser.add_argument("--sliding", type=str, default="sliding_results",
                        help="슬라이딩 결과 디렉토리 (default: sliding_results/)")
    parser.add_argument("--explain", type=str, default="explain_results",
                        help="설명 결과 디렉토리 (default: explain_results/)")
    parser.add_argument("--ui-data", type=str, default="data/ui",
                        help="UI parquet 디렉토리 (default: data/ui/)")
    parser.add_argument("--out", type=str, default="dashboard.html",
                        help="출력 HTML 파일 (default: dashboard.html)")
    parser.add_argument("--export-safe", action="store_true",
                        help="반출 안전 모드 — 고객 데이터 제외 (모델성능/요금/모델상세만)")
    args = parser.parse_args()

    ui_dir = ROOT / args.ui_data
    eval_dir = ROOT / args.results
    sliding_dir = ROOT / args.sliding
    explain_dir = ROOT / args.explain
    out_path = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out

    print(f"[load] UI data   : {ui_dir}")
    print(f"[load] Eval      : {eval_dir}")
    print(f"[load] Sliding   : {sliding_dir}")
    print(f"[load] Explain   : {explain_dir}")

    data = load_pipeline_data(ui_dir, eval_dir, sliding_dir, explain_dir)

    # 반출 안전 모드: 고객별 데이터 제거
    if args.export_safe:
        print("[export-safe] 고객 데이터 제거")
        for key in ("daily", "daily_cum", "monthly", "preds", "ctx"):
            data[key] = None

    # report loaded datasets
    for key, val in data.items():
        if isinstance(val, pd.DataFrame):
            print(f"  {key}: {len(val):,} rows")
        elif isinstance(val, dict):
            print(f"  {key}: {len(val)} items")
        elif val is None:
            print(f"  {key}: (not found)")

    print("[build] JSON payload ...")
    json_payload = build_json_payload(data)
    payload_mb = len(json_payload.encode("utf-8")) / 1024 / 1024
    print(f"  payload size: {payload_mb:.1f} MB")

    print("[build] HTML ...")
    html = build_html(json_payload)
    html_mb = len(html.encode("utf-8")) / 1024 / 1024
    print(f"  HTML size: {html_mb:.1f} MB")

    out_path.write_text(html, encoding="utf-8")
    print(f"[ok] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
