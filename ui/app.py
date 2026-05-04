"""Streamlit 대시보드 — Spotify-inspired dark UI.

탭 구성:
  1. 고객 뷰 — 개인 예측·알림·일자별 누적
  2. 운영 대시보드 — 집계 지표 + 모델 성능
  3. 요금 분석 — 누진제 시뮬레이터 + TOU
  4. 고객 리포트 — 알림 메시지 미리보기

실행: streamlit run ui/app.py --server.port 8765
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

from src.tariff import estimate_bill, bill_increase_analysis
from src.tou import get_season

# ── 스타일 ──

SPOTIFY_GREEN = "#1DB954"
SPOTIFY_GREEN_DARK = "#1AA34A"
SPOTIFY_BG = "#121212"
SPOTIFY_CARD = "#181818"
SPOTIFY_CARD_HOVER = "#282828"
SPOTIFY_TEXT = "#FFFFFF"
SPOTIFY_TEXT_SUB = "#B3B3B3"
SPOTIFY_RED = "#E74C3C"
SPOTIFY_YELLOW = "#F1C40F"

matplotlib.rcParams.update({
    "figure.facecolor": SPOTIFY_BG,
    "axes.facecolor": SPOTIFY_CARD,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": SPOTIFY_TEXT_SUB,
    "text.color": SPOTIFY_TEXT,
    "xtick.color": SPOTIFY_TEXT_SUB,
    "ytick.color": SPOTIFY_TEXT_SUB,
    "grid.color": "#333333",
    "grid.alpha": 0.3,
    "axes.unicode_minus": False,
})
for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        matplotlib.rcParams["font.family"] = f
        break
    except Exception:
        continue

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ui"
TH_PREV, TH_YOY, TH_MA3 = 1.30, 1.30, 1.50


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * { font-family: 'Inter', 'Malgun Gothic', sans-serif !important; }

    .main .block-container { padding-top: 2rem; max-width: 1400px; }

    /* 카드 스타일 */
    .metric-card {
        background: #181818;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
        border: 1px solid #282828;
        transition: background 0.2s;
    }
    .metric-card:hover { background: #282828; }
    .metric-card .label {
        color: #B3B3B3;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-card .value {
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-card .delta {
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 4px;
    }
    .delta-up { color: #E74C3C; }
    .delta-down { color: #1DB954; }
    .delta-neutral { color: #B3B3B3; }

    /* 알림 배지 */
    .alarm-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .alarm-danger {
        background: rgba(231, 76, 60, 0.15);
        color: #E74C3C;
        border: 1px solid rgba(231, 76, 60, 0.3);
    }
    .alarm-safe {
        background: rgba(29, 185, 84, 0.15);
        color: #1DB954;
        border: 1px solid rgba(29, 185, 84, 0.3);
    }

    /* 섹션 헤더 */
    .section-header {
        color: #FFFFFF;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        letter-spacing: -0.3px;
    }

    /* 리포트 카드 */
    .report-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 28px 32px;
        border: 1px solid #2a2a4a;
        margin: 16px 0;
    }
    .report-card h3 { color: #FFFFFF; margin-bottom: 16px; }
    .report-card table { width: 100%; border-collapse: collapse; }
    .report-card td, .report-card th {
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid #2a2a4a;
        color: #B3B3B3;
    }
    .report-card th { color: #FFFFFF; font-weight: 600; }
    .report-card .highlight { color: #1DB954; font-weight: 700; font-size: 1.1rem; }
    .report-card .warning { color: #E74C3C; font-weight: 700; }

    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* 데이터프레임 */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* 알림 박스 */
    .tip-box {
        background: rgba(29, 185, 84, 0.08);
        border-left: 4px solid #1DB954;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
        color: #B3B3B3;
    }
    .warn-box {
        background: rgba(231, 76, 60, 0.08);
        border-left: 4px solid #E74C3C;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
        color: #B3B3B3;
    }
    </style>
    """, unsafe_allow_html=True)


def metric_card(label, value, delta=None, delta_dir="neutral"):
    delta_class = {"up": "delta-up", "down": "delta-down"}.get(delta_dir, "delta-neutral")
    delta_html = f'<div class="delta {delta_class}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ── 데이터 로딩 ──

@st.cache_data
def load_all():
    daily = pd.read_parquet(DATA_DIR / "daily.parquet")
    daily_cum = pd.read_parquet(DATA_DIR / "daily_cum.parquet")
    monthly = pd.read_parquet(DATA_DIR / "monthly.parquet")
    preds = pd.read_parquet(DATA_DIR / "preds.parquet")
    ctx = pd.read_parquet(DATA_DIR / "ctx.parquet")
    metrics = pd.read_parquet(DATA_DIR / "metrics.parquet")
    monthly["year_month"] = pd.to_datetime(monthly["year_month"] + "-01")
    preds["year_month"] = pd.to_datetime(preds["year_month"] + "-01")
    ctx["year_month"] = pd.to_datetime(ctx["year_month"] + "-01")
    return daily, daily_cum, monthly, preds, ctx, metrics


# ── 탭 1: 고객 뷰 ──

def customer_view(daily_cum, monthly, preds, ctx):
    customers = sorted(monthly["customer_id"].unique())
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        cust = st.selectbox("고객 선택", customers, index=0)
    with c2:
        customer_months = sorted(monthly[monthly["customer_id"] == cust]["year_month"].unique())
        default_idx = max(0, len(customer_months) // 2)
        ym = st.selectbox("기준월", customer_months, index=default_idx,
                          format_func=lambda d: d.strftime("%Y-%m"))
    with c3:
        horizon = st.radio("horizon", [10, 20], horizontal=True)

    row = ctx[(ctx["customer_id"] == cust) & (ctx["year_month"] == ym) & (ctx["horizon_days"] == horizon)]
    if row.empty:
        st.warning("선택 조합에 데이터가 없습니다.")
        return
    row = row.iloc[0]

    pred_row = preds[(preds["customer_id"] == cust) & (preds["year_month"] == ym)
                     & (preds["horizon_days"] == horizon) & (preds["model"] == "partial_linear")]
    pred_kwh = float(pred_row["pred_monthly_kwh"].iloc[0]) if not pred_row.empty else np.nan

    prev = row.get("prev_month_kwh", np.nan)
    yoy = row.get("yoy_month_kwh", np.nan)
    ma3 = row.get("ma3_kwh", np.nan)

    cond_prev = pred_kwh > TH_PREV * prev if pd.notna(prev) else False
    cond_yoy = pred_kwh > TH_YOY * yoy if pd.notna(yoy) else False
    cond_ma = pred_kwh > TH_MA3 * ma3 if pd.notna(ma3) else False
    any3 = bool(cond_prev or cond_yoy or cond_ma)

    # KPI 카드
    cols = st.columns(4)
    with cols[0]:
        metric_card("예측 월 사용량", f"{pred_kwh:,.0f} kWh")
    with cols[1]:
        d = f"{(pred_kwh/prev-1)*100:+.1f}%" if pd.notna(prev) and prev > 0 else None
        dd = "up" if d and float(d.strip('%+')) > 0 else "down" if d else "neutral"
        metric_card("전월", f"{prev:,.0f} kWh" if pd.notna(prev) else "—", d, dd)
    with cols[2]:
        d = f"{(pred_kwh/yoy-1)*100:+.1f}%" if pd.notna(yoy) and yoy > 0 else None
        dd = "up" if d and float(d.strip('%+')) > 0 else "down" if d else "neutral"
        metric_card("전년 동월", f"{yoy:,.0f} kWh" if pd.notna(yoy) else "—", d, dd)
    with cols[3]:
        d = f"{(pred_kwh/ma3-1)*100:+.1f}%" if pd.notna(ma3) and ma3 > 0 else None
        dd = "up" if d and float(d.strip('%+')) > 0 else "down" if d else "neutral"
        metric_card("3개월 평균", f"{ma3:,.0f} kWh" if pd.notna(ma3) else "—", d, dd)

    # 알림 판정
    st.markdown('<div class="section-header">알림 판정</div>', unsafe_allow_html=True)
    alarm_cols = st.columns(3)
    for col, (label, triggered, rhs) in zip(alarm_cols, [
        ("전월 대비 +30%", cond_prev, prev * TH_PREV if pd.notna(prev) else np.nan),
        ("전년 동월 대비 +30%", cond_yoy, yoy * TH_YOY if pd.notna(yoy) else np.nan),
        ("3개월 평균 대비 +50%", cond_ma, ma3 * TH_MA3 if pd.notna(ma3) else np.nan),
    ]):
        with col:
            cls = "alarm-danger" if triggered else "alarm-safe"
            icon = "ALERT" if triggered else "OK"
            detail = f"예측 {pred_kwh:,.0f} vs 임계 {rhs:,.0f}" if pd.notna(rhs) else "기준 데이터 부족"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <span class="alarm-badge {cls}">{icon}</span>
                <div style="color: #B3B3B3; font-size: 0.8rem; margin-top: 8px;">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

    if any3:
        st.markdown('<div class="warn-box">⚠️ <strong>알림 발송 대상</strong> — 예측 사용량이 3조건 중 1건 이상 초과</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="tip-box">✅ 현재 평균적 소비 범위 내</div>', unsafe_allow_html=True)

    # 누적 차트
    st.markdown('<div class="section-header">기준월 일자별 누적 사용량</div>', unsafe_allow_html=True)
    ym_str = ym.strftime("%Y-%m")
    dcum = daily_cum[daily_cum["customer_id"] == cust].copy()
    dcum["year_month"] = dcum["year_month"].astype(str)
    cur = dcum[dcum["year_month"] == ym_str].sort_values("date")
    if len(cur):
        days_in_month = int(cur["day_of_month"].max())
        observed = cur[cur["day_of_month"] <= horizon]
        remainder = cur[cur["day_of_month"] > horizon]

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(observed["day_of_month"], observed["cum_kwh"], "-o",
                color=SPOTIFY_GREEN, label=f"관측 ({horizon}일)", markersize=4, linewidth=2)
        ax.plot(remainder["day_of_month"], remainder["cum_kwh"], "--o",
                color="#555555", label="실측 (잔여)", markersize=3, alpha=0.5)
        if not observed.empty:
            end_obs = observed.iloc[-1]
            scale = days_in_month / horizon
            ax.plot([end_obs["day_of_month"], days_in_month],
                    [end_obs["cum_kwh"], end_obs["cum_kwh"] * scale],
                    ":", color=SPOTIFY_RED, linewidth=2.5, label="예측")
            ax.scatter([days_in_month], [end_obs["cum_kwh"] * scale],
                       color=SPOTIFY_RED, zorder=5, s=60)
        ax.axvline(horizon, color="#444", linestyle="--", alpha=0.5)
        ax.set_xlabel("일자")
        ax.set_ylabel("누적 kWh")
        ax.legend(loc="upper left", framealpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # 12개월 막대
    st.markdown('<div class="section-header">최근 12개월 + 예측</div>', unsafe_allow_html=True)
    hist = monthly[monthly["customer_id"] == cust].sort_values("year_month")
    hist_12 = hist[hist["year_month"] < ym].tail(12)
    if not hist_12.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 3.2))
        x_labels = list(hist_12["year_month"].dt.strftime("%Y-%m")) + [ym.strftime("%Y-%m")]
        bars1 = ax2.bar(hist_12["year_month"].dt.strftime("%Y-%m"), hist_12["monthly_kwh"],
                        color="#535353", label="실측", width=0.7)
        bars2 = ax2.bar([ym.strftime("%Y-%m")], [pred_kwh],
                        color=SPOTIFY_GREEN if not any3 else SPOTIFY_RED,
                        label="예측", width=0.7, alpha=0.9)
        ax2.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("월 사용량 (kWh)")
        ax2.legend(loc="upper left", framealpha=0.3)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


# ── 탭 2: 운영 대시보드 ──

def fleet_view(daily, monthly, preds, ctx, metrics):
    n_customers = monthly["customer_id"].nunique()
    latest_ym = ctx["year_month"].max()
    latest_ctx = ctx[(ctx["year_month"] == latest_ym) & (ctx["horizon_days"] == 20)].copy()
    latest_preds = preds[(preds["year_month"] == latest_ym) & (preds["horizon_days"] == 20) & (preds["model"] == "partial_linear")]
    merged = latest_ctx.merge(latest_preds[["customer_id", "pred_monthly_kwh"]], on="customer_id", how="left")
    merged["cond_prev"] = merged["pred_monthly_kwh"] > merged["prev_month_kwh"] * TH_PREV
    merged["cond_ma"] = merged["pred_monthly_kwh"] > merged["ma3_kwh"] * TH_MA3
    merged["any3"] = merged[["cond_prev", "cond_ma"]].fillna(False).any(axis=1)
    alarm_n = int(merged["any3"].sum())
    alarm_rate = float(merged["any3"].mean() * 100)

    pl_all = metrics[metrics["model"] == "partial_linear"]
    mape_20 = pl_all[pl_all["horizon_days"] == 20]["mape_pct"].mean()
    f1_20 = pl_all[pl_all["horizon_days"] == 20]["alarm_f1"].mean()

    cols = st.columns(4)
    with cols[0]:
        metric_card("관리 고객 수", f"{n_customers:,}")
    with cols[1]:
        metric_card(f"{latest_ym.strftime('%Y-%m')} 알림 대상", f"{alarm_n}명", f"{alarm_rate:.1f}%", "up" if alarm_rate > 5 else "neutral")
    with cols[2]:
        metric_card("+20일 MAPE", f"{mape_20:.2f}%")
    with cols[3]:
        metric_card("+20일 알림 F1", f"{f1_20:.2f}")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown('<div class="section-header">계약종별 구성</div>', unsafe_allow_html=True)
        ct_counts = monthly.drop_duplicates("customer_id")["contract_type"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        colors = [SPOTIFY_GREEN, "#535353"]
        wedges, texts, autotexts = ax.pie(ct_counts.values, labels=ct_counts.index,
                                           autopct="%1.1f%%", startangle=90, colors=colors[:len(ct_counts)])
        for t in autotexts:
            t.set_color("white")
            t.set_fontweight("bold")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        st.markdown('<div class="section-header">모델 성능 비교</div>', unsafe_allow_html=True)
        display = metrics[["model", "horizon_days", "contract_type", "n", "mape_pct", "alarm_f1"]].copy()
        display.columns = ["모델", "horizon", "계약종", "n", "MAPE(%)", "알림 F1"]
        st.dataframe(display.round(3), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">월별 알림 발생률 추이</div>', unsafe_allow_html=True)
    monthly_alarm = []
    for ym_g, sub in ctx[ctx["horizon_days"] == 20].groupby("year_month"):
        ps = preds[(preds["year_month"] == ym_g) & (preds["horizon_days"] == 20) & (preds["model"] == "partial_linear")]
        m = sub.merge(ps[["customer_id", "pred_monthly_kwh"]], on="customer_id", how="left")
        cp = m["pred_monthly_kwh"] > m["prev_month_kwh"] * TH_PREV
        cm = m["pred_monthly_kwh"] > m["ma3_kwh"] * TH_MA3
        a = cp.fillna(False) | cm.fillna(False)
        if len(m) > 0:
            monthly_alarm.append({"year_month": ym_g, "rate": a.mean() * 100})
    mdf = pd.DataFrame(monthly_alarm).sort_values("year_month")
    if not mdf.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(range(len(mdf)), mdf["rate"], alpha=0.15, color=SPOTIFY_GREEN)
        ax.plot(range(len(mdf)), mdf["rate"], "-o", color=SPOTIFY_GREEN, markersize=5, linewidth=2)
        ax.set_xticks(range(len(mdf)))
        ax.set_xticklabels(mdf["year_month"].dt.strftime("%Y-%m"), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("알림 발생률 (%)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ── 탭 3: 요금 분석 ──

def tariff_view(monthly, preds, ctx):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">누진제 시뮬레이터 (주택용)</div>', unsafe_allow_html=True)
        sim_kwh = st.slider("월 사용량 (kWh)", 50, 1000, 300, 10)
        sim_month = st.selectbox("월", list(range(1, 13)), index=0, key="tariff_month")
        bill = estimate_bill(sim_kwh, "주택용", sim_month)
        metric_card("예상 요금", f"{bill['total_won']:,}원",
                    f"누진 {bill['tier']}구간 · 실효단가 {bill['effective_rate']:.1f}원/kWh", "neutral")

        st.markdown('<div class="section-header">누진 구간 경계 효과</div>', unsafe_allow_html=True)
        kwh_range = list(range(50, 801, 10))
        bills = [estimate_bill(k, "주택용", sim_month) for k in kwh_range]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax1.plot(kwh_range, [b["total_won"] for b in bills], color=SPOTIFY_GREEN, linewidth=2.5)
        ax1.axvline(sim_kwh, color=SPOTIFY_TEXT_SUB, linestyle="--", alpha=0.5)
        ax1.set_ylabel("예상 요금 (원)")
        ax2.plot(kwh_range, [b["effective_rate"] for b in bills], color=SPOTIFY_YELLOW, linewidth=2.5)
        ax2.axvline(sim_kwh, color=SPOTIFY_TEXT_SUB, linestyle="--", alpha=0.5)
        ax2.set_ylabel("실효단가 (원/kWh)")
        ax2.set_xlabel("월 사용량 (kWh)")
        tiers = [300, 450] if sim_month in (7, 8) else [200, 400]
        for t in tiers:
            ax1.axvline(t, color="#555", linestyle=":", alpha=0.4)
            ax2.axvline(t, color="#555", linestyle=":", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown('<div class="section-header">사용량 vs 요금 증가 비교</div>', unsafe_allow_html=True)
        st.caption("사용량 증가가 요금에 미치는 증폭 효과")
        base_kwh = st.number_input("기준 사용량 (kWh)", 100, 2000, 300, 50)
        rows = []
        for pct in [10, 20, 30, 40, 50]:
            a = bill_increase_analysis(base_kwh * (1 + pct / 100), base_kwh, "주택용", sim_month)
            rows.append({
                "사용량 증가": f"+{pct}%",
                "사용량": f"{a['pred_kwh']:,.0f} kWh",
                "기준 요금": f"{a['prev_bill']:,}원",
                "예측 요금": f"{a['pred_bill']:,}원",
                "요금 증가": f"+{a['won_change_pct']:.1f}%",
                "증폭률": f"{a['amplification']:.2f}x",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">일반용 TOU 요금 구조</div>', unsafe_allow_html=True)
        gen_kwh = st.number_input("월 사용량 (kWh)", 100, 50000, 1000, 100, key="gen_kwh")
        peak_ratio = st.slider("최대부하 시간대 비율 (%)", 10, 70, 30, 5)
        off_ratio = st.slider("경부하 시간대 비율 (%)", 10, 60, 30, 5)
        mid_ratio = 100 - peak_ratio - off_ratio
        if mid_ratio < 0:
            st.warning("비율 합이 100%를 초과합니다")
        else:
            season = get_season(sim_month)
            ratios = {"off_peak": off_ratio/100, "mid_peak": mid_ratio/100, "on_peak": peak_ratio/100}
            gen_bill = estimate_bill(gen_kwh, "일반용", sim_month, tou_ratios=ratios)
            metric_card("예상 전력량요금", f"{gen_bill['energy_won']:,}원",
                        f"실효단가 {gen_bill['effective_rate']:.1f}원/kWh · {season}", "neutral")


# ── 탭 4: 고객 리포트 ──

def customer_report(daily_cum, monthly, preds, ctx):
    customers = sorted(monthly["customer_id"].unique())
    cust = st.selectbox("고객 선택", customers, index=0, key="report_cust")

    cust_monthly = monthly[monthly["customer_id"] == cust].sort_values("year_month")
    if len(cust_monthly) < 2:
        st.warning("데이터 부족")
        return

    ct = cust_monthly["contract_type"].iloc[0]

    for latest_ym in reversed(cust_monthly["year_month"].tolist()):
        pred_row = preds[(preds["customer_id"] == cust) & (preds["year_month"] == latest_ym)
                         & (preds["model"] == "partial_linear") & (preds["horizon_days"] == 20)]
        if not pred_row.empty:
            break
    else:
        st.warning("예측 데이터 없음")
        return

    pred_kwh = float(pred_row["pred_monthly_kwh"].iloc[0])
    month_num = latest_ym.month

    ctx_row = ctx[(ctx["customer_id"] == cust) & (ctx["year_month"] == latest_ym) & (ctx["horizon_days"] == 20)]
    if ctx_row.empty:
        st.warning("컨텍스트 데이터 없음")
        return
    ctx_row = ctx_row.iloc[0]

    prev = ctx_row.get("prev_month_kwh", np.nan)
    yoy = ctx_row.get("yoy_month_kwh", np.nan)
    ma3 = ctx_row.get("ma3_kwh", np.nan)

    cond_prev = pred_kwh > TH_PREV * prev if pd.notna(prev) else False
    cond_yoy = pred_kwh > TH_YOY * yoy if pd.notna(yoy) else False
    cond_ma = pred_kwh > TH_MA3 * ma3 if pd.notna(ma3) else False
    any3 = bool(cond_prev or cond_yoy or cond_ma)

    ym_str = latest_ym.strftime("%Y년 %m월")
    prev_delta = f"{(pred_kwh/prev-1)*100:+.1f}%" if pd.notna(prev) and prev > 0 else "—"

    bill_analysis = None
    if pd.notna(prev) and prev > 0:
        bill_analysis = bill_increase_analysis(pred_kwh, prev, ct, month_num)

    # 리포트 카드
    if any3:
        alert_rows = ""
        if cond_prev:
            alert_rows += f'<tr><td>전월 대비</td><td class="warning">{prev_delta} ⚠️ 초과</td></tr>'
        else:
            alert_rows += f'<tr><td>전월 대비</td><td>{prev_delta} ✅</td></tr>'
        if pd.notna(yoy):
            yoy_d = f"{(pred_kwh/yoy-1)*100:+.1f}%"
            cls = 'warning' if cond_yoy else ''
            alert_rows += f'<tr><td>전년 동월 대비</td><td class="{cls}">{yoy_d} {"⚠️ 초과" if cond_yoy else "✅"}</td></tr>'
        if pd.notna(ma3):
            ma3_d = f"{(pred_kwh/ma3-1)*100:+.1f}%"
            cls = 'warning' if cond_ma else ''
            alert_rows += f'<tr><td>3개월 평균 대비</td><td class="{cls}">{ma3_d} {"⚠️ 초과" if cond_ma else "✅"}</td></tr>'

        st.markdown(f"""
        <div class="report-card">
            <h3>⚡ 전력사용량 과다발생 사전 알림</h3>
            <p style="color: #B3B3B3;">
                <strong style="color: white;">{cust}</strong> 고객님, {ym_str} 전력사용량이 평소보다 높을 것으로 예측됩니다.
            </p>
            <table>
                <tr><th>항목</th><th>값</th></tr>
                <tr><td>예측 사용량</td><td class="highlight">{pred_kwh:,.0f} kWh</td></tr>
                <tr><td>전월 사용량</td><td>{prev:,.0f} kWh</td></tr>
                {alert_rows}
            </table>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="report-card">
            <h3>📊 전력사용량 분석 리포트</h3>
            <p style="color: #B3B3B3;">
                <strong style="color: white;">{cust}</strong> 고객님, {ym_str} 전력사용량은 평균적인 소비 범위 내로 예측됩니다.
            </p>
            <table>
                <tr><th>항목</th><th>값</th></tr>
                <tr><td>예측 사용량</td><td class="highlight">{pred_kwh:,.0f} kWh</td></tr>
                <tr><td>전월 사용량</td><td>{prev:,.0f} kWh</td></tr>
                <tr><td>전월 대비</td><td>{prev_delta} ✅</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    if bill_analysis:
        amp = bill_analysis["amplification"]
        amp_color = "warning" if amp > 1.2 else "highlight"
        st.markdown(f"""
        <div class="report-card">
            <h3>💰 예상 요금 영향</h3>
            <table>
                <tr><th>항목</th><th>값</th></tr>
                <tr><td>전월 예상 요금</td><td>{bill_analysis['prev_bill']:,}원</td></tr>
                <tr><td>이번 달 예상 요금</td><td class="highlight">{bill_analysis['pred_bill']:,}원</td></tr>
                <tr><td>사용량 변동</td><td>{bill_analysis['kwh_change_pct']:+.1f}%</td></tr>
                <tr><td>요금 변동</td><td class="{amp_color}">{bill_analysis['won_change_pct']:+.1f}%</td></tr>
                <tr><td>증폭률</td><td class="{amp_color}">{amp:.2f}배</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        if amp > 1.2:
            st.markdown(f"""
            <div class="warn-box">
                사용량이 <strong>{bill_analysis['kwh_change_pct']:+.1f}%</strong> 변동하지만
                요금은 <strong>{bill_analysis['won_change_pct']:+.1f}%</strong> 변동합니다.
                {'누진 구간 상승' if ct == '주택용' else 'TOU 최대부하 시간대 집중'}에 의한 증폭 효과입니다.
            </div>
            """, unsafe_allow_html=True)

    if any3:
        st.markdown("""
        <div class="tip-box">
            <strong>💡 절약 제안</strong><br>
            • 대기전력 차단 (멀티탭 스위치 활용)<br>
            • 냉난방 설정온도 1~2°C 조정<br>
            • 사용 시간대 분산 (경부하 시간대 활용: 23시~09시)
        </div>
        """, unsafe_allow_html=True)


# ── 메인 ──

def main():
    st.set_page_config(
        page_title="전기요금 과다발생 예측",
        page_icon="⚡",
        layout="wide",
    )
    inject_css()

    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        <span style="font-size: 2.2rem;">⚡</span>
        <div>
            <div style="font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px;">전기요금 과다발생 사전 예측</div>
            <div style="color: #B3B3B3; font-size: 0.85rem;">한국전력공사 지정과제 · 2026 AX 아이디어 경진대회</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    daily, daily_cum, monthly, preds, ctx, metrics = load_all()

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚡ 고객 뷰", "📊 운영 대시보드", "💰 요금 분석", "📋 고객 리포트"
    ])
    with tab1:
        customer_view(daily_cum, monthly, preds, ctx)
    with tab2:
        fleet_view(daily, monthly, preds, ctx, metrics)
    with tab3:
        tariff_view(monthly, preds, ctx)
    with tab4:
        customer_report(daily_cum, monthly, preds, ctx)


if __name__ == "__main__":
    main()
