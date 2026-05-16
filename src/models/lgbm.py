"""글로벌 LightGBM 모델 — partial_kwh + 기상·달력·래그 피처로 월 사용량 예측.

설계 의도:
  - partial_linear 가 이미 매우 강력하므로 LGBM 은 '그 위의 잔차' 를 학습하는 구조
  - 피처에 partial_kwh 를 포함시켜 LGBM 이 자연스럽게 partial_linear 를 학습 후 개선
  - 시간 split: 가장 최근 연도를 test, 이전을 train (out-of-time, 누설 방지)
  - 반입용 가중치는 booster.save_model() + feature_meta.json 으로 저장

공동주택 공공데이터처럼 고객별 weather 컬럼이 같이 있는 소스면 피처 풍부,
합성 데이터나 weather 없는 소스면 기상 피처 없이도 동작 (NaN 허용).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..eval import build_horizon_table, daily_by_customer, monthly_by_customer
from ..schemas import CONTRACT_TYPE, CUSTOMER_ID, P_ACTIVE_KWH, P_REACTIVE_KWH, TS
from ..utils import hdd_cdd


_CALENDAR_FEATURES = ["month", "days_in_month", "days_observed"]
_NUMERIC_FEATURES_BASE = [
    "partial_kwh",
    "partial_rate",
    "prev_month_kwh",
    "ma3_kwh",
    "customer_mean_history",
]
_AMI_FEATURES = [
    "observed_peak_ratio",
    "observed_tou_on_peak",
    "observed_night_ratio",
    "observed_load_factor",
    "observed_weekend_weekday_ratio",
    "observed_daily_cv",
    "observed_power_factor",
    "pf_vs_prev_month",
    "observed_reactive_ratio",
]
_SPECIAL_DAY_FEATURES = [
    "n_workday",
    "n_holiday",
    "disruption_ratio",
]
_WEATHER_FEATURES = [
    "hdd_full",
    "cdd_full",
    "hdd_observed",
    "cdd_observed",
    "hdd_remainder",
    "cdd_remainder",
]


_EXTRA_NUMERIC = [
    "n_households",
    "contract_power_kw",
    "kwh_per_household",
]

@dataclass
class FeatureSpec:
    has_weather: bool
    has_ami: bool = False
    has_special_days: bool = False
    extra_numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)

    @property
    def numeric(self) -> list[str]:
        cols = list(_CALENDAR_FEATURES) + list(_NUMERIC_FEATURES_BASE)
        if self.has_weather:
            cols += list(_WEATHER_FEATURES)
        if self.has_ami:
            cols += list(_AMI_FEATURES)
        if self.has_special_days:
            cols += list(_SPECIAL_DAY_FEATURES)
        cols += self.extra_numeric
        return cols

    @property
    def all(self) -> list[str]:
        return self.numeric + self.categorical


def _load_asos_daily() -> pd.DataFrame | None:
    """ASOS/asos_all.parquet에서 서울(108) 일별 평균 기온 로드."""
    from pathlib import Path as _Path
    for p in [_Path("ASOS/asos_all.parquet"), _Path("data/ASOS/asos_all.parquet")]:
        if p.exists():
            _asos = pd.read_parquet(p, columns=["station_id", "ts", "temp_c"])
            _asos["ts"] = pd.to_datetime(_asos["ts"])
            _asos = _asos[_asos["station_id"] == 108]
            daily = _asos.groupby(_asos["ts"].dt.date)["temp_c"].mean().reset_index()
            daily.columns = ["date", "temp_c"]
            daily["date"] = pd.to_datetime(daily["date"])
            return daily
    return None


def _weather_aggs(
    df: pd.DataFrame, horizon_tbl: pd.DataFrame,
    remainder_temp_override: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """고객·월·horizon 별 기상 집계 (full/observed/remainder).

    ASOS에서 직접 일별 기온을 로드. observed는 항상 ASOS 실측.
    remainder_temp_override가 주어지면 잔여 기간 기온을 해당 데이터로 대체 (TMY 등).
    override 형식: DataFrame with columns [date, temp_c] (월별이면 [month, temp_c]).
    """
    asos_daily = _load_asos_daily()
    if asos_daily is None:
        print(f"  [weather_aggs] ASOS 파일 없음 → 스킵", flush=True)
        return None

    try:
        custs = df[CUSTOMER_ID].unique()
        dates = pd.to_datetime(df[TS]).dt.normalize().unique()
        cust_dates = pd.DataFrame([(c, dt) for c in custs for dt in dates],
                                  columns=[CUSTOMER_ID, "date"])
        d = cust_dates.merge(asos_daily, on="date", how="left")
        print(f"  [weather_aggs] ASOS 직접 로드: {len(asos_daily)}일, {len(custs)}고객", flush=True)
    except Exception as e:
        print(f"  [weather_aggs] ASOS 로드 실패: {e}", flush=True)
        return None

    d["year_month"] = d["date"].dt.to_period("M")
    d["day_of_month"] = d["date"].dt.day

    na_rate = d["temp_c"].isna().mean()
    print(f"  [weather_aggs] 기온 결측률: {na_rate:.1%}", flush=True)
    if na_rate > 0.5:
        print(f"  [weather_aggs] 결측 50% 초과 → 스킵", flush=True)
        return None

    # remainder override 준비 (TMY 등)
    d_rem = None
    if remainder_temp_override is not None:
        if "month" in remainder_temp_override.columns and "date" not in remainder_temp_override.columns:
            # 월별 TMY → 고객×일 확장 (같은 월의 모든 날에 동일 기온)
            d_rem = d[[CUSTOMER_ID, "date", "year_month", "day_of_month"]].copy()
            d_rem["_month"] = d_rem["date"].dt.month
            tmy_lookup = remainder_temp_override.groupby("month")["temp_c"].mean().to_dict()
            d_rem["temp_c"] = d_rem["_month"].map(tmy_lookup)
            d_rem = d_rem.drop(columns=["_month"])
            print(f"  [weather_aggs] remainder를 TMY로 대체 (월별)", flush=True)
        elif "date" in remainder_temp_override.columns:
            # 일별 TMY
            d_rem = cust_dates.merge(remainder_temp_override[["date", "temp_c"]], on="date", how="left")
            d_rem["year_month"] = d_rem["date"].dt.to_period("M")
            d_rem["day_of_month"] = d_rem["date"].dt.day
            print(f"  [weather_aggs] remainder를 TMY로 대체 (일별)", flush=True)

    # 월 전체 기온 평균
    full = d.groupby([CUSTOMER_ID, "year_month"], observed=True)["temp_c"].mean().reset_index(
        name="temp_full_mean"
    )
    hdd_f, cdd_f = hdd_cdd(full["temp_full_mean"].to_numpy())
    full["hdd_full"] = hdd_f
    full["cdd_full"] = cdd_f

    rows = []
    for h in horizon_tbl["horizon_days"].unique():
        # observed: 항상 ASOS 실측
        obs = (
            d[d["day_of_month"] <= h]
            .groupby([CUSTOMER_ID, "year_month"], observed=True)["temp_c"]
            .mean()
            .reset_index(name="temp_observed_mean")
        )
        # remainder: override 있으면 TMY, 없으면 ASOS
        rem_source = d_rem if d_rem is not None else d
        rem = (
            rem_source[rem_source["day_of_month"] > h]
            .groupby([CUSTOMER_ID, "year_month"], observed=True)["temp_c"]
            .mean()
            .reset_index(name="temp_remainder_mean")
        )
        merged = obs.merge(rem, on=[CUSTOMER_ID, "year_month"], how="outer")
        merged["horizon_days"] = h
        hdd_o, cdd_o = hdd_cdd(merged["temp_observed_mean"].to_numpy())
        hdd_r, cdd_r = hdd_cdd(merged["temp_remainder_mean"].to_numpy())
        merged["hdd_observed"] = hdd_o
        merged["cdd_observed"] = cdd_o
        merged["hdd_remainder"] = hdd_r
        merged["cdd_remainder"] = cdd_r
        rows.append(merged)
    obs_rem = pd.concat(rows, ignore_index=True)
    result = obs_rem.merge(full, on=[CUSTOMER_ID, "year_month"], how="left")
    rem_src = "TMY" if d_rem is not None else "ASOS"
    print(f"  [weather_aggs] 생성 완료: {len(result)}행, remainder={rem_src}, hdd_remainder 비null {result['hdd_remainder'].notna().sum()}", flush=True)
    return result


def _ami_features(df: pd.DataFrame, horizon_tbl: pd.DataFrame) -> pd.DataFrame | None:
    """관측 기간의 15분 AMI 패턴에서 피처 산출 — AMI 데이터 고유 가치."""
    # 일별 집계 데이터 감지 (15분이면 하루 96행, 일별이면 1행)
    sample_cust = df[CUSTOMER_ID].iloc[0]
    sample_date = pd.to_datetime(df[TS]).dt.date.iloc[0]
    n_per_day = len(df[(df[CUSTOMER_ID] == sample_cust) & (pd.to_datetime(df[TS]).dt.date == sample_date)])
    if n_per_day <= 1:
        print("  [AMI] 일별 데이터 감지 → AMI 피처 산출 불가, 스킵", flush=True)
        return None  # 일별 데이터 → AMI 피처 산출 불가

    cols = [CUSTOMER_ID, TS, P_ACTIVE_KWH]
    if P_REACTIVE_KWH in df.columns:
        cols.append(P_REACTIVE_KWH)
    d = df[cols].copy()
    t = pd.to_datetime(d[TS])
    d["date"] = t.dt.normalize()
    d["hour"] = t.dt.hour
    d["year_month"] = t.dt.to_period("M")
    d["day_of_month"] = t.dt.day
    d["dayofweek"] = t.dt.dayofweek

    rows = []
    for h in horizon_tbl["horizon_days"].unique():
        obs = d[d["day_of_month"] <= h].copy()

        # 고객×월 단위 집계
        for (cid, ym), grp in obs.groupby([CUSTOMER_ID, "year_month"], observed=True):
            vals = grp[P_ACTIVE_KWH].values
            if len(vals) < 4:
                continue

            total = vals.sum()
            if total <= 0:
                continue

            peak_mask = grp["hour"].between(10, 17)
            night_mask = (grp["hour"] >= 23) | (grp["hour"] < 9)
            weekend_mask = grp["dayofweek"] >= 5
            weekday_mask = ~weekend_mask

            peak_ratio = vals[peak_mask.values].sum() / total
            night_ratio = vals[night_mask.values].sum() / total

            # TOU on_peak (계절별) + 토/공휴일 재분류
            from ..tou import get_season, classify_tou
            month_num = ym.month
            season = get_season(month_num)
            tou_labels = np.array([classify_tou(h_, season) for h_ in grp["hour"]])
            # 토요일(5): on_peak → mid_peak
            dow = grp["dayofweek"].values
            sat_mask = (dow == 5) & (tou_labels == "on_peak")
            tou_labels[sat_mask] = "mid_peak"
            # 일요일(6): 전부 → off_peak
            sun_mask = dow == 6
            tou_labels[sun_mask] = "off_peak"
            on_peak_sum = vals[tou_labels == "on_peak"].sum()
            tou_on_peak = on_peak_sum / total

            # load factor: mean / max
            load_factor = vals.mean() / vals.max() if vals.max() > 0 else 0

            # weekend/weekday ratio
            wkend_mean = vals[weekend_mask.values].mean() if weekend_mask.any() else 0
            wkday_mean = vals[weekday_mask.values].mean() if weekday_mask.any() else 1e-9
            wk_ratio = wkend_mean / wkday_mean

            # daily CV
            daily_sum = grp.groupby("date")[P_ACTIVE_KWH].sum()
            daily_cv = daily_sum.std() / daily_sum.mean() if daily_sum.mean() > 0 else 0

            # 역률 관련 (무효전력 있을 때만)
            pf_val = np.nan
            reactive_ratio = np.nan
            if P_REACTIVE_KWH in d.columns:
                reactive_col = grp[P_REACTIVE_KWH] if P_REACTIVE_KWH in grp.columns else None
                if reactive_col is not None and reactive_col.notna().sum() > 0:
                    p = vals.copy()
                    q = reactive_col.values
                    valid = (p > 0) & np.isfinite(q)
                    if valid.sum() > 0:
                        p_valid = p[valid]
                        q_valid = np.abs(q[valid])
                        apparent = np.sqrt(p_valid**2 + q_valid**2)
                        pf_val = float((p_valid / apparent.clip(min=1e-9)).mean())
                        reactive_ratio = float(q_valid.sum() / p_valid.sum())

            rows.append({
                CUSTOMER_ID: cid,
                "year_month": ym,
                "horizon_days": h,
                "observed_peak_ratio": float(peak_ratio),
                "observed_tou_on_peak": float(tou_on_peak),
                "observed_night_ratio": float(night_ratio),
                "observed_load_factor": float(load_factor),
                "observed_weekend_weekday_ratio": float(wk_ratio),
                "observed_daily_cv": float(daily_cv),
                "observed_power_factor": float(pf_val),
                "observed_reactive_ratio": float(reactive_ratio),
            })

    return pd.DataFrame(rows) if rows else None


def _special_day_features(horizon_tbl: pd.DataFrame) -> pd.DataFrame:
    """특수일 피처 — 월별 공휴일·징검다리·선거일 구성."""
    from ..special_days import build_special_day_features
    year_months = horizon_tbl["year_month"].unique()
    sd = build_special_day_features(year_months)
    keep = ["year_month", "n_workday", "n_holiday", "disruption_ratio"]
    return sd[[c for c in keep if c in sd.columns]]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSpec]:
    """소스 DF → 학습·예측 공용 feature 매트릭스.

    한 행 = (customer_id, year_month, horizon_days) 로 고유.
    """
    daily = daily_by_customer(df)
    monthly = monthly_by_customer(daily)
    horizon = build_horizon_table(daily, horizons=(10, 20))

    # 래그 피처
    mlook = monthly.drop_duplicates([CUSTOMER_ID, "year_month"]).set_index([CUSTOMER_ID, "year_month"])["monthly_kwh"]
    mean_hist = (
        monthly.groupby(CUSTOMER_ID, observed=True)["monthly_kwh"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    monthly_with_hist = monthly.copy()
    monthly_with_hist["customer_mean_history"] = mean_hist.values

    hist_lookup = monthly_with_hist.drop_duplicates([CUSTOMER_ID, "year_month"]).set_index([CUSTOMER_ID, "year_month"])[
        "customer_mean_history"
    ]

    horizon["month"] = horizon["year_month"].dt.month
    horizon["partial_rate"] = horizon["partial_kwh"] / horizon["days_observed"].replace(
        0, np.nan
    )
    horizon["prev_month_kwh"] = [
        mlook.get((c, ym - 1), np.nan)
        for c, ym in zip(horizon[CUSTOMER_ID], horizon["year_month"])
    ]
    horizon["yoy_month_kwh"] = [
        mlook.get((c, ym - 12), np.nan)
        for c, ym in zip(horizon[CUSTOMER_ID], horizon["year_month"])
    ]
    horizon["ma3_kwh"] = [
        float(np.nanmean([float(mlook.get((c, ym - k), np.nan)) for k in (1, 2, 3)]))
        for c, ym in zip(horizon[CUSTOMER_ID], horizon["year_month"])
    ]
    horizon["customer_mean_history"] = [
        hist_lookup.get((c, ym), np.nan)
        for c, ym in zip(horizon[CUSTOMER_ID], horizon["year_month"])
    ]

    # 기상 피처
    weather = _weather_aggs(df, horizon)
    if weather is not None:
        horizon = horizon.merge(
            weather, on=[CUSTOMER_ID, "year_month", "horizon_days"], how="left"
        )

    # AMI 피처 (15분 패턴 활용)
    ami = _ami_features(df, horizon)
    has_ami = ami is not None and len(ami) > 0
    if has_ami:
        horizon = horizon.merge(
            ami, on=[CUSTOMER_ID, "year_month", "horizon_days"], how="left"
        )
        # 전월 대비 역률 변화 (냉방 시작 감지)
        if "observed_power_factor" in horizon.columns:
            _pf_df = horizon.drop_duplicates([CUSTOMER_ID, "year_month", "horizon_days"])
            pf_lookup = _pf_df.set_index(
                [CUSTOMER_ID, "year_month", "horizon_days"]
            )["observed_power_factor"].to_dict()
            horizon["pf_vs_prev_month"] = [
                float(pf_lookup.get((c, ym, h), np.nan))
                - float(pf_lookup.get((c, ym - 1, h), np.nan))
                for c, ym, h in zip(
                    horizon[CUSTOMER_ID],
                    horizon["year_month"],
                    horizon["horizon_days"],
                )
            ]

    # 특수일 피처
    try:
        sd = _special_day_features(horizon)
        horizon = horizon.merge(sd, on="year_month", how="left")
        has_special = True
    except Exception:
        has_special = False

    # 추가 컬럼 자동 감지 (세대수, 계약전력, 지역코드 등)
    extra_num = []
    cust_extra = df[[CUSTOMER_ID]].drop_duplicates()
    for col in ["n_households", "contract_power_kw"]:
        if col in df.columns:
            lookup = df.groupby(CUSTOMER_ID)[col].first().reset_index()
            horizon = horizon.merge(lookup, on=CUSTOMER_ID, how="left")
            extra_num.append(col)
    if "n_households" in horizon.columns and "partial_kwh" in horizon.columns:
        horizon["kwh_per_household"] = (
            horizon["partial_kwh"] / horizon["n_households"].clip(lower=1)
        )
        extra_num.append("kwh_per_household")

    # 카테고리 피처 자동 감지
    extra_cat = []
    _CAT_CANDIDATES = [
        "region_code",      # 본부
        "usage_purpose",    # 전기사용용도
        "industry_code",    # 산업분류
    ]
    for col in _CAT_CANDIDATES:
        if col in df.columns and df[col].notna().sum() > 0:
            nunique = df[col].nunique()
            if nunique > 1:
                lookup = df.groupby(CUSTOMER_ID)[col].first().reset_index()
                horizon = horizon.merge(lookup, on=CUSTOMER_ID, how="left")
                extra_cat.append(col)
                print(f"  [피처] {col}: {nunique}종 카테고리 추가")

    spec = FeatureSpec(
        has_weather=weather is not None,
        has_ami=has_ami,
        has_special_days=has_special,
        extra_numeric=extra_num,
        categorical=extra_cat,
    )
    return horizon, spec


def train(
    features: pd.DataFrame,
    spec: FeatureSpec,
    *,
    test_periods: Iterable[pd.Period] | None = None,
    train_end: pd.Period | None = None,
    params: dict | None = None,
    num_boost_round: int = 400,
    early_stopping_rounds: int = 30,
) -> tuple[lgb.Booster, pd.DataFrame]:
    """시간 분할 학습. train_end 이하 = train, 이후 = test."""
    features = features.dropna(subset=["full_month_kwh", "partial_kwh"]).copy()
    # categorical encoding — LightGBM 은 category dtype 지원
    for c in spec.categorical:
        features[c] = features[c].astype("category")

    if train_end is None:
        # 자동: 데이터의 마지막 연도를 test 로
        max_ym = features["year_month"].max()
        train_end = pd.Period(f"{max_ym.year - 1}-12", freq="M")

    train_mask = features["year_month"] <= train_end
    test_mask = ~train_mask
    if test_periods is not None:
        test_mask &= features["year_month"].isin(test_periods)

    # partial_linear 베이스라인: 잔차 학습 타겟
    pl_pred_all = (
        features["partial_kwh"] / features["days_observed"].clip(lower=1) * features["days_in_month"]
    )
    y_residual = features["full_month_kwh"] - pl_pred_all

    X_tr = features.loc[train_mask, spec.all]
    y_tr = y_residual.loc[train_mask]
    X_te = features.loc[test_mask, spec.all]
    y_te = y_residual.loc[test_mask]

    print(
        f"[lgbm] train={len(X_tr):,}  test={len(X_te):,}  "
        f"train_end={train_end}  features={len(spec.all)}  (잔차 학습)"
    )

    default_params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=spec.categorical)
    dte = lgb.Dataset(
        X_te,
        label=y_te,
        reference=dtr,
        categorical_feature=spec.categorical,
    )
    evals_result: dict = {}
    booster = lgb.train(
        default_params,
        dtr,
        num_boost_round=num_boost_round,
        valid_sets=[dtr, dte],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals_result),
        ],
    )

    # 학습 곡선 로그 저장 (generate_figures fig06용)
    if evals_result:
        try:
            import json as _json
            log_dir = Path("weights/dsz_lgbm")
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / "training_log.json", "w") as f:
                _json.dump(evals_result, f)
        except Exception:
            pass

    # 테스트 예측: partial_linear + 잔차 보정 + 클리핑
    test_preds = features.loc[test_mask, [CUSTOMER_ID, "year_month", "horizon_days"]].copy()
    residual_pred = booster.predict(X_te, num_iteration=booster.best_iteration)
    pl_base = pl_pred_all.loc[test_mask].values
    raw_pred = pl_base + residual_pred
    partial = features.loc[test_mask, "partial_kwh"].values
    test_preds["pred_monthly_kwh"] = np.maximum(raw_pred, partial)
    n_clipped = int((raw_pred < partial).sum())
    if n_clipped > 0:
        print(f"  [클리핑] pred < partial_kwh: {n_clipped}건 보정", flush=True)
    print(f"  [잔차 학습] 잔차 평균={residual_pred.mean():.1f}, std={residual_pred.std():.1f}", flush=True)
    return booster, test_preds


def save(
    booster: lgb.Booster,
    spec: FeatureSpec,
    out_dir: str | Path,
    *,
    notes: str = "",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.txt"))
    meta = {
        "feature_columns": spec.all,
        "numeric_columns": spec.numeric,
        "categorical_columns": spec.categorical,
        "extra_numeric": spec.extra_numeric,
        "has_weather": spec.has_weather,
        "has_ami": spec.has_ami,
        "has_special_days": spec.has_special_days,
        "best_iteration": booster.best_iteration,
        "num_trees": booster.num_trees(),
        "notes": notes,
    }
    with open(out / "feature_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load(weights_dir: str | Path) -> tuple[lgb.Booster, FeatureSpec]:
    p = Path(weights_dir)
    booster = lgb.Booster(model_file=str(p / "model.txt"))
    with open(p / "feature_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    spec = FeatureSpec(
        has_weather=bool(meta.get("has_weather", False)),
        has_ami=bool(meta.get("has_ami", False)),
        has_special_days=bool(meta.get("has_special_days", False)),
        extra_numeric=meta.get("extra_numeric", []),
        categorical=meta.get("categorical_columns", ["contract_type"]),
    )
    return booster, spec


def predict(
    booster: lgb.Booster, features: pd.DataFrame, spec: FeatureSpec
) -> pd.DataFrame:
    df = features.copy()
    for c in spec.categorical:
        df[c] = df[c].astype("category")
    preds = booster.predict(df[spec.all], num_iteration=booster.best_iteration)
    out = df[[CUSTOMER_ID, "year_month", "horizon_days"]].copy()
    out["pred_monthly_kwh"] = preds
    return out
