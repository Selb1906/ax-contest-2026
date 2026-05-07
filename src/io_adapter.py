"""소스-스위처블 LP 로더.

세 가지 소스를 동일한 표준 스키마로 정규화하여 하류에 전달.

1. synth    — `src.synth` 로 만든 parquet
2. public_apt — 한전 공공데이터 공동주택 전력·기상 융합 CSV (격자×시간)
3. dsz_lp   — 안심구역 내 실 LP (컬럼 매핑은 스카우팅 방문 후 확정)

설정은 YAML 한 파일로 관리:

    source:
      kind: synth | public_apt | dsz_lp
      path: <파일/디렉터리 경로>
      column_map:   # dsz_lp 에서만 사용
        customer_id: "고객번호"
        ts: "검침일시"
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import schemas
from .schemas import (
    CONTRACT_TYPE,
    CUSTOMER_ID,
    INTERVAL_MINUTES,
    MAX_DEMAND_KW,
    P_ACTIVE_KWH,
    P_APPARENT_KWH,
    P_REACTIVE_KWH,
    TS,
    normalize_contract_type,
)


@dataclass
class SourceConfig:
    kind: str
    path: Path
    column_map: dict[str, str]
    contract_type_default: str | None = None
    filters: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SourceConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        s = raw["source"]
        return cls(
            kind=str(s["kind"]),
            path=Path(s["path"]),
            column_map=dict(s.get("column_map", {})),
            contract_type_default=s.get("contract_type_default"),
            filters=s.get("filters"),
        )


# -----------------------------------------------------------
# 개별 로더
# -----------------------------------------------------------


def _load_synth(cfg: SourceConfig) -> pd.DataFrame:
    """src.synth 로 만든 parquet (partition=customer_id) 로드."""
    df = pd.read_parquet(cfg.path, engine="pyarrow")
    # partition column 은 읽을 때 카테고리 문자열로 복원됨
    if df[TS].dtype == object:
        df[TS] = pd.to_datetime(df[TS])
    return df


def _load_public_apt(cfg: SourceConfig) -> pd.DataFrame:
    """한전 공동주택 전력·기상 융합데이터(cp949 CSV, 격자×시간) 로드.

    격자 단위를 "고객" 으로 매핑한다 (customer_id = 'grid_{X}_{Y}').
    15분 단위가 아니라 시간 단위이므로 ts 는 `시:00` 으로만 설정.
    무효전력·최대수요 컬럼이 없어 NaN 으로 남긴다.
    """
    src_paths: list[Path] = []
    p = cfg.path
    if p.is_dir():
        src_paths = sorted(p.glob("한국전력공사_공동주택 전력·기상 융합데이터_*.csv"))
    elif p.suffix.lower() == ".csv":
        src_paths = [p]
    else:
        raise FileNotFoundError(f"public_apt: {p} 에서 CSV를 찾을 수 없음")

    frames: list[pd.DataFrame] = []
    for sp in src_paths:
        d = pd.read_csv(sp, encoding="cp949")
        ts = pd.to_datetime(
            dict(year=d["연도"], month=d["월"], day=d["일"], hour=d["시"].clip(upper=23)),
            errors="coerce",
        )
        cust = "grid_" + d["X격자"].astype(int).astype(str) + "_" + d["Y격자"].astype(int).astype(str)
        n_household = d["공동주택수"].astype(float)
        # 전력부하합계는 격자 총합 → 단지 평균으로 정규화 (가구 기준이 아니라 단지 평균)
        # 해석: 단지 평균 부하(kWh) — 이것만으로도 기상 반응 학습엔 충분
        active_per_unit = d["전력부하합계"].astype(float) / n_household.replace(0, np.nan)

        out = pd.DataFrame(
            {
                CUSTOMER_ID: cust,
                TS: ts,
                CONTRACT_TYPE: "주택용",
                P_ACTIVE_KWH: active_per_unit,
            }
        )
        # 기상 컬럼은 별도 테이블로 join 할 때를 위해 side-car로 남긴다
        out["_temp_c"] = d["기온"].astype(float)
        out["_humidity"] = d["상대습도"].astype(float)
        out["_wind"] = d["풍속"].astype(float)
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[TS, P_ACTIVE_KWH])
    return df


def _load_dsz_lp(cfg: SourceConfig) -> pd.DataFrame:
    """안심구역 실 LP 로드 — column_map 으로 컬럼을 표준으로 rename.

    AX 경진대회 데이터 컬럼 (한전 제공):
      계기번호, 계약번호, 검침년월일, 검침시분, 검침주기,
      유효전력량계, 지상무효전력량계, 진상무효전력량계, 피상전력량계,
      본부, 지사, 계약종별, 공급방식, 계약전력, 전기사용용도, 산업분류

    자동 처리:
      - 검침년월일 + 검침시분 → ts (2400→0000+1일 보정 포함)
      - 지상무효 + 진상무효 → p_reactive_kwh 합산
      - 공급방식/전기사용용도/산업분류/지사 → 카테고리 피처로 보존
    """
    if not cfg.column_map:
        raise ValueError("dsz_lp: column_map 이 필요합니다 (스카우팅 후 설정)")

    # 디렉터리면 csv/parquet 모두 탐색
    paths: list[Path] = []
    if cfg.path.is_dir():
        paths = sorted(
            list(cfg.path.glob("*.csv")) + list(cfg.path.glob("*.CSV"))
            + list(cfg.path.glob("*.parquet"))
            + list(cfg.path.glob("*.xlsx")) + list(cfg.path.glob("*.xls"))
        )
    else:
        paths = [cfg.path]
    if not paths:
        raise FileNotFoundError(f"dsz_lp: {cfg.path} 에 파일이 없음")

    frames: list[pd.DataFrame] = []
    reverse_map = {v: k for k, v in cfg.column_map.items()}
    for sp in paths:
        if sp.suffix.lower() == ".parquet":
            d = pd.read_parquet(sp)
        elif sp.suffix.lower() in (".xlsx", ".xls"):
            d = pd.read_excel(sp)
        else:
            # 구분자 자동 감지 (쉼표/탭)
            with open(sp, "r", encoding="utf-8-sig", errors="replace") as _f:
                first_line = _f.readline()
            sep = "\t" if "\t" in first_line else ","

            # 인코딩 자동 감지
            d = None
            for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
                try:
                    d = pd.read_csv(sp, encoding=enc, sep=sep, on_bad_lines="skip")
                    print(f"  [인코딩] {sp.name}: {enc}, 구분자: {'탭' if sep == chr(9) else '쉼표'}")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception:
                    try:
                        d = pd.read_csv(sp, encoding=enc, sep=sep,
                                        on_bad_lines="skip", engine="python")
                        print(f"  [인코딩] {sp.name}: {enc} (python engine)")
                        break
                    except Exception:
                        continue
            if d is None:
                d = pd.read_csv(sp, encoding="utf-8", errors="replace",
                                sep=sep, on_bad_lines="skip")
                print(f"  [인코딩] {sp.name}: utf-8 (깨진 문자 대체)")
        d = d.rename(columns=reverse_map)

        # 날짜+시간 분리 컬럼 처리 (검침년월일 + 검침시분 → ts)
        if TS not in d.columns and "ts_date" in d.columns and "ts_time" in d.columns:
            date_str = d["ts_date"].astype(str).str.strip()
            time_str = d["ts_time"].astype(str).str.strip()

            # 형식 자동 감지: "0:00" (H:MM) vs "0000" (HHMM)
            if time_str.str.contains(":").any():
                # H:MM 또는 HH:MM 형식 → 그대로 결합
                is_2400 = time_str.isin(["24:00", "2400"])
                time_str = time_str.replace("24:00", "00:00")
                combined = date_str + " " + time_str
            else:
                # HHMM 형식 → HH:MM으로 변환
                time_str = time_str.str.zfill(4)
                is_2400 = time_str == "2400"
                time_str = time_str.replace("2400", "0000")
                combined = date_str + " " + time_str.str[:2] + ":" + time_str.str[2:]

            d[TS] = pd.to_datetime(combined, errors="coerce", format="mixed")
            nat_count = d[TS].isna().sum()
            if nat_count == len(d):
                print(f"  [경고] ts 전부 파싱 실패! 검침시분 샘플: {time_str.head(3).tolist()}")
            elif nat_count > 0:
                print(f"  [경고] ts 파싱 실패 {nat_count}건 / {len(d)}건")
            if is_2400.any():
                d.loc[is_2400, TS] = d.loc[is_2400, TS] + pd.Timedelta(days=1)
            d = d.drop(columns=["ts_date", "ts_time"], errors="ignore")
            print(f"  [ts 조합] 검침년월일 + 검침시분 → ts ({d[TS].iloc[0]})")

        # 무효전력 지상/진상 합산
        if "p_reactive_lag" in d.columns or "p_reactive_lead" in d.columns:
            lag = pd.to_numeric(d.get("p_reactive_lag", 0), errors="coerce").fillna(0)
            lead = pd.to_numeric(d.get("p_reactive_lead", 0), errors="coerce").fillna(0)
            d[P_REACTIVE_KWH] = lag + lead
            d = d.drop(columns=["p_reactive_lag", "p_reactive_lead"], errors="ignore")
            print(f"  [무효전력] 지상 + 진상 → p_reactive_kwh 합산")

        if TS in d.columns:
            # 시간 형식 자동 감지 + 보정
            ts_raw = d[TS]

            # "20220101 2400" 또는 "2022-01-01 24:00" 같은 24시 형식 처리
            if ts_raw.dtype == object:
                ts_str = ts_raw.astype(str)
                has_24 = ts_str.str.contains("24:00|2400| 24:", regex=True).any()
                if has_24:
                    print(f"  [시간보정] 24시 형식 감지 → 00시+1일로 변환")
                    ts_str = ts_str.str.replace("24:00", "00:00", regex=False)
                    ts_str = ts_str.str.replace("2400", "0000", regex=False)
                    ts_str = ts_str.str.replace(" 24:", " 00:", regex=False)
                    d[TS] = pd.to_datetime(ts_str, errors="coerce") + pd.Timedelta(days=1)
                else:
                    d[TS] = pd.to_datetime(ts_raw, errors="coerce")
            else:
                d[TS] = pd.to_datetime(ts_raw, errors="coerce")

            # 01~24시 형식 감지 (시간이 1~24이고 0이 없는 경우)
            if d[TS].dt.hour.min() >= 1 and (d[TS].dt.hour == 0).sum() == 0:
                # 15분 단위에서 hour=24 문제는 위에서 처리됨
                # hour=1~23만 있고 0이 없으면 1시간 빼기
                print(f"  [시간보정] 01~24시 형식 의심 → 1시간 차감")
                d[TS] = d[TS] - pd.Timedelta(hours=1)

        if CONTRACT_TYPE not in d.columns and cfg.contract_type_default:
            d[CONTRACT_TYPE] = cfg.contract_type_default

        # 지사 → region_sub (본부보다 세분화된 지역)
        if "region_sub" in d.columns and "region_code" not in d.columns:
            d["region_code"] = d["region_sub"]
            d = d.drop(columns=["region_sub"])
        elif "region_sub" in d.columns and "region_code" in d.columns:
            pass  # 본부(region_code) 우선, 지사는 별도 유지

        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return df


LOADERS = {
    "synth": _load_synth,
    "public_apt": _load_public_apt,
    "dsz_lp": _load_dsz_lp,
}


# -----------------------------------------------------------
# 공용 엔트리
# -----------------------------------------------------------


def load(cfg: SourceConfig, *, validate: bool = True) -> pd.DataFrame:
    loader = LOADERS.get(cfg.kind)
    if loader is None:
        raise ValueError(f"unknown source kind: {cfg.kind} (known={list(LOADERS)})")
    df = loader(cfg)
    # 필수 컬럼 존재 보장 (side-car 컬럼은 남겨둠)
    df = schemas.ensure_columns(
        df, [P_REACTIVE_KWH, P_APPARENT_KWH, MAX_DEMAND_KW]
    )
    # 계약종별 정규화 (다양한 표기 → 표준 형태)
    if CONTRACT_TYPE in df.columns:
        raw_types = df[CONTRACT_TYPE].dropna().unique()
        df[CONTRACT_TYPE] = df[CONTRACT_TYPE].map(
            lambda x: normalize_contract_type(x) if isinstance(x, str) else x
        )
        new_types = df[CONTRACT_TYPE].dropna().unique()
        changed = set(raw_types) - set(new_types)
        if changed:
            print(f"  [계약종별 정규화] {len(raw_types)}종 → {len(new_types)}종")
        print(f"  계약종별: {sorted(new_types)}")
    if validate:
        schemas.validate(df, strict=False)
    return df


def load_from_yaml(path: str | Path, **kwargs) -> pd.DataFrame:
    return load(SourceConfig.from_yaml(path), **kwargs)
