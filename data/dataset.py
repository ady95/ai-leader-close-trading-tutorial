"""연구용 일봉 패널 — 기간 분할(개발/홀드아웃 봉인)과 품질 검사.

기간 정책 (실험 시작 전 확정, 이후 변경하지 않는다)
  개발(dev)      2023-01-02 ~ 2025-12-31   일봉 실험(5~7장)의 탐색·설계·튜닝
  홀드아웃        2026-01-01 ~             15장 최종 검증 전까지 열지 않는다

1분봉·틱 실험은 1년치뿐이라 별도의 시간순 Walk-Forward(15-1)로 검증한다.

라벨(다음 날 시가 등)은 같은 기간 안에서만 만든다. 개발 기간 마지막 날의 라벨은
홀드아웃 첫날 가격이 필요하므로 비워 둔다 — 봉인 경계에서 한 칸도 새지 않게 하기 위해서다.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

from data.store import path

DEV_START, DEV_END = "20230101", "20251231"
HOLDOUT_START = "20260101"
UNSEAL_ENV = "RESEARCH_UNSEAL_HOLDOUT"

AFTERHOURS_START = "20260914"   # 이날부터 키움 일봉 종가 = 장후 마지막 가격(19:59), 정규장 종가 아님 (3-3)
PRICE_LIMIT = 0.30          # 일일 가격제한폭 ±30%
LIMIT_UP_EPS = 0.295        # 이 이상 오른 날은 상한가(근처)로 본다 — 종가 매수 체결이 불확실


class HoldoutSealed(RuntimeError):
    pass


def _latest(pattern: str) -> pd.DataFrame:
    files = sorted(path("universe").glob(pattern))
    if not files:
        raise FileNotFoundError(f"universe/{pattern} 이 없습니다. scripts.backfill 을 먼저 실행하세요.")
    return pd.read_parquet(files[-1])


def universe() -> pd.DataFrame:
    """연구 대상 보통주 + 시장 구분 + 업종 (최신 스냅샷)."""
    u = _latest("stocks_*.parquet")
    u = u.loc[u["is_common"], ["code", "name", "marketCode", "listCount"]].rename(columns={"code": "symbol"})
    u["market"] = np.where(u["marketCode"].astype(str) == "0", "KOSPI", "KOSDAQ")
    try:
        sec = _latest("sectors_*.parquet").loc[:, ["symbol", "inds_cd", "inds_nm"]]
        u = u.merge(sec, on="symbol", how="left")
    except FileNotFoundError:
        u["inds_cd"], u["inds_nm"] = None, None
    # 코스피·코스닥의 같은 이름 업종(예: 전기/전자)은 하나의 섹터로 본다.
    u["sector"] = u["inds_nm"].fillna("미분류")
    return u.drop(columns=["marketCode"])


@lru_cache(maxsize=1)
def _raw_daily() -> pd.DataFrame:
    frames = []
    cols = ["symbol", "dt", "open", "high", "low", "close", "volume", "value_mil"]
    for p in sorted(path("daily").glob("*.parquet")):
        frames.append(pd.read_parquet(p, columns=cols))
    df = pd.concat(frames, ignore_index=True)
    for c in ("open", "high", "low", "close", "volume", "value_mil"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["dt"] = df["dt"].astype(str)
    return df


def regular_session_daily(symbol: str) -> pd.DataFrame:
    """1분봉으로 만든 KRX 정규장 일봉 (09:00~15:30). close = 15:30 종가 단일가 봉의 종가."""
    p = path("minute", f"{symbol}.parquet")
    if not p.exists():
        return pd.DataFrame()
    m = pd.read_parquet(p, columns=["ts", "open", "high", "low", "close", "volume"])
    m["dt"], m["hm"] = m["ts"].str[:8], m["ts"].str[8:12]
    m = m[(m["hm"] >= "0900") & (m["hm"] <= "1530")].sort_values("ts")
    g = m.groupby("dt")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": m[m["hm"] == "1530"].set_index("dt")["close"],
                        "volume": g["volume"].sum()})
    return out.dropna(subset=["close"]).reset_index()


def _fix_afterhours_close(df: pd.DataFrame) -> pd.DataFrame:
    """AFTERHOURS_START 이후 행의 OHLC를 1분봉 정규장 값으로 바꾼다. 1분봉이 없으면 close 를 비운다."""
    late = df["dt"] >= AFTERHOURS_START
    df["close_source"] = np.where(late, "unavailable", "daily")
    if not late.any():
        return df
    fixes = []
    for s in df.loc[late, "symbol"].unique():
        r = regular_session_daily(s)
        if not r.empty:
            fixes.append(r[r["dt"] >= AFTERHOURS_START].assign(symbol=s))
    if fixes:
        f = pd.concat(fixes).set_index(["symbol", "dt"])
        idx = pd.MultiIndex.from_frame(df.loc[late, ["symbol", "dt"]])
        hit = idx.isin(f.index)
        rows = df.index[late][hit]
        vals = f.loc[idx[hit]]
        for c in ("open", "high", "low", "close"):
            df.loc[rows, c] = vals[c].to_numpy()
        df.loc[rows, "close_source"] = "minute_1530"
    df.loc[late & (df["close_source"] == "unavailable"), "close"] = np.nan
    return df


def _check_period(period: str) -> tuple[str, str]:
    if period == "dev":
        return DEV_START, DEV_END
    if period in ("holdout", "all"):
        if os.environ.get(UNSEAL_ENV) != "1":
            raise HoldoutSealed(
                f"홀드아웃({HOLDOUT_START}~)은 봉인돼 있습니다. 최종 검증(15-2)에서만 "
                f"환경변수 {UNSEAL_ENV}=1 로 엽니다.")
        return (HOLDOUT_START if period == "holdout" else DEV_START), "99991231"
    raise ValueError(period)


def load_daily(period: str = "dev") -> pd.DataFrame:
    """일봉 패널. 다음 날 가격 라벨(nxt_*)과 기본 파생값을 붙인다.

    열: symbol, dt(YYYYMMDD), date, open..close, volume, value(원), ret(전일 종가 대비),
        tradable(거래량>0), limit_up, nxt_open, nxt_close, nxt_tradable,
        r_on(종가→익일 시가), r_id(익일 시가→종가)
    """
    start, end = _check_period(period)
    df = _raw_daily()
    df = df[(df["dt"] >= start) & (df["dt"] <= end)].copy()
    df = df.sort_values(["symbol", "dt"]).reset_index(drop=True)
    df = _fix_afterhours_close(df)
    df["date"] = pd.to_datetime(df["dt"])
    df["value"] = df["value_mil"] * 1e6
    g = df.groupby("symbol", sort=False)
    df["prev_close"] = g["close"].shift(1)
    df["ret"] = df["close"] / df["prev_close"] - 1
    df["tradable"] = df["volume"] > 0
    df["limit_up"] = df["ret"] >= LIMIT_UP_EPS
    df["nxt_dt"] = g["dt"].shift(-1)
    df["nxt_open"] = g["open"].shift(-1)
    df["nxt_close"] = g["close"].shift(-1)
    df["nxt_tradable"] = g["volume"].shift(-1) > 0
    df["r_on"] = df["nxt_open"] / df["close"] - 1
    df["r_id"] = df["nxt_close"] / df["nxt_open"] - 1
    u = universe().loc[:, ["symbol", "name", "market", "sector", "listCount"]]
    return df.merge(u, on="symbol", how="left")


def trading_calendar(period: str = "dev") -> list[str]:
    """코스피 지수 일봉의 날짜 = 거래일 달력 (지수가 없으면 일봉 패널에서 추정)."""
    start, end = _check_period(period)
    p = path("index", "001.parquet")
    if p.exists():
        d = pd.read_parquet(p, columns=["dt", "volume"])
        d = d[pd.to_numeric(d["volume"], errors="coerce") > 0]
        days = d["dt"].astype(str)
    else:
        days = _raw_daily()["dt"]
    return sorted(x for x in set(days) if start <= x <= end)


def quality_report(df: pd.DataFrame, calendar: list[str]) -> pd.Series:
    """일봉 패널 품질 검사. 0 이 아닌 항목은 원인을 확인하고 처리 방법을 기록한다."""
    first = df.groupby("symbol")["dt"].transform("min")
    expected = {s: sum(1 for c in calendar if c >= f) for s, f in
                df.groupby("symbol")["dt"].min().items()}
    have = df.groupby("symbol")["dt"].nunique()
    missing = sum(max(expected[s] - have[s], 0) for s in have.index)
    bad_ohlc = ((df["low"] > df[["open", "close"]].min(axis=1)) |
                (df["high"] < df[["open", "close"]].max(axis=1))) & df["tradable"]
    return pd.Series({
        "종목 수": df["symbol"].nunique(),
        "행 수": len(df),
        "중복 (symbol, dt)": int(df.duplicated(["symbol", "dt"]).sum()),
        "거래일 달력 밖 날짜": int((~df["dt"].isin(set(calendar))).sum()),
        "상장 이후 누락 거래일": int(missing),
        "거래량 0 (거래정지 등)": int((~df["tradable"]).sum()),
        "OHLC 모순": int(bad_ohlc.sum()),
        "가격제한폭 초과 등락 (|ret|>30%)": int((df["ret"].abs() > PRICE_LIMIT + 0.005).sum()),
        "기간 중 신규 상장": int((first > calendar[0]).groupby(df["symbol"]).first().sum()),
    })
