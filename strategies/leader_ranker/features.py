"""주도주 탐색 특징 (일봉 기준).

모든 특징은 t일 장 마감 시점에 알 수 있는 값이다. 과거 구간 통계(20일 평균 등)는 t-1일까지로
계산해 t일 값과 섞이지 않게 한다. 15:18 진입에서는 일봉 대신 15:18 부분봉으로 같은 특징을
다시 계산해야 한다(9-1). 이 모듈의 결과는 "장 마감 정보를 다 안다"는 가정의 상한선이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.store import path


def stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """종목별 시계열 특징."""
    df = df.sort_values(["symbol", "dt"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    df["tv_ratio20"] = df["value"] / g["value"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    rng = (df["high"] - df["low"]).where(lambda x: x > 0)
    df["clv"] = (df["close"] - df["low"]) / rng
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    for n in (5, 20, 60):
        df[f"ma{n}"] = g["close"].transform(lambda s, n=n: s.rolling(n, min_periods=n).mean())
    df["hh252_prev"] = g["high"].transform(lambda s: s.shift(1).rolling(252, min_periods=120).max())
    df["near_high"] = df["close"] / df["hh252_prev"]
    df["ret5"] = g["close"].pct_change(5)
    return df


def cross_section(df: pd.DataFrame, top: int = 100) -> pd.DataFrame:
    """날짜별 순위·섹터 특징."""
    ok = df["tradable"]
    df["tv_rank"] = df["value"].where(ok).groupby(df["dt"]).rank(ascending=False, method="first")
    df["in_top"] = df["tv_rank"] <= top
    # 섹터: 그날 거래대금 상위 top 안에 든 같은 섹터 종목 수(자기 제외), 섹터 거래대금 비중
    known = df["sector"].notna() & ~df["sector"].str.endswith("미분류")
    key = [df["dt"], df["sector"]]
    df["sector_top_peers"] = (df["in_top"] & known).groupby(key).transform("sum") - (df["in_top"] & known)
    df.loc[~known, "sector_top_peers"] = np.nan
    mkt_tv = df.groupby("dt")["value"].transform("sum")
    df["sector_tv_share"] = df.groupby(key)["value"].transform("sum") / mkt_tv
    df.loc[~known, "sector_tv_share"] = np.nan
    w = df["value"].where(ok)
    rw = (df["ret"] * w).groupby(key).transform("sum")
    ws = w.groupby(key).transform("sum")
    df["sector_ret"] = rw / ws
    # 자기 자신을 뺀 동료 종목의 거래대금 가중 등락률 (상위 종목은 자기 등락이 섹터 평균을 끌어올린다)
    df["sector_ret_ex"] = (rw - (df["ret"] * w).fillna(0)) / (ws - w.fillna(0)).where(lambda x: x > 0)
    # 섹터 거래대금 비중의 평소(직전 20거래일 평균) 대비 배수 = 돈이 새로 몰리는 정도
    sd = (df.loc[known, ["dt", "sector", "sector_tv_share"]].drop_duplicates(["dt", "sector"])
          .sort_values(["sector", "dt"]))
    sd["sector_share_ratio20"] = sd["sector_tv_share"] / sd.groupby("sector")["sector_tv_share"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    # 섹터 안 거래대금 1위 = 섹터 대장주 후보
    df["sector_tv_rank"] = df["value"].where(ok & known).groupby(key).rank(ascending=False, method="first")
    df = df.merge(sd.loc[:, ["dt", "sector", "sector_share_ratio20"]], on=["dt", "sector"], how="left")
    return df


def sector_daily(df: pd.DataFrame) -> pd.DataFrame:
    """섹터×날짜 집계: 거래대금 비중, 상위 종목 수, 거래대금 가중 등락률."""
    known = df["sector"].notna() & ~df["sector"].str.endswith("미분류")
    d = df[known & df["tradable"]]
    agg = d.groupby(["dt", "sector"]).agg(
        tv=("value", "sum"), n_top=("in_top", "sum"), n=("symbol", "size"),
        ret=("sector_ret", "first")).reset_index()
    agg["tv_share"] = agg["tv"] / agg.groupby("dt")["tv"].transform("sum")
    return agg


def market_regime(index_code: str) -> pd.DataFrame:
    """지수 일봉 → 국면. 상승: 종가>MA20>MA60, 하락: 종가<MA20<MA60, 그 밖은 횡보."""
    d = pd.read_parquet(path("index", f"{index_code}.parquet"))
    d = d[pd.to_numeric(d["volume"], errors="coerce") > 0].sort_values("dt").copy()
    d["dt"] = d["dt"].astype(str)
    c = d["close"].astype(float)
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    d["regime"] = np.select([(c > ma20) & (ma20 > ma60), (c < ma20) & (ma20 < ma60)],
                            ["상승", "하락"], "횡보")
    d.loc[ma60.isna(), "regime"] = None
    d["idx_ret"] = c.pct_change()
    return d.loc[:, ["dt", "regime", "idx_ret"]]


def build(df: pd.DataFrame, top: int = 100) -> pd.DataFrame:
    df = cross_section(stock_features(df), top=top)
    reg = pd.concat([market_regime("001").assign(market="KOSPI"),
                     market_regime("101").assign(market="KOSDAQ")])
    return df.merge(reg, on=["dt", "market"], how="left")
