"""P1 — 신고가 모멘텀 연장형 (일봉).

정의는 실험 전에 고정했다(6-4). 임계값을 결과를 보고 바꾸지 않는다.

  주도 후보:  그날 거래대금 상위 100, 상승, 상한가 아님
  돌파:       종가 > 직전 252거래일 최고가(고가 기준)
  추세:       MA5 > MA20 > MA60 (정배열)

데이터가 2023-01부터라 '역사적 신고가'는 판정할 수 없다. 252일 최고가는 직전 120거래일 이상이
쌓인 뒤부터 계산한다(그 전은 판정 안 함).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOP_N = 100


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "dt"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    df["hh60_prev"] = g["high"].transform(lambda s: s.shift(1).rolling(60, min_periods=40).max())
    df["breakout252"] = df["close"] > df["hh252_prev"]
    df["breakout60"] = df["close"] > df["hh60_prev"]
    df["margin252"] = df["close"] / df["hh252_prev"] - 1
    df["ma_aligned"] = (df["ma5"] > df["ma20"]) & (df["ma20"] > df["ma60"])
    df["ma20_slope5"] = df["ma20"] / g["ma20"].shift(5) - 1
    # 직전 20거래일 안에 이미 252일 신고가 돌파가 있었나 (첫 돌파 vs 연속 신고가)
    prev_bo = g["breakout252"].transform(lambda s: s.shift(1).rolling(20, min_periods=1).max())
    df["fresh_breakout"] = df["breakout252"] & (prev_bo.fillna(0) == 0)
    # 라벨(미래 값): 다음 날 종가까지, 5거래일 뒤 종가까지
    df["r_oc"] = df["nxt_close"] / df["close"] - 1
    df["r_5d"] = g["close"].shift(-5) / df["close"] - 1
    return df


def candidates(df: pd.DataFrame) -> pd.Series:
    return (df["tv_rank"] <= TOP_N) & df["tradable"] & (df["ret"] > 0) & ~df["limit_up"]


def is_p1(df: pd.DataFrame) -> pd.Series:
    return candidates(df) & df["breakout252"] & df["ma_aligned"]


def near_high_bucket(x: pd.Series) -> pd.Series:
    return pd.cut(x, [0, 0.8, 0.9, 0.97, 1.0, np.inf],
                  labels=["~80%", "80~90%", "90~97%", "97~100%", "돌파(>100%)"])
