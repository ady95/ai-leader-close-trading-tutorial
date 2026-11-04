"""P2 — 상승추세 눌림목 재상승형 (일봉).

정의는 실험 전에 고정했다(7-4). 결과를 보고 임계값을 바꾸지 않는다.

  주도 후보:  그날 거래대금 상위 100, 상승, 상한가 아님 (P1과 같음)
  상승추세:   MA20 > MA60, 종가 > MA20, 최근 20일 고점 ≥ 252일 고점의 90% (고점권 주도주)
  눌림:       최근 20일 고점이 2~10거래일 전에 나왔고,
              그 뒤 저점까지의 하락폭(눌림 깊이)이 3~15%,
              직전 5거래일 종가가 모두 MA20 위 (20일선 유지)
  재상승:     오늘 종가 > 전일 고가, 그리고 종가 ≤ 최근 20일 고점

P1·P2 겹침 정책: 최근 20일 고점을 넘긴 날은 P2가 아니다(고점 돌파는 P1의 영역).
252일 고점 ≥ 20일 고점이므로 P1(종가 > 252일 고점)과 P2(종가 ≤ 20일 고점)는 동시에 성립하지 않는다.

역추세 반등(제외 대상, 7-1 비교용):
  주도 후보 & 오늘 종가 > 전일 고가 &
  (직전 10거래일 중 5일 이상 종가 < MA20  또는  60일 고점 대비 낙폭 ≥ 25%)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.entry_patterns.p1 import candidates


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """p1.add_features 를 먼저 적용한 패널에 P2 특징을 더한다."""
    g = df.groupby("symbol", sort=False)
    prev_high = g["high"].shift(1)
    df["prev_high"] = prev_high
    df["hh20_prev"] = g["high"].transform(lambda s: s.shift(1).rolling(20, min_periods=20).max())
    # 최근 20일 고점이 며칠 전에 나왔나 (1 = 어제)
    df["days_since_high"] = g["high"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).apply(lambda w: len(w) - int(np.argmax(w)), raw=True))
    df["ll5_prev"] = g["low"].transform(lambda s: s.shift(1).rolling(5, min_periods=5).min())
    df["pullback_depth"] = 1 - df["ll5_prev"] / df["hh20_prev"]
    above20 = (df["close"] > df["ma20"]).astype(float)
    df["above20_5d"] = above20.groupby(df["symbol"]).transform(lambda s: s.shift(1).rolling(5, min_periods=5).sum())
    df["below20_10d"] = (1 - above20).groupby(df["symbol"]).transform(
        lambda s: s.shift(1).rolling(10, min_periods=10).sum())
    df["dd60"] = 1 - df["close"] / df["hh60_prev"]
    df["high_ratio"] = df["hh20_prev"] / df["hh252_prev"]
    # 눌림 저점의 5일선 거리, 저점 상승(어제 저점 > 직전 5일 최저가)
    df["low_vs_ma5"] = df["ll5_prev"] / g["ma5"].shift(1) - 1
    df["higher_low"] = g["low"].shift(1) > df["ll5_prev"]
    df["reclaim"] = df["close"] > prev_high
    df["below_recent_high"] = df["close"] <= df["hh20_prev"]
    return df


def uptrend(df: pd.DataFrame) -> pd.Series:
    return (df["ma20"] > df["ma60"]) & (df["close"] > df["ma20"]) & (df["high_ratio"] >= 0.90)


def pullback(df: pd.DataFrame) -> pd.Series:
    return (df["days_since_high"].between(2, 10) & df["pullback_depth"].between(0.03, 0.15)
            & (df["above20_5d"] == 5))


def is_p2(df: pd.DataFrame) -> pd.Series:
    return candidates(df) & uptrend(df) & pullback(df) & df["reclaim"] & df["below_recent_high"]


def is_countertrend(df: pd.DataFrame) -> pd.Series:
    return candidates(df) & df["reclaim"] & ((df["below20_10d"] >= 5) | (df["dd60"] >= 0.25))
