"""S0·S1 청산 정책 (1분봉 근사). 명세는 실험 전에 고정했다(10-2).

공통
  진입가 entry: 15:18~15:19 대표가격 + 반 호가 (접속매매 매수는 매도호가에 산다고 보는 보수적 가정)
  갭 = 다음 날 09:00 시가 단일가 ÷ entry - 1.  상승 ≥ +0.5%, 하락 ≤ -0.5%, 그 사이 보합.

S1  다음 날 시가 단일가에 전량 매도 (단일가라 스프레드 없음)
S0  갭 상승: 1/3 시가 단일가 매도, 2/3 는 09:00~09:04 거래량 가중가격에서 반 호가 불리하게 매도
    보합·갭 하락: 전량을 09:00~09:04 거래량 가중가격에서 반 호가 불리하게 매도 (09:05까지 청산 시도)
S0b S0 와 같되 갭 하락이면 시가 단일가에 전량 매도 (민감도 확인용 변형)

09:05까지 다 팔리는지는 1분봉으로 알 수 없으므로, 포지션 크기 ÷ 09:00~09:04 거래대금(참여율)을
함께 보고한다. 틱·호가로 하는 정밀 재현은 10-4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAP_BAND = 0.005


def tick_size(price: pd.Series) -> pd.Series:
    """KRX 호가단위 (2023-01 개편 이후 코스피·코스닥 공통)."""
    bins = [0, 2_000, 5_000, 20_000, 50_000, 200_000, 500_000, np.inf]
    return pd.cut(price, bins, right=False, labels=[1, 5, 10, 50, 100, 500, 1_000]).astype(float)


def entry_price(df: pd.DataFrame) -> pd.Series:
    return df["p_t1"] + 0.5 * tick_size(df["p_t1"])


def gap_type(df: pd.DataFrame, entry: pd.Series) -> pd.Series:
    g = df["nxt_open"] / entry - 1
    return pd.Series(np.select([g >= GAP_BAND, g <= -GAP_BAND], ["갭상승", "갭하락"], "보합"), index=df.index)


def exit_prices(df: pd.DataFrame) -> pd.DataFrame:
    """정책별 평균 매도가."""
    entry = entry_price(df)
    gt = gap_type(df, entry)
    open_ = df["nxt_open"]
    cont = df["nx_vwap5"] - 0.5 * tick_size(df["nx_vwap5"])
    s0 = np.where(gt == "갭상승", open_ / 3 + cont * 2 / 3, cont)
    s0b = np.where(gt == "갭하락", open_, s0)
    return pd.DataFrame({"entry": entry, "gap_type": gt, "S1": open_, "S0": s0, "S0b": s0b}, index=df.index)
