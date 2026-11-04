"""거래비용 모델 — 수수료·증권거래세(농특세 포함)·슬리피지.

세율은 매도일 기준이다. 수수료는 증권사·계좌·이벤트마다 다르므로 자신의 조건으로 바꿔 쓴다.
슬리피지는 일봉 실험에서 알 수 없으므로 별도 인자로 두고 민감도를 함께 보고한다(9-5).
"""
from __future__ import annotations

import pandas as pd

FEE_RATE = 0.00015            # 매수·매도 각각 (키움 비대면 계좌 기본 수수료 가정)
SELL_TAX_BY_YEAR = {          # 코스피(거래세+농특세)·코스닥 공통 합계 세율
    2023: 0.0020,
    2024: 0.0018,
    2025: 0.0015,
    2026: 0.0020,             # 2026년 세율은 매매 전에 다시 확인할 것
}


def sell_tax(year: int) -> float:
    return SELL_TAX_BY_YEAR.get(year, max(SELL_TAX_BY_YEAR.values()))


def round_trip_cost(sell_dates: pd.Series, slippage: float = 0.0) -> pd.Series:
    """1회 왕복(매수+매도) 비용률. slippage 는 편도 가격 불리분(예: 0.001 = 0.1%)."""
    years = pd.to_datetime(sell_dates.astype(str)).dt.year
    return 2 * FEE_RATE + years.map(sell_tax) + 2 * slippage


def net_return(gross: pd.Series, sell_dates: pd.Series, slippage: float = 0.0) -> pd.Series:
    return gross - round_trip_cost(sell_dates, slippage)
