"""성과 지표 — 거래 단위와 일별 포트폴리오 단위를 구분한다.

거래 단위: 평균(t값)·중앙값·승률·Profit Factor·최악 거래·CVaR(하위 5% 평균)
일별 포트폴리오: 같은 날 신호를 동일 비중으로 나눠 산 1일 수익의 누적 → MDD
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trade_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return {"건수": 0}
    gain, loss = r[r > 0].sum(), -r[r < 0].sum()
    q = r.quantile(0.05)
    return {
        "건수": int(len(r)),
        "평균": r.mean(),
        "t": r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan,
        "중앙값": r.median(),
        "승률": (r > 0).mean(),
        "PF": gain / loss if loss > 0 else np.inf,
        "최악": r.min(),
        "CVaR5": r[r <= q].mean(),
    }


def daily_portfolio(trades: pd.DataFrame, ret_col: str, date_col: str = "dt") -> pd.Series:
    """같은 날 진입한 거래를 동일 비중으로 묶은 일별 수익률 (신호 없는 날은 0)."""
    return trades.groupby(date_col)[ret_col].mean().sort_index()


def max_drawdown(daily: pd.Series) -> float:
    equity = (1 + daily.fillna(0)).cumprod()
    return float((equity / equity.cummax() - 1).min())


def summarize(trades: pd.DataFrame, ret_col: str, date_col: str = "dt") -> dict:
    out = trade_stats(trades[ret_col])
    if out["건수"]:
        d = daily_portfolio(trades, ret_col, date_col)
        out["거래일"] = int(len(d))
        out["누적(일별 동일비중)"] = float((1 + d).prod() - 1)
        out["MDD"] = max_drawdown(d)
    return out


def fmt_table(rows: dict[str, dict]) -> pd.DataFrame:
    """지표 dict 들을 책에 싣기 좋은 표로 (비율은 %)."""
    df = pd.DataFrame(rows).T
    pct = ["평균", "중앙값", "승률", "최악", "CVaR5", "누적(일별 동일비중)", "MDD"]
    for c in df.columns:
        if c in pct:
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{100 * x:.2f}%")
        elif c in ("PF", "t"):
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        else:
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{int(x):,}")
    return df
