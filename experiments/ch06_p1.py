"""6장 실험 — P1 신고가 모멘텀 연장형 (일봉, 개발 기간 2023-01~2025-12).

사용법 (먼저 python -m experiments.ch05_leaders build)
  python -m experiments.ch06_p1 high       # 6-1 252일 고점 대비 위치별 성과
  python -m experiments.ch06_p1 ma         # 6-2 이동평균 정배열·기울기
  python -m experiments.ch06_p1 strength   # 6-3 돌파 강도·거래대금·종가 위치·윗꼬리
  python -m experiments.ch06_p1 e1         # 6-4 P1 후보와 E1 (P1 → S1)

성과는 세 가지 보유 기간으로 본다: 다음 날 시가(S1, 비용 후), 다음 날 종가(비용 후), 5거래일 뒤 종가(비용 전).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from backtest.costs import net_return
from backtest.metrics import fmt_table, summarize
from experiments.ch05_leaders import FEAT, by, e0_select
from strategies.entry_patterns import p1

pd.set_option("display.width", 200)


def load(extra=None) -> pd.DataFrame:
    """특징 패널 + P1 특징. extra 는 거래정지일을 빼기 전에 적용할 추가 특징 함수."""
    df = p1.add_features(pd.read_parquet(FEAT))
    if extra is not None:
        df = extra(df)
    df = df[df["tradable"] & (df["ret"].abs() <= 0.305)].copy()
    ok = df["nxt_tradable"] & df["r_on"].notna()
    df.loc[~ok, ["r_on", "r_oc"]] = np.nan
    df["net_on"] = net_return(df["r_on"], df["nxt_dt"].fillna("20251231"))
    df["net_oc"] = net_return(df["r_oc"], df["nxt_dt"].fillna("20251231"))
    return df


def horizons(groups: dict[str, pd.DataFrame]) -> str:
    rows = {}
    for k, g in groups.items():
        on, oc = summarize(g, "net_on"), summarize(g, "net_oc")
        r5 = g["r_5d"].dropna()
        rows[k] = {"건수": on.get("건수", 0), "시가 평균": on.get("평균"), "시가 t": on.get("t"),
                   "시가 승률": on.get("승률"), "익일종가 평균": oc.get("평균"),
                   "5일 평균(비용 전)": r5.mean(), "5일 중앙값": r5.median()}
    out = pd.DataFrame(rows).T
    for c in out.columns:
        if c == "건수":
            out[c] = out[c].map(lambda x: f"{int(x):,}")
        elif c == "시가 t":
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        else:
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{100 * x:.2f}%")
    return out.to_string()


def cmd_high() -> None:
    df = load()
    pool = df[p1.candidates(df) & df["near_high"].notna()]
    print("후보 = 거래대금 상위 100, 상승, 상한가 제외. 종가 ÷ 직전 252일 최고가")
    print(horizons(by(pool, p1.near_high_bucket(pool["near_high"]))))
    bo = pool[pool["breakout252"]]
    print("\n돌파일 중 첫 돌파(직전 20일 돌파 없음) vs 연속 신고가")
    print(horizons(by(bo, np.where(bo["fresh_breakout"], "첫 돌파", "연속 신고가"))))
    b60 = pool[pool["breakout60"] & ~pool["breakout252"]]
    print("\n60일 전고점만 돌파(252일 고점 아래)")
    print(horizons({"60일 돌파·252일 미돌파": b60}))


def cmd_ma() -> None:
    df = load()
    pool = df[p1.candidates(df) & df["ma60"].notna()]
    print(horizons(by(pool, np.where(pool["ma_aligned"], "정배열 5>20>60", "정배열 아님"))))
    bo = pool[pool["breakout252"]]
    print("\n252일 돌파일만")
    print(horizons(by(bo, np.where(bo["ma_aligned"], "돌파+정배열", "돌파+정배열 아님"))))
    al = pool[pool["ma_aligned"]]
    print("\n정배열 종목의 MA20 5일 기울기")
    print(horizons(by(al, pd.cut(al["ma20_slope5"], [-1, 0.01, 0.03, 0.06, 1],
                                 labels=["~1%", "1~3%", "3~6%", "6%~"]))))


def cmd_strength() -> None:
    df = load()
    ev = df[p1.is_p1(df)]
    print(f"P1 이벤트 {len(ev):,}건")
    for name, col, bins, labels in [
        ("돌파폭 (종가 ÷ 252일 고점 - 1)", "margin252", [0, 0.02, 0.05, 0.10, 1], ["~2%", "2~5%", "5~10%", "10%~"]),
        ("거래대금 ÷ 20일 평균", "tv_ratio20", [0, 1.5, 2.5, 4, 1e9], ["~1.5", "1.5~2.5", "2.5~4", "4~"]),
        ("종가 위치 CLV", "clv", [-0.01, 0.5, 0.7, 0.9, 1.01], ["~0.5", "0.5~0.7", "0.7~0.9", "0.9~1"]),
        ("윗꼬리 비율", "upper_wick", [-0.01, 0.1, 0.25, 0.5, 1.01], ["~0.1", "0.1~0.25", "0.25~0.5", "0.5~"]),
        ("당일 등락률", "ret", [0, 0.03, 0.07, 0.15, 0.3], ["0~3%", "3~7%", "7~15%", "15%~"]),
    ]:
        print(f"\n{name}")
        print(horizons(by(ev, pd.cut(ev[col], bins, labels=labels))))


def cmd_e1() -> None:
    df = load()
    ev = df[p1.is_p1(df)]
    e0 = e0_select(df[df["nxt_tradable"] & df["r_on"].notna()])
    print(f"P1 이벤트 {len(ev):,}건, {ev['dt'].nunique()}거래일, {ev['symbol'].nunique()}종목, "
          f"하루 평균 {len(ev) / ev['dt'].nunique():.1f}건")
    print(f"E0와 겹침: P1의 {ev.index.isin(e0.index).mean() * 100:.1f}%가 E0에도 속함")
    t = {"E1 = P1 → S1 (비용 후)": ev, "E0 (비용 후)": e0}
    print(fmt_table({k: summarize(v, "net_on") for k, v in t.items()}).to_string())
    print("\n보유 기간별")
    print(horizons({"P1": ev, "E0": e0}))
    print("\n연도별 E1")
    print(fmt_table({k: summarize(v, "net_on") for k, v in by(ev, ev["dt"].str[:4]).items()}).iloc[:, :8].to_string())
    print("\n시장 국면별 E1")
    r = ev[ev["regime"].notna()]
    print(fmt_table({k: summarize(v, "net_on") for k, v in by(r, r["regime"]).items()}).iloc[:, :8].to_string())
    print("\nP1 상위 종목:", ev["name"].value_counts().head(8).to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["high", "ma", "strength", "e1"])
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    globals()[f"cmd_{a.cmd}"]()


if __name__ == "__main__":
    main()
