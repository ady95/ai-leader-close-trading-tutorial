"""7장 실험 — P2 상승추세 눌림목 재상승형 (일봉, 개발 기간 2023-01~2025-12).

사용법 (먼저 python -m experiments.ch05_leaders build)
  python -m experiments.ch07_p2 counter    # 7-1 P2 vs 역추세 반등 vs 기타 재상승
  python -m experiments.ch07_p2 support    # 7-2 5일선 거리·저점 상승·조정 일수
  python -m experiments.ch07_p2 depth      # 7-3 눌림 깊이·재상승 강도
  python -m experiments.ch07_p2 e2         # 7-4 E2 (P2 → S1)
  python -m experiments.ch07_p2 compare    # 7-5 P1·P2 비교와 통합
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from backtest.metrics import daily_portfolio, fmt_table, max_drawdown, summarize
from experiments.ch05_leaders import by, e0_select
from experiments.ch06_p1 import horizons
from experiments.ch06_p1 import load as load_p1
from strategies.entry_patterns import p1, p2

pd.set_option("display.width", 200)


def load() -> pd.DataFrame:
    return load_p1(extra=p2.add_features)


def stats(groups: dict[str, pd.DataFrame]) -> str:
    return fmt_table({k: summarize(v, "net_on") for k, v in groups.items()}).iloc[:, :8].to_string()


def cmd_counter() -> None:
    df = load()
    cand = df[p1.candidates(df) & df["reclaim"]]
    ev2, ct = p2.is_p2(df), p2.is_countertrend(df)
    groups = {"P2 눌림목 재상승": df[ev2], "역추세 반등 (제외 대상)": df[ct],
              "기타 재상승 (전일 고가 돌파)": cand[~ev2.loc[cand.index] & ~ct.loc[cand.index]]}
    print("후보 = 거래대금 상위 100, 상승, 상한가 제외, 종가 > 전일 고가")
    print(horizons(groups))
    print(f"\nP2 {ev2.sum():,}건 / 역추세 {ct.sum():,}건 / P1과 겹침 {(ev2 & p1.is_p1(df)).sum()}건")


def cmd_support() -> None:
    df = load()
    ev = df[p2.is_p2(df)]
    print(f"P2 {len(ev):,}건")
    print("\n눌림 저점의 5일선 대비 위치 (직전 5일 최저가 ÷ 어제 MA5 - 1)")
    print(horizons(by(ev, pd.cut(ev["low_vs_ma5"], [-1, -0.08, -0.05, -0.02, 1],
                                 labels=["-8% 미만", "-8~-5%", "-5~-2%", "-2% 이상"]))))
    print("\n저점 상승 (어제 저점 > 직전 5일 최저가)")
    print(horizons(by(ev, np.where(ev["higher_low"], "저점 상승", "어제가 최저점"))))
    print("\n조정 일수 (최근 20일 고점이 며칠 전)")
    print(horizons(by(ev, pd.cut(ev["days_since_high"], [1, 3, 5, 7, 10], labels=["2~3일", "4~5일", "6~7일", "8~10일"]))))


def cmd_depth() -> None:
    df = load()
    ev = df[p2.is_p2(df)]
    print("눌림 깊이 (최근 20일 고점 대비 직전 5일 최저가 하락폭)")
    print(horizons(by(ev, pd.cut(ev["pullback_depth"], [0.03, 0.06, 0.09, 0.12, 0.15],
                                 labels=["3~6%", "6~9%", "9~12%", "12~15%"], include_lowest=True))))
    print("\n재상승일 등락률")
    print(horizons(by(ev, pd.cut(ev["ret"], [0, 0.03, 0.07, 0.15, 0.3], labels=["0~3%", "3~7%", "7~15%", "15%~"]))))
    print("\n재상승일 거래대금 ÷ 20일 평균")
    print(horizons(by(ev, pd.cut(ev["tv_ratio20"], [0, 1, 1.5, 2.5, 1e9], labels=["~1", "1~1.5", "1.5~2.5", "2.5~"]))))
    print("\n재상승일 종가 위치 CLV")
    print(horizons(by(ev, pd.cut(ev["clv"], [-0.01, 0.5, 0.7, 0.9, 1.01], labels=["~0.5", "0.5~0.7", "0.7~0.9", "0.9~1"]))))


def cmd_e2() -> None:
    df = load()
    ev = df[p2.is_p2(df)]
    e0 = e0_select(df[df["nxt_tradable"] & df["r_on"].notna()])
    e1 = df[p1.is_p1(df)]
    print(f"P2 이벤트 {len(ev):,}건, {ev['dt'].nunique()}거래일, {ev['symbol'].nunique()}종목, "
          f"하루 평균 {len(ev) / ev['dt'].nunique():.1f}건, E0와 겹침 {ev.index.isin(e0.index).mean() * 100:.1f}%")
    print(fmt_table({k: summarize(v, "net_on") for k, v in
                     {"E2 = P2 → S1": ev, "E1 = P1 → S1": e1, "E0": e0}.items()}).to_string())
    print("\n보유 기간별")
    print(horizons({"P2": ev, "P1": e1, "E0": e0}))
    print("\n연도별 E2")
    print(stats(by(ev, ev["dt"].str[:4])))
    r = ev[ev["regime"].notna()]
    print("\n시장 국면별 E2")
    print(stats(by(r, r["regime"])))
    print("\nP2 상위 종목:", ev["name"].value_counts().head(8).to_dict())


def cmd_compare() -> None:
    df = load()
    e1, e2 = df[p1.is_p1(df)].copy(), df[p2.is_p2(df)].copy()
    e1["pattern_type"], e2["pattern_type"] = "P1", "P2"
    both = pd.concat([e1, e2])
    print("P1·P2 통합 (pattern_type 태그)")
    print(stats({"P1": e1, "P2": e2, "P1+P2 통합": both}))
    d1, d2 = daily_portfolio(e1.dropna(subset=["net_on"]), "net_on"), daily_portfolio(e2.dropna(subset=["net_on"]), "net_on")
    j = pd.concat([d1.rename("P1"), d2.rename("P2")], axis=1)
    print(f"\n신호일: P1 {d1.size}일, P2 {d2.size}일, 같은 날 {j.dropna().shape[0]}일, "
          f"같은 날 일별 수익 상관 {j.dropna().corr().iloc[0, 1]:.2f}")
    print(f"일별 동일비중 MDD: P1 {max_drawdown(d1) * 100:.2f}%, P2 {max_drawdown(d2) * 100:.2f}%, "
          f"통합 {max_drawdown(daily_portfolio(both.dropna(subset=['net_on']), 'net_on')) * 100:.2f}%")
    for name, key in [("연도", both["dt"].str[:4]), ("국면", both["regime"].fillna("국면 없음"))]:
        g = both.groupby([key, "pattern_type"])["net_on"]
        t = pd.concat({"건수": g.count().unstack(), "평균(%)": (g.mean() * 100).round(2).unstack()}, axis=1)
        print(f"\n{name}별 P1·P2 (비용 후)\n{t.to_string()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["counter", "support", "depth", "e2", "compare"])
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    globals()[f"cmd_{a.cmd}"]()


if __name__ == "__main__":
    main()
