"""10장 실험 — 다음 날 09:05 청산 S0 vs 시가 전량 S1 (1분봉 기간).

사용법 (먼저 python -m experiments.ch08_conditions build1518)
  python -m experiments.ch10_exit gap      # 10-1 갭 유형별 다음 날 아침 경로
  python -m experiments.ch10_exit s0s1     # 10-2·10-3 15:18 후보에 S0·S1 적용
  python -m experiments.ch10_exit e3e4     # 10-5 E3(P1·P2 + S0)·E4(같은 포지션 + S1)

E3·E4는 P1·P2 판정에 일봉이 필요하다. 1분봉 기간(2025-11~)은 일봉 홀드아웃과 겹치므로
'분봉 실험 모드'로 일봉을 읽는다(15-1). E0~E2 규칙은 이미 고정돼 있어 이 결과로 바꾸지 않는다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from backtest.costs import net_return
from backtest.metrics import fmt_table, summarize
from experiments.ch05_leaders import by
from experiments.ch08_conditions import load1518
from experiments.ch09_entry import pools, quarter
from strategies.exit_policies.s0_s1 import exit_prices

pd.set_option("display.width", 200)
POLICIES = ["S1", "S0", "S0b"]


def stats(groups: dict[str, pd.DataFrame], col: str) -> str:
    return fmt_table({k: summarize(v, col) for k, v in groups.items()}).iloc[:, :8].to_string()


def with_exits(x: pd.DataFrame) -> pd.DataFrame:
    x = x.dropna(subset=["nx_vwap5", "nxt_open", "p_t1"]).copy()
    x = x.join(exit_prices(x))
    for p in POLICIES:
        x[p + "_net"] = net_return(x[p] / x["entry"] - 1, x["nxt_dt"])
    return x


def compare(x: pd.DataFrame, title: str) -> None:
    print(f"\n[{title}] {len(x):,}건, {x['dt'].min()}~{x['dt'].max()}")
    print(stats({p: x.assign(r=x[p + "_net"]) for p in POLICIES}, "r"))
    for p in ("S0", "S0b"):
        d = x[p + "_net"] - x["S1_net"]
        print(f"  {p} − S1: {d.mean() * 100:+.3f}%p (t={d.mean() / d.std() * np.sqrt(d.count()):+.2f}), "
              f"{p}가 나은 비율 {(d > 0).mean() * 100:.1f}%")
    d = x["S0_net"] - x["S1_net"]
    print("  분기별 S0 − S1 (%p):", (d.groupby(quarter(x["dt"])).mean() * 100).round(3).to_dict())
    print("  갭 유형별 S0 − S1 (%p):", (d.groupby(x["gap_type"]).mean() * 100).round(3).to_dict())


def cmd_gap() -> None:
    x = with_exits(pools(load1518())["pool"])
    print(f"15:18 후보 {len(x):,}건, 갭 유형 비율:", (x["gap_type"].value_counts(normalize=True) * 100).round(1).to_dict())
    o = x["nxt_open"]
    path = pd.DataFrame({
        "시가→09:00~04 VWAP": x["nx_vwap5"] / o - 1, "시가→09:04": x["nx_c0904"] / o - 1,
        "시가→09:09": x["nx_c0909"] / o - 1, "시가→09:29": x["nx_c0929"] / o - 1, "시가→09:59": x["nx_c0959"] / o - 1})
    print("\n갭 유형별 시가 이후 평균 수익률 (%, 비용 전)")
    print((path.groupby(x["gap_type"]).mean() * 100).round(3).to_string())
    print("\n갭 유형별 시가 이후 중앙값 (%, 비용 전)")
    print((path.groupby(x["gap_type"]).median() * 100).round(3).to_string())
    g = x["nxt_open"] / x["entry"] - 1
    print(f"\n갭(진입가 대비 시가) 평균 {g.mean() * 100:.3f}%, 중앙값 {g.median() * 100:.3f}%")


def cmd_s0s1() -> None:
    p = pools(load1518())
    for k in ("pool", "e0_1518"):
        x = with_exits(p[k])
        compare(x, k)
    x = with_exits(p["pool"])
    for amt in (1e7, 1e8, 1e9):
        part = amt / (x["nx_vol5"] * x["nxt_open"])
        print(f"포지션 {amt / 1e8:g}억 원: 다음 날 09:00~09:04 거래대금 대비 참여율 중앙값 {part.median() * 100:.2f}%, "
              f"10% 초과 {(part > 0.1).mean() * 100:.1f}%")


def daily_patterns_minute_period() -> pd.DataFrame:
    """분봉 실험 모드: 1분봉 기간의 P1·P2 판정. 일봉 홀드아웃을 읽으므로 명시적으로 연다."""
    os.environ["RESEARCH_UNSEAL_HOLDOUT"] = "1"
    from data import dataset as ds
    from strategies.entry_patterns import p1, p2
    from strategies.leader_ranker import features as F
    d = ds.load_daily("all")
    d = d[d["dt"] >= "20241101"]          # 252일 고점 계산에 필요한 1년 + 1분봉 기간
    d = F.build(d)
    d = p2.add_features(p1.add_features(d))
    d = d[d["tradable"] & (d["ret"].abs() <= 0.305) & (d["dt"] >= "20251103")]
    d["pattern_type"] = np.select([p1.is_p1(d), p2.is_p2(d)], ["P1", "P2"], "")
    return d.loc[d["pattern_type"] != "", ["symbol", "dt", "pattern_type"]]


def cmd_e3e4() -> None:
    pat = daily_patterns_minute_period()
    m = load1518().merge(pat, on=["symbol", "dt"], how="inner")
    x = with_exits(m)
    print(f"1분봉이 있는 P1·P2 이벤트 {len(x):,}건 (P1 {(x['pattern_type'] == 'P1').sum()}, "
          f"P2 {(x['pattern_type'] == 'P2').sum()})")
    t = {"E3 = P1·P2 + S0": x.assign(r=x["S0_net"]), "E4 = 같은 포지션 + S1": x.assign(r=x["S1_net"])}
    print(fmt_table({k: summarize(v, "r") for k, v in t.items()}).to_string())
    compare(x, "P1·P2")
    for k, g in by(x, x["pattern_type"]).items():
        d = g["S0_net"] - g["S1_net"]
        print(f"  {k}: S0 {g['S0_net'].mean() * 100:.3f}%, S1 {g['S1_net'].mean() * 100:.3f}%, "
              f"차이 {d.mean() * 100:+.3f}%p (n={len(g)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gap", "s0s1", "e3e4"])
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    globals()[f"cmd_{a.cmd}"]()


if __name__ == "__main__":
    main()
