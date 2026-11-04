"""13장 실험 — LightGBM 조건부 매도 S7 (1분봉 기간, 월별 Walk-Forward).

사용법 (먼저 python -m experiments.ch08_conditions build1518)
  python -m experiments.ch13_exit_ai run

판단 시점은 다음 날 09:01:00 (09:00 봉 완성 직후). 시가 단일가 체결 여부는 시가를 보기 전에 정해야 하므로,
S7은 '09:01에 지금 팔지, 09:10·09:30까지 더 들고 갈지'를 고른다. 비교 기준은 S1(시가 전량).
라벨(미래 값): y10 = (09:09 봉 종가 - 09:00 봉 종가) / 진입가, y30 = (09:29 봉 종가 - 09:00 봉 종가) / 진입가.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from backtest.costs import net_return
from backtest.metrics import fmt_table, summarize
from data.store import path
from experiments.ch09_entry import quarter
from experiments.ch12_minute import FEATURES as F1518
from experiments.ch12_minute import frame, monthly_folds
from models.lightgbm import walkforward as W
from strategies.exit_policies.s0_s1 import tick_size

pd.set_option("display.width", 200)
EXIT_FEATURES = ["gap", "bar1_ret", "bar1_range", "bar1_value_ratio"] + F1518
CATEGORICAL = ["market"]


def sell(p: pd.Series) -> pd.Series:
    return p - 0.5 * tick_size(p)


def cmd_run() -> None:
    x = frame().dropna(subset=["nx_c0900", "nx_c0909", "nx_c0929"]).copy()
    o = x["nxt_open"]
    x["gap"] = o / x["entry"] - 1
    x["bar1_ret"] = x["nx_c0900"] / o - 1
    x["bar1_range"] = (x["nx_h0900"] - x["nx_l0900"]) / o
    x["bar1_value_ratio"] = x["nx_v0900"] * o / x["value_1518"]
    x["y10"] = (x["nx_c0909"] - x["nx_c0900"]) / x["entry"]
    x["y30"] = (x["nx_c0929"] - x["nx_c0900"]) / x["entry"]
    px = {"S1 시가 전량": o, "09:01 전량": sell(x["nx_c0900"]), "09:10 전량 (S5)": sell(x["nx_c0909"]),
          "09:30 전량 (S5)": sell(x["nx_c0929"])}
    fl = monthly_folds(x["dt"])
    x["p10"], imp = W.walk_forward(x, EXIT_FEATURES, "y10", fl, CATEGORICAL, clip=0.10)
    x["p30"], _ = W.walk_forward(x, EXIT_FEATURES, "y30", fl, CATEGORICAL, clip=0.10)
    t = x[x["p10"].notna()].copy()
    half = 0.5 * tick_size(t["nx_c0900"]) / t["entry"]           # 09:01에 팔 때 이미 내는 반 호가와 같은 크기
    hold10 = t["p10"] > half
    hold30 = t["p30"] > half
    px_t = {k: v.loc[t.index] for k, v in px.items()}
    px_t["S7 (09:10 보유 예측 시 보유)"] = np.where(hold10, sell(t["nx_c0909"]), sell(t["nx_c0900"]))
    px_t["S7 (09:30 보유 예측 시 보유)"] = np.where(hold30, sell(t["nx_c0929"]), sell(t["nx_c0900"]))
    # 13-3 즉시·부분·보유: 예측이 +반호가 초과 → 전량 09:10, -반호가 미만 → 전량 09:01, 그 사이 → 절반씩
    part = np.select([t["p10"] > half, t["p10"] < -half], [sell(t["nx_c0909"]), sell(t["nx_c0900"])],
                     (sell(t["nx_c0909"]) + sell(t["nx_c0900"])) / 2)
    px_t["S7 즉시·부분·보유"] = part
    rows = {}
    for k, p in px_t.items():
        g = t.assign(r=net_return(pd.Series(p, index=t.index) / t["entry"] - 1, t["nxt_dt"]))
        rows[k] = summarize(g, "r")
        t[k] = g["r"]
    print(f"같은 진입 포지션 {len(t):,}건 (15:18 후보, T-1 진입), {t['dt'].min()}~{t['dt'].max()}, 월별 폴드 {len(fl)}개")
    print(fmt_table(rows).iloc[:, :8].to_string())
    print(f"\n09:10까지 보유를 고른 비율 {hold10.mean() * 100:.1f}%, 09:30 보유 {hold30.mean() * 100:.1f}%")
    for k in ["09:01 전량", "S7 (09:10 보유 예측 시 보유)", "S7 (09:30 보유 예측 시 보유)", "S7 즉시·부분·보유"]:
        d = t[k] - t["S1 시가 전량"]
        print(f"  {k} − S1: {d.mean() * 100:+.3f}%p (t={d.mean() / d.std() * np.sqrt(d.count()):+.2f})")
    d = t["S7 (09:10 보유 예측 시 보유)"] - t["09:01 전량"]
    print(f"  S7(09:10) − 09:01 전량: {d.mean() * 100:+.3f}%p (t={d.mean() / d.std() * np.sqrt(d.count()):+.2f})")
    ic = t.groupby("dt").apply(lambda g: g["p10"].rank().corr(g["y10"].rank()) if len(g) > 2 else np.nan)
    print(f"\ny10 예측 일별 순위상관 {ic.mean():.3f} (t={ic.mean() / ic.std() * np.sqrt(ic.count()):.2f})")
    print("분기별 S7(09:10) − S1 (%p):",
          ((t["S7 (09:10 보유 예측 시 보유)"] - t["S1 시가 전량"]).groupby(quarter(t["dt"])).mean() * 100).round(3).to_dict())
    print("\n특징 중요도 (gain %, 상위 8)")
    print((imp.mean(axis=1) / imp.mean(axis=1).sum() * 100).sort_values(ascending=False).head(8).round(1).to_string())
    t.to_parquet(path("research", "ch13_s7.parquet"), index=False)


def cmd_breakdown() -> None:
    """13-5: P1·P2·그 밖, 소속 지수 국면별 S1·09:01·S7 비교 (run 결과 파일 사용)."""
    from experiments.ch10_exit import daily_patterns_minute_period
    from strategies.leader_ranker import features as Fz
    t = pd.read_parquet(path("research", "ch13_s7.parquet"))
    t = t.merge(daily_patterns_minute_period(), on=["symbol", "dt"], how="left")
    t["pattern_type"] = t["pattern_type"].fillna("그 밖")
    reg = pd.concat([Fz.market_regime("001").assign(market="KOSPI"), Fz.market_regime("101").assign(market="KOSDAQ")])
    t = t.merge(reg.loc[:, ["dt", "market", "regime"]], on=["dt", "market"], how="left")
    cols = ["S1 시가 전량", "09:01 전량", "S7 (09:10 보유 예측 시 보유)"]
    for key in ("pattern_type", "regime"):
        g = t.groupby(key)
        out = pd.concat({c: g[c].mean() * 100 for c in cols}, axis=1).round(3)
        out["건수"] = g.size()
        out["S7−S1"] = (out[cols[2]] - out[cols[0]]).round(3)
        print(f"\n{key}별 평균 (%, 비용 후)\n{out.to_string()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "breakdown"])
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    globals()[f"cmd_{a.cmd}"]()


if __name__ == "__main__":
    main()
