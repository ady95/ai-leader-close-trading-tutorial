"""11장 실험 — 다음 날 청산 정책 S1~S6 비교 (1분봉 기간, 같은 진입 포지션).

사용법 (먼저 python -m experiments.ch08_conditions build1518)
  python -m experiments.ch11_exits build    # 진입 포지션별 정책 매도가 계산 → research/exits.parquet
  python -m experiments.ch11_exits compare  # 11-1~11-5 정책 비교
  python -m experiments.ch11_exits pattern  # 11-6 P1·P2별 (분봉 실험 모드)
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from backtest.costs import net_return
from backtest.metrics import fmt_table, summarize
from data.dataset import regular_session_daily
from data.store import path
from experiments.ch08_conditions import load1518
from experiments.ch09_entry import pools, quarter
from strategies.exit_policies import intraday_exits as X
from strategies.exit_policies.s0_s1 import entry_price, exit_prices

pd.set_option("display.width", 200)
OUT = path("research", "exits.parquet")
POL = ["S1", "S0", "S2", "S3", "S4", "S5_0910", "S5_0930", "S5_1000", "S6"]


def _atr14(symbol: str) -> pd.Series:
    d = regular_session_daily(symbol).sort_values("dt")
    if d.empty:
        return pd.Series(dtype=float)
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    return pd.Series(tr.rolling(14, min_periods=10).mean().to_numpy(), index=d["dt"])


def _symbol_job(args) -> list[dict]:
    symbol, rows = args
    m = pd.read_parquet(path("minute", f"{symbol}.parquet"), columns=["ts", "open", "high", "low", "close", "volume"])
    m["dt"], m["hm"] = m["ts"].str[:8], m["ts"].str[8:12]
    m = m[(m["hm"] >= "0900") & (m["hm"] <= "1519")].sort_values("ts")
    days = {d: g for d, g in m.groupby("dt")}
    atr = _atr14(symbol)
    out = []
    for r in rows:
        g = days.get(r["nxt_dt"])
        if g is None:
            continue
        res = X.simulate(r["entry"], float(atr.get(r["dt"], np.nan)), g["hm"].to_numpy(),
                         *(g[c].to_numpy(float) for c in ("open", "high", "low", "close", "volume")))
        if res:
            out.append({"symbol": symbol, "dt": r["dt"], **res})
    return out


def cmd_build() -> None:
    x = pools(load1518())["pool"].dropna(subset=["p_t1", "nxt_open"]).copy()
    x["entry"] = entry_price(x)
    jobs = [(s, g.loc[:, ["dt", "nxt_dt", "entry"]].to_dict("records")) for s, g in x.groupby("symbol")]
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = [r for part in ex.map(_symbol_job, jobs, chunksize=4) for r in part]
    e = pd.DataFrame(res)
    base = x.loc[:, ["symbol", "dt", "nxt_dt", "entry", "nx_vwap5", "nxt_open", "p_t1", "rank_1518", "ret_1518"]]
    s0 = pd.concat([x.loc[:, ["symbol", "dt"]], exit_prices(x).loc[:, ["S0", "gap_type"]]], axis=1)
    e = base.merge(e, on=["symbol", "dt"]).merge(s0, on=["symbol", "dt"])
    e.to_parquet(OUT, index=False)
    print(f"{len(e):,}건 → {OUT}")


def load() -> pd.DataFrame:
    e = pd.read_parquet(OUT)
    for p in POL:
        e[p + "_net"] = net_return(e[p] / e["entry"] - 1, e["nxt_dt"])
    return e


def table(e: pd.DataFrame) -> str:
    return fmt_table({p: summarize(e.assign(r=e[p + "_net"]), "r") for p in POL if p in e}).iloc[:, :8].to_string()


def cmd_compare() -> None:
    e = load()
    print(f"같은 진입 포지션 {len(e):,}건 (15:18 거래대금 상위 100·상승·상한가 아님), {e['dt'].min()}~{e['dt'].max()}")
    print(table(e))
    print("\nS1 대비 차이 (%p, t)")
    for p in POL[1:]:
        d = e[p + "_net"] - e["S1_net"]
        print(f"  {p:8s} {d.mean() * 100:+.3f} (t={d.mean() / d.std() * np.sqrt(d.count()):+.2f}), 나은 비율 {(d > 0).mean() * 100:.1f}%")
    print("\n분기별 평균 (%)")
    q = quarter(e["dt"])
    print(pd.DataFrame({p: e.groupby(q)[p + "_net"].mean() * 100 for p in POL}).round(3).to_string())
    print("\n갭 유형별 평균 (%)")
    print(pd.DataFrame({p: e.groupby("gap_type")[p + "_net"].mean() * 100 for p in POL}).round(3).to_string())


def cmd_pattern() -> None:
    from experiments.ch10_exit import daily_patterns_minute_period
    e = load().merge(daily_patterns_minute_period(), on=["symbol", "dt"], how="inner")
    for k, g in e.groupby("pattern_type"):
        print(f"\n[{k}] {len(g):,}건")
        print(table(g))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "compare", "pattern"])
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    globals()[f"cmd_{a.cmd}"]()


if __name__ == "__main__":
    main()
