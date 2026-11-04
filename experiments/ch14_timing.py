"""14장 실험 — 매수 시점 T-1·T0·T1~T3 비교와 매수하지 않는 선택 T4 (1분봉 기간).

사용법 (먼저 build1518, ch11_exits build, ch12_minute run)
  python -m experiments.ch14_timing timing     # 14-1·14-2·14-3 같은 후보의 진입 시점별 성과·유동성
  python -m experiments.ch14_timing interact   # 14-4 진입 시점 × 청산 정책
  python -m experiments.ch14_timing t4         # 14-5 매수하지 않는 날의 가치

장후 거래(T1~T3)는 KRX 장후 거래가 열린 2026-09-14 이후만 있다. 진입 후보는 15:18에 정하고 바꾸지 않는다.
NXT 시세는 이 실험에 없다(1분봉은 KRX 코드로만 수집).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from backtest.costs import net_return
from backtest.metrics import fmt_table, summarize
from data.dataset import AFTERHOURS_START
from data.store import path
from experiments.ch08_conditions import load1518
from experiments.ch09_entry import pools
from strategies.exit_policies.s0_s1 import tick_size

pd.set_option("display.width", 200)


def entries(x: pd.DataFrame) -> dict[str, pd.Series]:
    """시점별 진입가. 접속매매(T-1·T1~T3)는 반 호가 불리, 단일가(T0)는 그대로."""
    up = lambda p: p + 0.5 * tick_size(p)  # noqa: E731
    return {"T-1 15:18~15:19": up(x["p_t1"]), "T0 종가 단일가": x["p_t0"], "T1 장후 16:00~16:29": up(x["p_ah_t1"]),
            "T2 장후 17:30~17:59": up(x["p_ah_t2"]), "T3 장후 19:30~19:59": up(x["p_ah_t3"])}


def cmd_timing() -> None:
    x = pools(load1518())["pool"]
    x = x[(x["dt"] >= AFTERHOURS_START)].dropna(subset=["p_t1", "p_t0", "p_ah_t1", "p_ah_t2", "p_ah_t3", "nxt_open"])
    print(f"장후 거래 기간 15:18 후보 {len(x):,}건, {x['dt'].min()}~{x['dt'].max()}, {x['dt'].nunique()}거래일")
    rows, nets = {}, {}
    for k, p in entries(x).items():
        nets[k] = net_return(x["nxt_open"] / p - 1, x["nxt_dt"])
        rows[k] = summarize(x.assign(r=nets[k]), "r")
    print(fmt_table(rows).iloc[:, :8].to_string())
    base = nets["T-1 15:18~15:19"]
    for k, r in nets.items():
        d = r - base
        if k.startswith("T-1"):
            continue
        print(f"  {k} − T-1: {d.mean() * 100:+.3f}%p (t={d.mean() / d.std() * np.sqrt(d.count()):+.2f})")
    ah = x["last_ah"] / x["p_t0"] - 1
    print(f"\n정규장 종가 → 장후 마지막 가격 변화: 평균 {ah.mean() * 100:+.3f}%, 중앙값 {ah.median() * 100:+.3f}%")
    print("\n[유동성] 구간별 거래대금 중앙값 (억 원)")
    liq = {"15:18~15:19 (2분)": x["value_t1"], "종가 단일가": x["vol_auction"] * x["p_t0"],
           "장후 16:00~16:29": x["value_ah_t1"], "장후 17:30~17:59": x["value_ah_t2"],
           "장후 19:30~19:59": x["value_ah_t3"], "장후 전체": x["value_ah"]}
    print({k: round(v.median() / 1e8, 2) for k, v in liq.items()})
    zero = {k: f"{(v <= 0).mean() * 100:.1f}%" for k, v in liq.items() if k.startswith("장후")}
    print("거래가 전혀 없었던 비율:", zero)
    # 14-2: 장후 가격을 보고 고르기 — 장후 초반 30분에 정규장 종가보다 오른 종목만 T3에 산다
    rise = x["p_ah_t1"] > x["p_t0"]
    print("\n장후 초반(16:00~16:29) 가격이 정규장 종가보다 높았던 종목만 T3 진입 vs 전체 T3")
    t3 = nets["T3 장후 19:30~19:59"]
    print(fmt_table({"장후 초반 상승 종목": summarize(x[rise].assign(r=t3[rise]), "r"),
                     "장후 초반 하락·보합 종목": summarize(x[~rise].assign(r=t3[~rise]), "r")}).iloc[:, :8].to_string())


def cmd_interact() -> None:
    e = pd.read_parquet(path("research", "exits.parquet"))
    x = pools(load1518())["pool"].loc[:, ["symbol", "dt", "p_t0", "p_t1"]]
    e = e.merge(x, on=["symbol", "dt"], suffixes=("", "_x"))
    ent = {"T-1": e["entry"], "T0": e["p_t0"]}
    rows = {}
    for en, p in ent.items():
        for s in ("S1", "S0", "S5_0910", "S6"):
            r = net_return(e[s] / p - 1, e["nxt_dt"])
            rows[f"{en} × {s}"] = summarize(e.assign(r=r), "r")
    print(f"같은 후보 {len(e):,}건 (1분봉 기간 전체)")
    print(fmt_table(rows).iloc[:, :6].to_string())


def cmd_t4() -> None:
    p = pd.read_parquet(path("research", "ch12_minute_preds.parquet"))
    top = p[p.groupby("dt")["pred"].rank(ascending=False, method="first") <= 5]
    day = top.groupby("dt").agg(pred=("pred", "mean"), risk=("risk", "mean"), net=("net", "mean"), n=("net", "size"))
    print(f"모델 상위 5를 매일 산다고 할 때 {len(day)}거래일 (월별 Walk-Forward 예측)")
    print(f"  매일 매수: 일평균 {day['net'].mean() * 100:.3f}%, 누적 {((1 + day['net']).prod() - 1) * 100:.1f}%")
    for name, keep in [("예측 평균 > 0.2% 인 날만", day["pred"] > 0.002),
                       ("예측 평균 > 0 인 날만", day["pred"] > 0),
                       ("갭하락 위험이 그동안의 중앙값 이하인 날만", day["risk"] <= day["risk"].expanding().median().shift(1)),
                       ]:
        d = day["net"].where(keep, 0.0)
        print(f"  {name}: 매수일 {int(keep.sum())}일, 일평균 {d.mean() * 100:.3f}%, "
              f"누적 {((1 + d).prod() - 1) * 100:.1f}%, 쉰 날의 실제 평균 {day.loc[~keep, 'net'].mean() * 100:+.3f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["timing", "interact", "t4"])
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    globals()[f"cmd_{a.cmd}"]()


if __name__ == "__main__":
    main()
