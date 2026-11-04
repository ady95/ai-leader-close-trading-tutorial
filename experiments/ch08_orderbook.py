"""8-5·9-5 실측 — 수집한 실시간 호가로 특정 시간대의 스프레드·잔량 측정.

사용법
  python -m experiments.ch08_orderbook spread --day 20261104 --start 151800 --end 151959
  python -m experiments.ch08_orderbook spread --day 20261105 --start 090000 --end 090459

0D(주식호가잔량) 필드: 21 호가시간, 41 매도1호가, 51 매수1호가, 61 매도1잔량, 71 매수1잔량,
121 총매도잔량, 125 총매수잔량. 수집 중인 파일도 읽을 수 있다(마지막 미완성 블록은 건너뜀).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import zlib

import numpy as np
import pandas as pd

from data.store import path
from strategies.exit_policies.s0_s1 import tick_size


def _num(x) -> float:
    try:
        return abs(float(str(x).replace("+", "").replace("-", "")))
    except ValueError:
        return np.nan


def read_orderbook(day: str, start: str, end: str) -> pd.DataFrame:
    rows = []
    for p in sorted(path("realtime", day).glob("orderbook*.jsonl.gz")):   # 재시작하면 파일이 나뉜다
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    v = r.get("values") or {}
                    hm = str(v.get("21", ""))
                    if not (start <= hm <= end):
                        continue
                    rows.append({"symbol": r["item"], "t": hm, "ask": _num(v.get("41")), "bid": _num(v.get("51")),
                                 "ask_q": _num(v.get("61")), "bid_q": _num(v.get("71")),
                                 "tot_ask": _num(v.get("121")), "tot_bid": _num(v.get("125"))})
        except (EOFError, zlib.error, OSError):
            pass                                       # 강제 종료·수집 중인 파일의 끝
    return pd.DataFrame(rows)


def cmd_spread(day: str, start: str, end: str) -> None:
    ob = read_orderbook(day, start, end)
    ob = ob[(ob["ask"] > 0) & (ob["bid"] > 0) & (ob["ask"] > ob["bid"])]
    if ob.empty:
        print("해당 구간 호가 없음")
        return
    ob["mid"] = (ob["ask"] + ob["bid"]) / 2
    ob["spread_pct"] = (ob["ask"] - ob["bid"]) / ob["mid"]
    ob["spread_ticks"] = (ob["ask"] - ob["bid"]) / tick_size(ob["bid"])
    ob["best_ask_value"] = ob["ask"] * ob["ask_q"]
    per = ob.groupby("symbol").agg(n=("t", "size"), spread_pct=("spread_pct", "median"),
                                   spread_ticks=("spread_ticks", "median"), ask_value=("best_ask_value", "median"),
                                   imbalance=("tot_bid", "median"))
    per["imbalance"] = (ob.groupby("symbol")["tot_bid"].median()
                        / (ob.groupby("symbol")["tot_bid"].median() + ob.groupby("symbol")["tot_ask"].median()))
    print(f"{day} {start}~{end}: 호가 메시지 {len(ob):,}건, 종목 {len(per)}개")
    print(f"종목별 스프레드 중앙값의 분포: 중앙값 {per['spread_pct'].median() * 100:.3f}%, "
          f"평균 {per['spread_pct'].mean() * 100:.3f}%, 상위 10% {per['spread_pct'].quantile(0.9) * 100:.3f}%")
    print(f"스프레드 = 1호가인 종목 비율 {(per['spread_ticks'] <= 1.0).mean() * 100:.1f}%, "
          f"2호가 이상 {(per['spread_ticks'] >= 2).mean() * 100:.1f}%")
    print(f"→ 반 스프레드(접속매매 편도 비용) 중앙값 {per['spread_pct'].median() / 2 * 100:.3f}%")
    print(f"매도1호가 잔량 금액 중앙값 {per['ask_value'].median() / 1e6:.1f}백만 원, "
          f"50만 원 주문이 매도1호가 잔량보다 큰 종목 {(per['ask_value'] < 5e5).mean() * 100:.1f}%")
    print(f"총매수잔량 비중 중앙값 {per['imbalance'].median() * 100:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["spread"])
    ap.add_argument("--day", required=True)
    ap.add_argument("--start", default="151800")
    ap.add_argument("--end", default="151959")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cmd_spread(a.day, a.start, a.end)


if __name__ == "__main__":
    main()
