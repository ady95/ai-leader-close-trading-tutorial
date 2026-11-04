"""8-5·9-5 실측 — 수집한 실시간 호가로 특정 시간대의 스프레드·잔량 측정.

사용법
  python -m experiments.ch08_orderbook spread --day 20261104 --start 151800 --end 151959
  python -m experiments.ch08_orderbook spread --day 20261105 --start 090000 --end 090459
  python -m experiments.ch08_orderbook flow --day 20261104      # 15:17:59 잔량·체결강도 vs 15:18→15:30 수익

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


def read_trades(day: str, start: str, end: str) -> pd.DataFrame:
    """0B(주식체결): 20 체결시간, 10 현재가, 228 체결강도."""
    rows = []
    for p in sorted(path("realtime", day).glob("trades*.jsonl.gz")):
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    v = r.get("values") or {}
                    hm = str(v.get("20", ""))
                    if not (start <= hm <= end):
                        continue
                    rows.append({"symbol": r["item"], "t": hm, "rt": r.get("rt", ""),
                                 "price": _num(v.get("10")), "strength": _num(v.get("228"))})
        except (EOFError, zlib.error, OSError):
            pass
    return pd.DataFrame(rows)


def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    return float(np.corrcoef(a.rank(), b.rank())[0, 1])


def cmd_flow(day: str) -> None:
    """15:17:59 시점의 호가잔량·체결강도가 15:18 → 15:30 종가 단일가 수익을 설명하는가 (하루 단면)."""
    ob = read_orderbook(day, "151500", "151759")
    ob = ob[(ob["ask"] > 0) & (ob["bid"] > 0)]
    last_ob = ob.sort_values("t").groupby("symbol").tail(1).set_index("symbol")
    tr = read_trades(day, "151500", "153059").sort_values(["t", "rt"])
    before = tr[tr["t"] <= "151759"].groupby("symbol").tail(1).set_index("symbol")
    close = tr[tr["t"] >= "153000"].groupby("symbol").head(1).set_index("symbol")["price"]
    f = pd.DataFrame({
        "book_bid_share": last_ob["tot_bid"] / (last_ob["tot_bid"] + last_ob["tot_ask"]),
        "top_bid_share": last_ob["bid_q"] / (last_ob["bid_q"] + last_ob["ask_q"]),
        "strength": before["strength"],
        "ret": close / before["price"] - 1,
    }).dropna()
    print(f"{day}: 15:17:59 호가·체결과 15:30 종가가 모두 있는 종목 {len(f)}개")
    print(f"15:18 → 15:30 수익: 평균 {f['ret'].mean() * 100:+.3f}%, 중앙값 {f['ret'].median() * 100:+.3f}%, "
          f"상승 {(f['ret'] > 0).mean() * 100:.1f}% · 하락 {(f['ret'] < 0).mean() * 100:.1f}%")
    for col, name in [("book_bid_share", "총매수잔량 비중"), ("top_bid_share", "매수1호가 잔량 비중"),
                      ("strength", "체결강도")]:
        q = pd.qcut(f[col].rank(method="first"), 3, labels=["하", "중", "상"])
        g = f.groupby(q, observed=True)["ret"].mean() * 100
        print(f"{name}: 분포 중앙값 {f[col].median():.3f}, 순위상관 {_rank_corr(f[col], f['ret']):+.3f}, "
              f"3분위 평균수익 하 {g['하']:+.3f}% · 중 {g['중']:+.3f}% · 상 {g['상']:+.3f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["spread", "flow"])
    ap.add_argument("--day", required=True)
    ap.add_argument("--start", default="151800")
    ap.add_argument("--end", default="151959")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "flow":
        cmd_flow(a.day)
    else:
        cmd_spread(a.day, a.start, a.end)


if __name__ == "__main__":
    main()
