"""9장 실험 — 15:18 판단과 진입 방식 (1분봉 기간 2025-11~2026-11, 1분봉 후보 종목).

사용법 (먼저 python -m experiments.ch08_conditions build1518)
  python -m experiments.ch09_entry overlap   # 9-1 15:18 신호 vs 장 마감 후 신호
  python -m experiments.ch09_entry entry     # 9-2·9-4 T-1(15:18~15:19) vs T0(종가 단일가)
  python -m experiments.ch09_entry split     # 9-3 일괄 vs 2~3회 분할
  python -m experiments.ch09_entry cost      # 9-5 호가단위 비용·참여율

1분봉 기간은 일봉 홀드아웃(2026)과 겹친다. E0~E2 규칙은 이미 고정했으므로 바꾸지 않고,
1분봉 기반 규칙은 시간순 검증(분기별 안정성, 15-1 Walk-Forward)으로 판단한다.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from backtest.costs import net_return
from backtest.metrics import fmt_table, summarize
from experiments.ch05_leaders import by
from experiments.ch08_conditions import load1518

pd.set_option("display.width", 200)


def stats(groups: dict[str, pd.DataFrame], col: str) -> str:
    return fmt_table({k: summarize(v, col) for k, v in groups.items()}).iloc[:, :8].to_string()


def quarter(dt: pd.Series) -> pd.Series:
    return dt.str[:4] + "Q" + ((dt.str[4:6].astype(int) - 1) // 3 + 1).astype(str)


def tick_size(price: pd.Series) -> pd.Series:
    """KRX 호가단위 (2023-01 개편 이후 코스피·코스닥 공통)."""
    bins = [0, 2_000, 5_000, 20_000, 50_000, 200_000, 500_000, np.inf]
    ticks = [1, 5, 10, 50, 100, 500, 1_000]
    return pd.cut(price, bins, right=False, labels=ticks).astype(float)


def pools(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    up = ~df["limit_1518"]
    return {
        "pool": df[(df["rank_1518"] <= 100) & (df["ret_1518"] > 0) & up],
        "e0_1518": df[(df["rank_1518"] <= 30) & (df["ret_1518"] >= 0.05) & up],
        "e0_final": df[(df["rank_final"] <= 30) & (df["ret_final"] >= 0.05) & (df["ret_final"] < 0.295)],
    }


def cmd_overlap() -> None:
    df = load1518()
    p = pools(df)
    a, b = p["e0_1518"], p["e0_final"]
    ka, kb = set(zip(a["symbol"], a["dt"])), set(zip(b["symbol"], b["dt"]))
    both = ka & kb
    print(f"기간 {df['dt'].min()}~{df['dt'].max()}, {df['dt'].nunique()}거래일")
    print(f"E0@15:18 {len(ka):,}건 / E0@장마감(사후) {len(kb):,}건 / 겹침 {len(both):,}건")
    print(f"  15:18 신호 중 장 마감에도 E0인 비율 {len(both) / len(ka) * 100:.1f}%")
    print(f"  장 마감 E0 중 15:18에 이미 잡힌 비율 {len(both) / len(kb) * 100:.1f}%")
    a = a.assign(also=[k in kb for k in zip(a["symbol"], a["dt"])])
    print(f"15:18 신호가 장 마감에 빠진 이유: 등락률 5% 미만 마감 {((a['ret_final'] < 0.05) & ~a['also']).sum()}건, "
          f"상한가 마감 {((a['ret_final'] >= 0.295) & ~a['also']).sum()}건")
    r = df[df["rank_1518"] <= 30]
    top = r.groupby("dt").apply(lambda d: (d["rank_final"] <= 30).mean())
    print(f"15:18 거래대금 상위 30 중 최종 상위 30 유지 비율 평균 {top.mean() * 100:.1f}%")
    print("\n성과 비교 (다음 날 시가 매도, 비용 후)")
    print(stats({"E0@15:18 → T-1 진입": a, "E0@장마감 → T-1 진입 (사후 정보)": b}, "net_t1"))
    print(stats({"E0@15:18 → T0 진입": a, "E0@장마감 → T0 진입 (사후 정보)": b}, "net_t0"))
    print("\n분기별 E0@15:18 → T-1 진입")
    print(stats(by(a, quarter(a["dt"])), "net_t1"))


def cmd_entry() -> None:
    df = load1518()
    p = pools(df)
    for name in ("pool", "e0_1518"):
        x = p[name]
        print(f"\n[{name}] {len(x):,}건")
        print(stats({"T-1 (15:18~15:19)": x, "T0 (종가 단일가)": x.assign(net_t1=x["net_t0"])}, "net_t1"))
        d = x["p_t0"] / x["p_t1"] - 1
        print(f"종가 단일가 가격 ÷ T-1 가격 - 1: 평균 {d.mean() * 100:.3f}%, 중앙값 {d.median() * 100:.3f}%, "
              f"T0가 더 비싼 비율 {(d > 0).mean() * 100:.1f}%, 같은 비율 {(d == 0).mean() * 100:.1f}%")
        diff = x["net_t1"] - x["net_t0"]
        print("분기별 T-1 − T0 (거래당 %p):",
              (diff.groupby(quarter(x["dt"])).mean() * 100).round(3).to_dict())
    x = p["pool"]
    share = x["vol_auction"] / x["vol_regular"]
    print(f"\n종가 단일가 거래량 비중 (정규장 대비): 평균 {share.mean() * 100:.1f}%, 중앙값 {share.median() * 100:.1f}%")
    print(f"15:18~15:19 2분 거래대금 중앙값 {x['value_t1'].median() / 1e8:.1f}억 원, "
          f"하위 10% {x['value_t1'].quantile(0.1) / 1e8:.1f}억 원")


def cmd_split() -> None:
    df = load1518()
    x = pools(df)["pool"].dropna(subset=["p_1518", "p_1519"]).copy()
    variants = {
        "일괄 15:18": x["p_1518"],
        "일괄 15:19": x["p_1519"],
        "2회 분할 (15:18·15:19)": (x["p_1518"] + x["p_1519"]) / 2,
        "3회 분할 (15:18·15:19·단일가)": (x["p_1518"] + x["p_1519"] + x["p_t0"]) / 3,
        "일괄 단일가 (T0)": x["p_t0"],
    }
    groups = {}
    for k, px in variants.items():
        g = x.copy()
        g["r"] = net_return(g["nxt_open"] / px - 1, g["nxt_dt"])
        groups[k] = g
    print(f"후보 {len(x):,}건 (15:18 거래대금 상위 100·상승·상한가 아님)")
    print(stats(groups, "r"))
    base = groups["일괄 15:18"]["r"]
    for k in list(variants)[1:]:
        d = groups[k]["r"] - base
        print(f"  {k} − 일괄 15:18: {d.mean() * 100:+.3f}%p (t={d.mean() / d.std() * np.sqrt(d.count()):+.2f})")


def cmd_cost() -> None:
    df = load1518()
    x = pools(df)["pool"].dropna(subset=["p_t0"])
    one = tick_size(x["p_t1"]) / x["p_t1"]
    print(f"후보 {len(x):,}건. 1호가 = 가격 대비 평균 {one.mean() * 100:.3f}%, 중앙값 {one.median() * 100:.3f}%")
    band = pd.cut(x["p_t1"], [0, 5e3, 2e4, 5e4, 2e5, 5e5, 1e7],
                  labels=["~5천", "5천~2만", "2만~5만", "5만~20만", "20만~50만", "50만~"])
    print("가격대별 1호가 비용 중앙값(%):", (one.groupby(band, observed=True).median() * 100).round(3).to_dict())
    # 접속매매 매수는 매도호가에 사야 하므로 봉 대표가격보다 평균 반 호가 비싸다고 본다.
    # 청산(다음 날 시가 단일가)과 T0 진입(종가 단일가)은 한 가격으로 체결되므로 스프레드가 없다.
    t0 = x["net_t0"]
    print(f"T0 (종가 단일가 진입) 거래당 평균 {t0.mean() * 100:.3f}%")
    for half in (0.0, 0.5, 1.0):
        r = x["net_t1"] - half * one
        d = r - t0
        print(f"T-1, 진입 {half}호가 불리 → 거래당 평균 {r.mean() * 100:.3f}%, "
              f"T0 대비 {d.mean() * 100:+.3f}%p (t={d.mean() / d.std() * np.sqrt(d.count()):+.2f})")
    auction_value = x["vol_auction"] * x["p_t0"]
    print()
    print(f"2분(15:18~15:19) 거래대금 중앙값 {x['value_t1'].median() / 1e8:.1f}억 원, "
          f"종가 단일가 거래대금 중앙값 {auction_value.median() / 1e8:.1f}억 원")
    for amt in (1e7, 1e8, 1e9):
        p1, p0 = amt / x["value_t1"], amt / auction_value
        print(f"주문 {amt / 1e8:g}억 원: 참여율 중앙값 T-1 {p1.median() * 100:.2f}% / T0 {p0.median() * 100:.2f}%, "
              f"10% 초과 비율 T-1 {(p1 > 0.1).mean() * 100:.1f}% / T0 {(p0 > 0.1).mean() * 100:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["overlap", "entry", "split", "cost"])
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    globals()[f"cmd_{a.cmd}"]()


if __name__ == "__main__":
    main()
