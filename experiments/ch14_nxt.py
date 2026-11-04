"""14-3 실험 — KRX vs NXT (거래대금 상위 50종목, 2026-09-14~). 먼저 python -m scripts.backfill nxt --top 50 --start 20260901

  python -m experiments.ch14_nxt run

NXT 세션: 프리마켓 08:00~08:49, 메인 09:00~15:19(+15:20 이후 체결 없음), 애프터 15:40~19:59.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from data.dataset import AFTERHOURS_START
from data.store import path

pd.set_option("display.width", 200)


def _load(folder: str, s: str) -> pd.DataFrame:
    p = path(folder, f"{s}.parquet")
    if not p.exists():
        return pd.DataFrame()
    m = pd.read_parquet(p, columns=["ts", "open", "high", "low", "close", "volume"])
    m["dt"], m["hm"] = m["ts"].str[:8], m["ts"].str[8:12]
    m["value"] = m["close"].astype(float) * m["volume"]
    return m[m["dt"] >= AFTERHOURS_START]


def seg(m: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return m[(m["hm"] >= a) & (m["hm"] <= b)]


def day_rows(s: str) -> list[dict]:
    k, n = _load("minute", s), _load("minute_nx", s)
    if k.empty or n.empty:
        return []
    out = []
    for dt in sorted(set(k["dt"]) & set(n["dt"])):
        kd, nd = k[k["dt"] == dt], n[n["dt"] == dt]
        c1530 = kd.loc[kd["hm"] == "1530", "close"]
        nx1519 = seg(nd, "0900", "1519")["close"]
        pre = seg(nd, "0800", "0849")
        out.append({
            "symbol": s, "dt": dt,
            "krx_reg": seg(kd, "0900", "1530")["value"].sum(), "krx_after": seg(kd, "1600", "1959")["value"].sum(),
            "nxt_pre": pre["value"].sum(), "nxt_main": seg(nd, "0900", "1519")["value"].sum(),
            "nxt_after": seg(nd, "1540", "1959")["value"].sum(),
            "krx_close": float(c1530.iloc[0]) if len(c1530) else np.nan,
            "nxt_main_last": float(nx1519.iloc[-1]) if len(nx1519) else np.nan,
            "krx_open": float(kd.loc[kd["hm"] == "0900", "open"].iloc[0]) if (kd["hm"] == "0900").any() else np.nan,
            "nxt_pre_last": float(pre["close"].iloc[-1]) if len(pre) else np.nan,
            "nxt_pre_vwap": float((pre["close"] * pre["volume"]).sum() / pre["volume"].sum()) if pre["volume"].sum() else np.nan,
        })
    return out


def cmd_run() -> None:
    cand = pd.read_parquet(path("universe", "minute_candidates.parquet")).head(50)["symbol"]
    df = pd.DataFrame([r for s in cand for r in day_rows(s)])
    print(f"{df['symbol'].nunique()}종목, {df['dt'].nunique()}거래일, {len(df):,}종목·일 ({df['dt'].min()}~{df['dt'].max()})")
    tot = df[["krx_reg", "krx_after", "nxt_pre", "nxt_main", "nxt_after"]].sum()
    print("\n세션별 거래대금 합계 비중 (%)")
    print((tot / tot.sum() * 100).round(1).to_dict())
    share = (df["nxt_pre"] + df["nxt_main"] + df["nxt_after"]) / (df[["krx_reg", "krx_after", "nxt_pre", "nxt_main", "nxt_after"]].sum(axis=1))
    print(f"종목·일별 NXT 비중: 중앙값 {share.median() * 100:.1f}%, 평균 {share.mean() * 100:.1f}%")
    after = df["nxt_after"] / (df["nxt_after"] + df["krx_after"])
    print(f"장후(16:00~19:59 KRX vs 15:40~19:59 NXT) 중 NXT 비중: 중앙값 {after.median() * 100:.1f}%")
    d = df["nxt_main_last"] / df["krx_close"] - 1
    print(f"\nNXT 메인 마지막 가격(15:19 무렵) ÷ KRX 종가 단일가 − 1: 평균 {d.mean() * 100:+.3f}%, "
          f"중앙값 {d.median() * 100:+.3f}%, |차이| 중앙값 {d.abs().median() * 100:.3f}%")
    # 다음 날 아침: NXT 프리마켓이 KRX 시가를 미리 보여 주나
    df = df.sort_values(["symbol", "dt"])
    g = df.groupby("symbol")
    nxt = pd.DataFrame({"krx_close": df["krx_close"], "next_open": g["krx_open"].shift(-1),
                        "next_pre_last": g["nxt_pre_last"].shift(-1), "next_pre_vwap": g["nxt_pre_vwap"].shift(-1)}).dropna()
    gap = nxt["next_open"] / nxt["krx_close"] - 1
    pre = nxt["next_pre_last"] / nxt["krx_close"] - 1
    print(f"\n다음 날 KRX 시가 갭 vs NXT 프리마켓 08:49 가격 갭: 상관 {gap.corr(pre):.3f}, 방향 일치 {(np.sign(gap) == np.sign(pre)).mean() * 100:.1f}%, "
          f"|차이| 중앙값 {(gap - pre).abs().median() * 100:.3f}%  (n={len(nxt):,})")
    sell = nxt["next_pre_vwap"] / nxt["next_open"] - 1
    print(f"NXT 프리마켓 VWAP 에 팔면 KRX 시가 대비: 평균 {sell.mean() * 100:+.3f}%, 중앙값 {sell.median() * 100:+.3f}%, "
          f"더 높은 비율 {(sell > 0).mean() * 100:.1f}% (스프레드 반영 전)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cmd_run()


if __name__ == "__main__":
    main()
