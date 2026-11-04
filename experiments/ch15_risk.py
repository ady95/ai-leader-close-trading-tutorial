"""15-3·15-5 위험 분석 — 홀드아웃(2026)과 개발 기간의 꼬리 손실, 국면 필터, 일별 손실 분포.

사용법 (먼저 RESEARCH_UNSEAL_HOLDOUT=1 python -m experiments.ch15_holdout run)
  RESEARCH_UNSEAL_HOLDOUT=1 python -m experiments.ch15_risk run

국면 필터는 14-5에서 홀드아웃을 열기 전에 정한 것이다. 15:18 판단에 맞게 '전 거래일' 국면을 쓴다.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from backtest.metrics import fmt_table, max_drawdown, summarize
from data import dataset as ds
from data.store import path
from experiments.ch05_leaders import e0_select
from experiments.ch12_lgbm import BASE_FEATURES, CATEGORICAL
from models.lightgbm import walkforward as W
from strategies.entry_patterns import p1

pd.set_option("display.width", 200)


def daily_stats(trades: pd.DataFrame, name: str) -> dict:
    d = trades.groupby("dt")["net_on"].mean().sort_index()
    q = d.quantile(0.05)
    return {"이름": name, "거래일": len(d), "일평균": d.mean(), "최악일": d.min(), "−2% 미만 날": (d < -0.02).mean(),
            "일별 CVaR5": d[d <= q].mean(), "MDD": max_drawdown(d)}


def fmt_daily(rows: list[dict]) -> str:
    df = pd.DataFrame(rows).set_index("이름")
    for c in ["일평균", "최악일", "−2% 미만 날", "일별 CVaR5", "MDD"]:
        df[c] = df[c].map(lambda x: f"{x * 100:.2f}%")
    return df.to_string()


def cmd_run() -> None:
    d = pd.read_parquet(path("research", "holdout_frame.parquet"))
    reg = d.loc[:, ["dt", "market", "regime"]].drop_duplicates(["dt", "market"]).sort_values("dt")
    reg["regime_prev"] = reg.groupby("market")["regime"].shift(1)
    d = d.merge(reg.loc[:, ["dt", "market", "regime_prev"]], on=["dt", "market"], how="left")
    dev = d[(d["dt"] >= "20230701") & (d["dt"] <= ds.DEV_END)]
    hold = d[d["dt"] >= ds.HOLDOUT_START]
    devv, hv = dev[dev["r_on"].notna()], hold[hold["r_on"].notna()]
    pool_dev, pool_h = devv[p1.candidates(devv)].copy(), hv[p1.candidates(hv)].copy()
    days = np.sort(pool_dev["dt"].unique())
    fold = [{"train": ("20230701", days[-(W.EMBARGO + 1)]), "test": (ds.HOLDOUT_START, "99991231")}]
    both = pd.concat([pool_dev, pool_h])
    pred, _ = W.walk_forward(both, BASE_FEATURES, "r_on", fold, CATEGORICAL)
    pool_h["pred"] = pred.loc[pool_h.index]
    lgb5 = pool_h[pool_h.groupby("dt")["pred"].rank(ascending=False, method="first") <= 5]
    e0_h, e0_d = e0_select(hv), e0_select(devv)

    print("[15-3] 거래 단위 꼬리 손실 (종가 → 다음 날 시가, 비용 후)")
    for name, t in [("E0 개발", e0_d), ("E0 홀드아웃", e0_h), ("후보 전체 개발", pool_dev), ("후보 전체 홀드아웃", pool_h),
                    ("LightGBM 상위 5 홀드아웃", lgb5)]:
        r = t["net_on"]
        print(f"  {name}: {len(r):,}건, −3% 이하 {(r <= -0.03).mean() * 100:.2f}%, −5% 이하 {(r <= -0.05).mean() * 100:.2f}%, "
              f"−10% 이하 {(r <= -0.10).mean() * 100:.2f}%, 하위 1% 평균 {r[r <= r.quantile(0.01)].mean() * 100:.2f}%")
    print("\n[15-3] 일별 포트폴리오 (그날 신호를 같은 금액으로 나눠 매수)")
    print(fmt_daily([daily_stats(e0_d, "E0 개발"), daily_stats(e0_h, "E0 홀드아웃"),
                     daily_stats(lgb5, "LightGBM 상위 5 홀드아웃"), daily_stats(pool_h, "후보 전체 홀드아웃")]))
    worst = lgb5.groupby("dt")["net_on"].mean().nsmallest(5)
    print("  LightGBM 상위 5 최악의 날:", {k: f"{v * 100:.2f}%" for k, v in worst.items()})
    idx = d.drop_duplicates("dt").set_index("dt")
    print("  그날 코스피 지수 등락:", {k: f"{idx.loc[k, 'idx_ret'] * 100:.2f}%" if k in idx.index else "" for k in worst.index})

    print("\n[15-5] 전일 국면 필터 (14-5에서 정한 T4: 전일 하락 국면이면 매수하지 않음)")
    rows = {}
    for name, t in [("E0 홀드아웃", e0_h), ("LightGBM 상위 5 홀드아웃", lgb5), ("E0 개발", e0_d)]:
        rows[f"{name} 전체"] = summarize(t, "net_on")
        rows[f"{name} 전일 하락 국면 제외"] = summarize(t[t["regime_prev"] != "하락"], "net_on")
        rows[f"{name} 전일 상승 국면만"] = summarize(t[t["regime_prev"] == "상승"], "net_on")
    print(fmt_table(rows).to_string())

    print("\n[15-5] 종목 집중도: LightGBM 상위 5 홀드아웃에서 자주 뽑힌 종목·업종")
    print("  종목:", lgb5["name"].value_counts().head(5).to_dict())
    print("  업종:", (lgb5["sector"].value_counts(normalize=True).head(5) * 100).round(1).to_dict())
    same = lgb5.groupby("dt")["sector"].agg(lambda s: s.value_counts().iloc[0])
    print(f"  같은 날 5종목 중 같은 업종 최대 수: 평균 {same.mean():.2f}, 3개 이상인 날 {(same >= 3).mean() * 100:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cmd_run()


if __name__ == "__main__":
    main()
