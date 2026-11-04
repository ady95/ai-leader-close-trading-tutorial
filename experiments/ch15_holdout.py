"""15-2 최종 홀드아웃 — 고정한 일봉 규칙(E0~E2)과 일봉 LightGBM을 2026-01~에 한 번 적용한다.

사용법
  RESEARCH_UNSEAL_HOLDOUT=1 python -m experiments.ch15_holdout run

규칙과 모델은 예제 저장소 커밋 0c284e0·0ad2b6d·a217057·edfe609 의 정의 그대로이며, 이 결과를 보고 바꾸지 않는다.
LightGBM은 개발 기간 전체(2023-07-01 ~ 개발 기간 마지막 5거래일 전)로 한 번 학습해 홀드아웃 전체를 예측한다.
2026-09-14 이후 일봉 종가는 1분봉 15:30 종가로 바꾼다(data.dataset). 1분봉이 없는 종목은 그 기간 종가가 비어 후보에서 빠진다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from backtest.costs import net_return
from backtest.metrics import fmt_table, summarize
from data import dataset as ds
from data.store import path
from experiments.ch05_leaders import e0_select, score
from experiments.ch08_conditions import add_disclosures
from experiments.ch12_lgbm import BASE_FEATURES, CATEGORICAL, FLOW_FEATURES, add_flows
from models.lightgbm import walkforward as W
from strategies.entry_patterns import p1, p2
from strategies.leader_ranker import features as F

pd.set_option("display.width", 200)
HOLDOUT_START = ds.HOLDOUT_START
OUT = path("research", "holdout_frame.parquet")


def build() -> pd.DataFrame:
    if os.environ.get(ds.UNSEAL_ENV) != "1":
        raise SystemExit("최종 검증에서만 실행합니다. RESEARCH_UNSEAL_HOLDOUT=1 을 명시하세요.")
    d = ds.load_daily("all")
    d = F.build(d)
    d = add_flows(add_disclosures(p2.add_features(p1.add_features(d))))
    d = d[d["tradable"] & (d["ret"].abs() <= 0.305)].copy()
    ok = d["nxt_tradable"] & d["r_on"].notna()
    d.loc[~ok, "r_on"] = np.nan
    d["net_on"] = net_return(d["r_on"], d["nxt_dt"].fillna("20261231"))
    d["pattern_type"] = np.select([p1.is_p1(d), p2.is_p2(d)], ["P1", "P2"], "없음")
    d["log_value"] = np.log10(d["value"].clip(lower=1))
    d["log_mcap"] = np.log10((d["listCount"] * d["close"]).clip(lower=1))
    d.to_parquet(OUT, index=False)
    return d


def stats(groups: dict[str, pd.DataFrame]) -> str:
    return fmt_table({k: summarize(v, "net_on") for k, v in groups.items()}).to_string()


def top(df: pd.DataFrame, col: str, k: int = 5, asc: bool = False) -> pd.DataFrame:
    return df[df.groupby("dt")[col].rank(ascending=asc, method="first") <= k]


def cmd_run() -> None:
    d = build()
    h = d[d["dt"] >= HOLDOUT_START]
    print(f"홀드아웃 {h['dt'].min()}~{h['dt'].max()}, {h['dt'].nunique()}거래일")
    src = h["close_source"].value_counts().to_dict() if "close_source" in h else {}
    print("종가 출처:", src)
    hv = h[h["r_on"].notna()]
    e0, e1, e2 = e0_select(hv), hv[p1.is_p1(hv)], hv[p2.is_p2(hv)]
    pool = hv[p1.candidates(hv)].copy()
    print("\n[고정 규칙] 종가 매수 → 다음 날 시가 매도 (S1), 비용 후")
    print(stats({"E0": e0, "E1 (P1)": e1, "E2 (P2)": e2, "P1+P2": pd.concat([e1, e2]), "후보 전체 (기준)": pool}))
    q = lambda x: x["dt"].str[:4] + "Q" + ((x["dt"].str[4:6].astype(int) - 1) // 3 + 1).astype(str)  # noqa: E731
    print("\n분기별 평균 (%)")
    print(pd.DataFrame({k: v.groupby(q(v))["net_on"].mean() * 100 for k, v in
                        {"E0": e0, "E1": e1, "E2": e2, "후보 전체": pool}.items()}).round(3).to_string())
    print("\n국면별 평균 (%)")
    print(pd.DataFrame({k: v.groupby("regime")["net_on"].mean() * 100 for k, v in
                        {"E0": e0, "E1": e1, "E2": e2, "후보 전체": pool}.items()}).round(3).to_string())

    # LightGBM: 개발 기간 전체로 한 번 학습
    dev = d[(d["dt"] >= "20230701") & (d["dt"] <= ds.DEV_END) & p1.candidates(d) & d["r_on"].notna()]
    days = np.sort(dev["dt"].unique())
    train_end = days[-(W.EMBARGO + 1)]
    fold = [{"train": ("20230701", train_end), "test": (HOLDOUT_START, "99991231")}]
    both = pd.concat([dev, pool])
    both = both[~both.index.duplicated()]
    for name, feats in (("LightGBM (특징 22)", BASE_FEATURES), ("LightGBM (+ 전일 수급, 26)", BASE_FEATURES + FLOW_FEATURES)):
        pred, _ = W.walk_forward(both, feats, "r_on", fold, CATEGORICAL)
        pool[name] = pred.loc[pool.index]
    pool["score"] = pool.groupby("dt", group_keys=False).apply(score)
    print(f"\n[LightGBM] 학습 20230701~{train_end} → 홀드아웃 예측. 매일 상위 5, S1, 비용 후")
    g = {k: top(pool, k) for k in ("LightGBM (특징 22)", "LightGBM (+ 전일 수급, 26)")}
    g |= {"LightGBM 하위 5 (특징 22)": top(pool, "LightGBM (특징 22)", asc=True),
          "거래대금 상위 5": top(pool, "tv_rank", asc=True), "고정 점수 상위 5": top(pool, "score"),
          "후보 전체": pool}
    print(stats(g))
    for k in ("LightGBM (특징 22)", "LightGBM (+ 전일 수급, 26)"):
        ic = pool.groupby("dt").apply(lambda x, k=k: x[k].rank().corr(x["r_on"].rank()))
        tp = g[k]
        print(f"  {k}: 일별 IC {ic.mean():.3f} (t={ic.mean() / ic.std() * np.sqrt(ic.count()):.2f}), "
              f"분기별 상위5 평균(%) {(tp.groupby(q(tp))['net_on'].mean() * 100).round(3).to_dict()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cmd_run()


if __name__ == "__main__":
    main()
