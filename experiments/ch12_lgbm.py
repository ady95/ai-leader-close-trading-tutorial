"""12장 실험 — LightGBM Leader Ranker·Overnight Predictor (일봉, 개발 기간 Walk-Forward).

사용법 (먼저 python -m experiments.ch05_leaders build)
  python -m experiments.ch12_lgbm daily        # 12-1·12-2·12-5 일봉 모델 Walk-Forward 평가
  python -m experiments.ch12_lgbm daily --flows   # 수급 특징 추가 (python -m scripts.backfill investor 이후)

테스트 구간(개발 기간 안): 2024 상반기 / 2024 하반기 / 2025 상반기 / 2025 하반기.
학습은 각 테스트 시작 전까지(2023-07 이후)로 확장하며, 5거래일 간격을 둔다.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from backtest.metrics import fmt_table, summarize
from data.store import path
from experiments.ch05_leaders import score
from experiments.ch06_p1 import load as load_p1
from experiments.ch08_conditions import add_disclosures
from models.lightgbm.walkforward import folds, walk_forward
from strategies.entry_patterns import p1, p2

pd.set_option("display.width", 200)
TEST_STARTS = ["20240101", "20240701", "20250101", "20250701"]
TEST_ENDS = ["20240630", "20241231", "20250630", "20251231"]
BASE_FEATURES = ["ret", "tv_ratio20", "clv", "upper_wick", "near_high", "margin252", "ma20_slope5",
                 "sector_top_peers", "sector_ret_ex", "sector_share_ratio20", "sector_tv_rank", "tv_rank",
                 "log_value", "ret5", "days_since_high", "pullback_depth", "log_mcap", "n_disc_prev",
                 "idx_ret", "pattern_type", "regime", "market"]
FLOW_FEATURES = ["frgn_prev", "orgn_prev", "frgn_5d", "orgn_5d"]
CATEGORICAL = ["pattern_type", "regime", "market"]


def add_flows(df: pd.DataFrame) -> pd.DataFrame:
    """일별 확정 수급 — 전일까지 값만 쓴다(당일 확정치는 장 마감 후 공개, 4-3). 거래대금으로 나눠 규모를 맞춘다."""
    frames = []
    for f in path("investor_daily").glob("*.parquet"):
        d = pd.read_parquet(f, columns=["symbol", "dt", "frgnr_invsr", "orgn"])
        frames.append(d)
    fl = pd.concat(frames, ignore_index=True)
    for c in ("frgnr_invsr", "orgn"):
        fl[c] = pd.to_numeric(fl[c], errors="coerce") * 1e6          # 백만 원 → 원
    df = df.merge(fl, on=["symbol", "dt"], how="left")
    g = df.groupby("symbol", sort=False)
    v = df["value"].where(df["value"] > 0)
    for src, name in (("frgnr_invsr", "frgn"), ("orgn", "orgn")):
        ratio = df[src] / v
        df[f"{name}_prev"] = ratio.groupby(df["symbol"]).shift(1)
        df[f"{name}_5d"] = (df[src].groupby(df["symbol"]).transform(lambda s: s.shift(1).rolling(5, min_periods=3).sum())
                            / g["value"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).sum()))
    return df


def build_frame(flows: bool) -> pd.DataFrame:
    def extra(d: pd.DataFrame) -> pd.DataFrame:
        d = add_disclosures(p2.add_features(d))
        return add_flows(d) if flows else d
    df = load_p1(extra=extra)
    df["pattern_type"] = np.select([p1.is_p1(df), p2.is_p2(df)], ["P1", "P2"], "없음")
    df["log_value"] = np.log10(df["value"].clip(lower=1))
    df["log_mcap"] = np.log10((df["listCount"] * df["close"]).clip(lower=1))
    pool = df[p1.candidates(df) & df["nxt_tradable"] & df["r_on"].notna()].copy()
    # 사후 '주도 지속' 라벨 (5-4와 같은 정의)
    feat = pd.read_parquet(path("research", "dev_features.parquet"), columns=["symbol", "dt", "tv_rank", "close"])
    feat = feat.sort_values(["symbol", "dt"])
    g = feat.groupby("symbol")
    fut = sum(g["tv_rank"].shift(-k).le(30).astype(float) for k in range(1, 6))
    n_fut = sum(g["tv_rank"].shift(-k).notna().astype(int) for k in range(1, 6))
    c5 = g["close"].shift(-5)
    feat["leader_next5"] = np.where((n_fut == 5) & c5.notna(), ((fut >= 3) & (c5 > feat["close"])).astype(float), np.nan)
    return pool.merge(feat.loc[:, ["symbol", "dt", "leader_next5"]], on=["symbol", "dt"], how="left")


def pick(df: pd.DataFrame, col: str, k: int = 5, ascending: bool = False) -> pd.DataFrame:
    return df[df.groupby("dt")[col].rank(ascending=ascending, method="first") <= k]


def cmd_daily(flows: bool) -> None:
    df = build_frame(flows)
    feats = BASE_FEATURES + (FLOW_FEATURES if flows else [])
    fl = folds(df["dt"], TEST_STARTS, TEST_ENDS, train_start="20230701")
    for f in fl:
        print("폴드", f)
    df["pred_on"], imp = walk_forward(df, feats, "r_on", fl, CATEGORICAL)
    df["pred_leader"], _ = walk_forward(df, feats, "leader_next5", fl, CATEGORICAL, classifier=True)
    test = df[df["pred_on"].notna()].copy()
    test["score"] = test.groupby("dt", group_keys=False).apply(score)
    print(f"\n테스트 행 {len(test):,}건, {test['dt'].min()}~{test['dt'].max()}, 특징 {len(feats)}개")
    print("\n[Leader Ranker] 매일 상위 5의 사후 '주도 지속' 적중률")
    for name, sel in {"모델 상위 5": pick(test, "pred_leader"), "거래대금 상위 5": pick(test, "tv_rank", ascending=True),
                      "고정 점수 상위 5": pick(test, "score"), "후보 전체": test}.items():
        print(f"  {name}: {sel['leader_next5'].mean() * 100:.1f}% (n={sel['leader_next5'].notna().sum():,})")
    print("\n[Overnight Predictor] 매일 상위 5 → 종가 매수·다음 날 시가 매도 (비용 후)")
    groups = {"모델 상위 5": pick(test, "pred_on"), "모델 하위 5": pick(test, "pred_on", ascending=True),
              "거래대금 상위 5": pick(test, "tv_rank", ascending=True), "고정 점수 상위 5": pick(test, "score"),
              "후보 전체": test}
    top = groups["모델 상위 5"]
    groups["모델 상위 5 & 예측 > 0.2%"] = top[top["pred_on"] > 0.002]
    print(fmt_table({k: summarize(v, "net_on") for k, v in groups.items()}).to_string())
    ic = test.groupby("dt").apply(lambda d: d["pred_on"].rank().corr(d["r_on"].rank()))
    print(f"\n예측과 실제의 일별 순위상관 평균 {ic.mean():.3f} (t={ic.mean() / ic.std() * np.sqrt(ic.count()):.2f}), "
          f"양수인 날 {(ic > 0).mean() * 100:.1f}%")
    print("\n폴드별 모델 상위 5 vs 후보 전체 (비용 후 평균 %)")
    for f in fl:
        a = top[top["dt"].between(*f["test"])]["net_on"].mean() * 100
        b = test[test["dt"].between(*f["test"])]["net_on"].mean() * 100
        print(f"  {f['test'][0]}~{f['test'][1]}: 모델 {a:+.3f} / 전체 {b:+.3f}")
    print("\n특징 중요도 (gain, 폴드 평균 상위 12)")
    print((imp.mean(axis=1) / imp.mean(axis=1).sum() * 100).sort_values(ascending=False).head(12).round(1).to_string())
    test.loc[:, ["symbol", "dt", "pred_on", "pred_leader", "net_on", "r_on"]].to_parquet(
        path("research", f"ch12_daily_preds{'_flows' if flows else ''}.parquet"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["daily"])
    ap.add_argument("--flows", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cmd_daily(a.flows)


if __name__ == "__main__":
    main()
