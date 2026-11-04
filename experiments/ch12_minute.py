"""12-3·12-4 실험 — 15:18 시점 Overnight Predictor와 갭하락 위험 모델 (1분봉 기간 Walk-Forward).

사용법 (먼저 python -m experiments.ch08_conditions build1518)
  python -m experiments.ch12_minute run

특징은 15:18까지의 1분봉 값과 '전일까지'의 일봉 값만 쓴다. 장 마감값은 하나도 넣지 않는다.
검증: 2026-02부터 매달 그 달을 테스트하고, 그 전 달까지 전부로 학습한다(확장 창, 1거래일 간격).
1분봉 기간은 일봉 홀드아웃과 겹치므로 일봉은 '분봉 실험 모드'로 읽는다(15-1).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from backtest.costs import net_return
from backtest.metrics import fmt_table, summarize
from experiments.ch08_conditions import load1518
from experiments.ch09_entry import pools
from models.lightgbm import walkforward as W
from strategies.exit_policies.s0_s1 import entry_price

pd.set_option("display.width", 200)
FEATURES = ["ret_1518", "clv_1518", "upper_wick_1518", "vs_vwap", "vs_ma20", "ma20_1m_slope", "low_rising_30m",
            "mom_30m", "rank_1518", "log_value_1518", "tv_ratio20_1518", "peer_ret_1518", "n_peers_top",
            "near_high_prev", "ma20_slope5_prev", "ret5_prev", "log_mcap", "market"]
CATEGORICAL = ["market"]


def prev_day_daily() -> pd.DataFrame:
    """전일까지의 일봉 특징 (분봉 실험 모드로 일봉 전체를 읽는다)."""
    os.environ["RESEARCH_UNSEAL_HOLDOUT"] = "1"
    from data import dataset as ds
    from strategies.leader_ranker import features as F
    d = ds.load_daily("all")
    d = F.stock_features(d[d["dt"] >= "20241001"])
    g = d.groupby("symbol", sort=False)
    out = pd.DataFrame({"symbol": d["symbol"], "dt": d["dt"], "sector": d["sector"], "market": d["market"],
                        "near_high_prev": g["near_high"].shift(1), "ma20_slope5_prev": (d["ma20"] / g["ma20"].shift(5) - 1)
                        .groupby(d["symbol"]).shift(1), "ret5_prev": g["ret5"].shift(1),
                        "value_avg20_prev": g["value"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()),
                        "log_mcap": np.log10((d["listCount"] * g["close"].shift(1)).clip(lower=1))})
    return out


def frame() -> pd.DataFrame:
    m = load1518()
    m = m.merge(prev_day_daily(), on=["symbol", "dt"], how="left")
    m["vs_vwap"] = m["last_1518"] / m["vwap_1518"] - 1
    m["vs_ma20"] = m["last_1518"] / m["ma20_1m"] - 1
    m["log_value_1518"] = np.log10(m["value_1518"].clip(lower=1))
    m["tv_ratio20_1518"] = m["value_1518"] / m["value_avg20_prev"]
    # 15:18 시점 업종 동료 등락률 (자기 제외, 15:18 누적 거래대금 가중) — 1분봉 후보 종목 안에서
    w = m["value_1518"]
    key = [m["dt"], m["sector"]]
    rw, ws = (m["ret_1518"] * w).groupby(key).transform("sum"), w.groupby(key).transform("sum")
    den = ws - w
    m["peer_ret_1518"] = (rw - m["ret_1518"] * w) / den.where(den > 0)
    top = (m["rank_1518"] <= 100).astype(int)
    m["n_peers_top"] = top.groupby(key).transform("sum") - top
    m.loc[m["sector"].isna() | (m["sector"] == "미분류"), ["peer_ret_1518", "n_peers_top"]] = np.nan
    x = pools(m)["pool"].dropna(subset=["p_t1", "nxt_open"]).copy()
    x["entry"] = entry_price(x)
    x["r_entry"] = x["nxt_open"] / x["entry"] - 1                 # 보수적 진입가 기준 다음 날 시가 수익
    x["net"] = net_return(x["r_entry"], x["nxt_dt"])
    x["gap_down2"] = (x["r_entry"] <= -0.02).astype(float)
    return x


def monthly_folds(dates: pd.Series, first_test: str = "202602") -> list[dict]:
    days = np.sort(dates.unique())
    months = sorted({d[:6] for d in days if d[:6] >= first_test})
    out = []
    for mth in months:
        test = days[[d[:6] == mth for d in days]]
        prior = days[days < test[0]]
        out.append({"train": (days[0], prior[-2]), "test": (test[0], test[-1])})   # 1거래일 간격
    return out


def cmd_run() -> None:
    x = frame()
    fl = monthly_folds(x["dt"])
    old = W.EMBARGO
    x["pred"], imp = W.walk_forward(x, FEATURES, "r_entry", fl, CATEGORICAL)
    x["risk"], _ = W.walk_forward(x, FEATURES, "gap_down2", fl, CATEGORICAL, classifier=True)
    W.EMBARGO = old
    t = x[x["pred"].notna()].copy()
    print(f"테스트 {len(t):,}건, {t['dt'].min()}~{t['dt'].max()}, 폴드 {len(fl)}개(월), 특징 {len(FEATURES)}개")

    def top(col, k=5, asc=False):
        return t[t.groupby("dt")[col].rank(ascending=asc, method="first") <= k]
    groups = {"모델 상위 5": top("pred"), "모델 하위 5": top("pred", asc=True),
              "15:18 거래대금 상위 5": top("rank_1518", asc=True), "후보 전체": t}
    tp = groups["모델 상위 5"]
    groups["모델 상위 5 & 예측 > 0.2%"] = tp[tp["pred"] > 0.002]
    safe = tp[tp["risk"] <= tp.groupby("dt")["risk"].transform("median")]
    groups["모델 상위 5 중 갭하락 위험 낮은 절반"] = safe
    print("\n종가 매수(T-1, 반 호가 불리) → 다음 날 시가 매도, 비용 후")
    print(fmt_table({k: summarize(v, "net") for k, v in groups.items()}).to_string())
    ic = t.groupby("dt").apply(lambda d: d["pred"].rank().corr(d["r_entry"].rank()))
    print(f"\n일별 순위상관 평균 {ic.mean():.3f} (t={ic.mean() / ic.std() * np.sqrt(ic.count()):.2f})")
    print("\n월별 모델 상위 5 vs 후보 전체 (비용 후 평균 %)")
    for f in fl:
        a = tp[tp["dt"].between(*f["test"])]["net"].mean() * 100
        b = t[t["dt"].between(*f["test"])]["net"].mean() * 100
        print(f"  {f['test'][0][:6]}: 모델 {a:+.3f} / 전체 {b:+.3f}")
    q = pd.qcut(t["risk"], 5, labels=["1(낮음)", "2", "3", "4", "5(높음)"])
    print("\n갭하락(-2% 이하) 위험 예측 5분위별 실제 갭하락 비율과 평균 수익")
    print(t.groupby(q, observed=True).agg(건수=("net", "size"), 실제갭하락=("gap_down2", "mean"),
                                          평균순수익=("net", "mean")).assign(
        실제갭하락=lambda d: (d["실제갭하락"] * 100).round(1), 평균순수익=lambda d: (d["평균순수익"] * 100).round(3)).to_string())
    print("\n특징 중요도 (gain %, 상위 10)")
    print((imp.mean(axis=1) / imp.mean(axis=1).sum() * 100).sort_values(ascending=False).head(10).round(1).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cmd_run()


if __name__ == "__main__":
    main()
