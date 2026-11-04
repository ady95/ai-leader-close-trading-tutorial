"""5장 실험 — 주도주 탐색 (일봉, 개발 기간 2023-01~2025-12).

사용법
  python -m experiments.ch05_leaders build     # 특징 계산 → data/lake/research/dev_features.parquet
  python -m experiments.ch05_leaders grid      # 5-1 거래대금 순위 × 등락률별 익일 시가 갭
  python -m experiments.ch05_leaders e0        # 5-1 E0 기준선 (비용·슬리피지·연도별)
  python -m experiments.ch05_leaders sector    # 5-2 섹터 특징별 E0 성과
  python -m experiments.ch05_leaders regime    # 5-3 시장 국면별 성과와 주도 섹터 지속성
  python -m experiments.ch05_leaders ranker    # 5-4 랭킹 비교 (고정 가중 점수)

모든 결과는 "장 마감 정보를 다 안다"는 일봉 근사다. 15:18 실제 판단 시점의 결과는 9장에서 다시 잰다.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from backtest.costs import net_return
from backtest.metrics import fmt_table, summarize
from data import dataset as ds
from data.store import path
from strategies.leader_ranker import features as F

FEAT = path("research", "dev_features.parquet")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def load() -> pd.DataFrame:
    df = pd.read_parquet(FEAT)
    base = df[df["tradable"] & df["nxt_tradable"] & (df["ret"].abs() <= 0.305) & df["r_on"].notna()].copy()
    base["net_on"] = net_return(base["r_on"], base["nxt_dt"])
    return base


def e0_select(base: pd.DataFrame, limit_up: bool = False) -> pd.DataFrame:
    """E0: 그날 거래대금 상위 30 + 등락률 5% 이상. 기본은 상한가 제외(종가 체결 불확실)."""
    sel = base[(base["tv_rank"] <= 30) & (base["ret"] >= 0.05)]
    return sel[sel["limit_up"]] if limit_up else sel[~sel["limit_up"]]


def table(groups: dict[str, pd.DataFrame], col: str = "net_on", cols: int = 8) -> str:
    return fmt_table({k: summarize(g, col) for k, g in groups.items()}).iloc[:, :cols].to_string()


def by(df: pd.DataFrame, key) -> dict[str, pd.DataFrame]:
    return {str(k): g for k, g in df.groupby(key, observed=True)}


# ---------------------------------------------------------------------------
def cmd_build() -> None:
    df = F.build(ds.load_daily("dev"))
    df.to_parquet(FEAT, index=False)
    print(f"{len(df):,}행, {df['symbol'].nunique():,}종목, {df['dt'].min()}~{df['dt'].max()} → {FEAT}")


def cmd_grid() -> None:
    b = load()
    b["거래대금 순위"] = pd.cut(b["tv_rank"], [0, 10, 30, 100, 300, 1e9],
                          labels=["1-10", "11-30", "31-100", "101-300", "301+"])
    b["등락률"] = pd.cut(b["ret"], [-1, -1e-4, 0.05, 0.10, 0.20, 0.295, 1],
                      labels=["하락", "0~5%", "5~10%", "10~20%", "20~29.5%", "상한가"])
    for name, f in [("평균(%)", "mean"), ("중앙값(%)", "median")]:
        print(f"\n익일 시가 갭 {name}, 비용 전")
        print((b.pivot_table(index="거래대금 순위", columns="등락률", values="r_on", aggfunc=f,
                             observed=True) * 100).round(2).to_string())
    print("\n표본 수")
    print(b.pivot_table(index="거래대금 순위", columns="등락률", values="r_on", aggfunc="size",
                        observed=True).to_string())


def cmd_e0() -> None:
    b = load()
    b["r_oc"] = b["nxt_close"] / b["close"] - 1
    b["net_oc"] = net_return(b["r_oc"], b["nxt_dt"])
    e0 = e0_select(b)
    lu = e0_select(b, limit_up=True)
    print("E0 = 거래대금 상위 30 & 등락률 5% 이상, 종가 매수 → 익일 시가 매도(S1)")
    print(table({"E0 비용 전": e0.assign(x=e0["r_on"])}, "x"))
    print(table({"E0 비용 후": e0, "E0 익일 종가 매도(비용 후)": e0.assign(net_on=e0["net_oc"])}))
    print(table({"상한가 종목 (체결 가정, 비용 후)": lu}, cols=7))
    print(f"\n신호 일수 {e0['dt'].nunique()} / 하루 평균 {len(e0) / e0['dt'].nunique():.1f}건")
    print(f"익일 시가→종가 평균 {e0['r_id'].mean() * 100:.2f}%, 중앙값 {e0['r_id'].median() * 100:.2f}%")
    print("\n연도별 (비용 후)")
    print(table(by(e0, e0["dt"].str[:4])))
    print("\n편도 슬리피지 민감도 (거래당 평균 순수익)")
    for s in (0, 0.0005, 0.001, 0.002):
        print(f"  {s * 100:.2f}% → {net_return(e0['r_on'], e0['nxt_dt'], s).mean() * 100:.2f}%")


def cmd_sector() -> None:
    e0 = e0_select(load())
    print(f"E0 {len(e0):,}건, 업종 미분류 {e0['sector_top_peers'].isna().sum()}건 제외")
    e0 = e0[e0["sector_top_peers"].notna()]
    print("\n같은 업종 종목 중 그날 거래대금 상위 100에 든 수 (자기 제외)")
    print(table(by(e0, pd.cut(e0["sector_top_peers"], [-1, 0, 2, 5, 100], labels=["0", "1-2", "3-5", "6+"]))))
    print("\n동료 종목 등락률 (자기 제외, 거래대금 가중)")
    print(table(by(e0, pd.cut(e0["sector_ret_ex"], [-1, 0, 0.01, 0.02, 0.04, 1],
                              labels=["<0", "0~1%", "1~2%", "2~4%", "4%+"]))))
    print("\n업종 거래대금 비중 / 직전 20일 평균")
    print(table(by(e0, pd.cut(e0["sector_share_ratio20"], [0, 0.8, 1.0, 1.2, 1.5, 100],
                              labels=["<0.8", "0.8~1.0", "1.0~1.2", "1.2~1.5", "1.5+"]))))
    print("\n업종 안 거래대금 순위")
    print(table(by(e0, np.where(e0["sector_tv_rank"] == 1, "업종 1위", "2위 이하"))))
    d = pd.read_parquet(FEAT, columns=["dt", "sector", "value", "tradable"])
    d = d[d["tradable"] & (d["sector"] != "미분류")]
    sh = d.groupby(["dt", "sector"])["value"].sum()
    sh = (sh / sh.groupby(level=0).transform("sum")).rename("share").reset_index()
    top = sh.sort_values("share", ascending=False).groupby("dt").head(1)
    print(f"\n업종 거래대금 비중 1위: {top['sector'].value_counts().head(3).to_dict()}"
          f", 평균 비중 {top['share'].mean() * 100:.1f}%")


def leading_sector(feat: pd.DataFrame) -> pd.DataFrame:
    """그날 '돈이 새로 몰린 업종': 거래대금 상위 100 종목 수가 직전 20일 평균보다 가장 많이 늘어난 업종."""
    d = feat[feat["sector"] != "미분류"]
    n = d.groupby(["dt", "sector"])["in_top"].sum().unstack(fill_value=0).sort_index()
    excess = (n - n.shift(1).rolling(20, min_periods=10).mean()).dropna(how="all")
    out = pd.DataFrame({"sector": excess.idxmax(axis=1), "excess": excess.max(axis=1)}).dropna()
    return out[out["excess"] > 0]


def cmd_regime() -> None:
    b = load()
    e0 = e0_select(b)
    e0 = e0[e0["regime"].notna()]
    print("소속 시장 지수의 국면별 E0 (비용 후). 상승: 종가>MA20>MA60, 하락: 종가<MA20<MA60")
    print(table(by(e0, ["market", "regime"])))
    days = b.groupby(["market", "regime"])["dt"].nunique()
    print("\n국면별 거래일 수\n" + days.to_string())
    feat = pd.read_parquet(FEAT, columns=["dt", "sector", "in_top"])
    ls = leading_sector(feat)
    same = (ls["sector"] == ls["sector"].shift(1)).iloc[1:]
    runs = (ls["sector"] != ls["sector"].shift(1)).cumsum()
    run_len = ls.groupby(runs).size()
    print(f"\n'돈이 새로 몰린 업종' 판정일 {len(ls)}일, 다음 날도 같은 업종일 확률 {same.mean() * 100:.1f}%,"
          f" 평균 연속 {run_len.mean():.2f}일, 최장 {run_len.max()}일")
    print("가장 자주 나온 업종:", ls["sector"].value_counts().head(5).to_dict())


def rank_ic(df: pd.DataFrame, col: str, target: str = "r_on") -> pd.Series:
    """날짜별 스피어만 순위상관 (순위를 매긴 뒤 피어슨 상관)."""
    r = df.loc[:, ["dt", col, target]].dropna()
    r[col] = r.groupby("dt")[col].rank()
    r[target] = r.groupby("dt")[target].rank()
    return r.groupby("dt").apply(lambda d: d[col].corr(d[target]) if len(d) > 2 else np.nan).dropna()


def score(day: pd.DataFrame) -> pd.Series:
    """고정 동일가중 점수 — 그날 후보 안 백분위의 평균. 가중치는 데이터를 보기 전에 정했다."""
    cols = ["ret", "tv_ratio20", "clv", "near_high", "sector_top_peers", "sector_ret_ex"]
    return day[cols].rank(pct=True).mean(axis=1, skipna=True)


def cmd_ranker() -> None:
    b = load()
    feat = pd.read_parquet(FEAT, columns=["symbol", "dt", "tv_rank", "close"])
    # 사후 '주도 지속' 라벨: 향후 5거래일 중 3일 이상 거래대금 상위 30 & 5일 뒤 종가가 오늘 종가보다 높음
    feat = feat.sort_values(["symbol", "dt"])
    g = feat.groupby("symbol")
    fut = sum(g["tv_rank"].shift(-k).le(30).astype(float) for k in range(1, 6))
    n_fut = sum(g["tv_rank"].shift(-k).notna().astype(int) for k in range(1, 6))
    c5 = g["close"].shift(-5)
    feat["leader_next5"] = np.where((n_fut == 5) & c5.notna(), ((fut >= 3) & (c5 > feat["close"])).astype(float),
                                    np.nan)
    b = b.merge(feat.loc[:, ["symbol", "dt", "leader_next5"]], on=["symbol", "dt"], how="left")
    pool = b[(b["tv_rank"] <= 100) & ~b["limit_up"] & (b["ret"] > 0)].copy()
    pool["score"] = pool.groupby("dt", group_keys=False).apply(score)
    picks = {
        "거래대금 상위 5": pool[pool.groupby("dt")["tv_rank"].rank(method="first") <= 5],
        "등락률 상위 5": pool[pool.groupby("dt")["ret"].rank(ascending=False, method="first") <= 5],
        "점수 상위 5": pool[pool.groupby("dt")["score"].rank(ascending=False, method="first") <= 5],
        "점수 하위 5": pool[pool.groupby("dt")["score"].rank(method="first") <= 5],
        "후보 전체 (상위100·상승)": pool,
    }
    print("후보 = 거래대금 상위 100, 상승, 상한가 제외. 매일 5종목, 종가 매수 → 익일 시가 매도(비용 후)")
    print(table(picks))
    print("\n사후 '주도 지속' 적중률 (향후 5일 중 3일+ 거래대금 상위 30 & 5일 뒤 상승)")
    for k, v in picks.items():
        print(f"  {k}: {v['leader_next5'].mean() * 100:.1f}% (n={v['leader_next5'].notna().sum():,})")
    ic = rank_ic(pool, "score")
    print(f"\n점수와 익일 시가 갭의 일별 순위상관(IC) 평균 {ic.mean():.3f}, "
          f"t={ic.mean() / ic.std() * np.sqrt(len(ic)):.2f}, 양수인 날 {(ic > 0).mean() * 100:.1f}%")
    cols = ["ret", "tv_ratio20", "clv", "near_high", "sector_top_peers", "sector_ret_ex"]
    print("\n특징별 일별 IC 평균 (익일 시가 갭)")
    for c in cols:
        icc = rank_ic(pool, c)
        print(f"  {c:18s} {icc.mean():+.3f} (t={icc.mean() / icc.std() * np.sqrt(len(icc)):+.2f})")
    print("\n연도별 점수 상위 5 (비용 후)")
    sp = picks["점수 상위 5"]
    print(table(by(sp, sp["dt"].str[:4])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "grid", "e0", "sector", "regime", "ranker"])
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    globals()[f"cmd_{a.cmd}"]()


if __name__ == "__main__":
    main()
