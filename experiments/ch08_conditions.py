"""8장 실험 — 6대 조건·재료·거래대금 (일봉, 개발 기간) + 15:18 1분봉 특징 (1분봉 기간).

사용법 (먼저 python -m experiments.ch05_leaders build)
  python -m experiments.ch08_conditions material   # 8-1 공시(재료)
  python -m experiments.ch08_conditions tv         # 8-1 거래대금 기준 (상위 50위, 5일 1,000억)
  python -m experiments.ch08_conditions six        # 8-6 6대 조건 완전충족과 조건별 기여도
  python -m experiments.ch08_conditions build1518  # 15:18 1분봉 특징 계산 (시간이 걸림)
  python -m experiments.ch08_conditions candle     # 8-2 15:18 부분봉 캔들 vs 최종 일봉
  python -m experiments.ch08_conditions minute     # 8-4 1분봉 20이평·VWAP·저점 상승
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from backtest.costs import net_return
from backtest.metrics import fmt_table, summarize
from experiments.ch05_leaders import by
from experiments.ch06_p1 import horizons
from experiments.ch06_p1 import load as load_p1
from data.store import path
from strategies.entry_patterns import p1

pd.set_option("display.width", 200)
F1518 = path("research", "intraday_1518.parquet")

# 공시 유형 (보고서명 키워드). 위에서부터 먼저 맞는 유형 하나만 붙인다.
DISC_TYPES = [
    ("공급계약", ["공급계약"]),
    ("실적", ["영업(잠정)실적", "매출액또는손익구조"]),
    ("유상증자·CB", ["유상증자결정", "전환사채권발행결정", "신주인수권부사채권발행결정", "교환사채권발행결정"]),
    ("무상증자", ["무상증자결정"]),
    ("자기주식", ["자기주식취득", "자기주식처분"]),
    ("최대주주·M&A", ["최대주주변경", "합병결정", "분할결정", "타법인주식및출자증권취득결정"]),
    ("임상·특허·기술", ["임상", "특허", "기술이전", "기술도입"]),
]


def stats(groups: dict[str, pd.DataFrame], col: str = "net_on") -> str:
    return fmt_table({k: summarize(v, col) for k, v in groups.items()}).iloc[:, :8].to_string()


def disclosures() -> pd.DataFrame:
    """DART 과거 공시 → (symbol, dt) 별 건수와 유형. 정정 공시([기재정정] 등)와 첨부추가는 뺀다."""
    files = sorted(path("dart_hist").glob("*.parquet"))
    cols = ["stock_code", "report_nm", "rcept_dt", "corp_cls"]
    frames = [pd.read_parquet(f) for f in files]
    d = pd.concat([x.loc[:, cols] for x in frames if not x.empty], ignore_index=True)  # 공시 없는 날 제외
    d = d[d["stock_code"].fillna("").str.len() == 6]
    d = d[~d["report_nm"].str.contains(r"^\[(?:기재정정|첨부정정|첨부추가|변경등록)\]", regex=True)]
    d["type"] = "기타"
    for name, keys in reversed(DISC_TYPES):  # 목록 앞쪽 유형이 우선하도록 거꾸로 덮어쓴다
        hit = np.zeros(len(d), dtype=bool)
        for k in keys:
            hit |= d["report_nm"].str.contains(k, regex=False).to_numpy()
        d.loc[hit, "type"] = name
    d = d.rename(columns={"stock_code": "symbol", "rcept_dt": "dt"})
    top = d[d["type"] != "기타"].groupby(["symbol", "dt"])["type"].first()
    cnt = d.groupby(["symbol", "dt"]).size().rename("n_disc")
    return pd.concat([cnt, top], axis=1).reset_index()


def add_disclosures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(disclosures(), on=["symbol", "dt"], how="left")
    df["n_disc"] = df["n_disc"].fillna(0)
    df["n_disc_prev"] = df.groupby("symbol", sort=False)["n_disc"].shift(1).fillna(0)
    return df


def add_tv5(df: pd.DataFrame) -> pd.DataFrame:
    df["tv5"] = df.groupby("symbol", sort=False)["value"].transform(lambda s: s.rolling(5, min_periods=5).sum())
    return df


def cmd_material() -> None:
    df = load_p1(extra=add_disclosures)
    pool = df[p1.candidates(df) & df["nxt_tradable"] & df["r_on"].notna()]
    print(f"후보 {len(pool):,}건 (거래대금 상위 100·상승·상한가 제외), 당일 공시 있음 {(pool['n_disc'] > 0).mean() * 100:.1f}%")
    print("\n당일 공시 여부 (접수 시각 불명 — 장 마감 후 공시 포함, 15:18에는 일부만 알 수 있음)")
    print(horizons(by(pool, np.where(pool["n_disc"] > 0, "당일 공시 있음", "없음"))))
    print("\n당일 공시 유형")
    fallback = pd.Series(np.where(pool["n_disc"] > 0, "기타", "공시 없음"), index=pool.index)
    t = pool.assign(type=pool["type"].fillna(fallback))
    print(horizons(by(t, t["type"])))
    print("\n전일 공시 여부 (15:18 이전에 확실히 알 수 있음)")
    print(horizons(by(pool, np.where(pool["n_disc_prev"] > 0, "전일 공시 있음", "없음"))))


def cmd_tv() -> None:
    df = load_p1(extra=add_tv5)
    pool = df[p1.candidates(df) & df["nxt_tradable"] & df["r_on"].notna()]
    print("거래대금 순위 (후보 = 상위 100·상승·상한가 제외)")
    print(horizons(by(pool, np.where(pool["tv_rank"] <= 50, "상위 50위 이내", "51~100위"))))
    print("\n최근 5거래일 누적 거래대금 (오늘 포함)")
    print(horizons(by(pool, pd.cut(pool["tv5"], [0, 1e11, 3e11, 1e12, 1e15],
                                   labels=["1,000억 미만", "1,000억~3,000억", "3,000억~1조", "1조 이상"]))))


def six_conditions(df: pd.DataFrame, rise: float = 0.15) -> pd.DataFrame:
    """참고자료 6대 조건을 원문 기준 그대로 옮긴 판정 (일봉, 장 마감 기준)."""
    g = df.groupby("symbol", sort=False)
    for n in (10, 120):
        df[f"ma{n}"] = g["close"].transform(lambda s, n=n: s.rolling(n, min_periods=n).mean())
    df = add_tv5(df)
    df["mcap"] = df["listCount"] * df["close"]     # 현재 상장주식수 × 당시 종가 (근사)
    # 당일 가장 강한 섹터: 구성 5종목 이상 업종 중 거래대금 가중 등락률 상위 3
    sec = df[df["tradable"] & (df["sector"] != "미분류")].groupby(["dt", "sector"]).agg(
        sret=("sector_ret", "first"), n=("symbol", "size")).reset_index()
    sec = sec[sec["n"] >= 5]
    sec["srank"] = sec.groupby("dt")["sret"].rank(ascending=False, method="first")
    df = df.merge(sec.loc[:, ["dt", "sector", "srank"]], on=["dt", "sector"], how="left")
    c = pd.DataFrame(index=df.index)
    c["C1 신고가"] = df["breakout252"]
    c["C2 정배열"] = (df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"]) & (df["ma20"] > df["ma60"]) & (df["ma60"] > df["ma120"])
    c["C3 주도 섹터"] = df["srank"] <= 3
    c["C4 상승률"] = df["ret"] >= rise
    c["C5 거래대금"] = (df["tv_rank"] <= 50) | (df["tv5"] >= 1e11)
    c["C6 시총"] = df["mcap"] >= 1e12
    return df.join(c.fillna(False))


COND = ["C1 신고가", "C2 정배열", "C3 주도 섹터", "C4 상승률", "C5 거래대금", "C6 시총"]


def cmd_six() -> None:
    df = load_p1(extra=six_conditions)
    base = df[df["tradable"] & ~df["limit_up"] & (df["ret"] > 0) & df["nxt_tradable"] & df["r_on"].notna()
              & df["ma120"].notna() & df["hh252_prev"].notna()]
    print(f"모집단: 상승·상한가 제외·252일 고점과 MA120 계산 가능 {len(base):,}건")
    print("\n조건별 충족 비율:", {k: f"{base[k].mean() * 100:.1f}%" for k in COND})
    allc = base[COND].all(axis=1)
    print("\n6대 조건 완전충족 (상승률 15% 이상)")
    print(horizons({"완전충족": base[allc]}))
    print("\n상승률 기준 바꾸기 (나머지 5개 조건 충족)")
    rest = base[[k for k in COND if k != "C4 상승률"]].all(axis=1)
    print(horizons({f"상승률 {int(r * 100)}% 이상": base[rest & (base["ret"] >= r)] for r in (0.0, 0.05, 0.10, 0.15)}))
    print("\n하나씩 빼기 (나머지 5개 충족, 상승률 15%)")
    print(horizons({f"{k} 제외": base[base[[x for x in COND if x != k]].all(axis=1)] for k in COND}))
    print("\n하나만 (그 조건만 충족해도 포함)")
    print(horizons({k: base[base[k]] for k in COND}))
    print("\n충족 개수별")
    n = base[COND].sum(axis=1)
    print(horizons(by(base, n)))


# ---------------------------------------------------------------- 15:18 1분봉
def cmd_build1518() -> None:
    from features import intraday
    cand = pd.read_parquet(path("universe", "minute_candidates.parquet"))["symbol"].tolist()
    df = intraday.build_all(cand)
    df.to_parquet(F1518, index=False)
    print(f"{len(df):,}행, {df['symbol'].nunique()}종목, {df['dt'].min()}~{df['dt'].max()} → {F1518}")


def load1518() -> pd.DataFrame:
    df = pd.read_parquet(F1518)
    df = df[df["nxt_open"].notna() & df["p_t1"].notna() & (df["ret_1518"].abs() <= 0.305)].copy()
    df["net_t1"] = net_return(df["r_on_t1"], df["nxt_dt"])
    df["net_t0"] = net_return(df["r_on_t0"], df["nxt_dt"])
    df["limit_1518"] = df["ret_1518"] >= 0.295
    return df


def cmd_candle() -> None:
    df = load1518()
    pool = df[(df["rank_1518"] <= 100) & (df["ret_1518"] > 0) & ~df["limit_1518"]]
    print(f"15:18 후보 {len(pool):,}건 (15:18 누적 거래대금 상위 100·상승·상한가 아님), "
          f"{pool['dt'].min()}~{pool['dt'].max()}")
    print("\n15:18 후보 중 최종 일봉 기준으로도 상승 마감한 비율:",
          f"{(pool['ret_final'] > 0).mean() * 100:.1f}%")
    print("\n15:18 CLV (T-1 진입 / T0 진입, 다음 날 시가, 비용 후)")
    b = pd.cut(pool["clv_1518"], [-0.01, 0.5, 0.7, 0.9, 1.01], labels=["~0.5", "0.5~0.7", "0.7~0.9", "0.9~1"])
    print(stats(by(pool, b), "net_t1"))
    print(stats(by(pool, b), "net_t0"))
    print("\n15:18 윗꼬리")
    b = pd.cut(pool["upper_wick_1518"], [-0.01, 0.1, 0.25, 0.5, 1.01], labels=["~0.1", "0.1~0.25", "0.25~0.5", "0.5~"])
    print(stats(by(pool, b), "net_t1"))


def cmd_minute() -> None:
    df = load1518()
    pool = df[(df["rank_1518"] <= 100) & (df["ret_1518"] > 0) & ~df["limit_1518"]].copy()
    pool["vs_vwap"] = pool["last_1518"] / pool["vwap_1518"] - 1
    pool["vs_ma20"] = pool["last_1518"] / pool["ma20_1m"] - 1
    print("15:18 현재가 vs 당일 VWAP")
    print(stats(by(pool, pd.cut(pool["vs_vwap"], [-1, -0.01, 0, 0.01, 0.03, 1],
                                labels=["-1% 미만", "-1~0%", "0~1%", "1~3%", "3% 이상"])), "net_t1"))
    print("\n15:18 현재가 vs 1분봉 20이평")
    print(stats(by(pool, pd.cut(pool["vs_ma20"], [-1, -0.005, 0, 0.005, 0.01, 1],
                                labels=["-0.5% 미만", "-0.5~0%", "0~0.5%", "0.5~1%", "1% 이상"])), "net_t1"))
    print("\n최근 30분 저점 상승 (14:48~15:17 최저가 > 14:18~14:47 최저가)")
    print(stats(by(pool, pool["low_rising_30m"].map({1.0: "저점 상승", 0.0: "저점 하락"})), "net_t1"))
    print("\n최근 30분 수익률 (14:48 종가 → 15:17 종가)")
    print(stats(by(pool, pd.cut(pool["mom_30m"], [-1, -0.01, 0, 0.01, 0.02, 1],
                                labels=["-1% 미만", "-1~0%", "0~1%", "1~2%", "2% 이상"])), "net_t1"))
    ok = (pool["vs_vwap"] > 0) & (pool["vs_ma20"] > 0) & (pool["low_rising_30m"] == 1)
    print("\n세 조건 동시 충족 (VWAP 위 & 20이평 위 & 저점 상승)")
    print(stats({"충족": pool[ok], "미충족": pool[~ok]}, "net_t1"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["material", "tv", "six", "build1518", "candle", "minute"])
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    globals()[f"cmd_{a.cmd}"]()


if __name__ == "__main__":
    main()
