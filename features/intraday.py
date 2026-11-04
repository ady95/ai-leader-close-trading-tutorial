"""15:18 판단 시점의 1분봉 특징과 진입·청산 가격 (KRX, 1분봉 후보 종목).

관측 시각 규칙 (4-4)
  1분봉의 ts 는 봉 시작 시각이다. 15:17 봉(15:17:00~15:17:59)은 15:18:00 에 완성된다.
  그래서 15:18 신호는 ts ≤ 151700 인 봉만 쓴다.

진입 가격
  T-1 (15:18~15:20 접속매매): 15:18·15:19 봉의 거래량 가중 대표가격 — 체결 순서를 모르는 근사값
  T0  (15:20~15:30 종가 단일가): 15:30 봉의 종가 = KRX 정규장 종가
청산 가격 (다음 거래일 아침, nx_ 접두사)
  09:00 봉의 시가 = KRX 시가 단일가 가격 (S1)
  09:00~09:04 봉의 거래량 가중 대표가격, 09:04·09:09·09:29·09:59 봉의 종가 (S0·S5 근사)
  09:00~09:04 고가·저가 (손절·익절 판정의 보수적 근사)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.store import path

CUTOFF = "1517"          # 15:18 신호에서 쓸 수 있는 마지막 봉 (시작 시각)


def _tp_vwap(b: pd.DataFrame) -> float:
    v = b["volume"].to_numpy(float)
    if v.sum() <= 0:
        return np.nan
    tp = ((b["high"] + b["low"] + b["close"]) / 3).to_numpy(float)
    return float((tp * v).sum() / v.sum())


def _close_at(d: pd.DataFrame, hm: str) -> float:
    """hm 시각 봉까지 중 마지막 봉의 종가 (그 분에 거래가 없으면 직전 봉)."""
    x = d.loc[(d["hm"] >= "0900") & (d["hm"] <= hm), "close"]
    return float(x.iloc[-1]) if len(x) else np.nan


def _after_hours(d: pd.DataFrame) -> dict:
    out = {}
    for name, a, b in (("t1", "1600", "1629"), ("t2", "1730", "1759"), ("t3", "1930", "1959")):
        w = d[(d["hm"] >= a) & (d["hm"] <= b)]
        out[f"p_ah_{name}"] = _tp_vwap(w) if len(w) else np.nan
        out[f"value_ah_{name}"] = float((w["close"].astype(float) * w["volume"]).sum()) if len(w) else 0.0
    ah = d[d["hm"] >= "1600"]
    out["value_ah"] = float((ah["close"].astype(float) * ah["volume"]).sum()) if len(ah) else 0.0
    out["last_ah"] = float(ah["close"].iloc[-1]) if len(ah) else np.nan
    return out


def _morning(d: pd.DataFrame) -> dict:
    """그날 아침 청산 후보 가격들 (다음 날로 shift 해서 쓴다)."""
    first5 = d[(d["hm"] >= "0900") & (d["hm"] <= "0904")]
    return {
        "m_vwap5": _tp_vwap(first5),
        "m_high5": float(first5["high"].max()) if len(first5) else np.nan,
        "m_low5": float(first5["low"].min()) if len(first5) else np.nan,
        "m_c0904": _close_at(d, "0904"), "m_c0909": _close_at(d, "0909"),
        "m_c0929": _close_at(d, "0929"), "m_c0959": _close_at(d, "0959"),
        "m_vol5": float(first5["volume"].sum()),
        # 09:00 봉 (S7 의 09:01 판단 시점 정보)
        "m_c0900": _close_at(d, "0900"),
        "m_h0900": float(d.loc[d["hm"] == "0900", "high"].max()) if (d["hm"] == "0900").any() else np.nan,
        "m_l0900": float(d.loc[d["hm"] == "0900", "low"].min()) if (d["hm"] == "0900").any() else np.nan,
        "m_v0900": float(d.loc[d["hm"] == "0900", "volume"].sum()),
        "m_c0901": _close_at(d, "0901"),
    }


def _day_features(d: pd.DataFrame) -> dict:
    reg = d[(d["hm"] >= "0900") & (d["hm"] <= CUTOFF)]
    if reg.empty or reg["volume"].sum() == 0:
        return {}
    c, v = reg["close"].to_numpy(float), reg["volume"].to_numpy(float)
    tp = (reg["high"].to_numpy(float) + reg["low"].to_numpy(float) + c) / 3
    o, h, l, last = float(reg["open"].iloc[0]), float(reg["high"].max()), float(reg["low"].min()), c[-1]
    rng = h - l
    ma20 = c[-20:].mean() if len(c) >= 20 else np.nan
    ma20_prev = c[-30:-10].mean() if len(c) >= 30 else np.nan
    w1 = reg[(reg["hm"] >= "1448") & (reg["hm"] <= CUTOFF)]["low"]
    w0 = reg[(reg["hm"] >= "1418") & (reg["hm"] < "1448")]["low"]
    c1448 = reg.loc[reg["hm"] <= "1448", "close"]
    ent = d[d["hm"].isin(["1518", "1519"])]
    ev = ent["volume"].to_numpy(float)
    etp = ((ent["high"] + ent["low"] + ent["close"]) / 3).to_numpy(float)
    auction = d.loc[d["hm"] == "1530", "close"]
    return {
        "o_1518": o, "h_1518": h, "l_1518": l, "last_1518": last,
        "vol_1518": v.sum(), "value_1518": float((c * v).sum()),
        "vwap_1518": float((tp * v).sum() / v.sum()),
        "ma20_1m": ma20, "ma20_1m_slope": ma20 / ma20_prev - 1 if ma20_prev == ma20_prev else np.nan,
        "clv_1518": (last - l) / rng if rng > 0 else np.nan,
        "upper_wick_1518": (h - max(o, last)) / rng if rng > 0 else np.nan,
        "low_rising_30m": float(w1.min() > w0.min()) if len(w1) and len(w0) else np.nan,
        "mom_30m": last / float(c1448.iloc[-1]) - 1 if len(c1448) else np.nan,
        "p_t1": float((etp * ev).sum() / ev.sum()) if ev.sum() > 0 else np.nan,
        "p_1518": _tp_vwap(d[d["hm"] == "1518"]), "p_1519": _tp_vwap(d[d["hm"] == "1519"]),
        "vol_t1": ev.sum(),
        "p_t0": float(auction.iloc[0]) if len(auction) else np.nan,
        "vol_auction": float(d.loc[d["hm"] == "1530", "volume"].sum()),
        "vol_regular": float(d.loc[(d["hm"] >= "0900") & (d["hm"] <= "1530"), "volume"].sum()),
        "value_regular": float((d.loc[(d["hm"] >= "0900") & (d["hm"] <= "1530"), "close"].astype(float)
                                * d.loc[(d["hm"] >= "0900") & (d["hm"] <= "1530"), "volume"]).sum()),
        "value_t1": float((ent["close"].astype(float) * ent["volume"]).sum()),
        "open_0900": np.nan,  # 아래에서 다음 날 값으로 채운다
        # KRX 장후 거래 (2026-09-14 이후에만 있음): T1 초기 16:00~16:29, T2 중반 17:30~17:59, T3 마감 전 19:30~19:59
        **_after_hours(d),
    }


def build_symbol(symbol: str) -> pd.DataFrame:
    p = path("minute", f"{symbol}.parquet")
    m = pd.read_parquet(p, columns=["ts", "open", "high", "low", "close", "volume"])
    m["dt"], m["hm"] = m["ts"].str[:8], m["ts"].str[8:12]
    m = m.sort_values("ts")
    rows = []
    for dt, d in m.groupby("dt", sort=True):
        f = _day_features(d)
        if f:
            o9 = d.loc[d["hm"] == "0900", "open"]
            f["open_0900"] = float(o9.iloc[0]) if len(o9) else np.nan
            rows.append({"symbol": symbol, "dt": dt, **f, **_morning(d)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["prev_close_1530"] = df["p_t0"].shift(1)
    df["nxt_dt"] = df["dt"].shift(-1)
    df["nxt_open"] = df["open_0900"].shift(-1)
    for c in [c for c in df.columns if c.startswith("m_")]:
        df["nx_" + c[2:]] = df[c].shift(-1)
    df["ret_1518"] = df["last_1518"] / df["prev_close_1530"] - 1
    df["ret_final"] = df["p_t0"] / df["prev_close_1530"] - 1
    df["r_on_t1"] = df["nxt_open"] / df["p_t1"] - 1      # T-1 진입 → 다음 날 시가
    df["r_on_t0"] = df["nxt_open"] / df["p_t0"] - 1      # T0 진입 → 다음 날 시가
    return df


def _safe_build(symbol: str) -> pd.DataFrame:
    try:
        return build_symbol(symbol)
    except FileNotFoundError:
        return pd.DataFrame()


def build_all(symbols: list[str], workers: int = 6) -> pd.DataFrame:
    """종목별 특징을 병렬로 계산해 합치고, 15:18 누적 거래대금 순위를 매긴다."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_safe_build, symbols, chunksize=8))
    df = pd.concat([x for x in out if not x.empty], ignore_index=True)
    ok = df["vol_1518"] > 0
    df["rank_1518"] = df["value_1518"].where(ok).groupby(df["dt"]).rank(ascending=False, method="first")
    df["rank_final"] = df["value_regular"].where(df["p_t0"].notna()).groupby(df["dt"]).rank(
        ascending=False, method="first")
    return df
