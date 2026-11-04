"""다음 날 장중 청산 정책 S1~S6 시뮬레이터 (1분봉 근사). 명세는 실험 전에 고정했다(11장).

공통 가정
  - 진입가: 15:18~15:19 대표가격 + 반 호가 (s0_s1.entry_price)
  - 09:00 시가 단일가 체결은 스프레드 없음. 그 밖의 장중 매도는 반 호가 불리.
  - 한 봉 안에서 익절선과 손절선을 모두 건드리면 손절이 먼저라고 본다 (봉 내부 순서는 모름).
  - 손절·익절선을 갭으로 건너뛰면 그 봉의 시가에 판다.
  - 시간 손절: 별도 언급이 없으면 11:00 (10:59 봉 종가).

정책
  S1  09:00 시가 단일가 전량
  S2  고정 익절 +3% / 손절 -2%, 11:00 시간 청산
  S3  ATR 트레일링: 장중 최고가 - 0.5 × ATR14(정규장 일봉), 손절 초기값 진입가 - 0.5 × ATR, 11:00
  S4  VWAP 이탈: 09:05 이후 1분봉 종가 < 당일 누적 VWAP 이면 다음 봉 시가에 매도, 11:00
  S5  시간 청산: 09:10 / 09:30 / 10:00 (각 직전 봉 종가)
  S6  +2% 도달 시 절반 익절, 나머지는 장중 최고가 -1.5% 트레일링 (초기 손절 -2%), 15:19 시간 청산
"""
from __future__ import annotations

import numpy as np

TIME_STOP = "1059"


def _half_tick(p: float) -> float:
    for lim, t in ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500)):
        if p < lim:
            return t / 2
    return 500.0


def _sell(p: float) -> float:
    return p - _half_tick(p)


def simulate(entry: float, atr: float, hm: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray,
             c: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """다음 날 정규장 1분봉(09:00~15:19)으로 정책별 평균 매도가를 계산한다."""
    out: dict[str, float] = {}
    if len(hm) == 0 or hm[0] != "0900":
        return out
    op = o[0]
    out["S1"] = op

    def until(t):
        return np.where(hm <= t)[0]

    # S5 시간 청산
    for t, name in (("0909", "S5_0910"), ("0929", "S5_0930"), ("0959", "S5_1000")):
        idx = until(t)
        out[name] = _sell(c[idx[-1]])

    ts = until(TIME_STOP)
    # S2 고정 익절·손절
    tp, sl = entry * 1.03, entry * 0.98
    out["S2"] = _bracket(entry, op, tp, sl, ts, o, h, l, c)

    # S3 ATR 트레일링
    if atr == atr and atr > 0:
        k = 0.5 * atr
        if op <= entry - k:
            out["S3"] = op
        else:
            peak, res = max(op, entry), None
            for i in ts:
                stop = max(entry - k, peak - k)
                if l[i] <= stop:
                    res = _sell(o[i] if o[i] < stop else stop)
                    break
                peak = max(peak, h[i])
            out["S3"] = res if res is not None else _sell(c[ts[-1]])

    # S4 VWAP 이탈
    tpx = (h + l + c) / 3
    cum_v = np.cumsum(v)
    vwap = np.divide(np.cumsum(tpx * v), cum_v, out=np.full(len(v), np.nan), where=cum_v > 0)
    res = None
    for i in ts:
        if hm[i] >= "0905" and c[i] < vwap[i] and i + 1 < len(hm):
            res = _sell(o[i + 1])
            break
    out["S4"] = res if res is not None else _sell(c[ts[-1]])

    # S6 분할 익절 + 잔여 트레일링
    out["S6"] = _scale_out(entry, op, hm, o, h, l, c)
    return out


def _bracket(entry, op, tp, sl, ts, o, h, l, c) -> float:
    if op <= sl or op >= tp:
        return op
    for i in ts:
        if l[i] <= sl:
            return _sell(o[i] if o[i] < sl else sl)
        if h[i] >= tp:
            return _sell(o[i] if o[i] > tp else tp)
    return _sell(c[ts[-1]])


def _scale_out(entry, op, hm, o, h, l, c) -> float:
    tp, sl0, trail = entry * 1.02, entry * 0.98, 0.015
    last = np.where(hm <= "1519")[0]
    if op <= sl0:
        return op
    half = None
    if op >= tp:
        half = op
    peak = op
    for i in last:
        stop = sl0 if half is None else max(sl0, peak * (1 - trail))
        if l[i] <= stop:
            px = _sell(o[i] if o[i] < stop else stop)
            return px if half is None else (half + px) / 2
        if half is None and h[i] >= tp:
            half = _sell(o[i] if o[i] > tp else tp)
        peak = max(peak, h[i])
    px = _sell(c[last[-1]])
    return px if half is None else (half + px) / 2
