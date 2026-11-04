"""종가배팅 통합 실행기 (16-1) — 15:18 매수, 다음 날 09:00 매도.

사용법
  python -m scripts.trade_daily buy      # 15:18:00 에 실행 (15:18 이전이면 기다린다)
  python -m scripts.trade_daily sell     # 다음 날 09:00:05 에 실행
  python -m scripts.trade_daily report   # 저널·체결로 결과 요약
  python -m scripts.trade_daily sell --at 20261105090000   # 날짜를 지정해 예약 실행

시세는 실전 키로 조회만 하고, 주문은 TRADING_MODE(.env, 기본 paper)의 키로 보낸다.
선택 규칙은 배관 검증용 기준선 E0@15:18 이다: 15:18 거래대금 상위(통합) 중 보통주, 등락률 5~29.5%, 상위 5.
연구에서 고른 모델로 바꾸려면 select() 만 바꾼다. 주문·위험·동기화는 그대로 쓴다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

from collectors import market
from collectors.kiwoom_rest import KiwoomREST
from config.settings import trading_mode
from data.store import path
from execution.orders import Intent, OrderManager
from risk.limits import RiskLimits

log = logging.getLogger("trade")
BUDGET_PER_NAME = 500_000
TOP_K = 5


def wait_until(hhmmss: str) -> None:
    while datetime.now().strftime("%H%M%S") < hhmmss:
        time.sleep(0.5)


def universe_common() -> set[str]:
    files = sorted(path("universe").glob("stocks_*.parquet"))
    u = pd.read_parquet(files[-1])
    return set(u.loc[u["is_common"] & ~u["is_halted"], "code"])


def select(rank: pd.DataFrame, common: set[str]) -> pd.DataFrame:
    r = rank[rank["symbol"].isin(common)].copy()
    r["flu_rt"] = pd.to_numeric(r["flu_rt"], errors="coerce")
    r = r[(r["flu_rt"] >= 5) & (r["flu_rt"] < 29.5)]
    return r.sort_values("now_rank", key=pd.to_numeric).head(TOP_K)


def journal_path(day: str):
    return path("trading", f"journal_{day}.jsonl")


def cmd_buy() -> None:
    real, mode = KiwoomREST("real"), trading_mode()
    om = OrderManager(KiwoomREST(mode), journal_path(datetime.now().strftime("%Y%m%d")), RiskLimits())
    common = universe_common()
    real.token(), om.api.token()                      # 토큰은 미리 확인 (16-3)
    wait_until("151800")
    t0 = datetime.now()
    rank = market.trade_value_rank(real)
    picks = select(rank, common)
    log.info("15:18 순위 조회 %.2fs, 후보 %d: %s", (datetime.now() - t0).total_seconds(), len(picks),
             picks.loc[:, ["symbol", "stk_nm", "flu_rt", "sel_bid"]].to_dict("records"))
    day = t0.strftime("%Y%m%d")
    for r in picks.itertuples():
        ask = abs(int(str(r.sel_bid).replace("+", "").replace("-", "")))
        qty = BUDGET_PER_NAME // ask
        if qty < 1:
            log.info("%s 1주 가격 %s 가 예산보다 큼 — 건너뜀", r.symbol, ask)
            continue
        it = om.submit(Intent(f"{day}-{r.symbol}-T-1-buy", r.symbol, "buy", int(qty), ask), ask)
        log.info("매수 %s %s주 @%s → %s %s %s", r.symbol, qty, ask, it.status, it.ord_no, it.msg)
    wait_until("151940")
    for o in om.open_orders():                        # 15:20 전에 미체결은 취소 (T-1 정의)
        if o.get("io_tp_nm", "").startswith("+매수"):
            om.cancel(o["ord_no"], o["stk_cd"].lstrip("A"))
            log.info("미체결 취소 %s %s 잔량 %s", o["stk_cd"], o["ord_no"], o.get("oso_qty"))
    time.sleep(2)
    log.info("15:19:40 잔고 %s", om.positions())


def cmd_sell() -> None:
    mode = trading_mode()
    y = (datetime.now() - timedelta(days=1))
    days = [(y - timedelta(days=k)).strftime("%Y%m%d") for k in range(0, 5)]   # 직전 거래일 저널 찾기
    jp = next((journal_path(d) for d in days if journal_path(d).exists()), None)
    if jp is None:
        log.info("매도할 저널 없음")
        return
    om = OrderManager(KiwoomREST(mode), journal_path(datetime.now().strftime("%Y%m%d")), RiskLimits())
    om.api.token()
    bought = {i.symbol for i in OrderManager(om.api, jp).journal.latest().values()
              if i.side == "buy" and i.status == "accepted"}
    wait_until("090005")      # 모의투자는 장 시작 전 주문을 받지 않는다 (16-3)
    pos = om.positions()
    today = datetime.now().strftime("%Y%m%d")
    for s, p in pos.items():
        if s in bought and p["qty"] > 0:
            it = om.submit(Intent(f"{today}-{s}-S1-sell", s, "sell", p["qty"], None), p["price"])
            log.info("매도 %s %s주 → %s %s", s, p["qty"], it.status, it.ord_no)
    time.sleep(3)
    log.info("09:00 매도 후 잔고 %s", om.positions())


def cmd_report() -> None:
    mode = trading_mode()
    om = OrderManager(KiwoomREST(mode), journal_path(datetime.now().strftime("%Y%m%d")))
    for f in om.fills():
        log.info("체결 %s", json.dumps({k: f.get(k) for k in ("ord_no", "stk_cd", "io_tp_nm", "cntr_qty", "cntr_pric",
                                                          "ord_tm", "tdy_trde_cmsn", "tdy_trde_tax")},
                                     ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["buy", "sell", "report"])
    ap.add_argument("--at", default="", help="YYYYMMDDHHMMSS 까지 기다렸다가 실행 (예약 실행용)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(path("logs", "trade.log"), encoding="utf-8")])
    log.info("모드 %s, 명령 %s, 예약 %s", trading_mode(), a.cmd, a.at or "-")
    if a.at:
        target = datetime.strptime(a.at, "%Y%m%d%H%M%S")
        while datetime.now() < target:
            time.sleep(5)
    globals()[f"cmd_{a.cmd}"]()


if __name__ == "__main__":
    main()
