"""과거 데이터 일괄 수집 (최초 1회 + 매일 증분).

사용법
  python -m scripts.backfill universe            # 전 종목 목록 스냅샷
  python -m scripts.backfill daily --start 20230101
  python -m scripts.backfill candidates --top 100 --days 260
  python -m scripts.backfill minute               # 후보 종목 1분봉 (증분)
  python -m scripts.backfill all                  # 위 4단계를 순서대로

분봉은 키움이 최근 1년치만 제공하므로 가장 먼저, 그리고 매일 실행해야 한다.
장중(08:00~20:00)에는 실시간 수집기와 호출 한도를 나누도록 호출 간격을 늘린다.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

from collectors import market
from collectors.kiwoom_rest import KiwoomError, KiwoomREST
from data.store import path, read_parquet, upsert_parquet

log = logging.getLogger("backfill")


def market_hours() -> bool:
    now = datetime.now()
    return now.weekday() < 5 and 8 <= now.hour < 20


def adaptive_interval(api: KiwoomREST) -> None:
    api.min_interval = 0.6 if market_hours() else 0.25
    LOCK.touch()  # 진행 중 표시 (수집 데몬이 동시 실행을 피하는 데 사용)


LOCK = path("logs", "backfill.lock")


def backfill_running(max_idle_min: int = 10) -> bool:
    """다른 백필 프로세스가 최근 max_idle_min 분 안에 진행 표시를 남겼는가."""
    return LOCK.exists() and (time.time() - LOCK.stat().st_mtime) < max_idle_min * 60


def latest_universe() -> pd.DataFrame:
    files = sorted(path("universe").glob("stocks_*.parquet"))
    if not files:
        raise SystemExit("종목 목록이 없습니다. 먼저 `universe` 를 실행하세요.")
    return pd.read_parquet(files[-1])


def cmd_universe(api: KiwoomREST, args) -> None:
    df = market.stock_list(api)
    p = path("universe", f"stocks_{market.today()}.parquet")
    df.to_parquet(p, index=False)
    log.info("종목 %d개 (연구 대상 보통주 %d개) → %s", len(df), int(df["is_common"].sum()), p)


def cmd_daily(api: KiwoomREST, args) -> None:
    uni = latest_universe()
    syms = uni.loc[uni["is_common"], "code"].tolist()
    base = market.today()
    for i, s in enumerate(syms, 1):
        adaptive_interval(api)
        p = path("daily", f"{s}.parquet")
        old = read_parquet(p)
        start = args.start if old.empty else max(old["dt"].max(), args.start)
        try:
            df = market.daily_bars(api, s, base, start)
        except KiwoomError as e:
            log.warning("%s 일봉 실패: %s", s, e)
            continue
        n = upsert_parquet(p, df, ["dt"]) if not df.empty else 0
        if i % 100 == 0 or i == len(syms):
            log.info("일봉 %d/%d (%s: %d행)", i, len(syms), s, n)


def cmd_candidates(api: KiwoomREST, args) -> None:
    """최근 N거래일 동안 하루라도 거래대금 상위 top 에 든 종목 = 분봉 수집 후보."""
    frames = []
    for p in path("daily").glob("*.parquet"):
        df = pd.read_parquet(p, columns=["symbol", "dt", "value_mil", "close"])
        frames.append(df)
    allday = pd.concat(frames, ignore_index=True)
    days = sorted(allday["dt"].unique())[-args.days:]
    recent = allday[allday["dt"].isin(days)].copy()
    recent["rank"] = recent.groupby("dt")["value_mil"].rank(ascending=False, method="first")
    top = recent[recent["rank"] <= args.top]
    cand = (top.groupby("symbol").agg(top_days=("dt", "nunique"), last_top=("dt", "max"))
            .reset_index().sort_values(["top_days", "last_top"], ascending=False))
    p = path("universe", "minute_candidates.parquet")
    old = read_parquet(p)
    if not old.empty:  # 한 번 후보가 된 종목은 계속 수집한다 (생존자 편향 방지)
        cand = pd.concat([cand, old[~old["symbol"].isin(cand["symbol"])]], ignore_index=True)
    cand.to_parquet(p, index=False)
    log.info("분봉 후보 %d종목 (최근 %d거래일, 일별 거래대금 상위 %d)", len(cand), len(days), args.top)


def cmd_minute(api: KiwoomREST, args) -> None:
    cand = read_parquet(path("universe", "minute_candidates.parquet"))
    if cand.empty:
        raise SystemExit("후보가 없습니다. 먼저 `candidates` 를 실행하세요.")
    syms = cand["symbol"].tolist()
    if args.limit:
        syms = syms[: args.limit]
    t0 = time.time()
    for i, s in enumerate(syms, 1):
        adaptive_interval(api)
        p = path("minute", f"{s}.parquet")
        old = read_parquet(p)
        since = None if old.empty else str(old["ts"].max())
        try:
            df = market.minute_bars(api, s, since=since)
        except KiwoomError as e:
            log.warning("%s 분봉 실패: %s", s, e)
            continue
        n = upsert_parquet(p, df, ["ts"]) if not df.empty else len(old)
        oldest = df["ts"].min() if not df.empty else "-"
        log.info("분봉 %d/%d %s +%d행 (누적 %d, 최고(古) %s) %.0f분 경과",
                 i, len(syms), s, len(df), n, oldest, (time.time() - t0) / 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["universe", "daily", "candidates", "minute", "all"])
    ap.add_argument("--start", default="20230101")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--days", type=int, default=260)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(path("logs", "backfill.log"), encoding="utf-8")])
    if backfill_running():
        log.warning("다른 백필이 진행 중입니다 (%s). 종료합니다.", LOCK)
        return
    api = KiwoomREST("real")
    steps = ["universe", "daily", "candidates", "minute"] if args.cmd == "all" else [args.cmd]
    for step in steps:
        log.info("=== %s 시작", step)
        LOCK.touch()
        try:
            globals()[f"cmd_{step}"](api, args)
        finally:
            LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
