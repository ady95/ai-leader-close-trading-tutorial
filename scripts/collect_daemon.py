"""상시 수집 데몬 — 하루 일정에 따라 실시간·장중·장마감 후 수집을 실행한다.

  07:50~20:05  WebSocket 실시간 체결·호가 (관심종목: 전일 거래대금 상위 + 장중 갱신)
  06:00~21:00  DART 공시 목록 폴링 (60초) → 최초 수신 시각 기록
  09:00~15:40  장중 투자자 잠정 수급 폴링 (10분, 관심종목)
  10:10        오전 틱 수집 (관심종목 상위 30, 08:30 이후)
  20:10        장마감 후 수집: 틱(14:50 이후)·시간별 프로그램매매·일별 수급·테마
  20:40        증분 백필: 종목목록 → 일봉 → 분봉 후보 → 분봉

사용법
  python -m scripts.collect_daemon            # 상시 실행
  python -m scripts.collect_daemon --once eod # 특정 작업만 즉시 실행 (eod|morning|backfill|dart|investor)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime, time as dtime, timedelta

import pandas as pd

from collectors import market
from collectors.dart import DartPoller
from collectors.kiwoom_rest import KiwoomError, KiwoomREST
from collectors.realtime import RealtimeCollector
from data.store import JsonlGzWriter, path, read_parquet, upsert_parquet

log = logging.getLogger("daemon")

WATCH_TOP = 100
WATCH_MAX = 200
WATCH_REFRESH = [dtime(9, 10), dtime(10, 0), dtime(11, 0), dtime(13, 0), dtime(14, 0), dtime(14, 50), dtime(15, 10)]


def is_weekday(d: datetime) -> bool:
    return d.weekday() < 5


def common_symbols() -> set[str]:
    files = sorted(path("universe").glob("stocks_*.parquet"))
    if not files:
        return set()
    u = pd.read_parquet(files[-1])
    return set(u.loc[u["is_common"], "code"])


class Daemon:
    def __init__(self):
        self.api = KiwoomREST("real", min_interval=0.3)
        self.rt = RealtimeCollector(self.api)
        self.dart = DartPoller()
        self.done: set[str] = set()
        self.common = common_symbols()

    # ------------------------------------------------------------ watchlist
    def ranked_symbols(self, n: int) -> list[str]:
        df = market.trade_value_rank(self.api, pages=2)
        syms = [s for s in df["symbol"] if not self.common or s in self.common]
        return syms[:n]

    def save_watchlist(self) -> None:
        day = datetime.now().strftime("%Y%m%d")
        p = path("realtime", day, "watchlist.json")
        p.write_text(json.dumps(sorted(self.rt.registered | self.rt.pending)), encoding="utf-8")

    def watchlist(self) -> list[str]:
        return sorted(self.rt.registered | self.rt.pending)

    async def refresh_watchlist(self, n: int) -> None:
        try:
            syms = await asyncio.to_thread(self.ranked_symbols, n)
        except KiwoomError as e:
            log.warning("관심종목 갱신 실패: %s", e)
            return
        room = WATCH_MAX - len(self.watchlist())
        new = [s for s in syms if s not in self.watchlist()][: max(room, 0)]
        self.rt.add_symbols(new)
        self.save_watchlist()
        log.info("관심종목 +%d (총 %d)", len(new), len(self.watchlist()))

    # ----------------------------------------------------------------- jobs
    def job_investor(self) -> None:
        day = datetime.now().strftime("%Y%m%d")
        w = JsonlGzWriter(path("realtime", day, "investor_intraday.jsonl.gz"))
        n = 0
        for s in self.watchlist():
            try:
                df = market.investor_intraday(self.api, s)
            except KiwoomError as e:
                log.warning("%s 장중수급 실패: %s", s, e)
                continue
            for rec in df.to_dict("records"):
                w.write(rec)
                n += 1
        w.close()
        log.info("장중 잠정 수급 %d종목 %d행", len(self.watchlist()), n)

    def job_ticks(self, symbols: list[str], since: str, tag: str) -> None:
        day = datetime.now().strftime("%Y%m%d")
        for s in symbols:
            try:
                df = market.ticks(self.api, s, until_ts=day + since)
            except KiwoomError as e:
                log.warning("%s 틱 실패: %s", s, e)
                continue
            if not df.empty:
                df.to_parquet(path("ticks", day, f"{s}_{tag}.parquet"), index=False)
        log.info("틱(%s) %d종목 저장", tag, len(symbols))

    def job_morning(self) -> None:
        self.job_ticks(self.watchlist()[:30], "083000", "am")

    def job_eod(self) -> None:
        day = datetime.now().strftime("%Y%m%d")
        top = self.ranked_symbols(WATCH_TOP)
        pd.DataFrame({"symbol": top, "rank": range(1, len(top) + 1)}).to_parquet(
            path("realtime", day, "eod_top_value.parquet"), index=False)
        self.job_ticks(top[:50], "145000", "pm")
        for s in top:
            try:
                pi = market.program_intraday(self.api, s, day)
                if not pi.empty:
                    pi.to_parquet(path("program_intraday", day, f"{s}.parquet"), index=False)
                inv = market.investor_daily(self.api, s, day)
                if not inv.empty:
                    upsert_parquet(path("investor_daily", f"{s}.parquet"), inv, ["dt"])
            except KiwoomError as e:
                log.warning("%s 장마감 수집 실패: %s", s, e)
        groups = market.theme_groups(self.api)
        groups.to_parquet(path("themes", f"{day}_groups.parquet"), index=False)
        members = []
        for code in groups["thema_grp_cd"]:
            try:
                members.append(market.theme_members(self.api, code))
            except KiwoomError as e:
                log.warning("테마 %s 실패: %s", code, e)
        if members:
            pd.concat(members, ignore_index=True).to_parquet(path("themes", f"{day}_members.parquet"), index=False)
        log.info("장마감 수집 완료: 상위 %d종목, 테마 %d개", len(top), len(groups))

    def job_backfill(self) -> None:
        from scripts.backfill import backfill_running
        if backfill_running():
            log.warning("백필이 이미 진행 중이라 증분 백필을 건너뜁니다")
            return
        for step in ("universe", "daily", "candidates", "minute"):
            log.info("증분 백필 %s", step)
            subprocess.run([sys.executable, "-m", "scripts.backfill", step], check=False)
        self.common = common_symbols()

    def job_dart(self) -> None:
        n = self.dart.poll()
        if n:
            log.info("DART 새 공시 %d건", n)

    # ------------------------------------------------------------ scheduler
    async def once(self, key: str, fn, *a) -> None:
        if key in self.done:
            return
        self.done.add(key)
        try:
            await asyncio.to_thread(fn, *a)
        except Exception as e:  # noqa: BLE001
            log.exception("%s 실패: %s", key, e)

    async def loop(self) -> None:
        ws_task: asyncio.Task | None = None
        last_dart = last_inv = datetime.min
        while True:
            now = datetime.now()
            day = now.strftime("%Y%m%d")
            t = now.time()
            wd = is_weekday(now)

            if wd and dtime(7, 50) <= t < dtime(20, 5) and (ws_task is None or ws_task.done()):
                if not self.rt.registered and not self.rt.pending:
                    await self.refresh_watchlist(WATCH_TOP)
                until = now.replace(hour=20, minute=5, second=0, microsecond=0)
                ws_task = asyncio.create_task(self.rt.run(until))

            for rt_ in WATCH_REFRESH:
                if wd and rt_ <= t < (datetime.combine(now.date(), rt_) + timedelta(minutes=5)).time():
                    key = f"{day}-watch-{rt_}"
                    if key not in self.done:
                        self.done.add(key)
                        await self.refresh_watchlist(WATCH_TOP)

            if dtime(6, 0) <= t < dtime(21, 0) and now - last_dart > timedelta(seconds=60):
                last_dart = now
                asyncio.create_task(asyncio.to_thread(self._safe, self.job_dart))

            if wd and dtime(9, 0) <= t < dtime(15, 40) and now - last_inv > timedelta(minutes=10):
                last_inv = now
                asyncio.create_task(asyncio.to_thread(self._safe, self.job_investor))

            if wd and dtime(10, 10) <= t < dtime(11, 0):
                asyncio.create_task(self.once(f"{day}-morning", self.job_morning))
            if wd and dtime(20, 10) <= t < dtime(20, 40):
                asyncio.create_task(self.once(f"{day}-eod", self.job_eod))
            if wd and dtime(20, 40) <= t < dtime(23, 59):
                asyncio.create_task(self.once(f"{day}-backfill", self.job_backfill))

            if t < dtime(0, 1):  # 날짜가 바뀌면 관심종목 초기화
                self.rt.registered, self.rt.pending = set(), set()
            await asyncio.sleep(15)

    @staticmethod
    def _safe(fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            log.exception("작업 실패: %s", e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", choices=["eod", "morning", "backfill", "dart", "investor"])
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(path("logs", "daemon.log"), encoding="utf-8")])
    d = Daemon()
    if args.once:
        if args.once in ("morning", "investor"):
            d.rt.add_symbols(d.ranked_symbols(WATCH_TOP))
        getattr(d, f"job_{args.once}")()
        return
    asyncio.run(d.loop())


if __name__ == "__main__":
    main()
