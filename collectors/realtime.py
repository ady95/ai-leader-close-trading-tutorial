"""키움 WebSocket 실시간 수집 — 주식체결(0B)·주식호가잔량(0D) 원본 저장.

- 메시지 원본(values)을 그대로 gzip JSONL 로 남기고, 수신 시각(rt)을 붙인다.
- 호가(0D)는 양이 많으므로 핵심 구간(장 시작·종가·장후 마감 전)에는 전부,
  그 밖에는 종목당 1초에 1건만 남긴다.
- 디스크 여유가 MIN_FREE_GB 아래로 떨어지면 호가 저장을 멈춘다(체결은 유지).
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from datetime import datetime, time as dtime

import websockets

from collectors.kiwoom_rest import KiwoomREST
from data.store import JsonlGzWriter, data_dir, now_iso, path

log = logging.getLogger(__name__)

FULL_ORDERBOOK_WINDOWS = [
    (dtime(8, 0), dtime(9, 15)),    # 장전·시가 단일가·S0(09:05) 청산 구간
    (dtime(15, 0), dtime(16, 0)),   # 15:18~15:20 진입·종가 단일가·장후 시작
    (dtime(19, 20), dtime(20, 0)),  # 장후 마감 전(T3)
]
MIN_FREE_GB = 10
REG_CHUNK = 50


def in_full_window(now: datetime) -> bool:
    t = now.time()
    return any(a <= t < b for a, b in FULL_ORDERBOOK_WINDOWS)


class RealtimeCollector:
    def __init__(self, api: KiwoomREST):
        self.api = api
        self.registered: set[str] = set()
        self.pending: set[str] = set()
        self.day = ""
        self.writers: dict[str, JsonlGzWriter] = {}
        self._last_ob: dict[str, float] = {}
        self.stats = {"0B": 0, "0D": 0, "0D_skip": 0}
        self._ws = None

    # ------------------------------------------------------------- storage
    def _writer(self, kind: str) -> JsonlGzWriter:
        day = datetime.now().strftime("%Y%m%d")
        if day != self.day:
            for w in self.writers.values():
                w.close()
            self.writers = {}
            self.day = day
        if kind not in self.writers:
            p = path("realtime", day, f"{kind}.jsonl.gz")
            if p.exists():   # 재시작: 이전 프로세스가 끝맺지 못한 gzip 뒤에 이어 쓰면 읽을 수 없게 되므로 새 파일
                p = path("realtime", day, f"{kind}_{datetime.now():%H%M%S}.jsonl.gz")
            self.writers[kind] = JsonlGzWriter(p)
        return self.writers[kind]

    def _disk_ok(self) -> bool:
        return shutil.disk_usage(data_dir()).free / 1e9 > MIN_FREE_GB

    def handle(self, msg: dict) -> None:
        now = datetime.now()
        rt = now_iso()
        for d in msg.get("data", []):
            typ, item = d.get("type"), d.get("item")
            rec = {"rt": rt, "type": typ, "item": item, "values": d.get("values")}
            if typ == "0D":
                if not self._disk_ok():
                    self.stats["0D_skip"] += 1
                    continue
                if not in_full_window(now):
                    t = time.monotonic()
                    if t - self._last_ob.get(item, 0) < 1.0:
                        self.stats["0D_skip"] += 1
                        continue
                    self._last_ob[item] = t
                self._writer("orderbook").write(rec)
                self.stats["0D"] += 1
            elif typ == "0B":
                self._writer("trades").write(rec)
                self.stats["0B"] += 1
            else:
                self._writer("other").write(rec)

    # ----------------------------------------------------------- watchlist
    def add_symbols(self, symbols) -> None:
        new = set(symbols) - self.registered
        if new:
            self.pending |= new

    async def _flush_registration(self, ws) -> None:
        while self.pending:
            chunk = sorted(self.pending)[:REG_CHUNK]
            await ws.send(json.dumps({"trnm": "REG", "grp_no": str(1 + len(self.registered) // REG_CHUNK),
                                      "refresh": "1",
                                      "data": [{"item": chunk, "type": ["0B", "0D"]}]}))
            self.registered |= set(chunk)
            self.pending -= set(chunk)
            log.info("실시간 등록 +%d (총 %d)", len(chunk), len(self.registered))
            await asyncio.sleep(0.5)

    # ---------------------------------------------------------------- loop
    async def run(self, until: datetime) -> None:
        """until 시각까지 접속을 유지한다. 끊기면 재접속하고 전체 종목을 다시 등록한다."""
        backoff = 2
        while datetime.now() < until:
            try:
                async with websockets.connect(self.api.cred.ws_host, ping_interval=None,
                                              max_size=2 ** 22) as ws:
                    await ws.send(json.dumps({"trnm": "LOGIN", "token": await asyncio.to_thread(self.api.token)}))
                    login = json.loads(await asyncio.wait_for(ws.recv(), 10))
                    if login.get("return_code") != 0:
                        raise RuntimeError(f"WS 로그인 실패: {login}")
                    log.info("WS 로그인 성공")
                    self.pending |= self.registered
                    self.registered = set()
                    backoff = 2
                    last_stat = time.monotonic()
                    while datetime.now() < until:
                        await self._flush_registration(ws)
                        try:
                            raw = await asyncio.wait_for(ws.recv(), 5)
                        except asyncio.TimeoutError:
                            continue
                        msg = json.loads(raw)
                        trnm = msg.get("trnm")
                        if trnm == "PING":
                            await ws.send(raw)
                        elif trnm == "REAL":
                            self.handle(msg)
                        elif trnm in ("REG", "REMOVE"):
                            if msg.get("return_code") != 0:
                                log.warning("WS %s 응답: %s", trnm, msg)
                        else:
                            log.info("WS 기타 메시지: %s", raw[:200])
                        if time.monotonic() - last_stat > 300:
                            for w in self.writers.values():
                                w.flush()
                            log.info("WS 통계 %s", self.stats)
                            last_stat = time.monotonic()
            except Exception as e:  # noqa: BLE001 — 수집기는 죽지 않고 재접속한다
                log.warning("WS 끊김: %s (%ds 후 재접속)", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
        for w in self.writers.values():
            w.close()
        self.writers = {}
        log.info("WS 종료 %s", self.stats)
