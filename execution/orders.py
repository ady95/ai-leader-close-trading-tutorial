"""주문 관리자 — 중복 주문 방지, 상태 대조, 체결 기준 포지션.

원칙
  1. 주문 의도(intent)를 저널에 먼저 쓰고 보낸다. 같은 client_id 는 두 번 보내지 않는다.
  2. 주문 요청은 재시도하지 않는다. 응답을 못 받으면 상태를 'unknown' 으로 두고
     미체결·체결 조회로 실제 접수 여부를 확인(reconcile)한다. 확인 전에는 다시 보내지 않는다.
  3. 포지션은 주문이 아니라 체결 조회 결과로만 만든다.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from collectors.kiwoom_rest import KiwoomError, KiwoomREST
from risk.limits import RiskLimits

log = logging.getLogger(__name__)
ORDR, ACNT = "/api/dostk/ordr", "/api/dostk/acnt"
TRDE_TP = {"limit": "0", "market": "3"}
CLOCK_TOLERANCE_S = 60


@dataclass
class Intent:
    client_id: str
    symbol: str
    side: str                    # buy | sell
    qty: int
    price: float | None          # None = 시장가
    venue: str = "KRX"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    status: str = "new"          # new → sent → accepted | rejected | unknown | blocked
    ord_no: str | None = None
    msg: str = ""


class Journal:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, it: Intent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")

    def latest(self) -> dict[str, Intent]:
        out: dict[str, Intent] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                out[d["client_id"]] = Intent(**d)
        return out


def _int(x) -> int:
    s = str(x or "0").replace("+", "").replace("-", "").lstrip("0")
    return int(s or 0)


class OrderManager:
    def __init__(self, api: KiwoomREST, journal: Path, limits: RiskLimits | None = None):
        self.api, self.journal, self.limits = api, Journal(journal), limits or RiskLimits()

    # ---------------------------------------------------------------- 조회
    def open_orders(self) -> list[dict]:
        j, _ = self.api.request(ACNT, "ka10075", {"all_stk_tp": "0", "trde_tp": "0", "stk_cd": "", "stex_tp": "0"})
        return j.get("oso") or []

    def fills(self) -> list[dict]:
        j, _ = self.api.request(ACNT, "ka10076", {"stk_cd": "", "qry_tp": "0", "sell_tp": "0", "ord_no": "", "stex_tp": "0"})
        return j.get("cntr") or []

    def positions(self) -> dict[str, dict]:
        """잔고 조회(kt00018) 기준 보유. 종목코드의 'A' 접두사와 0 채움을 정규화한다."""
        try:
            j, _ = self.api.request(ACNT, "kt00018", {"qry_tp": "1", "dmst_stex_tp": "KRX"})
        except KiwoomError as e:
            if "조회내역이 없습니다" in str(e):
                return {}
            raise
        out = {}
        for r in j.get("acnt_evlt_remn_indv_tot") or []:
            sym = str(r.get("stk_cd", "")).lstrip("A")
            out[sym] = {"qty": _int(r.get("rmnd_qty")), "avg": _int(r.get("pur_pric")), "price": _int(r.get("cur_prc"))}
        return out

    # ---------------------------------------------------------------- 주문
    def submit(self, it: Intent, ref_price: float) -> Intent:
        prev = self.journal.latest().get(it.client_id)
        if prev and prev.status not in ("new",):
            log.info("이미 처리된 주문 의도 %s (%s) — 다시 보내지 않음", it.client_id, prev.status)
            return prev
        held = {s: p["qty"] * p["price"] for s, p in self.positions().items()}
        today = datetime.now().strftime("%Y-%m-%d")
        n_today = sum(1 for x in self.journal.latest().values() if x.created_at.startswith(today) and x.status != "blocked")
        ok, why = self.limits.check(it.side, it.symbol, it.qty, it.price or ref_price, held, n_today, 0.0)
        if not ok:
            it.status, it.msg = "blocked", why
            self.journal.write(it)
            return it
        it.status = "sent"
        self.journal.write(it)                      # 보내기 전에 기록 (보낸 뒤 죽어도 흔적이 남는다)
        body = {"dmst_stex_tp": it.venue, "stk_cd": it.symbol, "ord_qty": str(it.qty),
                "ord_uv": "" if it.price is None else str(int(it.price)),
                "trde_tp": TRDE_TP["market" if it.price is None else "limit"], "cond_uv": ""}
        api_id = "kt10000" if it.side == "buy" else "kt10001"
        try:
            j, _ = self.api.request(ORDR, api_id, body, retries=1)   # 주문은 재시도하지 않는다
            it.status, it.ord_no, it.msg = "accepted", j.get("ord_no"), str(j.get("return_msg", "")).strip()
        except KiwoomError as e:
            if "rc=" in str(e):                     # 서버가 거부 사유를 돌려준 경우
                it.status, it.msg = "rejected", str(e)
            else:                                   # 응답을 못 받음 → 접수됐을 수도 있다
                it.status, it.msg = "unknown", str(e)
        self.journal.write(it)
        if it.status == "unknown":
            self.reconcile(it)
        return it

    def reconcile(self, it: Intent) -> Intent:
        """응답을 못 받은 주문이 실제로 접수됐는지 미체결·체결 조회로 확인한다."""
        side = "+매수" if it.side == "buy" else "-매도"
        known = {x.ord_no for x in self.journal.latest().values() if x.ord_no}     # 다른 의도의 주문은 제외
        # PC 시계와 서버 시계는 몇 초씩 다르다(실측 1~2초). 의도 생성 60초 전부터의 주문을 후보로 본다.
        since = (datetime.fromisoformat(it.created_at) - timedelta(seconds=CLOCK_TOLERANCE_S)).strftime("%H%M%S")
        cand = [r for r in self.open_orders() + self.fills()
                if r.get("stk_cd", "").lstrip("A") == it.symbol and r.get("io_tp_nm", "").startswith(side)
                and _int(r.get("ord_qty")) == it.qty and r.get("ord_tm", "000000") >= since
                and r.get("ord_no") not in known]
        if cand:
            it.status, it.ord_no, it.msg = "accepted", cand[0].get("ord_no"), "reconcile: 접수 확인"
        else:
            it.status, it.msg = "not_found", "reconcile: 접수 기록 없음 — 새 client_id 로만 다시 보낼 것"
        self.journal.write(it)
        return it

    def cancel(self, ord_no: str, symbol: str, venue: str = "KRX") -> dict:
        j, _ = self.api.request(ORDR, "kt10003", {"dmst_stex_tp": venue, "orig_ord_no": ord_no,
                                                  "stk_cd": symbol, "cncl_qty": "0"}, retries=1)
        return j
