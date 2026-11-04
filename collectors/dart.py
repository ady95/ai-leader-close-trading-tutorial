"""DART 공시 목록 폴링 — 공시마다 '최초 수신 시각(first_seen)'을 기록한다.

DART list.json 은 접수일(rcept_dt)만 주고 시각을 주지 않는다. 장중 신호(예: 15:18)에서
'그 시각에 알 수 있었던 공시'를 가리려면 폴링하며 처음 본 시각을 직접 남겨야 한다.
과거 기간(폴링 이전)은 당일 공시를 장 마감 후에 알려진 것으로 보수적으로 처리한다.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import requests

from config.settings import dart_api_key
from data.store import now_iso, path

log = logging.getLogger(__name__)
URL = "https://opendart.fss.or.kr/api/list.json"


class DartPoller:
    def __init__(self):
        self.key = dart_api_key()
        self.session = requests.Session()
        self.day = ""
        self.seen: set[str] = set()

    def _file(self, day: str):
        return path("dart", f"{day}.jsonl")

    def _roll(self, day: str) -> None:
        if day == self.day:
            return
        self.day = day
        self.seen = set()
        f = self._file(day)
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    self.seen.add(json.loads(line)["rcept_no"])
                except (ValueError, KeyError):
                    pass

    def poll(self) -> int:
        """오늘 공시 목록을 훑어 새 공시를 기록하고 새 건수를 돌려준다."""
        day = datetime.now().strftime("%Y%m%d")
        self._roll(day)
        new_rows = []
        page = 1
        while True:
            r = self.session.get(URL, params={"crtfc_key": self.key, "bgn_de": day, "end_de": day,
                                              "page_no": page, "page_count": 100}, timeout=20)
            j = r.json()
            status = j.get("status")
            if status == "013":  # 조회 데이터 없음
                break
            if status != "000":
                log.warning("DART 오류 %s %s", status, j.get("message"))
                break
            seen_at = now_iso()
            fresh = 0
            for x in j.get("list", []):
                if x["rcept_no"] not in self.seen:
                    self.seen.add(x["rcept_no"])
                    x["report_nm"] = x.get("report_nm", "").strip()
                    x["first_seen"] = seen_at
                    new_rows.append(x)
                    fresh += 1
            # 최신 공시가 앞에 오므로, 새 공시가 없는 페이지를 만나면 더 볼 필요가 없다.
            if fresh == 0 or page >= int(j.get("total_page", 1)):
                break
            page += 1
        if new_rows:
            with self._file(day).open("a", encoding="utf-8") as f:
                for x in new_rows:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
        return len(new_rows)


def history(day: str, session: requests.Session | None = None) -> list[dict]:
    """과거 하루치 공시 목록 전체 (폴링 이전 기간용). 시각 정보는 없다 — 접수일 장 마감 후로 간주한다."""
    s = session or requests.Session()
    key, rows, page = dart_api_key(), [], 1
    while True:
        j = s.get(URL, params={"crtfc_key": key, "bgn_de": day, "end_de": day,
                               "page_no": page, "page_count": 100}, timeout=30).json()
        if j.get("status") == "013":
            break
        if j.get("status") != "000":
            raise RuntimeError(f"DART 오류 {j.get('status')} {j.get('message')}")
        for x in j.get("list", []):
            x["report_nm"] = x.get("report_nm", "").strip()
            rows.append(x)
        if page >= int(j.get("total_page", 1)):
            break
        page += 1
    return rows
