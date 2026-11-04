"""키움 REST API 최소 클라이언트 (조회 전용 기능 중심).

특징
- 접근토큰을 파일에 캐시해 여러 프로세스가 같은 토큰을 재사용한다.
- 프로세스 내 호출 간격 제한(rate limit)과 재시도.
- 연속조회(cont-yn / next-key) 제너레이터.
- 키움 숫자 문자열("+1234", "-1234", "--1234") 파서.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Iterator

import requests

from config.settings import KiwoomCredential, data_dir, kiwoom_credential

log = logging.getLogger(__name__)

# 가격 필드: 키움은 등락 방향을 부호로 붙여 주므로 절댓값을 취한다.
PRICE_FIELDS = {
    "cur_prc", "open_pric", "high_pric", "low_pric", "sel_bid", "buy_bid",
    "pri_sel_bid_unit", "pri_buy_bid_unit",
}


def num(value: str | None, absolute: bool = False) -> float | int | None:
    """키움 숫자 문자열을 숫자로 바꾼다.

    "+1234" → 1234, "-1234" → -1234, "--1234" → -1234 (순매수 음수 표기), "" → None.
    absolute=True 이면 부호를 버린다 (가격 필드용).
    """
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if s == "":
        return None
    negative = s.startswith("-")
    body = s.lstrip("+-")
    if body == "":
        return None
    v = float(body) if "." in body else int(body)
    if absolute or not negative:
        return v
    return -v


def normalize_row(row: dict) -> dict:
    """응답 행의 숫자 문자열을 숫자로 변환한다. 숫자가 아닌 값은 그대로 둔다."""
    out = {}
    for k, v in row.items():
        if isinstance(v, str) and v.strip() == "":
            out[k] = None
            continue
        if isinstance(v, str) and k not in {"dt", "cntr_tm", "tm", "stk_cd", "stk_nm", "date"}:
            try:
                out[k] = num(v, absolute=k in PRICE_FIELDS)
                continue
            except ValueError:
                pass
        out[k] = v
    return out


class KiwoomError(RuntimeError):
    pass


class KiwoomREST:
    def __init__(self, mode: str = "real", min_interval: float = 0.25, timeout: float = 20.0):
        self.cred: KiwoomCredential = kiwoom_credential(mode)
        self.min_interval = min_interval
        self.timeout = timeout
        self.session = requests.Session()
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._token: str | None = None
        self._token_exp: datetime | None = None
        cache_dir = data_dir() / ".cache"
        cache_dir.mkdir(exist_ok=True)
        self._token_file = cache_dir / f"kiwoom_{mode}.token.json"

    # ---------------------------------------------------------------- token
    def _load_cached_token(self) -> bool:
        try:
            j = json.loads(self._token_file.read_text(encoding="utf-8"))
            exp = datetime.strptime(j["expires_dt"], "%Y%m%d%H%M%S")
            if exp - datetime.now() > timedelta(minutes=30):
                self._token, self._token_exp = j["token"], exp
                return True
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            pass
        return False

    def _issue_token(self) -> None:
        r = self.session.post(
            f"{self.cred.host}/oauth2/token",
            json={"grant_type": "client_credentials", "appkey": self.cred.app_key,
                  "secretkey": self.cred.app_secret},
            timeout=self.timeout,
        )
        j = r.json()
        if not j.get("token"):
            raise KiwoomError(f"토큰 발급 실패: {j.get('return_code')} {j.get('return_msg')}")
        self._token = j["token"]
        self._token_exp = datetime.strptime(j["expires_dt"], "%Y%m%d%H%M%S")
        self._token_file.write_text(json.dumps({"token": j["token"], "expires_dt": j["expires_dt"]}),
                                    encoding="utf-8")
        log.info("키움 %s 토큰 발급 (만료 %s)", self.cred.mode, j["expires_dt"])

    def token(self, force: bool = False) -> str:
        if not force and self._token and self._token_exp and self._token_exp - datetime.now() > timedelta(minutes=30):
            return self._token
        if force or not self._load_cached_token():
            self._issue_token()
        assert self._token
        return self._token

    # --------------------------------------------------------------- request
    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def request(self, path: str, api_id: str, body: dict, next_key: str | None = None,
                retries: int = 5) -> tuple[dict, str | None]:
        """단일 조회. (응답 JSON, 다음 연속조회 키 또는 None) 를 돌려준다."""
        delay = 1.0
        for attempt in range(retries):
            self._throttle()
            headers = {
                "authorization": f"Bearer {self.token()}",
                "api-id": api_id,
                "Content-Type": "application/json;charset=UTF-8",
            }
            if next_key:
                headers["cont-yn"] = "Y"
                headers["next-key"] = next_key
            try:
                r = self.session.post(f"{self.cred.host}{path}", headers=headers, json=body,
                                      timeout=self.timeout)
            except requests.RequestException as e:
                log.warning("%s 네트워크 오류 (%s), %.1fs 후 재시도", api_id, e, delay)
                time.sleep(delay); delay *= 2
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == retries - 1:                     # 마지막 시도면 기다리지 않고 바로 실패
                    raise KiwoomError(f"{api_id} HTTP {r.status_code} (재시도 없음)")
                log.warning("%s HTTP %s, %.1fs 후 재시도", api_id, r.status_code, delay)
                time.sleep(delay); delay *= 2
                continue
            try:
                j = r.json()
            except ValueError:
                raise KiwoomError(f"{api_id} JSON 아님: HTTP {r.status_code} {r.text[:200]}")
            rc = j.get("return_code")
            msg = str(j.get("return_msg", ""))
            if rc in (0, None):
                nk = r.headers.get("next-key") if r.headers.get("cont-yn") == "Y" else None
                return j, (nk or None)
            if "토큰" in msg or "token" in msg.lower() or r.status_code == 401:
                log.warning("%s 토큰 오류 → 재발급: %s", api_id, msg)
                self.token(force=True)
                continue
            if "허용" in msg or "초과" in msg or rc == 5:
                log.warning("%s 호출 제한(%s): %s, %.1fs 대기", api_id, rc, msg, delay)
                time.sleep(delay); delay *= 2
                continue
            raise KiwoomError(f"{api_id} 오류 rc={rc} msg={msg}")
        raise KiwoomError(f"{api_id} 재시도 {retries}회 실패")

    def paged(self, path: str, api_id: str, body: dict, list_key: str | None = None,
              stop: Callable[[list[dict]], bool] | None = None,
              max_pages: int | None = None) -> Iterator[list[dict]]:
        """연속조회 제너레이터. 페이지마다 행 리스트를 내준다.

        stop(rows) 가 True 를 돌려주면 그 페이지를 내준 뒤 멈춘다.
        """
        next_key = None
        pages = 0
        while True:
            j, next_key = self.request(path, api_id, body, next_key)
            rows = j.get(list_key) if list_key else next((v for v in j.values() if isinstance(v, list)), [])
            rows = rows or []
            pages += 1
            yield rows
            if not rows or not next_key:
                return
            if stop and stop(rows):
                return
            if max_pages and pages >= max_pages:
                return
