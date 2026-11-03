"""환경설정 로더.

- 비밀정보는 .env 에서만 읽는다 (저장소에 커밋하지 않는다).
- .env 탐색 순서: 현재 폴더 → 저장소 루트 → 저장소 상위 폴더.
- 시세 수집은 실전 키(조회 전용)를 사용하고, 주문은 TRADING_MODE 를 따른다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

for _cand in (Path.cwd() / ".env", REPO_ROOT / ".env", REPO_ROOT.parent / ".env"):
    if _cand.exists():
        load_dotenv(_cand, override=False)
        break

KIWOOM_HOSTS = {
    "real": "https://api.kiwoom.com",
    "paper": "https://mockapi.kiwoom.com",
}
KIWOOM_WS_HOSTS = {
    "real": "wss://api.kiwoom.com:10000/api/dostk/websocket",
    "paper": "wss://mockapi.kiwoom.com:10000/api/dostk/websocket",
}


@dataclass(frozen=True)
class KiwoomCredential:
    mode: str
    app_key: str
    app_secret: str
    account_no: str

    @property
    def host(self) -> str:
        return KIWOOM_HOSTS[self.mode]

    @property
    def ws_host(self) -> str:
        return KIWOOM_WS_HOSTS[self.mode]


def kiwoom_credential(mode: str) -> KiwoomCredential:
    if mode not in KIWOOM_HOSTS:
        raise ValueError(f"mode must be real|paper, got {mode!r}")
    p = mode.upper()
    key = os.getenv(f"KIWOOM_{p}_APP_KEY", "")
    secret = os.getenv(f"KIWOOM_{p}_APP_SECRET", "")
    if not key or not secret:
        raise RuntimeError(f"KIWOOM_{p}_APP_KEY / KIWOOM_{p}_APP_SECRET 가 .env 에 없습니다")
    return KiwoomCredential(mode, key, secret, os.getenv(f"KIWOOM_{p}_ACCOUNT_NO", ""))


def trading_mode() -> str:
    return os.getenv("TRADING_MODE", "paper").strip().lower()


def dart_api_key() -> str:
    key = os.getenv("DART_API_KEY", "")
    if not key:
        raise RuntimeError("DART_API_KEY 가 .env 에 없습니다")
    return key


def data_dir() -> Path:
    d = Path(os.getenv("DATA_DIR") or REPO_ROOT / "data" / "lake")
    d.mkdir(parents=True, exist_ok=True)
    return d
