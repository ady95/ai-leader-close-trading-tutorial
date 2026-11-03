"""수집 데이터 저장 유틸.

저장 구조 (DATA_DIR 기준)
  universe/stocks_YYYYMMDD.parquet        전 종목 목록 스냅샷
  daily/{symbol}.parquet                  일봉 (KRX 정규장 기준, 수정주가)
  minute/{symbol}.parquet                 1분봉 (KRX, 장전·장후 포함)
  ticks/YYYYMMDD/{symbol}_{window}.parquet  틱 (구간별)
  investor_daily/{symbol}.parquet         일별 투자자 수급 (확정치)
  program_intraday/YYYYMMDD/{symbol}.parquet 시간별 프로그램매매
  themes/YYYYMMDD_*.parquet               테마 그룹·구성 종목
  realtime/YYYYMMDD/*.jsonl.gz            실시간 체결·호가·잠정 수급 (원본)
  dart/YYYYMMDD.jsonl                     공시 최초 수신 기록

모든 행에는 수집 시각(received_at)을 남긴다. 과거 시점 재현(as_of) 때 기준이 된다.
"""
from __future__ import annotations

import gzip
import json
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import data_dir


def path(*parts: str) -> Path:
    p = data_dir().joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def upsert_parquet(p: Path, df: pd.DataFrame, key: list[str], sort: list[str] | None = None) -> int:
    """기존 파일과 합쳐 key 기준으로 중복 제거(새 값 우선) 후 저장. 저장된 행 수를 돌려준다."""
    if df.empty and not p.exists():
        return 0
    if p.exists():
        old = pd.read_parquet(p)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=key, keep="last")
    df = df.sort_values(sort or key).reset_index(drop=True)
    for c in df.columns:  # 문자열·숫자가 섞인 열은 문자열로 통일 (parquet 저장 오류 방지)
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: None if x is None or (isinstance(x, float) and x != x) else str(x))
    tmp = p.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(p)
    return len(df)


def read_parquet(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


class JsonlGzWriter:
    """하루 단위 gzip JSONL 기록기 (스레드 안전)."""

    def __init__(self, p: Path):
        p.parent.mkdir(parents=True, exist_ok=True)
        self.p = p
        self._f = gzip.open(p, "at", encoding="utf-8")
        self._lock = threading.Lock()
        self.count = 0

    def write(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._f.write(line + "\n")
            self.count += 1

    def flush(self) -> None:
        with self._lock:
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            self._f.close()
