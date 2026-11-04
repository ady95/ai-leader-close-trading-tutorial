"""상태 점검 (16-4) — 수집 데몬·실시간 수신·주문 저널·보유·위험 스위치를 한 화면에.

사용법
  python -m scripts.monitor            # 한 번 출력
  python -m scripts.monitor --watch 60 # 60초마다

점검 항목마다 OK/경고를 판정한다. 경고가 있으면 종료 코드 1 (cron·작업 스케줄러 알림용).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta

from config.settings import trading_mode
from data.store import path


def _last_line(pattern: str, lines: list[str]) -> str | None:
    for line in reversed(lines):
        if re.search(pattern, line):
            return line
    return None


def _age_min(line: str | None) -> float | None:
    if not line:
        return None
    try:
        t = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return (datetime.now() - t).total_seconds() / 60


def check() -> list[tuple[str, str, str]]:
    now = datetime.now()
    hm = now.strftime("%H%M")
    weekday = now.weekday() < 5
    out: list[tuple[str, str, str]] = []
    log = path("logs", "daemon.log")
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[-3000:] if log.exists() else []

    ws = _last_line(r"WS 통계", lines)
    age = _age_min(ws)
    in_ws = weekday and "0750" <= hm < "2005"
    stat = re.search(r"\{.*\}", ws).group(0) if ws else "-"
    out.append(("실시간 수신", "경고" if in_ws and (age is None or age > 10) else "OK",
                f"마지막 통계 {age:.0f}분 전 {stat}" if age is not None else "기록 없음"))
    dart = _last_line(r"DART", lines)
    age = _age_min(dart)
    in_dart = "0600" <= hm < "2100"
    out.append(("DART 폴링", "경고" if in_dart and (age is None or age > 30) else "OK",
                f"마지막 기록 {age:.0f}분 전" if age is not None else "기록 없음"))
    inv = _last_line(r"장중 잠정 수급", lines)
    age = _age_min(inv)
    in_inv = weekday and "0910" <= hm < "1540"
    out.append(("장중 잠정 수급", "경고" if in_inv and (age is None or age > 20) else "OK",
                f"마지막 수집 {age:.0f}분 전" if age is not None else "기록 없음"))
    errs = [x for x in lines if " ERROR " in x or "Traceback" in x]
    recent = [x for x in errs if (_age_min(x) or 1e9) < 60]
    out.append(("데몬 오류 (최근 1시간)", "경고" if recent else "OK", f"{len(recent)}건"))

    day = now.strftime("%Y%m%d")
    rt = path("realtime", day)
    size = sum(p.stat().st_size for p in rt.glob("*.jsonl.gz")) / 1e6 if rt.exists() else 0
    out.append(("오늘 실시간 저장량", "OK", f"{size:,.1f} MB"))

    jp = path("trading", f"journal_{day}.jsonl")
    if jp.exists():
        rows = [json.loads(x) for x in jp.read_text(encoding="utf-8").splitlines()]
        last = {r["client_id"]: r for r in rows}
        st = {}
        for r in last.values():
            st[r["status"]] = st.get(r["status"], 0) + 1
        bad = st.get("unknown", 0) + st.get("not_found", 0)
        out.append(("오늘 주문 의도", "경고" if bad else "OK", json.dumps(st, ensure_ascii=False)))
    else:
        out.append(("오늘 주문 의도", "OK", "없음"))
    ks = path("KILL_SWITCH")
    out.append(("킬 스위치", "경고" if ks.exists() else "OK", "작동 중" if ks.exists() else "꺼짐"))
    out.append(("주문 모드", "OK" if trading_mode() == "paper" else "경고", trading_mode()))
    return out


def show() -> int:
    rows = check()
    print(f"== {datetime.now():%Y-%m-%d %H:%M:%S} 상태 점검")
    for name, level, detail in rows:
        print(f"[{level:2s}] {name:18s} {detail}")
    return 1 if any(level == "경고" for _, level, _ in rows) else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not a.watch:
        sys.exit(show())
    while True:
        show()
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
