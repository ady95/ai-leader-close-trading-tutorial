"""독립 위험관리 — 주문을 보내기 전에 반드시 통과해야 하는 한도 검사.

전략·모델 코드는 이 모듈을 우회할 수 없다. 주문 관리자(execution.orders)가 매 주문마다 호출한다.
킬 스위치 파일이 있으면 모든 신규 매수를 막는다 (매도·취소는 허용 — 위험을 줄이는 방향).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RiskLimits:
    max_order_value: float = 1_000_000          # 주문 1건 최대 금액 (원)
    max_symbol_value: float = 2_000_000         # 종목당 최대 보유 금액
    max_total_value: float = 5_000_000          # 전체 최대 보유 금액
    max_orders_per_day: int = 40
    max_daily_loss: float = 300_000             # 당일 실현 손실 한도 (원). 넘으면 신규 매수 중단
    buy_windows: list[tuple[str, str]] = field(default_factory=lambda: [("1518", "1530")])
    kill_switch: Path = Path("data/lake/KILL_SWITCH")

    def check(self, side: str, symbol: str, qty: int, price: float, held_value: dict[str, float],
              orders_today: int, realized_pnl_today: float, now: datetime | None = None) -> tuple[bool, str]:
        now = now or datetime.now()
        value = qty * price
        if orders_today >= self.max_orders_per_day:
            return False, f"일일 주문 수 한도 {self.max_orders_per_day} 초과"
        if side == "sell":
            return True, "매도는 위험 축소 방향이므로 허용"
        if self.kill_switch.exists():
            return False, "킬 스위치 작동 중"
        hm = now.strftime("%H%M")
        if not any(a <= hm < b for a, b in self.buy_windows):
            return False, f"매수 허용 시간 밖 ({hm})"
        if realized_pnl_today <= -self.max_daily_loss:
            return False, "당일 손실 한도 도달"
        if value > self.max_order_value:
            return False, f"주문 금액 {value:,.0f} > 한도 {self.max_order_value:,.0f}"
        if held_value.get(symbol, 0) + value > self.max_symbol_value:
            return False, "종목당 보유 한도 초과"
        if sum(held_value.values()) + value > self.max_total_value:
            return False, "전체 보유 한도 초과"
        return True, "OK"
