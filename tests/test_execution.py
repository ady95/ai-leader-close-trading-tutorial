from datetime import datetime

from collectors.kiwoom_rest import KiwoomError
from execution.orders import Intent, OrderManager
from risk.limits import RiskLimits


class FakeAPI:
    """주문 요청을 기록하는 가짜 API. lose_response=True 면 서버는 받았지만 응답이 끊긴 상황."""

    def __init__(self, lose_response=False):
        self.orders, self.lose = [], lose_response

    def request(self, path, api_id, body, next_key=None, retries=5):
        if api_id in ("kt10000", "kt10001"):
            self.orders.append(body)
            if self.lose:
                raise KiwoomError(f"{api_id} 재시도 {retries}회 실패")
            return {"return_code": 0, "ord_no": f"{len(self.orders):07d}", "return_msg": "OK"}, None
        if api_id == "ka10075":
            return {"oso": []}, None
        if api_id == "ka10076":
            now = datetime.now().strftime("%H%M%S")
            return {"cntr": [{"ord_no": f"{i + 1:07d}", "stk_cd": b["stk_cd"], "io_tp_nm": "+매수",
                              "ord_qty": b["ord_qty"], "ord_tm": now} for i, b in enumerate(self.orders)]}, None
        if api_id == "kt00018":
            return {"acnt_evlt_remn_indv_tot": []}, None
        raise AssertionError(api_id)


OPEN = RiskLimits(buy_windows=[("0000", "2400")])


def test_same_intent_is_sent_once(tmp_path):
    api = FakeAPI()
    om = OrderManager(api, tmp_path / "j.jsonl", OPEN)
    om.submit(Intent("a", "005930", "buy", 1, 100000), 100000)
    om.submit(Intent("a", "005930", "buy", 1, 100000), 100000)
    assert len(api.orders) == 1


def test_lost_response_is_reconciled_not_resent(tmp_path):
    api = FakeAPI(lose_response=True)
    om = OrderManager(api, tmp_path / "j.jsonl", OPEN)
    it = om.submit(Intent("b", "005930", "buy", 1, 100000), 100000)
    assert it.status == "accepted" and it.ord_no == "0000001" and "reconcile" in it.msg
    om.submit(Intent("b", "005930", "buy", 1, 100000), 100000)
    assert len(api.orders) == 1


def test_risk_limits_block_buys_but_allow_sells(tmp_path):
    lim = RiskLimits(buy_windows=[("1518", "1530")], kill_switch=tmp_path / "KILL")
    t = datetime(2026, 11, 4, 10, 0)
    assert lim.check("buy", "A", 1, 1000, {}, 0, 0.0, now=t) == (False, "매수 허용 시간 밖 (1000)")
    (tmp_path / "KILL").touch()
    t2 = datetime(2026, 11, 4, 15, 19)
    assert lim.check("buy", "A", 1, 1000, {}, 0, 0.0, now=t2)[1] == "킬 스위치 작동 중"
    assert lim.check("sell", "A", 1, 1000, {}, 0, 0.0, now=t2)[0]
