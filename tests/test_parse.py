from datetime import datetime

from collectors.kiwoom_rest import normalize_row, num
from collectors.realtime import in_full_window


def test_num_signs():
    assert num("+1234") == 1234
    assert num("-1234") == -1234
    assert num("--8653") == -8653          # 키움 순매수 음수 표기
    assert num("-222000", absolute=True) == 222000
    assert num("+3.94") == 3.94
    assert num("") is None and num(None) is None


def test_normalize_row_keeps_time_and_code_strings():
    r = normalize_row({"cntr_tm": "20261103152000", "stk_cd": "005930", "cur_prc": "-222000",
                       "frgnr_invsr": "-210682", "upd_rt": ""})
    assert r["cntr_tm"] == "20261103152000" and r["stk_cd"] == "005930"
    assert r["cur_prc"] == 222000 and r["frgnr_invsr"] == -210682 and r["upd_rt"] is None


def test_orderbook_windows():
    assert in_full_window(datetime(2026, 11, 4, 9, 3))
    assert in_full_window(datetime(2026, 11, 4, 15, 19))
    assert in_full_window(datetime(2026, 11, 4, 19, 45))
    assert not in_full_window(datetime(2026, 11, 4, 11, 0))
