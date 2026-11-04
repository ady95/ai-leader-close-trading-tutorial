import pandas as pd
import pytest

from backtest.costs import round_trip_cost
from backtest.metrics import max_drawdown, trade_stats
from data import dataset as ds


def test_holdout_is_sealed(monkeypatch):
    monkeypatch.delenv(ds.UNSEAL_ENV, raising=False)
    with pytest.raises(ds.HoldoutSealed):
        ds._check_period("holdout")
    assert ds._check_period("dev") == (ds.DEV_START, ds.DEV_END)


def test_round_trip_cost_by_sell_year():
    c = round_trip_cost(pd.Series(["20231228", "20250102"]))
    assert c.round(6).tolist() == [0.0023, 0.0018]


def test_trade_stats_and_mdd():
    s = trade_stats(pd.Series([0.02, -0.01, 0.03, -0.02]))
    assert s["건수"] == 4 and s["PF"] == pytest.approx(5 / 3)
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_afterhours_close_is_replaced_by_1530(monkeypatch):
    reg = pd.DataFrame({"dt": ["20261103"], "open": [10.0], "high": [12.0], "low": [9.0], "close": [11.0],
                        "volume": [100]})
    monkeypatch.setattr(ds, "regular_session_daily", lambda s: reg if s == "A" else pd.DataFrame())
    df = pd.DataFrame({"symbol": ["A", "A", "B"], "dt": ["20260911", "20261103", "20261103"],
                       "open": [1.0] * 3, "high": [1.0] * 3, "low": [1.0] * 3, "close": [5.0] * 3})
    out = ds._fix_afterhours_close(df)
    assert out["close"].tolist()[:2] == [5.0, 11.0] and pd.isna(out["close"].iloc[2])
    assert out["close_source"].tolist() == ["daily", "minute_1530", "unavailable"]


def test_exit_simulator_conservative_order():
    import numpy as np

    from strategies.exit_policies.intraday_exits import simulate
    hm = np.array(["0900", "0901", "0902", "1059"])
    o = np.array([100.0, 101.0, 99.0, 99.0])
    h = np.array([101.0, 104.0, 99.5, 99.5])   # 09:01 봉에서 +3% 익절선(103) 도달
    l = np.array([99.5, 97.0, 98.5, 98.5])     # 같은 봉에서 -2% 손절선(98)도 도달 → 손절 먼저
    c = np.array([100.5, 99.0, 99.0, 99.0])
    v = np.ones(4)
    r = simulate(100.0, 2.0, hm, o, h, l, c, v)
    assert r["S1"] == 100.0
    assert r["S2"] == 98.0 - 0.5          # 손절가 98에서 반 호가(1원 호가의 절반) 불리
