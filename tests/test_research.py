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
