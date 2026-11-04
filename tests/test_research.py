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
