"""Tests for strategies/vol_target_sizer.py — Volatility-Targeting Position Sizing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def returns_df():
    """Synthetic daily returns for 4 assets."""
    np.random.seed(55)
    n = 300
    return pd.DataFrame({
        "SPY": np.random.normal(0.0004, 0.012, n),
        "TLT": np.random.normal(0.0001, 0.006, n),
        "GLD": np.random.normal(0.0002, 0.010, n),
        "VIX": np.random.normal(-0.0002, 0.035, n),
    })


class TestVolTargetSizer:
    def test_inverse_vol_weights(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer()
        weights = sizer.inverse_vol_weights(returns_df)
        assert len(weights) == 4
        assert sum(weights.values()) == pytest.approx(1.0)
        # Lower vol assets should get higher weight
        assert weights["TLT"] > weights["VIX"]

    def test_size_portfolio(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer(target_vol=0.15)
        result = sizer.size_portfolio(returns_df, capital=100_000.0)
        assert result.target_vol == 0.15
        assert result.portfolio_vol > 0
        assert result.scaling_factor > 0
        assert len(result.allocations) == 4

    def test_custom_weights(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer(target_vol=0.10)
        custom = {"SPY": 0.5, "TLT": 0.3, "GLD": 0.15, "VIX": 0.05}
        result = sizer.size_portfolio(returns_df, base_weights=custom)
        assert result.target_vol == 0.10
        assert len(result.allocations) == 4

    def test_max_leverage_cap(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer(target_vol=0.50, max_leverage=1.5)
        result = sizer.size_portfolio(returns_df)
        assert result.scaling_factor <= 1.5

    def test_notional_sizing(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer()
        result = sizer.size_portfolio(returns_df, capital=200_000.0)
        total_notional = sum(a.notional_size for a in result.allocations)
        assert total_notional > 0

    def test_vol_adjusted_kelly(self):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer()
        kelly = sizer.vol_adjusted_kelly(
            win_rate=0.55,
            avg_win=200.0,
            avg_loss=150.0,
            current_vol=0.20,
            historical_vol=0.15,
        )
        assert kelly >= 0
        assert kelly <= 2.0

    def test_vol_adjusted_kelly_high_vol(self):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer()
        normal = sizer.vol_adjusted_kelly(0.55, 200, 150, 0.15, 0.15)
        high_vol = sizer.vol_adjusted_kelly(0.55, 200, 150, 0.30, 0.15)
        # Higher vol → smaller position
        assert high_vol < normal

    def test_target_size_single(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer(target_vol=0.10)
        alloc = sizer.target_size_single(
            returns_df["VIX"],
            capital=100_000.0,
            asset_name="VIX",
        )
        assert alloc.asset == "VIX"
        # VIX is high vol, so weight should be < 1.0
        assert alloc.vol_target_weight < 1.0
        assert alloc.notional_size > 0

    def test_to_dict(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer()
        result = sizer.size_portfolio(returns_df)
        d = result.to_dict()
        assert "target_vol" in d
        assert "allocations" in d
        assert len(d["allocations"]) == 4
