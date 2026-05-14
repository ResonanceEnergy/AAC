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


# ---------------------------------------------------------------------------
# Sprint 25 — Comprehensive expansion
# ---------------------------------------------------------------------------

class TestVolTargetAllocation:
    def test_allocation_fields(self):
        from strategies.vol_target_sizer import VolTargetAllocation
        a = VolTargetAllocation(
            asset="SPY",
            raw_weight=0.25,
            vol_target_weight=0.30,
            realized_vol=0.15,
            notional_size=30_000.0,
            leverage=1.2,
        )
        assert a.asset == "SPY"
        assert a.raw_weight == 0.25
        assert a.vol_target_weight == 0.30
        assert a.realized_vol == 0.15
        assert a.notional_size == 30_000.0
        assert a.leverage == 1.2

    def test_allocation_default_leverage(self):
        from strategies.vol_target_sizer import VolTargetAllocation
        a = VolTargetAllocation(
            asset="TLT", raw_weight=0.5, vol_target_weight=0.5, realized_vol=0.08
        )
        assert a.leverage == 1.0
        assert a.notional_size == 0.0


class TestVolTargetResult:
    def test_result_to_dict_keys(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        result = VolTargetSizer().size_portfolio(returns_df)
        d = result.to_dict()
        for key in ("target_vol", "portfolio_vol", "scaling_factor", "total_leverage", "allocations"):
            assert key in d

    def test_result_allocation_dicts_have_fields(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        result = VolTargetSizer().size_portfolio(returns_df)
        d = result.to_dict()
        for alloc in d["allocations"]:
            assert "asset" in alloc
            assert "weight" in alloc
            assert "vol" in alloc

    def test_total_leverage_positive(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        result = VolTargetSizer().size_portfolio(returns_df)
        assert result.total_leverage > 0


class TestInverseVolWeights:
    def test_all_assets_present(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        weights = VolTargetSizer().inverse_vol_weights(returns_df)
        assert set(weights.keys()) == {"SPY", "TLT", "GLD", "VIX"}

    def test_weights_positive(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        weights = VolTargetSizer().inverse_vol_weights(returns_df)
        assert all(w > 0 for w in weights.values())

    def test_weights_sum_to_one(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        weights = VolTargetSizer().inverse_vol_weights(returns_df)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_low_vol_asset_higher_weight(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        weights = VolTargetSizer().inverse_vol_weights(returns_df)
        # TLT has lower vol than VIX
        assert weights["TLT"] > weights["VIX"]

    def test_custom_lookback_respected(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        w_short = VolTargetSizer(vol_lookback=20).inverse_vol_weights(returns_df)
        w_long = VolTargetSizer(vol_lookback=200).inverse_vol_weights(returns_df)
        # Both should still sum to 1 and have same assets
        assert sum(w_short.values()) == pytest.approx(1.0, abs=1e-6)
        assert sum(w_long.values()) == pytest.approx(1.0, abs=1e-6)


class TestSizePortfolio:
    def test_default_target_vol_attribute(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer(target_vol=0.12)
        result = sizer.size_portfolio(returns_df)
        assert result.target_vol == 0.12

    def test_scaling_factor_bounded_by_max_leverage(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer(target_vol=0.50, max_leverage=1.0)
        result = sizer.size_portfolio(returns_df)
        assert result.scaling_factor <= 1.0

    def test_allocations_count_matches_assets(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        result = VolTargetSizer().size_portfolio(returns_df)
        assert len(result.allocations) == len(returns_df.columns)

    def test_notional_scales_with_capital(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer()
        r1 = sizer.size_portfolio(returns_df, capital=100_000.0)
        r2 = sizer.size_portfolio(returns_df, capital=200_000.0)
        n1 = sum(a.notional_size for a in r1.allocations)
        n2 = sum(a.notional_size for a in r2.allocations)
        assert n2 == pytest.approx(n1 * 2, rel=0.01)

    def test_portfolio_vol_positive(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        result = VolTargetSizer().size_portfolio(returns_df)
        assert result.portfolio_vol > 0

    def test_custom_weights_all_assets(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        custom = {"SPY": 0.40, "TLT": 0.30, "GLD": 0.20, "VIX": 0.10}
        result = VolTargetSizer().size_portfolio(returns_df, base_weights=custom)
        names = {a.asset for a in result.allocations}
        assert names == {"SPY", "TLT", "GLD", "VIX"}

    def test_vol_floor_prevents_zero_division(self):
        """All-zero returns → vol_floor prevents ZeroDivisionError."""
        from strategies.vol_target_sizer import VolTargetSizer
        flat = pd.DataFrame({"A": [0.0] * 100, "B": [0.0] * 100})
        sizer = VolTargetSizer(vol_floor=0.05)
        result = sizer.size_portfolio(flat)
        assert result.portfolio_vol >= 0.05


class TestVolAdjustedKelly:
    def test_zero_avg_loss_returns_zero(self):
        from strategies.vol_target_sizer import VolTargetSizer
        assert VolTargetSizer().vol_adjusted_kelly(0.6, 100, 0.0, 0.15) == 0.0

    def test_zero_avg_win_returns_zero(self):
        from strategies.vol_target_sizer import VolTargetSizer
        assert VolTargetSizer().vol_adjusted_kelly(0.6, 0.0, 100, 0.15) == 0.0

    def test_negative_kelly_clamped_to_zero(self):
        from strategies.vol_target_sizer import VolTargetSizer
        # win_rate 0.1 → edge = 0.1*b - 0.9 ≈ negative
        k = VolTargetSizer().vol_adjusted_kelly(0.10, 200, 150, 0.15, 0.15)
        assert k == 0.0

    def test_capped_at_max_leverage(self):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer(max_leverage=1.5)
        k = sizer.vol_adjusted_kelly(0.99, 10_000, 1, 0.01, 0.15)
        assert k <= 1.5

    def test_vol_ratio_scaling(self):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer()
        k_normal = sizer.vol_adjusted_kelly(0.60, 200, 150, 0.15, 0.15)
        k_crisis = sizer.vol_adjusted_kelly(0.60, 200, 150, 0.45, 0.15)
        assert k_crisis < k_normal  # 3× vol → smaller position

    def test_zero_current_vol_uses_floor(self):
        from strategies.vol_target_sizer import VolTargetSizer
        # Should not raise; vol_floor prevents division by zero
        k = VolTargetSizer().vol_adjusted_kelly(0.55, 200, 150, 0.0, 0.15)
        assert k >= 0.0


class TestTargetSizeSingle:
    def test_low_vol_asset_gets_higher_weight(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer(target_vol=0.15)
        low_vol = sizer.target_size_single(returns_df["TLT"], asset_name="TLT")
        high_vol = sizer.target_size_single(returns_df["VIX"], asset_name="VIX")
        assert low_vol.vol_target_weight > high_vol.vol_target_weight

    def test_asset_name_preserved(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        alloc = VolTargetSizer().target_size_single(returns_df["GLD"], asset_name="GLD")
        assert alloc.asset == "GLD"

    def test_notional_size_positive(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        alloc = VolTargetSizer().target_size_single(returns_df["SPY"], capital=50_000.0)
        assert alloc.notional_size > 0

    def test_weight_capped_at_max_leverage(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        sizer = VolTargetSizer(target_vol=0.99, max_leverage=1.0)
        alloc = sizer.target_size_single(returns_df["TLT"])
        assert alloc.vol_target_weight <= 1.0

    def test_raw_weight_is_one(self, returns_df):
        from strategies.vol_target_sizer import VolTargetSizer
        alloc = VolTargetSizer().target_size_single(returns_df["SPY"])
        assert alloc.raw_weight == 1.0

