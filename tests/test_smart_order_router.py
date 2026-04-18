"""Tests for strategies/smart_order_router.py — TWAP/VWAP/Iceberg."""
from __future__ import annotations

import pytest


class TestSmartOrderRouter:
    def test_create_twap(self):
        from strategies.smart_order_router import SmartOrderRouter
        sor = SmartOrderRouter()
        slices = sor.create_twap(
            total_quantity=1000,
            duration_minutes=60,
            n_slices=6,
        )
        assert len(slices) == 6
        total = sum(s.quantity for s in slices)
        assert total == 1000

    def test_create_vwap(self):
        from strategies.smart_order_router import SmartOrderRouter
        sor = SmartOrderRouter()
        profile = [0.08, 0.06, 0.05, 0.05, 0.10, 0.12, 0.08, 0.06, 0.05, 0.05, 0.08, 0.10, 0.12]
        slices = sor.create_vwap(
            total_quantity=1000,
            volume_profile=profile,
        )
        total = sum(s.quantity for s in slices)
        assert total == 1000

    def test_create_iceberg(self):
        from strategies.smart_order_router import SmartOrderRouter
        sor = SmartOrderRouter()
        slices = sor.create_iceberg(
            total_quantity=500,
            show_quantity=50,
        )
        assert len(slices) == 10
        assert all(s.quantity == 50 for s in slices)

    def test_us_equity_volume_profile(self):
        from strategies.smart_order_router import SmartOrderRouter
        sor = SmartOrderRouter()
        profile = sor.us_equity_volume_profile()
        assert len(profile) == 13
        # Profile represents percentage weights (sum ≈ 100)
        assert abs(sum(profile) - 100.0) < 0.01

    def test_simulate_execution(self):
        from strategies.smart_order_router import SmartOrderRouter
        sor = SmartOrderRouter()
        slices = sor.create_twap(100, 60, 5)
        prices = [450.0, 451.0, 449.5, 450.5, 450.0]
        volumes = [1_000_000] * 5
        result = sor.simulate_execution(slices, prices, volumes)
        assert result is not None
        assert result.total_quantity == 100
        assert result.avg_fill_price > 0

    def test_participation_rate(self):
        from strategies.smart_order_router import SmartOrderRouter
        sor = SmartOrderRouter(max_participation_rate=0.10)
        slices = sor.create_twap(100_000, 60, 5)
        prices = [450.0] * 5
        volumes = [10_000] * 5  # very low volume
        result = sor.simulate_execution(slices, prices, volumes)
        assert result is not None
        # Some slices may be partially filled due to participation limit
        assert result.filled_quantity <= result.total_quantity

    def test_vwap_with_profile(self):
        from strategies.smart_order_router import SmartOrderRouter
        sor = SmartOrderRouter()
        profile = sor.us_equity_volume_profile()
        slices = sor.create_vwap(1000, profile)
        assert len(slices) == 13
        assert sum(s.quantity for s in slices) == 1000

    def test_zero_quantity(self):
        from strategies.smart_order_router import SmartOrderRouter
        sor = SmartOrderRouter()
        slices = sor.create_twap(0, 60, 5)
        total = sum(s.quantity for s in slices)
        assert total == 0

    def test_slippage_bps(self):
        from strategies.smart_order_router import SmartOrderRouter
        sor = SmartOrderRouter()
        slices = sor.create_twap(100, 60, 4)
        prices = [450.0, 451.0, 452.0, 453.0]
        volumes = [1_000_000] * 4
        result = sor.simulate_execution(slices, prices, volumes)
        assert result.slippage_bps is not None

    def test_algo_order_result_fill_rate(self):
        from strategies.smart_order_router import SmartOrderRouter
        sor = SmartOrderRouter()
        slices = sor.create_twap(100, 60, 4)
        prices = [450.0] * 4
        volumes = [1_000_000] * 4
        result = sor.simulate_execution(slices, prices, volumes)
        assert result.fill_rate == pytest.approx(1.0)
        d = result.to_dict()
        assert "algorithm" in d
