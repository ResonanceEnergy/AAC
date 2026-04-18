"""Tests for strategies/portfolio_optimizer.py — Portfolio Optimization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices():
    """Synthetic daily PRICE data for 5 assets, 500 days."""
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    data = {
        "SPY": 100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.012, n_days))),
        "TLT": 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.008, n_days))),
        "GLD": 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.010, n_days))),
        "VIX": 20 * np.exp(np.cumsum(np.random.normal(-0.0002, 0.030, n_days))),
        "IWM": 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n_days))),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def optimizer():
    from strategies.portfolio_optimizer import PortfolioOptimizer
    return PortfolioOptimizer()


class TestOptimizationResult:
    def test_dataclass_fields(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(
            weights={"SPY": 0.5, "TLT": 0.5},
            method="max_sharpe",
            expected_return=0.08,
            expected_volatility=0.12,
            sharpe_ratio=0.67,
        )
        assert r.expected_return == 0.08
        assert sum(r.weights.values()) == pytest.approx(1.0)


class TestPortfolioOptimizer:
    def test_min_volatility(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.MIN_VOLATILITY)
        assert result.method == "min_volatility"
        assert result.expected_volatility > 0
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.02)

    def test_risk_parity(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.RISK_PARITY)
        assert result.method == "risk_parity"
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.01)
        assert all(w > 0.01 for w in result.weights.values())

    def test_hrp(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.HRP)
        assert result.method == "hrp"
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_weights_sum_to_one(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        for method in [OptMethod.MIN_VOLATILITY, OptMethod.RISK_PARITY, OptMethod.HRP]:
            result = optimizer.optimize(sample_prices, method=method)
            assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.02)

    def test_compare_methods(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        results = optimizer.compare_methods(
            sample_prices,
            methods=[OptMethod.MIN_VOLATILITY, OptMethod.RISK_PARITY, OptMethod.HRP],
        )
        assert len(results) >= 3
        methods = {r.method for r in results}
        assert "min_volatility" in methods
        assert "risk_parity" in methods

    def test_risk_contributions(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.RISK_PARITY)
        if result.risk_contributions:
            total_rc = sum(result.risk_contributions.values())
            assert total_rc == pytest.approx(1.0, abs=0.1)

    def test_insufficient_data(self, optimizer):
        short = pd.DataFrame({"A": [100, 101], "B": [50, 51]})
        with pytest.raises(ValueError, match="Need"):
            optimizer.optimize(short)

    def test_single_asset(self, optimizer):
        np.random.seed(99)
        data = pd.DataFrame({"SPY": 100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.012, 300)))})
        with pytest.raises(ValueError, match="Need"):
            optimizer.optimize(data)

    def test_diversification_ratio(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.RISK_PARITY)
        assert result.diversification_ratio >= 0.0

    def test_to_dict(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.HRP)
        d = result.to_dict()
        assert "method" in d
        assert "weights" in d
        assert "sharpe_ratio" in d
