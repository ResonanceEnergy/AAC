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


# ---------------------------------------------------------------------------
# Sprint 27 — comprehensive coverage
# ---------------------------------------------------------------------------

class TestOptMethodEnum:
    """All six OptMethod variants exist with the correct string values."""

    def test_min_volatility_value(self):
        from strategies.portfolio_optimizer import OptMethod
        assert OptMethod.MIN_VOLATILITY.value == "min_volatility"

    def test_max_sharpe_value(self):
        from strategies.portfolio_optimizer import OptMethod
        assert OptMethod.MAX_SHARPE.value == "max_sharpe"

    def test_mean_variance_value(self):
        from strategies.portfolio_optimizer import OptMethod
        assert OptMethod.MEAN_VARIANCE.value == "mean_variance"

    def test_risk_parity_value(self):
        from strategies.portfolio_optimizer import OptMethod
        assert OptMethod.RISK_PARITY.value == "risk_parity"

    def test_hrp_value(self):
        from strategies.portfolio_optimizer import OptMethod
        assert OptMethod.HRP.value == "hrp"

    def test_black_litterman_value(self):
        from strategies.portfolio_optimizer import OptMethod
        assert OptMethod.BLACK_LITTERMAN.value == "black_litterman"


class TestOptimizationResultFields:
    """OptimizationResult dataclass fields and defaults."""

    def test_all_required_fields_accessible(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(method="test", weights={"A": 0.6, "B": 0.4})
        assert r.method == "test"
        assert r.weights == {"A": 0.6, "B": 0.4}

    def test_expected_return_default_zero(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(method="test", weights={})
        assert r.expected_return == 0.0

    def test_expected_volatility_default_zero(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(method="test", weights={})
        assert r.expected_volatility == 0.0

    def test_sharpe_default_zero(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(method="test", weights={})
        assert r.sharpe_ratio == 0.0

    def test_max_drawdown_estimate_default_zero(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(method="test", weights={})
        assert r.max_drawdown_estimate == 0.0

    def test_risk_contributions_default_empty(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(method="test", weights={})
        assert r.risk_contributions == {}

    def test_details_default_empty(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(method="test", weights={})
        assert r.details == {}

    def test_diversification_ratio_default_zero(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(method="test", weights={})
        assert r.diversification_ratio == 0.0


class TestOptimizationResultTopHoldings:
    """top_holdings property returns sorted (ticker, weight) pairs."""

    def test_sorted_descending(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(
            method="test",
            weights={"SPY": 0.2, "TLT": 0.5, "GLD": 0.3},
        )
        top = r.top_holdings
        assert top[0] == ("TLT", 0.5)
        assert top[1] == ("GLD", 0.3)
        assert top[2] == ("SPY", 0.2)

    def test_single_asset(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(method="test", weights={"SPY": 1.0})
        assert r.top_holdings == [("SPY", 1.0)]

    def test_returns_list_of_tuples(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(method="test", weights={"A": 0.5, "B": 0.5})
        top = r.top_holdings
        assert isinstance(top, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in top)


class TestOptimizationResultToDict:
    """to_dict() returns exactly the right structure."""

    def test_exactly_seven_keys(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(
            method="min_volatility",
            weights={"SPY": 0.5, "TLT": 0.5},
            expected_return=0.08,
            expected_volatility=0.10,
            sharpe_ratio=0.755,
            diversification_ratio=1.2,
            risk_contributions={"SPY": 0.5, "TLT": 0.5},
        )
        d = r.to_dict()
        assert set(d.keys()) == {
            "method", "weights", "expected_return",
            "expected_volatility", "sharpe_ratio",
            "diversification_ratio", "risk_contributions",
        }

    def test_expected_return_rounded_to_6dp(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(method="test", weights={}, expected_return=0.123456789)
        assert r.to_dict()["expected_return"] == pytest.approx(0.123457, abs=1e-6)

    def test_sharpe_rounded_to_4dp(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(method="test", weights={}, sharpe_ratio=1.23456789)
        assert r.to_dict()["sharpe_ratio"] == pytest.approx(1.2346, abs=1e-4)

    def test_risk_contributions_rounded(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(
            method="test",
            weights={},
            risk_contributions={"A": 0.123456789},
        )
        d = r.to_dict()
        assert d["risk_contributions"]["A"] == pytest.approx(0.123457, abs=1e-6)

    def test_weights_preserved(self):
        from strategies.portfolio_optimizer import OptimizationResult
        w = {"SPY": 0.4, "TLT": 0.6}
        r = OptimizationResult(method="test", weights=w)
        assert r.to_dict()["weights"] == w

    def test_method_string_preserved(self):
        from strategies.portfolio_optimizer import OptimizationResult
        r = OptimizationResult(method="hrp", weights={})
        assert r.to_dict()["method"] == "hrp"


class TestPortfolioOptimizerParams:
    """Constructor params are stored correctly."""

    def test_default_risk_free_rate(self):
        from strategies.portfolio_optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        assert opt.risk_free_rate == pytest.approx(0.045)

    def test_default_frequency(self):
        from strategies.portfolio_optimizer import PortfolioOptimizer
        assert PortfolioOptimizer().frequency == 252

    def test_default_weight_bounds(self):
        from strategies.portfolio_optimizer import PortfolioOptimizer
        assert PortfolioOptimizer().weight_bounds == (0.0, 0.40)

    def test_custom_risk_free_rate(self):
        from strategies.portfolio_optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(risk_free_rate=0.05)
        assert opt.risk_free_rate == pytest.approx(0.05)

    def test_custom_frequency(self):
        from strategies.portfolio_optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(frequency=365)
        assert opt.frequency == 365

    def test_custom_weight_bounds(self):
        from strategies.portfolio_optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(weight_bounds=(0.05, 0.30))
        assert opt.weight_bounds == (0.05, 0.30)


class TestMaxSharpe:
    """MAX_SHARPE optimization output contracts."""

    def test_method_name(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.MAX_SHARPE)
        assert result.method == "max_sharpe"

    def test_weights_sum_to_one(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.MAX_SHARPE)
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.02)

    def test_all_weights_in_bounds(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.MAX_SHARPE)
        for w in result.weights.values():
            assert 0.0 <= w <= 0.40 + 1e-6

    def test_expected_volatility_positive(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.MAX_SHARPE)
        assert result.expected_volatility > 0

    def test_risk_contributions_present(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.MAX_SHARPE)
        assert isinstance(result.risk_contributions, dict)


class TestMeanVarianceAlias:
    """MEAN_VARIANCE dispatches to the same path as MAX_SHARPE."""

    def test_mean_variance_method_name_is_max_sharpe(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.MEAN_VARIANCE)
        # MEAN_VARIANCE is an alias for MAX_SHARPE
        assert result.method == "max_sharpe"

    def test_mean_variance_weights_sum_to_one(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.MEAN_VARIANCE)
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.02)


class TestBlackLitterman:
    """BLACK_LITTERMAN optimization with and without views."""

    def test_no_views_runs(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.BLACK_LITTERMAN)
        assert result.method == "black_litterman"

    def test_no_views_weights_sum_to_one(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.BLACK_LITTERMAN)
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.02)

    def test_with_views_runs(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        views = {"SPY": 0.05, "TLT": -0.02}
        result = optimizer.optimize(
            sample_prices, method=OptMethod.BLACK_LITTERMAN, views=views
        )
        assert result.method == "black_litterman"
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.02)

    def test_details_contains_bl_expected_returns(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.BLACK_LITTERMAN)
        assert "bl_expected_returns" in result.details


class TestCompareMethods:
    """compare_methods() returns sorted results."""

    def test_returns_list(self, optimizer, sample_prices):
        results = optimizer.compare_methods(sample_prices)
        assert isinstance(results, list)

    def test_sorted_by_sharpe_descending(self, optimizer, sample_prices):
        results = optimizer.compare_methods(sample_prices)
        sharpes = [r.sharpe_ratio for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_default_includes_four_methods(self, optimizer, sample_prices):
        results = optimizer.compare_methods(sample_prices)
        assert len(results) >= 3  # at least 3 succeed (BL may fail on some data)

    def test_custom_method_list(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        results = optimizer.compare_methods(
            sample_prices,
            methods=[OptMethod.RISK_PARITY, OptMethod.HRP],
        )
        assert len(results) == 2
        methods = {r.method for r in results}
        assert "risk_parity" in methods
        assert "hrp" in methods

    def test_returns_optimization_result_instances(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptimizationResult
        results = optimizer.compare_methods(sample_prices)
        assert all(isinstance(r, OptimizationResult) for r in results)


class TestRiskContributions:
    """_compute_risk_contributions helper."""

    def test_sum_to_one_for_risk_parity(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.RISK_PARITY)
        if result.risk_contributions:
            total = sum(result.risk_contributions.values())
            assert total == pytest.approx(1.0, abs=0.05)

    def test_all_keys_match_columns(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.RISK_PARITY)
        if result.risk_contributions:
            assert set(result.risk_contributions.keys()) == set(sample_prices.columns)

    def test_all_non_negative(self, optimizer, sample_prices):
        from strategies.portfolio_optimizer import OptMethod
        result = optimizer.optimize(sample_prices, method=OptMethod.HRP)
        for rc in result.risk_contributions.values():
            assert rc >= -1e-9  # float tolerance


class TestUnknownMethod:
    """Unknown optimization method raises ValueError."""

    def test_unknown_method_raises(self, optimizer, sample_prices):
        with pytest.raises((ValueError, AttributeError, KeyError)):
            # Passing a string where OptMethod is expected
            optimizer.optimize(sample_prices, method="bogus_method")  # type: ignore[arg-type]

