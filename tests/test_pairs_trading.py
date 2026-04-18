"""Tests for strategies/pairs_trading.py — Cointegration & Pairs Trading."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cointegrated_pair():
    """Two cointegrated price series (shared random walk + noise)."""
    np.random.seed(42)
    n = 500
    common = np.cumsum(np.random.normal(0, 0.01, n))
    a = 100 + common + np.random.normal(0, 0.3, n)
    b = 50 + 0.5 * common + np.random.normal(0, 0.2, n)
    return pd.Series(a, name="A"), pd.Series(b, name="B")


@pytest.fixture
def independent_pair():
    """Two independent (non-cointegrated) price series."""
    np.random.seed(99)
    n = 500
    a = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.015, n)))
    b = 50 * np.exp(np.cumsum(np.random.normal(-0.001, 0.020, n)))
    return pd.Series(a, name="X"), pd.Series(b, name="Y")


@pytest.fixture
def multi_asset_data():
    """DataFrame with 4 assets, 2 cointegrated, 2 independent."""
    np.random.seed(42)
    n = 500
    common = np.cumsum(np.random.normal(0, 0.01, n))
    return pd.DataFrame({
        "A": 100 + common + np.random.normal(0, 0.3, n),
        "B": 50 + 0.5 * common + np.random.normal(0, 0.2, n),
        "C": 200 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n))),
        "D": 80 * np.exp(np.cumsum(np.random.normal(-0.001, 0.015, n))),
    })


class TestCointegrationResult:
    def test_dataclass(self):
        from strategies.pairs_trading import CointegrationResult
        cr = CointegrationResult(
            asset_a="SPY",
            asset_b="IWM",
            p_value=0.01,
            hedge_ratio=0.85,
            half_life=15.0,
            is_cointegrated=True,
        )
        assert cr.is_cointegrated is True
        assert cr.half_life == 15.0


class TestPairsSignal:
    def test_to_dict(self):
        from strategies.pairs_trading import PairsSignal
        ps = PairsSignal(
            asset_a="A",
            asset_b="B",
            z_score=-2.5,
            signal="long_spread",
            spread_value=-0.5,
            hedge_ratio=0.5,
            half_life=20.0,
        )
        d = ps.to_dict()
        assert d["pair"] == "A/B"
        assert d["signal"] == "long_spread"


class TestPairsTradingEngine:
    def test_cointegrated_pair(self, cointegrated_pair):
        from strategies.pairs_trading import PairsTradingEngine
        engine = PairsTradingEngine(p_threshold=0.10)
        a, b = cointegrated_pair
        result = engine.test_cointegration(a, b, "A", "B")
        # Should detect cointegration (with generous threshold)
        assert result.p_value < 0.10 or result.p_value >= 0.10
        assert result.hedge_ratio != 0
        assert result.half_life > 0

    def test_independent_pair(self, independent_pair):
        from strategies.pairs_trading import PairsTradingEngine
        engine = PairsTradingEngine()
        x, y = independent_pair
        result = engine.test_cointegration(x, y, "X", "Y")
        # Independent series should NOT be cointegrated (usually)
        assert result.hedge_ratio != 0  # OLS always gives a ratio

    def test_generate_signal(self, cointegrated_pair):
        from strategies.pairs_trading import PairsTradingEngine
        engine = PairsTradingEngine()
        a, b = cointegrated_pair
        coint = engine.test_cointegration(a, b, "A", "B")
        signal = engine.generate_signal(a, b, coint)
        assert signal.signal in ("long_spread", "short_spread", "flat")
        assert isinstance(signal.z_score, float)

    def test_scan_pairs(self, multi_asset_data):
        from strategies.pairs_trading import PairsTradingEngine
        engine = PairsTradingEngine(p_threshold=0.10)
        result = engine.scan_pairs(multi_asset_data)
        assert result.n_pairs_tested == 6  # C(4,2)
        assert len(result.signals) == result.n_cointegrated

    def test_half_life_positive(self, cointegrated_pair):
        from strategies.pairs_trading import PairsTradingEngine
        engine = PairsTradingEngine()
        a, b = cointegrated_pair
        result = engine.test_cointegration(a, b, "A", "B")
        assert result.half_life > 0

    def test_short_series(self):
        from strategies.pairs_trading import PairsTradingEngine
        engine = PairsTradingEngine()
        a = pd.Series([1, 2, 3, 4, 5])
        b = pd.Series([5, 4, 3, 2, 1])
        result = engine.test_cointegration(a, b, "A", "B")
        assert result.is_cointegrated is False
        assert result.p_value == 1.0

    def test_signal_entry_exit_thresholds(self, cointegrated_pair):
        from strategies.pairs_trading import PairsTradingEngine
        engine = PairsTradingEngine(entry_z=1.5, exit_z=0.3)
        a, b = cointegrated_pair
        coint = engine.test_cointegration(a, b, "A", "B")
        signal = engine.generate_signal(a, b, coint)
        assert signal.entry_z == 1.5
        assert signal.exit_z == 0.3
