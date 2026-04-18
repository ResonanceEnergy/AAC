"""Tests for strategies/correlation_tracker.py — Dynamic Correlation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def price_data():
    """Synthetic price data for 4 assets, 300 days."""
    np.random.seed(123)
    n = 300
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    base = np.random.normal(0, 0.01, n)
    data = pd.DataFrame({
        "SPY": 100 * np.exp(np.cumsum(base + np.random.normal(0, 0.005, n))),
        "IWM": 100 * np.exp(np.cumsum(base * 0.8 + np.random.normal(0, 0.007, n))),
        "TLT": 100 * np.exp(np.cumsum(-base * 0.3 + np.random.normal(0, 0.006, n))),
        "GLD": 100 * np.exp(np.cumsum(np.random.normal(0, 0.008, n))),
    }, index=dates)
    return data


@pytest.fixture
def tracker():
    from strategies.correlation_tracker import CorrelationTracker
    return CorrelationTracker(halflife=21)


class TestContagionAlert:
    def test_severity_levels(self):
        from strategies.correlation_tracker import ContagionAlert
        a = ContagionAlert(
            timestamp="2026-01-01",
            asset_a="SPY",
            asset_b="IWM",
            current_corr=0.95,
            baseline_corr=0.60,
            z_score=3.0,
            severity="critical",
            message="SPY/IWM correlation spiked",
        )
        assert a.severity == "critical"
        assert a.z_score == 3.0
        assert a.asset_a == "SPY"


class TestCorrelationSnapshot:
    def test_snapshot_fields(self):
        from strategies.correlation_tracker import CorrelationSnapshot
        s = CorrelationSnapshot(
            timestamp="2026-01-01",
            correlation_matrix=pd.DataFrame(),
            eigenvalues=[1.0, 0.5],
            effective_n_assets=1.5,
            absorption_ratio=0.5,
            regime="normal",
        )
        assert s.regime == "normal"
        assert s.absorption_ratio == 0.5


class TestCorrelationTracker:
    def test_update(self, tracker, price_data):
        snapshot = tracker.update(price_data)
        assert snapshot is not None
        assert snapshot.regime in ("normal", "decorrelating", "contagion")
        assert snapshot.absorption_ratio >= 0
        assert isinstance(snapshot.correlation_matrix, pd.DataFrame)
        assert snapshot.correlation_matrix.shape == (4, 4)

    def test_eigenvalues(self, tracker, price_data):
        snapshot = tracker.update(price_data)
        assert len(snapshot.eigenvalues) == 4
        assert all(v >= -1e-10 for v in snapshot.eigenvalues)

    def test_effective_n_assets(self, tracker, price_data):
        snapshot = tracker.update(price_data)
        assert 1.0 <= snapshot.effective_n_assets <= 4.0

    def test_contagion_detection(self, tracker, price_data):
        snapshot = tracker.update(price_data)
        assert snapshot.regime in ("normal", "decorrelating", "contagion")

    def test_ewm_covariance(self, tracker, price_data):
        cov = tracker.get_ewm_covariance(price_data)
        assert cov is not None
        assert cov.shape == (4, 4)
        for i in range(4):
            assert cov.iloc[i, i] > 0

    def test_alerts(self, tracker, price_data):
        snapshot = tracker.update(price_data)
        assert isinstance(snapshot.contagion_alerts, list)

    def test_short_history(self, tracker):
        short = pd.DataFrame({
            "A": [100, 101, 102],
            "B": [50, 51, 50],
        })
        with pytest.raises(ValueError, match="Need"):
            tracker.update(short)

    def test_regime_classification(self, tracker):
        """During high-correlation crisis, regime should shift."""
        np.random.seed(55)
        n = 300
        base = np.cumsum(np.random.normal(0, 0.01, n)) + 100
        crisis = pd.DataFrame({
            "A": base,
            "B": base * 1.1 + np.random.normal(0, 0.001, n),
            "C": base * 0.9 + np.random.normal(0, 0.001, n),
            "D": base * 1.05 + np.random.normal(0, 0.001, n),
        })
        snapshot = tracker.update(crisis)
        assert snapshot.absorption_ratio > 0.5

    def test_to_dict(self, tracker, price_data):
        snapshot = tracker.update(price_data)
        d = snapshot.to_dict()
        assert "regime" in d
        assert "absorption_ratio" in d
        assert "effective_n_assets" in d
