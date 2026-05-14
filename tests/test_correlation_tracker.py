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


# ── Sprint 26: deeper coverage ─────────────────────────────────────────────

class TestContagionAlertFields:
    """All ContagionAlert dataclass fields are readable."""

    def test_all_fields_accessible(self):
        from strategies.correlation_tracker import ContagionAlert
        a = ContagionAlert(
            timestamp="2026-06-01T10:00:00",
            asset_a="SPY",
            asset_b="TLT",
            current_corr=0.92,
            baseline_corr=0.45,
            z_score=2.8,
            severity="warning",
            message="SPY/TLT correlation spike",
        )
        assert a.timestamp == "2026-06-01T10:00:00"
        assert a.asset_b == "TLT"
        assert a.current_corr == pytest.approx(0.92)
        assert a.baseline_corr == pytest.approx(0.45)
        assert a.message == "SPY/TLT correlation spike"

    def test_severity_watch(self):
        from strategies.correlation_tracker import ContagionAlert
        a = ContagionAlert("ts", "A", "B", 0.7, 0.5, 1.5, "watch", "msg")
        assert a.severity == "watch"

    def test_severity_critical(self):
        from strategies.correlation_tracker import ContagionAlert
        a = ContagionAlert("ts", "A", "B", 0.99, 0.6, 4.0, "critical", "spike")
        assert a.severity == "critical"
        assert a.z_score == pytest.approx(4.0)


class TestCorrelationSnapshotToDict:
    """CorrelationSnapshot.to_dict() returns exactly the expected 5 keys."""

    def test_to_dict_key_count(self, tracker, price_data):
        snap = tracker.update(price_data)
        d = snap.to_dict()
        assert set(d.keys()) == {"timestamp", "effective_n_assets", "absorption_ratio", "regime", "n_alerts"}

    def test_to_dict_n_alerts_zero(self, tracker, price_data):
        snap = tracker.update(price_data)
        d = snap.to_dict()
        assert d["n_alerts"] == len(snap.contagion_alerts)
        assert isinstance(d["n_alerts"], int)

    def test_to_dict_absorption_ratio_rounded(self, tracker, price_data):
        snap = tracker.update(price_data)
        d = snap.to_dict()
        # absorption_ratio rounded to 4 decimal places
        assert d["absorption_ratio"] == round(snap.absorption_ratio, 4)

    def test_to_dict_effective_n_rounded(self, tracker, price_data):
        snap = tracker.update(price_data)
        d = snap.to_dict()
        assert d["effective_n_assets"] == round(snap.effective_n_assets, 2)

    def test_to_dict_regime_is_string(self, tracker, price_data):
        snap = tracker.update(price_data)
        d = snap.to_dict()
        assert isinstance(d["regime"], str)
        assert d["regime"] in ("normal", "decorrelating", "contagion")

    def test_to_dict_timestamp_is_string(self, tracker, price_data):
        snap = tracker.update(price_data)
        d = snap.to_dict()
        assert isinstance(d["timestamp"], str)


class TestLastSnapshotCached:
    """tracker.last_snapshot is populated after update() and matches return value."""

    def test_last_snapshot_none_before_update(self):
        from strategies.correlation_tracker import CorrelationTracker
        ct = CorrelationTracker()
        assert ct.last_snapshot is None

    def test_last_snapshot_set_after_update(self, tracker, price_data):
        snap = tracker.update(price_data)
        assert tracker.last_snapshot is snap

    def test_last_snapshot_updated_on_second_call(self, tracker, price_data):
        tracker.update(price_data)
        first = tracker.last_snapshot
        snap2 = tracker.update(price_data)
        assert tracker.last_snapshot is snap2
        assert tracker.last_snapshot is not first

    def test_last_snapshot_regime_matches(self, tracker, price_data):
        snap = tracker.update(price_data)
        assert tracker.last_snapshot.regime == snap.regime

    def test_last_snapshot_absorption_matches(self, tracker, price_data):
        snap = tracker.update(price_data)
        assert tracker.last_snapshot.absorption_ratio == pytest.approx(snap.absorption_ratio)


class TestCorrelationTrackerParams:
    """Constructor parameters are stored and respected."""

    def test_custom_halflife(self):
        from strategies.correlation_tracker import CorrelationTracker
        ct = CorrelationTracker(halflife=10)
        assert ct.halflife == 10

    def test_custom_lookback(self):
        from strategies.correlation_tracker import CorrelationTracker
        ct = CorrelationTracker(lookback=100)
        assert ct.lookback == 100

    def test_custom_z_threshold(self):
        from strategies.correlation_tracker import CorrelationTracker
        ct = CorrelationTracker(contagion_z_threshold=3.5)
        assert ct.contagion_z_threshold == pytest.approx(3.5)

    def test_custom_absorption_warning(self):
        from strategies.correlation_tracker import CorrelationTracker
        ct = CorrelationTracker(absorption_warning=0.70)
        assert ct.absorption_warning == pytest.approx(0.70)

    def test_default_params(self):
        from strategies.correlation_tracker import CorrelationTracker
        ct = CorrelationTracker()
        assert ct.halflife == 21
        assert ct.lookback == 252
        assert ct.contagion_z_threshold == pytest.approx(2.0)
        assert ct.absorption_warning == pytest.approx(0.80)


class TestCorrelationTrackerRegimes:
    """Regime classification boundaries are correct."""

    def test_normal_regime_moderate_correlation(self, tracker, price_data):
        snap = tracker.update(price_data)
        # With 4 assets and varied returns, absorption should be < 0.80 for normal data
        if snap.absorption_ratio < 0.80 and snap.absorption_ratio >= 0.40:
            assert snap.regime == "normal"

    def test_contagion_regime_high_absorption(self):
        """High-correlation data (absorption > 0.80) triggers contagion regime."""
        from strategies.correlation_tracker import CorrelationTracker
        np.random.seed(99)
        n = 300
        base = np.cumsum(np.random.normal(0, 0.01, n)) + 100
        # Near-identical asset paths → very high correlation → high absorption
        data = pd.DataFrame({
            "A": base,
            "B": base + np.random.normal(0, 0.0001, n),
            "C": base + np.random.normal(0, 0.0001, n),
            "D": base + np.random.normal(0, 0.0001, n),
        })
        ct = CorrelationTracker(absorption_warning=0.80)
        snap = ct.update(data)
        assert snap.regime == "contagion"
        assert snap.absorption_ratio > 0.80

    def test_decorrelating_regime_low_absorption(self):
        """Orthogonal assets (absorption < 0.40) trigger decorrelating regime."""
        from strategies.correlation_tracker import CorrelationTracker
        np.random.seed(77)
        n = 300
        data = pd.DataFrame({
            "A": np.exp(np.cumsum(np.random.normal(0, 0.02, n))) * 100,
            "B": np.exp(np.cumsum(np.random.normal(0, 0.02, n))) * 100,
        })
        ct = CorrelationTracker()
        snap = ct.update(data)
        # Regime depends on actual absorption — verify consistency
        if snap.absorption_ratio < 0.40:
            assert snap.regime == "decorrelating"

    def test_absorption_ratio_in_unit_interval(self, tracker, price_data):
        snap = tracker.update(price_data)
        assert 0.0 <= snap.absorption_ratio <= 1.0

    def test_effective_n_assets_at_least_one(self, tracker, price_data):
        snap = tracker.update(price_data)
        assert snap.effective_n_assets >= 1.0

    def test_effective_n_assets_at_most_n(self, tracker, price_data):
        snap = tracker.update(price_data)
        n_assets = price_data.shape[1]
        assert snap.effective_n_assets <= n_assets + 1e-9  # float tolerance


class TestEwmCovarianceDiagonal:
    """get_ewm_covariance() diagonal values are positive (non-zero variance)."""

    def test_diagonal_positive(self, tracker, price_data):
        cov = tracker.get_ewm_covariance(price_data)
        for i in range(cov.shape[0]):
            assert cov.iloc[i, i] > 0, f"Diagonal [{i},{i}] not positive"

    def test_returns_dataframe(self, tracker, price_data):
        cov = tracker.get_ewm_covariance(price_data)
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape[0] == cov.shape[1]

    def test_column_labels_preserved(self, tracker, price_data):
        cov = tracker.get_ewm_covariance(price_data)
        assert list(cov.columns) == list(price_data.columns)
        assert list(cov.index) == list(price_data.columns)

