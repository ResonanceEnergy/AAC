"""Tests for the NYSE breadth client (yfinance-backed)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from integrations.breadth_client import BreadthClient, BreadthSnapshot


def _fake_history(close_values: list[float]):
    """Build a minimal pandas-DataFrame-like object yfinance would return."""
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"Close": close_values})


def _patched_yf(close_map: dict[str, list[float]]):
    """Return a mock yfinance module whose Ticker(sym).history() yields the
    supplied close series for each symbol."""
    fake_yf = MagicMock()

    def ticker_factory(sym: str):
        ticker = MagicMock()
        ticker.history.return_value = _fake_history(close_map.get(sym, []))
        return ticker

    fake_yf.Ticker.side_effect = ticker_factory
    return fake_yf


class TestEMA:
    def test_returns_none_when_too_few_samples(self) -> None:
        c = BreadthClient()
        assert c._ema([1.0, 2.0, 3.0], period=5) is None

    def test_basic_ema(self) -> None:
        c = BreadthClient()
        out = c._ema([1.0] * 10, period=3)
        assert out == pytest.approx(1.0)


class TestMcClellan:
    def test_returns_none_with_short_series(self) -> None:
        c = BreadthClient()
        assert c.mcclellan_oscillator([1.0] * 30) is None

    def test_zero_oscillator_for_flat_series(self) -> None:
        c = BreadthClient()
        out = c.mcclellan_oscillator([100.0] * 50)
        assert out == pytest.approx(0.0, abs=1e-6)


class TestSnapshot:
    def test_snapshot_with_mock_yfinance(self) -> None:
        close_map = {
            "^TRIN": [0.85] * 5,
            "^TICK": [120.0] * 5,
            "^ADD": [400.0] * 50,    # adv-decl, enough for McClellan
            "^DECL": [800.0] * 5,
            "^ADV": [1800.0] * 5,
        }
        client = BreadthClient()
        with patch.object(client, "_yfinance", return_value=_patched_yf(close_map)):
            snap = client.get_snapshot()

        assert isinstance(snap, BreadthSnapshot)
        assert snap.trin == pytest.approx(0.85)
        assert snap.tick == pytest.approx(120.0)
        assert snap.advance_decline_ratio == pytest.approx(1800.0 / 800.0)
        assert snap.mcclellan_oscillator == pytest.approx(0.0, abs=1e-3)
        # TRIN<0.8 fails (0.85), but adv/decl=2.25>1.5 → 1 bullish vote, 0 bear
        # McClellan==0 → no vote. Result: only 1 bullish → neutral
        assert snap.regime in {"bullish", "neutral"}
        assert snap.error is None

    def test_snapshot_handles_empty_data(self) -> None:
        client = BreadthClient()
        with patch.object(client, "_yfinance", return_value=_patched_yf({})):
            snap = client.get_snapshot()
        assert snap.trin is None
        assert snap.advance_decline_ratio is None
        assert snap.mcclellan_oscillator is None
        assert snap.regime == "unknown"
