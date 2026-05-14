"""Tests for strategies/simple_backtest.py — run_walk_forward() Sprint 24."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yf_history(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic yfinance-style history DataFrame."""
    np.random.seed(seed)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Close": prices}, index=dates)
    return df


def _make_mock_ticker(n: int = 500, seed: int = 42):
    """Return a mock yfinance Ticker whose .history() returns synthetic data."""
    hist = _make_yf_history(n, seed)
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = hist
    return mock_ticker


# ---------------------------------------------------------------------------
# Tests: run_walk_forward() — happy path
# ---------------------------------------------------------------------------

class TestRunWalkForwardHappyPath:
    def _run(self, ticker="SPY", lookback_days=500, train_window=252, test_window=63, n=500):
        from strategies.simple_backtest import run_walk_forward
        mock_ticker = _make_mock_ticker(n=n)
        with patch("yfinance.Ticker", return_value=mock_ticker):
            return run_walk_forward(
                ticker=ticker,
                lookback_days=lookback_days,
                train_window=train_window,
                test_window=test_window,
            )

    def test_returns_dict(self):
        result = self._run()
        assert isinstance(result, dict)

    def test_ticker_in_result(self):
        result = self._run(ticker="QQQ")
        assert result["ticker"] == "QQQ"

    def test_train_window_in_result(self):
        result = self._run(train_window=100, test_window=30, n=300)
        assert result["train_window"] == 100

    def test_test_window_in_result(self):
        result = self._run(train_window=100, test_window=30, n=300)
        assert result["test_window"] == 30

    def test_no_error_on_success(self):
        result = self._run()
        assert result["error"] is None

    def test_mean_reversion_present(self):
        result = self._run()
        assert result["mean_reversion"] is not None
        assert isinstance(result["mean_reversion"], dict)

    def test_momentum_present(self):
        result = self._run()
        assert result["momentum"] is not None
        assert isinstance(result["momentum"], dict)

    def test_mean_reversion_has_oos_sharpe(self):
        result = self._run()
        assert "oos_sharpe" in result["mean_reversion"]

    def test_momentum_has_total_folds(self):
        result = self._run()
        assert "total_folds" in result["momentum"]

    def test_folds_positive(self):
        result = self._run(train_window=100, test_window=50, n=400)
        assert result["mean_reversion"]["total_folds"] > 0
        assert result["momentum"]["total_folds"] > 0

    def test_consistency_ratio_bounded(self):
        result = self._run(train_window=100, test_window=50, n=400)
        assert 0.0 <= result["mean_reversion"]["consistency_ratio"] <= 1.0
        assert 0.0 <= result["momentum"]["consistency_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: run_walk_forward() — error handling
# ---------------------------------------------------------------------------

class TestRunWalkForwardErrorHandling:
    def test_empty_history_returns_error(self):
        from strategies.simple_backtest import run_walk_forward
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = run_walk_forward(ticker="FAKE")
        assert result["error"] is not None
        assert result["mean_reversion"] is None
        assert result["momentum"] is None

    def test_insufficient_data_returns_error(self):
        from strategies.simple_backtest import run_walk_forward
        # Only 10 rows — far below train+test minimum
        hist = pd.DataFrame(
            {"Close": np.linspace(100, 110, 10)},
            index=pd.date_range("2024-01-01", periods=10, freq="B"),
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = run_walk_forward(ticker="TINY", train_window=252, test_window=63)
        assert result["error"] is not None

    def test_yfinance_exception_returns_error(self):
        from strategies.simple_backtest import run_walk_forward
        with patch("yfinance.Ticker", side_effect=RuntimeError("network down")):
            result = run_walk_forward(ticker="ERR")
        assert result["error"] is not None
        assert "network down" in result["error"]

    def test_result_structure_preserved_on_error(self):
        from strategies.simple_backtest import run_walk_forward
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = run_walk_forward(ticker="BADTICKER")
        for key in ("ticker", "train_window", "test_window", "mean_reversion", "momentum", "error"):
            assert key in result
