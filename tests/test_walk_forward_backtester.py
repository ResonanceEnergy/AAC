"""Tests for strategies/walk_forward_backtester.py — Walk-Forward OOS."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def price_df():
    """500-day trending + noisy price DataFrame."""
    np.random.seed(99)
    n = 500
    trend = np.linspace(0, 0.5, n)
    noise = np.cumsum(np.random.normal(0, 0.01, n))
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"price": 100 * np.exp(trend + noise)},
        index=dates,
    )


class TestWalkForwardResult:
    def test_dataclass(self):
        from strategies.walk_forward_backtester import WalkForwardResult
        wr = WalkForwardResult(
            window_mode="rolling",
            train_window=100,
            test_window=50,
            total_folds=5,
            oos_total_return=0.12,
            oos_sharpe=1.5,
            consistency_ratio=0.8,
            sharpe_std=0.3,
        )
        assert wr.oos_sharpe == 1.5
        assert wr.consistency_ratio == 0.8
        assert wr.total_folds == 5


class TestFoldResult:
    def test_dataclass(self):
        from strategies.walk_forward_backtester import FoldResult
        fr = FoldResult(
            fold_index=0,
            train_start="2024-01-01",
            train_end="2024-06-01",
            test_start="2024-06-01",
            test_end="2024-09-01",
            train_size=100,
            test_size=50,
            total_return=0.05,
            sharpe_ratio=1.2,
            max_drawdown=0.03,
            win_rate=0.55,
            profit_factor=1.5,
        )
        assert fr.total_return == 0.05
        assert fr.fold_index == 0


class TestWalkForwardBacktester:
    def test_basic_run(self, price_df):
        from strategies.walk_forward_backtester import (
            WalkForwardBacktester,
            MomentumWF,
        )
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        strat = MomentumWF()
        result = bt.run(price_df, strat)
        assert result.total_folds > 0
        assert len(result.folds) == result.total_folds

    def test_mean_reversion_strategy(self, price_df):
        from strategies.walk_forward_backtester import (
            WalkForwardBacktester,
            MeanReversionWF,
        )
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        strat = MeanReversionWF()
        result = bt.run(price_df, strat)
        assert result.total_folds > 0

    def test_run_with_callable(self, price_df):
        from strategies.walk_forward_backtester import WalkForwardBacktester

        def fit_fn(train_data):
            # Simple: just compute mean
            prices = train_data.iloc[:, 0]
            return {"mean": float(prices.pct_change().mean())}

        def predict_fn(test_data, model):
            prices = test_data.iloc[:, 0]
            returns = prices.pct_change().fillna(0.0)
            signals = pd.Series(0.0, index=returns.index)
            signals[returns > model["mean"]] = 1.0
            signals[returns < -model["mean"]] = -1.0
            return signals

        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run_with_callable(price_df, fit_fn, predict_fn)
        assert result.total_folds > 0

    def test_consistency_ratio(self, price_df):
        from strategies.walk_forward_backtester import (
            WalkForwardBacktester,
            MomentumWF,
        )
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run(price_df, MomentumWF())
        assert 0 <= result.consistency_ratio <= 1.0

    def test_expanding_window(self, price_df):
        from strategies.walk_forward_backtester import (
            WalkForwardBacktester,
            MomentumWF,
            WindowMode,
        )
        bt = WalkForwardBacktester(
            train_window=100,
            test_window=50,
            mode=WindowMode.EXPANDING,
        )
        result = bt.run(price_df, MomentumWF())
        assert result.total_folds > 0

    def test_short_series(self):
        from strategies.walk_forward_backtester import (
            WalkForwardBacktester,
            MomentumWF,
        )
        short = pd.DataFrame({"price": [100, 101, 102, 103, 104]})
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        with pytest.raises(ValueError, match="Need"):
            bt.run(short, MomentumWF())

    def test_fold_metrics(self, price_df):
        from strategies.walk_forward_backtester import (
            WalkForwardBacktester,
            MomentumWF,
        )
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run(price_df, MomentumWF())
        for fold in result.folds:
            assert fold.test_size > 0
            assert fold.train_size > 0

    def test_to_dict(self, price_df):
        from strategies.walk_forward_backtester import (
            WalkForwardBacktester,
            MomentumWF,
        )
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run(price_df, MomentumWF())
        d = result.to_dict()
        assert "oos_total_return" in d
        assert "total_folds" in d
