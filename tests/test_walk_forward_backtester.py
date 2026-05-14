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


# ---------------------------------------------------------------------------
# Sprint 24 additions — comprehensive coverage
# ---------------------------------------------------------------------------

class TestWindowMode:
    def test_rolling_value(self):
        from strategies.walk_forward_backtester import WindowMode
        assert WindowMode.ROLLING.value == "rolling"

    def test_expanding_value(self):
        from strategies.walk_forward_backtester import WindowMode
        assert WindowMode.EXPANDING.value == "expanding"

    def test_anchored_value(self):
        from strategies.walk_forward_backtester import WindowMode
        assert WindowMode.ANCHORED.value == "anchored"

    def test_all_modes_exist(self):
        from strategies.walk_forward_backtester import WindowMode
        assert {WindowMode.ROLLING, WindowMode.EXPANDING, WindowMode.ANCHORED}


class TestWalkForwardResultDefaults:
    def test_folds_default_empty(self):
        from strategies.walk_forward_backtester import WalkForwardResult
        wr = WalkForwardResult(
            window_mode="rolling",
            train_window=100,
            test_window=50,
            total_folds=0,
        )
        assert wr.folds == []

    def test_oos_metrics_default_zero(self):
        from strategies.walk_forward_backtester import WalkForwardResult
        wr = WalkForwardResult(
            window_mode="rolling",
            train_window=100,
            test_window=50,
            total_folds=0,
        )
        assert wr.oos_total_return == 0.0
        assert wr.oos_sharpe == 0.0
        assert wr.oos_win_rate == 0.0

    def test_to_dict_keys(self):
        from strategies.walk_forward_backtester import WalkForwardResult
        wr = WalkForwardResult(
            window_mode="rolling",
            train_window=100,
            test_window=50,
            total_folds=2,
            oos_sharpe=1.2,
            consistency_ratio=0.5,
        )
        d = wr.to_dict()
        for key in ("window_mode", "train_window", "test_window", "total_folds",
                    "oos_sharpe", "consistency_ratio", "oos_total_return",
                    "oos_max_drawdown", "oos_win_rate", "sharpe_std", "return_std"):
            assert key in d, f"missing key: {key}"

    def test_to_dict_values_round_trip(self):
        from strategies.walk_forward_backtester import WalkForwardResult
        wr = WalkForwardResult(
            window_mode="expanding",
            train_window=200,
            test_window=40,
            total_folds=3,
            oos_total_return=0.07,
            oos_sharpe=0.9,
        )
        d = wr.to_dict()
        assert d["window_mode"] == "expanding"
        assert d["oos_total_return"] == pytest.approx(0.07)


class TestFoldResultDefaults:
    def test_predictions_default_empty(self):
        from strategies.walk_forward_backtester import FoldResult
        fr = FoldResult(
            fold_index=0,
            train_start="2024-01-01",
            train_end="2024-06-01",
            test_start="2024-06-02",
            test_end="2024-09-01",
            train_size=100,
            test_size=50,
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=1.0,
        )
        assert fr.predictions == []
        assert fr.actuals == []

    def test_n_trades_default_zero(self):
        from strategies.walk_forward_backtester import FoldResult
        fr = FoldResult(
            fold_index=1,
            train_start="2024-01-01",
            train_end="2024-06-01",
            test_start="2024-06-02",
            test_end="2024-09-01",
            train_size=100,
            test_size=50,
            total_return=0.01,
            sharpe_ratio=0.5,
            max_drawdown=-0.02,
            win_rate=0.5,
            profit_factor=1.2,
        )
        assert fr.n_trades == 0


class TestMeanReversionWF:
    def test_fit_sets_mean_std(self):
        from strategies.walk_forward_backtester import MeanReversionWF
        np.random.seed(7)
        prices = pd.DataFrame({"price": 100 + np.cumsum(np.random.normal(0, 0.5, 60))})
        strat = MeanReversionWF(lookback=20)
        strat.fit(prices)
        assert isinstance(strat._mean, float)
        assert strat._std >= 0.0

    def test_predict_returns_series(self):
        from strategies.walk_forward_backtester import MeanReversionWF
        np.random.seed(7)
        prices = pd.DataFrame({"price": 100 + np.cumsum(np.random.normal(0, 0.5, 60))})
        strat = MeanReversionWF()
        strat.fit(prices)
        sig = strat.predict(prices)
        assert isinstance(sig, pd.Series)
        assert len(sig) == len(prices)

    def test_predict_values_in_range(self):
        from strategies.walk_forward_backtester import MeanReversionWF
        np.random.seed(7)
        prices = pd.DataFrame({"price": 100 + np.cumsum(np.random.normal(0, 1.0, 100))})
        strat = MeanReversionWF(lookback=20, z_entry=1.5)
        strat.fit(prices)
        sig = strat.predict(prices)
        assert set(sig.unique()).issubset({-1.0, 0.0, 1.0})

    def test_zero_std_returns_all_zeros(self):
        from strategies.walk_forward_backtester import MeanReversionWF
        flat = pd.DataFrame({"price": [100.0] * 50})
        strat = MeanReversionWF()
        strat.fit(flat)
        sig = strat.predict(flat)
        assert (sig == 0.0).all()


class TestMomentumWF:
    def test_fit_sets_threshold(self):
        from strategies.walk_forward_backtester import MomentumWF
        np.random.seed(13)
        prices = pd.DataFrame({"price": 100 + np.cumsum(np.random.normal(0.05, 0.5, 80))})
        strat = MomentumWF(lookback=20)
        strat.fit(prices)
        assert strat._threshold >= 0.0

    def test_predict_returns_series(self):
        from strategies.walk_forward_backtester import MomentumWF
        np.random.seed(13)
        prices = pd.DataFrame({"price": 100 + np.cumsum(np.random.normal(0.05, 0.5, 80))})
        strat = MomentumWF(lookback=20)
        strat.fit(prices)
        sig = strat.predict(prices)
        assert isinstance(sig, pd.Series)
        assert len(sig) == len(prices)

    def test_predict_signals_in_range(self):
        from strategies.walk_forward_backtester import MomentumWF
        np.random.seed(13)
        prices = pd.DataFrame({"price": 100 + np.cumsum(np.random.normal(0.05, 0.5, 80))})
        strat = MomentumWF(lookback=20)
        strat.fit(prices)
        sig = strat.predict(prices)
        assert set(sig.unique()).issubset({-1.0, 0.0, 1.0})


class TestWalkForwardBacktesterExtended:
    def test_fold_count_rolling(self, price_df):
        from strategies.walk_forward_backtester import (
            WalkForwardBacktester,
            MomentumWF,
            WindowMode,
        )
        bt = WalkForwardBacktester(
            train_window=100, test_window=50, mode=WindowMode.ROLLING
        )
        result = bt.run(price_df, MomentumWF())
        # With 500 rows, 100 train + 50 test = 150 min; step=50 → ⌊(500-150)/50⌋+1
        assert result.total_folds >= 1

    def test_anchored_mode(self, price_df):
        from strategies.walk_forward_backtester import (
            WalkForwardBacktester,
            MeanReversionWF,
            WindowMode,
        )
        bt = WalkForwardBacktester(
            train_window=100, test_window=50, mode=WindowMode.ANCHORED
        )
        result = bt.run(price_df, MeanReversionWF())
        assert result.window_mode == "anchored"
        assert result.total_folds > 0

    def test_custom_step_size(self, price_df):
        from strategies.walk_forward_backtester import WalkForwardBacktester, MomentumWF
        bt_default = WalkForwardBacktester(train_window=100, test_window=50)
        bt_custom = WalkForwardBacktester(train_window=100, test_window=50, step_size=25)
        r_default = bt_default.run(price_df, MomentumWF())
        r_custom = bt_custom.run(price_df, MomentumWF())
        # Smaller step → more folds
        assert r_custom.total_folds >= r_default.total_folds

    def test_price_col_selection(self, price_df):
        from strategies.walk_forward_backtester import WalkForwardBacktester, MomentumWF
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run(price_df, MomentumWF(), price_col="price")
        assert result.total_folds > 0

    def test_price_col_missing_uses_first(self, price_df):
        from strategies.walk_forward_backtester import WalkForwardBacktester, MomentumWF
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        # nonexistent col → silently falls back to first column
        result = bt.run(price_df, MomentumWF(), price_col="nonexistent")
        assert result.total_folds > 0

    def test_oos_max_drawdown_nonpositive(self, price_df):
        from strategies.walk_forward_backtester import WalkForwardBacktester, MomentumWF
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run(price_df, MomentumWF())
        assert result.oos_max_drawdown <= 0.0

    def test_sharpe_std_nonnegative(self, price_df):
        from strategies.walk_forward_backtester import WalkForwardBacktester, MomentumWF
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run(price_df, MomentumWF())
        assert result.sharpe_std >= 0.0

    def test_return_std_nonnegative(self, price_df):
        from strategies.walk_forward_backtester import WalkForwardBacktester, MomentumWF
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run(price_df, MomentumWF())
        assert result.return_std >= 0.0

    def test_folds_list_matches_count(self, price_df):
        from strategies.walk_forward_backtester import WalkForwardBacktester, MomentumWF
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run(price_df, MomentumWF())
        assert len(result.folds) == result.total_folds

    def test_fold_indices_sequential(self, price_df):
        from strategies.walk_forward_backtester import WalkForwardBacktester, MomentumWF
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run(price_df, MomentumWF())
        for i, fold in enumerate(result.folds):
            assert fold.fold_index == i

    def test_fold_train_size_equals_train_window(self, price_df):
        from strategies.walk_forward_backtester import (
            WalkForwardBacktester,
            MomentumWF,
            WindowMode,
        )
        bt = WalkForwardBacktester(
            train_window=100, test_window=50, mode=WindowMode.ROLLING
        )
        result = bt.run(price_df, MomentumWF())
        for fold in result.folds:
            assert fold.train_size == 100

    def test_empty_folds_returns_zero_result(self):
        from strategies.walk_forward_backtester import WalkForwardBacktester, MomentumWF
        # Exactly min data → might produce 1 fold or 0 depending on boundary
        # Provide exactly min: should raise OR produce ≥0 folds without error
        bt = WalkForwardBacktester(train_window=50, test_window=20)
        too_few = pd.DataFrame({"price": np.linspace(100, 120, 69)})  # 69 < 70 min
        with pytest.raises(ValueError):
            bt.run(too_few, MomentumWF())

    def test_win_rate_bounded(self, price_df):
        from strategies.walk_forward_backtester import WalkForwardBacktester, MomentumWF
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run(price_df, MomentumWF())
        assert 0.0 <= result.oos_win_rate <= 1.0

    def test_callable_without_subclass(self, price_df):
        from strategies.walk_forward_backtester import WalkForwardBacktester

        model_store = {}

        def fit_fn(train):
            model_store["threshold"] = float(train.iloc[:, 0].pct_change().std()) * 0.3
            return model_store.copy()

        def predict_fn(test, model):
            returns = test.iloc[:, 0].pct_change().fillna(0.0)
            sig = pd.Series(0.0, index=returns.index)
            sig[returns > model["threshold"]] = 1.0
            sig[returns < -model["threshold"]] = -1.0
            return sig

        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run_with_callable(price_df, fit_fn, predict_fn)
        assert result.total_folds > 0

    def test_aggregate_empty_folds(self):
        from strategies.walk_forward_backtester import WalkForwardBacktester
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        empty_result = bt._aggregate([])
        assert empty_result.total_folds == 0
        assert empty_result.oos_sharpe == 0.0

    def test_profit_factor_inf_capped(self, price_df):
        from strategies.walk_forward_backtester import WalkForwardBacktester, MomentumWF
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run(price_df, MomentumWF())
        for fold in result.folds:
            assert fold.profit_factor < float("inf")

    def test_predictions_stored_in_fold(self, price_df):
        from strategies.walk_forward_backtester import WalkForwardBacktester, MomentumWF
        bt = WalkForwardBacktester(train_window=100, test_window=50)
        result = bt.run(price_df, MomentumWF())
        fold = result.folds[0]
        assert isinstance(fold.predictions, list)
        assert len(fold.predictions) == fold.test_size
