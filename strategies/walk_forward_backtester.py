"""Walk-Forward Backtester — rolling-window out-of-sample validation.

Replaces in-sample-only backtesters with proper train/test splits,
expanding and rolling windows, and anchored walk-forward analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types & data classes
# ---------------------------------------------------------------------------

class WindowMode(Enum):
    ROLLING = "rolling"      # fixed-size training window slides forward
    EXPANDING = "expanding"  # training window grows from anchor
    ANCHORED = "anchored"    # same as expanding (alias)


@dataclass
class FoldResult:
    """Result from a single walk-forward fold."""

    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    profit_factor: float = 0.0
    predictions: list[float] = field(default_factory=list)
    actuals: list[float] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated result across all walk-forward folds."""

    window_mode: str
    train_window: int
    test_window: int
    total_folds: int
    folds: list[FoldResult] = field(default_factory=list)

    # Aggregated out-of-sample metrics
    oos_total_return: float = 0.0
    oos_sharpe: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_win_rate: float = 0.0
    oos_avg_profit_factor: float = 0.0

    # Stability metrics
    sharpe_std: float = 0.0
    return_std: float = 0.0
    consistency_ratio: float = 0.0  # % of folds with positive return
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_mode": self.window_mode,
            "total_folds": self.total_folds,
            "oos_total_return": round(self.oos_total_return, 4),
            "oos_sharpe": round(self.oos_sharpe, 4),
            "oos_max_drawdown": round(self.oos_max_drawdown, 4),
            "oos_win_rate": round(self.oos_win_rate, 4),
            "consistency_ratio": round(self.consistency_ratio, 4),
            "sharpe_std": round(self.sharpe_std, 4),
        }


# ---------------------------------------------------------------------------
# Strategy Protocol
# ---------------------------------------------------------------------------

class WalkForwardStrategy:
    """Base class for strategies used in walk-forward testing.

    Subclass and implement ``fit()`` + ``predict()`` to plug into
    the walk-forward harness.
    """

    def fit(self, train_data: pd.DataFrame) -> None:
        """Learn parameters from training data."""
        raise NotImplementedError

    def predict(self, test_data: pd.DataFrame) -> pd.Series:
        """Generate signals on test data. Return Series of floats [-1, +1]."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Built-in example strategies
# ---------------------------------------------------------------------------

class MeanReversionWF(WalkForwardStrategy):
    """Simple mean-reversion strategy for demonstration."""

    def __init__(self, lookback: int = 20, z_entry: float = 1.5) -> None:
        self.lookback = lookback
        self.z_entry = z_entry
        self._mean: float = 0.0
        self._std: float = 0.0

    def fit(self, train_data: pd.DataFrame) -> None:
        prices = train_data.iloc[:, 0] if isinstance(train_data, pd.DataFrame) else train_data
        returns = prices.pct_change().dropna()
        self._mean = float(returns.mean())
        self._std = float(returns.std())

    def predict(self, test_data: pd.DataFrame) -> pd.Series:
        prices = test_data.iloc[:, 0] if isinstance(test_data, pd.DataFrame) else test_data
        returns = prices.pct_change().fillna(0.0)
        if self._std < 1e-10:
            return pd.Series(0.0, index=returns.index)
        z_scores = (returns - self._mean) / self._std
        signals = pd.Series(0.0, index=returns.index)
        signals[z_scores < -self.z_entry] = 1.0   # buy on dip
        signals[z_scores > self.z_entry] = -1.0   # sell on spike
        return signals


class MomentumWF(WalkForwardStrategy):
    """Simple momentum strategy for demonstration."""

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback
        self._threshold: float = 0.0

    def fit(self, train_data: pd.DataFrame) -> None:
        prices = train_data.iloc[:, 0] if isinstance(train_data, pd.DataFrame) else train_data
        returns = prices.pct_change().dropna()
        self._threshold = float(returns.std()) * 0.5

    def predict(self, test_data: pd.DataFrame) -> pd.Series:
        prices = test_data.iloc[:, 0] if isinstance(test_data, pd.DataFrame) else test_data
        mom = prices.pct_change(self.lookback).fillna(0.0)
        signals = pd.Series(0.0, index=mom.index)
        signals[mom > self._threshold] = 1.0
        signals[mom < -self._threshold] = -1.0
        return signals


# ---------------------------------------------------------------------------
# Walk-Forward Backtester
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    """Rolling-window out-of-sample backtester.

    Parameters
    ----------
    train_window : int
        Number of trading days in each training fold.
    test_window : int
        Number of trading days in each test fold.
    mode : WindowMode
        ROLLING (fixed train size) or EXPANDING (growing train size).
    step_size : int
        How many days to slide forward between folds.
        Defaults to ``test_window`` (non-overlapping test sets).
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        mode: WindowMode = WindowMode.ROLLING,
        step_size: Optional[int] = None,
    ) -> None:
        self.train_window = train_window
        self.test_window = test_window
        self.mode = mode
        self.step_size = step_size or test_window

    def run(
        self,
        data: pd.DataFrame,
        strategy: WalkForwardStrategy,
        price_col: Optional[str] = None,
    ) -> WalkForwardResult:
        """Execute walk-forward backtest.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain price data sorted by date.
        strategy : WalkForwardStrategy
            Strategy with ``fit()`` and ``predict()`` methods.
        price_col : str, optional
            Column name for the target price series.
            If None, uses first column.
        """
        if price_col and price_col in data.columns:
            prices = data[[price_col]]
        else:
            prices = data.iloc[:, :1] if data.shape[1] > 0 else data

        n = len(prices)
        min_data = self.train_window + self.test_window
        if n < min_data:
            raise ValueError(
                f"Need ≥{min_data} rows (train={self.train_window} + "
                f"test={self.test_window}), got {n}"
            )

        folds: list[FoldResult] = []
        fold_idx = 0
        pos = 0

        while pos + self.train_window + self.test_window <= n:
            if self.mode == WindowMode.ROLLING:
                train_start = pos
                train_end = pos + self.train_window
            else:
                # Expanding / Anchored
                train_start = 0
                train_end = pos + self.train_window

            test_start = train_end
            test_end = min(test_start + self.test_window, n)

            train_data = prices.iloc[train_start:train_end]
            test_data = prices.iloc[test_start:test_end]

            # Fit on train, predict on test
            strategy.fit(train_data)
            signals = strategy.predict(test_data)

            # Convert signals to returns
            fold_result = self._evaluate_fold(
                fold_idx, train_data, test_data, signals,
            )
            folds.append(fold_result)

            fold_idx += 1
            pos += self.step_size

        result = self._aggregate(folds)
        _log.info(
            "walk_forward_complete",
            folds=len(folds),
            oos_sharpe=round(result.oos_sharpe, 3),
            consistency=round(result.consistency_ratio, 3),
        )
        return result

    def run_with_callable(
        self,
        data: pd.DataFrame,
        fit_fn: Callable[[pd.DataFrame], Any],
        predict_fn: Callable[[pd.DataFrame, Any], pd.Series],
        price_col: Optional[str] = None,
    ) -> WalkForwardResult:
        """Walk-forward with function-based strategy (no subclassing).

        Parameters
        ----------
        fit_fn : callable
            Takes training DataFrame, returns model/state object.
        predict_fn : callable
            Takes test DataFrame + model, returns signal Series.
        """

        class _FnStrategy(WalkForwardStrategy):
            def __init__(self) -> None:
                self._model: Any = None

            def fit(self, train_data: pd.DataFrame) -> None:
                self._model = fit_fn(train_data)

            def predict(self, test_data: pd.DataFrame) -> pd.Series:
                if self._model is None:
                    raise RuntimeError("Call fit() before predict()")
                return predict_fn(test_data, self._model)

        return self.run(data, _FnStrategy(), price_col=price_col)

    # ── Fold evaluation ───────────────────────────────────────────────────

    def _evaluate_fold(
        self,
        fold_idx: int,
        train: pd.DataFrame,
        test: pd.DataFrame,
        signals: pd.Series,
    ) -> FoldResult:
        returns = test.iloc[:, 0].pct_change().fillna(0.0)
        strat_returns = returns * signals.reindex(returns.index).fillna(0.0)

        cum = (1 + strat_returns).cumprod()
        total_ret = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0.0

        # Sharpe (annualized)
        mean_r = float(strat_returns.mean())
        std_r = float(strat_returns.std())
        sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 1e-10 else 0.0

        # Max drawdown
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max.replace(0, 1e-10)
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0

        # Win rate from daily returns
        trades = strat_returns[strat_returns != 0]
        n_trades = len(trades)
        winners = int((trades > 0).sum())
        losers = int((trades < 0).sum())
        win_rate = winners / n_trades if n_trades > 0 else 0.0

        gross_profit = float(trades[trades > 0].sum()) if winners > 0 else 0.0
        gross_loss = abs(float(trades[trades < 0].sum())) if losers > 0 else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        ts = lambda idx: str(idx[0]) if len(idx) > 0 else ""

        return FoldResult(
            fold_index=fold_idx,
            train_start=ts(train.index),
            train_end=str(train.index[-1]) if len(train) > 0 else "",
            test_start=ts(test.index),
            test_end=str(test.index[-1]) if len(test) > 0 else "",
            train_size=len(train),
            test_size=len(test),
            total_return=total_ret,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            n_trades=n_trades,
            profit_factor=pf if pf != float("inf") else 99.0,
            predictions=signals.tolist(),
            actuals=returns.tolist(),
        )

    # ── Aggregation ───────────────────────────────────────────────────────

    def _aggregate(self, folds: list[FoldResult]) -> WalkForwardResult:
        if not folds:
            return WalkForwardResult(
                window_mode=self.mode.value,
                train_window=self.train_window,
                test_window=self.test_window,
                total_folds=0,
            )

        returns = [f.total_return for f in folds]
        sharpes = [f.sharpe_ratio for f in folds]
        drawdowns = [f.max_drawdown for f in folds]

        # Chain fold returns to get total OOS return
        cum = 1.0
        for r in returns:
            cum *= (1 + r)
        oos_total = cum - 1.0

        positive_folds = sum(1 for r in returns if r > 0)
        consistency = positive_folds / len(folds)

        total_trades = sum(f.n_trades for f in folds)
        total_winners = sum(int(f.win_rate * f.n_trades) for f in folds)
        win_rate = total_winners / total_trades if total_trades > 0 else 0.0

        pf_values = [f.profit_factor for f in folds if f.profit_factor < 99.0]
        avg_pf = float(np.mean(pf_values)) if pf_values else 0.0

        return WalkForwardResult(
            window_mode=self.mode.value,
            train_window=self.train_window,
            test_window=self.test_window,
            total_folds=len(folds),
            folds=folds,
            oos_total_return=oos_total,
            oos_sharpe=float(np.mean(sharpes)),
            oos_max_drawdown=float(min(drawdowns)),
            oos_win_rate=win_rate,
            oos_avg_profit_factor=avg_pf,
            sharpe_std=float(np.std(sharpes)),
            return_std=float(np.std(returns)),
            consistency_ratio=consistency,
        )
