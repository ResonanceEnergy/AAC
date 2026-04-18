from __future__ import annotations

"""LDE Backtest — test doctrine signals against historical market data.

Uses yfinance for price history (our primary free data source).
Evaluates how doctrine signals would have performed as trading signals.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

_log = structlog.get_logger(__name__)


@dataclass
class BacktestResult:
    """Results from a doctrine backtest run."""

    ticker: str
    period: str
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trade_log: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Position:
    """Internal position tracker."""

    direction: int  # +1 long, -1 short, 0 flat
    entry_price: float = 0.0
    entry_date: str = ""
    size: float = 1.0


def backtest_doctrine_signal(
    signal_series: list[float],
    dates: list[str],
    prices: list[float],
    *,
    ticker: str = "SPY",
    signal_threshold: float = 0.3,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.10,
) -> BacktestResult:
    """Backtest a doctrine signal series against price data.

    Parameters
    ----------
    signal_series : list[float]
        Doctrine signal values [-1, 1] for each date.
    dates : list[str]
        Date strings (YYYY-MM-DD) corresponding to signals.
    prices : list[float]
        Close prices for each date.
    ticker : str
        Ticker symbol for reporting.
    signal_threshold : float
        Minimum absolute signal for entry (default 0.3).
    stop_loss_pct : float
        Stop loss as fraction (0.05 = 5%).
    take_profit_pct : float
        Take profit as fraction (0.10 = 10%).

    Returns
    -------
    BacktestResult
    """
    n = min(len(signal_series), len(dates), len(prices))
    if n < 2:
        return BacktestResult(ticker=ticker, period="N/A",
                              details={"error": "insufficient data"})

    trades: list[dict[str, Any]] = []
    pos = _Position(direction=0)
    equity = [1.0]

    for i in range(1, n):
        sig = signal_series[i]
        price = prices[i]
        prev_price = prices[i - 1]

        # Update equity if in a position
        if pos.direction != 0:
            pnl_pct = (price - prev_price) / prev_price * pos.direction
            equity.append(equity[-1] * (1.0 + pnl_pct))

            # Check stop loss / take profit
            entry_pnl = (price - pos.entry_price) / pos.entry_price * pos.direction
            if entry_pnl <= -stop_loss_pct or entry_pnl >= take_profit_pct:
                trades.append({
                    "entry_date": pos.entry_date,
                    "exit_date": dates[i],
                    "direction": "long" if pos.direction > 0 else "short",
                    "entry_price": pos.entry_price,
                    "exit_price": price,
                    "pnl_pct": round(entry_pnl * 100, 2),
                    "exit_reason": "stop_loss" if entry_pnl <= -stop_loss_pct else "take_profit",
                })
                pos = _Position(direction=0)
        else:
            equity.append(equity[-1])

        # Entry signals
        if pos.direction == 0:
            if sig >= signal_threshold:
                pos = _Position(direction=1, entry_price=price,
                                entry_date=dates[i])
            elif sig <= -signal_threshold:
                pos = _Position(direction=-1, entry_price=price,
                                entry_date=dates[i])

    # Close any open position at end
    if pos.direction != 0 and n > 1:
        entry_pnl = (prices[-1] - pos.entry_price) / pos.entry_price * pos.direction
        trades.append({
            "entry_date": pos.entry_date,
            "exit_date": dates[-1],
            "direction": "long" if pos.direction > 0 else "short",
            "entry_price": pos.entry_price,
            "exit_price": prices[-1],
            "pnl_pct": round(entry_pnl * 100, 2),
            "exit_reason": "end_of_data",
        })

    # Compute metrics
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    total = len(trades)

    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))

    total_return = (equity[-1] / equity[0] - 1.0) if equity[0] > 0 else 0.0

    # Annualize
    if n > 252:
        ann_factor = 252.0 / n
    else:
        ann_factor = 252.0 / max(n, 1)
    annualized = (1.0 + total_return) ** ann_factor - 1.0

    # Daily returns for Sharpe / Sortino
    daily_returns = []
    for j in range(1, len(equity)):
        if equity[j - 1] > 0:
            daily_returns.append(equity[j] / equity[j - 1] - 1.0)

    sharpe = _sharpe(daily_returns)
    sortino = _sortino(daily_returns)
    mdd = _max_drawdown(equity)

    period = f"{dates[0]} to {dates[-1]}" if dates else "N/A"

    return BacktestResult(
        ticker=ticker,
        period=period,
        total_return=round(total_return * 100, 2),
        annualized_return=round(annualized * 100, 2),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        max_drawdown=round(mdd * 100, 2),
        win_rate=round(len(wins) / max(total, 1) * 100, 1),
        profit_factor=round(gross_profit / max(gross_loss, 0.01), 2),
        total_trades=total,
        winning_trades=len(wins),
        losing_trades=len(losses),
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        avg_win=round(gross_profit / max(len(wins), 1), 2),
        avg_loss=round(gross_loss / max(len(losses), 1), 2),
        trade_log=trades,
        details={
            "signal_threshold": signal_threshold,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "data_points": n,
        },
    )


def backtest_against_market(
    signal_series: list[float],
    *,
    ticker: str = "SPY",
    lookback_days: int = 90,
    signal_threshold: float = 0.3,
) -> BacktestResult:
    """Backtest doctrine signals against a ticker using yfinance.

    Fetches historical data and runs the backtest.
    """
    try:
        import yfinance as yf
    except ImportError:
        _log.error("yfinance not installed")
        return BacktestResult(ticker=ticker, period="N/A",
                              details={"error": "yfinance not installed"})

    period_str = f"{lookback_days}d"
    hist = yf.Ticker(ticker).history(period=period_str)

    if hist.empty:
        return BacktestResult(ticker=ticker, period="N/A",
                              details={"error": f"no data for {ticker}"})

    dates = [d.strftime("%Y-%m-%d") for d in hist.index]
    prices = hist["Close"].tolist()

    # Pad or trim signal to match price data
    n_prices = len(prices)
    if len(signal_series) < n_prices:
        # Repeat last signal to fill
        padded = signal_series + [signal_series[-1] if signal_series else 0.0] * (
            n_prices - len(signal_series)
        )
    else:
        padded = signal_series[-n_prices:]

    return backtest_doctrine_signal(
        padded, dates, prices,
        ticker=ticker,
        signal_threshold=signal_threshold,
    )


# ============================================================================
# METRICS
# ============================================================================

def _sharpe(returns: list[float], risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio from daily returns."""
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns) - risk_free / 252
    std_r = _std(returns)
    if std_r < 1e-9:
        return 0.0
    return (mean_r / std_r) * math.sqrt(252)


def _sortino(returns: list[float], risk_free: float = 0.0) -> float:
    """Annualized Sortino ratio from daily returns."""
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns) - risk_free / 252
    downside = [r for r in returns if r < 0]
    if not downside:
        return 10.0  # no downside → very good
    down_std = _std(downside)
    if down_std < 1e-9:
        return 0.0
    return (mean_r / down_std) * math.sqrt(252)


def _std(values: list[float]) -> float:
    """Sample standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(var, 0.0))


def _max_drawdown(equity: list[float]) -> float:
    """Maximum drawdown from equity curve."""
    if not equity:
        return 0.0
    peak = equity[0]
    mdd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd
