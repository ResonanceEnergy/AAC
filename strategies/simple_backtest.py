from __future__ import annotations
"""strategies/simple_backtest.py — 90-day signal win-rate comparison (Sprint 6).

Compares three signal strategies over the trailing 90 trading days using
yfinance historical close prices.

Method (honest and simple)
──────────────────────────
For each strategy, a "signal day" is identified using only data available
up to that day (walk-forward, no lookahead bias):

  War Room proxy:
      Day's 30-day realised HV of SPY > 0.18 (≈ elevated regime threshold).
      This mimics the CRISIS/ELEVATED branch without needing FRED/macro live
      data in the past.

  Vol Premium:
      30-day realised HV of the target ticker / HV_FLOOR > _IV_HV_THRESHOLD.
      Since historical IV is not available in free tier, realised HV alone
      is used as the signal proxy (high HV → options were likely expensive).

  Combined:
      Signal fires when EITHER strategy fires (OR-union).

Win condition:
      Within 10 trading days after the signal, the underlying closed ≥ 2%
      BELOW the signal-day close (put goes in-the-money proxy).

Output: BacktestReport with per-strategy StrategyPerformance rows.
Never raises; returns an empty report on any failure.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

_BACKTEST_TICKERS: list[str] = ["SPY", "QQQ", "IWM", "HYG", "JNK"]
_HOLD_DAYS = 10          # check for win within this many trading days
_WIN_THRESHOLD = 0.02    # underlying must fall ≥ 2 % for a "win"
_HV_WINDOW = 30          # rolling window (trading days) for HV calculation
_WR_HV_THRESHOLD = 0.18  # SPY HV threshold mimicking ELEVATED+ regime
_VP_HV_THRESHOLD = 1.20  # vol premium proxy: ticker HV / floor ratio
_VP_HV_FLOOR = 0.12      # floor to avoid division issues


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """Single simulated put outcome."""

    ticker: str
    strategy: str
    signal_date: str
    entry_price: float
    min_price_in_window: float   # lowest close in the hold window
    max_drawdown_pct: float      # (entry - min) / entry; positive = fell
    win: bool                    # underlying fell ≥ _WIN_THRESHOLD


@dataclass
class StrategyPerformance:
    """Aggregated stats for one strategy over the backtest window."""

    strategy: str
    n_signals: int
    n_wins: int
    win_rate: float
    avg_drawdown_pct: float   # average max drawdown on signal days
    notes: str = ""


@dataclass
class BacktestReport:
    """Comparison of strategies over the backtest window."""

    window_days: int
    start_date: str
    end_date: str
    results: list[StrategyPerformance] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_days": self.window_days,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "strategies": [
                {
                    "strategy": r.strategy,
                    "n_signals": r.n_signals,
                    "n_wins": r.n_wins,
                    "win_rate": round(r.win_rate, 3),
                    "avg_drawdown_pct": round(r.avg_drawdown_pct * 100, 2),
                    "notes": r.notes,
                }
                for r in self.results
            ],
        }

    def format_report(self) -> str:
        lines = [
            f"  90-Day Backtest: {self.start_date} → {self.end_date}",
            "  " + "─" * 63,
            f"  {'Strategy':<22} {'Signals':>8} {'Wins':>6} {'Win%':>7} {'AvgDrop':>9}",
            "  " + "─" * 63,
        ]
        for r in self.results:
            lines.append(
                f"  {r.strategy:<22} {r.n_signals:>8} {r.n_wins:>6} "
                f"{r.win_rate * 100:>6.1f}% {r.avg_drawdown_pct * 100:>+8.1f}%"
            )
        lines.append("  " + "─" * 63)
        return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _rolling_hv(closes: list[float], window: int = _HV_WINDOW) -> list[float]:
    """Return annualised 30-day rolling HV aligned to ``closes`` (same length).

    The first ``window`` entries are 0.0 (insufficient history).
    """
    arr = np.array(closes, dtype=float)
    log_ret = np.concatenate([[0.0], np.diff(np.log(arr))])
    hvs: list[float] = []
    for i in range(len(arr)):
        if i < window:
            hvs.append(0.0)
        else:
            segment = log_ret[i - window + 1: i + 1]
            hvs.append(float(np.std(segment, ddof=1) * np.sqrt(252)))
    return hvs


def _fetch_history(ticker: str, days: int = 130) -> tuple[list[date], list[float]]:
    """Fetch closing prices and dates for a ticker via yfinance.

    Returns (dates, closes) or ([], []) on failure.
    """
    try:
        import yfinance as yf  # noqa: PLC0415

        period = f"{days}d"
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return [], []
        dates = [d.date() if hasattr(d, "date") else d for d in hist.index]
        closes = list(hist["Close"].dropna().values)
        min_len = min(len(dates), len(closes))
        return list(dates[:min_len]), list(closes[:min_len])
    except Exception as exc:
        _log.debug("History fetch failed for %s: %s", ticker, exc)
        return [], []


def _evaluate_trades(
    ticker: str,
    strategy: str,
    dates: list[date],
    closes: list[float],
    signal_mask: list[bool],
    hold_days: int = _HOLD_DAYS,
    win_threshold: float = _WIN_THRESHOLD,
) -> list[BacktestTrade]:
    """For each True in signal_mask, evaluate the put outcome."""
    trades: list[BacktestTrade] = []
    n = len(dates)

    for i, fired in enumerate(signal_mask):
        if not fired:
            continue
        entry = closes[i]
        future_end = min(i + hold_days + 1, n)
        future_closes = closes[i + 1: future_end]
        if not future_closes:
            continue

        min_price = min(future_closes)
        drawdown = (entry - min_price) / entry  # positive = underlying fell

        trades.append(BacktestTrade(
            ticker=ticker,
            strategy=strategy,
            signal_date=str(dates[i]),
            entry_price=round(entry, 2),
            min_price_in_window=round(min_price, 2),
            max_drawdown_pct=round(drawdown, 4),
            win=drawdown >= win_threshold,
        ))

    return trades


# ── Public API ────────────────────────────────────────────────────────────────

def run_walk_forward(
    ticker: str = "SPY",
    lookback_days: int = 500,
    train_window: int = 252,
    test_window: int = 63,
) -> dict[str, Any]:
    """Run walk-forward backtests on ``ticker`` using both built-in strategies.

    Downloads ``lookback_days`` of daily close history for ``ticker`` via
    yfinance, then runs ``WalkForwardBacktester`` with both ``MeanReversionWF``
    and ``MomentumWF`` strategies.  Returns a combined summary dict containing
    OOS metrics for each strategy.

    Args:
        ticker:       Ticker to fetch history for.
        lookback_days: Number of calendar days of history to request.
                       Must provide at least ``train_window + test_window`` trading days.
        train_window: Number of training days per fold (default 252 = 1 year).
        test_window:  Number of test days per fold (default 63 = 1 quarter).

    Returns:
        Dict with keys ``"ticker"``, ``"train_window"``, ``"test_window"``,
        ``"mean_reversion"`` and ``"momentum"`` (each a ``WalkForwardResult.to_dict()``),
        and ``"error"`` (None on success, message string on failure).
    """
    result: dict[str, Any] = {
        "ticker": ticker,
        "train_window": train_window,
        "test_window": test_window,
        "mean_reversion": None,
        "momentum": None,
        "error": None,
    }
    try:
        import yfinance as yf  # noqa: PLC0415
        from strategies.walk_forward_backtester import (  # noqa: PLC0415
            MeanReversionWF,
            MomentumWF,
            WalkForwardBacktester,
        )

        hist = yf.Ticker(ticker).history(period=f"{lookback_days}d")
        if hist.empty:
            result["error"] = f"No history returned for {ticker}"
            return result

        prices = hist[["Close"]].dropna().rename(columns={"Close": "price"})

        bt = WalkForwardBacktester(
            train_window=train_window,
            test_window=test_window,
        )

        mr_result = bt.run(prices, MeanReversionWF())
        mom_result = bt.run(prices, MomentumWF())

        result["mean_reversion"] = mr_result.to_dict()
        result["momentum"] = mom_result.to_dict()

        _log.info(
            "walk_forward_backtest_complete",
            ticker=ticker,
            mr_sharpe=round(mr_result.oos_sharpe, 3),
            mom_sharpe=round(mom_result.oos_sharpe, 3),
            folds=mr_result.total_folds,
        )
    except Exception as exc:
        _log.warning("run_walk_forward_failed ticker=%s: %s", ticker, exc)
        result["error"] = str(exc)

    return result


def run_backtest(
    tickers: list[str] | None = None,
    lookback_days: int = 130,
    hold_days: int = _HOLD_DAYS,
    win_threshold: float = _WIN_THRESHOLD,
) -> BacktestReport:
    """Run 90-day win-rate comparison across three strategy proxies.

    Args:
        tickers:       Override default universe.
        lookback_days: Days of history to download (≥ 100 recommended).
        hold_days:     Trading days after signal to check for win.
        win_threshold: Minimum underlying decline to count as a win.

    Returns:
        BacktestReport with StrategyPerformance for each strategy.
    """
    universe = tickers or _BACKTEST_TICKERS
    all_trades: list[BacktestTrade] = []

    # Fetch SPY HV for War Room proxy
    spy_dates, spy_closes = _fetch_history("SPY", lookback_days)
    spy_hv = _rolling_hv(spy_closes) if spy_closes else []

    # Restrict backtest window to last 90 trading days (leaving room for hold)
    window_start_offset = max(0, len(spy_dates) - 90 - hold_days) if spy_dates else 0
    start_date = str(spy_dates[window_start_offset]) if spy_dates else "N/A"
    end_date = str(spy_dates[-(hold_days + 1)]) if len(spy_dates) > hold_days + 1 else "N/A"

    for ticker in universe:
        if ticker == "SPY":
            dates, closes = spy_dates, spy_closes
            hv = spy_hv
        else:
            dates, closes = _fetch_history(ticker, lookback_days)
            hv = _rolling_hv(closes) if closes else []

        if not closes or not hv:
            continue

        # Restrict to the backtest window (avoid last hold_days — no future data)
        n = len(closes)
        backtest_end = n - hold_days
        if backtest_end <= window_start_offset:
            continue

        b_dates = dates[window_start_offset:backtest_end]
        b_closes = closes[window_start_offset:backtest_end]
        b_hv = hv[window_start_offset:backtest_end]

        # War Room proxy: signal when SPY HV elevated (use spy_hv aligned to same indices)
        spy_hv_slice = spy_hv[window_start_offset:backtest_end] if ticker != "SPY" else b_hv
        wr_mask = [h >= _WR_HV_THRESHOLD for h in spy_hv_slice]

        # Vol Premium proxy: signal when ticker HV exceeds floor by threshold
        vp_mask = [h >= (_VP_HV_FLOOR * _VP_HV_THRESHOLD) for h in b_hv]

        # Combined: OR-union (either strategy fires)
        comb_mask = [a or b for a, b in zip(wr_mask, vp_mask)]

        all_trades.extend(_evaluate_trades(ticker, "war_room_proxy", b_dates, b_closes, wr_mask, hold_days, win_threshold))
        all_trades.extend(_evaluate_trades(ticker, "vol_premium_proxy", b_dates, b_closes, vp_mask, hold_days, win_threshold))
        all_trades.extend(_evaluate_trades(ticker, "combined_proxy", b_dates, b_closes, comb_mask, hold_days, win_threshold))

    # Aggregate by strategy
    strategy_names = ["war_room_proxy", "vol_premium_proxy", "combined_proxy"]
    results: list[StrategyPerformance] = []

    for sname in strategy_names:
        trades = [t for t in all_trades if t.strategy == sname]
        n_signals = len(trades)
        n_wins = sum(1 for t in trades if t.win)
        win_rate = n_wins / n_signals if n_signals else 0.0
        avg_dd = sum(t.max_drawdown_pct for t in trades) / n_signals if n_signals else 0.0
        results.append(StrategyPerformance(
            strategy=sname,
            n_signals=n_signals,
            n_wins=n_wins,
            win_rate=round(win_rate, 3),
            avg_drawdown_pct=round(avg_dd, 4),
        ))

    return BacktestReport(
        window_days=90,
        start_date=start_date,
        end_date=end_date,
        results=results,
        trades=all_trades,
    )
