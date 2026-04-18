from __future__ import annotations

"""Enhanced performance metrics for paper trading.

Goes beyond win-rate and P&L to compute the institutional-grade metrics
that separate lucky runs from genuine edge:

- Sharpe Ratio
- Sortino Ratio
- Profit Factor
- Recovery Factor
- Calmar Ratio
- Max consecutive wins/losses
- Expectancy (avg win × win rate - avg loss × loss rate)
"""

import math
from dataclasses import dataclass
from typing import Any

import structlog

_log = structlog.get_logger(__name__)


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance summary."""

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    recovery_factor: float = 0.0
    calmar_ratio: float = 0.0
    expectancy: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    risk_reward_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "profit_factor": round(self.profit_factor, 3),
            "recovery_factor": round(self.recovery_factor, 3),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "expectancy": round(self.expectancy, 4),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "best_trade": round(self.best_trade, 2),
            "worst_trade": round(self.worst_trade, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 1),
            "risk_reward_ratio": round(self.risk_reward_ratio, 3),
        }

    def passes_minimum_bar(
        self,
        min_sharpe: float = 1.0,
        min_profit_factor: float = 1.2,
        max_drawdown_pct: float = 20.0,
        min_trades: int = 30,
    ) -> tuple[bool, list[str]]:
        """Check if metrics meet minimum viability thresholds.

        Returns (passes, list_of_failures).
        """
        failures: list[str] = []
        if self.total_trades < min_trades:
            failures.append(f"insufficient_trades ({self.total_trades} < {min_trades})")
        if self.sharpe_ratio < min_sharpe:
            failures.append(f"low_sharpe ({self.sharpe_ratio:.2f} < {min_sharpe})")
        if self.profit_factor < min_profit_factor:
            failures.append(f"low_profit_factor ({self.profit_factor:.2f} < {min_profit_factor})")
        return len(failures) == 0, failures


class MetricsTracker:
    """Compute enhanced performance metrics from trade history.

    Accepts trade P&L values and equity snapshots to compute
    institutional-grade metrics.

    Usage::

        tracker = MetricsTracker()
        for trade in completed_trades:
            tracker.record_trade(trade.pnl)
        tracker.update_equity(current_equity)
        snapshot = tracker.compute()
    """

    def __init__(self, risk_free_rate: float = 0.05) -> None:
        self._risk_free_rate = risk_free_rate  # annual
        self._trade_pnls: list[float] = []
        self._equity_curve: list[float] = []
        self._peak_equity: float = 0.0
        self._max_drawdown: float = 0.0
        self._starting_equity: float = 0.0

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade's P&L."""
        self._trade_pnls.append(pnl)

    def record_trades(self, pnls: list[float]) -> None:
        """Bulk-record multiple completed trades."""
        self._trade_pnls.extend(pnls)

    def update_equity(self, equity: float) -> None:
        """Record current equity snapshot."""
        if not self._equity_curve:
            self._starting_equity = equity
        self._equity_curve.append(equity)
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            if dd > self._max_drawdown:
                self._max_drawdown = dd

    def compute(self) -> PerformanceSnapshot:
        """Compute all metrics from recorded data."""
        snap = PerformanceSnapshot()
        pnls = self._trade_pnls

        if not pnls:
            return snap

        snap.total_trades = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        snap.win_rate = (len(wins) / len(pnls) * 100) if pnls else 0
        snap.avg_win = (sum(wins) / len(wins)) if wins else 0
        snap.avg_loss = (sum(losses) / len(losses)) if losses else 0
        snap.best_trade = max(pnls)
        snap.worst_trade = min(pnls)

        # Risk/Reward ratio
        if snap.avg_loss != 0:
            snap.risk_reward_ratio = abs(snap.avg_win / snap.avg_loss)

        # Expectancy
        win_rate_frac = len(wins) / len(pnls) if pnls else 0
        loss_rate_frac = len(losses) / len(pnls) if pnls else 0
        snap.expectancy = (snap.avg_win * win_rate_frac) + (snap.avg_loss * loss_rate_frac)

        # Profit Factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        snap.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        # Max consecutive wins/losses
        snap.max_consecutive_wins = self._max_consecutive(pnls, positive=True)
        snap.max_consecutive_losses = self._max_consecutive(pnls, positive=False)

        # Sharpe Ratio (annualised, assuming ~252 trading days)
        snap.sharpe_ratio = self._sharpe(pnls)

        # Sortino Ratio
        snap.sortino_ratio = self._sortino(pnls)

        # Recovery Factor = net profit / max drawdown
        total_pnl = sum(pnls)
        if self._max_drawdown > 0 and self._starting_equity > 0:
            max_dd_dollar = self._max_drawdown * self._starting_equity
            snap.recovery_factor = total_pnl / max_dd_dollar if max_dd_dollar > 0 else 0.0
        else:
            snap.recovery_factor = 0.0

        # Calmar Ratio = annualised return / max drawdown
        if self._max_drawdown > 0 and self._starting_equity > 0:
            total_return = total_pnl / self._starting_equity
            snap.calmar_ratio = total_return / self._max_drawdown
        else:
            snap.calmar_ratio = 0.0

        return snap

    def get_status(self) -> dict[str, Any]:
        snap = self.compute()
        return snap.to_dict()

    # -- Internal -------------------------------------------------------------

    def _sharpe(self, pnls: list[float]) -> float:
        if len(pnls) < 2:
            return 0.0
        mean = sum(pnls) / len(pnls)
        std = self._std(pnls)
        if std == 0:
            return 0.0
        # Daily risk-free = annual / 252
        rf_daily = self._risk_free_rate / 252
        return ((mean - rf_daily) / std) * math.sqrt(252)

    def _sortino(self, pnls: list[float]) -> float:
        if len(pnls) < 2:
            return 0.0
        mean = sum(pnls) / len(pnls)
        rf_daily = self._risk_free_rate / 252
        downside = [p for p in pnls if p < 0]
        if not downside:
            return float("inf") if mean > 0 else 0.0
        downside_std = self._std(downside)
        if downside_std == 0:
            return 0.0
        return ((mean - rf_daily) / downside_std) * math.sqrt(252)

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)

    @staticmethod
    def _max_consecutive(pnls: list[float], positive: bool = True) -> int:
        max_streak = 0
        current = 0
        for p in pnls:
            if (positive and p > 0) or (not positive and p < 0):
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak
