from __future__ import annotations

"""Strategy Optimizer — continuously assess and rank strategy performance.

Tracks per-strategy P&L, win rate, drawdown, and Sharpe-like metrics.
Periodically promotes/demotes strategies based on composite scores.
Allocates more capital to winning strategies and throttles losers.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from strategies.paper_trading.engine import PaperTradingEngine
from strategies.paper_trading.strategies import StrategyBase

_log = structlog.get_logger(__name__)


@dataclass
class StrategyScore:
    """Composite score for a single strategy."""

    strategy_name: str
    total_pnl: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    avg_pnl_per_trade: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    composite: float = 0.0
    rank: int = 0
    allocation_pct: float = 0.0  # recommended capital allocation
    status: str = "active"  # active, throttled, disabled
    scored_at: str = ""


class StrategyOptimizer:
    """Compares multiple strategies running on the same paper engine.

    Usage::

        optimizer = StrategyOptimizer(engine, strategies=[grid, dca, momentum])
        optimizer.run_cycle(market_data)  # evaluate + execute + score all
        report = optimizer.get_rankings()
    """

    def __init__(
        self,
        engine: PaperTradingEngine,
        strategies: list[StrategyBase] | None = None,
        min_trades_to_score: int = 5,
        persist_dir: Path | None = None,
    ) -> None:
        self.engine = engine
        self.strategies: list[StrategyBase] = strategies or []
        self._min_trades = min_trades_to_score
        self._scores: dict[str, StrategyScore] = {}
        self._cycle_count: int = 0
        self._persist_dir = persist_dir
        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)

    def add_strategy(self, strategy: StrategyBase) -> None:
        self.strategies.append(strategy)

    # -- Execution cycle ------------------------------------------------------

    def run_cycle(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Run one full optimization cycle:
        1. Update prices
        2. Evaluate all strategies
        3. Execute signals
        4. Check stop-loss/take-profit
        5. Score and rank
        """
        self._cycle_count += 1
        prices = market_data.get("prices", {})

        # Update positions with current prices
        self.engine.update_prices(prices)
        self.engine.check_limit_orders(prices)

        cycle_stats: dict[str, Any] = {"cycle": self._cycle_count, "strategies": {}}

        for strat in self.strategies:
            if not strat.config.enabled:
                continue

            # Skip throttled strategies (still track but don't trade)
            score = self._scores.get(strat.name)
            if score and score.status == "disabled":
                continue

            # Evaluate
            signals = strat.evaluate(market_data)

            # Throttled — reduce signal volume by 75%
            if score and score.status == "throttled":
                signals = signals[:max(1, len(signals) // 4)]

            # Execute
            fills = strat.execute_signals(signals, prices)

            # Check stops
            exits = strat.check_stops(prices)

            cycle_stats["strategies"][strat.name] = {
                "signals": len(signals),
                "fills": fills,
                "exits": exits,
            }

        # Score and rank all strategies
        self._score_all()
        self._allocate_capital()
        self._persist_scores()

        cycle_stats["rankings"] = self.get_rankings()
        return cycle_stats

    # -- Scoring --------------------------------------------------------------

    def _score_all(self) -> None:
        """Calculate composite scores for all strategies."""
        for strat in self.strategies:
            report = strat.report()
            trades = report["total_trades"]
            pnl = report["total_pnl"]
            win_rate = report["win_rate"]
            avg_pnl = report["avg_pnl"]

            # Calculate strategy-specific drawdown from trade history
            strat_trades = [t for t in self.engine.account.trade_history if t.strategy == strat.name]
            cumulative_pnl = 0.0
            peak_pnl = 0.0
            max_dd = 0.0
            for t in strat_trades:
                cumulative_pnl += t.pnl
                if cumulative_pnl > peak_pnl:
                    peak_pnl = cumulative_pnl
                dd = peak_pnl - cumulative_pnl
                if dd > max_dd:
                    max_dd = dd

            dd_pct = (max_dd / self.engine.account.starting_balance * 100) if self.engine.account.starting_balance else 0

            # Sharpe-like: avg_pnl / std_pnl (simplified)
            sharpe = 0.0
            if strat_trades:
                pnls = [t.pnl for t in strat_trades]
                mean_pnl = sum(pnls) / len(pnls)
                var = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls) if len(pnls) > 1 else 1.0
                std = var ** 0.5 if var > 0 else 1e-9
                sharpe = mean_pnl / std

            # Composite score (weights: PnL 30%, win-rate 25%, Sharpe 25%, drawdown -20%)
            if trades >= self._min_trades:
                composite = (
                    (pnl / max(abs(pnl), 1)) * 0.30 * 100  # normalized PnL direction
                    + win_rate * 0.25
                    + min(sharpe, 3.0) / 3.0 * 100 * 0.25  # cap sharpe contribution
                    - dd_pct * 0.20
                )
            else:
                composite = 0.0

            # Determine status
            status = "active"
            if trades >= self._min_trades:
                if composite < -10:
                    status = "disabled"
                elif composite < 10:
                    status = "throttled"

            self._scores[strat.name] = StrategyScore(
                strategy_name=strat.name,
                total_pnl=round(pnl, 2),
                win_rate=round(win_rate, 1),
                trade_count=trades,
                avg_pnl_per_trade=round(avg_pnl, 2),
                max_drawdown_pct=round(dd_pct, 2),
                sharpe_ratio=round(sharpe, 3),
                composite=round(composite, 2),
                status=status,
                scored_at=datetime.now(timezone.utc).isoformat(),
            )

    def _allocate_capital(self) -> None:
        """Allocate capital proportional to composite scores."""
        active = [s for s in self._scores.values() if s.status == "active" and s.composite > 0]
        if not active:
            # Equal allocation fallback
            n = len(self.strategies)
            for s in self._scores.values():
                s.allocation_pct = round(100.0 / n, 1) if n else 0
            return

        total_score = sum(s.composite for s in active)
        if total_score <= 0:
            for s in active:
                s.allocation_pct = round(100.0 / len(active), 1)
            return

        for s in self._scores.values():
            if s in active:
                s.allocation_pct = round((s.composite / total_score) * 100, 1)
            elif s.status == "throttled":
                s.allocation_pct = 5.0  # minimal allocation
            else:
                s.allocation_pct = 0.0

        # Apply allocation to strategy config
        for strat in self.strategies:
            score = self._scores.get(strat.name)
            if score:
                strat.config.max_position_pct = score.allocation_pct / 100

        # Rank
        ranked = sorted(self._scores.values(), key=lambda s: s.composite, reverse=True)
        for i, s in enumerate(ranked):
            s.rank = i + 1

    # -- Reporting ------------------------------------------------------------

    def get_rankings(self) -> list[dict[str, Any]]:
        """Return sorted strategy rankings."""
        ranked = sorted(self._scores.values(), key=lambda s: s.composite, reverse=True)
        return [asdict(s) for s in ranked]

    def get_best_strategy(self) -> str | None:
        """Return the name of the top-ranked strategy."""
        rankings = self.get_rankings()
        if rankings:
            return rankings[0]["strategy_name"]
        return None

    def get_report(self) -> dict[str, Any]:
        """Full optimizer report."""
        return {
            "cycle_count": self._cycle_count,
            "total_strategies": len(self.strategies),
            "active": sum(1 for s in self._scores.values() if s.status == "active"),
            "throttled": sum(1 for s in self._scores.values() if s.status == "throttled"),
            "disabled": sum(1 for s in self._scores.values() if s.status == "disabled"),
            "best_strategy": self.get_best_strategy(),
            "account_equity": round(self.engine.account.equity, 2),
            "account_pnl": round(self.engine.account.total_pnl, 2),
            "rankings": self.get_rankings(),
        }

    # -- Persistence ----------------------------------------------------------

    def _persist_scores(self) -> None:
        if not self._persist_dir:
            return
        path = self._persist_dir / "optimizer_scores.json"
        path.write_text(
            json.dumps(self.get_report(), indent=2),
            encoding="utf-8",
        )
