"""strategies/stop_manager.py — Sprint 11.

Scans live positions for stop-loss and take-profit triggers and returns
``StopDecision`` objects that can be converted to executable ``TradeSignal``s
by ``stop_signals.py``.

Design
------
* Delegates the P&L threshold calculation to ``ExposureCalculator`` — the
  thresholds live in ``RiskConfig`` (stop_loss_pct=-0.50, take_profit_pct=1.00)
  and are already battle-tested through Sprint 3.
* This module only decides *what to do* about triggered positions; it does
  **not** talk to any exchange or modify any state.
* Fails-open: any exception during evaluation leaves the position alone rather
  than firing a spurious close order.

Usage::

    from strategies.stop_manager import StopManager

    mgr = StopManager()
    decisions = mgr.scan(positions, account_value_usd=50_000)
    for d in decisions:
        print(d.symbol, d.reason, d.trigger)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import structlog

_log = structlog.get_logger(__name__)

# ── Decision dataclass ────────────────────────────────────────────────────────

@dataclass
class StopDecision:
    """A close directive for one position that hit its stop or take-profit.

    Attributes:
        symbol:         Ticker / instrument symbol.
        trigger:        ``"stop_loss"`` or ``"take_profit"``.
        reason:         Human-readable string explaining the trigger.
        pnl_pct:        Unrealised P&L as a fraction of cost basis (e.g. -0.55).
        market_value:   Current absolute market value of the position (USD).
        quantity:       Current quantity (negative = short position).
        decided_at:     ISO timestamp when the decision was made.
    """
    symbol: str
    trigger: str          # "stop_loss" | "take_profit"
    reason: str
    pnl_pct: float
    market_value: float
    quantity: float
    decided_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "trigger": self.trigger,
            "reason": self.reason,
            "pnl_pct": round(self.pnl_pct * 100, 1),
            "market_value": round(self.market_value, 2),
            "quantity": self.quantity,
            "decided_at": self.decided_at,
        }


# ── Manager ───────────────────────────────────────────────────────────────────

class StopManager:
    """Scan positions for stop-loss and take-profit triggers.

    Args:
        config: Optional ``RiskConfig`` — defaults to module-level defaults
                (stop_loss=-50%, take_profit=+100%).
    """

    def __init__(self, config=None) -> None:
        self._config = config   # None → ExposureCalculator uses its own default

    def scan(
        self,
        positions: list,
        account_value_usd: float = 0.0,
    ) -> List[StopDecision]:
        """Return ``StopDecision`` objects for every position that has triggered
        a stop-loss or take-profit threshold.

        Args:
            positions:          List of ``PositionSnapshot`` objects (from
                                ``PositionTracker.all()``).
            account_value_usd:  Account net liquidation value in USD.  If 0,
                                defaults to $50,000 (same as ExposureCalculator).

        Returns:
            List of ``StopDecision`` objects (empty list if nothing triggered).
            Never raises — errors are logged and an empty list is returned.
        """
        try:
            return self._scan(positions, account_value_usd)
        except Exception as exc:
            _log.error("stop_scan_failed", error=str(exc))
            return []

    def urgent_only(
        self,
        positions: list,
        account_value_usd: float = 0.0,
    ) -> List[StopDecision]:
        """Alias for :meth:`scan` — named for symmetry with ``RollManager.urgent_only``."""
        return self.scan(positions, account_value_usd)

    # ── internals ─────────────────────────────────────────────────────────────

    def _scan(
        self,
        positions: list,
        account_value_usd: float,
    ) -> List[StopDecision]:
        from strategies.risk_engine import ExposureCalculator, RiskConfig  # noqa: PLC0415

        config = self._config or RiskConfig()
        calc = ExposureCalculator(config)

        acct = account_value_usd if account_value_usd > 0 else 50_000.0
        report = calc.calculate(positions, acct)

        decisions: List[StopDecision] = []

        for pr in report.positions:
            if pr.stop_triggered:
                reason = (
                    f"unrealised P&L {pr.pnl_pct_of_cost:.0%} hit stop "
                    f"({config.stop_loss_pct:.0%} threshold)"
                )
                decisions.append(
                    StopDecision(
                        symbol=pr.symbol,
                        trigger="stop_loss",
                        reason=reason,
                        pnl_pct=pr.pnl_pct_of_cost,
                        market_value=abs(pr.market_value),
                        quantity=pr.quantity,
                    )
                )
                _log.warning(
                    "stop_loss_triggered",
                    symbol=pr.symbol,
                    pnl_pct=round(pr.pnl_pct_of_cost * 100, 1),
                )
            elif pr.take_profit_triggered:
                reason = (
                    f"unrealised P&L {pr.pnl_pct_of_cost:.0%} hit take-profit "
                    f"({config.take_profit_pct:.0%} threshold)"
                )
                decisions.append(
                    StopDecision(
                        symbol=pr.symbol,
                        trigger="take_profit",
                        reason=reason,
                        pnl_pct=pr.pnl_pct_of_cost,
                        market_value=abs(pr.market_value),
                        quantity=pr.quantity,
                    )
                )
                _log.info(
                    "take_profit_triggered",
                    symbol=pr.symbol,
                    pnl_pct=round(pr.pnl_pct_of_cost * 100, 1),
                )

        _log.info(
            "stop_scan_complete",
            n_positions=len(positions),
            n_triggered=len(decisions),
        )
        return decisions
