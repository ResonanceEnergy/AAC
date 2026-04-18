"""Hedge Auto-Execution — wire HedgeRecommendation → execution engine.

Closes the gap between the Greeks module (which generates hedge
recommendations) and the execution engine (which can place orders).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class HedgeAction(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class HedgeOrder:
    """A hedge translated into an executable order."""

    hedge_id: str
    action: str          # "buy" or "sell"
    instrument: str      # e.g., "SPY", "SPY 520P 2026-05-16"
    quantity: int
    order_type: str      # "market", "limit"
    limit_price: float = 0.0
    priority: str = "end_of_day"
    rationale: str = ""
    estimated_cost: float = 0.0
    greeks_impact: dict[str, float] = field(default_factory=dict)
    status: HedgeAction = HedgeAction.APPROVED
    fill_price: float = 0.0
    executed_at: str = ""
    error: str = ""


@dataclass
class HedgeExecutionResult:
    """Summary of a hedge execution cycle."""

    timestamp: str
    recommendations_received: int
    approved: int
    rejected: int
    deferred: int
    executed: int
    failed: int
    orders: list[HedgeOrder] = field(default_factory=list)
    total_cost: float = 0.0
    net_delta_change: float = 0.0
    net_gamma_change: float = 0.0
    net_vega_change: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "received": self.recommendations_received,
            "approved": self.approved,
            "executed": self.executed,
            "failed": self.failed,
            "total_cost": round(self.total_cost, 2),
            "net_delta_change": round(self.net_delta_change, 2),
        }


# ---------------------------------------------------------------------------
# Hedge Executor
# ---------------------------------------------------------------------------

class HedgeExecutor:
    """Wires HedgeRecommendation from GreeksPortfolioRisk to ExecutionEngine.

    Parameters
    ----------
    max_hedge_cost : float
        Maximum cost per hedge order in USD.
    max_daily_hedge_cost : float
        Maximum total hedge cost per day.
    auto_execute_priorities : list[str]
        Which priority levels to auto-execute ("immediate", "end_of_day").
    dry_run : bool
        If True, log orders but don't submit to exchange.
    """

    def __init__(
        self,
        max_hedge_cost: float = 500.0,
        max_daily_hedge_cost: float = 2000.0,
        auto_execute_priorities: Optional[list[str]] = None,
        dry_run: bool = True,
    ) -> None:
        self.max_hedge_cost = max_hedge_cost
        self.max_daily_hedge_cost = max_daily_hedge_cost
        self.auto_execute_priorities = auto_execute_priorities or ["immediate"]
        self.dry_run = dry_run
        self._daily_cost: float = 0.0
        self._daily_reset_date: str = ""
        self._history: list[HedgeExecutionResult] = []

    # ── Public API ─────────────────────────────────────────────────────────

    async def process_recommendations(
        self,
        recommendations: list[Any],
        execution_engine: Optional[Any] = None,
    ) -> HedgeExecutionResult:
        """Process HedgeRecommendation objects into executable orders.

        Parameters
        ----------
        recommendations : list[HedgeRecommendation]
            From ``HedgingEngine.generate_hedges()``.
        execution_engine : ExecutionEngine, optional
            If provided and not in dry_run, orders will be submitted.
        """
        self._check_daily_reset()

        orders: list[HedgeOrder] = []
        for i, rec in enumerate(recommendations):
            order = self._translate(rec, i)
            order = self._apply_risk_checks(order)
            orders.append(order)

        # Execute approved orders
        if not self.dry_run and execution_engine is not None:
            for order in orders:
                if order.status == HedgeAction.APPROVED:
                    await self._execute_order(order, execution_engine)

        result = self._build_result(orders, len(recommendations))
        self._history.append(result)

        _log.info(
            "hedge_cycle_complete received=%d approved=%d executed=%d failed=%d cost=%.2f",
            result.recommendations_received,
            result.approved,
            result.executed,
            result.failed,
            result.total_cost,
        )
        return result

    def get_history(self) -> list[HedgeExecutionResult]:
        """Return all historical hedge execution results."""
        return self._history

    # ── Translation ───────────────────────────────────────────────────────

    def _translate(self, rec: Any, idx: int) -> HedgeOrder:
        """Convert a HedgeRecommendation to a HedgeOrder."""
        action = getattr(rec, "action", "buy").lower()
        instrument = getattr(rec, "instrument", "SPY")
        quantity = getattr(rec, "quantity", 0)
        priority = getattr(rec, "priority", "end_of_day")
        rationale = getattr(rec, "rationale", "")
        estimated_cost = getattr(rec, "estimated_cost", 0.0)
        greeks_impact = getattr(rec, "greeks_impact", {})

        # Determine order type
        order_type = "market" if priority == "immediate" else "limit"

        return HedgeOrder(
            hedge_id=f"HEDGE-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{idx:03d}",
            action=action,
            instrument=instrument,
            quantity=abs(quantity),
            order_type=order_type,
            priority=priority,
            rationale=rationale,
            estimated_cost=abs(estimated_cost),
            greeks_impact=dict(greeks_impact) if greeks_impact else {},
            status=HedgeAction.APPROVED,
        )

    # ── Risk checks ───────────────────────────────────────────────────────

    def _apply_risk_checks(self, order: HedgeOrder) -> HedgeOrder:
        """Gate hedge orders through risk limits."""
        # Check per-order cost
        if order.estimated_cost > self.max_hedge_cost:
            order.status = HedgeAction.REJECTED
            order.error = f"Cost ${order.estimated_cost:.0f} exceeds per-order limit ${self.max_hedge_cost:.0f}"
            _log.warning("hedge_rejected_cost hedge_id=%s cost=%.2f", order.hedge_id, order.estimated_cost)
            return order

        # Check daily budget
        if self._daily_cost + order.estimated_cost > self.max_daily_hedge_cost:
            order.status = HedgeAction.DEFERRED
            order.error = f"Daily budget exhausted (${self._daily_cost:.0f} used of ${self.max_daily_hedge_cost:.0f})"
            _log.warning("hedge_deferred_budget hedge_id=%s", order.hedge_id)
            return order

        # Check priority
        if order.priority not in self.auto_execute_priorities:
            order.status = HedgeAction.DEFERRED
            order.error = f"Priority '{order.priority}' not in auto-execute list"
            return order

        # Zero quantity
        if order.quantity <= 0:
            order.status = HedgeAction.REJECTED
            order.error = "Zero quantity"
            return order

        return order

    # ── Execution ─────────────────────────────────────────────────────────

    async def _execute_order(self, order: HedgeOrder, engine: Any) -> None:
        """Submit an approved hedge order to the execution engine."""
        try:
            from TradingExecution.execution_engine import OrderSide, OrderType

            side = OrderSide.BUY if order.action == "buy" else OrderSide.SELL
            otype = OrderType.MARKET if order.order_type == "market" else OrderType.LIMIT

            created = await engine.create_order(
                symbol=order.instrument,
                side=side,
                order_type=otype,
                quantity=float(order.quantity),
                price=order.limit_price if otype == OrderType.LIMIT else None,
                metadata={"source": "hedge_executor", "hedge_id": order.hedge_id},
            )
            submitted = await engine.submit_order(created)

            order.status = HedgeAction.EXECUTED
            order.fill_price = submitted.price or 0.0
            order.executed_at = datetime.now(timezone.utc).isoformat()
            self._daily_cost += order.estimated_cost

            _log.info(
                "hedge_executed hedge_id=%s instrument=%s quantity=%d",
                order.hedge_id,
                order.instrument,
                order.quantity,
            )

        except ImportError:
            order.status = HedgeAction.FAILED
            order.error = "ExecutionEngine imports unavailable"
            _log.error("hedge_import_error hedge_id=%s", order.hedge_id)
        except (ConnectionError, OSError, RuntimeError) as exc:
            order.status = HedgeAction.FAILED
            order.error = str(exc)
            _log.error("hedge_execution_failed hedge_id=%s error=%s", order.hedge_id, str(exc))

    # ── Helpers ───────────────────────────────────────────────────────────

    def _check_daily_reset(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._daily_reset_date != today:
            self._daily_cost = 0.0
            self._daily_reset_date = today

    def _build_result(
        self, orders: list[HedgeOrder], n_received: int,
    ) -> HedgeExecutionResult:
        approved = sum(1 for o in orders if o.status in (HedgeAction.APPROVED, HedgeAction.EXECUTED))
        rejected = sum(1 for o in orders if o.status == HedgeAction.REJECTED)
        deferred = sum(1 for o in orders if o.status == HedgeAction.DEFERRED)
        executed = sum(1 for o in orders if o.status == HedgeAction.EXECUTED)
        failed = sum(1 for o in orders if o.status == HedgeAction.FAILED)

        total_cost = sum(o.estimated_cost for o in orders if o.status == HedgeAction.EXECUTED)

        net_delta = sum(o.greeks_impact.get("delta", 0.0) for o in orders if o.status == HedgeAction.EXECUTED)
        net_gamma = sum(o.greeks_impact.get("gamma", 0.0) for o in orders if o.status == HedgeAction.EXECUTED)
        net_vega = sum(o.greeks_impact.get("vega", 0.0) for o in orders if o.status == HedgeAction.EXECUTED)

        return HedgeExecutionResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            recommendations_received=n_received,
            approved=approved,
            rejected=rejected,
            deferred=deferred,
            executed=executed,
            failed=failed,
            orders=orders,
            total_cost=total_cost,
            net_delta_change=net_delta,
            net_gamma_change=net_gamma,
            net_vega_change=net_vega,
        )
