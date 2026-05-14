"""core/order_monitor.py — Sprint 14: Stale Order Monitor.

Tracks submitted ``OrderConfirmation`` objects produced by ``AutoTrader``
and cancels any that remain unfilled past ``stale_minutes`` (default 30).

Architecture
------------
``PendingOrderRegistry``
    Thread-safe in-memory store.  ``AutoTrader`` registers every SUBMITTED
    confirmation; ``AutoTrader`` (or any caller) marks orders filled so they
    leave the registry before the stale window.

``OrderMonitor``
    Drives the scan loop.  ``scan()`` iterates the registry, identifies stale
    orders, and calls ``_cancel_via_ibkr()`` (module-level, patchable) for
    each one.  Returns a ``OrderMonitorReport``.

``MarketScheduler.run_order_monitor()``
    Calls ``OrderMonitor.scan()`` as a periodic task during market hours.
    Injected ``OrderMonitor`` instance is shared between the scheduler and
    ``AutoTrader``.

Design decisions
----------------
* Fails-open — any per-order exception is logged and counted in ``errors``.
* ``_cancel_via_ibkr()`` is a module-level function so tests can patch it
  without reaching into the class internals.
* No persistence — registry is in-memory; a restart clears pending orders.
  IBKR itself retains the open orders; next ``get_open_orders()`` call would
  still see them.  Out-of-scope for Sprint 14.
"""
from __future__ import annotations

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog

_log = structlog.get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_STALE_MINUTES: int = 30


# ── PendingOrder (internal record) ───────────────────────────────────────────

@dataclass
class PendingOrder:
    """A submitted order being tracked by the registry."""

    order_id: str
    ticker: str
    submitted_at: datetime       # UTC; used to compute age

    def age_minutes(self, now: Optional[datetime] = None) -> float:
        """Minutes elapsed since ``submitted_at``."""
        ref = now or datetime.now(tz=timezone.utc)
        # Ensure both are tz-aware
        sa = self.submitted_at
        if sa.tzinfo is None:
            sa = sa.replace(tzinfo=timezone.utc)
        return (ref - sa).total_seconds() / 60.0


# ── PendingOrderRegistry ──────────────────────────────────────────────────────

class PendingOrderRegistry:
    """Thread-safe in-memory store of pending (SUBMITTED) orders.

    Usage::

        registry = PendingOrderRegistry()
        registry.add(order_id="ibkr-1234", ticker="SPY", submitted_at=datetime.utcnow())
        registry.mark_filled("ibkr-1234")
    """

    def __init__(self) -> None:
        self._orders: dict[str, PendingOrder] = {}
        self._lock = threading.Lock()

    def add(self, order_id: str, ticker: str, submitted_at: Optional[datetime] = None) -> None:
        """Register a new pending order.  Duplicate order_id is silently ignored."""
        if not order_id:
            return
        ts = submitted_at or datetime.now(tz=timezone.utc)
        with self._lock:
            if order_id not in self._orders:
                self._orders[order_id] = PendingOrder(
                    order_id=order_id,
                    ticker=ticker,
                    submitted_at=ts,
                )

    def remove(self, order_id: str) -> None:
        """Remove an order (filled, cancelled, or expired)."""
        with self._lock:
            self._orders.pop(order_id, None)

    # Alias matching the concept in AutoTrader
    mark_filled = remove

    def get_pending(self) -> list[PendingOrder]:
        """Snapshot of current pending orders (copy — safe to iterate)."""
        with self._lock:
            return list(self._orders.values())

    @property
    def size(self) -> int:
        """Number of currently tracked pending orders."""
        with self._lock:
            return len(self._orders)


# ── OrderMonitorReport ───────────────────────────────────────────────────────

@dataclass
class OrderMonitorReport:
    """Result of a single ``OrderMonitor.scan()`` call."""

    checked: int = 0
    cancelled: int = 0
    still_pending: int = 0
    errors: int = 0
    cancelled_ids: list[str] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "cancelled": self.cancelled,
            "still_pending": self.still_pending,
            "errors": self.errors,
            "cancelled_ids": list(self.cancelled_ids),
            "generated_at": self.generated_at,
        }


# ── IBKR cancel helper (patchable at module level) ───────────────────────────

def _cancel_via_ibkr(order_id: str, ticker: str) -> bool:
    """Attempt to cancel ``order_id`` on IBKR.

    Connects, cancels, disconnects.  Returns True on success.
    Raises on connection or cancellation failure.

    This is a module-level function so tests can patch
    ``core.order_monitor._cancel_via_ibkr`` without touching the class.
    """
    import asyncio  # noqa: PLC0415

    from TradingExecution.exchange_connectors.ibkr_connector import (  # noqa: PLC0415
        IBKRConnector,
    )

    async def _go() -> bool:
        conn = IBKRConnector()
        await conn.connect()
        try:
            return await conn.cancel_order(order_id=order_id, symbol=ticker)
        finally:
            await conn.disconnect()

    return asyncio.run(_go())


# ── OrderMonitor ─────────────────────────────────────────────────────────────

class OrderMonitor:
    """Scans the pending order registry and cancels stale orders.

    Args:
        registry:      Shared ``PendingOrderRegistry``.  Required.
        stale_minutes: Orders unfilled longer than this are cancelled.
                       Default 30 minutes.

    Usage::

        registry = PendingOrderRegistry()
        monitor  = OrderMonitor(registry, stale_minutes=30)

        # Called by AutoTrader after each execution cycle:
        monitor.register(order_id=conf.order_id, ticker=conf.signal_ticker,
                         submitted_at=datetime.utcnow())

        # Called by MarketScheduler every scan interval:
        report = monitor.scan()
    """

    def __init__(
        self,
        registry: PendingOrderRegistry,
        stale_minutes: int = _DEFAULT_STALE_MINUTES,
    ) -> None:
        self._registry = registry
        self._stale_minutes = stale_minutes

    def register(
        self,
        order_id: str,
        ticker: str,
        submitted_at: Optional[datetime] = None,
    ) -> None:
        """Register a new pending order.  Delegates to the registry."""
        self._registry.add(order_id=order_id, ticker=ticker, submitted_at=submitted_at)

    def mark_filled(self, order_id: str) -> None:
        """Remove an order that has been confirmed filled."""
        self._registry.mark_filled(order_id)

    def scan(self, now: Optional[datetime] = None) -> OrderMonitorReport:
        """Scan pending orders; cancel those that have exceeded the stale window.

        Args:
            now: Reference time for age computation.  Defaults to UTC now.
                 Inject in tests for deterministic results.

        Returns:
            ``OrderMonitorReport`` with counts and cancelled IDs.
        """
        ref = now or datetime.now(tz=timezone.utc)
        now_str = ref.strftime("%Y-%m-%d %H:%M UTC")

        pending = self._registry.get_pending()
        report = OrderMonitorReport(
            checked=len(pending),
            generated_at=now_str,
        )

        for po in pending:
            try:
                age = po.age_minutes(ref)
                if age < self._stale_minutes:
                    report.still_pending += 1
                    continue

                # Order is stale — attempt cancellation
                _log.warning(
                    "stale_order_detected",
                    order_id=po.order_id,
                    ticker=po.ticker,
                    age_minutes=round(age, 1),
                )
                cancelled = _cancel_via_ibkr(po.order_id, po.ticker)
                if cancelled:
                    self._registry.remove(po.order_id)
                    report.cancelled += 1
                    report.cancelled_ids.append(po.order_id)
                    _log.info(
                        "stale_order_cancelled",
                        order_id=po.order_id,
                        ticker=po.ticker,
                    )
                else:
                    # Cancel returned False — leave in registry, count as error
                    report.errors += 1
                    _log.error(
                        "stale_order_cancel_failed",
                        order_id=po.order_id,
                        ticker=po.ticker,
                    )

            except Exception as exc:
                report.errors += 1
                _log.error(
                    "order_monitor_scan_error",
                    order_id=po.order_id,
                    error=str(exc),
                )

        _log.info(
            "order_monitor_scan_complete",
            checked=report.checked,
            cancelled=report.cancelled,
            still_pending=report.still_pending,
            errors=report.errors,
        )
        return report
