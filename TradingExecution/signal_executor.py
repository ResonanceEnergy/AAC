"""TradingExecution/signal_executor.py — Sprint 2 execution path.

Clean path:  TradeSignal → risk check → order → OrderConfirmation

This module is the ONLY place that translates a TradeSignal into a real or
paper order.  It handles both modes transparently:

  * Live mode  → IBKRConnector.create_order() → wait for ack → confirm
  * Paper mode → PaperTradingDivision.OrderSimulator.submit_order() → confirm

Usage::

    executor = SignalExecutor(paper=True)
    await executor.connect()
    conf = await executor.execute(signal)
    print(conf.status, conf.order_id)

Dependencies
------------
  - TradingExecution.exchange_connectors.ibkr_connector  (live path)
  - PaperTradingDivision.order_simulator                  (paper path)
  - TradingExecution.risk_manager                         (pre-trade checks)
  - shared.signal.TradeSignal                             (input type)
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

import structlog

_log = structlog.get_logger(__name__)


# ── Status ────────────────────────────────────────────────────────────────────

class ConfirmationStatus(str, Enum):
    SUBMITTED = "submitted"    # sent to exchange / paper engine, awaiting fill
    FILLED = "filled"          # confirmed fill
    REJECTED = "rejected"      # rejected by risk or exchange
    ERROR = "error"            # unexpected failure


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class OrderConfirmation:
    """Result of attempting to execute a TradeSignal.

    Fields that are 0 / "" indicate "not yet known".  They are updated once
    the fill confirmation arrives from the exchange.
    """
    signal_ticker: str
    order_id: str                          # exchange or paper order ID
    status: ConfirmationStatus
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    exchange: str = "ibkr"
    paper: bool = False
    rejection_reason: str = ""             # populated on REJECTED
    submitted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    filled_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "signal_ticker": self.signal_ticker,
            "order_id": self.order_id,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "exchange": self.exchange,
            "paper": self.paper,
            "rejection_reason": self.rejection_reason,
            "submitted_at": self.submitted_at,
            "filled_at": self.filled_at,
        }


# ── Risk gate (lightweight, pre-trade) ────────────────────────────────────────

_MAX_SIZE_FRACTION = 0.10   # no single signal may deploy > 10% of account
_MIN_CONFIDENCE    = 0.50   # ignore signals below this threshold


def _risk_check(signal, account_value_usd: float) -> tuple[bool, str]:
    """Lightweight pre-trade risk gate.

    Returns (approved: bool, reason: str).
    """
    from shared.signal import Direction

    if signal.confidence < _MIN_CONFIDENCE:
        return False, f"confidence {signal.confidence:.2f} < minimum {_MIN_CONFIDENCE}"

    if signal.size > _MAX_SIZE_FRACTION:
        return False, f"size {signal.size:.2%} > max {_MAX_SIZE_FRACTION:.0%}"

    if signal.direction is Direction.FLAT:
        return False, "FLAT signal — no order needed"

    order_value = account_value_usd * signal.size
    if order_value < 5.0:
        return False, f"order value ${order_value:.2f} below $5 minimum"

    if account_value_usd <= 0:
        return False, "account value unknown or zero"

    return True, ""


# ── Main executor ─────────────────────────────────────────────────────────────

class SignalExecutor:
    """Executes TradeSignal objects through IBKR (live) or paper engine.

    Lifecycle::

        executor = SignalExecutor(paper=True)
        await executor.connect()
        confirmation = await executor.execute(signal)
        await executor.disconnect()

    Thread-safety: not thread-safe — call from a single asyncio task.
    """

    def __init__(
        self,
        paper: Optional[bool] = None,
        account_value_usd: float = 0.0,
    ) -> None:
        """
        Args:
            paper: Force paper mode. If None, reads PAPER_TRADING env var.
            account_value_usd: Starting account value for risk sizing.
                               Updated automatically when IBKR is connected.
        """
        if paper is None:
            paper = os.getenv("PAPER_TRADING", "false").lower() == "true"
        self.paper = paper
        self.account_value_usd = account_value_usd

        self._ibkr: Optional[object] = None      # IBKRConnector (live only)
        self._paper_sim: Optional[object] = None  # OrderSimulator (paper only)
        self._connected = False

    # ── connection ────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect to IBKR (live) or initialise paper simulator."""
        if self.paper:
            try:
                from PaperTradingDivision.order_simulator import OrderSimulator
                self._paper_sim = OrderSimulator()
                self._connected = True
                _log.info("SignalExecutor connected in PAPER mode")
                return True
            except ImportError as exc:
                _log.error("Paper simulator unavailable: %s", exc)
                return False
        else:
            try:
                from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
                self._ibkr = IBKRConnector()
                connected = await self._ibkr.connect()  # type: ignore[union-attr]
                self._connected = connected
                if connected:
                    # Pull live account value for risk sizing
                    try:
                        summary = await self._ibkr.get_account_summary()  # type: ignore[union-attr]
                        self.account_value_usd = float(
                            summary.get("NetLiquidation", self.account_value_usd) or self.account_value_usd
                        )
                    except Exception as exc:
                        _log.warning("Could not fetch account summary: %s", exc)
                    _log.info("SignalExecutor connected in LIVE mode, account=$%s", self.account_value_usd)
                return connected
            except Exception as exc:
                _log.error("IBKR connect failed: %s", exc)
                return False

    async def disconnect(self) -> None:
        """Disconnect cleanly."""
        if self._ibkr:
            try:
                await self._ibkr.disconnect()  # type: ignore[union-attr]
            except Exception as exc:
                _log.warning("IBKR disconnect error: %s", exc)
        self._connected = False
        _log.info("SignalExecutor disconnected")

    # ── core execute ──────────────────────────────────────────────────────────

    async def execute(self, signal) -> OrderConfirmation:
        """Execute a single TradeSignal.

        Returns an OrderConfirmation regardless of outcome — never raises.
        Check confirmation.status and rejection_reason for failures.
        """
        if not self._connected:
            return OrderConfirmation(
                signal_ticker=signal.ticker,
                order_id="",
                status=ConfirmationStatus.ERROR,
                paper=self.paper,
                rejection_reason="executor not connected",
            )

        # Pre-trade risk gate
        approved, reason = _risk_check(signal, self.account_value_usd)
        if not approved:
            _log.warning(
                "Signal rejected by risk gate: ticker=%s reason=%s",
                signal.ticker, reason,
            )
            return OrderConfirmation(
                signal_ticker=signal.ticker,
                order_id="",
                status=ConfirmationStatus.REJECTED,
                paper=self.paper,
                rejection_reason=reason,
            )

        if self.paper:
            return await self._execute_paper(signal)
        return await self._execute_live(signal)

    async def execute_batch(self, signals: List) -> List[OrderConfirmation]:
        """Execute a list of signals in sequence.  Returns one confirmation per signal."""
        results = []
        for sig in signals:
            conf = await self.execute(sig)
            results.append(conf)
            if conf.status == ConfirmationStatus.FILLED:
                _log.info(
                    "Fill confirmed: ticker=%s order_id=%s qty=%.4f @ %.2f",
                    sig.ticker, conf.order_id, conf.filled_quantity, conf.avg_fill_price,
                )
        return results

    # ── paper path ────────────────────────────────────────────────────────────

    async def _execute_paper(self, signal) -> OrderConfirmation:
        """Submit to the paper OrderSimulator."""
        from shared.signal import Direction

        try:
            order_value = self.account_value_usd * signal.size
            side = (
                "buy" if signal.direction in (Direction.LONG, Direction.LONG_CALL)
                else "sell"
            )
            # submit_order returns a str order_id (fire-and-forget fill)
            order_id = await self._paper_sim.submit_order(  # type: ignore[union-attr]
                symbol=signal.ticker,
                side=side,
                quantity=order_value / max(signal.entry, 1.0),
                order_type="market",
                price=signal.entry if signal.entry > 0 else None,
            )
            # Wait briefly for the async fill task to complete
            await asyncio.sleep(0.1)
            sim_order = await self._paper_sim.get_order_status(order_id)  # type: ignore[union-attr]

            if sim_order is None:
                return OrderConfirmation(
                    signal_ticker=signal.ticker,
                    order_id=order_id,
                    status=ConfirmationStatus.SUBMITTED,
                    exchange="paper",
                    paper=True,
                )

            status = (
                ConfirmationStatus.FILLED if sim_order.status == "filled"
                else ConfirmationStatus.SUBMITTED
            )
            return OrderConfirmation(
                signal_ticker=signal.ticker,
                order_id=order_id,
                status=status,
                filled_quantity=sim_order.filled_quantity,
                avg_fill_price=sim_order.avg_fill_price,
                exchange="paper",
                paper=True,
                filled_at=datetime.utcnow().isoformat() if status == ConfirmationStatus.FILLED else None,
            )
        except Exception as exc:
            _log.error("Paper execution error: ticker=%s exc=%s", signal.ticker, exc)
            return OrderConfirmation(
                signal_ticker=signal.ticker,
                order_id="",
                status=ConfirmationStatus.ERROR,
                paper=True,
                rejection_reason=str(exc),
            )

    # ── live (IBKR) path ──────────────────────────────────────────────────────

    async def _execute_live(self, signal) -> OrderConfirmation:
        """Submit to IBKR via IBKRConnector.create_order()."""
        from shared.signal import Direction

        try:
            side = (
                "buy" if signal.direction in (Direction.LONG, Direction.LONG_CALL)
                else "sell"
            )
            order_type = "market" if signal.entry == 0.0 else "limit"
            order_value = self.account_value_usd * signal.size
            quantity = order_value / max(signal.entry, 1.0)

            exchange_order = await self._ibkr.create_order(  # type: ignore[union-attr]
                symbol=signal.ticker,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=signal.entry if order_type == "limit" else None,
            )

            status = ConfirmationStatus.SUBMITTED
            filled_qty = 0.0
            fill_price = 0.0

            # If the connector returned a fill already (market orders ack fast)
            if hasattr(exchange_order, "filled_quantity") and exchange_order.filled_quantity > 0:
                status = ConfirmationStatus.FILLED
                filled_qty = exchange_order.filled_quantity
                fill_price = exchange_order.average_fill_price

            return OrderConfirmation(
                signal_ticker=signal.ticker,
                order_id=str(exchange_order.order_id),
                status=status,
                filled_quantity=filled_qty,
                avg_fill_price=fill_price,
                exchange="ibkr",
                paper=False,
                filled_at=datetime.utcnow().isoformat() if status == ConfirmationStatus.FILLED else None,
            )

        except Exception as exc:
            _log.error("IBKR execution error: ticker=%s exc=%s", signal.ticker, exc)
            return OrderConfirmation(
                signal_ticker=signal.ticker,
                order_id="",
                status=ConfirmationStatus.ERROR,
                paper=False,
                rejection_reason=str(exc),
            )

    # ── status query ──────────────────────────────────────────────────────────

    async def get_order_status(self, order_id: str, ticker: str) -> Optional[OrderConfirmation]:
        """Query the exchange for current status of a previously submitted order.

        Returns None if order not found or executor not connected.
        """
        if not self._connected or self.paper:
            return None

        try:
            order = await self._ibkr.get_order(order_id, ticker)  # type: ignore[union-attr]
            filled = order.filled_quantity if hasattr(order, "filled_quantity") else 0.0
            price = order.average_fill_price if hasattr(order, "average_fill_price") else 0.0
            status_str = str(getattr(order, "status", "")).lower()
            if "fill" in status_str:
                status = ConfirmationStatus.FILLED
            elif "cancel" in status_str or "reject" in status_str:
                status = ConfirmationStatus.REJECTED
            else:
                status = ConfirmationStatus.SUBMITTED

            return OrderConfirmation(
                signal_ticker=ticker,
                order_id=order_id,
                status=status,
                filled_quantity=filled,
                avg_fill_price=price,
                exchange="ibkr",
                paper=False,
            )
        except Exception as exc:
            _log.warning("get_order_status error: order_id=%s exc=%s", order_id, exc)
            return None
