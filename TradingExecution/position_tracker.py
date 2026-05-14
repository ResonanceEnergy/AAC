"""TradingExecution/position_tracker.py — Sprint 2.4.

Reads live positions from IBKR (via IBKRConnector) and caches them
in-memory.  Falls back to an empty snapshot on connection failures —
never raises, always returns the last known state.

Usage::

    tracker = PositionTracker()
    await tracker.refresh()          # pull from IBKR
    pos = tracker.get("SPY")         # None if not held
    all_pos = tracker.all()          # list[Position snapshot]
    total = tracker.total_exposure() # $ sum of abs(market_value)

The tracker does NOT place orders.  It is read-only.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import structlog

_log = structlog.get_logger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PositionSnapshot:
    """Single position as read from the exchange.

    All monetary fields are in the position's native currency (usually USD).
    """
    symbol: str
    sec_type: str          # STK, OPT, CASH, CRYPTO, …
    quantity: float        # positive = long, negative = short
    avg_cost: float        # per-share / per-contract cost basis
    market_price: float    # current mark price
    market_value: float    # abs(quantity) * market_price (signed)
    unrealized_pnl: float
    realized_pnl: float
    exchange: str = "SMART"
    currency: str = "USD"
    # Options-specific (None for equities/ETFs)
    expiry: Optional[str] = None          # YYYYMMDD or None
    strike: Optional[float] = None
    right: Optional[str] = None           # 'C', 'P', or None
    multiplier: int = 100
    refreshed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def pnl_pct(self) -> float:
        """Unrealized P&L as a percentage of cost basis."""
        cost = abs(self.avg_cost * self.quantity)
        return (self.unrealized_pnl / cost * 100) if cost > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "sec_type": self.sec_type,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "market_price": self.market_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "exchange": self.exchange,
            "currency": self.currency,
            "expiry": self.expiry,
            "strike": self.strike,
            "right": self.right,
            "multiplier": self.multiplier,
            "pnl_pct": round(self.pnl_pct, 2),
            "refreshed_at": self.refreshed_at,
        }


# ── Tracker ───────────────────────────────────────────────────────────────────

class PositionTracker:
    """Read-only view of live positions, with in-memory cache.

    The tracker stays connected to IBKR for the duration of the session.
    Call ``refresh()`` periodically (e.g. every 60s) to update the cache.
    """

    def __init__(self, paper: Optional[bool] = None) -> None:
        import os
        if paper is None:
            paper = os.getenv("PAPER_TRADING", "false").lower() == "true"
        self.paper = paper
        self._ibkr: Optional[object] = None
        self._cache: Dict[str, PositionSnapshot] = {}
        self._last_refresh: Optional[str] = None

    # ── connection ────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        try:
            from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
            self._ibkr = IBKRConnector()
            ok = await self._ibkr.connect()  # type: ignore[union-attr]
            if ok:
                _log.info("PositionTracker connected to IBKR (paper=%s)", self.paper)
            return ok
        except Exception as exc:
            _log.error("PositionTracker connect failed: %s", exc)
            return False

    async def disconnect(self) -> None:
        if self._ibkr:
            try:
                await self._ibkr.disconnect()  # type: ignore[union-attr]
            except Exception as exc:
                _log.warning("PositionTracker disconnect error: %s", exc)
        self._ibkr = None

    # ── refresh ───────────────────────────────────────────────────────────────

    async def refresh(self) -> List[PositionSnapshot]:
        """Pull current positions from IBKR and update the cache.

        Returns the refreshed position list.  On failure, returns the
        previously cached list (empty list on first call failure).
        """
        if not self._ibkr:
            _log.warning("PositionTracker not connected — returning cached snapshot")
            return list(self._cache.values())

        try:
            raw_positions = await self._ibkr.get_positions()  # type: ignore[union-attr]
            new_cache: Dict[str, PositionSnapshot] = {}

            for raw in raw_positions:
                sym = raw.get("symbol", "")
                if not sym:
                    continue
                snap = PositionSnapshot(
                    symbol=sym,
                    sec_type=raw.get("sec_type", "STK"),
                    quantity=float(raw.get("quantity", 0)),
                    avg_cost=float(raw.get("avg_cost", 0)),
                    market_price=float(raw.get("market_price", 0)),
                    market_value=float(raw.get("market_value", 0)),
                    unrealized_pnl=float(raw.get("unrealized_pnl", 0)),
                    realized_pnl=float(raw.get("realized_pnl", 0)),
                    exchange=raw.get("exchange", "SMART"),
                    currency=raw.get("currency", "USD"),
                    expiry=raw.get("expiry"),
                    strike=float(raw["strike"]) if raw.get("strike") is not None else None,
                    right=raw.get("right"),
                    multiplier=int(raw.get("multiplier") or 100),
                )
                new_cache[sym] = snap

            self._cache = new_cache
            self._last_refresh = datetime.utcnow().isoformat()

            _log.info(
                "Positions refreshed: count=%d total_exposure=$%.0f",
                len(self._cache), self.total_exposure(),
            )
            return list(self._cache.values())

        except Exception as exc:
            _log.error("Position refresh failed: %s", exc)
            return list(self._cache.values())

    # ── queries ───────────────────────────────────────────────────────────────

    def get(self, symbol: str) -> Optional[PositionSnapshot]:
        """Return position for ``symbol``, or None if not held."""
        return self._cache.get(symbol.upper())

    def all(self) -> List[PositionSnapshot]:
        """Return all cached positions."""
        return list(self._cache.values())

    def total_exposure(self) -> float:
        """Total absolute market value across all positions (USD)."""
        return sum(abs(p.market_value) for p in self._cache.values())

    def total_unrealized_pnl(self) -> float:
        """Sum of unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self._cache.values())

    def summary(self) -> dict:
        """JSON-safe summary for status endpoints."""
        return {
            "position_count": len(self._cache),
            "total_exposure_usd": round(self.total_exposure(), 2),
            "total_unrealized_pnl": round(self.total_unrealized_pnl(), 2),
            "last_refresh": self._last_refresh,
            "positions": [p.to_dict() for p in self._cache.values()],
        }
