"""
TradingExecution Models
=======================
Enums and dataclasses for orders and positions.
Extracted from execution_engine.py for lightweight imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionStatus(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class Order:
    """Trading order."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = field(init=False)
    average_fill_price: float = 0.0
    exchange: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize remaining quantity."""
        self.remaining_quantity = self.quantity - self.filled_quantity

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_fill_price": self.average_fill_price,
            "exchange": self.exchange,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
        }


@dataclass
class Position:
    """Trading position."""

    position_id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    status: PositionStatus = PositionStatus.OPEN
    exchange: str = ""
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: datetime | None = None
    realized_pnl: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def unrealized_pnl(self) -> float:
        if self.side == OrderSide.BUY:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.side == OrderSide.BUY:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100

    def should_stop_loss(self) -> bool:
        if self.stop_loss is None:
            return False
        if self.side == OrderSide.BUY:
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss

    def should_take_profit(self) -> bool:
        if self.take_profit is None:
            return False
        if self.side == OrderSide.BUY:
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "exchange": self.exchange,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "realized_pnl": self.realized_pnl,
        }
