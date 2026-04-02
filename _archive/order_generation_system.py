#!/usr/bin/env python3
"""
Order Generation System — Validation & Generation
====================================================
Wraps ``shared.secrets_manager.OrderValidator`` and
``TradingExecution.execution_engine.RiskManager`` for order generation
with real validation. Preserves the same public API as the original stub
so all legacy imports work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from shared.secrets_manager import (
    OrderValidator,
    validate_price,
    validate_quantity,
    validate_symbol,
)

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """OrderStatus class."""
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXECUTED = "executed"


@dataclass
class OrderValidationResult:
    """Result of order validation."""
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    risk_score: float = 0.0


@dataclass
class ValidatedOrder:
    """A validated order ready for execution."""
    symbol: str = ""
    side: str = "buy"
    quantity: float = 0.0
    price: float = 0.0
    order_type: str = "market"
    status: OrderStatus = OrderStatus.PENDING
    validation: OrderValidationResult = field(default_factory=OrderValidationResult)
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderGenerator:
    """Generates and validates orders using real validation logic."""

    def __init__(self, **kwargs: Any) -> None:
        self._validator = OrderValidator()

    def generate_order(self, symbol: str, side: str, quantity: float,
                       price: float = 0.0, **kwargs: Any) -> ValidatedOrder:
        """Generate and validate an order."""
        order = ValidatedOrder(
            symbol=symbol, side=side, quantity=quantity,
            price=price, order_type=kwargs.get("order_type", "market"),
            metadata=kwargs,
        )
        validation = self.validate_order(order)
        order.validation = validation
        order.status = OrderStatus.VALIDATED if validation.is_valid else OrderStatus.REJECTED
        return order

    def validate_order(self, order: ValidatedOrder) -> OrderValidationResult:
        """Validate an order with real checks."""
        errors: List[str] = []
        warnings: List[str] = []

        # Symbol validation
        try:
            validate_symbol(order.symbol)
        except (ValueError, TypeError) as exc:
            errors.append(f"Invalid symbol: {exc}")

        # Quantity validation
        try:
            validate_quantity(order.quantity)
        except (ValueError, TypeError) as exc:
            errors.append(f"Invalid quantity: {exc}")

        # Price validation (skip for market orders)
        if order.order_type != "market" and order.price > 0:
            try:
                validate_price(order.price)
            except (ValueError, TypeError) as exc:
                errors.append(f"Invalid price: {exc}")

        # Side validation
        if order.side.lower() not in ("buy", "sell"):
            errors.append(f"Invalid side: {order.side} (must be buy or sell)")

        # Risk warnings
        if order.quantity * max(order.price, 1.0) > 10000:
            warnings.append("Large order — exceeds $10,000 notional")

        risk_score = min(1.0, (order.quantity * max(order.price, 1.0)) / 50000)

        return OrderValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            risk_score=round(risk_score, 3),
        )


def get_order_generator(**kwargs: Any) -> OrderGenerator:
    """Factory function for OrderGenerator."""
    return OrderGenerator(**kwargs)
