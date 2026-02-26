#!/usr/bin/env python3
"""
AAC Order Generation System
============================
Converts strategy signals into executable trading orders.
Provides order validation, risk management, and execution routing.

CRITICAL GAP RESOLUTION: Order Generation
- Converts strategy signals into trading orders
- Provides order validation and risk checks
- Routes orders to appropriate execution engines
- Enables strategies to place actual trades
"""

import asyncio
import logging
import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config
from shared.audit_logger import get_audit_logger
# Removed circular import: from strategy_execution_engine import StrategySignal, StrategyExecutionMode

# Define enums locally to avoid circular imports
class StrategyExecutionMode(Enum):
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    SIMULATION = "simulation"

class StrategySignal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

from TradingExecution.execution_engine import ExecutionEngine, Order, OrderSide, OrderType, OrderStatus


class OrderValidationResult(Enum):
    """Order validation results"""
    VALID = "valid"
    INVALID = "invalid"
    REQUIRES_APPROVAL = "requires_approval"
    REJECTED = "rejected"


@dataclass
class ValidatedOrder:
    """Validated trading order ready for execution"""
    order: Order
    strategy_signal: StrategySignal
    validation_result: OrderValidationResult
    validation_errors: List[str] = field(default_factory=list)
    risk_checks: Dict[str, bool] = field(default_factory=dict)
    approval_required: bool = False
    approved_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class OrderGenerationConfig:
    """Configuration for order generation"""
    max_order_size: float = 10000.0  # Maximum order value in USD
    max_position_size: float = 50000.0  # Maximum position value in USD
    max_daily_trades: int = 100  # Maximum trades per day
    max_daily_volume: float = 100000.0  # Maximum daily volume in USD
    require_approval_above: float = 5000.0  # Require approval for orders above this value
    allowed_exchanges: List[str] = field(default_factory=lambda: ["NYSE", "NASDAQ", "AMEX"])
    risk_limits_enabled: bool = True
    market_hours_only: bool = True


class OrderGenerator:
    """
    Converts strategy signals into validated trading orders.
    Provides risk management and execution routing.
    """

    def __init__(self, execution_mode: StrategyExecutionMode = StrategyExecutionMode.PAPER_TRADING):
        self.execution_mode = execution_mode
        self.audit_logger = get_audit_logger()
        self.execution_engine = ExecutionEngine()

        # Configuration
        self.config = OrderGenerationConfig()

        # Tracking
        self.daily_trade_count = 0
        self.daily_volume = 0.0
        self.open_positions: Dict[str, float] = {}  # symbol -> quantity
        self.pending_orders: Dict[str, ValidatedOrder] = {}

        # Callbacks
        self.order_callbacks: List[Callable] = []

        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize order generation system"""
        try:
            self.logger.info("Initializing Order Generation System...")

            # Reset daily counters
            await self._reset_daily_counters()

            # Load existing positions
            await self._load_positions()

            self.logger.info("Order Generation System initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize order generation: {e}")
            raise

    async def generate_order_from_signal(self, signal: StrategySignal) -> Optional[ValidatedOrder]:
        """Generate and validate order from strategy signal"""
        try:
            # Create base order
            order = await self._create_order_from_signal(signal)

            if not order:
                return None

            # Validate order
            validated_order = await self._validate_order(order, signal)

            # Log validation result
            await self.audit_logger.log_system_event(
                action="order_validation",
                resource=f"order_{order.order_id}",
                status="success" if validated_order.validation_result == OrderValidationResult.VALID else "failure",
                details={
                    "strategy_id": signal.strategy_id,
                    "symbol": signal.symbol,
                    "order_value": order.quantity * (order.price or 0),
                    "validation_result": validated_order.validation_result.value,
                    "errors": validated_order.validation_errors
                }
            )

            return validated_order

        except Exception as e:
            self.logger.error(f"Error generating order from signal: {e}")
            await self.audit_logger.log_system_event(
                action="order_generation_error",
                resource=f"strategy_{signal.strategy_id}",
                status="failure",
                details={"error": str(e), "signal": signal.__dict__}
            )
            return None

    async def _create_order_from_signal(self, signal: StrategySignal) -> Optional[Order]:
        """Create trading order from strategy signal"""
        try:
            # Determine order side
            if signal.signal == "buy":
                side = OrderSide.BUY
            elif signal.signal == "sell":
                side = OrderSide.SELL
            else:
                self.logger.warning(f"Unsupported signal type: {signal.signal}")
                return None

            # Determine order type
            if signal.price:
                order_type = OrderType.LIMIT
                price = signal.price
            else:
                order_type = OrderType.MARKET
                price = None

            # Generate unique order ID
            order_id = f"AAC_STRATEGY_{signal.strategy_id}_{datetime.now().strftime('%H%M%S%f')}"

            # Create order
            order = Order(
                order_id=order_id,
                symbol=signal.symbol,
                side=side,
                order_type=order_type,
                quantity=signal.quantity,
                price=price,
                metadata={
                    "strategy_id": signal.strategy_id,
                    "strategy_name": signal.strategy_name,
                    "confidence": signal.confidence,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "signal_timestamp": signal.timestamp.isoformat()
                }
            )

            return order

        except Exception as e:
            self.logger.error(f"Error creating order from signal: {e}")
            return None

    async def _validate_order(self, order: Order, signal: StrategySignal) -> ValidatedOrder:
        """Validate order against risk limits and constraints"""
        errors = []
        risk_checks = {}

        # Basic validation
        if not order.symbol:
            errors.append("Symbol is required")

        if order.quantity <= 0:
            errors.append("Quantity must be positive")

        if order.order_type == OrderType.LIMIT and not order.price:
            errors.append("Limit orders require price")

        # Size limits
        order_value = order.quantity * (order.price or 100)  # Estimate if no price

        if order_value > self.config.max_order_size:
            errors.append(f"Order value ${order_value:.2f} exceeds maximum ${self.config.max_order_size:.2f}")
            risk_checks["size_limit"] = False
        else:
            risk_checks["size_limit"] = True

        # Daily volume limits
        if self.daily_volume + order_value > self.config.max_daily_volume:
            errors.append(f"Daily volume limit would be exceeded")
            risk_checks["daily_volume_limit"] = False
        else:
            risk_checks["daily_volume_limit"] = True

        # Daily trade count limits
        if self.daily_trade_count >= self.config.max_daily_trades:
            errors.append(f"Daily trade count limit ({self.config.max_daily_trades}) reached")
            risk_checks["daily_trade_limit"] = False
        else:
            risk_checks["daily_trade_limit"] = True

        # Position size limits
        current_position = self.open_positions.get(order.symbol, 0)
        new_position = current_position + (order.quantity if order.side == OrderSide.BUY else -order.quantity)
        position_value = abs(new_position) * (order.price or 100)

        if position_value > self.config.max_position_size:
            errors.append(f"Position value ${position_value:.2f} would exceed maximum ${self.config.max_position_size:.2f}")
            risk_checks["position_limit"] = False
        else:
            risk_checks["position_limit"] = True

        # Market hours check
        if self.config.market_hours_only:
            current_time = datetime.now().time()
            market_open = datetime.strptime("09:30", "%H:%M").time()
            market_close = datetime.strptime("16:00", "%H:%M").time()

            if not (market_open <= current_time <= market_close):
                errors.append("Order outside market hours")
                risk_checks["market_hours"] = False
            else:
                risk_checks["market_hours"] = True

        # Approval requirements
        approval_required = order_value > self.config.require_approval_above

        # Determine validation result
        if errors:
            if approval_required and len(errors) == 1 and "approval" in errors[0].lower():
                result = OrderValidationResult.REQUIRES_APPROVAL
            else:
                result = OrderValidationResult.INVALID
        else:
            result = OrderValidationResult.VALID

        return ValidatedOrder(
            order=order,
            strategy_signal=signal,
            validation_result=result,
            validation_errors=errors,
            risk_checks=risk_checks,
            approval_required=approval_required
        )

    async def submit_validated_order(self, validated_order: ValidatedOrder) -> bool:
        """Submit validated order for execution"""
        try:
            if validated_order.validation_result != OrderValidationResult.VALID:
                self.logger.warning(f"Attempted to submit invalid order: {validated_order.validation_result}")
                return False

            # Check if approval is required
            if validated_order.approval_required and not validated_order.approved_by:
                self.logger.warning(f"Order requires approval but not approved: {validated_order.order.order_id}")
                return False

            # Submit to execution engine
            if self.execution_mode == StrategyExecutionMode.LIVE_TRADING:
                success = await self.execution_engine.submit_order(validated_order.order)
            elif self.execution_mode == StrategyExecutionMode.PAPER_TRADING:
                success = await self.execution_engine.submit_paper_order(validated_order.order)
            else:  # SIMULATION
                success = await self._simulate_order_submission(validated_order.order)

            if success:
                # Update tracking
                self.daily_trade_count += 1
                order_value = validated_order.order.quantity * (validated_order.order.price or 0)
                self.daily_volume += order_value

                # Update positions
                symbol = validated_order.order.symbol
                quantity = validated_order.order.quantity
                if validated_order.order.side == OrderSide.BUY:
                    self.open_positions[symbol] = self.open_positions.get(symbol, 0) + quantity
                else:
                    self.open_positions[symbol] = self.open_positions.get(symbol, 0) - quantity

                # Store pending order
                self.pending_orders[validated_order.order.order_id] = validated_order

                # Notify callbacks
                for callback in self.order_callbacks:
                    try:
                        await callback(validated_order)
                    except Exception as e:
                        self.logger.error(f"Error in order callback: {e}")

                # Audit log
                await self.audit_logger.log_order(
                    exchange="MULTI",  # Multi-exchange
                    symbol=validated_order.order.symbol,
                    side=validated_order.order.side.value,
                    order_type=validated_order.order.order_type.value,
                    quantity=validated_order.order.quantity,
                    price=validated_order.order.price,
                    order_id=validated_order.order.order_id,
                    status="submitted"
                )

                self.logger.info(f"Order submitted: {validated_order.order.order_id} ({validated_order.order.symbol})")
                return True
            else:
                self.logger.error(f"Failed to submit order: {validated_order.order.order_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error submitting validated order: {e}")
            return False

    async def approve_order(self, order_id: str, approver: str) -> bool:
        """Approve order requiring approval"""
        try:
            if order_id not in self.pending_orders:
                self.logger.warning(f"Order not found for approval: {order_id}")
                return False

            validated_order = self.pending_orders[order_id]
            validated_order.approved_by = approver
            validated_order.validation_result = OrderValidationResult.VALID

            self.logger.info(f"Order approved: {order_id} by {approver}")
            return True

        except Exception as e:
            self.logger.error(f"Error approving order: {e}")
            return False

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        try:
            if order_id not in self.pending_orders:
                self.logger.warning(f"Order not found for cancellation: {order_id}")
                return False

            # Cancel in execution engine
            success = await self.execution_engine.cancel_order(order_id)

            if success:
                # Remove from pending
                del self.pending_orders[order_id]

                # Audit log
                await self.audit_logger.log_order(
                    exchange="MULTI",
                    symbol="",  # Will be filled from order data
                    side="cancel",
                    order_type="market",
                    quantity=0,
                    order_id=order_id,
                    status="cancelled"
                )

                self.logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel order: {order_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an order"""
        try:
            status = await self.execution_engine.get_order_status(order_id)
            if status:
                return {
                    "order_id": order_id,
                    "status": status.status.value if hasattr(status, 'status') else str(status),
                    "filled_quantity": getattr(status, 'filled_quantity', 0),
                    "remaining_quantity": getattr(status, 'remaining_quantity', 0),
                    "average_price": getattr(status, 'average_fill_price', 0),
                    "updated_at": getattr(status, 'updated_at', datetime.now())
                }
            return None

        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return None

    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        try:
            portfolio = {
                "open_positions": self.open_positions.copy(),
                "pending_orders": len(self.pending_orders),
                "daily_trade_count": self.daily_trade_count,
                "daily_volume": self.daily_volume,
                "total_positions": len([p for p in self.open_positions.values() if p != 0])
            }

            # Calculate total portfolio value (simplified)
            total_value = 0.0
            for symbol, quantity in self.open_positions.items():
                if quantity != 0:
                    # In real implementation, get current price
                    total_value += abs(quantity) * 100  # Placeholder

            portfolio["total_portfolio_value"] = total_value

            return portfolio

        except Exception as e:
            self.logger.error(f"Error getting portfolio status: {e}")
            return {}

    def add_order_callback(self, callback: Callable):
        """Add callback for order events"""
        self.order_callbacks.append(callback)

    async def _reset_daily_counters(self):
        """Reset daily trading counters"""
        self.daily_trade_count = 0
        self.daily_volume = 0.0

    async def _load_positions(self):
        """Load existing positions from execution engine"""
        try:
            # In real implementation, load from database
            self.open_positions = {}
        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")

    async def _simulate_order_submission(self, order: Order) -> bool:
        """Simulate order submission for testing"""
        # Simulate some processing time
        await asyncio.sleep(0.1)

        # Simulate success/failure (90% success rate)
        success = random.random() < 0.9

        if success:
            # Simulate fill
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = order.price or 100.0

        return success


# Global instance
_order_generator = None

async def get_order_generator(mode: StrategyExecutionMode = StrategyExecutionMode.PAPER_TRADING) -> OrderGenerator:
    """Get or create order generator instance"""
    global _order_generator
    if _order_generator is None:
        _order_generator = OrderGenerator(mode)
        await _order_generator.initialize()
    return _order_generator


async def main():
    """Main entry point for order generation testing"""
    import argparse

    parser = argparse.ArgumentParser(description="AAC Order Generation System")
    parser.add_argument("--mode", choices=["paper", "live", "simulation"],
                       default="paper", help="Execution mode")
    parser.add_argument("--test-signal", action="store_true",
                       help="Generate and test a sample signal")

    args = parser.parse_args()

    # Map mode
    mode_map = {
        "paper": StrategyExecutionMode.PAPER_TRADING,
        "live": StrategyExecutionMode.LIVE_TRADING,
        "simulation": StrategyExecutionMode.SIMULATION
    }

    mode = mode_map[args.mode]

    try:
        generator = await get_order_generator(mode)

        if args.test_signal:
            # Create test signal
            from strategy_execution_engine import StrategySignal as StratSignal

            test_signal = StratSignal(
                strategy_id=1,
                strategy_name="Test ETF Strategy",
                symbol="SPY",
                signal="buy",
                confidence=0.8,
                quantity=100,
                price=450.0
            )

            print("🧪 Testing Order Generation...")

            # Generate order
            validated_order = await generator.generate_order_from_signal(test_signal)

            if validated_order:
                print(f"📋 Order Generated: {validated_order.order.order_id}")
                print(f"   Symbol: {validated_order.order.symbol}")
                print(f"   Side: {validated_order.order.side.value}")
                print(f"   Quantity: {validated_order.order.quantity}")
                print(f"   Price: ${validated_order.order.price}")
                print(f"   Validation: {validated_order.validation_result.value}")

                if validated_order.validation_errors:
                    print(f"   Errors: {validated_order.validation_errors}")

                # Submit if valid
                if validated_order.validation_result == OrderValidationResult.VALID:
                    success = await generator.submit_validated_order(validated_order)
                    print(f"   Submission: {'✅ Success' if success else '❌ Failed'}")

                    # Get status
                    status = await generator.get_order_status(validated_order.order.order_id)
                    if status:
                        print(f"   Status: {status['status']}")

            # Get portfolio status
            portfolio = await generator.get_portfolio_status()
            print(f"\n💼 Portfolio Status:")
            print(f"   Open Positions: {portfolio.get('total_positions', 0)}")
            print(f"   Daily Trades: {portfolio.get('daily_trade_count', 0)}")
            print(f"   Daily Volume: ${portfolio.get('daily_volume', 0):.2f}")

        else:
            print("Order Generation System initialized. Use --test-signal to test order generation.")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())