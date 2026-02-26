#!/usr/bin/env python3
"""
Paper Trading Environment
========================
Simulated trading environment for strategy validation without financial risk.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import get_audit_logger
from shared.market_data_integration import market_data_integration


class OrderType(Enum):
    """Paper trading order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"


@dataclass
class PaperOrder:
    """Paper trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperPosition:
    """Paper trading position"""
    symbol: str
    side: PositionSide
    quantity: float
    average_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    market_value: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class PaperAccount:
    """Paper trading account"""
    account_id: str
    balance: float = 1000000.0  # $1M starting balance
    equity: float = 1000000.0
    margin_used: float = 0.0
    margin_available: float = 1000000.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    orders: Dict[str, PaperOrder] = field(default_factory=dict)
    trade_history: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class PaperTrade:
    """Paper trade record"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    strategy_id: Optional[str] = None
    pnl: float = 0.0
    commission: float = 0.0


class PaperTradingEngine:
    """Paper trading execution engine"""

    def __init__(self, account_id: str = "paper_account_1"):
        self.logger = logging.getLogger("PaperTradingEngine")
        self.audit_logger = get_audit_logger()
        self.account_id = account_id
        self.account = PaperAccount(account_id=account_id)
        self.slippage_model = "realistic"  # "none", "fixed", "realistic"
        self.commission_per_trade = 0.01  # $0.01 per trade
        self.min_commission = 1.0  # $1 minimum
        self.max_position_size_pct = 0.1  # Max 10% of account per position

    async def initialize(self):
        """Initialize the paper trading engine"""
        # Load existing account state
        await self._load_account_state()

        # Start price update monitoring
        asyncio.create_task(self._start_price_monitoring())

    async def _load_account_state(self):
        """Load account state from storage"""
        account_file = PROJECT_ROOT / "data" / "paper_trading" / f"{self.account_id}.json"
        if account_file.exists():
            try:
                with open(account_file, 'r') as f:
                    account_data = json.load(f)

                # Convert string dates back to datetime recursively
                def convert_datetimes(obj):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if isinstance(value, str) and 'T' in value:
                                try:
                                    obj[key] = datetime.fromisoformat(value)
                                except:
                                    pass
                            elif isinstance(value, (dict, list)):
                                convert_datetimes(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            convert_datetimes(item)

                convert_datetimes(account_data)

                # Reconstruct account object
                positions_data = account_data.pop('positions', {})
                orders_data = account_data.pop('orders', {})
                trade_history_data = account_data.pop('trade_history', [])

                self.account = PaperAccount(**account_data)

                # Reconstruct positions
                for symbol, pos_data in positions_data.items():
                    if isinstance(pos_data, dict):
                        # Convert string back to enum
                        if 'side' in pos_data:
                            pos_data['side'] = PositionSide(pos_data['side'])
                        self.account.positions[symbol] = PaperPosition(**pos_data)

                # Reconstruct orders
                for order_id, order_data in orders_data.items():
                    if isinstance(order_data, dict):
                        # Convert strings back to enums
                        if 'side' in order_data:
                            order_data['side'] = OrderSide(order_data['side'])
                        if 'order_type' in order_data:
                            order_data['order_type'] = OrderType(order_data['order_type'])
                        if 'status' in order_data:
                            order_data['status'] = OrderStatus(order_data['status'])
                        self.account.orders[order_id] = PaperOrder(**order_data)

                # Restore trade history
                self.account.trade_history = trade_history_data

                self.logger.info(f"Loaded paper account: {self.account_id}")

            except Exception as e:
                self.logger.error(f"Failed to load account state: {e}")

    async def _save_account_state(self):
        """Save account state to storage"""
        try:
            account_dir = PROJECT_ROOT / "data" / "paper_trading"
            account_dir.mkdir(parents=True, exist_ok=True)

            account_file = account_dir / f"{self.account_id}.json"

            # Convert to dict for JSON serialization
            account_dict = self.account.__dict__.copy()

            # Convert positions to dicts
            positions_dict = {}
            for symbol, position in account_dict['positions'].items():
                pos_dict = position.__dict__.copy()
                # Convert enums to strings
                if 'side' in pos_dict and hasattr(pos_dict['side'], 'value'):
                    pos_dict['side'] = pos_dict['side'].value
                # Convert datetime fields
                if 'created_at' in pos_dict:
                    pos_dict['created_at'] = pos_dict['created_at'].isoformat()
                if 'updated_at' in pos_dict:
                    pos_dict['updated_at'] = pos_dict['updated_at'].isoformat()
                positions_dict[symbol] = pos_dict

            # Convert orders to dicts
            orders_dict = {}
            for order_id, order in account_dict['orders'].items():
                order_dict = order.__dict__.copy()
                # Convert enums to strings
                if 'side' in order_dict and hasattr(order_dict['side'], 'value'):
                    order_dict['side'] = order_dict['side'].value
                if 'order_type' in order_dict and hasattr(order_dict['order_type'], 'value'):
                    order_dict['order_type'] = order_dict['order_type'].value
                if 'status' in order_dict and hasattr(order_dict['status'], 'value'):
                    order_dict['status'] = order_dict['status'].value
                # Convert datetime fields
                if 'created_at' in order_dict:
                    order_dict['created_at'] = order_dict['created_at'].isoformat()
                if 'updated_at' in order_dict:
                    order_dict['updated_at'] = order_dict['updated_at'].isoformat()
                orders_dict[order_id] = order_dict

            account_dict['positions'] = positions_dict
            account_dict['orders'] = orders_dict
            account_dict['created_at'] = account_dict['created_at'].isoformat()
            account_dict['updated_at'] = account_dict['updated_at'].isoformat()

            with open(account_file, 'w') as f:
                json.dump(account_dict, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save account state: {e}")

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        limit_price: Optional[float] = None,
        strategy_id: Optional[str] = None
    ) -> str:
        """Submit a paper trading order"""
        order_id = str(uuid.uuid4())

        # Validate order
        if not await self._validate_order(symbol, side, quantity, order_type, price):
            raise ValueError("Order validation failed")

        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            limit_price=limit_price,
            strategy_id=strategy_id
        )

        self.account.orders[order_id] = order

        # Process market orders immediately
        if order_type == OrderType.MARKET:
            await self._execute_market_order(order)
        else:
            # Queue limit/stop orders for monitoring
            asyncio.create_task(self._monitor_pending_order(order))

        await self._save_account_state()

        await self.audit_logger.log_order(
            exchange="paper_trading",
            symbol=symbol,
            side=side.value,
            order_type=order_type.value,
            quantity=quantity,
            price=price,
            order_id=order_id,
            status="submitted"
        )

        self.logger.info(f"Submitted paper order: {order_id} - {side.value} {quantity} {symbol}")
        return order_id

    async def _validate_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        price: Optional[float]
    ) -> bool:
        """Validate order parameters"""
        # Check position size limits
        current_position = self.account.positions.get(symbol, PaperPosition(symbol, PositionSide.LONG, 0, 0))
        new_position_size = abs(current_position.quantity + (quantity if side == OrderSide.BUY else -quantity))

        max_position_value = self.account.equity * self.max_position_size_pct

        # Get current price for position value check
        current_price = await self._get_current_price(symbol)
        if current_price and new_position_size * current_price > max_position_value:
            self.logger.warning(f"Order rejected: Position size limit exceeded for {symbol}")
            return False

        # Check account balance for buy orders
        if side == OrderSide.BUY:
            estimated_cost = quantity * (price or current_price or 0)
            if estimated_cost > self.account.balance:
                self.logger.warning(f"Order rejected: Insufficient balance for {symbol}")
                return False

        return True

    async def _execute_market_order(self, order: PaperOrder):
        """Execute a market order with simulated slippage"""
        current_price = await self._get_current_price(order.symbol)
        if not current_price:
            order.status = OrderStatus.REJECTED
            return

        # Apply slippage
        execution_price = self._apply_slippage(current_price, order.side)

        # Calculate commission
        commission = max(self.commission_per_trade * order.quantity, self.min_commission)

        # Execute the trade
        await self._execute_trade(order, execution_price, commission)

    async def _execute_trade(self, order: PaperOrder, execution_price: float, commission: float):
        """Execute a trade and update positions"""
        trade = PaperTrade(
            trade_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=datetime.now(),
            strategy_id=order.strategy_id,
            commission=commission
        )

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = execution_price
        order.updated_at = datetime.now()

        # Update position
        await self._update_position(trade)

        # Update account balance and P&L
        await self._update_account_pnl(trade)

        # Record trade
        self.account.trade_history.append({
            "trade_id": trade.trade_id,
            "order_id": trade.order_id,
            "symbol": trade.symbol,
            "side": trade.side.value,
            "quantity": trade.quantity,
            "price": trade.price,
            "commission": trade.commission,
            "pnl": trade.pnl,
            "timestamp": trade.timestamp.isoformat(),
            "strategy_id": trade.strategy_id,
        })

        await self._save_account_state()

        self.logger.info(f"Executed paper trade: {trade.symbol} {trade.side.value} {trade.quantity} @ ${trade.price:.2f}")

    def _apply_slippage(self, base_price: float, side: OrderSide) -> float:
        """Apply realistic slippage to execution price"""
        if self.slippage_model == "none":
            return base_price
        elif self.slippage_model == "fixed":
            slippage = 0.0001  # 1 bp
            return base_price * (1 + slippage if side == OrderSide.BUY else 1 - slippage)
        else:  # realistic
            # Random slippage based on typical market conditions
            volatility = random.uniform(0.0001, 0.001)  # 0.01% to 0.1%
            direction = 1 if side == OrderSide.BUY else -1
            slippage = random.gauss(0, volatility) * direction
            return base_price * (1 + slippage)

    async def _update_position(self, trade: PaperTrade):
        """Update position after trade execution"""
        symbol = trade.symbol
        quantity = trade.quantity
        price = trade.price

        if symbol not in self.account.positions:
            # New position
            side = PositionSide.LONG if trade.side == OrderSide.BUY else PositionSide.SHORT
            self.account.positions[symbol] = PaperPosition(
                symbol=symbol,
                side=side,
                quantity=quantity if trade.side == OrderSide.BUY else -quantity,
                average_price=price,
                current_price=price
            )
        else:
            # Update existing position
            position = self.account.positions[symbol]

            if trade.side == OrderSide.BUY:
                # Buying more
                total_quantity = position.quantity + quantity
                total_value = (position.quantity * position.average_price) + (quantity * price)
                position.average_price = total_value / total_quantity if total_quantity != 0 else price
                position.quantity = total_quantity
            else:
                # Selling
                position.quantity -= quantity

                # Close position if quantity goes to zero
                if abs(position.quantity) < 0.001:
                    position.quantity = 0
                    position.average_price = 0

        # Update position market value and P&L
        await self._update_position_values()

    async def _update_position_values(self):
        """Update all position values and P&L"""
        total_unrealized_pnl = 0.0

        for symbol, position in self.account.positions.items():
            if position.quantity == 0:
                continue

            current_price = await self._get_current_price(symbol)
            if current_price:
                position.current_price = current_price
                position.market_value = abs(position.quantity) * current_price

                # Calculate unrealized P&L
                if position.side == PositionSide.LONG:
                    position.unrealized_pnl = (current_price - position.average_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.average_price - current_price) * abs(position.quantity)

                total_unrealized_pnl += position.unrealized_pnl

        # Update account equity
        self.account.equity = self.account.balance + total_unrealized_pnl
        self.account.margin_used = sum(p.market_value for p in self.account.positions.values())
        self.account.margin_available = self.account.equity - self.account.margin_used

    async def _update_account_pnl(self, trade: PaperTrade):
        """Update account P&L after trade"""
        # Commission reduces balance
        self.account.balance -= trade.commission

        # For closing trades, realize P&L
        position = self.account.positions.get(trade.symbol)
        if position and position.quantity == 0:
            # Position closed - realize P&L
            trade.pnl = position.unrealized_pnl
            position.realized_pnl += trade.pnl
            self.account.total_pnl += trade.pnl

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            context = await market_data_integration.get_market_context(symbol)
            if context and context.current_price:
                return context.current_price.price
        except Exception as e:
            self.logger.debug(f"Could not get price for {symbol}: {e}")

        # Fallback to simulated price
        return self._simulate_price(symbol)

    def _simulate_price(self, symbol: str) -> float:
        """Generate simulated price for testing"""
        # Simple price simulation based on symbol
        base_prices = {
            "SPY": 450.0,
            "QQQ": 380.0,
            "AAPL": 180.0,
            "MSFT": 380.0,
            "BTC/USDT": 45000.0,
            "ETH/USDT": 2800.0,
        }

        base_price = base_prices.get(symbol, 100.0)
        # Add some random variation
        variation = random.uniform(-0.02, 0.02)  # ±2%
        return base_price * (1 + variation)

    async def _monitor_pending_order(self, order: PaperOrder):
        """Monitor pending limit/stop orders"""
        while order.status == OrderStatus.PENDING:
            try:
                current_price = await self._get_current_price(order.symbol)
                if not current_price:
                    await asyncio.sleep(1)
                    continue

                should_execute = False

                if order.order_type == OrderType.LIMIT:
                    if order.side == OrderSide.BUY and current_price <= order.limit_price:
                        should_execute = True
                    elif order.side == OrderSide.SELL and current_price >= order.limit_price:
                        should_execute = True

                elif order.order_type == OrderType.STOP:
                    if order.side == OrderSide.BUY and current_price >= order.stop_price:
                        should_execute = True
                    elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                        should_execute = True

                if should_execute:
                    await self._execute_market_order(order)
                    break

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                self.logger.error(f"Error monitoring order {order.order_id}: {e}")
                break

    async def _start_price_monitoring(self):
        """Start monitoring price updates for P&L calculations"""
        while True:
            try:
                await self._update_position_values()
                await asyncio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                self.logger.error(f"Price monitoring error: {e}")
                await asyncio.sleep(5)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id in self.account.orders:
            order = self.account.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                asyncio.create_task(self._save_account_state())
                self.logger.info(f"Cancelled paper order: {order_id}")
                return True
        return False

    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary"""
        return {
            "account_id": self.account.account_id,
            "balance": self.account.balance,
            "equity": self.account.equity,
            "margin_used": self.account.margin_used,
            "margin_available": self.account.margin_available,
            "total_pnl": self.account.total_pnl,
            "daily_pnl": self.account.daily_pnl,
            "positions_count": len([p for p in self.account.positions.values() if p.quantity != 0]),
            "orders_count": len([o for o in self.account.orders.values() if o.status == OrderStatus.PENDING]),
            "total_trades": len(self.account.trade_history),
        }

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        positions = []
        for position in self.account.positions.values():
            if position.quantity != 0:
                positions.append({
                    "symbol": position.symbol,
                    "side": position.side.value,
                    "quantity": position.quantity,
                    "average_price": position.average_price,
                    "current_price": position.current_price,
                    "market_value": position.market_value,
                    "unrealized_pnl": position.unrealized_pnl,
                    "realized_pnl": position.realized_pnl,
                })
        return positions

    def get_orders(self, status: Optional[OrderStatus] = None) -> List[Dict]:
        """Get orders"""
        orders = []
        for order in self.account.orders.values():
            if status is None or order.status == status:
                orders.append({
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "type": order.order_type.value,
                    "quantity": order.quantity,
                    "price": order.price,
                    "status": order.status.value,
                    "filled_quantity": order.filled_quantity,
                    "average_fill_price": order.average_fill_price,
                    "created_at": order.created_at.isoformat(),
                    "strategy_id": order.strategy_id,
                })
        return orders

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get recent trade history"""
        return self.account.trade_history[-limit:]

    async def reset_account(self):
        """Reset account to initial state"""
        self.account = PaperAccount(account_id=self.account_id)
        await self._save_account_state()
        self.logger.info(f"Reset paper account: {self.account_id}")


# Global paper trading engine instance
paper_trading_engine = PaperTradingEngine()


async def initialize_paper_trading():
    """Initialize the paper trading environment"""
    await paper_trading_engine.initialize()

    print("[OK] Paper trading environment initialized")
    summary = paper_trading_engine.get_account_summary()
    print(f"  Account: {summary['account_id']}")
    print(f"  Equity: ${summary['equity']:,.2f}")
    print(f"  Positions: {summary['positions_count']}")
    print(f"  Total Trades: {summary['total_trades']}")


if __name__ == "__main__":
    # Example usage
    asyncio.run(initialize_paper_trading())