#!/usr/bin/env python3
"""
Paper Trading Order Simulator
============================

Simulates order execution for paper trading validation.
"""

import asyncio
import random
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import time

@dataclass
class SimulatedOrder:
    """Simulated order"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market', 'limit'
    price: Optional[float]
    status: str  # 'pending', 'filled', 'cancelled'
    filled_quantity: float
    avg_fill_price: float
    created_at: datetime
    executed_at: Optional[datetime]

class OrderSimulator:
    """Simulates realistic order execution"""

    def __init__(self):
        self.pending_orders: Dict[str, SimulatedOrder] = {}
        self.completed_orders: List[SimulatedOrder] = []
        self.slippage_model = {
            'market_impact': 0.0001,  # 1 basis point
            'execution_delay': 0.1,   # 100ms average
            'fill_probability': 0.95  # 95% fill rate
        }

    async def submit_order(self, symbol: str, side: str, quantity: float,
                          order_type: str = 'market', price: Optional[float] = None) -> str:
        """Submit an order for simulation"""
        import uuid

        order_id = f"sim_{uuid.uuid4().hex[:8]}"

        order = SimulatedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            status='pending',
            filled_quantity=0,
            avg_fill_price=0,
            created_at=datetime.now(),
            executed_at=None
        )

        self.pending_orders[order_id] = order

        # Start execution simulation
        asyncio.create_task(self._execute_order(order))

        return order_id

    async def _execute_order(self, order: SimulatedOrder):
        """Simulate order execution"""
        # Simulate execution delay
        delay = random.expovariate(1.0 / self.slippage_model['execution_delay'])
        await asyncio.sleep(delay)

        # Simulate fill probability
        if random.random() < self.slippage_model['fill_probability']:
            # Execute the order
            current_price = await self._get_simulated_price(order.symbol)

            # Apply market impact
            if order.side == 'buy':
                execution_price = current_price * (1 + self.slippage_model['market_impact'])
            else:
                execution_price = current_price * (1 - self.slippage_model['market_impact'])

            # For limit orders, check price
            if order.order_type == 'limit' and order.price:
                if (order.side == 'buy' and execution_price > order.price) or                    (order.side == 'sell' and execution_price < order.price):
                    # Price not favorable, cancel order
                    order.status = 'cancelled'
                    return

            # Fill the order
            order.status = 'filled'
            order.filled_quantity = order.quantity
            order.avg_fill_price = execution_price
            order.executed_at = datetime.now()

            # Move to completed
            self.completed_orders.append(order)
            del self.pending_orders[order.order_id]

        else:
            # Order not filled, cancel it
            order.status = 'cancelled'
            del self.pending_orders[order.order_id]

    async def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated current price for symbol"""
        # Simple price simulation - in real implementation, this would
        # come from market data feeds
        base_prices = {
            'SPY': 450.0,
            'QQQ': 380.0,
            'IWM': 180.0,
            'AAPL': 180.0,
            'GOOGL': 140.0,
            'MSFT': 380.0,
            'TSLA': 220.0,
            'NVDA': 450.0
        }

        base_price = base_prices.get(symbol, 100.0)

        # Add some random variation
        variation = random.uniform(-0.02, 0.02)  # ±2%
        return base_price * (1 + variation)

    async def get_order_status(self, order_id: str) -> Optional[SimulatedOrder]:
        """Get order status"""
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]

        for order in self.completed_orders:
            if order.order_id == order_id:
                return order

        return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = 'cancelled'
            del self.pending_orders[order_id]
            return True

        return False
