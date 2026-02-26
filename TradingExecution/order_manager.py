#!/usr/bin/env python3
"""
TradingExecution - Order Manager
================================
Order lifecycle management, tracking, and persistence.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import asdict
import sys

# Add shared module to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_project_path
from .trading_engine import Order, OrderStatus, OrderSide, OrderType


class OrderManager:
    """
    Manages order lifecycle and persistence.
    
    Responsibilities:
    - Track all orders (open, filled, cancelled)
    - Persist order history
    - Provide order querying/filtering
    - Handle order state transitions
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.logger = logging.getLogger('OrderManager')
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
        # Persistence
        if persistence_path is None:
            self.persistence_path = get_project_path('TradingExecution', 'data', 'orders.json')
        else:
            self.persistence_path = persistence_path
        
        # Ensure data directory exists
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing orders
        self._load_orders()

    def _load_orders(self):
        """Load orders from persistence file"""
        if self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                    # Deserialize orders properly
                    for order_data in data.get('orders', []):
                        try:
                            order = Order(
                                order_id=order_data['order_id'],
                                exchange=order_data['exchange'],
                                symbol=order_data['symbol'],
                                side=OrderSide(order_data['side']),
                                order_type=OrderType(order_data['order_type']),
                                quantity=order_data['quantity'],
                                price=order_data.get('price'),
                                status=OrderStatus(order_data['status']),
                                filled_quantity=order_data.get('filled_quantity', 0.0),
                                average_price=order_data.get('average_price', 0.0),
                                created_at=datetime.fromisoformat(order_data['created_at']),
                                updated_at=datetime.fromisoformat(order_data['updated_at']),
                                metadata=order_data.get('metadata', {}),
                            )
                            self.orders[order.order_id] = order
                        except Exception as e:
                            self.logger.warning(f"Failed to deserialize order: {e}")
                    self.logger.info(f"Loaded {len(self.orders)} orders from persistence")
            except Exception as e:
                self.logger.error(f"Failed to load orders: {e}")

    def _save_orders(self):
        """Save orders to persistence file"""
        try:
            data = {
                'orders': [self._order_to_dict(o) for o in self.orders.values()],
                'history': [self._order_to_dict(o) for o in self.order_history[-1000:]],  # Keep last 1000
                'updated_at': datetime.now().isoformat(),
            }
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save orders: {e}")

    def _order_to_dict(self, order: Order) -> Dict:
        """Convert order to dictionary for serialization"""
        return {
            'order_id': order.order_id,
            'exchange': order.exchange,
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': order.quantity,
            'price': order.price,
            'status': order.status.value,
            'filled_quantity': order.filled_quantity,
            'average_price': order.average_price,
            'created_at': order.created_at.isoformat(),
            'updated_at': order.updated_at.isoformat(),
            'metadata': order.metadata,
        }

    def add_order(self, order: Order) -> bool:
        """Add a new order to tracking"""
        if order.order_id in self.orders:
            self.logger.warning(f"Order {order.order_id} already exists")
            return False
        
        self.orders[order.order_id] = order
        self._save_orders()
        self.logger.info(f"Added order {order.order_id}")
        return True

    def update_order(self, order_id: str, **updates) -> Optional[Order]:
        """Update an existing order"""
        if order_id not in self.orders:
            self.logger.warning(f"Order {order_id} not found")
            return None
        
        order = self.orders[order_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(order, key):
                setattr(order, key, value)
        
        order.updated_at = datetime.now()
        self._save_orders()
        
        # Move to history if terminal state
        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
            self.order_history.append(order)
            del self.orders[order_id]
        
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_open_orders(self, exchange: Optional[str] = None, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered"""
        orders = [o for o in self.orders.values() if o.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)]
        
        if exchange:
            orders = [o for o in orders if o.exchange == exchange]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return orders

    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get orders by status"""
        return [o for o in self.orders.values() if o.status == status]

    def cancel_order(self, order_id: str, reason: str = "User requested") -> bool:
        """Cancel an order"""
        order = self.update_order(
            order_id,
            status=OrderStatus.CANCELLED,
            metadata={'cancellation_reason': reason}
        )
        return order is not None

    def get_statistics(self) -> Dict:
        """Get order statistics"""
        total_orders = len(self.orders) + len(self.order_history)
        filled_orders = len([o for o in self.order_history if o.status == OrderStatus.FILLED])
        cancelled_orders = len([o for o in self.order_history if o.status == OrderStatus.CANCELLED])
        
        return {
            'total_orders': total_orders,
            'open_orders': len(self.orders),
            'filled_orders': filled_orders,
            'cancelled_orders': cancelled_orders,
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0.0,
        }
