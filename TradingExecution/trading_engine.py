#!/usr/bin/env python3
"""
TradingExecution - Core Trading Engine
======================================
Main orchestrator for trade execution across multiple exchanges.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import sys

# Add shared module to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.secrets_manager import validate_order, OrderValidator

# Import exchange connectors
try:
    from TradingExecution.exchange_connectors.binance_connector import BinanceConnector
    from TradingExecution.exchange_connectors.coinbase_connector import CoinbaseConnector
    from TradingExecution.exchange_connectors.kraken_connector import KrakenConnector
    CONNECTORS_AVAILABLE = True
except ImportError:
    CONNECTORS_AVAILABLE = False


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    exchange: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Represents an open position"""
    position_id: str
    exchange: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradingEngine:
    """
    Core trading engine for executing arbitrage strategies.
    
    Handles:
    - Order creation and execution
    - Position management
    - Exchange connectivity
    - Risk checks before execution
    """

    def __init__(self):
        self.config = get_config()
        self.logger = self._setup_logging()
        
        # State
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.exchange_connections: Dict[str, Any] = {}
        
        # Flags
        self.is_running = False
        self.dry_run = self.config.risk.dry_run
        self.paper_trading = self.config.risk.paper_trading
        
        self.logger.info(f"TradingEngine initialized (dry_run={self.dry_run}, paper_trading={self.paper_trading})")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for trading engine"""
        logger = logging.getLogger('TradingEngine')
        logger.setLevel(logging.INFO)

        # Create logs directory
        log_dir = get_project_path('TradingExecution', 'logs')
        log_dir.mkdir(parents=True, exist_ok=True)

        # File handler
        fh = logging.FileHandler(log_dir / 'trading_engine.log')
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    async def initialize_exchanges(self) -> Dict[str, bool]:
        """Initialize connections to configured exchanges"""
        results = {}
        enabled_exchanges = self.config.get_enabled_exchanges()
        
        if not enabled_exchanges:
            self.logger.warning("No exchanges configured with API keys")
            return results
        
        if not CONNECTORS_AVAILABLE:
            self.logger.error("Exchange connectors not available")
            return results
        
        for exchange_name, exchange_config in enabled_exchanges.items():
            try:
                self.logger.info(f"Connecting to {exchange_name}...")
                
                if self.dry_run:
                    self.logger.info(f"DRY RUN: Simulated connection to {exchange_name}")
                    results[exchange_name] = True
                    continue
                
                # Create actual exchange connector
                connector = None
                if exchange_name == 'binance':
                    connector = BinanceConnector(
                        api_key=exchange_config.api_key,
                        api_secret=exchange_config.api_secret,
                        testnet=exchange_config.testnet,
                    )
                elif exchange_name == 'coinbase':
                    connector = CoinbaseConnector(
                        api_key=exchange_config.api_key,
                        api_secret=exchange_config.api_secret,
                        passphrase=exchange_config.passphrase,
                        testnet=exchange_config.testnet,
                    )
                elif exchange_name == 'kraken':
                    connector = KrakenConnector(
                        api_key=exchange_config.api_key,
                        api_secret=exchange_config.api_secret,
                        testnet=exchange_config.testnet,
                    )
                
                if connector:
                    connected = await connector.connect()
                    if connected:
                        self.exchange_connections[exchange_name] = connector
                        results[exchange_name] = True
                        self.logger.info(f"Connected to {exchange_name}")
                    else:
                        results[exchange_name] = False
                        self.logger.error(f"Failed to connect to {exchange_name}")
                else:
                    results[exchange_name] = False
                    self.logger.warning(f"No connector available for {exchange_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to connect to {exchange_name}: {e}")
                results[exchange_name] = False
        
        return results

    async def create_order(
        self,
        exchange: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
    ) -> Optional[Order]:
        """
        Create a new order.
        
        Args:
            exchange: Target exchange name
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: Buy or sell
            order_type: Market, limit, etc.
            quantity: Amount to trade
            price: Price for limit orders
            
        Returns:
            Order object if successful, None otherwise
        """
        # Validate inputs
        validation = validate_order(
            symbol=symbol,
            side=side.value,
            order_type=order_type.value,
            quantity=quantity,
            price=price,
            exchange=exchange,
        )
        if not validation.valid:
            self.logger.warning(f"Order validation failed: {validation.error}")
            return None
        
        # Generate order ID
        order_id = f"{exchange}_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        order = Order(
            order_id=order_id,
            exchange=exchange,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )
        
        # Pre-execution risk checks
        if not await self._validate_order(order):
            self.logger.warning(f"Order {order_id} failed validation")
            order.status = OrderStatus.REJECTED
            return order
        
        # Execute order
        if self.dry_run or self.paper_trading:
            self.logger.info(f"{'DRY RUN' if self.dry_run else 'PAPER'}: {side.value} {quantity} {symbol} @ {price or 'market'}")
            order.status = OrderStatus.FILLED
            order.filled_quantity = quantity
            order.average_price = price or 0.0
        else:
            # Real exchange execution
            if exchange not in self.exchange_connections:
                self.logger.error(f"No connection to exchange: {exchange}")
                order.status = OrderStatus.REJECTED
                order.metadata['rejection_reason'] = f'Not connected to {exchange}'
            else:
                try:
                    connector = self.exchange_connections[exchange]
                    result = await connector.create_order(
                        symbol=symbol,
                        side=side.value,
                        order_type=order_type.value,
                        quantity=quantity,
                        price=price,
                    )
                    order.status = OrderStatus.FILLED if result.status == 'filled' else OrderStatus.OPEN
                    order.filled_quantity = result.filled_quantity
                    order.average_price = result.average_price
                    order.metadata['exchange_order_id'] = result.order_id
                    self.logger.info(f"Order executed: {order_id} -> {result.order_id}")
                except Exception as e:
                    self.logger.error(f"Order execution failed: {e}")
                    order.status = OrderStatus.REJECTED
                    order.metadata['rejection_reason'] = str(e)
        
        self.orders[order_id] = order
        return order

    async def _validate_order(self, order: Order) -> bool:
        """Validate order against risk parameters"""
        # Check position limits
        if len(self.positions) >= self.config.risk.max_open_positions:
            self.logger.warning("Max open positions limit reached")
            return False
        
        # Check position size (would need price conversion in real impl)
        # For now, assume quantity is in USD terms
        if order.quantity > self.config.risk.max_position_size_usd:
            self.logger.warning(f"Order size {order.quantity} exceeds max {self.config.risk.max_position_size_usd}")
            return False
        
        return True

    async def close_position(self, position_id: str) -> Optional[Order]:
        """Close an existing position"""
        if position_id not in self.positions:
            self.logger.warning(f"Position {position_id} not found")
            return None
        
        position = self.positions[position_id]
        
        # Create opposite order to close
        close_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
        
        order = await self.create_order(
            exchange=position.exchange,
            symbol=position.symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
        )
        
        if order and order.status == OrderStatus.FILLED:
            del self.positions[position_id]
            self.logger.info(f"Closed position {position_id}")
        
        return order

    async def get_account_balance(self, exchange: str) -> Dict[str, float]:
        """Get account balance from exchange"""
        if self.dry_run:
            # Return mock balance for testing
            return {
                'USD': 100000.0,
                'BTC': 1.0,
                'ETH': 10.0,
            }
        
        # Get balance from exchange connector
        if exchange not in self.exchange_connections:
            self.logger.warning(f"No connection to {exchange}")
            return {}
        
        try:
            connector = self.exchange_connections[exchange]
            balances = await connector.get_balances()
            return {b.asset: b.total for b in balances.values()}
        except Exception as e:
            self.logger.error(f"Failed to fetch balance from {exchange}: {e}")
            return {}

    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self.positions.values())

    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders"""
        return [o for o in self.orders.values() if o.status in (OrderStatus.PENDING, OrderStatus.OPEN)]

    async def start(self):
        """Start the trading engine"""
        self.logger.info("Starting TradingEngine...")
        self.is_running = True
        
        # Initialize exchange connections
        await self.initialize_exchanges()
        
        self.logger.info("TradingEngine started")

    async def stop(self):
        """Stop the trading engine gracefully"""
        self.logger.info("Stopping TradingEngine...")
        self.is_running = False
        
        # Close all exchange connections properly
        for exchange_name, connector in list(self.exchange_connections.items()):
            try:
                self.logger.info(f"Disconnecting from {exchange_name}...")
                await connector.disconnect()
                del self.exchange_connections[exchange_name]
                self.logger.info(f"Disconnected from {exchange_name}")
            except Exception as e:
                self.logger.error(f"Error closing {exchange_name} connection: {e}")
        
        self.logger.info("TradingEngine stopped")


# CLI for testing
if __name__ == '__main__':
    async def main():
        engine = TradingEngine()
        await engine.start()
        
        # Test order creation
        order = await engine.create_order(
            exchange='binance',
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,  # $100 worth
        )
        
        if order:
            print(f"Order created: {order.order_id}, Status: {order.status.value}")
        
        balance = await engine.get_account_balance('binance')
        print(f"Account balance: {balance}")
        
        await engine.stop()
    
    asyncio.run(main())
