"""
Strategy Integration Module
===========================

Integrates strategy execution engine with trading execution system.
Converts strategy signals into executable orders.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from shared.strategy_framework import TradingSignal, SignalType
from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger
from shared.strategy_execution_engine import StrategyExecutionEngine

logger = logging.getLogger(__name__)


class StrategyIntegrator:
    """
    Integrates strategy execution with trading execution system.

    Responsibilities:
    - Receive strategy signals
    - Convert signals to executable orders
    - Route orders to appropriate execution venues
    - Monitor order execution and feedback
    - Update strategy positions and P&L
    """

    def __init__(self, communication: CommunicationFramework, audit_logger: AuditLogger):
        self.communication = communication
        self.audit_logger = audit_logger

        self.strategy_engine: Optional[StrategyExecutionEngine] = None
        self.active_orders = {}  # order_id -> strategy_signal mapping
        self.strategy_positions = {}  # strategy_id -> positions

    async def initialize(self, strategy_engine: StrategyExecutionEngine) -> bool:
        """Initialize the strategy integrator."""
        try:
            logger.info("Initializing Strategy Integrator")

            self.strategy_engine = strategy_engine

            # Set up communication handlers
            await self._setup_communication_handlers()

            await self.audit_logger.log_event(
                'strategy_integrator_initialization',
                'Strategy Integrator initialized successfully'
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Strategy Integrator: {e}")
            return False

    async def process_market_data(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """Process market data through strategy engine."""
        if not self.strategy_engine:
            return []

        return await self.strategy_engine.process_market_data(data)

    async def execute_signals(self, signals: List[TradingSignal]) -> List[Dict[str, Any]]:
        """Convert strategy signals to executable orders."""
        orders = []

        for signal in signals:
            try:
                # Convert signal to order
                order = await self._signal_to_order(signal)
                if order:
                    orders.append(order)

                    # Track active order
                    self.active_orders[order['order_id']] = signal

                    # Update strategy position tracking
                    await self._update_strategy_position(signal)

            except Exception as e:
                logger.error(f"Error processing signal {signal.strategy_id}: {e}")

        # Send orders to execution
        if orders:
            await self._send_orders_to_execution(orders)

            await self.audit_logger.log_event(
                'strategy_orders_generated',
                f'Generated {len(orders)} orders from strategy signals',
                {'orders_count': len(orders)}
            )

        return orders

    async def handle_execution_feedback(self, feedback: Dict[str, Any]):
        """Handle order execution feedback."""
        try:
            order_id = feedback.get('order_id')
            if not order_id or order_id not in self.active_orders:
                return

            signal = self.active_orders[order_id]
            execution_status = feedback.get('status')

            if execution_status == 'filled':
                # Update strategy position with actual execution
                fill_quantity = feedback.get('fill_quantity', 0)
                fill_price = feedback.get('fill_price', 0)

                await self.strategy_engine.strategies[signal.strategy_id].update_position(
                    signal.symbol, fill_quantity, fill_price
                )

                # Remove from active orders
                del self.active_orders[order_id]

                await self.audit_logger.log_event(
                    'strategy_order_filled',
                    f'Strategy {signal.strategy_id} order filled: {fill_quantity} @ {fill_price}',
                    {
                        'strategy_id': signal.strategy_id,
                        'symbol': signal.symbol,
                        'quantity': fill_quantity,
                        'price': fill_price
                    }
                )

            elif execution_status == 'rejected':
                # Handle rejection
                rejection_reason = feedback.get('reason', 'unknown')
                logger.warning(f"Order {order_id} rejected: {rejection_reason}")

                await self.audit_logger.log_event(
                    'strategy_order_rejected',
                    f'Strategy {signal.strategy_id} order rejected: {rejection_reason}',
                    {'order_id': order_id, 'reason': rejection_reason}
                )

        except Exception as e:
            logger.error(f"Error handling execution feedback: {e}")

    async def get_strategy_status(self) -> Dict[str, Any]:
        """Get comprehensive strategy status."""
        if not self.strategy_engine:
            return {}

        status = await self.strategy_engine.get_strategy_status()

        # Add integration-specific status
        status['integration'] = {
            'active_orders': len(self.active_orders),
            'total_positions': len(self.strategy_positions)
        }

        return status

    async def _signal_to_order(self, signal: TradingSignal) -> Optional[Dict[str, Any]]:
        """Convert a trading signal to an executable order."""
        try:
            # Determine order type based on signal
            if signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
                order_type = 'market'  # Default to market orders for arbitrage
                side = 'buy' if signal.signal_type == SignalType.LONG else 'sell'
            elif signal.signal_type == SignalType.HEDGE:
                order_type = 'market'
                side = 'buy' if signal.quantity > 0 else 'sell'
            elif signal.signal_type == SignalType.CLOSE:
                order_type = 'market'
                side = 'buy' if signal.quantity > 0 else 'sell'  # Close position
            else:
                logger.warning(f"Unsupported signal type: {signal.signal_type}")
                return None

            # Determine execution venue based on symbol type
            venue = self._determine_execution_venue(signal.symbol)

            # Create order
            order = {
                'order_id': f"{signal.strategy_id}_{signal.symbol}_{datetime.now().timestamp()}",
                'strategy_id': signal.strategy_id,
                'symbol': signal.symbol,
                'side': side,
                'quantity': abs(signal.quantity),
                'order_type': order_type,
                'venue': venue,
                'confidence': signal.confidence,
                'metadata': signal.metadata or {},
                'timestamp': signal.timestamp
            }

            return order

        except Exception as e:
            logger.error(f"Error converting signal to order: {e}")
            return None

    def _determine_execution_venue(self, symbol: str) -> str:
        """Determine the appropriate execution venue for a symbol."""
        # ETF symbols
        if symbol in ['SPY', 'QQQ', 'IWM', 'EFA', 'VWO']:
            return 'NYSE'  # Primary listing exchange

        # Futures symbols
        elif symbol in ['ES', 'NQ', 'RTY', 'E7', 'M6C']:
            return 'CME'  # Chicago Mercantile Exchange

        # Crypto symbols
        elif symbol in ['BTC', 'ETH', 'ADA', 'SOL']:
            return 'COINBASE'  # Primary crypto exchange

        else:
            return 'AUTO'  # Let execution engine decide

    async def _send_orders_to_execution(self, orders: List[Dict[str, Any]]):
        """Send orders to the trading execution system."""
        try:
            # Send each order to execution
            for order in orders:
                await self.communication.send_message(
                    sender="strategy_integrator",
                    recipient="trading_execution",
                    message_type="new_order",
                    payload=order
                )

        except Exception as e:
            logger.error(f"Error sending orders to execution: {e}")

    async def _update_strategy_position(self, signal: TradingSignal):
        """Update strategy position tracking."""
        strategy_id = signal.strategy_id

        if strategy_id not in self.strategy_positions:
            self.strategy_positions[strategy_id] = {}

        positions = self.strategy_positions[strategy_id]

        if signal.symbol not in positions:
            positions[signal.symbol] = 0

        # Update position (simplified - doesn't account for fills yet)
        positions[signal.symbol] += signal.quantity

    async def _setup_communication_handlers(self):
        """Set up communication message handlers."""
        # Subscribe to execution feedback
        await self.communication.subscribe_to_messages(
            'strategy_integrator',
            ['trading_execution.order_feedback']
        )

        # Subscribe to market data
        await self.communication.subscribe_to_messages(
            'strategy_integrator',
            ['market_data.etf.*', 'bigbrain.nav.*', 'market_data.futures.*']
        )


async def get_strategy_integrator(communication: CommunicationFramework,
                                audit_logger: AuditLogger) -> StrategyIntegrator:
    """Factory function to create strategy integrator."""
    integrator = StrategyIntegrator(communication, audit_logger)
    return integrator