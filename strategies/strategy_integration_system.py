"""
AAC Strategy Integration System
================================

Connects all 50 arbitrage strategies to live market data and enables
paper trading environment for safe testing and deployment.

This system bridges the gap between strategy definitions and executable trading.
"""

import asyncio
import logging
import importlib
import inspect
from typing import Dict, List, Any, Optional, Type
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from shared.strategy_framework import BaseArbitrageStrategy, TradingSignal, SignalType, StrategyConfig
from shared.strategy_loader import StrategyLoader, StrategyCategory, StrategyStatus
from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger
from market_data_aggregator import MarketDataAggregator
from strategy_implementation_factory import StrategyImplementationFactory

logger = logging.getLogger(__name__)


class StrategyIntegrationSystem:
    """
    Integrates all arbitrage strategies with market data and trading execution.

    This system:
    1. Loads all 50 strategy definitions from CSV
    2. Generates executable implementations using templates
    3. Connects strategies to live market data feeds
    4. Enables paper trading environment
    5. Manages strategy lifecycle and risk
    """

    def __init__(self, data_aggregator: MarketDataAggregator,
                 communication: CommunicationFramework,
                 audit_logger: AuditLogger):
        self.data_aggregator = data_aggregator
        self.communication = communication
        self.audit_logger = audit_logger

        # Core components
        self.strategy_loader = StrategyLoader()
        self.implementation_factory = StrategyImplementationFactory(
            data_aggregator, communication, audit_logger
        )

        # Strategy management
        self.active_strategies = {}
        self.strategy_signals = {}
        self.paper_trading_enabled = True

        # Performance tracking
        self.strategy_performance = {}
        self.system_metrics = {
            'total_strategies': 0,
            'active_strategies': 0,
            'total_signals': 0,
            'total_trades': 0,
            'system_pnl': 0.0,
            'risk_score': 0.0
        }

    async def initialize_system(self):
        """Initialize the complete strategy integration system"""
        logger.info("Initializing AAC Strategy Integration System...")

        # Generate all strategy implementations
        await self._generate_all_strategies()

        # Connect strategies to market data
        await self._connect_market_data()

        # Initialize paper trading environment
        await self._initialize_paper_trading()

        # Start strategy monitoring
        asyncio.create_task(self._monitor_strategies())

        logger.info(f"Strategy Integration System initialized with {len(self.active_strategies)} strategies")

    async def _generate_all_strategies(self):
        """Generate executable implementations for all strategies"""
        logger.info("Generating strategy implementations...")

        strategies = await self.implementation_factory.generate_all_strategies()

        for strategy_name, strategy_instance in strategies.items():
            # Initialize the strategy
            success = await strategy_instance.initialize()
            if success:
                self.active_strategies[strategy_name] = strategy_instance
                self.strategy_signals[strategy_name] = []
                self.strategy_performance[strategy_name] = {
                    'signals_generated': 0,
                    'trades_executed': 0,
                    'pnl_realized': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }
                logger.info(f"Successfully initialized strategy: {strategy_name}")
            else:
                logger.warning(f"Failed to initialize strategy: {strategy_name}")

        logger.info(f"Generated and initialized {len(self.active_strategies)}/{len(strategies)} strategy implementations")

    async def _connect_market_data(self):
        """Connect all strategies to market data feeds"""
        logger.info("Connecting strategies to market data...")

        # Collect all required symbols from strategies
        all_symbols = set()
        for strategy in self.active_strategies.values():
            if hasattr(strategy, 'market_data_subscriptions'):
                all_symbols.update(strategy.market_data_subscriptions)

        # Subscribe to market data feeds
        if all_symbols:
            await self.data_aggregator.subscribe_symbols(list(all_symbols))

        # Register strategies as data callbacks
        for strategy in self.active_strategies.values():
            # Create a wrapper callback that formats data correctly for the strategy
            async def strategy_callback(symbol, data_type, data):
                # Format data as expected by strategy.process_market_data
                formatted_data = {
                    'symbol': symbol,
                    'type': data_type,
                    'data': data,
                    'timestamp': datetime.now()
                }
                await strategy.process_market_data(formatted_data)

            self.data_aggregator.data_callbacks.append(strategy_callback)

        logger.info(f"Connected to market data for {len(all_symbols)} symbols and registered {len(self.active_strategies)} strategy callbacks")

    async def _initialize_paper_trading(self):
        """Initialize paper trading environment"""
        logger.info("Initializing paper trading environment...")

        # Create paper trading accounts for each strategy
        for strategy_name, strategy in self.active_strategies.items():
            paper_account = PaperTradingAccount(
                strategy_id=strategy.config.strategy_id,
                initial_balance=1000000.0,  # $1M starting balance
                audit_logger=self.audit_logger
            )
            strategy.paper_account = paper_account

        self.paper_trading_enabled = True
        logger.info("Paper trading environment initialized")

    async def _monitor_strategies(self):
        """Continuously monitor and execute strategies"""
        logger.info("Starting strategy monitoring loop...")

        while True:
            try:
                # Generate signals from all strategies
                await self._generate_all_signals()

                # Execute signals in paper trading environment
                if self.paper_trading_enabled:
                    await self._execute_paper_trades()

                # Update performance metrics
                await self._update_performance_metrics()

                # Risk management checks
                await self._perform_risk_checks()

                # Wait before next cycle
                await asyncio.sleep(60)  # 1-minute cycles

            except Exception as e:
                logger.error(f"Error in strategy monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _generate_all_signals(self):
        """Generate trading signals from all active strategies"""
        for strategy_name, strategy in self.active_strategies.items():
            try:
                signals = await strategy.generate_signals()
                self.strategy_signals[strategy_name] = signals

                # Update performance tracking
                self.strategy_performance[strategy_name]['signals_generated'] += len(signals)

                # Log signals
                for signal in signals:
                    await self.audit_logger.log_event(
                        'strategy_signal',
                        f'{strategy_name} generated signal: {signal.signal_type} {signal.quantity} {signal.symbol}',
                        {
                            'strategy': strategy_name,
                            'signal_type': signal.signal_type.value,
                            'symbol': signal.symbol,
                            'quantity': signal.quantity,
                            'confidence': signal.confidence
                        }
                    )

            except Exception as e:
                logger.error(f"Error generating signals for {strategy_name}: {e}")

    async def _execute_paper_trades(self):
        """Execute signals in paper trading environment"""
        for strategy_name, signals in self.strategy_signals.items():
            strategy = self.active_strategies[strategy_name]
            paper_account = strategy.paper_account

            for signal in signals:
                try:
                    # Execute trade in paper account
                    trade_result = await paper_account.execute_trade(signal)

                    if trade_result['success']:
                        self.strategy_performance[strategy_name]['trades_executed'] += 1

                        await self.audit_logger.log_event(
                            'paper_trade_executed',
                            f'{strategy_name} executed paper trade: {signal.signal_type} {signal.quantity} {signal.symbol}',
                            {
                                'strategy': strategy_name,
                                'symbol': signal.symbol,
                                'quantity': signal.quantity,
                                'price': trade_result['price'],
                                'pnl': trade_result.get('pnl', 0)
                            }
                        )

                except Exception as e:
                    logger.error(f"Error executing paper trade for {strategy_name}: {e}")

    async def _update_performance_metrics(self):
        """Update strategy performance metrics"""
        for strategy_name, strategy in self.active_strategies.items():
            try:
                paper_account = strategy.paper_account
                performance = await paper_account.get_performance_metrics()

                self.strategy_performance[strategy_name].update({
                    'pnl_realized': performance.get('total_pnl', 0),
                    'sharpe_ratio': performance.get('sharpe_ratio', 0),
                    'max_drawdown': performance.get('max_drawdown', 0),
                    'current_balance': performance.get('current_balance', 1000000.0)
                })

            except Exception as e:
                logger.error(f"Error updating performance for {strategy_name}: {e}")

    async def _perform_risk_checks(self):
        """Perform risk management checks across all strategies"""
        total_system_pnl = sum(p['pnl_realized'] for p in self.strategy_performance.values())
        total_signals = sum(p['signals_generated'] for p in self.strategy_performance.values())

        # Update system metrics
        self.system_metrics = {
            'total_strategies': len(self.active_strategies),
            'active_strategies': len([s for s in self.active_strategies.values() if s.is_active]),
            'total_signals_today': total_signals,
            'total_pnl': total_system_pnl,
            'system_pnl': total_system_pnl,  # Alias for risk management compatibility
            'average_pnl_per_strategy': total_system_pnl / len(self.active_strategies) if self.active_strategies else 0
        }

        # Risk alerts
        if total_system_pnl < -50000:  # $50K loss threshold
            await self._trigger_risk_alert('system_loss_threshold', total_system_pnl)

        # Strategy-specific risk checks
        for strategy_name, performance in self.strategy_performance.items():
            if performance['max_drawdown'] > 0.1:  # 10% drawdown
                await self._trigger_risk_alert('strategy_drawdown', strategy_name, performance['max_drawdown'])

    async def _trigger_risk_alert(self, alert_type: str, *args):
        """Trigger risk management alert"""
        alert_message = f"Risk Alert: {alert_type.upper()}"
        alert_data = {'alert_type': alert_type, 'timestamp': datetime.now()}

        if alert_type == 'system_loss_threshold':
            alert_data['total_loss'] = args[0]
        elif alert_type == 'strategy_drawdown':
            alert_data['strategy'] = args[0]
            alert_data['drawdown'] = args[1]

        await self.audit_logger.log_event('risk_alert', alert_message, alert_data)

        # Send alert through communication system
        await self.communication.send_message(
            sender='StrategyIntegrationSystem',
            recipient='RiskManagement',
            message_type='risk_alert',
            payload=alert_data
        )

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'active_strategies': len(self.active_strategies),
            'paper_trading_enabled': self.paper_trading_enabled,
            'system_metrics': self.system_metrics,
            'strategy_performance': self.strategy_performance,
            'last_update': datetime.now()
        }

    async def enable_live_trading(self, strategy_name: Optional[str] = None):
        """Enable live trading for strategies (with safeguards)"""
        if strategy_name:
            if strategy_name in self.active_strategies:
                strategy = self.active_strategies[strategy_name]
                strategy.live_trading_enabled = True
                logger.warning(f"⚠️ LIVE TRADING ENABLED for {strategy_name}")
        else:
            # Enable for all strategies (dangerous!)
            logger.critical("🚨 LIVE TRADING ENABLED FOR ALL STRATEGIES - EXTREME RISK")
            for strategy in self.active_strategies.values():
                strategy.live_trading_enabled = True

    async def shutdown_system(self):
        """Gracefully shutdown the strategy integration system"""
        logger.info("Shutting down Strategy Integration System...")

        # Close all positions
        for strategy_name, strategy in self.active_strategies.items():
            try:
                await strategy.close_all_positions()
            except Exception as e:
                logger.error(f"Error closing positions for {strategy_name}: {e}")

        # Save final performance metrics
        await self._save_performance_snapshot()

        logger.info("Strategy Integration System shutdown complete")

    async def _save_performance_snapshot(self):
        """Save final performance snapshot to disk"""
        try:
            snapshot = await self.get_system_status()
            snapshot_file = Path("performance_snapshot.json")
            import json
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            logger.info(f"Performance snapshot saved to {snapshot_file}")
        except Exception as e:
            logger.error(f"Error saving performance snapshot: {e}")


class PaperTradingAccount:
    """Paper trading account for safe strategy testing"""

    def __init__(self, strategy_id: str, initial_balance: float, audit_logger: AuditLogger):
        self.strategy_id = strategy_id
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}  # symbol -> position data
        self.trade_history = []
        self.audit_logger = audit_logger

    async def execute_trade(self, signal: TradingSignal) -> Dict[str, Any]:
        """Execute a trade in the paper account"""
        try:
            symbol = signal.symbol
            quantity = signal.quantity
            signal_type = signal.signal_type

            # Get current market price
            current_price = await self._get_market_price(symbol)
            if not current_price:
                return {'success': False, 'error': 'No market price available'}

            # Calculate trade value
            trade_value = abs(quantity) * current_price

            # Check if we have sufficient balance for buy orders
            if signal_type in [SignalType.LONG, SignalType.BUY] and trade_value > self.balance:
                return {'success': False, 'error': 'Insufficient balance'}

            # Execute the trade
            if signal_type in [SignalType.LONG, SignalType.BUY]:
                # Buy order
                self.balance -= trade_value
                self._update_position(symbol, quantity, current_price)

            elif signal_type in [SignalType.SHORT, SignalType.SELL]:
                # Sell order (simplified - assuming we can short)
                self.balance += trade_value  # Credit from short sale
                self._update_position(symbol, -quantity, current_price)

            elif signal_type == SignalType.CLOSE:
                # Close position
                await self._close_position(symbol, current_price)

            # Record trade
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'signal_type': signal_type.value,
                'quantity': quantity,
                'price': current_price,
                'value': trade_value,
                'balance_after': self.balance
            }
            self.trade_history.append(trade_record)

            return {
                'success': True,
                'price': current_price,
                'value': trade_value,
                'balance': self.balance
            }

        except Exception as e:
            logger.error(f"Error executing paper trade: {e}")
            return {'success': False, 'error': str(e)}

    def _update_position(self, symbol: str, quantity: int, price: float):
        """Update position for a symbol"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0.0,
                'unrealized_pnl': 0.0
            }

        position = self.positions[symbol]
        current_qty = position['quantity']
        current_avg = position['avg_price']

        # Calculate new average price
        if current_qty + quantity != 0:
            total_cost = (current_qty * current_avg) + (quantity * price)
            new_qty = current_qty + quantity
            new_avg = total_cost / new_qty if new_qty != 0 else 0
        else:
            new_qty = 0
            new_avg = 0

        position['quantity'] = new_qty
        position['avg_price'] = new_avg

        # Remove position if flat
        if new_qty == 0:
            del self.positions[symbol]

    async def _close_position(self, symbol: str, price: float):
        """Close position in a symbol"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        quantity = position['quantity']
        avg_price = position['avg_price']

        # Calculate P&L
        pnl = (price - avg_price) * abs(quantity)
        self.balance += pnl

        # Remove position
        del self.positions[symbol]

    async def _get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        # In production, this would query live market data
        # For now, return a mock price
        return 100.0 + np.random.normal(0, 5)  # Mock price around $100

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the account"""
        total_pnl = self.balance - self.initial_balance
        returns = total_pnl / self.initial_balance

        # Calculate Sharpe ratio (simplified)
        if self.trade_history:
            daily_returns = [t.get('pnl', 0) for t in self.trade_history[-30:]]  # Last 30 trades
            if daily_returns:
                sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Calculate max drawdown (simplified)
        balance_history = [self.initial_balance] + [t['balance_after'] for t in self.trade_history]
        peak = max(balance_history)
        trough = min(balance_history)
        max_drawdown = (peak - trough) / peak if peak > 0 else 0

        return {
            'total_pnl': total_pnl,
            'total_return': returns,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'current_balance': self.balance,
            'total_trades': len(self.trade_history),
            'open_positions': len(self.positions)
        }


# Global system instance
_strategy_system_instance = None

def get_strategy_integration_system(data_aggregator: MarketDataAggregator = None,
                                   communication: CommunicationFramework = None,
                                   audit_logger: AuditLogger = None) -> StrategyIntegrationSystem:
    """Get singleton strategy integration system instance"""
    global _strategy_system_instance

    if _strategy_system_instance is None:
        if not all([data_aggregator, communication, audit_logger]):
            raise ValueError("All dependencies required for first system instantiation")

        _strategy_system_instance = StrategyIntegrationSystem(
            data_aggregator, communication, audit_logger
        )

    return _strategy_system_instance