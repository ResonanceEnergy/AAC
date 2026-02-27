"""
Paper Trading Environment with Premium Data Feeds
==================================================

Strategy ID: s55_paper_trading_premium
Description: Simulated trading environment using premium market data feeds.

Key Components:
- Real-time premium data integration
- Simulated order execution
- Portfolio tracking and P&L calculation
- Risk management simulation
- Performance analytics and reporting
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from shared.strategy_framework import BaseArbitrageStrategy, TradingSignal, SignalType, StrategyConfig
from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class PaperTradingPremiumStrategy(BaseArbitrageStrategy):
    """
    Paper Trading Strategy with Premium Data Feeds.

    This strategy provides a simulated trading environment that:
    1. Uses premium market data for signal generation
    2. Simulates order execution with realistic slippage and fees
    3. Tracks portfolio performance and risk metrics
    4. Provides detailed analytics and reporting
    5. Tests strategies in a risk-free environment

    Key Features:
    - Real-time premium data integration
    - Realistic execution simulation
    - Comprehensive performance tracking
    - Risk management validation
    - Strategy backtesting capabilities
    """

    def __init__(self, config: StrategyConfig, communication: CommunicationFramework,
                 audit_logger: AuditLogger):
        super().__init__(config, communication, audit_logger)

        # Paper trading parameters
        self.initial_capital = config.capital_allocation
        self.current_capital = self.initial_capital
        self.portfolio = {}  # Symbol -> quantity mapping
        self.trades_history = []  # List of executed trades
        self.daily_pnl = []  # Daily P&L tracking

        # Execution simulation parameters
        self.slippage_model = 'realistic'  # 'none', 'fixed', 'realistic'
        self.fixed_slippage = 0.0005  # 0.05% for fixed slippage
        self.commission_per_trade = 0.005  # $0.005 per share
        self.min_commission = 1.0  # Minimum commission
        self.max_slippage_pct = 0.002  # Maximum 0.2% slippage

        # Risk management
        self.max_drawdown_limit = 0.1  # 10% maximum drawdown
        self.daily_loss_limit = 0.05   # 5% daily loss limit
        self.max_position_size_pct = 0.1  # 10% max position size

        # Performance tracking
        self.start_time = datetime.now()
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0

        # Premium data validation
        self.premium_data_sources = [
            'polygon', 'finnhub', 'iex', 'twelve_data', 'intrinio', 'alpha_vantage'
        ]
        self.data_quality_checks = {}

    async def initialize(self) -> bool:
        """Initialize the paper trading environment."""
        try:
            logger.info("Initializing Paper Trading Environment with Premium Data...")

            # Validate premium data connectivity
            await self._validate_premium_data()

            # Initialize portfolio
            self.portfolio = {}
            self.trades_history = []
            self.daily_pnl = []

            # Set up performance tracking
            self.start_time = datetime.now()
            self.peak_capital = self.initial_capital
            self.current_drawdown = 0.0

            logger.info(f"Paper Trading initialized with ${self.initial_capital:,.2f} capital")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Paper Trading: {e}")
            return False

    async def _validate_premium_data(self):
        """Validate premium data source connectivity."""
        logger.info("Validating premium data sources...")

        for source in self.premium_data_sources:
            try:
                # Test basic connectivity
                test_data = await self.market_data.get_real_time_price('AAPL', source=source)
                if test_data and 'price' in test_data:
                    self.data_quality_checks[source] = {
                        'status': 'active',
                        'last_check': datetime.now(),
                        'sample_price': test_data['price']
                    }
                    logger.info(f"✓ {source}: Connected - AAPL @ ${test_data['price']:.2f}")
                else:
                    self.data_quality_checks[source] = {
                        'status': 'inactive',
                        'last_check': datetime.now(),
                        'error': 'No data received'
                    }
                    logger.warning(f"✗ {source}: No data available")
            except Exception as e:
                self.data_quality_checks[source] = {
                    'status': 'error',
                    'last_check': datetime.now(),
                    'error': str(e)
                }
                logger.warning(f"✗ {source}: Error - {e}")

    async def generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals using premium data."""
        signals = []

        try:
            # Get market overview using premium data
            market_data = await self._get_market_overview()

            if not market_data:
                logger.warning("No premium market data available")
                return signals

            # Generate signals based on premium data analysis
            momentum_signals = await self._generate_momentum_signals(market_data)
            signals.extend(momentum_signals)

            mean_reversion_signals = await self._generate_mean_reversion_signals(market_data)
            signals.extend(mean_reversion_signals)

            arbitrage_signals = await self._generate_arbitrage_signals(market_data)
            signals.extend(arbitrage_signals)

            # Apply risk management filters
            filtered_signals = await self._apply_risk_filters(signals)

            return filtered_signals

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []

    async def _get_market_overview(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive market overview using premium data."""
        try:
            # Major indices
            indices = ['SPY', 'QQQ', 'IWM', 'DIA']
            index_data = {}

            for symbol in indices:
                data = await self.market_data.get_real_time_price(symbol)
                if data:
                    index_data[symbol] = data

            # Major forex pairs
            forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
            forex_data = {}

            for pair in forex_pairs:
                data = await self.market_data.get_real_time_price(pair)
                if data:
                    forex_data[pair] = data

            # Crypto assets
            crypto_assets = ['BTC', 'ETH', 'BNB', 'ADA']
            crypto_data = {}

            for asset in crypto_assets:
                data = await self.market_data.get_real_time_price(asset)
                if data:
                    crypto_data[asset] = data

            return {
                'indices': index_data,
                'forex': forex_data,
                'crypto': crypto_data,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return None

    async def _generate_momentum_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate momentum-based signals using premium data."""
        signals = []

        try:
            # Analyze index momentum
            indices = market_data.get('indices', {})

            for symbol, data in indices.items():
                if 'change_pct' in data:
                    change_pct = data['change_pct']

                    # Strong upward momentum
                    if change_pct > 1.0:  # >1% gain
                        signal = TradingSignal(
                            strategy_id=self.strategy_id,
                            signal_type=SignalType.LONG,
                            symbol=symbol,
                            quantity=self._calculate_position_size(symbol, data['price']),
                            price=data['price'],
                            confidence=min(abs(change_pct) / 5.0, 1.0),  # Scale by 5%
                            timestamp=datetime.now(),
                            metadata={
                                'signal_type': 'momentum',
                                'change_pct': change_pct,
                                'market_data_source': 'premium'
                            }
                        )
                        signals.append(signal)

                    # Strong downward momentum (short opportunity)
                    elif change_pct < -1.0:  # >1% loss
                        signal = TradingSignal(
                            strategy_id=self.strategy_id,
                            signal_type=SignalType.SHORT,
                            symbol=symbol,
                            quantity=self._calculate_position_size(symbol, data['price']),
                            price=data['price'],
                            confidence=min(abs(change_pct) / 5.0, 1.0),
                            timestamp=datetime.now(),
                            metadata={
                                'signal_type': 'momentum_short',
                                'change_pct': change_pct,
                                'market_data_source': 'premium'
                            }
                        )
                        signals.append(signal)

        except Exception as e:
            logger.error(f"Error generating momentum signals: {e}")

        return signals

    async def _generate_mean_reversion_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate mean reversion signals using premium data."""
        signals = []

        try:
            # Analyze forex mean reversion opportunities
            forex_data = market_data.get('forex', {})

            for pair, data in forex_data.items():
                if 'price' in data:
                    price = data['price']

                    # Simple mean reversion: if price deviates significantly from recent average
                    # In a real implementation, this would use historical data
                    # For simulation, we'll use a basic threshold

                    # Assume fair value around current price ±2%
                    fair_value = price * (1 + np.random.uniform(-0.02, 0.02))

                    deviation = (price - fair_value) / fair_value

                    if abs(deviation) > 0.015:  # 1.5% deviation
                        if deviation > 0:
                            # Price above fair value - expect reversion down
                            signal_type = SignalType.SHORT
                        else:
                            # Price below fair value - expect reversion up
                            signal_type = SignalType.LONG

                        signal = TradingSignal(
                            strategy_id=self.strategy_id,
                            signal_type=signal_type,
                            symbol=pair,
                            quantity=self._calculate_position_size(pair, price),
                            price=price,
                            confidence=min(abs(deviation) / 0.05, 1.0),  # Scale by 5%
                            timestamp=datetime.now(),
                            metadata={
                                'signal_type': 'mean_reversion',
                                'deviation': deviation,
                                'fair_value': fair_value,
                                'market_data_source': 'premium'
                            }
                        )
                        signals.append(signal)

        except Exception as e:
            logger.error(f"Error generating mean reversion signals: {e}")

        return signals

    async def _generate_arbitrage_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate arbitrage signals using premium data."""
        signals = []

        try:
            # Cross-market arbitrage: Check for price discrepancies
            indices = market_data.get('indices', {})
            crypto = market_data.get('crypto', {})

            # Example: SPY vs QQQ ratio arbitrage
            if 'SPY' in indices and 'QQQ' in indices:
                spy_price = indices['SPY'].get('price', 0)
                qqq_price = indices['QQQ'].get('price', 0)

                if spy_price > 0 and qqq_price > 0:
                    # Typical SPY/QQQ ratio is around 1.3-1.4
                    current_ratio = spy_price / qqq_price
                    fair_ratio = 1.35  # Simplified fair value

                    deviation = (current_ratio - fair_ratio) / fair_ratio

                    if abs(deviation) > 0.02:  # 2% deviation
                        # If ratio is high, SPY is expensive relative to QQQ
                        if deviation > 0:
                            signal = TradingSignal(
                                strategy_id=self.strategy_id,
                                signal_type=SignalType.ARBITRAGE,
                                symbol="SPY_QQQ_arb",
                                quantity=self._calculate_position_size("SPY", spy_price),
                                price=spy_price,
                                confidence=min(abs(deviation) / 0.1, 1.0),
                                timestamp=datetime.now(),
                                metadata={
                                    'arbitrage_type': 'ratio_arbitrage',
                                    'ratio': current_ratio,
                                    'fair_ratio': fair_ratio,
                                    'deviation': deviation,
                                    'market_data_source': 'premium'
                                }
                            )
                            signals.append(signal)

        except Exception as e:
            logger.error(f"Error generating arbitrage signals: {e}")

        return signals

    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate appropriate position size based on risk management."""
        try:
            # Maximum position size as percentage of capital
            max_position_value = self.current_capital * self.max_position_size_pct

            # Calculate quantity based on price
            if price > 0:
                max_quantity = max_position_value / price

                # For crypto/forex, use smaller position sizes
                if len(symbol) <= 6 and not symbol.isalpha():  # Likely crypto/forex
                    max_quantity *= 0.1  # 10% of normal size

                return min(max_quantity, 10000)  # Cap at 10k units
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    async def _apply_risk_filters(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Apply risk management filters to signals."""
        filtered_signals = []

        try:
            # Check drawdown limit
            if self.current_drawdown > self.max_drawdown_limit:
                logger.warning(f"Drawdown limit exceeded: {self.current_drawdown:.4f}")
                return []

            # Check daily loss limit
            today_pnl = sum([pnl for pnl in self.daily_pnl if pnl < 0])
            if abs(today_pnl) > self.initial_capital * self.daily_loss_limit:
                logger.warning(f"Daily loss limit exceeded: ${abs(today_pnl):.2f}")
                return []

            # Filter signals based on portfolio constraints
            for signal in signals:
                # Check if we have enough capital
                position_value = signal.quantity * signal.price
                if position_value > self.current_capital * 0.8:  # Max 80% of capital per trade
                    continue

                # Check position concentration
                existing_position = self.portfolio.get(signal.symbol, 0)
                total_position = existing_position + signal.quantity
                total_position_value = total_position * signal.price

                if total_position_value > self.current_capital * self.max_position_size_pct:
                    continue

                filtered_signals.append(signal)

        except Exception as e:
            logger.error(f"Error applying risk filters: {e}")

        return filtered_signals

    async def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute a simulated trade."""
        try:
            # Simulate order execution with slippage and fees
            executed_price = await self._simulate_execution(signal)

            if executed_price <= 0:
                logger.warning(f"Failed to execute trade for {signal.symbol}")
                return False

            # Calculate trade value and fees
            trade_value = signal.quantity * executed_price
            commission = max(self.commission_per_trade * abs(signal.quantity), self.min_commission)

            # Update portfolio
            if signal.symbol in self.portfolio:
                self.portfolio[signal.symbol] += signal.quantity
            else:
                self.portfolio[signal.symbol] = signal.quantity

            # Update capital (simplified - doesn't account for margin/leverage)
            if signal.signal_type in [SignalType.LONG, SignalType.ARBITRAGE]:
                self.current_capital -= trade_value + commission
            elif signal.signal_type == SignalType.SHORT:
                self.current_capital += trade_value - commission  # Short sale proceeds

            # Record trade
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': signal.symbol,
                'signal_type': signal.signal_type.value,
                'quantity': signal.quantity,
                'signal_price': signal.price,
                'executed_price': executed_price,
                'commission': commission,
                'trade_value': trade_value,
                'capital_after': self.current_capital
            }

            self.trades_history.append(trade_record)

            # Log trade execution
            await self.audit_logger.log_event(
                'paper_trade_execution',
                trade_record
            )

            logger.info(f"Paper trade executed: {signal.symbol} {signal.signal_type.value} "
                       f"{signal.quantity:.2f} @ ${executed_price:.4f} "
                       f"(Commission: ${commission:.2f})")

            return True

        except Exception as e:
            logger.error(f"Error executing paper trade: {e}")
            return False

    async def _simulate_execution(self, signal: TradingSignal) -> float:
        """Simulate realistic order execution with slippage."""
        try:
            base_price = signal.price

            if self.slippage_model == 'none':
                return base_price
            elif self.slippage_model == 'fixed':
                slippage = self.fixed_slippage
            else:  # realistic
                # Volume-based slippage simulation
                volume_factor = np.random.uniform(0.5, 2.0)  # Random volume
                slippage = min(self.max_slippage_pct, self.fixed_slippage / volume_factor)

            # Apply slippage
            if signal.signal_type == SignalType.LONG:
                executed_price = base_price * (1 + slippage)
            elif signal.signal_type == SignalType.SHORT:
                executed_price = base_price * (1 - slippage)
            else:
                executed_price = base_price

            return executed_price

        except Exception as e:
            logger.error(f"Error simulating execution: {e}")
            return 0.0

    async def update_statistics(self):
        """Update portfolio statistics and P&L."""
        try:
            # Calculate current portfolio value
            portfolio_value = 0.0

            for symbol, quantity in self.portfolio.items():
                if quantity != 0:
                    current_price = await self._get_current_price(symbol)
                    if current_price > 0:
                        portfolio_value += quantity * current_price

            # Calculate total value (cash + portfolio)
            total_value = self.current_capital + portfolio_value

            # Update drawdown
            if total_value > self.peak_capital:
                self.peak_capital = total_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_capital - total_value) / self.peak_capital

            # Calculate daily P&L
            current_day = datetime.now().date()
            daily_trades = [
                trade for trade in self.trades_history
                if trade['timestamp'].date() == current_day
            ]

            if daily_trades:
                daily_pnl = sum([
                    trade['trade_value'] * (-1 if trade['signal_type'] == 'short' else 1) - trade['commission']
                    for trade in daily_trades
                ])
                self.daily_pnl.append(daily_pnl)

        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            data = await self.market_data.get_real_time_price(symbol)
            return data.get('price', 0.0) if data else 0.0
        except Exception as e:
            logger.debug(f"Error getting current price for {symbol}: {e}")
            return 0.0

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current paper trading status and metrics."""
        try:
            # Calculate performance metrics
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            total_trades = len(self.trades_history)

            winning_trades = len([t for t in self.trades_history if t.get('profit', 0) > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # Calculate Sharpe ratio (simplified)
            if self.daily_pnl:
                avg_daily_return = np.mean(self.daily_pnl) / self.initial_capital
                std_daily_return = np.std(self.daily_pnl) / self.initial_capital
                sharpe_ratio = avg_daily_return / std_daily_return if std_daily_return > 0 else 0
            else:
                sharpe_ratio = 0

            return {
                'strategy_id': self.strategy_id,
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'total_return_pct': total_return * 100,
                'current_drawdown_pct': self.current_drawdown * 100,
                'total_trades': total_trades,
                'win_rate_pct': win_rate * 100,
                'sharpe_ratio': sharpe_ratio,
                'active_positions': len([p for p in self.portfolio.values() if p != 0]),
                'premium_data_sources': len([s for s in self.data_quality_checks.values() if s['status'] == 'active']),
                'running_days': (datetime.now() - self.start_time).days
            }

        except Exception as e:
            logger.error(f"Error getting strategy status: {e}")
            return {
                'strategy_id': self.strategy_id,
                'status': 'error',
                'error': str(e)
            }