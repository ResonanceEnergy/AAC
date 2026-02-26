"""
Mean Reversion Strategy (Statistical Arbitrage)
===============================================

Strategy ID: s52_mean_reversion
Description: Identify assets that have deviated significantly from their historical mean and bet on reversion to the mean.

Key Components:
- Rolling mean and standard deviation calculation
- Z-score based entry/exit signals
- Multiple timeframe analysis
- Volume confirmation
- Risk management with position sizing
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


class MeanReversionStrategy(BaseArbitrageStrategy):
    """
    Mean Reversion Strategy Implementation.

    This strategy identifies assets that have deviated significantly from their
    historical mean price and bets on reversion to the mean. The strategy goes
    short (long) assets that are significantly above (below) their historical
    average, expecting price normalization.

    Key Parameters:
    - Lookback period: 20-50 trading days for mean calculation
    - Entry threshold: 2-3 standard deviations from mean
    - Exit threshold: 0.5 standard deviations from mean
    - Volume confirmation: Above average volume required
    - Maximum holding period: 5 trading days
    """

    def __init__(self, config: StrategyConfig, communication: CommunicationFramework,
                 audit_logger: AuditLogger):
        super().__init__(config, communication, audit_logger)

        # Strategy-specific parameters
        self.lookback_periods = [20, 50]  # Multiple lookback periods for robustness
        self.entry_threshold_std = 2.5  # Entry when price > 2.5 std dev from mean
        self.exit_threshold_std = 0.3  # Exit when price < 0.3 std dev from mean
        self.volume_multiplier = 1.2  # Volume must be 1.2x average
        self.max_holding_days = 5  # Maximum holding period
        self.min_price = 5.0  # Minimum price to avoid penny stocks
        self.max_price = 1000.0  # Maximum price filter

        # Universe of tradable assets
        self.universe = [
            # Large-cap stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
            'KO', 'PEP', 'MCD', 'WMT', 'HD', 'COST',
            'JNJ', 'PFE', 'MRK', 'ABT', 'TMO',
            'XOM', 'CVX', 'COP', 'EOG',

            # ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'EFA', 'VWO', 'BND', 'VNQ'
        ]

        # Asset statistics tracking
        self.asset_statistics = {}
        self.open_positions = {}  # symbol -> position data

    async def initialize(self) -> bool:
        """Initialize the mean reversion strategy."""
        try:
            logger.info("Initializing Mean Reversion Strategy...")

            # Initialize asset statistics
            await self._initialize_asset_statistics()

            # Validate market data availability
            await self._validate_data_availability()

            logger.info(f"Mean Reversion Strategy initialized with {len(self.asset_statistics)} assets")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Mean Reversion Strategy: {e}")
            return False

    async def _initialize_asset_statistics(self):
        """Calculate historical statistics for universe assets."""
        logger.info("Calculating asset statistics...")

        for symbol in self.universe:
            try:
                stats = await self._calculate_asset_statistics(symbol)
                if stats and stats['is_valid']:
                    self.asset_statistics[symbol] = stats
                    logger.info(f"Initialized {symbol}: mean=${stats['price_mean']:.2f}, "
                              f"std=${stats['price_std']:.2f}")

            except Exception as e:
                logger.warning(f"Failed to initialize asset {symbol}: {e}")

    async def _calculate_asset_statistics(self, symbol: str) -> Optional[Dict]:
        """Calculate statistical properties of an asset."""
        try:
            # Get historical price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)  # 100 trading days

            data = await self.market_data.get_historical_prices(
                symbol, start_date, end_date, interval='1d'
            )

            if not data or len(data) < 60:  # Need at least 60 days
                return None

            # Extract prices and volumes
            prices = [d['close'] for d in data]
            volumes = [d['volume'] for d in data]
            timestamps = [d['timestamp'] for d in data]

            # Convert to pandas for easier analysis
            df = pd.DataFrame({
                'price': prices,
                'volume': volumes
            }, index=timestamps)

            # Calculate statistics for each lookback period
            stats = {}
            valid_periods = 0

            for period in self.lookback_periods:
                if len(df) >= period:
                    # Rolling statistics
                    rolling_mean = df['price'].rolling(window=period).mean()
                    rolling_std = df['price'].rolling(window=period).std()
                    rolling_volume_mean = df['volume'].rolling(window=period).mean()

                    # Current values
                    current_price = df['price'].iloc[-1]
                    current_volume = df['volume'].iloc[-1]
                    current_mean = rolling_mean.iloc[-1]
                    current_std = rolling_std.iloc[-1]
                    current_volume_mean = rolling_volume_mean.iloc[-1]

                    # Z-score
                    z_score = (current_price - current_mean) / current_std if current_std > 0 else 0

                    stats[period] = {
                        'mean': current_mean,
                        'std': current_std,
                        'z_score': z_score,
                        'volume_mean': current_volume_mean,
                        'volume_ratio': current_volume / current_volume_mean if current_volume_mean > 0 else 1
                    }

                    valid_periods += 1

            if valid_periods == 0:
                return None

            # Overall statistics
            price_mean = np.mean(prices)
            price_std = np.std(prices)
            volume_mean = np.mean(volumes)

            # Check if asset meets criteria
            current_price = prices[-1]
            is_valid = (
                self.min_price <= current_price <= self.max_price and
                price_std > 0 and
                len(prices) >= 60
            )

            return {
                'symbol': symbol,
                'price_mean': price_mean,
                'price_std': price_std,
                'volume_mean': volume_mean,
                'current_price': current_price,
                'lookback_stats': stats,
                'data_points': len(prices),
                'is_valid': is_valid,
                'last_update': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error calculating statistics for {symbol}: {e}")
            return None

    async def _validate_data_availability(self):
        """Validate that market data is available for assets."""
        for symbol in list(self.asset_statistics.keys()):
            try:
                # Test real-time data availability
                data = await self.market_data.get_real_time_price(symbol)
                if not data:
                    logger.warning(f"Real-time data not available for {symbol}")
                    del self.asset_statistics[symbol]

            except Exception as e:
                logger.warning(f"Data validation failed for {symbol}: {e}")
                del self.asset_statistics[symbol]

    async def generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals based on mean reversion opportunities."""
        signals = []

        try:
            for symbol, stats in self.asset_statistics.items():
                symbol_signals = await self._analyze_asset(symbol, stats)
                signals.extend(symbol_signals)

                # Check for exit signals on open positions
                exit_signals = await self._check_exit_signals(symbol, stats)
                signals.extend(exit_signals)

        except Exception as e:
            logger.error(f"Error generating signals: {e}")

        return signals

    async def _analyze_asset(self, symbol: str, stats: Dict) -> List[TradingSignal]:
        """Analyze a single asset for mean reversion opportunities."""
        signals = []

        try:
            # Get current price and volume
            current_data = await self.market_data.get_real_time_price(symbol)
            if not current_data:
                return signals

            current_price = current_data['price']
            current_volume = current_data.get('volume', 0)

            # Analyze each lookback period
            consensus_z_score = 0
            valid_periods = 0
            volume_confirmed = False

            for period, period_stats in stats['lookback_stats'].items():
                z_score = period_stats['z_score']
                volume_ratio = period_stats['volume_ratio']

                consensus_z_score += z_score
                valid_periods += 1

                # Check volume confirmation
                if volume_ratio >= self.volume_multiplier:
                    volume_confirmed = True

            if valid_periods == 0:
                return signals

            consensus_z_score /= valid_periods

            # Check for entry signals
            if abs(consensus_z_score) > self.entry_threshold_std and volume_confirmed:
                # Determine trade direction
                if consensus_z_score > 0:
                    # Price is significantly above mean - SHORT
                    signal_type = SignalType.SHORT
                    confidence = min(abs(consensus_z_score) / 4.0, 1.0)
                else:
                    # Price is significantly below mean - LONG
                    signal_type = SignalType.LONG
                    confidence = min(abs(consensus_z_score) / 4.0, 1.0)

                # Calculate position size
                portfolio_value = self.config.capital_allocation
                max_position_value = portfolio_value * self.config.max_position_size_pct
                quantity = max_position_value / current_price

                # Risk management - limit position size
                max_quantity = portfolio_value * 0.1 / current_price  # Max 10% of portfolio
                quantity = min(quantity, max_quantity)

                # Create signal
                signal = TradingSignal(
                    strategy_id=self.strategy_id,
                    signal_type=signal_type,
                    symbol=symbol,
                    quantity=quantity,
                    price=current_price,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    metadata={
                        'z_score': consensus_z_score,
                        'volume_ratio': volume_confirmed,
                        'lookback_periods': list(stats['lookback_stats'].keys()),
                        'price_mean': stats['price_mean'],
                        'price_std': stats['price_std']
                    }
                )

                signals.append(signal)
                logger.info(f"Generated {signal_type.value} signal for {symbol}: "
                          f"z-score={consensus_z_score:.2f}, confidence={confidence:.2f}")

        except Exception as e:
            logger.error(f"Error analyzing asset {symbol}: {e}")

        return signals

    async def _check_exit_signals(self, symbol: str, stats: Dict) -> List[TradingSignal]:
        """Check for exit signals on open positions."""
        signals = []

        try:
            if symbol not in self.open_positions:
                return signals

            position = self.open_positions[symbol]

            # Get current price
            current_data = await self.market_data.get_real_time_price(symbol)
            if not current_data:
                return signals

            current_price = current_data['price']

            # Calculate current z-score
            consensus_z_score = 0
            valid_periods = 0

            for period, period_stats in stats['lookback_stats'].items():
                # Recalculate z-score with current price
                current_z = (current_price - period_stats['mean']) / period_stats['std'] if period_stats['std'] > 0 else 0
                consensus_z_score += current_z
                valid_periods += 1

            if valid_periods > 0:
                consensus_z_score /= valid_periods

            # Check exit conditions
            should_exit = (
                abs(consensus_z_score) < self.exit_threshold_std or  # Price reverted to mean
                (datetime.now() - position['entry_time']).days >= self.max_holding_days  # Max holding period
            )

            if should_exit:
                # Determine exit signal type (opposite of entry)
                if position['position_type'] == 'long':
                    exit_signal_type = SignalType.SELL
                else:
                    exit_signal_type = SignalType.BUY

                # Calculate P&L
                if position['position_type'] == 'long':
                    pnl = (current_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - current_price) * position['quantity']

                # Create exit signal
                signal = TradingSignal(
                    strategy_id=self.strategy_id,
                    signal_type=exit_signal_type,
                    symbol=symbol,
                    quantity=position['quantity'],
                    price=current_price,
                    confidence=1.0,
                    timestamp=datetime.now(),
                    metadata={
                        'exit_reason': 'reversion' if abs(consensus_z_score) < self.exit_threshold_std else 'max_holding_period',
                        'z_score': consensus_z_score,
                        'pnl': pnl,
                        'holding_days': (datetime.now() - position['entry_time']).days
                    }
                )

                signals.append(signal)

                # Remove from open positions
                del self.open_positions[symbol]

                logger.info(f"Generated exit signal for {symbol}: {signal.metadata['exit_reason']}, "
                          f"P&L = ${pnl:.2f}")

        except Exception as e:
            logger.error(f"Error checking exit signals for {symbol}: {e}")

        return signals

    async def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute a trading signal."""
        try:
            if signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
                # Opening position
                await self._execute_entry(signal)
            elif signal.signal_type in [SignalType.SELL, SignalType.BUY]:
                # Closing position
                await self._execute_exit(signal)

            return True

        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False

    async def _execute_entry(self, signal: TradingSignal):
        """Execute position entry."""
        try:
            position_type = 'long' if signal.signal_type == SignalType.LONG else 'short'

            # Record position
            self.open_positions[signal.symbol] = {
                'entry_time': datetime.now(),
                'position_type': position_type,
                'quantity': signal.quantity,
                'entry_price': signal.price,
                'z_score': signal.metadata['z_score']
            }

            logger.info(f"Opened {position_type} position in {signal.symbol}: "
                       f"{signal.quantity:.0f} shares @ ${signal.price:.2f}")

        except Exception as e:
            logger.error(f"Error executing entry: {e}")

    async def _execute_exit(self, signal: TradingSignal):
        """Execute position exit."""
        try:
            pnl = signal.metadata.get('pnl', 0)
            logger.info(f"Closed position in {signal.symbol}: P&L = ${pnl:.2f}")

        except Exception as e:
            logger.error(f"Error executing exit: {e}")

    async def update_statistics(self):
        """Update asset statistics periodically."""
        try:
            # Update statistics every 4 hours
            for symbol in list(self.asset_statistics.keys()):
                stats = await self._calculate_asset_statistics(symbol)
                if stats and stats['is_valid']:
                    self.asset_statistics[symbol] = stats
                else:
                    # Remove invalid assets
                    del self.asset_statistics[symbol]

        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status and metrics."""
        return {
            'strategy_id': self.strategy_id,
            'assets_tracked': len(self.asset_statistics),
            'open_positions': len(self.open_positions),
            'universe_size': len(self.universe),
            'lookback_periods': self.lookback_periods,
            'entry_threshold': self.entry_threshold_std,
            'exit_threshold': self.exit_threshold_std,
            'position_summary': {
                symbol: {
                    'type': pos['position_type'],
                    'quantity': pos['quantity'],
                    'entry_price': pos['entry_price'],
                    'holding_days': (datetime.now() - pos['entry_time']).days
                }
                for symbol, pos in self.open_positions.items()
            }
        }