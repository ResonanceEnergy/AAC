"""
Pairs Trading Strategy (Statistical Arbitrage)
==============================================

Strategy ID: s51_pairs_trading
Description: Identify correlated asset pairs and trade the spread when it deviates significantly from its historical mean.

Key Components:
- Pair selection based on cointegration and correlation
- Spread calculation and normalization
- Mean reversion signals on spread deviations
- Risk management with stop losses and position sizing
- Dynamic pair rebalancing and monitoring
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller

from shared.strategy_framework import BaseArbitrageStrategy, TradingSignal, SignalType, StrategyConfig
from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class PairsTradingStrategy(BaseArbitrageStrategy):
    """
    Pairs Trading Strategy Implementation.

    This strategy identifies pairs of correlated assets and trades the spread
    between them when it deviates significantly from its historical mean.
    The strategy goes long the underperforming asset and short the
    outperforming asset when the spread is wide, expecting convergence.

    Key Parameters:
    - Correlation threshold: > 0.7 for pair selection
    - Cointegration test: Augmented Dickey-Fuller test
    - Entry threshold: 2 standard deviations from mean
    - Exit threshold: 0.5 standard deviations from mean
    - Maximum holding period: 5 trading days
    """

    def __init__(self, config: StrategyConfig, communication: CommunicationFramework,
                 audit_logger: AuditLogger):
        super().__init__(config, communication, audit_logger)

        # Strategy-specific parameters
        self.correlation_threshold = 0.7  # Minimum correlation for pair consideration
        self.entry_threshold_std = 2.0  # Entry when spread > 2 std dev from mean
        self.exit_threshold_std = 0.5  # Exit when spread < 0.5 std dev from mean
        self.max_holding_days = 5  # Maximum holding period in trading days
        self.min_lookback_days = 60  # Minimum historical data for analysis
        self.max_lookback_days = 252  # Maximum historical data (1 year)

        # Pair universe - correlated assets
        self.potential_pairs = [
            # Technology sector pairs
            ('AAPL', 'MSFT'),  # Apple vs Microsoft
            ('GOOGL', 'AMZN'),  # Google vs Amazon
            ('NVDA', 'AMD'),  # NVIDIA vs AMD
            ('CRM', 'NOW'),  # Salesforce vs ServiceNow

            # Financial sector pairs
            ('JPM', 'BAC'),  # JPMorgan vs Bank of America
            ('WFC', 'C'),  # Wells Fargo vs Citigroup

            # Consumer sector pairs
            ('KO', 'PEP'),  # Coca-Cola vs Pepsi
            ('MCD', 'WMT'),  # McDonald's vs Walmart

            # ETF pairs
            ('SPY', 'QQQ'),  # S&P 500 vs Nasdaq 100
            ('IWM', 'VTI'),  # Russell 2000 vs Total Stock Market
        ]

        # Active pairs and their spread statistics
        self.active_pairs = {}
        self.pair_statistics = {}

        # Position tracking
        self.open_positions = {}  # pair_key -> position data

    async def initialize(self) -> bool:
        """Initialize the pairs trading strategy."""
        try:
            logger.info("Initializing Pairs Trading Strategy...")

            # Initialize pair statistics
            await self._initialize_pair_statistics()

            # Validate market data availability
            await self._validate_data_availability()

            logger.info(f"Pairs Trading Strategy initialized with {len(self.active_pairs)} active pairs")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Pairs Trading Strategy: {e}")
            return False

    async def _initialize_pair_statistics(self):
        """Calculate historical statistics for potential pairs."""
        logger.info("Calculating pair statistics...")

        for pair in self.potential_pairs:
            try:
                pair_key = f"{pair[0]}_{pair[1]}"
                stats = await self._calculate_pair_statistics(pair[0], pair[1])

                if stats and stats['is_valid_pair']:
                    self.active_pairs[pair_key] = pair
                    self.pair_statistics[pair_key] = stats
                    logger.info(f"Activated pair {pair_key}: correlation={stats['correlation']:.3f}, "
                              f"cointegration_p={stats['cointegration_p']:.3f}")

            except Exception as e:
                logger.warning(f"Failed to initialize pair {pair}: {e}")

    async def _calculate_pair_statistics(self, symbol1: str, symbol2: str) -> Optional[Dict]:
        """Calculate statistical properties of a pair."""
        try:
            # Get historical price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.max_lookback_days)

            data1 = await self.market_data.get_historical_prices(
                symbol1, start_date, end_date, interval='1d'
            )
            data2 = await self.market_data.get_historical_prices(
                symbol2, start_date, end_date, interval='1d'
            )

            if not data1 or not data2 or len(data1) < self.min_lookback_days:
                return None

            # Align data by date
            prices1 = pd.Series({d['timestamp']: d['close'] for d in data1})
            prices2 = pd.Series({d['timestamp']: d['close'] for d in data2})

            # Merge on common dates
            common_dates = prices1.index.intersection(prices2.index)
            if len(common_dates) < self.min_lookback_days:
                return None

            prices1 = prices1[common_dates]
            prices2 = prices2[common_dates]

            # Calculate returns
            returns1 = prices1.pct_change().dropna()
            returns2 = prices2.pct_change().dropna()

            # Calculate correlation
            correlation = returns1.corr(returns2)

            # Test for cointegration
            try:
                coint_t, coint_p, _ = coint(prices1, prices2)
                is_cointegrated = coint_p < 0.05  # 5% significance level
            except:
                is_cointegrated = False
                coint_p = 1.0

            # Calculate spread (normalized price difference)
            spread = (prices1 - prices2) / (prices1 + prices2) / 2

            # Test for stationarity (ADF test)
            try:
                adf_result = adfuller(spread.dropna())
                is_stationary = adf_result[1] < 0.05  # 5% significance level
            except:
                is_stationary = False

            # Calculate spread statistics
            spread_mean = spread.mean()
            spread_std = spread.std()

            # Determine if pair is valid for trading
            is_valid_pair = (
                abs(correlation) > self.correlation_threshold and
                is_cointegrated and
                is_stationary and
                spread_std > 0
            )

            return {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'correlation': correlation,
                'cointegration_p': coint_p,
                'is_cointegrated': is_cointegrated,
                'is_stationary': is_stationary,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'spread_current': spread.iloc[-1],
                'is_valid_pair': is_valid_pair,
                'data_points': len(spread),
                'last_update': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error calculating statistics for pair {symbol1}-{symbol2}: {e}")
            return None

    async def _validate_data_availability(self):
        """Validate that market data is available for active pairs."""
        for pair_key, pair in self.active_pairs.items():
            try:
                # Test real-time data availability
                data1 = await self.market_data.get_real_time_price(pair[0])
                data2 = await self.market_data.get_real_time_price(pair[1])

                if not data1 or not data2:
                    logger.warning(f"Real-time data not available for pair {pair_key}")
                    # Could remove from active pairs if needed

            except Exception as e:
                logger.warning(f"Data validation failed for pair {pair_key}: {e}")

    async def generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals based on pair spread deviations."""
        signals = []

        try:
            for pair_key, pair in self.active_pairs.items():
                pair_signals = await self._analyze_pair(pair_key, pair)
                signals.extend(pair_signals)

                # Check for exit signals on open positions
                exit_signals = await self._check_exit_signals(pair_key, pair)
                signals.extend(exit_signals)

        except Exception as e:
            logger.error(f"Error generating signals: {e}")

        return signals

    async def _analyze_pair(self, pair_key: str, pair: Tuple[str, str]) -> List[TradingSignal]:
        """Analyze a single pair for trading opportunities."""
        signals = []

        try:
            stats = self.pair_statistics.get(pair_key)
            if not stats:
                return signals

            # Get current prices
            price1 = await self.market_data.get_real_time_price(pair[0])
            price2 = await self.market_data.get_real_time_price(pair[1])

            if not price1 or not price2:
                return signals

            # Calculate current spread
            current_spread = (price1['price'] - price2['price']) / (price1['price'] + price2['price']) / 2

            # Calculate z-score
            z_score = (current_spread - stats['spread_mean']) / stats['spread_std']

            # Check for entry signals
            if abs(z_score) > self.entry_threshold_std:
                # Determine trade direction
                if z_score > 0:
                    # Spread is wide - long underperformer (symbol2), short outperformer (symbol1)
                    long_symbol = pair[1]  # Underperformer
                    short_symbol = pair[0]  # Outperformer
                    signal_type = SignalType.LONG_SHORT
                else:
                    # Spread is narrow - long underperformer (symbol1), short outperformer (symbol2)
                    long_symbol = pair[0]  # Underperformer
                    short_symbol = pair[1]  # Outperformer
                    signal_type = SignalType.LONG_SHORT

                # Calculate position sizes (equal dollar value)
                portfolio_value = self.config.capital_allocation
                position_size = portfolio_value * self.config.max_position_size_pct

                long_quantity = position_size / (2 * price1['price']) if long_symbol == pair[0] else position_size / (2 * price2['price'])
                short_quantity = position_size / (2 * price1['price']) if short_symbol == pair[0] else position_size / (2 * price2['price'])

                # Create signal
                signal = TradingSignal(
                    strategy_id=self.strategy_id,
                    signal_type=signal_type,
                    symbol=long_symbol,  # Primary symbol for the pair
                    pair_symbol=short_symbol,  # Counter symbol
                    quantity=long_quantity,
                    price=price1['price'] if long_symbol == pair[0] else price2['price'],
                    confidence=abs(z_score) / 4.0,  # Normalize confidence to 0-1
                    timestamp=datetime.now(),
                    metadata={
                        'pair_key': pair_key,
                        'z_score': z_score,
                        'spread_mean': stats['spread_mean'],
                        'spread_std': stats['spread_std'],
                        'correlation': stats['correlation'],
                        'short_symbol': short_symbol,
                        'short_quantity': short_quantity,
                        'short_price': price1['price'] if short_symbol == pair[0] else price2['price']
                    }
                )

                signals.append(signal)
                logger.info(f"Generated {signal_type.value} signal for pair {pair_key}: z-score={z_score:.2f}")

        except Exception as e:
            logger.error(f"Error analyzing pair {pair_key}: {e}")

        return signals

    async def _check_exit_signals(self, pair_key: str, pair: Tuple[str, str]) -> List[TradingSignal]:
        """Check for exit signals on open positions."""
        signals = []

        try:
            if pair_key not in self.open_positions:
                return signals

            position = self.open_positions[pair_key]
            stats = self.pair_statistics.get(pair_key)

            if not stats:
                return signals

            # Get current prices
            price1 = await self.market_data.get_real_time_price(pair[0])
            price2 = await self.market_data.get_real_time_price(pair[1])

            if not price1 or not price2:
                return signals

            # Calculate current spread
            current_spread = (price1['price'] - price2['price']) / (price1['price'] + price2['price']) / 2
            z_score = (current_spread - stats['spread_mean']) / stats['spread_std']

            # Check exit conditions
            should_exit = (
                abs(z_score) < self.exit_threshold_std or  # Spread converged
                (datetime.now() - position['entry_time']).days >= self.max_holding_days  # Max holding period
            )

            if should_exit:
                # Create exit signals (close both positions)
                long_symbol = position['long_symbol']
                short_symbol = position['short_symbol']

                # Long position exit (sell)
                long_exit_signal = TradingSignal(
                    strategy_id=self.strategy_id,
                    signal_type=SignalType.SELL,
                    symbol=long_symbol,
                    quantity=position['long_quantity'],
                    price=price1['price'] if long_symbol == pair[0] else price2['price'],
                    confidence=1.0,
                    timestamp=datetime.now(),
                    metadata={
                        'pair_key': pair_key,
                        'exit_reason': 'convergence' if abs(z_score) < self.exit_threshold_std else 'max_holding_period',
                        'z_score': z_score,
                        'pnl': self._calculate_pnl(position, price1['price'], price2['price'])
                    }
                )

                # Short position exit (buy back)
                short_exit_signal = TradingSignal(
                    strategy_id=self.strategy_id,
                    signal_type=SignalType.BUY,
                    symbol=short_symbol,
                    quantity=position['short_quantity'],
                    price=price1['price'] if short_symbol == pair[0] else price2['price'],
                    confidence=1.0,
                    timestamp=datetime.now(),
                    metadata={
                        'pair_key': pair_key,
                        'exit_reason': 'convergence' if abs(z_score) < self.exit_threshold_std else 'max_holding_period',
                        'z_score': z_score,
                        'pnl': self._calculate_pnl(position, price1['price'], price2['price'])
                    }
                )

                signals.extend([long_exit_signal, short_exit_signal])

                # Remove from open positions
                del self.open_positions[pair_key]

                logger.info(f"Generated exit signals for pair {pair_key}: {long_exit_signal.metadata['exit_reason']}")

        except Exception as e:
            logger.error(f"Error checking exit signals for pair {pair_key}: {e}")

        return signals

    def _calculate_pnl(self, position: Dict, price1: float, price2: float) -> float:
        """Calculate profit/loss for a position."""
        try:
            long_symbol = position['long_symbol']
            short_symbol = position['short_symbol']

            # Current values
            long_current_value = position['long_quantity'] * (price1 if long_symbol == position['symbols'][0] else price2)
            short_current_value = position['short_quantity'] * (price1 if short_symbol == position['symbols'][0] else price2)

            # Entry values
            long_entry_value = position['long_quantity'] * position['long_entry_price']
            short_entry_value = position['short_quantity'] * position['short_entry_price']

            # P&L calculation (long profit + short profit, since short profit is negative of price movement)
            pnl = (long_current_value - long_entry_value) + (short_entry_value - short_current_value)

            return pnl

        except Exception as e:
            logger.error(f"Error calculating P&L: {e}")
            return 0.0

    async def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute a trading signal."""
        try:
            # For pairs trading, we need to execute both legs
            if 'pair_key' in signal.metadata:
                pair_key = signal.metadata['pair_key']

                if signal.signal_type in [SignalType.LONG_SHORT]:
                    # Opening position - execute both legs
                    await self._execute_pair_entry(signal)
                elif signal.signal_type in [SignalType.SELL, SignalType.BUY]:
                    # Closing position - execute exit
                    await self._execute_pair_exit(signal)

            return True

        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False

    async def _execute_pair_entry(self, signal: TradingSignal):
        """Execute pair entry (both long and short legs)."""
        try:
            pair_key = signal.metadata['pair_key']
            short_symbol = signal.metadata['short_symbol']
            short_quantity = signal.metadata['short_quantity']
            short_price = signal.metadata['short_price']

            # Record position
            self.open_positions[pair_key] = {
                'entry_time': datetime.now(),
                'long_symbol': signal.symbol,
                'short_symbol': short_symbol,
                'long_quantity': signal.quantity,
                'short_quantity': short_quantity,
                'long_entry_price': signal.price,
                'short_entry_price': short_price,
                'symbols': pair_key.split('_')
            }

            logger.info(f"Opened pairs position for {pair_key}: Long {signal.symbol} {signal.quantity:.0f} shares, "
                       f"Short {short_symbol} {short_quantity:.0f} shares")

        except Exception as e:
            logger.error(f"Error executing pair entry: {e}")

    async def _execute_pair_exit(self, signal: TradingSignal):
        """Execute pair exit."""
        try:
            pair_key = signal.metadata['pair_key']
            pnl = signal.metadata.get('pnl', 0)

            logger.info(f"Closed pairs position for {pair_key}: P&L = ${pnl:.2f}")

        except Exception as e:
            logger.error(f"Error executing pair exit: {e}")

    async def update_statistics(self):
        """Update pair statistics periodically."""
        try:
            # Update statistics every hour
            for pair_key, pair in self.active_pairs.items():
                stats = await self._calculate_pair_statistics(pair[0], pair[1])
                if stats and stats['is_valid_pair']:
                    self.pair_statistics[pair_key] = stats

        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status and metrics."""
        return {
            'strategy_id': self.strategy_id,
            'active_pairs': len(self.active_pairs),
            'open_positions': len(self.open_positions),
            'total_pairs_analyzed': len(self.potential_pairs),
            'pair_statistics': {
                pair_key: {
                    'correlation': stats['correlation'],
                    'spread_std': stats['spread_std'],
                    'is_valid': stats['is_valid_pair']
                }
                for pair_key, stats in self.pair_statistics.items()
            }
        }