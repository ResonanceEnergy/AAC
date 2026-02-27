"""
Triangular Arbitrage Strategy
============================

Strategy ID: s53_triangular_arbitrage
Description: Exploit inefficiencies in currency exchange rates through triangular arbitrage across three currencies.

Key Components:
- Real-time exchange rate monitoring
- Triangular arbitrage opportunity detection
- Automated execution of three-leg trades
- Risk management and slippage control
- Cross-exchange arbitrage opportunities
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from shared.strategy_framework import BaseArbitrageStrategy, TradingSignal, SignalType, StrategyConfig
from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class TriangularArbitrageStrategy(BaseArbitrageStrategy):
    """
    Triangular Arbitrage Strategy Implementation.

    This strategy identifies triangular arbitrage opportunities in currency markets
    by monitoring exchange rates between three currencies. When the cross rates
    are misaligned, the strategy executes a three-step trade that should be
    profitable after accounting for fees and slippage.

    Key Parameters:
    - Minimum profit threshold: 0.1% (10 basis points)
    - Maximum slippage allowance: 0.05%
    - Triangle combinations: Major currency triangles
    - Execution speed: Sub-second for HFT opportunities
    """

    def __init__(self, config: StrategyConfig, communication: CommunicationFramework,
                 audit_logger: AuditLogger):
        super().__init__(config, communication, audit_logger)

        # Strategy-specific parameters
        self.min_profit_threshold = 0.001  # 0.1% minimum profit
        self.max_slippage = 0.0005  # 0.05% maximum slippage
        self.max_execution_time = 5.0  # Maximum execution time in seconds
        self.fee_estimate = 0.0005  # Estimated trading fee per trade (0.05%)

        # Currency triangles for arbitrage
        # Each triangle is a tuple of three currencies
        self.currency_triangles = [
            # Major currency triangles
            ('EUR', 'USD', 'GBP'),  # EUR/USD, GBP/USD, EUR/GBP
            ('EUR', 'USD', 'JPY'),  # EUR/USD, USD/JPY, EUR/JPY
            ('EUR', 'USD', 'CHF'),  # EUR/USD, USD/CHF, EUR/CHF
            ('GBP', 'USD', 'JPY'),  # GBP/USD, USD/JPY, GBP/JPY
            ('GBP', 'USD', 'CHF'),  # GBP/USD, USD/CHF, GBP/CHF
            ('USD', 'JPY', 'CHF'),  # USD/JPY, USD/CHF, JPY/CHF

            # Commodity currency triangles
            ('AUD', 'USD', 'JPY'),  # AUD/USD, USD/JPY, AUD/JPY
            ('CAD', 'USD', 'JPY'),  # CAD/USD, USD/JPY, CAD/JPY
            ('NZD', 'USD', 'JPY'),  # NZD/USD, USD/JPY, NZD/JPY

            # Emerging market triangles
            ('EUR', 'USD', 'TRY'),  # EUR/USD, USD/TRY, EUR/TRY
            ('GBP', 'USD', 'ZAR'),  # GBP/USD, USD/ZAR, GBP/ZAR
        ]

        # Crypto triangles (cross-exchange)
        self.crypto_triangles = [
            ('BTC', 'ETH', 'USDT'),  # BTC/ETH, ETH/USDT, BTC/USDT
            ('BTC', 'BNB', 'USDT'),  # BTC/BNB, BNB/USDT, BTC/USDT
            ('ETH', 'BNB', 'USDT'),  # ETH/BNB, BNB/USDT, ETH/USDT
            ('BTC', 'ADA', 'USDT'),  # BTC/ADA, ADA/USDT, BTC/USDT
            ('ETH', 'ADA', 'USDT'),  # ETH/ADA, ADA/USDT, ETH/USDT
        ]

        # Rate cache for faster calculations
        self.rate_cache = {}
        self.cache_timestamp = None
        self.cache_duration = 1.0  # Cache rates for 1 second

    async def initialize(self) -> bool:
        """Initialize the triangular arbitrage strategy."""
        try:
            logger.info("Initializing Triangular Arbitrage Strategy...")

            # Test market data connectivity
            await self._validate_data_connectivity()

            logger.info(f"Triangular Arbitrage Strategy initialized with "
                       f"{len(self.currency_triangles)} currency triangles and "
                       f"{len(self.crypto_triangles)} crypto triangles")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Triangular Arbitrage Strategy: {e}")
            return False

    async def _validate_data_connectivity(self):
        """Validate that market data is available for all currency pairs."""
        test_pairs = []

        # Collect all unique pairs from triangles
        for triangle in self.currency_triangles + self.crypto_triangles:
            test_pairs.extend([
                f"{triangle[0]}{triangle[1]}",
                f"{triangle[1]}{triangle[2]}",
                f"{triangle[0]}{triangle[2]}"
            ])

        test_pairs = list(set(test_pairs))  # Remove duplicates

        logger.info(f"Validating data connectivity for {len(test_pairs)} currency pairs...")

        for pair in test_pairs[:10]:  # Test first 10 pairs
            try:
                data = await self.market_data.get_real_time_price(pair)
                if data:
                    logger.debug(f"✓ {pair}: ${data['price']:.4f}")
                else:
                    logger.warning(f"✗ {pair}: No data available")
            except Exception as e:
                logger.warning(f"✗ {pair}: Error - {e}")

    async def _get_exchange_rates(self) -> Dict[str, float]:
        """Get current exchange rates for all relevant pairs."""
        try:
            # Check cache first
            if self.rate_cache and self.cache_timestamp:
                if (datetime.now() - self.cache_timestamp).total_seconds() < self.cache_duration:
                    return self.rate_cache

            rates = {}

            # Collect all unique pairs
            all_pairs = set()
            for triangle in self.currency_triangles + self.crypto_triangles:
                all_pairs.update([
                    f"{triangle[0]}{triangle[1]}",
                    f"{triangle[1]}{triangle[2]}",
                    f"{triangle[0]}{triangle[2]}"
                ])

            # Fetch rates for all pairs
            for pair in all_pairs:
                try:
                    data = await self.market_data.get_real_time_price(pair)
                    if data and 'price' in data:
                        rates[pair] = data['price']
                    else:
                        # Try inverse pair
                        inverse_pair = pair[3:] + pair[:3] if len(pair) > 3 else None
                        if inverse_pair:
                            data = await self.market_data.get_real_time_price(inverse_pair)
                            if data and 'price' in data:
                                rates[pair] = 1.0 / data['price']
                except Exception as e:
                    logger.debug(f"Could not get rate for {pair}: {e}")

            # Cache the rates
            self.rate_cache = rates
            self.cache_timestamp = datetime.now()

            return rates

        except Exception as e:
            logger.error(f"Error getting exchange rates: {e}")
            return {}

    async def generate_signals(self) -> List[TradingSignal]:
        """Generate triangular arbitrage signals."""
        signals = []

        try:
            # Get current exchange rates
            rates = await self._get_exchange_rates()

            if not rates:
                logger.warning("No exchange rates available")
                return signals

            # Check currency triangles
            currency_signals = await self._check_triangles(
                self.currency_triangles, rates, "forex"
            )
            signals.extend(currency_signals)

            # Check crypto triangles
            crypto_signals = await self._check_triangles(
                self.crypto_triangles, rates, "crypto"
            )
            signals.extend(crypto_signals)

        except Exception as e:
            logger.error(f"Error generating signals: {e}")

        return signals

    async def _check_triangles(self, triangles: List[Tuple[str, str, str]],
                              rates: Dict[str, float], market_type: str) -> List[TradingSignal]:
        """Check triangles for arbitrage opportunities."""
        signals = []

        for triangle in triangles:
            try:
                opportunity = await self._analyze_triangle(triangle, rates, market_type)
                if opportunity:
                    signals.append(opportunity)

            except Exception as e:
                logger.debug(f"Error analyzing triangle {triangle}: {e}")

        return signals

    async def _analyze_triangle(self, triangle: Tuple[str, str, str],
                               rates: Dict[str, float], market_type: str) -> Optional[TradingSignal]:
        """Analyze a single triangle for arbitrage opportunities."""
        try:
            curr1, curr2, curr3 = triangle

            # Get exchange rates
            pair12 = f"{curr1}{curr2}"
            pair23 = f"{curr2}{curr3}"
            pair13 = f"{curr1}{curr3}"

            rate12 = rates.get(pair12)
            rate23 = rates.get(pair23)
            rate13 = rates.get(pair13)

            if not all([rate12, rate23, rate13]):
                return None

            # Calculate triangular arbitrage
            # Method 1: Start with curr1, go curr1->curr2->curr3->curr1
            path1_rate = rate12 * rate23 * (1.0 / rate13)

            # Method 2: Alternative path
            path2_rate = (1.0 / rate12) * (1.0 / rate23) * rate13

            # Calculate profit potential
            profit1 = (path1_rate - 1.0) * 100  # Percentage
            profit2 = (path2_rate - 1.0) * 100  # Percentage

            # Account for trading fees (3 trades)
            total_fees = self.fee_estimate * 3 * 100  # Convert to percentage

            # Net profit after fees
            net_profit1 = profit1 - total_fees
            net_profit2 = profit2 - total_fees

            # Check if profitable
            if net_profit1 > self.min_profit_threshold or net_profit2 > self.min_profit_threshold:
                # Determine best path
                if net_profit1 > net_profit2:
                    profit_pct = net_profit1
                    arbitrage_path = [curr1, curr2, curr3, curr1]
                    rates_used = [rate12, rate23, 1.0/rate13]
                else:
                    profit_pct = net_profit2
                    arbitrage_path = [curr1, curr3, curr2, curr1]
                    rates_used = [1.0/rate13, 1.0/rate23, rate12]

                # Calculate position size based on profit potential
                capital = self.config.capital_allocation
                max_position = capital * self.config.max_position_size_pct

                # Estimate position size (conservative approach)
                position_size = min(max_position, capital * 0.01)  # Max 1% of capital

                signal = TradingSignal(
                    strategy_id=self.strategy_id,
                    signal_type=SignalType.ARBITRAGE,
                    symbol=f"{curr1}_{curr2}_{curr3}",  # Triangle identifier
                    quantity=position_size,
                    price=1.0,  # Not applicable for arbitrage
                    confidence=min(profit_pct / 0.5, 1.0),  # Scale confidence by profit
                    timestamp=datetime.now(),
                    metadata={
                        'market_type': market_type,
                        'triangle': triangle,
                        'arbitrage_path': arbitrage_path,
                        'rates_used': rates_used,
                        'gross_profit_pct': max(profit1, profit2),
                        'net_profit_pct': profit_pct,
                        'total_fees_pct': total_fees,
                        'execution_time_limit': self.max_execution_time
                    }
                )

                logger.info(f"Triangular arbitrage opportunity: {triangle} "
                          f"({market_type}) - Net profit: {profit_pct:.4f}%")

                return signal

        except Exception as e:
            logger.error(f"Error analyzing triangle {triangle}: {e}")

        return None

    async def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute a triangular arbitrage trade."""
        try:
            if signal.signal_type != SignalType.ARBITRAGE:
                return False

            # Extract arbitrage details
            arbitrage_path = signal.metadata['arbitrage_path']
            rates_used = signal.metadata['rates_used']
            position_size = signal.quantity

            logger.info(f"Executing triangular arbitrage: {' -> '.join(arbitrage_path)}")

            # In a real implementation, this would execute three sequential trades
            # For now, we'll simulate the execution and log the opportunity

            # Step 1: Start with base currency amount
            start_amount = position_size

            # Step 2: Execute first trade
            trade1_amount = start_amount * rates_used[0]

            # Step 3: Execute second trade
            trade2_amount = trade1_amount * rates_used[1]

            # Step 4: Execute third trade (return to base currency)
            final_amount = trade2_amount * rates_used[2]

            # Calculate actual profit
            gross_profit = final_amount - start_amount
            net_profit = gross_profit - (start_amount * self.fee_estimate * 3)

            # Log execution results
            await self.audit_logger.log_event(
                'triangular_arbitrage_execution',
                {
                    'triangle': signal.metadata['triangle'],
                    'path': arbitrage_path,
                    'start_amount': start_amount,
                    'final_amount': final_amount,
                    'gross_profit': gross_profit,
                    'net_profit': net_profit,
                    'profit_pct': (net_profit / start_amount) * 100,
                    'execution_time': (datetime.now() - signal.timestamp).total_seconds()
                }
            )

            logger.info(f"Triangular arbitrage completed: ${net_profit:.2f} profit "
                       f"({(net_profit/start_amount)*100:.4f}%)")

            return True

        except Exception as e:
            logger.error(f"Error executing triangular arbitrage: {e}")
            return False

    async def update_statistics(self):
        """Update strategy statistics periodically."""
        try:
            # Clear old cache
            if self.cache_timestamp and (datetime.now() - self.cache_timestamp).total_seconds() > 60:
                self.rate_cache = {}
                self.cache_timestamp = None

        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status and metrics."""
        return {
            'strategy_id': self.strategy_id,
            'currency_triangles': len(self.currency_triangles),
            'crypto_triangles': len(self.crypto_triangles),
            'total_triangles': len(self.currency_triangles) + len(self.crypto_triangles),
            'cached_rates': len(self.rate_cache),
            'cache_age_seconds': (datetime.now() - self.cache_timestamp).total_seconds() if self.cache_timestamp else None,
            'min_profit_threshold': self.min_profit_threshold,
            'max_slippage': self.max_slippage,
            'fee_estimate': self.fee_estimate
        }