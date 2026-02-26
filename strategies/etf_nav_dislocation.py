"""
ETF-NAV Dislocation Harvesting Strategy
=======================================

Captures mispricings between ETF shares and their underlying NAV.
This is one of the most profitable arbitrage opportunities in modern markets.

Strategy Logic:
- Monitor ETF prices vs real-time NAV calculations
- When dislocation exceeds threshold, trade the spread
- Hedge with underlying basket to eliminate systematic risk
- Capture risk-free profit from market inefficiencies
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from shared.strategy_framework import BaseArbitrageStrategy, TradingSignal, SignalType, StrategyConfig
from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class ETFNAVDIslocationStrategy(BaseArbitrageStrategy):
    """
    ETF-NAV Dislocation Harvesting Strategy

    Captures arbitrage opportunities when ETF price deviates from NAV.
    This strategy generates consistent, low-risk profits.
    """

    def __init__(self, config: StrategyConfig, communication: CommunicationFramework, audit_logger: AuditLogger):
        super().__init__(config, communication, audit_logger)

        # Strategy parameters
        self.dislocation_threshold = 0.001  # 0.1% minimum dislocation
        self.max_dislocation = 0.01         # 1% maximum dislocation (filter outliers)
        self.min_volume = 100000           # Minimum ETF volume
        self.max_position_size = 50000     # Max position size per trade
        self.hold_period = 300             # 5 minutes max hold

        # ETF universe with sector diversification
        self.etf_universe = {
            # Large Cap
            'SPY': {'sector': 'large_cap', 'expense_ratio': 0.0945},
            'QQQ': {'sector': 'large_cap', 'expense_ratio': 0.0020},
            'IWM': {'sector': 'small_cap', 'expense_ratio': 0.0019},

            # International
            'EFA': {'sector': 'international', 'expense_ratio': 0.0032},
            'VWO': {'sector': 'emerging', 'expense_ratio': 0.0010},

            # Bonds
            'AGG': {'sector': 'bonds', 'expense_ratio': 0.0003},
            'BND': {'sector': 'bonds', 'expense_ratio': 0.0003},

            # Sector ETFs
            'XLF': {'sector': 'financial', 'expense_ratio': 0.0010},
            'XLE': {'sector': 'energy', 'expense_ratio': 0.0021},
            'XLK': {'sector': 'technology', 'expense_ratio': 0.0010},
        }

        # Position tracking
        self.active_positions = {}
        self.nav_cache = {}

        # Futures hedge mappings
        self.futures_mappings = {
            'SPY': 'ES',  # S&P 500 futures
            'QQQ': 'NQ',  # Nasdaq 100 futures
            'IWM': 'RTY', # Russell 2000 futures
            'EFA': 'E7',  # EAFE futures
            'VWO': 'M6C' # MSCI Emerging Markets futures
        }

    async def _initialize_strategy(self):
        """Initialize ETF-NAV strategy components"""
        logger.info("Initializing ETF-NAV Dislocation Strategy")

        # Subscribe to ETF and constituent data
        symbols_to_subscribe = list(self.etf_universe.keys())

        # Add major index constituents for NAV calculation
        major_holdings = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'PFE'
        ]
        symbols_to_subscribe.extend(major_holdings)

        # Set market data subscriptions for the integration system
        self.market_data_subscriptions = set(symbols_to_subscribe)

        # Initialize NAV calculation components
        self.nav_calculator = NAVCalculator(self.etf_universe)

        logger.info(f"ETF-NAV strategy initialized for {len(self.etf_universe)} ETFs")

    async def _generate_signals(self) -> List[TradingSignal]:
        """Generate ETF-NAV arbitrage signals"""
        signals = []

        for etf_symbol, etf_info in self.etf_universe.items():
            try:
                signal = await self._analyze_etf_dislocation(etf_symbol, etf_info)
                if signal:
                    signals.append(signal)

            except Exception as e:
                logger.error(f"Error analyzing {etf_symbol}: {e}")
                continue

        # Check for position exits
        exit_signals = await self._generate_exit_signals()
        signals.extend(exit_signals)

        return signals

    async def _analyze_etf_dislocation(self, etf_symbol: str, etf_info: Dict) -> Optional[TradingSignal]:
        """Analyze single ETF for NAV dislocations"""
        # Get current ETF price
        etf_data = self.market_data.get(etf_symbol, {})
        if not etf_data or 'price' not in etf_data:
            return None

        etf_price = etf_data['price']
        volume = etf_data.get('volume', 0)

        # Check volume threshold
        if volume < self.min_volume:
            return None

        # Calculate real-time NAV
        nav = await self.nav_calculator.calculate_nav(etf_symbol, self.market_data)
        if nav <= 0:
            return None

        # Calculate dislocation
        dislocation = (etf_price - nav) / nav

        # Cache NAV for exit calculations
        self.nav_cache[etf_symbol] = {
            'nav': nav,
            'timestamp': datetime.now(),
            'entry_price': etf_price
        }

        # Check dislocation thresholds
        if abs(dislocation) < self.dislocation_threshold or abs(dislocation) > self.max_dislocation:
            return None

        # Determine trade direction
        if dislocation > 0:
            # ETF overvalued vs NAV - short ETF, long basket
            signal_type = SignalType.SHORT
            confidence = min(dislocation / self.max_dislocation, 1.0)
        else:
            # ETF undervalued vs NAV - long ETF, short basket
            signal_type = SignalType.LONG
            confidence = min(abs(dislocation) / self.max_dislocation, 1.0)

        # Calculate position size based on confidence and available capital
        position_size = int(self.max_position_size * confidence)

        # Create signal
        signal = TradingSignal(
            strategy_id=self.config.strategy_id,
            signal_type=signal_type,
            symbol=etf_symbol,
            quantity=position_size,
            confidence=confidence,
            metadata={
                'dislocation': dislocation,
                'etf_price': etf_price,
                'nav': nav,
                'volume': volume,
                'sector': etf_info['sector'],
                'expense_ratio': etf_info['expense_ratio'],
                'strategy_type': 'etf_nav_arbitrage',
                'expected_hold_period': self.hold_period,
                'hedge_required': True,
                'hedge_instruments': await self._get_hedge_instruments(etf_symbol)
            }
        )

        # Track position entry
        self.active_positions[etf_symbol] = {
            'entry_time': datetime.now(),
            'entry_price': etf_price,
            'nav_at_entry': nav,
            'position_size': position_size,
            'signal_type': signal_type
        }

        return signal

    async def _generate_exit_signals(self) -> List[TradingSignal]:
        """Generate exit signals for active positions"""
        exit_signals = []
        current_time = datetime.now()

        for etf_symbol, position in list(self.active_positions.items()):
            hold_duration = (current_time - position['entry_time']).seconds

            # Exit if hold period exceeded
            if hold_duration >= self.hold_period:
                exit_signal = TradingSignal(
                    strategy_id=self.config.strategy_id,
                    signal_type=SignalType.CLOSE if position['signal_type'] == SignalType.LONG else SignalType.COVER,
                    symbol=etf_symbol,
                    quantity=position['position_size'],
                    confidence=1.0,
                    metadata={
                        'exit_reason': 'time_limit',
                        'hold_duration': hold_duration,
                        'strategy_type': 'etf_nav_arbitrage'
                    }
                )
                exit_signals.append(exit_signal)
                del self.active_positions[etf_symbol]
                continue

            # Exit if dislocation has converged
            current_nav = self.nav_cache.get(etf_symbol, {}).get('nav', 0)
            current_price = self.market_data.get(etf_symbol, {}).get('price', 0)

            if current_nav > 0 and current_price > 0:
                current_dislocation = (current_price - current_nav) / current_nav

                # Exit if dislocation has reduced by 50% or more
                entry_dislocation = (position['entry_price'] - position['nav_at_entry']) / position['nav_at_entry']

                if abs(current_dislocation) <= abs(entry_dislocation) * 0.5:
                    exit_signal = TradingSignal(
                        strategy_id=self.config.strategy_id,
                        signal_type=SignalType.CLOSE if position['signal_type'] == SignalType.LONG else SignalType.COVER,
                        symbol=etf_symbol,
                        quantity=position['position_size'],
                        confidence=1.0,
                        metadata={
                            'exit_reason': 'convergence',
                            'entry_dislocation': entry_dislocation,
                            'current_dislocation': current_dislocation,
                            'strategy_type': 'etf_nav_arbitrage'
                        }
                    )
                    exit_signals.append(exit_signal)
                    del self.active_positions[etf_symbol]

        return exit_signals

    async def _get_hedge_instruments(self, etf_symbol: str) -> List[str]:
        """Get instruments needed for hedging the ETF position"""
        # Simplified hedge mapping - in production would use actual holdings
        hedge_mappings = {
            'SPY': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'],
            'QQQ': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA'],
            'IWM': ['JPM', 'JNJ', 'V', 'PG', 'UNH'],
            'EFA': ['NESN.SW', 'NOVN.SW', 'ROG.SW'],  # European stocks
            'VWO': ['000858.SZ', '600036.SS', '000002.SZ'],  # Chinese stocks
            'AGG': ['US10Y', 'US5Y', 'US2Y'],  # Treasury futures
            'BND': ['US10Y', 'US5Y', 'US2Y'],
            'XLF': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
            'XLE': ['XOM', 'CVX', 'COP'],
            'XLK': ['AAPL', 'MSFT', 'NVDA', 'GOOGL']
        }

        return hedge_mappings.get(etf_symbol, [])

    def _update_market_data(self, data: Dict[str, Any]):
        """Update internal market data cache."""
        data_type = data.get('type', 'unknown')

        if data_type == 'etf_price':
            symbol = data.get('symbol')
            if symbol:
                self.market_data[symbol] = data
        elif data_type == 'nav_calculation':
            symbol = data.get('symbol')
            if symbol:
                # Ensure timestamp is a datetime object
                if 'timestamp' in data and isinstance(data['timestamp'], str):
                    try:
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    except:
                        data['timestamp'] = datetime.now()
                self.nav_cache[symbol] = data
        elif data_type == 'futures_price':
            symbol = data.get('symbol')
            if symbol:
                self.futures_data[symbol] = data

    async def _initialize_strategy(self):
        """Initialize ETF-NAV strategy specific components."""
        logger.info("Initializing ETF-NAV Dislocation strategy")

        # Initialize NAV calculation requests for each ETF
        for etf in self.etf_universe:
            await self._request_nav_calculation(etf)

        # Set up futures hedge mappings
        self.futures_mappings = {
            'SPY': 'ES',  # S&P 500 futures
            'QQQ': 'NQ',  # Nasdaq 100 futures
            'IWM': 'RTY', # Russell 2000 futures
            'EFA': 'E7',  # EAFE futures
            'VWO': 'M6C' # MSCI Emerging Markets futures
        }

        logger.info(f"ETF-NAV strategy initialized for {len(self.etf_universe)} ETFs")

    def _should_generate_signal(self) -> bool:
        """Check if we have sufficient data to generate signals."""
        # Check if we have both price and NAV data for any ETF
        for etf_symbol in self.etf_universe:
            has_price = etf_symbol in self.market_data
            has_nav = etf_symbol in self.nav_cache
            logger.info(f"ETF {etf_symbol}: price={has_price}, nav={has_nav}")
            if has_price and has_nav:
                return True
        return False

    async def _generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals based on NAV dislocations."""
        signals = []

        for etf_symbol in self.etf_universe:
            # Check for dislocation opportunities
            opportunity = await self._analyze_dislocation(etf_symbol)
            if opportunity:
                # Generate arbitrage signals
                arb_signals = await self._generate_arbitrage_signals(etf_symbol, opportunity)
                signals.extend(arb_signals)

        return signals

    def _should_generate_signal(self) -> bool:
        """Check if market conditions are suitable for signal generation."""
        # Only trade during regular market hours
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False

        # Check if we have recent NAV data (within 5 minutes)
        current_time = datetime.now()
        for nav_data in self.nav_cache.values():
            timestamp = nav_data.get('timestamp')
            if timestamp and (current_time - timestamp) > timedelta(minutes=5):
                return False

        return True

    async def _analyze_dislocation(self, etf_symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze NAV vs price dislocation for an ETF."""
        try:
            # Get current ETF price
            etf_price = self._get_etf_price(etf_symbol)
            if not etf_price:
                return None

            # Get latest NAV calculation
            nav_data = self.nav_cache.get(etf_symbol)
            if not nav_data:
                return None

            nav_price = nav_data.get('nav_price')
            if not nav_price:
                return None

            # Calculate dislocation in basis points
            dislocation_bps = ((etf_price - nav_price) / nav_price) * 10000

            # Check if dislocation exceeds threshold
            if abs(dislocation_bps) < self.dislocation_threshold_bps:
                return None

            # Check liquidity requirements
            liquidity = nav_data.get('liquidity', 0)
            if liquidity < self.min_liquidity_threshold:
                return None

            # Determine arbitrage direction
            if dislocation_bps > 0:
                # ETF trading at premium to NAV - short ETF, long basket
                direction = 'premium'
                action = 'redeem'
            else:
                # ETF trading at discount to NAV - long ETF, short basket
                direction = 'discount'
                action = 'create'

            return {
                'symbol': etf_symbol,
                'etf_price': etf_price,
                'nav_price': nav_price,
                'dislocation_bps': dislocation_bps,
                'direction': direction,
                'action': action,
                'liquidity': liquidity,
                'basket_holdings': nav_data.get('basket_holdings', [])
            }

        except Exception as e:
            logger.error(f"Error analyzing dislocation for {etf_symbol}: {e}")
            return None

    async def _generate_arbitrage_signals(self, etf_symbol: str,
                                        opportunity: Dict[str, Any]) -> List[TradingSignal]:
        """Generate arbitrage signals for a dislocation opportunity."""
        signals = []

        try:
            direction = opportunity['direction']
            dislocation_bps = opportunity['dislocation_bps']

            # Calculate position size based on risk limits
            position_size = self._calculate_position_size(etf_symbol, opportunity)
            logger.info(f"Calculated position size for {etf_symbol}: {position_size}")

            if position_size <= 0:
                logger.info(f"Position size <= 0 for {etf_symbol}, skipping signal generation")
                return signals

            logger.info(f"Generating arbitrage signals for {etf_symbol}: direction={direction}, position_size={position_size}")

            # Generate ETF position signal
            if direction == 'premium':
                # Short ETF (expect price to fall to NAV)
                etf_signal = TradingSignal(
                    strategy_id=self.config.strategy_id,
                    signal_type=SignalType.SHORT,
                    symbol=etf_symbol,
                    quantity=-position_size,
                    confidence=min(abs(dislocation_bps) / 100, 1.0),  # Confidence based on dislocation size
                    metadata={
                        'opportunity_type': 'nav_dislocation',
                        'dislocation_bps': dislocation_bps,
                        'expected_pnl_bps': abs(dislocation_bps) * 0.8  # Expect 80% of dislocation to capture
                    }
                )
                signals.append(etf_signal)
            else:
                # Long ETF (expect price to rise to NAV)
                etf_signal = TradingSignal(
                    strategy_id=self.config.strategy_id,
                    signal_type=SignalType.LONG,
                    symbol=etf_symbol,
                    quantity=position_size,
                    confidence=min(abs(dislocation_bps) / 100, 1.0),
                    metadata={
                        'opportunity_type': 'nav_dislocation',
                        'dislocation_bps': dislocation_bps,
                        'expected_pnl_bps': abs(dislocation_bps) * 0.8
                    }
                )
                signals.append(etf_signal)

            # Generate hedging signals with futures
            hedge_signals = await self._generate_hedge_signals(etf_symbol, position_size, direction)
            signals.extend(hedge_signals)

            # Log the opportunity
            await self.audit_logger.log_event(
                'nav_dislocation_opportunity',
                f'Detected NAV dislocation for {etf_symbol}: {dislocation_bps:.1f} bps',
                {
                    'etf_symbol': etf_symbol,
                    'dislocation_bps': dislocation_bps,
                    'direction': direction,
                    'position_size': position_size
                }
            )

        except Exception as e:
            logger.error(f"Error generating arbitrage signals for {etf_symbol}: {e}")

        return signals

    async def _generate_hedge_signals(self, etf_symbol: str, position_size: float,
                                    direction: str) -> List[TradingSignal]:
        """Generate futures hedging signals."""
        signals = []

        try:
            futures_symbol = self.futures_mappings.get(etf_symbol)
            if not futures_symbol:
                return signals

            # Calculate hedge quantity (opposite direction, partial hedge)
            hedge_quantity = position_size * self.hedge_ratio

            if direction == 'premium':
                # We shorted ETF, so long futures to hedge
                hedge_signal = TradingSignal(
                    strategy_id=self.config.strategy_id,
                    signal_type=SignalType.HEDGE,
                    symbol=futures_symbol,
                    quantity=hedge_quantity,
                    confidence=0.9,  # High confidence for hedging
                    metadata={
                        'hedge_type': 'futures_hedge',
                        'underlying': etf_symbol,
                        'hedge_ratio': self.hedge_ratio
                    }
                )
            else:
                # We longed ETF, so short futures to hedge
                hedge_signal = TradingSignal(
                    strategy_id=self.config.strategy_id,
                    signal_type=SignalType.HEDGE,
                    symbol=futures_symbol,
                    quantity=-hedge_quantity,
                    confidence=0.9,
                    metadata={
                        'hedge_type': 'futures_hedge',
                        'underlying': etf_symbol,
                        'hedge_ratio': self.hedge_ratio
                    }
                )

            signals.append(hedge_signal)

        except Exception as e:
            logger.error(f"Error generating hedge signals for {etf_symbol}: {e}")

        return signals

    def _calculate_position_size(self, etf_symbol: str, opportunity: Dict[str, Any]) -> float:
        """Calculate appropriate position size based on risk limits."""
        try:
            # Get ETF AUM (Assets Under Management) for sizing
            nav_data = self.nav_cache.get(etf_symbol, {})
            aum = nav_data.get('aum', 1000000000)  # Default $1B if unknown

            # Maximum position as percentage of AUM
            max_position = aum * (self.max_position_size_pct / 100)

            # Adjust based on dislocation size and liquidity
            dislocation_bps = abs(opportunity['dislocation_bps'])
            liquidity = opportunity['liquidity']

            # Scale position size with dislocation (larger dislocations = larger positions)
            dislocation_factor = min(dislocation_bps / 100, 2.0)  # Cap at 2x

            # Scale with liquidity (higher liquidity = larger positions)
            liquidity_factor = min(liquidity / 10000000, 2.0)  # Cap at 2x for $10M+ liquidity

            position_size = max_position * dislocation_factor * liquidity_factor

            # Apply strategy-level risk limits
            position_size = min(position_size, max_position)

            # Convert to number of ETF shares (assuming $100 ETF price for sizing)
            shares = position_size / 100

            return shares

        except Exception as e:
            logger.error(f"Error calculating position size for {etf_symbol}: {e}")
            return 0

    def _get_etf_price(self, symbol: str) -> Optional[float]:
        """Get current ETF price from market data."""
        market_data = self.market_data.get(symbol)
        if market_data:
            return market_data.get('price')
        return None

    async def _request_nav_calculation(self, etf_symbol: str):
        """Request NAV calculation from BigBrainIntelligence."""
        try:
            # Send request to BigBrain for NAV calculation
            request = {
                'etf_symbol': etf_symbol,
                'request_type': 'nav_calculation',
                'timestamp': datetime.now()
            }

            await self.communication.send_message(
                sender=self.config.strategy_id,
                recipient='BigBrainIntelligence',
                message_type='nav_calculation_request',
                payload=request
            )

        except Exception as e:
            logger.error(f"Error requesting NAV calculation for {etf_symbol}: {e}")

    def _update_market_data(self, data: Dict[str, Any]):
        """Update internal market data cache."""
        data_type = data.get('type', 'unknown')

        if data_type == 'etf_price':
            symbol = data.get('symbol')
            if symbol:
                self.market_data[symbol] = data
        elif data_type == 'nav_calculation':
            symbol = data.get('symbol')
            if symbol:
                # Ensure timestamp is a datetime object
                if 'timestamp' in data and isinstance(data['timestamp'], str):
                    try:
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    except:
                        data['timestamp'] = datetime.now()
                self.nav_cache[symbol] = data
        elif data_type == 'futures_price':
            symbol = data.get('symbol')
            if symbol:
                self.futures_data[symbol] = data


class NAVCalculator:
    """Real-time NAV calculator for ETFs"""

    def __init__(self, etf_universe: Dict[str, Dict]):
        self.etf_universe = etf_universe

        # Simplified holdings data - in production would be from ETF provider APIs
        self.holdings_data = {
            
            'SPY': {
                'AAPL': 0.12, 'MSFT': 0.11, 'AMZN': 0.06, 'GOOGL': 0.04, 'META': 0.03,
                'TSLA': 0.02, 'NVDA': 0.04, 'JPM': 0.02, 'JNJ': 0.02, 'V': 0.02
            },
            'QQQ': {
                'AAPL': 0.12, 'MSFT': 0.11, 'AMZN': 0.11, 'GOOGL': 0.08, 'META': 0.05,
                'TSLA': 0.05, 'NVDA': 0.08, 'ADBE': 0.03, 'CRM': 0.03, 'NFLX': 0.02
            },
            'IWM': {
                'JPM': 0.03, 'JNJ': 0.02, 'V': 0.02, 'PG': 0.02, 'UNH': 0.02,
                'HD': 0.02, 'MA': 0.02, 'PFE': 0.02, 'KO': 0.02, 'DIS': 0.02
            }
        }

    async def calculate_nav(self, etf_symbol: str, market_data: Dict[str, Dict]) -> float:
        """Calculate real-time NAV for an ETF"""
        holdings = self.holdings_data.get(etf_symbol, {})

        if not holdings:
            return 0.0

        total_value = 0.0
        total_weight = 0.0

        for symbol, weight in holdings.items():
            price_data = market_data.get(symbol, {})
            price = price_data.get('price', 0)

            if price > 0:
                total_value += price * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        # NAV is the value of holdings divided by total weight
        nav = total_value / total_weight

        # Apply expense ratio adjustment (simplified)
        expense_ratio = self.etf_universe.get(etf_symbol, {}).get('expense_ratio', 0)
        nav = nav * (1 - expense_ratio / 252)  # Daily adjustment

        return nav

class NAVCalculator:
    """Real-time NAV calculator for ETFs"""

    def __init__(self, etf_universe: Dict[str, Dict]):
        self.etf_universe = etf_universe

        # Simplified holdings data - in production would be from ETF provider APIs
        self.holdings_data = {
            'SPY': {
                'AAPL': 0.12, 'MSFT': 0.11, 'AMZN': 0.06, 'GOOGL': 0.04, 'META': 0.03,
                'TSLA': 0.02, 'NVDA': 0.04, 'JPM': 0.02, 'JNJ': 0.02, 'V': 0.02
            },
            'QQQ': {
                'AAPL': 0.12, 'MSFT': 0.11, 'AMZN': 0.11, 'GOOGL': 0.08, 'META': 0.05,
                'TSLA': 0.05, 'NVDA': 0.08, 'ADBE': 0.03, 'CRM': 0.03, 'NFLX': 0.02
            },
            'IWM': {
                'JPM': 0.03, 'JNJ': 0.02, 'V': 0.02, 'PG': 0.02, 'UNH': 0.02,
                'HD': 0.02, 'MA': 0.02, 'PFE': 0.02, 'KO': 0.02, 'DIS': 0.02
            }
        }

    async def calculate_nav(self, etf_symbol: str, market_data: Dict[str, Dict]) -> float:
        """Calculate real-time NAV for an ETF"""
        holdings = self.holdings_data.get(etf_symbol, {})

        if not holdings:
            return 0.0

        total_value = 0.0
        total_weight = 0.0

        for symbol, weight in holdings.items():
            price_data = market_data.get(symbol, {})
            price = price_data.get('price', 0)

            if price > 0:
                total_value += price * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        # NAV is the value of holdings divided by total weight
        nav = total_value / total_weight

        # Apply expense ratio adjustment (simplified)
        expense_ratio = self.etf_universe.get(etf_symbol, {}).get('expense_ratio', 0)
        nav = nav * (1 - expense_ratio / 252)  # Daily adjustment

        return nav
