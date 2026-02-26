"""
Options Arbitrage Strategy
==========================

Strategy ID: s54_options_arbitrage
Description: Exploit inefficiencies in options markets through put-call parity and box spreads.

Key Components:
- Put-call parity arbitrage detection
- Box spread arbitrage opportunities
- Options pricing model validation
- Automated options trading execution
- Risk management for options positions
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm

from shared.strategy_framework import BaseArbitrageStrategy, TradingSignal, SignalType, StrategyConfig
from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class OptionsArbitrageStrategy(BaseArbitrageStrategy):
    """
    Options Arbitrage Strategy Implementation.

    This strategy identifies arbitrage opportunities in options markets by:
    1. Put-call parity violations
    2. Box spread mispricings
    3. Calendar spread inefficiencies
    4. Volatility arbitrage opportunities

    Key Parameters:
    - Minimum profit threshold: 0.05% (5 basis points)
    - Maximum time to expiration: 90 days
    - Risk-free rate: Dynamic from market data
    - Volatility calculation: Historical + implied
    """

    def __init__(self, config: StrategyConfig, communication: CommunicationFramework,
                 audit_logger: AuditLogger):
        super().__init__(config, communication, audit_logger)

        # Strategy-specific parameters
        self.min_profit_threshold = 0.0005  # 0.05% minimum profit
        self.max_time_to_expiry = 90  # Maximum days to expiration
        self.min_time_to_expiry = 7   # Minimum days to expiration
        self.max_strike_deviation = 0.1  # Maximum 10% strike deviation from spot
        self.fee_per_contract = 0.5   # Estimated fee per options contract

        # Risk-free rate (will be updated dynamically)
        self.risk_free_rate = 0.045  # 4.5% default

        # Underlyings to monitor
        self.underlyings = [
            'SPY', 'QQQ', 'IWM',  # ETFs
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Tech stocks
            'NVDA', 'META', 'NFLX'  # More tech
        ]

        # Options data cache
        self.options_cache = {}
        self.cache_timestamp = None
        self.cache_duration = 30.0  # Cache for 30 seconds

    async def initialize(self) -> bool:
        """Initialize the options arbitrage strategy."""
        try:
            logger.info("Initializing Options Arbitrage Strategy...")

            # Test options data connectivity
            await self._validate_options_data()

            # Update risk-free rate
            await self._update_risk_free_rate()

            logger.info(f"Options Arbitrage Strategy initialized for "
                       f"{len(self.underlyings)} underlyings")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Options Arbitrage Strategy: {e}")
            return False

    async def _validate_options_data(self):
        """Validate that options data is available."""
        logger.info("Validating options data connectivity...")

        for symbol in self.underlyings[:3]:  # Test first 3 symbols
            try:
                options_data = await self.market_data.get_options_chain(symbol)
                if options_data:
                    logger.debug(f"✓ {symbol}: {len(options_data)} options available")
                else:
                    logger.warning(f"✗ {symbol}: No options data available")
            except Exception as e:
                logger.warning(f"✗ {symbol}: Error - {e}")

    async def _update_risk_free_rate(self):
        """Update the risk-free rate from market data."""
        try:
            # Try to get Treasury yield data
            treasury_data = await self.market_data.get_real_time_price('^TNX')  # 10-year Treasury
            if treasury_data and 'price' in treasury_data:
                self.risk_free_rate = treasury_data['price'] / 100.0  # Convert from percentage
                logger.info(f"Updated risk-free rate to {self.risk_free_rate:.4f}")
            else:
                logger.warning("Could not update risk-free rate, using default")
        except Exception as e:
            logger.warning(f"Error updating risk-free rate: {e}")

    def _black_scholes_price(self, S: float, K: float, T: float, r: float,
                           sigma: float, option_type: str) -> float:
        """
        Calculate Black-Scholes option price.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        """
        try:
            if T <= 0 or sigma <= 0:
                return 0.0

            d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            elif option_type.lower() == 'put':
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            else:
                return 0.0

            return max(price, 0.0)

        except Exception as e:
            logger.error(f"Error calculating Black-Scholes price: {e}")
            return 0.0

    def _calculate_put_call_parity(self, call_price: float, put_price: float,
                                 S: float, K: float, T: float, r: float) -> Dict[str, float]:
        """
        Calculate put-call parity relationship.

        Put-call parity: C - P = S - K * e^(-rT)
        """
        try:
            # Theoretical difference
            theoretical_diff = S - K * np.exp(-r * T)

            # Actual difference
            actual_diff = call_price - put_price

            # Mispricing
            mispricing = actual_diff - theoretical_diff

            return {
                'theoretical_diff': theoretical_diff,
                'actual_diff': actual_diff,
                'mispricing': mispricing,
                'mispricing_pct': abs(mispricing) / max(theoretical_diff, 0.01)
            }

        except Exception as e:
            logger.error(f"Error calculating put-call parity: {e}")
            return {}

    async def _get_options_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get options chain data for a symbol."""
        try:
            # Check cache first
            if self.options_cache.get(symbol) and self.cache_timestamp:
                if (datetime.now() - self.cache_timestamp).total_seconds() < self.cache_duration:
                    return self.options_cache[symbol]

            # Fetch fresh data
            options_data = await self.market_data.get_options_chain(symbol)

            if options_data:
                # Cache the data
                self.options_cache[symbol] = options_data
                self.cache_timestamp = datetime.now()

            return options_data

        except Exception as e:
            logger.debug(f"Error getting options data for {symbol}: {e}")
            return None

    async def generate_signals(self) -> List[TradingSignal]:
        """Generate options arbitrage signals."""
        signals = []

        try:
            # Update risk-free rate periodically
            if np.random.random() < 0.1:  # 10% chance each cycle
                await self._update_risk_free_rate()

            # Check each underlying for arbitrage opportunities
            for symbol in self.underlyings:
                try:
                    symbol_signals = await self._check_symbol_arbitrage(symbol)
                    signals.extend(symbol_signals)

                except Exception as e:
                    logger.debug(f"Error checking {symbol}: {e}")

        except Exception as e:
            logger.error(f"Error generating signals: {e}")

        return signals

    async def _check_symbol_arbitrage(self, symbol: str) -> List[TradingSignal]:
        """Check a single symbol for options arbitrage opportunities."""
        signals = []

        try:
            # Get underlying price
            spot_data = await self.market_data.get_real_time_price(symbol)
            if not spot_data or 'price' not in spot_data:
                return signals

            spot_price = spot_data['price']

            # Get options data
            options_data = await self._get_options_data(symbol)
            if not options_data:
                return signals

            # Check put-call parity arbitrage
            parity_signals = await self._check_put_call_parity(
                symbol, spot_price, options_data
            )
            signals.extend(parity_signals)

            # Check box spread arbitrage
            box_signals = await self._check_box_spreads(
                symbol, spot_price, options_data
            )
            signals.extend(box_signals)

        except Exception as e:
            logger.debug(f"Error checking arbitrage for {symbol}: {e}")

        return signals

    async def _check_put_call_parity(self, symbol: str, spot_price: float,
                                   options_data: Dict[str, Any]) -> List[TradingSignal]:
        """Check for put-call parity arbitrage opportunities."""
        signals = []

        try:
            # Group options by strike and expiration
            strike_expiry_groups = {}

            for option in options_data.get('options', []):
                key = (option.get('strike'), option.get('expiration'))
                if key not in strike_expiry_groups:
                    strike_expiry_groups[key] = {'calls': [], 'puts': []}

                if option.get('type') == 'call':
                    strike_expiry_groups[key]['calls'].append(option)
                elif option.get('type') == 'put':
                    strike_expiry_groups[key]['puts'].append(option)

            # Check each strike/expiry combination
            for (strike, expiry), options in strike_expiry_groups.items():
                calls = options['calls']
                puts = options['puts']

                if not calls or not puts:
                    continue

                # Use the most liquid options (highest volume)
                call = max(calls, key=lambda x: x.get('volume', 0))
                put = max(puts, key=lambda x: x.get('volume', 0))

                call_price = call.get('price', 0)
                put_price = put.get('price', 0)

                if call_price <= 0 or put_price <= 0:
                    continue

                # Calculate time to expiration
                expiry_date = datetime.fromisoformat(expiry.replace('Z', '+00:00'))
                time_to_expiry = (expiry_date - datetime.now()).total_seconds() / (365.25 * 24 * 3600)

                if not (self.min_time_to_expiry <= time_to_expiry * 365 <= self.max_time_to_expiry):
                    continue

                # Calculate put-call parity
                parity_analysis = self._calculate_put_call_parity(
                    call_price, put_price, spot_price, strike,
                    time_to_expiry, self.risk_free_rate
                )

                if not parity_analysis:
                    continue

                mispricing = parity_analysis['mispricing']
                mispricing_pct = parity_analysis['mispricing_pct']

                # Check if mispricing exceeds threshold
                if abs(mispricing_pct) > self.min_profit_threshold:
                    # Calculate position size
                    capital = self.config.capital_allocation
                    max_position = capital * self.config.max_position_size_pct

                    # Conservative position sizing for options
                    position_size = min(max_position, capital * 0.005)  # Max 0.5% of capital

                    # Determine arbitrage direction
                    if mispricing > 0:
                        # Call overpriced relative to put - sell call, buy put
                        signal_type = SignalType.ARBITRAGE
                        action = "sell_call_buy_put"
                    else:
                        # Put overpriced relative to call - buy call, sell put
                        signal_type = SignalType.ARBITRAGE
                        action = "buy_call_sell_put"

                    signal = TradingSignal(
                        strategy_id=self.strategy_id,
                        signal_type=signal_type,
                        symbol=f"{symbol}_parity_{strike}_{expiry[:10]}",
                        quantity=position_size,
                        price=abs(mispricing),  # Mispricing amount
                        confidence=min(abs(mispricing_pct) / 0.01, 1.0),  # Scale by 1%
                        timestamp=datetime.now(),
                        metadata={
                            'arbitrage_type': 'put_call_parity',
                            'underlying': symbol,
                            'strike': strike,
                            'expiration': expiry,
                            'spot_price': spot_price,
                            'call_price': call_price,
                            'put_price': put_price,
                            'mispricing': mispricing,
                            'mispricing_pct': mispricing_pct,
                            'action': action,
                            'time_to_expiry_days': time_to_expiry * 365,
                            'risk_free_rate': self.risk_free_rate
                        }
                    )

                    signals.append(signal)

                    logger.info(f"Put-call parity arbitrage: {symbol} ${strike} "
                              f"{expiry[:10]} - Mispricing: {mispricing:.4f} "
                              f"({mispricing_pct:.4f}%)")

        except Exception as e:
            logger.error(f"Error checking put-call parity for {symbol}: {e}")

        return signals

    async def _check_box_spreads(self, symbol: str, spot_price: float,
                               options_data: Dict[str, Any]) -> List[TradingSignal]:
        """Check for box spread arbitrage opportunities."""
        signals = []

        try:
            # Box spread: Buy call K1, sell call K2, buy put K1, sell put K2
            # Should equal (K2 - K1) * e^(-rT)

            # Get all strikes for near-term expiration
            expirations = sorted(list(set([
                opt.get('expiration') for opt in options_data.get('options', [])
            ])))

            if not expirations:
                return signals

            near_expiry = expirations[0]
            near_options = [
                opt for opt in options_data.get('options', [])
                if opt.get('expiration') == near_expiry
            ]

            # Group by strike
            strikes = sorted(list(set([opt.get('strike') for opt in near_options])))

            # Check box spreads for adjacent strikes
            for i in range(len(strikes) - 1):
                k1, k2 = strikes[i], strikes[i + 1]

                # Find options for these strikes
                k1_options = [opt for opt in near_options if opt.get('strike') == k1]
                k2_options = [opt for opt in near_options if opt.get('strike') == k2]

                # Get call and put prices
                k1_call = next((opt for opt in k1_options if opt.get('type') == 'call'), None)
                k1_put = next((opt for opt in k1_options if opt.get('type') == 'put'), None)
                k2_call = next((opt for opt in k2_options if opt.get('type') == 'call'), None)
                k2_put = next((opt for opt in k2_options if opt.get('type') == 'put'), None)

                if not all([k1_call, k1_put, k2_call, k2_put]):
                    continue

                # Box spread prices
                box_buy_price = k1_call.get('price', 0) + k2_put.get('price', 0)
                box_sell_price = k2_call.get('price', 0) + k1_put.get('price', 0)

                # Calculate time to expiration
                expiry_date = datetime.fromisoformat(near_expiry.replace('Z', '+00:00'))
                time_to_expiry = (expiry_date - datetime.now()).total_seconds() / (365.25 * 24 * 3600)

                # Theoretical box spread value
                theoretical_value = (k2 - k1) * np.exp(-self.risk_free_rate * time_to_expiry)

                # Check for arbitrage
                box_cost = box_buy_price - box_sell_price
                mispricing = box_cost - theoretical_value

                if abs(mispricing) > self.min_profit_threshold * theoretical_value:
                    # Calculate position size
                    capital = self.config.capital_allocation
                    max_position = capital * self.config.max_position_size_pct
                    position_size = min(max_position, capital * 0.003)  # Max 0.3% of capital

                    signal = TradingSignal(
                        strategy_id=self.strategy_id,
                        signal_type=SignalType.ARBITRAGE,
                        symbol=f"{symbol}_box_{k1}_{k2}_{near_expiry[:10]}",
                        quantity=position_size,
                        price=abs(mispricing),
                        confidence=min(abs(mispricing) / (theoretical_value * 0.01), 1.0),
                        timestamp=datetime.now(),
                        metadata={
                            'arbitrage_type': 'box_spread',
                            'underlying': symbol,
                            'strike_low': k1,
                            'strike_high': k2,
                            'expiration': near_expiry,
                            'box_buy_price': box_buy_price,
                            'box_sell_price': box_sell_price,
                            'theoretical_value': theoretical_value,
                            'mispricing': mispricing,
                            'mispricing_pct': mispricing / theoretical_value,
                            'time_to_expiry_days': time_to_expiry * 365
                        }
                    )

                    signals.append(signal)

                    logger.info(f"Box spread arbitrage: {symbol} ${k1}-${k2} "
                              f"{near_expiry[:10]} - Mispricing: {mispricing:.4f}")

        except Exception as e:
            logger.error(f"Error checking box spreads for {symbol}: {e}")

        return signals

    async def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute an options arbitrage trade."""
        try:
            if signal.signal_type != SignalType.ARBITRAGE:
                return False

            arbitrage_type = signal.metadata.get('arbitrage_type')

            if arbitrage_type == 'put_call_parity':
                success = await self._execute_put_call_parity(signal)
            elif arbitrage_type == 'box_spread':
                success = await self._execute_box_spread(signal)
            else:
                logger.warning(f"Unknown arbitrage type: {arbitrage_type}")
                return False

            if success:
                logger.info(f"Executed {arbitrage_type} arbitrage for {signal.symbol}")

            return success

        except Exception as e:
            logger.error(f"Error executing options arbitrage: {e}")
            return False

    async def _execute_put_call_parity(self, signal: TradingSignal) -> bool:
        """Execute put-call parity arbitrage trade."""
        try:
            # In a real implementation, this would place actual options orders
            # For now, simulate the execution

            action = signal.metadata['action']
            position_size = signal.quantity

            # Log the synthetic trade
            await self.audit_logger.log_event(
                'options_parity_execution',
                {
                    'symbol': signal.symbol,
                    'action': action,
                    'position_size': position_size,
                    'mispricing': signal.metadata['mispricing'],
                    'expected_profit': signal.price
                }
            )

            return True

        except Exception as e:
            logger.error(f"Error executing put-call parity: {e}")
            return False

    async def _execute_box_spread(self, signal: TradingSignal) -> bool:
        """Execute box spread arbitrage trade."""
        try:
            # Simulate box spread execution
            position_size = signal.quantity

            await self.audit_logger.log_event(
                'options_box_execution',
                {
                    'symbol': signal.symbol,
                    'position_size': position_size,
                    'mispricing': signal.metadata['mispricing'],
                    'expected_profit': signal.price
                }
            )

            return True

        except Exception as e:
            logger.error(f"Error executing box spread: {e}")
            return False

    async def update_statistics(self):
        """Update strategy statistics periodically."""
        try:
            # Clear old cache
            if self.cache_timestamp and (datetime.now() - self.cache_timestamp).total_seconds() > 300:
                self.options_cache = {}
                self.cache_timestamp = None

        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status and metrics."""
        return {
            'strategy_id': self.strategy_id,
            'underlyings_monitored': len(self.underlyings),
            'cached_options': len(self.options_cache),
            'cache_age_seconds': (datetime.now() - self.cache_timestamp).total_seconds() if self.cache_timestamp else None,
            'risk_free_rate': self.risk_free_rate,
            'min_profit_threshold': self.min_profit_threshold,
            'max_time_to_expiry_days': self.max_time_to_expiry,
            'fee_per_contract': self.fee_per_contract
        }