"""
Overnight Jump Reversion Strategy
=================================

Strategy ID: s11_overnight_jump_reversion
Description: Fade large overnight jumps with short-horizon mean reversion controls.

Key Components:
- Overnight return calculation and threshold detection
- Mean reversion positioning against large jumps
- Short-horizon holding period (intraday)
- Risk management and position sizing
- Jump size filtering and confidence scoring
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


class OvernightJumpReversionStrategy(BaseArbitrageStrategy):
    """
    Overnight Jump Reversion Strategy Implementation.

    This strategy identifies large overnight price jumps and fades them with
    short-horizon mean reversion trades. The strategy goes short (long) assets
    that jumped up (down) significantly overnight, expecting reversion during
    the trading day.

    Key Parameters:
    - Jump threshold: > 2% absolute overnight return
    - Holding period: Same trading day (close position at market close)
    - Universe: Large-cap stocks and ETFs with high liquidity
    """

    def __init__(self, config: StrategyConfig, communication: CommunicationFramework,
                 audit_logger: AuditLogger):
        super().__init__(config, communication, audit_logger)

        # Strategy-specific parameters
        self.jump_threshold_pct = 2.0  # 2% minimum jump for consideration
        self.max_jump_pct = 10.0  # Maximum jump to avoid extreme events
        self.position_size_pct = 5.0  # 5% allocation per signal
        self.max_position_size_pct = 8.0  # Maximum position size limit
        self.holding_period_hours = 6  # Hold for ~6 hours (until market close)

        # Jump reversion parameters
        self.universe = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        self.min_volume_threshold = 1000000  # Minimum volume for liquidity
        self.max_gap_fill_pct = 50.0  # Maximum expected reversion (50% of gap)

        # Tracking variables
        self.active_positions = {}  # symbol -> position info
        self.overnight_returns = {}  # symbol -> overnight return data
        self.last_close_prices = {}  # symbol -> previous close price
        self.entry_times = {}  # symbol -> entry timestamp

    async def _initialize_strategy(self):
        """Initialize strategy-specific components."""
        await self._setup_market_data_subscriptions()

    async def _generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals based on current market conditions."""
        signals = []

        try:
            # Check for overnight jumps at market open
            current_time = datetime.now()

            if await self._is_market_open_time(current_time):
                # Check all symbols for overnight jumps
                for symbol in self.universe:
                    if symbol in self.last_close_prices:
                        overnight_return = await self._calculate_overnight_return(symbol, self.market_data.get(symbol, {}).get('price', 0), current_time)

                        if overnight_return and await self._is_significant_jump(overnight_return):
                            signals.extend(await self._generate_jump_reversion_signals(symbol, self.market_data.get(symbol, {}).get('price', 0), overnight_return))

            # Check for exit conditions at market close
            elif await self._is_market_close_time(current_time):
                signals.extend(await self._generate_exit_signals("market_close"))

        except Exception as e:
            logger.error(f"Error generating signals in jump reversion strategy: {e}")

        return signals

    def _should_generate_signal(self) -> bool:
        """Determine if conditions are right to generate signals."""
        # Generate signals at market open (for entries) or market close (for exits)
        current_time = datetime.now()
        current_hour = current_time.hour

        # Market open is ~9:30 AM ET (13:30 UTC), market close is ~4:00 PM ET (20:00 UTC)
        market_open_hour = 13
        market_close_hour = 20

        is_market_open = abs(current_hour - market_open_hour) <= 1
        is_market_close = abs(current_hour - market_close_hour) <= 1

        return is_market_open or is_market_close

    async def _setup_market_data_subscriptions(self):
        """Set up market data subscriptions for equity universe."""
        data_types = ['equity_price', 'equity_volume', 'market_open_close']

        await self.communication.subscribe_to_messages(
            agent_id=self.config.strategy_id,
            message_types=data_types
        )

        logger.info(f"Subscribed to market data types: {data_types}")

    async def process_market_data(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """Process market data and generate jump reversion signals."""
        signals = []

        try:
            data_type = data.get('type')
            symbol = data.get('symbol')

            if symbol not in self.universe:
                return signals

            if data_type == 'equity_price':
                signals.extend(await self._process_price_data(data))
            elif data_type == 'market_open_close':
                signals.extend(await self._check_exit_conditions(data))

        except Exception as e:
            logger.error(f"Error processing market data in jump reversion strategy: {e}")

        return signals

    async def _process_price_data(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """Process price data to detect overnight jumps and generate signals."""
        signals = []
        symbol = data.get('symbol')
        price = data.get('price')
        timestamp = data.get('timestamp')

        if not all([symbol, price, timestamp]):
            return signals

        try:
            # Update price tracking
            current_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            # Check if this is market open (potential jump detection)
            if await self._is_market_open_time(current_time):
                overnight_return = await self._calculate_overnight_return(symbol, price, current_time)

                if overnight_return and await self._is_significant_jump(overnight_return):
                    signals.extend(await self._generate_jump_reversion_signals(symbol, price, overnight_return))

            # Update last close price at market close
            elif await self._is_market_close_time(current_time):
                self.last_close_prices[symbol] = price

        except Exception as e:
            logger.error(f"Error processing price data for {symbol}: {e}")

        return signals

    async def _calculate_overnight_return(self, symbol: str, open_price: float, open_time: datetime) -> Optional[float]:
        """Calculate overnight return from previous close to current open."""
        if symbol not in self.last_close_prices:
            return None

        prev_close = self.last_close_prices[symbol]
        overnight_return = ((open_price - prev_close) / prev_close) * 100

        self.overnight_returns[symbol] = {
            'return_pct': overnight_return,
            'open_price': open_price,
            'prev_close': prev_close,
            'timestamp': open_time
        }

        return overnight_return

    async def _is_significant_jump(self, overnight_return: float) -> bool:
        """Determine if overnight return qualifies as a significant jump."""
        abs_return = abs(overnight_return)

        # Check if jump is within our threshold range
        if self.jump_threshold_pct <= abs_return <= self.max_jump_pct:
            return True

        return False

    async def _generate_jump_reversion_signals(self, symbol: str, current_price: float,
                                             overnight_return: float) -> List[TradingSignal]:
        """Generate mean reversion signals against overnight jumps."""
        signals = []

        try:
            # Determine direction: fade the jump
            if overnight_return > 0:
                # Positive jump -> expect reversion down -> SHORT
                signal_type = SignalType.SHORT
                direction = "short"
                confidence = min(0.75, abs(overnight_return) / 5.0)  # Higher jumps = higher confidence
            else:
                # Negative jump -> expect reversion up -> LONG
                signal_type = SignalType.LONG
                direction = "long"
                confidence = min(0.75, abs(overnight_return) / 5.0)

            # Calculate position size
            quantity = await self._calculate_position_quantity(symbol, current_price)

            if quantity > 0:
                signal = TradingSignal(
                    strategy_id=self.config.strategy_id,
                    signal_type=signal_type,
                    symbol=symbol,
                    quantity=quantity if direction == "long" else -quantity,
                    confidence=confidence,
                    metadata={
                        'strategy_type': 'overnight_jump_reversion',
                        'overnight_return_pct': overnight_return,
                        'jump_direction': 'up' if overnight_return > 0 else 'down',
                        'reversion_direction': direction,
                        'expected_holding_hours': self.holding_period_hours,
                        'entry_price': current_price,
                        'stop_loss_pct': abs(overnight_return) * 0.5  # 50% of jump as stop
                    }
                )
                signals.append(signal)

                # Track position entry
                self.entry_times[symbol] = datetime.now()

                await self.audit_logger.log_event(
                    'jump_reversion_entry',
                    f'Entering {direction} position for {symbol} after {overnight_return:.2f}% overnight {"jump" if overnight_return > 0 else "drop"}',
                    {
                        'symbol': symbol,
                        'overnight_return_pct': overnight_return,
                        'direction': direction,
                        'quantity': quantity,
                        'confidence': confidence,
                        'entry_price': current_price
                    }
                )

        except Exception as e:
            logger.error(f"Error generating jump reversion signal for {symbol}: {e}")

        return signals

    async def _check_exit_conditions(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """Check for position exit conditions (market close or time-based)."""
        signals = []
        event_type = data.get('event')

        if event_type == 'market_close':
            # Exit all open positions at market close
            signals.extend(await self._generate_exit_signals("market_close"))

        return signals

    async def _generate_exit_signals(self, exit_reason: str) -> List[TradingSignal]:
        """Generate signals to exit all open positions."""
        signals = []

        try:
            for symbol in list(self.active_positions.keys()):
                if symbol in self.active_positions:
                    position = self.active_positions[symbol]
                    quantity = -position['quantity']  # Close position

                    signal = TradingSignal(
                        strategy_id=self.config.strategy_id,
                        signal_type=SignalType.FLAT,
                        symbol=symbol,
                        quantity=quantity,
                        confidence=0.90,  # High confidence for planned exits
                        metadata={
                            'strategy_type': 'overnight_jump_reversion',
                            'exit_reason': exit_reason,
                            'holding_hours': (datetime.now() - self.entry_times.get(symbol, datetime.now())).total_seconds() / 3600
                        }
                    )
                    signals.append(signal)

                    await self.audit_logger.log_event(
                        'jump_reversion_exit',
                        f'Exiting position for {symbol}: {quantity} shares ({exit_reason})',
                        {
                            'symbol': symbol,
                            'quantity': quantity,
                            'exit_reason': exit_reason
                        }
                    )

                    # Clean up tracking
                    del self.active_positions[symbol]
                    if symbol in self.entry_times:
                        del self.entry_times[symbol]

        except Exception as e:
            logger.error(f"Error generating exit signals: {e}")

        return signals

    async def _calculate_position_quantity(self, symbol: str, price: float) -> int:
        """Calculate position quantity based on risk limits and price."""
        try:
            # Get portfolio value and calculate position size
            portfolio_value = await self._get_portfolio_value()
            position_value = portfolio_value * (self.position_size_pct / 100)

            # Calculate shares
            quantity = int(position_value / price)

            # Apply risk limits
            max_quantity = await self._calculate_max_position_size(symbol, price)
            quantity = min(quantity, max_quantity)

            # Ensure minimum liquidity
            volume_check = await self._check_volume_threshold(symbol)
            if not volume_check:
                quantity = 0

            return quantity

        except Exception as e:
            logger.error(f"Error calculating position quantity for {symbol}: {e}")
            return 0

    async def _calculate_max_position_size(self, symbol: str, price: float) -> int:
        """Calculate maximum position size based on risk limits."""
        portfolio_value = await self._get_portfolio_value()
        max_position_pct = self.config.risk_envelope.get('max_position_pct', self.max_position_size_pct)

        max_position_value = portfolio_value * (max_position_pct / 100)
        return int(max_position_value / price)

    async def _check_volume_threshold(self, symbol: str) -> bool:
        """Check if symbol meets minimum volume requirements."""
        # Simplified volume check - in production would query actual volume data
        return True  # Assume sufficient volume for universe stocks

    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value for position sizing."""
        return 10000000  # $10M portfolio

    async def _is_market_open_time(self, timestamp: datetime) -> bool:
        """Check if timestamp is around market open time."""
        # Market open is ~9:30 AM ET (13:30 UTC)
        market_open_hour = 13
        return abs(timestamp.hour - market_open_hour) <= 1  # Within 1 hour

    async def _is_market_close_time(self, timestamp: datetime) -> bool:
        """Check if timestamp is around market close time."""
        # Market close is ~4:00 PM ET (20:00 UTC)
        market_close_hour = 20
        return abs(timestamp.hour - market_close_hour) <= 1  # Within 1 hour

    async def shutdown(self) -> bool:
        """Shutdown the jump reversion strategy."""
        try:
            logger.info(f"Shutting down {self.config.name}")

            # Close any open positions
            if self.active_positions:
                await self._generate_exit_signals("strategy_shutdown")

            await self.audit_logger.log_event(
                'strategy_shutdown',
                f'Strategy {self.config.strategy_id} shut down successfully'
            )

            return True

        except Exception as e:
            logger.error(f"Error shutting down {self.config.name}: {e}")
            return False