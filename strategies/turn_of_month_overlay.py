"""
Turn-of-the-Month Overlay Strategy
==================================

Strategy ID: s10_turn_of_the_month_overlay
Description: Own equities from last trading day to +3 days; implement via futures for capital efficiency.

Key Components:
- Calendar-based entry timing (last trading day of month)
- Position holding through first 3 trading days of new month
- Futures-based implementation for capital efficiency
- Risk management and position sizing
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from calendar import monthrange
import numpy as np

from shared.strategy_framework import BaseArbitrageStrategy, TradingSignal, SignalType, StrategyConfig
from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class TurnOfMonthOverlayStrategy(BaseArbitrageStrategy):
    """
    Turn-of-the-Month Overlay Strategy Implementation.

    This strategy exploits the well-documented "Turn-of-the-Month" effect where
    equity returns are significantly higher around month-end and month-beginning.
    The strategy goes long equities from the last trading day of the month through
    the first 3 trading days of the new month, implemented via futures for efficiency.
    """

    def __init__(self, config: StrategyConfig, communication: CommunicationFramework,
                 audit_logger: AuditLogger):
        super().__init__(config, communication, audit_logger)

        # Strategy-specific parameters
        self.holding_period_days = 3  # Hold through first 3 trading days of month
        self.position_size_pct = 10.0  # 10% allocation to TOM effect
        self.max_position_size_pct = 15.0  # Maximum position size limit
        self.entry_buffer_hours = 2  # Enter within 2 hours of month-end

        # TOM effect parameters
        self.tom_universe = ['ES', 'NQ', 'RTY', 'YM']  # Major equity index futures
        self.tom_weights = {'ES': 0.5, 'NQ': 0.3, 'RTY': 0.15, 'YM': 0.05}  # SPY heavy

        # Tracking variables
        self.active_positions = {}  # symbol -> position info
        self.tom_schedule = {}  # month -> entry/exit dates
        self.current_month_position = False

    async def _initialize_strategy(self):
        """Initialize strategy-specific components."""
        await self._setup_market_data_subscriptions()
        await self._initialize_tom_calendar()

    async def _generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals based on current market conditions."""
        signals = []

        try:
            # Check for TOM entry or exit conditions
            current_time = datetime.now()
            tom_status = await self._get_current_tom_status(current_time)

            if tom_status['in_tom_window'] and not self.current_month_position:
                # Enter TOM position
                signals.extend(await self._generate_tom_entry_signals())
                self.current_month_position = True

            elif not tom_status['in_tom_window'] and self.current_month_position:
                # Exit TOM position
                signals.extend(await self._generate_tom_exit_signals())
                self.current_month_position = False

        except Exception as e:
            logger.error(f"Error generating signals in TOM strategy: {e}")

        return signals

    def _should_generate_signal(self) -> bool:
        """Determine if conditions are right to generate signals."""
        # Generate signals on market open/close or during TOM windows
        current_time = datetime.now()
        return self._is_market_open_close_time(current_time) or self._is_in_tom_window(current_time)

    def _is_market_open_close_time(self, current_time: datetime) -> bool:
        """Check if current time is market open or close."""
        current_hour = current_time.hour
        return abs(current_hour - 9) <= 1 or abs(current_hour - 16) <= 1  # Around 9 AM or 4 PM ET

    def _is_in_tom_window(self, current_time: datetime) -> bool:
        """Check if we're currently in a TOM window."""
        current_date = current_time.date()

        for month_data in self.tom_schedule.values():
            entry_date = month_data['entry_date'].date()
            exit_date = month_data['exit_date'].date()

            if entry_date <= current_date <= exit_date:
                return True

        return False

    async def _setup_market_data_subscriptions(self):
        """Set up market data subscriptions for futures contracts."""
        data_types = ['futures_price', 'futures_volume', 'market_calendar']

        await self.communication.subscribe_to_messages(
            agent_id=self.config.strategy_id,
            message_types=data_types
        )

        logger.info(f"Subscribed to market data types: {data_types}")

    async def _initialize_tom_calendar(self):
        """Initialize the turn-of-month calendar for upcoming months."""
        current_date = datetime.now()

        # Calculate TOM dates for next 6 months
        for months_ahead in range(6):
            target_date = current_date.replace(day=1) + timedelta(days=32 * months_ahead)
            target_date = target_date.replace(day=1)  # First day of target month

            # Find last trading day of previous month
            last_day_prev_month = target_date - timedelta(days=1)
            last_trading_day = await self._get_last_trading_day(last_day_prev_month)

            # Find first 3 trading days of target month
            first_trading_days = await self._get_first_trading_days(target_date, 3)

            self.tom_schedule[target_date.strftime('%Y-%m')] = {
                'entry_date': last_trading_day,
                'exit_date': first_trading_days[-1],  # Last of the 3 days
                'holding_days': first_trading_days
            }

        logger.info(f"Initialized TOM calendar for {len(self.tom_schedule)} months")

    async def _get_last_trading_day(self, month_end: datetime) -> datetime:
        """Get the last trading day of the month (excluding weekends)."""
        # Simple approximation - last business day of month
        # In production, this would check actual trading calendar
        last_day = month_end
        while last_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
            last_day -= timedelta(days=1)
        return last_day

    async def _get_first_trading_days(self, month_start: datetime, num_days: int) -> List[datetime]:
        """Get the first N trading days of the month."""
        trading_days = []
        current_date = month_start

        while len(trading_days) < num_days:
            if current_date.weekday() < 5:  # Monday-Friday
                trading_days.append(current_date)
            current_date += timedelta(days=1)

        return trading_days

    async def process_market_data(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """Process market data and generate TOM overlay signals."""
        signals = []

        try:
            data_type = data.get('type')

            if data_type == 'market_calendar':
                await self._update_tom_schedule(data)
            elif data_type in ['futures_price', 'futures_volume']:
                signals.extend(await self._check_tom_entry_exit(data))

        except Exception as e:
            logger.error(f"Error processing market data in TOM strategy: {e}")

        return signals

    async def _check_tom_entry_exit(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """Check for TOM entry or exit conditions."""
        signals = []
        current_time = datetime.now()

        # Check if we're in a TOM window
        tom_status = await self._get_current_tom_status(current_time)

        if tom_status['in_tom_window'] and not self.current_month_position:
            # Enter TOM position
            signals.extend(await self._generate_tom_entry_signals(data))
            self.current_month_position = True

        elif not tom_status['in_tom_window'] and self.current_month_position:
            # Exit TOM position
            signals.extend(await self._generate_tom_exit_signals(data))
            self.current_month_position = False

        return signals

    async def _get_current_tom_status(self, current_time: datetime) -> Dict[str, Any]:
        """Determine if we're currently in a TOM window."""
        current_date = current_time.date()

        for month_data in self.tom_schedule.values():
            entry_date = month_data['entry_date'].date()
            exit_date = month_data['exit_date'].date()

            if entry_date <= current_date <= exit_date:
                days_into_position = (current_date - entry_date).days
                return {
                    'in_tom_window': True,
                    'days_held': days_into_position,
                    'entry_date': entry_date,
                    'exit_date': exit_date
                }

        return {'in_tom_window': False}

    async def _generate_tom_entry_signals(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals to enter TOM positions."""
        signals = []

        try:
            # Calculate position sizes based on risk limits
            total_portfolio_value = await self._get_portfolio_value()
            max_position_value = total_portfolio_value * (self.position_size_pct / 100)

            for symbol, weight in self.tom_weights.items():
                position_value = max_position_value * weight
                quantity = await self._calculate_futures_quantity(symbol, position_value)

                if quantity > 0:
                    signal = TradingSignal(
                        strategy_id=self.config.strategy_id,
                        signal_type=SignalType.LONG,
                        symbol=symbol,
                        quantity=quantity,
                        confidence=0.75,  # TOM effect has ~75% success rate historically
                        metadata={
                            'strategy_type': 'turn_of_month_overlay',
                            'entry_type': 'tom_entry',
                            'expected_holding_days': self.holding_period_days,
                            'position_value': position_value,
                            'weight': weight
                        }
                    )
                    signals.append(signal)

                    await self.audit_logger.log_event(
                        'tom_position_entry',
                        f'Entering TOM position for {symbol}: {quantity} contracts',
                        {'symbol': symbol, 'quantity': quantity, 'position_value': position_value}
                    )

        except Exception as e:
            logger.error(f"Error generating TOM entry signals: {e}")

        return signals

    async def _generate_tom_exit_signals(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals to exit TOM positions."""
        signals = []

        try:
            # Exit all current TOM positions
            for symbol in self.tom_universe:
                if symbol in self.active_positions:
                    position = self.active_positions[symbol]
                    quantity = -position['quantity']  # Close position

                    signal = TradingSignal(
                        strategy_id=self.config.strategy_id,
                        signal_type=SignalType.FLAT,
                        symbol=symbol,
                        quantity=quantity,
                        confidence=0.80,  # High confidence for planned exits
                        metadata={
                            'strategy_type': 'turn_of_month_overlay',
                            'entry_type': 'tom_exit',
                            'exit_reason': 'tom_window_closed',
                            'holding_days': position.get('holding_days', 0)
                        }
                    )
                    signals.append(signal)

                    await self.audit_logger.log_event(
                        'tom_position_exit',
                        f'Exiting TOM position for {symbol}: {quantity} contracts',
                        {'symbol': symbol, 'quantity': quantity}
                    )

        except Exception as e:
            logger.error(f"Error generating TOM exit signals: {e}")

        return signals

    async def _calculate_futures_quantity(self, symbol: str, position_value: float) -> int:
        """Calculate number of futures contracts for target position value."""
        try:
            # Get current futures price (simplified - would use actual market data)
            futures_price = await self._get_futures_price(symbol)
            if not futures_price:
                return 0

            # Calculate contracts needed (futures use notional value)
            contract_multiplier = self._get_contract_multiplier(symbol)
            contract_value = futures_price * contract_multiplier

            quantity = int(position_value / contract_value)

            # Apply risk limits
            max_quantity = await self._calculate_max_position_size(symbol)
            quantity = min(quantity, max_quantity)

            return quantity

        except Exception as e:
            logger.error(f"Error calculating futures quantity for {symbol}: {e}")
            return 0

    def _get_contract_multiplier(self, symbol: str) -> int:
        """Get contract multiplier for futures symbol."""
        multipliers = {
            'ES': 50,    # E-mini S&P 500
            'NQ': 20,    # E-mini Nasdaq-100
            'RTY': 50,   # E-mini Russell 2000
            'YM': 100    # E-mini Dow
        }
        return multipliers.get(symbol, 50)

    async def _get_futures_price(self, symbol: str) -> Optional[float]:
        """Get current futures price (simplified implementation)."""
        # In production, this would query real-time market data
        # For now, return approximate prices
        base_prices = {
            'ES': 4800,   # S&P 500 futures
            'NQ': 16800,  # Nasdaq futures
            'RTY': 2050,  # Russell 2000 futures
            'YM': 37500   # Dow futures
        }
        return base_prices.get(symbol)

    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value for position sizing."""
        # Simplified - would integrate with portfolio management system
        return 10000000  # $10M portfolio

    async def _calculate_max_position_size(self, symbol: str) -> int:
        """Calculate maximum position size based on risk limits."""
        portfolio_value = await self._get_portfolio_value()
        max_position_pct = self.config.risk_envelope.get('max_position_pct', self.max_position_size_pct)

        max_position_value = portfolio_value * (max_position_pct / 100)
        futures_price = await self._get_futures_price(symbol)

        if futures_price:
            contract_multiplier = self._get_contract_multiplier(symbol)
            contract_value = futures_price * contract_multiplier
            return int(max_position_value / contract_value)

        return 0

    async def _update_tom_schedule(self, calendar_data: Dict[str, Any]):
        """Update TOM schedule based on market calendar changes."""
        # Handle calendar updates (holidays, trading days, etc.)
        # This would update the TOM schedule if there are calendar changes
        pass

    async def shutdown(self) -> bool:
        """Shutdown the TOM strategy."""
        try:
            logger.info(f"Shutting down {self.config.name}")

            # Close any open positions
            if self.current_month_position:
                # Generate emergency exit signals
                pass

            await self.audit_logger.log_event(
                'strategy_shutdown',
                f'Strategy {self.config.strategy_id} shut down successfully'
            )

            return True

        except Exception as e:
            logger.error(f"Error shutting down {self.config.name}: {e}")
            return False