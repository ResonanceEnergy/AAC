"""
TOM Futures-Only Overlay Strategy
=================================

Strategy ID: s49_tom_futures_only_overlay
Description: Exploit TOM via S&P futures; strong persistence documented in futures.

Key Components:
- Futures-based Turn-of-the-Month effect
- Last trading day entry through first 3 trading days
- Pure futures implementation for capital efficiency
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


class TOMFuturesOnlyOverlayStrategy(BaseArbitrageStrategy):
    """
    TOM Futures-Only Overlay Strategy Implementation.

    This strategy exploits the Turn-of-the-Month effect using futures contracts,
    which have shown strong persistence in the TOM effect. The strategy goes long
    equity index futures from the last trading day of the month through the first
    3 trading days of the new month.

    Key Advantages:
    - Futures provide 24/7 trading access
    - Capital efficient (no underlying stock ownership)
    - Strong historical persistence in futures markets
    - Lower transaction costs
    """

    def __init__(self, config: StrategyConfig, communication: CommunicationFramework,
                 audit_logger: AuditLogger):
        super().__init__(config, communication, audit_logger)

        # Strategy-specific parameters
        self.holding_period_days = 3  # Hold through first 3 trading days of month
        self.position_size_pct = 12.0  # 12% allocation to TOM futures effect
        self.max_position_size_pct = 18.0  # Maximum position size limit
        self.entry_buffer_hours = 1  # Enter within 1 hour of close/open

        # TOM futures parameters
        self.futures_universe = ['ES', 'NQ', 'RTY', 'YM']  # Major equity index futures
        self.futures_weights = {
            'ES': 0.5,   # S&P 500 futures - primary focus
            'NQ': 0.25,  # Nasdaq-100 futures
            'RTY': 0.15, # Russell 2000 futures
            'YM': 0.1    # Dow Jones futures
        }

        # Futures-specific parameters
        self.contract_roll_days = 5  # Roll contracts 5 days before expiration
        self.min_liquidity_threshold = 10000  # Minimum open interest

        # Tracking variables
        self.active_positions = {}  # symbol -> position info
        self.tom_schedule = {}  # month -> entry/exit dates
        self.current_month_position = False
        self.active_contracts = {}  # symbol -> current contract

    async def _initialize_strategy(self):
        """Initialize strategy-specific components."""
        await self._setup_market_data_subscriptions()
        await self._initialize_tom_calendar()
        await self._initialize_contract_tracking()

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
            logger.error(f"Error generating signals in TOM futures strategy: {e}")

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
        data_types = ['futures_price', 'futures_volume', 'futures_open_interest', 'market_calendar']

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

    async def _initialize_contract_tracking(self):
        """Initialize tracking of active futures contracts."""
        for symbol in self.futures_universe:
            self.active_contracts[symbol] = await self._get_active_contract(symbol)

        logger.info("Initialized futures contract tracking")

    async def _get_last_trading_day(self, month_end: datetime) -> datetime:
        """Get the last trading day of the month."""
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
        """Process market data and generate TOM futures overlay signals."""
        signals = []

        try:
            data_type = data.get('type')
            symbol = data.get('symbol')

            if symbol not in self.futures_universe:
                return signals

            if data_type == 'market_calendar':
                await self._update_tom_schedule(data)
            elif data_type in ['futures_price', 'futures_volume']:
                signals.extend(await self._check_tom_entry_exit(data))
            elif data_type == 'futures_open_interest':
                await self._check_contract_rollover(data)

        except Exception as e:
            logger.error(f"Error processing market data in TOM futures strategy: {e}")

        return signals

    async def _check_tom_entry_exit(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """Check for TOM entry or exit conditions."""
        signals = []
        current_time = datetime.now()

        # Check if we should roll contracts
        await self._check_contract_rollover(data)

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
        """Generate signals to enter TOM futures positions."""
        signals = []

        try:
            # Calculate position sizes based on risk limits
            total_portfolio_value = await self._get_portfolio_value()
            max_position_value = total_portfolio_value * (self.position_size_pct / 100)

            for symbol, weight in self.futures_weights.items():
                position_value = max_position_value * weight
                contract_symbol = self.active_contracts.get(symbol, symbol)
                quantity = await self._calculate_futures_quantity(contract_symbol, position_value)

                if quantity > 0:
                    signal = TradingSignal(
                        strategy_id=self.config.strategy_id,
                        signal_type=SignalType.LONG,
                        symbol=contract_symbol,
                        quantity=quantity,
                        confidence=0.80,  # TOM effect in futures has ~80% success rate
                        metadata={
                            'strategy_type': 'tom_futures_overlay',
                            'entry_type': 'tom_entry',
                            'expected_holding_days': self.holding_period_days,
                            'position_value': position_value,
                            'weight': weight,
                            'underlying_symbol': symbol
                        }
                    )
                    signals.append(signal)

                    await self.audit_logger.log_event(
                        'tom_futures_entry',
                        f'Entering TOM position for {contract_symbol}: {quantity} contracts',
                        {'symbol': contract_symbol, 'quantity': quantity, 'position_value': position_value}
                    )

        except Exception as e:
            logger.error(f"Error generating TOM futures entry signals: {e}")

        return signals

    async def _generate_tom_exit_signals(self, data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals to exit TOM futures positions."""
        signals = []

        try:
            # Exit all current TOM positions
            for symbol in self.futures_universe:
                contract_symbol = self.active_contracts.get(symbol, symbol)
                if contract_symbol in self.active_positions:
                    position = self.active_positions[contract_symbol]
                    quantity = -position['quantity']  # Close position

                    signal = TradingSignal(
                        strategy_id=self.config.strategy_id,
                        signal_type=SignalType.FLAT,
                        symbol=contract_symbol,
                        quantity=quantity,
                        confidence=0.85,  # High confidence for planned exits
                        metadata={
                            'strategy_type': 'tom_futures_overlay',
                            'entry_type': 'tom_exit',
                            'exit_reason': 'tom_window_closed',
                            'holding_days': position.get('holding_days', 0)
                        }
                    )
                    signals.append(signal)

                    await self.audit_logger.log_event(
                        'tom_futures_exit',
                        f'Exiting TOM position for {contract_symbol}: {quantity} contracts',
                        {'symbol': contract_symbol, 'quantity': quantity}
                    )

        except Exception as e:
            logger.error(f"Error generating TOM futures exit signals: {e}")

        return signals

    async def _calculate_futures_quantity(self, symbol: str, position_value: float) -> int:
        """Calculate number of futures contracts for target position value."""
        try:
            # Get current futures price
            futures_price = await self._get_futures_price(symbol)
            if not futures_price:
                return 0

            # Calculate contracts needed
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
        # Extract base symbol (remove month/year codes)
        base_symbol = symbol.rstrip('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if not base_symbol:
            base_symbol = symbol[:2]  # Fallback for symbols like ES, NQ

        multipliers = {
            'ES': 50,    # E-mini S&P 500
            'NQ': 20,    # E-mini Nasdaq-100
            'RTY': 50,   # E-mini Russell 2000
            'YM': 100    # E-mini Dow
        }
        return multipliers.get(base_symbol, 50)

    async def _get_futures_price(self, symbol: str) -> Optional[float]:
        """Get current futures price."""
        # In production, this would query real-time market data
        base_prices = {
            'ES': 4800,   # S&P 500 futures
            'NQ': 16800,  # Nasdaq futures
            'RTY': 2050,  # Russell 2000 futures
            'YM': 37500   # Dow futures
        }

        # Extract base symbol
        base_symbol = symbol.rstrip('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if not base_symbol:
            base_symbol = symbol[:2]

        return base_prices.get(base_symbol)

    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value for position sizing."""
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

    async def _get_active_contract(self, symbol: str) -> str:
        """Get the active contract symbol for the given futures symbol."""
        # Simplified - in production would determine correct contract month
        current_month = datetime.now().strftime('%m')
        current_year = datetime.now().strftime('%y')

        # Map symbols to contract codes
        contract_codes = {
            'ES': 'ESH',   # March S&P 500
            'NQ': 'NQH',   # March Nasdaq-100
            'RTY': 'RTH',  # March Russell 2000
            'YM': 'YMH'    # March Dow
        }

        return contract_codes.get(symbol, symbol)

    async def _check_contract_rollover(self, data: Dict[str, Any]):
        """Check if futures contracts need to be rolled over."""
        # Simplified rollover logic - in production would check expiration dates
        # and roll to next contract when current contract is close to expiration
        pass

    async def _update_tom_schedule(self, calendar_data: Dict[str, Any]):
        """Update TOM schedule based on market calendar changes."""
        # Handle calendar updates (holidays, trading days, etc.)
        pass

    async def shutdown(self) -> bool:
        """Shutdown the TOM futures overlay strategy."""
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