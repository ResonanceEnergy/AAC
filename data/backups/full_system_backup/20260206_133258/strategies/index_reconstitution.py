"""
Index Reconstitution & Closing-Auction Liquidity Strategy
========================================================

Captures profits from index changes and closing auction inefficiencies.
This strategy exploits the most predictable arbitrage opportunities.

Strategy Logic:
- Monitor index reconstitution events (additions/deletions)
- Trade around closing auctions with superior liquidity detection
- Capture price dislocations from index changes
- Profit from predictable flows and market maker imbalances
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


class IndexReconstitutionStrategy(BaseArbitrageStrategy):
    """
    Index Reconstitution & Closing Auction Liquidity Strategy

    Captures arbitrage from index changes and closing auction inefficiencies.
    This is one of the most profitable and predictable arbitrage strategies.
    """

    def __init__(self, config: StrategyConfig, communication: CommunicationFramework, audit_logger: AuditLogger):
        super().__init__(config, communication, audit_logger)

        # Strategy parameters
        self.reconstitution_window_days = 5  # Days around reconstitution
        self.closing_auction_threshold = 0.0005  # 0.05% imbalance threshold
        self.min_volume_threshold = 500000  # Minimum daily volume
        self.max_position_size = 25000  # Max position per trade
        self.auction_participation_rate = 0.15  # Target auction participation

        # Index mappings
        self.index_constituents = {
            'SPX': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V'],
            'NDX': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'ADBE', 'CRM', 'NFLX'],
            'RUT': ['JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'PFE', 'KO', 'DIS']
        }

        # Track reconstitution events
        self.reconstitution_events = {}
        self.auction_imbalances = {}

    async def _initialize_strategy(self):
        """Initialize index reconstitution strategy components"""
        logger.info("Initializing Index Reconstitution Strategy")

        # Subscribe to index constituents and ETF data
        symbols_to_subscribe = []
        for constituents in self.index_constituents.values():
            symbols_to_subscribe.extend(constituents)

        # Add corresponding ETFs
        etf_mappings = {'SPX': 'SPY', 'NDX': 'QQQ', 'RUT': 'IWM'}
        symbols_to_subscribe.extend(etf_mappings.values())

        # Set market data subscriptions for the integration system
        self.market_data_subscriptions = set(symbols_to_subscribe)

        # Initialize reconstitution monitor
        self.reconstitution_monitor = ReconstitutionMonitor(self.index_constituents)

        logger.info(f"Index reconstitution strategy initialized for {len(self.index_constituents)} indices")

    async def _generate_signals(self) -> List[TradingSignal]:
        """Generate index reconstitution and closing auction signals"""
        signals = []

        # Check for reconstitution opportunities
        reconstitution_signals = await self._generate_reconstitution_signals()
        signals.extend(reconstitution_signals)

        # Check for closing auction opportunities
        auction_signals = await self._generate_auction_signals()
        signals.extend(auction_signals)

        return signals

    def _should_generate_signal(self) -> bool:
        """Determine if conditions are right to generate signals."""
        return True  # Simplified for testing

    async def _generate_reconstitution_signals(self) -> List[TradingSignal]:
        """Generate signals for index reconstitution events"""
        signals = []

        # Check for upcoming reconstitution events
        events = await self.reconstitution_monitor.get_upcoming_events()

        for event in events:
            try:
                signal = await self._analyze_reconstitution_event(event)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error analyzing reconstitution event {event}: {e}")
                continue

        return signals

    async def _analyze_reconstitution_event(self, event: Dict[str, Any]) -> Optional[TradingSignal]:
        """Analyze a specific reconstitution event for trading opportunities"""
        symbol = event['symbol']
        event_type = event['type']  # 'addition' or 'deletion'
        index = event['index']
        effective_date = event['effective_date']

        # Get current market data
        market_data = self.market_data.get(symbol, {})
        if not market_data or 'price' not in market_data:
            return None

        current_price = market_data['price']
        volume = market_data.get('volume', 0)

        # Check volume threshold
        if volume < self.min_volume_threshold:
            return None

        # Calculate reconstitution impact
        impact = await self._calculate_reconstitution_impact(event)

        if abs(impact) < 0.001:  # Less than 0.1% expected impact
            return None

        # Determine trade direction based on event type
        if event_type == 'addition':
            # Stock being added to index - expect price increase
            signal_type = SignalType.LONG
            confidence = min(abs(impact) / 0.01, 1.0)  # Scale confidence with impact
        else:  # deletion
            # Stock being removed from index - expect price decrease
            signal_type = SignalType.SHORT
            confidence = min(abs(impact) / 0.01, 1.0)

        # Calculate position size
        position_size = int(self.max_position_size * confidence)

        # Create signal
        signal = TradingSignal(
            strategy_id=self.config.strategy_id,
            signal_type=signal_type,
            symbol=symbol,
            quantity=position_size,
            confidence=confidence,
            metadata={
                'event_type': 'reconstitution',
                'index': index,
                'reconstitution_type': event_type,
                'effective_date': effective_date.isoformat(),
                'expected_impact': impact,
                'current_price': current_price,
                'strategy_type': 'index_reconstitution',
                'hold_period_days': self.reconstitution_window_days
            }
        )

        return signal

    async def _generate_auction_signals(self) -> List[TradingSignal]:
        """Generate signals for closing auction imbalances"""
        signals = []

        # Check if we're in closing auction window (last 30 minutes)
        now = datetime.now()
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        if now.weekday() >= 5:  # Weekend
            return signals

        auction_window_start = market_close - timedelta(minutes=30)
        auction_window_end = market_close - timedelta(minutes=5)  # Leave buffer

        if not (auction_window_start <= now <= auction_window_end):
            return signals

        # Analyze auction imbalances for each symbol
        for symbol in self.market_data.keys():
            try:
                signal = await self._analyze_auction_imbalance(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error analyzing auction for {symbol}: {e}")
                continue

        return signals

    async def _analyze_auction_imbalance(self, symbol: str) -> Optional[TradingSignal]:
        """Analyze closing auction imbalance for a symbol"""
        market_data = self.market_data.get(symbol, {})
        if not market_data or 'price' not in market_data:
            return None

        # Get order book data (simplified - would need actual order book feed)
        bid_size = market_data.get('bid_size', 0)
        ask_size = market_data.get('ask_size', 0)

        if bid_size == 0 or ask_size == 0:
            return None

        # Calculate imbalance ratio
        total_volume = bid_size + ask_size
        imbalance_ratio = (bid_size - ask_size) / total_volume

        # Check if imbalance exceeds threshold
        if abs(imbalance_ratio) < self.closing_auction_threshold:
            return None

        # Determine direction
        if imbalance_ratio > 0:
            # More buy orders - expect price uptick at close
            signal_type = SignalType.LONG
            confidence = min(abs(imbalance_ratio) / 0.1, 1.0)  # Scale with imbalance
        else:
            # More sell orders - expect price downtick at close
            signal_type = SignalType.SHORT
            confidence = min(abs(imbalance_ratio) / 0.1, 1.0)

        # Calculate position size based on auction participation target
        total_auction_volume = total_volume * self.auction_participation_rate
        position_size = int(total_auction_volume / market_data['price'])

        # Cap position size
        position_size = min(position_size, self.max_position_size)

        signal = TradingSignal(
            strategy_id=self.config.strategy_id,
            signal_type=signal_type,
            symbol=symbol,
            quantity=position_size,
            confidence=confidence,
            metadata={
                'event_type': 'closing_auction',
                'imbalance_ratio': imbalance_ratio,
                'bid_size': bid_size,
                'ask_size': ask_size,
                'auction_participation': self.auction_participation_rate,
                'strategy_type': 'closing_auction_liquidity',
                'hold_period': 'at_close'
            }
        )

        return signal

    async def _calculate_reconstitution_impact(self, event: Dict[str, Any]) -> float:
        """Calculate expected price impact from reconstitution event"""
        symbol = event['symbol']
        event_type = event['type']
        index = event['index']

        # Get index weight and market cap
        index_weight = await self._get_index_weight(symbol, index)
        market_cap = await self._get_market_cap(symbol)

        if market_cap == 0:
            return 0.0

        # Estimate flow impact
        index_aum = await self._get_index_aum(index)
        expected_flow = index_aum * index_weight

        # Price impact estimate (simplified model)
        # Additions typically cause 1-3% price increases
        # Deletions typically cause 1-3% price decreases
        base_impact = 0.02 if event_type == 'addition' else -0.02

        # Scale by flow size relative to market cap
        flow_ratio = expected_flow / market_cap
        impact = base_impact * min(flow_ratio * 10, 1.0)  # Cap at base impact

        return impact

    async def _get_index_weight(self, symbol: str, index: str) -> float:
        """Get weight of symbol in index"""
        # Simplified - in production would query index provider
        weight_mappings = {
            ('AAPL', 'SPX'): 0.12,
            ('MSFT', 'SPX'): 0.11,
            ('AAPL', 'NDX'): 0.12,
            ('MSFT', 'NDX'): 0.11,
        }
        return weight_mappings.get((symbol, index), 0.01)  # Default 1%

    async def _get_market_cap(self, symbol: str) -> float:
        """Get market cap of symbol"""
        # Simplified - in production would query financial data provider
        market_cap_mappings = {
            'AAPL': 3000000000000,  # $3T
            'MSFT': 2500000000000,  # $2.5T
            'AMZN': 1500000000000,  # $1.5T
        }
        return market_cap_mappings.get(symbol, 100000000000)  # Default $100B

    async def _get_index_aum(self, index: str) -> float:
        """Get assets under management for index"""
        # Simplified - in production would query index provider
        aum_mappings = {
            'SPX': 50000000000000,   # $50T
            'NDX': 25000000000000,   # $25T
            'RUT': 3000000000000,    # $3T
        }
        return aum_mappings.get(index, 10000000000000)  # Default $10T


class ReconstitutionMonitor:
    """Monitor for index reconstitution events"""

    def __init__(self, index_constituents: Dict[str, List[str]]):
        self.index_constituents = index_constituents

        # Track known reconstitution events
        # In production, this would be populated from index provider APIs
        self.known_events = [
            {
                'symbol': 'NVDA',
                'index': 'SPX',
                'type': 'addition',
                'effective_date': datetime(2024, 12, 23),  # Example future event
                'announced_date': datetime(2024, 12, 1)
            }
        ]

    async def get_upcoming_events(self) -> List[Dict[str, Any]]:
        """Get upcoming reconstitution events"""
        now = datetime.now()
        upcoming = []

        for event in self.known_events:
            effective_date = event['effective_date']
            days_until = (effective_date - now).days

            # Include events within our trading window
            if 0 <= days_until <= 5:  # 5 days window
                upcoming.append(event)

        return upcoming

    async def _initialize_strategy(self):
        """Initialize strategy-specific components."""
        # Subscribe to reconstitution data feeds
        await self.communication.subscribe_to_messages(
            self.config.strategy_id,
            ['index_reconstitution', 'closing_auction_data', 'liquidity_signals']
        )
        logger.info("Index Reconstitution strategy initialized")

    async def _subscribe_market_data(self):
        """Subscribe to required market data."""
        data_types = [
            'equity_price',
            'equity_volume',
            'order_book',
            'market_schedule'
        ]

        await self.communication.subscribe_to_messages(self.config.strategy_id, data_types)
        logger.info(f"Subscribed to market data types: {data_types}")

    def _update_market_data(self, data: Dict[str, Any]):
        """Update internal market data storage."""
        data_type = data.get('type')

        if data_type == 'index_reconstitution':
            symbol = data.get('symbol')
            if symbol:
                self.reconstitution_predictions[symbol] = {
                    'probability': data.get('probability', 0),
                    'direction': data.get('direction'),  # 'add' or 'delete'
                    'effective_date': data.get('effective_date'),
                    'timestamp': datetime.now()
                }
        elif data_type == 'closing_auction_data':
            venue = data.get('venue')
            if venue:
                self.closing_auction_schedule[venue] = data.get('close_time')
        elif data_type in ['equity_price', 'equity_volume', 'order_book']:
            symbol = data.get('symbol')
            if symbol:
                self.market_data[symbol] = data

    async def _generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals based on reconstitution predictions."""
        signals = []

        # Check for high-confidence reconstitution predictions
        for symbol, prediction in self.reconstitution_predictions.items():
            if (prediction['probability'] >= self.reconstitution_probability_threshold and
                symbol not in self.active_liquidity_positions):

                # Generate liquidity provision signal
                signal = await self._generate_liquidity_signal(symbol, prediction)
                if signal:
                    signals.append(signal)
                    self.active_liquidity_positions[symbol] = {
                        'entry_time': datetime.now(),
                        'prediction': prediction,
                        'position': signal
                    }

        # Check for exit signals near market close
        exit_signals = await self._generate_exit_signals()
        signals.extend(exit_signals)

        return signals

    async def _generate_liquidity_signal(self, symbol: str, prediction: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate liquidity provision signal for predicted reconstitution."""
        # Calculate position size based on expected volume
        position_size = self._calculate_position_size(symbol)

        # Determine direction based on prediction
        if prediction['direction'] == 'add':
            # Provide liquidity to buyers (sell into strength)
            signal_type = SignalType.SHORT
        elif prediction['direction'] == 'delete':
            # Provide liquidity to sellers (buy into weakness)
            signal_type = SignalType.LONG
        else:
            return None

        signal = TradingSignal(
            strategy_id=self.config.strategy_id,
            signal_type=signal_type,
            symbol=symbol,
            quantity=position_size if signal_type == SignalType.LONG else -position_size,
            confidence=prediction['probability'],
            metadata={
                'strategy_type': 'index_reconstitution',
                'prediction_direction': prediction['direction'],
                'effective_date': prediction['effective_date'],
                'liquidity_provision': True
            }
        )

        return signal

    async def _generate_exit_signals(self) -> List[TradingSignal]:
        """Generate exit signals for active positions."""
        signals = []
        current_time = datetime.now()

        for symbol, position_info in list(self.active_liquidity_positions.items()):
            entry_time = position_info['entry_time']
            holding_time = current_time - entry_time

            # Exit if held for target period or near market close
            if (holding_time >= timedelta(minutes=self.closing_auction_window_minutes) or
                self._is_near_market_close()):

                # Close position
                existing_position = position_info['position']
                exit_signal = TradingSignal(
                    strategy_id=self.config.strategy_id,
                    signal_type=SignalType.CLOSE,
                    symbol=symbol,
                    quantity=-existing_position.quantity,  # Close opposite
                    confidence=0.9,
                    metadata={
                        'exit_reason': 'target_holding_period' if holding_time >= timedelta(minutes=self.closing_auction_window_minutes) else 'market_close',
                        'holding_time_minutes': holding_time.total_seconds() / 60
                    }
                )

                signals.append(exit_signal)
                del self.active_liquidity_positions[symbol]

        return signals

    def _calculate_position_size(self, symbol: str) -> float:
        """Calculate position size for liquidity provision."""
        # Get average daily volume (simplified)
        avg_volume = 1000000  # Placeholder - would come from data

        # Position size as percentage of expected reconstitution volume
        position_size = avg_volume * (self.max_position_size_pct / 100)

        return min(position_size, 500000)  # Cap at 500k shares

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        current_time = datetime.now().time()
        market_open = datetime.strptime("09:30", "%H:%M").time()
        market_close = datetime.strptime("16:00", "%H:%M").time()

        return market_open <= current_time <= market_close

    def _is_near_market_close(self) -> bool:
        """Check if we're near market close."""
        current_time = datetime.now().time()
        market_close = datetime.strptime("16:00", "%H:%M").time()
        close_window = timedelta(minutes=self.closing_auction_window_minutes)

        return (datetime.combine(datetime.today(), market_close) -
                datetime.combine(datetime.today(), current_time)) <= close_window

    async def _close_all_positions(self):
        """Close all active positions."""
        close_signals = []

        for symbol, position_info in self.active_liquidity_positions.items():
            existing_position = position_info['position']
            close_signal = TradingSignal(
                strategy_id=self.config.strategy_id,
                signal_type=SignalType.CLOSE,
                symbol=symbol,
                quantity=-existing_position.quantity,
                confidence=1.0,
                metadata={'reason': 'strategy_shutdown'}
            )
            close_signals.append(close_signal)

        if close_signals:
            await self.communication.publish('strategy_signals', close_signals)

        self.active_liquidity_positions.clear()