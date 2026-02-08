#!/usr/bin/env python3
"""
Market Data Integration for AAC Strategies
==========================================
Integrates the comprehensive market data system with arbitrage strategies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.market_data_connector import (
    MarketDataManager,
    MarketData,
    ETFNAVData,
    initialize_market_data_system,
    DataSourceType
)
from shared.strategy_framework import StrategyFactory, BaseArbitrageStrategy
from shared.audit_logger import get_audit_logger
from shared.communication_framework import CommunicationFramework


@dataclass
class ArbitrageSignal:
    """Arbitrage trading signal"""
    strategy_id: str
    symbol: str
    signal_type: str  # 'long', 'short', 'close'
    quantity: float
    price: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketDataContext:
    """Market data context for strategies"""
    symbol: str
    current_price: Optional[MarketData] = None
    historical_prices: List[MarketData] = field(default_factory=list)
    order_book: Optional[Dict] = None
    nav_data: Optional[ETFNAVData] = None
    related_symbols: Dict[str, MarketData] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)


class MarketDataIntegration:
    """
    Integrates market data feeds with arbitrage strategy execution
    """

    def __init__(self):
        self.logger = logging.getLogger("MarketDataIntegration")
        self.audit_logger = get_audit_logger()
        self.market_data_manager = MarketDataManager()
        self.strategy_factory = StrategyFactory()
        self.communication_manager = CommunicationFramework()

        # Strategy execution state
        self.active_strategies: Dict[str, BaseArbitrageStrategy] = {}
        self.market_contexts: Dict[str, MarketDataContext] = {}
        self.pending_signals: List[ArbitrageSignal] = []

        # Callbacks
        self._signal_callbacks: List[Callable] = []

    def add_signal_callback(self, callback: Callable):
        """Add callback for arbitrage signals"""
        self._signal_callbacks.append(callback)

    async def initialize(self):
        """Initialize the market data integration system"""
        self.logger.info("Initializing market data integration...")

        # Initialize market data system
        await initialize_market_data_system()

        # Subscribe to market data updates
        self.market_data_manager.market_data_manager.subscribe(self._on_market_data_update)

        # Load and initialize strategies
        await self._load_strategies()

        self.logger.info("Market data integration initialized")

    async def _load_strategies(self):
        """Load and initialize arbitrage strategies"""
        config = get_config()
        strategy_config = config.get('strategy_department_matrix', {})

        # Load strategies from CSV
        import csv
        import os

        csv_path = os.path.join(PROJECT_ROOT, '50_arbitrage_strategies.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    strategy_id = row['id']
                    strategy_name = row['strategy_name']

                    try:
                        # Create strategy instance
                        strategy = self.strategy_factory.create_strategy(strategy_id)
                        if strategy:
                            self.active_strategies[strategy_id] = strategy

                            # Initialize strategy with market data context
                            await strategy.initialize()

                            self.logger.info(f"Loaded strategy: {strategy_id} - {strategy_name}")
                        else:
                            self.logger.warning(f"Failed to create strategy: {strategy_id}")

                    except Exception as e:
                        self.logger.error(f"Error loading strategy {strategy_id}: {e}")

        self.logger.info(f"Loaded {len(self.active_strategies)} strategies")

    async def _on_market_data_update(self, data: MarketData):
        """Handle incoming market data updates"""
        symbol = data.symbol

        # Update market context
        if symbol not in self.market_contexts:
            self.market_contexts[symbol] = MarketDataContext(symbol=symbol)

        context = self.market_contexts[symbol]
        context.current_price = data
        context.last_update = datetime.now()

        # Update related symbols context
        await self._update_related_symbols_context(symbol)

        # Check strategies that use this symbol
        await self._evaluate_strategies_for_symbol(symbol)

    async def _update_related_symbols_context(self, symbol: str):
        """Update context for symbols related to the given symbol"""
        # For ETFs, get holdings data
        if symbol in ['SPY', 'QQQ', 'IWM']:  # Major ETFs
            nav_data = self.market_data_manager.market_data_manager.get_nav_data(symbol)
            if nav_data:
                self.market_contexts[symbol].nav_data = nav_data

                # Update holdings prices
                for holding in nav_data.holdings:
                    holding_symbol = holding.symbol
                    if holding_symbol not in self.market_contexts:
                        self.market_contexts[holding_symbol] = MarketDataContext(symbol=holding_symbol)

                    # Get current price for holding
                    holding_price = self.market_data_manager.market_data_manager.get_latest_data(holding_symbol)
                    if holding_price:
                        self.market_contexts[symbol].related_symbols[holding_symbol] = holding_price

    async def _evaluate_strategies_for_symbol(self, symbol: str):
        """Evaluate strategies that depend on this symbol"""
        context = self.market_contexts.get(symbol)
        if not context or not context.current_price:
            return

        for strategy_id, strategy in self.active_strategies.items():
            try:
                # Check if strategy should generate signal
                should_generate = await strategy._should_generate_signal(context)
                if should_generate:
                    # Generate arbitrage signals
                    signals = await strategy._generate_signals(context)
                    for signal in signals:
                        arbitrage_signal = ArbitrageSignal(
                            strategy_id=strategy_id,
                            symbol=symbol,
                            signal_type=signal.get('type', 'unknown'),
                            quantity=signal.get('quantity', 0),
                            price=signal.get('price', context.current_price.price or 0),
                            confidence=signal.get('confidence', 0.5),
                            timestamp=datetime.now(),
                            metadata=signal
                        )

                        self.pending_signals.append(arbitrage_signal)

                        # Notify callbacks
                        for callback in self._signal_callbacks:
                            try:
                                await callback(arbitrage_signal)
                            except Exception as e:
                                self.logger.error(f"Signal callback error: {e}")

                        await self.audit_logger.log_event("strategy", "signal_generated", {
                            "strategy_id": strategy_id,
                            "symbol": symbol,
                            "signal_type": arbitrage_signal.signal_type,
                            "confidence": arbitrage_signal.confidence
                        })

            except Exception as e:
                self.logger.error(f"Error evaluating strategy {strategy_id} for {symbol}: {e}")

    async def get_market_context(self, symbol: str, include_historical: bool = False) -> Optional[MarketDataContext]:
        """Get complete market context for a symbol"""
        context = self.market_contexts.get(symbol)
        if not context:
            return None

        if include_historical:
            # Load historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Last 30 days

            historical_data = await self.market_data_manager.market_data_manager.get_historical_data(
                symbol, start_date, end_date
            )
            context.historical_prices = historical_data

        return context

    async def get_arbitrage_opportunities(self) -> List[Dict]:
        """Get current arbitrage opportunities across all symbols"""
        opportunities = []

        for symbol, context in self.market_contexts.items():
            if context.current_price:
                # Check for price discrepancies across feeds
                prices = self.market_data_manager.market_data_manager.get_best_prices(symbol)

                if len(prices) > 1:
                    # Find best bid/ask spread
                    bids = [p.bid for p in prices.values() if p.bid]
                    asks = [p.ask for p in prices.values() if p.ask]

                    if bids and asks:
                        best_bid = max(bids)
                        best_ask = min(asks)

                        if best_bid > best_ask:  # Arbitrage opportunity
                            spread = best_bid - best_ask
                            spread_pct = (spread / best_ask) * 100

                            opportunities.append({
                                "symbol": symbol,
                                "best_bid": best_bid,
                                "best_ask": best_ask,
                                "spread": spread,
                                "spread_pct": spread_pct,
                                "timestamp": datetime.now(),
                                "feeds": list(prices.keys())
                            })

        return opportunities

    def get_pending_signals(self) -> List[ArbitrageSignal]:
        """Get pending arbitrage signals"""
        return self.pending_signals.copy()

    def clear_signals(self, signal_ids: List[str] = None):
        """Clear pending signals"""
        if signal_ids:
            self.pending_signals = [s for s in self.pending_signals if s.strategy_id not in signal_ids]
        else:
            self.pending_signals.clear()

    async def start(self):
        """Start the market data integration"""
        await self.market_data_manager.market_data_manager.start()

        # Start strategy monitoring
        asyncio.create_task(self._monitor_strategies())

        self.logger.info("Market data integration started")

    async def _monitor_strategies(self):
        """Monitor strategy performance and health"""
        while True:
            try:
                # Check strategy health
                for strategy_id, strategy in self.active_strategies.items():
                    # Strategy-specific health checks would go here
                    pass

                # Clean up old market contexts (older than 1 hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                to_remove = []
                for symbol, context in self.market_contexts.items():
                    if context.last_update < cutoff_time:
                        to_remove.append(symbol)

                for symbol in to_remove:
                    del self.market_contexts[symbol]

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Strategy monitoring error: {e}")
                await asyncio.sleep(60)

    async def stop(self):
        """Stop the market data integration"""
        await self.market_data_manager.market_data_manager.stop()

        # Stop strategies
        for strategy in self.active_strategies.values():
            await strategy.cleanup()

        self.active_strategies.clear()
        self.market_contexts.clear()
        self.pending_signals.clear()

        self.logger.info("Market data integration stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "market_data": self.market_data_manager.market_data_manager.get_status(),
            "active_strategies": len(self.active_strategies),
            "market_contexts": len(self.market_contexts),
            "pending_signals": len(self.pending_signals),
            "strategy_types": list(set(s.__class__.__name__ for s in self.active_strategies.values())),
        }


# Global integration instance
market_data_integration = MarketDataIntegration()


async def initialize_market_data_integration():
    """Initialize the complete market data integration system"""
    await market_data_integration.initialize()

    # Connect to communication framework for order routing
    await market_data_integration.communication_manager.initialize()

    # Add signal processing callback
    async def process_signal(signal: ArbitrageSignal):
        """Process arbitrage signal through communication framework"""
        message = {
            "type": "arbitrage_signal",
            "strategy_id": signal.strategy_id,
            "symbol": signal.symbol,
            "signal_type": signal.signal_type,
            "quantity": signal.quantity,
            "price": signal.price,
            "confidence": signal.confidence,
            "timestamp": signal.timestamp.isoformat(),
            "metadata": signal.metadata
        }

        # Send to trading execution department
        await market_data_integration.communication_manager.send_message(
            "TradingExecution",
            "arbitrage_signal",
            message
        )

    market_data_integration.add_signal_callback(process_signal)

    await market_data_integration.start()

    audit_logger = get_audit_logger()
    await audit_logger.log_event("market_data_integration", "system_started", {
        "strategies": len(market_data_integration.active_strategies),
        "connectors": len(market_data_integration.market_data_manager.market_data_manager.connectors)
    })


if __name__ == "__main__":
    # Example usage
    asyncio.run(initialize_market_data_integration())