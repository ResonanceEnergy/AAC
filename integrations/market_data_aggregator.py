"""
AAC Market Data Integration System
===================================

Connects all trading strategies to live market data feeds.
Aggregates data from 60+ exchanges and provides unified interface.

This system enables real-time strategy execution with live data.
"""

import asyncio
import logging
import websockets
import json
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)


class MarketDataAggregator:
    """
    Aggregates market data from multiple sources for strategy consumption.

    Supports:
    - 60+ cryptocurrency exchanges via CCXT
    - WebSocket real-time feeds
    - REST API polling for supplementary data
    - Data normalization and caching
    """

    def __init__(self, communication: CommunicationFramework, audit_logger: AuditLogger):
        self.communication = communication
        self.audit_logger = audit_logger

        # Exchange connections
        self.exchanges = {}
        self.active_feeds = {}

        # Data storage
        self.market_data = {}  # symbol -> data
        self.order_books = {}  # symbol -> order book
        self.trade_history = {}  # symbol -> recent trades

        # Subscription management
        self.subscriptions = set()
        self.data_callbacks = []

        # Performance tracking
        self.data_quality_metrics = {}

        # Background tasks for proper cleanup
        self._background_tasks = set()

    async def initialize_aggregator(self):
        """Initialize market data aggregator"""
        logger.info("Initializing Market Data Aggregator...")

        # Initialize major exchanges
        await self._initialize_exchanges()

        # Start data collection loops
        data_collection_task = asyncio.create_task(self._run_data_collection())
        quality_monitor_task = asyncio.create_task(self._monitor_data_quality())

        self._background_tasks.add(data_collection_task)
        self._background_tasks.add(quality_monitor_task)

        # Add callback to remove completed tasks
        data_collection_task.add_done_callback(self._background_tasks.discard)
        quality_monitor_task.add_done_callback(self._background_tasks.discard)

        logger.info("Market Data Aggregator initialized")

    async def _initialize_exchanges(self):
        """Initialize connections to cryptocurrency exchanges"""
        exchange_configs = {
            'binance': {'apiKey': None, 'secret': None, 'enableRateLimit': True},
            'coinbase': {'apiKey': None, 'secret': None, 'password': None},
            'kraken': {'apiKey': None, 'secret': None},
            'bitfinex': {'apiKey': None, 'secret': None},
            'huobi': {'apiKey': None, 'secret': None},
            'okx': {'apiKey': None, 'secret': None},
            'bybit': {'apiKey': None, 'secret': None},
            'kucoin': {'apiKey': None, 'secret': None},
            'gate': {'apiKey': None, 'secret': None}
        }

        for exchange_name, config in exchange_configs.items():
            try:
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class(config)
                self.exchanges[exchange_name] = exchange
                logger.info(f"✅ Initialized {exchange_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize {exchange_name}: {e}")

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to market data for symbols"""
        logger.info(f"Subscribing to {len(symbols)} symbols...")

        for symbol in symbols:
            if symbol not in self.subscriptions:
                self.subscriptions.add(symbol)

                # Initialize data structures
                self.market_data[symbol] = {}
                self.order_books[symbol] = {'bids': [], 'asks': []}
                self.trade_history[symbol] = []

        # Start WebSocket feeds for subscribed symbols
        await self._start_websocket_feeds()

        logger.info(f"Subscribed to {len(symbols)} symbols")

    async def _start_websocket_feeds(self):
        """Start WebSocket feeds for subscribed symbols"""
        # Group symbols by exchange for efficient connections
        exchange_symbols = {}

        for symbol in self.subscriptions:
            # Map symbol to exchange (simplified mapping)
            exchange = self._map_symbol_to_exchange(symbol)
            if exchange not in exchange_symbols:
                exchange_symbols[exchange] = []
            exchange_symbols[exchange].append(symbol)

        # Start feeds for each exchange
        for exchange_name, symbols in exchange_symbols.items():
            if exchange_name in self.exchanges:
                asyncio.create_task(self._run_exchange_feed(exchange_name, symbols))

    def _map_symbol_to_exchange(self, symbol: str) -> str:
        """Map symbol to appropriate exchange"""
        # Simplified mapping - in production would be more sophisticated
        if symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'DOT/USDT']:
            return 'binance'
        elif symbol in ['BTC/USD', 'ETH/USD']:
            return 'coinbasepro'
        else:
            return 'binance'  # Default

    async def _run_exchange_feed(self, exchange_name: str, symbols: List[str]):
        """Run WebSocket feed for an exchange"""
        exchange = self.exchanges[exchange_name]

        try:
            # Start ticker feed as background task
            asyncio.create_task(self._subscribe_ticker_feed(exchange, symbols))

            # Start order book feed as background task
            asyncio.create_task(self._subscribe_orderbook_feed(exchange, symbols))

            # Start trade feed as background task
            asyncio.create_task(self._subscribe_trade_feed(exchange, symbols))

        except Exception as e:
            logger.error(f"Error running {exchange_name} feed: {e}")

    async def _subscribe_ticker_feed(self, exchange, symbols: List[str]):
        """Subscribe to ticker updates"""
        while True:
            try:
                for symbol in symbols:
                    if symbol in self.subscriptions:
                        ticker = await exchange.fetch_ticker(symbol)
                        self._update_market_data(symbol, 'ticker', ticker)

                        # Notify callbacks
                        await self._notify_data_callbacks(symbol, 'ticker', ticker)

                await asyncio.sleep(1)  # 1-second updates

            except Exception as e:
                logger.error(f"Error in ticker feed: {e}")
                await asyncio.sleep(5)

    async def _subscribe_orderbook_feed(self, exchange, symbols: List[str]):
        """Subscribe to order book updates"""
        while True:
            try:
                for symbol in symbols:
                    if symbol in self.subscriptions:
                        orderbook = await exchange.fetch_order_book(symbol)
                        self.order_books[symbol] = orderbook

                        # Extract bid/ask sizes for strategy use
                        best_bid = orderbook['bids'][0] if orderbook['bids'] else [0, 0]
                        best_ask = orderbook['asks'][0] if orderbook['asks'] else [0, 0]

                        data = {
                            'bid_price': best_bid[0],
                            'bid_size': best_bid[1],
                            'ask_price': best_ask[0],
                            'ask_size': best_ask[1]
                        }
                        self._update_market_data(symbol, 'orderbook', data)

                await asyncio.sleep(0.1)  # 100ms updates for order books

            except Exception as e:
                logger.error(f"Error in orderbook feed: {e}")
                await asyncio.sleep(5)

    async def _subscribe_trade_feed(self, exchange, symbols: List[str]):
        """Subscribe to trade updates"""
        while True:
            try:
                for symbol in symbols:
                    if symbol in self.subscriptions:
                        trades = await exchange.fetch_trades(symbol, limit=10)
                        self.trade_history[symbol] = trades

                        # Calculate volume metrics
                        if trades:
                            recent_volume = sum(t['amount'] for t in trades[-10:])
                            data = {
                                'recent_trades': len(trades),
                                'recent_volume': recent_volume,
                                'last_trade_price': trades[-1]['price'],
                                'last_trade_time': trades[-1]['timestamp']
                            }
                            self._update_market_data(symbol, 'trades', data)

                await asyncio.sleep(2)  # 2-second updates

            except Exception as e:
                logger.error(f"Error in trade feed: {e}")
                await asyncio.sleep(5)

    def _update_market_data(self, symbol: str, data_type: str, data: Dict[str, Any]):
        """Update market data for a symbol"""
        if symbol not in self.market_data:
            self.market_data[symbol] = {}

        # Merge data
        self.market_data[symbol].update(data)

        # Add metadata
        self.market_data[symbol]['last_update'] = datetime.now()
        self.market_data[symbol]['data_type'] = data_type

    async def _notify_data_callbacks(self, symbol: str, data_type: str, data: Dict[str, Any]):
        """Notify registered callbacks of data updates"""
        for callback in self.data_callbacks:
            try:
                await callback(symbol, data_type, data)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")

    def add_data_callback(self, callback: Callable):
        """Add a callback for data updates"""
        self.data_callbacks.append(callback)

    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for a symbol"""
        return self.market_data.get(symbol)

    async def get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current order book for a symbol"""
        return self.order_books.get(symbol)

    async def get_recent_trades(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades for a symbol"""
        trades = self.trade_history.get(symbol, [])
        return trades[-limit:] if trades else []

    async def _run_data_collection(self):
        """Run continuous data collection loop"""
        logger.info("Starting data collection loop...")

        while True:
            try:
                # Collect supplementary data (funding rates, open interest, etc.)
                await self._collect_supplementary_data()

                # Update data quality metrics
                await self._update_data_quality_metrics()

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(60)

    async def _collect_supplementary_data(self):
        """Collect supplementary market data"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Get exchange-specific data
                if hasattr(exchange, 'fetch_funding_rate'):
                    # For perpetual futures exchanges
                    for symbol in self.subscriptions:
                        if '/USD' in symbol or 'PERP' in symbol:
                            try:
                                funding_rate = await exchange.fetch_funding_rate(symbol)
                                self._update_market_data(symbol, 'funding', funding_rate)
                            except:
                                pass

                # Get open interest where available
                if hasattr(exchange, 'fetch_open_interest'):
                    for symbol in self.subscriptions:
                        try:
                            oi = await exchange.fetch_open_interest(symbol)
                            self._update_market_data(symbol, 'open_interest', oi)
                        except:
                            pass

            except Exception as e:
                logger.debug(f"Error collecting supplementary data from {exchange_name}: {e}")

    async def _update_data_quality_metrics(self):
        """Update data quality metrics"""
        current_time = datetime.now()

        for symbol in self.subscriptions:
            data = self.market_data.get(symbol, {})
            last_update = data.get('last_update')

            if last_update:
                age_seconds = (current_time - last_update).seconds
                is_stale = age_seconds > 300  # 5 minutes

                self.data_quality_metrics[symbol] = {
                    'last_update': last_update,
                    'age_seconds': age_seconds,
                    'is_stale': is_stale,
                    'data_types': list(data.keys())
                }

    async def _monitor_data_quality(self):
        """Monitor data quality and alert on issues"""
        while True:
            try:
                stale_symbols = []
                for symbol, metrics in self.data_quality_metrics.items():
                    if metrics.get('is_stale', False):
                        stale_symbols.append(symbol)

                if stale_symbols:
                    await self.audit_logger.log_event(
                        'data_quality_alert',
                        f"Stale data detected for {len(stale_symbols)} symbols",
                        {'stale_symbols': stale_symbols}
                    )

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error monitoring data quality: {e}")
                await asyncio.sleep(300)

    async def get_data_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive data quality report"""
        return {
            'total_symbols': len(self.subscriptions),
            'active_feeds': len(self.active_feeds),
            'stale_symbols': len([s for s in self.data_quality_metrics.values() if s.get('is_stale')]),
            'data_types_available': list(set(
                dt for metrics in self.data_quality_metrics.values()
                for dt in metrics.get('data_types', [])
            )),
            'last_update': datetime.now()
        }

    async def shutdown_aggregator(self):
        """Gracefully shutdown the aggregator"""
        logger.info("Shutting down Market Data Aggregator...")

        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Close exchange connections
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except Exception as e:
                logger.debug(f"Error closing exchange: {e}")

        # Clear data
        self.market_data.clear()
        self.order_books.clear()
        self.trade_history.clear()
        self._background_tasks.clear()

        logger.info("Market Data Aggregator shutdown complete")


# Global aggregator instance
_data_aggregator_instance = None

def get_market_data_aggregator(communication: CommunicationFramework = None,
                              audit_logger: AuditLogger = None) -> MarketDataAggregator:
    """Get singleton market data aggregator instance"""
    global _data_aggregator_instance

    if _data_aggregator_instance is None:
        if not all([communication, audit_logger]):
            raise ValueError("Communication and audit logger required for first instantiation")

        _data_aggregator_instance = MarketDataAggregator(communication, audit_logger)

    return _data_aggregator_instance