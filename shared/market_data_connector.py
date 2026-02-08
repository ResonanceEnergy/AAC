#!/usr/bin/env python3
"""
Comprehensive Market Data Connector System
==========================================
Unified interface for 100+ worldwide market data feeds with redundancy and failover.
"""

import asyncio
import logging
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set, Union
from enum import Enum
from pathlib import Path
import aiohttp
import websockets
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.utils import retry, RetryStrategy, with_circuit_breaker
from shared.audit_logger import get_audit_logger

# Simple audit log wrapper
async def audit_log(category: str, action: str, details: dict = None):
    """Simple audit logging wrapper"""
    logger = get_audit_logger()
    # For now, just log to console - in production this would use the full audit system
    print(f"AUDIT: {category} | {action} | {details or {}}")


class DataSourceType(Enum):
    """Types of market data sources"""
    EXCHANGE_DIRECT = "exchange_direct"  # Direct exchange feeds (NYSE, CME, NASDAQ)
    DATA_VENDOR = "data_vendor"         # Bloomberg, Refinitiv, etc.
    CRYPTO_EXCHANGE = "crypto_exchange" # Coinbase, Binance, etc.
    FOREX_PROVIDER = "forex_provider"   # FX feeds
    COMMODITY_FEED = "commodity_feed"   # Commodity prices
    INDEX_FEED = "index_feed"          # Index data
    NEWS_FEED = "news_feed"            # News and sentiment


class FeedStatus(Enum):
    """Feed connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    STALE = "stale"


@dataclass
class MarketData:
    """Unified market data structure"""
    symbol: str
    asset_class: str  # 'equity', 'future', 'option', 'crypto', 'forex', 'commodity'
    exchange: str
    price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    volume_24h: Optional[float] = None
    change_24h: Optional[float] = None
    change_pct_24h: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    vwap: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    quality_score: float = 1.0  # 0-1, based on timeliness and completeness


@dataclass
class OrderBookData:
    """Order book snapshot"""
    symbol: str
    exchange: str
    bids: List[tuple] = field(default_factory=list)  # [(price, size), ...]
    asks: List[tuple] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""


@dataclass
class ETFHolding:
    """ETF holding data for NAV calculation"""
    symbol: str
    name: str
    shares: int
    weight: float  # Percentage weight in ETF
    price: Optional[float] = None
    market_value: Optional[float] = None


@dataclass
class ETFNAVData:
    """ETF NAV calculation data"""
    symbol: str
    nav_price: float
    market_price: float
    premium_discount: float  # (market - NAV) / NAV * 100
    holdings: List[ETFHolding]
    total_assets: float
    shares_outstanding: int
    timestamp: datetime
    source: str = "calculated"


class BaseMarketDataConnector(ABC):
    """Base class for all market data connectors"""

    def __init__(self, name: str, source_type: DataSourceType):
        self.name = name
        self.source_type = source_type
        self.config = get_config()
        self.logger = logging.getLogger(f"connector_{name}")
        self.status = FeedStatus.DISCONNECTED
        self.last_update = None
        self._callbacks: List[Callable] = []
        self._quality_metrics = {
            'messages_received': 0,
            'errors': 0,
            'latency_ms': [],
            'staleness_alerts': 0
        }

    def subscribe(self, callback: Callable):
        """Subscribe to data updates"""
        self._callbacks.append(callback)

    def unsubscribe(self, callback: Callable):
        """Unsubscribe from data updates"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def _notify(self, data: Any):
        """Notify all subscribers"""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from data source"""
        pass

    @abstractmethod
    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbol updates"""
        pass

    @abstractmethod
    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get historical data for symbol"""
        pass

    def get_quality_score(self) -> float:
        """Calculate data quality score based on metrics"""
        if self._quality_metrics['messages_received'] == 0:
            return 0.0

        error_rate = self._quality_metrics['errors'] / self._quality_metrics['messages_received']
        avg_latency = sum(self._quality_metrics['latency_ms'][-100:]) / len(self._quality_metrics['latency_ms'][-100:]) if self._quality_metrics['latency_ms'] else 1000

        # Quality score: lower error rate and latency = higher score
        error_score = max(0, 1 - error_rate * 10)  # Penalize high error rates
        latency_score = max(0, 1 - (avg_latency / 5000))  # 5 second latency = 0 score

        return (error_score + latency_score) / 2


class RESTConnector(BaseMarketDataConnector):
    """Base class for REST API connectors"""

    def __init__(self, name: str, source_type: DataSourceType, base_url: str):
        super().__init__(name, source_type)
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 1.0

    async def connect(self) -> bool:
        try:
            self.session = aiohttp.ClientSession()
            self.status = FeedStatus.CONNECTED
            self.logger.info(f"Connected to {self.name} REST API")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.name}: {e}")
            self.status = FeedStatus.ERROR
            return False

    async def disconnect(self):
        if self.session:
            await self.session.close()
        self.status = FeedStatus.DISCONNECTED

    @abstractmethod
    async def subscribe_symbols(self, symbols: List[str]):
        """REST APIs typically don't have subscriptions - implement polling"""
        pass


class WebSocketConnector(BaseMarketDataConnector):
    """Base class for WebSocket connectors"""

    def __init__(self, name: str, source_type: DataSourceType, ws_url: str):
        super().__init__(name, source_type)
        self.ws_url = ws_url
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.subscribed_symbols: Set[str] = set()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

    async def connect(self) -> bool:
        try:
            self.status = FeedStatus.CONNECTING
            self.websocket = await websockets.connect(self.ws_url)
            self.status = FeedStatus.CONNECTED
            self.reconnect_attempts = 0
            self.logger.info(f"Connected to {self.name} WebSocket")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.name} WebSocket: {e}")
            self.status = FeedStatus.ERROR
            return False

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
        self.status = FeedStatus.DISCONNECTED

    @abstractmethod
    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to WebSocket streams"""
        pass

    async def run(self):
        """Run WebSocket message loop"""
        while self.status == FeedStatus.CONNECTED and self.websocket:
            try:
                message = await self.websocket.recv()
                await self._handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning(f"{self.name} WebSocket connection closed")
                await self._reconnect()
            except Exception as e:
                self.logger.error(f"{self.name} WebSocket error: {e}")
                self._quality_metrics['errors'] += 1

    async def _reconnect(self):
        """Attempt to reconnect WebSocket"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.status = FeedStatus.ERROR
            return

        self.reconnect_attempts += 1
        self.status = FeedStatus.RECONNECTING

        delay = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff
        await asyncio.sleep(delay)

        if await self.connect():
            # Re-subscribe to symbols
            if self.subscribed_symbols:
                await self.subscribe_symbols(list(self.subscribed_symbols))

    @abstractmethod
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message"""
        pass


# ============================================
# EXCHANGE DIRECT CONNECTORS
# ============================================

class NYSEConnector(WebSocketConnector):
    """NYSE direct feed connector - implemented with free market data APIs"""

    def __init__(self):
        # Use Alpha Vantage as primary free API, with Yahoo Finance as backup
        super().__init__("nyse", DataSourceType.EXCHANGE_DIRECT, "wss://alphavantage.co/ws")
        self.api_key = os.getenv('ALPHAVANTAGE_API_KEY', '')
        self.use_fallback = not self.api_key
        self.session = None

    async def connect(self) -> bool:
        """Connect using REST API polling since WebSocket may not be available"""
        try:
            self.status = FeedStatus.CONNECTING
            if self.use_fallback:
                self.logger.info("Using Yahoo Finance fallback for NYSE data (no API key)")
                self.status = FeedStatus.CONNECTED
                return True
            else:
                # Try Alpha Vantage WebSocket if available
                try:
                    self.session = aiohttp.ClientSession()
                    self.status = FeedStatus.CONNECTED
                    self.logger.info("Connected to Alpha Vantage for NYSE data")
                    return True
                except Exception as e:
                    self.logger.warning(f"Alpha Vantage WebSocket failed, using REST: {e}")
                    self.session = aiohttp.ClientSession()
                    self.status = FeedStatus.CONNECTED
                    return True
        except Exception as e:
            self.logger.error(f"Failed to connect to NYSE data source: {e}")
            self.status = FeedStatus.ERROR
            return False

    async def disconnect(self):
        if self.session:
            await self.session.close()
        self.status = FeedStatus.DISCONNECTED

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbols - for free APIs, we'll poll periodically"""
        # Filter to only equity symbols for NYSE (not futures)
        equity_symbols = [s for s in symbols if not self._is_futures_symbol(s)]
        if equity_symbols:
            self.subscribed_symbols.update(equity_symbols)
            self.logger.info(f"Subscribed to NYSE symbols: {equity_symbols}")

            # Start polling for subscribed symbols
            if self.status == FeedStatus.CONNECTED:
                asyncio.create_task(self._poll_market_data())

    def _is_futures_symbol(self, symbol: str) -> bool:
        """Check if symbol is a futures contract"""
        # Futures symbols typically end with =F or have month codes
        return symbol.endswith('=F') or any(month in symbol.upper() for month in ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'])

    async def _poll_market_data(self):
        """Poll market data for subscribed symbols"""
        while self.status == FeedStatus.CONNECTED and self.subscribed_symbols:
            try:
                for symbol in self.subscribed_symbols:
                    data = await self._fetch_symbol_data(symbol)
                    if data:
                        await self._notify(data)
                await asyncio.sleep(60)  # Poll every minute for free tier
            except Exception as e:
                self.logger.error(f"Error polling NYSE data: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _fetch_symbol_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch data for a single symbol"""
        try:
            if self.use_fallback:
                # Use yfinance as fallback
                return await self._fetch_yahoo_data(symbol)
            else:
                # Try Alpha Vantage first
                return await self._fetch_alphavantage_data(symbol)
        except Exception as e:
            self.logger.warning(f"Failed to fetch {symbol}: {e}")
            # Fallback to Yahoo Finance
            try:
                return await self._fetch_yahoo_data(symbol)
            except Exception as e2:
                self.logger.error(f"All data sources failed for {symbol}: {e2}")
                return None

    async def _fetch_alphavantage_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch data from Alpha Vantage"""
        if not self.session:
            return None

        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.api_key}"
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if "Global Quote" in data:
                    quote = data["Global Quote"]
                    return MarketData(
                        symbol=symbol,
                        asset_class="equity",
                        exchange="NYSE",
                        price=float(quote.get("05. price", 0)),
                        change_pct_24h=float(quote.get("10. change percent", "0%").strip('%')),
                        volume=float(quote.get("06. volume", 0)),
                        timestamp=datetime.now(),
                        source="alphavantage"
                    )
        return None

    async def _fetch_yahoo_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch data from Yahoo Finance"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")

            if not hist.empty:
                latest = hist.iloc[-1]
                return MarketData(
                    symbol=symbol,
                    asset_class="equity",
                    exchange="NYSE",
                    price=latest['Close'],
                    open=latest['Open'],
                    high=latest['High'],
                    low=latest['Low'],
                    volume=latest['Volume'],
                    timestamp=datetime.now(),
                    source="yahoo_finance"
                )
        except Exception as e:
            self.logger.warning(f"Yahoo Finance fetch failed: {e}")
        return None

    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message - not used for REST polling"""
        # This method is required by the abstract class but not used
        # since we're using REST polling instead of WebSocket
        pass

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get historical data for symbol"""
        try:
            if self.use_fallback:
                return await self._get_yahoo_historical(symbol, start_date, end_date)
            else:
                return await self._get_alphavantage_historical(symbol, start_date, end_date)
        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []


class CMEConnector(WebSocketConnector):
    """CME Group direct feed connector - implemented with free market data APIs"""

    def __init__(self):
        # Use Yahoo Finance for futures data, with fallback to other sources
        super().__init__("cme", DataSourceType.EXCHANGE_DIRECT, "wss://api.cmegroup.com/ws")
        self.session = None

    async def connect(self) -> bool:
        """Connect using REST API polling"""
        try:
            self.status = FeedStatus.CONNECTING
            self.session = aiohttp.ClientSession()
            self.status = FeedStatus.CONNECTED
            self.logger.info("Connected to CME data source (Yahoo Finance)")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to CME data source: {e}")
            self.status = FeedStatus.ERROR
            return False

    async def disconnect(self):
        if self.session:
            await self.session.close()
        self.status = FeedStatus.DISCONNECTED

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbols - for futures, we'll poll periodically"""
        # Filter to only futures symbols for CME
        futures_symbols = [s for s in symbols if self._is_futures_symbol(s)]
        if futures_symbols:
            self.subscribed_symbols.update(futures_symbols)
            self.logger.info(f"Subscribed to CME futures symbols: {futures_symbols}")

            # Start polling for subscribed symbols
            if self.status == FeedStatus.CONNECTED:
                asyncio.create_task(self._poll_futures_data())

    def _is_futures_symbol(self, symbol: str) -> bool:
        """Check if symbol is a futures contract"""
        # Futures symbols typically end with =F or have month codes
        return symbol.endswith('=F') or any(month in symbol.upper() for month in ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'])

    async def _poll_futures_data(self):
        """Poll futures data for subscribed symbols"""
        while self.status == FeedStatus.CONNECTED and self.subscribed_symbols:
            try:
                for symbol in self.subscribed_symbols:
                    data = await self._fetch_futures_data(symbol)
                    if data:
                        await self._notify(data)
                await asyncio.sleep(60)  # Poll every minute
            except Exception as e:
                self.logger.error(f"Error polling CME data: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _fetch_futures_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch futures data for a symbol"""
        try:
            import yfinance as yf

            # Convert CME futures symbols to Yahoo format
            yahoo_symbol = self._convert_cme_to_yahoo(symbol)
            if not yahoo_symbol:
                return None

            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(period="1d")

            if not hist.empty:
                latest = hist.iloc[-1]
                return MarketData(
                    symbol=symbol,
                    asset_class="future",
                    exchange="CME",
                    price=latest['Close'],
                    open=latest['Open'],
                    high=latest['High'],
                    low=latest['Low'],
                    volume=latest['Volume'],
                    timestamp=datetime.now(),
                    source="yahoo_finance"
                )
        except Exception as e:
            self.logger.warning(f"Failed to fetch CME futures data for {symbol}: {e}")
        return None

    def _convert_cme_to_yahoo(self, cme_symbol: str) -> Optional[str]:
        """Convert CME symbol format to Yahoo Finance format"""
        # Common CME futures mappings
        mappings = {
            "ES": "ES=F",    # E-mini S&P 500
            "NQ": "NQ=F",    # E-mini Nasdaq-100
            "RTY": "RTY=F",  # E-mini Russell 2000
            "CL": "CL=F",    # Crude Oil
            "GC": "GC=F",    # Gold
            "SI": "SI=F",    # Silver
            "HG": "HG=F",    # Copper
        }

        # If it's already in Yahoo format (ends with =F), return as-is
        if cme_symbol.endswith("=F"):
            return cme_symbol

        # Try to map common symbols
        base_symbol = cme_symbol.split()[0] if " " in cme_symbol else cme_symbol
        return mappings.get(base_symbol, f"{base_symbol}=F")

    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message - not used for REST polling"""
        pass

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get historical futures data"""
        try:
            import yfinance as yf

            yahoo_symbol = self._convert_cme_to_yahoo(symbol)
            if not yahoo_symbol:
                return []

            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(start=start_date, end=end_date)

            result = []
            for index, row in hist.iterrows():
                result.append(MarketData(
                    symbol=symbol,
                    asset_class="future",
                    exchange="CME",
                    price=row['Close'],
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    volume=row['Volume'],
                    timestamp=index.to_pydatetime(),
                    source="yahoo_finance"
                ))
            return result
        except Exception as e:
            self.logger.error(f"Failed to get historical CME data for {symbol}: {e}")
            return []


class NASDAQConnector(WebSocketConnector):
    """NASDAQ direct feed connector - implemented with free market data APIs"""

    def __init__(self):
        # Use Yahoo Finance as primary source for NASDAQ stocks
        super().__init__("nasdaq", DataSourceType.EXCHANGE_DIRECT, "wss://api.nasdaq.com/ws")
        self.session = None

    async def connect(self) -> bool:
        """Connect using REST API polling"""
        try:
            self.status = FeedStatus.CONNECTING
            self.session = aiohttp.ClientSession()
            self.status = FeedStatus.CONNECTED
            self.logger.info("Connected to NASDAQ data source (Yahoo Finance)")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to NASDAQ data source: {e}")
            self.status = FeedStatus.ERROR
            return False

    async def disconnect(self):
        if self.session:
            await self.session.close()
        self.status = FeedStatus.DISCONNECTED

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbols - poll periodically"""
        # Filter to only equity symbols for NASDAQ (not futures)
        equity_symbols = [s for s in symbols if not self._is_futures_symbol(s)]
        if equity_symbols:
            self.subscribed_symbols.update(equity_symbols)
            self.logger.info(f"Subscribed to NASDAQ symbols: {equity_symbols}")

            # Start polling for subscribed symbols
            if self.status == FeedStatus.CONNECTED:
                asyncio.create_task(self._poll_market_data())

    def _is_futures_symbol(self, symbol: str) -> bool:
        """Check if symbol is a futures contract"""
        # Futures symbols typically end with =F or have month codes
        return symbol.endswith('=F') or any(month in symbol.upper() for month in ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'])

    async def _poll_market_data(self):
        """Poll market data for subscribed symbols"""
        while self.status == FeedStatus.CONNECTED and self.subscribed_symbols:
            try:
                for symbol in self.subscribed_symbols:
                    data = await self._fetch_symbol_data(symbol)
                    if data:
                        await self._notify(data)
                await asyncio.sleep(60)  # Poll every minute
            except Exception as e:
                self.logger.error(f"Error polling NASDAQ data: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _fetch_symbol_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch data for a NASDAQ symbol"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")

            if not hist.empty:
                latest = hist.iloc[-1]
                return MarketData(
                    symbol=symbol,
                    asset_class="equity",
                    exchange="NASDAQ",
                    price=latest['Close'],
                    open=latest['Open'],
                    high=latest['High'],
                    low=latest['Low'],
                    volume=latest['Volume'],
                    timestamp=datetime.now(),
                    source="yahoo_finance"
                )
        except Exception as e:
            self.logger.warning(f"Failed to fetch NASDAQ data for {symbol}: {e}")
        return None

    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message - not used for REST polling"""
        pass

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get historical data for NASDAQ symbol"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)

            result = []
            for index, row in hist.iterrows():
                result.append(MarketData(
                    symbol=symbol,
                    asset_class="equity",
                    exchange="NASDAQ",
                    price=row['Close'],
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    volume=row['Volume'],
                    timestamp=index.to_pydatetime(),
                    source="yahoo_finance"
                ))
            return result
        except Exception as e:
            self.logger.error(f"Failed to get historical NASDAQ data for {symbol}: {e}")
            return []


# ============================================
# DATA VENDOR CONNECTORS
# ============================================

class BloombergConnector(RESTConnector):
    """Bloomberg Terminal API connector"""

    def __init__(self):
        super().__init__("bloomberg", DataSourceType.DATA_VENDOR, "https://api.bloomberg.com")
        # Note: Bloomberg API requires Bloomberg Terminal subscription

    async def subscribe_symbols(self, symbols: List[str]):
        # Bloomberg uses polling for real-time data
        asyncio.create_task(self._poll_symbols(symbols))

    async def _poll_symbols(self, symbols: List[str]):
        """Poll Bloomberg API for symbol updates"""
        while self.status == FeedStatus.CONNECTED:
            for symbol in symbols:
                try:
                    data = await self._get_bloomberg_data(symbol)
                    if data:
                        await self._notify(data)
                except Exception as e:
                    self.logger.error(f"Bloomberg data error for {symbol}: {e}")

            await asyncio.sleep(1)  # Poll every second

    async def _get_bloomberg_data(self, symbol: str) -> Optional[MarketData]:
        """Get data from Bloomberg API"""
        # Implementation would use Bloomberg API
        # This is a placeholder
        return None

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        # Bloomberg historical data API
        return []


class RefinitivConnector(RESTConnector):
    """Refinitiv Eikon API connector"""

    def __init__(self):
        super().__init__("refinitiv", DataSourceType.DATA_VENDOR, "https://api.refinitiv.com")
        # Note: Refinitiv API requires subscription

    async def subscribe_symbols(self, symbols: List[str]):
        # Refinitiv uses streaming API
        asyncio.create_task(self._poll_symbols(symbols))

    async def _poll_symbols(self, symbols: List[str]):
        """Poll Refinitiv API for symbol updates"""
        while self.status == FeedStatus.CONNECTED:
            for symbol in symbols:
                try:
                    data = await self._get_refinitiv_data(symbol)
                    if data:
                        await self._notify(data)
                except Exception as e:
                    self.logger.error(f"Refinitiv data error for {symbol}: {e}")

            await asyncio.sleep(1)

    async def _get_refinitiv_data(self, symbol: str) -> Optional[MarketData]:
        """Get data from Refinitiv API"""
        # Implementation would use Refinitiv API
        return None

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        # Refinitiv historical data API
        return []


class IQFeedConnector(WebSocketConnector):
    """IQFeed WebSocket connector"""

    def __init__(self):
        super().__init__("iqfeed", DataSourceType.DATA_VENDOR, "wss://iqfeed.iqfeed.net")
        # Note: IQFeed requires DTN subscription

    async def subscribe_symbols(self, symbols: List[str]):
        # IQFeed protocol implementation
        pass

    async def _handle_message(self, message: str):
        # Parse IQFeed messages
        pass

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        # IQFeed historical data
        return []


# ============================================
# PREMIUM API CONNECTORS
# ============================================

class PolygonConnector(RESTConnector):
    """Polygon.io API connector - Excellent free tier + premium options"""

    def __init__(self):
        super().__init__("polygon", DataSourceType.DATA_VENDOR, "https://api.polygon.io")
        self.api_key = os.getenv('POLYGON_API_KEY', '')

    async def connect(self) -> bool:
        """Connect to Polygon API"""
        if not self.api_key:
            self.logger.warning("Polygon API key not configured - using free tier limits")
            return True  # Can still use free tier

        try:
            self.status = FeedStatus.CONNECTING
            self.session = aiohttp.ClientSession()
            self.status = FeedStatus.CONNECTED
            self.logger.info("Connected to Polygon.io API")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Polygon API: {e}")
            self.status = FeedStatus.ERROR
            return False

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbols - Polygon uses REST polling"""
        asyncio.create_task(self._poll_symbols(symbols))

    async def _poll_symbols(self, symbols: List[str]):
        """Poll Polygon API for symbol updates"""
        while self.status == FeedStatus.CONNECTED:
            for symbol in symbols:
                try:
                    data = await self._get_polygon_data(symbol)
                    if data:
                        await self._notify(data)
                except Exception as e:
                    self.logger.error(f"Polygon data error for {symbol}: {e}")

            # Respect rate limits: 5 calls/minute free, much higher paid
            await asyncio.sleep(12)  # Poll every 12 seconds to stay under free limit

    async def _get_polygon_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time data from Polygon"""
        if not self.session:
            return None

        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={self.api_key}"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('results'):
                        result = data['results'][0]
                        return MarketData(
                            symbol=symbol,
                            asset_class="equity",
                            exchange="NYSE",  # Polygon aggregates across exchanges
                            price=result.get('c', 0),  # Close price
                            open=result.get('o', 0),
                            high=result.get('h', 0),
                            low=result.get('l', 0),
                            volume=result.get('v', 0),
                            timestamp=datetime.fromtimestamp(result.get('t', 0) / 1000),
                            source="polygon"
                        )
        except Exception as e:
            self.logger.warning(f"Polygon API error: {e}")
        return None

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get historical bars from Polygon"""
        if not self.session:
            return []

        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_ms}/{end_ms}?apiKey={self.api_key}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    for result in data.get('results', []):
                        results.append(MarketData(
                            symbol=symbol,
                            asset_class="equity",
                            exchange="NYSE",
                            price=result.get('c', 0),
                            open=result.get('o', 0),
                            high=result.get('h', 0),
                            low=result.get('l', 0),
                            volume=result.get('v', 0),
                            timestamp=datetime.fromtimestamp(result.get('t', 0) / 1000),
                            source="polygon"
                        ))
                    return results
        except Exception as e:
            self.logger.error(f"Polygon historical data error: {e}")
        return []


class FinnhubConnector(RESTConnector):
    """Finnhub API connector - Good free tier for stocks"""

    def __init__(self):
        super().__init__("finnhub", DataSourceType.DATA_VENDOR, "https://finnhub.io/api/v1")
        self.api_key = os.getenv('FINNHUB_API_KEY', '')

    async def connect(self) -> bool:
        """Connect to Finnhub API"""
        if not self.api_key:
            self.logger.warning("Finnhub API key not configured - limited functionality")
            return True

        try:
            self.status = FeedStatus.CONNECTING
            self.session = aiohttp.ClientSession()
            self.status = FeedStatus.CONNECTED
            self.logger.info("Connected to Finnhub API")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Finnhub API: {e}")
            self.status = FeedStatus.ERROR
            return False

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbols - Finnhub uses REST polling"""
        asyncio.create_task(self._poll_symbols(symbols))

    async def _poll_symbols(self, symbols: List[str]):
        """Poll Finnhub API for symbol updates"""
        while self.status == FeedStatus.CONNECTED:
            for symbol in symbols:
                try:
                    data = await self._get_finnhub_data(symbol)
                    if data:
                        await self._notify(data)
                except Exception as e:
                    self.logger.error(f"Finnhub data error for {symbol}: {e}")

            # Respect rate limits: 60 calls/minute free
            await asyncio.sleep(1)

    async def _get_finnhub_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time quote from Finnhub"""
        if not self.session or not self.api_key:
            return None

        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.api_key}"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('c', 0) > 0:  # Current price > 0
                        return MarketData(
                            symbol=symbol,
                            asset_class="equity",
                            exchange="NYSE",  # Finnhub aggregates
                            price=data.get('c', 0),  # Current price
                            open=data.get('o', 0),
                            high=data.get('h', 0),
                            low=data.get('l', 0),
                            timestamp=datetime.now(),
                            source="finnhub"
                        )
        except Exception as e:
            self.logger.warning(f"Finnhub API error: {e}")
        return None

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get historical data from Finnhub (limited free tier)"""
        # Finnhub free tier has limited historical data
        return []


class IEXCloudConnector(RESTConnector):
    """IEX Cloud API connector - Good for US stocks"""

    def __init__(self):
        super().__init__("iex_cloud", DataSourceType.DATA_VENDOR, "https://cloud.iexapis.com")
        self.api_key = os.getenv('IEX_CLOUD_API_KEY', '')

    async def connect(self) -> bool:
        """Connect to IEX Cloud API"""
        if not self.api_key:
            self.logger.warning("IEX Cloud API key not configured - limited functionality")
            return True

        try:
            self.status = FeedStatus.CONNECTING
            self.session = aiohttp.ClientSession()
            self.status = FeedStatus.CONNECTED
            self.logger.info("Connected to IEX Cloud API")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to IEX Cloud API: {e}")
            self.status = FeedStatus.ERROR
            return False

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbols - IEX uses REST polling"""
        asyncio.create_task(self._poll_symbols(symbols))

    async def _poll_symbols(self, symbols: List[str]):
        """Poll IEX Cloud API for symbol updates"""
        while self.status == FeedStatus.CONNECTED:
            for symbol in symbols:
                try:
                    data = await self._get_iex_data(symbol)
                    if data:
                        await self._notify(data)
                except Exception as e:
                    self.logger.error(f"IEX Cloud data error for {symbol}: {e}")

            # Respect rate limits: 50,000 calls/month free
            await asyncio.sleep(1)

    async def _get_iex_data(self, symbol: str) -> Optional[MarketData]:
        """Get quote from IEX Cloud"""
        if not self.session or not self.api_key:
            return None

        url = f"https://cloud.iexapis.com/stable/stock/{symbol}/quote?token={self.api_key}"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return MarketData(
                        symbol=symbol,
                        asset_class="equity",
                        exchange="NYSE",  # IEX aggregates
                        price=data.get('latestPrice', 0),
                        bid=data.get('iexBidPrice', 0),
                        ask=data.get('iexAskPrice', 0),
                        volume=data.get('volume', 0),
                        change_pct_24h=data.get('changePercent', 0),
                        timestamp=datetime.now(),
                        source="iex_cloud"
                    )
        except Exception as e:
            self.logger.warning(f"IEX Cloud API error: {e}")
        return None

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get historical data from IEX Cloud"""
        if not self.session or not self.api_key:
            return []

        # IEX Cloud has good historical data support
        url = f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/max?token={self.api_key}"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    for item in data:
                        # Filter by date range
                        item_date = datetime.strptime(item['date'], '%Y-%m-%d')
                        if start_date <= item_date <= end_date:
                            results.append(MarketData(
                                symbol=symbol,
                                asset_class="equity",
                                exchange="NYSE",
                                price=item.get('close', 0),
                                open=item.get('open', 0),
                                high=item.get('high', 0),
                                low=item.get('low', 0),
                                volume=item.get('volume', 0),
                                timestamp=item_date,
                                source="iex_cloud"
                            ))
                    return results
        except Exception as e:
            self.logger.error(f"IEX Cloud historical data error: {e}")
        return []


class TwelveDataConnector(RESTConnector):
    """Twelve Data API connector - Comprehensive market data"""

    def __init__(self):
        super().__init__("twelve_data", DataSourceType.DATA_VENDOR, "https://api.twelvedata.com")
        self.api_key = os.getenv('TWELVE_DATA_API_KEY', '')

    async def connect(self) -> bool:
        """Connect to Twelve Data API"""
        if not self.api_key:
            self.logger.warning("Twelve Data API key not configured - limited functionality")
            return True

        try:
            self.status = FeedStatus.CONNECTING
            self.session = aiohttp.ClientSession()
            self.status = FeedStatus.CONNECTED
            self.logger.info("Connected to Twelve Data API")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Twelve Data API: {e}")
            self.status = FeedStatus.ERROR
            return False

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbols - Twelve Data uses REST polling"""
        asyncio.create_task(self._poll_symbols(symbols))

    async def _poll_symbols(self, symbols: List[str]):
        """Poll Twelve Data API for symbol updates"""
        while self.status == FeedStatus.CONNECTED:
            # Twelve Data supports batch requests
            try:
                data_list = await self._get_twelve_data_batch(symbols)
                for data in data_list:
                    if data:
                        await self._notify(data)
            except Exception as e:
                self.logger.error(f"Twelve Data batch error: {e}")

            # Respect rate limits: 800 calls/day free
            await asyncio.sleep(30)  # Poll every 30 seconds

    async def _get_twelve_data_batch(self, symbols: List[str]) -> List[Optional[MarketData]]:
        """Get batch quotes from Twelve Data"""
        if not self.session or not self.api_key:
            return []

        symbol_list = ','.join(symbols[:8])  # Max 8 symbols per request
        url = f"https://api.twelvedata.com/quote?symbol={symbol_list}&apikey={self.api_key}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    for symbol in symbols[:8]:
                        symbol_data = data.get(symbol, {})
                        if symbol_data and symbol_data.get('close'):
                            results.append(MarketData(
                                symbol=symbol,
                                asset_class="equity",
                                exchange=symbol_data.get('exchange', 'NYSE'),
                                price=float(symbol_data.get('close', 0)),
                                open=float(symbol_data.get('open', 0)),
                                high=float(symbol_data.get('high', 0)),
                                low=float(symbol_data.get('low', 0)),
                                volume=int(symbol_data.get('volume', 0)),
                                change_pct_24h=float(symbol_data.get('percent_change', 0)),
                                timestamp=datetime.now(),
                                source="twelve_data"
                            ))
                        else:
                            results.append(None)
                    return results
        except Exception as e:
            self.logger.warning(f"Twelve Data API error: {e}")
        return [None] * len(symbols[:8])

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get historical data from Twelve Data"""
        if not self.session or not self.api_key:
            return []

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&start_date={start_str}&end_date={end_str}&apikey={self.api_key}"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    for item in data.get('values', []):
                        results.append(MarketData(
                            symbol=symbol,
                            asset_class="equity",
                            exchange="NYSE",
                            price=float(item.get('close', 0)),
                            open=float(item.get('open', 0)),
                            high=float(item.get('high', 0)),
                            low=float(item.get('low', 0)),
                            volume=int(item.get('volume', 0)),
                            timestamp=datetime.strptime(item['datetime'], '%Y-%m-%d'),
                            source="twelve_data"
                        ))
                    return results
        except Exception as e:
            self.logger.error(f"Twelve Data historical data error: {e}")
        return []


class IntrinioConnector(RESTConnector):
    """Intrinio API connector - Institutional-grade data"""

    def __init__(self):
        super().__init__("intrinio", DataSourceType.DATA_VENDOR, "https://api-v2.intrinio.com")
        self.api_key = os.getenv('INTRINIO_API_KEY', '')
        self.username = os.getenv('INTRINIO_USERNAME', '')

    async def connect(self) -> bool:
        """Connect to Intrinio API"""
        if not self.api_key or not self.username:
            self.logger.warning("Intrinio credentials not configured - premium service")
            return False

        try:
            self.status = FeedStatus.CONNECTING
            self.session = aiohttp.ClientSession()
            self.status = FeedStatus.CONNECTED
            self.logger.info("Connected to Intrinio API")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Intrinio API: {e}")
            self.status = FeedStatus.ERROR
            return False

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to symbols - Intrinio uses REST polling"""
        asyncio.create_task(self._poll_symbols(symbols))

    async def _poll_symbols(self, symbols: List[str]):
        """Poll Intrinio API for symbol updates"""
        while self.status == FeedStatus.CONNECTED:
            for symbol in symbols:
                try:
                    data = await self._get_intrinio_data(symbol)
                    if data:
                        await self._notify(data)
                except Exception as e:
                    self.logger.error(f"Intrinio data error for {symbol}: {e}")

            # Intrinio has generous rate limits for paid plans
            await asyncio.sleep(1)

    async def _get_intrinio_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time data from Intrinio"""
        if not self.session:
            return None

        # Intrinio uses basic auth
        auth = aiohttp.BasicAuth(self.username, self.api_key)
        url = f"https://api-v2.intrinio.com/securities/{symbol}/prices/realtime"

        try:
            async with self.session.get(url, auth=auth) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('last_price'):
                        return MarketData(
                            symbol=symbol,
                            asset_class="equity",
                            exchange=data.get('source', 'NYSE'),
                            price=data.get('last_price', 0),
                            bid=data.get('bid_price', 0),
                            ask=data.get('ask_price', 0),
                            volume=data.get('volume', 0),
                            timestamp=datetime.now(),
                            source="intrinio"
                        )
        except Exception as e:
            self.logger.warning(f"Intrinio API error: {e}")
        return None

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get historical data from Intrinio"""
        if not self.session:
            return []

        auth = aiohttp.BasicAuth(self.username, self.api_key)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        url = f"https://api-v2.intrinio.com/securities/{symbol}/prices?start_date={start_str}&end_date={end_str}&frequency=daily"

        try:
            async with self.session.get(url, auth=auth) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    for item in data.get('stock_prices', []):
                        results.append(MarketData(
                            symbol=symbol,
                            asset_class="equity",
                            exchange="NYSE",
                            price=item.get('close', 0),
                            open=item.get('open', 0),
                            high=item.get('high', 0),
                            low=item.get('low', 0),
                            volume=item.get('volume', 0),
                            timestamp=datetime.strptime(item['date'], '%Y-%m-%d'),
                            source="intrinio"
                        ))
                    return results
        except Exception as e:
            self.logger.error(f"Intrinio historical data error: {e}")
        return []


# ============================================
# CRYPTO EXCHANGE CONNECTORS
# ============================================

class BinanceConnector(WebSocketConnector):
    """Binance WebSocket connector"""

    def __init__(self):
        super().__init__("binance", DataSourceType.CRYPTO_EXCHANGE, "wss://stream.binance.com:9443/ws")

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to Binance streams"""
        # Convert symbols to Binance format
        streams = []
        for symbol in symbols:
            if "/" in symbol:
                base, quote = symbol.split("/")
                binance_symbol = f"{base.lower()}{quote.lower()}@ticker"
                streams.append(binance_symbol)

        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }

        if self.websocket:
            await self.websocket.send(json.dumps(subscribe_msg))
            self.subscribed_symbols.update(symbols)

    async def _handle_message(self, message: str):
        """Handle Binance WebSocket messages"""
        try:
            data = json.loads(message)

            if 'stream' in data and data['stream'].endswith('@ticker'):
                ticker_data = data['data']
                symbol = self._parse_symbol(ticker_data['s'])

                market_data = MarketData(
                    symbol=symbol,
                    asset_class="crypto",
                    exchange="binance",
                    price=float(ticker_data['c']),
                    bid=float(ticker_data['b']),
                    ask=float(ticker_data['a']),
                    volume=float(ticker_data['v']),
                    volume_24h=float(ticker_data['v']),
                    change_24h=float(ticker_data['P']),
                    change_pct_24h=float(ticker_data['P']),
                    high=float(ticker_data['h']),
                    low=float(ticker_data['l']),
                    source="binance",
                    timestamp=datetime.fromtimestamp(ticker_data['E'] / 1000)
                )

                await self._notify(market_data)

        except Exception as e:
            self.logger.error(f"Binance message parse error: {e}")

    def _parse_symbol(self, binance_symbol: str) -> str:
        """Convert Binance symbol to standard format"""
        if binance_symbol.endswith('USDT'):
            base = binance_symbol[:-4]
            return f"{base}/USDT"
        elif binance_symbol.endswith('BTC'):
            base = binance_symbol[:-3]
            return f"{base}/BTC"
        return binance_symbol

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get Binance historical data via REST API"""
        # Implementation would use Binance REST API
        return []


class CoinbaseProConnector(WebSocketConnector):
    """Coinbase Pro WebSocket connector"""

    def __init__(self):
        super().__init__("coinbase_pro", DataSourceType.CRYPTO_EXCHANGE, "wss://ws-feed.exchange.coinbase.com")

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to Coinbase Pro products"""
        # Convert to Coinbase format
        product_ids = [s.replace("/", "-") for s in symbols]

        subscribe_msg = {
            "type": "subscribe",
            "product_ids": product_ids,
            "channels": ["ticker", "level2"]
        }

        if self.websocket:
            await self.websocket.send(json.dumps(subscribe_msg))
            self.subscribed_symbols.update(symbols)

    async def _handle_message(self, message: str):
        """Handle Coinbase Pro messages"""
        try:
            data = json.loads(message)

            if data.get('type') == 'ticker':
                symbol = data['product_id'].replace("-", "/")

                market_data = MarketData(
                    symbol=symbol,
                    asset_class="crypto",
                    exchange="coinbase_pro",
                    price=float(data['price']),
                    bid=float(data.get('best_bid', 0)),
                    ask=float(data.get('best_ask', 0)),
                    volume=float(data.get('volume_24h', 0)),
                    volume_24h=float(data.get('volume_24h', 0)),
                    source="coinbase_pro",
                    timestamp=datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
                )

                await self._notify(market_data)

        except Exception as e:
            self.logger.error(f"Coinbase Pro message parse error: {e}")

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get Coinbase Pro historical data"""
        return []


# ============================================
# NAV CALCULATION ENGINE
# ============================================

class NAVCalculator:
    """Real-time ETF NAV calculation engine"""

    def __init__(self):
        self.logger = logging.getLogger("NAVCalculator")
        self.holdings_cache: Dict[str, List[ETFHolding]] = {}
        self.nav_cache: Dict[str, ETFNAVData] = {}
        self.price_feeds: Dict[str, BaseMarketDataConnector] = {}

    def add_price_feed(self, feed: BaseMarketDataConnector):
        """Add price feed for NAV calculations"""
        self.price_feeds[feed.name] = feed
        feed.subscribe(self._on_price_update)

    async def _on_price_update(self, data: MarketData):
        """Handle price updates for NAV calculation"""
        # Update holdings prices and recalculate NAV
        await self._update_nav_for_symbol(data.symbol)

    async def load_etf_holdings(self, etf_symbol: str) -> List[ETFHolding]:
        """Load ETF holdings data"""
        # This would typically load from a data provider or file
        # For now, return empty list as placeholder
        if etf_symbol not in self.holdings_cache:
            self.holdings_cache[etf_symbol] = []

        return self.holdings_cache[etf_symbol]

    async def calculate_nav(self, etf_symbol: str) -> Optional[ETFNAVData]:
        """Calculate real-time NAV for ETF"""
        holdings = await self.load_etf_holdings(etf_symbol)

        if not holdings:
            return None

        total_value = 0.0
        total_weight = 0.0

        for holding in holdings:
            if holding.price:
                holding.market_value = holding.shares * holding.price
                total_value += holding.market_value
                total_weight += holding.weight

        if total_weight == 0:
            return None

        # Get ETF market price
        market_price = await self._get_market_price(etf_symbol)

        if not market_price:
            return None

        nav_price = total_value / 1000000  # Assuming shares outstanding normalization
        premium_discount = ((market_price - nav_price) / nav_price) * 100

        nav_data = ETFNAVData(
            symbol=etf_symbol,
            nav_price=nav_price,
            market_price=market_price,
            premium_discount=premium_discount,
            holdings=holdings,
            total_assets=total_value,
            shares_outstanding=1000000,  # Placeholder
            timestamp=datetime.now(),
            source="calculated"
        )

        self.nav_cache[etf_symbol] = nav_data
        return nav_data

    async def _get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        for feed in self.price_feeds.values():
            # Try to get price from feeds
            # This is a simplified implementation
            pass
        return None

    async def _update_nav_for_symbol(self, symbol: str):
        """Update NAV for ETFs that hold this symbol"""
        # Find ETFs that hold this symbol and recalculate their NAV
        for etf_symbol, holdings in self.holdings_cache.items():
            if any(h.symbol == symbol for h in holdings):
                await self.calculate_nav(etf_symbol)


# ============================================
# MASTER MARKET DATA MANAGER
# ============================================

class MarketDataManager:
    """
    Master market data manager coordinating 100+ feeds with redundancy and failover
    """

    def __init__(self):
        self.logger = logging.getLogger("MarketDataManager")
        self.connectors: Dict[str, BaseMarketDataConnector] = {}
        self.active_feeds: Dict[str, List[BaseMarketDataConnector]] = {}  # symbol -> feeds
        self.data_cache: Dict[str, MarketData] = {}  # Latest data by symbol
        self.nav_calculator = NAVCalculator()
        self._running = False
        self._tasks: List[asyncio.Task] = []

    def add_connector(self, connector: BaseMarketDataConnector):
        """Add a market data connector"""
        self.connectors[connector.name] = connector
        self.nav_calculator.add_price_feed(connector)

        # Subscribe to connector updates
        connector.subscribe(self._on_data_update)

        audit_log("market_data", "connector_added", {
            "connector": connector.name,
            "type": connector.source_type.value
        })

    async def initialize_connectors(self):
        """Initialize and connect all connectors"""
        # Primary feeds (highest priority)
        primary_connectors = [
            NYSEConnector(),
            CMEConnector(),
            NASDAQConnector(),
        ]

        # Alternative feeds (redundancy)
        alternative_connectors = [
            BloombergConnector(),
            RefinitivConnector(),
            IQFeedConnector(),
            # Premium API connectors
            PolygonConnector(),
            FinnhubConnector(),
            IEXCloudConnector(),
            TwelveDataConnector(),
            IntrinioConnector(),
        ]

        # Crypto feeds
        crypto_connectors = [
            BinanceConnector(),
            CoinbaseProConnector(),
        ]

        # Add more connectors from the research...
        # This is where we'd instantiate all 100+ connectors

        all_connectors = primary_connectors + alternative_connectors + crypto_connectors

        for connector in all_connectors:
            self.add_connector(connector)

    async def subscribe_symbols(self, symbols: List[str], required_sources: List[DataSourceType] = None):
        """Subscribe to symbols across multiple feeds for redundancy"""
        if required_sources is None:
            required_sources = [DataSourceType.EXCHANGE_DIRECT, DataSourceType.DATA_VENDOR, DataSourceType.CRYPTO_EXCHANGE]

        for symbol in symbols:
            self.active_feeds[symbol] = []

            # Find suitable connectors for this symbol
            for connector in self.connectors.values():
                if connector.source_type in required_sources:
                    # Check if connector supports this symbol type
                    if self._connector_supports_symbol(connector, symbol):
                        self.active_feeds[symbol].append(connector)

                        # Connect and subscribe
                        if connector.status != FeedStatus.CONNECTED:
                            await connector.connect()

                        await connector.subscribe_symbols([symbol])

            self.logger.info(f"Subscribed {symbol} to {len(self.active_feeds[symbol])} feeds")

    def _connector_supports_symbol(self, connector: BaseMarketDataConnector, symbol: str) -> bool:
        """Check if connector supports this symbol"""
        # Simple logic - in production, this would be more sophisticated
        if connector.source_type == DataSourceType.CRYPTO_EXCHANGE:
            return "/" in symbol  # Crypto symbols have /
        elif connector.source_type in [DataSourceType.EXCHANGE_DIRECT, DataSourceType.DATA_VENDOR]:
            return "/" not in symbol or symbol.split("/")[1] in ["USD", "EUR", "GBP"]
        return True

    async def _on_data_update(self, data: MarketData):
        """Handle incoming market data"""
        # Update cache with latest data
        self.data_cache[data.symbol] = data

        # Check for arbitrage opportunities
        await self._check_arbitrage_opportunities(data.symbol)

        # Update NAV calculations
        if data.asset_class == "equity":
            await self.nav_calculator._update_nav_for_symbol(data.symbol)

    async def _check_arbitrage_opportunities(self, symbol: str):
        """Check for arbitrage opportunities across feeds"""
        # Get prices from different sources
        prices = {}
        for connector_name, connector in self.connectors.items():
            # This would aggregate prices from different feeds
            pass

        # Calculate spreads and alert if profitable
        # Implementation would check for price discrepancies

    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for symbol"""
        return self.data_cache.get(symbol)

    def get_nav_data(self, etf_symbol: str) -> Optional[ETFNAVData]:
        """Get latest NAV data for ETF"""
        return self.nav_calculator.nav_cache.get(etf_symbol)

    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime,
                                sources: List[str] = None) -> List[MarketData]:
        """Get historical data from multiple sources"""
        if sources is None:
            sources = list(self.connectors.keys())

        all_data = []
        for source_name in sources:
            if source_name in self.connectors:
                connector = self.connectors[source_name]
                try:
                    data = await connector.get_historical_data(symbol, start_date, end_date)
                    all_data.extend(data)
                except Exception as e:
                    self.logger.error(f"Failed to get historical data from {source_name}: {e}")

        # Sort by timestamp and remove duplicates
        all_data.sort(key=lambda x: x.timestamp)
        return all_data

    async def start(self):
        """Start all connectors and monitoring"""
        self._running = True

        # Start WebSocket connectors
        for connector in self.connectors.values():
            if isinstance(connector, WebSocketConnector):
                task = asyncio.create_task(connector.run())
                self._tasks.append(task)

        self.logger.info(f"Started market data manager with {len(self.connectors)} connectors")

    async def stop(self):
        """Stop all connectors"""
        self._running = False

        for connector in self.connectors.values():
            await connector.disconnect()

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        self.logger.info("Market data manager stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all connectors"""
        return {
            name: {
                "status": connector.status.value,
                "type": connector.source_type.value,
                "quality_score": connector.get_quality_score(),
                "last_update": connector.last_update.isoformat() if connector.last_update else None,
            }
            for name, connector in self.connectors.items()
        }

    def get_feed_coverage(self, symbol: str) -> Dict[str, Any]:
        """Get feed coverage for a symbol"""
        feeds = self.active_feeds.get(symbol, [])
        return {
            "symbol": symbol,
            "active_feeds": len(feeds),
            "feed_types": list(set(f.source_type.value for f in feeds)),
            "feed_names": [f.name for f in feeds],
        }


# Global market data manager instance
market_data_manager = MarketDataManager()


async def initialize_market_data_system():
    """Initialize the complete market data system"""
    await market_data_manager.initialize_connectors()

    # Example subscriptions for arbitrage strategies
    arbitrage_symbols = [
        "SPY", "QQQ", "IWM",  # ETFs
        "ES=F", "NQ=F", "RTY=F",  # Futures
        "BTC/USDT", "ETH/USDT",  # Crypto
    ]

    await market_data_manager.subscribe_symbols(arbitrage_symbols)
    await market_data_manager.start()

    audit_log("market_data", "system_initialized", {
        "connectors": len(market_data_manager.connectors),
        "symbols": len(arbitrage_symbols)
    })


if __name__ == "__main__":
    # Example usage
    asyncio.run(initialize_market_data_system())