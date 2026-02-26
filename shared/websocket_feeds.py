#!/usr/bin/env python3
"""
WebSocket Price Feed Manager
============================
Real-time price feeds via WebSocket connections to exchanges.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class FeedState(Enum):
    """WebSocket feed connection state"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class PriceTick:
    """Real-time price update"""
    symbol: str
    exchange: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        return (self.spread / self.bid) * 100 if self.bid > 0 else 0


@dataclass
class OrderBookUpdate:
    """Real-time order book update"""
    symbol: str
    exchange: str
    bids: List[tuple]  # [(price, quantity), ...]
    asks: List[tuple]
    timestamp: datetime = field(default_factory=datetime.now)
    is_snapshot: bool = False


class ReconnectionPolicy:
    """Exponential backoff reconnection policy"""
    
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        max_attempts: int = 10,
        jitter: float = 0.1,
    ):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.max_attempts = max_attempts
        self.jitter = jitter
        self._attempt = 0
        
    def reset(self):
        """Reset attempt counter"""
        self._attempt = 0
    
    def next_delay(self) -> Optional[float]:
        """
        Get next reconnection delay.
        Returns None if max attempts exceeded.
        """
        if self._attempt >= self.max_attempts:
            return None
        
        import random
        delay = min(
            self.initial_delay * (self.multiplier ** self._attempt),
            self.max_delay
        )
        
        # Add jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        
        self._attempt += 1
        return max(0.1, delay)
    
    @property
    def attempts(self) -> int:
        return self._attempt


PriceCallback = Callable[[PriceTick], None]
OrderBookCallback = Callable[[OrderBookUpdate], None]


class BaseWebSocketFeed(ABC):
    """Abstract base class for exchange WebSocket feeds"""
    
    def __init__(
        self,
        exchange: str,
        reconnect_policy: Optional[ReconnectionPolicy] = None,
        staleness_threshold_seconds: float = 30.0,
    ):
        self.exchange = exchange
        self.logger = logging.getLogger(f"{exchange}WebSocketFeed")
        self.reconnect_policy = reconnect_policy or ReconnectionPolicy()
        self.staleness_threshold_seconds = staleness_threshold_seconds
        
        self._state = FeedState.DISCONNECTED
        self._ws = None
        self._subscribed_symbols: Set[str] = set()
        self._price_callbacks: List[PriceCallback] = []
        self._orderbook_callbacks: List[OrderBookCallback] = []
        self._running = False
        self._message_count = 0
        self._last_message_time: Optional[datetime] = None
        
        # Staleness tracking callbacks
        self._staleness_callbacks: List[Callable[[str, float], None]] = []
    
    @property
    def state(self) -> FeedState:
        return self._state
    
    @property
    def is_connected(self) -> bool:
        return self._state == FeedState.CONNECTED
    
    @property
    def is_stale(self) -> bool:
        """Check if feed has gone stale (no messages for threshold period)"""
        if self._last_message_time is None:
            return self._state == FeedState.CONNECTED  # Connected but no messages yet
        age = (datetime.now() - self._last_message_time).total_seconds()
        return age > self.staleness_threshold_seconds
    
    @property
    def time_since_last_message(self) -> Optional[float]:
        """Get seconds since last message, or None if never received"""
        if self._last_message_time is None:
            return None
        return (datetime.now() - self._last_message_time).total_seconds()
    
    def on_staleness(self, callback: Callable[[str, float], None]):
        """Register callback for when feed becomes stale. Args: (exchange, seconds_since_last)"""
        self._staleness_callbacks.append(callback)
    
    def _emit_staleness_alert(self):
        """Emit staleness alert to all registered callbacks"""
        if self._last_message_time:
            age = (datetime.now() - self._last_message_time).total_seconds()
            for callback in self._staleness_callbacks:
                try:
                    callback(self.exchange, age)
                except Exception as e:
                    self.logger.error(f"Staleness callback error: {e}")
    
    @property
    def state(self) -> FeedState:
        return self._state
    
    @property
    def is_connected(self) -> bool:
        return self._state == FeedState.CONNECTED
    
    @property
    @abstractmethod
    def websocket_url(self) -> str:
        """WebSocket endpoint URL"""
        pass
    
    def on_price(self, callback: PriceCallback):
        """Register price update callback"""
        self._price_callbacks.append(callback)
    
    def on_orderbook(self, callback: OrderBookCallback):
        """Register order book update callback"""
        self._orderbook_callbacks.append(callback)
    
    def _emit_price(self, tick: PriceTick):
        """Emit price update to all callbacks"""
        for callback in self._price_callbacks:
            try:
                callback(tick)
            except Exception as e:
                self.logger.error(f"Price callback error: {e}")
    
    def _emit_orderbook(self, update: OrderBookUpdate):
        """Emit order book update to all callbacks"""
        for callback in self._orderbook_callbacks:
            try:
                callback(update)
            except Exception as e:
                self.logger.error(f"OrderBook callback error: {e}")
    
    @abstractmethod
    async def _connect(self) -> bool:
        """Establish WebSocket connection"""
        pass
    
    @abstractmethod
    async def _subscribe(self, symbols: List[str]):
        """Subscribe to symbols"""
        pass
    
    @abstractmethod
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message"""
        pass
    
    async def connect(self, symbols: List[str]) -> bool:
        """Connect and subscribe to symbols"""
        self._state = FeedState.CONNECTING
        self._subscribed_symbols = set(symbols)
        
        try:
            if await self._connect():
                await self._subscribe(symbols)
                self._state = FeedState.CONNECTED
                self.reconnect_policy.reset()
                self.logger.info(f"Connected to {self.exchange} WebSocket")
                return True
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self._state = FeedState.ERROR
        
        return False
    
    async def disconnect(self):
        """Disconnect WebSocket"""
        self._running = False
        self._state = FeedState.DISCONNECTED
        
        if self._ws:
            await self._ws.close()
            self._ws = None
        
        self.logger.info(f"Disconnected from {self.exchange} WebSocket")
    
    async def run(self):
        """Run the WebSocket feed with auto-reconnection and staleness monitoring"""
        self._running = True
        staleness_check_interval = 5.0  # Check for staleness every 5 seconds
        last_staleness_check = datetime.now()
        was_stale = False
        
        while self._running:
            try:
                # Periodic staleness check
                now = datetime.now()
                if (now - last_staleness_check).total_seconds() > staleness_check_interval:
                    last_staleness_check = now
                    if self.is_stale and not was_stale:
                        self.logger.warning(f"{self.exchange} feed is stale - no messages for {self.time_since_last_message:.1f}s")
                        self._emit_staleness_alert()
                        was_stale = True
                    elif not self.is_stale and was_stale:
                        self.logger.info(f"{self.exchange} feed recovered from staleness")
                        was_stale = False
                
                if self._state == FeedState.CONNECTED and self._ws:
                    # Use timeout to allow periodic staleness checks
                    try:
                        import asyncio
                        message = await asyncio.wait_for(
                            self._ws.__anext__(),
                            timeout=staleness_check_interval
                        )
                        if not self._running:
                            break
                        
                        self._message_count += 1
                        self._last_message_time = datetime.now()
                        was_stale = False
                        
                        await self._handle_message(message.data)
                    except asyncio.TimeoutError:
                        # No message received, continue to staleness check
                        continue
                    except StopAsyncIteration:
                        # WebSocket closed
                        self._state = FeedState.ERROR
                        await self._reconnect()
                else:
                    # Not connected, try to reconnect
                    await self._reconnect()
                    
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                self._state = FeedState.ERROR
                await self._reconnect()
    
    async def _reconnect(self):
        """Attempt reconnection with backoff"""
        if not self._running:
            return
        
        self._state = FeedState.RECONNECTING
        delay = self.reconnect_policy.next_delay()
        
        if delay is None:
            self.logger.error(f"Max reconnection attempts ({self.reconnect_policy.max_attempts}) exceeded")
            self._state = FeedState.ERROR
            self._running = False
            return
        
        self.logger.info(f"Reconnecting in {delay:.1f}s (attempt {self.reconnect_policy.attempts})")
        await asyncio.sleep(delay)
        
        if self._running:
            await self.connect(list(self._subscribed_symbols))


class BinanceWebSocketFeed(BaseWebSocketFeed):
    """Binance WebSocket price feed"""
    
    def __init__(self, testnet: bool = True, **kwargs):
        super().__init__(exchange="binance", **kwargs)
        self.testnet = testnet
        self._session = None  # Track aiohttp session for cleanup
    
    @property
    def websocket_url(self) -> str:
        if self.testnet:
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"
    
    async def _connect(self) -> bool:
        try:
            import aiohttp
            # Clean up any existing session first
            if self._session and not self._session.closed:
                await self._session.close()
            
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(self.websocket_url)
            return True
        except ImportError:
            self.logger.error("aiohttp not installed")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            # Cleanup on failure
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None
            return False
    
    async def disconnect(self):
        """Disconnect WebSocket and cleanup session"""
        self._running = False
        self._state = FeedState.DISCONNECTED
        
        # Close WebSocket first
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                self.logger.debug(f"Error closing WebSocket: {e}")
            self._ws = None
        
        # Then close the aiohttp session to prevent memory leaks
        if self._session and not self._session.closed:
            try:
                await self._session.close()
            except Exception as e:
                self.logger.debug(f"Error closing session: {e}")
            self._session = None
        
        self.logger.info(f"Disconnected from {self.exchange} WebSocket")
    
    async def _subscribe(self, symbols: List[str]):
        """Subscribe to ticker streams"""
        # Convert symbols to Binance format (BTC/USDT -> btcusdt)
        streams = []
        for symbol in symbols:
            binance_symbol = symbol.replace("/", "").lower()
            streams.append(f"{binance_symbol}@ticker")
            streams.append(f"{binance_symbol}@depth5@100ms")
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }
        
        await self._ws.send_str(json.dumps(subscribe_msg))
        self.logger.info(f"Subscribed to {len(symbols)} symbols")
    
    async def _handle_message(self, message: str):
        """Parse Binance WebSocket message"""
        try:
            data = json.loads(message)
            
            # Skip subscription confirmations
            if 'result' in data:
                return
            
            event_type = data.get('e')
            
            if event_type == '24hrTicker':
                # Ticker update
                symbol = self._parse_symbol(data.get('s', ''))
                tick = PriceTick(
                    symbol=symbol,
                    exchange=self.exchange,
                    bid=float(data.get('b', 0)),
                    ask=float(data.get('a', 0)),
                    last=float(data.get('c', 0)),
                    volume=float(data.get('v', 0)),
                    timestamp=datetime.fromtimestamp(data.get('E', 0) / 1000),
                )
                self._emit_price(tick)
                
            elif event_type == 'depthUpdate':
                # Order book update
                symbol = self._parse_symbol(data.get('s', ''))
                update = OrderBookUpdate(
                    symbol=symbol,
                    exchange=self.exchange,
                    bids=[(float(p), float(q)) for p, q in data.get('b', [])],
                    asks=[(float(p), float(q)) for p, q in data.get('a', [])],
                    timestamp=datetime.fromtimestamp(data.get('E', 0) / 1000),
                )
                self._emit_orderbook(update)
                
        except Exception as e:
            self.logger.debug(f"Message parse error: {e}")
    
    def _parse_symbol(self, binance_symbol: str) -> str:
        """Convert Binance symbol to standard format"""
        # BTCUSDT -> BTC/USDT
        if binance_symbol.endswith('USDT'):
            base = binance_symbol[:-4]
            return f"{base}/USDT"
        elif binance_symbol.endswith('BTC'):
            base = binance_symbol[:-3]
            return f"{base}/BTC"
        return binance_symbol


class CoinbaseWebSocketFeed(BaseWebSocketFeed):
    """Coinbase WebSocket price feed"""
    
    def __init__(self, **kwargs):
        super().__init__(exchange="coinbase", **kwargs)
    
    @property
    def websocket_url(self) -> str:
        return "wss://ws-feed.exchange.coinbase.com"
    
    async def _connect(self) -> bool:
        try:
            import aiohttp
            session = aiohttp.ClientSession()
            self._ws = await session.ws_connect(self.websocket_url)
            self._session = session
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False
    
    async def _subscribe(self, symbols: List[str]):
        """Subscribe to ticker channel"""
        # Convert to Coinbase format (BTC/USDT -> BTC-USDT)
        product_ids = [s.replace("/", "-") for s in symbols]
        
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": product_ids,
            "channels": ["ticker", "level2_batch"]
        }
        
        await self._ws.send_str(json.dumps(subscribe_msg))
        self.logger.info(f"Subscribed to {len(symbols)} symbols")
    
    async def _handle_message(self, message: str):
        """Parse Coinbase WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'ticker':
                symbol = data.get('product_id', '').replace('-', '/')
                tick = PriceTick(
                    symbol=symbol,
                    exchange=self.exchange,
                    bid=float(data.get('best_bid', 0) or 0),
                    ask=float(data.get('best_ask', 0) or 0),
                    last=float(data.get('price', 0) or 0),
                    volume=float(data.get('volume_24h', 0) or 0),
                    timestamp=datetime.fromisoformat(
                        data.get('time', '').replace('Z', '+00:00')
                    ) if data.get('time') else datetime.now(),
                )
                self._emit_price(tick)
                
            elif msg_type == 'l2update':
                symbol = data.get('product_id', '').replace('-', '/')
                changes = data.get('changes', [])
                
                bids = [(float(c[1]), float(c[2])) for c in changes if c[0] == 'buy']
                asks = [(float(c[1]), float(c[2])) for c in changes if c[0] == 'sell']
                
                if bids or asks:
                    update = OrderBookUpdate(
                        symbol=symbol,
                        exchange=self.exchange,
                        bids=bids,
                        asks=asks,
                    )
                    self._emit_orderbook(update)
                    
        except Exception as e:
            self.logger.debug(f"Message parse error: {e}")


class KrakenWebSocketFeed(BaseWebSocketFeed):
    """Kraken WebSocket price feed"""
    
    def __init__(self, **kwargs):
        super().__init__(exchange="kraken", **kwargs)
    
    @property
    def websocket_url(self) -> str:
        return "wss://ws.kraken.com"
    
    async def _connect(self) -> bool:
        try:
            import aiohttp
            session = aiohttp.ClientSession()
            self._ws = await session.ws_connect(self.websocket_url)
            self._session = session
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False
    
    async def _subscribe(self, symbols: List[str]):
        """Subscribe to ticker and book channels"""
        # Convert to Kraken format (BTC/USDT -> XBT/USDT)
        pairs = []
        for symbol in symbols:
            kraken_symbol = self._to_kraken_symbol(symbol)
            pairs.append(kraken_symbol)
        
        # Subscribe to ticker
        ticker_msg = {
            "event": "subscribe",
            "pair": pairs,
            "subscription": {"name": "ticker"}
        }
        await self._ws.send_str(json.dumps(ticker_msg))
        
        # Subscribe to book (depth 10)
        book_msg = {
            "event": "subscribe",
            "pair": pairs,
            "subscription": {"name": "book", "depth": 10}
        }
        await self._ws.send_str(json.dumps(book_msg))
        
        self.logger.info(f"Subscribed to {len(symbols)} symbols")
    
    async def _handle_message(self, message: str):
        """Parse Kraken WebSocket message"""
        try:
            data = json.loads(message)
            
            # Skip system messages
            if isinstance(data, dict):
                event = data.get('event')
                if event in ('systemStatus', 'subscriptionStatus', 'heartbeat'):
                    return
            
            # Data messages are arrays: [channelID, data, channelName, pair]
            if isinstance(data, list) and len(data) >= 4:
                channel_name = data[2]
                pair = data[3]
                symbol = self._from_kraken_symbol(pair)
                
                if channel_name == 'ticker':
                    ticker_data = data[1]
                    tick = PriceTick(
                        symbol=symbol,
                        exchange=self.exchange,
                        bid=float(ticker_data['b'][0]),  # Best bid price
                        ask=float(ticker_data['a'][0]),  # Best ask price
                        last=float(ticker_data['c'][0]),  # Close price
                        volume=float(ticker_data['v'][1]),  # Volume today
                        timestamp=datetime.now(),
                    )
                    self._emit_price(tick)
                    
                elif channel_name.startswith('book'):
                    book_data = data[1]
                    bids = [(float(b[0]), float(b[1])) for b in book_data.get('b', book_data.get('bs', []))]
                    asks = [(float(a[0]), float(a[1])) for a in book_data.get('a', book_data.get('as', []))]
                    
                    if bids or asks:
                        update = OrderBookUpdate(
                            symbol=symbol,
                            exchange=self.exchange,
                            bids=bids,
                            asks=asks,
                            is_snapshot='as' in book_data or 'bs' in book_data,
                        )
                        self._emit_orderbook(update)
                        
        except Exception as e:
            self.logger.debug(f"Message parse error: {e}")
    
    def _to_kraken_symbol(self, symbol: str) -> str:
        """Convert standard symbol to Kraken format"""
        # BTC/USDT -> XBT/USDT
        symbol = symbol.replace('BTC', 'XBT')
        return symbol
    
    def _from_kraken_symbol(self, pair: str) -> str:
        """Convert Kraken pair to standard format"""
        # XBT/USDT -> BTC/USDT
        pair = pair.replace('XBT', 'BTC')
        return pair


@dataclass
class OrderBookDepth:
    """Order book depth analysis with liquidity metrics"""
    symbol: str
    exchange: str
    timestamp: datetime
    
    # Depth metrics
    bid_depth_5: float = 0.0  # Total volume within 5 price levels
    ask_depth_5: float = 0.0
    bid_depth_10: float = 0.0
    ask_depth_10: float = 0.0
    
    # Imbalance
    imbalance_5: float = 0.0  # (bid - ask) / (bid + ask)
    imbalance_10: float = 0.0
    
    # Spread metrics
    spread_bps: float = 0.0  # Spread in basis points
    weighted_mid: float = 0.0  # Volume-weighted mid price
    
    @classmethod
    def from_orderbook(cls, book: OrderBookUpdate) -> 'OrderBookDepth':
        """Calculate depth metrics from order book update"""
        bids = book.bids[:10] if book.bids else []
        asks = book.asks[:10] if book.asks else []
        
        # Calculate depths
        bid_depth_5 = sum(q for _, q in bids[:5])
        ask_depth_5 = sum(q for _, q in asks[:5])
        bid_depth_10 = sum(q for _, q in bids)
        ask_depth_10 = sum(q for _, q in asks)
        
        # Imbalance
        total_5 = bid_depth_5 + ask_depth_5
        total_10 = bid_depth_10 + ask_depth_10
        imbalance_5 = (bid_depth_5 - ask_depth_5) / total_5 if total_5 > 0 else 0
        imbalance_10 = (bid_depth_10 - ask_depth_10) / total_10 if total_10 > 0 else 0
        
        # Spread
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread_bps = ((best_ask - best_bid) / best_bid) * 10000 if best_bid > 0 else 0
        
        # Weighted mid
        bid_value = sum(p * q for p, q in bids[:5])
        ask_value = sum(p * q for p, q in asks[:5])
        total_value = bid_value + ask_value
        weighted_mid = (bid_value + ask_value) / (bid_depth_5 + ask_depth_5) if (bid_depth_5 + ask_depth_5) > 0 else 0
        
        return cls(
            symbol=book.symbol,
            exchange=book.exchange,
            timestamp=book.timestamp,
            bid_depth_5=bid_depth_5,
            ask_depth_5=ask_depth_5,
            bid_depth_10=bid_depth_10,
            ask_depth_10=ask_depth_10,
            imbalance_5=imbalance_5,
            imbalance_10=imbalance_10,
            spread_bps=spread_bps,
            weighted_mid=weighted_mid,
        )
    
    @property
    def is_bid_heavy(self) -> bool:
        """Returns True if order book is bid-heavy (bullish signal)"""
        return self.imbalance_10 > 0.2
    
    @property
    def is_ask_heavy(self) -> bool:
        """Returns True if order book is ask-heavy (bearish signal)"""
        return self.imbalance_10 < -0.2
    
    @property
    def liquidity_score(self) -> float:
        """Liquidity score 0-1 based on depth and spread"""
        # Lower spread and higher depth = better liquidity
        spread_score = max(0, 1 - (self.spread_bps / 100))  # 100 bps = 0 score
        depth_score = min(1, (self.bid_depth_10 + self.ask_depth_10) / 1000)  # Normalized
        return (spread_score + depth_score) / 2


class PriceFeedManager:
    """
    Manages multiple WebSocket price feeds.
    
    Provides unified interface for real-time price updates
    across multiple exchanges.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("PriceFeedManager")
        self.feeds: Dict[str, BaseWebSocketFeed] = {}
        self._prices: Dict[str, Dict[str, PriceTick]] = {}  # {symbol: {exchange: tick}}
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    def add_feed(self, feed: BaseWebSocketFeed):
        """Add a WebSocket feed"""
        self.feeds[feed.exchange] = feed
        
        # Register price callback
        feed.on_price(self._on_price_update)
    
    def _on_price_update(self, tick: PriceTick):
        """Handle incoming price update"""
        if tick.symbol not in self._prices:
            self._prices[tick.symbol] = {}
        self._prices[tick.symbol][tick.exchange] = tick
    
    def get_price(self, symbol: str, exchange: str) -> Optional[PriceTick]:
        """Get latest price for symbol on exchange"""
        return self._prices.get(symbol, {}).get(exchange)
    
    def get_best_prices(self, symbol: str) -> Dict[str, PriceTick]:
        """Get prices for symbol across all exchanges"""
        return self._prices.get(symbol, {})
    
    def get_arbitrage_spread(self, symbol: str) -> Optional[Dict]:
        """
        Calculate arbitrage opportunity between exchanges.
        
        Returns dict with buy_exchange, sell_exchange, spread_pct
        """
        prices = self.get_best_prices(symbol)
        
        if len(prices) < 2:
            return None
        
        # Find lowest ask (buy) and highest bid (sell)
        best_buy = None
        best_sell = None
        
        for exchange, tick in prices.items():
            if best_buy is None or tick.ask < best_buy[1]:
                best_buy = (exchange, tick.ask)
            if best_sell is None or tick.bid > best_sell[1]:
                best_sell = (exchange, tick.bid)
        
        if best_buy and best_sell and best_buy[0] != best_sell[0]:
            spread = best_sell[1] - best_buy[1]
            spread_pct = (spread / best_buy[1]) * 100
            
            return {
                "symbol": symbol,
                "buy_exchange": best_buy[0],
                "buy_price": best_buy[1],
                "sell_exchange": best_sell[0],
                "sell_price": best_sell[1],
                "spread": spread,
                "spread_pct": spread_pct,
                "profitable": spread_pct > 0.1,  # > 0.1% is potentially profitable
            }
        
        return None
    
    async def start(self, symbols: List[str]):
        """Start all feeds"""
        self._running = True
        
        for exchange, feed in self.feeds.items():
            if await feed.connect(symbols):
                task = asyncio.create_task(feed.run())
                self._tasks.append(task)
                self.logger.info(f"Started {exchange} feed")
            else:
                self.logger.error(f"Failed to start {exchange} feed")
    
    async def stop(self):
        """Stop all feeds"""
        self._running = False
        
        for feed in self.feeds.values():
            await feed.disconnect()
        
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        self.logger.info("All feeds stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all feeds"""
        return {
            exchange: {
                "state": feed.state.value,
                "message_count": feed._message_count,
                "last_message": feed._last_message_time.isoformat() if feed._last_message_time else None,
                "subscribed_symbols": list(feed._subscribed_symbols),
            }
            for exchange, feed in self.feeds.items()
        }
