#!/usr/bin/env python3
"""
Real-Time Data Sources
======================
WebSocket and API clients for live market data.
"""

import asyncio
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import aiohttp
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.utils import retry, RetryStrategy, with_circuit_breaker


@dataclass
class MarketTick:
    """Real-time price tick"""
    symbol: str
    price: float
    volume_24h: float
    change_24h: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"


@dataclass
class SocialMention:
    """Social media mention data"""
    platform: str
    content: str
    author: str
    engagement: int
    sentiment: float  # -1 to 1
    timestamp: datetime
    url: Optional[str] = None
    asset_mentions: List[str] = field(default_factory=list)


@dataclass
class BlockchainEvent:
    """On-chain event data"""
    chain: str
    event_type: str  # 'whale_transfer', 'contract_deploy', 'large_swap', etc.
    data: Dict[str, Any]
    block_number: int
    tx_hash: str
    timestamp: datetime


class BaseDataSource(ABC):
    """Base class for all data sources"""

    def __init__(self, source_id: str):
        self.source_id = source_id
        self.config = get_config()
        self.logger = logging.getLogger(f"datasource_{source_id}")
        self.is_connected = False
        self._callbacks: List[Callable] = []

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
    async def connect(self):
        """Connect to data source"""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from data source"""
        pass


# ============================================
# PRICE DATA SOURCES
# ============================================

class CoinGeckoClient(BaseDataSource):
    """CoinGecko API client for price data"""

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self):
        super().__init__("coingecko")
        self.session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_delay = 1.5  # CoinGecko free tier rate limit

    async def connect(self):
        self.session = aiohttp.ClientSession()
        self.is_connected = True
        self.logger.info("CoinGecko client connected")

    async def disconnect(self):
        if self.session:
            await self.session.close()
        self.is_connected = False

    @with_circuit_breaker("coingecko_api", failure_threshold=5, timeout=60.0)
    @retry(max_attempts=3, base_delay=2.0, retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def get_price(self, coin_id: str, vs_currency: str = "usd") -> Optional[MarketTick]:
        """Get current price for a coin"""
        if not self.session:
            await self.connect()

        url = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": vs_currency,
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
        }

        async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if coin_id in data:
                    coin_data = data[coin_id]
                    tick = MarketTick(
                        symbol=f"{coin_id.upper()}/{vs_currency.upper()}",
                        price=coin_data.get(vs_currency, 0),
                        volume_24h=coin_data.get(f"{vs_currency}_24h_vol", 0),
                        change_24h=coin_data.get(f"{vs_currency}_24h_change", 0),
                        source="coingecko",
                    )
                    await self._notify(tick)
                    await asyncio.sleep(self._rate_limit_delay)
                    return tick
            elif resp.status == 429:
                # Rate limited - raise to trigger retry
                raise aiohttp.ClientError("Rate limited by CoinGecko")

        await asyncio.sleep(self._rate_limit_delay)
        return None

    @with_circuit_breaker("coingecko_api", failure_threshold=5, timeout=60.0)
    @retry(max_attempts=3, base_delay=2.0, retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def get_prices_batch(self, coin_ids: List[str], vs_currency: str = "usd") -> List[MarketTick]:
        """Get prices for multiple coins"""
        if not self.session:
            await self.connect()

        ticks = []
        url = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": vs_currency,
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
        }

        async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 200:
                data = await resp.json()
                for coin_id in coin_ids:
                    if coin_id in data:
                        coin_data = data[coin_id]
                        tick = MarketTick(
                            symbol=f"{coin_id.upper()}/{vs_currency.upper()}",
                            price=coin_data.get(vs_currency, 0),
                            volume_24h=coin_data.get(f"{vs_currency}_24h_vol", 0),
                            change_24h=coin_data.get(f"{vs_currency}_24h_change", 0),
                            source="coingecko",
                        )
                        ticks.append(tick)
                        await self._notify(tick)
            elif resp.status == 429:
                raise aiohttp.ClientError("Rate limited by CoinGecko")

        await asyncio.sleep(self._rate_limit_delay)
        return ticks

    @with_circuit_breaker("coingecko_api", failure_threshold=5, timeout=60.0)
    @retry(max_attempts=3, base_delay=2.0, retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def get_trending(self) -> List[Dict]:
        """Get trending coins"""
        if not self.session:
            await self.connect()

        url = f"{self.BASE_URL}/search/trending"
        async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                await asyncio.sleep(self._rate_limit_delay)
                return data.get("coins", [])
            elif resp.status == 429:
                raise aiohttp.ClientError("Rate limited by CoinGecko")

        await asyncio.sleep(self._rate_limit_delay)
        return []


class BinanceWebSocket(BaseDataSource):
    """Binance WebSocket for real-time price data with auto-reconnection"""

    WS_URL = "wss://stream.binance.com:9443/ws"

    def __init__(self):
        super().__init__("binance_ws")
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._subscribed_symbols: List[str] = []
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 1.0  # Start with 1 second
        self._max_reconnect_delay = 60.0
        self._heartbeat_interval = 30.0
        self._last_message_time: Optional[datetime] = None

    async def connect(self):
        self.session = aiohttp.ClientSession()
        self.is_connected = True
        self._reconnect_attempts = 0
        self.logger.info("Binance WebSocket client ready")

    async def disconnect(self):
        self._running = False
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
        self.is_connected = False
        self.logger.info("Binance WebSocket disconnected")

    async def subscribe_ticker(self, symbols: List[str]):
        """Subscribe to ticker updates for symbols with auto-reconnection"""
        self._subscribed_symbols = symbols
        self._running = True
        
        while self._running:
            try:
                await self._connect_and_listen(symbols)
            except Exception as e:
                if not self._running:
                    break
                    
                self._reconnect_attempts += 1
                if self._reconnect_attempts > self._max_reconnect_attempts:
                    self.logger.error(
                        f"Max reconnection attempts ({self._max_reconnect_attempts}) reached. Stopping."
                    )
                    break
                
                # Exponential backoff with jitter
                delay = min(
                    self._reconnect_delay * (2 ** (self._reconnect_attempts - 1)),
                    self._max_reconnect_delay
                )
                delay *= (0.5 + random.random())  # Add jitter
                
                self.logger.warning(
                    f"WebSocket error: {e}. Reconnecting in {delay:.1f}s "
                    f"(attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})"
                )
                await asyncio.sleep(delay)

    async def _connect_and_listen(self, symbols: List[str]):
        """Internal method to connect and listen to WebSocket"""
        streams = [f"{s.lower()}@ticker" for s in symbols]
        stream_url = f"{self.WS_URL}/{'/'.join(streams)}"

        if not self.session:
            await self.connect()

        async with self.session.ws_connect(
            stream_url,
            heartbeat=self._heartbeat_interval,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as ws:
            self.ws = ws
            self._reconnect_attempts = 0  # Reset on successful connection
            self.logger.info(f"Subscribed to {len(symbols)} symbols")

            async for msg in ws:
                if not self._running:
                    break

                self._last_message_time = datetime.now()

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    tick = self._parse_ticker(data)
                    if tick:
                        await self._notify(tick)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {ws.exception()}")
                    raise Exception(f"WebSocket error: {ws.exception()}")
                    
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    self.logger.warning("WebSocket closed by server")
                    raise Exception("WebSocket closed by server")

    def _parse_ticker(self, data: Dict) -> Optional[MarketTick]:
        """Parse Binance ticker data"""
        try:
            return MarketTick(
                symbol=data.get("s", ""),
                price=float(data.get("c", 0)),
                volume_24h=float(data.get("v", 0)),
                change_24h=float(data.get("P", 0)),
                bid=float(data.get("b", 0)),
                ask=float(data.get("a", 0)),
                source="binance_ws",
            )
        except (KeyError, ValueError) as e:
            self.logger.error(f"Parse error: {e}")
            return None

    @property
    def is_healthy(self) -> bool:
        """Check if WebSocket connection is healthy"""
        if not self._last_message_time:
            return False
        elapsed = (datetime.now() - self._last_message_time).total_seconds()
        return elapsed < self._heartbeat_interval * 2


# ============================================
# SOCIAL DATA SOURCES
# ============================================

class RedditClient(BaseDataSource):
    """Reddit API client for social sentiment"""

    BASE_URL = "https://www.reddit.com"

    def __init__(self):
        super().__init__("reddit")
        self.session: Optional[aiohttp.ClientSession] = None
        self.subreddits = ["cryptocurrency", "bitcoin", "ethereum", "defi", "altcoin"]

    async def connect(self):
        headers = {"User-Agent": "ACC-Research-Agent/1.0"}
        self.session = aiohttp.ClientSession(headers=headers)
        self.is_connected = True
        self.logger.info("Reddit client connected")

    async def disconnect(self):
        if self.session:
            await self.session.close()
        self.is_connected = False

    @retry(max_attempts=3, base_delay=2.0, retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def get_hot_posts(self, subreddit: str, limit: int = 25) -> List[SocialMention]:
        """Get hot posts from a subreddit"""
        if not self.session:
            await self.connect()

        mentions = []
        try:
            url = f"{self.BASE_URL}/r/{subreddit}/hot.json?limit={limit}"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for post in data.get("data", {}).get("children", []):
                        post_data = post.get("data", {})
                        mention = SocialMention(
                            platform="reddit",
                            content=post_data.get("title", ""),
                            author=post_data.get("author", ""),
                            engagement=post_data.get("score", 0) + post_data.get("num_comments", 0),
                            sentiment=0.0,  # Would use NLP for real sentiment
                            timestamp=datetime.fromtimestamp(post_data.get("created_utc", 0)),
                            url=f"https://reddit.com{post_data.get('permalink', '')}",
                            asset_mentions=self._extract_assets(post_data.get("title", "")),
                        )
                        mentions.append(mention)
                        await self._notify(mention)

            await asyncio.sleep(1)  # Rate limiting
        except aiohttp.ClientError:
            raise  # Let retry handle it
        except asyncio.TimeoutError:
            raise  # Let retry handle it
        except Exception as e:
            self.logger.error(f"Reddit fetch error: {e}")

        return mentions

    async def scan_all_subreddits(self) -> List[SocialMention]:
        """Scan all tracked subreddits"""
        all_mentions = []
        for subreddit in self.subreddits:
            mentions = await self.get_hot_posts(subreddit)
            all_mentions.extend(mentions)
        return all_mentions

    def _extract_assets(self, text: str) -> List[str]:
        """Extract crypto asset mentions from text"""
        # Common crypto tickers
        assets = []
        text_upper = text.upper()
        
        common_tickers = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC", "LINK"]
        for ticker in common_tickers:
            if ticker in text_upper:
                assets.append(ticker)
        
        return assets


class TwitterClient(BaseDataSource):
    """Twitter/X API client for social sentiment"""

    def __init__(self):
        super().__init__("twitter")
        self.session: Optional[aiohttp.ClientSession] = None
        self.bearer_token = None

    async def connect(self):
        # Get bearer token from environment variable directly
        self.bearer_token = os.environ.get("TWITTER_BEARER_TOKEN", "")
        
        if self.bearer_token:
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            self.session = aiohttp.ClientSession(headers=headers)
            self.is_connected = True
            self.logger.info("Twitter client connected")
        else:
            self.logger.warning("Twitter bearer token not configured")

    async def disconnect(self):
        if self.session:
            await self.session.close()
        self.is_connected = False

    async def search_recent(self, query: str, max_results: int = 10) -> List[SocialMention]:
        """Search recent tweets"""
        if not self.bearer_token:
            self.logger.warning("Twitter not configured")
            return []

        mentions = []
        # Twitter API v2 implementation would go here
        # Requires paid API access
        return mentions


# ============================================
# BLOCKCHAIN DATA SOURCES
# ============================================

class EtherscanClient(BaseDataSource):
    """Etherscan API client for on-chain data"""

    BASE_URL = "https://api.etherscan.io/api"

    def __init__(self):
        super().__init__("etherscan")
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_key: Optional[str] = None

    async def connect(self):
        # Get API key from environment variable directly
        self.api_key = os.environ.get("ETHERSCAN_API_KEY", "")
        self.session = aiohttp.ClientSession()
        self.is_connected = True
        self.logger.info("Etherscan client connected")

    async def disconnect(self):
        if self.session:
            await self.session.close()
        self.is_connected = False

    @with_circuit_breaker("etherscan_api", failure_threshold=5, timeout=60.0)
    @retry(max_attempts=3, base_delay=2.0, retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def get_whale_transactions(self, min_value_eth: float = 1000) -> List[BlockchainEvent]:
        """Get recent whale transactions"""
        if not self.session:
            await self.connect()

        events = []
        try:
            # Known whale addresses to track
            whale_addresses = [
                "0x28C6c06298d514Db089934071355E5743bf21d60",  # Binance 14
                "0xDFd5293D8e347dFe59E90eFd55b2956a1343963d",  # Binance 8  
                "0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549",  # Binance 15
                "0x56Eddb7aa87536c09CCc2793473599fD21A8b17F",  # Coinbase 2
                "0x503828976D22510aad0201ac7EC88293211D23Da",  # Coinbase 4
                "0x6cC5F688a315f3dC28A7781717a9A798a59fDA7b",  # OKX
                "0x66f820a414680b5bcda5eeca5dea238543f42054",  # Jump Trading
                "0x40ec5B33f54e0E8A33A975908C5BA1c14e5BbbDf",  # Polygon Bridge
            ]
            
            # For each whale address, get recent transactions
            for address in whale_addresses[:3]:  # Limit to avoid rate limits
                params = {
                    "module": "account",
                    "action": "txlist",
                    "address": address,
                    "startblock": 0,
                    "endblock": 99999999,
                    "page": 1,
                    "offset": 10,
                    "sort": "desc",
                    "apikey": self.api_key or "",
                }

                async with self.session.get(self.BASE_URL, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for tx in data.get("result", []):
                            value_eth = int(tx.get("value", 0)) / 1e18
                            if value_eth >= min_value_eth:
                                event = BlockchainEvent(
                                    chain="ethereum",
                                    event_type="whale_transfer",
                                    contract_address=tx.get("to", ""),
                                    data={
                                        "from": tx.get("from", ""),
                                        "to": tx.get("to", ""),
                                        "value_eth": value_eth,
                                        "tx_hash": tx.get("hash", ""),
                                    },
                                    block_number=int(tx.get("blockNumber", 0)),
                                    tx_hash=tx.get("hash", ""),
                                    timestamp=datetime.fromtimestamp(int(tx.get("timeStamp", 0))),
                                )
                                events.append(event)
                                await self._notify(event)
                
                await asyncio.sleep(0.25)  # Rate limiting between addresses

        except aiohttp.ClientError:
            raise  # Let retry handle it
        except asyncio.TimeoutError:
            raise  # Let retry handle it
        except Exception as e:
            self.logger.error(f"Etherscan fetch error: {e}")

        return events

    @with_circuit_breaker("etherscan_api", failure_threshold=5, timeout=60.0)
    @retry(max_attempts=3, base_delay=2.0, retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def get_gas_prices(self) -> Dict[str, int]:
        """Get current gas prices"""
        if not self.session:
            await self.connect()

        try:
            params = {
                "module": "gastracker",
                "action": "gasoracle",
                "apikey": self.api_key or "",
            }

            async with self.session.get(self.BASE_URL, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {})
                    return {
                        "safe_low": int(result.get("SafeGasPrice", 0)),
                        "standard": int(result.get("ProposeGasPrice", 0)),
                        "fast": int(result.get("FastGasPrice", 0)),
                    }

        except Exception as e:
            self.logger.error(f"Gas price fetch error: {e}")

        return {"safe_low": 0, "standard": 0, "fast": 0}


# ============================================
# DATA AGGREGATOR
# ============================================

class DataAggregator:
    """Aggregates data from all sources"""

    def __init__(self):
        self.logger = logging.getLogger("data_aggregator")
        
        # Initialize all data sources
        self.coingecko = CoinGeckoClient()
        self.binance_ws = BinanceWebSocket()
        self.reddit = RedditClient()
        self.twitter = TwitterClient()
        self.etherscan = EtherscanClient()
        
        # Data storage
        self.latest_ticks: Dict[str, MarketTick] = {}
        self.social_mentions: List[SocialMention] = []
        self.blockchain_events: List[BlockchainEvent] = []
        
        # Subscribe to updates
        self.coingecko.subscribe(self._on_tick)
        self.binance_ws.subscribe(self._on_tick)
        self.reddit.subscribe(self._on_social)

    async def _on_tick(self, tick: MarketTick):
        """Handle price tick"""
        self.latest_ticks[tick.symbol] = tick

    async def _on_social(self, mention: SocialMention):
        """Handle social mention"""
        self.social_mentions.append(mention)
        # Keep only last 1000
        if len(self.social_mentions) > 1000:
            self.social_mentions = self.social_mentions[-1000:]

    async def connect_all(self):
        """Connect all data sources"""
        await asyncio.gather(
            self.coingecko.connect(),
            self.reddit.connect(),
            self.etherscan.connect(),
        )
        self.logger.info("All data sources connected")

    async def disconnect_all(self):
        """Disconnect all data sources"""
        await asyncio.gather(
            self.coingecko.disconnect(),
            self.binance_ws.disconnect(),
            self.reddit.disconnect(),
            self.twitter.disconnect(),
            self.etherscan.disconnect(),
        )

    async def get_market_snapshot(self, symbols: List[str]) -> Dict[str, MarketTick]:
        """
        Get current market snapshot with graceful degradation.
        
        Falls back to cached data if primary data source is unavailable.
        """
        # Map symbols to CoinGecko IDs
        symbol_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "XRP": "ripple",
            "ADA": "cardano",
            "AVAX": "avalanche-2",
            "DOT": "polkadot",
            "MATIC": "matic-network",
            "LINK": "chainlink",
            "DOGE": "dogecoin",
        }
        
        coin_ids = [symbol_map.get(s.upper(), s.lower()) for s in symbols]
        
        result: Dict[str, MarketTick] = {}
        
        try:
            # Try primary source (CoinGecko)
            ticks = await self.coingecko.get_prices_batch(coin_ids)
            
            for tick in ticks:
                result[tick.symbol] = tick
                # Cache the tick for fallback
                self.latest_ticks[tick.symbol] = tick
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Primary data source failed: {e}. Attempting fallback...")
            
            # Fallback 1: Use cached data if fresh enough (< 60 seconds old)
            fallback_ticks = []
            for symbol in symbols:
                # Check symbol variations
                for key in [symbol, f"{symbol}/USDT", f"{symbol}/USD"]:
                    if key in self.latest_ticks:
                        tick = self.latest_ticks[key]
                        age = (datetime.now() - tick.timestamp).total_seconds()
                        if age < 60:  # Accept data up to 60 seconds old
                            fallback_ticks.append(tick)
                            self.logger.debug(f"Using cached {key} data ({age:.1f}s old)")
                            break
            
            if fallback_ticks:
                self.logger.info(
                    f"Graceful degradation: serving {len(fallback_ticks)}/{len(symbols)} "
                    f"symbols from cache"
                )
                return {tick.symbol: tick for tick in fallback_ticks}
            
            # Fallback 2: Return empty with warning
            self.logger.error(
                f"No fallback data available for {symbols}. "
                f"Data source and cache both unavailable."
            )
            return {}

    async def get_market_snapshot_with_source(
        self, 
        symbols: List[str],
        prefer_websocket: bool = True
    ) -> Dict[str, MarketTick]:
        """
        Get market snapshot with explicit source preference and fallback chain.
        
        Fallback order:
        1. WebSocket feed (if enabled and connected)
        2. CoinGecko API
        3. Cached data
        
        Args:
            symbols: List of symbol strings (e.g., ["BTC", "ETH"])
            prefer_websocket: Whether to try WebSocket first
            
        Returns:
            Dict mapping symbol to MarketTick
        """
        result: Dict[str, MarketTick] = {}
        missing_symbols = list(symbols)
        
        # Try WebSocket first if preferred
        if prefer_websocket and self.binance_ws.is_connected:
            for symbol in list(missing_symbols):
                ws_symbol = f"{symbol}/USDT"
                if ws_symbol in self.latest_ticks:
                    tick = self.latest_ticks[ws_symbol]
                    age = (datetime.now() - tick.timestamp).total_seconds()
                    if age < 5:  # WebSocket data should be very fresh
                        result[tick.symbol] = tick
                        missing_symbols.remove(symbol)
        
        # Try CoinGecko for remaining symbols
        if missing_symbols:
            try:
                snapshot = await self.get_market_snapshot(missing_symbols)
                result.update(snapshot)
                missing_symbols = [s for s in missing_symbols if s not in result and f"{s}/USDT" not in result]
            except Exception as e:
                self.logger.warning(f"CoinGecko fallback failed: {e}")
        
        # Final cache check for any still missing
        if missing_symbols:
            self.logger.warning(f"Missing data for {missing_symbols} after all fallbacks")
        
        return result

    async def get_social_pulse(self) -> Dict[str, Any]:
        """Get current social media pulse"""
        mentions = await self.reddit.scan_all_subreddits()
        
        # Aggregate by asset
        asset_buzz: Dict[str, int] = {}
        for mention in mentions:
            for asset in mention.asset_mentions:
                asset_buzz[asset] = asset_buzz.get(asset, 0) + mention.engagement
        
        return {
            "total_mentions": len(mentions),
            "asset_buzz": asset_buzz,
            "top_posts": sorted(mentions, key=lambda m: m.engagement, reverse=True)[:10],
        }

    async def get_gas_status(self) -> Dict[str, int]:
        """Get current gas prices"""
        return await self.etherscan.get_gas_prices()


# CLI for testing
if __name__ == "__main__":
    async def test():
        print("=== Data Sources Test ===\n")
        
        aggregator = DataAggregator()
        await aggregator.connect_all()
        
        print("1. Fetching market snapshot...")
        snapshot = await aggregator.get_market_snapshot(["BTC", "ETH", "SOL"])
        for symbol, tick in snapshot.items():
            print(f"   {symbol}: ${tick.price:,.2f} ({tick.change_24h:+.2f}%)")
        
        print("\n2. Fetching trending coins...")
        trending = await aggregator.coingecko.get_trending()
        for i, coin in enumerate(trending[:5], 1):
            name = coin.get("item", {}).get("name", "Unknown")
            print(f"   {i}. {name}")
        
        print("\n3. Fetching gas prices...")
        gas = await aggregator.get_gas_status()
        print(f"   Safe: {gas['safe_low']} | Standard: {gas['standard']} | Fast: {gas['fast']}")
        
        print("\n4. Fetching social pulse (Reddit)...")
        pulse = await aggregator.get_social_pulse()
        print(f"   Total mentions: {pulse['total_mentions']}")
        print(f"   Asset buzz: {pulse['asset_buzz']}")
        
        await aggregator.disconnect_all()
        print("\n=== Test Complete ===")

    asyncio.run(test())
