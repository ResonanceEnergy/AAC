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
    contract_address: Optional[str] = None


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
        raise NotImplementedError(f"{type(self).__name__} must implement connect")

    @abstractmethod
    async def disconnect(self):
        """Disconnect from data source"""
        raise NotImplementedError(f"{type(self).__name__} must implement disconnect")


# ============================================
# PRICE DATA SOURCES
# ============================================

class CoinGeckoClient(BaseDataSource):
    """CoinGecko API client for price data (supports Free, Demo, and Pro tiers)"""

    FREE_URL = "https://api.coingecko.com/api/v3"
    PRO_URL = "https://pro-api.coingecko.com/api/v3"

    def __init__(self):
        super().__init__("coingecko")
        self.session: Optional[aiohttp.ClientSession] = None  # set in connect()
        self._api_key = self.config.coingecko_key
        self._is_pro = bool(self._api_key)
        self.BASE_URL = self.PRO_URL if self._is_pro else self.FREE_URL
        # Pro: 500 req/min, Free: ~10 req/min
        self._rate_limit_delay = 0.15 if self._is_pro else 1.5

    def _get_headers(self) -> Dict[str, str]:
        """Get auth headers — Pro API uses x-cg-pro-api-key"""
        if self._is_pro:
            return {'x-cg-pro-api-key': self._api_key}
        return {}

    @property
    def tier_name(self) -> str:
        """Tier name."""
        return "Pro" if self._is_pro else "Free"

    async def connect(self):
        # Use ThreadedResolver to avoid aiodns/pycares DNS failures on some systems
        """Connect."""
        connector = aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
        self.session = aiohttp.ClientSession(connector=connector, headers=self._get_headers())
        self.is_connected = True
        self.logger.info(f"CoinGecko client connected ({self.tier_name} tier)")

    async def disconnect(self):
        """Disconnect."""
        if self.session:
            await self.session.close()
        self.is_connected = False

    @with_circuit_breaker("coingecko_api", failure_threshold=5, timeout=60.0)
    @retry(max_attempts=3, base_delay=2.0, retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def get_price(self, coin_id: str, vs_currency: str = "usd") -> Optional[MarketTick]:
        """Get current price for a coin"""
        if not self.session:
            await self.connect()
        assert self.session is not None

        url = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": vs_currency,
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
        }

        async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:  # type: ignore[union-attr]
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
        assert self.session is not None

        ticks = []
        url = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": vs_currency,
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
        }

        async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:  # type: ignore[union-attr]
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
        assert self.session is not None

        url = f"{self.BASE_URL}/search/trending"
        async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:  # type: ignore[union-attr]
            if resp.status == 200:
                data = await resp.json()
                await asyncio.sleep(self._rate_limit_delay)
                return data.get("coins", [])
            elif resp.status == 429:
                raise aiohttp.ClientError("Rate limited by CoinGecko")

        await asyncio.sleep(self._rate_limit_delay)
        return []

    # --- Pro-tier methods (require API key) ---

    @with_circuit_breaker("coingecko_api", failure_threshold=5, timeout=60.0)
    @retry(max_attempts=3, base_delay=2.0, retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def get_coin_market_chart(
        self, coin_id: str, vs_currency: str = "usd", days: int = 30
    ) -> Dict[str, Any]:
        """Get historical market data (OHLCV available on Pro tier)"""
        if not self.session:
            await self.connect()
        assert self.session is not None

        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": str(days)}

        async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:  # type: ignore[union-attr]
            if resp.status == 200:
                data = await resp.json()
                await asyncio.sleep(self._rate_limit_delay)
                return data
            elif resp.status == 429:
                raise aiohttp.ClientError("Rate limited by CoinGecko")

        await asyncio.sleep(self._rate_limit_delay)
        return {}

    @with_circuit_breaker("coingecko_api", failure_threshold=5, timeout=60.0)
    @retry(max_attempts=3, base_delay=2.0, retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def get_global_data(self) -> Dict[str, Any]:
        """Get global crypto market data (market cap, volume, dominance)"""
        if not self.session:
            await self.connect()
        assert self.session is not None

        url = f"{self.BASE_URL}/global"
        async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:  # type: ignore[union-attr]
            if resp.status == 200:
                data = await resp.json()
                await asyncio.sleep(self._rate_limit_delay)
                return data.get("data", {})
            elif resp.status == 429:
                raise aiohttp.ClientError("Rate limited by CoinGecko")

        await asyncio.sleep(self._rate_limit_delay)
        return {}

    @with_circuit_breaker("coingecko_api", failure_threshold=5, timeout=60.0)
    @retry(max_attempts=3, base_delay=2.0, retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def get_coin_details(self, coin_id: str) -> Dict[str, Any]:
        """Get detailed coin information (desc, links, market data, dev stats)"""
        if not self.session:
            await self.connect()
        assert self.session is not None

        url = f"{self.BASE_URL}/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "true",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
        }

        async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:  # type: ignore[union-attr]
            if resp.status == 200:
                data = await resp.json()
                await asyncio.sleep(self._rate_limit_delay)
                return data
            elif resp.status == 429:
                raise aiohttp.ClientError("Rate limited by CoinGecko")

        await asyncio.sleep(self._rate_limit_delay)
        return {}

    @with_circuit_breaker("coingecko_api", failure_threshold=5, timeout=60.0)
    @retry(max_attempts=3, base_delay=2.0, retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def get_coins_markets(
        self,
        vs_currency: str = "usd",
        order: str = "market_cap_desc",
        per_page: int = 100,
        page: int = 1,
        sparkline: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get top coins by market cap with full market data"""
        if not self.session:
            await self.connect()
        assert self.session is not None

        url = f"{self.BASE_URL}/coins/markets"
        params = {
            "vs_currency": vs_currency,
            "order": order,
            "per_page": str(per_page),
            "page": str(page),
            "sparkline": str(sparkline).lower(),
        }

        async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:  # type: ignore[union-attr]
            if resp.status == 200:
                data = await resp.json()
                await asyncio.sleep(self._rate_limit_delay)
                return data if isinstance(data, list) else []
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
        """Connect."""
        self.session = aiohttp.ClientSession()
        self.is_connected = True
        self._reconnect_attempts = 0
        self.logger.info("Binance WebSocket client ready")

    async def disconnect(self):
        """Disconnect."""
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
        assert self.session is not None

        async with self.session.ws_connect(  # type: ignore[union-attr]
            stream_url,
            heartbeat=self._heartbeat_interval,
            timeout=aiohttp.ClientWSTimeout(ws_close=60.0)  # type: ignore[arg-type]
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
        """Connect."""
        headers = {"User-Agent": "ACC-Research-Agent/1.0"}
        self.session = aiohttp.ClientSession(headers=headers)
        self.is_connected = True
        self.logger.info("Reddit client connected")

    async def disconnect(self):
        """Disconnect."""
        if self.session:
            await self.session.close()
        self.is_connected = False

    @retry(max_attempts=3, base_delay=2.0, retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def get_hot_posts(self, subreddit: str, limit: int = 25) -> List[SocialMention]:
        """Get hot posts from a subreddit"""
        if not self.session:
            await self.connect()
        assert self.session is not None

        mentions: List[SocialMention] = []
        try:
            url = f"{self.BASE_URL}/r/{subreddit}/hot.json?limit={limit}"
            async with self.session.get(url) as resp:  # type: ignore[union-attr]
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
        """Connect."""
        self.bearer_token = os.environ.get("TWITTER_BEARER_TOKEN", "")
        
        if self.bearer_token:
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            self.session = aiohttp.ClientSession(headers=headers)
            self.is_connected = True
            self.logger.info("Twitter client connected")
        else:
            self.logger.warning("Twitter bearer token not configured")

    async def disconnect(self):
        """Disconnect."""
        if self.session:
            await self.session.close()
        self.is_connected = False

    async def search_recent(self, query: str, max_results: int = 10) -> List[SocialMention]:
        """Search recent tweets"""
        if not self.bearer_token:
            self.logger.warning("Twitter not configured")
            return []

        mentions: List[SocialMention] = []
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
        """Connect."""
        self.api_key = os.environ.get("ETHERSCAN_API_KEY", "")
        self.session = aiohttp.ClientSession()
        self.is_connected = True
        self.logger.info("Etherscan client connected")

    async def disconnect(self):
        """Disconnect."""
        if self.session:
            await self.session.close()
        self.is_connected = False

    @with_circuit_breaker("etherscan_api", failure_threshold=5, timeout=60.0)
    @retry(max_attempts=3, base_delay=2.0, retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def get_whale_transactions(self, min_value_eth: float = 1000) -> List[BlockchainEvent]:
        """Get recent whale transactions"""
        if not self.session:
            await self.connect()
        assert self.session is not None

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

                async with self.session.get(self.BASE_URL, params=params) as resp:  # type: ignore[union-attr]
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
        assert self.session is not None

        try:
            params = {
                "module": "gastracker",
                "action": "gasoracle",
                "apikey": self.api_key or "",
            }

            async with self.session.get(self.BASE_URL, params=params) as resp:  # type: ignore[union-attr]
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
# METAL BLOCKCHAIN DATA SOURCE (Vector 5)
# ============================================

class MetalBlockchainSource(BaseDataSource):
    """
    Metal Blockchain C-Chain data source.

    Monitors EVM-compatible C-Chain for:
      - Large token transfers (whale alerts)
      - Smart contract deployments
      - DeFi protocol interactions
      - Bridge activity between Metal Blockchain and other chains

    Metal Blockchain uses Avalanche subnet architecture with
    three chains: X-Chain (assets), P-Chain (validators), C-Chain (EVM).
    """

    DEFAULT_RPC = "https://tahoe.metalblockchain.org/ext/bc/C/rpc"

    def __init__(self):
        super().__init__("metal_blockchain")
        self.session: Optional[aiohttp.ClientSession] = None
        self._rpc_url = self.config.metal_blockchain_rpc_url if hasattr(self.config, 'metal_blockchain_rpc_url') and self.config.metal_blockchain_rpc_url else self.DEFAULT_RPC

    async def connect(self):
        """Connect."""
        connector = aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
        self.session = aiohttp.ClientSession(connector=connector)
        self.is_connected = True
        self.logger.info(f"Metal Blockchain source connected ({self._rpc_url})")

    async def disconnect(self):
        """Disconnect."""
        if self.session:
            await self.session.close()
        self.is_connected = False

    async def _rpc_call(self, method: str, params: Optional[list] = None) -> Any:
        """Make a JSON-RPC call to Metal Blockchain C-Chain."""
        if not self.session:
            await self.connect()
        assert self.session is not None

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or [],
        }

        try:
            async with self.session.post(self._rpc_url, json=payload) as resp:  # type: ignore[union-attr]
                resp.raise_for_status()
                data = await resp.json()
                if "error" in data:
                    self.logger.error(f"RPC error: {data['error']}")
                    return None
                return data.get("result")
        except Exception as e:
            self.logger.error(f"Metal Blockchain RPC error: {e}")
            return None

    async def get_latest_block(self) -> Optional[Dict]:
        """Get the latest block on C-Chain."""
        return await self._rpc_call("eth_getBlockByNumber", ["latest", False])

    async def get_block_number(self) -> Optional[int]:
        """Get the current block height."""
        result = await self._rpc_call("eth_blockNumber")
        return int(result, 16) if result else None

    async def get_balance(self, address: str) -> Optional[float]:
        """Get METAL balance for an address (in METAL, not wei)."""
        result = await self._rpc_call("eth_getBalance", [address, "latest"])
        if result:
            return int(result, 16) / 1e18
        return None

    async def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        """Get transaction details by hash."""
        return await self._rpc_call("eth_getTransactionByHash", [tx_hash])

    async def monitor_blocks(self, callback=None, poll_interval: float = 2.0):
        """
        Poll for new blocks and emit BlockchainEvents.

        In production, this runs as a background task to feed
        the XPR Intelligence Agent with on-chain events.
        """
        last_block = await self.get_block_number()
        if last_block is None:
            self.logger.error("Cannot get initial block number")
            return

        while self.is_connected:
            try:
                current = await self.get_block_number()
                if current and current > last_block:
                    for bn in range(last_block + 1, current + 1):
                        block = await self._rpc_call(
                            "eth_getBlockByNumber", [hex(bn), True]
                        )
                        if block:
                            event = BlockchainEvent(
                                chain="metal_blockchain",
                                event_type="new_block",
                                data={
                                    "block_number": bn,
                                    "tx_count": len(block.get("transactions", [])),
                                    "timestamp": int(block.get("timestamp", "0x0"), 16),
                                },
                                block_number=bn,
                                tx_hash=block.get("hash", ""),
                                timestamp=datetime.fromtimestamp(
                                    int(block.get("timestamp", "0x0"), 16)
                                ),
                            )
                            await self._notify(event)
                            if callback:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(event)
                                else:
                                    callback(event)

                    last_block = current

                await asyncio.sleep(poll_interval)

            except Exception as e:
                self.logger.error(f"Block monitor error: {e}")
                await asyncio.sleep(poll_interval * 2)


# ============================================
# DATA AGGREGATOR
# ============================================

from shared.symbol_classifier import EQUITY_SYMBOLS, is_equity


class DataAggregator:
    """Aggregates data from all sources"""

    # Re-export for backward compat — canonical set lives in symbol_classifier
    EQUITY_SYMBOLS = EQUITY_SYMBOLS

    def __init__(self):
        self.logger = logging.getLogger("data_aggregator")
        
        # Initialize all data sources
        self.coingecko = CoinGeckoClient()
        self.binance_ws = BinanceWebSocket()
        self.reddit = RedditClient()
        self.twitter = TwitterClient()
        self.etherscan = EtherscanClient()
        self.metal_blockchain = MetalBlockchainSource()

        # Forex (Knightsbridge FX)
        from shared.forex_data_source import ForexDataSource
        self.forex = ForexDataSource()

        # Exchange connectors (injected by orchestrator after init)
        self._ibkr_connector: Optional[Any] = None
        self._ndax_connector: Optional[Any] = None

        # Data storage
        self.latest_ticks: Dict[str, MarketTick] = {}
        self.social_mentions: List[SocialMention] = []
        self.blockchain_events: List[BlockchainEvent] = []
        
        # Subscribe to updates
        self.coingecko.subscribe(self._on_tick)
        self.binance_ws.subscribe(self._on_tick)
        self.forex.subscribe(self._on_tick)
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

    def set_ibkr_connector(self, connector: Any) -> None:
        """Inject IBKR connector for equity/ETF/options pricing."""
        self._ibkr_connector = connector
        self.logger.info("IBKR connector wired into DataAggregator")

    def set_ndax_connector(self, connector: Any) -> None:
        """Inject NDAX connector for CAD crypto pricing."""
        self._ndax_connector = connector
        self.logger.info("NDAX connector wired into DataAggregator")

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

    async def _ibkr_get_ticker(self, symbol: str) -> Optional[MarketTick]:
        """Fetch a single equity/ETF quote from IBKR."""
        if not self._ibkr_connector:
            return None
        try:
            ticker = await self._ibkr_connector.get_ticker(f"{symbol}/USD")
            if ticker and ticker.last > 0:
                tick = MarketTick(
                    symbol=symbol,
                    price=ticker.last,
                    volume_24h=getattr(ticker, 'volume_24h', 0),
                    change_24h=0.0,
                    bid=ticker.bid,
                    ask=ticker.ask,
                    timestamp=datetime.now(),
                    source="ibkr",
                )
                self.latest_ticks[symbol] = tick
                return tick
        except Exception as e:
            self.logger.debug(f"IBKR ticker failed for {symbol}: {e}")
        return None

    async def _ndax_get_ticker(self, symbol: str) -> Optional[MarketTick]:
        """Fetch a crypto/CAD quote from NDAX."""
        if not self._ndax_connector:
            return None
        try:
            pair = f"{symbol}/CAD"
            ticker = await self._ndax_connector.get_ticker(pair)
            if ticker and ticker.last > 0:
                tick = MarketTick(
                    symbol=pair,
                    price=ticker.last,
                    volume_24h=getattr(ticker, 'volume_24h', 0),
                    change_24h=0.0,
                    bid=ticker.bid,
                    ask=ticker.ask,
                    timestamp=datetime.now(),
                    source="ndax",
                )
                self.latest_ticks[pair] = tick
                return tick
        except Exception as e:
            self.logger.debug(f"NDAX ticker failed for {symbol}: {e}")
        return None

    async def get_market_snapshot(self, symbols: List[str]) -> Dict[str, MarketTick]:
        """
        Get current market snapshot with graceful degradation.
        
        Routes:
        - Equity/ETF symbols → IBKR connector
        - Crypto symbols → CoinGecko (primary), NDAX CAD (secondary)
        - Falls back to cached data if all sources unavailable.
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

        result: Dict[str, MarketTick] = {}

        # Partition symbols into equity vs crypto
        equity_syms = [s for s in symbols if is_equity(s)]
        crypto_syms = [s for s in symbols if not is_equity(s)]

        # ── IBKR path for equities / ETFs ──
        for sym in equity_syms:
            tick = await self._ibkr_get_ticker(sym.upper())
            if tick:
                result[tick.symbol] = tick

        # ── CoinGecko path for crypto ──
        if crypto_syms:
            coin_ids = [symbol_map.get(s.upper(), s.lower()) for s in crypto_syms]
            try:
                ticks = await self.coingecko.get_prices_batch(coin_ids)
                for tick in ticks:
                    result[tick.symbol] = tick
                    self.latest_ticks[tick.symbol] = tick
            except Exception as e:
                self.logger.warning(f"CoinGecko failed: {e}")

        # ── NDAX secondary path for crypto/CAD ──
        if self._ndax_connector:
            for sym in crypto_syms:
                su = sym.upper()
                if su not in result and f"{su}/USDT" not in result:
                    tick = await self._ndax_get_ticker(su)
                    if tick:
                        result[tick.symbol] = tick

        # ── Cache fallback for anything still missing ──
        all_requested = equity_syms + crypto_syms
        missing = [s for s in all_requested
                   if s not in result and s.upper() not in result
                   and f"{s}/USDT" not in result and f"{s}/USD" not in result
                   and f"{s.upper()}/CAD" not in result]
        if missing:
            for symbol in missing:
                for key in [symbol, symbol.upper(), f"{symbol}/USDT", f"{symbol}/USD", f"{symbol.upper()}/CAD"]:
                    if key in self.latest_ticks:
                        tick = self.latest_ticks[key]
                        age = (datetime.now() - tick.timestamp).total_seconds()
                        if age < 60:
                            result[tick.symbol] = tick
                            break

        if not result and symbols:
            self.logger.error(f"No data for {symbols} from any source")

        return result

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
        """Test."""
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
