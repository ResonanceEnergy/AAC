#!/usr/bin/env python3
"""
API Integration Hub - Phase 2
=============================
Centralized API management for all external services required for live trading.
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audit_logger import get_audit_logger
from shared.config_loader import get_config


@dataclass
class APIEndpoint:
    """API endpoint configuration"""
    name: str
    base_url: str
    auth_type: str  # 'bearer', 'api_key', 'basic', 'none'
    auth_header: str = ''
    auth_value: str = ''
    rate_limit: int = 60  # requests per minute
    timeout: int = 30
    retries: int = 3
    enabled: bool = True


@dataclass
class APIResponse:
    """Standardized API response"""
    success: bool
    data: Any = None
    error: str = ''
    status_code: int = 0
    response_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class APIClient:
    """Base API client with rate limiting and error handling"""

    def __init__(self, endpoint: APIEndpoint):
        self.endpoint = endpoint
        self.logger = logging.getLogger(f"APIClient.{endpoint.name}")
        self.audit_logger = get_audit_logger()

        # Rate limiting
        self.requests_made: list[datetime] = []
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.endpoint.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    async def close(self) -> None:
        """Close any persistent session."""
        if self.session:
            await self.session.close()
            self.session = None

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        window_start = now - timedelta(minutes=1)

        # Clean old requests
        self.requests_made = [req for req in self.requests_made if req > window_start]

        if len(self.requests_made) >= self.endpoint.rate_limit:
            self.logger.warning(f"Rate limit exceeded for {self.endpoint.name}")
            return False

        self.requests_made.append(now)
        return True

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        headers = {}

        if self.endpoint.auth_type == 'bearer' and self.endpoint.auth_value:
            headers['Authorization'] = f'Bearer {self.endpoint.auth_value}'
        elif self.endpoint.auth_type == 'api_key' and self.endpoint.auth_header and self.endpoint.auth_value:
            headers[self.endpoint.auth_header] = self.endpoint.auth_value
        elif self.endpoint.auth_type == 'basic':
            # Basic auth would be handled differently
            pass

        return headers

    async def _make_request(self, method: str, url: str, **kwargs) -> APIResponse:
        """Make HTTP request with error handling and retries"""
        if not self._check_rate_limit():
            return APIResponse(
                success=False,
                error="Rate limit exceeded",
                status_code=429
            )

        start_time = time.time()
        last_error = None
        owns_session = self.session is None
        session = self.session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.endpoint.timeout)
        )

        try:
            for attempt in range(self.endpoint.retries):
                try:
                    headers = self._get_auth_headers()
                    if 'headers' in kwargs:
                        headers.update(kwargs['headers'])
                    kwargs['headers'] = headers

                    async with session.request(method, url, **kwargs) as response:
                        response_time = time.time() - start_time

                        if response.status == 200:
                            try:
                                data = await response.json()
                            except (json.JSONDecodeError, ValueError) as e:
                                self.logger.warning(f"JSON parse error: {e}")
                                data = await response.text()

                            await self.audit_logger.log_event(
                                category="api",
                                action="api_call_success",
                                details={
                                    "endpoint": self.endpoint.name,
                                    "method": method,
                                    "url": url,
                                    "response_time": response_time
                                }
                            )

                            return APIResponse(
                                success=True,
                                data=data,
                                status_code=response.status,
                                response_time=response_time
                            )

                        error_text = await response.text()
                        last_error = f"HTTP {response.status}: {error_text}"

                        if response.status == 429:
                            await asyncio.sleep(60)
                            continue
                        if response.status >= 500:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        break

                except asyncio.TimeoutError:
                    last_error = "Request timeout"
                    await asyncio.sleep(2 ** attempt)
                    continue
                except Exception as e:
                    last_error = str(e)
                    await asyncio.sleep(2 ** attempt)
                    continue
        finally:
            if owns_session:
                await session.close()

        # All retries failed
        await self.audit_logger.log_event(
            category="api",
            action="api_call_failed",
            details={
                "endpoint": self.endpoint.name,
                "method": method,
                "url": url,
                "error": last_error,
                "attempts": self.endpoint.retries
            },
            severity="warning"  # type: ignore[arg-type]
        )

        return APIResponse(
            success=False,
            error=last_error or "Unknown error",
            status_code=500,
            response_time=time.time() - start_time
        )


class EthereumDEXClient(APIClient):
    """Ethereum DEX trading client using Web3"""

    def __init__(self, config):
        # Web3 doesn't use HTTP endpoints like other APIs
        endpoint = APIEndpoint(
            name="ethereum_dex",
            base_url=config.eth_rpc_url,
            auth_type="none"
        )
        super().__init__(endpoint)
        self.config = config
        self.web3 = None
        self.account = None

        self._initialize_web3()

    def _initialize_web3(self):
        """Initialize Web3 connection"""
        try:
            from web3 import Web3
            self.web3 = Web3(Web3.HTTPProvider(self.config.eth_rpc_url))

            if self.config.eth_private_key:
                from eth_account import Account
                self.account = Account.from_key(self.config.eth_private_key)  # type: ignore[call-arg]
                self.logger.info("Ethereum DEX client initialized with private key")
            else:
                self.logger.warning("Ethereum DEX client initialized without private key")

        except ImportError:
            self.logger.error("Web3.py not installed. Install with: pip install web3 eth-account")
        except Exception as e:
            self.logger.error(f"Failed to initialize Web3: {e}")

    async def get_balance(self, address: Optional[str] = None) -> APIResponse:
        """Get ETH balance for address"""
        if not self.web3:
            return APIResponse(success=False, error="Web3 not initialized")

        try:
            target_address = address or self.account.address
            balance_wei = self.web3.eth.get_balance(target_address)
            balance_eth = self.web3.from_wei(balance_wei, 'ether')

            return APIResponse(
                success=True,
                data={
                    "address": target_address,
                    "balance_wei": balance_wei,
                    "balance_eth": float(balance_eth)
                }
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    async def get_gas_price(self) -> APIResponse:
        """Get current gas price"""
        if not self.web3:
            return APIResponse(success=False, error="Web3 not initialized")

        try:
            gas_price = self.web3.eth.gas_price
            gas_price_gwei = self.web3.from_wei(gas_price, 'gwei')

            return APIResponse(
                success=True,
                data={
                    "gas_price_wei": gas_price,
                    "gas_price_gwei": float(gas_price_gwei)
                }
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))

    async def estimate_gas(self, to_address: str, value_eth: float = 0) -> APIResponse:
        """Estimate gas for transaction"""
        if not self.web3 or not self.account:
            return APIResponse(success=False, error="Web3 or account not initialized")

        try:
            value_wei = self.web3.to_wei(value_eth, 'ether')

            # Estimate gas for simple transfer
            gas_estimate = self.web3.eth.estimate_gas({
                'to': to_address,
                'from': self.account.address,
                'value': value_wei
            })

            return APIResponse(
                success=True,
                data={
                    "gas_estimate": gas_estimate,
                    "gas_cost_wei": gas_estimate * self.web3.eth.gas_price,
                    "gas_cost_eth": float(self.web3.from_wei(gas_estimate * self.web3.eth.gas_price, 'ether'))
                }
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))


class BigBrainAIClient(APIClient):
    """BigBrain Intelligence API client"""

    def __init__(self, config):
        endpoint = APIEndpoint(
            name="bigbrain_ai",
            base_url=config.bigbrain_url,
            auth_type="bearer",
            auth_value=config.bigbrain_token,
            rate_limit=30  # Conservative rate limit
        )
        super().__init__(endpoint)
        self.config = config

    async def get_market_analysis(self, symbol: str, timeframe: str = "1h") -> APIResponse:
        """Get AI-powered market analysis"""
        url = f"{self.endpoint.base_url}/market-analysis"
        params = {"symbol": symbol, "timeframe": timeframe}

        return await self._make_request("GET", url, params=params)

    async def get_arbitrage_opportunities(self, exchanges: Optional[List[str]] = None) -> APIResponse:
        """Get AI-detected arbitrage opportunities"""
        url = f"{self.endpoint.base_url}/arbitrage-opportunities"
        params = {}
        if exchanges:
            params["exchanges"] = ",".join(exchanges)

        return await self._make_request("GET", url, params=params)

    async def get_sentiment_analysis(self, symbol: str) -> APIResponse:
        """Get AI sentiment analysis for symbol"""
        url = f"{self.endpoint.base_url}/sentiment-analysis"
        params = {"symbol": symbol}

        return await self._make_request("GET", url, params=params)

    async def submit_trading_signal(self, signal_data: Dict[str, Any]) -> APIResponse:
        """Submit trading signal for AI processing"""
        url = f"{self.endpoint.base_url}/trading-signals"

        return await self._make_request("POST", url, json=signal_data)


class CoinMarketCapClient(APIClient):
    """CoinMarketCap API client"""

    def __init__(self, config):
        endpoint = APIEndpoint(
            name="coinmarketcap",
            base_url="https://pro-api.coinmarketcap.com/v1",
            auth_type="api_key",
            auth_header="X-CMC_PRO_API_KEY",
            auth_value=config.coinmarketcap_key,
            rate_limit=10  # CMC free tier limit
        )
        super().__init__(endpoint)
        self.config = config

    async def get_cryptocurrency_listings(self, limit: int = 100) -> APIResponse:
        """Get cryptocurrency listings"""
        url = f"{self.endpoint.base_url}/cryptocurrency/listings/latest"
        params = {
            "start": "1",
            "limit": str(limit),
            "convert": "USD"
        }

        return await self._make_request("GET", url, params=params)

    async def get_cryptocurrency_quotes(self, symbols: List[str]) -> APIResponse:
        """Get quotes for specific cryptocurrencies"""
        url = f"{self.endpoint.base_url}/cryptocurrency/quotes/latest"
        params = {
            "symbol": ",".join(symbols),
            "convert": "USD"
        }

        return await self._make_request("GET", url, params=params)

    async def get_global_metrics(self) -> APIResponse:
        """Get global cryptocurrency metrics"""
        url = f"{self.endpoint.base_url}/global-metrics/quotes/latest"
        params = {"convert": "USD"}

        return await self._make_request("GET", url, params=params)


class NewsAPIClient(APIClient):
    """News API client for financial news"""

    def __init__(self, config):
        endpoint = APIEndpoint(
            name="news_api",
            base_url="https://newsapi.org/v2",
            auth_type="api_key",
            auth_header="X-API-Key",
            auth_value=config.news_api_key,
            rate_limit=100  # NewsAPI rate limit
        )
        super().__init__(endpoint)
        self.config = config

    async def get_financial_news(self, query: str = "cryptocurrency OR bitcoin OR ethereum",
                                language: str = "en", page_size: int = 20) -> APIResponse:
        """Get financial news articles"""
        url = f"{self.endpoint.base_url}/everything"
        params = {
            "q": query,
            "language": language,
            "pageSize": page_size,
            "sortBy": "publishedAt"
        }

        return await self._make_request("GET", url, params=params)

    async def get_top_headlines(self, country: str = "us", category: str = "business") -> APIResponse:
        """Get top headlines"""
        url = f"{self.endpoint.base_url}/top-headlines"
        params = {
            "country": country,
            "category": category,
            "pageSize": 20
        }

        return await self._make_request("GET", url, params=params)


class TwitterAPIClient(APIClient):
    """Twitter API client for social sentiment"""

    def __init__(self, config):
        endpoint = APIEndpoint(
            name="twitter_api",
            base_url="https://api.twitter.com/2",
            auth_type="bearer",
            auth_value=config.twitter_bearer,
            rate_limit=300  # Twitter API v2 rate limits
        )
        super().__init__(endpoint)
        self.config = config

    async def search_tweets(self, query: str, max_results: int = 10) -> APIResponse:
        """Search for tweets matching query"""
        url = f"{self.endpoint.base_url}/tweets/search/recent"
        params = {
            "query": query,
            "max_results": max_results,
            "tweet.fields": "created_at,public_metrics,text,author_id"
        }

        return await self._make_request("GET", url, params=params)

    async def get_crypto_sentiment(self, symbol: str) -> APIResponse:
        """Get sentiment analysis for cryptocurrency"""
        # Search for recent tweets about the symbol
        query = f"#{symbol} OR ${symbol} -is:retweet"
        return await self.search_tweets(query, max_results=50)


class RedditAPIClient(APIClient):
    """Reddit API client for social sentiment"""

    def __init__(self, config):
        endpoint = APIEndpoint(
            name="reddit_api",
            base_url="https://www.reddit.com",
            auth_type="basic",  # Reddit uses basic auth with client credentials
            rate_limit=60  # Reddit rate limit
        )
        super().__init__(endpoint)
        self.config = config
        self.access_token = None
        self.token_expires = None

    async def _get_access_token(self) -> bool:
        """Get Reddit API access token"""
        if self.token_expires and datetime.now() < self.token_expires:
            return True  # Token still valid

        try:
            auth = aiohttp.BasicAuth(self.config.reddit_client_id, self.config.reddit_client_secret)

            async with self.session.post(  # type: ignore[union-attr]
                "https://www.reddit.com/api/v1/access_token",
                data={"grant_type": "client_credentials"},
                auth=auth
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.access_token = data.get("access_token")
                    expires_in = data.get("expires_in", 3600)
                    self.token_expires = datetime.now() + timedelta(seconds=expires_in)
                    return True
                else:
                    self.logger.error(f"Reddit auth failed: {response.status}")
                    return False

        except Exception as e:
            self.logger.error(f"Reddit auth error: {e}")
            return False

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get Reddit authentication headers"""
        headers = {"User-Agent": self.config.reddit_user_agent}
        if self.access_token:
            headers["Authorization"] = f"bearer {self.access_token}"
        return headers

    async def get_subreddit_posts(self, subreddit: str, limit: int = 25) -> APIResponse:
        """Get posts from subreddit"""
        if not await self._get_access_token():
            return APIResponse(success=False, error="Failed to authenticate with Reddit")

        url = f"{self.endpoint.base_url}/r/{subreddit}/hot.json"
        params = {"limit": limit}

        return await self._make_request("GET", url, params=params)

    async def get_crypto_sentiment(self, symbol: str) -> APIResponse:
        """Get sentiment from crypto-related subreddits"""
        subreddits = ["cryptocurrency", "bitcoin", "ethereum", "CryptoMarkets"]
        all_posts = []

        for subreddit in subreddits:
            response = await self.get_subreddit_posts(subreddit, limit=10)
            if response.success:
                posts = response.data.get("data", {}).get("children", [])
                # Filter posts mentioning the symbol
                relevant_posts = [
                    post for post in posts
                    if symbol.lower() in post.get("data", {}).get("title", "").lower() or
                       symbol.lower() in post.get("data", {}).get("selftext", "").lower()
                ]
                all_posts.extend(relevant_posts[:5])  # Max 5 per subreddit

        return APIResponse(
            success=True,
            data={"posts": all_posts, "total_found": len(all_posts)}
        )


class NCCCoordinatorClient(APIClient):
    """Neural Coordination Center client"""

    def __init__(self, config):
        endpoint = APIEndpoint(
            name="ncc_coordinator",
            base_url=config.ncc_endpoint,
            auth_type="bearer",
            auth_value=config.ncc_token,
            rate_limit=60
        )
        super().__init__(endpoint)
        self.config = config

    async def get_coordination_status(self) -> APIResponse:
        """Get NCC coordination status"""
        url = f"{self.endpoint.base_url}/status"
        return await self._make_request("GET", url)

    async def submit_agent_action(self, agent_id: str, action: Dict[str, Any]) -> APIResponse:
        """Submit agent action for coordination"""
        url = f"{self.endpoint.base_url}/agents/{agent_id}/actions"
        return await self._make_request("POST", url, json=action)

    async def get_arbitrage_signals(self) -> APIResponse:
        """Get coordinated arbitrage signals"""
        url = f"{self.endpoint.base_url}/signals/arbitrage"
        return await self._make_request("GET", url)


class KYCProviderClient(APIClient):
    """KYC/Identity verification provider client"""

    def __init__(self, config):
        endpoint = APIEndpoint(
            name="kyc_provider",
            base_url=config.kyc_provider_url,
            auth_type="api_key",
            auth_header="X-API-Key",
            auth_value=config.kyc_provider_key,
            rate_limit=10  # Conservative for KYC
        )
        super().__init__(endpoint)
        self.config = config

    async def verify_identity(self, user_data: Dict[str, Any]) -> APIResponse:
        """Submit identity verification request"""
        url = f"{self.endpoint.base_url}/verify"
        return await self._make_request("POST", url, json=user_data)

    async def check_verification_status(self, verification_id: str) -> APIResponse:
        """Check verification status"""
        url = f"{self.endpoint.base_url}/verification/{verification_id}"
        return await self._make_request("GET", url)


class FREDClient(APIClient):
    """Federal Reserve Economic Data (FRED) API client"""

    def __init__(self, config):
        endpoint = APIEndpoint(
            name="fred",
            base_url="https://api.stlouisfed.org/fred",
            auth_type="none",  # API key passed as query param
            rate_limit=120
        )
        super().__init__(endpoint)
        self.api_key = config.fred_key

    async def get_series(self, series_id: str, limit: int = 100) -> APIResponse:
        """Get observations for a FRED series (e.g. GDP, CPI, FEDFUNDS)"""
        url = f"{self.endpoint.base_url}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit
        }
        return await self._make_request("GET", url, params=params)

    async def search_series(self, query: str, limit: int = 20) -> APIResponse:
        """Search FRED for economic data series"""
        url = f"{self.endpoint.base_url}/series/search"
        params = {
            "search_text": query,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": limit
        }
        return await self._make_request("GET", url, params=params)

    async def get_interest_rates(self) -> APIResponse:
        """Get key interest rates (Fed Funds, 10Y Treasury, 30Y Mortgage)"""
        rates = {}
        series_ids = {
            "fed_funds_rate": "FEDFUNDS",
            "treasury_10y": "DGS10",
            "treasury_2y": "DGS2",
            "mortgage_30y": "MORTGAGE30US",
            "prime_rate": "DPRIME"
        }
        for name, series_id in series_ids.items():
            result = await self.get_series(series_id, limit=1)
            if result.success and result.data:
                obs = result.data.get("observations", [])
                if obs:
                    rates[name] = {
                        "value": obs[0].get("value"),
                        "date": obs[0].get("date")
                    }
        return APIResponse(success=bool(rates), data=rates)

    async def get_economic_indicators(self) -> APIResponse:
        """Get key economic indicators (GDP, CPI, unemployment)"""
        indicators = {}
        series_ids = {
            "gdp_growth": "A191RL1Q225SBEA",
            "cpi": "CPIAUCSL",
            "unemployment": "UNRATE",
            "consumer_sentiment": "UMCSENT",
            "industrial_production": "INDPRO"
        }
        for name, series_id in series_ids.items():
            result = await self.get_series(series_id, limit=1)
            if result.success and result.data:
                obs = result.data.get("observations", [])
                if obs:
                    indicators[name] = {
                        "value": obs[0].get("value"),
                        "date": obs[0].get("date")
                    }
        return APIResponse(success=bool(indicators), data=indicators)


class PolygonClient(APIClient):
    """Polygon.io API client for market data"""

    def __init__(self, config):
        endpoint = APIEndpoint(
            name="polygon",
            base_url="https://api.polygon.io",
            auth_type="none",  # API key passed as query param
            rate_limit=5  # Free tier: 5/min
        )
        super().__init__(endpoint)
        self.api_key = config.polygon_key

    async def get_ticker_snapshot(self, symbol: str) -> APIResponse:
        """Get real-time snapshot for a ticker"""
        url = f"{self.endpoint.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        params = {"apiKey": self.api_key}
        return await self._make_request("GET", url, params=params)

    async def get_aggregates(self, symbol: str, multiplier: int = 1,
                             timespan: str = "day", from_date: str = "",
                             to_date: str = "") -> APIResponse:
        """Get aggregate bars (OHLCV) for a symbol"""
        url = f"{self.endpoint.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"apiKey": self.api_key, "adjusted": "true", "sort": "asc"}
        return await self._make_request("GET", url, params=params)

    async def get_previous_close(self, symbol: str) -> APIResponse:
        """Get previous day's close for a symbol"""
        url = f"{self.endpoint.base_url}/v2/aggs/ticker/{symbol}/prev"
        params = {"apiKey": self.api_key}
        return await self._make_request("GET", url, params=params)

    async def get_crypto_snapshot(self, symbol: str = "X:BTCUSD") -> APIResponse:
        """Get crypto ticker snapshot"""
        url = f"{self.endpoint.base_url}/v2/snapshot/locale/global/markets/crypto/tickers/{symbol}"
        params = {"apiKey": self.api_key}
        return await self._make_request("GET", url, params=params)


class FinnhubClient(APIClient):
    """Finnhub API client for stock data and news"""

    def __init__(self, config):
        endpoint = APIEndpoint(
            name="finnhub",
            base_url="https://finnhub.io/api/v1",
            auth_type="none",
            rate_limit=60  # Free: 60/min
        )
        super().__init__(endpoint)
        self.api_key = config.finnhub_key

    async def get_quote(self, symbol: str) -> APIResponse:
        """Get real-time quote"""
        url = f"{self.endpoint.base_url}/quote"
        params = {"symbol": symbol, "token": self.api_key}
        return await self._make_request("GET", url, params=params)

    async def get_company_news(self, symbol: str, from_date: str = "",
                                to_date: str = "") -> APIResponse:
        """Get company-specific news"""
        url = f"{self.endpoint.base_url}/company-news"
        params = {"symbol": symbol, "from": from_date, "to": to_date, "token": self.api_key}
        return await self._make_request("GET", url, params=params)

    async def get_earnings_calendar(self, from_date: str = "",
                                     to_date: str = "") -> APIResponse:
        """Get earnings calendar"""
        url = f"{self.endpoint.base_url}/calendar/earnings"
        params = {"from": from_date, "to": to_date, "token": self.api_key}
        return await self._make_request("GET", url, params=params)

    async def get_market_news(self, category: str = "general") -> APIResponse:
        """Get general market news"""
        url = f"{self.endpoint.base_url}/news"
        params = {"category": category, "token": self.api_key}
        return await self._make_request("GET", url, params=params)

    async def get_insider_transactions(self, symbol: str) -> APIResponse:
        """Get insider transactions"""
        url = f"{self.endpoint.base_url}/stock/insider-transactions"
        params = {"symbol": symbol, "token": self.api_key}
        return await self._make_request("GET", url, params=params)


class EODHDHubClient(APIClient):
    """EODHD API client for global market data"""

    def __init__(self, config):
        endpoint = APIEndpoint(
            name="eodhd",
            base_url="https://eodhd.com/api",
            auth_type="none",
            rate_limit=100
        )
        super().__init__(endpoint)
        self.api_key = config.eodhd_key

    async def get_eod_data(self, symbol: str, from_date: str = "",
                           to_date: str = "") -> APIResponse:
        """Get end-of-day historical data"""
        url = f"{self.endpoint.base_url}/eod/{symbol}"
        params = {"api_token": self.api_key, "fmt": "json"}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._make_request("GET", url, params=params)

    async def get_real_time(self, symbol: str) -> APIResponse:
        """Get real-time price"""
        url = f"{self.endpoint.base_url}/real-time/{symbol}"
        params = {"api_token": self.api_key, "fmt": "json"}
        return await self._make_request("GET", url, params=params)

    async def get_fundamentals(self, symbol: str) -> APIResponse:
        """Get fundamental data for a stock"""
        url = f"{self.endpoint.base_url}/fundamentals/{symbol}"
        params = {"api_token": self.api_key}
        return await self._make_request("GET", url, params=params)


class TradeStieSentimentClient(APIClient):
    """TradeStie Reddit Sentiment API client"""

    def __init__(self, config):
        endpoint = APIEndpoint(
            name="tradestie_sentiment",
            base_url="https://tradestie.com/api/v1/apps/reddit",
            auth_type="none",
            rate_limit=20  # 20/min
        )
        super().__init__(endpoint)

    async def get_wsb_sentiment(self) -> APIResponse:
        """Get WallStreetBets top discussed tickers with sentiment"""
        url = self.endpoint.base_url
        return await self._make_request("GET", url)

    async def get_ticker_sentiment(self, ticker: str) -> APIResponse:
        """Get sentiment for a specific ticker"""
        result = await self.get_wsb_sentiment()
        if result.success and result.data:
            for item in result.data:
                if item.get("ticker", "").upper() == ticker.upper():
                    return APIResponse(success=True, data=item)
        return APIResponse(success=False, error=f"Ticker {ticker} not in WSB top discussed")


class EtherscanClient(APIClient):
    """Etherscan API client for blockchain data"""

    def __init__(self, config):
        endpoint = APIEndpoint(
            name="etherscan",
            base_url="https://api.etherscan.io/api",
            auth_type="none",
            rate_limit=5  # Free: 5/sec
        )
        super().__init__(endpoint)
        self.api_key = config.etherscan_key

    async def get_eth_price(self) -> APIResponse:
        """Get current ETH price"""
        url = self.endpoint.base_url
        params = {"module": "stats", "action": "ethprice", "apikey": self.api_key}
        return await self._make_request("GET", url, params=params)

    async def get_gas_oracle(self) -> APIResponse:
        """Get gas price oracle"""
        url = self.endpoint.base_url
        params = {"module": "gastracker", "action": "gasoracle", "apikey": self.api_key}
        return await self._make_request("GET", url, params=params)

    async def get_balance(self, address: str) -> APIResponse:
        """Get ETH balance for an address"""
        url = self.endpoint.base_url
        params = {
            "module": "account", "action": "balance",
            "address": address, "tag": "latest", "apikey": self.api_key
        }
        return await self._make_request("GET", url, params=params)

    async def get_token_balance(self, address: str, contract: str) -> APIResponse:
        """Get ERC-20 token balance"""
        url = self.endpoint.base_url
        params = {
            "module": "account", "action": "tokenbalance",
            "contractaddress": contract, "address": address,
            "tag": "latest", "apikey": self.api_key
        }
        return await self._make_request("GET", url, params=params)


# ═══════════════════════════════════════════════════════════════════════════
# POLYMARKET — Prediction Market APIs (Gamma + Data + CLOB)
# ═══════════════════════════════════════════════════════════════════════════


class PolymarketGammaClient(APIClient):
    """Polymarket Gamma API — markets, events, tags, search (public, no auth)"""

    def __init__(self, config=None):
        endpoint = APIEndpoint(
            name="polymarket_gamma",
            base_url="https://gamma-api.polymarket.com",
            auth_type="none",
            rate_limit=60
        )
        super().__init__(endpoint)

    async def get_events(self, active: bool = True, closed: bool = False,
                         limit: int = 100, offset: int = 0,
                         order: str = "volume_24hr",
                         ascending: bool = False) -> APIResponse:
        """Get prediction market events (containers of markets)"""
        url = f"{self.endpoint.base_url}/events"
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower()
        }
        return await self._make_request("GET", url, params=params)

    async def get_event_by_slug(self, slug: str) -> APIResponse:
        """Get a specific event by its URL slug"""
        url = f"{self.endpoint.base_url}/events/slug/{slug}"
        return await self._make_request("GET", url)

    async def get_markets(self, active: bool = True, closed: bool = False,
                          limit: int = 100, offset: int = 0) -> APIResponse:
        """Get individual prediction markets"""
        url = f"{self.endpoint.base_url}/markets"
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset
        }
        return await self._make_request("GET", url, params=params)

    async def get_market_by_slug(self, slug: str) -> APIResponse:
        """Get a specific market by slug"""
        url = f"{self.endpoint.base_url}/markets/slug/{slug}"
        return await self._make_request("GET", url)

    async def get_market_by_id(self, condition_id: str) -> APIResponse:
        """Get a specific market by condition ID"""
        url = f"{self.endpoint.base_url}/markets/{condition_id}"
        return await self._make_request("GET", url)

    async def get_tags(self) -> APIResponse:
        """Get all available market tags/categories"""
        url = f"{self.endpoint.base_url}/tags"
        return await self._make_request("GET", url)

    async def get_events_by_tag(self, tag_id: int, active: bool = True,
                                limit: int = 50) -> APIResponse:
        """Get events filtered by tag (e.g., crypto, politics, sports)"""
        url = f"{self.endpoint.base_url}/events"
        params = {
            "tag_id": tag_id,
            "active": str(active).lower(),
            "closed": "false",
            "limit": limit
        }
        return await self._make_request("GET", url, params=params)

    async def search_markets(self, query: str, limit: int = 20) -> APIResponse:
        """Search markets by keyword"""
        url = f"{self.endpoint.base_url}/markets"
        params = {"slug": query, "limit": limit}
        return await self._make_request("GET", url, params=params)


class PolymarketDataClient(APIClient):
    """Polymarket Data API — positions, trades, activity, leaderboards (public)"""

    def __init__(self, config=None):
        endpoint = APIEndpoint(
            name="polymarket_data",
            base_url="https://data-api.polymarket.com",
            auth_type="none",
            rate_limit=60
        )
        super().__init__(endpoint)

    async def get_market_trades(self, condition_id: str, limit: int = 100) -> APIResponse:
        """Get recent trades for a market"""
        url = f"{self.endpoint.base_url}/trades"
        params = {"market": condition_id, "limit": limit}
        return await self._make_request("GET", url, params=params)

    async def get_market_timeseries(self, condition_id: str,
                                     fidelity: int = 60) -> APIResponse:
        """Get price timeseries for a market (fidelity in minutes)"""
        url = f"{self.endpoint.base_url}/timeseries"
        params = {"market": condition_id, "fidelity": fidelity}
        return await self._make_request("GET", url, params=params)

    async def get_global_activity(self, limit: int = 50) -> APIResponse:
        """Get global platform activity"""
        url = f"{self.endpoint.base_url}/activity"
        params = {"limit": limit}
        return await self._make_request("GET", url, params=params)

    async def get_leaderboard(self, limit: int = 50) -> APIResponse:
        """Get platform leaderboard"""
        url = f"{self.endpoint.base_url}/leaderboard"
        params = {"limit": limit}
        return await self._make_request("GET", url, params=params)


class PolymarketCLOBClient(APIClient):
    """Polymarket CLOB API — orderbook, pricing, midpoints (public read endpoints)

    Trading endpoints (order placement/cancellation) require py-clob-client
    SDK with wallet auth.  This client covers read-only CLOB data.
    """

    def __init__(self, config=None):
        endpoint = APIEndpoint(
            name="polymarket_clob",
            base_url="https://clob.polymarket.com",
            auth_type="none",
            rate_limit=60
        )
        super().__init__(endpoint)

    async def get_midpoint(self, token_id: str) -> APIResponse:
        """Get midpoint price for a token"""
        url = f"{self.endpoint.base_url}/midpoint"
        params = {"token_id": token_id}
        return await self._make_request("GET", url, params=params)

    async def get_price(self, token_id: str, side: str = "BUY") -> APIResponse:
        """Get best price (BUY or SELL) for a token"""
        url = f"{self.endpoint.base_url}/price"
        params = {"token_id": token_id, "side": side}
        return await self._make_request("GET", url, params=params)

    async def get_spread(self, token_id: str) -> APIResponse:
        """Get bid-ask spread for a token"""
        url = f"{self.endpoint.base_url}/spread"
        params = {"token_id": token_id}
        return await self._make_request("GET", url, params=params)

    async def get_order_book(self, token_id: str) -> APIResponse:
        """Get full order book for a token"""
        url = f"{self.endpoint.base_url}/book"
        params = {"token_id": token_id}
        return await self._make_request("GET", url, params=params)

    async def get_simplified_markets(self) -> APIResponse:
        """Get simplified market list from CLOB"""
        url = f"{self.endpoint.base_url}/simplified-markets"
        return await self._make_request("GET", url)

    async def get_last_trade_price(self, token_id: str) -> APIResponse:
        """Get last trade price for a token"""
        url = f"{self.endpoint.base_url}/last-trade-price"
        params = {"token_id": token_id}
        return await self._make_request("GET", url, params=params)

    async def get_prices_history(self, token_id: str,
                                  interval: str = "1d",
                                  fidelity: int = 60) -> APIResponse:
        """Get historical price data for a token"""
        url = f"{self.endpoint.base_url}/prices-history"
        params = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity
        }
        return await self._make_request("GET", url, params=params)


class APIIntegrationHub:
    """Central hub for all API integrations"""

    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger("APIIntegrationHub")
        self.audit_logger = get_audit_logger()

        # API clients
        self.clients = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize all API clients"""
        # Ethereum DEX
        if self.config.eth_private_key:
            self.clients["ethereum_dex"] = EthereumDEXClient(self.config)

        # BigBrain AI
        if self.config.bigbrain_token:
            self.clients["bigbrain_ai"] = BigBrainAIClient(self.config)

        # CoinMarketCap
        if self.config.coinmarketcap_key:
            self.clients["coinmarketcap"] = CoinMarketCapClient(self.config)

        # News API
        if self.config.news_api_key:
            self.clients["news_api"] = NewsAPIClient(self.config)

        # Twitter API
        if self.config.twitter_bearer:
            self.clients["twitter_api"] = TwitterAPIClient(self.config)

        # Reddit API
        if self.config.reddit_client_id and self.config.reddit_client_secret:
            self.clients["reddit_api"] = RedditAPIClient(self.config)

        # NCC Coordinator
        if self.config.ncc_token:
            self.clients["ncc_coordinator"] = NCCCoordinatorClient(self.config)

        # KYC Provider
        if self.config.kyc_provider_key:
            self.clients["kyc_provider"] = KYCProviderClient(self.config)

        # FRED Economic Data
        if self.config.fred_key:
            self.clients["fred"] = FREDClient(self.config)

        # Polygon.io
        if self.config.polygon_key:
            self.clients["polygon"] = PolygonClient(self.config)

        # Finnhub
        if self.config.finnhub_key:
            self.clients["finnhub"] = FinnhubClient(self.config)

        # EODHD
        if self.config.eodhd_key:
            self.clients["eodhd"] = EODHDHubClient(self.config)

        # TradeStie Reddit Sentiment
        self.clients["tradestie_sentiment"] = TradeStieSentimentClient(self.config)

        # Etherscan
        if self.config.etherscan_key:
            self.clients["etherscan"] = EtherscanClient(self.config)

        # Polymarket (all public, no key required)
        self.clients["polymarket_gamma"] = PolymarketGammaClient(self.config)
        self.clients["polymarket_data"] = PolymarketDataClient(self.config)
        self.clients["polymarket_clob"] = PolymarketCLOBClient(self.config)

        self.logger.info(f"Initialized {len(self.clients)} API clients")

    def get_client(self, name: str) -> Optional[APIClient]:
        """Get API client by name"""
        return self.clients.get(name)

    def get_available_clients(self) -> List[str]:
        """Get list of available API clients"""
        return list(self.clients.keys())

    async def test_all_connections(self) -> Dict[str, APIResponse]:
        """Test connectivity to all configured APIs"""
        results = {}

        # Test Ethereum DEX
        if "ethereum_dex" in self.clients:
            eth_client = self.clients["ethereum_dex"]
            results["ethereum_dex"] = await eth_client.get_gas_price()

        # Test BigBrain AI
        if "bigbrain_ai" in self.clients:
            bb_client = self.clients["bigbrain_ai"]
            results["bigbrain_ai"] = await bb_client.get_arbitrage_opportunities()

        # Test CoinMarketCap
        if "coinmarketcap" in self.clients:
            cmc_client = self.clients["coinmarketcap"]
            results["coinmarketcap"] = await cmc_client.get_global_metrics()

        # Test News API
        if "news_api" in self.clients:
            news_client = self.clients["news_api"]
            results["news_api"] = await news_client.get_top_headlines()

        # Test Twitter API
        if "twitter_api" in self.clients:
            twitter_client = self.clients["twitter_api"]
            results["twitter_api"] = await twitter_client.search_tweets("bitcoin", max_results=1)

        # Test Reddit API
        if "reddit_api" in self.clients:
            reddit_client = self.clients["reddit_api"]
            results["reddit_api"] = await reddit_client.get_subreddit_posts("cryptocurrency", limit=1)

        # Test NCC Coordinator
        if "ncc_coordinator" in self.clients:
            ncc_client = self.clients["ncc_coordinator"]
            results["ncc_coordinator"] = await ncc_client.get_coordination_status()

        # Test KYC Provider
        if "kyc_provider" in self.clients:
            kyc_client = self.clients["kyc_provider"]
            # Note: KYC testing would require actual user data, so we skip functional test
            results["kyc_provider"] = APIResponse(success=True, data={"status": "configured"})

        # Test FRED
        if "fred" in self.clients:
            fred_client = self.clients["fred"]
            results["fred"] = await fred_client.get_series("FEDFUNDS", limit=1)

        # Test Polygon
        if "polygon" in self.clients:
            polygon_client = self.clients["polygon"]
            results["polygon"] = await polygon_client.get_previous_close("AAPL")

        # Test Finnhub
        if "finnhub" in self.clients:
            finnhub_client = self.clients["finnhub"]
            results["finnhub"] = await finnhub_client.get_quote("AAPL")

        # Test EODHD
        if "eodhd" in self.clients:
            eodhd_client = self.clients["eodhd"]
            results["eodhd"] = await eodhd_client.get_real_time("AAPL.US")

        # Test TradeStie Sentiment
        if "tradestie_sentiment" in self.clients:
            ts_client = self.clients["tradestie_sentiment"]
            results["tradestie_sentiment"] = await ts_client.get_wsb_sentiment()

        # Test Etherscan
        if "etherscan" in self.clients:
            eth_client_scan = self.clients["etherscan"]
            results["etherscan"] = await eth_client_scan.get_eth_price()

        # Test Polymarket Gamma
        if "polymarket_gamma" in self.clients:
            pm_gamma = self.clients["polymarket_gamma"]
            results["polymarket_gamma"] = await pm_gamma.get_events(limit=1)

        # Test Polymarket CLOB
        if "polymarket_clob" in self.clients:
            pm_clob = self.clients["polymarket_clob"]
            results["polymarket_clob"] = await pm_clob.get_simplified_markets()

        return results

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall API integration status"""
        available_clients = self.get_available_clients()
        connection_tests = await self.test_all_connections()

        successful_connections = sum(1 for result in connection_tests.values() if result.success)
        total_tests = len(connection_tests)

        return {
            "total_configured_apis": len(available_clients),
            "available_clients": available_clients,
            "connection_tests_run": total_tests,
            "successful_connections": successful_connections,
            "connection_success_rate": (successful_connections / total_tests * 100) if total_tests > 0 else 0,
            "connection_details": {
                name: {
                    "success": result.success,
                    "error": result.error if not result.success else None,
                    "response_time": result.response_time
                }
                for name, result in connection_tests.items()
            }
        }


# Global API integration hub
api_integration_hub = APIIntegrationHub()


async def test_api_integrations():
    """Test all API integrations"""
    logger.info("🔗 Testing API Integrations - Phase 2")
    logger.info("=" * 50)

    # Get system status
    status = await api_integration_hub.get_system_status()

    logger.info(f"📊 Configured APIs: {status['total_configured_apis']}")
    logger.info(f"🧪 Connection Tests: {status['connection_tests_run']}")
    logger.info(f"✅ Successful: {status['successful_connections']}")
    logger.info(f"📈 Success Rate: {status['connection_success_rate']:.1f}%")
    logger.info("")

    # Show available clients
    if status['available_clients']:
        logger.info("🔧 Available API Clients:")
        for client in status['available_clients']:
            logger.info(f"  • {client}")
    else:
        logger.info("⚠️  No API clients configured")
        logger.info("   Configure API keys in .env file:")
        logger.info("   - ETH_PRIVATE_KEY")
        logger.info("   - BIGBRAIN_AUTH_TOKEN")
        logger.info("   - COINMARKETCAP_API_KEY")
        logger.info("   - NEWS_API_KEY")
        logger.info("   - TWITTER_BEARER_TOKEN")
        logger.info("   - REDDIT_CLIENT_ID & REDDIT_CLIENT_SECRET")
        logger.info("   - NCC_AUTH_TOKEN")
        logger.info("   - KYC_PROVIDER_API_KEY")

    logger.info("")

    # Show connection test results
    if status['connection_details']:
        logger.info("🧪 Connection Test Results:")
        for api_name, details in status['connection_details'].items():
            status_icon = "✅" if details['success'] else "❌"
            logger.info(f"  {status_icon} {api_name}: {details['response_time']:.2f}s")
            if not details['success'] and details['error']:
                logger.info(f"    Error: {details['error']}")

    logger.info("")
    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_api_integrations())
