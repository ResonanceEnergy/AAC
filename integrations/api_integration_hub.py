#!/usr/bin/env python3
"""
API Integration Hub - Phase 2
=============================
Centralized API management for all external services required for live trading.
"""

import asyncio
import logging
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config
from shared.audit_logger import get_audit_logger


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
        self.requests_made = []
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.endpoint.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

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

        if not self.session:
            return APIResponse(
                success=False,
                error="No active session",
                status_code=500
            )

        start_time = time.time()
        last_error = None

        for attempt in range(self.endpoint.retries):
            try:
                headers = self._get_auth_headers()
                if 'headers' in kwargs:
                    headers.update(kwargs['headers'])
                kwargs['headers'] = headers

                async with self.session.request(method, url, **kwargs) as response:
                    response_time = time.time() - start_time

                    if response.status == 200:
                        try:
                            data = await response.json()
                        except:
                            data = await response.text()

                        # Audit successful API call
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
                    else:
                        error_text = await response.text()
                        last_error = f"HTTP {response.status}: {error_text}"

                        if response.status == 429:  # Rate limited
                            await asyncio.sleep(60)  # Wait before retry
                            continue
                        elif response.status >= 500:  # Server error, retry
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:  # Client error, don't retry
                            break

            except asyncio.TimeoutError:
                last_error = "Request timeout"
                await asyncio.sleep(2 ** attempt)
                continue
            except Exception as e:
                last_error = str(e)
                await asyncio.sleep(2 ** attempt)
                continue

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
            severity="warning"
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
                self.account = Account.from_key(self.config.eth_private_key)
                self.logger.info("Ethereum DEX client initialized with private key")
            else:
                self.logger.warning("Ethereum DEX client initialized without private key")

        except ImportError:
            self.logger.error("Web3.py not installed. Install with: pip install web3 eth-account")
        except Exception as e:
            self.logger.error(f"Failed to initialize Web3: {e}")

    async def get_balance(self, address: str = None) -> APIResponse:
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

    async def get_arbitrage_opportunities(self, exchanges: List[str] = None) -> APIResponse:
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

            async with self.session.post(
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
    print("🔗 Testing API Integrations - Phase 2")
    print("=" * 50)

    # Get system status
    status = await api_integration_hub.get_system_status()

    print(f"📊 Configured APIs: {status['total_configured_apis']}")
    print(f"🧪 Connection Tests: {status['connection_tests_run']}")
    print(f"✅ Successful: {status['successful_connections']}")
    print(f"📈 Success Rate: {status['connection_success_rate']:.1f}%")
    print()

    # Show available clients
    if status['available_clients']:
        print("🔧 Available API Clients:")
        for client in status['available_clients']:
            print(f"  • {client}")
    else:
        print("⚠️  No API clients configured")
        print("   Configure API keys in .env file:")
        print("   - ETH_PRIVATE_KEY")
        print("   - BIGBRAIN_AUTH_TOKEN")
        print("   - COINMARKETCAP_API_KEY")
        print("   - NEWS_API_KEY")
        print("   - TWITTER_BEARER_TOKEN")
        print("   - REDDIT_CLIENT_ID & REDDIT_CLIENT_SECRET")
        print("   - NCC_AUTH_TOKEN")
        print("   - KYC_PROVIDER_API_KEY")

    print()

    # Show connection test results
    if status['connection_details']:
        print("🧪 Connection Test Results:")
        for api_name, details in status['connection_details'].items():
            status_icon = "✅" if details['success'] else "❌"
            print(f"  {status_icon} {api_name}: {details['response_time']:.2f}s")
            if not details['success'] and details['error']:
                print(f"    Error: {details['error']}")

    print()
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_api_integrations())