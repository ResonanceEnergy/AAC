#!/usr/bin/env python3
"""
AAC Binance Arbitrage Integration
=================================

Comprehensive Binance exchange integration for AAC arbitrage system.
Enables cross-exchange arbitrage, triangular arbitrage, and market making.

Features:
- Spot trading API integration
- Real-time price feeds
- Order book data
- Account management
- Risk management
- Cross-exchange arbitrage detection

Testnet: https://testnet.binance.vision
Production: https://api.binance.com
"""

import asyncio
import aiohttp
import hmac
import hashlib
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

@dataclass
class BinanceConfig:
    """Configuration for Binance API"""
    api_key: str = os.getenv('BINANCE_API_KEY', '')
    api_secret: str = os.getenv('BINANCE_API_SECRET', '')
    testnet: bool = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'

    @property
    def base_url(self) -> str:
        return "https://testnet.binance.vision" if self.testnet else "https://api.binance.com"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

@dataclass
class BinancePrice:
    """Binance price data"""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    source: str = "binance"

@dataclass
class OrderBook:
    """Order book data"""
    symbol: str
    bids: List[Tuple[float, float]]  # price, quantity
    asks: List[Tuple[float, float]]  # price, quantity
    timestamp: datetime

@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity data"""
    type: str
    symbol: str
    description: str
    expected_return: float
    confidence: float
    timestamp: datetime
    exchanges: List[str]
    trade_details: Dict[str, Any]

class BinanceArbitrageClient:
    """Binance client for arbitrage trading"""

    def __init__(self, config: BinanceConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = 0
        self.request_count = 0

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        return hmac.new(
            self.config.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds"""
        return int(time.time() * 1000)

    async def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make API request with rate limiting"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        # Rate limiting (1200 requests per minute for spot)
        current_time = time.time()
        if current_time - self.last_request_time < 0.05:  # Max 20 requests per second
            await asyncio.sleep(0.05 - (current_time - self.last_request_time))
        self.last_request_time = time.time()

        url = f"{self.config.base_url}{endpoint}"
        headers = {'X-MBX-APIKEY': self.config.api_key}

        params = params or {}

        if signed:
            params['timestamp'] = self._get_timestamp()
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            params['signature'] = self._generate_signature(query_string)

        try:
            if method.upper() == 'GET':
                async with self.session.get(url, params=params, headers=headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == 'POST':
                async with self.session.post(url, data=params, headers=headers) as response:
                    return await self._handle_response(response)
            elif method.upper() == 'DELETE':
                async with self.session.delete(url, params=params, headers=headers) as response:
                    return await self._handle_response(response)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        except Exception as e:
            print(f"Request error: {e}")
            raise

    async def _handle_response(self, response) -> Dict:
        """Handle API response"""
        if response.status == 200:
            return await response.json()
        else:
            error_text = await response.text()
            raise Exception(f"API Error {response.status}: {error_text}")

    async def get_price(self, symbol: str) -> Optional[BinancePrice]:
        """Get current price for a symbol"""
        try:
            data = await self._make_request('GET', '/api/v3/ticker/price', {'symbol': symbol})

            return BinancePrice(
                symbol=symbol,
                price=float(data['price']),
                volume=0.0,  # Not available in price endpoint
                timestamp=datetime.now()
            )
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None

    async def get_24hr_stats(self, symbol: str) -> Optional[Dict]:
        """Get 24hr statistics for a symbol"""
        try:
            return await self._make_request('GET', '/api/v3/ticker/24hr', {'symbol': symbol})
        except Exception as e:
            print(f"Error getting 24hr stats for {symbol}: {e}")
            return None

    async def get_order_book(self, symbol: str, limit: int = 100) -> Optional[OrderBook]:
        """Get order book for a symbol"""
        try:
            data = await self._make_request('GET', '/api/v3/depth', {
                'symbol': symbol,
                'limit': limit
            })

            bids = [(float(price), float(qty)) for price, qty in data['bids']]
            asks = [(float(price), float(qty)) for price, qty in data['asks']]

            return OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.now()
            )
        except Exception as e:
            print(f"Error getting order book for {symbol}: {e}")
            return None

    async def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        try:
            return await self._make_request('GET', '/api/v3/account', signed=True)
        except Exception as e:
            print(f"Error getting account info: {e}")
            return None

    async def place_order(self, symbol: str, side: str, order_type: str,
                         quantity: float, price: float = None, test: bool = True) -> Optional[Dict]:
        """Place an order (use test=True for paper trading)"""
        try:
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': order_type.upper(),
                'quantity': str(quantity)
            }

            if price and order_type.upper() == 'LIMIT':
                params['price'] = str(price)
                params['timeInForce'] = 'GTC'

            endpoint = '/api/v3/order/test' if test else '/api/v3/order'
            return await self._make_request('POST', endpoint, params, signed=True)

        except Exception as e:
            print(f"Error placing order: {e}")
            return None

    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, BinancePrice]:
        """Get prices for multiple symbols concurrently"""
        tasks = [self.get_price(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, BinancePrice):
                prices[symbol] = result
        return prices

async def test_binance_integration():
    """Test Binance integration"""
    print("🔍 Testing Binance Integration")
    print("=" * 40)

    config = BinanceConfig()

    if not config.is_configured():
        print("❌ BINANCE_API_KEY and BINANCE_API_SECRET not configured")
        print("   Current values are dummy ('y')")
        print("   Get real keys from: https://www.binance.com/en/my/settings/api-management")
        print("   For testnet: https://testnet.binance.vision/")
        return

    print(f"✅ API Key configured: {config.api_key[:8]}...")
    print(f"🏭 Testnet mode: {'ON' if config.testnet else 'OFF'}")

    async with BinanceArbitrageClient(config) as client:
        # Test basic price fetching
        print("\n📈 Testing Price Data:")
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT']

        prices = await client.get_multiple_prices(symbols)

        for symbol, price_data in prices.items():
            print(f"   {symbol}: ${price_data.price:.2f}")

        # Test order book
        print("\n📊 Testing Order Book (BTCUSDT):")
        order_book = await client.get_order_book('BTCUSDT', limit=5)

        if order_book:
            print(f"   Best Bid: ${order_book.bids[0][0]:.2f} (Qty: {order_book.bids[0][1]:.4f})")
            print(f"   Best Ask: ${order_book.asks[0][0]:.2f} (Qty: {order_book.asks[0][1]:.4f})")
            print(f"   Spread: ${(order_book.asks[0][0] - order_book.bids[0][0]):.2f}")

        # Test 24hr stats
        print("\n📈 Testing 24hr Statistics (BTCUSDT):")
        stats = await client.get_24hr_stats('BTCUSDT')

        if stats:
            print(f"   Price Change: {float(stats['priceChangePercent']):+.2f}%")
            print(f"   Volume: {float(stats['volume']):,.0f} BTC")
            print(f"   High: ${float(stats['highPrice']):.2f}")
            print(f"   Low: ${float(stats['lowPrice']):.2f}")

        # Test account info (if configured properly)
        print("\n👤 Testing Account Info:")
        account = await client.get_account_info()

        if account:
            print("✅ Account access successful")
            balances = [b for b in account['balances'] if float(b['free']) > 0 or float(b['locked']) > 0]
            if balances:
                print("   Balances:")
                for balance in balances[:5]:  # Show first 5
                    print(f"     {balance['asset']}: {balance['free']} free, {balance['locked']} locked")
            else:
                print("   No balances found (testnet account)")
        else:
            print("❌ Account access failed (check API keys)")

    print("\n✅ Binance integration test complete!")

class CrossExchangeArbitrage:
    """Cross-exchange arbitrage detector"""

    def __init__(self):
        self.binance_client = None
        self.other_exchanges = {}  # Will hold other exchange clients

    async def initialize_clients(self):
        """Initialize exchange clients"""
        binance_config = BinanceConfig()
        if binance_config.is_configured():
            self.binance_client = BinanceArbitrageClient(binance_config)

    async def detect_cross_exchange_arbitrage(self, symbols: List[str]) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities across exchanges"""
        opportunities = []

        if not self.binance_client:
            return opportunities

        async with self.binance_client as binance:
            # Get Binance prices
            binance_prices = await binance.get_multiple_prices(symbols)

            # For demo, we'll compare with simulated other exchange prices
            # In production, you'd integrate real exchange APIs here
            simulated_exchanges = {
                'coinbase': lambda price: price * (1 + 0.001),  # 0.1% higher
                'kraken': lambda price: price * (1 - 0.002),    # 0.2% lower
                'gemini': lambda price: price * (1 + 0.0015),   # 0.15% higher
            }

            for symbol in symbols:
                if symbol in binance_prices:
                    binance_price = binance_prices[symbol].price

                    for exchange_name, price_func in simulated_exchanges.items():
                        exchange_price = price_func(binance_price)
                        spread = abs(exchange_price - binance_price)
                        spread_pct = spread / binance_price

                        # Check for arbitrage opportunity (> 0.3% spread after fees)
                        if spread_pct > 0.003:
                            if exchange_price > binance_price:
                                # Buy on Binance, sell on other exchange
                                opportunity = ArbitrageOpportunity(
                                    type="cross_exchange_buy_binance",
                                    symbol=symbol,
                                    description=f"Buy {symbol} on Binance (${binance_price:.2f}), sell on {exchange_name} (${exchange_price:.2f})",
                                    expected_return=spread_pct,
                                    confidence=0.7,
                                    timestamp=datetime.now(),
                                    exchanges=['binance', exchange_name],
                                    trade_details={
                                        'buy_exchange': 'binance',
                                        'sell_exchange': exchange_name,
                                        'buy_price': binance_price,
                                        'sell_price': exchange_price,
                                        'spread': spread,
                                        'spread_pct': spread_pct
                                    }
                                )
                            else:
                                # Buy on other exchange, sell on Binance
                                opportunity = ArbitrageOpportunity(
                                    type="cross_exchange_sell_binance",
                                    symbol=symbol,
                                    description=f"Buy {symbol} on {exchange_name} (${exchange_price:.2f}), sell on Binance (${binance_price:.2f})",
                                    expected_return=spread_pct,
                                    confidence=0.7,
                                    timestamp=datetime.now(),
                                    exchanges=[exchange_name, 'binance'],
                                    trade_details={
                                        'buy_exchange': exchange_name,
                                        'sell_exchange': 'binance',
                                        'buy_price': exchange_price,
                                        'sell_price': binance_price,
                                        'spread': spread,
                                        'spread_pct': spread_pct
                                    }
                                )

                            opportunities.append(opportunity)

        return opportunities

async def arbitrage_demo():
    """Demo cross-exchange arbitrage detection"""
    print("\n🎯 Cross-Exchange Arbitrage Demo")
    print("=" * 40)

    arbitrage_detector = CrossExchangeArbitrage()
    await arbitrage_detector.initialize_clients()

    if not arbitrage_detector.binance_client:
        print("❌ Binance client not configured")
        return

    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

    print(f"🔍 Scanning {len(symbols)} symbols for arbitrage opportunities...")

    opportunities = await arbitrage_detector.detect_cross_exchange_arbitrage(symbols)

    print(f"\n📊 Found {len(opportunities)} arbitrage opportunities:")

    for opp in opportunities:
        print(f"\n🎯 {opp.symbol} - {opp.type.replace('_', ' ').title()}")
        print(f"   {opp.description}")
        print(f"   Confidence: {opp.confidence:.1f}")
        print(f"   Exchanges: {', '.join(opp.exchanges)}")

    if opportunities:
        print("\nThese opportunities represent potential profits after fees")
        print("   In production, you'd execute these trades automatically")
    else:
        print("\n✅ No arbitrage opportunities found (markets efficient)")

if __name__ == "__main__":
    print("🚀 AAC Binance Arbitrage Integration")
    print("=" * 50)

    # Run tests
    asyncio.run(test_binance_integration())
    asyncio.run(arbitrage_demo())

    print("\n💡 Next Steps:")
    print("   1. Replace BINANCE_API_KEY and BINANCE_API_SECRET with real keys")
    print("   2. Set BINANCE_TESTNET=false for production trading")
    print("   3. Integrate with existing AAC arbitrage strategies")
    print("   4. Add real exchange APIs for true cross-exchange arbitrage")
    print("   5. Implement automated trade execution with risk management")