#!/usr/bin/env python3
"""
AAC Polygon.io Integration for Advanced Arbitrage
===============================================

Integrates Polygon.io for high-quality US market data and options chains.
Perfect for statistical arbitrage, volatility arbitrage, and cross-market opportunities.

Features:
- Real-time aggregates (bars)
- Options chains data
- Market microstructure
- High-frequency data for HFT arbitrage

Free Tier: 5M calls/month, 5 calls/minute
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class PolygonConfig:
    """Configuration for Polygon.io API"""
    api_key: str = os.getenv('POLYGON_API_KEY', '')
    base_url: str = "https://api.polygon.io"
    timeout: int = 30

    def is_configured(self) -> bool:
        return bool(self.api_key)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    exchange: str
    conditions: List[int]

@dataclass
class OptionsChain:
    """Options chain data"""
    underlying_symbol: str
    expiration_date: str
    strikes: List[float]
    calls: Dict[str, Any]
    puts: Dict[str, Any]

class PolygonArbitrageClient:
    """Polygon.io client for arbitrage trading"""

    def __init__(self, config: PolygonConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """Make authenticated API request"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        url = f"{self.config.base_url}{endpoint}"
        params = params or {}
        params['apiKey'] = self.config.api_key

        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"API Error {response.status}: {error_text}")

    async def get_real_time_quote(self, symbol: str) -> Optional[MarketData]:
        """Get real-time quote for a symbol"""
        try:
            data = await self._make_request(f"/v2/aggs/ticker/{symbol}/prev")

            if 'results' in data and data['results']:
                result = data['results'][0]
                return MarketData(
                    symbol=symbol,
                    price=result['c'],  # close price
                    volume=result['v'],
                    timestamp=datetime.fromtimestamp(result['t'] / 1000),
                    exchange="POLYGON",
                    conditions=[]
                )
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
        return None

    async def get_options_chain(self, underlying: str, expiration: str = None) -> Optional[OptionsChain]:
        """Get options chain for underlying symbol"""
        try:
            # Get next expiration if not specified
            if not expiration:
                exp_data = await self._make_request(f"/v3/reference/options/contracts?underlying_ticker={underlying}&limit=1")
                if 'results' in exp_data and exp_data['results']:
                    expiration = exp_data['results'][0]['expiration_date']

            if not expiration:
                return None

            # Get options chain
            chain_data = await self._make_request(
                f"/v3/reference/options/contracts",
                {
                    'underlying_ticker': underlying,
                    'expiration_date': expiration,
                    'limit': 100
                }
            )

            if 'results' not in chain_data:
                return None

            strikes = []
            calls = {}
            puts = {}

            for contract in chain_data['results']:
                strike = contract['strike_price']
                if strike not in strikes:
                    strikes.append(strike)

                contract_symbol = contract['ticker']

                if contract['contract_type'] == 'call':
                    calls[str(strike)] = {
                        'symbol': contract_symbol,
                        'bid': contract.get('bid', 0),
                        'ask': contract.get('ask', 0),
                        'last': contract.get('last_trade', {}).get('price', 0),
                        'volume': contract.get('day', {}).get('volume', 0),
                        'oi': contract.get('open_interest', 0)
                    }
                else:  # put
                    puts[str(strike)] = {
                        'symbol': contract_symbol,
                        'bid': contract.get('bid', 0),
                        'ask': contract.get('ask', 0),
                        'last': contract.get('last_trade', {}).get('price', 0),
                        'volume': contract.get('day', {}).get('volume', 0),
                        'oi': contract.get('open_interest', 0)
                    }

            return OptionsChain(
                underlying_symbol=underlying,
                expiration_date=expiration,
                strikes=sorted(strikes),
                calls=calls,
                puts=puts
            )

        except Exception as e:
            print(f"Error getting options chain for {underlying}: {e}")
        return None

    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get quotes for multiple symbols concurrently"""
        tasks = [self.get_real_time_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        quotes = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, MarketData):
                quotes[symbol] = result
        return quotes

async def test_polygon_integration():
    """Test Polygon.io integration"""
    print("🔍 Testing Polygon.io Integration")
    print("=" * 50)

    config = PolygonConfig()

    if not config.is_configured():
        print("❌ POLYGON_API_KEY not found in .env file")
        print("   Get your free API key from: https://polygon.io")
        print("   Add to .env: POLYGON_API_KEY=your_key_here")
        return

    async with PolygonArbitrageClient(config) as client:
        # Test basic quote
        print("\n📈 Testing Real-time Quotes:")
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

        quotes = await client.get_multiple_quotes(symbols)

        for symbol, quote in quotes.items():
            print(f"   {symbol}: ${quote.price:.2f} (Vol: {quote.volume:,})")

        # Test options chain
        print("\n📊 Testing Options Chain (AAPL):")
        chain = await client.get_options_chain('AAPL')

        if chain:
            print(f"   Underlying: {chain.underlying_symbol}")
            print(f"   Expiration: {chain.expiration_date}")
            print(f"   Strikes: {len(chain.strikes)} available")
            print(f"   Sample strike: ${chain.strikes[len(chain.strikes)//2]:.2f}")

            # Show some call/put data
            mid_strike = str(chain.strikes[len(chain.strikes)//2])
            if mid_strike in chain.calls:
                call = chain.calls[mid_strike]
                print(f"   Call @ ${mid_strike}: Bid ${call['bid']:.2f}, Ask ${call['ask']:.2f}")
            if mid_strike in chain.puts:
                put = chain.puts[mid_strike]
                print(f"   Put @ ${mid_strike}: Bid ${put['bid']:.2f}, Ask ${put['ask']:.2f}")
        else:
            print("   ❌ No options chain data available")

    print("\n✅ Polygon.io integration test complete!")

async def arbitrage_opportunities_demo():
    """Demo arbitrage opportunities using Polygon.io data"""
    print("\n🎯 Arbitrage Opportunities Demo")
    print("=" * 50)

    config = PolygonConfig()
    if not config.is_configured():
        return

    async with PolygonArbitrageClient(config) as client:
        # Cross-market arbitrage example
        print("\n🔄 Cross-Market Arbitrage Check:")
        symbols = ['AAPL', 'GOOGL', 'MSFT']

        quotes = await client.get_multiple_quotes(symbols)

        if len(quotes) >= 2:
            # Simple price comparison (in real arbitrage, you'd compare across exchanges)
            prices = {sym: quote.price for sym, quote in quotes.items()}
            max_price = max(prices.values())
            min_price = min(prices.values())
            spread = max_price - min_price

            print(f"   Price range: ${min_price:.2f} - ${max_price:.2f}")
            print(f"   Spread: ${spread:.2f} ({spread/min_price*100:.2f}%)")

            if spread/min_price > 0.001:  # 0.1% spread
                print("   ⚠️  Potential arbitrage opportunity detected!")
            else:
                print("   ✅ Markets appear efficient")

        # Options arbitrage example
        print("\n📊 Options Arbitrage Check (Put-Call Parity):")
        chain = await client.get_options_chain('AAPL')

        if chain and chain.strikes:
            # Check put-call parity for at-the-money options
            atm_strike = chain.strikes[len(chain.strikes)//2]

            call_key = str(atm_strike)
            put_key = str(atm_strike)

            if call_key in chain.calls and put_key in chain.puts:
                call_price = (chain.calls[call_key]['bid'] + chain.calls[call_key]['ask']) / 2
                put_price = (chain.puts[put_key]['bid'] + chain.puts[put_key]['ask']) / 2

                # Get underlying price
                underlying_quote = await client.get_real_time_quote('AAPL')
                if underlying_quote:
                    underlying_price = underlying_quote.price

                    # Simplified put-call parity check
                    parity_diff = abs((call_price - put_price) - (underlying_price - atm_strike))

                    print(f"   ATM Strike: ${atm_strike:.2f}")
                    print(f"   Underlying: ${underlying_price:.2f}")
                    print(f"   Call Price: ${call_price:.2f}")
                    print(f"   Put Price: ${put_price:.2f}")
                    print(f"   Parity Difference: ${parity_diff:.2f}")

                    if parity_diff > 0.1:  # $0.10 threshold
                        print("   ⚠️  Put-call parity violation detected!")
                    else:
                        print("   ✅ Put-call parity holds")

if __name__ == "__main__":
    print("🚀 AAC Polygon.io Arbitrage Integration")
    print("=" * 50)

    # Run tests
    asyncio.run(test_polygon_integration())
    asyncio.run(arbitrage_opportunities_demo())

    print("\n💡 Next Steps:")
    print("   1. Add POLYGON_API_KEY to your .env file")
    print("   2. Integrate with existing arbitrage strategies")
    print("   3. Add real-time monitoring for arbitrage opportunities")
    print("   4. Combine with other data sources for cross-validation")