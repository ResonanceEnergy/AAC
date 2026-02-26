#!/usr/bin/env python3
"""
AAC EODHD Integration
=====================

EODHD (End of Day Historical Data) integration for AAC arbitrage system.
Provides comprehensive historical market data for arbitrage analysis.

Features:
- End-of-day historical data
- Real-time data (premium)
- Fundamental data
- Options data
- Forex and crypto data
- Global market coverage

API Documentation: https://eodhd.com/financial-apis/
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
class EODHDConfig:
    """Configuration for EODHD API"""
    api_key: str = os.getenv('EODHD_API_KEY', '')
    base_url: str = "https://eodhistoricaldata.com/api"
    realtime_url: str = "https://ws.eodhistoricaldata.com/ws"

    def is_configured(self) -> bool:
        return bool(self.api_key)

class EODHDClient:
    """EODHD API client for market data"""

    def __init__(self, config: EODHDConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        url = f"{self.config.base_url}{endpoint}"
        params = params or {}
        params['api_token'] = self.config.api_key
        params['fmt'] = 'json'  # JSON response format

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"EODHD API Error {response.status}: {error_text}")
        except Exception as e:
            print(f"EODHD request error: {e}")
            raise

    async def get_end_of_day_data(self, symbol: str, exchange: str = None,
                                 from_date: str = None, to_date: str = None) -> List[Dict]:
        """Get end-of-day historical data"""
        try:
            # Format symbol (e.g., "AAPL.US" for Apple on US exchange)
            if exchange:
                full_symbol = f"{symbol}.{exchange}"
            else:
                full_symbol = symbol

            params = {'period': 'd'}  # Daily data

            if from_date:
                params['from'] = from_date
            if to_date:
                params['to'] = to_date

            data = await self._make_request(f"/eod/{full_symbol}", params)

            if isinstance(data, list):
                return data
            else:
                print(f"Unexpected response format for {full_symbol}")
                return []

        except Exception as e:
            print(f"Error getting EOD data for {symbol}: {e}")
            return []

    async def get_real_time_price(self, symbol: str, exchange: str = None) -> Optional[Dict]:
        """Get real-time price data (premium feature)"""
        try:
            if exchange:
                full_symbol = f"{symbol}.{exchange}"
            else:
                full_symbol = symbol

            data = await self._make_request(f"/real-time/{full_symbol}")

            if data:
                return data[0] if isinstance(data, list) and data else data
            return None

        except Exception as e:
            print(f"Error getting real-time price for {symbol}: {e}")
            return None

    async def get_fundamentals(self, symbol: str, exchange: str = None) -> Optional[Dict]:
        """Get fundamental data"""
        try:
            if exchange:
                full_symbol = f"{symbol}.{exchange}"
            else:
                full_symbol = symbol

            data = await self._make_request(f"/fundamentals/{full_symbol}")
            return data

        except Exception as e:
            print(f"Error getting fundamentals for {symbol}: {e}")
            return None

    async def get_options_data(self, symbol: str, exchange: str = "US",
                              from_date: str = None, to_date: str = None) -> List[Dict]:
        """Get options data"""
        try:
            full_symbol = f"{symbol}.{exchange}"

            params = {}
            if from_date:
                params['from'] = from_date
            if to_date:
                params['to'] = to_date

            data = await self._make_request(f"/options/{full_symbol}", params)

            if isinstance(data, list):
                return data
            return []

        except Exception as e:
            print(f"Error getting options data for {symbol}: {e}")
            return []

    async def get_forex_data(self, from_symbol: str, to_symbol: str,
                           from_date: str = None, to_date: str = None) -> List[Dict]:
        """Get forex historical data"""
        try:
            symbol = f"{from_symbol}{to_symbol}.FOREX"

            params = {}
            if from_date:
                params['from'] = from_date
            if to_date:
                params['to'] = to_date

            data = await self._make_request(f"/eod/{symbol}", params)

            if isinstance(data, list):
                return data
            return []

        except Exception as e:
            print(f"Error getting forex data for {from_symbol}/{to_symbol}: {e}")
            return []

    async def get_crypto_data(self, symbol: str, from_date: str = None, to_date: str = None) -> List[Dict]:
        """Get cryptocurrency historical data"""
        try:
            crypto_symbol = f"{symbol}-USD.CC"

            params = {}
            if from_date:
                params['from'] = from_date
            if to_date:
                params['to'] = to_date

            data = await self._make_request(f"/eod/{crypto_symbol}", params)

            if isinstance(data, list):
                return data
            return []

        except Exception as e:
            print(f"Error getting crypto data for {symbol}: {e}")
            return []

    async def search_instruments(self, query: str, exchange: str = None, limit: int = 10) -> List[Dict]:
        """Search for financial instruments"""
        try:
            params = {
                'q': query,
                'limit': limit
            }

            if exchange:
                params['exchange'] = exchange

            data = await self._make_request("/search/", params)

            if isinstance(data, list):
                return data
            return []

        except Exception as e:
            print(f"Error searching instruments: {e}")
            return []

class EODHDArbitrageAnalyzer:
    """EODHD-based arbitrage analysis"""

    def __init__(self, config: EODHDConfig):
        self.config = config
        self.client = EODHDClient(config)

    async def analyze_price_discrepancies(self, symbols: List[str],
                                        exchanges: List[str] = None) -> List[Dict]:
        """Analyze price discrepancies across exchanges"""
        discrepancies = []

        exchanges = exchanges or ['US', 'LSE', 'TSE', 'HKEX']

        async with self.client:
            for symbol in symbols:
                symbol_prices = {}

                # Get prices from different exchanges
                for exchange in exchanges:
                    try:
                        eod_data = await self.client.get_end_of_day_data(symbol, exchange)

                        if eod_data and len(eod_data) > 0:
                            latest_price = eod_data[-1].get('close', 0)
                            if latest_price > 0:
                                symbol_prices[exchange] = latest_price

                    except Exception as e:
                        print(f"Error getting {symbol} from {exchange}: {e}")
                        continue

                # Analyze discrepancies
                if len(symbol_prices) > 1:
                    prices = list(symbol_prices.values())
                    max_price = max(prices)
                    min_price = min(prices)
                    spread = max_price - min_price
                    spread_pct = (spread / min_price) * 100

                    if spread_pct > 0.1:  # More than 0.1% spread
                        discrepancies.append({
                            'symbol': symbol,
                            'spread': spread,
                            'spread_pct': spread_pct,
                            'max_price': max_price,
                            'min_price': min_price,
                            'exchanges': symbol_prices,
                            'type': 'cross_exchange_eod'
                        })

        return discrepancies

    async def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview data"""
        async with self.client:
            try:
                # Get major indices
                indices = ['GSPC.INDX', 'IXIC.INDX', 'DJI.INDX', 'FTSE.INDX', 'N225.INDX']
                index_data = {}

                for index in indices:
                    try:
                        data = await self.client.get_end_of_day_data(index)
                        if data and len(data) > 0:
                            latest = data[-1]
                            index_data[index] = {
                                'price': latest.get('close', 0),
                                'change': latest.get('change', 0),
                                'change_pct': latest.get('change_percent', 0)
                            }
                    except Exception as e:
                        print(f"Error getting index {index}: {e}")

                return {
                    'indices': index_data,
                    'timestamp': datetime.now().isoformat()
                }

            except Exception as e:
                print(f"Error getting market overview: {e}")
                return {}

async def test_eodhd_integration():
    """Test EODHD integration"""
    print("🧪 Testing EODHD Integration")
    print("=" * 40)

    config = EODHDConfig()

    if not config.is_configured():
        print("❌ EODHD API key not configured")
        print("💡 Set EODHD_API_KEY in your .env file")
        return

    async with EODHDClient(config) as client:
        # Test basic connectivity
        print("🔍 Testing basic connectivity...")

        try:
            # Test with Apple stock
            eod_data = await client.get_end_of_day_data('AAPL', 'US')
            if eod_data and len(eod_data) > 0:
                latest = eod_data[-1]
                print("✅ EODHD API connected")
                print(f"   AAPL latest price: ${latest.get('close', 'N/A')}")
                print(f"   Data points: {len(eod_data)}")
            else:
                print("❌ No data received for AAPL")
                return

        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return

        # Test arbitrage analysis
        print("\n🔍 Testing arbitrage analysis...")

        analyzer = EODHDArbitrageAnalyzer(config)

        # Test with major stocks
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        discrepancies = await analyzer.analyze_price_discrepancies(symbols)

        print(f"📊 Found {len(discrepancies)} price discrepancies:")

        for disc in discrepancies:
            print(f"   🎯 {disc['symbol']}: {disc['spread_pct']:.2f}% spread")
            print(f"      Price range: ${disc['min_price']:.2f} - ${disc['max_price']:.2f}")

        # Test market overview
        print("\n📈 Market Overview:")
        overview = await analyzer.get_market_overview()

        if overview.get('indices'):
            for index, data in overview['indices'].items():
                print(f"   {index}: ${data['price']:.2f} ({data['change_pct']:+.2f}%)")

        print("\n✅ EODHD integration test complete!")

if __name__ == "__main__":
    print("🚀 AAC EODHD Integration")
    print("=" * 30)

    asyncio.run(test_eodhd_integration())

    print("\n💡 EODHD Features:")
    print("   • End-of-day historical data")
    print("   • Real-time data (premium)")
    print("   • Global market coverage")
    print("   • Options and fundamentals")
    print("   • Forex and crypto data")
    print("   • Arbitrage opportunity detection")