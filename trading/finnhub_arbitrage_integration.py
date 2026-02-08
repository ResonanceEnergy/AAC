#!/usr/bin/env python3
"""
AAC Finnhub Integration for Real-Time Arbitrage
==============================================

Integrates Finnhub for real-time quotes, earnings data, and news sentiment.
Perfect for high-frequency arbitrage and market timing strategies.

Features:
- Real-time stock quotes
- Earnings calendar and surprises
- News sentiment analysis
- Technical indicators
- Insider trading data

Free Tier: 150 calls/day, 60 calls/minute
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
class FinnhubConfig:
    """Configuration for Finnhub API"""
    api_key: str = os.getenv('FINNHUB_API_KEY', '')
    base_url: str = "https://finnhub.io/api/v1"
    timeout: int = 30

    def is_configured(self) -> bool:
        return bool(self.api_key)

@dataclass
class RealTimeQuote:
    """Real-time quote data"""
    symbol: str
    current_price: float
    change: float
    change_percent: float
    high: float
    low: float
    open: float
    previous_close: float
    timestamp: datetime

@dataclass
class NewsSentiment:
    """News sentiment data"""
    symbol: str
    sentiment_score: float
    sentiment_label: str
    articles_count: int
    buzz_score: float

@dataclass
class EarningsData:
    """Earnings data"""
    symbol: str
    date: str
    eps_estimate: float
    eps_actual: float
    surprise: float
    surprise_percent: float

class FinnhubArbitrageClient:
    """Finnhub client for arbitrage trading"""

    def __init__(self, config: FinnhubConfig):
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
        params['token'] = self.config.api_key

        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"API Error {response.status}: {error_text}")

    async def get_real_time_quote(self, symbol: str) -> Optional[RealTimeQuote]:
        """Get real-time quote for a symbol"""
        try:
            data = await self._make_request(f"/quote?symbol={symbol}")

            # Finnhub returns data directly
            if 'c' in data:  # current price exists
                return RealTimeQuote(
                    symbol=symbol,
                    current_price=data.get('c', 0),
                    change=data.get('d', 0),
                    change_percent=data.get('dp', 0),
                    high=data.get('h', 0),
                    low=data.get('l', 0),
                    open=data.get('o', 0),
                    previous_close=data.get('pc', 0),
                    timestamp=datetime.now()
                )
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
        return None

    async def get_news_sentiment(self, symbol: str) -> Optional[NewsSentiment]:
        """Get news sentiment for a symbol"""
        try:
            data = await self._make_request(f"/news-sentiment?symbol={symbol}")

            if 'sentiment' in data:
                sentiment = data['sentiment']
                buzz = data.get('buzz', {})

                return NewsSentiment(
                    symbol=symbol,
                    sentiment_score=sentiment.get('score', 0),
                    sentiment_label=sentiment.get('label', 'neutral'),
                    articles_count=buzz.get('articlesInLastWeek', 0),
                    buzz_score=buzz.get('buzz', 0)
                )
        except Exception as e:
            print(f"Error getting news sentiment for {symbol}: {e}")
        return None

    async def get_earnings_calendar(self, symbol: str, limit: int = 5) -> List[EarningsData]:
        """Get earnings calendar for a symbol"""
        try:
            data = await self._make_request(f"/calendar/earnings?symbol={symbol}&limit={limit}")

            earnings = []
            if 'earningsCalendar' in data:
                for item in data['earningsCalendar']:
                    earnings.append(EarningsData(
                        symbol=symbol,
                        date=item.get('date', ''),
                        eps_estimate=item.get('epsEstimate', 0),
                        eps_actual=item.get('epsActual', 0),
                        surprise=item.get('surprise', 0),
                        surprise_percent=item.get('surprisePercent', 0)
                    ))
            return earnings
        except Exception as e:
            print(f"Error getting earnings calendar for {symbol}: {e}")
        return []

    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, RealTimeQuote]:
        """Get quotes for multiple symbols concurrently"""
        tasks = [self.get_real_time_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        quotes = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, RealTimeQuote):
                quotes[symbol] = result
        return quotes

    async def get_multiple_sentiments(self, symbols: List[str]) -> Dict[str, NewsSentiment]:
        """Get news sentiment for multiple symbols concurrently"""
        tasks = [self.get_news_sentiment(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        sentiments = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, NewsSentiment):
                sentiments[symbol] = result
        return sentiments

async def test_finnhub_integration():
    """Test Finnhub integration"""
    print("🔍 Testing Finnhub Integration")
    print("=" * 50)

    config = FinnhubConfig()

    if not config.is_configured():
        print("❌ FINNHUB_API_KEY not found in .env file")
        print("   Get your free API key from: https://finnhub.io")
        print("   Add to .env: FINNHUB_API_KEY=your_key_here")
        return

    async with FinnhubArbitrageClient(config) as client:
        # Test basic quotes
        print("\n📈 Testing Real-time Quotes:")
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

        quotes = await client.get_multiple_quotes(symbols)

        for symbol, quote in quotes.items():
            print(f"   {symbol}: ${quote.current_price:.2f} ({quote.change_percent:+.2f}%)")

        # Test news sentiment
        print("\n📰 Testing News Sentiment:")
        sentiments = await client.get_multiple_sentiments(symbols[:2])  # Test first 2 to save API calls

        for symbol, sentiment in sentiments.items():
            print(f"   {symbol}: {sentiment.sentiment_label} ({sentiment.sentiment_score:.2f})")
            print(f"      Articles: {sentiment.articles_count}, Buzz: {sentiment.buzz_score:.2f}")

        # Test earnings calendar
        print("\n📊 Testing Earnings Calendar (AAPL):")
        earnings = await client.get_earnings_calendar('AAPL', 3)

        for earning in earnings:
            print(f"   {earning.date}: EPS ${earning.eps_actual:.2f} (Est: ${earning.eps_estimate:.2f})")
            print(f"      Surprise: {earning.surprise_percent:+.1f}%")

    print("\n✅ Finnhub integration test complete!")

async def arbitrage_opportunities_demo():
    """Demo arbitrage opportunities using Finnhub data"""
    print("\n🎯 Arbitrage Opportunities Demo")
    print("=" * 50)

    config = FinnhubConfig()
    if not config.is_configured():
        return

    async with FinnhubArbitrageClient(config) as client:
        # Sentiment-based arbitrage
        print("\n📰 Sentiment-Based Arbitrage Analysis:")
        symbols = ['AAPL', 'GOOGL', 'MSFT']

        quotes = await client.get_multiple_quotes(symbols)
        sentiments = await client.get_multiple_sentiments(symbols)

        for symbol in symbols:
            if symbol in quotes and symbol in sentiments:
                quote = quotes[symbol]
                sentiment = sentiments[symbol]

                # Simple sentiment-arbitrage logic
                sentiment_signal = 0
                if sentiment.sentiment_label == 'positive' and sentiment.sentiment_score > 0.1:
                    sentiment_signal = 1
                elif sentiment.sentiment_label == 'negative' and sentiment.sentiment_score < -0.1:
                    sentiment_signal = -1

                print(f"   {symbol}:")
                print(f"      Price: ${quote.current_price:.2f} ({quote.change_percent:+.2f}%)")
                print(f"      Sentiment: {sentiment.sentiment_label} ({sentiment.sentiment_score:.2f})")
                print(f"      Signal: {'BUY' if sentiment_signal > 0 else 'SELL' if sentiment_signal < 0 else 'HOLD'}")

        # Earnings surprise arbitrage
        print("\n📊 Earnings Surprise Analysis:")
        earnings = await client.get_earnings_calendar('AAPL', 5)

        for earning in earnings:
            if abs(earning.surprise_percent) > 5:  # Significant surprise
                direction = "positive" if earning.surprise > 0 else "negative"
                print(f"   {earning.date}: {direction} surprise ({earning.surprise_percent:+.1f}%)")
                print("   ⚠️  Potential post-earnings arbitrage opportunity!")

if __name__ == "__main__":
    print("🚀 AAC Finnhub Arbitrage Integration")
    print("=" * 50)

    # Run tests
    asyncio.run(test_finnhub_integration())
    asyncio.run(arbitrage_opportunities_demo())

    print("\n💡 Next Steps:")
    print("   1. Add FINNHUB_API_KEY to your .env file")
    print("   2. Integrate sentiment analysis with existing strategies")
    print("   3. Add earnings-based arbitrage signals")
    print("   4. Combine with Polygon.io for comprehensive market data")