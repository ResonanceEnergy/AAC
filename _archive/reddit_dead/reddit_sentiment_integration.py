#!/usr/bin/env python3
"""
AAC Reddit Sentiment Integration
=================================

TradeStie Reddit sentiment analysis for WallStreetBets arbitrage.
Provides real-time sentiment data for stocks discussed on Reddit.

Features:
- WallStreetBets sentiment analysis
- Top 50 discussed stocks
- Sentiment scores and comment counts
- Real-time updates (every 15 minutes)
- Sentiment-based arbitrage signals

API: https://tradestie.com/apps/reddit
Rate Limit: 20 requests per minute
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class RedditSentimentConfig:
    """Configuration for Reddit sentiment API"""
    base_url: str = "https://api.tradestie.com/v1/apps/reddit"
    rate_limit: int = 20  # requests per minute
    update_interval: int = 15  # minutes

class RedditSentimentClient:
    """TradeStie Reddit sentiment API client"""

    def __init__(self, config: RedditSentimentConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = datetime.min
        self.request_count = 0
        self.rate_limit_reset = datetime.now()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _rate_limit_wait(self):
        """Handle rate limiting"""
        now = datetime.now()

        # Reset counter if minute has passed
        if now - self.rate_limit_reset >= timedelta(minutes=1):
            self.request_count = 0
            self.rate_limit_reset = now

        # Check rate limit
        if self.request_count >= self.config.rate_limit:
            wait_time = 60 - (now - self.rate_limit_reset).seconds
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.rate_limit_reset = datetime.now()

        self.request_count += 1

    async def get_reddit_sentiment(self, date: str = None) -> List[Dict]:
        """Get Reddit sentiment data"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        await self._rate_limit_wait()

        params = {}
        if date:
            params['date'] = date

        try:
            async with self.session.get(self.config.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data if isinstance(data, list) else []
                else:
                    error_text = await response.text()
                    raise Exception(f"Reddit API Error {response.status}: {error_text}")
        except Exception as e:
            logger.info(f"Error fetching Reddit sentiment: {e}")
            return []

class RedditSentimentAnalyzer:
    """Reddit sentiment analysis for arbitrage"""

    def __init__(self, config: RedditSentimentConfig):
        self.config = config
        self.client = RedditSentimentClient(config)

    async def analyze_sentiment_opportunities(self, sentiment_data: List[Dict]) -> List[Dict]:
        """Analyze sentiment data for arbitrage opportunities"""
        opportunities = []

        # Filter for high sentiment stocks with significant discussion
        high_sentiment_stocks = []
        for stock in sentiment_data:
            comments = stock.get('no_of_comments', 0)
            sentiment_score = stock.get('sentiment_score', 0)
            sentiment = stock.get('sentiment', '')

            # Look for stocks with high discussion and strong sentiment
            if comments >= 10:  # At least 10 comments
                if sentiment == 'Bullish' and sentiment_score >= 0.15:
                    high_sentiment_stocks.append({
                        'ticker': stock.get('ticker', ''),
                        'sentiment': sentiment,
                        'sentiment_score': sentiment_score,
                        'comments': comments,
                        'signal_type': 'bullish_momentum'
                    })
                elif sentiment == 'Bearish' and sentiment_score <= -0.15:
                    high_sentiment_stocks.append({
                        'ticker': stock.get('ticker', ''),
                        'sentiment': sentiment,
                        'sentiment_score': sentiment_score,
                        'comments': comments,
                        'signal_type': 'bearish_momentum'
                    })

        # Sort by sentiment strength and discussion volume
        high_sentiment_stocks.sort(key=lambda x: (abs(x['sentiment_score']), x['comments']), reverse=True)

        # Convert to arbitrage opportunities
        for stock in high_sentiment_stocks[:10]:  # Top 10 opportunities
            opportunities.append({
                'symbol': stock['ticker'],
                'type': 'sentiment_arbitrage',
                'sentiment': stock['sentiment'],
                'sentiment_score': stock['sentiment_score'],
                'comment_volume': stock['comments'],
                'signal_strength': 'strong' if abs(stock['sentiment_score']) >= 0.2 else 'moderate',
                'description': f"Reddit sentiment: {stock['sentiment']} ({stock['sentiment_score']:.3f}) with {stock['comments']} comments",
                'confidence': min(abs(stock['sentiment_score']) * 2, 0.9),  # Scale confidence
                'timestamp': datetime.now().isoformat()
            })

        # TurboQuant: record sentiment snapshot
        try:
            from strategies.turboquant_integrations import IntegrationHub
            _tq_hub = IntegrationHub()
            _tq_hub.record_sentiment(opportunities)
            _tq_hub.save_all()
        except Exception:
            pass

        return opportunities

    async def get_market_sentiment_overview(self, sentiment_data: List[Dict]) -> Dict[str, Any]:
        """Get overall market sentiment overview"""
        if not sentiment_data:
            return {}

        total_comments = sum(stock.get('no_of_comments', 0) for stock in sentiment_data)
        bullish_stocks = len([s for s in sentiment_data if s.get('sentiment') == 'Bullish'])
        bearish_stocks = len([s for s in sentiment_data if s.get('sentiment') == 'Bearish'])

        avg_sentiment = sum(s.get('sentiment_score', 0) for s in sentiment_data) / len(sentiment_data)

        # Top discussed stocks
        top_stocks = sorted(sentiment_data, key=lambda x: x.get('no_of_comments', 0), reverse=True)[:5]

        return {
            'total_stocks': len(sentiment_data),
            'total_comments': total_comments,
            'bullish_stocks': bullish_stocks,
            'bearish_stocks': bearish_stocks,
            'average_sentiment': avg_sentiment,
            'market_sentiment': 'bullish' if avg_sentiment > 0.05 else 'bearish' if avg_sentiment < -0.05 else 'neutral',
            'top_discussed_stocks': [
                {
                    'ticker': stock.get('ticker', ''),
                    'comments': stock.get('no_of_comments', 0),
                    'sentiment': stock.get('sentiment', ''),
                    'score': stock.get('sentiment_score', 0)
                } for stock in top_stocks
            ],
            'timestamp': datetime.now().isoformat()
        }

    async def detect_sentiment_arbitrage(self, current_prices: Dict[str, float] = None) -> List[Dict]:
        """Detect sentiment-based arbitrage opportunities"""
        async with self.client:
            # Get latest sentiment data
            sentiment_data = await self.client.get_reddit_sentiment()

            if not sentiment_data:
                return []

            # Analyze for opportunities
            opportunities = await self.analyze_sentiment_opportunities(sentiment_data)

            # If we have current prices, we could correlate sentiment with price action
            # For now, return sentiment-based opportunities
            return opportunities

async def test_reddit_sentiment_integration():
    """Test Reddit sentiment integration"""
    logger.info("🧪 Testing Reddit Sentiment Integration")
    logger.info("=" * 45)

    config = RedditSentimentConfig()
    analyzer = RedditSentimentAnalyzer(config)

    async with analyzer.client:
        try:
            # Get sentiment data
            logger.info("📊 Fetching Reddit sentiment data...")
            sentiment_data = await analyzer.client.get_reddit_sentiment()

            if sentiment_data:
                logger.info(f"✅ Retrieved sentiment data for {len(sentiment_data)} stocks")

                # Show top 5 stocks
                logger.info("\n🔥 Top 5 Most Discussed Stocks:")
                top_stocks = sorted(sentiment_data, key=lambda x: x.get('no_of_comments', 0), reverse=True)[:5]

                for i, stock in enumerate(top_stocks, 1):
                    ticker = stock.get('ticker', 'N/A')
                    comments = stock.get('no_of_comments', 0)
                    sentiment = stock.get('sentiment', 'N/A')
                    score = stock.get('sentiment_score', 0)
                    logger.info(f"   {i}. {ticker}: {comments} comments, {sentiment} ({score:.3f})")

                # Get market overview
                logger.info("\n📈 Market Sentiment Overview:")
                overview = await analyzer.get_market_sentiment_overview(sentiment_data)

                logger.info(f"   Total Stocks: {overview.get('total_stocks', 0)}")
                logger.info(f"   Total Comments: {overview.get('total_comments', 0)}")
                logger.info(f"   Bullish Stocks: {overview.get('bullish_stocks', 0)}")
                logger.info(f"   Bearish Stocks: {overview.get('bearish_stocks', 0)}")
                logger.info(f"   Average Sentiment: {overview.get('average_sentiment', 0):.3f}")
                logger.info(f"   Market Sentiment: {overview.get('market_sentiment', 'unknown').upper()}")

                # Analyze arbitrage opportunities
                logger.info("\n🎯 Sentiment-Based Arbitrage Opportunities:")
                opportunities = await analyzer.analyze_sentiment_opportunities(sentiment_data)

                if opportunities:
                    for opp in opportunities[:5]:  # Show top 5
                        logger.info(f"   🎯 {opp['symbol']}: {opp['sentiment']} ({opp['sentiment_score']:.3f})")
                        logger.info(f"      Comments: {opp['comment_volume']}, Confidence: {opp['confidence']:.1%}")
                else:
                    logger.info("   No strong sentiment opportunities detected")

            else:
                logger.info("❌ No sentiment data received")
                return

        except Exception as e:
            logger.info(f"❌ Test failed: {e}")
            return

    logger.info("\n✅ Reddit sentiment integration test complete!")

if __name__ == "__main__":
    print("🚀 AAC Reddit Sentiment Integration")
    print("=" * 40)

    asyncio.run(test_reddit_sentiment_integration())

    print("\n💡 Reddit Sentiment Features:")
    print("   • WallStreetBets sentiment analysis")
    print("   • Real-time updates every 15 minutes")
    print("   • Top 50 discussed stocks")
    print("   • Sentiment-based arbitrage signals")
    print("   • Market sentiment overview")
    print("   • Comment volume tracking")
