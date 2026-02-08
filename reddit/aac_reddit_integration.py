"""
AAC Reddit Data Integration
===========================

Integrates scraped Reddit data into the AAC arbitrage system.
Loads recent scrape data and makes it available for analysis and trading decisions.

Features:
- Loads latest Reddit scrape data
- Provides sentiment analysis for tickers
- Identifies arbitrage opportunities
- Feeds data into AAC decision engine
- Real-time data updates
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RedditSentimentData:
    """Reddit sentiment data for integration"""
    ticker: str
    total_mentions: int
    avg_sentiment: float
    bullish_count: int
    bearish_count: int
    neutral_count: int
    subreddits: List[str]
    last_updated: datetime
    confidence_score: float

@dataclass
class ArbitrageSignal:
    """Arbitrage signal from Reddit data"""
    signal_type: str
    confidence: float
    description: str
    tickers: List[str]
    subreddits: List[str]
    detected_at: datetime

class AACRedditIntegration:
    """
    Integrates Reddit scrape data into AAC system
    """

    def __init__(self, data_dir: str = "data/reddit_scrapes"):
        self.data_dir = data_dir
        self.sentiment_data: Dict[str, RedditSentimentData] = {}
        self.arbitrage_signals: List[ArbitrageSignal] = []
        self.last_update = None

    def load_latest_data(self) -> bool:
        """Load the most recent scrape data"""
        try:
            if not os.path.exists(self.data_dir):
                logger.warning(f"Data directory {self.data_dir} does not exist")
                return False

            # Find latest files
            files = os.listdir(self.data_dir)
            insights_files = [f for f in files if f.startswith("insights_") and f.endswith(".json")]
            posts_files = [f for f in files if f.startswith("posts_") and f.endswith(".json")]

            if not insights_files:
                logger.warning("No insights files found")
                return False

            # Get latest insights file
            latest_insights = max(insights_files)
            insights_path = os.path.join(self.data_dir, latest_insights)

            # Get corresponding posts file (same timestamp)
            timestamp = latest_insights.replace("insights_", "").replace(".json", "")
            posts_file = f"posts_{timestamp}.json"
            posts_path = os.path.join(self.data_dir, posts_file)

            # Load insights
            with open(insights_path, 'r', encoding='utf-8') as f:
                insights_data = json.load(f)

            # Load posts for detailed analysis
            posts_data = {}
            if os.path.exists(posts_path):
                with open(posts_path, 'r', encoding='utf-8') as f:
                    posts_data = json.load(f)

            # Process insights
            self._process_insights(insights_data, posts_data)
            self.last_update = datetime.now()

            logger.info(f"✅ Loaded Reddit data from {timestamp}")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading Reddit data: {e}")
            return False

    def _process_insights(self, insights: List[Dict], posts_data: Dict):
        """Process insights data into structured format"""
        self.sentiment_data = {}
        self.arbitrage_signals = []

        for insight in insights:
            insight_type = insight.get('type')
            generated_at = datetime.fromisoformat(insight.get('generated_at', datetime.now().isoformat()))

            if insight_type == 'ticker_sentiment':
                # Process ticker sentiment
                tickers = insight.get('tickers', [])
                for ticker in tickers:
                    sentiment_key = f"{ticker}_sentiment"

                    # Extract data from insight
                    data_points = insight.get('data_points', {})
                    avg_sentiment = data_points.get('avg_sentiment', 0)
                    total_mentions = data_points.get('total_mentions', 0)

                    # Create sentiment data object
                    sentiment_data = RedditSentimentData(
                        ticker=ticker,
                        total_mentions=total_mentions,
                        avg_sentiment=avg_sentiment,
                        bullish_count=0,  # Would need detailed post analysis
                        bearish_count=0,
                        neutral_count=0,
                        subreddits=insight.get('subreddits', []),
                        last_updated=generated_at,
                        confidence_score=insight.get('confidence', 0)
                    )

                    self.sentiment_data[ticker] = sentiment_data

            elif insight_type in ['arbitrage_opportunity']:
                # Process arbitrage signals
                signal = ArbitrageSignal(
                    signal_type=insight.get('type'),
                    confidence=insight.get('confidence', 0),
                    description=insight.get('description', ''),
                    tickers=insight.get('tickers', []),
                    subreddits=insight.get('subreddits', []),
                    detected_at=generated_at
                )
                self.arbitrage_signals.append(signal)

    def get_ticker_sentiment(self, ticker: str) -> Optional[RedditSentimentData]:
        """Get sentiment data for a specific ticker"""
        return self.sentiment_data.get(ticker.upper())

    def get_all_sentiment_data(self) -> Dict[str, RedditSentimentData]:
        """Get all available sentiment data"""
        return self.sentiment_data.copy()

    def get_arbitrage_signals(self, min_confidence: float = 0.0) -> List[ArbitrageSignal]:
        """Get arbitrage signals above confidence threshold"""
        return [s for s in self.arbitrage_signals if s.confidence >= min_confidence]

    def get_market_sentiment_summary(self) -> Dict[str, Any]:
        """Get overall market sentiment summary"""
        if not self.sentiment_data:
            return {"error": "No sentiment data available"}

        total_mentions = sum(data.total_mentions for data in self.sentiment_data.values())
        avg_sentiment = sum(data.avg_sentiment * data.total_mentions for data in self.sentiment_data.values()) / max(total_mentions, 1)

        # Top tickers by mentions
        top_tickers = sorted(
            self.sentiment_data.items(),
            key=lambda x: x[1].total_mentions,
            reverse=True
        )[:10]

        return {
            "total_tickers": len(self.sentiment_data),
            "total_mentions": total_mentions,
            "average_sentiment": avg_sentiment,
            "sentiment_trend": "bullish" if avg_sentiment > 0.1 else "bearish" if avg_sentiment < -0.1 else "neutral",
            "top_tickers": [
                {
                    "ticker": ticker,
                    "mentions": data.total_mentions,
                    "sentiment": data.avg_sentiment,
                    "confidence": data.confidence_score
                }
                for ticker, data in top_tickers
            ],
            "arbitrage_signals": len(self.arbitrage_signals),
            "last_update": self.last_update.isoformat() if self.last_update else None
        }

    def refresh_data(self) -> bool:
        """Refresh data from latest scrape files"""
        return self.load_latest_data()

    def export_for_trading_engine(self) -> Dict[str, Any]:
        """Export data in format suitable for trading engine"""
        return {
            "sentiment_signals": [
                {
                    "ticker": data.ticker,
                    "signal_type": "reddit_sentiment",
                    "strength": data.confidence_score,
                    "direction": "bullish" if data.avg_sentiment > 0 else "bearish",
                    "magnitude": abs(data.avg_sentiment),
                    "mentions": data.total_mentions,
                    "timestamp": data.last_updated.isoformat()
                }
                for data in self.sentiment_data.values()
                if data.confidence_score > 0.5  # Only high confidence signals
            ],
            "arbitrage_opportunities": [
                {
                    "type": signal.signal_type,
                    "confidence": signal.confidence,
                    "description": signal.description,
                    "tickers": signal.tickers,
                    "timestamp": signal.detected_at.isoformat()
                }
                for signal in self.arbitrage_signals
                if signal.confidence > 0.3  # Filter low confidence
            ],
            "market_summary": self.get_market_sentiment_summary()
        }


def demo_reddit_integration():
    """Demonstrate Reddit data integration"""
    print("AAC Reddit Data Integration Demo")
    print("=" * 40)

    integration = AACRedditIntegration()

    if integration.load_latest_data():
        print("✅ Successfully loaded Reddit data")

        # Show market summary
        summary = integration.get_market_sentiment_summary()
        print(f"\n📊 Market Summary:")
        print(f"   Total Tickers: {summary['total_tickers']}")
        print(f"   Total Mentions: {summary['total_mentions']}")
        print(".2f")
        print(f"   Sentiment Trend: {summary['sentiment_trend']}")
        print(f"   Arbitrage Signals: {summary['arbitrage_signals']}")

        # Show top tickers
        print(f"\n🔥 Top 10 Tickers:")
        for i, ticker_data in enumerate(summary['top_tickers'][:10], 1):
            print("2d")

        # Show arbitrage signals
        signals = integration.get_arbitrage_signals(0.3)
        if signals:
            print(f"\n🎯 Arbitrage Signals (confidence > 0.3):")
            for signal in signals[:5]:  # Show top 5
                print(f"   {signal.signal_type}: {signal.description[:60]}...")
                print(".2f")

        # Export for trading engine
        trading_data = integration.export_for_trading_engine()
        print(f"\n📤 Trading Engine Export:")
        print(f"   Sentiment Signals: {len(trading_data['sentiment_signals'])}")
        print(f"   Arbitrage Opportunities: {len(trading_data['arbitrage_opportunities'])}")

    else:
        print("❌ Failed to load Reddit data")
        print("Make sure the scraper has run and generated data files")


if __name__ == "__main__":
    demo_reddit_integration()