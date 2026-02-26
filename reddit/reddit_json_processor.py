#!/usr/bin/env python3
"""
AAC Reddit JSON Data Processor
===============================

Demonstrates how the PRAW integration processes raw Reddit API JSON data
to extract sentiment signals for arbitrage opportunities.

This script shows how the JSON response from Reddit's API gets converted
into actionable trading signals.
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import Counter

@dataclass
class RedditPost:
    """Reddit post data structure"""
    id: str
    title: str
    score: int
    num_comments: int
    created_utc: float
    author: str
    url: str
    selftext: str = ""
    tickers: List[str] = None

    def __post_init__(self):
        if self.tickers is None:
            self.tickers = self.extract_tickers()

    def extract_tickers(self) -> List[str]:
        """Extract stock tickers from title and content"""
        text = f"{self.title} {self.selftext}"

        # Common stock ticker patterns
        patterns = [
            r'\$([A-Z]{1,5})(?=\W|$)',  # $TICKER
            r'\b([A-Z]{1,5})\$',        # TICKER$
            r'\b([A-Z]{1,5})\b'         # standalone TICKER (3-5 chars)
        ]

        tickers = []
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            tickers.extend(matches)

        # Filter out common non-ticker words
        exclude_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN',
            'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'HOT', 'BUT', 'SHE',
            'NEW', 'NOW', 'WIN', 'BID', 'ASK', 'CEO', 'IPO', 'EPS', 'GDP',
            'USD', 'BTC', 'ETH', 'LTC', 'DOGE', 'ADA', 'SOL', 'DOT', 'LINK',
            'UNI', 'AAVE', 'COMP', 'MKR', 'SUSHI', 'YFI', 'BAL', 'REN', 'OMG',
            'BAT', 'ZRX', 'REP', 'GNT', 'STORJ', 'ANT', 'FUN', 'SNT', 'PAY',
            'QTM', 'BTU', 'BOIL', 'KOLD', 'UNG', 'USO', 'UWT', 'DWT', 'VXX',
            'UVXY', 'SVXY', 'TQQQ', 'SQQQ', 'SPY', 'QQQ', 'IWM', 'EFA', 'VWO',
            'BND', 'VNQ', 'GLD', 'SLV', 'USO', 'DBC', 'IEF', 'TLT', 'HYG', 'LQD'
        }

        # Keep only tickers that are likely real (3-5 chars, not excluded)
        filtered_tickers = []
        for ticker in tickers:
            if 2 <= len(ticker) <= 5 and ticker not in exclude_words:
                filtered_tickers.append(ticker)

        return list(set(filtered_tickers))  # Remove duplicates

@dataclass
class SentimentSignal:
    """Sentiment-based arbitrage signal"""
    ticker: str
    sentiment_score: float  # -1 to 1 (negative to positive)
    confidence: float  # 0 to 1
    mentions: int
    recent_posts: int
    avg_score: float
    description: str
    timestamp: datetime

class RedditJSONProcessor:
    """Process raw Reddit API JSON data"""

    def __init__(self):
        self.posts = []

    def process_reddit_json(self, json_data: Dict[str, Any]) -> List[RedditPost]:
        """Process raw Reddit API JSON response"""
        posts = []

        try:
            # Extract posts from the listing
            if 'data' in json_data and 'children' in json_data['data']:
                for child in json_data['data']['children']:
                    if child.get('kind') == 't3':  # t3 = post
                        post_data = child['data']

                        # Create RedditPost object
                        post = RedditPost(
                            id=post_data.get('id', ''),
                            title=post_data.get('title', ''),
                            score=post_data.get('score', 0),
                            num_comments=post_data.get('num_comments', 0),
                            created_utc=post_data.get('created_utc', 0),
                            author=post_data.get('author', '[deleted]'),
                            url=post_data.get('url', ''),
                            selftext=post_data.get('selftext', '')
                        )
                        posts.append(post)

            print(f"✅ Processed {len(posts)} posts from Reddit JSON")
            self.posts = posts
            return posts

        except Exception as e:
            print(f"❌ Error processing Reddit JSON: {e}")
            return []

    def analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis based on keywords"""
        positive_words = {
            'moon', 'bullish', 'bull', 'buy', 'long', 'up', 'green', 'profit',
            'gain', 'rise', 'higher', 'breakout', 'squeeze', 'diamond', 'hands',
            'hodl', 'diamond hands', 'yolo', 'tendies', 'stonks', 'pump'
        }

        negative_words = {
            'bearish', 'bear', 'sell', 'short', 'down', 'red', 'loss', 'drop',
            'lower', 'crash', 'dump', 'rekt', 'paper hands', 'selloff', 'bear trap'
        }

        words = re.findall(r'\b\w+\b', text.lower())
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words == 0:
            return 0.0

        return (positive_count - negative_count) / total_sentiment_words

    def analyze_ticker_sentiment(self, posts: List[RedditPost]) -> Dict[str, Dict]:
        """Analyze sentiment for each ticker mentioned"""
        ticker_data = {}

        for post in posts:
            sentiment = self.analyze_sentiment(f"{post.title} {post.selftext}")

            for ticker in post.tickers:
                if ticker not in ticker_data:
                    ticker_data[ticker] = {
                        'mentions': 0,
                        'total_sentiment': 0.0,
                        'posts': [],
                        'total_score': 0,
                        'total_comments': 0
                    }

                ticker_data[ticker]['mentions'] += 1
                ticker_data[ticker]['total_sentiment'] += sentiment
                ticker_data[ticker]['posts'].append({
                    'id': post.id,
                    'title': post.title,
                    'sentiment': sentiment,
                    'score': post.score,
                    'comments': post.num_comments
                })
                ticker_data[ticker]['total_score'] += post.score
                ticker_data[ticker]['total_comments'] += post.num_comments

        # Calculate averages
        for ticker, data in ticker_data.items():
            data['avg_sentiment'] = data['total_sentiment'] / data['mentions']
            data['avg_score'] = data['total_score'] / data['mentions']
            data['avg_comments'] = data['total_comments'] / data['mentions']

        return ticker_data

    def generate_arbitrage_signals(self, ticker_data: Dict[str, Dict],
                                 min_mentions: int = 3) -> List[SentimentSignal]:
        """Generate arbitrage signals based on sentiment analysis"""
        signals = []

        for ticker, data in ticker_data.items():
            if data['mentions'] < min_mentions:
                continue

            # Calculate confidence based on consistency and volume
            sentiment_consistency = abs(data['avg_sentiment'])
            volume_factor = min(data['mentions'] / 10, 1.0)  # Cap at 10 mentions
            score_factor = min(data['avg_score'] / 100, 1.0)  # Cap at 100 avg score

            confidence = (sentiment_consistency * 0.4 + volume_factor * 0.4 + score_factor * 0.2)

            # Determine signal strength
            if abs(data['avg_sentiment']) > 0.3 and confidence > 0.5:
                signal = SentimentSignal(
                    ticker=ticker,
                    sentiment_score=data['avg_sentiment'],
                    confidence=confidence,
                    mentions=data['mentions'],
                    recent_posts=len(data['posts']),
                    avg_score=data['avg_score'],
                    description=f"Reddit sentiment: {'Bullish' if data['avg_sentiment'] > 0 else 'Bearish'} "
                               f"({data['avg_sentiment']:.2f}) with {data['mentions']} mentions, "
                               f"avg score: {data['avg_score']:.1f}",
                    timestamp=datetime.now()
                )
                signals.append(signal)

        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals

def process_reddit_json_demo(json_text: str):
    """Demonstrate processing of Reddit JSON data"""
    print("🚀 AAC Reddit JSON Processing Demo")
    print("=" * 50)

    # Parse JSON
    try:
        reddit_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return

    # Process the data
    processor = RedditJSONProcessor()
    posts = processor.process_reddit_json(reddit_data)

    if not posts:
        print("❌ No posts found in JSON data")
        return

    # Show sample posts
    print(f"\n📊 Sample Posts ({len(posts)} total):")
    print("-" * 40)
    for i, post in enumerate(posts[:5], 1):  # Show first 5
        print(f"{i}. {post.title[:60]}{'...' if len(post.title) > 60 else ''}")
        print(f"   Score: {post.score}, Comments: {post.num_comments}")
        print(f"   Tickers: {post.tickers}")
        print(f"   Author: {post.author}")
        print()

    # Analyze sentiment
    ticker_data = processor.analyze_ticker_sentiment(posts)

    print("📈 Ticker Sentiment Analysis:")
    print("-" * 40)
    for ticker, data in sorted(ticker_data.items(), key=lambda x: x[1]['mentions'], reverse=True)[:10]:
        print(f"{ticker}: {data['mentions']} mentions, "
              f"Avg Sentiment: {data['avg_sentiment']:.2f}, "
              f"Avg Score: {data['avg_score']:.1f}")

    # Generate signals
    signals = processor.generate_arbitrage_signals(ticker_data)

    print(f"\n🎯 Arbitrage Signals Generated: {len(signals)}")
    print("-" * 40)
    for i, signal in enumerate(signals[:5], 1):  # Show top 5
        sentiment_icon = "🚀" if signal.sentiment_score > 0 else "📉"
        print(f"{i}. {sentiment_icon} {signal.ticker}")
        print(f"   Sentiment: {signal.sentiment_score:.2f} (Confidence: {signal.confidence:.2f})")
        print(f"   Mentions: {signal.mentions}, Avg Score: {signal.avg_score:.1f}")
        print(f"   {signal.description}")
        print()

    print("✅ Reddit JSON processing demo completed!")

if __name__ == "__main__":
    # This would be called with the JSON string provided by the user
    # For demo purposes, we'll show how it works
    print("Reddit JSON Processing Demo")
    print("This script processes the raw Reddit API JSON you provided.")
    print("To use it, call: process_reddit_json_demo(your_json_string)")