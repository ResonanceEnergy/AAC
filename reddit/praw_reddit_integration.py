#!/usr/bin/env python3
"""
AAC PRAW Reddit Integration
============================

Direct Reddit API access using PRAW for WallStreetBets sentiment analysis.
Provides real-time access to WallStreetBets discussions for arbitrage signals.

Features:
- Direct WallStreetBets access via PRAW
- Real-time hot posts and comments
- Sentiment analysis from post titles and content
- Stock ticker extraction and frequency analysis
- Arbitrage signal generation based on retail sentiment

API: https://www.reddit.com/dev/api/
Rate Limit: 600 requests per 10 minutes (for authenticated users)

For detailed API endpoint documentation, see: reddit_api_documentation.py
"""

import praw
import re
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import Counter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class PRAWConfig:
    """Configuration for PRAW Reddit API"""
    client_id: str
    client_secret: str
    user_agent: str
    username: Optional[str] = None
    password: Optional[str] = None
    subreddit: str = "wallstreetbets"
    post_limit: int = 100
    comment_limit: int = 50

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
        # $TICKER, TICKER$, or standalone TICKER
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

class PRAWRedditClient:
    """PRAW-based Reddit API client"""

    def __init__(self, config: PRAWConfig):
        self.config = config
        self.reddit: Optional[praw.Reddit] = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize PRAW Reddit client"""
        try:
            self.reddit = praw.Reddit(
                client_id=self.config.client_id,
                client_secret=self.config.client_secret,
                user_agent=self.config.user_agent,
                username=self.config.username,
                password=self.config.password
            )
            print("✅ PRAW Reddit client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize PRAW client: {e}")
            self.reddit = None

    def is_authenticated(self) -> bool:
        """Check if client is authenticated"""
        if not self.reddit:
            return False
        try:
            return self.reddit.user.me() is not None
        except:
            return False

    def get_hot_posts(self, limit: Optional[int] = None) -> List[RedditPost]:
        """
        Get hot posts from WallStreetBets

        Reddit API GET_hot Endpoint Documentation:
        ==========================================

        Endpoint: GET [/r/subreddit]/hot
        OAuth Scope: read
        RSS Support: Yes

        Description:
        This endpoint returns a listing of hot posts from the specified subreddit.
        Hot posts are determined by reddit's ranking algorithm which considers
        recency, score (upvotes - downvotes), and comment activity.

        Parameters:
        -----------
        g : str, optional
            Geographic region filter. One of:
            GLOBAL, US, AR, AU, BG, CA, CL, CO, HR, CZ, FI, FR, DE, GR, HU, IS,
            IN, IE, IT, JP, MY, MX, NZ, PH, PL, PT, PR, RO, RS, SG, ES, SE, TW,
            TH, TR, GB, US_WA, US_DE, US_DC, US_WI, US_WV, US_HI, US_FL, US_WY,
            US_NH, US_NJ, US_NM, US_TX, US_LA, US_NC, US_ND, US_NE, US_TN,
            US_NY, US_PA, US_CA, US_NV, US_VA, US_CO, US_AK, US_AL, US_AR,
            US_CT, US_DE, US_FL, US_GA, US_HI, US_ID, US_IL, US_IN, US_IA,
            US_KS, US_KY, US_LA, US_MA, US_MD, US_ME, US_MI, US_MN, US_MO,
            US_MS, US_MT, US_NC, US_ND, US_NE, US_NH, US_NJ, US_NM, US_NV,
            US_NY, US_OH, US_OK, US_OR, US_PA, US_RI, US_SC, US_SD, US_TN,
            US_TX, US_UT, US_VA, US_VT, US_WA, US_WI, US_WV, US_WY

        after : str, optional
            Fullname of a thing - used for pagination

        before : str, optional
            Fullname of a thing - used for pagination

        count : int, optional
            Number of items already seen in this listing (default: 0)

        limit : int, optional
            Maximum number of items to return (default: 25, maximum: 100)

        show : str, optional
            Optional parameter; if "all" is passed, filters such as
            "hide links that I have voted on" will be disabled

        sr_detail : bool, optional
            Expand subreddits in the response

        Response Format:
        ---------------
        Returns a JSON object with:
        - kind: "Listing"
        - data: Object containing:
            - after: Fullname for pagination
            - before: Fullname for pagination
            - children: Array of post objects
            - dist: Number of posts returned

        Rate Limits:
        -----------
        - 600 requests per 10 minutes for authenticated users
        - 60 requests per hour for non-authenticated users

        Usage in AAC:
        -------------
        This method retrieves hot posts from WallStreetBets for sentiment analysis
        and arbitrage signal generation. Hot posts represent the most engaging
        and timely discussions that may indicate market sentiment shifts.
        """
        if not self.reddit:
            print("❌ Reddit client not initialized")
            return []

        limit = limit or self.config.post_limit

        try:
            subreddit = self.reddit.subreddit(self.config.subreddit)
            posts = []

            for submission in subreddit.hot(limit=limit):
                post = RedditPost(
                    id=submission.id,
                    title=submission.title,
                    score=submission.score,
                    num_comments=submission.num_comments,
                    created_utc=submission.created_utc,
                    author=str(submission.author) if submission.author else "[deleted]",
                    url=submission.url,
                    selftext=submission.selftext
                )
                posts.append(post)

            print(f"✅ Retrieved {len(posts)} hot posts from r/{self.config.subreddit}")
            return posts

        except Exception as e:
            print(f"❌ Error fetching hot posts: {e}")
            return []

    def get_post_comments(self, post_id: str, limit: Optional[int] = None) -> List[str]:
        """Get comments from a specific post"""
        if not self.reddit:
            return []

        limit = limit or self.config.comment_limit

        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Remove "load more comments"

            comments = []
            for comment in submission.comments.list()[:limit]:
                if comment.body and len(comment.body.strip()) > 10:  # Filter short comments
                    comments.append(comment.body.strip())

            return comments

        except Exception as e:
            print(f"❌ Error fetching comments for post {post_id}: {e}")
            return []

class RedditSentimentAnalyzer:
    """Analyze Reddit sentiment for arbitrage signals"""

    def __init__(self, config: PRAWConfig):
        self.config = config
        self.client = PRAWRedditClient(config)

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

    async def get_sentiment_arbitrage_opportunities(self) -> List[SentimentSignal]:
        """Main method to get sentiment-based arbitrage opportunities"""
        print("🔍 Analyzing WallStreetBets sentiment for arbitrage opportunities...")

        # Get recent hot posts
        posts = self.client.get_hot_posts()

        if not posts:
            print("❌ No posts retrieved from Reddit")
            return []

        # Analyze ticker sentiment
        ticker_data = self.analyze_ticker_sentiment(posts)

        # Generate arbitrage signals
        signals = self.generate_arbitrage_signals(ticker_data)

        print(f"🎯 Found {len(signals)} potential sentiment arbitrage signals")

        return signals

async def test_praw_reddit_integration():
    """Test the PRAW Reddit integration"""
    print("🧪 Testing PRAW Reddit Integration")
    print("=" * 50)

    # Load configuration from environment
    config = PRAWConfig(
        client_id=os.getenv('REDDIT_CLIENT_ID', ''),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET', ''),
        user_agent=os.getenv('REDDIT_USER_AGENT', 'AAC-Arbitrage-Bot/1.0'),
        username=os.getenv('REDDIT_USERNAME'),
        password=os.getenv('REDDIT_PASSWORD')
    )

    # Check if credentials are configured
    if not config.client_id or not config.client_secret:
        print("❌ Reddit API credentials not configured!")
        print("Please run: python configure_api_keys.py")
        print("And set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
        return

    # Initialize analyzer
    analyzer = RedditSentimentAnalyzer(config)

    # Check authentication
    if analyzer.client.is_authenticated():
        print("✅ Authenticated Reddit access")
    else:
        print("⚠️  Read-only Reddit access (limited functionality)")

    # Get sentiment arbitrage opportunities
    signals = await analyzer.get_sentiment_arbitrage_opportunities()

    if signals:
        print("\n🎯 Top Sentiment Arbitrage Signals:")
        print("-" * 40)
        for i, signal in enumerate(signals[:10], 1):  # Show top 10
            sentiment_icon = "🚀" if signal.sentiment_score > 0 else "📉"
            print(f"{i}. {sentiment_icon} {signal.ticker}")
            print(f"   Sentiment: {signal.sentiment_score:.2f} (Confidence: {signal.confidence:.2f})")
            print(f"   Mentions: {signal.mentions}, Avg Score: {signal.avg_score:.1f}")
            print(f"   {signal.description}")
            print()
    else:
        print("❌ No arbitrage signals found")

    print("✅ PRAW Reddit integration test completed")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_praw_reddit_integration())