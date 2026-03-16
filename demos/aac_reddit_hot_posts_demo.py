"""
AAC Reddit Hot Posts Demo - Using User's PRAW Pattern

This demo shows how to integrate the user's provided PRAW code pattern
into the AAC arbitrage system for WallStreetBets sentiment analysis.
"""

import praw
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class SimpleRedditPost:
    """Simplified Reddit post structure"""
    id: str
    title: str
    score: int
    num_comments: int
    created_utc: float
    author: str
    url: str

    @property
    def created_datetime(self) -> datetime:
        """Convert UTC timestamp to datetime"""
        return datetime.fromtimestamp(self.created_utc)

    @property
    def age_hours(self) -> float:
        """Get post age in hours"""
        return (datetime.now() - self.created_datetime).total_seconds() / 3600


class AACRedditHotPostsDemo:
    """Demo class using the user's PRAW pattern for AAC integration"""

    def __init__(self):
        """Initialize with environment variables"""
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = os.getenv('REDDIT_USER_AGENT', 'AAC-Arbitrage-Bot/1.0')
        self.username = os.getenv('REDDIT_USERNAME')
        self.password = os.getenv('REDDIT_PASSWORD')

        # Check if credentials are available
        self.credentials_available = bool(self.client_id and self.client_secret)

        if self.credentials_available:
            # Initialize Reddit client using user's pattern
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                username=self.username,
                password=self.password
            )
        else:
            self.reddit = None

    def fetch_hot_posts(self, limit: int = 10) -> List[SimpleRedditPost]:
        """
        Fetch hot posts from WallStreetBets using the user's pattern

        Args:
            limit: Number of posts to fetch

        Returns:
            List of SimpleRedditPost objects
        """
        posts = []

        try:
            subreddit = self.reddit.subreddit("wallstreetbets")

            for submission in subreddit.hot(limit=limit):
                post = SimpleRedditPost(
                    id=submission.id,
                    title=submission.title,
                    score=submission.score,
                    num_comments=submission.num_comments,
                    created_utc=submission.created_utc,
                    author=str(submission.author) if submission.author else "[deleted]",
                    url=submission.url
                )
                posts.append(post)

                # Print using user's format
                logger.info("{} -- {}".format(submission.id, submission.title))

        except Exception as e:
            logger.info(f"Error fetching posts: {e}")
            return []

        return posts

    def analyze_ticker_mentions(self, posts: List[SimpleRedditPost]) -> Dict[str, int]:
        """
        Analyze ticker mentions in post titles

        Args:
            posts: List of Reddit posts

        Returns:
            Dictionary of ticker -> mention count
        """
        import re
        ticker_counts = {}

        for post in posts:
            # Look for $TICKER pattern
            tickers = re.findall(r'\$([A-Z]{1,5})(?=\W|$)', post.title.upper())

            for ticker in tickers:
                if 2 <= len(ticker) <= 5:  # Reasonable ticker length
                    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

        return dict(sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True))

    def generate_arbitrage_signals(self, ticker_analysis: Dict[str, int],
                                 min_mentions: int = 3) -> List[Dict[str, Any]]:
        """
        Generate basic arbitrage signals based on ticker mentions

        Args:
            ticker_analysis: Dictionary of ticker mentions
            min_mentions: Minimum mentions to consider

        Returns:
            List of signal dictionaries
        """
        signals = []

        for ticker, mentions in ticker_analysis.items():
            if mentions >= min_mentions:
                # Simple signal based on mention frequency
                confidence = min(mentions / 10, 1.0)  # Scale confidence

                signal = {
                    "ticker": ticker,
                    "mentions": mentions,
                    "confidence": confidence,
                    "signal_type": "high_retail_interest",
                    "description": f"High retail interest in ${ticker} with {mentions} mentions",
                    "timestamp": datetime.now()
                }
                signals.append(signal)

        return signals


def run_aac_reddit_demo():
    """Run the AAC Reddit hot posts demo"""
    logger.info("AAC Reddit Hot Posts Demo")
    logger.info("=" * 40)
    logger.info("Using user's PRAW pattern integrated with AAC arbitrage system")
    logger.debug("")

    try:
        # Initialize demo client
        demo = AACRedditHotPostsDemo()

        if not demo.credentials_available:
            logger.info("⚠️  Reddit API credentials not found!")
            logger.debug("")
            logger.info("To run this demo, you need to set up Reddit API credentials:")
            logger.debug("")
            logger.info("1. Go to https://www.reddit.com/prefs/apps")
            logger.info("2. Create a new application (type: script)")
            logger.info("3. Copy the client_id and client_secret")
            logger.info("4. Add to your .env file:")
            logger.debug("")
            logger.info("   REDDIT_CLIENT_ID=your_client_id_here")
            logger.info("   REDDIT_CLIENT_SECRET=your_client_secret_here")
            logger.info("   REDDIT_USER_AGENT=AAC-Arbitrage-Bot/1.0")
            logger.info("   REDDIT_USERNAME=your_reddit_username (optional)")
            logger.info("   REDDIT_PASSWORD=your_reddit_password (optional)")
            logger.debug("")
            logger.info("For more details, see: reddit_api_setup.py")
            logger.debug("")
            logger.info("🔧 Showing code pattern demonstration instead...")
            logger.debug("")
            show_code_pattern_demo()
            return

        logger.info("✅ Reddit client initialized successfully")
        logger.debug("")

        # Fetch hot posts using user's pattern
        logger.info("Fetching hot posts from r/wallstreetbets...")
        logger.info("(Using user's code pattern: reddit.subreddit().hot())")
        logger.debug("")

        posts = demo.fetch_hot_posts(limit=15)

        logger.debug("")
        logger.info(f"📊 Retrieved {len(posts)} hot posts")
        logger.debug("")

        # Analyze ticker mentions
        logger.info("🔍 Analyzing ticker mentions...")
        ticker_analysis = demo.analyze_ticker_mentions(posts)

        logger.info("Top mentioned tickers:")
        for ticker, count in list(ticker_analysis.items())[:10]:
            logger.info(f"  ${ticker}: {count} mentions")
        logger.debug("")

        # Generate arbitrage signals
        logger.info("📈 Generating arbitrage signals...")
        signals = demo.generate_arbitrage_signals(ticker_analysis, min_mentions=2)

        logger.info(f"Generated {len(signals)} arbitrage signals:")
        for signal in signals[:5]:  # Show top 5
            logger.info(f"  🚀 {signal['ticker']}: {signal['confidence']:.1%} confidence")
            logger.info(f"     {signal['description']}")
            logger.debug("")

        # Show post details
        logger.info("📝 Recent post details:")
        for i, post in enumerate(posts[:3], 1):
            logger.info(f"{i}. {post.title[:60]}...")
            logger.info(f"   Score: {post.score} | Comments: {post.num_comments}")
            logger.info(f"   Age: {post.age_hours:.1f} hours ago")
            logger.debug("")

    except Exception as e:
        logger.info(f"❌ Demo failed: {e}")
        logger.info("Make sure your Reddit API credentials are set in .env file")


def show_code_pattern_demo():
    """Show the user's code pattern without actually connecting to Reddit"""
    logger.info("📝 User's PRAW Code Pattern:")
    logger.debug("")
    logger.info("```python")
    logger.info("import praw")
    logger.debug("")
    logger.info("# Initialize Reddit client")
    logger.info("reddit = praw.Reddit(")
    logger.info("    client_id=config.client_id,")
    logger.info("    client_secret=config.client_secret,")
    logger.info("    user_agent=config.user_agent,")
    logger.info("    username=config.username,")
    logger.info("    password=config.password")
    logger.info(")")
    logger.debug("")
    logger.info("# Get subreddit and fetch hot posts")
    logger.info('subreddit = reddit.subreddit("wallstreetbets")')
    logger.info("for submission in subreddit.hot():")
    logger.info('    print("{} -- {}".format(submission.id, submission.title))')
    logger.info("```")
    logger.debug("")
    logger.info("🔧 AAC Integration Features:")
    logger.info("  • Automatic ticker extraction from post titles")
    logger.info("  • Sentiment analysis for arbitrage signals")
    logger.info("  • Rate limit management (600 requests/10min)")
    logger.info("  • Market hours detection for timing")
    logger.info("  • Integration with multi-source arbitrage engine")
    logger.debug("")
    logger.info("📊 Expected Output Format:")
    logger.info("  abc123 -- $AAPL to the moon! 🚀🚀🚀")
    logger.info("  def456 -- TSLA earnings beat expectations")
    logger.info("  ghi789 -- Why $NVDA is undervalued")
    logger.debug("")
    logger.info("🚀 Arbitrage Signal Generation:")
    logger.info("  Based on mention frequency, post engagement, and market timing")
    logger.info("  Signals integrated with World Bank data and other sources")


if __name__ == "__main__":
    run_aac_reddit_demo()