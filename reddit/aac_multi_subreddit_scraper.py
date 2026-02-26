"""
AAC Multi-Subreddit Continuous Scraper
======================================

Continuous 24/7 scraper for multiple arbitrage and trading subreddits.
Runs every 15 minutes, processes data, generates insights, and integrates
with AAC arbitrage system.

Features:
- Multi-subreddit scraping (60+ subreddits)
- 15-minute intervals (24/7 operation)
- Advanced sentiment analysis
- Ticker extraction and analysis
- Insight generation and cataloging
- Data integration with AAC system
- Comprehensive logging and monitoring

Subreddits monitored:
- r/algotrading, r/ArbitrageEd, r/ARBITRAGE, r/ArbitrageFBA, r/statarb
- r/cryptoarbitrage, r/arbitrageExpert, r/arbitrageCT, r/ArbitrageTrading
- r/ArbiSwap, r/options, r/CentArbitrage, r/SearchArbitrageAFD
- r/quantfinance, r/FiverrArbitrage, r/Valuation, r/investing_discussion
- r/Flipping, r/quant, r/AfriqArbitrage, r/SideHustleGold
- r/bitcointrader, r/algorithmictrading, r/PriceActionTrading
- r/resellprofits, r/IndianStreetBets, r/defi, r/InvestmentsTrading
- r/stocks, r/InvestingandTrading, r/algobetting, r/valueinvestorsclub
- r/cryptocentralai, r/hogefinance, r/BanCongressTrading, r/Economics
- r/ZeroExchange, r/TradeVol, r/ETFInvesting, r/ValueInvesting
- r/btc, r/BitcoinMarkets, r/highfreqtrading, r/thetagang
- r/ethtrader, r/CryptoMarkets, r/CryptoQuantTrade, r/SocialArbitrageTradin
- r/CryptoCurrency, r/CryptoMoonShots, r/swingtrading, r/p2pcryptoexchanges
- r/CryptoHopper, r/Wallstreetsilver, r/wallstreet, r/wallstreetdd
- r/WallStreetOasis, r/WallStreetSiren, r/wallstreet_apes
- r/WallstreetSluts, r/WallStreetNYSE, r/wallstreetbetsGER
- r/WallStreetbetsELITE, r/Wallstreetbetsnew, r/occupywallstreet
- r/wallstreetbetsOGs, r/WallStreetBetsCrypto, r/WallstreetBreakers
- r/WallStreetVR, r/wallstreetbets2, r/WallStreetBaggers
- r/wallstreetInvestment, r/wallstreetbets_wins, r/wallstreetsmallcaps
- r/Superstonk, r/WallStreetBetsTopMost, r/GME, r/GMEJungle
- r/DDintoGME, r/gmeoptions
"""

import praw
import json
import csv
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import re
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from textblob import TextBlob
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aac_reddit_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RedditPost:
    """Comprehensive Reddit post data structure"""
    id: str
    subreddit: str
    title: str
    selftext: str
    score: int
    num_comments: int
    created_utc: float
    author: str
    url: str
    upvote_ratio: float
    sentiment_score: float
    tickers_mentioned: List[str]
    arbitrage_signals: List[str]
    scraped_at: datetime

@dataclass
class SubredditMetrics:
    """Metrics for each subreddit scrape"""
    subreddit: str
    posts_scraped: int
    total_score: int
    avg_sentiment: float
    top_tickers: Dict[str, int]
    arbitrage_signals: int
    scraped_at: datetime

@dataclass
class ArbitrageInsight:
    """Generated arbitrage insights"""
    insight_id: str
    type: str  # 'ticker_sentiment', 'arbitrage_opportunity', 'market_trend'
    confidence: float
    description: str
    tickers: List[str]
    subreddits: List[str]
    generated_at: datetime
    data_points: Dict[str, Any]

class AACMultiSubredditScraper:
    """
    Continuous multi-subreddit scraper for AAC arbitrage system
    """

    def __init__(self):
        """Initialize the scraper with credentials and configuration"""
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = os.getenv('REDDIT_USER_AGENT', 'AAC-Arbitrage-Bot/1.0')
        self.username = os.getenv('REDDIT_USERNAME')
        self.password = os.getenv('REDDIT_PASSWORD')

        # Subreddits to monitor
        self.subreddits = [
            'algotrading', 'ArbitrageEd', 'ARBITRAGE', 'ArbitrageFBA', 'statarb',
            'cryptoarbitrage', 'arbitrageExpert', 'arbitrageCT', 'ArbitrageTrading',
            'ArbiSwap', 'options', 'CentArbitrage', 'SearchArbitrageAFD',
            'quantfinance', 'FiverrArbitrage', 'Valuation', 'investing_discussion',
            'Flipping', 'quant', 'AfriqArbitrage', 'SideHustleGold',
            'bitcointrader', 'algorithmictrading', 'PriceActionTrading',
            'resellprofits', 'IndianStreetBets', 'defi', 'InvestmentsTrading',
            'stocks', 'InvestingandTrading', 'algobetting', 'valueinvestorsclub',
            'cryptocentralai', 'hogefinance', 'BanCongressTrading', 'Economics',
            'ZeroExchange', 'TradeVol', 'ETFInvesting', 'ValueInvesting',
            'btc', 'BitcoinMarkets', 'highfreqtrading', 'thetagang',
            'ethtrader', 'CryptoMarkets', 'CryptoQuantTrade', 'SocialArbitrageTradin',
            'CryptoCurrency', 'CryptoMoonShots', 'swingtrading', 'p2pcryptoexchanges',
            'CryptoHopper', 'Wallstreetsilver', 'wallstreet', 'wallstreetdd',
            'WallStreetOasis', 'WallStreetSiren', 'wallstreet_apes',
            'WallstreetSluts', 'WallStreetNYSE', 'wallstreetbetsGER',
            'WallStreetbetsELITE', 'Wallstreetbetsnew', 'occupywallstreet',
            'wallstreetbetsOGs', 'WallStreetBetsCrypto', 'WallstreetBreakers',
            'WallStreetVR', 'wallstreetbets2', 'WallStreetBaggers',
            'wallstreetInvestment', 'wallstreetbets_wins', 'wallstreetsmallcaps',
            'Superstonk', 'WallStreetBetsTopMost', 'GME', 'GMEJungle',
            'DDintoGME', 'gmeoptions'
        ]

        # Initialize Reddit client
        self.reddit = None
        self._init_reddit_client()

        # Data storage
        self.data_dir = 'data/reddit_scrapes'
        os.makedirs(self.data_dir, exist_ok=True)

        # Sentiment keywords
        self.bullish_keywords = [
            'moon', 'to the moon', 'bullish', 'buy', 'long', 'calls', 'yolo',
            'diamond hands', 'tendies', 'green', 'up', 'pump', 'breakout'
        ]
        self.bearish_keywords = [
            'sell', 'short', 'bearish', 'puts', 'crash', 'dump', 'paper hands',
            'rekt', 'red', 'down', 'dump', 'bear market'
        ]

        # Arbitrage keywords
        self.arbitrage_keywords = [
            'arbitrage', 'arb', 'spread', 'mispricing', 'inefficiency',
            'statistical arbitrage', 'pairs trading', ' triangular arbitrage'
        ]

        logger.info(f"✅ AAC Multi-Subreddit Scraper initialized for {len(self.subreddits)} subreddits")

    def _init_reddit_client(self):
        """Initialize Reddit API client"""
        if not self.client_id or not self.client_secret:
            logger.error("❌ Reddit credentials not found. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
            return

        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                username=self.username,
                password=self.password
            )
            logger.info("✅ Reddit client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Reddit client: {e}")

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity  # -1 to 1
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {e}")
            return 0.0

    def extract_tickers(self, text: str) -> List[str]:
        """Extract stock/crypto tickers from text"""
        # Look for $TICKER pattern
        dollar_tickers = re.findall(r'\$([A-Z]{1,5})(?=\W|$)', text.upper())

        # Look for word-boundary tickers (common stocks)
        word_tickers = re.findall(r'\b([A-Z]{2,5})\b', text.upper())

        # Filter out common words that aren't tickers
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'HOT', 'BUT', 'SOME', 'WHAT', 'THERE', 'WHEN', 'FROM', 'YOUR', 'HOW', 'EACH', 'WHICH', 'THEIR', 'TIME', 'WILL', 'ABOUT', 'WOULD', 'THERE', 'COULD', 'OTHER'}

        tickers = []
        for ticker in dollar_tickers + word_tickers:
            if len(ticker) >= 2 and len(ticker) <= 5 and ticker not in common_words:
                tickers.append(ticker)

        return list(set(tickers))  # Remove duplicates

    def detect_arbitrage_signals(self, title: str, selftext: str) -> List[str]:
        """Detect potential arbitrage signals in post content"""
        signals = []
        content = f"{title} {selftext}".lower()

        for keyword in self.arbitrage_keywords:
            if keyword.lower() in content:
                signals.append(f"keyword_{keyword.replace(' ', '_')}")

        # Look for specific arbitrage patterns
        if 'stat arb' in content or 'statistical arbitrage' in content:
            signals.append('statistical_arbitrage')
        if 'pairs trading' in content:
            signals.append('pairs_trading')
        if 'triangular' in content and 'arbitrage' in content:
            signals.append('triangular_arbitrage')
        if 'cross-exchange' in content or 'cross exchange' in content:
            signals.append('cross_exchange_arb')

        return signals

    def scrape_subreddit(self, subreddit_name: str, limit: int = 25) -> List[RedditPost]:
        """Scrape posts from a specific subreddit"""
        posts = []

        if not self.reddit:
            logger.error("Reddit client not initialized")
            return posts

        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            scraped_at = datetime.now()

            # Get hot posts (most relevant for current sentiment)
            for submission in subreddit.hot(limit=limit):
                # Combine title and selftext for analysis
                full_text = f"{submission.title} {submission.selftext}"

                # Analyze content
                sentiment_score = self.analyze_sentiment(full_text)
                tickers = self.extract_tickers(full_text)
                arbitrage_signals = self.detect_arbitrage_signals(submission.title, submission.selftext)

                post = RedditPost(
                    id=submission.id,
                    subreddit=subreddit_name,
                    title=submission.title,
                    selftext=submission.selftext,
                    score=submission.score,
                    num_comments=submission.num_comments,
                    created_utc=submission.created_utc,
                    author=str(submission.author) if submission.author else '[deleted]',
                    url=submission.url,
                    upvote_ratio=getattr(submission, 'upvote_ratio', 0.5),
                    sentiment_score=sentiment_score,
                    tickers_mentioned=tickers,
                    arbitrage_signals=arbitrage_signals,
                    scraped_at=scraped_at
                )

                posts.append(post)

            logger.info(f"✅ Scraped {len(posts)} posts from r/{subreddit_name}")

        except Exception as e:
            logger.error(f"❌ Error scraping r/{subreddit_name}: {e}")

        return posts

    def scrape_all_subreddits(self) -> Dict[str, List[RedditPost]]:
        """Scrape all configured subreddits"""
        all_posts = {}
        total_posts = 0

        logger.info("🚀 Starting multi-subreddit scrape...")

        for subreddit in self.subreddits:
            posts = self.scrape_subreddit(subreddit)
            all_posts[subreddit] = posts
            total_posts += len(posts)

            # Rate limiting - be respectful to Reddit API
            time.sleep(1)

        logger.info(f"✅ Completed scraping {total_posts} posts from {len(self.subreddits)} subreddits")
        return all_posts

    def generate_subreddit_metrics(self, posts_data: Dict[str, List[RedditPost]]) -> List[SubredditMetrics]:
        """Generate metrics for each subreddit"""
        metrics = []

        for subreddit, posts in posts_data.items():
            if not posts:
                continue

            total_score = sum(post.score for post in posts)
            avg_sentiment = sum(post.sentiment_score for post in posts) / len(posts)

            # Count ticker mentions
            ticker_counter = Counter()
            arbitrage_signals = 0

            for post in posts:
                for ticker in post.tickers_mentioned:
                    ticker_counter[ticker] += 1
                arbitrage_signals += len(post.arbitrage_signals)

            top_tickers = dict(ticker_counter.most_common(10))

            metric = SubredditMetrics(
                subreddit=subreddit,
                posts_scraped=len(posts),
                total_score=total_score,
                avg_sentiment=avg_sentiment,
                top_tickers=top_tickers,
                arbitrage_signals=arbitrage_signals,
                scraped_at=datetime.now()
            )

            metrics.append(metric)

        return metrics

    def generate_insights(self, posts_data: Dict[str, List[RedditPost]]) -> List[ArbitrageInsight]:
        """Generate arbitrage insights from scraped data"""
        insights = []

        # Aggregate data across all subreddits
        all_tickers = Counter()
        sentiment_by_ticker = defaultdict(list)
        arbitrage_signals = Counter()

        for subreddit, posts in posts_data.items():
            for post in posts:
                for ticker in post.tickers_mentioned:
                    all_tickers[ticker] += 1
                    sentiment_by_ticker[ticker].append(post.sentiment_score)

                for signal in post.arbitrage_signals:
                    arbitrage_signals[signal] += 1

        # Generate ticker sentiment insights
        for ticker, mentions in all_tickers.most_common(20):
            sentiments = sentiment_by_ticker[ticker]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

            confidence = min(mentions / 50, 1.0)  # Scale confidence by mention volume

            if abs(avg_sentiment) > 0.1 and mentions >= 3:  # Significant sentiment
                insight_type = 'bullish_sentiment' if avg_sentiment > 0 else 'bearish_sentiment'

                insight = ArbitrageInsight(
                    insight_id=f"{ticker}_{insight_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    type='ticker_sentiment',
                    confidence=confidence,
                    description=f"{'Bullish' if avg_sentiment > 0 else 'Bearish'} sentiment for ${ticker} "
                               f"(avg: {avg_sentiment:.2f}, mentions: {mentions})",
                    tickers=[ticker],
                    subreddits=list(set(post.subreddit for posts in posts_data.values()
                                      for post in posts if ticker in post.tickers_mentioned)),
                    generated_at=datetime.now(),
                    data_points={
                        'avg_sentiment': avg_sentiment,
                        'total_mentions': mentions,
                        'sentiment_distribution': sentiments
                    }
                )
                insights.append(insight)

        # Generate arbitrage opportunity insights
        for signal_type, count in arbitrage_signals.most_common():
            if count >= 2:  # At least 2 mentions
                insight = ArbitrageInsight(
                    insight_id=f"{signal_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    type='arbitrage_opportunity',
                    confidence=min(count / 10, 1.0),
                    description=f"Potential {signal_type.replace('_', ' ')} opportunities detected "
                               f"({count} mentions across subreddits)",
                    tickers=[],  # Could be enhanced to link specific tickers
                    subreddits=list(set(post.subreddit for posts in posts_data.values()
                                      for post in posts if signal_type in post.arbitrage_signals)),
                    generated_at=datetime.now(),
                    data_points={'signal_count': count, 'signal_type': signal_type}
                )
                insights.append(insight)

        logger.info(f"✅ Generated {len(insights)} arbitrage insights")
        return insights

    def save_data(self, posts_data: Dict[str, List[RedditPost]],
                  metrics: List[SubredditMetrics], insights: List[ArbitrageInsight]):
        """Save all scraped data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save posts data
        posts_file = os.path.join(self.data_dir, f"posts_{timestamp}.json")
        with open(posts_file, 'w', encoding='utf-8') as f:
            json.dump({sub: [asdict(post) for post in posts]
                      for sub, posts in posts_data.items()}, f, indent=2, default=str)

        # Save metrics
        metrics_file = os.path.join(self.data_dir, f"metrics_{timestamp}.csv")
        with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
            if metrics:
                writer = csv.DictWriter(f, fieldnames=asdict(metrics[0]).keys())
                writer.writeheader()
                for metric in metrics:
                    writer.writerow(asdict(metric))

        # Save insights
        insights_file = os.path.join(self.data_dir, f"insights_{timestamp}.json")
        with open(insights_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(insight) for insight in insights], f, indent=2, default=str)

        logger.info(f"✅ Data saved: {posts_file}, {metrics_file}, {insights_file}")

    def run_scrape_cycle(self):
        """Run one complete scrape cycle"""
        try:
            logger.info("🔄 Starting scrape cycle...")

            # Scrape all subreddits
            posts_data = self.scrape_all_subreddits()

            # Generate metrics
            metrics = self.generate_subreddit_metrics(posts_data)

            # Generate insights
            insights = self.generate_insights(posts_data)

            # Save data
            self.save_data(posts_data, metrics, insights)

            # Log summary
            total_posts = sum(len(posts) for posts in posts_data.values())
            total_insights = len(insights)

            logger.info(f"✅ Scrape cycle completed: {total_posts} posts, {total_insights} insights")

        except Exception as e:
            logger.error(f"❌ Error in scrape cycle: {e}")

    def run_continuous_scraping(self):
        """Run continuous scraping with 15-minute intervals"""
        logger.info("🚀 Starting continuous 24/7 scraping (15-minute intervals)")

        # Use background scheduler for continuous operation
        scheduler = BackgroundScheduler()
        scheduler.add_job(self.run_scrape_cycle, 'interval', minutes=15)
        scheduler.start()

        logger.info("✅ Continuous scraping started. Press Ctrl+C to stop.")

        try:
            # Keep the main thread alive
            while True:
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("🛑 Stopping continuous scraping...")
            scheduler.shutdown()
            logger.info("✅ Continuous scraping stopped")

    def run_single_scrape(self):
        """Run a single scrape cycle for testing"""
        logger.info("🧪 Running single scrape cycle for testing...")
        self.run_scrape_cycle()
        logger.info("✅ Single scrape cycle completed")


def main():
    """Main entry point"""
    scraper = AACMultiSubredditScraper()

    if not scraper.reddit:
        logger.error("Cannot start scraping without Reddit credentials")
        return

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--single':
        scraper.run_single_scrape()
    else:
        scraper.run_continuous_scraping()


if __name__ == "__main__":
    main()