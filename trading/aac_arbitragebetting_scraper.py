"""
AAC ArbitrageBetting Daily Scraper
===================================

Daily scraper for r/arbitragebetting subreddit using PRAW.
Collects hot posts and analyzes betting arbitrage opportunities.

Features:
- Daily automated scraping
- Post data collection (title, score, comments, etc.)
- Basic arbitrage signal detection
- CSV export for analysis
- Integration with AAC arbitrage engine
"""

import praw
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, date
import os
import csv
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ArbitragePost:
    """Arbitrage betting post data structure"""
    id: str
    title: str
    score: int
    num_comments: int
    created_utc: float
    author: str
    url: str
    selftext: str
    upvote_ratio: float

    @property
    def created_datetime(self) -> datetime:
        """Convert UTC timestamp to datetime"""
        return datetime.fromtimestamp(self.created_utc)

    @property
    def age_hours(self) -> float:
        """Get post age in hours"""
        return (datetime.now() - self.created_datetime).total_seconds() / 3600


class AACArbitrageBettingScraper:
    """Daily scraper for r/arbitragebetting"""

    def __init__(self):
        """Initialize with Reddit API credentials"""
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = os.getenv('REDDIT_USER_AGENT', 'AAC-Arbitrage-Bot/1.0')
        self.username = os.getenv('REDDIT_USERNAME')
        self.password = os.getenv('REDDIT_PASSWORD')

        # Check credentials
        self.credentials_available = bool(self.client_id and self.client_secret)

        if self.credentials_available:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                username=self.username,
                password=self.password
            )
        else:
            self.reddit = None
            print("⚠️  Reddit credentials not found. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")

    def fetch_daily_posts(self, limit: int = 25, sort: str = 'hot') -> List[ArbitragePost]:
        """
        Fetch posts from r/arbitragebetting

        Args:
            limit: Number of posts to fetch
            sort: Sort method ('hot', 'new', 'top')

        Returns:
            List of ArbitragePost objects
        """
        posts = []

        if not self.reddit:
            print("❌ Reddit client not initialized")
            return []

        try:
            subreddit = self.reddit.subreddit("arbitragebetting")

            if sort == 'hot':
                submissions = subreddit.hot(limit=limit)
            elif sort == 'new':
                submissions = subreddit.new(limit=limit)
            elif sort == 'top':
                submissions = subreddit.top(limit=limit)
            else:
                submissions = subreddit.hot(limit=limit)

            for submission in submissions:
                post = ArbitragePost(
                    id=submission.id,
                    title=submission.title,
                    score=submission.score,
                    num_comments=submission.num_comments,
                    created_utc=submission.created_utc,
                    author=str(submission.author) if submission.author else "[deleted]",
                    url=submission.url,
                    selftext=submission.selftext,
                    upvote_ratio=submission.upvote_ratio
                )
                posts.append(post)

                print(f"📄 {submission.id} -- {submission.title[:60]}...")

        except Exception as e:
            print(f"❌ Error fetching posts: {e}")
            return []

        return posts

    def analyze_arbitrage_opportunities(self, posts: List[ArbitragePost]) -> List[Dict[str, Any]]:
        """
        Analyze posts for arbitrage betting opportunities

        Args:
            posts: List of ArbitragePost objects

        Returns:
            List of potential arbitrage opportunities
        """
        opportunities = []

        # Keywords that might indicate arbitrage opportunities
        arbitrage_keywords = [
            'arbitrage', 'arb', 'sure bet', 'risk free', 'guaranteed',
            'hedge', 'lock in', 'no risk', 'free money', 'mathematical',
            'probability', 'edge', 'advantage'
        ]

        for post in posts:
            # Check title and selftext for arbitrage keywords
            text_to_check = (post.title + " " + post.selftext).lower()

            keyword_matches = []
            for keyword in arbitrage_keywords:
                if keyword in text_to_check:
                    keyword_matches.append(keyword)

            if keyword_matches:
                opportunity = {
                    "post_id": post.id,
                    "title": post.title,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "author": post.author,
                    "url": post.url,
                    "created_datetime": post.created_datetime.isoformat(),
                    "age_hours": post.age_hours,
                    "matched_keywords": keyword_matches,
                    "confidence_score": len(keyword_matches) / len(arbitrage_keywords),  # Simple scoring
                    "scraped_date": date.today().isoformat()
                }
                opportunities.append(opportunity)

        return opportunities

    def export_to_csv(self, posts: List[ArbitragePost], opportunities: List[Dict[str, Any]],
                     filename_prefix: str = "arbitragebetting"):
        """
        Export scraped data to CSV files

        Args:
            posts: List of all posts
            opportunities: List of arbitrage opportunities
            filename_prefix: Prefix for output files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export all posts
        posts_filename = f"{filename_prefix}_posts_{timestamp}.csv"
        with open(posts_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'title', 'score', 'num_comments', 'author', 'url',
                         'created_datetime', 'age_hours', 'upvote_ratio']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for post in posts:
                writer.writerow({
                    'id': post.id,
                    'title': post.title,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'author': post.author,
                    'url': post.url,
                    'created_datetime': post.created_datetime.isoformat(),
                    'age_hours': post.age_hours,
                    'upvote_ratio': post.upvote_ratio
                })

        # Export arbitrage opportunities
        opportunities_filename = f"{filename_prefix}_opportunities_{timestamp}.csv"
        with open(opportunities_filename, 'w', newline='', encoding='utf-8') as csvfile:
            if opportunities:
                fieldnames = opportunities[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(opportunities)

        print(f"✅ Exported {len(posts)} posts to {posts_filename}")
        print(f"✅ Exported {len(opportunities)} opportunities to {opportunities_filename}")

    def run_daily_scrape(self):
        """Run the complete daily scraping pipeline"""
        print(f"🚀 Starting daily scrape for r/arbitragebetting - {date.today()}")

        # Fetch posts
        posts = self.fetch_daily_posts(limit=50, sort='hot')

        if not posts:
            print("❌ No posts fetched")
            return

        # Analyze for opportunities
        opportunities = self.analyze_arbitrage_opportunities(posts)

        # Export results
        self.export_to_csv(posts, opportunities)

        # Summary
        print("\n📊 Daily Scrape Summary:")
        print(f"   Posts scraped: {len(posts)}")
        print(f"   Arbitrage opportunities found: {len(opportunities)}")
        print(f"   Average post score: {sum(p.score for p in posts) / len(posts):.1f}")
        print(f"   Total comments: {sum(p.num_comments for p in posts)}")

        print("✅ Daily scrape completed successfully")


def main():
    """Main function for command line execution"""
    scraper = AACArbitrageBettingScraper()
    scraper.run_daily_scrape()


if __name__ == "__main__":
    main()
