"""
AAC Advanced Reddit Web Scraping Integration
=============================================

Enhanced WallStreetBets sentiment analysis using Selenium and Pushshift API.
Provides comprehensive scraping of Daily Discussion threads for arbitrage signals.

Based on: https://algotrading101.com/learn/reddit-wallstreetbets-web-scraping/

Features:
- Selenium browser automation for thread discovery
- Pushshift API for bulk comment data (no rate limits)
- Daily Discussion thread targeting
- Advanced stock ticker extraction
- Large-scale sentiment analysis (60k+ comments)
- Integration with AAC arbitrage engine

API Sources:
- Pushshift API: https://api.pushshift.io/reddit/
- Selenium WebDriver for browser automation
"""

import requests
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

import numpy as np
from collections import Counter
from datetime import date, timedelta, datetime
from dateutil.parser import parse
import time
import re
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json
import csv
import pandas as pd

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ScrapedRedditPost:
    """Data structure for scraped Reddit posts"""
    thread_id: str
    thread_title: str
    thread_date: date
    comment_count: int
    comments: List[str]
    tickers_mentioned: Dict[str, int]
    sentiment_score: float
    timestamp_scraped: datetime

@dataclass
class StockMention:
    """Stock ticker mention data"""
    ticker: str
    mention_count: int
    context_examples: List[str]
    sentiment_indicators: Dict[str, int]  # bullish/bearish/neutral counts

class AACRedditWebScraper:
    """
    Advanced Reddit web scraper for WallStreetBets sentiment analysis.

    Uses Selenium for browser automation and Pushshift API for bulk data.
    """

    def __init__(self, headless: bool = True):
        """
        Initialize the web scraper.

        Args:
            headless: Run browser in headless mode (no GUI)
        """
        self.headless = headless
        self.driver = None
        self.stocks_list = self._load_stocks_list()

        # Pushshift API base URLs
        self.pushshift_base = "https://api.pushshift.io/reddit"
        self.comment_ids_url = f"{self.pushshift_base}/submission/comment_ids/{{}}"
        self.comments_url = f"{self.pushshift_base}/comment/search"

        print("✅ AAC Reddit Web Scraper initialized")

    def _load_stocks_list(self) -> List[str]:
        """Load stock tickers list for analysis"""
        # Create a comprehensive list of US stock tickers
        # In production, this would be loaded from a file or database
        common_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'BABA', 'ORCL', 'CRM', 'AMD', 'INTC', 'UBER', 'LYFT', 'SPOT',
            'PYPL', 'SQ', 'SHOP', 'ETSY', 'PINS', 'SNAP', 'TWTR', 'SPCE',
            'PLTR', 'COIN', 'MSTR', 'RIVN', 'LCID', 'SOFI', 'UPST', 'ROKU',
            'ZM', 'DOCU', 'DDOG', 'NET', 'CRWD', 'ZS', 'OKTA', 'MDB', 'TEAM',
            'SPOT', 'FSLY', 'AFRM', 'HOOD', 'DKNG', 'RUM', 'FUBO', 'SE', 'BIDU',
            'JD', 'NTES', 'TCEHY', 'BILI', 'IQ', 'WB', 'MOMO', 'YY', 'HUYA',
            'VIPS', 'TAL', 'EDU', 'XPEV', 'LI', 'NIO', 'XPEV', 'BYDDF', 'TSM',
            'ASML', 'NVDA', 'AMD', 'QCOM', 'TXN', 'AVGO', 'MU', 'LRCX', 'KLAC',
            'AMAT', 'TER', 'ENTG', 'IPGP', 'CGNX', 'IRBT', 'ISRG', 'SYK', 'BSX',
            'MDT', 'ABT', 'TMO', 'DHR', 'A', 'IQV', 'ILMN', 'INCY', 'VRTX',
            'REGN', 'BIIB', 'GILD', 'AMGN', 'VTRS', 'PFE', 'JNJ', 'MRK', 'BMY',
            'LLY', 'ABBV', 'CVS', 'UNH', 'HUM', 'CI', 'ANTM', 'CNC', 'WCG',
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'DB', 'UBS', 'CS', 'RBS',
            'HSBC', 'BCS', 'SAN', 'ING', 'BNP', 'GLE', 'DBK', 'CBK', 'VOW',
            'BMW', 'MBG', 'PAH3', 'CON', 'BAS', 'SIE', 'ALV', 'DTE', 'EOAN',
            'RWE', 'ENR', 'IBE', 'ELE', 'ENG', 'REP', 'TEF', 'ITX', 'SAN', 'BBVA',
            'SAB', 'CABK', 'POP', 'BKT', 'ACS', 'FER', 'AENA', 'AMS', 'GRF',
            'IBE', 'ELE', 'ENG', 'REP', 'TEF', 'ITX', 'SAN', 'BBVA', 'SAB',
            'CABK', 'POP', 'BKT', 'ACS', 'FER', 'AENA', 'AMS', 'GRF', 'REE',
            'FDR', 'MAP', 'MTS', 'TL5', 'VNA', 'ZOT', 'ANA', 'IAG', 'SCYR',
            'VIS', 'SGRE', 'NTGY', 'ELE', 'ENG', 'REP', 'TEF', 'ITX', 'SAN',
            'BBVA', 'SAB', 'CABK', 'POP', 'BKT', 'ACS', 'FER', 'AENA', 'AMS',
            'GRF', 'REE', 'FDR', 'MAP', 'MTS', 'TL5', 'VNA', 'ZOT', 'ANA',
            'IAG', 'SCYR', 'VIS', 'SGRE', 'NTGY', 'RED', 'SYV', 'CLNX', 'MRL',
            'ACX', 'COL', 'MEL', 'CAF', 'EBRO', 'GSJ', 'TUBA', 'ENAG', 'IDR',
            'ARY', 'PSG', 'ECR', 'APE', 'CMC', 'DGI', 'CIE', 'APE', 'CMC',
            'DGI', 'CIE', 'OHL', 'RLIA', 'OHL', 'RLIA', 'FCC', 'TRE', 'ACS',
            'FER', 'AENA', 'AMS', 'GRF', 'REE', 'FDR', 'MAP', 'MTS', 'TL5',
            'VNA', 'ZOT', 'ANA', 'IAG', 'SCYR', 'VIS', 'SGRE', 'NTGY', 'RED',
            'SYV', 'CLNX', 'MRL', 'ACX', 'COL', 'MEL', 'CAF', 'EBRO', 'GSJ',
            'TUBA', 'ENAG', 'IDR', 'ARY', 'PSG', 'ECR', 'APE', 'CMC', 'DGI',
            'CIE', 'OHL', 'RLIA', 'FCC', 'TRE', 'ACS', 'FER', 'AENA', 'AMS',
            'GRF', 'REE', 'FDR', 'MAP', 'MTS', 'TL5', 'VNA', 'ZOT', 'ANA',
            'IAG', 'SCYR', 'VIS', 'SGRE', 'NTGY', 'RED', 'SYV', 'CLNX', 'MRL',
            'ACX', 'COL', 'MEL', 'CAF', 'EBRO', 'GSJ', 'TUBA', 'ENAG', 'IDR',
            'ARY', 'PSG', 'ECR', 'APE', 'CMC', 'DGI', 'CIE', 'OHL', 'RLIA',
            'FCC', 'TRE'
        ]

        return common_tickers

    def _setup_driver(self):
        """Set up Chrome WebDriver with appropriate options"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            print("✅ Chrome WebDriver initialized")
        except Exception as e:
            print(f"❌ Failed to initialize WebDriver: {e}")
            print("Make sure ChromeDriver is installed and in PATH")
            raise

    def _close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def find_daily_discussion_thread(self, target_date: Optional[date] = None) -> Optional[str]:
        """
        Find the Daily Discussion thread for a specific date.

        Args:
            target_date: Date to find thread for (default: yesterday)

        Returns:
            Thread ID if found, None otherwise
        """
        if target_date is None:
            target_date = date.today() - timedelta(days=1)

        if not self.driver:
            self._setup_driver()

        try:
            # Search for Daily Discussion threads
            search_url = "https://www.reddit.com/r/wallstreetbets/search/?q=flair%3A%22Daily%20Discussion%22&restrict_sr=1&sort=new"
            self.driver.get(search_url)

            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "_eYtD2XCVieq6emjKBH3m"))
            )

            # Find thread links
            thread_links = self.driver.find_elements(By.CLASS_NAME, "_eYtD2XCVieq6emjKBH3m")

            for link_element in thread_links:
                thread_text = link_element.text

                # Check for Daily Discussion threads
                if thread_text.startswith('Daily Discussion Thread'):
                    # Extract date from thread title
                    date_part = "".join(thread_text.split(' ')[-3:])
                    try:
                        thread_date = parse(date_part).date()
                        if thread_date == target_date:
                            # Get the thread URL
                            thread_url = link_element.find_element(By.XPATH, "../..").get_attribute('href')
                            thread_id = thread_url.split('/')[-3]
                            print(f"✅ Found Daily Discussion thread for {target_date}: {thread_id}")
                            return thread_id
                    except Exception as e:
                        continue

                # Check for Weekend Discussion threads
                elif thread_text.startswith('Weekend'):
                    try:
                        weekend_parts = thread_text.split(' ')
                        date_range = weekend_parts[-2]  # e.g., "28–30"

                        if "–" in date_range:
                            start_day = int(date_range.split("–")[0].replace(',', ''))
                            month_year = f"{weekend_parts[-3]} {weekend_parts[-1]}"

                            # Parse the start date
                            start_date_str = f"{month_year[:3]} {start_day}, {weekend_parts[-1]}"
                            thread_start_date = parse(start_date_str).date()

                            # Check if target date falls within weekend range
                            thread_end_date = thread_start_date + timedelta(days=2)  # Sat-Sun

                            if thread_start_date <= target_date <= thread_end_date:
                                thread_url = link_element.find_element(By.XPATH, "../..").get_attribute('href')
                                thread_id = thread_url.split('/')[-3]
                                print(f"✅ Found Weekend Discussion thread for {target_date}: {thread_id}")
                                return thread_id
                    except Exception as e:
                        continue

            print(f"❌ No Daily Discussion thread found for {target_date}")
            return None

        except Exception as e:
            print(f"❌ Error finding Daily Discussion thread: {e}")
            return None

    def get_comment_ids(self, thread_id: str) -> List[str]:
        """
        Get all comment IDs for a thread using Pushshift API.

        Args:
            thread_id: Reddit thread ID

        Returns:
            List of comment IDs
        """
        try:
            url = self.comment_ids_url.format(thread_id)
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()
            comment_ids = data.get('data', [])

            print(f"✅ Retrieved {len(comment_ids)} comment IDs for thread {thread_id}")
            return comment_ids

        except Exception as e:
            print(f"❌ Error getting comment IDs: {e}")
            return []

    def get_comments_batch(self, comment_ids: List[str], batch_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Get comment data in batches using Pushshift API.

        Args:
            comment_ids: List of comment IDs
            batch_size: Number of comments per batch

        Returns:
            List of comment data dictionaries
        """
        all_comments = []

        for i in range(0, len(comment_ids), batch_size):
            batch_ids = comment_ids[i:i + batch_size]
            ids_string = ",".join(batch_ids)

            try:
                params = {
                    'ids': ids_string,
                    'fields': 'body,author,id,score,created_utc',
                    'size': batch_size
                }

                response = requests.get(self.comments_url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()
                batch_comments = data.get('data', [])
                all_comments.extend(batch_comments)

                print(f"✅ Retrieved batch {i//batch_size + 1}: {len(batch_comments)} comments")

                # Rate limiting - be respectful to the API
                time.sleep(0.5)

            except Exception as e:
                print(f"❌ Error in batch {i//batch_size + 1}: {e}")
                continue

        print(f"✅ Total comments retrieved: {len(all_comments)}")
        return all_comments

    def analyze_ticker_mentions(self, comments: List[Dict[str, Any]]) -> Dict[str, StockMention]:
        """
        Analyze stock ticker mentions in comments.

        Args:
            comments: List of comment dictionaries

        Returns:
            Dictionary of ticker -> StockMention data
        """
        ticker_counter = Counter()
        ticker_contexts = {}

        # Sentiment keywords
        bullish_keywords = ['moon', 'to the moon', 'bullish', 'buy', 'long', 'calls', 'yolo', 'diamond hands', 'tendies']
        bearish_keywords = ['sell', 'short', 'bearish', 'puts', 'crash', 'dump', 'paper hands', 'rekt']

        for comment in comments:
            body = comment.get('body', '').upper()

            for ticker in self.stocks_list:
                if ticker in body:
                    ticker_counter[ticker] += 1

                    # Store context examples (first 100 chars around ticker)
                    if ticker not in ticker_contexts:
                        ticker_contexts[ticker] = []

                    # Find ticker in body and get context
                    ticker_pos = body.find(ticker)
                    if ticker_pos != -1:
                        start = max(0, ticker_pos - 50)
                        end = min(len(body), ticker_pos + len(ticker) + 50)
                        context = body[start:end].strip()
                        if len(ticker_contexts[ticker]) < 5:  # Limit examples
                            ticker_contexts[ticker].append(context)

        # Create StockMention objects
        stock_mentions = {}
        for ticker, count in ticker_counter.most_common():
            contexts = ticker_contexts.get(ticker, [])

            # Analyze sentiment in contexts
            sentiment_indicators = {'bullish': 0, 'bearish': 0, 'neutral': 0}

            for context in contexts:
                context_lower = context.lower()
                if any(keyword in context_lower for keyword in bullish_keywords):
                    sentiment_indicators['bullish'] += 1
                elif any(keyword in context_lower for keyword in bearish_keywords):
                    sentiment_indicators['bearish'] += 1
                else:
                    sentiment_indicators['neutral'] += 1

            stock_mentions[ticker] = StockMention(
                ticker=ticker,
                mention_count=count,
                context_examples=contexts,
                sentiment_indicators=sentiment_indicators
            )

        print(f"✅ Analyzed {len(stock_mentions)} tickers mentioned in {len(comments)} comments")
        return stock_mentions

    def scrape_daily_discussion(self, target_date: Optional[date] = None) -> Optional[ScrapedRedditPost]:
        """
        Complete scraping pipeline for Daily Discussion thread.

        Args:
            target_date: Date to scrape (default: yesterday)

        Returns:
            ScrapedRedditPost object or None if failed
        """
        if target_date is None:
            target_date = date.today() - timedelta(days=1)

        try:
            print(f"🔍 Starting scrape for Daily Discussion on {target_date}")

            # Find the thread
            thread_id = self.find_daily_discussion_thread(target_date)
            if not thread_id:
                return None

            # Get comment IDs
            comment_ids = self.get_comment_ids(thread_id)
            if not comment_ids:
                return None

            # Get comment data
            comments_data = self.get_comments_batch(comment_ids)

            # Extract comment bodies
            comment_bodies = [comment.get('body', '') for comment in comments_data if comment.get('body')]

            # Analyze ticker mentions
            ticker_analysis = self.analyze_ticker_mentions(comments_data)

            # Calculate overall sentiment score (simplified)
            total_mentions = sum(mention.mention_count for mention in ticker_analysis.values())
            bullish_mentions = sum(mention.sentiment_indicators['bullish'] for mention in ticker_analysis.values())
            bearish_mentions = sum(mention.sentiment_indicators['bearish'] for mention in ticker_analysis.values())

            if total_mentions > 0:
                sentiment_score = (bullish_mentions - bearish_mentions) / total_mentions
            else:
                sentiment_score = 0.0

            # Create result object
            result = ScrapedRedditPost(
                thread_id=thread_id,
                thread_title=f"Daily Discussion Thread - {target_date.strftime('%B %d, %Y')}",
                thread_date=target_date,
                comment_count=len(comment_bodies),
                comments=comment_bodies,
                tickers_mentioned={ticker: mention.mention_count for ticker, mention in ticker_analysis.items()},
                sentiment_score=sentiment_score,
                timestamp_scraped=datetime.now()
            )

            print(f"✅ Successfully scraped Daily Discussion thread")
            print(f"   Comments: {result.comment_count}")
            print(f"   Tickers mentioned: {len(result.tickers_mentioned)}")
            print(".2f")

            return result

        except Exception as e:
            print(f"❌ Error during scraping: {e}")
            return None

        finally:
            self._close_driver()

    def export_to_csv(self, stock_mentions: Dict[str, StockMention], filename: str = "wallstreetbets_sentiment.csv"):
        """
        Export ticker analysis to CSV file.

        Args:
            stock_mentions: Dictionary of stock mentions
            filename: Output filename
        """
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['ticker', 'mention_count', 'bullish', 'bearish', 'neutral', 'top_context']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for ticker, mention in stock_mentions.items():
                    writer.writerow({
                        'ticker': ticker,
                        'mention_count': mention.mention_count,
                        'bullish': mention.sentiment_indicators['bullish'],
                        'bearish': mention.sentiment_indicators['bearish'],
                        'neutral': mention.sentiment_indicators['neutral'],
                        'top_context': mention.context_examples[0] if mention.context_examples else ''
                    })

            print(f"✅ Exported data to {filename}")

        except Exception as e:
            print(f"❌ Error exporting to CSV: {e}")

    def export_to_dataframe(self, stock_mentions: Dict[str, StockMention]) -> pd.DataFrame:
        """
        Convert stock mentions to pandas DataFrame.

        Args:
            stock_mentions: Dictionary of stock mentions

        Returns:
            pandas DataFrame
        """
        data = []
        for ticker, mention in stock_mentions.items():
            data.append({
                'ticker': ticker,
                'mentions': mention.mention_count,
                'bullish': mention.sentiment_indicators['bullish'],
                'bearish': mention.sentiment_indicators['bearish'],
                'neutral': mention.sentiment_indicators['neutral'],
                'sentiment_ratio': (mention.sentiment_indicators['bullish'] - mention.sentiment_indicators['bearish']) / max(mention.mention_count, 1),
                'top_context': mention.context_examples[0] if mention.context_examples else ''
            })

        df = pd.DataFrame(data)
        df = df.sort_values('mentions', ascending=False)
        return df


def run_aac_web_scraping_demo():
    """Run the AAC Reddit web scraping demo"""
    print("AAC Advanced Reddit Web Scraping Demo")
    print("=" * 50)
    print("Using Selenium + Pushshift API for comprehensive sentiment analysis")
    print()

    scraper = AACRedditWebScraper(headless=True)

    try:
        # Scrape yesterday's Daily Discussion
        result = scraper.scrape_daily_discussion()

        if result:
            print()
            print("📊 Scraping Results:")
            print(f"Thread: {result.thread_title}")
            print(f"Comments: {result.comment_count:,}")
            print(f"Tickers Found: {len(result.tickers_mentioned)}")
            print(".2f")

            # Show top 10 mentioned tickers
            print()
            print("🔥 Top 10 Mentioned Tickers:")
            sorted_tickers = sorted(result.tickers_mentioned.items(), key=lambda x: x[1], reverse=True)
            for i, (ticker, count) in enumerate(sorted_tickers[:10], 1):
                print("2d")

            # Export to CSV
            print()
            print("💾 Exporting data...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"wallstreetbets_sentiment_{timestamp}.csv"

            # Get detailed analysis for export
            detailed_analysis = scraper.analyze_ticker_mentions([
                {'body': comment} for comment in result.comments
            ])

            scraper.export_to_csv(detailed_analysis, csv_filename)

            # Show DataFrame preview
            df = scraper.export_to_dataframe(detailed_analysis)
            print()
            print("📋 Data Preview:")
            print(df.head().to_string(index=False))

        else:
            print("❌ Failed to scrape Daily Discussion thread")
            print("This could be due to:")
            print("- No Daily Discussion thread for the target date")
            print("- Network connectivity issues")
            print("- Reddit website changes")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure you have:")
        print("- Chrome browser installed")
        print("- ChromeDriver in PATH")
        print("- Internet connection")
        print("- Required Python packages: selenium, requests, pandas")


if __name__ == "__main__":
    run_aac_web_scraping_demo()