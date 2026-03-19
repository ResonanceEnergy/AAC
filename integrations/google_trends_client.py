#!/usr/bin/env python3
"""
Google Trends Integration
==========================
Tracks search interest for tickers, crypto, and financial terms
using the pytrends library (unofficial Google Trends API).

No API key required — uses public Google Trends data.

Requires:
    - pip install pytrends
"""

import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False


class GoogleTrendsClient:
    """
    Google Trends integration for market sentiment signals.

    Use cases:
        - Track retail interest in specific tickers (AAPL, TSLA, etc.)
        - Monitor crypto search trends (Bitcoin, Ethereum, etc.)
        - Detect breakout interest before price moves
        - Compare relative interest between assets
        - Get trending related queries
    """

    def __init__(
        self,
        hl: str = 'en-US',
        tz: int = 360,
        retries: int = 2,
        backoff_seconds: float = 2.0,
    ):
        """
        Args:
            hl: Language/locale
            tz: Timezone offset (360 = US Central)
        """
        self.logger = logging.getLogger("GoogleTrends")
        self.hl = hl
        self.tz = tz
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self._pytrends: Optional[Any] = None
        self._rate_limited_until = 0.0

    def _ensure_client(self):
        """Lazily initialize pytrends client"""
        if not PYTRENDS_AVAILABLE:
            raise ImportError("pytrends not installed. Run: pip install pytrends")
        if self._pytrends is None:
            self._pytrends = TrendReq(
                hl=self.hl,
                tz=self.tz,
                requests_args={
                    'headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AAC-GoogleTrends/1.0',
                    }
                },
            )

    def _cooldown_remaining(self) -> float:
        return max(0.0, self._rate_limited_until - time.monotonic())

    def _run_request(self, operation: Any, empty_result: Any):
        """Run a pytrends operation with 429-aware retry and cooldown handling."""
        remaining = self._cooldown_remaining()
        if remaining > 0:
            self.logger.warning(
                "Google Trends cooldown active for %.1fs after rate limiting",
                remaining,
            )
            return empty_result

        attempts = self.retries + 1
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                message = str(exc)
                is_rate_limited = '429' in message or 'Too Many Requests' in message
                if not is_rate_limited or attempt >= attempts:
                    break

                delay = self.backoff_seconds * attempt + random.uniform(0.0, 0.5)
                self.logger.warning(
                    "Google Trends rate limited on attempt %s/%s; retrying in %.1fs",
                    attempt,
                    attempts,
                    delay,
                )
                time.sleep(delay)

        if last_error is not None and ('429' in str(last_error) or 'Too Many Requests' in str(last_error)):
            cooldown = self.backoff_seconds * (self.retries + 2)
            self._rate_limited_until = time.monotonic() + cooldown
            self.logger.warning(
                "Google Trends entered cooldown for %.1fs after repeated rate limits",
                cooldown,
            )
        elif last_error is not None:
            self.logger.error("Google Trends request failed: %s", last_error)

        return empty_result

    def get_interest_over_time(
        self,
        keywords: List[str],
        timeframe: str = 'now 7-d',
        geo: str = '',
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get search interest over time for keywords.

        Args:
            keywords: Up to 5 search terms (Google limit)
            timeframe: Time range. Options:
                'now 1-H'   - past hour
                'now 4-H'   - past 4 hours
                'now 1-d'   - past day
                'now 7-d'   - past 7 days
                'today 1-m' - past month
                'today 3-m' - past 3 months
                'today 12-m'- past year
                'today 5-y' - past 5 years
            geo: Country code ('' for worldwide, 'US', 'CA', etc.)

        Returns:
            Dict with keyword -> list of {date, interest} entries (0-100 scale)
        """
        self._ensure_client()

        if len(keywords) > 5:
            keywords = keywords[:5]
            self.logger.warning("Google Trends limited to 5 keywords, truncated")

        try:
            def operation():
                self._pytrends.build_payload(
                    kw_list=keywords,
                    timeframe=timeframe,
                    geo=geo,
                )
                return self._pytrends.interest_over_time()

            df = self._run_request(operation, None)

            if df is None or df.empty:
                return {kw: [] for kw in keywords}

            result = {}
            for kw in keywords:
                if kw in df.columns:
                    result[kw] = [
                        {'date': str(idx), 'interest': int(row[kw])}
                        for idx, row in df.iterrows()
                    ]
                else:
                    result[kw] = []

            return result

        except Exception as e:
            self.logger.error(f"Failed to get interest over time: {e}")
            return {kw: [] for kw in keywords}

    def get_interest_by_region(
        self,
        keywords: List[str],
        timeframe: str = 'today 12-m',
        geo: str = '',
        resolution: str = 'COUNTRY',
    ) -> Dict[str, Dict[str, int]]:
        """
        Get search interest by region/country.

        Args:
            keywords: Up to 5 search terms
            resolution: 'COUNTRY', 'REGION', 'DMA', 'CITY'

        Returns:
            Dict with keyword -> {region: interest_score}
        """
        self._ensure_client()

        if len(keywords) > 5:
            keywords = keywords[:5]

        try:
            def operation():
                self._pytrends.build_payload(
                    kw_list=keywords,
                    timeframe=timeframe,
                    geo=geo,
                )
                return self._pytrends.interest_by_region(
                    resolution=resolution,
                    inc_low_vol=True,
                    inc_geo_code=False,
                )

            df = self._run_request(operation, None)

            if df is None:
                return {kw: {} for kw in keywords}

            result = {}
            for kw in keywords:
                if kw in df.columns:
                    result[kw] = {
                        region: int(score)
                        for region, score in df[kw].items()
                        if score > 0
                    }
                else:
                    result[kw] = {}

            return result

        except Exception as e:
            self.logger.error(f"Failed to get interest by region: {e}")
            return {kw: {} for kw in keywords}

    def get_related_queries(
        self,
        keywords: List[str],
        timeframe: str = 'now 7-d',
        geo: str = '',
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Get related search queries for keywords.

        Returns:
            Dict with keyword -> {'top': [...], 'rising': [...]}
            'rising' queries indicate breakout interest.
        """
        self._ensure_client()

        if len(keywords) > 5:
            keywords = keywords[:5]

        try:
            def operation():
                self._pytrends.build_payload(
                    kw_list=keywords,
                    timeframe=timeframe,
                    geo=geo,
                )
                return self._pytrends.related_queries()

            related = self._run_request(operation, None)

            if related is None:
                return {kw: {'top': [], 'rising': []} for kw in keywords}

            result = {}
            for kw in keywords:
                kw_data = related.get(kw, {})
                top_df = kw_data.get('top')
                rising_df = kw_data.get('rising')

                result[kw] = {
                    'top': top_df.to_dict('records') if top_df is not None and not top_df.empty else [],
                    'rising': rising_df.to_dict('records') if rising_df is not None and not rising_df.empty else [],
                }

            return result

        except Exception as e:
            self.logger.error(f"Failed to get related queries: {e}")
            return {kw: {'top': [], 'rising': []} for kw in keywords}

    def get_trending_searches(self, geo: str = 'united_states') -> List[str]:
        """
        Get today's trending searches.

        Args:
            geo: Country name (lowercase, underscored)

        Returns:
            List of trending search terms
        """
        self._ensure_client()
        pytrends = self._pytrends
        if pytrends is None:
            return []

        try:
            df = self._run_request(lambda: pytrends.trending_searches(pn=geo), None)
            if df is None:
                return []
            return df[0].tolist() if not df.empty else []
        except Exception as e:
            self.logger.error(f"Failed to get trending searches: {e}")
            return []

    def get_realtime_trending(self, geo: str = 'US', category: str = 'b') -> List[Dict[str, Any]]:
        """
        Get real-time trending searches.

        Args:
            geo: Country code
            category: Category ('b' = business, 'e' = entertainment, etc.)

        Returns:
            List of trending topic dicts
        """
        self._ensure_client()
        pytrends = self._pytrends
        if pytrends is None:
            return []

        try:
            df = self._run_request(
                lambda: pytrends.realtime_trending_searches(pn=geo, cat=category),
                None,
            )
            if df is not None and not df.empty:
                return df.head(20).to_dict('records')
            return []
        except Exception as e:
            self.logger.error(f"Failed to get realtime trending: {e}")
            return []

    def get_ticker_sentiment_score(
        self,
        ticker: str,
        company_name: str = '',
    ) -> Dict[str, Any]:
        """
        Get a sentiment score based on search interest for a ticker.

        Compares current interest to historical baseline.
        Score > 1.0 means above-average interest (potential catalyst).
        """
        self._ensure_client()

        keywords = [ticker]
        if company_name:
            keywords.append(company_name)

        try:
            def recent_operation():
                self._pytrends.build_payload(
                    kw_list=keywords,
                    timeframe='now 7-d',
                )
                return self._pytrends.interest_over_time()

            def baseline_operation():
                self._pytrends.build_payload(
                    kw_list=keywords,
                    timeframe='today 12-m',
                )
                return self._pytrends.interest_over_time()

            recent_df = self._run_request(recent_operation, None)
            baseline_df = self._run_request(baseline_operation, None)

            if recent_df is None or baseline_df is None:
                return {
                    'ticker': ticker,
                    'sentiment_score': 0,
                    'signal': 'rate_limited',
                    'error': 'Google Trends request throttled',
                }

            recent_avg = float(recent_df[ticker].mean()) if not recent_df.empty and ticker in recent_df.columns else 0
            baseline_avg = float(baseline_df[ticker].mean()) if not baseline_df.empty and ticker in baseline_df.columns else 1

            score = recent_avg / baseline_avg if baseline_avg > 0 else 0

            return {
                'ticker': ticker,
                'recent_interest': round(recent_avg, 2),
                'baseline_interest': round(baseline_avg, 2),
                'sentiment_score': round(score, 3),
                'signal': 'bullish' if score > 1.5 else 'neutral' if score > 0.7 else 'bearish',
                'timestamp': datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get sentiment score for {ticker}: {e}")
            return {
                'ticker': ticker,
                'sentiment_score': 0,
                'signal': 'unknown',
                'error': str(e),
            }
