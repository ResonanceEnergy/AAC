#!/usr/bin/env python3
"""
Finnhub API Client
===================
Real-time stock data, earnings, IPO calendar, insider transactions,
SEC filings, economic calendar, and financial news.

API: https://finnhub.io/docs/api

Requires:
    - FINNHUB_API_KEY in .env (free tier: 60 req/min at https://finnhub.io/)
"""

import logging
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.api_integration_hub import APIClient, APIEndpoint, APIResponse


@dataclass
class EarningsEntry:
    """Upcoming or past earnings report"""
    symbol: str
    date: str
    eps_estimate: float
    eps_actual: float
    revenue_estimate: float
    revenue_actual: float
    quarter: int
    year: int


@dataclass
class InsiderTransaction:
    """Insider trade filing"""
    symbol: str
    name: str
    share: int
    change: int
    filing_date: str
    transaction_date: str
    transaction_type: str  # 'P' (purchase) or 'S' (sale)
    price: float


@dataclass
class NewsArticle:
    """Financial news article"""
    headline: str
    source: str
    url: str
    summary: str
    category: str
    datetime: datetime
    related: str


class FinnhubClient(APIClient):
    """
    Finnhub API client — comprehensive financial data.

    Free tier: 60 API calls/min, real-time US stock data,
    earnings calendar, insider transactions, news, and more.
    """

    def __init__(self):
        self._api_key = os.environ.get('FINNHUB_API_KEY', '')

        endpoint = APIEndpoint(
            name='finnhub',
            base_url='https://finnhub.io/api/v1',
            auth_type='api_key',
            auth_header='X-Finnhub-Token',
            auth_value=self._api_key,
            rate_limit=60,
            timeout=30,
        )
        super().__init__(endpoint)
        self.logger = logging.getLogger('FinnhubClient')

    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote for a stock"""
        url = f"{self.endpoint.base_url}/quote"
        response = await self._make_request('GET', url, params={'symbol': symbol})

        if not response.success or not response.data:
            return None

        q = response.data
        return {
            'symbol': symbol,
            'current': float(q.get('c', 0)),
            'change': float(q.get('d', 0)),
            'change_pct': float(q.get('dp', 0)),
            'high': float(q.get('h', 0)),
            'low': float(q.get('l', 0)),
            'open': float(q.get('o', 0)),
            'prev_close': float(q.get('pc', 0)),
            'timestamp': datetime.fromtimestamp(q.get('t', 0)),
        }

    async def get_earnings_calendar(
        self,
        from_date: str = '',
        to_date: str = '',
        symbol: str = '',
    ) -> List[EarningsEntry]:
        """
        Get upcoming and recent earnings.

        Args:
            from_date: Start date (YYYY-MM-DD), default: today
            to_date: End date (YYYY-MM-DD), default: +7 days
            symbol: Filter by ticker (optional)
        """
        if not from_date:
            from_date = date.today().isoformat()
        if not to_date:
            to_date = (date.today() + timedelta(days=7)).isoformat()

        url = f"{self.endpoint.base_url}/calendar/earnings"
        params = {'from': from_date, 'to': to_date}
        if symbol:
            params['symbol'] = symbol

        response = await self._make_request('GET', url, params=params)
        if not response.success or not response.data:
            return []

        return [
            EarningsEntry(
                symbol=e.get('symbol', ''),
                date=e.get('date', ''),
                eps_estimate=float(e.get('epsEstimate', 0) or 0),
                eps_actual=float(e.get('epsActual', 0) or 0),
                revenue_estimate=float(e.get('revenueEstimate', 0) or 0),
                revenue_actual=float(e.get('revenueActual', 0) or 0),
                quarter=int(e.get('quarter', 0) or 0),
                year=int(e.get('year', 0) or 0),
            )
            for e in response.data.get('earningsCalendar', [])
        ]

    async def get_ipo_calendar(
        self,
        from_date: str = '',
        to_date: str = '',
    ) -> List[Dict]:
        """Get upcoming IPOs"""
        if not from_date:
            from_date = date.today().isoformat()
        if not to_date:
            to_date = (date.today() + timedelta(days=30)).isoformat()

        url = f"{self.endpoint.base_url}/calendar/ipo"
        response = await self._make_request('GET', url, params={
            'from': from_date, 'to': to_date
        })
        if response.success and response.data:
            return response.data.get('ipoCalendar', [])
        return []

    async def get_insider_transactions(
        self,
        symbol: str,
        from_date: str = '',
        to_date: str = '',
    ) -> List[InsiderTransaction]:
        """
        Get insider transactions for a stock.

        Insider buying → bullish signal.
        Insider selling → bearish or routine.
        """
        url = f"{self.endpoint.base_url}/stock/insider-transactions"
        params = {'symbol': symbol}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date

        response = await self._make_request('GET', url, params=params)
        if not response.success or not response.data:
            return []

        return [
            InsiderTransaction(
                symbol=symbol,
                name=tx.get('name', ''),
                share=int(tx.get('share', 0) or 0),
                change=int(tx.get('change', 0) or 0),
                filing_date=tx.get('filingDate', ''),
                transaction_date=tx.get('transactionDate', ''),
                transaction_type=tx.get('transactionCode', ''),
                price=float(tx.get('transactionPrice', 0) or 0),
            )
            for tx in response.data.get('data', [])
        ]

    async def get_news(
        self,
        category: str = 'general',
        min_id: int = 0,
    ) -> List[NewsArticle]:
        """
        Get market news.

        Args:
            category: 'general', 'forex', 'crypto', 'merger'
            min_id: Filter news after this ID
        """
        url = f"{self.endpoint.base_url}/news"
        params = {'category': category}
        if min_id:
            params['minId'] = min_id

        response = await self._make_request('GET', url, params=params)
        if not response.success or not response.data:
            return []

        return [
            NewsArticle(
                headline=article.get('headline', ''),
                source=article.get('source', ''),
                url=article.get('url', ''),
                summary=article.get('summary', ''),
                category=article.get('category', ''),
                datetime=datetime.fromtimestamp(article.get('datetime', 0)),
                related=article.get('related', ''),
            )
            for article in response.data
            if isinstance(article, dict)
        ]

    async def get_company_news(
        self,
        symbol: str,
        from_date: str = '',
        to_date: str = '',
    ) -> List[NewsArticle]:
        """Get news for a specific company"""
        if not from_date:
            from_date = (date.today() - timedelta(days=7)).isoformat()
        if not to_date:
            to_date = date.today().isoformat()

        url = f"{self.endpoint.base_url}/company-news"
        response = await self._make_request('GET', url, params={
            'symbol': symbol, 'from': from_date, 'to': to_date
        })

        if not response.success or not response.data:
            return []

        return [
            NewsArticle(
                headline=a.get('headline', ''),
                source=a.get('source', ''),
                url=a.get('url', ''),
                summary=a.get('summary', ''),
                category=a.get('category', ''),
                datetime=datetime.fromtimestamp(a.get('datetime', 0)),
                related=a.get('related', symbol),
            )
            for a in response.data
            if isinstance(a, dict)
        ]

    async def get_recommendation_trends(self, symbol: str) -> List[Dict]:
        """Get analyst recommendation trends (buy/hold/sell counts)"""
        url = f"{self.endpoint.base_url}/stock/recommendation"
        response = await self._make_request('GET', url, params={'symbol': symbol})
        return response.data if response.success and isinstance(response.data, list) else []

    async def get_price_target(self, symbol: str) -> Optional[Dict]:
        """Get analyst price target consensus"""
        url = f"{self.endpoint.base_url}/stock/price-target"
        response = await self._make_request('GET', url, params={'symbol': symbol})
        return response.data if response.success else None

    async def get_sec_filings(self, symbol: str) -> List[Dict]:
        """Get recent SEC filings (10-K, 10-Q, 8-K, etc.)"""
        url = f"{self.endpoint.base_url}/stock/filings"
        response = await self._make_request('GET', url, params={'symbol': symbol})
        return response.data if response.success and isinstance(response.data, list) else []

    async def get_economic_calendar(
        self,
        from_date: str = '',
        to_date: str = '',
    ) -> List[Dict]:
        """Get upcoming economic events (FOMC, CPI, jobs report, etc.)"""
        if not from_date:
            from_date = date.today().isoformat()
        if not to_date:
            to_date = (date.today() + timedelta(days=7)).isoformat()

        url = f"{self.endpoint.base_url}/calendar/economic"
        response = await self._make_request('GET', url, params={
            'from': from_date, 'to': to_date
        })
        if response.success and response.data:
            return response.data.get('economicCalendar', [])
        return []
