#!/usr/bin/env python3
"""
Polygon.io API Client
======================
Real-time and historical US market data: stocks, options, forex, crypto.
Aggregates, snapshots, trades, quotes, websockets.

API: https://polygon.io/docs

Requires:
    - POLYGON_API_KEY in .env
"""

import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config
from integrations.api_integration_hub import APIClient, APIEndpoint, APIResponse


@dataclass
class AggBar:
    """OHLCV aggregate bar"""
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    timestamp: datetime
    transactions: int = 0


@dataclass
class TickerSnapshot:
    """Real-time ticker snapshot"""
    ticker: str
    day_open: float
    day_high: float
    day_low: float
    day_close: float
    day_volume: float
    prev_close: float
    change: float
    change_pct: float
    updated: datetime


class PolygonClient(APIClient):
    """
    Polygon.io market data client.

    Provides real-time snapshots, historical aggregates, ticker details,
    market status, and grouped daily bars.
    """

    def __init__(self):
        config = get_config()
        self._api_key = config.__dict__.get('polygon_key', '') or config.__dict__.get('alphavantage_key', '')
        # Try polygon-specific env var
        import os
        self._api_key = os.environ.get('POLYGON_API_KEY', self._api_key)

        endpoint = APIEndpoint(
            name='polygon',
            base_url='https://api.polygon.io',
            auth_type='api_key',
            auth_header='Authorization',
            auth_value=f'Bearer {self._api_key}',
            rate_limit=5 if not self._api_key else 100,
            timeout=30,
        )
        super().__init__(endpoint)
        self.logger = logging.getLogger('PolygonClient')

    async def get_aggregates(
        self,
        ticker: str,
        multiplier: int = 1,
        timespan: str = 'day',
        from_date: str = '',
        to_date: str = '',
        limit: int = 120,
    ) -> List[AggBar]:
        """
        Get aggregate bars (OHLCV).

        Args:
            ticker: Stock ticker (e.g. 'AAPL')
            multiplier: Size of the timespan multiplier
            timespan: day, minute, hour, week, month, quarter, year
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Max results
        """
        if not from_date:
            from_date = (datetime.now().date().replace(day=1)).isoformat()
        if not to_date:
            to_date = date.today().isoformat()

        url = (
            f"{self.endpoint.base_url}/v2/aggs/ticker/{ticker}/range/"
            f"{multiplier}/{timespan}/{from_date}/{to_date}"
        )
        response = await self._make_request('GET', url, params={
            'adjusted': 'true',
            'sort': 'asc',
            'limit': limit,
        })

        if not response.success or not response.data:
            return []

        results = response.data.get('results', [])
        return [
            AggBar(
                ticker=ticker,
                open=float(bar.get('o', 0)),
                high=float(bar.get('h', 0)),
                low=float(bar.get('l', 0)),
                close=float(bar.get('c', 0)),
                volume=float(bar.get('v', 0)),
                vwap=float(bar.get('vw', 0)),
                timestamp=datetime.fromtimestamp(bar['t'] / 1000),
                transactions=int(bar.get('n', 0)),
            )
            for bar in results if 't' in bar
        ]

    async def get_snapshot(self, ticker: str) -> Optional[TickerSnapshot]:
        """Get real-time snapshot for a single ticker"""
        url = f"{self.endpoint.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        response = await self._make_request('GET', url)

        if not response.success or not response.data:
            return None

        t = response.data.get('ticker', {})
        day = t.get('day', {})
        prev = t.get('prevDay', {})

        return TickerSnapshot(
            ticker=t.get('ticker', ticker),
            day_open=float(day.get('o', 0)),
            day_high=float(day.get('h', 0)),
            day_low=float(day.get('l', 0)),
            day_close=float(day.get('c', 0)),
            day_volume=float(day.get('v', 0)),
            prev_close=float(prev.get('c', 0)),
            change=float(t.get('todaysChange', 0)),
            change_pct=float(t.get('todaysChangePerc', 0)),
            updated=datetime.fromtimestamp(t.get('updated', 0) / 1e9) if t.get('updated') else datetime.now(),
        )

    async def get_all_snapshots(self, tickers: Optional[List[str]] = None) -> List[TickerSnapshot]:
        """Get snapshots for all or specific tickers"""
        url = f"{self.endpoint.base_url}/v2/snapshot/locale/us/markets/stocks/tickers"
        params = {}
        if tickers:
            params['tickers'] = ','.join(tickers)

        response = await self._make_request('GET', url, params=params)
        if not response.success or not response.data:
            return []

        snapshots = []
        for t in response.data.get('tickers', []):
            day = t.get('day', {})
            prev = t.get('prevDay', {})
            snapshots.append(TickerSnapshot(
                ticker=t.get('ticker', ''),
                day_open=float(day.get('o', 0)),
                day_high=float(day.get('h', 0)),
                day_low=float(day.get('l', 0)),
                day_close=float(day.get('c', 0)),
                day_volume=float(day.get('v', 0)),
                prev_close=float(prev.get('c', 0)),
                change=float(t.get('todaysChange', 0)),
                change_pct=float(t.get('todaysChangePerc', 0)),
                updated=datetime.now(),
            ))
        return snapshots

    async def get_ticker_details(self, ticker: str) -> Optional[Dict]:
        """Get company details for a ticker"""
        url = f"{self.endpoint.base_url}/v3/reference/tickers/{ticker}"
        response = await self._make_request('GET', url)
        if response.success and response.data:
            return response.data.get('results', {})
        return None

    async def get_market_status(self) -> Optional[Dict]:
        """Get current market status (open/closed/early hours)"""
        url = f"{self.endpoint.base_url}/v1/marketstatus/now"
        response = await self._make_request('GET', url)
        return response.data if response.success else None

    async def get_grouped_daily(self, date_str: str = '') -> List[AggBar]:
        """Get grouped daily bars for the entire market on a date"""
        if not date_str:
            date_str = date.today().isoformat()

        url = f"{self.endpoint.base_url}/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
        response = await self._make_request('GET', url, params={'adjusted': 'true'})

        if not response.success or not response.data:
            return []

        return [
            AggBar(
                ticker=bar.get('T', ''),
                open=float(bar.get('o', 0)),
                high=float(bar.get('h', 0)),
                low=float(bar.get('l', 0)),
                close=float(bar.get('c', 0)),
                volume=float(bar.get('v', 0)),
                vwap=float(bar.get('vw', 0)),
                timestamp=datetime.fromtimestamp(bar['t'] / 1000) if 't' in bar else datetime.now(),
                transactions=int(bar.get('n', 0)),
            )
            for bar in response.data.get('results', [])
        ]

    async def get_previous_close(self, ticker: str) -> Optional[AggBar]:
        """Get previous day's close for a ticker"""
        url = f"{self.endpoint.base_url}/v2/aggs/ticker/{ticker}/prev"
        response = await self._make_request('GET', url, params={'adjusted': 'true'})
        if response.success and response.data:
            results = response.data.get('results', [])
            if results:
                bar = results[0]
                return AggBar(
                    ticker=ticker,
                    open=float(bar.get('o', 0)),
                    high=float(bar.get('h', 0)),
                    low=float(bar.get('l', 0)),
                    close=float(bar.get('c', 0)),
                    volume=float(bar.get('v', 0)),
                    vwap=float(bar.get('vw', 0)),
                    timestamp=datetime.fromtimestamp(bar['t'] / 1000) if 't' in bar else datetime.now(),
                )
        return None
