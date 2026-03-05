#!/usr/bin/env python3
"""
FRED (Federal Reserve Economic Data) Client
=============================================
Free macroeconomic data from the Federal Reserve Bank of St. Louis.
GDP, CPI, unemployment, interest rates, yield curves, money supply.

API: https://fred.stlouisfed.org/docs/api/fred/
No API key required for basic use (optional for higher limits).

Requires:
    - FRED_API_KEY in .env (optional — free at https://fred.stlouisfed.org/docs/api/api_key.html)
"""

import logging
import os
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.api_integration_hub import APIClient, APIEndpoint, APIResponse


# Key FRED series for trading signals
MACRO_SERIES = {
    'GDP': 'Gross Domestic Product (quarterly)',
    'CPIAUCSL': 'Consumer Price Index (monthly)',
    'UNRATE': 'Unemployment Rate (monthly)',
    'FEDFUNDS': 'Federal Funds Rate (monthly)',
    'DFF': 'Federal Funds Effective Rate (daily)',
    'T10Y2Y': '10Y-2Y Treasury Spread (daily, recession indicator)',
    'T10YIE': '10Y Breakeven Inflation (daily)',
    'VIXCLS': 'CBOE Volatility Index (daily)',
    'DGS10': '10-Year Treasury Rate (daily)',
    'DGS2': '2-Year Treasury Rate (daily)',
    'M2SL': 'M2 Money Supply (monthly)',
    'DEXUSEU': 'USD/EUR Exchange Rate (daily)',
    'DCOILWTICO': 'WTI Crude Oil Price (daily)',
    'GOLDAMGBD228NLBM': 'Gold Price (daily)',
    'BAMLH0A0HYM2': 'High Yield Bond Spread (daily)',
    'UMCSENT': 'U of Michigan Consumer Sentiment (monthly)',
    'RSAFS': 'Retail Sales (monthly)',
    'INDPRO': 'Industrial Production Index (monthly)',
    'PAYEMS': 'Total Nonfarm Payrolls (monthly)',
    'ICSA': 'Initial Jobless Claims (weekly)',
}


@dataclass
class FredObservation:
    """Single data point from a FRED series"""
    series_id: str
    date: str
    value: float
    realtime_start: str = ''
    realtime_end: str = ''


@dataclass
class FredSeries:
    """FRED series metadata"""
    series_id: str
    title: str
    frequency: str
    units: str
    last_updated: str
    observation_start: str
    observation_end: str


class FredClient(APIClient):
    """
    FRED API client for macroeconomic data.

    Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
    """

    def __init__(self):
        self._api_key = os.environ.get('FRED_API_KEY', '')

        endpoint = APIEndpoint(
            name='fred',
            base_url='https://api.stlouisfed.org/fred',
            auth_type='none',
            rate_limit=120,
            timeout=30,
        )
        super().__init__(endpoint)
        self.logger = logging.getLogger('FredClient')

    def _params(self, **kwargs) -> Dict:
        """Build request params with API key and file_type"""
        params = {'file_type': 'json'}
        if self._api_key:
            params['api_key'] = self._api_key
        params.update(kwargs)
        return params

    async def get_series_observations(
        self,
        series_id: str,
        observation_start: str = '',
        observation_end: str = '',
        limit: int = 100,
        sort_order: str = 'desc',
    ) -> List[FredObservation]:
        """
        Get observations (data points) for a FRED series.

        Args:
            series_id: FRED series ID (e.g. 'GDP', 'CPIAUCSL', 'FEDFUNDS')
            observation_start: Start date (YYYY-MM-DD)
            observation_end: End date (YYYY-MM-DD)
            limit: Max observations
            sort_order: 'asc' or 'desc'
        """
        params = self._params(
            series_id=series_id,
            limit=limit,
            sort_order=sort_order,
        )
        if observation_start:
            params['observation_start'] = observation_start
        if observation_end:
            params['observation_end'] = observation_end

        url = f"{self.endpoint.base_url}/series/observations"
        response = await self._make_request('GET', url, params=params)

        if not response.success or not response.data:
            return []

        observations = []
        for obs in response.data.get('observations', []):
            try:
                value = float(obs['value']) if obs['value'] != '.' else 0.0
            except (ValueError, KeyError):
                continue
            observations.append(FredObservation(
                series_id=series_id,
                date=obs.get('date', ''),
                value=value,
                realtime_start=obs.get('realtime_start', ''),
                realtime_end=obs.get('realtime_end', ''),
            ))
        return observations

    async def get_series_info(self, series_id: str) -> Optional[FredSeries]:
        """Get metadata about a FRED series"""
        url = f"{self.endpoint.base_url}/series"
        response = await self._make_request('GET', url, params=self._params(series_id=series_id))

        if not response.success or not response.data:
            return None

        series_list = response.data.get('seriess', [])
        if not series_list:
            return None

        s = series_list[0]
        return FredSeries(
            series_id=s.get('id', series_id),
            title=s.get('title', ''),
            frequency=s.get('frequency', ''),
            units=s.get('units', ''),
            last_updated=s.get('last_updated', ''),
            observation_start=s.get('observation_start', ''),
            observation_end=s.get('observation_end', ''),
        )

    async def get_latest_value(self, series_id: str) -> Optional[FredObservation]:
        """Get the most recent observation for a series"""
        obs = await self.get_series_observations(series_id, limit=1, sort_order='desc')
        return obs[0] if obs else None

    async def get_macro_dashboard(self) -> Dict[str, Optional[FredObservation]]:
        """
        Get latest values for all key macro indicators.

        Returns dict mapping series ID -> latest observation.
        """
        dashboard = {}
        for series_id in MACRO_SERIES:
            dashboard[series_id] = await self.get_latest_value(series_id)
        return dashboard

    async def get_yield_curve(self) -> Dict[str, float]:
        """Get current Treasury yield curve"""
        maturities = {
            'DGS1MO': '1M', 'DGS3MO': '3M', 'DGS6MO': '6M',
            'DGS1': '1Y', 'DGS2': '2Y', 'DGS3': '3Y',
            'DGS5': '5Y', 'DGS7': '7Y', 'DGS10': '10Y',
            'DGS20': '20Y', 'DGS30': '30Y',
        }
        curve = {}
        for series_id, label in maturities.items():
            obs = await self.get_latest_value(series_id)
            if obs:
                curve[label] = obs.value
        return curve

    async def is_yield_curve_inverted(self) -> Dict[str, Any]:
        """
        Check if the yield curve is inverted (recession signal).

        Returns inversion status and spread between 10Y and 2Y.
        """
        obs_10y = await self.get_latest_value('DGS10')
        obs_2y = await self.get_latest_value('DGS2')

        if not obs_10y or not obs_2y:
            return {'inverted': None, 'spread': None, 'error': 'Could not fetch data'}

        spread = obs_10y.value - obs_2y.value
        return {
            'inverted': spread < 0,
            'spread': spread,
            '10y_rate': obs_10y.value,
            '2y_rate': obs_2y.value,
            'signal': 'RECESSION_WARNING' if spread < 0 else 'NORMAL',
        }

    async def search_series(self, search_text: str, limit: int = 10) -> List[FredSeries]:
        """Search for FRED series by keyword"""
        url = f"{self.endpoint.base_url}/series/search"
        response = await self._make_request('GET', url, params=self._params(
            search_text=search_text,
            limit=limit,
        ))

        if not response.success or not response.data:
            return []

        return [
            FredSeries(
                series_id=s.get('id', ''),
                title=s.get('title', ''),
                frequency=s.get('frequency', ''),
                units=s.get('units', ''),
                last_updated=s.get('last_updated', ''),
                observation_start=s.get('observation_start', ''),
                observation_end=s.get('observation_end', ''),
            )
            for s in response.data.get('seriess', [])
        ]
