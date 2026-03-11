#!/usr/bin/env python3
"""
Santiment API Client
=====================
On-chain and social metrics for crypto. Network activity, social volume,
development activity, exchange flow, whale behavior, and sentiment.

API: https://api.santiment.net/graphiql

Requires:
    - SANTIMENT_API_KEY in .env (free tier available at https://app.santiment.net/)
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.api_integration_hub import APIClient, APIEndpoint, APIResponse


@dataclass
class SantimentMetric:
    """A single metric data point"""
    datetime: str
    value: float


class SantimentClient(APIClient):
    """
    Santiment API client — on-chain analytics and social metrics.

    Uses GraphQL API to query crypto metrics including:
    - Social volume and sentiment
    - Development activity
    - On-chain transaction volume
    - Active addresses
    - Exchange inflow/outflow
    - MVRV ratio
    - NVT ratio
    """

    def __init__(self):
        self._api_key = os.environ.get('SANTIMENT_API_KEY', '')

        endpoint = APIEndpoint(
            name='santiment',
            base_url='https://api.santiment.net/graphql',
            auth_type='bearer',
            auth_value=self._api_key,
            rate_limit=60,
            timeout=30,
        )
        super().__init__(endpoint)
        self.logger = logging.getLogger('SantimentClient')

    async def _query(self, graphql_query: str) -> Optional[Dict]:
        """Execute a GraphQL query against Santiment API"""
        response = await self._make_request(
            'POST',
            self.endpoint.base_url,
            json={'query': graphql_query},
            headers={'Content-Type': 'application/json'},
        )
        if response.success and response.data:
            return response.data.get('data', {})
        return None

    async def get_metric(
        self,
        metric: str,
        slug: str = 'bitcoin',
        from_date: str = '',
        to_date: str = '',
        interval: str = '1d',
    ) -> List[SantimentMetric]:
        """
        Get a time-series metric for a crypto asset.

        Args:
            metric: Metric name (e.g. 'social_volume_total', 'dev_activity', 'daily_active_addresses')
            slug: Asset slug (e.g. 'bitcoin', 'ethereum')
            from_date: ISO date string
            to_date: ISO date string
            interval: '1h', '1d', '1w'
        """
        if not from_date:
            from_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%dT00:00:00Z')
        if not to_date:
            to_date = datetime.utcnow().strftime('%Y-%m-%dT00:00:00Z')

        query = f'''{{
            getMetric(metric: "{metric}") {{
                timeseriesData(
                    slug: "{slug}"
                    from: "{from_date}"
                    to: "{to_date}"
                    interval: "{interval}"
                ) {{
                    datetime
                    value
                }}
            }}
        }}'''

        data = await self._query(query)
        if not data:
            return []

        timeseries = data.get('getMetric', {}).get('timeseriesData', [])
        return [
            SantimentMetric(datetime=point['datetime'], value=float(point['value']))
            for point in timeseries
            if 'datetime' in point and 'value' in point
        ]

    async def get_social_volume(self, slug: str = 'bitcoin', days: int = 30) -> List[SantimentMetric]:
        """Get social volume (mentions across social media)"""
        return await self.get_metric('social_volume_total', slug, interval='1d')

    async def get_dev_activity(self, slug: str = 'bitcoin', days: int = 30) -> List[SantimentMetric]:
        """Get development activity (GitHub commits, events)"""
        return await self.get_metric('dev_activity', slug, interval='1d')

    async def get_active_addresses(self, slug: str = 'bitcoin', days: int = 30) -> List[SantimentMetric]:
        """Get daily active addresses (on-chain usage)"""
        return await self.get_metric('daily_active_addresses', slug, interval='1d')

    async def get_exchange_inflow(self, slug: str = 'bitcoin', days: int = 30) -> List[SantimentMetric]:
        """Get exchange inflow (selling pressure indicator)"""
        return await self.get_metric('exchange_inflow', slug, interval='1d')

    async def get_exchange_outflow(self, slug: str = 'bitcoin', days: int = 30) -> List[SantimentMetric]:
        """Get exchange outflow (accumulation indicator)"""
        return await self.get_metric('exchange_outflow', slug, interval='1d')

    async def get_mvrv_ratio(self, slug: str = 'bitcoin', days: int = 90) -> List[SantimentMetric]:
        """
        Get MVRV (Market Value to Realized Value) ratio.
        
        MVRV > 3.5 → market top signal (overvalued).
        MVRV < 1.0 → market bottom signal (undervalued).
        """
        return await self.get_metric('mvrv_usd', slug, interval='1d')

    async def get_nvt_ratio(self, slug: str = 'bitcoin', days: int = 90) -> List[SantimentMetric]:
        """
        Get NVT (Network Value to Transactions) ratio.
        
        High NVT → overvalued relative to on-chain activity.
        Low NVT → undervalued relative to on-chain activity.
        """
        return await self.get_metric('nvt', slug, interval='1d')

    async def get_sentiment_summary(self, slug: str = 'bitcoin') -> Dict[str, Any]:
        """
        Get a combined sentiment summary for a crypto asset.

        Pulls social volume, active addresses, exchange flows, and MVRV.
        """
        social = await self.get_social_volume(slug, days=7)
        active = await self.get_active_addresses(slug, days=7)
        inflow = await self.get_exchange_inflow(slug, days=7)
        outflow = await self.get_exchange_outflow(slug, days=7)
        mvrv = await self.get_mvrv_ratio(slug, days=7)

        avg_social = sum(s.value for s in social) / max(len(social), 1)
        avg_inflow = sum(s.value for s in inflow) / max(len(inflow), 1)
        avg_outflow = sum(s.value for s in outflow) / max(len(outflow), 1)
        latest_mvrv = mvrv[-1].value if mvrv else 0

        net_flow = avg_outflow - avg_inflow

        return {
            'slug': slug,
            'avg_social_volume_7d': avg_social,
            'avg_active_addresses_7d': sum(s.value for s in active) / max(len(active), 1),
            'avg_exchange_inflow_7d': avg_inflow,
            'avg_exchange_outflow_7d': avg_outflow,
            'net_exchange_flow_7d': net_flow,
            'mvrv_ratio': latest_mvrv,
            'flow_signal': 'BULLISH' if net_flow > 0 else 'BEARISH',
            'mvrv_signal': 'OVERVALUED' if latest_mvrv > 3.5 else 'UNDERVALUED' if latest_mvrv < 1.0 else 'NEUTRAL',
        }
