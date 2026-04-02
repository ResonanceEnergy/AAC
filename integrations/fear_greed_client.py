#!/usr/bin/env python3
"""
Fear & Greed Index Client
===========================
Crypto and stock market sentiment indices.

Sources:
    - Alternative.me Crypto Fear & Greed Index (free, no key)
    - CNN Fear & Greed Index (scraped, no key)

No API key required.
"""

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.api_integration_hub import APIClient, APIEndpoint, APIResponse


@dataclass
class FearGreedReading:
    """Fear & Greed Index data point"""
    value: int           # 0 (Extreme Fear) — 100 (Extreme Greed)
    classification: str  # 'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'
    timestamp: datetime
    market: str          # 'crypto' or 'stock'


class FearGreedClient(APIClient):
    """
    Fear & Greed Index client.

    Crypto: Alternative.me API (free, no key).
    0-25 = Extreme Fear, 25-45 = Fear, 45-55 = Neutral,
    55-75 = Greed, 75-100 = Extreme Greed.

    Trading signals:
    - Extreme Fear → contrarian buy signal
    - Extreme Greed → contrarian sell signal
    """

    def __init__(self):
        endpoint = APIEndpoint(
            name='fear_greed',
            base_url='https://api.alternative.me/fng/',
            auth_type='none',
            rate_limit=30,
            timeout=15,
        )
        super().__init__(endpoint)
        self.logger = logging.getLogger('FearGreedClient')

    async def get_current(self) -> Optional[FearGreedReading]:
        """Get current Fear & Greed Index value"""
        response = await self._make_request('GET', self.endpoint.base_url)

        if not response.success or not response.data:
            return None

        data_list = response.data.get('data', [])
        if not data_list:
            return None

        entry = data_list[0]
        return FearGreedReading(
            value=int(entry.get('value', 50)),
            classification=entry.get('value_classification', 'Neutral'),
            timestamp=datetime.fromtimestamp(int(entry.get('timestamp', 0))),
            market='crypto',
        )

    async def get_historical(self, limit: int = 30) -> List[FearGreedReading]:
        """
        Get historical Fear & Greed Index values.

        Args:
            limit: Number of days of history (max ~365)
        """
        url = f"{self.endpoint.base_url}?limit={limit}&format=json"
        response = await self._make_request('GET', url)

        if not response.success or not response.data:
            return []

        return [
            FearGreedReading(
                value=int(entry.get('value', 50)),
                classification=entry.get('value_classification', 'Neutral'),
                timestamp=datetime.fromtimestamp(int(entry.get('timestamp', 0))),
                market='crypto',
            )
            for entry in response.data.get('data', [])
        ]

    async def get_signal(self) -> Dict[str, Any]:
        """
        Get Fear & Greed trading signal with context.

        Uses current + 7-day average to determine if sentiment is
        at an extreme worth acting on.
        """
        current = await self.get_current()
        history = await self.get_historical(limit=7)

        if not current:
            return {'signal': 'NO_DATA', 'value': None}

        avg_7d = sum(r.value for r in history) / max(len(history), 1) if history else current.value

        # Generate contrarian signal
        if current.value <= 20:
            signal = 'STRONG_BUY'
        elif current.value <= 35:
            signal = 'BUY'
        elif current.value >= 80:
            signal = 'STRONG_SELL'
        elif current.value >= 65:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'

        # Detect rapid sentiment shifts
        trend = 'STABLE'
        if history and len(history) >= 3:
            recent_avg = sum(r.value for r in history[:3]) / 3
            if recent_avg - avg_7d > 15:
                trend = 'RAPIDLY_GREEDY'
            elif avg_7d - recent_avg > 15:
                trend = 'RAPIDLY_FEARFUL'

        return {
            'current_value': current.value,
            'classification': current.classification,
            'avg_7d': round(avg_7d, 1),
            'signal': signal,
            'trend': trend,
            'timestamp': current.timestamp.isoformat(),
        }
