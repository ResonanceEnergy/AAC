#!/usr/bin/env python3
"""
Unusual Whales API Client
==========================
Client for Unusual Whales options flow, dark pool, and whale alert data.

API: https://api.unusualwhales.com/docs

Requires:
    - UNUSUAL_WHALES_API_KEY in .env
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config
from integrations.api_integration_hub import APIClient, APIEndpoint, APIResponse


@dataclass
class OptionsFlow:
    """Parsed options flow entry"""
    ticker: str
    strike: float
    expiry: str
    option_type: str  # 'call' or 'put'
    sentiment: str
    premium: float
    volume: int
    open_interest: int
    timestamp: datetime


@dataclass
class DarkPoolTrade:
    """Parsed dark pool trade"""
    ticker: str
    price: float
    size: int
    notional: float
    exchange: str
    timestamp: datetime


class UnusualWhalesClient(APIClient):
    """
    Client for Unusual Whales API.

    Provides:
        - Options flow (unusual activity)
        - Dark pool trades
        - Market-wide flow summaries
        - Ticker-specific flow
        - Congress/Senate trading data
    """

    BASE_URL = "https://api.unusualwhales.com/api"

    # Cloudflare blocks default Python User-Agent
    _USER_AGENT = "AAC/2.7.0 UnusualWhalesClient"

    def __init__(self, api_key: str = ''):
        config = get_config()
        self.api_key = api_key or config.__dict__.get('unusual_whales_key', '')

        endpoint = APIEndpoint(
            name="unusual_whales",
            base_url=self.BASE_URL,
            auth_type='bearer',
            auth_value=self.api_key,
            rate_limit=120,  # 120 req/min
            timeout=30,
            retries=3,
            enabled=bool(self.api_key),
        )
        super().__init__(endpoint)
        self.logger = logging.getLogger("UnusualWhales")

    def _get_auth_headers(self) -> Dict[str, str]:
        headers = super()._get_auth_headers()
        headers["User-Agent"] = self._USER_AGENT
        return headers

    async def get_flow(
        self,
        ticker: Optional[str] = None,
        min_premium: float = 100000,
        limit: int = 50,
    ) -> List[OptionsFlow]:
        """
        Get unusual options flow.

        Args:
            ticker: Filter by ticker symbol (None for all)
            min_premium: Minimum premium filter
            limit: Max results
        """
        url = f"{self.BASE_URL}/stock/{ticker}/flow-recent" if ticker else f"{self.BASE_URL}/option-trades/flow-alerts"
        params = {'limit': str(limit)}

        response = await self._make_request("GET", url, params=params)

        if not response.success:
            self.logger.error(f"Failed to get options flow: {response.error}")
            return []

        results = []
        data_list = response.data if isinstance(response.data, list) else response.data.get('data', [])

        for item in data_list[:limit]:
            premium = float(item.get('premium', 0) or 0)
            if premium < min_premium:
                continue

            results.append(OptionsFlow(
                ticker=item.get('ticker', item.get('underlying_symbol', '')),
                strike=float(item.get('strike_price', 0) or 0),
                expiry=item.get('expires_date', item.get('expiry', '')),
                option_type=item.get('put_call', item.get('option_type', '')).lower(),
                sentiment=item.get('sentiment', 'neutral'),
                premium=premium,
                volume=int(item.get('volume', 0) or 0),
                open_interest=int(item.get('open_interest', 0) or 0),
                timestamp=datetime.now(),
            ))

        return results

    async def get_dark_pool(
        self,
        ticker: Optional[str] = None,
        limit: int = 50,
    ) -> List[DarkPoolTrade]:
        """
        Get dark pool trades.

        Args:
            ticker: Filter by ticker (None for all)
            limit: Max results
        """
        url = f"{self.BASE_URL}/darkpool/{ticker}" if ticker else f"{self.BASE_URL}/darkpool/recent"
        params = {'limit': str(limit)}

        response = await self._make_request("GET", url, params=params)

        if not response.success:
            self.logger.error(f"Failed to get dark pool data: {response.error}")
            return []

        results = []
        data_list = response.data if isinstance(response.data, list) else response.data.get('data', [])

        for item in data_list[:limit]:
            results.append(DarkPoolTrade(
                ticker=item.get('ticker', item.get('symbol', '')),
                price=float(item.get('price', 0) or 0),
                size=int(item.get('size', item.get('volume', 0)) or 0),
                notional=float(item.get('notional_value', 0) or 0),
                exchange=item.get('exchange', item.get('market_center', '')),
                timestamp=datetime.now(),
            ))

        return results

    async def get_market_flow_summary(self) -> Dict[str, Any]:
        """Get market-wide options flow summary (market tide data)."""
        url = f"{self.BASE_URL}/market/market-tide"

        response = await self._make_request("GET", url)

        if not response.success:
            self.logger.error(f"Failed to get market flow summary: {response.error}")
            return {}

        return response.data if isinstance(response.data, dict) else {}

    async def get_ticker_overview(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive overview for a ticker (info, sector, marketcap)."""
        url = f"{self.BASE_URL}/stock/{ticker}/info"

        response = await self._make_request("GET", url)

        if not response.success:
            self.logger.error(f"Failed to get ticker overview for {ticker}: {response.error}")
            return {}

        return response.data if isinstance(response.data, dict) else {}

    async def get_congress_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent Congressional / Senate trading disclosures."""
        url = f"{self.BASE_URL}/congress/recent-trades"
        params = {'limit': str(limit)}

        response = await self._make_request("GET", url, params=params)

        if not response.success:
            self.logger.error(f"Failed to get congress trades: {response.error}")
            return []

        data = response.data
        if isinstance(data, dict):
            return data.get('data', [])[:limit]
        return data[:limit] if isinstance(data, list) else []

    async def get_etf_flow(self, etf_ticker: str = 'SPY') -> Dict[str, Any]:
        """Get ETF options flow"""
        return await self.get_ticker_overview(etf_ticker)

    async def get_hottest_chains(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get hottest options chains by volume."""
        url = f"{self.BASE_URL}/market/spike"
        params = {'limit': str(limit)}

        response = await self._make_request("GET", url, params=params)

        if not response.success:
            return []

        data = response.data
        if isinstance(data, dict):
            return data.get('data', [])[:limit]
        return data[:limit] if isinstance(data, list) else []

    async def get_sector_etfs(self) -> List[Dict[str, Any]]:
        """Get sector ETF performance data."""
        url = f"{self.BASE_URL}/market/sector-etfs"
        response = await self._make_request("GET", url)
        if not response.success:
            return []
        data = response.data
        if isinstance(data, dict):
            return data.get('data', [])
        return data if isinstance(data, list) else []

    async def get_insider_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent insider buy/sell transactions."""
        url = f"{self.BASE_URL}/insider/transactions"
        params = {'limit': str(limit)}
        response = await self._make_request("GET", url, params=params)
        if not response.success:
            return []
        data = response.data
        if isinstance(data, dict):
            return data.get('data', [])[:limit]
        return data[:limit] if isinstance(data, list) else []

    async def get_news_headlines(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get latest financial news headlines."""
        url = f"{self.BASE_URL}/news/headlines"
        params = {'limit': str(limit)}
        response = await self._make_request("GET", url, params=params)
        if not response.success:
            return []
        data = response.data
        if isinstance(data, dict):
            return data.get('data', [])[:limit]
        return data[:limit] if isinstance(data, list) else []

    async def get_max_pain(self, ticker: str) -> Dict[str, Any]:
        """Get max pain data for a ticker."""
        url = f"{self.BASE_URL}/stock/{ticker}/max-pain"
        response = await self._make_request("GET", url)
        if not response.success:
            return {}
        return response.data if isinstance(response.data, dict) else {}

    async def get_flow_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get latest options flow alerts (unusual activity)."""
        url = f"{self.BASE_URL}/option-trades/flow-alerts"
        response = await self._make_request("GET", url)
        if not response.success:
            return []
        data = response.data
        if isinstance(data, dict):
            return data.get('data', [])[:limit]
        return data[:limit] if isinstance(data, list) else []
