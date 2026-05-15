#!/usr/bin/env python3
from __future__ import annotations

"""
Unusual Whales API Client
==========================
Client for Unusual Whales options flow, dark pool, and whale alert data.

API: https://api.unusualwhales.com/docs
SKILL: https://unusualwhales.com/skill.md

Requires:
    - UNUSUAL_WHALES_API_KEY in .env
"""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.api_integration_hub import APIClient, APIEndpoint, APIResponse
from shared.config_loader import get_config


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


def _parse_occ_symbol(symbol: str) -> Dict[str, Any]:
    """Parse an OCC option symbol like ``AAPL241115C00150000``.

    UW endpoints (e.g. ``/option-trades/flow-alerts``,
    ``/stock/{t}/option-contracts``) often only return the encoded chain string
    rather than separate ``strike``/``type``/``expiry`` fields. This helper
    decodes them so the rest of the code does not silently see $0 strikes.

    Returns dict with keys: ticker, expiry (YYYY-MM-DD), option_type, strike.
    Returns empty dict if the symbol is not parseable.
    """
    if not symbol or not isinstance(symbol, str):
        return {}
    s = symbol.strip().upper()
    # OCC: ROOT (1-6) + YYMMDD (6) + C/P (1) + STRIKE*1000 (8) → tail length 15
    if len(s) < 16 or len(s) > 21:
        return {}
    tail = s[-15:]
    root = s[:-15]
    if not root or not tail[:6].isdigit() or tail[6] not in ("C", "P") or not tail[7:].isdigit():
        return {}
    yy, mm, dd = tail[0:2], tail[2:4], tail[4:6]
    cp = tail[6]
    try:
        strike = int(tail[7:]) / 1000.0
    except ValueError:
        return {}
    return {
        "ticker": root,
        "expiry": f"20{yy}-{mm}-{dd}",
        "option_type": "call" if cp == "C" else "put",
        "strike": strike,
    }


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
    _USER_AGENT = "AAC/3.6.0 UnusualWhalesClient"
    # Required per UW API docs (skill.md)
    _CLIENT_API_ID = "100001"

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
        headers["UW-CLIENT-API-ID"] = self._CLIENT_API_ID
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
        url = f"{self.BASE_URL}/stock/{ticker}/option-contracts" if ticker else f"{self.BASE_URL}/option-trades/flow-alerts"
        params = {'limit': str(limit)}

        response = await self._make_request("GET", url, params=params)

        if not response.success:
            self.logger.error(f"Failed to get options flow: {response.error}")
            return []

        results = []
        data_list = response.data if isinstance(response.data, list) else response.data.get('data', [])

        for item in data_list[:limit]:
            premium = float(
                item.get('total_premium')
                or item.get('premium')
                or item.get('total_ask_side_prem')
                or 0
            )
            if premium < min_premium:
                continue

            # Field aliases vary by endpoint; OCC chain symbol is the source of
            # truth when present. Fall back to whatever top-level fields the
            # endpoint did include.
            occ = _parse_occ_symbol(
                item.get('option_chain')
                or item.get('option_symbol')
                or item.get('chain')
                or ''
            )

            ticker_field = (
                item.get('ticker')
                or item.get('underlying_symbol')
                or item.get('underlying_ticker')
                or occ.get('ticker', '')
            )
            strike_field = (
                item.get('strike')
                or item.get('strike_price')
                or occ.get('strike', 0)
            )
            expiry_field = (
                item.get('expiry')
                or item.get('expires_date')
                or item.get('expiration_date')
                or item.get('expiration')
                or occ.get('expiry', '')
            )
            type_field = (
                item.get('type')
                or item.get('put_call')
                or item.get('option_type')
                or item.get('side')
                or occ.get('option_type', '')
            )

            results.append(OptionsFlow(
                ticker=str(ticker_field),
                strike=float(strike_field or 0),
                expiry=str(expiry_field),
                option_type=str(type_field).lower(),
                sentiment=item.get('sentiment', 'neutral'),
                premium=premium,
                volume=int(
                    float(item.get('total_size') or item.get('volume') or item.get('size') or 0)
                ),
                open_interest=int(float(item.get('open_interest') or 0)),
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
            price = float(item.get('price', 0) or 0)
            size = int(item.get('size', item.get('volume', 0)) or 0)
            # API returns 'premium' as the notional value (price * size)
            notional = float(item.get('premium', 0) or 0) or (price * size)
            results.append(DarkPoolTrade(
                ticker=item.get('ticker', item.get('symbol', '')),
                price=price,
                size=size,
                notional=notional,
                exchange=item.get('market_center', item.get('exchange', '')),
                timestamp=datetime.now(),
            ))

        return results

    async def get_market_flow_summary(self) -> Dict[str, Any]:
        """Get market-wide options flow summary."""
        url = f"{self.BASE_URL}/option-trades/flow-alerts"

        response = await self._make_request("GET", url)

        if not response.success:
            self.logger.error(f"Failed to get market flow summary: {response.error}")
            return {}

        return response.data if isinstance(response.data, dict) else {}

    async def get_ticker_overview(self, ticker: str) -> Dict[str, Any]:
        """Get option contracts for a ticker."""
        url = f"{self.BASE_URL}/stock/{ticker}/option-contracts"

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
        """Get ETF options flow (option contracts for this ETF)."""
        url = f"{self.BASE_URL}/stock/{etf_ticker}/option-contracts"
        params = {'limit': '50'}

        response = await self._make_request("GET", url, params=params)

        if not response.success:
            self.logger.error(f"Failed to get ETF flow for {etf_ticker}: {response.error}")
            return {}

        return response.data if isinstance(response.data, dict) else {'data': response.data if isinstance(response.data, list) else []}

    async def get_hottest_chains(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get hottest options chains by volume."""
        url = f"{self.BASE_URL}/screener/option-contracts"
        params = {'limit': str(limit)}

        response = await self._make_request("GET", url, params=params)

        if not response.success:
            return []

        data = response.data
        if isinstance(data, dict):
            return data.get('data', [])[:limit]
        return data[:limit] if isinstance(data, list) else []

    async def get_sector_etfs(self) -> List[Dict[str, Any]]:
        """Get sector ETF list."""
        url = f"{self.BASE_URL}/etf/list"
        response = await self._make_request("GET", url)
        if not response.success:
            return []
        data = response.data
        if isinstance(data, dict):
            return data.get('data', [])
        return data if isinstance(data, list) else []

    async def get_insider_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent insider buy/sell transactions.

        NOTE: Finnhub also exposes insider transactions via
        ``integrations.finnhub_client.FinnhubClient.get_insider_transactions``.
        Prefer Finnhub for ticker-scoped queries; use this UW endpoint for the
        firehose of recent market-wide insider activity.
        """
        url = f"{self.BASE_URL}/insider/transactions"
        params = {'limit': str(limit)}
        response = await self._make_request("GET", url, params=params)
        if not response.success:
            return []
        data = response.data
        if isinstance(data, dict):
            return data.get('data', [])[:limit]
        return data[:limit] if isinstance(data, list) else []

    async def get_news_headlines(self, ticker: str = "SPY", limit: int = 50) -> List[Dict[str, Any]]:
        """Get financial news headlines (optionally filtered to a ticker).

        NOTE: Finnhub already supplies company + general news via
        ``FinnhubClient.get_company_news`` / ``get_news``. Prefer Finnhub for
        primary news; use this UW endpoint when you specifically want UW's
        flow-tagged news stream.
        """
        url = f"{self.BASE_URL}/news/headlines"
        params = {'limit': str(limit), 'ticker_symbol': ticker}
        response = await self._make_request("GET", url, params=params)
        if not response.success:
            return []
        data = response.data
        if isinstance(data, dict):
            return data.get('data', [])[:limit]
        return data[:limit] if isinstance(data, list) else []

    async def get_max_pain(self, ticker: str) -> Dict[str, Any]:
        """Get options volume data for a ticker (max pain can be derived from this)."""
        url = f"{self.BASE_URL}/stock/{ticker}/option-contracts"
        response = await self._make_request("GET", url)
        if not response.success:
            return {}
        return response.data if isinstance(response.data, dict) else {}

    # ------------------------------------------------------------------
    # Tier-1 endpoints unique to UW (no overlap with Finnhub/FRED/yfinance)
    # ------------------------------------------------------------------

    async def get_market_tide(self, interval_5m: bool = False) -> List[Dict[str, Any]]:
        """Market-wide net call/put premium time series (sentiment).

        Endpoint: ``/api/market/market-tide``. Use for regime detection;
        complements VIX/FRED macro by reacting in real time to options flow.
        """
        url = f"{self.BASE_URL}/market/market-tide"
        params = {'interval_5m': 'true' if interval_5m else 'false'}
        response = await self._make_request("GET", url, params=params)
        if not response.success:
            self.logger.error(f"Failed to get market tide: {response.error}")
            return []
        data = response.data
        if isinstance(data, dict):
            return data.get('data', []) or []
        return data if isinstance(data, list) else []

    async def get_spot_gex(self, ticker: str) -> List[Dict[str, Any]]:
        """Spot Gamma Exposure (GEX) by strike for a ticker.

        Endpoint: ``/api/stock/{ticker}/spot-exposures/strike``. Returns the
        per-strike dealer gamma profile — critical for put-strike selection
        (sell at positive-GEX walls, avoid short puts below negative-GEX zones).
        """
        url = f"{self.BASE_URL}/stock/{ticker}/spot-exposures/strike"
        response = await self._make_request("GET", url)
        if not response.success:
            self.logger.error(f"Failed to get spot GEX for {ticker}: {response.error}")
            return []
        data = response.data
        if isinstance(data, dict):
            return data.get('data', []) or []
        return data if isinstance(data, list) else []

    async def get_greeks(self, ticker: str) -> List[Dict[str, Any]]:
        """Per-strike Greeks (delta/gamma/vega/theta) for a ticker.

        Endpoint: ``/api/stock/{ticker}/greeks``. Replaces local Black-Scholes
        recomputation; use for roll-down decisions and theta-budget tracking.
        """
        url = f"{self.BASE_URL}/stock/{ticker}/greeks"
        response = await self._make_request("GET", url)
        if not response.success:
            self.logger.error(f"Failed to get greeks for {ticker}: {response.error}")
            return []
        data = response.data
        if isinstance(data, dict):
            return data.get('data', []) or []
        return data if isinstance(data, list) else []

    async def get_interpolated_iv(self, ticker: str) -> Dict[str, Any]:
        """Interpolated IV term structure + percentile rank for a ticker.

        Endpoint: ``/api/stock/{ticker}/interpolated-iv``. The single best
        entry gate for short-put strategies (require IV rank > 40 etc).
        """
        url = f"{self.BASE_URL}/stock/{ticker}/interpolated-iv"
        response = await self._make_request("GET", url)
        if not response.success:
            self.logger.error(f"Failed to get interpolated IV for {ticker}: {response.error}")
            return {}
        return response.data if isinstance(response.data, dict) else {}

    async def get_net_prem_ticks(self, ticker: str) -> List[Dict[str, Any]]:
        """Per-ticker net premium tick stream (intraday sentiment).

        Endpoint: ``/api/stock/{ticker}/net-prem-ticks``. Use to detect
        intraday sentiment flips on currently-held names.
        """
        url = f"{self.BASE_URL}/stock/{ticker}/net-prem-ticks"
        response = await self._make_request("GET", url)
        if not response.success:
            self.logger.error(f"Failed to get net premium ticks for {ticker}: {response.error}")
            return []
        data = response.data
        if isinstance(data, dict):
            return data.get('data', []) or []
        return data if isinstance(data, list) else []

    async def get_flow_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get latest options flow alerts (unusual activity) as raw dicts.

        For structured OptionsFlow dataclasses, use get_flow() instead.
        """
        url = f"{self.BASE_URL}/option-trades/flow-alerts"
        params = {'limit': str(limit)}
        response = await self._make_request("GET", url, params=params)
        if not response.success:
            return []
        data = response.data
        if isinstance(data, dict):
            return data.get('data', [])[:limit]
        return data[:limit] if isinstance(data, list) else []
