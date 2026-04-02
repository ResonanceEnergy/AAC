#!/usr/bin/env python3
"""
Tradier API Client
===================
Options-focused broker API with excellent options chain data, Greeks,
historical options data, and order execution.

API: https://documentation.tradier.com/

Requires:
    - TRADIER_API_KEY in .env (free sandbox at https://developer.tradier.com/)
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
class OptionQuote:
    """Options contract quote"""
    symbol: str
    underlying: str
    strike: float
    option_type: str  # 'call' or 'put'
    expiration: str
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    greeks: Dict[str, float]


@dataclass
class StockQuote:
    """Stock quote from Tradier"""
    symbol: str
    last: float
    change: float
    change_pct: float
    volume: int
    open: float
    high: float
    low: float
    close: float
    bid: float
    ask: float


class TradierClient(APIClient):
    """
    Tradier API client — options chains, Greeks, quotes, and market data.

    Free sandbox API key at https://developer.tradier.com/
    """

    SANDBOX_URL = 'https://sandbox.tradier.com/v1'
    PRODUCTION_URL = 'https://api.tradier.com/v1'

    def __init__(self):
        self._api_key = os.environ.get('TRADIER_API_KEY', '')
        is_sandbox = os.environ.get('TRADIER_SANDBOX', 'true').lower() in ('true', '1', 'yes')

        base_url = self.SANDBOX_URL if is_sandbox else self.PRODUCTION_URL

        endpoint = APIEndpoint(
            name='tradier',
            base_url=base_url,
            auth_type='bearer',
            auth_value=self._api_key,
            rate_limit=120,
            timeout=30,
        )
        super().__init__(endpoint)
        self.logger = logging.getLogger('TradierClient')

    async def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get real-time quote for a stock"""
        url = f"{self.endpoint.base_url}/markets/quotes"
        response = await self._make_request('GET', url, params={
            'symbols': symbol,
            'greeks': 'false',
        }, headers={'Accept': 'application/json'})

        if not response.success or not response.data:
            return None

        quotes = response.data.get('quotes', {})
        q = quotes.get('quote', {})
        if not q:
            return None

        return StockQuote(
            symbol=q.get('symbol', symbol),
            last=float(q.get('last', 0)),
            change=float(q.get('change', 0)),
            change_pct=float(q.get('change_percentage', 0)),
            volume=int(q.get('volume', 0)),
            open=float(q.get('open', 0)),
            high=float(q.get('high', 0)),
            low=float(q.get('low', 0)),
            close=float(q.get('close') or q.get('prevclose', 0)),
            bid=float(q.get('bid', 0)),
            ask=float(q.get('ask', 0)),
        )

    async def get_option_chain(
        self,
        symbol: str,
        expiration: str = '',
        greeks: bool = True,
    ) -> List[OptionQuote]:
        """
        Get options chain for a stock.

        Args:
            symbol: Underlying stock ticker
            expiration: Expiration date (YYYY-MM-DD). If empty, uses nearest.
            greeks: Include Greeks (delta, gamma, theta, vega, rho)
        """
        url = f"{self.endpoint.base_url}/markets/options/chains"
        params = {
            'symbol': symbol,
            'greeks': str(greeks).lower(),
        }
        if expiration:
            params['expiration'] = expiration

        response = await self._make_request('GET', url, params=params,
                                            headers={'Accept': 'application/json'})

        if not response.success or not response.data:
            return []

        options_data = response.data.get('options', {})
        option_list = options_data.get('option', [])
        if isinstance(option_list, dict):
            option_list = [option_list]

        results = []
        for opt in option_list:
            greeks_data = opt.get('greeks', {}) or {}
            results.append(OptionQuote(
                symbol=opt.get('symbol', ''),
                underlying=opt.get('underlying', symbol),
                strike=float(opt.get('strike', 0)),
                option_type=opt.get('option_type', ''),
                expiration=opt.get('expiration_date', ''),
                bid=float(opt.get('bid', 0)),
                ask=float(opt.get('ask', 0)),
                last=float(opt.get('last', 0)),
                volume=int(opt.get('volume', 0)),
                open_interest=int(opt.get('open_interest', 0)),
                greeks={
                    'delta': float(greeks_data.get('delta', 0)),
                    'gamma': float(greeks_data.get('gamma', 0)),
                    'theta': float(greeks_data.get('theta', 0)),
                    'vega': float(greeks_data.get('vega', 0)),
                    'rho': float(greeks_data.get('rho', 0)),
                    'iv': float(greeks_data.get('mid_iv', 0)),
                },
            ))
        return results

    async def get_option_expirations(self, symbol: str) -> List[str]:
        """Get available expiration dates for a symbol"""
        url = f"{self.endpoint.base_url}/markets/options/expirations"
        response = await self._make_request('GET', url, params={
            'symbol': symbol,
        }, headers={'Accept': 'application/json'})

        if not response.success or not response.data:
            return []

        expirations = response.data.get('expirations', {})
        dates = expirations.get('date', [])
        if isinstance(dates, str):
            dates = [dates]
        return dates

    async def get_option_strikes(self, symbol: str, expiration: str) -> List[float]:
        """Get available strikes for a symbol and expiration"""
        url = f"{self.endpoint.base_url}/markets/options/strikes"
        response = await self._make_request('GET', url, params={
            'symbol': symbol,
            'expiration': expiration,
        }, headers={'Accept': 'application/json'})

        if not response.success or not response.data:
            return []

        strikes_data = response.data.get('strikes', {})
        strikes = strikes_data.get('strike', [])
        if isinstance(strikes, (int, float)):
            strikes = [strikes]
        return [float(s) for s in strikes]

    async def get_historical(
        self,
        symbol: str,
        interval: str = 'daily',
        start: str = '',
        end: str = '',
    ) -> List[Dict]:
        """Get historical price data"""
        url = f"{self.endpoint.base_url}/markets/history"
        params = {'symbol': symbol, 'interval': interval}
        if start:
            params['start'] = start
        if end:
            params['end'] = end

        response = await self._make_request('GET', url, params=params,
                                            headers={'Accept': 'application/json'})
        if not response.success or not response.data:
            return []

        history = response.data.get('history', {})
        days = history.get('day', [])
        if isinstance(days, dict):
            days = [days]
        return days

    async def get_market_calendar(self, month: int = 0, year: int = 0) -> Optional[Dict]:
        """Get market calendar (trading days, holidays)"""
        url = f"{self.endpoint.base_url}/markets/calendar"
        params = {}
        if month:
            params['month'] = month
        if year:
            params['year'] = year

        response = await self._make_request('GET', url, params=params,
                                            headers={'Accept': 'application/json'})
        return response.data if response.success else None
