#!/usr/bin/env python3
"""
Metal X DEX API Client
======================
REST client for the Metal X decentralized exchange API.
Handles market data, order submission, account balances, and trade history.

API Documentation: https://api.dex.docs.metalx.com/reference
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class MetalXAPIError(Exception):
    """Metal X API error"""
    def __init__(self, message: str, status_code: int = 0, response: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response or {}


class MetalXClient:
    """
    Async REST client for the Metal X DEX API.

    Metal X is a fully on-chain centralized limit order book (CLOB) decentralized
    exchange with zero gas fees, built on XPR Network.

    Endpoints:
        - Markets: list all trading pairs, daily stats
        - Orders: submit, cancel, open orders, history, lifecycle, depth
        - Trades: recent trades, trade history
        - Account: balances
        - Chart: OHLCV data
        - Referrals: referral stats
        - Tax: export tax data
    """

    BASE_URL = "https://api.metalx.com"
    RATE_LIMIT_DELAY = 0.1  # 100ms between requests

    def __init__(
        self,
        api_url: str = "",
        account_name: str = "",
        rate_limit_delay: float = 0.1,
    ):
        self.api_url = (api_url or self.BASE_URL).rstrip("/")
        self.account_name = account_name
        self._rate_limit_delay = rate_limit_delay
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0

    async def connect(self) -> bool:
        """Create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return True

    async def disconnect(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _rate_limit_wait(self):
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.monotonic()

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
    ) -> Any:
        """Make an API request with rate limiting."""
        if self._session is None or self._session.closed:
            await self.connect()

        await self._rate_limit_wait()
        url = f"{self.api_url}{path}"

        try:
            async with self._session.request(
                method, url, params=params, json=json_data
            ) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    raise MetalXAPIError(
                        f"Metal X API error {resp.status}: {data}",
                        status_code=resp.status,
                        response=data if isinstance(data, dict) else {"raw": data},
                    )
                return data
        except aiohttp.ClientError as e:
            raise MetalXAPIError(f"Metal X connection error: {e}")

    # ─── Market Data ────────────────────────────────────────

    async def get_markets(self) -> List[Dict]:
        """Get all available trading markets."""
        return await self._request("GET", "/dex/v1/markets/all")

    async def get_daily_stats(self) -> List[Dict]:
        """Get 24h trading statistics for all markets."""
        return await self._request("GET", "/dex/v1/trades/daily")

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get OHLCV candlestick data.

        Args:
            symbol: Trading pair (e.g., 'XBTC_XMD')
            interval: Candle interval ('1m','5m','15m','1h','4h','1d')
            limit: Number of candles
        """
        return await self._request(
            "GET",
            "/dex/v1/chart",
            params={"symbol": symbol, "interval": interval, "limit": str(limit)},
        )

    # ─── Order Book ─────────────────────────────────────────

    async def get_orderbook_depth(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Dict:
        """
        Get order book depth for a market.

        Returns:
            Dict with 'bids' and 'asks' arrays of [price, quantity].
        """
        return await self._request(
            "GET",
            "/dex/v1/orders/depth",
            params={"symbol": symbol, "limit": str(limit)},
        )

    # ─── Orders ─────────────────────────────────────────────

    async def submit_order(self, order_data: Dict) -> Dict:
        """
        Submit a new order to Metal X.

        Args:
            order_data: Order parameters including symbol, side, type,
                       quantity, price, account, etc.
        """
        return await self._request("POST", "/dex/v1/orders/submit", json_data=order_data)

    async def serialize_order(self, order_data: Dict) -> Dict:
        """Serialize an order for signing before submission."""
        return await self._request("POST", "/dex/v1/orders/serialize", json_data=order_data)

    async def get_open_orders(
        self,
        account: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict]:
        """Get all open orders for an account."""
        params: Dict[str, str] = {}
        if account or self.account_name:
            params["account"] = account or self.account_name
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/dex/v1/orders/open", params=params)

    async def get_order_history(
        self,
        account: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Get order history for an account."""
        params: Dict[str, str] = {"limit": str(limit)}
        if account or self.account_name:
            params["account"] = account or self.account_name
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/dex/v1/orders/history", params=params)

    async def get_order_lifecycle(
        self,
        order_id: str,
    ) -> Dict:
        """Get full lifecycle of a specific order."""
        return await self._request(
            "GET", "/dex/v1/orders/lifecycle", params={"order_id": order_id}
        )

    # ─── Trades ─────────────────────────────────────────────

    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 50,
    ) -> List[Dict]:
        """Get recent trades for a market."""
        return await self._request(
            "GET",
            "/dex/v1/trades/recent",
            params={"symbol": symbol, "limit": str(limit)},
        )

    async def get_trade_history(
        self,
        account: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Get trade fill history for an account."""
        params: Dict[str, str] = {"limit": str(limit)}
        if account or self.account_name:
            params["account"] = account or self.account_name
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/dex/v1/trades/history", params=params)

    # ─── Account ────────────────────────────────────────────

    async def get_balances(
        self,
        account: Optional[str] = None,
    ) -> List[Dict]:
        """Get token balances for an account."""
        params: Dict[str, str] = {}
        if account or self.account_name:
            params["account"] = account or self.account_name
        return await self._request("GET", "/dex/v1/account/balances", params=params)

    # ─── Transfers ──────────────────────────────────────────

    async def get_transfers(
        self,
        account: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Get deposit/withdrawal transfer history."""
        params: Dict[str, str] = {"limit": str(limit)}
        if account or self.account_name:
            params["account"] = account or self.account_name
        return await self._request("GET", "/dex/v1/history/transfers", params=params)

    # ─── Referrals ──────────────────────────────────────────

    async def get_referral_totals(
        self,
        account: Optional[str] = None,
    ) -> Dict:
        """Get referral commission totals."""
        params: Dict[str, str] = {}
        if account or self.account_name:
            params["account"] = account or self.account_name
        return await self._request("GET", "/dex/v1/referrals/totals", params=params)

    async def get_referrals(
        self,
        account: Optional[str] = None,
    ) -> List[Dict]:
        """Get list of referred accounts."""
        params: Dict[str, str] = {}
        if account or self.account_name:
            params["account"] = account or self.account_name
        return await self._request("GET", "/dex/v1/referrals/list", params=params)

    # ─── Tax ────────────────────────────────────────────────

    async def get_tax_export(
        self,
        account: Optional[str] = None,
        year: Optional[int] = None,
    ) -> Dict:
        """Get tax export data for an account."""
        params: Dict[str, str] = {}
        if account or self.account_name:
            params["account"] = account or self.account_name
        if year:
            params["year"] = str(year)
        return await self._request("GET", "/dex/v1/tax/user", params=params)

    # ─── Status ─────────────────────────────────────────────

    async def get_sync_status(self) -> Dict:
        """Get latest chain sync status."""
        return await self._request("GET", "/dex/v1/status/sync")

    # ─── Leaderboard ────────────────────────────────────────

    async def get_leaderboard(self, limit: int = 50) -> List[Dict]:
        """Get trading leaderboard."""
        return await self._request(
            "GET", "/dex/v1/leaderboard/list", params={"limit": str(limit)}
        )
