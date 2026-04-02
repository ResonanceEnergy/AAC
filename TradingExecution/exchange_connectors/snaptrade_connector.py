#!/usr/bin/env python3
"""
SnapTrade Exchange Connector — Wealthsimple Access
===================================================
Uses SnapTrade API as intermediary to access Wealthsimple accounts.
Supports: list accounts, get balances, get positions (read-only).

SnapTrade Setup:
    1. Sign up at https://dashboard.snaptrade.com
    2. Get clientId + consumerKey
    3. Set SNAPTRADE_CLIENT_ID + SNAPTRADE_CONSUMER_KEY in .env
    4. Run: python _setup_snaptrade.py --register   (creates user)
    5. Run: python _setup_snaptrade.py --connect     (OAuth link to Wealthsimple)

Requirements:
    pip install snaptrade-python-sdk
"""

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_env, get_env_bool, load_env_file

load_env_file()

try:
    from snaptrade_client.client import SnapTrade
    SNAPTRADE_AVAILABLE = True
except ImportError:
    SNAPTRADE_AVAILABLE = False

from .base_connector import (
    AuthenticationError,
    Balance,
    BaseExchangeConnector,
    ConnectionError,
    ExchangeError,
    ExchangeOrder,
    OrderBook,
    OrderError,
    Ticker,
)


class SnapTradeConnector(BaseExchangeConnector):
    """
    SnapTrade connector for Wealthsimple access.

    Read-only connector — SnapTrade provides balance/position data
    but trading is done directly through Wealthsimple UI.
    """

    @property
    def name(self) -> str:
        return "snaptrade"

    def __init__(
        self,
        client_id: str = '',
        consumer_key: str = '',
        user_id: str = '',
        user_secret: str = '',
        rate_limit: int = 30,
    ):
        super().__init__(
            api_key=client_id,
            api_secret=consumer_key,
            testnet=False,
            rate_limit=rate_limit,
        )
        self.client_id = client_id or get_env('SNAPTRADE_CLIENT_ID', '')
        self.consumer_key = consumer_key or get_env('SNAPTRADE_CONSUMER_KEY', '')
        self.user_id = user_id or get_env('SNAPTRADE_USER_ID', '')
        self.user_secret = user_secret or get_env('SNAPTRADE_USER_SECRET', '')

        self._snap: Any = None
        self._accounts: List[Dict] = []

    async def connect(self) -> bool:
        """Initialize SnapTrade client and verify credentials."""
        if not SNAPTRADE_AVAILABLE:
            raise ConnectionError(
                "snaptrade-python-sdk not installed. Run: pip install snaptrade-python-sdk"
            )

        if not self.client_id or not self.consumer_key:
            raise AuthenticationError(
                "SNAPTRADE_CLIENT_ID / SNAPTRADE_CONSUMER_KEY not set. "
                "Sign up at https://dashboard.snaptrade.com"
            )

        if not self.user_id or not self.user_secret:
            raise AuthenticationError(
                "SNAPTRADE_USER_ID / SNAPTRADE_USER_SECRET not set. "
                "Run: python _setup_snaptrade.py --register"
            )

        try:
            self._snap = SnapTrade(
                consumer_key=self.consumer_key,
                client_id=self.client_id,
            )

            # Verify by listing accounts
            accounts_resp = self._snap.account_information.get_all_user_account_balances(
                user_id=self.user_id,
                user_secret=self.user_secret,
            )
            self._accounts = accounts_resp if isinstance(accounts_resp, list) else []
            self._connected = True

            self.logger.info(
                f"Connected to SnapTrade — {len(self._accounts)} account(s) found"
            )
            return True

        except Exception as e:
            self.logger.error(f"SnapTrade connection failed: {e}")
            raise ConnectionError(f"SnapTrade connection failed: {e}") from e

    async def disconnect(self) -> None:
        """SnapTrade is stateless REST — just clear state."""
        self._snap = None
        self._accounts = []
        self._connected = False

    async def get_ticker(self, symbol: str) -> Ticker:
        """Not supported — SnapTrade is for account data, not market data."""
        raise ExchangeError("SnapTrade does not provide market data. Use IBKR or Moomoo.")

    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Not supported."""
        raise ExchangeError("SnapTrade does not provide order book data.")

    async def get_balances(self) -> Dict[str, Balance]:
        """Get balances from all linked Wealthsimple accounts."""
        if not self._snap:
            raise ConnectionError("Not connected. Call connect() first.")

        try:
            accounts_resp = self._snap.account_information.get_all_user_account_balances(
                user_id=self.user_id,
                user_secret=self.user_secret,
            )

            result: Dict[str, Balance] = {}

            for acct in (accounts_resp if isinstance(accounts_resp, list) else []):
                acct_id = str(acct.get("id", "unknown"))
                acct_name = acct.get("name", acct_id)
                cash = acct.get("cash", {})
                amount = float(cash.get("amount", 0))
                currency = cash.get("currency", "CAD")

                result[f"{acct_name}_{currency}"] = Balance(
                    asset=f"{acct_name}_{currency}",
                    free=amount,
                    locked=0.0,
                )

            return result

        except Exception as e:
            self.logger.error(f"Failed to get SnapTrade balances: {e}")
            raise ExchangeError(f"Balance fetch failed: {e}") from e

    async def get_positions(self) -> List[Dict]:
        """Get positions from all linked Wealthsimple accounts."""
        if not self._snap:
            raise ConnectionError("Not connected. Call connect() first.")

        try:
            holdings_resp = self._snap.account_information.get_user_holdings(
                user_id=self.user_id,
                user_secret=self.user_secret,
            )

            positions = []
            if hasattr(holdings_resp, "positions"):
                for pos in holdings_resp.positions:
                    symbol_info = getattr(pos, "symbol", {})
                    positions.append({
                        "symbol": symbol_info.get("symbol", "?") if isinstance(symbol_info, dict) else str(symbol_info),
                        "units": float(getattr(pos, "units", 0)),
                        "market_value": float(getattr(pos, "market_value", 0)),
                        "average_purchase_price": float(getattr(pos, "average_purchase_price", 0)),
                    })

            return positions

        except Exception as e:
            self.logger.error(f"Failed to get SnapTrade positions: {e}")
            raise ExchangeError(f"Positions fetch failed: {e}") from e

    async def create_order(self, symbol: str, side: str, order_type: str,
                           quantity: float, price: Optional[float] = None,
                           client_order_id: Optional[str] = None) -> ExchangeOrder:
        """Trading not supported via SnapTrade in this connector."""
        raise ExchangeError(
            "Trading via SnapTrade not implemented. "
            "Use Wealthsimple UI directly for trades."
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        raise ExchangeError("Order cancellation not supported via SnapTrade.")

    async def get_order(self, order_id: str, symbol: str) -> ExchangeOrder:
        raise ExchangeError("Order lookup not supported via SnapTrade.")

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        raise ExchangeError("Open orders not supported via SnapTrade.")

    async def get_trade_fee(self, symbol: str) -> Dict[str, float]:
        """Wealthsimple charges $0 commission on most trades."""
        return {"maker": 0.0, "taker": 0.0}

    async def create_stop_loss_order(self, symbol: str, side: str, quantity: float,
                                     stop_price: float, limit_price: Optional[float] = None,
                                     client_order_id: Optional[str] = None) -> ExchangeOrder:
        raise ExchangeError("Stop-loss orders not supported via SnapTrade.")
