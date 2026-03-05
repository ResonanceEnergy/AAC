#!/usr/bin/env python3
"""
NDAX Exchange Connector
========================
Implementation of the exchange connector for NDAX (National Digital Asset Exchange).
Canadian crypto exchange. Uses CCXT which has native NDAX support.

Requires:
    - pip install ccxt
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config

from .base_connector import (
    BaseExchangeConnector,
    Ticker,
    OrderBook,
    Balance,
    ExchangeOrder,
    ExchangeError,
    ConnectionError,
    AuthenticationError,
    InsufficientFundsError,
    OrderError,
)

try:
    import ccxt
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


class NDAXConnector(BaseExchangeConnector):
    """
    NDAX exchange connector using ccxt library.

    NDAX is a Canadian-regulated cryptocurrency exchange supporting
    BTC, ETH, XRP, LTC, EOS, DOGE, ADA, USDT, and more CAD pairs.
    """

    @property
    def name(self) -> str:
        return "ndax"

    def __init__(
        self,
        api_key: str = '',
        api_secret: str = '',
        user_id: str = '',
        account_id: str = '',
        testnet: bool = False,
        rate_limit: int = 600,
    ):
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            rate_limit=rate_limit,
        )

        config = get_config()
        if not api_key:
            self.api_key = config.__dict__.get('ndax_api_key', '')
            self.api_secret = config.__dict__.get('ndax_api_secret', '')
        self.user_id = user_id or config.__dict__.get('ndax_user_id', '')
        self.account_id = account_id or config.__dict__.get('ndax_account_id', '')

    async def connect(self) -> bool:
        """Connect to NDAX via CCXT"""
        import time
        start_time = time.time()

        if not CCXT_AVAILABLE:
            self.logger.error("ccxt library not installed. Run: pip install ccxt")
            await self._audit_auth("failure", "ccxt library not installed")
            return False

        try:
            options = {
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
            }

            if self.user_id:
                options['uid'] = self.user_id

            self._client = ccxt_async.ndax(options)

            # Test connection by loading markets
            await self._client.load_markets()

            self._connected = True
            self.logger.info("Connected to NDAX")

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("load_markets", "GET", "success", duration_ms)
            await self._audit_auth("success")

            return True

        except ccxt.AuthenticationError as e:
            self.logger.error(f"NDAX authentication failed: {e}")
            await self._audit_auth("failure", str(e))
            raise AuthenticationError(str(e))
        except Exception as e:
            self.logger.error(f"Failed to connect to NDAX: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("connect", "GET", "failure", duration_ms, error_message=str(e))
            raise ConnectionError(str(e))

    async def disconnect(self) -> None:
        """Disconnect from NDAX"""
        if self._client:
            await self._client.close()
            self._client = None
            self._connected = False
            self.logger.info("Disconnected from NDAX")

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker data"""
        await self._rate_limit_wait()

        try:
            ticker = await self._client.fetch_ticker(symbol)
            return Ticker(
                symbol=symbol,
                bid=ticker.get('bid', 0) or 0,
                ask=ticker.get('ask', 0) or 0,
                last=ticker.get('last', 0) or 0,
                volume_24h=ticker.get('quoteVolume', 0) or 0,
                timestamp=datetime.fromtimestamp(
                    ticker['timestamp'] / 1000
                ) if ticker.get('timestamp') else datetime.now(),
            )
        except Exception as e:
            self.logger.error(f"Failed to fetch NDAX ticker for {symbol}: {e}")
            raise ExchangeError(str(e))

    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get order book"""
        await self._rate_limit_wait()

        try:
            book = await self._client.fetch_order_book(symbol, limit)
            return OrderBook(
                symbol=symbol,
                bids=[(b[0], b[1]) for b in book.get('bids', [])],
                asks=[(a[0], a[1]) for a in book.get('asks', [])],
                timestamp=datetime.fromtimestamp(
                    book['timestamp'] / 1000
                ) if book.get('timestamp') else datetime.now(),
            )
        except Exception as e:
            self.logger.error(f"Failed to fetch NDAX orderbook for {symbol}: {e}")
            raise ExchangeError(str(e))

    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances"""
        if not self._check_credentials():
            self.logger.warning("No NDAX API credentials - returning empty balances")
            return {}

        await self._rate_limit_wait()

        try:
            balance = await self._client.fetch_balance()
            result = {}

            for asset, data in balance.get('total', {}).items():
                if data > 0:
                    result[asset] = Balance(
                        asset=asset,
                        free=balance.get('free', {}).get(asset, 0),
                        locked=balance.get('used', {}).get(asset, 0),
                    )

            return result
        except ccxt.AuthenticationError as e:
            raise AuthenticationError(str(e))
        except Exception as e:
            self.logger.error(f"Failed to fetch NDAX balances: {e}")
            raise ExchangeError(str(e))

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """Create an order on NDAX"""
        if not self._check_credentials():
            raise AuthenticationError("NDAX API credentials required for trading")

        await self._rate_limit_wait()

        import time
        start_time = time.time()

        try:
            params = {}
            if client_order_id:
                params['clientOrderId'] = client_order_id

            order = await self._client.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=quantity,
                price=price,
                params=params,
            )

            result = self._parse_ccxt_order(order)

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "POST", "success", duration_ms)
            await self._audit_order(symbol, side, order_type, quantity, price, result.order_id, "created")

            return result

        except ccxt.InsufficientFunds as e:
            await self._audit_order(symbol, side, order_type, quantity, price, None, "failed", str(e))
            raise InsufficientFundsError(str(e))
        except ccxt.InvalidOrder as e:
            await self._audit_order(symbol, side, order_type, quantity, price, None, "failed", str(e))
            raise OrderError(str(e))
        except Exception as e:
            self.logger.error(f"Failed to create NDAX order: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "POST", "failure", duration_ms, error_message=str(e))
            raise ExchangeError(str(e))

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        if not self._check_credentials():
            raise AuthenticationError("NDAX API credentials required")

        await self._rate_limit_wait()

        try:
            await self._client.cancel_order(order_id, symbol)
            return True
        except ccxt.OrderNotFound:
            self.logger.warning(f"NDAX order {order_id} not found")
            return False
        except Exception as e:
            self.logger.error(f"Failed to cancel NDAX order {order_id}: {e}")
            raise ExchangeError(str(e))

    async def get_order(self, order_id: str, symbol: str) -> ExchangeOrder:
        """Get order details"""
        if not self._check_credentials():
            raise AuthenticationError("NDAX API credentials required")

        await self._rate_limit_wait()

        try:
            order = await self._client.fetch_order(order_id, symbol)
            return self._parse_ccxt_order(order)
        except ccxt.OrderNotFound:
            raise OrderError(f"Order {order_id} not found on NDAX")
        except Exception as e:
            self.logger.error(f"Failed to fetch NDAX order {order_id}: {e}")
            raise ExchangeError(str(e))

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get all open orders"""
        if not self._check_credentials():
            return []

        await self._rate_limit_wait()

        try:
            orders = await self._client.fetch_open_orders(symbol)
            return [self._parse_ccxt_order(o) for o in orders]
        except Exception as e:
            self.logger.error(f"Failed to fetch NDAX open orders: {e}")
            raise ExchangeError(str(e))

    async def get_trade_fee(self, symbol: str) -> Dict[str, float]:
        """Get trading fees"""
        return {'maker': 0.002, 'taker': 0.002}

    @staticmethod
    def _parse_ccxt_order(order: dict) -> ExchangeOrder:
        """Convert CCXT order dict to ExchangeOrder"""
        return ExchangeOrder(
            order_id=str(order.get('id', '')),
            client_order_id=order.get('clientOrderId'),
            symbol=order.get('symbol', ''),
            side=order.get('side', ''),
            order_type=order.get('type', ''),
            quantity=float(order.get('amount', 0)),
            price=float(order.get('price', 0)) if order.get('price') else None,
            status=order.get('status', 'unknown'),
            filled_quantity=float(order.get('filled', 0)),
            average_price=float(order.get('average', 0) or 0),
            fee=float(order.get('fee', {}).get('cost', 0) or 0) if order.get('fee') else 0,
        )
