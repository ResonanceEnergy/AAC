#!/usr/bin/env python3
"""
NDAX Exchange Connector
=======================
Implementation of the exchange connector for NDAX (National Digital Asset Exchange).

NDAX is a Canadian cryptocurrency exchange supporting CAD/crypto pairs.
Uses ccxt library (NDAX is a built-in ccxt exchange).

Requirements:
    pip install ccxt

Configuration via .env:
    NDAX_API_KEY=your_key
    NDAX_API_SECRET=your_secret
    NDAX_USER_ID=your_user_id
    NDAX_TESTNET=true
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_env, get_env_bool
from shared.utils import with_circuit_breaker, CircuitOpenError

try:
    from shared.utils import with_circuit_breaker, CircuitOpenError
except ImportError:
    def with_circuit_breaker(*args, **kwargs):
        """With circuit breaker."""
        def decorator(func):
            """Decorator."""
            return func
        return decorator
    class CircuitOpenError(Exception):
        """CircuitOpenError class."""
        pass

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

# Try to import ccxt
try:
    import ccxt
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


class NDAXConnector(BaseExchangeConnector):
    """
    NDAX (National Digital Asset Exchange) connector using ccxt.

    Canadian crypto exchange supporting BTC/CAD, ETH/CAD, and other
    CAD-denominated cryptocurrency pairs. Also supports USD pairs.
    """

    @property
    def name(self) -> str:
        """Name."""
        return "ndax"

    def __init__(
        self,
        api_key: str = '',
        api_secret: str = '',
        testnet: bool = True,
        rate_limit: int = 1000,
    ):
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            rate_limit=rate_limit,
        )

        # NDAX-specific: user ID is required for some endpoints
        self._user_id = get_env('NDAX_USER_ID', '')

        # Load from config if not provided
        if not api_key:
            config = get_config()
            if hasattr(config, 'ndax') and config.ndax.is_configured():
                self.api_key = config.ndax.api_key
                self.api_secret = config.ndax.api_secret
                self.testnet = config.ndax.testnet

    async def connect(self) -> bool:
        """Connect to NDAX."""
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
                'options': {
                    'defaultType': 'spot',
                },
            }

            if self._user_id:
                options['uid'] = self._user_id

            if self.testnet:
                options['sandbox'] = True
                self.logger.info("Connecting to NDAX TESTNET")
            else:
                self.logger.info("Connecting to NDAX MAINNET")

            self._client = ccxt_async.ndax(options)

            if self.testnet:
                self._client.set_sandbox_mode(True)

            # Test connection by loading markets
            await self._client.load_markets()

            self._connected = True
            duration_ms = (time.time() - start_time) * 1000

            market_count = len(self._client.markets) if self._client.markets else 0
            self.logger.info(
                f"Connected to NDAX ({'testnet' if self.testnet else 'mainnet'}) "
                f"- {market_count} markets loaded in {duration_ms:.0f}ms"
            )
            await self._audit_api_call("load_markets", "GET", "success", duration_ms)
            await self._audit_auth("success")
            return True

        except ccxt.AuthenticationError as e:
            self.logger.error(f"NDAX authentication failed: {e}")
            await self._audit_auth("failure", str(e))
            raise AuthenticationError(str(e))
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Failed to connect to NDAX: {e}")
            await self._audit_api_call("connect", "GET", "failure", duration_ms, error_message=str(e))
            raise ConnectionError(str(e))

    async def disconnect(self) -> None:
        """Disconnect from NDAX."""
        if self._client:
            await self._client.close()
            self._client = None
            self._connected = False
            self.logger.info("Disconnected from NDAX")

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker for symbol (e.g., BTC/CAD)."""
        await self._rate_limit_wait()
        return await self._get_ticker_with_breaker(symbol)

    @with_circuit_breaker("ndax_ticker", failure_threshold=5, timeout=30.0)
    async def _get_ticker_with_breaker(self, symbol: str) -> Ticker:
        try:
            ticker = await self._client.fetch_ticker(symbol)
            return Ticker(
                symbol=symbol,
                bid=float(ticker.get('bid', 0) or 0),
                ask=float(ticker.get('ask', 0) or 0),
                last=float(ticker.get('last', 0) or 0),
                volume_24h=float(ticker.get('quoteVolume', 0) or 0),
                timestamp=datetime.fromtimestamp(
                    ticker['timestamp'] / 1000
                ) if ticker.get('timestamp') else datetime.now(),
            )
        except CircuitOpenError:
            raise
        except Exception as e:
            self.logger.error(f"NDAX ticker fetch failed for {symbol}: {e}")
            raise ExchangeError(str(e))

    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get order book for symbol."""
        await self._rate_limit_wait()
        return await self._get_orderbook_with_breaker(symbol, limit)

    @with_circuit_breaker("ndax_orderbook", failure_threshold=5, timeout=30.0)
    async def _get_orderbook_with_breaker(self, symbol: str, limit: int = 20) -> OrderBook:
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
        except CircuitOpenError:
            raise
        except Exception as e:
            self.logger.error(f"NDAX orderbook fetch failed for {symbol}: {e}")
            raise ExchangeError(str(e))

    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances."""
        if not self._check_credentials():
            raise AuthenticationError("API credentials required for balance query")

        await self._rate_limit_wait()
        start_time = time.time()

        try:
            balance = await self._client.fetch_balance()
            duration_ms = (time.time() - start_time) * 1000

            balances = {}
            for asset, info in balance.items():
                if isinstance(info, dict) and (info.get('free', 0) or info.get('used', 0)):
                    free = float(info.get('free', 0) or 0)
                    locked = float(info.get('used', 0) or 0)
                    if free > 0 or locked > 0:
                        balances[asset] = Balance(
                            asset=asset,
                            free=free,
                            locked=locked,
                        )

            await self._audit_api_call("fetch_balance", "GET", "success", duration_ms)
            return balances

        except ccxt.AuthenticationError as e:
            raise AuthenticationError(str(e))
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("fetch_balance", "GET", "failure", duration_ms, error_message=str(e))
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
        """Place an order on NDAX."""
        if not self._check_credentials():
            raise AuthenticationError("API credentials required for trading")

        await self._rate_limit_wait()
        return await self._create_order_with_breaker(
            symbol, side, order_type, quantity, price, client_order_id
        )

    @with_circuit_breaker("ndax_order", failure_threshold=3, timeout=60.0)
    async def _create_order_with_breaker(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        import time
        start_time = time.time()

        try:
            params = {}
            if client_order_id:
                params['clientOrderId'] = client_order_id

            result = await self._client.create_order(
                symbol=symbol,
                type=order_type.lower(),
                side=side.lower(),
                amount=quantity,
                price=price,
                params=params,
            )

            duration_ms = (time.time() - start_time) * 1000
            order_id = str(result.get('id', ''))

            order = ExchangeOrder(
                order_id=order_id,
                client_order_id=result.get('clientOrderId', client_order_id),
                symbol=symbol,
                side=side.lower(),
                order_type=order_type.lower(),
                quantity=quantity,
                price=price,
                status=result.get('status', 'open'),
                filled_quantity=float(result.get('filled', 0) or 0),
                average_price=float(result.get('average', 0) or 0),
                raw=result,
            )

            await self._audit_api_call("create_order", "POST", "success", duration_ms)
            await self._audit_order(symbol, side, order_type, quantity, price, order_id, "submitted")

            self.logger.info(
                f"NDAX order placed: {side.upper()} {quantity} {symbol} "
                f"@ {'MKT' if order_type == 'market' else price} -> {order_id}"
            )
            return order

        except ccxt.InsufficientFunds as e:
            await self._audit_order(symbol, side, order_type, quantity, price, None, "failed", str(e))
            raise InsufficientFundsError(str(e))
        except ccxt.AuthenticationError as e:
            raise AuthenticationError(str(e))
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "POST", "failure", duration_ms, error_message=str(e))
            await self._audit_order(symbol, side, order_type, quantity, price, None, "failed", str(e))
            raise OrderError(str(e))

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        if not self._check_credentials():
            raise AuthenticationError("API credentials required")

        await self._rate_limit_wait()
        start_time = time.time()

        try:
            await self._client.cancel_order(order_id, symbol)
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("cancel_order", "DELETE", "success", duration_ms)
            self.logger.info(f"NDAX order {order_id} cancelled")
            return True
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("cancel_order", "DELETE", "failure", duration_ms, error_message=str(e))
            self.logger.error(f"Failed to cancel NDAX order {order_id}: {e}")
            return False

    async def get_order(self, order_id: str, symbol: str) -> ExchangeOrder:
        """Get order details."""
        if not self._check_credentials():
            raise AuthenticationError("API credentials required")

        await self._rate_limit_wait()

        try:
            result = await self._client.fetch_order(order_id, symbol)
            return ExchangeOrder(
                order_id=str(result.get('id', '')),
                client_order_id=result.get('clientOrderId'),
                symbol=symbol,
                side=result.get('side', ''),
                order_type=result.get('type', ''),
                quantity=float(result.get('amount', 0) or 0),
                price=float(result.get('price', 0) or 0),
                status=result.get('status', 'open'),
                filled_quantity=float(result.get('filled', 0) or 0),
                average_price=float(result.get('average', 0) or 0),
                raw=result,
            )
        except Exception as e:
            raise ExchangeError(f"Failed to get NDAX order {order_id}: {e}")

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get all open orders."""
        if not self._check_credentials():
            raise AuthenticationError("API credentials required")

        await self._rate_limit_wait()

        try:
            orders = await self._client.fetch_open_orders(symbol)
            return [
                ExchangeOrder(
                    order_id=str(o.get('id', '')),
                    client_order_id=o.get('clientOrderId'),
                    symbol=o.get('symbol', symbol or ''),
                    side=o.get('side', ''),
                    order_type=o.get('type', ''),
                    quantity=float(o.get('amount', 0) or 0),
                    price=float(o.get('price', 0) or 0),
                    status=o.get('status', 'open'),
                    filled_quantity=float(o.get('filled', 0) or 0),
                    average_price=float(o.get('average', 0) or 0),
                    raw=o,
                )
                for o in orders
            ]
        except Exception as e:
            raise ExchangeError(f"Failed to get NDAX open orders: {e}")

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """Create a stop-loss order on NDAX."""
        if not self._check_credentials():
            raise AuthenticationError("API credentials required")

        await self._rate_limit_wait()
        start_time = time.time()

        try:
            params = {'stopPrice': stop_price}
            if client_order_id:
                params['clientOrderId'] = client_order_id

            if limit_price is not None:
                # Stop-limit order
                result = await self._client.create_order(
                    symbol=symbol,
                    type='stopLimit',
                    side=side.lower(),
                    amount=quantity,
                    price=limit_price,
                    params=params,
                )
            else:
                # Stop-market order
                result = await self._client.create_order(
                    symbol=symbol,
                    type='stopMarket',
                    side=side.lower(),
                    amount=quantity,
                    price=stop_price,
                    params=params,
                )

            duration_ms = (time.time() - start_time) * 1000
            order_id = str(result.get('id', ''))

            order = ExchangeOrder(
                order_id=order_id,
                client_order_id=result.get('clientOrderId', client_order_id),
                symbol=symbol,
                side=side.lower(),
                order_type='stop_loss',
                quantity=quantity,
                price=limit_price or stop_price,
                status=result.get('status', 'open'),
                raw=result,
            )

            await self._audit_api_call("create_stop_loss", "POST", "success", duration_ms)
            await self._audit_order(symbol, side, "stop_loss", quantity, stop_price, order_id, "submitted")
            return order

        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(str(e))
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_stop_loss", "POST", "failure", duration_ms, error_message=str(e))
            raise OrderError(f"NDAX stop-loss failed: {e}") from e

    async def get_trade_fee(self, symbol: str) -> Dict[str, float]:
        """Get trade fees for a symbol."""
        try:
            if hasattr(self._client, 'fetch_trading_fee'):
                fee = await self._client.fetch_trading_fee(symbol)
                return {
                    'maker': float(fee.get('maker', 0.002)),
                    'taker': float(fee.get('taker', 0.002)),
                }
        except Exception as e:
            logger.exception("Unexpected error: %s", e)
        # NDAX default fees: 0.20% maker/taker
        return {'maker': 0.002, 'taker': 0.002}

    async def get_available_cad_pairs(self) -> List[str]:
        """Get all available CAD trading pairs on NDAX."""
        if not self._client or not self._client.markets:
            raise ConnectionError("Not connected to NDAX")
        return [
            symbol for symbol in self._client.markets
            if '/CAD' in symbol
        ]
