#!/usr/bin/env python3
"""
Noxi Rise / MetaTrader 5 Connector
====================================
Connector for Noxi Rise — institutional multi-asset trading platform
that uses MetaTrader 5 as its execution engine.

Noxi Rise (noxirise.com) is an MT5-based broker providing forex,
commodities, indices, and crypto CFD trading.

Requires:
    - MetaTrader 5 terminal installed and logged in to a Noxi Rise account
    - pip install MetaTrader5
    - Windows only (MT5 Python API is Windows-native)
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config

try:
    from shared.utils import CircuitOpenError, with_circuit_breaker
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
    AuthenticationError,
    Balance,
    BaseExchangeConnector,
    ConnectionError,
    ExchangeError,
    ExchangeOrder,
    InsufficientFundsError,
    OrderBook,
    OrderError,
    Ticker,
)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


class NoxiRiseConnector(BaseExchangeConnector):
    """
    Noxi Rise connector via MetaTrader 5 Python API.

    Supports: Forex, Indices, Commodities, Crypto CFDs.
    Uses the MT5 terminal as the execution gateway.
    """

    @property
    def name(self) -> str:
        """Name."""
        return "noxi_rise"

    def __init__(
        self,
        mt5_path: str = '',
        login: int = 0,
        password: str = '',
        server: str = 'NoxiRise-Live',
        api_key: str = '',
        api_secret: str = '',
        testnet: bool = True,
        rate_limit: int = 30,
    ):
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            rate_limit=rate_limit,
        )

        config = get_config()
        self.mt5_path = mt5_path or config.__dict__.get('mt5_path', '')
        self.login = login or config.__dict__.get('mt5_login', 0)
        self.password = password or config.__dict__.get('mt5_password', '')
        self.server = server or config.__dict__.get('mt5_server', 'NoxiRise-Live')

    async def connect(self) -> bool:
        """Initialize MT5 terminal and login"""
        import time
        start_time = time.time()

        if not MT5_AVAILABLE:
            self.logger.error(
                "MetaTrader5 not installed. Run: pip install MetaTrader5 (Windows only)"
            )
            await self._audit_auth("failure", "MetaTrader5 library not installed")
            return False

        try:
            # Initialize MT5
            init_kwargs = {}
            if self.mt5_path:
                init_kwargs['path'] = self.mt5_path

            if self.login and self.password:
                init_kwargs['login'] = self.login
                init_kwargs['password'] = self.password
                init_kwargs['server'] = self.server

            initialized = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: mt5.initialize(**init_kwargs) if init_kwargs else mt5.initialize()
            )

            if not initialized:
                error = mt5.last_error()
                raise ConnectionError(
                    f"MT5 initialization failed: {error}. "
                    "Ensure MetaTrader 5 terminal is installed and a Noxi Rise account is configured."
                )

            # Get account info to verify connection
            account_info = mt5.account_info()
            if account_info is None:
                raise AuthenticationError("Failed to get MT5 account info. Check login credentials.")

            self._connected = True
            self.logger.info(
                f"Connected to Noxi Rise MT5 — Login: {account_info.login}, "
                f"Server: {account_info.server}, Balance: {account_info.balance}"
            )

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("connect", "MT5", "success", duration_ms)
            await self._audit_auth("success")

            return True

        except (ConnectionError, AuthenticationError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to connect to Noxi Rise MT5: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("connect", "MT5", "failure", duration_ms, error_message=str(e))
            raise ConnectionError(str(e))

    async def disconnect(self) -> None:
        """Shutdown MT5 connection"""
        if MT5_AVAILABLE:
            mt5.shutdown()
        self._connected = False
        self.logger.info("Disconnected from Noxi Rise MT5")

    def _ensure_connected(self):
        if not self._connected:
            raise ConnectionError("Not connected to MT5. Call connect() first.")

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price for a symbol (e.g. 'EURUSD', 'BTCUSD', 'XAUUSD')"""
        self._ensure_connected()
        await self._rate_limit_wait()
        return await self._get_ticker_with_breaker(symbol)

    @with_circuit_breaker("noxirise_ticker", failure_threshold=5, timeout=30.0)
    async def _get_ticker_with_breaker(self, symbol: str) -> Ticker:
        try:
            tick = await asyncio.get_event_loop().run_in_executor(
                None, lambda: mt5.symbol_info_tick(symbol)
            )

            if tick is None:
                # Try to enable the symbol first
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: mt5.symbol_select(symbol, True)
                )
                tick = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: mt5.symbol_info_tick(symbol)
                )

            if tick is None:
                raise ExchangeError(f"Symbol {symbol} not found on Noxi Rise")

            return Ticker(
                symbol=symbol,
                bid=float(tick.bid),
                ask=float(tick.ask),
                last=float(tick.last) if tick.last > 0 else float(tick.bid),
                volume_24h=float(tick.volume) if tick.volume else 0.0,
                timestamp=datetime.fromtimestamp(tick.time),
            )

        except ExchangeError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to fetch Noxi Rise ticker for {symbol}: {e}")
            raise ExchangeError(str(e))

    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """
        Get order book data.
        MT5 provides book data via symbol_info_tick (bid/ask only)
        and market depth via market_book_add/get.
        """
        self._ensure_connected()
        await self._rate_limit_wait()
        return await self._get_orderbook_with_breaker(symbol, limit)

    @with_circuit_breaker("noxirise_orderbook", failure_threshold=5, timeout=30.0)
    async def _get_orderbook_with_breaker(self, symbol: str, limit: int = 20) -> OrderBook:
        try:
            # Enable market depth for the symbol
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: mt5.market_book_add(symbol)
            )
            await asyncio.sleep(0.5)

            book = await asyncio.get_event_loop().run_in_executor(
                None, lambda: mt5.market_book_get(symbol)
            )

            bids = []
            asks = []

            if book:
                for entry in book:
                    if entry.type == mt5.BOOK_TYPE_SELL or entry.type == mt5.BOOK_TYPE_SELL_MARKET:
                        asks.append((float(entry.price), float(entry.volume)))
                    elif entry.type == mt5.BOOK_TYPE_BUY or entry.type == mt5.BOOK_TYPE_BUY_MARKET:
                        bids.append((float(entry.price), float(entry.volume)))

            # Release market depth
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: mt5.market_book_release(symbol)
            )

            return OrderBook(
                symbol=symbol,
                bids=sorted(bids, key=lambda x: -x[0])[:limit],
                asks=sorted(asks, key=lambda x: x[0])[:limit],
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Failed to fetch Noxi Rise orderbook for {symbol}: {e}")
            raise ExchangeError(str(e))

    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balance and equity"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            info = await asyncio.get_event_loop().run_in_executor(
                None, mt5.account_info
            )

            if info is None:
                raise ExchangeError("Failed to get MT5 account info")

            result = {}

            # Account currency balance
            result[info.currency] = Balance(
                asset=info.currency,
                free=float(info.margin_free),
                locked=float(info.margin),
            )

            # Open positions as balances
            positions = await asyncio.get_event_loop().run_in_executor(
                None, mt5.positions_get
            )

            if positions:
                for pos in positions:
                    result[pos.symbol] = Balance(
                        asset=pos.symbol,
                        free=float(pos.volume),
                        locked=0.0,
                    )

            return result

        except ExchangeError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to fetch Noxi Rise balances: {e}")
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
        """Create an order via MT5"""
        self._ensure_connected()
        await self._rate_limit_wait()
        return await self._create_order_with_breaker(
            symbol, side, order_type, quantity, price, client_order_id
        )

    @with_circuit_breaker("noxirise_order", failure_threshold=3, timeout=60.0)
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
            # Get current price for market orders
            tick = await asyncio.get_event_loop().run_in_executor(
                None, lambda: mt5.symbol_info_tick(symbol)
            )
            if tick is None:
                raise ExchangeError(f"Cannot get price for {symbol}")

            request = {
                'action': mt5.TRADE_ACTION_DEAL if order_type.lower() == 'market' else mt5.TRADE_ACTION_PENDING,
                'symbol': symbol,
                'volume': float(quantity),
                'type': mt5.ORDER_TYPE_BUY if side.lower() == 'buy' else mt5.ORDER_TYPE_SELL,
                'deviation': 20,  # Max price deviation in points
                'magic': 234000,  # Magic number for AAC orders
                'comment': client_order_id or 'AAC_order',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }

            if order_type.lower() == 'market':
                request['price'] = tick.ask if side.lower() == 'buy' else tick.bid
            elif order_type.lower() == 'limit':
                if price is None:
                    raise OrderError("Price required for limit orders")
                request['price'] = price
                request['type'] = (
                    mt5.ORDER_TYPE_BUY_LIMIT if side.lower() == 'buy'
                    else mt5.ORDER_TYPE_SELL_LIMIT
                )

            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: mt5.order_send(request)
            )

            if result is None:
                raise ExchangeError(f"MT5 order_send returned None: {mt5.last_error()}")

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise OrderError(
                    f"MT5 order rejected: {result.comment} (code: {result.retcode})"
                )

            order_result = ExchangeOrder(
                order_id=str(result.order),
                client_order_id=client_order_id,
                symbol=symbol,
                side=side.lower(),
                order_type=order_type.lower(),
                quantity=quantity,
                price=float(result.price) if result.price else price,
                status='filled' if order_type.lower() == 'market' else 'open',
                filled_quantity=float(result.volume),
                average_price=float(result.price) if result.price else 0,
                fee=0.0,
            )

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "MT5", "success", duration_ms)
            await self._audit_order(symbol, side, order_type, quantity, price, order_result.order_id, "created")

            return order_result

        except (OrderError, ExchangeError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to create Noxi Rise order: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "MT5", "failure", duration_ms, error_message=str(e))
            raise ExchangeError(str(e))

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a pending order"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            request = {
                'action': mt5.TRADE_ACTION_REMOVE,
                'order': int(order_id),
            }

            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: mt5.order_send(request)
            )

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return True

            self.logger.warning(f"Failed to cancel MT5 order {order_id}: {result}")
            return False

        except Exception as e:
            self.logger.error(f"Failed to cancel Noxi Rise order {order_id}: {e}")
            raise ExchangeError(str(e))

    async def get_order(self, order_id: str, symbol: str) -> ExchangeOrder:
        """Get order details"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            # Check pending orders
            orders = await asyncio.get_event_loop().run_in_executor(
                None, mt5.orders_get
            )
            if orders:
                for order in orders:
                    if str(order.ticket) == order_id:
                        return self._parse_mt5_order(order, symbol)

            # Check completed deals
            deals = await asyncio.get_event_loop().run_in_executor(
                None, lambda: mt5.history_orders_get(ticket=int(order_id))
            )
            if deals and len(deals) > 0:
                return self._parse_mt5_order(deals[0], symbol)

            raise OrderError(f"Order {order_id} not found on Noxi Rise")

        except OrderError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get Noxi Rise order {order_id}: {e}")
            raise ExchangeError(str(e))

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get all pending orders"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            if symbol:
                orders = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: mt5.orders_get(symbol=symbol)
                )
            else:
                orders = await asyncio.get_event_loop().run_in_executor(
                    None, mt5.orders_get
                )

            if not orders:
                return []

            return [self._parse_mt5_order(o, symbol or o.symbol) for o in orders]

        except Exception as e:
            self.logger.error(f"Failed to fetch Noxi Rise open orders: {e}")
            raise ExchangeError(str(e))

    async def get_trade_fee(self, symbol: str) -> Dict[str, float]:
        """Get estimated spread-based fees"""
        return {'maker': 0.0, 'taker': 0.0}  # MT5 brokers charge via spread

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """Create a stop order via MT5"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            if limit_price:
                order_type_mt5 = (
                    mt5.ORDER_TYPE_BUY_STOP_LIMIT if side.lower() == 'buy'
                    else mt5.ORDER_TYPE_SELL_STOP_LIMIT
                )
            else:
                order_type_mt5 = (
                    mt5.ORDER_TYPE_BUY_STOP if side.lower() == 'buy'
                    else mt5.ORDER_TYPE_SELL_STOP
                )

            request = {
                'action': mt5.TRADE_ACTION_PENDING,
                'symbol': symbol,
                'volume': float(quantity),
                'type': order_type_mt5,
                'price': stop_price,
                'deviation': 20,
                'magic': 234000,
                'comment': client_order_id or 'AAC_stop',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }

            if limit_price:
                request['stoplimit'] = limit_price

            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: mt5.order_send(request)
            )

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                raise OrderError(f"MT5 stop order rejected: {result}")

            return ExchangeOrder(
                order_id=str(result.order),
                client_order_id=client_order_id,
                symbol=symbol,
                side=side.lower(),
                order_type='stop_limit' if limit_price else 'stop',
                quantity=quantity,
                price=stop_price,
                status='open',
                filled_quantity=0.0,
                average_price=0.0,
                fee=0.0,
            )

        except OrderError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to create Noxi Rise stop order: {e}")
            raise ExchangeError(str(e))

    @staticmethod
    def _parse_mt5_order(order: Any, symbol: str) -> ExchangeOrder:
        """Parse MT5 order object to ExchangeOrder"""
        side_map = {
            0: 'buy',   # ORDER_TYPE_BUY
            1: 'sell',  # ORDER_TYPE_SELL
            2: 'buy',   # ORDER_TYPE_BUY_LIMIT
            3: 'sell',  # ORDER_TYPE_SELL_LIMIT
            4: 'buy',   # ORDER_TYPE_BUY_STOP
            5: 'sell',  # ORDER_TYPE_SELL_STOP
        }
        type_map = {
            0: 'market',
            1: 'market',
            2: 'limit',
            3: 'limit',
            4: 'stop',
            5: 'stop',
        }
        state_map = {
            0: 'pending',   # ORDER_STATE_STARTED
            1: 'open',      # ORDER_STATE_PLACED
            2: 'cancelled', # ORDER_STATE_CANCELED
            3: 'partially_filled',  # ORDER_STATE_PARTIAL
            4: 'filled',    # ORDER_STATE_FILLED
            5: 'rejected',  # ORDER_STATE_REJECTED
            6: 'expired',   # ORDER_STATE_EXPIRED
        }

        order_type_int = getattr(order, 'type', 0)

        return ExchangeOrder(
            order_id=str(getattr(order, 'ticket', '')),
            client_order_id=getattr(order, 'comment', None),
            symbol=symbol,
            side=side_map.get(order_type_int, 'buy'),
            order_type=type_map.get(order_type_int, 'market'),
            quantity=float(getattr(order, 'volume_initial', 0)),
            price=float(getattr(order, 'price_open', 0)),
            status=state_map.get(getattr(order, 'state', 0), 'unknown'),
            filled_quantity=float(getattr(order, 'volume_current', 0)),
            average_price=float(getattr(order, 'price_current', 0)),
            fee=0.0,
        )
