#!/usr/bin/env python3
"""
Moomoo (Futu) Exchange Connector
==================================
Connector for Moomoo/Futu — US, HK, CN stock and options trading.

Moomoo is the international brand of Futu Holdings (FUTU).
Uses the moomoo-api (OpenD gateway) SDK for data and trading.

Requires:
    - Moomoo OpenD gateway running locally (port 11111)
    - pip install moomoo-api
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
    from moomoo import (
        OpenQuoteContext,
        OpenSecTradeContext,
        TrdEnv,
        TrdMarket,
        TrdSide,
        OrderType as MooOrderType,
        RET_OK,
        SubType,
        SecurityFirm,
    )
    MOOMOO_AVAILABLE = True
except ImportError:
    MOOMOO_AVAILABLE = False


class MoomooConnector(BaseExchangeConnector):
    """
    Moomoo/Futu exchange connector.

    Supports US stocks, HK stocks, CN A-shares, options.
    Requires Moomoo OpenD gateway running locally.
    """

    @property
    def name(self) -> str:
        return "moomoo"

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 11111,
        paper: bool = True,
        market: str = 'US',
        security_firm: str = 'FUTUINC',
        api_key: str = '',
        api_secret: str = '',
        testnet: bool = True,
        rate_limit: int = 30,
    ):
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            testnet=paper,
            rate_limit=rate_limit,
        )

        config = get_config()
        self.host = host
        self.port = port
        self.paper = paper or config.__dict__.get('moomoo_paper', True)
        self.market = market
        self.security_firm = security_firm

        self._quote_ctx: Optional[Any] = None
        self._trade_ctx: Optional[Any] = None

    def _get_trd_env(self):
        """Get trading environment (paper vs real)"""
        return TrdEnv.SIMULATE if self.paper else TrdEnv.REAL

    def _get_trd_market(self):
        """Map market string to TrdMarket enum"""
        market_map = {
            'US': TrdMarket.US,
            'HK': TrdMarket.HK,
            'CN': TrdMarket.CN,
            'SG': TrdMarket.SG,
        }
        return market_map.get(self.market.upper(), TrdMarket.US)

    def _format_symbol(self, symbol: str) -> str:
        """
        Format symbol for Moomoo API.

        Input:  'AAPL' or 'AAPL:US' or 'US.AAPL'
        Output: 'US.AAPL'
        """
        if '.' in symbol and symbol.split('.')[0] in ('US', 'HK', 'CN', 'SG'):
            return symbol

        parts = symbol.split(':')
        ticker = parts[0]
        market = parts[1].upper() if len(parts) > 1 else self.market.upper()
        return f"{market}.{ticker}"

    async def connect(self) -> bool:
        """Connect to Moomoo OpenD"""
        import time
        start_time = time.time()

        if not MOOMOO_AVAILABLE:
            self.logger.error("moomoo-api not installed. Run: pip install moomoo-api")
            await self._audit_auth("failure", "moomoo-api library not installed")
            return False

        try:
            # Quote context for market data
            self._quote_ctx = OpenQuoteContext(host=self.host, port=self.port)

            # Trade context for order management
            firm = SecurityFirm.FUTUINC
            if self.security_firm.upper() == 'FUTUSG':
                firm = SecurityFirm.FUTUSG

            self._trade_ctx = OpenSecTradeContext(
                host=self.host,
                port=self.port,
                security_firm=firm,
                filter_trdmarket=self._get_trd_market(),
            )

            self._connected = True
            mode = 'paper' if self.paper else 'live'
            self.logger.info(f"Connected to Moomoo ({mode}) at {self.host}:{self.port}")

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("connect", "MOOMOO", "success", duration_ms)
            await self._audit_auth("success")

            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to Moomoo: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("connect", "MOOMOO", "failure", duration_ms, error_message=str(e))
            raise ConnectionError(
                f"Cannot connect to Moomoo OpenD at {self.host}:{self.port}. "
                "Ensure OpenD gateway is running."
            )

    async def disconnect(self) -> None:
        """Disconnect from Moomoo"""
        if self._quote_ctx:
            self._quote_ctx.close()
            self._quote_ctx = None
        if self._trade_ctx:
            self._trade_ctx.close()
            self._trade_ctx = None
        self._connected = False
        self.logger.info("Disconnected from Moomoo")

    def _ensure_connected(self):
        if not self._quote_ctx or not self._trade_ctx:
            raise ConnectionError("Not connected to Moomoo. Call connect() first.")

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price data"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            moo_symbol = self._format_symbol(symbol)

            ret, data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._quote_ctx.get_stock_quote([moo_symbol])
            )

            if ret != RET_OK:
                raise ExchangeError(f"Moomoo get_stock_quote failed: {data}")

            if data is not None and len(data) > 0:
                row = data.iloc[0]
                return Ticker(
                    symbol=symbol,
                    bid=float(row.get('bid_price', 0) or 0),
                    ask=float(row.get('ask_price', 0) or 0),
                    last=float(row.get('last_price', 0) or 0),
                    volume_24h=float(row.get('volume', 0) or 0),
                    timestamp=datetime.now(),
                )

            raise ExchangeError(f"No data returned for {symbol}")

        except ExchangeError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to fetch Moomoo ticker for {symbol}: {e}")
            raise ExchangeError(str(e))

    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get order book / market depth"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            moo_symbol = self._format_symbol(symbol)

            ret, data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._quote_ctx.get_order_book(moo_symbol, num=limit)
            )

            if ret != RET_OK:
                raise ExchangeError(f"Moomoo get_order_book failed: {data}")

            bids = []
            asks = []

            if data is not None:
                bid_df = data.get('Bid', None) if isinstance(data, dict) else None
                ask_df = data.get('Ask', None) if isinstance(data, dict) else None

                if bid_df is not None:
                    for _, row in bid_df.iterrows():
                        bids.append((float(row['price']), float(row['volume'])))
                if ask_df is not None:
                    for _, row in ask_df.iterrows():
                        asks.append((float(row['price']), float(row['volume'])))

            return OrderBook(
                symbol=symbol,
                bids=bids[:limit],
                asks=asks[:limit],
                timestamp=datetime.now(),
            )

        except ExchangeError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to fetch Moomoo orderbook for {symbol}: {e}")
            raise ExchangeError(str(e))

    async def get_balances(self) -> Dict[str, Balance]:
        """Get account cash balances"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            ret, data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._trade_ctx.accinfo_query(trd_env=self._get_trd_env())
            )

            if ret != RET_OK:
                raise ExchangeError(f"Moomoo accinfo_query failed: {data}")

            result = {}
            if data is not None and len(data) > 0:
                row = data.iloc[0]
                cash = float(row.get('cash', 0) or 0)
                frozen = float(row.get('frozen_cash', 0) or 0)
                currency = row.get('currency', 'USD')

                result[currency] = Balance(
                    asset=currency,
                    free=cash - frozen,
                    locked=frozen,
                )

            # Also get stock positions
            ret2, pos_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._trade_ctx.position_list_query(trd_env=self._get_trd_env())
            )

            if ret2 == RET_OK and pos_data is not None:
                for _, pos in pos_data.iterrows():
                    sym = str(pos.get('stock_name', pos.get('code', '')))
                    qty = float(pos.get('qty', 0) or 0)
                    if qty != 0:
                        result[sym] = Balance(
                            asset=sym,
                            free=qty,
                            locked=0.0,
                        )

            return result

        except ExchangeError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to fetch Moomoo balances: {e}")
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
        """Create an order on Moomoo"""
        self._ensure_connected()
        await self._rate_limit_wait()

        import time
        start_time = time.time()

        try:
            moo_symbol = self._format_symbol(symbol)
            trd_side = TrdSide.BUY if side.lower() == 'buy' else TrdSide.SELL

            if order_type.lower() == 'market':
                moo_order_type = MooOrderType.MARKET
                price = 0.0  # Moomoo requires price even for market orders
            elif order_type.lower() == 'limit':
                moo_order_type = MooOrderType.NORMAL
                if price is None:
                    raise OrderError("Price required for limit orders")
            else:
                raise OrderError(f"Unsupported order type: {order_type}")

            ret, data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._trade_ctx.place_order(
                    price=price or 0,
                    qty=quantity,
                    code=moo_symbol,
                    trd_side=trd_side,
                    order_type=moo_order_type,
                    trd_env=self._get_trd_env(),
                    remark=client_order_id or '',
                )
            )

            if ret != RET_OK:
                raise OrderError(f"Moomoo place_order failed: {data}")

            row = data.iloc[0] if data is not None and len(data) > 0 else {}

            result = ExchangeOrder(
                order_id=str(row.get('order_id', '')),
                client_order_id=client_order_id,
                symbol=symbol,
                side=side.lower(),
                order_type=order_type.lower(),
                quantity=quantity,
                price=price,
                status=self._map_moo_status(str(row.get('order_status', ''))),
                filled_quantity=float(row.get('dealt_qty', 0) or 0),
                average_price=float(row.get('dealt_avg_price', 0) or 0),
                fee=0.0,
            )

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "MOOMOO", "success", duration_ms)
            await self._audit_order(symbol, side, order_type, quantity, price, result.order_id, "created")

            return result

        except (OrderError, ExchangeError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to create Moomoo order: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "MOOMOO", "failure", duration_ms, error_message=str(e))
            raise ExchangeError(str(e))

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            ret, data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._trade_ctx.modify_order(
                    modify_order_op=1,  # Cancel
                    order_id=order_id,
                    qty=0,
                    price=0,
                    trd_env=self._get_trd_env(),
                )
            )

            if ret != RET_OK:
                self.logger.warning(f"Moomoo cancel_order failed: {data}")
                return False
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel Moomoo order {order_id}: {e}")
            raise ExchangeError(str(e))

    async def get_order(self, order_id: str, symbol: str) -> ExchangeOrder:
        """Get order status"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            ret, data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._trade_ctx.order_list_query(
                    order_id=order_id,
                    trd_env=self._get_trd_env(),
                )
            )

            if ret != RET_OK or data is None or len(data) == 0:
                raise OrderError(f"Order {order_id} not found on Moomoo")

            row = data.iloc[0]
            return ExchangeOrder(
                order_id=order_id,
                client_order_id=str(row.get('remark', '')) or None,
                symbol=symbol,
                side='buy' if 'BUY' in str(row.get('trd_side', '')).upper() else 'sell',
                order_type='limit' if 'NORMAL' in str(row.get('order_type', '')).upper() else 'market',
                quantity=float(row.get('qty', 0) or 0),
                price=float(row.get('price', 0) or 0),
                status=self._map_moo_status(str(row.get('order_status', ''))),
                filled_quantity=float(row.get('dealt_qty', 0) or 0),
                average_price=float(row.get('dealt_avg_price', 0) or 0),
                fee=0.0,
            )

        except OrderError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get Moomoo order {order_id}: {e}")
            raise ExchangeError(str(e))

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get all open orders"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            ret, data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._trade_ctx.order_list_query(
                    trd_env=self._get_trd_env(),
                    status_filter_list=['SUBMITTED', 'WAITING_SUBMIT', 'SUBMITTING'],
                )
            )

            if ret != RET_OK or data is None:
                return []

            orders = []
            for _, row in data.iterrows():
                code = str(row.get('code', ''))
                if symbol:
                    moo_symbol = self._format_symbol(symbol)
                    if code != moo_symbol:
                        continue

                orders.append(ExchangeOrder(
                    order_id=str(row.get('order_id', '')),
                    client_order_id=str(row.get('remark', '')) or None,
                    symbol=code,
                    side='buy' if 'BUY' in str(row.get('trd_side', '')).upper() else 'sell',
                    order_type='limit',
                    quantity=float(row.get('qty', 0) or 0),
                    price=float(row.get('price', 0) or 0),
                    status='open',
                    filled_quantity=float(row.get('dealt_qty', 0) or 0),
                    average_price=float(row.get('dealt_avg_price', 0) or 0),
                    fee=0.0,
                ))

            return orders

        except Exception as e:
            self.logger.error(f"Failed to fetch Moomoo open orders: {e}")
            raise ExchangeError(str(e))

    async def get_trade_fee(self, symbol: str) -> Dict[str, float]:
        """Moomoo US stock commissions"""
        return {'maker': 0.0, 'taker': 0.0099}  # $0.0099/share, $0.99 minimum

    @staticmethod
    def _map_moo_status(status: str) -> str:
        """Map Moomoo order status to unified status"""
        status_upper = status.upper()
        if 'FILLED_ALL' in status_upper:
            return 'filled'
        if 'FILLED_PART' in status_upper:
            return 'partially_filled'
        if 'SUBMITTED' in status_upper or 'SUBMITTING' in status_upper or 'WAITING' in status_upper:
            return 'open'
        if 'CANCELLED' in status_upper or 'DELETED' in status_upper:
            return 'cancelled'
        if 'FAILED' in status_upper or 'DISABLED' in status_upper:
            return 'rejected'
        return 'unknown'
