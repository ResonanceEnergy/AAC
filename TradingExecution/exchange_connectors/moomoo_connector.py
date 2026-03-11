#!/usr/bin/env python3
"""
Moomoo Exchange Connector
=========================
Implementation of the exchange connector for Moomoo (Futu) brokerage
via the official moomoo-api SDK.

Requirements:
    pip install moomoo-api

Connection:
    Requires Moomoo OpenD gateway running locally (or on a reachable host).
    Download OpenD from: https://www.moomoo.com/download/OpenD
    - Default host: 127.0.0.1
    - Default port: 11111
    - Supports: US, HK, CN, SG, AU, JP, CA markets

Configuration via .env:
    MOOMOO_HOST=127.0.0.1
    MOOMOO_PORT=11111
    MOOMOO_TRADE_ENV=SIMULATE    # SIMULATE or REAL
    MOOMOO_MARKET=US             # US, HK, CN, SG, AU, JP, CA
    MOOMOO_SECURITY_FIRM=FUTUINC
    MOOMOO_TRADE_PASSWORD=       # Required for live trading unlock
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

from shared.config_loader import get_env, get_env_int, get_env_bool

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

# Try to import moomoo SDK
try:
    from moomoo import (
        OpenQuoteContext,
        OpenSecTradeContext,
        OpenUSTradeContext,
        TrdSide,
        TrdEnv,
        OrderType as MooOrderType,
        OrderStatus as MooOrderStatus,
        TrdMarket,
        Market,
        SecurityFirm,
        RET_OK,
        RET_ERROR,
    )
    MOOMOO_AVAILABLE = True
except ImportError:
    MOOMOO_AVAILABLE = False


# Symbol mapping: AAC format -> Moomoo format
# AAC uses "AAPL/USD", Moomoo uses "US.AAPL"
MARKET_PREFIX_MAP = {
    "US": "US",
    "HK": "HK",
    "SH": "SH",
    "SZ": "SZ",
    "SG": "SG",
    "AU": "AU",
    "JP": "JP",
    "CA": "CA",
}


def _aac_to_moomoo_symbol(symbol: str, market: str = "US") -> str:
    """Convert AAC symbol (AAPL/USD) to Moomoo format (US.AAPL)."""
    base = symbol.split("/")[0].strip().upper()
    prefix = MARKET_PREFIX_MAP.get(market.upper(), "US")
    return f"{prefix}.{base}"


def _moomoo_to_aac_symbol(moomoo_symbol: str) -> str:
    """Convert Moomoo symbol (US.AAPL) to AAC format (AAPL/USD)."""
    parts = moomoo_symbol.split(".")
    if len(parts) == 2:
        market, ticker = parts
        currency = "USD" if market == "US" else "HKD" if market == "HK" else market
        return f"{ticker}/{currency}"
    return moomoo_symbol


class MoomooConnector(BaseExchangeConnector):
    """
    Moomoo/Futu exchange connector using the official moomoo-api SDK.

    Supports US, HK, CN, SG, AU, JP, CA markets for equities and options.
    Requires Moomoo OpenD gateway running locally.
    """

    @property
    def name(self) -> str:
        return "moomoo"

    def __init__(
        self,
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

        # Moomoo-specific config
        self._host = get_env('MOOMOO_HOST', '127.0.0.1')
        self._port = get_env_int('MOOMOO_PORT', 11111)
        self._market = get_env('MOOMOO_MARKET', 'US')
        self._security_firm_str = get_env('MOOMOO_SECURITY_FIRM', 'FUTUINC')
        self._trade_password = get_env('MOOMOO_TRADE_PASSWORD', '')
        self._trade_env_str = get_env('MOOMOO_TRADE_ENV', 'SIMULATE')

        # SDK contexts (initialized on connect)
        self._quote_ctx: Optional[Any] = None
        self._trade_ctx: Optional[Any] = None

    def _get_trd_env(self):
        """Get moomoo TrdEnv from config."""
        if not MOOMOO_AVAILABLE:
            return None
        return TrdEnv.SIMULATE if self._trade_env_str.upper() == 'SIMULATE' else TrdEnv.REAL

    def _get_trd_market(self):
        """Get moomoo TrdMarket from config."""
        if not MOOMOO_AVAILABLE:
            return None
        market_map = {
            'US': TrdMarket.US,
            'HK': TrdMarket.HK,
            'CN': TrdMarket.CN,
            'SG': TrdMarket.SG,
            'AU': TrdMarket.AU,
            'JP': TrdMarket.JP,
            'CA': TrdMarket.CA,
        }
        return market_map.get(self._market.upper(), TrdMarket.US)

    async def connect(self) -> bool:
        """Connect to Moomoo OpenD gateway."""
        start_time = time.time()

        if not MOOMOO_AVAILABLE:
            self.logger.error("moomoo-api not installed. Run: pip install moomoo-api")
            await self._audit_auth("failure", "moomoo-api not installed")
            return False

        try:
            # Quote context (market data)
            self._quote_ctx = OpenQuoteContext(
                host=self._host,
                port=self._port,
            )

            # Trade context
            self._trade_ctx = OpenSecTradeContext(
                host=self._host,
                port=self._port,
                security_firm=getattr(SecurityFirm, self._security_firm_str, SecurityFirm.FUTUINC),
                filter_trdmarket=self._get_trd_market(),
            )

            # Unlock trade if real mode and password provided
            if self._get_trd_env() == TrdEnv.REAL and self._trade_password:
                ret, data = self._trade_ctx.unlock_trade(self._trade_password)
                if ret != RET_OK:
                    self.logger.warning(f"Trade unlock failed: {data}")
                    await self._audit_auth("failure", f"Trade unlock failed: {data}")
                else:
                    self.logger.info("Moomoo trade unlocked for REAL trading")

            self._connected = True
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                f"Connected to Moomoo OpenD at {self._host}:{self._port} "
                f"(market={self._market}, env={self._trade_env_str}) "
                f"in {duration_ms:.0f}ms"
            )
            await self._audit_auth("success", f"Connected to {self._host}:{self._port}")
            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Moomoo connection failed: {e}")
            await self._audit_auth("failure", str(e))
            return False

    async def disconnect(self) -> None:
        """Close Moomoo connections."""
        if self._quote_ctx:
            self._quote_ctx.close()
            self._quote_ctx = None
        if self._trade_ctx:
            self._trade_ctx.close()
            self._trade_ctx = None
        self._connected = False
        self.logger.info("Moomoo disconnected")

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticker data."""
        if not self._quote_ctx:
            raise ConnectionError("Not connected to Moomoo")

        moo_sym = _aac_to_moomoo_symbol(symbol, self._market)

        def _fetch():
            ret, data = self._quote_ctx.get_market_snapshot([moo_sym])
            if ret != RET_OK:
                raise ExchangeError(f"Failed to get ticker: {data}")
            return data

        data = await asyncio.get_event_loop().run_in_executor(None, _fetch)

        if data.empty:
            raise ExchangeError(f"No data for {symbol}")

        row = data.iloc[0]
        start_time = time.time()
        ticker = Ticker(
            symbol=symbol,
            bid=float(row.get('bid_price', 0) or 0),
            ask=float(row.get('ask_price', 0) or 0),
            last=float(row.get('last_price', 0) or 0),
            volume_24h=float(row.get('volume', 0) or 0),
            timestamp=datetime.now(),
        )
        duration_ms = (time.time() - start_time) * 1000
        await self._audit_api_call("get_ticker", "GET", "success", duration_ms)
        return ticker

    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get order book."""
        if not self._quote_ctx:
            raise ConnectionError("Not connected to Moomoo")

        moo_sym = _aac_to_moomoo_symbol(symbol, self._market)

        def _fetch():
            ret, data = self._quote_ctx.get_order_book(moo_sym, num=limit)
            if ret != RET_OK:
                raise ExchangeError(f"Failed to get order book: {data}")
            return data

        data = await asyncio.get_event_loop().run_in_executor(None, _fetch)

        bids = []
        asks = []
        for _, row in data.iterrows():
            price = float(row.get('price', 0))
            volume = float(row.get('volume', 0))
            side = row.get('side', '')
            if side == 'Bid':
                bids.append((price, volume))
            elif side == 'Ask':
                asks.append((price, volume))

        return OrderBook(
            symbol=symbol,
            bids=sorted(bids, key=lambda x: x[0], reverse=True)[:limit],
            asks=sorted(asks, key=lambda x: x[0])[:limit],
            timestamp=datetime.now(),
        )

    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances."""
        if not self._trade_ctx:
            raise ConnectionError("Not connected to Moomoo")

        trd_env = self._get_trd_env()

        def _fetch():
            ret, data = self._trade_ctx.accinfo_query(trd_env=trd_env)
            if ret != RET_OK:
                raise ExchangeError(f"Failed to get account info: {data}")
            return data

        data = await asyncio.get_event_loop().run_in_executor(None, _fetch)

        balances = {}
        if not data.empty:
            row = data.iloc[0]
            cash = float(row.get('cash', 0) or 0)
            market_val = float(row.get('market_val', 0) or 0)
            frozen = float(row.get('frozen_cash', 0) or 0)
            currency = "USD" if self._market == "US" else "HKD"

            balances[currency] = Balance(
                asset=currency,
                free=cash - frozen,
                locked=frozen,
            )
            balances['PORTFOLIO'] = Balance(
                asset='PORTFOLIO',
                free=market_val,
                locked=0.0,
            )

        start_time = time.time()
        duration_ms = (time.time() - start_time) * 1000
        await self._audit_api_call("get_balances", "GET", "success", duration_ms)
        return balances

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """Place an order on Moomoo."""
        if not self._trade_ctx:
            raise ConnectionError("Not connected to Moomoo")

        start_time = time.time()
        moo_sym = _aac_to_moomoo_symbol(symbol, self._market)
        trd_env = self._get_trd_env()

        # Map side
        moo_side = TrdSide.BUY if side.lower() == 'buy' else TrdSide.SELL

        # Map order type
        if order_type.lower() == 'market':
            moo_type = MooOrderType.MARKET
        elif order_type.lower() == 'limit':
            moo_type = MooOrderType.NORMAL
            if price is None:
                raise OrderError("Limit orders require a price")
        else:
            moo_type = MooOrderType.NORMAL

        def _place():
            ret, data = self._trade_ctx.place_order(
                price=price or 0,
                qty=quantity,
                code=moo_sym,
                trd_side=moo_side,
                order_type=moo_type,
                trd_env=trd_env,
            )
            if ret != RET_OK:
                raise OrderError(f"Order placement failed: {data}")
            return data

        try:
            data = await asyncio.get_event_loop().run_in_executor(None, _place)
            row = data.iloc[0]
            order_id = str(row.get('order_id', ''))

            result = ExchangeOrder(
                order_id=order_id,
                client_order_id=client_order_id,
                symbol=symbol,
                side=side.lower(),
                order_type=order_type.lower(),
                quantity=quantity,
                price=price,
                status='open',
                raw=row.to_dict() if hasattr(row, 'to_dict') else {},
            )

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "POST", "success", duration_ms)
            await self._audit_order(symbol, side, order_type, quantity, price, result.order_id, "submitted")

            self.logger.info(
                f"Moomoo order placed: {side.upper()} {quantity} {symbol} "
                f"@ {'MKT' if order_type == 'market' else price} -> {order_id}"
            )
            return result

        except OrderError:
            raise
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "POST", "failure", duration_ms, error_message=str(e))
            await self._audit_order(symbol, side, order_type, quantity, price, None, "failed", str(e))
            raise OrderError(f"Moomoo order failed: {e}") from e

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        if not self._trade_ctx:
            raise ConnectionError("Not connected to Moomoo")

        start_time = time.time()
        trd_env = self._get_trd_env()

        def _cancel():
            from moomoo import ModifyOrderOp
            ret, data = self._trade_ctx.modify_order(
                modify_order_op=ModifyOrderOp.CANCEL,
                order_id=order_id,
                qty=0,
                price=0,
                trd_env=trd_env,
            )
            return ret == RET_OK

        try:
            success = await asyncio.get_event_loop().run_in_executor(None, _cancel)
            duration_ms = (time.time() - start_time) * 1000
            status = "success" if success else "failure"
            await self._audit_api_call("cancel_order", "POST", status, duration_ms)
            return success
        except Exception as e:
            self.logger.error(f"Cancel order failed: {e}")
            return False

    async def get_order(self, order_id: str, symbol: str) -> ExchangeOrder:
        """Get order details."""
        if not self._trade_ctx:
            raise ConnectionError("Not connected to Moomoo")

        trd_env = self._get_trd_env()

        def _fetch():
            ret, data = self._trade_ctx.order_list_query(
                order_id=order_id,
                trd_env=trd_env,
            )
            if ret != RET_OK:
                raise ExchangeError(f"Failed to get order: {data}")
            return data

        data = await asyncio.get_event_loop().run_in_executor(None, _fetch)

        if data.empty:
            raise ExchangeError(f"Order {order_id} not found")

        row = data.iloc[0]
        status_str = str(row.get('order_status', 'UNKNOWN')).lower()

        # Map Moomoo status to our status
        status_map = {
            'submitted': 'open',
            'filled_all': 'filled',
            'filled_part': 'partial',
            'cancelled_all': 'cancelled',
            'cancelled_part': 'cancelled',
            'failed': 'rejected',
        }
        mapped_status = status_map.get(status_str, 'open')

        return ExchangeOrder(
            order_id=order_id,
            client_order_id=None,
            symbol=_moomoo_to_aac_symbol(str(row.get('code', symbol))),
            side=str(row.get('trd_side', '')).lower(),
            order_type=str(row.get('order_type', '')).lower(),
            quantity=float(row.get('qty', 0)),
            price=float(row.get('price', 0) or 0),
            status=mapped_status,
            filled_quantity=float(row.get('dealt_qty', 0) or 0),
            average_price=float(row.get('dealt_avg_price', 0) or 0),
            raw=row.to_dict() if hasattr(row, 'to_dict') else {},
        )

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get all open orders."""
        if not self._trade_ctx:
            raise ConnectionError("Not connected to Moomoo")

        trd_env = self._get_trd_env()

        def _fetch():
            from moomoo import OrderStatus as MooOS
            ret, data = self._trade_ctx.order_list_query(
                trd_env=trd_env,
                status_filter_list=[
                    MooOS.SUBMITTED,
                    MooOS.FILLED_PART,
                ],
            )
            if ret != RET_OK:
                raise ExchangeError(f"Failed to get open orders: {data}")
            return data

        data = await asyncio.get_event_loop().run_in_executor(None, _fetch)

        orders = []
        if not data.empty:
            for _, row in data.iterrows():
                code = str(row.get('code', ''))
                aac_symbol = _moomoo_to_aac_symbol(code)

                if symbol and aac_symbol != symbol:
                    continue

                orders.append(ExchangeOrder(
                    order_id=str(row.get('order_id', '')),
                    client_order_id=None,
                    symbol=aac_symbol,
                    side=str(row.get('trd_side', '')).lower(),
                    order_type=str(row.get('order_type', '')).lower(),
                    quantity=float(row.get('qty', 0)),
                    price=float(row.get('price', 0) or 0),
                    status='open',
                    filled_quantity=float(row.get('dealt_qty', 0) or 0),
                    average_price=float(row.get('dealt_avg_price', 0) or 0),
                    raw=row.to_dict() if hasattr(row, 'to_dict') else {},
                ))

        return orders

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        if not self._trade_ctx:
            raise ConnectionError("Not connected to Moomoo")

        trd_env = self._get_trd_env()

        def _fetch():
            ret, data = self._trade_ctx.position_list_query(trd_env=trd_env)
            if ret != RET_OK:
                raise ExchangeError(f"Failed to get positions: {data}")
            return data

        data = await asyncio.get_event_loop().run_in_executor(None, _fetch)

        positions = []
        if not data.empty:
            for _, row in data.iterrows():
                positions.append({
                    'symbol': _moomoo_to_aac_symbol(str(row.get('code', ''))),
                    'quantity': float(row.get('qty', 0)),
                    'cost_price': float(row.get('cost_price', 0) or 0),
                    'market_value': float(row.get('market_val', 0) or 0),
                    'unrealized_pnl': float(row.get('pl_val', 0) or 0),
                    'unrealized_pnl_pct': float(row.get('pl_ratio', 0) or 0),
                    'side': str(row.get('position_side', 'LONG')),
                })

        return positions

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """Create a stop-loss order on Moomoo."""
        if not self._trade_ctx:
            raise ConnectionError("Not connected to Moomoo")

        start_time = time.time()
        moo_sym = _aac_to_moomoo_symbol(symbol, self._market)
        trd_env = self._get_trd_env()
        moo_side = TrdSide.BUY if side.lower() == 'buy' else TrdSide.SELL

        # Moomoo uses STOP or STOP_LIMIT order types
        if limit_price is not None:
            moo_type = MooOrderType.STOP_LIMIT
            order_price = limit_price
        else:
            moo_type = MooOrderType.STOP
            order_price = stop_price

        def _place():
            ret, data = self._trade_ctx.place_order(
                price=order_price,
                qty=quantity,
                code=moo_sym,
                trd_side=moo_side,
                order_type=moo_type,
                aux_price=stop_price,
                trd_env=trd_env,
            )
            if ret != RET_OK:
                raise OrderError(f"Stop-loss order failed: {data}")
            return data

        try:
            data = await asyncio.get_event_loop().run_in_executor(None, _place)
            row = data.iloc[0]
            order_id = str(row.get('order_id', ''))

            result = ExchangeOrder(
                order_id=order_id,
                client_order_id=client_order_id,
                symbol=symbol,
                side=side.lower(),
                order_type='stop_loss',
                quantity=quantity,
                price=limit_price or stop_price,
                status='open',
                raw=row.to_dict() if hasattr(row, 'to_dict') else {},
            )

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_stop_loss", "POST", "success", duration_ms)
            await self._audit_order(symbol, side, "stop_loss", quantity, order_price, order_id, "submitted")
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_stop_loss", "POST", "failure", duration_ms, error_message=str(e))
            await self._audit_order(symbol, side, "stop_loss", quantity, order_price, None, "failed", str(e))
            raise OrderError(f"Moomoo stop-loss failed: {e}") from e

    async def get_trade_fee(self, symbol: str) -> Dict[str, float]:
        """Get trading fees — Moomoo US equities fee structure."""
        # Moomoo US: Free commission, $0.99/order platform fee + regulatory fees
        return {'maker': 0.0, 'taker': 0.0, 'platform_fee_per_order': 0.99}
