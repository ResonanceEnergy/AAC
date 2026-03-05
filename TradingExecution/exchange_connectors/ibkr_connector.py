#!/usr/bin/env python3
"""
Interactive Brokers (IBKR) Exchange Connector
==============================================
Implementation of the exchange connector for Interactive Brokers
using the ib_insync library (async wrapper around the official IB API).

Requires:
    - IB Gateway or Trader Workstation (TWS) running locally
    - pip install ib_insync
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
    from ib_insync import IB, Stock, Forex, Crypto, Contract
    from ib_insync import MarketOrder, LimitOrder, StopOrder, StopLimitOrder
    from ib_insync import util as ib_util
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False
    # Placeholder classes for when ib_insync is not installed
    Stock = Forex = Crypto = Contract = None
    MarketOrder = LimitOrder = StopOrder = StopLimitOrder = None


class IBKRConnector(BaseExchangeConnector):
    """
    Interactive Brokers connector using ib_insync.

    Requires IB Gateway or TWS running locally.
    Supports stocks, forex, crypto, futures, and options.
    """

    # Symbol mapping: unified format -> IB contract
    ASSET_TYPE_MAP = {
        'stock': Stock,
        'forex': Forex,
        'crypto': Crypto,
    }

    @property
    def name(self) -> str:
        return "ibkr"

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 7497,
        client_id: int = 1,
        account: str = '',
        paper: bool = True,
        rate_limit: int = 50,
        api_key: str = '',
        api_secret: str = '',
        testnet: bool = True,
    ):
        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            testnet=paper,
            rate_limit=rate_limit,
        )

        config = get_config()
        self.host = host or config.__dict__.get('ibkr_host', '127.0.0.1')
        self.port = port or config.__dict__.get('ibkr_port', 7497)
        self.client_id = client_id or config.__dict__.get('ibkr_client_id', 1)
        self.account = account or config.__dict__.get('ibkr_account', '')
        self.paper = paper
        self._ib: Optional[Any] = None
        self._contracts_cache: Dict[str, Contract] = {}

    def _parse_symbol(self, symbol: str) -> 'Contract':
        """
        Parse unified symbol format into an IB Contract.

        Formats:
            'AAPL'          -> Stock('AAPL', 'SMART', 'USD')
            'AAPL:stock'    -> Stock('AAPL', 'SMART', 'USD')
            'EUR/USD:forex' -> Forex('EURUSD')
            'BTC/USD:crypto'-> Crypto('BTC', 'PAXOS', 'USD')
            'AAPL:NASDAQ'   -> Stock('AAPL', 'NASDAQ', 'USD')
        """
        if symbol in self._contracts_cache:
            return self._contracts_cache[symbol]

        parts = symbol.split(':')
        ticker = parts[0]
        asset_type = parts[1].lower() if len(parts) > 1 else 'stock'

        if asset_type == 'forex':
            pair = ticker.replace('/', '')
            contract = Forex(pair)
        elif asset_type == 'crypto':
            base = ticker.split('/')[0] if '/' in ticker else ticker
            contract = Crypto(base, 'PAXOS', 'USD')
        elif asset_type in ('stock', 'SMART', 'NASDAQ', 'NYSE', 'ARCA', 'AMEX'):
            exchange = asset_type if asset_type not in ('stock',) else 'SMART'
            contract = Stock(ticker, exchange, 'USD')
        else:
            contract = Stock(ticker, 'SMART', 'USD')

        self._contracts_cache[symbol] = contract
        return contract

    async def connect(self) -> bool:
        """Connect to IB Gateway / TWS"""
        import time
        start_time = time.time()

        if not IB_INSYNC_AVAILABLE:
            self.logger.error("ib_insync not installed. Run: pip install ib_insync")
            await self._audit_auth("failure", "ib_insync library not installed")
            return False

        try:
            self._ib = IB()
            # ib_insync connect is synchronous but we wrap it
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._ib.connect(
                    self.host,
                    self.port,
                    clientId=self.client_id,
                    readonly=False,
                    account=self.account or '',
                )
            )

            self._connected = True

            # Get account info
            accounts = self._ib.managedAccounts()
            if not self.account and accounts:
                self.account = accounts[0]

            mode = 'paper' if self.paper else 'live'
            self.logger.info(
                f"Connected to IBKR ({mode}) at {self.host}:{self.port} "
                f"account={self.account}"
            )

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("connect", "IB", "success", duration_ms)
            await self._audit_auth("success")

            return True

        except Exception as e:
            err_msg = str(e)
            self.logger.error(f"Failed to connect to IBKR: {err_msg}")
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("connect", "IB", "failure", duration_ms, error_message=err_msg)

            if 'connect' in err_msg.lower() or 'refused' in err_msg.lower():
                raise ConnectionError(
                    f"Cannot connect to IBKR at {self.host}:{self.port}. "
                    "Ensure IB Gateway or TWS is running."
                )
            raise ConnectionError(err_msg)

    async def disconnect(self) -> None:
        """Disconnect from IB Gateway / TWS"""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
        self._ib = None
        self._connected = False
        self._contracts_cache.clear()
        self.logger.info("Disconnected from IBKR")

    def _ensure_connected(self):
        """Raise if not connected"""
        if not self._ib or not self._ib.isConnected():
            raise ConnectionError("Not connected to IBKR. Call connect() first.")

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current market data for a symbol"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            contract = self._parse_symbol(symbol)
            self._ib.qualifyContracts(contract)

            # Request market data snapshot
            ticker_data = self._ib.reqMktData(contract, '', snapshot=True)
            # Wait for data to arrive
            for _ in range(50):  # max 5 seconds
                if ticker_data.last is not None or ticker_data.bid is not None:
                    break
                await asyncio.sleep(0.1)

            bid = float(ticker_data.bid) if ticker_data.bid and ticker_data.bid > 0 else 0.0
            ask = float(ticker_data.ask) if ticker_data.ask and ticker_data.ask > 0 else 0.0
            last = float(ticker_data.last) if ticker_data.last and ticker_data.last > 0 else 0.0
            volume = float(ticker_data.volume) if ticker_data.volume and ticker_data.volume > 0 else 0.0

            self._ib.cancelMktData(contract)

            return Ticker(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last,
                volume_24h=volume,
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Failed to fetch IBKR ticker for {symbol}: {e}")
            raise ExchangeError(str(e))

    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get Level II market depth"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            contract = self._parse_symbol(symbol)
            self._ib.qualifyContracts(contract)

            # Request market depth
            depth_data = self._ib.reqMktDepth(contract, numRows=limit)
            await asyncio.sleep(2)  # Wait for depth data

            bids = []
            asks = []

            if hasattr(depth_data, 'domBids'):
                for entry in depth_data.domBids:
                    bids.append((float(entry.price), float(entry.size)))
            if hasattr(depth_data, 'domAsks'):
                for entry in depth_data.domAsks:
                    asks.append((float(entry.price), float(entry.size)))

            self._ib.cancelMktDepth(contract)

            return OrderBook(
                symbol=symbol,
                bids=bids[:limit],
                asks=asks[:limit],
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Failed to fetch IBKR orderbook for {symbol}: {e}")
            raise ExchangeError(str(e))

    async def get_balances(self) -> Dict[str, Balance]:
        """Get account balances from IBKR"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            account_values = self._ib.accountValues(self.account)
            result = {}

            for av in account_values:
                if av.tag == 'CashBalance' and av.currency != 'BASE':
                    amount = float(av.value)
                    if amount != 0:
                        result[av.currency] = Balance(
                            asset=av.currency,
                            free=amount,
                            locked=0.0,
                        )

            # Also get portfolio positions
            positions = self._ib.positions(self.account)
            for pos in positions:
                sym = pos.contract.symbol
                qty = float(pos.position)
                if qty != 0:
                    result[sym] = Balance(
                        asset=sym,
                        free=qty,
                        locked=0.0,
                    )

            return result

        except Exception as e:
            self.logger.error(f"Failed to fetch IBKR balances: {e}")
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
        """Create an order on IBKR"""
        self._ensure_connected()
        await self._rate_limit_wait()

        import time
        start_time = time.time()

        try:
            contract = self._parse_symbol(symbol)
            self._ib.qualifyContracts(contract)

            action = 'BUY' if side.lower() == 'buy' else 'SELL'

            if order_type.lower() == 'market':
                ib_order = MarketOrder(action, quantity)
            elif order_type.lower() == 'limit':
                if price is None:
                    raise OrderError("Price required for limit orders")
                ib_order = LimitOrder(action, quantity, price)
            else:
                raise OrderError(f"Unsupported order type: {order_type}")

            if client_order_id:
                ib_order.orderRef = client_order_id

            trade = self._ib.placeOrder(contract, ib_order)
            # Wait for order acknowledgement
            await asyncio.sleep(0.5)

            result = ExchangeOrder(
                order_id=str(trade.order.orderId),
                client_order_id=client_order_id,
                symbol=symbol,
                side=side.lower(),
                order_type=order_type.lower(),
                quantity=quantity,
                price=price,
                status=self._map_ib_status(trade.orderStatus.status),
                filled_quantity=float(trade.orderStatus.filled),
                average_price=float(trade.orderStatus.avgFillPrice or 0),
                fee=float(trade.orderStatus.commission or 0),
            )

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "IB", "success", duration_ms)
            await self._audit_order(symbol, side, order_type, quantity, price, result.order_id, "created")

            return result

        except OrderError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to create IBKR order: {e}")
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "IB", "failure", duration_ms, error_message=str(e))
            await self._audit_order(symbol, side, order_type, quantity, price, None, "failed", str(e))
            raise ExchangeError(str(e))

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    await asyncio.sleep(0.5)
                    return True

            self.logger.warning(f"IBKR order {order_id} not found in open trades")
            return False

        except Exception as e:
            self.logger.error(f"Failed to cancel IBKR order {order_id}: {e}")
            raise ExchangeError(str(e))

    async def get_order(self, order_id: str, symbol: str) -> ExchangeOrder:
        """Get order status"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            for trade in self._ib.trades():
                if str(trade.order.orderId) == order_id:
                    return ExchangeOrder(
                        order_id=order_id,
                        client_order_id=trade.order.orderRef or None,
                        symbol=symbol,
                        side='buy' if trade.order.action == 'BUY' else 'sell',
                        order_type=trade.order.orderType.lower(),
                        quantity=float(trade.order.totalQuantity),
                        price=float(trade.order.lmtPrice or 0),
                        status=self._map_ib_status(trade.orderStatus.status),
                        filled_quantity=float(trade.orderStatus.filled),
                        average_price=float(trade.orderStatus.avgFillPrice or 0),
                        fee=float(trade.orderStatus.commission or 0),
                    )

            raise OrderError(f"Order {order_id} not found")

        except OrderError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get IBKR order {order_id}: {e}")
            raise ExchangeError(str(e))

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get all open orders"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            orders = []
            for trade in self._ib.openTrades():
                contract_sym = trade.contract.symbol
                if symbol and contract_sym.upper() != symbol.split(':')[0].upper():
                    continue

                orders.append(ExchangeOrder(
                    order_id=str(trade.order.orderId),
                    client_order_id=trade.order.orderRef or None,
                    symbol=f"{contract_sym}:stock",
                    side='buy' if trade.order.action == 'BUY' else 'sell',
                    order_type=trade.order.orderType.lower(),
                    quantity=float(trade.order.totalQuantity),
                    price=float(trade.order.lmtPrice or 0),
                    status=self._map_ib_status(trade.orderStatus.status),
                    filled_quantity=float(trade.orderStatus.filled),
                    average_price=float(trade.orderStatus.avgFillPrice or 0),
                    fee=float(trade.orderStatus.commission or 0),
                ))

            return orders

        except Exception as e:
            self.logger.error(f"Failed to fetch IBKR open orders: {e}")
            raise ExchangeError(str(e))

    async def get_trade_fee(self, symbol: str) -> Dict[str, float]:
        """Get estimated trading fees (IBKR tiered pricing defaults)"""
        return {'maker': 0.0005, 'taker': 0.0005}

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """Create a stop or stop-limit order"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            contract = self._parse_symbol(symbol)
            self._ib.qualifyContracts(contract)

            action = 'BUY' if side.lower() == 'buy' else 'SELL'

            if limit_price:
                ib_order = StopLimitOrder(action, quantity, limit_price, stop_price)
            else:
                ib_order = StopOrder(action, quantity, stop_price)

            if client_order_id:
                ib_order.orderRef = client_order_id

            trade = self._ib.placeOrder(contract, ib_order)
            await asyncio.sleep(0.5)

            return ExchangeOrder(
                order_id=str(trade.order.orderId),
                client_order_id=client_order_id,
                symbol=symbol,
                side=side.lower(),
                order_type='stop_limit' if limit_price else 'stop',
                quantity=quantity,
                price=limit_price or stop_price,
                status=self._map_ib_status(trade.orderStatus.status),
                filled_quantity=float(trade.orderStatus.filled),
                average_price=float(trade.orderStatus.avgFillPrice or 0),
                fee=float(trade.orderStatus.commission or 0),
            )

        except Exception as e:
            self.logger.error(f"Failed to create IBKR stop order: {e}")
            raise ExchangeError(str(e))

    # --- IBKR-Specific Methods ---

    async def get_portfolio(self) -> List[Dict[str, Any]]:
        """Get full portfolio positions with P&L"""
        self._ensure_connected()

        try:
            portfolio = self._ib.portfolio(self.account)
            result = []
            for item in portfolio:
                result.append({
                    'symbol': item.contract.symbol,
                    'sec_type': item.contract.secType,
                    'exchange': item.contract.exchange,
                    'position': float(item.position),
                    'market_price': float(item.marketPrice),
                    'market_value': float(item.marketValue),
                    'average_cost': float(item.averageCost),
                    'unrealized_pnl': float(item.unrealizedPNL),
                    'realized_pnl': float(item.realizedPNL),
                })
            return result

        except Exception as e:
            self.logger.error(f"Failed to get IBKR portfolio: {e}")
            raise ExchangeError(str(e))

    async def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary (net liquidation, buying power, etc.)"""
        self._ensure_connected()

        try:
            summary = self._ib.accountSummary(self.account)
            result = {}
            for item in summary:
                try:
                    result[item.tag] = float(item.value)
                except (ValueError, TypeError):
                    result[item.tag] = item.value
            return result

        except Exception as e:
            self.logger.error(f"Failed to get IBKR account summary: {e}")
            raise ExchangeError(str(e))

    async def get_historical_data(
        self,
        symbol: str,
        duration: str = '1 D',
        bar_size: str = '1 min',
        what_to_show: str = 'TRADES',
    ) -> List[Dict[str, Any]]:
        """Get historical OHLCV bars"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            contract = self._parse_symbol(symbol)
            self._ib.qualifyContracts(contract)

            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1,
            )

            return [
                {
                    'date': str(bar.date),
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume),
                }
                for bar in bars
            ]

        except Exception as e:
            self.logger.error(f"Failed to get IBKR historical data for {symbol}: {e}")
            raise ExchangeError(str(e))

    @staticmethod
    def _map_ib_status(ib_status: str) -> str:
        """Map IB order status to unified status"""
        status_map = {
            'PendingSubmit': 'pending',
            'PendingCancel': 'cancelling',
            'PreSubmitted': 'pending',
            'Submitted': 'open',
            'ApiPending': 'pending',
            'ApiCancelled': 'cancelled',
            'Cancelled': 'cancelled',
            'Filled': 'filled',
            'Inactive': 'rejected',
        }
        return status_map.get(ib_status, 'unknown')
