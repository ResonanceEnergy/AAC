#!/usr/bin/env python3
"""
Interactive Brokers (IBKR) Exchange Connector
==============================================
Full-featured IBKR connector: stocks, options, futures, forex, crypto.
Streaming market data, auto-reconnection, error handling, heartbeat.

Requires:
    - IB Gateway or Trader Workstation (TWS) running locally
    - pip install ib_insync
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
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
    OrderError,
)

try:
    from ib_insync import IB, Stock, Forex, Crypto, Future, Option, Contract
    from ib_insync import MarketOrder, LimitOrder, StopOrder, StopLimitOrder
    from ib_insync import TrailingStopOrder
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False
    Stock = Forex = Crypto = Future = Option = Contract = None
    MarketOrder = LimitOrder = StopOrder = StopLimitOrder = TrailingStopOrder = None

logger = logging.getLogger(__name__)

# IB error codes and their meanings for smart error handling
IB_ERROR_CODES = {
    # Connection errors
    502: ("connection_lost", "Cannot connect to TWS/Gateway. Ensure it is running."),
    504: ("not_connected", "Not connected to TWS/Gateway."),
    1100: ("connection_lost", "Connectivity between IB and TWS has been lost."),
    1101: ("connection_restored", "Connectivity restored — data lost."),
    1102: ("connection_restored", "Connectivity restored — data maintained."),
    # Order errors
    103: ("duplicate_order", "Duplicate order ID."),
    104: ("cannot_modify", "Cannot modify a filled order."),
    105: ("cannot_modify", "Order being modified does not match original."),
    106: ("cannot_transmit", "Cannot transmit order — price does not conform to min tick."),
    110: ("price_error", "Price does not conform to minimum price variation."),
    135: ("cannot_cancel", "Cannot cancel order — not found."),
    161: ("cancel_attempted", "Cancel attempted while order already being cancelled."),
    201: ("order_rejected", "Order rejected — reason not specified."),
    202: ("order_cancelled", "Order cancelled."),
    # Market data errors
    354: ("no_subscription", "Requested market data is not subscribed. Delayed data available."),
    10090: ("no_market_data_perms", "Part of requested market data is not subscribed."),
    # Account errors
    321: ("server_error", "Server error when validating an API client request."),
    502: ("cant_connect", "Couldn't connect to TWS."),
    # Rate limiting
    100: ("max_rate", "Max rate of messages per second has been exceeded."),
    162: ("historical_data_pacing", "Historical market data pacing violation."),
}


class IBKRConnector(BaseExchangeConnector):
    """
    Full-featured Interactive Brokers connector.

    Supports: stocks, options, futures, forex, crypto.
    Features: streaming data, auto-reconnect, heartbeat, bracket orders.
    """

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
        auto_reconnect: bool = True,
        heartbeat_interval: int = 30,
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
        self.auto_reconnect = auto_reconnect
        self.heartbeat_interval = heartbeat_interval

        self._ib: Optional[Any] = None
        self._contracts_cache: Dict[str, Any] = {}
        self._streaming_subscriptions: Dict[str, Any] = {}
        self._market_data_callbacks: Dict[str, List[Callable]] = {}
        self._order_callbacks: List[Callable] = []
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._error_log: List[Dict[str, Any]] = []

    # ─── SYMBOL PARSING ──────────────────────────────────────────────

    def _parse_symbol(self, symbol: str) -> 'Contract':
        """
        Parse unified symbol format into an IB Contract.

        Stock:    'AAPL' | 'AAPL:stock' | 'AAPL:NASDAQ'
        Forex:    'EUR/USD:forex'
        Crypto:   'BTC/USD:crypto'
        Futures:  'ES:future' | 'ES:future:202412' (with expiry)
        Options:  'AAPL:option:20241220:150:C' (symbol:option:expiry:strike:right)
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
        elif asset_type == 'future':
            expiry = parts[2] if len(parts) > 2 else ''
            contract = Future(ticker, expiry, 'CME', 'USD')
        elif asset_type == 'option':
            if len(parts) < 5:
                raise OrderError(
                    f"Options format: SYMBOL:option:YYYYMMDD:STRIKE:C/P "
                    f"(got '{symbol}')"
                )
            expiry = parts[2]
            strike = float(parts[3])
            right = parts[4].upper()  # 'C' or 'P'
            contract = Option(ticker, expiry, strike, right, 'SMART', '100', 'USD')
        elif asset_type.upper() in ('STOCK', 'SMART', 'NASDAQ', 'NYSE', 'ARCA', 'AMEX', 'BATS', 'IEX'):
            exchange = parts[1] if len(parts) > 1 and asset_type != 'stock' else 'SMART'
            contract = Stock(ticker, exchange, 'USD')
        else:
            contract = Stock(ticker, 'SMART', 'USD')

        self._contracts_cache[symbol] = contract
        return contract

    # ─── CONNECTION MANAGEMENT ────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect to IB Gateway / TWS with error handling and event registration"""
        import time
        start_time = time.time()

        if not IB_INSYNC_AVAILABLE:
            logger.error("ib_insync not installed. Run: pip install ib_insync")
            await self._audit_auth("failure", "ib_insync library not installed")
            return False

        try:
            self._ib = IB()

            # Register event handlers
            self._ib.errorEvent += self._on_error
            self._ib.disconnectedEvent += self._on_disconnect
            self._ib.connectedEvent += self._on_connected

            # Connect (ib_insync connect is synchronous — wrap in executor)
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
            self._reconnect_attempts = 0

            # Get account info
            accounts = self._ib.managedAccounts()
            if not self.account and accounts:
                self.account = accounts[0]

            mode = 'paper' if self.paper else 'LIVE'
            logger.info(
                f"Connected to IBKR ({mode}) at {self.host}:{self.port} "
                f"account={self.account}"
            )

            # Start heartbeat monitor
            if self.heartbeat_interval > 0:
                self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("connect", "IB", "success", duration_ms)
            await self._audit_auth("success")

            return True

        except Exception as e:
            err_msg = str(e)
            logger.error(f"Failed to connect to IBKR: {err_msg}")
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
        # Cancel heartbeat
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        # Cancel streaming subscriptions
        await self.unsubscribe_all()

        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
        self._ib = None
        self._connected = False
        self._contracts_cache.clear()
        logger.info("Disconnected from IBKR")

    def _ensure_connected(self):
        """Raise if not connected"""
        if not self._ib or not self._ib.isConnected():
            raise ConnectionError("Not connected to IBKR. Call connect() first.")

    # ─── EVENT HANDLERS ───────────────────────────────────────────────

    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle IB error/warning events"""
        error_info = IB_ERROR_CODES.get(errorCode)
        entry = {
            'timestamp': datetime.now().isoformat(),
            'reqId': reqId,
            'code': errorCode,
            'message': errorString,
            'contract': str(contract) if contract else None,
            'category': error_info[0] if error_info else 'unknown',
        }
        self._error_log.append(entry)

        # Keep error log bounded
        if len(self._error_log) > 1000:
            self._error_log = self._error_log[-500:]

        # Classify severity
        if errorCode in (1100, 502, 504):
            logger.error(f"IBKR CONNECTION ERROR [{errorCode}]: {errorString}")
        elif errorCode in (1101, 1102):
            logger.info(f"IBKR CONNECTION RESTORED [{errorCode}]: {errorString}")
        elif errorCode in (201, 202, 103, 104, 135):
            logger.warning(f"IBKR ORDER ERROR [{errorCode}]: {errorString}")
        elif errorCode in (354, 10090):
            logger.warning(f"IBKR MARKET DATA [{errorCode}]: {errorString}")
        elif errorCode in (100, 162):
            logger.warning(f"IBKR RATE LIMIT [{errorCode}]: {errorString}")
        elif errorCode >= 2100:
            # 2100+ are warnings, not errors
            logger.debug(f"IBKR warning [{errorCode}]: {errorString}")
        else:
            logger.warning(f"IBKR [{errorCode}]: {errorString}")

    def _on_disconnect(self):
        """Handle disconnection — trigger auto-reconnect if enabled"""
        self._connected = False
        logger.warning("IBKR disconnected")

        if self.auto_reconnect:
            asyncio.ensure_future(self._auto_reconnect())

    def _on_connected(self):
        """Handle successful connection"""
        logger.info("IBKR connected event received")

    # ─── AUTO-RECONNECT ───────────────────────────────────────────────

    async def _auto_reconnect(self):
        """Automatically reconnect with exponential backoff"""
        if self._reconnect_task and not self._reconnect_task.done():
            return  # Already reconnecting

        self._reconnect_task = asyncio.ensure_future(self._reconnect_loop())

    async def _reconnect_loop(self):
        """Reconnection loop with exponential backoff"""
        while self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            wait_time = min(2 ** self._reconnect_attempts, 60)
            logger.info(
                f"IBKR reconnect attempt {self._reconnect_attempts}/"
                f"{self._max_reconnect_attempts} in {wait_time}s..."
            )
            await asyncio.sleep(wait_time)

            try:
                if self._ib:
                    self._ib.disconnect()

                self._ib = IB()
                self._ib.errorEvent += self._on_error
                self._ib.disconnectedEvent += self._on_disconnect
                self._ib.connectedEvent += self._on_connected

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
                self._reconnect_attempts = 0
                logger.info("IBKR reconnected successfully")

                # Resubscribe to any active streaming data
                await self._resubscribe_streaming()
                return

            except Exception as e:
                logger.error(f"IBKR reconnect failed: {e}")

        logger.error(
            f"IBKR reconnect failed after {self._max_reconnect_attempts} attempts. "
            "Manual intervention required."
        )

    async def _resubscribe_streaming(self):
        """Resubscribe to streaming market data after reconnection"""
        symbols = list(self._streaming_subscriptions.keys())
        self._streaming_subscriptions.clear()
        for symbol in symbols:
            callbacks = self._market_data_callbacks.get(symbol, [])
            if callbacks:
                await self.subscribe_market_data(symbol, callbacks[0])

    # ─── HEARTBEAT ────────────────────────────────────────────────────

    async def _heartbeat_loop(self):
        """Periodic heartbeat to detect stale connections"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if self._ib and self._ib.isConnected():
                    # Request server time as heartbeat
                    server_time = self._ib.reqCurrentTime()
                    if server_time:
                        logger.debug(f"IBKR heartbeat OK — server time: {server_time}")
                    else:
                        logger.warning("IBKR heartbeat — no response")
                else:
                    logger.warning("IBKR heartbeat — connection lost")
                    if self.auto_reconnect:
                        await self._auto_reconnect()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"IBKR heartbeat error: {e}")

    # ─── MARKET DATA (SNAPSHOT) ───────────────────────────────────────

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current market data for a symbol"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            contract = self._parse_symbol(symbol)
            self._ib.qualifyContracts(contract)

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
            logger.error(f"Failed to fetch IBKR ticker for {symbol}: {e}")
            raise ExchangeError(str(e))

    # ─── STREAMING MARKET DATA ────────────────────────────────────────

    async def subscribe_market_data(
        self, symbol: str, callback: Callable[[str, Ticker], Any]
    ) -> bool:
        """
        Subscribe to real-time streaming market data.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'EUR/USD:forex')
            callback: async function(symbol, ticker) called on each update
        """
        self._ensure_connected()

        try:
            contract = self._parse_symbol(symbol)
            self._ib.qualifyContracts(contract)

            ticker_data = self._ib.reqMktData(contract, '', snapshot=False)

            # Register callback
            if symbol not in self._market_data_callbacks:
                self._market_data_callbacks[symbol] = []
            self._market_data_callbacks[symbol].append(callback)

            # Set up the update handler
            def on_ticker_update(ticker_obj):
                bid = float(ticker_obj.bid) if ticker_obj.bid and ticker_obj.bid > 0 else 0.0
                ask = float(ticker_obj.ask) if ticker_obj.ask and ticker_obj.ask > 0 else 0.0
                last = float(ticker_obj.last) if ticker_obj.last and ticker_obj.last > 0 else 0.0
                volume = float(ticker_obj.volume) if ticker_obj.volume and ticker_obj.volume > 0 else 0.0

                tick = Ticker(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    last=last,
                    volume_24h=volume,
                    timestamp=datetime.now(),
                )

                for cb in self._market_data_callbacks.get(symbol, []):
                    try:
                        result = cb(symbol, tick)
                        if asyncio.iscoroutine(result):
                            asyncio.ensure_future(result)
                    except Exception as e:
                        logger.error(f"Market data callback error for {symbol}: {e}")

            ticker_data.updateEvent += on_ticker_update
            self._streaming_subscriptions[symbol] = (contract, ticker_data)

            logger.info(f"Subscribed to IBKR streaming data for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to IBKR streaming for {symbol}: {e}")
            return False

    async def unsubscribe_market_data(self, symbol: str) -> bool:
        """Unsubscribe from streaming market data for a symbol"""
        if symbol in self._streaming_subscriptions:
            contract, ticker_data = self._streaming_subscriptions.pop(symbol)
            try:
                self._ib.cancelMktData(contract)
            except Exception:
                pass
            self._market_data_callbacks.pop(symbol, None)
            logger.info(f"Unsubscribed from IBKR streaming for {symbol}")
            return True
        return False

    async def unsubscribe_all(self):
        """Unsubscribe from all streaming data"""
        for symbol in list(self._streaming_subscriptions.keys()):
            await self.unsubscribe_market_data(symbol)

    # ─── ORDER BOOK ───────────────────────────────────────────────────

    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get Level II market depth"""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            contract = self._parse_symbol(symbol)
            self._ib.qualifyContracts(contract)

            depth_data = self._ib.reqMktDepth(contract, numRows=limit)
            await asyncio.sleep(2)

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
            logger.error(f"Failed to fetch IBKR orderbook for {symbol}: {e}")
            raise ExchangeError(str(e))

    # ─── ACCOUNT DATA ────────────────────────────────────────────────

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
            logger.error(f"Failed to fetch IBKR balances: {e}")
            raise ExchangeError(str(e))

    # ─── ORDER MANAGEMENT ─────────────────────────────────────────────

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
                raise OrderError(f"Unsupported order type: {order_type}. Use 'market' or 'limit'.")

            if client_order_id:
                ib_order.orderRef = client_order_id

            trade = self._ib.placeOrder(contract, ib_order)
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

            # Notify order callbacks
            for cb in self._order_callbacks:
                try:
                    cb_result = cb('created', result)
                    if asyncio.iscoroutine(cb_result):
                        await cb_result
                except Exception as e:
                    logger.error(f"Order callback error: {e}")

            return result

        except OrderError:
            raise
        except Exception as e:
            logger.error(f"Failed to create IBKR order: {e}")
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

            logger.warning(f"IBKR order {order_id} not found in open trades")
            return False

        except Exception as e:
            logger.error(f"Failed to cancel IBKR order {order_id}: {e}")
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
            logger.error(f"Failed to get IBKR order {order_id}: {e}")
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
                    symbol=f"{contract_sym}:{trade.contract.secType.lower()}",
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
            logger.error(f"Failed to fetch IBKR open orders: {e}")
            raise ExchangeError(str(e))

    async def get_trade_fee(self, symbol: str) -> Dict[str, float]:
        """Get estimated trading fees (IBKR tiered pricing defaults)"""
        return {'maker': 0.0005, 'taker': 0.0005}

    # ─── ADVANCED ORDER TYPES ─────────────────────────────────────────

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
            logger.error(f"Failed to create IBKR stop order: {e}")
            raise ExchangeError(str(e))

    async def create_trailing_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        trail_amount: Optional[float] = None,
        trail_percent: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """Create a trailing stop order"""
        self._ensure_connected()
        await self._rate_limit_wait()

        if trail_amount is None and trail_percent is None:
            raise OrderError("Either trail_amount or trail_percent is required")

        try:
            contract = self._parse_symbol(symbol)
            self._ib.qualifyContracts(contract)

            action = 'BUY' if side.lower() == 'buy' else 'SELL'
            ib_order = TrailingStopOrder(action, quantity, trailingPercent=trail_percent, auxPrice=trail_amount)

            if client_order_id:
                ib_order.orderRef = client_order_id

            trade = self._ib.placeOrder(contract, ib_order)
            await asyncio.sleep(0.5)

            return ExchangeOrder(
                order_id=str(trade.order.orderId),
                client_order_id=client_order_id,
                symbol=symbol,
                side=side.lower(),
                order_type='trailing_stop',
                quantity=quantity,
                price=trail_amount or 0,
                status=self._map_ib_status(trade.orderStatus.status),
                filled_quantity=float(trade.orderStatus.filled),
                average_price=float(trade.orderStatus.avgFillPrice or 0),
                fee=float(trade.orderStatus.commission or 0),
            )

        except OrderError:
            raise
        except Exception as e:
            logger.error(f"Failed to create IBKR trailing stop: {e}")
            raise ExchangeError(str(e))

    async def create_bracket_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        take_profit_price: float,
        stop_loss_price: float,
        client_order_id: Optional[str] = None,
    ) -> List[ExchangeOrder]:
        """
        Create a bracket order (entry + take profit + stop loss).
        All three orders are linked via OCA (One Cancels All).
        """
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            contract = self._parse_symbol(symbol)
            self._ib.qualifyContracts(contract)

            action = 'BUY' if side.lower() == 'buy' else 'SELL'
            reverse_action = 'SELL' if action == 'BUY' else 'BUY'

            bracket = self._ib.bracketOrder(
                action, quantity, entry_price,
                take_profit_price, stop_loss_price,
            )

            results = []
            for ib_order in bracket:
                if client_order_id:
                    ib_order.orderRef = client_order_id

                trade = self._ib.placeOrder(contract, ib_order)
                await asyncio.sleep(0.3)

                results.append(ExchangeOrder(
                    order_id=str(trade.order.orderId),
                    client_order_id=client_order_id,
                    symbol=symbol,
                    side=side.lower() if trade.order.action == action else ('sell' if side.lower() == 'buy' else 'buy'),
                    order_type=trade.order.orderType.lower(),
                    quantity=quantity,
                    price=float(trade.order.lmtPrice or trade.order.auxPrice or 0),
                    status=self._map_ib_status(trade.orderStatus.status),
                    filled_quantity=float(trade.orderStatus.filled),
                    average_price=float(trade.orderStatus.avgFillPrice or 0),
                    fee=float(trade.orderStatus.commission or 0),
                ))

            logger.info(
                f"IBKR bracket order created: {len(results)} legs for {symbol} "
                f"entry={entry_price} TP={take_profit_price} SL={stop_loss_price}"
            )
            return results

        except Exception as e:
            logger.error(f"Failed to create IBKR bracket order: {e}")
            raise ExchangeError(str(e))

    # ─── ORDER EVENT SUBSCRIPTION ─────────────────────────────────────

    def on_order_update(self, callback: Callable):
        """Register callback for order status updates"""
        self._order_callbacks.append(callback)

    # ─── IBKR-SPECIFIC: PORTFOLIO & ACCOUNT ───────────────────────────

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
            logger.error(f"Failed to get IBKR portfolio: {e}")
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
            logger.error(f"Failed to get IBKR account summary: {e}")
            raise ExchangeError(str(e))

    # ─── IBKR-SPECIFIC: HISTORICAL DATA ──────────────────────────────

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
            logger.error(f"Failed to get IBKR historical data for {symbol}: {e}")
            raise ExchangeError(str(e))

    # ─── IBKR-SPECIFIC: OPTIONS CHAIN ─────────────────────────────────

    async def get_option_chain(self, symbol: str) -> Dict[str, Any]:
        """
        Get the full options chain for a stock symbol.
        Returns available expirations and strikes.
        """
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            contract = Stock(symbol.split(':')[0], 'SMART', 'USD')
            self._ib.qualifyContracts(contract)

            chains = self._ib.reqSecDefOptParams(
                contract.symbol, '', contract.secType, contract.conId
            )

            result = {
                'symbol': symbol,
                'exchanges': [],
            }

            for chain in chains:
                result['exchanges'].append({
                    'exchange': chain.exchange,
                    'underlying_conId': chain.underlyingConId,
                    'trading_class': chain.tradingClass,
                    'multiplier': chain.multiplier,
                    'expirations': sorted(chain.expirations),
                    'strikes': sorted(chain.strikes),
                })

            return result

        except Exception as e:
            logger.error(f"Failed to get IBKR option chain for {symbol}: {e}")
            raise ExchangeError(str(e))

    async def get_option_greeks(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
    ) -> Dict[str, Any]:
        """
        Get Greeks for a specific option contract.

        Args:
            symbol: Underlying stock symbol (e.g., 'AAPL')
            expiry: Expiration date (YYYYMMDD)
            strike: Strike price
            right: 'C' for call, 'P' for put
        """
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            option_symbol = f"{symbol}:option:{expiry}:{strike}:{right}"
            contract = self._parse_symbol(option_symbol)
            self._ib.qualifyContracts(contract)

            ticker_data = self._ib.reqMktData(contract, '106', snapshot=True)
            # Wait for Greeks
            for _ in range(100):
                if ticker_data.modelGreeks is not None:
                    break
                await asyncio.sleep(0.1)

            self._ib.cancelMktData(contract)

            greeks = ticker_data.modelGreeks
            if greeks:
                return {
                    'symbol': option_symbol,
                    'delta': greeks.delta,
                    'gamma': greeks.gamma,
                    'theta': greeks.theta,
                    'vega': greeks.vega,
                    'implied_volatility': greeks.impliedVol,
                    'underlying_price': greeks.undPrice,
                    'option_price': greeks.optPrice,
                    'pv_dividend': greeks.pvDividend,
                }
            else:
                return {
                    'symbol': option_symbol,
                    'error': 'Greeks not available (market data subscription may be required)',
                }

        except Exception as e:
            logger.error(f"Failed to get IBKR option Greeks: {e}")
            raise ExchangeError(str(e))

    # ─── DIAGNOSTICS ──────────────────────────────────────────────────

    def get_error_log(self, last_n: int = 50) -> List[Dict[str, Any]]:
        """Get recent IB error/warning log entries"""
        return self._error_log[-last_n:]

    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status"""
        return {
            'connected': self._connected,
            'ib_connected': self._ib.isConnected() if self._ib else False,
            'host': self.host,
            'port': self.port,
            'client_id': self.client_id,
            'account': self.account,
            'paper': self.paper,
            'auto_reconnect': self.auto_reconnect,
            'reconnect_attempts': self._reconnect_attempts,
            'streaming_subscriptions': list(self._streaming_subscriptions.keys()),
            'cached_contracts': len(self._contracts_cache),
            'recent_errors': len(self._error_log),
        }

    # ─── HELPERS ──────────────────────────────────────────────────────

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
