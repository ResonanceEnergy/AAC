#!/usr/bin/env python3
"""
Interactive Brokers (IBKR) Exchange Connector
==============================================
Implementation of the exchange connector for Interactive Brokers
via the TWS API using ib_insync.

Requirements:
    pip install ib_insync

Connection:
    Requires TWS or IB Gateway running locally (or on a reachable host).
    - TWS Paper Trading: port 7497
    - TWS Live Trading: port 7496
    - IB Gateway Paper: port 4002
    - IB Gateway Live: port 4001

Configuration via .env:
    IBKR_HOST=127.0.0.1
    IBKR_PORT=7497
    IBKR_CLIENT_ID=1
    IBKR_ACCOUNT=DU1234567
    IBKR_PAPER=true
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

# Add project root to path
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

# Try to import ib_insync
try:
    from ib_insync import (
        IB, Stock, Forex, Contract, Order as IBOrder,
        LimitOrder, MarketOrder, StopOrder, StopLimitOrder,
        Trade, util,
    )
    IB_INSYNC_AVAILABLE = True
except (ImportError, RuntimeError):
    IB_INSYNC_AVAILABLE = False
    IB = None
    Stock = Forex = Crypto = Future = Option = Contract = None
    MarketOrder = LimitOrder = StopOrder = StopLimitOrder = TrailingStopOrder = None


# Symbol mapping: AAC format -> IBKR contract
# AAC uses "AAPL/USD", IBKR uses Stock("AAPL", "SMART", "USD")
def _parse_symbol(symbol: str) -> dict:
    """
    Parse AAC symbol format to IBKR contract parameters.

    Supported formats:
        'AAPL/USD'      -> Stock AAPL on SMART, USD
        'EUR/USD'       -> Forex EURUSD
        'BTC/USD'       -> Crypto BTCUSD (via PAXOS)
        'AAPL'          -> Stock AAPL on SMART, USD (default)
    """
    parts = symbol.split('/')
    base = parts[0].strip().upper()
    quote = parts[1].strip().upper() if len(parts) > 1 else 'USD'

    # Known forex pairs
    forex_bases = {
        'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD',
        'SEK', 'NOK', 'DKK', 'HKD', 'SGD', 'MXN', 'ZAR',
    }

    # Known crypto
    crypto_bases = {'BTC', 'ETH', 'LTC', 'BCH', 'XRP', 'SOL', 'ADA', 'DOT', 'AVAX'}

    if base in forex_bases or quote in forex_bases:
        return {'type': 'forex', 'pair': base + quote, 'base': base, 'quote': quote}
    elif base in crypto_bases:
        return {'type': 'crypto', 'symbol': base, 'currency': quote}
    else:
        return {'type': 'stock', 'symbol': base, 'currency': quote}


def _make_contract(parsed: dict):
    """Create an ib_insync Contract from parsed symbol info."""
    if not IB_INSYNC_AVAILABLE:
        raise ExchangeError("ib_insync not installed. Run: pip install ib_insync")

    if parsed['type'] == 'forex':
        return Forex(parsed['pair'])
    elif parsed['type'] == 'crypto':
        contract = Contract()
        contract.symbol = parsed['symbol']
        contract.secType = 'CRYPTO'
        contract.exchange = 'PAXOS'
        contract.currency = parsed['currency']
        return contract
    else:
        return Stock(parsed['symbol'], 'SMART', parsed['currency'])


class IBKRConnector(BaseExchangeConnector):
    """
    Interactive Brokers connector using ib_insync.

    Connects to TWS or IB Gateway via socket. Supports stocks, forex,
    and crypto through IBKR's unified API.

    Your $1,000 balance is accessible through this connector.
    """

    @property
    def name(self) -> str:
        return "ibkr"

    def __init__(
        self,
        host: str = '',
        port: int = 0,
        client_id: int = 0,
        account: str = '',
        paper: bool = True,
        rate_limit: int = 50,  # IBKR allows ~50 req/sec
        timeout: int = 20,
    ):
        # IBKR doesn't use api_key/secret — uses socket connection
        super().__init__(
            api_key='',
            api_secret='',
            testnet=paper,
            rate_limit=rate_limit,
        )

        # Load from env if not provided
        self.host = host or get_env('IBKR_HOST', '127.0.0.1')
        self.port = port or get_env_int('IBKR_PORT', 7497)
        self.client_id = client_id or get_env_int('IBKR_CLIENT_ID', 1)
        self.account = account or get_env('IBKR_ACCOUNT', '')
        self.paper = paper if paper is not True else get_env_bool('IBKR_PAPER', True)
        self.timeout = timeout

        self._ib: Optional[Any] = None  # IB instance
        self._account_values: Dict[str, Any] = {}

    async def connect(self) -> bool:
        """Connect to TWS or IB Gateway."""
        start_time = time.time()

        if not IB_INSYNC_AVAILABLE:
            self.logger.error("ib_insync not installed. Run: pip install ib_insync")
            return False

        try:
            self._ib = IB()
            
            # Use async connect to avoid event loop conflicts
            await self._ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout,
                readonly=False,
            )

            # Verify connection
            if not self._ib.isConnected():
                raise ConnectionError("Failed to establish connection to TWS/Gateway")

            # Get managed accounts
            accounts = self._ib.managedAccounts()
            if not accounts:
                raise AuthenticationError("No managed accounts found")

            # Use specified account or first available
            if self.account and self.account in accounts:
                pass  # keep self.account
            elif self.account and self.account not in accounts:
                self.logger.warning(
                    f"Account {self.account} not found. Available: {accounts}. "
                    f"Using {accounts[0]}"
                )
                self.account = accounts[0]
            else:
                self.account = accounts[0]

            self._connected = True
            mode = "PAPER" if self.paper else "LIVE"
            duration_ms = (time.time() - start_time) * 1000

            self.logger.info(
                f"Connected to IBKR {mode} — account {self.account} "
                f"via {self.host}:{self.port} ({duration_ms:.0f}ms)"
            )

            await self._audit_api_call("connect", "SOCKET", "success", duration_ms)

            return True

        except ConnectionError:
            raise
        except AuthenticationError:
            raise
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"IBKR connection failed: {e}")
            await self._audit_api_call(
                "connect", "SOCKET", "failure", duration_ms, error_message=str(e)
            )
            raise ConnectionError(
                f"Cannot connect to TWS/Gateway at {self.host}:{self.port}: {e}. "
                f"Ensure TWS or IB Gateway is running."
            )

    async def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            self.logger.info("Disconnected from IBKR")
        self._ib = None
        self._connected = False

    def _ensure_connected(self):
        """Raise if not connected."""
        if not self._ib or not self._ib.isConnected():
            raise ConnectionError("Not connected to IBKR. Call connect() first.")

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current market data for a symbol."""
        self._ensure_connected()
        await self._rate_limit_wait()
        start_time = time.time()

        try:
            parsed = _parse_symbol(symbol)
            contract = _make_contract(parsed)

            # Qualify the contract (resolve to specific exchange contract)
            qualified = self._ib.qualifyContracts(contract)
            if not qualified:
                raise ExchangeError(f"Could not qualify contract for {symbol}")
            contract = qualified[0]

            # Request market data snapshot
            ticker_data = self._ib.reqMktData(contract, '', True, False)

            # Wait for data to arrive (snapshot mode)
            await asyncio.sleep(2)

            bid = ticker_data.bid if ticker_data.bid and ticker_data.bid > 0 else 0.0
            ask = ticker_data.ask if ticker_data.ask and ticker_data.ask > 0 else 0.0
            last = ticker_data.last if ticker_data.last and ticker_data.last > 0 else 0.0
            volume = ticker_data.volume if ticker_data.volume and ticker_data.volume > 0 else 0.0

            # Cancel market data subscription
            self._ib.cancelMktData(contract)

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("get_ticker", "GET", "success", duration_ms)

            return Ticker(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last,
                volume_24h=volume,
                timestamp=datetime.now(),
            )

        except ExchangeError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise ExchangeError(f"Ticker fetch failed for {symbol}: {e}")

    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get order book (market depth) for a symbol."""
        self._ensure_connected()
        await self._rate_limit_wait()
        start_time = time.time()

        try:
            parsed = _parse_symbol(symbol)
            contract = _make_contract(parsed)
            qualified = self._ib.qualifyContracts(contract)
            if not qualified:
                raise ExchangeError(f"Could not qualify contract for {symbol}")
            contract = qualified[0]

            # Request market depth
            depth_data = self._ib.reqMktDepth(contract, numRows=limit)
            await asyncio.sleep(2)  # wait for depth data

            # Collect depth entries
            bids = []
            asks = []
            for entry in self._ib.ticker(contract).domBids or []:
                if entry.price > 0:
                    bids.append((entry.price, entry.size))
            for entry in self._ib.ticker(contract).domAsks or []:
                if entry.price > 0:
                    asks.append((entry.price, entry.size))

            # Sort: bids descending, asks ascending
            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])

            # Cancel depth subscription
            self._ib.cancelMktDepth(contract)

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("get_orderbook", "GET", "success", duration_ms)

            return OrderBook(
                symbol=symbol,
                bids=bids[:limit],
                asks=asks[:limit],
                timestamp=datetime.now(),
            )

        except ExchangeError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get order book for {symbol}: {e}")
            raise ExchangeError(f"Order book fetch failed for {symbol}: {e}")

    async def get_balances(self) -> Dict[str, Balance]:
        """
        Get account balances from IBKR.

        Returns all asset balances including cash and positions.
        """
        self._ensure_connected()
        await self._rate_limit_wait()
        start_time = time.time()

        try:
            # Request account summary
            account_values = self._ib.accountValues(self.account)

            result = {}

            for av in account_values:
                if av.tag == 'CashBalance' and av.currency != 'BASE':
                    # Cash balances per currency
                    currency = av.currency
                    amount = float(av.value)
                    if amount != 0:
                        result[currency] = Balance(
                            asset=currency,
                            free=amount,
                            locked=0.0,
                        )
                elif av.tag == 'TotalCashBalance' and av.currency == 'BASE':
                    # Total cash in base currency
                    result['TOTAL_CASH'] = Balance(
                        asset='TOTAL_CASH',
                        free=float(av.value),
                        locked=0.0,
                    )
                elif av.tag == 'NetLiquidation' and av.currency == 'BASE':
                    result['NET_LIQUIDATION'] = Balance(
                        asset='NET_LIQUIDATION',
                        free=float(av.value),
                        locked=0.0,
                    )

            # Get portfolio positions
            positions = self._ib.positions(self.account)
            for pos in positions:
                symbol = pos.contract.symbol
                qty = pos.position
                market_value = pos.marketValue or 0.0
                if qty != 0:
                    result[symbol] = Balance(
                        asset=symbol,
                        free=abs(qty),
                        locked=0.0,
                    )

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("get_balances", "GET", "success", duration_ms)

            return result

        except Exception as e:
            self.logger.error(f"Failed to get balances: {e}")
            raise ExchangeError(f"Balance fetch failed: {e}")

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """
        Create a new order on IBKR.

        Args:
            symbol: e.g. 'AAPL/USD', 'EUR/USD', 'BTC/USD'
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            quantity: Number of shares/units
            price: Limit price (required for limit orders)
            client_order_id: Optional custom reference
        """
        self._ensure_connected()
        await self._rate_limit_wait()
        start_time = time.time()

        try:
            parsed = _parse_symbol(symbol)
            contract = _make_contract(parsed)
            qualified = self._ib.qualifyContracts(contract)
            if not qualified:
                raise OrderError(f"Could not qualify contract for {symbol}")
            contract = qualified[0]

            # Build IBKR order
            action = 'BUY' if side.lower() == 'buy' else 'SELL'

            if order_type.lower() == 'market':
                ib_order = MarketOrder(action, quantity)
            elif order_type.lower() == 'limit':
                if price is None:
                    raise OrderError("Price required for limit orders")
                ib_order = LimitOrder(action, quantity, price)
            else:
                raise OrderError(f"Unsupported order type: {order_type}")

            # Set account
            ib_order.account = self.account

            # Place the order
            trade: Trade = self._ib.placeOrder(contract, ib_order)

            # Wait briefly for order acknowledgment
            await asyncio.sleep(0.5)

            result = self._parse_trade(trade, symbol)

            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_order", "POST", "success", duration_ms)
            await self._audit_order(
                symbol, side, order_type, quantity, price,
                result.order_id, "created"
            )

            self.logger.info(
                f"Order placed: {side} {quantity} {symbol} "
                f"@ {'MKT' if order_type == 'market' else price} "
                f"-> order_id={result.order_id}"
            )

            return result

        except (OrderError, ExchangeError):
            raise
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Failed to create order: {e}")
            await self._audit_api_call(
                "create_order", "POST", "failure", duration_ms, error_message=str(e)
            )
            raise ExchangeError(f"Order creation failed: {e}")

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order by order ID."""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            # Find the trade by order ID
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == str(order_id):
                    self._ib.cancelOrder(trade.order)
                    await asyncio.sleep(0.5)
                    self.logger.info(f"Order {order_id} cancelled")
                    return True

            self.logger.warning(f"Order {order_id} not found in open trades")
            return False

        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            raise ExchangeError(f"Cancel failed for order {order_id}: {e}")

    async def get_order(self, order_id: str, symbol: str) -> ExchangeOrder:
        """Get details of a specific order."""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            # Search open trades
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == str(order_id):
                    return self._parse_trade(trade, symbol)

            # Search completed trades
            for trade in self._ib.trades():
                if str(trade.order.orderId) == str(order_id):
                    return self._parse_trade(trade, symbol)

            raise OrderError(f"Order {order_id} not found")

        except OrderError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to get order {order_id}: {e}")
            raise ExchangeError(f"Order lookup failed for {order_id}: {e}")

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[ExchangeOrder]:
        """Get all open orders, optionally filtered by symbol."""
        self._ensure_connected()
        await self._rate_limit_wait()

        try:
            trades = self._ib.openTrades()
            result = []

            for trade in trades:
                trade_symbol = trade.contract.symbol
                if symbol:
                    parsed = _parse_symbol(symbol)
                    if parsed['type'] == 'forex':
                        target = parsed['base']
                    else:
                        target = parsed.get('symbol', '')
                    if trade_symbol != target:
                        continue

                result.append(self._parse_trade(trade, f"{trade_symbol}/USD"))

            return result

        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            raise ExchangeError(f"Open orders fetch failed: {e}")

    async def get_trade_fee(self, symbol: str) -> Dict[str, float]:
        """
        IBKR fee structure.
        
        Actual fees depend on account type, volume tier, etc.
        These are approximate for IBKR Pro tiered pricing (stocks).
        """
        parsed = _parse_symbol(symbol)
        if parsed['type'] == 'stock':
            # IBKR Pro: ~$0.005/share, min $1.00
            return {'maker': 0.0001, 'taker': 0.0001}
        elif parsed['type'] == 'forex':
            # Forex: ~0.2 basis points
            return {'maker': 0.00002, 'taker': 0.00002}
        elif parsed['type'] == 'crypto':
            # Crypto: 0.12%-0.18%
            return {'maker': 0.0012, 'taker': 0.0018}
        return {'maker': 0.001, 'taker': 0.001}

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """Create a stop or stop-limit order on IBKR."""
        self._ensure_connected()
        await self._rate_limit_wait()
        start_time = time.time()

        try:
            parsed = _parse_symbol(symbol)
            contract = _make_contract(parsed)
            qualified = self._ib.qualifyContracts(contract)
            if not qualified:
                raise OrderError(f"Could not qualify contract for {symbol}")
            contract = qualified[0]

            action = 'BUY' if side.lower() == 'buy' else 'SELL'

            if limit_price:
                ib_order = StopLimitOrder(action, quantity, limit_price, stop_price)
            else:
                ib_order = StopOrder(action, quantity, stop_price)

            ib_order.account = self.account
            trade = self._ib.placeOrder(contract, ib_order)
            await asyncio.sleep(0.5)

            result = self._parse_trade(trade, symbol)
            duration_ms = (time.time() - start_time) * 1000
            await self._audit_api_call("create_stop_loss", "POST", "success", duration_ms)

            self.logger.info(
                f"Stop-loss order placed: {side} {quantity} {symbol} "
                f"stop@{stop_price} -> order_id={result.order_id}"
            )

            return result

        except (OrderError, ExchangeError):
            raise
        except Exception as e:
            self.logger.error(f"Failed to create stop-loss: {e}")
            raise ExchangeError(f"Stop-loss creation failed: {e}")

    async def get_account_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive account summary.
        
        Returns key account metrics including net liquidation value,
        buying power, cash balances, margin requirements, etc.
        """
        self._ensure_connected()

        try:
            account_values = self._ib.accountValues(self.account)
            summary = {}

            important_tags = {
                'NetLiquidation', 'TotalCashValue', 'SettledCash',
                'BuyingPower', 'AvailableFunds', 'ExcessLiquidity',
                'GrossPositionValue', 'MaintMarginReq', 'InitMarginReq',
                'UnrealizedPnL', 'RealizedPnL',
            }

            for av in account_values:
                if av.tag in important_tags and av.currency in ('BASE', 'USD', 'CAD'):
                    key = f"{av.tag}_{av.currency}"
                    summary[key] = float(av.value)

            return summary

        except Exception as e:
            self.logger.error(f"Failed to get account summary: {e}")
            raise ExchangeError(f"Account summary failed: {e}")

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions with P&L."""
        self._ensure_connected()

        try:
            positions = self._ib.positions(self.account)
            result = []

            for pos in positions:
                result.append({
                    'symbol': pos.contract.symbol,
                    'sec_type': pos.contract.secType,
                    'exchange': pos.contract.exchange,
                    'currency': pos.contract.currency,
                    'quantity': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_value': pos.marketValue or 0.0,
                })

            return result

        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            raise ExchangeError(f"Positions fetch failed: {e}")

    def _parse_trade(self, trade, symbol: str) -> ExchangeOrder:
        """Convert an ib_insync Trade to ExchangeOrder."""
        order = trade.order
        status = trade.orderStatus

        # Map IBKR status to our status
        status_map = {
            'PendingSubmit': 'open',
            'PendingCancel': 'cancelling',
            'PreSubmitted': 'open',
            'Submitted': 'open',
            'ApiPending': 'open',
            'ApiCancelled': 'cancelled',
            'Cancelled': 'cancelled',
            'Filled': 'filled',
            'Inactive': 'cancelled',
        }

        ibkr_status = status.status if status else 'unknown'
        mapped_status = status_map.get(ibkr_status, 'unknown')

        # Map IBKR order type
        type_map = {
            'MKT': 'market',
            'LMT': 'limit',
            'STP': 'stop',
            'STP LMT': 'stop_limit',
        }

        return ExchangeOrder(
            order_id=str(order.orderId),
            client_order_id=str(order.clientId) if order.clientId else None,
            symbol=symbol,
            side=order.action.lower(),
            order_type=type_map.get(order.orderType, order.orderType),
            quantity=order.totalQuantity,
            price=order.lmtPrice if order.lmtPrice else None,
            status=mapped_status,
            filled_quantity=status.filled if status else 0.0,
            average_price=status.avgFillPrice if status else 0.0,
            fee=float(status.commission) if status and status.commission else 0.0,
            fee_currency='USD',
            created_at=datetime.now(),
            updated_at=datetime.now(),
            raw={
                'ibkr_status': ibkr_status,
                'perm_id': order.permId,
                'account': order.account,
            },
        )
