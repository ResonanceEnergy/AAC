#!/usr/bin/env python3
"""
Tests for the IBKR Exchange Connector
======================================
Unit tests with mocked ib_insync to validate connector behavior
without requiring a live IB Gateway / TWS connection.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from datetime import datetime
from types import SimpleNamespace
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_ib():
    """Create a mocked IB instance"""
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.managedAccounts.return_value = ['DUP782762']
    ib.reqCurrentTime.return_value = datetime.now()
    ib.accountValues.return_value = []
    ib.positions.return_value = []
    ib.portfolio.return_value = []
    ib.accountSummary.return_value = []
    ib.openTrades.return_value = []
    ib.trades.return_value = []

    # Event attributes
    ib.errorEvent = MagicMock()
    ib.errorEvent.__iadd__ = MagicMock(return_value=ib.errorEvent)
    ib.disconnectedEvent = MagicMock()
    ib.disconnectedEvent.__iadd__ = MagicMock(return_value=ib.disconnectedEvent)
    ib.connectedEvent = MagicMock()
    ib.connectedEvent.__iadd__ = MagicMock(return_value=ib.connectedEvent)

    return ib


@pytest.fixture
def connector(mock_ib):
    """Create an IBKRConnector with mocked IB connection"""
    with patch('TradingExecution.exchange_connectors.ibkr_connector.IB_INSYNC_AVAILABLE', True), \
         patch('TradingExecution.exchange_connectors.ibkr_connector.Stock') as mock_stock, \
         patch('TradingExecution.exchange_connectors.ibkr_connector.Forex') as mock_forex, \
         patch('TradingExecution.exchange_connectors.ibkr_connector.Crypto') as mock_crypto, \
         patch('TradingExecution.exchange_connectors.ibkr_connector.Future') as mock_future, \
         patch('TradingExecution.exchange_connectors.ibkr_connector.Option') as mock_option, \
         patch('TradingExecution.exchange_connectors.ibkr_connector.MarketOrder') as mock_mkt_order, \
         patch('TradingExecution.exchange_connectors.ibkr_connector.LimitOrder') as mock_lmt_order, \
         patch('TradingExecution.exchange_connectors.ibkr_connector.StopOrder') as mock_stp_order, \
         patch('TradingExecution.exchange_connectors.ibkr_connector.StopLimitOrder') as mock_stplmt_order, \
         patch('TradingExecution.exchange_connectors.ibkr_connector.TrailingStopOrder') as mock_trail_order:

        from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector

        conn = IBKRConnector(
            host='127.0.0.1',
            port=7497,
            client_id=1,
            account='DUP782762',
            paper=True,
        )
        conn._ib = mock_ib
        conn._connected = True
        conn.auto_reconnect = False  # Disable for tests

        # Store mock contract classes for assertions
        conn._mock_stock = mock_stock
        conn._mock_forex = mock_forex
        conn._mock_crypto = mock_crypto
        conn._mock_future = mock_future
        conn._mock_option = mock_option

        yield conn


# ─── BASIC PROPERTIES ─────────────────────────────────────────────────────────

class TestIBKRBasics:
    def test_name(self, connector):
        assert connector.name == "ibkr"

    def test_initial_state(self, connector):
        assert connector.host == '127.0.0.1'
        assert connector.port == 7497
        assert connector.client_id == 1
        assert connector.account == 'DUP782762'
        assert connector.paper is True


# ─── SYMBOL PARSING ──────────────────────────────────────────────────────────

class TestSymbolParsing:
    def test_stock_plain(self, connector):
        connector._parse_symbol('AAPL')
        connector._mock_stock.assert_called_with('AAPL', 'SMART', 'USD')

    def test_stock_explicit(self, connector):
        connector._parse_symbol('AAPL:stock')
        connector._mock_stock.assert_called_with('AAPL', 'SMART', 'USD')

    def test_stock_exchange(self, connector):
        connector._parse_symbol('AAPL:NASDAQ')
        connector._mock_stock.assert_called_with('AAPL', 'NASDAQ', 'USD')

    def test_forex(self, connector):
        connector._parse_symbol('EUR/USD:forex')
        connector._mock_forex.assert_called_with('EURUSD')

    def test_crypto(self, connector):
        connector._parse_symbol('BTC/USD:crypto')
        connector._mock_crypto.assert_called_with('BTC', 'PAXOS', 'USD')

    def test_future(self, connector):
        connector._parse_symbol('ES:future:202412')
        connector._mock_future.assert_called_with('ES', '202412', 'CME', 'USD')

    def test_future_no_expiry(self, connector):
        connector._parse_symbol('ES:future')
        connector._mock_future.assert_called_with('ES', '', 'CME', 'USD')

    def test_option(self, connector):
        connector._parse_symbol('AAPL:option:20241220:150:C')
        connector._mock_option.assert_called_with(
            'AAPL', '20241220', 150.0, 'C', 'SMART', '100', 'USD'
        )

    def test_option_put(self, connector):
        connector._parse_symbol('SPY:option:20250117:450.5:P')
        connector._mock_option.assert_called_with(
            'SPY', '20250117', 450.5, 'P', 'SMART', '100', 'USD'
        )

    def test_option_missing_fields_raises(self, connector):
        from TradingExecution.exchange_connectors.base_connector import OrderError
        with pytest.raises(OrderError, match="Options format"):
            connector._parse_symbol('AAPL:option:20241220')

    def test_cache_hit(self, connector):
        """Second call for same symbol should use cache"""
        connector._parse_symbol('MSFT')
        call_count_1 = connector._mock_stock.call_count
        connector._parse_symbol('MSFT')
        assert connector._mock_stock.call_count == call_count_1  # No new call

    def test_unknown_type_defaults_to_stock(self, connector):
        connector._parse_symbol('XYZ:unknown')
        connector._mock_stock.assert_called_with('XYZ', 'SMART', 'USD')


# ─── CONNECTION ──────────────────────────────────────────────────────────────

class TestConnection:
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_ib):
        """Test successful connection"""
        with patch('TradingExecution.exchange_connectors.ibkr_connector.IB_INSYNC_AVAILABLE', True), \
             patch('TradingExecution.exchange_connectors.ibkr_connector.IB', return_value=mock_ib), \
             patch('TradingExecution.exchange_connectors.ibkr_connector.Stock'), \
             patch('TradingExecution.exchange_connectors.ibkr_connector.Forex'), \
             patch('TradingExecution.exchange_connectors.ibkr_connector.Crypto'), \
             patch('TradingExecution.exchange_connectors.ibkr_connector.Future'), \
             patch('TradingExecution.exchange_connectors.ibkr_connector.Option'):

            from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
            conn = IBKRConnector(account='DUP782762', paper=True)
            conn.auto_reconnect = False
            conn.heartbeat_interval = 0

            result = await conn.connect()

            assert result is True
            assert conn._connected is True
            assert conn.account == 'DUP782762'

    @pytest.mark.asyncio
    async def test_connect_no_library(self):
        """Test connect fails gracefully when ib_insync not installed"""
        with patch('TradingExecution.exchange_connectors.ibkr_connector.IB_INSYNC_AVAILABLE', False), \
             patch('TradingExecution.exchange_connectors.ibkr_connector.Stock', None), \
             patch('TradingExecution.exchange_connectors.ibkr_connector.Forex', None), \
             patch('TradingExecution.exchange_connectors.ibkr_connector.Crypto', None), \
             patch('TradingExecution.exchange_connectors.ibkr_connector.Future', None), \
             patch('TradingExecution.exchange_connectors.ibkr_connector.Option', None):

            from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
            conn = IBKRConnector(account='TEST', paper=True)
            result = await conn.connect()
            assert result is False

    @pytest.mark.asyncio
    async def test_disconnect(self, connector, mock_ib):
        await connector.disconnect()
        mock_ib.disconnect.assert_called_once()
        assert connector._connected is False
        assert connector._ib is None

    def test_ensure_connected_raises(self, connector, mock_ib):
        from TradingExecution.exchange_connectors.base_connector import ConnectionError
        mock_ib.isConnected.return_value = False
        with pytest.raises(ConnectionError, match="Not connected"):
            connector._ensure_connected()


# ─── MARKET DATA ─────────────────────────────────────────────────────────────

class TestMarketData:
    @pytest.mark.asyncio
    async def test_get_ticker(self, connector, mock_ib):
        """Test fetching a market data snapshot"""
        mock_ticker = MagicMock()
        mock_ticker.bid = 150.10
        mock_ticker.ask = 150.20
        mock_ticker.last = 150.15
        mock_ticker.volume = 1000000
        mock_ib.reqMktData.return_value = mock_ticker

        ticker = await connector.get_ticker('AAPL')

        assert ticker.symbol == 'AAPL'
        assert ticker.bid == 150.10
        assert ticker.ask == 150.20
        assert ticker.last == 150.15
        assert ticker.volume_24h == 1000000.0
        mock_ib.cancelMktData.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_ticker_no_data(self, connector, mock_ib):
        """Test ticker with empty data returns zeros"""
        mock_ticker = MagicMock()
        mock_ticker.bid = None
        mock_ticker.ask = None
        mock_ticker.last = None
        mock_ticker.volume = None
        mock_ib.reqMktData.return_value = mock_ticker

        ticker = await connector.get_ticker('AAPL')

        assert ticker.bid == 0.0
        assert ticker.ask == 0.0
        assert ticker.last == 0.0


# ─── ACCOUNT BALANCES ────────────────────────────────────────────────────────

class TestBalances:
    @pytest.mark.asyncio
    async def test_get_balances(self, connector, mock_ib):
        """Test fetching account balances"""
        mock_ib.accountValues.return_value = [
            SimpleNamespace(tag='CashBalance', currency='USD', value='50000.00'),
            SimpleNamespace(tag='CashBalance', currency='CAD', value='10000.00'),
            SimpleNamespace(tag='CashBalance', currency='BASE', value='60000.00'),
            SimpleNamespace(tag='NetLiquidation', currency='USD', value='100000.00'),
        ]
        mock_ib.positions.return_value = [
            SimpleNamespace(
                contract=SimpleNamespace(symbol='AAPL'),
                position=100,
            ),
        ]

        balances = await connector.get_balances()

        assert 'USD' in balances
        assert balances['USD'].free == 50000.0
        assert 'CAD' in balances
        assert balances['CAD'].free == 10000.0
        assert 'BASE' not in balances  # BASE currency filtered out
        assert 'AAPL' in balances
        assert balances['AAPL'].free == 100.0

    @pytest.mark.asyncio
    async def test_get_balances_zero_filtered(self, connector, mock_ib):
        """Zero balances should be filtered out"""
        mock_ib.accountValues.return_value = [
            SimpleNamespace(tag='CashBalance', currency='USD', value='0'),
        ]
        mock_ib.positions.return_value = []

        balances = await connector.get_balances()
        assert 'USD' not in balances


# ─── ORDERS ───────────────────────────────────────────────────────────────────

class TestOrders:
    @pytest.mark.asyncio
    async def test_create_market_order(self, connector, mock_ib):
        """Test creating a market order"""
        mock_trade = MagicMock()
        mock_trade.order.orderId = 42
        mock_trade.order.orderRef = None
        mock_trade.orderStatus.status = 'Submitted'
        mock_trade.orderStatus.filled = 0
        mock_trade.orderStatus.avgFillPrice = 0
        mock_trade.orderStatus.commission = 0
        mock_ib.placeOrder.return_value = mock_trade

        order = await connector.create_order(
            symbol='AAPL',
            side='buy',
            order_type='market',
            quantity=100,
        )

        assert order.order_id == '42'
        assert order.symbol == 'AAPL'
        assert order.side == 'buy'
        assert order.order_type == 'market'
        assert order.quantity == 100
        assert order.status == 'open'
        mock_ib.placeOrder.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_limit_order(self, connector, mock_ib):
        """Test creating a limit order"""
        mock_trade = MagicMock()
        mock_trade.order.orderId = 43
        mock_trade.order.orderRef = 'my-ref'
        mock_trade.orderStatus.status = 'PreSubmitted'
        mock_trade.orderStatus.filled = 0
        mock_trade.orderStatus.avgFillPrice = 0
        mock_trade.orderStatus.commission = 0
        mock_ib.placeOrder.return_value = mock_trade

        order = await connector.create_order(
            symbol='MSFT',
            side='sell',
            order_type='limit',
            quantity=50,
            price=400.00,
            client_order_id='my-ref',
        )

        assert order.order_id == '43'
        assert order.side == 'sell'
        assert order.order_type == 'limit'
        assert order.price == 400.00
        assert order.status == 'pending'

    @pytest.mark.asyncio
    async def test_create_limit_order_no_price_raises(self, connector):
        """Limit order without price should raise"""
        from TradingExecution.exchange_connectors.base_connector import OrderError
        with pytest.raises(OrderError, match="Price required"):
            await connector.create_order(
                symbol='AAPL',
                side='buy',
                order_type='limit',
                quantity=10,
            )

    @pytest.mark.asyncio
    async def test_create_order_unsupported_type(self, connector):
        """Unsupported order type should raise"""
        from TradingExecution.exchange_connectors.base_connector import OrderError
        with pytest.raises(OrderError, match="Unsupported order type"):
            await connector.create_order(
                symbol='AAPL',
                side='buy',
                order_type='iceberg',
                quantity=10,
            )

    @pytest.mark.asyncio
    async def test_cancel_order(self, connector, mock_ib):
        """Test cancelling an open order"""
        mock_trade = MagicMock()
        mock_trade.order.orderId = 42
        mock_ib.openTrades.return_value = [mock_trade]

        result = await connector.cancel_order('42', 'AAPL')
        assert result is True
        mock_ib.cancelOrder.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, connector, mock_ib):
        """Cancelling non-existent order returns False"""
        mock_ib.openTrades.return_value = []
        result = await connector.cancel_order('999', 'AAPL')
        assert result is False

    @pytest.mark.asyncio
    async def test_get_order(self, connector, mock_ib):
        """Test fetching order status"""
        mock_trade = MagicMock()
        mock_trade.order.orderId = 42
        mock_trade.order.orderRef = 'ref1'
        mock_trade.order.action = 'BUY'
        mock_trade.order.orderType = 'LMT'
        mock_trade.order.totalQuantity = 100
        mock_trade.order.lmtPrice = 150.0
        mock_trade.orderStatus.status = 'Filled'
        mock_trade.orderStatus.filled = 100
        mock_trade.orderStatus.avgFillPrice = 149.50
        mock_trade.orderStatus.commission = 1.00
        mock_ib.trades.return_value = [mock_trade]

        order = await connector.get_order('42', 'AAPL')

        assert order.order_id == '42'
        assert order.side == 'buy'
        assert order.status == 'filled'
        assert order.filled_quantity == 100
        assert order.average_price == 149.50
        assert order.fee == 1.00

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, connector, mock_ib):
        """Getting non-existent order raises OrderError"""
        from TradingExecution.exchange_connectors.base_connector import OrderError
        mock_ib.trades.return_value = []
        with pytest.raises(OrderError, match="not found"):
            await connector.get_order('999', 'AAPL')

    @pytest.mark.asyncio
    async def test_get_open_orders(self, connector, mock_ib):
        """Test listing open orders"""
        mock_trade = MagicMock()
        mock_trade.contract.symbol = 'AAPL'
        mock_trade.contract.secType = 'STK'
        mock_trade.order.orderId = 42
        mock_trade.order.orderRef = None
        mock_trade.order.action = 'BUY'
        mock_trade.order.orderType = 'LMT'
        mock_trade.order.totalQuantity = 50
        mock_trade.order.lmtPrice = 150.0
        mock_trade.orderStatus.status = 'Submitted'
        mock_trade.orderStatus.filled = 0
        mock_trade.orderStatus.avgFillPrice = 0
        mock_trade.orderStatus.commission = 0
        mock_ib.openTrades.return_value = [mock_trade]

        orders = await connector.get_open_orders()
        assert len(orders) == 1
        assert orders[0].symbol == 'AAPL:stk'

    @pytest.mark.asyncio
    async def test_get_open_orders_filter_symbol(self, connector, mock_ib):
        """Test filtering open orders by symbol"""
        mock_aapl = MagicMock()
        mock_aapl.contract.symbol = 'AAPL'
        mock_aapl.contract.secType = 'STK'
        mock_aapl.order.orderId = 1
        mock_aapl.order.orderRef = None
        mock_aapl.order.action = 'BUY'
        mock_aapl.order.orderType = 'MKT'
        mock_aapl.order.totalQuantity = 10
        mock_aapl.order.lmtPrice = 0
        mock_aapl.orderStatus.status = 'Submitted'
        mock_aapl.orderStatus.filled = 0
        mock_aapl.orderStatus.avgFillPrice = 0
        mock_aapl.orderStatus.commission = 0

        mock_msft = MagicMock()
        mock_msft.contract.symbol = 'MSFT'
        mock_msft.contract.secType = 'STK'
        mock_msft.order.orderId = 2
        mock_msft.order.orderRef = None
        mock_msft.order.action = 'SELL'
        mock_msft.order.orderType = 'LMT'
        mock_msft.order.totalQuantity = 20
        mock_msft.order.lmtPrice = 400
        mock_msft.orderStatus.status = 'Submitted'
        mock_msft.orderStatus.filled = 0
        mock_msft.orderStatus.avgFillPrice = 0
        mock_msft.orderStatus.commission = 0

        mock_ib.openTrades.return_value = [mock_aapl, mock_msft]

        orders = await connector.get_open_orders(symbol='AAPL')
        assert len(orders) == 1
        assert orders[0].order_id == '1'


# ─── STOP / TRAILING STOP / BRACKET ──────────────────────────────────────────

class TestAdvancedOrders:
    @pytest.mark.asyncio
    async def test_create_stop_loss_order(self, connector, mock_ib):
        mock_trade = MagicMock()
        mock_trade.order.orderId = 50
        mock_trade.orderStatus.status = 'PreSubmitted'
        mock_trade.orderStatus.filled = 0
        mock_trade.orderStatus.avgFillPrice = 0
        mock_trade.orderStatus.commission = 0
        mock_ib.placeOrder.return_value = mock_trade

        order = await connector.create_stop_loss_order(
            symbol='AAPL',
            side='sell',
            quantity=100,
            stop_price=140.0,
        )

        assert order.order_id == '50'
        assert order.order_type == 'stop'

    @pytest.mark.asyncio
    async def test_create_stop_limit_order(self, connector, mock_ib):
        mock_trade = MagicMock()
        mock_trade.order.orderId = 51
        mock_trade.orderStatus.status = 'PreSubmitted'
        mock_trade.orderStatus.filled = 0
        mock_trade.orderStatus.avgFillPrice = 0
        mock_trade.orderStatus.commission = 0
        mock_ib.placeOrder.return_value = mock_trade

        order = await connector.create_stop_loss_order(
            symbol='AAPL',
            side='sell',
            quantity=100,
            stop_price=140.0,
            limit_price=139.50,
        )

        assert order.order_type == 'stop_limit'

    @pytest.mark.asyncio
    async def test_create_trailing_stop(self, connector, mock_ib):
        mock_trade = MagicMock()
        mock_trade.order.orderId = 52
        mock_trade.orderStatus.status = 'PreSubmitted'
        mock_trade.orderStatus.filled = 0
        mock_trade.orderStatus.avgFillPrice = 0
        mock_trade.orderStatus.commission = 0
        mock_ib.placeOrder.return_value = mock_trade

        order = await connector.create_trailing_stop_order(
            symbol='AAPL',
            side='sell',
            quantity=100,
            trail_percent=2.0,
        )

        assert order.order_type == 'trailing_stop'
        assert order.order_id == '52'

    @pytest.mark.asyncio
    async def test_trailing_stop_no_params_raises(self, connector):
        from TradingExecution.exchange_connectors.base_connector import OrderError
        with pytest.raises(OrderError, match="trail_amount or trail_percent"):
            await connector.create_trailing_stop_order(
                symbol='AAPL',
                side='sell',
                quantity=100,
            )

    @pytest.mark.asyncio
    async def test_create_bracket_order(self, connector, mock_ib):
        mock_trades = []
        for oid in [60, 61, 62]:
            mt = MagicMock()
            mt.order.orderId = oid
            mt.order.action = 'BUY' if oid == 60 else 'SELL'
            mt.order.orderType = 'LMT'
            mt.order.lmtPrice = 150.0
            mt.order.auxPrice = 0
            mt.orderStatus.status = 'PreSubmitted'
            mt.orderStatus.filled = 0
            mt.orderStatus.avgFillPrice = 0
            mt.orderStatus.commission = 0
            mock_trades.append(mt)

        mock_ib.placeOrder.side_effect = mock_trades
        mock_ib.bracketOrder.return_value = [MagicMock(), MagicMock(), MagicMock()]

        orders = await connector.create_bracket_order(
            symbol='AAPL',
            side='buy',
            quantity=100,
            entry_price=150.0,
            take_profit_price=160.0,
            stop_loss_price=140.0,
        )

        assert len(orders) == 3
        mock_ib.bracketOrder.assert_called_once_with(
            'BUY', 100, 150.0, 160.0, 140.0,
        )


# ─── TRADE FEES ──────────────────────────────────────────────────────────────

class TestFees:
    @pytest.mark.asyncio
    async def test_trade_fee(self, connector):
        fees = await connector.get_trade_fee('AAPL')
        assert 'maker' in fees
        assert 'taker' in fees
        assert fees['maker'] == 0.0005


# ─── ERROR HANDLING ──────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_on_error_connection(self, connector):
        """Connection errors logged at ERROR level"""
        connector._on_error(1, 1100, "Connectivity lost", None)
        assert len(connector._error_log) == 1
        assert connector._error_log[0]['code'] == 1100
        assert connector._error_log[0]['category'] == 'connection_lost'

    def test_on_error_order_rejected(self, connector):
        connector._on_error(2, 201, "Order rejected", None)
        assert connector._error_log[-1]['category'] == 'order_rejected'

    def test_on_error_market_data(self, connector):
        connector._on_error(3, 354, "No subscription", None)
        assert connector._error_log[-1]['category'] == 'no_subscription'

    def test_on_error_rate_limit(self, connector):
        connector._on_error(4, 100, "Max rate exceeded", None)
        assert connector._error_log[-1]['category'] == 'max_rate'

    def test_error_log_bounded(self, connector):
        """Error log should not grow unbounded"""
        for i in range(1100):
            connector._on_error(i, 999, "test", None)
        assert len(connector._error_log) <= 1000

    def test_get_error_log(self, connector):
        connector._on_error(1, 200, "err1", None)
        connector._on_error(2, 201, "err2", None)
        log = connector.get_error_log(last_n=1)
        assert len(log) == 1
        assert log[0]['code'] == 201


# ─── CONNECTION STATUS ────────────────────────────────────────────────────────

class TestConnectionStatus:
    def test_get_connection_status(self, connector, mock_ib):
        status = connector.get_connection_status()

        assert status['connected'] is True
        assert status['ib_connected'] is True
        assert status['host'] == '127.0.0.1'
        assert status['port'] == 7497
        assert status['account'] == 'DUP782762'
        assert status['paper'] is True
        assert status['streaming_subscriptions'] == []


# ─── IB STATUS MAPPING ───────────────────────────────────────────────────────

class TestStatusMapping:
    def test_filled(self, connector):
        assert connector._map_ib_status('Filled') == 'filled'

    def test_submitted(self, connector):
        assert connector._map_ib_status('Submitted') == 'open'

    def test_cancelled(self, connector):
        assert connector._map_ib_status('Cancelled') == 'cancelled'

    def test_pending_submit(self, connector):
        assert connector._map_ib_status('PendingSubmit') == 'pending'

    def test_inactive(self, connector):
        assert connector._map_ib_status('Inactive') == 'rejected'

    def test_unknown(self, connector):
        assert connector._map_ib_status('SomethingNew') == 'unknown'


# ─── PORTFOLIO & ACCOUNT ─────────────────────────────────────────────────────

class TestPortfolio:
    @pytest.mark.asyncio
    async def test_get_portfolio(self, connector, mock_ib):
        mock_ib.portfolio.return_value = [
            SimpleNamespace(
                contract=SimpleNamespace(symbol='AAPL', secType='STK', exchange='SMART'),
                position=100,
                marketPrice=150.0,
                marketValue=15000.0,
                averageCost=140.0,
                unrealizedPNL=1000.0,
                realizedPNL=500.0,
            ),
        ]

        portfolio = await connector.get_portfolio()
        assert len(portfolio) == 1
        assert portfolio[0]['symbol'] == 'AAPL'
        assert portfolio[0]['position'] == 100.0
        assert portfolio[0]['unrealized_pnl'] == 1000.0

    @pytest.mark.asyncio
    async def test_get_account_summary(self, connector, mock_ib):
        mock_ib.accountSummary.return_value = [
            SimpleNamespace(tag='NetLiquidation', value='100000.0'),
            SimpleNamespace(tag='BuyingPower', value='200000.0'),
            SimpleNamespace(tag='AccountType', value='INDIVIDUAL'),
        ]

        summary = await connector.get_account_summary()
        assert summary['NetLiquidation'] == 100000.0
        assert summary['BuyingPower'] == 200000.0
        assert summary['AccountType'] == 'INDIVIDUAL'


# ─── HISTORICAL DATA ─────────────────────────────────────────────────────────

class TestHistoricalData:
    @pytest.mark.asyncio
    async def test_get_historical_data(self, connector, mock_ib):
        mock_ib.reqHistoricalData.return_value = [
            SimpleNamespace(
                date='2024-01-15',
                open=150.0, high=155.0, low=149.0, close=153.0, volume=5000000,
            ),
            SimpleNamespace(
                date='2024-01-16',
                open=153.0, high=156.0, low=152.0, close=154.5, volume=4500000,
            ),
        ]

        bars = await connector.get_historical_data('AAPL', duration='2 D', bar_size='1 day')

        assert len(bars) == 2
        assert bars[0]['date'] == '2024-01-15'
        assert bars[0]['close'] == 153.0
        assert bars[1]['volume'] == 4500000


# ─── ORDER CALLBACKS ─────────────────────────────────────────────────────────

class TestOrderCallbacks:
    @pytest.mark.asyncio
    async def test_order_callback_fires(self, connector, mock_ib):
        """Order callbacks should fire on order creation"""
        callback_data = []

        def on_order(event_type, order):
            callback_data.append((event_type, order.order_id))

        connector.on_order_update(on_order)

        mock_trade = MagicMock()
        mock_trade.order.orderId = 99
        mock_trade.order.orderRef = None
        mock_trade.orderStatus.status = 'Submitted'
        mock_trade.orderStatus.filled = 0
        mock_trade.orderStatus.avgFillPrice = 0
        mock_trade.orderStatus.commission = 0
        mock_ib.placeOrder.return_value = mock_trade

        await connector.create_order('AAPL', 'buy', 'market', 10)

        assert len(callback_data) == 1
        assert callback_data[0] == ('created', '99')
