#!/usr/bin/env python3
"""
Tests for IBKR Exchange Connector
==================================
Unit tests with ib_insync fully mocked — no TWS/Gateway required.
"""

import pytest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Mock ib_insync before importing connector ──────────────────────────────────

mock_ib_module = MagicMock()

# Create mock classes that ib_insync exports
mock_ib_module.IB = MagicMock
mock_ib_module.Stock = MagicMock
mock_ib_module.Forex = MagicMock
mock_ib_module.Contract = MagicMock
mock_ib_module.Order = MagicMock
mock_ib_module.LimitOrder = MagicMock
mock_ib_module.MarketOrder = MagicMock
mock_ib_module.StopOrder = MagicMock
mock_ib_module.StopLimitOrder = MagicMock
mock_ib_module.Trade = MagicMock
mock_ib_module.util = MagicMock

sys.modules['ib_insync'] = mock_ib_module

from TradingExecution.exchange_connectors.ibkr_connector import (
    IBKRConnector,
    _parse_symbol,
)
from TradingExecution.exchange_connectors.base_connector import (
    ConnectionError,
    OrderError,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def connector():
    """Create a connector instance with test config."""
    c = IBKRConnector(
        host='127.0.0.1',
        port=7497,
        client_id=1,
        account='DU1234567',
        paper=True,
    )
    return c


@pytest.fixture
def mock_ib():
    """Create a mock IB instance with common methods."""
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.managedAccounts.return_value = ['DU1234567']
    ib.connect.return_value = None
    return ib


@pytest.fixture
def connected_connector(connector, mock_ib):
    """Create a connector that appears connected."""
    connector._ib = mock_ib
    connector._connected = True
    connector.account = 'DU1234567'
    return connector


# ── Symbol Parsing Tests ───────────────────────────────────────────────────────

class TestSymbolParsing:
    """Test AAC symbol -> IBKR contract parsing."""

    def test_stock_with_currency(self):
        result = _parse_symbol('AAPL/USD')
        assert result['type'] == 'stock'
        assert result['symbol'] == 'AAPL'
        assert result['currency'] == 'USD'

    def test_stock_without_currency(self):
        result = _parse_symbol('TSLA')
        assert result['type'] == 'stock'
        assert result['symbol'] == 'TSLA'
        assert result['currency'] == 'USD'

    def test_forex_pair(self):
        result = _parse_symbol('EUR/USD')
        assert result['type'] == 'forex'
        assert result['pair'] == 'EURUSD'

    def test_forex_reverse(self):
        result = _parse_symbol('USD/CAD')
        assert result['type'] == 'forex'
        assert result['pair'] == 'USDCAD'

    def test_crypto_pair(self):
        result = _parse_symbol('BTC/USD')
        assert result['type'] == 'crypto'
        assert result['symbol'] == 'BTC'
        assert result['currency'] == 'USD'

    def test_eth_pair(self):
        result = _parse_symbol('ETH/USD')
        assert result['type'] == 'crypto'
        assert result['symbol'] == 'ETH'

    def test_case_insensitive(self):
        result = _parse_symbol('aapl/usd')
        assert result['symbol'] == 'AAPL'
        assert result['currency'] == 'USD'

    def test_whitespace_handling(self):
        result = _parse_symbol(' AAPL / USD ')
        assert result['symbol'] == 'AAPL'
        assert result['currency'] == 'USD'


class TestConnectorProperties:
    """Test basic connector properties."""

    def test_name(self, connector):
        assert connector.name == 'ibkr'

    def test_default_config(self, connector):
        assert connector.host == '127.0.0.1'
        assert connector.port == 7497
        assert connector.client_id == 1
        assert connector.account == 'DU1234567'
        assert connector.paper is True

    def test_not_connected_initially(self, connector):
        assert connector.is_connected is False

    def test_rate_limit(self, connector):
        assert connector.rate_limit == 50


class TestConnect:
    """Test connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_success(self, connector, mock_ib):
        """Test successful connection to TWS."""
        with patch('TradingExecution.exchange_connectors.ibkr_connector.IB', return_value=mock_ib):
            result = await connector.connect()

        assert result is True
        assert connector.is_connected is True
        assert connector.account == 'DU1234567'

    @pytest.mark.asyncio
    async def test_connect_no_ib_insync(self, connector):
        """Test graceful failure when ib_insync not installed."""
        with patch('TradingExecution.exchange_connectors.ibkr_connector.IB_INSYNC_AVAILABLE', False):
            result = await connector.connect()
            assert result is False

    @pytest.mark.asyncio
    async def test_connect_uses_first_account_when_unspecified(self, mock_ib):
        """When no account specified, use first managed account."""
        mock_ib.managedAccounts.return_value = ['DU9999999', 'DU1111111']
        c = IBKRConnector(host='127.0.0.1', port=7497, account='')

        with patch('TradingExecution.exchange_connectors.ibkr_connector.IB', return_value=mock_ib):
            await c.connect()

        assert c.account == 'DU9999999'

    @pytest.mark.asyncio
    async def test_disconnect(self, connected_connector, mock_ib):
        """Test clean disconnection."""
        await connected_connector.disconnect()
        assert connected_connector.is_connected is False
        mock_ib.disconnect.assert_called_once()


class TestGetBalances:
    """Test balance retrieval — your $1,000 starts here."""

    @pytest.mark.asyncio
    async def test_get_cash_balances(self, connected_connector, mock_ib):
        """Test retrieving cash balances from IBKR."""
        # Mock account values — simulating $1,000 USD balance
        mock_av_usd = MagicMock()
        mock_av_usd.tag = 'CashBalance'
        mock_av_usd.currency = 'USD'
        mock_av_usd.value = '1000.00'

        mock_av_base = MagicMock()
        mock_av_base.tag = 'TotalCashBalance'
        mock_av_base.currency = 'BASE'
        mock_av_base.value = '1000.00'

        mock_av_net = MagicMock()
        mock_av_net.tag = 'NetLiquidation'
        mock_av_net.currency = 'BASE'
        mock_av_net.value = '1000.00'

        mock_ib.accountValues.return_value = [mock_av_usd, mock_av_base, mock_av_net]
        mock_ib.positions.return_value = []

        balances = await connected_connector.get_balances()

        assert 'USD' in balances
        assert balances['USD'].free == 1000.00
        assert balances['USD'].total == 1000.00
        assert 'TOTAL_CASH' in balances
        assert balances['NET_LIQUIDATION'].free == 1000.00

    @pytest.mark.asyncio
    async def test_get_balances_with_positions(self, connected_connector, mock_ib):
        """Test balances include stock positions."""
        mock_av = MagicMock()
        mock_av.tag = 'CashBalance'
        mock_av.currency = 'USD'
        mock_av.value = '500.00'
        mock_ib.accountValues.return_value = [mock_av]

        # Mock a position
        mock_pos = MagicMock()
        mock_pos.contract.symbol = 'AAPL'
        mock_pos.position = 5.0
        mock_pos.marketValue = 850.0
        mock_ib.positions.return_value = [mock_pos]

        balances = await connected_connector.get_balances()
        assert 'USD' in balances
        assert 'AAPL' in balances
        assert balances['AAPL'].free == 5.0


class TestCreateOrder:
    """Test order creation."""

    @pytest.mark.asyncio
    async def test_market_buy(self, connected_connector, mock_ib):
        """Test placing a market buy order."""
        # Mock contract qualification
        mock_contract = MagicMock()
        mock_ib.qualifyContracts.return_value = [mock_contract]

        # Mock trade result
        mock_trade = MagicMock()
        mock_trade.order.orderId = 12345
        mock_trade.order.clientId = 1
        mock_trade.order.action = 'BUY'
        mock_trade.order.orderType = 'MKT'
        mock_trade.order.totalQuantity = 10.0
        mock_trade.order.lmtPrice = None
        mock_trade.order.permId = 99999
        mock_trade.order.account = 'DU1234567'
        mock_trade.orderStatus.status = 'Submitted'
        mock_trade.orderStatus.filled = 0.0
        mock_trade.orderStatus.avgFillPrice = 0.0
        mock_trade.orderStatus.commission = 0.0
        mock_ib.placeOrder.return_value = mock_trade

        order = await connected_connector.create_order(
            symbol='AAPL/USD',
            side='buy',
            order_type='market',
            quantity=10.0,
        )

        assert order.order_id == '12345'
        assert order.side == 'buy'
        assert order.order_type == 'market'
        assert order.quantity == 10.0
        assert order.status == 'open'

    @pytest.mark.asyncio
    async def test_limit_sell(self, connected_connector, mock_ib):
        """Test placing a limit sell order."""
        mock_contract = MagicMock()
        mock_ib.qualifyContracts.return_value = [mock_contract]

        mock_trade = MagicMock()
        mock_trade.order.orderId = 12346
        mock_trade.order.clientId = 1
        mock_trade.order.action = 'SELL'
        mock_trade.order.orderType = 'LMT'
        mock_trade.order.totalQuantity = 5.0
        mock_trade.order.lmtPrice = 150.00
        mock_trade.order.permId = 99998
        mock_trade.order.account = 'DU1234567'
        mock_trade.orderStatus.status = 'Submitted'
        mock_trade.orderStatus.filled = 0.0
        mock_trade.orderStatus.avgFillPrice = 0.0
        mock_trade.orderStatus.commission = 0.0
        mock_ib.placeOrder.return_value = mock_trade

        order = await connected_connector.create_order(
            symbol='AAPL/USD',
            side='sell',
            order_type='limit',
            quantity=5.0,
            price=150.00,
        )

        assert order.order_id == '12346'
        assert order.side == 'sell'
        assert order.order_type == 'limit'
        assert order.price == 150.00

    @pytest.mark.asyncio
    async def test_limit_order_requires_price(self, connected_connector, mock_ib):
        """Limit order without price should raise OrderError."""
        mock_ib.qualifyContracts.return_value = [MagicMock()]

        with pytest.raises(OrderError, match="Price required"):
            await connected_connector.create_order(
                symbol='AAPL/USD',
                side='buy',
                order_type='limit',
                quantity=10.0,
                price=None,
            )

    @pytest.mark.asyncio
    async def test_unsupported_order_type(self, connected_connector, mock_ib):
        """Unsupported order type should raise OrderError."""
        mock_ib.qualifyContracts.return_value = [MagicMock()]

        with pytest.raises(OrderError, match="Unsupported order type"):
            await connected_connector.create_order(
                symbol='AAPL/USD',
                side='buy',
                order_type='trailing_stop',
                quantity=10.0,
            )

    @pytest.mark.asyncio
    async def test_unqualifiable_contract(self, connected_connector, mock_ib):
        """Order for invalid symbol should raise OrderError."""
        mock_ib.qualifyContracts.return_value = []

        with pytest.raises(OrderError, match="Could not qualify"):
            await connected_connector.create_order(
                symbol='FAKE/USD',
                side='buy',
                order_type='market',
                quantity=1.0,
            )


class TestCancelOrder:
    """Test order cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_existing_order(self, connected_connector, mock_ib):
        """Test cancelling an open order."""
        mock_trade = MagicMock()
        mock_trade.order.orderId = 12345
        mock_ib.openTrades.return_value = [mock_trade]

        result = await connected_connector.cancel_order('12345', 'AAPL/USD')
        assert result is True
        mock_ib.cancelOrder.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, connected_connector, mock_ib):
        """Test cancelling an order that doesn't exist."""
        mock_ib.openTrades.return_value = []

        result = await connected_connector.cancel_order('99999', 'AAPL/USD')
        assert result is False


class TestGetOrder:
    """Test order retrieval."""

    @pytest.mark.asyncio
    async def test_get_open_order(self, connected_connector, mock_ib):
        """Test getting an open order by ID."""
        mock_trade = MagicMock()
        mock_trade.order.orderId = 12345
        mock_trade.order.clientId = 1
        mock_trade.order.action = 'BUY'
        mock_trade.order.orderType = 'LMT'
        mock_trade.order.totalQuantity = 10.0
        mock_trade.order.lmtPrice = 145.00
        mock_trade.order.permId = 99999
        mock_trade.order.account = 'DU1234567'
        mock_trade.orderStatus.status = 'Submitted'
        mock_trade.orderStatus.filled = 0.0
        mock_trade.orderStatus.avgFillPrice = 0.0
        mock_trade.orderStatus.commission = 0.0
        mock_ib.openTrades.return_value = [mock_trade]

        order = await connected_connector.get_order('12345', 'AAPL/USD')
        assert order.order_id == '12345'
        assert order.status == 'open'

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, connected_connector, mock_ib):
        """Getting a nonexistent order should raise OrderError."""
        mock_ib.openTrades.return_value = []
        mock_ib.trades.return_value = []

        with pytest.raises(OrderError, match="not found"):
            await connected_connector.get_order('99999', 'AAPL/USD')


class TestGetOpenOrders:
    """Test listing open orders."""

    @pytest.mark.asyncio
    async def test_get_all_open_orders(self, connected_connector, mock_ib):
        """Test getting all open orders."""
        mock_trade1 = MagicMock()
        mock_trade1.order.orderId = 1
        mock_trade1.order.clientId = 1
        mock_trade1.order.action = 'BUY'
        mock_trade1.order.orderType = 'LMT'
        mock_trade1.order.totalQuantity = 10.0
        mock_trade1.order.lmtPrice = 145.00
        mock_trade1.order.permId = 1
        mock_trade1.order.account = 'DU1234567'
        mock_trade1.orderStatus.status = 'Submitted'
        mock_trade1.orderStatus.filled = 0.0
        mock_trade1.orderStatus.avgFillPrice = 0.0
        mock_trade1.orderStatus.commission = 0.0
        mock_trade1.contract.symbol = 'AAPL'

        mock_trade2 = MagicMock()
        mock_trade2.order.orderId = 2
        mock_trade2.order.clientId = 1
        mock_trade2.order.action = 'SELL'
        mock_trade2.order.orderType = 'MKT'
        mock_trade2.order.totalQuantity = 5.0
        mock_trade2.order.lmtPrice = None
        mock_trade2.order.permId = 2
        mock_trade2.order.account = 'DU1234567'
        mock_trade2.orderStatus.status = 'Submitted'
        mock_trade2.orderStatus.filled = 0.0
        mock_trade2.orderStatus.avgFillPrice = 0.0
        mock_trade2.orderStatus.commission = 0.0
        mock_trade2.contract.symbol = 'TSLA'

        mock_ib.openTrades.return_value = [mock_trade1, mock_trade2]

        orders = await connected_connector.get_open_orders()
        assert len(orders) == 2

    @pytest.mark.asyncio
    async def test_no_open_orders(self, connected_connector, mock_ib):
        """Test when there are no open orders."""
        mock_ib.openTrades.return_value = []
        orders = await connected_connector.get_open_orders()
        assert orders == []


class TestTradeFees:
    """Test fee structure lookup."""

    @pytest.mark.asyncio
    async def test_stock_fees(self, connector):
        fees = await connector.get_trade_fee('AAPL/USD')
        assert fees['maker'] == 0.0001
        assert fees['taker'] == 0.0001

    @pytest.mark.asyncio
    async def test_forex_fees(self, connector):
        fees = await connector.get_trade_fee('EUR/USD')
        assert fees['maker'] == 0.00002

    @pytest.mark.asyncio
    async def test_crypto_fees(self, connector):
        fees = await connector.get_trade_fee('BTC/USD')
        assert fees['maker'] == 0.0012
        assert fees['taker'] == 0.0018


class TestStopLoss:
    """Test stop-loss order creation."""

    @pytest.mark.asyncio
    async def test_stop_order(self, connected_connector, mock_ib):
        """Test plain stop order."""
        mock_ib.qualifyContracts.return_value = [MagicMock()]

        mock_trade = MagicMock()
        mock_trade.order.orderId = 55555
        mock_trade.order.clientId = 1
        mock_trade.order.action = 'SELL'
        mock_trade.order.orderType = 'STP'
        mock_trade.order.totalQuantity = 10.0
        mock_trade.order.lmtPrice = None
        mock_trade.order.permId = 88888
        mock_trade.order.account = 'DU1234567'
        mock_trade.orderStatus.status = 'Submitted'
        mock_trade.orderStatus.filled = 0.0
        mock_trade.orderStatus.avgFillPrice = 0.0
        mock_trade.orderStatus.commission = 0.0
        mock_ib.placeOrder.return_value = mock_trade

        order = await connected_connector.create_stop_loss_order(
            symbol='AAPL/USD',
            side='sell',
            quantity=10.0,
            stop_price=140.00,
        )

        assert order.order_id == '55555'
        assert order.order_type == 'stop'


class TestAccountSummary:
    """Test account summary (IBKR-specific feature)."""

    @pytest.mark.asyncio
    async def test_account_summary(self, connected_connector, mock_ib):
        """Test comprehensive account summary."""
        mock_values = []
        for tag, val in [
            ('NetLiquidation', '1000.00'),
            ('TotalCashValue', '1000.00'),
            ('BuyingPower', '4000.00'),
            ('AvailableFunds', '1000.00'),
        ]:
            mv = MagicMock()
            mv.tag = tag
            mv.currency = 'USD'
            mv.value = val
            mock_values.append(mv)

        mock_ib.accountValues.return_value = mock_values
        summary = await connected_connector.get_account_summary()

        assert summary['NetLiquidation_USD'] == 1000.00
        assert summary['BuyingPower_USD'] == 4000.00


class TestConnectionGuards:
    """Test that operations fail properly when not connected."""

    @pytest.mark.asyncio
    async def test_ticker_when_disconnected(self, connector):
        with pytest.raises(ConnectionError, match="Not connected"):
            await connector.get_ticker('AAPL/USD')

    @pytest.mark.asyncio
    async def test_balances_when_disconnected(self, connector):
        with pytest.raises(ConnectionError, match="Not connected"):
            await connector.get_balances()

    @pytest.mark.asyncio
    async def test_create_order_when_disconnected(self, connector):
        with pytest.raises(ConnectionError, match="Not connected"):
            await connector.create_order('AAPL/USD', 'buy', 'market', 10)
