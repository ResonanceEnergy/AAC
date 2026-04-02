"""Unit tests for TradingExecution.trading_engine — Order and Position dataclasses."""
from datetime import datetime

import pytest

from TradingExecution.trading_engine import Order, OrderSide, OrderStatus, OrderType, Position


class TestOrderSideEnum:
    def test_buy(self):
        assert OrderSide.BUY.value == "buy"

    def test_sell(self):
        assert OrderSide.SELL.value == "sell"


class TestOrderTypeEnum:
    def test_market(self):
        assert OrderType.MARKET.value == "market"

    def test_limit(self):
        assert OrderType.LIMIT.value == "limit"

    def test_stop_loss(self):
        assert OrderType.STOP_LOSS.value == "stop_loss"

    def test_take_profit(self):
        assert OrderType.TAKE_PROFIT.value == "take_profit"


class TestOrderStatusEnum:
    def test_statuses(self):
        expected = {"pending", "submitted", "partial", "filled", "cancelled", "rejected", "expired"}
        actual = {s.value for s in OrderStatus}
        assert actual == expected


class TestOrder:
    def test_create_market_order(self):
        o = Order(
            order_id="ORD-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )
        assert o.order_id == "ORD-001"
        assert o.status == OrderStatus.PENDING
        assert o.filled_quantity == 0.0
        assert o.average_fill_price == 0.0

    def test_create_limit_order_with_price(self):
        o = Order(
            order_id="ORD-002",
            exchange="kraken",
            symbol="ETH/USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=3000.0,
        )
        assert o.price == 3000.0
        assert o.order_type == OrderType.LIMIT

    def test_default_metadata(self):
        o = Order(
            order_id="ORD-003",
            exchange="coinbase",
            symbol="SOL/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
        )
        assert o.metadata == {}

    def test_created_at_auto(self):
        before = datetime.now()
        o = Order(
            order_id="ORD-004",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        assert o.created_at >= before


class TestPosition:
    def test_create_long_position(self):
        p = Position(
            position_id="POS-001",
            exchange="binance",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.5,
            entry_price=60000.0,
            current_price=60000.0,
        )
        assert p.position_id == "POS-001"
        assert p.unrealized_pnl == 0.0
        assert p.realized_pnl == 0.0

    def test_position_defaults(self):
        p = Position(
            position_id="POS-002",
            exchange="kraken",
            symbol="ETH/USD",
            side=OrderSide.SELL,
            quantity=10.0,
            entry_price=3000.0,
        )
        assert p.current_price == 0.0
        assert p.metadata == {}

    def test_position_with_pnl(self):
        p = Position(
            position_id="POS-003",
            exchange="coinbase",
            symbol="SOL/USD",
            side=OrderSide.BUY,
            quantity=100.0,
            entry_price=150.0,
            current_price=160.0,
        )
        assert p.unrealized_pnl == 1000.0
