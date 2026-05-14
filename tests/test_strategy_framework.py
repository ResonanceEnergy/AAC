from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.strategy_framework import (
    BaseArbitrageStrategy,
    SignalType,
    StrategyConfig,
    StrategyStatus,
    TradingSignal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    data_requirements=None,
    risk_envelope=None,
    symbols=None,
    strategy_id="strat-1",
):
    return StrategyConfig(
        strategy_id=strategy_id,
        name="Test Strategy",
        strategy_type="arb",
        edge_source="test",
        time_horizon="intraday",
        complexity="low",
        data_requirements=data_requirements if data_requirements is not None else ["etf_prices"],
        execution_requirements=["ibkr"],
        risk_envelope=risk_envelope if risk_envelope is not None else {"max_position_pct": 5.0},
        cross_department_dependencies={},
        symbols=symbols if symbols is not None else ["SPY"],
    )


class _ConcreteStrategy(BaseArbitrageStrategy):
    """Minimal concrete subclass for testing the abstract base."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_called = 0
        self.signals_to_emit: list[TradingSignal] = []
        self.should_emit = True

    async def _initialize_strategy(self):
        self.init_called += 1

    async def _generate_signals(self):
        return list(self.signals_to_emit)

    def _should_generate_signal(self) -> bool:
        return self.should_emit


def _make_strategy(**cfg_kwargs):
    config = _make_config(**cfg_kwargs)
    comm = MagicMock()
    comm.subscribe_to_messages = AsyncMock(return_value=None)
    comm.unsubscribe_from_messages = AsyncMock(return_value=None)
    comm.send_message = AsyncMock(return_value=None)
    audit = MagicMock()
    audit.log_event = AsyncMock(return_value=None)
    return _ConcreteStrategy(config, comm, audit), comm, audit


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestStrategyStatusEnum:
    def test_values(self):
        assert StrategyStatus.INACTIVE.value == "inactive"
        assert StrategyStatus.ACTIVE.value == "active"
        assert StrategyStatus.PAUSED.value == "paused"
        assert StrategyStatus.ERROR.value == "error"

    def test_count(self):
        assert len(list(StrategyStatus)) == 4


class TestSignalTypeEnum:
    def test_values(self):
        assert SignalType.LONG.value == "long"
        assert SignalType.SHORT.value == "short"
        assert SignalType.CLOSE.value == "close"
        assert SignalType.HEDGE.value == "hedge"
        assert SignalType.FLAT.value == "flat"

    def test_count(self):
        assert len(list(SignalType)) == 5


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestTradingSignal:
    def test_post_init_defaults(self):
        sig = TradingSignal(
            strategy_id="s1", signal_type=SignalType.LONG, symbol="SPY", quantity=10
        )
        assert isinstance(sig.timestamp, datetime)
        assert sig.metadata == {}

    def test_explicit_timestamp_and_metadata(self):
        ts = datetime(2026, 1, 1)
        sig = TradingSignal(
            strategy_id="s1",
            signal_type=SignalType.SHORT,
            symbol="QQQ",
            quantity=-5,
            price=400.0,
            confidence=0.8,
            metadata={"k": "v"},
            timestamp=ts,
        )
        assert sig.timestamp == ts
        assert sig.metadata == {"k": "v"}
        assert sig.price == 400.0

    def test_metadata_independent_per_instance(self):
        a = TradingSignal(strategy_id="s", signal_type=SignalType.LONG, symbol="X", quantity=1)
        b = TradingSignal(strategy_id="s", signal_type=SignalType.LONG, symbol="X", quantity=1)
        a.metadata["x"] = 1
        assert b.metadata == {}


class TestStrategyConfig:
    def test_required_fields(self):
        cfg = _make_config()
        assert cfg.strategy_id == "strat-1"
        assert cfg.symbols == ["SPY"]

    def test_default_symbols_empty(self):
        cfg = StrategyConfig(
            strategy_id="x",
            name="x",
            strategy_type="x",
            edge_source="x",
            time_horizon="x",
            complexity="x",
            data_requirements=[],
            execution_requirements=[],
            risk_envelope={},
            cross_department_dependencies={},
        )
        assert cfg.symbols == []


# ---------------------------------------------------------------------------
# initialize / shutdown
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        strat, comm, audit = _make_strategy()
        ok = await strat.initialize()
        assert ok is True
        assert strat.status == StrategyStatus.ACTIVE
        assert strat.init_called == 1
        comm.subscribe_to_messages.assert_awaited_once()
        audit.log_event.assert_awaited()

    @pytest.mark.asyncio
    async def test_initialize_failure_sets_error(self):
        strat, comm, _ = _make_strategy()
        comm.subscribe_to_messages = AsyncMock(side_effect=RuntimeError("boom"))
        ok = await strat.initialize()
        assert ok is False
        assert strat.status == StrategyStatus.ERROR


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_success(self):
        strat, comm, audit = _make_strategy()
        await strat.initialize()
        ok = await strat.shutdown()
        assert ok is True
        assert strat.status == StrategyStatus.INACTIVE
        comm.unsubscribe_from_messages.assert_awaited()

    @pytest.mark.asyncio
    async def test_shutdown_with_position_sends_close(self):
        strat, comm, _ = _make_strategy()
        await strat.initialize()
        strat.position_size = 10.0
        await strat.shutdown()
        comm.send_message.assert_awaited()
        topic, payload = comm.send_message.await_args.args
        assert topic == "trading_execution.close_position"
        assert payload["quantity"] == -10.0

    @pytest.mark.asyncio
    async def test_shutdown_failure_returns_false(self):
        strat, comm, _ = _make_strategy()
        await strat.initialize()
        comm.unsubscribe_from_messages = AsyncMock(side_effect=RuntimeError("x"))
        ok = await strat.shutdown()
        assert ok is False


# ---------------------------------------------------------------------------
# Subscribe topic mapping
# ---------------------------------------------------------------------------


class TestSubscribeTopics:
    @pytest.mark.asyncio
    async def test_etf_prices_maps(self):
        strat, comm, _ = _make_strategy(data_requirements=["etf_prices"])
        await strat._subscribe_market_data()
        comm.subscribe_to_messages.assert_awaited_once_with("strat-1", ["market_data.etf.*"])

    @pytest.mark.asyncio
    async def test_nav_and_futures_map(self):
        strat, comm, _ = _make_strategy(data_requirements=["nav_calculations", "index_futures"])
        await strat._subscribe_market_data()
        _, topics = comm.subscribe_to_messages.await_args.args
        assert "bigbrain.nav.*" in topics
        assert "market_data.futures.*" in topics

    @pytest.mark.asyncio
    async def test_unknown_requirement_yields_no_subscribe(self):
        strat, comm, _ = _make_strategy(data_requirements=["mystery"])
        await strat._subscribe_market_data()
        comm.subscribe_to_messages.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unsubscribe_mirrors_subscribe(self):
        strat, comm, _ = _make_strategy(data_requirements=["etf_prices"])
        await strat._unsubscribe_market_data()
        comm.unsubscribe_from_messages.assert_awaited_once_with("strat-1", ["market_data.etf.*"])

    @pytest.mark.asyncio
    async def test_unsubscribe_skips_when_no_topics(self):
        strat, comm, _ = _make_strategy(data_requirements=["mystery"])
        await strat._unsubscribe_market_data()
        comm.unsubscribe_from_messages.assert_not_awaited()


# ---------------------------------------------------------------------------
# _update_market_data
# ---------------------------------------------------------------------------


class TestUpdateMarketData:
    def test_etf_price_routes_to_market_data(self):
        strat, _, _ = _make_strategy()
        strat._update_market_data({"type": "etf_price", "symbol": "SPY", "price": 500})
        assert strat.market_data["SPY"]["price"] == 500

    def test_nav_routes_to_nav_data(self):
        strat, _, _ = _make_strategy()
        strat._update_market_data({"type": "nav_calculation", "symbol": "SPY", "nav": 499.5})
        assert strat.nav_data["SPY"]["nav"] == 499.5

    def test_futures_routes_to_futures_data(self):
        strat, _, _ = _make_strategy()
        strat._update_market_data({"type": "futures_price", "symbol": "ES", "price": 5000})
        assert strat.futures_data["ES"]["price"] == 5000

    def test_unknown_type_ignored(self):
        strat, _, _ = _make_strategy()
        strat._update_market_data({"type": "weather", "symbol": "X"})
        assert strat.market_data == {}
        assert strat.nav_data == {}
        assert strat.futures_data == {}

    def test_missing_symbol_does_not_store(self):
        strat, _, _ = _make_strategy()
        strat._update_market_data({"type": "etf_price"})
        assert strat.market_data == {}


# ---------------------------------------------------------------------------
# process_market_data
# ---------------------------------------------------------------------------


class TestProcessMarketData:
    @pytest.mark.asyncio
    async def test_inactive_returns_empty(self):
        strat, _, _ = _make_strategy()
        # Never initialized → INACTIVE
        out = await strat.process_market_data({"type": "etf_price"})
        assert out == []

    @pytest.mark.asyncio
    async def test_should_not_generate_returns_empty(self):
        strat, _, _ = _make_strategy()
        await strat.initialize()
        strat.should_emit = False
        out = await strat.process_market_data({"type": "etf_price", "symbol": "SPY"})
        assert out == []

    @pytest.mark.asyncio
    async def test_emits_valid_signals(self):
        strat, _, audit = _make_strategy(risk_envelope={"max_position_pct": 100.0})
        await strat.initialize()
        sig = TradingSignal(
            strategy_id="strat-1", signal_type=SignalType.LONG, symbol="SPY", quantity=1
        )
        strat.signals_to_emit = [sig]
        out = await strat.process_market_data({"type": "etf_price", "symbol": "SPY"})
        assert out == [sig]
        # Audit log called for signal generation (in addition to init)
        assert audit.log_event.await_count >= 2

    @pytest.mark.asyncio
    async def test_filters_signals_violating_risk(self):
        strat, _, _ = _make_strategy(risk_envelope={"max_position_pct": 1.0})
        await strat.initialize()
        # quantity exceeds max_position_pct=1.0
        bad = TradingSignal(
            strategy_id="strat-1", signal_type=SignalType.LONG, symbol="SPY", quantity=10
        )
        strat.signals_to_emit = [bad]
        out = await strat.process_market_data({"type": "etf_price", "symbol": "SPY"})
        assert out == []

    @pytest.mark.asyncio
    async def test_swallows_internal_exceptions(self):
        strat, _, _ = _make_strategy()
        await strat.initialize()

        async def boom():
            raise RuntimeError("nope")

        strat._generate_signals = boom  # type: ignore[assignment]
        out = await strat.process_market_data({"type": "etf_price", "symbol": "SPY"})
        assert out == []


# ---------------------------------------------------------------------------
# update_position
# ---------------------------------------------------------------------------


class TestUpdatePosition:
    @pytest.mark.asyncio
    async def test_accumulates(self):
        strat, _, _ = _make_strategy()
        await strat.update_position("SPY", 5, 100.0)
        await strat.update_position("SPY", -2, 105.0)
        assert strat.position_size == 3
        assert strat.unrealized_pnl == pytest.approx(5 * 100.0 + -2 * 105.0)


# ---------------------------------------------------------------------------
# _check_risk_limits
# ---------------------------------------------------------------------------


class TestCheckRiskLimits:
    def test_under_limit_passes(self):
        strat, _, _ = _make_strategy(risk_envelope={"max_position_pct": 5.0})
        sig = TradingSignal("s", SignalType.LONG, "SPY", quantity=2)
        assert strat._check_risk_limits(sig) is True

    def test_over_limit_blocks(self):
        strat, _, _ = _make_strategy(risk_envelope={"max_position_pct": 5.0})
        sig = TradingSignal("s", SignalType.LONG, "SPY", quantity=10)
        assert strat._check_risk_limits(sig) is False

    def test_max_holding_1_day_blocks_after_24h(self):
        strat, _, _ = _make_strategy(
            risk_envelope={"max_position_pct": 100.0, "max_holding_period": "1_day"}
        )
        strat.last_signal_time = datetime.now() - timedelta(days=2)
        sig = TradingSignal("s", SignalType.LONG, "SPY", quantity=1)
        assert strat._check_risk_limits(sig) is False


# ---------------------------------------------------------------------------
# Close-all helpers
# ---------------------------------------------------------------------------


class TestCloseAllPositions:
    @pytest.mark.asyncio
    async def test_no_position_no_message(self):
        strat, comm, _ = _make_strategy()
        await strat._close_all_positions()
        comm.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_with_position_sends_close(self):
        strat, comm, _ = _make_strategy()
        strat.position_size = 7.0
        await strat._close_all_positions()
        topic, payload = comm.send_message.await_args.args
        assert topic == "trading_execution.close_position"
        assert payload["signal_type"] == SignalType.CLOSE
        assert payload["quantity"] == -7.0
        assert payload["symbol"] == "ALL"

    @pytest.mark.asyncio
    async def test_public_close_all_positions(self):
        strat, comm, _ = _make_strategy()
        strat.position_size = 3.0
        await strat.close_all_positions()
        comm.send_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# Public properties / methods
# ---------------------------------------------------------------------------


class TestPublicProperties:
    @pytest.mark.asyncio
    async def test_is_active_false_initially(self):
        strat, _, _ = _make_strategy()
        assert strat.is_active is False

    @pytest.mark.asyncio
    async def test_is_active_true_after_init(self):
        strat, _, _ = _make_strategy()
        await strat.initialize()
        assert strat.is_active is True

    @pytest.mark.asyncio
    async def test_generate_signals_proxies_to_private(self):
        strat, _, _ = _make_strategy()
        sig = TradingSignal("strat-1", SignalType.LONG, "SPY", quantity=1)
        strat.signals_to_emit = [sig]
        out = await strat.generate_signals()
        assert out == [sig]
