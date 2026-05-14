from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.strategy_execution_engine import (
    StrategyExecutionEngine,
    get_strategy_execution_engine,
)
from shared.strategy_framework import StrategyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(strategy_id="s01_x", data_requirements=None):
    return StrategyConfig(
        strategy_id=strategy_id,
        name="X",
        strategy_type="stat_arb",
        edge_source="pricing_inefficiency",
        time_horizon="intraday",
        complexity="medium",
        data_requirements=data_requirements if data_requirements is not None else [],
        execution_requirements=[],
        risk_envelope={},
        cross_department_dependencies={},
    )


def _make_engine():
    comm = MagicMock()
    comm.subscribe_to_messages = AsyncMock(return_value=None)
    comm.unsubscribe_from_messages = AsyncMock(return_value=None)
    audit = MagicMock()
    audit.log_event = AsyncMock(return_value=None)
    return StrategyExecutionEngine(comm, audit), comm, audit


def _mock_strategy(name="X", status_value="active"):
    s = MagicMock()
    s.config = MagicMock()
    s.config.name = name
    s.status = MagicMock()
    s.status.value = status_value
    s.position_size = 0.0
    s.unrealized_pnl = 0.0
    s.last_signal_time = None
    s.initialize = AsyncMock(return_value=True)
    s.shutdown = AsyncMock(return_value=True)
    s.process_market_data = AsyncMock(return_value=[])
    return s


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestEngineInit:
    def test_starts_empty(self):
        eng, _, _ = _make_engine()
        assert eng.strategies == {}
        assert eng.strategy_configs == {}
        assert eng.market_data_subscriptions == set()

    def test_default_paths(self):
        eng, _, _ = _make_engine()
        assert eng.csv_path.name == "50_arbitrage_strategies.csv"
        assert eng.config_path.name == "strategy_department_matrix.yaml"


# ---------------------------------------------------------------------------
# initialize / shutdown
# ---------------------------------------------------------------------------


class TestInitializeShutdown:
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        eng, _, audit = _make_engine()
        eng._load_strategy_definitions = AsyncMock(return_value=None)
        eng._load_strategy_configs = AsyncMock(return_value=None)
        eng._instantiate_strategies = AsyncMock(return_value=None)
        eng._setup_market_data_routing = AsyncMock(return_value=None)
        ok = await eng.initialize()
        assert ok is True
        audit.log_event.assert_awaited()

    @pytest.mark.asyncio
    async def test_initialize_failure_returns_false(self):
        eng, _, _ = _make_engine()
        eng._load_strategy_definitions = AsyncMock(side_effect=RuntimeError("boom"))
        ok = await eng.initialize()
        assert ok is False

    @pytest.mark.asyncio
    async def test_shutdown_calls_each_strategy(self):
        eng, _, audit = _make_engine()
        a = _mock_strategy()
        b = _mock_strategy()
        eng.strategies = {"a": a, "b": b}
        ok = await eng.shutdown()
        assert ok is True
        a.shutdown.assert_awaited_once()
        b.shutdown.assert_awaited_once()
        assert eng.strategies == {}
        audit.log_event.assert_awaited()

    @pytest.mark.asyncio
    async def test_shutdown_failure_returns_false(self):
        eng, _, _ = _make_engine()
        bad = _mock_strategy()
        bad.shutdown = AsyncMock(side_effect=RuntimeError("nope"))
        eng.strategies = {"bad": bad}
        ok = await eng.shutdown()
        assert ok is False


# ---------------------------------------------------------------------------
# process_market_data
# ---------------------------------------------------------------------------


class TestProcessMarketData:
    @pytest.mark.asyncio
    async def test_no_relevant_strategies(self):
        eng, _, _ = _make_engine()
        out = await eng.process_market_data({"type": "etf_price"})
        assert out == []

    @pytest.mark.asyncio
    async def test_routes_to_relevant(self):
        eng, _, audit = _make_engine()
        cfg = _make_config(strategy_id="s01", data_requirements=["etf_prices"])
        eng.strategy_configs = {"s01": cfg}
        s = _mock_strategy()
        s.process_market_data = AsyncMock(return_value=["sig1", "sig2"])
        eng.strategies = {"s01": s}
        out = await eng.process_market_data({"type": "etf_price"})
        assert out == ["sig1", "sig2"]
        s.process_market_data.assert_awaited_once()
        # audit logged signal generation
        audit.log_event.assert_awaited()

    @pytest.mark.asyncio
    async def test_skips_missing_strategy_implementations(self):
        eng, _, _ = _make_engine()
        # config exists but no strategy instance
        eng.strategy_configs = {"s01": _make_config("s01", ["etf_prices"])}
        eng.strategies = {}
        out = await eng.process_market_data({"type": "etf_price"})
        assert out == []

    @pytest.mark.asyncio
    async def test_swallows_strategy_exceptions(self):
        eng, _, _ = _make_engine()
        cfg = _make_config("s01", ["etf_prices"])
        eng.strategy_configs = {"s01": cfg}
        s = _mock_strategy()
        s.process_market_data = AsyncMock(side_effect=RuntimeError("kaboom"))
        eng.strategies = {"s01": s}
        out = await eng.process_market_data({"type": "etf_price"})
        # gather with return_exceptions absorbs
        assert out == []

    @pytest.mark.asyncio
    async def test_outer_exception_returns_empty(self):
        eng, _, _ = _make_engine()
        eng._get_relevant_strategies = MagicMock(side_effect=RuntimeError("boom"))
        out = await eng.process_market_data({"type": "etf_price"})
        assert out == []


# ---------------------------------------------------------------------------
# get_strategy_status (second definition wins — includes not_implemented)
# ---------------------------------------------------------------------------


class TestGetStrategyStatus:
    @pytest.mark.asyncio
    async def test_includes_active_strategies(self):
        eng, _, _ = _make_engine()
        cfg = _make_config("s01")
        eng.strategy_configs = {"s01": cfg}
        s = _mock_strategy(name="MyStrat", status_value="active")
        s.position_size = 12.5
        s.unrealized_pnl = -3.0
        eng.strategies = {"s01": s}
        out = await eng.get_strategy_status()
        assert out["s01"]["name"] == "X"  # name pulled from config
        assert out["s01"]["status"] == "active"
        assert out["s01"]["position_size"] == 12.5
        assert out["s01"]["unrealized_pnl"] == -3.0

    @pytest.mark.asyncio
    async def test_includes_not_implemented_definitions(self):
        eng, _, _ = _make_engine()
        eng.strategy_configs = {"s01": _make_config("s01"), "s02": _make_config("s02")}
        # only s01 has an instance
        eng.strategies = {"s01": _mock_strategy()}
        out = await eng.get_strategy_status()
        assert "s02" in out
        assert out["s02"]["status"] == "not_implemented"
        assert out["s02"]["position_size"] == 0.0


# ---------------------------------------------------------------------------
# activate_strategy / deactivate_strategy
# ---------------------------------------------------------------------------


class TestActivateDeactivate:
    @pytest.mark.asyncio
    async def test_activate_known(self):
        eng, _, _ = _make_engine()
        s = _mock_strategy()
        eng.strategies = {"s01": s}
        ok = await eng.activate_strategy("s01")
        assert ok is True
        s.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_activate_unknown(self):
        eng, _, _ = _make_engine()
        ok = await eng.activate_strategy("missing")
        assert ok is False

    @pytest.mark.asyncio
    async def test_deactivate_known(self):
        eng, _, _ = _make_engine()
        s = _mock_strategy()
        eng.strategies = {"s01": s}
        ok = await eng.deactivate_strategy("s01")
        assert ok is True
        s.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_deactivate_unknown(self):
        eng, _, _ = _make_engine()
        assert await eng.deactivate_strategy("missing") is False


# ---------------------------------------------------------------------------
# _load_strategy_definitions (CSV)
# ---------------------------------------------------------------------------


class TestLoadStrategyDefinitions:
    @pytest.mark.asyncio
    async def test_loads_rows(self, tmp_path):
        csv_file = tmp_path / "strats.csv"
        csv_file.write_text(
            "id,strategy_name\n"
            "1,ETF NAV Arb\n"
            "2,Vol Premium\n",
            encoding="utf-8",
        )
        eng, _, _ = _make_engine()
        eng.csv_path = csv_file
        await eng._load_strategy_definitions()
        assert len(eng.strategy_configs) == 2
        # ID format: s + zero-padded id + "_" + sanitized name
        keys = list(eng.strategy_configs.keys())
        assert any(k.startswith("s01_") for k in keys)
        assert any(k.startswith("s02_") for k in keys)

    @pytest.mark.asyncio
    async def test_missing_file_raises(self, tmp_path):
        eng, _, _ = _make_engine()
        eng.csv_path = tmp_path / "absent.csv"
        with pytest.raises(Exception):
            await eng._load_strategy_definitions()


# ---------------------------------------------------------------------------
# _load_strategy_configs (YAML)
# ---------------------------------------------------------------------------


class TestLoadStrategyConfigs:
    @pytest.mark.asyncio
    async def test_updates_existing_config(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            "strategies:\n"
            "  s01_test:\n"
            "    strategy_type: vrp\n"
            "    edge_source: vol_premium\n"
            "    time_horizon: weekly\n"
            "    complexity: high\n"
            "    data_requirements: [etf_prices, options_data]\n"
            "    execution_requirements: [ibkr]\n"
            "    risk_envelope:\n"
            "      max_position_pct: 5.0\n"
            "    cross_department_dependencies:\n"
            "      bigbrain: [iv_curve]\n",
            encoding="utf-8",
        )
        eng, _, _ = _make_engine()
        eng.config_path = yaml_file
        eng.strategy_configs = {"s01_test": _make_config("s01_test")}
        await eng._load_strategy_configs()
        cfg = eng.strategy_configs["s01_test"]
        assert cfg.strategy_type == "vrp"
        assert cfg.complexity == "high"
        assert cfg.data_requirements == ["etf_prices", "options_data"]
        assert cfg.risk_envelope["max_position_pct"] == 5.0
        assert cfg.cross_department_dependencies["bigbrain"] == ["iv_curve"]

    @pytest.mark.asyncio
    async def test_skips_unknown_strategy_keys(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            "strategies:\n  unknown_strat:\n    strategy_type: foo\n",
            encoding="utf-8",
        )
        eng, _, _ = _make_engine()
        eng.config_path = yaml_file
        eng.strategy_configs = {}
        await eng._load_strategy_configs()  # no error
        assert eng.strategy_configs == {}

    @pytest.mark.asyncio
    async def test_missing_file_does_not_raise(self, tmp_path):
        eng, _, _ = _make_engine()
        eng.config_path = tmp_path / "absent.yaml"
        # YAML is optional — should not raise
        await eng._load_strategy_configs()


# ---------------------------------------------------------------------------
# _instantiate_strategies
# ---------------------------------------------------------------------------


class TestInstantiateStrategies:
    @pytest.mark.asyncio
    async def test_creates_and_initializes(self):
        eng, _, _ = _make_engine()
        eng.strategy_configs = {"s01": _make_config("s01")}
        good = _mock_strategy()
        with patch("shared.strategy_execution_engine.StrategyFactory") as F:
            F.create_strategy = AsyncMock(return_value=good)
            await eng._instantiate_strategies()
        assert "s01" in eng.strategies
        good.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_when_factory_returns_none(self):
        eng, _, _ = _make_engine()
        eng.strategy_configs = {"s01": _make_config("s01")}
        with patch("shared.strategy_execution_engine.StrategyFactory") as F:
            F.create_strategy = AsyncMock(return_value=None)
            await eng._instantiate_strategies()
        assert eng.strategies == {}

    @pytest.mark.asyncio
    async def test_skips_when_initialize_fails(self):
        eng, _, _ = _make_engine()
        eng.strategy_configs = {"s01": _make_config("s01")}
        bad = _mock_strategy()
        bad.initialize = AsyncMock(return_value=False)
        with patch("shared.strategy_execution_engine.StrategyFactory") as F:
            F.create_strategy = AsyncMock(return_value=bad)
            await eng._instantiate_strategies()
        assert eng.strategies == {}

    @pytest.mark.asyncio
    async def test_swallows_factory_exception(self):
        eng, _, _ = _make_engine()
        eng.strategy_configs = {"s01": _make_config("s01")}
        with patch("shared.strategy_execution_engine.StrategyFactory") as F:
            F.create_strategy = AsyncMock(side_effect=RuntimeError("kaboom"))
            await eng._instantiate_strategies()
        assert eng.strategies == {}


# ---------------------------------------------------------------------------
# _setup_market_data_routing
# ---------------------------------------------------------------------------


class TestSetupMarketDataRouting:
    @pytest.mark.asyncio
    async def test_subscribes_for_each_data_requirement(self):
        eng, comm, _ = _make_engine()
        eng.strategy_configs = {
            "s01": _make_config("s01", ["etf_prices"]),
            "s02": _make_config("s02", ["nav_calculations", "options_data"]),
            "s03": _make_config("s03", ["index_futures", "crypto_prices"]),
        }
        await eng._setup_market_data_routing()
        topics_subscribed = []
        for call in comm.subscribe_to_messages.await_args_list:
            topics_subscribed.extend(call.args[1])
        assert "market_data.etf.*" in topics_subscribed
        assert "bigbrain.nav.*" in topics_subscribed
        assert "market_data.futures.*" in topics_subscribed
        assert "market_data.crypto.*" in topics_subscribed
        assert "market_data.options.*" in topics_subscribed

    @pytest.mark.asyncio
    async def test_unknown_requirement_skipped(self):
        eng, comm, _ = _make_engine()
        eng.strategy_configs = {"s01": _make_config("s01", ["mystery_data"])}
        await eng._setup_market_data_routing()
        comm.subscribe_to_messages.assert_not_called()

    @pytest.mark.asyncio
    async def test_swallows_subscribe_errors(self):
        eng, comm, _ = _make_engine()
        comm.subscribe_to_messages = AsyncMock(side_effect=RuntimeError("nope"))
        eng.strategy_configs = {"s01": _make_config("s01", ["etf_prices"])}
        # Should not raise
        await eng._setup_market_data_routing()


# ---------------------------------------------------------------------------
# _get_relevant_strategies / _strategy_needs_data
# ---------------------------------------------------------------------------


class TestRelevantStrategies:
    def test_etf_price_routing(self):
        eng, _, _ = _make_engine()
        eng.strategy_configs = {
            "s01": _make_config("s01", ["etf_prices"]),
            "s02": _make_config("s02", ["nav_calculations"]),
        }
        out = eng._get_relevant_strategies({"type": "etf_price"})
        assert out == ["s01"]

    def test_nav_calculation_routing(self):
        eng, _, _ = _make_engine()
        eng.strategy_configs = {"s01": _make_config("s01", ["nav_calculations"])}
        assert eng._get_relevant_strategies({"type": "nav_calculation"}) == ["s01"]

    def test_futures_price_routing(self):
        eng, _, _ = _make_engine()
        eng.strategy_configs = {"s01": _make_config("s01", ["index_futures"])}
        assert eng._get_relevant_strategies({"type": "futures_price"}) == ["s01"]

    def test_crypto_routing(self):
        eng, _, _ = _make_engine()
        eng.strategy_configs = {"s01": _make_config("s01", ["crypto_prices"])}
        assert eng._get_relevant_strategies({"type": "crypto_price"}) == ["s01"]

    def test_options_routing(self):
        eng, _, _ = _make_engine()
        eng.strategy_configs = {"s01": _make_config("s01", ["options_data"])}
        assert eng._get_relevant_strategies({"type": "options_data"}) == ["s01"]

    def test_unknown_data_type_returns_empty(self):
        eng, _, _ = _make_engine()
        eng.strategy_configs = {"s01": _make_config("s01", ["etf_prices"])}
        assert eng._get_relevant_strategies({"type": "weather"}) == []


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------


class TestGetStrategyExecutionEngine:
    @pytest.mark.asyncio
    async def test_returns_initialized_engine(self):
        comm = MagicMock()
        audit = MagicMock()
        with patch.object(StrategyExecutionEngine, "initialize", new=AsyncMock(return_value=True)):
            eng = await get_strategy_execution_engine(comm, audit)
        assert isinstance(eng, StrategyExecutionEngine)

    @pytest.mark.asyncio
    async def test_raises_when_init_fails(self):
        comm = MagicMock()
        audit = MagicMock()
        with patch.object(StrategyExecutionEngine, "initialize", new=AsyncMock(return_value=False)):
            with pytest.raises(RuntimeError):
                await get_strategy_execution_engine(comm, audit)
