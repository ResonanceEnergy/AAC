#!/usr/bin/env python3
"""
Tests for Knightsbridge FX / Foreign Exchange integration.

Covers:
  - KnightsbridgeFXClient offline mode (no API key)
  - FXRate construction and spread calculation
  - Triangular arbitrage scanner
  - Spread comparison vs bank rates
  - ForexDataSource MarketTick emission
  - ForexArbitrageStrategy signal generation
  - Config fields for FX
  - Strategy mapping registration
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


# ─── Config ──────────────────────────────────────────────────────────────

def test_config_has_fx_fields():
    """Config dataclass includes FX fields."""
    from shared.config_loader import Config
    cfg = Config()
    assert hasattr(cfg, 'fx_api_key')
    assert hasattr(cfg, 'fx_spread_bps')
    assert hasattr(cfg, 'fx_poll_interval')
    assert cfg.fx_spread_bps == 50.0
    assert cfg.fx_poll_interval == 60


# ─── FXRate ──────────────────────────────────────────────────────────────

def test_fx_rate_construction():
    """FXRate computes mid and spread correctly."""
    from integrations.knightsbridge_fx_client import FXRate
    rate = FXRate("USD/CAD", bid=1.3550, ask=1.3630)
    assert rate.pair == "USD/CAD"
    assert rate.bid == 1.3550
    assert rate.ask == 1.3630
    assert abs(rate.mid - 1.3590) < 0.001
    assert rate.spread_bps > 0
    assert rate.source == "exchangerate-api"


def test_fx_rate_repr():
    """FXRate repr is readable."""
    from integrations.knightsbridge_fx_client import FXRate
    rate = FXRate("EUR/USD", bid=0.9150, ask=0.9210)
    text = repr(rate)
    assert "EUR/USD" in text
    assert "bid=" in text
    assert "spread=" in text


# ─── KnightsbridgeFXClient (offline) ────────────────────────────────────

def test_client_offline_rates_usd():
    """Offline demo rates for USD base."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    client = KnightsbridgeFXClient()  # no api_key → offline
    rates = client._offline_demo_rates("USD")
    assert len(rates) > 10
    assert "USD/CAD" in rates
    assert "USD/EUR" in rates
    assert "USD/JPY" in rates
    assert rates["USD/CAD"].bid < rates["USD/CAD"].ask


def test_client_offline_rates_cad():
    """Offline demo rates for CAD base."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    client = KnightsbridgeFXClient()
    rates = client._offline_demo_rates("CAD")
    assert len(rates) > 5
    assert "CAD/USD" in rates
    assert "CAD/EUR" in rates


def test_client_offline_rates_unknown_base():
    """Unknown base returns empty dict."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    client = KnightsbridgeFXClient()
    rates = client._offline_demo_rates("XYZ")
    assert rates == {}


def test_client_supported_currencies():
    """Client returns 30+ supported currencies."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    client = KnightsbridgeFXClient()
    ccys = client.get_supported_currencies()
    assert len(ccys) >= 30
    assert "USD" in ccys
    assert "CAD" in ccys
    assert "UYU" in ccys


def test_client_supported_pairs():
    """Client returns all tracked FX pairs."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    client = KnightsbridgeFXClient()
    pairs = client.get_supported_pairs()
    assert len(pairs) >= 20
    assert "USD/CAD" in pairs
    assert "EUR/USD" in pairs
    assert "USD/UYU" in pairs


# ─── Async Client Tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_rates_offline():
    """get_rates() returns offline rates when no API key."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    client = KnightsbridgeFXClient()
    rates = await client.get_rates("USD")
    assert len(rates) > 10
    assert "USD/CAD" in rates


@pytest.mark.asyncio
async def test_get_pair_offline():
    """get_pair() returns offline rate when no API key."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    client = KnightsbridgeFXClient()
    rate = await client.get_pair("USD", "CAD")
    assert rate is not None
    assert rate.pair == "USD/CAD"
    assert rate.bid > 0


@pytest.mark.asyncio
async def test_get_cad_rates():
    """get_cad_rates() fetches CAD-centric rates."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    client = KnightsbridgeFXClient()
    rates = await client.get_cad_rates()
    assert len(rates) > 5
    assert "CAD/USD" in rates


@pytest.mark.asyncio
async def test_get_uyu_rates():
    """get_uyu_rates() fetches UYU-centric rates."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    client = KnightsbridgeFXClient()
    rates = await client.get_uyu_rates()
    assert len(rates) > 5


@pytest.mark.asyncio
async def test_triangular_arb_scanner():
    """Triangular arb scanner runs without errors on offline data."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    client = KnightsbridgeFXClient()
    opps = await client.find_triangular_arb("USD", min_profit_bps=0.01)
    # Offline demo rates are self-consistent, so arb opps may be empty
    # but the scanner itself should not crash
    assert isinstance(opps, list)


@pytest.mark.asyncio
async def test_compare_spreads():
    """Spread comparison vs bank rates."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    client = KnightsbridgeFXClient()
    results = await client.compare_spreads()
    assert isinstance(results, list)
    # Should find at least some of the CAD pairs
    if results:
        r = results[0]
        assert "pair" in r
        assert "knightsbridge_spread_bps" in r
        assert "bank_spread_bps" in r
        assert "savings_bps" in r


@pytest.mark.asyncio
async def test_client_cache():
    """Rates cache returns equal data on second call."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    client = KnightsbridgeFXClient(api_key="test_key")  # non-empty key to enable caching path
    # Seed the cache manually
    demo_rates = client._offline_demo_rates("USD")
    client._cache_set("rates_USD", demo_rates)
    r1 = await client.get_rates("USD")
    r2 = await client.get_rates("USD")
    assert r1 is r2


@pytest.mark.asyncio
async def test_client_connect_disconnect():
    """Client connect/disconnect lifecycle."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    client = KnightsbridgeFXClient()
    await client.connect()
    assert client._session is not None
    await client.disconnect()
    assert client._session is None


@pytest.mark.asyncio
async def test_client_context_manager():
    """Client works as async context manager."""
    from integrations.knightsbridge_fx_client import KnightsbridgeFXClient
    async with KnightsbridgeFXClient() as client:
        assert client._session is not None
    assert client._session is None


# ─── ForexDataSource ─────────────────────────────────────────────────────

def test_forex_data_source_init():
    """ForexDataSource initializes with KnightsbridgeFXClient."""
    from shared.forex_data_source import ForexDataSource
    ds = ForexDataSource()
    assert ds.source_id == "forex"
    assert ds._client is not None


@pytest.mark.asyncio
async def test_forex_data_source_get_fx_tick():
    """ForexDataSource.get_fx_tick returns a MarketTick."""
    from shared.forex_data_source import ForexDataSource
    from shared.data_sources import MarketTick
    ds = ForexDataSource()
    tick = await ds.get_fx_tick("USD", "CAD")
    assert tick is not None
    assert isinstance(tick, MarketTick)
    assert tick.symbol == "USD/CAD"
    assert tick.bid is not None
    assert tick.ask is not None


@pytest.mark.asyncio
async def test_forex_data_source_get_all_rates():
    """ForexDataSource.get_all_rates returns dict of MarketTicks."""
    from shared.forex_data_source import ForexDataSource
    ds = ForexDataSource()
    ticks = await ds.get_all_rates("USD")
    assert len(ticks) > 10
    assert "USD/CAD" in ticks


# ─── ForexArbitrageStrategy ─────────────────────────────────────────────

def test_forex_strategy_init():
    """ForexArbitrageStrategy initializes correctly."""
    from strategies.forex_arb_strategy import ForexArbitrageStrategy
    from shared.strategy_framework import StrategyConfig

    config = StrategyConfig(
        strategy_id="53",
        name="Forex Triangular Arbitrage",
        strategy_type="FX_Arbitrage",
        edge_source="triangular_arb",
        time_horizon="intraday",
        complexity="medium",
        data_requirements=["fx_rates"],
        execution_requirements=["fx_broker"],
        risk_envelope={"min_profit_bps": 10, "max_position_usd": 50000},
        cross_department_dependencies={},
    )
    comm = MagicMock()
    audit = MagicMock()
    strategy = ForexArbitrageStrategy(config, comm, audit)
    assert strategy.min_profit_bps == 10
    assert strategy.max_position_usd == 50000
    assert strategy.cad_weight == 1.5


@pytest.mark.asyncio
async def test_forex_strategy_generate_signals_empty():
    """No signals when no market data loaded."""
    from strategies.forex_arb_strategy import ForexArbitrageStrategy
    from shared.strategy_framework import StrategyConfig

    config = StrategyConfig(
        strategy_id="53",
        name="Forex Triangular Arbitrage",
        strategy_type="FX_Arbitrage",
        edge_source="triangular_arb",
        time_horizon="intraday",
        complexity="medium",
        data_requirements=["fx_rates"],
        execution_requirements=["fx_broker"],
        risk_envelope={},
        cross_department_dependencies={},
    )
    comm = MagicMock()
    audit = MagicMock()
    strategy = ForexArbitrageStrategy(config, comm, audit)
    signals = await strategy._generate_signals()
    assert signals == []


@pytest.mark.asyncio
async def test_forex_strategy_generate_signals_with_arb():
    """Generates signals when triangular arb opportunities exist."""
    from strategies.forex_arb_strategy import ForexArbitrageStrategy
    from shared.strategy_framework import StrategyConfig

    config = StrategyConfig(
        strategy_id="53",
        name="Forex Triangular Arbitrage",
        strategy_type="FX_Arbitrage",
        edge_source="triangular_arb",
        time_horizon="intraday",
        complexity="medium",
        data_requirements=["fx_rates"],
        execution_requirements=["fx_broker"],
        risk_envelope={"min_profit_bps": 5},
        cross_department_dependencies={},
    )
    comm = MagicMock()
    audit = MagicMock()
    strategy = ForexArbitrageStrategy(config, comm, audit)

    # Inject a fake arb opportunity
    strategy.market_data["fx_arb_opportunities"] = [
        {
            "path": "USD→EUR→GBP→USD",
            "profit_bps": 12.5,
            "direction": "forward",
            "leg_1": "USD/EUR @ 0.91800",
            "leg_2": "EUR/GBP implied @ 0.85900",
            "leg_3": "GBP/USD @ 1.26740",
        }
    ]
    signals = await strategy._generate_signals()
    assert len(signals) == 1
    assert "fx_tri_arb" in signals[0].metadata["signal_tag"]
    assert signals[0].metadata["strategy"] == "forex_triangular_arb"


@pytest.mark.asyncio
async def test_forex_strategy_cad_weight_boost():
    """CAD corridor arbs get boosted priority."""
    from strategies.forex_arb_strategy import ForexArbitrageStrategy
    from shared.strategy_framework import StrategyConfig

    config = StrategyConfig(
        strategy_id="53",
        name="Forex Triangular Arbitrage",
        strategy_type="FX_Arbitrage",
        edge_source="triangular_arb",
        time_horizon="intraday",
        complexity="medium",
        data_requirements=["fx_rates"],
        execution_requirements=["fx_broker"],
        risk_envelope={"min_profit_bps": 5, "cad_corridor_weight": 2.0},
        cross_department_dependencies={},
    )
    comm = MagicMock()
    audit = MagicMock()
    strategy = ForexArbitrageStrategy(config, comm, audit)
    assert strategy.cad_weight == 2.0

    # CAD arb
    strategy.market_data["fx_arb_opportunities"] = [
        {
            "path": "USD→CAD→EUR→USD",
            "profit_bps": 8.0,
            "direction": "forward",
            "leg_1": "USD/CAD @ 1.35900",
            "leg_2": "CAD/EUR implied @ 0.67570",
            "leg_3": "EUR/USD @ 1.08932",
        }
    ]
    signals = await strategy._generate_signals()
    assert len(signals) == 1
    assert signals[0].metadata["cad_corridor"] is True
    assert signals[0].metadata["adjusted_profit_bps"] == 16.0  # 8 * 2.0


# ─── Strategy Mapping ───────────────────────────────────────────────────

def test_strategy_mapping_has_forex():
    """Strategy mapping includes FX strategies (IDs 53-55)."""
    from strategies.strategy_agent_master_mapping import STRATEGIES_DATA
    ids = {s["id"] for s in STRATEGIES_DATA}
    assert 53 in ids
    assert 54 in ids
    assert 55 in ids

    fx_strategies = [s for s in STRATEGIES_DATA if s["category"] == "FX_Arbitrage"]
    assert len(fx_strategies) == 3
    names = {s["name"] for s in fx_strategies}
    assert "Forex Triangular Arbitrage" in names
    assert "Forex Spread Dislocation" in names
    assert "CAD Corridor FX Optimization" in names


def test_strategy_count():
    """Total strategy count is 56 (49 original + 3 Metallicus + 3 FX + 1 Matrix Maximizer)."""
    from strategies.strategy_agent_master_mapping import STRATEGIES_DATA
    assert len(STRATEGIES_DATA) == 56


# ─── DataAggregator Registration ────────────────────────────────────────

def test_data_aggregator_has_forex():
    """DataAggregator has forex data source registered."""
    from shared.data_sources import DataAggregator
    agg = DataAggregator()
    assert hasattr(agg, 'forex')
    assert agg.forex.source_id == "forex"
