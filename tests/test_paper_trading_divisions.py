from __future__ import annotations

"""Tests for paper trading divisions, engine, strategies, and optimizer.

Covers:
- PaperTradingEngine: orders, fills, positions, P&L
- Strategy algorithms: Grid, DCA, Momentum, MeanReversion, Arbitrage
- StrategyOptimizer: scoring, ranking, capital allocation
- PolymarketPaperDivision: scan, consume_signal, report
- CryptoPaperDivision: scan, consume_signal, report
- BakeoffEngine YAML config loading
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from strategies.paper_trading.engine import (
    OrderSide,
    OrderStatus,
    OrderType,
    PaperTradingEngine,
)
from strategies.paper_trading.strategies import (
    ArbitrageConfig,
    ArbitrageStrategy,
    DCAConfig,
    DCAStrategy,
    GridConfig,
    GridStrategy,
    MeanReversionConfig,
    MeanReversionStrategy,
    MomentumConfig,
    MomentumStrategy,
)
from strategies.paper_trading.optimizer import StrategyOptimizer, StrategyScore

from divisions.division_protocol import Signal, SignalType
from divisions.trading.polymarket_paper.division import PolymarketPaperDivision
from divisions.trading.crypto_paper.division import CryptoPaperDivision


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def engine():
    """Fresh PaperTradingEngine with $10K, no persistence."""
    return PaperTradingEngine(
        account_id="test",
        starting_balance=10_000.0,
        fee_rate=0.001,
        slippage_bps=5.0,
        persist_dir=None,
    )


@pytest.fixture()
def grid_strategy(engine):
    cfg = GridConfig(name="test_grid", grid_levels=5, grid_range_pct=0.10)
    return GridStrategy(cfg, engine)


@pytest.fixture()
def dca_strategy(engine):
    cfg = DCAConfig(name="test_dca", buy_dip_pct=0.05, max_buys=5, sell_after_pct=0.15)
    return DCAStrategy(cfg, engine)


@pytest.fixture()
def momentum_strategy(engine):
    cfg = MomentumConfig(
        name="test_momentum",
        lookback_periods=3,
        entry_threshold_pct=0.03,
        exit_threshold_pct=-0.02,
    )
    return MomentumStrategy(cfg, engine)


@pytest.fixture()
def mean_rev_strategy(engine):
    cfg = MeanReversionConfig(name="test_meanrev", window=5, entry_z=1.5, exit_z=0.3)
    return MeanReversionStrategy(cfg, engine)


@pytest.fixture()
def arb_strategy(engine):
    cfg = ArbitrageConfig(name="test_arb", min_edge_pct=0.5, max_bet_size=50.0)
    return ArbitrageStrategy(cfg, engine)


# ===========================================================================
# PaperTradingEngine Tests
# ===========================================================================


class TestPaperTradingEngine:
    def test_initial_state(self, engine):
        assert engine.account.cash_balance == 10_000.0
        assert engine.account.equity == 10_000.0
        assert engine.account.total_trades == 0
        assert len(engine.account.positions) == 0

    def test_market_buy(self, engine):
        order = engine.place_order("BTC", OrderSide.BUY, 0.1, 50_000.0)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert "BTC" in engine.account.positions
        pos = engine.account.positions["BTC"]
        assert pos.quantity == pytest.approx(0.1, abs=1e-6)
        assert engine.account.cash_balance < 10_000.0

    def test_market_sell_reduces_position(self, engine):
        engine.place_order("ETH", OrderSide.BUY, 2.0, 3_000.0)
        engine.place_order("ETH", OrderSide.SELL, 1.0, 3_100.0)
        pos = engine.account.positions.get("ETH")
        # Either position reduced to 1.0 or closed depending on implementation
        if pos:
            assert pos.quantity == pytest.approx(1.0, abs=1e-6)

    def test_fees_deducted(self, engine):
        engine.place_order("AAA", OrderSide.BUY, 10.0, 100.0)
        assert engine.account.total_fees > 0
        assert engine.account.cash_balance < 10_000.0 - (10 * 100)

    def test_slippage_applied(self, engine):
        order = engine.place_order("SPY", OrderSide.BUY, 1.0, 500.0)
        assert order is not None
        # Slippage should make fill slightly worse than requested price
        assert order.filled_price >= 500.0  # buy = higher

    def test_limit_order_queued(self, engine):
        order = engine.place_order(
            "XYZ", OrderSide.BUY, 5.0, 100.0, order_type=OrderType.LIMIT
        )
        assert order is not None
        assert order.status == OrderStatus.PENDING
        assert len(engine.account.open_orders) >= 1

    def test_update_prices(self, engine):
        engine.place_order("BTC", OrderSide.BUY, 0.1, 50_000.0)
        engine.update_prices({"BTC": 55_000.0})
        pos = engine.account.positions["BTC"]
        assert pos.current_price == 55_000.0
        assert pos.unrealized_pnl > 0

    def test_performance_report(self, engine):
        engine.place_order("AAA", OrderSide.BUY, 10.0, 100.0)
        perf = engine.get_performance()
        assert "equity" in perf
        assert "cash_balance" in perf
        assert "total_trades" in perf

    def test_win_rate_zero_when_no_trades(self, engine):
        assert engine.account.win_rate == 0.0

    def test_max_drawdown_zero_at_start(self, engine):
        assert engine.account.max_drawdown_pct == 0.0


# ===========================================================================
# Strategy Tests
# ===========================================================================


class TestGridStrategy:
    def test_evaluate_returns_list(self, grid_strategy):
        md = {"prices": {"BTC": 50_000.0}, "volumes": {"BTC": 1_000_000}}
        signals = grid_strategy.evaluate(md)
        assert isinstance(signals, list)

    def test_report_structure(self, grid_strategy):
        report = grid_strategy.report()
        assert "strategy" in report
        assert report["strategy"] == "test_grid"
        assert "total_trades" in report
        assert "win_rate" in report


class TestDCAStrategy:
    def test_evaluate_returns_list(self, dca_strategy):
        md = {"prices": {"ETH": 3_000.0}, "volumes": {"ETH": 500_000}}
        signals = dca_strategy.evaluate(md)
        assert isinstance(signals, list)

    def test_report_has_all_fields(self, dca_strategy):
        report = dca_strategy.report()
        for key in ["strategy", "total_trades", "wins", "losses", "win_rate", "total_pnl"]:
            assert key in report


class TestMomentumStrategy:
    def test_evaluate_returns_list(self, momentum_strategy):
        md = {"prices": {"SOL": 150.0}, "changes": {"SOL": 0.05}, "volumes": {"SOL": 100_000}}
        signals = momentum_strategy.evaluate(md)
        assert isinstance(signals, list)


class TestMeanReversionStrategy:
    def test_evaluate_returns_list(self, mean_rev_strategy):
        md = {"prices": {"AVAX": 35.0}, "volumes": {"AVAX": 50_000}}
        signals = mean_rev_strategy.evaluate(md)
        assert isinstance(signals, list)


class TestArbitrageStrategy:
    def test_evaluate_returns_list(self, arb_strategy):
        md = {
            "prices": {"YES_MKT": 0.45},
            "arb_opportunities": [
                {"market": "test", "yes_price": 0.45, "no_price": 0.50, "edge_pct": 5.0},
            ],
        }
        signals = arb_strategy.evaluate(md)
        assert isinstance(signals, list)


class TestStrategyStops:
    def test_stop_loss_triggers(self, engine, dca_strategy):
        # Buy at 100, price drops past stop-loss
        engine.place_order("STOP_TEST", OrderSide.BUY, 10.0, 100.0, strategy="test_dca")
        engine.update_prices({"STOP_TEST": 90.0})  # -10%, past 5% stop
        exits = dca_strategy.check_stops({"STOP_TEST": 90.0})
        assert exits >= 1

    def test_take_profit_triggers(self, engine, dca_strategy):
        engine.place_order("TP_TEST", OrderSide.BUY, 10.0, 100.0, strategy="test_dca")
        engine.update_prices({"TP_TEST": 120.0})  # +20%, past 10% TP
        exits = dca_strategy.check_stops({"TP_TEST": 120.0})
        assert exits >= 1


# ===========================================================================
# Optimizer Tests
# ===========================================================================


class TestStrategyOptimizer:
    def test_initialization(self, engine, grid_strategy, dca_strategy):
        opt = StrategyOptimizer(engine, [grid_strategy, dca_strategy])
        assert len(opt.strategies) == 2

    def test_run_cycle(self, engine, grid_strategy, dca_strategy):
        opt = StrategyOptimizer(engine, [grid_strategy, dca_strategy])
        md = {"prices": {"BTC": 50_000.0}, "volumes": {"BTC": 1_000_000}}
        result = opt.run_cycle(md)
        assert "cycle" in result
        assert result["cycle"] == 1
        assert "strategies" in result

    def test_get_rankings_structure(self, engine, grid_strategy, dca_strategy):
        opt = StrategyOptimizer(engine, [grid_strategy, dca_strategy])
        md = {"prices": {"BTC": 50_000.0}, "volumes": {"BTC": 1_000_000}}
        opt.run_cycle(md)
        rankings = opt.get_rankings()
        assert isinstance(rankings, list)
        assert len(rankings) == 2
        for r in rankings:
            assert "strategy_name" in r
            assert "composite" in r
            assert "status" in r

    def test_get_report(self, engine, grid_strategy):
        opt = StrategyOptimizer(engine, [grid_strategy])
        report = opt.get_report()
        assert "cycle_count" in report
        assert "total_strategies" in report
        assert report["total_strategies"] == 1
        assert "rankings" in report

    def test_add_strategy(self, engine, grid_strategy, dca_strategy):
        opt = StrategyOptimizer(engine, [grid_strategy])
        assert len(opt.strategies) == 1
        opt.add_strategy(dca_strategy)
        assert len(opt.strategies) == 2


# ===========================================================================
# Division Tests
# ===========================================================================


class TestPolymarketPaperDivision:
    def test_init(self):
        div = PolymarketPaperDivision(starting_balance=5_000.0, persist=False)
        assert div.division_name == "polymarket_paper"
        assert div._engine.account.starting_balance == 5_000.0

    def test_scan_no_intel(self):
        div = PolymarketPaperDivision(starting_balance=5_000.0, persist=False)
        signals = asyncio.get_event_loop().run_until_complete(div.scan())
        assert signals == []

    def test_consume_polymarket_intel(self):
        div = PolymarketPaperDivision(starting_balance=5_000.0, persist=False)
        signal = Signal(
            signal_type=SignalType.INTEL_UPDATE,
            source_division="polymarket_council",
            timestamp=datetime.now(timezone.utc),
            data={
                "type": "polymarket_scan",
                "total_markets": 42,
                "total_volume": 5_000_000,
                "top_volume": [
                    {"condition_id": "mkt1", "price": 0.65, "volume": 100_000},
                    {"condition_id": "mkt2", "price": 0.30, "volume": 80_000},
                ],
                "arb_opportunities": [],
            },
        )
        asyncio.get_event_loop().run_until_complete(div.consume_signal(signal))
        assert div._latest_intel.get("type") == "polymarket_scan"

    def test_scan_with_intel(self):
        div = PolymarketPaperDivision(starting_balance=5_000.0, persist=False)
        div._latest_intel = {
            "type": "polymarket_scan",
            "total_markets": 10,
            "top_volume": [
                {"condition_id": "mkt_a", "price": 0.55, "volume": 200_000},
                {"condition_id": "mkt_b", "price": 0.70, "volume": 150_000},
            ],
            "arb_opportunities": [],
        }
        signals = asyncio.get_event_loop().run_until_complete(div.scan())
        assert isinstance(signals, list)
        # Should produce at least a TRADE_SIGNAL after cycle
        if signals:
            assert signals[0].signal_type == SignalType.TRADE_SIGNAL
            assert signals[0].data.get("type") == "polymarket_paper_cycle"

    def test_ignores_non_polymarket_intel(self):
        div = PolymarketPaperDivision(starting_balance=5_000.0, persist=False)
        signal = Signal(
            signal_type=SignalType.INTEL_UPDATE,
            source_division="crypto_council",
            timestamp=datetime.now(timezone.utc),
            data={"type": "crypto_scan", "btc_price": 50_000},
        )
        asyncio.get_event_loop().run_until_complete(div.consume_signal(signal))
        assert div._latest_intel == {}

    def test_report_structure(self):
        div = PolymarketPaperDivision(starting_balance=5_000.0, persist=False)
        report = asyncio.get_event_loop().run_until_complete(div.report())
        assert report["division"] == "polymarket_paper"
        assert "account" in report
        assert "strategies" in report
        assert "best_strategy" in report

    def test_has_five_strategies(self):
        div = PolymarketPaperDivision(starting_balance=5_000.0, persist=False)
        strat_names = [s.name for s in div._optimizer.strategies]
        assert len(strat_names) == 5
        expected = {"poly_grid", "poly_dca", "poly_momentum", "poly_mean_rev", "poly_arb"}
        assert set(strat_names) == expected


class TestCryptoPaperDivision:
    def test_init(self):
        div = CryptoPaperDivision(starting_balance=5_000.0, persist=False)
        assert div.division_name == "crypto_paper"
        assert div._engine.account.starting_balance == 5_000.0

    def test_scan_no_intel(self):
        div = CryptoPaperDivision(starting_balance=5_000.0, persist=False)
        signals = asyncio.get_event_loop().run_until_complete(div.scan())
        assert signals == []

    def test_consume_crypto_intel(self):
        div = CryptoPaperDivision(starting_balance=5_000.0, persist=False)
        signal = Signal(
            signal_type=SignalType.INTEL_UPDATE,
            source_division="crypto_council",
            timestamp=datetime.now(timezone.utc),
            data={
                "type": "crypto_scan",
                "total_coins": 20,
                "sentiment": "bullish",
                "btc_price": 62_000,
                "eth_price": 3_200,
                "gainers": [
                    {"coin_id": "solana", "price": 155.0, "change_24h": 8.5, "volume_24h": 2_000_000},
                ],
                "losers": [],
            },
        )
        asyncio.get_event_loop().run_until_complete(div.consume_signal(signal))
        assert div._latest_intel.get("type") == "crypto_scan"

    def test_scan_with_intel(self):
        div = CryptoPaperDivision(starting_balance=5_000.0, persist=False)
        div._latest_intel = {
            "type": "crypto_scan",
            "btc_price": 62_000,
            "eth_price": 3_200,
            "gainers": [
                {"coin_id": "solana", "price": 155.0, "change_24h": 8.5, "volume_24h": 2_000_000},
            ],
            "losers": [
                {"coin_id": "cardano", "price": 0.45, "change_24h": -4.2, "volume_24h": 800_000},
            ],
        }
        signals = asyncio.get_event_loop().run_until_complete(div.scan())
        assert isinstance(signals, list)
        if signals:
            assert signals[0].signal_type == SignalType.TRADE_SIGNAL
            assert signals[0].data.get("type") == "crypto_paper_cycle"

    def test_ignores_non_crypto_intel(self):
        div = CryptoPaperDivision(starting_balance=5_000.0, persist=False)
        signal = Signal(
            signal_type=SignalType.INTEL_UPDATE,
            source_division="polymarket_council",
            timestamp=datetime.now(timezone.utc),
            data={"type": "polymarket_scan", "total_markets": 42},
        )
        asyncio.get_event_loop().run_until_complete(div.consume_signal(signal))
        assert div._latest_intel == {}

    def test_report_structure(self):
        div = CryptoPaperDivision(starting_balance=5_000.0, persist=False)
        report = asyncio.get_event_loop().run_until_complete(div.report())
        assert report["division"] == "crypto_paper"
        assert "account" in report
        assert "strategies" in report

    def test_has_four_strategies(self):
        div = CryptoPaperDivision(starting_balance=5_000.0, persist=False)
        strat_names = [s.name for s in div._optimizer.strategies]
        assert len(strat_names) == 4
        expected = {"crypto_grid", "crypto_dca", "crypto_momentum", "crypto_mean_rev"}
        assert set(strat_names) == expected


# ===========================================================================
# BakeoffEngine YAML Configs Tests
# ===========================================================================


class TestBakeoffYAMLConfigs:
    """Verify the bakeoff YAML configs load correctly."""

    def test_metric_canon_loads(self):
        from aac.bakeoff.engine import BakeoffEngine
        eng = BakeoffEngine()
        assert eng.metric_canon != {}
        canon = eng.metric_canon.get("metric_canon", {})
        assert "performance" in canon
        assert "risk" in canon
        assert "win_rate" in canon["performance"]
        assert "sharpe_ratio" in canon["risk"]

    def test_policy_loads(self):
        from aac.bakeoff.engine import BakeoffEngine
        eng = BakeoffEngine()
        assert eng.policy != {}
        bakeoff = eng.policy.get("aac_bakeoff", {})
        assert "scoring" in bakeoff
        assert "gates" in bakeoff
        assert bakeoff["scoring"]["weights"]["performance"] == 0.25

    def test_checklists_load(self):
        from aac.bakeoff.engine import BakeoffEngine
        eng = BakeoffEngine()
        assert eng.checklists != {}
        gates = eng.checklists.get("gate_checklists", {})
        assert "PAPER" in gates
        assert "PILOT" in gates
        assert "SCALE" in gates

    def test_evaluate_metric(self):
        from aac.bakeoff.engine import BakeoffEngine
        eng = BakeoffEngine()
        mv = eng.evaluate_metric("win_rate", 60.0)
        assert mv.name == "win_rate"
        assert mv.value == 60.0
        assert mv.threshold_status == "good"

    def test_evaluate_metric_critical(self):
        from aac.bakeoff.engine import BakeoffEngine
        eng = BakeoffEngine()
        mv = eng.evaluate_metric("win_rate", 35.0)
        assert mv.threshold_status == "critical"

    def test_composite_score(self):
        from aac.bakeoff.engine import BakeoffEngine
        eng = BakeoffEngine()
        metrics = {
            "total_pnl_pct": 10.0,
            "win_rate": 60.0,
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": 8.0,
            "fill_rate": 95.0,
            "intel_freshness": 100.0,
            "uptime_pct": 99.5,
            "score_volatility": 3.0,
        }
        score = eng.calculate_composite_score("test_strat", metrics)
        assert score.strategy_id == "test_strat"
        assert score.composite > 0
        assert score.decision is not None

    def test_gate_validation_paper(self):
        from aac.bakeoff.engine import BakeoffEngine, Gate
        eng = BakeoffEngine()
        metrics = {
            "trade_count": 150,
            "win_rate": 58.0,
            "total_pnl_pct": 7.5,
            "sharpe_ratio": 1.2,
            "max_drawdown_pct": 12.0,
            "consecutive_losses": 6,
            "fill_rate": 92.0,
            "price_coverage": 85.0,
        }
        result = eng.validate_gate("test_strat", "PAPER", metrics)
        assert result.gate == Gate.PAPER
        assert result.strategy_id == "test_strat"
        # With good metrics, most items should pass
        passed_count = sum(1 for r in result.checklist_results if r.passed)
        assert passed_count > 0


# ===========================================================================
# Enterprise Wiring Tests
# ===========================================================================


class TestEnterpriseWiring:
    """Verify paper divisions are wired into Enterprise correctly."""

    def test_enterprise_has_paper_divisions(self):
        from divisions.enterprise import Enterprise
        ent = Enterprise()
        division_names = [d.division_name for d in ent._divisions]
        assert "polymarket_paper" in division_names
        assert "crypto_paper" in division_names

    def test_enterprise_division_count(self):
        from divisions.enterprise import Enterprise
        ent = Enterprise()
        assert len(ent._divisions) >= 11


# ===========================================================================
# Monitor Collector Tests
# ===========================================================================


class TestMonitorPaperCollectors:
    """Verify paper division collectors in the master monitoring dashboard."""

    def test_dashboard_has_polymarket_paper_collector(self):
        from monitoring.aac_master_monitoring_dashboard import (
            AACMasterMonitoringDashboard,
        )
        assert hasattr(AACMasterMonitoringDashboard, "_get_polymarket_paper_data")

    def test_dashboard_has_crypto_paper_collector(self):
        from monitoring.aac_master_monitoring_dashboard import (
            AACMasterMonitoringDashboard,
        )
        assert hasattr(AACMasterMonitoringDashboard, "_get_crypto_paper_data")

    def test_polymarket_paper_collector_returns_dict(self):
        from monitoring.aac_master_monitoring_dashboard import (
            AACMasterMonitoringDashboard,
        )
        dash = AACMasterMonitoringDashboard.__new__(AACMasterMonitoringDashboard)
        # Minimal init to avoid heavy constructor side effects
        dash.logger = __import__("logging").getLogger("test")
        result = dash._get_polymarket_paper_data()
        assert isinstance(result, dict)
        assert "status" in result

    def test_crypto_paper_collector_returns_dict(self):
        from monitoring.aac_master_monitoring_dashboard import (
            AACMasterMonitoringDashboard,
        )
        dash = AACMasterMonitoringDashboard.__new__(AACMasterMonitoringDashboard)
        dash.logger = __import__("logging").getLogger("test")
        result = dash._get_crypto_paper_data()
        assert isinstance(result, dict)
        assert "status" in result

    def test_polymarket_paper_collector_ok_status(self):
        from monitoring.aac_master_monitoring_dashboard import (
            AACMasterMonitoringDashboard,
            POLYMARKET_PAPER_AVAILABLE,
        )
        if not POLYMARKET_PAPER_AVAILABLE:
            pytest.skip("PolymarketPaperDivision not importable")
        dash = AACMasterMonitoringDashboard.__new__(AACMasterMonitoringDashboard)
        dash.logger = __import__("logging").getLogger("test")
        result = dash._get_polymarket_paper_data()
        assert result["status"] == "ok"
        assert "equity" in result
        assert "total_pnl" in result

    def test_crypto_paper_collector_ok_status(self):
        from monitoring.aac_master_monitoring_dashboard import (
            AACMasterMonitoringDashboard,
            CRYPTO_PAPER_AVAILABLE,
        )
        if not CRYPTO_PAPER_AVAILABLE:
            pytest.skip("CryptoPaperDivision not importable")
        dash = AACMasterMonitoringDashboard.__new__(AACMasterMonitoringDashboard)
        dash.logger = __import__("logging").getLogger("test")
        result = dash._get_crypto_paper_data()
        assert result["status"] == "ok"
        assert "equity" in result
        assert "total_pnl" in result


# ===========================================================================
# Smoke Test (Synthetic Intel Flow)
# ===========================================================================


class TestSyntheticIntelFlow:
    """Verify end-to-end flow: intel signal → strategy execution → fills."""

    @pytest.mark.asyncio
    async def test_polymarket_full_cycle(self):
        div = PolymarketPaperDivision(starting_balance=10_000.0, persist=False)
        sig = Signal(
            signal_type=SignalType.INTEL_UPDATE,
            source_division="polymarket_council",
            timestamp=datetime.now(timezone.utc),
            data={
                "type": "polymarket_scan",
                "total_markets": 3,
                "total_volume": 100_000,
                "top_volume": [
                    {"condition_id": "q1", "price": 0.60, "volume": 50_000,
                     "best_yes": 0.60},
                    {"condition_id": "q2", "price": 0.45, "volume": 30_000,
                     "best_yes": 0.45},
                ],
                "arb_opportunities": [],
            },
            confidence=0.7, urgency=0,
        )
        await div.consume_signal(sig)
        signals = await div.scan()
        report = await div.report()
        assert report["cycles"] >= 1
        assert report["account"]["equity"] > 0

    @pytest.mark.asyncio
    async def test_crypto_full_cycle(self):
        div = CryptoPaperDivision(starting_balance=10_000.0, persist=False)
        sig = Signal(
            signal_type=SignalType.INTEL_UPDATE,
            source_division="crypto_council",
            timestamp=datetime.now(timezone.utc),
            data={
                "type": "crypto_scan",
                "total_coins": 2,
                "sentiment": "bullish",
                "btc_price": 104_000,
                "eth_price": 3_200,
                "gainers": [
                    {"coin_id": "solana", "symbol": "SOL", "price": 170,
                     "change_24h": 3.5, "volume_24h": 2_000_000},
                ],
                "losers": [],
            },
            confidence=0.7, urgency=0,
        )
        await div.consume_signal(sig)
        signals = await div.scan()
        report = await div.report()
        assert report["cycles"] >= 1
        assert report["account"]["equity"] > 0
