from __future__ import annotations

"""Tests for paper trading risk management, regime detection, and metrics."""

from unittest.mock import MagicMock

import pytest

from strategies.paper_trading.engine import (
    OrderSide,
    PaperPosition,
    PaperTradingEngine,
)
from strategies.paper_trading.metrics import MetricsTracker, PerformanceSnapshot
from strategies.paper_trading.regime_detector import (
    MarketRegime,
    RegimeConfig,
    RegimeDetector,
)
from strategies.paper_trading.risk_manager import RiskConfig, RiskManager, RiskState


# =============================================================================
# RiskManager tests
# =============================================================================


class TestRiskManager:
    """Account-level risk manager tests."""

    def _make_positions(self, count: int = 0) -> dict[str, PaperPosition]:
        positions: dict[str, PaperPosition] = {}
        for i in range(count):
            sym = f"SYM{i}"
            positions[sym] = PaperPosition(
                symbol=sym,
                side=OrderSide.BUY,
                quantity=10.0,
                avg_entry_price=100.0,
                current_price=100.0,
                strategy="test_strat",
            )
        return positions

    def test_allows_normal_trade(self) -> None:
        rm = RiskManager(RiskConfig(max_open_positions=10))
        ok, reason = rm.pre_trade_check(
            account_equity=100_000,
            account_positions=self._make_positions(2),
            symbol="AAPL",
            side="buy",
            quantity=10,
            price=150.0,
            strategy="test",
        )
        assert ok
        assert reason == ""

    def test_blocks_max_open_positions(self) -> None:
        rm = RiskManager(RiskConfig(max_open_positions=5))
        ok, reason = rm.pre_trade_check(
            account_equity=100_000,
            account_positions=self._make_positions(5),
            symbol="NEW",
            side="buy",
            quantity=10,
            price=100.0,
            strategy="test",
        )
        assert not ok
        assert "max_open_positions" in reason

    def test_allows_sell_even_at_max_positions(self) -> None:
        rm = RiskManager(RiskConfig(max_open_positions=5))
        positions = self._make_positions(5)
        ok, reason = rm.pre_trade_check(
            account_equity=100_000,
            account_positions=positions,
            symbol="SYM0",
            side="sell",
            quantity=5,
            price=100.0,
            strategy="test",
        )
        assert ok

    def test_blocks_oversized_position(self) -> None:
        rm = RiskManager(RiskConfig(max_position_pct=0.10))
        ok, reason = rm.pre_trade_check(
            account_equity=100_000,
            account_positions={},
            symbol="TSLA",
            side="buy",
            quantity=100,
            price=200.0,  # 20k = 20% of 100k equity
            strategy="test",
        )
        assert not ok
        assert "position_size" in reason

    def test_blocks_strategy_allocation(self) -> None:
        rm = RiskManager(RiskConfig(max_strategy_allocation_pct=30.0))
        # Pre-seed strategy exposure
        rm.state.strategy_exposure["heavy_strat"] = 25_000
        ok, reason = rm.pre_trade_check(
            account_equity=100_000,
            account_positions={},
            symbol="X",
            side="buy",
            quantity=100,
            price=100.0,  # adds 10k → total 35k = 35% > 30%
            strategy="heavy_strat",
        )
        assert not ok
        assert "strategy" in reason

    def test_blocks_when_halted(self) -> None:
        rm = RiskManager()
        rm.halt("test_halt")
        ok, reason = rm.pre_trade_check(
            account_equity=100_000,
            account_positions={},
            symbol="X",
            side="buy",
            quantity=1,
            price=10.0,
            strategy="test",
        )
        assert not ok
        assert "halted" in reason

    def test_resume_after_halt(self) -> None:
        rm = RiskManager()
        rm.halt("test")
        assert rm.state.is_halted
        rm.resume()
        assert not rm.state.is_halted

    def test_daily_loss_limit_halts(self) -> None:
        rm = RiskManager(RiskConfig(daily_loss_limit_pct=5.0))
        # Simulate daily start at 100k, now at 94k → 6% loss
        # Use today's date so reset_daily doesn't overwrite
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rm.state.daily_start_equity = 100_000
        rm.state.daily_start_date = today
        ok, reason = rm.pre_trade_check(
            account_equity=94_000,
            account_positions={},
            symbol="X",
            side="buy",
            quantity=1,
            price=10.0,
            strategy="test",
        )
        assert not ok
        assert "daily_loss_limit" in reason
        assert rm.state.is_halted

    def test_post_update_circuit_breaker(self) -> None:
        rm = RiskManager(RiskConfig(max_drawdown_pct=15.0))
        alerts = rm.post_update_check(
            account_equity=80_000,
            peak_equity=100_000,  # 20% drawdown
            positions={},
        )
        assert any(a["type"] == "circuit_breaker" for a in alerts)
        assert rm.state.is_halted

    def test_trailing_stop_detection(self) -> None:
        rm = RiskManager(RiskConfig(trailing_stop_pct=0.10))
        pos = PaperPosition(
            symbol="BTC",
            side=OrderSide.BUY,
            quantity=1,
            avg_entry_price=50_000,
            current_price=60_000,
            strategy="test",
        )
        # First update: record high at 60k
        rm.check_trailing_stops({"BTC": pos})
        assert rm.state.trailing_highs["BTC"] == 60_000

        # Price drops 15% from peak → triggers trailing stop
        pos.current_price = 50_000  # 16.7% drop
        triggered = rm.check_trailing_stops({"BTC": pos})
        assert "BTC" in triggered

    def test_get_status(self) -> None:
        rm = RiskManager()
        status = rm.get_status()
        assert "config" in status
        assert "state" in status
        assert status["config"]["max_drawdown_pct"] == 20.0


class TestRiskState:
    def test_daily_reset(self) -> None:
        state = RiskState()
        state.reset_daily(50_000)
        assert state.daily_start_equity == 50_000
        assert state.daily_start_date != ""

    def test_to_dict(self) -> None:
        state = RiskState()
        d = state.to_dict()
        assert "is_halted" in d
        assert "breach_count" in d


# =============================================================================
# RegimeDetector tests
# =============================================================================


class TestRegimeDetector:
    def test_unknown_with_no_data(self) -> None:
        rd = RegimeDetector()
        assert rd.current_regime == MarketRegime.UNKNOWN

    def test_classifies_trending_up(self) -> None:
        rd = RegimeDetector(RegimeConfig(
            trend_threshold=0.03,
            lookback_periods=10,
            min_observations=5,
        ))
        # Simulate uptrend: 100 → 110 over 10 periods
        for i in range(10):
            rd.update_prices({"BTC": 100 + i})
        regime = rd.current_regime
        # With 10% rise over lookback this should be trending up
        assert regime in (MarketRegime.TRENDING_UP, MarketRegime.VOLATILE)

    def test_classifies_trending_down(self) -> None:
        rd = RegimeDetector(RegimeConfig(
            trend_threshold=0.03,
            lookback_periods=10,
            min_observations=5,
        ))
        for i in range(10):
            rd.update_prices({"BTC": 100 - i})
        regime = rd.current_regime
        assert regime in (MarketRegime.TRENDING_DOWN, MarketRegime.CRISIS, MarketRegime.VOLATILE)

    def test_classifies_ranging(self) -> None:
        rd = RegimeDetector(RegimeConfig(
            trend_threshold=0.03,
            volatility_threshold=0.10,
            lookback_periods=10,
            min_observations=5,
        ))
        # Small oscillation around 100
        prices = [100, 100.5, 99.5, 100, 100.5, 99.5, 100, 100.5, 99.5, 100]
        for p in prices:
            rd.update_prices({"STABLE": p})
        assert rd.current_regime == MarketRegime.RANGING

    def test_get_guidance(self) -> None:
        rd = RegimeDetector()
        guidance = rd.get_guidance()
        assert "regime" in guidance
        assert "stop_loss_mult" in guidance
        assert "position_size_mult" in guidance

    def test_multiple_symbols(self) -> None:
        rd = RegimeDetector(RegimeConfig(min_observations=3, lookback_periods=5))
        for i in range(5):
            rd.update_prices({
                "BTC": 100 + i * 2,
                "ETH": 50 + i,
            })
        status = rd.get_status()
        assert status["symbols_tracked"] == 2


# =============================================================================
# MetricsTracker tests
# =============================================================================


class TestMetricsTracker:
    def test_empty_metrics(self) -> None:
        mt = MetricsTracker()
        snap = mt.compute()
        assert snap.total_trades == 0
        assert snap.sharpe_ratio == 0.0

    def test_all_winning_trades(self) -> None:
        mt = MetricsTracker()
        for _ in range(10):
            mt.record_trade(100.0)
        snap = mt.compute()
        assert snap.total_trades == 10
        assert snap.win_rate == 100.0
        assert snap.avg_win == 100.0
        assert snap.max_consecutive_wins == 10
        assert snap.max_consecutive_losses == 0
        assert snap.profit_factor == float("inf")

    def test_mixed_trades(self) -> None:
        mt = MetricsTracker()
        pnls = [50, -20, 80, -30, 40, -10, 60, -25, 30, -15]
        mt.record_trades(pnls)
        mt.update_equity(100_000)
        snap = mt.compute()
        assert snap.total_trades == 10
        assert snap.win_rate == 50.0
        assert snap.avg_win == pytest.approx(52.0)
        assert snap.avg_loss == pytest.approx(-20.0)
        assert snap.best_trade == 80
        assert snap.worst_trade == -30
        assert snap.profit_factor > 1.0
        assert snap.expectancy > 0  # net positive
        assert snap.risk_reward_ratio > 0

    def test_sharpe_calculation(self) -> None:
        mt = MetricsTracker(risk_free_rate=0.0)
        # Consistent returns → high Sharpe
        for _ in range(50):
            mt.record_trade(10.0)
        snap = mt.compute()
        # All identical → std=0 → but we handle this
        # Actually all same = std 0, sharpe 0
        assert snap.sharpe_ratio == 0.0  # std is 0

        # Now with some variance
        mt2 = MetricsTracker(risk_free_rate=0.0)
        for i in range(50):
            mt2.record_trade(10.0 + (i % 3) - 1)
        snap2 = mt2.compute()
        assert snap2.sharpe_ratio > 0

    def test_max_drawdown_tracking(self) -> None:
        mt = MetricsTracker()
        mt.update_equity(100_000)
        mt.update_equity(105_000)
        mt.update_equity(95_000)  # 9.5% DD from 105k peak
        mt.update_equity(98_000)
        snap = mt.compute()
        # Recovery factor depends on trades, but DD should be tracked
        assert mt._max_drawdown > 0

    def test_consecutive_streaks(self) -> None:
        mt = MetricsTracker()
        pnls = [10, 20, 30, -5, -10, -15, -20, 5, 10]
        mt.record_trades(pnls)
        snap = mt.compute()
        assert snap.max_consecutive_wins == 3
        assert snap.max_consecutive_losses == 4

    def test_passes_minimum_bar(self) -> None:
        snap = PerformanceSnapshot(
            sharpe_ratio=1.5,
            profit_factor=1.8,
            total_trades=50,
        )
        passes, failures = snap.passes_minimum_bar(min_sharpe=1.0, min_profit_factor=1.2, min_trades=30)
        assert passes
        assert len(failures) == 0

    def test_fails_minimum_bar(self) -> None:
        snap = PerformanceSnapshot(
            sharpe_ratio=0.5,
            profit_factor=0.9,
            total_trades=10,
        )
        passes, failures = snap.passes_minimum_bar(min_sharpe=1.0, min_profit_factor=1.2, min_trades=30)
        assert not passes
        assert len(failures) == 3

    def test_to_dict(self) -> None:
        snap = PerformanceSnapshot(sharpe_ratio=1.234567)
        d = snap.to_dict()
        assert d["sharpe_ratio"] == 1.235  # rounded to 3


# =============================================================================
# Engine + RiskManager integration tests
# =============================================================================


class TestEngineRiskIntegration:
    """Test that RiskManager blocks orders through the engine."""

    def test_engine_blocks_when_halted(self) -> None:
        rm = RiskManager()
        rm.halt("test")
        engine = PaperTradingEngine(
            account_id="test_risk",
            starting_balance=100_000,
            risk_manager=rm,
        )
        result = engine.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            price=150.0,
            strategy="test",
        )
        assert result is None  # blocked

    def test_engine_allows_after_resume(self) -> None:
        rm = RiskManager()
        rm.halt("test")
        rm.resume()
        engine = PaperTradingEngine(
            account_id="test_risk2",
            starting_balance=100_000,
            risk_manager=rm,
        )
        result = engine.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            price=150.0,
            strategy="test",
        )
        assert result is not None
        assert result.status.value == "filled"

    def test_engine_triggers_circuit_breaker_on_update(self) -> None:
        rm = RiskManager(RiskConfig(max_drawdown_pct=10.0))
        engine = PaperTradingEngine(
            account_id="test_cb",
            starting_balance=100_000,
            risk_manager=rm,
        )
        # Buy something
        engine.place_order("BTC", OrderSide.BUY, 100, 100.0, strategy="test")
        # Price crashes — update triggers circuit breaker
        engine.update_prices({"BTC": 50.0})  # massive drop
        # Now orders should be blocked
        result = engine.place_order("ETH", OrderSide.BUY, 1, 10.0, strategy="test")
        assert result is None

    def test_engine_blocks_max_positions(self) -> None:
        rm = RiskManager(RiskConfig(max_open_positions=2))
        engine = PaperTradingEngine(
            account_id="test_maxpos",
            starting_balance=100_000,
            risk_manager=rm,
        )
        engine.place_order("A", OrderSide.BUY, 1, 10.0, strategy="test")
        engine.place_order("B", OrderSide.BUY, 1, 10.0, strategy="test")
        # Third position should be blocked
        result = engine.place_order("C", OrderSide.BUY, 1, 10.0, strategy="test")
        assert result is None

    def test_engine_works_without_risk_manager(self) -> None:
        """Backward compatibility — no risk_manager means no checks."""
        engine = PaperTradingEngine(
            account_id="test_no_rm",
            starting_balance=100_000,
        )
        result = engine.place_order("AAPL", OrderSide.BUY, 10, 150.0, strategy="test")
        assert result is not None
        assert result.status.value == "filled"
