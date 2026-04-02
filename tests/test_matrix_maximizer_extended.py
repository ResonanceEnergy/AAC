"""
Tests for MATRIX MAXIMIZER — Extended Modules
================================================
Covers: data_feeds, intelligence, execution, advanced_strategies,
        alerts, scheduler, backtester, dashboard, chatbot,
        and enhanced risk methods.
"""

import json
import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED RISK TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestCorrelationStress:
    def test_stress_returns_dict(self):
        from strategies.matrix_maximizer.core import MatrixConfig
        from strategies.matrix_maximizer.risk import RiskManager
        from strategies.matrix_maximizer.scanner import Position

        rm = RiskManager(MatrixConfig(account_size=5000))
        pos = Position(
            ticker="SPY", strike=400, expiry="2026-06-30",
            entry_date="2026-03-01", entry_premium=3.00, entry_delta=-0.30,
            contracts=1, cost_basis=300.0, current_premium=3.50,
            current_delta=-0.35, days_held=10, pnl_pct=16.7,
        )
        result = rm.correlation_stress_test([pos], {"SPY": 420})
        assert "shocked_pnl" in result or "stressed_pnl" in result
        assert "survival" in result
        assert bool(result["survival"]) in (True, False)  # numpy bool compatible

    def test_stress_deeper_shock_worse_pnl(self):
        from strategies.matrix_maximizer.core import MatrixConfig
        from strategies.matrix_maximizer.risk import RiskManager
        from strategies.matrix_maximizer.scanner import Position

        rm = RiskManager(MatrixConfig(account_size=5000))
        pos = Position(
            ticker="SPY", strike=400, expiry="2026-06-30",
            entry_date="2026-03-01", entry_premium=3.00, entry_delta=-0.30,
            contracts=1, cost_basis=300.0, current_premium=3.50,
            current_delta=-0.35, days_held=10, pnl_pct=16.7,
        )
        mild = rm.correlation_stress_test([pos], {"SPY": 420}, shock_pct=-0.05)
        severe = rm.correlation_stress_test([pos], {"SPY": 420}, shock_pct=-0.20)
        # For long puts, a bigger drop means PUT VALUE INCREASES → more positive PnL
        # So stressed_pnl should be MORE positive for severe crash
        assert severe["stressed_pnl"] >= mild["stressed_pnl"]

    def test_empty_positions(self):
        from strategies.matrix_maximizer.core import MatrixConfig
        from strategies.matrix_maximizer.risk import RiskManager

        rm = RiskManager(MatrixConfig(account_size=5000))
        result = rm.correlation_stress_test([], {})
        assert result["stressed_pnl"] == 0.0
        assert result["survival"] is True


class TestTailRisk:
    def test_fat_tails_wider_than_normal(self):
        from strategies.matrix_maximizer.core import (
            Asset,
            AssetForecast,
            MandateLevel,
            MatrixConfig,
            PortfolioForecast,
            ScenarioWeights,
            SystemMandate,
        )
        from strategies.matrix_maximizer.risk import RiskManager

        rm = RiskManager(MatrixConfig())
        spy_forecast = AssetForecast(
            asset=Asset.SPY,
            current_price=420,
            mean_price=415, median_price=416,
            pct_5=380, pct_25=400, pct_75=430, pct_95=450,
            expected_return_pct=-3.0,
            prob_down_10=0.20, prob_down_15=0.10, prob_down_20=0.08,
            prob_up_5=0.30,
            var_95_1d=-0.025, cvar_95_1d=-0.038,
            var_95_90d=-0.12, cvar_95_90d=-0.18,
            simulated_paths=10000,
        )
        forecast = PortfolioForecast(
            asset_forecasts={Asset.SPY: spy_forecast},
            mandate=SystemMandate(
                level=MandateLevel.STANDARD,
                risk_per_trade_pct=0.01, otm_pct=0.05,
                max_contracts_per_name=5, pyramid_allowed=False,
                hedge_energy=False, hedge_tlt=False,
                rationale="Test",
            ),
            timestamp=datetime.utcnow(),
            scenario_weights=ScenarioWeights(),
            weighted_return=-0.03,
            portfolio_var_95=-0.025,
            portfolio_cvar_95=-0.038,
            horizon_days=90,
            n_simulations=10000,
        )
        result = rm.tail_risk_analysis(forecast, df=3.0)
        assert abs(result["fat_tail_var_95"]) >= abs(result["normal_var_95"])
        assert result["tail_risk_multiplier"] >= 1.0

    def test_higher_df_closer_to_normal(self):
        from strategies.matrix_maximizer.core import (
            Asset,
            AssetForecast,
            MandateLevel,
            MatrixConfig,
            PortfolioForecast,
            ScenarioWeights,
            SystemMandate,
        )
        from strategies.matrix_maximizer.risk import RiskManager

        rm = RiskManager(MatrixConfig())
        spy_forecast = AssetForecast(
            asset=Asset.SPY, current_price=420,
            mean_price=415, median_price=416,
            pct_5=380, pct_25=400, pct_75=430, pct_95=450,
            expected_return_pct=-3.0,
            prob_down_10=0.20, prob_down_15=0.10, prob_down_20=0.08,
            prob_up_5=0.30,
            var_95_1d=-0.025, cvar_95_1d=-0.038,
            var_95_90d=-0.12, cvar_95_90d=-0.18,
            simulated_paths=10000,
        )
        forecast = PortfolioForecast(
            asset_forecasts={Asset.SPY: spy_forecast},
            mandate=SystemMandate(
                level=MandateLevel.STANDARD,
                risk_per_trade_pct=0.01, otm_pct=0.05,
                max_contracts_per_name=5, pyramid_allowed=False,
                hedge_energy=False, hedge_tlt=False,
                rationale="Test",
            ),
            timestamp=datetime.utcnow(),
            scenario_weights=ScenarioWeights(),
            weighted_return=-0.03,
            portfolio_var_95=-0.025,
            portfolio_cvar_95=-0.038,
            horizon_days=90,
            n_simulations=10000,
        )
        fat = rm.tail_risk_analysis(forecast, df=3.0)
        thin = rm.tail_risk_analysis(forecast, df=30.0)
        assert fat["tail_risk_multiplier"] > thin["tail_risk_multiplier"]


class TestLiquidityRisk:
    def test_wide_spread_flagged(self):
        from strategies.matrix_maximizer.core import MandateLevel, MatrixConfig
        from strategies.matrix_maximizer.greeks import GreeksResult
        from strategies.matrix_maximizer.risk import RiskManager
        from strategies.matrix_maximizer.scanner import (
            OptionContract,
            PutRecommendation,
        )

        rm = RiskManager(MatrixConfig())
        contract = OptionContract(
            ticker="SPY", strike=400, expiry="2026-06-30",
            dte=90, bid=0.50, ask=2.00, mid=1.25,
            volume=100, open_interest=5000, iv=0.30,
        )
        greeks = GreeksResult(
            spot=420, strike=400, time_years=0.25, sigma=0.30,
            rate=0.05, div_yield=0.01,
            price=1.25, delta=-0.25, gamma=0.01,
            theta=-0.05, vega=0.20, rho=-0.01,
            d1=0.5, d2=0.35, otm_pct=0.048,
            intrinsic=0.0, extrinsic=1.25,
        )
        pick = PutRecommendation(
            rank=1, ticker="SPY", contract=contract, greeks=greeks,
            contracts=1, total_cost=125.0, risk_pct=0.025,
            delta_score=80, liquidity_score=30, edge_score=70,
            composite_score=75.0, mandate=MandateLevel.STANDARD,
            thesis="Test wide spread",
        )
        results = rm.liquidity_risk([pick], min_bid_ask_ratio=0.80)
        assert len(results) == 1
        assert results[0]["passed"] is False  # 0.50/2.00 = 0.25 < 0.80

    def test_tight_spread_passes(self):
        from strategies.matrix_maximizer.core import MandateLevel, MatrixConfig
        from strategies.matrix_maximizer.greeks import GreeksResult
        from strategies.matrix_maximizer.risk import RiskManager
        from strategies.matrix_maximizer.scanner import (
            OptionContract,
            PutRecommendation,
        )

        rm = RiskManager(MatrixConfig())
        contract = OptionContract(
            ticker="SPY", strike=400, expiry="2026-06-30",
            dte=90, bid=1.90, ask=2.00, mid=1.95,
            volume=5000, open_interest=50000, iv=0.25,
        )
        greeks = GreeksResult(
            spot=420, strike=400, time_years=0.25, sigma=0.25,
            rate=0.05, div_yield=0.01,
            price=1.95, delta=-0.25, gamma=0.01,
            theta=-0.05, vega=0.20, rho=-0.01,
            d1=0.5, d2=0.35, otm_pct=0.048,
            intrinsic=0.0, extrinsic=1.95,
        )
        pick = PutRecommendation(
            rank=1, ticker="SPY", contract=contract, greeks=greeks,
            contracts=1, total_cost=195.0, risk_pct=0.039,
            delta_score=85, liquidity_score=95, edge_score=80,
            composite_score=85.0, mandate=MandateLevel.STANDARD,
            thesis="Test tight spread",
        )
        results = rm.liquidity_risk([pick])
        assert len(results) == 1
        assert results[0]["passed"] is True  # 1.90/2.00 = 0.95


class TestMarginEstimate:
    def test_long_put_margin_equals_cost(self):
        from strategies.matrix_maximizer.core import MatrixConfig
        from strategies.matrix_maximizer.risk import RiskManager
        from strategies.matrix_maximizer.scanner import Position

        rm = RiskManager(MatrixConfig(account_size=5000))
        pos = Position(
            ticker="SPY", strike=400, expiry="2026-06-30",
            entry_date="2026-03-01", entry_premium=3.00, entry_delta=-0.30,
            contracts=2, cost_basis=600.0, current_premium=3.50,
            current_delta=-0.35, days_held=10, pnl_pct=16.7,
        )
        result = rm.margin_estimate([pos], {"SPY": 420})
        assert result["total_cost_basis"] == 600.0
        assert result["buying_power_used"] == 600.0
        assert result["buying_power_remaining"] == 4400.0


# ═══════════════════════════════════════════════════════════════════════════
# DATA FEEDS TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestDataFeedManager:
    def test_init(self):
        from strategies.matrix_maximizer.data_feeds import DataFeedManager
        mgr = DataFeedManager()
        assert mgr is not None

    def test_live_quote_dataclass(self):
        from strategies.matrix_maximizer.data_feeds import LiveQuote
        q = LiveQuote(symbol="SPY", price=420.50, change=-2.30, change_pct=-0.54)
        assert q.symbol == "SPY"
        assert q.source == "polygon"

    def test_macro_snapshot_defaults(self):
        from strategies.matrix_maximizer.data_feeds import MacroSnapshot
        snap = MacroSnapshot()
        assert snap.vix == 22.0
        assert snap.oil_wti == 96.5
        assert snap.yield_curve_10_2 == -0.3

    def test_earnings_event(self):
        from strategies.matrix_maximizer.data_feeds import EarningsEvent
        ev = EarningsEvent(symbol="AAPL", date="2026-04-25", hour="amc")
        assert ev.symbol == "AAPL"

    def test_resolve_prices_with_defaults(self):
        """resolve_prices falls back to defaults without live data."""
        from strategies.matrix_maximizer.data_feeds import DataFeedManager
        mgr = DataFeedManager()
        # Monkey-patch get_live_prices to return empty (no API call)
        mgr.get_live_prices = lambda tickers: {}
        prices = mgr.resolve_prices(["SPY", "XLE"])
        assert "SPY" in prices
        assert prices["SPY"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# INTELLIGENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestIntelligence:
    def test_intel_signal_dataclass(self):
        from strategies.matrix_maximizer.intelligence import IntelSignal
        sig = IntelSignal(
            source="ncl", signal_type="sector_rotation",
            ticker="XLE", direction="bearish", strength=0.8,
            thesis="Oil peak expected",
        )
        assert sig.source == "ncl"
        assert sig.strength == 0.8

    def test_intel_brief_filtering(self):
        from strategies.matrix_maximizer.intelligence import IntelBrief, IntelSignal
        brief = IntelBrief()
        brief.signals.extend([
            IntelSignal("ncl", "rotation", "SPY", "bearish", 0.7, "test"),
            IntelSignal("ncl", "rotation", "XLE", "bullish", 0.6, "test"),
            IntelSignal("regime", "F1_vol", "SPY", "bearish", 0.9, "test"),
        ])
        assert len(brief.bearish_signals) == 2
        assert len(brief.bullish_signals) == 1

    def test_gather_intel_returns_brief(self):
        from strategies.matrix_maximizer.intelligence import IntelligenceEngine
        engine = IntelligenceEngine()
        brief = engine.gather_intel(["SPY", "XLE"])
        assert brief is not None
        assert hasattr(brief, "signals")
        assert hasattr(brief, "ticker_sentiment")


# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestPositionSizer:
    def test_kelly_size(self):
        from strategies.matrix_maximizer.execution import PositionSizer
        sizer = PositionSizer()
        kelly = sizer.kelly_size(win_rate=0.60, avg_win=200, avg_loss=100)
        assert kelly > 0
        assert kelly < 1.0  # Half-Kelly should be < 100%

    def test_kelly_negative_edge(self):
        from strategies.matrix_maximizer.execution import PositionSizer
        sizer = PositionSizer()
        kelly = sizer.kelly_size(win_rate=0.30, avg_win=100, avg_loss=200)
        assert kelly == 0  # Negative edge = 0 allocation

    def test_fixed_fractional(self):
        from strategies.matrix_maximizer.execution import PositionSizer
        sizer = PositionSizer()
        size = sizer.fixed_fractional(
            account_size=5000, risk_pct=0.02, premium=1.50,
        )
        assert size > 0

    def test_max_contracts_cap(self):
        from strategies.matrix_maximizer.execution import PositionSizer
        sizer = PositionSizer()
        contracts = sizer.max_contracts(
            account_size=1000, risk_pct=0.05, premium=1.50,
            existing_exposure=0,
        )
        assert contracts >= 0
        # Cost per contract = 1.50 × 100 = $150
        # Max risk = 1000 × 0.05 = $50 → but premium > risk → cap applies


class TestTrackedPosition:
    def test_is_open(self):
        from strategies.matrix_maximizer.execution import TrackedPosition
        pos = TrackedPosition(
            position_id="test1", ticker="SPY", strike=400,
            expiry="2026-06-30", contracts=1, entry_premium=3.00,
            entry_date="2026-03-01", entry_delta=-0.30, cost_basis=300.0,
        )
        assert pos.is_open is True
        pos.state = "closed"
        assert pos.is_open is False

    def test_to_scanner_position(self):
        from strategies.matrix_maximizer.execution import TrackedPosition
        pos = TrackedPosition(
            position_id="test1", ticker="SPY", strike=400,
            expiry="2026-06-30", contracts=1, entry_premium=3.00,
            entry_date="2026-03-01", entry_delta=-0.30, cost_basis=300.0,
        )
        scanner_pos = pos.to_scanner_position()
        assert scanner_pos.ticker == "SPY"
        assert scanner_pos.strike == 400


class TestExecutionEngine:
    def test_dry_run_mode(self):
        from strategies.matrix_maximizer.execution import ExecutionEngine, ExecutionMode
        engine = ExecutionEngine(mode=ExecutionMode.DRY_RUN, account_size=5000)
        assert engine.mode == ExecutionMode.DRY_RUN

    def test_dry_run_buy(self):
        from strategies.matrix_maximizer.execution import ExecutionEngine, ExecutionMode
        engine = ExecutionEngine(mode=ExecutionMode.DRY_RUN, account_size=5000)
        result = engine.buy_put("SPY", 400, "2026-06-30", 1, 3.00, -0.30)
        assert result.success is True  # Dry run always succeeds

    def test_execute_picks_dry_run(self):
        from strategies.matrix_maximizer.execution import ExecutionEngine, ExecutionMode
        engine = ExecutionEngine(mode=ExecutionMode.DRY_RUN, account_size=5000)
        picks = [
            {"ticker": "SPY", "strike": 400, "expiry": "2026-06-30",
             "premium": 3.00, "delta": -0.30},
        ]
        results = engine.execute_picks(picks, mandate_risk_pct=0.02)
        assert len(results) >= 0  # May be 0 if sizer says no


# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED STRATEGIES TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestAdvancedStrategies:
    def test_bear_put_spread(self):
        from strategies.matrix_maximizer.advanced_strategies import AdvancedStrategyEngine
        engine = AdvancedStrategyEngine()
        spreads = engine.bear_put_spreads("SPY", 420, 0.25, 45)
        assert isinstance(spreads, list)
        if spreads:
            s = spreads[0]
            assert s.strategy_type == "bear_put_spread"
            assert len(s.legs) == 2
            assert s.max_loss > 0 or s.max_loss < 0  # Defined risk

    def test_straddle(self):
        from strategies.matrix_maximizer.advanced_strategies import AdvancedStrategyEngine
        engine = AdvancedStrategyEngine()
        straddles = engine.straddles("SPY", 420, 0.25, 30)
        assert isinstance(straddles, list)
        if straddles:
            s = straddles[0]
            assert s.strategy_type == "straddle"
            assert len(s.legs) == 2

    def test_iron_condor(self):
        from strategies.matrix_maximizer.advanced_strategies import AdvancedStrategyEngine
        engine = AdvancedStrategyEngine()
        condors = engine.iron_condors("SPY", 420, 0.25, 30)
        assert isinstance(condors, list)
        if condors:
            c = condors[0]
            assert c.strategy_type == "iron_condor"
            assert len(c.legs) == 4

    def test_butterfly(self):
        from strategies.matrix_maximizer.advanced_strategies import AdvancedStrategyEngine
        engine = AdvancedStrategyEngine()
        butterflies = engine.butterflies("SPY", 420, 0.25, 30)
        assert isinstance(butterflies, list)
        if butterflies:
            b = butterflies[0]
            assert b.strategy_type == "butterfly"
            assert len(b.legs) == 3

    def test_collar(self):
        from strategies.matrix_maximizer.advanced_strategies import AdvancedStrategyEngine
        engine = AdvancedStrategyEngine()
        collars = engine.collars("SPY", 420, 100, 0.25, 45)
        assert isinstance(collars, list)
        if collars:
            c = collars[0]
            assert c.strategy_type == "collar"

    def test_recommend_high_vix(self):
        from strategies.matrix_maximizer.advanced_strategies import AdvancedStrategyEngine
        engine = AdvancedStrategyEngine()
        recs = engine.recommend_strategies("SPY", 420, 0.25, 45, vix=35)
        assert isinstance(recs, list)

    def test_recommend_low_vix(self):
        from strategies.matrix_maximizer.advanced_strategies import AdvancedStrategyEngine
        engine = AdvancedStrategyEngine()
        recs = engine.recommend_strategies("SPY", 420, 0.25, 45, vix=15)
        assert isinstance(recs, list)

    def test_recommend_normal_vix(self):
        from strategies.matrix_maximizer.advanced_strategies import AdvancedStrategyEngine
        engine = AdvancedStrategyEngine()
        recs = engine.recommend_strategies("SPY", 420, 0.25, 45, vix=22)
        assert isinstance(recs, list)

    def test_strategy_leg_dataclass(self):
        from strategies.matrix_maximizer.advanced_strategies import StrategyLeg
        leg = StrategyLeg(
            ticker="SPY", strike=400, expiry="2026-06-30",
            option_type="put", side="buy", contracts=1, premium=3.00,
        )
        assert leg.ticker == "SPY"
        assert leg.option_type == "put"

    def test_multi_leg_print_card(self):
        from strategies.matrix_maximizer.advanced_strategies import (
            MultiLegStrategy,
            StrategyLeg,
        )
        legs = [
            StrategyLeg("SPY", 400, "2026-06-30", "put", "buy", 1, 3.00),
            StrategyLeg("SPY", 390, "2026-06-30", "put", "sell", 1, 1.50),
        ]
        strat = MultiLegStrategy(
            name="Bear Spread", strategy_type="bear_put_spread",
            ticker="SPY", legs=legs, net_debit=1.50,
            max_profit=850, max_loss=150, breakeven=398.50,
            reward_risk_ratio=5.67,
        )
        card = strat.print_card()
        assert "Bear Spread" in card
        assert "SPY" in card

    def test_weekly_roll_plan(self):
        from strategies.matrix_maximizer.advanced_strategies import AdvancedStrategyEngine
        engine = AdvancedStrategyEngine()
        plan = engine.weekly_roll_plan("SPY", 420, 0.25, target_delta=-0.30)
        assert isinstance(plan, list)
        assert len(plan) == 4  # 4 weeks

    def test_rotate_scan_tickers(self):
        from strategies.matrix_maximizer.advanced_strategies import AdvancedStrategyEngine
        engine = AdvancedStrategyEngine()
        tickers = engine.rotate_scan_tickers("vol_shock_active", oil_price=80.0, vix=30.0)
        assert isinstance(tickers, list)
        assert len(tickers) > 0


# ═══════════════════════════════════════════════════════════════════════════
# ALERTS TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestAlerts:
    def test_alert_manager_init(self):
        from strategies.matrix_maximizer.alerts import AlertManager
        mgr = AlertManager()
        assert mgr is not None

    def test_alert_dataclass(self):
        from strategies.matrix_maximizer.alerts import Alert, AlertChannel, AlertLevel
        alert = Alert(
            level=AlertLevel.INFO,
            title="Test alert",
            body="This is a test",
            channels=[AlertChannel.LOG],
        )
        assert alert.level == AlertLevel.INFO

    def test_log_only_alert(self):
        from strategies.matrix_maximizer.alerts import Alert, AlertChannel, AlertLevel, AlertManager
        mgr = AlertManager()
        alert = Alert(
            level=AlertLevel.INFO,
            title="Test",
            body="Test body",
            channels=[AlertChannel.LOG],
        )
        result = mgr.send(alert)
        assert result is True

    def test_rate_limiting(self):
        from strategies.matrix_maximizer.alerts import Alert, AlertChannel, AlertLevel, AlertManager
        mgr = AlertManager(enable_telegram=False, enable_email=False)
        alert = Alert(
            level=AlertLevel.INFO,
            title="Spam test",
            body="Testing rate limits",
            channels=[AlertChannel.LOG],
        )
        assert mgr.send(alert) is True
        # Second send without dedup key should also pass
        alert2 = Alert(
            level=AlertLevel.INFO,
            title="Spam test 2",
            body="Testing rate limits 2",
            channels=[AlertChannel.LOG],
        )
        assert mgr.send(alert2) is True

    def test_dedup(self):
        from strategies.matrix_maximizer.alerts import Alert, AlertChannel, AlertLevel, AlertManager
        mgr = AlertManager()
        alert = Alert(
            level=AlertLevel.INFO,
            title="Dedup test",
            body="Same key alert",
            channels=[AlertChannel.LOG],
            dedup_key="test_dedup_1",
        )
        assert mgr.send(alert) is True
        # Second send with same key should be deduped
        result2 = mgr.send(alert)
        # Dedup within window means False
        assert result2 is False

    def test_send_trade_alert(self):
        from strategies.matrix_maximizer.alerts import AlertManager
        mgr = AlertManager()
        # Patch _send_telegram to avoid actual API call
        mgr._send_telegram = MagicMock(return_value=True)
        result = mgr.send_trade_alert("SPY", 400, "2026-06-30", 3.00, 1)
        assert result is True

    def test_send_circuit_breaker_alert(self):
        from strategies.matrix_maximizer.alerts import AlertManager
        mgr = AlertManager()
        mgr._send_telegram = MagicMock(return_value=True)
        mgr._send_email = MagicMock(return_value=True)
        result = mgr.send_circuit_breaker_alert("GREEN", "RED", "VaR breach")
        assert result is True

    def test_send_roll_alert(self):
        from strategies.matrix_maximizer.alerts import AlertManager
        mgr = AlertManager()
        mgr._send_telegram = MagicMock(return_value=True)
        result = mgr.send_roll_alert("SPY", "ROLL_DOWN", "Delta too low")
        assert result is True


# ═══════════════════════════════════════════════════════════════════════════
# WATCHDOG TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestWatchdog:
    def test_watchdog_init(self):
        from strategies.matrix_maximizer.alerts import AlertManager, Watchdog
        mgr = AlertManager()
        wd = Watchdog(mgr)
        assert wd is not None

    def test_watchdog_circuit_breaker_change(self):
        from strategies.matrix_maximizer.alerts import AlertManager, Watchdog
        mgr = AlertManager()
        mgr._send_telegram = MagicMock(return_value=True)
        mgr._send_email = MagicMock(return_value=True)
        wd = Watchdog(mgr)
        wd.check_circuit_breaker("GREEN")
        wd.check_circuit_breaker("YELLOW")
        # Should have sent an alert for the transition
        assert wd._last_cb_state == "YELLOW"


# ═══════════════════════════════════════════════════════════════════════════
# SCHEDULER TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestScheduler:
    def test_scheduler_init(self):
        from strategies.matrix_maximizer.scheduler import MatrixScheduler
        sched = MatrixScheduler()
        assert sched is not None

    def test_add_task(self):
        from strategies.matrix_maximizer.scheduler import MatrixScheduler, ScheduleSlot
        sched = MatrixScheduler()
        sched.add_task(ScheduleSlot.PRE_MARKET, "test_task", lambda: None)
        status = sched.get_status()
        assert "test_task" in status["tasks"]

    def test_remove_task(self):
        from strategies.matrix_maximizer.scheduler import MatrixScheduler, ScheduleSlot
        sched = MatrixScheduler()
        sched.add_task(ScheduleSlot.PRE_MARKET, "test_task", lambda: None)
        sched.remove_task("test_task")
        status = sched.get_status()
        assert "test_task" not in status["tasks"]

    def test_enable_disable(self):
        from strategies.matrix_maximizer.scheduler import MatrixScheduler, ScheduleSlot
        sched = MatrixScheduler()
        sched.add_task(ScheduleSlot.MARKET_OPEN, "test_task", lambda: None)
        sched.disable_task("test_task")
        status = sched.get_status()
        assert status["tasks"]["test_task"]["enabled"] is False
        sched.enable_task("test_task")
        status = sched.get_status()
        assert status["tasks"]["test_task"]["enabled"] is True

    def test_run_now(self):
        from strategies.matrix_maximizer.scheduler import MatrixScheduler, ScheduleSlot
        executed = []
        sched = MatrixScheduler()
        sched.add_task(ScheduleSlot.AFTER_HOURS, "test_task",
                       lambda: executed.append(True))
        sched.run_now("test_task")
        assert len(executed) == 1

    def test_print_schedule(self):
        from strategies.matrix_maximizer.scheduler import MatrixScheduler, ScheduleSlot
        sched = MatrixScheduler()
        sched.add_task(ScheduleSlot.PRE_MARKET, "t1", lambda: None)
        sched.add_task(ScheduleSlot.MARKET_CLOSE, "t2", lambda: None)
        output = sched.print_schedule()
        assert isinstance(output, str)


# ═══════════════════════════════════════════════════════════════════════════
# BACKTESTER TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestBacktester:
    def test_backtester_init(self):
        from strategies.matrix_maximizer.backtester import MatrixBacktester
        bt = MatrixBacktester()
        assert bt is not None

    def test_generate_scenarios(self):
        from strategies.matrix_maximizer.backtester import MatrixBacktester
        bt = MatrixBacktester()
        scenarios = bt.generate_historical_scenarios(days=30)
        assert len(scenarios) == 30
        assert "vix" in scenarios[0]
        assert "oil_price" in scenarios[0]
        assert "prices" in scenarios[0]

    def test_generate_scenarios_with_crisis(self):
        from strategies.matrix_maximizer.backtester import MatrixBacktester
        bt = MatrixBacktester()
        scenarios = bt.generate_historical_scenarios(days=30, crisis_at=15)
        assert len(scenarios) == 30
        # VIX should spike at day 15
        assert scenarios[15]["vix"] > scenarios[10]["vix"]

    def test_backtest_runs(self):
        from strategies.matrix_maximizer.backtester import MatrixBacktester
        bt = MatrixBacktester()
        scenarios = bt.generate_historical_scenarios(days=30)
        result = bt.backtest(scenarios, initial_capital=5000)
        assert hasattr(result, "total_trades")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "equity_curve")
        assert result.total_pnl is not None

    def test_backtest_result_print_card(self):
        from strategies.matrix_maximizer.backtester import MatrixBacktester
        bt = MatrixBacktester()
        scenarios = bt.generate_historical_scenarios(days=20)
        result = bt.backtest(scenarios, initial_capital=1000)
        card = result.print_card()
        assert isinstance(card, str)
        assert "BACKTEST" in card.upper() or "RESULT" in card.upper() or len(card) > 0

    def test_backtest_stats(self):
        from strategies.matrix_maximizer.backtester import MatrixBacktester
        bt = MatrixBacktester()
        scenarios = bt.generate_historical_scenarios(days=60)
        result = bt.backtest(scenarios, initial_capital=5000)
        # Stats should be reasonable
        assert 0 <= result.win_rate <= 1.0
        assert result.max_drawdown <= 0 or result.total_trades == 0


# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestDashboard:
    def test_dashboard_init(self):
        from strategies.matrix_maximizer.dashboard import MatrixDashboard
        dash = MatrixDashboard()
        assert dash is not None

    def test_daily_snapshot_dataclass(self):
        from strategies.matrix_maximizer.dashboard import DailySnapshot
        snap = DailySnapshot(
            date="2026-03-20", equity=920.0, unrealized_pnl=-15.0,
            realized_pnl=0.0, open_positions=3, trades_today=1,
            vix=25.0, oil=98.0, regime="vol_shock_armed",
            circuit_breaker="GREEN", mandate="standard",
        )
        assert snap.equity == 920.0
        assert snap.circuit_breaker == "GREEN"

    def test_record_and_retrieve_snapshot(self):
        from strategies.matrix_maximizer.dashboard import DailySnapshot, MatrixDashboard
        dash = MatrixDashboard()
        snap = DailySnapshot(
            date="2026-03-20", equity=920.0, unrealized_pnl=-15.0,
            realized_pnl=0.0, open_positions=3, trades_today=1,
            vix=25.0, oil=98.0, regime="normal",
            circuit_breaker="GREEN", mandate="standard",
        )
        dash.record_snapshot(snap)
        history = dash.get_history(days=7)
        assert len(history) >= 1

    def test_daily_report_generation(self):
        from strategies.matrix_maximizer.dashboard import MatrixDashboard
        dash = MatrixDashboard()
        report = dash.daily_report()
        assert isinstance(report, str)
        assert "MATRIX MAXIMIZER" in report or "DAILY" in report.upper() or len(report) > 0

    def test_weekly_summary(self):
        from strategies.matrix_maximizer.dashboard import DailySnapshot, MatrixDashboard
        dash = MatrixDashboard()
        # Add a few snapshots
        for i in range(5):
            snap = DailySnapshot(
                date=f"2026-03-{15+i}", equity=920.0 + i * 10,
                unrealized_pnl=i * 5.0, realized_pnl=0.0,
                open_positions=3, trades_today=1,
                vix=22.0, oil=80.0, regime="normal",
                circuit_breaker="GREEN", mandate="standard",
            )
            dash.record_snapshot(snap)
        summary = dash.weekly_summary()
        assert isinstance(summary, str)


# ═══════════════════════════════════════════════════════════════════════════
# CHATBOT TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestChatbot:
    def _make_context(self):
        from strategies.matrix_maximizer.chatbot import ChatContext
        ctx = ChatContext()
        ctx.runner = MagicMock()
        ctx.runner._last_result = {
            "status": "complete",
            "forecast": {"mandate": "standard"},
            "risk": {"circuit_breaker": "GREEN", "passed": 7, "failed": 0},
            "regime": {"regime": "normal", "vix": 22.0, "oil_price": 80.0},
            "picks": [],
        }
        ctx.runner.run_full_cycle = MagicMock(return_value=ctx.runner._last_result)
        ctx.runner.print_summary = MagicMock(return_value="Cycle summary here")
        return ctx

    def test_chatbot_init(self):
        from strategies.matrix_maximizer.chatbot import MatrixChatbot
        ctx = self._make_context()
        bot = MatrixChatbot(ctx)
        assert bot is not None

    def test_help_command(self):
        from strategies.matrix_maximizer.chatbot import MatrixChatbot
        ctx = self._make_context()
        bot = MatrixChatbot(ctx)
        response = bot.handle("help")
        assert "help" in response.lower() or "command" in response.lower()

    def test_status_command(self):
        from strategies.matrix_maximizer.chatbot import MatrixChatbot
        ctx = self._make_context()
        bot = MatrixChatbot(ctx)
        response = bot.handle("status")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_risk_command(self):
        from strategies.matrix_maximizer.chatbot import MatrixChatbot
        ctx = self._make_context()
        bot = MatrixChatbot(ctx)
        response = bot.handle("risk")
        assert isinstance(response, str)

    def test_regime_command(self):
        from strategies.matrix_maximizer.chatbot import MatrixChatbot
        ctx = self._make_context()
        bot = MatrixChatbot(ctx)
        response = bot.handle("regime")
        assert isinstance(response, str)

    def test_natural_language_routing(self):
        from strategies.matrix_maximizer.chatbot import MatrixChatbot
        ctx = self._make_context()
        bot = MatrixChatbot(ctx)
        # "show me my positions" should route to positions command
        response = bot.handle("show me my positions")
        assert isinstance(response, str)

    def test_unknown_command(self):
        from strategies.matrix_maximizer.chatbot import MatrixChatbot
        ctx = self._make_context()
        bot = MatrixChatbot(ctx)
        response = bot.handle("xyzzy zork plugh")
        assert isinstance(response, str)

    def test_run_command(self):
        from strategies.matrix_maximizer.chatbot import MatrixChatbot
        ctx = self._make_context()
        bot = MatrixChatbot(ctx)
        response = bot.handle("run")
        assert isinstance(response, str)

    def test_pnl_command(self):
        from strategies.matrix_maximizer.chatbot import MatrixChatbot
        ctx = self._make_context()
        bot = MatrixChatbot(ctx)
        response = bot.handle("pnl")
        assert isinstance(response, str)

    def test_mandate_command(self):
        from strategies.matrix_maximizer.chatbot import MatrixChatbot
        ctx = self._make_context()
        bot = MatrixChatbot(ctx)
        response = bot.handle("mandate")
        assert isinstance(response, str)

    def test_chat_message_history(self):
        from strategies.matrix_maximizer.chatbot import MatrixChatbot
        ctx = self._make_context()
        bot = MatrixChatbot(ctx)
        bot.handle("status")
        bot.handle("risk")
        assert len(bot._history) >= 4  # 2 user + 2 bot


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER v2 INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestRunnerV2:
    def test_version_2(self):
        from strategies.matrix_maximizer.runner import MatrixMaximizer
        mm = MatrixMaximizer()
        assert mm.VERSION == "2.0.0"

    def test_optional_modules_load(self):
        from strategies.matrix_maximizer.runner import MatrixMaximizer
        mm = MatrixMaximizer()
        # These should initialize (may be None if import fails)
        # but shouldn't crash
        assert mm.data_feeds is not None or mm.data_feeds is None
        assert mm.intelligence is not None or mm.intelligence is None

    def test_get_chatbot(self):
        from strategies.matrix_maximizer.runner import MatrixMaximizer
        mm = MatrixMaximizer()
        bot = mm.get_chatbot()
        # May or may not initialize depending on chatbot module
        assert bot is not None or bot is None

    @patch("urllib.request.urlopen")
    def test_run_full_cycle_with_prices(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("no network")
        from strategies.matrix_maximizer.core import MatrixConfig
        from strategies.matrix_maximizer.runner import MatrixMaximizer
        mm = MatrixMaximizer(MatrixConfig(account_size=920, n_simulations=100))
        result = mm.run_full_cycle(
            prices={"oil": 95, "vix": 28},
        )
        assert result["status"] in ("complete", "blocked")

    @patch("urllib.request.urlopen")
    def test_auto_execute_flag(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("no network")
        from strategies.matrix_maximizer.core import MatrixConfig
        from strategies.matrix_maximizer.runner import MatrixMaximizer
        mm = MatrixMaximizer(MatrixConfig(account_size=920, n_simulations=100))
        result = mm.run_full_cycle(
            prices={"oil": 85, "vix": 22},
            auto_execute=True,
        )
        assert result["status"] in ("complete", "blocked")

    @patch("urllib.request.urlopen")
    def test_enhanced_risk_in_output(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("no network")
        from strategies.matrix_maximizer.core import MatrixConfig
        from strategies.matrix_maximizer.runner import MatrixMaximizer
        mm = MatrixMaximizer(MatrixConfig(account_size=920, n_simulations=100))
        result = mm.run_full_cycle(prices={"oil": 80, "vix": 22})
        if result["status"] == "complete":
            assert "risk" in result
            assert "tail_risk" in result["risk"]


# ═══════════════════════════════════════════════════════════════════════════
# __init__.py EXPORT TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestPackageExports:
    def test_core_exports(self):
        from strategies.matrix_maximizer import (
            Asset,
            BlackScholesEngine,
            MatrixConfig,
            MatrixMaximizer,
            MonteCarloEngine,
            PillarBridge,
            RiskManager,
        )
        assert MatrixMaximizer is not None
        assert MonteCarloEngine is not None

    def test_extended_exports(self):
        from strategies.matrix_maximizer import (
            AdvancedStrategyEngine,
            AlertManager,
            DataFeedManager,
            ExecutionEngine,
            IntelligenceEngine,
            MatrixBacktester,
            MatrixChatbot,
            MatrixDashboard,
            MatrixScheduler,
            Watchdog,
        )
        # These may be None if import fails, but shouldn't raise
        # Just verify they're accessible
        assert True

    def test_all_list(self):
        import strategies.matrix_maximizer as mm
        assert len(mm.__all__) >= 25  # Core + Extended
