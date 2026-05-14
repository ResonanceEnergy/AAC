from __future__ import annotations

import math
from datetime import time

import pytest

from strategies.zero_dte_gamma_engine import (
    GammaScalpPosition,
    GammaScalpTracker,
    IntradayAnalytics,
    IntradayLevel,
    MaxPainPredictor,
    OpeningRangeEngine,
    RiskMode,
    SessionDetector,
    SessionPhase,
    ZeroDTEEngine,
    ZeroDTESetup,
    ZeroDTEStrategy,
)


# ═══════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════

class TestDataclasses:
    def test_intraday_level(self):
        lvl = IntradayLevel(price=520, level_type="support", strength=80, source="gex")
        assert lvl.price == 520
        assert lvl.strength == 80

    def test_zero_dte_setup(self):
        s = ZeroDTESetup(
            strategy=ZeroDTEStrategy.SCALP_GAMMA,
            entry_time="morning", exit_deadline="15:30",
            strikes=[520], estimated_cost=100, max_risk=100,
            target_profit=50, risk_reward=0.5,
            session_phase=SessionPhase.MORNING,
            conviction=50, notes=[]
        )
        assert s.strategy == ZeroDTEStrategy.SCALP_GAMMA

    def test_gamma_scalp_total_pnl(self):
        p = GammaScalpPosition(
            symbol="SPY", entry_price=520, option_strike=520,
            option_type="call", option_cost=200, delta_at_entry=0.5,
            realized_scalp_pnl=150,
        )
        assert p.total_pnl == -50  # 150 - 200

    def test_intraday_analytics_defaults(self):
        ia = IntradayAnalytics(session_date="2026-04-22")
        assert ia.opening_range_high == 0.0
        assert ia.total_0dte_volume == 0


# ═══════════════════════════════════════════════════════════════════════════
# SessionDetector
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionDetector:
    @pytest.mark.parametrize("t,expected", [
        (time(9, 0), SessionPhase.PRE_MARKET),
        (time(9, 29), SessionPhase.PRE_MARKET),
        (time(9, 30), SessionPhase.OPENING_RANGE),
        (time(9, 59), SessionPhase.OPENING_RANGE),
        (time(10, 0), SessionPhase.MORNING),
        (time(11, 0), SessionPhase.MORNING),
        (time(11, 30), SessionPhase.MIDDAY),
        (time(13, 59), SessionPhase.MIDDAY),
        (time(14, 0), SessionPhase.AFTERNOON),
        (time(15, 29), SessionPhase.AFTERNOON),
        (time(15, 30), SessionPhase.POWER_HOUR),
        (time(15, 44), SessionPhase.POWER_HOUR),
        (time(15, 45), SessionPhase.FINAL_15),
        (time(15, 59), SessionPhase.FINAL_15),
        (time(16, 0), SessionPhase.AFTER_HOURS),
        (time(20, 0), SessionPhase.AFTER_HOURS),
    ])
    def test_detect_phase_boundaries(self, t, expected):
        assert SessionDetector.detect_phase(t) == expected

    def test_get_characteristics_known_phase(self):
        c = SessionDetector.get_characteristics(SessionPhase.MORNING)
        assert c["phase"] == "morning"
        assert "volatility" in c
        assert isinstance(c["best_strategies"], list)
        assert all(isinstance(s, str) for s in c["best_strategies"])

    def test_get_characteristics_unknown_phase(self):
        # PRE_MARKET not in PHASE_CHARACTERISTICS
        c = SessionDetector.get_characteristics(SessionPhase.PRE_MARKET)
        assert c["phase"] == "pre_market"
        assert c["best_strategies"] == []
        assert c["avoid"] == []

    def test_opening_range_avoids_iron_condor(self):
        c = SessionDetector.get_characteristics(SessionPhase.OPENING_RANGE)
        assert "sell_iron_condor" in c["avoid"]

    def test_final_15_avoids_everything(self):
        c = SessionDetector.get_characteristics(SessionPhase.FINAL_15)
        assert c["best_strategies"] == []
        # All ZeroDTEStrategy enum values in avoid
        assert len(c["avoid"]) == len(list(ZeroDTEStrategy))


# ═══════════════════════════════════════════════════════════════════════════
# ZeroDTEEngine — Iron Condor
# ═══════════════════════════════════════════════════════════════════════════

class TestZeroDTEEngineInit:
    def test_conservative_risk(self):
        e = ZeroDTEEngine(100_000, RiskMode.CONSERVATIVE)
        assert e.max_risk_pct == 0.005

    def test_moderate_risk(self):
        e = ZeroDTEEngine(100_000, RiskMode.MODERATE)
        assert e.max_risk_pct == 0.01

    def test_aggressive_risk(self):
        e = ZeroDTEEngine(100_000, RiskMode.AGGRESSIVE)
        assert e.max_risk_pct == 0.02

    def test_underlying_price_stored(self):
        e = ZeroDTEEngine(100_000, underlying_price=550)
        assert e.S == 550


class TestIronCondor:
    def _engine(self, **k):
        return ZeroDTEEngine(
            capital=k.get("capital", 100_000),
            risk_mode=k.get("risk_mode", RiskMode.MODERATE),
            underlying_price=k.get("S", 520),
        )

    def test_opening_range_returns_none(self):
        e = self._engine()
        assert e.generate_iron_condor(SessionPhase.OPENING_RANGE, 3.5, 0.20) is None

    def test_final_15_returns_none(self):
        e = self._engine()
        assert e.generate_iron_condor(SessionPhase.FINAL_15, 3.5, 0.20) is None

    def test_power_hour_returns_none(self):
        e = self._engine()
        assert e.generate_iron_condor(SessionPhase.POWER_HOUR, 3.5, 0.20) is None

    def test_morning_generates(self):
        e = self._engine()
        s = e.generate_iron_condor(SessionPhase.MORNING, 3.5, 0.20)
        assert s is not None
        assert s.strategy == ZeroDTEStrategy.SELL_IRON_CONDOR
        # or_half=1.75, margin=2.275 → put_short=round(520-2.275)=518, call_short=522
        assert s.strikes == [513.0, 518.0, 522.0, 527.0]
        assert s.conviction == 55

    def test_midday_higher_conviction(self):
        e = self._engine()
        s = e.generate_iron_condor(SessionPhase.MIDDAY, 3.5, 0.20)
        assert s is not None
        assert s.conviction == 65

    def test_afternoon_generates(self):
        e = self._engine()
        s = e.generate_iron_condor(SessionPhase.AFTERNOON, 3.5, 0.20)
        assert s is not None
        # time_factor=0.25, est_credit=5*0.25=1.25, max_loss=(5-1.25)*100=375 per spread
        # budget=100k*0.01=1000; contracts=int(1000/375)=2 → total max_risk=750
        assert s.max_risk == 750.0

    def test_credit_scales_by_phase(self):
        e = self._engine()
        # Morning tf=0.45, midday=0.35, afternoon=0.25
        morning = e.generate_iron_condor(SessionPhase.MORNING, 3.5, 0.20)
        midday = e.generate_iron_condor(SessionPhase.MIDDAY, 3.5, 0.20)
        assert morning.estimated_cost > midday.estimated_cost

    def test_wing_width_is_5(self):
        e = self._engine()
        s = e.generate_iron_condor(SessionPhase.MORNING, 3.5, 0.20)
        # put_long should be 5 below put_short; call_long 5 above call_short
        put_long, put_short, call_short, call_long = s.strikes
        assert put_short - put_long == 5
        assert call_long - call_short == 5


# ═══════════════════════════════════════════════════════════════════════════
# ZeroDTEEngine — Gamma Scalp
# ═══════════════════════════════════════════════════════════════════════════

class TestGammaScalp:
    def _engine(self):
        return ZeroDTEEngine(100_000, RiskMode.MODERATE, underlying_price=520)

    def test_midday_returns_none(self):
        assert self._engine().generate_gamma_scalp(SessionPhase.MIDDAY, 0.20) is None

    def test_final_15_returns_none(self):
        assert self._engine().generate_gamma_scalp(SessionPhase.FINAL_15, 0.20) is None

    def test_opening_range_generates(self):
        s = self._engine().generate_gamma_scalp(SessionPhase.OPENING_RANGE, 0.20)
        assert s is not None
        assert s.strategy == ZeroDTEStrategy.SCALP_GAMMA
        assert s.conviction == 50
        # hours_left=6.5, option_cost = 520*0.20*sqrt(6.5/(252*6.5))
        expected_cost = 520 * 0.20 * math.sqrt(6.5 / (252 * 6.5))
        max_risk = expected_cost * 100
        assert s.max_risk == pytest.approx(round(max_risk, 2))

    def test_morning_generates(self):
        s = self._engine().generate_gamma_scalp(SessionPhase.MORNING, 0.20)
        assert s is not None

    def test_afternoon_generates(self):
        s = self._engine().generate_gamma_scalp(SessionPhase.AFTERNOON, 0.20)
        assert s is not None

    def test_power_hour_generates(self):
        s = self._engine().generate_gamma_scalp(SessionPhase.POWER_HOUR, 0.20)
        assert s is not None

    def test_strike_at_underlying(self):
        s = self._engine().generate_gamma_scalp(SessionPhase.MORNING, 0.20)
        assert s.strikes == [520.0]

    def test_risk_reward_is_half(self):
        s = self._engine().generate_gamma_scalp(SessionPhase.MORNING, 0.20)
        assert s.risk_reward == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# GammaScalpTracker
# ═══════════════════════════════════════════════════════════════════════════

class TestGammaScalpTracker:
    def _pos(self, **k):
        return GammaScalpPosition(
            symbol="SPY", entry_price=k.get("entry_price", 520),
            option_strike=520, option_type="call",
            option_cost=k.get("option_cost", 200),
            delta_at_entry=0.5,
            current_delta=k.get("current_delta", 0.5),
            shares_hedged=k.get("shares_hedged", 0),
            realized_scalp_pnl=k.get("realized_scalp_pnl", 0),
        )

    def test_should_not_hedge_small_move(self):
        t = GammaScalpTracker(self._pos(), hedge_interval=1.0)
        out = t.should_hedge(current_price=520.5, current_delta=0.52)
        assert out["should_hedge"] is False
        assert "next_hedge_at" in out

    def test_should_hedge_on_price_move(self):
        t = GammaScalpTracker(self._pos(), hedge_interval=1.0)
        out = t.should_hedge(current_price=521.5, current_delta=0.60)
        assert out["should_hedge"] is True
        assert "action" in out
        assert out["action"] in ("BUY", "SELL")

    def test_should_hedge_on_delta_change(self):
        t = GammaScalpTracker(self._pos(), hedge_interval=5.0)
        # Small price move but big delta change
        out = t.should_hedge(current_price=520.5, current_delta=0.70)
        assert out["should_hedge"] is True

    def test_hedge_action_sell_when_positive_trade(self):
        # shares_needed = int((0.75-0.5)*100)=25, current=0, trade=25 → SELL
        t = GammaScalpTracker(self._pos(shares_hedged=0), hedge_interval=1.0)
        out = t.should_hedge(current_price=522, current_delta=0.75)
        assert out["action"] == "SELL"
        assert out["shares"] == 25

    def test_hedge_action_buy_when_negative_trade(self):
        # shares_needed = int((0.30-0.5)*100) = int(-20) = -20
        # current=0 → trade=-20 → BUY
        t = GammaScalpTracker(self._pos(shares_hedged=0), hedge_interval=1.0)
        out = t.should_hedge(current_price=518, current_delta=0.30)
        assert out["action"] == "BUY"
        assert out["shares"] == 20

    def test_theta_burn_expired(self):
        t = GammaScalpTracker(self._pos(), hedge_interval=1.0)
        out = t.theta_burn_rate(hours_remaining=0, option_cost=200)
        assert out["status"] == "EXPIRED"
        assert out["action"] == "CLOSE_NOW"

    def test_theta_burn_negative_hours(self):
        t = GammaScalpTracker(self._pos(), hedge_interval=1.0)
        out = t.theta_burn_rate(hours_remaining=-1, option_cost=200)
        assert out["status"] == "EXPIRED"

    def test_theta_burn_active(self):
        t = GammaScalpTracker(self._pos(realized_scalp_pnl=50), hedge_interval=1.0)
        out = t.theta_burn_rate(hours_remaining=4, option_cost=200)
        assert "theta_per_hour" in out
        assert "theta_remaining" in out
        assert out["scalp_pnl"] == 50
        assert out["hours_left"] == 4.0
        assert isinstance(out["profitable"], bool)

    def test_theta_burn_profitable(self):
        t = GammaScalpTracker(self._pos(realized_scalp_pnl=1000), hedge_interval=1.0)
        out = t.theta_burn_rate(hours_remaining=4, option_cost=200)
        assert out["profitable"] is True
        assert out["net_pnl"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# OpeningRangeEngine
# ═══════════════════════════════════════════════════════════════════════════

class TestOpeningRangeEngine:
    def test_compute_opening_range(self):
        out = OpeningRangeEngine.compute_opening_range(521.50, 518.20, 519.80)
        assert out["or_high"] == 521.50
        assert out["or_low"] == 518.20
        assert out["or_width"] == 3.3
        assert out["midpoint"] == 519.85
        assert out["vwap"] == 519.80
        # break_up_t1 = or_high + width*1.0 = 521.50 + 3.30 = 524.80
        assert out["break_up_t1"] == 524.80
        # break_down_t1 = or_low - width*1.0 = 514.90
        assert out["break_down_t1"] == 514.90

    def test_compute_opening_range_extensions(self):
        out = OpeningRangeEngine.compute_opening_range(10, 0, 5)
        # width=10, midpoint=5
        assert out["or_width"] == 10
        assert out["break_up_t1"] == 20  # high + 10
        assert out["break_up_t2"] == 25  # high + 15
        assert out["break_up_t3"] == 30  # high + 20
        assert out["break_down_t3"] == -20  # low - 20

    def test_generate_breakout_trade_up(self):
        or_levels = OpeningRangeEngine.compute_opening_range(521.50, 518.20, 519.80)
        t = OpeningRangeEngine.generate_breakout_trade(or_levels, "up", 100_000, 521.00)
        assert t["direction"] == "up"
        assert t["option_type"] == "call"
        assert t["entry_trigger"] == 521.50
        assert t["target"] == 524.80
        assert t["stop_loss"] == 519.85
        assert t["strike"] == 522.0
        assert t["contracts"] >= 1
        assert "notes" in t
        assert t["max_risk"] == 500.0  # 100k * 0.005

    def test_generate_breakout_trade_down(self):
        or_levels = OpeningRangeEngine.compute_opening_range(521.50, 518.20, 519.80)
        t = OpeningRangeEngine.generate_breakout_trade(or_levels, "down", 100_000, 518.00)
        assert t["direction"] == "down"
        assert t["option_type"] == "put"
        assert t["entry_trigger"] == 518.20
        assert t["target"] == 514.90

    def test_breakout_risk_reward(self):
        or_levels = OpeningRangeEngine.compute_opening_range(10, 0, 5)
        t = OpeningRangeEngine.generate_breakout_trade(or_levels, "up", 100_000, 10.0)
        # trigger=10, target=20, stop=5 → rr = (20-10)/(10-5) = 2.0
        assert t["risk_reward"] == 2.0


# ═══════════════════════════════════════════════════════════════════════════
# MaxPainPredictor
# ═══════════════════════════════════════════════════════════════════════════

class TestMaxPainPredictor:
    def test_basic_max_pain(self):
        strikes = [510, 515, 520, 525, 530]
        call_oi = {510: 5000, 515: 12000, 520: 25000, 525: 18000, 530: 8000}
        put_oi = {510: 8000, 515: 20000, 520: 22000, 525: 10000, 530: 3000}
        out = MaxPainPredictor.calculate_max_pain(strikes, call_oi, put_oi)
        assert "max_pain" in out
        assert out["max_pain"] in strikes
        assert "total_pain_at_mp" in out
        assert len(out["magnetic_strikes"]) == 3

    def test_magnetic_sorted_by_oi_desc(self):
        strikes = [510, 515, 520]
        call_oi = {510: 100, 515: 500, 520: 1000}
        put_oi = {510: 100, 515: 100, 520: 100}
        out = MaxPainPredictor.calculate_max_pain(strikes, call_oi, put_oi)
        # combined: 510=200, 515=600, 520=1100
        mag = out["magnetic_strikes"]
        assert mag[0]["strike"] == 520
        assert mag[0]["total_oi"] == 1100
        assert mag[1]["strike"] == 515

    def test_max_pain_symmetric_oi(self):
        # Equal call/put OI at center strike → max pain should be center
        strikes = [500, 510, 520]
        call_oi = {500: 100, 510: 1000, 520: 100}
        put_oi = {500: 100, 510: 1000, 520: 100}
        out = MaxPainPredictor.calculate_max_pain(strikes, call_oi, put_oi)
        assert out["max_pain"] == 510

    def test_pain_landscape_in_output(self):
        strikes = [500, 510, 520]
        out = MaxPainPredictor.calculate_max_pain(
            strikes, {510: 100}, {510: 100}
        )
        assert "pain_landscape" in out
        assert isinstance(out["pain_landscape"], dict)

    def test_empty_oi(self):
        strikes = [500, 510, 520]
        out = MaxPainPredictor.calculate_max_pain(strikes, {}, {})
        assert out["max_pain"] in strikes
        # All pain = 0 → min just returns first
