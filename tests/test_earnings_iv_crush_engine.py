from __future__ import annotations

import math

import pytest

from strategies.earnings_iv_crush_engine import (
    CrushTradeSetup,
    EarningsEvent,
    EarningsSeasonScanner,
    ExpectedMoveCalculator,
    IVCrushEngine,
)


# ═══════════════════════════════════════════════════════════════════════════
# EarningsEvent
# ═══════════════════════════════════════════════════════════════════════════

class TestEarningsEvent:
    def test_defaults(self):
        e = EarningsEvent(symbol="AAPL", report_date="2025-07-31",
                          report_timing="AMC", stock_price=200)
        assert e.symbol == "AAPL"
        assert e.front_week_iv == 0.0
        assert e.last_4_moves == []
        assert e.beat_rate == 0.0

    def test_iv_premium_zero_history(self):
        e = EarningsEvent(symbol="X", report_date="d", report_timing="AMC",
                          stock_price=100, front_week_iv=0.40, historical_iv_mean=0)
        assert e.iv_premium == 0

    def test_iv_premium_computed(self):
        e = EarningsEvent(symbol="X", report_date="d", report_timing="AMC",
                          stock_price=100, front_week_iv=0.45, historical_iv_mean=0.30)
        # (0.45/0.30 - 1)*100 = 50
        assert e.iv_premium == pytest.approx(50.0)

    def test_move_vs_expected_zero_em(self):
        e = EarningsEvent(symbol="X", report_date="d", report_timing="AMC",
                          stock_price=100, expected_move_pct=0,
                          avg_historical_move=3.0)
        assert e.move_vs_expected == 0

    def test_move_vs_expected_computed(self):
        e = EarningsEvent(symbol="X", report_date="d", report_timing="AMC",
                          stock_price=100, expected_move_pct=5.0,
                          avg_historical_move=4.0)
        assert e.move_vs_expected == pytest.approx(0.8)


# ═══════════════════════════════════════════════════════════════════════════
# CrushTradeSetup
# ═══════════════════════════════════════════════════════════════════════════

class TestCrushTradeSetup:
    def test_construction(self):
        s = CrushTradeSetup(
            strategy="x", strikes=[1, 2], credit=1.0, max_profit=100,
            max_loss=400, expected_iv_crush=0.20, expected_pnl=50,
            confidence=70, edge="e", risk_notes=["a"],
        )
        assert s.strategy == "x"
        assert s.strikes == [1, 2]
        assert len(s.risk_notes) == 1


# ═══════════════════════════════════════════════════════════════════════════
# ExpectedMoveCalculator
# ═══════════════════════════════════════════════════════════════════════════

class TestExpectedMoveFromStraddle:
    def test_basic(self):
        r = ExpectedMoveCalculator.from_straddle(stock_price=100, straddle_price=10)
        assert r["expected_move"] == 8.5
        assert r["expected_pct"] == 8.5
        assert r["upper_bound"] == 108.5
        assert r["lower_bound"] == 91.5
        assert r["straddle_cost"] == 10
        assert r["straddle_pct"] == 10.0
        assert r["method"] == "straddle_85pct_rule"


class TestExpectedMoveFromIv:
    def test_basic(self):
        r = ExpectedMoveCalculator.from_iv(stock_price=100, iv=0.30, dte=30)
        # one_sd = 100 * 0.30 * sqrt(30/365) = ~8.6
        expected = 100 * 0.30 * math.sqrt(30 / 365)
        assert r["expected_move_1sd"] == round(expected, 2)
        assert r["expected_move_2sd"] == round(expected * 2, 2)
        assert r["prob_within_1sd"] == 68.2
        assert r["prob_within_2sd"] == 95.4
        assert r["method"] == "implied_volatility"
        assert r["upper_1sd"] == round(100 + expected, 2)
        assert r["lower_1sd"] == round(100 - expected, 2)


class TestCompareToHistorical:
    def test_no_history(self):
        r = ExpectedMoveCalculator.compare_to_historical(5.0, [])
        assert r == {"comparison": "no_history"}

    def test_overpriced_sell_premium(self):
        # avg = 2.0, expected = 5.0 → ratio 2.5 → SELL_PREMIUM
        r = ExpectedMoveCalculator.compare_to_historical(5.0, [1.0, 2.0, 3.0, 2.0])
        assert r["expected_move_pct"] == 5.0
        assert r["avg_historical_move_pct"] == 2.0
        assert r["max_historical_move_pct"] == 3.0
        assert r["expected_vs_avg_ratio"] == 2.5
        assert r["overpriced"] is True
        assert r["edge"] == "SELL_PREMIUM"
        assert r["times_exceeded"] == 0
        assert r["pct_exceeded"] == 0.0

    def test_neutral_edge(self):
        # avg=5, exp=5 → ratio=1.0 → between 0.85 and 1.15 → NEUTRAL
        r = ExpectedMoveCalculator.compare_to_historical(5.0, [5.0, 5.0])
        assert r["edge"] == "NEUTRAL"
        assert r["overpriced"] is False

    def test_buy_premium_edge(self):
        # avg=10, exp=5 → ratio=0.5 → BUY_PREMIUM
        r = ExpectedMoveCalculator.compare_to_historical(5.0, [10.0, 10.0])
        assert r["edge"] == "BUY_PREMIUM"

    def test_negative_moves_use_abs(self):
        r = ExpectedMoveCalculator.compare_to_historical(5.0, [-3.0, 3.0])
        assert r["avg_historical_move_pct"] == 3.0

    def test_times_exceeded_count(self):
        # exp=3.0; |moves|=4,5,2,1 → 2 exceed
        r = ExpectedMoveCalculator.compare_to_historical(3.0, [4.0, -5.0, 2.0, 1.0])
        assert r["times_exceeded"] == 2
        assert r["pct_exceeded"] == 50.0

    def test_zero_avg_ratio_zero(self):
        # All zeros → avg=0 → ratio=0
        r = ExpectedMoveCalculator.compare_to_historical(5.0, [0.0, 0.0])
        assert r["expected_vs_avg_ratio"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# IVCrushEngine
# ═══════════════════════════════════════════════════════════════════════════

def _event(**overrides) -> EarningsEvent:
    base = dict(
        symbol="AAPL", report_date="2025-07-31", report_timing="AMC",
        stock_price=200, front_week_iv=0.40, back_week_iv=0.25,
        historical_iv_mean=0.25, iv_rank=70,
        atm_straddle_price=10.0, expected_move_dollar=8.5,
        expected_move_pct=4.25, avg_historical_move=3.0,
        last_4_moves=[2.5, -3.0, 3.2, -2.8],
    )
    base.update(overrides)
    return EarningsEvent(**base)


class TestIronCondorSetup:
    def test_strikes_around_expected_move(self):
        eng = IVCrushEngine(_event(stock_price=200, expected_move_dollar=10))
        s = eng.iron_condor_setup(wing_width=5)
        # short strikes ±10 from 200; longs ±5 beyond
        assert s.strikes == [185, 190, 210, 215]
        assert s.strategy.startswith("Iron Condor")

    def test_zero_expected_move_falls_back_to_iv(self):
        eng = IVCrushEngine(_event(expected_move_dollar=0,
                                   stock_price=100, front_week_iv=0.30))
        s = eng.iron_condor_setup(wing_width=5)
        # em = 100 * 0.30 * sqrt(1/365) ≈ 1.57 → short = 98 / 102
        assert s.strikes[1] == round(100 - 100 * 0.30 * math.sqrt(1/365))
        assert s.strikes[2] == round(100 + 100 * 0.30 * math.sqrt(1/365))

    def test_credit_and_max_loss(self):
        eng = IVCrushEngine(_event())
        s = eng.iron_condor_setup(wing_width=5)
        # est_credit = 5 * 0.35 = 1.75; max_loss = (5-1.75)*100 = 325
        assert s.credit == 1.75
        assert s.max_profit == 175.0
        assert s.max_loss == 325.0

    def test_high_confidence_when_move_overpriced(self):
        # move_vs_expected = 3.0 / 4.25 = 0.706 < 1 → conf=70
        eng = IVCrushEngine(_event())
        assert eng.iron_condor_setup().confidence == 70

    def test_low_confidence_when_move_underpriced(self):
        eng = IVCrushEngine(_event(avg_historical_move=10.0, expected_move_pct=4.0))
        # ratio 2.5 > 1 → conf=55
        assert eng.iron_condor_setup().confidence == 55

    def test_risk_notes_count(self):
        s = IVCrushEngine(_event()).iron_condor_setup()
        assert len(s.risk_notes) == 4

    def test_expected_iv_crush_50pct(self):
        eng = IVCrushEngine(_event(front_week_iv=0.40))
        s = eng.iron_condor_setup()
        assert s.expected_iv_crush == 0.20  # 0.40 * 0.5


class TestCalendarSpreadSetup:
    def test_strikes_atm(self):
        eng = IVCrushEngine(_event(stock_price=200))
        s = eng.calendar_spread_setup()
        assert s.strikes == [200]
        assert s.strategy.startswith("Calendar Spread")

    def test_back_iv_default_when_zero(self):
        # back_iv=0 → falls back to front * 0.85
        eng = IVCrushEngine(_event(front_week_iv=0.40, back_week_iv=0))
        s = eng.calendar_spread_setup()
        # No assertion error and strategy returned
        assert s.strategy.startswith("Calendar Spread")

    def test_debit_trade_negative_credit(self):
        eng = IVCrushEngine(_event(stock_price=100))
        s = eng.calendar_spread_setup()
        # debit = 100 * 0.015 = 1.5
        assert s.credit == -1.5
        assert s.max_loss == 150.0
        assert s.max_profit == 225.0  # 1.5*1.5*100

    def test_confidence_fixed_65(self):
        s = IVCrushEngine(_event()).calendar_spread_setup()
        assert s.confidence == 65

    def test_risk_notes_count(self):
        s = IVCrushEngine(_event()).calendar_spread_setup()
        assert len(s.risk_notes) == 4


class TestButterflyPinSetup:
    def test_strikes(self):
        eng = IVCrushEngine(_event(stock_price=200))
        s = eng.butterfly_pin_setup(width=5)
        assert s.strikes == [195, 200, 205]
        assert s.strategy.startswith("ATM Butterfly")

    def test_debit_and_max_profit(self):
        eng = IVCrushEngine(_event())
        s = eng.butterfly_pin_setup(width=10)
        # debit = 10*0.10 = 1.0; max_profit = (10-1)*100 = 900; max_loss=100
        assert s.credit == -1.0
        assert s.max_loss == 100.0
        assert s.max_profit == 900.0

    def test_confidence_low(self):
        s = IVCrushEngine(_event()).butterfly_pin_setup()
        assert s.confidence == 30

    def test_expected_pnl_15pct_of_max(self):
        s = IVCrushEngine(_event()).butterfly_pin_setup(width=5)
        assert s.expected_pnl == round(s.max_profit * 0.15, 2)


class TestRankStrategies:
    def test_returns_three_sorted_descending(self):
        ranked = IVCrushEngine(_event()).rank_strategies()
        assert len(ranked) == 3
        scores = [r["score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_each_entry_has_required_keys(self):
        ranked = IVCrushEngine(_event()).rank_strategies()
        for r in ranked:
            assert {"strategy", "score", "credit", "max_profit",
                    "max_loss", "confidence", "edge"} <= set(r.keys())

    def test_score_penalty_when_move_underpriced(self):
        # move_vs_expected < 0.8 deducts 10
        eng_under = IVCrushEngine(_event(avg_historical_move=2.0,
                                         expected_move_pct=4.0))  # ratio 0.5
        eng_normal = IVCrushEngine(_event(avg_historical_move=4.0,
                                          expected_move_pct=4.0))  # ratio 1.0
        # IronCondor confidence differs (70 vs 55), so compare butterfly which has fixed conf=30
        ranked_under = {r["strategy"]: r["score"] for r in eng_under.rank_strategies()}
        ranked_normal = {r["strategy"]: r["score"] for r in eng_normal.rank_strategies()}
        bf = "ATM Butterfly (Earnings Pin Play)"
        # Both have same butterfly setup numerics; under-event applies -10 penalty
        assert ranked_under[bf] == ranked_normal[bf] - 10


# ═══════════════════════════════════════════════════════════════════════════
# EarningsSeasonScanner
# ═══════════════════════════════════════════════════════════════════════════

class TestEarningsSeasonScanner:
    def test_empty(self):
        assert EarningsSeasonScanner.scan([]) == []

    def test_high_score_event_recommended(self):
        # iv_premium=80, move_vs_expected=0.5, ivr=80, low std → max score
        e = _event(
            front_week_iv=0.45, historical_iv_mean=0.25,  # premium = 80
            avg_historical_move=2.0, expected_move_pct=4.0,  # ratio 0.5
            iv_rank=80,
            last_4_moves=[2.0, 2.1, 1.9, 2.0],  # std ≈ 0
        )
        result = EarningsSeasonScanner.scan([e])
        assert len(result) == 1
        r = result[0]
        # 25 (iv_prem>50) + 25 (move<0.8) + 20 (ivr>70) + 15 (std<2) = 85
        assert r["score"] == 85.0
        assert r["recommended"] is True
        assert "HIGH_IV_PREMIUM" in r["flags"]
        assert "MOVE_OVERPRICED" in r["flags"]
        assert "HIGH_IVR" in r["flags"]
        assert "CONSISTENT_MOVER" in r["flags"]

    def test_mid_tier_scoring(self):
        # iv_premium=40, move_vs_expected=0.9, ivr=60 → 15+15+10 = 40
        e = _event(
            front_week_iv=0.35, historical_iv_mean=0.25,  # premium = 40
            avg_historical_move=3.6, expected_move_pct=4.0,  # ratio 0.9
            iv_rank=60,
            last_4_moves=[1.0, 5.0, 1.0, 5.0],  # std=2 → no bonus
        )
        result = EarningsSeasonScanner.scan([e])
        # 15 (iv>30) + 15 (ratio<1) + 10 (ivr>50) = 40
        assert result[0]["score"] == 40.0
        assert result[0]["recommended"] is False

    def test_low_score_no_flags(self):
        e = _event(
            front_week_iv=0.26, historical_iv_mean=0.25,  # premium ≈ 4
            avg_historical_move=5.0, expected_move_pct=4.0,  # ratio 1.25
            iv_rank=30,
            last_4_moves=[],
        )
        result = EarningsSeasonScanner.scan([e])
        assert result[0]["score"] == 0
        assert result[0]["flags"] == []
        assert result[0]["recommended"] is False

    def test_results_sorted_descending(self):
        events = [
            _event(symbol="LOW", front_week_iv=0.26, historical_iv_mean=0.25,
                   avg_historical_move=10, expected_move_pct=4, iv_rank=30,
                   last_4_moves=[]),
            _event(symbol="HIGH", front_week_iv=0.45, historical_iv_mean=0.25,
                   avg_historical_move=2, expected_move_pct=4, iv_rank=80,
                   last_4_moves=[2.0, 2.0, 2.0, 2.0]),
        ]
        result = EarningsSeasonScanner.scan(events)
        assert result[0]["symbol"] == "HIGH"
        assert result[1]["symbol"] == "LOW"

    def test_required_keys(self):
        result = EarningsSeasonScanner.scan([_event()])[0]
        for key in ("symbol", "report_date", "timing", "score", "iv_premium",
                    "expected_move_pct", "avg_actual_move_pct", "move_ratio",
                    "iv_rank", "flags", "recommended"):
            assert key in result

    def test_consistent_mover_threshold(self):
        # std=2 exactly → not bonused (< not <=)
        e = _event(last_4_moves=[1.0, 5.0, 1.0, 5.0])
        # std = sqrt(((1-3)^2 + (5-3)^2 + (1-3)^2 + (5-3)^2)/4) = sqrt(16/4) = 2
        result = EarningsSeasonScanner.scan([e])
        assert "CONSISTENT_MOVER" not in result[0]["flags"]

    def test_consistent_mover_added_when_low_std(self):
        e = _event(last_4_moves=[3.0, 3.0, 3.0, 3.0])  # std=0
        result = EarningsSeasonScanner.scan([e])
        assert "CONSISTENT_MOVER" in result[0]["flags"]
