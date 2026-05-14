from __future__ import annotations

import math

import pytest

from strategies.options_income_systems import (
    CoveredCallScreener,
    CreditSpreadEngine,
    IncomePortfolio,
    IncomePosition,
    IncomeYieldTracker,
    IronCondorEngine,
    ManagementAction,
    WheelEngine,
    WheelPhase,
    WheelPosition,
)


# ═══════════════════════════════════════════════════════════════════════════
# IncomePosition
# ═══════════════════════════════════════════════════════════════════════════

def _pos(**k):
    base = dict(symbol="SPY", strategy="iron_condor", entry_date="2026-01-01",
                expiry_date="2026-02-15", strikes=[490, 500, 540, 550],
                credit_received=2.0, current_value=1.0, quantity=1,
                dte_at_entry=45, dte_remaining=20)
    base.update(k)
    return IncomePosition(**base)


class TestIncomePosition:
    def test_pnl(self):
        p = _pos(credit_received=2.0, current_value=0.5)
        assert p.pnl == 150  # (2.0-0.5)*100

    def test_pnl_pct(self):
        p = _pos(credit_received=2.0, current_value=1.0)
        assert p.pnl_pct == 50

    def test_pnl_pct_zero_credit(self):
        p = _pos(credit_received=0)
        assert p.pnl_pct == 0

    def test_theta_decay_pct_zero_dte_at_entry(self):
        p = _pos(dte_at_entry=0)
        assert p.theta_decay_pct == 100

    def test_theta_decay_pct_basic(self):
        p = _pos(dte_at_entry=45, dte_remaining=20)
        # (1 - sqrt(20/45)) * 100
        expected = (1 - math.sqrt(20/45)) * 100
        assert p.theta_decay_pct == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════════
# WheelPosition
# ═══════════════════════════════════════════════════════════════════════════

class TestWheelPosition:
    def test_effective_cost_basis(self):
        w = WheelPosition(symbol="X", phase=WheelPhase.IDLE,
                          cost_basis=10000, total_premium_collected=500)
        assert w.effective_cost_basis == 9500

    def test_annualized_yield_zero_cost_basis(self):
        w = WheelPosition(symbol="X", phase=WheelPhase.IDLE)
        assert w.annualized_yield == 0

    def test_annualized_yield_no_history(self):
        w = WheelPosition(symbol="X", phase=WheelPhase.IDLE, cost_basis=10000)
        assert w.annualized_yield == 0

    def test_annualized_yield_invalid_date(self):
        w = WheelPosition(symbol="X", phase=WheelPhase.IDLE,
                          cost_basis=10000, total_premium_collected=500,
                          history=[{"date": "not-a-date"}])
        assert w.annualized_yield == 0

    def test_annualized_yield_empty_date(self):
        w = WheelPosition(symbol="X", phase=WheelPhase.IDLE,
                          cost_basis=10000, total_premium_collected=500,
                          history=[{"date": ""}])
        assert w.annualized_yield == 0


# ═══════════════════════════════════════════════════════════════════════════
# IncomePortfolio
# ═══════════════════════════════════════════════════════════════════════════

class TestIncomePortfolio:
    def test_total_credits(self):
        p = IncomePortfolio()
        p.positions = [_pos(credit_received=2.0, quantity=2),
                       _pos(credit_received=1.5, status="closed")]
        # 2.0*2*100=400; closed excluded
        assert p.total_credits == 400

    def test_total_pnl_open_only(self):
        p = IncomePortfolio()
        p.positions = [_pos(credit_received=2.0, current_value=0.5, quantity=2),
                       _pos(credit_received=2.0, current_value=0.5, status="closed")]
        # pnl=150 per contract * 2 quantity = 300; closed excluded
        assert p.total_pnl == 300

    def test_positions_count(self):
        p = IncomePortfolio()
        p.positions = [_pos(), _pos(status="closed"), _pos()]
        assert p.positions_count == 2

    def test_capital_deployed_covered_call(self):
        p = IncomePortfolio()
        p.positions = [_pos(strategy="covered_call", underlying_at_entry=500, quantity=1)]
        assert p.capital_deployed == 50000  # 500*100*1

    def test_capital_deployed_iron_condor(self):
        p = IncomePortfolio()
        # width = 550-490 = 60; (60-2.0)*100*1 = 5800
        p.positions = [_pos(strategy="iron_condor", strikes=[490, 500, 540, 550],
                            credit_received=2.0)]
        assert p.capital_deployed == 5800

    def test_capital_deployed_bull_put_spread(self):
        p = IncomePortfolio()
        # width=10; (10-1.5)*100=850
        p.positions = [_pos(strategy="bull_put_spread", strikes=[490, 500],
                            credit_received=1.5)]
        assert p.capital_deployed == 850

    def test_capital_deployed_skips_closed(self):
        p = IncomePortfolio()
        p.positions = [_pos(strategy="covered_call", underlying_at_entry=500, status="closed")]
        assert p.capital_deployed == 0

    def test_capital_deployed_unknown_strategy(self):
        p = IncomePortfolio()
        p.positions = [_pos(strategy="unknown")]
        assert p.capital_deployed == 0

    def test_capital_utilization_zero_capital(self):
        p = IncomePortfolio()
        assert p.capital_utilization == 0

    def test_capital_utilization(self):
        p = IncomePortfolio(total_capital=100000)
        p.positions = [_pos(strategy="covered_call", underlying_at_entry=500)]
        assert p.capital_utilization == 50.0


# ═══════════════════════════════════════════════════════════════════════════
# WheelEngine
# ═══════════════════════════════════════════════════════════════════════════

class TestWheelEngine:
    def test_init_default_allocation(self):
        e = WheelEngine(capital=100000)
        assert e.max_allocation == 0.15

    def test_screen_filters_capital(self):
        # max_capital_per = 100000*0.15 = 15000
        e = WheelEngine(capital=100000)
        candidates = [
            {"symbol": "OK", "price": 100, "iv_rank": 60, "bid_ask_spread": 0.05, "avg_option_volume": 1000},
            {"symbol": "TOO_BIG", "price": 200, "iv_rank": 60, "bid_ask_spread": 0.05, "avg_option_volume": 1000},
        ]
        out = e.screen_csp_candidates(candidates)
        assert any(c["symbol"] == "OK" for c in out)
        assert not any(c["symbol"] == "TOO_BIG" for c in out)

    def test_screen_filters_low_ivr(self):
        e = WheelEngine(capital=100000)
        candidates = [
            {"symbol": "LOW", "price": 100, "iv_rank": 30, "bid_ask_spread": 0.05, "avg_option_volume": 1000},
        ]
        assert e.screen_csp_candidates(candidates) == []

    def test_screen_filters_low_volume(self):
        e = WheelEngine(capital=100000)
        candidates = [
            {"symbol": "ILLIQ", "price": 100, "iv_rank": 60, "bid_ask_spread": 0.05, "avg_option_volume": 100},
        ]
        assert e.screen_csp_candidates(candidates) == []

    def test_screen_filters_wide_spread(self):
        e = WheelEngine(capital=100000)
        candidates = [
            {"symbol": "WIDE", "price": 100, "iv_rank": 60, "bid_ask_spread": 0.30, "avg_option_volume": 1000},
        ]
        assert e.screen_csp_candidates(candidates) == []

    def test_screen_includes_score_and_capital(self):
        e = WheelEngine(capital=100000)
        candidates = [
            {"symbol": "OK", "price": 100, "iv_rank": 60, "bid_ask_spread": 0.05, "avg_option_volume": 1000},
        ]
        out = e.screen_csp_candidates(candidates)
        assert "score" in out[0]
        assert out[0]["capital_needed"] == 10000
        assert out[0]["capital_pct"] == 10.0

    def test_screen_sorted_desc(self):
        e = WheelEngine(capital=1_000_000)
        candidates = [
            {"symbol": f"S{i}", "price": 100, "iv_rank": 50 + i*5,
             "bid_ask_spread": 0.05, "avg_option_volume": 1000}
            for i in range(3)
        ]
        out = e.screen_csp_candidates(candidates)
        scores = [c["score"] for c in out]
        assert scores == sorted(scores, reverse=True)

    def test_calculate_csp_yield(self):
        e = WheelEngine(capital=100000)
        out = e.calculate_csp_yield(strike=495, premium=3.50, dte=35)
        assert out["capital_required"] == 49500
        assert out["credit"] == 350
        assert out["breakeven"] == 491.5
        # yield_pct = 350/49500*100
        assert out["yield_pct"] == round(350/49500*100, 2)

    def test_should_roll_close_at_50pct_profit(self):
        e = WheelEngine(capital=100000)
        # pnl_pct=50: credit=2, current=1 → (2-1)/2*100=50
        p = _pos(strategy="wheel_csp", strikes=[100], credit_received=2.0,
                 current_value=1.0, dte_remaining=30)
        rolled, action, _ = e.should_roll(p, underlying_price=110)
        assert rolled is True
        assert action == ManagementAction.CLOSE

    def test_should_roll_at_21dte_tested(self):
        e = WheelEngine(capital=100000)
        p = _pos(strategy="wheel_csp", strikes=[100], credit_received=2.0,
                 current_value=1.5, dte_remaining=21)  # 25% profit
        rolled, action, _ = e.should_roll(p, underlying_price=95)  # tested
        assert rolled is True
        assert action == ManagementAction.ROLL_OUT_AND_DOWN

    def test_should_roll_at_21dte_untested(self):
        e = WheelEngine(capital=100000)
        p = _pos(strategy="wheel_csp", strikes=[100], credit_received=2.0,
                 current_value=1.5, dte_remaining=21)  # 25% profit
        rolled, action, _ = e.should_roll(p, underlying_price=105)
        assert rolled is True
        assert action == ManagementAction.ROLL_OUT

    def test_should_take_assignment_deep_itm_14dte(self):
        e = WheelEngine(capital=100000)
        # TAKE_ASSIGNMENT only reachable when pnl in [40, 50) at 14 DTE tested
        # (else 21-DTE branch fires first).
        p = _pos(strategy="wheel_csp", strikes=[100], credit_received=2.0,
                 current_value=1.1, dte_remaining=14)  # 45% profit
        rolled, action, _ = e.should_roll(p, underlying_price=90)
        assert rolled is True
        assert action == ManagementAction.TAKE_ASSIGNMENT

    def test_should_hold_within_params(self):
        e = WheelEngine(capital=100000)
        p = _pos(strategy="wheel_csp", strikes=[100], credit_received=2.0,
                 current_value=1.5, dte_remaining=30)  # 25% profit, 30 DTE
        rolled, action, _ = e.should_roll(p, underlying_price=110)
        assert rolled is False
        assert action == ManagementAction.HOLD


# ═══════════════════════════════════════════════════════════════════════════
# CreditSpreadEngine
# ═══════════════════════════════════════════════════════════════════════════

class TestCreditSpreadEngine:
    def _chain(self, **overrides):
        base = {"strike": 500, "delta": 0.20, "bid": 1.75, "iv": 0.30}
        base.update(overrides)
        return [base]

    def test_select_spread_bullish(self):
        out = CreditSpreadEngine.select_spread(self._chain(), outlook="bullish")
        assert out is not None
        assert out["strategy"] == "bull_put_spread"
        assert out["short_strike"] == 500
        assert out["long_strike"] == 495  # 500 - 5
        assert out["max_profit"] == 175  # 1.75*100

    def test_select_spread_bearish(self):
        out = CreditSpreadEngine.select_spread(self._chain(), outlook="bearish")
        assert out is not None
        assert out["strategy"] == "bear_call_spread"
        assert out["long_strike"] == 505  # 500 + 5

    def test_select_spread_no_match_delta(self):
        # delta_match = 1 - |0.05-0.20|/0.20 = 0.25 < 0.5
        out = CreditSpreadEngine.select_spread(self._chain(delta=0.05), outlook="bullish")
        assert out is None

    def test_select_spread_below_min_credit(self):
        # credit/spread = 0.5/5 = 0.10 < 0.30
        out = CreditSpreadEngine.select_spread(self._chain(bid=0.5), outlook="bullish")
        assert out is None

    def test_manage_spread_close_at_50pct(self):
        p = _pos(credit_received=2.0, current_value=1.0, dte_remaining=30,
                 strikes=[490, 500])
        action, _ = CreditSpreadEngine.manage_spread(p, underlying_price=510)
        assert action == ManagementAction.CLOSE

    def test_manage_spread_close_at_21dte_winning(self):
        p = _pos(credit_received=2.0, current_value=1.5, dte_remaining=21,
                 strikes=[490, 500])
        action, _ = CreditSpreadEngine.manage_spread(p, underlying_price=510)
        assert action == ManagementAction.CLOSE

    def test_manage_spread_roll_at_21dte_losing(self):
        p = _pos(credit_received=2.0, current_value=3.0, dte_remaining=21,
                 strikes=[490, 500])
        action, _ = CreditSpreadEngine.manage_spread(p, underlying_price=505)
        assert action == ManagementAction.ROLL_OUT

    def test_manage_spread_defend_when_close(self):
        # Code uses min(strikes) as short_strike. With strikes=[490,500],
        # short=490, underlying=494 -> distance = 4/494*100 ≈ 0.81% < 1.0.
        p = _pos(credit_received=2.0, current_value=1.8, dte_remaining=30,
                 strikes=[490, 500])
        action, _ = CreditSpreadEngine.manage_spread(p, underlying_price=494)
        assert action == ManagementAction.DEFEND

    def test_manage_spread_hold(self):
        p = _pos(credit_received=2.0, current_value=1.5, dte_remaining=30,
                 strikes=[490, 500])
        action, _ = CreditSpreadEngine.manage_spread(p, underlying_price=520)
        assert action == ManagementAction.HOLD


# ═══════════════════════════════════════════════════════════════════════════
# IronCondorEngine
# ═══════════════════════════════════════════════════════════════════════════

class TestIronCondorEngine:
    def test_construct(self):
        out = IronCondorEngine.construct(
            put_short=500, put_long=490, call_short=540, call_long=550,
            put_credit=1.50, call_credit=1.20, dte=35,
        )
        assert out["total_credit"] == 2.70
        assert out["total_credit_dollars"] == 270
        # max_width = max(10, 10) = 10; max_loss = (10-2.7)*100 = 730
        assert out["max_loss"] == 730
        assert out["risk_reward"] == round(730/270, 2)
        assert out["dte"] == 35
        assert out["profit_target"] == 135  # 2.7 * 0.5 * 100

    def test_construct_credit_pct(self):
        out = IronCondorEngine.construct(
            put_short=500, put_long=490, call_short=540, call_long=550,
            put_credit=1.50, call_credit=1.50, dte=35,
        )
        # 3.0/10 = 30%
        assert out["credit_pct_of_width"] == 30.0

    def test_adjustment_insufficient_strikes(self):
        p = _pos(strikes=[100, 110])
        out = IronCondorEngine.adjustment_decision(p, underlying_price=105)
        assert out["action"] == "HOLD"

    def test_adjustment_close_at_50pct(self):
        p = _pos(strikes=[490, 500, 540, 550], credit_received=2.0,
                 current_value=1.0, dte_remaining=30)
        out = IronCondorEngine.adjustment_decision(p, underlying_price=520)
        assert out["action"] == "CLOSE_ALL"
        assert out["priority"] == "HIGH"

    def test_adjustment_close_at_21dte(self):
        p = _pos(strikes=[490, 500, 540, 550], credit_received=2.0,
                 current_value=1.5, dte_remaining=21)
        out = IronCondorEngine.adjustment_decision(p, underlying_price=520)
        assert out["action"] == "CLOSE_ALL"
        assert out["priority"] == "MEDIUM"

    def test_adjustment_put_threatened(self):
        # underlying near put_short=500
        p = _pos(strikes=[490, 500, 540, 550], credit_received=2.0,
                 current_value=1.5, dte_remaining=30)
        out = IronCondorEngine.adjustment_decision(p, underlying_price=505)
        assert out["action"] == "ROLL_CALL_SIDE_DOWN"
        assert "new_call_short" in out

    def test_adjustment_call_threatened(self):
        # underlying near call_short=540
        p = _pos(strikes=[490, 500, 540, 550], credit_received=2.0,
                 current_value=1.5, dte_remaining=30)
        out = IronCondorEngine.adjustment_decision(p, underlying_price=535)
        assert out["action"] == "ROLL_PUT_SIDE_UP"
        assert "new_put_short" in out

    def test_adjustment_hold(self):
        p = _pos(strikes=[490, 500, 540, 550], credit_received=2.0,
                 current_value=1.5, dte_remaining=30)
        out = IronCondorEngine.adjustment_decision(p, underlying_price=520)
        assert out["action"] == "HOLD"
        assert out["priority"] == "LOW"


# ═══════════════════════════════════════════════════════════════════════════
# CoveredCallScreener
# ═══════════════════════════════════════════════════════════════════════════

class TestCoveredCallScreener:
    def _stock(self, **k):
        base = dict(price=100, call_premium=2.0, call_delta=0.30, dte=30,
                    iv=0.30, div_yield=0, earnings_in_cycle=False)
        base.update(k)
        return base

    def test_screen_income_mode(self):
        out = CoveredCallScreener.screen([self._stock()], mode="income")
        assert len(out) == 1
        assert out[0]["mode"] == "income"
        # monthly = 2/100 * 30/30 * 100 = 2.0
        assert out[0]["monthly_yield_pct"] == 2.0
        assert out[0]["annualized_yield_pct"] == 24.0

    def test_screen_skips_earnings_in_cycle(self):
        assert CoveredCallScreener.screen([self._stock(earnings_in_cycle=True)]) == []

    def test_screen_skips_zero_price(self):
        assert CoveredCallScreener.screen([self._stock(price=0)]) == []

    def test_screen_skips_zero_premium(self):
        assert CoveredCallScreener.screen([self._stock(call_premium=0)]) == []

    def test_screen_filters_low_yield_income(self):
        # min_yield=0.8 in income mode; 0.5/100*100=0.5
        assert CoveredCallScreener.screen([self._stock(call_premium=0.5)], mode="income") == []

    def test_screen_filters_low_yield_growth(self):
        # min_yield=0.4 in growth mode; 0.3/100*100=0.3
        assert CoveredCallScreener.screen([self._stock(call_premium=0.3, call_delta=0.18)],
                                          mode="growth") == []

    def test_screen_growth_mode(self):
        out = CoveredCallScreener.screen([self._stock(call_delta=0.18, call_premium=0.5)],
                                         mode="growth")
        assert len(out) == 1
        assert out[0]["mode"] == "growth"

    def test_screen_sorted_desc(self):
        stocks = [
            self._stock(symbol="A", call_premium=1.5),
            self._stock(symbol="B", call_premium=3.0),
        ]
        out = CoveredCallScreener.screen(stocks, mode="income")
        assert out[0]["call_premium"] >= out[1]["call_premium"]

    def test_screen_includes_total_annual_with_dividend(self):
        out = CoveredCallScreener.screen([self._stock(div_yield=2.5)], mode="income")
        assert out[0]["total_annual_return_pct"] == 26.5  # 24.0 + 2.5


# ═══════════════════════════════════════════════════════════════════════════
# IncomeYieldTracker
# ═══════════════════════════════════════════════════════════════════════════

class TestIncomeYieldTracker:
    def test_record_trade(self):
        t = IncomeYieldTracker(100000)
        t.record_trade("SPY", "iron_condor", 2.70, 135, "2025-01-15", 35)
        assert len(t.trades) == 1
        assert t.trades[0]["return_pct"] == 0.135  # 135/100000*100
        assert t.monthly_income["2025-01"] == 135

    def test_record_trade_aggregates_month(self):
        t = IncomeYieldTracker(100000)
        t.record_trade("A", "x", 1.0, 100, "2025-01-15", 30)
        t.record_trade("B", "x", 1.0, 200, "2025-01-22", 30)
        assert t.monthly_income["2025-01"] == 300

    def test_summary_empty(self):
        assert IncomeYieldTracker(100000).summary() == {"total_trades": 0}

    def test_summary_basic(self):
        t = IncomeYieldTracker(100000)
        t.record_trade("A", "x", 1.0, 100, "2025-01-15", 30)
        t.record_trade("B", "x", 1.0, -50, "2025-01-22", 30)
        s = t.summary()
        assert s["total_trades"] == 2
        assert s["total_pnl"] == 50
        assert s["win_rate"] == 50.0
        assert s["avg_win"] == 100
        assert s["avg_loss"] == -50
        assert s["profit_factor"] == 2.0

    def test_summary_no_losers_returns_inf(self):
        t = IncomeYieldTracker(100000)
        t.record_trade("A", "x", 1.0, 100, "2025-01-15", 30)
        s = t.summary()
        assert s["profit_factor"] == float("inf")

    def test_summary_monthly_average(self):
        t = IncomeYieldTracker(100000)
        t.record_trade("A", "x", 1.0, 100, "2025-01-15", 30)
        t.record_trade("B", "x", 1.0, 200, "2025-02-15", 30)
        s = t.summary()
        assert s["avg_monthly_income"] == 150  # (100+200)/2
        assert s["months_tracked"] == 2

    def test_summary_total_return_pct(self):
        t = IncomeYieldTracker(100000)
        t.record_trade("A", "x", 1.0, 1000, "2025-01-15", 30)
        s = t.summary()
        assert s["total_return_pct"] == 1.0  # 1000/100000*100
