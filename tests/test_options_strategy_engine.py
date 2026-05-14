from __future__ import annotations

import math

import pytest

from strategies.options_strategy_engine import (
    Direction,
    OptionLeg,
    OptionType,
    OptionsRiskCalculator,
    OptionsStrategy,
    RiskMetrics,
    RiskProfile,
    StrategyBuilder,
    StrategyGreeks,
    StrategyOutlook,
    StrategyScanner,
    _norm_cdf,
    _norm_pdf,
    black_scholes_price,
    compute_greeks,
)


# ─────────────────────────────────────────────────────────────────────────
# TestEnums
# ─────────────────────────────────────────────────────────────────────────

class TestEnums:
    def test_option_type_values(self):
        assert OptionType.CALL.value == "call"
        assert OptionType.PUT.value == "put"

    def test_direction_values(self):
        assert Direction.LONG.value == "long"
        assert Direction.SHORT.value == "short"

    def test_outlook_values(self):
        assert {o.value for o in StrategyOutlook} == {"bullish", "bearish", "neutral", "volatile"}

    def test_risk_profile_values(self):
        assert {r.value for r in RiskProfile} == {"defined", "undefined", "semi_defined"}


# ─────────────────────────────────────────────────────────────────────────
# TestOptionLeg
# ─────────────────────────────────────────────────────────────────────────

class TestOptionLeg:
    def test_is_long_true(self):
        leg = OptionLeg(OptionType.CALL, Direction.LONG, 100.0, 30, premium=2.0)
        assert leg.is_long

    def test_is_long_false_for_short(self):
        leg = OptionLeg(OptionType.PUT, Direction.SHORT, 100.0, 30, premium=2.0)
        assert not leg.is_long

    def test_net_premium_long_debit_positive(self):
        leg = OptionLeg(OptionType.CALL, Direction.LONG, 100.0, 30, quantity=2, premium=1.5)
        # long → debit = +3.0
        assert leg.net_premium == pytest.approx(3.0)

    def test_net_premium_short_credit_negative(self):
        leg = OptionLeg(OptionType.PUT, Direction.SHORT, 100.0, 30, quantity=3, premium=1.0)
        # short → credit = -3.0
        assert leg.net_premium == pytest.approx(-3.0)


# ─────────────────────────────────────────────────────────────────────────
# TestStrategyGreeks
# ─────────────────────────────────────────────────────────────────────────

class TestStrategyGreeks:
    def test_defaults_zero(self):
        g = StrategyGreeks()
        assert g.delta == 0.0 and g.gamma == 0.0 and g.theta == 0.0 and g.vega == 0.0

    def test_repr_contains_symbols(self):
        g = StrategyGreeks(delta=0.5, gamma=0.01, theta=-0.05, vega=0.1)
        s = repr(g)
        assert "Δ" in s and "Γ" in s and "Θ" in s and "V" in s


# ─────────────────────────────────────────────────────────────────────────
# TestRiskMetrics
# ─────────────────────────────────────────────────────────────────────────

class TestRiskMetrics:
    def test_expectancy_positive(self):
        m = RiskMetrics(
            max_profit=100, max_loss=-50, breakeven_prices=[105.0],
            probability_of_profit=0.7, risk_reward_ratio=0.5,
            capital_required=50, return_on_risk=200.0,
        )
        # 0.7*100 - 0.3*50 = 70 - 15 = 55
        assert m.expectancy == pytest.approx(55.0)

    def test_expectancy_negative(self):
        m = RiskMetrics(
            max_profit=10, max_loss=-100, breakeven_prices=[],
            probability_of_profit=0.3, risk_reward_ratio=10.0,
            capital_required=100, return_on_risk=10.0,
        )
        # 0.3*10 - 0.7*100 = 3 - 70 = -67
        assert m.expectancy == pytest.approx(-67.0)


# ─────────────────────────────────────────────────────────────────────────
# TestNormalDistribution
# ─────────────────────────────────────────────────────────────────────────

class TestNormalDistribution:
    def test_norm_cdf_zero(self):
        assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-3)

    def test_norm_cdf_symmetric(self):
        assert _norm_cdf(1.0) + _norm_cdf(-1.0) == pytest.approx(1.0, abs=1e-3)

    def test_norm_cdf_extremes(self):
        assert _norm_cdf(5.0) == pytest.approx(1.0, abs=1e-4)
        assert _norm_cdf(-5.0) == pytest.approx(0.0, abs=1e-4)

    def test_norm_pdf_zero(self):
        assert _norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2 * math.pi))

    def test_norm_pdf_symmetric(self):
        assert _norm_pdf(1.5) == pytest.approx(_norm_pdf(-1.5))


# ─────────────────────────────────────────────────────────────────────────
# TestBlackScholes
# ─────────────────────────────────────────────────────────────────────────

class TestBlackScholes:
    def test_call_atm_positive(self):
        p = black_scholes_price(100, 100, 0.5, 0.05, 0.20, OptionType.CALL)
        assert p > 0

    def test_put_atm_positive(self):
        p = black_scholes_price(100, 100, 0.5, 0.05, 0.20, OptionType.PUT)
        assert p > 0

    def test_call_intrinsic_when_t_zero(self):
        p = black_scholes_price(110, 100, 0.0, 0.05, 0.20, OptionType.CALL)
        assert p == pytest.approx(10.0)

    def test_put_intrinsic_when_t_zero(self):
        p = black_scholes_price(90, 100, 0.0, 0.05, 0.20, OptionType.PUT)
        assert p == pytest.approx(10.0)

    def test_call_otm_intrinsic_zero(self):
        p = black_scholes_price(90, 100, 0.0, 0.05, 0.20, OptionType.CALL)
        assert p == 0.0

    def test_zero_sigma_returns_intrinsic(self):
        p = black_scholes_price(110, 100, 0.5, 0.05, 0.0, OptionType.CALL)
        assert p == pytest.approx(10.0)

    def test_call_more_valuable_deep_itm(self):
        atm = black_scholes_price(100, 100, 0.5, 0.05, 0.20, OptionType.CALL)
        itm = black_scholes_price(120, 100, 0.5, 0.05, 0.20, OptionType.CALL)
        assert itm > atm


# ─────────────────────────────────────────────────────────────────────────
# TestComputeGreeks
# ─────────────────────────────────────────────────────────────────────────

class TestComputeGreeks:
    def test_call_delta_in_range(self):
        g = compute_greeks(100, 100, 0.5, 0.05, 0.20, OptionType.CALL)
        assert 0.0 < g["delta"] < 1.0

    def test_put_delta_negative(self):
        g = compute_greeks(100, 100, 0.5, 0.05, 0.20, OptionType.PUT)
        assert -1.0 < g["delta"] < 0.0

    def test_gamma_positive(self):
        g = compute_greeks(100, 100, 0.5, 0.05, 0.20, OptionType.CALL)
        assert g["gamma"] > 0

    def test_vega_positive(self):
        g = compute_greeks(100, 100, 0.5, 0.05, 0.20, OptionType.CALL)
        assert g["vega"] > 0

    def test_theta_negative_for_long(self):
        g = compute_greeks(100, 100, 0.5, 0.05, 0.20, OptionType.CALL)
        assert g["theta"] < 0

    def test_zero_t_returns_zeros(self):
        g = compute_greeks(100, 100, 0.0, 0.05, 0.20, OptionType.CALL)
        assert g == {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

    def test_zero_sigma_returns_zeros(self):
        g = compute_greeks(100, 100, 0.5, 0.05, 0.0, OptionType.CALL)
        assert g["delta"] == 0


# ─────────────────────────────────────────────────────────────────────────
# TestOptionsStrategyDataclass
# ─────────────────────────────────────────────────────────────────────────

class TestOptionsStrategyDataclass:
    def _make_strategy(self):
        legs = [
            OptionLeg(OptionType.CALL, Direction.LONG, 100, 30, premium=3.0,
                      delta=0.5, gamma=0.02, theta=-0.05, vega=0.1),
            OptionLeg(OptionType.CALL, Direction.SHORT, 110, 30, premium=1.0,
                      delta=0.3, gamma=0.015, theta=-0.03, vega=0.08),
        ]
        return OptionsStrategy(
            name="TestStrategy", outlook=StrategyOutlook.BULLISH,
            risk_profile=RiskProfile.DEFINED, legs=legs,
            underlying_price=105.0,
        )

    def test_net_credit_for_debit_strategy_negative(self):
        s = self._make_strategy()
        # long 3.0 - short 1.0 = +2.0 debit; net_credit = -2.0
        assert s.net_credit == pytest.approx(-2.0)

    def test_greeks_aggregate_sign_long_minus_short(self):
        s = self._make_strategy()
        g = s.greeks
        # delta: +0.5 - 0.3 = 0.2
        assert g.delta == pytest.approx(0.2)
        assert g.gamma == pytest.approx(0.005)
        assert g.theta == pytest.approx(-0.02)
        assert g.vega == pytest.approx(0.02)

    def test_leg_count(self):
        s = self._make_strategy()
        assert s.leg_count == 2

    def test_to_dict_keys(self):
        s = self._make_strategy()
        d = s.to_dict()
        assert d["name"] == "TestStrategy"
        assert d["outlook"] == "bullish"
        assert d["risk_profile"] == "defined"
        assert d["leg_count"] == 2
        assert "net_credit" in d
        assert set(d["greeks"].keys()) == {"delta", "gamma", "theta", "vega"}
        assert d["underlying_price"] == 105.0


# ─────────────────────────────────────────────────────────────────────────
# TestStrategyBuilder
# ─────────────────────────────────────────────────────────────────────────

class TestStrategyBuilder:
    def setup_method(self):
        self.builder = StrategyBuilder(underlying_price=100.0)

    def test_init_defaults(self):
        b = StrategyBuilder(50.0)
        assert b.S == 50.0
        assert b.r == 0.05

    def test_long_call(self):
        s = self.builder.long_call(100, 30, 0.20)
        assert s.name == "Long Call"
        assert s.outlook == StrategyOutlook.BULLISH
        assert s.risk_profile == RiskProfile.DEFINED
        assert s.leg_count == 1
        assert s.legs[0].is_long
        assert s.legs[0].option_type == OptionType.CALL

    def test_long_put(self):
        s = self.builder.long_put(100, 30, 0.20)
        assert s.name == "Long Put"
        assert s.outlook == StrategyOutlook.BEARISH
        assert s.legs[0].option_type == OptionType.PUT
        assert s.legs[0].is_long

    def test_bull_call_spread_two_legs(self):
        s = self.builder.bull_call_spread(95, 105, 30, 0.20)
        assert s.leg_count == 2
        assert s.legs[0].direction == Direction.LONG
        assert s.legs[1].direction == Direction.SHORT
        assert s.outlook == StrategyOutlook.BULLISH

    def test_bull_put_spread_credit(self):
        s = self.builder.bull_put_spread(105, 95, 30, 0.20)
        # short higher strike put + long lower strike put → credit
        assert s.net_credit > 0

    def test_bear_call_spread(self):
        s = self.builder.bear_call_spread(95, 105, 30, 0.20)
        assert s.outlook == StrategyOutlook.BEARISH
        assert s.legs[0].direction == Direction.SHORT
        assert s.net_credit > 0

    def test_bear_put_spread_debit(self):
        s = self.builder.bear_put_spread(105, 95, 30, 0.20)
        assert s.outlook == StrategyOutlook.BEARISH
        # long high strike put + short low strike put → debit
        assert s.net_credit < 0

    def test_iron_condor_four_legs(self):
        s = self.builder.iron_condor(85, 95, 105, 115, 30, 0.20)
        assert s.leg_count == 4
        assert s.outlook == StrategyOutlook.NEUTRAL
        assert s.net_credit > 0

    def test_iron_butterfly(self):
        s = self.builder.iron_butterfly(90, 100, 110, 30, 0.20)
        assert s.leg_count == 4
        assert s.outlook == StrategyOutlook.NEUTRAL

    def test_short_straddle_undefined_risk(self):
        s = self.builder.short_straddle(100, 30, 0.20)
        assert s.leg_count == 2
        assert s.risk_profile == RiskProfile.UNDEFINED
        assert s.net_credit > 0

    def test_long_straddle_volatile(self):
        s = self.builder.long_straddle(100, 30, 0.20)
        assert s.outlook == StrategyOutlook.VOLATILE
        assert s.risk_profile == RiskProfile.DEFINED

    def test_short_strangle(self):
        s = self.builder.short_strangle(95, 105, 30, 0.20)
        assert s.leg_count == 2
        assert s.risk_profile == RiskProfile.UNDEFINED

    def test_butterfly_calls_three_legs(self):
        s = self.builder.butterfly_spread(95, 100, 105, 30, 0.20, use_calls=True)
        assert s.leg_count == 3
        assert s.legs[1].quantity == 2
        assert s.legs[1].option_type == OptionType.CALL
        assert "Calls" in s.name

    def test_butterfly_puts(self):
        s = self.builder.butterfly_spread(95, 100, 105, 30, 0.20, use_calls=False)
        assert s.legs[1].option_type == OptionType.PUT
        assert "Puts" in s.name

    def test_calendar_spread(self):
        s = self.builder.calendar_spread(100, 30, 60, 0.25, 0.20, use_calls=True)
        assert s.leg_count == 2
        assert s.legs[0].expiry_days == 30
        assert s.legs[1].expiry_days == 60

    def test_covered_call_has_shares(self):
        s = self.builder.covered_call(105, 30, 0.20)
        assert s.underlying_shares == 100
        assert s.risk_profile == RiskProfile.SEMI_DEFINED

    def test_wheel_strategy(self):
        s = self.builder.wheel_strategy(95, 30, 0.20)
        assert s.legs[0].option_type == OptionType.PUT
        assert s.legs[0].direction == Direction.SHORT

    def test_pmcc_two_legs(self):
        s = self.builder.poor_mans_covered_call(80, 365, 105, 30, 0.20)
        assert s.leg_count == 2
        assert s.legs[0].expiry_days == 365
        assert s.legs[1].expiry_days == 30

    def test_collar(self):
        s = self.builder.collar(95, 105, 30, 0.20)
        assert s.underlying_shares == 100
        assert s.legs[0].option_type == OptionType.PUT
        assert s.legs[0].is_long
        assert s.legs[1].option_type == OptionType.CALL
        assert not s.legs[1].is_long

    def test_jade_lizard_three_legs(self):
        s = self.builder.jade_lizard(90, 115, 105, 30, 0.20)
        assert s.leg_count == 3

    def test_broken_wing_butterfly_bullish_uses_puts(self):
        s = self.builder.broken_wing_butterfly(90, 100, 105, 30, 0.20, bullish=True)
        assert s.outlook == StrategyOutlook.BULLISH
        assert all(leg.option_type == OptionType.PUT for leg in s.legs)

    def test_broken_wing_butterfly_bearish_uses_calls(self):
        s = self.builder.broken_wing_butterfly(95, 100, 110, 30, 0.20, bullish=False)
        assert s.outlook == StrategyOutlook.BEARISH
        assert all(leg.option_type == OptionType.CALL for leg in s.legs)

    def test_christmas_tree_three_legs(self):
        s = self.builder.christmas_tree_spread(100, 105, 110, 30, 0.20, use_calls=True)
        assert s.leg_count == 3

    def test_ratio_spread_quantity(self):
        s = self.builder.ratio_spread(100, 110, 30, 0.20, ratio=3, use_calls=True)
        assert s.legs[1].quantity == 3
        assert "1x3" in s.name


# ─────────────────────────────────────────────────────────────────────────
# TestOptionsRiskCalculator
# ─────────────────────────────────────────────────────────────────────────

class TestOptionsRiskCalculator:
    def test_payoff_long_call_above_strike(self):
        leg = OptionLeg(OptionType.CALL, Direction.LONG, 100, 30, premium=2.0)
        s = OptionsStrategy(name="LC", outlook=StrategyOutlook.BULLISH,
                            risk_profile=RiskProfile.DEFINED, legs=[leg],
                            underlying_price=100.0)
        # at 110: intrinsic 10 - premium 2 = 8 per share * 100
        assert OptionsRiskCalculator.calculate_payoff(s, 110.0) == pytest.approx(800.0)

    def test_payoff_long_call_below_strike(self):
        leg = OptionLeg(OptionType.CALL, Direction.LONG, 100, 30, premium=2.0)
        s = OptionsStrategy(name="LC", outlook=StrategyOutlook.BULLISH,
                            risk_profile=RiskProfile.DEFINED, legs=[leg],
                            underlying_price=100.0)
        # at 95: intrinsic 0 - premium 2 = -200
        assert OptionsRiskCalculator.calculate_payoff(s, 95.0) == pytest.approx(-200.0)

    def test_payoff_short_put_above_strike(self):
        leg = OptionLeg(OptionType.PUT, Direction.SHORT, 100, 30, premium=2.0)
        s = OptionsStrategy(name="SP", outlook=StrategyOutlook.BULLISH,
                            risk_profile=RiskProfile.SEMI_DEFINED, legs=[leg],
                            underlying_price=100.0)
        # at 110: intrinsic 0; short keeps full premium 200
        assert OptionsRiskCalculator.calculate_payoff(s, 110.0) == pytest.approx(200.0)

    def test_payoff_includes_underlying_shares(self):
        leg = OptionLeg(OptionType.CALL, Direction.SHORT, 110, 30, premium=1.0)
        s = OptionsStrategy(name="CC", outlook=StrategyOutlook.BULLISH,
                            risk_profile=RiskProfile.SEMI_DEFINED, legs=[leg],
                            underlying_price=100.0, underlying_shares=100)
        # stock: 100 * (105-100) = 500; call below strike: +100
        assert OptionsRiskCalculator.calculate_payoff(s, 105.0) == pytest.approx(600.0)

    def test_compute_risk_metrics_long_call(self):
        builder = StrategyBuilder(100.0)
        s = builder.long_call(100, 30, 0.20)
        m = OptionsRiskCalculator.compute_risk_metrics(s)
        assert isinstance(m, RiskMetrics)
        assert m.max_loss < 0
        assert m.max_profit > 0
        assert 0 <= m.probability_of_profit <= 1

    def test_compute_risk_metrics_iron_condor(self):
        builder = StrategyBuilder(100.0)
        s = builder.iron_condor(85, 95, 105, 115, 30, 0.20)
        m = OptionsRiskCalculator.compute_risk_metrics(s)
        # iron condor: defined risk
        assert m.max_loss < 0
        assert m.max_profit > 0
        assert m.capital_required > 0
        assert m.return_on_risk > 0

    def test_compute_risk_metrics_finds_breakevens(self):
        builder = StrategyBuilder(100.0)
        s = builder.bull_call_spread(95, 105, 30, 0.20)
        m = OptionsRiskCalculator.compute_risk_metrics(s)
        assert len(m.breakeven_prices) >= 1

    def test_compute_risk_metrics_with_custom_range(self):
        builder = StrategyBuilder(100.0)
        s = builder.long_call(100, 30, 0.20)
        m = OptionsRiskCalculator.compute_risk_metrics(s, price_range=(80.0, 120.0), steps=200)
        assert isinstance(m, RiskMetrics)


# ─────────────────────────────────────────────────────────────────────────
# TestStrategyScanner
# ─────────────────────────────────────────────────────────────────────────

class TestStrategyScanner:
    def test_recommend_bullish_high_iv(self):
        recs = StrategyScanner.recommend("bullish", 75)
        assert "Bull Put Spread" in recs
        assert "Wheel" in recs

    def test_recommend_bullish_low_iv(self):
        recs = StrategyScanner.recommend("bullish", 25)
        assert "Long Call" in recs
        assert "PMCC" in recs

    def test_recommend_bearish_high_iv(self):
        recs = StrategyScanner.recommend("bearish", 80)
        assert "Bear Call Spread" in recs

    def test_recommend_bearish_low_iv(self):
        recs = StrategyScanner.recommend("bearish", 20)
        assert "Long Put" in recs

    def test_recommend_neutral_high_iv(self):
        recs = StrategyScanner.recommend("neutral", 70)
        assert "Iron Condor" in recs
        assert "Short Strangle" in recs

    def test_recommend_neutral_low_iv(self):
        recs = StrategyScanner.recommend("neutral", 25)
        assert "Calendar Spread" in recs

    def test_recommend_volatile_high_iv(self):
        recs = StrategyScanner.recommend("volatile", 80)
        assert "Long Straddle" in recs

    def test_recommend_unknown_outlook_empty(self):
        recs = StrategyScanner.recommend("crazy", 50)
        assert recs == []

    def test_iv_boundary_50_means_low(self):
        # iv_rank > 50 → high; 50 itself → low
        recs = StrategyScanner.recommend("bullish", 50)
        assert "Long Call" in recs  # low_iv list

    def test_filter_by_risk_defined_only(self):
        all_strats = ["Long Call", "Short Strangle", "Iron Condor", "Covered Call"]
        filtered = StrategyScanner.filter_by_risk(all_strats, RiskProfile.DEFINED)
        assert "Long Call" in filtered
        assert "Iron Condor" in filtered
        assert "Short Strangle" not in filtered
        assert "Covered Call" not in filtered

    def test_filter_by_risk_semi_defined_includes_both(self):
        all_strats = ["Long Call", "Short Strangle", "Covered Call", "Wheel"]
        filtered = StrategyScanner.filter_by_risk(all_strats, RiskProfile.SEMI_DEFINED)
        assert "Long Call" in filtered
        assert "Covered Call" in filtered
        assert "Wheel" in filtered
        assert "Short Strangle" not in filtered

    def test_filter_by_risk_undefined_returns_all(self):
        all_strats = ["Long Call", "Short Strangle", "Covered Call"]
        filtered = StrategyScanner.filter_by_risk(all_strats, RiskProfile.UNDEFINED)
        assert filtered == all_strats
