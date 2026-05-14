from __future__ import annotations

from datetime import datetime

import pytest

from strategies.regime_engine import FormulaTag, Regime, RegimeState, SignalRiskClass
from strategies.stock_forecaster import (
    INDUSTRY_PLAYBOOK,
    Direction,
    ExpressionType,
    FailureMode,
    Horizon,
    Industry,
    IndustryForecast,
    IndustrySpec,
    StockForecaster,
    TradeOpportunity,
    print_industry_regime_matrix,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _state(
    primary=Regime.CREDIT_STRESS,
    secondary=None,
    confidence=1.0,
    armed_formulas=None,
    vol_readiness=50.0,
    bear=3,
    bull=0,
):
    return RegimeState(
        timestamp=datetime(2026, 4, 23),
        primary_regime=primary,
        secondary_regime=secondary,
        regime_confidence=confidence,
        formula_results=[],
        armed_formulas=armed_formulas or [],
        vol_shock_readiness=vol_readiness,
        bear_signals=bear,
        bull_signals=bull,
        summary="test",
    )


def _opp(**k):
    base = dict(
        rank=1,
        industry=Industry.CREDIT,
        direction=Direction.BEARISH,
        tickers=["HYG"],
        primary_ticker="HYG",
        horizon=Horizon.SHORT,
        expression=ExpressionType.PUT_SPREAD,
        thesis="t",
        catalyst="c",
        failure_modes=[FailureMode.POLICY_BACKSTOP],
        roi_score=80.0,
        risk_score=30.0,
        speed_score=85.0,
        composite_score=80.0,
        structure_hint="hint",
        expiry_range_days=(14, 42),
        otm_pct=0.03,
        risk_class=SignalRiskClass.INSTITUTIONAL,
        formula_sources=[],
    )
    base.update(k)
    return TradeOpportunity(**base)


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class TestEnums:
    def test_industry_values(self):
        assert Industry.CREDIT.value == "credit"
        assert Industry.ENERGY.value == "energy"

    def test_direction_values(self):
        assert {d.value for d in Direction} == {"bearish", "bullish", "neutral"}

    def test_expression_values(self):
        assert ExpressionType.AVOID.value == "avoid"
        assert ExpressionType.PUT_SPREAD.value == "put_spread"

    def test_horizon_values(self):
        assert {h.value for h in Horizon} == {"intraday", "short", "medium"}

    def test_failure_mode_values(self):
        assert FailureMode.POLICY_BACKSTOP.value == "policy_backstop"


# ═══════════════════════════════════════════════════════════════════════════
# IndustryForecast
# ═══════════════════════════════════════════════════════════════════════════

class TestIndustryForecast:
    def test_best_when_empty(self):
        fc = IndustryForecast(
            regime_state=_state(), horizon=Horizon.SHORT,
            opportunities=[], top_3=[], industry_map={},
        )
        assert fc.best is None

    def test_best_returns_first(self):
        a, b = _opp(primary_ticker="HYG"), _opp(primary_ticker="KRE")
        fc = IndustryForecast(
            regime_state=_state(), horizon=Horizon.SHORT,
            opportunities=[a, b], top_3=[a, b], industry_map={},
        )
        assert fc.best is a

    def test_print_plan_renders(self):
        fc = IndustryForecast(
            regime_state=_state(), horizon=Horizon.SHORT,
            opportunities=[_opp()], top_3=[_opp()], industry_map={},
        )
        out = fc.print_plan()
        assert "STOCK FORECASTER" in out
        assert "SHORT" in out
        assert "CREDIT STRESS" in out
        assert "HYG" in out
        assert "Thesis:" in out
        assert "Structure:" in out
        assert "Failure:" in out

    def test_print_plan_max_rows(self):
        opps = [_opp(rank=i, primary_ticker=f"T{i}") for i in range(15)]
        fc = IndustryForecast(
            regime_state=_state(), horizon=Horizon.SHORT,
            opportunities=opps, top_3=opps[:3], industry_map={},
        )
        out = fc.print_plan(max_rows=3)
        assert "T0" in out and "T1" in out and "T2" in out
        assert "T5" not in out


# ═══════════════════════════════════════════════════════════════════════════
# INDUSTRY_PLAYBOOK
# ═══════════════════════════════════════════════════════════════════════════

class TestIndustryPlaybook:
    def test_all_industries_unique(self):
        inds = [s.industry for s in INDUSTRY_PLAYBOOK]
        assert len(set(inds)) == len(inds)

    def test_energy_marked_avoid(self):
        spec = next(s for s in INDUSTRY_PLAYBOOK if s.industry == Industry.ENERGY)
        assert spec.warn_do_not_short is True
        assert spec.expression == ExpressionType.AVOID

    def test_credit_first_mover_high_speed(self):
        spec = next(s for s in INDUSTRY_PLAYBOOK if s.industry == Industry.CREDIT)
        assert spec.base_speed_score >= 90
        assert spec.primary_ticker == "HYG"

    def test_each_spec_primary_in_tickers(self):
        for spec in INDUSTRY_PLAYBOOK:
            assert spec.primary_ticker in spec.tickers

    def test_each_spec_dte_ranges_ordered(self):
        for spec in INDUSTRY_PLAYBOOK:
            assert spec.short_expiry[0] <= spec.short_expiry[1]
            assert spec.medium_expiry[0] <= spec.medium_expiry[1]


# ═══════════════════════════════════════════════════════════════════════════
# StockForecaster.forecast
# ═══════════════════════════════════════════════════════════════════════════

class TestForecast:
    def test_credit_stress_includes_credit_and_banks(self):
        f = StockForecaster()
        fc = f.forecast(_state(primary=Regime.CREDIT_STRESS), horizon=Horizon.SHORT)
        inds = {o.industry for o in fc.opportunities}
        assert Industry.CREDIT in inds
        assert Industry.BANKS in inds
        # Energy must be excluded (AVOID)
        assert Industry.ENERGY not in inds

    def test_excludes_energy_always(self):
        f = StockForecaster()
        for regime in Regime:
            fc = f.forecast(_state(primary=regime, bear=3), horizon=Horizon.SHORT)
            assert Industry.ENERGY not in {o.industry for o in fc.opportunities}

    def test_uncertain_regime_with_low_bear_signals_filters_all(self):
        f = StockForecaster()
        # primary=UNCERTAIN → empty active_vectors; bear<2 → relevance=0 skips
        fc = f.forecast(_state(primary=Regime.UNCERTAIN, bear=0), horizon=Horizon.SHORT)
        assert fc.opportunities == []
        assert fc.best is None
        assert fc.top_3 == []

    def test_uncertain_with_high_bear_signals_includes_industries(self):
        f = StockForecaster()
        fc = f.forecast(_state(primary=Regime.UNCERTAIN, bear=3), horizon=Horizon.SHORT)
        assert len(fc.opportunities) > 0
        for o in fc.opportunities:
            assert o.risk_class == SignalRiskClass.FRINGE

    def test_credit_stress_credit_industry_is_institutional(self):
        f = StockForecaster()
        fc = f.forecast(_state(primary=Regime.CREDIT_STRESS), horizon=Horizon.SHORT)
        credit = next(o for o in fc.opportunities if o.industry == Industry.CREDIT)
        assert credit.risk_class == SignalRiskClass.INSTITUTIONAL

    def test_top_n_limit(self):
        f = StockForecaster()
        fc = f.forecast(_state(primary=Regime.CREDIT_STRESS, bear=3), top_n=2)
        assert len(fc.opportunities) <= 2

    def test_ranks_assigned_descending(self):
        f = StockForecaster()
        fc = f.forecast(_state(primary=Regime.CREDIT_STRESS, bear=3))
        for i, o in enumerate(fc.opportunities, 1):
            assert o.rank == i
        scores = [o.composite_score for o in fc.opportunities]
        assert scores == sorted(scores, reverse=True)

    def test_top_3_is_first_three(self):
        f = StockForecaster()
        fc = f.forecast(_state(primary=Regime.CREDIT_STRESS, bear=3))
        assert fc.top_3 == fc.opportunities[:3]

    def test_industry_map_grouping(self):
        f = StockForecaster()
        fc = f.forecast(_state(primary=Regime.CREDIT_STRESS, bear=3))
        for ind, opps in fc.industry_map.items():
            for o in opps:
                assert o.industry == ind

    def test_horizon_short_uses_short_expiry(self):
        f = StockForecaster()
        fc = f.forecast(_state(primary=Regime.CREDIT_STRESS), horizon=Horizon.SHORT)
        credit = next(o for o in fc.opportunities if o.industry == Industry.CREDIT)
        spec = next(s for s in INDUSTRY_PLAYBOOK if s.industry == Industry.CREDIT)
        assert credit.expiry_range_days == spec.short_expiry

    def test_horizon_medium_uses_medium_expiry(self):
        f = StockForecaster()
        fc = f.forecast(_state(primary=Regime.CREDIT_STRESS, bear=3),
                        horizon=Horizon.MEDIUM)
        credit = next(o for o in fc.opportunities if o.industry == Industry.CREDIT)
        spec = next(s for s in INDUSTRY_PLAYBOOK if s.industry == Industry.CREDIT)
        assert credit.expiry_range_days == spec.medium_expiry

    def test_medium_horizon_boosts_private_credit_roi(self):
        f = StockForecaster()
        # PRIVATE_CREDIT activates on credit_stress
        short = f.forecast(_state(primary=Regime.CREDIT_STRESS), Horizon.SHORT)
        med = f.forecast(_state(primary=Regime.CREDIT_STRESS), Horizon.MEDIUM)
        pc_s = next(o for o in short.opportunities if o.industry == Industry.PRIVATE_CREDIT)
        pc_m = next(o for o in med.opportunities if o.industry == Industry.PRIVATE_CREDIT)
        # Medium adds +10 to roi (capped 100); short multiplies speed by 1.1
        assert pc_m.roi_score >= pc_s.roi_score

    def test_formula_boost_increases_credit_roi(self):
        f = StockForecaster()
        no_boost = f.forecast(_state(primary=Regime.CREDIT_STRESS, armed_formulas=[]))
        with_boost = f.forecast(_state(
            primary=Regime.CREDIT_STRESS,
            armed_formulas=[FormulaTag.F1_CREDIT_LED_BREAKDOWN],
        ))
        c_no = next(o for o in no_boost.opportunities if o.industry == Industry.CREDIT)
        c_yes = next(o for o in with_boost.opportunities if o.industry == Industry.CREDIT)
        assert c_yes.roi_score > c_no.roi_score
        assert FormulaTag.F1_CREDIT_LED_BREAKDOWN in c_yes.formula_sources

    def test_regime_confidence_scales_roi_and_speed(self):
        f = StockForecaster()
        full = f.forecast(_state(primary=Regime.CREDIT_STRESS, confidence=1.0))
        half = f.forecast(_state(primary=Regime.CREDIT_STRESS, confidence=0.5))
        c_full = next(o for o in full.opportunities if o.industry == Industry.CREDIT)
        c_half = next(o for o in half.opportunities if o.industry == Industry.CREDIT)
        # half-confidence gives ~50% of roi
        assert c_half.roi_score < c_full.roi_score
        assert c_half.speed_score < c_full.speed_score

    def test_policy_delay_trap_lowers_risk_for_defined_risk(self):
        f = StockForecaster()
        baseline = f.forecast(_state(primary=Regime.CREDIT_STRESS))
        trap = f.forecast(_state(primary=Regime.POLICY_DELAY_TRAP))
        c_base = next(o for o in baseline.opportunities if o.industry == Industry.CREDIT)
        c_trap = next(o for o in trap.opportunities if o.industry == Industry.CREDIT)
        # CREDIT uses PUT_SPREAD which qualifies for -15 risk reduction
        assert c_trap.risk_score == max(10, c_base.risk_score - 15)

    def test_secondary_regime_unions_vectors(self):
        f = StockForecaster()
        # primary=RISK_OFF (gives risk_off) + secondary=STAGFLATION (adds stagflation)
        fc = f.forecast(_state(primary=Regime.RISK_OFF,
                               secondary=Regime.STAGFLATION))
        inds = {o.industry for o in fc.opportunities}
        # Airlines activates only on stagflation/risk_off
        assert Industry.AIRLINES in inds

    def test_composite_formula(self):
        f = StockForecaster()
        fc = f.forecast(_state(primary=Regime.CREDIT_STRESS))
        for o in fc.opportunities:
            expected = round(o.roi_score * 0.40 + o.speed_score * 0.35
                             + (100 - o.risk_score) * 0.25, 1)
            assert o.composite_score == pytest.approx(expected)

    def test_all_opportunities_bearish(self):
        f = StockForecaster()
        fc = f.forecast(_state(primary=Regime.CREDIT_STRESS, bear=3))
        assert all(o.direction == Direction.BEARISH for o in fc.opportunities)

    def test_structure_hint_mentions_dte_range(self):
        f = StockForecaster()
        fc = f.forecast(_state(primary=Regime.CREDIT_STRESS), horizon=Horizon.SHORT)
        credit = next(o for o in fc.opportunities if o.industry == Industry.CREDIT)
        assert "DTE: 14-42" in credit.structure_hint
        assert "Short-term" in credit.structure_hint
        assert "HYG" in credit.structure_hint


# ═══════════════════════════════════════════════════════════════════════════
# Convenience methods
# ═══════════════════════════════════════════════════════════════════════════

class TestConvenienceMethods:
    def test_two_trade_stack_credit_stress(self):
        f = StockForecaster()
        credit, banks = f.two_trade_stack(_state(primary=Regime.CREDIT_STRESS))
        assert credit is not None
        assert credit.industry == Industry.CREDIT
        assert banks is not None
        assert banks.industry == Industry.BANKS

    def test_two_trade_stack_uncertain_returns_none(self):
        f = StockForecaster()
        credit, banks = f.two_trade_stack(_state(primary=Regime.UNCERTAIN, bear=0))
        assert credit is None
        assert banks is None

    def test_top_n_stack_returns_n(self):
        f = StockForecaster()
        out = f.top_n_stack(_state(primary=Regime.CREDIT_STRESS, bear=3), n=3)
        assert len(out) <= 3

    def test_top_n_stack_default(self):
        f = StockForecaster()
        out = f.top_n_stack(_state(primary=Regime.CREDIT_STRESS, bear=3))
        assert len(out) <= 3


# ═══════════════════════════════════════════════════════════════════════════
# print_industry_regime_matrix
# ═══════════════════════════════════════════════════════════════════════════

class TestPrintMatrix:
    def test_renders_matrix(self):
        out = print_industry_regime_matrix()
        assert "INDUSTRY × REGIME MATRIX" in out
        assert "CREDIT_STRESS" in out
        assert "STAGFLATION" in out
        assert "VOL_SHOCK_ARMED" in out

    def test_matrix_marks_energy_do_not_short(self):
        out = print_industry_regime_matrix()
        assert "DO NOT SHORT" in out
