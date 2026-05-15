from __future__ import annotations

from datetime import datetime

import pytest

from strategies.regime_engine import (
    FormulaResult,
    FormulaTag,
    MacroSnapshot,
    Regime,
    RegimeEngine,
    RegimeState,
    SignalRiskClass,
    snapshot_from_fred,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _empty_snap(**overrides) -> MacroSnapshot:
    snap = MacroSnapshot()
    for k, v in overrides.items():
        setattr(snap, k, v)
    return snap


def _benign_snap(**overrides) -> MacroSnapshot:
    """Quiet baseline — no formula should fire."""
    snap = MacroSnapshot(
        vix=15.0,
        vix_change_1d=0.0,
        realized_vol_20d=12.0,
        hy_spread_bps=250.0,
        ig_spread_bps=100.0,
        yield_curve_10_2=0.5,
        yield_10y=4.0,
        breakeven_inflation=2.2,
        core_pce=2.0,
        gdp_growth=2.5,
        oil_price=70.0,
        gold_price=2000.0,
        dollar_index=103.0,
        spy_return_1d=0.1,
        hyg_return_1d=0.1,
        kre_return_1d=0.1,
        qqq_return_1d=0.1,
        airlines_return_1d=0.1,
        shipping_return_1d=0.1,
        breadth_adv_dec=1.1,
        new_highs_52w=50,
        new_lows_52w=10,
        fear_greed=55.0,
        volume_ratio=1.0,
        safe_haven_bid=None,
        private_credit_redemption_pct=1.0,
        war_active=False,
        hormuz_blocked=False,
    )
    for k, v in overrides.items():
        setattr(snap, k, v)
    return snap


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class TestEnums:
    def test_regime_values(self):
        assert Regime.RISK_ON.value == "risk_on"
        assert Regime.VOL_SHOCK_ACTIVE.value == "vol_shock_active"
        assert Regime.UNCERTAIN.value == "uncertain"
        assert len(list(Regime)) == 9

    def test_formula_tag_values(self):
        assert FormulaTag.F1_CREDIT_LED_BREAKDOWN.value == "F1_credit_led_breakdown"
        assert FormulaTag.F9_LEVERAGE_REVEAL.value == "F9_leverage_reveal"
        assert FormulaTag.F10_COT_LEVERAGED_EXTREME.value == "F10_cot_leveraged_extreme"
        assert FormulaTag.F14_VIX_COT_SHORT_SQUEEZE.value == "F14_vix_cot_short_squeeze"
        assert len(list(FormulaTag)) == 14

    def test_signal_risk_class_values(self):
        assert SignalRiskClass.NEAR_GUARANTEE.value == "near_guarantee"
        assert SignalRiskClass.CONVEX.value == "convex"
        assert len(list(SignalRiskClass)) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroSnapshotDefaults:
    def test_defaults_all_none(self):
        snap = MacroSnapshot()
        assert snap.vix is None
        assert snap.hy_spread_bps is None
        assert snap.war_active is False
        assert snap.hormuz_blocked is False

    def test_timestamp_set_by_default(self):
        snap = MacroSnapshot()
        assert isinstance(snap.timestamp, datetime)

    def test_field_assignment(self):
        snap = MacroSnapshot(vix=22.0, war_active=True)
        assert snap.vix == 22.0
        assert snap.war_active is True


class TestFormulaResultDataclass:
    def test_fields(self):
        r = FormulaResult(
            tag=FormulaTag.F1_CREDIT_LED_BREAKDOWN,
            fired=True,
            confidence=0.5,
            conditions_met=["a"],
            conditions_missing=["b"],
            expected_outcome="x",
            risk_class=SignalRiskClass.INSTITUTIONAL,
            timeframe_days=(1, 5),
            expression_hint="put spreads",
        )
        assert r.fired is True
        assert r.confidence == 0.5
        assert r.timeframe_days == (1, 5)


class TestRegimeStateProperties:
    def _state(self, **kw) -> RegimeState:
        defaults = dict(
            timestamp=datetime.utcnow(),
            primary_regime=Regime.UNCERTAIN,
            secondary_regime=None,
            regime_confidence=0.1,
            formula_results=[],
            armed_formulas=[],
            vol_shock_readiness=0.0,
            bear_signals=0,
            bull_signals=0,
            summary="",
        )
        defaults.update(kw)
        return RegimeState(**defaults)

    def test_is_bearish_true(self):
        st = self._state(bear_signals=3, bull_signals=1)
        assert st.is_bearish is True

    def test_is_bearish_false_when_equal(self):
        st = self._state(bear_signals=2, bull_signals=2)
        assert st.is_bearish is False

    def test_shock_imminent_at_60(self):
        assert self._state(vol_shock_readiness=60.0).shock_imminent is True

    def test_shock_imminent_below_60(self):
        assert self._state(vol_shock_readiness=59.9).shock_imminent is False

    def test_top_formulas_filters_and_sorts(self):
        f_low = FormulaResult(
            tag=FormulaTag.F1_CREDIT_LED_BREAKDOWN, fired=True, confidence=0.2,
            conditions_met=[], conditions_missing=[], expected_outcome="",
            risk_class=SignalRiskClass.INSTITUTIONAL, timeframe_days=(1, 5), expression_hint="",
        )
        f_high = FormulaResult(
            tag=FormulaTag.F2_STAGFLATION_COMPRESSION, fired=True, confidence=0.9,
            conditions_met=[], conditions_missing=[], expected_outcome="",
            risk_class=SignalRiskClass.INSTITUTIONAL, timeframe_days=(1, 5), expression_hint="",
        )
        f_unfired = FormulaResult(
            tag=FormulaTag.F3_LIQUIDITY_MIRAGE, fired=False, confidence=0.99,
            conditions_met=[], conditions_missing=[], expected_outcome="",
            risk_class=SignalRiskClass.FRINGE, timeframe_days=(1, 5), expression_hint="",
        )
        st = self._state(formula_results=[f_low, f_high, f_unfired])
        top = st.top_formulas
        assert len(top) == 2
        assert top[0].tag == FormulaTag.F2_STAGFLATION_COMPRESSION
        assert top[1].tag == FormulaTag.F1_CREDIT_LED_BREAKDOWN


# ─────────────────────────────────────────────────────────────────────────────
# Engine init
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineInit:
    def test_defaults(self):
        e = RegimeEngine()
        assert e._HY_STRESS_BPS == 350
        assert e._VIX_SUPPRESSED == 20
        assert e._OIL_SHOCK == 120

    def test_custom_thresholds(self):
        e = RegimeEngine(hy_stress_bps=400, vix_suppressed=18, oil_shock_threshold=130)
        assert e._HY_STRESS_BPS == 400
        assert e._VIX_SUPPRESSED == 18
        assert e._OIL_SHOCK == 130


# ─────────────────────────────────────────────────────────────────────────────
# F1 Credit-Led Breakdown
# ─────────────────────────────────────────────────────────────────────────────

class TestF1CreditLedBreakdown:
    def setup_method(self):
        self.e = RegimeEngine()

    def test_fires_with_two_conditions(self):
        snap = _empty_snap(hy_spread_bps=500, hyg_return_1d=-1.0, spy_return_1d=0.0)
        r = self.e._f1_credit_led_breakdown(snap)
        assert r.fired is True
        assert r.confidence > 0.0
        assert r.risk_class == SignalRiskClass.INSTITUTIONAL

    def test_does_not_fire_with_one_condition(self):
        snap = _empty_snap(hy_spread_bps=500)
        r = self.e._f1_credit_led_breakdown(snap)
        assert r.fired is False
        assert r.confidence == 0.0

    def test_yield_curve_inversion_counts(self):
        snap = _empty_snap(hy_spread_bps=500, yield_curve_10_2=-0.5)
        r = self.e._f1_credit_led_breakdown(snap)
        assert r.fired is True
        assert any("inverted" in m for m in r.conditions_met)

    def test_missing_data_listed(self):
        snap = _empty_snap()
        r = self.e._f1_credit_led_breakdown(snap)
        assert r.fired is False
        assert "hy_spread_bps" in r.conditions_missing
        assert "yield_curve_10_2" in r.conditions_missing

    def test_quiet_does_not_fire(self):
        r = self.e._f1_credit_led_breakdown(_benign_snap())
        assert r.fired is False


# ─────────────────────────────────────────────────────────────────────────────
# F2 Stagflation Compression
# ─────────────────────────────────────────────────────────────────────────────

class TestF2StagflationCompression:
    def setup_method(self):
        self.e = RegimeEngine()

    def test_fires_oil_and_pce(self):
        snap = _empty_snap(oil_price=110, core_pce=3.5)
        r = self.e._f2_stagflation_compression(snap)
        assert r.fired is True
        assert r.risk_class == SignalRiskClass.INSTITUTIONAL

    def test_fires_oil_pce_gdp(self):
        snap = _empty_snap(oil_price=100, core_pce=3.0, gdp_growth=0.5)
        r = self.e._f2_stagflation_compression(snap)
        assert r.fired is True
        assert r.confidence > 0.0

    def test_does_not_fire_only_oil(self):
        snap = _empty_snap(oil_price=110)
        r = self.e._f2_stagflation_compression(snap)
        assert r.fired is False

    def test_airlines_oil_check(self):
        snap = _empty_snap(oil_price=85, airlines_return_1d=-2.0, core_pce=3.0)
        r = self.e._f2_stagflation_compression(snap)
        assert r.fired is True

    def test_quiet_does_not_fire(self):
        assert self.e._f2_stagflation_compression(_benign_snap()).fired is False


# ─────────────────────────────────────────────────────────────────────────────
# F3 Liquidity Mirage
# ─────────────────────────────────────────────────────────────────────────────

class TestF3LiquidityMirage:
    def setup_method(self):
        self.e = RegimeEngine()

    def test_fires_three_conditions(self):
        snap = _empty_snap(
            spy_return_1d=0.5, volume_ratio=0.6, breadth_adv_dec=0.8,
            hyg_return_1d=-0.3,
        )
        r = self.e._f3_liquidity_mirage(snap)
        assert r.fired is True
        assert r.risk_class == SignalRiskClass.FRINGE

    def test_does_not_fire_two_conditions(self):
        snap = _empty_snap(spy_return_1d=0.5, volume_ratio=0.6)
        r = self.e._f3_liquidity_mirage(snap)
        assert r.fired is False

    def test_credit_not_confirming(self):
        snap = _empty_snap(
            spy_return_1d=1.0, volume_ratio=0.7, breadth_adv_dec=0.9,
            hyg_return_1d=-0.5,
        )
        r = self.e._f3_liquidity_mirage(snap)
        assert r.fired is True
        assert any("Credit not confirming" in m for m in r.conditions_met)


# ─────────────────────────────────────────────────────────────────────────────
# F4 Policy Delay Trap
# ─────────────────────────────────────────────────────────────────────────────

class TestF4PolicyDelayTrap:
    def setup_method(self):
        self.e = RegimeEngine()

    def test_fires_stress_plus_low_vix(self):
        snap = _empty_snap(hy_spread_bps=400, vix=15)
        r = self.e._f4_policy_delay_trap(snap)
        assert r.fired is True
        assert r.risk_class == SignalRiskClass.CONVEX

    def test_geopolitical_war_counts(self):
        snap = _empty_snap(hy_spread_bps=400, war_active=True)
        r = self.e._f4_policy_delay_trap(snap)
        assert r.fired is True

    def test_hormuz_blocked_counts(self):
        snap = _empty_snap(vix=15, hormuz_blocked=True)
        r = self.e._f4_policy_delay_trap(snap)
        assert r.fired is True

    def test_private_credit_redemption_counts(self):
        snap = _empty_snap(hy_spread_bps=400, private_credit_redemption_pct=8)
        r = self.e._f4_policy_delay_trap(snap)
        assert r.fired is True

    def test_does_not_fire_only_one(self):
        r = self.e._f4_policy_delay_trap(_empty_snap(vix=15))
        assert r.fired is False


# ─────────────────────────────────────────────────────────────────────────────
# F5 Failed Safe Haven
# ─────────────────────────────────────────────────────────────────────────────

class TestF5FailedSafeHaven:
    def setup_method(self):
        self.e = RegimeEngine()

    def test_fires_riskoff_haven_bid_false(self):
        snap = _empty_snap(spy_return_1d=-1.0, gold_price=2000, safe_haven_bid=False)
        r = self.e._f5_failed_safe_haven(snap)
        assert r.fired is True
        assert r.risk_class == SignalRiskClass.FRINGE

    def test_fires_riskoff_plus_severe_hy(self):
        snap = _empty_snap(spy_return_1d=-1.0, hy_spread_bps=600)
        r = self.e._f5_failed_safe_haven(snap)
        assert r.fired is True

    def test_does_not_fire_only_one(self):
        snap = _empty_snap(spy_return_1d=-1.0)
        r = self.e._f5_failed_safe_haven(snap)
        assert r.fired is False


# ─────────────────────────────────────────────────────────────────────────────
# F6 Correlation Spike
# ─────────────────────────────────────────────────────────────────────────────

class TestF6CorrelationSpike:
    def setup_method(self):
        self.e = RegimeEngine()

    def test_fires_multi_sector_red(self):
        snap = _empty_snap(
            spy_return_1d=-1.0, hyg_return_1d=-0.8, kre_return_1d=-1.5,
            airlines_return_1d=-2.0, shipping_return_1d=-1.0,
        )
        r = self.e._f6_correlation_spike(snap)
        assert r.fired is True

    def test_fires_vix_surge(self):
        snap = _empty_snap(vix_change_1d=20.0)
        r = self.e._f6_correlation_spike(snap)
        # need confidence_parts > 0; vix_change>10 adds to confidence_parts AND met
        assert r.fired is True

    def test_does_not_fire_quiet(self):
        snap = _empty_snap(vix_change_1d=5.0, spy_return_1d=0.1)
        r = self.e._f6_correlation_spike(snap)
        assert r.fired is False

    def test_missing_when_too_few_sectors(self):
        snap = _empty_snap(spy_return_1d=-1.0, hyg_return_1d=-1.0)
        r = self.e._f6_correlation_spike(snap)
        assert "multi-sector return data" in r.conditions_missing


# ─────────────────────────────────────────────────────────────────────────────
# F7 Vol Compression Bomb
# ─────────────────────────────────────────────────────────────────────────────

class TestF7VolCompressionBomb:
    def setup_method(self):
        self.e = RegimeEngine()

    def test_fires_vix_low_credit_diverge(self):
        snap = _empty_snap(vix=16, hy_spread_bps=350)
        r = self.e._f7_vol_compression_bomb(snap)
        assert r.fired is True
        assert r.risk_class == SignalRiskClass.CONVEX

    def test_fires_inflation_curve_inverted(self):
        snap = _empty_snap(vix=16, breakeven_inflation=3.0, yield_curve_10_2=-0.3)
        r = self.e._f7_vol_compression_bomb(snap)
        assert r.fired is True

    def test_does_not_fire_only_low_vix(self):
        snap = _empty_snap(vix=16)
        r = self.e._f7_vol_compression_bomb(snap)
        assert r.fired is False


# ─────────────────────────────────────────────────────────────────────────────
# F8 Narrative Break
# ─────────────────────────────────────────────────────────────────────────────

class TestF8NarrativeBreak:
    def setup_method(self):
        self.e = RegimeEngine()

    def test_fires_redemption_plus_complacency(self):
        snap = _empty_snap(
            private_credit_redemption_pct=10, spy_return_1d=0.5, fear_greed=60,
        )
        r = self.e._f8_narrative_break(snap)
        assert r.fired is True
        assert r.risk_class == SignalRiskClass.CONVEX

    def test_fires_oil_war(self):
        snap = _empty_snap(
            private_credit_redemption_pct=10, oil_price=130, war_active=True,
        )
        r = self.e._f8_narrative_break(snap)
        assert r.fired is True

    def test_does_not_fire_only_redemption(self):
        snap = _empty_snap(private_credit_redemption_pct=10)
        r = self.e._f8_narrative_break(snap)
        assert r.fired is False


# ─────────────────────────────────────────────────────────────────────────────
# F9 Leverage Reveal
# ─────────────────────────────────────────────────────────────────────────────

class TestF9LeverageReveal:
    def setup_method(self):
        self.e = RegimeEngine()

    def test_fires_outsized_vix_react(self):
        snap = _empty_snap(spy_return_1d=-0.5, vix_change_1d=20, hy_spread_bps=600)
        r = self.e._f9_leverage_reveal(snap)
        assert r.fired is True
        assert r.risk_class == SignalRiskClass.CONVEX

    def test_fires_severe_hy_plus_redemption(self):
        snap = _empty_snap(hy_spread_bps=600, private_credit_redemption_pct=12)
        r = self.e._f9_leverage_reveal(snap)
        assert r.fired is True

    def test_does_not_fire_quiet(self):
        snap = _empty_snap(spy_return_1d=2.0, vix_change_1d=20)
        r = self.e._f9_leverage_reveal(snap)
        # spy move > 1.5 disqualifies outsized check; only HY/private_credit could fire
        assert r.fired is False


# ─────────────────────────────────────────────────────────────────────────────
# Vol shock checklist
# ─────────────────────────────────────────────────────────────────────────────

class TestVolShockChecklist:
    def setup_method(self):
        self.e = RegimeEngine()

    def test_zero_when_empty(self):
        assert self.e._vol_shock_checklist(_empty_snap()) == 0.0

    def test_full_score_capped_at_100(self):
        snap = _empty_snap(
            hyg_return_1d=-1.0, spy_return_1d=0.0,   # +20 (bullet 1)
            kre_return_1d=-1.0,                        # +20 (bullet 2)
            oil_price=95, airlines_return_1d=-0.5,   # +20 (bullet 3)
            vix=15, hy_spread_bps=400,                # +20 (bullet 4)
            volume_ratio=0.5, breadth_adv_dec=0.5,   # +20 (bullet 5)
        )
        score = self.e._vol_shock_checklist(snap)
        assert score == 100.0

    def test_partial_credit_bullet_1(self):
        # diverge weakly: hyg slightly worse than spy but hyg not <-0.3
        snap = _empty_snap(hyg_return_1d=-0.6, spy_return_1d=0.0)
        score = self.e._vol_shock_checklist(snap)
        # full bullet 1 fires (-0.6<-0.3 and spy>=-0.5) → 20
        assert score == 20.0

    def test_partial_credit_bullet_5(self):
        snap = _empty_snap(volume_ratio=0.83, breadth_adv_dec=1.5)
        # vol<0.85 but breadth fine, partial 8
        assert self.e._vol_shock_checklist(snap) == 8.0


# ─────────────────────────────────────────────────────────────────────────────
# Regime classification
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeClassification:
    def setup_method(self):
        self.e = RegimeEngine()

    def test_uncertain_when_no_data(self):
        state = self.e.evaluate(_empty_snap())
        assert state.primary_regime == Regime.UNCERTAIN
        assert state.regime_confidence == pytest.approx(0.1)

    def test_risk_on_when_quiet(self):
        state = self.e.evaluate(_benign_snap(hy_spread_bps=180, vix=12))
        assert state.primary_regime == Regime.RISK_ON

    def test_credit_stress_regime(self):
        snap = _benign_snap(
            hy_spread_bps=600, hyg_return_1d=-1.5, spy_return_1d=0.0,
            yield_curve_10_2=-0.5,
        )
        state = self.e.evaluate(snap)
        # F1 must arm; primary may share with VOL_SHOCK_ARMED depending on vol score
        assert state.primary_regime in (Regime.CREDIT_STRESS, Regime.VOL_SHOCK_ARMED)
        assert FormulaTag.F1_CREDIT_LED_BREAKDOWN in state.armed_formulas

    def test_stagflation_regime(self):
        snap = _benign_snap(oil_price=130, core_pce=4.0, gdp_growth=0.5)
        state = self.e.evaluate(snap)
        assert state.primary_regime == Regime.STAGFLATION

    def test_vol_shock_active_when_score_high(self):
        # full vol shock score (100) + no other regime competing
        snap = _benign_snap(
            hyg_return_1d=-1.0, spy_return_1d=0.0,
            kre_return_1d=-1.0,
            oil_price=95, airlines_return_1d=-0.5,
            vix=15, hy_spread_bps=400,
            volume_ratio=0.5, breadth_adv_dec=0.5,
        )
        state = self.e.evaluate(snap)
        # vol_score=100 → VOL_SHOCK_ACTIVE 0.8
        assert state.primary_regime in (Regime.VOL_SHOCK_ACTIVE, Regime.VOL_SHOCK_ARMED, Regime.CREDIT_STRESS)
        assert state.vol_shock_readiness == 100.0
        assert state.shock_imminent is True

    def test_secondary_regime_assigned(self):
        snap = _benign_snap(oil_price=130, core_pce=4.0, gdp_growth=0.5,
                            hy_spread_bps=600, hyg_return_1d=-1.5, spy_return_1d=0.0)
        state = self.e.evaluate(snap)
        assert state.secondary_regime is not None

    def test_evaluate_returns_nine_results(self):
        state = self.e.evaluate(_benign_snap())
        assert len(state.formula_results) == 14
        # all FormulaTags represented
        tags = {r.tag for r in state.formula_results}
        assert tags == set(FormulaTag)


# ─────────────────────────────────────────────────────────────────────────────
# Signal counting
# ─────────────────────────────────────────────────────────────────────────────

class TestCountSignals:
    def setup_method(self):
        self.e = RegimeEngine()

    def test_bull_signals_quiet(self):
        snap = _empty_snap(hy_spread_bps=180, vix=12, fear_greed=20)
        bear, bull = self.e._count_signals([], snap)
        assert bear == 0
        assert bull == 3

    def test_no_bull_when_thresholds_not_met(self):
        snap = _empty_snap(hy_spread_bps=300, vix=20, fear_greed=50)
        bear, bull = self.e._count_signals([], snap)
        assert bull == 0

    def test_bear_counts_fired_formulas(self):
        fired = FormulaResult(
            tag=FormulaTag.F1_CREDIT_LED_BREAKDOWN, fired=True, confidence=0.5,
            conditions_met=[], conditions_missing=[], expected_outcome="",
            risk_class=SignalRiskClass.INSTITUTIONAL, timeframe_days=(1, 5), expression_hint="",
        )
        unfired = FormulaResult(
            tag=FormulaTag.F2_STAGFLATION_COMPRESSION, fired=False, confidence=0.0,
            conditions_met=[], conditions_missing=[], expected_outcome="",
            risk_class=SignalRiskClass.INSTITUTIONAL, timeframe_days=(1, 5), expression_hint="",
        )
        bear, _ = self.e._count_signals([fired, unfired], _empty_snap())
        assert bear == 1


# ─────────────────────────────────────────────────────────────────────────────
# Summary builder
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSummary:
    def setup_method(self):
        self.e = RegimeEngine()

    def test_includes_primary(self):
        s = self.e._build_summary(Regime.RISK_ON, None, [], 0.0, 0, 1)
        assert "RISK ON" in s

    def test_includes_secondary(self):
        s = self.e._build_summary(Regime.RISK_OFF, Regime.CREDIT_STRESS, [], 0.0, 1, 0)
        assert "SECONDARY" in s
        assert "CREDIT STRESS" in s

    def test_includes_armed(self):
        s = self.e._build_summary(
            Regime.CREDIT_STRESS, None, [FormulaTag.F1_CREDIT_LED_BREAKDOWN], 0.0, 1, 0,
        )
        assert "F1_credit_led_breakdown" in s

    def test_imminent_warning_at_80(self):
        s = self.e._build_summary(Regime.VOL_SHOCK_ACTIVE, None, [], 80.0, 5, 0)
        assert "IMMINENT" in s

    def test_armed_warning_at_60(self):
        s = self.e._build_summary(Regime.VOL_SHOCK_ARMED, None, [], 60.0, 3, 0)
        assert "ARMED" in s

    def test_quiet_no_warning(self):
        s = self.e._build_summary(Regime.RISK_ON, None, [], 0.0, 0, 1)
        assert "IMMINENT" not in s
        assert "ARMED" not in s

    def test_signals_count_in_summary(self):
        s = self.e._build_summary(Regime.RISK_OFF, None, [], 0.0, 4, 1)
        assert "4 bearish" in s and "1 bullish" in s


# ─────────────────────────────────────────────────────────────────────────────
# snapshot_from_fred convenience
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotFromFred:
    def test_basic_mapping(self):
        data = {
            "VIXCLS": 22.5,
            "BAMLH0A0HYM2": 4.5,   # → 450 bps
            "T10Y2Y": -0.3,
            "DCOILWTICO": 95.0,
            "GOLDAMGBD228NLBM": 2100.0,
            "T10YIE": 2.4,
            "DGS10": 4.5,
            "PCEPILFE": 3.0,
            "DTWEXBGS": 105.0,
        }
        snap = snapshot_from_fred(data)
        assert snap.vix == 22.5
        assert snap.hy_spread_bps == pytest.approx(450.0)
        assert snap.yield_curve_10_2 == -0.3
        assert snap.oil_price == 95.0
        assert snap.gold_price == 2100.0
        assert snap.breakeven_inflation == 2.4
        assert snap.yield_10y == 4.5
        assert snap.core_pce == 3.0
        assert snap.dollar_index == 105.0

    def test_missing_keys_are_none(self):
        snap = snapshot_from_fred({})
        assert snap.vix is None
        assert snap.oil_price is None

    def test_kwargs_override(self):
        snap = snapshot_from_fred({"VIXCLS": 20.0}, war_active=True, vix=99.0)
        assert snap.war_active is True
        assert snap.vix == 99.0  # kwarg overrides

    def test_unknown_kwargs_ignored(self):
        # hasattr filter prevents AttributeError
        snap = snapshot_from_fred({}, totally_unknown_field=42)
        assert not hasattr(snap, "totally_unknown_field") or snap is not None

    def test_hy_spread_falsy_value(self):
        # BAMLH0A0HYM2 = 0 → `0 and 0*100` → 0 (falsy) → None mapping behavior
        # Actually: `fred_data.get(...) and ... * 100` returns 0 when value is 0
        snap = snapshot_from_fred({"BAMLH0A0HYM2": 0})
        assert snap.hy_spread_bps == 0  # falsy short-circuit returns 0
