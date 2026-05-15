"""Tests for F10–F14 (COT, ETF flow, breadth) formulas added 2026-04-13."""
from __future__ import annotations

from strategies.regime_engine import (
    FormulaTag,
    MacroSnapshot,
    RegimeEngine,
)


def _by_tag(results, tag):
    return next(r for r in results if r.tag is tag)


def test_f10_cot_extreme_long_fires_with_greed_confirmation():
    snap = MacroSnapshot(
        cot_es_extreme="extreme_long",
        cot_es_zscore=2.4,
        fear_greed=82,
    )
    engine = RegimeEngine()
    state = engine.evaluate(snap)
    f10 = _by_tag(state.formula_results, FormulaTag.F10_COT_LEVERAGED_EXTREME)
    assert f10.fired is True
    assert f10.confidence > 0.0


def test_f10_cot_neutral_does_not_fire():
    snap = MacroSnapshot(
        cot_es_extreme="neutral",
        cot_nq_extreme="neutral",
    )
    state = RegimeEngine().evaluate(snap)
    f10 = _by_tag(state.formula_results, FormulaTag.F10_COT_LEVERAGED_EXTREME)
    assert f10.fired is False


def test_f11_etf_outflow_capitulation_fires():
    snap = MacroSnapshot(
        etf_net_flow_usd=-5_000_000_000.0,
        etf_flow_samples=8,
        fear_greed=15,
    )
    state = RegimeEngine().evaluate(snap)
    f11 = _by_tag(state.formula_results, FormulaTag.F11_ETF_OUTFLOW_CAPITULATION)
    assert f11.fired is True
    assert f11.confidence > 0.3


def test_f11_etf_neutral_does_not_fire_without_samples():
    # Below sample threshold even if flow is huge
    snap = MacroSnapshot(
        etf_net_flow_usd=-5_000_000_000.0,
        etf_flow_samples=2,
    )
    state = RegimeEngine().evaluate(snap)
    f11 = _by_tag(state.formula_results, FormulaTag.F11_ETF_OUTFLOW_CAPITULATION)
    assert f11.fired is False


def test_f12_breadth_thrust_up_fires():
    snap = MacroSnapshot(
        mcclellan_oscillator=85.0,
        breadth_regime="bullish",
    )
    state = RegimeEngine().evaluate(snap)
    f12 = _by_tag(state.formula_results, FormulaTag.F12_BREADTH_THRUST)
    assert f12.fired is True


def test_f13_trin_capitulation_fires_with_tick_confirmation():
    snap = MacroSnapshot(trin=2.5, tick=-900.0)
    state = RegimeEngine().evaluate(snap)
    f13 = _by_tag(state.formula_results, FormulaTag.F13_TRIN_CAPITULATION)
    assert f13.fired is True
    assert f13.confidence > 0.0


def test_f14_vix_cot_short_squeeze_arms():
    snap = MacroSnapshot(
        cot_vx_extreme="extreme_short",
        cot_vx_zscore=-2.5,
        vix=14.0,
        hy_spread_bps=380,
    )
    state = RegimeEngine().evaluate(snap)
    f14 = _by_tag(state.formula_results, FormulaTag.F14_VIX_COT_SHORT_SQUEEZE)
    assert f14.fired is True


def test_evaluate_returns_14_results():
    snap = MacroSnapshot()
    state = RegimeEngine().evaluate(snap)
    assert len(state.formula_results) == 14
