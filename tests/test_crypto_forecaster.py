from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from strategies.crypto_forecaster import (
    CryptoDirection,
    CryptoExpressionType,
    CryptoForecaster,
    CryptoFormula,
    CryptoFormulaResult,
    CryptoRegime,
    CryptoRegimeState,
    CryptoSnapshot,
    snapshot_from_coingecko,
)


def _snap(**kwargs: Any) -> CryptoSnapshot:
    return CryptoSnapshot(**kwargs)


def _full_snap(**overrides: Any) -> CryptoSnapshot:
    """Snapshot with neutral-ish defaults so individual formulas can be toggled."""
    base: dict[str, Any] = dict(
        btc_price=65000.0,
        btc_return_1d=0.5,
        btc_return_7d=2.0,
        eth_price=3500.0,
        eth_return_1d=0.4,
        eth_btc_ratio=0.054,
        btc_dominance_pct=52.0,
        btc_dominance_change_1d=0.1,
        funding_rate_btc=0.005,
        funding_rate_eth=0.005,
        open_interest_btc_usd=1.5e10,
        oi_change_pct_24h=2.0,
        exchange_inflow_btc=100.0,
        exchange_inflow_change_pct=5.0,
        iv_btc_1w=55.0,
        iv_btc_1m=55.0,
        realized_vol_7d=55.0,
        fear_greed=50.0,
        spx_return_1d=0.2,
        dxy_change_1d=0.0,
        m2_global_growth=4.0,
        global_liquidity_trend="neutral",
    )
    base.update(overrides)
    return CryptoSnapshot(**base)


# ─────────────────────────────────────────────────────────────────────────
# TestEnums
# ─────────────────────────────────────────────────────────────────────────

class TestEnums:
    def test_crypto_regime_values(self):
        assert CryptoRegime.REFLEXIVE_MELT_UP.value == "reflexive_melt_up"
        assert CryptoRegime.UNCERTAIN.value == "uncertain"
        assert len(list(CryptoRegime)) == 10

    def test_crypto_direction_values(self):
        assert CryptoDirection.LONG.value == "long"
        assert CryptoDirection.REDUCE_ALTS.value == "reduce_alts"
        assert {d.value for d in CryptoDirection} == {"long", "short", "neutral", "reduce_alts"}

    def test_crypto_expression_types(self):
        assert CryptoExpressionType.AVOID.value == "avoid"
        assert CryptoExpressionType.DELTA_NEUTRAL.value == "delta_neutral"
        assert len(list(CryptoExpressionType)) == 9

    def test_crypto_formula_set(self):
        formulas = {f.value for f in CryptoFormula}
        assert "C1_liquidity_reflexive_melt" in formulas
        assert "C8_correlation_breakout" in formulas
        assert len(formulas) == 8


# ─────────────────────────────────────────────────────────────────────────
# TestCryptoSnapshot
# ─────────────────────────────────────────────────────────────────────────

class TestCryptoSnapshot:
    def test_defaults_all_optional_none(self):
        s = CryptoSnapshot()
        assert s.btc_price is None
        assert s.fear_greed is None
        assert s.global_liquidity_trend is None
        assert isinstance(s.timestamp, datetime)

    def test_field_assignment(self):
        s = CryptoSnapshot(btc_price=70000.0, funding_rate_btc=0.025)
        assert s.btc_price == 70000.0
        assert s.funding_rate_btc == pytest.approx(0.025)


# ─────────────────────────────────────────────────────────────────────────
# TestCryptoRegimeStatePrint
# ─────────────────────────────────────────────────────────────────────────

class TestCryptoRegimeStatePrint:
    def _state_with_results(self, results: list[CryptoFormulaResult]) -> CryptoRegimeState:
        return CryptoRegimeState(
            timestamp=datetime.utcnow(),
            primary_regime=CryptoRegime.LEVERAGE_FRAGILITY,
            secondary_regime=None,
            regime_confidence=0.8,
            formula_results=results,
            armed_formulas=[r.formula for r in results if r.fired],
            long_signals=0,
            short_signals=sum(1 for r in results if r.fired),
            net_bias=CryptoDirection.SHORT,
        )

    def test_print_plan_contains_header(self):
        st = self._state_with_results([])
        out = st.print_plan()
        assert "CRYPTO FORECASTER" in out
        assert "LEVERAGE FRAGILITY" in out
        assert "SHORT" in out

    def test_print_plan_renders_fired_results(self):
        r = CryptoFormulaResult(
            formula=CryptoFormula.C2_LEVERAGE_FLUSH,
            fired=True,
            confidence=0.7,
            direction=CryptoDirection.SHORT,
            conditions_met=["funding extreme"],
            conditions_missing=[],
            expected_outcome="cascade",
            timeframe_str="hours",
            expression=CryptoExpressionType.PERP_SHORT,
            risk_level="high",
        )
        out = self._state_with_results([r]).print_plan()
        assert "C2_leverage_fragility_flush" in out
        assert "cascade" in out
        assert "perp_short" in out
        assert "high" in out
        assert "funding extreme" in out

    def test_print_plan_skips_unfired(self):
        r = CryptoFormulaResult(
            formula=CryptoFormula.C1_LIQUIDITY_MELT,
            fired=False,
            confidence=0.0,
            direction=CryptoDirection.LONG,
            conditions_met=[],
            conditions_missing=["x"],
            expected_outcome="never shown",
            timeframe_str="n/a",
            expression=CryptoExpressionType.SPOT_LONG,
            risk_level="low",
        )
        out = self._state_with_results([r]).print_plan()
        assert "never shown" not in out


# ─────────────────────────────────────────────────────────────────────────
# TestC1LiquidityMelt
# ─────────────────────────────────────────────────────────────────────────

class TestC1LiquidityMelt:
    def test_fires_with_three_conditions(self):
        snap = _full_snap(
            global_liquidity_trend="expanding",
            dxy_change_1d=-0.5,
            funding_rate_btc=0.003,
            iv_btc_1m=50.0,
        )
        r = CryptoForecaster()._c1_liquidity_melt(snap)
        assert r.fired
        assert r.direction == CryptoDirection.LONG
        assert r.expression == CryptoExpressionType.SPOT_LONG
        assert r.confidence > 0.0

    def test_does_not_fire_on_contracting_liquidity_alone(self):
        snap = _full_snap(global_liquidity_trend="contracting", dxy_change_1d=0.5)
        r = CryptoForecaster()._c1_liquidity_melt(snap)
        assert not r.fired
        assert r.confidence == 0.0
        assert any("Liquidity contracting" in m for m in r.conditions_missing)


# ─────────────────────────────────────────────────────────────────────────
# TestC2LeverageFlush
# ─────────────────────────────────────────────────────────────────────────

class TestC2LeverageFlush:
    def test_fires_on_extreme_funding_plus_oi_spike(self):
        snap = _full_snap(funding_rate_btc=0.02, oi_change_pct_24h=15.0, fear_greed=80.0)
        r = CryptoForecaster()._c2_leverage_flush(snap)
        assert r.fired
        assert r.direction == CryptoDirection.SHORT
        assert r.expression == CryptoExpressionType.PERP_SHORT

    def test_distribution_branch_oi_up_price_down(self):
        snap = _full_snap(
            funding_rate_btc=0.015,
            oi_change_pct_24h=8.0,  # below SPIKE=10 but >5 for distribution branch
            btc_return_1d=-1.0,
        )
        r = CryptoForecaster()._c2_leverage_flush(snap)
        # met: funding extreme + distribution = 2 met → fires
        assert r.fired
        assert any("distribution" in m for m in r.conditions_met)

    def test_does_not_fire_with_only_one_signal(self):
        snap = _full_snap(funding_rate_btc=0.005, oi_change_pct_24h=2.0, fear_greed=50.0)
        r = CryptoForecaster()._c2_leverage_flush(snap)
        assert not r.fired


# ─────────────────────────────────────────────────────────────────────────
# TestC3Decoupling
# ─────────────────────────────────────────────────────────────────────────

class TestC3Decoupling:
    def test_fires_on_underperformance_plus_squeeze_setup(self):
        snap = _full_snap(btc_return_1d=-3.0, spx_return_1d=0.5, funding_rate_btc=0.005)
        # condition1: btc < spx-2 (-3 < -1.5) ✓
        # condition2: btc<-2 AND funding>0 ✓
        r = CryptoForecaster()._c3_decoupling(snap)
        assert r.fired
        assert r.direction == CryptoDirection.SHORT
        assert r.expression == CryptoExpressionType.REDUCE_LEVERAGE

    def test_fires_on_underperformance_plus_tradfi_stress(self):
        snap = _full_snap(btc_return_1d=-5.0, spx_return_1d=-2.0, funding_rate_btc=-0.001)
        # cond1: -5 < -2-2 = -4 ✓; tradfi stress: -2<-1.5 ✓
        r = CryptoForecaster()._c3_decoupling(snap)
        assert r.fired

    def test_does_not_fire_when_btc_outperforms(self):
        snap = _full_snap(btc_return_1d=1.0, spx_return_1d=0.5)
        r = CryptoForecaster()._c3_decoupling(snap)
        assert not r.fired


# ─────────────────────────────────────────────────────────────────────────
# TestC4VolCompression
# ─────────────────────────────────────────────────────────────────────────

class TestC4VolCompression:
    def test_fires_compressed_iv_plus_iv_below_realized(self):
        snap = _full_snap(iv_btc_1m=30.0, realized_vol_7d=45.0, spx_return_1d=0.0)
        r = CryptoForecaster()._c4_vol_compression(snap)
        assert r.fired
        assert r.direction == CryptoDirection.NEUTRAL
        assert r.expression == CryptoExpressionType.OPTIONS_CALL

    def test_direction_short_when_spx_weak(self):
        snap = _full_snap(iv_btc_1m=30.0, realized_vol_7d=45.0, spx_return_1d=-1.5)
        r = CryptoForecaster()._c4_vol_compression(snap)
        assert r.fired
        assert r.direction == CryptoDirection.SHORT
        assert r.expression == CryptoExpressionType.OPTIONS_PUT

    def test_does_not_fire_with_normal_iv(self):
        snap = _full_snap(iv_btc_1m=60.0, realized_vol_7d=55.0, spx_return_1d=0.0)
        r = CryptoForecaster()._c4_vol_compression(snap)
        assert not r.fired


# ─────────────────────────────────────────────────────────────────────────
# TestC5ExchangeInflow
# ─────────────────────────────────────────────────────────────────────────

class TestC5ExchangeInflow:
    def test_fires_on_inflow_spike(self):
        snap = _full_snap(exchange_inflow_change_pct=30.0)
        r = CryptoForecaster()._c5_exchange_inflow(snap)
        assert r.fired
        assert r.direction == CryptoDirection.SHORT
        assert r.risk_level == "low"

    def test_distribution_branch_with_7d_pump(self):
        snap = _full_snap(exchange_inflow_change_pct=25.0, btc_return_7d=15.0)
        r = CryptoForecaster()._c5_exchange_inflow(snap)
        assert r.fired
        assert any("distribution" in m for m in r.conditions_met)

    def test_does_not_fire_below_threshold(self):
        snap = _full_snap(exchange_inflow_change_pct=10.0)
        r = CryptoForecaster()._c5_exchange_inflow(snap)
        assert not r.fired

    def test_missing_data_not_fired(self):
        snap = _full_snap(exchange_inflow_change_pct=None)
        r = CryptoForecaster()._c5_exchange_inflow(snap)
        assert not r.fired
        assert "exchange_inflow_change_pct" in r.conditions_missing


# ─────────────────────────────────────────────────────────────────────────
# TestC6BtcDominance
# ─────────────────────────────────────────────────────────────────────────

class TestC6BtcDominance:
    def test_fires_on_dominance_rising(self):
        snap = _full_snap(btc_dominance_change_1d=2.0)
        r = CryptoForecaster()._c6_btc_dominance(snap)
        assert r.fired
        assert r.direction == CryptoDirection.REDUCE_ALTS

    def test_does_not_fire_when_dominance_flat(self):
        snap = _full_snap(btc_dominance_change_1d=0.2)
        r = CryptoForecaster()._c6_btc_dominance(snap)
        assert not r.fired


# ─────────────────────────────────────────────────────────────────────────
# TestC7FundingReversion
# ─────────────────────────────────────────────────────────────────────────

class TestC7FundingReversion:
    def test_fires_on_extreme_positive_funding_short(self):
        snap = _full_snap(funding_rate_btc=0.04)
        r = CryptoForecaster()._c7_funding_reversion(snap)
        assert r.fired
        assert r.direction == CryptoDirection.SHORT
        assert r.expression == CryptoExpressionType.DELTA_NEUTRAL

    def test_fires_on_extreme_negative_funding_long(self):
        snap = _full_snap(funding_rate_btc=-0.01)
        r = CryptoForecaster()._c7_funding_reversion(snap)
        assert r.fired
        assert r.direction == CryptoDirection.LONG
        assert r.expression == CryptoExpressionType.SPOT_LONG

    def test_neutral_funding_does_not_fire(self):
        snap = _full_snap(funding_rate_btc=0.005)
        r = CryptoForecaster()._c7_funding_reversion(snap)
        assert not r.fired
        assert r.direction == CryptoDirection.NEUTRAL

    def test_missing_funding_not_fired(self):
        snap = _full_snap(funding_rate_btc=None)
        r = CryptoForecaster()._c7_funding_reversion(snap)
        assert not r.fired
        assert "funding_rate_btc" in r.conditions_missing


# ─────────────────────────────────────────────────────────────────────────
# TestC8CorrelationBreak
# ─────────────────────────────────────────────────────────────────────────

class TestC8CorrelationBreak:
    def test_fires_on_btc_up_spx_down(self):
        snap = _full_snap(btc_return_1d=3.0, spx_return_1d=-1.0)
        r = CryptoForecaster()._c8_correlation_break(snap)
        assert r.fired
        assert r.direction == CryptoDirection.LONG
        assert r.expression == CryptoExpressionType.SPOT_LONG

    def test_fires_on_btc_down_spx_up(self):
        snap = _full_snap(btc_return_1d=-3.0, spx_return_1d=1.0)
        r = CryptoForecaster()._c8_correlation_break(snap)
        assert r.fired
        assert r.direction == CryptoDirection.SHORT
        assert r.expression == CryptoExpressionType.PERP_SHORT

    def test_does_not_fire_when_aligned(self):
        snap = _full_snap(btc_return_1d=2.5, spx_return_1d=1.0)
        r = CryptoForecaster()._c8_correlation_break(snap)
        assert not r.fired

    def test_does_not_fire_on_small_btc_move(self):
        snap = _full_snap(btc_return_1d=1.0, spx_return_1d=-1.0)  # |btc|<=2
        r = CryptoForecaster()._c8_correlation_break(snap)
        assert not r.fired


# ─────────────────────────────────────────────────────────────────────────
# TestEvaluateAggregation
# ─────────────────────────────────────────────────────────────────────────

class TestEvaluateAggregation:
    def test_evaluate_returns_state_with_eight_results(self):
        snap = _full_snap()
        st = CryptoForecaster().evaluate(snap)
        assert isinstance(st, CryptoRegimeState)
        assert len(st.formula_results) == 8

    def test_evaluate_long_bias_when_c1_alone_fires(self):
        snap = _full_snap(
            global_liquidity_trend="expanding",
            dxy_change_1d=-0.5,
            funding_rate_btc=0.003,
            iv_btc_1m=50.0,
        )
        st = CryptoForecaster().evaluate(snap)
        assert st.long_signals >= 1
        assert st.net_bias == CryptoDirection.LONG

    def test_evaluate_short_bias_on_leverage_flush(self):
        snap = _full_snap(funding_rate_btc=0.025, oi_change_pct_24h=20.0, fear_greed=85.0)
        st = CryptoForecaster().evaluate(snap)
        assert st.short_signals >= st.long_signals
        assert st.net_bias in (CryptoDirection.SHORT, CryptoDirection.NEUTRAL)

    def test_evaluate_neutral_when_nothing_fires(self):
        snap = _full_snap()
        st = CryptoForecaster().evaluate(snap)
        assert st.long_signals == 0
        assert st.short_signals == 0
        assert st.net_bias == CryptoDirection.NEUTRAL


# ─────────────────────────────────────────────────────────────────────────
# TestRegimeClassification
# ─────────────────────────────────────────────────────────────────────────

class TestRegimeClassification:
    def test_uncertain_when_no_formula_fires(self):
        snap = _full_snap()
        st = CryptoForecaster().evaluate(snap)
        assert st.primary_regime == CryptoRegime.UNCERTAIN
        assert st.regime_confidence == 0.0

    def test_leverage_fragility_dominant(self):
        snap = _full_snap(funding_rate_btc=0.025, oi_change_pct_24h=20.0, fear_greed=85.0)
        st = CryptoForecaster().evaluate(snap)
        assert st.primary_regime == CryptoRegime.LEVERAGE_FRAGILITY
        assert st.regime_confidence > 0.0

    def test_reflexive_melt_up_classification(self):
        snap = _full_snap(
            global_liquidity_trend="expanding",
            dxy_change_1d=-0.5,
            funding_rate_btc=0.003,
            iv_btc_1m=50.0,
        )
        st = CryptoForecaster().evaluate(snap)
        assert st.primary_regime == CryptoRegime.REFLEXIVE_MELT_UP

    def test_accumulation_added_on_extreme_fear(self):
        snap = _full_snap(fear_greed=10.0)
        st = CryptoForecaster().evaluate(snap)
        assert st.primary_regime == CryptoRegime.ACCUMULATION

    def test_secondary_regime_set_when_two_formulas_fire(self):
        snap = _full_snap(
            funding_rate_btc=0.025,           # C2
            oi_change_pct_24h=20.0,
            fear_greed=85.0,
            btc_dominance_change_1d=2.0,      # C6
        )
        st = CryptoForecaster().evaluate(snap)
        assert st.primary_regime == CryptoRegime.LEVERAGE_FRAGILITY
        assert st.secondary_regime == CryptoRegime.ALT_LIQUIDATION

    def test_confidence_capped_at_one(self):
        snap = _full_snap(funding_rate_btc=0.025, oi_change_pct_24h=20.0, fear_greed=10.0)
        st = CryptoForecaster().evaluate(snap)
        assert 0.0 <= st.regime_confidence <= 1.0


# ─────────────────────────────────────────────────────────────────────────
# TestSnapshotFromCoinGecko
# ─────────────────────────────────────────────────────────────────────────

class TestSnapshotFromCoinGecko:
    def test_basic_mapping(self):
        snap = snapshot_from_coingecko({
            "btc_price": 65000.0,
            "btc_change_24h": 1.5,
            "eth_price": 3500.0,
            "eth_change_24h": 0.8,
            "btc_dominance": 52.3,
            "fear_greed_value": 60,
        })
        assert snap.btc_price == 65000.0
        assert snap.btc_return_1d == pytest.approx(1.5)
        assert snap.eth_price == 3500.0
        assert snap.btc_dominance_pct == pytest.approx(52.3)
        assert snap.fear_greed == 60

    def test_kwargs_set_known_attrs(self):
        snap = snapshot_from_coingecko({}, funding_rate_btc=0.018, spx_return_1d=-1.2)
        assert snap.funding_rate_btc == pytest.approx(0.018)
        assert snap.spx_return_1d == pytest.approx(-1.2)

    def test_kwargs_ignore_unknown_attrs(self):
        snap = snapshot_from_coingecko({}, made_up_field=999)
        assert not hasattr(snap, "made_up_field")

    def test_empty_dict_yields_none_values(self):
        snap = snapshot_from_coingecko({})
        assert snap.btc_price is None
        assert snap.fear_greed is None
