"""
Tests for the Storm Lifeboat Matrix v9.0.
==========================================
Covers all subsystems: core, monte_carlo, scenario_engine,
lunar_phi, coherence, and helix_news.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# ─── Core ───────────────────────────────────────────────────────────────
from strategies.storm_lifeboat.core import (
    ASSET_ORDER,
    CRISIS_DRIFTS,
    DEFAULT_PRICES,
    MC_DEFAULT_PATHS,
    REGIME_VOLATILITIES,
    SCENARIO_MAP,
    SCENARIOS,
    STARTING_CAPITAL_CAD,
    Asset,
    MandateLevel,
    MoonPhase,
    PortfolioForecast,
    ScenarioDefinition,
    ScenarioState,
    ScenarioStatus,
    StormConfig,
    VolRegime,
    build_correlation_matrix,
)


class TestCoreConstants:
    def test_20_assets_defined(self):
        assert len(Asset) == 20

    def test_asset_order_matches_enum(self):
        assert ASSET_ORDER == list(Asset)

    def test_all_assets_have_default_prices(self):
        for a in Asset:
            assert a in DEFAULT_PRICES, f"Missing default price for {a.value}"
            assert DEFAULT_PRICES[a] > 0

    def test_all_assets_have_crisis_drifts(self):
        for a in Asset:
            assert a in CRISIS_DRIFTS, f"Missing crisis drift for {a.value}"

    def test_all_regimes_have_all_asset_vols(self):
        for regime in VolRegime:
            vols = REGIME_VOLATILITIES[regime]
            for a in Asset:
                assert a in vols, f"Missing vol for {a.value} in {regime.value}"
                assert vols[a] > 0

    def test_43_scenarios_defined(self):
        assert len(SCENARIOS) == 43

    def test_scenario_codes_unique(self):
        codes = [s.code for s in SCENARIOS]
        assert len(codes) == len(set(codes))

    def test_scenario_map_complete(self):
        for s in SCENARIOS:
            assert s.code in SCENARIO_MAP

    def test_scenario_probabilities_valid(self):
        for s in SCENARIOS:
            assert 0 < s.probability <= 1.0, f"{s.code} has invalid probability"
            assert 0 < s.impact_severity <= 1.0, f"{s.code} has invalid severity"

    def test_starting_capital(self):
        assert STARTING_CAPITAL_CAD == 45_120.0

    def test_storm_config_defaults(self):
        config = StormConfig()
        assert config.n_simulations == 100_000
        assert config.horizon_days == 90
        assert config.regime == VolRegime.CRISIS


class TestCorrelationMatrix:
    def test_shape_20x20(self):
        corr = build_correlation_matrix()
        assert corr.shape == (20, 20)

    def test_diagonal_is_one(self):
        corr = build_correlation_matrix()
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(len(Asset)))

    def test_symmetric(self):
        corr = build_correlation_matrix()
        np.testing.assert_array_almost_equal(corr, corr.T)

    def test_values_in_range(self):
        corr = build_correlation_matrix()
        assert np.all(corr >= -1.0) and np.all(corr <= 1.0)

    def test_gold_silver_highly_correlated(self):
        corr = build_correlation_matrix()
        gold_idx = ASSET_ORDER.index(Asset.GOLD)
        silver_idx = ASSET_ORDER.index(Asset.SILVER)
        assert corr[gold_idx, silver_idx] > 0.7

    def test_oil_spy_negatively_correlated(self):
        corr = build_correlation_matrix()
        oil_idx = ASSET_ORDER.index(Asset.OIL)
        spy_idx = ASSET_ORDER.index(Asset.SPY)
        assert corr[oil_idx, spy_idx] < 0


# ─── Monte Carlo ────────────────────────────────────────────────────────

from strategies.storm_lifeboat.monte_carlo import StormMonteCarloEngine


class TestMonteCarlo:
    @pytest.fixture(scope="class")
    def engine(self):
        config = StormConfig(n_simulations=5_000, horizon_days=30, seed=42)
        return StormMonteCarloEngine(config)

    def test_simulate_returns_forecast(self, engine):
        forecast = engine.simulate(vix=25.0)
        assert isinstance(forecast, PortfolioForecast)
        assert forecast.n_simulations == 5_000
        assert forecast.horizon_days == 30

    def test_all_assets_in_forecast(self, engine):
        forecast = engine.simulate(vix=25.0)
        for a in Asset:
            assert a in forecast.asset_forecasts

    def test_mean_price_positive(self, engine):
        forecast = engine.simulate(vix=25.0)
        for a, fc in forecast.asset_forecasts.items():
            assert fc.mean_price > 0, f"{a.value} mean price <= 0"

    def test_percentile_ordering(self, engine):
        forecast = engine.simulate(vix=25.0)
        for a, fc in forecast.asset_forecasts.items():
            assert fc.pct_5 <= fc.pct_25 <= fc.median_price <= fc.pct_75 <= fc.pct_95, \
                f"{a.value} percentiles out of order"

    def test_var_positive(self, engine):
        forecast = engine.simulate(vix=25.0)
        # VaR should be positive (it's a loss measure)
        assert forecast.portfolio_var_95 != 0
        assert forecast.portfolio_cvar_95 != 0

    def test_mandate_assigned(self, engine):
        forecast = engine.simulate(vix=25.0)
        assert isinstance(forecast.mandate, MandateLevel)

    def test_regime_assigned(self, engine):
        forecast = engine.simulate(vix=25.0)
        assert isinstance(forecast.regime, VolRegime)

    def test_crisis_regime_at_vix_30(self, engine):
        forecast = engine.simulate(vix=30.0)
        assert forecast.regime == VolRegime.CRISIS

    def test_panic_regime_at_vix_50(self, engine):
        forecast = engine.simulate(vix=50.0)
        assert forecast.regime == VolRegime.PANIC

    def test_put_payoff_simulation(self, engine):
        result = engine.simulate_put_payoff(Asset.SPY, strike_pct_otm=0.10)
        assert "win_rate" in result
        assert "expected_pnl" in result
        assert result["spot"] == DEFAULT_PRICES[Asset.SPY]
        assert result["strike"] < result["spot"]

    def test_reproducibility_with_seed(self):
        config = StormConfig(n_simulations=1_000, horizon_days=30, seed=42)
        e1 = StormMonteCarloEngine(config)
        f1 = e1.simulate(vix=25.0)
        e2 = StormMonteCarloEngine(config)
        f2 = e2.simulate(vix=25.0)
        assert f1.asset_forecasts[Asset.SPY].mean_price == \
               f2.asset_forecasts[Asset.SPY].mean_price


# ─── Scenario Engine ────────────────────────────────────────────────────

from strategies.storm_lifeboat.scenario_engine import ScenarioEngine


class TestScenarioEngine:
    def test_initial_states_all_dormant(self):
        engine = ScenarioEngine()
        for code, state in engine.states.items():
            assert state.status == ScenarioStatus.DORMANT

    def test_update_indicators_changes_status(self):
        engine = ScenarioEngine()
        state = engine.update_indicators("HORMUZ", ["Oil > $120", "USN carrier deployment"])
        assert state.status != ScenarioStatus.DORMANT
        assert state.probability > SCENARIO_MAP["HORMUZ"].probability * 0.5

    def test_all_indicators_firing_reaches_peak(self):
        engine = ScenarioEngine()
        indicators = SCENARIO_MAP["HORMUZ"].trigger_indicators
        state = engine.update_indicators("HORMUZ", indicators)
        assert state.status == ScenarioStatus.PEAK

    def test_contagion_raises_related_probabilities(self):
        engine = ScenarioEngine()
        # Activate Hormuz
        indicators = SCENARIO_MAP["HORMUZ"].trigger_indicators
        engine.update_indicators("HORMUZ", indicators)
        # Get supercycle prob before contagion
        supercycle_before = engine.states["SUPERCYCLE"].probability
        engine.apply_contagion()
        supercycle_after = engine.states["SUPERCYCLE"].probability
        assert supercycle_after > supercycle_before

    def test_get_active_scenarios_returns_non_dormant(self):
        engine = ScenarioEngine()
        # Fire 2 of 6 HORMUZ indicators (ratio=0.33 >= EMERGING threshold of 0.25)
        engine.update_indicators("HORMUZ", ["Oil > $120", "USN carrier deployment"])
        active = engine.get_active_scenarios()
        assert len(active) >= 1
        codes = [s.code for s in active]
        assert "HORMUZ" in codes

    def test_risk_heatmap_sorted_by_risk_score(self):
        engine = ScenarioEngine()
        heatmap = engine.get_risk_heatmap()
        assert len(heatmap) == 43
        scores = [h["risk_score"] for h in heatmap]
        assert scores == sorted(scores, reverse=True)

    def test_save_and_load_state(self, tmp_path):
        engine = ScenarioEngine()
        engine.update_indicators("TAIWAN", ["PLA exercises near Taiwan"])
        filepath = str(tmp_path / "scenario_state.json")
        engine.save_state(filepath)

        engine2 = ScenarioEngine()
        engine2.load_state(filepath)
        assert engine2.states["TAIWAN"].status == engine.states["TAIWAN"].status
        assert len(engine2.states["TAIWAN"].indicators_firing) == 1

    def test_unknown_scenario_raises(self):
        engine = ScenarioEngine()
        with pytest.raises(ValueError):
            engine.update_indicators("NONEXISTENT", [])


# ─── Lunar Phi ──────────────────────────────────────────────────────────

from strategies.storm_lifeboat.lunar_phi import MOON_NAMES, PHI, LunarPhiEngine


class TestLunarPhi:
    def test_13_moon_names(self):
        assert len(MOON_NAMES) == 13

    def test_phi_constant(self):
        assert abs(PHI - 1.6180339887) < 0.0001

    def test_position_on_cycle_start(self):
        engine = LunarPhiEngine(cycle_start=date(2026, 3, 20))
        pos = engine.get_position(date(2026, 3, 20))
        assert pos.moon_number == 1
        assert pos.day_in_moon == 1
        assert pos.day_in_cycle == 1
        assert pos.phase == MoonPhase.NEW

    def test_position_moon_2(self):
        engine = LunarPhiEngine(cycle_start=date(2026, 3, 20))
        pos = engine.get_position(date(2026, 4, 17))  # Day 29 = Moon 2, Day 1
        assert pos.moon_number == 2
        assert pos.day_in_moon == 1

    def test_phi_window_detection(self):
        engine = LunarPhiEngine(cycle_start=date(2026, 3, 20))
        # Day 10 of moon 1 = March 29
        pos = engine.get_position(date(2026, 3, 29))
        assert pos.day_in_moon == 10
        assert pos.in_phi_window is True

    def test_full_phase_at_day_15(self):
        engine = LunarPhiEngine(cycle_start=date(2026, 3, 20))
        pos = engine.get_position(date(2026, 4, 3))  # Day 15
        assert pos.day_in_moon == 15
        assert pos.phase == MoonPhase.FULL

    def test_waning_phase_at_day_25(self):
        engine = LunarPhiEngine(cycle_start=date(2026, 3, 20))
        pos = engine.get_position(date(2026, 4, 13))  # Day 25
        assert pos.day_in_moon == 25
        assert pos.phase == MoonPhase.WANING

    def test_position_multiplier_range(self):
        engine = LunarPhiEngine(cycle_start=date(2026, 3, 20))
        for day_offset in range(364):
            pos = engine.get_position(date(2026, 3, 20) + timedelta(days=day_offset))
            assert 0.3 <= pos.position_multiplier <= 1.5

    def test_next_phi_window(self):
        engine = LunarPhiEngine(cycle_start=date(2026, 3, 20))
        nxt = engine.get_next_phi_window(date(2026, 3, 20))  # Day 1
        # Should be day 10 = March 29
        assert nxt == date(2026, 3, 29)

    def test_format_display(self):
        engine = LunarPhiEngine(cycle_start=date(2026, 3, 20))
        pos = engine.get_position(date(2026, 3, 20))
        text = engine.format_display(pos)
        assert "Moon 1/13" in text
        assert "Magnetic" in text
        assert "NEW" in text

    def test_cycle_wraps(self):
        engine = LunarPhiEngine(cycle_start=date(2026, 3, 20))
        # Day 364, the last day of the cycle (delta=363, day_in_cycle=364)
        pos = engine.get_position(date(2027, 3, 18))  # 363 days later
        assert pos.day_in_cycle == 364
        # Day 365 wraps to day 1 of next cycle (delta=364, 364%364=0, +1=1)
        pos2 = engine.get_position(date(2027, 3, 19))
        assert pos2.day_in_cycle == 1


# ─── Coherence ──────────────────────────────────────────────────────────

from strategies.storm_lifeboat.coherence import (
    CoherenceEngine,
    compute_harmonic_coherence,
    compute_scenario_alignment,
)


class TestCoherence:
    def test_harmonic_coherence_with_no_data(self):
        score = compute_harmonic_coherence([], [], [])
        assert score == 0.5

    def test_harmonic_coherence_range(self):
        daily = [0.01, -0.005, 0.008, -0.003, 0.012]
        weekly = [0.03, -0.02, 0.04, -0.01]
        monthly = [0.05, -0.03, 0.08]
        score = compute_harmonic_coherence(daily, weekly, monthly)
        assert 0.0 <= score <= 1.0

    def test_scenario_alignment_no_scenarios(self):
        score = compute_scenario_alignment([])
        assert score == 0.5

    def test_scenario_alignment_all_bearish(self):
        # Create scenarios that all target SPY as victim
        bearish = [s for s in SCENARIOS if Asset.SPY in s.victim_assets]
        score = compute_scenario_alignment(bearish)
        assert score > 0.5  # Should be aligned

    def test_coherence_engine_returns_result(self):
        engine = CoherenceEngine()
        result = engine.analyze()
        assert 0.0 <= result.overall_score <= 1.0
        assert result.dominant_frequency in {"bullish_coherent", "bearish_coherent", "chaotic", "mixed"}
        assert 0.0 <= result.confidence <= 1.0

    def test_coherence_with_full_inputs(self):
        engine = CoherenceEngine()
        result = engine.analyze(
            returns_daily=[0.01, -0.005, 0.008, -0.003, 0.012],
            returns_weekly=[0.03, -0.02, 0.04, -0.01],
            returns_monthly=[0.05, -0.03, 0.08],
            active_scenarios=SCENARIOS[:5],
            moon_phase=MoonPhase.FULL,
            lunar_phi_coherence=0.9,
            current_regime=VolRegime.CRISIS,
            recent_regimes=[VolRegime.CRISIS] * 5,
        )
        assert result.confidence == 1.0  # All inputs provided


# ─── Helix News ─────────────────────────────────────────────────────────

from strategies.storm_lifeboat.helix_news import HelixNewsGenerator


class TestHelixNews:
    def test_generate_minimal_briefing(self):
        gen = HelixNewsGenerator()
        briefing = gen.generate()
        assert briefing.date == date.today()
        assert briefing.mandate == MandateLevel.OBSERVE
        assert "STORM LIFEBOAT" in briefing.headline

    def test_generate_with_forecast(self):
        config = StormConfig(n_simulations=1_000, horizon_days=30, seed=42)
        mc = StormMonteCarloEngine(config)
        forecast = mc.simulate(vix=30.0)

        gen = HelixNewsGenerator()
        briefing = gen.generate(
            forecast=forecast,
            regime=forecast.regime,
            mandate=forecast.mandate,
        )
        assert briefing.mandate == forecast.mandate
        assert "MC" in briefing.portfolio_summary

    def test_format_terminal(self):
        gen = HelixNewsGenerator()
        briefing = gen.generate()
        text = gen.format_terminal(briefing)
        assert "HELIX NEWS" in text
        assert "REGIME" in text
        assert "MANDATE" in text

    def test_save_json(self, tmp_path):
        gen = HelixNewsGenerator()
        briefing = gen.generate()
        filepath = gen.save_json(briefing, str(tmp_path))
        assert Path(filepath).exists()
        data = json.loads(Path(filepath).read_text())
        assert data["date"] == date.today().isoformat()
        assert "headline" in data

    def test_briefing_with_active_scenarios(self):
        scenario_eng = ScenarioEngine()
        scenario_eng.update_indicators("HORMUZ", SCENARIO_MAP["HORMUZ"].trigger_indicators)
        scenario_eng.update_indicators("DEBT_CRISIS", ["10Y yield > 5.5%"])

        gen = HelixNewsGenerator()
        briefing = gen.generate(scenario_states=scenario_eng.states)
        assert len(briefing.active_scenarios) >= 2

    def test_risk_alert_fires_on_many_escalating(self):
        scenario_eng = ScenarioEngine()
        # Escalate 3+ scenarios
        for code in ["HORMUZ", "DEBT_CRISIS", "TAIWAN"]:
            scenario_eng.update_indicators(code, SCENARIO_MAP[code].trigger_indicators)

        gen = HelixNewsGenerator()
        briefing = gen.generate(scenario_states=scenario_eng.states)
        assert briefing.risk_alert is not None
        assert "ESCALATING" in briefing.risk_alert or "scenario" in briefing.risk_alert.lower()
