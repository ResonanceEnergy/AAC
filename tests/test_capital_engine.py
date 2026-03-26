"""
Tests for the Gold-Oil-Silver See-Saw Capital Engine.
======================================================
Covers: phase detection, signal generation, stop-loss mechanics,
position tracking, cycle reporting, and persistence.
"""
from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strategies.storm_lifeboat.capital_engine import (
    CycleReport,
    DEFAULT_WEIGHTS,
    LifeboatCapitalEngine,
    SeeSawPhase,
    SeeSawSignal,
    StopType,
    TrackedPosition,
)
from strategies.storm_lifeboat.core import Asset, DEFAULT_PRICES, MoonPhase, VolRegime


# ═══════════════════════════════════════════════════════════════════════════
# TrackedPosition — stop-loss mechanics
# ═══════════════════════════════════════════════════════════════════════════

class TestTrackedPosition:
    def _make_pos(self, entry_price=100.0, **kwargs) -> TrackedPosition:
        defaults = dict(
            asset=Asset.GOLD, etf="GLD", entry_price=entry_price,
            current_price=entry_price, quantity=10_000.0,
            entry_time=datetime.now(), high_water_mark=entry_price,
        )
        defaults.update(kwargs)
        return TrackedPosition(**defaults)

    def test_update_price_tracks_hwm(self):
        pos = self._make_pos(entry_price=100.0)
        pos.update_price(110.0)
        assert pos.high_water_mark == 110.0
        assert pos.pnl_pct == pytest.approx(0.10, abs=0.001)

    def test_update_price_hwm_not_lowered(self):
        pos = self._make_pos(entry_price=100.0)
        pos.update_price(120.0)
        pos.update_price(115.0)
        assert pos.high_water_mark == 120.0

    def test_fixed_stop_triggers(self):
        pos = self._make_pos(entry_price=100.0, fixed_stop_pct=0.08)
        pos.update_price(91.0)  # -9%
        triggered = pos.check_stop_loss()
        assert triggered
        assert pos.stopped_out
        assert "FIXED_STOP" in pos.stop_trigger

    def test_fixed_stop_not_triggered_within_threshold(self):
        pos = self._make_pos(entry_price=100.0, fixed_stop_pct=0.08, trailing_stop_pct=0.10)
        pos.update_price(93.0)  # -7%, within 8% fixed and 10% trailing
        assert not pos.check_stop_loss()
        assert not pos.stopped_out

    def test_trailing_stop_triggers(self):
        pos = self._make_pos(entry_price=100.0, trailing_stop_pct=0.05)
        pos.update_price(120.0)  # New HWM
        pos.update_price(113.0)  # -5.83% from HWM of 120
        triggered = pos.check_stop_loss()
        assert triggered
        assert pos.stopped_out
        assert "TRAILING_STOP" in pos.stop_trigger

    def test_trailing_stop_not_triggered_within_threshold(self):
        pos = self._make_pos(entry_price=100.0, trailing_stop_pct=0.05)
        pos.update_price(120.0)
        pos.update_price(115.0)  # -4.17% from HWM, within threshold
        assert not pos.check_stop_loss()

    def test_already_stopped_stays_stopped(self):
        pos = self._make_pos(entry_price=100.0, fixed_stop_pct=0.05)
        pos.update_price(90.0)
        pos.check_stop_loss()
        assert pos.stopped_out
        # Even if price recovers, still stopped
        pos.update_price(105.0)
        assert pos.check_stop_loss()

    def test_zero_entry_price_no_crash(self):
        pos = self._make_pos(entry_price=0.0)
        pos.update_price(10.0)
        assert not pos.check_stop_loss()


# ═══════════════════════════════════════════════════════════════════════════
# SeeSawPhase detection
# ═══════════════════════════════════════════════════════════════════════════

class TestPhaseDetection:
    def _make_engine(self) -> LifeboatCapitalEngine:
        return LifeboatCapitalEngine(starting_capital=50_000.0, data_dir="data/test_ce")

    def test_neutral_phase_default(self):
        eng = self._make_engine()
        # VIX < 20 and no active scenarios → RECOVERY
        phase = eng._detect_phase(dict(DEFAULT_PRICES), vix=15.0, heatmap=[])
        assert phase == SeeSawPhase.RECOVERY

    def test_neutral_on_moderate_vix(self):
        eng = self._make_engine()
        # VIX 22, no strong moves, normal g/s ratio, no scenarios → NEUTRAL
        eng._prev_oil_price = DEFAULT_PRICES[Asset.OIL]
        eng._prev_gold_price = DEFAULT_PRICES[Asset.GOLD]
        prices = dict(DEFAULT_PRICES)
        prices[Asset.SILVER] = 60.0  # G/S ratio = 4861/60 = ~81 (above 65)
        phase = eng._detect_phase(prices, vix=22.0, heatmap=[])
        assert phase == SeeSawPhase.NEUTRAL

    def test_oil_spike_detected(self):
        eng = self._make_engine()
        eng._prev_oil_price = 100.0
        prices = dict(DEFAULT_PRICES)
        prices[Asset.OIL] = 106.0  # +6%
        phase = eng._detect_phase(prices, vix=30.0, heatmap=[])
        assert phase == SeeSawPhase.OIL_SPIKE

    def test_recovery_on_low_vix_no_scenarios(self):
        eng = self._make_engine()
        eng._prev_oil_price = 100.0
        eng._prev_gold_price = 4800.0
        prices = dict(DEFAULT_PRICES)
        prices[Asset.OIL] = 101.0  # +1%
        phase = eng._detect_phase(prices, vix=15.0, heatmap=[])
        assert phase == SeeSawPhase.RECOVERY

    def test_gold_breakout_on_gold_surge(self):
        eng = self._make_engine()
        eng._prev_gold_price = 4700.0
        prices = dict(DEFAULT_PRICES)
        prices[Asset.GOLD] = 4900.0  # +4.3%
        phase = eng._detect_phase(prices, vix=22.0, heatmap=[])
        assert phase == SeeSawPhase.GOLD_BREAKOUT

    def test_silver_amplifier_on_low_ratio(self):
        eng = self._make_engine()
        eng._prev_oil_price = 100.0
        eng._prev_gold_price = 4800.0
        prices = dict(DEFAULT_PRICES)
        prices[Asset.GOLD] = 4800.0
        prices[Asset.SILVER] = 80.0  # ratio = 60
        phase = eng._detect_phase(prices, vix=22.0, heatmap=[])
        assert phase == SeeSawPhase.SILVER_AMPLIFIER


# ═══════════════════════════════════════════════════════════════════════════
# Signal generation
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalGeneration:
    def _make_engine(self) -> LifeboatCapitalEngine:
        return LifeboatCapitalEngine(starting_capital=50_000.0, data_dir="data/test_ce")

    def _mock_lunar(self):
        m = MagicMock()
        m.position_multiplier = 1.0
        m.phase = MoonPhase.WAXING
        m.moon_name = "Magnetic"
        return m

    def test_oil_spike_generates_long_and_short(self):
        eng = self._make_engine()
        signals = eng._generate_signals(
            SeeSawPhase.OIL_SPIKE, dict(DEFAULT_PRICES),
            self._mock_lunar(), 0.7, [],
        )
        dirs = {s.direction for s in signals}
        assert "LONG" in dirs
        assert "SHORT" in dirs
        # Should long energy/gold/silver
        long_etfs = {s.etf for s in signals if s.direction == "LONG"}
        assert "XLE" in long_etfs
        assert "GLD" in long_etfs
        assert "SLV" in long_etfs

    def test_inflation_rotation_heavy_on_gold_silver(self):
        eng = self._make_engine()
        signals = eng._generate_signals(
            SeeSawPhase.INFLATION_ROTATION, dict(DEFAULT_PRICES),
            self._mock_lunar(), 0.8, [],
        )
        etfs = {s.etf for s in signals}
        assert "GLD" in etfs
        assert "SLV" in etfs
        assert "GDX" in etfs

    def test_neutral_minimal_allocation(self):
        eng = self._make_engine()
        signals = eng._generate_signals(
            SeeSawPhase.NEUTRAL, dict(DEFAULT_PRICES),
            self._mock_lunar(), 0.5, [],
        )
        assert len(signals) >= 1
        total_weight = sum(s.weight for s in signals)
        assert total_weight < 0.50  # Mostly cash in neutral

    def test_new_moon_reduces_confidence(self):
        eng = self._make_engine()
        lunar = self._mock_lunar()
        lunar.phase = MoonPhase.NEW
        signals = eng._generate_signals(
            SeeSawPhase.OIL_SPIKE, dict(DEFAULT_PRICES),
            lunar, 0.8, [],
        )
        for s in signals:
            assert s.confidence < 0.45  # 0.5x multiplier applied
            assert "NEW MOON" in s.reason

    def test_waning_trims_long_weights(self):
        eng = self._make_engine()
        lunar = self._mock_lunar()
        lunar.phase = MoonPhase.WANING
        signals_wax = eng._generate_signals(
            SeeSawPhase.OIL_SPIKE, dict(DEFAULT_PRICES),
            self._mock_lunar(), 0.8, [],
        )
        signals_wane = eng._generate_signals(
            SeeSawPhase.OIL_SPIKE, dict(DEFAULT_PRICES),
            lunar, 0.8, [],
        )
        for sw, sn in zip(
            [s for s in signals_wax if s.direction == "LONG"],
            [s for s in signals_wane if s.direction == "LONG"],
        ):
            assert sn.weight <= sw.weight

    def test_all_signals_have_option_info(self):
        eng = self._make_engine()
        signals = eng._generate_signals(
            SeeSawPhase.OIL_SPIKE, dict(DEFAULT_PRICES),
            self._mock_lunar(), 0.8, [],
        )
        for s in signals:
            assert s.option_type != ""
            assert s.strike_hint != ""


# ═══════════════════════════════════════════════════════════════════════════
# Position management & stop-loss in engine context
# ═══════════════════════════════════════════════════════════════════════════

class TestPositionManagement:
    def _make_engine(self) -> LifeboatCapitalEngine:
        return LifeboatCapitalEngine(starting_capital=50_000.0, data_dir="data/test_ce")

    def test_stop_loss_returns_capital_to_cash(self):
        eng = self._make_engine()
        eng.cash = 40_000.0
        eng.positions[Asset.GOLD] = TrackedPosition(
            asset=Asset.GOLD, etf="GLD", entry_price=100.0,
            current_price=100.0, quantity=10_000.0,
            entry_time=datetime.now(), high_water_mark=100.0,
            fixed_stop_pct=0.05,
        )
        prices = {Asset.GOLD: 94.0}  # -6% → triggers 5% stop
        triggered = eng._update_positions(prices)
        assert len(triggered) == 1
        assert "FIXED_STOP" in triggered[0]
        assert Asset.GOLD not in eng.positions
        assert eng.cash > 40_000.0  # Got capital back (minus loss)

    def test_apply_signals_opens_positions(self):
        eng = self._make_engine()
        eng.portfolio_value = eng.cash
        signals = [
            SeeSawSignal(
                asset=Asset.GOLD, etf="GLD", direction="LONG",
                weight=0.20, confidence=0.80, reason="Test",
            ),
        ]
        prices = {Asset.GOLD: 200.0}
        eng._apply_signals(signals, prices)
        assert Asset.GOLD in eng.positions
        assert eng.positions[Asset.GOLD].entry_price == 200.0
        assert eng.cash < eng.starting_capital

    def test_apply_signals_skips_low_confidence(self):
        eng = self._make_engine()
        eng.portfolio_value = eng.cash
        signals = [
            SeeSawSignal(
                asset=Asset.GOLD, etf="GLD", direction="LONG",
                weight=0.20, confidence=0.30, reason="Low conf",
            ),
        ]
        eng._apply_signals(signals, {Asset.GOLD: 200.0})
        assert Asset.GOLD not in eng.positions  # Skipped

    def test_apply_close_signal_removes_position(self):
        eng = self._make_engine()
        eng.cash = 40_000.0
        eng.positions[Asset.SILVER] = TrackedPosition(
            asset=Asset.SILVER, etf="SLV", entry_price=30.0,
            current_price=32.0, quantity=10_000.0,
            entry_time=datetime.now(), high_water_mark=32.0,
        )
        eng.positions[Asset.SILVER].update_price(32.0)
        signals = [
            SeeSawSignal(
                asset=Asset.SILVER, etf="SLV", direction="CLOSE",
                weight=0.0, confidence=0.90, reason="Close out",
            ),
        ]
        eng._apply_signals(signals, {Asset.SILVER: 32.0})
        assert Asset.SILVER not in eng.positions


# ═══════════════════════════════════════════════════════════════════════════
# Full cycle (mocked data sources)
# ═══════════════════════════════════════════════════════════════════════════

class TestFullCycle:
    @pytest.fixture
    def engine(self, tmp_path):
        return LifeboatCapitalEngine(
            starting_capital=48_100.0,
            data_dir=str(tmp_path / "ce_data"),
        )

    def _mock_snapshot(self):
        """Build a mock LiveFeedSnapshot."""
        snap = MagicMock()
        snap.prices = dict(DEFAULT_PRICES)
        snap.vix = 28.0
        snap.regime = VolRegime.CRISIS
        snap.sources_ok = ["polygon", "fred"]
        snap.sources_failed = []
        snap.firing_indicators = {}
        return snap

    def _mock_lunar_pos(self):
        pos = MagicMock()
        pos.phase = MoonPhase.WAXING
        pos.moon_number = 3
        pos.moon_name = "Electric"
        pos.in_phi_window = False
        pos.phi_coherence = 0.618
        pos.position_multiplier = 1.0
        return pos

    def _mock_coherence(self):
        coh = MagicMock()
        coh.overall_score = 0.72
        return coh

    @pytest.mark.asyncio
    async def test_run_hourly_returns_cycle_report(self, engine):
        snap = self._mock_snapshot()
        lunar = self._mock_lunar_pos()
        coherence = self._mock_coherence()

        with patch.object(engine, '_scrape', new_callable=AsyncMock) as mock_scrape:
            mock_scrape.return_value = {
                "snapshot": snap,
                "lunar": lunar,
                "coherence": coherence,
                "active_scenarios": [],
                "heatmap": [],
            }
            report = await engine.run_hourly()

        assert isinstance(report, CycleReport)
        assert report.portfolio_value > 0
        assert report.gold_price > 0
        assert report.oil_price > 0
        assert report.silver_price > 0
        assert report.gold_oil_ratio > 0
        assert report.gold_silver_ratio > 0

    @pytest.mark.asyncio
    async def test_run_hourly_persists_report(self, engine, tmp_path):
        snap = self._mock_snapshot()
        lunar = self._mock_lunar_pos()
        coherence = self._mock_coherence()

        with patch.object(engine, '_scrape', new_callable=AsyncMock) as mock_scrape:
            mock_scrape.return_value = {
                "snapshot": snap,
                "lunar": lunar,
                "coherence": coherence,
                "active_scenarios": [],
                "heatmap": [],
            }
            await engine.run_hourly()

        report_path = tmp_path / "ce_data" / "capital_engine_reports.jsonl"
        assert report_path.exists()
        lines = report_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "phase" in record
        assert "portfolio" in record
        assert "signals" in record

    @pytest.mark.asyncio
    async def test_run_hourly_graceful_on_scrape_failure(self, engine):
        """Engine should produce a valid report even when scrape fails."""
        with patch.object(engine, '_scrape', new_callable=AsyncMock) as mock_scrape:
            mock_scrape.side_effect = RuntimeError("Network down")
            report = await engine.run_hourly()

        assert isinstance(report, CycleReport)
        assert report.portfolio_value == engine.starting_capital
        assert "all_sources" in report.sources_failed


# ═══════════════════════════════════════════════════════════════════════════
# Status / health
# ═══════════════════════════════════════════════════════════════════════════

class TestStatus:
    def test_get_status_contains_required_keys(self):
        eng = LifeboatCapitalEngine(starting_capital=50_000.0, data_dir="data/test_ce")
        status = eng.get_status()
        assert status["engine"] == "LifeboatCapitalEngine"
        assert status["phase"] == "neutral"
        assert status["portfolio_value"] == 50_000.0
        assert "positions" in status
        assert "cycles_run" in status

    def test_shutdown_sets_flag(self):
        eng = LifeboatCapitalEngine(data_dir="data/test_ce")
        assert not eng._shutdown
        eng.shutdown()
        assert eng._shutdown


# ═══════════════════════════════════════════════════════════════════════════
# Default weights validation
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigConstants:
    def test_all_phases_have_weights(self):
        for phase in ["oil_spike", "inflation_rotation", "recovery", "neutral"]:
            assert phase in DEFAULT_WEIGHTS
            total = sum(DEFAULT_WEIGHTS[phase].values())
            assert 0.99 <= total <= 1.01, f"Weights for {phase} sum to {total}"

    def test_seesaw_phases_defined(self):
        assert len(SeeSawPhase) >= 5
        values = {p.value for p in SeeSawPhase}
        assert "neutral" in values
        assert "oil_spike" in values
        assert "inflation_rotation" in values
