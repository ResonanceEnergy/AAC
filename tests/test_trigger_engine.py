"""Tests for strategies.rocket_ship.trigger_engine — Ignition trigger logic."""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from strategies.rocket_ship.core import (
    INDICATORS_REQUIRED_FOR_IGNITION,
    LIFEBOAT_INCEPTION,
    SystemPhase,
    TriggerStatus,
)
from strategies.rocket_ship.trigger_engine import TriggerEngine, TriggerReport


@pytest.fixture
def engine() -> TriggerEngine:
    return TriggerEngine()


# ═══════════════════════════════════════════════════════════════════════════
# Phase determination
# ═══════════════════════════════════════════════════════════════════════════

class TestDeterminePhase:
    """_determine_phase static method tests."""

    def test_life_boat_when_not_ignited(self) -> None:
        phase = TriggerEngine._determine_phase(
            ignited=False, moon_number=5, gulf_status=TriggerStatus.WATCHING,
        )
        assert phase == SystemPhase.LIFE_BOAT

    def test_igniting_when_ignited(self) -> None:
        phase = TriggerEngine._determine_phase(
            ignited=True, moon_number=5, gulf_status=TriggerStatus.WATCHING,
        )
        assert phase == SystemPhase.IGNITING

    def test_igniting_when_gulf_confirmed(self) -> None:
        phase = TriggerEngine._determine_phase(
            ignited=False, moon_number=5, gulf_status=TriggerStatus.CONFIRMED,
        )
        assert phase == SystemPhase.IGNITING

    def test_rocket_phase_at_moon_13(self) -> None:
        phase = TriggerEngine._determine_phase(
            ignited=False, moon_number=13, gulf_status=TriggerStatus.WATCHING,
        )
        assert phase == SystemPhase.ROCKET

    def test_rocket_at_moon_39(self) -> None:
        phase = TriggerEngine._determine_phase(
            ignited=False, moon_number=39, gulf_status=TriggerStatus.WATCHING,
        )
        assert phase == SystemPhase.ROCKET

    def test_orbit_at_moon_40(self) -> None:
        phase = TriggerEngine._determine_phase(
            ignited=False, moon_number=40, gulf_status=TriggerStatus.WATCHING,
        )
        assert phase == SystemPhase.ORBIT

    def test_orbit_overrides_ignited(self) -> None:
        # Moon >= 40 always returns ORBIT, regardless of ignition
        phase = TriggerEngine._determine_phase(
            ignited=True, moon_number=42, gulf_status=TriggerStatus.CONFIRMED,
        )
        assert phase == SystemPhase.ORBIT


# ═══════════════════════════════════════════════════════════════════════════
# Alert levels
# ═══════════════════════════════════════════════════════════════════════════

class TestAlertLevel:
    """_compute_alert_level static method tests."""

    def test_low_when_few_green(self) -> None:
        level = TriggerEngine._compute_alert_level(
            green_count=5, gulf_status=TriggerStatus.WATCHING, ignited=False,
        )
        assert level == "LOW"

    def test_elevated_at_threshold(self) -> None:
        level = TriggerEngine._compute_alert_level(
            green_count=INDICATORS_REQUIRED_FOR_IGNITION,
            gulf_status=TriggerStatus.WATCHING, ignited=False,
        )
        assert level == "ELEVATED"

    def test_high_when_12_green(self) -> None:
        level = TriggerEngine._compute_alert_level(
            green_count=12, gulf_status=TriggerStatus.WATCHING, ignited=False,
        )
        assert level == "HIGH"

    def test_high_when_emerging(self) -> None:
        level = TriggerEngine._compute_alert_level(
            green_count=5, gulf_status=TriggerStatus.EMERGING, ignited=False,
        )
        assert level == "HIGH"

    def test_ignited_when_ignited(self) -> None:
        level = TriggerEngine._compute_alert_level(
            green_count=13, gulf_status=TriggerStatus.WATCHING, ignited=True,
        )
        assert level == "IGNITED"

    def test_ignited_when_gulf_confirmed(self) -> None:
        level = TriggerEngine._compute_alert_level(
            green_count=3, gulf_status=TriggerStatus.CONFIRMED, ignited=False,
        )
        assert level == "IGNITED"


# ═══════════════════════════════════════════════════════════════════════════
# Probability estimation
# ═══════════════════════════════════════════════════════════════════════════

class TestProbability:
    """_estimate_probability static method tests."""

    def test_confirmed_returns_1(self) -> None:
        prob = TriggerEngine._estimate_probability(
            green_count=5, gulf_status=TriggerStatus.CONFIRMED, days_to_default=300,
        )
        assert prob == 1.0

    def test_zero_green_low_prob(self) -> None:
        prob = TriggerEngine._estimate_probability(
            green_count=0, gulf_status=TriggerStatus.WATCHING, days_to_default=500,
        )
        assert prob == 0.0

    def test_emerging_boosts_prob(self) -> None:
        base = TriggerEngine._estimate_probability(
            green_count=8, gulf_status=TriggerStatus.WATCHING, days_to_default=300,
        )
        boosted = TriggerEngine._estimate_probability(
            green_count=8, gulf_status=TriggerStatus.EMERGING, days_to_default=300,
        )
        assert boosted > base

    def test_time_proximity_boost(self) -> None:
        far = TriggerEngine._estimate_probability(
            green_count=8, gulf_status=TriggerStatus.WATCHING, days_to_default=300,
        )
        near = TriggerEngine._estimate_probability(
            green_count=8, gulf_status=TriggerStatus.WATCHING, days_to_default=50,
        )
        assert near > far

    def test_capped_at_1(self) -> None:
        prob = TriggerEngine._estimate_probability(
            green_count=15, gulf_status=TriggerStatus.EMERGING, days_to_default=30,
        )
        assert prob <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Action recommendations
# ═══════════════════════════════════════════════════════════════════════════

class TestRecommendActions:
    """_recommend_actions static method tests."""

    def test_life_boat_has_maintain_action(self) -> None:
        actions = TriggerEngine._recommend_actions(
            SystemPhase.LIFE_BOAT, green_count=5,
            gulf_status=TriggerStatus.WATCHING, moon_number=3,
        )
        assert any("Life Boat" in a for a in actions)

    def test_life_boat_alert_when_high_green(self) -> None:
        actions = TriggerEngine._recommend_actions(
            SystemPhase.LIFE_BOAT, green_count=12,
            gulf_status=TriggerStatus.WATCHING, moon_number=3,
        )
        assert any("ALERT" in a for a in actions)

    def test_igniting_has_deployment_action(self) -> None:
        actions = TriggerEngine._recommend_actions(
            SystemPhase.IGNITING, green_count=13,
            gulf_status=TriggerStatus.WATCHING, moon_number=5,
        )
        assert any("deployment" in a.lower() for a in actions)
        assert len(actions) >= 3

    def test_rocket_references_moon(self) -> None:
        actions = TriggerEngine._recommend_actions(
            SystemPhase.ROCKET, green_count=13,
            gulf_status=TriggerStatus.WATCHING, moon_number=15,
        )
        assert any("Moon 15" in a for a in actions)

    def test_orbit_has_stable_action(self) -> None:
        actions = TriggerEngine._recommend_actions(
            SystemPhase.ORBIT, green_count=13,
            gulf_status=TriggerStatus.WATCHING, moon_number=42,
        )
        assert any("orbit" in a.lower() or "stable" in a.lower() for a in actions)


# ═══════════════════════════════════════════════════════════════════════════
# Full evaluate integration
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluate:
    """TriggerEngine.evaluate() integration tests."""

    def test_evaluate_returns_trigger_report(self, engine: TriggerEngine) -> None:
        report = engine.evaluate(LIFEBOAT_INCEPTION)
        assert isinstance(report, TriggerReport)

    def test_report_has_all_fields(self, engine: TriggerEngine) -> None:
        report = engine.evaluate(LIFEBOAT_INCEPTION)
        assert isinstance(report.phase, SystemPhase)
        assert report.alert_level in ("LOW", "ELEVATED", "HIGH", "IGNITED")
        assert isinstance(report.is_ignited, bool)
        assert 0.0 <= report.ignition_probability <= 1.0
        assert isinstance(report.green_count, int)
        assert isinstance(report.gulf_trigger_status, str)
        assert isinstance(report.days_to_default_ignition, int)
        assert isinstance(report.immediate_actions, list)
        assert isinstance(report.reason, str)

    def test_report_is_frozen(self, engine: TriggerEngine) -> None:
        report = engine.evaluate(LIFEBOAT_INCEPTION)
        with pytest.raises(AttributeError):
            report.phase = SystemPhase.ORBIT  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard formatting
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatDashboard:
    """format_dashboard() output tests."""

    def test_returns_string(self, engine: TriggerEngine) -> None:
        result = engine.format_dashboard(LIFEBOAT_INCEPTION)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_contains_phase(self, engine: TriggerEngine) -> None:
        result = engine.format_dashboard(LIFEBOAT_INCEPTION)
        # Should contain one of the phase labels
        assert any(
            label in result
            for label in ("LIFE BOAT", "IGNITING", "ROCKET", "ORBIT")
        )

    def test_contains_alert_level(self, engine: TriggerEngine) -> None:
        result = engine.format_dashboard(LIFEBOAT_INCEPTION)
        assert any(
            level in result
            for level in ("LOW", "ELEVATED", "HIGH", "IGNITED")
        )

    def test_contains_green_count(self, engine: TriggerEngine) -> None:
        result = engine.format_dashboard(LIFEBOAT_INCEPTION)
        assert "/15 GREEN" in result

    def test_contains_actions(self, engine: TriggerEngine) -> None:
        result = engine.format_dashboard(LIFEBOAT_INCEPTION)
        assert "Immediate Actions:" in result
