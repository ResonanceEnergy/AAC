"""Tests for strategies.rocket_ship.lunar_cycles — Moon cycle timing engine."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from strategies.rocket_ship.core import (
    LIFEBOAT_INCEPTION,
    MoonPhase,
    SystemPhase,
)
from strategies.rocket_ship.lunar_cycles import RocketLunarEngine


@pytest.fixture
def engine() -> RocketLunarEngine:
    return RocketLunarEngine()


class TestRocketLunarEngine:
    """Core timing logic tests."""

    def test_inception_day_is_moon_1(self, engine: RocketLunarEngine) -> None:
        state = engine.get_current_state(LIFEBOAT_INCEPTION)
        assert state.moon_number == 1
        assert state.day_in_moon == 1
        assert state.moon_phase == MoonPhase.NEW
        assert state.system_phase == SystemPhase.LIFE_BOAT

    def test_mid_moon_1(self, engine: RocketLunarEngine) -> None:
        # Day 15 (full phase, near phi window 2)
        ref = date(2026, 4, 5)  # 14 days after inception
        state = engine.get_current_state(ref)
        assert state.moon_number == 1
        assert state.day_in_moon == 15
        assert state.moon_phase == MoonPhase.FULL

    def test_phi_window_1_active(self, engine: RocketLunarEngine) -> None:
        # Day 10 of moon 1 = inception + 9 days
        ref = date(2026, 3, 31)
        state = engine.get_current_state(ref)
        assert state.day_in_moon == 10
        assert state.in_phi_window_1 is True
        assert state.in_phi_window_2 is False

    def test_phi_window_2_active(self, engine: RocketLunarEngine) -> None:
        # Day 17 of moon 1 = inception + 16 days
        ref = date(2026, 4, 7)
        state = engine.get_current_state(ref)
        assert state.day_in_moon == 17
        assert state.in_phi_window_2 is True
        assert state.in_phi_window_1 is False

    def test_moon_2_transition(self, engine: RocketLunarEngine) -> None:
        # ~30 days after inception should be moon 2
        ref = date(2026, 4, 21)  # 30 days after 2026-03-22
        state = engine.get_current_state(ref)
        assert state.moon_number == 2
        assert state.moon_name == "Fortification"

    def test_days_to_rocket_start_decreases(self, engine: RocketLunarEngine) -> None:
        s1 = engine.get_current_state(date(2026, 4, 1))
        s2 = engine.get_current_state(date(2026, 5, 1))
        assert s1.days_to_rocket_start > s2.days_to_rocket_start
        assert s1.days_to_rocket_start > 0
        assert s2.days_to_rocket_start > 0

    def test_rocket_phase_moon_13(self, engine: RocketLunarEngine) -> None:
        # Moon 13 start ≈ inception + 12 * 29.53 days ≈ 354 days
        rocket_start = engine._moon_start_date(13)
        state = engine.get_current_state(rocket_start)
        assert state.moon_number == 13
        assert state.is_rocket_phase is True
        assert state.days_to_rocket_start == 0
        assert state.system_phase == SystemPhase.ROCKET
        assert state.moon_name == "Ignition"

    def test_orbit_phase_moon_40(self, engine: RocketLunarEngine) -> None:
        orbit_start = engine._moon_start_date(40)
        state = engine.get_current_state(orbit_start)
        assert state.moon_number == 40
        assert state.system_phase == SystemPhase.ORBIT
        assert state.moon_name == "Orbit Alpha"

    def test_pre_inception_clamps_to_moon_1(self, engine: RocketLunarEngine) -> None:
        state = engine.get_current_state(date(2025, 1, 1))
        assert state.moon_number == 1

    def test_milestone_present_on_moon_1(self, engine: RocketLunarEngine) -> None:
        state = engine.get_current_state(LIFEBOAT_INCEPTION)
        assert state.milestone is not None
        assert "inception" in state.milestone.lower()

    def test_milestone_none_on_non_milestone_moon(self, engine: RocketLunarEngine) -> None:
        # Moon 3 has no milestone — use a date safely inside moon 3
        moon3_start = engine._moon_start_date(3)
        state = engine.get_current_state(moon3_start + timedelta(days=5))
        assert state.moon_number == 3
        assert state.milestone is None

    def test_new_moon_date_matches_start(self, engine: RocketLunarEngine) -> None:
        state = engine.get_current_state(LIFEBOAT_INCEPTION)
        assert state.new_moon_date == LIFEBOAT_INCEPTION

    def test_days_in_moon_is_positive(self, engine: RocketLunarEngine) -> None:
        state = engine.get_current_state(LIFEBOAT_INCEPTION)
        assert state.days_in_this_moon >= 29  # synodic month ≈ 29.53


class TestFormatDashboard:
    """Dashboard formatting tests."""

    def test_format_returns_string(self, engine: RocketLunarEngine) -> None:
        result = engine.format_dashboard()
        assert isinstance(result, str)
        assert len(result) > 50

    def test_format_contains_phase(self, engine: RocketLunarEngine) -> None:
        result = engine.format_dashboard(date(2026, 3, 25))
        assert "LIFE BOAT" in result

    def test_format_contains_progress(self, engine: RocketLunarEngine) -> None:
        result = engine.format_dashboard()
        assert "Progress:" in result

    def test_format_rocket_phase_shows_active(self, engine: RocketLunarEngine) -> None:
        rocket_start = engine._moon_start_date(13)
        result = engine.format_dashboard(rocket_start)
        assert "ACTIVE" in result
        assert "ROCKET" in result
