"""
Storm Lifeboat Matrix — 13-Moon Phi Cycle Engine
==================================================
Tracks the 13-moon cycle (13 × 28 = 364 days) with golden ratio (phi)
weighting for position timing and risk modulation.

The 13-moon calendar:
    Each synodic month ≈ 29.53 days, but the 13-moon system uses a
    regularized 28-day cycle (13 × 28 = 364, +1 "Day Out of Time").

Phi integration:
    Within each 28-day moon, peak action windows are placed at
    phi-proportional days (days 10-11 and 17-18 out of 28),
    corresponding to the golden ratio division of the cycle.

    28 / phi ≈ 17.3 (first cut)
    28 / phi^2 ≈ 10.7 (second cut)

    These "phi windows" are peak coherence times for entries/exits.

Moon phases and trading mandates:
    NEW (days 1-7):     Reset, reassess — close losing positions, review thesis
    WAXING (days 8-14): Accumulate — scale into positions during phi window (day 10-11)
    FULL (days 15-21):  Execute — max conviction entries at phi window (day 17-18)
    WANING (days 22-28): Hedge — trim, roll, protect gains

The engine computes:
    - Current moon number (1-13) and day within moon (1-28)
    - Current phase (NEW/WAXING/FULL/WANING)
    - Whether we're in a phi window
    - Position sizing multiplier based on phase + phi alignment
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

from strategies.storm_lifeboat.core import MoonPhase

logger = logging.getLogger(__name__)

PHI = (1 + math.sqrt(5)) / 2  # 1.6180339887...
MOON_LENGTH = 28               # Days per moon
TOTAL_MOONS = 13               # Moons per cycle
CYCLE_LENGTH = MOON_LENGTH * TOTAL_MOONS  # 364 days

# Phi-proportional windows within a 28-day moon
PHI_WINDOW_1 = (10, 11)  # 28 / phi^2 ≈ 10.7
PHI_WINDOW_2 = (17, 18)  # 28 / phi ≈ 17.3

# Moon names (resonance/indigenous tradition inspired)
MOON_NAMES = [
    "Magnetic",      # Moon 1: Set intention
    "Lunar",         # Moon 2: Polarise, challenge
    "Electric",      # Moon 3: Activate, bond
    "Self-Existing", # Moon 4: Define, measure
    "Overtone",      # Moon 5: Empower, command
    "Rhythmic",      # Moon 6: Organize, balance
    "Resonant",      # Moon 7: Channel, inspire (midpoint)
    "Galactic",      # Moon 8: Harmonize, model
    "Solar",         # Moon 9: Pulse, realize
    "Planetary",     # Moon 10: Perfect, produce
    "Spectral",      # Moon 11: Dissolve, release
    "Crystal",       # Moon 12: Dedicate, cooperate
    "Cosmic",        # Moon 13: Endure, transcend
]


@dataclass
class LunarPosition:
    """Current position within the 13-moon cycle."""
    cycle_start: date          # When this 364-day cycle began
    moon_number: int           # 1-13
    moon_name: str             # e.g. "Resonant"
    day_in_moon: int           # 1-28
    day_in_cycle: int          # 1-364
    phase: MoonPhase           # NEW / WAXING / FULL / WANING
    in_phi_window: bool        # True if in a golden-ratio timing window
    phi_coherence: float       # 0-1, how close to exact phi proportion
    position_multiplier: float # 0.5-1.5, sizing weight based on phase + phi


def _day_phase(day: int) -> MoonPhase:
    """Determine moon phase from day within moon (1-28)."""
    if day <= 7:
        return MoonPhase.NEW
    if day <= 14:
        return MoonPhase.WAXING
    if day <= 21:
        return MoonPhase.FULL
    return MoonPhase.WANING


def _phi_coherence(day: int) -> float:
    """Compute phi coherence (0-1) based on proximity to phi windows.

    Maximum coherence (1.0) at exact phi days (10.7, 17.3).
    Coherence decays as a cosine curve away from those points.
    """
    phi_day_1 = MOON_LENGTH / (PHI ** 2)  # ~10.69
    phi_day_2 = MOON_LENGTH / PHI          # ~17.31

    dist1 = abs(day - phi_day_1)
    dist2 = abs(day - phi_day_2)
    min_dist = min(dist1, dist2)

    # Coherence from 1.0 (at phi day) to 0.0 (at 7+ days away)
    coherence = max(0.0, math.cos(min_dist * math.pi / 14))
    return round(coherence, 4)


def _position_multiplier(phase: MoonPhase, in_phi: bool, coherence: float) -> float:
    """Compute position sizing multiplier.

    Base multipliers by phase:
        NEW:    0.5 (reduce exposure)
        WAXING: 0.8 (building)
        FULL:   1.2 (max conviction)
        WANING: 0.7 (protecting)

    Phi window bonus: +0.3 (capped at 1.5)
    """
    base = {
        MoonPhase.NEW: 0.50,
        MoonPhase.WAXING: 0.80,
        MoonPhase.FULL: 1.20,
        MoonPhase.WANING: 0.70,
    }[phase]

    if in_phi:
        base += 0.30 * coherence

    return round(min(1.50, max(0.30, base)), 3)


class LunarPhiEngine:
    """13-moon phi-cycle tracking engine.

    Anchors the 364-day cycle to a configurable start date.
    Default anchor: March 20, 2026 (spring equinox — natural year start).
    """

    def __init__(self, cycle_start: Optional[date] = None) -> None:
        self.cycle_start = cycle_start or date(2026, 3, 20)

    def get_position(self, target: Optional[date] = None) -> LunarPosition:
        """Compute the 13-moon position for a given date.

        Args:
            target: Date to compute for (defaults to today)

        Returns:
            LunarPosition with full cycle coordinates
        """
        target = target or date.today()

        # Compute day within the current or most recent cycle
        delta = (target - self.cycle_start).days
        if delta < 0:
            # Before cycle start — walk back to find the previous cycle start
            cycles_back = (-delta // CYCLE_LENGTH) + 1
            effective_start = self.cycle_start - timedelta(days=cycles_back * CYCLE_LENGTH)
            delta = (target - effective_start).days
        else:
            # Wrap into current cycle
            delta = delta % CYCLE_LENGTH

        day_in_cycle = delta + 1  # 1-indexed
        moon_number = (delta // MOON_LENGTH) + 1  # 1-13
        day_in_moon = (delta % MOON_LENGTH) + 1    # 1-28

        # Clamp moon_number to 1-13
        moon_number = min(moon_number, TOTAL_MOONS)
        moon_name = MOON_NAMES[moon_number - 1]

        phase = _day_phase(day_in_moon)
        in_phi = (PHI_WINDOW_1[0] <= day_in_moon <= PHI_WINDOW_1[1] or
                  PHI_WINDOW_2[0] <= day_in_moon <= PHI_WINDOW_2[1])
        coherence = _phi_coherence(day_in_moon)
        multiplier = _position_multiplier(phase, in_phi, coherence)

        return LunarPosition(
            cycle_start=self.cycle_start,
            moon_number=moon_number,
            moon_name=moon_name,
            day_in_moon=day_in_moon,
            day_in_cycle=day_in_cycle,
            phase=phase,
            in_phi_window=in_phi,
            phi_coherence=coherence,
            position_multiplier=multiplier,
        )

    def get_next_phi_window(self, target: Optional[date] = None) -> date:
        """Find the next phi window date from the given date."""
        target = target or date.today()
        pos = self.get_position(target)

        day = pos.day_in_moon
        if day < PHI_WINDOW_1[0]:
            days_ahead = PHI_WINDOW_1[0] - day
        elif day < PHI_WINDOW_2[0]:
            days_ahead = PHI_WINDOW_2[0] - day
        else:
            # Next moon's first phi window
            days_ahead = (MOON_LENGTH - day) + PHI_WINDOW_1[0]

        return target + timedelta(days=days_ahead)

    def format_display(self, pos: Optional[LunarPosition] = None) -> str:
        """Format a human-readable lunar position string."""
        pos = pos or self.get_position()
        phi_marker = " ** PHI WINDOW **" if pos.in_phi_window else ""
        return (
            f"Moon {pos.moon_number}/13 ({pos.moon_name}) | "
            f"Day {pos.day_in_moon}/28 | Phase: {pos.phase.value.upper()} | "
            f"Phi: {pos.phi_coherence:.2f} | Size: {pos.position_multiplier:.2f}x"
            f"{phi_marker}"
        )
