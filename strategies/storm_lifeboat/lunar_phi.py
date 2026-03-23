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
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

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

# ═══════════════════════════════════════════════════════════════════════════
# ASTRONOMICAL EVENT CALENDAR — Stamps for the 39-Moon Map
# ═══════════════════════════════════════════════════════════════════════════
# Stamp key:
#   *    Full Moon Fire Peak (high-volatility day within phi window)
#   **   Equinox/Solstice Amplifier (seasonal inflection)
#   ***  Major Eclipse (solar or lunar — max disruption signal)
#   S    Solar-Lunar Synergy Spike (new/full moon + solar cycle peak alignment)
#   G    Geomagnetic Storm Window (CME/storm watch from NOAA data)
#   P    Major Planetary Alignment (Jupiter-Saturn, etc.)
#   †    Analysis Cycle Review (mandated strategy reassessment point)

# Equinoxes and solstices 2026-2029 (fixed astronomical dates)
EQUINOX_SOLSTICE_DATES: List[date] = [
    date(2026, 3, 20),   # Vernal equinox 2026
    date(2026, 6, 21),   # Summer solstice 2026
    date(2026, 9, 22),   # Autumnal equinox 2026
    date(2026, 12, 21),  # Winter solstice 2026
    date(2027, 3, 20),   # Vernal equinox 2027
    date(2027, 6, 21),   # Summer solstice 2027
    date(2027, 9, 22),   # Autumnal equinox 2027
    date(2027, 12, 21),  # Winter solstice 2027
    date(2028, 3, 20),   # Vernal equinox 2028
    date(2028, 6, 20),   # Summer solstice 2028
    date(2028, 9, 22),   # Autumnal equinox 2028
    date(2028, 12, 21),  # Winter solstice 2028
    date(2029, 3, 20),   # Vernal equinox 2029
]

# Major eclipses 2026-2029 (solar + lunar, visible impact on markets)
ECLIPSE_DATES: List[date] = [
    date(2026, 3, 3),    # Moon 0 anchor — total lunar eclipse
    date(2026, 3, 29),   # Partial solar eclipse
    date(2026, 8, 12),   # Total solar eclipse
    date(2026, 8, 28),   # Partial lunar eclipse
    date(2027, 2, 6),    # Annular solar eclipse
    date(2027, 2, 20),   # Total lunar eclipse
    date(2027, 7, 18),   # Total lunar eclipse
    date(2027, 8, 2),    # Total solar eclipse
    date(2028, 1, 12),   # Partial solar eclipse
    date(2028, 1, 26),   # Total lunar eclipse
    date(2028, 7, 22),   # Total solar eclipse
    date(2028, 12, 31),  # Total lunar eclipse
]

# Solar cycle 25 peak window (2024-2026 peak, elevated geomagnetic activity)
# These mark the strongest G-stamp windows where CME probability is highest
SOLAR_PEAK_WINDOW = (date(2025, 6, 1), date(2026, 12, 31))

# Planetary alignments of note (Jupiter-Saturn, etc.)
PLANETARY_ALIGNMENTS: List[date] = [
    date(2026, 4, 18),   # Jupiter-Neptune conjunction approach
    date(2026, 8, 27),   # Saturn opposition
    date(2027, 2, 19),   # Jupiter-Uranus trine
    date(2027, 6, 14),   # Grand trine formation
    date(2028, 1, 8),    # Jupiter-Saturn square
]

# Analysis cycle review points (every 3rd moon = mandatory thesis reassessment)
REVIEW_MOON_INTERVAL = 3

# Moon 0 anchor: March 3, 2026 total lunar eclipse
MOON_ZERO_ANCHOR = date(2026, 3, 3)


@dataclass
class AstroStamp:
    """An astronomical event stamp for a specific date."""
    date: date
    code: str       # *, **, ***, S, G, P, †
    label: str      # Human-readable description
    volatility_multiplier: float = 1.0  # Additional sizing modifier


@dataclass
class MoonMapEntry:
    """One moon in the 39-moon extended map."""
    moon_number: int        # 0-38 (Moon 0 = anchor eclipse)
    start_date: date
    end_date: date
    moon_name: str          # Cycles through the 13 moon names
    stamps: List[AstroStamp] = field(default_factory=list)
    is_review: bool = False  # True every REVIEW_MOON_INTERVAL moons


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

    # ─── 39-Moon Map with Astronomical Stamps ─────────────────────────

    def build_moon_map(self, num_moons: int = 39) -> List[MoonMapEntry]:
        """Generate the full multi-cycle moon map with astronomical stamps.

        Extends from Moon 0 (anchor eclipse, March 3, 2026) through 39 moons.
        Each moon gets stamped with any astronomical events that fall within it.

        Args:
            num_moons: Number of moons to map (default 39 = ~3 years).

        Returns:
            List of MoonMapEntry objects with stamps attached.
        """
        entries: List[MoonMapEntry] = []

        for i in range(num_moons):
            start = MOON_ZERO_ANCHOR + timedelta(days=i * MOON_LENGTH)
            end = start + timedelta(days=MOON_LENGTH - 1)
            moon_name = MOON_NAMES[i % TOTAL_MOONS]
            is_review = (i > 0) and (i % REVIEW_MOON_INTERVAL == 0)

            stamps = self._compute_stamps(start, end, i, is_review)

            entries.append(MoonMapEntry(
                moon_number=i,
                start_date=start,
                end_date=end,
                moon_name=moon_name,
                stamps=stamps,
                is_review=is_review,
            ))

        return entries

    def _compute_stamps(
        self, start: date, end: date, moon_num: int, is_review: bool,
    ) -> List[AstroStamp]:
        """Compute all astronomical stamps for a given moon window."""
        stamps: List[AstroStamp] = []

        # *** Eclipse stamps
        for d in ECLIPSE_DATES:
            if start <= d <= end:
                stamps.append(AstroStamp(
                    date=d, code="***",
                    label=f"Eclipse ({d.isoformat()})",
                    volatility_multiplier=1.40,
                ))

        # ** Equinox/Solstice stamps
        for d in EQUINOX_SOLSTICE_DATES:
            if start <= d <= end:
                stamps.append(AstroStamp(
                    date=d, code="**",
                    label=f"Equinox/Solstice ({d.isoformat()})",
                    volatility_multiplier=1.25,
                ))

        # P Planetary alignment stamps
        for d in PLANETARY_ALIGNMENTS:
            if start <= d <= end:
                stamps.append(AstroStamp(
                    date=d, code="P",
                    label=f"Planetary alignment ({d.isoformat()})",
                    volatility_multiplier=1.15,
                ))

        # G Geomagnetic storm window — during solar cycle peak
        if SOLAR_PEAK_WINDOW[0] <= start <= SOLAR_PEAK_WINDOW[1]:
            # Phi windows during solar peak get G stamp
            phi_day_1_date = start + timedelta(days=PHI_WINDOW_1[0] - 1)
            if start <= phi_day_1_date <= end:
                stamps.append(AstroStamp(
                    date=phi_day_1_date, code="G",
                    label=f"Geomagnetic storm window (solar peak + phi)",
                    volatility_multiplier=1.20,
                ))

        # S Solar-Lunar synergy — new/full moon during equinox/solstice month
        moon_mid = start + timedelta(days=14)
        for eq_date in EQUINOX_SOLSTICE_DATES:
            if abs((moon_mid - eq_date).days) <= 14:
                # New moon (day 1) and full moon (day 15) get synergy stamps
                stamps.append(AstroStamp(
                    date=start, code="S",
                    label=f"Solar-Lunar synergy (near {eq_date.isoformat()})",
                    volatility_multiplier=1.30,
                ))
                break

        # * Full Moon Fire Peak — day 15+phi window overlap
        full_moon_date = start + timedelta(days=14)  # Day 15
        if not any(s.code == "***" for s in stamps):  # Don't double-stamp eclipses
            stamps.append(AstroStamp(
                date=full_moon_date, code="*",
                label="Full Moon Fire Peak",
                volatility_multiplier=1.10,
            ))

        # † Analysis Cycle Review
        if is_review:
            stamps.append(AstroStamp(
                date=start, code="†",
                label=f"Analysis Cycle Review (Moon {moon_num})",
                volatility_multiplier=1.0,
            ))

        return stamps

    def get_current_stamps(self, target: Optional[date] = None) -> List[AstroStamp]:
        """Get astronomical stamps active for the current moon."""
        target = target or date.today()
        moon_map = self.build_moon_map()
        for entry in moon_map:
            if entry.start_date <= target <= entry.end_date:
                return entry.stamps
        return []

    def get_astro_multiplier(self, target: Optional[date] = None) -> float:
        """Get the combined astronomical volatility multiplier for today.

        Multiplies all active stamp multipliers together.
        """
        stamps = self.get_current_stamps(target)
        multiplier = 1.0
        for s in stamps:
            multiplier *= s.volatility_multiplier
        return round(min(2.0, multiplier), 3)  # Cap at 2x

    def format_moon_map(self, num_moons: int = 39) -> str:
        """Format the full moon map as a readable table."""
        entries = self.build_moon_map(num_moons)
        lines = [
            "═══ STORM LIFEBOAT 39-MOON MAP ═══",
            f"Anchor: Moon 0 = {MOON_ZERO_ANCHOR.isoformat()} (Total Lunar Eclipse)",
            f"Stamp key: * Fire Peak | ** Equinox/Solstice | *** Eclipse | "
            f"S Synergy | G Geomagnetic | P Planetary | † Review",
            "",
            f"{'Moon':>6s}  {'Name':>14s}  {'Start':>12s}  {'End':>12s}  {'Stamps':>30s}",
            "-" * 80,
        ]
        for e in entries:
            stamp_str = " ".join(s.code for s in e.stamps) if e.stamps else "—"
            lines.append(
                f"  {e.moon_number:>3d}   {e.moon_name:>14s}  "
                f"{e.start_date.isoformat():>12s}  {e.end_date.isoformat():>12s}  "
                f"{stamp_str:>30s}"
            )
        return "\n".join(lines)
