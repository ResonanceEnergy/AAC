"""
Rocket Ship — Moon Cycles 13–39 Timing Engine
===============================================
Tracks the post-Life-Boat window (Moon Cycles 13–39) for disciplined
rebalancing and deployment of Rocket Ship allocations.

Calendar basis:
    Life Boat inception: March 22, 2026
    Each "moon" uses the real synodic month (29.530589 days) for accuracy.

    Moon 1:  March 22, 2026    (Life Boat begins)
    Moon 13: ~March 11, 2027   (Default Rocket Ship ignition — also LIFE BOAT END)
    Moon 39: ~June 17, 2029    (Rocket window closes → ORBIT stabilization)

Phi timing (within each 29.53-day moon):
    Same as Storm Lifeboat — peak action windows at golden-ratio positions.
    PHI_WINDOW_1: days 10-11   (28/phi^2 ≈ 10.7)
    PHI_WINDOW_2: days 17-18   (28/phi  ≈ 17.3)

    Use NEW MOON days (day 1) for major rebalancing decisions.
    Use FULL MOON days (days 14-15) for execution of entries.
    Phi windows = peak coherence for deployment.

Moon phases (within each 29.53-day cycle):
    NEW    (days 1-7):   Reset, re-read indicators, decide if ignition
    WAXING (days 8-15):  Accumulate — scale into Rocket positions
    FULL   (days 16-22): Execute — max conviction deployment
    WANING (days 23-29): Hedge — trim, protect, rebalance to targets

Rebalance discipline:
    - Check ignition on NEW MOON only (no emotional intra-cycle decisions)
    - Bridge XRP → FXRP on PHI_WINDOW_1 or PHI_WINDOW_2 of WAXING moon
    - Reallocate BRICS/Unit exposure on FULL MOON
    - Review geo-plan task list on WANING moon
    - Each rebalance action logged with moon number and day
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from strategies.rocket_ship.core import (
    LIFEBOAT_INCEPTION,
    LIFEBOAT_MOON_END,
    ROCKET_MOON_END,
    ROCKET_MOON_START,
    SYNODIC_MONTH_DAYS,
    MoonPhase,
)

logger = logging.getLogger(__name__)

PHI = (1 + math.sqrt(5)) / 2   # 1.6180339887...
MOON_CYCLE_DAYS = SYNODIC_MONTH_DAYS  # 29.530589 days

# Phi-proportional action windows within a regularized 28-day moon
PHI_WINDOW_1: Tuple[int, int] = (10, 11)   # 28 / phi^2 ≈ 10.7
PHI_WINDOW_2: Tuple[int, int] = (17, 18)   # 28 / phi  ≈ 17.3

# Astronomical new moon dates 2026-2030 (UTC, accurate to ±1 day)
# Source: NASA / USNO new moon almanac
NEW_MOON_DATES: List[date] = [
    # 2026
    date(2026, 3, 29),   # Moon 1 base (closest new moon after inception Mar 22)
    date(2026, 4, 27),   # Moon 2
    date(2026, 5, 27),   # Moon 3
    date(2026, 6, 25),   # Moon 4 — Panama scout trip target ★
    date(2026, 7, 25),   # Moon 5
    date(2026, 8, 23),   # Moon 6
    date(2026, 9, 21),   # Moon 7
    date(2026, 10, 21),  # Moon 8
    date(2026, 11, 20),  # Moon 9
    date(2026, 12, 19),  # Moon 10
    # 2027
    date(2027, 1, 18),   # Moon 11
    date(2027, 2, 17),   # Moon 12  — Last Life Boat moon
    date(2027, 3, 19),   # Moon 13  ★ DEFAULT IGNITION — Rocket Ship begins
    date(2027, 4, 17),   # Moon 14
    date(2027, 5, 17),   # Moon 15
    date(2027, 6, 15),   # Moon 16
    date(2027, 7, 15),   # Moon 17
    date(2027, 8, 13),   # Moon 18
    date(2027, 9, 11),   # Moon 19
    date(2027, 10, 11),  # Moon 20
    date(2027, 11, 9),   # Moon 21
    date(2027, 12, 9),   # Moon 22
    # 2028
    date(2028, 1, 7),    # Moon 23
    date(2028, 2, 5),    # Moon 24
    date(2028, 3, 6),    # Moon 25
    date(2028, 4, 5),    # Moon 26
    date(2028, 5, 4),    # Moon 27
    date(2028, 6, 2),    # Moon 28  — UAE Golden Visa target window ★
    date(2028, 7, 2),    # Moon 29
    date(2028, 8, 1),    # Moon 30
    date(2028, 8, 30),   # Moon 31
    date(2028, 9, 28),   # Moon 32
    date(2028, 10, 27),  # Moon 33
    date(2028, 11, 26),  # Moon 34
    date(2028, 12, 26),  # Moon 35
    # 2029
    date(2029, 1, 24),   # Moon 36
    date(2029, 2, 23),   # Moon 37
    date(2029, 3, 24),   # Moon 38
    date(2029, 4, 22),   # Moon 39  ★ ROCKET WINDOW CLOSES — enter ORBIT
    date(2029, 5, 22),   # Moon 40  — Orbit phase begins
    date(2029, 6, 20),   # Moon 41
]

# Moon names (Resonance / 13-moon tradition — reset from moon 13)
ROCKET_MOON_NAMES: Dict[int, str] = {
    13: "Ignition",        # Default launch moon
    14: "Ascent",          # Initial climb
    15: "Acceleration",    # Speed building
    16: "Orbit Lock",      # Locks into new system rails
    17: "Resonant Fire",   # Peak phi alignment
    18: "Bridge",          # CBDC/DeFi bridge activation
    19: "Yield",           # Morpho/Flare yield running
    20: "Harvest",         # Mid-window harvest and review
    21: "Panama Base",     # Geo-plan Panama fully operational
    22: "Expansion",       # Paraguay + Panama running
    23: "UAE Entry",       # UAE Golden Visa target window
    24: "Asia Pivot",      # Singapore/Malaysia residency research
    25: "Gravity",         # Gravity assist — compound yield doing work
    26: "Solana Peak",     # Solana stablecoin volume thesis check
    27: "BRICS Unit",      # BRICS Unit retail access expected
    28: "Deep Space",      # UAE Golden Visa submitted
    29: "Interop",         # ISO-20022 CBDC interoperability milestone
    30: "Midpoint",        # Rebalance review (Cycle 30 = halfway 13-39)
    31: "Gold Anchor",     # Physical gold/PAXG rebalance
    32: "Swiss Layer",     # Zug/Switzerland research trip
    33: "Consolidation",   # Trim, protect, consolidate gains
    34: "Singapore",       # SEA base activation
    35: "RWA",             # Real-world asset tokenization thesis check
    36: "Final Approach",  # Approaching 2030 full diversification
    37: "Five Flags",      # All 5 residencies active
    38: "Orbit Prep",      # Reduce DeFi risk, build stability
    39: "Orbit",           # Rocket window closes → ORBIT stability
}

# Special milestone moons
MILESTONE_MOONS: Dict[int, str] = {
    4:  "Panama Scouting Trip — June 2026 target",
    6:  "Paraguay Residency Filed — August 2026 target",
    13: "DEFAULT IGNITION — Rocket Ship launches",
    15: "Panama Residency Final — 6 months after filing",
    28: "UAE Golden Visa Application Window",
    30: "Mid-Rocket Review — Rebalance all positions",
    39: "ROCKET WINDOW CLOSES — Full 2030 ORBIT achieved",
}


# ═══════════════════════════════════════════════════════════════════════════
# MOON STATE DATACLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MoonState:
    """Complete state of the current moon cycle position."""
    # Absolute moon number (1 = Life Boat start, 13 = default ignition)
    moon_number: int
    # Day within current moon (1-29.53)
    day_in_moon: int
    # Phase based on day position
    phase: MoonPhase
    # Dates
    current_date: date
    new_moon_date: date          # Start of current moon
    next_new_moon: date         # When next moon begins
    # Phi windows
    in_phi_window_1: bool        # Days 10-11 within moon
    in_phi_window_2: bool        # Days 17-18 within moon
    in_any_phi_window: bool
    # Phase identification
    is_life_boat_phase: bool     # Moon 1-12
    is_rocket_phase: bool        # Moon 13-39
    is_orbit_phase: bool         # Moon 40+
    # Days remaining
    days_to_rocket_start: int    # Days until Moon 13
    days_in_rocket_window: int   # Days elapsed if in rocket phase
    days_remaining_rocket: int   # Days left in rocket window
    # Names and labels
    moon_name: str
    milestone: Optional[str]
    # Recommended action
    action_label: str


# ═══════════════════════════════════════════════════════════════════════════
# LUNAR CYCLE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class RocketLunarEngine:
    """
    Computes current moon position and all timing for Rocket Ship operations.

    Usage:
        engine = RocketLunarEngine()
        state = engine.get_current_state()
        print(state.moon_name, state.phase.value, state.action_label)
    """

    def __init__(self, inception: date = LIFEBOAT_INCEPTION) -> None:
        self.inception = inception
        self._new_moons = NEW_MOON_DATES

    def _find_moon_window(self, target: date) -> Tuple[int, date, date]:
        """
        Returns (moon_index_0based, new_moon_start, next_new_moon)
        where moon_index_0based=0 means Moon 1.
        Before the first new moon (inception → Moon 1 new moon) counts as Moon 1.
        """
        # Before Moon 1's new moon — treat as early days of Moon 1
        if target < self._new_moons[0]:
            return 0, self._new_moons[0], self._new_moons[1]

        for i, nm in enumerate(self._new_moons[:-1]):
            next_nm = self._new_moons[i + 1]
            if nm <= target < next_nm:
                return i, nm, next_nm
        # After last known date: extrapolate
        last = self._new_moons[-1]
        idx = len(self._new_moons) - 1
        while target >= last + timedelta(days=MOON_CYCLE_DAYS):
            last += timedelta(days=MOON_CYCLE_DAYS)
            idx += 1
        next_nm = last + timedelta(days=MOON_CYCLE_DAYS)
        return idx, last, next_nm

    def _compute_phase(self, day_in_moon: int) -> MoonPhase:
        """Classify moon phase from day number (1-29)."""
        if day_in_moon <= 7:
            return MoonPhase.NEW
        if day_in_moon <= 15:
            return MoonPhase.WAXING
        if day_in_moon <= 22:
            return MoonPhase.FULL
        return MoonPhase.WANING

    def _get_action_label(self, phase: MoonPhase, moon_num: int, in_phi: bool) -> str:
        """Return a human-readable action recommendation."""
        is_lifeboat = moon_num < ROCKET_MOON_START
        is_rocket = ROCKET_MOON_START <= moon_num <= ROCKET_MOON_END

        if is_lifeboat:
            actions = {
                MoonPhase.NEW:    "LIFEBOAT: Review indicators + prepare docs",
                MoonPhase.WAXING: "LIFEBOAT: Build XRP position + test bridges",
                MoonPhase.FULL:   "LIFEBOAT: Execute any needed stablecoin moves",
                MoonPhase.WANING: "LIFEBOAT: Hedge, check Gulf Yuan news",
            }
        elif is_rocket:
            actions = {
                MoonPhase.NEW:    "ROCKET: Check ignition criteria + review 15 indicators",
                MoonPhase.WAXING: "ROCKET: Bridge XRP → FXRP + accumulate DeFi positions",
                MoonPhase.FULL:   "ROCKET: Execute max allocation + review geo-plan tasks",
                MoonPhase.WANING: "ROCKET: Trim, rebalance to targets + protect gains",
            }
        else:
            actions = {
                MoonPhase.NEW:    "ORBIT: Review 2030 diversification status",
                MoonPhase.WAXING: "ORBIT: Compound yields",
                MoonPhase.FULL:   "ORBIT: Execute any remaining residency/banking tasks",
                MoonPhase.WANING: "ORBIT: Maintain allocations",
            }
        label = actions[phase]
        if in_phi:
            label += " ★ PHI WINDOW — Peak coherence, high-conviction action"
        return label

    def get_current_state(self, today: Optional[date] = None) -> MoonState:
        """Compute full moon state for the given date (defaults to today)."""
        target = today or date.today()

        moon_idx_0, nm_start, nm_next = self._find_moon_window(target)
        moon_number = moon_idx_0 + 1         # 1-based

        # If we're before Moon 1's new moon, count days from inception
        if target < nm_start:
            day_in_moon = max(1, (target - self.inception).days + 1)
        else:
            day_in_moon = (target - nm_start).days + 1
        phase = self._compute_phase(day_in_moon)

        in_phi1 = day_in_moon in range(PHI_WINDOW_1[0], PHI_WINDOW_1[1] + 1)
        in_phi2 = day_in_moon in range(PHI_WINDOW_2[0], PHI_WINDOW_2[1] + 1)
        in_any_phi = in_phi1 or in_phi2

        is_life_boat = moon_number < ROCKET_MOON_START
        is_rocket = ROCKET_MOON_START <= moon_number <= ROCKET_MOON_END
        is_orbit = moon_number > ROCKET_MOON_END

        # Days to Rocket start (Moon 13)
        if moon_number < ROCKET_MOON_START:
            rocket_start_date = NEW_MOON_DATES[ROCKET_MOON_START - 1]
            days_to_rocket = (rocket_start_date - target).days
        else:
            days_to_rocket = 0

        # Days in/remaining rocket window
        if is_rocket:
            rocket_start_date = NEW_MOON_DATES[ROCKET_MOON_START - 1]
            rocket_end_date = NEW_MOON_DATES[ROCKET_MOON_END - 1]
            days_in_rocket = (target - rocket_start_date).days
            days_remaining_rocket = (rocket_end_date - target).days
        else:
            days_in_rocket = 0
            days_remaining_rocket = max(
                0, (NEW_MOON_DATES[ROCKET_MOON_END - 1] - target).days
            )

        moon_name = ROCKET_MOON_NAMES.get(moon_number, f"Moon {moon_number}")
        milestone = MILESTONE_MOONS.get(moon_number)
        action = self._get_action_label(phase, moon_number, in_any_phi)

        return MoonState(
            moon_number=moon_number,
            day_in_moon=day_in_moon,
            phase=phase,
            current_date=target,
            new_moon_date=nm_start,
            next_new_moon=nm_next,
            in_phi_window_1=in_phi1,
            in_phi_window_2=in_phi2,
            in_any_phi_window=in_any_phi,
            is_life_boat_phase=is_life_boat,
            is_rocket_phase=is_rocket,
            is_orbit_phase=is_orbit,
            days_to_rocket_start=days_to_rocket,
            days_in_rocket_window=days_in_rocket,
            days_remaining_rocket=days_remaining_rocket,
            moon_name=moon_name,
            milestone=milestone,
            action_label=action,
        )

    def format_dashboard(self, today: Optional[date] = None) -> str:
        """Return formatted ASCII lunar position dashboard."""
        state = self.get_current_state(today)

        phase_bar = {
            MoonPhase.NEW:    "●○○○",
            MoonPhase.WAXING: "●●○○",
            MoonPhase.FULL:   "●●●○",
            MoonPhase.WANING: "●●●●",
        }

        if state.is_life_boat_phase:
            system_label = f"LIFE BOAT — {state.days_to_rocket_start} days to Moon 13"
            phase_color = "🛥"
        elif state.is_rocket_phase:
            pct = 100 * (1 - state.days_remaining_rocket / (ROCKET_MOON_END - ROCKET_MOON_START + 1) / MOON_CYCLE_DAYS)
            system_label = f"ROCKET SHIP — {state.days_remaining_rocket} days remaining in window"
            phase_color = "🚀"
        else:
            system_label = "ORBIT — Full 2030 diversification active"
            phase_color = "🌍"

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════════════╗",
            "║          ROCKET SHIP — LUNAR TIMING ENGINE                             ║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
            f"║  Moon #{state.moon_number:<3}  │  {state.moon_name:<20}  │  Day {state.day_in_moon:>2}/29     ║",
            f"║  Phase: {state.phase.value.upper():<10}  {phase_bar[state.phase]}  │  {system_label:<30}║",
            f"║  New moon: {state.new_moon_date}  →  Next: {state.next_new_moon}              ║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
        ]

        if state.in_any_phi_window:
            lines.append("║  ★  PHI WINDOW ACTIVE — Peak coherence for high-conviction action   ★  ║")
        else:
            next_phi = PHI_WINDOW_1[0] - state.day_in_moon
            if next_phi < 0:
                next_phi = PHI_WINDOW_2[0] - state.day_in_moon
            days_to_phi = max(0, next_phi)
            lines.append(f"║  Next phi window in {days_to_phi:>2} days  (days {PHI_WINDOW_1[0]}-{PHI_WINDOW_1[1]} or {PHI_WINDOW_2[0]}-{PHI_WINDOW_2[1]})                     ║")

        lines.append("╠══════════════════════════════════════════════════════════════════════════╣")
        lines.append(f"║  ACTION: {state.action_label[:64]:<64}║")

        if state.milestone:
            lines.append("╠══════════════════════════════════════════════════════════════════════════╣")
            lines.append(f"║  ★ MILESTONE: {state.milestone[:60]:<60}  ║")

        lines.append("╠══════════════════════════════════════════════════════════════════════════╣")
        lines.append("║  UPCOMING KEY MOONS:                                                    ║")
        for mn, label in MILESTONE_MOONS.items():
            if mn > state.moon_number:
                nm_date = NEW_MOON_DATES[mn - 1] if mn <= len(NEW_MOON_DATES) else "?"
                lines.append(f"║    Moon {mn:>2} ({nm_date})  {label[:45]:<45}  ║")
                if len([x for x in MILESTONE_MOONS if x > state.moon_number and x <= mn]) >= 3:
                    break

        lines.append("╚══════════════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)
