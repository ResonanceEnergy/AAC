"""
Rocket Ship — Lunar Cycle Timing Engine
=========================================
Moon cycles 1-39+ timing engine providing rebalance windows and
phi-ratio accumulation / execution windows for the Rocket Ship thesis.

Architecture:
    Moon 1-12   Life Boat phase  (March 2026 — March 2027)
    Moon 13-39  Rocket Ship phase (April 2027 — June 2029)
    Moon 40+    Orbit phase      (July 2029+)

Each moon uses the synodic month (~29.53 days) for date calculations
and a regularised 28-day grid for phi-window placement:

    Days 1-7    NEW    — reset, reassess thesis
    Days 8-14   WAXING — accumulate; phi window at days 10-11
    Days 15-21  FULL   — execute max conviction; phi window at days 17-18
    Days 22-28+ WANING — hedge, trim, protect
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from strategies.rocket_ship.core import (
    LIFEBOAT_INCEPTION,
    LIFEBOAT_MOON_END,
    ORBIT_MOON_START,
    ROCKET_MOON_END,
    ROCKET_MOON_START,
    SYNODIC_MONTH_DAYS,
    MoonPhase,
    SystemPhase,
)

# ═══════════════════════════════════════════════════════════════════════════
# PHI WINDOW CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# φ ≈ 1.618 — used to derive optimal accumulation / execution days
# Phi Window 1 (accumulate): day ≈ 28/φ² ≈ 10.7 → days 10-11
# Phi Window 2 (execute):    day ≈ 28/φ  ≈ 17.3 → days 17-18
PHI_WINDOW_1_START: int = 10
PHI_WINDOW_1_END: int = 11
PHI_WINDOW_2_START: int = 17
PHI_WINDOW_2_END: int = 18


# ═══════════════════════════════════════════════════════════════════════════
# MOON NAMES — thesis-aligned naming for each cycle
# ═══════════════════════════════════════════════════════════════════════════

MOON_NAMES: dict[int, str] = {
    # Life Boat (1-12)
    1: "Foundation",
    2: "Fortification",
    3: "Reconnaissance",
    4: "Accumulation",
    5: "Calibration",
    6: "Endurance",
    7: "Vigilance",
    8: "Observation",
    9: "Assessment",
    10: "Preparation",
    11: "Anticipation",
    12: "Threshold",
    # Rocket Ship (13-39)
    13: "Ignition",
    14: "Ascent",
    15: "Deployment",
    16: "Expansion",
    17: "Acceleration",
    18: "Diversification",
    19: "Consolidation",
    20: "Amplification",
    21: "Integration",
    22: "Stabilization",
    23: "Optimization",
    24: "Propagation",
    25: "Replication",
    26: "Maturation",
    27: "Fortification II",
    28: "Convergence",
    29: "Elevation",
    30: "Harmonization",
    31: "Calibration II",
    32: "Transcendence",
    33: "Sovereignty",
    34: "Liberation",
    35: "Culmination",
    36: "Synthesis",
    37: "Crystallization",
    38: "Completion",
    39: "Transition",
    # Orbit (40+)
    40: "Orbit Alpha",
}

# Key milestones at specific moon numbers
MILESTONES: dict[int, str] = {
    1: "Life Boat inception — survival allocation active",
    6: "Mid-Life Boat assessment checkpoint",
    12: "Life Boat final — ignition readiness review",
    13: "Default Rocket ignition (or earlier on Gulf trigger)",
    20: "Yield deployment target: 60% of Rocket allocation",
    26: "Rocket Ship midpoint — full geo diversification",
    33: "Sovereignty milestone — multi-jurisdiction active",
    39: "Rocket Ship final — Orbit transition preparation",
    40: "Orbit begins — 2030 stable-state diversification",
}


# ═══════════════════════════════════════════════════════════════════════════
# LUNAR STATE DATACLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LunarState:
    """Snapshot of the current position in the Rocket Ship lunar cycle."""

    moon_number: int
    moon_name: str
    new_moon_date: date
    day_in_moon: int
    days_to_rocket_start: int
    in_phi_window_1: bool
    in_phi_window_2: bool
    is_rocket_phase: bool
    milestone: str | None
    moon_phase: MoonPhase
    system_phase: SystemPhase
    days_in_this_moon: int


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class RocketLunarEngine:
    """Timing engine mapping calendar dates to Rocket Ship lunar cycles."""

    def __init__(self) -> None:
        self._inception: date = LIFEBOAT_INCEPTION
        self._synodic_days: float = SYNODIC_MONTH_DAYS

    # ── Core computations ────────────────────────────────────────────────

    def _moon_start_date(self, moon_number: int) -> date:
        """Compute the start (new-moon) date for a given moon number."""
        days_offset = (moon_number - 1) * self._synodic_days
        return self._inception + timedelta(days=days_offset)

    def _current_moon_number(self, ref_date: date) -> int:
        """Determine which moon number contains *ref_date*."""
        if ref_date < self._inception:
            return 0  # Pre-inception
        elapsed = (ref_date - self._inception).days
        # Initial estimate from floating-point division
        moon_num = int(elapsed / self._synodic_days) + 1
        # Correct boundary: if the next moon's start date falls on or before
        # ref_date, we've crossed into that moon (handles truncation mismatch).
        while self._moon_start_date(moon_num + 1) <= ref_date:
            moon_num += 1
        return moon_num

    def _day_in_moon(self, ref_date: date, moon_number: int) -> int:
        """1-based day within the current moon cycle."""
        moon_start = self._moon_start_date(moon_number)
        return (ref_date - moon_start).days + 1

    def _days_in_moon(self, moon_number: int) -> int:
        """Total calendar days in this specific moon."""
        start = self._moon_start_date(moon_number)
        end = self._moon_start_date(moon_number + 1)
        return (end - start).days

    @staticmethod
    def _moon_phase(day: int) -> MoonPhase:
        """Map day-in-moon to the 4-phase framework."""
        if day <= 7:
            return MoonPhase.NEW
        if day <= 14:
            return MoonPhase.WAXING
        if day <= 21:
            return MoonPhase.FULL
        return MoonPhase.WANING

    @staticmethod
    def _system_phase_for_moon(moon_number: int) -> SystemPhase:
        """Determine system phase from moon number."""
        if moon_number <= LIFEBOAT_MOON_END:
            return SystemPhase.LIFE_BOAT
        if moon_number <= ROCKET_MOON_END:
            return SystemPhase.ROCKET
        return SystemPhase.ORBIT

    # ── Public API ───────────────────────────────────────────────────────

    def get_current_state(self, ref_date: date | None = None) -> LunarState:
        """Compute full lunar state for the given date (default: today)."""
        ref = ref_date or date.today()
        moon_num = self._current_moon_number(ref)
        if moon_num < 1:
            moon_num = 1  # Clamp to Moon 1 if before inception

        day = self._day_in_moon(ref, moon_num)
        total_days = self._days_in_moon(moon_num)

        rocket_start_date = self._moon_start_date(ROCKET_MOON_START)
        days_to_rocket = (rocket_start_date - ref).days

        return LunarState(
            moon_number=moon_num,
            moon_name=MOON_NAMES.get(moon_num, f"Moon {moon_num}"),
            new_moon_date=self._moon_start_date(moon_num),
            day_in_moon=day,
            days_to_rocket_start=max(days_to_rocket, 0),
            in_phi_window_1=PHI_WINDOW_1_START <= day <= PHI_WINDOW_1_END,
            in_phi_window_2=PHI_WINDOW_2_START <= day <= PHI_WINDOW_2_END,
            is_rocket_phase=moon_num >= ROCKET_MOON_START,
            milestone=MILESTONES.get(moon_num),
            moon_phase=self._moon_phase(day),
            system_phase=self._system_phase_for_moon(moon_num),
            days_in_this_moon=total_days,
        )

    def format_dashboard(self, ref_date: date | None = None) -> str:
        """Format a human-readable dashboard of the current lunar state."""
        state = self.get_current_state(ref_date)

        phase_label = state.system_phase.value.replace("_", " ").upper()
        lines: list[str] = []

        # Phase banner
        lines.append(f"  Phase: {phase_label}  |  Moon {state.moon_number}: {state.moon_name}")
        lines.append(f"  New Moon Date: {state.new_moon_date.isoformat()}")
        lines.append(
            f"  Day in Moon:   {state.day_in_moon} / {state.days_in_this_moon}"
            f"  ({state.moon_phase.value.upper()} phase)"
        )
        lines.append("")

        # Phi windows
        phi1 = (
            ">>> ACTIVE <<<"
            if state.in_phi_window_1
            else f"days {PHI_WINDOW_1_START}-{PHI_WINDOW_1_END}"
        )
        phi2 = (
            ">>> ACTIVE <<<"
            if state.in_phi_window_2
            else f"days {PHI_WINDOW_2_START}-{PHI_WINDOW_2_END}"
        )
        lines.append(f"  Phi Window 1 (Accumulate): {phi1}")
        lines.append(f"  Phi Window 2 (Execute):    {phi2}")
        lines.append("")

        # Rocket countdown
        if state.is_rocket_phase:
            remaining = ROCKET_MOON_END - state.moon_number
            lines.append(f"  Rocket Phase: ACTIVE  |  {remaining} moons remaining to Orbit")
        else:
            lines.append(f"  Days to Rocket Start: {state.days_to_rocket_start}")
            moons_left = ROCKET_MOON_START - state.moon_number
            lines.append(f"  Moons to Ignition:    {moons_left}")
        lines.append("")

        # Milestone
        if state.milestone:
            lines.append(f"  * Milestone: {state.milestone}")
            lines.append("")

        # Progress bar
        pct = min(state.moon_number / ORBIT_MOON_START * 100, 100)
        filled = int(pct / 2.5)  # 40 chars wide
        bar = "#" * filled + "-" * (40 - filled)
        lines.append(f"  Progress: [{bar}] {pct:.1f}%")
        lines.append(f"           Moon {state.moon_number} of {ORBIT_MOON_START}  |  {phase_label}")

        return "\n".join(lines)
