"""
Rocket Ship — Trigger Engine
==============================
Evaluates ignition criteria by combining the 15 de-dollarization
indicators with the Gulf Yuan Oil trigger to determine the current
system phase and generate actionable recommendations.

Ignition criteria:
    a) Gulf Trigger CONFIRMED (override) → IGNITE immediately
    b) ≥ 10 of 15 indicators GREEN     → IGNITE on schedule

Phase determination:
    LIFE_BOAT  — Ignition criteria NOT met
    IGNITING   — Criteria met, deployment in progress
    ROCKET     — Full Rocket allocation active
    ORBIT      — Moon 40+, stable-state
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import List

from strategies.rocket_ship.core import (
    INDICATORS_REQUIRED_FOR_IGNITION,
    ORBIT_MOON_START,
    ROCKET_MOON_START,
    SystemPhase,
    TriggerStatus,
)
from strategies.rocket_ship.indicators import IndicatorEngine
from strategies.rocket_ship.lunar_cycles import RocketLunarEngine

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# TRIGGER REPORT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TriggerReport:
    """Result of a full ignition evaluation."""

    phase: SystemPhase
    alert_level: str               # "LOW" | "ELEVATED" | "HIGH" | "IGNITED"
    is_ignited: bool
    ignition_probability: float    # 0.0 – 1.0 estimated near-term probability
    green_count: int
    gulf_trigger_status: str       # TriggerStatus value
    days_to_default_ignition: int  # Calendar days to Moon 13
    immediate_actions: List[str]
    reason: str


# ═══════════════════════════════════════════════════════════════════════════
# TRIGGER ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class TriggerEngine:
    """
    Evaluates ignition readiness by combining indicator state,
    Gulf trigger status, and lunar cycle position.
    """

    def __init__(self) -> None:
        self._indicators = IndicatorEngine()
        self._indicators.load_state()
        self._lunar = RocketLunarEngine()

    def evaluate(self, ref_date: date | None = None) -> TriggerReport:
        """Run full ignition evaluation and return a TriggerReport."""
        ref = ref_date or date.today()
        lunar_state = self._lunar.get_current_state(ref)

        green_count = self._indicators.green_count()
        ignited, reason = self._indicators.check_ignition()
        gulf_status = self._indicators.gulf_trigger.status
        days_to_default = lunar_state.days_to_rocket_start

        # Phase determination
        phase = self._determine_phase(
            ignited, lunar_state.moon_number, gulf_status,
        )

        # Alert level
        alert_level = self._compute_alert_level(
            green_count, gulf_status, ignited,
        )

        # Ignition probability estimate
        prob = self._estimate_probability(
            green_count, gulf_status, days_to_default,
        )

        # Immediate actions
        actions = self._recommend_actions(
            phase, green_count, gulf_status, lunar_state.moon_number,
        )

        return TriggerReport(
            phase=phase,
            alert_level=alert_level,
            is_ignited=ignited,
            ignition_probability=round(prob, 2),
            green_count=green_count,
            gulf_trigger_status=gulf_status.value,
            days_to_default_ignition=days_to_default,
            immediate_actions=actions,
            reason=reason,
        )

    # ── Phase logic ──────────────────────────────────────────────────────

    @staticmethod
    def _determine_phase(
        ignited: bool,
        moon_number: int,
        gulf_status: TriggerStatus,
    ) -> SystemPhase:
        if moon_number >= ORBIT_MOON_START:
            return SystemPhase.ORBIT
        if moon_number >= ROCKET_MOON_START:
            return SystemPhase.ROCKET
        if ignited or gulf_status == TriggerStatus.CONFIRMED:
            return SystemPhase.IGNITING
        return SystemPhase.LIFE_BOAT

    @staticmethod
    def _compute_alert_level(
        green_count: int,
        gulf_status: TriggerStatus,
        ignited: bool,
    ) -> str:
        if ignited or gulf_status == TriggerStatus.CONFIRMED:
            return "IGNITED"
        if gulf_status == TriggerStatus.EMERGING or green_count >= 12:
            return "HIGH"
        if green_count >= INDICATORS_REQUIRED_FOR_IGNITION:
            return "ELEVATED"
        return "LOW"

    @staticmethod
    def _estimate_probability(
        green_count: int,
        gulf_status: TriggerStatus,
        days_to_default: int,
    ) -> float:
        """Rough heuristic for near-term ignition probability."""
        if gulf_status == TriggerStatus.CONFIRMED:
            return 1.0
        base = green_count / 15.0 * 0.5  # 0–0.5 from indicators
        if gulf_status == TriggerStatus.EMERGING:
            base += 0.3
        # Time proximity boost (closer to Moon 13 → higher baseline)
        if days_to_default <= 60:
            base += 0.15
        elif days_to_default <= 180:
            base += 0.05
        return min(base, 1.0)

    @staticmethod
    def _recommend_actions(
        phase: SystemPhase,
        green_count: int,
        gulf_status: TriggerStatus,
        moon_number: int,
    ) -> List[str]:
        actions: List[str] = []
        if phase == SystemPhase.LIFE_BOAT:
            actions.append("Maintain Life Boat allocation (BTC/gold/stables/XRP)")
            if green_count >= INDICATORS_REQUIRED_FOR_IGNITION:
                actions.append(
                    f"ALERT: {green_count}/15 GREEN — prepare Rocket deployment checklist"
                )
            if gulf_status == TriggerStatus.EMERGING:
                actions.append(
                    "Gulf trigger EMERGING — monitor mBridge reports and Saudi energy news daily"
                )
            actions.append("Review indicator data on scheduled update dates")
        elif phase == SystemPhase.IGNITING:
            actions.append("Begin phased Rocket allocation deployment")
            actions.append("Open exchange accounts for Flare/Solana/Ethereum on-ramps")
            actions.append("Bridge initial 10% XRP → FXRP on Flare for Morpho test")
            actions.append("Set up hardware wallet multi-sig for >$50K positions")
        elif phase == SystemPhase.ROCKET:
            actions.append(f"Moon {moon_number}: Execute Rocket allocation rebalance")
            actions.append("Monitor yield tactics — review APYs weekly")
            actions.append("Advance geo plan tasks for current moon")
        elif phase == SystemPhase.ORBIT:
            actions.append("Orbit phase — stable-state portfolio management")
            actions.append("Focus on geo diversification and tax optimization")

        return actions

    # ── Dashboard ────────────────────────────────────────────────────────

    def format_dashboard(self, ref_date: date | None = None) -> str:
        """Return a formatted ASCII trigger report."""
        report = self.evaluate(ref_date)

        phase_labels = {
            SystemPhase.LIFE_BOAT: "LIFE BOAT",
            SystemPhase.IGNITING:  "IGNITING",
            SystemPhase.ROCKET:    "ROCKET SHIP",
            SystemPhase.ORBIT:     "ORBIT",
        }
        p_label = phase_labels.get(report.phase, report.phase.value)

        alert_symbols = {
            "LOW":      "🟢",
            "ELEVATED": "🟡",
            "HIGH":     "🟠",
            "IGNITED":  "🔴",
        }
        alert_sym = alert_symbols.get(report.alert_level, "⚪")

        lines: list[str] = [
            f"  Current Phase:    {p_label}",
            f"  Alert Level:      {alert_sym} {report.alert_level}",
            f"  Ignition Status:  {'YES — IGNITED' if report.is_ignited else 'NO — Monitoring'}",
            "",
            f"  Indicators:       {report.green_count}/15 GREEN"
            f"  (need {INDICATORS_REQUIRED_FOR_IGNITION})",
            f"  Gulf Trigger:     [{report.gulf_trigger_status.upper():^12}]",
            f"  Ignition Prob:    {report.ignition_probability:.0%}",
            f"  Days to Default:  {report.days_to_default_ignition}",
            "",
            f"  Reason: {report.reason}",
            "",
            "  Immediate Actions:",
        ]
        for i, action in enumerate(report.immediate_actions, 1):
            lines.append(f"    {i}. {action}")

        return "\n".join(lines)
