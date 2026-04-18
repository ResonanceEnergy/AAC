"""Alert Engine -- threshold-based alerts, milestone tracking, strategy adjustments.

Monitors portfolio NAV against $10M target phase gates and publishes
alerts when milestones are approached or indicator regimes shift.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

_log = structlog.get_logger()

# $10M target phase gates (phi-ratio spacing from War Room engine)
PHASE_GATES: dict[str, float] = {
    "SEED": 0.0,
    "SPROUT": 50_000.0,
    "GROWTH": 150_000.0,
    "MOMENTUM": 500_000.0,
    "SCALE": 1_000_000.0,
    "COMPOUND": 2_500_000.0,
    "ACCELERATE": 5_000_000.0,
    "TARGET": 10_000_000.0,
}

# Proximity thresholds for alerts (percentage of next gate)
PROXIMITY_THRESHOLDS = [0.10, 0.05, 0.01]  # 10%, 5%, 1%


class AlertSeverity(enum.Enum):
    """Alert severity levels."""

    INFO = "info"
    ATTENTION = "attention"
    ACTION = "action"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A generated alert."""

    alert_id: str
    severity: AlertSeverity
    category: str  # milestone, regime_shift, indicator, risk
    title: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False


class MilestoneTracker:
    """Track portfolio NAV against $10M phase gates."""

    def __init__(self) -> None:
        self._current_nav: float = 0.0
        self._current_phase: str = "SEED"
        self._alerts_fired: set[str] = set()
        self._alert_counter: int = 0

    def update_nav(self, nav: float) -> list[Alert]:
        """Update NAV and check for milestone proximity alerts."""
        self._current_nav = nav
        alerts: list[Alert] = []

        # Determine current phase
        current_phase = "SEED"
        for phase, gate in PHASE_GATES.items():
            if nav >= gate:
                current_phase = phase

        # Phase transition alert
        if current_phase != self._current_phase:
            prev = self._current_phase
            self._current_phase = current_phase
            alert = self._make_alert(
                AlertSeverity.ACTION,
                "milestone",
                f"Phase Gate Crossed: {prev} -> {current_phase}",
                f"Portfolio NAV ${nav:,.2f} crossed into {current_phase} phase",
                {"previous_phase": prev, "new_phase": current_phase, "nav": nav},
            )
            alerts.append(alert)
            _log.info("milestone.phase_crossed", previous=prev, current=current_phase, nav=nav)

        # Proximity alerts for next phase gate
        gates = list(PHASE_GATES.items())
        for i, (phase, gate) in enumerate(gates):
            if gate > nav and gate > 0:
                distance = gate - nav
                for threshold in PROXIMITY_THRESHOLDS:
                    pct_remaining = distance / gate
                    alert_key = f"{phase}_{threshold}"
                    if pct_remaining <= threshold and alert_key not in self._alerts_fired:
                        self._alerts_fired.add(alert_key)
                        pct_str = f"{threshold * 100:.0f}%"
                        alert = self._make_alert(
                            AlertSeverity.ATTENTION,
                            "milestone",
                            f"Approaching {phase}: within {pct_str}",
                            f"NAV ${nav:,.2f} is within {pct_str} of {phase} gate (${gate:,.0f}). Distance: ${distance:,.2f}",
                            {"target_phase": phase, "gate": gate, "nav": nav, "distance": distance},
                        )
                        alerts.append(alert)
                break  # Only alert for next gate

        return alerts

    def get_progress(self) -> dict[str, Any]:
        """Get current progress toward $10M target."""
        target = PHASE_GATES["TARGET"]
        return {
            "current_nav": self._current_nav,
            "current_phase": self._current_phase,
            "target": target,
            "progress_pct": (self._current_nav / target * 100) if target > 0 else 0,
            "remaining": target - self._current_nav,
            "phases": {
                phase: {"gate": gate, "reached": self._current_nav >= gate}
                for phase, gate in PHASE_GATES.items()
            },
        }

    def _make_alert(
        self,
        severity: AlertSeverity,
        category: str,
        title: str,
        message: str,
        data: dict[str, Any],
    ) -> Alert:
        self._alert_counter += 1
        return Alert(
            alert_id=f"milestone_{self._alert_counter}",
            severity=severity,
            category=category,
            title=title,
            message=message,
            data=data,
        )
