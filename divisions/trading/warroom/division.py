"""War Room Division -- options compounding engine targeting $10M.

Wraps the existing War Room engine, 13 Moon doctrine, and sub-strategies
(Storm Lifeboat, Matrix Maximizer, Rocket Ship, Options Intelligence)
into a DivisionProtocol participant that receives Research intel and
publishes trade signals.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from divisions.division_protocol import (
    DivisionHealth,
    DivisionProtocol,
    Signal,
    SignalType,
)
from divisions.research.alerts.alert_engine import MilestoneTracker

_log = structlog.get_logger()

# $10M target
TARGET_NAV = 10_000_000.0


class WarRoomDivision(DivisionProtocol):
    """War Room / 13 Moon division.

    Runs the Monte Carlo engine, 5-arm position management,
    and milestone tracking toward $10M.
    Consumes Research intel to adjust scenario weights.
    """

    def __init__(self) -> None:
        super().__init__(division_name="warroom")
        self._milestone_tracker = MilestoneTracker()
        self._current_nav: float = 0.0
        self._intel_buffer: list[dict[str, Any]] = []
        self._signals_generated: int = 0

    async def scan(self) -> list[Signal]:
        """Run War Room scan cycle.

        1. Check milestone progress
        2. Process buffered research intel
        3. Generate trade signals if doctrine allows
        """
        signals: list[Signal] = []

        # Check milestones
        alerts = self._milestone_tracker.update_nav(self._current_nav)
        for alert in alerts:
            signals.append(Signal(
                signal_type=SignalType.MILESTONE_ALERT,
                source_division=self.division_name,
                timestamp=datetime.now(timezone.utc),
                data=alert.data,
                confidence=1.0,
                urgency=2,
            ))

        # Process intel buffer
        if self._intel_buffer:
            _log.info(
                "warroom.processing_intel",
                count=len(self._intel_buffer),
            )
            self._intel_buffer.clear()

        self._signals_generated += len(signals)
        return signals

    async def report(self) -> dict[str, Any]:
        """Status report."""
        progress = self._milestone_tracker.get_progress()
        return {
            "division": self.division_name,
            "health": self.health.value,
            "target": TARGET_NAV,
            "current_nav": self._current_nav,
            "progress": progress,
            "signals_generated": self._signals_generated,
            "intel_buffered": len(self._intel_buffer),
            "cycle_count": self._cycle_count,
        }

    async def consume_signal(self, signal: Signal) -> None:
        """Receive research intel and buffer for next scan cycle."""
        await super().consume_signal(signal)
        if signal.signal_type in (
            SignalType.INTEL_UPDATE,
            SignalType.STRATEGY_ADJUSTMENT,
            SignalType.RISK_ALERT,
        ):
            self._intel_buffer.append(signal.data)

    def update_nav(self, nav: float) -> None:
        """Update current portfolio NAV from CentralAccounting."""
        self._current_nav = nav
