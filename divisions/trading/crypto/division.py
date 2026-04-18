"""Crypto Division -- venue health, whale tracking, and crypto trading.

Wraps the CryptoIntelligence engine as a DivisionProtocol participant.
Reports venue health scores to War Room for crypto arm allocation.
Currently dormant (NDAX liquidated) but ready to reactivate.
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

_log = structlog.get_logger()


class CryptoDivision(DivisionProtocol):
    """Crypto intelligence and trading arm.

    Monitors venue health, whale activity, DeFi yields, on-chain
    analysis, scam detection, and MEV protection.
    """

    def __init__(self) -> None:
        super().__init__(division_name="crypto")
        self._venue_health: dict[str, float] = {}
        self._whale_alerts: list[dict[str, Any]] = []
        self._signals_sent: int = 0
        # Start as DEGRADED since no active positions post-NDAX liquidation
        self.health = DivisionHealth.DEGRADED

    async def scan(self) -> list[Signal]:
        """Scan crypto venues and on-chain activity."""
        signals: list[Signal] = []

        # Publish venue health to War Room
        if self._venue_health:
            signals.append(Signal(
                signal_type=SignalType.INTEL_UPDATE,
                source_division=self.division_name,
                timestamp=datetime.now(timezone.utc),
                data={"venue_health": self._venue_health},
                confidence=0.8,
                urgency=0,
            ))

        # Publish whale alerts
        for alert in self._whale_alerts:
            signals.append(Signal(
                signal_type=SignalType.RISK_ALERT,
                source_division=self.division_name,
                timestamp=datetime.now(timezone.utc),
                data=alert,
                confidence=0.7,
                urgency=2,
            ))

        self._whale_alerts.clear()
        self._signals_sent += len(signals)

        _log.info(
            "crypto.scan_complete",
            venues=len(self._venue_health),
            signals=len(signals),
        )
        return signals

    async def report(self) -> dict[str, Any]:
        """Status report."""
        return {
            "division": self.division_name,
            "health": self.health.value,
            "venue_count": len(self._venue_health),
            "signals_sent": self._signals_sent,
            "cycle_count": self._cycle_count,
        }

    async def consume_signal(self, signal: Signal) -> None:
        """Receive macro/research intel relevant to crypto markets."""
        await super().consume_signal(signal)
