"""Division Protocol — base contract all AAC divisions implement.

Every division (Trading arms, Research) inherits this protocol to enable:
- Heartbeat monitoring (5s alive check)
- Health status reporting (HEALTHY / DEGRADED / OFFLINE)
- Inter-division signal publishing/consumption
- Doctrine-gated execution (NORMAL / CAUTION / SAFE_MODE / HALT)
"""
from __future__ import annotations

import asyncio
import enum
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

_log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DivisionHealth(enum.Enum):
    """Health status reported by every division."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class SignalType(enum.Enum):
    """Types of inter-division signals."""

    INTEL_UPDATE = "intel_update"
    STRATEGY_ADJUSTMENT = "strategy_adjustment"
    MILESTONE_ALERT = "milestone_alert"
    RISK_ALERT = "risk_alert"
    HEARTBEAT = "heartbeat"
    TRADE_SIGNAL = "trade_signal"


class DoctrineState(enum.Enum):
    """Doctrine enforcement states — gates all execution."""

    NORMAL = "normal"
    CAUTION = "caution"
    SAFE_MODE = "safe_mode"
    HALT = "halt"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DivisionStatus:
    """Snapshot of a division's current state."""

    division_name: str
    health: DivisionHealth
    uptime_seconds: float
    cycle_count: int
    last_heartbeat: datetime
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Inter-division communication payload."""

    signal_type: SignalType
    source_division: str
    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    urgency: int = 0  # 0=info, 1=attention, 2=action, 3=critical


# ---------------------------------------------------------------------------
# Division Protocol — abstract base
# ---------------------------------------------------------------------------

class DivisionProtocol(ABC):
    """Base protocol all AAC divisions implement.

    Lifecycle: start() → [heartbeat + scan loop] → stop()
    """

    def __init__(self, division_name: str) -> None:
        self.division_name = division_name
        self.health = DivisionHealth.OFFLINE
        self.doctrine_state = DoctrineState.NORMAL
        self._start_time: float = 0.0
        self._cycle_count: int = 0
        self._last_heartbeat = datetime.now(timezone.utc)
        self._subscribers: list[DivisionProtocol] = []
        self._running = False

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Bring division online."""
        self._start_time = time.monotonic()
        self._running = True
        self.health = DivisionHealth.HEALTHY
        _log.info("division.started", division=self.division_name)

    async def stop(self) -> None:
        """Take division offline gracefully."""
        self._running = False
        self.health = DivisionHealth.OFFLINE
        _log.info("division.stopped", division=self.division_name)

    # -- Heartbeat -----------------------------------------------------------

    async def heartbeat(self) -> DivisionStatus:
        """Prove alive — called every 5 seconds by orchestrator."""
        self._last_heartbeat = datetime.now(timezone.utc)
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
        status = DivisionStatus(
            division_name=self.division_name,
            health=self.health,
            uptime_seconds=uptime,
            cycle_count=self._cycle_count,
            last_heartbeat=self._last_heartbeat,
        )
        return status

    # -- Core work loop (divisions override these) ---------------------------

    @abstractmethod
    async def scan(self) -> list[Signal]:
        """Main work cycle — find opportunities, produce signals."""

    @abstractmethod
    async def report(self) -> dict[str, Any]:
        """Status report for dashboards and cross-division consumption."""

    # -- Inter-division communication ----------------------------------------

    def subscribe(self, division: DivisionProtocol) -> None:
        """Register another division to receive our signals."""
        if division not in self._subscribers:
            self._subscribers.append(division)

    async def publish(self, signal: Signal) -> None:
        """Broadcast a signal to all subscribers."""
        for sub in self._subscribers:
            try:
                await sub.consume_signal(signal)
            except Exception:
                _log.warning(
                    "signal.delivery_failed",
                    source=self.division_name,
                    target=sub.division_name,
                    signal_type=signal.signal_type.value,
                )

    async def consume_signal(self, signal: Signal) -> None:
        """Receive a signal from another division. Override to handle."""
        _log.debug(
            "signal.received",
            division=self.division_name,
            signal_type=signal.signal_type.value,
            source=signal.source_division,
        )

    # -- Doctrine enforcement ------------------------------------------------

    def set_doctrine_state(self, state: DoctrineState) -> None:
        """Update doctrine state — gates execution decisions."""
        prev = self.doctrine_state
        self.doctrine_state = state
        if prev != state:
            _log.warning(
                "doctrine.state_change",
                division=self.division_name,
                previous=prev.value,
                current=state.value,
            )

    def can_execute(self) -> bool:
        """Check if doctrine allows execution (not in HALT or SAFE_MODE)."""
        return self.doctrine_state in (DoctrineState.NORMAL, DoctrineState.CAUTION)

    # -- Run loop ------------------------------------------------------------

    async def run_loop(self, interval: float = 60.0) -> None:
        """Main division loop — heartbeat + scan at configured interval."""
        await self.start()
        try:
            while self._running:
                try:
                    await self.heartbeat()
                    signals = await self.scan()
                    self._cycle_count += 1
                    for sig in signals:
                        await self.publish(sig)
                except Exception:
                    self.health = DivisionHealth.DEGRADED
                    _log.exception("division.scan_error", division=self.division_name)
                await asyncio.sleep(interval)
        finally:
            await self.stop()
