"""Research Coordinator -- active intel feed from Research to Trading divisions.

Runs all research agents on a schedule, filters findings by confidence/urgency,
and publishes actionable signals to subscribed Trading divisions (War Room,
AgentBravo, Crypto, Polymarket).

Signal types pushed:
- INTEL_UPDATE:         Market data, flow signals, sentiment shifts
- STRATEGY_ADJUSTMENT:  Scenario weight changes, arm rebalance triggers
- MILESTONE_ALERT:      Proximity to $10M phase gates
- RISK_ALERT:           Geopolitical trigger, vol regime shift, counterparty risk
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
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

# Minimum confidence to publish a finding as a signal
CONFIDENCE_THRESHOLD = 0.5

# Urgency mapping: finding urgency string -> Signal urgency int
URGENCY_MAP: dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}


@dataclass
class ResearchFinding:
    """Standardized research finding -- canonical definition.

    Backward-compatible with BigBrainIntelligence.agents.ResearchFinding.
    """

    finding_id: str
    agent_id: str
    theater: str
    finding_type: str
    title: str
    description: str
    confidence: float
    urgency: str  # 'low', 'medium', 'high', 'critical'
    data: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "finding_id": self.finding_id,
            "agent_id": self.agent_id,
            "theater": self.theater,
            "finding_type": self.finding_type,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "data": self.data,
            "sources": self.sources,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class ResearchCoordinator(DivisionProtocol):
    """Research Department coordinator.

    Manages research agents, runs scans on schedule, filters findings,
    and pushes actionable intel to Trading divisions via the Division Protocol.
    """

    def __init__(self) -> None:
        super().__init__(division_name="research")
        self._agents: list[Any] = []  # BaseResearchAgent instances
        self._latest_findings: list[ResearchFinding] = []
        self._total_findings: int = 0
        self._total_published: int = 0

    # -- Agent management ----------------------------------------------------

    def register_agent(self, agent: Any) -> None:
        """Register a research agent (must have async scan() method)."""
        self._agents.append(agent)
        _log.info("research.agent_registered", agent=getattr(agent, "agent_id", str(agent)))

    # -- DivisionProtocol implementation ------------------------------------

    async def scan(self) -> list[Signal]:
        """Run all research agents, filter, convert to signals."""
        signals: list[Signal] = []
        findings: list[ResearchFinding] = []

        for agent in self._agents:
            try:
                agent_findings = await agent.scan()
                if agent_findings:
                    for f in agent_findings:
                        # Convert legacy ResearchFinding to our canonical type
                        if not isinstance(f, ResearchFinding):
                            f = self._convert_finding(f)
                        findings.append(f)
            except Exception:
                _log.warning("research.agent_scan_failed", agent=getattr(agent, "agent_id", "unknown"))

        self._latest_findings = findings
        self._total_findings += len(findings)

        # Filter and convert to signals
        for finding in findings:
            if finding.confidence >= CONFIDENCE_THRESHOLD:
                signal = self._finding_to_signal(finding)
                signals.append(signal)
                self._total_published += 1

        _log.info(
            "research.scan_complete",
            findings=len(findings),
            signals_published=len(signals),
            agents=len(self._agents),
        )
        return signals

    async def report(self) -> dict[str, Any]:
        """Status report."""
        return {
            "division": self.division_name,
            "health": self.health.value,
            "agents_registered": len(self._agents),
            "total_findings": self._total_findings,
            "total_published": self._total_published,
            "latest_findings_count": len(self._latest_findings),
            "cycle_count": self._cycle_count,
        }

    # -- Conversion helpers --------------------------------------------------

    def _finding_to_signal(self, finding: ResearchFinding) -> Signal:
        """Convert a ResearchFinding into an inter-division Signal."""
        # Map finding type to signal type
        signal_type = self._classify_signal(finding)
        return Signal(
            signal_type=signal_type,
            source_division=self.division_name,
            timestamp=finding.timestamp,
            data=finding.to_dict(),
            confidence=finding.confidence,
            urgency=URGENCY_MAP.get(finding.urgency, 0),
        )

    @staticmethod
    def _classify_signal(finding: ResearchFinding) -> SignalType:
        """Classify a research finding into a signal type."""
        urgency = finding.urgency.lower()
        ftype = finding.finding_type.lower()

        if urgency == "critical" or "risk" in ftype:
            return SignalType.RISK_ALERT
        if "milestone" in ftype or "phase" in ftype:
            return SignalType.MILESTONE_ALERT
        if "adjust" in ftype or "rebalance" in ftype or "weight" in ftype:
            return SignalType.STRATEGY_ADJUSTMENT
        return SignalType.INTEL_UPDATE

    @staticmethod
    def _convert_finding(legacy: Any) -> ResearchFinding:
        """Convert a legacy BigBrainIntelligence ResearchFinding."""
        return ResearchFinding(
            finding_id=getattr(legacy, "finding_id", ""),
            agent_id=getattr(legacy, "agent_id", ""),
            theater=getattr(legacy, "theater", ""),
            finding_type=getattr(legacy, "finding_type", ""),
            title=getattr(legacy, "title", ""),
            description=getattr(legacy, "description", ""),
            confidence=getattr(legacy, "confidence", 0.0),
            urgency=getattr(legacy, "urgency", "low"),
            data=getattr(legacy, "data", {}),
            sources=getattr(legacy, "sources", []),
            timestamp=getattr(legacy, "timestamp", datetime.now(timezone.utc)),
            expires_at=getattr(legacy, "expires_at", None),
        )
