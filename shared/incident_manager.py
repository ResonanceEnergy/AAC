"""
shared.incident_manager — Incident tracking and escalation for AAC.

Provides IncidentManager for recording, escalating, and resolving
operational incidents across all departments.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """IncidentSeverity class."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """IncidentStatus class."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"


@dataclass
class Incident:
    """Incident class."""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    source: str
    status: IncidentStatus = IncidentStatus.OPEN
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    assignee: Optional[str] = None
    notes: List[str] = field(default_factory=list)


class IncidentManager:
    """Central incident management for AAC operations."""

    def __init__(self):
        self.incidents: Dict[str, Incident] = {}
        self._counter = 0
        logger.info("IncidentManager initialized")

    def create_incident(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity = IncidentSeverity.MEDIUM,
        source: str = "system",
    ) -> Incident:
        """Create and register a new incident."""
        self._counter += 1
        incident_id = f"INC-{self._counter:05d}"
        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            source=source,
        )
        self.incidents[incident_id] = incident
        logger.warning(f"Incident created: {incident_id} [{severity.value}] {title}")
        return incident

    def escalate(self, incident_id: str, new_severity: IncidentSeverity) -> bool:
        """Escalate an incident to a higher severity."""
        incident = self.incidents.get(incident_id)
        if not incident:
            logger.error(f"Incident {incident_id} not found for escalation")
            return False
        old = incident.severity
        incident.severity = new_severity
        incident.updated_at = datetime.now()
        incident.notes.append(f"Escalated from {old.value} to {new_severity.value}")
        logger.warning(f"Incident {incident_id} escalated: {old.value} → {new_severity.value}")
        return True

    def resolve(self, incident_id: str, resolution_note: str = "") -> bool:
        """Resolve an incident."""
        incident = self.incidents.get(incident_id)
        if not incident:
            logger.error(f"Incident {incident_id} not found for resolution")
            return False
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.now()
        incident.updated_at = datetime.now()
        if resolution_note:
            incident.notes.append(f"Resolved: {resolution_note}")
        logger.info(f"Incident {incident_id} resolved")
        return True

    def get_open_incidents(self) -> List[Incident]:
        """Return all non-resolved incidents."""
        return [i for i in self.incidents.values() if i.status != IncidentStatus.RESOLVED]

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident."""
        return self.incidents.get(incident_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Return incident metrics summary."""
        total = len(self.incidents)
        open_count = len(self.get_open_incidents())
        return {
            "total_incidents": total,
            "open_incidents": open_count,
            "resolved_incidents": total - open_count,
            "by_severity": {
                s.value: len([i for i in self.incidents.values() if i.severity == s])
                for s in IncidentSeverity
            },
        }
