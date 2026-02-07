#!/usr/bin/env python3
"""
Incident and Postmortem Automation
==================================

Automated monitoring and incident creation for doctrine compliance:

- Audit Gap Detection: Monitors audit log completeness (< 99% triggers incident)
- Postmortem Completeness: Ensures postmortems are completed for incidents > 7 days old
- Incident Lifecycle Management: Tracks incident age, status, and required actions

This automation ensures compliance with doctrine pack failure modes.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from shared.audit_logger import get_audit_logger, AuditCategory, AuditSeverity

logger = logging.getLogger(__name__)


@dataclass
class IncidentRecord:
    """Record of an incident and its lifecycle."""
    incident_id: str
    created_at: datetime
    severity: str  # 'sev1', 'sev2', 'sev3'
    description: str
    status: str  # 'active', 'resolved', 'postmortem_pending', 'closed'
    postmortem_completed: bool = False
    postmortem_due_date: Optional[datetime] = None
    action_items: List[str] = None
    root_cause: Optional[str] = None

    def __post_init__(self):
        if self.action_items is None:
            self.action_items = []
        if self.postmortem_due_date is None and self.status == 'resolved':
            # Postmortem due 7 days after resolution
            self.postmortem_due_date = self.created_at + timedelta(days=7)

    @property
    def age_days(self) -> float:
        """Get incident age in days."""
        return (datetime.now() - self.created_at).total_seconds() / (24 * 3600)

    @property
    def is_postmortem_overdue(self) -> bool:
        """Check if postmortem is overdue."""
        if self.status != 'resolved' or self.postmortem_completed:
            return False
        return datetime.now() > (self.postmortem_due_date or self.created_at + timedelta(days=7))

    @property
    def is_postmortem_incomplete(self) -> bool:
        """Check if postmortem is incomplete."""
        if self.status != 'resolved':
            return False
        return not self.postmortem_completed and self.is_postmortem_overdue


class IncidentPostmortemAutomation:
    """
    Automates incident management and postmortem compliance.

    Monitors for doctrine-defined failure modes:
    - Audit gap: audit_log_completeness < 99%
    - Incomplete postmortem: postmortem_incomplete && incident_age > 7 days
    """

    def __init__(self):
        self.audit = get_audit_logger()
        self.incidents: Dict[str, IncidentRecord] = {}
        self.audit_completeness_threshold = 99.0  # %
        self.monitoring_active = False

    async def start_monitoring(self) -> None:
        """Start the automated monitoring loops."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        logger.info("Starting incident and postmortem automation monitoring")

        # Start monitoring tasks
        asyncio.create_task(self._monitor_audit_completeness())
        asyncio.create_task(self._monitor_postmortem_completeness())
        asyncio.create_task(self._cleanup_old_incidents())

        await self._audit_event("monitoring_started", "info")

    async def stop_monitoring(self) -> None:
        """Stop the automated monitoring."""
        self.monitoring_active = False
        logger.info("Stopped incident and postmortem automation monitoring")
        await self._audit_event("monitoring_stopped", "info")

    async def _monitor_audit_completeness(self) -> None:
        """Monitor audit log completeness and create incidents for gaps."""
        while self.monitoring_active:
            try:
                # Get current audit completeness (would integrate with actual audit system)
                completeness = await self._get_audit_completeness()

                if completeness < self.audit_completeness_threshold:
                    gap_percentage = self.audit_completeness_threshold - completeness
                    await self._create_audit_gap_incident(completeness, gap_percentage)

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error monitoring audit completeness: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute

    async def _monitor_postmortem_completeness(self) -> None:
        """Monitor for incomplete postmortems on old incidents."""
        while self.monitoring_active:
            try:
                incomplete_count = 0

                for incident_id, incident in self.incidents.items():
                    if incident.is_postmortem_incomplete:
                        incomplete_count += 1
                        await self._create_incomplete_postmortem_incident(incident)

                if incomplete_count > 0:
                    logger.warning(f"Found {incomplete_count} incidents with incomplete postmortems")

                await asyncio.sleep(3600)  # Check hourly

            except Exception as e:
                logger.error(f"Error monitoring postmortem completeness: {e}")
                await asyncio.sleep(300)  # Retry after 5 minutes

    async def _cleanup_old_incidents(self) -> None:
        """Clean up old closed incidents."""
        while self.monitoring_active:
            try:
                cutoff_date = datetime.now() - timedelta(days=90)  # Keep 90 days of history
                to_remove = []

                for incident_id, incident in self.incidents.items():
                    if incident.status == 'closed' and incident.created_at < cutoff_date:
                        to_remove.append(incident_id)

                for incident_id in to_remove:
                    del self.incidents[incident_id]
                    logger.info(f"Cleaned up old incident: {incident_id}")

                await asyncio.sleep(86400)  # Clean up daily

            except Exception as e:
                logger.error(f"Error cleaning up old incidents: {e}")
                await asyncio.sleep(3600)  # Retry after 1 hour

    async def _get_audit_completeness(self) -> float:
        """Get current audit log completeness percentage."""
        # This would integrate with the actual audit logging system
        # For now, simulate based on system health
        try:
            # Simulate audit completeness check
            # In real implementation, this would query audit logs
            base_completeness = 99.5  # Assume generally good

            # Add some variance based on "system load"
            import random
            variance = random.uniform(-0.5, 0.5)
            completeness = max(95.0, min(100.0, base_completeness + variance))

            return completeness

        except Exception as e:
            logger.warning(f"Failed to get audit completeness: {e}")
            return 98.0  # Conservative fallback

    async def _create_audit_gap_incident(self, completeness: float, gap_percentage: float) -> None:
        """Create an incident for audit log gaps."""
        incident_id = f"AUDIT-GAP-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        incident = IncidentRecord(
            incident_id=incident_id,
            created_at=datetime.now(),
            severity='sev2',  # Audit gaps are serious but not critical
            description=f"Audit log completeness below threshold: {completeness:.1f}% (gap: {gap_percentage:.1f}%)",
            status='active'
        )

        self.incidents[incident_id] = incident

        # Create the incident via doctrine system
        await self._trigger_doctrine_incident(
            incident_type="audit_gap",
            description=incident.description,
            severity=incident.severity,
            context={
                'audit_completeness': completeness,
                'gap_percentage': gap_percentage,
                'threshold': self.audit_completeness_threshold
            }
        )

        logger.warning(f"Created audit gap incident: {incident_id}")
        await self._audit_event("audit_gap_incident_created", "warning",
                               incident_id=incident_id, completeness=completeness)

    async def _create_incomplete_postmortem_incident(self, original_incident: IncidentRecord) -> None:
        """Create an incident for incomplete postmortem."""
        incident_id = f"POSTMORTEM-{original_incident.incident_id}-{datetime.now().strftime('%Y%m%d')}"

        # Only create one postmortem incident per original incident
        if any(inc.incident_id.startswith(f"POSTMORTEM-{original_incident.incident_id}")
               for inc in self.incidents.values()):
            return

        incident = IncidentRecord(
            incident_id=incident_id,
            created_at=datetime.now(),
            severity='sev3',  # Postmortem issues are important but not critical
            description=f"Incomplete postmortem for incident {original_incident.incident_id} (age: {original_incident.age_days:.1f} days)",
            status='active'
        )

        self.incidents[incident_id] = incident

        # Create the incident via doctrine system
        await self._trigger_doctrine_incident(
            incident_type="incomplete_postmortem",
            description=incident.description,
            severity=incident.severity,
            context={
                'original_incident_id': original_incident.incident_id,
                'original_incident_age_days': original_incident.age_days,
                'postmortem_due_date': original_incident.postmortem_due_date.isoformat() if original_incident.postmortem_due_date else None
            }
        )

        logger.warning(f"Created incomplete postmortem incident: {incident_id}")
        await self._audit_event("incomplete_postmortem_incident_created", "warning",
                               incident_id=incident_id, original_incident=original_incident.incident_id)

    async def _trigger_doctrine_incident(self, incident_type: str, description: str,
                                       severity: str, context: Dict[str, Any]) -> None:
        """Trigger incident creation through the doctrine system."""
        try:
            # Import doctrine orchestrator
            from aac.doctrine.doctrine_integration import get_doctrine_orchestrator

            orchestrator = await get_doctrine_orchestrator()

            # Create incident via doctrine action
            from aac.doctrine.doctrine_engine import ActionType

            await orchestrator.execute_action(
                ActionType.A_CREATE_INCIDENT,
                {
                    'incident_type': incident_type,
                    'description': description,
                    'severity': severity,
                    'automation_source': 'incident_postmortem_automation',
                    'context': context
                }
            )

        except Exception as e:
            logger.error(f"Failed to trigger doctrine incident: {e}")
            # Fallback: log the incident anyway
            await self._audit_event("doctrine_incident_trigger_failed", "error",
                                   incident_type=incident_type, error=str(e))

    async def record_incident_resolution(self, incident_id: str, resolution_details: Dict[str, Any]) -> None:
        """Record that an incident has been resolved."""
        if incident_id not in self.incidents:
            logger.warning(f"Unknown incident for resolution: {incident_id}")
            return

        incident = self.incidents[incident_id]
        incident.status = 'resolved'

        # Set postmortem due date
        incident.postmortem_due_date = datetime.now() + timedelta(days=7)

        logger.info(f"Recorded resolution for incident: {incident_id}")
        await self._audit_event("incident_resolved", "info",
                               incident_id=incident_id, resolution_details=resolution_details)

    async def complete_postmortem(self, incident_id: str, action_items: List[str],
                                root_cause: str) -> None:
        """Mark postmortem as completed for an incident."""
        if incident_id not in self.incidents:
            logger.warning(f"Unknown incident for postmortem completion: {incident_id}")
            return

        incident = self.incidents[incident_id]
        incident.postmortem_completed = True
        incident.action_items = action_items
        incident.root_cause = root_cause
        incident.status = 'closed'

        logger.info(f"Completed postmortem for incident: {incident_id}")
        await self._audit_event("postmortem_completed", "info",
                               incident_id=incident_id, action_items_count=len(action_items))

    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and metrics."""
        total_incidents = len(self.incidents)
        active_incidents = len([inc for inc in self.incidents.values() if inc.status == 'active'])
        resolved_incidents = len([inc for inc in self.incidents.values() if inc.status == 'resolved'])
        incomplete_postmortems = len([inc for inc in self.incidents.values() if inc.is_postmortem_incomplete])

        return {
            'monitoring_active': self.monitoring_active,
            'total_incidents': total_incidents,
            'active_incidents': active_incidents,
            'resolved_incidents': resolved_incidents,
            'incomplete_postmortems': incomplete_postmortems,
            'audit_completeness_threshold': self.audit_completeness_threshold,
            'timestamp': datetime.now().isoformat()
        }

    async def _audit_event(self, event_type: str, severity: str, **kwargs) -> None:
        """Audit automation events."""
        await self.audit.log_event(
            category=AuditCategory.SYSTEM,
            action=event_type,
            resource="incident_automation",
            status="info" if severity == "info" else "success",
            severity=getattr(AuditSeverity, severity.upper()),
            user="system",
            details={
                'event_type': event_type,
                'automation_component': 'incident_postmortem_automation',
                'timestamp': datetime.now().isoformat(),
                **kwargs
            }
        )


# Global automation instance
_automation_instance: Optional[IncidentPostmortemAutomation] = None


async def get_incident_automation() -> IncidentPostmortemAutomation:
    """Get or create the global incident automation instance."""
    global _automation_instance
    if _automation_instance is None:
        _automation_instance = IncidentPostmortemAutomation()
        await _automation_instance.start_monitoring()
    return _automation_instance


async def start_incident_monitoring() -> None:
    """Convenience function to start incident monitoring."""
    automation = await get_incident_automation()
    await automation.start_monitoring()


async def get_monitoring_status() -> Dict[str, Any]:
    """Convenience function to get monitoring status."""
    automation = await get_incident_automation()
    return await automation.get_monitoring_status()