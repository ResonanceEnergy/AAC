"""
shared.alert_manager — Alert routing and deduplication for AAC.

Provides AlertManager for creating, routing, deduplicating, and
acknowledging operational alerts across all departments.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """AlertSeverity class."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """AlertStatus class."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert class."""
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    source: str
    category: str = "general"
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    count: int = 1


class AlertManager:
    """Central alert management with deduplication and routing."""

    def __init__(self, dedup_window_seconds: int = 300):
        self.alerts: Dict[str, Alert] = {}
        self._counter = 0
        self._dedup_window = timedelta(seconds=dedup_window_seconds)
        self._handlers: List[Callable] = []
        logger.info("AlertManager initialized")

    def register_handler(self, handler: Callable):
        """Register a callback invoked on new alerts."""
        self._handlers.append(handler)

    def fire(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        source: str = "system",
        category: str = "general",
    ) -> Alert:
        """Fire an alert with automatic deduplication."""
        dedup_key = f"{source}:{category}:{title}"

        # Deduplicate within window
        existing = self.alerts.get(dedup_key)
        if existing and existing.status == AlertStatus.ACTIVE:
            age = datetime.now() - existing.created_at
            if age < self._dedup_window:
                existing.count += 1
                logger.debug(f"Alert deduplicated ({existing.count}x): {title}")
                return existing

        self._counter += 1
        alert_id = f"ALERT-{self._counter:05d}"
        alert = Alert(
            alert_id=alert_id,
            title=title,
            message=message,
            severity=severity,
            source=source,
            category=category,
        )
        self.alerts[dedup_key] = alert
        logger.warning(f"Alert fired: {alert_id} [{severity.value}] {title}")

        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

        return alert

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts.values():
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False

    def resolve(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts.values():
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return [a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE]

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics."""
        total = len(self.alerts)
        active = len(self.get_active_alerts())
        return {
            "total_alerts": total,
            "active_alerts": active,
            "acknowledged": len([a for a in self.alerts.values() if a.status == AlertStatus.ACKNOWLEDGED]),
            "resolved": len([a for a in self.alerts.values() if a.status == AlertStatus.RESOLVED]),
        }
