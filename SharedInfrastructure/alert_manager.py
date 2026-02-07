"""
Shared Infrastructure - Alert Manager
Manages system alerts and notifications across all departments.
"""

import asyncio
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from SharedInfrastructure.audit_logger import AuditLogger

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"

@dataclass
class Alert:
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    source: str
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AlertRule:
    rule_id: str
    name: str
    condition: str  # Expression to evaluate
    severity: AlertSeverity
    message_template: str
    enabled: bool = True
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None

class AlertManager:
    """
    Manages system alerts, notifications, and escalation policies.
    """

    def __init__(self):
        self.audit_logger = AuditLogger()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}

        # Notification settings
        self.notification_channels = {
            "email": self._send_email_notification,
            "log": self._send_log_notification,
            "console": self._send_console_notification
        }

        self.enabled_channels = ["log", "console"]  # Default channels

        # Email configuration (would be loaded from config)
        self.email_config = {
            "smtp_server": "localhost",
            "smtp_port": 587,
            "sender_email": "alerts@acc-system.local",
            "recipient_emails": ["ops@acc-system.local"]
        }

        # Escalation policies
        self.escalation_policies = {
            AlertSeverity.INFO: {"channels": ["log"]},
            AlertSeverity.WARNING: {"channels": ["log", "console"]},
            AlertSeverity.ERROR: {"channels": ["log", "console", "email"]},
            AlertSeverity.CRITICAL: {"channels": ["log", "console", "email"], "escalate_after_minutes": 5}
        }

        # Initialize default alert rules
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                condition="system.cpu_usage > 80",
                severity=AlertSeverity.WARNING,
                message_template="CPU usage is {value:.1f}%, above threshold of 80%",
                cooldown_minutes=10
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                condition="system.memory_usage > 85",
                severity=AlertSeverity.ERROR,
                message_template="Memory usage is {value:.1f}%, above threshold of 85%",
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id="component_down",
                name="Component Down",
                condition="component.status == 'DOWN'",
                severity=AlertSeverity.CRITICAL,
                message_template="Component {component} is down: {message}",
                cooldown_minutes=1
            ),
            AlertRule(
                rule_id="security_threat",
                name="Security Threat Detected",
                condition="security.events_detected > 0",
                severity=AlertSeverity.CRITICAL,
                message_template="Security threat detected: {details}",
                cooldown_minutes=0  # No cooldown for security alerts
            )
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

    def create_alert(self, title: str, message: str, severity: AlertSeverity,
                    source: str, tags: Optional[Dict[str, str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new alert.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            source: Alert source component
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Alert ID
        """
        alert_id = f"{source}_{int(datetime.now().timestamp())}_{hash(title) % 10000}"

        alert = Alert(
            alert_id=alert_id,
            title=title,
            message=message,
            severity=severity,
            source=source,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=tags or {},
            metadata=metadata or {}
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Send notifications
        asyncio.create_task(self._send_notifications(alert))

        # Log alert creation
        self.audit_logger.log_event(
            "alert_manager",
            "alert_created",
            f"Alert created: {title} ({severity.value})",
            severity.value
        )

        return alert_id

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User who acknowledged the alert
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by
            alert.updated_at = datetime.now()

            self.audit_logger.log_event(
                "alert_manager",
                "alert_acknowledged",
                f"Alert {alert_id} acknowledged by {acknowledged_by}",
                "info"
            )

    async def resolve_alert(self, alert_id: str, resolved_by: str):
        """
        Resolve an alert.

        Args:
            alert_id: Alert ID to resolve
            resolved_by: User who resolved the alert
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.resolved_by = resolved_by
            alert.updated_at = datetime.now()

            # Remove from active alerts
            del self.active_alerts[alert_id]

            self.audit_logger.log_event(
                "alert_manager",
                "alert_resolved",
                f"Alert {alert_id} resolved by {resolved_by}",
                "info"
            )

    async def check_alert_rules(self, metrics_context: Dict[str, Any]):
        """
        Check alert rules against current metrics.

        Args:
            metrics_context: Current system metrics
        """
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            # Check cooldown
            if rule.last_triggered:
                cooldown_end = rule.last_triggered + timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() < cooldown_end:
                    continue

            # Evaluate condition
            if self._evaluate_condition(rule.condition, metrics_context):
                # Create alert
                message = rule.message_template.format(**metrics_context)
                alert_id = self.create_alert(
                    title=rule.name,
                    message=message,
                    severity=rule.severity,
                    source="alert_rule",
                    tags={"rule_id": rule.rule_id},
                    metadata={"rule": rule.rule_id, "condition": rule.condition}
                )

                rule.last_triggered = datetime.now()

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate an alert condition expression.

        Args:
            condition: Condition expression
            context: Context variables

        Returns:
            True if condition is met
        """
        try:
            # Simple expression evaluator (in production, use a proper expression parser)
            # This is a basic implementation - in reality, you'd want a safer expression evaluator

            # Replace variable references with values
            eval_condition = condition
            for key, value in context.items():
                if isinstance(value, str):
                    eval_condition = eval_condition.replace(key, f"'{value}'")
                else:
                    eval_condition = eval_condition.replace(key, str(value))

            # Evaluate the condition
            return bool(eval(eval_condition))

        except Exception:
            return False

    async def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels."""
        policy = self.escalation_policies.get(alert.severity, {"channels": ["log"]})
        channels = policy.get("channels", ["log"])

        # Filter to enabled channels
        channels = [c for c in channels if c in self.enabled_channels]

        for channel in channels:
            try:
                notification_func = self.notification_channels.get(channel)
                if notification_func:
                    await notification_func(alert)
            except Exception as e:
                self.audit_logger.log_event(
                    "alert_manager",
                    "notification_failed",
                    f"Failed to send {channel} notification for alert {alert.alert_id}: {str(e)}",
                    "error"
                )

        # Handle escalation
        escalate_after = policy.get("escalate_after_minutes")
        if escalate_after and alert.status == AlertStatus.ACTIVE:
            asyncio.create_task(self._escalate_alert(alert, escalate_after))

    async def _escalate_alert(self, alert: Alert, delay_minutes: int):
        """Escalate an alert after delay if still active."""
        await asyncio.sleep(delay_minutes * 60)

        # Check if alert is still active
        if (alert.alert_id in self.active_alerts and
            self.active_alerts[alert.alert_id].status == AlertStatus.ACTIVE):

            # Escalate by sending to additional channels
            escalation_channels = ["email"]  # Could be configured

            for channel in escalation_channels:
                if channel not in self.enabled_channels:
                    continue

                try:
                    notification_func = self.notification_channels.get(channel)
                    if notification_func:
                        alert_copy = alert
                        alert_copy.title = f"ESCALATED: {alert.title}"
                        await notification_func(alert_copy)
                except Exception as e:
                    self.audit_logger.log_event(
                        "alert_manager",
                        "escalation_failed",
                        f"Failed to escalate alert {alert.alert_id}: {str(e)}",
                        "error"
                    )

    async def _send_email_notification(self, alert: Alert):
        """Send email notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipient_emails'])
            msg['Subject'] = f"ACC Alert: {alert.title}"

            body = f"""
ACC System Alert

Severity: {alert.severity.value.upper()}
Source: {alert.source}
Time: {alert.created_at.isoformat()}

{alert.message}

Status: {alert.status.value}
Alert ID: {alert.alert_id}
"""

            if alert.tags:
                body += f"\nTags: {alert.tags}"

            msg.attach(MIMEText(body, 'plain'))

            # Note: In production, this would actually send the email
            # For now, just log it
            self.audit_logger.log_event(
                "alert_manager",
                "email_notification",
                f"Email notification sent for alert {alert.alert_id}",
                "info"
            )

        except Exception as e:
            raise Exception(f"Email notification failed: {str(e)}")

    async def _send_log_notification(self, alert: Alert):
        """Send log notification."""
        self.audit_logger.log_event(
            "alert_manager",
            "alert_notification",
            f"Alert: {alert.title} - {alert.message}",
            alert.severity.value
        )

    async def _send_console_notification(self, alert: Alert):
        """Send console notification."""
        print(f"[ALERT] ALERT [{alert.severity.value.upper()}]: {alert.title}")
        print(f"   {alert.message}")
        print(f"   Source: {alert.source} | Time: {alert.created_at}")
        print()

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "by_severity": {
                severity.value: 0 for severity in AlertSeverity
            },
            "by_status": {
                status.value: 0 for status in AlertStatus
            }
        }

        for alert in self.active_alerts.values():
            summary["by_severity"][alert.severity.value] += 1
            summary["by_status"][alert.status.value] += 1

        for alert in self.alert_history:
            summary["by_status"][alert.status.value] += 1

        return summary

    def configure_email(self, smtp_server: str, smtp_port: int,
                       sender_email: str, recipient_emails: List[str]):
        """Configure email notifications."""
        self.email_config.update({
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "sender_email": sender_email,
            "recipient_emails": recipient_emails
        })

        if "email" not in self.enabled_channels:
            self.enabled_channels.append("email")

# Global alert manager instance
alert_manager = AlertManager()

async def get_alert_manager():
    """Get the global alert manager instance."""
    return alert_manager