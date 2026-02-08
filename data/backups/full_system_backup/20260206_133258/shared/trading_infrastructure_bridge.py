#!/usr/bin/env python3
"""
TradingExecution ↔ SharedInfrastructure Bridge
==============================================

Bridge between TradingExecution and SharedInfrastructure departments
for execution monitoring, incident reporting, and audit logging.

This bridge enables:
- Real-time execution monitoring and alerting
- Incident reporting and escalation
- Audit logging of all trading activities
- Infrastructure health monitoring for trading
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from shared.bridge_orchestrator import BridgeMessage, BridgeMessageType, MessagePriority, Department
from shared.audit_logger import get_audit_logger, AuditCategory, AuditSeverity

logger = logging.getLogger(__name__)


class TradingInfrastructureBridge:
    """
    Bridge between TradingExecution and SharedInfrastructure departments.
    Handles execution monitoring, incident reporting, and audit logging.
    """

    def __init__(self):
        self.audit_logger = get_audit_logger()

        # Bridge state
        self.is_initialized = False
        self.last_execution_monitor = None
        self.last_incident_report = None

        # Monitoring state
        self.active_monitors: Dict[str, Dict] = {}
        self.incident_queue: List[Dict] = []

        # Audit trail
        self.audit_buffer: List[Dict] = []
        self.audit_batch_size = 100

        # Performance metrics
        self.performance_metrics = {
            "executions_monitored": 0,
            "incidents_reported": 0,
            "audit_logs_processed": 0,
            "alerts_sent": 0
        }

    async def initialize(self) -> bool:
        """Initialize the bridge."""
        try:
            logger.info("Initializing TradingExecution ↔ SharedInfrastructure bridge")

            # Initialize state
            self.is_initialized = True

            await self.audit_logger.log_event(
                category=AuditCategory.SYSTEM,
                action="bridge_initialized",
                resource="trading_infrastructure_bridge",
                severity=AuditSeverity.INFO,
                details={"bridge_type": "trading_infrastructure"}
            )

            logger.info("TradingExecution ↔ SharedInfrastructure bridge initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize trading-infrastructure bridge: {e}")
            return False

    async def handle_message(self, message: BridgeMessage) -> bool:
        """Handle incoming bridge messages."""
        try:
            if message.message_type == BridgeMessageType.EXECUTION_MONITORING:
                return await self._handle_execution_monitoring(message)
            elif message.message_type == BridgeMessageType.INCIDENT_REPORTING:
                return await self._handle_incident_reporting(message)
            elif message.message_type == BridgeMessageType.AUDIT_LOGGING:
                return await self._handle_audit_logging(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return False

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return False

    async def _handle_execution_monitoring(self, message: BridgeMessage) -> bool:
        """Handle execution monitoring requests."""
        try:
            monitoring_data = message.data
            execution_id = monitoring_data.get("execution_id")
            metrics = monitoring_data.get("metrics", {})

            # Set up monitoring for execution
            monitor_id = await self._setup_execution_monitor(execution_id, metrics)

            if monitor_id:
                self.last_execution_monitor = datetime.now()
                self.performance_metrics["executions_monitored"] += 1

                await self.audit_logger.log_event(
                    category=AuditCategory.TRADING,
                    action="execution_monitoring_started",
                    resource="trading_infrastructure_bridge",
                    severity=AuditSeverity.INFO,
                    details={
                        "execution_id": execution_id,
                        "monitor_id": monitor_id,
                        "metrics": list(metrics.keys())
                    }
                )

                logger.info(f"Started monitoring execution {execution_id}")
                return True
            else:
                logger.error(f"Failed to setup monitoring for execution {execution_id}")
                return False

        except Exception as e:
            logger.error(f"Error handling execution monitoring: {e}")
            return False

    async def _handle_incident_reporting(self, message: BridgeMessage) -> bool:
        """Handle incident reporting."""
        try:
            incident_data = message.data
            incident_id = incident_data.get("incident_id")
            severity = incident_data.get("severity", "medium")
            description = incident_data.get("description", "")

            # Report incident to infrastructure
            report_result = await self._report_incident(incident_id, severity, description, incident_data)

            if report_result:
                self.last_incident_report = datetime.now()
                self.performance_metrics["incidents_reported"] += 1

                await self.audit_logger.log_event(
                    category=AuditCategory.TRADING,
                    action="incident_reported",
                    resource="trading_infrastructure_bridge",
                    severity=AuditSeverity.WARNING if severity in ["high", "critical"] else AuditSeverity.INFO,
                    details={
                        "incident_id": incident_id,
                        "severity": severity,
                        "description": description
                    }
                )

                logger.info(f"Reported incident {incident_id} with severity {severity}")
                return True
            else:
                logger.error(f"Failed to report incident {incident_id}")
                return False

        except Exception as e:
            logger.error(f"Error handling incident reporting: {e}")
            return False

    async def _handle_audit_logging(self, message: BridgeMessage) -> bool:
        """Handle audit logging requests."""
        try:
            audit_data = message.data
            activity_type = audit_data.get("activity_type")
            details = audit_data.get("details", {})

            # Add to audit buffer
            audit_entry = {
                "timestamp": datetime.now(),
                "activity_type": activity_type,
                "details": details,
                "source": "trading_execution"
            }

            self.audit_buffer.append(audit_entry)
            self.performance_metrics["audit_logs_processed"] += 1

            # Flush buffer if it reaches batch size
            if len(self.audit_buffer) >= self.audit_batch_size:
                await self._flush_audit_buffer()

            await self.audit_logger.log_event(
                category=AuditCategory.TRADING,
                action="audit_log_processed",
                resource="trading_infrastructure_bridge",
                severity=AuditSeverity.INFO,
                details={
                    "activity_type": activity_type,
                    "buffer_size": len(self.audit_buffer)
                }
            )

            logger.info(f"Processed audit log for activity: {activity_type}")
            return True

        except Exception as e:
            logger.error(f"Error handling audit logging: {e}")
            return False

    async def _setup_execution_monitor(self, execution_id: str, metrics: Dict) -> Optional[str]:
        """Setup monitoring for an execution."""
        try:
            monitor_id = f"monitor_{execution_id}_{int(datetime.now().timestamp())}"

            self.active_monitors[monitor_id] = {
                "execution_id": execution_id,
                "metrics": metrics,
                "start_time": datetime.now(),
                "status": "active",
                "alerts_triggered": 0
            }

            # Start monitoring task
            asyncio.create_task(self._monitor_execution(monitor_id))

            return monitor_id

        except Exception as e:
            logger.error(f"Error setting up execution monitor: {e}")
            return None

    async def _monitor_execution(self, monitor_id: str):
        """Monitor an active execution."""
        try:
            monitor_data = self.active_monitors.get(monitor_id)
            if not monitor_data:
                return

            # Monitor for 5 minutes (example duration)
            monitor_duration = timedelta(minutes=5)
            end_time = monitor_data["start_time"] + monitor_duration

            while datetime.now() < end_time and monitor_data["status"] == "active":
                # Check execution health (placeholder logic)
                await self._check_execution_health(monitor_id)

                await asyncio.sleep(30)  # Check every 30 seconds

            # Mark monitor as completed
            monitor_data["status"] = "completed"
            monitor_data["end_time"] = datetime.now()

        except Exception as e:
            logger.error(f"Error monitoring execution {monitor_id}: {e}")

    async def _check_execution_health(self, monitor_id: str):
        """Check health of monitored execution."""
        # Placeholder - would check actual execution metrics
        # If issues detected, trigger alerts
        pass

    async def _report_incident(self, incident_id: str, severity: str, description: str, incident_data: Dict) -> bool:
        """Report incident to infrastructure."""
        try:
            incident_report = {
                "incident_id": incident_id,
                "severity": severity,
                "description": description,
                "data": incident_data,
                "reported_at": datetime.now(),
                "source": "trading_execution"
            }

            self.incident_queue.append(incident_report)

            # Process incident (placeholder - would escalate based on severity)
            await self._process_incident(incident_report)

            return True

        except Exception as e:
            logger.error(f"Error reporting incident: {e}")
            return False

    async def _process_incident(self, incident_report: Dict):
        """Process reported incident."""
        severity = incident_report.get("severity", "low")

        # Escalate based on severity
        if severity in ["high", "critical"]:
            # Immediate escalation
            self.performance_metrics["alerts_sent"] += 1
            logger.warning(f"CRITICAL INCIDENT: {incident_report['description']}")
        elif severity == "medium":
            # Standard processing
            logger.info(f"Incident reported: {incident_report['description']}")
        else:
            # Low priority
            logger.debug(f"Minor incident: {incident_report['description']}")

    async def _flush_audit_buffer(self):
        """Flush accumulated audit logs."""
        try:
            if not self.audit_buffer:
                return

            # Batch process audit logs (placeholder - would send to infrastructure)
            batch_size = len(self.audit_buffer)
            logger.info(f"Flushing {batch_size} audit log entries")

            # Clear buffer
            self.audit_buffer.clear()

        except Exception as e:
            logger.error(f"Error flushing audit buffer: {e}")

    async def get_bridge_health(self) -> Dict[str, Any]:
        """Get bridge health status."""
        return {
            "is_initialized": self.is_initialized,
            "last_execution_monitor": self.last_execution_monitor.isoformat() if self.last_execution_monitor else None,
            "last_incident_report": self.last_incident_report.isoformat() if self.last_incident_report else None,
            "active_monitors": len(self.active_monitors),
            "incident_queue_size": len(self.incident_queue),
            "audit_buffer_size": len(self.audit_buffer),
            "performance_metrics": self.performance_metrics
        }

    async def shutdown(self):
        """Shutdown the bridge."""
        logger.info("Shutting down TradingExecution ↔ SharedInfrastructure bridge")
        # Flush any remaining audit logs
        await self._flush_audit_buffer()
        self.is_initialized = False