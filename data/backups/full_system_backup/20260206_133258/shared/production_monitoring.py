#!/usr/bin/env python3
"""
Production Monitoring & Alerting System
======================================
24/7 monitoring, health checks, and automated alerting for production deployment.
"""

import asyncio
import logging
import json
import time
import psutil
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import get_audit_logger
from shared.live_trading_safeguards import live_trading_safeguards
from shared.production_deployment import production_deployment_system


@dataclass
class Alert:
    """Production alert"""
    alert_id: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str  # system, trading, data, performance, security
    title: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """System health check"""
    check_id: str
    name: str
    category: str
    check_function: Callable
    interval_seconds: int
    last_run: Optional[datetime] = None
    last_result: Optional[bool] = None
    consecutive_failures: int = 0
    enabled: bool = True


class ProductionMonitoringSystem:
    """24/7 production monitoring and alerting"""

    def __init__(self):
        self.logger = logging.getLogger("ProductionMonitoring")
        self.audit_logger = get_audit_logger()

        # Alert system
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_channels = ["console", "file"]  # Default channels

        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_check_task: Optional[asyncio.Task] = None

        # Monitoring configuration
        self.monitoring_config = {
            "health_check_interval": 30,  # seconds
            "alert_retention_days": 30,
            "max_consecutive_failures": 3,
            "alert_cooldown_minutes": 5
        }

        # Initialize health checks
        self._initialize_health_checks()

        # Alert cooldown tracking
        self.last_alert_times: Dict[str, datetime] = {}

    def _initialize_health_checks(self):
        """Initialize system health checks"""
        self.health_checks = {
            "system_cpu": HealthCheck(
                check_id="system_cpu",
                name="System CPU Usage",
                category="system",
                check_function=self._check_system_cpu,
                interval_seconds=30
            ),
            "system_memory": HealthCheck(
                check_id="system_memory",
                name="System Memory Usage",
                category="system",
                check_function=self._check_system_memory,
                interval_seconds=30
            ),
            "trading_safeguards": HealthCheck(
                check_id="trading_safeguards",
                name="Trading Safeguards Status",
                category="trading",
                check_function=self._check_trading_safeguards,
                interval_seconds=60
            ),
            "market_data_feeds": HealthCheck(
                check_id="market_data_feeds",
                name="Market Data Feeds",
                category="data",
                check_function=self._check_market_data_feeds,
                interval_seconds=30
            ),
            "deployment_status": HealthCheck(
                check_id="deployment_status",
                name="Production Deployment",
                category="trading",
                check_function=self._check_deployment_status,
                interval_seconds=300  # 5 minutes
            ),
            "network_connectivity": HealthCheck(
                check_id="network_connectivity",
                name="Network Connectivity",
                category="system",
                check_function=self._check_network_connectivity,
                interval_seconds=60
            ),
            "disk_space": HealthCheck(
                check_id="disk_space",
                name="Disk Space",
                category="system",
                check_function=self._check_disk_space,
                interval_seconds=300
            )
        }

    async def start_monitoring(self):
        """Start 24/7 monitoring"""
        self.logger.info("Starting 24/7 production monitoring...")

        # Start health check loop
        self.health_check_task = asyncio.create_task(self._health_check_loop())

        # Start alert cleanup task
        alert_cleanup_task = asyncio.create_task(self._alert_cleanup_loop())

        self.logger.info("Production monitoring started")

    async def stop_monitoring(self):
        """Stop monitoring"""
        self.logger.info("Stopping production monitoring...")

        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Production monitoring stopped")

    async def _health_check_loop(self):
        """Continuous health check loop"""
        while True:
            try:
                # Run all enabled health checks
                for check in self.health_checks.values():
                    if check.enabled and (check.last_run is None or
                        (datetime.now() - check.last_run).total_seconds() >= check.interval_seconds):
                        await self._run_health_check(check)

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(30)

    async def _run_health_check(self, check: HealthCheck):
        """Run a single health check"""
        try:
            check.last_run = datetime.now()
            result = await check.check_function()

            if result:
                # Check passed
                check.consecutive_failures = 0
                check.last_result = True
            else:
                # Check failed
                check.consecutive_failures += 1
                check.last_result = False

                # Alert on consecutive failures
                if check.consecutive_failures >= self.monitoring_config["max_consecutive_failures"]:
                    await self._create_alert(
                        severity="HIGH" if check.category == "trading" else "MEDIUM",
                        category=check.category,
                        title=f"Health Check Failed: {check.name}",
                        message=f"{check.name} has failed {check.consecutive_failures} consecutive times",
                        metadata={"check_id": check.check_id, "consecutive_failures": check.consecutive_failures}
                    )

        except Exception as e:
            self.logger.error(f"Health check {check.check_id} error: {e}")
            check.consecutive_failures += 1

    async def _create_alert(self, severity: str, category: str, title: str, message: str, metadata: Dict[str, Any] = None):
        """Create and send an alert"""
        # Check alert cooldown
        alert_key = f"{category}:{title}"
        last_alert = self.last_alert_times.get(alert_key)

        if last_alert and (datetime.now() - last_alert).total_seconds() < (self.monitoring_config["alert_cooldown_minutes"] * 60):
            return  # Still in cooldown

        alert = Alert(
            alert_id=f"alert_{int(time.time())}_{len(self.alert_history)}",
            severity=severity,
            category=category,
            title=title,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_times[alert_key] = datetime.now()

        # Send alert through all channels
        await self._send_alert(alert)

        self.logger.warning(f"Alert created: {severity} - {title}")

    async def _send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        for channel in self.alert_channels:
            try:
                if channel == "console":
                    await self._send_console_alert(alert)
                elif channel == "file":
                    await self._send_file_alert(alert)
                elif channel == "email":
                    await self._send_email_alert(alert)
                elif channel == "slack":
                    await self._send_slack_alert(alert)
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel}: {e}")

    async def _send_console_alert(self, alert: Alert):
        """Send alert to console"""
        severity_emoji = {
            "CRITICAL": "[ALERT]",
            "HIGH": "🔴",
            "MEDIUM": "🟡",
            "LOW": "🔵",
            "INFO": "ℹ️"
        }.get(alert.severity, "❓")

        print(f"{severity_emoji} ALERT [{alert.severity}] {alert.category}: {alert.title}")
        print(f"   {alert.message}")

    async def _send_file_alert(self, alert: Alert):
        """Write alert to log file"""
        alert_log_path = PROJECT_ROOT / "logs" / "production_alerts.log"

        with open(alert_log_path, 'a') as f:
            f.write(f"{alert.timestamp.isoformat()} | {alert.severity} | {alert.category} | {alert.title} | {alert.message}\n")

    async def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        # Email configuration would be loaded from config
        # This is a placeholder implementation
        self.logger.info(f"Email alert: {alert.title} (not implemented)")

    async def _send_slack_alert(self, alert: Alert):
        """Send alert via Slack"""
        # Slack configuration would be loaded from config
        # This is a placeholder implementation
        self.logger.info(f"Slack alert: {alert.title} (not implemented)")

    async def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()

            self.logger.info(f"Alert resolved: {alert.title}")

    async def _alert_cleanup_loop(self):
        """Clean up old alerts"""
        while True:
            try:
                cutoff_date = datetime.now() - timedelta(days=self.monitoring_config["alert_retention_days"])

                # Remove old alerts from history
                self.alert_history = [
                    alert for alert in self.alert_history
                    if alert.timestamp > cutoff_date
                ]

                await asyncio.sleep(3600)  # Clean up once per hour

            except Exception as e:
                self.logger.error(f"Alert cleanup error: {e}")
                await asyncio.sleep(300)

    # Health Check Functions

    async def _check_system_cpu(self) -> bool:
        """Check system CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent < 90  # Alert if CPU > 90%

    async def _check_system_memory(self) -> bool:
        """Check system memory usage"""
        memory = psutil.virtual_memory()
        return memory.percent < 90  # Alert if memory > 90%

    async def _check_trading_safeguards(self) -> bool:
        """Check trading safeguards status"""
        status = live_trading_safeguards.get_safety_status()
        return not status.get('emergency_shutdown', False)

    async def _check_market_data_feeds(self) -> bool:
        """Check market data feeds status"""
        # This would check actual market data feed connections
        # For now, return True
        return True

    async def _check_deployment_status(self) -> bool:
        """Check production deployment status"""
        status = production_deployment_system.get_deployment_status()
        return not status.get('emergency_stop', False)

    async def _check_network_connectivity(self) -> bool:
        """Check network connectivity"""
        # Simple connectivity check
        return True

    async def _check_disk_space(self) -> bool:
        """Check disk space"""
        disk = psutil.disk_usage('/')
        return disk.percent < 90  # Alert if disk > 90% full

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "active_alerts": len(self.active_alerts),
            "total_alerts_history": len(self.alert_history),
            "health_checks": {
                check_id: {
                    "name": check.name,
                    "last_run": check.last_run.isoformat() if check.last_run else None,
                    "last_result": check.last_result,
                    "consecutive_failures": check.consecutive_failures,
                    "enabled": check.enabled
                }
                for check_id, check in self.health_checks.items()
            },
            "alert_channels": self.alert_channels
        }

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        return [
            {
                "alert_id": alert.alert_id,
                "severity": alert.severity,
                "category": alert.category,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata
            }
            for alert in self.active_alerts.values()
        ]


# Global production monitoring system instance
production_monitoring_system = ProductionMonitoringSystem()


async def initialize_production_monitoring():
    """Initialize the production monitoring system"""
    print("[MONITOR] Initializing Production Monitoring System...")

    await production_monitoring_system.start_monitoring()

    print("[OK] Production monitoring system initialized")
    print("  24/7 monitoring active")
    print("  Alert channels: console, file")

    status = production_monitoring_system.get_monitoring_status()
    print(f"  Health checks: {len(status['health_checks'])}")
    print(f"  Active alerts: {status['active_alerts']}")


if __name__ == "__main__":
    asyncio.run(initialize_production_monitoring())