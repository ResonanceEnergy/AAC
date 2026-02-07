"""
Shared Infrastructure - System Monitor
Monitors overall system health and performance across all departments.
"""

import asyncio
import psutil
import platform
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from SharedInfrastructure.audit_logger import AuditLogger

class SystemComponent(Enum):
    DOCTRINE_ORCHESTRATOR = "doctrine_orchestrator"
    FINANCIAL_ENGINE = "financial_engine"
    CRYPTO_ENGINE = "crypto_engine"
    RESEARCH_AGENTS = "research_agents"
    BRIDGE_SERVICES = "bridge_services"
    SECURITY_MONITOR = "security_monitor"
    DATABASE = "database"
    NETWORK = "network"

@dataclass
class HealthCheck:
    component: SystemComponent
    status: str  # HEALTHY, DEGRADED, UNHEALTHY, DOWN
    response_time: float
    last_check: datetime
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class SystemMonitor:
    """
    Monitors system health, performance, and component status.
    """

    def __init__(self):
        self.audit_logger = AuditLogger()
        self.health_checks: Dict[SystemComponent, HealthCheck] = {}
        self.monitoring_interval = 60  # seconds
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "response_time_max": 5.0,  # seconds
        }

    async def start_monitoring(self):
        """Start the system monitoring loop."""
        self.audit_logger.log_event(
            "system_monitor",
            "monitoring_started",
            "System monitoring service started",
            "info"
        )

        while True:
            try:
                await self._perform_health_checks()
                await self._check_system_resources()
                await self._generate_health_report()

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                self.audit_logger.log_event(
                    "system_monitor",
                    "monitoring_error",
                    f"Monitoring error: {str(e)}",
                    "error"
                )
                await asyncio.sleep(30)  # Shorter interval on error

    async def _perform_health_checks(self):
        """Perform health checks on all system components."""
        components_to_check = [
            SystemComponent.DOCTRINE_ORCHESTRATOR,
            SystemComponent.FINANCIAL_ENGINE,
            SystemComponent.CRYPTO_ENGINE,
            SystemComponent.RESEARCH_AGENTS,
            SystemComponent.BRIDGE_SERVICES,
            SystemComponent.SECURITY_MONITOR,
        ]

        for component in components_to_check:
            try:
                health_check = await self._check_component_health(component)
                self.health_checks[component] = health_check

                # Alert on unhealthy components
                if health_check.status in ["UNHEALTHY", "DOWN"]:
                    await self._alert_unhealthy_component(health_check)

            except Exception as e:
                self.audit_logger.log_event(
                    "system_monitor",
                    "health_check_failed",
                    f"Health check failed for {component.value}: {str(e)}",
                    "error"
                )

    async def _check_component_health(self, component: SystemComponent) -> HealthCheck:
        """Check health of a specific component."""
        start_time = datetime.now()

        try:
            # Component-specific health checks
            if component == SystemComponent.DOCTRINE_ORCHESTRATOR:
                status, metrics = await self._check_doctrine_orchestrator()
            elif component == SystemComponent.FINANCIAL_ENGINE:
                status, metrics = await self._check_financial_engine()
            elif component == SystemComponent.CRYPTO_ENGINE:
                status, metrics = await self._check_crypto_engine()
            elif component == SystemComponent.RESEARCH_AGENTS:
                status, metrics = await self._check_research_agents()
            elif component == SystemComponent.BRIDGE_SERVICES:
                status, metrics = await self._check_bridge_services()
            elif component == SystemComponent.SECURITY_MONITOR:
                status, metrics = await self._check_security_monitor()
            else:
                status, metrics = "UNKNOWN", {}

            response_time = (datetime.now() - start_time).total_seconds()

            return HealthCheck(
                component=component,
                status=status,
                response_time=response_time,
                last_check=datetime.now(),
                metrics=metrics
            )

        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            return HealthCheck(
                component=component,
                status="DOWN",
                response_time=response_time,
                last_check=datetime.now(),
                error_message=str(e)
            )

    async def _check_doctrine_orchestrator(self) -> tuple[str, Dict]:
        """Check doctrine orchestrator health."""
        try:
            # Import here to avoid circular imports
            from aac.doctrine.doctrine_integration import DoctrineIntegration
            doctrine = DoctrineIntegration()
            metrics = await doctrine.get_health_metrics()

            # Determine status based on metrics
            if metrics.get("doctrine_checks_completed", 0) > 0:
                status = "HEALTHY"
            else:
                status = "DEGRADED"

            return status, metrics

        except Exception:
            return "DOWN", {}

    async def _check_financial_engine(self) -> tuple[str, Dict]:
        """Check financial analysis engine health."""
        try:
            from CentralAccounting.financial_analysis_engine import FinancialAnalysisEngine
            engine = FinancialAnalysisEngine()
            metrics = await engine.get_health_status()

            status = "HEALTHY" if metrics.get("operational", False) else "DEGRADED"
            return status, metrics

        except Exception:
            return "DOWN", {}

    async def _check_crypto_engine(self) -> tuple[str, Dict]:
        """Check crypto intelligence engine health."""
        try:
            from CryptoIntelligence.crypto_intelligence_engine import CryptoIntelligenceEngine
            engine = CryptoIntelligenceEngine()
            metrics = await engine.get_health_status()

            status = "HEALTHY" if metrics.get("operational", False) else "DEGRADED"
            return status, metrics

        except Exception:
            return "DOWN", {}

    async def _check_research_agents(self) -> tuple[str, Dict]:
        """Check research agents health."""
        try:
            from BigBrainIntelligence.agents import ResearchAgentManager
            manager = ResearchAgentManager()
            metrics = await manager.get_health_status()

            status = "HEALTHY" if metrics.get("agents_active", 0) > 0 else "DEGRADED"
            return status, metrics

        except Exception:
            return "DOWN", {}

    async def _check_bridge_services(self) -> tuple[str, Dict]:
        """Check bridge services health."""
        try:
            from shared.crypto_bigbrain_bridge import CryptoBigBrainBridge
            bridge = CryptoBigBrainBridge()
            metrics = await bridge.get_health_status()

            status = "HEALTHY" if metrics.get("connections_active", 0) > 0 else "DEGRADED"
            return status, metrics

        except Exception:
            return "DOWN", {}

    async def _check_security_monitor(self) -> tuple[str, Dict]:
        """Check security monitor health."""
        try:
            from SharedInfrastructure.security_monitor import SecurityMonitor
            monitor = SecurityMonitor()
            metrics = await monitor.get_security_status()

            status = "HEALTHY" if metrics.get("monitoring_active", False) else "DOWN"
            return status, metrics

        except Exception:
            return "DOWN", {}

    async def _check_system_resources(self):
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.alert_thresholds["cpu_percent"]:
                await self._alert_resource_usage("cpu", cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.alert_thresholds["memory_percent"]:
                await self._alert_resource_usage("memory", memory.percent)

            # Disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > self.alert_thresholds["disk_percent"]:
                await self._alert_resource_usage("disk", disk.percent)

        except Exception as e:
            self.audit_logger.log_event(
                "system_monitor",
                "resource_check_error",
                f"Failed to check system resources: {str(e)}",
                "error"
            )

    async def _alert_resource_usage(self, resource: str, usage: float):
        """Alert on high resource usage."""
        self.audit_logger.log_event(
            "system_monitor",
            "resource_alert",
            f"High {resource} usage: {usage:.1f}%",
            "warning"
        )

    async def _alert_unhealthy_component(self, health_check: HealthCheck):
        """Alert on unhealthy component."""
        self.audit_logger.log_event(
            "system_monitor",
            "component_alert",
            f"Component {health_check.component.value} is {health_check.status}: {health_check.error_message or 'No details'}",
            "error"
        )

    async def _generate_health_report(self):
        """Generate periodic health report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "component_health": {
                comp.value: {
                    "status": hc.status,
                    "response_time": hc.response_time,
                    "last_check": hc.last_check.isoformat(),
                    "error": hc.error_message
                }
                for comp, hc in self.health_checks.items()
            },
            "system_resources": self._get_system_resources()
        }

        # Log summary
        healthy_count = sum(1 for hc in self.health_checks.values() if hc.status == "HEALTHY")
        total_count = len(self.health_checks)

        self.audit_logger.log_event(
            "system_monitor",
            "health_report",
            f"Health check complete: {healthy_count}/{total_count} components healthy",
            "info"
        )

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "hostname": platform.node()
        }

    def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage('/').total,
                    "free": psutil.disk_usage('/').free,
                    "percent": psutil.disk_usage('/').percent
                },
                "network": {
                    "bytes_sent": psutil.net_io_counters().bytes_sent,
                    "bytes_recv": psutil.net_io_counters().bytes_recv
                }
            }
        except Exception:
            return {"error": "Unable to retrieve system resources"}

    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all components."""
        return {
            "overall_status": self._calculate_overall_status(),
            "components": {
                comp.value: {
                    "status": hc.status,
                    "response_time": hc.response_time,
                    "last_check": hc.last_check.isoformat(),
                    "metrics": hc.metrics
                }
                for comp, hc in self.health_checks.items()
            },
            "system_resources": self._get_system_resources(),
            "last_report": datetime.now().isoformat()
        }

    def _calculate_overall_status(self) -> str:
        """Calculate overall system health status."""
        if not self.health_checks:
            return "UNKNOWN"

        statuses = [hc.status for hc in self.health_checks.values()]

        if "DOWN" in statuses:
            return "CRITICAL"
        elif "UNHEALTHY" in statuses:
            return "DEGRADED"
        elif all(s == "HEALTHY" for s in statuses):
            return "HEALTHY"
        else:
            return "WARNING"

# Global system monitor instance
system_monitor = SystemMonitor()

async def get_system_monitor():
    """Get the global system monitor instance."""
    return system_monitor