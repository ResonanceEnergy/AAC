"""
shared.health_checker — System health checking for AAC.

Provides HealthChecker for running component health checks,
aggregating status, and reporting system-wide health.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """HealthStatus class."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """ComponentHealth class."""
    name: str
    status: HealthStatus
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


class HealthChecker:
    """System-wide health checker with pluggable component checks."""

    def __init__(self):
        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, ComponentHealth] = {}
        logger.info("HealthChecker initialized")

    def register_check(self, name: str, check_fn: Callable):
        """Register a health check function. It should return a dict with at least 'status'."""
        self._checks[name] = check_fn

    def run_all(self) -> Dict[str, ComponentHealth]:
        """Run all registered health checks synchronously."""
        for name, check_fn in self._checks.items():
            start = time.monotonic()
            try:
                result = check_fn()
                latency = (time.monotonic() - start) * 1000
                status = HealthStatus(result.get("status", "healthy"))
                self._results[name] = ComponentHealth(
                    name=name,
                    status=status,
                    latency_ms=round(latency, 2),
                    details=result,
                )
            except Exception as e:
                latency = (time.monotonic() - start) * 1000
                self._results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=round(latency, 2),
                    error=str(e),
                )
                logger.error(f"Health check '{name}' failed: {e}")
        return self._results

    def check_system(self) -> ComponentHealth:
        """Built-in system resource health check."""
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        try:
            disk = psutil.disk_usage("/").percent
        except Exception:
            disk = 0.0

        if cpu > 95 or mem > 95:
            status = HealthStatus.UNHEALTHY
        elif cpu > 80 or mem > 80:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return ComponentHealth(
            name="system",
            status=status,
            details={"cpu_percent": cpu, "memory_percent": mem, "disk_percent": disk},
        )

    def overall_status(self) -> HealthStatus:
        """Aggregate overall status from all results."""
        if not self._results:
            return HealthStatus.UNKNOWN
        statuses = [r.status for r in self._results.values()]
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def get_report(self) -> Dict[str, Any]:
        """Return a summary report dict."""
        return {
            "overall": self.overall_status().value,
            "components": {
                name: {
                    "status": h.status.value,
                    "latency_ms": h.latency_ms,
                    "error": h.error,
                }
                for name, h in self._results.items()
            },
            "checked_at": datetime.now().isoformat(),
        }
