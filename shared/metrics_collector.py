"""
shared.metrics_collector — System metrics collection engine.

Provides MetricsCollector for gathering system and trading metrics,
and get_metrics_collector() singleton accessor.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores system, trading, and infrastructure metrics."""

    def __init__(self):
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self._running = False
        self._collection_interval = 30  # seconds
        self.logger = logging.getLogger(self.__class__.__name__)

    async def start_collection(self):
        """Start periodic metrics collection."""
        self._running = True
        self.logger.info("📊 Metrics collection started")
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self._collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(self._collection_interval)

    async def stop_collection(self):
        """Stop metrics collection."""
        self._running = False
        self.logger.info("📊 Metrics collection stopped")

    async def _collect_system_metrics(self):
        """Collect basic system metrics."""
        try:
            import psutil
            self.record("system.cpu_percent", psutil.cpu_percent())
            mem = psutil.virtual_memory()
            self.record("system.memory_percent", mem.percent)
            self.record("system.memory_available_gb", mem.available / (1024**3))
        except ImportError:
            pass

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric data point."""
        point = MetricPoint(name=name, value=value, tags=tags or {})
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(point)
        # Keep last 1000 points per metric
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-500:]

    def get_latest(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        points = self.metrics.get(name, [])
        return points[-1].value if points else None

    def get_all_latest(self) -> Dict[str, float]:
        """Get latest values for all metrics."""
        return {
            name: points[-1].value
            for name, points in self.metrics.items()
            if points
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_metrics": len(self.metrics),
            "total_points": sum(len(v) for v in self.metrics.values()),
            "collecting": self._running,
            "latest": self.get_all_latest(),
        }


# Singleton
_metrics_collector: Optional[MetricsCollector] = None


async def get_metrics_collector() -> MetricsCollector:
    """Get the global MetricsCollector singleton."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
