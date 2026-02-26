"""
Shared Infrastructure - Metrics Collector
Collects and aggregates system metrics across all departments.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import statistics

from SharedInfrastructure.audit_logger import AuditLogger

@dataclass
class MetricPoint:
    timestamp: datetime
    value: float
    tags: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MetricSeries:
    name: str
    points: List[MetricPoint]
    aggregation_type: str  # count, gauge, histogram, summary

class MetricsCollector:
    """
    Collects and aggregates system metrics from all departments.
    """

    def __init__(self):
        self.audit_logger = AuditLogger()
        self.metrics: Dict[str, MetricSeries] = {}
        self.collection_interval = 30  # seconds
        self.retention_period = timedelta(hours=24)  # Keep metrics for 24 hours

        # Metric definitions
        self.metric_definitions = {
            "system.cpu_usage": {"type": "gauge", "unit": "percent"},
            "system.memory_usage": {"type": "gauge", "unit": "percent"},
            "system.disk_usage": {"type": "gauge", "unit": "percent"},
            "doctrine.checks_completed": {"type": "count", "unit": "checks"},
            "financial.pnl_total": {"type": "gauge", "unit": "currency"},
            "crypto.venues_monitored": {"type": "gauge", "unit": "count"},
            "research.agents_active": {"type": "gauge", "unit": "count"},
            "security.events_detected": {"type": "count", "unit": "events"},
            "bridge.messages_processed": {"type": "count", "unit": "messages"}
        }

    async def start_collection(self):
        """Start the metrics collection loop."""
        self.audit_logger.log_event(
            "metrics_collector",
            "collection_started",
            "Metrics collection service started",
            "info"
        )

        while True:
            try:
                await self._collect_all_metrics()
                await self._cleanup_old_metrics()

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                self.audit_logger.log_event(
                    "metrics_collector",
                    "collection_error",
                    f"Metrics collection error: {str(e)}",
                    "error"
                )
                await asyncio.sleep(60)  # Wait longer on error

    async def _collect_all_metrics(self):
        """Collect metrics from all system components."""
        # System metrics
        await self._collect_system_metrics()

        # Department metrics
        await self._collect_doctrine_metrics()
        await self._collect_financial_metrics()
        await self._collect_crypto_metrics()
        await self._collect_research_metrics()
        await self._collect_security_metrics()
        await self._collect_bridge_metrics()

    async def _collect_system_metrics(self):
        """Collect basic system metrics."""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system.cpu_usage", cpu_percent, {"host": "localhost"})

            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("system.memory_usage", memory.percent, {"host": "localhost"})

            # Disk usage
            disk = psutil.disk_usage('/')
            self.record_metric("system.disk_usage", disk.percent, {"host": "localhost"})

        except Exception as e:
            self.audit_logger.log_event(
                "metrics_collector",
                "system_metrics_error",
                f"Failed to collect system metrics: {str(e)}",
                "error"
            )

    async def _collect_doctrine_metrics(self):
        """Collect doctrine orchestrator metrics."""
        try:
            from aac.doctrine.doctrine_integration import DoctrineIntegration
            doctrine = DoctrineIntegration()
            metrics = await doctrine.get_health_metrics()

            checks_completed = metrics.get("doctrine_checks_completed", 0)
            self.record_metric("doctrine.checks_completed", checks_completed,
                             {"component": "orchestrator"})

        except Exception as e:
            self.audit_logger.log_event(
                "metrics_collector",
                "doctrine_metrics_error",
                f"Failed to collect doctrine metrics: {str(e)}",
                "error"
            )

    async def _collect_financial_metrics(self):
        """Collect financial analysis metrics."""
        try:
            from CentralAccounting.financial_analysis_engine import FinancialAnalysisEngine
            engine = FinancialAnalysisEngine()
            metrics = await engine.get_health_status()

            pnl_total = metrics.get("pnl_total", 0.0)
            self.record_metric("financial.pnl_total", pnl_total,
                             {"department": "accounting"})

        except Exception as e:
            self.audit_logger.log_event(
                "metrics_collector",
                "financial_metrics_error",
                f"Failed to collect financial metrics: {str(e)}",
                "error"
            )

    async def _collect_crypto_metrics(self):
        """Collect crypto intelligence metrics."""
        try:
            from CryptoIntelligence.crypto_intelligence_engine import CryptoIntelligenceEngine
            engine = CryptoIntelligenceEngine()
            metrics = await engine.get_health_status()

            venues_monitored = metrics.get("venues_monitored", 0)
            self.record_metric("crypto.venues_monitored", venues_monitored,
                             {"department": "intelligence"})

        except Exception as e:
            self.audit_logger.log_event(
                "metrics_collector",
                "crypto_metrics_error",
                f"Failed to collect crypto metrics: {str(e)}",
                "error"
            )

    async def _collect_research_metrics(self):
        """Collect research agent metrics."""
        try:
            from BigBrainIntelligence.agents import ResearchAgentManager
            manager = ResearchAgentManager()
            metrics = await manager.get_health_status()

            agents_active = metrics.get("agents_active", 0)
            self.record_metric("research.agents_active", agents_active,
                             {"department": "intelligence"})

        except Exception as e:
            self.audit_logger.log_event(
                "metrics_collector",
                "research_metrics_error",
                f"Failed to collect research metrics: {str(e)}",
                "error"
            )

    async def _collect_security_metrics(self):
        """Collect security monitoring metrics."""
        try:
            from SharedInfrastructure.security_monitor import SecurityMonitor
            monitor = SecurityMonitor()
            metrics = await monitor.get_security_status()

            events_detected = metrics.get("active_threats", 0)
            self.record_metric("security.events_detected", events_detected,
                             {"component": "monitor"})

        except Exception as e:
            self.audit_logger.log_event(
                "metrics_collector",
                "security_metrics_error",
                f"Failed to collect security metrics: {str(e)}",
                "error"
            )

    async def _collect_bridge_metrics(self):
        """Collect bridge service metrics."""
        try:
            from shared.crypto_bigbrain_bridge import CryptoBigBrainBridge
            bridge = CryptoBigBrainBridge()
            metrics = await bridge.get_health_status()

            messages_processed = metrics.get("messages_processed", 0)
            self.record_metric("bridge.messages_processed", messages_processed,
                             {"component": "bridge"})

        except Exception as e:
            self.audit_logger.log_event(
                "metrics_collector",
                "bridge_metrics_error",
                f"Failed to collect bridge metrics: {str(e)}",
                "error"
            )

    def record_metric(self, name: str, value: float, tags: Dict[str, str],
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Metric tags for categorization
            metadata: Additional metadata
        """
        if name not in self.metrics:
            metric_type = self.metric_definitions.get(name, {}).get("type", "gauge")
            self.metrics[name] = MetricSeries(name=name, points=[], aggregation_type=metric_type)

        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags,
            metadata=metadata
        )

        self.metrics[name].points.append(point)

        # Keep only recent points
        cutoff_time = datetime.now() - self.retention_period
        self.metrics[name].points = [
            p for p in self.metrics[name].points
            if p.timestamp > cutoff_time
        ]

    async def _cleanup_old_metrics(self):
        """Clean up metrics older than retention period."""
        cutoff_time = datetime.now() - self.retention_period

        for name, series in self.metrics.items():
            original_count = len(series.points)
            series.points = [p for p in series.points if p.timestamp > cutoff_time]

            if len(series.points) < original_count:
                self.audit_logger.log_event(
                    "metrics_collector",
                    "metrics_cleanup",
                    f"Cleaned up {original_count - len(series.points)} old points for {name}",
                    "info"
                )

    def get_metric_series(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[MetricSeries]:
        """
        Get a metric series by name and optional tags.

        Args:
            name: Metric name
            tags: Optional tags to filter by

        Returns:
            MetricSeries if found, None otherwise
        """
        series = self.metrics.get(name)
        if not series or not tags:
            return series

        # Filter points by tags
        filtered_points = []
        for point in series.points:
            if all(point.tags.get(k) == v for k, v in tags.items()):
                filtered_points.append(point)

        if filtered_points:
            return MetricSeries(
                name=name,
                points=filtered_points,
                aggregation_type=series.aggregation_type
            )

        return None

    def get_metric_value(self, name: str, tags: Optional[Dict[str, str]] = None,
                        aggregation: str = "latest") -> Optional[float]:
        """
        Get the current value of a metric.

        Args:
            name: Metric name
            tags: Optional tags to filter by
            aggregation: Aggregation method (latest, average, sum, min, max)

        Returns:
            Metric value or None if not found
        """
        series = self.get_metric_series(name, tags)
        if not series or not series.points:
            return None

        values = [p.value for p in series.points]

        if aggregation == "latest":
            return values[-1]
        elif aggregation == "average":
            return statistics.mean(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)

        return None

    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all collected metrics.

        Returns:
            Summary of metrics
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_metrics": len(self.metrics),
            "metrics": {}
        }

        for name, series in self.metrics.items():
            points = series.points
            if points:
                values = [p.value for p in points]
                summary["metrics"][name] = {
                    "count": len(points),
                    "latest": values[-1],
                    "average": statistics.mean(values) if len(values) > 1 else values[0],
                    "min": min(values),
                    "max": max(values),
                    "type": series.aggregation_type
                }
            else:
                summary["metrics"][name] = {
                    "count": 0,
                    "type": series.aggregation_type
                }

        return summary

    def export_metrics(self, format: str = "json") -> str:
        """
        Export all metrics in the specified format.

        Args:
            format: Export format (json, prometheus)

        Returns:
            Exported metrics as string
        """
        if format == "json":
            return self._export_json()
        elif format == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self) -> str:
        """Export metrics as JSON."""
        import json

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }

        for name, series in self.metrics.items():
            export_data["metrics"][name] = {
                "type": series.aggregation_type,
                "points": [
                    {
                        "timestamp": p.timestamp.isoformat(),
                        "value": p.value,
                        "tags": p.tags,
                        "metadata": p.metadata
                    }
                    for p in series.points[-100:]  # Last 100 points
                ]
            }

        return json.dumps(export_data, indent=2, default=str)

    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, series in self.metrics.items():
            if series.points:
                latest_point = series.points[-1]
                # Convert metric name to Prometheus format
                prom_name = name.replace(".", "_").replace("-", "_")

                # Add metric value
                tags_str = ",".join(f'{k}="{v}"' for k, v in latest_point.tags.items())
                if tags_str:
                    line = f'{prom_name}{{{tags_str}}} {latest_point.value}'
                else:
                    line = f'{prom_name} {latest_point.value}'

                lines.append(line)

        return "\n".join(lines)

# Global metrics collector instance
metrics_collector = MetricsCollector()

async def get_metrics_collector():
    """Get the global metrics collector instance."""
    return metrics_collector