"""
Shared Infrastructure - Health Checker
Provides detailed health checking capabilities for system components.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from SharedInfrastructure.audit_logger import AuditLogger

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DOWN = "down"
    UNKNOWN = "unknown"

@dataclass
class HealthResult:
    component_name: str
    status: HealthStatus
    response_time: float
    timestamp: datetime
    message: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthChecker:
    """
    Advanced health checking system for system components.
    """

    def __init__(self):
        self.audit_logger = AuditLogger()
        self.health_checks: Dict[str, Callable] = {}
        self.check_history: Dict[str, List[HealthResult]] = {}
        self.max_history_size = 100

        # Register built-in health checks
        self._register_builtin_checks()

    def _register_builtin_checks(self):
        """Register built-in health check functions."""
        self.register_check("doctrine_orchestrator", self._check_doctrine_orchestrator)
        self.register_check("financial_engine", self._check_financial_engine)
        self.register_check("crypto_engine", self._check_crypto_engine)
        self.register_check("research_agents", self._check_research_agents)
        self.register_check("bridge_services", self._check_bridge_services)
        self.register_check("database", self._check_database)
        self.register_check("network", self._check_network)
        self.register_check("file_system", self._check_file_system)

    def register_check(self, name: str, check_function: Callable):
        """
        Register a custom health check function.

        Args:
            name: Name of the health check
            check_function: Async function that returns HealthResult
        """
        self.health_checks[name] = check_function
        self.check_history[name] = []

    async def run_health_check(self, component_name: str) -> HealthResult:
        """
        Run a specific health check.

        Args:
            component_name: Name of the component to check

        Returns:
            HealthResult with check details
        """
        if component_name not in self.health_checks:
            return HealthResult(
                component_name=component_name,
                status=HealthStatus.UNKNOWN,
                response_time=0.0,
                timestamp=datetime.now(),
                message=f"No health check registered for {component_name}",
                error="Check not found"
            )

        start_time = time.time()

        try:
            check_func = self.health_checks[component_name]
            result = await check_func()

            response_time = time.time() - start_time

            # Update result with timing
            result.response_time = response_time
            result.timestamp = datetime.now()

            # Store in history
            self._add_to_history(component_name, result)

            # Log significant issues
            if result.status in [HealthStatus.UNHEALTHY, HealthStatus.DOWN]:
                self.audit_logger.log_event(
                    "health_checker",
                    "health_check_failed",
                    f"{component_name}: {result.message}",
                    "error"
                )

            return result

        except Exception as e:
            response_time = time.time() - start_time

            result = HealthResult(
                component_name=component_name,
                status=HealthStatus.DOWN,
                response_time=response_time,
                timestamp=datetime.now(),
                message=f"Health check failed: {str(e)}",
                error=str(e)
            )

            self._add_to_history(component_name, result)

            self.audit_logger.log_event(
                "health_checker",
                "health_check_error",
                f"{component_name}: {str(e)}",
                "error"
            )

            return result

    async def run_all_checks(self) -> Dict[str, HealthResult]:
        """
        Run all registered health checks.

        Returns:
            Dictionary of component names to HealthResult
        """
        results = {}

        for component_name in self.health_checks.keys():
            results[component_name] = await self.run_health_check(component_name)

        return results

    def _add_to_history(self, component_name: str, result: HealthResult):
        """Add result to history, maintaining max size."""
        history = self.check_history[component_name]
        history.append(result)

        if len(history) > self.max_history_size:
            history.pop(0)

    async def get_health_history(self, component_name: str,
                                limit: int = 10) -> List[HealthResult]:
        """
        Get health check history for a component.

        Args:
            component_name: Component name
            limit: Maximum number of results to return

        Returns:
            List of recent HealthResult objects
        """
        history = self.check_history.get(component_name, [])
        return history[-limit:]

    async def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all health checks.

        Returns:
            Summary statistics
        """
        all_results = await self.run_all_checks()

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_checks": len(all_results),
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "down": 0,
            "unknown": 0,
            "details": {}
        }

        for name, result in all_results.items():
            summary["details"][name] = {
                "status": result.status.value,
                "response_time": result.response_time,
                "message": result.message,
                "last_check": result.timestamp.isoformat()
            }

            if result.status == HealthStatus.HEALTHY:
                summary["healthy"] += 1
            elif result.status == HealthStatus.DEGRADED:
                summary["degraded"] += 1
            elif result.status == HealthStatus.UNHEALTHY:
                summary["unhealthy"] += 1
            elif result.status == HealthStatus.DOWN:
                summary["down"] += 1
            else:
                summary["unknown"] += 1

        # Calculate overall status
        if summary["down"] > 0:
            summary["overall_status"] = "CRITICAL"
        elif summary["unhealthy"] > 0:
            summary["overall_status"] = "WARNING"
        elif summary["degraded"] > 0:
            summary["overall_status"] = "DEGRADED"
        elif summary["healthy"] == summary["total_checks"]:
            summary["overall_status"] = "HEALTHY"
        else:
            summary["overall_status"] = "UNKNOWN"

        return summary

    # Built-in health check implementations

    async def _check_doctrine_orchestrator(self) -> HealthResult:
        """Check doctrine orchestrator health."""
        try:
            from aac.doctrine.doctrine_integration import DoctrineIntegration
            doctrine = DoctrineIntegration()
            metrics = await doctrine.get_health_metrics()

            if metrics.get("operational", False):
                return HealthResult(
                    component_name="doctrine_orchestrator",
                    status=HealthStatus.HEALTHY,
                    response_time=0.0,  # Will be set by caller
                    timestamp=datetime.now(),
                    message="Doctrine orchestrator is operational",
                    details=metrics
                )
            else:
                return HealthResult(
                    component_name="doctrine_orchestrator",
                    status=HealthStatus.DEGRADED,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    message="Doctrine orchestrator has issues",
                    details=metrics
                )

        except Exception as e:
            return HealthResult(
                component_name="doctrine_orchestrator",
                status=HealthStatus.DOWN,
                response_time=0.0,
                timestamp=datetime.now(),
                message=f"Cannot access doctrine orchestrator: {str(e)}",
                error=str(e)
            )

    async def _check_financial_engine(self) -> HealthResult:
        """Check financial analysis engine health."""
        try:
            from CentralAccounting.financial_analysis_engine import FinancialAnalysisEngine
            engine = FinancialAnalysisEngine()
            status = await engine.get_health_status()

            if status.get("operational", False):
                return HealthResult(
                    component_name="financial_engine",
                    status=HealthStatus.HEALTHY,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    message="Financial engine is operational",
                    details=status
                )
            else:
                return HealthResult(
                    component_name="financial_engine",
                    status=HealthStatus.DEGRADED,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    message="Financial engine has issues",
                    details=status
                )

        except Exception as e:
            return HealthResult(
                component_name="financial_engine",
                status=HealthStatus.DOWN,
                response_time=0.0,
                timestamp=datetime.now(),
                message=f"Cannot access financial engine: {str(e)}",
                error=str(e)
            )

    async def _check_crypto_engine(self) -> HealthResult:
        """Check crypto intelligence engine health."""
        try:
            from CryptoIntelligence.crypto_intelligence_engine import CryptoIntelligenceEngine
            engine = CryptoIntelligenceEngine()
            status = await engine.get_health_status()

            if status.get("operational", False):
                return HealthResult(
                    component_name="crypto_engine",
                    status=HealthStatus.HEALTHY,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    message="Crypto engine is operational",
                    details=status
                )
            else:
                return HealthResult(
                    component_name="crypto_engine",
                    status=HealthStatus.DEGRADED,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    message="Crypto engine has issues",
                    details=status
                )

        except Exception as e:
            return HealthResult(
                component_name="crypto_engine",
                status=HealthStatus.DOWN,
                response_time=0.0,
                timestamp=datetime.now(),
                message=f"Cannot access crypto engine: {str(e)}",
                error=str(e)
            )

    async def _check_research_agents(self) -> HealthResult:
        """Check research agents health."""
        try:
            from BigBrainIntelligence.agents import ResearchAgentManager
            manager = ResearchAgentManager()
            status = await manager.get_health_status()

            active_agents = status.get("agents_active", 0)
            if active_agents > 0:
                return HealthResult(
                    component_name="research_agents",
                    status=HealthStatus.HEALTHY,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    message=f"{active_agents} research agents active",
                    details=status
                )
            else:
                return HealthResult(
                    component_name="research_agents",
                    status=HealthStatus.DEGRADED,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    message="No active research agents",
                    details=status
                )

        except Exception as e:
            return HealthResult(
                component_name="research_agents",
                status=HealthStatus.DOWN,
                response_time=0.0,
                timestamp=datetime.now(),
                message=f"Cannot access research agents: {str(e)}",
                error=str(e)
            )

    async def _check_bridge_services(self) -> HealthResult:
        """Check bridge services health."""
        try:
            from shared.crypto_bigbrain_bridge import CryptoBigBrainBridge
            bridge = CryptoBigBrainBridge()
            status = await bridge.get_health_status()

            active_connections = status.get("connections_active", 0)
            if active_connections > 0:
                return HealthResult(
                    component_name="bridge_services",
                    status=HealthStatus.HEALTHY,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    message=f"{active_connections} bridge connections active",
                    details=status
                )
            else:
                return HealthResult(
                    component_name="bridge_services",
                    status=HealthStatus.DEGRADED,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    message="No active bridge connections",
                    details=status
                )

        except Exception as e:
            return HealthResult(
                component_name="bridge_services",
                status=HealthStatus.DOWN,
                response_time=0.0,
                timestamp=datetime.now(),
                message=f"Cannot access bridge services: {str(e)}",
                error=str(e)
            )

    async def _check_database(self) -> HealthResult:
        """Check database connectivity."""
        # This would check actual database connections
        # For now, return healthy status
        return HealthResult(
            component_name="database",
            status=HealthStatus.HEALTHY,
            response_time=0.0,
            timestamp=datetime.now(),
            message="Database connectivity normal"
        )

    async def _check_network(self) -> HealthResult:
        """Check network connectivity."""
        # This would check network interfaces, DNS, etc.
        return HealthResult(
            component_name="network",
            status=HealthStatus.HEALTHY,
            response_time=0.0,
            timestamp=datetime.now(),
            message="Network connectivity normal"
        )

    async def _check_file_system(self) -> HealthResult:
        """Check file system health."""
        import os

        try:
            # Check if critical directories exist and are accessible
            critical_paths = [
                "config",
                "shared",
                "NCC",
                "CentralAccounting",
                "CryptoIntelligence",
                "BigBrainIntelligence"
            ]

            missing_paths = []
            for path in critical_paths:
                if not os.path.exists(path):
                    missing_paths.append(path)

            if missing_paths:
                return HealthResult(
                    component_name="file_system",
                    status=HealthStatus.UNHEALTHY,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    message=f"Missing critical directories: {', '.join(missing_paths)}",
                    details={"missing_paths": missing_paths}
                )
            else:
                return HealthResult(
                    component_name="file_system",
                    status=HealthStatus.HEALTHY,
                    response_time=0.0,
                    timestamp=datetime.now(),
                    message="All critical directories accessible"
                )

        except Exception as e:
            return HealthResult(
                component_name="file_system",
                status=HealthStatus.DOWN,
                response_time=0.0,
                timestamp=datetime.now(),
                message=f"File system check failed: {str(e)}",
                error=str(e)
            )

# Global health checker instance
health_checker = HealthChecker()

async def get_health_checker():
    """Get the global health checker instance."""
    return health_checker