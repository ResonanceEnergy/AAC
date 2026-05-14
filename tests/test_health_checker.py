from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from shared.health_checker import (
    ComponentHealth,
    HealthChecker,
    HealthStatus,
)


# ---------------------------------------------------------------------------
# Enum & dataclass
# ---------------------------------------------------------------------------


class TestHealthStatusEnum:
    def test_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_count(self):
        assert len(list(HealthStatus)) == 4


class TestComponentHealth:
    def test_required_fields(self):
        ch = ComponentHealth(name="db", status=HealthStatus.HEALTHY)
        assert ch.name == "db"
        assert ch.status is HealthStatus.HEALTHY
        assert ch.latency_ms == 0.0
        assert ch.details == {}
        assert isinstance(ch.checked_at, datetime)
        assert ch.error is None

    def test_with_details_and_error(self):
        ch = ComponentHealth(
            name="api",
            status=HealthStatus.UNHEALTHY,
            latency_ms=12.5,
            details={"code": 500},
            error="boom",
        )
        assert ch.latency_ms == 12.5
        assert ch.details == {"code": 500}
        assert ch.error == "boom"


# ---------------------------------------------------------------------------
# HealthChecker init / register
# ---------------------------------------------------------------------------


class TestHealthCheckerInit:
    def test_starts_empty(self):
        hc = HealthChecker()
        assert hc._checks == {}
        assert hc._results == {}

    def test_register_check(self):
        hc = HealthChecker()
        fn = lambda: {"status": "healthy"}
        hc.register_check("api", fn)
        assert hc._checks == {"api": fn}

    def test_register_overwrites_same_name(self):
        hc = HealthChecker()
        fn1 = lambda: {"status": "healthy"}
        fn2 = lambda: {"status": "degraded"}
        hc.register_check("api", fn1)
        hc.register_check("api", fn2)
        assert hc._checks["api"] is fn2


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


class TestRunAll:
    def test_no_checks_returns_empty(self):
        hc = HealthChecker()
        assert hc.run_all() == {}

    def test_single_healthy_check(self):
        hc = HealthChecker()
        hc.register_check("api", lambda: {"status": "healthy", "x": 1})
        results = hc.run_all()
        assert "api" in results
        assert results["api"].status is HealthStatus.HEALTHY
        assert results["api"].details == {"status": "healthy", "x": 1}
        assert results["api"].error is None
        assert results["api"].latency_ms >= 0.0

    def test_degraded_status_parsed(self):
        hc = HealthChecker()
        hc.register_check("api", lambda: {"status": "degraded"})
        results = hc.run_all()
        assert results["api"].status is HealthStatus.DEGRADED

    def test_unhealthy_status_parsed(self):
        hc = HealthChecker()
        hc.register_check("api", lambda: {"status": "unhealthy"})
        results = hc.run_all()
        assert results["api"].status is HealthStatus.UNHEALTHY

    def test_default_status_is_healthy(self):
        hc = HealthChecker()
        hc.register_check("api", lambda: {})
        results = hc.run_all()
        assert results["api"].status is HealthStatus.HEALTHY

    def test_exception_marks_unhealthy_with_error(self):
        hc = HealthChecker()

        def bad():
            raise RuntimeError("kaboom")

        hc.register_check("flaky", bad)
        results = hc.run_all()
        assert results["flaky"].status is HealthStatus.UNHEALTHY
        assert results["flaky"].error == "kaboom"
        assert results["flaky"].latency_ms >= 0.0

    def test_invalid_status_value_marks_unhealthy(self):
        hc = HealthChecker()
        hc.register_check("api", lambda: {"status": "bogus"})
        results = hc.run_all()
        # ValueError from HealthStatus("bogus") caught by except → UNHEALTHY
        assert results["api"].status is HealthStatus.UNHEALTHY
        assert results["api"].error is not None

    def test_multiple_checks_all_executed(self):
        hc = HealthChecker()
        hc.register_check("a", lambda: {"status": "healthy"})
        hc.register_check("b", lambda: {"status": "degraded"})
        hc.register_check("c", lambda: {"status": "unhealthy"})
        results = hc.run_all()
        assert set(results.keys()) == {"a", "b", "c"}
        assert results["a"].status is HealthStatus.HEALTHY
        assert results["b"].status is HealthStatus.DEGRADED
        assert results["c"].status is HealthStatus.UNHEALTHY


# ---------------------------------------------------------------------------
# check_system
# ---------------------------------------------------------------------------


class TestCheckSystem:
    def test_healthy_low_usage(self):
        hc = HealthChecker()
        with patch("shared.health_checker.psutil") as ps:
            ps.cpu_percent.return_value = 10.0
            ps.virtual_memory.return_value = MagicMock(percent=20.0)
            ps.disk_usage.return_value = MagicMock(percent=30.0)
            ch = hc.check_system()
        assert ch.name == "system"
        assert ch.status is HealthStatus.HEALTHY
        assert ch.details["cpu_percent"] == 10.0
        assert ch.details["memory_percent"] == 20.0
        assert ch.details["disk_percent"] == 30.0

    def test_degraded_at_cpu_85(self):
        hc = HealthChecker()
        with patch("shared.health_checker.psutil") as ps:
            ps.cpu_percent.return_value = 85.0
            ps.virtual_memory.return_value = MagicMock(percent=50.0)
            ps.disk_usage.return_value = MagicMock(percent=10.0)
            ch = hc.check_system()
        assert ch.status is HealthStatus.DEGRADED

    def test_degraded_at_mem_85(self):
        hc = HealthChecker()
        with patch("shared.health_checker.psutil") as ps:
            ps.cpu_percent.return_value = 50.0
            ps.virtual_memory.return_value = MagicMock(percent=85.0)
            ps.disk_usage.return_value = MagicMock(percent=10.0)
            ch = hc.check_system()
        assert ch.status is HealthStatus.DEGRADED

    def test_unhealthy_at_cpu_99(self):
        hc = HealthChecker()
        with patch("shared.health_checker.psutil") as ps:
            ps.cpu_percent.return_value = 99.0
            ps.virtual_memory.return_value = MagicMock(percent=10.0)
            ps.disk_usage.return_value = MagicMock(percent=10.0)
            ch = hc.check_system()
        assert ch.status is HealthStatus.UNHEALTHY

    def test_unhealthy_at_mem_99(self):
        hc = HealthChecker()
        with patch("shared.health_checker.psutil") as ps:
            ps.cpu_percent.return_value = 10.0
            ps.virtual_memory.return_value = MagicMock(percent=99.0)
            ps.disk_usage.return_value = MagicMock(percent=10.0)
            ch = hc.check_system()
        assert ch.status is HealthStatus.UNHEALTHY

    def test_disk_failure_falls_back_to_zero(self):
        hc = HealthChecker()
        with patch("shared.health_checker.psutil") as ps:
            ps.cpu_percent.return_value = 5.0
            ps.virtual_memory.return_value = MagicMock(percent=5.0)
            ps.disk_usage.side_effect = OSError("no disk")
            ch = hc.check_system()
        assert ch.status is HealthStatus.HEALTHY
        assert ch.details["disk_percent"] == 0.0


# ---------------------------------------------------------------------------
# overall_status
# ---------------------------------------------------------------------------


class TestOverallStatus:
    def test_unknown_when_empty(self):
        hc = HealthChecker()
        assert hc.overall_status() is HealthStatus.UNKNOWN

    def test_healthy_when_all_healthy(self):
        hc = HealthChecker()
        hc.register_check("a", lambda: {"status": "healthy"})
        hc.register_check("b", lambda: {"status": "healthy"})
        hc.run_all()
        assert hc.overall_status() is HealthStatus.HEALTHY

    def test_degraded_when_any_degraded(self):
        hc = HealthChecker()
        hc.register_check("a", lambda: {"status": "healthy"})
        hc.register_check("b", lambda: {"status": "degraded"})
        hc.run_all()
        assert hc.overall_status() is HealthStatus.DEGRADED

    def test_unhealthy_dominates(self):
        hc = HealthChecker()
        hc.register_check("a", lambda: {"status": "healthy"})
        hc.register_check("b", lambda: {"status": "degraded"})
        hc.register_check("c", lambda: {"status": "unhealthy"})
        hc.run_all()
        assert hc.overall_status() is HealthStatus.UNHEALTHY


# ---------------------------------------------------------------------------
# get_report
# ---------------------------------------------------------------------------


class TestGetReport:
    def test_empty_report(self):
        hc = HealthChecker()
        report = hc.get_report()
        assert report["overall"] == "unknown"
        assert report["components"] == {}
        # ISO format string
        datetime.fromisoformat(report["checked_at"])

    def test_report_includes_components(self):
        hc = HealthChecker()
        hc.register_check("api", lambda: {"status": "healthy"})
        hc.register_check("db", lambda: {"status": "degraded"})
        hc.run_all()
        report = hc.get_report()
        assert report["overall"] == "degraded"
        assert set(report["components"].keys()) == {"api", "db"}
        assert report["components"]["api"]["status"] == "healthy"
        assert report["components"]["db"]["status"] == "degraded"
        assert "latency_ms" in report["components"]["api"]
        assert report["components"]["api"]["error"] is None

    def test_report_carries_error(self):
        hc = HealthChecker()

        def bad():
            raise RuntimeError("nope")

        hc.register_check("bad", bad)
        hc.run_all()
        report = hc.get_report()
        assert report["overall"] == "unhealthy"
        assert report["components"]["bad"]["status"] == "unhealthy"
        assert report["components"]["bad"]["error"] == "nope"
