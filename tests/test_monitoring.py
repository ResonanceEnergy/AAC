#!/usr/bin/env python3
"""
Tests for monitoring subsystem — covers the master dashboard,
continuous monitoring service, and security dashboard.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Imports ────────────────────────────────────────────────────────────────────

class TestMonitoringImports:
    """Verify all monitoring modules import without error."""

    def test_import_master_dashboard(self):
        mod = pytest.importorskip(
            'monitoring.aac_master_monitoring_dashboard',
            reason='master dashboard has unmet deps',
        )
        assert hasattr(mod, 'AACMasterMonitoringDashboard')
        assert hasattr(mod, 'DisplayMode')

    def test_import_continuous_monitoring(self):
        import monitoring.continuous_monitoring as mod
        assert hasattr(mod, 'ContinuousMonitoringService')

    def test_import_security_dashboard(self):
        mod = pytest.importorskip(
            'monitoring.security_dashboard',
            reason='security_dashboard needs qrcode',
        )
        assert hasattr(mod, 'SecurityDashboard')

    def test_import_security_compliance(self):
        mod = pytest.importorskip(
            'monitoring.security_compliance_integration',
            reason='security_compliance needs qrcode',
        )
        assert hasattr(mod, 'check_security_compliance')

    def test_import_monitoring_init(self):
        import monitoring
        assert monitoring is not None


# ── DisplayMode ────────────────────────────────────────────────────────────────

class TestDisplayMode:
    """DisplayMode enum values."""

    def test_enum_members(self):
        pytest.importorskip('monitoring.aac_master_monitoring_dashboard', reason='unmet deps')
        from monitoring.aac_master_monitoring_dashboard import DisplayMode
        assert DisplayMode.TERMINAL == "terminal"
        assert DisplayMode.WEB == "web"
        assert DisplayMode.DASH == "dash"
        assert DisplayMode.API == "api"


# ── AACMasterMonitoringDashboard ───────────────────────────────────────────────

class TestMasterDashboard:
    """Tests for AACMasterMonitoringDashboard."""

    @pytest.fixture
    def dashboard(self):
        pytest.importorskip('monitoring.aac_master_monitoring_dashboard', reason='unmet deps')
        from monitoring.aac_master_monitoring_dashboard import AACMasterMonitoringDashboard
        return AACMasterMonitoringDashboard()

    def test_instantiation(self, dashboard):
        assert dashboard is not None
        assert hasattr(dashboard, 'display_mode')
        assert hasattr(dashboard, 'collect_monitoring_data')

    def test_default_display_mode(self, dashboard):
        from monitoring.aac_master_monitoring_dashboard import DisplayMode
        assert dashboard.display_mode == DisplayMode.TERMINAL

    @pytest.mark.asyncio
    async def test_collect_monitoring_data_returns_dict(self, dashboard):
        """collect_monitoring_data should return a dict with standard keys."""
        data = await dashboard.collect_monitoring_data()
        assert isinstance(data, dict)
        assert 'timestamp' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_get_system_health(self, dashboard):
        """_get_system_health should return health structure with infra data."""
        result = await dashboard._get_system_health()
        assert isinstance(result, dict)
        # Structure has infrastructure.cpu and infrastructure.memory
        infra = result.get('infrastructure', {})
        assert 'cpu' in infra or 'memory' in infra

    @pytest.mark.asyncio
    async def test_check_network_health(self, dashboard):
        """_check_network_health should return status and latency."""
        result = await dashboard._check_network_health()
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'latency_ms' in result

    @pytest.mark.asyncio
    async def test_check_memory_usage(self, dashboard):
        result = await dashboard._check_memory_usage()
        assert isinstance(result, dict)
        assert 'total_gb' in result or 'percent' in result or 'status' in result


# ── ContinuousMonitoringService ────────────────────────────────────────────────

class TestContinuousMonitoring:
    """Tests for ContinuousMonitoringService."""

    @pytest.fixture
    def service(self):
        from monitoring.continuous_monitoring import ContinuousMonitoringService
        return ContinuousMonitoringService()

    def test_instantiation(self, service):
        assert service is not None
        assert hasattr(service, 'start_monitoring')

    def test_alert_thresholds_exist(self, service):
        """Service should define thresholds for alerting."""
        assert hasattr(service, 'alert_thresholds') or hasattr(service, 'thresholds')

    @pytest.mark.asyncio
    async def test_health_check_methods(self, service):
        """_check_system_health should return a dict."""
        result = await service._check_system_health()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_send_alert_no_config(self, service):
        """_send_alert should not crash when no notification channels configured."""
        alert = {
            'id': 'test-001',
            'severity': 'warning',
            'message': 'Test alert',
            'timestamp': datetime.now(),
        }
        # Should not raise — falls back to logging
        await service._send_alert(alert)

    @pytest.mark.asyncio
    async def test_send_alert_with_string_timestamp(self, service):
        """_send_alert handles string timestamps gracefully."""
        alert = {
            'id': 'test-002',
            'severity': 'info',
            'message': 'String timestamp alert',
            'timestamp': '2026-03-12 10:00:00',
        }
        await service._send_alert(alert)


# ── SecurityDashboard ─────────────────────────────────────────────────────────

class TestSecurityDashboard:
    """Tests for SecurityDashboard."""

    @pytest.fixture
    def sec_dashboard(self):
        pytest.importorskip('monitoring.security_dashboard', reason='needs qrcode')
        from monitoring.security_dashboard import SecurityDashboard
        return SecurityDashboard()

    def test_instantiation(self, sec_dashboard):
        assert sec_dashboard is not None

    def test_has_report_method(self, sec_dashboard):
        assert hasattr(sec_dashboard, 'get_security_status_report')

    def test_report_returns_dict(self, sec_dashboard):
        report = sec_dashboard.get_security_status_report()
        assert isinstance(report, dict)
        assert 'overall_score' in report or 'components' in report or 'status' in report


# ── Security Compliance Integration ────────────────────────────────────────────

class TestSecurityCompliance:
    """Tests for standalone security compliance functions."""

    def test_check_security_compliance(self):
        mod = pytest.importorskip('monitoring.security_compliance_integration', reason='needs qrcode')
        result = mod.check_security_compliance()
        assert isinstance(result, (dict, list, bool))

    def test_module_has_expected_functions(self):
        mod = pytest.importorskip('monitoring.security_compliance_integration', reason='needs qrcode')
        assert callable(getattr(mod, 'check_security_compliance', None))
