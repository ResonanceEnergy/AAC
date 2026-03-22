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
    async def test_market_intelligence_snapshot_is_included(self, dashboard):
        expected_snapshot = {
            'status': 'healthy',
            'market_tone': 'bullish',
            'put_call_ratio': 0.82,
        }
        dashboard.unusual_whales.get_snapshot = AsyncMock(return_value=expected_snapshot)

        result = await dashboard._get_market_intelligence()

        assert result == expected_snapshot

    @pytest.mark.asyncio
    async def test_get_pnl_data_uses_financial_engine_shape(self, dashboard):
        dashboard.financial_engine.update_risk_metrics = AsyncMock(
            return_value=MagicMock(max_drawdown_pct=4.2)
        )
        dashboard.financial_engine.calculate_portfolio_value = AsyncMock(return_value=1250000.0)
        dashboard.financial_engine.calculate_daily_pnl = AsyncMock(return_value=2500.0)
        dashboard.financial_engine.positions = {
            'SPY': MagicMock(unrealized_pnl=125.0),
            'QQQ': MagicMock(unrealized_pnl=-25.0),
        }

        result = await dashboard._get_pnl_data()

        assert result['daily_pnl'] == 2500.0
        assert result['total_equity'] == 1250000.0
        assert result['unrealized_pnl'] == 100.0
        assert result['max_drawdown'] == 4.2

    @pytest.mark.asyncio
    async def test_get_risk_metrics_maps_available_fields(self, dashboard):
        dashboard.financial_engine.update_risk_metrics = AsyncMock(
            return_value=MagicMock(
                stressed_var_99=2.5,
                tail_loss_p99=2.0,
                strategy_correlation=0.3,
                portfolio_heat=18.0,
                margin_buffer=92.0,
            )
        )

        result = await dashboard._get_risk_metrics()

        assert result['var_99'] == 2.5
        assert result['expected_shortfall'] == 2.0
        assert result['correlation_matrix']['strategy_correlation'] == 0.3
        assert result['stress_test_results']['portfolio_heat'] == 18.0

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

    @pytest.mark.asyncio
    async def test_report_returns_dict(self, sec_dashboard):
        report = await sec_dashboard.get_security_status_report()
        assert isinstance(report, dict)
        assert 'overall_security_score' in report or 'security_components' in report or 'status' in report


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


# ── Dashboard Fix Regression Tests ─────────────────────────────────────────────

class TestDashboardFixes:
    """Regression tests for critical dashboard bug fixes."""

    def test_data_lock_exists(self):
        """CRITICAL #5: _data_lock must exist for thread safety."""
        import threading
        from monitoring.aac_master_monitoring_dashboard import AACMasterMonitoringDashboard
        d = AACMasterMonitoringDashboard()
        assert hasattr(d, '_data_lock')
        assert isinstance(d._data_lock, type(threading.Lock()))

    def test_refresh_rate_minimum(self):
        """HIGH: refresh rate must be at least 1s even with bad env var."""
        import os
        old = os.environ.get('DASHBOARD_REFRESH_RATE')
        try:
            os.environ['DASHBOARD_REFRESH_RATE'] = '0.01'
            from monitoring.aac_master_monitoring_dashboard import AACMasterMonitoringDashboard
            d = AACMasterMonitoringDashboard()
            assert d.refresh_rate >= 1.0
        finally:
            if old is not None:
                os.environ['DASHBOARD_REFRESH_RATE'] = old
            else:
                os.environ.pop('DASHBOARD_REFRESH_RATE', None)

    def test_text_dashboard_uses_print(self):
        """CRITICAL #1: _display_text_dashboard must use print(), not logger.info()."""
        import inspect
        from monitoring.aac_master_monitoring_dashboard import AACMasterMonitoringDashboard
        source = inspect.getsource(AACMasterMonitoringDashboard._display_text_dashboard)
        # Should have print calls, not logger.info for display output
        assert 'print(' in source
        # logger.info should NOT appear in display method (it was the bug)
        assert 'logger.info(' not in source

    def test_ibkr_orders_no_hardcoded_balance(self):
        """CRITICAL #3: _get_ibkr_orders must not have hardcoded 920.0 balance."""
        import inspect
        from monitoring.aac_master_monitoring_dashboard import AACMasterMonitoringDashboard
        source = inspect.getsource(AACMasterMonitoringDashboard._get_ibkr_orders)
        assert '920.0' not in source
        assert '920' not in source.split('get_balances')[0]  # no hardcoded before balance fetch

    def test_collect_monitoring_data_logs_exc_info(self):
        """CRITICAL #4: collect_monitoring_data exception handler must include exc_info."""
        import inspect
        from monitoring.aac_master_monitoring_dashboard import AACMasterMonitoringDashboard
        source = inspect.getsource(AACMasterMonitoringDashboard.collect_monitoring_data)
        assert 'exc_info=True' in source


# ── Storm Lifeboat Integration Tests ──────────────────────────────────────────

class TestStormLifeboatIntegration:
    """Verify Storm Lifeboat is wired into all monitoring touch-points."""

    def test_storm_lifeboat_import_flag_exists(self):
        """Master dashboard should declare STORM_LIFEBOAT_AVAILABLE."""
        from monitoring import aac_master_monitoring_dashboard as mod
        assert hasattr(mod, 'STORM_LIFEBOAT_AVAILABLE')

    def test_collect_monitoring_data_has_storm_lifeboat_key(self):
        """collect_monitoring_data return dict must include 'storm_lifeboat'."""
        import inspect
        from monitoring.aac_master_monitoring_dashboard import AACMasterMonitoringDashboard
        source = inspect.getsource(AACMasterMonitoringDashboard.collect_monitoring_data)
        assert "'storm_lifeboat'" in source

    def test_get_storm_lifeboat_data_method_exists(self):
        """Dashboard must have _get_storm_lifeboat_data method."""
        from monitoring.aac_master_monitoring_dashboard import AACMasterMonitoringDashboard
        assert hasattr(AACMasterMonitoringDashboard, '_get_storm_lifeboat_data')
        assert callable(AACMasterMonitoringDashboard._get_storm_lifeboat_data)

    def test_storm_lifeboat_data_returns_dict(self):
        """_get_storm_lifeboat_data should return a dict with 'status' key."""
        from monitoring.aac_master_monitoring_dashboard import AACMasterMonitoringDashboard
        d = AACMasterMonitoringDashboard()
        result = d._get_storm_lifeboat_data()
        assert isinstance(result, dict)
        assert 'status' in result

    def test_storm_lifeboat_data_has_scenario_heatmap(self):
        """When available, result must include scenario_heatmap."""
        from monitoring.aac_master_monitoring_dashboard import (
            AACMasterMonitoringDashboard, STORM_LIFEBOAT_AVAILABLE,
        )
        if not STORM_LIFEBOAT_AVAILABLE:
            pytest.skip("Storm Lifeboat not importable")
        d = AACMasterMonitoringDashboard()
        result = d._get_storm_lifeboat_data()
        assert 'scenario_heatmap' in result
        assert isinstance(result['scenario_heatmap'], list)

    def test_storm_lifeboat_data_has_lunar(self):
        """When available, result must include lunar position."""
        from monitoring.aac_master_monitoring_dashboard import (
            AACMasterMonitoringDashboard, STORM_LIFEBOAT_AVAILABLE,
        )
        if not STORM_LIFEBOAT_AVAILABLE:
            pytest.skip("Storm Lifeboat not importable")
        d = AACMasterMonitoringDashboard()
        result = d._get_storm_lifeboat_data()
        assert 'lunar' in result
        lunar = result['lunar']
        assert 'moon_number' in lunar
        assert 'phase' in lunar
        assert 'in_phi_window' in lunar

    def test_registry_probes_storm_lifeboat(self):
        """SystemRegistry._probe_strategies must include Storm Lifeboat."""
        from monitoring.aac_system_registry import _probe_strategies
        results = _probe_strategies()
        names = [c.name for c in results]
        assert 'Storm Lifeboat Matrix' in names

    def test_continuous_monitoring_has_storm_lifeboat_alerts(self):
        """ContinuousMonitoringService must have _check_storm_lifeboat_alerts."""
        from monitoring.continuous_monitoring import ContinuousMonitoringService
        assert hasattr(ContinuousMonitoringService, '_check_storm_lifeboat_alerts')
        assert callable(ContinuousMonitoringService._check_storm_lifeboat_alerts)