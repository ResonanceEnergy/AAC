from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared import health_server as hs_mod
from shared.health_server import (
    HealthServer,
    TradingMetrics,
    get_trading_metrics,
)
from shared.monitoring import HealthCheckResult, HealthStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_trading_metrics():
    hs_mod._trading_metrics = None
    yield
    hs_mod._trading_metrics = None


def _fake_metrics(**overrides):
    base = SimpleNamespace(
        cpu_percent=12.5,
        memory_percent=42.0,
        memory_used_mb=2048.0,
        disk_percent=55.0,
        disk_used_gb=120.0,
        open_files=10,
        threads=8,
        process_cpu_percent=3.0,
        process_memory_mb=128.0,
        timestamp=datetime(2026, 4, 24, 12, 0, 0),
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _hr(name, status, message="", latency_ms=1.0, details=None):
    return HealthCheckResult(
        name=name,
        status=status,
        message=message,
        latency_ms=latency_ms,
        details=details or {},
    )


@pytest.fixture
def server():
    hc = MagicMock()
    hc.run_all_checks = AsyncMock(return_value={})
    hc.get_overall_status = MagicMock(return_value=HealthStatus.HEALTHY)
    mc = MagicMock()
    mc.collect = MagicMock(return_value=_fake_metrics())
    mc.get_averages = MagicMock(return_value={"avg_cpu": 10.0})
    s = HealthServer(host="127.0.0.1", port=0, health_checker=hc, metrics_collector=mc)
    return s


def _body_json(resp):
    return json.loads(resp.body)


# ---------------------------------------------------------------------------
# TradingMetrics
# ---------------------------------------------------------------------------


class TestTradingMetricsCounters:
    @pytest.mark.asyncio
    async def test_increment_orders_default_only_total(self):
        m = TradingMetrics()
        await m.increment_orders()
        assert m.orders_total == 1
        assert m.orders_filled == 0
        assert m.orders_rejected == 0
        assert m.orders_cancelled == 0

    @pytest.mark.asyncio
    async def test_increment_orders_filled(self):
        m = TradingMetrics()
        await m.increment_orders("filled")
        assert m.orders_total == 1
        assert m.orders_filled == 1

    @pytest.mark.asyncio
    async def test_increment_orders_rejected(self):
        m = TradingMetrics()
        await m.increment_orders("rejected")
        assert m.orders_rejected == 1

    @pytest.mark.asyncio
    async def test_increment_orders_cancelled(self):
        m = TradingMetrics()
        await m.increment_orders("cancelled")
        assert m.orders_cancelled == 1

    @pytest.mark.asyncio
    async def test_increment_signals_default_one(self):
        m = TradingMetrics()
        await m.increment_signals()
        assert m.signals_total == 1

    @pytest.mark.asyncio
    async def test_increment_signals_count(self):
        m = TradingMetrics()
        await m.increment_signals(7)
        assert m.signals_total == 7

    @pytest.mark.asyncio
    async def test_record_arb_opportunity_not_taken(self):
        m = TradingMetrics()
        await m.record_arb_opportunity()
        assert m.arb_opportunities_total == 1
        assert m.arb_opportunities_taken == 0

    @pytest.mark.asyncio
    async def test_record_arb_opportunity_taken(self):
        m = TradingMetrics()
        await m.record_arb_opportunity(True)
        assert m.arb_opportunities_total == 1
        assert m.arb_opportunities_taken == 1


class TestTradingMetricsGauges:
    @pytest.mark.asyncio
    async def test_set_open_positions(self):
        m = TradingMetrics()
        await m.set_open_positions(5)
        assert m.open_positions == 5

    @pytest.mark.asyncio
    async def test_set_daily_pnl(self):
        m = TradingMetrics()
        await m.set_daily_pnl(123.45)
        assert m.daily_pnl_usd == pytest.approx(123.45)

    @pytest.mark.asyncio
    async def test_set_total_exposure(self):
        m = TradingMetrics()
        await m.set_total_exposure(50000.0)
        assert m.total_exposure_usd == pytest.approx(50000.0)


class TestTradingMetricsLatency:
    def test_avg_latency_zero_when_no_samples(self):
        m = TradingMetrics()
        assert m.avg_order_latency_ms == 0.0

    @pytest.mark.asyncio
    async def test_avg_latency_computed(self):
        m = TradingMetrics()
        await m.record_order_latency(10.0)
        await m.record_order_latency(20.0)
        await m.record_order_latency(30.0)
        assert m.avg_order_latency_ms == pytest.approx(20.0)


class TestTradingMetricsPrometheus:
    def test_to_prometheus_includes_all_metrics(self):
        m = TradingMetrics()
        out = "\n".join(m.to_prometheus())
        for key in [
            "acc_orders_total",
            "acc_orders_filled",
            "acc_orders_rejected",
            "acc_orders_cancelled",
            "acc_signals_total",
            "acc_arb_opportunities_total",
            "acc_arb_opportunities_taken",
            "acc_open_positions",
            "acc_daily_pnl_usd",
            "acc_total_exposure_usd",
            "acc_order_latency_avg_ms",
        ]:
            assert key in out

    def test_to_prometheus_returns_list(self):
        m = TradingMetrics()
        out = m.to_prometheus()
        assert isinstance(out, list)
        assert all(isinstance(line, str) for line in out)

    @pytest.mark.asyncio
    async def test_prometheus_reflects_state(self):
        m = TradingMetrics()
        await m.set_open_positions(3)
        await m.set_daily_pnl(100.5)
        out = "\n".join(m.to_prometheus())
        assert "acc_open_positions 3" in out
        assert "acc_daily_pnl_usd 100.50" in out


class TestGetTradingMetricsSingleton:
    def test_returns_singleton(self):
        a = get_trading_metrics()
        b = get_trading_metrics()
        assert a is b
        assert isinstance(a, TradingMetrics)


# ---------------------------------------------------------------------------
# HealthServer init
# ---------------------------------------------------------------------------


class TestHealthServerInit:
    def test_defaults(self):
        s = HealthServer()
        assert s.host == "127.0.0.1"
        assert s.port == 8080
        assert s._running is False
        assert s._started_at is None

    def test_custom_host_port(self):
        s = HealthServer(host="0.0.0.0", port=9999)
        assert s.host == "0.0.0.0"
        assert s.port == 9999

    def test_uses_provided_health_checker(self):
        hc = MagicMock()
        s = HealthServer(health_checker=hc)
        assert s.health_checker is hc

    def test_uses_provided_metrics_collector(self):
        mc = MagicMock()
        s = HealthServer(metrics_collector=mc)
        assert s.metrics_collector is mc

    def test_raises_when_aiohttp_missing(self):
        with patch.object(hs_mod, "AIOHTTP_AVAILABLE", False):
            with pytest.raises(ImportError):
                HealthServer()


class TestRegisterWebSocketFeeds:
    def test_register(self, server):
        feed_mgr = MagicMock()
        server.register_websocket_feeds(feed_mgr)
        assert server._websocket_feeds is feed_mgr


class TestAddReadinessCheck:
    def test_add_readiness_check(self, server):
        def chk():
            return True
        server.add_readiness_check("custom", chk)
        assert "custom" in server._readiness_checks


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


class TestHandleHealth:
    @pytest.mark.asyncio
    async def test_returns_ok(self, server):
        resp = await server._handle_health(MagicMock())
        body = _body_json(resp)
        assert body["status"] == "ok"
        assert "timestamp" in body


class TestHandleLiveness:
    @pytest.mark.asyncio
    async def test_alive_true(self, server):
        resp = await server._handle_liveness(MagicMock())
        body = _body_json(resp)
        assert body["alive"] is True
        assert body["uptime_seconds"] == 0.0  # not started

    @pytest.mark.asyncio
    async def test_uptime_after_started(self, server):
        server._started_at = datetime.now() - timedelta(seconds=5)
        resp = await server._handle_liveness(MagicMock())
        body = _body_json(resp)
        assert body["uptime_seconds"] >= 5.0


class TestHandleReadiness:
    @pytest.mark.asyncio
    async def test_ready_when_healthy(self, server):
        server.health_checker.run_all_checks = AsyncMock(
            return_value={"db": _hr("db", HealthStatus.HEALTHY)}
        )
        server.health_checker.get_overall_status = MagicMock(return_value=HealthStatus.HEALTHY)
        resp = await server._handle_readiness(MagicMock())
        assert resp.status == 200
        body = _body_json(resp)
        assert body["ready"] is True
        assert body["status"] == "healthy"
        assert body["checks"]["db"] == "healthy"

    @pytest.mark.asyncio
    async def test_ready_when_degraded(self, server):
        server.health_checker.get_overall_status = MagicMock(return_value=HealthStatus.DEGRADED)
        resp = await server._handle_readiness(MagicMock())
        assert resp.status == 200
        assert _body_json(resp)["ready"] is True

    @pytest.mark.asyncio
    async def test_not_ready_when_unhealthy(self, server):
        server.health_checker.get_overall_status = MagicMock(return_value=HealthStatus.UNHEALTHY)
        resp = await server._handle_readiness(MagicMock())
        assert resp.status == 503
        assert _body_json(resp)["ready"] is False

    @pytest.mark.asyncio
    async def test_custom_sync_check_failing_makes_unhealthy(self, server):
        server.add_readiness_check("nope", lambda: False)
        resp = await server._handle_readiness(MagicMock())
        assert resp.status == 503

    @pytest.mark.asyncio
    async def test_custom_async_check_passing(self, server):
        async def chk():
            return True
        server.add_readiness_check("ok", chk)
        resp = await server._handle_readiness(MagicMock())
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_custom_check_exception_makes_unhealthy(self, server):
        def boom():
            raise RuntimeError("x")
        server.add_readiness_check("boom", boom)
        resp = await server._handle_readiness(MagicMock())
        assert resp.status == 503


class TestHandleDetailedHealth:
    @pytest.mark.asyncio
    async def test_returns_full_payload(self, server):
        server.health_checker.run_all_checks = AsyncMock(
            return_value={"db": _hr("db", HealthStatus.HEALTHY, "ok", 12.0, {"k": "v"})}
        )
        server.health_checker.get_overall_status = MagicMock(return_value=HealthStatus.HEALTHY)
        resp = await server._handle_detailed_health(MagicMock())
        body = _body_json(resp)
        assert body["status"] == "healthy"
        assert "uptime_seconds" in body
        assert body["checks"]["db"]["status"] == "healthy"
        assert body["checks"]["db"]["latency_ms"] == 12.0
        assert body["checks"]["db"]["details"] == {"k": "v"}

    @pytest.mark.asyncio
    async def test_includes_websocket_status_when_registered(self, server):
        feed_mgr = MagicMock()
        feed_mgr.get_status = MagicMock(return_value={"binance": {"connected": True}})
        server.register_websocket_feeds(feed_mgr)
        resp = await server._handle_detailed_health(MagicMock())
        body = _body_json(resp)
        assert body["websocket_feeds"] == {"binance": {"connected": True}}

    @pytest.mark.asyncio
    async def test_websocket_error_recorded(self, server):
        feed_mgr = MagicMock()
        feed_mgr.get_status = MagicMock(side_effect=RuntimeError("dead"))
        server.register_websocket_feeds(feed_mgr)
        resp = await server._handle_detailed_health(MagicMock())
        body = _body_json(resp)
        assert "error" in body["websocket_feeds"]


class TestHandleMetrics:
    @pytest.mark.asyncio
    async def test_returns_metrics(self, server):
        resp = await server._handle_metrics(MagicMock())
        body = _body_json(resp)
        assert body["current"]["cpu_percent"] == 12.5
        assert body["current"]["memory_percent"] == 42.0
        assert body["averages_5min"] == {"avg_cpu": 10.0}
        assert "timestamp" in body


class TestHandlePrometheus:
    @pytest.mark.asyncio
    async def test_basic_system_metrics(self, server):
        resp = await server._handle_prometheus(MagicMock())
        text = resp.text
        for k in [
            "acc_cpu_percent 12.5",
            "acc_memory_percent 42.0",
            "acc_disk_percent 55.0",
            "acc_process_threads 8",
            "acc_uptime_seconds",
            "# ===== Trading Metrics =====",
            "acc_orders_total 0",
        ]:
            assert k in text

    @pytest.mark.asyncio
    async def test_content_type_is_prometheus_text(self, server):
        resp = await server._handle_prometheus(MagicMock())
        assert "text/plain" in resp.content_type

    @pytest.mark.asyncio
    async def test_circuit_breaker_states_emitted(self, server):
        breaker = MagicMock()
        breaker.state = MagicMock(value="open")
        breaker.failure_count = 4
        with patch.object(hs_mod, "CIRCUIT_BREAKERS_AVAILABLE", True), \
             patch.object(hs_mod, "_circuit_breakers", {"ibkr": breaker}):
            resp = await server._handle_prometheus(MagicMock())
        text = resp.text
        assert "acc_circuit_breaker_state" in text
        assert 'name="ibkr"} 2' in text
        assert "acc_circuit_breaker_failures" in text
        assert 'name="ibkr"} 4' in text

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_state(self, server):
        breaker = MagicMock()
        breaker.state = MagicMock(value="half_open")
        breaker.failure_count = 1
        with patch.object(hs_mod, "CIRCUIT_BREAKERS_AVAILABLE", True), \
             patch.object(hs_mod, "_circuit_breakers", {"ibkr": breaker}):
            resp = await server._handle_prometheus(MagicMock())
        assert 'name="ibkr"} 1' in resp.text

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self, server):
        breaker = MagicMock()
        breaker.state = MagicMock(value="closed")
        breaker.failure_count = 0
        with patch.object(hs_mod, "CIRCUIT_BREAKERS_AVAILABLE", True), \
             patch.object(hs_mod, "_circuit_breakers", {"ibkr": breaker}):
            resp = await server._handle_prometheus(MagicMock())
        assert 'acc_circuit_breaker_state{name="ibkr"} 0' in resp.text

    @pytest.mark.asyncio
    async def test_websocket_feed_status_emitted(self, server):
        feed_mgr = MagicMock()
        feed_mgr.get_status = MagicMock(
            return_value={"binance": {"connected": True}, "kraken": {"connected": False}}
        )
        server.register_websocket_feeds(feed_mgr)
        resp = await server._handle_prometheus(MagicMock())
        text = resp.text
        assert 'acc_websocket_connected{exchange="binance"} 1' in text
        assert 'acc_websocket_connected{exchange="kraken"} 0' in text

    @pytest.mark.asyncio
    async def test_websocket_failure_swallowed(self, server):
        feed_mgr = MagicMock()
        feed_mgr.get_status = MagicMock(side_effect=RuntimeError("dead"))
        server.register_websocket_feeds(feed_mgr)
        resp = await server._handle_prometheus(MagicMock())
        # Should still respond successfully
        assert resp.status == 200


# ---------------------------------------------------------------------------
# Uptime
# ---------------------------------------------------------------------------


class TestGetUptime:
    def test_zero_when_not_started(self, server):
        assert server._get_uptime() == 0.0

    def test_positive_when_started(self, server):
        server._started_at = datetime.now() - timedelta(seconds=10)
        assert server._get_uptime() >= 10.0


# ---------------------------------------------------------------------------
# Start / Stop
# ---------------------------------------------------------------------------


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_and_stop(self, server):
        # Use port 0 → OS picks free port
        server.port = 0
        await server.start()
        assert server._running is True
        assert server._started_at is not None
        assert server._app is not None
        await server.stop()
        assert server._running is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self, server):
        server.port = 0
        await server.start()
        runner_first = server._runner
        await server.start()  # second call no-op
        assert server._runner is runner_first
        await server.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running_noop(self, server):
        await server.stop()  # should not raise
        assert server._running is False


# ---------------------------------------------------------------------------
# run_health_server convenience
# ---------------------------------------------------------------------------


class TestRunHealthServer:
    @pytest.mark.asyncio
    async def test_starts_and_cancels(self):
        # Create the coroutine task and cancel it after a short wait
        task = asyncio.create_task(hs_mod.run_health_server(port=0))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Should not raise other exceptions
