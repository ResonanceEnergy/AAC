#!/usr/bin/env python3
"""
HTTP Health Server
==================
Simple HTTP server for health checks and metrics endpoints.
Designed for Kubernetes/Docker health probes and monitoring.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import aiohttp
try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None

from shared.monitoring import (
    HealthChecker,
    MetricsCollector,
    HealthStatus,
)

# Import circuit breaker to expose state
try:
    from shared.utils import _circuit_breakers
    CIRCUIT_BREAKERS_AVAILABLE = True
except ImportError:
    _circuit_breakers = {}
    CIRCUIT_BREAKERS_AVAILABLE = False

logger = logging.getLogger('HealthServer')


class TradingMetrics:
    """
    Trading-specific metrics for Prometheus monitoring.
    
    Thread-safe counters and gauges for trading activity.
    """
    
    def __init__(self):
        self._lock = asyncio.Lock() if asyncio else None
        
        # Counters (monotonically increasing)
        self.orders_total = 0
        self.orders_filled = 0
        self.orders_rejected = 0
        self.orders_cancelled = 0
        self.signals_total = 0
        self.arb_opportunities_total = 0
        self.arb_opportunities_taken = 0
        
        # Gauges (current values)
        self.open_positions = 0
        self.daily_pnl_usd = 0.0
        self.total_exposure_usd = 0.0
        
        # Histograms (we'll track averages for simplicity)
        self._order_latency_sum = 0.0
        self._order_latency_count = 0
        
    async def increment_orders(self, status: str = "created"):
        """Increment order counter by status"""
        self.orders_total += 1
        if status == "filled":
            self.orders_filled += 1
        elif status == "rejected":
            self.orders_rejected += 1
        elif status == "cancelled":
            self.orders_cancelled += 1
    
    async def increment_signals(self, count: int = 1):
        """Increment signal counter"""
        self.signals_total += count
    
    async def record_arb_opportunity(self, taken: bool = False):
        """Record an arbitrage opportunity"""
        self.arb_opportunities_total += 1
        if taken:
            self.arb_opportunities_taken += 1
    
    async def set_open_positions(self, count: int):
        """Set current open positions gauge"""
        self.open_positions = count
    
    async def set_daily_pnl(self, pnl: float):
        """Set daily P&L gauge"""
        self.daily_pnl_usd = pnl
    
    async def set_total_exposure(self, exposure: float):
        """Set total exposure gauge"""
        self.total_exposure_usd = exposure
    
    async def record_order_latency(self, latency_ms: float):
        """Record order execution latency"""
        self._order_latency_sum += latency_ms
        self._order_latency_count += 1
    
    @property
    def avg_order_latency_ms(self) -> float:
        """Get average order latency"""
        if self._order_latency_count == 0:
            return 0.0
        return self._order_latency_sum / self._order_latency_count
    
    def to_prometheus(self) -> List[str]:
        """Generate Prometheus-format metrics lines"""
        lines = [
            '# HELP acc_orders_total Total number of orders created',
            '# TYPE acc_orders_total counter',
            f'acc_orders_total {self.orders_total}',
            '',
            '# HELP acc_orders_filled Total orders filled',
            '# TYPE acc_orders_filled counter',
            f'acc_orders_filled {self.orders_filled}',
            '',
            '# HELP acc_orders_rejected Total orders rejected',
            '# TYPE acc_orders_rejected counter',
            f'acc_orders_rejected {self.orders_rejected}',
            '',
            '# HELP acc_orders_cancelled Total orders cancelled',
            '# TYPE acc_orders_cancelled counter',
            f'acc_orders_cancelled {self.orders_cancelled}',
            '',
            '# HELP acc_signals_total Total trading signals generated',
            '# TYPE acc_signals_total counter',
            f'acc_signals_total {self.signals_total}',
            '',
            '# HELP acc_arb_opportunities_total Total arbitrage opportunities detected',
            '# TYPE acc_arb_opportunities_total counter',
            f'acc_arb_opportunities_total {self.arb_opportunities_total}',
            '',
            '# HELP acc_arb_opportunities_taken Arbitrage opportunities executed',
            '# TYPE acc_arb_opportunities_taken counter',
            f'acc_arb_opportunities_taken {self.arb_opportunities_taken}',
            '',
            '# HELP acc_open_positions Current number of open positions',
            '# TYPE acc_open_positions gauge',
            f'acc_open_positions {self.open_positions}',
            '',
            '# HELP acc_daily_pnl_usd Daily profit/loss in USD',
            '# TYPE acc_daily_pnl_usd gauge',
            f'acc_daily_pnl_usd {self.daily_pnl_usd:.2f}',
            '',
            '# HELP acc_total_exposure_usd Total portfolio exposure in USD',
            '# TYPE acc_total_exposure_usd gauge',
            f'acc_total_exposure_usd {self.total_exposure_usd:.2f}',
            '',
            '# HELP acc_order_latency_avg_ms Average order execution latency',
            '# TYPE acc_order_latency_avg_ms gauge',
            f'acc_order_latency_avg_ms {self.avg_order_latency_ms:.2f}',
        ]
        return lines


# Global trading metrics instance
_trading_metrics: Optional['TradingMetrics'] = None


def get_trading_metrics() -> TradingMetrics:
    """Get or create the global trading metrics instance"""
    global _trading_metrics
    if _trading_metrics is None:
        _trading_metrics = TradingMetrics()
    return _trading_metrics


class HealthServer:
    """
    HTTP server providing health check and metrics endpoints.
    
    Endpoints:
    - GET /health - Basic health check (returns 200 if alive)
    - GET /health/live - Liveness probe (always 200 if server running)
    - GET /health/ready - Readiness probe (200 if all checks pass)
    - GET /health/detailed - Detailed health status JSON
    - GET /metrics - System metrics JSON
    
    Usage:
        server = HealthServer(port=8080)
        await server.start()
        # ... application runs ...
        await server.stop()
    """
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 8080,
        health_checker: Optional[HealthChecker] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp required for health server. Install: pip install aiohttp")
        
        self.host = host
        self.port = port
        self.health_checker = health_checker or HealthChecker()
        self.metrics_collector = metrics_collector or MetricsCollector()
        
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._running = False
        
        # Track startup time
        self._started_at: Optional[datetime] = None
        
        # Custom readiness checks
        self._readiness_checks: Dict[str, callable] = {}
        
        # WebSocket feed manager reference (optional)
        self._websocket_feeds = None
    
    def register_websocket_feeds(self, feed_manager):
        """
        Register WebSocket feed manager for health monitoring.
        
        Args:
            feed_manager: PriceFeedManager instance
        """
        self._websocket_feeds = feed_manager
        logger.info("WebSocket feeds registered with health server")
    
    def add_readiness_check(self, name: str, check_fn: callable):
        """Add a custom readiness check"""
        self._readiness_checks[name] = check_fn
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Basic health check - returns 200 if server is running"""
        return web.json_response({
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
        })
    
    async def _handle_liveness(self, request: web.Request) -> web.Response:
        """
        Liveness probe - always returns 200 if server is responding.
        Kubernetes uses this to know if container needs restart.
        """
        return web.json_response({
            'alive': True,
            'uptime_seconds': self._get_uptime(),
        })
    
    async def _handle_readiness(self, request: web.Request) -> web.Response:
        """
        Readiness probe - returns 200 only if system is ready to serve.
        Kubernetes uses this to know if pod should receive traffic.
        """
        # Run all health checks
        results = await self.health_checker.run_all_checks()
        overall = self.health_checker.get_overall_status()
        
        # Run custom readiness checks
        for name, check_fn in self._readiness_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_fn):
                    passed = await check_fn()
                else:
                    passed = check_fn()
                if not passed:
                    overall = HealthStatus.UNHEALTHY
            except Exception as e:
                logger.error(f"Readiness check {name} failed: {e}")
                overall = HealthStatus.UNHEALTHY
        
        is_ready = overall in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        status_code = 200 if is_ready else 503
        
        return web.json_response(
            {
                'ready': is_ready,
                'status': overall.value,
                'checks': {
                    name: result.status.value 
                    for name, result in results.items()
                },
            },
            status=status_code,
        )
    
    async def _handle_detailed_health(self, request: web.Request) -> web.Response:
        """Detailed health status with all check results"""
        results = await self.health_checker.run_all_checks()
        overall = self.health_checker.get_overall_status()
        
        response_data = {
            'status': overall.value,
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': self._get_uptime(),
            'checks': {
                name: {
                    'status': result.status.value,
                    'message': result.message,
                    'latency_ms': result.latency_ms,
                    'details': result.details,
                }
                for name, result in results.items()
            },
        }
        
        # Add WebSocket feed status if registered
        if self._websocket_feeds:
            try:
                feed_status = self._websocket_feeds.get_status()
                response_data['websocket_feeds'] = feed_status
            except Exception as e:
                response_data['websocket_feeds'] = {'error': str(e)}
        
        return web.json_response(response_data)
    
    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """System metrics endpoint"""
        metrics = self.metrics_collector.collect()
        averages = self.metrics_collector.get_averages(minutes=5)
        
        return web.json_response({
            'current': {
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'memory_used_mb': metrics.memory_used_mb,
                'disk_percent': metrics.disk_percent,
                'open_files': metrics.open_files,
                'threads': metrics.threads,
                'process_cpu': metrics.process_cpu_percent,
                'process_memory_mb': metrics.process_memory_mb,
            },
            'averages_5min': averages,
            'timestamp': metrics.timestamp.isoformat(),
        })
    
    async def _handle_prometheus(self, request: web.Request) -> web.Response:
        """
        Prometheus-compatible metrics endpoint.
        Returns metrics in Prometheus text format.
        """
        metrics = self.metrics_collector.collect()
        
        # System metrics
        lines = [
            '# HELP acc_cpu_percent CPU usage percentage',
            '# TYPE acc_cpu_percent gauge',
            f'acc_cpu_percent {metrics.cpu_percent}',
            '',
            '# HELP acc_memory_percent Memory usage percentage',
            '# TYPE acc_memory_percent gauge',
            f'acc_memory_percent {metrics.memory_percent}',
            '',
            '# HELP acc_memory_used_bytes Memory used in bytes',
            '# TYPE acc_memory_used_bytes gauge',
            f'acc_memory_used_bytes {int(metrics.memory_used_mb * 1024 * 1024)}',
            '',
            '# HELP acc_disk_percent Disk usage percentage',
            '# TYPE acc_disk_percent gauge',
            f'acc_disk_percent {metrics.disk_percent}',
            '',
            '# HELP acc_process_threads Number of threads',
            '# TYPE acc_process_threads gauge',
            f'acc_process_threads {metrics.threads}',
            '',
            '# HELP acc_uptime_seconds Uptime in seconds',
            '# TYPE acc_uptime_seconds counter',
            f'acc_uptime_seconds {self._get_uptime()}',
        ]
        
        # Add trading metrics
        trading_metrics = get_trading_metrics()
        lines.extend(['', '# ===== Trading Metrics ====='])
        lines.extend(trading_metrics.to_prometheus())
        
        # Add circuit breaker states
        if CIRCUIT_BREAKERS_AVAILABLE and _circuit_breakers:
            lines.extend([
                '',
                '# ===== Circuit Breaker States =====',
                '# HELP acc_circuit_breaker_state Circuit breaker state (0=closed, 1=half-open, 2=open)',
                '# TYPE acc_circuit_breaker_state gauge',
            ])
            for name, breaker in _circuit_breakers.items():
                state_value = 0  # closed
                if breaker.state.value == 'half_open':
                    state_value = 1
                elif breaker.state.value == 'open':
                    state_value = 2
                lines.append(f'acc_circuit_breaker_state{{name="{name}"}} {state_value}')
            
            lines.extend([
                '',
                '# HELP acc_circuit_breaker_failures Circuit breaker failure count',
                '# TYPE acc_circuit_breaker_failures gauge',
            ])
            for name, breaker in _circuit_breakers.items():
                lines.append(f'acc_circuit_breaker_failures{{name="{name}"}} {breaker.failure_count}')
        
        # Add WebSocket feed status if registered
        if self._websocket_feeds:
            try:
                feed_status = self._websocket_feeds.get_status()
                lines.extend([
                    '',
                    '# ===== WebSocket Feed Status =====',
                    '# HELP acc_websocket_connected WebSocket feed connection status (1=connected)',
                    '# TYPE acc_websocket_connected gauge',
                ])
                for exchange, status in feed_status.items():
                    connected = 1 if status.get('connected', False) else 0
                    lines.append(f'acc_websocket_connected{{exchange="{exchange}"}} {connected}')
            except Exception:
                pass
        
        return web.Response(
            text='\n'.join(lines) + '\n',
            content_type='text/plain; version=0.0.4',
        )
    
    def _get_uptime(self) -> float:
        """Get server uptime in seconds"""
        if self._started_at:
            return (datetime.now() - self._started_at).total_seconds()
        return 0.0
    
    async def start(self):
        """Start the health server"""
        if self._running:
            return
        
        self._app = web.Application()
        self._app.router.add_get('/health', self._handle_health)
        self._app.router.add_get('/health/live', self._handle_liveness)
        self._app.router.add_get('/health/ready', self._handle_readiness)
        self._app.router.add_get('/health/detailed', self._handle_detailed_health)
        self._app.router.add_get('/metrics', self._handle_metrics)
        self._app.router.add_get('/metrics/prometheus', self._handle_prometheus)
        
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        
        self._started_at = datetime.now()
        self._running = True
        
        logger.info(f"Health server started on http://{self.host}:{self.port}")
        logger.info(f"  Endpoints: /health, /health/live, /health/ready, /health/detailed, /metrics")
    
    async def stop(self):
        """Stop the health server"""
        if not self._running:
            return
        
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        
        self._running = False
        logger.info("Health server stopped")


# Convenience function to run standalone
async def run_health_server(port: int = 8080):
    """Run health server standalone (for testing)"""
    server = HealthServer(port=port)
    await server.start()
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await server.stop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_health_server())
