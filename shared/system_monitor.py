"""
shared.system_monitor — AAC System Monitor (terminal-based)
============================================================

Provides a terminal-based system monitoring view that refreshes
periodically. This is the canonical `SystemMonitor`; the
`SharedInfrastructure.system_monitor` module re-exports from here.

Usage:
    python -m shared.system_monitor          # 30-second refresh
    python -m shared.system_monitor --fast   # 5-second refresh
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from shared.monitoring import (
    get_monitoring_service,
)

logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Terminal-based system monitor with periodic refresh.

    Wraps `MonitoringService` and renders a live dashboard
    to stdout every `refresh_seconds`.
    """

    def __init__(self, refresh_seconds: int = 30):
        self.refresh_seconds = refresh_seconds
        self.service = get_monitoring_service()
        self._running = False
        self._cycle = 0

    # ── rendering helpers ────────────────────────────────────────

    @staticmethod
    def _clear() -> None:
        # Use ANSI escape codes instead of os.system() to avoid shell injection
        print("\033[2J\033[H", end="", flush=True)

    def _render_banner(self) -> None:
        print()
        print("  ╔══════════════════════════════════════════════╗")
        print("  ║   BARREN WUFFET — System Monitor             ║")
        print("  ║   Codename: AZ SUPREME                       ║")
        print("  ╚══════════════════════════════════════════════╝")
        print()

    def _render_metrics(self) -> None:
        if not PSUTIL_AVAILABLE:
            print("  [!] psutil not installed — metrics unavailable")
            return

        metrics = self.service.metrics_collector.collect()
        print("  ┌─── System Resources ────────────────────────┐")
        print(f"  │  CPU         {metrics.cpu_percent:6.1f}%                      │")
        print(f"  │  Memory      {metrics.memory_percent:6.1f}%  ({metrics.memory_used_mb:,.0f} MB)       │")
        print(f"  │  Disk        {metrics.disk_percent:6.1f}%  ({metrics.disk_used_gb:,.1f} GB)        │")
        print(f"  │  Threads     {metrics.threads:6d}                       │")
        print("  └─────────────────────────────────────────────┘")
        print()

    async def _render_health(self) -> None:
        results = await self.service.health_checker.run_all_checks()
        overall = self.service.health_checker.get_overall_status()

        icon = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}.get(overall.value, "❓")
        print(f"  Overall Health: {icon} {overall.value.upper()}")
        print()
        for name, r in results.items():
            ic = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}.get(r.status.value, "❓")
            print(f"    {ic} {name:<20s}  {r.message}")
        print()

    def _render_alerts(self) -> None:
        unack = self.service.alert_manager.get_unacknowledged()
        critical = [a for a in unack if a.severity == "critical"]
        warnings = [a for a in unack if a.severity == "warning"]

        if critical:
            print(f"  🚨 Critical Alerts: {len(critical)}")
            for a in critical[-3:]:
                print(f"       {a.title}: {a.message}")
        if warnings:
            print(f"  ⚠️  Warnings: {len(warnings)}")
        if not critical and not warnings:
            print("  ✅ No active alerts")
        print()

    def _render_footer(self) -> None:
        print(f"  Cycle #{self._cycle}  |  Refresh every {self.refresh_seconds}s  |  Ctrl+C to quit")
        print(f"  Last update: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print()

    # ── main loop ────────────────────────────────────────────────

    async def run(self) -> None:
        """Run the monitor loop until interrupted."""
        self._running = True
        logger.info("SystemMonitor started (refresh=%ds)", self.refresh_seconds)

        try:
            while self._running:
                self._cycle += 1
                self._clear()
                self._render_banner()
                self._render_metrics()
                await self._render_health()
                self._render_alerts()
                self._render_footer()
                await asyncio.sleep(self.refresh_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            print("\n  System Monitor stopped.\n")

    def stop(self) -> None:
        """Stop."""
        self._running = False


# ── module-level entry point ────────────────────────────────────

async def _async_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="BARREN WUFFET System Monitor")
    parser.add_argument("--fast", action="store_true", help="5-second refresh instead of 30")
    parser.add_argument("--interval", type=int, default=None, help="Custom refresh interval")
    args = parser.parse_args()

    interval = args.interval or (5 if args.fast else 30)
    monitor = SystemMonitor(refresh_seconds=interval)
    await monitor.run()


def main() -> None:
    """Main."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
