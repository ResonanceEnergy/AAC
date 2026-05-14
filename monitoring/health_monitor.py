from __future__ import annotations

"""
monitoring.health_monitor — Lightweight real-data health monitor for AAC.

Collects:
  - API health for the working data sources (yfinance, FRED, Finnhub,
    CoinGecko, IBKR port, Unusual Whales)
  - Open position count + today's P&L from PnLTracker
  - Active alerts via shared.alert_manager

Usage:
    python -m monitoring.health_monitor          # single snapshot
    python -m monitoring.health_monitor --loop   # refresh every 30 s
"""

import os
import socket
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any

import structlog

from shared.alert_manager import AlertManager
from shared.alert_manager import AlertSeverity
from shared.health_checker import ComponentHealth
from shared.health_checker import HealthChecker
from shared.health_checker import HealthStatus

_log = structlog.get_logger()


# ── individual check functions ────────────────────────────────────────────
# Each returns {"status": "healthy"|"degraded"|"unhealthy", "note": str}.
# HealthChecker wraps each in try/except and measures latency.


def _check_yfinance() -> dict[str, Any]:
    """Verify yfinance can return a live SPY price quote."""
    try:
        import yfinance as yf  # noqa: PLC0415

        ticker = yf.Ticker("SPY")
        fi = ticker.fast_info
        # yfinance renamed lastPrice -> last_price; FastInfo also supports dict access
        price = getattr(fi, "last_price", None)
        if price is None:
            try:
                price = fi["last_price"]
            except (KeyError, TypeError):
                price = None
        note = f"SPY ${price:.2f}" if isinstance(price, (int, float)) else "SPY ok"
        return {"status": "healthy", "note": note}
    except Exception as exc:
        return {"status": "unhealthy", "note": str(exc)[:60]}


def _check_fred() -> dict[str, Any]:
    """FRED: verify key is configured."""
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        return {"status": "degraded", "note": "FRED_API_KEY not set"}
    return {"status": "healthy", "note": "key configured"}


def _check_finnhub() -> dict[str, Any]:
    """Finnhub: verify key is configured."""
    key = os.getenv("FINNHUB_API_KEY", "")
    if not key:
        return {"status": "degraded", "note": "FINNHUB_API_KEY not set"}
    return {"status": "healthy", "note": "key configured"}


def _check_coingecko() -> dict[str, Any]:
    """CoinGecko: free-tier always reachable; report whether key is set."""
    key = os.getenv("COINGECKO_API_KEY", "")
    note = "key configured (free tier)" if key else "free tier (no key)"
    return {"status": "healthy", "note": note}


def _check_ibkr_port() -> dict[str, Any]:
    """IBKR: verify TWS/Gateway is listening on port 7496 or 7497."""
    for port in (7496, 7497):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return {"status": "healthy", "note": f"port {port} reachable"}
        except OSError:
            continue
    return {"status": "degraded", "note": "ports 7496/7497 closed"}


def _check_unusual_whales() -> dict[str, Any]:
    """Unusual Whales: key status + known field-parsing issue."""
    key = os.getenv("UNUSUAL_WHALES_API_KEY", "")
    if not key:
        return {"status": "degraded", "note": "no key set"}
    return {"status": "degraded", "note": "key set; field parsing broken"}


# Default check registry used by HealthMonitor
_DEFAULT_CHECKS: dict[str, Any] = {
    "yfinance": _check_yfinance,
    "fred": _check_fred,
    "finnhub": _check_finnhub,
    "coingecko": _check_coingecko,
    "ibkr": _check_ibkr_port,
    "unusual_whales": _check_unusual_whales,
}


# ── snapshot dataclass ─────────────────────────────────────────────────────


@dataclass
class SystemSnapshot:
    """Aggregated real-time system state for display and alerting."""

    checked_at: datetime
    api_health: dict[str, ComponentHealth]
    overall_status: HealthStatus
    system_resources: ComponentHealth | None
    positions_count: int
    pnl_today: float | None
    last_trade_symbol: str | None
    last_trade_at: str | None
    active_alerts: list[dict[str, Any]] = field(default_factory=list)


# ── main class ─────────────────────────────────────────────────────────────


class HealthMonitor:
    """
    Lightweight health monitor for AAC.

    Collects real system state — API health, P&L, positions, alerts —
    and renders it for terminal display.

    Args:
        checks:        Custom check registry (name → callable).  Defaults
                       to _DEFAULT_CHECKS (yfinance, FRED, Finnhub, …).
        alert_manager: Shared AlertManager instance.  A new one is created
                       per HealthMonitor if not provided.
    """

    def __init__(
        self,
        checks: dict[str, Any] | None = None,
        alert_manager: AlertManager | None = None,
    ) -> None:
        self._checker = HealthChecker()
        self._alerts = alert_manager or AlertManager()
        check_map = checks if checks is not None else _DEFAULT_CHECKS
        for name, fn in check_map.items():
            self._checker.register_check(name, fn)

    # ── collection ──────────────────────────────────────────────────

    def collect_snapshot(self) -> SystemSnapshot:
        """Run all health checks and assemble a SystemSnapshot."""
        api_health = self._checker.run_all()
        overall = self._checker.overall_status()
        system = self._checker.check_system()

        # P&L and positions from PnLTracker (graceful fallback)
        positions_count = 0
        pnl_today: float | None = None
        last_trade_symbol: str | None = None
        last_trade_at: str | None = None
        try:
            from CentralAccounting.pnl_tracker import PnLTracker  # noqa: PLC0415

            tracker = PnLTracker()
            report = tracker.today_report()
            positions_count = len(report.get("positions", []))
            pnl_row = report.get("daily_pnl")
            if isinstance(pnl_row, dict):
                pnl_today = float(pnl_row.get("total_unrealized_pnl", 0.0)) + float(
                    pnl_row.get("total_realized_pnl", 0.0)
                )
            trades = report.get("today_trades", [])
            if trades:
                t = trades[-1]
                last_trade_symbol = t.get("symbol")
                raw_at = t.get("logged_at", "")
                last_trade_at = raw_at[:16] if raw_at else None
        except Exception as exc:
            _log.warning("pnl_tracker unavailable", error=str(exc))

        # Fire ERROR alert for any fully-down API
        for name, health in api_health.items():
            if health.status == HealthStatus.UNHEALTHY:
                self._alerts.fire(
                    title=f"API DOWN: {name}",
                    message=health.error or "health check failed",
                    severity=AlertSeverity.ERROR,
                    source="health_monitor",
                    category="api",
                )

        active_alerts = [
            {
                "title": a.title,
                "severity": a.severity.value,
                "count": a.count,
            }
            for a in self._alerts.get_active_alerts()
        ]

        return SystemSnapshot(
            checked_at=datetime.now(),
            api_health=api_health,
            overall_status=overall,
            system_resources=system,
            positions_count=positions_count,
            pnl_today=pnl_today,
            last_trade_symbol=last_trade_symbol,
            last_trade_at=last_trade_at,
            active_alerts=active_alerts,
        )

    # ── display ─────────────────────────────────────────────────────

    @staticmethod
    def format_terminal(snapshot: SystemSnapshot) -> str:
        """Return a formatted terminal string for the snapshot."""
        lines: list[str] = []
        ts = snapshot.checked_at.strftime("%Y-%m-%d %H:%M:%S")
        overall_label = {
            "healthy": "OK  ",
            "degraded": "WARN",
            "unhealthy": "DOWN",
            "unknown": "????",
        }.get(snapshot.overall_status.value, "????")

        lines.append("  ╔═══════════════════════════════════════════════════════╗")
        lines.append("  ║   BARREN WUFFET -- Health Monitor                     ║")
        lines.append(f"  ║   {ts}   Overall: {overall_label}                   ║")
        lines.append("  ╚═══════════════════════════════════════════════════════╝")
        lines.append("")

        # API health table
        lines.append("  ┌── API Health ─────────────────────────────────────────┐")
        for name, h in snapshot.api_health.items():
            icon = {"healthy": "OK  ", "degraded": "WARN", "unhealthy": "DOWN"}.get(
                h.status.value, "????"
            )
            note = (h.details.get("note") or h.error or "")[:40]
            lines.append(f"  │  {icon}  {name:<18s}  {note:<40s}│")
        lines.append("  └────────────────────────────────────────────────────────┘")
        lines.append("")

        # System resources
        if snapshot.system_resources:
            det = snapshot.system_resources.details
            cpu = det.get("cpu_percent", 0.0)
            mem = det.get("memory_percent", 0.0)
            disk = det.get("disk_percent", 0.0)
            lines.append(f"  System:  CPU {cpu:.1f}%   Mem {mem:.1f}%   Disk {disk:.1f}%")
            lines.append("")

        # P&L / positions panel
        lines.append("  ┌── P&L & Positions ────────────────────────────────────┐")
        lines.append(f"  │  Open positions : {snapshot.positions_count:<3}                                 │")
        pnl_str = f"{snapshot.pnl_today:+.2f}" if snapshot.pnl_today is not None else "n/a"
        lines.append(f"  │  Today P&L      : {pnl_str:<10}                              │")
        sym = snapshot.last_trade_symbol or "none"
        at_ = snapshot.last_trade_at or "--"
        lines.append(f"  │  Last trade     : {sym:<8}  at {at_:<16}             │")
        lines.append("  └────────────────────────────────────────────────────────┘")
        lines.append("")

        # Active alerts
        if snapshot.active_alerts:
            lines.append(f"  Alerts ({len(snapshot.active_alerts)}):")
            for a in snapshot.active_alerts[-5:]:
                lines.append(f"    [{a['severity'].upper():<8}] {a['title']}  (x{a['count']})")
        else:
            lines.append("  No active alerts.")
        lines.append("")

        return "\n".join(lines)

    # ── run modes ────────────────────────────────────────────────────

    def run_once(self) -> int:
        """Collect one snapshot, print it, and return 0."""
        snap = self.collect_snapshot()
        print(self.format_terminal(snap))
        return 0

    def run_loop(self, interval: int = 30) -> int:
        """
        Collect snapshots in a loop, clearing the terminal before each render.
        Stops on Ctrl+C (KeyboardInterrupt).
        """
        _log.info("HealthMonitor loop started", interval_seconds=interval)
        try:
            while True:
                snap = self.collect_snapshot()
                print("\033[2J\033[H", end="", flush=True)
                print(self.format_terminal(snap), flush=True)
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        return 0


# ── module entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    loop = "--loop" in sys.argv
    monitor = HealthMonitor()
    if loop:
        sys.exit(monitor.run_loop(30))
    else:
        sys.exit(monitor.run_once())
