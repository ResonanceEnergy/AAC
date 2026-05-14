"""core/eod_reporter.py — Sprint 13: End-of-Day Summary Report.

Generates a structured daily brief after each end-of-day P&L snapshot.

The report aggregates four data sources that already exist:
  1. PnLTracker.today_report()  — P&L and trades
  2. PositionSnapshot list      — live positions from IBKR
  3. RollManager.urgent_only()  — positions needing attention
  4. SystemSnapshot (optional)  — API health from HealthMonitor

Output:
  * ``EodReport`` dataclass — structured, machine-readable
  * ``EodReport.format_text()`` — human-readable multi-section brief
  * ``EodReport.write_to_file(path)`` — writes to ``reports/daily_brief.txt``

Design:
  * Never raises — every block is wrapped in try/except; missing data
    is rendered as "N/A" rather than crashing the report.
  * All external imports are lazy (inside functions) with ``# noqa: PLC0415``
    so the module loads instantly.
  * ``EodReporter.generate()`` is the only public entry point.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

# ── Default report path ───────────────────────────────────────────────────────

_DEFAULT_REPORT_PATH = Path(__file__).parent.parent / "reports" / "daily_brief.txt"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PositionSummary:
    """Condensed view of a single position for EOD reporting."""

    symbol: str
    quantity: float
    market_value: float
    unrealized_pnl: float
    roll_urgent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "roll_urgent": self.roll_urgent,
        }


@dataclass
class EodReport:
    """Structured end-of-day report.

    Attributes:
        report_date:        ISO date string (YYYY-MM-DD).
        account_value_usd:  Reported account value.
        total_unrealized_pnl: Sum of unrealised P&L across all positions.
        total_realized_pnl:   Sum of realised P&L for today's trades.
        pnl_delta:          Change vs previous day (0.0 if unavailable).
        position_count:     Number of open positions.
        positions:          List of PositionSummary objects.
        roll_urgent_count:  Number of positions flagged for roll/close.
        trades_today:       Number of trades executed today.
        api_health_summary: Dict of api_name → "OK" | "DEGRADED" | "DOWN".
        overall_api_status: Worst status across all APIs.
        active_alerts:      List of active alert dicts from AlertManager.
        written_at:         UTC timestamp when the report was written.
    """

    report_date: str
    account_value_usd: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    pnl_delta: float = 0.0
    position_count: int = 0
    positions: list[PositionSummary] = field(default_factory=list)
    roll_urgent_count: int = 0
    trades_today: int = 0
    api_health_summary: dict[str, str] = field(default_factory=dict)
    overall_api_status: str = "UNKNOWN"
    active_alerts: list[dict] = field(default_factory=list)
    written_at: str = ""
    drawdown_pct: float = 0.0
    drawdown_tripped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_date": self.report_date,
            "account_value_usd": self.account_value_usd,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_realized_pnl": self.total_realized_pnl,
            "pnl_delta": self.pnl_delta,
            "position_count": self.position_count,
            "positions": [p.to_dict() for p in self.positions],
            "roll_urgent_count": self.roll_urgent_count,
            "trades_today": self.trades_today,
            "api_health_summary": self.api_health_summary,
            "overall_api_status": self.overall_api_status,
            "active_alerts": self.active_alerts,
            "written_at": self.written_at,
            "drawdown_pct": self.drawdown_pct,
            "drawdown_tripped": self.drawdown_tripped,
        }

    def format_text(self) -> str:
        """Render the report as a human-readable multi-section text brief."""
        lines: list[str] = []

        def _hr(label: str = "") -> str:
            bar = "─" * 56
            return f"\n  {bar}\n  {label}\n  {bar}" if label else f"\n  {'─' * 56}"

        # ── Header ──────────────────────────────────────────────────────────
        lines.append(_hr(f"AAC End-of-Day Brief — {self.report_date}"))
        lines.append(f"  Generated : {self.written_at or 'N/A'}")
        lines.append(f"  API Status: {self.overall_api_status}")

        # ── P&L Summary ──────────────────────────────────────────────────────
        lines.append(_hr("P&L Summary"))
        lines.append(f"  Account Value  : ${self.account_value_usd:>14,.2f}")
        drawdown_warn = "  !! DRAWDOWN CIRCUIT BREAKER TRIPPED" if self.drawdown_tripped else ""
        lines.append(f"  Drawdown       : {self.drawdown_pct:.2%}{drawdown_warn}")
        lines.append(f"  Unrealised P&L : ${self.total_unrealized_pnl:>14,.2f}")
        lines.append(f"  Realised P&L   : ${self.total_realized_pnl:>14,.2f}")
        total = self.total_unrealized_pnl + self.total_realized_pnl
        lines.append(f"  Total P&L      : ${total:>14,.2f}")
        delta_prefix = "+" if self.pnl_delta >= 0 else ""
        lines.append(f"  P&L Delta (1d) : {delta_prefix}${self.pnl_delta:,.2f}")
        lines.append(f"  Open Positions : {self.position_count}")
        lines.append(f"  Trades Today   : {self.trades_today}")

        # ── Roll Urgency ─────────────────────────────────────────────────────
        if self.roll_urgent_count > 0:
            urgent_symbols = [p.symbol for p in self.positions if p.roll_urgent]
            lines.append(_hr(f"Roll Urgency — {self.roll_urgent_count} position(s) need attention"))
            for sym in urgent_symbols:
                lines.append(f"  !! {sym}")
        else:
            lines.append(_hr("Roll Urgency"))
            lines.append("  No positions require rolling today.")

        # ── Positions ────────────────────────────────────────────────────────
        if self.positions:
            lines.append(_hr(f"Positions ({self.position_count})"))
            lines.append(
                f"  {'Symbol':<8} {'Qty':>7} {'Mkt Val':>13} {'Unr P&L':>11} {'Roll?':>6}"
            )
            lines.append(
                f"  {'─'*8} {'─'*7} {'─'*13} {'─'*11} {'─'*6}"
            )
            for p in self.positions:
                roll_flag = " !! " if p.roll_urgent else "    "
                lines.append(
                    f"  {p.symbol:<8} {p.quantity:>7.1f} "
                    f"${p.market_value:>11,.2f} "
                    f"${p.unrealized_pnl:>9,.2f}"
                    f"{roll_flag}"
                )
        else:
            lines.append(_hr("Positions"))
            lines.append("  No open positions.")

        # ── API Health ───────────────────────────────────────────────────────
        if self.api_health_summary:
            lines.append(_hr("API Health"))
            for api, status in self.api_health_summary.items():
                icon = "OK " if status == "OK" else "!!!"
                lines.append(f"  [{icon}] {api}: {status}")

        # ── Alerts ───────────────────────────────────────────────────────────
        if self.active_alerts:
            lines.append(_hr(f"Active Alerts ({len(self.active_alerts)})"))
            for alert in self.active_alerts[-5:]:
                msg = alert.get("message", str(alert))
                sev = alert.get("severity", "?")
                lines.append(f"  [{sev}] {msg}")

        lines.append(_hr())
        return "\n".join(lines)

    def write_to_file(self, path: str | Path | None = None) -> str:
        """Write the formatted report to disk.

        Args:
            path: Override path.  Defaults to ``reports/daily_brief.txt``.

        Returns:
            Absolute path of the written file as a string.
            Empty string if write fails.
        """
        target = Path(path) if path else _DEFAULT_REPORT_PATH
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            content = self.format_text()
            target.write_text(content, encoding="utf-8")
            _log.info("eod_report_written", path=str(target))
            return str(target)
        except Exception as exc:
            _log.error("eod_report_write_failed", path=str(target), error=str(exc))
            return ""


# ── EodReporter ───────────────────────────────────────────────────────────────

class EodReporter:
    """Generates the end-of-day brief from available data sources.

    Usage::

        reporter = EodReporter(alerter=alerter)
        report   = reporter.generate(
            pnl_report   = pnl_tracker.today_report(),
            positions    = position_snapshots,
            health_snap  = health_monitor.collect_snapshot(),   # optional
            pnl_delta    = pnl_tracker.pnl_delta(),             # optional
        )
        report.write_to_file()

    All arguments are optional — passing ``None`` for any of them is safe.
    The reporter fails-open: a missing health snapshot yields "UNKNOWN" API
    status, not a crash.

    Parameters
    ----------
    alerter:
        Optional :class:`shared.alerter.Alerter` instance.  When set, a
        condensed daily brief is sent to Telegram after ``write_to_file()``.
        Defaults to ``None`` (no push notification).
    """

    def __init__(self, alerter: object | None = None) -> None:
        self._alerter = alerter

    def generate(
        self,
        pnl_report: dict | None = None,
        positions: list | None = None,
        health_snap: Any | None = None,
        pnl_delta: float = 0.0,
        report_path: str | Path | None = None,
    ) -> EodReport:
        """Generate, write, and return the EodReport.

        Args:
            pnl_report:  Output of ``PnLTracker.today_report()``.
            positions:   List of ``PositionSnapshot`` objects.
            health_snap: ``SystemSnapshot`` from ``HealthMonitor`` (optional).
            pnl_delta:   Float P&L change vs previous day (default 0.0).
            report_path: Override write path (default ``reports/daily_brief.txt``).

        Returns:
            ``EodReport`` — always, even if all inputs are None.
        """
        today = datetime.now(tz=timezone.utc).date().isoformat()
        now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # ── P&L data ─────────────────────────────────────────────────────────
        acct_val = 0.0
        unrealized = 0.0
        realized = 0.0
        pos_count = 0
        trades_count = 0

        try:
            if pnl_report:
                daily = pnl_report.get("daily_pnl") or {}
                if daily:
                    acct_val = float(daily.get("account_value_usd") or 0)
                    unrealized = float(daily.get("total_unrealized_pnl") or 0)
                    realized = float(daily.get("total_realized_pnl") or 0)
                    pos_count = int(daily.get("position_count") or 0)
                trades_count = len(pnl_report.get("today_trades") or [])
        except Exception as exc:
            _log.warning("eod_reporter_pnl_parse_failed", error=str(exc))

        # ── Position summaries + roll urgency ─────────────────────────────────
        position_summaries: list[PositionSummary] = []
        roll_urgent_count = 0

        try:
            urgent_symbols = _get_urgent_symbols(positions or [])
            for pos in (positions or []):
                sym = getattr(pos, "symbol", "?")
                roll_flag = sym in urgent_symbols
                if roll_flag:
                    roll_urgent_count += 1
                position_summaries.append(
                    PositionSummary(
                        symbol=sym,
                        quantity=float(getattr(pos, "quantity", 0)),
                        market_value=float(getattr(pos, "market_value", 0)),
                        unrealized_pnl=float(getattr(pos, "unrealized_pnl", 0)),
                        roll_urgent=roll_flag,
                    )
                )
        except Exception as exc:
            _log.warning("eod_reporter_position_parse_failed", error=str(exc))

        # ── API health ────────────────────────────────────────────────────────
        api_health_summary: dict[str, str] = {}
        overall_api_status = "UNKNOWN"
        active_alerts: list[dict] = []

        try:
            if health_snap is not None:
                api_health_summary, overall_api_status = _extract_api_health(health_snap)
                active_alerts = list(getattr(health_snap, "active_alerts", []))
        except Exception as exc:
            _log.warning("eod_reporter_health_parse_failed", error=str(exc))

        # ── Drawdown state ────────────────────────────────────────────────────
        drawdown_pct = 0.0
        drawdown_tripped = False
        try:
            from strategies.drawdown_circuit_breaker import DrawdownCircuitBreaker  # noqa: PLC0415

            dd_state = DrawdownCircuitBreaker().current_state()
            drawdown_pct = dd_state.drawdown_pct
            drawdown_tripped = dd_state.tripped
        except Exception as exc:
            _log.warning("eod_reporter_drawdown_state_failed", error=str(exc))

        report = EodReport(
            report_date=today,
            account_value_usd=acct_val,
            total_unrealized_pnl=unrealized,
            total_realized_pnl=realized,
            pnl_delta=pnl_delta,
            position_count=pos_count,
            positions=position_summaries,
            roll_urgent_count=roll_urgent_count,
            trades_today=trades_count,
            api_health_summary=api_health_summary,
            overall_api_status=overall_api_status,
            active_alerts=active_alerts,
            written_at=now_str,
            drawdown_pct=drawdown_pct,
            drawdown_tripped=drawdown_tripped,
        )

        report.write_to_file(report_path)

        # ── Sprint 21: Telegram daily brief ──────────────────────────────────
        if self._alerter is not None:
            try:
                brief = (
                    f"Date: {report.report_date}\n"
                    f"Account: ${report.account_value_usd:,.0f}\n"
                    f"Unrealized P&L: ${report.total_unrealized_pnl:+,.0f}\n"
                    f"Realized P&L: ${report.total_realized_pnl:+,.0f}\n"
                    f"Positions: {report.position_count} | Trades today: {report.trades_today}\n"
                    f"Roll urgent: {report.roll_urgent_count}\n"
                    f"API status: {report.overall_api_status}\n"
                    + (f"⚠️ DRAWDOWN {report.drawdown_pct:.1%}" if report.drawdown_tripped else "")
                ).strip()
                self._alerter.send("EOD_BRIEF", brief)
            except Exception as exc:
                _log.warning("eod_reporter_alert_failed", error=str(exc))

        return report


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_urgent_symbols(positions: list) -> set[str]:
    """Return set of symbols flagged by RollManager.urgent_only()."""
    try:
        from strategies.roll_manager import RollManager  # noqa: PLC0415

        decisions = RollManager().urgent_only(positions)
        return {d.symbol for d in decisions}
    except Exception as exc:
        _log.debug("eod_reporter_roll_manager_unavailable", error=str(exc))
        return set()


def _extract_api_health(health_snap: Any) -> tuple[dict[str, str], str]:
    """Extract API health dict and worst-status string from a SystemSnapshot."""
    api_map: dict[str, str] = {}
    worst = "UNKNOWN"

    try:
        raw = getattr(health_snap, "api_health", {})
        for name, comp in raw.items():
            status_val = getattr(getattr(comp, "status", None), "value", str(comp))
            api_map[name] = status_val

        overall = getattr(health_snap, "overall_status", None)
        if overall is not None:
            worst = getattr(overall, "value", str(overall))
    except Exception as exc:
        _log.debug("eod_reporter_health_extract_failed", error=str(exc))

    return api_map, worst
