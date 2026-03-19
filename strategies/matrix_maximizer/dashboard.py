"""
MATRIX MAXIMIZER — Dashboard & Reporting
============================================
P&L tracking, daily reports, and performance dashboards:
  - Real-time P&L snapshot
  - End-of-day report generation
  - Position heat map
  - Weekly/monthly summaries
  - Export as text, JSON, or Markdown
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_REPORT_DIR = Path("data/matrix_maximizer/reports")


@dataclass
class DailySnapshot:
    """Single day's performance snapshot."""
    date: str
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    open_positions: int
    trades_today: int
    vix: float
    oil: float
    regime: str
    circuit_breaker: str
    mandate: str
    top_picks: List[str] = field(default_factory=list)


class MatrixDashboard:
    """Performance dashboard and report generator.

    Usage:
        dash = MatrixDashboard()
        dash.record_snapshot(snapshot)
        report = dash.daily_report(cycle_output)
        dash.save_report(report, "daily")
        history = dash.get_history(days=30)
    """

    def __init__(self) -> None:
        _REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self._snapshots: List[DailySnapshot] = []
        self._load_history()

    def record_snapshot(self, snapshot: DailySnapshot) -> None:
        """Record a daily snapshot."""
        self._snapshots.append(snapshot)
        self._save_history()

    def daily_report(self, cycle_output: Optional[Dict[str, Any]] = None,
                     positions: Optional[List[Any]] = None,
                     account: Optional[Any] = None) -> str:
        """Generate a comprehensive daily report.

        Args:
            cycle_output: Output from runner.run_full_cycle()
            positions: List of TrackedPosition objects
            account: AccountSnapshot from ExecutionEngine

        Returns:
            Formatted report string
        """
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║           MATRIX MAXIMIZER — DAILY REPORT                   ║",
            f"║           {now:<49s}║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
        ]

        # Account Summary
        if account:
            lines.extend([
                "┌─ ACCOUNT ─────────────────────────────────────────────────┐",
                f"│  Mode:     {account.mode:<15s}                              │",
                f"│  Equity:   ${account.total_value:>10,.2f}                          │",
                f"│  Cash:     ${account.cash:>10,.2f}                          │",
                f"│  Exposure: ${account.put_exposure:>10,.2f}                          │",
                f"│  Unreal:   ${account.unrealized_pnl:>+10,.2f}                          │",
                f"│  Realized: ${account.realized_pnl:>+10,.2f}                          │",
                "└───────────────────────────────────────────────────────────┘",
                "",
            ])

        # Cycle Output
        if cycle_output:
            lines.extend([
                "┌─ CYCLE OUTPUT ─────────────────────────────────────────────┐",
            ])

            # Scenario weights
            weights = cycle_output.get("scenario_weights", {})
            if weights:
                lines.append(f"│  Scenarios: Base={weights.get('base', 0):.0%}  "
                             f"Bear={weights.get('bear', 0):.0%}  "
                             f"Bull={weights.get('bull', 0):.0%}")

            # Mandate
            mandate = cycle_output.get("mandate", {})
            if mandate:
                lines.append(f"│  Mandate:  {mandate.get('level', 'OBSERVE')} "
                             f"({mandate.get('conviction', 0):.0%} conviction)")

            # Circuit breaker
            cb = cycle_output.get("circuit_breaker", "")
            lines.append(f"│  Circuit:  {cb}")

            # Top picks
            picks = cycle_output.get("picks", [])
            if picks:
                lines.append("│  TOP PICKS:")
                for p in picks[:5]:
                    lines.append(
                        f"│    {p.get('ticker', ''):<6s} ${p.get('strike', 0):.0f}P  "
                        f"${p.get('premium', 0):.2f}  Δ={p.get('delta', 0):.2f}  "
                        f"Score={p.get('score', 0):.0f}"
                    )

            lines.extend([
                "└───────────────────────────────────────────────────────────┘",
                "",
            ])

        # Positions
        if positions:
            lines.extend([
                "┌─ POSITIONS ─────────────────────────────────────────────────┐",
                "│  Ticker  Strike  Expiry      Qty  Entry   Current  P&L     │",
                "│  " + "-" * 56 + "│",
            ])
            for p in positions:
                if hasattr(p, "is_open") and p.is_open:
                    lines.append(
                        f"│  {p.ticker:<7s} ${p.strike:<6.0f} {p.expiry}  "
                        f"{p.contracts:<4d} ${p.entry_premium:<6.2f} "
                        f"${p.current_premium:<6.2f} ${p.unrealized_pnl:>+7.0f}│"
                    )
            lines.extend([
                "└───────────────────────────────────────────────────────────┘",
                "",
            ])

        # Historical performance
        if self._snapshots:
            lines.extend(self._performance_section())

        return "\n".join(lines)

    def weekly_summary(self, days: int = 7) -> str:
        """Generate weekly performance summary."""
        recent = self._get_recent(days)
        if not recent:
            return "  No data for weekly summary"

        total_realized = sum(s.realized_pnl for s in recent)
        avg_equity = sum(s.equity for s in recent) / len(recent)
        trading_days = sum(1 for s in recent if s.trades_today > 0)
        total_trades = sum(s.trades_today for s in recent)

        lines = [
            f"  WEEKLY SUMMARY ({recent[0].date} → {recent[-1].date})",
            f"    Trading Days: {trading_days}/{len(recent)}",
            f"    Total Trades: {total_trades}",
            f"    Realized P&L: ${total_realized:+,.2f}",
            f"    Avg Equity: ${avg_equity:,.2f}",
        ]

        # Regime breakdown
        regimes = {}
        for s in recent:
            regimes[s.regime] = regimes.get(s.regime, 0) + 1
        lines.append(f"    Regimes: {', '.join(f'{r}({c}d)' for r, c in regimes.items())}")

        return "\n".join(lines)

    def position_heatmap(self, positions: List[Any]) -> str:
        """Text-based position heat map showing P&L intensity."""
        if not positions:
            return "  No positions for heatmap"

        lines = ["  POSITION HEATMAP:"]
        for p in positions:
            if not hasattr(p, "is_open") or not p.is_open:
                continue

            pnl_pct = p.pnl_pct if hasattr(p, "pnl_pct") else 0
            # Visual bar: green for profit, red for loss
            bar_len = min(20, abs(int(pnl_pct * 100)))
            if pnl_pct >= 0:
                bar = "█" * bar_len
                color_label = "+"
            else:
                bar = "░" * bar_len
                color_label = "-"

            lines.append(
                f"    {p.ticker:<6s} ${p.strike:<6.0f} "
                f"[{color_label}{bar:<20s}] {pnl_pct:+.0%} (${p.unrealized_pnl:+.0f})"
            )

        return "\n".join(lines)

    def save_report(self, report: str, report_type: str = "daily") -> Path:
        """Save report to file."""
        date_str = datetime.utcnow().strftime("%Y%m%d_%H%M")
        filename = f"mm_{report_type}_{date_str}.txt"
        path = _REPORT_DIR / filename
        path.write_text(report, encoding="utf-8")
        logger.info("Report saved: %s", path)
        return path

    def save_json(self, data: Dict[str, Any], report_type: str = "cycle") -> Path:
        """Save structured data as JSON."""
        date_str = datetime.utcnow().strftime("%Y%m%d_%H%M")
        filename = f"mm_{report_type}_{date_str}.json"
        path = _REPORT_DIR / filename
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def get_history(self, days: int = 30) -> List[DailySnapshot]:
        """Get recent snapshot history."""
        return self._get_recent(days)

    # ═══════════════════════════════════════════════════════════════════════
    # INTERNAL
    # ═══════════════════════════════════════════════════════════════════════

    def _performance_section(self) -> List[str]:
        """Generate performance stats from history."""
        if len(self._snapshots) < 2:
            return []

        recent = self._snapshots[-30:]
        equities = [s.equity for s in recent]
        returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                returns.append((equities[i] - equities[i - 1]) / equities[i - 1])

        peak = max(equities)
        trough = min(equities)
        max_dd = trough - peak if peak > 0 else 0

        lines = [
            "┌─ PERFORMANCE (last 30 snapshots) ──────────────────────────┐",
            f"│  Start: ${equities[0]:,.2f} → Current: ${equities[-1]:,.2f}",
            f"│  Peak: ${peak:,.2f} | Trough: ${trough:,.2f} | Max DD: ${max_dd:+,.2f}",
        ]

        if returns:
            avg_ret = sum(returns) / len(returns)
            lines.append(f"│  Avg Daily Return: {avg_ret:.2%}")

        lines.append("└───────────────────────────────────────────────────────────┘")
        return lines

    def _get_recent(self, days: int) -> List[DailySnapshot]:
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [s for s in self._snapshots if s.date >= cutoff]

    def _save_history(self) -> None:
        path = _REPORT_DIR / "snapshot_history.json"
        data = []
        for s in self._snapshots[-365:]:  # Keep 1 year
            data.append({
                "date": s.date,
                "equity": s.equity,
                "unrealized_pnl": s.unrealized_pnl,
                "realized_pnl": s.realized_pnl,
                "open_positions": s.open_positions,
                "trades_today": s.trades_today,
                "vix": s.vix,
                "oil": s.oil,
                "regime": s.regime,
                "circuit_breaker": s.circuit_breaker,
                "mandate": s.mandate,
                "top_picks": s.top_picks,
            })
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_history(self) -> None:
        path = _REPORT_DIR / "snapshot_history.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for d in data:
                self._snapshots.append(DailySnapshot(**d))
            logger.info("Loaded %d snapshots from history", len(self._snapshots))
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.warning("Failed to load snapshot history: %s", exc)
