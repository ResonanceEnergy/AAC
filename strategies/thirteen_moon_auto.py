"""
Thirteen Moon Auto Engine
=========================

Plugs into the autonomous engine's ScheduledTask framework to provide
push-based event alerting from the thirteen-moon doctrine timeline.

Architecture:
    ┌───────────────────────────────────────────────────┐
    │          THIRTEEN MOON AUTO ENGINE                 │
    │                                                   │
    │  SCHEDULED TASKS                                  │
    │    tm_event_scan     300s  14-day horizon scan    │
    │    tm_daily_brief  86400s  morning moon briefing  │
    │    tm_critical_watch  60s  today's CRITICAL only  │
    │                                                   │
    │  ALERT DEDUPLICATION                              │
    │    Tracks fired alerts by (date, name) hash       │
    │    Resets daily for recurring lead-time actions    │
    │                                                   │
    │  PERSISTENCE                                      │
    │    data/war_engine/moon_alerts.jsonl               │
    │    data/war_engine/moon_daily_briefs.jsonl         │
    └───────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("ThirteenMoonAuto")


# ════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ════════════════════════════════════════════════════════════════════════

@dataclass
class MoonAutoParams:
    """Parameters governing the moon auto-alert loop."""

    # ── Scan Intervals ────────────────────────────────────────────────
    event_scan_interval: float = 300.0          # 5 min — full horizon scan
    daily_brief_interval: float = 86400.0       # 24 hr — morning briefing
    critical_watch_interval: float = 60.0       # 1 min — today's CRITICAL events only

    # ── Horizon Settings ──────────────────────────────────────────────
    scan_horizon_days: int = 14                 # How far ahead to scan
    critical_horizon_days: int = 1              # Only today for critical watch

    # ── Alert Thresholds ──────────────────────────────────────────────
    alert_priorities: List[str] = field(
        default_factory=lambda: ["CRITICAL", "HIGH"]
    )  # Which priorities trigger notifications
    brief_priorities: List[str] = field(
        default_factory=lambda: ["CRITICAL", "HIGH", "MEDIUM"]
    )  # Which priorities appear in daily brief

    # ── Dedup Settings ────────────────────────────────────────────────
    dedup_reset_hour: int = 6                   # Reset fired alerts at 6 AM local
    max_alerts_per_scan: int = 10               # Cap alerts per scan cycle


# ════════════════════════════════════════════════════════════════════════
# THIRTEEN MOON AUTO ENGINE
# ════════════════════════════════════════════════════════════════════════

class ThirteenMoonAutoEngine:
    """
    Push-based event alerting from the thirteen-moon doctrine timeline.
    Can run standalone or be registered into the autonomous engine.
    """

    STATE_DIR = PROJECT_ROOT / "data" / "war_engine"

    def __init__(
        self,
        params: Optional[MoonAutoParams] = None,
    ) -> None:
        self.params = params or MoonAutoParams()

        # Lazy-init doctrine to avoid import cost at registration
        self._doctrine: Any = None
        self._doctrine_loaded = False

        # Alert dedup state
        self._fired_alerts: Set[str] = set()
        self._last_dedup_reset: Optional[date] = None

        # Stats
        self._total_alerts_sent = 0
        self._last_scan_time: Optional[str] = None
        self._last_brief_time: Optional[str] = None

        # Ensure state directory
        self.STATE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Doctrine Access ───────────────────────────────────────────────

    def _get_doctrine(self) -> Any:
        """Lazy-load the ThirteenMoonDoctrine to avoid circular imports."""
        if not self._doctrine_loaded:
            try:
                from strategies.thirteen_moon_doctrine import ThirteenMoonDoctrine
                self._doctrine = ThirteenMoonDoctrine()
                self._doctrine_loaded = True
                logger.info("ThirteenMoonDoctrine loaded: %d cycles",
                            len(self._doctrine.moon_cycles))
            except Exception as e:
                logger.warning("Failed to load ThirteenMoonDoctrine: %s", e)
                self._doctrine_loaded = True  # Don't retry every cycle
        return self._doctrine

    # ── Dedup Management ──────────────────────────────────────────────

    def _alert_key(self, alert_date: date, alert_name: str) -> str:
        """Generate a dedup key for an alert."""
        return f"{alert_date.isoformat()}|{alert_name}"

    def _maybe_reset_dedup(self) -> None:
        """Reset fired alerts once per day at the configured hour."""
        today = date.today()
        if self._last_dedup_reset != today:
            now = datetime.now()
            if now.hour >= self.params.dedup_reset_hour:
                self._fired_alerts.clear()
                self._last_dedup_reset = today
                logger.info("Moon alert dedup reset for %s", today.isoformat())

    def _is_new_alert(self, alert_date: date, alert_name: str) -> bool:
        """Check if this alert has already been fired today."""
        return self._alert_key(alert_date, alert_name) not in self._fired_alerts

    def _mark_fired(self, alert_date: date, alert_name: str) -> None:
        """Mark an alert as fired."""
        self._fired_alerts.add(self._alert_key(alert_date, alert_name))

    # ── Persistence ───────────────────────────────────────────────────

    def _append_jsonl(self, filename: str, record: dict) -> None:
        """Append a JSON record to a JSONL file."""
        path = self.STATE_DIR / filename
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ══════════════════════════════════════════════════════════════════
    # SCHEDULED TASKS
    # ══════════════════════════════════════════════════════════════════

    async def task_event_scan(self) -> str:
        """
        Scan the 14-day horizon for upcoming events, fire alerts for
        CRITICAL/HIGH items that haven't been notified yet.
        """
        self._maybe_reset_dedup()
        doctrine = self._get_doctrine()
        if doctrine is None:
            return "Moon doctrine not available"

        now_iso = datetime.now(timezone.utc).isoformat()
        self._last_scan_time = now_iso

        alerts = doctrine.get_events_with_lead_time(
            days_ahead=self.params.scan_horizon_days,
        )

        # Filter to notifiable priorities and new alerts only
        actionable = []
        for alert in alerts:
            if alert.priority not in self.params.alert_priorities:
                continue
            if not self._is_new_alert(alert.event_date, alert.event_name):
                continue
            actionable.append(alert)

        if not actionable:
            logger.debug("Moon scan: no new actionable alerts")
            return "No new moon alerts"

        # Cap and process
        actionable = actionable[: self.params.max_alerts_per_scan]
        lines = []
        for alert in actionable:
            self._mark_fired(alert.event_date, alert.event_name)
            self._total_alerts_sent += 1

            line = (
                f"[{alert.priority}] {alert.event_name} "
                f"({alert.event_type}) in {alert.days_until}d "
                f"-> {alert.lead_time_action}"
            )
            lines.append(line)

            # Persist
            self._append_jsonl("moon_alerts.jsonl", {
                "ts": now_iso,
                "date": alert.event_date.isoformat(),
                "name": alert.event_name,
                "type": alert.event_type,
                "moon": alert.moon_number,
                "days_until": alert.days_until,
                "priority": alert.priority,
                "action": alert.lead_time_action,
            })

        summary = (
            f"MOON ALERT SCAN — {len(actionable)} actionable events:\n"
            + "\n".join(lines)
        )
        logger.info(summary)
        return summary

    async def task_daily_brief(self) -> str:
        """
        Generate a morning moon briefing: current phase, today's events,
        and the next 7 days outlook.
        """
        doctrine = self._get_doctrine()
        if doctrine is None:
            return "Moon doctrine not available"

        now_iso = datetime.now(timezone.utc).isoformat()
        self._last_brief_time = now_iso
        today = date.today()

        # Current moon phase
        current_moon = doctrine.get_current_moon(today)
        moon_info = "Unknown"
        if current_moon:
            days_in = (today - current_moon.start_date).days
            days_left = (current_moon.end_date - today).days
            moon_info = (
                f"Moon {current_moon.moon_number}: {current_moon.lunar_phase_name} "
                f"(Day {days_in}/{(current_moon.end_date - current_moon.start_date).days})"
            )
            if current_moon.fire_peak_date:
                fire_delta = (current_moon.fire_peak_date - today).days
                if fire_delta >= 0:
                    moon_info += f" | Fire Peak in {fire_delta}d"
                else:
                    moon_info += f" | Fire Peak was {abs(fire_delta)}d ago"

        # Today's events
        today_alerts = doctrine.get_events_with_lead_time(days_ahead=0, target=today)
        today_lines = []
        for a in today_alerts:
            if a.priority in self.params.brief_priorities:
                today_lines.append(
                    f"  [{a.priority}] {a.event_name} ({a.event_type}) -> {a.lead_time_action}"
                )

        # Next 7 days
        week_alerts = doctrine.get_events_with_lead_time(days_ahead=7, target=today)
        # Exclude today's events from the week list
        future_alerts = [a for a in week_alerts if a.days_until > 0]
        week_lines = []
        for a in future_alerts:
            if a.priority in self.params.brief_priorities:
                week_lines.append(
                    f"  [{a.priority}] {a.event_name} in {a.days_until}d ({a.event_type})"
                )

        # Build brief
        parts = [
            "=" * 50,
            "  THIRTEEN MOON DAILY BRIEF",
            f"  {today.strftime('%A, %B %d, %Y')}",
            "=" * 50,
            f"  Phase: {moon_info}",
        ]

        if current_moon and current_moon.doctrine_action:
            parts.append(f"  Mandate: {current_moon.doctrine_action.mandate}")

        parts.append("")

        if today_lines:
            parts.append(f"  TODAY ({len(today_lines)} events):")
            parts.extend(today_lines)
        else:
            parts.append("  TODAY: No scheduled events")

        parts.append("")

        if week_lines:
            parts.append(f"  NEXT 7 DAYS ({len(week_lines)} events):")
            parts.extend(week_lines)
        else:
            parts.append("  NEXT 7 DAYS: Clear horizon")

        parts.append("=" * 50)
        brief = "\n".join(parts)

        # Persist
        self._append_jsonl("moon_daily_briefs.jsonl", {
            "ts": now_iso,
            "date": today.isoformat(),
            "moon": current_moon.moon_number if current_moon else -1,
            "phase": current_moon.lunar_phase_name if current_moon else "unknown",
            "today_count": len(today_lines),
            "week_count": len(week_lines),
        })

        logger.info(brief)
        return brief

    async def task_critical_watch(self) -> str:
        """
        Fast-cycle check for today's CRITICAL events only.
        Fires on every scan (no dedup) to keep critical events visible.
        """
        doctrine = self._get_doctrine()
        if doctrine is None:
            return "Moon doctrine not available"

        today_alerts = doctrine.get_events_with_lead_time(
            days_ahead=self.params.critical_horizon_days,
        )
        critical = [a for a in today_alerts if a.priority == "CRITICAL"]

        if not critical:
            return "No critical moon events today"

        lines = []
        for a in critical:
            tag = "TODAY" if a.days_until == 0 else f"in {a.days_until}d"
            lines.append(f"  !!! {a.event_name} ({a.event_type}) {tag} -> {a.lead_time_action}")

        summary = f"CRITICAL MOON WATCH — {len(critical)} events:\n" + "\n".join(lines)
        return summary

    # ══════════════════════════════════════════════════════════════════
    # TASK REGISTRY — For autonomous engine registration
    # ══════════════════════════════════════════════════════════════════

    def get_task_registry(self) -> List[Dict[str, Any]]:
        """
        Return task definitions for autonomous engine registration.
        Each entry: {"name": str, "interval": float, "callback": coroutine, "critical": bool}
        """
        p = self.params
        return [
            {
                "name": "tm_event_scan",
                "interval": p.event_scan_interval,
                "callback": self.task_event_scan,
                "critical": False,
            },
            {
                "name": "tm_daily_brief",
                "interval": p.daily_brief_interval,
                "callback": self.task_daily_brief,
                "critical": False,
            },
            {
                "name": "tm_critical_watch",
                "interval": p.critical_watch_interval,
                "callback": self.task_critical_watch,
                "critical": True,
            },
        ]

    # ══════════════════════════════════════════════════════════════════
    # STATUS / DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict[str, Any]:
        """Return current status for dashboards."""
        doctrine = self._get_doctrine()
        current_moon = doctrine.get_current_moon() if doctrine else None
        return {
            "doctrine_loaded": self._doctrine is not None,
            "current_moon": current_moon.lunar_phase_name if current_moon else "unknown",
            "moon_number": current_moon.moon_number if current_moon else -1,
            "total_alerts_sent": self._total_alerts_sent,
            "fired_alerts_today": len(self._fired_alerts),
            "last_scan": self._last_scan_time,
            "last_brief": self._last_brief_time,
            "params": {
                "scan_interval": f"{self.params.event_scan_interval}s",
                "brief_interval": f"{self.params.daily_brief_interval}s",
                "critical_interval": f"{self.params.critical_watch_interval}s",
                "horizon_days": self.params.scan_horizon_days,
                "alert_priorities": self.params.alert_priorities,
            },
        }

    def render_status(self) -> str:
        """Render a human-readable status report."""
        s = self.get_status()
        return "\n".join([
            "=" * 50,
            "  THIRTEEN MOON AUTO ENGINE — STATUS",
            "=" * 50,
            f"  Moon:         {s['current_moon']} (#{s['moon_number']})",
            f"  Doctrine:     {'loaded' if s['doctrine_loaded'] else 'NOT LOADED'}",
            f"  Alerts sent:  {s['total_alerts_sent']} total, {s['fired_alerts_today']} today",
            f"  Last scan:    {s['last_scan'] or 'never'}",
            f"  Last brief:   {s['last_brief'] or 'never'}",
            "",
            "  INTERVALS:",
            f"    event_scan:     {s['params']['scan_interval']}",
            f"    daily_brief:    {s['params']['brief_interval']}",
            f"    critical_watch: {s['params']['critical_interval']}",
            f"    horizon:        {s['params']['horizon_days']}d",
            "=" * 50,
        ])


# ════════════════════════════════════════════════════════════════════════
# CLI — Standalone test / manual run
# ════════════════════════════════════════════════════════════════════════

async def _run_one_shot():
    """Run a single scan + brief for testing."""
    import asyncio

    engine = ThirteenMoonAutoEngine()
    print(engine.render_status())
    print()

    brief = await engine.task_daily_brief()
    print(brief)
    print()

    scan = await engine.task_event_scan()
    print(scan)
    print()

    watch = await engine.task_critical_watch()
    print(watch)


if __name__ == "__main__":
    import asyncio
    asyncio.run(_run_one_shot())
