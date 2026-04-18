"""AAC Daily Tasks — fluid daily task aggregator for dashboards.

Pulls from 13-Moon doctrine events, war room regime, options lifecycle,
autonomous engine tasks, and portfolio positions. Organizes into
time-slotted daily tasks with priorities and categories.

Used by mission_control.py /api/daily_tasks endpoint.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import structlog

_log = structlog.get_logger()


# ── Time Slots ──────────────────────────────────────────────────────────────

TIME_SLOTS = {
    "pre_market":   {"label": "Pre-Market",   "start": "06:00", "end": "09:30", "emoji": "🌅"},
    "market_open":  {"label": "Market Open",  "start": "09:30", "end": "10:30", "emoji": "🔔"},
    "mid_day":      {"label": "Mid-Day",      "start": "10:30", "end": "14:00", "emoji": "☀️"},
    "power_hour":   {"label": "Power Hour",   "start": "14:00", "end": "16:00", "emoji": "⚡"},
    "after_hours":  {"label": "After Hours",  "start": "16:00", "end": "20:00", "emoji": "🌙"},
    "overnight":    {"label": "Overnight",    "start": "20:00", "end": "06:00", "emoji": "🌌"},
}

# Map event categories → suggested time slots
CATEGORY_SLOT_MAP: dict[str, str] = {
    # Pre-market
    "earnings": "pre_market",
    "macro_data": "pre_market",
    "fed": "pre_market",
    "economic": "pre_market",
    "daily_brief": "pre_market",
    "regime": "pre_market",
    "review": "pre_market",
    "ops": "pre_market",
    # Market open
    "execution": "market_open",
    "roll_window": "market_open",
    "capital": "market_open",
    "expiry": "market_open",
    "inflection": "market_open",
    # Mid-day
    "risk_management": "mid_day",
    "dte_check": "mid_day",
    "milestone": "mid_day",
    "dispersion": "mid_day",
    "event_driven": "mid_day",
    # Power hour
    "calendar": "power_hour",
    "macro_event": "power_hour",
    "assignment_risk": "power_hour",
    # After hours
    "inflation_rotation": "after_hours",
    "financial": "after_hours",
    "geopolitical": "after_hours",
    "scenario": "after_hours",
    # Overnight
    "system": "overnight",
    "policy_change": "overnight",
}


@dataclass
class DailyTask:
    """A single daily task with time, priority, and source."""
    name: str
    description: str
    priority: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    time_slot: str  # key from TIME_SLOTS
    source: str  # moon, war_room, options, portfolio, strategy, engine
    assets: list[str] = field(default_factory=list)
    due_date: str = ""
    days_until: int = 0
    action_required: str = ""
    conviction: float = 0.0
    completed: bool = False
    task_id: str = ""


# ── Aggregator ──────────────────────────────────────────────────────────────

class DailyTaskAggregator:
    """Pulls from all AAC subsystems and builds a prioritized daily task list."""

    def __init__(self, target_date: datetime.date | None = None, horizon_days: int = 3):
        self.target_date = target_date or datetime.date.today()
        self.horizon_days = horizon_days
        self._tasks: list[DailyTask] = []
        self._doctrine = None
        self._completions = self._load_completions()

    # ── Public API ──────────────────────────────────────────────────────

    def collect_all(self) -> dict[str, Any]:
        """Collect daily tasks from all sources. Returns dashboard-ready dict."""
        self._tasks = []

        self._collect_moon_events()
        self._collect_options_lifecycle()
        self._collect_war_room_events()
        self._collect_strategy_tasks()
        self._collect_engine_daily_tasks()
        self._collect_portfolio_checks()

        # Deduplicate by task_id
        seen: set[str] = set()
        unique: list[DailyTask] = []
        for t in self._tasks:
            if t.task_id not in seen:
                seen.add(t.task_id)
                # Apply saved completions
                if t.task_id in self._completions:
                    t.completed = True
                unique.append(t)
        self._tasks = unique

        return self._format_output()

    def mark_complete(self, task_id: str) -> bool:
        """Mark a task as completed — persists to file."""
        today_str = self.target_date.isoformat()
        comp_path = self._completions_path()
        entry = {"task_id": task_id, "date": today_str, "completed_at": datetime.datetime.now().isoformat()}

        try:
            comp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(comp_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            self._completions.add(task_id)
            return True
        except OSError:
            _log.warning("daily_tasks_mark_fail", task_id=task_id)
            return False

    # ── Moon Events ─────────────────────────────────────────────────────

    def _collect_moon_events(self) -> None:
        """Pull scheduled alerts from 13-Moon doctrine."""
        try:
            from strategies.thirteen_moon_doctrine import ThirteenMoonDoctrine
            doctrine = ThirteenMoonDoctrine()
            self._doctrine = doctrine

            alerts = doctrine.get_events_with_lead_time(days_ahead=self.horizon_days, target=self.target_date)

            for alert in alerts:
                slot = CATEGORY_SLOT_MAP.get(alert.event_type, "pre_market")

                # Try to get richer data from the underlying moon cycle
                assets: list[str] = []
                category = alert.event_type
                conviction = 0.0
                desc = alert.lead_time_action or alert.event_name

                moon = doctrine.get_current_moon(target=alert.event_date)
                if moon:
                    # Search AAC events for matching event
                    for aac_evt in (moon.aac_events or []):
                        if aac_evt.name == alert.event_name or (
                            hasattr(aac_evt, "date") and str(aac_evt.date) == str(alert.event_date)
                            and aac_evt.name in alert.event_name
                        ):
                            assets = list(aac_evt.assets) if hasattr(aac_evt, "assets") else []
                            category = aac_evt.category if hasattr(aac_evt, "category") else category
                            conviction = aac_evt.conviction if hasattr(aac_evt, "conviction") else 0.0
                            desc = aac_evt.description if hasattr(aac_evt, "description") else desc
                            slot = CATEGORY_SLOT_MAP.get(category, slot)
                            break

                task_id = f"moon_{alert.event_date}_{alert.event_name[:30].replace(' ', '_')}"

                self._tasks.append(DailyTask(
                    name=alert.event_name,
                    description=desc,
                    priority=alert.priority,
                    category=category,
                    time_slot=slot,
                    source="moon",
                    assets=assets,
                    due_date=str(alert.event_date),
                    days_until=alert.days_until,
                    action_required=alert.lead_time_action,
                    conviction=conviction,
                    task_id=task_id,
                ))
        except Exception as e:
            _log.warning("daily_tasks_moon_error", error=str(e))

    # ── Options Lifecycle ───────────────────────────────────────────────

    def _collect_options_lifecycle(self) -> None:
        """Generate DTE-based tasks from live portfolio positions."""
        try:
            positions = self._get_option_positions()

            for pos in positions:
                expiry_str = pos.get("expiry", "")
                if not expiry_str:
                    continue

                try:
                    expiry_date = datetime.date.fromisoformat(expiry_str.replace("/", "-")[:10])
                except ValueError:
                    continue

                dte = (expiry_date - self.target_date).days
                if dte < 0:
                    continue

                symbol = pos.get("symbol", "?")
                strike = pos.get("strike", "?")
                ptype = pos.get("type", "put")
                qty = pos.get("qty", 0)
                account = pos.get("account", "")
                label = f"{symbol} ${strike}{ptype[0].upper()} {expiry_str}"

                # Generate DTE-based tasks
                if dte == 0:
                    self._tasks.append(DailyTask(
                        name=f"EXPIRY DAY: {label}",
                        description=f"Expires TODAY. Close by 3:30 PM ET to avoid assignment risk. Qty: {qty}",
                        priority="CRITICAL",
                        category="expiry",
                        time_slot="market_open",
                        source="options",
                        assets=[symbol],
                        due_date=expiry_str,
                        days_until=0,
                        action_required="CLOSE or let expire. Check assignment risk.",
                        task_id=f"opt_expiry_{symbol}_{expiry_str}",
                    ))
                elif dte == 1:
                    self._tasks.append(DailyTask(
                        name=f"EXPIRY EVE: {label}",
                        description=f"Expires TOMORROW. Close all by 3:30 PM ET. Qty: {qty} ({account})",
                        priority="CRITICAL",
                        category="assignment_risk",
                        time_slot="power_hour",
                        source="options",
                        assets=[symbol],
                        due_date=expiry_str,
                        days_until=1,
                        action_required="Close & roll. Do NOT hold to expiry.",
                        task_id=f"opt_eve_{symbol}_{expiry_str}",
                    ))
                elif dte <= 7:
                    self._tasks.append(DailyTask(
                        name=f"FINAL WEEK ({dte} DTE): {label}",
                        description=f"Gamma spike zone. Roll to next month or close. Qty: {qty} ({account})",
                        priority="CRITICAL" if dte <= 3 else "HIGH",
                        category="risk_management",
                        time_slot="mid_day",
                        source="options",
                        assets=[symbol],
                        due_date=expiry_str,
                        days_until=dte,
                        action_required=f"Roll or close within {dte} days.",
                        task_id=f"opt_final_{symbol}_{expiry_str}_{dte}dte",
                    ))
                elif dte <= 21:
                    self._tasks.append(DailyTask(
                        name=f"ROLL WINDOW ({dte} DTE): {label}",
                        description=f"Approaching 21-DTE roll trigger. Evaluate theta decay. Qty: {qty}",
                        priority="HIGH" if dte <= 14 else "MEDIUM",
                        category="roll_window",
                        time_slot="market_open",
                        source="options",
                        assets=[symbol],
                        due_date=expiry_str,
                        days_until=dte,
                        action_required="Evaluate roll at 21 DTE per ROLL_DISCIPLINE.",
                        task_id=f"opt_roll_{symbol}_{expiry_str}",
                    ))
                elif dte <= 45 and dte % 30 == 0:
                    self._tasks.append(DailyTask(
                        name=f"DTE CHECK ({dte}): {label}",
                        description=f"Periodic check. Monitor theta and credit conditions. Qty: {qty}",
                        priority="MEDIUM",
                        category="dte_check",
                        time_slot="mid_day",
                        source="options",
                        assets=[symbol],
                        due_date=expiry_str,
                        days_until=dte,
                        action_required="Monitor. No action unless thesis changed.",
                        task_id=f"opt_check_{symbol}_{expiry_str}_{dte}dte",
                    ))
        except Exception as e:
            _log.warning("daily_tasks_options_error", error=str(e))

    # ── War Room Events ─────────────────────────────────────────────────

    def _collect_war_room_events(self) -> None:
        """Pull war room regime-driven tasks."""
        try:
            from strategies.war_room_engine import WarRoomEngine
            engine = WarRoomEngine()

            # Composite score → regime task
            try:
                status = engine.get_status()
                composite = status.get("composite_score", 0)
                regime = status.get("regime", "UNKNOWN")

                if composite >= 70:
                    self._tasks.append(DailyTask(
                        name=f"CRISIS REGIME: Score {composite:.0f}/100",
                        description=f"Regime: {regime}. Heightened vigilance. Review all positions and hedges.",
                        priority="CRITICAL",
                        category="regime",
                        time_slot="pre_market",
                        source="war_room",
                        action_required="Review positions against crisis playbook.",
                        task_id=f"wr_regime_{self.target_date}",
                    ))
                elif composite >= 50:
                    self._tasks.append(DailyTask(
                        name=f"ELEVATED WATCH: Score {composite:.0f}/100",
                        description=f"Regime: {regime}. Monitor closely. Tighten stops.",
                        priority="HIGH",
                        category="regime",
                        time_slot="pre_market",
                        source="war_room",
                        action_required="Tighten stops. Review hedges.",
                        task_id=f"wr_regime_{self.target_date}",
                    ))
            except Exception:
                pass

            # Daily mandate
            try:
                mandate = engine.get_mandate()
                if mandate:
                    mandate_text = str(mandate) if not isinstance(mandate, dict) else mandate.get("text", str(mandate))
                    self._tasks.append(DailyTask(
                        name="Daily Mandate",
                        description=mandate_text[:200],
                        priority="HIGH",
                        category="review",
                        time_slot="pre_market",
                        source="war_room",
                        action_required="Execute daily mandate.",
                        task_id=f"wr_mandate_{self.target_date}",
                    ))
            except Exception:
                pass

        except ImportError:
            _log.debug("daily_tasks_war_room_unavailable")
        except Exception as e:
            _log.warning("daily_tasks_war_room_error", error=str(e))

    # ── Strategy Tasks ──────────────────────────────────────────────────

    def _collect_strategy_tasks(self) -> None:
        """Pull strategy-layer tasks from moon events."""
        # Strategy events already get pulled via moon events with layer="strategy"
        # This method adds standing daily strategy tasks

        today_weekday = self.target_date.weekday()  # 0=Mon, 4=Fri

        # End-of-week review on Fridays
        if today_weekday == 4:
            self._tasks.append(DailyTask(
                name="Weekly Strategy Review",
                description="End-of-week review: P&L, position sizing, thesis validation, next week setup.",
                priority="HIGH",
                category="review",
                time_slot="after_hours",
                source="strategy",
                action_required="Review all positions. Update thesis probabilities.",
                task_id=f"strat_weekly_{self.target_date}",
            ))

        # Monday morning briefing
        if today_weekday == 0:
            self._tasks.append(DailyTask(
                name="Monday Morning Briefing",
                description="Week start: Check overnight moves, Asia/Europe session, weekend events.",
                priority="HIGH",
                category="daily_brief",
                time_slot="pre_market",
                source="strategy",
                action_required="Review weekend news. Set weekly priorities.",
                task_id=f"strat_monday_{self.target_date}",
            ))

        # First of month
        if self.target_date.day == 1:
            self._tasks.append(DailyTask(
                name="Monthly Doctrine Review",
                description="Recalibrate all indicators. Update composite score. Review P&L. Adjust thesis probabilities.",
                priority="HIGH",
                category="review",
                time_slot="after_hours",
                source="strategy",
                action_required="Full monthly review cycle.",
                task_id=f"strat_monthly_{self.target_date}",
            ))

    # ── Engine Daily Tasks ──────────────────────────────────────────────

    def _collect_engine_daily_tasks(self) -> None:
        """Pull standing daily tasks from the autonomous engine schedule."""
        daily_engine_tasks = [
            ("Pre-Market Data Refresh", "All live feeds, API data, composite score update.", "pre_market", "HIGH"),
            ("Position Reconciliation", "Cross-check IBKR/Moomoo positions against internal state.", "market_open", "HIGH"),
            ("Moon Phase Check", "Current moon phase alignment, active doctrine events.", "pre_market", "MEDIUM"),
            ("EOD Snapshot", "End-of-day: Save indicator state, composite, regime to JSONL.", "after_hours", "MEDIUM"),
        ]

        for name, desc, slot, priority in daily_engine_tasks:
            self._tasks.append(DailyTask(
                name=name,
                description=desc,
                priority=priority,
                category="ops",
                time_slot=slot,
                source="engine",
                action_required="Automated — verify execution.",
                task_id=f"eng_{name.replace(' ', '_').lower()}_{self.target_date}",
            ))

    # ── Portfolio Checks ────────────────────────────────────────────────

    def _collect_portfolio_checks(self) -> None:
        """Portfolio-level tasks based on account state."""
        try:
            bal_path = _ROOT / "data" / "account_balances.json"
            if not bal_path.exists():
                return

            raw = json.loads(bal_path.read_text(encoding="utf-8"))
            accounts = raw.get("accounts", {})

            for key, acct in accounts.items():
                positions = acct.get("positions") or []
                if not positions:
                    continue

                # Count options vs equity
                option_count = sum(1 for p in positions if p.get("right") in ("P", "C"))
                if option_count > 0:
                    self._tasks.append(DailyTask(
                        name=f"{key.upper()}: {option_count} Options Monitor",
                        description=f"Monitor {option_count} option positions. Check P&L, DTE, Greeks.",
                        priority="MEDIUM",
                        category="risk_management",
                        time_slot="mid_day",
                        source="portfolio",
                        action_required="Review Greeks and P&L.",
                        task_id=f"port_monitor_{key}_{self.target_date}",
                    ))
        except Exception as e:
            _log.warning("daily_tasks_portfolio_error", error=str(e))

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_option_positions(self) -> list[dict[str, Any]]:
        """Extract option positions from account_balances.json."""
        bal_path = _ROOT / "data" / "account_balances.json"
        if not bal_path.exists():
            return []

        raw = json.loads(bal_path.read_text(encoding="utf-8"))
        accounts = raw.get("accounts", {})
        positions: list[dict[str, Any]] = []

        for key, acct in accounts.items():
            for pos in acct.get("positions") or []:
                right = pos.get("right", "")
                if right in ("P", "C"):
                    positions.append({
                        "symbol": pos.get("symbol", "?"),
                        "type": "put" if right == "P" else "call",
                        "strike": pos.get("strike"),
                        "expiry": pos.get("expiry", ""),
                        "qty": pos.get("qty", 0),
                        "account": key,
                        "market_price": pos.get("marketPrice", 0),
                        "unrealized_pnl": pos.get("unrealizedPNL", pos.get("pl_val", 0)),
                    })

        return positions

    def _completions_path(self) -> Path:
        return _ROOT / "data" / "war_engine" / "daily_task_completions.jsonl"

    def _load_completions(self) -> set[str]:
        """Load today's completed task IDs."""
        comp_path = self._completions_path()
        today_str = self.target_date.isoformat()
        completed: set[str] = set()

        if not comp_path.exists():
            return completed

        try:
            for line in comp_path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("date") == today_str:
                    completed.add(entry.get("task_id", ""))
        except (json.JSONDecodeError, OSError):
            pass

        return completed

    def _format_output(self) -> dict[str, Any]:
        """Format tasks into dashboard-ready grouped structure."""
        # Sort: CRITICAL first, then by time slot order
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        slot_order = list(TIME_SLOTS.keys())

        self._tasks.sort(key=lambda t: (
            slot_order.index(t.time_slot) if t.time_slot in slot_order else 99,
            priority_order.get(t.priority, 9),
            t.days_until,
        ))

        # Group by time slot
        by_slot: dict[str, list[dict[str, Any]]] = {}
        for slot_key in TIME_SLOTS:
            slot_tasks = [asdict(t) for t in self._tasks if t.time_slot == slot_key]
            if slot_tasks:
                by_slot[slot_key] = slot_tasks

        # Summary counts
        total = len(self._tasks)
        completed = sum(1 for t in self._tasks if t.completed)
        by_priority: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for t in self._tasks:
            by_priority[t.priority] = by_priority.get(t.priority, 0) + 1
            by_source[t.source] = by_source.get(t.source, 0) + 1

        # Today vs upcoming
        today_tasks = [asdict(t) for t in self._tasks if t.days_until == 0 or t.due_date == str(self.target_date) or t.due_date == ""]
        upcoming_tasks = [asdict(t) for t in self._tasks if t.days_until > 0 and t.due_date and t.due_date != str(self.target_date)]

        return {
            "date": str(self.target_date),
            "day_name": self.target_date.strftime("%A"),
            "horizon_days": self.horizon_days,
            "total_tasks": total,
            "completed": completed,
            "remaining": total - completed,
            "by_priority": by_priority,
            "by_source": by_source,
            "by_slot": by_slot,
            "slots": {k: v for k, v in TIME_SLOTS.items()},
            "today_tasks": today_tasks,
            "upcoming_tasks": upcoming_tasks,
            "all_tasks": [asdict(t) for t in self._tasks],
            "ts": datetime.datetime.now().isoformat(),
        }


# ── Standalone CLI ──────────────────────────────────────────────────────────

def _run_cli():
    """CLI test: python monitoring/daily_tasks.py"""
    import argparse

    parser = argparse.ArgumentParser(description="AAC Daily Tasks")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=3, help="Days ahead to scan")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    target = datetime.date.fromisoformat(args.date) if args.date else None
    agg = DailyTaskAggregator(target_date=target, horizon_days=args.horizon)
    result = agg.collect_all()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    # Pretty print
    print(f"\n{'='*60}")
    print(f"  AAC DAILY TASKS — {result['day_name']} {result['date']}")
    print(f"  Total: {result['total_tasks']} | Completed: {result['completed']} | Remaining: {result['remaining']}")
    print(f"  Priority: {result['by_priority']}")
    print(f"  Sources: {result['by_source']}")
    print(f"{'='*60}")

    for slot_key, slot_meta in TIME_SLOTS.items():
        tasks_in_slot = result["by_slot"].get(slot_key, [])
        if not tasks_in_slot:
            continue

        print(f"\n{slot_meta['emoji']}  {slot_meta['label']} ({slot_meta['start']}–{slot_meta['end']} ET)")
        print(f"{'─'*55}")

        for t in tasks_in_slot:
            check = "✅" if t["completed"] else "⬜"
            pri_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(t["priority"], "⚪")
            src_icon = {"moon": "🌙", "options": "📊", "war_room": "🏛️", "strategy": "📋", "engine": "⚙️", "portfolio": "💼"}.get(t["source"], "•")
            days_label = f" [{t['days_until']}d]" if t["days_until"] > 0 else ""
            assets_label = f" [{', '.join(t['assets'][:4])}]" if t["assets"] else ""
            print(f"  {check} {pri_icon} {src_icon} {t['name']}{days_label}{assets_label}")
            if t["action_required"]:
                print(f"       → {t['action_required']}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    _run_cli()
