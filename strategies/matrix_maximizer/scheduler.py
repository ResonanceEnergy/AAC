"""
MATRIX MAXIMIZER — Scheduler
================================
Automated daily cycle scheduling:
  - Pre-market run (9:15 AM ET)
  - Market open (9:30 AM ET)
  - Mid-session check (12:30 PM ET)
  - Market close (4:00 PM ET)
  - After-hours report (4:15 PM ET)
  - Custom interval runs
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class ScheduleSlot(Enum):
    PRE_MARKET = "pre_market"       # 9:15 AM ET
    MARKET_OPEN = "market_open"     # 9:30 AM ET
    MID_SESSION = "mid_session"     # 12:30 PM ET
    MARKET_CLOSE = "market_close"   # 4:00 PM ET
    AFTER_HOURS = "after_hours"     # 4:15 PM ET
    CUSTOM = "custom"


# Default schedule times (hour, minute) in ET
DEFAULT_SCHEDULE = {
    ScheduleSlot.PRE_MARKET: (9, 15),
    ScheduleSlot.MARKET_OPEN: (9, 30),
    ScheduleSlot.MID_SESSION: (12, 30),
    ScheduleSlot.MARKET_CLOSE: (16, 0),
    ScheduleSlot.AFTER_HOURS: (16, 15),
}


@dataclass
class ScheduledTask:
    """A scheduled task."""
    slot: ScheduleSlot
    name: str
    callback: Callable
    args: tuple = ()
    kwargs: Optional[Dict[str, Any]] = None
    enabled: bool = True
    last_run: Optional[str] = None
    run_count: int = 0

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class MatrixScheduler:
    """Scheduler for automated MATRIX MAXIMIZER cycles.

    Usage:
        scheduler = MatrixScheduler()
        scheduler.add_task(ScheduleSlot.PRE_MARKET, "scan", runner.run_full_cycle, ...)
        scheduler.add_task(ScheduleSlot.AFTER_HOURS, "report", dashboard.generate_report)
        scheduler.start()  # Runs in background thread
        scheduler.stop()

    Manual overrides:
        scheduler.run_now("scan")           # Run a specific task immediately
        scheduler.run_slot(ScheduleSlot.PRE_MARKET)  # Run all tasks in a slot
    """

    def __init__(self, check_interval: int = 30) -> None:
        self._tasks: Dict[str, ScheduledTask] = {}
        self._check_interval = check_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_slot_run: Dict[ScheduleSlot, str] = {}  # Prevents double-runs

    def add_task(self, slot: ScheduleSlot, name: str,
                 callback: Callable, *args, **kwargs) -> None:
        """Register a task for a schedule slot."""
        self._tasks[name] = ScheduledTask(
            slot=slot,
            name=name,
            callback=callback,
            args=args,
            kwargs=kwargs,
        )
        logger.info("Registered task '%s' at %s", name, slot.value)

    def remove_task(self, name: str) -> None:
        """Remove a scheduled task."""
        self._tasks.pop(name, None)

    def enable_task(self, name: str) -> None:
        if name in self._tasks:
            self._tasks[name].enabled = True

    def disable_task(self, name: str) -> None:
        if name in self._tasks:
            self._tasks[name].enabled = False

    def start(self) -> None:
        """Start the scheduler in a background daemon thread."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="MM-Scheduler")
        self._thread.start()
        logger.info("Matrix Maximizer scheduler started (check every %ds)", self._check_interval)

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Matrix Maximizer scheduler stopped")

    def is_running(self) -> bool:
        return self._running

    def run_now(self, task_name: str) -> Any:
        """Run a specific task immediately, bypassing schedule."""
        task = self._tasks.get(task_name)
        if not task:
            logger.warning("Task '%s' not found", task_name)
            return None
        return self._execute_task(task)

    def run_slot(self, slot: ScheduleSlot) -> List[Any]:
        """Run all tasks in a schedule slot immediately."""
        results = []
        for task in self._tasks.values():
            if task.slot == slot and task.enabled:
                result = self._execute_task(task)
                results.append(result)
        return results

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        now_et = datetime.now(ET)
        next_slots = self._get_upcoming_slots(now_et)

        return {
            "running": self._running,
            "current_time_et": now_et.strftime("%H:%M:%S"),
            "is_market_hours": self._is_market_hours(now_et),
            "is_trading_day": self._is_trading_day(now_et),
            "tasks": {
                name: {
                    "slot": t.slot.value,
                    "enabled": t.enabled,
                    "last_run": t.last_run,
                    "run_count": t.run_count,
                }
                for name, t in self._tasks.items()
            },
            "next_slots": next_slots,
        }

    def print_schedule(self) -> str:
        """Human-readable schedule display."""
        status = self.get_status()
        lines = [
            f"  Scheduler: {'RUNNING' if status['running'] else 'STOPPED'}",
            f"  Time (ET): {status['current_time_et']}",
            f"  Market:    {'OPEN' if status['is_market_hours'] else 'CLOSED'}",
            f"  Trading:   {'YES' if status['is_trading_day'] else 'NO (weekend)'}",
            "",
            "  SCHEDULE:",
        ]

        for slot, (hour, minute) in sorted(DEFAULT_SCHEDULE.items(), key=lambda x: x[1]):
            slot_tasks = [t for t in self._tasks.values() if t.slot == slot and t.enabled]
            task_names = ", ".join(t.name for t in slot_tasks) or "(none)"
            lines.append(f"    {hour:02d}:{minute:02d} ET  {slot.value:<15s}  → {task_names}")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # INTERNAL
    # ═══════════════════════════════════════════════════════════════════════

    def _loop(self) -> None:
        """Main scheduler loop — runs in background thread."""
        while self._running:
            try:
                now_et = datetime.now(ET)

                if not self._is_trading_day(now_et):
                    time.sleep(self._check_interval)
                    continue

                # Check each slot
                for slot, (hour, minute) in DEFAULT_SCHEDULE.items():
                    slot_time = now_et.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    diff = (now_et - slot_time).total_seconds()

                    # Trigger if within the check window and not already run today
                    if 0 <= diff < self._check_interval:
                        today_key = f"{slot.value}_{now_et.date().isoformat()}"
                        if today_key not in self._last_slot_run:
                            self._last_slot_run[today_key] = now_et.isoformat()
                            self.run_slot(slot)

            except Exception as exc:
                logger.error("Scheduler loop error: %s", exc)

            time.sleep(self._check_interval)

    def _execute_task(self, task: ScheduledTask) -> Any:
        """Execute a single task with error handling."""
        logger.info("Executing task: %s", task.name)
        try:
            result = task.callback(*task.args, **task.kwargs)
            task.last_run = datetime.utcnow().isoformat()
            task.run_count += 1
            logger.info("Task '%s' completed (run #%d)", task.name, task.run_count)
            return result
        except Exception as exc:
            logger.error("Task '%s' failed: %s", task.name, exc)
            return None

    @staticmethod
    def _is_market_hours(now_et: datetime) -> bool:
        """Check if within NYSE market hours (9:30-16:00 ET)."""
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_et <= market_close

    @staticmethod
    def _is_trading_day(now_et: datetime) -> bool:
        """Check if today is a trading day (Mon-Fri, no major holidays)."""
        return now_et.weekday() < 5  # 0=Mon, 4=Fri

    @staticmethod
    def _get_upcoming_slots(now_et: datetime) -> List[Dict[str, str]]:
        """Get remaining schedule slots for today."""
        upcoming = []
        for slot, (hour, minute) in sorted(DEFAULT_SCHEDULE.items(), key=lambda x: x[1]):
            slot_time = now_et.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if slot_time > now_et:
                upcoming.append({
                    "slot": slot.value,
                    "time": f"{hour:02d}:{minute:02d} ET",
                    "in_minutes": int((slot_time - now_et).total_seconds() / 60),
                })
        return upcoming
