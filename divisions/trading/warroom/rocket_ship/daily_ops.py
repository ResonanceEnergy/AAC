#!/usr/bin/env python3
"""
Rocket Ship — Daily Operations Scheduler
==========================================
Runs the full Rocket Ship briefing on a daily schedule and persists
every result to data/rocket_ship_daily_log.jsonl.

Schedule (local time):
  • 07:00 — morning_briefing   (all 5 dashboards)
  • 20:00 — evening_review     (lunar + trigger only)

Usage:
    python -m strategies.rocket_ship.daily_ops           # daemon mode (blocks)
    python -m strategies.rocket_ship.daily_ops --once    # single full briefing + exit
    python -m strategies.rocket_ship.daily_ops --evening # evening review + exit
    python -m strategies.rocket_ship.daily_ops --status  # show last log entry + exit
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# ── stdout/stderr guard (pythonw.exe / Windows Task Scheduler) ─────────────
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")  # noqa: WPS515
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")  # noqa: WPS515
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # …/AAC_fresh/
LOG_FILE = PROJECT_ROOT / "data" / "rocket_ship_daily_log.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [rocket_ship.daily_ops] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Schedule constants (local 24-hour clock) ───────────────────────────────
MORNING_BRIEFING_HOUR: int = 7
MORNING_BRIEFING_MIN:  int = 0
EVENING_REVIEW_HOUR:   int = 20
EVENING_REVIEW_MIN:    int = 0


# ════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════

def _make_args(json_output: bool = False) -> SimpleNamespace:
    """Build a dummy argparse.Namespace that satisfies the runner's cmd_* functions."""
    return SimpleNamespace(
        indicators=False,
        lunar=False,
        trigger=False,
        allocation=False,
        geo=None,
        geo_base=None,
        all=True,
        phase=None,
        capital=None,
        json=json_output,
    )


def _log_result(event: str, result: dict[str, Any]) -> None:
    """Append a structured entry to the daily JSONL log file."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts":    datetime.now(timezone.utc).isoformat(),
        "date":  date.today().isoformat(),
        "event": event,
        "data":  result,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ════════════════════════════════════════════════════════════════════════
# TASK RUNNERS  (pure functions — easy to call from the engine or CLI)
# ════════════════════════════════════════════════════════════════════════

def run_morning_briefing() -> dict[str, Any]:
    """
    Full Rocket Ship morning briefing.

    Runs all 5 dashboards (indicators, lunar, trigger, allocation, geo),
    prints them to stdout, and persists the structured result to JSONL.
    Returns the result dict for programmatic use.
    """
    logger.info("=== MORNING BRIEFING START ===")
    try:
        from strategies.rocket_ship.runner import cmd_full_briefing
        args   = _make_args(json_output=False)
        result = cmd_full_briefing(args)
        _log_result("morning_briefing", result)

        phase = result.get("trigger", {}).get("phase", "?")
        green = result.get("indicators", {}).get("green_count", "?")
        prob  = result.get("trigger", {}).get("ignition_prob", 0)
        moon  = result.get("lunar", {}).get("moon_number", "?")
        days  = result.get("lunar", {}).get("days_to_rocket_start", "?")
        logger.info(
            "Briefing complete — Phase=%s | Green=%s/15 | Prob=%.0f%% | Moon#%s | T-%sd",
            phase, green, prob * 100, moon, days,
        )
        return result
    except Exception as exc:
        logger.error("Morning briefing failed: %s", exc)
        error_result: dict[str, Any] = {"error": str(exc)}
        _log_result("morning_briefing_error", error_result)
        return error_result


def run_evening_review() -> dict[str, Any]:
    """
    Lightweight evening review — lunar position + ignition trigger only.

    Faster than the full briefing; no allocation or geo dashboards.
    Persists result to JSONL.
    """
    logger.info("=== EVENING REVIEW START ===")
    try:
        from strategies.rocket_ship.runner import cmd_lunar, cmd_trigger
        args      = _make_args(json_output=False)
        lunar_r   = cmd_lunar(args)
        trigger_r = cmd_trigger(args)
        result: dict[str, Any] = {"lunar": lunar_r, "trigger": trigger_r}
        _log_result("evening_review", result)

        phase = trigger_r.get("phase", "?")
        prob  = trigger_r.get("ignition_prob", 0)
        moon  = lunar_r.get("moon_number", "?")
        logger.info(
            "Evening review — Phase=%s | Prob=%.0f%% | Moon#%s",
            phase, prob * 100, moon,
        )
        return result
    except Exception as exc:
        logger.error("Evening review failed: %s", exc)
        error_result = {"error": str(exc)}
        _log_result("evening_review_error", error_result)
        return error_result


def show_last_status() -> None:
    """Print a human-readable summary of the most recent log entry."""
    if not LOG_FILE.exists():
        print(f"No log file yet at {LOG_FILE}")
        return

    last_line = ""
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line.strip()

    if not last_line:
        print("Log file is empty.")
        return

    entry = json.loads(last_line)
    print(f"\nLast event:  {entry['event']}")
    print(f"Timestamp:   {entry['ts']}")
    data = entry.get("data", {})

    trigger = data.get("trigger", {})
    if trigger:
        print(f"Phase:       {trigger.get('phase', '?')}")
        print(f"Ignition %:  {trigger.get('ignition_prob', 0):.0%}")
        print(f"Alert:       {trigger.get('alert_level', '?')}")

    indicators = data.get("indicators", {})
    if indicators:
        print(f"Green count: {indicators.get('green_count', '?')}/15")

    lunar = data.get("lunar", {})
    if lunar:
        print(f"Moon #:      {lunar.get('moon_number', '?')}")
        print(f"Days left:   {lunar.get('days_to_rocket_start', '?')}")

    geo = data.get("geo", {})
    if geo:
        done  = geo.get("complete_tasks", "?")
        total = geo.get("total_tasks", "?")
        print(f"Geo tasks:   {done}/{total} complete")


# ════════════════════════════════════════════════════════════════════════
# BACKGROUND SCHEDULER
# ════════════════════════════════════════════════════════════════════════

class RocketDailyOps:
    """
    Thread-based daily ops scheduler.

    Fires two tasks per calendar day at fixed local times:
      • morning_briefing at MORNING_BRIEFING_HOUR:MORNING_BRIEFING_MIN
      • evening_review   at EVENING_REVIEW_HOUR:EVENING_REVIEW_MIN

    Each task runs in its own short-lived daemon thread so the scheduler
    loop is never blocked by a slow briefing.

    Usage::

        ops = RocketDailyOps()
        ops.start()
        # … keep main thread alive …
        ops.stop()
    """

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Launch the background polling thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="rocket_daily_ops",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "RocketDailyOps started — morning=%02d:%02d  evening=%02d:%02d (local time)",
            MORNING_BRIEFING_HOUR, MORNING_BRIEFING_MIN,
            EVENING_REVIEW_HOUR,   EVENING_REVIEW_MIN,
        )

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("RocketDailyOps stopped")

    def _run_loop(self) -> None:
        """Poll local time every 30 seconds; fire tasks when hour:minute matches."""
        _fired_today: set[str] = set()

        while not self._stop_event.is_set():
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")

            # Clear the fired-today tracker at midnight
            if now.hour == 0 and now.minute < 1:
                _fired_today.clear()

            morning_key = f"morning_{today}"
            evening_key = f"evening_{today}"

            if (
                now.hour   == MORNING_BRIEFING_HOUR
                and now.minute == MORNING_BRIEFING_MIN
                and morning_key not in _fired_today
            ):
                _fired_today.add(morning_key)
                threading.Thread(
                    target=run_morning_briefing,
                    name="rocket_morning_briefing",
                    daemon=True,
                ).start()

            if (
                now.hour   == EVENING_REVIEW_HOUR
                and now.minute == EVENING_REVIEW_MIN
                and evening_key not in _fired_today
            ):
                _fired_today.add(evening_key)
                threading.Thread(
                    target=run_evening_review,
                    name="rocket_evening_review",
                    daemon=True,
                ).start()

            self._stop_event.wait(timeout=30.0)


# ════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m strategies.rocket_ship.daily_ops",
        description="Rocket Ship Daily Ops — scheduled briefing daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Daemon mode (default) blocks until Ctrl-C and sends briefings\n"
            "every day at 07:00 (morning) and 20:00 (evening) local time.\n"
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run full morning briefing once and exit",
    )
    parser.add_argument(
        "--evening",
        action="store_true",
        help="Run evening review once and exit",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print last logged briefing and exit",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args   = parser.parse_args()

    if args.status:
        show_last_status()
        return

    if args.once:
        run_morning_briefing()
        return

    if args.evening:
        run_evening_review()
        return

    # ── Daemon mode — run until Ctrl-C ─────────────────────────────────
    ops = RocketDailyOps()
    ops.start()
    logger.info("Rocket Ship Daily Ops daemon running — Ctrl-C to stop")
    logger.info("  Morning briefing : %02d:%02d local", MORNING_BRIEFING_HOUR, MORNING_BRIEFING_MIN)
    logger.info("  Evening review   : %02d:%02d local", EVENING_REVIEW_HOUR,   EVENING_REVIEW_MIN)
    logger.info("  Log file         : %s", LOG_FILE)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        ops.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
