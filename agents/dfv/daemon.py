from __future__ import annotations

"""DFV daemon — 24/7 cadence runner.

Lightweight scheduler that calls routines at the times defined in
config/doctrine/dfv_doctrine.yaml::cadence.  No external deps; sleeps in
60-second slices and fires when wall-clock matches.

Run:
    python -m agents.dfv.daemon
or:
    python launch.py dfv
"""

import signal
import time
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

import structlog

from agents.dfv.decision_engine import DFV
from agents.dfv.routines import brief, eod, midday, weekend_dd

_log = structlog.get_logger(__name__)
ET = ZoneInfo("America/New_York")

# routine_key -> (callable, default HH:MM ET)
ROUTINES: dict[str, Callable[[], dict]] = {
    "asia_digest":    brief,
    "pre_market":     brief,
    "open_bell_prep": brief,
    "midday":         midday,
    "eod_prep":       eod,
    "close_debrief":  eod,
    "asia_watch":     midday,
    "weekend_dd":     weekend_dd,
}

_RUN_FLAG = True


def _stop(_signum, _frame) -> None:  # pragma: no cover — signal handler
    global _RUN_FLAG
    _RUN_FLAG = False
    _log.info("dfv.daemon.stop_signal")


def _parse_time(spec: str) -> tuple[str, str]:
    """Return (weekday_filter, 'HH:MM').  Weekday filter is '' (any) or 'Mon'..'Sun'."""
    spec = spec.strip()
    if " " in spec:
        day, hhmm = spec.split(" ", 1)
        return day, hhmm.strip()
    return "", spec


def _is_due(spec: str, now: datetime) -> bool:
    day, hhmm = _parse_time(spec)
    if day:
        if now.strftime("%a") != day:
            return False
    try:
        h, m = (int(x) for x in hhmm.split(":"))
    except ValueError:
        return False
    return now.hour == h and now.minute == m


def run_forever(tick_seconds: int = 60) -> None:
    dfv = DFV()
    cadence = dfv.doctrine.get("cadence", {})
    _log.info("dfv.daemon.start", cadence=cadence)

    signal.signal(signal.SIGINT, _stop)
    try:
        signal.signal(signal.SIGTERM, _stop)
    except (AttributeError, ValueError):
        pass  # Windows: SIGTERM may not be available in all contexts

    fired_this_minute: set[tuple[str, str]] = set()
    last_minute = ""

    while _RUN_FLAG:
        now = datetime.now(ET)
        cur_minute = now.strftime("%Y-%m-%d %H:%M")
        if cur_minute != last_minute:
            fired_this_minute.clear()
            last_minute = cur_minute

        for name, spec in cadence.items():
            key = (name, cur_minute)
            if key in fired_this_minute:
                continue
            if _is_due(spec, now):
                fn = ROUTINES.get(name)
                if not fn:
                    continue
                try:
                    out = fn()
                    _log.info("dfv.daemon.fired", routine=name, headline=out.get("headline", ""))
                except Exception as e:  # noqa: BLE001 — daemon must survive routine errors
                    _log.error("dfv.daemon.routine_error", routine=name, error=str(e))
                fired_this_minute.add(key)

        time.sleep(tick_seconds)

    _log.info("dfv.daemon.stopped")


if __name__ == "__main__":
    run_forever()
