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

import json
import os
import signal
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import structlog

from agents.dfv.decision_engine import DFV
from agents.dfv.routines import brief, eod, midday, weekend_dd

_log = structlog.get_logger(__name__)
ET = ZoneInfo("America/New_York")

REPO_ROOT = Path(__file__).resolve().parents[2]
HEARTBEAT_PATH = REPO_ROOT / "agents" / "dfv" / "memory" / "daemon_heartbeat.json"
HEARTBEAT_STALE_SECONDS = 2 * 60 * 60  # 2h


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_heartbeat(
    *,
    last_routine: str | None = None,
    last_routine_ts: str | None = None,
    extra: dict | None = None,
) -> None:
    """Write a heartbeat file. Routines and the daemon both call this."""
    payload: dict = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ts_et": datetime.now(ET).isoformat(timespec="seconds"),
        "pid": os.getpid(),
    }
    if last_routine:
        payload["last_routine"] = last_routine
    if last_routine_ts:
        payload["last_routine_ts"] = last_routine_ts
    if extra:
        payload.update(extra)
    try:
        _atomic_write_json(HEARTBEAT_PATH, payload)
    except OSError as e:
        _log.warning("dfv.daemon.heartbeat_write_failed", error=str(e))


def heartbeat_status() -> dict:
    """Return {alive: bool, age_seconds: int|None, last_ts: str|None, ...}."""
    if not HEARTBEAT_PATH.exists():
        return {"alive": False, "age_seconds": None, "last_ts": None, "reason": "no heartbeat file"}
    try:
        data = json.loads(HEARTBEAT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"alive": False, "age_seconds": None, "last_ts": None, "reason": f"unreadable: {e}"}
    last_ts = data.get("ts_utc")
    age_seconds: int | None = None
    alive = False
    if last_ts:
        try:
            ts = datetime.fromisoformat(last_ts)
            age_seconds = int((datetime.now(timezone.utc) - ts).total_seconds())
            alive = age_seconds < HEARTBEAT_STALE_SECONDS
        except ValueError:
            pass
    return {
        "alive": alive,
        "age_seconds": age_seconds,
        "last_ts": last_ts,
        "pid": data.get("pid"),
        "last_routine": data.get("last_routine"),
        "last_routine_ts": data.get("last_routine_ts"),
    }

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
    last_routine: str | None = None
    last_routine_ts: str | None = None

    # Initial heartbeat so consumers see liveness immediately.
    write_heartbeat()

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
                    last_routine = name
                    last_routine_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
                except Exception as e:  # noqa: BLE001 — daemon must survive routine errors
                    _log.error("dfv.daemon.routine_error", routine=name, error=str(e))
                fired_this_minute.add(key)

        write_heartbeat(last_routine=last_routine, last_routine_ts=last_routine_ts)
        time.sleep(tick_seconds)

    _log.info("dfv.daemon.stopped")


if __name__ == "__main__":
    run_forever()
