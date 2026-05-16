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
from agents.dfv.routines import (
    asia_digest,
    asia_watch,
    brief,
    close_debrief,
    eod,
    midday,
    open_bell_prep,
    retail_pulse,
    weekend_dd,
)

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
    "asia_digest":    asia_digest,
    "pre_market":     brief,
    "open_bell_prep": open_bell_prep,
    "midday":         midday,
    "retail_pulse":   retail_pulse,
    "eod_prep":       eod,
    "close_debrief":  close_debrief,
    "asia_watch":     asia_watch,
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
    orphan_cfg = dfv.doctrine.get("orphan_guard") or {}
    reconcile_cfg = dfv.doctrine.get("reconciler") or {}
    orphan_interval = int(orphan_cfg.get("scan_interval_seconds", 300))
    reconcile_interval = int(reconcile_cfg.get("scan_interval_seconds", 300))
    _log.info("dfv.daemon.start", cadence=cadence,
              orphan_interval=orphan_interval, reconcile_interval=reconcile_interval)

    signal.signal(signal.SIGINT, _stop)
    try:
        signal.signal(signal.SIGTERM, _stop)
    except (AttributeError, ValueError):
        pass  # Windows: SIGTERM may not be available in all contexts

    fired_this_minute: set[tuple[str, str]] = set()
    last_minute = ""
    last_routine: str | None = None
    last_routine_ts: str | None = None
    last_orphan_scan = 0.0
    last_reconcile_scan = 0.0

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

        # Orphan-position guard (separate cadence from briefs)
        now_mono = time.monotonic()
        if orphan_cfg.get("enabled", True) and (now_mono - last_orphan_scan) >= orphan_interval:
            try:
                from agents.dfv import orphan_guard  # noqa: PLC0415
                result = orphan_guard.scan_and_stub(dfv=dfv)
                if result.get("written"):
                    _log.warning("dfv.daemon.orphan_scan",
                                 written=result["written"], orphans=result["orphans"])
            except Exception as e:  # noqa: BLE001
                _log.error("dfv.daemon.orphan_scan_failed", error=str(e))
            last_orphan_scan = now_mono

        # Position reconciler
        if reconcile_cfg.get("enabled", True) and (now_mono - last_reconcile_scan) >= reconcile_interval:
            try:
                from agents.dfv import reconciler  # noqa: PLC0415
                snap = reconciler.reconcile(dfv=dfv)
                if snap.get("mismatch_count"):
                    _log.warning("dfv.daemon.reconcile",
                                 mismatches=snap["mismatch_count"])
            except Exception as e:  # noqa: BLE001
                _log.error("dfv.daemon.reconcile_failed", error=str(e))
            last_reconcile_scan = now_mono

        write_heartbeat(last_routine=last_routine, last_routine_ts=last_routine_ts)
        time.sleep(tick_seconds)

    _log.info("dfv.daemon.stopped")


if __name__ == "__main__":
    run_forever()
