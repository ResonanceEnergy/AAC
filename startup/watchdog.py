"""
startup.watchdog — Process supervision and health polling.

Manages child processes with auto-restart, health endpoint polling,
PID tracking, and graceful shutdown.

Usage:
    dog = ProcessWatchdog()
    dog.register("orchestrator", [sys.executable, "-m", "core.orchestrator"])
    dog.register("health", [sys.executable, "health_server.py"])
    dog.run_forever()  # blocks until SIGINT/SIGTERM
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Managed Process ─────────────────────────────────────────────────────────


@dataclass
class ManagedProcess:
    """A supervised child process."""

    name: str
    cmd: list[str]
    cwd: str = str(PROJECT_ROOT)
    env: dict[str, str] | None = None
    max_restarts: int = 3
    backoff_base: float = 5.0
    backoff_max: float = 120.0
    health_url: str | None = None
    health_timeout: float = 5.0
    critical: bool = False  # If True, watchdog exits when max restarts exceeded

    # Runtime state (not user-settable)
    proc: subprocess.Popen | None = field(default=None, repr=False)
    pid: int | None = field(default=None, repr=False)
    restart_count: int = field(default=0, repr=False)
    last_start: float = field(default=0.0, repr=False)
    last_exit_code: int | None = field(default=None, repr=False)
    state: str = field(default="stopped", repr=False)  # stopped, running, restarting, failed


# ── Watchdog ────────────────────────────────────────────────────────────────


class ProcessWatchdog:
    """Supervises child processes with auto-restart and health polling.

    Features:
    - Start/stop/restart managed processes
    - Auto-restart on crash with exponential backoff
    - HTTP health endpoint polling
    - Graceful shutdown on SIGINT/SIGTERM (and SIGBREAK on Windows)
    - JSON status file for external tooling
    - Thread-safe operations
    """

    def __init__(
        self,
        poll_interval: float = 10.0,
        status_file: str | None = None,
    ) -> None:
        self._processes: dict[str, ManagedProcess] = {}
        self._lock = Lock()
        self._stop_event = Event()
        self._poll_interval = poll_interval
        self._status_file = Path(status_file) if status_file else PROJECT_ROOT / "logs" / "watchdog_status.json"
        self._started_at: float | None = None

    # ── Registration ────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        cmd: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        max_restarts: int = 3,
        backoff_base: float = 5.0,
        health_url: str | None = None,
        critical: bool = False,
    ) -> None:
        """Register a process to be supervised."""
        with self._lock:
            if name in self._processes:
                logger.warning("Process %r already registered — replacing", name)
            self._processes[name] = ManagedProcess(
                name=name,
                cmd=cmd,
                cwd=cwd or str(PROJECT_ROOT),
                env=env,
                max_restarts=max_restarts,
                backoff_base=backoff_base,
                health_url=health_url,
                critical=critical,
            )
            logger.info("Registered managed process: %s (%s)", name, " ".join(cmd[:3]))

    # ── Process Lifecycle ───────────────────────────────────────────────

    def _start_process(self, mp: ManagedProcess) -> bool:
        """Start a single managed process. Returns True on success."""
        try:
            merged_env = {**os.environ, **(mp.env or {})}
            mp.proc = subprocess.Popen(
                mp.cmd,
                cwd=mp.cwd,
                env=merged_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            mp.pid = mp.proc.pid
            mp.last_start = time.monotonic()
            mp.state = "running"
            logger.info("[watchdog] Started %s (PID %d)", mp.name, mp.pid)
            return True
        except Exception as exc:
            mp.state = "failed"
            logger.error("[watchdog] Failed to start %s: %s", mp.name, exc)
            return False

    def _stop_process(self, mp: ManagedProcess, timeout: float = 10.0) -> None:
        """Stop a managed process gracefully, then forcefully."""
        if mp.proc is None:
            return
        try:
            mp.proc.terminate()
            mp.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("[watchdog] %s didn't stop in %ds — killing", mp.name, timeout)
            mp.proc.kill()
            mp.proc.wait(timeout=5)
        except Exception as exc:
            logger.error("[watchdog] Error stopping %s: %s", mp.name, exc)
        finally:
            mp.last_exit_code = mp.proc.returncode if mp.proc else None
            mp.proc = None
            mp.pid = None
            mp.state = "stopped"
            logger.info("[watchdog] Stopped %s (exit=%s)", mp.name, mp.last_exit_code)

    def _check_health(self, mp: ManagedProcess) -> bool:
        """Poll health URL. Returns True if healthy."""
        if not mp.health_url:
            return True  # No health URL = assume healthy if running
        try:
            req = urllib.request.Request(mp.health_url, method="GET")
            with urllib.request.urlopen(req, timeout=mp.health_timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _handle_restart(self, mp: ManagedProcess) -> bool:
        """Handle restart logic with backoff. Returns True if restarted."""
        if mp.restart_count >= mp.max_restarts:
            mp.state = "failed"
            logger.error(
                "[watchdog] %s exhausted %d restarts — marking FAILED",
                mp.name, mp.max_restarts,
            )
            return False

        mp.restart_count += 1
        delay = min(mp.backoff_base * (2 ** (mp.restart_count - 1)), mp.backoff_max)
        mp.state = "restarting"
        logger.warning(
            "[watchdog] %s died (exit=%s) — restart %d/%d in %.0fs",
            mp.name, mp.last_exit_code, mp.restart_count, mp.max_restarts, delay,
        )

        # Wait with backoff (interruptible)
        if self._stop_event.wait(timeout=delay):
            return False  # Shutdown requested during backoff

        return self._start_process(mp)

    # ── Main Loop ───────────────────────────────────────────────────────

    def start_all(self) -> dict[str, bool]:
        """Start all registered processes. Returns {name: success}."""
        results = {}
        with self._lock:
            for name, mp in self._processes.items():
                results[name] = self._start_process(mp)
        return results

    def stop_all(self) -> None:
        """Stop all managed processes gracefully."""
        with self._lock:
            for mp in self._processes.values():
                if mp.state == "running":
                    self._stop_process(mp)

    def run_forever(self) -> None:
        """Main supervision loop. Blocks until SIGINT/SIGTERM."""
        self._started_at = time.monotonic()
        self._install_signal_handlers()

        logger.info("[watchdog] === PROCESS WATCHDOG ACTIVE ===")
        logger.info("[watchdog] Supervising %d processes, poll every %.0fs",
                     len(self._processes), self._poll_interval)

        # Start all processes
        results = self.start_all()
        for name, ok in results.items():
            if not ok:
                logger.error("[watchdog] Initial start failed: %s", name)

        # Supervision loop
        while not self._stop_event.is_set():
            self._supervision_tick()
            self._write_status()
            self._stop_event.wait(timeout=self._poll_interval)

        # Shutdown
        logger.info("[watchdog] Shutdown signal received — stopping all processes")
        self.stop_all()
        self._write_status()
        logger.info("[watchdog] === WATCHDOG STOPPED ===")

    def _supervision_tick(self) -> None:
        """One pass of the supervision loop."""
        with self._lock:
            for mp in self._processes.values():
                if mp.state == "failed":
                    continue  # Exhausted restarts
                if mp.proc is None:
                    continue  # Not started

                rc = mp.proc.poll()
                if rc is not None:
                    # Process died
                    mp.last_exit_code = rc
                    mp.proc = None
                    mp.pid = None
                    self._handle_restart(mp)
                elif mp.health_url:
                    # Process running — check health
                    if not self._check_health(mp):
                        uptime = time.monotonic() - mp.last_start
                        if uptime > 30:  # Grace period after start
                            logger.warning(
                                "[watchdog] %s health check failed (uptime %.0fs)",
                                mp.name, uptime,
                            )

                # Check for critical failures
                if mp.critical and mp.state == "failed":
                    logger.critical(
                        "[watchdog] CRITICAL process %s exhausted %d/%d restarts "
                        "(last exit code: %s) — initiating graceful shutdown",
                        mp.name, mp.restart_count, mp.max_restarts,
                        mp.last_exit_code,
                    )
                    logger.critical(
                        "[watchdog] To recover: restart manually with "
                        "'python launch.py all' or check logs/watchdog_status.json",
                    )
                    self._write_status()  # persist final state before exit
                    self._stop_event.set()

    # ── Signal Handling ─────────────────────────────────────────────────

    def _install_signal_handlers(self) -> None:
        """Install handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        # Windows-specific
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("[watchdog] Received %s", sig_name)
        self._stop_event.set()

    # ── Status Reporting ────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return current watchdog status as a dict."""
        uptime = time.monotonic() - self._started_at if self._started_at else 0
        procs = {}
        with self._lock:
            for name, mp in self._processes.items():
                procs[name] = {
                    "state": mp.state,
                    "pid": mp.pid,
                    "restarts": mp.restart_count,
                    "max_restarts": mp.max_restarts,
                    "last_exit_code": mp.last_exit_code,
                    "critical": mp.critical,
                }
        return {
            "watchdog": "running" if not self._stop_event.is_set() else "stopping",
            "uptime_seconds": round(uptime, 1),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processes": procs,
        }

    def _write_status(self) -> None:
        """Write status to JSON file for external monitoring."""
        try:
            self._status_file.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(self.status(), indent=2)
            # Atomic write (write tmp, rename)
            tmp = self._status_file.with_suffix(".tmp")
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(self._status_file)
        except Exception as exc:
            logger.debug("[watchdog] Status write failed: %s", exc)

    def shutdown(self) -> None:
        """Request graceful shutdown (can be called from any thread)."""
        self._stop_event.set()
