"""Tests for startup.watchdog — Process supervision."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock, patch

import pytest

from startup.watchdog import ManagedProcess, ProcessWatchdog

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── ManagedProcess defaults ─────────────────────────────────────────────────


class TestManagedProcess:
    def test_defaults(self):
        mp = ManagedProcess(name="test", cmd=["echo", "hi"])
        assert mp.name == "test"
        assert mp.state == "stopped"
        assert mp.pid is None
        assert mp.restart_count == 0
        assert mp.max_restarts == 3
        assert mp.proc is None
        assert mp.critical is False

    def test_repr_no_crash(self):
        mp = ManagedProcess(name="x", cmd=["y"])
        assert "x" in repr(mp)


# ── ProcessWatchdog registration ────────────────────────────────────────────


class TestWatchdogRegistration:
    def test_register(self):
        dog = ProcessWatchdog()
        dog.register("svc", [sys.executable, "-c", "pass"])
        assert "svc" in dog._processes
        assert dog._processes["svc"].state == "stopped"

    def test_register_replaces(self):
        dog = ProcessWatchdog()
        dog.register("svc", ["old"])
        dog.register("svc", ["new"])
        assert dog._processes["svc"].cmd == ["new"]

    def test_register_with_options(self):
        dog = ProcessWatchdog()
        dog.register(
            "svc",
            ["cmd"],
            max_restarts=5,
            backoff_base=2.0,
            health_url="http://localhost:9999/health",
            critical=True,
        )
        mp = dog._processes["svc"]
        assert mp.max_restarts == 5
        assert mp.backoff_base == 2.0
        assert mp.health_url == "http://localhost:9999/health"
        assert mp.critical is True


# ── Start / Stop ────────────────────────────────────────────────────────────


class TestStartStop:
    def test_start_process(self):
        dog = ProcessWatchdog()
        mp = ManagedProcess(
            name="sleeper",
            cmd=[sys.executable, "-c", "import time; time.sleep(60)"],
        )
        ok = dog._start_process(mp)
        assert ok
        assert mp.state == "running"
        assert mp.pid is not None
        assert mp.proc is not None
        # Cleanup
        dog._stop_process(mp)
        assert mp.state == "stopped"
        assert mp.pid is None

    def test_start_bad_command(self):
        dog = ProcessWatchdog()
        mp = ManagedProcess(name="bad", cmd=["__nonexistent_binary__"])
        ok = dog._start_process(mp)
        assert not ok
        assert mp.state == "failed"

    def test_stop_already_stopped(self):
        dog = ProcessWatchdog()
        mp = ManagedProcess(name="x", cmd=["y"])
        dog._stop_process(mp)  # Should not raise
        assert mp.state == "stopped"

    def test_start_all(self):
        dog = ProcessWatchdog()
        dog.register("a", [sys.executable, "-c", "import time; time.sleep(60)"])
        dog.register("b", [sys.executable, "-c", "import time; time.sleep(60)"])
        results = dog.start_all()
        assert results["a"] is True
        assert results["b"] is True
        dog.stop_all()
        for mp in dog._processes.values():
            assert mp.state == "stopped"

    def test_stop_all(self):
        dog = ProcessWatchdog()
        dog.register("s", [sys.executable, "-c", "import time; time.sleep(60)"])
        dog.start_all()
        dog.stop_all()
        assert dog._processes["s"].state == "stopped"


# ── Restart Logic ───────────────────────────────────────────────────────────


class TestRestartLogic:
    def test_restart_increments_count(self):
        dog = ProcessWatchdog()
        mp = ManagedProcess(
            name="restartable",
            cmd=[sys.executable, "-c", "import time; time.sleep(60)"],
            max_restarts=3,
            backoff_base=0.01,  # Fast for tests
        )
        mp.last_exit_code = 1
        ok = dog._handle_restart(mp)
        assert ok
        assert mp.restart_count == 1
        dog._stop_process(mp)

    def test_restart_exhausted(self):
        dog = ProcessWatchdog()
        mp = ManagedProcess(
            name="doomed",
            cmd=[sys.executable, "-c", "pass"],
            max_restarts=2,
        )
        mp.restart_count = 2  # Already at limit
        mp.last_exit_code = 1
        ok = dog._handle_restart(mp)
        assert not ok
        assert mp.state == "failed"

    def test_backoff_capped(self):
        mp = ManagedProcess(
            name="x",
            cmd=["y"],
            backoff_base=10.0,
            backoff_max=30.0,
        )
        mp.restart_count = 0
        # First restart: 10 * 2^0 = 10
        # Second: 10 * 2^1 = 20
        # Third: 10 * 2^2 = 40 → capped at 30
        delay1 = min(mp.backoff_base * (2 ** 0), mp.backoff_max)
        delay3 = min(mp.backoff_base * (2 ** 2), mp.backoff_max)
        assert delay1 == 10.0
        assert delay3 == 30.0


# ── Health Checking ─────────────────────────────────────────────────────────


class TestHealthCheck:
    def test_no_health_url_returns_true(self):
        dog = ProcessWatchdog()
        mp = ManagedProcess(name="x", cmd=["y"], health_url=None)
        assert dog._check_health(mp) is True

    @patch("startup.watchdog.urllib.request.urlopen")
    def test_health_success(self, mock_urlopen):
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        dog = ProcessWatchdog()
        mp = ManagedProcess(name="x", cmd=["y"], health_url="http://127.0.0.1:8080/health")
        assert dog._check_health(mp) is True

    @patch("startup.watchdog.urllib.request.urlopen", side_effect=Exception("timeout"))
    def test_health_failure(self, _):
        dog = ProcessWatchdog()
        mp = ManagedProcess(name="x", cmd=["y"], health_url="http://127.0.0.1:9999/health")
        assert dog._check_health(mp) is False


# ── Status Reporting ────────────────────────────────────────────────────────


class TestStatus:
    def test_status_dict(self):
        dog = ProcessWatchdog()
        dog._started_at = time.monotonic() - 60
        dog.register("svc", ["cmd"])
        status = dog.status()
        assert status["watchdog"] == "running"
        assert "processes" in status
        assert "svc" in status["processes"]
        assert status["processes"]["svc"]["state"] == "stopped"

    def test_write_status_creates_file(self, tmp_path):
        f = tmp_path / "status.json"
        dog = ProcessWatchdog(status_file=str(f))
        dog._started_at = time.monotonic()
        dog.register("x", ["y"])
        dog._write_status()
        assert f.exists()
        data = json.loads(f.read_text())
        assert data["watchdog"] == "running"
        assert "x" in data["processes"]


# ── Shutdown ────────────────────────────────────────────────────────────────


class TestShutdown:
    def test_shutdown_sets_event(self):
        dog = ProcessWatchdog()
        assert not dog._stop_event.is_set()
        dog.shutdown()
        assert dog._stop_event.is_set()


# ── Supervision Loop (quick) ───────────────────────────────────────────────


class TestSupervisionLoop:
    def test_run_forever_exits_on_shutdown(self):
        """Watchdog exits promptly when shutdown() is called."""
        dog = ProcessWatchdog(poll_interval=0.5)
        dog.register("quick", [sys.executable, "-c", "import time; time.sleep(300)"])

        def stop_soon():
            time.sleep(1.0)
            dog.shutdown()

        Thread(target=stop_soon, daemon=True).start()
        dog.run_forever()  # Should return within ~1-2s
        assert dog._stop_event.is_set()
        assert dog._processes["quick"].state == "stopped"

    def test_auto_restart_on_crash(self):
        """Process that exits immediately gets restarted."""
        dog = ProcessWatchdog(poll_interval=0.5)
        # This process exits immediately with code 42
        dog.register(
            "crasher",
            [sys.executable, "-c", "import sys; sys.exit(42)"],
            max_restarts=1,
            backoff_base=0.1,
        )

        def stop_later():
            time.sleep(3.0)
            dog.shutdown()

        Thread(target=stop_later, daemon=True).start()
        dog.run_forever()

        mp = dog._processes["crasher"]
        assert mp.restart_count >= 1
        assert mp.last_exit_code == 42
