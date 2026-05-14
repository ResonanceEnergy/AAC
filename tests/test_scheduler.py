from __future__ import annotations
"""tests/test_scheduler.py — Sprint 7: MarketScheduler tests.

All tests are deterministic and fully offline — external dependencies
(signal_aggregator, roll_manager, pnl_tracker, health_monitor, position_tracker)
are mocked throughout.
"""

import threading
import time as _time_module
from datetime import date, datetime, time
from unittest.mock import MagicMock, patch, call
from zoneinfo import ZoneInfo

import pytest

from core.market_scheduler import MarketScheduler, _ET

# ── fixtures / helpers ────────────────────────────────────────────────────────

_ET_ZONE = ZoneInfo("America/New_York")


def _dt(weekday: int, hour: int, minute: int = 0) -> datetime:
    """Build a timezone-aware datetime with a specific weekday/time.

    weekday: 0=Mon, 1=Tue, ..., 4=Fri, 5=Sat, 6=Sun
    We anchor to 2025-01-06 (Mon) as the base week.
    """
    base_monday = datetime(2025, 1, 6, tzinfo=_ET_ZONE)
    from datetime import timedelta
    return base_monday + timedelta(days=weekday, hours=hour, minutes=minute)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Market-hour classification
# ══════════════════════════════════════════════════════════════════════════════


class TestIsTradingDay:
    def test_monday_is_trading_day(self):
        assert MarketScheduler.is_trading_day(_dt(0, 10)) is True

    def test_friday_is_trading_day(self):
        assert MarketScheduler.is_trading_day(_dt(4, 10)) is True

    def test_saturday_not_trading_day(self):
        assert MarketScheduler.is_trading_day(_dt(5, 10)) is False

    def test_sunday_not_trading_day(self):
        assert MarketScheduler.is_trading_day(_dt(6, 10)) is False


class TestIsMarketHours:
    def test_during_session(self):
        assert MarketScheduler.is_market_hours(_dt(0, 12)) is True

    def test_at_open(self):
        assert MarketScheduler.is_market_hours(_dt(1, 9, 30)) is True

    def test_at_close_boundary_excluded(self):
        # 16:00 is NOT market hours (boundary is exclusive)
        assert MarketScheduler.is_market_hours(_dt(1, 16, 0)) is False

    def test_pre_market(self):
        assert MarketScheduler.is_market_hours(_dt(1, 9, 0)) is False

    def test_after_hours(self):
        assert MarketScheduler.is_market_hours(_dt(1, 17, 0)) is False

    def test_weekend_always_false(self):
        assert MarketScheduler.is_market_hours(_dt(5, 12)) is False


class TestIsMarketOpenWindow:
    def test_inside_window(self):
        assert MarketScheduler.is_market_open_window(_dt(0, 9, 32)) is True

    def test_at_window_start(self):
        assert MarketScheduler.is_market_open_window(_dt(0, 9, 30)) is True

    def test_after_window_end(self):
        assert MarketScheduler.is_market_open_window(_dt(0, 9, 40)) is False

    def test_weekend_false(self):
        assert MarketScheduler.is_market_open_window(_dt(5, 9, 32)) is False


class TestIsMarketCloseWindow:
    def test_inside_window(self):
        assert MarketScheduler.is_market_close_window(_dt(0, 16, 3)) is True

    def test_at_window_start(self):
        assert MarketScheduler.is_market_close_window(_dt(0, 16, 0)) is True

    def test_after_window_end(self):
        assert MarketScheduler.is_market_close_window(_dt(0, 16, 10)) is False

    def test_before_close(self):
        assert MarketScheduler.is_market_close_window(_dt(0, 15, 59)) is False

    def test_weekend_false(self):
        assert MarketScheduler.is_market_close_window(_dt(5, 16, 3)) is False


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Interval tracking: _should_run / _mark_done
# ══════════════════════════════════════════════════════════════════════════════


class TestShouldRun:
    def test_never_run_returns_true(self):
        sched = MarketScheduler()
        assert sched._should_run("health_check", 300) is True

    def test_just_marked_returns_false(self):
        import time as _t
        sched = MarketScheduler()
        sched._last_run["health_check"] = _t.monotonic()
        assert sched._should_run("health_check", 300) is False

    def test_after_interval_returns_true(self):
        import time as _t
        sched = MarketScheduler()
        # Simulate the task ran 400 seconds ago
        sched._last_run["health_check"] = _t.monotonic() - 400
        assert sched._should_run("health_check", 300) is True

    def test_mark_done_updates_last_run(self):
        import time as _t
        sched = MarketScheduler()
        before = _t.monotonic()
        sched._mark_done("signal_scan")
        assert sched._last_run["signal_scan"] >= before


# ══════════════════════════════════════════════════════════════════════════════
# 3.  _run_task: success / failure
# ══════════════════════════════════════════════════════════════════════════════


class TestRunTask:
    def test_success_marks_done_and_returns_true(self):
        sched = MarketScheduler()
        calls = []
        result = sched._run_task("demo", lambda: calls.append(1))
        assert result is True
        assert calls == [1]
        assert "demo" in sched._last_run

    def test_exception_returns_false_does_not_raise(self):
        sched = MarketScheduler()
        result = sched._run_task("boom", lambda: (_ for _ in ()).throw(ValueError("oops")))
        assert result is False
        assert "boom" not in sched._last_run

    def test_exception_does_not_affect_other_tasks(self):
        sched = MarketScheduler()
        sched._run_task("bad", lambda: (_ for _ in ()).throw(RuntimeError("x")))
        # A subsequent good task still works
        result = sched._run_task("good", lambda: None)
        assert result is True


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Individual task methods (all external calls mocked)
# ══════════════════════════════════════════════════════════════════════════════


class TestRunSignalScan:
    @patch("strategies.signal_aggregator.get_combined_signals", return_value=["sig1"])
    def test_returns_signals(self, mock_fn):
        sched = MarketScheduler()
        result = sched.run_signal_scan()
        assert result == ["sig1"]
        mock_fn.assert_called_once()

    @patch("strategies.signal_aggregator.get_combined_signals", return_value=[])
    def test_empty_signals_ok(self, mock_fn):
        sched = MarketScheduler()
        assert sched.run_signal_scan() == []


class TestRunRollCheck:
    @patch("asyncio.run", return_value=[])
    @patch("TradingExecution.position_tracker.PositionTracker")
    @patch("strategies.roll_manager.RollManager")
    def test_returns_urgent_decisions(self, MockRM, MockPT, mock_asyncio_run):
        decision = MagicMock()
        decision.symbol = "IWM"
        instance = MockRM.return_value
        instance.urgent_only.return_value = [decision]

        sched = MarketScheduler()
        result = sched.run_roll_check()
        assert len(result) == 1
        assert result[0].symbol == "IWM"

    @patch("asyncio.run", side_effect=ConnectionRefusedError("IBKR offline"))
    @patch("TradingExecution.position_tracker.PositionTracker")
    @patch("strategies.roll_manager.RollManager")
    def test_ibkr_unavailable_returns_empty_decisions(self, MockRM, MockPT, mock_asyncio_run):
        instance = MockRM.return_value
        instance.urgent_only.return_value = []

        sched = MarketScheduler()
        result = sched.run_roll_check()
        assert isinstance(result, list)


class TestRunPnlSnapshot:
    @patch("asyncio.run", return_value=[])
    @patch("TradingExecution.position_tracker.PositionTracker")
    @patch("CentralAccounting.pnl_tracker.PnLTracker")
    def test_returns_report_dict(self, MockPT, MockPos, mock_asyncio_run):
        report = {"total_pnl": 150.0, "positions": []}
        instance = MockPT.return_value
        instance.take_snapshot.return_value = report

        sched = MarketScheduler()
        result = sched.run_pnl_snapshot()
        assert result["total_pnl"] == 150.0
        # Shared tracker is NOT closed after each snapshot call
        instance.close.assert_not_called()

    @patch("asyncio.run", return_value=[])
    @patch("TradingExecution.position_tracker.PositionTracker")
    @patch("CentralAccounting.pnl_tracker.PnLTracker")
    def test_pnl_snapshot_error_returns_empty_dict(self, MockPT, MockPos, mock_asyncio_run):
        """Snapshot errors are caught and logged; run_pnl_snapshot returns {}."""
        instance = MockPT.return_value
        instance.take_snapshot.side_effect = RuntimeError("db locked")

        sched = MarketScheduler()
        result = sched.run_pnl_snapshot()
        assert result == {}

    @patch("asyncio.run", side_effect=ConnectionRefusedError("IBKR offline"))
    @patch("TradingExecution.position_tracker.PositionTracker")
    @patch("CentralAccounting.pnl_tracker.PnLTracker")
    def test_ibkr_offline_still_snapshots(self, MockPT, MockPos, mock_asyncio_run):
        report = {"total_pnl": 0.0}
        instance = MockPT.return_value
        instance.take_snapshot.return_value = report
        instance.close = MagicMock()

        sched = MarketScheduler()
        result = sched.run_pnl_snapshot()
        assert result == {"total_pnl": 0.0}


class TestRunHealthCheck:
    @patch("monitoring.health_monitor.HealthMonitor")
    def test_returns_snapshot(self, MockHM):
        snap = MagicMock()
        snap.overall_status.value = "healthy"
        snap.active_alerts = []
        MockHM.return_value.collect_snapshot.return_value = snap

        sched = MarketScheduler()
        result = sched.run_health_check()
        assert result is snap
        MockHM.return_value.collect_snapshot.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Scheduler.run() — integration with mocked tasks
# ══════════════════════════════════════════════════════════════════════════════


def _make_sched_with_mocked_tasks(
    tick: int = 0,
    scan_interval: int = 1,
    health_interval: int = 1,
) -> tuple[MarketScheduler, dict]:
    """Helper: build a scheduler with all task methods replaced by mocks."""
    sched = MarketScheduler(
        scan_interval=scan_interval,
        health_interval=health_interval,
        tick=tick,
    )
    mocks = {
        "health": MagicMock(return_value=MagicMock(overall_status=MagicMock(value="healthy"), active_alerts=[])),
        "scan": MagicMock(return_value=[]),
        "roll": MagicMock(return_value=[]),
        "pnl": MagicMock(return_value={}),
    }
    sched.run_health_check = mocks["health"]
    sched.run_signal_scan = mocks["scan"]
    sched.run_roll_check = mocks["roll"]
    sched.run_pnl_snapshot = mocks["pnl"]
    return sched, mocks


def _mark_tasks_done(sched: MarketScheduler, *task_names: str) -> None:
    """Pre-mark tasks so they won't fire on the next scheduler tick.

    Sets last-run time to far future so _should_run always returns False.
    """
    far_future = _time_module.monotonic() + 999_999
    for name in task_names:
        sched._last_run[name] = far_future


class TestRunLoopHealthAlwaysRuns:
    def test_health_check_fires_outside_market_hours(self):
        """Health check should run even at midnight on Sunday."""
        sched, mocks = _make_sched_with_mocked_tasks()
        sunday_midnight = _dt(6, 0)

        # Stop the loop after health check fires
        mocks["health"].side_effect = sched.stop
        _mark_tasks_done(sched, "signal_scan", "roll_check", "pnl_snapshot")

        with patch("core.market_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = sunday_midnight
            sched.run()

        mocks["health"].assert_called_once()

    def test_signal_scan_skipped_outside_market_hours(self):
        """Signal scan must NOT fire on weekends or outside 9:30–16:00."""
        sched, mocks = _make_sched_with_mocked_tasks()
        sunday = _dt(6, 12)

        # Stop after health check fires (health always runs); scan should not
        mocks["health"].side_effect = sched.stop
        _mark_tasks_done(sched, "signal_scan", "roll_check", "pnl_snapshot")

        with patch("core.market_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = sunday
            sched.run()

        mocks["scan"].assert_not_called()


class TestRunLoopMarketHourTasks:
    def test_signal_scan_fires_during_market_hours(self):
        sched, mocks = _make_sched_with_mocked_tasks(scan_interval=0, health_interval=0)
        tuesday_noon = _dt(1, 12)

        # Pre-mark health so only scan fires first; stop loop when scan runs
        _mark_tasks_done(sched, "health_check", "roll_check", "pnl_snapshot")
        mocks["scan"].side_effect = sched.stop

        with patch("core.market_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = tuesday_noon
            sched.run()

        mocks["scan"].assert_called_once()

    def test_roll_check_fires_at_open_window(self):
        sched, mocks = _make_sched_with_mocked_tasks(health_interval=0, scan_interval=0)
        wednesday_open = _dt(2, 9, 33)

        _mark_tasks_done(sched, "health_check", "signal_scan", "pnl_snapshot")
        mocks["roll"].side_effect = sched.stop

        with patch("core.market_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = wednesday_open
            sched.run()

        mocks["roll"].assert_called_once()

    def test_pnl_snapshot_fires_at_close_window(self):
        sched, mocks = _make_sched_with_mocked_tasks(health_interval=0, scan_interval=0)
        thursday_close = _dt(3, 16, 2)

        _mark_tasks_done(sched, "health_check", "signal_scan", "roll_check")
        mocks["pnl"].side_effect = sched.stop

        with patch("core.market_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = thursday_close
            sched.run()

        mocks["pnl"].assert_called_once()

    def test_roll_check_does_not_fire_outside_open_window(self):
        sched, mocks = _make_sched_with_mocked_tasks(health_interval=0, scan_interval=0)
        wednesday_afternoon = _dt(2, 14, 0)

        # Stop after health check; roll should not run
        _mark_tasks_done(sched, "signal_scan", "pnl_snapshot")
        mocks["health"].side_effect = sched.stop

        with patch("core.market_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = wednesday_afternoon
            sched.run()

        mocks["roll"].assert_not_called()

    def test_pnl_snapshot_does_not_fire_before_close(self):
        sched, mocks = _make_sched_with_mocked_tasks(health_interval=0, scan_interval=0)
        friday_noon = _dt(4, 12)

        _mark_tasks_done(sched, "signal_scan", "roll_check")
        mocks["health"].side_effect = sched.stop

        with patch("core.market_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = friday_noon
            sched.run()

        mocks["pnl"].assert_not_called()


class TestOncePerDayGuard:
    def test_roll_check_runs_only_once_per_day(self):
        sched, mocks = _make_sched_with_mocked_tasks(health_interval=0, scan_interval=0)
        monday_open = _dt(0, 9, 33)
        today = monday_open.date()
        sched._last_roll_date = today  # already ran today

        # Stop after health check; roll must not fire again
        _mark_tasks_done(sched, "signal_scan", "pnl_snapshot")
        mocks["health"].side_effect = sched.stop

        with patch("core.market_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = monday_open
            sched.run()

        mocks["roll"].assert_not_called()

    def test_pnl_snapshot_runs_only_once_per_day(self):
        sched, mocks = _make_sched_with_mocked_tasks(health_interval=0, scan_interval=0)
        monday_close = _dt(0, 16, 3)
        today = monday_close.date()
        sched._last_pnl_date = today  # already ran today

        _mark_tasks_done(sched, "signal_scan", "roll_check")
        mocks["health"].side_effect = sched.stop

        with patch("core.market_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = monday_close
            sched.run()

        mocks["pnl"].assert_not_called()

    def test_roll_check_runs_again_next_day(self):
        sched, mocks = _make_sched_with_mocked_tasks(health_interval=0, scan_interval=0)
        tuesday_open = _dt(1, 9, 33)
        yesterday = _dt(0, 9, 33).date()
        sched._last_roll_date = yesterday  # ran yesterday, not today

        _mark_tasks_done(sched, "health_check", "signal_scan", "pnl_snapshot")
        mocks["roll"].side_effect = sched.stop

        with patch("core.market_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = tuesday_open
            sched.run()

        mocks["roll"].assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  run_forever — crash recovery (7.5)
# ══════════════════════════════════════════════════════════════════════════════


class TestRunForever:
    def test_clean_stop_does_not_increment_restart_count(self):
        sched = MarketScheduler(tick=0)

        def _stop_immediately():
            sched.stop()

        sched.run_health_check = MagicMock(side_effect=_stop_immediately)
        sched.run_signal_scan = MagicMock(return_value=[])
        sched.run_roll_check = MagicMock(return_value=[])
        sched.run_pnl_snapshot = MagicMock(return_value={})

        monday_open = _dt(0, 9, 33)
        with patch("core.market_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = monday_open
            sched.run_forever(max_restarts=5)

        assert sched.restart_count == 0

    def test_crash_in_run_increments_restart_count(self):
        """Patch run() to raise on first call, succeed on second."""
        sched = MarketScheduler(tick=0)
        call_count = [0]

        original_stop = sched.stop

        def _patched_run():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated crash")
            # Second call: set stop and return normally
            original_stop()
            sched._stop.wait(0)  # yield

        with patch.object(sched, "run", side_effect=_patched_run):
            # run_forever only restarts if run() raises; we need to make it
            # NOT raise on second call for clean exit
            try:
                sched.run_forever(max_restarts=3)
            except StopIteration:
                pass

        assert sched.restart_count == 1

    def test_exceeding_max_restarts_raises(self):
        sched = MarketScheduler(tick=0)

        with patch.object(sched, "run", side_effect=RuntimeError("repeated crash")):
            with pytest.raises(RuntimeError, match="repeated crash"):
                sched.run_forever(max_restarts=2)

        assert sched.restart_count == 3

    def test_restart_count_starts_at_zero(self):
        sched = MarketScheduler()
        assert sched.restart_count == 0


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Stop behaviour
# ══════════════════════════════════════════════════════════════════════════════


class TestStop:
    def test_stop_sets_event(self):
        sched = MarketScheduler()
        assert not sched._stop.is_set()
        sched.stop()
        assert sched._stop.is_set()

    def test_run_exits_when_already_stopped(self):
        sched = MarketScheduler(tick=0)
        sched.stop()  # pre-set
        monday_noon = _dt(0, 12)
        with patch("core.market_scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = monday_noon
            sched.run()  # should return immediately


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Configuration
# ══════════════════════════════════════════════════════════════════════════════


class TestConfiguration:
    def test_defaults(self):
        sched = MarketScheduler()
        assert sched._scan_interval == 900
        assert sched._health_interval == 300
        assert sched._tick == 30

    def test_custom_intervals(self):
        sched = MarketScheduler(scan_interval=60, health_interval=30, tick=5)
        assert sched._scan_interval == 60
        assert sched._health_interval == 30
        assert sched._tick == 5

    def test_paper_mode_from_env(self, monkeypatch):
        monkeypatch.setenv("PAPER_TRADING", "true")
        sched = MarketScheduler()
        assert sched._paper is True

    def test_paper_mode_default_false(self, monkeypatch):
        monkeypatch.delenv("PAPER_TRADING", raising=False)
        sched = MarketScheduler()
        assert sched._paper is False

    def test_paper_explicit_override(self):
        sched = MarketScheduler(paper=True)
        assert sched._paper is True
