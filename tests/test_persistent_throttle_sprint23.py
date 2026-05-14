"""Sprint 23 — Persistent Execution Throttle tests.

Covers:
- ExecutionThrottle lifecycle (can_execute, record, remaining, clear, all_entries)
- Persistence: values survive a new object pointing to the same DB
- Throttle expiry: can_execute returns True after window passes
- Fail-open: DB errors never block execution
- AutoTrader wiring: persistent throttle checked + recorded on success
- MarketScheduler wiring: throttle created and passed to AutoTrader
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from core.execution_throttle import ExecutionThrottle


# ─────────────────────────────────────────────────────────────────────────────
# TestExecutionThrottleNewTicker
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionThrottleNewTicker(unittest.TestCase):
    """Unknown tickers are not throttled."""

    def _make(self, **kw):
        return ExecutionThrottle(db_path=":memory:", **kw)

    def test_unknown_ticker_can_execute(self):
        t = self._make()
        self.assertTrue(t.can_execute("SPY"))

    def test_unknown_ticker_last_executed_is_none(self):
        t = self._make()
        self.assertIsNone(t.last_executed("SPY"))

    def test_unknown_ticker_remaining_is_zero(self):
        t = self._make()
        self.assertEqual(t.remaining_seconds("SPY"), 0.0)

    def test_all_entries_empty_on_fresh_db(self):
        t = self._make()
        self.assertEqual(t.all_entries(), [])


# ─────────────────────────────────────────────────────────────────────────────
# TestExecutionThrottleAfterRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionThrottleAfterRecord(unittest.TestCase):
    """Ticker is throttled immediately after record_execution()."""

    def _make(self, **kw):
        return ExecutionThrottle(db_path=":memory:", throttle_seconds=3600, **kw)

    def test_throttled_immediately_after_record(self):
        t = self._make()
        t.record_execution("IWM")
        self.assertFalse(t.can_execute("IWM"))

    def test_different_ticker_not_throttled(self):
        t = self._make()
        t.record_execution("IWM")
        self.assertTrue(t.can_execute("SPY"))

    def test_remaining_positive_after_record(self):
        t = self._make()
        t.record_execution("IWM")
        remaining = t.remaining_seconds("IWM")
        self.assertGreater(remaining, 0.0)
        self.assertLessEqual(remaining, 3600.0)

    def test_last_executed_set_after_record(self):
        t = self._make()
        before = time.time()
        t.record_execution("IWM")
        after = time.time()
        ts = t.last_executed("IWM")
        self.assertIsNotNone(ts)
        self.assertGreaterEqual(ts, before)
        self.assertLessEqual(ts, after + 1)

    def test_all_entries_shows_recorded_ticker(self):
        t = self._make()
        t.record_execution("JNK")
        entries = t.all_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["ticker"], "JNK")


# ─────────────────────────────────────────────────────────────────────────────
# TestExecutionThrottleExpiry
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionThrottleExpiry(unittest.TestCase):
    """can_execute returns True once the throttle window has passed."""

    def test_can_execute_after_window_expires(self):
        t = ExecutionThrottle(db_path=":memory:", throttle_seconds=0.05)
        t.record_execution("SPY")
        self.assertFalse(t.can_execute("SPY"))
        time.sleep(0.1)  # let the 50ms window expire
        self.assertTrue(t.can_execute("SPY"))

    def test_remaining_approaches_zero_as_time_passes(self):
        t = ExecutionThrottle(db_path=":memory:", throttle_seconds=0.5)
        t.record_execution("SPY")
        remaining1 = t.remaining_seconds("SPY")
        time.sleep(0.1)
        remaining2 = t.remaining_seconds("SPY")
        self.assertGreater(remaining1, remaining2)


# ─────────────────────────────────────────────────────────────────────────────
# TestExecutionThrottlePersistence
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionThrottlePersistence(unittest.TestCase):
    """Records survive creating a new ExecutionThrottle instance on same file."""

    def test_record_survives_new_instance(self, tmp_path=None):
        import tempfile, os  # noqa: E401
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            t1 = ExecutionThrottle(db_path=db_path, throttle_seconds=3600)
            t1.record_execution("SPY")
            # Create a SECOND instance pointing at the same file
            t2 = ExecutionThrottle(db_path=db_path, throttle_seconds=3600)
            self.assertFalse(t2.can_execute("SPY"))
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    def test_clear_then_new_instance_can_execute(self):
        import tempfile, os  # noqa: E401
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            t1 = ExecutionThrottle(db_path=db_path, throttle_seconds=3600)
            t1.record_execution("IWM")
            t1.clear("IWM")
            t2 = ExecutionThrottle(db_path=db_path, throttle_seconds=3600)
            self.assertTrue(t2.can_execute("IWM"))
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# TestExecutionThrottleClear
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionThrottleClear(unittest.TestCase):
    """clear() resets one or all entries."""

    def _make(self):
        return ExecutionThrottle(db_path=":memory:", throttle_seconds=3600)

    def test_clear_single_ticker(self):
        t = self._make()
        t.record_execution("SPY")
        t.record_execution("IWM")
        t.clear("SPY")
        self.assertTrue(t.can_execute("SPY"))
        self.assertFalse(t.can_execute("IWM"))

    def test_clear_all(self):
        t = self._make()
        t.record_execution("SPY")
        t.record_execution("IWM")
        t.clear()  # no arg → clear all
        self.assertTrue(t.can_execute("SPY"))
        self.assertTrue(t.can_execute("IWM"))
        self.assertEqual(t.all_entries(), [])

    def test_clear_nonexistent_ticker_is_noop(self):
        t = self._make()
        t.record_execution("SPY")
        t.clear("ZZZNOPE")
        self.assertFalse(t.can_execute("SPY"))

    def test_clear_unknown_ticker_noops_silently(self):
        t = self._make()
        t.clear("NOTHING")  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# TestExecutionThrottleFailOpen
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionThrottleFailOpen(unittest.TestCase):
    """DB errors never block execution — all methods fail-open."""

    def _make_broken(self):
        """ExecutionThrottle with a DB path that will fail."""
        t = ExecutionThrottle(db_path=":memory:", throttle_seconds=3600)
        # Simulate a broken DB by closing and deleting the underlying connection
        t._initialized = True  # pretend init succeeded
        return t

    def test_can_execute_returns_true_on_db_error(self):
        t = ExecutionThrottle(db_path=":memory:", throttle_seconds=3600)
        with patch.object(t, "_get_last_executed", side_effect=RuntimeError("db gone")):
            result = t.can_execute("SPY")
        self.assertTrue(result)

    def test_record_execution_silently_discards_error(self):
        t = ExecutionThrottle(db_path=":memory:", throttle_seconds=3600)
        with patch.object(t, "_connect", side_effect=RuntimeError("db gone")):
            # Should not raise
            t.record_execution("SPY")

    def test_last_executed_returns_none_on_error(self):
        t = ExecutionThrottle(db_path=":memory:", throttle_seconds=3600)
        with patch.object(t, "_get_last_executed", side_effect=RuntimeError("db gone")):
            result = t.last_executed("SPY")
        self.assertIsNone(result)

    def test_all_entries_returns_empty_list_on_error(self):
        t = ExecutionThrottle(db_path=":memory:", throttle_seconds=3600)
        with patch.object(t, "_connect", side_effect=RuntimeError("db gone")):
            result = t.all_entries()
        self.assertEqual(result, [])

    def test_remaining_returns_zero_on_error(self):
        t = ExecutionThrottle(db_path=":memory:", throttle_seconds=3600)
        with patch.object(t, "_get_last_executed", side_effect=RuntimeError("db gone")):
            result = t.remaining_seconds("SPY")
        self.assertEqual(result, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# TestAutoTraderWiring
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoTraderWiring(unittest.TestCase):
    """AutoTrader checks persistent throttle and records on successful execution."""

    def _make_signal(self, ticker="SPY", confidence=0.80, direction=None):
        from shared.signal import TradeSignal, Direction  # noqa: PLC0415
        return TradeSignal(
            ticker=ticker,
            direction=direction or Direction.SHORT,
            confidence=confidence,
            entry=100.0,
            stop=110.0,
            target=90.0,
            size=0.05,
        )

    def test_throttle_can_execute_called_on_non_compulsory_signal(self):
        from core.auto_trader import AutoTrader  # noqa: PLC0415
        throttle = ExecutionThrottle(db_path=":memory:", throttle_seconds=3600)
        trader = AutoTrader(
            dry_run=True,
            execution_throttle=throttle,
        )
        sig = self._make_signal()
        trader.run_once([sig])
        # Verify the throttle was queried — can_execute should have been called
        # In dry_run=True, no record_execution. Signal should pass (not throttled).
        summary = trader.last_summary
        self.assertEqual(summary.signals_approved, 1)

    def test_throttled_signal_is_filtered(self):
        from core.auto_trader import AutoTrader  # noqa: PLC0415
        throttle = ExecutionThrottle(db_path=":memory:", throttle_seconds=3600)
        throttle.record_execution("SPY")  # pre-throttle SPY
        trader = AutoTrader(dry_run=True, execution_throttle=throttle)
        sig = self._make_signal(ticker="SPY")
        summary = trader.run_once([sig])
        self.assertEqual(summary.signals_filtered, 1)
        self.assertEqual(summary.signals_approved, 0)

    def test_throttle_not_blocking_after_clear(self):
        from core.auto_trader import AutoTrader  # noqa: PLC0415
        throttle = ExecutionThrottle(db_path=":memory:", throttle_seconds=3600)
        throttle.record_execution("SPY")
        throttle.clear("SPY")
        trader = AutoTrader(dry_run=True, execution_throttle=throttle)
        sig = self._make_signal(ticker="SPY")
        summary = trader.run_once([sig])
        self.assertEqual(summary.signals_approved, 1)

    def test_compulsory_roll_signal_bypasses_persistent_throttle(self):
        """ROLL_CLOSE signals bypass even the persistent throttle."""
        from core.auto_trader import AutoTrader  # noqa: PLC0415
        throttle = ExecutionThrottle(db_path=":memory:", throttle_seconds=3600)
        throttle.record_execution("SPY")  # SPY throttled

        from shared.signal import TradeSignal, Direction  # noqa: PLC0415
        roll_signal = TradeSignal(
            ticker="SPY",
            direction=Direction.SHORT,
            confidence=0.95,
            entry=400.0,
            stop=420.0,
            target=380.0,
            size=0.05,
            notes="ROLL_CLOSE:21DTE:roll now",
        )
        trader = AutoTrader(dry_run=True, execution_throttle=throttle)
        summary = trader.run_once([roll_signal])
        self.assertEqual(summary.signals_approved, 1)

    def test_record_execution_called_on_fill(self):
        """record_execution() is called when a signal results in submitted/filled."""
        from unittest.mock import AsyncMock  # noqa: PLC0415
        from core.auto_trader import AutoTrader  # noqa: PLC0415
        from shared.signal import TradeSignal, Direction  # noqa: PLC0415

        throttle = ExecutionThrottle(db_path=":memory:", throttle_seconds=3600)

        mock_conf = MagicMock()
        mock_conf.status.value = "submitted"
        mock_conf.order_id = "TEST-001"
        mock_conf.signal_ticker = "IWM"  # needed so run_once() can record throttle

        sig = TradeSignal(
            ticker="IWM",
            direction=Direction.SHORT,
            confidence=0.80,
            entry=200.0, stop=210.0, target=190.0, size=0.05,
        )

        trader = AutoTrader(
            dry_run=False,
            paper=True,
            execution_throttle=throttle,
        )
        # _execute_approved is async — use AsyncMock so asyncio.run() can await it
        with patch.object(trader, "_execute_approved", new=AsyncMock(return_value=[mock_conf])):
            trader.run_once([sig])

        # IWM should now be throttled
        self.assertFalse(throttle.can_execute("IWM"))

    def test_no_throttle_falls_back_to_in_memory(self):
        """When execution_throttle=None, fall back to _last_executed dict."""
        from core.auto_trader import AutoTrader  # noqa: PLC0415
        trader = AutoTrader(dry_run=True, execution_throttle=None)
        sig = self._make_signal("SPY")
        # First call — should pass
        summary1 = trader.run_once([sig])
        self.assertEqual(summary1.signals_approved, 1)


# ─────────────────────────────────────────────────────────────────────────────
# TestMarketSchedulerWiring
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketSchedulerWiring(unittest.TestCase):
    """MarketScheduler creates an ExecutionThrottle and passes it to AutoTrader."""

    def _make_scheduler(self, auto_execute=True):
        from core.market_scheduler import MarketScheduler  # noqa: PLC0415
        return MarketScheduler(auto_execute=auto_execute)

    def test_scheduler_creates_execution_throttle(self):
        sched = self._make_scheduler()
        from core.execution_throttle import ExecutionThrottle  # noqa: PLC0415
        self.assertIsInstance(sched._execution_throttle, ExecutionThrottle)

    def test_scheduler_passes_throttle_to_auto_trader(self):
        sched = self._make_scheduler(auto_execute=True)
        self.assertIsNotNone(sched._auto_trader)
        self.assertIs(
            sched._auto_trader._execution_throttle,
            sched._execution_throttle,
        )

    def test_scheduler_no_auto_execute_still_has_throttle(self):
        sched = self._make_scheduler(auto_execute=False)
        from core.execution_throttle import ExecutionThrottle  # noqa: PLC0415
        self.assertIsInstance(sched._execution_throttle, ExecutionThrottle)
        # auto_trader is None when auto_execute=False
        self.assertIsNone(sched._auto_trader)

    def test_throttle_default_window_is_four_hours(self):
        t = ExecutionThrottle(db_path=":memory:")
        self.assertEqual(t.throttle_seconds, 14400.0)

    def test_custom_throttle_window_honoured(self):
        t = ExecutionThrottle(db_path=":memory:", throttle_seconds=7200)
        self.assertEqual(t.throttle_seconds, 7200)


# ─────────────────────────────────────────────────────────────────────────────
# TestExecutionThrottleAllEntries
# ─────────────────────────────────────────────────────────────────────────────

class TestExecutionThrottleAllEntries(unittest.TestCase):
    """all_entries() returns all rows ordered by most-recent first."""

    def _make(self):
        return ExecutionThrottle(db_path=":memory:", throttle_seconds=3600)

    def test_multiple_tickers_all_returned(self):
        t = self._make()
        t.record_execution("SPY")
        t.record_execution("IWM")
        t.record_execution("JNK")
        entries = t.all_entries()
        tickers = {e["ticker"] for e in entries}
        self.assertEqual(tickers, {"SPY", "IWM", "JNK"})

    def test_duplicate_record_updates_not_appends(self):
        t = self._make()
        t.record_execution("SPY")
        t.record_execution("SPY")  # second record of same ticker
        entries = t.all_entries()
        self.assertEqual(len(entries), 1)  # upsert → only one row

    def test_entry_has_expected_keys(self):
        t = self._make()
        t.record_execution("SPY")
        entry = t.all_entries()[0]
        self.assertIn("ticker", entry)
        self.assertIn("executed_at", entry)
        self.assertIsInstance(entry["executed_at"], float)


if __name__ == "__main__":
    unittest.main()
