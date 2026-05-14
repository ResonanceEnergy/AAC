"""tests/test_order_monitor.py — Sprint 14: Stale Order Monitor tests.

Coverage:
    TestPendingOrder          (4)  — age_minutes, tz-naive, tz-aware, zero age
    TestPendingOrderRegistry  (7)  — add, duplicate ignored, remove, mark_filled, get_pending, size, empty
    TestOrderMonitorReport    (3)  — defaults, to_dict keys, to_dict values
    TestOrderMonitorScan      (7)  — empty, fresh not cancelled, stale cancelled, report counts,
                                     cancel returns False (error), exception per order, mixed orders
    TestOrderMonitorRegister  (4)  — register delegates, mark_filled delegates, no duplicate, missing order_id
    TestAutoTraderWiring      (5)  — monitor param stored, submitted registered, filled not registered,
                                     no monitor no crash, empty confirmations no crash
    TestMarketSchedulerWiring (4)  — run_order_monitor returns report, zero cancelled on empty registry,
                                     exception returns empty report, order_monitor created at init
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# TestPendingOrder
# ─────────────────────────────────────────────────────────────────────────────

class TestPendingOrder:
    def _make(self, minutes_ago: float = 0.0):
        from core.order_monitor import PendingOrder
        ts = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes_ago)
        return PendingOrder(order_id="ord-1", ticker="SPY", submitted_at=ts)

    def test_age_minutes_positive(self):
        po = self._make(minutes_ago=20)
        assert 19.0 <= po.age_minutes() <= 21.0

    def test_age_minutes_zero(self):
        po = self._make(minutes_ago=0)
        assert po.age_minutes() < 1.0

    def test_tz_naive_submitted_at(self):
        """tz-naive submitted_at is treated as UTC."""
        from core.order_monitor import PendingOrder
        naive_ts = datetime.utcnow() - timedelta(minutes=10)
        po = PendingOrder(order_id="x", ticker="IWM", submitted_at=naive_ts)
        assert 9.0 <= po.age_minutes() <= 11.0

    def test_age_minutes_with_injected_now(self):
        from core.order_monitor import PendingOrder
        ts = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        po = PendingOrder(order_id="y", ticker="QQQ", submitted_at=ts)
        now = datetime(2026, 1, 1, 10, 35, 0, tzinfo=timezone.utc)
        assert po.age_minutes(now) == pytest.approx(35.0)


# ─────────────────────────────────────────────────────────────────────────────
# TestPendingOrderRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestPendingOrderRegistry:
    def _reg(self):
        from core.order_monitor import PendingOrderRegistry
        return PendingOrderRegistry()

    def test_add_order(self):
        reg = self._reg()
        reg.add("ord-1", "SPY")
        assert reg.size == 1

    def test_duplicate_order_id_ignored(self):
        reg = self._reg()
        reg.add("ord-1", "SPY")
        reg.add("ord-1", "SPY")  # duplicate
        assert reg.size == 1

    def test_remove_order(self):
        reg = self._reg()
        reg.add("ord-1", "SPY")
        reg.remove("ord-1")
        assert reg.size == 0

    def test_remove_nonexistent_no_error(self):
        reg = self._reg()
        reg.remove("does-not-exist")  # must not raise
        assert reg.size == 0

    def test_mark_filled_alias(self):
        reg = self._reg()
        reg.add("ord-2", "QQQ")
        reg.mark_filled("ord-2")
        assert reg.size == 0

    def test_get_pending_returns_snapshot(self):
        reg = self._reg()
        reg.add("ord-1", "SPY")
        reg.add("ord-2", "IWM")
        pending = reg.get_pending()
        assert len(pending) == 2
        ids = {p.order_id for p in pending}
        assert ids == {"ord-1", "ord-2"}

    def test_empty_registry_size_zero(self):
        reg = self._reg()
        assert reg.size == 0
        assert reg.get_pending() == []


# ─────────────────────────────────────────────────────────────────────────────
# TestOrderMonitorReport
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderMonitorReport:
    def test_defaults(self):
        from core.order_monitor import OrderMonitorReport
        r = OrderMonitorReport()
        assert r.checked == 0
        assert r.cancelled == 0
        assert r.still_pending == 0
        assert r.errors == 0
        assert r.cancelled_ids == []

    def test_to_dict_has_keys(self):
        from core.order_monitor import OrderMonitorReport
        r = OrderMonitorReport(checked=2, cancelled=1, still_pending=1, errors=0,
                               cancelled_ids=["x"], generated_at="now")
        d = r.to_dict()
        for key in ("checked", "cancelled", "still_pending", "errors", "cancelled_ids", "generated_at"):
            assert key in d

    def test_to_dict_values(self):
        from core.order_monitor import OrderMonitorReport
        r = OrderMonitorReport(checked=3, cancelled=2, still_pending=1,
                               errors=1, cancelled_ids=["a", "b"], generated_at="2026-01-01")
        d = r.to_dict()
        assert d["checked"] == 3
        assert d["cancelled"] == 2
        assert d["cancelled_ids"] == ["a", "b"]


# ─────────────────────────────────────────────────────────────────────────────
# TestOrderMonitorScan
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderMonitorScan:
    def _monitor_and_reg(self, stale_minutes=30):
        from core.order_monitor import OrderMonitor, PendingOrderRegistry
        reg = PendingOrderRegistry()
        mon = OrderMonitor(reg, stale_minutes=stale_minutes)
        return mon, reg

    def _now(self):
        return datetime.now(tz=timezone.utc)

    def test_empty_registry_returns_empty_report(self):
        mon, _ = self._monitor_and_reg()
        report = mon.scan()
        assert report.checked == 0
        assert report.cancelled == 0
        assert report.still_pending == 0

    def test_fresh_order_not_cancelled(self):
        mon, reg = self._monitor_and_reg(stale_minutes=30)
        reg.add("ord-1", "SPY", submitted_at=self._now())
        with patch("core.order_monitor._cancel_via_ibkr") as mock_cancel:
            report = mon.scan()
        mock_cancel.assert_not_called()
        assert report.still_pending == 1
        assert report.cancelled == 0

    def test_stale_order_cancelled(self):
        mon, reg = self._monitor_and_reg(stale_minutes=30)
        old_ts = self._now() - timedelta(minutes=45)
        reg.add("ord-stale", "IWM", submitted_at=old_ts)
        with patch("core.order_monitor._cancel_via_ibkr", return_value=True):
            report = mon.scan()
        assert report.cancelled == 1
        assert "ord-stale" in report.cancelled_ids
        assert reg.size == 0  # removed from registry

    def test_cancel_returns_false_counts_as_error(self):
        mon, reg = self._monitor_and_reg(stale_minutes=10)
        old_ts = self._now() - timedelta(minutes=20)
        reg.add("ord-fail", "QQQ", submitted_at=old_ts)
        with patch("core.order_monitor._cancel_via_ibkr", return_value=False):
            report = mon.scan()
        assert report.cancelled == 0
        assert report.errors == 1
        # Order stays in registry
        assert reg.size == 1

    def test_exception_in_cancel_counts_as_error_does_not_raise(self):
        mon, reg = self._monitor_and_reg(stale_minutes=5)
        old_ts = self._now() - timedelta(minutes=10)
        reg.add("ord-exc", "HYG", submitted_at=old_ts)
        with patch("core.order_monitor._cancel_via_ibkr", side_effect=RuntimeError("IBKR down")):
            report = mon.scan()  # must not raise
        assert report.errors == 1
        assert report.cancelled == 0

    def test_mixed_fresh_and_stale(self):
        mon, reg = self._monitor_and_reg(stale_minutes=30)
        fresh = self._now()
        stale = self._now() - timedelta(minutes=60)
        reg.add("ord-fresh", "SPY", submitted_at=fresh)
        reg.add("ord-stale", "JNK", submitted_at=stale)
        with patch("core.order_monitor._cancel_via_ibkr", return_value=True):
            report = mon.scan()
        assert report.checked == 2
        assert report.cancelled == 1
        assert report.still_pending == 1
        assert "ord-stale" in report.cancelled_ids

    def test_report_generated_at_non_empty(self):
        mon, _ = self._monitor_and_reg()
        report = mon.scan()
        assert report.generated_at != ""


# ─────────────────────────────────────────────────────────────────────────────
# TestOrderMonitorRegister
# ─────────────────────────────────────────────────────────────────────────────

class TestOrderMonitorRegister:
    def _monitor(self):
        from core.order_monitor import OrderMonitor, PendingOrderRegistry
        reg = PendingOrderRegistry()
        return OrderMonitor(reg), reg

    def test_register_delegates_to_registry(self):
        mon, reg = self._monitor()
        mon.register("ord-1", "SPY")
        assert reg.size == 1

    def test_mark_filled_delegates_to_registry(self):
        mon, reg = self._monitor()
        mon.register("ord-1", "SPY")
        mon.mark_filled("ord-1")
        assert reg.size == 0

    def test_duplicate_registration_ignored(self):
        mon, reg = self._monitor()
        mon.register("ord-1", "SPY")
        mon.register("ord-1", "SPY")
        assert reg.size == 1

    def test_empty_order_id_not_added(self):
        mon, reg = self._monitor()
        mon.register("", "SPY")
        assert reg.size == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestAutoTraderWiring
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoTraderWiring:
    def _make_confirmation(self, status_value: str, order_id: str = "ord-1", ticker: str = "SPY"):
        conf = MagicMock()
        conf.status.value = status_value
        status_mock = MagicMock()
        status_mock.value = status_value
        # Make `conf.status is not ConfirmationStatus.SUBMITTED` test work
        conf.status = status_mock
        conf.order_id = order_id
        conf.signal_ticker = ticker
        conf.submitted_at = datetime.now(tz=timezone.utc).isoformat()
        return conf

    def test_order_monitor_stored(self):
        from core.auto_trader import AutoTrader
        from core.order_monitor import OrderMonitor, PendingOrderRegistry
        reg = PendingOrderRegistry()
        mon = OrderMonitor(reg)
        at = AutoTrader(dry_run=True, order_monitor=mon)
        assert at._order_monitor is mon

    def test_none_order_monitor_does_not_crash(self):
        from core.auto_trader import AutoTrader
        at = AutoTrader(dry_run=True, order_monitor=None)
        # _register_pending_with_monitor must not be called — but just instantiating is enough
        assert at._order_monitor is None

    def test_submitted_confirmation_registered(self):
        from core.auto_trader import AutoTrader
        from core.order_monitor import OrderMonitor, PendingOrderRegistry
        from TradingExecution.signal_executor import ConfirmationStatus

        reg = PendingOrderRegistry()
        mon = OrderMonitor(reg)
        at = AutoTrader(dry_run=True, order_monitor=mon)

        conf = MagicMock()
        conf.status = ConfirmationStatus.SUBMITTED
        conf.order_id = "ibkr-999"
        conf.signal_ticker = "SPY"
        conf.submitted_at = datetime.now(tz=timezone.utc).isoformat()

        at._register_pending_with_monitor([conf])
        assert reg.size == 1

    def test_filled_confirmation_not_registered(self):
        from core.auto_trader import AutoTrader
        from core.order_monitor import OrderMonitor, PendingOrderRegistry
        from TradingExecution.signal_executor import ConfirmationStatus

        reg = PendingOrderRegistry()
        mon = OrderMonitor(reg)
        at = AutoTrader(dry_run=True, order_monitor=mon)

        conf = MagicMock()
        conf.status = ConfirmationStatus.FILLED
        conf.order_id = "ibkr-888"
        conf.signal_ticker = "QQQ"
        conf.submitted_at = datetime.now(tz=timezone.utc).isoformat()

        at._register_pending_with_monitor([conf])
        assert reg.size == 0  # FILLED orders are not tracked

    def test_empty_confirmations_no_crash(self):
        from core.auto_trader import AutoTrader
        from core.order_monitor import OrderMonitor, PendingOrderRegistry
        reg = PendingOrderRegistry()
        at = AutoTrader(dry_run=True, order_monitor=OrderMonitor(reg))
        at._register_pending_with_monitor([])  # must not raise
        assert reg.size == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestMarketSchedulerWiring
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketSchedulerWiring:
    def _scheduler(self):
        from core.market_scheduler import MarketScheduler
        return MarketScheduler(auto_execute=False)

    def test_order_monitor_created_at_init(self):
        sched = self._scheduler()
        from core.order_monitor import OrderMonitor
        assert isinstance(sched._order_monitor, OrderMonitor)

    def test_run_order_monitor_returns_report(self):
        sched = self._scheduler()
        from core.order_monitor import OrderMonitorReport
        with patch("core.order_monitor._cancel_via_ibkr"):
            report = sched.run_order_monitor()
        assert isinstance(report, OrderMonitorReport)

    def test_run_order_monitor_empty_registry(self):
        sched = self._scheduler()
        report = sched.run_order_monitor()
        assert report.checked == 0
        assert report.cancelled == 0

    def test_run_order_monitor_exception_returns_empty_report(self):
        sched = self._scheduler()
        # Make scan raise
        sched._order_monitor.scan = MagicMock(side_effect=RuntimeError("boom"))
        from core.order_monitor import OrderMonitorReport
        report = sched.run_order_monitor()  # must not raise
        assert isinstance(report, OrderMonitorReport)
        assert report.checked == 0
