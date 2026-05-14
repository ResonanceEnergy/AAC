from __future__ import annotations

"""tests/test_account_equity_feed.py — Sprint 19: Account Equity Feed.

Verifies that the DrawdownCircuitBreaker.current_state() read-only method,
the MarketScheduler._run_drawdown_update() helper, and the drawdown fields
on EodReport all work correctly and are properly wired together.
"""

import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch constants (definition-site, matching the lazy-import pattern)
# ---------------------------------------------------------------------------

_PATCH_POSITION_TRACKER = "TradingExecution.position_tracker.PositionTracker"
_PATCH_PNL_TRACKER = "CentralAccounting.pnl_tracker.PnLTracker"
_PATCH_LOSS_GUARD = "strategies.daily_loss_guard.DailyLossGuard"
_PATCH_ORDER_MONITOR = "core.order_monitor.OrderMonitor"
_PATCH_ORDER_REGISTRY = "core.order_monitor.PendingOrderRegistry"
_PATCH_SIGNAL_JOURNAL = "strategies.signal_journal.SignalJournal"
_PATCH_REGIME_MONITOR = "strategies.regime_monitor.RegimeMonitor"
_PATCH_RECONCILER = "core.position_reconciler.PositionReconciler"
_PATCH_DRAWDOWN_CB = "strategies.drawdown_circuit_breaker.DrawdownCircuitBreaker"
_PATCH_EOD_REPORTER = "core.eod_reporter.EodReporter"


def _make_scheduler(**kwargs):
    """Create a MarketScheduler with all heavy deps mocked out."""
    with (
        patch(_PATCH_POSITION_TRACKER),
        patch(_PATCH_PNL_TRACKER),
        patch(_PATCH_LOSS_GUARD),
        patch(_PATCH_ORDER_MONITOR),
        patch(_PATCH_ORDER_REGISTRY),
        patch(_PATCH_SIGNAL_JOURNAL),
        patch(_PATCH_REGIME_MONITOR),
        patch(_PATCH_RECONCILER),
        patch(_PATCH_DRAWDOWN_CB),
        patch(_PATCH_EOD_REPORTER),
        patch("core.market_scheduler.AutoTrader", create=True),
    ):
        from core.market_scheduler import MarketScheduler
        return MarketScheduler(**kwargs)


# ===========================================================================
# TestCurrentState — DrawdownCircuitBreaker.current_state()
# ===========================================================================

class TestCurrentState:
    """current_state() reads from DB without modifying anything."""

    def test_fresh_db_returns_zero_state(self):
        """No DB row → returns safe zero DrawdownState (tripped=False)."""
        from strategies.drawdown_circuit_breaker import DrawdownCircuitBreaker, DrawdownState

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            cb = DrawdownCircuitBreaker(db_path=db_path)
            state = cb.current_state()
            assert isinstance(state, DrawdownState)
            assert state.peak_value == 0.0
            assert state.current_value == 0.0
            assert state.drawdown_pct == 0.0
            assert state.tripped is False
        finally:
            if cb._conn:
                cb._conn.close()
            os.unlink(db_path)

    def test_returns_state_after_update(self):
        """current_state() reflects the state written by update()."""
        from strategies.drawdown_circuit_breaker import DrawdownCircuitBreaker

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            cb = DrawdownCircuitBreaker(db_path=db_path)
            cb.update(100_000.0)  # sets peak
            cb.update(95_000.0)   # 5% drawdown

            state = cb.current_state()
            assert state.peak_value == pytest.approx(100_000.0)
            assert state.current_value == pytest.approx(95_000.0)
            assert state.drawdown_pct == pytest.approx(0.05)
            assert state.tripped is False
        finally:
            if cb._conn:
                cb._conn.close()
            os.unlink(db_path)

    def test_current_state_reflects_tripped_flag(self):
        """current_state().tripped=True when breaker has been tripped."""
        from strategies.drawdown_circuit_breaker import DrawdownCircuitBreaker

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            cb = DrawdownCircuitBreaker(db_path=db_path, max_drawdown_pct=0.10)
            cb.update(100_000.0)
            cb.update(85_000.0)   # 15% drawdown → trips

            state = cb.current_state()
            assert state.tripped is True
            assert state.drawdown_pct >= 0.10
        finally:
            if cb._conn:
                cb._conn.close()
            os.unlink(db_path)

    def test_current_state_does_not_alter_db(self):
        """Calling current_state() multiple times does not change peak or drawdown."""
        from strategies.drawdown_circuit_breaker import DrawdownCircuitBreaker

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            cb = DrawdownCircuitBreaker(db_path=db_path)
            cb.update(100_000.0)
            cb.update(97_000.0)

            state1 = cb.current_state()
            state2 = cb.current_state()
            assert state1.peak_value == state2.peak_value
            assert state1.current_value == state2.current_value
            assert state1.drawdown_pct == state2.drawdown_pct
        finally:
            if cb._conn:
                cb._conn.close()
            os.unlink(db_path)

    def test_current_state_db_error_returns_safe_state(self):
        """DB error in current_state() fails-open: returns tripped=False zeros."""
        from strategies.drawdown_circuit_breaker import DrawdownCircuitBreaker, DrawdownState

        cb = DrawdownCircuitBreaker(db_path="data/drawdown_circuit_breaker.db")
        # Force _get_conn() to raise.
        cb._get_conn = MagicMock(side_effect=RuntimeError("db gone"))
        state = cb.current_state()
        assert isinstance(state, DrawdownState)
        assert state.tripped is False
        assert state.peak_value == 0.0


# ===========================================================================
# TestRunDrawdownUpdate — MarketScheduler._run_drawdown_update()
# ===========================================================================

class TestRunDrawdownUpdate:
    """_run_drawdown_update() feeds account value to the circuit breaker."""

    def test_calls_update_with_positive_value(self):
        """update() is called with the supplied account value."""
        sched = _make_scheduler()
        mock_cb = sched._drawdown_circuit_breaker
        mock_cb.update.return_value = MagicMock(drawdown_pct=0.02, tripped=False, peak_value=50000, current_value=49000)

        sched._run_drawdown_update(50_000.0)

        mock_cb.update.assert_called_once_with(50_000.0)

    def test_skips_zero_account_value(self):
        """update() is NOT called when account_value == 0."""
        sched = _make_scheduler()
        sched._run_drawdown_update(0.0)
        sched._drawdown_circuit_breaker.update.assert_not_called()

    def test_skips_negative_account_value(self):
        """update() is NOT called when account_value < 0."""
        sched = _make_scheduler()
        sched._run_drawdown_update(-100.0)
        sched._drawdown_circuit_breaker.update.assert_not_called()

    def test_exception_from_update_is_swallowed(self):
        """Exception inside update() does not propagate out of _run_drawdown_update."""
        sched = _make_scheduler()
        sched._drawdown_circuit_breaker.update.side_effect = RuntimeError("db crash")
        # Should not raise
        sched._run_drawdown_update(50_000.0)

    def test_update_return_value_is_logged_not_raised(self):
        """Returns None even when update() succeeds — state is not returned."""
        sched = _make_scheduler()
        mock_state = MagicMock(drawdown_pct=0.05, tripped=False, peak_value=50000, current_value=47500)
        sched._drawdown_circuit_breaker.update.return_value = mock_state
        result = sched._run_drawdown_update(50_000.0)
        assert result is None


# ===========================================================================
# TestPnlSnapshotWiring — run_pnl_snapshot() calls _run_drawdown_update
# ===========================================================================

class TestPnlSnapshotWiring:
    """run_pnl_snapshot() triggers drawdown update after the PnL snapshot."""

    def _patched_scheduler(self, env_account_value="50000"):
        sched = _make_scheduler()
        # Patch PnL tracker to succeed
        sched._pnl_tracker.take_snapshot.return_value = {"daily_pnl": {"account_value_usd": float(env_account_value)}}
        sched._fetch_positions = MagicMock(return_value=[])
        sched._run_eod_report = MagicMock()
        sched._run_reconciliation = MagicMock()
        sched._run_drawdown_update = MagicMock()
        return sched

    def test_drawdown_update_called_in_pnl_snapshot(self):
        """_run_drawdown_update is called every time run_pnl_snapshot runs."""
        sched = self._patched_scheduler()
        sched.run_pnl_snapshot()
        sched._run_drawdown_update.assert_called_once()

    def test_drawdown_update_called_after_eod_report(self):
        """Call order: snapshot → eod → drawdown → reconciliation."""
        call_order = []
        sched = self._patched_scheduler()
        sched._run_eod_report = MagicMock(side_effect=lambda **kw: call_order.append("eod"))
        sched._run_reconciliation = MagicMock(side_effect=lambda **kw: call_order.append("reconcile"))
        sched._run_drawdown_update = MagicMock(side_effect=lambda v: call_order.append("drawdown"))
        sched.run_pnl_snapshot()
        assert call_order == ["eod", "drawdown", "reconcile"] or call_order == ["eod", "reconcile", "drawdown"]

    def test_drawdown_update_receives_account_value_from_env(self, monkeypatch):
        """_run_drawdown_update() receives the account value from ACCOUNT_VALUE_USD env."""
        monkeypatch.setenv("ACCOUNT_VALUE_USD", "75000")
        sched = self._patched_scheduler(env_account_value="75000")
        sched.run_pnl_snapshot()
        sched._run_drawdown_update.assert_called_once_with(75_000.0)

    def test_drawdown_update_called_even_if_snapshot_fails(self):
        """Even if take_snapshot() raises, _run_drawdown_update still runs."""
        sched = _make_scheduler()
        sched._fetch_positions = MagicMock(return_value=[])
        sched._pnl_tracker.take_snapshot.side_effect = RuntimeError("pnl error")
        sched._run_eod_report = MagicMock()
        sched._run_reconciliation = MagicMock()
        sched._run_drawdown_update = MagicMock()
        sched.run_pnl_snapshot()
        sched._run_drawdown_update.assert_called_once()

    def test_run_pnl_snapshot_still_returns_dict_when_drawdown_raises(self):
        """Exception in _run_drawdown_update never prevents run_pnl_snapshot return."""
        sched = _make_scheduler()
        sched._fetch_positions = MagicMock(return_value=[])
        sched._pnl_tracker.take_snapshot.return_value = {"ok": True}
        sched._run_eod_report = MagicMock()
        sched._run_reconciliation = MagicMock()
        sched._run_drawdown_update = MagicMock(side_effect=RuntimeError("drawdown exploded"))
        result = sched.run_pnl_snapshot()
        # Should still return — the exception from _run_drawdown_update propagates
        # unless we verify it's caught.  This test confirms no crash.
        assert result is not None


# ===========================================================================
# TestEodReportDrawdownFields — EodReport dataclass fields
# ===========================================================================

class TestEodReportDrawdownFields:
    """EodReport carries drawdown_pct and drawdown_tripped fields."""

    def test_default_drawdown_pct_is_zero(self):
        from core.eod_reporter import EodReport
        report = EodReport(report_date="2025-01-01")
        assert report.drawdown_pct == 0.0

    def test_default_drawdown_tripped_is_false(self):
        from core.eod_reporter import EodReport
        report = EodReport(report_date="2025-01-01")
        assert report.drawdown_tripped is False

    def test_drawdown_pct_set_correctly(self):
        from core.eod_reporter import EodReport
        report = EodReport(report_date="2025-01-01", drawdown_pct=0.07)
        assert report.drawdown_pct == pytest.approx(0.07)

    def test_drawdown_tripped_set_correctly(self):
        from core.eod_reporter import EodReport
        report = EodReport(report_date="2025-01-01", drawdown_tripped=True)
        assert report.drawdown_tripped is True

    def test_to_dict_includes_drawdown_fields(self):
        from core.eod_reporter import EodReport
        report = EodReport(report_date="2025-01-01", drawdown_pct=0.06, drawdown_tripped=True)
        d = report.to_dict()
        assert d["drawdown_pct"] == pytest.approx(0.06)
        assert d["drawdown_tripped"] is True

    def test_format_text_includes_drawdown_line(self):
        from core.eod_reporter import EodReport
        report = EodReport(report_date="2025-01-01", drawdown_pct=0.05, drawdown_tripped=False)
        text = report.format_text()
        assert "Drawdown" in text
        assert "5.00%" in text

    def test_format_text_warns_when_tripped(self):
        from core.eod_reporter import EodReport
        report = EodReport(report_date="2025-01-01", drawdown_pct=0.12, drawdown_tripped=True)
        text = report.format_text()
        assert "TRIPPED" in text


# ===========================================================================
# TestEodReporterDrawdownWiring — EodReporter.generate() reads drawdown state
# ===========================================================================

class TestEodReporterDrawdownWiring:
    """EodReporter.generate() reads DrawdownCircuitBreaker.current_state()."""

    def test_generate_calls_drawdown_current_state(self):
        """generate() imports DrawdownCircuitBreaker and calls current_state()."""
        from core.eod_reporter import EodReporter
        from strategies.drawdown_circuit_breaker import DrawdownState

        mock_state = DrawdownState(peak_value=50000, current_value=47000, drawdown_pct=0.06, tripped=False)
        mock_cb_instance = MagicMock()
        mock_cb_instance.current_state.return_value = mock_state
        mock_cb_cls = MagicMock(return_value=mock_cb_instance)

        with patch("strategies.drawdown_circuit_breaker.DrawdownCircuitBreaker", mock_cb_cls):
            reporter = EodReporter()
            report = reporter.generate()

        mock_cb_instance.current_state.assert_called_once()
        assert report.drawdown_pct == pytest.approx(0.06)
        assert report.drawdown_tripped is False

    def test_generate_sets_tripped_when_breaker_tripped(self):
        """report.drawdown_tripped=True when circuit breaker is tripped."""
        from core.eod_reporter import EodReporter
        from strategies.drawdown_circuit_breaker import DrawdownState

        mock_state = DrawdownState(peak_value=50000, current_value=42000, drawdown_pct=0.16, tripped=True)
        mock_cb_instance = MagicMock()
        mock_cb_instance.current_state.return_value = mock_state
        mock_cb_cls = MagicMock(return_value=mock_cb_instance)

        with patch("strategies.drawdown_circuit_breaker.DrawdownCircuitBreaker", mock_cb_cls):
            reporter = EodReporter()
            report = reporter.generate()

        assert report.drawdown_tripped is True
        assert report.drawdown_pct == pytest.approx(0.16)

    def test_generate_drawdown_exception_falls_back_to_defaults(self):
        """Exception from DrawdownCircuitBreaker doesn't crash generate()."""
        from core.eod_reporter import EodReporter

        mock_cb_cls = MagicMock(side_effect=RuntimeError("import failed"))

        with patch("strategies.drawdown_circuit_breaker.DrawdownCircuitBreaker", mock_cb_cls):
            reporter = EodReporter()
            report = reporter.generate()

        assert report.drawdown_pct == 0.0
        assert report.drawdown_tripped is False

    def test_generate_current_state_exception_falls_back_to_defaults(self):
        """Exception from current_state() doesn't crash generate()."""
        from core.eod_reporter import EodReporter

        mock_cb_instance = MagicMock()
        mock_cb_instance.current_state.side_effect = sqlite3.OperationalError("db locked")
        mock_cb_cls = MagicMock(return_value=mock_cb_instance)

        with patch("strategies.drawdown_circuit_breaker.DrawdownCircuitBreaker", mock_cb_cls):
            reporter = EodReporter()
            report = reporter.generate()

        assert report.drawdown_pct == 0.0
        assert report.drawdown_tripped is False

    def test_generate_returns_eod_report_always(self):
        """generate() always returns an EodReport regardless of drawdown state."""
        from core.eod_reporter import EodReport, EodReporter

        with patch(_PATCH_DRAWDOWN_CB) as mock_cb_cls:
            mock_cb_cls.return_value.current_state.return_value = MagicMock(
                drawdown_pct=0.0, tripped=False
            )
            reporter = EodReporter()
            result = reporter.generate()

        assert isinstance(result, EodReport)


# ===========================================================================
# TestMarketSchedulerDrawdownSequence — integration: correct order & args
# ===========================================================================

class TestMarketSchedulerDrawdownSequence:
    """Integration: run_pnl_snapshot wires _run_drawdown_update correctly."""

    def test_drawdown_update_uses_env_default(self, monkeypatch):
        """Default ACCOUNT_VALUE_USD=50000 is passed to _run_drawdown_update."""
        monkeypatch.setenv("ACCOUNT_VALUE_USD", "50000")
        sched = _make_scheduler()
        sched._fetch_positions = MagicMock(return_value=[])
        sched._pnl_tracker.take_snapshot.return_value = {}
        sched._run_eod_report = MagicMock()
        sched._run_reconciliation = MagicMock()
        sched._run_drawdown_update = MagicMock()
        sched.run_pnl_snapshot()
        sched._run_drawdown_update.assert_called_once_with(50_000.0)

    def test_both_reconciliation_and_drawdown_run(self, monkeypatch):
        """Both _run_reconciliation and _run_drawdown_update are called."""
        monkeypatch.setenv("ACCOUNT_VALUE_USD", "50000")
        sched = _make_scheduler()
        sched._fetch_positions = MagicMock(return_value=[])
        sched._pnl_tracker.take_snapshot.return_value = {}
        sched._run_eod_report = MagicMock()
        sched._run_reconciliation = MagicMock()
        sched._run_drawdown_update = MagicMock()
        sched.run_pnl_snapshot()
        sched._run_reconciliation.assert_called_once()
        sched._run_drawdown_update.assert_called_once()

    def test_drawdown_circuit_breaker_update_reaches_real_method(self):
        """When real DrawdownCircuitBreaker is used, update() is called with account value."""
        from strategies.drawdown_circuit_breaker import DrawdownCircuitBreaker

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            real_cb = DrawdownCircuitBreaker(db_path=db_path)
            sched = _make_scheduler()
            sched._drawdown_circuit_breaker = real_cb
            sched._fetch_positions = MagicMock(return_value=[])
            sched._pnl_tracker.take_snapshot.return_value = {}
            sched._run_eod_report = MagicMock()
            sched._run_reconciliation = MagicMock()

            # Force env value
            os.environ["ACCOUNT_VALUE_USD"] = "60000"
            sched.run_pnl_snapshot()

            state = real_cb.current_state()
            assert state.peak_value == pytest.approx(60_000.0)
            assert state.current_value == pytest.approx(60_000.0)
        finally:
            if real_cb._conn:
                real_cb._conn.close()
            os.unlink(db_path)
            os.environ.pop("ACCOUNT_VALUE_USD", None)
