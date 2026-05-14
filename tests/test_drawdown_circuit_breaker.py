from __future__ import annotations

"""tests/test_drawdown_circuit_breaker.py — Sprint 18: Drawdown Circuit Breaker.

Tests for:
  * DrawdownState dataclass
  * DrawdownCircuitBreaker.update() state machine
  * Peak tracking (peak only moves up)
  * DB persistence across instances
  * is_tripped() read-only query
  * reset() clears trip flag
  * Fail-open behaviour
  * AutoTrader wiring (drawdown_circuit_breaker param)
  * MarketScheduler wiring (definition-site patches)
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from strategies.drawdown_circuit_breaker import DrawdownCircuitBreaker, DrawdownState

# ── Patch constants (definition-site) ─────────────────────────────────────────
_PATCH_POSITION_TRACKER = "TradingExecution.position_tracker.PositionTracker"
_PATCH_PNL_TRACKER = "CentralAccounting.pnl_tracker.PnLTracker"
_PATCH_LOSS_GUARD = "strategies.daily_loss_guard.DailyLossGuard"
_PATCH_ORDER_MONITOR = "core.order_monitor.OrderMonitor"
_PATCH_ORDER_REGISTRY = "core.order_monitor.PendingOrderRegistry"
_PATCH_SIGNAL_JOURNAL = "strategies.signal_journal.SignalJournal"
_PATCH_REGIME_MONITOR = "strategies.regime_monitor.RegimeMonitor"
_PATCH_RECONCILER = "core.position_reconciler.PositionReconciler"
_PATCH_DRAWDOWN_CB = "strategies.drawdown_circuit_breaker.DrawdownCircuitBreaker"


# ── DrawdownState tests ────────────────────────────────────────────────────────


class TestDrawdownState:
    def test_to_dict_not_tripped(self):
        state = DrawdownState(
            peak_value=100_000.0,
            current_value=95_000.0,
            drawdown_pct=0.05,
            tripped=False,
        )
        d = state.to_dict()
        assert d["peak_value"] == 100_000.0
        assert d["current_value"] == 95_000.0
        assert d["drawdown_pct"] == pytest.approx(0.05, abs=1e-6)
        assert d["tripped"] is False
        assert d["tripped_at"] is None

    def test_to_dict_tripped(self):
        state = DrawdownState(
            peak_value=100_000.0,
            current_value=88_000.0,
            drawdown_pct=0.12,
            tripped=True,
            tripped_at="2026-04-22T12:00:00",
        )
        d = state.to_dict()
        assert d["tripped"] is True
        assert d["tripped_at"] == "2026-04-22T12:00:00"

    def test_drawdown_pct_rounded_in_dict(self):
        state = DrawdownState(
            peak_value=100_000.0,
            current_value=90_001.0,
            drawdown_pct=0.09999,
            tripped=False,
        )
        d = state.to_dict()
        # Should be rounded to 6 decimal places
        assert len(str(d["drawdown_pct"]).split(".")[-1]) <= 6

    def test_tripped_at_defaults_to_none(self):
        state = DrawdownState(
            peak_value=50_000.0,
            current_value=50_000.0,
            drawdown_pct=0.0,
            tripped=False,
        )
        assert state.tripped_at is None


# ── update() — clean state (no trip) ─────────────────────────────────────────


class TestUpdateNoTrip:
    def test_first_update_sets_peak(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.10)
        state = breaker.update(100_000.0)
        assert state.peak_value == pytest.approx(100_000.0)
        assert state.current_value == pytest.approx(100_000.0)
        assert state.drawdown_pct == pytest.approx(0.0)
        assert state.tripped is False

    def test_drawdown_below_threshold_not_tripped(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.10)
        breaker.update(100_000.0)
        state = breaker.update(95_000.0)  # 5% drawdown — under 10% threshold
        assert state.drawdown_pct == pytest.approx(0.05, abs=1e-6)
        assert state.tripped is False

    def test_drawdown_exactly_at_threshold_trips(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.10)
        breaker.update(100_000.0)
        state = breaker.update(90_000.0)  # exactly 10%
        assert state.tripped is True

    def test_drawdown_above_threshold_trips(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.10)
        breaker.update(100_000.0)
        state = breaker.update(85_000.0)  # 15% drawdown
        assert state.drawdown_pct == pytest.approx(0.15, abs=1e-6)
        assert state.tripped is True
        assert state.tripped_at is not None


# ── Peak tracking — peak only moves up ────────────────────────────────────────


class TestPeakTracking:
    def test_peak_rises_on_new_high(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.20)
        breaker.update(100_000.0)
        state = breaker.update(120_000.0)  # new high
        assert state.peak_value == pytest.approx(120_000.0)

    def test_peak_does_not_fall_after_drawdown(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.20)
        breaker.update(100_000.0)
        breaker.update(120_000.0)  # new high
        state = breaker.update(90_000.0)  # drawdown — peak stays at 120k
        assert state.peak_value == pytest.approx(120_000.0)
        assert state.drawdown_pct == pytest.approx(0.25, abs=1e-6)

    def test_stays_tripped_once_tripped(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.10)
        breaker.update(100_000.0)
        breaker.update(85_000.0)  # trip
        state = breaker.update(99_000.0)  # recovered but NOT reset
        assert state.tripped is True  # stays tripped until explicit reset()


# ── DB persistence across instances ───────────────────────────────────────────


class TestPersistence:
    def test_state_persists_across_instances(self, tmp_path):
        db = str(tmp_path / "dd.db")
        b1 = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.10)
        b1.update(100_000.0)
        b1.update(85_000.0)  # trip

        b2 = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.10)
        assert b2.is_tripped() is True

    def test_peak_persists_across_instances(self, tmp_path):
        db = str(tmp_path / "dd.db")
        b1 = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.20)
        b1.update(100_000.0)
        b1.update(150_000.0)  # new peak

        b2 = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.20)
        state = b2.update(140_000.0)  # drawdown from 150k peak
        assert state.peak_value == pytest.approx(150_000.0)


# ── is_tripped() ──────────────────────────────────────────────────────────────


class TestIsTripped:
    def test_false_before_any_update(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db)
        assert breaker.is_tripped() is False

    def test_false_below_threshold(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.10)
        breaker.update(100_000.0)
        breaker.update(95_000.0)
        assert breaker.is_tripped() is False

    def test_true_after_threshold_exceeded(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.10)
        breaker.update(100_000.0)
        breaker.update(88_000.0)  # 12% drawdown
        assert breaker.is_tripped() is True


# ── reset() ───────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_trip_flag(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.10)
        breaker.update(100_000.0)
        breaker.update(85_000.0)  # trip
        assert breaker.is_tripped() is True
        breaker.reset()
        assert breaker.is_tripped() is False

    def test_reset_with_explicit_new_peak(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.10)
        breaker.update(100_000.0)
        breaker.update(85_000.0)  # trip
        breaker.reset(new_peak_value=90_000.0)
        # After reset, update with 82k — should be 8.9% from 90k peak (under 10%)
        state = breaker.update(82_000.0)
        assert state.tripped is False

    def test_reset_without_new_peak_uses_current_value(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db, max_drawdown_pct=0.10)
        breaker.update(100_000.0)
        breaker.update(85_000.0)  # trip; current=85k
        breaker.reset()  # peak becomes 85k
        # Update with 80k — should be 5.9% from 85k, not tripped
        state = breaker.update(80_000.0)
        assert state.tripped is False
        assert state.peak_value == pytest.approx(85_000.0)


# ── Fail-open behaviour ────────────────────────────────────────────────────────


class TestFailOpen:
    def test_update_with_zero_value_returns_safe_state(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db)
        state = breaker.update(0.0)
        assert state.tripped is False

    def test_update_with_negative_value_returns_safe_state(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db)
        state = breaker.update(-1000.0)
        assert state.tripped is False

    def test_is_tripped_returns_false_on_db_error(self, tmp_path):
        db = str(tmp_path / "nonexistent_dir" / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db)
        # Inject a bad connection
        bad_conn = MagicMock()
        bad_conn.execute.side_effect = sqlite3.OperationalError("simulated error")
        breaker._conn = bad_conn
        assert breaker.is_tripped() is False

    def test_update_returns_safe_state_on_db_error(self, tmp_path):
        db = str(tmp_path / "dd.db")
        breaker = DrawdownCircuitBreaker(db_path=db)
        bad_conn = MagicMock()
        bad_conn.execute.side_effect = sqlite3.OperationalError("simulated error")
        breaker._conn = bad_conn
        state = breaker.update(50_000.0)
        assert state.tripped is False


# ── AutoTrader wiring ─────────────────────────────────────────────────────────


class TestAutoTraderWiring:
    def test_auto_trader_accepts_drawdown_cb_param(self):
        from core.auto_trader import AutoTrader
        mock_cb = MagicMock()
        mock_cb.is_tripped.return_value = False
        at = AutoTrader(drawdown_circuit_breaker=mock_cb)
        assert at._drawdown_circuit_breaker is mock_cb

    def test_auto_trader_halts_when_tripped(self):
        from core.auto_trader import AutoTrader
        from shared.signal import Direction, TradeSignal

        mock_cb = MagicMock()
        mock_cb.is_tripped.return_value = True
        mock_cb.max_drawdown_pct = 0.10

        at = AutoTrader(
            drawdown_circuit_breaker=mock_cb,
            dry_run=True,
        )
        sig = TradeSignal(
            ticker="SPY",
            direction=Direction.LONG_PUT,
            confidence=0.90,
            entry=5.0,
            stop=2.5,
            target=10.0,
            size=0.05,
        )
        summary = at.run_once([sig])
        assert summary.signals_executed == 0
        assert summary.signals_filtered == 1
        assert any("DRAWDOWN" in r for r in summary.filter_reasons)

    def test_auto_trader_proceeds_when_not_tripped(self):
        from core.auto_trader import AutoTrader
        from shared.signal import Direction, TradeSignal

        mock_cb = MagicMock()
        mock_cb.is_tripped.return_value = False

        at = AutoTrader(
            drawdown_circuit_breaker=mock_cb,
            dry_run=True,
        )
        sig = TradeSignal(
            ticker="SPY",
            direction=Direction.LONG_PUT,
            confidence=0.90,
            entry=5.0,
            stop=2.5,
            target=10.0,
            size=0.05,
        )
        summary = at.run_once([sig])
        # dry_run means executed=0 but it was approved (not filtered by drawdown)
        assert summary.signals_approved >= 0  # may still be filtered by other gates
        assert not any("DRAWDOWN" in r for r in summary.filter_reasons)

    def test_auto_trader_drawdown_checked_before_daily_loss(self):
        """Drawdown guard short-circuits before daily loss guard is consulted."""
        from core.auto_trader import AutoTrader

        mock_cb = MagicMock()
        mock_cb.is_tripped.return_value = True
        mock_cb.max_drawdown_pct = 0.10

        mock_dlg = MagicMock()
        mock_dlg.is_limit_reached.return_value = (True, "daily limit")

        at = AutoTrader(
            drawdown_circuit_breaker=mock_cb,
            daily_loss_guard=mock_dlg,
            dry_run=True,
        )
        at.run_once([])
        mock_cb.is_tripped.assert_called_once()
        mock_dlg.is_limit_reached.assert_not_called()

    def test_cb_exception_does_not_halt_trading(self):
        """If is_tripped() raises, trading continues (fail-open)."""
        from core.auto_trader import AutoTrader

        mock_cb = MagicMock()
        mock_cb.is_tripped.side_effect = RuntimeError("DB unavailable")

        at = AutoTrader(drawdown_circuit_breaker=mock_cb, dry_run=True)
        summary = at.run_once([])
        assert not any("DRAWDOWN" in r for r in summary.filter_reasons)


# ── MarketScheduler wiring ────────────────────────────────────────────────────


class TestMarketSchedulerWiring:
    @pytest.fixture()
    def patched_scheduler(self):
        with (
            patch(_PATCH_POSITION_TRACKER) as mock_pt,
            patch(_PATCH_PNL_TRACKER) as mock_pnl,
            patch(_PATCH_LOSS_GUARD) as mock_dlg,
            patch(_PATCH_ORDER_REGISTRY) as mock_reg,
            patch(_PATCH_ORDER_MONITOR) as mock_om,
            patch(_PATCH_SIGNAL_JOURNAL) as mock_sj,
            patch(_PATCH_REGIME_MONITOR) as mock_rm,
            patch(_PATCH_RECONCILER) as mock_rec,
            patch(_PATCH_DRAWDOWN_CB) as mock_cb,
        ):
            mock_pt.return_value = MagicMock()
            mock_pnl.return_value = MagicMock()
            mock_dlg.return_value = MagicMock()
            mock_reg.return_value = MagicMock()
            mock_om.return_value = MagicMock()
            mock_sj.return_value = MagicMock()
            mock_rm.return_value = MagicMock()
            mock_rec.return_value = MagicMock()
            mock_cb.return_value = MagicMock()

            from core.market_scheduler import MarketScheduler
            sched = MarketScheduler()
            yield sched, mock_cb

    def test_drawdown_cb_attribute_exists(self, patched_scheduler):
        sched, _ = patched_scheduler
        assert hasattr(sched, "_drawdown_circuit_breaker")

    def test_drawdown_cb_created_once_at_init(self, patched_scheduler):
        _, mock_cb_cls = patched_scheduler
        mock_cb_cls.assert_called_once()

    def test_drawdown_cb_passed_to_auto_trader(self):
        """When auto_execute=True, DrawdownCircuitBreaker is passed to AutoTrader."""
        with (
            patch(_PATCH_POSITION_TRACKER) as mock_pt,
            patch(_PATCH_PNL_TRACKER) as mock_pnl,
            patch(_PATCH_LOSS_GUARD) as mock_dlg,
            patch(_PATCH_ORDER_REGISTRY) as mock_reg,
            patch(_PATCH_ORDER_MONITOR) as mock_om,
            patch(_PATCH_SIGNAL_JOURNAL) as mock_sj,
            patch(_PATCH_REGIME_MONITOR) as mock_rm,
            patch(_PATCH_RECONCILER) as mock_rec,
            patch(_PATCH_DRAWDOWN_CB) as mock_cb_cls,
            patch("core.auto_trader.AutoTrader") as mock_at_cls,
        ):
            mock_pt.return_value = MagicMock()
            mock_pnl.return_value = MagicMock()
            mock_dlg.return_value = MagicMock()
            mock_reg.return_value = MagicMock()
            mock_om.return_value = MagicMock()
            mock_sj.return_value = MagicMock()
            mock_rm.return_value = MagicMock()
            mock_rec.return_value = MagicMock()
            mock_cb_inst = MagicMock()
            mock_cb_cls.return_value = mock_cb_inst
            mock_at_cls.return_value = MagicMock()

            from core.market_scheduler import MarketScheduler
            MarketScheduler(auto_execute=True)

            call_kwargs = mock_at_cls.call_args.kwargs
            assert "drawdown_circuit_breaker" in call_kwargs
            assert call_kwargs["drawdown_circuit_breaker"] is mock_cb_inst
