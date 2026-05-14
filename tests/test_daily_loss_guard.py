"""tests/test_daily_loss_guard.py — Sprint 10 tests.

Covers:
  * DailyLossGuard — is_limit_reached, today_loss_pct, fail-open behaviour
  * AutoTrader — daily loss guard trips the cycle; trade journal auto-logging
  * MarketScheduler — guard + pnl_tracker wired at construction;
                      run_pnl_snapshot uses shared tracker
"""
from __future__ import annotations

import types
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_tracker_with_pnl(pnl_value: float, today_str: str | None = None):
    """Return a mock PnLTracker whose today_report() returns a daily_pnl row."""
    today = today_str or date.today().isoformat()
    tracker = MagicMock()
    tracker.today_report.return_value = {
        "date": today,
        "daily_pnl": {
            "snapshot_date": today,
            "total_unrealized_pnl": pnl_value,
            "total_realized_pnl": 0.0,
            "account_value_usd": 50_000.0,
        },
        "positions": [],
        "today_trades": [],
    }
    return tracker


def _make_tracker_no_snapshot():
    """Return a mock PnLTracker with no snapshot for today."""
    tracker = MagicMock()
    tracker.today_report.return_value = {
        "date": date.today().isoformat(),
        "daily_pnl": None,
        "positions": [],
        "today_trades": [],
    }
    return tracker


# ── DailyLossGuard ────────────────────────────────────────────────────────────


class TestDailyLossGuardInit:
    def test_default_max_loss_pct(self):
        from strategies.daily_loss_guard import DailyLossGuard
        g = DailyLossGuard(max_loss_pct=0.05, account_value_usd=50_000)
        assert g.max_loss_pct == 0.05
        assert g.account_value_usd == 50_000

    def test_env_fallback_max_loss_pct(self, monkeypatch):
        monkeypatch.setenv("MAX_DAILY_LOSS_PCT", "0.03")
        monkeypatch.setenv("ACCOUNT_VALUE_USD", "80000")
        from strategies.daily_loss_guard import DailyLossGuard
        g = DailyLossGuard(max_loss_pct=0.0, account_value_usd=0.0)
        assert g.max_loss_pct == pytest.approx(0.03)
        assert g.account_value_usd == pytest.approx(80_000.0)

    def test_db_path_stored(self):
        from strategies.daily_loss_guard import DailyLossGuard
        g = DailyLossGuard(db_path="/tmp/test.db")
        assert g._db_path == "/tmp/test.db"


class TestDailyLossGuardIsLimitReached:
    def _guard(self, pnl_value: float, max_loss_pct: float = 0.05):
        from strategies.daily_loss_guard import DailyLossGuard
        g = DailyLossGuard(max_loss_pct=max_loss_pct, account_value_usd=50_000)
        g._pnl_tracker = _make_tracker_with_pnl(pnl_value)
        return g

    def test_no_loss_returns_false(self):
        g = self._guard(pnl_value=500.0)   # positive P&L
        tripped, reason = g.is_limit_reached()
        assert tripped is False
        assert reason == ""

    def test_zero_pnl_returns_false(self):
        g = self._guard(pnl_value=0.0)
        tripped, reason = g.is_limit_reached()
        assert tripped is False

    def test_loss_below_ceiling_returns_false(self):
        # 5% ceiling on $50k = $2500; loss of $2400 < ceiling
        g = self._guard(pnl_value=-2_400.0)
        tripped, _ = g.is_limit_reached()
        assert tripped is False

    def test_loss_exactly_at_ceiling_returns_true(self):
        # Exactly $2500 loss on 5% ceiling
        g = self._guard(pnl_value=-2_500.0)
        tripped, reason = g.is_limit_reached()
        assert tripped is True
        assert "daily loss" in reason
        assert "5.0%" in reason

    def test_loss_above_ceiling_returns_true(self):
        g = self._guard(pnl_value=-3_000.0)
        tripped, reason = g.is_limit_reached()
        assert tripped is True
        assert "$3,000" in reason or "3000" in reason

    def test_account_value_override(self):
        from strategies.daily_loss_guard import DailyLossGuard
        # Guard has $50k default but we override with $100k → ceiling doubles
        g = DailyLossGuard(max_loss_pct=0.05, account_value_usd=50_000)
        g._pnl_tracker = _make_tracker_with_pnl(-3_000.0)
        # $3k loss on $50k account = 6% → tripped
        tripped_small, _ = g.is_limit_reached()
        assert tripped_small is True
        # Same loss on $100k account = 3% → not tripped
        tripped_large, _ = g.is_limit_reached(account_value_usd=100_000)
        assert tripped_large is False

    def test_no_snapshot_today_returns_false(self):
        from strategies.daily_loss_guard import DailyLossGuard
        g = DailyLossGuard(max_loss_pct=0.05, account_value_usd=50_000)
        g._pnl_tracker = _make_tracker_no_snapshot()
        tripped, _ = g.is_limit_reached()
        assert tripped is False   # no data = fail-open

    def test_no_account_value_returns_false(self):
        from strategies.daily_loss_guard import DailyLossGuard
        g = DailyLossGuard(max_loss_pct=0.05, account_value_usd=0.0)
        # Even with bad pnl, no account value → can't compute ceiling → fail-open
        with patch.dict("os.environ", {"ACCOUNT_VALUE_USD": "0"}):
            tripped, _ = g.is_limit_reached(account_value_usd=0.0)
        assert tripped is False

    def test_db_error_fails_open(self):
        from strategies.daily_loss_guard import DailyLossGuard
        g = DailyLossGuard(max_loss_pct=0.05, account_value_usd=50_000)
        bad_tracker = MagicMock()
        bad_tracker.today_report.side_effect = RuntimeError("DB locked")
        g._pnl_tracker = bad_tracker
        # Must not raise, must fail-open
        tripped, _ = g.is_limit_reached()
        assert tripped is False

    def test_as_of_date_passed_to_tracker(self):
        from strategies.daily_loss_guard import DailyLossGuard
        target_date = date(2026, 1, 15)
        g = DailyLossGuard(max_loss_pct=0.05, account_value_usd=50_000)
        mock_tracker = _make_tracker_with_pnl(0.0, today_str=target_date.isoformat())
        g._pnl_tracker = mock_tracker
        g.is_limit_reached(as_of=target_date)
        mock_tracker.today_report.assert_called_once_with(snapshot_date=target_date.isoformat())

    def test_reason_contains_ceiling_and_account(self):
        g = self._guard(pnl_value=-2_600.0)   # $50k account, $2500 ceiling
        tripped, reason = g.is_limit_reached()
        assert tripped is True
        assert "$2,500" in reason   # ceiling
        assert "$50,000" in reason  # account


class TestTodayLossPct:
    def test_profit_returns_positive(self):
        from strategies.daily_loss_guard import DailyLossGuard
        g = DailyLossGuard(max_loss_pct=0.05, account_value_usd=50_000)
        g._pnl_tracker = _make_tracker_with_pnl(1_000.0)
        pct = g.today_loss_pct()
        assert pct > 0

    def test_loss_returns_negative(self):
        from strategies.daily_loss_guard import DailyLossGuard
        g = DailyLossGuard(max_loss_pct=0.05, account_value_usd=50_000)
        g._pnl_tracker = _make_tracker_with_pnl(-2_500.0)
        pct = g.today_loss_pct()
        assert pct == pytest.approx(-0.05)

    def test_no_snapshot_returns_zero(self):
        from strategies.daily_loss_guard import DailyLossGuard
        g = DailyLossGuard(max_loss_pct=0.05, account_value_usd=50_000)
        g._pnl_tracker = _make_tracker_no_snapshot()
        assert g.today_loss_pct() == 0.0

    def test_no_account_value_returns_zero(self):
        from strategies.daily_loss_guard import DailyLossGuard
        g = DailyLossGuard(max_loss_pct=0.05, account_value_usd=0.0)
        with patch.dict("os.environ", {"ACCOUNT_VALUE_USD": "0"}):
            assert g.today_loss_pct() == 0.0


# ── AutoTrader — daily loss guard integration ─────────────────────────────────


def _make_signal(ticker: str = "SPY", confidence: float = 0.80, size: float = 0.05):
    """Build a minimal TradeSignal-like object."""
    from shared.signal import Direction, TradeSignal
    return TradeSignal(
        ticker=ticker,
        direction=Direction.SHORT,
        confidence=confidence,
        entry=100.0,
        stop=110.0,
        target=80.0,
        size=size,
        strategy="test",
    )


def _tripped_guard(account_value: float = 50_000):
    """Return a DailyLossGuard-like mock that always trips."""
    g = MagicMock()
    g.is_limit_reached.return_value = (True, "daily loss -$2,600 exceeds 5.0% ceiling ($2,500)")
    return g


def _clear_guard():
    """Return a DailyLossGuard-like mock that never trips."""
    g = MagicMock()
    g.is_limit_reached.return_value = (False, "")
    return g


class TestAutoTraderDailyLossGuard:
    def test_guard_trips_blocks_all_signals(self):
        from core.auto_trader import AutoTrader
        signals = [_make_signal("SPY"), _make_signal("IWM")]
        trader = AutoTrader(
            dry_run=True,
            account_value_usd=50_000,
            daily_loss_guard=_tripped_guard(),
        )
        summary = trader.run_once(signals)
        assert summary.signals_received == 2
        assert summary.signals_filtered == 2
        assert summary.signals_approved == 0
        assert summary.signals_executed == 0

    def test_guard_trips_reason_in_filter_reasons(self):
        from core.auto_trader import AutoTrader
        trader = AutoTrader(
            dry_run=True,
            account_value_usd=50_000,
            daily_loss_guard=_tripped_guard(),
        )
        summary = trader.run_once([_make_signal()])
        assert any("DAILY_LOSS_GUARD" in r for r in summary.filter_reasons)

    def test_guard_clear_allows_normal_flow(self):
        from core.auto_trader import AutoTrader
        trader = AutoTrader(
            dry_run=True,   # dry run so we don't hit IBKR
            account_value_usd=50_000,
            daily_loss_guard=_clear_guard(),
        )
        signals = [_make_signal("SPY")]
        summary = trader.run_once(signals)
        # dry-run with clear guard: signals are approved, just not executed
        assert summary.signals_approved >= 1
        assert summary.signals_executed == 0  # dry_run

    def test_no_guard_executes_normally(self):
        """AutoTrader without guard still works (guard=None)."""
        from core.auto_trader import AutoTrader
        trader = AutoTrader(dry_run=True, account_value_usd=50_000)
        summary = trader.run_once([_make_signal()])
        assert summary.signals_received == 1
        # No guard → flow proceeds normally (dry_run stops actual exchange call)
        assert summary.signals_executed == 0

    def test_guard_trips_returns_summary_with_correct_received(self):
        from core.auto_trader import AutoTrader
        n_signals = 5
        signals = [_make_signal(f"S{i}") for i in range(n_signals)]
        trader = AutoTrader(
            dry_run=True,
            account_value_usd=50_000,
            daily_loss_guard=_tripped_guard(),
        )
        summary = trader.run_once(signals)
        assert summary.signals_received == n_signals

    def test_guard_account_value_passed_to_is_limit_reached(self):
        from core.auto_trader import AutoTrader
        guard = _clear_guard()
        trader = AutoTrader(
            dry_run=True,
            account_value_usd=75_000,
            daily_loss_guard=guard,
        )
        trader.run_once([_make_signal()])
        guard.is_limit_reached.assert_called_once_with(account_value_usd=75_000)


# ── AutoTrader — trade journal auto-logging ───────────────────────────────────


def _make_confirmation(ticker: str, status: str = "filled"):
    """Build a minimal OrderConfirmation-like namespace."""
    conf = types.SimpleNamespace()
    conf.signal_ticker = ticker
    conf.status = types.SimpleNamespace(value=status)
    conf.filled_quantity = 1
    conf.avg_fill_price = 99.5
    conf.order_id = f"ORD-{ticker}"

    def to_dict():
        return {
            "signal_ticker": ticker,
            "status": status,
            "filled_quantity": 1,
            "avg_fill_price": 99.5,
            "order_id": conf.order_id,
        }

    conf.to_dict = to_dict
    return conf


class TestAutoTraderTradeJournal:
    def test_filled_confirmation_logged(self):
        from core.auto_trader import AutoTrader
        pnl_tracker = MagicMock()
        trader = AutoTrader(dry_run=False, account_value_usd=50_000, pnl_tracker=pnl_tracker)
        conf = _make_confirmation("SPY", "filled")
        trader._log_confirmations_to_journal([conf])
        pnl_tracker.log_trade_from_confirmation.assert_called_once_with(conf)

    def test_submitted_confirmation_logged(self):
        from core.auto_trader import AutoTrader
        pnl_tracker = MagicMock()
        trader = AutoTrader(dry_run=False, account_value_usd=50_000, pnl_tracker=pnl_tracker)
        conf = _make_confirmation("IWM", "submitted")
        trader._log_confirmations_to_journal([conf])
        pnl_tracker.log_trade_from_confirmation.assert_called_once_with(conf)

    def test_rejected_confirmation_not_logged(self):
        from core.auto_trader import AutoTrader
        pnl_tracker = MagicMock()
        trader = AutoTrader(dry_run=False, account_value_usd=50_000, pnl_tracker=pnl_tracker)
        conf = _make_confirmation("QQQ", "rejected")
        trader._log_confirmations_to_journal([conf])
        pnl_tracker.log_trade_from_confirmation.assert_not_called()

    def test_no_pnl_tracker_does_not_crash(self):
        from core.auto_trader import AutoTrader
        trader = AutoTrader(dry_run=False, account_value_usd=50_000, pnl_tracker=None)
        conf = _make_confirmation("AAPL", "filled")
        # Must not raise
        trader._log_confirmations_to_journal([conf])

    def test_journal_error_does_not_raise(self):
        from core.auto_trader import AutoTrader
        pnl_tracker = MagicMock()
        pnl_tracker.log_trade_from_confirmation.side_effect = RuntimeError("DB error")
        trader = AutoTrader(dry_run=False, account_value_usd=50_000, pnl_tracker=pnl_tracker)
        conf = _make_confirmation("SPY", "filled")
        # Must not raise even though logger throws
        trader._log_confirmations_to_journal([conf])

    def test_multiple_confirmations_all_logged(self):
        from core.auto_trader import AutoTrader
        pnl_tracker = MagicMock()
        trader = AutoTrader(dry_run=False, account_value_usd=50_000, pnl_tracker=pnl_tracker)
        confs = [
            _make_confirmation("SPY", "filled"),
            _make_confirmation("IWM", "submitted"),
            _make_confirmation("QQQ", "rejected"),  # should NOT be logged
        ]
        trader._log_confirmations_to_journal(confs)
        assert pnl_tracker.log_trade_from_confirmation.call_count == 2


# ── MarketScheduler — guard + pnl_tracker wiring ─────────────────────────────


class TestMarketSchedulerSprintTenWiring:
    def _make_scheduler(self, auto_execute: bool = False) -> object:
        """Create a MarketScheduler with mocked dependencies."""
        with patch("TradingExecution.position_tracker.PositionTracker"), \
             patch("CentralAccounting.pnl_tracker.PnLTracker"), \
             patch("strategies.daily_loss_guard.DailyLossGuard"):
            from core.market_scheduler import MarketScheduler
            return MarketScheduler(auto_execute=auto_execute)

    def test_scheduler_creates_pnl_tracker(self):
        with patch("TradingExecution.position_tracker.PositionTracker"), \
             patch("CentralAccounting.pnl_tracker.PnLTracker") as mock_pnl, \
             patch("strategies.daily_loss_guard.DailyLossGuard"):
            from core.market_scheduler import MarketScheduler
            sched = MarketScheduler()
        assert mock_pnl.called
        assert sched._pnl_tracker is not None

    def test_scheduler_creates_daily_loss_guard(self):
        with patch("TradingExecution.position_tracker.PositionTracker"), \
             patch("CentralAccounting.pnl_tracker.PnLTracker"), \
             patch("strategies.daily_loss_guard.DailyLossGuard") as mock_guard:
            from core.market_scheduler import MarketScheduler
            sched = MarketScheduler()
        assert mock_guard.called
        assert sched._daily_loss_guard is not None

    def test_auto_execute_passes_guard_to_auto_trader(self):
        with patch("TradingExecution.position_tracker.PositionTracker"), \
             patch("CentralAccounting.pnl_tracker.PnLTracker"), \
             patch("strategies.daily_loss_guard.DailyLossGuard"), \
             patch("core.auto_trader.AutoTrader") as mock_at:
            from core.market_scheduler import MarketScheduler
            sched = MarketScheduler(auto_execute=True)
        call_kwargs = mock_at.call_args.kwargs
        assert "daily_loss_guard" in call_kwargs
        assert call_kwargs["daily_loss_guard"] is sched._daily_loss_guard

    def test_auto_execute_passes_pnl_tracker_to_auto_trader(self):
        with patch("TradingExecution.position_tracker.PositionTracker"), \
             patch("CentralAccounting.pnl_tracker.PnLTracker"), \
             patch("strategies.daily_loss_guard.DailyLossGuard"), \
             patch("core.auto_trader.AutoTrader") as mock_at:
            from core.market_scheduler import MarketScheduler
            sched = MarketScheduler(auto_execute=True)
        call_kwargs = mock_at.call_args.kwargs
        assert "pnl_tracker" in call_kwargs
        assert call_kwargs["pnl_tracker"] is sched._pnl_tracker

    def test_run_pnl_snapshot_uses_shared_tracker(self):
        with patch("TradingExecution.position_tracker.PositionTracker"), \
             patch("CentralAccounting.pnl_tracker.PnLTracker"), \
             patch("strategies.daily_loss_guard.DailyLossGuard"):
            from core.market_scheduler import MarketScheduler
            sched = MarketScheduler()

        # Override shared tracker with a mock
        mock_pnl = MagicMock()
        mock_pnl.take_snapshot.return_value = {"date": "2026-01-01", "daily_pnl": None,
                                                 "positions": [], "today_trades": []}
        sched._pnl_tracker = mock_pnl

        with patch.object(sched, "_fetch_positions", return_value=[]):
            sched.run_pnl_snapshot()

        mock_pnl.take_snapshot.assert_called_once()

    def test_run_pnl_snapshot_returns_empty_on_error(self):
        with patch("TradingExecution.position_tracker.PositionTracker"), \
             patch("CentralAccounting.pnl_tracker.PnLTracker"), \
             patch("strategies.daily_loss_guard.DailyLossGuard"):
            from core.market_scheduler import MarketScheduler
            sched = MarketScheduler()

        bad_pnl = MagicMock()
        bad_pnl.take_snapshot.side_effect = RuntimeError("DB exploded")
        sched._pnl_tracker = bad_pnl

        with patch.object(sched, "_fetch_positions", return_value=[]):
            result = sched.run_pnl_snapshot()

        assert result == {}
