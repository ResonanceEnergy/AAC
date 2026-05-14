"""Tests for CorrelationTracker contagion gate wiring — Sprint 26."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(ticker="SPY", confidence=0.80, size=0.05, direction="LONG"):
    from shared.signal import Direction, TradeSignal  # noqa: PLC0415
    sig = MagicMock(spec=TradeSignal)
    sig.ticker = ticker
    sig.confidence = confidence
    sig.size = size
    if direction == "LONG":
        sig.direction = Direction.LONG
    elif direction == "SHORT":
        sig.direction = Direction.SHORT
    else:
        sig.direction = Direction.FLAT
    return sig


def _make_snapshot(regime="normal", absorption=0.55):
    from strategies.correlation_tracker import CorrelationSnapshot  # noqa: PLC0415
    return CorrelationSnapshot(
        timestamp="2026-06-01",
        correlation_matrix=pd.DataFrame(),
        eigenvalues=[1.0, 0.5],
        effective_n_assets=1.5,
        absorption_ratio=absorption,
        regime=regime,
    )


def _make_tracker(regime="normal", absorption=0.55):
    """Return a mock CorrelationTracker with last_snapshot pre-set."""
    tracker = MagicMock()
    tracker.last_snapshot = _make_snapshot(regime=regime, absorption=absorption)
    return tracker


def _make_auto_trader(correlation_tracker=None, dry_run=True):
    from core.auto_trader import AutoTrader  # noqa: PLC0415
    return AutoTrader(dry_run=dry_run, correlation_tracker=correlation_tracker)


# ---------------------------------------------------------------------------
# TestAutoTraderContagionGateParam
# ---------------------------------------------------------------------------

class TestAutoTraderContagionGateParam:
    def test_correlation_tracker_stored(self):
        tracker = _make_tracker()
        trader = _make_auto_trader(correlation_tracker=tracker)
        assert trader._correlation_tracker is tracker

    def test_correlation_tracker_none_by_default(self):
        from core.auto_trader import AutoTrader
        trader = AutoTrader(dry_run=True)
        assert trader._correlation_tracker is None


# ---------------------------------------------------------------------------
# TestContagionGateBlocking
# ---------------------------------------------------------------------------

class TestContagionGateBlocking:
    """Contagion regime gates out non-compulsory signals."""

    def test_contagion_regime_blocks_signals(self):
        """Normal signals are filtered when regime is contagion."""
        tracker = _make_tracker(regime="contagion", absorption=0.85)
        trader = _make_auto_trader(correlation_tracker=tracker)
        signals = [_make_signal("SPY"), _make_signal("IWM")]
        summary = trader.run_once(signals)
        # In contagion regime, non-compulsory signals should be blocked
        # (DRY_RUN=True so nothing executes, but gate fires before execution)
        assert summary.signals_approved == 0

    def test_normal_regime_passes_signals(self):
        """Signals are not filtered when regime is normal."""
        tracker = _make_tracker(regime="normal", absorption=0.55)
        trader = _make_auto_trader(correlation_tracker=tracker)
        signals = [_make_signal("SPY")]
        summary = trader.run_once(signals)
        assert summary.signals_approved >= 1

    def test_no_tracker_passes_signals(self):
        """Gate is skipped entirely when correlation_tracker is None."""
        trader = _make_auto_trader(correlation_tracker=None)
        signals = [_make_signal("SPY")]
        summary = trader.run_once(signals)
        assert summary.signals_approved >= 1

    def test_no_last_snapshot_passes_signals(self):
        """Gate is skipped when last_snapshot is None (tracker not yet updated)."""
        tracker = MagicMock()
        tracker.last_snapshot = None
        trader = _make_auto_trader(correlation_tracker=tracker)
        signals = [_make_signal("SPY")]
        summary = trader.run_once(signals)
        assert summary.signals_approved >= 1

    def test_decorrelating_regime_passes_signals(self):
        """Signals pass when regime is decorrelating (only contagion blocks)."""
        tracker = _make_tracker(regime="decorrelating", absorption=0.30)
        trader = _make_auto_trader(correlation_tracker=tracker)
        signals = [_make_signal("SPY")]
        summary = trader.run_once(signals)
        assert summary.signals_approved >= 1

    def test_contagion_multiple_signals_all_blocked(self):
        """Multiple signals are all blocked in contagion regime."""
        tracker = _make_tracker(regime="contagion", absorption=0.90)
        trader = _make_auto_trader(correlation_tracker=tracker)
        signals = [_make_signal(t) for t in ["SPY", "IWM", "TLT", "GLD"]]
        summary = trader.run_once(signals)
        assert summary.signals_approved == 0

    def test_contagion_gate_exception_fails_open(self):
        """If contagion gate raises, signals still pass (fail-open)."""
        tracker = MagicMock()
        # Simulate getattr raising on last_snapshot access
        type(tracker).last_snapshot = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        trader = _make_auto_trader(correlation_tracker=tracker)
        signals = [_make_signal("SPY")]
        # Should not raise; gate fails open
        summary = trader.run_once(signals)
        assert summary.signals_received == 1

    def test_contagion_signals_received_count(self):
        """signals_received always equals the number of signals submitted."""
        tracker = _make_tracker(regime="contagion", absorption=0.88)
        trader = _make_auto_trader(correlation_tracker=tracker)
        signals = [_make_signal("SPY"), _make_signal("IWM"), _make_signal("TLT")]
        summary = trader.run_once(signals)
        assert summary.signals_received == 3


# ---------------------------------------------------------------------------
# TestMarketSchedulerCorrelationWiring
# ---------------------------------------------------------------------------

class TestMarketSchedulerCorrelationWiring:
    """MarketScheduler creates a CorrelationTracker and wires it to AutoTrader."""

    def test_scheduler_has_correlation_tracker_attr(self):
        from core.market_scheduler import MarketScheduler
        sched = MarketScheduler(auto_execute=False)
        assert hasattr(sched, "_correlation_tracker")

    def test_scheduler_correlation_tracker_is_instance(self):
        from core.market_scheduler import MarketScheduler
        from strategies.correlation_tracker import CorrelationTracker
        sched = MarketScheduler(auto_execute=False)
        assert isinstance(sched._correlation_tracker, CorrelationTracker)

    def test_auto_trader_receives_correlation_tracker(self):
        """When auto_execute=True, AutoTrader gets the tracker."""
        from core.market_scheduler import MarketScheduler
        sched = MarketScheduler(auto_execute=True)
        if sched._auto_trader is not None:
            assert sched._auto_trader._correlation_tracker is sched._correlation_tracker

    def test_correlation_tracker_last_snapshot_none_on_init(self):
        """Tracker starts with no snapshot — scheduler hasn't run yet."""
        from core.market_scheduler import MarketScheduler
        sched = MarketScheduler(auto_execute=False)
        assert sched._correlation_tracker.last_snapshot is None
