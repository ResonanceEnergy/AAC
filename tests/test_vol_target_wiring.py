"""Tests for AutoTrader vol_sizer wiring — Sprint 25."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

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


def _make_vol_sizer(kelly_return=0.5):
    sizer = MagicMock()
    sizer.vol_adjusted_kelly.return_value = kelly_return
    return sizer


def _make_auto_trader(vol_sizer=None, dry_run=True):
    from core.auto_trader import AutoTrader  # noqa: PLC0415
    return AutoTrader(dry_run=dry_run, vol_sizer=vol_sizer)


# ---------------------------------------------------------------------------
# TestAutoTraderVolSizerParam
# ---------------------------------------------------------------------------

class TestAutoTraderVolSizerParam:
    def test_vol_sizer_stored_as_attribute(self):
        sizer = _make_vol_sizer()
        trader = _make_auto_trader(vol_sizer=sizer)
        assert trader._vol_sizer is sizer

    def test_vol_sizer_none_by_default(self):
        from core.auto_trader import AutoTrader
        trader = AutoTrader(dry_run=True)
        assert trader._vol_sizer is None

    def test_none_vol_sizer_does_not_filter_signals(self):
        trader = _make_auto_trader(vol_sizer=None)
        sig = _make_signal(confidence=0.90)
        summary = trader.run_once([sig])
        assert summary.signals_approved >= 1

    def test_vol_sizer_called_once_per_signal(self):
        sizer = _make_vol_sizer(kelly_return=1.0)
        trader = _make_auto_trader(vol_sizer=sizer, dry_run=True)
        sig1 = _make_signal("SPY")
        sig2 = _make_signal("TLT")
        trader.run_once([sig1, sig2])
        assert sizer.vol_adjusted_kelly.call_count == 2


# ---------------------------------------------------------------------------
# TestVolGateFiltering
# ---------------------------------------------------------------------------

class TestVolGateFiltering:
    def test_signal_filtered_when_kelly_below_threshold(self):
        sizer = _make_vol_sizer(kelly_return=0.005)  # below 0.01
        trader = _make_auto_trader(vol_sizer=sizer, dry_run=True)
        sig = _make_signal(confidence=0.90)
        summary = trader.run_once([sig])
        assert summary.signals_approved == 0
        assert any("VOL_GATE" in r for r in summary.filter_reasons)

    def test_signal_approved_when_kelly_above_threshold(self):
        sizer = _make_vol_sizer(kelly_return=0.50)
        trader = _make_auto_trader(vol_sizer=sizer, dry_run=True)
        sig = _make_signal(confidence=0.90)
        summary = trader.run_once([sig])
        assert summary.signals_approved >= 1

    def test_kelly_exactly_at_threshold_approved(self):
        sizer = _make_vol_sizer(kelly_return=0.01)
        trader = _make_auto_trader(vol_sizer=sizer, dry_run=True)
        sig = _make_signal(confidence=0.90)
        summary = trader.run_once([sig])
        assert summary.signals_approved >= 1

    def test_multiple_signals_partial_filter(self):
        call_count = [0]
        results = [0.5, 0.001]  # first passes, second filtered

        def side_effect(**kwargs):  # noqa: ARG001
            idx = call_count[0]
            call_count[0] += 1
            return results[idx] if idx < len(results) else 0.5

        sizer = MagicMock()
        sizer.vol_adjusted_kelly.side_effect = side_effect
        trader = _make_auto_trader(vol_sizer=sizer, dry_run=True)
        s1 = _make_signal("SPY", confidence=0.90)
        s2 = _make_signal("TLT", confidence=0.90)
        summary = trader.run_once([s1, s2])
        assert summary.signals_approved == 1
        assert summary.signals_filtered >= 1

    def test_filter_reason_contains_ticker(self):
        sizer = _make_vol_sizer(kelly_return=0.0)
        trader = _make_auto_trader(vol_sizer=sizer, dry_run=True)
        sig = _make_signal("XYZ", confidence=0.90)
        summary = trader.run_once([sig])
        assert any("XYZ" in r for r in summary.filter_reasons)

    def test_vol_gate_exception_fails_open(self):
        sizer = MagicMock()
        sizer.vol_adjusted_kelly.side_effect = RuntimeError("bad vol calc")
        trader = _make_auto_trader(vol_sizer=sizer, dry_run=True)
        sig = _make_signal(confidence=0.90)
        summary = trader.run_once([sig])
        # Fails-open: signal approved despite exception
        assert summary.signals_approved >= 1


# ---------------------------------------------------------------------------
# TestMarketSchedulerVolSizerWiring
# ---------------------------------------------------------------------------

class TestMarketSchedulerVolSizerWiring:
    def test_scheduler_creates_vol_sizer(self):
        from core.market_scheduler import MarketScheduler
        sched = MarketScheduler(auto_execute=False)
        from strategies.vol_target_sizer import VolTargetSizer
        assert isinstance(sched._vol_sizer, VolTargetSizer)

    def test_scheduler_wires_vol_sizer_to_auto_trader(self):
        from core.market_scheduler import MarketScheduler
        sched = MarketScheduler(auto_execute=True)
        assert sched._auto_trader is not None
        assert sched._auto_trader._vol_sizer is sched._vol_sizer
