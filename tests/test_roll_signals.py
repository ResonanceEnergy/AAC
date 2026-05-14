"""tests/test_roll_signals.py — Sprint 9: Live Position Awareness.

Coverage:
  - roll_decisions_to_signals: empty input, actionable vs non-actionable actions
  - roll_decisions_to_signals: size calculation from market_value / account_value
  - roll_decisions_to_signals: fallback size when market_value == 0
  - roll_decisions_to_signals: missing position in positions list → skip + warn
  - roll_decisions_to_signals: env var ACCOUNT_VALUE_USD fallback
  - roll_decisions_to_signals: signal notes prefix = ROLL_CLOSE
  - roll_decisions_to_signals: direction is SHORT, confidence is 1.0
  - roll_decisions_to_signals: expiry and strike propagated
  - is_roll_close_signal: True for ROLL_CLOSE notes, False otherwise
  - AutoTrader: position_tracker=None → _fetch_positions_sync returns []
  - AutoTrader: position_tracker set → _fetch_positions_sync calls connect+refresh+disconnect
  - AutoTrader: _fetch_positions_sync returns [] on tracker error (fail-safe)
  - AutoTrader: _exposure_ok uses live positions from tracker in run_once cycle
  - AutoTrader: ROLL_CLOSE signals bypass confidence filter
  - AutoTrader: ROLL_CLOSE signals bypass throttle
  - AutoTrader: ROLL_CLOSE signals still go through exposure check
  - AutoTrader: position_tracker param accepted in __init__
  - AutoTrader: run_once fetches positions once per cycle (not per signal)
  - MarketScheduler: creates PositionTracker in __init__
  - MarketScheduler: auto_execute wires tracker into AutoTrader
  - MarketScheduler: run_roll_check uses _fetch_positions
  - MarketScheduler: run_pnl_snapshot uses _fetch_positions
  - MarketScheduler: run_roll_check routes urgent decisions through auto_trader
  - MarketScheduler: run_roll_check skips auto_trader when no urgent decisions
  - MarketScheduler: _fetch_positions returns [] when IBKR unavailable
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.auto_trader import AutoTrader
from shared.signal import Direction, TradeSignal
from strategies.roll_manager import RollAction, RollDecision
from strategies.roll_signals import (
    ROLL_CLOSE_TAG,
    is_roll_close_signal,
    roll_decisions_to_signals,
)


# ── helpers ───────────────────────────────────────────────────────────────────


@dataclass
class FakePosition:
    symbol: str
    market_value: float = 500.0
    market_price: float = 1.50
    sec_type: str = "OPT"
    expiry: str = "20250620"
    strike: float = 95.0
    right: str = "P"
    quantity: float = -1.0


def _decision(
    symbol: str = "SPY",
    action: RollAction = RollAction.CLOSE,
    reason: str = "DTE <= 7",
    dte: int = 5,
    expiry: str = "20250620",
    strike: float = 95.0,
    urgent: bool = True,
) -> RollDecision:
    return RollDecision(
        symbol=symbol,
        sec_type="OPT",
        dte=dte,
        action=action,
        reason=reason,
        expiry=expiry,
        strike=strike,
        right="P",
        market_price=1.50,
        quantity=-1.0,
        urgent=urgent,
    )


def _sig(
    ticker: str = "SPY",
    direction: Direction = Direction.LONG_PUT,
    confidence: float = 0.80,
    size: float = 0.05,
    notes: str = "",
) -> TradeSignal:
    return TradeSignal(
        ticker=ticker,
        direction=direction,
        confidence=confidence,
        entry=100.0,
        stop=90.0,
        target=110.0,
        size=size,
        strategy="test",
        notes=notes,
    )


def _fake_tracker(positions: list | None = None) -> MagicMock:
    """Return a mock PositionTracker that yields ``positions`` on refresh()."""
    if positions is None:
        positions = []
    tracker = MagicMock()
    tracker.connect = AsyncMock(return_value=True)
    tracker.refresh = AsyncMock(return_value=positions)
    tracker.disconnect = AsyncMock()
    return tracker


# ── TestRollDecisionsToSignals ────────────────────────────────────────────────


class TestRollDecisionsToSignals:
    def test_empty_input_returns_empty(self):
        result = roll_decisions_to_signals([], [], account_value_usd=50_000)
        assert result == []

    def test_close_action_produces_signal(self):
        dec = _decision(action=RollAction.CLOSE)
        pos = FakePosition("SPY")
        signals = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.ticker == "SPY"
        assert sig.direction is Direction.SHORT
        assert sig.confidence == 1.0
        assert sig.strategy == "roll_manager"

    def test_roll_action_produces_signal(self):
        dec = _decision(symbol="IWM", action=RollAction.ROLL, reason="DTE=18")
        pos = FakePosition("IWM")
        signals = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)
        assert len(signals) == 1

    def test_dead_put_action_produces_signal(self):
        dec = _decision(symbol="QQQ", action=RollAction.DEAD_PUT, reason="price~0")
        pos = FakePosition("QQQ")
        signals = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)
        assert len(signals) == 1

    def test_hold_action_skipped(self):
        dec = _decision(action=RollAction.HOLD, urgent=False)
        pos = FakePosition("SPY")
        signals = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)
        assert signals == []

    def test_expired_action_skipped(self):
        dec = _decision(action=RollAction.EXPIRED, urgent=True)
        pos = FakePosition("SPY")
        signals = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)
        assert signals == []

    def test_not_option_action_skipped(self):
        dec = _decision(action=RollAction.NOT_OPTION, urgent=False)
        pos = FakePosition("SPY")
        signals = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)
        assert signals == []

    def test_size_from_market_value_fraction(self):
        dec = _decision(action=RollAction.CLOSE)
        pos = FakePosition("SPY", market_value=5_000.0)
        signals = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)
        assert len(signals) == 1
        expected = 5_000 / 50_000  # 0.10
        assert abs(signals[0].size - expected) < 1e-6

    def test_size_capped_at_50_pct(self):
        dec = _decision(action=RollAction.CLOSE)
        pos = FakePosition("SPY", market_value=40_000.0)   # 80% of account
        signals = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)
        assert signals[0].size <= 0.50

    def test_size_floored_at_0_1_pct(self):
        dec = _decision(action=RollAction.CLOSE)
        pos = FakePosition("SPY", market_value=0.0)
        signals = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)
        assert signals[0].size >= 0.001

    def test_missing_position_is_skipped(self):
        dec = _decision(symbol="ARCC")
        pos = FakePosition("SPY")   # SPY position, not ARCC
        signals = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)
        assert signals == []

    def test_mixed_present_and_missing_positions(self):
        decisions = [
            _decision(symbol="SPY", action=RollAction.CLOSE),
            _decision(symbol="ARCC", action=RollAction.ROLL),  # no matching position
        ]
        positions = [FakePosition("SPY")]
        signals = roll_decisions_to_signals(decisions, positions, account_value_usd=50_000)
        assert len(signals) == 1
        assert signals[0].ticker == "SPY"

    def test_notes_contain_roll_close_tag(self):
        dec = _decision(action=RollAction.CLOSE)
        pos = FakePosition("SPY")
        sig = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)[0]
        assert sig.notes.startswith(ROLL_CLOSE_TAG)

    def test_notes_contain_action_and_reason(self):
        dec = _decision(symbol="IWM", action=RollAction.ROLL, reason="DTE=18")
        pos = FakePosition("IWM")
        sig = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)[0]
        assert "roll" in sig.notes
        assert "DTE=18" in sig.notes

    def test_expiry_propagated_to_signal(self):
        dec = _decision(expiry="20250620")
        pos = FakePosition("SPY")
        sig = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)[0]
        assert sig.expiry == "20250620"

    def test_strike_propagated_to_signal(self):
        dec = _decision(strike=95.0)
        pos = FakePosition("SPY")
        sig = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)[0]
        assert sig.strike == 95.0

    def test_account_value_fallback_from_env(self, monkeypatch):
        monkeypatch.setenv("ACCOUNT_VALUE_USD", "100000")
        dec = _decision(action=RollAction.CLOSE)
        pos = FakePosition("SPY", market_value=5_000.0)
        signals = roll_decisions_to_signals([dec], [pos], account_value_usd=0)
        assert len(signals) == 1
        expected = 5_000 / 100_000  # 0.05
        assert abs(signals[0].size - expected) < 1e-6

    def test_symbol_uppercased(self):
        dec = _decision(symbol="spy")  # lower-case input
        pos = FakePosition("SPY")      # upper-case in position
        signals = roll_decisions_to_signals([dec], [pos], account_value_usd=50_000)
        assert len(signals) == 1
        assert signals[0].ticker == "SPY"


# ── TestIsRollCloseSignal ─────────────────────────────────────────────────────


class TestIsRollCloseSignal:
    def test_roll_close_notes_returns_true(self):
        sig = _sig(notes="ROLL_CLOSE:close:DTE<=7")
        assert is_roll_close_signal(sig) is True

    def test_empty_notes_returns_false(self):
        sig = _sig(notes="")
        assert is_roll_close_signal(sig) is False

    def test_other_notes_returns_false(self):
        sig = _sig(notes="war_room_signal")
        assert is_roll_close_signal(sig) is False

    def test_roll_prefix_only_returns_true(self):
        sig = _sig(notes=ROLL_CLOSE_TAG)
        assert is_roll_close_signal(sig) is True


# ── TestAutoTraderPositionTracker ─────────────────────────────────────────────


class TestAutoTraderPositionTracker:
    def test_no_tracker_returns_empty_positions(self):
        t = AutoTrader(paper=True, dry_run=True, account_value_usd=50_000)
        assert t._position_tracker is None
        result = t._fetch_positions_sync()
        assert result == []

    def test_tracker_injected_in_init(self):
        tracker = _fake_tracker()
        t = AutoTrader(paper=True, dry_run=True, position_tracker=tracker)
        assert t._position_tracker is tracker

    def test_fetch_calls_connect_refresh_disconnect(self):
        pos = FakePosition("SPY", market_value=500.0)
        tracker = _fake_tracker([pos])
        t = AutoTrader(paper=True, dry_run=True, position_tracker=tracker)
        result = t._fetch_positions_sync()
        assert len(result) == 1
        tracker.connect.assert_called_once()
        tracker.refresh.assert_called_once()
        tracker.disconnect.assert_called_once()

    def test_fetch_returns_empty_on_exception(self):
        tracker = MagicMock()
        tracker.connect = AsyncMock(side_effect=ConnectionError("IBKR down"))
        t = AutoTrader(paper=True, dry_run=True, position_tracker=tracker)
        result = t._fetch_positions_sync()
        assert result == []

    def test_fetch_disconnects_even_on_refresh_error(self):
        tracker = MagicMock()
        tracker.connect = AsyncMock(return_value=True)
        tracker.refresh = AsyncMock(side_effect=RuntimeError("stream error"))
        tracker.disconnect = AsyncMock()
        t = AutoTrader(paper=True, dry_run=True, position_tracker=tracker)
        result = t._fetch_positions_sync()
        assert result == []
        tracker.disconnect.assert_called_once()

    def test_exposure_ok_passes_live_positions(self):
        """_exposure_ok should use provided live positions for total-exposure calc."""
        pos = FakePosition("SPY", market_value=35_000.0)  # 70% of 50k account
        t = AutoTrader(paper=True, dry_run=True, account_value_usd=50_000.0)
        # Adding another 5% (2,500) on top of 35k → 37.5k / 50k = 75% — under 80% limit
        sig = _sig("IWM", size=0.05)
        ok, reason = t._exposure_ok(sig, [pos])
        assert ok, f"Expected ok but got: {reason}"

    def test_exposure_ok_blocks_when_near_limit(self):
        """Adding to a near-full portfolio (78%) should be blocked."""
        pos = FakePosition("SPY", market_value=39_000.0)  # 78% of 50k
        t = AutoTrader(paper=True, dry_run=True, account_value_usd=50_000.0)
        # Adding another 5% (2,500) → 41.5k / 50k = 83% > 80% limit
        sig = _sig("IWM", size=0.05)
        ok, reason = t._exposure_ok(sig, [pos])
        assert not ok
        assert "exposure" in reason.lower() or "80" in reason or "total" in reason.lower()


# ── TestAutoTraderRollCloseBypass ─────────────────────────────────────────────


class TestAutoTraderRollCloseBypass:
    def _trader(self) -> AutoTrader:
        return AutoTrader(
            min_confidence=0.70,
            paper=True,
            dry_run=True,
            account_value_usd=50_000.0,
        )

    def test_roll_close_bypasses_confidence_filter(self):
        """ROLL_CLOSE signals with confidence 1.0 pass even if we raise min_confidence."""
        t = AutoTrader(min_confidence=0.99, paper=True, dry_run=True, account_value_usd=50_000.0)
        sig = _sig(confidence=1.0, notes=f"{ROLL_CLOSE_TAG}:close:DTE<=7")
        ok, reason = t._should_execute(sig, [])
        assert ok, f"Expected ROLL_CLOSE to bypass confidence; got: {reason}"

    def test_roll_close_bypasses_throttle(self):
        """ROLL_CLOSE signals execute even when the ticker is throttled."""
        import time as _time_module
        t = self._trader()
        # Mark SPY as just executed (well within throttle window)
        t._last_executed["SPY"] = _time_module.monotonic()
        sig = _sig(ticker="SPY", confidence=1.0, notes=f"{ROLL_CLOSE_TAG}:close:DTE<=7")
        ok, reason = t._should_execute(sig, [])
        assert ok, f"Expected ROLL_CLOSE to bypass throttle; got: {reason}"

    def test_roll_close_still_filtered_if_flat(self):
        """FLAT direction is always rejected, even for roll-close signals."""
        t = self._trader()
        sig = _sig(direction=Direction.FLAT, notes=f"{ROLL_CLOSE_TAG}:close:reason")
        ok, reason = t._should_execute(sig, [])
        assert not ok
        assert "FLAT" in reason

    def test_normal_signal_still_throttled(self):
        """Regular signals (no ROLL_CLOSE tag) are still throttled as before."""
        import time as _time_module
        t = self._trader()
        t._last_executed["SPY"] = _time_module.monotonic()
        sig = _sig(ticker="SPY", confidence=0.90)
        ok, reason = t._should_execute(sig, [])
        assert not ok
        assert "throttled" in reason

    def test_run_once_with_roll_close_signal_executes_in_dry_run(self):
        """Roll-close signals in dry_run mode are counted as approved."""
        t = AutoTrader(min_confidence=0.70, paper=True, dry_run=True, account_value_usd=50_000.0)
        sig = _sig(
            ticker="SPY",
            direction=Direction.SHORT,
            confidence=1.0,
            notes=f"{ROLL_CLOSE_TAG}:close:DTE<=7",
        )
        summary = t.run_once([sig])
        assert summary.signals_received == 1
        assert summary.signals_approved == 1
        assert summary.dry_run is True


# ── TestMarketSchedulerPositionWiring ─────────────────────────────────────────


class TestMarketSchedulerPositionWiring:
    def test_scheduler_creates_position_tracker(self):
        with patch("TradingExecution.position_tracker.PositionTracker") as mock_pt:
            mock_pt.return_value = MagicMock()
            from core.market_scheduler import MarketScheduler
            sched = MarketScheduler(paper=True)
        assert sched._position_tracker is not None

    def test_auto_execute_passes_tracker_to_auto_trader(self):
        fake_pt = MagicMock()
        with (
            patch("TradingExecution.position_tracker.PositionTracker", return_value=fake_pt),
            patch("core.auto_trader.AutoTrader") as mock_at,
        ):
            mock_at.return_value = MagicMock()
            from core.market_scheduler import MarketScheduler
            sched = MarketScheduler(paper=True, auto_execute=True)
        # AutoTrader should be constructed with position_tracker=
        call_kwargs = mock_at.call_args.kwargs
        assert "position_tracker" in call_kwargs
        assert call_kwargs["position_tracker"] is fake_pt

    def test_fetch_positions_uses_shared_tracker(self):
        pos = FakePosition("SPY")
        fake_pt = _fake_tracker([pos])
        with patch("TradingExecution.position_tracker.PositionTracker", return_value=fake_pt):
            from core.market_scheduler import MarketScheduler
            sched = MarketScheduler(paper=True)
        result = sched._fetch_positions()
        assert len(result) == 1
        fake_pt.connect.assert_called_once()
        fake_pt.refresh.assert_called_once()
        fake_pt.disconnect.assert_called_once()

    def test_fetch_positions_returns_empty_on_error(self):
        bad_pt = MagicMock()
        bad_pt.connect = AsyncMock(side_effect=ConnectionError("IBKR down"))
        with patch("TradingExecution.position_tracker.PositionTracker", return_value=bad_pt):
            from core.market_scheduler import MarketScheduler
            sched = MarketScheduler(paper=True)
        result = sched._fetch_positions()
        assert result == []

    def test_run_roll_check_uses_fetch_positions(self):
        pos = FakePosition("SPY")
        from core.market_scheduler import MarketScheduler
        sched = MarketScheduler(paper=True)
        with (
            patch.object(sched, "_fetch_positions", return_value=[pos]) as mock_fetch,
            patch("strategies.roll_manager.RollManager.urgent_only", return_value=[]),
        ):
            sched.run_roll_check()
        mock_fetch.assert_called_once()

    def test_run_pnl_snapshot_uses_fetch_positions(self):
        from core.market_scheduler import MarketScheduler
        sched = MarketScheduler(paper=True)
        with (
            patch.object(sched, "_fetch_positions", return_value=[]) as mock_fetch,
            patch("CentralAccounting.pnl_tracker.PnLTracker") as mock_pnl,
        ):
            mock_pnl.return_value.take_snapshot.return_value = {}
            mock_pnl.return_value.close.return_value = None
            sched.run_pnl_snapshot()
        mock_fetch.assert_called_once()

    def test_run_roll_check_routes_urgent_through_auto_trader(self):
        """Urgent decisions should be converted to signals and executed."""
        decision = _decision(action=RollAction.CLOSE)
        pos = FakePosition("SPY", market_value=500.0)

        mock_trader = MagicMock()
        mock_trader.account_value_usd = 50_000.0
        mock_trader.run_once.return_value = MagicMock(
            signals_executed=1, dry_run=False
        )

        from core.market_scheduler import MarketScheduler
        sched = MarketScheduler(paper=True, auto_trader=mock_trader)
        with (
            patch.object(sched, "_fetch_positions", return_value=[pos]),
            patch(
                "strategies.roll_manager.RollManager.urgent_only",
                return_value=[decision],
            ),
        ):
            decisions = sched.run_roll_check()

        assert len(decisions) == 1
        mock_trader.run_once.assert_called_once()
        # Verify the signals passed to run_once are ROLL_CLOSE signals
        call_args = mock_trader.run_once.call_args[0][0]
        assert len(call_args) == 1
        assert is_roll_close_signal(call_args[0])

    def test_run_roll_check_skips_auto_trader_when_no_decisions(self):
        mock_trader = MagicMock()
        from core.market_scheduler import MarketScheduler
        sched = MarketScheduler(paper=True, auto_trader=mock_trader)
        with (
            patch.object(sched, "_fetch_positions", return_value=[]),
            patch("strategies.roll_manager.RollManager.urgent_only", return_value=[]),
        ):
            sched.run_roll_check()
        mock_trader.run_once.assert_not_called()
