"""tests/test_stop_manager.py — Sprint 11 tests.

Covers:
  * StopManager.scan() — stop_loss and take_profit triggers; fail-open; empty
  * stop_decisions_to_signals() — signal construction, sizing, notes tag
  * is_stop_close_signal() — tag detection
  * AutoTrader — STOP_CLOSE signals bypass throttle and confidence filter
  * MarketScheduler — run_signal_scan() wires StopManager; stop signals executed
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock, call, patch

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_position(
    symbol: str,
    quantity: float = -1.0,
    market_value: float = -500.0,
    market_price: float = 5.0,
    avg_cost: float = 10.0,
    unrealized_pnl: float = -500.0,
    multiplier: float = 100.0,
    sec_type: str = "OPT",
):
    """Return a simple namespace that looks like a PositionSnapshot."""
    return types.SimpleNamespace(
        symbol=symbol,
        quantity=quantity,
        market_value=market_value,
        market_price=market_price,
        avg_cost=avg_cost,
        unrealized_pnl=unrealized_pnl,
        multiplier=multiplier,
        sec_type=sec_type,
        realized_pnl=0.0,
        expiry=None,
        strike=None,
        right=None,
    )


def _make_stop_decision(
    symbol: str = "IWM",
    trigger: str = "stop_loss",
    reason: str = "unrealised P&L -55% hit stop (-50% threshold)",
    pnl_pct: float = -0.55,
    market_value: float = 450.0,
    quantity: float = -1.0,
):
    from strategies.stop_manager import StopDecision
    return StopDecision(
        symbol=symbol,
        trigger=trigger,
        reason=reason,
        pnl_pct=pnl_pct,
        market_value=market_value,
        quantity=quantity,
    )


# ── StopManager.scan() ────────────────────────────────────────────────────────

class TestStopManagerScan:
    def test_empty_positions_returns_empty(self):
        from strategies.stop_manager import StopManager
        mgr = StopManager()
        result = mgr.scan([], account_value_usd=50_000)
        assert result == []

    def test_no_trigger_returns_empty(self):
        """Position with small loss should not trigger."""
        from strategies.stop_manager import StopManager
        # avg_cost=10, quantity=-1, multiplier=100 → cost_basis=1000
        # unrealized_pnl=-100 → pnl_pct = -100/1000 = -10% (above -50% stop)
        pos = _make_position("SPY", unrealized_pnl=-100.0, avg_cost=10.0)
        mgr = StopManager()
        result = mgr.scan([pos], account_value_usd=50_000)
        assert result == []

    def test_stop_loss_triggered(self):
        """Position with -55% unrealised P&L should produce a stop_loss decision."""
        from strategies.stop_manager import StopManager
        # avg_cost=10, quantity=-1, multiplier=100 → cost_basis=1000
        # unrealized_pnl=-550 → pnl_pct=-55% → triggers stop at -50%
        pos = _make_position("IWM", unrealized_pnl=-550.0, avg_cost=10.0)
        mgr = StopManager()
        result = mgr.scan([pos], account_value_usd=50_000)
        assert len(result) == 1
        assert result[0].symbol == "IWM"
        assert result[0].trigger == "stop_loss"

    def test_take_profit_triggered(self):
        """Position with +110% unrealised P&L should produce a take_profit decision."""
        from strategies.stop_manager import StopManager
        # avg_cost=10, quantity=-1, multiplier=100 → cost_basis=1000
        # unrealized_pnl=+1100 → pnl_pct=+110% → triggers TP at +100%
        pos = _make_position("QQQ", unrealized_pnl=1100.0, avg_cost=10.0)
        mgr = StopManager()
        result = mgr.scan([pos], account_value_usd=50_000)
        assert len(result) == 1
        assert result[0].symbol == "QQQ"
        assert result[0].trigger == "take_profit"

    def test_multiple_positions_both_triggered(self):
        from strategies.stop_manager import StopManager
        stop_pos = _make_position("IWM", unrealized_pnl=-600.0, avg_cost=10.0)
        tp_pos = _make_position("QQQ", unrealized_pnl=1200.0, avg_cost=10.0)
        safe_pos = _make_position("SPY", unrealized_pnl=-50.0, avg_cost=10.0)
        mgr = StopManager()
        result = mgr.scan([stop_pos, tp_pos, safe_pos], account_value_usd=100_000)
        symbols = {d.symbol for d in result}
        assert "IWM" in symbols
        assert "QQQ" in symbols
        assert "SPY" not in symbols
        assert len(result) == 2

    def test_stop_decision_has_reason_string(self):
        from strategies.stop_manager import StopManager
        pos = _make_position("KRE", unrealized_pnl=-600.0, avg_cost=10.0)
        mgr = StopManager()
        result = mgr.scan([pos], account_value_usd=50_000)
        assert len(result) == 1
        assert "stop" in result[0].reason.lower()
        assert result[0].reason  # non-empty

    def test_stop_decision_market_value_positive(self):
        """market_value on the decision is always positive (absolute)."""
        from strategies.stop_manager import StopManager
        pos = _make_position("JNK", market_value=-450.0, unrealized_pnl=-600.0, avg_cost=10.0)
        mgr = StopManager()
        result = mgr.scan([pos], account_value_usd=50_000)
        assert len(result) == 1
        assert result[0].market_value >= 0.0

    def test_exception_in_scan_returns_empty(self):
        """If ExposureCalculator raises, scan() fails-open → empty list."""
        from strategies.stop_manager import StopManager
        mgr = StopManager()
        with patch(
            "strategies.stop_manager.StopManager._scan",
            side_effect=RuntimeError("boom"),
        ):
            result = mgr.scan([_make_position("SPY")], 50_000)
        assert result == []

    def test_urgent_only_alias(self):
        """urgent_only() is a direct alias for scan()."""
        from strategies.stop_manager import StopManager
        pos = _make_position("IWM", unrealized_pnl=-600.0, avg_cost=10.0)
        mgr = StopManager()
        via_scan = mgr.scan([pos], 50_000)
        via_urgent = mgr.urgent_only([pos], 50_000)
        assert len(via_scan) == len(via_urgent) == 1


# ── StopDecision.to_dict() ────────────────────────────────────────────────────

class TestStopDecisionToDict:
    def test_to_dict_keys(self):
        d = _make_stop_decision()
        result = d.to_dict()
        assert "symbol" in result
        assert "trigger" in result
        assert "reason" in result
        assert "pnl_pct" in result
        assert "market_value" in result
        assert "quantity" in result
        assert "decided_at" in result

    def test_to_dict_pnl_pct_is_percentage(self):
        d = _make_stop_decision(pnl_pct=-0.55)
        result = d.to_dict()
        assert result["pnl_pct"] == pytest.approx(-55.0, abs=0.1)


# ── stop_decisions_to_signals() ───────────────────────────────────────────────

class TestStopDecisionsToSignals:
    def test_returns_one_signal_per_decision(self):
        from strategies.stop_signals import stop_decisions_to_signals
        d1 = _make_stop_decision("IWM")
        d2 = _make_stop_decision("KRE", trigger="take_profit")
        result = stop_decisions_to_signals([d1, d2], [], 50_000)
        assert len(result) == 2

    def test_signal_direction_is_short(self):
        from shared.signal import Direction
        from strategies.stop_signals import stop_decisions_to_signals
        d = _make_stop_decision()
        sigs = stop_decisions_to_signals([d], [], 50_000)
        assert sigs[0].direction == Direction.SHORT

    def test_signal_confidence_is_one(self):
        from strategies.stop_signals import stop_decisions_to_signals
        d = _make_stop_decision()
        sigs = stop_decisions_to_signals([d], [], 50_000)
        assert sigs[0].confidence == 1.0

    def test_notes_starts_with_stop_close_tag(self):
        from strategies.stop_signals import STOP_CLOSE_TAG, stop_decisions_to_signals
        d = _make_stop_decision()
        sigs = stop_decisions_to_signals([d], [], 50_000)
        assert sigs[0].notes.startswith(STOP_CLOSE_TAG)

    def test_notes_includes_trigger_and_reason(self):
        from strategies.stop_signals import stop_decisions_to_signals
        d = _make_stop_decision(trigger="stop_loss", reason="test reason")
        sigs = stop_decisions_to_signals([d], [], 50_000)
        assert "stop_loss" in sigs[0].notes
        assert "test reason" in sigs[0].notes

    def test_size_fraction_from_market_value(self):
        """$5,000 position in a $50,000 account → size = 0.10."""
        from strategies.stop_signals import stop_decisions_to_signals
        d = _make_stop_decision(market_value=5_000.0)
        sigs = stop_decisions_to_signals([d], [], 50_000)
        assert sigs[0].size == pytest.approx(0.10, abs=0.001)

    def test_size_fraction_capped_at_50pct(self):
        """Very large position → size capped at 0.50."""
        from strategies.stop_signals import stop_decisions_to_signals
        d = _make_stop_decision(market_value=999_999.0)
        sigs = stop_decisions_to_signals([d], [], 50_000)
        assert sigs[0].size == pytest.approx(0.50, abs=0.001)

    def test_entry_uses_market_price_from_position(self):
        from strategies.stop_signals import stop_decisions_to_signals
        pos = _make_position("IWM", market_price=3.75)
        d = _make_stop_decision("IWM")
        sigs = stop_decisions_to_signals([d], [pos], 50_000)
        assert sigs[0].entry == pytest.approx(3.75)

    def test_entry_zero_when_no_matching_position(self):
        from strategies.stop_signals import stop_decisions_to_signals
        d = _make_stop_decision("XYZ")
        sigs = stop_decisions_to_signals([d], [], 50_000)
        assert sigs[0].entry == pytest.approx(0.0)

    def test_empty_decisions_returns_empty(self):
        from strategies.stop_signals import stop_decisions_to_signals
        assert stop_decisions_to_signals([], [], 50_000) == []

    def test_strategy_label(self):
        from strategies.stop_signals import stop_decisions_to_signals
        d = _make_stop_decision()
        sigs = stop_decisions_to_signals([d], [], 50_000)
        assert sigs[0].strategy == "stop_manager"


# ── is_stop_close_signal() ────────────────────────────────────────────────────

class TestIsStopCloseSignal:
    def test_stop_signal_detected(self):
        from strategies.stop_signals import STOP_CLOSE_TAG, is_stop_close_signal
        sig = types.SimpleNamespace(notes=f"{STOP_CLOSE_TAG}:stop_loss:test")
        assert is_stop_close_signal(sig) is True

    def test_roll_signal_not_detected(self):
        from strategies.stop_signals import is_stop_close_signal
        sig = types.SimpleNamespace(notes="ROLL_CLOSE:close:expired")
        assert is_stop_close_signal(sig) is False

    def test_plain_signal_not_detected(self):
        from strategies.stop_signals import is_stop_close_signal
        sig = types.SimpleNamespace(notes="war_room signal")
        assert is_stop_close_signal(sig) is False

    def test_empty_notes_not_detected(self):
        from strategies.stop_signals import is_stop_close_signal
        sig = types.SimpleNamespace(notes="")
        assert is_stop_close_signal(sig) is False

    def test_no_notes_attr_not_detected(self):
        from strategies.stop_signals import is_stop_close_signal
        sig = types.SimpleNamespace()
        assert is_stop_close_signal(sig) is False


# ── AutoTrader STOP_CLOSE bypass ──────────────────────────────────────────────

class TestAutoTraderStopCloseBypass:
    """STOP_CLOSE signals must bypass throttle and confidence filters."""

    def _make_stop_signal(self):
        from shared.signal import Direction, TradeSignal
        from strategies.stop_signals import STOP_CLOSE_TAG
        return TradeSignal(
            ticker="IWM",
            direction=Direction.SHORT,
            confidence=1.0,
            entry=5.0,
            stop=0.0,
            target=0.0,
            size=0.05,
            strategy="stop_manager",
            notes=f"{STOP_CLOSE_TAG}:stop_loss:pnl -55% hit stop",
        )

    def test_stop_signal_bypasses_confidence_filter(self):
        """A STOP_CLOSE signal with confidence 1.0 should pass even when
        AutoTrader.min_confidence is not met (shouldn't matter — 1.0 >= any min)
        and more critically, bypass the throttle even when ticker was just traded."""
        from core.auto_trader import AutoTrader
        import time as _time_mod
        trader = AutoTrader(min_confidence=0.90, dry_run=True, throttle_seconds=3600)
        # Fake that IWM was executed 1 second ago (well within 3600s throttle).
        trader._last_executed["IWM"] = _time_mod.monotonic()

        sig = self._make_stop_signal()
        ok, reason = trader._should_execute(sig, [])
        assert ok is True, f"Expected bypass, got: {reason}"

    def test_stop_signal_bypasses_throttle(self):
        """Even with a 4-hour throttle, STOP_CLOSE must execute."""
        from core.auto_trader import AutoTrader
        import time as _time_mod
        trader = AutoTrader(dry_run=True, throttle_seconds=14400)
        trader._last_executed["KRE"] = _time_mod.monotonic()  # just executed

        from shared.signal import Direction, TradeSignal
        from strategies.stop_signals import STOP_CLOSE_TAG
        sig = TradeSignal(
            ticker="KRE",
            direction=Direction.SHORT,
            confidence=1.0,
            entry=0.0,
            stop=0.0,
            target=0.0,
            size=0.05,
            strategy="stop_manager",
            notes=f"{STOP_CLOSE_TAG}:stop_loss:stop hit",
        )
        ok, reason = trader._should_execute(sig, [])
        assert ok is True, f"Expected bypass, got: {reason}"

    def test_normal_signal_still_throttled(self):
        """Ensure STOP_CLOSE bypass doesn't break throttle for normal signals."""
        from core.auto_trader import AutoTrader
        from shared.signal import Direction, TradeSignal
        import time as _time_mod
        trader = AutoTrader(dry_run=True, throttle_seconds=14400)
        trader._last_executed["SPY"] = _time_mod.monotonic()

        sig = TradeSignal(
            ticker="SPY",
            direction=Direction.SHORT,
            confidence=0.95,
            entry=0.0,
            stop=0.0,
            target=0.0,
            size=0.05,
            strategy="war_room",
            notes="",
        )
        ok, _reason = trader._should_execute(sig, [])
        assert ok is False

    def test_run_once_with_stop_signal_executes(self):
        """run_once() with a STOP_CLOSE signal should show signals_approved >= 1."""
        from core.auto_trader import AutoTrader
        import time as _time_mod
        trader = AutoTrader(dry_run=True, throttle_seconds=14400)
        # Throttle IWM so a normal signal would be blocked.
        trader._last_executed["IWM"] = _time_mod.monotonic()
        sig = self._make_stop_signal()
        summary = trader.run_once([sig])
        assert summary.signals_approved >= 1


# ── MarketScheduler Sprint 11 wiring ─────────────────────────────────────────

class TestMarketSchedulerSprintElevenWiring:
    """run_signal_scan() must check stop/TP triggers after the regular scan."""

    def _make_scheduler_with_mock_trader(self):
        """Return (scheduler, mock_auto_trader) with auto_execute wired."""
        from core.market_scheduler import MarketScheduler
        mock_trader = MagicMock()
        mock_trader.account_value_usd = 50_000.0
        summary_mock = MagicMock()
        summary_mock.signals_received = 1
        summary_mock.signals_approved = 1
        summary_mock.signals_executed = 1
        summary_mock.dry_run = True
        mock_trader.run_once.return_value = summary_mock
        sched = MarketScheduler(auto_trader=mock_trader)
        return sched, mock_trader

    def test_stop_manager_called_during_signal_scan(self):
        """StopManager.scan() must be invoked once per run_signal_scan() cycle."""
        sched, mock_trader = self._make_scheduler_with_mock_trader()
        with (
            patch("strategies.signal_aggregator.get_combined_signals", return_value=[]),
            patch("strategies.stop_manager.StopManager.scan", return_value=[]) as mock_scan,
            patch.object(sched, "_fetch_positions", return_value=[]),
        ):
            sched.run_signal_scan()
        mock_scan.assert_called_once()

    def test_stop_signals_routed_when_triggered(self):
        """When stop decisions exist, run_once() is called with STOP_CLOSE signals."""
        from strategies.stop_manager import StopDecision
        sched, mock_trader = self._make_scheduler_with_mock_trader()
        decision = _make_stop_decision("IWM")
        with (
            patch("strategies.signal_aggregator.get_combined_signals", return_value=[]),
            patch("strategies.stop_manager.StopManager.scan", return_value=[decision]),
            patch.object(sched, "_fetch_positions", return_value=[_make_position("IWM")]),
        ):
            sched.run_signal_scan()

        # run_once must have been called with at least one STOP_CLOSE signal.
        from strategies.stop_signals import STOP_CLOSE_TAG
        all_calls = mock_trader.run_once.call_args_list
        stop_calls = [
            c for c in all_calls
            if any(
                str(getattr(sig, "notes", "")).startswith(STOP_CLOSE_TAG)
                for sig in (c.args[0] if c.args else [])
            )
        ]
        assert len(stop_calls) >= 1

    def test_no_stop_run_once_when_no_triggers(self):
        """If StopManager returns no decisions, the extra run_once must NOT be called."""
        sched, mock_trader = self._make_scheduler_with_mock_trader()
        with (
            patch("strategies.signal_aggregator.get_combined_signals", return_value=[]),
            patch("strategies.stop_manager.StopManager.scan", return_value=[]),
            patch.object(sched, "_fetch_positions", return_value=[]),
        ):
            sched.run_signal_scan()
        # run_once should not have been called at all (no strategy signals, no stop signals).
        mock_trader.run_once.assert_not_called()

    def test_stop_scan_error_does_not_raise(self):
        """If StopManager.scan() raises, run_signal_scan() must NOT propagate it."""
        sched, _mock_trader = self._make_scheduler_with_mock_trader()
        with (
            patch("strategies.signal_aggregator.get_combined_signals", return_value=[]),
            patch("strategies.stop_manager.StopManager.scan", side_effect=RuntimeError("boom")),
            patch.object(sched, "_fetch_positions", return_value=[]),
        ):
            result = sched.run_signal_scan()   # must not raise
        assert result == []

    def test_run_signal_scan_still_returns_strategy_signals(self):
        """Return value is the raw strategy signal list — stop signals are not included."""
        from shared.signal import Direction, TradeSignal
        sched, _mock_trader = self._make_scheduler_with_mock_trader()
        strat_signal = TradeSignal(
            ticker="SPY",
            direction=Direction.SHORT,
            confidence=0.75,
            entry=0.0,
            stop=0.0,
            target=0.0,
            size=0.05,
            strategy="war_room",
        )
        with (
            patch(
                "strategies.signal_aggregator.get_combined_signals",
                return_value=[strat_signal],
            ),
            patch("strategies.stop_manager.StopManager.scan", return_value=[]),
            patch.object(sched, "_fetch_positions", return_value=[]),
        ):
            result = sched.run_signal_scan()
        assert len(result) == 1
        assert result[0].ticker == "SPY"
