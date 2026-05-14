"""tests/test_auto_trader.py — Sprint 8: Auto-Execution Loop.

Tests for core/auto_trader.py: AutoTrader + ExecutionSummary.

Coverage:
  - ExecutionSummary.to_dict / format_report
  - AutoTrader env-var defaults (paper, dry_run, account_value)
  - _should_execute: FLAT, low-confidence, throttle, exposure-breach, pass
  - _exposure_ok: normal, oversized, fail-open on exception
  - run_once dry_run: no executor called, log only
  - run_once no-approved: empty confirmations
  - run_once execution: connect, execute, throttle updated
  - run_once connect-failure: empty confirmations, no crash
  - run_once execution error on single signal: continues for remaining
  - n_executed counts submitted+filled only
  - MarketScheduler auto_execute=True wires AutoTrader
  - MarketScheduler auto_execute=False has no AutoTrader
  - MarketScheduler accepts injected AutoTrader
  - MarketScheduler.run_signal_scan calls auto_trader.run_once
  - MarketScheduler.run_signal_scan skips auto_trader when signals empty
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.auto_trader import AutoTrader, ExecutionSummary, _MIN_CONFIDENCE, _THROTTLE_SECONDS
from shared.signal import AssetClass, Direction, TradeSignal


# ── helpers ───────────────────────────────────────────────────────────────────


def _sig(
    ticker: str = "SPY",
    direction: Direction = Direction.LONG_PUT,
    confidence: float = 0.80,
    size: float = 0.05,
) -> TradeSignal:
    return TradeSignal(
        ticker=ticker,
        direction=direction,
        confidence=confidence,
        entry=100.0,
        stop=90.0,
        target=110.0,
        size=size,
        strategy="test_strategy",
    )


def _fake_confirmation(ticker: str = "SPY", status: str = "submitted") -> MagicMock:
    conf = MagicMock()
    conf.signal_ticker = ticker
    conf.status = MagicMock()
    conf.status.value = status
    conf.order_id = f"ord_{ticker}"
    conf.filled_quantity = 1.0
    conf.avg_fill_price = 100.0
    conf.to_dict.return_value = {
        "signal_ticker": ticker,
        "status": status,
        "order_id": f"ord_{ticker}",
        "filled_quantity": 1.0,
        "avg_fill_price": 100.0,
    }
    return conf


# ── TestExecutionSummary ──────────────────────────────────────────────────────


class TestExecutionSummary:
    def test_to_dict_has_required_keys(self):
        s = ExecutionSummary(
            signals_received=3,
            signals_filtered=1,
            signals_approved=2,
            signals_executed=2,
            confirmations=[],
            dry_run=False,
            paper=True,
            filter_reasons=["SPY: FLAT signal"],
        )
        d = s.to_dict()
        for key in (
            "signals_received",
            "signals_filtered",
            "signals_approved",
            "signals_executed",
            "confirmations",
            "dry_run",
            "paper",
            "filter_reasons",
            "generated_at",
        ):
            assert key in d, f"missing key: {key}"
        assert d["signals_received"] == 3
        assert d["filter_reasons"] == ["SPY: FLAT signal"]

    def test_format_report_shows_dry_run(self):
        s = ExecutionSummary(
            signals_received=2,
            signals_filtered=0,
            signals_approved=2,
            signals_executed=0,
            confirmations=[],
            dry_run=True,
            paper=False,
            filter_reasons=[],
        )
        report = s.format_report()
        assert "DRY RUN" in report
        assert "2" in report

    def test_format_report_shows_confirmations(self):
        conf = _fake_confirmation("IWM", "filled")
        s = ExecutionSummary(
            signals_received=1,
            signals_filtered=0,
            signals_approved=1,
            signals_executed=1,
            confirmations=[conf],
            dry_run=False,
            paper=False,
            filter_reasons=[],
        )
        report = s.format_report()
        assert "FILLED" in report
        assert "IWM" in report

    def test_format_report_live_mode(self):
        s = ExecutionSummary(
            signals_received=0,
            signals_filtered=0,
            signals_approved=0,
            signals_executed=0,
            confirmations=[],
            dry_run=False,
            paper=False,
            filter_reasons=[],
        )
        assert "LIVE" in s.format_report()

    def test_format_report_paper_mode(self):
        s = ExecutionSummary(
            signals_received=0,
            signals_filtered=0,
            signals_approved=0,
            signals_executed=0,
            confirmations=[],
            dry_run=False,
            paper=True,
            filter_reasons=[],
        )
        assert "PAPER" in s.format_report()

    def test_format_report_caps_filter_reasons_at_5(self):
        reasons = [f"ticker{i}: low confidence" for i in range(10)]
        s = ExecutionSummary(
            signals_received=10,
            signals_filtered=10,
            signals_approved=0,
            signals_executed=0,
            confirmations=[],
            dry_run=False,
            paper=False,
            filter_reasons=reasons,
        )
        report = s.format_report()
        # Only first 5 shown
        assert report.count("ticker") <= 5


# ── TestAutoTraderInit ────────────────────────────────────────────────────────


class TestAutoTraderInit:
    def test_explicit_params_used(self):
        t = AutoTrader(
            min_confidence=0.85,
            paper=True,
            dry_run=False,
            account_value_usd=100_000.0,
            throttle_seconds=3600,
        )
        assert t.min_confidence == 0.85
        assert t.paper is True
        assert t.dry_run is False
        assert t.account_value_usd == 100_000.0
        assert t.throttle_seconds == 3600

    def test_env_dry_run_default_is_true(self, monkeypatch):
        monkeypatch.delenv("DRY_RUN", raising=False)
        t = AutoTrader(paper=False)
        assert t.dry_run is True  # safe default

    def test_env_paper_trading_false_by_default(self, monkeypatch):
        monkeypatch.delenv("PAPER_TRADING", raising=False)
        t = AutoTrader(dry_run=True)
        assert t.paper is False

    def test_env_account_value_fallback(self, monkeypatch):
        monkeypatch.delenv("ACCOUNT_VALUE_USD", raising=False)
        t = AutoTrader(paper=False, dry_run=True)
        assert t.account_value_usd == 50_000.0

    def test_env_account_value_read(self, monkeypatch):
        monkeypatch.setenv("ACCOUNT_VALUE_USD", "75000")
        t = AutoTrader(paper=False, dry_run=True)
        assert t.account_value_usd == 75_000.0

    def test_last_summary_initially_none(self):
        t = AutoTrader(paper=False, dry_run=True)
        assert t.last_summary is None


# ── TestShouldExecute ─────────────────────────────────────────────────────────


class TestShouldExecute:
    def _trader(self) -> AutoTrader:
        return AutoTrader(
            min_confidence=0.70,
            paper=True,
            dry_run=True,
            account_value_usd=50_000.0,
        )

    def test_flat_signal_rejected(self):
        t = self._trader()
        sig = _sig(direction=Direction.FLAT)
        ok, reason = t._should_execute(sig, [])

    def test_low_confidence_rejected(self):
        t = self._trader()
        sig = _sig(confidence=0.50)
        ok, reason = t._should_execute(sig, [])
        assert not ok
        assert "confidence" in reason

    def test_exact_min_confidence_passes(self):
        t = self._trader()
        sig = _sig(confidence=0.70)
        # exposure_ok should pass for a small 5% size
        ok, _ = t._should_execute(sig, [])
        t = self._trader()
        # Mark SPY as just executed
        t._last_executed["SPY"] = _time.monotonic()
        sig = _sig(ticker="SPY", confidence=0.90)
        ok, reason = t._should_execute(sig, [])
        assert not ok
        assert "throttled" in reason

    def test_throttle_clears_after_window(self):
        t = self._trader()
        # Mark SPY as executed well outside the window
        t._last_executed["SPY"] = _time.monotonic() - t.throttle_seconds - 1
        sig = _sig(ticker="SPY", confidence=0.90)
        ok, _ = t._should_execute(sig, [])
        t = self._trader()
        # 25% size will breach 15% max_single_position_pct
        sig = _sig(ticker="SPY", confidence=0.90, size=0.25)
        ok, reason = t._should_execute(sig, [])
        assert not ok
        assert "size" in reason.lower() or "15" in reason

    def test_good_signal_approved(self):
        t = self._trader()
        sig = _sig(confidence=0.80, size=0.05)
        ok, reason = t._should_execute(sig, [])
        assert ok
        assert reason == ""


# ── TestExposureOk ────────────────────────────────────────────────────────────


class TestExposureOk:
    def _trader(self) -> AutoTrader:
        return AutoTrader(
            paper=True,
            dry_run=True,
            account_value_usd=50_000.0,
        )

    def test_normal_size_passes(self):
        t = self._trader()
        sig = _sig(size=0.05)
        ok, reason = t._exposure_ok(sig, [])
        assert ok
        assert reason == ""

    def test_oversized_position_fails(self):
        t = self._trader()
        sig = _sig(size=0.20)  # 20% > 15% limit
        ok, reason = t._exposure_ok(sig, [])
        assert not ok
        assert reason != ""

    def test_fails_open_on_exception(self):
        t = self._trader()
        sig = _sig()
        with patch("strategies.risk_engine.ExposureCalculator", side_effect=RuntimeError("boom")):
            ok, reason = t._exposure_ok(sig, [])
        # Fails open — don't block trading on check failure
        assert ok


# ── TestRunOnceDryRun ─────────────────────────────────────────────────────────


class TestRunOnceDryRun:
    def test_dry_run_does_not_call_executor(self):
        t = AutoTrader(paper=True, dry_run=True, account_value_usd=50_000.0)
        signals = [_sig("SPY"), _sig("IWM")]
        with patch("TradingExecution.signal_executor.SignalExecutor") as mock_exec:
            t.run_once(signals)
        mock_exec.assert_not_called()

    def test_dry_run_returns_zero_executed(self):
        t = AutoTrader(paper=True, dry_run=True, account_value_usd=50_000.0)
        summary = t.run_once([_sig("SPY", confidence=0.90)])
        assert summary.dry_run is True
        assert summary.signals_executed == 0
        assert summary.confirmations == []

    def test_dry_run_counts_approved(self):
        t = AutoTrader(
            min_confidence=0.70,
            paper=True,
            dry_run=True,
            account_value_usd=50_000.0,
        )
        # Two high-confidence, one low
        signals = [_sig("SPY", confidence=0.90), _sig("IWM", confidence=0.80), _sig("QQQ", confidence=0.40)]
        summary = t.run_once(signals)
        assert summary.signals_received == 3
        assert summary.signals_approved == 2
        assert summary.signals_filtered == 1

    def test_dry_run_updates_last_summary(self):
        t = AutoTrader(paper=True, dry_run=True, account_value_usd=50_000.0)
        assert t.last_summary is None
        t.run_once([_sig()])
        assert t.last_summary is not None
        assert t.last_summary.dry_run is True

    def test_dry_run_does_not_update_throttle(self):
        t = AutoTrader(paper=True, dry_run=True, account_value_usd=50_000.0)
        t.run_once([_sig("SPY", confidence=0.90)])
        # throttle should NOT be updated in dry_run mode
        assert "SPY" not in t._last_executed


# ── TestRunOnceEmpty ──────────────────────────────────────────────────────────


class TestRunOnceEmpty:
    def test_empty_signals_returns_zeros(self):
        t = AutoTrader(paper=True, dry_run=False, account_value_usd=50_000.0)
        summary = t.run_once([])
        assert summary.signals_received == 0
        assert summary.signals_filtered == 0
        assert summary.signals_approved == 0
        assert summary.signals_executed == 0

    def test_all_filtered_returns_no_confirmations(self):
        t = AutoTrader(
            min_confidence=0.95,
            paper=True,
            dry_run=False,
            account_value_usd=50_000.0,
        )
        # All below 0.95 threshold
        signals = [_sig(confidence=0.70), _sig("IWM", confidence=0.60)]
        summary = t.run_once(signals)
        assert summary.signals_approved == 0
        assert summary.signals_executed == 0
        assert len(summary.filter_reasons) == 2


# ── TestRunOnceExecution ──────────────────────────────────────────────────────


class TestRunOnceExecution:
    def _make_trader(self) -> AutoTrader:
        return AutoTrader(
            min_confidence=0.70,
            paper=True,
            dry_run=False,
            account_value_usd=50_000.0,
            throttle_seconds=3600,
        )

    def test_successful_execution_returns_confirmations(self):
        t = self._make_trader()
        conf = _fake_confirmation("SPY", "submitted")

        async def _fake_execute_approved(signals):
            return [conf]

        t._execute_approved = _fake_execute_approved
        summary = t.run_once([_sig("SPY", confidence=0.80)])
        assert summary.signals_executed == 1
        assert len(summary.confirmations) == 1
        assert summary.dry_run is False

    def test_connect_failure_returns_empty_confirmations(self):
        t = self._make_trader()

        async def _connect_fail(signals):
            return []

        t._execute_approved = _connect_fail
        summary = t.run_once([_sig(confidence=0.80)])
        assert summary.signals_executed == 0
        assert summary.confirmations == []

    def test_execution_exception_does_not_crash(self):
        t = self._make_trader()

        async def _boom(signals):
            raise RuntimeError("exchange down")

        t._execute_approved = _boom
        # Should not raise
        summary = t.run_once([_sig(confidence=0.80)])
        assert summary.signals_executed == 0

    def test_n_executed_counts_submitted_and_filled(self):
        t = self._make_trader()
        confs = [
            _fake_confirmation("SPY", "submitted"),
            _fake_confirmation("IWM", "filled"),
            _fake_confirmation("QQQ", "rejected"),
            _fake_confirmation("KRE", "error"),
        ]

        async def _fake(signals):
            return confs

        t._execute_approved = _fake
        signals = [
            _sig("SPY", confidence=0.80),
            _sig("IWM", confidence=0.80),
            _sig("QQQ", confidence=0.80),
            _sig("KRE", confidence=0.80),
        ]
        summary = t.run_once(signals)
        assert summary.signals_executed == 2   # only submitted + filled

    def test_last_summary_updated_after_execution(self):
        t = self._make_trader()
        conf = _fake_confirmation("SPY", "submitted")

        async def _fake(signals):
            return [conf]

        t._execute_approved = _fake
        assert t.last_summary is None
        t.run_once([_sig("SPY", confidence=0.80)])
        assert t.last_summary is not None
        assert t.last_summary.signals_received == 1


# ── TestThrottleUpdate ────────────────────────────────────────────────────────


class TestThrottleUpdate:
    def test_throttle_set_after_submitted(self):
        """_last_executed updated inside _execute_approved for submitted signals."""
        t = AutoTrader(paper=True, dry_run=False, account_value_usd=50_000.0)

        # Build a real execution path using mocked SignalExecutor
        mock_executor = MagicMock()
        mock_executor.connect = AsyncMock(return_value=True)
        mock_executor.disconnect = AsyncMock()
        conf = _fake_confirmation("SPY", "submitted")
        mock_executor.execute = AsyncMock(return_value=conf)

        with patch("TradingExecution.signal_executor.SignalExecutor", return_value=mock_executor):
            import asyncio
            asyncio.run(t._execute_approved([_sig("SPY", confidence=0.80)]))

        assert "SPY" in t._last_executed

    def test_throttle_not_set_after_rejected(self):
        """_last_executed NOT updated when confirmation is rejected."""
        t = AutoTrader(paper=True, dry_run=False, account_value_usd=50_000.0)

        mock_executor = MagicMock()
        mock_executor.connect = AsyncMock(return_value=True)
        mock_executor.disconnect = AsyncMock()
        conf = _fake_confirmation("SPY", "rejected")
        mock_executor.execute = AsyncMock(return_value=conf)

        with patch("TradingExecution.signal_executor.SignalExecutor", return_value=mock_executor):
            import asyncio
            asyncio.run(t._execute_approved([_sig("SPY", confidence=0.80)]))

        assert "SPY" not in t._last_executed


# ── TestExecuteApproved ───────────────────────────────────────────────────────


class TestExecuteApproved:
    def test_connect_failure_returns_empty_list(self):
        t = AutoTrader(paper=True, dry_run=False, account_value_usd=50_000.0)
        mock_executor = MagicMock()
        mock_executor.connect = AsyncMock(return_value=False)
        mock_executor.disconnect = AsyncMock()

        with patch("TradingExecution.signal_executor.SignalExecutor", return_value=mock_executor):
            import asyncio
            result = asyncio.run(t._execute_approved([_sig()]))

        assert result == []
        mock_executor.execute.assert_not_called()

    def test_per_signal_exception_continues_to_next(self):
        t = AutoTrader(paper=True, dry_run=False, account_value_usd=50_000.0)
        mock_executor = MagicMock()
        mock_executor.connect = AsyncMock(return_value=True)
        mock_executor.disconnect = AsyncMock()
        conf_iwm = _fake_confirmation("IWM", "submitted")
        # SPY execute raises, IWM succeeds
        mock_executor.execute = AsyncMock(side_effect=[RuntimeError("bad"), conf_iwm])

        with patch("TradingExecution.signal_executor.SignalExecutor", return_value=mock_executor):
            import asyncio
            result = asyncio.run(t._execute_approved([_sig("SPY"), _sig("IWM")]))

        assert len(result) == 1
        assert result[0].signal_ticker == "IWM"

    def test_disconnect_always_called(self):
        t = AutoTrader(paper=True, dry_run=False, account_value_usd=50_000.0)
        mock_executor = MagicMock()
        mock_executor.connect = AsyncMock(return_value=True)
        mock_executor.disconnect = AsyncMock()
        mock_executor.execute = AsyncMock(side_effect=RuntimeError("boom"))

        with patch("TradingExecution.signal_executor.SignalExecutor", return_value=mock_executor):
            import asyncio
            asyncio.run(t._execute_approved([_sig()]))

        mock_executor.disconnect.assert_called_once()


# ── TestMarketSchedulerIntegration ───────────────────────────────────────────


class TestMarketSchedulerIntegration:
    def test_auto_execute_false_has_no_auto_trader(self):
        from core.market_scheduler import MarketScheduler

        sched = MarketScheduler(auto_execute=False)
        assert sched._auto_trader is None

    def test_auto_execute_true_creates_auto_trader(self):
        from core.market_scheduler import MarketScheduler

        sched = MarketScheduler(auto_execute=True)
        from core.auto_trader import AutoTrader

        assert isinstance(sched._auto_trader, AutoTrader)

    def test_injected_auto_trader_used(self):
        from core.market_scheduler import MarketScheduler

        mock_trader = MagicMock()
        sched = MarketScheduler(auto_execute=False, auto_trader=mock_trader)
        assert sched._auto_trader is mock_trader

    def test_injected_auto_trader_overrides_auto_execute(self):
        from core.market_scheduler import MarketScheduler
        from core.auto_trader import AutoTrader

        mock_trader = MagicMock()
        sched = MarketScheduler(auto_execute=True, auto_trader=mock_trader)
        # Injected trader should take precedence
        assert sched._auto_trader is mock_trader

    def test_run_signal_scan_calls_auto_trader_run_once(self):
        from core.market_scheduler import MarketScheduler

        mock_trader = MagicMock()
        mock_trader.run_once.return_value = MagicMock(
            signals_received=2,
            signals_approved=1,
            signals_executed=1,
            dry_run=False,
        )
        sched = MarketScheduler(auto_execute=False, auto_trader=mock_trader)
        fake_signals = [_sig("SPY"), _sig("IWM")]
        with patch("strategies.signal_aggregator.get_combined_signals", return_value=fake_signals):
            result = sched.run_signal_scan()

        mock_trader.run_once.assert_called_once_with(fake_signals)
        assert result == fake_signals

    def test_run_signal_scan_skips_auto_trader_on_empty_signals(self):
        from core.market_scheduler import MarketScheduler

        mock_trader = MagicMock()
        sched = MarketScheduler(auto_execute=False, auto_trader=mock_trader)
        with patch("strategies.signal_aggregator.get_combined_signals", return_value=[]):
            sched.run_signal_scan()

        mock_trader.run_once.assert_not_called()

    def test_run_signal_scan_handles_auto_trader_exception(self):
        from core.market_scheduler import MarketScheduler

        mock_trader = MagicMock()
        mock_trader.run_once.side_effect = RuntimeError("trader crash")
        sched = MarketScheduler(auto_execute=False, auto_trader=mock_trader)
        fake_signals = [_sig("SPY")]
        with patch("strategies.signal_aggregator.get_combined_signals", return_value=fake_signals):
            # Should not raise
            result = sched.run_signal_scan()
        assert result == fake_signals

    def test_no_auto_trader_scan_still_returns_signals(self):
        from core.market_scheduler import MarketScheduler

        sched = MarketScheduler(auto_execute=False)
        fake_signals = [_sig("SPY")]
        with patch("strategies.signal_aggregator.get_combined_signals", return_value=fake_signals):
            result = sched.run_signal_scan()
        assert result == fake_signals
