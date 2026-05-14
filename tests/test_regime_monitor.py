from __future__ import annotations

"""tests/test_regime_monitor.py — Sprint 16: Regime Change Alerting.

Tests for:
  * Regime.from_score() threshold mapping
  * RegimeMonitor SQLite persistence and transition detection
  * RegimeTransition helpers (to_dict, escalation flags)
  * RegimeAlerts (is_escalation, multipliers, format_alert, apply_regime_filter)
  * MarketScheduler wiring (monitor created at init, called in run_signal_scan)
"""

import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Module imports ─────────────────────────────────────────────────────────

from strategies.regime_monitor import (
    Regime,
    RegimeMonitor,
    RegimeRecord,
    RegimeTransition,
)
from strategies.regime_alerts import (
    apply_regime_filter,
    confidence_multiplier,
    format_alert,
    is_de_escalation,
    is_escalation,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_transition(prev: str, new: str, score: float = 55.0) -> RegimeTransition:
    return RegimeTransition(
        prev_regime=prev,
        new_regime=new,
        composite_score=score,
        detected_at="2026-04-21T12:00:00+00:00",
    )


def _make_signal(ticker: str = "SPY", confidence: float = 0.70) -> Any:
    """Return a minimal TradeSignal suitable for regime filter tests."""
    from shared.signal import Direction, TradeSignal  # noqa: PLC0415

    class _FakeDir:
        value = "LONG_PUT"

    return TradeSignal(
        ticker=ticker,
        direction=Direction.LONG_PUT,
        confidence=confidence,
        entry=100.0,
        stop=90.0,
        target=80.0,
        size=0.05,
    )


# ══════════════════════════════════════════════════════════════════════════
# TestRegime — from_score threshold mapping
# ══════════════════════════════════════════════════════════════════════════


class TestRegime:
    def test_from_score_calm(self) -> None:
        assert Regime.from_score(0.0) == Regime.CALM
        assert Regime.from_score(15.0) == Regime.CALM
        assert Regime.from_score(29.9) == Regime.CALM

    def test_from_score_watch(self) -> None:
        assert Regime.from_score(30.0) == Regime.WATCH
        assert Regime.from_score(40.0) == Regime.WATCH
        assert Regime.from_score(49.9) == Regime.WATCH

    def test_from_score_elevated(self) -> None:
        assert Regime.from_score(50.0) == Regime.ELEVATED
        assert Regime.from_score(60.0) == Regime.ELEVATED
        assert Regime.from_score(69.9) == Regime.ELEVATED

    def test_from_score_crisis(self) -> None:
        assert Regime.from_score(70.0) == Regime.CRISIS
        assert Regime.from_score(85.0) == Regime.CRISIS
        assert Regime.from_score(100.0) == Regime.CRISIS

    def test_severity_ordering(self) -> None:
        assert Regime.CALM.severity < Regime.WATCH.severity
        assert Regime.WATCH.severity < Regime.ELEVATED.severity
        assert Regime.ELEVATED.severity < Regime.CRISIS.severity


# ══════════════════════════════════════════════════════════════════════════
# TestRegimeTransition — DTO helpers
# ══════════════════════════════════════════════════════════════════════════


class TestRegimeTransition:
    def test_to_dict_keys(self) -> None:
        t = _make_transition("CALM", "ELEVATED", 55.0)
        d = t.to_dict()
        assert set(d.keys()) == {"prev_regime", "new_regime", "composite_score", "detected_at"}
        assert d["prev_regime"] == "CALM"
        assert d["new_regime"] == "ELEVATED"

    def test_frozen_dataclass(self) -> None:
        t = _make_transition("WATCH", "CRISIS")
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            t.prev_regime = "CALM"  # type: ignore[misc]

    def test_composite_score_preserved(self) -> None:
        t = _make_transition("CALM", "CRISIS", score=72.5)
        assert t.composite_score == 72.5


# ══════════════════════════════════════════════════════════════════════════
# TestRegimeMonitor — SQLite persistence
# ══════════════════════════════════════════════════════════════════════════


class TestRegimeMonitor:
    def test_db_file_created(self, tmp_path: Path) -> None:
        db = tmp_path / "test_regime.db"
        mon = RegimeMonitor(db_path=db)
        mon.close()
        assert db.exists()

    def test_first_record_returns_no_transition(self, tmp_path: Path) -> None:
        mon = RegimeMonitor(db_path=tmp_path / "r.db")
        result = mon.record(25.0)  # CALM — first ever record
        mon.close()
        assert result is None

    def test_same_regime_returns_no_transition(self, tmp_path: Path) -> None:
        mon = RegimeMonitor(db_path=tmp_path / "r.db")
        mon.record(25.0)   # CALM
        result = mon.record(28.0)  # CALM again
        mon.close()
        assert result is None

    def test_transition_returned_on_change(self, tmp_path: Path) -> None:
        mon = RegimeMonitor(db_path=tmp_path / "r.db")
        mon.record(25.0)   # CALM
        result = mon.record(55.0)  # ELEVATED
        mon.close()
        assert isinstance(result, RegimeTransition)
        assert result.prev_regime == "CALM"
        assert result.new_regime == "ELEVATED"

    def test_transition_composite_score_recorded(self, tmp_path: Path) -> None:
        mon = RegimeMonitor(db_path=tmp_path / "r.db")
        mon.record(20.0)
        result = mon.record(75.0)  # CRISIS
        mon.close()
        assert result is not None
        assert result.composite_score == 75.0

    def test_current_regime_none_when_empty(self, tmp_path: Path) -> None:
        mon = RegimeMonitor(db_path=tmp_path / "r.db")
        assert mon.current_regime() is None
        mon.close()

    def test_current_regime_after_record(self, tmp_path: Path) -> None:
        mon = RegimeMonitor(db_path=tmp_path / "r.db")
        mon.record(60.0)
        assert mon.current_regime() == Regime.ELEVATED
        mon.close()

    def test_get_history_returns_records(self, tmp_path: Path) -> None:
        mon = RegimeMonitor(db_path=tmp_path / "r.db")
        mon.record(20.0)
        mon.record(55.0)
        mon.record(75.0)
        history = mon.get_history()
        mon.close()
        assert len(history) == 3
        # newest first
        assert history[0].regime == "CRISIS"

    def test_get_history_respects_limit(self, tmp_path: Path) -> None:
        mon = RegimeMonitor(db_path=tmp_path / "r.db")
        for score in [20.0, 35.0, 55.0, 75.0, 80.0]:
            mon.record(score)
        history = mon.get_history(limit=2)
        mon.close()
        assert len(history) == 2

    def test_record_error_returns_none(self, tmp_path: Path) -> None:
        mon = RegimeMonitor(db_path=tmp_path / "r.db")
        mon._conn.close()  # break the connection
        result = mon.record(55.0)
        assert result is None  # fails-open

    def test_get_history_error_returns_empty(self, tmp_path: Path) -> None:
        mon = RegimeMonitor(db_path=tmp_path / "r.db")
        mon._conn.close()
        result = mon.get_history()
        assert result == []

    def test_current_regime_error_returns_none(self, tmp_path: Path) -> None:
        mon = RegimeMonitor(db_path=tmp_path / "r.db")
        mon._conn.close()
        result = mon.current_regime()
        assert result is None


# ══════════════════════════════════════════════════════════════════════════
# TestRegimeAlerts — direction helpers, multipliers, formatting
# ══════════════════════════════════════════════════════════════════════════


class TestRegimeAlerts:
    def test_is_escalation_true(self) -> None:
        t = _make_transition("CALM", "CRISIS")
        assert is_escalation(t) is True

    def test_is_escalation_false_same_severity(self) -> None:
        t = _make_transition("ELEVATED", "ELEVATED")
        assert is_escalation(t) is False

    def test_is_escalation_false_de_escalation(self) -> None:
        t = _make_transition("CRISIS", "CALM")
        assert is_escalation(t) is False

    def test_is_de_escalation_true(self) -> None:
        t = _make_transition("CRISIS", "WATCH")
        assert is_de_escalation(t) is True

    def test_is_de_escalation_false_escalation(self) -> None:
        t = _make_transition("CALM", "WATCH")
        assert is_de_escalation(t) is False

    def test_confidence_multiplier_escalation(self) -> None:
        t = _make_transition("CALM", "ELEVATED")
        assert confidence_multiplier(t) == pytest.approx(1.20)

    def test_confidence_multiplier_de_escalation(self) -> None:
        t = _make_transition("CRISIS", "WATCH")
        assert confidence_multiplier(t) == pytest.approx(0.85)

    def test_confidence_multiplier_lateral(self) -> None:
        t = _make_transition("WATCH", "WATCH")
        assert confidence_multiplier(t) == pytest.approx(1.00)

    def test_format_alert_escalation_contains_keywords(self) -> None:
        t = _make_transition("CALM", "CRISIS", 75.0)
        msg = format_alert(t)
        assert "ESCALATION" in msg
        assert "CALM" in msg
        assert "CRISIS" in msg
        assert "75.0" in msg

    def test_format_alert_de_escalation_label(self) -> None:
        t = _make_transition("CRISIS", "CALM", 20.0)
        msg = format_alert(t)
        assert "DE-ESCALATION" in msg


# ══════════════════════════════════════════════════════════════════════════
# TestApplyRegimeFilter — signal confidence adjustment
# ══════════════════════════════════════════════════════════════════════════


class TestApplyRegimeFilter:
    def test_escalation_boosts_confidence(self) -> None:
        sig = _make_signal(confidence=0.70)
        t = _make_transition("CALM", "ELEVATED")
        result = apply_regime_filter([sig], t)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.70 * 1.20)

    def test_de_escalation_dampens_confidence(self) -> None:
        sig = _make_signal(confidence=0.80)
        t = _make_transition("CRISIS", "CALM")
        result = apply_regime_filter([sig], t)
        assert result[0].confidence == pytest.approx(0.80 * 0.85)

    def test_confidence_capped_at_max(self) -> None:
        sig = _make_signal(confidence=0.90)  # 0.90 * 1.20 = 1.08 > 0.95
        t = _make_transition("WATCH", "CRISIS")
        result = apply_regime_filter([sig], t)
        assert result[0].confidence <= 0.95

    def test_lateral_returns_original_list(self) -> None:
        sig = _make_signal(confidence=0.70)
        t = _make_transition("WATCH", "WATCH")
        result = apply_regime_filter([sig], t)
        # Same list object returned (no-op path)
        assert result is [sig] or result[0].confidence == pytest.approx(0.70)

    def test_empty_signals_returns_empty(self) -> None:
        t = _make_transition("CALM", "CRISIS")
        result = apply_regime_filter([], t)
        assert result == []

    def test_original_signal_not_mutated(self) -> None:
        sig = _make_signal(confidence=0.70)
        t = _make_transition("CALM", "CRISIS")
        result = apply_regime_filter([sig], t)
        assert sig.confidence == pytest.approx(0.70)  # original unchanged
        assert result[0].confidence != pytest.approx(0.70)

    def test_multiple_signals_all_adjusted(self) -> None:
        sigs = [_make_signal("SPY", 0.60), _make_signal("QQQ", 0.75)]
        t = _make_transition("CALM", "ELEVATED")
        result = apply_regime_filter(sigs, t)
        assert len(result) == 2
        assert result[0].confidence == pytest.approx(0.60 * 1.20)
        assert result[1].confidence == pytest.approx(0.75 * 1.20)


# ══════════════════════════════════════════════════════════════════════════
# TestMarketSchedulerRegimeWiring — scheduler integration
# ══════════════════════════════════════════════════════════════════════════

# Lazy-import patch paths (imports happen inside __init__ / methods, not at
# module level, so we patch at the *definition site*, not at market_scheduler.*)
_PATCH_POSITION_TRACKER = "TradingExecution.position_tracker.PositionTracker"
_PATCH_PNL_TRACKER = "CentralAccounting.pnl_tracker.PnLTracker"
_PATCH_LOSS_GUARD = "strategies.daily_loss_guard.DailyLossGuard"
_PATCH_ORDER_MONITOR = "core.order_monitor.OrderMonitor"
_PATCH_ORDER_REGISTRY = "core.order_monitor.PendingOrderRegistry"
_PATCH_SIGNAL_JOURNAL = "strategies.signal_journal.SignalJournal"
_PATCH_REGIME_MONITOR = "strategies.regime_monitor.RegimeMonitor"
_PATCH_COMBINED_SIGNALS = "strategies.signal_aggregator.get_combined_signals"


class TestMarketSchedulerRegimeWiring:
    def test_regime_monitor_attribute_exists(self) -> None:
        """MarketScheduler must have a _regime_monitor attribute at init."""
        from core.market_scheduler import MarketScheduler  # noqa: PLC0415

        with (
            patch(_PATCH_POSITION_TRACKER),
            patch(_PATCH_PNL_TRACKER),
            patch(_PATCH_LOSS_GUARD),
            patch(_PATCH_ORDER_MONITOR),
            patch(_PATCH_ORDER_REGISTRY),
            patch(_PATCH_SIGNAL_JOURNAL),
            patch(_PATCH_REGIME_MONITOR),
        ):
            sched = MarketScheduler()
        assert hasattr(sched, "_regime_monitor")

    def test_regime_monitor_is_created_at_init(self) -> None:
        """RegimeMonitor() must be instantiated during MarketScheduler.__init__."""
        from core.market_scheduler import MarketScheduler  # noqa: PLC0415

        with (
            patch(_PATCH_POSITION_TRACKER),
            patch(_PATCH_PNL_TRACKER),
            patch(_PATCH_LOSS_GUARD),
            patch(_PATCH_ORDER_MONITOR),
            patch(_PATCH_ORDER_REGISTRY),
            patch(_PATCH_SIGNAL_JOURNAL),
            patch(_PATCH_REGIME_MONITOR) as mock_rm_cls,
        ):
            MarketScheduler()
        mock_rm_cls.assert_called_once()

    def test_run_signal_scan_calls_regime_record(self) -> None:
        """run_signal_scan() must call _regime_monitor.record() with a float score."""
        from core.market_scheduler import MarketScheduler  # noqa: PLC0415

        mock_rm = MagicMock()
        mock_rm.record.return_value = None  # no transition

        with (
            patch(_PATCH_POSITION_TRACKER),
            patch(_PATCH_PNL_TRACKER),
            patch(_PATCH_LOSS_GUARD),
            patch(_PATCH_ORDER_MONITOR),
            patch(_PATCH_ORDER_REGISTRY),
            patch(_PATCH_SIGNAL_JOURNAL),
            patch(_PATCH_REGIME_MONITOR, return_value=mock_rm),
            patch(_PATCH_COMBINED_SIGNALS, return_value=[]),
            patch.object(MarketScheduler, "_get_composite_score", return_value=55.0),
        ):
            sched = MarketScheduler()
            sched.run_signal_scan()

        mock_rm.record.assert_called_once_with(55.0)

    def test_regime_transition_applies_filter_to_signals(self) -> None:
        """When a RegimeTransition is returned, apply_regime_filter is called."""
        from core.market_scheduler import MarketScheduler  # noqa: PLC0415

        fake_transition = _make_transition("CALM", "ELEVATED", 55.0)
        mock_rm = MagicMock()
        mock_rm.record.return_value = fake_transition

        fake_signal = _make_signal("SPY", 0.70)

        with (
            patch(_PATCH_POSITION_TRACKER),
            patch(_PATCH_PNL_TRACKER),
            patch(_PATCH_LOSS_GUARD),
            patch(_PATCH_ORDER_MONITOR),
            patch(_PATCH_ORDER_REGISTRY),
            patch(_PATCH_SIGNAL_JOURNAL),
            patch(_PATCH_REGIME_MONITOR, return_value=mock_rm),
            patch(_PATCH_COMBINED_SIGNALS, return_value=[fake_signal]),
            patch.object(MarketScheduler, "_get_composite_score", return_value=55.0),
        ):
            sched = MarketScheduler()
            result = sched.run_signal_scan()

        # Signal confidence should have been boosted by escalation multiplier (1.20)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.70 * 1.20)

    def test_regime_error_does_not_block_scan(self) -> None:
        """If _regime_monitor.record() raises, run_signal_scan still returns signals."""
        from core.market_scheduler import MarketScheduler  # noqa: PLC0415

        mock_rm = MagicMock()
        mock_rm.record.side_effect = RuntimeError("DB exploded")

        fake_signal = _make_signal("IWM", 0.65)

        with (
            patch(_PATCH_POSITION_TRACKER),
            patch(_PATCH_PNL_TRACKER),
            patch(_PATCH_LOSS_GUARD),
            patch(_PATCH_ORDER_MONITOR),
            patch(_PATCH_ORDER_REGISTRY),
            patch(_PATCH_SIGNAL_JOURNAL),
            patch(_PATCH_REGIME_MONITOR, return_value=mock_rm),
            patch(_PATCH_COMBINED_SIGNALS, return_value=[fake_signal]),
            patch.object(MarketScheduler, "_get_composite_score", return_value=55.0),
        ):
            sched = MarketScheduler()
            result = sched.run_signal_scan()

        assert len(result) == 1  # signal still delivered despite regime error

    def test_get_composite_score_fallback(self) -> None:
        """_get_composite_score() must return 50.0 when war_room_engine raises."""
        from core.market_scheduler import MarketScheduler  # noqa: PLC0415

        with (
            patch(_PATCH_POSITION_TRACKER),
            patch(_PATCH_PNL_TRACKER),
            patch(_PATCH_LOSS_GUARD),
            patch(_PATCH_ORDER_MONITOR),
            patch(_PATCH_ORDER_REGISTRY),
            patch(_PATCH_SIGNAL_JOURNAL),
            patch(_PATCH_REGIME_MONITOR),
        ):
            sched = MarketScheduler()

        with patch(
            "strategies.war_room_engine.IndicatorState",
            side_effect=RuntimeError("engine broken"),
        ):
            score = sched._get_composite_score()

        assert score == pytest.approx(50.0)
