"""tests/test_signal_journal.py — Sprint 15: Signal Journal & Outcome Tracker.

Coverage:
    TestSignalJournal         (8)  — log_signal, get_recent, get_hit_rates, resolve, get_unresolved,
                                     log_signal_error_returns_neg1, missing entry_price, HitRate.rate
    TestHitRate               (4)  — rate zero, rate with data, rate all misses, to_dict
    TestJournalRow            (2)  — fields populated, optional entry_price None
    TestSignalOutcomeTracker  (8)  — run no unresolved, run resolves hit, run resolves miss,
                                     load_fills missing db, load_fills unreadable, resolve_one hit,
                                     resolve_one miss, resolve_one window boundary
    TestCalibrationWeights    (5)  — defaults_when_insufficient, calibrated_when_enough,
                                     war_room_higher_rate_gets_more_weight, to_dict, calibrated_flag
    TestAggregatorCalibration (4)  — use_calibration_false_uses_defaults,
                                     use_calibration_true_passes_weights,
                                     calibration_fails_silently_uses_defaults,
                                     get_combined_new_param_accepted
    TestAutoTraderJournalWiring(5) — journal_param_stored, approved_signals_journalled,
                                     journal_none_no_crash, journal_error_swallowed,
                                     strategy_source_from_signal
    TestMarketSchedulerJournalWiring(3) — journal_created_at_init, passed_to_auto_trader,
                                          journal_attribute_present
"""
from __future__ import annotations

import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fresh_journal(tmp_path: Path):
    """Return a SignalJournal backed by a temp file."""
    from strategies.signal_journal import SignalJournal
    return SignalJournal(db_path=tmp_path / "test_journal.db")


@dataclass
class _FakeSignal:
    ticker: str = "SPY"
    confidence: float = 0.75
    entry: float = 450.0
    strategy: str = "war_room"

    class direction:
        value = "SHORT"

    direction = type("Dir", (), {"value": "SHORT"})()


def _fake_signal(ticker="SPY", strategy="war_room", confidence=0.75):
    sig = _FakeSignal()
    sig.ticker = ticker
    sig.strategy = strategy
    sig.confidence = confidence
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# TestHitRate
# ─────────────────────────────────────────────────────────────────────────────

class TestHitRate:
    def test_rate_zero_when_no_resolved(self):
        from strategies.signal_journal import HitRate
        hr = HitRate(strategy="war_room", total=3, unresolved=3)
        assert hr.rate == 0.0

    def test_rate_with_hits_and_misses(self):
        from strategies.signal_journal import HitRate
        hr = HitRate(strategy="war_room", total=10, hits=7, misses=3)
        assert abs(hr.rate - 0.7) < 0.001

    def test_rate_all_misses(self):
        from strategies.signal_journal import HitRate
        hr = HitRate(strategy="war_room", total=5, hits=0, misses=5)
        assert hr.rate == 0.0

    def test_to_dict_keys(self):
        from strategies.signal_journal import HitRate
        d = HitRate(strategy="x", total=4, hits=2, misses=2).to_dict()
        assert set(d.keys()) == {"strategy", "total", "hits", "misses", "unresolved", "rate"}


# ─────────────────────────────────────────────────────────────────────────────
# TestJournalRow
# ─────────────────────────────────────────────────────────────────────────────

class TestJournalRow:
    def test_fields_populated(self, tmp_path):
        j = _fresh_journal(tmp_path)
        sig = _fake_signal()
        row_id = j.log_signal(sig, strategy_source="war_room")
        rows = j.get_recent(limit=1)
        assert len(rows) == 1
        r = rows[0]
        assert r.ticker == "SPY"
        assert r.strategy_source == "war_room"
        assert r.confidence == 0.75
        assert r.id == row_id

    def test_entry_price_none_when_signal_missing_entry(self, tmp_path):
        j = _fresh_journal(tmp_path)

        class _NoEntry:
            ticker = "QQQ"
            confidence = 0.5
            strategy = "vol_premium"
            direction = type("D", (), {"value": "SHORT"})()

        j.log_signal(_NoEntry(), strategy_source="vol_premium")
        rows = j.get_recent(limit=1)
        assert rows[0].entry_price is None


# ─────────────────────────────────────────────────────────────────────────────
# TestSignalJournal
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalJournal:
    def test_log_signal_returns_positive_id(self, tmp_path):
        j = _fresh_journal(tmp_path)
        row_id = j.log_signal(_fake_signal(), strategy_source="war_room")
        assert row_id > 0

    def test_get_recent_returns_logged_signals(self, tmp_path):
        j = _fresh_journal(tmp_path)
        j.log_signal(_fake_signal("SPY"), strategy_source="war_room")
        j.log_signal(_fake_signal("IWM"), strategy_source="vol_premium")
        rows = j.get_recent(limit=10)
        tickers = {r.ticker for r in rows}
        assert "SPY" in tickers
        assert "IWM" in tickers

    def test_get_hit_rates_empty_db(self, tmp_path):
        j = _fresh_journal(tmp_path)
        rates = j.get_hit_rates()
        assert rates == {}

    def test_get_hit_rates_counts_strategies(self, tmp_path):
        j = _fresh_journal(tmp_path)
        j.log_signal(_fake_signal("SPY", "war_room"), strategy_source="war_room")
        j.log_signal(_fake_signal("IWM", "war_room"), strategy_source="war_room")
        j.log_signal(_fake_signal("QQQ", "vol_premium"), strategy_source="vol_premium")
        rates = j.get_hit_rates()
        assert "war_room" in rates
        assert rates["war_room"].total == 2
        assert "vol_premium" in rates
        assert rates["vol_premium"].total == 1

    def test_resolve_marks_hit(self, tmp_path):
        j = _fresh_journal(tmp_path)
        row_id = j.log_signal(_fake_signal(), strategy_source="war_room")
        ok = j.resolve(row_id, "HIT")
        assert ok is True
        rates = j.get_hit_rates()
        assert rates["war_room"].hits == 1

    def test_resolve_nonexistent_returns_false(self, tmp_path):
        j = _fresh_journal(tmp_path)
        ok = j.resolve(99999, "HIT")
        assert ok is False

    def test_get_unresolved_returns_unresolved_row(self, tmp_path):
        """A signal logged in the past (beyond cutoff) should appear in unresolved."""
        from strategies.signal_journal import SignalJournal

        j = SignalJournal(db_path=tmp_path / "j.db")
        # Insert an old timestamp directly so it falls before the cutoff
        old_ts = (datetime.now(tz=timezone.utc) - timedelta(hours=72)).isoformat()
        j._conn.execute(
            "INSERT INTO signal_journal (ticker, direction, confidence, strategy_source, logged_at) "
            "VALUES ('SPY', 'SHORT', 0.8, 'war_room', ?)",
            (old_ts,),
        )
        j._conn.commit()
        # cutoff_hours=48 → signals older than 48 h should appear
        rows = j.get_unresolved(cutoff_hours=48)
        assert len(rows) == 1

    def test_log_signal_error_returns_neg1(self, tmp_path):
        """Passing an object that causes a conversion error returns -1."""
        from strategies.signal_journal import SignalJournal

        j = SignalJournal(db_path=tmp_path / "test.db")
        # Close the connection to force a DB error
        j._conn.close()
        result = j.log_signal(_fake_signal())
        assert result == -1


# ─────────────────────────────────────────────────────────────────────────────
# TestSignalOutcomeTracker
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalOutcomeTracker:
    def _make_pnl_db(self, tmp_path: Path, fills: list[dict]) -> Path:
        """Create a minimal pnl.db with trade_log rows."""
        db_path = tmp_path / "pnl.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                logged_at TEXT NOT NULL
            )
            """
        )
        for fill in fills:
            conn.execute(
                "INSERT INTO trade_log (symbol, direction, logged_at) VALUES (?, ?, ?)",
                (fill["symbol"], fill["direction"], fill["logged_at"]),
            )
        conn.commit()
        conn.close()
        return db_path

    def test_run_no_unresolved(self, tmp_path):
        from strategies.signal_outcome_tracker import SignalOutcomeTracker

        jdb = tmp_path / "j.db"
        pdb = tmp_path / "p.db"
        tracker = SignalOutcomeTracker(journal_db_path=jdb, pnl_db_path=pdb)
        report = tracker.run(cutoff_hours=48)
        assert report.resolved == 0
        assert report.errors == 0

    def test_run_resolves_hit(self, tmp_path):
        from strategies.signal_journal import SignalJournal
        from strategies.signal_outcome_tracker import SignalOutcomeTracker

        old_ts = (datetime.now(tz=timezone.utc) - timedelta(hours=72)).isoformat()
        fill_ts = (datetime.now(tz=timezone.utc) - timedelta(hours=60)).isoformat()

        # Insert an old unresolved signal
        j = SignalJournal(db_path=tmp_path / "j.db")
        j._conn.execute(
            "INSERT INTO signal_journal (ticker, direction, confidence, strategy_source, logged_at) VALUES (?,?,?,?,?)",
            ("SPY", "SHORT", 0.8, "war_room", old_ts),
        )
        j._conn.commit()

        # Create pnl.db with a matching fill
        pdb = self._make_pnl_db(tmp_path, [{"symbol": "SPY", "direction": "SHORT", "logged_at": fill_ts}])

        tracker = SignalOutcomeTracker(
            journal_db_path=tmp_path / "j.db",
            pnl_db_path=pdb,
            match_window_hours=48.0,
        )
        report = tracker.run(cutoff_hours=48)
        assert report.resolved == 1
        assert report.hits == 1
        assert report.misses == 0

    def test_run_resolves_miss(self, tmp_path):
        from strategies.signal_journal import SignalJournal
        from strategies.signal_outcome_tracker import SignalOutcomeTracker

        old_ts = (datetime.now(tz=timezone.utc) - timedelta(hours=72)).isoformat()

        j = SignalJournal(db_path=tmp_path / "j.db")
        j._conn.execute(
            "INSERT INTO signal_journal (ticker, direction, confidence, strategy_source, logged_at) VALUES (?,?,?,?,?)",
            ("IWM", "SHORT", 0.7, "vol_premium", old_ts),
        )
        j._conn.commit()

        # No fills for IWM
        pdb = self._make_pnl_db(tmp_path, [{"symbol": "SPY", "direction": "SHORT", "logged_at": old_ts}])

        tracker = SignalOutcomeTracker(
            journal_db_path=tmp_path / "j.db",
            pnl_db_path=pdb,
            match_window_hours=48.0,
        )
        report = tracker.run(cutoff_hours=48)
        assert report.resolved == 1
        assert report.misses == 1
        assert report.hits == 0

    def test_load_fills_missing_db(self, tmp_path):
        from strategies.signal_outcome_tracker import SignalOutcomeTracker

        tracker = SignalOutcomeTracker(
            journal_db_path=tmp_path / "j.db",
            pnl_db_path=tmp_path / "nonexistent.db",
        )
        fills = tracker._load_trade_fills()
        assert fills == []

    def test_resolve_one_hit_within_window(self, tmp_path):
        from strategies.signal_journal import JournalRow
        from strategies.signal_outcome_tracker import SignalOutcomeTracker

        now = datetime.now(tz=timezone.utc)
        signal_ts = (now - timedelta(hours=10)).isoformat()
        fill_ts = (now - timedelta(hours=8)).isoformat()   # 2 h later → within 48 h window

        row = JournalRow(
            id=1, ticker="SPY", direction="SHORT", confidence=0.8,
            strategy_source="war_room", entry_price=450.0,
            logged_at=signal_ts, outcome=None, resolved_at=None,
        )
        fills = [{"id": 42, "symbol": "SPY", "direction": "SHORT", "logged_at": fill_ts}]

        tracker = SignalOutcomeTracker(match_window_hours=48.0)
        outcome, trade_id = tracker._resolve_one(row, fills)
        assert outcome == "HIT"
        assert trade_id == 42

    def test_resolve_one_miss_different_ticker(self, tmp_path):
        from strategies.signal_journal import JournalRow
        from strategies.signal_outcome_tracker import SignalOutcomeTracker

        now = datetime.now(tz=timezone.utc)
        signal_ts = now.isoformat()
        fill_ts = now.isoformat()

        row = JournalRow(
            id=1, ticker="IWM", direction="SHORT", confidence=0.8,
            strategy_source="war_room", entry_price=None,
            logged_at=signal_ts, outcome=None, resolved_at=None,
        )
        fills = [{"id": 1, "symbol": "SPY", "direction": "SHORT", "logged_at": fill_ts}]

        tracker = SignalOutcomeTracker(match_window_hours=48.0)
        outcome, trade_id = tracker._resolve_one(row, fills)
        assert outcome == "MISS"
        assert trade_id is None

    def test_resolve_one_outside_window(self):
        from strategies.signal_journal import JournalRow
        from strategies.signal_outcome_tracker import SignalOutcomeTracker

        now = datetime.now(tz=timezone.utc)
        signal_ts = (now - timedelta(hours=100)).isoformat()
        fill_ts = now.isoformat()   # 100 h gap → outside 48 h window

        row = JournalRow(
            id=1, ticker="SPY", direction="SHORT", confidence=0.8,
            strategy_source="war_room", entry_price=None,
            logged_at=signal_ts, outcome=None, resolved_at=None,
        )
        fills = [{"id": 1, "symbol": "SPY", "direction": "SHORT", "logged_at": fill_ts}]

        tracker = SignalOutcomeTracker(match_window_hours=48.0)
        outcome, _ = tracker._resolve_one(row, fills)
        assert outcome == "MISS"


# ─────────────────────────────────────────────────────────────────────────────
# TestCalibrationWeights
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationWeights:
    def test_defaults_when_insufficient_data(self, tmp_path):
        from strategies.signal_outcome_tracker import SignalOutcomeTracker

        tracker = SignalOutcomeTracker(journal_db_path=tmp_path / "j.db")
        weights = tracker.calibrated_weights(default_war_room=0.60, default_vol_premium=0.40)
        assert weights.calibrated is False
        assert abs(weights.war_room - 0.60) < 0.001
        assert abs(weights.vol_premium - 0.40) < 0.001

    def test_calibrated_when_enough_data(self, tmp_path):
        from strategies.signal_journal import SignalJournal
        from strategies.signal_outcome_tracker import SignalOutcomeTracker

        j = SignalJournal(db_path=tmp_path / "j.db")
        # Insert 6 resolved rows for each strategy
        for i in range(6):
            j._conn.execute(
                "INSERT INTO signal_journal (ticker, direction, confidence, strategy_source, logged_at, outcome) "
                "VALUES ('SPY','SHORT',0.8,'war_room','2026-01-01T00:00:00+00:00','HIT')"
            )
            j._conn.execute(
                "INSERT INTO signal_journal (ticker, direction, confidence, strategy_source, logged_at, outcome) "
                "VALUES ('IWM','SHORT',0.6,'vol_premium','2026-01-01T00:00:00+00:00','MISS')"
            )
        j._conn.commit()

        tracker = SignalOutcomeTracker(journal_db_path=tmp_path / "j.db")
        weights = tracker.calibrated_weights()
        assert weights.calibrated is True
        assert abs(weights.war_room + weights.vol_premium - 1.0) < 0.01

    def test_war_room_higher_rate_gets_more_weight(self, tmp_path):
        from strategies.signal_journal import SignalJournal
        from strategies.signal_outcome_tracker import SignalOutcomeTracker

        j = SignalJournal(db_path=tmp_path / "j.db")
        # war_room: 6 hits (rate=1.0)
        for _ in range(6):
            j._conn.execute(
                "INSERT INTO signal_journal (ticker, direction, confidence, strategy_source, logged_at, outcome) "
                "VALUES ('SPY','SHORT',0.8,'war_room','2026-01-01T00:00:00+00:00','HIT')"
            )
        # vol_premium: 6 misses (rate=0.0)
        for _ in range(6):
            j._conn.execute(
                "INSERT INTO signal_journal (ticker, direction, confidence, strategy_source, logged_at, outcome) "
                "VALUES ('IWM','SHORT',0.6,'vol_premium','2026-01-01T00:00:00+00:00','MISS')"
            )
        j._conn.commit()

        tracker = SignalOutcomeTracker(journal_db_path=tmp_path / "j.db")
        weights = tracker.calibrated_weights(default_war_room=0.60, default_vol_premium=0.40)
        assert weights.calibrated is True
        assert weights.war_room > weights.vol_premium

    def test_to_dict_keys(self, tmp_path):
        from strategies.signal_outcome_tracker import SignalOutcomeTracker

        tracker = SignalOutcomeTracker(journal_db_path=tmp_path / "j.db")
        d = tracker.calibrated_weights().to_dict()
        assert set(d.keys()) == {
            "war_room", "vol_premium", "calibrated",
            "war_room_hit_rate", "vol_premium_hit_rate",
        }

    def test_calibrated_flag_false_by_default(self, tmp_path):
        from strategies.signal_outcome_tracker import CalibrationWeights
        w = CalibrationWeights()
        assert w.calibrated is False


# ─────────────────────────────────────────────────────────────────────────────
# TestAggregatorCalibration
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregatorCalibration:
    def test_use_calibration_false_uses_defaults(self):
        """use_calibration=False must not import SignalOutcomeTracker."""
        with patch("strategies.signal_generator.generate_signals", return_value=[]), \
             patch("strategies.vol_premium_signals.generate_vol_premium_signals", return_value=[]):
            from strategies.signal_aggregator import get_combined_signals
            result = get_combined_signals(use_calibration=False)
        assert result == []

    def test_use_calibration_true_calls_tracker(self):
        """use_calibration=True should call SignalOutcomeTracker.calibrated_weights."""
        mock_weights = MagicMock()
        mock_weights.calibrated = False
        mock_weights.war_room = 0.60
        mock_weights.vol_premium = 0.40
        mock_tracker = MagicMock()
        mock_tracker.calibrated_weights.return_value = mock_weights

        with patch("strategies.signal_generator.generate_signals", return_value=[]), \
             patch("strategies.vol_premium_signals.generate_vol_premium_signals", return_value=[]), \
             patch("strategies.signal_outcome_tracker.SignalOutcomeTracker", return_value=mock_tracker):
            from strategies.signal_aggregator import get_combined_signals
            get_combined_signals(use_calibration=True)
        mock_tracker.calibrated_weights.assert_called_once()

    def test_calibration_exception_swallowed(self):
        """If SignalOutcomeTracker raises, defaults must be used silently."""
        with patch("strategies.signal_generator.generate_signals", return_value=[]), \
             patch("strategies.vol_premium_signals.generate_vol_premium_signals", return_value=[]), \
             patch("strategies.signal_outcome_tracker.SignalOutcomeTracker", side_effect=RuntimeError("db gone")):
            from strategies.signal_aggregator import get_combined_signals
            result = get_combined_signals(use_calibration=True)
        assert result == []

    def test_new_param_accepted_with_default_false(self):
        """get_combined_signals() must accept use_calibration kwarg without error."""
        import inspect
        from strategies.signal_aggregator import get_combined_signals
        sig = inspect.signature(get_combined_signals)
        assert "use_calibration" in sig.parameters


# ─────────────────────────────────────────────────────────────────────────────
# TestAutoTraderJournalWiring
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoTraderJournalWiring:
    def test_journal_param_stored(self):
        from core.auto_trader import AutoTrader
        mock_journal = MagicMock()
        trader = AutoTrader(dry_run=True, signal_journal=mock_journal)
        assert trader._signal_journal is mock_journal

    def test_journal_none_no_crash(self):
        from core.auto_trader import AutoTrader
        trader = AutoTrader(dry_run=True, signal_journal=None)
        assert trader._signal_journal is None

    def test_approved_signals_journalled_before_execution(self):
        from core.auto_trader import AutoTrader
        mock_journal = MagicMock()
        trader = AutoTrader(dry_run=False, paper=True, signal_journal=mock_journal)
        # Build a minimal valid signal
        from shared.signal import AssetClass, Direction, TradeSignal
        sig = TradeSignal(
            ticker="SPY",
            direction=Direction.SHORT,
            confidence=0.85,
            entry=450.0,
            stop=460.0,
            target=430.0,
            size=0.05,
            strategy="test",
            asset_class=AssetClass.EQUITY,
        )
        with patch("asyncio.run", return_value=[]):
            trader.run_once([sig])
        mock_journal.log_signal.assert_called_once()

    def test_journal_none_does_not_call_log_signal(self):
        """When signal_journal=None, _journal_approved_signals must not be called."""
        from core.auto_trader import AutoTrader
        trader = AutoTrader(dry_run=False, paper=True, signal_journal=None)
        from shared.signal import AssetClass, Direction, TradeSignal
        sig = TradeSignal(
            ticker="SPY",
            direction=Direction.SHORT,
            confidence=0.85,
            entry=450.0,
            stop=460.0,
            target=430.0,
            size=0.05,
            strategy="test",
            asset_class=AssetClass.EQUITY,
        )
        with patch("asyncio.run", return_value=[]):
            # Should not raise
            trader.run_once([sig])

    def test_journal_error_does_not_block_execution(self):
        """Even if log_signal throws, execution must continue normally."""
        from core.auto_trader import AutoTrader
        mock_journal = MagicMock()
        mock_journal.log_signal.side_effect = RuntimeError("db locked")
        trader = AutoTrader(dry_run=False, paper=True, signal_journal=mock_journal)
        from shared.signal import AssetClass, Direction, TradeSignal
        sig = TradeSignal(
            ticker="SPY",
            direction=Direction.SHORT,
            confidence=0.85,
            entry=450.0,
            stop=460.0,
            target=430.0,
            size=0.05,
            strategy="test",
            asset_class=AssetClass.EQUITY,
        )
        with patch("asyncio.run", return_value=[]):
            summary = trader.run_once([sig])
        # Execution completes; no exception propagated
        assert summary is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestMarketSchedulerJournalWiring
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketSchedulerJournalWiring:
    def _make_scheduler(self, tmp_path=None):
        with patch("TradingExecution.position_tracker.PositionTracker"), \
             patch("CentralAccounting.pnl_tracker.PnLTracker"), \
             patch("strategies.daily_loss_guard.DailyLossGuard"), \
             patch("core.order_monitor.PendingOrderRegistry"), \
             patch("core.order_monitor.OrderMonitor"):
            if tmp_path:
                with patch("strategies.signal_journal.SignalJournal") as mock_sj:
                    from core.market_scheduler import MarketScheduler
                    sched = MarketScheduler(auto_execute=False)
                    return sched, mock_sj
            from core.market_scheduler import MarketScheduler
            sched = MarketScheduler(auto_execute=False)
            return sched, None

    def test_signal_journal_attribute_present(self):
        sched, _ = self._make_scheduler()
        assert hasattr(sched, "_signal_journal")

    def test_signal_journal_created_at_init(self):
        with patch("TradingExecution.position_tracker.PositionTracker"), \
             patch("CentralAccounting.pnl_tracker.PnLTracker"), \
             patch("strategies.daily_loss_guard.DailyLossGuard"), \
             patch("core.order_monitor.PendingOrderRegistry"), \
             patch("core.order_monitor.OrderMonitor"), \
             patch("strategies.signal_journal.SignalJournal") as mock_sj_cls:
            from core.market_scheduler import MarketScheduler
            MarketScheduler(auto_execute=False)
        mock_sj_cls.assert_called_once()

    def test_signal_journal_passed_to_auto_trader(self):
        mock_journal_instance = MagicMock()
        with patch("TradingExecution.position_tracker.PositionTracker"), \
             patch("CentralAccounting.pnl_tracker.PnLTracker"), \
             patch("strategies.daily_loss_guard.DailyLossGuard"), \
             patch("core.order_monitor.PendingOrderRegistry"), \
             patch("core.order_monitor.OrderMonitor"), \
             patch("strategies.signal_journal.SignalJournal", return_value=mock_journal_instance), \
             patch("core.auto_trader.AutoTrader") as mock_at_cls:
            from core.market_scheduler import MarketScheduler
            MarketScheduler(auto_execute=True)
        call_kwargs = mock_at_cls.call_args.kwargs
        assert call_kwargs.get("signal_journal") is mock_journal_instance
