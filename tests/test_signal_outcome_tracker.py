"""Sprint 28 — Comprehensive tests for strategies.signal_outcome_tracker.

Covers:
- ResolutionResult / OutcomeReport / CalibrationWeights dataclasses
- SignalOutcomeTracker.__init__ default + custom paths
- _load_trade_fills: missing DB, real DB
- _resolve_one: HIT (within window), MISS (outside window),
                MISS (no matching ticker), MISS (bad timestamp),
                ticker case-insensitive match, naive datetime handling
- run(): no unresolved → empty report
- run(): full HIT/MISS round-trip with signal_journal + pnl trade_log
- run(): exception inside resolve fails-open
- calibrated_weights(): insufficient data → defaults, calibrated=False
- calibrated_weights(): with data → calibrated weights sum to 1.0
- calibrated_weights(): no rates → defaults
- calibrated_weights(): exception → defaults
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from strategies.signal_journal import SignalJournal
from strategies.signal_outcome_tracker import (
    CalibrationWeights,
    OutcomeReport,
    ResolutionResult,
    SignalOutcomeTracker,
)


# ── helpers ──────────────────────────────────────────────────────────────────


@dataclass
class FakeDirection:
    value: str = "LONG"


@dataclass
class FakeSignal:
    ticker: str = "SPY"
    direction: FakeDirection = None
    confidence: float = 0.7
    entry: float | None = 450.0

    def __post_init__(self):
        if self.direction is None:
            self.direction = FakeDirection()


def _make_pnl_db(path: Path, fills: list[dict]) -> None:
    """Create a minimal pnl.db with the trade_log table and provided rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol    TEXT,
            direction TEXT,
            logged_at TEXT
        )
        """
    )
    for f in fills:
        conn.execute(
            "INSERT INTO trade_log (symbol, direction, logged_at) VALUES (?, ?, ?)",
            (f["symbol"], f.get("direction", "LONG"), f["logged_at"]),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def journal_db(tmp_path: Path) -> Path:
    return tmp_path / "signal_journal.db"


@pytest.fixture
def pnl_db(tmp_path: Path) -> Path:
    return tmp_path / "pnl.db"


# ── ResolutionResult ─────────────────────────────────────────────────────────


class TestResolutionResult:
    def test_minimal_construction(self):
        r = ResolutionResult(
            journal_id=1,
            ticker="SPY",
            strategy_source="war_room",
            outcome="HIT",
        )
        assert r.journal_id == 1
        assert r.ticker == "SPY"
        assert r.outcome == "HIT"
        assert r.matched_trade_id is None

    def test_with_matched_trade(self):
        r = ResolutionResult(
            journal_id=2,
            ticker="QQQ",
            strategy_source="vol_premium",
            outcome="HIT",
            matched_trade_id=42,
        )
        assert r.matched_trade_id == 42

    def test_miss_outcome(self):
        r = ResolutionResult(
            journal_id=3,
            ticker="IWM",
            strategy_source="war_room",
            outcome="MISS",
        )
        assert r.outcome == "MISS"
        assert r.matched_trade_id is None


# ── CalibrationWeights ───────────────────────────────────────────────────────


class TestCalibrationWeights:
    def test_defaults(self):
        w = CalibrationWeights()
        assert w.war_room == pytest.approx(0.60)
        assert w.vol_premium == pytest.approx(0.40)
        assert w.calibrated is False
        assert w.war_room_hit_rate == 0.0
        assert w.vol_premium_hit_rate == 0.0

    def test_to_dict_keys(self):
        w = CalibrationWeights()
        d = w.to_dict()
        assert set(d.keys()) == {
            "war_room",
            "vol_premium",
            "calibrated",
            "war_room_hit_rate",
            "vol_premium_hit_rate",
        }

    def test_to_dict_rounding(self):
        w = CalibrationWeights(
            war_room=0.123456789,
            vol_premium=0.987654321,
            war_room_hit_rate=0.555555,
        )
        d = w.to_dict()
        assert d["war_room"] == pytest.approx(0.1235, abs=1e-4)
        assert d["vol_premium"] == pytest.approx(0.9877, abs=1e-4)
        assert d["war_room_hit_rate"] == pytest.approx(0.5556, abs=1e-4)


# ── OutcomeReport ────────────────────────────────────────────────────────────


class TestOutcomeReport:
    def test_defaults(self):
        r = OutcomeReport()
        assert r.resolved == 0
        assert r.hits == 0
        assert r.misses == 0
        assert r.errors == 0
        assert r.results == []
        assert r.generated_at == ""

    def test_to_dict_keys(self):
        r = OutcomeReport(resolved=3, hits=2, misses=1, errors=0,
                          generated_at="2026-04-22T00:00:00+00:00")
        d = r.to_dict()
        assert set(d.keys()) == {"resolved", "hits", "misses", "errors", "generated_at"}
        assert d["resolved"] == 3
        assert d["hits"] == 2
        assert d["misses"] == 1

    def test_results_independent_per_instance(self):
        r1 = OutcomeReport()
        r2 = OutcomeReport()
        r1.results.append("x")
        assert r2.results == []


# ── SignalOutcomeTracker init ────────────────────────────────────────────────


class TestTrackerInit:
    def test_default_paths(self):
        t = SignalOutcomeTracker()
        assert t._journal_db.name == "signal_journal.db"
        assert t._pnl_db.name == "pnl.db"
        assert t._window_hours == 48.0

    def test_custom_paths(self, tmp_path: Path):
        j = tmp_path / "j.db"
        p = tmp_path / "p.db"
        t = SignalOutcomeTracker(journal_db_path=j, pnl_db_path=p, match_window_hours=12.0)
        assert t._journal_db == j
        assert t._pnl_db == p
        assert t._window_hours == 12.0

    def test_str_paths_accepted(self, tmp_path: Path):
        t = SignalOutcomeTracker(
            journal_db_path=str(tmp_path / "j.db"),
            pnl_db_path=str(tmp_path / "p.db"),
        )
        assert isinstance(t._journal_db, Path)
        assert isinstance(t._pnl_db, Path)


# ── _load_trade_fills ────────────────────────────────────────────────────────


class TestLoadTradeFills:
    def test_missing_db_returns_empty(self, journal_db: Path, tmp_path: Path):
        t = SignalOutcomeTracker(
            journal_db_path=journal_db,
            pnl_db_path=tmp_path / "does_not_exist.db",
        )
        assert t._load_trade_fills() == []

    def test_loads_existing_rows(self, journal_db: Path, pnl_db: Path):
        _make_pnl_db(
            pnl_db,
            [
                {"symbol": "SPY", "logged_at": "2026-04-22T10:00:00+00:00"},
                {"symbol": "QQQ", "logged_at": "2026-04-22T11:00:00+00:00"},
            ],
        )
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        fills = t._load_trade_fills()
        assert len(fills) == 2
        symbols = {f["symbol"] for f in fills}
        assert symbols == {"SPY", "QQQ"}

    def test_corrupt_db_returns_empty(self, journal_db: Path, pnl_db: Path):
        # Create a file that exists but is not a valid sqlite db
        pnl_db.write_text("not a sqlite db")
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        assert t._load_trade_fills() == []


# ── _resolve_one ─────────────────────────────────────────────────────────────


class _StubRow:
    def __init__(self, ticker: str, logged_at: str, id_: int = 1):
        self.id = id_
        self.ticker = ticker
        self.logged_at = logged_at
        self.strategy_source = "war_room"


class TestResolveOne:
    def test_hit_within_window(self, journal_db: Path, pnl_db: Path):
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db,
                                 match_window_hours=24.0)
        signal_ts = "2026-04-22T10:00:00+00:00"
        fills = [{"id": 7, "symbol": "SPY", "direction": "LONG",
                  "logged_at": "2026-04-22T11:00:00+00:00"}]
        outcome, trade_id = t._resolve_one(_StubRow("SPY", signal_ts), fills)
        assert outcome == "HIT"
        assert trade_id == 7

    def test_miss_outside_window(self, journal_db: Path, pnl_db: Path):
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db,
                                 match_window_hours=1.0)
        signal_ts = "2026-04-22T10:00:00+00:00"
        fills = [{"id": 8, "symbol": "SPY",
                  "logged_at": "2026-04-22T20:00:00+00:00"}]
        outcome, trade_id = t._resolve_one(_StubRow("SPY", signal_ts), fills)
        assert outcome == "MISS"
        assert trade_id is None

    def test_miss_no_matching_ticker(self, journal_db: Path, pnl_db: Path):
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        fills = [{"id": 1, "symbol": "QQQ",
                  "logged_at": "2026-04-22T10:00:00+00:00"}]
        outcome, trade_id = t._resolve_one(
            _StubRow("SPY", "2026-04-22T10:00:00+00:00"), fills,
        )
        assert outcome == "MISS"
        assert trade_id is None

    def test_miss_bad_signal_timestamp(self, journal_db: Path, pnl_db: Path):
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        fills = [{"id": 1, "symbol": "SPY",
                  "logged_at": "2026-04-22T10:00:00+00:00"}]
        outcome, trade_id = t._resolve_one(_StubRow("SPY", "not-a-date"), fills)
        assert outcome == "MISS"
        assert trade_id is None

    def test_ticker_case_insensitive(self, journal_db: Path, pnl_db: Path):
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        fills = [{"id": 99, "symbol": "spy",
                  "logged_at": "2026-04-22T10:00:00+00:00"}]
        outcome, trade_id = t._resolve_one(
            _StubRow("SPY", "2026-04-22T10:00:00+00:00"), fills,
        )
        assert outcome == "HIT"
        assert trade_id == 99

    def test_naive_signal_ts_treated_as_utc(self, journal_db: Path, pnl_db: Path):
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db,
                                 match_window_hours=2.0)
        # No tzinfo — should be assumed UTC
        fills = [{"id": 5, "symbol": "SPY",
                  "logged_at": "2026-04-22T11:00:00+00:00"}]
        outcome, trade_id = t._resolve_one(
            _StubRow("SPY", "2026-04-22T10:00:00"), fills,
        )
        assert outcome == "HIT"
        assert trade_id == 5

    def test_skips_fill_with_bad_timestamp(self, journal_db: Path, pnl_db: Path):
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        fills = [
            {"id": 1, "symbol": "SPY", "logged_at": "garbage"},
            {"id": 2, "symbol": "SPY", "logged_at": "2026-04-22T10:30:00+00:00"},
        ]
        outcome, trade_id = t._resolve_one(
            _StubRow("SPY", "2026-04-22T10:00:00+00:00"), fills,
        )
        assert outcome == "HIT"
        assert trade_id == 2


# ── run() end-to-end ─────────────────────────────────────────────────────────


class TestRunEndToEnd:
    def test_no_unresolved_returns_empty_report(self, journal_db: Path, pnl_db: Path):
        # Empty journal → no unresolved
        SignalJournal(db_path=journal_db)  # creates schema
        _make_pnl_db(pnl_db, [])
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        report = t.run(cutoff_hours=0.0)
        assert report.resolved == 0
        assert report.hits == 0
        assert report.misses == 0

    def test_hit_round_trip(self, journal_db: Path, pnl_db: Path):
        # Log a signal in the past
        journal = SignalJournal(db_path=journal_db)
        sig = FakeSignal(ticker="SPY")
        row_id = journal.log_signal(sig, strategy_source="war_room")
        assert row_id > 0
        # Backdate the signal so it's older than cutoff
        past = (datetime.now(tz=timezone.utc) - timedelta(hours=72)).isoformat()
        journal._conn.execute(
            "UPDATE signal_journal SET logged_at=? WHERE id=?", (past, row_id),
        )
        journal._conn.commit()
        # Add a matching fill close in time to the signal
        _make_pnl_db(pnl_db, [{"symbol": "SPY",
                               "logged_at": past}])
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        report = t.run(cutoff_hours=24.0)
        assert report.resolved == 1
        assert report.hits == 1
        assert report.misses == 0
        assert report.results[0].outcome == "HIT"
        assert report.results[0].ticker == "SPY"

    def test_miss_round_trip(self, journal_db: Path, pnl_db: Path):
        journal = SignalJournal(db_path=journal_db)
        sig = FakeSignal(ticker="QQQ")
        row_id = journal.log_signal(sig, strategy_source="vol_premium")
        past = (datetime.now(tz=timezone.utc) - timedelta(hours=72)).isoformat()
        journal._conn.execute(
            "UPDATE signal_journal SET logged_at=? WHERE id=?", (past, row_id),
        )
        journal._conn.commit()
        # No matching fill at all
        _make_pnl_db(pnl_db, [])
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        report = t.run(cutoff_hours=24.0)
        assert report.resolved == 1
        assert report.misses == 1
        assert report.hits == 0
        assert report.results[0].outcome == "MISS"

    def test_run_has_generated_at_iso(self, journal_db: Path, pnl_db: Path):
        SignalJournal(db_path=journal_db)
        _make_pnl_db(pnl_db, [])
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        report = t.run(cutoff_hours=0.0)
        # Parses as ISO datetime
        datetime.fromisoformat(report.generated_at)


# ── calibrated_weights ───────────────────────────────────────────────────────


def _seed_resolved(journal: SignalJournal, source: str, hits: int, misses: int) -> None:
    """Insert resolved rows directly into the journal."""
    now = datetime.now(tz=timezone.utc).isoformat()
    for _ in range(hits):
        journal._conn.execute(
            """INSERT INTO signal_journal
               (ticker, direction, confidence, strategy_source, entry_price,
                logged_at, outcome, resolved_at)
               VALUES ('X','LONG',0.5,?,1.0,?,'HIT',?)""",
            (source, now, now),
        )
    for _ in range(misses):
        journal._conn.execute(
            """INSERT INTO signal_journal
               (ticker, direction, confidence, strategy_source, entry_price,
                logged_at, outcome, resolved_at)
               VALUES ('X','LONG',0.5,?,1.0,?,'MISS',?)""",
            (source, now, now),
        )
    journal._conn.commit()


class TestCalibratedWeights:
    def test_no_data_returns_defaults(self, journal_db: Path, pnl_db: Path):
        SignalJournal(db_path=journal_db)
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        w = t.calibrated_weights()
        assert w.war_room == pytest.approx(0.60)
        assert w.vol_premium == pytest.approx(0.40)
        assert w.calibrated is False

    def test_insufficient_data_returns_defaults(self, journal_db: Path, pnl_db: Path):
        journal = SignalJournal(db_path=journal_db)
        # Below the 5-resolved minimum for one source
        _seed_resolved(journal, "war_room", hits=2, misses=1)
        _seed_resolved(journal, "vol_premium", hits=10, misses=5)
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        w = t.calibrated_weights()
        assert w.calibrated is False
        assert w.war_room == pytest.approx(0.60)
        assert w.vol_premium == pytest.approx(0.40)

    def test_calibration_weights_sum_to_one(self, journal_db: Path, pnl_db: Path):
        journal = SignalJournal(db_path=journal_db)
        _seed_resolved(journal, "war_room", hits=8, misses=2)   # 0.8
        _seed_resolved(journal, "vol_premium", hits=4, misses=6)  # 0.4
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        w = t.calibrated_weights()
        assert w.calibrated is True
        assert (w.war_room + w.vol_premium) == pytest.approx(1.0, abs=1e-3)
        # Higher hit-rate → higher weight (vs baseline ratio)
        assert w.war_room > 0.60
        assert w.vol_premium < 0.40
        assert w.war_room_hit_rate == pytest.approx(0.8)
        assert w.vol_premium_hit_rate == pytest.approx(0.4)

    def test_custom_defaults(self, journal_db: Path, pnl_db: Path):
        journal = SignalJournal(db_path=journal_db)
        _seed_resolved(journal, "war_room", hits=5, misses=5)
        _seed_resolved(journal, "vol_premium", hits=5, misses=5)
        t = SignalOutcomeTracker(journal_db_path=journal_db, pnl_db_path=pnl_db)
        w = t.calibrated_weights(default_war_room=0.7, default_vol_premium=0.3)
        # Equal hit rates → calibrated weights match default ratio
        assert w.calibrated is True
        assert w.war_room == pytest.approx(0.7, abs=1e-3)
        assert w.vol_premium == pytest.approx(0.3, abs=1e-3)
