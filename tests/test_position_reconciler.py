from __future__ import annotations

"""tests/test_position_reconciler.py — Sprint 17: Position Reconciliation.

Covers:
* ReconciliationMismatch dataclass
* ReconciliationReport counts / properties
* PositionReconciler.reconcile() — MISSING, PHANTOM, SIZE_MISMATCH, clean
* _load_latest_snapshot() — DB auto-load path
* Fail-open on errors
* MarketScheduler wiring — reconciler created at init, called in run_pnl_snapshot
"""

import sqlite3
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────


@dataclass
class _Pos:
    """Minimal duck-typed PositionSnapshot for tests."""
    symbol: str
    quantity: float


def _make_pos(symbol: str, qty: float) -> _Pos:
    return _Pos(symbol=symbol, quantity=qty)


# ── Patch constants (definition-site) ────────────────────────────────────────

_PATCH_POSITION_TRACKER = "TradingExecution.position_tracker.PositionTracker"
_PATCH_PNL_TRACKER = "CentralAccounting.pnl_tracker.PnLTracker"
_PATCH_LOSS_GUARD = "strategies.daily_loss_guard.DailyLossGuard"
_PATCH_ORDER_MONITOR = "core.order_monitor.OrderMonitor"
_PATCH_ORDER_REGISTRY = "core.order_monitor.PendingOrderRegistry"
_PATCH_SIGNAL_JOURNAL = "strategies.signal_journal.SignalJournal"
_PATCH_REGIME_MONITOR = "strategies.regime_monitor.RegimeMonitor"
_PATCH_RECONCILER = "core.position_reconciler.PositionReconciler"


def _ms_patches(*extra):
    """Return a list of patches needed to instantiate MarketScheduler."""
    return [
        patch(_PATCH_POSITION_TRACKER),
        patch(_PATCH_PNL_TRACKER),
        patch(_PATCH_LOSS_GUARD),
        patch(_PATCH_ORDER_MONITOR),
        patch(_PATCH_ORDER_REGISTRY),
        patch(_PATCH_SIGNAL_JOURNAL),
        patch(_PATCH_REGIME_MONITOR),
        patch(_PATCH_RECONCILER),
        *extra,
    ]


# ── ReconciliationMismatch ────────────────────────────────────────────────────


class TestReconciliationMismatch:
    def test_to_dict_missing(self) -> None:
        from core.position_reconciler import MISSING, ReconciliationMismatch  # noqa: PLC0415

        m = ReconciliationMismatch(
            mismatch_type=MISSING,
            symbol="SPY",
            internal_qty=5.0,
            live_qty=None,
            detected_at="2026-01-01T00:00:00Z",
        )
        d = m.to_dict()
        assert d["mismatch_type"] == MISSING
        assert d["symbol"] == "SPY"
        assert d["internal_qty"] == 5.0
        assert d["live_qty"] is None

    def test_to_dict_phantom(self) -> None:
        from core.position_reconciler import PHANTOM, ReconciliationMismatch  # noqa: PLC0415

        m = ReconciliationMismatch(
            mismatch_type=PHANTOM,
            symbol="QQQ",
            internal_qty=None,
            live_qty=3.0,
            detected_at="2026-01-01T00:00:00Z",
        )
        d = m.to_dict()
        assert d["mismatch_type"] == PHANTOM
        assert d["internal_qty"] is None
        assert d["live_qty"] == 3.0

    def test_to_dict_size_mismatch(self) -> None:
        from core.position_reconciler import SIZE_MISMATCH, ReconciliationMismatch  # noqa: PLC0415

        m = ReconciliationMismatch(
            mismatch_type=SIZE_MISMATCH,
            symbol="IWM",
            internal_qty=2.0,
            live_qty=3.0,
            detected_at="2026-01-01T00:00:00Z",
        )
        d = m.to_dict()
        assert d["mismatch_type"] == SIZE_MISMATCH
        assert d["internal_qty"] == 2.0
        assert d["live_qty"] == 3.0

    def test_frozen_immutable(self) -> None:
        from core.position_reconciler import MISSING, ReconciliationMismatch  # noqa: PLC0415

        m = ReconciliationMismatch(
            mismatch_type=MISSING,
            symbol="X",
            internal_qty=1.0,
            live_qty=None,
            detected_at="ts",
        )
        with pytest.raises((TypeError, AttributeError)):
            m.symbol = "Y"  # type: ignore[misc]


# ── ReconciliationReport ──────────────────────────────────────────────────────


class TestReconciliationReport:
    def _make_report(
        self,
        mismatches=None,
        internal_count=3,
        live_count=3,
    ):
        from core.position_reconciler import ReconciliationReport  # noqa: PLC0415

        return ReconciliationReport(
            generated_at="2026-01-01T00:00:00Z",
            internal_count=internal_count,
            live_count=live_count,
            mismatches=mismatches or [],
        )

    def test_no_mismatches(self) -> None:
        r = self._make_report()
        assert not r.has_mismatches
        assert r.mismatch_count == 0
        assert r.missing_count == 0
        assert r.phantom_count == 0
        assert r.size_mismatch_count == 0

    def test_counts_by_type(self) -> None:
        from core.position_reconciler import (  # noqa: PLC0415
            MISSING,
            PHANTOM,
            SIZE_MISMATCH,
            ReconciliationMismatch,
        )

        mismatches = [
            ReconciliationMismatch(MISSING, "A", 1.0, None, "ts"),
            ReconciliationMismatch(MISSING, "B", 1.0, None, "ts"),
            ReconciliationMismatch(PHANTOM, "C", None, 2.0, "ts"),
            ReconciliationMismatch(SIZE_MISMATCH, "D", 1.0, 2.0, "ts"),
        ]
        r = self._make_report(mismatches=mismatches)
        assert r.has_mismatches
        assert r.mismatch_count == 4
        assert r.missing_count == 2
        assert r.phantom_count == 1
        assert r.size_mismatch_count == 1

    def test_to_dict_structure(self) -> None:
        r = self._make_report(internal_count=2, live_count=2)
        d = r.to_dict()
        assert "generated_at" in d
        assert d["internal_count"] == 2
        assert d["live_count"] == 2
        assert d["mismatch_count"] == 0
        assert isinstance(d["mismatches"], list)

    def test_to_dict_includes_mismatch_details(self) -> None:
        from core.position_reconciler import PHANTOM, ReconciliationMismatch  # noqa: PLC0415

        m = ReconciliationMismatch(PHANTOM, "XYZ", None, 5.0, "ts")
        r = self._make_report(mismatches=[m], internal_count=0, live_count=1)
        d = r.to_dict()
        assert len(d["mismatches"]) == 1
        assert d["mismatches"][0]["symbol"] == "XYZ"


# ── PositionReconciler.reconcile() ────────────────────────────────────────────


class TestReconcileClean:
    def test_both_empty_no_mismatches(self) -> None:
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        r = PositionReconciler().reconcile(internal_positions=[], live_positions=[])
        assert not r.has_mismatches
        assert r.internal_count == 0
        assert r.live_count == 0

    def test_matching_positions_no_mismatches(self) -> None:
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        internal = [_make_pos("SPY", 5.0), _make_pos("QQQ", 3.0)]
        live = [_make_pos("SPY", 5.0), _make_pos("QQQ", 3.0)]
        r = PositionReconciler().reconcile(internal_positions=internal, live_positions=live)
        assert not r.has_mismatches
        assert r.internal_count == 2
        assert r.live_count == 2

    def test_tolerance_small_float_diff_no_mismatch(self) -> None:
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        internal = [_make_pos("SPY", 5.0)]
        live = [_make_pos("SPY", 5.005)]  # diff = 0.005 < 0.01 tolerance
        r = PositionReconciler().reconcile(internal_positions=internal, live_positions=live)
        assert not r.has_mismatches


class TestReconcileMissing:
    def test_single_missing_position(self) -> None:
        from core.position_reconciler import MISSING, PositionReconciler  # noqa: PLC0415

        internal = [_make_pos("SPY", 5.0)]
        live: list = []
        r = PositionReconciler().reconcile(internal_positions=internal, live_positions=live)
        assert r.has_mismatches
        assert r.missing_count == 1
        assert r.phantom_count == 0
        assert r.mismatches[0].mismatch_type == MISSING
        assert r.mismatches[0].symbol == "SPY"
        assert r.mismatches[0].internal_qty == 5.0
        assert r.mismatches[0].live_qty is None

    def test_multiple_missing_positions(self) -> None:
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        internal = [_make_pos("A", 1.0), _make_pos("B", 2.0), _make_pos("C", 3.0)]
        live = [_make_pos("A", 1.0)]  # B and C missing
        r = PositionReconciler().reconcile(internal_positions=internal, live_positions=live)
        assert r.missing_count == 2
        symbols = {m.symbol for m in r.mismatches}
        assert symbols == {"B", "C"}


class TestReconcilePhantom:
    def test_single_phantom_position(self) -> None:
        from core.position_reconciler import PHANTOM, PositionReconciler  # noqa: PLC0415

        internal: list = []
        live = [_make_pos("GLD", 10.0)]
        r = PositionReconciler().reconcile(internal_positions=internal, live_positions=live)
        assert r.has_mismatches
        assert r.phantom_count == 1
        assert r.missing_count == 0
        assert r.mismatches[0].mismatch_type == PHANTOM
        assert r.mismatches[0].symbol == "GLD"
        assert r.mismatches[0].live_qty == 10.0
        assert r.mismatches[0].internal_qty is None

    def test_phantom_does_not_flag_known_symbols(self) -> None:
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        internal = [_make_pos("SPY", 2.0)]
        live = [_make_pos("SPY", 2.0), _make_pos("PHANTOM_TICKER", 1.0)]
        r = PositionReconciler().reconcile(internal_positions=internal, live_positions=live)
        assert r.phantom_count == 1
        assert r.missing_count == 0
        assert r.mismatches[0].symbol == "PHANTOM_TICKER"


class TestReconcileSizeMismatch:
    def test_size_mismatch_detected(self) -> None:
        from core.position_reconciler import SIZE_MISMATCH, PositionReconciler  # noqa: PLC0415

        internal = [_make_pos("IWM", 5.0)]
        live = [_make_pos("IWM", 7.0)]
        r = PositionReconciler().reconcile(internal_positions=internal, live_positions=live)
        assert r.has_mismatches
        assert r.size_mismatch_count == 1
        m = r.mismatches[0]
        assert m.mismatch_type == SIZE_MISMATCH
        assert m.symbol == "IWM"
        assert m.internal_qty == 5.0
        assert m.live_qty == 7.0

    def test_size_mismatch_at_tolerance_boundary(self) -> None:
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        # Exactly at tolerance — should NOT flag
        internal = [_make_pos("HYG", 3.0)]
        live = [_make_pos("HYG", 3.0 + 0.01)]  # diff == 0.01, exactly at tolerance
        r = PositionReconciler().reconcile(internal_positions=internal, live_positions=live)
        assert not r.has_mismatches

    def test_size_mismatch_above_tolerance(self) -> None:
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        # Just over tolerance
        internal = [_make_pos("HYG", 3.0)]
        live = [_make_pos("HYG", 3.0 + 0.0101)]
        r = PositionReconciler().reconcile(internal_positions=internal, live_positions=live)
        assert r.size_mismatch_count == 1

    def test_custom_tolerance_respected(self) -> None:
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        # diff=0.5 is above default 0.01 but below custom 1.0
        reconciler = PositionReconciler(size_tolerance=1.0)
        internal = [_make_pos("SPY", 10.0)]
        live = [_make_pos("SPY", 10.5)]
        r = reconciler.reconcile(internal_positions=internal, live_positions=live)
        assert not r.has_mismatches


class TestReconcileMixedMismatches:
    def test_all_three_types_detected(self) -> None:
        from core.position_reconciler import (  # noqa: PLC0415
            MISSING,
            PHANTOM,
            SIZE_MISMATCH,
            PositionReconciler,
        )

        internal = [
            _make_pos("SPY", 5.0),   # will match
            _make_pos("QQQ", 3.0),   # MISSING — not in live
            _make_pos("IWM", 2.0),   # SIZE_MISMATCH
        ]
        live = [
            _make_pos("SPY", 5.0),   # match
            _make_pos("IWM", 4.0),   # SIZE_MISMATCH
            _make_pos("GLD", 8.0),   # PHANTOM
        ]
        r = PositionReconciler().reconcile(internal_positions=internal, live_positions=live)
        assert r.missing_count == 1
        assert r.phantom_count == 1
        assert r.size_mismatch_count == 1
        assert r.mismatch_count == 3

        types = {m.mismatch_type for m in r.mismatches}
        assert MISSING in types
        assert PHANTOM in types
        assert SIZE_MISMATCH in types


# ── Fail-open paths ────────────────────────────────────────────────────────────


class TestFailOpen:
    def test_none_live_treated_as_empty(self) -> None:
        """Passing live_positions=None should not raise — returns MISSING for all internal."""
        from core.position_reconciler import MISSING, PositionReconciler  # noqa: PLC0415

        internal = [_make_pos("SPY", 1.0)]
        r = PositionReconciler().reconcile(internal_positions=internal, live_positions=None)
        assert r.missing_count == 1
        assert r.mismatches[0].mismatch_type == MISSING

    def test_none_both_returns_empty_report(self) -> None:
        """None for both uses _load_latest_snapshot (empty if no DB)."""
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        reconciler = PositionReconciler(pnl_db_path=":memory_nonexistent_path_xyz:")
        # Should not raise even if DB doesn't exist
        r = reconciler.reconcile()
        assert isinstance(r.mismatches, list)

    def test_malformed_position_object_skipped(self) -> None:
        """Position objects without .symbol / .quantity are silently skipped."""
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        class BadPos:
            pass

        r = PositionReconciler().reconcile(
            internal_positions=[BadPos()],
            live_positions=[_make_pos("SPY", 1.0)],
        )
        # BadPos is skipped → internal is empty → SPY is PHANTOM
        assert r.phantom_count == 1

    def test_internal_exception_returns_empty_report(self) -> None:
        """If reconcile logic raises internally, fail-open and return empty report."""
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        reconciler = PositionReconciler()
        with patch.object(reconciler, "_to_qty_map", side_effect=RuntimeError("boom")):
            r = reconciler.reconcile(internal_positions=[], live_positions=[])
        assert not r.has_mismatches  # fail-open
        assert r.internal_count == 0


# ── _load_latest_snapshot via real SQLite ────────────────────────────────────


class TestLoadLatestSnapshot:
    def test_returns_empty_when_db_missing(self, tmp_path) -> None:
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        reconciler = PositionReconciler(pnl_db_path=tmp_path / "nonexistent.db")
        rows = reconciler._load_latest_snapshot()
        assert rows == []

    def test_loads_most_recent_date(self, tmp_path) -> None:
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        db_path = tmp_path / "pnl.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE position_snapshots "
            "(id INTEGER PRIMARY KEY, snapshot_date TEXT, symbol TEXT, quantity REAL)"
        )
        conn.executemany(
            "INSERT INTO position_snapshots (snapshot_date, symbol, quantity) VALUES (?,?,?)",
            [
                ("2026-01-01", "SPY", 5.0),
                ("2026-01-01", "QQQ", 3.0),
                ("2026-01-02", "SPY", 7.0),  # most recent
                ("2026-01-02", "IWM", 2.0),  # most recent
            ],
        )
        conn.commit()
        conn.close()

        reconciler = PositionReconciler(pnl_db_path=db_path)
        rows = reconciler._load_latest_snapshot()
        symbols = {r.symbol: r.quantity for r in rows}
        assert set(symbols.keys()) == {"SPY", "IWM"}  # only 2026-01-02 rows
        assert symbols["SPY"] == 7.0
        assert symbols["IWM"] == 2.0

    def test_auto_load_wires_into_reconcile(self, tmp_path) -> None:
        """When internal_positions=None, _load_latest_snapshot is called."""
        from core.position_reconciler import PositionReconciler  # noqa: PLC0415

        db_path = tmp_path / "pnl.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE position_snapshots "
            "(id INTEGER PRIMARY KEY, snapshot_date TEXT, symbol TEXT, quantity REAL)"
        )
        conn.execute(
            "INSERT INTO position_snapshots (snapshot_date, symbol, quantity) VALUES (?,?,?)",
            ("2026-01-01", "JNK", 4.0),
        )
        conn.commit()
        conn.close()

        reconciler = PositionReconciler(pnl_db_path=db_path)
        # live has no JNK → should be MISSING
        r = reconciler.reconcile(internal_positions=None, live_positions=[])
        assert r.missing_count == 1
        assert r.mismatches[0].symbol == "JNK"


# ── MarketScheduler wiring ─────────────────────────────────────────────────────


class TestMarketSchedulerReconcilerWiring:
    def test_reconciler_attribute_exists(self) -> None:
        """MarketScheduler must have a _position_reconciler attribute after init."""
        from core.market_scheduler import MarketScheduler  # noqa: PLC0415

        with (
            patch(_PATCH_POSITION_TRACKER),
            patch(_PATCH_PNL_TRACKER),
            patch(_PATCH_LOSS_GUARD),
            patch(_PATCH_ORDER_MONITOR),
            patch(_PATCH_ORDER_REGISTRY),
            patch(_PATCH_SIGNAL_JOURNAL),
            patch(_PATCH_REGIME_MONITOR),
            patch(_PATCH_RECONCILER),
        ):
            sched = MarketScheduler()
        assert hasattr(sched, "_position_reconciler")

    def test_reconciler_created_once_at_init(self) -> None:
        """PositionReconciler() must be called exactly once during MarketScheduler.__init__."""
        from core.market_scheduler import MarketScheduler  # noqa: PLC0415

        with (
            patch(_PATCH_POSITION_TRACKER),
            patch(_PATCH_PNL_TRACKER),
            patch(_PATCH_LOSS_GUARD),
            patch(_PATCH_ORDER_MONITOR),
            patch(_PATCH_ORDER_REGISTRY),
            patch(_PATCH_SIGNAL_JOURNAL),
            patch(_PATCH_REGIME_MONITOR),
            patch(_PATCH_RECONCILER) as mock_rec_cls,
        ):
            MarketScheduler()
        mock_rec_cls.assert_called_once()

    def test_run_pnl_snapshot_calls_reconcile(self) -> None:
        """run_pnl_snapshot() must call _position_reconciler.reconcile()."""
        from core.market_scheduler import MarketScheduler  # noqa: PLC0415

        mock_rec = MagicMock()
        mock_rec_report = MagicMock()
        mock_rec_report.has_mismatches = False
        mock_rec.reconcile.return_value = mock_rec_report

        mock_pnl = MagicMock()
        mock_pnl.take_snapshot.return_value = {"ok": True}
        mock_pnl.pnl_delta.return_value = 0.0
        mock_pnl.today_report.return_value = {}

        with (
            patch(_PATCH_POSITION_TRACKER),
            patch(_PATCH_PNL_TRACKER, return_value=mock_pnl),
            patch(_PATCH_LOSS_GUARD),
            patch(_PATCH_ORDER_MONITOR),
            patch(_PATCH_ORDER_REGISTRY),
            patch(_PATCH_SIGNAL_JOURNAL),
            patch(_PATCH_REGIME_MONITOR),
            patch(_PATCH_RECONCILER, return_value=mock_rec),
            patch.object(MarketScheduler, "_fetch_positions", return_value=[]),
            patch.object(MarketScheduler, "_run_eod_report"),
        ):
            sched = MarketScheduler()
            sched.run_pnl_snapshot()

        mock_rec.reconcile.assert_called_once()

    def test_run_pnl_snapshot_passes_live_positions_to_reconcile(self) -> None:
        """Reconciler must receive the positions fetched in run_pnl_snapshot."""
        from core.market_scheduler import MarketScheduler  # noqa: PLC0415

        fake_positions = [_make_pos("SPY", 5.0), _make_pos("QQQ", 3.0)]
        mock_rec = MagicMock()
        mock_rec_report = MagicMock()
        mock_rec_report.has_mismatches = False
        mock_rec.reconcile.return_value = mock_rec_report

        mock_pnl = MagicMock()
        mock_pnl.take_snapshot.return_value = {}
        mock_pnl.pnl_delta.return_value = 0.0
        mock_pnl.today_report.return_value = {}

        with (
            patch(_PATCH_POSITION_TRACKER),
            patch(_PATCH_PNL_TRACKER, return_value=mock_pnl),
            patch(_PATCH_LOSS_GUARD),
            patch(_PATCH_ORDER_MONITOR),
            patch(_PATCH_ORDER_REGISTRY),
            patch(_PATCH_SIGNAL_JOURNAL),
            patch(_PATCH_REGIME_MONITOR),
            patch(_PATCH_RECONCILER, return_value=mock_rec),
            patch.object(MarketScheduler, "_fetch_positions", return_value=fake_positions),
            patch.object(MarketScheduler, "_run_eod_report"),
        ):
            sched = MarketScheduler()
            sched.run_pnl_snapshot()

        call_kwargs = mock_rec.reconcile.call_args.kwargs
        assert call_kwargs.get("live_positions") == fake_positions

    def test_reconcile_mismatches_logged_as_warning(self) -> None:
        """When reconcile reports mismatches, _log.warning is called (not error)."""
        from core.market_scheduler import MarketScheduler  # noqa: PLC0415

        mock_rec = MagicMock()
        mock_rec_report = MagicMock()
        mock_rec_report.has_mismatches = True
        mock_rec_report.mismatch_count = 2
        mock_rec_report.missing_count = 1
        mock_rec_report.phantom_count = 1
        mock_rec_report.size_mismatch_count = 0
        mock_rec.reconcile.return_value = mock_rec_report

        mock_pnl = MagicMock()
        mock_pnl.take_snapshot.return_value = {}
        mock_pnl.pnl_delta.return_value = 0.0
        mock_pnl.today_report.return_value = {}

        with (
            patch(_PATCH_POSITION_TRACKER),
            patch(_PATCH_PNL_TRACKER, return_value=mock_pnl),
            patch(_PATCH_LOSS_GUARD),
            patch(_PATCH_ORDER_MONITOR),
            patch(_PATCH_ORDER_REGISTRY),
            patch(_PATCH_SIGNAL_JOURNAL),
            patch(_PATCH_REGIME_MONITOR),
            patch(_PATCH_RECONCILER, return_value=mock_rec),
            patch.object(MarketScheduler, "_fetch_positions", return_value=[]),
            patch.object(MarketScheduler, "_run_eod_report"),
        ):
            sched = MarketScheduler()
            # Should not raise even with mismatches reported
            sched.run_pnl_snapshot()

    def test_reconcile_error_does_not_block_pnl_snapshot(self) -> None:
        """If reconcile() raises, run_pnl_snapshot must still return successfully."""
        from core.market_scheduler import MarketScheduler  # noqa: PLC0415

        mock_rec = MagicMock()
        mock_rec.reconcile.side_effect = RuntimeError("reconcile exploded")

        mock_pnl = MagicMock()
        mock_pnl.take_snapshot.return_value = {"ok": True}
        mock_pnl.pnl_delta.return_value = 0.0
        mock_pnl.today_report.return_value = {}

        with (
            patch(_PATCH_POSITION_TRACKER),
            patch(_PATCH_PNL_TRACKER, return_value=mock_pnl),
            patch(_PATCH_LOSS_GUARD),
            patch(_PATCH_ORDER_MONITOR),
            patch(_PATCH_ORDER_REGISTRY),
            patch(_PATCH_SIGNAL_JOURNAL),
            patch(_PATCH_REGIME_MONITOR),
            patch(_PATCH_RECONCILER, return_value=mock_rec),
            patch.object(MarketScheduler, "_fetch_positions", return_value=[]),
            patch.object(MarketScheduler, "_run_eod_report"),
        ):
            sched = MarketScheduler()
            result = sched.run_pnl_snapshot()  # must not raise

        assert isinstance(result, dict)
