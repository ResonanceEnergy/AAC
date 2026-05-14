from __future__ import annotations

"""core/position_reconciler.py — Sprint 17: Position Reconciliation.

Compares the system's internal position view (most-recent ``position_snapshots``
in ``pnl.db``) against the live exchange positions (``PositionSnapshot`` list
from PositionTracker) and flags three classes of discrepancy:

* ``MISSING``       — symbol is in the internal snapshot but absent from IBKR
* ``PHANTOM``       — symbol exists in IBKR but is absent from the snapshot
* ``SIZE_MISMATCH`` — both views agree the position exists but quantities differ
                       by more than ``size_tolerance``

Usage::

    from core.position_reconciler import PositionReconciler
    report = PositionReconciler().reconcile(
        internal_positions=pnl_tracker.latest_positions(),
        live_positions=position_tracker_list,
    )
    if report.has_mismatches:
        for m in report.mismatches:
            log.warning("reconciliation_mismatch", **m.to_dict())
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

_log = structlog.get_logger(__name__)

_DEFAULT_PNL_DB = Path("CentralAccounting") / "data" / "pnl.db"

# How much quantity difference (absolute) is allowed before flagging SIZE_MISMATCH.
_DEFAULT_SIZE_TOLERANCE: float = 0.01


# ── Mismatch types ─────────────────────────────────────────────────────────

MISSING = "MISSING"          # in snapshot, not in IBKR
PHANTOM = "PHANTOM"          # in IBKR, not in snapshot
SIZE_MISMATCH = "SIZE_MISMATCH"  # quantities differ


# ── Data transfer objects ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ReconciliationMismatch:
    """A single discrepancy between internal and live positions."""

    mismatch_type: str        # MISSING | PHANTOM | SIZE_MISMATCH
    symbol: str
    internal_qty: Optional[float]   # None for PHANTOM
    live_qty: Optional[float]       # None for MISSING
    detected_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mismatch_type": self.mismatch_type,
            "symbol": self.symbol,
            "internal_qty": self.internal_qty,
            "live_qty": self.live_qty,
            "detected_at": self.detected_at,
        }


@dataclass
class ReconciliationReport:
    """Full reconciliation result for one run."""

    generated_at: str
    internal_count: int
    live_count: int
    mismatches: list[ReconciliationMismatch] = field(default_factory=list)

    @property
    def has_mismatches(self) -> bool:
        return bool(self.mismatches)

    @property
    def mismatch_count(self) -> int:
        return len(self.mismatches)

    @property
    def missing_count(self) -> int:
        return sum(1 for m in self.mismatches if m.mismatch_type == MISSING)

    @property
    def phantom_count(self) -> int:
        return sum(1 for m in self.mismatches if m.mismatch_type == PHANTOM)

    @property
    def size_mismatch_count(self) -> int:
        return sum(1 for m in self.mismatches if m.mismatch_type == SIZE_MISMATCH)

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "internal_count": self.internal_count,
            "live_count": self.live_count,
            "mismatch_count": self.mismatch_count,
            "missing_count": self.missing_count,
            "phantom_count": self.phantom_count,
            "size_mismatch_count": self.size_mismatch_count,
            "mismatches": [m.to_dict() for m in self.mismatches],
        }


# ── PositionReconciler ─────────────────────────────────────────────────────


class PositionReconciler:
    """Reconciles internal position snapshots against live exchange positions.

    Args:
        pnl_db_path:     Path to the PnLTracker SQLite database.  Defaults to
                         ``CentralAccounting/data/pnl.db``.
        size_tolerance:  Absolute quantity difference allowed before flagging
                         a SIZE_MISMATCH (default 0.01 — handles float rounding).
    """

    def __init__(
        self,
        pnl_db_path: Optional[Path | str] = None,
        size_tolerance: float = _DEFAULT_SIZE_TOLERANCE,
    ) -> None:
        self._db_path = Path(pnl_db_path) if pnl_db_path else _DEFAULT_PNL_DB
        self._size_tolerance = size_tolerance

    # ── public API ────────────────────────────────────────────────────────

    def reconcile(
        self,
        internal_positions: Optional[list] = None,
        live_positions: Optional[list] = None,
    ) -> ReconciliationReport:
        """Compare internal and live positions; return a ``ReconciliationReport``.

        If ``internal_positions`` is ``None``, the most-recent positions are
        loaded from ``pnl.db`` automatically.  If ``live_positions`` is ``None``
        an empty list is used (all internal positions become MISSING).

        Never raises — all errors are caught and logged; on error a report
        with zero mismatches is returned so reconciliation failures never
        block the trading loop.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        try:
            internal = internal_positions
            if internal is None:
                internal = self._load_latest_snapshot()

            live = live_positions if live_positions is not None else []

            # Build lookup maps: symbol → quantity
            internal_map = self._to_qty_map(internal, key_attr="symbol", qty_attr="quantity")
            live_map = self._to_qty_map(live, key_attr="symbol", qty_attr="quantity")

            mismatches: list[ReconciliationMismatch] = []

            # MISSING: in internal but not in live
            for sym, qty in internal_map.items():
                if sym not in live_map:
                    mismatches.append(
                        ReconciliationMismatch(
                            mismatch_type=MISSING,
                            symbol=sym,
                            internal_qty=qty,
                            live_qty=None,
                            detected_at=now,
                        )
                    )

            # PHANTOM: in live but not in internal
            for sym, qty in live_map.items():
                if sym not in internal_map:
                    mismatches.append(
                        ReconciliationMismatch(
                            mismatch_type=PHANTOM,
                            symbol=sym,
                            internal_qty=None,
                            live_qty=qty,
                            detected_at=now,
                        )
                    )

            # SIZE_MISMATCH: both present but quantities differ
            for sym in internal_map:
                if sym in live_map:
                    diff = abs(internal_map[sym] - live_map[sym])
                    if diff > self._size_tolerance:
                        mismatches.append(
                            ReconciliationMismatch(
                                mismatch_type=SIZE_MISMATCH,
                                symbol=sym,
                                internal_qty=internal_map[sym],
                                live_qty=live_map[sym],
                                detected_at=now,
                            )
                        )

            report = ReconciliationReport(
                generated_at=now,
                internal_count=len(internal_map),
                live_count=len(live_map),
                mismatches=mismatches,
            )

            if report.has_mismatches:
                _log.warning(
                    "position_reconciliation_mismatches",
                    total=report.mismatch_count,
                    missing=report.missing_count,
                    phantom=report.phantom_count,
                    size_mismatch=report.size_mismatch_count,
                )
            else:
                _log.info(
                    "position_reconciliation_clean",
                    internal=report.internal_count,
                    live=report.live_count,
                )

            return report

        except Exception as exc:  # noqa: BLE001
            _log.warning("reconciliation_error", error=str(exc))
            return ReconciliationReport(
                generated_at=now,
                internal_count=0,
                live_count=0,
                mismatches=[],
            )

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_qty_map(
        positions: list,
        key_attr: str,
        qty_attr: str,
    ) -> dict[str, float]:
        """Convert a list of position objects to ``{symbol: quantity}``."""
        result: dict[str, float] = {}
        for pos in positions:
            try:
                sym = getattr(pos, key_attr)
                qty = float(getattr(pos, qty_attr))
                result[sym] = qty
            except Exception:  # noqa: BLE001
                pass
        return result

    def _load_latest_snapshot(self) -> list[_SnapshotRow]:
        """Load the most-recent position_snapshots row per symbol from pnl.db."""
        if not self._db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(
                    """
                    SELECT symbol, quantity
                    FROM position_snapshots
                    WHERE snapshot_date = (
                        SELECT MAX(snapshot_date) FROM position_snapshots
                    )
                    """
                )
                return [_SnapshotRow(symbol=r["symbol"], quantity=r["quantity"]) for r in cur]
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            _log.warning("snapshot_load_error", error=str(exc))
            return []


@dataclass
class _SnapshotRow:
    """Minimal row returned by _load_latest_snapshot (duck-typed as PositionSnapshot)."""

    symbol: str
    quantity: float
