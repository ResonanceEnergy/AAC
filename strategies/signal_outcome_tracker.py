"""strategies/signal_outcome_tracker.py — Sprint 15: Outcome Resolution.

Cross-references the ``signal_journal`` against ``trade_log`` to determine
which signals were acted on (HIT) and which were not (MISS).  The result is
used to compute per-strategy accuracy and to produce calibrated aggregator
weights.

HIT definition
--------------
A signal is a HIT if a row exists in ``trade_log`` for the same ticker
within ``match_window_hours`` of the signal's ``logged_at`` timestamp.
The fill direction is not checked — the mere fact that a trade was executed
for the same ticker indicates the signal influenced an order.

MISS definition
---------------
No matching trade fill in ``trade_log`` within the window.

Calibrated weights
------------------
Given hit rates for ``war_room`` and ``vol_premium``, the calibrator
scales each source's default weight proportionally to its hit rate.  If
insufficient data exists for a source, the original default weight is kept.

Usage::

    tracker = SignalOutcomeTracker()
    report = tracker.run(cutoff_hours=48)
    weights = tracker.calibrated_weights(
        report,
        default_war_room=0.60,
        default_vol_premium=0.40,
    )
"""
from __future__ import annotations

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import structlog

_log = structlog.get_logger(__name__)

# ── Default paths ─────────────────────────────────────────────────────────────

_DEFAULT_JOURNAL_DB = Path("data") / "signal_journal.db"
_DEFAULT_PNL_DB = Path("CentralAccounting") / "data" / "pnl.db"

# Minimum resolved signals needed before calibration overrides defaults.
_MIN_RESOLVED_FOR_CALIBRATION = 5


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ResolutionResult:
    """Outcome of a single signal resolution attempt."""

    journal_id: int
    ticker: str
    strategy_source: str
    outcome: str          # "HIT" or "MISS"
    matched_trade_id: Optional[int] = None


@dataclass
class CalibrationWeights:
    """Aggregator weights derived from observed hit rates."""

    war_room: float = 0.60
    vol_premium: float = 0.40
    calibrated: bool = False          # False → defaults used
    war_room_hit_rate: float = 0.0
    vol_premium_hit_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "war_room": round(self.war_room, 4),
            "vol_premium": round(self.vol_premium, 4),
            "calibrated": self.calibrated,
            "war_room_hit_rate": round(self.war_room_hit_rate, 4),
            "vol_premium_hit_rate": round(self.vol_premium_hit_rate, 4),
        }


@dataclass
class OutcomeReport:
    """Summary of a single outcome-resolution run."""

    resolved: int = 0
    hits: int = 0
    misses: int = 0
    errors: int = 0
    results: list[ResolutionResult] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "resolved": self.resolved,
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "generated_at": self.generated_at,
        }


# ── SignalOutcomeTracker ──────────────────────────────────────────────────────

class SignalOutcomeTracker:
    """Resolves unresolved signal journal entries against the trade log.

    Args:
        journal_db_path: Path to ``signal_journal.db``.
        pnl_db_path:     Path to PnLTracker's ``pnl.db``.
        match_window_hours: A fill within this many hours of the signal is a HIT.
    """

    def __init__(
        self,
        journal_db_path: Optional[str | Path] = None,
        pnl_db_path: Optional[str | Path] = None,
        match_window_hours: float = 48.0,
    ) -> None:
        self._journal_db = Path(journal_db_path) if journal_db_path else _DEFAULT_JOURNAL_DB
        self._pnl_db = Path(pnl_db_path) if pnl_db_path else _DEFAULT_PNL_DB
        self._window_hours = match_window_hours

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, cutoff_hours: float = 48.0) -> OutcomeReport:
        """Resolve all unresolved signals older than ``cutoff_hours``.

        Pulls unresolved rows from signal_journal, fetches trade_log rows
        from pnl.db, cross-references, and updates journal outcomes.

        Args:
            cutoff_hours: Only resolve signals older than this many hours.

        Returns:
            ``OutcomeReport`` summarising what was resolved.
        """
        report = OutcomeReport(
            generated_at=datetime.now(tz=timezone.utc).isoformat()
        )

        try:
            from strategies.signal_journal import SignalJournal  # noqa: PLC0415

            journal = SignalJournal(db_path=self._journal_db)
            unresolved = journal.get_unresolved(cutoff_hours=cutoff_hours)

            if not unresolved:
                _log.info("outcome_tracker_no_unresolved")
                return report

            trade_fills = self._load_trade_fills()

            for row in unresolved:
                try:
                    outcome, trade_id = self._resolve_one(row, trade_fills)
                    success = journal.resolve(row.id, outcome)
                    if success:
                        report.resolved += 1
                        if outcome == "HIT":
                            report.hits += 1
                        else:
                            report.misses += 1
                        report.results.append(ResolutionResult(
                            journal_id=row.id,
                            ticker=row.ticker,
                            strategy_source=row.strategy_source,
                            outcome=outcome,
                            matched_trade_id=trade_id,
                        ))
                    else:
                        report.errors += 1
                except Exception as exc:
                    _log.warning(
                        "outcome_tracker_resolve_error",
                        journal_id=row.id,
                        error=str(exc),
                    )
                    report.errors += 1

            _log.info(
                "outcome_tracker_complete",
                resolved=report.resolved,
                hits=report.hits,
                misses=report.misses,
            )
        except Exception as exc:
            _log.warning("outcome_tracker_run_failed", error=str(exc))

        return report

    def calibrated_weights(
        self,
        report: OutcomeReport | None = None,
        *,
        default_war_room: float = 0.60,
        default_vol_premium: float = 0.40,
    ) -> CalibrationWeights:
        """Compute aggregator weights calibrated to observed hit rates.

        Reads current hit rates from the signal journal.  If either strategy
        has fewer than ``_MIN_RESOLVED_FOR_CALIBRATION`` resolved signals, the
        original default weights are returned (``calibrated=False``).

        The calibrated weight for each source is:

            w_i = default_i * (1 + hit_rate_i) / normaliser

        where normaliser ensures the two weights still sum to 1.0.

        Args:
            report:              OutcomeReport from the latest ``run()`` call
                                 (unused currently, reserved for extension).
            default_war_room:    Default weight to use when data is insufficient.
            default_vol_premium: Default weight to use when data is insufficient.

        Returns:
            ``CalibrationWeights`` instance.
        """
        weights = CalibrationWeights(
            war_room=default_war_room,
            vol_premium=default_vol_premium,
        )
        try:
            from strategies.signal_journal import SignalJournal  # noqa: PLC0415

            journal = SignalJournal(db_path=self._journal_db)
            rates = journal.get_hit_rates()

            wr_data = rates.get("war_room")
            vp_data = rates.get("vol_premium")

            wr_resolved = (wr_data.hits + wr_data.misses) if wr_data else 0
            vp_resolved = (vp_data.hits + vp_data.misses) if vp_data else 0

            if wr_resolved < _MIN_RESOLVED_FOR_CALIBRATION or vp_resolved < _MIN_RESOLVED_FOR_CALIBRATION:
                _log.info(
                    "calibration_insufficient_data",
                    wr_resolved=wr_resolved,
                    vp_resolved=vp_resolved,
                    minimum=_MIN_RESOLVED_FOR_CALIBRATION,
                )
                return weights

            wr_rate = wr_data.rate  # type: ignore[union-attr]
            vp_rate = vp_data.rate  # type: ignore[union-attr]

            # Scale default weights by (1 + hit_rate), then normalise to sum=1.
            raw_wr = default_war_room * (1.0 + wr_rate)
            raw_vp = default_vol_premium * (1.0 + vp_rate)
            total = raw_wr + raw_vp
            if total <= 0.0:
                return weights

            weights.war_room = round(raw_wr / total, 4)
            weights.vol_premium = round(raw_vp / total, 4)
            weights.calibrated = True
            weights.war_room_hit_rate = round(wr_rate, 4)
            weights.vol_premium_hit_rate = round(vp_rate, 4)

            _log.info(
                "calibration_weights_computed",
                war_room=weights.war_room,
                vol_premium=weights.vol_premium,
                wr_hit_rate=wr_rate,
                vp_hit_rate=vp_rate,
            )
        except Exception as exc:
            _log.warning("calibration_weights_failed", error=str(exc))

        return weights

    # ── private ───────────────────────────────────────────────────────────────

    def _load_trade_fills(self) -> list[dict]:
        """Load all trade_log rows from pnl.db.

        Returns an empty list if the database is missing or unreadable.
        """
        try:
            if not self._pnl_db.exists():
                return []
            conn = sqlite3.connect(str(self._pnl_db), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT id, symbol, direction, logged_at FROM trade_log"
            )
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return rows
        except Exception as exc:
            _log.warning("outcome_tracker_load_fills_failed", error=str(exc))
            return []

    def _resolve_one(
        self,
        row: object,
        trade_fills: list[dict],
    ) -> tuple[str, Optional[int]]:
        """Determine outcome for a single journal row.

        Searches ``trade_fills`` for a row whose ``symbol`` matches the
        signal's ``ticker`` and whose ``logged_at`` is within
        ``self._window_hours`` of the signal's ``logged_at``.

        Returns:
            A 2-tuple ``(outcome, matched_trade_id)`` where outcome is
            "HIT" or "MISS" and matched_trade_id is the trade_log id if hit.
        """
        ticker: str = getattr(row, "ticker", "")
        logged_at_str: str = getattr(row, "logged_at", "")

        try:
            signal_ts = datetime.fromisoformat(logged_at_str)
            if signal_ts.tzinfo is None:
                signal_ts = signal_ts.replace(tzinfo=timezone.utc)
        except ValueError:
            return "MISS", None

        window = timedelta(hours=self._window_hours)

        for fill in trade_fills:
            if fill.get("symbol", "").upper() != ticker.upper():
                continue
            try:
                fill_ts = datetime.fromisoformat(fill["logged_at"])
                if fill_ts.tzinfo is None:
                    fill_ts = fill_ts.replace(tzinfo=timezone.utc)
            except (ValueError, KeyError):
                continue

            if abs(fill_ts - signal_ts) <= window:
                return "HIT", fill.get("id")

        return "MISS", None
