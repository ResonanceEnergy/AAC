"""strategies/signal_journal.py — Sprint 15: Signal Journal.

Persists every generated ``TradeSignal`` to a SQLite journal so that outcome
tracking and strategy calibration can retrospectively measure which signals
were acted on (became fills in ``trade_log``).

Schema
------
``signal_journal`` table:
    id              INTEGER PK AUTOINCREMENT
    ticker          TEXT
    direction       TEXT    (LONG / SHORT / LONG_PUT / FLAT)
    confidence      REAL
    strategy_source TEXT    (e.g. "war_room", "vol_premium", "combined")
    entry_price     REAL    (signal's .entry field)
    logged_at       TEXT    ISO-8601 UTC
    outcome         TEXT    NULL | "HIT" | "MISS"
    resolved_at     TEXT    NULL | ISO-8601 UTC

Outcome definitions
-------------------
HIT  — a matching trade fill was found in ``trade_log`` for the same ticker
       within the resolution window (default 48 h), confirming the signal
       was acted on.
MISS — no matching fill found within the window.
NULL — not yet resolved (signal is too recent or resolution not yet run).

Usage::

    journal = SignalJournal()
    journal.log_signal(signal, strategy_source="war_room")
    ...
    rows = journal.get_unresolved(cutoff_hours=48)
    journal.resolve(row_id=1, outcome="HIT")
    rates = journal.get_hit_rates()  # {"war_room": HitRate(...), ...}
"""
from __future__ import annotations

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

_log = structlog.get_logger(__name__)

# ── Default paths ─────────────────────────────────────────────────────────────

_DEFAULT_DB = Path("data") / "signal_journal.db"

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS signal_journal (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    confidence      REAL    NOT NULL DEFAULT 0.0,
    strategy_source TEXT    NOT NULL DEFAULT 'unknown',
    entry_price     REAL,
    logged_at       TEXT    NOT NULL,
    outcome         TEXT,
    resolved_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_sj_logged_at ON signal_journal(logged_at);
CREATE INDEX IF NOT EXISTS idx_sj_outcome   ON signal_journal(outcome);
"""


# ── HitRate ───────────────────────────────────────────────────────────────────

@dataclass
class HitRate:
    """Per-strategy hit rate summary."""

    strategy: str
    total: int = 0
    hits: int = 0
    misses: int = 0
    unresolved: int = 0

    @property
    def rate(self) -> float:
        """Fraction of resolved signals that were hits.  0.0 if no resolved data."""
        resolved = self.hits + self.misses
        return self.hits / resolved if resolved > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total": self.total,
            "hits": self.hits,
            "misses": self.misses,
            "unresolved": self.unresolved,
            "rate": round(self.rate, 4),
        }


# ── JournalRow (lightweight read-only view) ───────────────────────────────────

@dataclass
class JournalRow:
    """A single row from the signal_journal table."""

    id: int
    ticker: str
    direction: str
    confidence: float
    strategy_source: str
    entry_price: Optional[float]
    logged_at: str
    outcome: Optional[str]
    resolved_at: Optional[str]


# ── SignalJournal ─────────────────────────────────────────────────────────────

class SignalJournal:
    """SQLite-backed journal of generated trade signals.

    Thread-safety: SQLite WAL mode; single writer at a time is sufficient for
    the scheduler use-case.  Uses ``check_same_thread=False`` so the journal
    can be passed between scheduler and AutoTrader threads.

    Args:
        db_path: Path to the SQLite database file.  Created if absent.
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._apply_schema()

    def _apply_schema(self) -> None:
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ── write ─────────────────────────────────────────────────────────────────

    def log_signal(
        self,
        signal: object,
        strategy_source: str = "unknown",
    ) -> int:
        """Insert a ``TradeSignal`` into the journal.

        Args:
            signal:          Any object with ``.ticker``, ``.direction``,
                             ``.confidence``, ``.entry`` attributes.
            strategy_source: Label identifying which strategy produced it
                             (e.g. "war_room", "vol_premium", "combined").

        Returns:
            The new row id.  Returns -1 on any error.
        """
        try:
            ticker = str(getattr(signal, "ticker", "?"))
            direction = str(getattr(getattr(signal, "direction", "?"), "value", "?"))
            confidence = float(getattr(signal, "confidence", 0.0))
            entry_price_raw = getattr(signal, "entry", None)
            entry_price = float(entry_price_raw) if entry_price_raw is not None else None
            logged_at = datetime.now(tz=timezone.utc).isoformat()

            cur = self._conn.execute(
                """
                INSERT INTO signal_journal
                    (ticker, direction, confidence, strategy_source, entry_price, logged_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ticker, direction, confidence, strategy_source, entry_price, logged_at),
            )
            self._conn.commit()
            row_id = cur.lastrowid or -1
            _log.debug(
                "signal_journalled",
                ticker=ticker,
                direction=direction,
                strategy=strategy_source,
                id=row_id,
            )
            return row_id
        except Exception as exc:
            _log.warning("signal_journal_log_failed", error=str(exc))
            return -1

    def resolve(self, row_id: int, outcome: str) -> bool:
        """Mark a journal entry as HIT or MISS.

        Args:
            row_id:  The signal_journal.id to update.
            outcome: "HIT" or "MISS".

        Returns:
            True if a row was updated; False otherwise.
        """
        try:
            resolved_at = datetime.now(tz=timezone.utc).isoformat()
            cur = self._conn.execute(
                "UPDATE signal_journal SET outcome=?, resolved_at=? WHERE id=?",
                (outcome, resolved_at, row_id),
            )
            self._conn.commit()
            return cur.rowcount > 0
        except Exception as exc:
            _log.warning("signal_journal_resolve_failed", row_id=row_id, error=str(exc))
            return False

    # ── read ──────────────────────────────────────────────────────────────────

    def get_unresolved(self, cutoff_hours: float = 48.0) -> list[JournalRow]:
        """Return unresolved signals older than ``cutoff_hours``.

        Signals younger than the cutoff have not had enough time for a fill
        to appear in trade_log and should not yet be resolved.

        Args:
            cutoff_hours: Minimum age (in hours) of unresolved signals to return.

        Returns:
            List of ``JournalRow`` objects.
        """
        try:
            from datetime import timedelta  # noqa: PLC0415

            cutoff_ts = (
                datetime.now(tz=timezone.utc) - timedelta(hours=cutoff_hours)
            ).isoformat()

            cur = self._conn.execute(
                """
                SELECT id, ticker, direction, confidence, strategy_source,
                       entry_price, logged_at, outcome, resolved_at
                FROM signal_journal
                WHERE outcome IS NULL
                  AND logged_at <= ?
                ORDER BY logged_at ASC
                """,
                (cutoff_ts,),
            )
            return [_row_to_journal(r) for r in cur.fetchall()]
        except Exception as exc:
            _log.warning("signal_journal_get_unresolved_failed", error=str(exc))
            return []

    def get_recent(self, limit: int = 50) -> list[JournalRow]:
        """Return the most recent ``limit`` signal rows (any outcome)."""
        try:
            cur = self._conn.execute(
                """
                SELECT id, ticker, direction, confidence, strategy_source,
                       entry_price, logged_at, outcome, resolved_at
                FROM signal_journal
                ORDER BY logged_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [_row_to_journal(r) for r in cur.fetchall()]
        except Exception as exc:
            _log.warning("signal_journal_get_recent_failed", error=str(exc))
            return []

    def get_hit_rates(self) -> dict[str, HitRate]:
        """Compute hit rates per strategy from all resolved rows.

        Returns:
            ``{strategy_source: HitRate}`` for every strategy that has at
            least one recorded signal.  Includes unresolved counts.
        """
        try:
            cur = self._conn.execute(
                """
                SELECT strategy_source,
                       COUNT(*)                                          AS total,
                       SUM(CASE WHEN outcome = 'HIT'  THEN 1 ELSE 0 END) AS hits,
                       SUM(CASE WHEN outcome = 'MISS' THEN 1 ELSE 0 END) AS misses,
                       SUM(CASE WHEN outcome IS NULL   THEN 1 ELSE 0 END) AS unresolved
                FROM signal_journal
                GROUP BY strategy_source
                """
            )
            result: dict[str, HitRate] = {}
            for row in cur.fetchall():
                source = row["strategy_source"]
                result[source] = HitRate(
                    strategy=source,
                    total=int(row["total"] or 0),
                    hits=int(row["hits"] or 0),
                    misses=int(row["misses"] or 0),
                    unresolved=int(row["unresolved"] or 0),
                )
            return result
        except Exception as exc:
            _log.warning("signal_journal_get_hit_rates_failed", error=str(exc))
            return {}

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception as exc:  # noqa: BLE001
            _log = __import__('structlog').get_logger() if '_log' not in dir() else _log
            _log.warning('suppressed_exception', error=str(exc))
# ── helpers ───────────────────────────────────────────────────────────────────

def _row_to_journal(row: sqlite3.Row) -> JournalRow:
    return JournalRow(
        id=row["id"],
        ticker=row["ticker"],
        direction=row["direction"],
        confidence=float(row["confidence"]),
        strategy_source=row["strategy_source"],
        entry_price=float(row["entry_price"]) if row["entry_price"] is not None else None,
        logged_at=row["logged_at"],
        outcome=row["outcome"],
        resolved_at=row["resolved_at"],
    )
