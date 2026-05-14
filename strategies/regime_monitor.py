from __future__ import annotations

"""regime_monitor.py — Sprint 16: Regime Change Alerting.

Persists composite-score regime state (CALM / WATCH / ELEVATED / CRISIS) to
SQLite.  Each call to ``record()`` stores the current regime and returns a
``RegimeTransition`` if the regime has changed since the previous record.

Usage::

    from strategies.regime_monitor import RegimeMonitor, Regime
    monitor = RegimeMonitor()
    transition = monitor.record(composite_score=72.5)
    if transition:
        print(f"Regime changed: {transition.prev_regime} → {transition.new_regime}")
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import structlog

_log = structlog.get_logger(__name__)

_DEFAULT_DB = Path("data") / "regime_monitor.db"

# ── Regime enum ────────────────────────────────────────────────────────────


class Regime(str, Enum):
    """Market stress regime, derived from composite score thresholds.

    Thresholds mirror the War Room engine:
      CALM     < 30
      WATCH    30 – 50
      ELEVATED 50 – 70
      CRISIS   ≥ 70
    """

    CALM = "CALM"
    WATCH = "WATCH"
    ELEVATED = "ELEVATED"
    CRISIS = "CRISIS"

    @classmethod
    def from_score(cls, score: float) -> Regime:
        """Return the regime that ``score`` maps to."""
        if score < 30.0:
            return cls.CALM
        if score < 50.0:
            return cls.WATCH
        if score < 70.0:
            return cls.ELEVATED
        return cls.CRISIS

    @property
    def severity(self) -> int:
        """Ordinal severity (0 = calmest, 3 = most severe)."""
        return {"CALM": 0, "WATCH": 1, "ELEVATED": 2, "CRISIS": 3}[self.value]


# ── Data transfer objects ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RegimeRecord:
    """A single row from ``regime_history``."""

    id: int
    regime: str
    composite_score: float
    detected_at: str


@dataclass(frozen=True)
class RegimeTransition:
    """Returned by ``RegimeMonitor.record()`` when the regime has changed."""

    prev_regime: str
    new_regime: str
    composite_score: float
    detected_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "prev_regime": self.prev_regime,
            "new_regime": self.new_regime,
            "composite_score": self.composite_score,
            "detected_at": self.detected_at,
        }


# ── RegimeMonitor ──────────────────────────────────────────────────────────


class RegimeMonitor:
    """Persists composite-score regime state and emits transitions.

    Args:
        db_path: Path to the SQLite database.  Defaults to
                 ``data/regime_monitor.db``.
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_schema()

    # ── schema ────────────────────────────────────────────────────────────

    def _apply_schema(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS regime_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                regime          TEXT    NOT NULL,
                composite_score REAL    NOT NULL,
                detected_at     TEXT    NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_regime_detected ON regime_history(detected_at)"
        )
        self._conn.commit()

    # ── public API ────────────────────────────────────────────────────────

    def record(self, composite_score: float) -> Optional[RegimeTransition]:
        """Store the current composite score and detect regime transitions.

        Returns a ``RegimeTransition`` if the regime has changed since the
        previous call, or ``None`` if the regime is the same or on error.
        """
        try:
            new_regime = Regime.from_score(composite_score)
            now = datetime.now(tz=timezone.utc).isoformat()

            cur = self._conn.execute(
                "SELECT regime FROM regime_history ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            prev_regime_str: Optional[str] = row["regime"] if row else None

            self._conn.execute(
                "INSERT INTO regime_history (regime, composite_score, detected_at) VALUES (?, ?, ?)",
                (new_regime.value, composite_score, now),
            )
            self._conn.commit()

            if prev_regime_str is None or prev_regime_str == new_regime.value:
                return None

            transition = RegimeTransition(
                prev_regime=prev_regime_str,
                new_regime=new_regime.value,
                composite_score=composite_score,
                detected_at=now,
            )
            _log.info(
                "regime_transition",
                prev=prev_regime_str,
                new=new_regime.value,
                score=composite_score,
            )
            return transition
        except Exception as exc:  # noqa: BLE001
            _log.warning("regime_record_error", error=str(exc))
            return None

    def current_regime(self) -> Optional[Regime]:
        """Return the last recorded regime, or ``None`` if no records exist."""
        try:
            cur = self._conn.execute(
                "SELECT regime FROM regime_history ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            return Regime(row["regime"]) if row else None
        except Exception as exc:  # noqa: BLE001
            _log.warning("regime_current_error", error=str(exc))
            return None

    def get_history(self, limit: int = 50) -> list[RegimeRecord]:
        """Return the most recent ``limit`` regime records, newest first."""
        try:
            cur = self._conn.execute(
                """
                SELECT id, regime, composite_score, detected_at
                FROM regime_history ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            return [
                RegimeRecord(
                    id=r["id"],
                    regime=r["regime"],
                    composite_score=r["composite_score"],
                    detected_at=r["detected_at"],
                )
                for r in cur.fetchall()
            ]
        except Exception as exc:  # noqa: BLE001
            _log.warning("regime_history_error", error=str(exc))
            return []

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
