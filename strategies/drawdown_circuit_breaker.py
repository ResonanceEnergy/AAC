from __future__ import annotations

"""strategies/drawdown_circuit_breaker.py — Sprint 18: Drawdown Circuit Breaker.

Monitors cumulative account drawdown from peak equity and trips a
"no-new-trades" flag when drawdown exceeds a configurable threshold.

Unlike ``DailyLossGuard`` (single-day P&L ceiling), this guard protects
against *multi-day* drawdowns from a rolling account-value peak.  Both
guards run independently — either can halt trading.

Key rules
---------
* Peak only moves **up** — it never resets automatically.  ``reset()`` must be
  called explicitly (e.g. after the drawdown has been reviewed and resolved).
* ``update(account_value_usd)`` persists state atomically via SQLite WAL.
* ``is_tripped() -> bool`` is a cheap read-only query.
* Fails-**open**: any DB or unexpected exception returns a safe default
  (tripped=False) so a broken guard never halts all trading.

Usage::

    breaker = DrawdownCircuitBreaker(max_drawdown_pct=0.10)
    state = breaker.update(current_account_value)
    if state.tripped:
        print(f"Drawdown circuit breaker tripped: {state.drawdown_pct:.1%}")

Design notes
------------
* SQLite WAL mode + ``check_same_thread=False`` — safe for use from async
  contexts called via ``asyncio.run()``.
* Default DB path: ``data/drawdown_circuit_breaker.db`` (relative to CWD,
  matching other SQLite-backed modules in the codebase).
* ``DrawdownState`` is a plain dataclass (not frozen) so it can be returned
  from update() with minimal allocation overhead.
"""

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import structlog

_log = structlog.get_logger(__name__)

_DEFAULT_DB_PATH = "data/drawdown_circuit_breaker.db"
_DEFAULT_MAX_DRAWDOWN_PCT = 0.10  # 10 %

_DDL = """
CREATE TABLE IF NOT EXISTS drawdown_state (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    peak_value    REAL    NOT NULL,
    current_value REAL    NOT NULL,
    drawdown_pct  REAL    NOT NULL DEFAULT 0.0,
    tripped       INTEGER NOT NULL DEFAULT 0,
    tripped_at    TEXT,
    updated_at    TEXT    NOT NULL
);
"""


# ── DrawdownState ─────────────────────────────────────────────────────────────


@dataclass
class DrawdownState:
    """Snapshot of the circuit breaker's current state.

    Attributes:
        peak_value:    Highest account value ever recorded (USD).
        current_value: Most-recently recorded account value (USD).
        drawdown_pct:  Fraction drawdown from peak — 0.12 means 12%.
                       0.0 when peak_value is 0 (no history yet).
        tripped:       True when drawdown_pct >= max_drawdown_pct.
        tripped_at:    UTC ISO timestamp of when the breaker was first tripped,
                       or None if it has never been tripped.
    """

    peak_value: float
    current_value: float
    drawdown_pct: float
    tripped: bool
    tripped_at: Optional[str] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "peak_value": self.peak_value,
            "current_value": self.current_value,
            "drawdown_pct": round(self.drawdown_pct, 6),
            "tripped": self.tripped,
            "tripped_at": self.tripped_at,
        }


# ── DrawdownCircuitBreaker ────────────────────────────────────────────────────


class DrawdownCircuitBreaker:
    """Tracks rolling peak account value and trips when drawdown is too deep.

    Args:
        db_path:          Path to SQLite database file.  ``None`` uses the
                          default path ``data/drawdown_circuit_breaker.db``.
        max_drawdown_pct: Maximum allowed drawdown fraction before the breaker
                          trips (default 0.10 = 10 %).  Reads
                          ``MAX_DRAWDOWN_PCT`` env var when 0.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_drawdown_pct: float = _DEFAULT_MAX_DRAWDOWN_PCT,
    ) -> None:
        if max_drawdown_pct <= 0:
            max_drawdown_pct = float(os.getenv("MAX_DRAWDOWN_PCT", "0.10"))
        self.max_drawdown_pct = max_drawdown_pct
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._conn: sqlite3.Connection | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, account_value_usd: float) -> DrawdownState:
        """Record a new account value, update peak, calculate drawdown.

        Trips the breaker if ``drawdown_pct >= max_drawdown_pct``.

        Args:
            account_value_usd: Current total account value in USD.

        Returns:
            ``DrawdownState`` reflecting the new state.  On any error,
            returns a safe ``DrawdownState(tripped=False, ...)`` so callers
            can always read ``.tripped`` safely.
        """
        if account_value_usd <= 0:
            _log.warning("drawdown_cb_invalid_account_value", value=account_value_usd)
            return DrawdownState(
                peak_value=0.0,
                current_value=0.0,
                drawdown_pct=0.0,
                tripped=False,
            )

        try:
            conn = self._get_conn()
            now_iso = datetime.utcnow().isoformat()

            with conn:
                existing = conn.execute(
                    "SELECT peak_value, tripped, tripped_at FROM drawdown_state WHERE id = 1"
                ).fetchone()

                if existing is None:
                    # First ever record — peak = current value, not tripped.
                    peak_value = account_value_usd
                    tripped = False
                    tripped_at = None
                else:
                    peak_value = max(float(existing["peak_value"]), account_value_usd)
                    tripped = bool(existing["tripped"])
                    tripped_at = existing["tripped_at"]

                # Recalculate drawdown.
                if peak_value > 0:
                    drawdown_pct = (peak_value - account_value_usd) / peak_value
                else:
                    drawdown_pct = 0.0

                # Trip if threshold exceeded (once tripped, stays tripped until reset).
                if not tripped and drawdown_pct >= self.max_drawdown_pct:
                    tripped = True
                    tripped_at = now_iso
                    _log.warning(
                        "drawdown_circuit_breaker_tripped",
                        drawdown_pct=round(drawdown_pct, 4),
                        peak_value=round(peak_value, 2),
                        current_value=round(account_value_usd, 2),
                        threshold=self.max_drawdown_pct,
                    )

                conn.execute(
                    """
                    INSERT INTO drawdown_state
                        (id, peak_value, current_value, drawdown_pct, tripped, tripped_at, updated_at)
                    VALUES (1, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        peak_value    = excluded.peak_value,
                        current_value = excluded.current_value,
                        drawdown_pct  = excluded.drawdown_pct,
                        tripped       = excluded.tripped,
                        tripped_at    = excluded.tripped_at,
                        updated_at    = excluded.updated_at
                    """,
                    (
                        peak_value,
                        account_value_usd,
                        drawdown_pct,
                        int(tripped),
                        tripped_at,
                        now_iso,
                    ),
                )

            return DrawdownState(
                peak_value=peak_value,
                current_value=account_value_usd,
                drawdown_pct=drawdown_pct,
                tripped=tripped,
                tripped_at=tripped_at,
            )

        except Exception as exc:
            _log.warning("drawdown_cb_update_failed", error=str(exc))
            return DrawdownState(
                peak_value=0.0,
                current_value=account_value_usd,
                drawdown_pct=0.0,
                tripped=False,
            )

    def is_tripped(self) -> bool:
        """Return True if the circuit breaker is currently tripped.

        Reads directly from SQLite so it always reflects the latest persisted
        state (even if the instance was created on a different call).

        Returns False on any error (fail-open).
        """
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT tripped FROM drawdown_state WHERE id = 1"
            ).fetchone()
            if row is None:
                return False
            return bool(row["tripped"])
        except Exception as exc:
            _log.warning("drawdown_cb_is_tripped_failed", error=str(exc))
            return False

    def current_state(self) -> DrawdownState:
        """Return the current drawdown state without modifying anything.

        Useful for read-only reporting (e.g. EodReporter) that needs the
        current drawdown percentage without recording a new account value.

        Returns a safe ``DrawdownState(tripped=False, ...)`` on any error.
        """
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT peak_value, current_value, drawdown_pct, tripped, tripped_at "
                "FROM drawdown_state WHERE id = 1"
            ).fetchone()
            if row is None:
                return DrawdownState(
                    peak_value=0.0,
                    current_value=0.0,
                    drawdown_pct=0.0,
                    tripped=False,
                )
            return DrawdownState(
                peak_value=float(row["peak_value"]),
                current_value=float(row["current_value"]),
                drawdown_pct=float(row["drawdown_pct"]),
                tripped=bool(row["tripped"]),
                tripped_at=row["tripped_at"],
            )
        except Exception as exc:
            _log.warning("drawdown_cb_current_state_failed", error=str(exc))
            return DrawdownState(
                peak_value=0.0,
                current_value=0.0,
                drawdown_pct=0.0,
                tripped=False,
            )

    def reset(self, new_peak_value: Optional[float] = None) -> None:
        """Manually reset the circuit breaker.

        Clears the trip flag.  If ``new_peak_value`` is provided it becomes
        the new peak (useful when starting a new trading period after a
        drawdown recovery).  If omitted, the current ``current_value``
        is used as the new peak.

        Args:
            new_peak_value: New peak to start measuring drawdown from.
                            ``None`` → read current_value from DB and use that.
        """
        try:
            conn = self._get_conn()
            now_iso = datetime.utcnow().isoformat()

            with conn:
                if new_peak_value is not None and new_peak_value > 0:
                    peak = new_peak_value
                else:
                    row = conn.execute(
                        "SELECT current_value FROM drawdown_state WHERE id = 1"
                    ).fetchone()
                    peak = float(row["current_value"]) if row else 0.0

                conn.execute(
                    """
                    UPDATE drawdown_state
                    SET peak_value   = ?,
                        drawdown_pct = 0.0,
                        tripped      = 0,
                        tripped_at   = NULL,
                        updated_at   = ?
                    WHERE id = 1
                    """,
                    (peak, now_iso),
                )
            _log.info("drawdown_circuit_breaker_reset", new_peak=round(peak, 2))
        except Exception as exc:
            _log.warning("drawdown_cb_reset_failed", error=str(exc))

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return (or lazily create) the SQLite connection."""
        if self._conn is None:
            os.makedirs(os.path.dirname(os.path.abspath(self._db_path)), exist_ok=True)
            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_DDL)
            self._conn.commit()
        return self._conn
