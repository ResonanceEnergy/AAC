"""Persistent Execution Throttle — Sprint 23.

Replaces the in-memory ``AutoTrader._last_executed`` dict with a
SQLite-backed throttle that survives process restarts.

When the system crashes and restarts, the in-memory throttle is lost and
every ticker looks fresh — risking immediate re-entry on the same positions.
This module persists throttle state to disk so the cooldown window is honoured
across restarts.

Key properties:
- Fails-open: any DB error → ``can_execute()`` returns ``True`` so trading
  is never blocked by a broken throttle, and ``record_execution()`` silently
  discards the write.
- Thread-safe: SQLite WAL mode + per-call connections.
- ``clear()`` is provided for tests and manual resets.
"""
from __future__ import annotations

import sqlite3
import time
import os
from pathlib import Path

import structlog

_log = structlog.get_logger(__name__)

# Default cooldown: 4 hours (14400 seconds)
_DEFAULT_THROTTLE_SECONDS: float = 14_400.0
_DEFAULT_DB_PATH = Path("data") / "execution_throttle.db"

_DDL = """
CREATE TABLE IF NOT EXISTS execution_log (
    ticker       TEXT    NOT NULL,
    executed_at  REAL    NOT NULL,
    PRIMARY KEY (ticker)
);
"""


class ExecutionThrottle:
    """SQLite-backed execution throttle — survives process restarts.

    Args:
        db_path:           Path to the SQLite database file.  Defaults to
                           ``data/execution_throttle.db``.  Pass ``:memory:``
                           in tests for isolation.
        throttle_seconds:  Cooldown window in seconds between executions of
                           the same ticker.  Defaults to 4 hours.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        throttle_seconds: float = _DEFAULT_THROTTLE_SECONDS,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        self.throttle_seconds = throttle_seconds
        self._initialized = False
        # For :memory: databases, reuse a single connection across all calls
        # (each new sqlite3.connect(':memory:') creates a separate empty DB)
        self._mem_conn: sqlite3.Connection | None = None
        self._init_db()

    # ── public API ────────────────────────────────────────────────────────

    def can_execute(self, ticker: str) -> bool:
        """Return True if the ticker is not currently throttled.

        Fails-open on any DB or logic error — never blocks execution.
        """
        try:
            self._ensure_init()
            last_ts = self._get_last_executed(ticker)
            if last_ts is None:
                return True
            elapsed = time.time() - last_ts
            return elapsed >= self.throttle_seconds
        except Exception as exc:
            _log.warning(
                "execution_throttle_can_execute_error",
                ticker=ticker,
                error=str(exc),
            )
            return True  # fail-open

    def record_execution(self, ticker: str) -> None:
        """Persist the current timestamp as the last execution for *ticker*.

        Silently discards errors so throttle recording never blocks trading.
        """
        try:
            self._ensure_init()
            now = time.time()
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO execution_log (ticker, executed_at)
                    VALUES (?, ?)
                    ON CONFLICT(ticker) DO UPDATE SET executed_at = excluded.executed_at
                    """,
                    (ticker, now),
                )
        except Exception as exc:
            _log.warning(
                "execution_throttle_record_error",
                ticker=ticker,
                error=str(exc),
            )

    def last_executed(self, ticker: str) -> float | None:
        """Return the Unix timestamp of the last recorded execution, or None.

        Fails-open: returns None on any error.
        """
        try:
            self._ensure_init()
            return self._get_last_executed(ticker)
        except Exception as exc:
            _log.warning(
                "execution_throttle_last_executed_error",
                ticker=ticker,
                error=str(exc),
            )
            return None

    def remaining_seconds(self, ticker: str) -> float:
        """Return seconds remaining in the throttle window (0 if not throttled).

        Fails-open: returns 0 on error.
        """
        try:
            last_ts = self._get_last_executed(ticker)
            if last_ts is None:
                return 0.0
            elapsed = time.time() - last_ts
            remaining = self.throttle_seconds - elapsed
            return max(0.0, remaining)
        except Exception as exc:
            _log.warning(
                "execution_throttle_remaining_error",
                ticker=ticker,
                error=str(exc),
            )
            return 0.0

    def clear(self, ticker: str | None = None) -> None:
        """Clear throttle state.

        Args:
            ticker: If provided, clears only that ticker.  If None, clears all.

        Silently discards errors.
        """
        try:
            self._ensure_init()
            with self._connect() as conn:
                if ticker is not None:
                    conn.execute(
                        "DELETE FROM execution_log WHERE ticker = ?", (ticker,)
                    )
                else:
                    conn.execute("DELETE FROM execution_log")
        except Exception as exc:
            _log.warning(
                "execution_throttle_clear_error",
                ticker=ticker,
                error=str(exc),
            )

    def all_entries(self) -> list[dict[str, float]]:
        """Return all throttle entries as a list of dicts (for diagnostics).

        Fails-open: returns [] on error.
        """
        try:
            self._ensure_init()
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT ticker, executed_at FROM execution_log ORDER BY executed_at DESC"
                ).fetchall()
            return [{"ticker": row[0], "executed_at": row[1]} for row in rows]
        except Exception as exc:
            _log.warning("execution_throttle_all_entries_error", error=str(exc))
            return []

    # ── internal ──────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create the database file and schema.  Silently discards errors."""
        try:
            if str(self.db_path) != ":memory:":
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(_DDL)
            self._initialized = True
        except Exception as exc:
            _log.warning("execution_throttle_init_failed", error=str(exc))
            self._initialized = False

    def _ensure_init(self) -> None:
        """Re-attempt init if first attempt failed."""
        if not self._initialized:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if str(self.db_path) == ":memory:":
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            return self._mem_conn
        return sqlite3.connect(str(self.db_path), timeout=5.0)

    def _get_last_executed(self, ticker: str) -> float | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT executed_at FROM execution_log WHERE ticker = ?",
                (ticker,),
            ).fetchone()
        return row[0] if row else None
