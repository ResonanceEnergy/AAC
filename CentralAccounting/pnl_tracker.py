"""CentralAccounting/pnl_tracker.py — Sprint 4.

High-level P&L tracking facade.

Responsibilities
----------------
* take_snapshot(positions, account_value_usd)
  — Writes today's position state to SQLite.
  — Upserts a daily_pnl row so repeated calls are idempotent.

* log_trade(symbol, direction, qty, fill_price, ...)
  — Appends a trade_log row whenever SignalExecutor confirms a fill.

* today_report() → dict
  — Returns structured report suitable for CLI display or monitoring.

* historical_summary(days=30) → list[dict]
  — Rolls up daily_pnl rows for the past N days.

* pnl_delta(days=2) → float
  — Return change in unrealised P&L over last N days.

Database
--------
Uses a *separate* minimal SQLite (default: CentralAccounting/data/pnl.db).
It does NOT depend on AccountingDatabase foreign keys so it stays simple and
testable with :memory: without pre-seeding accounts.

Schema tables
-------------
  position_snapshots  — one row per position per day
  daily_pnl           — one aggregate row per day (UNIQUE on snapshot_date)
  trade_log           — one row per OrderConfirmation received
"""
from __future__ import annotations

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import structlog

_log = structlog.get_logger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS position_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date   TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    sec_type        TEXT    NOT NULL DEFAULT 'STK',
    quantity        REAL    NOT NULL DEFAULT 0,
    avg_cost        REAL,
    market_price    REAL,
    market_value    REAL    NOT NULL DEFAULT 0,
    unrealized_pnl  REAL    NOT NULL DEFAULT 0,
    realized_pnl    REAL    NOT NULL DEFAULT 0,
    expiry          TEXT,
    strike          REAL,
    right           TEXT,
    account         TEXT    NOT NULL DEFAULT 'IBKR',
    captured_at     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date       TEXT    NOT NULL UNIQUE,
    account_value_usd   REAL    NOT NULL DEFAULT 0,
    total_exposure_usd  REAL    NOT NULL DEFAULT 0,
    total_unrealized_pnl REAL   NOT NULL DEFAULT 0,
    total_realized_pnl  REAL    NOT NULL DEFAULT 0,
    position_count      INTEGER NOT NULL DEFAULT 0,
    captured_at         TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    direction   TEXT    NOT NULL,
    quantity    REAL    NOT NULL DEFAULT 0,
    fill_price  REAL,
    order_id    TEXT,
    status      TEXT    NOT NULL DEFAULT 'unknown',
    strategy    TEXT,
    confidence  REAL,
    logged_at   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pnl_date     ON daily_pnl(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_snap_date    ON position_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_trade_logged ON trade_log(logged_at);
"""


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class DailyPnlRow:
    snapshot_date: str
    account_value_usd: float
    total_exposure_usd: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    position_count: int
    captured_at: str

    @property
    def total_pnl(self) -> float:
        return self.total_unrealized_pnl + self.total_realized_pnl

    def to_dict(self) -> dict:
        return {
            "date": self.snapshot_date,
            "account_value_usd": round(self.account_value_usd, 2),
            "total_exposure_usd": round(self.total_exposure_usd, 2),
            "total_unrealized_pnl": round(self.total_unrealized_pnl, 2),
            "total_realized_pnl": round(self.total_realized_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "position_count": self.position_count,
            "captured_at": self.captured_at,
        }


# ── Store (thin SQLite wrapper) ───────────────────────────────────────────────

class _PnlStore:
    """Minimal SQLite wrapper — not thread-pooled, designed for single-threaded CLI."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        if self._conn is not None:
            return
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> _PnlStore:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        cur = self._c().execute(sql, params)
        self._c().commit()
        return cur

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self._c().execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[dict]:
        cur = self._c().execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None


# ── Tracker ───────────────────────────────────────────────────────────────────

class PnLTracker:
    """Records position snapshots, trade fills, and daily P&L to SQLite.

    Usage::

        tracker = PnLTracker()           # uses default CentralAccounting/data/pnl.db
        report = tracker.take_snapshot(positions, account_value_usd=50_000)
        print(tracker.format_report(report))

    Testing::

        tracker = PnLTracker(":memory:")
    """

    #: Default DB path (relative to project root — overridden in tests)
    DEFAULT_DB_PATH: str = "CentralAccounting/data/pnl.db"

    def __init__(self, db_path: str | Path | None = None) -> None:
        resolved = db_path if db_path is not None else self.DEFAULT_DB_PATH
        self._store = _PnlStore(resolved)
        self._store.connect()

    def close(self) -> None:
        self._store.close()

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def take_snapshot(
        self,
        positions: list,
        account_value_usd: float,
        snapshot_date: Optional[str] = None,
        account: str = "IBKR",
    ) -> dict:
        """Persist today's position state and return today_report().

        Args:
            positions: List[PositionSnapshot] from PositionTracker.all()
            account_value_usd: Net liquidation value of the account.
            snapshot_date: YYYY-MM-DD override (defaults to today UTC).
            account: Exchange label (IBKR, Moomoo, WS).

        Returns:
            today_report() dict.
        """
        today = snapshot_date or date.today().isoformat()
        captured_at = datetime.now().isoformat()

        total_unrealized = 0.0
        total_realized = 0.0
        total_exposure = 0.0

        for pos in positions:
            unr = float(getattr(pos, "unrealized_pnl", 0) or 0)
            rea = float(getattr(pos, "realized_pnl", 0) or 0)
            mv = float(getattr(pos, "market_value", 0) or 0)
            total_unrealized += unr
            total_realized += rea
            total_exposure += abs(mv)

            self._store.execute(
                """
                INSERT INTO position_snapshots (
                    snapshot_date, symbol, sec_type, quantity, avg_cost,
                    market_price, market_value, unrealized_pnl, realized_pnl,
                    expiry, strike, right, account, captured_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    today,
                    str(getattr(pos, "symbol", "?")),
                    str(getattr(pos, "sec_type", "STK")),
                    float(getattr(pos, "quantity", 0) or 0),
                    float(getattr(pos, "avg_cost", 0) or 0),
                    float(getattr(pos, "market_price", 0) or 0),
                    mv,
                    unr,
                    rea,
                    getattr(pos, "expiry", None),
                    getattr(pos, "strike", None),
                    getattr(pos, "right", None),
                    account,
                    captured_at,
                ),
            )

        # Upsert daily_pnl (INSERT OR REPLACE so repeated calls are idempotent)
        self._store.execute(
            """
            INSERT OR REPLACE INTO daily_pnl (
                snapshot_date, account_value_usd, total_exposure_usd,
                total_unrealized_pnl, total_realized_pnl, position_count,
                captured_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                today,
                account_value_usd,
                total_exposure,
                total_unrealized,
                total_realized,
                len(positions),
                captured_at,
            ),
        )

        _log.info(
            "P&L snapshot taken",
            date=today,
            positions=len(positions),
            unrealized_pnl=round(total_unrealized, 2),
            account_value=account_value_usd,
        )
        return self.today_report(snapshot_date=today)

    # ── Trade log ─────────────────────────────────────────────────────────────

    def log_trade(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        fill_price: Optional[float] = None,
        order_id: Optional[str] = None,
        status: str = "filled",
        strategy: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """Record a trade execution.

        Returns the new trade_log row id.
        """
        logged_at = datetime.now().isoformat()
        cur = self._store.execute(
            """
            INSERT INTO trade_log (
                symbol, direction, quantity, fill_price, order_id,
                status, strategy, confidence, logged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (symbol, direction, quantity, fill_price, order_id,
             status, strategy, confidence, logged_at),
        )
        return cur.lastrowid

    def log_trade_from_confirmation(self, confirmation) -> int:
        """Log from an OrderConfirmation object (from signal_executor.py).

        Accepts any object with .symbol, .status, .filled_price, .order_id attrs.
        """
        symbol = str(getattr(confirmation, "symbol", "?"))
        direction = str(getattr(confirmation, "direction", "?"))
        qty = float(getattr(confirmation, "quantity", 0) or 0)
        fill_price = getattr(confirmation, "filled_price", None)
        order_id = str(getattr(confirmation, "order_id", "") or "")
        status = getattr(confirmation, "status", "unknown")
        if hasattr(status, "value"):
            status = status.value
        status = str(status)
        strategy = getattr(confirmation, "strategy", None)
        confidence = getattr(confirmation, "confidence", None)

        return self.log_trade(
            symbol=symbol,
            direction=direction,
            quantity=qty,
            fill_price=fill_price,
            order_id=order_id or None,
            status=status,
            strategy=strategy,
            confidence=confidence,
        )

    # ── Queries ───────────────────────────────────────────────────────────────

    def today_report(self, snapshot_date: Optional[str] = None) -> dict:
        """Return today's P&L report dict."""
        today = snapshot_date or date.today().isoformat()

        pnl_row = self._store.fetchone(
            "SELECT * FROM daily_pnl WHERE snapshot_date = ?", (today,)
        )
        positions = self._store.fetchall(
            "SELECT * FROM position_snapshots WHERE snapshot_date = ? ORDER BY symbol",
            (today,),
        )
        trades = self._store.fetchall(
            "SELECT * FROM trade_log WHERE substr(logged_at, 1, 10) = ? ORDER BY logged_at DESC",
            (today,),
        )

        return {
            "date": today,
            "daily_pnl": pnl_row,
            "positions": positions,
            "today_trades": trades,
        }

    def historical_summary(self, days: int = 30) -> list[dict]:
        """Return daily_pnl rows for the last N days, most recent first."""
        return self._store.fetchall(
            "SELECT * FROM daily_pnl ORDER BY snapshot_date DESC LIMIT ?",
            (days,),
        )

    def pnl_delta(self, days: int = 2) -> float:
        """Change in total unrealised P&L between the two most recent snapshots.

        Returns positive when improving, negative when deteriorating.
        """
        rows = self.historical_summary(days=days)
        if len(rows) < 2:
            return 0.0
        newest = rows[0]["total_unrealized_pnl"] + rows[0]["total_realized_pnl"]
        oldest = rows[-1]["total_unrealized_pnl"] + rows[-1]["total_realized_pnl"]
        return round(newest - oldest, 2)

    def recent_trades(self, limit: int = 20) -> list[dict]:
        """Return most recent trade_log entries."""
        return self._store.fetchall(
            "SELECT * FROM trade_log ORDER BY logged_at DESC LIMIT ?",
            (limit,),
        )

    def all_pnl_rows(self) -> list[DailyPnlRow]:
        """Return all daily_pnl rows as typed objects."""
        raw = self._store.fetchall("SELECT * FROM daily_pnl ORDER BY snapshot_date")
        return [
            DailyPnlRow(
                snapshot_date=r["snapshot_date"],
                account_value_usd=r["account_value_usd"],
                total_exposure_usd=r["total_exposure_usd"],
                total_unrealized_pnl=r["total_unrealized_pnl"],
                total_realized_pnl=r["total_realized_pnl"],
                position_count=r["position_count"],
                captured_at=r["captured_at"],
            )
            for r in raw
        ]

    # ── CLI formatter ─────────────────────────────────────────────────────────

    @staticmethod
    def format_report(report: dict, colorize: bool = False) -> str:
        """Format today_report() as a human-readable CLI string."""
        lines: list[str] = []

        def _h(text: str) -> str:
            return f"\n  {'─' * 50}\n  {text}\n  {'─' * 50}"

        lines.append(_h(f"AAC P&L Report — {report['date']}"))

        pnl = report.get("daily_pnl")
        if pnl:
            lines.append(f"\n  Account Value  : ${pnl['account_value_usd']:>12,.2f}")
            lines.append(f"  Total Exposure : ${pnl['total_exposure_usd']:>12,.2f}")
            lines.append(f"  Unrealised P&L : ${pnl['total_unrealized_pnl']:>12,.2f}")
            lines.append(f"  Realised P&L   : ${pnl['total_realized_pnl']:>12,.2f}")
            total = pnl["total_unrealized_pnl"] + pnl["total_realized_pnl"]
            lines.append(f"  Total P&L      : ${total:>12,.2f}")
            lines.append(f"  Open Positions : {pnl['position_count']}")
        else:
            lines.append("\n  No snapshot taken today — run with live positions to record.")

        positions = report.get("positions", [])
        if positions:
            lines.append(_h(f"Positions ({len(positions)})"))
            lines.append(f"  {'Symbol':<8} {'Type':<4} {'Qty':>6} {'Mkt Val':>12} {'Unr P&L':>10}")
            lines.append(f"  {'─'*8} {'─'*4} {'─'*6} {'─'*12} {'─'*10}")
            for p in positions:
                lines.append(
                    f"  {p['symbol']:<8} {p['sec_type']:<4} {p['quantity']:>6.1f} "
                    f"${p['market_value']:>10,.2f} ${p['unrealized_pnl']:>8,.2f}"
                )

        trades = report.get("today_trades", [])
        if trades:
            lines.append(_h(f"Today's Trades ({len(trades)})"))
            for t in trades:
                price_str = f"@ ${t['fill_price']:.4f}" if t.get("fill_price") else ""
                lines.append(
                    f"  {t['symbol']:<8} {t['direction']:<8} x{t['quantity']:.0f}  "
                    f"{price_str}  [{t['status']}]"
                )
        else:
            lines.append("\n  No trades logged today.")

        lines.append(f"\n  {'─' * 50}\n")
        return "\n".join(lines)
