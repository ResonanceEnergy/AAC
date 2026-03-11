"""
Polymarket Paper Trading Schema
================================

Database models and schema for paper trading on prediction markets.
Supports Polymarket, Metaculus, and Manifold market interfaces.

Uses SQLite for local paper trading, with PostgreSQL migration path
defined in the SQL constants below.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════


class MarketStatus(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    OPEN = "open"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class Venue(Enum):
    POLYMARKET = "polymarket"
    METACULUS = "metaculus"
    MANIFOLD = "manifold"


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA SQL
# ═══════════════════════════════════════════════════════════════════════════

PAPER_TRADING_SCHEMA = """
-- Prediction markets we track
CREATE TABLE IF NOT EXISTS markets (
    market_id       TEXT PRIMARY KEY,
    venue           TEXT NOT NULL DEFAULT 'polymarket',
    title           TEXT NOT NULL,
    description     TEXT,
    category        TEXT,
    end_date        TEXT,
    status          TEXT NOT NULL DEFAULT 'active',
    resolution      TEXT,                       -- 'YES', 'NO', numeric, etc.
    current_yes_price REAL DEFAULT 0.50,
    current_no_price  REAL DEFAULT 0.50,
    liquidity_usd   REAL DEFAULT 0,
    volume_usd      REAL DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

-- Paper trading portfolio (virtual balances)
CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL DEFAULT 'default',
    initial_balance REAL NOT NULL DEFAULT 10000.0,
    current_balance REAL NOT NULL DEFAULT 10000.0,
    total_pnl       REAL DEFAULT 0.0,
    win_count       INTEGER DEFAULT 0,
    loss_count      INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Individual paper orders
CREATE TABLE IF NOT EXISTS orders (
    order_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id    INTEGER NOT NULL,
    market_id       TEXT NOT NULL,
    venue           TEXT NOT NULL DEFAULT 'polymarket',
    side            TEXT NOT NULL,               -- 'buy' or 'sell'
    outcome         TEXT NOT NULL DEFAULT 'YES',  -- 'YES', 'NO', or specific outcome
    quantity        REAL NOT NULL,
    price           REAL NOT NULL,                -- 0.00 to 1.00
    cost_basis      REAL NOT NULL,                -- quantity * price
    status          TEXT NOT NULL DEFAULT 'open',
    fill_price      REAL,
    pnl             REAL,
    strategy_tag    TEXT,                          -- e.g. 'bw-contrarian', 'bw-momentum'
    notes           TEXT,
    placed_at       TEXT DEFAULT (datetime('now')),
    filled_at       TEXT,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
    FOREIGN KEY (market_id) REFERENCES markets(market_id)
);

-- Position log (aggregated per market)
CREATE TABLE IF NOT EXISTS positions (
    position_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id    INTEGER NOT NULL,
    market_id       TEXT NOT NULL,
    outcome         TEXT NOT NULL DEFAULT 'YES',
    avg_entry_price REAL NOT NULL,
    quantity        REAL NOT NULL,
    current_value   REAL DEFAULT 0.0,
    unrealized_pnl  REAL DEFAULT 0.0,
    opened_at       TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
    FOREIGN KEY (market_id) REFERENCES markets(market_id)
);

-- Trade journal for Jonny Bravo analysis
CREATE TABLE IF NOT EXISTS trade_journal (
    entry_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id    INTEGER NOT NULL,
    order_id        INTEGER,
    market_id       TEXT,
    thesis          TEXT,
    confidence      REAL,
    methodology     TEXT,               -- 'supply_demand', 'price_action', etc.
    outcome_notes   TEXT,
    lesson_learned  TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_orders_portfolio ON orders(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_orders_market ON orders(market_id);
CREATE INDEX IF NOT EXISTS idx_positions_portfolio ON positions(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_markets_status ON markets(status);
CREATE INDEX IF NOT EXISTS idx_markets_venue ON markets(venue);
"""


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════════════


class PaperTradingDB:
    """
    SQLite-backed paper trading database for prediction markets.

    Usage:
        db = PaperTradingDB()
        db.initialize()
        portfolio_id = db.create_portfolio("BARREN WUFFET Paper", balance=10000)
        db.place_order(portfolio_id, market_id="poly-123", side="buy",
                       outcome="YES", quantity=100, price=0.65)
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or Path("data/paper_trading/paper_trades.db")
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Create database and tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(PAPER_TRADING_SCHEMA)
        self._conn.commit()
        logger.info(f"Paper trading DB initialized at {self.db_path}")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if not self._conn:
            self.initialize()
        return self._conn  # type: ignore

    # ── Portfolio ──────────────────────────────────────────────────────

    def create_portfolio(self, name: str = "default", balance: float = 10_000.0) -> int:
        """Create a new paper trading portfolio."""
        cur = self.conn.execute(
            "INSERT INTO portfolios (name, initial_balance, current_balance) VALUES (?, ?, ?)",
            (name, balance, balance),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore

    def get_portfolio(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        """Get portfolio details."""
        row = self.conn.execute(
            "SELECT * FROM portfolios WHERE portfolio_id = ?", (portfolio_id,)
        ).fetchone()
        return dict(row) if row else None

    # ── Markets ────────────────────────────────────────────────────────

    def upsert_market(
        self,
        market_id: str,
        title: str,
        venue: str = "polymarket",
        yes_price: float = 0.5,
        no_price: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Insert or update a prediction market."""
        self.conn.execute(
            """INSERT INTO markets (market_id, venue, title, current_yes_price, current_no_price,
                                     description, category, end_date, liquidity_usd, volume_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(market_id) DO UPDATE SET
                   current_yes_price = excluded.current_yes_price,
                   current_no_price = excluded.current_no_price,
                   updated_at = datetime('now')""",
            (
                market_id, venue, title, yes_price, no_price,
                kwargs.get("description"), kwargs.get("category"),
                kwargs.get("end_date"), kwargs.get("liquidity_usd", 0),
                kwargs.get("volume_usd", 0),
            ),
        )
        self.conn.commit()

    # ── Orders ─────────────────────────────────────────────────────────

    def place_order(
        self,
        portfolio_id: int,
        market_id: str,
        side: str,
        outcome: str,
        quantity: float,
        price: float,
        strategy_tag: Optional[str] = None,
        notes: Optional[str] = None,
        venue: str = "polymarket",
    ) -> int:
        """Place a paper trade order."""
        cost_basis = quantity * price
        cur = self.conn.execute(
            """INSERT INTO orders (portfolio_id, market_id, venue, side, outcome,
                                    quantity, price, cost_basis, strategy_tag, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (portfolio_id, market_id, venue, side, outcome, quantity, price,
             cost_basis, strategy_tag, notes),
        )
        self.conn.commit()
        logger.info(f"Paper order placed: {side} {quantity}x {outcome} @ {price} on {market_id}")
        return cur.lastrowid  # type: ignore

    def get_open_orders(self, portfolio_id: int) -> List[Dict[str, Any]]:
        """Get all open orders for a portfolio."""
        rows = self.conn.execute(
            "SELECT * FROM orders WHERE portfolio_id = ? AND status = 'open' ORDER BY placed_at DESC",
            (portfolio_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Journal ────────────────────────────────────────────────────────

    def add_journal_entry(
        self,
        portfolio_id: int,
        thesis: str,
        confidence: float = 0.5,
        methodology: str = "price_action",
        order_id: Optional[int] = None,
        market_id: Optional[str] = None,
    ) -> int:
        """Add a trade journal entry."""
        cur = self.conn.execute(
            """INSERT INTO trade_journal (portfolio_id, order_id, market_id,
                                          thesis, confidence, methodology)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (portfolio_id, order_id, market_id, thesis, confidence, methodology),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore

    # ── Stats ──────────────────────────────────────────────────────────

    def get_portfolio_stats(self, portfolio_id: int) -> Dict[str, Any]:
        """Get aggregate stats for a portfolio."""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return {}

        order_count = self.conn.execute(
            "SELECT COUNT(*) FROM orders WHERE portfolio_id = ?", (portfolio_id,)
        ).fetchone()[0]

        open_count = self.conn.execute(
            "SELECT COUNT(*) FROM orders WHERE portfolio_id = ? AND status = 'open'",
            (portfolio_id,),
        ).fetchone()[0]

        return {
            **portfolio,
            "total_orders": order_count,
            "open_orders": open_count,
            "roi_pct": round(
                (portfolio["current_balance"] - portfolio["initial_balance"])
                / portfolio["initial_balance"]
                * 100,
                2,
            ),
        }
