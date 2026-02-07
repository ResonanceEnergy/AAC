#!/usr/bin/env python3
"""
Central Accounting Database Schema and Migrations
=================================================
SQLite database schema for transactions, positions, and P&L tracking.
"""

import sqlite3
import logging
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Optional, Generator, List, Dict, Any
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_project_path

logger = logging.getLogger('AccountingDB')


# Schema version for migrations
SCHEMA_VERSION = 2

SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Accounts (wallets, exchange accounts, etc.)
CREATE TABLE IF NOT EXISTS accounts (
    account_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    account_type TEXT NOT NULL,  -- 'exchange', 'wallet', 'bank'
    exchange TEXT,  -- 'binance', 'coinbase', etc.
    currency TEXT DEFAULT 'USD',
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Asset balances by account
CREATE TABLE IF NOT EXISTS balances (
    balance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL,
    asset TEXT NOT NULL,
    free_balance REAL DEFAULT 0,
    locked_balance REAL DEFAULT 0,
    total_balance REAL GENERATED ALWAYS AS (free_balance + locked_balance) STORED,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id),
    UNIQUE(account_id, asset)
);

-- Transactions (all financial movements)
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT,  -- Exchange order ID, blockchain tx hash, etc.
    account_id INTEGER NOT NULL,
    transaction_type TEXT NOT NULL,  -- 'trade', 'deposit', 'withdrawal', 'fee', 'transfer'
    side TEXT,  -- 'buy', 'sell' for trades
    symbol TEXT,  -- Trading pair for trades
    asset TEXT NOT NULL,  -- Asset involved
    quantity REAL NOT NULL,
    price REAL,  -- Price per unit (for trades)
    total_value REAL,  -- Total value in base currency
    fee REAL DEFAULT 0,
    fee_currency TEXT,
    status TEXT DEFAULT 'pending',  -- 'pending', 'completed', 'failed', 'cancelled'
    notes TEXT,
    executed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

-- Open positions (accounting view)
CREATE TABLE IF NOT EXISTS positions (
    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'long', 'short'
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL,
    unrealized_pnl REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    stop_loss REAL,
    take_profit REAL,
    leverage REAL DEFAULT 1,
    status TEXT DEFAULT 'open',  -- 'open', 'closed', 'liquidated'
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    metadata TEXT,  -- JSON for additional data
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

-- Daily P&L snapshots
CREATE TABLE IF NOT EXISTS daily_pnl (
    pnl_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL UNIQUE,
    starting_equity REAL NOT NULL,
    ending_equity REAL NOT NULL,
    realized_pnl REAL DEFAULT 0,
    unrealized_pnl REAL DEFAULT 0,
    total_pnl REAL GENERATED ALWAYS AS (realized_pnl + unrealized_pnl) STORED,
    fees_paid REAL DEFAULT 0,
    trades_count INTEGER DEFAULT 0,
    win_count INTEGER DEFAULT 0,
    loss_count INTEGER DEFAULT 0,
    largest_win REAL DEFAULT 0,
    largest_loss REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trade history (completed trades with P&L)
CREATE TABLE IF NOT EXISTS trade_history (
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER,
    account_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    quantity REAL NOT NULL,
    gross_pnl REAL NOT NULL,
    fees REAL DEFAULT 0,
    net_pnl REAL GENERATED ALWAYS AS (gross_pnl - fees) STORED,
    hold_duration_seconds INTEGER,
    strategy TEXT,
    notes TEXT,
    opened_at TIMESTAMP,
    closed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (position_id) REFERENCES positions(position_id),
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

-- Alerts and notifications log
CREATE TABLE IF NOT EXISTS alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type TEXT NOT NULL,  -- 'risk', 'profit', 'loss', 'system'
    severity TEXT NOT NULL,  -- 'info', 'warning', 'critical'
    message TEXT NOT NULL,
    data TEXT,  -- JSON
    acknowledged BOOLEAN DEFAULT 0,
    acknowledged_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System events log
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    data TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading positions (execution engine state)
CREATE TABLE IF NOT EXISTS trading_positions (
    position_id TEXT PRIMARY KEY,  -- UUID from execution engine
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'buy', 'sell'
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL,
    stop_loss REAL,
    take_profit REAL,
    status TEXT DEFAULT 'open',  -- 'open', 'closed'
    exchange TEXT,
    unrealized_pnl REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    opened_at TIMESTAMP,
    closed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders (execution engine state)
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,  -- UUID from execution engine
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'buy', 'sell'
    order_type TEXT NOT NULL,  -- 'market', 'limit', 'stop_loss', 'take_profit'
    quantity REAL NOT NULL,
    price REAL,
    status TEXT DEFAULT 'pending',  -- 'pending', 'submitted', 'filled', 'cancelled', 'rejected'
    exchange TEXT,
    filled_quantity REAL DEFAULT 0,
    average_price REAL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    metadata TEXT  -- JSON for additional data
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_transactions_account ON transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_transactions_executed ON transactions(executed_at);
CREATE INDEX IF NOT EXISTS idx_transactions_symbol ON transactions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_account ON positions(account_id);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_history_closed ON trade_history(closed_at);
CREATE INDEX IF NOT EXISTS idx_daily_pnl_date ON daily_pnl(date);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_trading_positions_status ON trading_positions(status);
CREATE INDEX IF NOT EXISTS idx_trading_positions_symbol ON trading_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
"""

# Default accounts to create
DEFAULT_ACCOUNTS = [
    ('Binance Spot', 'exchange', 'binance', 'USD'),
    ('Coinbase Pro', 'exchange', 'coinbase', 'USD'),
    ('Kraken', 'exchange', 'kraken', 'USD'),
    ('Paper Trading', 'paper', None, 'USD'),
]


class ConnectionPool:
    """
    SQLite connection pool for thread-safe database access.
    
    SQLite in Python is not thread-safe by default, so we maintain
    a pool of connections with one per thread.
    """
    
    def __init__(self, db_path, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: Queue = Queue(maxsize=pool_size)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._initialized = False
        self.logger = logging.getLogger('ConnectionPool')
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection"""
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,  # Allow multi-thread access with our pool
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # Better concurrent access
        conn.execute("PRAGMA busy_timeout = 5000")  # Wait up to 5s if locked
        return conn
    
    def initialize(self):
        """Pre-create connections for the pool"""
        with self._lock:
            if self._initialized:
                return
            for _ in range(self.pool_size):
                self._pool.put(self._create_connection())
            self._initialized = True
            self.logger.info(f"Connection pool initialized with {self.pool_size} connections")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool with health check"""
        if not self._initialized:
            self.initialize()
        
        # Try to get thread-local connection first
        if hasattr(self._local, 'connection') and self._local.connection:
            conn = self._local.connection
            # Validate the connection is still healthy
            if self._validate_connection(conn):
                return conn
            else:
                # Connection is stale, remove it
                self._local.connection = None
                self.logger.warning("Thread-local connection was stale, getting new one")
        
        # Get from pool
        conn = self._pool.get(timeout=10)
        
        # Validate connection before returning
        if not self._validate_connection(conn):
            self.logger.warning("Pool connection was stale, creating new one")
            try:
                conn.close()
            except:
                pass
            conn = self._create_connection()
        
        self._local.connection = conn
        return conn
    
    def _validate_connection(self, conn: sqlite3.Connection) -> bool:
        """Check if a connection is still healthy"""
        try:
            # Quick ping test
            cursor = conn.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except (sqlite3.Error, sqlite3.ProgrammingError) as e:
            self.logger.debug(f"Connection validation failed: {e}")
            return False
    
    def release_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool"""
        if hasattr(self._local, 'connection'):
            self._local.connection = None
        try:
            self._pool.put_nowait(conn)
        except:
            pass  # Pool full, close the connection
    
    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for getting a pooled connection"""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.release_connection(conn)
    
    def close_all(self):
        """Close all connections in the pool"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except:
                pass
        self._initialized = False
        self.logger.info("All pool connections closed")


class AccountingDatabase:
    """
    SQLite database manager for accounting data.
    Supports both single-connection and pooled modes.
    """

    def __init__(self, db_path: Optional[Path] = None, use_pool: bool = False, pool_size: int = 5):
        if db_path is None:
            db_path = get_project_path('CentralAccounting', 'data', 'accounting.db')
        
        # Handle string paths and special SQLite paths like :memory:
        if isinstance(db_path, str):
            if db_path == ':memory:':
                self.db_path = db_path
                use_pool = False  # Can't pool in-memory databases
            else:
                self.db_path = Path(db_path)
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.db_path = db_path
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._use_pool = use_pool
        self._pool: Optional[ConnectionPool] = None
        self._connection: Optional[sqlite3.Connection] = None
        self.logger = logging.getLogger('AccountingDB')
        
        if use_pool:
            self._pool = ConnectionPool(self.db_path, pool_size)

    def connect(self) -> sqlite3.Connection:
        """Get database connection (from pool if enabled)"""
        if self._use_pool and self._pool:
            return self._pool.get_connection()
        
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            self._connection.row_factory = sqlite3.Row
            # Enable foreign keys and WAL mode
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
        return self._connection
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections"""
        if self._use_pool and self._pool:
            with self._pool.connection() as conn:
                yield conn
        else:
            yield self.connect()

    def close(self):
        """Close database connection(s)"""
        if self._use_pool and self._pool:
            self._pool.close_all()
        elif self._connection:
            self._connection.close()
            self._connection = None

    # ==========================================
    # Async Wrapper Methods for Execution Engine
    # ==========================================

    async def execute_async(self, query: str, params: Optional[tuple] = None):
        """
        Execute a query asynchronously (wraps sync execution).
        For use with async execution engine.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_sync, query, params)

    def _execute_sync(self, query: str, params: Optional[tuple] = None):
        """Synchronous query execution"""
        conn = self.connect()
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        conn.commit()
        return cursor.lastrowid

    async def fetch_all_async(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """
        Fetch all results asynchronously.
        Returns list of dicts.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_all_sync, query, params)

    def _fetch_all_sync(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Synchronous fetch all"""
        conn = self.connect()
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    async def fetch_one_async(self, query: str, params: Optional[tuple] = None) -> Optional[Dict]:
        """
        Fetch one result asynchronously.
        Returns dict or None.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch_one_sync, query, params)

    def _fetch_one_sync(self, query: str, params: Optional[tuple] = None) -> Optional[Dict]:
        """Synchronous fetch one"""
        conn = self.connect()
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        row = cursor.fetchone()
        return dict(row) if row else None

    def initialize(self) -> bool:
        """
        Initialize the database schema.
        Creates tables if they don't exist.
        """
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Check current schema version
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='schema_version'
            """)
            
            if cursor.fetchone() is None:
                # Fresh database - create schema
                self.logger.info(f"Creating new database at {self.db_path}")
                cursor.executescript(SCHEMA_SQL)
                
                # Record schema version
                cursor.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,)
                )
                
                # Create default accounts
                for name, acc_type, exchange, currency in DEFAULT_ACCOUNTS:
                    cursor.execute("""
                        INSERT OR IGNORE INTO accounts (name, account_type, exchange, currency)
                        VALUES (?, ?, ?, ?)
                    """, (name, acc_type, exchange, currency))
                
                conn.commit()
                self.logger.info(f"Database initialized with schema version {SCHEMA_VERSION}")
            else:
                # Check for migrations
                cursor.execute("SELECT MAX(version) FROM schema_version")
                current_version = cursor.fetchone()[0] or 0
                
                if current_version < SCHEMA_VERSION:
                    self._run_migrations(current_version, SCHEMA_VERSION)
                else:
                    self.logger.info(f"Database already at version {current_version}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            return False

    def _run_migrations(self, from_version: int, to_version: int):
        """Run database migrations"""
        self.logger.info(f"Migrating database from v{from_version} to v{to_version}")
        
        conn = self.connect()
        cursor = conn.cursor()
        
        # Add migration logic here as schema evolves
        # Example:
        # if from_version < 2:
        #     cursor.execute("ALTER TABLE positions ADD COLUMN margin REAL")
        
        cursor.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (to_version,)
        )
        conn.commit()
        
        self.logger.info(f"Migration complete")

    # ==========================================
    # Account Operations
    # ==========================================

    def get_accounts(self, active_only: bool = True):
        """Get all accounts"""
        conn = self.connect()
        cursor = conn.cursor()
        
        if active_only:
            cursor.execute("SELECT * FROM accounts WHERE is_active = 1")
        else:
            cursor.execute("SELECT * FROM accounts")
        
        return [dict(row) for row in cursor.fetchall()]

    def get_account_by_exchange(self, exchange: str):
        """Get account for a specific exchange"""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM accounts WHERE exchange = ? AND is_active = 1",
            (exchange,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # ==========================================
    # Balance Operations
    # ==========================================

    def update_balance(
        self,
        account_id: int,
        asset: str,
        free_balance: float,
        locked_balance: float = 0
    ):
        """Update or insert balance for an account"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO balances (account_id, asset, free_balance, locked_balance, last_updated)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(account_id, asset) DO UPDATE SET
                free_balance = excluded.free_balance,
                locked_balance = excluded.locked_balance,
                last_updated = CURRENT_TIMESTAMP
        """, (account_id, asset, free_balance, locked_balance))
        
        conn.commit()

    def get_balances(self, account_id: Optional[int] = None):
        """Get balances, optionally filtered by account"""
        conn = self.connect()
        cursor = conn.cursor()
        
        if account_id:
            cursor.execute("""
                SELECT b.*, a.name as account_name, a.exchange
                FROM balances b
                JOIN accounts a ON b.account_id = a.account_id
                WHERE b.account_id = ? AND b.total_balance > 0
            """, (account_id,))
        else:
            cursor.execute("""
                SELECT b.*, a.name as account_name, a.exchange
                FROM balances b
                JOIN accounts a ON b.account_id = a.account_id
                WHERE b.total_balance > 0
            """)
        
        return [dict(row) for row in cursor.fetchall()]

    # ==========================================
    # Transaction Operations
    # ==========================================

    def record_transaction(
        self,
        account_id: int,
        transaction_type: str,
        asset: str,
        quantity: float,
        side: Optional[str] = None,
        symbol: Optional[str] = None,
        price: Optional[float] = None,
        total_value: Optional[float] = None,
        fee: float = 0,
        fee_currency: Optional[str] = None,
        external_id: Optional[str] = None,
        status: str = 'completed',
        notes: Optional[str] = None,
    ) -> int:
        """Record a transaction"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO transactions (
                account_id, transaction_type, side, symbol, asset, quantity,
                price, total_value, fee, fee_currency, external_id, status,
                notes, executed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            account_id, transaction_type, side, symbol, asset, quantity,
            price, total_value, fee, fee_currency, external_id, status, notes
        ))
        
        conn.commit()
        return cursor.lastrowid

    def get_transactions(
        self,
        account_id: Optional[int] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ):
        """Get recent transactions"""
        conn = self.connect()
        cursor = conn.cursor()
        
        query = "SELECT * FROM transactions WHERE 1=1"
        params = []
        
        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # ==========================================
    # Position Operations
    # ==========================================

    def open_position(
        self,
        account_id: int,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: float = 1,
        metadata: Optional[str] = None,
    ) -> int:
        """Open a new position"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO positions (
                account_id, symbol, side, quantity, entry_price,
                current_price, stop_loss, take_profit, leverage, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            account_id, symbol, side, quantity, entry_price,
            entry_price, stop_loss, take_profit, leverage, metadata
        ))
        
        conn.commit()
        return cursor.lastrowid

    def update_position(
        self,
        position_id: int,
        current_price: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ):
        """Update position with current market data"""
        conn = self.connect()
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if current_price is not None:
            updates.append("current_price = ?")
            params.append(current_price)
        if unrealized_pnl is not None:
            updates.append("unrealized_pnl = ?")
            params.append(unrealized_pnl)
        if stop_loss is not None:
            updates.append("stop_loss = ?")
            params.append(stop_loss)
        if take_profit is not None:
            updates.append("take_profit = ?")
            params.append(take_profit)
        
        if updates:
            params.append(position_id)
            cursor.execute(
                f"UPDATE positions SET {', '.join(updates)} WHERE position_id = ?",
                params
            )
            conn.commit()

    def close_position(
        self,
        position_id: int,
        exit_price: float,
        realized_pnl: float,
        fees: float = 0,
    ):
        """Close a position and record in trade history"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Get position details
        cursor.execute("SELECT * FROM positions WHERE position_id = ?", (position_id,))
        position = cursor.fetchone()
        
        if not position:
            raise ValueError(f"Position {position_id} not found")
        
        # Calculate hold duration
        opened_at = datetime.fromisoformat(position['opened_at']) if position['opened_at'] else datetime.now()
        hold_duration = int((datetime.now() - opened_at).total_seconds())
        
        # Record in trade history
        cursor.execute("""
            INSERT INTO trade_history (
                position_id, account_id, symbol, side, entry_price, exit_price,
                quantity, gross_pnl, fees, hold_duration_seconds, opened_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position_id, position['account_id'], position['symbol'],
            position['side'], position['entry_price'], exit_price,
            position['quantity'], realized_pnl, fees, hold_duration, position['opened_at']
        ))
        
        # Update position status
        cursor.execute("""
            UPDATE positions SET
                status = 'closed',
                realized_pnl = ?,
                closed_at = CURRENT_TIMESTAMP
            WHERE position_id = ?
        """, (realized_pnl, position_id))
        
        conn.commit()

    def get_open_positions(self, account_id: Optional[int] = None):
        """Get all open positions"""
        conn = self.connect()
        cursor = conn.cursor()
        
        if account_id:
            cursor.execute("""
                SELECT p.*, a.name as account_name
                FROM positions p
                JOIN accounts a ON p.account_id = a.account_id
                WHERE p.status = 'open' AND p.account_id = ?
            """, (account_id,))
        else:
            cursor.execute("""
                SELECT p.*, a.name as account_name
                FROM positions p
                JOIN accounts a ON p.account_id = a.account_id
                WHERE p.status = 'open'
            """)
        
        return [dict(row) for row in cursor.fetchall()]

    # ==========================================
    # P&L Operations
    # ==========================================

    def record_daily_pnl(
        self,
        date: str,
        starting_equity: float,
        ending_equity: float,
        realized_pnl: float,
        unrealized_pnl: float,
        fees_paid: float,
        trades_count: int,
        win_count: int,
        loss_count: int,
        largest_win: float,
        largest_loss: float,
    ):
        """Record daily P&L snapshot"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO daily_pnl (
                date, starting_equity, ending_equity, realized_pnl,
                unrealized_pnl, fees_paid, trades_count, win_count,
                loss_count, largest_win, largest_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date, starting_equity, ending_equity, realized_pnl,
            unrealized_pnl, fees_paid, trades_count, win_count,
            loss_count, largest_win, largest_loss
        ))
        
        conn.commit()

    def get_daily_pnl(self, days: int = 30):
        """Get daily P&L for the last N days"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM daily_pnl
            ORDER BY date DESC
            LIMIT ?
        """, (days,))
        
        return [dict(row) for row in cursor.fetchall()]

    def get_pnl_summary(self):
        """Get overall P&L summary"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT
                COUNT(*) as total_days,
                SUM(realized_pnl) as total_realized_pnl,
                SUM(fees_paid) as total_fees,
                SUM(trades_count) as total_trades,
                SUM(win_count) as total_wins,
                SUM(loss_count) as total_losses,
                AVG(realized_pnl) as avg_daily_pnl,
                MAX(largest_win) as best_trade,
                MIN(largest_loss) as worst_trade
            FROM daily_pnl
        """)
        
        row = cursor.fetchone()
        return dict(row) if row else {}


def init_database():
    """Initialize the accounting database"""
    db = AccountingDatabase()
    success = db.initialize()
    db.close()
    return success


# CLI
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing Accounting Database...")
    db = AccountingDatabase()
    
    if db.initialize():
        print(f"Database created at: {db.db_path}")
        
        # Show accounts
        accounts = db.get_accounts()
        print(f"\nAccounts ({len(accounts)}):")
        for acc in accounts:
            print(f"  - {acc['name']} ({acc['account_type']})")
        
        # Test creating a transaction
        if accounts:
            tx_id = db.record_transaction(
                account_id=accounts[0]['account_id'],
                transaction_type='deposit',
                asset='USD',
                quantity=10000,
                notes='Initial paper trading balance'
            )
            print(f"\nCreated test transaction: {tx_id}")
        
        db.close()
        print("\nDatabase initialization complete!")
    else:
        print("Database initialization failed!")
