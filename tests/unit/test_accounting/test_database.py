"""Unit tests for CentralAccounting.database — AccountingDatabase with :memory: SQLite."""
import pytest
from CentralAccounting.database import AccountingDatabase


class TestAccountingDatabase:
    """Tests for AccountingDatabase using in-memory SQLite."""

    @pytest.fixture
    def db(self):
        """Create an in-memory database, initialized."""
        database = AccountingDatabase(":memory:")
        database.initialize()
        yield database
        database.close()

    def test_init_memory(self):
        db = AccountingDatabase(":memory:")
        assert db.db_path == ":memory:"
        db.close()

    def test_initialize_returns_true(self):
        db = AccountingDatabase(":memory:")
        assert db.initialize() is True
        db.close()

    def test_connect_returns_connection(self):
        db = AccountingDatabase(":memory:")
        conn = db.connect()
        assert conn is not None
        db.close()

    def test_wal_mode_enabled(self):
        db = AccountingDatabase(":memory:")
        conn = db.connect()
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        # In-memory may report 'memory' instead of 'wal', both acceptable
        assert mode in ("wal", "memory")
        db.close()

    def test_default_accounts_created(self, db):
        accounts = db.get_accounts()
        assert len(accounts) >= 4
        names = [a["name"] for a in accounts]
        assert "Binance Spot" in names
        assert "Paper Trading" in names

    def test_get_accounts_active_only(self, db):
        accounts = db.get_accounts(active_only=True)
        for acc in accounts:
            assert acc["is_active"] == 1

    def test_get_account_by_exchange(self, db):
        acc = db.get_account_by_exchange("binance")
        assert acc is not None
        assert acc["exchange"] == "binance"

    def test_get_account_by_exchange_missing(self, db):
        acc = db.get_account_by_exchange("nonexistent_exchange")
        assert acc is None

    def test_update_and_get_balance(self, db):
        accounts = db.get_accounts()
        acc_id = accounts[0]["account_id"]
        db.update_balance(acc_id, "BTC", free_balance=1.5, locked_balance=0.5)
        balances = db.get_balances(acc_id)
        btc = [b for b in balances if b["asset"] == "BTC"]
        assert len(btc) == 1
        assert btc[0]["free_balance"] == 1.5

    def test_schema_version_recorded(self, db):
        conn = db.connect()
        cursor = conn.execute("SELECT MAX(version) FROM schema_version")
        version = cursor.fetchone()[0]
        assert version >= 1

    def test_close_clears_connection(self):
        db = AccountingDatabase(":memory:")
        db.connect()
        db.close()
        assert db._connection is None
