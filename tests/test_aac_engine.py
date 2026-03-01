import pytest
from aac_engine import AccountingEngine, Account, Transaction, AccountingError
from decimal import Decimal
from datetime import datetime, date
import os

@pytest.fixture
def sample_account():
    return Account(account_id="A001", name="Cash", account_type="Asset", description="Main Cash Account")

@pytest.fixture
def sample_transaction():
    entries = [
        {'account_id': "A001", 'debit': Decimal('100.00'), 'credit': Decimal('0.00')},
        {'account_id': "A002", 'debit': Decimal('0.00'), 'credit': Decimal('100.00')}
    ]
    return Transaction(transaction_id="T001", date=date.today(), description="Service Payment", entries=entries)

@pytest.fixture
def accounting_engine():
    engine = AccountingEngine(db_path=":memory:")  # Use in-memory database for testing
    yield engine
    engine.conn.close()  # Clean up the connection after tests

def test_account_creation(sample_account):
    account = sample_account
    assert account.account_id == "A001", "Account ID should be 'A001'"
    assert account.name == "Cash", "Account name should be 'Cash'"
    assert account.account_type == "Asset", "Account type should be 'Asset'"
    assert account.balance == Decimal('0.00'), "Initial balance should be zero"

def test_account_to_dict(sample_account):
    account = sample_account.to_dict()
    assert account['account_id'] == "A001", "Account ID should be 'A001'"
    assert account['name'] == "Cash", "Account name should be 'Cash'"
    assert 'created_date' in account, "'created_date' should be present in account dictionary"

def test_transaction_creation(sample_transaction):
    transaction = sample_transaction
    assert transaction.transaction_id == "T001", "Transaction ID should be 'T001'"
    assert transaction.description == "Service Payment", "Transaction description should be 'Service Payment'"
    assert len(transaction.entries) == 2, "Transaction should have 2 entries"
    
def test_transaction_balancing(sample_transaction):
    transaction = sample_transaction
    assert transaction.is_balanced(), "Transaction should be balanced"

def test_transaction_unbalanced():
    entries = [
        {'account_id': "A001", 'debit': Decimal('50.00'), 'credit': Decimal('0.00')},
        {'account_id': "A002", 'debit': Decimal('0.00'), 'credit': Decimal('100.00')}
    ]
    transaction = Transaction(transaction_id="T002", date=date.today(), description="Unbalanced Transaction", entries=entries)

    assert not transaction.is_balanced(), "Transaction should not be balanced"

@pytest.mark.parametrize("debit, credit", [
    (Decimal('100.00'), Decimal('50.00')),
    (Decimal('0.00'), Decimal('100.00'))
])
def test_transaction_total_debit_and_credit(debit, credit):
    entries = [{'account_id': "A001", 'debit': debit, 'credit': credit}]
    transaction = Transaction(transaction_id="T003", date=date.today(), description="Debit Credit Test", entries=entries)

    assert transaction.get_total_debit() == debit, f"Total debit should be {debit}"
    assert transaction.get_total_credit() == credit, f"Total credit should be {credit}"

def test_initialize_database(accounting_engine):
    engine = accounting_engine
    assert engine.db_path == ":memory:", "Database path should be ':memory:' for testing"
    assert engine.conn is not None, "Database connection should be established"

def test_missing_db_initialization():
    # Trying to initialize AccountingEngine without a proper database path
    with pytest.raises(sqlite3.OperationalError):
        AccountingEngine(db_path="/invalid/path/to/database.db")

# If needed, additional tests for the rest of the module can go here.
