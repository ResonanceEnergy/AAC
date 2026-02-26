"""
Internal Money Monitor
======================

Mock implementation for financial monitoring and account management.
"""

import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class Account:
    """Internal account representation"""
    account_code: str
    department: str
    balance: float
    available_balance: float
    account_type: str = "checking"
    status: str = "active"

class InternalMoneyMonitor:
    """Mock internal money monitoring system"""

    def __init__(self):
        self.accounts = {}
        self.transactions = []
        self._initialize_mock_accounts()

    def _initialize_mock_accounts(self):
        """Initialize mock accounts for testing"""
        mock_accounts = [
            Account("AAC-001", "TradingExecution", 1000000.0, 950000.0),
            Account("AAC-002", "CryptoIntelligence", 500000.0, 480000.0),
            Account("AAC-003", "CentralAccounting", 2000000.0, 1950000.0),
            Account("AAC-004", "SharedInfrastructure", 300000.0, 290000.0),
            Account("AAC-005", "NCC", 1500000.0, 1480000.0),
            Account("AAC-006", "BigBrainIntelligence", 750000.0, 740000.0),
            Account("AAC-007", "Doctrine", 250000.0, 245000.0),
            Account("AAC-008", "Bakeoff", 100000.0, 98000.0),
            Account("AAC-009", "Integration", 400000.0, 395000.0),
            Account("AAC-010", "Reserve", 5000000.0, 4950000.0)
        ]

        for account in mock_accounts:
            self.accounts[account.account_code] = account

    async def get_all_accounts(self) -> List[Dict[str, Any]]:
        """Get all accounts as dictionaries"""
        return [
            {
                "account_code": acc.account_code,
                "department": acc.department,
                "balance": acc.balance,
                "available_balance": acc.available_balance,
                "account_type": acc.account_type,
                "status": acc.status
            }
            for acc in self.accounts.values()
        ]

    async def get_account_balance(self, account_code: str) -> float:
        """Get account balance"""
        if account_code in self.accounts:
            return self.accounts[account_code].balance
        return 0.0

    async def transfer_funds(self, from_account: str, to_account: str, amount: float) -> bool:
        """Transfer funds between accounts"""
        if (from_account in self.accounts and
            to_account in self.accounts and
            self.accounts[from_account].available_balance >= amount):

            self.accounts[from_account].balance -= amount
            self.accounts[from_account].available_balance -= amount
            self.accounts[to_account].balance += amount
            self.accounts[to_account].available_balance += amount

            # Record transaction
            self.transactions.append({
                "from_account": from_account,
                "to_account": to_account,
                "amount": amount,
                "timestamp": asyncio.get_event_loop().time(),
                "type": "transfer"
            })

            return True
        return False

    async def approve_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Approve a transaction"""
        # Simple approval logic
        amount = transaction.get("amount", 0)
        account_code = transaction.get("account_code", "")

        if account_code in self.accounts:
            available_balance = self.accounts[account_code].available_balance
            if available_balance >= amount:
                return {
                    "approved": True,
                    "approval_code": f"APPROVED-{len(self.transactions)}",
                    "available_balance": available_balance,
                    "remaining_balance": available_balance - amount
                }

        return {
            "approved": False,
            "reason": "Insufficient funds or invalid account",
            "available_balance": self.accounts.get(account_code, Account("", "", 0, 0)).available_balance
        }

# Global instance
_money_monitor = None

def get_money_monitor() -> InternalMoneyMonitor:
    """Get the global money monitor instance"""
    global _money_monitor
    if _money_monitor is None:
        _money_monitor = InternalMoneyMonitor()
    return _money_monitor