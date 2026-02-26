#!/usr/bin/env python3
"""
Paper Trading Account Manager
============================

Manages virtual trading accounts for safe strategy validation.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid

@dataclass
class PaperAccount:
    """Virtual trading account"""
    account_id: str
    balance: float
    positions: Dict[str, Dict[str, Any]]
    orders: List[Dict[str, Any]]
    pnl: float
    created_at: datetime
    last_updated: datetime

class PaperAccountManager:
    """Manages paper trading accounts"""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/paper_trading")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.accounts_file = self.data_dir / "accounts.json"
        self.accounts: Dict[str, PaperAccount] = {}

    async def initialize(self):
        """Initialize paper trading accounts"""
        await self._load_accounts()

        # Create default account if none exist
        if not self.accounts:
            await self.create_account("default", 100000.0)

    async def create_account(self, name: str, initial_balance: float) -> str:
        """Create a new paper trading account"""
        account_id = f"paper_{name}_{uuid.uuid4().hex[:8]}"

        account = PaperAccount(
            account_id=account_id,
            balance=initial_balance,
            positions={},
            orders=[],
            pnl=0.0,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        self.accounts[account_id] = account
        await self._save_accounts()

        return account_id

    async def get_account(self, account_id: str) -> Optional[PaperAccount]:
        """Get account by ID"""
        return self.accounts.get(account_id)

    async def update_balance(self, account_id: str, amount: float):
        """Update account balance"""
        if account_id in self.accounts:
            self.accounts[account_id].balance += amount
            self.accounts[account_id].last_updated = datetime.now()
            await self._save_accounts()

    async def add_position(self, account_id: str, symbol: str, quantity: float, price: float):
        """Add or update position"""
        if account_id in self.accounts:
            account = self.accounts[account_id]

            if symbol not in account.positions:
                account.positions[symbol] = {
                    'quantity': 0,
                    'avg_price': 0,
                    'current_price': price
                }

            position = account.positions[symbol]
            total_quantity = position['quantity'] + quantity
            total_cost = (position['quantity'] * position['avg_price']) + (quantity * price)

            if total_quantity != 0:
                position['avg_price'] = total_cost / total_quantity
            else:
                position['avg_price'] = 0

            position['quantity'] = total_quantity
            position['current_price'] = price

            account.last_updated = datetime.now()
            await self._save_accounts()

    async def _load_accounts(self):
        """Load accounts from file"""
        if self.accounts_file.exists():
            try:
                with open(self.accounts_file, 'r') as f:
                    data = json.load(f)

                for account_data in data.values():
                    # Convert datetime strings back to datetime objects
                    account_data['created_at'] = datetime.fromisoformat(account_data['created_at'])
                    account_data['last_updated'] = datetime.fromisoformat(account_data['last_updated'])

                    account = PaperAccount(**account_data)
                    self.accounts[account.account_id] = account

            except Exception as e:
                print(f"Error loading accounts: {e}")

    async def _save_accounts(self):
        """Save accounts to file"""
        data = {}
        for account_id, account in self.accounts.items():
            account_dict = asdict(account)
            # Convert datetime to ISO format
            account_dict['created_at'] = account.created_at.isoformat()
            account_dict['last_updated'] = account.last_updated.isoformat()
            data[account_id] = account_dict

        with open(self.accounts_file, 'w') as f:
            json.dump(data, f, indent=2)
