#!/usr/bin/env python3
"""
EQ Bank Funding Integration
============================
Integration module for EQ Bank (Canadian digital bank) to track and manage
funding flows to/from brokerage accounts.

EQ Bank does not have a public trading API, so this module provides:
    1. Funding tracker — record EFT/wire transfers to broker accounts
    2. Balance tracking — monitor available funding
    3. Transfer history — audit trail of all funding movements
    4. Broker account mapping — which broker accounts are funded from EQ

For automated EFT transfers, consider using Plaid for bank connectivity
or Flinks (Canadian-focused bank aggregation).

Configuration via .env:
    EQ_BANK_ENABLED=true
    EQ_BANK_ACCOUNT_LABEL=EQ Bank Savings Plus
    EQ_BANK_LINKED_BROKERS=ibkr,moomoo,ndax
    PLAID_CLIENT_ID=             # Optional: Plaid for automated bank reads
    PLAID_SECRET=                # Optional: Plaid secret
    PLAID_ENV=sandbox            # sandbox | development | production
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_env, get_env_bool

logger = logging.getLogger(__name__)


class TransferStatus(Enum):
    """Status of a funding transfer."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TransferType(Enum):
    """Type of transfer."""
    EFT = "eft"              # Electronic Funds Transfer (2-3 business days)
    WIRE = "wire"            # Wire transfer (same day)
    INTERNAL = "internal"    # Within same institution
    MANUAL = "manual"        # Manually recorded


@dataclass
class FundingTransfer:
    """Record of a funding transfer between EQ Bank and a broker."""
    transfer_id: str
    timestamp: str
    from_account: str
    to_account: str
    amount: float
    currency: str = "CAD"
    transfer_type: str = "eft"
    status: str = "pending"
    reference: str = ""
    notes: str = ""
    broker: str = ""
    estimated_arrival: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """To dict."""
        return asdict(self)


@dataclass
class BrokerFundingAccount:
    """Mapping of a broker account that receives funding from EQ Bank."""
    broker: str
    account_id: str
    account_name: str
    currency: str = "CAD"
    total_funded: float = 0.0
    last_transfer: str = ""
    enabled: bool = True


class EQBankFundingManager:
    """
    Manages funding flows from EQ Bank to trading broker accounts.

    Provides tracking, recording, and audit trail for all funding movements.
    Does NOT execute bank transactions — records must be created manually
    or via Plaid integration when available.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.enabled = get_env_bool('EQ_BANK_ENABLED', False)
        self.account_label = get_env('EQ_BANK_ACCOUNT_LABEL', 'EQ Bank Savings Plus')
        self._linked_brokers_str = get_env('EQ_BANK_LINKED_BROKERS', 'ibkr,moomoo,ndax')

        # Data persistence
        if data_dir:
            self._data_dir = Path(data_dir)
        else:
            self._data_dir = Path(PROJECT_ROOT) / 'data' / 'funding'
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._transfers_file = self._data_dir / 'eq_bank_transfers.json'
        self._accounts_file = self._data_dir / 'broker_funding_accounts.json'

        # In-memory state
        self._transfers: List[FundingTransfer] = []
        self._broker_accounts: Dict[str, BrokerFundingAccount] = {}

        # Load persisted data
        self._load_data()

        if self.enabled:
            logger.info(
                f"EQ Bank Funding Manager initialized: {self.account_label} "
                f"-> linked brokers: {self._linked_brokers_str}"
            )

    @property
    def linked_brokers(self) -> List[str]:
        """Get list of linked broker names."""
        return [b.strip() for b in self._linked_brokers_str.split(',') if b.strip()]

    def _load_data(self) -> None:
        """Load persisted transfer and account data."""
        if self._transfers_file.exists():
            try:
                with open(self._transfers_file, 'r') as f:
                    data = json.load(f)
                self._transfers = [FundingTransfer(**t) for t in data]
                logger.info(f"Loaded {len(self._transfers)} transfer records")
            except Exception as e:
                logger.warning(f"Failed to load transfers: {e}")

        if self._accounts_file.exists():
            try:
                with open(self._accounts_file, 'r') as f:
                    data = json.load(f)
                self._broker_accounts = {
                    a['broker']: BrokerFundingAccount(**a)
                    for a in data
                }
            except Exception as e:
                logger.warning(f"Failed to load broker accounts: {e}")

    def _save_transfers(self) -> None:
        """Persist transfer records."""
        try:
            with open(self._transfers_file, 'w') as f:
                json.dump([t.to_dict() for t in self._transfers], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save transfers: {e}")

    def _save_accounts(self) -> None:
        """Persist broker account mappings."""
        try:
            with open(self._accounts_file, 'w') as f:
                json.dump([asdict(a) for a in self._broker_accounts.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save broker accounts: {e}")

    def register_broker_account(
        self,
        broker: str,
        account_id: str,
        account_name: str = "",
        currency: str = "CAD",
    ) -> BrokerFundingAccount:
        """Register a broker account for funding from EQ Bank."""
        account = BrokerFundingAccount(
            broker=broker,
            account_id=account_id,
            account_name=account_name or f"{broker.upper()} Trading Account",
            currency=currency,
        )
        self._broker_accounts[broker] = account
        self._save_accounts()
        logger.info(f"Registered broker funding account: {broker} -> {account_id}")
        return account

    def record_transfer(
        self,
        to_broker: str,
        amount: float,
        currency: str = "CAD",
        transfer_type: str = "eft",
        reference: str = "",
        notes: str = "",
    ) -> FundingTransfer:
        """
        Record a funding transfer from EQ Bank to a broker account.

        This creates a record — the actual bank transfer must be initiated
        through EQ Bank's banking interface or via Plaid.
        """
        if amount <= 0:
            raise ValueError("Transfer amount must be positive")

        transfer_id = f"EQ-{to_broker.upper()}-{int(time.time())}"
        now = datetime.now().isoformat()

        to_account = ""
        if to_broker in self._broker_accounts:
            to_account = self._broker_accounts[to_broker].account_id

        transfer = FundingTransfer(
            transfer_id=transfer_id,
            timestamp=now,
            from_account=self.account_label,
            to_account=to_account or f"{to_broker.upper()} Account",
            amount=amount,
            currency=currency,
            transfer_type=transfer_type,
            status="pending",
            reference=reference,
            notes=notes,
            broker=to_broker,
        )

        self._transfers.append(transfer)

        # Update broker account totals
        if to_broker in self._broker_accounts:
            self._broker_accounts[to_broker].total_funded += amount
            self._broker_accounts[to_broker].last_transfer = now
            self._save_accounts()

        self._save_transfers()
        logger.info(f"Recorded transfer {transfer_id}: ${amount:.2f} {currency} -> {to_broker}")
        return transfer

    def update_transfer_status(self, transfer_id: str, status: str) -> Optional[FundingTransfer]:
        """Update the status of a transfer."""
        valid_statuses = {s.value for s in TransferStatus}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status '{status}'. Valid: {valid_statuses}")

        for transfer in self._transfers:
            if transfer.transfer_id == transfer_id:
                transfer.status = status
                self._save_transfers()
                logger.info(f"Transfer {transfer_id} status -> {status}")
                return transfer
        return None

    def get_transfers(
        self,
        broker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[FundingTransfer]:
        """Get transfer history with optional filters."""
        result = self._transfers
        if broker:
            result = [t for t in result if t.broker == broker]
        if status:
            result = [t for t in result if t.status == status]
        return sorted(result, key=lambda t: t.timestamp, reverse=True)[:limit]

    def get_funding_summary(self) -> Dict[str, Any]:
        """Get a summary of all funding activity."""
        total_funded = sum(t.amount for t in self._transfers if t.status == 'completed')
        pending = sum(t.amount for t in self._transfers if t.status == 'pending')

        by_broker = {}
        for t in self._transfers:
            if t.status == 'completed':
                by_broker.setdefault(t.broker, 0.0)
                by_broker[t.broker] += t.amount

        return {
            'account': self.account_label,
            'total_funded': total_funded,
            'pending_transfers': pending,
            'total_transfers': len(self._transfers),
            'by_broker': by_broker,
            'linked_brokers': self.linked_brokers,
            'registered_accounts': {
                k: asdict(v) for k, v in self._broker_accounts.items()
            },
        }

    def get_broker_funding_status(self, broker: str) -> Dict[str, Any]:
        """Get funding status for a specific broker."""
        transfers = self.get_transfers(broker=broker)
        account = self._broker_accounts.get(broker)

        completed = [t for t in transfers if t.status == 'completed']
        pending = [t for t in transfers if t.status == 'pending']

        return {
            'broker': broker,
            'account': asdict(account) if account else None,
            'total_funded': sum(t.amount for t in completed),
            'pending_amount': sum(t.amount for t in pending),
            'total_transfers': len(transfers),
            'last_transfer': transfers[0].to_dict() if transfers else None,
        }
