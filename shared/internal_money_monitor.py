"""Internal Money Monitor — live IBKR-backed account monitoring.

Connects to IBKR via ib_insync to pull real account balances,
positions, and P&L. Falls back to cached data on connection failure.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


@dataclass
class Account:
    """Internal account representation."""

    account_code: str
    department: str
    balance: float
    available_balance: float
    account_type: str = "brokerage"
    status: str = "active"
    currency: str = "CAD"
    last_updated: str = ""


@dataclass
class LiveAccountData:
    """Parsed IBKR account summary."""

    net_liquidation: float = 0.0
    total_cash: float = 0.0
    buying_power: float = 0.0
    gross_position_value: float = 0.0
    maintenance_margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    available_funds: float = 0.0
    excess_liquidity: float = 0.0
    cushion: float = 0.0
    currency: str = "CAD"
    account_id: str = ""
    positions: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""


class InternalMoneyMonitor:
    """IBKR-backed money monitor with fallback to last-known values.

    Parameters
    ----------
    ibkr_host : str
        TWS/Gateway host.
    ibkr_port : int
        TWS/Gateway port (7496=live, 7497=paper).
    ibkr_client_id : int
        Client ID for this connection (use a unique ID to avoid conflicts).
    account_id : str
        IBKR account ID (e.g., "U24346218").
    cache_ttl : int
        How long (seconds) to use cached data before refreshing.
    """

    def __init__(
        self,
        ibkr_host: str = "127.0.0.1",
        ibkr_port: int = 0,
        ibkr_client_id: int = 10,
        account_id: str = "",
        cache_ttl: int = 60,
    ) -> None:
        self.ibkr_host = ibkr_host
        self.ibkr_port = ibkr_port or int(os.environ.get("IBKR_PORT", "7497"))
        self.ibkr_client_id = ibkr_client_id
        self.account_id = account_id or os.environ.get("IBKR_ACCOUNT", "U24346218")
        self.cache_ttl = cache_ttl

        self.accounts: dict[str, Account] = {}
        self.transactions: list[dict[str, Any]] = []
        self._live_data: Optional[LiveAccountData] = None
        self._last_fetch: float = 0.0
        self._is_live: bool = False

    # ── IBKR connection ───────────────────────────────────────────────────

    async def _fetch_ibkr_data(self) -> Optional[LiveAccountData]:
        """Connect to IBKR and pull account summary + positions."""
        try:
            from ib_insync import IB
        except ImportError:
            _log.debug("ib_insync not installed, using fallback")
            return None

        ib = IB()
        try:
            await asyncio.wait_for(
                ib.connectAsync(
                    self.ibkr_host,
                    self.ibkr_port,
                    clientId=self.ibkr_client_id,
                ),
                timeout=10.0,
            )
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as exc:
            _log.warning("ibkr_connect_failed", error=str(exc))
            return None

        try:
            # Request account summary
            summary = ib.accountSummary(self.account_id)
            data = LiveAccountData(
                account_id=self.account_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            field_map = {
                "NetLiquidation": "net_liquidation",
                "TotalCashValue": "total_cash",
                "BuyingPower": "buying_power",
                "GrossPositionValue": "gross_position_value",
                "MaintMarginReq": "maintenance_margin",
                "UnrealizedPnL": "unrealized_pnl",
                "RealizedPnL": "realized_pnl",
                "AvailableFunds": "available_funds",
                "ExcessLiquidity": "excess_liquidity",
                "Cushion": "cushion",
            }

            for item in summary:
                attr = field_map.get(item.tag)
                if attr:
                    try:
                        setattr(data, attr, float(item.value))
                    except (ValueError, TypeError):
                        pass
                if item.tag == "Currency":
                    data.currency = item.value

            # Fetch positions
            positions = ib.positions(self.account_id)
            for pos in positions:
                data.positions.append({
                    "symbol": pos.contract.symbol,
                    "sec_type": pos.contract.secType,
                    "quantity": float(pos.position),
                    "avg_cost": float(pos.avgCost),
                    "market_value": float(pos.position) * float(pos.avgCost),
                })

            self._is_live = True
            _log.info(
                "ibkr_data_fetched",
                net_liq=round(data.net_liquidation, 2),
                cash=round(data.total_cash, 2),
                positions=len(data.positions),
            )
            return data

        finally:
            ib.disconnect()

    async def _ensure_data(self) -> LiveAccountData:
        """Refresh from IBKR if cache expired, or return cached."""
        now = time.monotonic()
        if self._live_data and (now - self._last_fetch) < self.cache_ttl:
            return self._live_data

        fresh = await self._fetch_ibkr_data()
        if fresh:
            self._live_data = fresh
            self._last_fetch = now
            self._sync_accounts(fresh)
            return fresh

        # Fallback to last-known or empty
        if self._live_data:
            _log.info("using_cached_ibkr_data", age_s=round(now - self._last_fetch, 0))
            return self._live_data

        _log.warning("no_ibkr_data_available")
        return LiveAccountData()

    def _sync_accounts(self, data: LiveAccountData) -> None:
        """Sync IBKR data into the Account objects."""
        ts = data.timestamp
        self.accounts["IBKR-MAIN"] = Account(
            account_code="IBKR-MAIN",
            department="TradingExecution",
            balance=data.net_liquidation,
            available_balance=data.available_funds,
            account_type="brokerage",
            status="active",
            currency=data.currency,
            last_updated=ts,
        )
        self.accounts["IBKR-CASH"] = Account(
            account_code="IBKR-CASH",
            department="TradingExecution",
            balance=data.total_cash,
            available_balance=data.total_cash,
            account_type="cash",
            status="active",
            currency=data.currency,
            last_updated=ts,
        )

    # ── Public API (same interface as old mock) ───────────────────────────

    async def get_all_accounts(self) -> List[Dict[str, Any]]:
        """Get all accounts as dictionaries."""
        await self._ensure_data()
        return [
            {
                "account_code": acc.account_code,
                "department": acc.department,
                "balance": acc.balance,
                "available_balance": acc.available_balance,
                "account_type": acc.account_type,
                "status": acc.status,
                "currency": acc.currency,
                "last_updated": acc.last_updated,
            }
            for acc in self.accounts.values()
        ]

    async def get_account_balance(self, account_code: str) -> float:
        """Get account balance."""
        await self._ensure_data()
        if account_code in self.accounts:
            return self.accounts[account_code].balance
        # Legacy codes map to IBKR-MAIN
        if account_code == "AAC-001":
            return self.accounts.get("IBKR-MAIN", Account("", "", 0.0, 0.0)).balance
        return 0.0

    async def get_live_snapshot(self) -> LiveAccountData:
        """Get full IBKR snapshot including positions."""
        return await self._ensure_data()

    async def transfer_funds(self, from_account: str, to_account: str, amount: float) -> bool:
        """Transfer funds between accounts (internal ledger only)."""
        if (
            from_account in self.accounts
            and to_account in self.accounts
            and self.accounts[from_account].available_balance >= amount
        ):
            self.accounts[from_account].balance -= amount
            self.accounts[from_account].available_balance -= amount
            self.accounts[to_account].balance += amount
            self.accounts[to_account].available_balance += amount
            self.transactions.append({
                "from_account": from_account,
                "to_account": to_account,
                "amount": amount,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "transfer",
            })
            return True
        return False

    async def approve_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Approve a transaction against real available funds."""
        amount = transaction.get("amount", 0)
        account_code = transaction.get("account_code", "")

        await self._ensure_data()

        acct = self.accounts.get(account_code) or self.accounts.get("IBKR-MAIN")
        if acct and acct.available_balance >= amount:
            return {
                "approved": True,
                "approval_code": f"APPROVED-{len(self.transactions)}",
                "available_balance": acct.available_balance,
                "remaining_balance": acct.available_balance - amount,
                "live": self._is_live,
            }

        avail = acct.available_balance if acct else 0.0
        return {
            "approved": False,
            "reason": "Insufficient funds or invalid account",
            "available_balance": avail,
            "live": self._is_live,
        }

    @property
    def is_live(self) -> bool:
        """Whether we're running against real IBKR data."""
        return self._is_live


# Global instance
_money_monitor: Optional[InternalMoneyMonitor] = None


def get_money_monitor() -> InternalMoneyMonitor:
    """Get the global money monitor instance."""
    global _money_monitor
    if _money_monitor is None:
        _money_monitor = InternalMoneyMonitor()
    return _money_monitor
