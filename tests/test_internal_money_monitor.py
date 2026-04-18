"""Tests for shared/internal_money_monitor.py — IBKR-backed P&L monitor."""
from __future__ import annotations

import pytest


def _set_ibkr_env(monkeypatch):
    monkeypatch.setenv("IBKR_ACCOUNT", "U24346218")
    monkeypatch.setenv("IBKR_PORT", "7497")


class TestInternalMoneyMonitor:
    def test_init(self, monkeypatch):
        _set_ibkr_env(monkeypatch)
        from shared.internal_money_monitor import InternalMoneyMonitor
        monitor = InternalMoneyMonitor()
        assert monitor is not None
        assert monitor.account_id == "U24346218"
        assert monitor.ibkr_port == 7497

    @pytest.mark.asyncio
    async def test_get_live_snapshot_connected(self, monkeypatch):
        _set_ibkr_env(monkeypatch)
        from shared.internal_money_monitor import InternalMoneyMonitor, LiveAccountData
        monitor = InternalMoneyMonitor()

        async def mock_fetch():
            return LiveAccountData(
                net_liquidation=25000.0,
                total_cash=5000.0,
                buying_power=15000.0,
                available_funds=12000.0,
                account_id="U24346218",
                positions=[{"symbol": "ARCC", "sec_type": "OPT", "quantity": -1, "avg_cost": 25.0, "market_value": -25.0}],
            )
        monitor._fetch_ibkr_data = mock_fetch
        snapshot = await monitor.get_live_snapshot()
        assert snapshot.net_liquidation == 25000.0
        assert snapshot.total_cash == 5000.0

    @pytest.mark.asyncio
    async def test_get_live_snapshot_disconnected(self, monkeypatch):
        _set_ibkr_env(monkeypatch)
        from shared.internal_money_monitor import InternalMoneyMonitor
        monitor = InternalMoneyMonitor()

        async def mock_fetch():
            return None
        monitor._fetch_ibkr_data = mock_fetch

        snapshot = await monitor.get_live_snapshot()
        assert snapshot.net_liquidation == 0.0

    @pytest.mark.asyncio
    async def test_get_all_accounts(self, monkeypatch):
        _set_ibkr_env(monkeypatch)
        from shared.internal_money_monitor import InternalMoneyMonitor, LiveAccountData
        monitor = InternalMoneyMonitor()

        async def mock_fetch():
            return LiveAccountData(
                net_liquidation=25000.0,
                total_cash=5000.0,
                available_funds=12000.0,
                account_id="U24346218",
            )
        monitor._fetch_ibkr_data = mock_fetch

        accounts = await monitor.get_all_accounts()
        assert isinstance(accounts, list)
        assert len(accounts) >= 1

    @pytest.mark.asyncio
    async def test_get_account_balance(self, monkeypatch):
        _set_ibkr_env(monkeypatch)
        from shared.internal_money_monitor import InternalMoneyMonitor, LiveAccountData
        monitor = InternalMoneyMonitor()

        async def mock_fetch():
            return LiveAccountData(
                net_liquidation=25000.0,
                total_cash=5000.0,
                available_funds=12000.0,
                account_id="U24346218",
            )
        monitor._fetch_ibkr_data = mock_fetch

        balance = await monitor.get_account_balance("IBKR-MAIN")
        assert isinstance(balance, float)
        assert balance == 25000.0

    @pytest.mark.asyncio
    async def test_legacy_account_mapping(self, monkeypatch):
        _set_ibkr_env(monkeypatch)
        from shared.internal_money_monitor import InternalMoneyMonitor, LiveAccountData
        monitor = InternalMoneyMonitor()

        async def mock_fetch():
            return LiveAccountData(
                net_liquidation=25000.0,
                total_cash=5000.0,
                available_funds=12000.0,
                account_id="U24346218",
            )
        monitor._fetch_ibkr_data = mock_fetch

        balance = await monitor.get_account_balance("AAC-001")
        assert isinstance(balance, float)
        assert balance == 25000.0

    @pytest.mark.asyncio
    async def test_cache_ttl(self, monkeypatch):
        _set_ibkr_env(monkeypatch)
        from shared.internal_money_monitor import InternalMoneyMonitor, LiveAccountData
        monitor = InternalMoneyMonitor(cache_ttl=60)

        call_count = 0
        async def mock_fetch():
            nonlocal call_count
            call_count += 1
            return LiveAccountData(
                net_liquidation=25000.0,
                total_cash=5000.0,
                available_funds=12000.0,
                account_id="U24346218",
            )
        monitor._fetch_ibkr_data = mock_fetch

        await monitor.get_live_snapshot()
        await monitor.get_live_snapshot()  # should use cache
        assert call_count == 1  # only fetched once
