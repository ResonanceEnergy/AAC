#!/usr/bin/env python3
"""
Whale Alert API Client
=======================
Real-time on-chain whale transaction tracking. Monitors large crypto
transfers between wallets, exchanges, and unknown addresses.

API: https://docs.whale-alert.io/

Requires:
    - WHALE_ALERT_API_KEY in .env (free tier: 10 req/min)
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.api_integration_hub import APIClient, APIEndpoint, APIResponse


@dataclass
class WhaleTransaction:
    """Large on-chain transaction"""
    blockchain: str
    symbol: str
    tx_hash: str
    from_address: str
    from_owner: str  # e.g. 'binance', 'unknown'
    from_type: str   # 'exchange', 'unknown'
    to_address: str
    to_owner: str
    to_type: str
    amount: float
    amount_usd: float
    timestamp: datetime


class WhaleAlertClient(APIClient):
    """
    Whale Alert API client.

    Tracks large crypto transfers in real-time. Useful for detecting
    exchange inflows/outflows that signal selling/buying pressure.
    """

    def __init__(self):
        self._api_key = os.environ.get('WHALE_ALERT_API_KEY', '')

        endpoint = APIEndpoint(
            name='whale_alert',
            base_url='https://api.whale-alert.io/v1',
            auth_type='none',  # Uses query param
            rate_limit=10,
            timeout=30,
        )
        super().__init__(endpoint)
        self.logger = logging.getLogger('WhaleAlertClient')

    async def get_transactions(
        self,
        min_value: int = 500000,
        start: Optional[int] = None,
        end: Optional[int] = None,
        currency: str = '',
        limit: int = 100,
    ) -> List[WhaleTransaction]:
        """
        Get recent whale transactions.

        Args:
            min_value: Minimum USD value to filter
            start: Unix timestamp for start (default: 1 hour ago)
            end: Unix timestamp for end (default: now)
            currency: Filter by currency (e.g. 'btc', 'eth')
            limit: Max results
        """
        if start is None:
            start = int((datetime.now() - timedelta(hours=1)).timestamp())

        params = {
            'api_key': self._api_key,
            'min_value': min_value,
            'start': start,
            'limit': limit,
        }
        if end:
            params['end'] = end
        if currency:
            params['currency'] = currency.lower()

        url = f"{self.endpoint.base_url}/transactions"
        response = await self._make_request('GET', url, params=params)

        if not response.success or not response.data:
            return []

        transactions = []
        for tx in response.data.get('transactions', []):
            from_data = tx.get('from', {})
            to_data = tx.get('to', {})

            transactions.append(WhaleTransaction(
                blockchain=tx.get('blockchain', ''),
                symbol=tx.get('symbol', ''),
                tx_hash=tx.get('hash', ''),
                from_address=from_data.get('address', ''),
                from_owner=from_data.get('owner', 'unknown'),
                from_type=from_data.get('owner_type', 'unknown'),
                to_address=to_data.get('address', ''),
                to_owner=to_data.get('owner', 'unknown'),
                to_type=to_data.get('owner_type', 'unknown'),
                amount=float(tx.get('amount', 0)),
                amount_usd=float(tx.get('amount_usd', 0)),
                timestamp=datetime.fromtimestamp(tx.get('timestamp', 0)),
            ))

        return transactions

    async def get_exchange_flows(
        self,
        currency: str = 'btc',
        min_value: int = 1000000,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Analyze exchange inflow/outflow for a currency.

        High inflows → potential selling pressure.
        High outflows → accumulation (bullish signal).
        """
        start = int((datetime.now() - timedelta(hours=hours)).timestamp())
        txs = await self.get_transactions(
            min_value=min_value,
            start=start,
            currency=currency,
        )

        inflows = []   # To exchange
        outflows = []  # From exchange

        for tx in txs:
            if tx.to_type == 'exchange':
                inflows.append(tx)
            elif tx.from_type == 'exchange':
                outflows.append(tx)

        total_inflow = sum(t.amount_usd for t in inflows)
        total_outflow = sum(t.amount_usd for t in outflows)
        net_flow = total_outflow - total_inflow

        return {
            'currency': currency,
            'period_hours': hours,
            'inflow_count': len(inflows),
            'outflow_count': len(outflows),
            'total_inflow_usd': total_inflow,
            'total_outflow_usd': total_outflow,
            'net_flow_usd': net_flow,
            'signal': 'BULLISH' if net_flow > 0 else 'BEARISH' if net_flow < 0 else 'NEUTRAL',
            'top_inflows': sorted(inflows, key=lambda t: -t.amount_usd)[:5],
            'top_outflows': sorted(outflows, key=lambda t: -t.amount_usd)[:5],
        }

    async def get_status(self) -> Optional[Dict]:
        """Get API status and supported blockchains"""
        url = f"{self.endpoint.base_url}/status"
        response = await self._make_request('GET', url, params={'api_key': self._api_key})
        return response.data if response.success else None
