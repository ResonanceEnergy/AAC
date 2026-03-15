#!/usr/bin/env python3
"""
Metal Pay B2B API Client (Vector 7)
=====================================
Client for the Metal Pay B2B (Business-to-Business) payment platform.

Metal Pay enables:
  - Instant fiat-to-crypto conversions
  - Business payroll in crypto
  - B2B settlement rails
  - Compliance-ready KYC/AML integration
  - NMLS licensed (#2057807)
  - Pop-in checkout for e-commerce

This client is used by AAC for:
  1. Fiat on/off ramp for the treasury
  2. Automated portfolio rebalancing via fiat legs
  3. Business expense settlement
  4. Payroll automation for agent-generated revenue
"""

import asyncio
import hashlib
import hmac
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class PaymentTransaction:
    """A Metal Pay payment transaction."""
    tx_id: str
    from_account: str
    to_account: str
    amount: float
    currency: str
    status: str  # pending, completed, failed, cancelled
    fee: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetalPayClient:
    """
    Metal Pay B2B API client for fiat on/off ramp and payments.

    Usage:
        client = MetalPayClient(
            api_key="mp_...",
            api_secret="mps_...",
        )
        await client.connect()

        # Get available currencies
        currencies = await client.get_currencies()

        # Get a conversion quote
        quote = await client.get_quote("USD", "BTC", 1000.0)

        # Execute conversion
        tx = await client.execute_conversion("USD", "BTC", 1000.0)
    """

    BASE_URL = "https://api.metalpay.com/v1"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "",
    ):
        self.api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url or self.BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_delay = 0.2  # 5 req/s

    def _sign_request(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate HMAC-SHA256 signature for authenticated requests."""
        message = f"{timestamp}{method.upper()}{path}{body}"
        return hmac.new(
            self._api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Build authenticated request headers."""
        timestamp = str(int(time.time()))
        signature = self._sign_request(timestamp, method, path, body)
        return {
            "X-API-Key": self.api_key,
            "X-Timestamp": timestamp,
            "X-Signature": signature,
            "Content-Type": "application/json",
        }

    async def connect(self) -> bool:
        """Initialize HTTP session."""
        try:
            connector = aiohttp.TCPConnector(
                resolver=aiohttp.resolver.ThreadedResolver()
            )
            self._session = aiohttp.ClientSession(connector=connector)
            logger.info("Metal Pay client connected")
            return True
        except Exception as e:
            logger.error(f"Metal Pay connection failed: {e}")
            return False

    async def disconnect(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("Metal Pay client disconnected")

    async def _request(
        self, method: str, path: str, data: Optional[Dict] = None
    ) -> Dict:
        """Make an authenticated API request."""
        if not self._session:
            raise ConnectionError("Metal Pay client not connected")

        url = f"{self._base_url}{path}"
        body = ""
        if data:
            import json
            body = json.dumps(data)

        headers = self._auth_headers(method, path, body)

        await asyncio.sleep(self._rate_limit_delay)

        try:
            if method.upper() == "GET":
                async with self._session.get(url, headers=headers) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            elif method.upper() == "POST":
                async with self._session.post(url, headers=headers, data=body) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        except aiohttp.ClientResponseError as e:
            logger.error(f"Metal Pay API error: {e.status} {e.message}")
            raise
        except Exception as e:
            logger.error(f"Metal Pay request failed: {e}")
            raise

    # ==========================================
    # Currency & Account
    # ==========================================

    async def get_currencies(self) -> List[Dict]:
        """Get supported currencies and their status."""
        return await self._request("GET", "/currencies")

    async def get_account_info(self) -> Dict:
        """Get business account details."""
        return await self._request("GET", "/account")

    async def get_balances(self) -> Dict[str, float]:
        """Get account balances by currency."""
        data = await self._request("GET", "/balances")
        return {
            bal["currency"]: float(bal["available"])
            for bal in data.get("balances", [])
        }

    # ==========================================
    # Conversions (Fiat ↔ Crypto)
    # ==========================================

    async def get_quote(
        self, from_currency: str, to_currency: str, amount: float
    ) -> Dict:
        """
        Get a conversion quote.

        Returns: {rate, from_amount, to_amount, fee, expires_at}
        """
        return await self._request("POST", "/quotes", {
            "from_currency": from_currency,
            "to_currency": to_currency,
            "amount": amount,
        })

    async def execute_conversion(
        self,
        from_currency: str,
        to_currency: str,
        amount: float,
        quote_id: Optional[str] = None,
    ) -> PaymentTransaction:
        """
        Execute a currency conversion.

        Optionally lock in a quote_id for guaranteed rate.
        """
        data: Dict[str, Any] = {
            "from_currency": from_currency,
            "to_currency": to_currency,
            "amount": amount,
        }
        if quote_id:
            data["quote_id"] = quote_id

        result = await self._request("POST", "/conversions", data)

        return PaymentTransaction(
            tx_id=result.get("transaction_id", ""),
            from_account="self",
            to_account="self",
            amount=amount,
            currency=from_currency,
            status=result.get("status", "pending"),
            fee=float(result.get("fee", 0)),
            metadata={
                "to_currency": to_currency,
                "rate": result.get("rate"),
                "to_amount": result.get("to_amount"),
            },
        )

    # ==========================================
    # Payments
    # ==========================================

    async def send_payment(
        self,
        recipient: str,
        amount: float,
        currency: str,
        memo: str = "",
    ) -> PaymentTransaction:
        """Send a payment to another Metal Pay user or external address."""
        result = await self._request("POST", "/payments", {
            "recipient": recipient,
            "amount": amount,
            "currency": currency,
            "memo": memo,
        })

        return PaymentTransaction(
            tx_id=result.get("transaction_id", ""),
            from_account="self",
            to_account=recipient,
            amount=amount,
            currency=currency,
            status=result.get("status", "pending"),
            fee=float(result.get("fee", 0)),
        )

    async def get_transaction_history(
        self,
        currency: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Get transaction history with optional currency filter."""
        path = f"/transactions?limit={limit}"
        if currency:
            path += f"&currency={currency}"
        return await self._request("GET", path)

    async def get_transaction(self, tx_id: str) -> Dict:
        """Get details of a specific transaction."""
        return await self._request("GET", f"/transactions/{tx_id}")
