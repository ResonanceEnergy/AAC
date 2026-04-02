#!/usr/bin/env python3
"""
WebAuth Wallet Integration (Vector 10)
========================================
Client for WebAuth.com — the identity and wallet layer for XPR Network.

WebAuth provides:
  - Non-custodial wallet with biometric (Face ID / fingerprint) signing
  - WebAuthn/FIDO2 standards — no private keys exposed
  - Multi-chain support (XPR Network, Metal Blockchain)
  - On-chain identity verification
  - SSO for DApps via Proton signing requests

AAC uses WebAuth for:
  1. Secure transaction signing without exposing keys
  2. Identity verification for XPR Agents marketplace
  3. Multi-sig authorization for treasury operations
  4. Admin/operator authentication
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class WebAuthSession:
    """Active WebAuth login session."""
    session_id: str
    account_name: str
    permission: str  # 'active' or 'owner'
    chain: str  # 'proton' or 'metal'
    authenticated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SigningRequest:
    """A transaction signing request sent to WebAuth."""
    request_id: str
    actions: List[Dict[str, Any]]
    status: str  # pending, approved, rejected, expired
    created_at: datetime = field(default_factory=datetime.now)
    signed_at: Optional[datetime] = None
    tx_id: Optional[str] = None


class WebAuthClient:
    """
    WebAuth wallet integration for secure signing and identity.

    Usage:
        client = WebAuthClient(
            app_id="aac_trading",
            callback_url="https://aac.local/auth/callback",
        )
        await client.connect()

        # Generate login URI for operator
        login_uri = client.generate_login_uri(["login", "trade"])

        # After user approves in WebAuth app:
        session = await client.verify_session(session_token)

        # Request transaction signing
        request = await client.request_signing(
            account="aac.treasury",
            actions=[{
                "account": "eosio.token",
                "name": "transfer",
                "data": {...},
            }],
        )
    """

    API_URL = "https://api.webauth.com/v1"
    SIGNING_URL = "https://sign.webauth.com"

    def __init__(
        self,
        app_id: str = "aac_trading",
        callback_url: str = "",
        api_url: str = "",
    ):
        self.app_id = app_id
        self.callback_url = callback_url
        self._api_url = api_url or self.API_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._active_sessions: Dict[str, WebAuthSession] = {}

    async def connect(self) -> bool:
        """Initialize HTTP session."""
        try:
            connector = aiohttp.TCPConnector(
                resolver=aiohttp.resolver.ThreadedResolver()
            )
            self._session = aiohttp.ClientSession(connector=connector)
            logger.info("WebAuth client connected")
            return True
        except Exception as e:
            logger.error(f"WebAuth connection failed: {e}")
            return False

    async def disconnect(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._active_sessions.clear()
        logger.info("WebAuth client disconnected")

    # ==========================================
    # Authentication
    # ==========================================

    def generate_login_uri(
        self,
        scopes: Optional[List[str]] = None,
        chain: str = "proton",
    ) -> str:
        """
        Generate a WebAuth login URI.

        The operator opens this in their browser or WebAuth app.
        After approval, WebAuth calls back with a session token.
        """
        scopes = scopes or ["login"]
        scope_str = ",".join(scopes)

        return (
            f"{self.SIGNING_URL}/login"
            f"?app={self.app_id}"
            f"&chain={chain}"
            f"&scope={scope_str}"
            f"&callback={self.callback_url}"
        )

    async def verify_session(self, session_token: str) -> Optional[WebAuthSession]:
        """
        Verify a session token returned by WebAuth after login.

        Returns a WebAuthSession if valid, None if invalid/expired.
        """
        if not self._session:
            raise ConnectionError("WebAuth client not connected")

        try:
            async with self._session.post(
                f"{self._api_url}/sessions/verify",
                json={"token": session_token, "app_id": self.app_id},
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"WebAuth session verification failed: {resp.status}")
                    return None

                data = await resp.json()
                session = WebAuthSession(
                    session_id=data.get("session_id", ""),
                    account_name=data.get("account", ""),
                    permission=data.get("permission", "active"),
                    chain=data.get("chain", "proton"),
                    metadata=data.get("metadata", {}),
                )
                self._active_sessions[session.session_id] = session
                logger.info(f"WebAuth session verified: {session.account_name}")
                return session

        except Exception as e:
            logger.error(f"Session verification error: {e}")
            return None

    # ==========================================
    # Transaction Signing
    # ==========================================

    async def request_signing(
        self,
        account: str,
        actions: List[Dict[str, Any]],
        memo: str = "",
    ) -> SigningRequest:
        """
        Submit a signing request to WebAuth.

        The wallet owner sees the request on their device and
        can approve with biometric authentication.
        """
        if not self._session:
            raise ConnectionError("WebAuth client not connected")

        request_data = {
            "app_id": self.app_id,
            "account": account,
            "actions": actions,
            "memo": memo,
        }

        try:
            async with self._session.post(
                f"{self._api_url}/signing/request",
                json=request_data,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

                request = SigningRequest(
                    request_id=data.get("request_id", ""),
                    actions=actions,
                    status="pending",
                )

                logger.info(
                    f"Signing request submitted: {request.request_id} "
                    f"({len(actions)} actions for {account})"
                )
                return request

        except Exception as e:
            logger.error(f"Signing request failed: {e}")
            raise

    async def check_signing_status(self, request_id: str) -> SigningRequest:
        """Check the status of a pending signing request."""
        if not self._session:
            raise ConnectionError("WebAuth client not connected")

        try:
            async with self._session.get(
                f"{self._api_url}/signing/status/{request_id}",
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

                return SigningRequest(
                    request_id=request_id,
                    actions=data.get("actions", []),
                    status=data.get("status", "unknown"),
                    tx_id=data.get("transaction_id"),
                )

        except Exception as e:
            logger.error(f"Signing status check failed: {e}")
            raise

    # ==========================================
    # Identity / On-Chain Verification
    # ==========================================

    async def get_account_info(self, account_name: str) -> Dict[str, Any]:
        """Get on-chain account info including KYC status."""
        if not self._session:
            raise ConnectionError("WebAuth client not connected")

        try:
            async with self._session.get(
                f"{self._api_url}/accounts/{account_name}",
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

        except Exception as e:
            logger.error(f"Account info lookup failed: {e}")
            return {}

    async def verify_identity(self, account_name: str) -> bool:
        """Check if an account has passed on-chain KYC verification."""
        info = await self.get_account_info(account_name)
        return info.get("kyc_verified", False)
