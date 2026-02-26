#!/usr/bin/env python3
"""
API Key Management System
=========================
Secure API key acquisition, storage, rotation, and validation for 100+ data feeds.
"""

import asyncio
import logging
import json
import os
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
import secrets
import keyring
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import get_audit_logger


class KeyStatus(Enum):
    """API key status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"
    RATE_LIMITED = "rate_limited"
    INVALID = "invalid"


class KeyProvider(Enum):
    """API key providers"""
    ENVIRONMENT = "environment"
    KEYRING = "keyring"
    VAULT = "vault"
    CONFIG_FILE = "config_file"


@dataclass
class APIKey:
    """API key with metadata"""
    key_id: str
    provider: str
    service: str
    key_name: str  # e.g., "binance_api_key"
    encrypted_key: str
    status: KeyStatus = KeyStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyValidationResult:
    """Result of key validation"""
    is_valid: bool
    status: KeyStatus
    message: str
    rate_limit_info: Optional[Dict] = None
    expires_in_days: Optional[int] = None


class APIKeyManager:
    """Secure API key management system"""

    def __init__(self):
        self.logger = logging.getLogger("APIKeyManager")
        self.audit_logger = get_audit_logger()
        self.keys: Dict[str, APIKey] = {}
        self.encryption_key = self._get_or_create_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        self.key_store_path = PROJECT_ROOT / "config" / "api_keys.enc"

        # Load existing keys
        asyncio.create_task(self._load_keys())

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for API keys"""
        key_file = PROJECT_ROOT / "config" / ".key_encryption_key"

        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new encryption key
            salt = os.urandom(16)
            password = secrets.token_bytes(32)

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))

            # Store the key (in production, this should be in a secure vault)
            with open(key_file, 'wb') as f:
                f.write(key)

            return key

    async def _load_keys(self):
        """Load encrypted keys from storage"""
        if self.key_store_path.exists():
            try:
                with open(self.key_store_path, 'r') as f:
                    encrypted_data = f.read()

                decrypted_data = self.fernet.decrypt(encrypted_data.encode())
                keys_data = json.loads(decrypted_data.decode())

                for key_data in keys_data:
                    key = APIKey(**key_data)
                    self.keys[key.key_id] = key

                self.logger.info(f"Loaded {len(self.keys)} API keys")

            except Exception as e:
                self.logger.error(f"Failed to load API keys: {e}")

    async def _save_keys(self):
        """Save encrypted keys to storage"""
        try:
            keys_data = [key.__dict__ for key in self.keys.values()]
            json_data = json.dumps(keys_data, default=str)
            encrypted_data = self.fernet.encrypt(json_data.encode())

            with open(self.key_store_path, 'w') as f:
                f.write(encrypted_data.decode())

        except Exception as e:
            self.logger.error(f"Failed to save API keys: {e}")

    async def add_key(
        self,
        provider: str,
        service: str,
        key_name: str,
        key_value: str,
        expires_at: Optional[datetime] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add a new API key"""
        key_id = f"{provider}_{service}_{key_name}_{secrets.token_hex(4)}"

        # Encrypt the key
        encrypted_key = self.fernet.encrypt(key_value.encode()).decode()

        key = APIKey(
            key_id=key_id,
            provider=provider,
            service=service,
            key_name=key_name,
            encrypted_key=encrypted_key,
            expires_at=expires_at,
            metadata=metadata or {}
        )

        self.keys[key_id] = key
        await self._save_keys()

        await self.audit_logger.log_api_call(
            exchange=service,
            method="ADD_KEY",
            endpoint=f"key:{key_name}",
            status="success",
            details={"key_id": key_id, "provider": provider}
        )

        self.logger.info(f"Added API key: {key_id} for {service}")
        return key_id

    async def get_key(self, service: str, key_name: str) -> Optional[str]:
        """Get decrypted API key"""
        for key in self.keys.values():
            if key.service == service and key.key_name == key_name and key.status == KeyStatus.ACTIVE:
                try:
                    # Check expiration
                    if key.expires_at and datetime.now() > key.expires_at:
                        key.status = KeyStatus.EXPIRED
                        await self._save_keys()
                        return None

                    # Decrypt and return
                    decrypted = self.fernet.decrypt(key.encrypted_key.encode()).decode()
                    key.last_used = datetime.now()
                    key.usage_count += 1
                    await self._save_keys()

                    return decrypted

                except Exception as e:
                    self.logger.error(f"Failed to decrypt key {key.key_id}: {e}")
                    return None

        return None

    async def validate_key(self, service: str, key_name: str) -> KeyValidationResult:
        """Validate an API key by testing it"""
        key_value = await self.get_key(service, key_name)
        if not key_value:
            return KeyValidationResult(
                is_valid=False,
                status=KeyStatus.INVALID,
                message="Key not found or inactive"
            )

        # Service-specific validation
        try:
            if service == "binance":
                return await self._validate_binance_key(key_value)
            elif service == "coinbase_pro":
                return await self._validate_coinbase_key(key_value)
            elif service == "alpha_vantage":
                return await self._validate_alpha_vantage_key(key_value)
            else:
                # Generic validation - just check if key exists
                return KeyValidationResult(
                    is_valid=True,
                    status=KeyStatus.ACTIVE,
                    message="Key exists (no specific validation available)"
                )

        except Exception as e:
            return KeyValidationResult(
                is_valid=False,
                status=KeyStatus.INVALID,
                message=f"Validation failed: {str(e)}"
            )

    async def _validate_binance_key(self, api_key: str) -> KeyValidationResult:
        """Validate Binance API key"""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"X-MBX-APIKEY": api_key}
                async with session.get("https://api.binance.com/api/v3/account", headers=headers) as resp:
                    if resp.status == 200:
                        return KeyValidationResult(
                            is_valid=True,
                            status=KeyStatus.ACTIVE,
                            message="Binance API key valid"
                        )
                    elif resp.status == 401:
                        return KeyValidationResult(
                            is_valid=False,
                            status=KeyStatus.INVALID,
                            message="Invalid Binance API key"
                        )
                    else:
                        return KeyValidationResult(
                            is_valid=False,
                            status=KeyStatus.RATE_LIMITED,
                            message=f"Binance API error: {resp.status}"
                        )
        except Exception as e:
            return KeyValidationResult(
                is_valid=False,
                status=KeyStatus.INVALID,
                message=f"Binance validation error: {str(e)}"
            )

    async def _validate_coinbase_key(self, api_key: str) -> KeyValidationResult:
        """Validate Coinbase Pro API key"""
        import aiohttp
        import hmac
        import hashlib
        import time

        try:
            timestamp = str(int(time.time()))
            message = timestamp + "GET" + "/accounts"
            signature = hmac.new(api_key.encode(), message.encode(), hashlib.sha256).hexdigest()

            headers = {
                "CB-ACCESS-KEY": api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-ACCESS-PASSPHRASE": "",  # Would need passphrase from config
            }

            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.pro.coinbase.com/accounts", headers=headers) as resp:
                    if resp.status == 200:
                        return KeyValidationResult(
                            is_valid=True,
                            status=KeyStatus.ACTIVE,
                            message="Coinbase Pro API key valid"
                        )
                    else:
                        return KeyValidationResult(
                            is_valid=False,
                            status=KeyStatus.INVALID,
                            message=f"Coinbase API error: {resp.status}"
                        )
        except Exception as e:
            return KeyValidationResult(
                is_valid=False,
                status=KeyStatus.INVALID,
                message=f"Coinbase validation error: {str(e)}"
            )

    async def _validate_alpha_vantage_key(self, api_key: str) -> KeyValidationResult:
        """Validate Alpha Vantage API key"""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                params = {"function": "TIME_SERIES_INTRADAY", "symbol": "IBM", "interval": "1min", "apikey": api_key}
                async with session.get("https://www.alphavantage.co/query", params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "Error Message" in data:
                            return KeyValidationResult(
                                is_valid=False,
                                status=KeyStatus.INVALID,
                                message="Invalid Alpha Vantage API key"
                            )
                        else:
                            return KeyValidationResult(
                                is_valid=True,
                                status=KeyStatus.ACTIVE,
                                message="Alpha Vantage API key valid"
                            )
                    else:
                        return KeyValidationResult(
                            is_valid=False,
                            status=KeyStatus.INVALID,
                            message=f"Alpha Vantage API error: {resp.status}"
                        )
        except Exception as e:
            return KeyValidationResult(
                is_valid=False,
                status=KeyStatus.INVALID,
                message=f"Alpha Vantage validation error: {str(e)}"
            )

    async def rotate_key(self, key_id: str, new_key_value: str) -> bool:
        """Rotate an API key"""
        if key_id not in self.keys:
            return False

        key = self.keys[key_id]
        key.encrypted_key = self.fernet.encrypt(new_key_value.encode()).decode()
        key.created_at = datetime.now()
        key.usage_count = 0

        await self._save_keys()

        await self.audit_logger.log_api_call(
            exchange=key.service,
            method="ROTATE_KEY",
            endpoint=f"key:{key.key_name}",
            status="success",
            details={"key_id": key_id}
        )

        self.logger.info(f"Rotated API key: {key_id}")
        return True

    async def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        if key_id not in self.keys:
            return False

        key = self.keys[key_id]
        key.status = KeyStatus.REVOKED

        await self._save_keys()

        await self.audit_logger.log_api_call(
            exchange=key.service,
            method="REVOKE_KEY",
            endpoint=f"key:{key.key_name}",
            status="success",
            details={"key_id": key_id}
        )

        self.logger.info(f"Revoked API key: {key_id}")
        return True

    def get_key_status(self, service: str, key_name: str) -> Optional[KeyStatus]:
        """Get status of an API key"""
        for key in self.keys.values():
            if key.service == service and key.key_name == key_name:
                return key.status
        return None

    def list_keys(self, service: Optional[str] = None) -> List[Dict]:
        """List API keys"""
        keys_list = []
        for key in self.keys.values():
            if service is None or key.service == service:
                keys_list.append({
                    "key_id": key.key_id,
                    "service": key.service,
                    "key_name": key.key_name,
                    "status": key.status.value,
                    "created_at": key.created_at.isoformat(),
                    "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                    "usage_count": key.usage_count,
                    "last_used": key.last_used.isoformat() if key.last_used else None,
                })
        return keys_list

    async def cleanup_expired_keys(self):
        """Clean up expired keys"""
        expired_keys = []
        for key in self.keys.values():
            if key.expires_at and datetime.now() > key.expires_at:
                key.status = KeyStatus.EXPIRED
                expired_keys.append(key.key_id)

        if expired_keys:
            await self._save_keys()
            self.logger.info(f"Marked {len(expired_keys)} keys as expired")

    async def import_from_environment(self):
        """Import API keys from environment variables"""
        # Common API key environment variables
        env_mappings = {
            "binance": ["BINANCE_API_KEY", "BINANCE_API_SECRET"],
            "coinbase_pro": ["COINBASE_API_KEY", "COINBASE_API_SECRET", "COINBASE_PASSPHRASE"],
            "alpha_vantage": ["ALPHA_VANTAGE_API_KEY"],
            "polygon": ["POLYGON_API_KEY"],
            "finnhub": ["FINNHUB_API_KEY"],
            "iex_cloud": ["IEX_CLOUD_API_KEY"],
            "tiingo": ["TIINGO_API_KEY"],
            "quandl": ["QUANDL_API_KEY"],
            "oanda": ["OANDA_API_KEY", "OANDA_ACCOUNT_ID"],
            "fxcm": ["FXCM_API_KEY"],
            "kraken": ["KRAKEN_API_KEY", "KRAKEN_API_SECRET"],
            "bybit": ["BYBIT_API_KEY", "BYBIT_API_SECRET"],
        }

        imported_count = 0
        for service, env_vars in env_mappings.items():
            for env_var in env_vars:
                key_value = os.getenv(env_var)
                if key_value:
                    await self.add_key(
                        provider="environment",
                        service=service,
                        key_name=env_var.lower(),
                        key_value=key_value
                    )
                    imported_count += 1

        self.logger.info(f"Imported {imported_count} API keys from environment variables")
        return imported_count


# Global API key manager instance
api_key_manager = APIKeyManager()


async def initialize_api_key_system():
    """Initialize the API key management system"""
    await api_key_manager._load_keys()
    await api_key_manager.cleanup_expired_keys()

    # Import keys from environment if available
    imported = await api_key_manager.import_from_environment()
    if imported > 0:
        print(f"[OK] Imported {imported} API keys from environment")

    print("[OK] API key management system initialized")


if __name__ == "__main__":
    # Example usage
    asyncio.run(initialize_api_key_system())