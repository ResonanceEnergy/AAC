#!/usr/bin/env python3
"""
Secrets Manager
===============
Secure handling of API keys and sensitive configuration.
Uses Fernet symmetric encryption for secrets at rest.
"""

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

# Try to import cryptography
try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None
    InvalidToken = Exception

logger = logging.getLogger('SecretsManager')


class SecretsError(Exception):
    """Base exception for secrets-related errors"""
    pass


class EncryptionError(SecretsError):
    """Encryption/decryption failed"""
    pass


class SecretsManager:
    """
    Manages encrypted storage and retrieval of sensitive data.
    
    Usage:
        # Initialize with master password (from env or prompt)
        sm = SecretsManager(master_password=os.getenv('ACC_MASTER_PASSWORD'))
        
        # Store encrypted secrets
        sm.set_secret('binance_api_key', 'your-api-key')
        sm.save()
        
        # Retrieve secrets
        api_key = sm.get_secret('binance_api_key')
    """
    
    def __init__(
        self,
        master_password: Optional[str] = None,
        secrets_file: Optional[Path] = None,
        auto_load: bool = True,
    ):
        if not CRYPTO_AVAILABLE:
            logger.warning(
                "cryptography library not installed. "
                "Secrets will be stored in plaintext. "
                "Install with: pip install cryptography"
            )
        
        self.secrets_file = secrets_file or self._default_secrets_path()
        self._secrets: Dict[str, str] = {}
        self._fernet: Optional[Any] = None
        self._encrypted = False
        
        if master_password:
            self._init_encryption(master_password)
        
        if auto_load and self.secrets_file.exists():
            self.load()
    
    def _default_secrets_path(self) -> Path:
        """Get default secrets file path"""
        from shared.config_loader import get_project_path
        return get_project_path('data', 'secrets.enc')
    
    def _get_or_create_salt(self) -> bytes:
        """Get or create a unique salt for this installation"""
        from shared.config_loader import get_project_path
        salt_file = get_project_path('data', '.salt')
        
        if salt_file.exists():
            with open(salt_file, 'rb') as f:
                return f.read()
        
        # Generate new random salt
        import secrets as py_secrets
        salt = py_secrets.token_bytes(32)
        
        # Save it
        salt_file.parent.mkdir(parents=True, exist_ok=True)
        with open(salt_file, 'wb') as f:
            f.write(salt)
        
        logger.info("Generated new installation-specific salt")
        return salt
    
    def _init_encryption(self, password: str):
        """Initialize Fernet encryption with password-derived key"""
        if not CRYPTO_AVAILABLE:
            logger.warning("Encryption not available - secrets stored in plaintext")
            return
        
        # Use PBKDF2 to derive key from password with installation-specific salt
        salt = self._get_or_create_salt()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,  # OWASP recommended minimum
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self._fernet = Fernet(key)
        self._encrypted = True
        logger.info("Encryption initialized with installation-specific salt")
    
    def set_secret(self, key: str, value: str):
        """Store a secret (in memory until save() is called)"""
        self._secrets[key] = value
        logger.debug(f"Secret set: {key}")
    
    def get_secret(self, key: str, default: str = '') -> str:
        """Retrieve a secret by key"""
        return self._secrets.get(key, default)
    
    def has_secret(self, key: str) -> bool:
        """Check if a secret exists"""
        return key in self._secrets
    
    def delete_secret(self, key: str) -> bool:
        """Delete a secret"""
        if key in self._secrets:
            del self._secrets[key]
            return True
        return False
    
    def list_secrets(self) -> list:
        """List all secret keys (not values)"""
        return list(self._secrets.keys())
    
    def save(self):
        """Save secrets to encrypted file"""
        self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = json.dumps(self._secrets)
        
        if self._encrypted and self._fernet:
            # Encrypt the data
            encrypted = self._fernet.encrypt(data.encode())
            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted)
            logger.info(f"Saved {len(self._secrets)} encrypted secrets")
        else:
            # Store as base64-encoded JSON (not secure, but obfuscated)
            encoded = base64.b64encode(data.encode())
            with open(self.secrets_file, 'wb') as f:
                f.write(b'PLAIN:' + encoded)
            logger.warning(f"Saved {len(self._secrets)} secrets WITHOUT encryption")
    
    def load(self) -> bool:
        """Load secrets from file"""
        if not self.secrets_file.exists():
            logger.info("No secrets file found")
            return False
        
        try:
            with open(self.secrets_file, 'rb') as f:
                content = f.read()
            
            if content.startswith(b'PLAIN:'):
                # Plaintext (base64 encoded)
                data = base64.b64decode(content[6:]).decode()
                self._secrets = json.loads(data)
                logger.warning("Loaded unencrypted secrets")
            else:
                # Encrypted
                if not self._fernet:
                    raise EncryptionError(
                        "Cannot decrypt secrets - no master password provided"
                    )
                try:
                    decrypted = self._fernet.decrypt(content)
                    self._secrets = json.loads(decrypted.decode())
                    logger.info(f"Loaded {len(self._secrets)} encrypted secrets")
                except InvalidToken:
                    raise EncryptionError("Invalid master password or corrupted secrets file")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            return False
    
    def migrate_from_env(self, env_keys: Dict[str, str]):
        """
        Migrate secrets from environment variables to encrypted storage.
        
        Args:
            env_keys: Dict mapping secret_name -> env_var_name
                      e.g., {'binance_api_key': 'BINANCE_API_KEY'}
        """
        migrated = 0
        for secret_name, env_var in env_keys.items():
            value = os.environ.get(env_var, '')
            if value:
                self.set_secret(secret_name, value)
                migrated += 1
                logger.info(f"Migrated {env_var} -> {secret_name}")
        
        if migrated:
            self.save()
            logger.info(f"Migrated {migrated} secrets from environment")
        
        return migrated


# Singleton instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager(master_password: Optional[str] = None) -> SecretsManager:
    """Get or create the global secrets manager instance"""
    global _secrets_manager
    
    if _secrets_manager is None:
        password = master_password or os.environ.get('ACC_MASTER_PASSWORD', '')
        _secrets_manager = SecretsManager(master_password=password)
    
    return _secrets_manager


def get_secret(key: str, default: str = '') -> str:
    """Convenience function to get a secret"""
    return get_secrets_manager().get_secret(key, default)


# ============================================
# INPUT VALIDATION
# ============================================

@dataclass
class ValidationResult:
    """Result of input validation"""
    valid: bool
    error: str = ''
    sanitized_value: Any = None


def validate_symbol(symbol: str) -> ValidationResult:
    """
    Validate a trading symbol format.
    
    Valid formats:
    - BTC/USDT, ETH/BTC (spot)
    - BTCUSDT (compact)
    - BTC-PERP (futures)
    """
    if not symbol:
        return ValidationResult(False, "Symbol cannot be empty")
    
    # Remove whitespace
    symbol = symbol.strip().upper()
    
    # Check length
    if len(symbol) < 3 or len(symbol) > 20:
        return ValidationResult(False, f"Invalid symbol length: {len(symbol)}")
    
    # Check for invalid characters
    allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-_')
    if not all(c in allowed_chars for c in symbol):
        return ValidationResult(False, "Symbol contains invalid characters")
    
    return ValidationResult(True, sanitized_value=symbol)


def validate_quantity(quantity: float, min_qty: float = 0.0, max_qty: float = 1e9) -> ValidationResult:
    """Validate order quantity"""
    if not isinstance(quantity, (int, float)):
        return ValidationResult(False, "Quantity must be a number")
    
    if quantity <= min_qty:
        return ValidationResult(False, f"Quantity must be greater than {min_qty}")
    
    if quantity > max_qty:
        return ValidationResult(False, f"Quantity exceeds maximum {max_qty}")
    
    # Check for NaN/Inf
    if quantity != quantity or quantity == float('inf'):
        return ValidationResult(False, "Quantity cannot be NaN or Infinity")
    
    return ValidationResult(True, sanitized_value=float(quantity))


def validate_price(price: Optional[float], allow_none: bool = True) -> ValidationResult:
    """Validate order price"""
    if price is None:
        if allow_none:
            return ValidationResult(True, sanitized_value=None)
        return ValidationResult(False, "Price is required")
    
    if not isinstance(price, (int, float)):
        return ValidationResult(False, "Price must be a number")
    
    if price <= 0:
        return ValidationResult(False, "Price must be positive")
    
    if price != price or price == float('inf'):
        return ValidationResult(False, "Price cannot be NaN or Infinity")
    
    return ValidationResult(True, sanitized_value=float(price))


def validate_order_side(side: str) -> ValidationResult:
    """Validate order side"""
    side = side.strip().lower()
    if side not in ('buy', 'sell'):
        return ValidationResult(False, f"Invalid order side: {side}")
    return ValidationResult(True, sanitized_value=side)


def validate_order_type(order_type: str) -> ValidationResult:
    """Validate order type"""
    order_type = order_type.strip().lower()
    valid_types = ('market', 'limit', 'stop_loss', 'stop_limit', 'take_profit')
    if order_type not in valid_types:
        return ValidationResult(False, f"Invalid order type: {order_type}")
    return ValidationResult(True, sanitized_value=order_type)


def validate_exchange(exchange: str) -> ValidationResult:
    """Validate exchange name"""
    exchange = exchange.strip().lower()
    valid_exchanges = ('binance', 'coinbase', 'kraken', 'paper')
    if exchange not in valid_exchanges:
        return ValidationResult(False, f"Unknown exchange: {exchange}")
    return ValidationResult(True, sanitized_value=exchange)


class OrderValidator:
    """
    Validates complete order parameters before submission.
    """
    
    def __init__(
        self,
        min_quantity: float = 0.0,
        max_quantity: float = 1e9,
        max_price: float = 1e9,
    ):
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.max_price = max_price
    
    def validate_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        exchange: str = 'binance',
    ) -> ValidationResult:
        """Validate all order parameters"""
        errors = []
        
        # Validate each field
        symbol_result = validate_symbol(symbol)
        if not symbol_result.valid:
            errors.append(f"Symbol: {symbol_result.error}")
        
        side_result = validate_order_side(side)
        if not side_result.valid:
            errors.append(f"Side: {side_result.error}")
        
        type_result = validate_order_type(order_type)
        if not type_result.valid:
            errors.append(f"Type: {type_result.error}")
        
        qty_result = validate_quantity(quantity, self.min_quantity, self.max_quantity)
        if not qty_result.valid:
            errors.append(f"Quantity: {qty_result.error}")
        
        # Price required for limit orders
        allow_none = type_result.sanitized_value == 'market' if type_result.valid else True
        price_result = validate_price(price, allow_none=allow_none)
        if not price_result.valid:
            errors.append(f"Price: {price_result.error}")
        
        exchange_result = validate_exchange(exchange)
        if not exchange_result.valid:
            errors.append(f"Exchange: {exchange_result.error}")
        
        if errors:
            return ValidationResult(False, "; ".join(errors))
        
        return ValidationResult(
            True,
            sanitized_value={
                'symbol': symbol_result.sanitized_value,
                'side': side_result.sanitized_value,
                'order_type': type_result.sanitized_value,
                'quantity': qty_result.sanitized_value,
                'price': price_result.sanitized_value,
                'exchange': exchange_result.sanitized_value,
            }
        )


# Global validator instance
_order_validator = OrderValidator()


def validate_order(**kwargs) -> ValidationResult:
    """Convenience function to validate an order"""
    return _order_validator.validate_order(**kwargs)
