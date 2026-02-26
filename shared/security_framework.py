#!/usr/bin/env python3
"""
Advanced Security Framework
==========================
Multi-factor authentication, encryption, RBAC, and API security for production deployment.
"""

import asyncio
import logging
import json
import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import pyotp
import qrcode
import io
import base64
import uuid
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import get_audit_logger


@dataclass
class User:
    """User account with security profile"""
    user_id: str
    username: str
    email: str
    role: str
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    password_hash: Optional[str] = None
    api_keys: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Role:
    """Security role with permissions"""
    role_id: str
    name: str
    description: str
    permissions: List[str]
    inherits_from: Optional[str] = None


@dataclass
class APIKey:
    """API key with security controls"""
    key_id: str
    user_id: str
    name: str
    key_hash: str
    permissions: List[str]
    rate_limit: int  # requests per minute
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    request_count: int = 0


@dataclass
class SecurityEvent:
    """Security event for monitoring"""
    event_id: str
    event_type: str
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    details: Dict[str, Any]
    timestamp: datetime
    severity: str


class MultiFactorAuthentication:
    """MFA implementation using TOTP"""

    def __init__(self):
        self.logger = logging.getLogger("MFA")

    def generate_secret(self) -> str:
        """Generate a new TOTP secret"""
        return pyotp.random_base32()

    def get_totp_uri(self, username: str, secret: str, issuer: str = "AAC Trading") -> str:
        """Generate TOTP URI for QR code"""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=username, issuer_name=issuer)

    def generate_qr_code(self, uri: str) -> str:
        """Generate QR code as base64 string"""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def verify_totp(self, secret: str, code: str) -> bool:
        """Verify TOTP code"""
        totp = pyotp.TOTP(secret)
        return totp.verify(code)

    def get_backup_codes(self, secret: str, count: int = 10) -> List[str]:
        """Generate backup codes"""
        codes = []
        for i in range(count):
            # Create backup codes as HMAC of secret with counter
            code = hmac.new(
                secret.encode(),
                f"backup_{i}".encode(),
                hashlib.sha256
            ).hexdigest()[:8].upper()
            codes.append(code)
        return codes


class AdvancedEncryption:
    """Advanced encryption for data protection"""

    def __init__(self):
        self.logger = logging.getLogger("Encryption")
        self.audit_logger = get_audit_logger()

        # Encryption keys
        self.master_key = None
        self.key_rotation_schedule = {}

        # Key storage
        self.keys_dir = PROJECT_ROOT / "config" / "crypto"
        self.keys_dir.mkdir(parents=True, exist_ok=True)

        self._load_master_key()

    def _load_master_key(self):
        """Load or generate master encryption key"""
        key_file = self.keys_dir / "master_key.enc"

        try:
            if key_file.exists():
                # Load existing key
                with open(key_file, 'rb') as f:
                    encrypted_key = f.read()
                # In production, this would be decrypted with a password or HSM
                self.master_key = base64.b64decode(encrypted_key)
            else:
                # Generate new key
                self.master_key = Fernet.generate_key()

                # Save encrypted key (simplified - in production use proper key encryption)
                with open(key_file, 'wb') as f:
                    f.write(base64.b64encode(self.master_key))

                self.logger.info("Generated new master encryption key")

        except Exception as e:
            self.logger.error(f"Error loading master key: {e}")
            # Fallback to generated key
            self.master_key = Fernet.generate_key()

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not self.master_key:
            raise ValueError("Master key not available")

        f = Fernet(self.master_key)
        encrypted = f.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self.master_key:
            raise ValueError("Master key not available")

        f = Fernet(self.master_key)
        encrypted = base64.b64decode(encrypted_data)
        decrypted = f.decrypt(encrypted)
        return decrypted.decode()

    def rotate_key(self) -> bool:
        """Rotate encryption key"""
        try:
            old_key = self.master_key
            new_key = Fernet.generate_key()

            # In production, would re-encrypt all data with new key
            # For now, just update the key
            self.master_key = new_key

            # Save new key
            key_file = self.keys_dir / "master_key.enc"
            with open(key_file, 'wb') as f:
                f.write(base64.b64encode(new_key))

            # Audit the key rotation
            self.audit_logger.log_event(
                category="security",
                action="key_rotated",
                details={"key_type": "master_encryption"},
                severity="info"
            )

            self.logger.info("Encryption key rotated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Key rotation failed: {e}")
            return False

    def hash_password(self, password: str) -> str:
        """Hash password using Argon2 or PBKDF2"""
        # Use PBKDF2 for simplicity (in production, consider Argon2)
        salt = secrets.token_bytes(16)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt,
            100000  # 100k iterations
        )
        return base64.b64encode(salt + key).decode()

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            decoded = base64.b64decode(hashed)
            salt = decoded[:16]
            stored_key = decoded[16:]

            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt,
                100000
            )

            return secrets.compare_digest(key, stored_key)
        except Exception:
            return False


class RoleBasedAccessControl:
    """RBAC system with hierarchical permissions"""

    def __init__(self):
        self.logger = logging.getLogger("RBAC")
        self.audit_logger = get_audit_logger()

        # Roles and permissions
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}

        # Permission hierarchy
        self.permission_hierarchy = {
            "read": ["read"],
            "write": ["read", "write"],
            "admin": ["read", "write", "admin", "delete"],
            "super_admin": ["read", "write", "admin", "delete", "system"]
        }

        # Initialize default roles
        self._initialize_roles()

    def _initialize_roles(self):
        """Initialize default security roles"""
        roles_data = [
            {
                "role_id": "viewer",
                "name": "Viewer",
                "description": "Read-only access to system information",
                "permissions": ["read:dashboard", "read:reports"]
            },
            {
                "role_id": "trader",
                "name": "Trader",
                "description": "Can execute trades and view positions",
                "permissions": ["read:dashboard", "read:positions", "write:orders", "read:reports"],
                "inherits_from": "viewer"
            },
            {
                "role_id": "manager",
                "name": "Manager",
                "description": "Can manage trading strategies and risk limits",
                "permissions": ["write:strategies", "write:risk_limits", "read:audit_logs"],
                "inherits_from": "trader"
            },
            {
                "role_id": "admin",
                "name": "Administrator",
                "description": "Full system administration access",
                "permissions": ["admin:users", "admin:system", "write:config", "delete:data"],
                "inherits_from": "manager"
            },
            {
                "role_id": "super_admin",
                "name": "Super Administrator",
                "description": "Complete system access including security",
                "permissions": ["system:shutdown", "admin:security", "admin:audit"],
                "inherits_from": "admin"
            }
        ]

        for role_data in roles_data:
            role = Role(**role_data)
            self.roles[role.role_id] = role

        self.logger.info(f"Initialized {len(roles_data)} default roles")

    def create_user(self, username: str, email: str, role: str, password: str = None) -> Optional[User]:
        """Create a new user account"""
        if username in [u.username for u in self.users.values()]:
            self.logger.error(f"Username {username} already exists")
            return None

        if role not in self.roles:
            self.logger.error(f"Role {role} does not exist")
            return None

        user = User(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            role=role
        )

        if password:
            user.password_hash = advanced_encryption.hash_password(password)

        self.users[user.user_id] = user

        # Audit user creation
        self.audit_logger.log_event(
            category="security",
            action="user_created",
            details={
                "user_id": user.user_id,
                "username": username,
                "role": role
            }
        )

        self.logger.info(f"Created user: {username} with role {role}")
        return user

    def authenticate_user(self, username: str, password: str, mfa_code: str = None) -> Optional[User]:
        """Authenticate user with password and optional MFA"""
        user = next((u for u in self.users.values() if u.username == username), None)

        if not user:
            self.logger.warning(f"Authentication failed: user {username} not found")
            return None

        # Check if account is locked
        if user.locked_until and datetime.now() < user.locked_until:
            self.logger.warning(f"Authentication failed: account {username} is locked")
            return None

        # Verify password
        if user.password_hash and not advanced_encryption.verify_password(password, user.password_hash):
            user.failed_attempts += 1

            # Lock account after 5 failed attempts
            if user.failed_attempts >= 5:
                user.locked_until = datetime.now() + timedelta(hours=1)
                self.logger.warning(f"Account {username} locked due to failed attempts")

            # Audit failed authentication
            self.audit_logger.log_event(
                category="security",
                action="auth_failed",
                details={
                    "username": username,
                    "failed_attempts": user.failed_attempts
                },
                severity="warning"
            )
            return None

        # Reset failed attempts on successful password auth
        user.failed_attempts = 0

        # Verify MFA if enabled
        if user.mfa_enabled and user.mfa_secret:
            if not mfa_code:
                self.logger.warning(f"MFA required for user {username}")
                return None

            if not mfa.verify_totp(user.mfa_secret, mfa_code):
                self.logger.warning(f"MFA verification failed for user {username}")
                self.audit_logger.log_event(
                    category="security",
                    action="mfa_failed",
                    details={"username": username},
                    severity="warning"
                )
                return None

        # Update last login
        user.last_login = datetime.now()

        # Audit successful authentication
        self.audit_logger.log_event(
            category="security",
            action="auth_success",
            details={
                "user_id": user.user_id,
                "username": username,
                "mfa_used": user.mfa_enabled
            }
        )

        self.logger.info(f"User {username} authenticated successfully")
        return user

    def check_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission"""
        if not user:
            return False

        # Get user role
        role = self.roles.get(user.role)
        if not role:
            return False

        # Check direct permissions
        user_permissions = set(role.permissions)

        # Add inherited permissions
        if role.inherits_from:
            parent_role = self.roles.get(role.inherits_from)
            if parent_role:
                user_permissions.update(parent_role.permissions)

        # Check permission hierarchy
        for perm in user_permissions:
            if perm in self.permission_hierarchy:
                if permission in self.permission_hierarchy[perm]:
                    return True

        # Direct permission match
        return permission in user_permissions

    def enable_mfa(self, user_id: str) -> Optional[str]:
        """Enable MFA for user and return QR code URI"""
        user = self.users.get(user_id)
        if not user:
            return None

        # Generate MFA secret
        secret = mfa.generate_secret()
        user.mfa_secret = secret
        user.mfa_enabled = False  # Will be enabled after verification

        # Return QR code URI for setup
        return mfa.get_totp_uri(user.username, secret)

    def verify_mfa_setup(self, user_id: str, code: str) -> bool:
        """Verify MFA setup and enable it"""
        user = self.users.get(user_id)
        if not user or not user.mfa_secret:
            return False

        if mfa.verify_totp(user.mfa_secret, code):
            user.mfa_enabled = True

            # Audit MFA enablement
            self.audit_logger.log_event(
                category="security",
                action="mfa_enabled",
                details={"user_id": user_id}
            )

            self.logger.info(f"MFA enabled for user {user.username}")
            return True

        return False


class APISecurity:
    """API security with rate limiting and validation"""

    def __init__(self):
        self.logger = logging.getLogger("APISecurity")
        self.audit_logger = get_audit_logger()

        # API keys
        self.api_keys: Dict[str, APIKey] = {}

        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = {}

        # Request validation rules
        self.validation_rules = {
            "max_request_size": 1024 * 1024,  # 1MB
            "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
            "required_headers": ["Authorization", "Content-Type"],
            "suspicious_patterns": ["<script", "javascript:", "eval(", "union select"]
        }

    def create_api_key(self, user_id: str, name: str, permissions: List[str], rate_limit: int = 60) -> Optional[str]:
        """Create a new API key for user"""
        # Generate secure API key
        key = secrets.token_urlsafe(32)

        # Hash the key for storage
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        api_key = APIKey(
            key_id=str(uuid.uuid4()),
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            permissions=permissions,
            rate_limit=rate_limit,
            created_at=datetime.now()
        )

        self.api_keys[api_key.key_id] = api_key

        # Audit API key creation
        self.audit_logger.log_event(
            category="security",
            action="api_key_created",
            details={
                "key_id": api_key.key_id,
                "user_id": user_id,
                "name": name,
                "permissions": permissions
            }
        )

        self.logger.info(f"Created API key: {name} for user {user_id}")
        return key

    def validate_api_request(self, api_key: str, endpoint: str, method: str, request_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate API request"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Find API key by hashing the provided key and comparing
        provided_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_record = None
        for key in self.api_keys.values():
            if hmac.compare_digest(provided_key_hash, key.key_hash):
                key_record = key
                break

        if not key_record:
            result["valid"] = False
            result["errors"].append("Invalid API key")
            return result

        # Check expiration
        if key_record.expires_at and datetime.now() > key_record.expires_at:
            result["valid"] = False
            result["errors"].append("API key expired")
            return result

        # Check rate limit
        if not self._check_rate_limit(key_record.key_id, key_record.rate_limit):
            result["valid"] = False
            result["errors"].append("Rate limit exceeded")
            return result

        # Check permissions
        required_permission = f"{method.lower()}:{endpoint}"
        # Also check generic read/write permissions
        generic_permission = f"{'read' if method.upper() in ['GET', 'HEAD'] else 'write'}:{endpoint}"

        has_specific_permission = required_permission in key_record.permissions
        has_generic_permission = generic_permission in key_record.permissions

        if not (has_specific_permission or has_generic_permission):
            result["valid"] = False
            result["errors"].append("Insufficient permissions")
            return result

        # Validate request data
        if request_data:
            validation = self._validate_request_data(request_data)
            if not validation["valid"]:
                result["valid"] = False
                result["errors"].extend(validation["errors"])

        # Update usage
        key_record.last_used = datetime.now()
        key_record.request_count += 1

        return result

    def _check_rate_limit(self, key_id: str, limit: int) -> bool:
        """Check if request is within rate limit"""
        now = datetime.now()
        window_start = now - timedelta(minutes=1)

        if key_id not in self.rate_limits:
            self.rate_limits[key_id] = {"requests": [], "blocked_until": None}

        limits = self.rate_limits[key_id]

        # Check if currently blocked
        if limits["blocked_until"] and now < limits["blocked_until"]:
            return False

        # Clean old requests
        limits["requests"] = [req for req in limits["requests"] if req > window_start]

        # Check limit
        if len(limits["requests"]) >= limit:
            # Block for 1 minute
            limits["blocked_until"] = now + timedelta(minutes=1)
            self.logger.warning(f"Rate limit exceeded for API key {key_id}")
            return False

        # Add current request
        limits["requests"].append(now)
        return True

    def _validate_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request data for security"""
        result = {"valid": True, "errors": []}

        # Check for suspicious patterns
        data_str = json.dumps(data, default=str)
        for pattern in self.validation_rules["suspicious_patterns"]:
            if pattern.lower() in data_str.lower():
                result["valid"] = False
                result["errors"].append(f"Suspicious pattern detected: {pattern}")

        # Check data size
        if len(data_str) > self.validation_rules["max_request_size"]:
            result["valid"] = False
            result["errors"].append("Request data too large")

        return result

    def rotate_api_key(self, key_id: str) -> Optional[str]:
        """Rotate an existing API key"""
        if key_id not in self.api_keys:
            return None

        old_key = self.api_keys[key_id]

        # Create new key with same permissions
        new_key = self.create_api_key(
            old_key.user_id,
            f"{old_key.name}_rotated",
            old_key.permissions,
            old_key.rate_limit
        )

        # Mark old key as expired
        old_key.expires_at = datetime.now()

        # Audit key rotation
        self.audit_logger.log_event(
            category="security",
            action="api_key_rotated",
            details={
                "old_key_id": key_id,
                "new_key_id": self.api_keys[list(self.api_keys.keys())[-1]].key_id,
                "user_id": old_key.user_id
            }
        )

        return new_key


class SecurityMonitoring:
    """Security event monitoring and alerting"""

    def __init__(self):
        self.logger = logging.getLogger("SecurityMonitor")
        self.audit_logger = get_audit_logger()

        # Security events
        self.security_events: List[SecurityEvent] = []

        # Alert thresholds
        self.alert_thresholds = {
            "failed_logins_per_hour": 5,
            "suspicious_requests_per_minute": 10,
            "api_rate_limit_hits_per_hour": 20
        }

        # Active alerts
        self.active_alerts: Dict[str, Dict[str, Any]] = {}

    def log_security_event(self, event_type: str, user_id: str = None, ip_address: str = None,
                          user_agent: str = None, details: Dict[str, Any] = None, severity: str = "info"):
        """Log a security event"""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            timestamp=datetime.now(),
            severity=severity
        )

        self.security_events.append(event)

        # Check for alerts
        self._check_alerts(event)

        # Audit the security event
        self.audit_logger.log_event(
            category="security",
            action="security_event",
            details={
                "event_type": event_type,
                "user_id": user_id,
                "severity": severity,
                "details": details
            },
            severity=severity
        )

    def _check_alerts(self, event: SecurityEvent):
        """Check if event triggers an alert"""
        # Count recent events of same type
        recent_events = [
            e for e in self.security_events
            if e.event_type == event.event_type and
            (datetime.now() - e.timestamp).total_seconds() < 3600  # Last hour
        ]

        threshold_key = f"{event.event_type}_per_hour"
        if threshold_key in self.alert_thresholds:
            threshold = self.alert_thresholds[threshold_key]
            if len(recent_events) >= threshold:
                self._trigger_alert(event.event_type, len(recent_events), threshold)

    def _trigger_alert(self, event_type: str, count: int, threshold: int):
        """Trigger a security alert"""
        alert_key = f"{event_type}_threshold"

        if alert_key not in self.active_alerts:
            alert = {
                "alert_id": str(uuid.uuid4()),
                "event_type": event_type,
                "count": count,
                "threshold": threshold,
                "triggered_at": datetime.now(),
                "status": "active"
            }

            self.active_alerts[alert_key] = alert

            self.logger.warning(f"SECURITY ALERT: {event_type} threshold exceeded ({count}/{threshold})")

            # In production, this would send notifications
            # send_alert_notification(alert)

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        recent_events = [
            e for e in self.security_events
            if (datetime.now() - e.timestamp).total_seconds() < 3600  # Last hour
        ]

        return {
            "total_events_last_hour": len(recent_events),
            "active_alerts": len(self.active_alerts),
            "alerts": list(self.active_alerts.values()),
            "critical_events": len([e for e in recent_events if e.severity == "critical"]),
            "warning_events": len([e for e in recent_events if e.severity == "warning"])
        }


# Global security system instances
mfa = MultiFactorAuthentication()
advanced_encryption = AdvancedEncryption()
rbac = RoleBasedAccessControl()
api_security = APISecurity()
security_monitoring = SecurityMonitoring()


async def initialize_security_framework():
    """Initialize the complete security framework"""
    print("[SECURITY] Initializing Advanced Security Framework...")

    # Create default admin user
    admin_user = rbac.create_user("admin", "admin@aac.com", "super_admin", "default_password_123!")
    if admin_user:
        print("✅ Created default admin user")

        # Enable MFA for admin (in production, this would be done through UI)
        mfa_uri = rbac.enable_mfa(admin_user.user_id)
        if mfa_uri:
            print("✅ MFA setup initiated for admin user")
            print(f"   MFA URI: {mfa_uri}")

    # Create API key for system
    system_key = api_security.create_api_key(
        admin_user.user_id if admin_user else "system",
        "system_api_key",
        ["read:*", "write:orders", "admin:system"],
        rate_limit=1000
    )
    if system_key:
        print("✅ Created system API key")

    # Test encryption
    test_data = "sensitive_test_data"
    encrypted = advanced_encryption.encrypt_data(test_data)
    decrypted = advanced_encryption.decrypt_data(encrypted)
    if decrypted == test_data:
        print("✅ Encryption/decryption working")
    else:
        print("❌ Encryption test failed")

    security_status = security_monitoring.get_security_status()

    print("[OK] Security framework initialized")
    print(f"  Users: {len(rbac.users)}")
    print(f"  Roles: {len(rbac.roles)}")
    print(f"  API Keys: {len(api_security.api_keys)}")
    print(f"  Security Events (last hour): {security_status['total_events_last_hour']}")
    print(f"  Active Alerts: {security_status['active_alerts']}")

    return True


if __name__ == "__main__":
    asyncio.run(initialize_security_framework())