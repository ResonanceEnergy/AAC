#!/usr/bin/env python3
"""
License Management System
=========================
Enterprise license management for direct exchange feeds and data vendors.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import get_audit_logger


class LicenseType(Enum):
    """Types of data licenses"""
    EXCHANGE_DIRECT = "exchange_direct"  # NYSE, CME, NASDAQ direct feeds
    DATA_VENDOR = "data_vendor"         # Bloomberg, Refinitiv, etc.
    MARKET_DATA = "market_data"         # General market data
    NEWS_FEED = "news_feed"            # News and sentiment data
    ALTERNATIVE_DATA = "alternative_data"  # Social media, web data


class LicenseStatus(Enum):
    """License status"""
    ACTIVE = "active"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    PENDING_RENEWAL = "pending_renewal"
    OVER_LIMIT = "over_limit"


@dataclass
class LicenseUsage:
    """License usage tracking"""
    requests_today: int = 0
    requests_this_month: int = 0
    data_volume_mb: float = 0.0
    last_request: Optional[datetime] = None
    peak_usage_hour: Optional[int] = None


@dataclass
class License:
    """Enterprise license information"""
    license_id: str
    provider: str  # "nyse", "cme", "bloomberg", etc.
    license_type: LicenseType
    license_name: str
    status: LicenseStatus = LicenseStatus.ACTIVE
    issued_date: datetime = field(default_factory=datetime.now)
    expiry_date: Optional[datetime] = None
    auto_renew: bool = False
    renewal_period_days: int = 365

    # Usage limits
    daily_request_limit: Optional[int] = None
    monthly_request_limit: Optional[int] = None
    data_volume_limit_mb: Optional[float] = None

    # Current usage
    usage: LicenseUsage = field(default_factory=LicenseUsage)

    # Contact and compliance
    vendor_contact: Optional[str] = None
    compliance_officer: Optional[str] = None
    contract_terms: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    cost_per_month: Optional[float] = None
    notes: str = ""


@dataclass
class LicenseAlert:
    """License compliance alert"""
    alert_id: str
    license_id: str
    alert_type: str  # "expiry_warning", "usage_limit", "compliance_issue"
    severity: str  # "low", "medium", "high", "critical"
    message: str
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


class LicenseManager:
    """Enterprise license management system"""

    def __init__(self):
        self.logger = logging.getLogger("LicenseManager")
        self.audit_logger = get_audit_logger()
        self.licenses: Dict[str, License] = {}
        self.alerts: List[LicenseAlert] = []
        self.license_store_path = PROJECT_ROOT / "config" / "licenses.json"

        # Load existing licenses
        asyncio.create_task(self._load_licenses())

        # Start monitoring tasks
        asyncio.create_task(self._start_monitoring())

    async def _load_licenses(self):
        """Load licenses from storage"""
        if self.license_store_path.exists():
            try:
                with open(self.license_store_path, 'r') as f:
                    licenses_data = json.load(f)

                for license_data in licenses_data:
                    # Convert string dates back to datetime
                    if license_data.get('issued_date'):
                        license_data['issued_date'] = datetime.fromisoformat(license_data['issued_date'])
                    if license_data.get('expiry_date'):
                        license_data['expiry_date'] = datetime.fromisoformat(license_data['expiry_date'])
                    if license_data['usage'].get('last_request'):
                        license_data['usage']['last_request'] = datetime.fromisoformat(license_data['usage']['last_request'])

                    license_obj = License(**license_data)
                    self.licenses[license_obj.license_id] = license_obj

                self.logger.info(f"Loaded {len(self.licenses)} licenses")

            except Exception as e:
                self.logger.error(f"Failed to load licenses: {e}")

    async def _save_licenses(self):
        """Save licenses to storage"""
        try:
            licenses_data = []
            for license in self.licenses.values():
                license_dict = license.__dict__.copy()
                # Convert datetime objects to strings
                license_dict['issued_date'] = license.issued_date.isoformat()
                if license.expiry_date:
                    license_dict['expiry_date'] = license.expiry_date.isoformat()
                if license.usage.last_request:
                    license_dict['usage'] = license.usage.__dict__.copy()
                    license_dict['usage']['last_request'] = license.usage.last_request.isoformat()

                licenses_data.append(license_dict)

            with open(self.license_store_path, 'w') as f:
                json.dump(licenses_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save licenses: {e}")

    async def add_license(
        self,
        provider: str,
        license_type: LicenseType,
        license_name: str,
        expiry_date: Optional[datetime] = None,
        daily_limit: Optional[int] = None,
        monthly_limit: Optional[int] = None,
        data_limit_mb: Optional[float] = None,
        cost_per_month: Optional[float] = None,
        **kwargs
    ) -> str:
        """Add a new enterprise license"""
        license_id = f"{provider}_{license_type.value}_{len(self.licenses) + 1}"

        license = License(
            license_id=license_id,
            provider=provider,
            license_type=license_type,
            license_name=license_name,
            expiry_date=expiry_date,
            daily_request_limit=daily_limit,
            monthly_request_limit=monthly_limit,
            data_volume_limit_mb=data_limit_mb,
            cost_per_month=cost_per_month,
            **kwargs
        )

        self.licenses[license_id] = license
        await self._save_licenses()

        await self.audit_logger.log_api_call(
            exchange=provider,
            method="ADD_LICENSE",
            endpoint=f"license:{license_name}",
            status="success",
            details={"license_id": license_id, "type": license_type.value}
        )

        self.logger.info(f"Added license: {license_id} for {provider}")
        return license_id

    def get_license(self, provider: str, license_type: LicenseType) -> Optional[License]:
        """Get license for a provider and type"""
        for license in self.licenses.values():
            if license.provider == provider and license.license_type == license_type and license.status == LicenseStatus.ACTIVE:
                return license
        return None

    async def check_usage_limit(self, license_id: str, request_count: int = 1, data_volume_mb: float = 0.0) -> bool:
        """Check if usage is within license limits"""
        if license_id not in self.licenses:
            return False

        license = self.licenses[license_id]

        # Update usage
        license.usage.requests_today += request_count
        license.usage.requests_this_month += request_count
        license.usage.data_volume_mb += data_volume_mb
        license.usage.last_request = datetime.now()

        # Check limits
        if license.daily_request_limit and license.usage.requests_today > license.daily_request_limit:
            await self._create_alert(license_id, "usage_limit", "high", f"Daily request limit exceeded: {license.usage.requests_today}/{license.daily_request_limit}")
            license.status = LicenseStatus.OVER_LIMIT
            return False

        if license.monthly_request_limit and license.usage.requests_this_month > license.monthly_request_limit:
            await self._create_alert(license_id, "usage_limit", "critical", f"Monthly request limit exceeded: {license.usage.requests_this_month}/{license.monthly_request_limit}")
            license.status = LicenseStatus.OVER_LIMIT
            return False

        if license.data_volume_limit_mb and license.usage.data_volume_mb > license.data_volume_limit_mb:
            await self._create_alert(license_id, "usage_limit", "high", f"Data volume limit exceeded: {license.usage.data_volume_mb:.1f}MB/{license.data_volume_limit_mb}MB")
            license.status = LicenseStatus.OVER_LIMIT
            return False

        await self._save_licenses()
        return True

    async def _create_alert(self, license_id: str, alert_type: str, severity: str, message: str):
        """Create a license alert"""
        alert = LicenseAlert(
            alert_id=f"alert_{license_id}_{len(self.alerts) + 1}",
            license_id=license_id,
            alert_type=alert_type,
            severity=severity,
            message=message
        )

        self.alerts.append(alert)

        # Log alert
        await self.audit_logger.log_api_call(
            exchange=self.licenses[license_id].provider,
            method="LICENSE_ALERT",
            endpoint=f"license:{license_id}",
            status="warning",
            details={"alert_type": alert_type, "severity": severity, "message": message}
        )

        self.logger.warning(f"License alert: {message}")

    async def _start_monitoring(self):
        """Start license monitoring tasks"""
        while True:
            try:
                await self._check_expiring_licenses()
                await self._reset_daily_usage()
                await self._reset_monthly_usage()
                await asyncio.sleep(3600)  # Check every hour
            except Exception as e:
                self.logger.error(f"License monitoring error: {e}")
                await asyncio.sleep(3600)

    async def _check_expiring_licenses(self):
        """Check for licenses expiring soon"""
        now = datetime.now()

        for license in self.licenses.values():
            if license.expiry_date and license.status == LicenseStatus.ACTIVE:
                days_until_expiry = (license.expiry_date - now).days

                if days_until_expiry <= 30 and days_until_expiry > 0:
                    await self._create_alert(
                        license.license_id,
                        "expiry_warning",
                        "medium" if days_until_expiry > 7 else "high",
                        f"License expires in {days_until_expiry} days: {license.license_name}"
                    )

                elif days_until_expiry <= 0:
                    license.status = LicenseStatus.EXPIRED
                    await self._create_alert(
                        license.license_id,
                        "expiry_warning",
                        "critical",
                        f"License has expired: {license.license_name}"
                    )

        await self._save_licenses()

    async def _reset_daily_usage(self):
        """Reset daily usage counters"""
        now = datetime.now()

        # Reset at midnight
        if now.hour == 0 and now.minute < 5:  # Within first 5 minutes of day
            for license in self.licenses.values():
                license.usage.requests_today = 0
                license.usage.peak_usage_hour = None

            await self._save_licenses()
            self.logger.info("Reset daily license usage counters")

    async def _reset_monthly_usage(self):
        """Reset monthly usage counters"""
        now = datetime.now()

        # Reset on the 1st of each month
        if now.day == 1 and now.hour == 0 and now.minute < 5:
            for license in self.licenses.values():
                license.usage.requests_this_month = 0

            await self._save_licenses()
            self.logger.info("Reset monthly license usage counters")

    async def renew_license(self, license_id: str, new_expiry_date: Optional[datetime] = None) -> bool:
        """Renew a license"""
        if license_id not in self.licenses:
            return False

        license = self.licenses[license_id]

        if new_expiry_date:
            license.expiry_date = new_expiry_date
        elif license.renewal_period_days:
            license.expiry_date = datetime.now() + timedelta(days=license.renewal_period_days)

        license.status = LicenseStatus.ACTIVE

        await self._save_licenses()

        await self.audit_logger.log_api_call(
            exchange=license.provider,
            method="RENEW_LICENSE",
            endpoint=f"license:{license.license_name}",
            status="success",
            details={"license_id": license_id, "new_expiry": license.expiry_date.isoformat() if license.expiry_date else None}
        )

        self.logger.info(f"Renewed license: {license_id}")
        return True

    def get_license_status(self, provider: str, license_type: LicenseType) -> Optional[LicenseStatus]:
        """Get license status"""
        license = self.get_license(provider, license_type)
        return license.status if license else None

    def list_licenses(self, provider: Optional[str] = None, status: Optional[LicenseStatus] = None) -> List[Dict]:
        """List licenses with optional filtering"""
        licenses_list = []
        for license in self.licenses.values():
            if provider and license.provider != provider:
                continue
            if status and license.status != status:
                continue

            license_dict = {
                "license_id": license.license_id,
                "provider": license.provider,
                "type": license.license_type.value,
                "name": license.license_name,
                "status": license.status.value,
                "issued_date": license.issued_date.isoformat(),
                "expiry_date": license.expiry_date.isoformat() if license.expiry_date else None,
                "daily_limit": license.daily_request_limit,
                "monthly_limit": license.monthly_request_limit,
                "current_daily_usage": license.usage.requests_today,
                "current_monthly_usage": license.usage.requests_this_month,
                "data_usage_mb": license.usage.data_volume_mb,
                "cost_per_month": license.cost_per_month,
            }
            licenses_list.append(license_dict)

        return licenses_list

    def get_alerts(self, acknowledged: bool = False) -> List[Dict]:
        """Get license alerts"""
        alerts_list = []
        for alert in self.alerts:
            if alert.acknowledged == acknowledged:
                alerts_list.append({
                    "alert_id": alert.alert_id,
                    "license_id": alert.license_id,
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "created_at": alert.created_at.isoformat(),
                    "acknowledged": alert.acknowledged,
                })
        return alerts_list

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a license alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self.logger.info(f"Acknowledged alert: {alert_id}")
                return True
        return False

    async def initialize_default_licenses(self):
        """Initialize default license configurations for common providers"""
        default_licenses = [
            {
                "provider": "nyse",
                "license_type": LicenseType.EXCHANGE_DIRECT,
                "license_name": "NYSE Direct Feed License",
                "daily_limit": 100000,
                "monthly_limit": 2000000,
                "cost_per_month": 5000.0,
                "expiry_date": datetime.now() + timedelta(days=365),
            },
            {
                "provider": "cme",
                "license_type": LicenseType.EXCHANGE_DIRECT,
                "license_name": "CME Direct Feed License",
                "daily_limit": 200000,
                "monthly_limit": 4000000,
                "cost_per_month": 8000.0,
                "expiry_date": datetime.now() + timedelta(days=365),
            },
            {
                "provider": "bloomberg",
                "license_type": LicenseType.DATA_VENDOR,
                "license_name": "Bloomberg Terminal License",
                "daily_limit": 50000,
                "monthly_limit": 1000000,
                "cost_per_month": 2000.0,
                "expiry_date": datetime.now() + timedelta(days=365),
            },
            {
                "provider": "refinitiv",
                "license_type": LicenseType.DATA_VENDOR,
                "license_name": "Refinitiv Eikon License",
                "daily_limit": 75000,
                "monthly_limit": 1500000,
                "cost_per_month": 1800.0,
                "expiry_date": datetime.now() + timedelta(days=365),
            },
        ]

        for license_config in default_licenses:
            license_type = license_config.pop("license_type")
            await self.add_license(license_type=license_type, **license_config)

        self.logger.info("Initialized default enterprise licenses")


# Global license manager instance
license_manager = LicenseManager()


async def initialize_license_system():
    """Initialize the license management system"""
    await license_manager._load_licenses()
    await license_manager.initialize_default_licenses()

    print("[OK] License management system initialized")
    print(f"  Active licenses: {len([l for l in license_manager.licenses.values() if l.status == LicenseStatus.ACTIVE])}")
    print(f"  Pending alerts: {len(license_manager.get_alerts(acknowledged=False))}")


if __name__ == "__main__":
    # Example usage
    asyncio.run(initialize_license_system())