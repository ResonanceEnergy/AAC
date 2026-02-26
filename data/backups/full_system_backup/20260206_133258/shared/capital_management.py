#!/usr/bin/env python3
"""
Capital Management & Adequacy System
====================================
Real-time capital tracking, regulatory compliance, and risk management for production deployment.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from decimal import Decimal, ROUND_DOWN
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_env
from shared.audit_logger import get_audit_logger


@dataclass
class CapitalRequirement:
    """Regulatory capital requirement"""
    jurisdiction: str
    minimum_capital: Decimal
    risk_weighted_assets_multiplier: float
    liquidity_ratio: float  # LCR requirement
    leverage_ratio: float   # Leverage ratio requirement
    description: str


@dataclass
class CapitalPosition:
    """Current capital position"""
    total_capital: Decimal
    available_capital: Decimal
    allocated_capital: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    margin_used: Decimal
    margin_available: Decimal
    last_updated: datetime
    currency: str = "USD"


@dataclass
class CapitalThreshold:
    """Capital threshold for alerts"""
    threshold_type: str  # "minimum", "warning", "critical"
    percentage: float
    amount: Decimal
    action_required: str
    auto_stop_trading: bool = False


class CapitalManagementSystem:
    """Real-time capital management and regulatory compliance"""

    def __init__(self):
        self.logger = logging.getLogger("CapitalManagement")
        self.audit_logger = get_audit_logger()

        # Capital requirements by jurisdiction
        self.capital_requirements = self._initialize_requirements()

        # Current capital position
        self.capital_position: Optional[CapitalPosition] = None

        # Capital thresholds
        self.capital_thresholds = self._initialize_thresholds()

        # Capital monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # Capital history for compliance reporting
        self.capital_history: List[Dict[str, Any]] = []

        # Load initial capital from config
        self._load_initial_capital()

    def _initialize_requirements(self) -> Dict[str, CapitalRequirement]:
        """Initialize regulatory capital requirements"""
        return {
            "FINRA": CapitalRequirement(
                jurisdiction="FINRA",
                minimum_capital=Decimal("100000"),  # $100K for introducing brokers
                risk_weighted_assets_multiplier=0.08,  # 8% risk-weighted capital
                liquidity_ratio=1.0,  # 100% LCR
                leverage_ratio=0.1,   # 10:1 leverage max
                description="FINRA capital requirements for retail forex dealers"
            ),
            "CFTC": CapitalRequirement(
                jurisdiction="CFTC",
                minimum_capital=Decimal("20000000"),  # $20M for FCMs
                risk_weighted_assets_multiplier=0.08,
                liquidity_ratio=1.0,
                leverage_ratio=0.1,
                description="CFTC capital requirements for futures commission merchants"
            ),
            "SEC": CapitalRequirement(
                jurisdiction="SEC",
                minimum_capital=Decimal("25000000"),  # $25M for broker-dealers
                risk_weighted_assets_multiplier=0.08,
                liquidity_ratio=1.0,
                leverage_ratio=0.1,
                description="SEC capital requirements for registered broker-dealers"
            ),
            "BASEL_III": CapitalRequirement(
                jurisdiction="BASEL_III",
                minimum_capital=Decimal("0"),  # No fixed minimum, risk-based
                risk_weighted_assets_multiplier=0.08,  # 8% CET1 ratio
                liquidity_ratio=1.0,  # 100% LCR
                leverage_ratio=0.03,  # 3% leverage ratio
                description="Basel III international banking standards"
            )
        }

    def _initialize_thresholds(self) -> List[CapitalThreshold]:
        """Initialize capital alert thresholds"""
        return [
            CapitalThreshold(
                threshold_type="minimum",
                percentage=10.0,
                amount=Decimal("0"),  # Will be calculated based on requirements
                action_required="Immediate capital infusion required",
                auto_stop_trading=True
            ),
            CapitalThreshold(
                threshold_type="warning",
                percentage=20.0,
                amount=Decimal("0"),
                action_required="Monitor capital closely, reduce risk exposure",
                auto_stop_trading=False
            ),
            CapitalThreshold(
                threshold_type="critical",
                percentage=5.0,
                amount=Decimal("0"),
                action_required="EMERGENCY: Halt all trading operations immediately",
                auto_stop_trading=True
            )
        ]

    def _load_initial_capital(self):
        """Load initial capital from configuration"""
        config = get_config()

        # Default capital if not configured
        initial_capital = getattr(config.risk, 'initial_capital', None)
        if initial_capital is None:
            # Try to get from environment or use default
            initial_capital = Decimal(str(get_env('INITIAL_CAPITAL', '1000000.0')))
        else:
            initial_capital = Decimal(str(initial_capital))

        self.capital_position = CapitalPosition(
            total_capital=initial_capital,
            available_capital=initial_capital,
            allocated_capital=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            margin_used=Decimal("0"),
            margin_available=initial_capital,
            last_updated=datetime.now(),
            currency="USD"
        )

        self.logger.info(f"Initial capital loaded: ${initial_capital:,.2f}")

    async def update_capital_position(self,
                                    realized_pnl: Optional[Decimal] = None,
                                    unrealized_pnl: Optional[Decimal] = None,
                                    margin_used: Optional[Decimal] = None) -> bool:
        """Update current capital position"""
        if not self.capital_position:
            return False

        # Update P&L
        if realized_pnl is not None:
            self.capital_position.realized_pnl += realized_pnl
            self.capital_position.total_capital += realized_pnl
            self.capital_position.available_capital += realized_pnl

        if unrealized_pnl is not None:
            # Remove old unrealized P&L from total
            self.capital_position.total_capital -= self.capital_position.unrealized_pnl
            # Add new unrealized P&L
            self.capital_position.unrealized_pnl = unrealized_pnl
            self.capital_position.total_capital += unrealized_pnl

        # Update margin
        if margin_used is not None:
            old_margin = self.capital_position.margin_used
            self.capital_position.margin_used = margin_used
            self.capital_position.margin_available = self.capital_position.available_capital - margin_used

            # Update allocated capital based on margin usage
            margin_change = margin_used - old_margin
            self.capital_position.allocated_capital += margin_change
            self.capital_position.available_capital -= margin_change

        self.capital_position.last_updated = datetime.now()

        # Record in history
        await self._record_capital_history()

        # Check thresholds
        await self._check_capital_thresholds()

        # Audit the update
        await self.audit_logger.log_event(
            category="capital",
            action="position_updated",
            details={
                "total_capital": float(self.capital_position.total_capital),
                "available_capital": float(self.capital_position.available_capital),
                "allocated_capital": float(self.capital_position.allocated_capital),
                "realized_pnl": float(self.capital_position.realized_pnl),
                "unrealized_pnl": float(self.capital_position.unrealized_pnl),
                "margin_used": float(self.capital_position.margin_used)
            }
        )

        return True

    async def _record_capital_history(self):
        """Record capital position in history for compliance"""
        if not self.capital_position:
            return

        history_entry = {
            "timestamp": self.capital_position.last_updated.isoformat(),
            "total_capital": float(self.capital_position.total_capital),
            "available_capital": float(self.capital_position.available_capital),
            "allocated_capital": float(self.capital_position.allocated_capital),
            "unrealized_pnl": float(self.capital_position.unrealized_pnl),
            "realized_pnl": float(self.capital_position.realized_pnl),
            "margin_used": float(self.capital_position.margin_used),
            "margin_available": float(self.capital_position.margin_available)
        }

        self.capital_history.append(history_entry)

        # Keep only last 90 days of history
        cutoff_date = datetime.now() - timedelta(days=90)
        self.capital_history = [
            entry for entry in self.capital_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
        ]

    async def _check_capital_thresholds(self):
        """Check if capital has breached any thresholds"""
        if not self.capital_position:
            return

        total_capital = self.capital_position.total_capital

        for threshold in self.capital_thresholds:
            # Calculate threshold amount based on initial capital
            initial_capital = self.capital_history[0]["total_capital"] if self.capital_history else float(total_capital)
            threshold.amount = Decimal(str(initial_capital * (threshold.percentage / 100.0)))

            if total_capital <= threshold.amount:
                self.logger.warning(f"CAPITAL THRESHOLD BREACHED: {threshold.threshold_type.upper()}")
                self.logger.warning(f"Current: ${total_capital:,.2f}, Threshold: ${threshold.amount:,.2f}")
                self.logger.warning(f"Action Required: {threshold.action_required}")

                # Audit the threshold breach
                await self.audit_logger.log_event(
                    category="capital",
                    action="threshold_breached",
                    details={
                        "threshold_type": threshold.threshold_type,
                        "current_capital": float(total_capital),
                        "threshold_amount": float(threshold.amount),
                        "action_required": threshold.action_required,
                        "auto_stop_trading": threshold.auto_stop_trading
                    },
                    severity="high" if threshold.auto_stop_trading else "medium"
                )

                # Emergency stop if required
                if threshold.auto_stop_trading:
                    await self._emergency_stop_trading()

    async def _emergency_stop_trading(self):
        """Emergency stop all trading operations"""
        self.logger.critical("EMERGENCY CAPITAL STOP: Halting all trading operations")

        # This would integrate with the trading engine to stop all operations
        # For now, just log the emergency stop
        await self.audit_logger.log_event(
            category="capital",
            action="emergency_stop",
            details={
                "reason": "capital_threshold_breached",
                "current_capital": float(self.capital_position.total_capital) if self.capital_position else 0
            },
            severity="critical"
        )

    async def check_capital_adequacy(self, jurisdiction: str = "FINRA") -> Dict[str, Any]:
        """Check if current capital meets regulatory requirements"""
        if not self.capital_position:
            return {"compliant": False, "error": "No capital position available"}

        requirement = self.capital_requirements.get(jurisdiction)
        if not requirement:
            return {"compliant": False, "error": f"Unknown jurisdiction: {jurisdiction}"}

        total_capital = self.capital_position.total_capital
        available_capital = self.capital_position.available_capital

        # Check minimum capital requirement
        min_capital_compliant = total_capital >= requirement.minimum_capital

        # Check risk-weighted assets (simplified - assume 12.5x leverage = 8% requirement)
        risk_weighted_compliant = available_capital >= (total_capital * Decimal(str(requirement.risk_weighted_assets_multiplier)))

        # Check liquidity ratio (simplified)
        liquidity_compliant = available_capital >= (total_capital * Decimal(str(requirement.liquidity_ratio)))

        # Check leverage ratio
        max_leveraged_exposure = total_capital * Decimal(str(1.0 / requirement.leverage_ratio))
        current_leverage = self.capital_position.margin_used / total_capital if total_capital > 0 else Decimal("0")
        leverage_compliant = current_leverage <= Decimal(str(requirement.leverage_ratio))

        overall_compliant = min_capital_compliant and risk_weighted_compliant and liquidity_compliant and leverage_compliant

        result = {
            "jurisdiction": jurisdiction,
            "compliant": overall_compliant,
            "total_capital": float(total_capital),
            "available_capital": float(available_capital),
            "minimum_capital_required": float(requirement.minimum_capital),
            "checks": {
                "minimum_capital": {
                    "compliant": min_capital_compliant,
                    "required": float(requirement.minimum_capital),
                    "current": float(total_capital)
                },
                "risk_weighted_assets": {
                    "compliant": risk_weighted_compliant,
                    "ratio": requirement.risk_weighted_assets_multiplier
                },
                "liquidity_ratio": {
                    "compliant": liquidity_compliant,
                    "ratio": requirement.liquidity_ratio
                },
                "leverage_ratio": {
                    "compliant": leverage_compliant,
                    "max_ratio": requirement.leverage_ratio,
                    "current_ratio": float(current_leverage)
                }
            }
        }

        # Audit the compliance check
        await self.audit_logger.log_event(
            category="compliance",
            action="capital_adequacy_check",
            details=result
        )

        return result

    async def allocate_capital(self, amount: Decimal, purpose: str) -> bool:
        """Allocate capital for trading"""
        if not self.capital_position:
            return False

        if self.capital_position.available_capital < amount:
            self.logger.error(f"Insufficient capital for allocation: requested ${amount:,.2f}, available ${self.capital_position.available_capital:,.2f}")
            return False

        self.capital_position.allocated_capital += amount
        self.capital_position.available_capital -= amount

        # Audit the allocation
        await self.audit_logger.log_event(
            category="capital",
            action="capital_allocated",
            details={
                "amount": float(amount),
                "purpose": purpose,
                "remaining_available": float(self.capital_position.available_capital)
            }
        )

        return True

    async def deallocate_capital(self, amount: Decimal, purpose: str) -> bool:
        """Deallocate capital after trading"""
        if not self.capital_position:
            return False

        if self.capital_position.allocated_capital < amount:
            self.logger.warning(f"Deallocating more capital than allocated: ${amount:,.2f} > ${self.capital_position.allocated_capital:,.2f}")
            amount = self.capital_position.allocated_capital

        self.capital_position.allocated_capital -= amount
        self.capital_position.available_capital += amount

        # Audit the deallocation
        await self.audit_logger.log_event(
            category="capital",
            action="capital_deallocated",
            details={
                "amount": float(amount),
                "purpose": purpose,
                "remaining_allocated": float(self.capital_position.allocated_capital)
            }
        )

        return True

    def get_capital_status(self) -> Dict[str, Any]:
        """Get current capital status"""
        if not self.capital_position:
            return {"status": "no_capital_position"}

        return {
            "total_capital": float(self.capital_position.total_capital),
            "available_capital": float(self.capital_position.available_capital),
            "allocated_capital": float(self.capital_position.allocated_capital),
            "unrealized_pnl": float(self.capital_position.unrealized_pnl),
            "realized_pnl": float(self.capital_position.realized_pnl),
            "margin_used": float(self.capital_position.margin_used),
            "margin_available": float(self.capital_position.margin_available),
            "last_updated": self.capital_position.last_updated.isoformat(),
            "currency": self.capital_position.currency
        }

    async def start_capital_monitoring(self):
        """Start real-time capital monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._capital_monitoring_loop())

        self.logger.info("Capital monitoring started")

    async def stop_capital_monitoring(self):
        """Stop capital monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Capital monitoring stopped")

    async def _capital_monitoring_loop(self):
        """Continuous capital monitoring loop"""
        while self.monitoring_active:
            try:
                # Check capital adequacy every 5 minutes
                adequacy = await self.check_capital_adequacy()
                if not adequacy.get("compliant", False):
                    self.logger.warning("Capital adequacy check failed during monitoring")

                # Update thresholds
                await self._check_capital_thresholds()

                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                self.logger.error(f"Error in capital monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    def get_capital_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get capital history for the specified number of days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            entry for entry in self.capital_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
        ]


# Global capital management system instance
capital_management_system = CapitalManagementSystem()


async def initialize_capital_management():
    """Initialize the capital management system"""
    print("[CAPITAL] Initializing Capital Management System...")

    # Start monitoring
    await capital_management_system.start_capital_monitoring()

    # Check initial compliance
    adequacy = await capital_management_system.check_capital_adequacy()
    status = capital_management_system.get_capital_status()

    print("[OK] Capital management system initialized")
    print(f"  Total Capital: ${status['total_capital']:,.2f}")
    print(f"  Available Capital: ${status['available_capital']:,.2f}")
    print(f"  Capital Adequacy: {'COMPLIANT' if adequacy['compliant'] else 'NON-COMPLIANT'}")

    return True


if __name__ == "__main__":
    asyncio.run(initialize_capital_management())