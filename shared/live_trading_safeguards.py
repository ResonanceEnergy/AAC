#!/usr/bin/env python3
"""
Live Trading Safeguards
======================
Comprehensive risk management and safety controls for live trading.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
import threading
import psutil
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import get_audit_logger
from shared.monitoring import MonitoringService


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SafetyAction(Enum):
    """Safety actions that can be taken"""
    NONE = "none"
    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    HALT_TRADING = "halt_trading"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class RiskLimits:
    """Risk management limits"""
    max_position_size: float = 10000.0  # Max position per symbol
    max_portfolio_value: float = 100000.0  # Max total portfolio value
    max_daily_loss: float = 5000.0  # Max daily loss
    max_drawdown: float = 0.1  # Max drawdown (10%)
    max_leverage: float = 2.0  # Max leverage ratio
    max_concentration: float = 0.2  # Max concentration per symbol (20%)
    max_trades_per_hour: int = 50  # Max trades per hour
    max_trades_per_day: int = 200  # Max trades per day


@dataclass
class SafetyRule:
    """Safety rule definition"""
    rule_id: str
    name: str
    description: str
    condition: Callable[[Dict[str, Any]], bool]
    action: SafetyAction
    alert_level: AlertLevel
    enabled: bool = True
    cooldown_period: int = 300  # seconds
    last_triggered: Optional[datetime] = None


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    leverage_ratio: float = 0.0
    concentration_risk: float = 0.0
    volatility_risk: float = 0.0
    liquidity_risk: float = 0.0
    trades_today: int = 0
    trades_this_hour: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SafetyAlert:
    """Safety alert record"""
    alert_id: str
    rule_id: str
    message: str
    level: AlertLevel
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    action_taken: SafetyAction = SafetyAction.NONE
    details: Dict[str, Any] = field(default_factory=dict)


class LiveTradingSafeguards:
    """Comprehensive live trading safety system"""

    def __init__(self):
        self.logger = logging.getLogger("LiveTradingSafeguards")
        self.audit_logger = get_audit_logger()

        # Risk limits
        self.risk_limits = RiskLimits()

        # Safety rules
        self.safety_rules: Dict[str, SafetyRule] = {}
        self._initialize_safety_rules()

        # Risk metrics
        self.risk_metrics = RiskMetrics()

        # Alerts and incidents
        self.active_alerts: Dict[str, SafetyAlert] = {}
        self.alert_history: List[SafetyAlert] = []

        # Trading state
        self.trading_halted = False
        self.emergency_shutdown = False
        self.last_trade_time = None
        self.daily_trade_count = 0
        self.hourly_trade_count = 0
        self.hour_start_time = datetime.now()

        # Circuit breakers
        self.circuit_breakers = {
            "daily_loss_limit": False,
            "drawdown_limit": False,
            "volatility_spike": False,
            "connectivity_loss": False,
        }

        # Monitoring
        self.monitoring_active = True
        self.monitoring_interval = 30  # seconds
        self.health_check_interval = 60  # seconds

        # Load configuration
        # asyncio.create_task(self._load_configuration())  # Moved to initialize()

        # Start monitoring loops
        # asyncio.create_task(self._start_monitoring_loop())  # Moved to initialize()
        # asyncio.create_task(self._start_health_check_loop())  # Moved to initialize()

    async def initialize(self):
        """Initialize the live trading safeguards with async tasks"""
        # Load configuration
        await self._load_configuration()

        # Start monitoring loops
        asyncio.create_task(self._start_monitoring_loop())
        asyncio.create_task(self._start_health_check_loop())

    async def _load_configuration(self):
        """Load safety configuration"""
        try:
            config = get_config()
            # For now, skip loading custom configuration as Config class doesn't support dict-like access
            # safety_config = config.get("live_trading_safeguards", {})

            # Use default risk limits
            self.logger.info("Using default live trading safeguards configuration")

        except Exception as e:
            self.logger.error(f"Failed to load safety configuration: {e}")

    def _initialize_safety_rules(self):
        """Initialize default safety rules"""

        # Daily loss limit
        self.safety_rules["daily_loss_limit"] = SafetyRule(
            rule_id="daily_loss_limit",
            name="Daily Loss Limit",
            description="Stop trading if daily loss exceeds limit",
            condition=lambda metrics: abs(metrics.get("daily_pnl", 0)) > self.risk_limits.max_daily_loss,
            action=SafetyAction.HALT_TRADING,
            alert_level=AlertLevel.CRITICAL
        )

        # Drawdown limit
        self.safety_rules["drawdown_limit"] = SafetyRule(
            rule_id="drawdown_limit",
            name="Drawdown Limit",
            description="Stop trading if drawdown exceeds limit",
            condition=lambda metrics: metrics.get("max_drawdown", 0) > self.risk_limits.max_drawdown,
            action=SafetyAction.REDUCE_POSITION,
            alert_level=AlertLevel.CRITICAL
        )

        # Position size limit
        self.safety_rules["position_size_limit"] = SafetyRule(
            rule_id="position_size_limit",
            name="Position Size Limit",
            description="Reduce position if size exceeds limit",
            condition=lambda metrics: metrics.get("largest_position", 0) > self.risk_limits.max_position_size,
            action=SafetyAction.REDUCE_POSITION,
            alert_level=AlertLevel.WARNING
        )

        # Concentration risk
        self.safety_rules["concentration_risk"] = SafetyRule(
            rule_id="concentration_risk",
            name="Concentration Risk",
            description="Alert on high concentration in single symbol",
            condition=lambda metrics: metrics.get("concentration_risk", 0) > self.risk_limits.max_concentration,
            action=SafetyAction.NONE,
            alert_level=AlertLevel.WARNING
        )

        # Trading frequency limit
        self.safety_rules["trading_frequency"] = SafetyRule(
            rule_id="trading_frequency",
            name="Trading Frequency",
            description="Slow down trading if too frequent",
            condition=lambda metrics: metrics.get("trades_this_hour", 0) > self.risk_limits.max_trades_per_hour,
            action=SafetyAction.HALT_TRADING,
            alert_level=AlertLevel.ERROR
        )

        # Leverage limit
        self.safety_rules["leverage_limit"] = SafetyRule(
            rule_id="leverage_limit",
            name="Leverage Limit",
            description="Reduce leverage if too high",
            condition=lambda metrics: metrics.get("leverage_ratio", 0) > self.risk_limits.max_leverage,
            action=SafetyAction.REDUCE_POSITION,
            alert_level=AlertLevel.WARNING
        )

        # System health
        self.safety_rules["system_health"] = SafetyRule(
            rule_id="system_health",
            name="System Health",
            description="Emergency shutdown on critical system issues",
            condition=lambda metrics: metrics.get("system_health", 1.0) < 0.5,
            action=SafetyAction.EMERGENCY_SHUTDOWN,
            alert_level=AlertLevel.CRITICAL
        )

    def _add_custom_rule(self, rule_config: Dict[str, Any]):
        """Add a custom safety rule"""
        try:
            # Create condition function from string
            condition_code = rule_config.get("condition_code", "")
            if condition_code:
                # This is a simplified implementation - in production you'd want proper sandboxing
                condition_func = eval(f"lambda metrics: {condition_code}")
            else:
                condition_func = lambda metrics: False

            rule = SafetyRule(
                rule_id=rule_config.get("rule_id", f"custom_{len(self.safety_rules)}"),
                name=rule_config.get("name", "Custom Rule"),
                description=rule_config.get("description", ""),
                condition=condition_func,
                action=SafetyAction(rule_config.get("action", "none")),
                alert_level=AlertLevel(rule_config.get("alert_level", "warning")),
                enabled=rule_config.get("enabled", True),
                cooldown_period=rule_config.get("cooldown_period", 300)
            )

            self.safety_rules[rule.rule_id] = rule
            self.logger.info(f"Added custom safety rule: {rule.rule_id}")

        except Exception as e:
            self.logger.error(f"Failed to add custom rule: {e}")

    async def check_trade_safety(self, trade_details: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if a trade is safe to execute"""
        if self.emergency_shutdown:
            return False, "Emergency shutdown active"

        if self.trading_halted:
            return False, "Trading halted by safety system"

        # Update trade counters
        await self._update_trade_counters()

        # Check trading frequency
        if self.hourly_trade_count >= self.risk_limits.max_trades_per_hour:
            return False, f"Hourly trade limit exceeded ({self.risk_limits.max_trades_per_hour})"

        if self.daily_trade_count >= self.risk_limits.max_trades_per_day:
            return False, f"Daily trade limit exceeded ({self.risk_limits.max_trades_per_day})"

        # Check position size
        position_size = trade_details.get("quantity", 0) * trade_details.get("price", 0)
        if position_size > self.risk_limits.max_position_size:
            return False, f"Position size exceeds limit (${self.risk_limits.max_position_size:,.2f})"

        # Check portfolio impact
        new_portfolio_value = self.risk_metrics.portfolio_value + position_size
        if new_portfolio_value > self.risk_limits.max_portfolio_value:
            return False, f"Portfolio value would exceed limit (${self.risk_limits.max_portfolio_value:,.2f})"

        return True, "Trade approved"

    async def execute_safety_check(self) -> List[SafetyAlert]:
        """Execute all safety rules and return triggered alerts"""
        alerts = []
        current_metrics = self._get_current_metrics()

        for rule in self.safety_rules.values():
            if not rule.enabled:
                continue

            # Check cooldown period
            if rule.last_triggered:
                time_since_trigger = (datetime.now() - rule.last_triggered).total_seconds()
                if time_since_trigger < rule.cooldown_period:
                    continue

            try:
                if rule.condition(current_metrics):
                    # Rule triggered
                    alert = SafetyAlert(
                        alert_id=f"alert_{rule.rule_id}_{int(datetime.now().timestamp())}",
                        rule_id=rule.rule_id,
                        message=f"Safety rule triggered: {rule.name} - {rule.description}",
                        level=rule.alert_level,
                        triggered_at=datetime.now(),
                        details={
                            "rule": rule.__dict__,
                            "metrics": current_metrics
                        }
                    )

                    alerts.append(alert)
                    self.active_alerts[alert.alert_id] = alert
                    self.alert_history.append(alert)

                    # Execute safety action
                    await self._execute_safety_action(rule.action, alert)

                    # Update last triggered
                    rule.last_triggered = datetime.now()

                    await self.audit_logger.log_event(
                        "safety_rule_triggered",
                        {
                            "rule_id": rule.rule_id,
                            "alert_id": alert.alert_id,
                            "action": rule.action.value,
                            "level": rule.alert_level.value
                        },
                        rule.alert_level.value
                    )

            except Exception as e:
                self.logger.error(f"Error checking rule {rule.rule_id}: {e}")

        return alerts

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        return {
            "portfolio_value": self.risk_metrics.portfolio_value,
            "daily_pnl": self.risk_metrics.daily_pnl,
            "total_pnl": self.risk_metrics.total_pnl,
            "max_drawdown": self.risk_metrics.max_drawdown,
            "leverage_ratio": self.risk_metrics.leverage_ratio,
            "concentration_risk": self.risk_metrics.concentration_risk,
            "volatility_risk": self.risk_metrics.volatility_risk,
            "liquidity_risk": self.risk_metrics.liquidity_risk,
            "trades_today": self.daily_trade_count,
            "trades_this_hour": self.hourly_trade_count,
            "largest_position": getattr(self.risk_metrics, 'largest_position', 0),
            "system_health": self._check_system_health(),
            "last_updated": datetime.now().isoformat(),
        }

    async def _execute_safety_action(self, action: SafetyAction, alert: SafetyAlert):
        """Execute a safety action"""
        if action == SafetyAction.NONE:
            return

        alert.action_taken = action

        if action == SafetyAction.REDUCE_POSITION:
            self.logger.warning("REDUCING POSITIONS due to safety rule trigger")
            # Implementation would reduce positions here

        elif action == SafetyAction.CLOSE_POSITION:
            self.logger.warning("CLOSING POSITIONS due to safety rule trigger")
            # Implementation would close positions here

        elif action == SafetyAction.HALT_TRADING:
            self.logger.error("HALTING TRADING due to safety rule trigger")
            self.trading_halted = True

        elif action == SafetyAction.EMERGENCY_SHUTDOWN:
            self.logger.critical("EMERGENCY SHUTDOWN initiated due to safety rule trigger")
            self.emergency_shutdown = True
            await self._emergency_shutdown()

    async def _emergency_shutdown(self):
        """Execute emergency shutdown procedure"""
        self.logger.critical("Executing emergency shutdown...")

        # Close all positions
        # Cancel all orders
        # Save state
        # Notify administrators

        await self.audit_logger.log_event(
            "emergency_shutdown",
            {"reason": "safety_rule_trigger"},
            "critical"
        )

    def _check_system_health(self) -> float:
        """Check overall system health (0.0 to 1.0)"""
        health_score = 1.0

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 90:
                health_score -= 0.3
            elif cpu_percent > 70:
                health_score -= 0.1

            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                health_score -= 0.3
            elif memory.percent > 80:
                health_score -= 0.1

            # Disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                health_score -= 0.2
            elif disk.percent > 85:
                health_score -= 0.1

            # Network connectivity (simplified)
            # In production, check actual connectivity to exchanges

        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            health_score -= 0.2

        return max(0.0, health_score)

    async def _update_trade_counters(self):
        """Update trade frequency counters"""
        now = datetime.now()

        # Reset hourly counter if hour changed
        if (now - self.hour_start_time).total_seconds() >= 3600:
            self.hourly_trade_count = 0
            self.hour_start_time = now

        # Reset daily counter at midnight
        if now.date() != getattr(self, '_last_reset_date', now.date()):
            self.daily_trade_count = 0
            self._last_reset_date = now.date()

    async def _start_monitoring_loop(self):
        """Start continuous risk monitoring"""
        while self.monitoring_active:
            try:
                # Update risk metrics
                await self._update_risk_metrics()

                # Execute safety checks
                alerts = await self.execute_safety_check()

                if alerts:
                    self.logger.warning(f"Triggered {len(alerts)} safety alerts")

                # Log metrics periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    await self._log_risk_metrics()

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)

    async def _start_health_check_loop(self):
        """Start system health monitoring"""
        while self.monitoring_active:
            try:
                health_score = self._check_system_health()

                if health_score < 0.7:
                    self.logger.warning(f"System health degraded: {health_score:.2f}")

                    # Create health alert
                    alert = SafetyAlert(
                        alert_id=f"health_{int(datetime.now().timestamp())}",
                        rule_id="system_health",
                        message=f"System health degraded: {health_score:.2f}",
                        level=AlertLevel.WARNING if health_score > 0.5 else AlertLevel.CRITICAL,
                        triggered_at=datetime.now(),
                        details={"health_score": health_score}
                    )

                    self.active_alerts[alert.alert_id] = alert
                    self.alert_history.append(alert)

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)

    async def _update_risk_metrics(self):
        """Update current risk metrics"""
        try:
            # This would integrate with actual portfolio data
            # For now, using placeholder values
            self.risk_metrics.last_updated = datetime.now()

            # In production, calculate from actual positions:
            # - portfolio_value
            # - daily_pnl, total_pnl
            # - max_drawdown
            # - leverage_ratio
            # - concentration_risk
            # etc.

        except Exception as e:
            self.logger.error(f"Failed to update risk metrics: {e}")

    async def _log_risk_metrics(self):
        """Log current risk metrics"""
        metrics = self._get_current_metrics()

        await self.audit_logger.log_event(
            "risk_metrics_update",
            metrics,
            "info"
        )

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety system status"""
        return {
            "trading_halted": self.trading_halted,
            "emergency_shutdown": self.emergency_shutdown,
            "active_alerts": len(self.active_alerts),
            "total_alerts_today": len([a for a in self.alert_history
                                     if a.triggered_at.date() == datetime.now().date()]),
            "circuit_breakers": self.circuit_breakers,
            "risk_limits": self.risk_limits.__dict__,
            "current_metrics": self._get_current_metrics(),
        }

    def get_active_alerts(self) -> List[SafetyAlert]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())

    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_id]

            self.logger.info(f"Resolved alert: {alert_id}")

    def reset_trading_halt(self):
        """Reset trading halt (admin function)"""
        if not self.emergency_shutdown:
            self.trading_halted = False
            self.logger.info("Trading halt reset by administrator")

    async def manual_emergency_shutdown(self):
        """Manually trigger emergency shutdown"""
        self.logger.critical("Manual emergency shutdown initiated")
        self.emergency_shutdown = True
        await self._emergency_shutdown()


# Global live trading safeguards instance
live_trading_safeguards = LiveTradingSafeguards()


async def initialize_live_trading_safeguards():
    """Initialize the live trading safeguards system"""
    print("[SHIELD]️  Initializing Live Trading Safeguards...")

    # Initialize the safeguards instance
    await live_trading_safeguards.initialize()

    # Wait a moment for initialization
    await asyncio.sleep(1)

    print("[OK] Live trading safeguards initialized")
    status = live_trading_safeguards.get_safety_status()
    print(f"  Trading halted: {status['trading_halted']}")
    print(f"  Emergency shutdown: {status['emergency_shutdown']}")
    status = live_trading_safeguards.get_safety_status()
    print(f"  Trading halted: {status['trading_halted']}")
    print(f"  Emergency shutdown: {status['emergency_shutdown']}")
    print(f"  Active alerts: {status['active_alerts']}")
    print(f"  Risk limits loaded: {len(status['risk_limits'])} parameters")


if __name__ == "__main__":
    # Example usage
    asyncio.run(initialize_live_trading_safeguards())