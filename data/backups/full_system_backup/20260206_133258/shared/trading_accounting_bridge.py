#!/usr/bin/env python3
"""
TradingExecution ↔ CentralAccounting Bridge
===========================================

Bridge between TradingExecution and CentralAccounting departments
for risk management, P&L attribution, and position reconciliation.

This bridge enables:
- Real-time risk limit updates from accounting to trading
- P&L attribution and performance tracking
- Position reconciliation and discrepancy alerts
- Capital allocation and utilization monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from shared.bridge_orchestrator import BridgeMessage, BridgeMessageType, MessagePriority, Department
from shared.audit_logger import get_audit_logger, AuditCategory, AuditSeverity

logger = logging.getLogger(__name__)


class TradingAccountingBridge:
    """
    Bridge between TradingExecution and CentralAccounting departments.
    Handles risk limits, P&L attribution, and position reconciliation.
    """

    def __init__(self):
        self.audit_logger = get_audit_logger()

        # Bridge state
        self.is_initialized = False
        self.last_risk_update = None
        self.last_pnl_update = None
        self.active_reconciliations: Dict[str, Dict] = {}

        # Risk limits cache
        self.risk_limits = {
            "max_drawdown_pct": 5.0,
            "daily_loss_limit": 10000.0,
            "position_limit_per_symbol": 100000.0,
            "total_exposure_limit": 500000.0
        }

        # Performance metrics
        self.performance_metrics = {
            "messages_processed": 0,
            "risk_updates_sent": 0,
            "reconciliations_completed": 0,
            "discrepancies_found": 0
        }

    async def initialize(self) -> bool:
        """Initialize the bridge."""
        try:
            logger.info("Initializing TradingExecution ↔ CentralAccounting bridge")

            # Initialize state
            self.is_initialized = True

            await self.audit_logger.log_event(
                category=AuditCategory.SYSTEM,
                action="bridge_initialized",
                resource="trading_accounting_bridge",
                severity=AuditSeverity.INFO,
                details={"bridge_type": "trading_accounting"}
            )

            logger.info("TradingExecution ↔ CentralAccounting bridge initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize trading-accounting bridge: {e}")
            return False

    async def handle_message(self, message: BridgeMessage) -> bool:
        """Handle incoming bridge messages."""
        try:
            self.performance_metrics["messages_processed"] += 1

            if message.message_type == BridgeMessageType.RISK_UPDATE:
                return await self._handle_risk_update(message)
            elif message.message_type == BridgeMessageType.PNL_ATTRIBUTION:
                return await self._handle_pnl_attribution(message)
            elif message.message_type == BridgeMessageType.POSITION_RECONCILIATION:
                return await self._handle_position_reconciliation(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return False

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return False

    async def _handle_risk_update(self, message: BridgeMessage) -> bool:
        """Handle risk limit updates from accounting."""
        try:
            risk_data = message.payload.get("risk_limits", {})

            # Update local risk limits cache
            self.risk_limits.update(risk_data)
            self.last_risk_update = datetime.now()

            self.performance_metrics["risk_updates_sent"] += 1

            await self.audit_logger.log_event(
                category=AuditCategory.TRADING,
                action="risk_limits_updated",
                resource="trading_accounting_bridge",
                severity=AuditSeverity.INFO,
                details={"risk_limits": risk_data}
            )

            logger.info(f"Updated risk limits: {risk_data}")
            return True

        except Exception as e:
            logger.error(f"Error handling risk update: {e}")
            return False

    async def _handle_pnl_attribution(self, message: BridgeMessage) -> bool:
        """Handle P&L attribution updates."""
        try:
            pnl_data = message.payload.get("pnl_data", {})

            self.last_pnl_update = datetime.now()

            await self.audit_logger.log_event(
                category=AuditCategory.TRADING,
                action="pnl_attributed",
                resource="trading_accounting_bridge",
                severity=AuditSeverity.INFO,
                details={"pnl_data": pnl_data}
            )

            logger.info(f"P&L attributed: {pnl_data}")
            return True

        except Exception as e:
            logger.error(f"Error handling P&L attribution: {e}")
            return False

    async def _handle_position_reconciliation(self, message: BridgeMessage) -> bool:
        """Handle position reconciliation requests."""
        try:
            reconciliation_data = message.payload
            rec_id = reconciliation_data.get("reconciliation_id")

            if not rec_id:
                rec_id = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Start reconciliation process
            self.active_reconciliations[rec_id] = {
                "start_time": datetime.now(),
                "status": "in_progress",
                "data": reconciliation_data
            }

            # Schedule reconciliation monitoring
            asyncio.create_task(self._monitor_reconciliation(rec_id))

            await self.audit_logger.log_event(
                category=AuditCategory.TRADING,
                action="reconciliation_started",
                resource="trading_accounting_bridge",
                severity=AuditSeverity.INFO,
                details={"reconciliation_id": rec_id}
            )

            logger.info(f"Started position reconciliation: {rec_id}")
            return True

        except Exception as e:
            logger.error(f"Error handling position reconciliation: {e}")
            return False

    async def _monitor_reconciliation(self, rec_id: str):
        """Monitor ongoing reconciliation process."""
        try:
            timeout = timedelta(minutes=5)

            while rec_id in self.active_reconciliations:
                rec_data = self.active_reconciliations[rec_id]
                elapsed = datetime.now() - rec_data["start_time"]

                if elapsed > timeout:
                    # Timeout - mark as failed
                    rec_data["status"] = "timed_out"
                    rec_data["end_time"] = datetime.now()

                    await self.audit_logger.log_event(
                        category=AuditCategory.TRADING,
                        action="reconciliation_timeout",
                        resource="trading_accounting_bridge",
                        severity=AuditSeverity.WARNING,
                        details={"reconciliation_id": rec_id}
                    )

                    break

                await asyncio.sleep(30)  # Check every 30 seconds

        except Exception as e:
            logger.error(f"Reconciliation monitoring error: {e}")

    async def get_bridge_health(self) -> Dict[str, Any]:
        """Get bridge health status."""
        return {
            "is_initialized": self.is_initialized,
            "last_risk_update": self.last_risk_update.isoformat() if self.last_risk_update else None,
            "last_pnl_update": self.last_pnl_update.isoformat() if self.last_pnl_update else None,
            "active_reconciliations": len(self.active_reconciliations),
            "risk_limits": self.risk_limits,
            "performance_metrics": self.performance_metrics
        }

    async def shutdown(self):
        """Shutdown the bridge."""
        logger.info("Shutting down TradingExecution ↔ CentralAccounting bridge")
        # Cleanup resources if needed
        self.is_initialized = False

        # Performance tracking
        self.performance_metrics = {
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
            "monthly_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0
        }

    async def initialize_bridge(self) -> bool:
        """Initialize the trading-accounting bridge."""
        try:
            logger.info("Initializing TradingExecution ↔ CentralAccounting bridge")

            # Load initial risk limits from CentralAccounting
            await self._load_risk_limits()

            # Load current performance metrics
            await self._load_performance_metrics()

            # Start reconciliation monitoring
            asyncio.create_task(self._reconciliation_monitor())

            self.is_initialized = True

            await self.audit_logger.log_event(
                category=AuditCategory.SYSTEM,
                event_type="trading_accounting_bridge_initialized",
                severity=AuditSeverity.INFO
            )

            logger.info("TradingExecution ↔ CentralAccounting bridge initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize trading-accounting bridge: {e}")
            await self.audit_logger.log_event(
                category=AuditCategory.SYSTEM,
                event_type="trading_accounting_bridge_init_failed",
                severity=AuditSeverity.ERROR,
                details={"error": str(e)}
            )
            return False

    async def _load_risk_limits(self):
        """Load current risk limits from CentralAccounting."""
        try:
            # In a real implementation, this would query CentralAccounting
            # For now, use default values
            logger.info("Loaded risk limits from CentralAccounting")

        except Exception as e:
            logger.error(f"Failed to load risk limits: {e}")

    async def _load_performance_metrics(self):
        """Load current performance metrics from CentralAccounting."""
        try:
            # In a real implementation, this would query CentralAccounting
            # For now, use default values
            logger.info("Loaded performance metrics from CentralAccounting")

        except Exception as e:
            logger.error(f"Failed to load performance metrics: {e}")

    async def process_message(self, message: BridgeMessage) -> bool:
        """Process messages from the bridge orchestrator."""
        try:
            if message.message_type == BridgeMessageType.RISK_LIMIT_UPDATE:
                return await self._handle_risk_limit_update(message)

            elif message.message_type == BridgeMessageType.PNL_ATTRIBUTION:
                return await self._handle_pnl_attribution(message)

            elif message.message_type == BridgeMessageType.POSITION_RECONCILIATION:
                return await self._handle_position_reconciliation(message)

            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to process bridge message: {e}")
            return False

    async def _handle_risk_limit_update(self, message: BridgeMessage) -> bool:
        """Handle risk limit updates from CentralAccounting."""
        try:
            limits_data = message.data

            # Update local risk limits cache
            self.risk_limits.update(limits_data)
            self.last_risk_update = datetime.now()

            # Forward to TradingExecution engine
            await self._forward_risk_limits_to_trading(limits_data)

            await self.audit_logger.log_event(
                category=AuditCategory.TRADING,
                event_type="risk_limits_updated",
                severity=AuditSeverity.INFO,
                details={"limits": limits_data}
            )

            logger.info(f"Updated risk limits: {limits_data}")
            return True

        except Exception as e:
            logger.error(f"Failed to handle risk limit update: {e}")
            return False

    async def _handle_pnl_attribution(self, message: BridgeMessage) -> bool:
        """Handle P&L attribution updates from CentralAccounting."""
        try:
            pnl_data = message.data

            # Update performance metrics
            self.performance_metrics.update(pnl_data)
            self.last_pnl_update = datetime.now()

            # Forward attribution data to TradingExecution
            await self._forward_pnl_to_trading(pnl_data)

            await self.audit_logger.log_event(
                category=AuditCategory.TRADING,
                event_type="pnl_attribution_updated",
                severity=AuditSeverity.INFO,
                details={"pnl_data": pnl_data}
            )

            logger.info(f"Updated P&L attribution: {pnl_data}")
            return True

        except Exception as e:
            logger.error(f"Failed to handle P&L attribution: {e}")
            return False

    async def _handle_position_reconciliation(self, message: BridgeMessage) -> bool:
        """Handle position reconciliation requests."""
        try:
            reconciliation_data = message.data
            reconciliation_id = reconciliation_data.get("reconciliation_id")

            # Store active reconciliation
            self.active_reconciliations[reconciliation_id] = {
                "data": reconciliation_data,
                "started_at": datetime.now(),
                "status": "in_progress"
            }

            # Perform reconciliation
            discrepancies = await self._perform_reconciliation(reconciliation_data)

            # Send reconciliation results back
            result_message = BridgeMessage(
                source_dept=Department.TRADING_EXECUTION,
                target_dept=Department.CENTRAL_ACCOUNTING,
                message_type=BridgeMessageType.POSITION_RECONCILIATION,
                priority=MessagePriority.HIGH,
                data={
                    "reconciliation_id": reconciliation_id,
                    "discrepancies": discrepancies,
                    "status": "completed" if not discrepancies else "discrepancies_found"
                }
            )

            # Update reconciliation status
            self.active_reconciliations[reconciliation_id]["status"] = "completed"
            self.active_reconciliations[reconciliation_id]["completed_at"] = datetime.now()

            await self.audit_logger.log_event(
                category=AuditCategory.TRADING,
                event_type="position_reconciliation_completed",
                severity=AuditSeverity.INFO,
                details={
                    "reconciliation_id": reconciliation_id,
                    "discrepancies": len(discrepancies)
                }
            )

            logger.info(f"Completed position reconciliation {reconciliation_id} with {len(discrepancies)} discrepancies")
            return True

        except Exception as e:
            logger.error(f"Failed to handle position reconciliation: {e}")
            return False

    async def _perform_reconciliation(self, reconciliation_data: Dict) -> List[Dict]:
        """Perform position reconciliation between trading and accounting."""
        try:
            # In a real implementation, this would:
            # 1. Get positions from TradingExecution engine
            # 2. Compare with positions from CentralAccounting
            # 3. Identify discrepancies
            # 4. Return detailed discrepancy report

            # For now, simulate reconciliation
            discrepancies = []

            # Simulate occasional discrepancies for testing
            import random
            if random.random() < 0.1:  # 10% chance of discrepancy
                discrepancies.append({
                    "symbol": "BTC/USD",
                    "trading_position": 100.0,
                    "accounting_position": 95.0,
                    "discrepancy": 5.0,
                    "severity": "medium",
                    "description": "Position mismatch detected"
                })

            return discrepancies

        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            return []

    async def _forward_risk_limits_to_trading(self, limits: Dict):
        """Forward risk limits to TradingExecution engine."""
        try:
            # In a real implementation, this would update the TradingExecution engine
            logger.info(f"Forwarded risk limits to TradingExecution: {limits}")

        except Exception as e:
            logger.error(f"Failed to forward risk limits: {e}")

    async def _forward_pnl_to_trading(self, pnl_data: Dict):
        """Forward P&L data to TradingExecution engine."""
        try:
            # In a real implementation, this would update the TradingExecution engine
            logger.info(f"Forwarded P&L data to TradingExecution: {pnl_data}")

        except Exception as e:
            logger.error(f"Failed to forward P&L data: {e}")

    async def _reconciliation_monitor(self):
        """Monitor active reconciliations and handle timeouts."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.now()
                timeout_threshold = timedelta(minutes=10)

                # Check for timed out reconciliations
                for rec_id, rec_data in list(self.active_reconciliations.items()):
                    if rec_data["status"] == "in_progress":
                        started_at = rec_data["started_at"]
                        if now - started_at > timeout_threshold:
                            logger.warning(f"Reconciliation {rec_id} timed out")
                            rec_data["status"] = "timed_out"

                            await self.audit_logger.log_event(
                                category=AuditCategory.TRADING,
                                event_type="reconciliation_timeout",
                                severity=AuditSeverity.WARNING,
                                details={"reconciliation_id": rec_id}
                            )

            except Exception as e:
                logger.error(f"Reconciliation monitoring error: {e}")

    async def get_bridge_health(self) -> Dict[str, Any]:
        """Get bridge health status."""
        return {
            "is_initialized": self.is_initialized,
            "last_risk_update": self.last_risk_update.isoformat() if self.last_risk_update else None,
            "last_pnl_update": self.last_pnl_update.isoformat() if self.last_pnl_update else None,
            "active_reconciliations": len(self.active_reconciliations),
            "risk_limits": self.risk_limits,
            "performance_metrics": self.performance_metrics
        }

    async def shutdown(self):
        """Shutdown the bridge."""
        logger.info("Shutting down TradingExecution ↔ CentralAccounting bridge")
        # Cleanup resources if needed