#!/usr/bin/env python3
"""
ACC Cross-Department Bridge System
===================================

Unified bridge orchestrator managing all cross-department communications
for the Accelerated Arbitrage Corp doctrine system.

Departments:
- TradingExecution (TE)
- BigBrainIntelligence (BBI)
- CentralAccounting (CA)
- CryptoIntelligence (CI)
- SharedInfrastructure (SI)

Bridge Types:
- TE ↔ BBI: Strategy signals, execution feedback
- TE ↔ CA: Risk limits, P&L attribution, position reconciliation
- TE ↔ CI: Venue selection, execution routing, withdrawal coordination
- TE ↔ SI: Execution monitoring, incident reporting, audit logging
- BBI ↔ CA: Research insights, risk model validation, strategy attribution
- BBI ↔ SI: Research monitoring, gap analysis, intelligence synthesis
- CA ↔ CI: Counterparty risk assessment, exposure monitoring
- CA ↔ SI: Audit compliance, financial reporting, risk aggregation
- CI ↔ SI: Security monitoring, venue health tracking, incident response
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json

from shared.audit_logger import get_audit_logger, AuditCategory, AuditSeverity

logger = logging.getLogger(__name__)


class Department(Enum):
    """ACC Department enumeration."""
    TRADING_EXECUTION = "TE"
    BIGBRAIN_INTELLIGENCE = "BBI"
    CENTRAL_ACCOUNTING = "CA"
    CRYPTO_INTELLIGENCE = "CI"
    SHARED_INFRASTRUCTURE = "SI"


class BridgeMessageType(Enum):
    """Types of messages that can be sent across bridges."""
    # Trading ↔ Intelligence
    STRATEGY_SIGNAL = "strategy_signal"
    EXECUTION_FEEDBACK = "execution_feedback"

    # Trading ↔ Accounting
    RISK_LIMIT_UPDATE = "risk_limit_update"
    PNL_ATTRIBUTION = "pnl_attribution"
    POSITION_RECONCILIATION = "position_reconciliation"

    # Trading ↔ Crypto
    VENUE_SELECTION = "venue_selection"
    EXECUTION_ROUTING = "execution_routing"
    WITHDRAWAL_COORDINATION = "withdrawal_coordination"

    # Trading ↔ Infrastructure
    EXECUTION_MONITORING = "execution_monitoring"
    INCIDENT_REPORTING = "incident_reporting"
    AUDIT_LOGGING = "audit_logging"

    # Intelligence ↔ Accounting
    RESEARCH_INSIGHT = "research_insight"
    RISK_MODEL_VALIDATION = "risk_model_validation"
    STRATEGY_ATTRIBUTION = "strategy_attribution"

    # Intelligence ↔ Infrastructure
    RESEARCH_MONITORING = "research_monitoring"
    GAP_ANALYSIS = "gap_analysis"
    INTELLIGENCE_SYNTHESIS = "intelligence_synthesis"

    # Accounting ↔ Crypto
    COUNTERPARTY_RISK = "counterparty_risk"
    EXPOSURE_MONITORING = "exposure_monitoring"

    # Accounting ↔ Infrastructure
    AUDIT_COMPLIANCE = "audit_compliance"
    FINANCIAL_REPORTING = "financial_reporting"
    RISK_AGGREGATION = "risk_aggregation"

    # Crypto ↔ Infrastructure
    SECURITY_MONITORING = "security_monitoring"
    VENUE_HEALTH_TRACKING = "venue_health_tracking"
    INCIDENT_RESPONSE = "incident_response"


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BridgeMessage:
    """Unified message format for cross-department communication."""
    source_dept: Department
    target_dept: Department
    message_type: BridgeMessageType
    priority: MessagePriority
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = None
    ttl_seconds: int = 300  # 5 minutes default TTL

    def __post_init__(self):
        if self.correlation_id is None:
            self.correlation_id = f"{self.source_dept.value}_{self.target_dept.value}_{int(self.timestamp.timestamp())}"

    @property
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl_seconds


@dataclass
class BridgeConnection:
    """Represents a connection between two departments."""
    dept_a: Department
    dept_b: Department
    is_active: bool = False
    last_message: Optional[datetime] = None
    message_count: int = 0
    error_count: int = 0
    health_score: float = 1.0

    @property
    def connection_key(self) -> str:
        """Get unique key for this connection."""
        depts = sorted([self.dept_a.value, self.dept_b.value])
        return f"{depts[0]}_{depts[1]}"


class BridgeOrchestrator:
    """
    Central orchestrator for all cross-department bridge communications.
    Manages message routing, connection health, and bridge lifecycle.
    """

    def __init__(self):
        self.audit_logger = get_audit_logger()

        # Bridge connections between all department pairs
        self.connections: Dict[str, BridgeConnection] = {}
        self._initialize_connections()

        # Message queues and routing
        self.message_queues: Dict[Department, asyncio.Queue] = {
            dept: asyncio.Queue() for dept in Department
        }

        # Bridge handlers for specific department pairs
        self.bridge_handlers: Dict[str, Any] = {}

        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.last_health_check = datetime.now()

        # Message processing
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False

    def _initialize_connections(self):
        """Initialize all possible department-to-department connections."""
        departments = list(Department)

        for i, dept_a in enumerate(departments):
            for dept_b in departments[i+1:]:  # Avoid duplicate connections
                connection = BridgeConnection(dept_a, dept_b)
                self.connections[connection.connection_key] = connection

    async def initialize(self) -> bool:
        """Initialize the bridge orchestrator."""
        try:
            logger.info("Initializing ACC Bridge Orchestrator...")

            # Initialize specific bridge handlers
            await self._initialize_bridge_handlers()

            # Start message processing
            self.processing_task = asyncio.create_task(self._process_messages())
            self.is_running = True

            # Start health monitoring
            asyncio.create_task(self._health_monitoring_loop())

            await self.audit_logger.log_event(
                category=AuditCategory.SYSTEM,
                action="bridge_orchestrator_initialized",
                resource="bridge_orchestrator",
                severity=AuditSeverity.INFO,
                details={"connections": len(self.connections)}
            )

            logger.info(f"Bridge Orchestrator initialized with {len(self.connections)} connections")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize bridge orchestrator: {e}")
            await self.audit_logger.log_event(
                category=AuditCategory.SYSTEM,
                action="bridge_orchestrator_init_failed",
                resource="bridge_orchestrator",
                severity=AuditSeverity.ERROR,
                details={"error": str(e)}
            )
            return False

    async def _initialize_bridge_handlers(self):
        """Initialize specific bridge handlers for department pairs."""
        # Import and initialize specific bridge implementations
        try:
            # CryptoIntelligence ↔ BigBrainIntelligence bridge
            from shared.crypto_bigbrain_bridge import CryptoBigBrainBridge
            crypto_bigbrain_bridge = CryptoBigBrainBridge()
            await crypto_bigbrain_bridge.initialize_bridge()
            self.bridge_handlers["BBI_CI"] = crypto_bigbrain_bridge
            self.connections["BBI_CI"].is_active = True

            logger.info("Initialized CryptoIntelligence ↔ BigBrainIntelligence bridge")

        except Exception as e:
            logger.warning(f"Failed to initialize crypto-bigbrain bridge: {e}")

        try:
            # TradingExecution ↔ CentralAccounting bridge
            from shared.trading_accounting_bridge import TradingAccountingBridge
            trading_accounting_bridge = TradingAccountingBridge()
            await trading_accounting_bridge.initialize_bridge()
            self.bridge_handlers["TE_CA"] = trading_accounting_bridge
            self.connections["TE_CA"].is_active = True

            logger.info("Initialized TradingExecution ↔ CentralAccounting bridge")

        except Exception as e:
            logger.warning(f"Failed to initialize trading-accounting bridge: {e}")

        try:
            # TradingExecution ↔ BigBrainIntelligence bridge
            from shared.trading_intelligence_bridge import TradingIntelligenceBridge
            trading_intelligence_bridge = TradingIntelligenceBridge()
            await trading_intelligence_bridge.initialize_bridge()
            self.bridge_handlers["TE_BBI"] = trading_intelligence_bridge
            self.connections["TE_BBI"].is_active = True

            logger.info("Initialized TradingExecution ↔ BigBrainIntelligence bridge")

        except Exception as e:
            logger.warning(f"Failed to initialize trading-intelligence bridge: {e}")

        try:
            # TradingExecution ↔ CryptoIntelligence bridge
            from shared.trading_crypto_bridge import TradingCryptoBridge
            trading_crypto_bridge = TradingCryptoBridge()
            await trading_crypto_bridge.initialize()
            self.bridge_handlers["TE_CI"] = trading_crypto_bridge
            self.connections["TE_CI"].is_active = True

            logger.info("Initialized TradingExecution ↔ CryptoIntelligence bridge")

        except Exception as e:
            logger.warning(f"Failed to initialize trading-crypto bridge: {e}")

        try:
            # TradingExecution ↔ SharedInfrastructure bridge
            from shared.trading_infrastructure_bridge import TradingInfrastructureBridge
            trading_infrastructure_bridge = TradingInfrastructureBridge()
            await trading_infrastructure_bridge.initialize()
            self.bridge_handlers["TE_SI"] = trading_infrastructure_bridge
            self.connections["TE_SI"].is_active = True

            logger.info("Initialized TradingExecution ↔ SharedInfrastructure bridge")

        except Exception as e:
            logger.warning(f"Failed to initialize trading-infrastructure bridge: {e}")

        try:
            # BigBrainIntelligence ↔ CentralAccounting bridge
            from shared.intelligence_accounting_bridge import IntelligenceAccountingBridge
            intelligence_accounting_bridge = IntelligenceAccountingBridge()
            await intelligence_accounting_bridge.initialize()
            self.bridge_handlers["BBI_CA"] = intelligence_accounting_bridge
            self.connections["BBI_CA"].is_active = True

            logger.info("Initialized BigBrainIntelligence ↔ CentralAccounting bridge")

        except Exception as e:
            logger.warning(f"Failed to initialize intelligence-accounting bridge: {e}")

        try:
            # BigBrainIntelligence ↔ SharedInfrastructure bridge
            from shared.intelligence_infrastructure_bridge import IntelligenceInfrastructureBridge
            intelligence_infrastructure_bridge = IntelligenceInfrastructureBridge()
            await intelligence_infrastructure_bridge.initialize()
            self.bridge_handlers["BBI_SI"] = intelligence_infrastructure_bridge
            self.connections["BBI_SI"].is_active = True

            logger.info("Initialized BigBrainIntelligence ↔ SharedInfrastructure bridge")

        except Exception as e:
            logger.warning(f"Failed to initialize intelligence-infrastructure bridge: {e}")

        try:
            # CentralAccounting ↔ CryptoIntelligence bridge
            from shared.accounting_crypto_bridge import AccountingCryptoBridge
            accounting_crypto_bridge = AccountingCryptoBridge()
            await accounting_crypto_bridge.initialize()
            self.bridge_handlers["CA_CI"] = accounting_crypto_bridge
            self.connections["CA_CI"].is_active = True

            logger.info("Initialized CentralAccounting ↔ CryptoIntelligence bridge")

        except Exception as e:
            logger.warning(f"Failed to initialize accounting-crypto bridge: {e}")

        try:
            # CentralAccounting ↔ SharedInfrastructure bridge
            from shared.accounting_infrastructure_bridge import AccountingInfrastructureBridge
            accounting_infrastructure_bridge = AccountingInfrastructureBridge()
            await accounting_infrastructure_bridge.initialize()
            self.bridge_handlers["CA_SI"] = accounting_infrastructure_bridge
            self.connections["CA_SI"].is_active = True

            logger.info("Initialized CentralAccounting ↔ SharedInfrastructure bridge")

        except Exception as e:
            logger.warning(f"Failed to initialize accounting-infrastructure bridge: {e}")

        try:
            # CryptoIntelligence ↔ SharedInfrastructure bridge
            from shared.crypto_infrastructure_bridge import CryptoInfrastructureBridge
            crypto_infrastructure_bridge = CryptoInfrastructureBridge()
            await crypto_infrastructure_bridge.initialize()
            self.bridge_handlers["CI_SI"] = crypto_infrastructure_bridge
            self.connections["CI_SI"].is_active = True

            logger.info("Initialized CryptoIntelligence ↔ SharedInfrastructure bridge")

        except Exception as e:
            logger.warning(f"Failed to initialize crypto-infrastructure bridge: {e}")

    async def send_message(self, message: BridgeMessage) -> bool:
        """Send a message across the bridge system."""
        try:
            # Validate message
            if message.is_expired:
                logger.warning(f"Discarding expired message: {message.correlation_id}")
                return False

            # Get connection key
            connection_key = BridgeConnection(message.source_dept, message.target_dept).connection_key

            # Check if connection exists and is active
            if connection_key not in self.connections:
                logger.error(f"No connection found for {connection_key}")
                return False

            connection = self.connections[connection_key]
            if not connection.is_active:
                logger.warning(f"Connection {connection_key} is not active")
                return False

            # Route message to target department's queue
            await self.message_queues[message.target_dept].put(message)

            # Update connection stats
            connection.last_message = datetime.now()
            connection.message_count += 1

            # Audit the message
            await self.audit_logger.log_event(
                category=AuditCategory.SYSTEM,
                event_type="bridge_message_sent",
                severity=AuditSeverity.INFO,
                details={
                    "correlation_id": message.correlation_id,
                    "source": message.source_dept.value,
                    "target": message.target_dept.value,
                    "type": message.message_type.value,
                    "priority": message.priority.value
                }
            )

            logger.debug(f"Message sent: {message.source_dept.value} → {message.target_dept.value} ({message.message_type.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to send bridge message: {e}")
            # Update error count
            connection_key = BridgeConnection(message.source_dept, message.target_dept).connection_key
            if connection_key in self.connections:
                self.connections[connection_key].error_count += 1

            await self.audit_logger.log_event(
                category=AuditCategory.SYSTEM,
                event_type="bridge_message_send_failed",
                severity=AuditSeverity.ERROR,
                details={
                    "correlation_id": message.correlation_id,
                    "error": str(e)
                }
            )
            return False

    async def _process_messages(self):
        """Process messages from all department queues."""
        while self.is_running:
            try:
                # Process messages from all queues concurrently
                tasks = []
                for dept in Department:
                    if not self.message_queues[dept].empty():
                        tasks.append(self._process_department_messages(dept))

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await asyncio.sleep(1)

    async def _process_department_messages(self, department: Department):
        """Process messages for a specific department."""
        try:
            while not self.message_queues[department].empty():
                message = self.message_queues[department].get_nowait()

                # Route to appropriate handler based on source and message type
                await self._route_message(message)

        except Exception as e:
            logger.error(f"Error processing messages for {department.value}: {e}")

    async def _route_message(self, message: BridgeMessage):
        """Route message to appropriate handler."""
        try:
            # Get connection key for handler lookup
            connection_key = BridgeConnection(message.source_dept, message.target_dept).connection_key

            # Check if we have a specific handler for this connection
            if connection_key in self.bridge_handlers:
                handler = self.bridge_handlers[connection_key]
                await handler.process_message(message)
            else:
                # Generic message processing
                await self._process_generic_message(message)

        except Exception as e:
            logger.error(f"Failed to route message {message.correlation_id}: {e}")

    async def _process_generic_message(self, message: BridgeMessage):
        """Process messages that don't have specific handlers."""
        # For now, just log the message and acknowledge receipt
        logger.info(f"Processed generic message: {message.source_dept.value} → {message.target_dept.value} ({message.message_type.value})")

        # In a full implementation, this would:
        # - Validate message content
        # - Apply business logic rules
        # - Forward to appropriate internal systems
        # - Generate responses if needed

    async def _health_monitoring_loop(self):
        """Monitor bridge health and connection status."""
        while self.is_running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    async def _perform_health_checks(self):
        """Perform health checks on all connections."""
        self.last_health_check = datetime.now()

        for connection in self.connections.values():
            try:
                # Check connection health based on recent activity and error rates
                time_since_last_message = (datetime.now() - (connection.last_message or datetime.min)).total_seconds()

                # Health score based on activity and errors
                activity_score = min(1.0, 300 / max(time_since_last_message, 1))  # Prefer recent activity
                error_score = max(0.0, 1.0 - (connection.error_count / max(connection.message_count + 1, 10)))

                connection.health_score = (activity_score + error_score) / 2

                # Mark as inactive if health is too low
                connection.is_active = connection.health_score > 0.3

            except Exception as e:
                logger.error(f"Health check failed for {connection.connection_key}: {e}")
                connection.health_score = 0.0
                connection.is_active = False

    async def get_bridge_status(self) -> Dict[str, Any]:
        """Get comprehensive bridge system status."""
        return {
            "timestamp": datetime.now().isoformat(),
            "is_running": self.is_running,
            "connections": {
                key: {
                    "departments": f"{conn.dept_a.value} ↔ {conn.dept_b.value}",
                    "is_active": conn.is_active,
                    "health_score": conn.health_score,
                    "message_count": conn.message_count,
                    "error_count": conn.error_count,
                    "last_message": conn.last_message.isoformat() if conn.last_message else None
                }
                for key, conn in self.connections.items()
            },
            "handlers": list(self.bridge_handlers.keys()),
            "last_health_check": self.last_health_check.isoformat(),
            "queue_sizes": {
                dept.value: self.message_queues[dept].qsize() for dept in Department
            }
        }

    async def shutdown(self):
        """Shutdown the bridge orchestrator."""
        logger.info("Shutting down Bridge Orchestrator...")
        self.is_running = False

        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        # Shutdown bridge handlers
        for handler in self.bridge_handlers.values():
            if hasattr(handler, 'shutdown'):
                await handler.shutdown()

        await self.audit_logger.log_event(
            category=AuditCategory.SYSTEM,
            action="bridge_orchestrator_shutdown",
            resource="bridge_orchestrator",
            severity=AuditSeverity.INFO
        )


# Global bridge orchestrator instance
bridge_orchestrator = BridgeOrchestrator()

async def get_bridge_orchestrator():
    """Get the global bridge orchestrator instance."""
    return bridge_orchestrator