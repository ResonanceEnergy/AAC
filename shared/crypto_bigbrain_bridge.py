#!/usr/bin/env python3
"""
CryptoIntelligence ↔ BigBrainIntelligence Bridge
===============================================

Cross-department bridge enabling information flow between:
- CryptoIntelligence: Venue health, counterparty data, withdrawal monitoring
- BigBrainIntelligence: Research analysis, gap detection, intelligence synthesis

This bridge enables:
- Venue health data feeding into research analysis
- Counterparty intelligence informing gap detection
- Withdrawal risk signals triggering research alerts
- Cross-validation of intelligence sources
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from shared.audit_logger import get_audit_logger, AuditCategory, AuditSeverity

logger = logging.getLogger(__name__)


@dataclass
class IntelligenceBridgeMessage:
    """Message format for cross-department intelligence sharing."""
    source_dept: str
    target_dept: str
    message_type: str  # 'venue_health', 'counterparty_alert', 'withdrawal_risk', 'research_insight'
    priority: str  # 'low', 'medium', 'high', 'critical'
    data: Dict[str, Any]
    timestamp: datetime = None
    correlation_id: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.correlation_id is None:
            self.correlation_id = f"{self.source_dept}_{self.target_dept}_{int(self.timestamp.timestamp())}"


class CryptoBigBrainBridge:
    """
    Bridge between CryptoIntelligence and BigBrainIntelligence departments.

    Handles bidirectional information flow:
    - CryptoIntel → BigBrain: Venue data, counterparty signals, withdrawal risks
    - BigBrain → CryptoIntel: Research insights, gap analysis, intelligence synthesis
    """

    def __init__(self):
        self.audit = get_audit_logger()
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.active_connections = {
            'crypto_intel': False,
            'big_brain': False
        }
        self.bridge_metrics = {
            'messages_processed': 0,
            'alerts_generated': 0,
            'insights_shared': 0,
            'last_bridge_check': None
        }

    async def initialize_bridge(self) -> bool:
        """Initialize the cross-department bridge."""
        try:
            logger.info("Initializing CryptoIntelligence ↔ BigBrainIntelligence bridge")

            # Test connections to both departments
            crypto_connected = await self._test_crypto_connection()
            bigbrain_connected = await self._test_bigbrain_connection()

            self.active_connections['crypto_intel'] = crypto_connected
            self.active_connections['big_brain'] = bigbrain_connected

            if crypto_connected and bigbrain_connected:
                logger.info("Bridge initialized successfully")
                await self._audit_bridge_event("bridge_initialized", "success")
                return True
            else:
                logger.warning("Bridge partially initialized - some connections failed")
                await self._audit_bridge_event("bridge_partial_init", "warning")
                return False

        except Exception as e:
            logger.error(f"Bridge initialization failed: {e}")
            await self._audit_bridge_event("bridge_init_failed", "error", error=str(e))
            return False

    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the bridge."""
        try:
            # Test connections
            crypto_connected = await self._test_crypto_connection()
            bigbrain_connected = await self._test_bigbrain_connection()

            # Update connection status
            self.active_connections['crypto_intel'] = crypto_connected
            self.active_connections['big_brain'] = bigbrain_connected

            connections_active = sum(self.active_connections.values())

            return {
                "operational": connections_active == 2,
                "connections_active": connections_active,
                "crypto_intel_connected": crypto_connected,
                "big_brain_connected": bigbrain_connected,
                "messages_processed": self.bridge_metrics['messages_processed'],
                "alerts_generated": self.bridge_metrics['alerts_generated'],
                "insights_shared": self.bridge_metrics['insights_shared'],
                "last_bridge_check": self.bridge_metrics['last_bridge_check'].isoformat() if self.bridge_metrics['last_bridge_check'] else None
            }

        except Exception as e:
            logger.error(f"Failed to get bridge health status: {e}")
            return {
                "operational": False,
                "error": str(e),
                "connections_active": 0,
                "crypto_intel_connected": False,
                "big_brain_connected": False
            }

    async def _test_crypto_connection(self) -> bool:
        """Test connection to CryptoIntelligence department."""
        try:
            # Import and test CryptoIntelligence components
            from CryptoIntelligence.crypto_intelligence_engine import CryptoIntelligenceEngine
            engine = CryptoIntelligenceEngine()
            # Test basic functionality
            health = await engine.get_venue_health("test")
            return health is not None
        except Exception as e:
            logger.warning(f"CryptoIntelligence connection test failed: {e}")
            return False

    async def _test_bigbrain_connection(self) -> bool:
        """Test connection to BigBrainIntelligence department."""
        try:
            # Import and test BigBrainIntelligence components
            from BigBrainIntelligence.research_agent import ResearchAgent
            agent = ResearchAgent()
            # Test basic functionality
            status = await agent.get_status()
            return status is not None
        except Exception as e:
            logger.warning(f"BigBrainIntelligence connection test failed: {e}")
            return False

    async def send_message(self, message: IntelligenceBridgeMessage) -> bool:
        """Send a message across the bridge."""
        try:
            await self.message_queue.put(message)
            self.bridge_metrics['messages_processed'] += 1

            await self._audit_bridge_event(
                "message_sent",
                "info",
                message_type=message.message_type,
                priority=message.priority
            )

            logger.debug(f"Message sent: {message.message_type} ({message.priority})")
            return True

        except Exception as e:
            logger.error(f"Failed to send bridge message: {e}")
            return False

    async def process_messages(self) -> None:
        """Process messages in the bridge queue."""
        while True:
            try:
                message = await self.message_queue.get()

                # Route message based on target department
                if message.target_dept == "BigBrainIntelligence":
                    await self._route_to_bigbrain(message)
                elif message.target_dept == "CryptoIntelligence":
                    await self._route_to_crypto(message)
                else:
                    logger.warning(f"Unknown target department: {message.target_dept}")

                self.message_queue.task_done()

            except Exception as e:
                logger.error(f"Error processing bridge message: {e}")
                await asyncio.sleep(1)

    async def _route_to_bigbrain(self, message: IntelligenceBridgeMessage) -> None:
        """Route message to BigBrainIntelligence for analysis."""
        try:
            if not self.active_connections['big_brain']:
                logger.warning("BigBrain connection not active, queuing message")
                return

            # Import BigBrain components
            from BigBrainIntelligence.research_agent import ResearchAgent
            agent = ResearchAgent()

            # Process based on message type
            if message.message_type == "venue_health":
                await self._process_venue_health_for_research(message, agent)
            elif message.message_type == "counterparty_alert":
                await self._process_counterparty_alert_for_research(message, agent)
            elif message.message_type == "withdrawal_risk":
                await self._process_withdrawal_risk_for_research(message, agent)

            self.bridge_metrics['insights_shared'] += 1

        except Exception as e:
            logger.error(f"Failed to route message to BigBrain: {e}")

    async def _route_to_crypto(self, message: IntelligenceBridgeMessage) -> None:
        """Route message to CryptoIntelligence for action."""
        try:
            if not self.active_connections['crypto_intel']:
                logger.warning("CryptoIntel connection not active, queuing message")
                return

            # Import CryptoIntel components
            from CryptoIntelligence.crypto_intelligence_engine import CryptoIntelligenceEngine
            engine = CryptoIntelligenceEngine()

            # Process based on message type
            if message.message_type == "research_insight":
                await self._process_research_insight_for_crypto(message, engine)

            self.bridge_metrics['alerts_generated'] += 1

        except Exception as e:
            logger.error(f"Failed to route message to CryptoIntel: {e}")

    async def _process_venue_health_for_research(self, message: IntelligenceBridgeMessage, agent) -> None:
        """Process venue health data for research analysis."""
        venue_data = message.data

        # Create research finding from venue health data
        finding = {
            'type': 'venue_health_analysis',
            'venue': venue_data.get('venue'),
            'health_score': venue_data.get('health_score'),
            'risk_factors': venue_data.get('risk_factors', []),
            'timestamp': message.timestamp.isoformat()
        }

        await agent.analyze_venue_health(finding)

    async def _process_counterparty_alert_for_research(self, message: IntelligenceBridgeMessage, agent) -> None:
        """Process counterparty alerts for research analysis."""
        alert_data = message.data

        # Create research finding from counterparty data
        finding = {
            'type': 'counterparty_risk_analysis',
            'counterparty': alert_data.get('counterparty'),
            'risk_level': alert_data.get('risk_level'),
            'exposure_amount': alert_data.get('exposure'),
            'timestamp': message.timestamp.isoformat()
        }

        await agent.analyze_counterparty_risk(finding)

    async def _process_withdrawal_risk_for_research(self, message: IntelligenceBridgeMessage, agent) -> None:
        """Process withdrawal risk signals for research analysis."""
        risk_data = message.data

        # Create research finding from withdrawal risk
        finding = {
            'type': 'withdrawal_risk_analysis',
            'venue': risk_data.get('venue'),
            'risk_score': risk_data.get('risk_score'),
            'withdrawal_queue': risk_data.get('queue_length'),
            'timestamp': message.timestamp.isoformat()
        }

        await agent.analyze_withdrawal_risk(finding)

    async def _process_research_insight_for_crypto(self, message: IntelligenceBridgeMessage, engine) -> None:
        """Process research insights for crypto intelligence action."""
        insight_data = message.data

        # Route insight to appropriate crypto intelligence handler
        insight_type = insight_data.get('insight_type')

        if insight_type == "gap_opportunity":
            await engine.process_gap_opportunity(insight_data)
        elif insight_type == "venue_anomaly":
            await engine.process_venue_anomaly(insight_data)
        elif insight_type == "counterparty_shift":
            await engine.process_counterparty_shift(insight_data)

    async def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status and metrics."""
        status = {
            'connections': self.active_connections.copy(),
            'metrics': self.bridge_metrics.copy(),
            'queue_size': self.message_queue.qsize(),
            'timestamp': datetime.now().isoformat()
        }

        # Update last check time
        self.bridge_metrics['last_bridge_check'] = datetime.now()

        return status

    async def _audit_bridge_event(self, event_type: str, severity: str, **kwargs) -> None:
        """Audit bridge events."""
        await self.audit.log_event(
            category=AuditCategory.SYSTEM,
            action=event_type,
            resource="crypto_bigbrain_bridge",
            status="info" if severity == "info" else "success",
            severity=getattr(AuditSeverity, severity.upper()),
            user="system",
            details={
                'event_type': event_type,
                'bridge_type': 'crypto_bigbrain',
                'timestamp': datetime.now().isoformat(),
                **kwargs
            }
        )


# Global bridge instance
_bridge_instance: Optional[CryptoBigBrainBridge] = None


async def get_bridge() -> CryptoBigBrainBridge:
    """Get or create the global bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = CryptoBigBrainBridge()
        await _bridge_instance.initialize_bridge()
    return _bridge_instance


async def send_crypto_to_bigbrain(message_type: str, data: Dict[str, Any], priority: str = "medium") -> bool:
    """Convenience function to send messages from CryptoIntel to BigBrain."""
    bridge = await get_bridge()
    message = IntelligenceBridgeMessage(
        source_dept="CryptoIntelligence",
        target_dept="BigBrainIntelligence",
        message_type=message_type,
        priority=priority,
        data=data
    )
    return await bridge.send_message(message)


async def send_bigbrain_to_crypto(message_type: str, data: Dict[str, Any], priority: str = "medium") -> bool:
    """Convenience function to send messages from BigBrain to CryptoIntel."""
    bridge = await get_bridge()
    message = IntelligenceBridgeMessage(
        source_dept="BigBrainIntelligence",
        target_dept="CryptoIntelligence",
        message_type=message_type,
        priority=priority,
        data=data
    )
    return await bridge.send_message(message)