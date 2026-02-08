#!/usr/bin/env python3
"""
TradingExecution ↔ CryptoIntelligence Bridge
=============================================

Bridge between TradingExecution and CryptoIntelligence departments
for venue selection, execution routing, and withdrawal coordination.

This bridge enables:
- Venue selection based on trading requirements
- Execution routing to optimal crypto venues
- Withdrawal coordination and settlement
- Real-time venue health and capacity monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from shared.bridge_orchestrator import BridgeMessage, BridgeMessageType, MessagePriority, Department
from shared.audit_logger import get_audit_logger, AuditCategory, AuditSeverity

logger = logging.getLogger(__name__)


class TradingCryptoBridge:
    """
    Bridge between TradingExecution and CryptoIntelligence departments.
    Handles venue selection, execution routing, and withdrawal coordination.
    """

    def __init__(self):
        self.audit_logger = get_audit_logger()

        # Bridge state
        self.is_initialized = False
        self.last_venue_selection = None
        self.last_execution_routing = None

        # Venue management
        self.venue_capacity: Dict[str, Dict] = {}
        self.active_routes: Dict[str, Dict] = {}

        # Withdrawal coordination
        self.pending_withdrawals: Dict[str, Dict] = {}

        # Performance metrics
        self.performance_metrics = {
            "venue_selections": 0,
            "execution_routes": 0,
            "withdrawals_coordinated": 0,
            "venue_health_checks": 0
        }

    async def initialize(self) -> bool:
        """Initialize the bridge."""
        try:
            logger.info("Initializing TradingExecution ↔ CryptoIntelligence bridge")

            # Initialize state
            self.is_initialized = True

            await self.audit_logger.log_event(
                category=AuditCategory.SYSTEM,
                action="bridge_initialized",
                resource="trading_crypto_bridge",
                severity=AuditSeverity.INFO,
                details={"bridge_type": "trading_crypto"}
            )

            logger.info("TradingExecution ↔ CryptoIntelligence bridge initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize trading-crypto bridge: {e}")
            return False

    async def handle_message(self, message: BridgeMessage) -> bool:
        """Handle incoming bridge messages."""
        try:
            if message.message_type == BridgeMessageType.VENUE_SELECTION:
                return await self._handle_venue_selection(message)
            elif message.message_type == BridgeMessageType.EXECUTION_ROUTING:
                return await self._handle_execution_routing(message)
            elif message.message_type == BridgeMessageType.WITHDRAWAL_COORDINATION:
                return await self._handle_withdrawal_coordination(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return False

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return False

    async def _handle_venue_selection(self, message: BridgeMessage) -> bool:
        """Handle venue selection requests."""
        try:
            selection_data = message.data
            asset = selection_data.get("asset")
            amount = selection_data.get("amount", 0)
            strategy = selection_data.get("strategy", "arbitrage")

            # Select optimal venue based on criteria
            selected_venue = await self._select_optimal_venue(asset, amount, strategy)

            if selected_venue:
                self.last_venue_selection = datetime.now()
                self.performance_metrics["venue_selections"] += 1

                await self.audit_logger.log_event(
                    category=AuditCategory.TRADING,
                    action="venue_selected",
                    resource="trading_crypto_bridge",
                    severity=AuditSeverity.INFO,
                    details={
                        "asset": asset,
                        "amount": amount,
                        "selected_venue": selected_venue,
                        "strategy": strategy
                    }
                )

                logger.info(f"Selected venue {selected_venue} for {asset} trade")
                return True
            else:
                logger.warning(f"No suitable venue found for {asset}")
                return False

        except Exception as e:
            logger.error(f"Error handling venue selection: {e}")
            return False

    async def _handle_execution_routing(self, message: BridgeMessage) -> bool:
        """Handle execution routing requests."""
        try:
            routing_data = message.data
            order_id = routing_data.get("order_id")
            venue = routing_data.get("venue")
            execution_params = routing_data.get("execution_params", {})

            # Route execution to specified venue
            route_id = await self._route_execution(order_id, venue, execution_params)

            if route_id:
                self.last_execution_routing = datetime.now()
                self.performance_metrics["execution_routes"] += 1

                await self.audit_logger.log_event(
                    category=AuditCategory.TRADING,
                    action="execution_routed",
                    resource="trading_crypto_bridge",
                    severity=AuditSeverity.INFO,
                    details={
                        "order_id": order_id,
                        "venue": venue,
                        "route_id": route_id
                    }
                )

                logger.info(f"Routed execution {order_id} to venue {venue}")
                return True
            else:
                logger.error(f"Failed to route execution {order_id}")
                return False

        except Exception as e:
            logger.error(f"Error handling execution routing: {e}")
            return False

    async def _handle_withdrawal_coordination(self, message: BridgeMessage) -> bool:
        """Handle withdrawal coordination requests."""
        try:
            withdrawal_data = message.data
            withdrawal_id = withdrawal_data.get("withdrawal_id")
            asset = withdrawal_data.get("asset")
            amount = withdrawal_data.get("amount", 0)
            destination = withdrawal_data.get("destination")

            # Coordinate withdrawal
            coordination_result = await self._coordinate_withdrawal(
                withdrawal_id, asset, amount, destination
            )

            if coordination_result:
                self.performance_metrics["withdrawals_coordinated"] += 1

                await self.audit_logger.log_event(
                    category=AuditCategory.TRADING,
                    action="withdrawal_coordinated",
                    resource="trading_crypto_bridge",
                    severity=AuditSeverity.INFO,
                    details={
                        "withdrawal_id": withdrawal_id,
                        "asset": asset,
                        "amount": amount,
                        "destination": destination
                    }
                )

                logger.info(f"Coordinated withdrawal {withdrawal_id} for {amount} {asset}")
                return True
            else:
                logger.error(f"Failed to coordinate withdrawal {withdrawal_id}")
                return False

        except Exception as e:
            logger.error(f"Error handling withdrawal coordination: {e}")
            return False

    async def _select_optimal_venue(self, asset: str, amount: float, strategy: str) -> Optional[str]:
        """Select the optimal venue for trading."""
        try:
            # Get venue health and capacity data
            venue_scores = {}

            for venue_name in ["binance", "coinbase", "kraken", "gemini"]:  # Example venues
                health_score = await self._get_venue_health(venue_name)
                capacity_score = await self._get_venue_capacity(venue_name, asset, amount)
                liquidity_score = await self._get_venue_liquidity(venue_name, asset)

                # Calculate composite score
                total_score = (health_score * 0.4) + (capacity_score * 0.3) + (liquidity_score * 0.3)
                venue_scores[venue_name] = total_score

            # Select venue with highest score
            if venue_scores:
                best_venue = max(venue_scores, key=venue_scores.get)
                return best_venue if venue_scores[best_venue] > 0.6 else None

            return None

        except Exception as e:
            logger.error(f"Error selecting optimal venue: {e}")
            return None

    async def _get_venue_health(self, venue_name: str) -> float:
        """Get venue health score (0.0 to 1.0)."""
        # Placeholder - would integrate with CryptoIntelligence engine
        return 0.85  # Mock healthy venue

    async def _get_venue_capacity(self, venue_name: str, asset: str, amount: float) -> float:
        """Get venue capacity score for asset/amount."""
        # Placeholder - would check venue limits and current utilization
        return 0.90  # Mock good capacity

    async def _get_venue_liquidity(self, venue_name: str, asset: str) -> float:
        """Get venue liquidity score for asset."""
        # Placeholder - would check order book depth
        return 0.80  # Mock good liquidity

    async def _route_execution(self, order_id: str, venue: str, execution_params: Dict) -> Optional[str]:
        """Route execution to venue."""
        try:
            route_id = f"route_{order_id}_{int(datetime.now().timestamp())}"

            self.active_routes[route_id] = {
                "order_id": order_id,
                "venue": venue,
                "params": execution_params,
                "status": "routed",
                "timestamp": datetime.now()
            }

            return route_id

        except Exception as e:
            logger.error(f"Error routing execution: {e}")
            return None

    async def _coordinate_withdrawal(self, withdrawal_id: str, asset: str, amount: float, destination: str) -> bool:
        """Coordinate withdrawal with venue."""
        try:
            self.pending_withdrawals[withdrawal_id] = {
                "asset": asset,
                "amount": amount,
                "destination": destination,
                "status": "coordinated",
                "timestamp": datetime.now()
            }

            return True

        except Exception as e:
            logger.error(f"Error coordinating withdrawal: {e}")
            return False

    async def get_bridge_health(self) -> Dict[str, Any]:
        """Get bridge health status."""
        return {
            "is_initialized": self.is_initialized,
            "last_venue_selection": self.last_venue_selection.isoformat() if self.last_venue_selection else None,
            "last_execution_routing": self.last_execution_routing.isoformat() if self.last_execution_routing else None,
            "active_routes": len(self.active_routes),
            "pending_withdrawals": len(self.pending_withdrawals),
            "performance_metrics": self.performance_metrics
        }

    async def shutdown(self):
        """Shutdown the bridge."""
        logger.info("Shutting down TradingExecution ↔ CryptoIntelligence bridge")
        # Cleanup resources if needed
        self.is_initialized = False