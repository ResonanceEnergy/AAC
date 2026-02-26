#!/usr/bin/env python3
"""
TradingExecution ↔ BigBrainIntelligence Bridge
==============================================

Bridge between TradingExecution and BigBrainIntelligence departments
for strategy signals, execution feedback, and research-driven trading.

This bridge enables:
- Research insights feeding into trading strategies
- Execution feedback improving research models
- Strategy performance attribution
- Real-time signal processing and validation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from shared.bridge_orchestrator import BridgeMessage, BridgeMessageType, MessagePriority, Department
from shared.audit_logger import get_audit_logger, AuditCategory, AuditSeverity

logger = logging.getLogger(__name__)


class TradingIntelligenceBridge:
    """
    Bridge between TradingExecution and BigBrainIntelligence departments.
    Handles strategy signals, execution feedback, and research integration.
    """

    def __init__(self):
        self.audit_logger = get_audit_logger()

        # Bridge state
        self.is_initialized = False
        self.last_signal_processed = None
        self.last_feedback_sent = None

        # Signal processing
        self.active_signals: Dict[str, Dict] = {}
        self.signal_performance: Dict[str, Dict] = {}

        # Strategy attribution
        self.strategy_pnl: Dict[str, float] = {}
        self.research_contribution: Dict[str, float] = {}

        # Feedback queue
        self.feedback_queue: List[Dict] = []

        # Performance metrics
        self.performance_metrics = {
            "signals_processed": 0,
            "feedback_sent": 0,
            "strategies_attributed": 0,
            "research_insights_applied": 0
        }

    async def initialize(self) -> bool:
        """Initialize the bridge."""
        try:
            logger.info("Initializing TradingExecution ↔ BigBrainIntelligence bridge")

            # Initialize state
            self.is_initialized = True

            await self.audit_logger.log_event(
                category=AuditCategory.SYSTEM,
                action="bridge_initialized",
                resource="trading_intelligence_bridge",
                severity=AuditSeverity.INFO,
                details={"bridge_type": "trading_intelligence"}
            )

            logger.info("TradingExecution ↔ BigBrainIntelligence bridge initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize trading-intelligence bridge: {e}")
            return False

    async def handle_message(self, message: BridgeMessage) -> bool:
        """Handle incoming bridge messages."""
        try:
            if message.message_type == BridgeMessageType.STRATEGY_SIGNAL:
                return await self._handle_strategy_signal(message)
            elif message.message_type == BridgeMessageType.EXECUTION_FEEDBACK:
                return await self._handle_execution_feedback(message)
            elif message.message_type == BridgeMessageType.RESEARCH_INSIGHT:
                return await self._handle_research_insight(message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return False

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return False

    async def _handle_strategy_signal(self, message: BridgeMessage) -> bool:
        """Handle strategy signals from research."""
        try:
            signal_data = message.payload
            signal_id = signal_data.get("signal_id")

            if not signal_id:
                signal_id = f"sig_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Process and validate signal
            self.active_signals[signal_id] = {
                "data": signal_data,
                "received_at": datetime.now(),
                "status": "processing"
            }

            self.last_signal_processed = datetime.now()
            self.performance_metrics["signals_processed"] += 1

            await self.audit_logger.log_event(
                category=AuditCategory.TRADING,
                action="strategy_signal_received",
                resource="trading_intelligence_bridge",
                severity=AuditSeverity.INFO,
                details={"signal_id": signal_id, "signal_data": signal_data}
            )

            logger.info(f"Processed strategy signal: {signal_id}")
            return True

        except Exception as e:
            logger.error(f"Error handling strategy signal: {e}")
            return False

    async def _handle_execution_feedback(self, message: BridgeMessage) -> bool:
        """Handle execution feedback for research improvement."""
        try:
            feedback_data = message.payload

            # Queue feedback for processing
            self.feedback_queue.append({
                "data": feedback_data,
                "received_at": datetime.now()
            })

            self.last_feedback_sent = datetime.now()
            self.performance_metrics["feedback_sent"] += 1

            await self.audit_logger.log_event(
                category=AuditCategory.TRADING,
                action="execution_feedback_received",
                resource="trading_intelligence_bridge",
                severity=AuditSeverity.INFO,
                details={"feedback_data": feedback_data}
            )

            logger.info("Execution feedback queued for research")
            return True

        except Exception as e:
            logger.error(f"Error handling execution feedback: {e}")
            return False

    async def _handle_research_insight(self, message: BridgeMessage) -> bool:
        """Handle research insights for trading strategies."""
        try:
            insight_data = message.payload

            # Apply research insights to strategy attribution
            strategy_id = insight_data.get("strategy_id")
            if strategy_id:
                self.research_contribution[strategy_id] = insight_data.get("contribution_score", 0.0)
                self.performance_metrics["research_insights_applied"] += 1

            await self.audit_logger.log_event(
                category=AuditCategory.TRADING,
                action="research_insight_applied",
                resource="trading_intelligence_bridge",
                severity=AuditSeverity.INFO,
                details={"insight_data": insight_data}
            )

            logger.info(f"Applied research insight for strategy: {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"Error handling research insight: {e}")
            return False

    async def get_signal_performance(self, signal_id: str) -> Optional[Dict]:
        """Get performance data for a specific signal."""
        return self.signal_performance.get(signal_id)

    async def get_strategy_attribution(self, strategy_id: str) -> Dict[str, Any]:
        """Get attribution data for a strategy."""
        return {
            "strategy_id": strategy_id,
            "pnl": self.strategy_pnl.get(strategy_id, 0.0),
            "research_contribution": self.research_contribution.get(strategy_id, 0.0)
        }

    async def get_bridge_health(self) -> Dict[str, Any]:
        """Get bridge health status."""
        return {
            "is_initialized": self.is_initialized,
            "last_signal_processed": self.last_signal_processed.isoformat() if self.last_signal_processed else None,
            "last_feedback_sent": self.last_feedback_sent.isoformat() if self.last_feedback_sent else None,
            "active_signals": len(self.active_signals),
            "feedback_queue_size": len(self.feedback_queue),
            "performance_metrics": self.performance_metrics
        }

    async def shutdown(self):
        """Shutdown the bridge."""
        logger.info("Shutting down TradingExecution ↔ BigBrainIntelligence bridge")
        # Cleanup resources if needed
        self.is_initialized = False
        self.execution_feedback_queue: asyncio.Queue = asyncio.Queue()

    async def initialize_bridge(self) -> bool:
        """Initialize the trading-intelligence bridge."""
        try:
            logger.info("Initializing TradingExecution ↔ BigBrainIntelligence bridge")

            # Load active signals from BigBrainIntelligence
            await self._load_active_signals()

            # Start feedback processing
            asyncio.create_task(self._process_execution_feedback())

            # Start signal validation
            asyncio.create_task(self._signal_validation_loop())

            self.is_initialized = True

            await self.audit_logger.log_event(
                category=AuditCategory.SYSTEM,
                event_type="trading_intelligence_bridge_initialized",
                severity=AuditSeverity.INFO
            )

            logger.info("TradingExecution ↔ BigBrainIntelligence bridge initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize trading-intelligence bridge: {e}")
            await self.audit_logger.log_event(
                category=AuditCategory.SYSTEM,
                event_type="trading_intelligence_bridge_init_failed",
                severity=AuditSeverity.ERROR,
                details={"error": str(e)}
            )
            return False

    async def _load_active_signals(self):
        """Load active trading signals from BigBrainIntelligence."""
        try:
            # In a real implementation, this would query BigBrainIntelligence
            # For now, initialize with sample signals
            logger.info("Loaded active signals from BigBrainIntelligence")

        except Exception as e:
            logger.error(f"Failed to load active signals: {e}")

    async def process_message(self, message: BridgeMessage) -> bool:
        """Process messages from the bridge orchestrator."""
        try:
            if message.message_type == BridgeMessageType.STRATEGY_SIGNAL:
                return await self._handle_strategy_signal(message)

            elif message.message_type == BridgeMessageType.EXECUTION_FEEDBACK:
                return await self._handle_execution_feedback(message)

            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to process bridge message: {e}")
            return False

    async def _handle_strategy_signal(self, message: BridgeMessage) -> bool:
        """Handle strategy signals from BigBrainIntelligence."""
        try:
            signal_data = message.data
            signal_id = signal_data.get("signal_id")

            # Validate signal
            if not await self._validate_signal(signal_data):
                logger.warning(f"Signal validation failed for {signal_id}")
                return False

            # Store active signal
            self.active_signals[signal_id] = {
                "data": signal_data,
                "received_at": datetime.now(),
                "status": "active",
                "performance": {}
            }

            # Forward signal to TradingExecution
            await self._forward_signal_to_trading(signal_data)

            # Initialize performance tracking
            self.signal_performance[signal_id] = {
                "pnl": 0.0,
                "trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0
            }

            await self.audit_logger.log_event(
                category=AuditCategory.TRADING,
                event_type="strategy_signal_received",
                severity=AuditSeverity.INFO,
                details={
                    "signal_id": signal_id,
                    "signal_type": signal_data.get("type"),
                    "confidence": signal_data.get("confidence")
                }
            )

            logger.info(f"Processed strategy signal: {signal_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to handle strategy signal: {e}")
            return False

    async def _handle_execution_feedback(self, message: BridgeMessage) -> bool:
        """Handle execution feedback from TradingExecution."""
        try:
            feedback_data = message.data

            # Queue feedback for processing
            await self.execution_feedback_queue.put(feedback_data)

            await self.audit_logger.log_event(
                category=AuditCategory.TRADING,
                event_type="execution_feedback_received",
                severity=AuditSeverity.INFO,
                details={"feedback_type": feedback_data.get("type")}
            )

            logger.info("Queued execution feedback for processing")
            return True

        except Exception as e:
            logger.error(f"Failed to handle execution feedback: {e}")
            return False

    async def _validate_signal(self, signal_data: Dict) -> bool:
        """Validate strategy signal before forwarding to trading."""
        try:
            # Check required fields
            required_fields = ["signal_id", "type", "confidence", "timestamp"]
            for field in required_fields:
                if field not in signal_data:
                    logger.error(f"Signal missing required field: {field}")
                    return False

            # Validate confidence level
            confidence = signal_data.get("confidence", 0)
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                logger.error(f"Invalid confidence level: {confidence}")
                return False

            # Check signal age (not too old)
            signal_time = datetime.fromisoformat(signal_data["timestamp"])
            age = datetime.now() - signal_time
            if age > timedelta(minutes=5):
                logger.warning(f"Signal too old: {age}")
                return False

            return True

        except Exception as e:
            logger.error(f"Signal validation error: {e}")
            return False

    async def _forward_signal_to_trading(self, signal_data: Dict):
        """Forward validated signal to TradingExecution engine."""
        try:
            # In a real implementation, this would send to TradingExecution engine
            logger.info(f"Forwarded signal to TradingExecution: {signal_data.get('signal_id')}")

        except Exception as e:
            logger.error(f"Failed to forward signal to trading: {e}")

    async def _process_execution_feedback(self):
        """Process execution feedback and update research models."""
        while True:
            try:
                feedback = await self.execution_feedback_queue.get()

                # Process feedback based on type
                feedback_type = feedback.get("type")

                if feedback_type == "trade_executed":
                    await self._process_trade_feedback(feedback)
                elif feedback_type == "signal_performance":
                    await self._process_signal_performance(feedback)
                elif feedback_type == "market_conditions":
                    await self._process_market_feedback(feedback)

                self.last_feedback_sent = datetime.now()

            except Exception as e:
                logger.error(f"Feedback processing error: {e}")

    async def _process_trade_feedback(self, feedback: Dict):
        """Process individual trade execution feedback."""
        try:
            signal_id = feedback.get("signal_id")
            pnl = feedback.get("realized_pnl", 0)

            if signal_id in self.signal_performance:
                perf = self.signal_performance[signal_id]
                perf["trades"] += 1
                perf["pnl"] += pnl

                # Update win rate
                if pnl > 0:
                    perf["win_rate"] = ((perf["win_rate"] * (perf["trades"] - 1)) + 1) / perf["trades"]
                else:
                    perf["win_rate"] = (perf["win_rate"] * (perf["trades"] - 1)) / perf["trades"]

                # Update average return
                perf["avg_return"] = perf["pnl"] / perf["trades"]

            # Forward feedback to BigBrainIntelligence for model improvement
            await self._send_feedback_to_research(feedback)

        except Exception as e:
            logger.error(f"Trade feedback processing error: {e}")

    async def _process_signal_performance(self, feedback: Dict):
        """Process overall signal performance feedback."""
        try:
            signal_id = feedback.get("signal_id")
            performance_data = feedback.get("performance", {})

            if signal_id in self.signal_performance:
                self.signal_performance[signal_id].update(performance_data)

            # Send performance data to research for model refinement
            await self._send_performance_to_research(signal_id, performance_data)

        except Exception as e:
            logger.error(f"Signal performance processing error: {e}")

    async def _process_market_feedback(self, feedback: Dict):
        """Process market condition feedback."""
        try:
            # Forward market data to research for signal validation
            await self._send_market_data_to_research(feedback)

        except Exception as e:
            logger.error(f"Market feedback processing error: {e}")

    async def _send_feedback_to_research(self, feedback: Dict):
        """Send execution feedback to BigBrainIntelligence."""
        try:
            # In a real implementation, this would send to BigBrainIntelligence
            logger.info("Sent execution feedback to BigBrainIntelligence")

        except Exception as e:
            logger.error(f"Failed to send feedback to research: {e}")

    async def _send_performance_to_research(self, signal_id: str, performance: Dict):
        """Send signal performance data to BigBrainIntelligence."""
        try:
            # In a real implementation, this would send to BigBrainIntelligence
            logger.info(f"Sent performance data for signal {signal_id} to BigBrainIntelligence")

        except Exception as e:
            logger.error(f"Failed to send performance to research: {e}")

    async def _send_market_data_to_research(self, market_data: Dict):
        """Send market data to BigBrainIntelligence."""
        try:
            # In a real implementation, this would send to BigBrainIntelligence
            logger.info("Sent market data to BigBrainIntelligence")

        except Exception as e:
            logger.error(f"Failed to send market data to research: {e}")

    async def _signal_validation_loop(self):
        """Periodically validate active signals."""
        while True:
            try:
                await asyncio.sleep(60)  # Validate every minute

                now = datetime.now()
                expired_signals = []

                for signal_id, signal_info in self.active_signals.items():
                    # Check if signal has expired (e.g., after 1 hour)
                    age = now - signal_info["received_at"]
                    if age > timedelta(hours=1):
                        expired_signals.append(signal_id)
                        continue

                    # Validate signal performance
                    perf = self.signal_performance.get(signal_id, {})
                    if perf.get("trades", 0) > 10:  # After 10 trades
                        win_rate = perf.get("win_rate", 0)
                        if win_rate < 0.3:  # Poor performance
                            logger.warning(f"Signal {signal_id} showing poor performance (win_rate: {win_rate})")
                            # Could deactivate signal or alert research

                # Clean up expired signals
                for signal_id in expired_signals:
                    del self.active_signals[signal_id]
                    logger.info(f"Expired signal: {signal_id}")

            except Exception as e:
                logger.error(f"Signal validation error: {e}")

    async def get_signal_performance(self, signal_id: str) -> Optional[Dict]:
        """Get performance data for a specific signal."""
        return self.signal_performance.get(signal_id)

    async def get_active_signals_summary(self) -> Dict[str, Any]:
        """Get summary of active signals."""
        return {
            "active_signals": len(self.active_signals),
            "total_signals_processed": len(self.signal_performance),
            "signals_by_type": {},
            "average_performance": {}
        }

    async def get_bridge_health(self) -> Dict[str, Any]:
        """Get bridge health status."""
        return {
            "is_initialized": self.is_initialized,
            "active_signals": len(self.active_signals),
            "feedback_queue_size": self.execution_feedback_queue.qsize(),
            "last_signal_processed": self.last_signal_processed.isoformat() if self.last_signal_processed else None,
            "last_feedback_sent": self.last_feedback_sent.isoformat() if self.last_feedback_sent else None,
            "signal_performance_tracked": len(self.signal_performance)
        }

    async def shutdown(self):
        """Shutdown the bridge."""
        logger.info("Shutting down TradingExecution ↔ BigBrainIntelligence bridge")
        # Cleanup resources if needed