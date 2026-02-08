#!/usr/bin/env python3
"""
AAC Agent-Based Trading Integration
====================================

Integration layer between the agent-based trading contest and AAC infrastructure.
Connects trading agents to live market data, security systems, and audit logging.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent_based_trading import AgentContestOrchestrator, TradingAgent, InnovationAgent
from shared.audit_logger import AuditLogger, AuditCategory, AuditSeverity
from shared.communication import CommunicationFramework
from shared.data_sources import DataAggregator
from shared.config_loader import get_config

# Import AAC system components (with fallbacks for missing modules)
try:
    from market_data_aggregator import MarketDataAggregator
except ImportError:
    MarketDataAggregator = None

try:
    from trading_desk_security import TradingDeskSecurity
except ImportError:
    TradingDeskSecurity = None

logger = logging.getLogger("AAC.AgentIntegration")


class AACAgentIntegration:
    """
    Integration layer for agent-based trading contest within AAC ecosystem.

    Connects the contest system to:
    - Live market data feeds
    - Trading desk security
    - Audit logging
    - Communication framework
    - BigBrainIntelligence/CryptoIntelligence
    """

    def __init__(self):
        self.config = get_config()
        self.contest_orchestrator = None
        self.audit_logger = AuditLogger()
        self.communication = CommunicationFramework()
        self.data_aggregator = None
        self.market_data_aggregator = None
        self.trading_security = None

        # Integration status
        self.market_data_connected = False
        self.security_initialized = False
        self.communication_registered = False

        # Contest monitoring
        self.contest_monitor_task = None
        self.performance_monitor_task = None

    async def initialize_integration(self) -> bool:
        """
        Initialize all AAC system integrations.

        Returns:
            bool: True if all integrations successful
        """
        logger.info("🔗 Initializing AAC Agent Integration...")

        success = True

        try:
            # Initialize data aggregator
            self.data_aggregator = DataAggregator()
            logger.info("✅ Data aggregator initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize data aggregator: {e}")
            success = False

        try:
            # Initialize market data aggregator if available
            if MarketDataAggregator:
                try:
                    self.market_data_aggregator = MarketDataAggregator(self.communication, self.audit_logger)
                    await self.market_data_aggregator.initialize()
                    self.market_data_connected = True
                    logger.info("✅ Market data aggregator connected")
                except Exception as init_error:
                    logger.warning(f"⚠️ Market data aggregator initialization failed: {init_error}")
                    self.market_data_aggregator = None
            else:
                logger.warning("⚠️ Market data aggregator not available - using mock data")

        except Exception as e:
            logger.error(f"❌ Failed to connect market data: {e}")
            success = False

        try:
            # Initialize trading desk security if available
            if TradingDeskSecurity:
                try:
                    self.trading_security = TradingDeskSecurity(audit_logger=self.audit_logger)
                    self.security_initialized = True
                    logger.info("✅ Trading desk security initialized")
                except Exception as init_error:
                    logger.warning(f"⚠️ Trading desk security initialization failed: {init_error}")
                    self.trading_security = None
            else:
                logger.warning("⚠️ Trading desk security not available - using mock security")

        except Exception as e:
            logger.error(f"❌ Failed to initialize trading security: {e}")
            success = False

        try:
            # Register with communication framework
            await self.communication.register_channel("agent_contest", "trading")
            await self.communication.register_channel("intelligence_routing", "intelligence")
            self.communication_registered = True
            logger.info("✅ Communication framework registered")

        except Exception as e:
            logger.error(f"❌ Failed to register communication: {e}")
            success = False

        # Log integration status
        await self._log_integration_status(success)

        return success

    async def _log_integration_status(self, success: bool):
        """Log the integration initialization status"""
        status_data = {
            'market_data_connected': self.market_data_connected,
            'security_initialized': self.security_initialized,
            'communication_registered': self.communication_registered,
            'overall_success': success
        }

        await self.audit_logger.log_event(
            category="system",
            action="agent_integration_init",
            user="AAC_AgentIntegration",
            status="success" if success else "failure",
            details=status_data
        )

    async def create_integrated_contest(self) -> AgentContestOrchestrator:
        """
        Create contest orchestrator with full AAC integration.

        Returns:
            AgentContestOrchestrator: Fully integrated contest system
        """
        logger.info("🎯 Creating integrated agent contest...")

        # Create base orchestrator
        orchestrator = AgentContestOrchestrator()

        # Enhance with AAC integrations
        orchestrator.audit_logger = self.audit_logger
        orchestrator.communication = self.communication
        orchestrator.market_data = self.market_data_aggregator or self.data_aggregator
        orchestrator.trading_security = self.trading_security

        # Store reference
        self.contest_orchestrator = orchestrator

        logger.info("✅ Integrated contest orchestrator created")
        return orchestrator

    async def start_contest_monitoring(self):
        """Start background monitoring of the contest"""
        if not self.contest_orchestrator:
            logger.error("❌ No contest orchestrator available for monitoring")
            return

        # Start contest monitor
        self.contest_monitor_task = asyncio.create_task(self._monitor_contest())
        logger.info("📊 Contest monitoring started")

        # Start performance monitor
        self.performance_monitor_task = asyncio.create_task(self._monitor_performance())
        logger.info("📈 Performance monitoring started")

    async def stop_contest_monitoring(self):
        """Stop background monitoring"""
        if self.contest_monitor_task:
            self.contest_monitor_task.cancel()
            try:
                await self.contest_monitor_task
            except asyncio.CancelledError:
                pass

        if self.performance_monitor_task:
            self.performance_monitor_task.cancel()
            try:
                await self.performance_monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 Contest monitoring stopped")

    async def _monitor_contest(self):
        """Monitor contest status and log significant events"""
        while True:
            try:
                if self.contest_orchestrator and self.contest_orchestrator.status == 'active':
                    status = await self.contest_orchestrator.get_contest_status()

                    # Log winner if contest ended
                    if status.get('winner'):
                        winner_agent = status['winner']
                        await self._log_contest_winner(winner_agent, status)

                    # Log hourly status updates
                    current_time = datetime.now()
                    if current_time.minute == 0 and current_time.second < 10:  # Top of hour
                        await self._log_hourly_status(status)

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Contest monitoring error: {e}")
                await asyncio.sleep(60)

    async def _monitor_performance(self):
        """Monitor agent performance and system health"""
        while True:
            try:
                if self.contest_orchestrator and self.contest_orchestrator.status == 'active':
                    # Get performance metrics
                    performance_data = await self._collect_performance_metrics()

                    # Log performance data
                    await self.audit_logger.log_event(
                        category="system",
                        action="performance_monitoring",
                        user="AAC_AgentContest",
                        status="success",
                        details=performance_data
                    )

                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Performance monitoring error: {e}")
                await asyncio.sleep(300)

    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive performance metrics"""
        if not self.contest_orchestrator:
            return {}

        trading_agents = [a for a in self.contest_orchestrator.agents.values()
                         if isinstance(a, TradingAgent)]

        metrics = {
            'total_agents': len(trading_agents),
            'active_agents': len([a for a in trading_agents if a.status.name == 'ACTIVE']),
            'winner_count': len([a for a in trading_agents if a.contest_result.winner]),
            'average_portfolio_value': sum(a.contest_result.total_value for a in trading_agents) / len(trading_agents) if trading_agents else 0,
            'total_trades': sum(a.contest_result.total_trades for a in trading_agents),
            'total_fees': sum(a.contest_result.total_fees for a in trading_agents),
            'market_data_connected': self.market_data_connected,
            'security_active': self.security_initialized,
            'communication_active': self.communication_registered
        }

        return metrics

    async def _log_contest_winner(self, winner_agent_id: str, status: Dict[str, Any]):
        """Log contest winner event"""
        winner_data = next((agent for agent in status['leaderboard'] if agent['agent_id'] == winner_agent_id), {})

        await self.audit_logger.log_event(
            category="trade",
            action="contest_winner",
            user="AAC_AgentContest",
            status="success",
            details={
                'winner_agent': winner_agent_id,
                'strategy': winner_data.get('strategy'),
                'final_value': winner_data.get('portfolio_value'),
                'net_profit': winner_data.get('net_profit'),
                'trades': winner_data.get('trades'),
                'win_rate': winner_data.get('win_rate'),
                'contest_duration': status.get('contest_start')
            }
        )

    async def _log_hourly_status(self, status: Dict[str, Any]):
        """Log hourly contest status"""
        await self.audit_logger.log_event(
            category="system",
            action="contest_status_update",
            user="AAC_AgentContest",
            status="success",
            details=status
        )

    async def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        return {
            'market_data_connected': self.market_data_connected,
            'security_initialized': self.security_initialized,
            'communication_registered': self.communication_registered,
            'contest_active': self.contest_orchestrator.status if self.contest_orchestrator else 'none',
            'monitoring_active': not (self.contest_monitor_task.done() if self.contest_monitor_task else True)
        }

    async def shutdown_integration(self):
        """Shutdown all integrations gracefully"""
        logger.info("🔄 Shutting down AAC Agent Integration...")

        # Stop monitoring
        await self.stop_contest_monitoring()

        # Close connections
        if self.market_data_aggregator and hasattr(self.market_data_aggregator, 'close'):
            await self.market_data_aggregator.close()

        if self.trading_security and hasattr(self.trading_security, 'shutdown'):
            await self.trading_security.shutdown()

        # Log shutdown
        await self.audit_logger.log_event(
            category="system",
            action="integration_shutdown",
            user="AAC_AgentIntegration",
            status="success"
        )

        logger.info("✅ AAC Agent Integration shutdown complete")


# Global integration instance
_integration_instance = None

async def get_agent_integration() -> AACAgentIntegration:
    """Get or create the global agent integration instance"""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = AACAgentIntegration()
        await _integration_instance.initialize_integration()
    return _integration_instance


async def run_integrated_contest():
    """Run the agent contest with full AAC integration"""
    logger.info("🚀 Starting AAC Integrated Agent Contest!")

    # Initialize integration
    integration = await get_agent_integration()

    # Create integrated contest
    orchestrator = await integration.create_integrated_contest()

    # Initialize contest
    await orchestrator.initialize_contest()

    # Start monitoring
    await integration.start_contest_monitoring()

    try:
        # Run contest
        round_num = 0
        max_rounds = 1000

        while round_num < max_rounds:
            round_num += 1
            logger.info(f"🎲 Round {round_num} starting...")

            contest_over = await orchestrator.run_contest_round()

            if contest_over:
                logger.info(f"🏁 Contest ended in round {round_num}")
                break

            # Progress update every 10 rounds
            if round_num % 10 == 0:
                status = await orchestrator.get_contest_status()
                logger.info(f"📊 Round {round_num}: {status['active_agents']} agents still active")
                if status['leaderboard']:
                    top_agent = status['leaderboard'][0]
                    logger.info(f"🥇 Leader: {top_agent['agent_id']} (${top_agent['portfolio_value']:.2f})")

            # Small delay between rounds
            await asyncio.sleep(0.1)

        # Final report
        final_report = await orchestrator.generate_contest_report()
        logger.info("📈 Final contest report generated")

        return final_report

    finally:
        # Shutdown integration
        await integration.shutdown_integration()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run integrated contest
    result = asyncio.run(run_integrated_contest())
    print("\n🎉 Integrated Contest Complete! Check reports/contest_report.json for full results!")