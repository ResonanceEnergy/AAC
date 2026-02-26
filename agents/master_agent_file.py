#!/usr/bin/env python3
"""
AAC MASTER AGENT CONSOLIDATION FILE
====================================

Unified interface for all AAC agent systems.
Consolidates 26 agents across research, operational, and strategic domains.

Agent Categories:
- 20 Research Agents (BigBrain Intelligence)
- 6 Super Agents (Department Operations)
- Contest Agents (Trading Competition)
- Security Agents (System Protection)

Usage:
    from master_agent_file import get_all_agents, initialize_agent_system
    agents = get_all_agents()
    await initialize_agent_system()
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Core agent imports
from BigBrainIntelligence.agents import (
    ResearchAgentManager,
    get_research_agent_manager,
    AGENT_REGISTRY,
    get_agent,
    get_all_agents as get_all_research_agents
)

from shared.department_super_agents import (
    DEPARTMENT_SUPER_AGENTS,
    initialize_all_department_super_agents,
    get_department_super_agent
)

# Optional agent integrations
try:
    from agent_based_trading_integration import get_agent_integration
except ImportError:
    get_agent_integration = None

try:
    from agent_based_trading import AgentBasedTradingSystem
except ImportError:
    AgentBasedTradingSystem = None

# Configure logging
logger = logging.getLogger("AAC.MasterAgent")

class AACMasterAgentSystem:
    """
    Master controller for all AAC agent systems.
    Provides unified interface for agent management and coordination.
    """

    def __init__(self):
        self.research_manager: Optional[ResearchAgentManager] = None
        self.super_agents: Dict[str, Any] = {}
        self.agent_integration = None
        self.trading_system = None
        self.initialized = False
        self.agent_counts = {
            'research': 0,
            'super': 0,
            'contest': 0,
            'security': 0,
            'total': 0
        }

    async def initialize(self) -> bool:
        """
        Initialize all agent systems.
        Returns True if successful, False otherwise.
        """
        try:
            logger.info("🔬 Initializing AAC Master Agent System...")

            # Initialize research agents
            self.research_manager = await get_research_agent_manager()
            self.agent_counts['research'] = len(AGENT_REGISTRY)

            # Initialize department super agents
            self.super_agents = await initialize_all_department_super_agents()
            super_count = 0
            for dept_agents in self.super_agents.values():
                super_count += len(dept_agents)
            self.agent_counts['super'] = super_count

            # Initialize agent integration (contest system)
            if get_agent_integration:
                self.agent_integration = await get_agent_integration()
                if self.agent_integration:
                    contest_status = await self.agent_integration.get_integration_status()
                    self.agent_counts['contest'] = contest_status.get('total_agents', 0)

            # Initialize trading system
            if AgentBasedTradingSystem:
                self.trading_system = AgentBasedTradingSystem()
                await self.trading_system.initialize()

            # Calculate totals
            self.agent_counts['total'] = (
                self.agent_counts['research'] +
                self.agent_counts['super'] +
                self.agent_counts['contest']
            )

            self.initialized = True
            logger.info(f"✅ Master Agent System initialized: {self.agent_counts['total']} agents")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize master agent system: {e}")
            return False

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all agent systems."""
        if not self.initialized:
            return {'status': 'not_initialized'}

        status = {
            'status': 'operational',
            'timestamp': datetime.now().isoformat(),
            'agent_counts': self.agent_counts.copy(),
            'research_agents': {},
            'super_agents': {},
            'integration_status': None
        }

        # Research agent status
        if self.research_manager:
            for agent_id, agent in self.research_manager.agents.items():
                status['research_agents'][agent_id] = {
                    'status': self.research_manager.agent_status[agent_id]['status'],
                    'health_score': self.research_manager.agent_status[agent_id]['health_score'],
                    'last_scan': self.research_manager.agent_status[agent_id]['last_scan']
                }

        # Super agent status
        status['super_agents'] = self.super_agents

        # Integration status
        if self.agent_integration:
            status['integration_status'] = await self.agent_integration.get_integration_status()

        return status

    async def run_agent_scan(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Run a scan on a specific research agent."""
        if not self.research_manager:
            return None

        agent = self.research_manager.agents.get(agent_id)
        if agent:
            try:
                findings = await agent.run_scan()
                # Update status
                self.research_manager.agent_status[agent_id]['last_scan'] = datetime.now()
                self.research_manager.agent_status[agent_id]['findings_count'] = len(findings) if findings else 0
                return findings
            except Exception as e:
                logger.error(f"Failed to run scan for {agent_id}: {e}")
                self.research_manager.agent_status[agent_id]['error_message'] = str(e)
                return None
        return None

    async def get_department_agent(self, department: str, agent_type: str) -> Optional[Any]:
        """Get a department super agent."""
        return await get_department_super_agent(department, agent_type)

    async def shutdown(self):
        """Shutdown all agent systems gracefully."""
        logger.info("🔽 Shutting down AAC Master Agent System...")

        # Shutdown research agents
        if self.research_manager:
            # Research agents don't have explicit shutdown, but we can log
            pass

        # Shutdown integration
        if self.agent_integration:
            # Integration shutdown if available
            pass

        # Shutdown trading system
        if self.trading_system:
            await self.trading_system.shutdown()

        logger.info("✅ Master Agent System shutdown complete")

# Global instance
_master_agent_system: Optional[AACMasterAgentSystem] = None

async def get_master_agent_system() -> AACMasterAgentSystem:
    """Get or create the global master agent system instance."""
    global _master_agent_system
    if _master_agent_system is None:
        _master_agent_system = AACMasterAgentSystem()
        await _master_agent_system.initialize()
    return _master_agent_system

def get_all_agents() -> Dict[str, List[str]]:
    """Get a comprehensive list of all agents by category."""
    return {
        'research_agents': list(AGENT_REGISTRY.keys()),
        'super_agents': {
            dept: list(agents.keys())
            for dept, agents in DEPARTMENT_SUPER_AGENTS.items()
        },
        'total_research': len(AGENT_REGISTRY),
        'total_super': sum(len(agents) for agents in DEPARTMENT_SUPER_AGENTS.values()),
        'total_overall': len(AGENT_REGISTRY) + sum(len(agents) for agents in DEPARTMENT_SUPER_AGENTS.values())
    }

async def initialize_agent_system() -> bool:
    """Initialize the complete agent system."""
    system = await get_master_agent_system()
    return system.initialized

async def get_agent_status_report() -> Dict[str, Any]:
    """Get a comprehensive agent status report."""
    system = await get_master_agent_system()
    return await system.get_system_status()

# CLI interface for testing
if __name__ == '__main__':
    async def main():
        print("=== AAC MASTER AGENT SYSTEM ===")
        print()

        # Initialize system
        print("Initializing agent system...")
        success = await initialize_agent_system()
        if not success:
            print("Failed to initialize agent system")
            return

        # Get agent inventory
        agents = get_all_agents()
        print(f"Research Agents: {agents['total_research']}")
        print(f"Super Agents: {agents['total_super']}")
        print(f"Total Agents: {agents['total_overall']}")
        print()

        # Get status report
        print("Getting system status...")
        status = await get_agent_status_report()
        print(f"System Status: {status['status']}")
        print(f"Agent Counts: {status['agent_counts']}")
        print()

        # List research agents
        print("Research Agents:")
        for agent_id in agents['research_agents']:
            print(f"  - {agent_id}")
        print()

        # List super agents
        print("Super Agents by Department:")
        for dept, dept_agents in agents['super_agents'].items():
            print(f"  {dept}:")
            for agent in dept_agents:
                print(f"    - {agent}")
        print()

        print("=== AGENT SYSTEM READY ===")

    asyncio.run(main())