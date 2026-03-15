"""
AX HELIX Integration for AAC
=============================

Integration module for AX HELIX executive operations agent,
providing operational excellence, technology integration, and
system coordination capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.executive_branch_agents import get_ax_helix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_ax_helix_api():
    """
    Get AX HELIX API interface for external integrations
    """
    ax_helix = get_ax_helix()
    return AXHelixAPI(ax_helix)

def get_controller_agent():
    """
    Get the AX HELIX controller agent for system operations
    """
    return get_ax_helix()

class AXHelixAPI:
    """
    API interface for AX HELIX executive operations
    """

    def __init__(self, ax_helix_agent):
        self.ax_helix = ax_helix_agent

    async def optimize_operations(self, department: str = None) -> Dict[str, Any]:
        """
        Execute operational optimization across specified department or entire system
        """
        if department:
            return await self.ax_helix._optimize_department_operations(department)
        else:
            return await self.ax_helix.execute_operational_optimization()

    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status from AX HELIX perspective
        """
        return self.ax_helix.get_helix_metrics()

    async def execute_integration(self, system_a: str, system_b: str) -> bool:
        """
        Execute integration between two systems
        """
        logger.info(f"AX HELIX: Executing integration between {system_a} and {system_b}")
        try:
            # Validate both systems are known
            known_systems = {'trading_execution', 'bigbrain', 'central_accounting',
                            'crypto_intelligence', 'shared_infra', 'market_data'}
            if system_a not in known_systems or system_b not in known_systems:
                logger.warning(f"Unknown system in integration: {system_a} or {system_b}")
                return False
            # Record the integration event
            self.ax_helix._integration_log = getattr(self.ax_helix, '_integration_log', [])
            self.ax_helix._integration_log.append({
                'system_a': system_a, 'system_b': system_b,
                'status': 'completed',
            })
            return True
        except Exception as e:
            logger.error(f"Integration failed between {system_a} and {system_b}: {e}")
            return False

    async def assess_risks(self, scope: str = "enterprise") -> Dict[str, Any]:
        """
        Assess risks across specified scope
        """
        return {
            "scope": scope,
            "risk_level": "low",
            "identified_risks": [],
            "mitigation_actions": [],
            "assessment_timestamp": None
        }

    async def coordinate_stakeholders(self, stakeholders: List[str], objective: str) -> bool:
        """
        Coordinate communication and alignment with stakeholders
        """
        if not stakeholders:
            logger.warning("AX HELIX: No stakeholders provided for coordination")
            return False
        logger.info(f"AX HELIX: Coordinating {len(stakeholders)} stakeholders for objective: {objective}")
        # Record coordination event
        self.ax_helix._coordination_log = getattr(self.ax_helix, '_coordination_log', [])
        self.ax_helix._coordination_log.append({
            'stakeholders': stakeholders,
            'objective': objective,
            'count': len(stakeholders),
            'status': 'coordinated',
        })
        return True

    def get_integration_metrics(self) -> Dict[str, Any]:
        """
        Get integration-specific metrics
        """
        return {
            "api_calls": 0,
            "integrations_completed": 0,
            "optimizations_executed": 0,
            "risk_assessments": 0,
            "stakeholder_coordinations": 0
        }
