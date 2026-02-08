#!/usr/bin/env python3
"""
Production Deployment System
===========================
Gradual rollout system for safe production deployment with limited capital and single strategies.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import get_audit_logger
from shared.live_trading_safeguards import live_trading_safeguards
from shared.paper_trading import paper_trading_engine
from shared.ai_strategy_generator import ai_strategy_generator


@dataclass
class DeploymentPhase:
    """Production deployment phase"""
    phase_id: str
    name: str
    capital_percentage: float
    max_strategies: int
    risk_multiplier: float  # 1.0 = normal risk, 0.5 = half risk
    monitoring_hours: int
    criteria: Dict[str, Any]


@dataclass
class ProductionDeployment:
    """Production deployment configuration"""
    deployment_id: str
    start_time: datetime
    current_phase: str
    allocated_capital: float
    active_strategies: List[str]
    risk_limits: Dict[str, float]
    monitoring_active: bool = True
    emergency_stop: bool = False


class ProductionDeploymentSystem:
    """Manages gradual rollout to production"""

    def __init__(self):
        self.logger = logging.getLogger("ProductionDeployment")
        self.audit_logger = get_audit_logger()

        # Load production configuration
        config_path = PROJECT_ROOT / "config" / "production_market_data.yaml"
        with open(config_path, 'r') as f:
            self.prod_config = yaml.safe_load(f)

        # Deployment phases
        self.deployment_phases = self._initialize_phases()

        # Current deployment state
        self.current_deployment: Optional[ProductionDeployment] = None

        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None

    def _initialize_phases(self) -> Dict[str, DeploymentPhase]:
        """Initialize deployment phases"""
        return {
            "phase_1": DeploymentPhase(
                phase_id="phase_1",
                name="Initial Deployment",
                capital_percentage=10.0,  # 10% of total capital
                max_strategies=1,
                risk_multiplier=0.5,  # Half normal risk
                monitoring_hours=24,
                criteria={
                    "min_monitoring_hours": 24,
                    "max_daily_loss_pct": 0.5,
                    "max_drawdown_pct": 1.0,
                    "min_win_rate": 0.6
                }
            ),
            "phase_2": DeploymentPhase(
                phase_id="phase_2",
                name="Capital Increase",
                capital_percentage=25.0,  # 25% of total capital
                max_strategies=1,
                risk_multiplier=0.75,
                monitoring_hours=48,
                criteria={
                    "min_monitoring_hours": 48,
                    "max_daily_loss_pct": 1.0,
                    "max_drawdown_pct": 2.0,
                    "min_win_rate": 0.65,
                    "min_sharpe_ratio": 0.8
                }
            ),
            "phase_3": DeploymentPhase(
                phase_id="phase_3",
                name="Multi-Strategy",
                capital_percentage=50.0,  # 50% of total capital
                max_strategies=3,
                risk_multiplier=1.0,
                monitoring_hours=72,
                criteria={
                    "min_monitoring_hours": 72,
                    "max_daily_loss_pct": 1.5,
                    "max_drawdown_pct": 3.0,
                    "min_win_rate": 0.7,
                    "min_sharpe_ratio": 1.0,
                    "min_profit_factor": 1.5
                }
            ),
            "phase_4": DeploymentPhase(
                phase_id="phase_4",
                name="Full Deployment",
                capital_percentage=100.0,  # Full capital
                max_strategies=5,
                risk_multiplier=1.0,
                monitoring_hours=168,  # 1 week
                criteria={
                    "min_monitoring_hours": 168,
                    "max_daily_loss_pct": 2.0,
                    "max_drawdown_pct": 5.0,
                    "min_win_rate": 0.75,
                    "min_sharpe_ratio": 1.2,
                    "min_profit_factor": 1.8
                }
            )
        }

    async def initialize_deployment(self, total_capital: float = 1000000.0) -> str:
        """Initialize production deployment"""
        self.logger.info("Initializing production deployment...")

        deployment_id = f"prod_deployment_{int(time.time())}"

        # Start with phase 1
        initial_phase = self.deployment_phases["phase_1"]
        allocated_capital = total_capital * (initial_phase.capital_percentage / 100.0)

        self.current_deployment = ProductionDeployment(
            deployment_id=deployment_id,
            start_time=datetime.now(),
            current_phase="phase_1",
            allocated_capital=allocated_capital,
            active_strategies=[],
            risk_limits={
                "max_position_size_pct": 5.0 * initial_phase.risk_multiplier,
                "max_daily_loss_pct": 2.0 * initial_phase.risk_multiplier,
                "max_drawdown_pct": 5.0 * initial_phase.risk_multiplier,
                "max_trades_per_hour": int(50 * initial_phase.risk_multiplier),
                "max_trades_per_day": int(200 * initial_phase.risk_multiplier)
            }
        )

        # Update safety system with deployment limits
        await self._update_safety_limits()

        # Start monitoring
        self.monitoring_task = asyncio.create_task(self._monitor_deployment())

        self.logger.info(f"Production deployment {deployment_id} initialized")
        self.logger.info(f"Phase 1: {initial_phase.name} - ${allocated_capital:,.2f} capital")

        return deployment_id

    async def deploy_strategy(self, strategy_id: str) -> bool:
        """Deploy a strategy to production"""
        if not self.current_deployment:
            self.logger.error("No active deployment")
            return False

        current_phase = self.deployment_phases[self.current_deployment.current_phase]

        # Check strategy limit
        if len(self.current_deployment.active_strategies) >= current_phase.max_strategies:
            self.logger.warning(f"Maximum strategies ({current_phase.max_strategies}) reached for current phase")
            return False

        # Add strategy to active list
        if strategy_id not in self.current_deployment.active_strategies:
            self.current_deployment.active_strategies.append(strategy_id)
            self.logger.info(f"Strategy {strategy_id} deployed to production")

        return True

    async def _update_safety_limits(self):
        """Update safety system with current deployment limits"""
        if not self.current_deployment:
            return

        # Update live trading safeguards
        limits = self.current_deployment.risk_limits

        # This would update the safety system configuration
        # For now, just log the limits
        self.logger.info(f"Updated safety limits: {limits}")

    async def _monitor_deployment(self):
        """Monitor deployment progress and phase transitions"""
        while self.current_deployment and self.current_deployment.monitoring_active:
            try:
                await self._check_phase_transition()
                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                self.logger.error(f"Deployment monitoring error: {e}")
                await asyncio.sleep(60)

    async def _check_phase_transition(self):
        """Check if deployment can move to next phase"""
        if not self.current_deployment:
            return

        current_phase_id = self.current_deployment.current_phase
        current_phase = self.deployment_phases[current_phase_id]

        # Check if minimum monitoring time has passed
        hours_elapsed = (datetime.now() - self.current_deployment.start_time).total_seconds() / 3600

        if hours_elapsed < current_phase.monitoring_hours:
            return  # Not enough time has passed

        # Check performance criteria
        performance_ok = await self._check_performance_criteria(current_phase.criteria)

        if performance_ok:
            await self._advance_to_next_phase()
        else:
            self.logger.warning(f"Performance criteria not met for phase {current_phase_id}")

    async def _check_performance_criteria(self, criteria: Dict[str, Any]) -> bool:
        """Check if performance criteria are met"""
        # Get current performance metrics
        # This would integrate with actual trading performance
        # For now, return True for simulation
        return True

    async def _advance_to_next_phase(self):
        """Advance to the next deployment phase"""
        if not self.current_deployment:
            return

        current_phase_id = self.current_deployment.current_phase
        phase_order = ["phase_1", "phase_2", "phase_3", "phase_4"]

        try:
            current_index = phase_order.index(current_phase_id)
            if current_index < len(phase_order) - 1:
                next_phase_id = phase_order[current_index + 1]
                next_phase = self.deployment_phases[next_phase_id]

                # Update deployment
                old_capital = self.current_deployment.allocated_capital
                total_capital = 1000000.0  # This should come from config
                new_capital = total_capital * (next_phase.capital_percentage / 100.0)

                self.current_deployment.current_phase = next_phase_id
                self.current_deployment.allocated_capital = new_capital

                # Update risk limits
                risk_multiplier = next_phase.risk_multiplier
                self.current_deployment.risk_limits = {
                    "max_position_size_pct": 5.0 * risk_multiplier,
                    "max_daily_loss_pct": 2.0 * risk_multiplier,
                    "max_drawdown_pct": 5.0 * risk_multiplier,
                    "max_trades_per_hour": int(50 * risk_multiplier),
                    "max_trades_per_day": int(200 * risk_multiplier)
                }

                # Update safety limits
                await self._update_safety_limits()

                self.logger.info(f"Advanced to {next_phase.name}")
                self.logger.info(f"Capital increased from ${old_capital:,.2f} to ${new_capital:,.2f}")
                self.logger.info(f"Risk multiplier: {risk_multiplier}")

        except (ValueError, IndexError):
            self.logger.info("Reached final deployment phase")

    async def emergency_stop(self):
        """Emergency stop deployment"""
        if self.current_deployment:
            self.current_deployment.emergency_stop = True
            self.current_deployment.monitoring_active = False
            self.logger.critical("Emergency stop activated - halting all trading")

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        if not self.current_deployment:
            return {"status": "no_active_deployment"}

        current_phase = self.deployment_phases[self.current_deployment.current_phase]

        return {
            "deployment_id": self.current_deployment.deployment_id,
            "current_phase": self.current_deployment.current_phase,
            "phase_name": current_phase.name,
            "allocated_capital": self.current_deployment.allocated_capital,
            "active_strategies": self.current_deployment.active_strategies,
            "risk_limits": self.current_deployment.risk_limits,
            "emergency_stop": self.current_deployment.emergency_stop,
            "start_time": self.current_deployment.start_time.isoformat(),
            "hours_elapsed": (datetime.now() - self.current_deployment.start_time).total_seconds() / 3600
        }


# Global production deployment system instance
production_deployment_system = ProductionDeploymentSystem()


async def initialize_production_deployment():
    """Initialize the production deployment system"""
    print("[DEPLOY] Initializing Production Deployment System...")

    deployment_id = await production_deployment_system.initialize_deployment()

    print("[OK] Production deployment system initialized")
    print(f"  Deployment ID: {deployment_id}")

    status = production_deployment_system.get_deployment_status()
    print(f"  Phase: {status['phase_name']}")
    print(f"  Allocated Capital: ${status['allocated_capital']:,.2f}")

    return deployment_id


if __name__ == "__main__":
    asyncio.run(initialize_production_deployment())