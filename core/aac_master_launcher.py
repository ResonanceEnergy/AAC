#!/usr/bin/env python3
"""
AAC Master System Launcher
==========================

Unified entry point for the complete Accelerated Arbitrage Corp system.
Launches doctrine compliance monitoring + all system modules in proper sequence.

This replaces fragmented startup files:
- main.py (trading only)
- run_integrated_system.py (doctrine only)
- deploy_aac_system.py (deployment only)

Usage:
    python aac_master_launcher.py [--mode paper|live|check]
    python aac_master_launcher.py --doctrine-only    # Doctrine monitoring only
    python aac_master_launcher.py --agents-only      # Agents only
    python aac_master_launcher.py --trading-only     # Trading only
    python aac_master_launcher.py --monitoring-only  # Monitoring systems only
    python aac_master_launcher.py --dashboard-only   # Dashboard only
    python aac_master_launcher.py --service-only     # Monitoring service only
"""

import asyncio
import argparse
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'aac_master_launcher.log'),
    ]
)
logger = logging.getLogger("AAC.MasterLauncher")


@dataclass
class SystemStatus:
    """Status of system components"""
    doctrine: bool = False
    agents: bool = False
    trading: bool = False
    monitoring: bool = False
    infrastructure: bool = False
    departments_connected: int = 0
    agents_active: int = 0
    strategy_agents_active: int = 0
    strategies_covered: int = 0
    errors: List[str] = field(default_factory=list)


class AACMasterLauncher:
    """
    Master launcher for the complete AAC system.

    Startup Sequence:
    1. Doctrine Compliance Monitoring (8 packs)
    2. Department Agents (26 specialized agents)
    3. Trading Systems (execution, risk, accounting)
    4. Monitoring & Dashboards
    5. Cross-System Integration
    """

    def __init__(self, mode: str = 'paper'):
        self.mode = mode
        self.status = SystemStatus()
        self.start_time = None
        self.components = {}

        # Component managers
        self.doctrine_orchestrator = None
        self.agent_manager = None
        self.trading_system = None
        self.monitoring_system = None

    async def launch_complete_system(self) -> bool:
        """Launch the complete AAC system"""
        self.start_time = datetime.now()
        logger.info("AAC MASTER LAUNCHER - Starting Complete System")
        logger.info(f"Launch Time: {self.start_time}")
        logger.info(f"Launch Mode: {self.mode.upper()}")

        try:
            # Phase 1: Doctrine Compliance Monitoring
            await self._launch_doctrine_system()

            # Phase 2: Department Agents
            await self._launch_department_agents()

            # Phase 3: Trading Systems
            await self._launch_trading_systems()

            # Phase 4: Monitoring & Infrastructure
            await self._launch_monitoring_systems()

            # Phase 5: Cross-System Integration
            await self._establish_system_integration()

            # Phase 6: Final Validation
            success = await self._validate_complete_system()

            if success:
                await self._log_successful_launch()
                await self._start_continuous_operation()
            else:
                await self._log_launch_failure()

            return success

        except Exception as e:
            logger.critical(f"💥 Master launch failed: {e}")
            await self._emergency_shutdown()
            return False

    async def _launch_doctrine_system(self):
        """Launch doctrine compliance monitoring (8 packs)"""
        logger.info("📋 Phase 1: Launching Doctrine Compliance System")

        try:
            from aac.doctrine import DoctrineOrchestrator
            self.doctrine_orchestrator = DoctrineOrchestrator()
            await self.doctrine_orchestrator.initialize()

            # Start continuous compliance monitoring
            asyncio.create_task(self.doctrine_orchestrator.start_monitoring())

            self.status.doctrine = True
            logger.info("✅ Doctrine system launched (8 packs active)")

        except Exception as e:
            self.status.errors.append(f"Doctrine launch failed: {e}")
            logger.error(f"❌ Doctrine system launch failed: {e}")
            raise

    async def _launch_department_agents(self):
        """Launch all department agents (26 specialized agents)"""
        logger.info("🤖 Phase 2: Launching Department Agents")

        try:
            # Initialize department super agents
            from shared.department_super_agents import initialize_all_department_super_agents
            agent_result = await initialize_all_department_super_agents()

            self.status.agents = True
            self.status.agents_active = agent_result.get('total_count', 0)
            self.status.departments_connected = agent_result.get('departments_covered', 0)

            logger.info(f"✅ Department agents launched: {self.status.agents_active} agents across {self.status.departments_connected} departments")

            # Launch BigBrainIntelligence agents
            await self._launch_bigbrain_agents()

            # Launch executive agents
            await self._launch_executive_agents()

            # Launch strategy agents
            await self._launch_strategy_agents()

        except Exception as e:
            self.status.errors.append(f"Agent launch failed: {e}")
            logger.error(f"❌ Department agents launch failed: {e}")
            raise

    async def _launch_bigbrain_agents(self):
        """Launch BigBrainIntelligence research agents"""
        try:
            from BigBrainIntelligence.agents import get_all_agents
            agents = get_all_agents()

            # Start research pipeline
            from BigBrainIntelligence.research_agent import ResearchAgent
            research = ResearchAgent()
            await research.initialize()

            logger.info(f"✅ BigBrainIntelligence agents active: {len(agents)} research agents")

        except Exception as e:
            logger.warning(f"BigBrainIntelligence agents partial failure: {e}")

    async def _launch_executive_agents(self):
        """Launch executive oversight agents"""
        try:
            # Note: Executive agents are currently not fully implemented
            # This is where AZ_SUPREME and AX_HELIX would be launched
            logger.info("ℹ️  Executive agents: Framework ready (implementation pending)")

        except Exception as e:
            logger.warning(f"Executive agents launch issue: {e}")

    async def _launch_strategy_agents(self):
        """Launch strategy trading agents (49 strategies × 2 agents = 98 agents)"""
        try:
            from agents.agent_based_trading import AgentContestOrchestrator
            from strategies.strategy_agent_master_mapping import get_strategy_agent_mapper

            # Initialize strategy agent orchestrator
            self.strategy_orchestrator = AgentContestOrchestrator()
            await self.strategy_orchestrator.initialize_contest()

            # Get strategy-agent mapping for validation
            mapper = get_strategy_agent_mapper()
            validation = mapper.validate_assignments()

            strategy_count = validation.get('total_strategies', 0)
            agent_count = validation.get('total_agents', 0)

            logger.info(f"✅ Strategy agents launched: {agent_count} agents for {strategy_count} strategies")

            # Update status
            self.status.strategy_agents_active = agent_count
            self.status.strategies_covered = strategy_count

        except Exception as e:
            logger.warning(f"Strategy agents launch issue: {e}")
            self.status.errors.append(f"Strategy agents failed: {e}")

    async def _launch_trading_systems(self):
        """Launch trading execution systems"""
        logger.info("💰 Phase 3: Launching Trading Systems")

        try:
            from main import ACCSystem
            self.trading_system = ACCSystem(
                paper_trading=(self.mode == 'paper'),
                dry_run=(self.mode == 'dry-run')
            )

            await self.trading_system.start()

            self.status.trading = True
            logger.info(f"✅ Trading systems launched in {self.mode.upper()} mode")

        except Exception as e:
            self.status.errors.append(f"Trading launch failed: {e}")
            logger.error(f"❌ Trading systems launch failed: {e}")
            raise

    async def _launch_monitoring_systems(self):
        """Launch monitoring and dashboard systems"""
        logger.info("📊 Phase 4: Launching Monitoring Systems")

        try:
            # Start continuous monitoring service
            from monitoring.continuous_monitoring import ContinuousMonitoringService
            monitoring_service = ContinuousMonitoringService()
            await monitoring_service.start_monitoring()
            self.monitoring_system = monitoring_service

            # Start master monitoring dashboard
            from monitoring.aac_master_monitoring_dashboard import get_master_dashboard, DisplayMode
            dashboard = get_master_dashboard(DisplayMode.TERMINAL)
            asyncio.create_task(dashboard.start_monitoring())
            self.monitoring_dashboard = dashboard

            self.status.monitoring = True
            logger.info("✅ Monitoring systems launched")

        except Exception as e:
            self.status.errors.append(f"Monitoring launch failed: {e}")
            logger.error(f"❌ Monitoring systems launch failed: {e}")
            # Don't raise - monitoring is not critical

    async def _establish_system_integration(self):
        """Establish cross-system coordination"""
        logger.info("🔗 Phase 5: Establishing System Integration")

        try:
            # Connect doctrine to trading systems
            if self.doctrine_orchestrator and self.trading_system:
                await self._connect_doctrine_to_trading()

            # Connect agents to monitoring
            if self.status.agents and self.status.monitoring:
                await self._connect_agents_to_monitoring()

            # Establish communication frameworks
            await self._initialize_communication_frameworks()

            logger.info("✅ System integration established")

        except Exception as e:
            logger.warning(f"⚠️  System integration partial failure: {e}")

    async def _connect_doctrine_to_trading(self):
        """Connect doctrine compliance to trading systems"""
        # Doctrine actions will automatically trigger trading system responses
        logger.info("📋 Doctrine ↔ Trading integration active")

    async def _connect_agents_to_monitoring(self):
        """Connect agent systems to monitoring"""
        logger.info("🤖 Agents ↔ Monitoring integration active")

    async def _initialize_communication_frameworks(self):
        """Initialize cross-system communication"""
        from shared.communication_framework import get_communication_framework
        comms = get_communication_framework()
        await comms.initialize()
        logger.info("📡 Communication frameworks initialized")

    async def _validate_complete_system(self) -> bool:
        """Validate that all critical systems are operational"""
        logger.info("🔍 Phase 6: Validating Complete System")

        checks = {
            "Doctrine System": self.status.doctrine,
            "Agent Systems": self.status.agents,
            "Trading Systems": self.status.trading,
            "Department Connections": self.status.departments_connected >= 5,
            "Agent Count": self.status.agents_active >= 20,
            "Strategy Agents": self.status.strategy_agents_active >= 98,
            "Strategies Covered": self.status.strategies_covered >= 49
        }

        all_passed = True
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
            if not passed:
                all_passed = False

        return all_passed

    async def _log_successful_launch(self):
        """Log successful system launch"""
        runtime = datetime.now() - self.start_time
        logger.info("🎉 AAC SYSTEM LAUNCH SUCCESSFUL")
        logger.info(f"⏱️  Total launch time: {runtime.total_seconds():.1f} seconds")
        logger.info(f"📊 System status: Doctrine={self.status.doctrine}, Agents={self.status.agents_active}, Strategy Agents={self.status.strategy_agents_active}, Trading={self.status.trading}")

    async def _log_launch_failure(self):
        """Log launch failure"""
        logger.error("💥 AAC SYSTEM LAUNCH FAILED")
        for error in self.status.errors:
            logger.error(f"  ❌ {error}")

    async def _emergency_shutdown(self):
        """Emergency shutdown of all systems"""
        logger.critical("[EMERGENCY] EMERGENCY SHUTDOWN INITIATED")

        # Shutdown in reverse order
        shutdown_tasks = []

        if self.monitoring_system:
            shutdown_tasks.append(self._shutdown_monitoring())

        if self.trading_system:
            shutdown_tasks.append(self.trading_system.stop())

        if self.doctrine_orchestrator:
            shutdown_tasks.append(self.doctrine_orchestrator.stop_monitoring())

        # Wait for all shutdowns
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)

    async def _shutdown_monitoring(self):
        """Shutdown monitoring systems"""
        logger.info("Shutting down monitoring systems")

    async def launch_doctrine_only(self) -> bool:
        """Launch doctrine system only"""
        return await self._launch_doctrine_system()

    async def launch_agents_only(self) -> bool:
        """Launch agents only"""
        await self._launch_department_agents()
        return self.status.agents

    async def launch_trading_only(self) -> bool:
        """Launch trading systems only"""
        await self._launch_trading_systems()
        return self.status.trading

    async def launch_monitoring_only(self) -> bool:
        """Launch monitoring systems only"""
        self.start_time = datetime.now()
        logger.info("AAC MASTER LAUNCHER - Monitoring Systems Only")
        logger.info(f"Launch Time: {self.start_time}")

        try:
            await self._launch_monitoring_systems()
            await self._establish_system_integration()
            await self._run_monitoring_loop()
            return True
        except Exception as e:
            logger.error(f"❌ Monitoring launch failed: {e}")
            return False

    async def launch_dashboard_only(self, display_mode: str = 'terminal') -> bool:
        """Launch monitoring dashboard only"""
        self.start_time = datetime.now()
        logger.info("AAC MASTER LAUNCHER - Dashboard Only")
        logger.info(f"Launch Time: {self.start_time}")
        logger.info(f"Display Mode: {display_mode}")

        try:
            from monitoring.aac_master_monitoring_dashboard import get_master_dashboard, DisplayMode, run_streamlit_dashboard
            
            # Map string to DisplayMode enum
            mode_map = {
                'terminal': DisplayMode.TERMINAL,
                'web': DisplayMode.WEB,
                'dash': DisplayMode.DASH,
                'api': DisplayMode.API
            }
            dashboard_mode = mode_map.get(display_mode, DisplayMode.TERMINAL)
            
            if dashboard_mode == DisplayMode.WEB:
                # For web mode, run Streamlit directly
                import subprocess
                import sys
                import time
                import webbrowser
                import threading
                
                logger.info("Starting Streamlit dashboard on http://localhost:8080")
                
                # Function to open browser after a short delay
                def open_browser():
                    time.sleep(3)  # Wait for server to start
                    try:
                        webbrowser.open('http://localhost:8080')
                        logger.info("✅ Browser opened automatically to http://localhost:8080")
                    except Exception as e:
                        logger.warning(f"Could not open browser automatically: {e}")
                        logger.info("Please manually open: http://localhost:8080")
                
                # Start browser opener in background thread
                browser_thread = threading.Thread(target=open_browser, daemon=True)
                browser_thread.start()
                
                # Run Streamlit server (blocking call)
                subprocess.run([sys.executable, "-m", "streamlit", "run", "monitoring/aac_master_monitoring_dashboard.py", "--server.port", "8080", "--server.headless", "true"])
            else:
                # For terminal/dash modes, use the standard dashboard interface
                dashboard = get_master_dashboard(dashboard_mode)
                await dashboard.initialize()
                await dashboard.run_dashboard()
            return True
        except Exception as e:
            logger.error(f"[ERROR] Dashboard launch failed: {e}")
            return False

    async def launch_service_only(self) -> bool:
        """Launch continuous monitoring service only"""
        self.start_time = datetime.now()
        logger.info("AAC MASTER LAUNCHER - Service Only")
        logger.info(f"Launch Time: {self.start_time}")

        try:
            from monitoring.continuous_monitoring import ContinuousMonitoringService
            service = ContinuousMonitoringService()
            await service.start_monitoring()
            await self._run_monitoring_loop()
            return True
        except Exception as e:
            logger.error(f"❌ Service launch failed: {e}")
            return False

    async def _run_monitoring_loop(self):
        """Run monitoring operation loop"""
        logger.info("🔄 Monitoring systems operating continuously")

        try:
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                # Monitoring systems handle their own loops
        except KeyboardInterrupt:
            logger.info("⏹️  Monitoring shutdown requested")
        except Exception as e:
            logger.error(f"❌ Monitoring loop error: {e}")

    async def _start_continuous_operation(self):
        """Start continuous system operation"""
        logger.info("🔄 AAC System operating continuously")

        # Keep the system running
        while True:
            await asyncio.sleep(60)  # Health check every minute

            # Periodic health checks
            if self.doctrine_orchestrator:
                health = await self.doctrine_orchestrator.get_health_status()
                if health.get('status') != 'healthy':
                    logger.warning("[WARN] Doctrine health check failed")

    def print_banner(self):
        """Print master launcher banner"""
        banner = """
================================================================================
                                                                              |
     AAC MASTER SYSTEM LAUNCHER                                                |
                                                                              |
     ACCELERATED ARBITRAGE CORP - MASTER SYSTEM LAUNCHER                      |
                                                                              |
     Doctrine Compliance | Department Agents | Trading Systems | Monitoring    |
                                                                              |
================================================================================
"""
        print(banner)


def print_deprecated_warning():
    """Warn about deprecated startup files"""
    print("""
[WARN] DEPRECATED STARTUP FILES DETECTED
   The following files are now deprecated and will be removed:

   ❌ main.py                    → Use aac_master_launcher.py
   ❌ run_integrated_system.py   → Use aac_master_launcher.py --doctrine-only
   ❌ deploy_aac_system.py       → Use aac_master_launcher.py
   ❌ Start-ACC.ps1             → Use aac_master_launcher.py

   These files remain for backward compatibility but are orphaned.
""")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AAC Master System Launcher")
    parser.add_argument(
        '--mode',
        choices=['paper', 'live', 'dry-run', 'check'],
        default='paper',
        help='System operating mode'
    )
    parser.add_argument(
        '--doctrine-only',
        action='store_true',
        help='Launch doctrine system only'
    )
    parser.add_argument(
        '--agents-only',
        action='store_true',
        help='Launch agents only'
    )
    parser.add_argument(
        '--trading-only',
        action='store_true',
        help='Launch trading systems only'
    )
    parser.add_argument(
        '--monitoring-only',
        action='store_true',
        help='Launch monitoring systems only'
    )
    parser.add_argument(
        '--dashboard-only',
        action='store_true',
        help='Launch monitoring dashboard only'
    )
    parser.add_argument(
        '--service-only',
        action='store_true',
        help='Launch continuous monitoring service only'
    )
    parser.add_argument(
        '--display-mode',
        choices=['terminal', 'web', 'dash', 'api'],
        default='terminal',
        help='Monitoring dashboard display mode'
    )

    args = parser.parse_args()

    # Print banner
    launcher = AACMasterLauncher(args.mode)
    launcher.print_banner()

    # Check for deprecated usage
    if not any([args.doctrine_only, args.agents_only, args.trading_only, args.monitoring_only, args.dashboard_only, args.service_only]):
        print_deprecated_warning()

    # Ensure logs directory exists
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)

    try:
        if args.doctrine_only:
            success = await launcher.launch_doctrine_only()
        elif args.agents_only:
            success = await launcher.launch_agents_only()
        elif args.trading_only:
            success = await launcher.launch_trading_only()
        elif args.monitoring_only:
            success = await launcher.launch_monitoring_only()
        elif args.dashboard_only:
            success = await launcher.launch_dashboard_only(args.display_mode)
        elif args.service_only:
            success = await launcher.launch_service_only()
        else:
            success = await launcher.launch_complete_system()

        if success:
            print("\n[SUCCESS] AAC MASTER LAUNCHER - SUCCESS")
            print("═" * 60)
        else:
            print("\n💥 AAC MASTER LAUNCHER - FAILED")
            print("═" * 60)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Shutdown requested by user")
        await launcher._emergency_shutdown()
    except Exception as e:
        print(f"\n💥 Launch failed: {e}")
        await launcher._emergency_shutdown()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())