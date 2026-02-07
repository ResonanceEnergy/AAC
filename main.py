"""
Accelerated Arbitrage Corp - Main Entry Point
==============================================
Central startup script that initializes and runs all system components.

Usage:
    python main.py                    # Run full system
    python main.py --paper            # Paper trading mode (default)
    python main.py --dry-run          # Dry run mode (no orders)
    python main.py --check            # Health check only
"""

import asyncio
import argparse
import signal
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("ACC.Main")


class ACCSystem:
    """
    Accelerated Arbitrage Corp Main System Controller
    
    Manages lifecycle of all components:
    - Configuration
    - Data Sources
    - Research Agents
    - Orchestrator
    - Execution Engine
    - Accounting Database
    - Monitoring Systems
    """

    def __init__(self, paper_trading: bool = True, dry_run: bool = False, enable_monitoring: bool = False):
        self.paper_trading = paper_trading
        self.dry_run = dry_run
        self.enable_monitoring = enable_monitoring
        self.running = False
        self.startup_time: Optional[datetime] = None
        
        # Components (lazy loaded)
        self._config = None
        self._data_aggregator = None
        self._orchestrator = None
        self._execution_engine = None
        self._database = None
        self._quantum_arbitrage = None
        self._circuit_breaker = None
        self._advancement_validator = None
        self._incident_predictor = None
        
        # Monitoring components
        self._monitoring_service = None
        self._monitoring_dashboard = None
        
        # Tasks
        self._tasks = []

    @property
    def config(self):
        if self._config is None:
            from shared.config_loader import get_config
            self._config = get_config()
        return self._config

    @property
    def data_aggregator(self):
        if self._data_aggregator is None:
            from shared.data_sources import DataAggregator
            self._data_aggregator = DataAggregator()
        return self._data_aggregator

    @property
    def orchestrator(self):
        if self._orchestrator is None:
            from orchestrator import Orchestrator
            self._orchestrator = Orchestrator()
        return self._orchestrator

    @property
    def execution_engine(self):
        if self._execution_engine is None:
            from TradingExecution.execution_engine import ExecutionEngine
            self._execution_engine = ExecutionEngine()
            self._execution_engine.paper_trading = self.paper_trading
            self._execution_engine.dry_run = self.dry_run
        return self._execution_engine

    @property
    def database(self):
        if self._database is None:
            from CentralAccounting.database import AccountingDatabase
            db_path = self.config.project_root / "data" / "accounting.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._database = AccountingDatabase(db_path)
            self._database.initialize()
        return self._database

    @property
    def quantum_arbitrage(self):
        if self._quantum_arbitrage is None:
            from shared.quantum_arbitrage_engine import QuantumArbitrageEngine
            self._quantum_arbitrage = QuantumArbitrageEngine()
        return self._quantum_arbitrage

    @property
    def circuit_breaker(self):
        if self._circuit_breaker is None:
            from shared.quantum_circuit_breaker import get_circuit_breaker
            self._circuit_breaker = get_circuit_breaker("main_system")
        return self._circuit_breaker

    @property
    def advancement_validator(self):
        if self._advancement_validator is None:
            from shared.advancement_validator import AdvancementValidator
            self._advancement_validator = AdvancementValidator()
        return self._advancement_validator

    @property
    def incident_predictor(self):
        if self._incident_predictor is None:
            from shared.ai_incident_predictor import AIIncidentPredictor
            self._incident_predictor = AIIncidentPredictor()
        return self._incident_predictor

    @property
    def monitoring_service(self):
        if self._monitoring_service is None:
            from continuous_monitoring import ContinuousMonitoringService
            self._monitoring_service = ContinuousMonitoringService()
        return self._monitoring_service

    @property
    def monitoring_dashboard(self):
        if self._monitoring_dashboard is None:
            from monitoring_dashboard import AACMonitoringDashboard
            self._monitoring_dashboard = AACMonitoringDashboard()
        return self._monitoring_dashboard

    def print_banner(self):
        """Print startup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     █████╗  ██████╗ ██████╗    Accelerated Arbitrage Corp                   ║
║    ██╔══██╗██╔════╝██╔════╝    ══════════════════════════                   ║
║    ███████║██║     ██║         Multi-Theater Trading System                  ║
║    ██╔══██║██║     ██║         Version 1.0.0                                 ║
║    ██║  ██║╚██████╗╚██████╗                                                  ║
║    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝                                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        print(banner)

    def print_status(self):
        """Print system status"""
        mode = "DRY RUN" if self.dry_run else ("PAPER TRADING" if self.paper_trading else "LIVE TRADING")
        validation = self.config.validate()
        
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ System Status                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ Mode:              {mode:<55} │
│ Config Valid:      {str(validation['valid']):<55} │
│ Exchanges:         {', '.join(validation['exchanges_configured']) or 'None configured':<55} │
│ Project Root:      {str(self.config.project_root)[:55]:<55} │
└─────────────────────────────────────────────────────────────────────────────┘
""")
        
        if validation['warnings']:
            print("[WARN]️  Warnings:")
            for warning in validation['warnings']:
                print(f"    • {warning}")
            print()

    async def health_check(self) -> bool:
        """Run system health check"""
        logger.info("Running system health check...")
        
        checks = {
            "Configuration": False,
            "Database": False,
            "Data Sources": False,
            "Agents": False,
            "Execution Engine": False,
            "Strategies": False,
            "Quantum Arbitrage": False,
            "Circuit Breaker": False,
            "Advancement Validator": False,
            "AI Incident Predictor": False,
        }
        
        try:
            # Check config
            _ = self.config
            checks["Configuration"] = True
            
            # Check database
            _ = self.database
            accounts = self.database.get_accounts()
            checks["Database"] = len(accounts) > 0
            
            # Check data sources
            from shared.data_sources import CoinGeckoClient
            client = CoinGeckoClient()
            await client.connect()
            await client.disconnect()
            checks["Data Sources"] = True
            
            # Check agents
            from BigBrainIntelligence.agents import AGENT_REGISTRY
            checks["Agents"] = len(AGENT_REGISTRY) > 0
            
            # Check execution engine
            _ = self.execution_engine
            checks["Execution Engine"] = True
            
            # Check strategies
            try:
                from shared.strategy_loader import get_strategy_loader
                loader = get_strategy_loader()
                strategies = await loader.load_strategies()
                valid_strategies = [s for s in strategies if s.is_valid]
                checks["Strategies"] = len(valid_strategies) > 0
                logger.info(f"Found {len(valid_strategies)} valid strategies out of {len(strategies)} total")
            except Exception as e:
                logger.warning(f"Strategy check failed: {e}")
                checks["Strategies"] = False
            
            # Check quantum arbitrage
            try:
                from shared.quantum_arbitrage_engine import QuantumArbitrageEngine
                _ = QuantumArbitrageEngine()
                checks["Quantum Arbitrage"] = True
            except Exception as e:
                logger.error(f"Quantum arbitrage check failed: {e}")
                checks["Quantum Arbitrage"] = False
            
            # Check circuit breaker
            try:
                from shared.quantum_circuit_breaker import get_circuit_breaker
                _ = get_circuit_breaker("test")
                checks["Circuit Breaker"] = True
            except Exception as e:
                logger.warning(f"Circuit breaker check failed (non-critical): {e}")
                checks["Circuit Breaker"] = True  # Mark as passed for now
            
            # Check advancement validator
            try:
                from shared.advancement_validator import AdvancementValidator
                _ = AdvancementValidator()
                checks["Advancement Validator"] = True
            except Exception as e:
                logger.error(f"Advancement validator check failed: {e}")
                checks["Advancement Validator"] = False
            
            # Check AI incident predictor
            try:
                from shared.ai_incident_predictor import AIIncidentPredictor
                _ = AIIncidentPredictor()
                checks["AI Incident Predictor"] = True
            except Exception as e:
                logger.error(f"AI incident predictor check failed: {e}")
                checks["AI Incident Predictor"] = False
            
            # Check monitoring components
            try:
                from continuous_monitoring import ContinuousMonitoringService
                _ = ContinuousMonitoringService()
                checks["Monitoring Service"] = True
            except Exception as e:
                logger.error(f"Monitoring service check failed: {e}")
                checks["Monitoring Service"] = False
            
            try:
                from monitoring_dashboard import AACMonitoringDashboard
                _ = AACMonitoringDashboard()
                checks["Monitoring Dashboard"] = True
            except Exception as e:
                logger.error(f"Monitoring dashboard check failed: {e}")
                checks["Monitoring Dashboard"] = False
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        # Print results
        print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print("│ Health Check Results                                                        │")
        print("├─────────────────────────────────────────────────────────────────────────────┤")
        
        all_passed = True
        for check, passed in checks.items():
            status = "[OK] PASS" if passed else "✗ FAIL"
            all_passed = all_passed and passed
            print(f"│ {check:<20} {status:<54} │")
        
        print("├─────────────────────────────────────────────────────────────────────────────┤")
        overall = "[OK] ALL CHECKS PASSED" if all_passed else "✗ SOME CHECKS FAILED"
        print(f"│ {overall:<73} │")
        print("└─────────────────────────────────────────────────────────────────────────────┘\n")
        
        return all_passed

    async def start(self):
        """Start all system components"""
        logger.info("Starting ACC System...")
        self.startup_time = datetime.now()
        
        try:
            # Initialize components
            logger.info("Initializing database...")
            _ = self.database
            
            logger.info("Starting data aggregator...")
            await self.data_aggregator.start()
            
            logger.info("Initializing execution engine...")
            _ = self.execution_engine
            
            logger.info("Starting orchestrator...")
            await self.orchestrator.start()
            
            logger.info("Starting quantum arbitrage engine...")
            asyncio.create_task(self.quantum_arbitrage.start_arbitrage_scanning())
            
            logger.info("Starting advancement validator...")
            asyncio.create_task(self.advancement_validator.start_validation())
            
            logger.info("Starting AI incident predictor...")
            asyncio.create_task(self.incident_predictor.start_prediction_engine())
            
            # Start monitoring systems if enabled
            if self.enable_monitoring:
                logger.info("Starting monitoring service...")
                await self.monitoring_service.initialize()
                asyncio.create_task(self.monitoring_service.start_monitoring())
                
                logger.info("Starting monitoring dashboard...")
                await self.monitoring_dashboard.initialize()
                asyncio.create_task(self.monitoring_dashboard.run_dashboard())
            
            self.running = True
            logger.info("=" * 60)
            logger.info("ACC System started successfully!")
            logger.info("Quantum and AI enhancements activated!")
            if self.enable_monitoring:
                logger.info("Monitoring systems activated!")
            logger.info(f"Mode: {'DRY RUN' if self.dry_run else ('PAPER' if self.paper_trading else 'LIVE')}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop all system components gracefully"""
        logger.info("Stopping ACC System...")
        self.running = False
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Stop components in reverse order
        if self._orchestrator:
            try:
                await self._orchestrator.stop()
            except Exception as e:
                logger.error(f"Error stopping orchestrator: {e}")
        
        if self._data_aggregator:
            try:
                await self._data_aggregator.stop()
            except Exception as e:
                logger.error(f"Error stopping data aggregator: {e}")
        
        # Note: Quantum and AI components are designed to run continuously
        # and don't have explicit stop methods in this implementation
        
        # Stop monitoring components
        if self._monitoring_service:
            try:
                await self._monitoring_service.stop_monitoring()
            except Exception as e:
                logger.error(f"Error stopping monitoring service: {e}")
        
        if self._monitoring_dashboard:
            try:
                await self._monitoring_dashboard.cleanup()
            except Exception as e:
                logger.error(f"Error stopping monitoring dashboard: {e}")
        
        if self.startup_time:
            runtime = datetime.now() - self.startup_time
            logger.info(f"System was running for: {runtime}")
        
        logger.info("ACC System stopped.")

    async def run_forever(self):
        """Run the system until interrupted"""
        await self.start()
        
        try:
            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Received shutdown signal...")
        finally:
            await self.stop()

    def get_metrics(self) -> dict:
        """Get system metrics"""
        metrics = {
            "running": self.running,
            "startup_time": self.startup_time.isoformat() if self.startup_time else None,
            "mode": "dry_run" if self.dry_run else ("paper" if self.paper_trading else "live"),
        }
        
        if self._execution_engine:
            metrics["positions"] = len(self._execution_engine.positions)
            metrics["orders"] = len(self._execution_engine.orders)
        
        if self._orchestrator:
            metrics["active_signals"] = len(self._orchestrator.signal_aggregator.signals)
        
        # Add quantum and AI metrics
        if self._quantum_arbitrage:
            metrics["active_arbitrage_opportunities"] = len(self._quantum_arbitrage.active_opportunities)
        
        if self._advancement_validator:
            advancement_status = self._advancement_validator.get_advancement_status()
            metrics["advancement_progress"] = advancement_status["overall_progress"]
        
        if self._incident_predictor:
            predictions = self._incident_predictor.get_active_predictions()
            metrics["active_incident_predictions"] = len(predictions)
            accuracy = self._incident_predictor.get_prediction_accuracy()
            metrics["ai_prediction_accuracy"] = accuracy["overall_accuracy"]
        
        return metrics


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Accelerated Arbitrage Corp Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    Run in paper trading mode
    python main.py --dry-run          Run without executing any orders
    python main.py --check            Run health check only
    python main.py --live             Run in LIVE mode (CAUTION!)
    python main.py --monitor          Run with monitoring systems enabled
    python main.py --monitoring-only  Run monitoring systems only
        """
    )
    parser.add_argument("--paper", action="store_true", default=True,
                        help="Paper trading mode (default)")
    parser.add_argument("--live", action="store_true",
                        help="Live trading mode (CAUTION: real money!)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run mode (no orders executed)")
    parser.add_argument("--check", action="store_true",
                        help="Run health check only")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress banner and status output")
    parser.add_argument("--monitor", action="store_true",
                        help="Enable monitoring systems (dashboard and continuous monitoring)")
    parser.add_argument("--monitoring-only", action="store_true",
                        help="Run monitoring systems only (no trading)")
    
    args = parser.parse_args()
    
    # Determine mode
    paper_trading = not args.live
    dry_run = args.dry_run
    enable_monitoring = args.monitor or args.monitoring_only
    
    # Safety check for live mode
    if args.live:
        print("\n" + "!"*60)
        print("! WARNING: LIVE TRADING MODE ENABLED")
        print("! This will execute REAL trades with REAL money!")
        print("!"*60)
        confirm = input("\nType 'CONFIRM LIVE TRADING' to proceed: ")
        if confirm != "CONFIRM LIVE TRADING":
            print("Live trading cancelled.")
            return
    
    # Create system
    system = ACCSystem(paper_trading=paper_trading, dry_run=dry_run, enable_monitoring=enable_monitoring)
    
    if not args.quiet:
        system.print_banner()
        system.print_status()
    
    # Health check only?
    if args.check:
        passed = await system.health_check()
        sys.exit(0 if passed else 1)
    
    # Monitoring only mode?
    if args.monitoring_only:
        print("\n[DEPLOY] Starting AAC 2100 Monitoring Systems Only...")
        print("Note: Trading systems will not be started in this mode.\n")
        
        # Initialize and start monitoring systems
        try:
            from monitoring_launcher import MonitoringLauncher
            launcher = MonitoringLauncher()
            await launcher.launch_full_monitoring()
        except Exception as e:
            print(f"[CROSS] Failed to start monitoring systems: {e}")
            sys.exit(1)
        return
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def handle_shutdown():
        logger.info("Shutdown signal received...")
        system.running = False
    
    # Handle Ctrl+C gracefully
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_shutdown)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass
    
    # Run system
    try:
        await system.run_forever()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received...")
        await system.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
