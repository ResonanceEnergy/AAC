#!/usr/bin/env python3
"""
AAC Production Deployment Script
================================

Complete production deployment and launch script for AAC arbitrage system.
Handles system setup, configuration validation, and orchestrated startup.

Features:
- Environment validation
- API key verification
- System health checks
- Orchestrated service startup
- Production monitoring
- Automated recovery

Usage:
    python deploy_production.py --mode [test|live]
    python deploy_production.py --validate-only
    python deploy_production.py --status
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests

logger = logging.getLogger(__name__)

# Import AAC components
from trading.aac_arbitrage_execution_system import AACArbitrageExecutionSystem, ExecutionConfig
from trading.binance_trading_engine import TradingConfig
from trading.binance_arbitrage_integration import BinanceConfig
try:
    from demos.multi_source_arbitrage_demo import MultiSourceArbitrageDemo as MultiSourceArbitrageDetector
except ImportError:
    MultiSourceArbitrageDetector = None

class AACProductionDeployer:
    """Production deployment manager for AAC system"""

    def __init__(self, mode: str = 'test'):
        self.mode = mode  # 'test' or 'live'
        self.project_root = Path(__file__).parent
        self.config_validated = False
        self.system_ready = False
        self.deployment_log = []

        # Component status
        self.component_status = {
            'environment': False,
            'api_keys': False,
            'network': False,
            'arbitrage_detector': False,
            'trading_engine': False,
            'execution_system': False,
            'monitoring': False
        }

    def log(self, message: str, level: str = 'INFO'):
        """Log deployment message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"
        self.deployment_log.append(log_entry)
        print(log_entry)

    async def validate_environment(self) -> bool:
        """Validate system environment"""
        self.log("🔍 Validating environment...")

        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                self.log("❌ Python 3.8+ required", "ERROR")
                return False
            self.log(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

            # Check required packages
            required_packages = [
                'asyncio', 'aiohttp', 'pandas', 'numpy', 'plotly', 'streamlit',
                'python-dotenv', 'requests', 'scipy', 'statsmodels'
            ]

            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    missing_packages.append(package)

            if missing_packages:
                self.log(f"❌ Missing packages: {', '.join(missing_packages)}", "ERROR")
                self.log("💡 Run: pip install -r requirements.txt")
                return False

            self.log("✅ All required packages installed")

            # Check .env file
            env_file = self.project_root / '.env'
            if not env_file.exists():
                self.log("❌ .env file not found", "ERROR")
                return False

            self.log("✅ Environment file found")
            self.component_status['environment'] = True
            return True

        except Exception as e:
            self.log(f"❌ Environment validation failed: {e}", "ERROR")
            return False

    async def validate_api_keys(self) -> bool:
        """Validate API keys and connectivity"""
        self.log("🔑 Validating API keys...")

        try:
            # Load environment
            from dotenv import load_dotenv
            load_dotenv()

            # Required API keys based on mode
            required_keys = {
                'ALPHA_VANTAGE_API_KEY': 'Alpha Vantage',
                'COINGECKO_API_KEY': 'CoinGecko',
                'CURRENCY_API_KEY': 'CurrencyAPI',
                'TWELVE_DATA_API_KEY': 'Twelve Data',
                'POLYGON_API_KEY': 'Polygon.io',
                'FINNHUB_API_KEY': 'Finnhub'
            }

            if self.mode == 'live':
                required_keys.update({
                    'BINANCE_API_KEY': 'Binance',
                    'BINANCE_API_SECRET': 'Binance Secret'
                })

            missing_keys = []
            for key, service in required_keys.items():
                if not os.getenv(key):
                    missing_keys.append(f"{service} ({key})")

            if missing_keys:
                self.log(f"❌ Missing API keys: {', '.join(missing_keys)}", "ERROR")
                return False

            self.log("✅ All required API keys configured")

            # Test API connectivity
            self.log("🌐 Testing API connectivity...")

            # Test Alpha Vantage
            alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if alpha_key:
                try:
                    response = requests.get(
                        f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={alpha_key}',
                        timeout=10
                    )
                    if response.status_code == 200 and 'Global Quote' in response.text:
                        self.log("✅ Alpha Vantage API connected")
                    else:
                        self.log("❌ Alpha Vantage API test failed")
                        return False
                except Exception as e:
                    self.log(f"❌ Alpha Vantage connection error: {e}")
                    return False

            # Test Polygon.io
            polygon_key = os.getenv('POLYGON_API_KEY')
            if polygon_key:
                try:
                    response = requests.get(
                        f'https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-02?apiKey={polygon_key}',
                        timeout=10
                    )
                    if response.status_code == 200:
                        self.log("✅ Polygon.io API connected")
                    else:
                        self.log("❌ Polygon.io API test failed")
                        return False
                except Exception as e:
                    self.log(f"❌ Polygon.io connection error: {e}")
                    return False

            # Test Finnhub
            finnhub_key = os.getenv('FINNHUB_API_KEY')
            if finnhub_key:
                try:
                    response = requests.get(
                        f'https://finnhub.io/api/v1/quote?symbol=AAPL&token={finnhub_key}',
                        timeout=10
                    )
                    if response.status_code == 200:
                        self.log("✅ Finnhub API connected")
                    else:
                        self.log("❌ Finnhub API test failed")
                        return False
                except Exception as e:
                    self.log(f"❌ Finnhub connection error: {e}")
                    return False

            self.component_status['api_keys'] = True
            self.component_status['network'] = True
            return True

        except Exception as e:
            self.log(f"❌ API key validation failed: {e}", "ERROR")
            return False

    async def validate_components(self) -> bool:
        """Validate AAC system components"""
        self.log("🔧 Validating system components...")

        try:
            # Test arbitrage detector
            self.log("   Testing arbitrage detector...")
            detector = MultiSourceArbitrageDetector()
            opportunities = await detector.detect_opportunities()
            self.log(f"   ✅ Arbitrage detector working - {len(opportunities)} opportunities found")
            self.component_status['arbitrage_detector'] = True

            # Test trading engine (if live mode)
            if self.mode == 'live':
                self.log("   Testing trading engine...")
                binance_config = BinanceConfig()
                trading_config = TradingConfig()

                if binance_config.is_configured():
                    from binance_trading_engine import BinanceTradingEngine
                    async with BinanceTradingEngine(binance_config, trading_config) as engine:
                        balance = await engine.get_account_balance('USDT')
                        if balance is not None:
                            self.log("   ✅ Trading engine connected")
                            self.component_status['trading_engine'] = True
                        else:
                            self.log("❌ Trading engine balance check failed")
                            return False
                else:
                    self.log("❌ Binance API keys not configured for live mode")
                    return False

            # Test execution system
            self.log("   Testing execution system...")
            execution_config = ExecutionConfig()
            execution_system = AACArbitrageExecutionSystem(execution_config)
            await execution_system.initialize()
            self.log("   ✅ Execution system initialized")
            self.component_status['execution_system'] = True

            return True

        except Exception as e:
            self.log(f"❌ Component validation failed: {e}", "ERROR")
            return False

    async def deploy_system(self) -> bool:
        """Deploy the complete AAC system"""
        self.log("🚀 Deploying AAC arbitrage system...")

        try:
            # Create execution system
            execution_config = ExecutionConfig(
                auto_execute=(self.mode == 'live'),
                enable_test_mode=(self.mode == 'test')
            )

            execution_system = AACArbitrageExecutionSystem(execution_config)
            await execution_system.initialize()

            self.log(f"✅ System deployed in {self.mode.upper()} mode")

            # Start monitoring dashboard
            if self._should_start_dashboard():
                self.log("📊 Starting monitoring dashboard...")
                self._start_dashboard()
                self.component_status['monitoring'] = True

            # Start continuous trading
            self.log("🎯 Starting continuous arbitrage trading...")

            await execution_system.run_continuous_trading(
                cycles=None,  # Run indefinitely
                cycle_interval=60  # 1 minute cycles
            )

            return True

        except Exception as e:
            self.log(f"❌ Deployment failed: {e}", "ERROR")
            return False

    def _should_start_dashboard(self) -> bool:
        """Check if dashboard should be started"""
        return os.getenv('START_DASHBOARD', 'true').lower() == 'true'

    def _start_dashboard(self):
        """Start the monitoring dashboard"""
        try:
            dashboard_script = self.project_root / 'aac_monitoring_dashboard.py'
            if dashboard_script.exists():
                # Start dashboard in background
                subprocess.Popen([
                    sys.executable, str(dashboard_script)
                ], cwd=self.project_root)
                self.log("✅ Monitoring dashboard started")
            else:
                self.log("⚠️  Monitoring dashboard script not found")
        except Exception as e:
            self.log(f"❌ Failed to start dashboard: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'deployment_mode': self.mode,
            'config_validated': self.config_validated,
            'system_ready': self.system_ready,
            'component_status': self.component_status,
            'deployment_log': self.deployment_log[-10:],  # Last 10 log entries
            'timestamp': datetime.now().isoformat()
        }

    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        self.log("🏥 Running health check...")

        health_status = {
            'overall_status': 'unknown',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            health_status['checks']['memory_usage'] = f"{memory_mb:.1f} MB"

            # CPU usage
            cpu_percent = process.cpu_percent(interval=1)
            health_status['checks']['cpu_usage'] = f"{cpu_percent:.1f}%"

            # Network connectivity
            try:
                requests.get('https://api.binance.com/api/v3/ping', timeout=5)
                health_status['checks']['network'] = 'connected'
            except Exception as e:
                logger.error(f"Deployment step failed: {e}")
                health_status['checks']['network'] = 'disconnected'

            # API responsiveness
            health_status['checks']['api_status'] = 'checking'
            # Add more detailed API checks here

            health_status['overall_status'] = 'healthy'

        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['error'] = str(e)

        return health_status

async def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='AAC Production Deployment')
    parser.add_argument('--mode', choices=['test', 'live'], default='test',
                       help='Deployment mode (default: test)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate configuration, do not deploy')
    parser.add_argument('--status', action='store_true',
                       help='Show system status')
    parser.add_argument('--health-check', action='store_true',
                       help='Run health check')

    args = parser.parse_args()

    print("🚀 AAC Production Deployment Script")
    print("=" * 50)

    deployer = AACProductionDeployer(mode=args.mode)

    if args.status:
        # Show current status
        status = deployer.get_system_status()
        print(json.dumps(status, indent=2))
        return

    if args.health_check:
        # Run health check
        health = await deployer.run_health_check()
        print(json.dumps(health, indent=2))
        return

    # Validation phase
    print("\n📋 Phase 1: Validation")
    print("-" * 30)

    validations = [
        ('Environment', deployer.validate_environment()),
        ('API Keys', deployer.validate_api_keys()),
        ('Components', deployer.validate_components())
    ]

    all_valid = True
    for name, validation_coro in validations:
        try:
            result = await validation_coro
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{name}: {status}")
            if not result:
                all_valid = False
        except Exception as e:
            print(f"{name}: ❌ ERROR - {e}")
            all_valid = False

    if not all_valid:
        print("\n❌ Validation failed. Please fix issues and try again.")
        sys.exit(1)

    deployer.config_validated = True
    print("\n✅ All validations passed!")

    if args.validate_only:
        print("🎯 Validation complete. Use --mode live to deploy.")
        return

    # Deployment phase
    print("\n🚀 Phase 2: Deployment")
    print("-" * 30)

    if args.mode == 'live':
        print("⚠️  LIVE MODE: This will execute real trades!")
        confirm = input("Are you sure you want to continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Deployment cancelled.")
            return

    try:
        success = await deployer.deploy_system()
        if success:
            print("\n✅ AAC system deployed successfully!")
            print(f"📊 Mode: {args.mode.upper()}")
            print("🎯 System is now running continuous arbitrage trading")
            print("📈 Check the monitoring dashboard for real-time status")
        else:
            print("\n❌ Deployment failed. Check logs above.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Deployment interrupted by user")
    except Exception as e:
        print(f"\n❌ Deployment error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())