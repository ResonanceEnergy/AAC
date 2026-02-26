#!/usr/bin/env python3
"""
AAC 2100 Production Setup Script
================================
Configures all production components for live trading deployment.

This script will:
1. Set up exchange API keys
2. Configure master password for secrets encryption
3. Set up alert/notification systems (Telegram/Slack)
4. Configure monitoring (Prometheus/Grafana)
5. Implement production safeguards (rate limiting, circuit breakers)
"""

import os
import sys
import json
import secrets
import string
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, Config
from shared.secrets_manager import SecretsManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductionSetup:
    """Production setup orchestrator"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.env_file = self.project_root / '.env'
        self.secrets_manager = SecretsManager()

    def generate_master_password(self) -> str:
        """Generate a secure master password"""
        alphabet = string.ascii_letters + string.digits + string.punctuation
        password = ''.join(secrets.choice(alphabet) for i in range(32))
        return password

    def setup_master_password(self) -> bool:
        """Set up master password for secrets encryption"""
        print("\n🔐 MASTER PASSWORD SETUP")
        print("=" * 30)

        if os.getenv('ACC_MASTER_PASSWORD'):
            print("✅ Master password already set in environment")
            return True

        # Generate secure password
        master_password = self.generate_master_password()

        # Set environment variable
        os.environ['ACC_MASTER_PASSWORD'] = master_password

        # Also save to .env file for persistence
        self.update_env_var('ACC_MASTER_PASSWORD', master_password)

        print(f"✅ Generated and set master password: {master_password[:8]}...")
        print("[WARN]️  IMPORTANT: Save this password securely - it's required for secrets decryption!")

        return True

    def setup_exchange_apis(self) -> bool:
        """Configure exchange API keys"""
        print("\n🔑 EXCHANGE API CONFIGURATION")
        print("=" * 35)

        exchanges = {
            'binance': {
                'name': 'Binance',
                'vars': ['BINANCE_API_KEY', 'BINANCE_API_SECRET'],
                'testnet_var': 'BINANCE_TESTNET'
            },
            'coinbase': {
                'name': 'Coinbase Pro',
                'vars': ['COINBASE_API_KEY', 'COINBASE_API_SECRET', 'COINBASE_PASSPHRASE']
            },
            'kraken': {
                'name': 'Kraken',
                'vars': ['KRAKEN_API_KEY', 'KRAKEN_API_SECRET']
            }
        }

        configured = 0

        for exchange_id, exchange_info in exchanges.items():
            print(f"\n📈 {exchange_info['name']} Setup:")

            # Check if already configured
            if all(os.getenv(var) for var in exchange_info['vars']):
                print(f"  ✅ Already configured")
                configured += 1
                continue

            # Prompt for configuration
            configure = input(f"  Configure {exchange_info['name']} API keys? (y/n): ").lower().strip()

            if configure == 'y':
                print("  Enter your API credentials:")

                for var in exchange_info['vars']:
                    current = os.getenv(var, '')
                    if current:
                        use_existing = input(f"  {var} already set. Use existing? (y/n): ").lower().strip()
                        if use_existing != 'y':
                            current = input(f"  {var}: ").strip()
                    else:
                        current = input(f"  {var}: ").strip()

                    if current:
                        os.environ[var] = current
                        self.update_env_var(var, current)

                # Special handling for Binance testnet
                if exchange_id == 'binance':
                    testnet = input("  Use testnet? (y/n) [y]: ").lower().strip()
                    testnet = testnet != 'n'  # Default to True
                    os.environ['BINANCE_TESTNET'] = str(testnet).lower()
                    self.update_env_var('BINANCE_TESTNET', str(testnet).lower())

                print(f"  ✅ {exchange_info['name']} configured")
                configured += 1
            else:
                print(f"  ⏭️  Skipped {exchange_info['name']}")

        print(f"\n[MONITOR] Exchange Configuration: {configured}/{len(exchanges)} exchanges configured")
        return configured > 0

    def setup_notifications(self) -> bool:
        """Configure notification systems"""
        print("\n[ALERT] NOTIFICATION SYSTEM SETUP")
        print("=" * 32)

        notification_options = {
            'telegram': {
                'name': 'Telegram',
                'vars': ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID'],
                'instructions': 'Create a bot at @BotFather and get your chat ID'
            },
            'slack': {
                'name': 'Slack',
                'vars': ['SLACK_WEBHOOK_URL'],
                'instructions': 'Create a webhook in Slack settings'
            },
            'email': {
                'name': 'Email (SMTP)',
                'vars': ['SMTP_HOST', 'SMTP_PORT', 'SMTP_USER', 'SMTP_PASSWORD', 'ALERT_EMAIL_TO'],
                'instructions': 'Configure SMTP settings for email alerts'
            }
        }

        configured = 0

        for notif_id, notif_info in notification_options.items():
            print(f"\n📢 {notif_info['name']} Setup:")

            # Check if already configured
            if all(os.getenv(var) for var in notif_info['vars']):
                print(f"  ✅ Already configured")
                configured += 1
                continue

            configure = input(f"  Configure {notif_info['name']} notifications? (y/n): ").lower().strip()

            if configure == 'y':
                print(f"  Instructions: {notif_info['instructions']}")
                print("  Enter your configuration:")

                for var in notif_info['vars']:
                    current = os.getenv(var, '')
                    if current:
                        use_existing = input(f"  {var} already set. Use existing? (y/n): ").lower().strip()
                        if use_existing != 'y':
                            current = input(f"  {var}: ").strip()
                    else:
                        current = input(f"  {var}: ").strip()

                    if current:
                        os.environ[var] = current
                        self.update_env_var(var, current)

                print(f"  ✅ {notif_info['name']} configured")
                configured += 1
            else:
                print(f"  ⏭️  Skipped {notif_info['name']}")

        print(f"\n[MONITOR] Notifications: {configured}/{len(notification_options)} systems configured")
        return configured > 0

    def setup_monitoring(self) -> bool:
        """Configure monitoring systems"""
        print("\n[MONITOR] MONITORING SETUP")
        print("=" * 20)

        # Check if Prometheus config exists
        prometheus_config = self.project_root / 'config' / 'prometheus.yml'
        if prometheus_config.exists():
            print("✅ Prometheus config exists")
        else:
            print("[WARN]️  Prometheus config missing - creating basic config")
            self.create_prometheus_config()

        # Check Grafana config
        grafana_config = self.project_root / 'config' / 'grafana'
        if grafana_config.exists():
            print("✅ Grafana config directory exists")
        else:
            print("[WARN]️  Grafana config missing - creating directory")
            grafana_config.mkdir(parents=True, exist_ok=True)

        # Set monitoring environment variables
        monitoring_vars = {
            'PROMETHEUS_PORT': '9090',
            'GRAFANA_PORT': '3000',
            'METRICS_RETENTION_DAYS': '30'
        }

        for var, default in monitoring_vars.items():
            if not os.getenv(var):
                os.environ[var] = default
                self.update_env_var(var, default)

        print("✅ Monitoring configuration updated")
        return True

    def setup_production_safeguards(self) -> bool:
        """Implement production safeguards"""
        print("\n[SHIELD]️ PRODUCTION SAFEGUARDS SETUP")
        print("=" * 32)

        safeguards = {
            'rate_limiting': {
                'name': 'Rate Limiting',
                'description': 'Prevents excessive API calls to exchanges',
                'vars': {
                    'RATE_LIMIT_REQUESTS_PER_MINUTE': '60',
                    'RATE_LIMIT_BURST_SIZE': '10'
                }
            },
            'circuit_breaker': {
                'name': 'Circuit Breaker',
                'description': 'Automatically stops trading if exchange APIs fail',
                'vars': {
                    'CIRCUIT_BREAKER_FAILURE_THRESHOLD': '5',
                    'CIRCUIT_BREAKER_RECOVERY_TIMEOUT': '60'
                }
            },
            'risk_limits': {
                'name': 'Risk Limits',
                'description': 'Position size and loss limits',
                'vars': {
                    'MAX_POSITION_SIZE_USD': '10000',
                    'MAX_DAILY_LOSS_USD': '1000',
                    'MAX_OPEN_POSITIONS': '10'
                }
            }
        }

        for safeguard_id, safeguard_info in safeguards.items():
            print(f"\n🔧 {safeguard_info['name']}:")
            print(f"  {safeguard_info['description']}")

            for var, default in safeguard_info['vars'].items():
                if not os.getenv(var):
                    os.environ[var] = default
                    self.update_env_var(var, default)
                    print(f"  ✅ Set {var} = {default}")
                else:
                    print(f"  ✅ {var} already set")

        # Switch from paper trading to live (with confirmation)
        print("\n[TARGET] TRADING MODE:")
        current_paper = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
        current_dry = os.getenv('DRY_RUN', 'true').lower() == 'true'

        if current_paper or current_dry:
            print("  Current mode: PAPER TRADING / DRY RUN")
            enable_live = input("  Enable LIVE trading? (y/n) [n]: ").lower().strip()

            if enable_live == 'y':
                os.environ['PAPER_TRADING'] = 'false'
                os.environ['DRY_RUN'] = 'false'
                self.update_env_var('PAPER_TRADING', 'false')
                self.update_env_var('DRY_RUN', 'false')
                print("  [DEPLOY] LIVE TRADING ENABLED!")
            else:
                print("  📝 Staying in paper trading mode")
        else:
            print("  [DEPLOY] Already configured for live trading")

        print("✅ Production safeguards configured")
        return True

    def update_env_var(self, key: str, value: str):
        """Update environment variable in .env file"""
        if not self.env_file.exists():
            # Create .env file if it doesn't exist
            self.env_file.write_text("")

        # Read current content
        content = self.env_file.read_text()

        # Check if variable already exists
        lines = content.split('\n')
        updated = False

        for i, line in enumerate(lines):
            if line.startswith(f'{key}='):
                lines[i] = f'{key}={value}'
                updated = True
                break

        if not updated:
            # Add new variable
            if content and not content.endswith('\n'):
                content += '\n'
            content += f'{key}={value}\n'
            lines = content.split('\n')

        # Write back
        self.env_file.write_text('\n'.join(lines))

    def create_prometheus_config(self):
        """Create basic Prometheus configuration"""
        prometheus_config = self.project_root / 'config' / 'prometheus.yml'

        config_content = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'aac-orchestrator'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

  - job_name: 'aac-bigbrain'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/metrics'

  - job_name: 'aac-crypto-intel'
    static_configs:
      - targets: ['localhost:8002']
    metrics_path: '/metrics'

  - job_name: 'aac-accounting'
    static_configs:
      - targets: ['localhost:8003']
    metrics_path: '/metrics'
"""

        prometheus_config.parent.mkdir(parents=True, exist_ok=True)
        prometheus_config.write_text(config_content)
        print(f"  Created {prometheus_config}")

    def validate_setup(self) -> Dict[str, Any]:
        """Validate the production setup"""
        print("\n🔍 VALIDATION")
        print("=" * 12)

        config = get_config()
        validation = config.validate()

        print("Configuration Status:")
        print(f"  Environment: {validation['environment']}")
        print(f"  Dry Run: {validation['dry_run']}")
        print(f"  Exchanges: {', '.join(validation['exchanges_configured']) if validation['exchanges_configured'] else 'None'}")

        if validation['issues']:
            print("  [CROSS] Issues:")
            for issue in validation['issues']:
                print(f"    - {issue}")

        if validation['warnings']:
            print("  [WARN]️  Warnings:")
            for warning in validation['warnings']:
                print(f"    - {warning}")

        if validation['valid'] and validation['exchanges_configured']:
            print("  ✅ Setup appears ready for live trading!")
        else:
            print("  [WARN]️  Setup incomplete - review issues above")

        return validation

    def run_setup(self):
        """Run the complete production setup"""
        print("[DEPLOY] AAC 2100 PRODUCTION SETUP")
        print("=" * 30)
        print("This will configure your system for live trading.")
        print("Make sure you have your API keys and credentials ready.\n")

        # Run each setup component
        components = [
            ('Master Password', self.setup_master_password),
            ('Exchange APIs', self.setup_exchange_apis),
            ('Notifications', self.setup_notifications),
            ('Monitoring', self.setup_monitoring),
            ('Production Safeguards', self.setup_production_safeguards),
        ]

        results = {}
        for name, setup_func in components:
            try:
                result = setup_func()
                results[name] = result
                print(f"✅ {name}: {'SUCCESS' if result else 'SKIPPED'}")
            except Exception as e:
                print(f"[CROSS] {name}: FAILED - {e}")
                results[name] = False

        # Validate final setup
        validation = self.validate_setup()

        # Summary
        print("\n📋 SETUP SUMMARY")
        print("=" * 15)

        successful = sum(1 for r in results.values() if r)
        total = len(results)

        print(f"Components configured: {successful}/{total}")

        if validation['valid'] and validation['exchanges_configured']:
            print("\n[CELEBRATION] PRODUCTION SETUP COMPLETE!")
            print("Your AAC 2100 system is ready for live trading.")
            print("\nNext steps:")
            print("1. Start the system: python run_integrated_system.py")
            print("2. Monitor logs and metrics")
            print("3. Gradually increase position sizes")
        else:
            print("\n[WARN]️  SETUP INCOMPLETE")
            print("Address the issues above before going live.")

        return results


def main():
    """Main entry point"""
    setup = ProductionSetup()
    setup.run_setup()


if __name__ == "__main__":
    main()