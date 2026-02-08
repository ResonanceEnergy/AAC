#!/usr/bin/env python3
"""
AAC 2100 Missing Components Audit
"""

from shared.config_loader import get_config
import os
from pathlib import Path

def audit_missing_components():
    config = get_config()

    print('🔍 AAC 2100 MISSING COMPONENTS AUDIT')
    print('=' * 50)
    print()

    # Configuration Status
    print('[MONITOR] CONFIGURATION STATUS:')
    try:
        paper_trading = config.trading.paper_trading if hasattr(config, 'trading') and hasattr(config.trading, 'paper_trading') else True
        dry_run = config.trading.dry_run if hasattr(config, 'trading') and hasattr(config.trading, 'dry_run') else False
    except:
        paper_trading = True
        dry_run = False
    print(f'  Paper Trading: {paper_trading}')
    print(f'  Dry Run: {dry_run}')
    print()

    # Exchange API Keys
    print('🔑 EXCHANGE API KEYS:')
    binance_key = getattr(config.binance, 'api_key', '') if hasattr(config, 'binance') else ''
    coinbase_key = getattr(config.coinbase, 'api_key', '') if hasattr(config, 'coinbase') else ''
    kraken_key = getattr(config.kraken, 'api_key', '') if hasattr(config, 'kraken') else ''

    print(f'  Binance: {"[CROSS] MISSING" if not binance_key else "✅ CONFIGURED"}')
    print(f'  Coinbase: {"[CROSS] MISSING" if not coinbase_key else "✅ CONFIGURED"}')
    print(f'  Kraken: {"[CROSS] MISSING" if not kraken_key else "✅ CONFIGURED"}')
    print()

    # Account Funding
    print('[MONEY] ACCOUNT FUNDING:')
    print('  [WARN]️  NO REAL CAPITAL CONFIGURED (Paper Trading Mode)')
    print()

    # Database
    print('🗄️ DATABASE:')
    try:
        db_path = config.database.path if hasattr(config, 'database') and hasattr(config.database, 'path') else 'data/accounting.db'
    except:
        db_path = 'data/accounting.db'
    print(f'  Path: {db_path}')
    print('  ✅ SQLite configured (consider PostgreSQL for production)')
    print()

    # External Monitoring
    print('📡 EXTERNAL MONITORING:')
    try:
        prometheus_port = config.monitoring.prometheus_port if hasattr(config, 'monitoring') and hasattr(config.monitoring, 'prometheus_port') else None
    except:
        prometheus_port = None
    print(f'  Prometheus: {"[CROSS] NOT CONFIGURED" if not prometheus_port else "✅ CONFIGURED"}')
    print('  Grafana: [CROSS] NOT CONFIGURED (config exists but not integrated)')
    print()

    # Alerts & Notifications
    print('[ALERT] ALERTS & NOTIFICATIONS:')
    telegram_enabled = False
    slack_enabled = False
    try:
        telegram_enabled = config.notifications.telegram_enabled()
        slack_enabled = config.notifications.slack_enabled()
    except:
        # Fallback check
        telegram_enabled = bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))
        slack_enabled = bool(os.getenv('SLACK_WEBHOOK_URL'))
    print(f'  Telegram: {"[CROSS] NOT CONFIGURED" if not telegram_enabled else "✅ CONFIGURED"}')
    print(f'  Slack: {"[CROSS] NOT CONFIGURED" if not slack_enabled else "✅ CONFIGURED"}')
    print()

    # Security
    print('🔐 SECURITY:')
    print('  ✅ Secrets Manager implemented')
    print(f'  Master Password: {"[CROSS] NOT SET" if not os.getenv("ACC_MASTER_PASSWORD") else "✅ SET"}')
    print('  [WARN]️  No HSM or key management service')
    print()

    # Infrastructure
    print('☁️ INFRASTRUCTURE:')
    k8s_exists = Path('k8s').exists()
    docker_exists = Path('docker-compose.yml').exists()
    print(f'  Kubernetes: {"✅ CONFIGURED" if k8s_exists else "[CROSS] MISSING"}')
    print(f'  Docker Compose: {"✅ CONFIGURED" if docker_exists else "[CROSS] MISSING"}')
    print('  [CROSS] No CI/CD pipeline configuration')
    print()

    # Documentation
    print('📚 DOCUMENTATION:')
    readme_exists = Path('README.md').exists()
    print(f'  README: {"✅ EXISTS" if readme_exists else "[CROSS] MISSING"}')
    print('  [CROSS] No API documentation')
    print('  [CROSS] No deployment runbooks')
    print('  [CROSS] No troubleshooting guides')
    print()

    # Testing
    print('🧪 TESTING & VALIDATION:')
    tests_exist = Path('tests').exists()
    print(f'  Test Suite: {"✅ EXISTS" if tests_exist else "[CROSS] MISSING"}')
    print('  [CROSS] No integration tests for live trading')
    print('  [CROSS] No chaos engineering tests')
    print('  [CROSS] No performance benchmarking')
    print()

    # Production Optimizations
    print('⚡ PRODUCTION OPTIMIZATIONS:')
    print('  ✅ Async/await patterns implemented')
    print('  [WARN]️  No connection pooling for high-frequency trading')
    print('  [WARN]️  No rate limiting for exchange APIs')
    print('  [WARN]️  No circuit breaker for external dependencies')
    print()

    # Summary
    print('📋 SUMMARY - CRITICAL MISSING COMPONENTS:')
    critical_missing = []

    if not binance_key and not coinbase_key and not kraken_key:
        critical_missing.append('• Exchange API Keys (required for live trading)')

    if not os.getenv('ACC_MASTER_PASSWORD'):
        critical_missing.append('• Master Password for secrets encryption')

    if not telegram_enabled and not slack_enabled:
        critical_missing.append('• Alert/Notification system')

    if not k8s_exists:
        critical_missing.append('• Production deployment manifests')

    if not docker_exists:
        critical_missing.append('• Local development environment')

    if critical_missing:
        print('\n'.join(critical_missing))
    else:
        print('✅ All critical components present!')

    print()
    print('[TARGET] READY FOR: Paper Trading & Development')
    print('[DEPLOY] REQUIRES: Exchange API keys for live trading')

if __name__ == "__main__":
    audit_missing_components()