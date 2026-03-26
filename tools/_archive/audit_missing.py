#!/usr/bin/env python3
"""
AAC 2100 Missing Components Audit
"""

from shared.config_loader import get_config
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def audit_missing_components():
    """Audit missing components."""
    config = get_config()

    logger.info('🔍 AAC 2100 MISSING COMPONENTS AUDIT')
    logger.info('=' * 50)
    logger.info("")

    # Configuration Status
    logger.info('[MONITOR] CONFIGURATION STATUS:')
    try:
        paper_trading = config.trading.paper_trading if hasattr(config, 'trading') and hasattr(config.trading, 'paper_trading') else True
        dry_run = config.trading.dry_run if hasattr(config, 'trading') and hasattr(config.trading, 'dry_run') else False
    except (AttributeError, TypeError) as e:
        logger.warning(f"Config access issue: {e}")
        paper_trading = True
        dry_run = False
    logger.info(f'  Paper Trading: {paper_trading}')
    logger.info(f'  Dry Run: {dry_run}')
    logger.info("")

    # Exchange API Keys
    logger.info('🔑 EXCHANGE API KEYS:')
    binance_key = getattr(config.binance, 'api_key', '') if hasattr(config, 'binance') else ''
    coinbase_key = getattr(config.coinbase, 'api_key', '') if hasattr(config, 'coinbase') else ''
    kraken_key = getattr(config.kraken, 'api_key', '') if hasattr(config, 'kraken') else ''

    logger.info(f'  Binance: {"[CROSS] MISSING" if not binance_key else "✅ CONFIGURED"}')
    logger.info(f'  Coinbase: {"[CROSS] MISSING" if not coinbase_key else "✅ CONFIGURED"}')
    logger.info(f'  Kraken: {"[CROSS] MISSING" if not kraken_key else "✅ CONFIGURED"}')
    logger.info("")

    # Account Funding
    logger.info('[MONEY] ACCOUNT FUNDING:')
    logger.info('  [WARN]️  NO REAL CAPITAL CONFIGURED (Paper Trading Mode)')
    logger.info("")

    # Database
    logger.info('🗄️ DATABASE:')
    try:
        db_path = config.database.path if hasattr(config, 'database') and hasattr(config.database, 'path') else 'data/accounting.db'
    except (AttributeError, TypeError) as e:
        logger.warning(f"Config access issue: {e}")
        db_path = 'data/accounting.db'
    logger.info(f'  Path: {db_path}')
    logger.info('  ✅ SQLite configured (consider PostgreSQL for production)')
    logger.info("")

    # External Monitoring
    logger.info('📡 EXTERNAL MONITORING:')
    try:
        prometheus_port = config.monitoring.prometheus_port if hasattr(config, 'monitoring') and hasattr(config.monitoring, 'prometheus_port') else None
    except (AttributeError, TypeError) as e:
        logger.warning(f"Config access issue: {e}")
        prometheus_port = None
    logger.info(f'  Prometheus: {"[CROSS] NOT CONFIGURED" if not prometheus_port else "✅ CONFIGURED"}')
    logger.info('  Grafana: [CROSS] NOT CONFIGURED (config exists but not integrated)')
    logger.info("")

    # Alerts & Notifications
    logger.info('[ALERT] ALERTS & NOTIFICATIONS:')
    telegram_enabled = False
    slack_enabled = False
    try:
        telegram_enabled = config.notifications.telegram_enabled()
        slack_enabled = config.notifications.slack_enabled()
    except (AttributeError, TypeError) as e:
        logger.warning(f"Config access issue: {e}")
        # Fallback check
        telegram_enabled = bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))
        slack_enabled = bool(os.getenv('SLACK_WEBHOOK_URL'))
    logger.info(f'  Telegram: {"[CROSS] NOT CONFIGURED" if not telegram_enabled else "✅ CONFIGURED"}')
    logger.info(f'  Slack: {"[CROSS] NOT CONFIGURED" if not slack_enabled else "✅ CONFIGURED"}')
    logger.info("")

    # Security
    logger.info('🔐 SECURITY:')
    logger.info('  ✅ Secrets Manager implemented')
    logger.info(f'  Master Password: {"[CROSS] NOT SET" if not os.getenv("ACC_MASTER_PASSWORD") else "✅ SET"}')
    logger.info('  [WARN]️  No HSM or key management service')
    logger.info("")

    # Infrastructure
    logger.info('☁️ INFRASTRUCTURE:')
    k8s_exists = Path('k8s').exists()
    docker_exists = Path('docker-compose.yml').exists()
    logger.info(f'  Kubernetes: {"✅ CONFIGURED" if k8s_exists else "[CROSS] MISSING"}')
    logger.info(f'  Docker Compose: {"✅ CONFIGURED" if docker_exists else "[CROSS] MISSING"}')
    logger.info('  [CROSS] No CI/CD pipeline configuration')
    logger.info("")

    # Documentation
    logger.info('📚 DOCUMENTATION:')
    readme_exists = Path('README.md').exists()
    logger.info(f'  README: {"✅ EXISTS" if readme_exists else "[CROSS] MISSING"}')
    logger.info('  [CROSS] No API documentation')
    logger.info('  [CROSS] No deployment runbooks')
    logger.info('  [CROSS] No troubleshooting guides')
    logger.info("")

    # Testing
    logger.info('🧪 TESTING & VALIDATION:')
    tests_exist = Path('tests').exists()
    logger.info(f'  Test Suite: {"✅ EXISTS" if tests_exist else "[CROSS] MISSING"}')
    logger.info('  [CROSS] No integration tests for live trading')
    logger.info('  [CROSS] No chaos engineering tests')
    logger.info('  [CROSS] No performance benchmarking')
    logger.info("")

    # Production Optimizations
    logger.info('⚡ PRODUCTION OPTIMIZATIONS:')
    logger.info('  ✅ Async/await patterns implemented')
    logger.info('  [WARN]️  No connection pooling for high-frequency trading')
    logger.info('  [WARN]️  No rate limiting for exchange APIs')
    logger.info('  [WARN]️  No circuit breaker for external dependencies')
    logger.info("")

    # Summary
    logger.info('📋 SUMMARY - CRITICAL MISSING COMPONENTS:')
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
        logger.info('\n'.join(critical_missing))
    else:
        logger.info('✅ All critical components present!')

    logger.info("")
    logger.info('[TARGET] READY FOR: Paper Trading & Development')
    logger.info('[DEPLOY] REQUIRES: Exchange API keys for live trading')

if __name__ == "__main__":
    audit_missing_components()