#!/usr/bin/env python3
"""
Startup Validator
=================
Validates configuration and credentials before system startup.
"""

import os
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path

from .config_loader import get_config, get_project_path


class ValidationSeverity(Enum):
    """Validation result severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    check_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        status = "✓" if self.passed else "✗"
        return f"[{status}] {self.check_name}: {self.message}"


@dataclass
class StartupValidation:
    """Complete startup validation results"""
    results: List[ValidationResult] = field(default_factory=list)
    start_time: float = 0
    end_time: float = 0
    
    @property
    def all_passed(self) -> bool:
        """Check if all validations passed"""
        return all(r.passed for r in self.results)
    
    @property
    def critical_passed(self) -> bool:
        """Check if all critical validations passed"""
        return all(
            r.passed for r in self.results 
            if r.severity == ValidationSeverity.CRITICAL
        )
    
    @property
    def errors(self) -> List[ValidationResult]:
        """Get all failed checks"""
        return [r for r in self.results if not r.passed]
    
    @property
    def warnings(self) -> List[ValidationResult]:
        """Get all warnings"""
        return [
            r for r in self.results 
            if r.severity == ValidationSeverity.WARNING and not r.passed
        ]
    
    def add(self, result: ValidationResult):
        """Add a validation result"""
        self.results.append(result)
    
    def summary(self) -> str:
        """Get validation summary"""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        errors = len(self.errors)
        warnings = len(self.warnings)
        
        lines = [
            f"\n{'='*60}",
            f"  Startup Validation Summary",
            f"{'='*60}",
            f"  Passed: {passed}/{total}",
        ]
        
        if errors:
            lines.append(f"  Errors: {errors}")
        if warnings:
            lines.append(f"  Warnings: {warnings}")
        
        lines.append(f"{'='*60}")
        
        # List failures
        if self.errors:
            lines.append("\n  ❌ Failed Checks:")
            for r in self.errors:
                lines.append(f"    • {r.check_name}: {r.message}")
        
        # List warnings
        if self.warnings:
            lines.append("\n  ⚠️  Warnings:")
            for r in self.warnings:
                lines.append(f"    • {r.check_name}: {r.message}")
        
        lines.append("")
        return "\n".join(lines)


class StartupValidator:
    """
    Validates system configuration and dependencies before startup.
    
    Performs checks for:
    - Required configuration files
    - Exchange API credentials
    - Database connectivity
    - Directory permissions
    - Optional service availability
    """
    
    def __init__(self):
        self.logger = logging.getLogger("StartupValidator")
        self.config = None
        
    async def validate_all(self) -> StartupValidation:
        """Run all validation checks"""
        import time
        
        validation = StartupValidation()
        validation.start_time = time.time()
        
        # Configuration checks
        validation.add(self._check_config_file())
        
        if validation.results[-1].passed:
            self.config = get_config()
            
            # Credential checks
            validation.add(self._check_binance_credentials())
            validation.add(self._check_coinbase_credentials())
            validation.add(self._check_kraken_credentials())
            
            # Database checks
            validation.add(self._check_database_path())
            
            # Directory checks
            validation.add(self._check_data_directory())
            validation.add(self._check_logs_directory())
            
            # Optional service checks
            validation.add(await self._check_redis_connection())
            validation.add(self._check_notification_config())
            validation.add(self._check_infura_rpc())
            
            # Trading mode checks
            validation.add(self._check_trading_mode())
        
        validation.end_time = time.time()
        return validation
    
    def _check_config_file(self) -> ValidationResult:
        """Check if main config file exists"""
        config_path = get_project_path("config", "config.yaml")
        
        if config_path.exists():
            return ValidationResult(
                check_name="Configuration File",
                passed=True,
                severity=ValidationSeverity.CRITICAL,
                message=f"Found at {config_path}",
            )
        else:
            return ValidationResult(
                check_name="Configuration File",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Missing config/config.yaml",
            )
    
    def _check_binance_credentials(self) -> ValidationResult:
        """Check Binance API credentials"""
        api_key = os.getenv('BINANCE_API_KEY') or getattr(self.config.binance, 'api_key', '')
        api_secret = os.getenv('BINANCE_API_SECRET') or getattr(self.config.binance, 'api_secret', '')
        
        if api_key and api_secret:
            # Mask for logging
            masked_key = api_key[:8] + "..." if len(api_key) > 8 else "***"
            return ValidationResult(
                check_name="Binance Credentials",
                passed=True,
                severity=ValidationSeverity.WARNING,
                message=f"API key configured ({masked_key})",
            )
        else:
            return ValidationResult(
                check_name="Binance Credentials",
                passed=False,
                severity=ValidationSeverity.WARNING,
                message="Missing BINANCE_API_KEY or BINANCE_API_SECRET",
                details={"hint": "Set in .env file or environment variables"},
            )
    
    def _check_coinbase_credentials(self) -> ValidationResult:
        """Check Coinbase API credentials"""
        api_key = os.getenv('COINBASE_API_KEY') or getattr(self.config.coinbase, 'api_key', '')
        api_secret = os.getenv('COINBASE_API_SECRET') or getattr(self.config.coinbase, 'api_secret', '')
        
        if api_key and api_secret:
            masked_key = api_key[:8] + "..." if len(api_key) > 8 else "***"
            return ValidationResult(
                check_name="Coinbase Credentials",
                passed=True,
                severity=ValidationSeverity.WARNING,
                message=f"API key configured ({masked_key})",
            )
        else:
            return ValidationResult(
                check_name="Coinbase Credentials",
                passed=False,
                severity=ValidationSeverity.WARNING,
                message="Missing COINBASE_API_KEY or COINBASE_API_SECRET",
            )
    
    def _check_kraken_credentials(self) -> ValidationResult:
        """Check Kraken API credentials"""
        api_key = os.getenv('KRAKEN_API_KEY') or getattr(self.config.kraken, 'api_key', '')
        api_secret = os.getenv('KRAKEN_API_SECRET') or getattr(self.config.kraken, 'api_secret', '')
        
        if api_key and api_secret:
            masked_key = api_key[:8] + "..." if len(api_key) > 8 else "***"
            return ValidationResult(
                check_name="Kraken Credentials",
                passed=True,
                severity=ValidationSeverity.WARNING,
                message=f"API key configured ({masked_key})",
            )
        else:
            return ValidationResult(
                check_name="Kraken Credentials",
                passed=False,
                severity=ValidationSeverity.WARNING,
                message="Missing KRAKEN_API_KEY or KRAKEN_API_SECRET",
            )
    
    def _check_database_path(self) -> ValidationResult:
        """Check database path is writable"""
        db_path = get_project_path("data", "accounting.db")
        db_dir = db_path.parent
        
        if db_dir.exists():
            # Check write permission
            try:
                test_file = db_dir / ".write_test"
                test_file.touch()
                test_file.unlink()
                return ValidationResult(
                    check_name="Database Path",
                    passed=True,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Writable: {db_dir}",
                )
            except PermissionError:
                return ValidationResult(
                    check_name="Database Path",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"No write permission: {db_dir}",
                )
        else:
            # Try to create it
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
                return ValidationResult(
                    check_name="Database Path",
                    passed=True,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Created: {db_dir}",
                )
            except Exception as e:
                return ValidationResult(
                    check_name="Database Path",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Cannot create: {e}",
                )
    
    def _check_data_directory(self) -> ValidationResult:
        """Check data directory exists"""
        data_dir = get_project_path("data")
        
        if data_dir.exists():
            return ValidationResult(
                check_name="Data Directory",
                passed=True,
                severity=ValidationSeverity.ERROR,
                message=f"Exists: {data_dir}",
            )
        else:
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
                return ValidationResult(
                    check_name="Data Directory",
                    passed=True,
                    severity=ValidationSeverity.ERROR,
                    message=f"Created: {data_dir}",
                )
            except Exception as e:
                return ValidationResult(
                    check_name="Data Directory",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Cannot create: {e}",
                )
    
    def _check_logs_directory(self) -> ValidationResult:
        """Check logs directory exists"""
        logs_dir = get_project_path("logs")
        
        if logs_dir.exists():
            return ValidationResult(
                check_name="Logs Directory",
                passed=True,
                severity=ValidationSeverity.WARNING,
                message=f"Exists: {logs_dir}",
            )
        else:
            try:
                logs_dir.mkdir(parents=True, exist_ok=True)
                return ValidationResult(
                    check_name="Logs Directory",
                    passed=True,
                    severity=ValidationSeverity.WARNING,
                    message=f"Created: {logs_dir}",
                )
            except Exception as e:
                return ValidationResult(
                    check_name="Logs Directory",
                    passed=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Cannot create: {e}",
                )
    
    async def _check_redis_connection(self) -> ValidationResult:
        """Check Redis connectivity (optional)"""
        try:
            import redis.asyncio as redis
            
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            client = redis.from_url(redis_url)
            
            await asyncio.wait_for(client.ping(), timeout=2.0)
            await client.close()
            
            return ValidationResult(
                check_name="Redis Connection",
                passed=True,
                severity=ValidationSeverity.INFO,
                message=f"Connected to {redis_url}",
            )
        except asyncio.TimeoutError:
            return ValidationResult(
                check_name="Redis Connection",
                passed=False,
                severity=ValidationSeverity.INFO,
                message="Connection timeout (Redis is optional)",
            )
        except ImportError:
            return ValidationResult(
                check_name="Redis Connection",
                passed=False,
                severity=ValidationSeverity.INFO,
                message="redis package not installed (optional)",
            )
        except Exception as e:
            return ValidationResult(
                check_name="Redis Connection",
                passed=False,
                severity=ValidationSeverity.INFO,
                message=f"Not available: {type(e).__name__} (optional)",
            )
    
    def _check_notification_config(self) -> ValidationResult:
        """Check notification configuration"""
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        
        notifications = []
        if telegram_token:
            notifications.append("Telegram")
        if slack_webhook:
            notifications.append("Slack")
        
        if notifications:
            return ValidationResult(
                check_name="Notifications",
                passed=True,
                severity=ValidationSeverity.INFO,
                message=f"Configured: {', '.join(notifications)}",
            )
        else:
            return ValidationResult(
                check_name="Notifications",
                passed=False,
                severity=ValidationSeverity.INFO,
                message="No notification channels configured (optional)",
            )
    
    def _check_infura_rpc(self) -> ValidationResult:
        """Check Infura RPC URL configuration"""
        infura_project_id = os.getenv('INFURA_PROJECT_ID', '')
        infura_url = os.getenv('INFURA_URL', '')
        
        # Check for placeholder values
        placeholder_values = [
            'YOUR_PROJECT_ID',
            'your-project-id',
            'YOUR_INFURA_KEY',
            'your-infura-key',
            '',
        ]
        
        if infura_project_id and infura_project_id not in placeholder_values:
            return ValidationResult(
                check_name="Infura RPC",
                passed=True,
                severity=ValidationSeverity.WARNING,
                message=f"Project ID configured ({infura_project_id[:8]}...)",
            )
        elif infura_url and 'YOUR_PROJECT_ID' not in infura_url:
            return ValidationResult(
                check_name="Infura RPC",
                passed=True,
                severity=ValidationSeverity.WARNING,
                message="Custom Infura URL configured",
            )
        else:
            return ValidationResult(
                check_name="Infura RPC",
                passed=False,
                severity=ValidationSeverity.WARNING,
                message="Infura not configured - Web3 features disabled",
                details={"hint": "Set INFURA_PROJECT_ID in .env or environment"},
            )
    
    def _check_trading_mode(self) -> ValidationResult:
        """Check trading mode configuration"""
        paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
        dry_run = os.getenv('DRY_RUN', 'false').lower() == 'true'
        
        if paper_trading:
            return ValidationResult(
                check_name="Trading Mode",
                passed=True,
                severity=ValidationSeverity.CRITICAL,
                message="PAPER TRADING enabled (safe mode)",
                details={"paper_trading": True, "dry_run": dry_run},
            )
        else:
            return ValidationResult(
                check_name="Trading Mode",
                passed=True,
                severity=ValidationSeverity.CRITICAL,
                message="⚠️  LIVE TRADING enabled - real money at risk!",
                details={"paper_trading": False, "dry_run": dry_run},
            )


async def validate_startup(fail_on_critical: bool = True) -> bool:
    """
    Run startup validation and optionally fail on critical errors.
    
    Args:
        fail_on_critical: If True, raises exception on critical failures
        
    Returns:
        True if all critical checks passed
    """
    validator = StartupValidator()
    validation = await validator.validate_all()
    
    # Log summary
    logger = logging.getLogger("StartupValidator")
    logger.info(validation.summary())
    
    if not validation.critical_passed:
        if fail_on_critical:
            raise RuntimeError(
                f"Startup validation failed: {len(validation.errors)} critical errors"
            )
        return False
    
    return True
