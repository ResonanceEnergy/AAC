"""
Accelerated Arbitrage Corp - Configuration Loader
==================================================
Centralized configuration management using environment variables.
Loads from .env file and provides typed access to all config values.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# Try to load python-dotenv, fall back gracefully if not installed
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Find the project root directory by looking for .env or .env.example"""
    current = Path(__file__).resolve().parent
    
    for _ in range(5):  # Max 5 levels up
        if (current / '.env').exists() or (current / '.env.example').exists():
            return current
        if (current / 'parallel_orchestrator.py').exists():  # Known root file
            return current
        current = current.parent
    
    # Fallback to parent of shared/
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = find_project_root()


def load_env_file(env_path: Optional[Path] = None) -> bool:
    """Load environment variables from .env file"""
    if env_path is None:
        env_path = PROJECT_ROOT / '.env'
    
    if not env_path.exists():
        logger.warning(f".env file not found at {env_path}. Using system environment variables only.")
        logger.info("Copy .env.example to .env and configure your values.")
        return False
    
    if DOTENV_AVAILABLE:
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
        return True
    else:
        # Manual .env parsing fallback
        logger.warning("python-dotenv not installed. Using manual .env parsing.")
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            return True
        except Exception as e:
            logger.error(f"Failed to parse .env file: {e}")
            return False


def get_env(key: str, default: str = '', required: bool = False) -> str:
    """Get environment variable with optional default and required check"""
    value = os.environ.get(key, default)
    if required and not value:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable"""
    value = os.environ.get(key, '').lower()
    if value in ('true', '1', 'yes', 'on'):
        return True
    if value in ('false', '0', 'no', 'off'):
        return False
    return default


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable"""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable"""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


@dataclass
class ExchangeConfig:
    """Configuration for a single exchange"""
    api_key: str = ''
    api_secret: str = ''
    passphrase: str = ''  # For Coinbase
    testnet: bool = True
    enabled: bool = False
    
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = ''
    redis_url: str = ''
    
    @property
    def is_sqlite(self) -> bool:
        return self.url.startswith('sqlite')
    
    @property
    def is_postgres(self) -> bool:
        return self.url.startswith('postgresql')


@dataclass
class NotificationConfig:
    """Notification services configuration"""
    telegram_token: str = ''
    telegram_chat_id: str = ''
    slack_webhook: str = ''
    slack_channel: str = '#trading-alerts'
    discord_webhook: str = ''
    smtp_host: str = ''
    smtp_port: int = 587
    smtp_user: str = ''
    smtp_password: str = ''
    email_to: str = ''
    
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)
    
    def slack_enabled(self) -> bool:
        return bool(self.slack_webhook)
    
    def email_enabled(self) -> bool:
        return bool(self.smtp_host and self.smtp_user and self.email_to)


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size_usd: float = 10000.0
    max_daily_loss_usd: float = 1000.0
    max_open_positions: int = 10
    dry_run: bool = True
    paper_trading: bool = True


@dataclass 
class Config:
    """Main configuration container"""
    # Environment
    environment: str = 'development'
    debug: bool = True
    log_level: str = 'INFO'
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    
    # Exchanges
    binance: ExchangeConfig = field(default_factory=ExchangeConfig)
    coinbase: ExchangeConfig = field(default_factory=ExchangeConfig)
    kraken: ExchangeConfig = field(default_factory=ExchangeConfig)
    
    # Web3
    eth_private_key: str = ''
    eth_rpc_url: str = ''
    polygon_rpc_url: str = ''
    arbitrum_rpc_url: str = ''
    
    # Services
    bigbrain_url: str = 'http://localhost:8001/api/v1'
    bigbrain_token: str = ''
    crypto_intel_url: str = 'http://localhost:8002/api/v1'
    accounting_url: str = 'http://localhost:8003/api/v1'
    ncc_endpoint: str = 'http://localhost:8000/api/v1'
    ncc_token: str = ''
    
    # Database
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Notifications
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    
    # Risk
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    # External APIs
    coingecko_key: str = ''
    coinmarketcap_key: str = ''
    alphavantage_key: str = ''
    news_api_key: str = ''
    twitter_bearer: str = ''
    reddit_client_id: str = ''
    reddit_client_secret: str = ''
    reddit_user_agent: str = 'AAC-Trading-Bot/1.0'
    kyc_provider_key: str = ''
    kyc_provider_url: str = ''
    
    # Dashboard
    dashboard_url: str = 'http://localhost:3000'
    dashboard_secret: str = ''
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        load_env_file()
        
        return cls(
            # Environment
            environment=get_env('ENVIRONMENT', 'development'),
            debug=get_env_bool('DEBUG', True),
            log_level=get_env('LOG_LEVEL', 'INFO'),
            
            # Binance
            binance=ExchangeConfig(
                api_key=get_env('BINANCE_API_KEY'),
                api_secret=get_env('BINANCE_API_SECRET'),
                testnet=get_env_bool('BINANCE_TESTNET', True),
                enabled=bool(get_env('BINANCE_API_KEY')),
            ),
            
            # Coinbase
            coinbase=ExchangeConfig(
                api_key=get_env('COINBASE_API_KEY'),
                api_secret=get_env('COINBASE_API_SECRET'),
                passphrase=get_env('COINBASE_PASSPHRASE'),
                enabled=bool(get_env('COINBASE_API_KEY')),
            ),
            
            # Kraken
            kraken=ExchangeConfig(
                api_key=get_env('KRAKEN_API_KEY'),
                api_secret=get_env('KRAKEN_API_SECRET'),
                enabled=bool(get_env('KRAKEN_API_KEY')),
            ),
            
            # Web3
            eth_private_key=get_env('ETH_PRIVATE_KEY'),
            eth_rpc_url=get_env('ETH_RPC_URL', 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID'),
            polygon_rpc_url=get_env('POLYGON_RPC_URL', 'https://polygon-rpc.com'),
            arbitrum_rpc_url=get_env('ARBITRUM_RPC_URL', 'https://arb1.arbitrum.io/rpc'),
            
            # Services
            bigbrain_url=get_env('BIGBRAIN_API_URL', 'http://localhost:8001/api/v1'),
            bigbrain_token=get_env('BIGBRAIN_AUTH_TOKEN'),
            crypto_intel_url=get_env('CRYPTO_INTEL_API_URL', 'http://localhost:8002/api/v1'),
            accounting_url=get_env('ACCOUNTING_API_URL', 'http://localhost:8003/api/v1'),
            ncc_endpoint=get_env('NCC_COORDINATOR_ENDPOINT', 'http://localhost:8000/api/v1'),
            ncc_token=get_env('NCC_AUTH_TOKEN'),
            
            # Database
            database=DatabaseConfig(
                url=get_env('DATABASE_URL', f'sqlite:///{PROJECT_ROOT}/CentralAccounting/data/accounting.db'),
                redis_url=get_env('REDIS_URL', 'redis://localhost:6379/0'),
            ),
            
            # Notifications
            notifications=NotificationConfig(
                telegram_token=get_env('TELEGRAM_BOT_TOKEN'),
                telegram_chat_id=get_env('TELEGRAM_CHAT_ID'),
                slack_webhook=get_env('SLACK_WEBHOOK_URL'),
                slack_channel=get_env('SLACK_CHANNEL', '#trading-alerts'),
                discord_webhook=get_env('DISCORD_WEBHOOK_URL'),
                smtp_host=get_env('SMTP_HOST', 'smtp.gmail.com'),
                smtp_port=get_env_int('SMTP_PORT', 587),
                smtp_user=get_env('SMTP_USER'),
                smtp_password=get_env('SMTP_PASSWORD'),
                email_to=get_env('ALERT_EMAIL_TO'),
            ),
            
            # Risk
            risk=RiskConfig(
                max_position_size_usd=get_env_float('MAX_POSITION_SIZE_USD', 10000.0),
                max_daily_loss_usd=get_env_float('MAX_DAILY_LOSS_USD', 1000.0),
                max_open_positions=get_env_int('MAX_OPEN_POSITIONS', 10),
                dry_run=get_env_bool('DRY_RUN', True),
                paper_trading=get_env_bool('PAPER_TRADING', True),
            ),
            
            # External APIs
            coingecko_key=get_env('COINGECKO_API_KEY'),
            coinmarketcap_key=get_env('COINMARKETCAP_API_KEY'),
            alphavantage_key=get_env('ALPHAVANTAGE_API_KEY'),
            news_api_key=get_env('NEWS_API_KEY'),
            twitter_bearer=get_env('TWITTER_BEARER_TOKEN'),
            reddit_client_id=get_env('REDDIT_CLIENT_ID'),
            reddit_client_secret=get_env('REDDIT_CLIENT_SECRET'),
            reddit_user_agent=get_env('REDDIT_USER_AGENT', 'AAC-Trading-Bot/1.0'),
            kyc_provider_key=get_env('KYC_PROVIDER_API_KEY'),
            kyc_provider_url=get_env('KYC_PROVIDER_URL'),
            
            # Dashboard
            dashboard_url=get_env('DASHBOARD_URL', 'http://localhost:3000'),
            dashboard_secret=get_env('DASHBOARD_SECRET_KEY'),
        )
    
    def get_enabled_exchanges(self) -> Dict[str, ExchangeConfig]:
        """Return dict of exchanges that have API keys configured"""
        exchanges = {}
        if self.binance.is_configured():
            exchanges['binance'] = self.binance
        if self.coinbase.is_configured():
            exchanges['coinbase'] = self.coinbase
        if self.kraken.is_configured():
            exchanges['kraken'] = self.kraken
        return exchanges
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return status report"""
        issues = []
        warnings = []
        
        # Check exchanges
        if not self.get_enabled_exchanges():
            warnings.append("No exchange API keys configured - trading disabled")
        
        # Check database
        if not self.database.url:
            issues.append("DATABASE_URL not configured")
        
        # Check if dry run
        if self.risk.dry_run:
            warnings.append("DRY_RUN=true - no real trades will execute")
        
        # Check notifications
        if not (self.notifications.telegram_enabled() or 
                self.notifications.slack_enabled() or 
                self.notifications.email_enabled()):
            warnings.append("No notification channels configured")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'exchanges_configured': list(self.get_enabled_exchanges().keys()),
            'environment': self.environment,
            'dry_run': self.risk.dry_run,
        }


class ConfigValidationError(Exception):
    """Raised when config validation fails"""
    def __init__(self, errors: list):
        self.errors = errors
        super().__init__(f"Configuration validation failed: {len(errors)} error(s)")


class ConfigSchema:
    """
    Schema-based configuration validation.
    
    Validates configuration values against defined schemas with type checking,
    range validation, and required field enforcement.
    """
    
    # Schema definitions for each config section
    SCHEMAS = {
        "environment": {
            "type": str,
            "allowed": ["development", "staging", "production"],
            "required": True,
        },
        "log_level": {
            "type": str,
            "allowed": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "required": False,
            "default": "INFO",
        },
        "risk.max_position_size_usd": {
            "type": (int, float),
            "min": 0,
            "max": 1000000,
            "required": False,
        },
        "risk.max_daily_loss_usd": {
            "type": (int, float),
            "min": 0,
            "max": 100000,
            "required": False,
        },
        "risk.max_open_positions": {
            "type": int,
            "min": 1,
            "max": 100,
            "required": False,
        },
        "database.url": {
            "type": str,
            "pattern": r"^(sqlite|postgresql|mysql)://",
            "required": False,
        },
        "notifications.telegram_chat_id": {
            "type": str,
            "pattern": r"^-?\d+$",  # Telegram chat IDs are numeric
            "required": False,
        },
        "notifications.slack_webhook": {
            "type": str,
            "pattern": r"^https://hooks\.slack\.com/",
            "required": False,
        },
        "notifications.smtp_port": {
            "type": int,
            "min": 1,
            "max": 65535,
            "required": False,
        },
    }
    
    @classmethod
    def validate(cls, config: 'Config', strict: bool = False) -> Dict[str, Any]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration object to validate
            strict: If True, raise ConfigValidationError on any error
            
        Returns:
            Dict with validation results
        """
        import re
        
        errors = []
        warnings = []
        
        def get_nested_value(obj, path: str):
            """Get nested attribute by dotted path"""
            parts = path.split(".")
            current = obj
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return None
            return current
        
        for field_path, schema in cls.SCHEMAS.items():
            value = get_nested_value(config, field_path)
            
            # Check required
            if schema.get("required") and value in (None, ""):
                errors.append(f"{field_path}: Required field is missing")
                continue
            
            if value is None or value == "":
                continue  # Skip optional empty fields
            
            # Check type
            expected_type = schema.get("type")
            if expected_type:
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        errors.append(
                            f"{field_path}: Expected {expected_type}, got {type(value).__name__}"
                        )
                        continue
                elif not isinstance(value, expected_type):
                    errors.append(
                        f"{field_path}: Expected {expected_type.__name__}, got {type(value).__name__}"
                    )
                    continue
            
            # Check allowed values
            if "allowed" in schema and value not in schema["allowed"]:
                errors.append(
                    f"{field_path}: Value '{value}' not in allowed values: {schema['allowed']}"
                )
            
            # Check numeric range
            if isinstance(value, (int, float)):
                if "min" in schema and value < schema["min"]:
                    errors.append(f"{field_path}: Value {value} below minimum {schema['min']}")
                if "max" in schema and value > schema["max"]:
                    errors.append(f"{field_path}: Value {value} above maximum {schema['max']}")
            
            # Check pattern
            if "pattern" in schema and isinstance(value, str) and value:
                if not re.match(schema["pattern"], value):
                    warnings.append(
                        f"{field_path}: Value doesn't match expected pattern {schema['pattern']}"
                    )
        
        # Additional cross-field validations
        if config.risk.dry_run and config.environment == "production":
            warnings.append(
                "dry_run=True in production environment - no actual trades will execute"
            )
        
        if config.risk.paper_trading and not config.risk.dry_run:
            warnings.append(
                "paper_trading=True but dry_run=False - this is unusual"
            )
        
        if config.environment == "production":
            # Production-specific validations
            if not any([
                config.notifications.telegram_enabled(),
                config.notifications.slack_enabled(),
                config.notifications.email_enabled(),
            ]):
                errors.append(
                    "Production environment requires at least one notification channel"
                )
            
            if config.risk.max_position_size_usd > 50000:
                warnings.append(
                    f"High max_position_size_usd={config.risk.max_position_size_usd} in production"
                )
        
        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "fields_validated": len(cls.SCHEMAS),
        }
        
        if strict and errors:
            raise ConfigValidationError(errors)
        
        return result
    
    @classmethod
    def validate_yaml_file(cls, yaml_path: Path) -> Dict[str, Any]:
        """
        Validate a YAML configuration file before loading.
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            Dict with validation results
        """
        import yaml
        
        if not yaml_path.exists():
            return {
                "valid": False,
                "errors": [f"Config file not found: {yaml_path}"],
                "warnings": [],
            }
        
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if not isinstance(data, dict):
                return {
                    "valid": False,
                    "errors": ["YAML root must be a dictionary"],
                    "warnings": [],
                }
            
            # Check for unknown top-level keys
            known_keys = {
                "environment", "debug", "log_level", "binance", "coinbase", "kraken",
                "database", "notifications", "risk", "services", "web3", "apis"
            }
            unknown = set(data.keys()) - known_keys
            warnings = [f"Unknown config key: {k}" for k in unknown]
            
            return {
                "valid": True,
                "errors": [],
                "warnings": warnings,
                "keys_found": list(data.keys()),
            }
            
        except yaml.YAMLError as e:
            return {
                "valid": False,
                "errors": [f"YAML parse error: {e}"],
                "warnings": [],
            }


def validate_config(config: Optional['Config'] = None, strict: bool = False) -> Dict[str, Any]:
    """
    Validate configuration with schema.
    
    Args:
        config: Config object to validate (uses global if None)
        strict: If True, raise exception on errors
        
    Returns:
        Validation results dict
    """
    if config is None:
        config = get_config()
    
    return ConfigSchema.validate(config, strict=strict)


# Global config instance - lazy loaded
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reload_config() -> Config:
    """Force reload configuration from environment"""
    global _config
    _config = Config.from_env()
    return _config


# Convenience exports
def get_project_path(*parts: str) -> Path:
    """Get absolute path relative to project root"""
    return PROJECT_ROOT.joinpath(*parts)


if __name__ == '__main__':
    # Test configuration loading
    logging.basicConfig(level=logging.INFO)
    
    config = get_config()
    validation = config.validate()
    
    print("\n=== ACC Configuration Status ===")
    print(f"Environment: {config.environment}")
    print(f"Debug: {config.debug}")
    print(f"Project Root: {config.project_root}")
    print(f"\nExchanges Configured: {validation['exchanges_configured'] or 'None'}")
    print(f"Dry Run: {validation['dry_run']}")
    print(f"\nValidation: {'PASSED' if validation['valid'] else 'FAILED'}")
    
    if validation['issues']:
        print("\nIssues:")
        for issue in validation['issues']:
            print(f"  [CROSS] {issue}")
    
    if validation['warnings']:
        print("\nWarnings:")
        for warning in validation['warnings']:
            print(f"  [WARN]️ {warning}")
