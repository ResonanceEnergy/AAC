"""
Accelerated Arbitrage Corp - Configuration Loader
==================================================
Centralized configuration management using environment variables.
Loads from .env file and provides typed access to all config values.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field

# Try to load python-dotenv, fall back gracefully if not installed
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


def _read_secret_file(secret_path: str) -> str:
    """Read a secret value from a local file path."""
    if not secret_path:
        return ''

    path = Path(secret_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    try:
        return path.read_text(encoding='utf-8').strip()
    except OSError as exc:
        logger.warning("Failed to read secret file %s: %s", path, exc)
        return ''


def _get_secret_manager_value(key: str) -> str:
    """Get a secret from the encrypted secrets manager when configured."""
    if not os.environ.get('ACC_MASTER_PASSWORD'):
        return ''

    try:
        from shared.secrets_manager import get_secrets_manager

        manager = get_secrets_manager()
        return manager.get_secret(key, '')
    except Exception as exc:
        logger.debug("Encrypted secret lookup skipped for %s: %s", key, exc)
        return ''


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
    value = os.environ.get(key, '')
    if value:
        result = value
    else:
        file_value = _read_secret_file(os.environ.get(f'{key}_FILE', ''))
        encrypted_value = _get_secret_manager_value(key) if not file_value else ''
        result = file_value or encrypted_value or default

    if required and not result:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return result


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
    api_key: str = field(default='', repr=False)
    api_secret: str = field(default='', repr=False)
    passphrase: str = field(default='', repr=False)  # For Coinbase
    testnet: bool = True
    enabled: bool = False
    
    def is_configured(self) -> bool:
        """Is configured."""
        return bool(self.api_key and self.api_secret)


@dataclass
class IBKRConfig:
    """Configuration for Interactive Brokers (socket-based, no API keys)"""
    host: str = '127.0.0.1'
    port: int = 7497
    client_id: int = 1
    account: str = ''
    paper: bool = True
    enabled: bool = False

    def is_configured(self) -> bool:
        """Is configured."""
        return bool(self.account)


@dataclass
class MoomooConfig:
    """Configuration for Moomoo/Futu (gateway-based, similar to IBKR)"""
    host: str = '127.0.0.1'
    port: int = 11111
    trade_env: str = 'SIMULATE'  # SIMULATE or REAL
    market: str = 'US'           # US, HK, CN, SG, AU, JP, CA
    security_firm: str = 'FUTUINC'
    trade_password: str = field(default='', repr=False)
    enabled: bool = False

    def is_configured(self) -> bool:
        """Is configured."""
        return self.enabled


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = ''
    redis_url: str = ''
    
    @property
    def is_sqlite(self) -> bool:
        """Is sqlite."""
        return self.url.startswith('sqlite')
    
    @property
    def is_postgres(self) -> bool:
        """Is postgres."""
        return self.url.startswith('postgresql')


@dataclass
class NotificationConfig:
    """Notification services configuration"""
    telegram_token: str = field(default='', repr=False)
    telegram_chat_id: str = ''
    slack_webhook: str = field(default='', repr=False)
    slack_channel: str = '#trading-alerts'
    discord_webhook: str = field(default='', repr=False)
    smtp_host: str = ''
    smtp_port: int = 587
    smtp_user: str = ''
    smtp_password: str = field(default='', repr=False)
    email_to: str = ''
    
    def telegram_enabled(self) -> bool:
        """Telegram enabled."""
        return bool(self.telegram_token and self.telegram_chat_id)
    
    def slack_enabled(self) -> bool:
        """Slack enabled."""
        return bool(self.slack_webhook)
    
    def email_enabled(self) -> bool:
        """Email enabled."""
        return bool(self.smtp_host and self.smtp_user and self.email_to)


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size_usd: float = 10000.0
    max_daily_loss_usd: float = 1000.0
    max_open_positions: int = 10
    dry_run: bool = True
    paper_trading: bool = True
    live_trading_confirmation: str = ''  # Must be "YES_I_UNDERSTAND" to go live
    # Per-strategy allocation limits (percentage of total capital, 0-100)
    strategy_max_allocation_pct: float = 25.0  # No single strategy gets >25%
    max_daily_trades: int = 100  # Circuit-breaker on runaway strategies


@dataclass 
class Config:
    """Main configuration container"""

    # Fields whose values must never appear in logs, repr, or str output.
    _SENSITIVE_FIELDS: frozenset = field(
        default_factory=lambda: frozenset({
            'eth_private_key', 'api_key', 'api_secret', 'passphrase',
            'bigbrain_token', 'ncc_token', 'dashboard_secret',
            'coingecko_key', 'coinmarketcap_key', 'alphavantage_key',
            'news_api_key', 'twitter_bearer', 'reddit_client_secret',
            'kyc_provider_key', 'telegram_token', 'slack_webhook',
            'discord_webhook', 'smtp_password',
            'openclaw_gateway_token', 'clawhub_api_key', 'aac_api_key',
        }),

        repr=False,
    )

    # Environment
    environment: str = 'development'
    debug: bool = False
    log_level: str = 'INFO'
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    
    # Exchanges
    binance: ExchangeConfig = field(default_factory=ExchangeConfig)
    coinbase: ExchangeConfig = field(default_factory=ExchangeConfig)
    kraken: ExchangeConfig = field(default_factory=ExchangeConfig)
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    moomoo: MoomooConfig = field(default_factory=MoomooConfig)
    ndax: ExchangeConfig = field(default_factory=ExchangeConfig)
    metalx: ExchangeConfig = field(default_factory=ExchangeConfig)
    
    # IBKR-specific
    ibkr_host: str = '127.0.0.1'
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    ibkr_account: str = ''
    
    # NDAX-specific
    ndax_user_id: str = ''
    ndax_account_id: str = ''
    
    # Moomoo-specific
    moomoo_paper: bool = True
    
    # MT5 / Noxi Rise
    mt5_path: str = ''
    mt5_login: int = 0
    mt5_password: str = field(default='', repr=False)
    mt5_server: str = 'NoxiRise-Live'
    
    # Metal X / Metallicus
    metalx_account_name: str = ''
    metalx_private_key: str = field(default='', repr=False)
    metal_blockchain_rpc_url: str = ''
    xpr_rpc_url: str = ''
    xpr_account_name: str = ''
    xpr_private_key: str = field(default='', repr=False)
    metalpay_api_key: str = field(default='', repr=False)
    metalpay_api_secret: str = field(default='', repr=False)
    webauth_app_id: str = 'aac_trading'
    webauth_callback_url: str = ''
    
    # Foreign Exchange (Knightsbridge FX)
    fx_api_key: str = field(default='', repr=False)
    fx_spread_bps: float = 50.0
    fx_poll_interval: int = 60

    # Web3
    eth_private_key: str = field(default='', repr=False)
    eth_rpc_url: str = ''
    polygon_rpc_url: str = ''
    arbitrum_rpc_url: str = ''
    
    # Services
    # DEV-ONLY defaults — overridden by from_env() in production
    bigbrain_url: str = 'http://localhost:8001/api/v1'
    bigbrain_token: str = field(default='', repr=False)
    crypto_intel_url: str = 'http://localhost:8002/api/v1'
    accounting_url: str = 'http://localhost:8003/api/v1'
    ncc_endpoint: str = 'http://localhost:8000/api/v1'
    ncc_token: str = field(default='', repr=False)
    
    # Database
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Notifications
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    
    # Risk
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    # External APIs
    coingecko_key: str = field(default='', repr=False)
    coinmarketcap_key: str = field(default='', repr=False)
    alphavantage_key: str = field(default='', repr=False)
    news_api_key: str = field(default='', repr=False)
    twitter_bearer: str = field(default='', repr=False)
    reddit_client_id: str = ''
    reddit_client_secret: str = field(default='', repr=False)
    reddit_user_agent: str = 'AAC-Trading-Bot/1.0'
    kyc_provider_key: str = field(default='', repr=False)
    kyc_provider_url: str = ''
    unusual_whales_key: str = field(default='', repr=False)
    eodhd_key: str = field(default='', repr=False)
    polygon_key: str = field(default='', repr=False)
    finnhub_key: str = field(default='', repr=False)
    tradier_key: str = field(default='', repr=False)
    fred_key: str = field(default='', repr=False)
    whale_alert_key: str = field(default='', repr=False)
    santiment_key: str = field(default='', repr=False)
    twelve_data_key: str = field(default='', repr=False)
    iex_cloud_key: str = field(default='', repr=False)
    intrinio_key: str = field(default='', repr=False)
    intrinio_username: str = ''
    etherscan_key: str = field(default='', repr=False)
    tradestie_key: str = field(default='', repr=False)
    wallstreetodds_key: str = field(default='', repr=False)
    
    # AI/LLM APIs
    openai_key: str = field(default='', repr=False)
    anthropic_key: str = field(default='', repr=False)
    google_ai_key: str = field(default='', repr=False)
    xai_key: str = field(default='', repr=False)
    
    # Polymarket (prediction markets)
    polymarket_private_key: str = field(default='', repr=False)
    polymarket_funder: str = ''
    polymarket_chain_id: int = 137  # Polygon
    
    # Dashboard
    # DEV-ONLY defaults — overridden by from_env() in production
    dashboard_url: str = 'http://localhost:3000'
    dashboard_secret: str = field(default='', repr=False)
    
    # OpenClaw / ClawHub
    openclaw_skills_dir: str = ''
    openclaw_gateway_url: str = 'ws://127.0.0.1:18789'
    openclaw_gateway_token: str = field(default='', repr=False)
    openclaw_daily_spend_limit: float = 10.0
    clawhub_api_key: str = field(default='', repr=False)
    aac_api_key: str = field(default='', repr=False)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        load_env_file()
        
        config = cls(
            # Environment
            environment=get_env('ENVIRONMENT', 'development'),
            debug=get_env_bool('DEBUG', False),
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
            
            # Interactive Brokers
            ibkr=IBKRConfig(
                host=get_env('IBKR_HOST', '127.0.0.1'),
                port=get_env_int('IBKR_PORT', 7497),
                client_id=get_env_int('IBKR_CLIENT_ID', 1),
                account=get_env('IBKR_ACCOUNT', ''),
                paper=get_env_bool('IBKR_PAPER', True),
                enabled=bool(get_env('IBKR_ACCOUNT')),
            ),
            
            # Moomoo/Futu
            moomoo=MoomooConfig(
                host=get_env('MOOMOO_HOST', '127.0.0.1'),
                port=get_env_int('MOOMOO_PORT', 11111),
                trade_env=get_env('MOOMOO_TRADE_ENV', 'SIMULATE'),
                market=get_env('MOOMOO_MARKET', 'US'),
                security_firm=get_env('MOOMOO_SECURITY_FIRM', 'FUTUINC'),
                trade_password=get_env('MOOMOO_TRADE_PASSWORD'),
                enabled=get_env_bool('MOOMOO_ENABLED', False),
            ),
            
            # NDAX (Canadian crypto exchange)
            ndax=ExchangeConfig(
                api_key=get_env('NDAX_API_KEY'),
                api_secret=get_env('NDAX_API_SECRET'),
                testnet=get_env_bool('NDAX_TESTNET', True),
                enabled=bool(get_env('NDAX_API_KEY')),
            ),
            ndax_user_id=get_env('NDAX_USER_ID'),
            ndax_account_id=get_env('NDAX_ACCOUNT_ID'),
            
            moomoo_paper=get_env_bool('MOOMOO_PAPER', True),
            
            # Metal X
            metalx=ExchangeConfig(
                api_key=get_env('METALX_ACCOUNT_NAME'),
                api_secret=get_env('METALX_PRIVATE_KEY'),
                enabled=bool(get_env('METALX_ACCOUNT_NAME')),
            ),
            metalx_account_name=get_env('METALX_ACCOUNT_NAME'),
            metalx_private_key=get_env('METALX_PRIVATE_KEY'),
            metal_blockchain_rpc_url=get_env('METAL_BLOCKCHAIN_RPC_URL', ''),
            xpr_rpc_url=get_env('XPR_RPC_URL', ''),
            xpr_account_name=get_env('XPR_ACCOUNT_NAME'),
            xpr_private_key=get_env('XPR_PRIVATE_KEY'),
            metalpay_api_key=get_env('METALPAY_API_KEY'),
            metalpay_api_secret=get_env('METALPAY_API_SECRET'),
            webauth_app_id=get_env('WEBAUTH_APP_ID', 'aac_trading'),
            webauth_callback_url=get_env('WEBAUTH_CALLBACK_URL'),
            
            # Foreign Exchange (Knightsbridge FX)
            fx_api_key=get_env('FX_API_KEY'),
            fx_spread_bps=get_env_float('FX_SPREAD_BPS', 50.0),
            fx_poll_interval=get_env_int('FX_POLL_INTERVAL', 60),

            # MT5 / Noxi Rise
            mt5_path=get_env('MT5_PATH'),
            mt5_login=get_env_int('MT5_LOGIN', 0),
            mt5_password=get_env('MT5_PASSWORD'),
            mt5_server=get_env('MT5_SERVER', 'NoxiRise-Live'),
            
            # Web3
            eth_private_key=get_env('ETH_PRIVATE_KEY'),
            eth_rpc_url=get_env('ETH_RPC_URL', ''),
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
                smtp_host=get_env('SMTP_HOST', ''),
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
                live_trading_confirmation=get_env('LIVE_TRADING_CONFIRMATION', ''),
                strategy_max_allocation_pct=get_env_float('STRATEGY_MAX_ALLOCATION_PCT', 25.0),
                max_daily_trades=get_env_int('MAX_DAILY_TRADES', 100),
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
            unusual_whales_key=get_env('UNUSUAL_WHALES_API_KEY'),
            eodhd_key=get_env('EODHD_API_KEY'),
            polygon_key=get_env('POLYGON_API_KEY'),
            finnhub_key=get_env('FINNHUB_API_KEY'),
            tradier_key=get_env('TRADIER_API_KEY'),
            fred_key=get_env('FRED_API_KEY'),
            whale_alert_key=get_env('WHALE_ALERT_API_KEY'),
            santiment_key=get_env('SANTIMENT_API_KEY'),
            twelve_data_key=get_env('TWELVE_DATA_API_KEY'),
            iex_cloud_key=get_env('IEX_CLOUD_API_KEY'),
            intrinio_key=get_env('INTRINIO_API_KEY'),
            intrinio_username=get_env('INTRINIO_USERNAME'),
            etherscan_key=get_env('ETHERSCAN_API_KEY'),
            tradestie_key=get_env('TRADESTIE_API_KEY'),
            wallstreetodds_key=get_env('WALLSTREETODDS_API_KEY'),
            
            # AI/LLM APIs
            openai_key=get_env('OPENAI_API_KEY'),
            anthropic_key=get_env('ANTHROPIC_API_KEY'),
            google_ai_key=get_env('GOOGLE_AI_API_KEY'),
            xai_key=get_env('XAI_API_KEY'),
            
            # Polymarket
            polymarket_private_key=get_env('POLYMARKET_PRIVATE_KEY'),
            polymarket_funder=get_env('POLYMARKET_FUNDER_ADDRESS'),
            polymarket_chain_id=int(get_env('POLYMARKET_CHAIN_ID', '137')),
            
            # Dashboard
            dashboard_url=get_env('DASHBOARD_URL', 'http://localhost:3000'),
            dashboard_secret=get_env('DASHBOARD_SECRET_KEY'),
            
            # OpenClaw / ClawHub
            openclaw_skills_dir=get_env('OPENCLAW_SKILLS_DIR'),
            openclaw_gateway_url=get_env('OPENCLAW_GATEWAY_URL', 'ws://127.0.0.1:18789'),
            openclaw_gateway_token=get_env('OPENCLAW_GATEWAY_TOKEN'),
            openclaw_daily_spend_limit=get_env_float('OPENCLAW_DAILY_SPEND_LIMIT', 10.0),
            clawhub_api_key=get_env('CLAWHUB_API_KEY'),
            aac_api_key=get_env('AAC_API_KEY'),
        )

        # Post-load validation for risk config
        if config.risk.max_position_size_usd <= 0:
            logger.warning(f"Invalid max_position_size_usd: {config.risk.max_position_size_usd}, using default 10000.0")
            config.risk.max_position_size_usd = 10000.0
        if config.risk.max_daily_loss_usd <= 0:
            logger.warning(f"Invalid max_daily_loss_usd: {config.risk.max_daily_loss_usd}, using default 1000.0")
            config.risk.max_daily_loss_usd = 1000.0

        return config
    
    def get_enabled_exchanges(self) -> Dict[str, Union[ExchangeConfig, IBKRConfig, MoomooConfig]]:
        """Return dict of exchanges that have API keys configured"""
        exchanges: Dict[str, Union[ExchangeConfig, IBKRConfig, MoomooConfig]] = {}
        if self.binance.is_configured():
            exchanges['binance'] = self.binance
        if self.coinbase.is_configured():
            exchanges['coinbase'] = self.coinbase
        if self.kraken.is_configured():
            exchanges['kraken'] = self.kraken
        if self.ibkr.is_configured():
            exchanges['ibkr'] = self.ibkr
        if self.moomoo.is_configured():
            exchanges['moomoo'] = self.moomoo
        if self.metalx.is_configured():
            exchanges['metalx'] = self.metalx
        return exchanges
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return status report"""
        issues = []
        warnings = []
        
        enabled = self.get_enabled_exchanges()
        paper = self.risk.paper_trading or self.risk.dry_run
        
        # Critical: no exchanges AND not paper trading = system does nothing
        if not enabled and not paper:
            issues.append(
                "No exchange API keys configured and PAPER_TRADING/DRY_RUN not enabled. "
                "Set at least one exchange API key or DRY_RUN=true in .env"
            )
        elif not enabled:
            warnings.append("No exchange API keys configured - running in paper/dry-run mode only")
        
        # Check database
        if not self.database.url:
            issues.append("DATABASE_URL not configured")
        
        # Check if dry run
        if self.risk.dry_run:
            warnings.append("DRY_RUN=true - no real trades will execute")
        
        # Live trading safety guard
        if not paper and not self.risk.dry_run:
            if self.risk.live_trading_confirmation != 'YES_I_UNDERSTAND':
                issues.append(
                    "Live trading requires LIVE_TRADING_CONFIRMATION=YES_I_UNDERSTAND in .env. "
                    "This confirms you accept the risk of real money trading."
                )
        
        # Check notifications
        if not (self.notifications.telegram_enabled() or 
                self.notifications.slack_enabled() or 
                self.notifications.email_enabled()):
            warnings.append("No notification channels configured")
        
        # Check market data sources
        has_market_data = any([
            self.coingecko_key,
            self.alphavantage_key,
            self.finnhub_key,
        ])
        if not has_market_data:
            warnings.append(
                "No market data API keys configured (COINGECKO_API_KEY, ALPHAVANTAGE_API_KEY, FINNHUB_API_KEY). "
                "Market data feeds will be limited"
            )
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'exchanges_configured': list(enabled.keys()),
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
    SCHEMAS: Dict[str, Dict[str, Any]] = {
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
            expected_type: Any = schema.get("type")
            if expected_type:
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):  # type: ignore[arg-type]
                        errors.append(
                            f"{field_path}: Expected {expected_type}, got {type(value).__name__}"
                        )
                        continue
                elif not isinstance(value, expected_type):  # type: ignore[arg-type]
                    errors.append(
                        f"{field_path}: Expected {expected_type.__name__}, got {type(value).__name__}"
                    )
                    continue
            
            # Check allowed values
            allowed: Any = schema.get("allowed")
            if allowed is not None and value not in allowed:
                errors.append(
                    f"{field_path}: Value '{value}' not in allowed values: {allowed}"
                )
            
            # Check numeric range
            if isinstance(value, (int, float)):
                min_val: Any = schema.get("min")
                max_val: Any = schema.get("max")
                if min_val is not None and value < min_val:
                    errors.append(f"{field_path}: Value {value} below minimum {min_val}")
                if max_val is not None and value > max_val:
                    errors.append(f"{field_path}: Value {value} above maximum {max_val}")
            
            # Check pattern
            pattern: Any = schema.get("pattern")
            if pattern is not None and isinstance(value, str) and value:
                if not re.match(pattern, value):
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
        import yaml  # type: ignore[import-untyped]
        
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


def validate_startup_requirements(config: Optional['Config'] = None) -> bool:
    """
    Run startup validation and log results.
    
    Returns True if config is valid for operation, False if fatal issues found.
    Logs warnings for non-fatal issues.
    """
    if config is None:
        config = get_config()
    
    result = config.validate()
    
    for warning in result.get('warnings', []):
        logger.warning(f"[CONFIG] {warning}")
    
    for issue in result.get('issues', []):
        logger.error(f"[CONFIG] {issue}")
    
    if result['valid']:
        exchanges = result.get('exchanges_configured', [])
        logger.info(
            f"[CONFIG] Startup validation passed. "
            f"Environment={result['environment']}, "
            f"Exchanges={exchanges or 'none (paper mode)'}, "
            f"DryRun={result['dry_run']}"
        )
    else:
        logger.error(
            f"[CONFIG] Startup validation FAILED with {len(result['issues'])} issue(s). "
            f"Fix .env configuration before launching."
        )
    
    return result['valid']


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
