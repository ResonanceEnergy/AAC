"""
AAC Configuration Loader — DEPRECATED
═══════════════════════════════════════

**DEPRECATED:** Use ``shared.config_loader`` (``get_config()``) instead.
This file is kept for backward compatibility only.

Loads configuration from environment variables, .env files,
and config files with validation and defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class ExchangeConfig:
    """Exchange API configuration."""
    name: str
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    passphrase: str = ""  # Coinbase


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size_usd: float = 10_000.0
    max_daily_loss_usd: float = 1_000.0
    max_drawdown_percent: float = 0.05
    default_risk_percent: float = 0.01
    emergency_stop_loss: float = 0.10


@dataclass
class OpenClawConfig:
    """OpenClaw / BARREN WUFFET configuration."""
    skills_dir: str = ""
    daily_spend_limit: float = 10.0
    model: str = "claude-opus"
    provider: str = "anthropic"


@dataclass
class AACConfig:
    """Master AAC configuration."""
    env: str = "development"
    paper_trading: bool = True
    live_trading_enabled: bool = False
    debug: bool = False

    # Sub-configs
    risk: RiskConfig = field(default_factory=RiskConfig)
    openclaw: OpenClawConfig = field(default_factory=OpenClawConfig)
    exchanges: Dict[str, ExchangeConfig] = field(default_factory=dict)

    # Infrastructure
    database_url: str = "sqlite:///data/aac.db"
    # NOTE: Redis and Kafka are NOT deployed. These placeholders are retained
    # for future infrastructure expansion only. Current transport uses
    # NCC Relay (HTTP :8787) with NDJSON outbox fallback.
    redis_url: str = ""   # UNUSED — set REDIS_URL env var when deployed
    kafka_broker: str = ""  # UNUSED — set KAFKA_BROKER env var when deployed

    # Notifications
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    slack_webhook_url: str = ""

    # AI/LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_ai_api_key: str = ""


def load_config() -> AACConfig:
    """Load configuration from environment variables.

    Returns:
        AACConfig with all settings populated from .env / environment.
    """
    cfg = AACConfig(
        env=os.getenv("AAC_ENV", "development"),
        paper_trading=os.getenv("PAPER_TRADING", "true").lower() == "true",
        live_trading_enabled=os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true",
        debug=os.getenv("DEBUG", "false").lower() == "true",
        database_url=os.getenv("DATABASE_URL", "sqlite:///data/aac.db"),
        redis_url=os.getenv("REDIS_URL", ""),  # UNUSED — no Redis deployed
        kafka_broker=os.getenv("KAFKA_BROKER", ""),  # UNUSED — no Kafka deployed
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        google_ai_api_key=os.getenv("GOOGLE_AI_API_KEY", ""),
    )

    # Risk config
    cfg.risk = RiskConfig(
        max_position_size_usd=float(os.getenv("MAX_POSITION_SIZE_USD", "10000")),
        max_daily_loss_usd=float(os.getenv("MAX_DAILY_LOSS_USD", "1000")),
        max_drawdown_percent=float(os.getenv("MAX_DRAWDOWN_PERCENT", "0.05")),
        default_risk_percent=float(os.getenv("DEFAULT_RISK_PERCENT", "0.01")),
        emergency_stop_loss=float(os.getenv("EMERGENCY_STOP_LOSS", "0.10")),
    )

    # OpenClaw config
    cfg.openclaw = OpenClawConfig(
        skills_dir=os.getenv("OPENCLAW_SKILLS_DIR", ""),
        daily_spend_limit=float(os.getenv("OPENCLAW_DAILY_SPEND_LIMIT", "10.0")),
    )

    # Exchange configs
    exchanges = {
        "binance": ExchangeConfig(
            name="binance",
            api_key=os.getenv("BINANCE_API_KEY", ""),
            api_secret=os.getenv("BINANCE_API_SECRET", ""),
            testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
        ),
        "coinbase": ExchangeConfig(
            name="coinbase",
            api_key=os.getenv("COINBASE_API_KEY", ""),
            api_secret=os.getenv("COINBASE_API_SECRET", ""),
            passphrase=os.getenv("COINBASE_PASSPHRASE", ""),
        ),
        "kraken": ExchangeConfig(
            name="kraken",
            api_key=os.getenv("KRAKEN_API_KEY", ""),
            api_secret=os.getenv("KRAKEN_API_SECRET", ""),
        ),
    }
    cfg.exchanges = exchanges

    return cfg


# Singleton config instance
_config: Optional[AACConfig] = None


def get_config() -> AACConfig:
    """Get or create the singleton configuration."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def is_paper_mode() -> bool:
    """Quick check: are we in paper trading mode?"""
    return get_config().paper_trading


def is_live_enabled() -> bool:
    """Quick check: is live trading explicitly enabled?"""
    cfg = get_config()
    return cfg.live_trading_enabled and not cfg.paper_trading
