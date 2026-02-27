"""
AAC Platform Constants
======================

Central source-of-truth for magic numbers, default values, and
system-wide constants used across the AAC trading platform.

Import usage::

    from shared.constants import DEFAULT_EXCHANGE, MAX_SLIPPAGE_BPS
"""

from __future__ import annotations

# ── Version ──────────────────────────────────────────────────────────────────

VERSION = "2.3.0"
APP_NAME = "AAC"
APP_FULL_NAME = "Accelerated Arbitrage Corp"

# ── Environment ──────────────────────────────────────────────────────────────

ENV_PRODUCTION = "production"
ENV_STAGING = "staging"
ENV_TEST = "test"
ENV_DEV = "development"

# ── Trading Defaults ────────────────────────────────────────────────────────

DEFAULT_EXCHANGE = "binance"
DEFAULT_BASE_CURRENCY = "USDT"
DEFAULT_QUOTE_CURRENCY = "USD"

# Position sizing
MAX_POSITION_SIZE_USD = 100_000.0
MIN_POSITION_SIZE_USD = 10.0
DEFAULT_POSITION_PCT = 0.02  # 2% of portfolio per trade

# Slippage
MAX_SLIPPAGE_BPS = 10.0        # 10 basis points (0.10%)
DEFAULT_SLIPPAGE_BPS = 1.5     # 1.5 basis points (0.015%)

# Risk management
MAX_DRAWDOWN_PCT = 0.05        # 5% max drawdown before circuit breaker
MAX_DAILY_LOSS_PCT = 0.02      # 2% max daily loss
MAX_OPEN_POSITIONS = 50
DEFAULT_STOP_LOSS_PCT = 0.02   # 2% default stop loss
DEFAULT_TAKE_PROFIT_PCT = 0.05 # 5% default take profit

# ── Market Data ──────────────────────────────────────────────────────────────

STALE_DATA_THRESHOLD_SEC = 60  # seconds before data is considered stale
ORDERBOOK_DEPTH_DEFAULT = 20
OHLCV_TIMEFRAME_DEFAULT = "1h"

# ── Timeouts & Retries ──────────────────────────────────────────────────────

HTTP_TIMEOUT_SEC = 30
WS_PING_INTERVAL_SEC = 20
WS_RECONNECT_DELAY_SEC = 5
MAX_API_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2.0

# ── Database ─────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "data/aac.db"
DB_WAL_MODE = True

# ── Paper Trading ────────────────────────────────────────────────────────────

PAPER_INITIAL_BALANCE = 100_000.0
PAPER_COMMISSION_PCT = 0.001  # 0.1% per trade
