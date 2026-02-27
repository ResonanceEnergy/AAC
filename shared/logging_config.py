"""
AAC Logging Configuration
=========================

Centralised logging setup for the Accelerated Arbitrage Corp platform.
Provides a single ``configure_logging()`` call that every entry-point
should invoke at startup so all modules get consistent formatting,
level control, and optional JSON output.

Usage::

    from shared.logging_config import configure_logging
    configure_logging()          # human-readable for dev
    configure_logging(json=True) # structured JSON for prod
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


# ── Defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_LEVEL = os.getenv("AAC_LOG_LEVEL", "INFO").upper()
_DEFAULT_FORMAT = (
    "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s"
)
_JSON_FORMAT = (
    '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
)
_LOG_DIR = Path(os.getenv("AAC_LOG_DIR", "logs"))


def configure_logging(
    *,
    level: str = _DEFAULT_LEVEL,
    json: bool = False,
    log_file: Optional[str] = None,
    max_bytes: int = 10_485_760,  # 10 MB
    backup_count: int = 5,
) -> None:
    """Configure root logger for the AAC platform.

    Parameters
    ----------
    level:
        Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    json:
        If ``True``, use one-line JSON format suitable for log aggregators.
    log_file:
        Optional filename inside ``AAC_LOG_DIR`` (default ``logs/``).
        A rotating file handler is added when provided.
    max_bytes:
        Maximum size of each log file before rotation (default 10 MB).
    backup_count:
        Number of rotated log files to keep (default 5).
    """
    fmt = _JSON_FORMAT if json else _DEFAULT_FORMAT
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))

    # Clear existing handlers to allow re-configuration
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Optional rotating file handler
    if log_file:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_DIR / log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quiet down noisy third-party loggers
    for noisy in ("urllib3", "asyncio", "websockets", "ccxt"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger scoped to *name*, typically ``__name__``."""
    return logging.getLogger(name)
