"""
Accelerated Arbitrage Corp - Shared Utilities
==============================================
Common utilities and configuration management for all AAC components.
"""

from .config_loader import (
    PROJECT_ROOT,
    Config,
    get_config,
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_project_path,
    reload_config,
)
from .constants import VERSION

__all__ = [
    'Config',
    'get_config',
    'reload_config',
    'get_project_path',
    'get_env',
    'get_env_bool',
    'get_env_int',
    'get_env_float',
    'PROJECT_ROOT',
    'VERSION',
]
