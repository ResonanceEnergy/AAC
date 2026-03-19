"""
Shared Strategy Enums
======================
Canonical definitions for strategy-related enums used across
strategies/ and trading/ modules. Extracted to avoid circular imports.
"""

from enum import Enum


class StrategySignal(Enum):
    """Strategy signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class StrategyExecutionMode(Enum):
    """Strategy execution modes"""
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    SIMULATION = "simulation"
