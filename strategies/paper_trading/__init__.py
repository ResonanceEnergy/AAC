from __future__ import annotations

"""Paper Trading Engine — simulated portfolio and order execution."""

from strategies.paper_trading.engine import PaperAccount, PaperOrder, PaperPosition, PaperTradingEngine
from strategies.paper_trading.metrics import MetricsTracker, PerformanceSnapshot
from strategies.paper_trading.regime_detector import MarketRegime, RegimeDetector
from strategies.paper_trading.risk_manager import RiskConfig, RiskManager
from strategies.paper_trading.strategies import (
    ArbitrageStrategy,
    DCAStrategy,
    GridStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    StrategyBase,
)
from strategies.paper_trading.optimizer import StrategyOptimizer, StrategyScore

__all__ = [
    "ArbitrageStrategy",
    "DCAStrategy",
    "GridStrategy",
    "MarketRegime",
    "MeanReversionStrategy",
    "MetricsTracker",
    "MomentumStrategy",
    "PaperAccount",
    "PaperOrder",
    "PaperPosition",
    "PaperTradingEngine",
    "PerformanceSnapshot",
    "RegimeDetector",
    "RiskConfig",
    "RiskManager",
    "StrategyBase",
    "StrategyOptimizer",
    "StrategyScore",
]
