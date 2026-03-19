"""
MATRIX MAXIMIZER — Geopolitical Bear Options Engine
=====================================================
TOP LEVEL / TOP PRIORITY system for AAC BANK pillar.

Deep integration with:
  - NCC (Natrix Command & Control) — governance gates, doctrine enforcement
  - NCL (BRAIN) — intelligence feeds, market sentiment, forecasts
  - AAC (BANK) — execution, accounting, risk management

Architecture (16 modules):
  Core:
    1.  core              — Data structures, enums, constants, scenario engine
    2.  monte_carlo       — 10,000-path correlated GBM with oil-shock integration
    3.  greeks            — Full Black-Scholes (price, delta, gamma, vega, theta, rho)
    4.  scanner           — Options chain scraper, auto-roll logic, delta-decay detection
    5.  risk              — 7-layer + enhanced risk (VaR, tail risk, correlation stress, margin)
    6.  bridge            — NCC/NCL/AAC integration layer
    7.  runner            — Supreme orchestrator, 13-step cycle, CLI

  Extended:
    8.  data_feeds        — Live prices (Polygon), FRED macro, Finnhub, UW flow
    9.  intelligence      — NCL signals, StockForecaster, RegimeEngine, earnings
    10. execution         — IBKR order execution, position tracker, Kelly sizing
    11. advanced_strategies — Spreads, collars, straddles, iron condors, butterflies
    12. alerts            — Telegram/Email/Log multi-channel alerts + watchdog
    13. scheduler         — 5-slot daily trading schedule, background daemon
    14. backtester        — Historical scenario simulation, performance attribution
    15. dashboard         — Daily/weekly reports, position heatmap, snapshot history
    16. chatbot           — Natural language command interface (13+ commands)

Usage:
    from strategies.matrix_maximizer import MatrixMaximizer
    mm = MatrixMaximizer()
    result = mm.run_full_cycle()
"""

from strategies.matrix_maximizer.core import (
    Asset,
    Scenario,
    ScenarioWeights,
    MatrixConfig,
    AssetForecast,
    PortfolioForecast,
    MandateLevel,
    SystemMandate,
)
from strategies.matrix_maximizer.monte_carlo import MonteCarloEngine
from strategies.matrix_maximizer.greeks import BlackScholesEngine, GreeksResult
from strategies.matrix_maximizer.scanner import OptionsScanner, PutRecommendation, RollSignal
from strategies.matrix_maximizer.risk import RiskManager, RiskSnapshot, CircuitBreaker
from strategies.matrix_maximizer.bridge import PillarBridge
from strategies.matrix_maximizer.runner import MatrixMaximizer

# Extended modules — imported safely for optional use
try:
    from strategies.matrix_maximizer.data_feeds import DataFeedManager
except ImportError:
    DataFeedManager = None  # type: ignore[misc,assignment]

try:
    from strategies.matrix_maximizer.intelligence import IntelligenceEngine
except ImportError:
    IntelligenceEngine = None  # type: ignore[misc,assignment]

try:
    from strategies.matrix_maximizer.execution import ExecutionEngine
except ImportError:
    ExecutionEngine = None  # type: ignore[misc,assignment]

try:
    from strategies.matrix_maximizer.advanced_strategies import AdvancedStrategyEngine
except ImportError:
    AdvancedStrategyEngine = None  # type: ignore[misc,assignment]

try:
    from strategies.matrix_maximizer.alerts import AlertManager, Watchdog
except ImportError:
    AlertManager = None  # type: ignore[misc,assignment]
    Watchdog = None  # type: ignore[misc,assignment]

try:
    from strategies.matrix_maximizer.scheduler import MatrixScheduler
except ImportError:
    MatrixScheduler = None  # type: ignore[misc,assignment]

try:
    from strategies.matrix_maximizer.backtester import MatrixBacktester
except ImportError:
    MatrixBacktester = None  # type: ignore[misc,assignment]

try:
    from strategies.matrix_maximizer.dashboard import MatrixDashboard
except ImportError:
    MatrixDashboard = None  # type: ignore[misc,assignment]

try:
    from strategies.matrix_maximizer.chatbot import MatrixChatbot
except ImportError:
    MatrixChatbot = None  # type: ignore[misc,assignment]

try:
    from strategies.matrix_maximizer.http_health import HTTPHealthCheck
except ImportError:
    HTTPHealthCheck = None  # type: ignore[misc,assignment]

__all__ = [
    # Core
    "MatrixMaximizer",
    "MonteCarloEngine",
    "BlackScholesEngine",
    "GreeksResult",
    "OptionsScanner",
    "PutRecommendation",
    "RollSignal",
    "RiskManager",
    "RiskSnapshot",
    "CircuitBreaker",
    "PillarBridge",
    "Asset",
    "Scenario",
    "ScenarioWeights",
    "MatrixConfig",
    "AssetForecast",
    "PortfolioForecast",
    "MandateLevel",
    "SystemMandate",
    # Extended
    "DataFeedManager",
    "IntelligenceEngine",
    "ExecutionEngine",
    "AdvancedStrategyEngine",
    "AlertManager",
    "Watchdog",
    "MatrixScheduler",
    "MatrixBacktester",
    "MatrixDashboard",
    "MatrixChatbot",
    "HTTPHealthCheck",
]
