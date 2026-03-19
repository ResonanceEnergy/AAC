"""
MATRIX MAXIMIZER — Geopolitical Bear Options Engine
=====================================================
TOP LEVEL / TOP PRIORITY system for AAC BANK pillar.

Deep integration with:
  - NCC (Natrix Command & Control) — governance gates, doctrine enforcement
  - NCL (BRAIN) — intelligence feeds, market sentiment, forecasts
  - AAC (BANK) — execution, accounting, risk management

Architecture (7 modules):
  1. core        — Data structures, enums, constants, scenario engine
  2. monte_carlo — 10,000-path correlated GBM with oil-shock integration
  3. greeks      — Full Black-Scholes (price, delta, gamma, vega, theta, rho)
  4. scanner     — Options chain scraper, auto-roll logic, delta-decay detection
  5. risk        — 7-layer risk management (VaR, CVaR, drawdown, Greeks limits)
  6. bridge      — NCC/NCL/AAC integration layer (governance, intelligence, execution)
  7. runner      — Orchestrator entry point, semi-daily cycle, CLI

Usage:
    from strategies.matrix_maximizer import MatrixMaximizer
    mm = MatrixMaximizer(account_size=50000)
    mm.run_full_cycle()  # synchronous wrapper
    # or: await mm.async_run_full_cycle()
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

__all__ = [
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
]
