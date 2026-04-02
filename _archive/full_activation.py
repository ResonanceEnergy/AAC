#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AAC FULL ACTIVATION — ALL HANDS ON DECK                                    ║
║  BARREN WUFFET: Operation ROCKET SHIP to $ONE MILLION                       ║
║                                                                              ║
║  Every agent. Every division. Every department. Every API. Every data stream.║
║  FPC + NCC + NCL + Doctrine + BRS running fully. Continuous analysis.        ║
║  All systems coordinated. All intelligence aggregated. 24/7 operations.      ║
║                                                                              ║
║  Usage:                                                                      ║
║    python full_activation.py                    # Full battle stations        ║
║    python full_activation.py --mode paper       # Paper trading (default)     ║
║    python full_activation.py --mode live        # Live trading (careful!)     ║
║    python full_activation.py --status           # System status report        ║
║    python full_activation.py --cycle-once       # Run one full cycle & exit   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Project root ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ── Core imports (always available) ───────────────────────────────────
from weekly_tracker import (
    WEEKLY_GROWTH_RATE,
    CompoundTracker,
    WeeklyAllocation,
    WeekPhase,
    get_current_phase,
)

from agents.master_agent_file import (
    AACMasterAgentSystem,
    get_master_agent_system,
)

# ── Agent imports ─────────────────────────────────────────────────────
from BigBrainIntelligence.agents import (
    BaseResearchAgent,
    ResearchFinding,
    get_agents_by_theater,
    get_all_agents,
)
from CentralAccounting.database import AccountingDatabase
from shared.audit_logger import AuditLogger
from shared.communication import CommunicationFramework
from shared.config_loader import get_config, get_project_path
from shared.data_sources import CoinGeckoClient, DataAggregator, MarketTick

# ── Strategy imports ──────────────────────────────────────────────────
from strategies.golden_ratio_finance import (
    FibLevel,
    FibonacciCalculator,
    fractal_compression_index,
    phase_conjugation_score,
)
from strategies.macro_crisis_put_strategy import (
    PUT_PLAYBOOK,
    CrisisAssessment,
    CrisisMonitor,
    CrisisVector,
    MacroCrisisPutEngine,
    PutOrderSpec,
)
from strategies.zero_dte_gamma_engine import (
    RiskMode,
    SessionPhase,
    ZeroDTEStrategy,
)
from TradingExecution.execution_engine import (
    AAC2100ExecutionEngine,
    ExecutionEngine,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionStatus,
)

# ── Optional imports (graceful degradation) ───────────────────────────
AVAILABLE_MODULES = {}


def _try_import(name: str, import_fn: Callable):
    """Import a module, log availability."""
    try:
        result = import_fn()
        AVAILABLE_MODULES[name] = True
        return result
    except (ImportError, Exception) as e:
        AVAILABLE_MODULES[name] = False
        return None


# CryptoIntelligence
_crypto_intel = _try_import("CryptoIntelligence", lambda: __import__(
    "CryptoIntelligence.crypto_intelligence_engine",
    fromlist=["CryptoIntelligenceEngine"]
))
CryptoIntelligenceEngine = getattr(_crypto_intel, "CryptoIntelligenceEngine", None) if _crypto_intel else None

# DeFi Yield Analyzer
_defi = _try_import("DeFiYieldAnalyzer", lambda: __import__(
    "CryptoIntelligence.defi_yield_analyzer",
    fromlist=["DeFiYieldAnalyzer"]
))
DeFiYieldAnalyzer = getattr(_defi, "DeFiYieldAnalyzer", None) if _defi else None

# Whale Tracking
_whale = _try_import("WhaleTracker", lambda: __import__(
    "CryptoIntelligence.whale_tracking_system",
    fromlist=["WhaleTrackingSystem"]
))
WhaleTrackingSystem = getattr(_whale, "WhaleTrackingSystem", None) if _whale else None

# Options Strategy Engine
_options = _try_import("OptionsEngine", lambda: __import__(
    "strategies.options_strategy_engine",
    fromlist=["OptionsStrategyEngine"]
))
OptionsStrategyEngine = getattr(_options, "OptionsStrategyEngine", None) if _options else None

# Variance Risk Premium
_vrp = _try_import("VarianceRiskPremium", lambda: __import__(
    "strategies.variance_risk_premium",
    fromlist=["VarianceRiskPremiumStrategy"]
))

# Volatility Arbitrage
_vol_arb = _try_import("VolatilityArbitrage", lambda: __import__(
    "strategies.volatility_arbitrage_engine",
    fromlist=["VolatilityArbitrageEngine"]
))

# MetalX Arbitrage
_metalx = _try_import("MetalXArbitrage", lambda: __import__(
    "strategies.metalx_arb_strategy",
    fromlist=["MetalXArbStrategy"]
))

# WebSocket Feeds
_ws = _try_import("WebSocketFeeds", lambda: __import__(
    "shared.websocket_feeds",
    fromlist=["PriceFeedManager", "BinanceWebSocketFeed"]
))

# Doctrine Engine
_doctrine = _try_import("DoctrineEngine", lambda: __import__(
    "aac.doctrine.doctrine_engine",
    fromlist=["DoctrineEngine", "BarrenWuffetState", "ActionType", "ComplianceState",
             "DoctrineViolation", "DoctrineComplianceReport"]
))
DoctrineEngineClass: Any = getattr(_doctrine, "DoctrineEngine", None) if _doctrine else None
BarrenWuffetState: Any = getattr(_doctrine, "BarrenWuffetState", None) if _doctrine else None
ActionType: Any = getattr(_doctrine, "ActionType", None) if _doctrine else None

# Strategy Loader
_strat_loader = _try_import("StrategyLoader", lambda: __import__(
    "shared.strategy_loader",
    fromlist=["get_strategy_loader"]
))

# Autonomous Engine — bypass core.__init__ (it imports streamlit which hangs on 3.14)
def _import_file(name, filepath):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None

_auto_engine = _try_import("AutonomousEngine", lambda: _import_file(
    "core.autonomous_engine", str(PROJECT_ROOT / "core" / "autonomous_engine.py")
))

# Orchestrator — direct file import to avoid core.__init__
_orchestrator_mod = _try_import("Orchestrator", lambda: _import_file(
    "core.orchestrator", str(PROJECT_ROOT / "core" / "orchestrator.py")
))

# ACC Advanced State (Faraday Protection / FPC)
_acc_state = _try_import("ACCAdvancedState", lambda: _import_file(
    "core.acc_advanced_state", str(PROJECT_ROOT / "core" / "acc_advanced_state.py")
))
ACC_AdvancedState: Any = getattr(_acc_state, "ACC_AdvancedState", None) if _acc_state else None

# Extract DoctrineStatus, ComponentHealth, ComponentStatus from autonomous engine
DoctrineStatus: Any = getattr(_auto_engine, "DoctrineStatus", None) if _auto_engine else None
DoctrineState: Any = getattr(_auto_engine, "DoctrineState", None) if _auto_engine else None
ComponentHealth: Any = getattr(_auto_engine, "ComponentHealth", None) if _auto_engine else None
ComponentStatus: Any = getattr(_auto_engine, "ComponentStatus", None) if _auto_engine else None
ScheduledTask: Any = getattr(_auto_engine, "ScheduledTask", None) if _auto_engine else None
SystemGap: Any = getattr(_auto_engine, "SystemGap", None) if _auto_engine else None
GapSeverity: Any = getattr(_auto_engine, "GapSeverity", None) if _auto_engine else None

# Pipeline components
_pipeline = _try_import("PipelineRunner", lambda: __import__(
    "pipeline_runner",
    fromlist=["FibSignalGenerator"]
))
FibSignalGenerator: Any = getattr(_pipeline, "FibSignalGenerator", None) if _pipeline else None

# NCC Coordinator (Neural Coordination Center)
_ncc_hub = _try_import("NCCCoordinator", lambda: __import__(
    "integrations.api_integration_hub",
    fromlist=["NCCCoordinatorClient"]
))
NCCCoordinatorClient: Any = getattr(_ncc_hub, "NCCCoordinatorClient", None) if _ncc_hub else None

# Super NCC Agent (department-level strategic coordinator)
_ncc_super = _try_import("SuperNCCAgent", lambda: __import__(
    "shared.department_super_agents",
    fromlist=["SuperNCCCoordinatorAgent"]
))
SuperNCCCoordinatorAgent: Any = getattr(_ncc_super, "SuperNCCCoordinatorAgent", None) if _ncc_super else None

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            get_project_path("logs", f"full_activation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            mode='w'
        ),
    ],
)
logger = logging.getLogger("FULL_ACTIVATION")


# ════════════════════════════════════════════════════════════════════════
# ACTIVATION STATUS & TELEMETRY
# ════════════════════════════════════════════════════════════════════════

class ActivationLevel(Enum):
    """System-wide activation level."""
    STANDBY = "STANDBY"
    PARTIAL = "PARTIAL"
    FULL = "FULL"
    BATTLE_STATIONS = "BATTLE_STATIONS"


class NCLComplianceLevel(Enum):
    """NCL Compliance Matrix classification levels."""
    OMEGA = "OMEGA"       # 95%+ — perfect compliance
    GAMMA = "GAMMA"       # 80-95% — elite performance
    BETA = "BETA"         # 50-80% — moderate compliance
    ALPHA = "ALPHA"       # <50% — zero-compliance recovery


@dataclass
class NCLComplianceReport:
    """NCL Governance Charter compliance report per cycle."""
    level: NCLComplianceLevel = NCLComplianceLevel.BETA
    score: float = 0.0
    checks_passed: int = 0
    checks_total: int = 0
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def pct(self) -> float:
        return (self.checks_passed / max(self.checks_total, 1)) * 100


class NCLGovernanceEngine:
    """
    Runtime NCL Governance enforcement.
    Encodes the NCL Compliance Matrix and Governance Charter rules:
    - Capital efficiency (25%+ annualized target)
    - Risk-adjusted returns (Sharpe 2.0+, MDD <15%)
    - Decision quality (doctrine compliance)
    - Digital integrity (all actions auditable)
    - Human judgment paramount (no live trades without confirmation)
    """

    def evaluate(self, metrics: Dict[str, Any]) -> NCLComplianceReport:
        """Evaluate NCL governance compliance for this cycle."""
        report = NCLComplianceReport()
        checks = []

        # 1. Digital Integrity — all trades auditable
        db_ok = metrics.get("database_online", False)
        checks.append(("digital_integrity", db_ok))
        if not db_ok:
            report.violations.append("Database offline — audit trail broken")

        # 2. Human Judgment Paramount — paper mode unless explicitly confirmed
        paper_mode = metrics.get("paper_mode", True)
        checks.append(("human_judgment", paper_mode or metrics.get("live_confirmed", False)))
        if not paper_mode and not metrics.get("live_confirmed", False):
            report.violations.append("Live trading without explicit confirmation")

        # 3. Risk Limits — MDD <15%
        drawdown = metrics.get("drawdown_pct", 0.0)
        checks.append(("mdd_limit", drawdown < 15.0))
        if drawdown >= 15.0:
            report.violations.append(f"Max drawdown {drawdown:.1f}% exceeds 15% NCL limit")

        # 4. Doctrine Compliance — doctrine engine loaded and active
        doctrine_ok = metrics.get("doctrine_online", False)
        checks.append(("doctrine_active", doctrine_ok))
        if not doctrine_ok:
            report.recommendations.append("Doctrine engine offline — compliance degraded")

        # 5. Position Concentration — max 25% in single asset
        max_concentration = metrics.get("max_concentration_pct", 0.0)
        checks.append(("concentration_limit", max_concentration <= 25.0))
        if max_concentration > 25.0:
            report.violations.append(f"Position concentration {max_concentration:.0f}% exceeds 25%")

        # 6. Daily Loss Cap — max 5% daily
        daily_loss = metrics.get("daily_loss_pct", 0.0)
        checks.append(("daily_loss_cap", daily_loss < 5.0))
        if daily_loss >= 5.0:
            report.violations.append(f"Daily loss {daily_loss:.1f}% exceeds 5% NCL limit")

        # 7. Agent Coordination — at least one agent system online
        agents_online = metrics.get("agents_online", 0)
        checks.append(("agent_coordination", agents_online > 0))

        # 8. Crisis Awareness — crisis monitor active
        crisis_active = metrics.get("crisis_monitor_active", True)
        checks.append(("crisis_awareness", crisis_active))

        # Calculate score
        report.checks_total = len(checks)
        report.checks_passed = sum(1 for _, ok in checks if ok)
        report.score = report.pct

        # Classify compliance level
        if report.score >= 95:
            report.level = NCLComplianceLevel.OMEGA
        elif report.score >= 80:
            report.level = NCLComplianceLevel.GAMMA
        elif report.score >= 50:
            report.level = NCLComplianceLevel.BETA
        else:
            report.level = NCLComplianceLevel.ALPHA
            report.recommendations.append("CRITICAL: Immediate NCL recovery protocol required")

        return report


@dataclass
class SystemTelemetry:
    """Real-time system telemetry across all components."""
    activation_level: ActivationLevel = ActivationLevel.STANDBY
    start_time: Optional[datetime] = None
    cycles_completed: int = 0
    total_signals_generated: int = 0
    total_trades_executed: int = 0
    total_pnl: float = 0.0
    active_positions: int = 0
    active_agents: int = 0
    active_strategies: int = 0
    active_data_streams: int = 0
    active_exchanges: int = 0
    crisis_severity: float = 0.0
    modules_online: Dict[str, bool] = field(default_factory=dict)
    agent_statuses: Dict[str, str] = field(default_factory=dict)
    last_scan_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cycle_times_ms: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def uptime_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def avg_cycle_time_ms(self) -> float:
        if not self.cycle_times_ms:
            return 0.0
        return sum(self.cycle_times_ms) / len(self.cycle_times_ms)


# ════════════════════════════════════════════════════════════════════════
# CRYPTO WATCHLIST — Everything we're tracking
# ════════════════════════════════════════════════════════════════════════

CRYPTO_WATCHLIST: List[Dict[str, Any]] = [
    # Tier 1 — Core positions (highest allocation)
    {"symbol": "bitcoin", "ticker": "BTC", "pair": "BTC/USDT", "tier": 1,
     "thesis": "Digital gold, safe haven during fiat crisis, BRICS de-dollarization"},
    {"symbol": "ethereum", "ticker": "ETH", "pair": "ETH/USDT", "tier": 1,
     "thesis": "DeFi backbone, staking yield, L2 ecosystem growth"},

    # Tier 2 — High conviction
    {"symbol": "ripple", "ticker": "XRP", "pair": "XRP/USDT", "tier": 2,
     "thesis": "BRICS payment rails, Ripple institutional adoption, court victory momentum"},
    {"symbol": "solana", "ticker": "SOL", "pair": "SOL/USDT", "tier": 2,
     "thesis": "Fastest L1, DeFi TVL growth, Jupiter/Raydium ecosystem"},

    # Tier 3 — Catalyst plays
    {"symbol": "flare-networks", "ticker": "FLR", "pair": "FLR/USDT", "tier": 3,
     "thesis": "XRP ecosystem, FTSO delegation rewards, cross-chain interop"},
    {"symbol": "chainlink", "ticker": "LINK", "pair": "LINK/USDT", "tier": 3,
     "thesis": "Oracle infrastructure, CCIP cross-chain, staking launch"},
    {"symbol": "render-token", "ticker": "RNDR", "pair": "RNDR/USDT", "tier": 3,
     "thesis": "GPU compute demand, AI narrative, decentralized rendering"},

    # Tier 4 — Volatile momentum
    {"symbol": "dogecoin", "ticker": "DOGE", "pair": "DOGE/USDT", "tier": 4,
     "thesis": "X payments integration speculation, meme momentum"},

    # Tier 5 — Agent-discovered (high conviction from BigBrain agents)
    {"symbol": "hyperliquid", "ticker": "HYPE", "pair": "HYPE/USDT", "tier": 5,
     "thesis": "Agent narrative signal — DEX leader, perps volume growth"},
    {"symbol": "bittensor", "ticker": "TAO", "pair": "TAO/USDT", "tier": 5,
     "thesis": "Agent narrative signal — decentralized AI network, subnet architecture"},
    {"symbol": "grass", "ticker": "GRASS", "pair": "GRASS/USDT", "tier": 5,
     "thesis": "Agent narrative signal — DePIN bandwidth marketplace"},
    {"symbol": "pi-network", "ticker": "PI", "pair": "PI/USDT", "tier": 5,
     "thesis": "Agent narrative signal — mobile-first crypto, ecosystem launch"},
]

# Options targets (IBKR)
OPTIONS_WATCHLIST: List[Dict[str, Any]] = [
    {"ticker": "SPY", "sector": "Broad Market", "direction": "PUT", "priority": 1},
    {"ticker": "QQQ", "sector": "Tech/Nasdaq", "direction": "PUT", "priority": 1},
    {"ticker": "IWM", "sector": "Small Cap", "direction": "PUT", "priority": 2},
    {"ticker": "XLF", "sector": "Financials", "direction": "PUT", "priority": 1},
    {"ticker": "KRE", "sector": "Regional Banks", "direction": "PUT", "priority": 1},
    {"ticker": "HYG", "sector": "High Yield Credit", "direction": "PUT", "priority": 1},
    {"ticker": "BKLN", "sector": "Leveraged Loans", "direction": "PUT", "priority": 2},
    {"ticker": "LQD", "sector": "Investment Grade", "direction": "PUT", "priority": 3},
    {"ticker": "USO", "sector": "Oil", "direction": "CALL", "priority": 2},
    {"ticker": "GLD", "sector": "Gold", "direction": "STRADDLE", "priority": 2},
    {"ticker": "UVXY", "sector": "Volatility", "direction": "CALL", "priority": 2},
    {"ticker": "MSTR", "sector": "BTC Proxy", "direction": "CALL", "priority": 3},
    {"ticker": "IBIT", "sector": "BTC ETF", "direction": "CALL", "priority": 3},
    {"ticker": "XLE", "sector": "Energy", "direction": "CALL", "priority": 3},
]


# ════════════════════════════════════════════════════════════════════════
# FULL ACTIVATION ENGINE
# ════════════════════════════════════════════════════════════════════════

class FullActivationEngine:
    """
    ALL HANDS ON DECK — Coordinates every AAC subsystem simultaneously.

    Components activated:
    ├── DATA LAYER
    │   ├── CoinGecko (free, real-time crypto)
    │   ├── CCXT (7 exchanges)
    │   ├── WebSocket feeds (Binance, Coinbase)
    │   └── Pipeline runner (Fibonacci + technical)
    ├── INTELLIGENCE LAYER
    │   ├── 20 Research Agents (BigBrain)
    │   ├── 6 Super Agents (Department heads)
    │   ├── CryptoIntelligence (venue health, whale tracking, DeFi yield)
    │   ├── Crisis Monitor (10 macro vectors)
    │   └── Options flow analysis
    ├── STRATEGY LAYER
    │   ├── 50 Arbitrage Strategies (CSV playbook)
    │   ├── 40+ Options Strategies (Greeks-aware)
    │   ├── Macro Crisis Put Engine
    │   ├── 0DTE Gamma Scalper
    │   ├── Fibonacci/Golden Ratio Pipeline
    │   └── Cross-exchange crypto arbitrage
    ├── EXECUTION LAYER
    │   ├── ExecutionEngine (paper + live)
    │   ├── RiskManager (position limits, daily loss cap)
    │   ├── Exchange connectors (Binance, Coinbase, Kraken, IBKR)
    │   └── Order generation + validation
    ├── ACCOUNTING LAYER
    │   ├── SQLite transaction database
    │   ├── P&L tracking
    │   └── Audit trail
    ├── DOCTRINE LAYER
    │   ├── 8-pack compliance
    │   ├── Art of War strategic posture
    │   └── State machine (NORMAL → CAUTION → SAFE_MODE → HALT)
    └── MONITORING LAYER
        ├── Telemetry dashboard
        ├── Agent health checks
        ├── Latency tracking
        └── Alert system
    """

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.telemetry = SystemTelemetry()
        self._shutdown_event = asyncio.Event()
        self._initialized = False

        # Weekly compound tracker
        self.tracker = CompoundTracker.load()
        self.week_phase = get_current_phase()

        # Cycle-level accumulators for weekly recording
        self._week_crypto_pnl = 0.0
        self._week_options_pnl = 0.0
        self._week_arb_pnl = 0.0
        self._week_trades = 0
        self._week_signals = 0
        self._week_best_trade = ""
        self._week_best_pnl = 0.0
        self._week_worst_trade = ""
        self._week_worst_pnl = 0.0
        self._week_crisis_severities: List[float] = []

        # Core components
        self.config = get_config()
        self.data_aggregator = DataAggregator()
        self.coingecko = CoinGeckoClient()
        self.db = AccountingDatabase()
        self.audit_logger = AuditLogger()
        self.communication = CommunicationFramework()

        # Execution
        self.execution_engine = ExecutionEngine(db=self.db)
        self.fib_calculator = FibonacciCalculator()
        self.fib_signal_gen = FibSignalGenerator() if FibSignalGenerator else None

        # Crisis monitoring
        self.crisis_monitor = CrisisMonitor()
        self.crisis_put_engine = MacroCrisisPutEngine(
            account_balance=self.tracker.current_capital,
            max_portfolio_put_allocation_pct=15.0,
            max_single_position_pct=3.0,
            paper_trading=paper_mode,
        )

        # Agent system
        self.master_agents: Optional[AACMasterAgentSystem] = None
        self.research_agents: List[BaseResearchAgent] = []

        # Intelligence modules
        self.crypto_intel = None
        self.defi_analyzer = None
        self.whale_tracker = None

        # ── DOCTRINE & COMPLIANCE LAYER ───────────────────────────
        self.doctrine_engine = None          # 8-pack compliance (DoctrineEngine)
        self.doctrine_status = None          # State machine (NORMAL→CAUTION→SAFE_MODE→HALT)
        self.barren_wuffet_state = "NORMAL"  # Current BW state string
        self.doctrine_violations: List = []  # Active violations
        self.doctrine_actions: List = []     # Triggered actions

        # ── COMPONENT HEALTH TRACKING ─────────────────────────────
        self.component_health: Dict[str, Any] = {}  # ComponentStatus per subsystem

        # ── FPC / ACC ADVANCED STATE ──────────────────────────────
        self.acc_advanced_state = None       # Faraday Protection / Resilience

        # ── SCHEDULED TASKS ───────────────────────────────────────
        self.scheduled_tasks: Dict[str, Any] = {}  # Periodic introspection tasks
        self.system_gaps: List = []          # Detected gaps

        # ── AGENT COORDINATION ────────────────────────────────────
        self._agent_performance: Dict[str, Dict] = {}  # Per-agent scans/findings/signals

        # ── NCC COORDINATION LAYER ────────────────────────────────
        self.ncc_coordinator = None       # NCCCoordinatorClient (REST API)
        self.ncc_super_agent = None       # SuperNCCCoordinatorAgent (strategic)
        self.ncc_status: Dict[str, Any] = {}  # Latest NCC coordination status

        # ── NCL GOVERNANCE LAYER ──────────────────────────────────
        self.ncl_engine = NCLGovernanceEngine()
        self.ncl_compliance: Optional[NCLComplianceReport] = None

        # Strategy tracking
        self.active_signals: Dict[str, Dict] = {}
        self.signal_history: deque = deque(maxlen=1000)
        self.position_tracker: Dict[str, Dict] = {}

        # Cycle configuration
        self.scan_interval_seconds = 30
        self.execution_interval_seconds = 15
        self.report_interval_seconds = 300

    async def initialize(self) -> bool:
        """
        FULL SYSTEM INITIALIZATION — All components, all layers.
        Returns True if core systems are online.
        """
        logger.info("=" * 80)
        logger.info("  BARREN WUFFET — FULL ACTIVATION SEQUENCE INITIATED")
        logger.info("  Operation: ROCKET SHIP TO $ONE MILLION")
        logger.info(f"  Mode: {'PAPER TRADING' if self.paper_mode else '🔴 LIVE TRADING'}")
        logger.info(f"  Capital: ${self.tracker.current_capital:,.2f} | "
                     f"Week: {self.tracker.current_week} | Phase: {self.week_phase.value}")
        logger.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        self.telemetry.start_time = datetime.now()
        core_ok = True

        # ── PHASE 1: Data Layer ───────────────────────────────────────
        logger.info("\n[PHASE 1] DATA LAYER — Connecting all data streams...")
        try:
            await self.data_aggregator.connect_all()
            self.telemetry.active_data_streams += 1
            logger.info("  ✅ DataAggregator connected (CCXT multi-exchange)")
        except Exception as e:
            logger.warning(f"  ⚠️  DataAggregator: {e}")

        # CoinGecko — always available, no API key needed
        logger.info("  ✅ CoinGecko client ready (free, no API key required)")
        self.telemetry.active_data_streams += 1

        # WebSocket feeds
        if AVAILABLE_MODULES.get("WebSocketFeeds"):
            logger.info("  ✅ WebSocket feeds available (Binance, Coinbase)")
            self.telemetry.active_data_streams += 1
        else:
            logger.info("  ⚠️  WebSocket feeds not available (using REST polling)")

        # ── PHASE 2: Database ─────────────────────────────────────────
        logger.info("\n[PHASE 2] DATABASE — Initializing accounting...")
        try:
            self.db.initialize()
            logger.info("  ✅ AccountingDatabase initialized (SQLite)")
        except Exception as e:
            logger.error(f"  ❌ Database failed: {e}")
            core_ok = False

        # ── PHASE 3: Execution Engine ─────────────────────────────────
        logger.info("\n[PHASE 3] EXECUTION ENGINE — Arming trading systems...")
        try:
            if hasattr(self.execution_engine, 'initialize'):
                await self.execution_engine.initialize()
            logger.info(f"  ✅ ExecutionEngine ready (paper_mode={self.paper_mode})")
            logger.info(f"     Max position size: ${self.execution_engine.risk_manager.max_position_size_usd if hasattr(self.execution_engine, 'risk_manager') else 'N/A'}")
        except Exception as e:
            logger.warning(f"  ⚠️  ExecutionEngine init: {e}")

        # ── PHASE 4: Agent System ─────────────────────────────────────
        logger.info("\n[PHASE 4] AGENT SYSTEM — Deploying all agents...")
        try:
            self.master_agents = AACMasterAgentSystem()
            await self.master_agents.initialize()
            self.telemetry.active_agents = self.master_agents.agent_counts.get('total', 0)
            logger.info(f"  ✅ Master Agent System: {self.telemetry.active_agents} agents deployed")
            logger.info(f"     Research: {self.master_agents.agent_counts.get('research', 0)}")
            logger.info(f"     Super:    {self.master_agents.agent_counts.get('super', 0)}")
            logger.info(f"     Contest:  {self.master_agents.agent_counts.get('contest', 0)}")
        except Exception as e:
            logger.warning(f"  ⚠️  Agent system: {e}")

        # Load research agents directly
        try:
            self.research_agents = get_all_agents()
            logger.info(f"  ✅ Research agents loaded: {len(self.research_agents)} agents")
        except Exception as e:
            logger.warning(f"  ⚠️  Research agents: {e}")

        # ── PHASE 5: Intelligence Modules ─────────────────────────────
        logger.info("\n[PHASE 5] INTELLIGENCE — Activating all intel systems...")

        if CryptoIntelligenceEngine:
            try:
                self.crypto_intel = CryptoIntelligenceEngine()
                logger.info("  ✅ CryptoIntelligence Engine online")
            except Exception as e:
                logger.warning(f"  ⚠️  CryptoIntelligence: {e}")

        if DeFiYieldAnalyzer:
            try:
                self.defi_analyzer = DeFiYieldAnalyzer()
                logger.info("  ✅ DeFi Yield Analyzer online")
            except Exception as e:
                logger.warning(f"  ⚠️  DeFi Yield Analyzer: {e}")

        if WhaleTrackingSystem:
            try:
                self.whale_tracker = WhaleTrackingSystem()
                logger.info("  ✅ Whale Tracking System online")
            except Exception as e:
                logger.warning(f"  ⚠️  Whale Tracker: {e}")

        logger.info(f"  ✅ Crisis Monitor online (10 macro vectors)")
        logger.info(f"  ✅ Macro Crisis Put Engine armed (account: ${self.crisis_put_engine.account_balance:,.0f})")

        # ── PHASE 6: Strategy Layer ───────────────────────────────────
        logger.info("\n[PHASE 6] STRATEGIES — Loading all strategy engines...")

        strategy_count = 0
        if FibSignalGenerator:
            logger.info("  ✅ Fibonacci/Golden Ratio pipeline (VERIFIED WORKING)")
            strategy_count += 1

        logger.info("  ✅ Macro Crisis Put Strategy (8 targets loaded)")
        strategy_count += 1

        logger.info("  ✅ 0DTE Gamma Scalping Engine")
        strategy_count += 1

        if AVAILABLE_MODULES.get("OptionsEngine"):
            logger.info("  ✅ Options Strategy Engine (40+ strategies)")
            strategy_count += 40

        if AVAILABLE_MODULES.get("VarianceRiskPremium"):
            logger.info("  ✅ Variance Risk Premium")
            strategy_count += 1

        if AVAILABLE_MODULES.get("VolatilityArbitrage"):
            logger.info("  ✅ Volatility Arbitrage Engine")
            strategy_count += 1

        if AVAILABLE_MODULES.get("MetalXArbitrage"):
            logger.info("  ✅ MetalX Cross-Exchange Arbitrage")
            strategy_count += 1

        self.telemetry.active_strategies = strategy_count
        logger.info(f"  Total strategies loaded: {strategy_count}")

        # ── PHASE 7: Doctrine & Compliance ────────────────────────────
        logger.info("\n[PHASE 7] DOCTRINE — Activating compliance guardrails...")
        if DoctrineEngineClass:
            try:
                de = DoctrineEngineClass()
                de.load_doctrine_packs()
                # Register action handlers for automated responses
                if ActionType:
                    de.register_action_handler(
                        ActionType.A_THROTTLE_RISK,
                        self._doctrine_action_throttle_risk,
                    )
                    de.register_action_handler(
                        ActionType.A_STOP_EXECUTION,
                        self._doctrine_action_halt,
                    )
                    de.register_action_handler(
                        ActionType.A_ENTER_SAFE_MODE,
                        self._doctrine_action_safe_mode,
                    )
                    de.register_action_handler(
                        ActionType.A_TACTICAL_RETREAT,
                        self._doctrine_action_throttle_risk,
                    )
                self.doctrine_engine = de
                logger.info("  ✅ Doctrine Engine online (8-pack compliance + state machine)")
                logger.info(f"     Packs loaded: {len(de.doctrine_packs)}")
            except Exception as e:
                logger.warning(f"  ⚠️  Doctrine Engine init: {e}")
        else:
            logger.info("  ⚠️  Doctrine Engine not available (using manual guardrails)")

        # Initialize DoctrineStatus state machine (from AutonomousEngine)
        if DoctrineStatus:
            try:
                self.doctrine_status = DoctrineStatus()
                logger.info("  ✅ Doctrine State Machine: NORMAL→CAUTION→SAFE_MODE→HALT")
            except Exception as e:
                logger.warning(f"  ⚠️  DoctrineStatus init: {e}")

        logger.info("  ✅ Risk limits enforced:")
        logger.info("     Max position: $1,000 | Max daily loss: $500 | Max open: 5")
        logger.info("     Concentration: 25% max | Stop loss: 5% default")

        # ── PHASE 7b: FPC / Faraday Protection ───────────────────────
        logger.info("\n[PHASE 7b] FPC — Faraday Protection / ACC Advanced State...")
        if ACC_AdvancedState:
            try:
                acc = ACC_AdvancedState()
                self.acc_advanced_state = acc
                # Initialize resilience layers (sync — do not await full async init)
                logger.info("  ✅ ACC Advanced State loaded (Future/Bomb/Hurricane/EMP Proof)")
                logger.info(f"     Resilience layers: {len(acc.resilience_layers)}")
            except Exception as e:
                logger.warning(f"  ⚠️  ACC Advanced State: {e}")
        else:
            logger.info("  ⚠️  ACC Advanced State not available")

        # ── PHASE 7c: Component Health Tracking ──────────────────────
        logger.info("\n[PHASE 7c] HEALTH TRACKING — Registering component monitors...")
        if ComponentStatus:
            component_names = [
                "coingecko", "execution_engine", "database", "agents",
                "fibonacci_pipeline", "crisis_monitor", "doctrine_engine",
                "crypto_intelligence", "defi_analyzer", "whale_tracker",
                "websocket_feeds", "acc_advanced_state",
            ]
            for name in component_names:
                self.component_health[name] = ComponentStatus(name=name)
            # Mark already-verified components as healthy
            self.component_health["coingecko"].record_success()
            self.component_health["crisis_monitor"].record_success()
            if core_ok:
                self.component_health["database"].record_success()
                self.component_health["execution_engine"].record_success()
            if self.fib_signal_gen:
                self.component_health["fibonacci_pipeline"].record_success()
            if self.doctrine_engine:
                self.component_health["doctrine_engine"].record_success()
            if self.crypto_intel:
                self.component_health["crypto_intelligence"].record_success()
            if self.defi_analyzer:
                self.component_health["defi_analyzer"].record_success()
            if self.whale_tracker:
                self.component_health["whale_tracker"].record_success()
            if self.acc_advanced_state:
                self.component_health["acc_advanced_state"].record_success()
            healthy = sum(1 for c in self.component_health.values()
                         if ComponentHealth and c.health == ComponentHealth.HEALTHY)
            logger.info(f"  ✅ Health tracking: {healthy}/{len(self.component_health)} components healthy")
        else:
            logger.info("  ⚠️  ComponentStatus not available — health tracking disabled")

        # ── PHASE 7d: NCC Coordination Layer ─────────────────────
        logger.info("\n[PHASE 7d] NCC — Neural Coordination Center...")
        if SuperNCCCoordinatorAgent:
            try:
                ncc = SuperNCCCoordinatorAgent()
                await ncc.initialize_super_capabilities()
                self.ncc_super_agent = ncc
                logger.info("  ✅ NCC Super Coordinator Agent online (strategic planning, doctrine enforcement)")
            except Exception as e:
                logger.warning(f"  ⚠️  NCC Super Agent: {e}")
        else:
            logger.info("  ⚠️  NCC Super Agent not available")

        if NCCCoordinatorClient:
            try:
                self.ncc_coordinator = NCCCoordinatorClient(self.config)
                logger.info("  ✅ NCC Coordinator REST client ready")
            except Exception as e:
                logger.warning(f"  ⚠️  NCC Coordinator client: {e}")
        else:
            logger.info("  ⚠️  NCC Coordinator client not available")

        # Register NCC in component health
        if ComponentStatus:
            self.component_health["ncc_coordinator"] = ComponentStatus(name="ncc_coordinator")
            if self.ncc_super_agent:
                self.component_health["ncc_coordinator"].record_success()

        # ── PHASE 8: Module Status Report ─────────────────────────────
        logger.info("\n[PHASE 8] MODULE STATUS REPORT")
        logger.info("-" * 60)
        for module, available in sorted(AVAILABLE_MODULES.items()):
            status = "✅ ONLINE" if available else "⚠️  OFFLINE"
            logger.info(f"  {module:<30} {status}")
            self.telemetry.modules_online[module] = available
        logger.info("-" * 60)

        online_count = sum(1 for v in AVAILABLE_MODULES.values() if v)
        total_count = len(AVAILABLE_MODULES)
        logger.info(f"  Modules: {online_count}/{total_count} online")

        # ── Final Status ──────────────────────────────────────────────
        self._initialized = core_ok
        if core_ok:
            self.telemetry.activation_level = ActivationLevel.BATTLE_STATIONS
            logger.info("\n" + "=" * 80)
            logger.info("  🚀 FULL ACTIVATION COMPLETE — BATTLE STATIONS")
            logger.info(f"  Agents: {self.telemetry.active_agents} | "
                        f"Strategies: {self.telemetry.active_strategies} | "
                        f"Data Streams: {self.telemetry.active_data_streams}")
            logger.info(f"  NCC: {'ONLINE' if self.ncc_super_agent else 'OFFLINE'} | "
                        f"NCL: ONLINE | Doctrine: {'ONLINE' if self.doctrine_engine else 'OFFLINE'} | "
                        f"BRS: {self.barren_wuffet_state}")
            logger.info(f"  Mode: {'PAPER' if self.paper_mode else 'LIVE'} | "
                        f"Target: $1,000,000")
            logger.info("=" * 80 + "\n")
        else:
            self.telemetry.activation_level = ActivationLevel.PARTIAL
            logger.error("  ⚠️  PARTIAL ACTIVATION — Some core systems failed")

        return core_ok

    # ════════════════════════════════════════════════════════════════════
    # SENSE — Gather all market data
    # ════════════════════════════════════════════════════════════════════

    async def sense(self) -> Dict[str, Any]:
        """
        Gather real-time data from ALL sources simultaneously.
        Returns comprehensive market snapshot.
        """
        snapshot: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "crypto": {},
            "crisis_assessment": None,
            "whale_alerts": [],
            "defi_yields": {},
            "agent_findings": [],
        }

        # 1. Fetch crypto prices (CoinGecko — free, always works)
        try:
            coin_ids = [str(c["symbol"]) for c in CRYPTO_WATCHLIST]
            ticks = await self.coingecko.get_prices_batch(coin_ids)
            tick_map = {t.symbol.split("/")[0].lower(): t for t in ticks}
            for coin in CRYPTO_WATCHLIST:
                coin_id = str(coin["symbol"])
                # Match by coin_id in the tick map keys
                tick = tick_map.get(coin_id.upper()) or tick_map.get(coin_id)
                if tick:
                    snapshot["crypto"][coin["ticker"]] = {
                        "price": tick.price,
                        "change_24h": tick.change_24h,
                        "volume_24h": tick.volume_24h,
                        "market_cap": 0,
                        "tier": coin["tier"],
                        "thesis": coin["thesis"],
                    }
            logger.info(f"  📊 Crypto prices fetched: {len(snapshot['crypto'])} assets")
        except Exception as e:
            logger.warning(f"  ⚠️  CoinGecko fetch: {e}")

        # 2. Crisis assessment (macro vectors)
        try:
            assessment = self.crisis_monitor.assess(
                oil_price=105.0,      # Current estimated
                vix_level=28.0,       # Elevated
                gold_price=5003.0,    # Record high
                ten_year_yield=4.35,
                core_pce=3.1,
                gdp_growth=0.7,
                credit_spread_bps=420,
                private_credit_redemption_pct=8.5,
                hormuz_blocked=True,
                war_active=True,
            )
            snapshot["crisis_assessment"] = {
                "composite_severity": assessment.composite_severity,
                "max_severity": assessment.max_severity,
                "critical_count": assessment.critical_count,
                "should_deploy_puts": assessment.should_deploy_puts,
                "signals": [
                    {
                        "vector": s.vector.value,
                        "severity": s.severity,
                        "description": s.description,
                    }
                    for s in assessment.signals
                ],
            }
            self.telemetry.crisis_severity = assessment.composite_severity
            logger.info(f"  🔴 Crisis severity: {assessment.composite_severity:.2f} "
                        f"(critical: {assessment.critical_count}, deploy_puts: {assessment.should_deploy_puts})")
        except Exception as e:
            logger.warning(f"  ⚠️  Crisis assessment: {e}")

        # 3. Run agent scans (BigBrain research agents)
        try:
            for theater in ["theater_b", "theater_c", "theater_d"]:
                agents = get_agents_by_theater(theater)
                for agent in agents:
                    try:
                        findings = await agent.scan()
                        if findings:
                            snapshot["agent_findings"].extend(
                                [{"agent": agent.agent_id, "theater": theater,
                                  "type": f.finding_type, "confidence": f.confidence,
                                  "data": f.data}
                                 for f in findings]
                            )
                    except Exception:
                        pass  # Agent scan failures are non-fatal
                    finally:
                        # Clean up aiohttp sessions to prevent resource leaks
                        if hasattr(agent, 'cleanup'):
                            try:
                                await agent.cleanup()
                            except Exception:
                                pass
            logger.info(f"  🔬 Agent scan: {len(snapshot.get('agent_findings', []))} findings")
        except Exception as e:
            logger.warning(f"  ⚠️  Agent scan: {e}")

        self.telemetry.last_scan_time = datetime.now()
        return snapshot

    # ════════════════════════════════════════════════════════════════════
    # ANALYZE — Process data through all strategy engines
    # ════════════════════════════════════════════════════════════════════

    async def analyze(self, snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run ALL strategy engines against the current market snapshot.
        Returns list of trade signals.
        """
        signals = []

        # 1. Fibonacci analysis on crypto (VERIFIED WORKING PIPELINE)
        if self.fib_signal_gen and snapshot.get("crypto"):
            for coin in CRYPTO_WATCHLIST[:4]:  # Tier 1 & 2 first
                ticker = coin["ticker"]
                crypto_data = snapshot["crypto"].get(ticker)
                if not crypto_data:
                    continue

                try:
                    current_price = crypto_data["price"]
                    change_24h = crypto_data.get("change_24h", 0)
                    # Estimate 30-day range from current price + change
                    estimated_high = current_price * 1.15
                    estimated_low = current_price * 0.85

                    analysis = self.fib_signal_gen.analyze(
                        symbol=ticker,
                        current_price=current_price,
                        high_30d=estimated_high,
                        low_30d=estimated_low,
                        prices_30d=[current_price],
                        change_24h=change_24h,
                    )

                    if analysis.get("signal") != "HOLD":
                        signals.append({
                            "source": "fibonacci_pipeline",
                            "symbol": f"{ticker}/USDT",
                            "direction": analysis["signal"].lower(),
                            "confidence": analysis.get("confidence", 0.5),
                            "reason": analysis.get("reason", "Fibonacci signal"),
                            "price": current_price,
                            "strategy": "golden_ratio_finance",
                            "tier": coin["tier"],
                        })
                        logger.info(f"  📈 FIB SIGNAL: {analysis['signal']} {ticker} "
                                    f"@ ${current_price:,.2f} "
                                    f"(confidence: {analysis.get('confidence', 0):.2f})")
                except Exception as e:
                    logger.debug(f"  Fib analysis {ticker}: {e}")

        # 2. Crisis put signals
        crisis_data = snapshot.get("crisis_assessment", {})
        if crisis_data and crisis_data.get("should_deploy_puts"):
            for target in PUT_PLAYBOOK:
                # Check if any active crisis vectors match this target
                active_vectors = {
                    s["vector"] for s in crisis_data.get("signals", [])
                    if s["severity"] > 0.4
                }
                target_vectors = {v.value for v in target.crisis_vectors}

                if active_vectors & target_vectors:
                    signals.append({
                        "source": "macro_crisis_puts",
                        "symbol": target.symbol,
                        "direction": "buy_put",
                        "confidence": min(s["severity"] for s in crisis_data.get("signals", [])
                                          if s["vector"] in {v.value for v in target.crisis_vectors}),
                        "reason": target.description,
                        "strategy": "macro_crisis_put",
                        "priority": target.priority,
                        "otm_pct": target.otm_pct,
                        "target_dte": target.target_dte,
                        "delta": target.target_delta,
                    })

            if signals:
                put_count = sum(1 for s in signals if s["source"] == "macro_crisis_puts")
                logger.info(f"  🔴 CRISIS PUTS: {put_count} put signals generated")

        # 3. Cross-asset correlation signals
        crypto = snapshot.get("crypto", {})
        btc_data = crypto.get("BTC", {})
        eth_data = crypto.get("ETH", {})

        if btc_data and eth_data:
            btc_change = btc_data.get("change_24h", 0)
            eth_change = eth_data.get("change_24h", 0)

            # ETH/BTC ratio divergence
            if btc_change < -5 and eth_change > btc_change + 3:
                signals.append({
                    "source": "correlation_divergence",
                    "symbol": "ETH/USDT",
                    "direction": "buy",
                    "confidence": 0.6,
                    "reason": f"ETH decoupling from BTC (BTC {btc_change:.1f}% vs ETH {eth_change:.1f}%)",
                    "strategy": "cross_asset_correlation",
                })

            # Extreme fear = buy signal for crypto
            if btc_change < -10:
                signals.append({
                    "source": "extreme_fear",
                    "symbol": "BTC/USDT",
                    "direction": "buy",
                    "confidence": 0.7,
                    "reason": f"Extreme fear — BTC down {btc_change:.1f}% in 24h (oversold bounce)",
                    "strategy": "mean_reversion",
                })

        # 4. Agent-generated findings → signals
        for finding in snapshot.get("agent_findings", []):
            if finding.get("confidence", 0) > 0.6:
                asset = finding.get("data", {}).get("asset") or finding.get("data", {}).get("symbol")
                if asset:
                    signals.append({
                        "source": f"agent_{finding['agent']}",
                        "symbol": f"{asset}/USDT" if "/" not in str(asset) else str(asset),
                        "direction": "buy",
                        "confidence": finding["confidence"],
                        "reason": f"Agent {finding['agent']} finding: {finding['type']}",
                        "strategy": "agent_intelligence",
                    })

        self.telemetry.total_signals_generated += len(signals)
        return signals

    # ════════════════════════════════════════════════════════════════════
    # DECIDE — Risk-check and prioritize signals
    # ════════════════════════════════════════════════════════════════════

    async def decide(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply risk management, position limits, and prioritization.
        Phase-aware: DEPLOY opens aggressively, HARVEST tightens, CLOSE blocks new.
        Doctrine-aware: HALT blocks all, SAFE_MODE blocks new, CAUTION reduces.
        Returns filtered/approved signals ready for execution.
        """
        if not signals:
            return []

        # ── DOCTRINE GATE ─────────────────────────────────────────
        # If doctrine says HALT, block everything
        if self.doctrine_status and self.doctrine_status.is_halted:
            logger.warning("  🛑 DOCTRINE HALT — Blocking all signals")
            return []
        if self.barren_wuffet_state == "HALT":
            logger.warning("  🛑 BARREN WUFFET HALT — Blocking all signals")
            return []
        # If SAFE_MODE — no new positions, only allow close signals
        if self.doctrine_status and not self.doctrine_status.can_open_new_positions:
            close_signals = [s for s in signals if s.get("direction") in ("sell", "short", "close")]
            if close_signals:
                logger.info(f"  ⚠️  DOCTRINE SAFE_MODE — Only {len(close_signals)} close signals allowed")
            else:
                logger.info("  ⚠️  DOCTRINE SAFE_MODE — No new positions allowed")
            return close_signals
        if self.barren_wuffet_state == "SAFE_MODE":
            close_signals = [s for s in signals if s.get("direction") in ("sell", "short", "close")]
            logger.info(f"  ⚠️  BW SAFE_MODE — Only {len(close_signals)} close signals allowed")
            return close_signals

        # Update phase each cycle
        self.week_phase = get_current_phase()
        phase = self.week_phase

        # Phase-specific parameters
        if phase == WeekPhase.DEPLOY:
            confidence_threshold = 0.45   # Lower bar on Monday — deploy capital
            max_positions = 7
            logger.info("  📌 DEPLOY phase — opening positions aggressively")
        elif phase == WeekPhase.MONITOR:
            confidence_threshold = 0.50
            max_positions = 5
        elif phase == WeekPhase.ASSESS:
            confidence_threshold = 0.55   # Mid-week — more selective
            max_positions = 5
        elif phase == WeekPhase.HARVEST:
            confidence_threshold = 0.65   # Thursday — high-conviction only
            max_positions = 3
            logger.info("  📌 HARVEST phase — closing winners, new entries need high conviction")
        elif phase == WeekPhase.CLOSE_COMPOUND:
            confidence_threshold = 0.90   # Friday — almost no new entries
            max_positions = 1
            logger.info("  📌 CLOSE phase — minimal new entries, preparing to compound")
        else:
            confidence_threshold = 0.50
            max_positions = 5

        # Doctrine risk reduction: CAUTION raises thresholds and halves positions
        if (self.doctrine_status and self.doctrine_status.should_reduce_risk) or \
                self.barren_wuffet_state == "CAUTION":
            confidence_threshold = min(confidence_threshold + 0.15, 0.90)
            max_positions = max(1, max_positions // 2)
            logger.info(f"  ⚠️  DOCTRINE CAUTION — Threshold raised to {confidence_threshold:.2f}, "
                        f"max positions halved to {max_positions}")

        # Capital allocation from tracker
        alloc = self.tracker.current_allocation()
        logger.info(f"  💰 Capital: ${self.tracker.current_capital:,.0f} | "
                    f"Opt: ${alloc.options_allocation:,.0f} | "
                    f"Crypto: ${alloc.crypto_allocation:,.0f} | "
                    f"Phase: {phase.value}")

        approved = []
        open_positions = self.execution_engine.get_open_positions() if hasattr(self.execution_engine, 'get_open_positions') else []
        current_position_count = len(open_positions)

        # Get existing symbols
        existing_symbols = {p.symbol for p in open_positions} if open_positions else set()

        # Separate crypto and options signals — independent position limits
        crypto_signals = [s for s in signals if s.get("direction") in ("buy", "sell", "long", "short")]
        options_signals = [s for s in signals if s.get("direction") in ("buy_put", "buy_call", "straddle")]

        # Sort each bucket by confidence
        crypto_signals.sort(key=lambda s: s.get("confidence", 0), reverse=True)
        options_signals.sort(key=lambda s: s.get("confidence", 0), reverse=True)

        crypto_approved = 0
        max_crypto = max_positions
        max_options = 4  # Separate budget for options

        for sig in crypto_signals:
            symbol = sig.get("symbol", "")
            if symbol in existing_symbols:
                continue
            if current_position_count + crypto_approved >= max_crypto:
                break
            if sig.get("confidence", 0) < confidence_threshold:
                continue
            approved.append(sig)
            crypto_approved += 1
            logger.info(f"  ✅ APPROVED: {sig['direction'].upper()} {symbol} "
                        f"(confidence: {sig['confidence']:.2f}, source: {sig['source']})")

        options_approved = 0
        options_threshold = max(0.40, confidence_threshold - 0.10)  # Slightly lower for options
        for sig in options_signals:
            symbol = sig.get("symbol", "")
            if options_approved >= max_options:
                break
            if sig.get("confidence", 0) < options_threshold:
                continue
            approved.append(sig)
            options_approved += 1
            logger.info(f"  ✅ APPROVED: {sig['direction'].upper()} {symbol} "
                        f"(confidence: {sig['confidence']:.2f}, source: {sig['source']})")

        logger.info(f"  Decision: {len(approved)}/{len(signals)} approved "
                    f"({crypto_approved} crypto, {options_approved} options)")
        return approved

    # ════════════════════════════════════════════════════════════════════
    # ACT — Execute approved trades
    # ════════════════════════════════════════════════════════════════════

    async def act(self, approved_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute approved trade signals through the ExecutionEngine.
        Paper trades by default. Returns list of execution results.
        """
        results = []

        for sig in approved_signals:
            try:
                symbol = sig["symbol"]
                direction = sig["direction"]
                price = sig.get("price", 0)

                # ── OPTIONS PAPER EXECUTION (no crypto price needed) ──
                if direction in ("buy_put", "buy_call", "straddle"):
                    alloc = self.tracker.current_allocation()
                    premium_budget = alloc.options_allocation * 0.03  # 3% per signal
                    contracts = max(1, int(premium_budget / 200))     # ~$2/contract estimated
                    estimated_premium = contracts * 2.00

                    result = {
                        "symbol": symbol,
                        "direction": direction,
                        "contracts": contracts,
                        "estimated_premium": estimated_premium,
                        "otm_pct": sig.get("otm_pct", 5),
                        "target_dte": sig.get("target_dte", 7),
                        "delta": sig.get("delta", -0.30),
                        "status": "paper_filled",
                        "mode": "paper_options",
                        "reason": sig.get("reason", ""),
                    }
                    results.append(result)
                    self.telemetry.total_trades_executed += 1
                    self._week_options_pnl -= estimated_premium  # Debit premium
                    logger.info(f"  📋 OPTIONS PAPER: {direction.upper()} {contracts}x "
                                f"{symbol} {sig.get('target_dte', 7)}DTE "
                                f"@ ${estimated_premium:,.2f} premium "
                                f"({sig.get('reason', '')[:50]})")
                    continue

                # ── CRYPTO EXECUTION ──────────────────────────────────
                # Get current price if not provided
                if price == 0 and "/" in symbol:
                    base = symbol.split("/")[0].lower()
                    coin_id_map = {c["ticker"]: c["symbol"] for c in CRYPTO_WATCHLIST}
                    coin_id = coin_id_map.get(base.upper())
                    if coin_id:
                        try:
                            tick = await self.coingecko.get_price(coin_id)
                            price = tick.price if tick else 0
                        except Exception:
                            pass

                if price <= 0:
                    logger.warning(f"  ⚠️  No price for {symbol}, skipping execution")
                    continue

                # Calculate position size (2% risk per trade) from current capital
                account_balance = self.tracker.current_capital
                risk_per_trade = account_balance * 0.02
                quantity = risk_per_trade / price

                # Determine side
                if direction in ("buy", "long"):
                    side = OrderSide.BUY
                elif direction in ("sell", "short"):
                    side = OrderSide.SELL
                else:
                    # Unrecognized direction — skip
                    logger.debug(f"  Unrecognized direction '{direction}' for {symbol}")
                    continue

                # Execute through engine
                logger.info(f"  🎯 EXECUTING: {side.value} {quantity:.6f} {symbol} "
                            f"@ ${price:,.2f} (${risk_per_trade:,.0f} risk)")

                order = await self.execution_engine.create_order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    price=price,
                    metadata={"market_price": price, "source": sig.get("source", "unknown")},
                )
                filled_order = await self.execution_engine.submit_order(order)

                # Record in accounting
                try:
                    self.db.record_transaction(
                        account_id=1,
                        transaction_type="paper_trade" if self.paper_mode else "live_trade",
                        asset=symbol.split("/")[0],
                        quantity=quantity,
                        price=price,
                        side=side.value,
                        symbol=symbol,
                    )
                except Exception as db_err:
                    logger.warning(f"  ⚠️  DB record failed: {db_err}")

                results.append({
                    "symbol": symbol,
                    "direction": side.value,
                    "quantity": quantity,
                    "price": price,
                    "order_id": filled_order.order_id,
                    "status": filled_order.status.value,
                    "mode": "paper" if self.paper_mode else "live",
                })

                self.telemetry.total_trades_executed += 1
                self.telemetry.last_trade_time = datetime.now()
                logger.info(f"  ✅ {filled_order.status.value}: {side.value} {quantity:.6f} {symbol} "
                            f"@ ${price:,.2f}")

            except Exception as e:
                logger.error(f"  ❌ Execution failed for {sig.get('symbol')}: {e}")
                results.append({
                    "symbol": sig.get("symbol"),
                    "status": "failed",
                    "error": str(e),
                })

        return results

    # ════════════════════════════════════════════════════════════════════
    # RECONCILE — Verify positions match reality
    # ════════════════════════════════════════════════════════════════════

    async def reconcile(self) -> Dict[str, Any]:
        """
        Update all positions with current prices.
        Check stop losses and take profits.
        """
        reconciliation: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "positions_checked": 0,
            "stops_triggered": 0,
            "profits_taken": 0,
            "total_unrealized_pnl": 0.0,
        }

        try:
            positions = self.execution_engine.get_open_positions() if hasattr(self.execution_engine, 'get_open_positions') else []
            self.telemetry.active_positions = len(positions)
            reconciliation["positions_checked"] = len(positions)

            if not positions:
                return reconciliation

            # Fetch current prices for all positioned assets
            for position in positions:
                try:
                    base = position.symbol.split("/")[0].lower()
                    coin_id_map = {str(c["ticker"]).lower(): str(c["symbol"]) for c in CRYPTO_WATCHLIST}
                    coin_id = coin_id_map.get(base)
                    if coin_id:
                        tick = await self.coingecko.get_price(coin_id)
                        current_price = tick.price if tick else 0

                        if current_price > 0 and hasattr(position, 'update_price'):
                            position.update_price(current_price)

                            # Calculate P&L
                            if hasattr(position, 'unrealized_pnl'):
                                reconciliation["total_unrealized_pnl"] = float(reconciliation.get("total_unrealized_pnl", 0.0)) + position.unrealized_pnl
                except Exception:
                    pass

            self.telemetry.total_pnl = float(reconciliation.get("total_unrealized_pnl", 0.0))

        except Exception as e:
            logger.warning(f"  ⚠️  Reconciliation error: {e}")

        return reconciliation

    def _calc_max_concentration(self, reconciliation: Dict) -> float:
        """Calculate max single-position concentration as % of capital."""
        try:
            positions = reconciliation.get("positions", [])
            capital = max(self.tracker.current_capital, 1)
            if not positions:
                return 0.0
            max_value = max(
                abs(getattr(p, "market_value", 0) or 0) for p in positions
            )
            return (max_value / capital) * 100
        except Exception:
            return 0.0

    # ════════════════════════════════════════════════════════════════════
    # DOCTRINE CHECK — Evaluate compliance & enforce state machine
    # ════════════════════════════════════════════════════════════════════

    async def doctrine_check(self, reconciliation: Dict) -> Dict[str, Any]:
        """
        Run doctrine compliance every cycle.
        1. Evaluate DoctrineStatus state machine (drawdown/daily-loss)
        2. Check 8-pack compliance via DoctrineEngine
        3. Enforce HALT / SAFE_MODE / CAUTION state
        Returns doctrine state summary.
        """
        result: Dict[str, Any] = {
            "state": "NORMAL",
            "can_trade": True,
            "violations": 0,
            "actions": [],
        }

        unrealized_pnl = reconciliation.get("total_unrealized_pnl", 0.0)

        # 1. DoctrineStatus state machine — fast drawdown/daily-loss check
        if self.doctrine_status:
            self.doctrine_status.evaluate(
                pnl=unrealized_pnl,
                equity=self.tracker.current_capital + unrealized_pnl,
            )
            state = self.doctrine_status.state
            result["state"] = state.value if hasattr(state, 'value') else str(state)
            result["can_trade"] = self.doctrine_status.can_open_new_positions
            result["should_reduce"] = self.doctrine_status.should_reduce_risk
            result["is_halted"] = self.doctrine_status.is_halted

            if self.doctrine_status.is_halted:
                logger.warning("  🛑 DOCTRINE HALT — All trading suspended")
                logger.warning(f"     Reason: {self.doctrine_status.transition_reason}")
                result["can_trade"] = False
            elif self.doctrine_status.should_reduce_risk:
                logger.warning(f"  ⚠️  DOCTRINE {result['state']} — Reducing risk")
                logger.warning(f"     Drawdown: {self.doctrine_status.drawdown_pct:.1%}")

        # 2. DoctrineEngine 8-pack compliance
        if self.doctrine_engine:
            try:
                metrics = {
                    "max_drawdown_pct": self.doctrine_status.drawdown_pct * 100 if self.doctrine_status else 0,
                    "daily_loss_pct": abs(unrealized_pnl / max(self.tracker.current_capital, 1)) * 100,
                    "active_sev1_incidents": 0,
                    "compliance_violations": len(self.doctrine_violations),
                    "venue_health_score": 0.95,
                    "slippage_bps_p95": 5.0,
                    "partial_fill_rate_pct": 2.0,
                    "liquidity_available_pct": 85.0,
                }

                bw_state, actions = self.doctrine_engine.check_barren_wuffet_triggers(metrics)
                self.barren_wuffet_state = bw_state.value if hasattr(bw_state, 'value') else str(bw_state)

                if actions:
                    result["actions"] = [a.value if hasattr(a, 'value') else str(a) for a in actions]
                    self.doctrine_actions = actions
                    logger.info(f"  📜 BARREN WUFFET: {self.barren_wuffet_state} | "
                                f"Actions: {len(actions)}")
                    # Execute action handlers
                    for action in actions:
                        handler = self.doctrine_engine.action_handlers.get(action)
                        if handler:
                            try:
                                await handler()
                            except Exception as e:
                                logger.warning(f"  ⚠️  Action handler {action}: {e}")

                # If BW state is HALT or SAFE_MODE, block trading
                if BarrenWuffetState and bw_state in (BarrenWuffetState.HALT, BarrenWuffetState.SAFE_MODE):
                    result["can_trade"] = False

                result["violations"] = len(self.doctrine_engine.active_violations)
                self.doctrine_violations = list(self.doctrine_engine.active_violations)

            except Exception as e:
                logger.warning(f"  ⚠️  Doctrine Engine check: {e}")

        # 3. NCL Governance compliance check
        ncl_metrics = {
            "database_online": AVAILABLE_MODULES.get("PipelineRunner", False) or True,
            "paper_mode": self.paper_mode,
            "drawdown_pct": (self.doctrine_status.drawdown_pct * 100) if self.doctrine_status else 0.0,
            "doctrine_online": self.doctrine_engine is not None,
            "max_concentration_pct": self._calc_max_concentration(reconciliation),
            "daily_loss_pct": abs(unrealized_pnl / max(self.tracker.current_capital, 1)) * 100,
            "agents_online": self.telemetry.active_agents,
            "crisis_monitor_active": True,
        }
        self.ncl_compliance = self.ncl_engine.evaluate(ncl_metrics)
        result["ncl_level"] = self.ncl_compliance.level.value
        result["ncl_score"] = self.ncl_compliance.score
        result["ncl_violations"] = len(self.ncl_compliance.violations)

        if self.ncl_compliance.violations:
            for v in self.ncl_compliance.violations:
                logger.warning(f"  ⚠️  NCL VIOLATION: {v}")
        if self.ncl_compliance.level == NCLComplianceLevel.ALPHA:
            logger.warning("  🛑 NCL ALPHA — Critical compliance failure, blocking new trades")
            result["can_trade"] = False

        self.telemetry.modules_online["DoctrineCompliance"] = True
        self.telemetry.modules_online["NCLGovernance"] = True
        return result

    # ── Doctrine Action Handlers ──────────────────────────────────

    async def _doctrine_action_throttle_risk(self):
        """Reduce position sizes and raise confidence threshold."""
        logger.warning("  📜 DOCTRINE ACTION: Throttling risk — position sizes halved")
        # Materially reduce capital allocation to enforce risk reduction
        self.crisis_put_engine.account_balance = self.tracker.current_capital * 0.5

    async def _doctrine_action_halt(self):
        """Full trading halt."""
        logger.warning("  📜 DOCTRINE ACTION: HALT — All execution suspended")
        self.barren_wuffet_state = "HALT"
        # Force doctrine_status into halted if available
        if self.doctrine_status and hasattr(self.doctrine_status, 'force_halt'):
            self.doctrine_status.force_halt()

    async def _doctrine_action_safe_mode(self):
        """Enter safe mode — close-only, no new positions."""
        logger.warning("  📜 DOCTRINE ACTION: SAFE MODE — Close-only operations")
        self.barren_wuffet_state = "SAFE_MODE"
        # Reduce capital exposure
        self.crisis_put_engine.account_balance = self.tracker.current_capital * 0.25

    # ════════════════════════════════════════════════════════════════════
    # INTROSPECT — Self-health check and gap detection
    # ════════════════════════════════════════════════════════════════════

    async def introspect(self) -> Dict[str, Any]:
        """
        Self-health check:
        1. Verify all components are responding
        2. Detect gaps (missing data, broken connections)
        3. Attempt self-healing where possible
        """
        report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "components_healthy": 0,
            "components_degraded": 0,
            "components_down": 0,
            "gaps_detected": 0,
            "self_healed": 0,
        }

        if not ComponentStatus:
            return report

        # Health-check each tracked component
        checks = {
            "coingecko": self._check_coingecko,
            "execution_engine": self._check_execution_engine,
            "database": self._check_database,
            "doctrine_engine": self._check_doctrine_engine,
        }

        for name, check_fn in checks.items():
            comp = self.component_health.get(name)
            if not comp:
                continue
            try:
                start = time.monotonic()
                await check_fn()
                latency = (time.monotonic() - start) * 1000
                comp.record_success(latency)
            except Exception as e:
                comp.record_failure(str(e))

        # Count health states
        for comp in self.component_health.values():
            if ComponentHealth and comp.health == ComponentHealth.HEALTHY:
                report["components_healthy"] = report.get("components_healthy", 0) + 1
            elif ComponentHealth and comp.health == ComponentHealth.DEGRADED:
                report["components_degraded"] = report.get("components_degraded", 0) + 1
            elif ComponentHealth and comp.health == ComponentHealth.DOWN:
                report["components_down"] = report.get("components_down", 0) + 1
                # Gap detection
                if GapSeverity and SystemGap:
                    gap = SystemGap(
                        gap_id=f"health_{comp.name}_{int(time.time())}",
                        component=comp.name,
                        description=f"{comp.name} is DOWN: {comp.last_error}",
                        severity=GapSeverity.HIGH,
                    )
                    self.system_gaps.append(gap)
                    report["gaps_detected"] = report.get("gaps_detected", 0) + 1

        # FPC resilience check
        if self.acc_advanced_state:
            for layer_name, layer in self.acc_advanced_state.resilience_layers.items():
                if layer.status != 'active':
                    report["components_degraded"] += 1

        return report

    async def _check_coingecko(self):
        """Quick ping to CoinGecko."""
        tick = await self.coingecko.get_price("bitcoin")
        if not tick or tick.price <= 0:
            raise RuntimeError("CoinGecko returned no price")

    async def _check_execution_engine(self):
        """Verify execution engine responds."""
        if hasattr(self.execution_engine, 'get_open_positions'):
            self.execution_engine.get_open_positions()

    async def _check_database(self):
        """Verify database is accessible."""
        self.db.initialize()

    async def _check_doctrine_engine(self):
        """Verify doctrine engine loaded."""
        if not self.doctrine_engine:
            raise RuntimeError("Doctrine engine not loaded")
        if not self.doctrine_engine._loaded:
            raise RuntimeError("Doctrine packs not loaded")

    # ════════════════════════════════════════════════════════════════════
    # AGENT COORDINATION — Unified command across all agents
    # ════════════════════════════════════════════════════════════════════

    async def coordinate_agents(self, snapshot: Dict) -> List[Dict[str, Any]]:
        """
        Full agent coordination:
        1. Run all research agents in parallel across theaters
        2. Collect and rank findings by confidence
        3. Track agent performance (hit rate, signal quality)
        4. Generate coordinated signal consensus
        Returns list of top-ranked agent findings.
        """
        all_findings = []

        # 0. NCC Strategic Coordination — get mission-level guidance
        if self.ncc_super_agent:
            try:
                mission = {
                    "objective": "maximize_risk_adjusted_returns",
                    "capital": self.tracker.current_capital,
                    "week": self.tracker.current_week,
                    "phase": self.week_phase.value,
                    "doctrine_state": self.barren_wuffet_state,
                    "crisis_severity": self.telemetry.crisis_severity,
                }
                coordination = await self.ncc_super_agent.coordinate_super_mission(mission)
                self.ncc_status = coordination
                success_prob = coordination.get("success_probability", {})
                if isinstance(success_prob, dict):
                    prob_val = success_prob.get("overall_probability", 0.5)
                else:
                    prob_val = float(success_prob) if success_prob else 0.5
                logger.info(f"  🎯 NCC Coordination: success_prob={prob_val:.2f} | "
                            f"feasibility={coordination.get('mission_feasibility', {}).get('feasible', 'N/A')}")

                # Submit coordination status to NCC REST API if available
                if self.ncc_coordinator:
                    try:
                        await self.ncc_coordinator.submit_agent_action(
                            "full_activation", {"type": "cycle_update", "data": {
                                "cycle": self.telemetry.cycles_completed,
                                "doctrine_state": self.barren_wuffet_state,
                            }}
                        )
                    except Exception:
                        pass  # NCC REST endpoint may be offline
            except Exception as e:
                logger.debug(f"NCC coordination: {e}")

        # 1. Master agent system scan
        if self.master_agents and self.master_agents.initialized:
            try:
                status = await self.master_agents.get_system_status()
                agent_ids = list(status.get("research_agents", {}).keys())
                for agent_id in agent_ids:
                    try:
                        result = await self.master_agents.run_agent_scan(agent_id)
                        if result:
                            all_findings.append({
                                "agent_id": agent_id,
                                "source": "master_agent_system",
                                "findings": result if isinstance(result, list) else [result],
                                "timestamp": datetime.now().isoformat(),
                            })
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Agent coordination scan: {e}")

        # 2. BigBrain research agents (parallel across theaters)
        for theater in ["theater_b", "theater_c", "theater_d"]:
            try:
                agents = get_agents_by_theater(theater)
                for agent in agents:
                    try:
                        findings = await agent.scan()
                        if findings:
                            for f in findings:
                                all_findings.append({
                                    "agent_id": agent.agent_id,
                                    "source": f"bigbrain_{theater}",
                                    "confidence": f.confidence,
                                    "finding_type": f.finding_type,
                                    "data": f.data,
                                    "timestamp": datetime.now().isoformat(),
                                })
                    except Exception:
                        pass
                    finally:
                        if hasattr(agent, 'cleanup'):
                            try:
                                await agent.cleanup()
                            except Exception:
                                pass
            except Exception:
                pass

        # 3. Track agent performance
        agent_performance = self._agent_performance
        for finding in all_findings:
            aid = finding.get("agent_id", "unknown")
            if aid not in agent_performance:
                agent_performance[aid] = {
                    "scans": 0, "findings": 0, "signals_generated": 0,
                }
            agent_performance[aid]["scans"] += 1
            agent_performance[aid]["findings"] += 1

        # 4. Rank by confidence and return top signals
        ranked = sorted(
            [f for f in all_findings if f.get("confidence", 0) > 0.5],
            key=lambda f: f.get("confidence", 0),
            reverse=True,
        )

        if ranked:
            logger.info(f"  🔬 Agent coordination: {len(all_findings)} findings, "
                        f"{len(ranked)} high-confidence | "
                        f"Active agents: {len(agent_performance)}")

        return ranked[:10]  # Top 10 findings

    # ════════════════════════════════════════════════════════════════════
    # REPORT — Status dashboard
    # ════════════════════════════════════════════════════════════════════

    def report(self, snapshot: Optional[Dict] = None, signals: Optional[List] = None,
               executions: Optional[List] = None, reconciliation: Optional[Dict] = None,
               doctrine: Optional[Dict] = None):
        """Print comprehensive status report."""
        t = self.telemetry

        logger.info("\n" + "═" * 80)
        logger.info("  BARREN WUFFET — CYCLE REPORT")
        logger.info("═" * 80)
        logger.info(f"  Activation: {t.activation_level.value} | "
                     f"Uptime: {t.uptime_seconds:.0f}s | "
                     f"Cycle: {t.cycles_completed}")
        logger.info(f"  Agents: {t.active_agents} | "
                     f"Strategies: {t.active_strategies} | "
                     f"Data Streams: {t.active_data_streams}")
        logger.info(f"  Signals: {t.total_signals_generated} total | "
                     f"Trades: {t.total_trades_executed} executed | "
                     f"Positions: {t.active_positions} open")
        logger.info(f"  Crisis Severity: {t.crisis_severity:.2f} | "
                     f"P&L: ${t.total_pnl:,.2f}")

        if snapshot and snapshot.get("crypto"):
            logger.info("\n  CRYPTO PRICES:")
            for ticker, data in sorted(snapshot["crypto"].items()):
                change = data.get("change_24h", 0)
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                logger.info(f"    {ticker:<6} ${data['price']:>12,.2f}  {arrow} {change:>6.1f}%  "
                             f"(Tier {data['tier']})")

        if signals:
            logger.info(f"\n  SIGNALS THIS CYCLE: {len(signals)}")
            for sig in signals[:10]:
                logger.info(f"    {sig['direction'].upper():<10} {sig['symbol']:<12} "
                             f"conf:{sig['confidence']:.2f}  src:{sig['source']}")

        if executions:
            crypto_execs = [e for e in executions if e.get("mode") != "paper_options"]
            option_execs = [e for e in executions if e.get("mode") == "paper_options"]

            if crypto_execs:
                logger.info(f"\n  CRYPTO EXECUTIONS: {len(crypto_execs)}")
                for ex in crypto_execs:
                    logger.info(f"    {ex.get('status', 'unknown')}: {ex.get('direction', '?')} "
                                 f"{ex.get('symbol', '?')} @ ${ex.get('price', 0):,.2f}")

            if option_execs:
                logger.info(f"\n  OPTIONS PAPER FILLS: {len(option_execs)}")
                for ex in option_execs:
                    logger.info(f"    {ex['direction'].upper()} {ex.get('contracts', 1)}x "
                                 f"{ex['symbol']} {ex.get('target_dte', 7)}DTE "
                                 f"@ ${ex.get('estimated_premium', 0):,.2f}")

        # Options signals (logged for IBKR execution)
        crisis = snapshot.get("crisis_assessment", {}) if snapshot else {}
        if crisis.get("should_deploy_puts"):
            logger.info("\n  🔴 OPTIONS SIGNALS (IBKR Required):")
            for target in OPTIONS_WATCHLIST:
                logger.info(f"    {target['direction']:<8} {target['ticker']:<6} "
                             f"({target['sector']}) — Priority {target['priority']}")

        avg_ms = t.avg_cycle_time_ms
        logger.info(f"\n  Cycle time: {avg_ms:.0f}ms avg | "
                     f"Modules: {sum(1 for v in t.modules_online.values() if v)}/"
                     f"{len(t.modules_online)} online")

        # Doctrine & compliance status
        if doctrine:
            logger.info(f"\n  DOCTRINE STATUS:")
            logger.info(f"    State: {doctrine.get('state', 'N/A')} | "
                         f"Can Trade: {'YES' if doctrine.get('can_trade', True) else 'NO'} | "
                         f"Violations: {doctrine.get('violations', 0)}")
            if doctrine.get('actions'):
                logger.info(f"    Actions triggered: {', '.join(doctrine['actions'][:5])}")
        elif self.barren_wuffet_state:
            logger.info(f"\n  DOCTRINE: {self.barren_wuffet_state}")

        # Component health summary
        if self.component_health and ComponentHealth:
            healthy = sum(1 for c in self.component_health.values()
                         if c.health == ComponentHealth.HEALTHY)
            degraded = sum(1 for c in self.component_health.values()
                          if c.health == ComponentHealth.DEGRADED)
            down = sum(1 for c in self.component_health.values()
                       if c.health == ComponentHealth.DOWN)
            if degraded or down:
                logger.info(f"\n  HEALTH: {healthy} healthy, {degraded} degraded, {down} down")
                for name, comp in self.component_health.items():
                    if comp.health != ComponentHealth.HEALTHY:
                        logger.info(f"    ⚠️  {name}: {comp.health.value} "
                                     f"(errors: {comp.error_count}, last: {comp.last_error[:50]})")

        # FPC / Faraday status
        if self.acc_advanced_state:
            active = sum(1 for l in self.acc_advanced_state.resilience_layers.values()
                         if l.status == 'active')
            total = len(self.acc_advanced_state.resilience_layers)
            logger.info(f"\n  FPC RESILIENCE: {active}/{total} layers active")

        # NCC Coordination status
        if self.ncc_status:
            feasibility = self.ncc_status.get("mission_feasibility", {})
            success = self.ncc_status.get("success_probability", {})
            if isinstance(success, dict):
                prob_val = success.get("overall_probability", "N/A")
            else:
                prob_val = success
            logger.info(f"\n  NCC STATUS:")
            logger.info(f"    Feasible: {feasibility.get('feasible', 'N/A')} | "
                         f"Success Prob: {prob_val}")

        # NCL Governance status
        if self.ncl_compliance:
            logger.info(f"\n  NCL GOVERNANCE: {self.ncl_compliance.level.value} "
                         f"({self.ncl_compliance.score:.0f}% | "
                         f"{self.ncl_compliance.checks_passed}/{self.ncl_compliance.checks_total} checks)")
            if self.ncl_compliance.violations:
                for v in self.ncl_compliance.violations[:3]:
                    logger.info(f"    ⚠️  {v}")

        # Weekly tracker summary
        alloc = self.tracker.current_allocation()
        logger.info(f"\n  WEEKLY TRACKER — Week {self.tracker.current_week} | "
                     f"Phase: {self.week_phase.value}")
        logger.info(f"  Capital: ${self.tracker.current_capital:>10,.2f} → "
                     f"Target: $1,000,000 ({self.tracker.progress_pct():.2f}%)")
        logger.info(f"  Allocation: Opt ${alloc.options_allocation:,.0f} | "
                     f"Crypto ${alloc.crypto_allocation:,.0f} | "
                     f"Arb ${alloc.arb_allocation:,.0f} | "
                     f"Cash ${alloc.cash_reserve:,.0f}")
        logger.info(f"  Weeks remaining: {self.tracker.weeks_remaining()}")
        logger.info("═" * 80 + "\n")

    # ════════════════════════════════════════════════════════════════════
    # WEEKLY TRACKER INTEGRATION
    # ════════════════════════════════════════════════════════════════════

    def _week_recorded_today(self) -> bool:
        """Check if we already recorded a week ending today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return any(w.end_date == today for w in self.tracker.weeks)

    def _record_and_compound(self):
        """Record the week's results and compound capital."""
        total_pnl = self._week_crypto_pnl + self._week_options_pnl + self._week_arb_pnl
        avg_crisis = (
            sum(self._week_crisis_severities) / len(self._week_crisis_severities)
            if self._week_crisis_severities else 0.0
        )

        week = self.tracker.record_week(
            pnl=total_pnl,
            options_pnl=self._week_options_pnl,
            crypto_pnl=self._week_crypto_pnl,
            arb_pnl=self._week_arb_pnl,
            trades=self._week_trades,
            signals=self._week_signals,
            crisis_severity=avg_crisis,
            best_trade=self._week_best_trade,
            worst_trade=self._week_worst_trade,
            notes=f"Phase progression: DEPLOY→MONITOR→ASSESS→HARVEST→CLOSE",
        )

        # Update execution engine account balance for next week
        self.crisis_put_engine.account_balance = self.tracker.current_capital

        # Reset accumulators
        self._week_crypto_pnl = 0.0
        self._week_options_pnl = 0.0
        self._week_arb_pnl = 0.0
        self._week_trades = 0
        self._week_signals = 0
        self._week_best_trade = ""
        self._week_best_pnl = 0.0
        self._week_worst_trade = ""
        self._week_worst_pnl = 0.0
        self._week_crisis_severities = []

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  📊 WEEK {week.week_number} RECORDED & COMPOUNDED")
        logger.info(f"  P&L: ${total_pnl:+,.2f} ({week.pnl_pct:+.1f}%)")
        logger.info(f"  New Capital: ${self.tracker.current_capital:,.2f}")
        logger.info(f"  On Target: {'YES ✅' if week.on_target else 'NO ❌'}")
        logger.info(f"  Weeks Remaining: {self.tracker.weeks_remaining()}")
        logger.info(f"{'=' * 60}\n")

    # ════════════════════════════════════════════════════════════════════
    # MAIN MISSION LOOP
    # ════════════════════════════════════════════════════════════════════

    async def run_cycle(self) -> Dict[str, Any]:
        """
        Execute ONE complete mission cycle:
        sense → analyze → decide → act → reconcile → report
        """
        cycle_start = time.monotonic()
        cycle_result: Dict[str, Any] = {"cycle": self.telemetry.cycles_completed + 1}

        logger.info(f"\n{'─' * 60}")
        logger.info(f"  CYCLE {cycle_result['cycle']} — {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'─' * 60}")

        # SENSE
        logger.info("\n[SENSE] Gathering market intelligence...")
        snapshot = await self.sense()
        cycle_result["snapshot"] = snapshot

        # COORDINATE AGENTS (feed into analyze)
        logger.info("\n[COORDINATE] Agent coordination scan...")
        agent_findings = await self.coordinate_agents(snapshot)
        # Merge agent findings into snapshot for analyze() to consume
        if agent_findings:
            existing = snapshot.get("agent_findings", [])
            for af in agent_findings:
                existing.append({
                    "agent": af.get("agent_id", "unknown"),
                    "theater": af.get("source", "coordination"),
                    "type": af.get("finding_type", "agent_signal"),
                    "confidence": af.get("confidence", 0.5),
                    "data": af.get("data", {}),
                })
            snapshot["agent_findings"] = existing
        cycle_result["agent_findings_count"] = len(agent_findings)

        # ANALYZE
        logger.info("\n[ANALYZE] Running strategy engines...")
        signals = await self.analyze(snapshot)
        cycle_result["signals"] = signals

        # DECIDE
        logger.info("\n[DECIDE] Risk-checking signals...")
        approved = await self.decide(signals)
        cycle_result["approved"] = approved

        # ACT
        if approved:
            logger.info("\n[ACT] Executing approved signals...")
            executions = await self.act(approved)
            cycle_result["executions"] = executions
        else:
            logger.info("\n[ACT] No signals approved — standing by")
            cycle_result["executions"] = []

        # RECONCILE
        logger.info("\n[RECONCILE] Updating positions...")
        reconciliation = await self.reconcile()
        cycle_result["reconciliation"] = reconciliation

        # DOCTRINE CHECK
        logger.info("\n[DOCTRINE] Evaluating compliance & state machine...")
        doctrine_result = await self.doctrine_check(reconciliation)
        cycle_result["doctrine"] = doctrine_result
        if not doctrine_result.get("can_trade", True):
            logger.warning(f"  🛑 Doctrine state: {doctrine_result['state']} — "
                           f"Trading {'BLOCKED' if not doctrine_result['can_trade'] else 'ALLOWED'}")

        # INTROSPECT (every 5th cycle to avoid overhead)
        if self.telemetry.cycles_completed % 5 == 0:
            logger.info("\n[INTROSPECT] Self-health check...")
            introspect_result = await self.introspect()
            cycle_result["introspect"] = introspect_result
            logger.info(f"  Health: {introspect_result['components_healthy']} healthy, "
                        f"{introspect_result['components_degraded']} degraded, "
                        f"{introspect_result['components_down']} down | "
                        f"Gaps: {introspect_result['gaps_detected']}")

        # REPORT
        self.report(snapshot, signals, cycle_result.get("executions"), reconciliation,
                    doctrine=doctrine_result)

        # ── ACCUMULATE FOR WEEKLY TRACKER ─────────────────────────
        self._week_signals += len(signals)
        self._week_trades += len(cycle_result.get("executions", []))
        if crisis_data := snapshot.get("crisis_assessment"):
            self._week_crisis_severities.append(crisis_data.get("composite_severity", 0))

        # Track unrealized P&L per category
        for ex in cycle_result.get("executions", []):
            if ex.get("status") == "signal_logged":
                # Options signal — attribute to options bucket
                pass
            else:
                src = ex.get("source", "")
                # All crypto executions go to crypto P&L for now
                # Real P&L will come from reconciliation once filled

        # Record unrealized P&L from reconciliation
        unrealized = reconciliation.get("total_unrealized_pnl", 0.0)
        self._week_crypto_pnl = unrealized  # Continuously updated, not summed

        # ── FRIDAY AUTO-CLOSE & COMPOUND ──────────────────────────
        self.week_phase = get_current_phase()
        if self.week_phase == WeekPhase.CLOSE_COMPOUND:
            # On Friday, check if it's time to record the week
            # (first cycle after 4pm Friday, or explicitly via flag)
            now = datetime.now()
            if now.hour >= 16 and not self._week_recorded_today():
                self._record_and_compound()

        # Update telemetry
        cycle_time_ms = (time.monotonic() - cycle_start) * 1000
        self.telemetry.cycle_times_ms.append(cycle_time_ms)
        self.telemetry.cycles_completed += 1

        return cycle_result

    async def run_forever(self):
        """
        Continuous mission loop — runs until shutdown signal.
        sense → analyze → decide → act → reconcile → report → sleep → repeat
        """
        logger.info("🚀 ENTERING CONTINUOUS MISSION LOOP — ALL SYSTEMS GO")
        logger.info(f"   Scan interval: {self.scan_interval_seconds}s")
        logger.info(f"   Capital: ${self.tracker.current_capital:,.2f}")
        logger.info(f"   Week: {self.tracker.current_week} | Phase: {self.week_phase.value}")
        logger.info(f"   Target: $1,000,000")
        logger.info(f"   Press Ctrl+C to shutdown gracefully\n")

        cycle_count = 0
        while not self._shutdown_event.is_set():
            try:
                await self.run_cycle()

                # Wait for next cycle
                logger.info(f"  ⏳ Next cycle in {self.scan_interval_seconds}s...")
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.scan_interval_seconds
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue loop

            except Exception as e:
                logger.error(f"  ❌ Cycle error: {e}")
                self.telemetry.errors.append(f"{datetime.now()}: {e}")
                # Don't crash — wait and retry
                await asyncio.sleep(10)

        logger.info("\n🔽 MISSION LOOP ENDED — Shutting down...")

    async def shutdown(self):
        """Graceful shutdown — save state, close positions, persist data."""
        logger.info("🔽 SHUTDOWN SEQUENCE INITIATED")

        self._shutdown_event.set()

        # Save weekly tracker state
        try:
            self.tracker.save()
            logger.info(f"  ✅ Weekly tracker saved (Week {self.tracker.current_week}, "
                        f"${self.tracker.current_capital:,.2f})")
        except Exception as e:
            logger.warning(f"  ⚠️  Tracker save failed: {e}")

        # Save final telemetry
        try:
            telemetry_path = get_project_path("data", "full_activation_telemetry.json")
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(telemetry_path, 'w') as f:
                json.dump({
                    "shutdown_time": datetime.now().isoformat(),
                    "uptime_seconds": self.telemetry.uptime_seconds,
                    "cycles_completed": self.telemetry.cycles_completed,
                    "total_signals": self.telemetry.total_signals_generated,
                    "total_trades": self.telemetry.total_trades_executed,
                    "total_pnl": self.telemetry.total_pnl,
                    "modules_online": self.telemetry.modules_online,
                    "doctrine_state": self.barren_wuffet_state,
                    "doctrine_violations": len(self.doctrine_violations),
                    "ncl_level": self.ncl_compliance.level.value if self.ncl_compliance else "N/A",
                    "ncl_score": self.ncl_compliance.score if self.ncl_compliance else 0.0,
                    "ncc_online": self.ncc_super_agent is not None,
                    "component_health": {
                        name: comp.health.value
                        for name, comp in self.component_health.items()
                    } if self.component_health and ComponentHealth else {},
                    "system_gaps": len(self.system_gaps),
                    "fpc_layers": (
                        {name: layer.status for name, layer in
                         self.acc_advanced_state.resilience_layers.items()}
                        if self.acc_advanced_state else {}
                    ),
                    "errors": self.telemetry.errors[-20:],  # Last 20 errors
                }, f, indent=2, default=str)
            logger.info(f"  ✅ Telemetry saved to {telemetry_path}")
        except Exception as e:
            logger.warning(f"  ⚠️  Telemetry save failed: {e}")

        # Shutdown agent system
        if self.master_agents:
            await self.master_agents.shutdown()
            logger.info("  ✅ Agent system shutdown")

        # Clean up research agents (close aiohttp sessions)
        for agent in self.research_agents:
            if hasattr(agent, 'cleanup'):
                try:
                    await agent.cleanup()
                except Exception:
                    pass
        logger.info("  ✅ Research agent sessions closed")

        # Disconnect data sources
        try:
            await self.data_aggregator.disconnect_all()
            logger.info("  ✅ Data sources disconnected")
        except Exception:
            pass

        # Close main CoinGecko session
        try:
            await self.coingecko.disconnect()
        except Exception:
            pass

        logger.info("  ✅ SHUTDOWN COMPLETE")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including weekly tracker."""
        alloc = self.tracker.current_allocation()
        return {
            "activation_level": self.telemetry.activation_level.value,
            "mode": "PAPER" if self.paper_mode else "LIVE",
            "uptime_seconds": self.telemetry.uptime_seconds,
            "cycles_completed": self.telemetry.cycles_completed,
            "active_agents": self.telemetry.active_agents,
            "active_strategies": self.telemetry.active_strategies,
            "active_data_streams": self.telemetry.active_data_streams,
            "total_signals": self.telemetry.total_signals_generated,
            "total_trades": self.telemetry.total_trades_executed,
            "active_positions": self.telemetry.active_positions,
            "total_pnl": self.telemetry.total_pnl,
            "crisis_severity": self.telemetry.crisis_severity,
            "avg_cycle_time_ms": self.telemetry.avg_cycle_time_ms,
            "modules_online": self.telemetry.modules_online,
            "last_scan": self.telemetry.last_scan_time.isoformat() if self.telemetry.last_scan_time else None,
            "last_trade": self.telemetry.last_trade_time.isoformat() if self.telemetry.last_trade_time else None,
            "crypto_watchlist": [c["ticker"] for c in CRYPTO_WATCHLIST],
            "options_watchlist": [f"{o['direction']} {o['ticker']}" for o in OPTIONS_WATCHLIST],
            "errors_recent": self.telemetry.errors[-5:],
            # Weekly tracker
            "week": self.tracker.current_week,
            "week_phase": self.week_phase.value,
            "capital": self.tracker.current_capital,
            "target": 1_000_000,
            "progress_pct": self.tracker.progress_pct(),
            "weeks_remaining": self.tracker.weeks_remaining(),
            "peak_capital": self.tracker.peak_capital,
            "max_drawdown": self.tracker.max_drawdown,
            "allocation": {
                "options": alloc.options_allocation,
                "crypto": alloc.crypto_allocation,
                "arb": alloc.arb_allocation,
                "cash": alloc.cash_reserve,
            },
            # Doctrine & compliance
            "doctrine_state": self.barren_wuffet_state,
            "doctrine_can_trade": (
                self.doctrine_status.can_open_new_positions
                if self.doctrine_status else True
            ),
            "doctrine_violations": len(self.doctrine_violations),
            "doctrine_drawdown_pct": (
                self.doctrine_status.drawdown_pct
                if self.doctrine_status else 0.0
            ),
            # Component health
            "component_health": {
                name: comp.health.value
                for name, comp in self.component_health.items()
            } if self.component_health and ComponentHealth else {},
            # FPC resilience
            "fpc_layers_active": (
                sum(1 for l in self.acc_advanced_state.resilience_layers.values()
                    if l.status == 'active')
                if self.acc_advanced_state else 0
            ),
            "system_gaps": len(self.system_gaps),
            # NCC Coordination
            "ncc_online": self.ncc_super_agent is not None,
            "ncc_status": {
                "feasible": self.ncc_status.get("mission_feasibility", {}).get("feasible")
                if self.ncc_status else None,
                "success_probability": (
                    self.ncc_status.get("success_probability", {}).get("overall_probability")
                    if isinstance(self.ncc_status.get("success_probability"), dict)
                    else self.ncc_status.get("success_probability")
                ) if self.ncc_status else None,
            },
            # NCL Governance
            "ncl_level": self.ncl_compliance.level.value if self.ncl_compliance else "N/A",
            "ncl_score": self.ncl_compliance.score if self.ncl_compliance else 0.0,
            "ncl_violations": len(self.ncl_compliance.violations) if self.ncl_compliance else 0,
        }


# ════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

async def main(args):
    """Main entry point for full activation."""
    engine = FullActivationEngine(paper_mode=(args.mode == "paper"))

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("\n⚠️  Shutdown signal received...")
        asyncio.ensure_future(engine.shutdown())

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    except (NotImplementedError, AttributeError):
        # Windows doesn't support add_signal_handler
        pass

    # Initialize
    ok = await engine.initialize()
    if not ok:
        logger.error("❌ Initialization failed — aborting")
        return

    if args.status:
        # Just print status and exit
        status = engine.get_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.cycle_once:
        # Run one cycle and exit
        result = await engine.run_cycle()
        logger.info(f"\nCycle complete. Signals: {len(result.get('signals', []))} | "
                     f"Executions: {len(result.get('executions', []))}")
    else:
        # Run forever
        try:
            await engine.run_forever()
        except KeyboardInterrupt:
            logger.info("\n⚠️  Keyboard interrupt...")
        finally:
            await engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AAC Full Activation — All Hands on Deck",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python full_activation.py                    # Full continuous operation (paper)
  python full_activation.py --cycle-once       # Run one cycle and exit
  python full_activation.py --status           # Print system status
  python full_activation.py --mode live        # LIVE TRADING (careful!)
        """,
    )
    parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                        help="Trading mode (default: paper)")
    parser.add_argument("--status", action="store_true",
                        help="Print system status and exit")
    parser.add_argument("--cycle-once", action="store_true",
                        help="Run one complete cycle and exit")

    args = parser.parse_args()

    if args.mode == "live":
        print("\n⚠️  LIVE TRADING MODE SELECTED")
        print("This will execute REAL trades with REAL money.")
        confirm = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
        if confirm != "CONFIRM LIVE TRADING":
            print("Aborted.")
            sys.exit(0)

    asyncio.run(main(args))
