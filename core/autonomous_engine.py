#!/usr/bin/env python3
"""
AAC Autonomous Engine — BARREN WUFFET Self-Running Core
========================================================

This is the brain. It runs 24/7, self-organizes, self-monitors, self-heals.
No hand-holding. When it doesn't know the next step, it looks inward.
When it can't find internal answers, it searches externally.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                 AUTONOMOUS ENGINE                    │
    │                                                     │
    │  ┌─────────┐  ┌──────────┐  ┌────────────────────┐ │
    │  │Heartbeat│  │ Scheduler│  │  Doctrine Enforcer │ │
    │  │ (5s)    │  │ (cron)   │  │  (state machine)   │ │
    │  └────┬────┘  └────┬─────┘  └────────┬───────────┘ │
    │       │            │                  │             │
    │       ▼            ▼                  ▼             │
    │  ┌──────────────────────────────────────────────┐   │
    │  │           MISSION LOOP                       │   │
    │  │  sense → analyze → decide → act → reconcile  │   │
    │  └──────────────────────┬───────────────────────┘   │
    │                         │                           │
    │       ┌─────────────────┼──────────────────┐        │
    │       ▼                 ▼                  ▼        │
    │  ┌─────────┐    ┌──────────────┐   ┌───────────┐   │
    │  │Self-Check│    │ Gap Analyzer │   │ Self-Heal │   │
    │  │Validator │    │ (intro-      │   │ Recovery  │   │
    │  │         │    │  spection)   │   │           │   │
    │  └─────────┘    └──────────────┘   └───────────┘   │
    │                                                     │
    └─────────────────────────────────────────────────────┘

Protocol:
    1. HEARTBEAT: Every 5s, prove we're alive
    2. SENSE:     Fetch live market data from all connected sources
    3. ANALYZE:   Run strategies, generate signals, aggregate
    4. DECIDE:    Check doctrine compliance, risk limits, consensus
    5. ACT:       Execute paper trades (or live if authorized)
    6. RECONCILE: Verify positions match reality
    7. INTROSPECT: Scan self for gaps, broken paths, stale data
    8. HEAL:      Fix what can be fixed, log what can't
    9. REPORT:    Status update to OpenClaw/logs every cycle
    10. SCHEDULE: Manage cron-like periodic tasks
"""

import asyncio
import json
import logging
import os
import socket
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

# ── Project setup ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import AuditLogger

logger = logging.getLogger("AutonomousEngine")


# ════════════════════════════════════════════════════════════════════════
# DOCTRINE STATE MACHINE — The guardrails that never sleep
# ════════════════════════════════════════════════════════════════════════

class DoctrineState(Enum):
    """DoctrineState class."""
    NORMAL = "NORMAL"         # Full operations
    CAUTION = "CAUTION"       # Drawdown 5-10%, reduce risk
    SAFE_MODE = "SAFE_MODE"   # Drawdown 10-15%, stop new positions
    HALT = "HALT"             # Daily loss > 2%, full stop


@dataclass
class DoctrineStatus:
    """DoctrineStatus class."""
    state: DoctrineState = DoctrineState.NORMAL
    daily_pnl: float = 0.0
    drawdown_pct: float = 0.0
    total_equity: float = 0.0
    open_positions: int = 0
    last_transition: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transition_reason: str = "initial"

    def evaluate(self, pnl: float, equity: float) -> DoctrineState:
        """Evaluate and transition doctrine state based on live metrics.

        pnl: cumulative profit/loss from starting equity (negative = loss)
        equity: current account equity
        Drawdown and daily-loss are computed relative to starting equity
        (equity - pnl).
        """
        self.daily_pnl = pnl
        self.total_equity = equity
        starting_equity = equity - pnl  # reconstruct starting capital
        if starting_equity > 0:
            self.drawdown_pct = abs(min(0, pnl)) / starting_equity
        daily_loss_pct = (abs(pnl) / starting_equity) if (starting_equity > 0 and pnl < 0) else 0.0
        old = self.state
        # Priority: extreme drawdown -> safe_mode -> caution -> daily-loss halt -> normal
        if self.drawdown_pct > 0.15:
            self.state = DoctrineState.HALT
            self.transition_reason = f"extreme_drawdown={self.drawdown_pct:.1%}"
        elif self.drawdown_pct > 0.10:
            self.state = DoctrineState.SAFE_MODE
            self.transition_reason = f"drawdown={self.drawdown_pct:.1%}"
        elif self.drawdown_pct > 0.05:
            self.state = DoctrineState.CAUTION
            self.transition_reason = f"drawdown={self.drawdown_pct:.1%}"
        elif daily_loss_pct > 0.02:
            self.state = DoctrineState.HALT
            self.transition_reason = f"daily_loss={daily_loss_pct:.1%}"
        else:
            self.state = DoctrineState.NORMAL
            self.transition_reason = "nominal"
        if self.state != old:
            self.last_transition = datetime.now(timezone.utc)
            logger.warning(f"DOCTRINE TRANSITION: {old.value} → {self.state.value} | {self.transition_reason}")
            self._transition_callback = None  # Will be set externally
        return self.state

    def set_transition_callback(self, callback):
        """Set async callback to fire on state transitions."""
        self._transition_callback = callback

    @property
    def can_open_new_positions(self) -> bool:
        """Can open new positions."""
        return self.state in (DoctrineState.NORMAL, DoctrineState.CAUTION)

    @property
    def should_reduce_risk(self) -> bool:
        """Should reduce risk."""
        return self.state in (DoctrineState.CAUTION, DoctrineState.SAFE_MODE)

    @property
    def is_halted(self) -> bool:
        """Is halted."""
        return self.state == DoctrineState.HALT


# ════════════════════════════════════════════════════════════════════════
# SYSTEM HEALTH — Track what's alive, what's dead, what's degraded
# ════════════════════════════════════════════════════════════════════════

class ComponentHealth(Enum):
    """ComponentHealth class."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


@dataclass
class ComponentStatus:
    """ComponentStatus class."""
    name: str
    health: ComponentHealth = ComponentHealth.UNKNOWN
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    error_count: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    latency_ms: float = 0.0

    def record_success(self, latency_ms: float = 0.0):
        """Record success."""
        now = datetime.now(timezone.utc)
        self.health = ComponentHealth.HEALTHY
        self.last_check = now
        self.last_success = now
        self.consecutive_failures = 0
        self.latency_ms = latency_ms

    def record_failure(self, error: str):
        """Record failure."""
        self.last_check = datetime.now(timezone.utc)
        self.error_count += 1
        self.consecutive_failures += 1
        self.last_error = error
        if self.consecutive_failures >= 3:
            self.health = ComponentHealth.DOWN
        else:
            self.health = ComponentHealth.DEGRADED


# ════════════════════════════════════════════════════════════════════════
# SIGNAL — What strategies produce, what the engine consumes
# ════════════════════════════════════════════════════════════════════════

class SignalAction(Enum):
    """SignalAction class."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """TradingSignal class."""
    strategy_name: str
    symbol: str
    action: SignalAction
    confidence: float  # 0.0 - 1.0
    price: float
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedSignal:
    """AggregatedSignal class."""
    symbol: str
    action: SignalAction
    consensus_score: float  # -1.0 (all sell) to +1.0 (all buy)
    contributing_signals: List[TradingSignal] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ════════════════════════════════════════════════════════════════════════
# SCHEDULED TASK — Cron-like system for periodic work
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ScheduledTask:
    """ScheduledTask class."""
    name: str
    interval_seconds: float
    callback: Callable[..., Coroutine]
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True
    run_count: int = 0
    error_count: int = 0
    last_error: str = ""
    critical: bool = False  # if True, failure halts the engine

    def is_due(self) -> bool:
        """Is due."""
        if not self.enabled:
            return False
        if self.next_run is None:
            return True
        return datetime.now(timezone.utc) >= self.next_run

    def mark_run(self):
        """Mark run."""
        now = datetime.now(timezone.utc)
        self.last_run = now
        self.next_run = now + timedelta(seconds=self.interval_seconds)
        self.run_count += 1


# ════════════════════════════════════════════════════════════════════════
# GAP RECORD — What introspection finds
# ════════════════════════════════════════════════════════════════════════

class GapSeverity(Enum):
    """GapSeverity class."""
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class SystemGap:
    """SystemGap class."""
    gap_id: str
    component: str
    description: str
    severity: GapSeverity
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolution: str = ""
    auto_fixable: bool = False


# ════════════════════════════════════════════════════════════════════════
# THE ENGINE — The autonomous core
# ════════════════════════════════════════════════════════════════════════

class AutonomousEngine:
    """
    The self-running, self-organizing, self-healing core of AAC.

    Runs a continuous loop:
      sense → analyze → decide → act → reconcile → introspect → report
    
    When stuck internally, searches for answers in logs/state.
    When stuck externally, can fetch web resources for solutions.
    
    Only escalates to human for CRITICAL_ONLY situations:
      - Doctrine HALT requiring manual override
      - Live trading authorization
      - Unrecoverable system failures
    """

    CYCLE_INTERVAL = 60.0       # seconds between full cycles
    HEARTBEAT_INTERVAL = 5.0    # seconds between heartbeats
    INTROSPECT_INTERVAL = 300.0 # seconds between deep self-checks
    REPORT_INTERVAL = 3600.0    # seconds between full reports (1hr)
    MAX_HISTORY = 1000          # keep last N cycle results

    def __init__(self):
        self.config = get_config()
        self.audit = AuditLogger()
        self.doctrine = DoctrineStatus()
        self.components: Dict[str, ComponentStatus] = {}
        self.gaps: List[SystemGap] = []
        self.signals_history: deque = deque(maxlen=self.MAX_HISTORY)
        self.cycle_history: deque = deque(maxlen=self.MAX_HISTORY)
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.start_time: Optional[datetime] = None
        self.cycle_count = 0
        self.error_count = 0
        self.last_report: Optional[datetime] = None
        self._market_data: Dict[str, MarketDataSnapshot] = {}
        self._positions: Dict[str, Any] = {}
        self._daily_trades: List[Dict] = []

        # Lazy-loaded subsystems
        self._coingecko = None
        self._fib_calculator = None
        self._signal_generator = None
        self._execution_engine = None
        self._accounting_db = None
        self._data_aggregator = None
        self._openclaw = None
        self._recommendation_engine = None
        self._intelligence_model = None
        self._connectors: Dict[str, Any] = {}

        # Register core components for health tracking
        for name in ["coingecko", "ndax", "ibkr", "moomoo", "execution_engine",
                      "accounting_db", "doctrine", "scheduler", "openclaw"]:
            self.components[name] = ComponentStatus(name=name)

        self._register_default_tasks()

    # ── Subsystem Initialization (lazy, resilient) ────────────────────

    async def _init_coingecko(self) -> bool:
        try:
            from shared.data_sources import CoinGeckoClient
            self._coingecko = CoinGeckoClient()
            await self._coingecko.connect()
            tick = await self._coingecko.get_price("bitcoin")
            if tick and tick.price > 0:
                self.components["coingecko"].record_success()
                logger.info(f"CoinGecko LIVE: BTC=${tick.price:,.2f}")
                return True
            self.components["coingecko"].record_failure("No price data")
            return False
        except Exception as e:
            self.components["coingecko"].record_failure(str(e))
            logger.error(f"CoinGecko init failed: {e}")
            return False

    async def _init_ndax(self) -> bool:
        try:
            from TradingExecution.exchange_connectors.ndax_connector import NDAXConnector
            conn = NDAXConnector(testnet=False)
            result = await conn.connect()
            if result:
                self._connectors["ndax"] = conn
                self.components["ndax"].record_success()
                logger.info("NDAX LIVE")
                return True
            self.components["ndax"].record_failure("connect() returned False")
            return False
        except Exception as e:
            self.components["ndax"].record_failure(str(e))
            logger.warning(f"NDAX init failed (non-critical): {e}")
            return False

    async def _init_ibkr(self) -> bool:
        try:
            ibkr_port = int(os.environ.get("IBKR_PORT", "7497"))
            if not _port_open("127.0.0.1", ibkr_port):
                self.components["ibkr"].record_failure(f"TWS not running on port {ibkr_port}")
                return False
            from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
            conn = IBKRConnector()
            result = await conn.connect()
            if result:
                self._connectors["ibkr"] = conn
                self.components["ibkr"].record_success()
                logger.info("IBKR LIVE")
                return True
            self.components["ibkr"].record_failure("connect() returned False")
            return False
        except Exception as e:
            self.components["ibkr"].record_failure(str(e))
            logger.warning(f"IBKR init failed (non-critical): {e}")
            return False

    async def _init_moomoo(self) -> bool:
        try:
            if not _port_open("127.0.0.1", 11111):
                self.components["moomoo"].record_failure("OpenD not running on port 11111")
                return False
            from TradingExecution.exchange_connectors.moomoo_connector import MoomooConnector
            conn = MoomooConnector()
            result = await conn.connect()
            if result:
                self._connectors["moomoo"] = conn
                self.components["moomoo"].record_success()
                logger.info("Moomoo LIVE")
                return True
            self.components["moomoo"].record_failure("connect() returned False")
            return False
        except Exception as e:
            self.components["moomoo"].record_failure(str(e))
            logger.warning(f"Moomoo init failed (non-critical): {e}")
            return False

    async def _init_execution_engine(self) -> bool:
        try:
            from TradingExecution.execution_engine import ExecutionEngine
            self._execution_engine = ExecutionEngine()
            self.components["execution_engine"].record_success()
            return True
        except Exception as e:
            self.components["execution_engine"].record_failure(str(e))
            logger.error(f"ExecutionEngine init failed: {e}")
            return False

    async def _init_accounting(self) -> bool:
        try:
            from CentralAccounting.database import AccountingDatabase
            self._accounting_db = AccountingDatabase()
            self._accounting_db.initialize()
            self.components["accounting_db"].record_success()
            return True
        except Exception as e:
            self.components["accounting_db"].record_failure(str(e))
            logger.error(f"AccountingDB init failed: {e}")
            return False

    async def _init_strategies(self) -> bool:
        try:
            from strategies.golden_ratio_finance import (
                FibonacciCalculator, fractal_compression_index,
                phase_conjugation_score,
            )
            self._fib_calculator = FibonacciCalculator()

            # Import signal generator from pipeline_runner
            from pipeline_runner import FibSignalGenerator
            self._signal_generator = FibSignalGenerator()

            logger.info("Strategies initialized: FibSignalGenerator + FibonacciCalculator")
            return True
        except Exception as e:
            logger.error(f"Strategy init failed: {e}")
            return False

    async def _init_recommendation_engine(self) -> bool:
        """Initialize the daily recommendation engine."""
        try:
            from core.daily_recommendation_engine import DailyRecommendationEngine
            self._recommendation_engine = DailyRecommendationEngine()
            logger.info("DailyRecommendationEngine initialized")
            return True
        except Exception as e:
            logger.warning(f"DailyRecommendationEngine init failed (non-critical): {e}")
            return False

    async def _init_intelligence_model(self) -> bool:
        """Initialize the 24/7 Market Intelligence Model (sentiment + event ingest)."""
        try:
            from strategies.market_intelligence_model import MarketIntelligenceModel
            balance = float(os.getenv("ACCOUNT_BALANCE_USD", "920"))
            doctrine_mult = 0.5 if self.doctrine.is_caution else 1.0
            self._intelligence_model = MarketIntelligenceModel(
                available_capital=balance,
                doctrine_risk_mult=doctrine_mult,
            )
            # Kick off the internal loop as a background task
            asyncio.create_task(self._intelligence_model.run())
            logger.info("MarketIntelligenceModel LIVE — 24/7 sentiment engine running")
            return True
        except Exception as exc:
            logger.warning("MarketIntelligenceModel init failed (non-critical): %s", exc)
            return False

    async def _init_openclaw(self) -> bool:
        """Connect to OpenClaw Gateway for multi-channel messaging."""
        try:
            from integrations.openclaw_gateway_bridge import (
                OpenClawGatewayBridge, OpenClawChannel, OpenClawCronJob,
            )
            gateway_url = os.getenv("OPENCLAW_GATEWAY_URL", "ws://127.0.0.1:18789")
            self._openclaw = OpenClawGatewayBridge(gateway_url=gateway_url)
            connected = await self._openclaw.connect()
            if connected:
                self.components["openclaw"].record_success()
                logger.info(f"OpenClaw Gateway LIVE at {gateway_url}")

                # Register daily cron jobs per SOUL.md doctrine
                morning_briefing = OpenClawCronJob(
                    job_id="morning_briefing",
                    name="Morning Market Briefing",
                    schedule="0 7 * * *",  # 07:00 MT daily
                    message="Generate morning market briefing with overnight analysis",
                    session_key="main",
                )
                evening_recap = OpenClawCronJob(
                    job_id="evening_recap",
                    name="Evening Performance Recap",
                    schedule="0 18 * * *",  # 18:00 MT daily
                    message="Generate evening performance recap and P&L summary",
                    session_key="main",
                )
                await self._openclaw.register_cron_job(morning_briefing)
                await self._openclaw.register_cron_job(evening_recap)
                return True
            else:
                self.components["openclaw"].record_failure("connect() returned False")
                return False
        except Exception as e:
            self.components["openclaw"].record_failure(str(e))
            logger.warning(f"OpenClaw init failed (non-critical): {e}")
            return False

    # ── Task Registration ─────────────────────────────────────────────

    def _register_default_tasks(self):
        """Register all periodic tasks that make the system autonomous."""
        self.register_task("market_scan", 60.0, self._task_market_scan, critical=True)
        self.register_task("strategy_signals", 60.0, self._task_generate_signals)
        self.register_task("position_reconcile", 300.0, self._task_reconcile_positions)
        self.register_task("connector_health", 120.0, self._task_check_connectors)
        self.register_task("introspection", self.INTROSPECT_INTERVAL, self._task_introspect)
        self.register_task("status_report", self.REPORT_INTERVAL, self._task_status_report)
        self.register_task("daily_pnl_reset", 86400.0, self._task_daily_reset)
        self.register_task("gap_analysis", 600.0, self._task_gap_analysis)
        self.register_task("daily_brief", 86400.0, self._task_daily_brief)
        self.register_task("intelligence_cycle", 3600.0, self._task_intelligence_cycle)
        self.register_task("rocket_ship_brief", 86400.0, self._task_rocket_ship_brief)

    def register_task(self, name: str, interval: float, callback, critical: bool = False):
        """Register task."""
        self.tasks[name] = ScheduledTask(
            name=name, interval_seconds=interval,
            callback=callback, critical=critical,
        )

    # ══════════════════════════════════════════════════════════════════
    # MAIN LOOP — The beating heart
    # ══════════════════════════════════════════════════════════════════

    async def start(self):
        """Boot up and run forever."""
        logger.info("=" * 70)
        logger.info("  BARREN WUFFET AUTONOMOUS ENGINE — INITIALIZING")
        logger.info("  Codename: AZ SUPREME | Protocol: SELF-ORGANIZING")
        logger.info("=" * 70)

        self.running = True
        self.start_time = datetime.now(timezone.utc)

        # Phase 1: Initialize all subsystems (resilient — failures logged, not fatal)
        await self._bootstrap()

        # Phase 2: Run the heartbeat in parallel with the mission loop
        logger.info("AUTONOMOUS ENGINE ONLINE — entering mission loop")
        await asyncio.gather(
            self._heartbeat_loop(),
            self._mission_loop(),
            self._scheduler_loop(),
        )

    async def stop(self):
        """Graceful shutdown."""
        logger.info("AUTONOMOUS ENGINE — shutting down gracefully")
        self.running = False
        # Disconnect all connectors
        for name, conn in self._connectors.items():
            try:
                if hasattr(conn, 'disconnect'):
                    await conn.disconnect()
                    logger.info(f"Disconnected: {name}")
            except Exception as e:
                logger.warning(f"Error disconnecting {name}: {e}")
        if self._coingecko and hasattr(self._coingecko, 'disconnect'):
            try:
                await self._coingecko.disconnect()
            except Exception as e:
                logger.exception("Unexpected error: %s", e)

    async def _bootstrap(self):
        """Initialize everything. Failures are logged but don't block startup."""
        logger.info("Phase 1: Bootstrapping subsystems...")

        results = {}
        # Init in dependency order
        for name, init_fn in [
            ("strategies", self._init_strategies),
            ("coingecko", self._init_coingecko),
            ("execution_engine", self._init_execution_engine),
            ("accounting", self._init_accounting),
            ("ndax", self._init_ndax),
            ("ibkr", self._init_ibkr),
            ("moomoo", self._init_moomoo),
            ("openclaw", self._init_openclaw),
            ("recommendation_engine", self._init_recommendation_engine),
            ("intelligence_model", self._init_intelligence_model),
        ]:
            try:
                results[name] = await init_fn()
            except Exception as e:
                results[name] = False
                logger.error(f"Bootstrap {name} FAILED: {e}")

        live = sum(1 for v in results.values() if v)
        total = len(results)
        logger.info(f"Bootstrap complete: {live}/{total} subsystems online")
        for name, ok in results.items():
            status = "LIVE" if ok else "OFFLINE"
            logger.info(f"  {name:20s} {status}")

        # Must have at least CoinGecko + strategies to run
        if not results.get("strategies"):
            logger.critical("CRITICAL: Strategy engine failed to init — cannot trade")
        if not results.get("coingecko"):
            logger.critical("CRITICAL: CoinGecko failed — no market data source")

    # ── Heartbeat ─────────────────────────────────────────────────────

    async def _heartbeat_loop(self):
        """Prove we're alive. Log heartbeat every HEARTBEAT_INTERVAL seconds."""
        while self.running:
            try:
                uptime = datetime.now(timezone.utc) - self.start_time
                hours = uptime.total_seconds() / 3600
                healthy = sum(1 for c in self.components.values()
                              if c.health == ComponentHealth.HEALTHY)
                total = len(self.components)
                logger.debug(
                    f"HEARTBEAT | uptime={hours:.1f}h | cycles={self.cycle_count} | "
                    f"doctrine={self.doctrine.state.value} | "
                    f"components={healthy}/{total} healthy | "
                    f"errors={self.error_count}"
                )
            except Exception as e:
                logger.exception("Unexpected error: %s", e)
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)

    # ── Scheduler ─────────────────────────────────────────────────────

    async def _scheduler_loop(self):
        """Run scheduled tasks when they're due."""
        self.components["scheduler"].record_success()
        while self.running:
            for name, task in self.tasks.items():
                if task.is_due():
                    try:
                        await task.callback()
                        task.mark_run()
                        task.last_error = ""
                    except Exception as e:
                        task.error_count += 1
                        task.last_error = str(e)
                        task.mark_run()  # still mark as run to avoid tight retry loops
                        logger.error(f"Scheduled task '{name}' failed: {e}")
                        if task.critical:
                            logger.critical(f"CRITICAL task '{name}' failed — "
                                            f"errors={task.error_count}")
            await asyncio.sleep(1.0)

    # ══════════════════════════════════════════════════════════════════
    # MISSION LOOP — sense → analyze → decide → act → reconcile
    # ══════════════════════════════════════════════════════════════════

    async def _mission_loop(self):
        """The core trading cycle. Runs every CYCLE_INTERVAL seconds."""
        # Wait for first data to arrive
        await asyncio.sleep(5.0)

        while self.running:
            cycle_start = time.monotonic()
            cycle_result = {
                "cycle": self.cycle_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phases": {},
                "error": None,
            }

            try:
                # ── SENSE ─────────────────────────────────────────
                market_data = await self._sense()
                cycle_result["phases"]["sense"] = {
                    "symbols": list(market_data.keys()),
                    "ok": len(market_data) > 0,
                }

                if not market_data:
                    cycle_result["phases"]["sense"]["error"] = "No market data"
                    self.cycle_history.append(cycle_result)
                    self.cycle_count += 1
                    await asyncio.sleep(self.CYCLE_INTERVAL)
                    continue

                # ── ANALYZE ───────────────────────────────────────
                signals = await self._analyze(market_data)
                cycle_result["phases"]["analyze"] = {
                    "signals": len(signals),
                    "buys": sum(1 for s in signals if s.action == SignalAction.BUY),
                    "sells": sum(1 for s in signals if s.action == SignalAction.SELL),
                    "holds": sum(1 for s in signals if s.action == SignalAction.HOLD),
                }

                # ── DECIDE ────────────────────────────────────────
                approved = await self._decide(signals)
                cycle_result["phases"]["decide"] = {
                    "approved": len(approved),
                    "doctrine_state": self.doctrine.state.value,
                }

                # ── ACT ───────────────────────────────────────────
                if approved:
                    executions = await self._act(approved)
                    cycle_result["phases"]["act"] = {
                        "executed": len(executions),
                        "details": executions,
                    }
                else:
                    cycle_result["phases"]["act"] = {"executed": 0, "reason": "no approved signals"}

                # ── RECONCILE ─────────────────────────────────────
                recon = await self._reconcile()
                cycle_result["phases"]["reconcile"] = recon

            except Exception as e:
                self.error_count += 1
                cycle_result["error"] = str(e)
                logger.error(f"Cycle {self.cycle_count} error: {e}")
                logger.debug(traceback.format_exc())

            elapsed = time.monotonic() - cycle_start
            cycle_result["elapsed_ms"] = round(elapsed * 1000, 2)
            self.cycle_history.append(cycle_result)
            self.cycle_count += 1

            # Log cycle summary
            phases = cycle_result.get("phases", {})
            sense_ok = phases.get("sense", {}).get("ok", False)
            n_signals = phases.get("analyze", {}).get("signals", 0)
            n_approved = phases.get("decide", {}).get("approved", 0)
            n_executed = phases.get("act", {}).get("executed", 0)

            if sense_ok:
                logger.info(
                    f"CYCLE {self.cycle_count:04d} | "
                    f"{elapsed*1000:.0f}ms | "
                    f"signals={n_signals} approved={n_approved} executed={n_executed} | "
                    f"doctrine={self.doctrine.state.value}"
                )
            else:
                logger.warning(f"CYCLE {self.cycle_count:04d} | NO DATA | {elapsed*1000:.0f}ms")

            await asyncio.sleep(self.CYCLE_INTERVAL)

    # ══════════════════════════════════════════════════════════════════
    # PHASE: SENSE — Gather market data from all live sources
    # ══════════════════════════════════════════════════════════════════

    async def _sense(self) -> Dict[str, "MarketDataSnapshot"]:
        """Fetch prices from all available data sources."""
        snapshots = {}

        # CoinGecko (always available, free)
        if self._coingecko:
            try:
                from shared.data_sources import MarketTick
                coins = ["bitcoin", "ethereum"]
                for coin in coins:
                    tick = await self._coingecko.get_price(coin)
                    if tick and tick.price > 0:
                        symbol = "BTC/USD" if coin == "bitcoin" else "ETH/USD"
                        snapshots[symbol] = MarketDataSnapshot(
                            symbol=symbol, price=tick.price,
                            bid=tick.bid, ask=tick.ask,
                            volume_24h=tick.volume_24h,
                            source="coingecko",
                            timestamp=datetime.now(timezone.utc),
                        )
                self.components["coingecko"].record_success()
            except Exception as e:
                self.components["coingecko"].record_failure(str(e))

        # NDAX (if connected)
        if "ndax" in self._connectors:
            try:
                conn = self._connectors["ndax"]
                for pair in ["BTC/CAD", "ETH/CAD"]:
                    try:
                        ticker = await conn.get_ticker(pair)
                        if ticker and ticker.last > 0:
                            snapshots[pair] = MarketDataSnapshot(
                                symbol=pair, price=ticker.last,
                                bid=ticker.bid, ask=ticker.ask,
                                volume_24h=getattr(ticker, 'volume_24h', 0),
                                source="ndax",
                                timestamp=datetime.now(timezone.utc),
                            )
                    except Exception as e:
                        logger.exception("Unexpected error: %s", e)
                self.components["ndax"].record_success()
            except Exception as e:
                self.components["ndax"].record_failure(str(e))

        self._market_data = snapshots
        return snapshots

    # ══════════════════════════════════════════════════════════════════
    # PHASE: ANALYZE — Run strategies on market data, generate signals
    # ══════════════════════════════════════════════════════════════════

    async def _analyze(self, market_data: Dict[str, "MarketDataSnapshot"]) -> List[TradingSignal]:
        """Run all active strategies and collect signals."""
        signals = []

        # Fibonacci / Golden Ratio strategy (THE working one)
        if self._signal_generator and "BTC/USD" in market_data:
            try:
                btc = market_data["BTC/USD"]
                # Fetch 30-day history for Fibonacci calculation
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    url = ("https://api.coingecko.com/api/v3/coins/bitcoin/"
                           "market_chart?vs_currency=usd&days=30&interval=daily")
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            prices = [p[1] for p in data.get("prices", [])]
                        else:
                            prices = []

                if len(prices) >= 10:
                    analysis = self._signal_generator.analyze(
                        symbol="BTC/USD",
                        current_price=btc.price,
                        high_30d=max(prices),
                        low_30d=min(prices),
                        prices_30d=prices,
                        change_24h=0.0,
                    )
                    action_str = analysis.get("signal", "HOLD")
                    action = (SignalAction.BUY if action_str == "BUY"
                              else SignalAction.SELL if action_str == "SELL"
                              else SignalAction.HOLD)
                    signals.append(TradingSignal(
                        strategy_name="fibonacci_golden_ratio",
                        symbol="BTC/USD",
                        action=action,
                        confidence=analysis.get("confidence", 0.5),
                        price=btc.price,
                        reason=analysis.get("reason", "Fibonacci analysis"),
                        metadata=analysis,
                    ))
            except Exception as e:
                logger.warning(f"Fibonacci strategy error: {e}")

        # ETH analysis if available
        if self._signal_generator and "ETH/USD" in market_data:
            try:
                eth = market_data["ETH/USD"]
                async with aiohttp.ClientSession() as session:
                    url = ("https://api.coingecko.com/api/v3/coins/ethereum/"
                           "market_chart?vs_currency=usd&days=30&interval=daily")
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            prices = [p[1] for p in data.get("prices", [])]
                        else:
                            prices = []

                if len(prices) >= 10:
                    analysis = self._signal_generator.analyze(
                        symbol="ETH/USD",
                        current_price=eth.price,
                        high_30d=max(prices),
                        low_30d=min(prices),
                        prices_30d=prices,
                        change_24h=0.0,
                    )
                    action_str = analysis.get("signal", "HOLD")
                    action = (SignalAction.BUY if action_str == "BUY"
                              else SignalAction.SELL if action_str == "SELL"
                              else SignalAction.HOLD)
                    signals.append(TradingSignal(
                        strategy_name="fibonacci_golden_ratio",
                        symbol="ETH/USD",
                        action=action,
                        confidence=analysis.get("confidence", 0.5),
                        price=eth.price,
                        reason=analysis.get("reason", "Fibonacci analysis"),
                        metadata=analysis,
                    ))
            except Exception as e:
                logger.warning(f"ETH strategy error: {e}")

        # Daily Recommendation Engine signals (RSI, MACD, MA, Volume, Bollinger, UW)
        if self._recommendation_engine:
            try:
                rec_signals = await self._recommendation_engine.generate_signals()
                for rsig in rec_signals:
                    action_map = {
                        "BUY": SignalAction.BUY,
                        "SELL": SignalAction.SELL,
                        "HOLD": SignalAction.HOLD,
                    }
                    action = action_map.get(rsig.direction.value, SignalAction.HOLD)
                    price = rsig.metadata.get("current_price", 0.0)
                    if not price and rsig.symbol in market_data:
                        price = market_data[rsig.symbol].price
                    signals.append(TradingSignal(
                        strategy_name=rsig.generator,
                        symbol=rsig.symbol,
                        action=action,
                        confidence=rsig.confidence,
                        price=price,
                        reason=rsig.reason,
                        metadata=rsig.metadata,
                    ))
            except Exception as e:
                logger.warning(f"Recommendation engine signals error: {e}")

        # Market Intelligence Model — structured regime × sentiment signals
        if self._intelligence_model:
            try:
                intel_recs = self._intelligence_model.get_recommendations()
                for rec in intel_recs:
                    if rec.entry_urgency in ("IMMEDIATE", "NEXT_OPEN"):
                        signals.append(TradingSignal(
                            strategy_name="market_intelligence_model",
                            symbol=rec.ticker,
                            action=SignalAction.SELL,
                            confidence=rec.composite_score / 100.0,
                            price=0.0,
                            reason=(
                                f"[{rec.conviction_tier}] {rec.entry_urgency} | "
                                f"regime={rec.regime} sentiment={rec.sector_sentiment:+.0f}"
                            ),
                            metadata=rec.to_dict(),
                        ))
            except Exception as exc:
                logger.warning("Intelligence model signal injection error: %s", exc)

        # Store signals
        for sig in signals:
            self.signals_history.append(sig)

        return signals

    # ══════════════════════════════════════════════════════════════════
    # PHASE: DECIDE — Doctrine + risk check on signals
    # ══════════════════════════════════════════════════════════════════

    async def _decide(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals through doctrine and risk management."""
        if self.doctrine.is_halted:
            logger.warning("DOCTRINE HALT — all signals rejected")
            return []

        approved = []
        for sig in signals:
            # Skip HOLD signals
            if sig.action == SignalAction.HOLD:
                continue

            # Confidence threshold
            if sig.confidence < 0.6:
                logger.debug(f"Signal rejected (low confidence {sig.confidence:.2f}): "
                             f"{sig.strategy_name} {sig.action.value} {sig.symbol}")
                continue

            # Doctrine: can we open new positions?
            if not self.doctrine.can_open_new_positions:
                logger.info(f"Signal rejected (doctrine {self.doctrine.state.value}): "
                            f"{sig.action.value} {sig.symbol}")
                continue

            # Risk: daily trade limit (paper trading: 200 max)
            if len(self._daily_trades) >= 200:
                logger.info("Daily trade limit reached (200)")
                break

            approved.append(sig)

        return approved

    # ══════════════════════════════════════════════════════════════════
    # PHASE: ACT — Execute approved signals
    # ══════════════════════════════════════════════════════════════════

    async def _act(self, signals: List[TradingSignal]) -> List[Dict]:
        """Execute approved signals. Paper trading only unless doctrine permits live."""
        executions = []

        for sig in signals:
            try:
                # Paper execution via the working pipeline pattern
                trade_record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "strategy": sig.strategy_name,
                    "symbol": sig.symbol,
                    "action": sig.action.value,
                    "price": sig.price,
                    "confidence": sig.confidence,
                    "reason": sig.reason,
                    "mode": "paper",
                    "status": "executed",
                }

                # Record to accounting if available
                if self._accounting_db:
                    try:
                        self._accounting_db.record_transaction(
                            account_id=1,  # paper_trading account
                            transaction_type="trade",
                            asset=sig.symbol,
                            quantity=0.001 if "BTC" in sig.symbol else 0.01,
                            symbol=sig.symbol,
                            price=sig.price,
                            side=sig.action.value.lower(),
                            fee=0.0,
                            notes=f"Auto: {sig.strategy_name} | {sig.reason}",
                        )
                        trade_record["recorded"] = True
                    except Exception as e:
                        trade_record["recorded"] = False
                        trade_record["record_error"] = str(e)

                self._daily_trades.append(trade_record)
                executions.append(trade_record)

                logger.info(
                    f"TRADE | {sig.action.value} {sig.symbol} @ ${sig.price:,.2f} | "
                    f"strategy={sig.strategy_name} conf={sig.confidence:.2f} | PAPER"
                )

            except Exception as e:
                logger.error(f"Execution failed for {sig.symbol}: {e}")
                executions.append({"symbol": sig.symbol, "status": "failed", "error": str(e)})

        return executions

    # ══════════════════════════════════════════════════════════════════
    # PHASE: RECONCILE — Verify positions, update doctrine state
    # ══════════════════════════════════════════════════════════════════

    async def _reconcile(self) -> Dict:
        """Reconcile positions and update doctrine state."""
        result = {"positions": 0, "daily_pnl": 0.0, "errors": []}

        try:
            # Calculate daily P&L from executed trades
            daily_pnl = 0.0
            for trade in self._daily_trades:
                action = trade.get("action", trade.get("side", ""))
                price = trade.get("price", 0.0)
                quantity = trade.get("quantity", trade.get("size", 0.0))
                fill_price = trade.get("fill_price", price)
                if action in ("sell", "SELL"):
                    daily_pnl += (fill_price - price) * quantity
                elif action in ("buy", "BUY"):
                    daily_pnl -= (fill_price - price) * quantity

            result["positions"] = len(self._daily_trades)
            result["daily_pnl"] = round(daily_pnl, 2)

            # Update doctrine state
            equity = 100_000.0 + daily_pnl  # Paper trading starts at $100K
            self.doctrine.evaluate(daily_pnl, equity)
            result["doctrine_state"] = self.doctrine.state.value

            self.components["doctrine"].record_success()

        except Exception as e:
            result["errors"].append(str(e))
            self.components["doctrine"].record_failure(str(e))

        return result

    # ══════════════════════════════════════════════════════════════════
    # SCHEDULED TASKS — Periodic autonomous work
    # ══════════════════════════════════════════════════════════════════

    async def _task_market_scan(self):
        """Fetch latest market data (runs every 30s)."""
        await self._sense()

    async def _task_generate_signals(self):
        """Generate fresh signals from market data (runs every 60s)."""
        if self._market_data:
            signals = await self._analyze(self._market_data)
            if signals:
                logger.debug(f"Generated {len(signals)} signals from scheduled scan")

    async def _task_reconcile_positions(self):
        """Reconcile positions with exchanges (runs every 5min)."""
        await self._reconcile()

    async def _task_check_connectors(self):
        """Health-check all exchange connectors (runs every 2min)."""
        # Check CoinGecko
        if self._coingecko:
            try:
                tick = await self._coingecko.get_price("bitcoin")
                if tick and tick.price > 0:
                    self.components["coingecko"].record_success()
                else:
                    self.components["coingecko"].record_failure("No data")
            except Exception as e:
                self.components["coingecko"].record_failure(str(e))

        # Check NDAX connectivity
        if "ndax" in self._connectors:
            try:
                ticker = await self._connectors["ndax"].get_ticker("BTC/CAD")
                if ticker:
                    self.components["ndax"].record_success()
            except Exception as e:
                self.components["ndax"].record_failure(str(e))

        # Check gateway ports
        ibkr_port = int(os.environ.get("IBKR_PORT", "7497"))
        if _port_open("127.0.0.1", ibkr_port):
            if self.components["ibkr"].health == ComponentHealth.DOWN:
                # Try to reconnect
                await self._init_ibkr()
        else:
            self.components["ibkr"].record_failure(f"Port {ibkr_port} closed")

        if _port_open("127.0.0.1", 11111):
            if self.components["moomoo"].health == ComponentHealth.DOWN:
                await self._init_moomoo()
        else:
            self.components["moomoo"].record_failure("Port 11111 closed")

    async def _task_introspect(self):
        """Deep self-analysis: find gaps, broken connections, stale data (every 5min)."""
        logger.info("INTROSPECTION — scanning self for gaps...")

        gaps_found = []

        # Check 1: Are any data sources completely dead?
        for name, comp in self.components.items():
            if comp.health == ComponentHealth.DOWN and comp.consecutive_failures > 5:
                gaps_found.append(SystemGap(
                    gap_id=f"DOWN-{name}",
                    component=name,
                    description=f"{name} has been DOWN for {comp.consecutive_failures} consecutive checks: {comp.last_error}",
                    severity=GapSeverity.HIGH,
                ))

        # Check 2: Are we generating signals?
        recent_signals = [s for s in self.signals_history
                          if (datetime.now(timezone.utc) - s.timestamp).total_seconds() < 600]
        if not recent_signals and self.cycle_count > 10:
            gaps_found.append(SystemGap(
                gap_id="NO-SIGNALS",
                component="strategy_engine",
                description="No signals generated in last 10 minutes",
                severity=GapSeverity.MEDIUM,
            ))

        # Check 3: Error rate too high?
        if self.cycle_count > 0:
            error_rate = self.error_count / self.cycle_count
            if error_rate > 0.3:
                gaps_found.append(SystemGap(
                    gap_id="HIGH-ERROR-RATE",
                    component="mission_loop",
                    description=f"Error rate {error_rate:.0%} exceeds 30% threshold",
                    severity=GapSeverity.CRITICAL,
                ))

        # Check 4: Are scheduled tasks running?
        for name, task in self.tasks.items():
            if task.enabled and task.error_count > 3 and task.last_error:
                gaps_found.append(SystemGap(
                    gap_id=f"TASK-FAIL-{name}",
                    component=f"task:{name}",
                    description=f"Task '{name}' has failed {task.error_count} times: {task.last_error}",
                    severity=GapSeverity.HIGH if task.critical else GapSeverity.MEDIUM,
                ))

        # Check 5: Market data stale?
        for symbol, snap in self._market_data.items():
            age = (datetime.now(timezone.utc) - snap.timestamp).total_seconds()
            if age > 300:
                gaps_found.append(SystemGap(
                    gap_id=f"STALE-{symbol}",
                    component="market_data",
                    description=f"{symbol} data is {age:.0f}s old (>5min stale)",
                    severity=GapSeverity.MEDIUM,
                ))

        # Check 6: Doctrine stuck in bad state too long?
        if self.doctrine.state != DoctrineState.NORMAL:
            duration = (datetime.now(timezone.utc) - self.doctrine.last_transition).total_seconds()
            if duration > 3600:  # 1 hour in non-normal state
                gaps_found.append(SystemGap(
                    gap_id="DOCTRINE-STUCK",
                    component="doctrine",
                    description=f"Doctrine in {self.doctrine.state.value} for {duration/3600:.1f} hours",
                    severity=GapSeverity.HIGH,
                ))

        # Store and log gaps
        self.gaps = gaps_found
        if gaps_found:
            logger.warning(f"INTROSPECTION found {len(gaps_found)} gaps:")
            for gap in gaps_found:
                logger.warning(f"  [{gap.severity.value}] {gap.gap_id}: {gap.description}")
            # Auto-heal what we can
            await self.self_heal()
            # Notify on critical gaps
            critical_gaps = [g for g in gaps_found if g.severity == GapSeverity.CRITICAL and not g.resolved]
            if critical_gaps:
                msg = f"CRITICAL GAPS DETECTED ({len(critical_gaps)}):\n"
                msg += "\n".join(f"  - {g.description}" for g in critical_gaps)
                await self._notify(msg, critical=True)
        else:
            logger.info("INTROSPECTION: all clear — no gaps detected")

    async def _task_gap_analysis(self):
        """Deeper gap analysis — scan codebase for broken imports, stubs (every 10min)."""
        # Check if key modules are importable
        import_checks = {
            "shared.data_sources": "Market data client",
            "strategies.golden_ratio_finance": "Fibonacci strategy",
            "TradingExecution.execution_engine": "Execution engine",
            "CentralAccounting.database": "Accounting database",
            "shared.config_loader": "Configuration",
            "shared.audit_logger": "Audit logging",
        }
        import importlib
        for module_path, label in import_checks.items():
            try:
                importlib.import_module(module_path)
            except Exception as e:
                self.gaps.append(SystemGap(
                    gap_id=f"IMPORT-FAIL-{module_path}",
                    component="imports",
                    description=f"Cannot import {module_path} ({label}): {e}",
                    severity=GapSeverity.CRITICAL,
                    auto_fixable=False,
                ))

    async def _task_status_report(self):
        """Generate comprehensive status report (every 1 hour)."""
        logger.info("=" * 70)
        logger.info("  AUTONOMOUS ENGINE STATUS REPORT")
        logger.info("=" * 70)

        uptime = datetime.now(timezone.utc) - self.start_time
        hours = uptime.total_seconds() / 3600

        logger.info(f"  Uptime:           {hours:.1f} hours")
        logger.info(f"  Cycles completed: {self.cycle_count}")
        logger.info(f"  Total errors:     {self.error_count}")
        logger.info(f"  Error rate:       {(self.error_count/max(1,self.cycle_count))*100:.1f}%")
        logger.info(f"  Doctrine state:   {self.doctrine.state.value}")
        logger.info(f"  Daily trades:     {len(self._daily_trades)}")

        logger.info("\n  Component Health:")
        for name, comp in self.components.items():
            status = comp.health.value
            err = f" ({comp.last_error[:40]})" if comp.last_error else ""
            logger.info(f"    {name:20s} {status:8s} errors={comp.error_count}{err}")

        logger.info("\n  Scheduled Tasks:")
        for name, task in self.tasks.items():
            status = "ON" if task.enabled else "OFF"
            runs = task.run_count
            errs = task.error_count
            logger.info(f"    {name:20s} {status:3s} runs={runs} errors={errs}")

        if self.gaps:
            logger.info(f"\n  Active Gaps: {len(self.gaps)}")
            for gap in self.gaps[:5]:
                logger.info(f"    [{gap.severity.value}] {gap.description[:60]}")

        # Recent signals summary
        recent = list(self.signals_history)[-10:]
        if recent:
            logger.info(f"\n  Last {len(recent)} signals:")
            for sig in recent:
                logger.info(f"    {sig.timestamp.strftime('%H:%M:%S')} "
                            f"{sig.action.value:4s} {sig.symbol} @ ${sig.price:,.2f} "
                            f"conf={sig.confidence:.2f} ({sig.strategy_name})")

        logger.info("=" * 70)

        # Send summary via OpenClaw
        report_summary = (
            f"HOURLY STATUS | uptime={hours:.1f}h | cycles={self.cycle_count} | "
            f"errors={self.error_count} | doctrine={self.doctrine.state.value} | "
            f"trades_today={len(self._daily_trades)} | "
            f"gaps={len(self.gaps)}"
        )
        await self._notify(report_summary)

    async def _task_intelligence_cycle(self):
        """Hourly: run an intraday intelligence cycle (event ingest → sentiment update → NCL push)."""
        if not self._intelligence_model:
            return
        try:
            balance = float(os.getenv("ACCOUNT_BALANCE_USD", str(self._intelligence_model._capital)))
            self._intelligence_model.update_capital(balance)
            doctrine_mult = 0.5 if self.doctrine.is_caution else 1.0
            self._intelligence_model.update_doctrine_mult(doctrine_mult)
            await self._intelligence_model.intraday_cycle()
        except Exception as exc:
            logger.warning("Intelligence cycle task failed: %s", exc)

    async def _task_daily_brief(self):
        """Generate and distribute the daily trade recommendation brief."""
        if not self._recommendation_engine:
            logger.warning("DailyRecommendationEngine not available — skipping daily brief")
            return
        try:
            brief = await self._recommendation_engine.generate_daily_brief()
            logger.info(f"Daily brief generated ({len(brief)} chars)")
            # Send via OpenClaw / Telegram if available
            await self._notify(brief[:2000])  # Truncate for notification channels
        except Exception as e:
            logger.error(f"Daily brief generation failed: {e}")

    async def _task_rocket_ship_brief(self):
        """Run the Rocket Ship morning briefing in a thread and notify via OpenClaw."""
        try:
            loop = asyncio.get_event_loop()
            from strategies.rocket_ship.daily_ops import run_morning_briefing
            result = await loop.run_in_executor(None, run_morning_briefing)
            phase  = result.get("trigger", {}).get("phase", "?")
            green  = result.get("indicators", {}).get("green_count", "?")
            prob   = result.get("trigger", {}).get("ignition_prob", 0)
            moon   = result.get("lunar", {}).get("moon_number", "?")
            days   = result.get("lunar", {}).get("days_to_rocket_start", "?")
            summary = (
                f"ROCKET SHIP BRIEF — Phase={phase} | Green={green}/15 | "
                f"Ignition={prob:.0%} | Moon#{moon} | T-{days}d"
            )
            logger.info(summary)
            await self._notify(summary)
        except Exception as exc:
            logger.error(f"Rocket Ship brief failed: {exc}")

    async def _task_daily_reset(self):
        """Reset daily metrics at midnight UTC (runs every 24h)."""
        logger.info("DAILY RESET — clearing daily trades and P&L")
        self._daily_trades.clear()
        self.doctrine.daily_pnl = 0.0

    # ══════════════════════════════════════════════════════════════════
    # SELF-HEAL — Attempt to fix identified gaps
    # ══════════════════════════════════════════════════════════════════

    async def self_heal(self):
        """Try to automatically fix gaps found during introspection."""
        for gap in self.gaps:
            if gap.resolved:
                continue

            # Reconnect dead connectors
            if gap.gap_id.startswith("DOWN-"):
                component = gap.component
                if component == "coingecko":
                    ok = await self._init_coingecko()
                    if ok:
                        gap.resolved = True
                        gap.resolution = "Reconnected successfully"
                elif component == "ndax":
                    ok = await self._init_ndax()
                    if ok:
                        gap.resolved = True
                        gap.resolution = "Reconnected successfully"
                elif component == "ibkr":
                    ok = await self._init_ibkr()
                    if ok:
                        gap.resolved = True
                        gap.resolution = "Reconnected to TWS"
                elif component == "moomoo":
                    ok = await self._init_moomoo()
                    if ok:
                        gap.resolved = True
                        gap.resolution = "Reconnected to OpenD"

        healed = sum(1 for g in self.gaps if g.resolved)
        if healed:
            logger.info(f"SELF-HEAL: resolved {healed}/{len(self.gaps)} gaps")

    # ══════════════════════════════════════════════════════════════════
    # OPENCLAW MESSAGING — Send alerts and reports through all channels
    # ══════════════════════════════════════════════════════════════════

    async def _notify(self, message: str, critical: bool = False):
        """Send a notification through OpenClaw (if connected) and log it."""
        logger.info(f"{'CRITICAL ALERT' if critical else 'NOTIFY'}: {message}")
        if self._openclaw:
            try:
                from integrations.openclaw_gateway_bridge import OpenClawChannel
                channel = OpenClawChannel.WEBCHAT  # Default channel
                metadata = {"critical": critical, "engine_cycle": self.cycle_count}
                await self._openclaw.send_proactive_message(
                    channel=channel,
                    session_key="main",
                    content=message,
                    metadata=metadata,
                )
            except Exception as e:
                logger.debug(f"OpenClaw notify failed (non-blocking): {e}")

    # ══════════════════════════════════════════════════════════════════
    # STATUS API — For external tools / OpenClaw to query
    # ══════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict:
        """Return full engine status as a dict (for API/OpenClaw consumption)."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
        return {
            "engine": "AutonomousEngine",
            "version": "1.0.0",
            "running": self.running,
            "uptime_hours": round(uptime / 3600, 2),
            "cycle_count": self.cycle_count,
            "error_count": self.error_count,
            "error_rate": round(self.error_count / max(1, self.cycle_count), 4),
            "doctrine": {
                "state": self.doctrine.state.value,
                "daily_pnl": self.doctrine.daily_pnl,
                "drawdown_pct": self.doctrine.drawdown_pct,
                "can_trade": self.doctrine.can_open_new_positions,
            },
            "components": {
                name: {
                    "health": c.health.value,
                    "errors": c.error_count,
                    "last_error": c.last_error[:100] if c.last_error else "",
                }
                for name, c in self.components.items()
            },
            "gaps": [
                {"id": g.gap_id, "severity": g.severity.value, "resolved": g.resolved}
                for g in self.gaps
            ],
            "daily_trades": len(self._daily_trades),
            "recent_signals": [
                {
                    "symbol": s.symbol,
                    "action": s.action.value,
                    "confidence": s.confidence,
                    "strategy": s.strategy_name,
                    "time": s.timestamp.isoformat(),
                }
                for s in list(self.signals_history)[-5:]
            ],
            "market_data": {
                symbol: {"price": snap.price, "source": snap.source}
                for symbol, snap in self._market_data.items()
            },
        }


# ════════════════════════════════════════════════════════════════════════
# MARKET DATA SNAPSHOT — Normalized price data from any source
# ════════════════════════════════════════════════════════════════════════

@dataclass
class MarketDataSnapshot:
    """MarketDataSnapshot class."""
    symbol: str
    price: float
    bid: float = 0.0
    ask: float = 0.0
    volume_24h: float = 0.0
    source: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ════════════════════════════════════════════════════════════════════════
# UTILITY
# ════════════════════════════════════════════════════════════════════════

def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT — Run the autonomous engine
# ════════════════════════════════════════════════════════════════════════

async def main():
    """Main."""
    engine = AutonomousEngine()

    # Handle graceful shutdown
    import signal as sig

    def handle_shutdown(signum, frame):
        """Handle shutdown."""
        logger.info(f"Received signal {signum} — initiating shutdown")
        asyncio.get_event_loop().create_task(engine.stop())

    sig.signal(sig.SIGINT, handle_shutdown)
    sig.signal(sig.SIGTERM, handle_shutdown)

    try:
        await engine.start()
    except KeyboardInterrupt:
        await engine.stop()
    except Exception as e:
        logger.critical(f"AUTONOMOUS ENGINE CRASHED: {e}")
        logger.critical(traceback.format_exc())
        await engine.stop()
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                PROJECT_ROOT / "logs" / "autonomous_engine.log",
                mode="a", encoding="utf-8",
            ),
        ],
    )
    asyncio.run(main())
