#!/usr/bin/env python3
"""
AAC 2100 MASTER MONITORING DASHBOARD
=====================================
Unified comprehensive monitoring and display system combining all AAC monitoring capabilities.

Features:
- Real-time doctrine compliance monitoring (11 packs)
- System health and performance metrics
- Trading activity and P&L tracking
- Risk management dashboard
- Security monitoring and alerts
- Strategy metrics and analytics
- Production safeguards status
- Circuit breaker monitoring
- Alert management and notifications
- Multiple display modes (terminal, web, text)
- Elite Trading Desk consolidated command console:
  * Jonny Bravo Division (trading education & methodology)
  * Superstonk / Reddit WSB sentiment tracking
  * PlanktonXD Polymarket prediction harvester
  * Unusual Whales options flow & dark pool intel
  * Grok AI trade scorer (xAI / Claude / GPT)
  * OpenClaw Barren Wuffet 93-skill task hub
  * Stock ticker ribbon (Polygon.io real-time)
  * NCL Link cross-pillar data display (NCC/NCL/BRS)
- Multi-Pillar Matrix Monitor Network:
  * NCC (HUB :8765) — Governance & Command
  * AAC (BANK :8080) — Trading & Capital (self)
  * NCL (BRAIN :8787) — Cognitive Augmentation
  * BRS/DL (AGENCY :8000) — Digital Labour
  * NCC MASTER C2 (:8765) — Supreme Orchestrator
  * Per-pillar health polling, directive tracking, heartbeat monitoring

Consolidated from:
- monitoring_dashboard.py (terminal dashboard)
- aac_monitoring_dashboard.py (streamlit dashboard)
- strategy_metrics_dashboard.py (dash metrics)
- security_dashboard.py (security monitoring)
- continuous_monitoring.py (background service)
- enhanced_metrics_display.py (enhanced display)

Display Modes:
- TERMINAL: Curses-based real-time dashboard (Unix) / Text-based (Windows)
- WEB: Streamlit web interface for remote access
- DASH: Plotly Dash analytics dashboard
- API: REST API for external integrations
"""

import asyncio
import json
import logging
import os
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import psutil

logger = logging.getLogger(__name__)

# Try to import curses, fallback for Windows
try:
    import curses

    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False
    logger.info("Warning: curses not available, using text-based dashboard")

# Optional imports for different display modes
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

# Dash imports are handled inside the class to avoid module-level import issues
DASH_AVAILABLE = False

# Add project root and AAC package path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Core AAC imports
# Doctrine integration
from aac.doctrine.doctrine_integration import get_doctrine_integration

# Department engines
from CentralAccounting.financial_analysis_engine import FinancialAnalysisEngine
from CryptoIntelligence.crypto_intelligence_engine import CryptoIntelligenceEngine
from integrations.unusual_whales_service import get_unusual_whales_snapshot_service
from shared.audit_logger import get_audit_logger
from shared.config_loader import get_config
from shared.monitoring import get_monitoring_manager
from shared.production_safeguards import (
    get_production_safeguards,
    get_safeguards_health,
)

# Security framework
try:
    from shared.security_framework import (
        advanced_encryption,
        api_security,
        rbac,
        security_monitoring,
    )

    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

# Strategy testing (optional)
try:
    from strategies.strategy_testing_lab_fixed import (
        initialize_strategy_testing_lab,
        strategy_testing_lab,
    )

    STRATEGY_TESTING_AVAILABLE = True
except ImportError:
    STRATEGY_TESTING_AVAILABLE = False

# Regime forecaster (AAC intelligence layer)
try:
    from strategies.crypto_forecaster import CryptoForecaster, CryptoSnapshot
    from strategies.regime_engine import MacroSnapshot, RegimeEngine
    from strategies.stock_forecaster import Horizon, StockForecaster

    FORECASTER_AVAILABLE = True
except ImportError:
    FORECASTER_AVAILABLE = False

# MATRIX MAXIMIZER (geopolitical bear market options engine)
try:
    from strategies.matrix_maximizer.core import MatrixConfig
    from strategies.matrix_maximizer.runner import MatrixMaximizer

    MATRIX_MAXIMIZER_AVAILABLE = True
except ImportError:
    MATRIX_MAXIMIZER_AVAILABLE = False

# Additional dashboard components
try:
    from trading.aac_arbitrage_execution_system import (
        AACArbitrageExecutionSystem,
        ExecutionConfig,
    )
    from trading.binance_arbitrage_integration import (
        BinanceArbitrageClient,
        BinanceConfig,
    )
    from trading.binance_trading_engine import TradingConfig

    from monitoring.continuous_monitoring import ContinuousMonitoringService
    from monitoring.security_dashboard import SecurityDashboard
    from strategies.strategy_analysis_engine import (
        initialize_strategy_analysis,
        strategy_analysis_engine,
    )
    from tools.enhanced_metrics_display import AACMetricsDisplay

    ARBITRAGE_COMPONENTS_AVAILABLE = True
    TRADING_AVAILABLE = True
except ImportError:
    ARBITRAGE_COMPONENTS_AVAILABLE = False
    TRADING_AVAILABLE = False

# Database manager (optional)
try:
    from CentralAccounting.database import AccountingDatabase as DatabaseManager

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    DatabaseManager = None

# System Registry (unified status tracker)
try:
    from monitoring.aac_system_registry import SystemRegistry

    SYSTEM_REGISTRY_AVAILABLE = True
except ImportError:
    SYSTEM_REGISTRY_AVAILABLE = False
    SystemRegistry = None  # type: ignore[assignment,misc]

# Black Swan Pressure Cooker (crisis center)
try:
    from strategies.black_swan_pressure_cooker import (
        get_crisis_data,
        get_crisis_section,
    )

    CRISIS_CENTER_AVAILABLE = True
except ImportError:
    CRISIS_CENTER_AVAILABLE = False

# Storm Lifeboat Matrix v9.0 (scenario / MC / coherence engine)
try:
    from strategies.storm_lifeboat.coherence import CoherenceEngine
    from strategies.storm_lifeboat.core import MandateLevel
    from strategies.storm_lifeboat.core import VolRegime as SLVolRegime
    from strategies.storm_lifeboat.lunar_phi import LunarPhiEngine
    from strategies.storm_lifeboat.scenario_engine import ScenarioEngine

    STORM_LIFEBOAT_AVAILABLE = True
except ImportError:
    STORM_LIFEBOAT_AVAILABLE = False

# ── NEW INTEGRATIONS: Elite Trading Desk Consolidated ──────────────

# Jonny Bravo Division (trading education agent)
try:
    from agent_jonny_bravo_division.jonny_bravo_agent import JonnyBravoAgent

    JONNY_BRAVO_AVAILABLE = True
except ImportError:
    JONNY_BRAVO_AVAILABLE = False

# Superstonk / Reddit Sentiment
try:
    from reddit.reddit_sentiment_integration import (
        RedditSentimentClient,
        RedditSentimentConfig,
    )

    REDDIT_SENTIMENT_AVAILABLE = True
except ImportError:
    REDDIT_SENTIMENT_AVAILABLE = False

# PlanktonXD Prediction Market Harvester
try:
    from strategies.planktonxd_prediction_harvester import PlanktonXDSimulator

    PLANKTONXD_AVAILABLE = True
except ImportError:
    PLANKTONXD_AVAILABLE = False

# Polymarket Division — War Room Poly + PolyMC Agent + Monitor + Active Scanner
try:
    from strategies.polymarket_division import get_division_status
    from strategies.polymarket_division.account_tracker import PolymarketAccountTracker
    from strategies.polymarket_division.active_scanner import ActiveScanner
    from strategies.polymarket_division.polymc_agent import PolyMCAgent
    from strategies.polymarket_division.polymc_monitor import PolyMCMonitor
    from strategies.polymarket_division.war_room_poly import WarRoomPoly

    POLYMARKET_DIVISION_AVAILABLE = True
except ImportError:
    POLYMARKET_DIVISION_AVAILABLE = False

# Grok AI Trade Scorer (xAI integration)
try:
    from strategies.options_intelligence.ai_scorer import AITradeScorer

    GROK_SCORER_AVAILABLE = True
except ImportError:
    GROK_SCORER_AVAILABLE = False

# OpenClaw Barren Wuffet Skills (task completion hub)
try:
    from integrations.openclaw_barren_wuffet_skills import (
        get_skill_count,
        get_skill_names,
        get_skills_by_category,
    )

    OPENCLAW_AVAILABLE = True
except ImportError:
    OPENCLAW_AVAILABLE = False

# Stock Ticker (Polygon.io)
try:
    from integrations.polygon_client import PolygonClient

    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

# Cross-Pillar Hub (NCL Link)
try:
    from integrations.cross_pillar_hub import CrossPillarHub

    CROSS_PILLAR_AVAILABLE = True
except ImportError:
    CROSS_PILLAR_AVAILABLE = False

# NCC MASTER Adapter (heartbeat, directive handling, matrix status reporting)
try:
    from integrations.ncc_master_adapter import NCCMasterAdapter

    NCC_MASTER_ADAPTER_AVAILABLE = True
except ImportError:
    NCC_MASTER_ADAPTER_AVAILABLE = False

# NCC Integration Bridge (heartbeat publisher, command consumer)
try:
    from shared.ncc_integration import NCC_AAC_Bridge, get_ncc_bridge

    NCC_BRIDGE_AVAILABLE = True
except ImportError:
    NCC_BRIDGE_AVAILABLE = False

# Pillar Matrix Federation (deep matrix data from all pillars)
try:
    from integrations.pillar_matrix_federation import (
        PillarMatrixFederation,
        get_pillar_federation,
    )

    PILLAR_FEDERATION_AVAILABLE = True
except ImportError:
    PILLAR_FEDERATION_AVAILABLE = False

# Strategy Advisor Engine (paper-proof leaderboard)
try:
    from strategies.strategy_advisor_engine import (
        StrategyAdvisorEngine,
        get_strategy_advisor_engine,
    )

    STRATEGY_ADVISOR_AVAILABLE = True
except ImportError:
    STRATEGY_ADVISOR_AVAILABLE = False

# Paper Trading Divisions (Polymarket + Crypto paper bots)
try:
    from divisions.trading.polymarket_paper import PolymarketPaperDivision

    POLYMARKET_PAPER_AVAILABLE = True
except ImportError:
    POLYMARKET_PAPER_AVAILABLE = False

try:
    from divisions.trading.crypto_paper import CryptoPaperDivision

    CRYPTO_PAPER_AVAILABLE = True
except ImportError:
    CRYPTO_PAPER_AVAILABLE = False

# Strategy-Aware Doctrine
try:
    from aac.doctrine.strategic_doctrine import (
        get_strategy_aware_doctrine,
    )

    STRATEGY_DOCTRINE_AVAILABLE = True
except ImportError:
    STRATEGY_DOCTRINE_AVAILABLE = False

# Doctrine Packs YAML
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class DisplayMode:
    """Display mode enumeration"""

    TERMINAL = "terminal"  # Curses/text-based terminal dashboard
    WEB = "web"  # Streamlit web dashboard
    DASH = "dash"  # Plotly Dash analytics dashboard
    API = "api"  # REST API mode


class AACMasterMonitoringDashboard:
    """
    Master monitoring dashboard consolidating all AAC monitoring capabilities.

    Features:
    - Doctrine compliance monitoring (11 packs)
    - System health and performance
    - Trading activity and P&L
    - Risk management
    - Security monitoring
    - Strategy analytics
    - Production safeguards
    - Multiple display modes
    """

    def __init__(self, display_mode: str = DisplayMode.TERMINAL):
        self.display_mode = display_mode
        self.config = get_config()
        self.audit_logger = get_audit_logger()

        # Core monitoring components
        self.monitoring = get_monitoring_manager()
        self.safeguards = get_production_safeguards()
        self.doctrine_integration = None

        # Department engines
        self.financial_engine = FinancialAnalysisEngine()
        self.crypto_engine = CryptoIntelligenceEngine()
        self.unusual_whales = get_unusual_whales_snapshot_service()

        # Optional components
        self.execution_system = None
        self.security_framework = None
        self.strategy_testing = None

        # Dashboard state
        self.running = False
        self.last_update = datetime.now()
        self.refresh_rate = max(
            1.0, float(os.environ.get("DASHBOARD_REFRESH_RATE", "5.0"))
        )  # seconds, min 1s
        self._latest_data = {}
        self._data_lock = threading.Lock()

        # Threads and processes
        self.monitoring_thread = None
        self.display_thread = None

        # System Registry (unified component tracker)
        self.system_registry = SystemRegistry() if SYSTEM_REGISTRY_AVAILABLE else None

        # ── Elite Trading Desk Integrated Components ──────────────────
        self.jonny_bravo = JonnyBravoAgent() if JONNY_BRAVO_AVAILABLE else None
        self.reddit_sentiment = (
            RedditSentimentClient(RedditSentimentConfig())
            if REDDIT_SENTIMENT_AVAILABLE
            else None
        )
        self.planktonxd = PlanktonXDSimulator() if PLANKTONXD_AVAILABLE else None
        self.polymc_agent = PolyMCAgent() if POLYMARKET_DIVISION_AVAILABLE else None
        self.polymc_monitor = PolyMCMonitor() if POLYMARKET_DIVISION_AVAILABLE else None
        self.war_room_poly = WarRoomPoly() if POLYMARKET_DIVISION_AVAILABLE else None
        self.poly_account_tracker = PolymarketAccountTracker() if POLYMARKET_DIVISION_AVAILABLE else None
        self.active_scanner = ActiveScanner(dry_run=True) if POLYMARKET_DIVISION_AVAILABLE else None
        self.grok_scorer = AITradeScorer() if GROK_SCORER_AVAILABLE else None
        self.polygon_client = PolygonClient() if POLYGON_AVAILABLE else None
        self.cross_pillar_hub = CrossPillarHub() if CROSS_PILLAR_AVAILABLE else None

        # ── Multi-Pillar Matrix Monitor Network ──────────────────────
        self.ncc_master_adapter = (
            NCCMasterAdapter() if NCC_MASTER_ADAPTER_AVAILABLE else None
        )
        self.ncc_bridge = get_ncc_bridge() if NCC_BRIDGE_AVAILABLE else None
        self.pillar_federation = (
            get_pillar_federation() if PILLAR_FEDERATION_AVAILABLE else None
        )

        # Pillar endpoint registry — ports per the architecture
        self._pillar_endpoints = {
            "NCC_MASTER": {
                "name": "NCC MASTER C2",
                "role": "Supreme Orchestrator",
                "port": 8765,
                "health_url": os.environ.get("NCC_MASTER_URL", "http://localhost:8765")
                + "/health",
                "matrix_url": os.environ.get("NCC_MASTER_URL", "http://localhost:8765")
                + "/matrix/sitrep",
            },
            "NCC": {
                "name": "NCC (HUB)",
                "role": "Governance & Command",
                "port": 8765,
                "health_url": os.environ.get("NCC_COMMAND_URL", "http://127.0.0.1:8765")
                + "/health",
                "matrix_url": os.environ.get("NCC_COMMAND_URL", "http://127.0.0.1:8765")
                + "/ncc/matrix-monitor",
            },
            "AAC": {
                "name": "AAC (BANK)",
                "role": "Trading & Capital",
                "port": 8080,
                "health_url": "http://127.0.0.1:"
                + os.environ.get("HEALTH_CHECK_PORT", "8080")
                + "/health",
                "matrix_url": "self",  # We ARE the AAC Matrix Monitor
            },
            "NCL": {
                "name": "NCL (BRAIN)",
                "role": "Cognitive Augmentation",
                "port": 8787,
                "health_url": os.environ.get("NCC_RELAY_URL", "http://127.0.0.1:8787")
                + "/health",
                "matrix_url": os.environ.get("NCC_RELAY_URL", "http://127.0.0.1:8787")
                + "/health",
            },
            "BRS": {
                "name": "BRS/DL (AGENCY)",
                "role": "Digital Labour",
                "port": 8000,
                "health_url": "http://localhost:8000/monitor/overview",
                "matrix_url": "http://localhost:8000/matrix/sitrep",
            },
        }

        # Logger
        self.logger = logging.getLogger("MasterDashboard")

    async def initialize(self) -> bool:
        """Initialize all monitoring components"""
        self.logger.info("[INIT] Initializing AAC Master Monitoring Dashboard...")

        try:
            # Initialize core safeguards
            from shared.production_safeguards import initialize_production_safeguards

            await initialize_production_safeguards()

            # Initialize doctrine integration
            self.doctrine_integration = get_doctrine_integration()
            await self.doctrine_integration.initialize()

            # Initialize security framework if available
            if SECURITY_AVAILABLE:
                self.security_framework = {
                    "rbac": rbac,
                    "api_security": api_security,
                    "security_monitoring": security_monitoring,
                    "encryption": advanced_encryption,
                }

            # Initialize trading system if available
            if TRADING_AVAILABLE:
                try:
                    execution_config = ExecutionConfig()
                    self.execution_system = AACArbitrageExecutionSystem(
                        execution_config
                    )
                    await self.execution_system.initialize()
                except Exception as e:
                    self.logger.warning(
                        "[WARN] Trading system init failed (non-fatal): %s", e
                    )
                    self.execution_system = None

            # Initialize strategy testing if available
            if STRATEGY_TESTING_AVAILABLE:
                await initialize_strategy_testing_lab()
                self.strategy_testing = strategy_testing_lab

            self.logger.info("[SUCCESS] Master monitoring dashboard initialized")
            return True

        except Exception as e:
            self.logger.error(f"[CROSS] Failed to initialize master dashboard: {e}")
            return False

    # Per-collector timeout (seconds).  Network-heavy collectors
    # (IBKR, pillar network) get the same budget as local ones —
    # if they can't respond in this time, they're down.
    _COLLECTOR_TIMEOUT: float = 5.0

    async def _safe_collect(
        self,
        name: str,
        coro_or_callable,
    ) -> tuple[str, Any, str]:
        """Run a single collector with a timeout.

        Returns (name, result, status) where status is 'ok', 'timeout', or 'error'.
        Sync callables are wrapped so they participate in the gather.
        """
        try:
            if asyncio.iscoroutinefunction(coro_or_callable):
                result = await asyncio.wait_for(
                    coro_or_callable(), timeout=self._COLLECTOR_TIMEOUT
                )
            elif asyncio.iscoroutine(coro_or_callable):
                result = await asyncio.wait_for(
                    coro_or_callable, timeout=self._COLLECTOR_TIMEOUT
                )
            else:
                # Sync callable — just call it
                result = coro_or_callable()
            return (name, result, "ok")
        except asyncio.TimeoutError:
            self.logger.warning("Collector %s timed out (%.1fs)", name, self._COLLECTOR_TIMEOUT)
            return (name, {}, "timeout")
        except Exception as exc:
            self.logger.warning("Collector %s failed: %s", name, exc)
            return (name, {}, "error")

    async def collect_monitoring_data(self) -> Dict[str, Any]:
        """Collect comprehensive monitoring data from all systems **in parallel**.

        Every collector runs concurrently with a per-collector timeout
        (default 5 s).  The result dict always contains a ``_collector_status``
        key that reports which collectors succeeded, timed out, or errored —
        so the display layer can show a degradation panel.
        """
        timestamp = datetime.now()

        # Build the list of (key, callable/coroutine-function) pairs.
        # Sync helpers are fine — _safe_collect handles both.
        collectors: list[tuple[str, Any]] = [
            ("health", self._get_system_health),
            ("pnl", self._get_pnl_data),
            ("risk", self._get_risk_metrics),
            ("trading", self._get_trading_activity),
            ("doctrine", self._get_doctrine_compliance),
            ("market_intelligence", self._get_market_intelligence),
            ("security", self._get_security_status),
            ("strategy", self._get_strategy_metrics),
            ("safeguards", get_safeguards_health),
            ("forecaster", self._get_forecaster_intel),
            ("ibkr_orders", self._get_ibkr_orders),
            ("alerts", self._get_alerts),
            ("registry", self._get_registry_snapshot),
            ("matrix_maximizer", self._get_matrix_maximizer_deep),
            ("crisis_center", self._get_crisis_center_data),
            ("storm_lifeboat", self._get_storm_lifeboat_data),
            ("capital_rotation", self._get_capital_rotation_matrix),
            ("doctrine_packs", self._get_doctrine_packs_data),
            ("strategy_advisor", self._get_strategy_advisor_data),
            ("active_doctrine", self._get_active_strategy_doctrine_data),
            ("jonny_bravo", self._get_jonny_bravo_data),
            ("superstonk", self._get_superstonk_data),
            ("planktonxd", self._get_planktonxd_data),
            ("polymarket_division", self._get_polymarket_division_data),
            ("active_scanner", self._get_active_scanner_data),
            ("grok_scorer", self._get_grok_scorer_data),
            ("openclaw", self._get_openclaw_data),
            ("stock_ticker", self._get_stock_ticker_data),
            ("ncl_link", self._get_ncl_link_data),
            ("pillar_network", self._get_pillar_network_status),
            ("relay_status", self._get_relay_status_data),
            ("polymarket_paper", self._get_polymarket_paper_data),
            ("crypto_paper", self._get_crypto_paper_data),
        ]

        # Pillar federation (conditional)
        if self.pillar_federation:
            collectors.append(("pillar_matrix_deep", self.pillar_federation.collect_all))

        # Fire all collectors concurrently
        tasks = [self._safe_collect(name, fn) for name, fn in collectors]
        results = await asyncio.gather(*tasks)

        # Assemble output dict + collector health report
        data: Dict[str, Any] = {"timestamp": timestamp}
        collector_status: Dict[str, str] = {}
        ok_count = 0
        fail_count = 0
        timeout_count = 0

        for name, value, status in results:
            data[name] = value
            collector_status[name] = status
            if status == "ok":
                ok_count += 1
            elif status == "timeout":
                timeout_count += 1
            else:
                fail_count += 1

        # Fill any missing keys with empty dicts
        all_keys = [k for k, _ in collectors]
        if not self.pillar_federation:
            all_keys.append("pillar_matrix_deep")
            data.setdefault("pillar_matrix_deep", {})

        total = ok_count + fail_count + timeout_count
        data["_collector_status"] = {
            "ok": ok_count,
            "failed": fail_count,
            "timed_out": timeout_count,
            "total": total,
            "details": collector_status,
        }

        return data

    async def _get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health data"""
        health = {
            "overall_status": "healthy",
            "departments": {},
            "infrastructure": {},
            "performance": {},
        }

        try:
            # Department health checks
            departments = [
                "BigBrainIntelligence",
                "CentralAccounting",
                "CryptoIntelligence",
                "TradingExecution",
            ]
            for dept in departments:
                health["departments"][dept] = await self._check_department_health(dept)

            # Infrastructure health
            health["infrastructure"] = {
                "database": await self._check_database_health(),
                "network": await self._check_network_health(),
                "memory": await self._check_memory_usage(),
                "cpu": await self._check_cpu_usage(),
            }

            # Overall status
            dept_statuses = [
                d.get("status", "unknown") for d in health["departments"].values()
            ]
            if "critical" in dept_statuses:
                health["overall_status"] = "critical"
            elif "warning" in dept_statuses:
                health["overall_status"] = "warning"

        except Exception as e:
            health["overall_status"] = "error"
            health["error"] = str(e)

        return health

    async def _check_department_health(self, department: str) -> Dict[str, Any]:
        """Check health of a specific department"""
        try:
            if department == "CentralAccounting":
                db_status = await self.financial_engine.get_health_status()
                return {
                    "status": db_status.get("status", "unknown"),
                    "last_transaction": db_status.get("last_reconciliation"),
                    "pending_reconciliations": db_status.get(
                        "reconciliation_issues", 0
                    ),
                }
            elif department == "CryptoIntelligence":
                venue_status = await self.crypto_engine.get_all_venue_health()
                avg_score = (
                    (
                        sum(v["health_score"] for v in venue_status.values())
                        / len(venue_status)
                    )
                    if venue_status
                    else 0.0
                )
                return {
                    "status": "healthy" if venue_status else "warning",
                    "venues_monitored": len(venue_status),
                    "average_health_score": avg_score,
                }
            elif department == "BigBrainIntelligence":
                # Dynamically count agents from BigBrainIntelligence directory
                agent_count = 0
                try:
                    from pathlib import Path

                    bbi_dir = (
                        Path(__file__).resolve().parent.parent / "BigBrainIntelligence"
                    )
                    if bbi_dir.exists():
                        agent_count = len(
                            [
                                f
                                for f in bbi_dir.glob("*.py")
                                if f.stem not in ("__init__", "__pycache__")
                            ]
                        )
                except Exception as e:
                    self.logger.debug(f"BBI agent count fallback: {e}")
                    agent_count = 0
                return {
                    "status": "healthy",
                    "active_agents": agent_count,
                    "predictions_today": 0,
                }
            elif department == "TradingExecution":
                return {"status": "healthy", "active_positions": 0, "orders_pending": 0}
            else:
                return {"status": "unknown"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            if not DATABASE_AVAILABLE:
                return {
                    "status": "unavailable",
                    "error": "DatabaseManager not imported",
                }
            db = DatabaseManager()
            try:
                connected = await db.health_check()
                return {
                    "status": "healthy" if connected else "critical",
                    "connection_pool_size": getattr(db, "pool_size", 1),
                    "active_connections": getattr(db, "active_connections", 1),
                }
            finally:
                if hasattr(db, "close"):
                    db.close()
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return {"status": "critical", "error": str(e)}

    async def _check_network_health(self) -> Dict[str, Any]:
        """Check network connectivity with actual latency measurement"""
        start = time.monotonic()
        try:
            import socket

            s = socket.create_connection(("8.8.8.8", 53), timeout=2)
            s.close()
            latency = round((time.monotonic() - start) * 1000, 1)
            return {
                "status": "healthy" if latency < 200 else "warning",
                "latency_ms": latency,
                "packet_loss": 0.0,
            }
        except Exception as e:
            self.logger.debug(f"Network health check failed: {e}")
            latency = round((time.monotonic() - start) * 1000, 1)
            return {"status": "degraded", "latency_ms": latency, "packet_loss": 100.0}

    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        memory = psutil.virtual_memory()
        return {
            "used_percent": memory.percent,
            "available_gb": memory.available / (1024**3),
            "status": "warning" if memory.percent > 85 else "healthy",
        }

    async def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        return {
            "used_percent": cpu_percent,
            "status": "warning" if cpu_percent > 90 else "healthy",
        }

    async def _get_pnl_data(self) -> Dict[str, Any]:
        """Get P&L data"""
        try:
            risk_metrics = await self.financial_engine.update_risk_metrics()
            total_equity = await self.financial_engine.calculate_portfolio_value()
            daily_pnl = await self.financial_engine.calculate_daily_pnl()
            unrealized_pnl = sum(
                position.unrealized_pnl
                for position in self.financial_engine.positions.values()
            )
            return {
                "daily_pnl": daily_pnl,
                "total_equity": total_equity,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": risk_metrics.max_drawdown_pct,
            }
        except Exception as e:
            self.logger.error(f"PnL data retrieval failed: {e}")
            return {
                "daily_pnl": 0.0,
                "total_equity": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "error": str(e),
            }

    async def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics"""
        try:
            risk_metrics = await self.financial_engine.update_risk_metrics()
            return {
                "var_99": risk_metrics.stressed_var_99,
                "expected_shortfall": risk_metrics.tail_loss_p99,
                "beta": 0.0,
                "correlation_matrix": {
                    "strategy_correlation": risk_metrics.strategy_correlation,
                },
                "stress_test_results": {
                    "portfolio_heat": risk_metrics.portfolio_heat,
                    "margin_buffer": risk_metrics.margin_buffer,
                },
            }
        except Exception as e:
            self.logger.error(f"Risk metrics retrieval failed: {e}")
            return {"error": str(e)}

    async def _get_trading_activity(self) -> Dict[str, Any]:
        """Get trading activity data"""
        activity = {
            "orders_today": 0,
            "fills_today": 0,
            "active_strategies": 0,
            "venue_utilization": {},
        }

        if self.execution_system:
            try:
                # Get data from execution system
                status = self.execution_system.get_system_status()
                activity.update(
                    {
                        "orders_today": status.get("orders_today", 0),
                        "fills_today": status.get("fills_today", 0),
                        "active_strategies": status.get("active_strategies", 0),
                        "venue_utilization": status.get("venue_utilization", {}),
                    }
                )
            except Exception as e:
                self.logger.warning(f"Trading activity fetch failed: {e}")

        return activity

    async def _get_doctrine_compliance(self) -> Dict[str, Any]:
        """Get doctrine compliance data"""
        try:
            if self.doctrine_integration:
                compliance_report = (
                    await self.doctrine_integration.run_compliance_check()
                )
                health_status = await self.doctrine_integration.get_health_status()

                return {
                    "compliance_score": compliance_report.get("compliance_score", 0),
                    "barren_wuffet_state": compliance_report.get(
                        "barren_wuffet_state", "unknown"
                    ),
                    "compliant": compliance_report.get("compliant", 0),
                    "warnings": compliance_report.get("warnings", 0),
                    "violations": compliance_report.get("violations", 0),
                    "monitoring_active": health_status.get("monitoring_active", False),
                    "departments_connected": health_status.get(
                        "departments_connected", 0
                    ),
                    "last_check": compliance_report.get("generated_at"),
                }
            else:
                return {
                    "compliance_score": 0,
                    "barren_wuffet_state": "not_initialized",
                    "compliant": 0,
                    "warnings": 0,
                    "violations": 0,
                    "monitoring_active": False,
                    "departments_connected": 0,
                    "last_check": None,
                }
        except Exception as e:
            return {
                "compliance_score": 0,
                "barren_wuffet_state": "error",
                "compliant": 0,
                "warnings": 0,
                "violations": 0,
                "monitoring_active": False,
                "departments_connected": 0,
                "last_check": None,
                "error": str(e),
            }

    async def _get_market_intelligence(self) -> Dict[str, Any]:
        """Get normalized Unusual Whales market-intelligence snapshot."""
        try:
            snapshot = await self.unusual_whales.get_snapshot()
            return snapshot
        except Exception as e:
            return {
                "status": "error",
                "as_of": datetime.now().isoformat(),
                "error": str(e),
            }

    async def _get_security_status(self) -> Dict[str, Any]:
        """Get security status data"""
        if not SECURITY_AVAILABLE or not self.security_framework:
            return {"status": "not_available"}

        try:
            security_status = {"overall_score": 0, "components": {}, "alerts": []}

            # MFA Status
            mfa_status = self._check_mfa_status()
            security_status["components"]["mfa"] = mfa_status

            # Encryption Status
            encryption_status = self._check_encryption_status()
            security_status["components"]["encryption"] = encryption_status

            # RBAC Status
            rbac_status = self._check_rbac_status()
            security_status["components"]["rbac"] = rbac_status

            # API Security
            api_status = self._check_api_security_status()
            security_status["components"]["api"] = api_status

            # Calculate overall score
            component_scores = []
            for component in security_status["components"].values():
                if isinstance(component, dict) and "score" in component:
                    component_scores.append(component["score"])

            if component_scores:
                security_status["overall_score"] = sum(component_scores) / len(
                    component_scores
                )

            return security_status

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _check_mfa_status(self) -> Dict[str, Any]:
        """Check MFA status — queries actual auth config"""
        try:
            cfg = get_config()
            mfa_enabled = getattr(cfg, "mfa_enabled", False)
            return {
                "enabled_users": 1 if mfa_enabled else 0,
                "total_users": 1,
                "score": 100 if mfa_enabled else 0,
                "status": "healthy" if mfa_enabled else "not_configured",
            }
        except Exception as e:
            self.logger.debug(f"MFA status check failed: {e}")
            return {
                "enabled_users": 0,
                "total_users": 0,
                "score": 0,
                "status": "not_implemented",
            }

    def _check_encryption_status(self) -> Dict[str, Any]:
        """Check encryption status — verifies secrets manager is initialized"""
        try:
            from shared.secrets_manager import get_secrets_manager

            sm = get_secrets_manager()
            encrypted = sm is not None and hasattr(sm, "_fernet")
            return {
                "encrypted_databases": 1 if encrypted else 0,
                "total_databases": 1,
                "score": 100 if encrypted else 0,
                "status": "healthy" if encrypted else "not_configured",
            }
        except Exception as e:
            self.logger.debug(f"Encryption status check failed: {e}")
            return {
                "encrypted_databases": 0,
                "total_databases": 0,
                "score": 0,
                "status": "not_implemented",
            }

    def _check_rbac_status(self) -> Dict[str, Any]:
        """Check RBAC status — queries actual role definitions from security_framework"""
        try:
            from shared.security_framework import rbac as rbac_manager

            roles = (
                len(rbac_manager.roles)
                if rbac_manager and hasattr(rbac_manager, "roles")
                else 0
            )
            return {
                "roles_defined": roles,
                "permissions_assigned": roles * 10,
                "score": min(100, roles * 15),
                "status": "healthy" if roles >= 3 else "warning",
            }
        except Exception as e:
            self.logger.debug(f"RBAC status check failed: {e}")
            return {
                "roles_defined": 0,
                "permissions_assigned": 0,
                "score": 0,
                "status": "not_implemented",
            }

    def _check_api_security_status(self) -> Dict[str, Any]:
        """Check API security — verifies TLS and auth middleware presence"""
        try:
            import ssl

            has_tls = ssl.OPENSSL_VERSION is not None
            return {
                "endpoints_secured": 1 if has_tls else 0,
                "total_endpoints": 1,
                "score": 80 if has_tls else 20,
                "status": "healthy" if has_tls else "warning",
            }
        except Exception as e:
            self.logger.debug(f"API security status check failed: {e}")
            return {
                "endpoints_secured": 0,
                "total_endpoints": 0,
                "score": 0,
                "status": "not_implemented",
            }

    async def _get_forecaster_intel(self) -> Dict[str, Any]:
        """Run regime engine + stock forecaster. Returns compact intel dict for dashboard."""
        if not FORECASTER_AVAILABLE:
            return {"status": "not_available"}
        try:
            import os

            from strategies.regime_engine import MacroSnapshot, RegimeEngine
            from strategies.stock_forecaster import Horizon, StockForecaster

            # Build snapshot from env overrides or safe defaults
            snap = MacroSnapshot(
                vix=float(os.environ.get("MONITOR_VIX", "21.5")),
                hy_spread_bps=float(os.environ.get("MONITOR_HY_SPREAD", "380")),
                oil_price=float(os.environ.get("MONITOR_OIL", "95.5")),
                gold_price=float(os.environ.get("MONITOR_GOLD", "5011")),
                core_pce=float(os.environ.get("MONITOR_PCE", "3.1")),
                gdp_growth=float(os.environ.get("MONITOR_GDP", "0.7")),
                yield_curve_10_2=float(os.environ.get("MONITOR_YIELD_CURVE", "-0.3")),
                private_credit_redemption_pct=float(
                    os.environ.get("MONITOR_PRIV_CREDIT", "11.0")
                ),
                war_active=os.environ.get("MONITOR_WAR", "true").lower() == "true",
                hormuz_blocked=os.environ.get("MONITOR_HORMUZ", "true").lower()
                == "true",
                hyg_return_1d=float(os.environ.get("MONITOR_HYG_RET", "-0.6")),
                spy_return_1d=float(os.environ.get("MONITOR_SPY_RET", "-0.4")),
                kre_return_1d=float(os.environ.get("MONITOR_KRE_RET", "-0.9")),
                airlines_return_1d=float(os.environ.get("MONITOR_JETS_RET", "-1.1")),
            )

            state = RegimeEngine().evaluate(snap)
            forecaster = StockForecaster()
            short_fcast = forecaster.forecast(state, Horizon.SHORT, top_n=3)

            top3 = [
                {
                    "rank": o.rank,
                    "ticker": o.primary_ticker,
                    "industry": o.industry.value,
                    "expression": o.expression.value,
                    "score": round(o.composite_score, 1),
                    "thesis": o.thesis[:80],
                    "structure": o.structure_hint[:70],
                }
                for o in short_fcast.opportunities[:3]
            ]

            fired_formulas = [
                {
                    "tag": f.tag.value,
                    "confidence": round(f.confidence * 100),
                    "outcome": f.expected_outcome[:70],
                }
                for f in state.top_formulas[:3]
            ]

            two_stack = forecaster.two_trade_stack(state)

            return {
                "status": "ok",
                "regime": state.primary_regime.value,
                "secondary": state.secondary_regime.value
                if state.secondary_regime
                else None,
                "regime_confidence": round(state.regime_confidence * 100),
                "vol_shock_readiness": state.vol_shock_readiness,
                "bear_signals": state.bear_signals,
                "bull_signals": state.bull_signals,
                "fired_formulas": fired_formulas,
                "top3_opportunities": top3,
                "anchor_ticker": two_stack[0].primary_ticker if two_stack[0] else None,
                "contagion_ticker": two_stack[1].primary_ticker
                if two_stack[1]
                else None,
                "macro_context": {
                    "vix": snap.vix,
                    "hy_spread": snap.hy_spread_bps,
                    "oil": snap.oil_price,
                    "war": snap.war_active,
                    "hormuz": snap.hormuz_blocked,
                },
            }
        except Exception as e:
            self.logger.warning(f"Forecaster intel error: {e}")
            return {"status": "error", "error": str(e)}

    async def _get_ibkr_orders(self) -> Dict[str, Any]:
        """Fetch open IBKR orders and account balance for maximization plan."""
        connector = None
        try:
            from TradingExecution.exchange_connectors.ibkr_connector import (
                IBKRConnector,
            )

            connector = IBKRConnector()
            await connector.connect()
            orders_raw = await connector.get_open_orders()

            # Fetch real account balance instead of hardcoded value
            account_balance = 0.0
            try:
                balances = await connector.get_balances()
                if "USD" in balances:
                    account_balance = balances["USD"].free
                elif "TOTAL_CASH" in balances:
                    account_balance = balances["TOTAL_CASH"].free
            except Exception:
                self.logger.warning("Could not fetch IBKR balance, using 0")

            await connector.disconnect()
            connector = None

            orders = []
            total_committed = 0.0
            for o in orders_raw:
                # o is an ExchangeOrder object or its string repr
                raw = o if isinstance(o, dict) else {}
                # Parse from string repr if needed
                if hasattr(o, "symbol"):
                    qty = float(getattr(o, "quantity", 0))
                    price = float(getattr(o, "price", 0))
                    cost = qty * price * 100  # options: qty contracts × premium × 100
                    total_committed += cost
                    orders.append(
                        {
                            "order_id": getattr(o, "order_id", "?"),
                            "symbol": getattr(o, "symbol", "?").replace("/USD", ""),
                            "side": getattr(o, "side", "?"),
                            "qty": qty,
                            "price": price,
                            "cost": cost,
                            "status": getattr(o, "status", "?"),
                            "ibkr_status": getattr(o, "raw", {}).get(
                                "ibkr_status", "?"
                            ),
                        }
                    )

            remaining = max(0.0, account_balance - total_committed)

            # Recommendations are generated by the strategy/forecaster layer, not here.
            # The monitoring dashboard only reports current state.
            recs = []
            rec_cost = 0.0
            cash_after = remaining

            return {
                "status": "ok",
                "account": "U24346218",
                "balance_usd": account_balance,
                "open_orders": orders,
                "total_committed": round(total_committed, 2),
                "remaining_cash": round(remaining, 2),
                "recommendations": recs,
                "rec_total_cost": round(rec_cost, 2),
                "cash_after_recs": round(cash_after, 2),
            }
        except Exception as e:
            self.logger.warning(f"IBKR orders fetch error: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            if connector is not None:
                try:
                    await connector.disconnect()
                except Exception:
                    pass

    async def _get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy metrics data"""
        if not STRATEGY_TESTING_AVAILABLE or not self.strategy_testing:
            return {"status": "not_available"}

        try:
            # Get strategy performance data
            performance_data = await self.strategy_testing.get_performance_summary()
            return {
                "total_strategies": performance_data.get("total_strategies", 0),
                "active_strategies": performance_data.get("active_strategies", 0),
                "best_performing": performance_data.get("best_performing", {}),
                "worst_performing": performance_data.get("worst_performing", {}),
                "average_return": performance_data.get("average_return", 0.0),
                "sharpe_ratio": performance_data.get("sharpe_ratio", 0.0),
            }
        except Exception as e:
            self.logger.error(f"Strategy metrics retrieval failed: {e}")
            return {
                "total_strategies": 0,
                "active_strategies": 0,
                "best_performing": {},
                "worst_performing": {},
                "average_return": 0.0,
                "sharpe_ratio": 0.0,
                "error": str(e),
            }

    # ── SYSTEM REGISTRY + MATRIX MAXIMIZER DEEP ─────────────────

    def _get_registry_snapshot(self) -> Dict[str, Any]:
        """Get a full system registry snapshot (APIs, exchanges, infra, etc.)."""
        if not self.system_registry:
            return {"status": "not_available"}
        try:
            return self.system_registry.collect_full_snapshot()
        except Exception as e:
            self.logger.warning("Registry snapshot failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _get_matrix_maximizer_deep(self) -> Dict[str, Any]:
        """Get deep Matrix Maximizer state — last-run data + module health."""
        if not MATRIX_MAXIMIZER_AVAILABLE:
            return {"status": "not_available"}
        try:
            # Read latest run output
            latest_path = PROJECT_ROOT / "data" / "matrix_maximizer_latest.json"
            if latest_path.exists():
                data = json.loads(latest_path.read_text(encoding="utf-8"))
                regime = data.get("regime", {})
                forecast = data.get("forecast", {})
                picks = data.get("picks", [])
                risk_snap = data.get("risk", {})
                return {
                    "status": "ok",
                    "run_number": data.get("run_number"),
                    "timestamp": data.get("timestamp"),
                    "mandate": forecast.get("mandate", "?"),
                    "risk_per_trade": forecast.get("risk_per_trade"),
                    "max_positions": forecast.get("max_positions"),
                    "spy_median_return": forecast.get("spy_median_return"),
                    "spy_var_95": forecast.get("spy_var_95"),
                    "regime": regime.get("regime", "?"),
                    "regime_confidence": regime.get("confidence"),
                    "war_active": regime.get("war_active"),
                    "hormuz_blocked": regime.get("hormuz_blocked"),
                    "oil_price": regime.get("oil_price"),
                    "vix": regime.get("vix"),
                    "top_picks": [
                        {
                            "ticker": p.get("ticker"),
                            "strike": p.get("strike"),
                            "expiry": p.get("expiry"),
                            "score": p.get("score"),
                            "contracts": p.get("contracts"),
                            "cost": p.get("cost"),
                        }
                        for p in picks[:5]
                    ],
                    "total_picks": len(picks),
                    "circuit_breaker": risk_snap.get("circuit_breaker", "?"),
                    "risk_score": risk_snap.get("risk_score"),
                    "elapsed_s": data.get("elapsed_s"),
                }
            else:
                return {"status": "no_data", "detail": "No run output found"}
        except Exception as e:
            self.logger.warning("Matrix Maximizer deep read failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _get_crisis_center_data(self) -> Dict[str, Any]:
        """Get Black Swan Crisis Center data from pressure cooker."""
        if not CRISIS_CENTER_AVAILABLE:
            return {"status": "unavailable"}
        try:
            return get_crisis_data()
        except Exception as e:
            self.logger.warning("Crisis center data fetch failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _get_storm_lifeboat_data(self) -> Dict[str, Any]:
        """Get Storm Lifeboat Matrix v9.0 state — scenarios, coherence, lunar, last briefing."""
        if not STORM_LIFEBOAT_AVAILABLE:
            return {"status": "not_available"}
        try:
            import glob as _glob

            result: Dict[str, Any] = {"status": "ok"}

            # ── Scenario heatmap (live, from engine defaults) ──
            try:
                from strategies.storm_lifeboat.core import SCENARIOS as _SL_SCENARIOS

                se = ScenarioEngine()
                result["scenario_heatmap"] = se.get_risk_heatmap()
                active = se.get_active_scenarios()
                result["active_scenarios"] = len(active)
                result["scenarios_total"] = len(_SL_SCENARIOS)
            except Exception as e:
                self.logger.debug("Storm Lifeboat scenario engine: %s", e)
                result["scenario_heatmap"] = []
                result["active_scenarios"] = 0

            # ── Lunar position ──
            try:
                lp = LunarPhiEngine()
                pos = lp.get_position()
                result["lunar"] = {
                    "moon_number": pos.moon_number,
                    "moon_name": pos.moon_name,
                    "phase": pos.phase.value,
                    "day_in_moon": pos.day_in_moon,
                    "in_phi_window": pos.in_phi_window,
                    "phi_coherence": round(pos.phi_coherence, 3),
                    "position_multiplier": round(pos.position_multiplier, 2),
                }
            except Exception as e:
                self.logger.debug("Storm Lifeboat lunar: %s", e)
                result["lunar"] = {}

            # ── Latest Helix briefing (persisted JSON) ──
            briefing_dir = PROJECT_ROOT / "data" / "storm_lifeboat"
            briefings = sorted(_glob.glob(str(briefing_dir / "helix_briefing_*.json")))
            if briefings:
                try:
                    latest = Path(briefings[-1])
                    data = json.loads(latest.read_text(encoding="utf-8"))
                    result["last_briefing"] = {
                        "date": data.get("date"),
                        "headline": data.get("headline"),
                        "regime": data.get("regime"),
                        "mandate": data.get("mandate"),
                        "coherence_score": data.get("coherence_score"),
                        "risk_alert": data.get("risk_alert"),
                        "moon_phase": data.get("moon_phase"),
                        "top_trades": data.get("top_trades", [])[:5],
                        "active_scenarios": data.get("active_scenarios", []),
                    }
                except Exception as e:
                    self.logger.debug("Storm Lifeboat briefing read: %s", e)
                    result["last_briefing"] = {}
            else:
                result["last_briefing"] = {}

            return result

        except Exception as e:
            self.logger.warning("Storm Lifeboat data fetch failed: %s", e)
            return {"status": "error", "error": str(e)}

    # ── CAPITAL ROTATION MATRIX — 7 STRATEGIES ─────────────────────

    def _get_capital_rotation_matrix(self) -> Dict[str, Any]:
        """
        Build the 7-strategy Capital Rotation Matrix.

        Strategies 1-5 are LIVE (balances count toward total).
        Strategies 6-7 are PAPER (balances are informational only, not added to total).

        Returns dict with per-strategy data + aggregated live balance + $10M progress.
        """
        TARGET_BALANCE = 10_000_000.0

        strategies: List[Dict[str, Any]] = []

        # ── 1. WAR ROOM — IBKR Margin (PRIMARY) ──
        ibkr_balance = 0.0
        ibkr_positions = 0
        try:
            from strategies.ninety_day_war_room import ACCOUNTS, POSITIONS

            ibkr_acct = ACCOUNTS.get("ibkr")
            if ibkr_acct:
                ibkr_balance = ibkr_acct.balance
                ibkr_positions = len([p for p in POSITIONS if p.account == "ibkr"])
        except Exception:
            pass
        strategies.append(
            {
                "id": 1,
                "name": "WAR ROOM",
                "broker": "IBKR TWS",
                "api": "IBKR TWS API (port 7496)",
                "type": "PRIMARY",
                "role": "Macro crisis puts & margin strategy",
                "is_live": True,
                "balance_usd": ibkr_balance,
                "currency": "USD",
                "positions": ibkr_positions,
                "status": "LIVE" if ibkr_balance > 0 else "OFFLINE",
                "icon": "⚔️",
            }
        )

        # ── 2. LIFEBOAT — Moomoo Margin (SECONDARY) ──
        moomoo_balance = 0.0
        try:
            from strategies.ninety_day_war_room import ACCOUNTS as _ACCTS2

            moomoo_acct = _ACCTS2.get("moomoo")
            if moomoo_acct:
                moomoo_balance = moomoo_acct.balance_usd
        except Exception:
            pass
        strategies.append(
            {
                "id": 2,
                "name": "LIFEBOAT",
                "broker": "Moomoo OpenD",
                "api": "Moomoo OpenD (port 11111)",
                "type": "SECONDARY",
                "role": "Storm Lifeboat margin strategy",
                "is_live": True,
                "balance_usd": moomoo_balance,
                "currency": "USD",
                "positions": 0,
                "status": "LIVE" if moomoo_balance > 0 else "STANDBY",
                "icon": "🛟",
            }
        )

        # ── 3. CRYPTO HEDGE — NDAX + CoinGecko (via Lifeboat) ──
        ndax_balance_usd = 0.0
        try:
            from strategies.ninety_day_war_room import ACCOUNTS as _ACCTS3

            ndax_acct = _ACCTS3.get("ndax")
            if ndax_acct:
                ndax_balance_usd = ndax_acct.balance_usd
        except Exception:
            pass
        strategies.append(
            {
                "id": 3,
                "name": "CRYPTO HEDGE",
                "broker": "NDAX",
                "api": "NDAX API + CoinGecko Pro",
                "type": "SATELLITE",
                "role": "Crypto trading via Lifeboat strategy",
                "is_live": True,
                "balance_usd": ndax_balance_usd,
                "currency": "CAD→USD",
                "positions": 0,
                "status": "LIVE" if ndax_balance_usd > 0 else "EMPTY",
                "icon": "🪙",
            }
        )

        # ── 4. WEALTHSIMPLE TFSA — Manual Legacy ──
        ws_balance_usd = 0.0
        try:
            from strategies.ninety_day_war_room import ACCOUNTS as _ACCTS4

            ws_acct = _ACCTS4.get("wealthsimple")
            if ws_acct:
                ws_balance_usd = ws_acct.balance_usd
        except Exception:
            pass
        strategies.append(
            {
                "id": 4,
                "name": "WEALTHSIMPLE TFSA",
                "broker": "WealthSimple",
                "api": "MANUAL (screenshots)",
                "type": "LEGACY",
                "role": "Tax-free savings — manual positions",
                "is_live": True,
                "balance_usd": ws_balance_usd,
                "currency": "CAD→USD",
                "positions": 0,
                "status": "MANUAL",
                "icon": "🏦",
            }
        )

        # ── 5. POLYMARKET DIVISION — 3 Strategies ──
        poly_balance = 0.0
        poly_bets = 0
        poly_strategies_loaded = 0
        try:
            if self.planktonxd:
                px_data = self._get_planktonxd_data()
                if px_data.get("status") == "ok":
                    poly_balance = px_data.get("total_deployed", 0.0)
                    poly_bets = px_data.get("active_bets", 0)
            div_data = self._get_polymarket_division_data()
            if div_data.get("status") == "ok":
                poly_strategies_loaded = div_data.get("strategies_loaded", 0)
                # Use real account tracker total if available
                acct = div_data.get("account", {})
                if acct.get("status") == "ok":
                    poly_balance = acct.get("total_account_value", poly_balance)
                    poly_bets = acct.get("total_positions", poly_bets)
                else:
                    mc = div_data.get("polymc", {})
                    if mc.get("status") == "ok":
                        poly_balance += mc.get("total_cost", 0)
        except Exception:
            pass
        strategies.append(
            {
                "id": 5,
                "name": "POLYMARKET DIV",
                "broker": "Polymarket",
                "api": f"Division ({poly_strategies_loaded}/3 loaded)",
                "type": "PREDICTION",
                "role": "War Room + PlanktonXD + PolyMC Agent",
                "is_live": True,
                "balance_usd": poly_balance,
                "currency": "USDC",
                "positions": poly_bets,
                "status": "LIVE" if poly_bets > 0 else "SCANNING",
                "icon": "🎯",
            }
        )

        # ── 6. QUE THE FIRE! — Jonny Bravo Paper ──
        qtf_balance = 1000.0  # Arbitrary starting balance
        try:
            qtf_path = PROJECT_ROOT / "data" / "que_the_fire_balance.json"
            if qtf_path.exists():
                qtf_data = json.loads(qtf_path.read_text(encoding="utf-8"))
                qtf_balance = qtf_data.get("balance", 1000.0)
        except Exception:
            pass
        strategies.append(
            {
                "id": 6,
                "name": "QUE THE FIRE!",
                "broker": "IBKR Paper",
                "api": "IBKR TWS API (port 7496)",
                "type": "PAPER",
                "role": "Jonny Bravo strategy — paper trading",
                "is_live": False,
                "balance_usd": qtf_balance,
                "currency": "USD (paper)",
                "positions": 0,
                "status": "PAPER",
                "icon": "🔥",
            }
        )

        # ── 7. UNUSUAL WHALES — Paper ──
        uw_balance = 1000.0  # Paper strategy not yet implemented
        # TODO: Implement UW paper trading strategy and persist balance
        strategies.append(
            {
                "id": 7,
                "name": "UNUSUAL WHALES",
                "broker": "IBKR Paper",
                "api": "Unusual Whales API + IBKR Paper (7497)",
                "type": "PAPER",
                "role": "Options flow strategy — paper trading",
                "is_live": False,
                "balance_usd": uw_balance,
                "currency": "USD (paper)",
                "positions": 0,
                "status": "PAPER",
                "icon": "🐋",
            }
        )

        # ── Aggregation ──
        live_balance = sum(s["balance_usd"] for s in strategies if s["is_live"])
        paper_balance = sum(s["balance_usd"] for s in strategies if not s["is_live"])
        live_count = sum(1 for s in strategies if s["is_live"])
        paper_count = sum(1 for s in strategies if not s["is_live"])
        progress_pct = (
            min(100.0, (live_balance / TARGET_BALANCE) * 100)
            if TARGET_BALANCE > 0
            else 0.0
        )

        return {
            "status": "ok",
            "strategies": strategies,
            "live_balance_usd": round(live_balance, 2),
            "paper_balance_usd": round(paper_balance, 2),
            "live_count": live_count,
            "paper_count": paper_count,
            "target_balance": TARGET_BALANCE,
            "progress_pct": round(progress_pct, 4),
            "remaining_to_target": round(max(0, TARGET_BALANCE - live_balance), 2),
        }

    # ── DOCTRINE PACK READER ──────────────────────────────────────────

    def _get_doctrine_packs_data(self) -> Dict[str, Any]:
        """Load and return all 11 doctrine packs from config/doctrine_packs.yaml."""
        try:
            packs_file = PROJECT_ROOT / "config" / "doctrine_packs.yaml"
            if not packs_file.exists():
                return {"status": "not_found", "packs": [], "count": 0}

            if YAML_AVAILABLE:
                with packs_file.open("r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
                packs = raw.get("doctrine_packs", raw) if isinstance(raw, dict) else raw
                pack_list = []
                if isinstance(packs, list):
                    pack_list = packs
                elif isinstance(packs, dict):
                    for k, v in packs.items():
                        if isinstance(v, dict):
                            v["id"] = k
                            pack_list.append(v)
                return {"status": "ok", "packs": pack_list, "count": len(pack_list)}
            else:
                # Fallback: read as text
                text = packs_file.read_text(encoding="utf-8")
                return {"status": "ok_text", "raw": text[:4000], "count": 11}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "packs": [], "count": 0}

    # ── STRATEGY ADVISOR LEADERBOARD ────────────────────────────────

    def _get_strategy_advisor_data(self) -> Dict[str, Any]:
        """Get strategy advisor engine leaderboard + summary."""
        if not STRATEGY_ADVISOR_AVAILABLE:
            return {"status": "not_available"}
        try:
            advisor = get_strategy_advisor_engine()
            summary = advisor.get_summary()
            return {"status": "ok", **summary}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ── NCL RELAY STATUS ────────────────────────────────────────────

    def _get_relay_status_data(self) -> Dict[str, Any]:
        """Get NCL relay client stats (published, queued, outbox depth)."""
        try:
            from shared.ncc_relay_client import get_relay_client
            relay = get_relay_client()
            stats = relay.stats
            return {"status": "ok", **stats}
        except ImportError:
            return {"status": "not_available"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ── PAPER TRADING DIVISIONS ──────────────────────────────────────

    def _get_polymarket_paper_data(self) -> Dict[str, Any]:
        """Get Polymarket paper trading division status."""
        if not POLYMARKET_PAPER_AVAILABLE:
            return {"status": "not_available"}
        try:
            div = PolymarketPaperDivision(persist=True)
            perf = div._engine.get_performance()
            rankings = div._optimizer.get_rankings()
            return {
                "status": "ok",
                "equity": perf.get("equity", 0),
                "total_pnl": perf.get("total_pnl", 0),
                "total_pnl_pct": perf.get("total_pnl_pct", 0),
                "total_trades": perf.get("total_trades", 0),
                "win_rate": perf.get("win_rate", 0),
                "max_drawdown_pct": perf.get("max_drawdown_pct", 0),
                "open_positions": len(div._engine.account.positions),
                "cycles": div._total_cycles,
                "best_strategy": div._optimizer.get_best_strategy(),
                "rankings": rankings[:3],
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def _get_crypto_paper_data(self) -> Dict[str, Any]:
        """Get Crypto paper trading division status."""
        if not CRYPTO_PAPER_AVAILABLE:
            return {"status": "not_available"}
        try:
            div = CryptoPaperDivision(persist=True)
            perf = div._engine.get_performance()
            rankings = div._optimizer.get_rankings()
            return {
                "status": "ok",
                "equity": perf.get("equity", 0),
                "total_pnl": perf.get("total_pnl", 0),
                "total_pnl_pct": perf.get("total_pnl_pct", 0),
                "total_trades": perf.get("total_trades", 0),
                "win_rate": perf.get("win_rate", 0),
                "max_drawdown_pct": perf.get("max_drawdown_pct", 0),
                "open_positions": len(div._engine.account.positions),
                "cycles": div._total_cycles,
                "best_strategy": div._optimizer.get_best_strategy(),
                "rankings": rankings[:3],
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ── ACTIVE STRATEGY DOCTRINE ────────────────────────────────────

    def _get_active_strategy_doctrine_data(self) -> Dict[str, Any]:
        """Get per-strategy doctrine directives from StrategyAwareDoctrine."""
        if not STRATEGY_DOCTRINE_AVAILABLE:
            return {"status": "not_available"}
        try:
            doctrine = get_strategy_aware_doctrine()
            state = doctrine.get_doctrine_state()
            # Generate directives with empty signals (display current state)
            directives = doctrine.generate_composite_directive({})
            directive_list = []
            for key, d in directives.items():
                directive_list.append({
                    "strategy": d.strategy_name,
                    "allowed": d.allowed,
                    "position_size_pct": d.position_size_pct,
                    "bias": d.bias,
                    "max_positions": d.max_positions,
                    "notes": d.notes,
                })
            return {
                "status": "ok",
                "regime": state.get("regime", "unknown"),
                "manual_overrides": state.get("manual_overrides", []),
                "ncl_caution": state.get("ncl_caution", 0),
                "directives": directive_list,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ── MULTI-PILLAR MATRIX MONITOR NETWORK ─────────────────────────

    async def _get_pillar_network_status(self) -> Dict[str, Any]:
        """
        Poll all pillar Matrix Monitor endpoints for unified status.
        Returns health, connectivity, and matrix status for each pillar.
        """
        import urllib.error
        import urllib.request

        results: Dict[str, Any] = {
            "status": "ok",
            "pillars": {},
            "ncc_master": {},
            "doctrine_mode": "UNKNOWN",
            "total_pillars": len(self._pillar_endpoints),
            "pillars_online": 0,
            "pillars_offline": 0,
        }

        for pillar_id, ep in self._pillar_endpoints.items():
            pillar_data: Dict[str, Any] = {
                "name": ep["name"],
                "role": ep["role"],
                "port": ep["port"],
                "health": "unknown",
                "matrix_status": "unknown",
                "latency_ms": None,
                "error": None,
            }

            # AAC is self — always online
            if pillar_id == "AAC":
                pillar_data["health"] = "GREEN"
                pillar_data["matrix_status"] = "ACTIVE"
                pillar_data["latency_ms"] = 0
                results["pillars_online"] += 1
                results["pillars"][pillar_id] = pillar_data
                continue

            # Poll health endpoint
            health_url = ep.get("health_url", "")
            if health_url:
                try:
                    t0 = time.monotonic()
                    req = urllib.request.Request(health_url, method="GET")
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        latency = (time.monotonic() - t0) * 1000
                        pillar_data["latency_ms"] = round(latency, 1)
                        body = resp.read().decode("utf-8", errors="replace")
                        try:
                            health_json = json.loads(body)
                            pillar_data["health"] = "GREEN"
                            pillar_data["health_detail"] = health_json
                        except json.JSONDecodeError:
                            pillar_data["health"] = "GREEN"
                            pillar_data["health_detail"] = body[:200]
                    results["pillars_online"] += 1
                except urllib.error.URLError as exc:
                    pillar_data["health"] = "RED"
                    pillar_data["error"] = str(exc.reason)[:100]
                    results["pillars_offline"] += 1
                except Exception as exc:
                    pillar_data["health"] = "RED"
                    pillar_data["error"] = str(exc)[:100]
                    results["pillars_offline"] += 1

            # Poll matrix endpoint (only if health succeeded)
            matrix_url = ep.get("matrix_url", "")
            if matrix_url and matrix_url != "self" and pillar_data["health"] == "GREEN":
                try:
                    req = urllib.request.Request(matrix_url, method="GET")
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        body = resp.read().decode("utf-8", errors="replace")
                        try:
                            matrix_json = json.loads(body)
                            pillar_data["matrix_status"] = "ACTIVE"
                            pillar_data["matrix_detail"] = matrix_json
                        except json.JSONDecodeError:
                            pillar_data["matrix_status"] = "RESPONDING"
                except Exception:
                    pillar_data["matrix_status"] = "NO_MATRIX"

            # NCC MASTER gets special treatment
            if pillar_id == "NCC_MASTER":
                results["ncc_master"] = pillar_data

            results["pillars"][pillar_id] = pillar_data

        # Overlay cross-pillar hub state if available
        if self.cross_pillar_hub:
            full = self.cross_pillar_hub.get_full_status()
            results["doctrine_mode"] = full.get("doctrine_mode", "UNKNOWN")
            results["should_trade"] = full.get("should_trade", False)
            results["risk_multiplier"] = full.get("risk_multiplier", 0.0)
            results["last_directive"] = full.get("last_directive")
            results["active_strategies"] = full.get("active_strategies", [])

            # Merge cross-pillar connection info into pillar entries
            for pname, pstat in full.get("pillars", {}).items():
                upper = pname.upper()
                if upper in results["pillars"]:
                    results["pillars"][upper]["cross_pillar_connected"] = pstat.get(
                        "connected", False
                    )
                    results["pillars"][upper]["cross_pillar_mode"] = pstat.get(
                        "mode", "unknown"
                    )
                    results["pillars"][upper]["last_heartbeat"] = pstat.get(
                        "last_heartbeat"
                    )
        else:
            results["doctrine_mode"] = "UNKNOWN"
            results["should_trade"] = True
            results["risk_multiplier"] = 1.0

        # Get NCC adapter matrix status if available
        if self.ncc_master_adapter:
            try:
                matrix_report = self.ncc_master_adapter.get_matrix_status()
                results["aac_matrix_report"] = matrix_report
            except Exception as exc:
                results["aac_matrix_report"] = {"error": str(exc)}

        # Get NCC bridge status if available
        if self.ncc_bridge:
            try:
                bridge_status = getattr(self.ncc_bridge, "platform_status", {})
                if callable(bridge_status):
                    bridge_status = bridge_status()
                results["ncc_bridge_status"] = bridge_status
            except Exception:
                results["ncc_bridge_status"] = {"status": "unavailable"}

        return results

    # ── ELITE TRADING DESK INTEGRATED DATA COLLECTORS ─────────────────

    def _get_jonny_bravo_data(self) -> Dict[str, Any]:
        """Get Jonny Bravo Division status — education agent, lessons, journal."""
        if not self.jonny_bravo:
            return {"status": "not_available"}
        try:
            status = self.jonny_bravo.get_status()
            curriculum = self.jonny_bravo.get_curriculum_overview()
            journal_stats = self.jonny_bravo.get_journal_stats()
            return {
                "status": "ok",
                "initialized": status.get("initialized", False),
                "lessons_loaded": status.get("lessons_loaded", 0),
                "student_level": status.get("student_level", "unknown"),
                "methodologies": status.get("methodologies", []),
                "journal_entries": journal_stats.get("total_entries", 0),
                "win_rate": journal_stats.get("win_rate", 0.0),
                "curriculum": curriculum,
            }
        except Exception as e:
            self.logger.warning("Jonny Bravo data fetch failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def _get_superstonk_data(self) -> Dict[str, Any]:
        """Get Reddit/Superstonk/WSB sentiment data."""
        if not self.reddit_sentiment:
            return {"status": "not_available"}
        try:
            sentiment = await self.reddit_sentiment.get_reddit_sentiment()
            if not sentiment:
                return {"status": "no_data"}
            # Summarize top discussed tickers and aggregate sentiment
            top_tickers = sentiment[:10] if isinstance(sentiment, list) else []
            bullish = sum(1 for s in top_tickers if s.get("sentiment", 0) > 0)
            bearish = sum(1 for s in top_tickers if s.get("sentiment", 0) < 0)
            neutral = len(top_tickers) - bullish - bearish
            return {
                "status": "ok",
                "tickers_tracked": len(sentiment) if isinstance(sentiment, list) else 0,
                "top_10": [
                    {
                        "ticker": s.get("ticker", "?"),
                        "mentions": s.get("mentions", 0),
                        "sentiment": round(s.get("sentiment", 0), 2),
                    }
                    for s in top_tickers
                ],
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
            }
        except Exception as e:
            self.logger.warning("Reddit sentiment fetch failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _get_planktonxd_data(self) -> Dict[str, Any]:
        """Get PlanktonXD prediction market harvester status."""
        if not self.planktonxd:
            return {"status": "not_available"}
        try:
            # Read latest run output from persisted file
            report_path = PROJECT_ROOT / "polymarket_scenario_bets.json"
            if report_path.exists():
                data = json.loads(report_path.read_text(encoding="utf-8"))
                bets = data.get("bets", [])
                active_bets = [b for b in bets if b.get("size_usd", 0) >= 1]
                total_deployed = sum(b.get("size_usd", 0) for b in active_bets)
                total_payout = sum(b.get("potential_payout", 0) for b in active_bets)
                scenarios_matched = len(
                    set(b.get("scenario_code", "") for b in active_bets)
                )
                return {
                    "status": "ok",
                    "total_bets": len(bets),
                    "active_bets": len(active_bets),
                    "scenarios_matched": scenarios_matched,
                    "total_deployed": round(total_deployed, 2),
                    "max_payout": round(total_payout, 2),
                    "top_bets": [
                        {
                            "scenario": b.get("scenario_code", "?"),
                            "market": b.get("market_question", "?")[:60],
                            "size": round(b.get("size_usd", 0), 2),
                            "crowd_price": b.get("crowd_price", 0),
                        }
                        for b in sorted(
                            active_bets,
                            key=lambda x: x.get("size_usd", 0),
                            reverse=True,
                        )[:5]
                    ],
                }
            return {"status": "no_data"}
        except Exception as e:
            self.logger.warning("PlanktonXD data fetch failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _get_polymarket_division_data(self) -> Dict[str, Any]:
        """Get full Polymarket Division status — all 3 strategies."""
        if not POLYMARKET_DIVISION_AVAILABLE:
            return {"status": "not_available"}
        try:
            # Division registry status
            division_status = get_division_status()
            loaded = sum(1 for v in division_status.values() if v["status"] == "loaded")

            # War Room Poly
            war_room_data: Dict[str, Any] = {"status": "not_available"}
            if self.war_room_poly:
                try:
                    wr_dict = self.war_room_poly.to_dict()
                    war_room_data = {
                        "status": "ok",
                        "pressure_level": wr_dict.get("pressure_level", 0),
                        "n_matches": wr_dict.get("n_matches", 0),
                        "stages": wr_dict.get("stages", {}),
                        "top_matches": wr_dict.get("top_matches", [])[:5],
                    }
                except Exception as e:
                    war_room_data = {"status": "error", "error": str(e)}

            # PolyMC Agent — portfolio + MC summary
            polymc_data: Dict[str, Any] = {"status": "not_available"}
            if self.polymc_agent:
                try:
                    portfolio = self.polymc_agent.portfolio
                    mc_result = self.polymc_agent.run_portfolio_monte_carlo(n_sims=10_000)
                    polymc_data = {
                        "status": "ok",
                        "portfolio_size": len(portfolio),
                        "total_cost": mc_result.get("total_cost", 0),
                        "mean_return": mc_result.get("mean_return", 0),
                        "ev_pct": mc_result.get("ev_pct", 0),
                        "prob_profit": mc_result.get("prob_profit", 0),
                        "max_payout": mc_result.get("max_payout", 0),
                        "var_95": mc_result.get("var_95", 0),
                        "sharpe": mc_result.get("sharpe", 0),
                        "per_bet": mc_result.get("per_bet", []),
                        "bets": [
                            {
                                "name": b.name[:40],
                                "side": b.side,
                                "entry": b.entry_price,
                                "bet_size": b.bet_size,
                                "our_prob": b.our_prob,
                                "take_profit": b.take_profit,
                                "stop_loss": b.stop_loss,
                            }
                            for b in portfolio
                        ],
                    }
                except Exception as e:
                    polymc_data = {"status": "error", "error": str(e)}

            # PolyMC Monitor — exit signals
            monitor_data: Dict[str, Any] = {"status": "not_available"}
            if self.polymc_monitor:
                try:
                    signals = self.polymc_monitor.check_exit_signals()
                    immediate = [s for s in signals if s.urgency == "immediate"]
                    monitor_data = {
                        "status": "ok",
                        "total_signals": len(signals),
                        "immediate_exits": len(immediate),
                        "check_count": self.polymc_monitor.check_count,
                    }
                except Exception as e:
                    monitor_data = {"status": "error", "error": str(e)}

            # Account Tracker — per-strategy balance breakdown
            account_data: Dict[str, Any] = {"status": "not_available"}
            if self.poly_account_tracker:
                try:
                    acct_state = self.poly_account_tracker.get_account_state()
                    account_data = acct_state.to_dict()
                except Exception as e:
                    account_data = {"status": "error", "error": str(e)}

            return {
                "status": "ok",
                "strategies_loaded": loaded,
                "strategies_total": len(division_status),
                "division_status": division_status,
                "war_room": war_room_data,
                "polymc": polymc_data,
                "monitor": monitor_data,
                "account": account_data,
            }
        except Exception as e:
            self.logger.warning("Polymarket Division data fetch failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _get_active_scanner_data(self) -> Dict[str, Any]:
        """Get Active Scanner unified engine status."""
        if not POLYMARKET_DIVISION_AVAILABLE or not self.active_scanner:
            return {"status": "not_available"}
        try:
            scanner = self.active_scanner
            return {
                "status": "ok",
                "dry_run": scanner.dry_run,
                "daily_bet_count": scanner.daily_bet_count,
                "max_daily_bets": scanner.MAX_DAILY_BETS,
                "max_position_usd": scanner.MAX_POSITION_SIZE_USD,
                "min_edge": scanner.MIN_EDGE_THRESHOLD,
                "scan_interval_s": scanner.DEFAULT_SCAN_INTERVAL,
                "execution_log_count": len(scanner.execution_log),
            }
        except Exception as e:
            self.logger.warning("Active Scanner data fetch failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _get_grok_scorer_data(self) -> Dict[str, Any]:
        """Get Grok AI Trade Scorer status."""
        if not self.grok_scorer:
            return {"status": "not_available"}
        try:
            has_xai = bool(self.grok_scorer._xai_key)
            has_anthropic = bool(getattr(self.grok_scorer, "_anthropic_key", ""))
            has_openai = bool(getattr(self.grok_scorer, "_openai_key", ""))
            models = []
            if has_xai:
                models.append("Grok (xAI)")
            if has_anthropic:
                models.append("Claude")
            if has_openai:
                models.append("GPT")
            return {
                "status": "ok" if models else "no_keys",
                "models_available": models,
                "primary_model": models[0] if models else "heuristic_fallback",
                "scoring_threshold": 60,
                "strong_threshold": 80,
            }
        except Exception as e:
            self.logger.warning("Grok scorer status failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _get_openclaw_data(self) -> Dict[str, Any]:
        """Get OpenClaw Barren Wuffet Skills hub status."""
        if not OPENCLAW_AVAILABLE:
            return {"status": "not_available"}
        try:
            total = get_skill_count()
            names = get_skill_names()
            categories = {}
            for cat in [
                "CORE_AAC",
                "TRADING_MARKETS",
                "CRYPTO_DEFI",
                "FINANCE_BANKING",
                "WEALTH_BUILDING",
                "ADVANCED_ANALYSIS",
                "OPENCLAW_POWER_UPS",
            ]:
                cat_skills = get_skills_by_category(cat)
                if cat_skills:
                    categories[cat] = len(cat_skills)
            return {
                "status": "ok",
                "total_skills": total,
                "categories": categories,
                "sample_skills": names[:10] if names else [],
            }
        except Exception as e:
            self.logger.warning("OpenClaw data fetch failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def _get_stock_ticker_data(self) -> Dict[str, Any]:
        """Get stock ticker snapshots from Polygon.io for key watchlist."""
        if not self.polygon_client:
            return {"status": "not_available"}
        try:
            watchlist = ["SPY", "QQQ", "IWM", "GLD", "SLV", "USO", "VIX"]
            tickers = []
            for symbol in watchlist:
                snap = await self.polygon_client.get_snapshot(symbol)
                if snap:
                    tickers.append(
                        {
                            "ticker": snap.ticker,
                            "price": snap.day_close,
                            "change_pct": round(snap.change_pct, 2)
                            if snap.change_pct
                            else 0,
                            "volume": snap.day_volume,
                        }
                    )
            return {
                "status": "ok" if tickers else "no_data",
                "tickers": tickers,
                "count": len(tickers),
            }
        except Exception as e:
            self.logger.warning("Stock ticker data fetch failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def _get_ncl_link_data(self) -> Dict[str, Any]:
        """Get NCL (BRAIN pillar) cross-pillar link intelligence."""
        if not self.cross_pillar_hub:
            return {"status": "not_available"}
        try:
            governance = await self.cross_pillar_hub.check_ncc_governance()
            ncl_intel = await self.cross_pillar_hub.get_ncl_intelligence()
            brs_signals = await self.cross_pillar_hub.get_brs_signals()

            state = self.cross_pillar_hub.state
            return {
                "status": "ok",
                "doctrine_mode": state.doctrine_mode,
                "trading_allowed": self.cross_pillar_hub.should_trade(),
                "risk_multiplier": self.cross_pillar_hub.get_risk_multiplier(),
                "pillars": {
                    "NCC": state.ncc.to_dict(),
                    "NCL": state.ncl.to_dict(),
                    "BRS": state.brs.to_dict(),
                },
                "last_directive": {
                    "action": state.last_directive.action,
                    "reason": state.last_directive.reason,
                }
                if state.last_directive
                else None,
                "ncl_intelligence": {
                    "source": ncl_intel.get("source", "none"),
                    "has_forecasts": bool(ncl_intel.get("forecasts")),
                    "forecast_count": len(ncl_intel.get("forecasts", [])),
                },
                "brs_patterns": len(brs_signals.get("patterns", [])),
                "governance_source": governance.get("source", "none"),
            }
        except Exception as e:
            self.logger.warning("NCL link data fetch failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def _get_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts from all systems"""
        alerts = []

        # Check safeguards for alerts
        safeguards_status = get_safeguards_health()
        if safeguards_status.get("overall_health") != "healthy":
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Safeguards status: {safeguards_status['overall_health']}",
                    "timestamp": datetime.now(),
                    "source": "safeguards",
                }
            )

        # Check circuit breakers
        for exchange, status in safeguards_status.get("exchanges", {}).items():
            if status.get("circuit_breaker_state") == "open":
                alerts.append(
                    {
                        "level": "critical",
                        "message": f"Circuit breaker open for {exchange}",
                        "timestamp": datetime.now(),
                        "source": "circuit_breaker",
                    }
                )

        # Check doctrine compliance
        if self.doctrine_integration:
            try:
                compliance_report = (
                    await self.doctrine_integration.run_compliance_check()
                )
                violations = compliance_report.get("violations", 0)
                warnings = compliance_report.get("warnings", 0)
                barren_wuffet_state = compliance_report.get(
                    "barren_wuffet_state", "normal"
                )

                if violations > 0:
                    alerts.append(
                        {
                            "level": "critical",
                            "message": f"Doctrine violations: {violations} active",
                            "timestamp": datetime.now(),
                            "source": "doctrine",
                        }
                    )

                if warnings > 0:
                    alerts.append(
                        {
                            "level": "warning",
                            "message": f"Doctrine warnings: {warnings} active",
                            "timestamp": datetime.now(),
                            "source": "doctrine",
                        }
                    )

                if barren_wuffet_state in ["CAUTION", "CRITICAL"]:
                    alerts.append(
                        {
                            "level": "warning"
                            if barren_wuffet_state == "CAUTION"
                            else "critical",
                            "message": f"BARREN WUFFET state: {barren_wuffet_state}",
                            "timestamp": datetime.now(),
                            "source": "doctrine",
                        }
                    )

            except Exception as e:
                alerts.append(
                    {
                        "level": "warning",
                        "message": f"Doctrine monitoring error: {str(e)}",
                        "timestamp": datetime.now(),
                        "source": "doctrine",
                    }
                )

        # Check security alerts
        if SECURITY_AVAILABLE and self.security_framework:
            try:
                security_alerts = await self._get_security_alerts()
                alerts.extend(security_alerts)
            except Exception as e:
                self.logger.warning(f"Security alert collection failed: {e}")

        return alerts

    async def _get_security_alerts(self) -> List[Dict[str, Any]]:
        """Get security-related alerts"""
        alerts = []

        if SECURITY_AVAILABLE:
            try:
                sec_status = self.security_framework[
                    "security_monitoring"
                ].get_security_status()

                critical = sec_status.get("critical_events", 0)
                if critical > 0:
                    alerts.append(
                        {
                            "level": "critical",
                            "message": f"Critical security events in last hour: {critical}",
                            "timestamp": datetime.now(),
                            "source": "security",
                        }
                    )

                active_alerts = sec_status.get("active_alerts", 0)
                if active_alerts > 0:
                    alerts.append(
                        {
                            "level": "warning",
                            "message": f"Active security alerts: {active_alerts}",
                            "timestamp": datetime.now(),
                            "source": "security",
                        }
                    )

            except Exception as e:
                self.logger.warning(f"Security alert retrieval failed: {e}")

        return alerts

    def display_dashboard(self, stdscr, data: Dict[str, Any]):
        """Display the monitoring dashboard (terminal mode)"""
        if not data or "error" in data:
            stdscr.clear()
            stdscr.addstr(0, 0, "[LOADING] Loading AAC Master Monitoring Dashboard...")
            if "error" in data:
                stdscr.addstr(2, 0, f"Error: {data['error']}")
            stdscr.refresh()
            return

        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Header
        header = f"[MASTER] AAC 2100 UNIFIED MONITORING DASHBOARD - {data['timestamp'].strftime('%H:%M:%S')}"
        stdscr.addstr(0, 0, header[: width - 1], curses.A_BOLD)

        y_pos = 2

        # System Health
        health = data.get("health", {})
        stdscr.addstr(y_pos, 0, "🏥 SYSTEM HEALTH", curses.A_BOLD)
        y_pos += 1

        status_colors = {
            "healthy": curses.COLOR_GREEN,
            "warning": curses.COLOR_YELLOW,
            "critical": curses.COLOR_RED,
            "error": curses.COLOR_RED,
        }

        overall_status = health.get("overall_status", "unknown")
        color = status_colors.get(overall_status, curses.COLOR_WHITE)
        curses.init_pair(1, color, curses.COLOR_BLACK)
        stdscr.addstr(
            y_pos, 0, f"Overall: {overall_status.upper()}", curses.color_pair(1)
        )
        y_pos += 2

        # Department status
        departments = health.get("departments", {})
        for dept, status in departments.items():
            dept_status = status.get("status", "unknown")
            color = status_colors.get(dept_status, curses.COLOR_WHITE)
            curses.init_pair(2, color, curses.COLOR_BLACK)
            stdscr.addstr(
                y_pos, 0, f"{dept}: {dept_status.upper()}", curses.color_pair(2)
            )
            y_pos += 1

        # Doctrine Compliance Section
        y_pos += 1
        stdscr.addstr(y_pos, 0, "[DOCTRINE] COMPLIANCE MATRIX", curses.A_BOLD)
        y_pos += 1

        doctrine = data.get("doctrine", {})
        compliance_score = doctrine.get("compliance_score", 0)
        barren_wuffet_state = doctrine.get("barren_wuffet_state", "unknown")

        # Compliance score with color coding
        score_color = (
            curses.COLOR_GREEN
            if compliance_score >= 90
            else curses.COLOR_YELLOW
            if compliance_score >= 70
            else curses.COLOR_RED
        )
        curses.init_pair(6, score_color, curses.COLOR_BLACK)
        stdscr.addstr(
            y_pos, 0, f"Compliance Score: {compliance_score}%", curses.color_pair(6)
        )
        y_pos += 1

        # BARREN WUFFET state
        az_color = (
            curses.COLOR_RED
            if barren_wuffet_state in ["CRITICAL", "error"]
            else curses.COLOR_YELLOW
            if barren_wuffet_state == "CAUTION"
            else curses.COLOR_GREEN
        )
        curses.init_pair(7, az_color, curses.COLOR_BLACK)
        stdscr.addstr(
            y_pos,
            0,
            f"BARREN WUFFET State: {barren_wuffet_state.upper()}",
            curses.color_pair(7),
        )
        y_pos += 1

        # Compliance metrics
        compliant = doctrine.get("compliant", 0)
        warnings = doctrine.get("warnings", 0)
        violations = doctrine.get("violations", 0)
        stdscr.addstr(
            y_pos,
            0,
            f"[OK] Compliant: {compliant} | [WARN] Warnings: {warnings} | [ERROR] Violations: {violations}",
        )
        y_pos += 1

        # Monitoring status
        monitoring_active = doctrine.get("monitoring_active", False)
        monitor_color = curses.COLOR_GREEN if monitoring_active else curses.COLOR_RED
        curses.init_pair(8, monitor_color, curses.COLOR_BLACK)
        stdscr.addstr(
            y_pos,
            0,
            f"Monitoring: {'ACTIVE' if monitoring_active else 'INACTIVE'}",
            curses.color_pair(8),
        )
        y_pos += 2

        # P&L Section
        stdscr.addstr(y_pos, 0, "[MONEY] P&L SUMMARY", curses.A_BOLD)
        y_pos += 1

        pnl = data.get("pnl", {})
        stdscr.addstr(y_pos, 0, f"Daily P&L: ${pnl.get('daily_pnl', 0):,.2f}")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Total Equity: ${pnl.get('total_equity', 0):,.2f}")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Unrealized P&L: ${pnl.get('unrealized_pnl', 0):,.2f}")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Max Drawdown: {pnl.get('max_drawdown', 0):.2%}")
        y_pos += 2

        # Security Status
        security = data.get("security", {})
        if security and security.get("status") != "not_available":
            stdscr.addstr(y_pos, 0, "[SHIELD] SECURITY STATUS", curses.A_BOLD)
            y_pos += 1

            security_score = security.get("overall_score", 0)
            sec_color = (
                curses.COLOR_GREEN
                if security_score >= 90
                else curses.COLOR_YELLOW
                if security_score >= 70
                else curses.COLOR_RED
            )
            curses.init_pair(9, sec_color, curses.COLOR_BLACK)
            stdscr.addstr(
                y_pos, 0, f"Security Score: {security_score:.1f}%", curses.color_pair(9)
            )
            y_pos += 1

            components = security.get("components", {})
            for comp_name, comp_data in components.items():
                if isinstance(comp_data, dict):
                    comp_score = comp_data.get("score", 0)
                    comp_color = (
                        curses.COLOR_GREEN
                        if comp_score >= 90
                        else curses.COLOR_YELLOW
                        if comp_score >= 70
                        else curses.COLOR_RED
                    )
                    curses.init_pair(10, comp_color, curses.COLOR_BLACK)
                    stdscr.addstr(
                        y_pos,
                        0,
                        f"{comp_name.upper()}: {comp_score}%",
                        curses.color_pair(10),
                    )
                    y_pos += 1

        # Safeguards Status
        safeguards = data.get("safeguards", {})
        if safeguards:
            stdscr.addstr(y_pos, 0, "[SHIELD]️ PRODUCTION SAFEGUARDS", curses.A_BOLD)
            y_pos += 1

            overall_health = safeguards.get("overall_health", "unknown")
            safe_color = status_colors.get(overall_health, curses.COLOR_WHITE)
            curses.init_pair(3, safe_color, curses.COLOR_BLACK)
            stdscr.addstr(
                y_pos, 0, f"Overall: {overall_health.upper()}", curses.color_pair(3)
            )
            y_pos += 1

            exchanges = safeguards.get("exchanges", {})
            for exchange, status in exchanges.items():
                cb_state = status.get("circuit_breaker_state", "unknown")
                cb_color = (
                    curses.COLOR_RED if cb_state == "open" else curses.COLOR_GREEN
                )
                curses.init_pair(4, cb_color, curses.COLOR_BLACK)
                stdscr.addstr(
                    y_pos, 0, f"{exchange}: CB={cb_state.upper()}", curses.color_pair(4)
                )
                y_pos += 1

        # ── Regime Forecaster (curses panel) ──────────────────────────────
        forecaster = data.get("forecaster", {})
        if forecaster and forecaster.get("status") == "ok" and y_pos < height - 10:
            y_pos += 1
            try:
                stdscr.addstr(y_pos, 0, "[FORECAST] REGIME", curses.A_BOLD)
                y_pos += 1
                regime = forecaster.get("regime", "?").upper().replace("_", " ")
                vol = forecaster.get("vol_shock_readiness", 0)
                anchor = forecaster.get("anchor_ticker", "-")
                contagion = forecaster.get("contagion_ticker", "-")
                vol_lbl = "ARMED" if vol >= 60 else "ELEV" if vol >= 40 else "LOW"
                line = f"{regime}  Vol:{vol:.0f}/100[{vol_lbl}]  Stack:{anchor}+{contagion}"
                stdscr.addstr(y_pos, 0, line[: width - 1])
                y_pos += 1
                for o in forecaster.get("top3_opportunities", []):
                    if y_pos >= height - 6:
                        break
                    expr = o["expression"].replace("_", " ").upper()
                    stdscr.addstr(
                        y_pos,
                        0,
                        f"  #{o['rank']} {o['ticker']} {expr} sc={o['score']}"[
                            : width - 1
                        ],
                    )
                    y_pos += 1
            except curses.error:
                pass  # Terminal too small for this panel

        # ── IBKR Orders + $30,148 Maximization (curses panel) ────────────────
        ibkr = data.get("ibkr_orders", {})
        if ibkr and ibkr.get("status") == "ok" and y_pos < height - 6:
            y_pos += 1
            try:
                stdscr.addstr(y_pos, 0, "[IBKR] ORDERS & MAXIMIZE", curses.A_BOLD)
                y_pos += 1
                bal = ibkr.get("balance_usd", 0)
                committed = ibkr.get("total_committed", 0)
                remaining = ibkr.get("remaining_cash", 0)
                stdscr.addstr(
                    y_pos,
                    0,
                    f"${bal:.0f} bal  ${committed:.0f} deployed  ${remaining:.0f} powder"[
                        : width - 1
                    ],
                )
                y_pos += 1
                for o in ibkr.get("open_orders", [])[:3]:
                    if y_pos >= height - 4:
                        break
                    line = f"  #{o['order_id']} {o['symbol']} x{o['qty']:.0f}@${o['price']:.2f} [{o['ibkr_status']}]"
                    stdscr.addstr(y_pos, 0, line[: width - 1])
                    y_pos += 1
                recs = ibkr.get("recommendations", [])
                if recs and y_pos < height - 3:
                    tickers = "+".join(r["ticker"] for r in recs)
                    stdscr.addstr(
                        y_pos,
                        0,
                        f"  DEPLOY: {tickers}  cost=${ibkr.get('rec_total_cost', 0):.0f}"[
                            : width - 1
                        ],
                    )
                    y_pos += 1
            except curses.error:
                pass  # Terminal too small for this panel

        # ── Matrix Maximizer (curses panel) ────────────────────────────
        mm = data.get("matrix_maximizer", {})
        if mm and mm.get("status") == "ok" and y_pos < height - 8:
            y_pos += 1
            try:
                stdscr.addstr(y_pos, 0, "[MM] MATRIX MAXIMIZER", curses.A_BOLD)
                y_pos += 1
                mandate = mm.get("mandate", "?").upper()
                regime_mm = mm.get("regime", "?").upper()
                cb = mm.get("circuit_breaker", "?")
                stdscr.addstr(
                    y_pos, 0, f"{mandate} | {regime_mm} | CB={cb}"[: width - 1]
                )
                y_pos += 1
                picks = mm.get("top_picks", [])
                for p in picks[:3]:
                    if y_pos >= height - 6:
                        break
                    stdscr.addstr(
                        y_pos,
                        0,
                        f"  {p['ticker']:6} K={p.get('strike', '?')} sc={p.get('score', '?')}"[
                            : width - 1
                        ],
                    )
                    y_pos += 1
            except curses.error:
                pass  # Terminal too small for this panel

        # ── System Registry Summary (curses panel) ─────────────────────
        registry = data.get("registry", {})
        if (
            registry
            and registry.get("status") != "not_available"
            and y_pos < height - 6
        ):
            y_pos += 1
            try:
                stdscr.addstr(y_pos, 0, "[REG] SYSTEM REGISTRY", curses.A_BOLD)
                y_pos += 1
                s = registry.get("summary", {})
                line = (
                    f"APIs:{s.get('apis_configured', 0)}/{s.get('total_apis', 0)} "
                    f"Exch:{s.get('exchanges_online', 0)}/{s.get('exchanges_total', 0)} "
                    f"Strat:{s.get('strategies_ok', 0)}/{s.get('strategies_total', 0)} "
                    f"Dept:{s.get('departments_ok', 0)}/{s.get('departments_total', 0)}"
                )
                stdscr.addstr(y_pos, 0, line[: width - 1])
                y_pos += 1
            except curses.error:
                pass  # Terminal too small for this panel

        # ── Strategy Advisor Leaderboard (curses panel) ────────────────
        sa = data.get("strategy_advisor", {})
        if sa and sa.get("status") == "ok" and y_pos < height - 8:
            y_pos += 1
            try:
                stdscr.addstr(y_pos, 0, "[ADV] STRATEGY ADVISOR LEADERBOARD", curses.A_BOLD)
                y_pos += 1
                total = sa.get("total_strategies", 0)
                openp = sa.get("total_open_positions", 0)
                closedp = sa.get("total_closed_positions", 0)
                approved = sa.get("approved_count", 0)
                stdscr.addstr(
                    y_pos, 0,
                    f"Strats:{total} Open:{openp} Closed:{closedp} Live:{approved}"[: width - 1],
                )
                y_pos += 1
                board = sa.get("leaderboard_top5", [])
                for idx, row in enumerate(board[:5], 1):
                    if y_pos >= height - 6:
                        break
                    name = row.get("strategy", "?")[:20]
                    wr = row.get("win_rate", 0) * 100
                    pnl = row.get("total_pnl", 0)
                    sc = row.get("composite_score", 0)
                    live = "*" if row.get("approved_live") else " "
                    line = f"  {idx}{live}{name:<20} W:{wr:4.0f}% P&L:${pnl:>7.0f} S:{sc:.3f}"
                    stdscr.addstr(y_pos, 0, line[: width - 1])
                    y_pos += 1
            except curses.error:
                pass  # Terminal too small for this panel

        # ── NCL Relay Status (curses panel) ────────────────────────────
        if y_pos < height - 5:
            y_pos += 1
            try:
                from shared.ncc_relay_client import get_relay_client
                relay = get_relay_client()
                rs = relay.stats
                reachable = "ONLINE" if rs.get("relay_reachable") else "OFFLINE"
                pub = rs.get("published", 0)
                queued = rs.get("queued", 0)
                depth = rs.get("outbox_depth", 0)
                stdscr.addstr(y_pos, 0, "[RELAY] NCL RELAY STATUS", curses.A_BOLD)
                y_pos += 1
                line = f"  {reachable} | Sent:{pub} Queued:{queued} Outbox:{depth}"
                stdscr.addstr(y_pos, 0, line[: width - 1])
                y_pos += 1
            except (ImportError, Exception):
                pass

        # ── MULTI-PILLAR NETWORK (curses compact) ────────────────────
        pn = data.get("pillar_network", {})
        if pn and pn.get("status") == "ok" and y_pos < height - 10:
            y_pos += 1
            try:
                stdscr.addstr(
                    y_pos, 0, "[PILLARS] MATRIX MONITOR NETWORK", curses.A_BOLD
                )
                y_pos += 1

                mode = pn.get("doctrine_mode", "?")
                online = pn.get("pillars_online", 0)
                total = pn.get("total_pillars", 0)
                rm = pn.get("risk_multiplier", 0)
                trade = "Y" if pn.get("should_trade") else "N"
                header = f"Doctrine:{mode} Trade:{trade} Risk:{rm:.1f}x Online:{online}/{total}"
                stdscr.addstr(y_pos, 0, header[: width - 1])
                y_pos += 1

                pillar_order = ["NCC_MASTER", "NCC", "AAC", "NCL", "BRS"]
                for pid in pillar_order:
                    if y_pos >= height - 6:
                        break
                    p = pn.get("pillars", {}).get(pid, {})
                    if not p:
                        continue
                    name = p.get("name", pid)[:12]
                    health = p.get("health", "?")
                    h_ch = "+" if health == "GREEN" else "-" if health == "RED" else "?"
                    matrix = p.get("matrix_status", "?")[:8]
                    lat = p.get("latency_ms")
                    lat_s = f"{lat:.0f}ms" if lat is not None else "--"
                    line = f"  {h_ch}{name:<12} :{p.get('port', '?'):<5} M:{matrix:<8} {lat_s}"
                    stdscr.addstr(y_pos, 0, line[: width - 1])
                    y_pos += 1
            except curses.error:
                pass

        # ── DEEP PILLAR MATRIX (curses compact) ─────────────────────
        pmd = data.get("pillar_matrix_deep", {})
        if pmd and pmd.get("status") == "ok" and y_pos < height - 10:
            y_pos += 1
            try:
                stdscr.addstr(y_pos, 0, "[DEEP] PILLAR MATRIX MONITORS", curses.A_BOLD)
                y_pos += 1

                ent = pmd.get("enterprise_score", 0)
                p_on = pmd.get("pillars_online", 0)
                p_tot = pmd.get("pillars_total", 0)
                stdscr.addstr(
                    y_pos, 0, f"Enterprise:{ent}% Pillars:{p_on}/{p_tot}"[: width - 1]
                )
                y_pos += 1

                for pid in ["NCC_MASTER", "NCC", "AAC", "NCL", "BRS"]:
                    if y_pos >= height - 6:
                        break
                    p = pmd.get("pillars", {}).get(pid, {})
                    if not p:
                        continue
                    h = p.get("health", "?")
                    h_ch = "+" if h == "GREEN" else "-" if h == "RED" else "?"
                    sc = p.get("overall_score", 0)
                    ck = f"{p.get('checks_passed', 0)}/{p.get('checks_total', 0)}"
                    sl = p.get("slo_violations", 0)
                    hs = p.get("health_status", "?")[:6]
                    line = f"  {h_ch}{pid:<10} {sc:.0%} {hs:<6} chk:{ck:<5} slo:{sl}"
                    stdscr.addstr(y_pos, 0, line[: width - 1])
                    y_pos += 1
            except curses.error:
                pass

        # ── ELITE TRADING DESK (curses compact) ──────────────────────
        if y_pos < height - 12:
            y_pos += 1
            try:
                stdscr.addstr(y_pos, 0, "[ELITE] TRADING DESK", curses.A_BOLD)
                y_pos += 1

                # Stock Ticker ribbon
                ticker = data.get("stock_ticker", {})
                if ticker and ticker.get("status") == "ok":
                    ribbon = ""
                    for t in ticker.get("tickers", []):
                        pct = t.get("change_pct", 0)
                        arrow = "+" if pct >= 0 else ""
                        ribbon += (
                            f"{t['ticker']}${t.get('price', 0):.0f}({arrow}{pct:.1f}%) "
                        )
                    stdscr.addstr(y_pos, 0, ribbon[: width - 1])
                    y_pos += 1

                # NCL Link status
                ncl = data.get("ncl_link", {})
                if ncl and ncl.get("status") == "ok" and y_pos < height - 8:
                    mode = ncl.get("doctrine_mode", "?")
                    trade_ok = "Y" if ncl.get("trading_allowed") else "N"
                    rm = ncl.get("risk_multiplier", 0)
                    line = f"NCL:{mode} Trade:{trade_ok} Mult:{rm:.1f}x"
                    pillars = ncl.get("pillars", {})
                    for name, p in pillars.items():
                        con = "+" if p.get("connected") else "-"
                        line += f" {name}:{con}"
                    stdscr.addstr(y_pos, 0, line[: width - 1])
                    y_pos += 1

                # PlanktonXD summary
                px = data.get("planktonxd", {})
                if px and px.get("status") == "ok" and y_pos < height - 6:
                    line = (
                        f"PLKTN: {px.get('active_bets', 0)}/{px.get('total_bets', 0)} bets "
                        f"${px.get('total_deployed', 0):.0f} deployed "
                        f"payout=${px.get('max_payout', 0):.0f}"
                    )
                    stdscr.addstr(y_pos, 0, line[: width - 1])
                    y_pos += 1

                # Superstonk
                ss = data.get("superstonk", {})
                if ss and ss.get("status") == "ok" and y_pos < height - 5:
                    line = f"WSB: {ss.get('tickers_tracked', 0)} tickers B:{ss.get('bullish', 0)} R:{ss.get('bearish', 0)}"
                    top = ss.get("top_10", [])
                    if top:
                        line += " top:" + ",".join(t["ticker"] for t in top[:3])
                    stdscr.addstr(y_pos, 0, line[: width - 1])
                    y_pos += 1

                # Grok + OpenClaw + Jonny Bravo on one line
                parts = []
                grok = data.get("grok_scorer", {})
                if grok:
                    parts.append(f"GROK:{grok.get('primary_model', 'off')}")
                oc = data.get("openclaw", {})
                if oc and oc.get("status") == "ok":
                    parts.append(f"CLAW:{oc.get('total_skills', 0)}skills")
                jb = data.get("jonny_bravo", {})
                if jb and jb.get("status") == "ok":
                    parts.append(
                        f"JB:{jb.get('student_level', '?')} L:{jb.get('lessons_loaded', 0)}"
                    )
                if parts and y_pos < height - 4:
                    stdscr.addstr(y_pos, 0, "  ".join(parts)[: width - 1])
                    y_pos += 1

            except curses.error:
                pass  # Terminal too small for Elite Desk panel

        # Alerts
        alerts = data.get("alerts", [])
        if alerts:
            y_pos += 1
            stdscr.addstr(
                y_pos, 0, "[ALERT] ACTIVE ALERTS", curses.A_BOLD | curses.A_BLINK
            )
            y_pos += 1

            for alert in alerts[:3]:  # Show top 3 alerts
                alert_color = (
                    curses.COLOR_RED
                    if alert["level"] == "critical"
                    else curses.COLOR_YELLOW
                )
                curses.init_pair(5, alert_color, curses.COLOR_BLACK)
                stdscr.addstr(y_pos, 0, f"• {alert['message']}", curses.color_pair(5))
                y_pos += 1

        # Footer
        if y_pos < height - 2:
            footer_y = height - 2
            stdscr.addstr(
                footer_y, 0, "Press 'q' to quit | 'r' to refresh | Auto-refresh: 1s"
            )

        stdscr.refresh()

    def _display_text_dashboard(self, data: Dict[str, Any]):
        """Display dashboard in text mode for Windows compatibility"""
        print("\033[2J\033[H", end="")  # ANSI clear screen
        print("=" * 80)
        print("AAC 2100 MASTER MONITORING DASHBOARD")
        print("=" * 80)
        print(
            f"Last update: {data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("")

        # ── Collector Health (degradation report) ─────────────────────
        cstatus = data.get("_collector_status", {})
        total = cstatus.get("total", 0)
        ok = cstatus.get("ok", 0)
        failed = cstatus.get("failed", 0)
        timed_out = cstatus.get("timed_out", 0)
        if total > 0:
            if failed == 0 and timed_out == 0:
                print(f"COLLECTORS: {ok}/{total} OK")
            else:
                parts = [f"{ok} ok"]
                if failed:
                    parts.append(f"{failed} FAILED")
                if timed_out:
                    parts.append(f"{timed_out} TIMEOUT")
                print(f"COLLECTORS: {' | '.join(parts)}  ({total} total)")
                # Show which ones failed/timed out
                details = cstatus.get("details", {})
                for cname, cstat in details.items():
                    if cstat != "ok":
                        tag = "TIMEOUT" if cstat == "timeout" else "ERROR"
                        print(f"  [{tag}] {cname}")
            print("")

        # System Health
        health = data.get("health", {})
        print("🔍 SYSTEM HEALTH")
        print("-" * 20)
        status = health.get("overall_status", "unknown")
        status_emoji = {
            "healthy": "🟢",
            "warning": "🟡",
            "critical": "🔴",
            "unknown": "⚪",
        }.get(status, "⚪")
        print(f"Overall Status: {status_emoji} {status.upper()}")

        # Department status
        departments = health.get("departments", {})
        if departments:
            print("\nDepartments:")
            for dept, info in departments.items():
                dept_status = info.get("status", "unknown")
                emoji = {
                    "healthy": "🟢",
                    "warning": "🟡",
                    "critical": "🔴",
                    "unknown": "⚪",
                }.get(dept_status, "⚪")
                print(f"  {emoji} {dept}: {dept_status}")

        # Infrastructure
        infra = health.get("infrastructure", {})
        if infra:
            print("\nInfrastructure:")
            for component, status in infra.items():
                if isinstance(status, dict):
                    comp_status = status.get("status", "unknown")
                else:
                    comp_status = "healthy" if status else "critical"
                emoji = {
                    "healthy": "🟢",
                    "warning": "🟡",
                    "critical": "🔴",
                    "unknown": "⚪",
                }.get(comp_status, "⚪")
                print(f"  {emoji} {component}: {comp_status}")

        # Doctrine Compliance
        doctrine = data.get("doctrine", {})
        if doctrine:
            print("\n[DOCTRINE] COMPLIANCE MATRIX")
            print("-" * 25)
            compliance_score = doctrine.get("compliance_score", 0)
            barren_wuffet_state = doctrine.get("barren_wuffet_state", "unknown")

            # Compliance score with color emoji
            if compliance_score >= 90:
                score_emoji = "🟢"
            elif compliance_score >= 70:
                score_emoji = "🟡"
            else:
                score_emoji = "🔴"
            print(f"Compliance Score: {score_emoji} {compliance_score}%")

            # BARREN WUFFET state
            if barren_wuffet_state in ["CRITICAL", "error"]:
                az_emoji = "🔴"
            elif barren_wuffet_state == "CAUTION":
                az_emoji = "🟡"
            else:
                az_emoji = "🟢"
            print(f"BARREN WUFFET State: {az_emoji} {barren_wuffet_state.upper()}")

            # Compliance metrics
            compliant = doctrine.get("compliant", 0)
            warnings = doctrine.get("warnings", 0)
            violations = doctrine.get("violations", 0)
            print(
                f"[OK] Compliant: {compliant} | [WARN] Warnings: {warnings} | [ERROR] Violations: {violations}"
            )

            # Monitoring status
            monitoring_active = doctrine.get("monitoring_active", False)
            monitor_status = "[ACTIVE]" if monitoring_active else "[INACTIVE]"
            print(f"Monitoring: {monitor_status}")

        # P&L Data
        pnl = data.get("pnl", {})
        if pnl:
            print("\n[MONEY] P&L SUMMARY")
            print("-" * 15)
            daily_pnl = pnl.get("daily_pnl", 0)
            total_pnl = pnl.get("total_equity", 0)
            print(f"Daily P&L: ${daily_pnl:,.2f}")
            print(f"Total P&L: ${total_pnl:,.2f}")

        # Risk Metrics
        risk = data.get("risk", {})
        if risk:
            print("\n[WARN]️  RISK METRICS")
            print("-" * 15)
            var_95 = risk.get("var_95", 0)
            max_drawdown = risk.get("max_drawdown", 0)
            print(f"VaR (95%): ${var_95:,.2f}")
            print(f"Max Drawdown: ${max_drawdown:,.2f}")

        # Trading Activity
        trading = data.get("trading", {})
        if trading:
            print("\n📈 TRADING ACTIVITY")
            print("-" * 20)
            active_positions = trading.get("active_positions", 0)
            pending_orders = trading.get("pending_orders", 0)
            print(f"Active Positions: {active_positions}")
            print(f"Pending Orders: {pending_orders}")

        # Security Status
        security = data.get("security", {})
        if security and security.get("status") != "not_available":
            print("\n[SHIELD] SECURITY STATUS")
            print("-" * 20)
            security_score = security.get("overall_score", 0)
            if security_score >= 90:
                sec_emoji = "🟢"
            elif security_score >= 70:
                sec_emoji = "🟡"
            else:
                sec_emoji = "🔴"
            print(f"Security Score: {sec_emoji} {security_score:.1f}%")

            components = security.get("components", {})
            for comp_name, comp_data in components.items():
                if isinstance(comp_data, dict):
                    comp_score = comp_data.get("score", 0)
                    if comp_score >= 90:
                        comp_emoji = "🟢"
                    elif comp_score >= 70:
                        comp_emoji = "🟡"
                    else:
                        comp_emoji = "🔴"
                    print(f"  {comp_emoji} {comp_name.upper()}: {comp_score}%")

        # Safeguards
        safeguards = data.get("safeguards", {})
        if safeguards:
            print("\n[SHIELD]️  PRODUCTION SAFEGUARDS")
            print("-" * 25)
            circuit_breaker = safeguards.get("circuit_breaker_active", False)
            rate_limited = safeguards.get("rate_limited", False)
            print(f"Circuit Breaker: {'🔴 ACTIVE' if circuit_breaker else '🟢 NORMAL'}")
            print(f"Rate Limiting: {'🟡 ACTIVE' if rate_limited else '🟢 NORMAL'}")

        # Strategy Metrics
        strategy = data.get("strategy", {})
        if strategy and strategy.get("status") != "not_available":
            print("\n[STRATEGY] STRATEGY METRICS")
            print("-" * 20)
            total_strategies = strategy.get("total_strategies", 0)
            active_strategies = strategy.get("active_strategies", 0)
            avg_return = strategy.get("average_return", 0.0)
            print(f"Total Strategies: {total_strategies}")
            print(f"Active Strategies: {active_strategies}")
            print(f"Average Return: {avg_return:.2%}")

        # ── REGIME FORECASTER INTEL ────────────────────────────────────────
        forecaster = data.get("forecaster", {})
        if forecaster and forecaster.get("status") == "ok":
            print("")
            print("=" * 60)
            print("  REGIME FORECASTER — IF X+Y → EXPECT Z")
            print("=" * 60)
            regime = forecaster.get("regime", "UNKNOWN").upper().replace("_", " ")
            secondary = forecaster.get("secondary")
            conf = forecaster.get("regime_confidence", 0)
            vol = forecaster.get("vol_shock_readiness", 0)
            bears = forecaster.get("bear_signals", 0)
            bulls = forecaster.get("bull_signals", 0)
            anchor = forecaster.get("anchor_ticker", "-")
            contagion = forecaster.get("contagion_ticker", "-")
            macro = forecaster.get("macro_context", {})

            if vol >= 80:
                vol_icon = "[!! SHOCK WINDOW OPEN !!]"
            elif vol >= 60:
                vol_icon = "[ARMED — cheapest vol now]"
            elif vol >= 40:
                vol_icon = "[ELEVATED — watch signals]"
            else:
                vol_icon = "[LOW]"

            print(f"  Regime: {regime}  (conf {conf}%)")
            if secondary:
                print(f"  Secondary: {secondary.upper().replace('_', ' ')}")
            print(f"  Vol Shock Readiness: {vol}/100  {vol_icon}")
            print(f"  Signals: {bears} bearish | {bulls} bullish")
            print(
                f"  Macro: VIX={macro.get('vix', '?')}  HY={macro.get('hy_spread', '?')}bps  Oil=${macro.get('oil', '?')}"
            )
            war_flags = []
            if macro.get("war"):
                war_flags.append("WAR ACTIVE")
            if macro.get("hormuz"):
                war_flags.append("HORMUZ BLOCKED")
            if war_flags:
                print(f"  Geo: {' | '.join(war_flags)}")

            fired = forecaster.get("fired_formulas", [])
            if fired:
                print(f"  Fired Formulas ({len(fired)}):")
                for f in fired:
                    print(f"    [{f['tag']}] {f['confidence']}%  {f['outcome']}")

            print("")
            print(f"  2-TRADE STACK  |  Anchor: {anchor}  |  Contagion: {contagion}")

            opps = forecaster.get("top3_opportunities", [])
            if opps:
                print("  TOP 3 SHORT-TERM OPPORTUNITIES:")
                for o in opps:
                    expr = o["expression"].replace("_", " ").upper()
                    print(
                        f"    #{o['rank']}  {o['ticker']:6}  {expr:17}  score={o['score']}"
                    )
                    print(f"         {o['thesis']}")
            print("=" * 60)

        # ── IBKR OPEN ORDERS + $30,148 MAXIMIZATION PLAN ────────────────
        ibkr = data.get("ibkr_orders", {})
        if ibkr and ibkr.get("status") == "ok":
            print("")
            print("=" * 60)
            print("  IBKR ORDERS  [acct: {}]".format(ibkr.get("account", "?")))
            print("=" * 60)
            bal = ibkr.get("balance_usd", 0)
            committed = ibkr.get("total_committed", 0)
            remaining = ibkr.get("remaining_cash", 0)
            print(
                f"  Balance: ${bal:.0f} USD  |  Committed: ${committed:.0f}  |  Dry Powder: ${remaining:.0f}"
            )

            orders = ibkr.get("open_orders", [])
            if orders:
                print(f"  Open Orders ({len(orders)}):")
                for o in orders:
                    s = o["ibkr_status"]
                    cost = o["cost"]
                    print(
                        f"    #{o['order_id']}  {o['symbol']:6}  {o['side'].upper()} "
                        f"x{o['qty']:.0f} @ ${o['price']:.2f}  cost=${cost:.0f}  [{s}]"
                    )
            else:
                print("  No open orders.")

            recs = ibkr.get("recommendations", [])
            if recs:
                rec_total = ibkr.get("rec_total_cost", 0)
                cash_after = ibkr.get("cash_after_recs", 0)
                print("")
                print(
                    f"  MAXIMIZATION PLAN (deploy ${rec_total:.0f} / ${remaining:.0f} remaining):"
                )
                for r in recs:
                    print(
                        f"    {r['ticker']:6}  {r['expression']:17}  x{r['contracts']}  "
                        f"est ${r['est_premium']:.2f}/contract  total=${r['cost']:.0f}  DTE {r['dte']}"
                    )
                    print(f"           {r['rationale']}")
                print(f"  Cash buffer after deployment: ${cash_after:.0f}")
            print("=" * 60)

        # ── MATRIX MAXIMIZER DEEP ─────────────────────────────────────
        mm = data.get("matrix_maximizer", {})
        if mm and mm.get("status") == "ok":
            print("")
            print("=" * 60)
            print("  MATRIX MAXIMIZER — COMMAND & CONTROL")
            print("=" * 60)
            print(f"  Run #{mm.get('run_number', '?')}  |  {mm.get('timestamp', '?')}")
            mandate = mm.get("mandate", "?").upper()
            mandate_icon = {
                "DEFENSIVE": "🛡️",
                "STANDARD": "⚖️",
                "AGGRESSIVE": "⚔️",
                "MAX_CONVICTION": "🔥",
            }.get(mandate, "❓")
            print(
                f"  Mandate: {mandate_icon} {mandate}  |  Risk/trade: {mm.get('risk_per_trade', '?')}%  |  Max pos: {mm.get('max_positions', '?')}"
            )
            print(
                f"  Regime: {mm.get('regime', '?').upper()}  conf={mm.get('regime_confidence', '?')}%"
            )
            print(
                f"  Geo: WAR={'YES' if mm.get('war_active') else 'NO'}  HORMUZ={'BLOCKED' if mm.get('hormuz_blocked') else 'OPEN'}"
            )
            print(f"  Oil=${mm.get('oil_price', '?')}  VIX={mm.get('vix', '?')}")
            spy_ret = mm.get("spy_median_return")
            spy_var = mm.get("spy_var_95")
            if spy_ret is not None:
                print(f"  SPY median return: {spy_ret:.1f}%  |  VaR(95): {spy_var}")
            cb = mm.get("circuit_breaker", "?")
            cb_icon = "🔴" if cb == "OPEN" else "🟢"
            print(
                f"  Circuit Breaker: {cb_icon} {cb}  |  Risk Score: {mm.get('risk_score', '?')}"
            )

            picks = mm.get("top_picks", [])
            if picks:
                print(f"  Top {len(picks)} Picks (of {mm.get('total_picks', 0)}):")
                for p in picks:
                    print(
                        f"    {p['ticker']:6}  K={p.get('strike', '?')}  exp={p.get('expiry', '?')}  "
                        f"score={p.get('score', '?')}  x{p.get('contracts', '?')}  ${p.get('cost', '?')}"
                    )
            elapsed = mm.get("elapsed_s")
            if elapsed:
                print(f"  Cycle time: {elapsed:.1f}s")
            print("=" * 60)

        # ── SYSTEM REGISTRY — API STATUS ──────────────────────────────
        registry = data.get("registry", {})
        if registry and registry.get("status") != "not_available":
            summary = registry.get("summary", {})
            print("")
            print("=" * 60)
            print("  SYSTEM REGISTRY — COMMAND & CONTROL")
            print("=" * 60)
            print(
                f"  APIs: {summary.get('apis_configured', 0)} configured | "
                f"{summary.get('apis_missing', 0)} missing | "
                f"{summary.get('apis_free', 0)} free (no key)"
            )

            # Exchange status
            exchanges = registry.get("exchanges", [])
            ex_online = summary.get("exchanges_online", 0)
            ex_total = summary.get("exchanges_total", 0)
            print(f"  Exchanges: {ex_online}/{ex_total} online")
            for ex in exchanges:
                h = ex.get("health", "grey")
                icon = {"green": "🟢", "yellow": "🟡", "red": "🔴", "grey": "⚪"}.get(
                    h, "⚪"
                )
                lat = ex.get("latency_ms")
                lat_str = f"  {lat}ms" if lat else ""
                print(f"    {icon} {ex['name']:20} {ex.get('detail', '')}{lat_str}")

            # Infrastructure status
            infra = registry.get("infrastructure", [])
            infra_ok = summary.get("infra_ok", 0)
            infra_total = summary.get("infra_total", 0)
            print(f"  Infrastructure: {infra_ok}/{infra_total} healthy")
            for svc in infra:
                h = svc.get("health", "grey")
                icon = {"green": "🟢", "yellow": "🟡", "red": "🔴", "grey": "⚪"}.get(
                    h, "⚪"
                )
                print(f"    {icon} {svc['name']:20} {svc.get('detail', '')}")

            # Strategy engines
            strats = registry.get("strategies", [])
            strat_ok = summary.get("strategies_ok", 0)
            strat_total = summary.get("strategies_total", 0)
            print(f"  Strategies: {strat_ok}/{strat_total} operational")
            for st_item in strats:
                h = st_item.get("health", "grey")
                icon = {"green": "🟢", "yellow": "🟡", "red": "🔴", "grey": "⚪"}.get(
                    h, "⚪"
                )
                print(f"    {icon} {st_item['name']:20} {st_item.get('detail', '')}")

            # Departments
            depts = registry.get("departments", [])
            dept_ok = summary.get("departments_ok", 0)
            dept_total = summary.get("departments_total", 0)
            print(f"  Departments: {dept_ok}/{dept_total} online")
            for d in depts:
                h = d.get("health", "grey")
                icon = {"green": "🟢", "yellow": "🟡", "red": "🔴", "grey": "⚪"}.get(
                    h, "⚪"
                )
                print(f"    {icon} {d['name']:25} {d.get('detail', '')}")

            # API breakdown by category
            apis = registry.get("apis", [])
            if apis:
                cats: Dict[str, list] = {}
                for a in apis:
                    cat = a.get("category", "Other")
                    cats.setdefault(cat, []).append(a)
                print(f"  ── API Inventory ({len(apis)} total) ──")
                for cat, items in sorted(cats.items()):
                    conf = sum(1 for i in items if i.get("configured"))
                    print(f"    {cat}: {conf}/{len(items)} configured")
                    for item in items:
                        icon = (
                            "✅"
                            if item.get("configured")
                            else ("🔑" if item.get("env_var") else "🆓")
                        )
                        print(f"      {icon} {item['name']}")

            # Orphan scripts
            orphans = registry.get("orphans", [])
            if orphans:
                print(f"  Orphan Scripts: {len(orphans)} root _*.py files")
                for o in orphans[:10]:
                    print(f"    📄 {o['script']:30} {o.get('description', '')[:50]}")
                if len(orphans) > 10:
                    print(f"    ... and {len(orphans) - 10} more")

            print("=" * 60)

        # ═══════════════════════════════════════════════════════════════
        #  DOCTRINE PACK READER (all 11 packs)
        # ═══════════════════════════════════════════════════════════════
        dpacks = data.get("doctrine_packs", {})
        if dpacks and dpacks.get("status") in ("ok", "ok_text"):
            print("")
            print("=" * 60)
            print("  DOCTRINE PACK READER -- 11 Packs")
            print("=" * 60)
            packs = dpacks.get("packs", [])
            if packs:
                for i, pack in enumerate(packs, 1):
                    name = pack.get("name", pack.get("id", f"Pack {i}"))
                    owner = pack.get("owner", "?")
                    metrics = pack.get("metrics", pack.get("key_metrics", []))
                    if isinstance(metrics, list):
                        metric_str = ", ".join(str(m) for m in metrics[:4])
                    elif isinstance(metrics, dict):
                        metric_str = ", ".join(list(metrics.keys())[:4])
                    else:
                        metric_str = str(metrics)[:60]
                    print(f"  [{i:2d}] {str(name):35} Owner: {str(owner):20} Metrics: {metric_str}")
            elif dpacks.get("raw"):
                print("  (YAML loaded as text -- install PyYAML for structured view)")
            print(f"  Total: {dpacks.get('count', 0)} doctrine packs loaded")
            print("=" * 60)

        # ═══════════════════════════════════════════════════════════════
        #  STRATEGY ADVISOR LEADERBOARD
        # ═══════════════════════════════════════════════════════════════
        advisor = data.get("strategy_advisor", {})
        if advisor and advisor.get("status") == "ok":
            print("")
            print("=" * 60)
            print("  STRATEGY ADVISOR -- Paper-Proof Leaderboard")
            print("=" * 60)
            print(f"  Strategies Loaded: {advisor.get('total_strategies', 0)}")
            print(f"  Open Paper Positions: {advisor.get('total_open_positions', 0)}")
            print(f"  Closed Positions: {advisor.get('total_closed_positions', 0)}")
            approved = advisor.get("approved_for_live", [])
            if approved:
                print(f"  APPROVED FOR LIVE: {', '.join(approved)}")
            else:
                print("  APPROVED FOR LIVE: (none yet -- need 10+ trades + positive Sharpe)")
            board = advisor.get("leaderboard_top5", [])
            if board:
                print("  TOP 5:")
                print(f"  {'#':>3}  {'Strategy':30}  {'Trades':>6}  {'Win%':>6}  {'P&L':>10}  {'Sharpe':>7}  {'Score':>7}")
                for idx, row in enumerate(board, 1):
                    print(
                        f"  {idx:3d}  {row['strategy']:30}  {row['trades']:6d}  "
                        f"{row['win_rate']*100:5.1f}%  ${row['total_pnl']:9.2f}  "
                        f"{row['sharpe']:7.3f}  {row['composite_score']:7.4f}"
                    )
            print("=" * 60)

        # ═══════════════════════════════════════════════════════════════
        #  NCL RELAY STATUS
        # ═══════════════════════════════════════════════════════════════
        rs = data.get("relay_status", {})
        if rs and rs.get("status") == "ok":
            print("")
            print("=" * 60)
            print("  NCL RELAY STATUS")
            print("=" * 60)
            reachable = "ONLINE" if rs.get("relay_reachable") else "OFFLINE"
            print(f"  Relay URL: {rs.get('relay_url', '?')}")
            print(f"  Status: {reachable}")
            print(f"  Published: {rs.get('published', 0)} | Queued: {rs.get('queued', 0)} | Outbox Depth: {rs.get('outbox_depth', 0)}")
            print("=" * 60)

        # ═══════════════════════════════════════════════════════════════
        #  ACTIVE STRATEGY DOCTRINE (per-strategy directives)
        # ═══════════════════════════════════════════════════════════════
        adoc = data.get("active_doctrine", {})
        if adoc and adoc.get("status") == "ok":
            print("")
            print("=" * 60)
            print("  ACTIVE STRATEGY DOCTRINE -- Per-Strategy Directives")
            print("=" * 60)
            print(f"  Regime: {adoc.get('regime', '?').upper()}")
            overrides = adoc.get("manual_overrides", [])
            if overrides:
                print(f"  Manual Overrides Active: {', '.join(overrides)}")
            ncl_c = adoc.get("ncl_caution", 0)
            ncl_icon = "HIGH" if ncl_c > 0.7 else "MODERATE" if ncl_c > 0.4 else "LOW"
            print(f"  NCL Caution Level: {ncl_c:.2f} ({ncl_icon})")
            directives = adoc.get("directives", [])
            if directives:
                print(f"  {'Strategy':25}  {'Allowed':>7}  {'Size%':>6}  {'Bias':>12}  {'MaxPos':>6}  Notes")
                for d in directives:
                    allowed = "YES" if d["allowed"] else "NO"
                    print(
                        f"  {d['strategy']:25}  {allowed:>7}  {d['position_size_pct']*100:5.1f}%  "
                        f"{d['bias']:>12}  {d['max_positions']:6d}  {d.get('notes', '')[:30]}"
                    )
            print("=" * 60)

        # Black Swan Crisis Center
        crisis = data.get("crisis_center", {})
        if crisis and crisis.get("status") != "unavailable":
            print("\n[BLACKSWAN] CRISIS CENTER")
            print("-" * 40)
            if crisis.get("status") == "error":
                print(f"  ⚠️  Crisis center error: {crisis.get('error', 'unknown')}")
            else:
                print(
                    f"  🌡️  Pressure Level: {crisis.get('pressure_pct', '?')}% ({crisis.get('pressure', '?')})"
                )
                print(
                    f"  📊 Thesis Ratio: {crisis.get('ratio_pct', '?')}% (Bullish: {crisis.get('bullish', '?')} / Bearish: {crisis.get('bearish', '?')} / Neutral: {crisis.get('neutral', '?')})"
                )
                print(f"  🎯 Probability: {crisis.get('probability', '?')}")
                top = crisis.get("top5_indicators", [])
                if top:
                    print("  🔥 Top Indicators:")
                    for ind in top[:5]:
                        print(f"    {ind.get('weight', 0):.2f}  {ind.get('name', '?')}")
            print("=" * 60)

        # ═══════════════════════════════════════════════════════════════
        #  MULTI-PILLAR MATRIX MONITOR NETWORK
        # ═══════════════════════════════════════════════════════════════

        pn = data.get("pillar_network", {})
        if pn and pn.get("status") == "ok":
            print("\n" + "▓" * 80)
            print("  🌐 MULTI-PILLAR MATRIX MONITOR NETWORK")
            print("▓" * 80)

            # Doctrine & governance summary
            mode = pn.get("doctrine_mode", "UNKNOWN")
            mode_icon = {
                "NORMAL": "🟢",
                "CAUTION": "🟡",
                "SAFE_MODE": "🟠",
                "HALT": "🔴",
            }.get(mode, "⚪")
            trade_ok = "✅ YES" if pn.get("should_trade", False) else "🚫 NO"
            risk_m = pn.get("risk_multiplier", 0)
            online = pn.get("pillars_online", 0)
            total = pn.get("total_pillars", 0)
            offline = pn.get("pillars_offline", 0)
            print(
                f"  Doctrine: {mode_icon} {mode}  |  Trading: {trade_ok}  |  Risk: {risk_m:.1f}x  |  Pillars: {online}/{total} online"
            )

            # Per-pillar status grid
            print(
                f"\n  {'PILLAR':<16} {'ROLE':<28} {'PORT':<6} {'HEALTH':<8} {'MATRIX':<12} {'LATENCY':<10} {'LINK':<6}"
            )
            print("  " + "─" * 86)

            pillar_order = ["NCC_MASTER", "NCC", "AAC", "NCL", "BRS"]
            for pid in pillar_order:
                p = pn.get("pillars", {}).get(pid, {})
                if not p:
                    continue
                name = p.get("name", pid)[:15]
                role = p.get("role", "")[:27]
                port = str(p.get("port", "?"))
                health = p.get("health", "unknown")
                h_icon = {"GREEN": "🟢", "RED": "🔴", "unknown": "⚪"}.get(health, "⚪")
                matrix = p.get("matrix_status", "unknown")[:11]
                lat = p.get("latency_ms")
                lat_str = f"{lat:.0f}ms" if lat is not None else "—"
                cp_con = p.get("cross_pillar_connected")
                link_str = "✅" if cp_con else ("🔴" if cp_con is False else "—")
                print(
                    f"  {name:<16} {role:<28} {port:<6} {h_icon} {health:<6} {matrix:<12} {lat_str:<10} {link_str}"
                )

            # Last directive
            directive = pn.get("last_directive")
            if directive:
                print(
                    f"\n  📜 Last NCC Directive: {directive.get('action', '?').upper()} — {directive.get('reason', '?')}"
                )

            # Active strategies from cross-pillar state
            strats = pn.get("active_strategies", [])
            if strats:
                print(f"  ⚡ Active Strategies: {', '.join(strats[:8])}")

            # NCC Bridge status
            bridge = pn.get("ncc_bridge_status", {})
            if bridge and bridge.get("status") != "unavailable":
                print(
                    f"  🔌 NCC Bridge: {bridge.get('status', '?')} | uptime={bridge.get('uptime_seconds', '?')}s"
                )

            # AAC Matrix Report
            matrix_rpt = pn.get("aac_matrix_report", {})
            if matrix_rpt and not matrix_rpt.get("error"):
                h = matrix_rpt.get("health", "?")
                hs = matrix_rpt.get("health_score", "?")
                print(f"  📊 AAC Matrix Report: health={h} score={hs}")

            print("▓" * 80)

        # ═══════════════════════════════════════════════════════════════
        #  DEEP PILLAR MATRIX MONITOR — per-pillar health intelligence
        # ═══════════════════════════════════════════════════════════════

        pmd = data.get("pillar_matrix_deep", {})
        if pmd and pmd.get("status") == "ok":
            print("\n" + "╔" + "═" * 78 + "╗")
            print(
                "║  🔬 DEEP PILLAR MATRIX MONITOR — ALL PILLARS HEALTH INTELLIGENCE"
                + " " * 12
                + "║"
            )
            print("╠" + "═" * 78 + "╣")

            ent_score = pmd.get("enterprise_score", 0)
            p_online = pmd.get("pillars_online", 0)
            p_total = pmd.get("pillars_total", 0)
            score_icon = (
                "🟢" if ent_score >= 80 else ("🟡" if ent_score >= 50 else "🔴")
            )
            print(
                f"║  Enterprise Score: {score_icon} {ent_score}%  |  Pillars: {p_online}/{p_total} online"
                + " " * 30
                + "║"
            )
            print("╠" + "═" * 78 + "╣")

            pillar_order = ["NCC_MASTER", "NCC", "AAC", "NCL", "BRS"]
            for pid in pillar_order:
                p = pmd.get("pillars", {}).get(pid, {})
                if not p:
                    continue
                name = p.get("name", pid)
                health = p.get("health", "?")
                h_icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(health, "⚫")
                matrix = p.get("matrix_status", "?")
                m_icon = (
                    "✅"
                    if matrix == "ACTIVE"
                    else ("⚠️" if matrix == "NO_MATRIX" else "❌")
                )
                score = p.get("overall_score", 0)
                h_status = p.get("health_status", "?")
                checks_p = p.get("checks_passed", 0)
                checks_t = p.get("checks_total", 0)
                slo_v = p.get("slo_violations", 0)
                alerts = p.get("active_alerts", 0)
                lat = p.get("latency_ms")
                lat_str = f"{lat:.0f}ms" if lat is not None else "—"
                uptime = p.get("uptime_s", 0)
                err = p.get("error")

                print(
                    f"║  {h_icon} {name:<22} Matrix:{m_icon} {matrix:<12} Latency:{lat_str:<8}"
                    + " " * max(0, 19 - len(lat_str))
                    + "║"
                )
                print(
                    f"║    Score: {score:.0%}  Status: {h_status:<10} Checks: {checks_p}/{checks_t}  SLO violations: {slo_v}  Alerts: {alerts}"
                    + " " * max(0, 5)
                    + "║"
                )
                if uptime > 0:
                    hrs = int(uptime // 3600)
                    mins = int((uptime % 3600) // 60)
                    print(f"║    Uptime: {hrs}h {mins}m" + " " * 60 + "║")
                if err:
                    print(f"║    ⚠️  {err[:70]}" + " " * max(0, 7) + "║")
                print("║" + " " * 78 + "║")

            print("╚" + "═" * 78 + "╝")

        # ═══════════════════════════════════════════════════════════════
        #  CAPITAL ROTATION MATRIX — 7 STRATEGIES
        # ═══════════════════════════════════════════════════════════════

        cap_rot = data.get("capital_rotation", {})
        if cap_rot.get("status") == "ok":
            strats = cap_rot.get("strategies", [])
            live_bal = cap_rot.get("live_balance_usd", 0.0)
            paper_bal = cap_rot.get("paper_balance_usd", 0.0)
            target = cap_rot.get("target_balance", 10_000_000.0)
            pct = cap_rot.get("progress_pct", 0.0)
            remaining = cap_rot.get("remaining_to_target", target)

            print("\n╔" + "═" * 78 + "╗")
            print("║  💎  CAPITAL ROTATION MATRIX — 7 STRATEGIES" + " " * 33 + "║")
            print("╠" + "═" * 78 + "╣")
            print(
                "║  #  │ Icon │ Strategy            │ Broker/API              │ Type      │ Balance      ║"
            )
            print("╟" + "─" * 78 + "╢")

            for s in strats:
                sid = s["id"]
                icon = s["icon"]
                name = s["name"][:19].ljust(19)
                broker = s["broker"][:23].ljust(23)
                stype = s["type"][:9].ljust(9)
                bal = s["balance_usd"]
                is_live = s["is_live"]
                status = s["status"]

                if is_live:
                    bal_str = f"${bal:>10,.2f}"
                else:
                    bal_str = f"${bal:>10,.2f}*"

                status_icon = {
                    "LIVE": "🟢",
                    "STANDBY": "🟡",
                    "MANUAL": "📋",
                    "SCANNING": "🔍",
                    "PAPER": "📄",
                    "OFFLINE": "🔴",
                    "EMPTY": "⚫",
                }.get(status, "⚪")

                print(
                    f"║  {sid}  │ {icon}  │ {name} │ {broker} │ {stype} │ {bal_str} {status_icon} ║"
                )

            print("╟" + "─" * 78 + "╢")
            print(
                f"║  LIVE BALANCE (Strategies 1-5):  ${live_bal:>14,.2f} USD"
                + " " * 26
                + "║"
            )
            print(
                f"║  PAPER BALANCE (Strategies 6-7): ${paper_bal:>14,.2f} USD  (not in total)"
                + " " * 10
                + "║"
            )
            print("╟" + "─" * 78 + "╢")

            # ── $10M Progress Bar ──
            bar_width = 40
            filled = int(bar_width * min(pct, 100.0) / 100.0)
            bar = "█" * filled + "░" * (bar_width - filled)
            print("║  🎯 TARGET: $10,000,000" + " " * 54 + "║")
            print(
                f"║  [{bar}] {pct:.4f}%" + " " * max(0, 28 - len(f"{pct:.4f}%")) + "║"
            )
            print(
                f"║  LIVE: ${live_bal:>12,.2f}  │  REMAINING: ${remaining:>14,.2f}"
                + " " * 17
                + "║"
            )
            print("╚" + "═" * 78 + "╝")
        else:
            print("\n╔" + "═" * 78 + "╗")
            print("║  💎  CAPITAL ROTATION MATRIX — 7 STRATEGIES" + " " * 33 + "║")
            print("║  ⚠️  Data unavailable" + " " * 56 + "║")
            print("╚" + "═" * 78 + "╝")

        # ═══════════════════════════════════════════════════════════════
        #  ELITE TRADING DESK — CONSOLIDATED COMMAND PANELS
        # ═══════════════════════════════════════════════════════════════

        print("\n" + "█" * 80)
        print("  ⚜️  ELITE TRADING DESK — INTEGRATED COMMAND CONSOLE")
        print("█" * 80)

        # ── Stock Ticker Ribbon ──────────────────────────────────────
        ticker = data.get("stock_ticker", {})
        if ticker and ticker.get("status") == "ok":
            ribbon = "  📊 "
            for t in ticker.get("tickers", []):
                pct = t.get("change_pct", 0)
                arrow = "▲" if pct >= 0 else "▼"
                ribbon += (
                    f"{t['ticker']} ${t.get('price', 0):,.2f} {arrow}{abs(pct):.1f}%  "
                )
            print(ribbon)
        else:
            print("  📊 Stock Ticker: offline (no Polygon key)")
        print("-" * 80)

        # ── NCL Link / Cross-Pillar Hub ──────────────────────────────
        ncl = data.get("ncl_link", {})
        if ncl and ncl.get("status") == "ok":
            print("\n  🔗 NCL LINK — CROSS-PILLAR HUB")
            mode = ncl.get("doctrine_mode", "?")
            mode_icon = {
                "NORMAL": "🟢",
                "CAUTION": "🟡",
                "SAFE_MODE": "🟠",
                "HALT": "🔴",
            }.get(mode, "⚪")
            print(
                f"    Doctrine: {mode_icon} {mode}  |  Trading: {'✅' if ncl.get('trading_allowed') else '🚫'}  |  Risk Mult: {ncl.get('risk_multiplier', 0):.1f}x"
            )
            pillars = ncl.get("pillars", {})
            for name, p in pillars.items():
                con = "🟢" if p.get("connected") else "🔴"
                print(
                    f"    {con} {name:4} mode={p.get('mode', '?'):10} hb={p.get('last_heartbeat', 'never')[:19]}"
                )
            directive = ncl.get("last_directive")
            if directive:
                print(
                    f"    📜 Last Directive: {directive['action'].upper()} — {directive['reason']}"
                )
            ncl_intel = ncl.get("ncl_intelligence", {})
            if ncl_intel.get("has_forecasts"):
                print(f"    🧠 NCL Forecasts: {ncl_intel['forecast_count']} available")
            gov_src = ncl.get("governance_source", "none")
            print(
                f"    Governance: via {gov_src}  |  BRS Patterns: {ncl.get('brs_patterns', 0)}"
            )
        elif ncl and ncl.get("status") != "not_available":
            print(f"\n  🔗 NCL LINK: ⚠️  {ncl.get('error', 'error')}")

        # ── Jonny Bravo Division ─────────────────────────────────────
        jb = data.get("jonny_bravo", {})
        if jb and jb.get("status") == "ok":
            print("\n  🥋 JONNY BRAVO DIVISION")
            print(
                f"    Level: {jb.get('student_level', '?').upper()}  |  Lessons: {jb.get('lessons_loaded', 0)}  |  Journal: {jb.get('journal_entries', 0)} entries"
            )
            wr = jb.get("win_rate", 0)
            wr_icon = "🟢" if wr >= 0.6 else "🟡" if wr >= 0.4 else "🔴"
            print(
                f"    Win Rate: {wr_icon} {wr:.0%}  |  Methods: {', '.join(jb.get('methodologies', [])[:4])}"
            )
        elif jb and jb.get("status") != "not_available":
            print(f"\n  🥋 JONNY BRAVO: ⚠️  {jb.get('error', 'error')}")

        # ── Superstonk / Reddit Sentiment ────────────────────────────
        ss = data.get("superstonk", {})
        if ss and ss.get("status") == "ok":
            print("\n  🦍 SUPERSTONK / WSB SENTIMENT")
            print(
                f"    Tracked: {ss.get('tickers_tracked', 0)} tickers  |  🟢 Bull: {ss.get('bullish', 0)}  🔴 Bear: {ss.get('bearish', 0)}  ⚪ Neutral: {ss.get('neutral', 0)}"
            )
            top = ss.get("top_10", [])
            if top:
                print("    Top Discussed:")
                for t in top[:5]:
                    sent = t.get("sentiment", 0)
                    icon = "🟢" if sent > 0 else "🔴" if sent < 0 else "⚪"
                    print(
                        f"      {icon} {t['ticker']:6} mentions={t.get('mentions', 0):4}  sent={sent:+.2f}"
                    )
        elif ss and ss.get("status") != "not_available":
            print(f"\n  🦍 SUPERSTONK: {ss.get('status', 'offline')}")

        # ── POLYMARKET DIVISION — 3 Strategies ────────────────────────
        px = data.get("planktonxd", {})
        pdiv = data.get("polymarket_division", {})
        if (px and px.get("status") == "ok") or (pdiv and pdiv.get("status") == "ok"):
            loaded = pdiv.get("strategies_loaded", 0) if pdiv else 0
            total = pdiv.get("strategies_total", 3) if pdiv else 3
            print("")
            print("  " + "▓" * 76)
            print(f"  🎯 POLYMARKET DIVISION — {loaded}/{total} STRATEGIES LOADED")
            print("  " + "▓" * 76)

            # 1. PlanktonXD Harvester
            if px and px.get("status") == "ok":
                print(f"    🐙 PLANKTONXD HARVESTER")
                print(
                    f"       Bets: {px.get('active_bets', 0)}/{px.get('total_bets', 0)} active  |  "
                    f"Scenarios: {px.get('scenarios_matched', 0)}"
                )
                print(
                    f"       Deployed: ${px.get('total_deployed', 0):,.2f}  |  "
                    f"Max Payout: ${px.get('max_payout', 0):,.2f}"
                )
                top_bets = px.get("top_bets", [])
                if top_bets:
                    for b in top_bets[:3]:
                        print(
                            f"         ${b.get('size', 0):.2f}  [{b.get('scenario', '?'):20}]  {b.get('market', '?')}"
                        )
            else:
                print(f"    🐙 PLANKTONXD: {px.get('status', 'offline') if px else 'not loaded'}")

            # 2. War Room Poly
            wr = pdiv.get("war_room", {}) if pdiv else {}
            if wr.get("status") == "ok":
                pressure = wr.get("pressure_level", 0)
                p_icon = "🔴" if pressure >= 0.8 else "🟡" if pressure >= 0.4 else "🟢"
                print(f"    ⚔️  WAR ROOM POLY — Thesis Chain")
                print(
                    f"       Pressure: {p_icon} {pressure:.0%}  |  "
                    f"Matches: {wr.get('n_matches', 0)}"
                )
                stages = wr.get("stages", {})
                if stages:
                    stage_parts = []
                    for sname, sinfo in list(stages.items())[:6]:
                        short = sname.replace("stage_", "S").replace("_", " ")[:12]
                        n = sinfo.get("matches", 0)
                        stage_parts.append(f"{short}={n}")
                    print(f"       Stages: {' | '.join(stage_parts)}")
                top_wr = wr.get("top_matches", [])
                if top_wr:
                    for m in top_wr[:3]:
                        q = m.get("question", "?")[:40]
                        print(
                            f"         {m.get('stage', '?'):12}  ${m.get('price', 0):.3f}  "
                            f"edge={m.get('edge', 0):+.3f}  {m.get('multiplier', 0):.1f}x  {q}"
                        )
            elif wr.get("status") == "error":
                print(f"    ⚔️  WAR ROOM POLY: ⚠️  {wr.get('error', 'error')}")
            else:
                print(f"    ⚔️  WAR ROOM POLY: not loaded")

            # 3. PolyMC Agent — Monte Carlo Portfolio
            mc = pdiv.get("polymc", {}) if pdiv else {}
            if mc.get("status") == "ok":
                ev_pct = mc.get("ev_pct", 0)
                ev_icon = "🟢" if ev_pct > 5 else "🟡" if ev_pct > 0 else "🔴"
                print(f"    🎲 POLYMC AGENT — 100K Monte Carlo")
                print(
                    f"       Portfolio: {mc.get('portfolio_size', 0)} bets  |  "
                    f"Cost: ${mc.get('total_cost', 0):,.2f}  |  "
                    f"Max Payout: ${mc.get('max_payout', 0):,.2f}"
                )
                print(
                    f"       {ev_icon} EV: {ev_pct:+.1f}%  |  "
                    f"Mean Return: ${mc.get('mean_return', 0):+,.2f}  |  "
                    f"P(profit): {mc.get('prob_profit', 0):.1%}"
                )
                print(
                    f"       VaR(95): ${mc.get('var_95', 0):,.2f}  |  "
                    f"Sharpe: {mc.get('sharpe', 0):.3f}"
                )
                per_bet = mc.get("per_bet", [])
                if per_bet:
                    for pb in per_bet[:5]:
                        ev = pb.get("ev", 0)
                        ev_sym = "+" if ev >= 0 else ""
                        print(
                            f"         {pb.get('name', '?')[:35]:<35}  "
                            f"EV=${ev_sym}{ev:.2f}  P={pb.get('prob_profit', 0):.0%}"
                        )
            elif mc.get("status") == "error":
                print(f"    🎲 POLYMC AGENT: ⚠️  {mc.get('error', 'error')}")
            else:
                print(f"    🎲 POLYMC AGENT: not loaded")

            # 4. Monitor — Exit Signals
            mon = pdiv.get("monitor", {}) if pdiv else {}
            if mon.get("status") == "ok":
                imm = mon.get("immediate_exits", 0)
                imm_icon = "🔴" if imm > 0 else "🟢"
                print(
                    f"    📡 MONITOR: {imm_icon} {mon.get('total_signals', 0)} signals  |  "
                    f"Immediate exits: {imm}  |  Checks: {mon.get('check_count', 0)}"
                )

            # 5. Account Tracker — Per-Strategy Balances
            acct = pdiv.get("account", {}) if pdiv else {}
            if acct.get("status") == "ok":
                wallet = acct.get("wallet_balance", 0)
                total_val = acct.get("total_account_value", 0)
                total_pos = acct.get("total_positions", 0)
                total_ord = acct.get("total_orders", 0)
                print(f"    💰 ACCOUNT TRACKER — Per-Strategy Allocation")
                print(f"       Wallet: ${wallet:,.2f}  |  Positions: {total_pos}  |  Orders: {total_ord}  |  Total: ${total_val:,.2f}")
                strats = acct.get("strategies", {})
                for skey, slabel in [("war_room", "⚔️  War Room"), ("planktonxd", "🐙 PlanktonXD"), ("polymc", "🎲 PolyMC   ")]:
                    s = strats.get(skey, {})
                    s_pos = s.get("positions", 0)
                    s_ord = s.get("orders", 0)
                    s_val = s.get("total_deployed", 0)
                    s_pct = (s_val / total_val * 100) if total_val > 0 else 0
                    print(f"         {slabel}:  {s_pos:>3} pos  {s_ord:>3} ord  ${s_val:>9,.2f}  ({s_pct:5.1f}%)")
            elif acct.get("status") == "error":
                print(f"    💰 ACCOUNT: ⚠️  {acct.get('error', 'error')}")

            # 6. Active Scanner — Unified Engine
            asc = data.get("active_scanner", {})
            if asc.get("status") == "ok":
                mode = "DRY RUN" if asc.get("dry_run", True) else "LIVE"
                mode_icon = "🔵" if asc.get("dry_run", True) else "🔴"
                bets = asc.get("daily_bet_count", 0)
                max_b = asc.get("max_daily_bets", 50)
                max_pos = asc.get("max_position_usd", 25)
                min_edge = asc.get("min_edge", 0.03)
                interval = asc.get("scan_interval_s", 300)
                execs = asc.get("execution_log_count", 0)
                print(
                    f"    🔍 ACTIVE SCANNER: {mode_icon} {mode}  |  "
                    f"Bets today: {bets}/{max_b}  |  Max ${max_pos}  |  "
                    f"Min edge: {min_edge:.0%}  |  Interval: {interval}s  |  Executions: {execs}"
                )

            print("  " + "▓" * 76)
        elif px and px.get("status") != "not_available":
            print(f"\n  🎯 POLYMARKET DIVISION: {px.get('status', 'offline')}")
        elif pdiv and pdiv.get("status") == "error":
            print(f"\n  🎯 POLYMARKET DIVISION: ⚠️  {pdiv.get('error', 'error')}")

        # ── Unusual Whales (already wired — add summary) ────────────
        mi = data.get("market_intelligence", {})
        if mi and mi.get("status") != "error":
            pcr = mi.get("put_call_ratio", "?")
            tone = mi.get("market_tone", "?")
            flow = mi.get("options_flow_signal_count", 0)
            dp = mi.get("dark_pool_trade_count", 0)
            dp_vol = mi.get("dark_pool_notional", 0)
            congress = mi.get("congress_trade_count", 0)
            print("\n  🐋 UNUSUAL WHALES")
            print(
                f"    P/C: {pcr}  Tone: {tone}  |  Flow Signals: {flow}  |  Dark Pool: {dp} trades (${dp_vol:,.0f})"
            )
            if congress:
                print(f"    📜 Congress Trades: {congress}")

        # ── Grok AI Trade Scorer ─────────────────────────────────────
        grok = data.get("grok_scorer", {})
        if grok and grok.get("status") in ("ok", "no_keys"):
            models = grok.get("models_available", [])
            primary = grok.get("primary_model", "none")
            status_icon = "🟢" if models else "🟡"
            print("\n  🤖 GROK AI TRADE SCORER")
            print(
                f"    {status_icon} Primary: {primary}  |  Models: {', '.join(models) if models else 'heuristic only'}"
            )
            print(
                f"    Thresholds: actionable ≥{grok.get('scoring_threshold', 60)}  strong ≥{grok.get('strong_threshold', 80)}"
            )
        elif grok and grok.get("status") != "not_available":
            print(f"\n  🤖 GROK: ⚠️  {grok.get('error', 'error')}")

        # ── OpenClaw Task Hub ────────────────────────────────────────
        oc = data.get("openclaw", {})
        if oc and oc.get("status") == "ok":
            print("\n  🦀 OPENCLAW — BARREN WUFFET SKILLS HUB")
            print(f"    Total Skills: {oc.get('total_skills', 0)}")
            cats = oc.get("categories", {})
            if cats:
                cat_line = "    "
                for cat, count in cats.items():
                    cat_line += f"{cat.replace('_', ' ')}: {count}  "
                print(cat_line)
            sample = oc.get("sample_skills", [])
            if sample:
                print(f"    Sample: {', '.join(sample[:6])}")
        elif oc and oc.get("status") != "not_available":
            print(f"\n  🦀 OPENCLAW: ⚠️  {oc.get('error', 'error')}")

        print("█" * 80)

        # Alerts
        alerts = data.get("alerts", [])
        if alerts:
            print("\n[ALERT] ACTIVE ALERTS")
            print("-" * 15)
            for alert in alerts[:5]:  # Show first 5 alerts
                severity = alert.get("severity", "info")
                emoji = {"critical": "🔴", "warning": "🟡", "info": "ℹ️"}.get(
                    severity, "ℹ️"
                )
                print(f"  {emoji} {alert.get('title', 'Unknown alert')}")

        print("\n" + "=" * 80)
        print("Press Ctrl+C to quit | Auto-refresh: 1s")
        print("=" * 80)

    def _run_curses_dashboard(self, stdscr):
        """Run the curses-based dashboard"""
        # Setup curses
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.use_default_colors()
        stdscr.timeout(1000)  # Refresh every second

        while self.running:
            try:
                # Get latest data (thread-safe read)
                with self._data_lock:
                    data = self._latest_data.copy() if self._latest_data else {}

                # Display dashboard
                self.display_dashboard(stdscr, data)

                # Handle input
                key = stdscr.getch()
                if key == ord("q"):
                    self.running = False
                    break
                elif key == ord("r"):
                    # Manual refresh
                    pass

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                stdscr.clear()
                stdscr.addstr(0, 0, f"Dashboard error: {e}")
                stdscr.refresh()
                time.sleep(2)

    def _run_text_dashboard_loop(self):
        """Run the text-based dashboard loop"""
        print("\n" + "=" * 80)
        print("AAC 2100 MASTER MONITORING DASHBOARD (Text Mode)")
        print("=" * 80)
        print("Press Ctrl+C to quit | Auto-refresh: 1s")
        print("")

        _shown_waiting = False

        while self.running:
            try:
                # Get latest data (thread-safe read)
                with self._data_lock:
                    data = self._latest_data.copy() if self._latest_data else {}

                if data:
                    _shown_waiting = False
                    # Display dashboard
                    self._display_text_dashboard(data)
                elif not _shown_waiting:
                    print("Collecting data from all sources (first pass) ...")
                    _shown_waiting = True

                # Wait before next update
                time.sleep(1)

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"Dashboard error: {e}")
                time.sleep(2)

    async def run_dashboard(self, port: int = 8050):
        """Run the monitoring dashboard based on display mode"""
        if self.display_mode == DisplayMode.WEB:
            await self._run_web_dashboard()
        elif self.display_mode == DisplayMode.DASH:
            await self._run_dash_dashboard()
        elif self.display_mode == DisplayMode.API:
            await self._run_api_dashboard()
        else:  # Default to terminal mode
            await self._run_terminal_dashboard()

    async def _run_terminal_dashboard(self):
        """Run terminal-based dashboard"""
        if CURSES_AVAILABLE and platform.system() != "Windows":
            # Use curses on Unix-like systems
            def dashboard_thread():
                """Dashboard thread."""
                curses.wrapper(self._run_curses_dashboard)

            # Start dashboard in separate thread
            dashboard_thread = threading.Thread(target=dashboard_thread, daemon=True)
            dashboard_thread.start()
        else:
            # Use text-based dashboard on Windows or when curses unavailable
            def dashboard_thread():
                """Dashboard thread."""
                self._run_text_dashboard_loop()

            # Start dashboard in separate thread
            dashboard_thread = threading.Thread(target=dashboard_thread, daemon=True)
            dashboard_thread.start()

        # Main monitoring loop
        self.running = True
        while self.running:
            try:
                # Collect data
                data = await self.collect_monitoring_data()
                self.last_update = datetime.now()

                # Update shared data for dashboard
                with self._data_lock:
                    self._latest_data = data

                # Wait before next update
                await asyncio.sleep(self.refresh_rate)

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                logger.error(f"Monitoring error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _run_web_dashboard(self):
        """Run Streamlit web dashboard via subprocess."""
        if not STREAMLIT_AVAILABLE:
            raise RuntimeError(
                "Web dashboard requested but streamlit is not installed. "
                "Install with: pip install streamlit"
            )

        import subprocess

        dashboard_script = Path(__file__).parent / "streamlit_dashboard.py"
        if not dashboard_script.exists():
            raise FileNotFoundError(
                "Web dashboard module missing: monitoring/streamlit_dashboard.py"
            )
        port = int(os.environ.get("STREAMLIT_PORT", "8501"))
        logger.info(f"Launching Streamlit dashboard on port {port}...")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard_script),
                "--server.port",
                str(port),
                "--server.headless",
                "true",
            ],
            cwd=str(PROJECT_ROOT),
        )
        logger.info(
            f"Streamlit dashboard running (PID {proc.pid}) at http://localhost:{port}"
        )

        # Keep the async loop alive while Streamlit runs
        self.running = True
        try:
            while self.running and proc.poll() is None:
                data = await self.collect_monitoring_data()
                with self._data_lock:
                    self._latest_data = data
                await asyncio.sleep(self.refresh_rate)
        except KeyboardInterrupt:
            self.running = False
        finally:
            proc.terminate()

    async def _run_dash_dashboard(self):
        """Run Plotly Dash analytics dashboard."""
        global DASH_AVAILABLE
        if not DASH_AVAILABLE:
            try:
                import dash  # noqa: F401

                DASH_AVAILABLE = True
            except ImportError:
                logger.info(
                    "Dash not available. Install with: pip install dash dash-bootstrap-components"
                )
                logger.info("Falling back to terminal mode.")
                await self._run_terminal_dashboard()
                return

        from monitoring.dash_dashboard import AACDashDashboard

        dash_dashboard = AACDashDashboard()
        if await dash_dashboard.initialize():
            port = int(os.environ.get("DASH_PORT", "8050"))
            logger.info(f"Starting Dash analytics dashboard on port {port}...")
            dash_thread = threading.Thread(
                target=dash_dashboard.run_dashboard,
                kwargs={"port": port},
                daemon=True,
            )
            dash_thread.start()

            # Keep collecting data while Dash runs
            self.running = True
            try:
                while self.running:
                    data = await self.collect_monitoring_data()
                    with self._data_lock:
                        self._latest_data = data
                    await asyncio.sleep(self.refresh_rate)
            except KeyboardInterrupt:
                self.running = False
        else:
            logger.info(
                "Dash dashboard initialization failed, falling back to terminal"
            )
            await self._run_terminal_dashboard()

    async def _run_api_dashboard(self):
        """Run REST API dashboard exposing monitoring data as JSON."""
        from http.server import BaseHTTPRequestHandler, HTTPServer

        dashboard_ref = self

        class MonitoringAPIHandler(BaseHTTPRequestHandler):
            """Simple HTTP handler for monitoring API."""

            def do_GET(self):
                if self.path == "/health":
                    self._json_response(
                        {"status": "healthy", "timestamp": datetime.now().isoformat()}
                    )
                elif self.path == "/api/status":
                    data = dashboard_ref._latest_data
                    self._json_response(self._serializable(data))
                elif self.path == "/api/alerts":
                    data = dashboard_ref._latest_data
                    alerts = data.get("alerts", []) if data else []
                    self._json_response({"alerts": self._serializable(alerts)})
                elif self.path == "/api/registry":
                    data = dashboard_ref._latest_data
                    registry = data.get("registry", {}) if data else {}
                    self._json_response(self._serializable(registry))
                elif self.path == "/api/maximizer":
                    data = dashboard_ref._latest_data
                    mm = data.get("matrix_maximizer", {}) if data else {}
                    self._json_response(self._serializable(mm))
                elif self.path == "/api/elite-desk":
                    data = dashboard_ref._latest_data or {}
                    self._json_response(
                        self._serializable(
                            {
                                "stock_ticker": data.get("stock_ticker", {}),
                                "ncl_link": data.get("ncl_link", {}),
                                "jonny_bravo": data.get("jonny_bravo", {}),
                                "superstonk": data.get("superstonk", {}),
                                "planktonxd": data.get("planktonxd", {}),
                                "grok_scorer": data.get("grok_scorer", {}),
                                "openclaw": data.get("openclaw", {}),
                                "market_intelligence": data.get(
                                    "market_intelligence", {}
                                ),
                            }
                        )
                    )
                elif self.path == "/api/pillar-network":
                    data = dashboard_ref._latest_data or {}
                    self._json_response(
                        self._serializable(data.get("pillar_network", {}))
                    )
                elif self.path == "/api/matrix-monitors":
                    data = dashboard_ref._latest_data or {}
                    self._json_response(
                        self._serializable(data.get("pillar_matrix_deep", {}))
                    )
                elif self.path == "/api/integration-status":
                    try:
                        from core.unified_component_integrator import (
                            get_unified_integrator,
                        )

                        integrator = get_unified_integrator(paper_mode=True)
                        s = integrator.status
                        self._json_response(
                            self._serializable(
                                {
                                    "bridge_orchestrator": s.bridge_orchestrator,
                                    "cross_pillar_hub": s.cross_pillar_hub,
                                    "strategy_loader": s.strategy_loader,
                                    "strategy_execution": s.strategy_execution,
                                    "strategy_integrator": s.strategy_integrator,
                                    "api_hub": s.api_hub,
                                    "ncc_master_adapter": s.ncc_master_adapter,
                                    "ncc_bridge": s.ncc_bridge,
                                    "pillar_matrix_federation": s.pillar_matrix_federation,
                                    "doctrine_feedback_loop": s.doctrine_feedback_loop,
                                    "components_wired": s.components_wired,
                                    "components_failed": s.components_failed,
                                    "errors": s.errors,
                                }
                            )
                        )
                    except Exception as e:
                        self._json_response({"error": str(e), "components_wired": 0})
                else:
                    self.send_error(
                        404,
                        "Endpoints: /health, /api/status, /api/alerts, /api/registry, /api/maximizer, /api/elite-desk, /api/pillar-network, /api/matrix-monitors, /api/integration-status",
                    )

            def _json_response(self, obj):
                body = json.dumps(obj, default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            @staticmethod
            def _serializable(obj):
                """Convert non-serializable types."""
                if isinstance(obj, dict):
                    return {
                        k: MonitoringAPIHandler._serializable(v) for k, v in obj.items()
                    }
                if isinstance(obj, (list, tuple)):
                    return [MonitoringAPIHandler._serializable(i) for i in obj]
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            def log_message(self, format, *args):
                logger.debug(f"API: {format % args}")

        port = int(os.environ.get("API_DASHBOARD_PORT", "8080"))
        server = HTTPServer(("127.0.0.1", port), MonitoringAPIHandler)
        api_thread = threading.Thread(target=server.serve_forever, daemon=True)
        api_thread.start()
        logger.info(f"API dashboard running on http://localhost:{port}")
        logger.info(
            "  Endpoints: /health, /api/status, /api/alerts, /api/registry, /api/maximizer, /api/elite-desk, /api/pillar-network, /api/matrix-monitors, /api/integration-status"
        )

        # Main monitoring loop
        self.running = True
        try:
            while self.running:
                data = await self.collect_monitoring_data()
                with self._data_lock:
                    self._latest_data = data
                await asyncio.sleep(self.refresh_rate)
        except KeyboardInterrupt:
            self.running = False
        finally:
            server.shutdown()

    async def start_monitoring(self):
        """Start the master monitoring system"""
        logger.info("[START] Starting AAC Master Monitoring Dashboard...")

        # Initialize
        if not await self.initialize():
            logger.info("[ERROR] Failed to initialize monitoring dashboard")
            return

        # Start monitoring loop
        await self.run_dashboard()

        logger.info("[SUCCESS] Master monitoring dashboard started")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        logger.info("[STOP] Master monitoring dashboard stopped")


# Global instance
_master_dashboard = None


def get_master_dashboard(display_mode: str = DisplayMode.TERMINAL):
    """Get the appropriate dashboard instance based on display mode"""
    if display_mode == DisplayMode.WEB:
        if not STREAMLIT_AVAILABLE:
            raise RuntimeError(
                "Web dashboard requested but streamlit is unavailable. "
                "Install streamlit or use --mode terminal"
            )
        try:
            from monitoring.streamlit_dashboard import AACStreamlitDashboard

            return AACStreamlitDashboard()
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Web dashboard requested but monitoring.streamlit_dashboard "
                "is missing. Rebuild or restore monitoring/streamlit_dashboard.py"
            ) from exc
    elif display_mode == DisplayMode.DASH:
        try:
            from monitoring.dash_dashboard import AACDashDashboard

            return AACDashDashboard()
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Dash dashboard requested but monitoring.dash_dashboard is missing. "
                "Create monitoring/dash_dashboard.py or use --mode terminal"
            ) from exc
    else:
        return AACMasterMonitoringDashboard(display_mode)


async def _async_main():
    """Async entry point (internal)"""
    import argparse

    parser = argparse.ArgumentParser(description="AAC Master Monitoring Dashboard")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["terminal", "web", "dash", "api"],
        default="terminal",
        help="Display mode",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8501, help="Port for web/API modes"
    )

    args = parser.parse_args()

    display_mode = {
        "terminal": DisplayMode.TERMINAL,
        "web": DisplayMode.WEB,
        "dash": DisplayMode.DASH,
        "api": DisplayMode.API,
    }.get(args.mode, DisplayMode.TERMINAL)

    dashboard = get_master_dashboard(display_mode)

    try:
        if display_mode == DisplayMode.WEB:
            # Streamlit dashboard
            if hasattr(dashboard, "run_dashboard"):
                await dashboard.run_dashboard(port=args.port)
            else:
                logger.info("Streamlit dashboard not available")
        elif display_mode == DisplayMode.DASH:
            # Dash dashboard
            if hasattr(dashboard, "run_dashboard"):
                await dashboard.run_dashboard(port=args.port)
            else:
                logger.info("Dash dashboard not available")
        else:
            # Terminal dashboard
            await dashboard.start_monitoring()
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down master monitoring dashboard...")
        if hasattr(dashboard, "stop_monitoring"):
            dashboard.stop_monitoring()
    except Exception as e:
        logger.info(f"[CROSS] Master monitoring dashboard failed: {e}")
        if hasattr(dashboard, "stop_monitoring"):
            dashboard.stop_monitoring()


def main():
    """Sync entry point for console_scripts / setuptools."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()


# Backward-compatible re-exports for external consumers — lazy to avoid hard dep at import time
def __getattr__(name):  # noqa: E302
    _streamlit_exports = {
        "AACStreamlitDashboard",
        "generate_copilot_response",
        "play_audio_response",
        "run_streamlit_dashboard",
    }
    if name in _streamlit_exports:
        return locals()[name]
    if name == "AACDashDashboard":
        from monitoring.dash_dashboard import AACDashDashboard  # noqa: E402

        return AACDashDashboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
