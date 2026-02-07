#!/usr/bin/env python3
"""
ACC Orchestrator
================
Central coordinator that manages all theaters, agents, and execution.
"""

import asyncio
import logging
import signal
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.data_sources import DataAggregator, MarketTick
from BigBrainIntelligence.agents import (
    get_all_agents, get_agents_by_theater, BaseResearchAgent, ResearchFinding
)
from TradingExecution.execution_engine import ExecutionEngine, OrderSide, Position
from CentralAccounting.database import AccountingDatabase

# Import health server (optional)
try:
    from shared.health_server import HealthServer
    HEALTH_SERVER_AVAILABLE = True
except ImportError:
    HEALTH_SERVER_AVAILABLE = False

# Import startup validator (optional)
try:
    from shared.startup_validator import validate_startup
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False

# Import CryptoIntelligence integration (optional)
try:
    from CryptoIntelligence.crypto_bigbrain_integration import CryptoBigBrainIntegration
    CRYPTO_INTEL_AVAILABLE = True
except ImportError:
    CRYPTO_INTEL_AVAILABLE = False

# Import cache manager (optional)
try:
    from shared.cache_manager import get_cache, CacheManager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# Import WebSocket feeds (optional)
try:
    from shared.websocket_feeds import PriceFeedManager, BinanceWebSocketFeed, CoinbaseWebSocketFeed
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


class OrchestratorState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class TheaterStatus:
    """Status of a theater"""
    name: str
    is_active: bool = True
    last_scan: Optional[datetime] = None
    findings_count: int = 0
    actions_count: int = 0
    errors_count: int = 0


@dataclass
class Signal:
    """Trading signal from agents"""
    signal_id: str
    source_agent: str
    theater: str
    signal_type: str
    symbol: str
    direction: str  # 'long', 'short', 'neutral'
    strength: float  # 0-1
    confidence: float  # 0-1
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def score(self) -> float:
        """Combined signal score"""
        return self.strength * self.confidence

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class SignalAggregator:
    """Aggregates and scores signals from multiple agents"""

    def __init__(self):
        self.signals: List[Signal] = []
        self.signal_weights = {
            "theater_b": 0.3,  # Attention/narrative signals
            "theater_c": 0.4,  # Infrastructure signals (higher weight - more actionable)
            "theater_d": 0.3,  # Information asymmetry signals
        }

    def add_signal(self, signal: Signal):
        """Add a new signal"""
        self.signals.append(signal)
        # Keep only recent signals
        self._cleanup_expired()

    def _cleanup_expired(self):
        """Remove expired signals"""
        self.signals = [s for s in self.signals if not s.is_expired()]

    def get_consensus(self, symbol: str) -> Dict[str, Any]:
        """Get consensus signal for a symbol"""
        symbol_signals = [s for s in self.signals if s.symbol == symbol and not s.is_expired()]
        
        if not symbol_signals:
            return {"direction": "neutral", "confidence": 0, "signal_count": 0}

        # Weight signals by theater
        weighted_direction = 0
        total_weight = 0
        
        for signal in symbol_signals:
            weight = self.signal_weights.get(signal.theater, 0.33)
            direction_value = 1 if signal.direction == "long" else (-1 if signal.direction == "short" else 0)
            weighted_direction += direction_value * signal.score * weight
            total_weight += weight

        if total_weight == 0:
            return {"direction": "neutral", "confidence": 0, "signal_count": len(symbol_signals)}

        avg_direction = weighted_direction / total_weight
        avg_confidence = sum(s.confidence for s in symbol_signals) / len(symbol_signals)

        return {
            "direction": "long" if avg_direction > 0.3 else ("short" if avg_direction < -0.3 else "neutral"),
            "strength": abs(avg_direction),
            "confidence": avg_confidence,
            "signal_count": len(symbol_signals),
            "signals": symbol_signals,
        }

    def get_top_opportunities(self, min_score: float = 0.5, limit: int = 10) -> List[Dict]:
        """Get top trading opportunities"""
        self._cleanup_expired()
        
        # Group by symbol
        symbols = set(s.symbol for s in self.signals)
        opportunities = []
        
        for symbol in symbols:
            consensus = self.get_consensus(symbol)
            if consensus["confidence"] >= min_score and consensus["direction"] != "neutral":
                opportunities.append({
                    "symbol": symbol,
                    **consensus,
                })
        
        # Sort by combined score
        opportunities.sort(key=lambda x: x["strength"] * x["confidence"], reverse=True)
        
        return opportunities[:limit]


class Orchestrator:
    """Main orchestrator for ACC system"""

    def __init__(
        self,
        enable_health_server: bool = True,
        health_port: int = 8080,
        enable_crypto_intel: bool = True,
        enable_websocket_feeds: bool = True,
        enable_cache: bool = True,
        validate_on_startup: bool = True,
    ):
        self.config = get_config()
        self.logger = self._setup_logging()
        
        # State
        self.state = OrchestratorState.STOPPED
        self._shutdown_event = asyncio.Event()
        
        # Feature flags
        self._enable_health_server = enable_health_server and HEALTH_SERVER_AVAILABLE
        self._health_port = health_port
        self._enable_crypto_intel = enable_crypto_intel and CRYPTO_INTEL_AVAILABLE
        self._enable_websocket = enable_websocket_feeds and WEBSOCKET_AVAILABLE
        self._enable_cache = enable_cache and CACHE_AVAILABLE
        self._validate_startup = validate_on_startup and VALIDATOR_AVAILABLE
        
        # Components
        self.data_aggregator = DataAggregator()
        self.execution_engine = ExecutionEngine()
        self.signal_aggregator = SignalAggregator()
        self.db = AccountingDatabase()
        
        # Optional components
        self.health_server: Optional[Any] = None
        self.crypto_intel: Optional[Any] = None
        self.price_feeds: Optional[Any] = None
        self.cache: Optional[Any] = None
        
        # Agents
        self.agents: List[BaseResearchAgent] = []
        
        # Theater status
        self.theaters: Dict[str, TheaterStatus] = {
            "theater_b": TheaterStatus("Theater B - Attention"),
            "theater_c": TheaterStatus("Theater C - Infrastructure"),
            "theater_d": TheaterStatus("Theater D - Information"),
        }
        
        # Configuration
        self.scan_interval = 60  # seconds
        self.execution_interval = 30  # seconds
        self.min_signal_score = 0.6
        self.max_concurrent_trades = 3
        
        # Auto-execution configuration
        self.auto_execute_signals = False  # Disabled by default for safety
        self.auto_execute_min_confidence = 0.75  # Higher threshold for auto-execution
        self.auto_execute_confirmation_signals = 2  # Require multiple signals
        
        # Checkpoint path for graceful shutdown
        self._checkpoint_path = get_project_path('data', 'orchestrator_checkpoint.json')
        
        # Background task references
        self._auto_execute_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = {
            "scans_completed": 0,
            "signals_generated": 0,
            "trades_executed": 0,
            "auto_executed": 0,
            "total_pnl": 0.0,
            "start_time": None,
        }

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("orchestrator")
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(ch)
        
        # File handler
        log_dir = get_project_path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        fh = logging.FileHandler(log_dir / f"orchestrator_{datetime.now().strftime('%Y%m%d')}.log")
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(fh)
        
        return logger

    async def initialize(self):
        """Initialize all components"""
        self.state = OrchestratorState.STARTING
        self.logger.info("Initializing ACC Orchestrator...")
        
        try:
            # Run startup validation if enabled
            if self._validate_startup:
                self.logger.info("Running startup validation...")
                await validate_startup(fail_on_critical=True)
            
            # Initialize cache if enabled
            if self._enable_cache:
                self.cache = await get_cache()
                self.logger.info("Cache manager initialized")
            
            # Load agents
            self.agents = get_all_agents()
            self.logger.info(f"Loaded {len(self.agents)} research agents")
            
            # Connect data sources
            await self.data_aggregator.connect_all()
            self.logger.info("Data sources connected")
            
            # Initialize database
            self.db.initialize()
            self.logger.info("Database initialized")
            
            # Initialize CryptoIntelligence if enabled
            if self._enable_crypto_intel:
                self.crypto_intel = CryptoBigBrainIntegration()
                self.logger.info("CryptoIntelligence integration initialized")
            
            # Initialize WebSocket price feeds if enabled
            if self._enable_websocket:
                self.price_feeds = PriceFeedManager()
                self.price_feeds.add_feed(BinanceWebSocketFeed(
                    testnet=getattr(self.config.binance, 'testnet', True)
                ))
                self.price_feeds.add_feed(CoinbaseWebSocketFeed())
                self.logger.info("WebSocket price feeds initialized")
            
            # Start health server if enabled
            if self._enable_health_server:
                self.health_server = HealthServer(port=self._health_port)
                await self.health_server.start()
                self.logger.info(f"Health server started on port {self._health_port}")
            
            # Load checkpoint if exists
            self._load_checkpoint()
            
            self.state = OrchestratorState.RUNNING
            self.metrics["start_time"] = datetime.now()
            self.logger.info("Orchestrator initialized successfully")
            
        except Exception as e:
            self.state = OrchestratorState.ERROR
            self.logger.error(f"Initialization failed: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown with state persistence"""
        self.state = OrchestratorState.STOPPING
        self.logger.info("Shutting down orchestrator...")
        
        self._shutdown_event.set()
        
        # Save checkpoint before closing positions
        self._save_checkpoint()
        self.logger.info("Checkpoint saved")
        
        # Export final metrics
        self._export_final_metrics()
        
        # Close all positions if configured
        if getattr(self.config.risk, "close_positions_on_shutdown", True):
            for position in self.execution_engine.get_open_positions():
                self.logger.info(f"Closing position: {position.position_id}")
                await self.execution_engine.close_position(position.position_id)
        
        # Save state
        self.execution_engine.save_state()
        
        # Stop WebSocket feeds if running
        if self.price_feeds:
            await self.price_feeds.stop()
            self.logger.info("WebSocket feeds stopped")
        
        # Stop health server
        if self.health_server:
            await self.health_server.stop()
        
        # Disconnect data sources
        await self.data_aggregator.disconnect_all()
        
        self.state = OrchestratorState.STOPPED
        self.logger.info("Orchestrator shutdown complete")
    
    def _export_final_metrics(self):
        """Export metrics to file before shutdown"""
        try:
            metrics_path = get_project_path('data', 'final_metrics.json')
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            
            final_metrics = {
                "timestamp": datetime.now().isoformat(),
                "session_duration_seconds": (
                    (datetime.now() - self.metrics["start_time"]).total_seconds()
                    if self.metrics["start_time"] else 0
                ),
                "scans_completed": self.metrics["scans_completed"],
                "signals_generated": self.metrics["signals_generated"],
                "trades_executed": self.metrics["trades_executed"],
                "total_pnl": self.metrics["total_pnl"],
                "final_positions": len(self.execution_engine.get_open_positions()),
                "final_exposure": self.execution_engine.get_total_exposure(),
                "unrealized_pnl": self.execution_engine.get_total_unrealized_pnl(),
                "theaters": {
                    k: {
                        "findings": v.findings_count,
                        "actions": v.actions_count,
                        "errors": v.errors_count,
                    }
                    for k, v in self.theaters.items()
                },
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(final_metrics, f, indent=2, default=str)
            
            self.logger.info(f"Final metrics exported to {metrics_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export final metrics: {e}")

    def _save_checkpoint(self):
        """Save current state to checkpoint file"""
        try:
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics,
                "theaters": {
                    k: {
                        "findings_count": v.findings_count,
                        "actions_count": v.actions_count,
                        "errors_count": v.errors_count,
                    }
                    for k, v in self.theaters.items()
                },
                "pending_signals": [
                    {
                        "signal_id": s.signal_id,
                        "symbol": s.symbol,
                        "direction": s.direction,
                        "strength": s.strength,
                        "confidence": s.confidence,
                        "theater": s.theater,
                    }
                    for s in self.signal_aggregator.signals if not s.is_expired()
                ],
            }
            with open(self._checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
            self.logger.info(f"Saved checkpoint with {len(checkpoint['pending_signals'])} pending signals")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self):
        """Load state from checkpoint file"""
        if not self._checkpoint_path.exists():
            return
        
        try:
            with open(self._checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            # Restore metrics counters
            for key in ['scans_completed', 'signals_generated', 'trades_executed', 'total_pnl']:
                if key in checkpoint.get('metrics', {}):
                    self.metrics[key] = checkpoint['metrics'][key]
            
            # Restore theater counters
            for theater_id, data in checkpoint.get('theaters', {}).items():
                if theater_id in self.theaters:
                    self.theaters[theater_id].findings_count = data.get('findings_count', 0)
                    self.theaters[theater_id].actions_count = data.get('actions_count', 0)
                    self.theaters[theater_id].errors_count = data.get('errors_count', 0)
            
            self.logger.info(f"Loaded checkpoint from {checkpoint.get('timestamp', 'unknown')}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")

    async def run_agent_scan(self, theater: str) -> List[ResearchFinding]:
        """Run scans for all agents in a theater"""
        theater_status = self.theaters.get(theater)
        if not theater_status or not theater_status.is_active:
            return []

        findings = []
        agents = get_agents_by_theater(theater)
        
        for agent in agents:
            try:
                agent_findings = await agent.scan()
                findings.extend(agent_findings)
                
                # Convert findings to signals
                for finding in agent_findings:
                    signal = self._finding_to_signal(finding)
                    if signal:
                        self.signal_aggregator.add_signal(signal)
                        self.metrics["signals_generated"] += 1
                
            except Exception as e:
                self.logger.error(f"Agent {agent.agent_id} scan failed: {e}")
                theater_status.errors_count += 1

        theater_status.last_scan = datetime.now()
        theater_status.findings_count += len(findings)
        self.metrics["scans_completed"] += 1
        
        return findings

    def _finding_to_signal(self, finding: ResearchFinding) -> Optional[Signal]:
        """Convert a research finding to a trading signal"""
        # Map finding types to signal directions
        direction_map = {
            "narrative_shift": "long",
            "engagement_spike": "long",
            "bridge_arbitrage": "long",
            "liquidity_imbalance": "long",
            "api_signal": "long",
            "access_opportunity": "long",
        }
        
        direction = direction_map.get(finding.finding_type, "neutral")
        
        # Extract symbol from finding data
        symbol = finding.data.get("asset") or finding.data.get("symbol") or finding.data.get("narrative")
        if not symbol:
            return None

        # Normalize symbol format
        if "/" not in symbol:
            symbol = f"{symbol}/USDT"

        return Signal(
            signal_id=finding.finding_id,
            source_agent=finding.agent_id,
            theater=finding.theater,
            signal_type=finding.finding_type,
            symbol=symbol,
            direction=direction,
            strength=finding.confidence,
            confidence=finding.confidence,
            expires_at=finding.expires_at,
            metadata=finding.data,
        )

    async def execute_signals(self):
        """Execute trading signals"""
        if self.state != OrchestratorState.RUNNING:
            return

        # Get top opportunities
        opportunities = self.signal_aggregator.get_top_opportunities(
            min_score=self.min_signal_score,
            limit=self.max_concurrent_trades,
        )
        
        current_positions = len(self.execution_engine.get_open_positions())
        available_slots = self.max_concurrent_trades - current_positions
        
        for opp in opportunities[:available_slots]:
            if opp["direction"] == "neutral":
                continue

            symbol = opp["symbol"]
            
            # Check if we already have a position
            existing = [p for p in self.execution_engine.get_open_positions() if p.symbol == symbol]
            if existing:
                continue

            # Get current price
            snapshot = await self.data_aggregator.get_market_snapshot([symbol.split("/")[0]])
            if not snapshot:
                continue
            
            tick = list(snapshot.values())[0] if snapshot else None
            if not tick:
                continue

            # Open position
            side = OrderSide.BUY if opp["direction"] == "long" else OrderSide.SELL
            
            # Get actual account balance
            account_balance = await self.get_account_balance()
            
            # Calculate position size
            position_size = self.execution_engine.risk_manager.calculate_position_size(
                account_balance=account_balance,
                risk_per_trade_pct=2.0,
            )
            quantity = position_size / tick.price

            self.logger.info(
                f"Executing signal: {side.value} {quantity:.6f} {symbol} @ ${tick.price:,.2f} "
                f"(score: {opp['strength'] * opp['confidence']:.2f})"
            )
            
            position = await self.execution_engine.open_position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=tick.price,
            )
            
            if position:
                self.metrics["trades_executed"] += 1
                self.theaters[opp["signals"][0].theater].actions_count += 1
                
                # Log to database
                self.db.record_transaction(
                    account_id=1,  # Default account
                    transaction_type="trade",
                    asset=symbol.split("/")[0],
                    quantity=quantity,
                    price=tick.price,
                    side=side.value,
                    symbol=symbol,
                )

    async def update_positions(self):
        """Update position prices and check for exits"""
        positions = self.execution_engine.get_open_positions()
        if not positions:
            return

        # Get symbols we need prices for
        symbols = list(set(p.symbol.split("/")[0] for p in positions))
        
        # Fetch prices
        snapshot = await self.data_aggregator.get_market_snapshot(symbols)
        
        # Build price map
        prices = {}
        for tick in snapshot.values():
            # Extract base symbol
            base = tick.symbol.split("/")[0]
            prices[f"{base}/USDT"] = tick.price
        
        # Update positions
        await self.execution_engine.update_positions(prices)

    async def scan_loop(self):
        """Main scanning loop"""
        while not self._shutdown_event.is_set():
            if self.state == OrchestratorState.RUNNING:
                try:
                    # Scan all theaters
                    for theater in ["theater_b", "theater_c", "theater_d"]:
                        await self.run_agent_scan(theater)
                    
                    self.logger.info(
                        f"Scan complete: {self.metrics['signals_generated']} total signals"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Scan loop error: {e}")
            
            await asyncio.sleep(self.scan_interval)

    async def execution_loop(self):
        """Main execution loop"""
        while not self._shutdown_event.is_set():
            if self.state == OrchestratorState.RUNNING:
                try:
                    # Update positions
                    await self.update_positions()
                    
                    # Execute signals
                    await self.execute_signals()
                    
                except Exception as e:
                    self.logger.error(f"Execution loop error: {e}")
            
            await asyncio.sleep(self.execution_interval)

    async def metrics_loop(self):
        """Periodic metrics reporting"""
        while not self._shutdown_event.is_set():
            if self.state == OrchestratorState.RUNNING:
                self._log_metrics()
            
            await asyncio.sleep(300)  # Every 5 minutes

    def _log_metrics(self):
        """Log current metrics"""
        open_positions = self.execution_engine.get_open_positions()
        unrealized_pnl = self.execution_engine.get_total_unrealized_pnl()
        
        self.logger.info(
            f"METRICS | Scans: {self.metrics['scans_completed']} | "
            f"Signals: {self.metrics['signals_generated']} | "
            f"Trades: {self.metrics['trades_executed']} | "
            f"Open Positions: {len(open_positions)} | "
            f"Unrealized P&L: ${unrealized_pnl:.2f}"
        )

    async def get_account_balance(self, currency: str = 'USD') -> float:
        """
        Get actual account balance from database/exchange.
        Falls back to config default if unavailable.
        """
        try:
            # Try to get balance from database first
            balances = self.db.get_account_balances(account_id=1)
            if balances:
                total = sum(
                    b.get('free_balance', 0) + b.get('locked_balance', 0)
                    for b in balances
                    if b.get('asset', '').upper() in (currency.upper(), 'USDT', 'USDC')
                )
                if total > 0:
                    return total
            
            # Fall back to config default
            default_balance = getattr(self.config.risk, 'default_account_balance', 10000)
            self.logger.debug(f"Using default account balance: ${default_balance}")
            return default_balance
            
        except Exception as e:
            self.logger.warning(f"Failed to get account balance: {e}")
            return getattr(self.config.risk, 'default_account_balance', 10000)

    async def run(self):
        """Main run method"""
        await self.initialize()
        
        # Setup signal handlers (platform-specific)
        if platform.system() != 'Windows':
            # Unix-style signal handlers
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        else:
            # Windows: Use keyboard interrupt handling instead
            self.logger.info("Windows detected - use Ctrl+C to stop")
        
        # Start execution engine background tasks
        await self.execution_engine.start_background_tasks(
            reconciliation_interval=300,  # 5 minutes
            order_poll_interval=30,       # 30 seconds
        )
        
        # Initial position reconciliation on startup
        self.logger.info("Running initial position reconciliation...")
        for exchange in ["binance", "coinbase", "kraken"]:
            try:
                results = await self.execution_engine.reconcile_positions(exchange)
                if results.get("error"):
                    self.logger.debug(f"Skipped {exchange} reconciliation: not connected")
                elif results.get("local_positions") or results.get("exchange_positions"):
                    self.logger.info(
                        f"{exchange.capitalize()} reconciliation: {len(results.get('matched', []))} matched"
                    )
            except Exception as e:
                self.logger.debug(f"Reconciliation skipped for {exchange}: {e}")
        
        # Build task list
        tasks = [
            self.scan_loop(),
            self.execution_loop(),
            self.metrics_loop(),
        ]
        
        # Add auto-execution task if enabled
        if self.auto_execute_signals:
            self.logger.info(
                f"Auto-execution ENABLED: min_confidence={self.auto_execute_min_confidence}, "
                f"confirmation_signals={self.auto_execute_confirmation_signals}"
            )
            tasks.append(self._auto_execute_loop())
        
        # Run all loops concurrently
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            # Stop execution engine background tasks
            await self.execution_engine.stop_background_tasks()
            await self.shutdown()
    
    async def _auto_execute_loop(self):
        """
        Background loop for automatic signal-to-trade execution.
        
        Only executes trades when:
        1. Signal confidence >= auto_execute_min_confidence
        2. At least auto_execute_confirmation_signals agree
        3. No existing position for the symbol
        4. Within risk limits
        """
        while not self._shutdown_event.is_set():
            if self.state != OrchestratorState.RUNNING:
                await asyncio.sleep(5)
                continue
            
            try:
                # Get high-confidence opportunities
                opportunities = self.signal_aggregator.get_top_opportunities(
                    min_score=self.auto_execute_min_confidence,
                    limit=self.max_concurrent_trades * 2,
                )
                
                current_positions = len(self.execution_engine.get_open_positions())
                available_slots = self.max_concurrent_trades - current_positions
                
                executed_count = 0
                for opp in opportunities[:available_slots]:
                    # Skip neutral signals
                    if opp["direction"] == "neutral":
                        continue
                    
                    # Check confirmation threshold
                    if opp["signal_count"] < self.auto_execute_confirmation_signals:
                        self.logger.debug(
                            f"Skipping {opp['symbol']}: only {opp['signal_count']} signals "
                            f"(need {self.auto_execute_confirmation_signals})"
                        )
                        continue
                    
                    # Check if already have position
                    existing = [
                        p for p in self.execution_engine.get_open_positions()
                        if p.symbol == opp["symbol"]
                    ]
                    if existing:
                        continue
                    
                    symbol = opp["symbol"]
                    
                    # Get current price
                    snapshot = await self.data_aggregator.get_market_snapshot(
                        [symbol.split("/")[0]]
                    )
                    if not snapshot:
                        continue
                    
                    tick = list(snapshot.values())[0] if snapshot else None
                    if not tick:
                        continue
                    
                    # Calculate position size with reduced risk for auto-execution
                    side = OrderSide.BUY if opp["direction"] == "long" else OrderSide.SELL
                    account_balance = await self.get_account_balance()
                    
                    # Use half the normal risk per trade for auto-execution
                    position_size = self.execution_engine.risk_manager.calculate_position_size(
                        account_balance=account_balance,
                        risk_per_trade_pct=1.0,  # Half of manual execution
                    )
                    quantity = position_size / tick.price
                    
                    # Log auto-execution
                    self.logger.info(
                        f"AUTO-EXECUTE: {side.value} {quantity:.6f} {symbol} @ ${tick.price:,.2f} | "
                        f"confidence={opp['confidence']:.2f}, signals={opp['signal_count']}"
                    )
                    
                    # Execute the trade
                    position = await self.execution_engine.open_position(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        entry_price=tick.price,
                    )
                    
                    if position:
                        executed_count += 1
                        self.metrics["trades_executed"] += 1
                        self.metrics["auto_executed"] += 1
                        
                        # Record in database
                        self.db.record_transaction(
                            account_id=1,
                            transaction_type="auto_trade",
                            asset=symbol.split("/")[0],
                            quantity=quantity,
                            price=tick.price,
                            side=side.value,
                            symbol=symbol,
                        )
                
                if executed_count > 0:
                    self.logger.info(f"Auto-execution cycle: {executed_count} trades opened")
                    
            except Exception as e:
                self.logger.error(f"Auto-execution loop error: {e}")
            
            # Check less frequently than manual execution
            await asyncio.sleep(self.execution_interval * 2)

    def enable_auto_execution(
        self,
        min_confidence: float = 0.75,
        confirmation_signals: int = 2
    ):
        """
        Enable automatic signal-to-trade execution.
        
        Args:
            min_confidence: Minimum signal confidence to execute (0.0-1.0)
            confirmation_signals: Minimum number of agreeing signals required
        """
        self.auto_execute_signals = True
        self.auto_execute_min_confidence = min_confidence
        self.auto_execute_confirmation_signals = confirmation_signals
        self.logger.info(
            f"Auto-execution enabled: min_confidence={min_confidence}, "
            f"confirmation_signals={confirmation_signals}"
        )
    
    def disable_auto_execution(self):
        """Disable automatic signal-to-trade execution"""
        self.auto_execute_signals = False
        self.logger.info("Auto-execution disabled")

    def get_status(self) -> Dict:
        """Get current orchestrator status"""
        return {
            "state": self.state.value,
            "uptime": str(datetime.now() - self.metrics["start_time"]) if self.metrics["start_time"] else "N/A",
            "theaters": {k: {
                "name": v.name,
                "active": v.is_active,
                "last_scan": v.last_scan.isoformat() if v.last_scan else None,
                "findings": v.findings_count,
                "actions": v.actions_count,
                "errors": v.errors_count,
            } for k, v in self.theaters.items()},
            "metrics": self.metrics,
            "open_positions": len(self.execution_engine.get_open_positions()),
            "total_exposure": self.execution_engine.get_total_exposure(),
            "unrealized_pnl": self.execution_engine.get_total_unrealized_pnl(),
        }


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ACC Orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--status", action="store_true", help="Show status only")
    args = parser.parse_args()
    
    async def main():
        orchestrator = Orchestrator()
        
        if args.status:
            await orchestrator.initialize()
            status = orchestrator.get_status()
            print(json.dumps(status, indent=2, default=str))
            await orchestrator.shutdown()
        else:
            print("=== ACC Orchestrator Starting ===")
            print("Press Ctrl+C to stop")
            await orchestrator.run()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
