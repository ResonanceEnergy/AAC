"""
AAC Cross-Department Integration Engine
=======================================

Connects the Bake-Off Engine to all AAC departments for unified
strategy management, risk monitoring, and performance tracking.

Departments:
- TradingExecution: Order management, execution, fills
- BigBrainIntelligence: Signals, research, ML models
- CentralAccounting: P&L, reconciliation, risk budget
- CryptoIntelligence: Venue routing, withdrawals, health
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DEPARTMENT INTERFACES
# ═══════════════════════════════════════════════════════════════════════════

class Department(Enum):
    """AAC organizational departments."""
    TRADING_EXECUTION = "TradingExecution"
    BIGBRAIN_INTELLIGENCE = "BigBrainIntelligence"
    CENTRAL_ACCOUNTING = "CentralAccounting"
    CRYPTO_INTELLIGENCE = "CryptoIntelligence"
    SHARED_INFRASTRUCTURE = "SharedInfrastructure"


@dataclass
class DepartmentMetric:
    """A metric produced or consumed by a department."""
    name: str
    department: Department
    value: float
    timestamp: datetime
    unit: str = ""
    is_healthy: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "department": self.department.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "unit": self.unit,
            "is_healthy": self.is_healthy
        }


@dataclass
class CrossDepartmentEvent:
    """An event that needs to be communicated across departments."""
    event_type: str
    source_department: Department
    target_departments: List[Department]
    payload: Dict[str, Any]
    priority: int = 1  # 1=highest
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "source": self.source_department.value,
            "targets": [d.value for d in self.target_departments],
            "payload": self.payload,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat()
        }


# ═══════════════════════════════════════════════════════════════════════════
# DEPARTMENT ADAPTERS
# ═══════════════════════════════════════════════════════════════════════════

class DepartmentAdapter:
    """Base class for department integration adapters."""
    
    def __init__(self, department: Department):
        self.department = department
        self.is_connected = False
        self._event_handlers: Dict[str, List[Callable]] = {}
        
    async def connect(self) -> bool:
        """Connect to department systems."""
        raise NotImplementedError
        
    async def disconnect(self) -> None:
        """Disconnect from department systems."""
        raise NotImplementedError
        
    async def get_metrics(self) -> List[DepartmentMetric]:
        """Get current metrics from department."""
        raise NotImplementedError
        
    async def send_event(self, event: CrossDepartmentEvent) -> bool:
        """Send an event to this department."""
        raise NotImplementedError
        
    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)


class TradingExecutionAdapter(DepartmentAdapter):
    """Adapter for TradingExecution department."""
    
    def __init__(self):
        super().__init__(Department.TRADING_EXECUTION)
        self.active_orders: Dict[str, Any] = {}
        self.positions: Dict[str, float] = {}
        
    async def connect(self) -> bool:
        """Connect to trading systems."""
        try:
            # In production: connect to execution_engine, order_manager
            logger.info("TradingExecution: Connecting to execution systems...")
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"TradingExecution connection failed: {e}")
            return False
            
    async def disconnect(self) -> None:
        """Disconnect from trading systems."""
        self.is_connected = False
        logger.info("TradingExecution: Disconnected")
        
    async def get_metrics(self) -> List[DepartmentMetric]:
        """Get execution metrics."""
        now = datetime.now()
        return [
            DepartmentMetric(
                name="fill_rate",
                department=self.department,
                value=0.95,  # Would query actual fill rate
                timestamp=now,
                unit="ratio",
                is_healthy=True
            ),
            DepartmentMetric(
                name="slippage_bps_p95",
                department=self.department,
                value=8.5,  # Would query actual slippage
                timestamp=now,
                unit="bps",
                is_healthy=True
            ),
            DepartmentMetric(
                name="time_to_fill_p95",
                department=self.department,
                value=450,  # ms
                timestamp=now,
                unit="ms",
                is_healthy=True
            ),
            DepartmentMetric(
                name="partial_fill_rate",
                department=self.department,
                value=0.12,
                timestamp=now,
                unit="ratio",
                is_healthy=True
            ),
        ]
        
    async def send_event(self, event: CrossDepartmentEvent) -> bool:
        """Handle incoming events."""
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"TradingExecution handler error: {e}")
        return True
        
    # TradingExecution specific methods
    async def submit_order(self, order: Dict[str, Any]) -> str:
        """Submit an order for execution."""
        order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.active_orders[order_id] = order
        logger.info(f"Order submitted: {order_id}")
        return order_id
        
    async def get_position(self, symbol: str) -> float:
        """Get current position for symbol."""
        return self.positions.get(symbol, 0.0)
        
    async def freeze_strategy(self, strategy_id: str) -> bool:
        """Freeze a strategy (A_FREEZE_STRATEGY action)."""
        logger.warning(f"FREEZING strategy: {strategy_id}")
        return True


class BigBrainIntelligenceAdapter(DepartmentAdapter):
    """Adapter for BigBrainIntelligence department."""
    
    def __init__(self):
        super().__init__(Department.BIGBRAIN_INTELLIGENCE)
        self.active_signals: Dict[str, Any] = {}
        self.research_queue: List[str] = []
        
    async def connect(self) -> bool:
        """Connect to research systems."""
        try:
            logger.info("BigBrainIntelligence: Connecting to research systems...")
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"BigBrainIntelligence connection failed: {e}")
            return False
            
    async def disconnect(self) -> None:
        self.is_connected = False
        logger.info("BigBrainIntelligence: Disconnected")
        
    async def get_metrics(self) -> List[DepartmentMetric]:
        """Get research/signal metrics."""
        now = datetime.now()
        return [
            DepartmentMetric(
                name="signal_strength",
                department=self.department,
                value=0.72,
                timestamp=now,
                unit="score",
                is_healthy=True
            ),
            DepartmentMetric(
                name="data_freshness_p95",
                department=self.department,
                value=2.5,  # seconds
                timestamp=now,
                unit="seconds",
                is_healthy=True
            ),
            DepartmentMetric(
                name="schema_mismatch_count",
                department=self.department,
                value=0,
                timestamp=now,
                unit="count",
                is_healthy=True
            ),
            DepartmentMetric(
                name="source_reliability",
                department=self.department,
                value=0.98,
                timestamp=now,
                unit="ratio",
                is_healthy=True
            ),
        ]
        
    async def send_event(self, event: CrossDepartmentEvent) -> bool:
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"BigBrainIntelligence handler error: {e}")
        return True
        
    # BigBrain specific methods
    async def get_signal(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get latest signal for a strategy."""
        return self.active_signals.get(strategy_id)
        
    async def quarantine_source(self, source_id: str) -> bool:
        """Quarantine a data source (A_QUARANTINE_SOURCE action)."""
        logger.warning(f"QUARANTINING data source: {source_id}")
        return True
        
    async def trigger_backtest(self, strategy_id: str, params: Dict[str, Any]) -> str:
        """Trigger a backtest job."""
        job_id = f"BT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        logger.info(f"Backtest triggered: {job_id} for strategy {strategy_id}")
        return job_id


class CentralAccountingAdapter(DepartmentAdapter):
    """Adapter for CentralAccounting department."""
    
    def __init__(self):
        super().__init__(Department.CENTRAL_ACCOUNTING)
        self.positions: Dict[str, float] = {}
        self.pnl_cache: Dict[str, float] = {}
        
    async def connect(self) -> bool:
        try:
            logger.info("CentralAccounting: Connecting to accounting systems...")
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"CentralAccounting connection failed: {e}")
            return False
            
    async def disconnect(self) -> None:
        self.is_connected = False
        logger.info("CentralAccounting: Disconnected")
        
    async def get_metrics(self) -> List[DepartmentMetric]:
        """Get accounting/risk metrics."""
        now = datetime.now()
        return [
            DepartmentMetric(
                name="reconciled_net_return",
                department=self.department,
                value=0.012,  # 1.2% return
                timestamp=now,
                unit="ratio",
                is_healthy=True
            ),
            DepartmentMetric(
                name="max_drawdown_pct",
                department=self.department,
                value=3.2,  # 3.2% drawdown
                timestamp=now,
                unit="percent",
                is_healthy=True
            ),
            DepartmentMetric(
                name="daily_loss_pct",
                department=self.department,
                value=0.5,
                timestamp=now,
                unit="percent",
                is_healthy=True
            ),
            DepartmentMetric(
                name="recon_backlog_minutes",
                department=self.department,
                value=2,
                timestamp=now,
                unit="minutes",
                is_healthy=True
            ),
            DepartmentMetric(
                name="capital_utilization",
                department=self.department,
                value=0.45,  # 45%
                timestamp=now,
                unit="ratio",
                is_healthy=True
            ),
        ]
        
    async def send_event(self, event: CrossDepartmentEvent) -> bool:
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"CentralAccounting handler error: {e}")
        return True
        
    # CentralAccounting specific methods
    async def get_reconciled_pnl(self, strategy_id: str) -> float:
        """Get reconciled P&L for a strategy."""
        return self.pnl_cache.get(strategy_id, 0.0)
        
    async def get_risk_budget_remaining(self) -> float:
        """Get remaining risk budget."""
        return 0.55  # 55% of risk budget remaining
        
    async def force_reconciliation(self, strategy_id: str) -> bool:
        """Force immediate reconciliation (A_FORCE_RECON action)."""
        logger.warning(f"FORCING reconciliation for: {strategy_id}")
        return True


class CryptoIntelligenceAdapter(DepartmentAdapter):
    """Adapter for CryptoIntelligence department."""
    
    def __init__(self):
        super().__init__(Department.CRYPTO_INTELLIGENCE)
        self.venue_health: Dict[str, float] = {}
        
    async def connect(self) -> bool:
        try:
            logger.info("CryptoIntelligence: Connecting to venue systems...")
            self.is_connected = True
            return True
        except Exception as e:
            logger.error(f"CryptoIntelligence connection failed: {e}")
            return False
            
    async def disconnect(self) -> None:
        self.is_connected = False
        logger.info("CryptoIntelligence: Disconnected")
        
    async def get_metrics(self) -> List[DepartmentMetric]:
        """Get venue health metrics."""
        now = datetime.now()
        return [
            DepartmentMetric(
                name="venue_health_score",
                department=self.department,
                value=0.92,
                timestamp=now,
                unit="score",
                is_healthy=True
            ),
            DepartmentMetric(
                name="withdrawal_reliability",
                department=self.department,
                value=0.99,
                timestamp=now,
                unit="ratio",
                is_healthy=True
            ),
            DepartmentMetric(
                name="latency_p99",
                department=self.department,
                value=85,  # ms
                timestamp=now,
                unit="ms",
                is_healthy=True
            ),
            DepartmentMetric(
                name="counterparty_exposure_pct",
                department=self.department,
                value=0.18,  # 18%
                timestamp=now,
                unit="ratio",
                is_healthy=True
            ),
        ]
        
    async def send_event(self, event: CrossDepartmentEvent) -> bool:
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"CryptoIntelligence handler error: {e}")
        return True
        
    # CryptoIntelligence specific methods
    async def get_venue_health(self, venue_id: str) -> float:
        """Get health score for a venue."""
        return self.venue_health.get(venue_id, 0.85)
        
    async def route_failover(self, from_venue: str, to_venue: str) -> bool:
        """Execute venue failover (A_ROUTE_FAILOVER action)."""
        logger.warning(f"FAILOVER: {from_venue} -> {to_venue}")
        return True


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-DEPARTMENT INTEGRATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class CrossDepartmentIntegrationEngine:
    """
    Central engine for cross-department communication and coordination.
    
    Responsibilities:
    - Connect to all department adapters
    - Route events between departments
    - Aggregate metrics for unified view
    - Coordinate AZ PRIME safety actions
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.adapters: Dict[Department, DepartmentAdapter] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self._metric_cache: Dict[str, DepartmentMetric] = {}
        
        # Initialize adapters
        self._init_adapters()
        
    def _init_adapters(self) -> None:
        """Initialize all department adapters."""
        self.adapters = {
            Department.TRADING_EXECUTION: TradingExecutionAdapter(),
            Department.BIGBRAIN_INTELLIGENCE: BigBrainIntelligenceAdapter(),
            Department.CENTRAL_ACCOUNTING: CentralAccountingAdapter(),
            Department.CRYPTO_INTELLIGENCE: CryptoIntelligenceAdapter(),
        }
        
    async def start(self) -> bool:
        """Start the integration engine and connect to all departments."""
        logger.info("Starting Cross-Department Integration Engine...")
        
        # Connect to all departments
        connect_results = await asyncio.gather(*[
            adapter.connect() for adapter in self.adapters.values()
        ])
        
        if not all(connect_results):
            logger.error("Failed to connect to all departments")
            return False
            
        self.is_running = True
        
        # Start event processing loop
        asyncio.create_task(self._process_events())
        
        # Start metric collection loop
        asyncio.create_task(self._collect_metrics())
        
        logger.info("Integration Engine started successfully")
        return True
        
    async def stop(self) -> None:
        """Stop the integration engine."""
        self.is_running = False
        
        # Disconnect from all departments
        await asyncio.gather(*[
            adapter.disconnect() for adapter in self.adapters.values()
        ])
        
        logger.info("Integration Engine stopped")
        
    async def _process_events(self) -> None:
        """Process cross-department events."""
        while self.is_running:
            try:
                # Wait for events with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                await self._route_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                
    async def _route_event(self, event: CrossDepartmentEvent) -> None:
        """Route an event to target departments."""
        logger.info(f"Routing event: {event.event_type} from {event.source_department.value}")
        
        for target_dept in event.target_departments:
            if target_dept in self.adapters:
                adapter = self.adapters[target_dept]
                await adapter.send_event(event)
                
    async def _collect_metrics(self) -> None:
        """Periodically collect metrics from all departments."""
        while self.is_running:
            try:
                # Collect metrics from all departments
                all_metrics = await asyncio.gather(*[
                    adapter.get_metrics() for adapter in self.adapters.values()
                ])
                
                # Flatten and cache metrics
                for dept_metrics in all_metrics:
                    for metric in dept_metrics:
                        key = f"{metric.department.value}.{metric.name}"
                        self._metric_cache[key] = metric
                        
                await asyncio.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                logger.error(f"Metric collection error: {e}")
                await asyncio.sleep(5)
                
    async def publish_event(self, event: CrossDepartmentEvent) -> None:
        """Publish an event to be routed to target departments."""
        await self.event_queue.put(event)
        
    def get_all_metrics(self) -> Dict[str, DepartmentMetric]:
        """Get all cached metrics."""
        return self._metric_cache.copy()
        
    def get_department_metrics(self, department: Department) -> List[DepartmentMetric]:
        """Get metrics for a specific department."""
        return [
            m for m in self._metric_cache.values()
            if m.department == department
        ]
        
    async def get_unified_health_status(self) -> Dict[str, Any]:
        """Get unified health status across all departments."""
        metrics = self.get_all_metrics()
        
        # Check critical thresholds
        alerts = []
        
        # Check risk metrics
        max_dd = metrics.get("CentralAccounting.max_drawdown_pct")
        if max_dd and max_dd.value > 10:
            alerts.append({
                "severity": "critical",
                "metric": "max_drawdown_pct",
                "value": max_dd.value,
                "action": "A_ENTER_SAFE_MODE"
            })
            
        daily_loss = metrics.get("CentralAccounting.daily_loss_pct")
        if daily_loss and daily_loss.value > 2:
            alerts.append({
                "severity": "critical",
                "metric": "daily_loss_pct",
                "value": daily_loss.value,
                "action": "A_STOP_EXECUTION"
            })
            
        # Check execution metrics
        fill_rate = metrics.get("TradingExecution.fill_rate")
        if fill_rate and fill_rate.value < 0.9:
            alerts.append({
                "severity": "warning",
                "metric": "fill_rate",
                "value": fill_rate.value,
                "action": "A_THROTTLE_RISK"
            })
            
        # Check venue health
        venue_health = metrics.get("CryptoIntelligence.venue_health_score")
        if venue_health and venue_health.value < 0.7:
            alerts.append({
                "severity": "warning",
                "metric": "venue_health_score",
                "value": venue_health.value,
                "action": "A_ROUTE_FAILOVER"
            })
            
        # Determine overall health
        if any(a["severity"] == "critical" for a in alerts):
            overall_health = "CRITICAL"
        elif any(a["severity"] == "warning" for a in alerts):
            overall_health = "WARNING"
        else:
            overall_health = "HEALTHY"
            
        return {
            "overall_health": overall_health,
            "timestamp": datetime.now().isoformat(),
            "departments_connected": len([a for a in self.adapters.values() if a.is_connected]),
            "total_departments": len(self.adapters),
            "metric_count": len(metrics),
            "alerts": alerts
        }
        
    # ═══════════════════════════════════════════════════════════════════════
    # AZ PRIME SAFETY ACTIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    async def execute_safety_action(self, action: str, params: Dict[str, Any] = None) -> bool:
        """Execute an AZ PRIME safety action."""
        params = params or {}
        
        logger.warning(f"EXECUTING SAFETY ACTION: {action}")
        
        if action == "A_THROTTLE_RISK":
            # Reduce order size and concurrency across TradingExecution
            event = CrossDepartmentEvent(
                event_type="THROTTLE_RISK",
                source_department=Department.SHARED_INFRASTRUCTURE,
                target_departments=[Department.TRADING_EXECUTION],
                payload={"reduction_factor": 0.5},
                priority=1
            )
            await self.publish_event(event)
            return True
            
        elif action == "A_QUARANTINE_SOURCE":
            # Quarantine bad data source via BigBrain
            source_id = params.get("source_id")
            if source_id:
                adapter = self.adapters[Department.BIGBRAIN_INTELLIGENCE]
                return await adapter.quarantine_source(source_id)
            return False
            
        elif action == "A_ROUTE_FAILOVER":
            # Failover to backup venue via CryptoIntelligence
            from_venue = params.get("from_venue")
            to_venue = params.get("to_venue")
            if from_venue and to_venue:
                adapter = self.adapters[Department.CRYPTO_INTELLIGENCE]
                return await adapter.route_failover(from_venue, to_venue)
            return False
            
        elif action == "A_STOP_EXECUTION":
            # Stop all execution immediately
            event = CrossDepartmentEvent(
                event_type="STOP_EXECUTION",
                source_department=Department.SHARED_INFRASTRUCTURE,
                target_departments=[Department.TRADING_EXECUTION],
                payload={"reason": params.get("reason", "Safety action triggered")},
                priority=1
            )
            await self.publish_event(event)
            return True
            
        elif action == "A_FREEZE_STRATEGY":
            # Freeze specific strategy
            strategy_id = params.get("strategy_id")
            if strategy_id:
                adapter = self.adapters[Department.TRADING_EXECUTION]
                return await adapter.freeze_strategy(strategy_id)
            return False
            
        elif action == "A_FORCE_RECON":
            # Force reconciliation
            strategy_id = params.get("strategy_id", "ALL")
            adapter = self.adapters[Department.CENTRAL_ACCOUNTING]
            return await adapter.force_reconciliation(strategy_id)
            
        elif action == "A_CREATE_INCIDENT":
            # Create incident (would integrate with incident management)
            logger.critical(f"INCIDENT CREATED: {params}")
            return True
            
        elif action == "A_PAGE_ONCALL":
            # Page on-call (would integrate with PagerDuty/similar)
            logger.critical(f"PAGING ON-CALL: {params}")
            return True
            
        else:
            logger.error(f"Unknown safety action: {action}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH BAKE-OFF ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class BakeoffIntegration:
    """
    Integrates the Bake-Off Engine with cross-department systems.
    """
    
    def __init__(self, integration_engine: CrossDepartmentIntegrationEngine):
        self.integration_engine = integration_engine
        
    async def get_strategy_metrics_for_bakeoff(self, strategy_id: str) -> Dict[str, float]:
        """
        Collect metrics for bake-off composite scoring from all departments.
        
        Returns metrics needed for P-R-E-D-O-F scoring:
        - P: Performance (from CentralAccounting)
        - R: Risk (from CentralAccounting)
        - E: Execution (from TradingExecution)
        - D: Data (from BigBrainIntelligence)
        - O: Ops (from SharedInfrastructure)
        - F: Fragility (from all departments)
        """
        all_metrics = self.integration_engine.get_all_metrics()
        
        return {
            # Performance metrics
            "reconciled_net_return": all_metrics.get("CentralAccounting.reconciled_net_return", DepartmentMetric(
                name="reconciled_net_return", department=Department.CENTRAL_ACCOUNTING, value=0, timestamp=datetime.now()
            )).value,
            "sharpe_ratio": 1.2,  # Would calculate from returns
            
            # Risk metrics
            "max_drawdown_pct": all_metrics.get("CentralAccounting.max_drawdown_pct", DepartmentMetric(
                name="max_drawdown_pct", department=Department.CENTRAL_ACCOUNTING, value=0, timestamp=datetime.now()
            )).value,
            "daily_loss_pct": all_metrics.get("CentralAccounting.daily_loss_pct", DepartmentMetric(
                name="daily_loss_pct", department=Department.CENTRAL_ACCOUNTING, value=0, timestamp=datetime.now()
            )).value,
            
            # Execution metrics
            "fill_rate": all_metrics.get("TradingExecution.fill_rate", DepartmentMetric(
                name="fill_rate", department=Department.TRADING_EXECUTION, value=0.95, timestamp=datetime.now()
            )).value,
            "slippage_bps_p95": all_metrics.get("TradingExecution.slippage_bps_p95", DepartmentMetric(
                name="slippage_bps_p95", department=Department.TRADING_EXECUTION, value=10, timestamp=datetime.now()
            )).value,
            
            # Data metrics
            "data_freshness_p95": all_metrics.get("BigBrainIntelligence.data_freshness_p95", DepartmentMetric(
                name="data_freshness_p95", department=Department.BIGBRAIN_INTELLIGENCE, value=2, timestamp=datetime.now()
            )).value,
            "source_reliability": all_metrics.get("BigBrainIntelligence.source_reliability", DepartmentMetric(
                name="source_reliability", department=Department.BIGBRAIN_INTELLIGENCE, value=0.98, timestamp=datetime.now()
            )).value,
            
            # Ops metrics
            "recon_backlog_minutes": all_metrics.get("CentralAccounting.recon_backlog_minutes", DepartmentMetric(
                name="recon_backlog_minutes", department=Department.CENTRAL_ACCOUNTING, value=2, timestamp=datetime.now()
            )).value,
            
            # Fragility metrics
            "venue_health_score": all_metrics.get("CryptoIntelligence.venue_health_score", DepartmentMetric(
                name="venue_health_score", department=Department.CRYPTO_INTELLIGENCE, value=0.9, timestamp=datetime.now()
            )).value,
        }
        
    async def check_gate_requirements(self, strategy_id: str, gate: str) -> Dict[str, Any]:
        """
        Check if a strategy meets requirements for a bake-off gate.
        
        Aggregates checks across all relevant departments.
        """
        metrics = await self.get_strategy_metrics_for_bakeoff(strategy_id)
        
        gate_checks = {
            "SPEC": {
                "hypothesis_documented": True,  # Would check BigBrain
                "data_contracts_defined": True,  # Would check BigBrain
                "risk_envelope_specified": True,  # Would check CentralAccounting
            },
            "SIM": {
                "backtest_complete": True,  # Would check BigBrain
                "stress_test_passed": True,  # Would check BigBrain
                "metrics_within_bounds": metrics["max_drawdown_pct"] < 15,
            },
            "PAPER": {
                "live_data_connected": True,  # Would check BigBrain
                "latency_meets_slo": metrics["data_freshness_p95"] < 5,
                "2_weeks_runtime": True,  # Would check log history
            },
            "PILOT": {
                "paper_gate_passed": True,
                "risk_approval": True,  # Would check CentralAccounting
                "kill_switch_armed": True,  # Would check TradingExecution
            },
            "POST_ANALYSIS": {
                "4_weeks_data": True,  # Would check log history
                "reconciliation_complete": metrics["recon_backlog_minutes"] < 60,
                "attribution_done": True,  # Would check CentralAccounting
            },
        }
        
        checks = gate_checks.get(gate, {})
        passed = all(checks.values())
        
        return {
            "gate": gate,
            "strategy_id": strategy_id,
            "passed": passed,
            "checks": checks,
            "metrics_snapshot": metrics
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Demo the cross-department integration engine."""
    
    # Create and start integration engine
    engine = CrossDepartmentIntegrationEngine()
    await engine.start()
    
    # Wait for initial metrics collection
    await asyncio.sleep(2)
    
    # Get unified health status
    health = await engine.get_unified_health_status()
    print("\n=== UNIFIED HEALTH STATUS ===")
    print(f"Overall: {health['overall_health']}")
    print(f"Departments: {health['departments_connected']}/{health['total_departments']}")
    print(f"Metrics: {health['metric_count']}")
    print(f"Alerts: {len(health['alerts'])}")
    
    # Get all metrics
    metrics = engine.get_all_metrics()
    print("\n=== ALL METRICS ===")
    for key, metric in metrics.items():
        print(f"  {key}: {metric.value} {metric.unit}")
    
    # Test bake-off integration
    bakeoff = BakeoffIntegration(engine)
    
    print("\n=== BAKE-OFF METRICS ===")
    bakeoff_metrics = await bakeoff.get_strategy_metrics_for_bakeoff("s01_etf_nav")
    for key, value in bakeoff_metrics.items():
        print(f"  {key}: {value}")
    
    print("\n=== GATE CHECK ===")
    gate_result = await bakeoff.check_gate_requirements("s01_etf_nav", "PAPER")
    print(f"Gate: {gate_result['gate']}")
    print(f"Passed: {gate_result['passed']}")
    for check, result in gate_result['checks'].items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}")
    
    # Test safety action
    print("\n=== SAFETY ACTION TEST ===")
    await engine.execute_safety_action("A_THROTTLE_RISK")
    
    # Stop engine
    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
