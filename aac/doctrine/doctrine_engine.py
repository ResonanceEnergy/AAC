"""
AAC Doctrine Application Engine
===============================

Analyzes and applies all 8 Doctrine Packs across the organization.
Provides automated compliance checking, gap detection, and enforcement.

Doctrine Packs:
1. Risk Envelope & Capital Allocation (CentralAccounting)
2. Security / Secrets / IAM / Key Custody (SharedInfrastructure)
3. Testing / Simulation / Replay / Chaos (BigBrainIntelligence)
4. Incident Response + On-Call + Postmortems (SharedInfrastructure)
5. Liquidity / Market Impact / Partial Fill Logic (TradingExecution)
6. Counterparty Scoring + Venue Health (CryptoIntelligence)
7. Research Factory + Experimentation (BigBrainIntelligence)
8. Metric Canon + Truth Arbitration (CentralAccounting)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import yaml
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("DoctrineEngine")


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS AND CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

class Department(Enum):
    """AAC organizational departments."""
    TRADING_EXECUTION = "TradingExecution"
    BIGBRAIN_INTELLIGENCE = "BigBrainIntelligence"
    CENTRAL_ACCOUNTING = "CentralAccounting"
    CRYPTO_INTELLIGENCE = "CryptoIntelligence"
    SHARED_INFRASTRUCTURE = "SharedInfrastructure"


class ComplianceState(Enum):
    """Doctrine compliance states."""
    COMPLIANT = "COMPLIANT"
    WARNING = "WARNING"
    VIOLATION = "VIOLATION"
    UNKNOWN = "UNKNOWN"


class AZPrimeState(Enum):
    """AZ Prime operational states."""
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    SAFE_MODE = "SAFE_MODE"
    HALT = "HALT"


class ActionType(Enum):
    """Automated action types from doctrine packs."""
    A_STOP_EXECUTION = "A_STOP_EXECUTION"
    A_THROTTLE_RISK = "A_THROTTLE_RISK"
    A_ENTER_SAFE_MODE = "A_ENTER_SAFE_MODE"
    A_PAGE_ONCALL = "A_PAGE_ONCALL"
    A_CREATE_INCIDENT = "A_CREATE_INCIDENT"
    A_FREEZE_STRATEGY = "A_FREEZE_STRATEGY"
    A_ROUTE_FAILOVER = "A_ROUTE_FAILOVER"
    A_LOCK_KEYS = "A_LOCK_KEYS"
    A_QUARANTINE_SOURCE = "A_QUARANTINE_SOURCE"
    A_FORCE_RECON = "A_FORCE_RECON"


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MetricValue:
    """A metric value with threshold evaluation."""
    name: str
    value: Any
    threshold_good: str
    threshold_warning: str
    threshold_critical: str
    state: ComplianceState = ComplianceState.UNKNOWN
    department: Optional[Department] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DoctrineViolation:
    """A doctrine violation requiring action."""
    pack_id: int
    pack_name: str
    rule_type: str  # 'metric', 'failure_mode', 'az_prime_hook'
    rule_id: str
    description: str
    severity: str  # 'warning', 'critical'
    department: Department
    recommended_action: ActionType
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False


@dataclass
class DoctrineComplianceReport:
    """Compliance report for a department or the entire org."""
    generated_at: datetime
    scope: str  # department name or 'organization'
    total_rules: int
    compliant: int
    warnings: int
    violations: int
    violations_list: List[DoctrineViolation]
    compliance_score: float  # 0-100
    az_prime_state: AZPrimeState


# ═══════════════════════════════════════════════════════════════════════════
# DOCTRINE PACK DEFINITIONS
# Complete metric registry aligned with all department adapters
# ═══════════════════════════════════════════════════════════════════════════

DOCTRINE_PACKS = {
    1: {
        "name": "Risk Envelope & Capital Allocation",
        "owner": Department.CENTRAL_ACCOUNTING,
        "key_metrics": [
            # Core risk metrics
            "max_drawdown_pct", "daily_loss_pct", "tail_loss_p99",
            "capital_utilization", "margin_buffer", "strategy_correlation_matrix",
            "stressed_var_99", "portfolio_heat"
        ],
        "failure_modes": [
            "cascading_liquidation", "correlated_drawdown", "leverage_trap"
        ],
        "az_prime_triggers": [
            ("drawdown_exceeds_5pct", AZPrimeState.CAUTION, ActionType.A_THROTTLE_RISK),
            ("drawdown_exceeds_10pct", AZPrimeState.SAFE_MODE, ActionType.A_STOP_EXECUTION),
            ("daily_loss_exceeds_2pct", AZPrimeState.HALT, ActionType.A_STOP_EXECUTION),
        ]
    },
    2: {
        "name": "Security / Secrets / IAM / Key Custody",
        "owner": Department.SHARED_INFRASTRUCTURE,
        "key_metrics": [
            "key_age_days", "failed_auth_rate", "audit_log_completeness",
            "mfa_compliance_rate", "secret_scan_coverage"
        ],
        "failure_modes": [
            "key_compromise", "audit_gap"
        ],
        "az_prime_triggers": [
            ("failed_auth_count > 5/min", AZPrimeState.SAFE_MODE, ActionType.A_LOCK_KEYS),
            ("key_exposure_detected", AZPrimeState.HALT, ActionType.A_LOCK_KEYS),
        ]
    },
    3: {
        "name": "Testing / Simulation / Replay / Chaos",
        "owner": Department.BIGBRAIN_INTELLIGENCE,
        "key_metrics": [
            "backtest_vs_live_correlation", "chaos_test_pass_rate",
            "regression_test_pass_rate", "replay_fidelity_score"
        ],
        "failure_modes": [
            "test_environment_drift", "flaky_test_syndrome",
            "backtest_overfitting", "replay_data_corruption"
        ],
        "az_prime_triggers": [
            ("test_coverage_pct < 70", AZPrimeState.CAUTION, ActionType.A_CREATE_INCIDENT),
            ("regression_test_failures > 0", AZPrimeState.CAUTION, ActionType.A_STOP_EXECUTION),
            ("chaos_test_failed", AZPrimeState.SAFE_MODE, ActionType.A_ENTER_SAFE_MODE),
            ("backtest_vs_live_drift > 50%", AZPrimeState.CAUTION, ActionType.A_FREEZE_STRATEGY),
        ]
    },
    4: {
        "name": "Incident Response + On-Call + Postmortems",
        "owner": Department.SHARED_INFRASTRUCTURE,
        "key_metrics": [
            "mttd_minutes", "mttr_minutes", "incident_recurrence_rate",
            "active_sev1_count"
        ],
        "failure_modes": [
            "alert_fatigue", "escalation_failure",
            "communication_breakdown", "incomplete_postmortem"
        ],
        "az_prime_triggers": [
            ("active_sev1_count > 0", AZPrimeState.CAUTION, ActionType.A_THROTTLE_RISK),
            ("mttd > 10min for sev1", AZPrimeState.CAUTION, ActionType.A_CREATE_INCIDENT),
            ("mttr > 60min for sev1", AZPrimeState.SAFE_MODE, ActionType.A_ENTER_SAFE_MODE),
            ("incident_recurrence within 7 days", AZPrimeState.CAUTION, ActionType.A_CREATE_INCIDENT),
        ]
    },
    5: {
        "name": "Liquidity / Market Impact / Partial Fill Logic",
        "owner": Department.TRADING_EXECUTION,
        "key_metrics": [
            "fill_rate", "time_to_fill_p95", "slippage_bps",
            "partial_fill_rate", "adverse_selection_cost",
            "market_impact_bps", "liquidity_available_pct"
        ],
        "failure_modes": [
            "liquidity_mirage", "market_impact_underestimation",
            "partial_fill_cascade", "adverse_selection_trap"
        ],
        "az_prime_triggers": [
            ("slippage_bps_p95 > 10", AZPrimeState.CAUTION, ActionType.A_THROTTLE_RISK),
            ("partial_fill_rate > 30%", AZPrimeState.CAUTION, ActionType.A_CREATE_INCIDENT),
            ("market_impact_bps > 20", AZPrimeState.SAFE_MODE, ActionType.A_STOP_EXECUTION),
            ("liquidity_available_pct < 100%", AZPrimeState.CAUTION, ActionType.A_THROTTLE_RISK),
        ]
    },
    6: {
        "name": "Counterparty Scoring + Venue Health + Withdrawal Risk",
        "owner": Department.CRYPTO_INTELLIGENCE,
        "key_metrics": [
            "venue_health_score", "withdrawal_success_rate",
            "counterparty_exposure_pct", "settlement_failure_rate",
            "counterparty_credit_score"
        ],
        "failure_modes": [
            "venue_insolvency", "withdrawal_freeze"
        ],
        "az_prime_triggers": [
            ("venue_health_score < 0.70", AZPrimeState.CAUTION, ActionType.A_ROUTE_FAILOVER),
            ("withdrawal_frozen", AZPrimeState.SAFE_MODE, ActionType.A_CREATE_INCIDENT),
            ("settlement_failure_rate > 1%", AZPrimeState.CAUTION, ActionType.A_THROTTLE_RISK),
            ("counterparty_credit_score < 50", AZPrimeState.SAFE_MODE, ActionType.A_ROUTE_FAILOVER),
        ]
    },
    7: {
        "name": "Research Factory + Experimentation + Strategy Retirement",
        "owner": Department.BIGBRAIN_INTELLIGENCE,
        "key_metrics": [
            "research_pipeline_velocity", "strategy_survival_rate",
            "feature_reuse_rate", "experiment_completion_rate"
        ],
        "failure_modes": [
            "research_stagnation", "strategy_overfitting",
            "feature_bloat", "experiment_sprawl"
        ],
        "az_prime_triggers": [
            ("research_pipeline_velocity < 1", AZPrimeState.CAUTION, ActionType.A_CREATE_INCIDENT),
            ("strategy_survival_rate < 25%", AZPrimeState.CAUTION, ActionType.A_FREEZE_STRATEGY),
            ("experiment_completion_rate < 50%", AZPrimeState.CAUTION, ActionType.A_CREATE_INCIDENT),
            ("model_version_rollback_count > 2 in 7 days", AZPrimeState.SAFE_MODE, ActionType.A_FREEZE_STRATEGY),
        ]
    },
    8: {
        "name": "Metric Canon + Truth Arbitration + Retention/Privacy",
        "owner": Department.CENTRAL_ACCOUNTING,
        "key_metrics": [
            "data_quality_score", "metric_lineage_coverage",
            "reconciliation_accuracy", "truth_arbitration_latency"
        ],
        "failure_modes": [
            "metric_drift", "truth_conflict_stalemate",
            "retention_policy_violation", "dashboard_stale_data"
        ],
        "az_prime_triggers": [
            ("data_quality_score < 0.85", AZPrimeState.CAUTION, ActionType.A_QUARANTINE_SOURCE),
            ("reconciliation_accuracy < 95%", AZPrimeState.CAUTION, ActionType.A_FORCE_RECON),
            ("truth_conflict_unresolved > 30min", AZPrimeState.SAFE_MODE, ActionType.A_PAGE_ONCALL),
            ("metric_lineage_coverage < 70%", AZPrimeState.CAUTION, ActionType.A_CREATE_INCIDENT),
        ]
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DOCTRINE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class DoctrineEngine:
    """
    Core engine for doctrine analysis and application.
    
    Responsibilities:
    - Load and parse doctrine packs from YAML
    - Check compliance across all departments
    - Generate violations and recommended actions
    - Track AZ Prime state transitions
    - Execute automated remediation actions
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent.parent / "config" / "doctrine_packs.yaml"
        self.doctrine_packs: Dict[int, Dict] = {}
        self.current_az_state: AZPrimeState = AZPrimeState.NORMAL
        self.active_violations: List[DoctrineViolation] = []
        self.metric_values: Dict[str, MetricValue] = {}
        self.action_handlers: Dict[ActionType, Callable] = {}
        self._loaded = False
        
    def load_doctrine_packs(self) -> bool:
        """Load doctrine packs from YAML configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    raw = yaml.safe_load(f)
                    self.doctrine_packs = raw.get('doctrine_packs', {})
                    self._loaded = True
                    logger.info(f"Loaded {len(self.doctrine_packs)} doctrine packs from {self.config_path}")
                    return True
            else:
                logger.warning(f"Doctrine config not found at {self.config_path}, using defaults")
                self.doctrine_packs = DOCTRINE_PACKS
                self._loaded = True
                return True
        except Exception as e:
            logger.error(f"Failed to load doctrine packs: {e}")
            self.doctrine_packs = DOCTRINE_PACKS
            self._loaded = True
            return False
    
    def register_action_handler(self, action: ActionType, handler: Callable) -> None:
        """Register a handler for an automated action type."""
        self.action_handlers[action] = handler
        logger.info(f"Registered handler for {action.value}")
    
    def update_metric(self, name: str, value: Any, department: Department) -> MetricValue:
        """Update a metric value and check compliance."""
        metric = self._find_metric_definition(name)
        if metric:
            mv = MetricValue(
                name=name,
                value=value,
                threshold_good=metric.get('thresholds', {}).get('good', ''),
                threshold_warning=metric.get('thresholds', {}).get('warning', ''),
                threshold_critical=metric.get('thresholds', {}).get('critical', ''),
                department=department,
                timestamp=datetime.now()
            )
            mv.state = self._evaluate_threshold(value, mv.threshold_good, mv.threshold_warning, mv.threshold_critical)
            self.metric_values[name] = mv
            
            # Check for violations
            if mv.state == ComplianceState.VIOLATION:
                self._create_violation_from_metric(mv)
            
            return mv
        else:
            logger.warning(f"Metric {name} not found in doctrine packs")
            return MetricValue(name=name, value=value, threshold_good='', threshold_warning='', threshold_critical='')
    
    def _find_metric_definition(self, name: str) -> Optional[Dict]:
        """Find a metric definition in doctrine packs."""
        for pack_key, pack_data in self.doctrine_packs.items():
            if isinstance(pack_data, dict):
                metrics = pack_data.get('required_metrics', [])
                if isinstance(metrics, list):
                    for m in metrics:
                        if isinstance(m, dict) and m.get('metric') == name:
                            return m
        return None
    
    def _evaluate_threshold(self, value: Any, good: str, warning: str, critical: str) -> ComplianceState:
        """Evaluate a value against thresholds."""
        try:
            if isinstance(value, (int, float)):
                # Parse threshold strings like "< 5%", "> 0.85", "5-10%"
                # Check good first - if in good range, compliant
                if self._check_threshold(value, good):
                    return ComplianceState.COMPLIANT
                # Check if in critical zone
                elif self._check_threshold(value, critical):
                    return ComplianceState.VIOLATION
                # Otherwise warning
                elif self._check_threshold(value, warning):
                    return ComplianceState.WARNING
                # Default to warning if no thresholds matched
                else:
                    return ComplianceState.WARNING
        except Exception:
            pass
        return ComplianceState.UNKNOWN
    
    def _check_threshold(self, value: float, threshold_str: str) -> bool:
        """Check if a value satisfies a threshold string."""
        if not threshold_str:
            return False
            
        threshold_str = threshold_str.strip().replace('%', '')
        
        try:
            if threshold_str.startswith('<'):
                limit = float(threshold_str[1:].strip())
                return value < limit
            elif threshold_str.startswith('>'):
                limit = float(threshold_str[1:].strip())
                return value > limit
            elif '-' in threshold_str and not threshold_str.startswith('-'):
                parts = threshold_str.split('-')
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return low <= value <= high
        except (ValueError, IndexError):
            pass
        return False
    
    def _create_violation_from_metric(self, metric: MetricValue) -> None:
        """Create a violation record from a metric breach."""
        # Find the pack this metric belongs to
        for pack_id, pack_info in DOCTRINE_PACKS.items():
            if metric.name in pack_info.get('key_metrics', []):
                violation = DoctrineViolation(
                    pack_id=pack_id,
                    pack_name=pack_info['name'],
                    rule_type='metric',
                    rule_id=metric.name,
                    description=f"Metric {metric.name} = {metric.value} exceeds critical threshold",
                    severity='critical',
                    department=pack_info['owner'],
                    recommended_action=self._get_recommended_action(pack_id, metric.name)
                )
                self.active_violations.append(violation)
                logger.warning(f"VIOLATION: {violation.description}")
                break
    
    def _get_recommended_action(self, pack_id: int, metric_name: str) -> ActionType:
        """Get recommended action for a metric violation."""
        pack = DOCTRINE_PACKS.get(pack_id, {})
        triggers = pack.get('az_prime_triggers', [])
        
        for trigger_str, state, action in triggers:
            if metric_name in trigger_str:
                return action
        
        # Default actions by pack
        default_actions = {
            1: ActionType.A_THROTTLE_RISK,
            2: ActionType.A_LOCK_KEYS,
            3: ActionType.A_CREATE_INCIDENT,
            4: ActionType.A_PAGE_ONCALL,
            5: ActionType.A_THROTTLE_RISK,
            6: ActionType.A_ROUTE_FAILOVER,
            7: ActionType.A_FREEZE_STRATEGY,
            8: ActionType.A_FORCE_RECON,
        }
        return default_actions.get(pack_id, ActionType.A_CREATE_INCIDENT)
    
    def check_az_prime_triggers(self, metrics: Dict[str, Any]) -> Tuple[AZPrimeState, List[ActionType]]:
        """Check all AZ Prime triggers and return state + required actions."""
        triggered_actions: List[ActionType] = []
        worst_state = AZPrimeState.NORMAL
        
        for pack_id, pack_info in DOCTRINE_PACKS.items():
            triggers = pack_info.get('az_prime_triggers', [])
            
            for trigger_str, target_state, action in triggers:
                if self._evaluate_trigger(trigger_str, metrics):
                    triggered_actions.append(action)
                    if self._state_severity(target_state) > self._state_severity(worst_state):
                        worst_state = target_state
                    logger.warning(f"AZ Prime trigger fired: {trigger_str} → {target_state.value}")
        
        return worst_state, triggered_actions
    
    def _evaluate_trigger(self, trigger_str: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate if a trigger condition is met."""
        # Simple parsing for common patterns
        try:
            if '>' in trigger_str:
                parts = trigger_str.split('>')
                metric_name = parts[0].strip().replace('_', ' ').replace(' ', '_')
                threshold = float(parts[1].strip().replace('%', '').split()[0])
                value = metrics.get(metric_name, 0)
                return value > threshold
            elif '<' in trigger_str:
                parts = trigger_str.split('<')
                metric_name = parts[0].strip().replace('_', ' ').replace(' ', '_')
                threshold = float(parts[1].strip().replace('%', '').split()[0])
                value = metrics.get(metric_name, float('inf'))
                return value < threshold
        except Exception:
            pass
        return False
    
    def _state_severity(self, state: AZPrimeState) -> int:
        """Get numeric severity for state comparison."""
        severities = {
            AZPrimeState.NORMAL: 0,
            AZPrimeState.CAUTION: 1,
            AZPrimeState.SAFE_MODE: 2,
            AZPrimeState.HALT: 3,
        }
        return severities.get(state, 0)
    
    async def execute_action(self, action: ActionType, context: Dict[str, Any] = None) -> bool:
        """Execute an automated action."""
        context = context or {}
        
        if action in self.action_handlers:
            try:
                result = self.action_handlers[action](context)
                if asyncio.iscoroutine(result):
                    result = await result
                logger.info(f"Executed action {action.value}: {result}")
                return True
            except Exception as e:
                logger.error(f"Action {action.value} failed: {e}")
                return False
        else:
            logger.warning(f"No handler registered for action {action.value}")
            return False
    
    def generate_compliance_report(self, department: Optional[Department] = None) -> DoctrineComplianceReport:
        """Generate a compliance report for a department or entire org."""
        scope = department.value if department else "organization"
        violations = [v for v in self.active_violations if not department or v.department == department]
        
        # Count metrics by state
        compliant = sum(1 for m in self.metric_values.values() 
                       if m.state == ComplianceState.COMPLIANT 
                       and (not department or m.department == department))
        warnings = sum(1 for m in self.metric_values.values() 
                      if m.state == ComplianceState.WARNING 
                      and (not department or m.department == department))
        violation_count = len(violations)
        
        total = compliant + warnings + violation_count
        score = (compliant / total * 100) if total > 0 else 100.0
        
        return DoctrineComplianceReport(
            generated_at=datetime.now(),
            scope=scope,
            total_rules=total,
            compliant=compliant,
            warnings=warnings,
            violations=violation_count,
            violations_list=violations,
            compliance_score=round(score, 2),
            az_prime_state=self.current_az_state
        )
    
    def get_department_doctrine_summary(self, department: Department) -> Dict[str, Any]:
        """Get a summary of doctrines applicable to a department."""
        applicable_packs = []
        
        for pack_id, pack_info in DOCTRINE_PACKS.items():
            if pack_info.get('owner') == department:
                applicable_packs.append({
                    'pack_id': pack_id,
                    'name': pack_info['name'],
                    'metrics': pack_info.get('key_metrics', []),
                    'failure_modes': pack_info.get('failure_modes', []),
                    'trigger_count': len(pack_info.get('az_prime_triggers', []))
                })
        
        return {
            'department': department.value,
            'primary_packs': applicable_packs,
            'total_metrics': sum(len(p['metrics']) for p in applicable_packs),
            'total_failure_modes': sum(len(p['failure_modes']) for p in applicable_packs),
        }


# ═══════════════════════════════════════════════════════════════════════════
# DOCTRINE APPLICATION SERVICE
# ═══════════════════════════════════════════════════════════════════════════

class DoctrineApplicationService:
    """
    High-level service for applying doctrine across the organization.
    
    Integrates with:
    - Cross-Department Engine for coordination
    - Individual department systems for metrics
    - Alerting systems for notifications
    """
    
    def __init__(self, engine: DoctrineEngine):
        self.engine = engine
        self.department_connections: Dict[Department, bool] = {}
        
    async def initialize(self) -> bool:
        """Initialize the doctrine application service."""
        # Load doctrine packs
        self.engine.load_doctrine_packs()
        
        # Register default action handlers
        self._register_default_handlers()
        
        logger.info("DoctrineApplicationService initialized")
        return True
    
    def _register_default_handlers(self) -> None:
        """Register default handlers for all action types."""
        
        def create_incident(ctx):
            logger.warning(f"[INCIDENT] Creating incident: {ctx}")
            return True
        
        def page_oncall(ctx):
            logger.critical(f"[PAGE] Paging on-call: {ctx}")
            return True
        
        def throttle_risk(ctx):
            logger.warning(f"[RISK] Throttling risk: {ctx}")
            return True
        
        def stop_execution(ctx):
            logger.critical(f"[EXEC] Stopping execution: {ctx}")
            return True
        
        def enter_safe_mode(ctx):
            logger.critical(f"[SAFE_MODE] Entering safe mode: {ctx}")
            return True
        
        def freeze_strategy(ctx):
            logger.warning(f"[STRATEGY] Freezing strategy: {ctx}")
            return True
        
        def route_failover(ctx):
            logger.warning(f"[ROUTING] Initiating failover: {ctx}")
            return True
        
        def lock_keys(ctx):
            logger.critical(f"[SECURITY] Locking keys: {ctx}")
            return True
        
        def quarantine_source(ctx):
            logger.warning(f"[DATA] Quarantining source: {ctx}")
            return True
        
        def force_recon(ctx):
            logger.warning(f"[RECON] Forcing reconciliation: {ctx}")
            return True
        
        self.engine.register_action_handler(ActionType.A_CREATE_INCIDENT, create_incident)
        self.engine.register_action_handler(ActionType.A_PAGE_ONCALL, page_oncall)
        self.engine.register_action_handler(ActionType.A_THROTTLE_RISK, throttle_risk)
        self.engine.register_action_handler(ActionType.A_STOP_EXECUTION, stop_execution)
        self.engine.register_action_handler(ActionType.A_ENTER_SAFE_MODE, enter_safe_mode)
        self.engine.register_action_handler(ActionType.A_FREEZE_STRATEGY, freeze_strategy)
        self.engine.register_action_handler(ActionType.A_ROUTE_FAILOVER, route_failover)
        self.engine.register_action_handler(ActionType.A_LOCK_KEYS, lock_keys)
        self.engine.register_action_handler(ActionType.A_QUARANTINE_SOURCE, quarantine_source)
        self.engine.register_action_handler(ActionType.A_FORCE_RECON, force_recon)
    
    async def run_compliance_check(self, sample_metrics: Optional[Dict[str, Any]] = None) -> DoctrineComplianceReport:
        """Run a full compliance check across all departments."""
        
        # Use sample metrics for demonstration or fetch real ones
        metrics = sample_metrics or self._get_sample_metrics()
        
        # Update all metrics in engine
        for name, value in metrics.items():
            dept = self._metric_to_department(name)
            self.engine.update_metric(name, value, dept)
        
        # Check AZ Prime triggers
        new_state, actions = self.engine.check_az_prime_triggers(metrics)
        
        # Execute triggered actions
        for action in actions:
            await self.engine.execute_action(action, {"metrics": metrics})
        
        # Update state if needed
        if self.engine._state_severity(new_state) > self.engine._state_severity(self.engine.current_az_state):
            logger.warning(f"AZ Prime state transition: {self.engine.current_az_state.value} → {new_state.value}")
            self.engine.current_az_state = new_state
        
        return self.engine.generate_compliance_report()
    
    def _metric_to_department(self, metric_name: str) -> Department:
        """Map a metric name to its owning department."""
        for pack_id, pack_info in DOCTRINE_PACKS.items():
            if metric_name in pack_info.get('key_metrics', []):
                return pack_info['owner']
        return Department.SHARED_INFRASTRUCTURE
    
    def _get_sample_metrics(self) -> Dict[str, Any]:
        """Get sample metrics for demonstration."""
        return {
            # Pack 1: Risk Envelope
            "max_drawdown_pct": 3.5,
            "daily_loss_pct": 0.8,
            "capital_utilization": 45.0,
            "margin_buffer": 55.0,
            "portfolio_heat": 42.0,
            
            # Pack 2: Security
            "key_age_days": 25,
            "failed_auth_rate": 0.5,
            "audit_log_completeness": 99.95,
            "mfa_compliance_rate": 100.0,
            
            # Pack 3: Testing
            "backtest_vs_live_correlation": 0.85,
            "chaos_test_pass_rate": 97.0,
            "regression_test_pass_rate": 100.0,
            
            # Pack 4: Incident Response
            "mttd_minutes": 1.5,
            "mttr_minutes": 12.0,
            "incident_recurrence_rate": 3.0,
            
            # Pack 5: Liquidity
            "fill_rate": 96.0,
            "slippage_bps": 3.5,
            "partial_fill_rate": 8.0,
            
            # Pack 6: Counterparty
            "venue_health_score": 0.92,
            "withdrawal_success_rate": 99.5,
            "counterparty_exposure_pct": 15.0,
            
            # Pack 7: Research
            "research_pipeline_velocity": 4.0,
            "strategy_survival_rate": 55.0,
            "experiment_completion_rate": 72.0,
            
            # Pack 8: Metrics
            "data_quality_score": 0.97,
            "reconciliation_accuracy": 99.2,
            "metric_lineage_coverage": 85.0,
        }
    
    def print_doctrine_summary(self) -> None:
        """Print a summary of all doctrine packs."""
        print("\n" + "═" * 80)
        print("AAC DOCTRINE PACKS SUMMARY")
        print("═" * 80)
        
        total_metrics = 0
        total_failure_modes = 0
        total_triggers = 0
        
        for pack_id in sorted(DOCTRINE_PACKS.keys()):
            pack = DOCTRINE_PACKS[pack_id]
            metrics = len(pack.get('key_metrics', []))
            failures = len(pack.get('failure_modes', []))
            triggers = len(pack.get('az_prime_triggers', []))
            
            total_metrics += metrics
            total_failure_modes += failures
            total_triggers += triggers
            
            print(f"\n📦 Pack {pack_id}: {pack['name']}")
            print(f"   Owner: {pack['owner'].value}")
            print(f"   Metrics: {metrics} | Failure Modes: {failures} | AZ Triggers: {triggers}")
        
        print("\n" + "─" * 80)
        print(f"TOTALS: {total_metrics} metrics | {total_failure_modes} failure modes | {total_triggers} AZ triggers")
        print("═" * 80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Main entry point for doctrine engine demonstration."""
    
    print("\n" + "█" * 80)
    print("  AAC DOCTRINE APPLICATION ENGINE")
    print("  Analyzing and Applying 8 Doctrine Packs")
    print("█" * 80)
    
    # Initialize engine and service
    engine = DoctrineEngine()
    service = DoctrineApplicationService(engine)
    await service.initialize()
    
    # Print doctrine summary
    service.print_doctrine_summary()
    
    # Run compliance check with sample metrics
    print("\n🔍 Running Compliance Check...")
    print("─" * 80)
    
    report = await service.run_compliance_check()
    
    print(f"\n📊 COMPLIANCE REPORT")
    print(f"   Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Scope: {report.scope}")
    print(f"   AZ Prime State: {report.az_prime_state.value}")
    print(f"\n   ✅ Compliant: {report.compliant}")
    print(f"   ⚠️  Warnings: {report.warnings}")
    print(f"   ❌ Violations: {report.violations}")
    print(f"\n   📈 Compliance Score: {report.compliance_score}%")
    
    if report.violations_list:
        print(f"\n   Active Violations:")
        for v in report.violations_list:
            print(f"   - [{v.pack_name}] {v.description}")
            print(f"     Action: {v.recommended_action.value}")
    
    # Per-department summary
    print("\n\n📋 DEPARTMENT DOCTRINE ASSIGNMENTS")
    print("─" * 80)
    
    for dept in Department:
        summary = engine.get_department_doctrine_summary(dept)
        if summary['primary_packs']:
            print(f"\n🏢 {dept.value}")
            for pack in summary['primary_packs']:
                print(f"   • Pack {pack['pack_id']}: {pack['name']}")
                print(f"     - {len(pack['metrics'])} metrics to track")
                print(f"     - {len(pack['failure_modes'])} failure modes to monitor")
    
    print("\n" + "═" * 80)
    print("✅ Doctrine Application Complete")
    print("═" * 80 + "\n")
    
    return report


if __name__ == "__main__":
    asyncio.run(main())
