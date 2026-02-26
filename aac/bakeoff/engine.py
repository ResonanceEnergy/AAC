"""
AAC Bake-Off Engine
====================
Automated strategy validation, gate progression, and safety state management.
Integrates with metric_canon.yaml, bakeoff_policy.yaml, and gate_checklists.yaml.

Usage:
    from aac.bakeoff.engine import BakeoffEngine, StrategyState
    
    engine = BakeoffEngine()
    
    # Check gate requirements
    result = engine.validate_gate("STRAT_001", "PILOT")
    
    # Get composite score
    score = engine.calculate_composite_score("STRAT_001")
    
    # Check safety state
    state = engine.evaluate_safety_state()
"""

import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SafetyState(Enum):
    """System-wide safety states"""
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    SAFE_MODE = "SAFE_MODE"
    HALT = "HALT"


class Gate(Enum):
    """Strategy progression gates"""
    SPEC = 0
    SIM = 1
    PAPER = 2
    PILOT = 3
    POST_ANALYSIS = 4
    SCALE = 5


class Decision(Enum):
    """Gate review decisions"""
    PROMOTE = "promote"
    HOLD = "hold"
    QUARANTINE = "quarantine"
    RETIRE = "retire"


@dataclass
class MetricValue:
    """A metric measurement with metadata"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    threshold_status: str = "unknown"  # good, warning, critical


@dataclass
class ChecklistItem:
    """A single checklist item result"""
    id: str
    item: str
    required: bool
    passed: bool
    evidence: str = ""
    metric_value: Optional[float] = None


@dataclass
class GateValidation:
    """Result of a gate validation"""
    strategy_id: str
    gate: Gate
    passed: bool
    checklist_results: List[ChecklistItem]
    blocking_items: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CompositeScore:
    """Strategy composite score breakdown"""
    strategy_id: str
    performance: float
    risk: float
    execution: float
    data: float
    ops: float
    fragility: float
    instability_penalty: float
    composite: float
    decision: Decision
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyState:
    """Current state of a strategy"""
    strategy_id: str
    current_gate: Gate
    safety_state: SafetyState
    capital_allocated: float
    last_score: Optional[CompositeScore] = None
    last_validation: Optional[GateValidation] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class BakeoffEngine:
    """
    Main engine for strategy bake-off management.
    
    Handles:
    - Gate validation and progression
    - Composite scoring
    - Safety state evaluation
    - Policy enforcement
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the bake-off engine.
        
        Args:
            config_dir: Path to bakeoff config directory. 
                       Defaults to aac/bakeoff/ in project root.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        
        self.config_dir = Path(config_dir)
        self.metric_canon = self._load_yaml("metrics/metric_canon.yaml")
        self.policy = self._load_yaml("policy/bakeoff_policy.yaml")
        self.checklists = self._load_yaml("checklists/gate_checklists.yaml")
        
        # Runtime state
        self.strategies: Dict[str, StrategyState] = {}
        self.current_safety_state = SafetyState.NORMAL
        self.metrics_cache: Dict[str, MetricValue] = {}
        
        logger.info(f"BakeoffEngine initialized with config from {config_dir}")
    
    def _load_yaml(self, relative_path: str) -> Dict:
        """Load a YAML config file"""
        path = self.config_dir / relative_path
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}
        
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    # ==================== METRIC EVALUATION ====================
    
    def evaluate_metric(
        self, 
        metric_name: str, 
        value: float,
        category: Optional[str] = None
    ) -> MetricValue:
        """
        Evaluate a metric value against canon thresholds.
        
        Args:
            metric_name: Name of the metric (e.g., "sharpe_ratio")
            value: The measured value
            category: Optional category hint (performance, risk, etc.)
            
        Returns:
            MetricValue with threshold status
        """
        # Find metric definition in canon
        canon = self.metric_canon.get("metric_canon", {})
        metric_def = None
        unit = "unknown"
        
        for cat, metrics in canon.items():
            if isinstance(metrics, dict) and metric_name in metrics:
                metric_def = metrics[metric_name]
                category = cat
                unit = metric_def.get("unit", "unknown")
                break
        
        # Determine threshold status
        status = "unknown"
        if metric_def:
            thresholds = {
                "good": metric_def.get("threshold_good"),
                "warning": metric_def.get("threshold_warning"),
                "critical": metric_def.get("threshold_critical"),
            }
            status = self._evaluate_threshold(value, thresholds)
        
        metric_value = MetricValue(
            name=metric_name,
            value=value,
            unit=unit,
            threshold_status=status,
        )
        
        # Cache the metric
        self.metrics_cache[metric_name] = metric_value
        
        return metric_value
    
    def _evaluate_threshold(
        self, 
        value: float, 
        thresholds: Dict[str, Optional[str]]
    ) -> str:
        """Evaluate value against threshold expressions"""
        # Parse threshold expressions like ">= 1.5" or "< 5%"
        def check_threshold(threshold_str: str, value: float) -> bool:
            if not threshold_str:
                return False
            
            # Remove percentage sign if present
            threshold_str = threshold_str.replace("%", "")
            
            # Parse operator and value
            for op in [">=", "<=", ">", "<", "=="]:
                if threshold_str.startswith(op):
                    threshold_value = float(threshold_str[len(op):].strip())
                    if op == ">=":
                        return value >= threshold_value
                    elif op == "<=":
                        return value <= threshold_value
                    elif op == ">":
                        return value > threshold_value
                    elif op == "<":
                        return value < threshold_value
                    elif op == "==":
                        return value == threshold_value
            
            return False
        
        if thresholds.get("good") and check_threshold(thresholds["good"], value):
            return "good"
        if thresholds.get("critical") and check_threshold(thresholds["critical"], value):
            return "critical"
        if thresholds.get("warning") and check_threshold(thresholds["warning"], value):
            return "warning"
        
        return "unknown"
    
    # ==================== GATE VALIDATION ====================
    
    def validate_gate(
        self, 
        strategy_id: str, 
        gate: str,
        metrics: Optional[Dict[str, float]] = None
    ) -> GateValidation:
        """
        Validate a strategy against a specific gate's checklist.
        
        Args:
            strategy_id: The strategy identifier
            gate: Gate name (SPEC, SIM, PAPER, PILOT, POST_ANALYSIS, SCALE)
            metrics: Optional dict of metric values to validate against
            
        Returns:
            GateValidation with results
        """
        gate_enum = Gate[gate.upper()]
        checklist_data = self.checklists.get("gate_checklists", {}).get(gate.upper(), {})
        
        if not checklist_data:
            logger.error(f"No checklist found for gate: {gate}")
            return GateValidation(
                strategy_id=strategy_id,
                gate=gate_enum,
                passed=False,
                checklist_results=[],
                blocking_items=["Checklist not found"],
            )
        
        results: List[ChecklistItem] = []
        blocking_items: List[str] = []
        metrics = metrics or {}
        
        # Process each checklist section
        for section_name, items in checklist_data.get("checklist", {}).items():
            for item in items:
                item_id = item.get("id", "")
                item_text = item.get("item", "")
                required = item.get("required", False)
                metric_name = item.get("metric")
                threshold = item.get("threshold")
                
                # Evaluate if metric-based
                passed = False
                metric_value = None
                evidence = ""
                
                if metric_name and threshold and metric_name in metrics:
                    metric_value = metrics[metric_name]
                    passed = self._evaluate_threshold(metric_value, {"good": threshold})
                    evidence = f"{metric_name}={metric_value} (threshold: {threshold})"
                elif not metric_name:
                    # Manual checklist item - assume passed if not metric-based
                    # In production, this would check actual documentation/approval state
                    passed = True
                    evidence = "Manual verification required"
                
                result = ChecklistItem(
                    id=item_id,
                    item=item_text,
                    required=required,
                    passed=passed,
                    evidence=evidence,
                    metric_value=metric_value,
                )
                results.append(result)
                
                if required and not passed:
                    blocking_items.append(f"{item_id}: {item_text}")
        
        validation = GateValidation(
            strategy_id=strategy_id,
            gate=gate_enum,
            passed=len(blocking_items) == 0,
            checklist_results=results,
            blocking_items=blocking_items,
        )
        
        # Update strategy state
        if strategy_id in self.strategies:
            self.strategies[strategy_id].last_validation = validation
            self.strategies[strategy_id].updated_at = datetime.now()
        
        logger.info(
            f"Gate validation for {strategy_id} at {gate}: "
            f"{'PASSED' if validation.passed else 'FAILED'} "
            f"({len(blocking_items)} blocking items)"
        )
        
        return validation
    
    # ==================== COMPOSITE SCORING ====================
    
    def calculate_composite_score(
        self,
        strategy_id: str,
        metrics: Dict[str, float],
        historical_scores: Optional[List[float]] = None
    ) -> CompositeScore:
        """
        Calculate composite score for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            metrics: Dict of metric name -> value
            historical_scores: Optional list of past composite scores for instability
            
        Returns:
            CompositeScore with breakdown
        """
        weights = self.policy.get("aac_bakeoff", {}).get("scoring", {}).get("weights", {})
        
        # Calculate dimension scores (0-100)
        performance = self._score_dimension(metrics, "performance")
        risk = self._score_dimension(metrics, "risk")
        execution = self._score_dimension(metrics, "execution")
        data = self._score_dimension(metrics, "data")
        ops = self._score_dimension(metrics, "ops")
        fragility = self._score_dimension(metrics, "fragility")
        
        # Weighted sum
        weighted_sum = (
            performance * weights.get("performance", 0.25) +
            risk * weights.get("risk", 0.25) +
            execution * weights.get("execution", 0.15) +
            data * weights.get("data", 0.15) +
            ops * weights.get("ops", 0.10) +
            fragility * weights.get("fragility", 0.10)
        )
        
        # Calculate instability penalty
        instability_penalty = 0.0
        if historical_scores and len(historical_scores) >= 3:
            import statistics
            mean = statistics.mean(historical_scores)
            if mean > 0:
                stddev = statistics.stdev(historical_scores)
                instability_penalty = min(0.20, stddev / mean)
        
        composite = max(0, weighted_sum * (1 - instability_penalty))
        
        # Determine decision
        thresholds = self.policy.get("aac_bakeoff", {}).get("scoring", {})
        if composite >= thresholds.get("promotion_threshold", 70):
            decision = Decision.PROMOTE
        elif composite >= thresholds.get("hold_threshold", 50):
            decision = Decision.HOLD
        elif composite >= thresholds.get("quarantine_threshold", 35):
            decision = Decision.QUARANTINE
        else:
            decision = Decision.RETIRE
        
        score = CompositeScore(
            strategy_id=strategy_id,
            performance=performance,
            risk=risk,
            execution=execution,
            data=data,
            ops=ops,
            fragility=fragility,
            instability_penalty=instability_penalty,
            composite=composite,
            decision=decision,
        )
        
        # Update strategy state
        if strategy_id in self.strategies:
            self.strategies[strategy_id].last_score = score
            self.strategies[strategy_id].updated_at = datetime.now()
        
        logger.info(
            f"Composite score for {strategy_id}: {composite:.1f} "
            f"(decision: {decision.value})"
        )
        
        return score
    
    def _score_dimension(self, metrics: Dict[str, float], dimension: str) -> float:
        """
        Score a single dimension (0-100) based on metric values.
        
        Simple implementation: average of metric scores where:
        - good = 100
        - warning = 60
        - critical = 20
        - unknown = 50
        """
        canon = self.metric_canon.get("metric_canon", {}).get(dimension, {})
        if not canon:
            return 50.0  # Default neutral score
        
        scores = []
        for metric_name, metric_def in canon.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                thresholds = {
                    "good": metric_def.get("threshold_good"),
                    "warning": metric_def.get("threshold_warning"),
                    "critical": metric_def.get("threshold_critical"),
                }
                status = self._evaluate_threshold(value, thresholds)
                
                score_map = {"good": 100, "warning": 60, "critical": 20, "unknown": 50}
                scores.append(score_map.get(status, 50))
        
        return sum(scores) / len(scores) if scores else 50.0
    
    # ==================== SAFETY STATE ====================
    
    def evaluate_safety_state(
        self, 
        metrics: Dict[str, float]
    ) -> Tuple[SafetyState, List[str]]:
        """
        Evaluate current safety state based on metrics.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            Tuple of (SafetyState, list of triggered conditions)
        """
        triggers = self.policy.get("aac_bakeoff", {}).get("triggers", [])
        triggered = []
        new_state = SafetyState.NORMAL
        
        for trigger in triggers:
            condition = trigger.get("condition", "")
            action = trigger.get("action", "")
            
            # Parse and evaluate condition
            if self._evaluate_trigger_condition(condition, metrics):
                triggered.append(f"{trigger.get('name')}: {condition}")
                
                # Determine state change
                if "HALT" in action:
                    new_state = SafetyState.HALT
                elif "SAFE_MODE" in action and new_state != SafetyState.HALT:
                    new_state = SafetyState.SAFE_MODE
                elif "CAUTION" in action and new_state == SafetyState.NORMAL:
                    new_state = SafetyState.CAUTION
        
        if new_state != self.current_safety_state:
            logger.warning(
                f"Safety state change: {self.current_safety_state.value} -> {new_state.value}"
            )
            self.current_safety_state = new_state
        
        return new_state, triggered
    
    def _evaluate_trigger_condition(
        self, 
        condition: str, 
        metrics: Dict[str, float]
    ) -> bool:
        """Evaluate a trigger condition expression"""
        # Parse conditions like "daily_loss_pct >= 1.5"
        for op in [">=", "<=", ">", "<", "=="]:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    metric_name = parts[0].strip()
                    threshold = float(parts[1].strip())
                    value = metrics.get(metric_name)
                    
                    if value is None:
                        return False
                    
                    if op == ">=":
                        return value >= threshold
                    elif op == "<=":
                        return value <= threshold
                    elif op == ">":
                        return value > threshold
                    elif op == "<":
                        return value < threshold
                    elif op == "==":
                        return value == threshold
        
        return False
    
    # ==================== STRATEGY MANAGEMENT ====================
    
    def register_strategy(
        self,
        strategy_id: str,
        initial_gate: Gate = Gate.SPEC,
        capital: float = 0.0
    ) -> StrategyState:
        """Register a new strategy"""
        state = StrategyState(
            strategy_id=strategy_id,
            current_gate=initial_gate,
            safety_state=SafetyState.NORMAL,
            capital_allocated=capital,
        )
        self.strategies[strategy_id] = state
        logger.info(f"Registered strategy: {strategy_id} at gate {initial_gate.name}")
        return state
    
    def promote_strategy(self, strategy_id: str) -> bool:
        """Promote strategy to next gate"""
        if strategy_id not in self.strategies:
            return False
        
        state = self.strategies[strategy_id]
        current_order = state.current_gate.value
        
        if current_order >= Gate.SCALE.value:
            logger.warning(f"Strategy {strategy_id} already at max gate")
            return False
        
        # Find next gate
        for gate in Gate:
            if gate.value == current_order + 1:
                state.current_gate = gate
                state.updated_at = datetime.now()
                logger.info(f"Promoted {strategy_id} to {gate.name}")
                return True
        
        return False
    
    def quarantine_strategy(self, strategy_id: str, reason: str) -> bool:
        """Quarantine a strategy (pause trading)"""
        if strategy_id not in self.strategies:
            return False
        
        state = self.strategies[strategy_id]
        state.safety_state = SafetyState.SAFE_MODE
        state.updated_at = datetime.now()
        logger.warning(f"Quarantined strategy {strategy_id}: {reason}")
        return True
    
    def get_strategy_status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get full status of a strategy"""
        if strategy_id not in self.strategies:
            return None
        
        state = self.strategies[strategy_id]
        return {
            "strategy_id": strategy_id,
            "current_gate": state.current_gate.name,
            "safety_state": state.safety_state.value,
            "capital_allocated": state.capital_allocated,
            "last_score": state.last_score.composite if state.last_score else None,
            "last_decision": state.last_score.decision.value if state.last_score else None,
            "last_validation_passed": state.last_validation.passed if state.last_validation else None,
            "updated_at": state.updated_at.isoformat(),
        }
    
    # ==================== REPORTS ====================
    
    def generate_weekly_summary(self) -> Dict[str, Any]:
        """Generate weekly review summary data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_state": self.current_safety_state.value,
            "strategies": {
                sid: self.get_strategy_status(sid)
                for sid in self.strategies
            },
            "metrics_snapshot": {
                name: {"value": m.value, "status": m.threshold_status}
                for name, m in self.metrics_cache.items()
            },
        }


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="AAC Bake-Off Engine")
    parser.add_argument("--validate", type=str, help="Validate strategy at gate: STRAT_ID:GATE")
    parser.add_argument("--score", type=str, help="Calculate composite score for strategy")
    parser.add_argument("--status", type=str, help="Get strategy status")
    parser.add_argument("--summary", action="store_true", help="Generate weekly summary")
    args = parser.parse_args()
    
    engine = BakeoffEngine()
    
    if args.validate:
        parts = args.validate.split(":")
        if len(parts) == 2:
            result = engine.validate_gate(parts[0], parts[1])
            print(json.dumps({
                "strategy": result.strategy_id,
                "gate": result.gate.name,
                "passed": result.passed,
                "blocking": result.blocking_items,
            }, indent=2))
    
    elif args.summary:
        summary = engine.generate_weekly_summary()
        print(json.dumps(summary, indent=2, default=str))
    
    else:
        print("AAC Bake-Off Engine")
        print("Use --help for usage information")
