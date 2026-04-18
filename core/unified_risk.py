"""Unified Risk Aggregation — AAC v3.6.0

Central risk dashboard that aggregates risk metrics from ALL subsystems:
  - Greeks portfolio risk (delta, gamma, theta, vega)
  - Matrix Maximizer risk (VaR, circuit breaker, mandate)
  - Trading execution risk (limits, state, violations)
  - Paper trading risk (drawdown, halt state)

Provides cross-strategy risk limits, total portfolio exposure,
and risk budget allocation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data classes
# ---------------------------------------------------------------------------

class OverallRiskLevel(Enum):
    """Aggregate risk level across all subsystems."""

    GREEN = "green"       # all limits within thresholds
    YELLOW = "yellow"     # approaching limits
    ORANGE = "orange"     # some limits breached
    RED = "red"           # multiple breaches, consider halting
    HALT = "halt"         # circuit breaker tripped


@dataclass
class SubsystemRisk:
    """Risk summary from one subsystem."""

    name: str
    risk_level: str       # "low", "moderate", "high", "critical", etc.
    utilization: float    # 0..1 — how much of risk budget is used
    details: dict[str, Any] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)


@dataclass
class RiskBudget:
    """Risk budget for a strategy or subsystem."""

    name: str
    allocated_var: float      # max VaR allowed in dollars
    current_var: float        # current VaR estimate
    allocated_delta: float    # max absolute delta
    current_delta: float
    utilization: float = 0.0  # max of VaR and delta utilization

    def compute_utilization(self) -> None:
        var_util = self.current_var / self.allocated_var if self.allocated_var > 0 else 0.0
        delta_util = abs(self.current_delta) / self.allocated_delta if self.allocated_delta > 0 else 0.0
        self.utilization = max(var_util, delta_util)


@dataclass
class UnifiedRiskSnapshot:
    """Complete risk view across all subsystems."""

    timestamp: str
    overall_level: OverallRiskLevel = OverallRiskLevel.GREEN
    subsystems: list[SubsystemRisk] = field(default_factory=list)
    risk_budgets: list[RiskBudget] = field(default_factory=list)

    # Aggregated metrics
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0
    total_daily_theta: float = 0.0
    total_var_95: float = 0.0
    total_exposure_usd: float = 0.0
    total_daily_pnl: float = 0.0
    max_drawdown_pct: float = 0.0

    # Limits
    portfolio_var_limit: float = 5000.0    # max 95% 1-day VaR
    portfolio_delta_limit: float = 200.0
    portfolio_exposure_limit: float = 500_000.0
    daily_loss_limit: float = 3000.0

    alerts: list[str] = field(default_factory=list)
    circuit_breaker_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_level": self.overall_level.value,
            "total_delta": round(self.total_delta, 2),
            "total_gamma": round(self.total_gamma, 3),
            "total_vega": round(self.total_vega, 2),
            "total_daily_theta": round(self.total_daily_theta, 2),
            "total_var_95": round(self.total_var_95, 2),
            "total_exposure_usd": round(self.total_exposure_usd, 0),
            "daily_pnl": round(self.total_daily_pnl, 2),
            "circuit_breaker": self.circuit_breaker_active,
            "n_alerts": len(self.alerts),
            "subsystems": [
                {"name": s.name, "level": s.risk_level, "util": round(s.utilization, 2)}
                for s in self.subsystems
            ],
        }


# ---------------------------------------------------------------------------
# Unified Risk Aggregator
# ---------------------------------------------------------------------------

class UnifiedRiskAggregator:
    """Aggregate risk from all AAC subsystems.

    Parameters
    ----------
    var_limit : float
        Portfolio-wide 95% 1-day VaR limit.
    delta_limit : float
        Max absolute portfolio delta (SPY-equivalent).
    exposure_limit : float
        Max total exposure in USD.
    daily_loss_limit : float
        Max daily P&L loss before circuit breaker.
    """

    def __init__(
        self,
        var_limit: float = 5000.0,
        delta_limit: float = 200.0,
        exposure_limit: float = 500_000.0,
        daily_loss_limit: float = 3000.0,
    ) -> None:
        self.var_limit = var_limit
        self.delta_limit = delta_limit
        self.exposure_limit = exposure_limit
        self.daily_loss_limit = daily_loss_limit
        self._budgets: dict[str, RiskBudget] = {}

    # ── Budget management ─────────────────────────────────────────────────

    def set_risk_budget(
        self,
        name: str,
        allocated_var: float,
        allocated_delta: float,
    ) -> None:
        """Allocate risk budget to a strategy/subsystem."""
        self._budgets[name] = RiskBudget(
            name=name,
            allocated_var=allocated_var,
            current_var=0.0,
            allocated_delta=allocated_delta,
            current_delta=0.0,
        )

    def update_budget(
        self,
        name: str,
        current_var: float,
        current_delta: float,
    ) -> None:
        """Update current risk usage for a budget."""
        if name not in self._budgets:
            return
        b = self._budgets[name]
        b.current_var = current_var
        b.current_delta = current_delta
        b.compute_utilization()

    # ── Aggregation ───────────────────────────────────────────────────────

    def aggregate(
        self,
        greeks_snapshot: Optional[dict[str, Any]] = None,
        matrix_risk: Optional[dict[str, Any]] = None,
        execution_risk: Optional[dict[str, Any]] = None,
        paper_risk: Optional[dict[str, Any]] = None,
    ) -> UnifiedRiskSnapshot:
        """Build unified risk snapshot from subsystem data.

        All inputs are dicts from the respective .to_dict() methods
        or equivalent raw data.
        """
        now = datetime.now(timezone.utc).isoformat()
        snapshot = UnifiedRiskSnapshot(
            timestamp=now,
            portfolio_var_limit=self.var_limit,
            portfolio_delta_limit=self.delta_limit,
            portfolio_exposure_limit=self.exposure_limit,
            daily_loss_limit=self.daily_loss_limit,
        )

        # ── Greeks subsystem ──────────────────────────────────────────────
        if greeks_snapshot:
            sub = self._ingest_greeks(greeks_snapshot)
            snapshot.subsystems.append(sub)
            snapshot.total_delta += greeks_snapshot.get("delta", 0.0)
            snapshot.total_gamma += greeks_snapshot.get("gamma", 0.0)
            snapshot.total_vega += greeks_snapshot.get("vega", 0.0)
            snapshot.total_daily_theta += greeks_snapshot.get("daily_theta", 0.0)
            snapshot.alerts.extend(greeks_snapshot.get("alerts", []))

        # ── Matrix Maximizer risk ─────────────────────────────────────────
        if matrix_risk:
            sub = self._ingest_matrix(matrix_risk)
            snapshot.subsystems.append(sub)
            snapshot.total_var_95 += matrix_risk.get("var_95_1d", 0.0)
            snapshot.total_exposure_usd += matrix_risk.get("total_exposure", 0.0)
            snapshot.total_daily_pnl += matrix_risk.get("daily_pnl", 0.0)
            if matrix_risk.get("circuit_breaker", {}).get("tripped"):
                snapshot.circuit_breaker_active = True

        # ── Execution risk ────────────────────────────────────────────────
        if execution_risk:
            sub = self._ingest_execution(execution_risk)
            snapshot.subsystems.append(sub)
            snapshot.total_exposure_usd += execution_risk.get("total_exposure_usd", 0.0)
            snapshot.total_daily_pnl += execution_risk.get("daily_pnl", 0.0)

        # ── Paper trading risk ────────────────────────────────────────────
        if paper_risk:
            sub = self._ingest_paper(paper_risk)
            snapshot.subsystems.append(sub)
            snapshot.max_drawdown_pct = max(
                snapshot.max_drawdown_pct,
                paper_risk.get("max_drawdown_pct", 0.0),
            )

        # ── Risk budgets ─────────────────────────────────────────────────
        snapshot.risk_budgets = list(self._budgets.values())

        # ── Overall level ─────────────────────────────────────────────────
        snapshot.overall_level = self._classify(snapshot)

        return snapshot

    # ── Subsystem ingestors ───────────────────────────────────────────────

    def _ingest_greeks(self, data: dict[str, Any]) -> SubsystemRisk:
        level = data.get("risk_level", "low")
        delta_util = abs(data.get("delta", 0)) / self.delta_limit if self.delta_limit > 0 else 0
        return SubsystemRisk(
            name="greeks_portfolio",
            risk_level=level,
            utilization=min(delta_util, 1.0),
            details=data,
            alerts=data.get("alerts", []),
        )

    def _ingest_matrix(self, data: dict[str, Any]) -> SubsystemRisk:
        var_util = data.get("var_95_1d", 0.0) / self.var_limit if self.var_limit > 0 else 0
        level = "low"
        if var_util > 0.8:
            level = "high"
        elif var_util > 0.5:
            level = "moderate"
        return SubsystemRisk(
            name="matrix_maximizer",
            risk_level=level,
            utilization=min(var_util, 1.0),
            details=data,
            alerts=data.get("hedge_alerts", []),
        )

    def _ingest_execution(self, data: dict[str, Any]) -> SubsystemRisk:
        violations = data.get("violations", [])
        level = "critical" if violations else "low"
        exp_util = data.get("total_exposure_usd", 0) / self.exposure_limit if self.exposure_limit > 0 else 0
        return SubsystemRisk(
            name="execution_risk",
            risk_level=level,
            utilization=min(exp_util, 1.0),
            details=data,
            alerts=violations,
        )

    def _ingest_paper(self, data: dict[str, Any]) -> SubsystemRisk:
        is_halted = data.get("is_halted", False)
        level = "critical" if is_halted else "low"
        dd = data.get("max_drawdown_pct", 0.0)
        util = dd / 0.10 if dd > 0 else 0.0  # 10% max drawdown baseline
        return SubsystemRisk(
            name="paper_trading",
            risk_level=level,
            utilization=min(util, 1.0),
            details=data,
            alerts=[data.get("halt_reason", "")] if is_halted else [],
        )

    # ── Classification ────────────────────────────────────────────────────

    def _classify(self, snap: UnifiedRiskSnapshot) -> OverallRiskLevel:
        """Classify overall risk level."""
        if snap.circuit_breaker_active:
            return OverallRiskLevel.HALT

        breaches = 0
        if abs(snap.total_delta) > snap.portfolio_delta_limit:
            snap.alerts.append(f"Delta breach: {snap.total_delta:.1f} > {snap.portfolio_delta_limit}")
            breaches += 1
        if snap.total_var_95 > snap.portfolio_var_limit:
            snap.alerts.append(f"VaR breach: ${snap.total_var_95:.0f} > ${snap.portfolio_var_limit:.0f}")
            breaches += 1
        if snap.total_exposure_usd > snap.portfolio_exposure_limit:
            snap.alerts.append(f"Exposure breach: ${snap.total_exposure_usd:.0f}")
            breaches += 1
        if snap.total_daily_pnl < -snap.daily_loss_limit:
            snap.alerts.append(f"Daily loss breach: ${snap.total_daily_pnl:.0f}")
            breaches += 1

        if breaches >= 3:
            return OverallRiskLevel.RED
        if breaches >= 2:
            return OverallRiskLevel.ORANGE
        if breaches >= 1:
            return OverallRiskLevel.YELLOW

        # Check subsystem utilization
        max_util = max((s.utilization for s in snap.subsystems), default=0.0)
        if max_util > 0.8:
            return OverallRiskLevel.YELLOW

        return OverallRiskLevel.GREEN
