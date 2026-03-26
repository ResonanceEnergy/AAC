"""
NCL Governance Engine — Runtime NCL Compliance Matrix enforcement.

Extracted from full_activation.py into a standalone module so it can be
used by the orchestrator, autonomous engine, and matrix monitor.

Encodes the NCL Compliance Matrix and Governance Charter rules:
- Capital efficiency (25%+ annualized target)
- Risk-adjusted returns (Sharpe 2.0+, MDD <15%)
- Decision quality (doctrine compliance)
- Digital integrity (all actions auditable)
- Human judgment paramount (no live trades without confirmation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class NCLComplianceLevel(Enum):
    """NCL Compliance Matrix classification levels."""
    OMEGA = "OMEGA"       # 95%+ -- perfect compliance
    GAMMA = "GAMMA"       # 80-95% -- elite performance
    BETA = "BETA"         # 50-80% -- moderate compliance
    ALPHA = "ALPHA"       # <50% -- zero-compliance recovery


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

    8-criteria evaluation:
    1. Digital Integrity -- all trades auditable (DB online)
    2. Human Judgment Paramount -- paper mode unless confirmed live
    3. Risk Limits -- MDD < 15%
    4. Doctrine Compliance -- doctrine engine active
    5. Position Concentration -- max 25% in single asset
    6. Daily Loss Cap -- max 5% daily
    7. Agent Coordination -- at least 1 agent system online
    8. Crisis Awareness -- crisis monitor active
    """

    def evaluate(self, metrics: Dict[str, Any]) -> NCLComplianceReport:
        """Evaluate NCL governance compliance for this cycle."""
        report = NCLComplianceReport()
        checks: List[tuple] = []

        # 1. Digital Integrity -- all trades auditable
        db_ok = metrics.get("database_online", False)
        checks.append(("digital_integrity", db_ok))
        if not db_ok:
            report.violations.append("Database offline -- audit trail broken")

        # 2. Human Judgment Paramount -- paper mode unless explicitly confirmed
        paper_mode = metrics.get("paper_mode", True)
        checks.append(("human_judgment", paper_mode or metrics.get("live_confirmed", False)))
        if not paper_mode and not metrics.get("live_confirmed", False):
            report.violations.append("Live trading without explicit confirmation")

        # 3. Risk Limits -- MDD < 15%
        drawdown = metrics.get("drawdown_pct", 0.0)
        checks.append(("mdd_limit", drawdown < 15.0))
        if drawdown >= 15.0:
            report.violations.append(f"Max drawdown {drawdown:.1f}% exceeds 15% NCL limit")

        # 4. Doctrine Compliance -- doctrine engine loaded and active
        doctrine_ok = metrics.get("doctrine_online", False)
        checks.append(("doctrine_active", doctrine_ok))
        if not doctrine_ok:
            report.recommendations.append("Doctrine engine offline -- compliance degraded")

        # 5. Position Concentration -- max 25% in single asset
        max_concentration = metrics.get("max_concentration_pct", 0.0)
        checks.append(("concentration_limit", max_concentration <= 25.0))
        if max_concentration > 25.0:
            report.violations.append(
                f"Position concentration {max_concentration:.0f}% exceeds 25%"
            )

        # 6. Daily Loss Cap -- max 5% daily
        daily_loss = metrics.get("daily_loss_pct", 0.0)
        checks.append(("daily_loss_cap", daily_loss < 5.0))
        if daily_loss >= 5.0:
            report.violations.append(f"Daily loss {daily_loss:.1f}% exceeds 5% NCL limit")

        # 7. Agent Coordination -- at least one agent system online
        agents_online = metrics.get("agents_online", 0)
        checks.append(("agent_coordination", agents_online > 0))

        # 8. Crisis Awareness -- crisis monitor active
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
            report.recommendations.append(
                "CRITICAL: Immediate NCL recovery protocol required"
            )

        return report
