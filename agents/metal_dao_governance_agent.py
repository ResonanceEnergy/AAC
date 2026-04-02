#!/usr/bin/env python3
"""
Metal DAO Governance Agent (Vector 12)
========================================
Autonomous agent for monitoring and participating in Metal DAO governance.

The Metal DAO governs:
  - Protocol parameter changes (fees, limits, collateral ratios)
  - Treasury allocation and grant funding
  - Validator/block producer elections
  - Protocol upgrades and smart contract deployments
  - Partnership and integration approvals

This agent:
  1. Monitors governance proposals in real-time
  2. Analyzes proposal impact on AAC strategy performance
  3. Generates voting recommendations based on strategic doctrine
  4. Tracks governance outcomes for post-mortem analysis
  5. Alerts on proposals affecting AAC's trading operations
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from BigBrainIntelligence.agents import BaseResearchAgent, ResearchFinding
from shared.config_loader import get_config

logger = logging.getLogger(__name__)


@dataclass
class GovernanceProposal:
    """A governance proposal on Metal DAO."""
    proposal_id: str
    title: str
    description: str
    proposer: str
    category: str  # parameter_change, treasury, election, upgrade, partnership
    status: str  # pending, active, passed, rejected, executed
    vote_start: Optional[datetime] = None
    vote_end: Optional[datetime] = None
    votes_for: float = 0.0
    votes_against: float = 0.0
    quorum_pct: float = 0.0
    impact_assessment: Optional[str] = None
    aac_recommendation: Optional[str] = None  # support, oppose, abstain
    data: Dict[str, Any] = field(default_factory=dict)


class MetalDAOGovernanceAgent(BaseResearchAgent):
    """
    Monitors Metal DAO governance and generates strategic recommendations.

    Theater: METAL_BLOCKCHAIN

    This agent evaluates proposals through the lens of AAC's strategic
    doctrine (Art of War + 48 Laws integration) to determine whether
    proposals help or harm AAC's trading operations.

    Impact categories:
      - FEE_CHANGE: Proposals affecting trading fees → directly impacts strategy PnL
      - LIQUIDITY: Proposals affecting pool liquidity → impacts execution quality
      - COMPLIANCE: Regulatory/compliance changes → impacts operational risk
      - INTEGRATION: New partnerships/integrations → impacts expansion vectors
      - UPGRADE: Protocol upgrades → impacts technical stability
    """

    def __init__(self):
        super().__init__(
            agent_id="metal_dao_governance",
            theater="METAL_BLOCKCHAIN",
        )
        self._active_proposals: Dict[str, GovernanceProposal] = {}
        self._historical_proposals: List[GovernanceProposal] = []

    async def run_scan(self) -> List[ResearchFinding]:
        """
        Scan for new and updated governance proposals.

        Returns findings for each actionable proposal.
        """
        findings: List[ResearchFinding] = []
        self.is_running = True

        try:
            findings.extend(await self._scan_active_proposals())
            findings.extend(await self._scan_upcoming_votes())
            findings.extend(await self._scan_executed_proposals())

            self.last_scan = datetime.now()
            self.findings.extend(findings)

            if findings:
                self.logger.info(
                    f"DAO Governance scan — {len(findings)} findings, "
                    f"{len(self._active_proposals)} active proposals"
                )

        except Exception as e:
            self.logger.error(f"Governance scan error: {e}")
        finally:
            self.is_running = False

        return findings

    async def _scan_active_proposals(self) -> List[ResearchFinding]:
        """Check for new active proposals."""
        findings = []

        try:
            self.logger.debug("Scanning active DAO proposals...")

            # Production: query Metal DAO smart contract for active proposals
            # For each new proposal not in self._active_proposals:
            #   1. Parse the proposal details
            #   2. Run impact assessment
            #   3. Generate recommendation
            #   4. Create finding

        except Exception as e:
            self.logger.error(f"Active proposals scan error: {e}")

        return findings

    async def _scan_upcoming_votes(self) -> List[ResearchFinding]:
        """Alert on proposals nearing vote deadline."""
        findings = []

        for pid, proposal in self._active_proposals.items():
            if proposal.status != "active" or not proposal.vote_end:
                continue

            time_left = proposal.vote_end - datetime.now()

            # Alert if vote ends within 24 hours
            if timedelta(0) < time_left < timedelta(hours=24):
                findings.append(
                    self.create_finding(
                        finding_type="governance_vote_deadline",
                        title=f"Vote deadline approaching: {proposal.title}",
                        description=(
                            f"Proposal {proposal.proposal_id} vote ends in "
                            f"{time_left.total_seconds() / 3600:.1f} hours. "
                            f"Current: {proposal.votes_for:.0f} for / "
                            f"{proposal.votes_against:.0f} against. "
                            f"AAC recommendation: {proposal.aac_recommendation or 'pending'}"
                        ),
                        confidence=0.9,
                        urgency="high",
                        data={"proposal": proposal.__dict__},
                        sources=["metal_dao_contract"],
                    )
                )

        return findings

    async def _scan_executed_proposals(self) -> List[ResearchFinding]:
        """Track recently executed proposals for impact analysis."""
        findings = []

        try:
            self.logger.debug("Scanning executed DAO proposals...")

            # Production: find proposals that transitioned to 'executed'
            # Generate post-mortem findings for strategy adjustments

        except Exception as e:
            self.logger.error(f"Executed proposals scan error: {e}")

        return findings

    def assess_impact(self, proposal: GovernanceProposal) -> str:
        """
        Assess the impact of a proposal on AAC trading operations.

        Uses strategic doctrine principles to evaluate.
        """
        category = proposal.category.lower()

        if category == "parameter_change":
            # Fee changes directly impact strategy PnL
            return "HIGH — May affect trading fee structure and strategy profitability"
        elif category == "treasury":
            return "MEDIUM — Treasury changes may affect liquidity"
        elif category == "election":
            return "LOW — Validator changes have minimal direct trading impact"
        elif category == "upgrade":
            return "HIGH — Protocol upgrades may require connector updates"
        elif category == "partnership":
            return "MEDIUM — New integrations may open new trading vectors"
        else:
            return "LOW — Limited impact on current operations"

    def generate_recommendation(self, proposal: GovernanceProposal) -> str:
        """
        Generate a voting recommendation based on strategic analysis.

        Principles applied:
          - Art of War: "The supreme art of war is to subdue the enemy without fighting"
            → Support proposals that expand market access without cost
          - 48 Laws: Law 2 "Never put too much trust in friends"
            → Evaluate proposer's track record and incentives
        """
        impact = self.assess_impact(proposal)

        if "HIGH" in impact:
            # Need careful analysis — abstain by default until reviewed
            return "abstain"
        elif proposal.category in ("treasury", "partnership"):
            # Generally supportive of expansion
            return "support"
        else:
            return "abstain"

    def get_status_report(self) -> Dict[str, Any]:
        """Get governance agent status for dashboard."""
        return {
            "agent_id": self.agent_id,
            "theater": self.theater,
            "is_running": self.is_running,
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "active_proposals": len(self._active_proposals),
            "historical_proposals": len(self._historical_proposals),
            "total_findings": len(self.findings),
        }
