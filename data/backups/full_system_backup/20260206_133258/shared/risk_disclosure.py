#!/usr/bin/env python3
"""
Risk Disclosure Framework
========================
Standardized risk warnings, client risk profiling, and regulatory disclosure templates.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path
from shared.audit_logger import get_audit_logger


class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


class RiskCategory(Enum):
    """Categories of risk"""
    MARKET_RISK = "market_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CREDIT_RISK = "credit_risk"
    OPERATIONAL_RISK = "operational_risk"
    REGULATORY_RISK = "regulatory_risk"
    STRATEGY_RISK = "strategy_risk"


@dataclass
class RiskDisclosure:
    """Standardized risk disclosure"""
    disclosure_id: str
    title: str
    category: RiskCategory
    severity: str  # "low", "medium", "high", "critical"
    description: str
    regulatory_requirements: List[str]
    warning_text: str
    mitigation_measures: List[str]
    last_updated: datetime
    version: str


@dataclass
class ClientRiskProfile:
    """Client risk profile and preferences"""
    client_id: str
    risk_tolerance: RiskLevel
    investment_amount: float
    time_horizon: str
    experience_level: str
    risk_acknowledgments: List[str] = field(default_factory=list)
    disclosures_accepted: List[str] = field(default_factory=list)
    profile_created: datetime = field(default_factory=datetime.now)
    last_reviewed: Optional[datetime] = None


class RiskDisclosureFramework:
    """Comprehensive risk disclosure and client profiling system"""

    def __init__(self):
        self.logger = logging.getLogger("RiskDisclosure")
        self.audit_logger = get_audit_logger()

        # Risk disclosures library
        self.risk_disclosures: Dict[str, RiskDisclosure] = {}

        # Client risk profiles
        self.client_profiles: Dict[str, ClientRiskProfile] = {}

        # Regulatory templates
        self.regulatory_templates: Dict[str, Dict[str, Any]] = {}

        # Storage paths
        self.disclosures_dir = PROJECT_ROOT / "docs" / "risk_disclosures"
        self.profiles_dir = PROJECT_ROOT / "data" / "client_profiles"
        self.templates_dir = PROJECT_ROOT / "docs" / "regulatory_templates"
        self.disclosures_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Initialize standard disclosures
        self._initialize_standard_disclosures()

        # Load existing data
        self._load_client_profiles()

    def _initialize_standard_disclosures(self):
        """Initialize standard regulatory risk disclosures"""
        standard_disclosures = [
            RiskDisclosure(
                disclosure_id="market_volatility",
                title="Market Volatility and Price Risk",
                category=RiskCategory.MARKET_RISK,
                severity="high",
                description="Cryptocurrency and arbitrage trading involves significant price volatility",
                regulatory_requirements=["FINRA Rule 2210", "SEC Risk Disclosure"],
                warning_text="""
                WARNING: Cryptocurrency markets are highly volatile. Prices can change rapidly and significantly,
                potentially resulting in substantial losses. Past performance does not guarantee future results.
                Arbitrage strategies, while designed to be market-neutral, carry execution and timing risks.
                """,
                mitigation_measures=[
                    "Position size limits",
                    "Stop-loss orders",
                    "Diversification across strategies",
                    "Real-time risk monitoring"
                ],
                last_updated=datetime.now(),
                version="1.0"
            ),
            RiskDisclosure(
                disclosure_id="liquidity_risk",
                title="Liquidity and Market Access Risk",
                category=RiskCategory.LIQUIDITY_RISK,
                severity="medium",
                description="Limited liquidity in some markets may affect trade execution",
                regulatory_requirements=["FINRA Rule 2111", "SEC Market Access Rule"],
                warning_text="""
                WARNING: Some cryptocurrency markets have limited liquidity. This may result in:
                - Difficulty executing trades at desired prices
                - Wider bid-ask spreads
                - Potential slippage on large orders
                - Inability to exit positions quickly during market stress
                """,
                mitigation_measures=[
                    "Pre-trade liquidity checks",
                    "Position size limits based on market depth",
                    "Multiple exchange connectivity",
                    "Emergency liquidation protocols"
                ],
                last_updated=datetime.now(),
                version="1.0"
            ),
            RiskDisclosure(
                disclosure_id="operational_risk",
                title="Operational and Technology Risk",
                category=RiskCategory.OPERATIONAL_RISK,
                severity="high",
                description="System failures, connectivity issues, and algorithmic errors",
                regulatory_requirements=["FINRA Rule 4370", "SEC Regulation SCI"],
                warning_text="""
                WARNING: Trading operations depend on technology systems that may experience:
                - Internet connectivity failures
                - Software bugs or algorithmic errors
                - Exchange API outages
                - Power or hardware failures
                - Cyber security incidents
                These issues could result in unexecuted trades, delayed responses, or unintended positions.
                """,
                mitigation_measures=[
                    "Redundant internet connections",
                    "Multi-region deployment",
                    "Automated failover systems",
                    "Regular system testing and updates",
                    "24/7 monitoring and incident response"
                ],
                last_updated=datetime.now(),
                version="1.0"
            ),
            RiskDisclosure(
                disclosure_id="regulatory_risk",
                title="Regulatory and Compliance Risk",
                category=RiskCategory.REGULATORY_RISK,
                severity="critical",
                description="Changes in laws and regulations affecting cryptocurrency trading",
                regulatory_requirements=["Dodd-Frank Act", "CFTC Regulations"],
                warning_text="""
                WARNING: Cryptocurrency regulation is evolving rapidly. Future regulatory changes may:
                - Restrict or prohibit certain trading activities
                - Impose new capital or reporting requirements
                - Affect the legality or tax treatment of profits
                - Require changes to trading strategies or operations
                """,
                mitigation_measures=[
                    "Regular regulatory monitoring",
                    "Legal and compliance review",
                    "Conservative capital allocation",
                    "Diversification across jurisdictions",
                    "Professional legal counsel engagement"
                ],
                last_updated=datetime.now(),
                version="1.0"
            ),
            RiskDisclosure(
                disclosure_id="strategy_risk",
                title="Arbitrage Strategy Specific Risks",
                category=RiskCategory.STRATEGY_RISK,
                severity="high",
                description="Unique risks associated with arbitrage trading strategies",
                regulatory_requirements=["FINRA Rule 2210", "SEC Risk Disclosure"],
                warning_text="""
                WARNING: Arbitrage strategies, while designed to be low-risk, carry specific risks:
                - Execution timing differences between markets
                - Temporary dislocations in pricing relationships
                - Transaction costs eroding small profit margins
                - Counterparty or exchange-specific risks
                - Model assumptions may not hold during extreme market conditions
                """,
                mitigation_measures=[
                    "Real-time spread monitoring",
                    "Dynamic position sizing",
                    "Cost-benefit analysis for each trade",
                    "Stress testing under various market conditions",
                    "Continuous strategy performance monitoring"
                ],
                last_updated=datetime.now(),
                version="1.0"
            )
        ]

        for disclosure in standard_disclosures:
            self.risk_disclosures[disclosure.disclosure_id] = disclosure

        self.logger.info(f"Initialized {len(standard_disclosures)} standard risk disclosures")

    def _load_client_profiles(self):
        """Load existing client risk profiles"""
        profiles_file = self.profiles_dir / "client_profiles.json"

        if profiles_file.exists():
            try:
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)

                for profile_data in profiles_data:
                    profile = ClientRiskProfile(
                        client_id=profile_data["client_id"],
                        risk_tolerance=RiskLevel(profile_data["risk_tolerance"]),
                        investment_amount=profile_data["investment_amount"],
                        time_horizon=profile_data["time_horizon"],
                        experience_level=profile_data["experience_level"],
                        risk_acknowledgments=profile_data.get("risk_acknowledgments", []),
                        disclosures_accepted=profile_data.get("disclosures_accepted", []),
                        profile_created=datetime.fromisoformat(profile_data["profile_created"]),
                        last_reviewed=datetime.fromisoformat(profile_data["last_reviewed"]) if profile_data.get("last_reviewed") else None
                    )
                    self.client_profiles[profile.client_id] = profile

                self.logger.info(f"Loaded {len(self.client_profiles)} client risk profiles")

            except Exception as e:
                self.logger.error(f"Error loading client profiles: {e}")

    async def create_client_risk_profile(self,
                                       client_id: str,
                                       risk_tolerance: RiskLevel,
                                       investment_amount: float,
                                       time_horizon: str,
                                       experience_level: str) -> ClientRiskProfile:
        """Create a new client risk profile"""

        profile = ClientRiskProfile(
            client_id=client_id,
            risk_tolerance=risk_tolerance,
            investment_amount=investment_amount,
            time_horizon=time_horizon,
            experience_level=experience_level
        )

        self.client_profiles[client_id] = profile

        # Save profiles
        await self._save_client_profiles()

        # Audit the profile creation
        await self.audit_logger.log_event(
            category="compliance",
            action="risk_profile_created",
            details={
                "client_id": client_id,
                "risk_tolerance": risk_tolerance.value,
                "investment_amount": investment_amount,
                "time_horizon": time_horizon,
                "experience_level": experience_level
            }
        )

        self.logger.info(f"Created risk profile for client: {client_id}")
        return profile

    async def accept_risk_disclosure(self, client_id: str, disclosure_id: str) -> bool:
        """Record client acceptance of a risk disclosure"""
        if client_id not in self.client_profiles:
            self.logger.error(f"Client profile not found: {client_id}")
            return False

        if disclosure_id not in self.risk_disclosures:
            self.logger.error(f"Risk disclosure not found: {disclosure_id}")
            return False

        profile = self.client_profiles[client_id]

        if disclosure_id not in profile.disclosures_accepted:
            profile.disclosures_accepted.append(disclosure_id)
            profile.last_reviewed = datetime.now()

            # Save profiles
            await self._save_client_profiles()

            # Audit the acceptance
            await self.audit_logger.log_event(
                category="compliance",
                action="disclosure_accepted",
                details={
                    "client_id": client_id,
                    "disclosure_id": disclosure_id,
                    "timestamp": datetime.now().isoformat()
                }
            )

        return True

    async def _save_client_profiles(self):
        """Save client risk profiles to storage"""
        profiles_file = self.profiles_dir / "client_profiles.json"

        profiles_data = []
        for profile in self.client_profiles.values():
            profile_data = {
                "client_id": profile.client_id,
                "risk_tolerance": profile.risk_tolerance.value,
                "investment_amount": profile.investment_amount,
                "time_horizon": profile.time_horizon,
                "experience_level": profile.experience_level,
                "risk_acknowledgments": profile.risk_acknowledgments,
                "disclosures_accepted": profile.disclosures_accepted,
                "profile_created": profile.profile_created.isoformat(),
                "last_reviewed": profile.last_reviewed.isoformat() if profile.last_reviewed else None
            }
            profiles_data.append(profile_data)

        try:
            with open(profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving client profiles: {e}")

    def get_required_disclosures(self, client_id: str) -> List[RiskDisclosure]:
        """Get disclosures required for a client based on their risk profile"""
        if client_id not in self.client_profiles:
            # Return all disclosures for unknown clients
            return list(self.risk_disclosures.values())

        profile = self.client_profiles[client_id]

        # Determine required disclosures based on risk tolerance and amount
        required_disclosures = []

        # All clients get basic disclosures
        required_disclosures.extend([
            self.risk_disclosures["market_volatility"],
            self.risk_disclosures["operational_risk"],
            self.risk_disclosures["regulatory_risk"]
        ])

        # High-risk clients get additional disclosures
        if profile.risk_tolerance in [RiskLevel.AGGRESSIVE, RiskLevel.VERY_AGGRESSIVE]:
            required_disclosures.append(self.risk_disclosures["liquidity_risk"])

        # Large investment amounts require additional disclosures
        if profile.investment_amount >= 100000:
            required_disclosures.append(self.risk_disclosures["strategy_risk"])

        return required_disclosures

    def generate_risk_disclosure_document(self, client_id: str) -> str:
        """Generate a comprehensive risk disclosure document for a client"""
        if client_id not in self.client_profiles:
            raise ValueError(f"Client profile not found: {client_id}")

        profile = self.client_profiles[client_id]
        required_disclosures = self.get_required_disclosures(client_id)

        document = f"""
# RISK DISCLOSURE DOCUMENT

**Client ID:** {client_id}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Risk Tolerance:** {profile.risk_tolerance.value.title()}
**Investment Amount:** ${profile.investment_amount:,.2f}

## IMPORTANT RISK WARNINGS

This document outlines the significant risks associated with automated arbitrage trading in cryptocurrency markets.
By proceeding with this investment, you acknowledge that you have read, understood, and accept these risks.

"""

        for disclosure in required_disclosures:
            document += f"""
### {disclosure.title}

**Risk Category:** {disclosure.category.value.replace('_', ' ').title()}
**Severity:** {disclosure.severity.title()}

{disclosure.warning_text}

**Regulatory Requirements:**
"""
            for req in disclosure.regulatory_requirements:
                document += f"- {req}\n"

            document += f"""
**Mitigation Measures:**
"""
            for measure in disclosure.mitigation_measures:
                document += f"- {measure}\n"

            document += "\n---\n"

        document += f"""
## ACKNOWLEDGMENT OF RISKS

I, {client_id}, hereby acknowledge that:

1. I have read and understood all risk disclosures presented above
2. I understand that cryptocurrency trading involves substantial risk of loss
3. I am financially able to bear the risks of this investment
4. I have the necessary experience and knowledge to evaluate these risks
5. I accept full responsibility for investment decisions made

**Accepted Disclosures:** {', '.join(profile.disclosures_accepted)}
**Profile Created:** {profile.profile_created.strftime('%Y-%m-%d')}
**Last Reviewed:** {profile.last_reviewed.strftime('%Y-%m-%d') if profile.last_reviewed else 'Not reviewed'}

---
*This document is generated in compliance with FINRA Rule 2210 and SEC risk disclosure requirements.*
"""

        return document

    def check_disclosure_compliance(self, client_id: str) -> Dict[str, Any]:
        """Check if a client has accepted all required disclosures"""
        if client_id not in self.client_profiles:
            return {
                "compliant": False,
                "error": "Client profile not found",
                "required_disclosures": len(self.risk_disclosures),
                "accepted_disclosures": 0
            }

        profile = self.client_profiles[client_id]
        required_disclosures = self.get_required_disclosures(client_id)
        required_ids = {d.disclosure_id for d in required_disclosures}
        accepted_ids = set(profile.disclosures_accepted)

        missing_disclosures = required_ids - accepted_ids

        return {
            "compliant": len(missing_disclosures) == 0,
            "required_disclosures": len(required_ids),
            "accepted_disclosures": len(accepted_ids),
            "missing_disclosures": list(missing_disclosures),
            "profile_current": profile.last_reviewed and (datetime.now() - profile.last_reviewed).days < 365
        }

    async def update_disclosure(self, disclosure_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing risk disclosure"""
        if disclosure_id not in self.risk_disclosures:
            return False

        disclosure = self.risk_disclosures[disclosure_id]

        # Update fields
        for key, value in updates.items():
            if hasattr(disclosure, key):
                setattr(disclosure, key, value)

        disclosure.last_updated = datetime.now()
        disclosure.version = f"{float(disclosure.version) + 0.1:.1f}"

        # Audit the update
        await self.audit_logger.log_event(
            category="compliance",
            action="disclosure_updated",
            details={
                "disclosure_id": disclosure_id,
                "new_version": disclosure.version,
                "updated_fields": list(updates.keys())
            }
        )

        self.logger.info(f"Updated risk disclosure: {disclosure_id} v{disclosure.version}")
        return True

    def get_disclosure_summary(self) -> Dict[str, Any]:
        """Get summary of risk disclosure framework"""
        total_clients = len(self.client_profiles)
        compliant_clients = sum(
            1 for client_id in self.client_profiles.keys()
            if self.check_disclosure_compliance(client_id)["compliant"]
        )

        return {
            "total_disclosures": len(self.risk_disclosures),
            "total_clients": total_clients,
            "compliant_clients": compliant_clients,
            "compliance_rate": compliant_clients / total_clients if total_clients > 0 else 0,
            "disclosure_categories": list(set(d.category.value for d in self.risk_disclosures.values()))
        }


# Global risk disclosure framework instance
risk_disclosure_framework = RiskDisclosureFramework()


async def initialize_risk_disclosure():
    """Initialize the risk disclosure framework"""
    print("[RISK] Initializing Risk Disclosure Framework...")

    # Create default client profile for system
    await risk_disclosure_framework.create_client_risk_profile(
        client_id="system_admin",
        risk_tolerance=RiskLevel.CONSERVATIVE,
        investment_amount=1000000.0,
        time_horizon="long_term",
        experience_level="expert"
    )

    # Accept all disclosures for system admin
    for disclosure_id in risk_disclosure_framework.risk_disclosures.keys():
        await risk_disclosure_framework.accept_risk_disclosure("system_admin", disclosure_id)

    summary = risk_disclosure_framework.get_disclosure_summary()

    print("[OK] Risk disclosure framework initialized")
    print(f"  Total Disclosures: {summary['total_disclosures']}")
    print(f"  Total Clients: {summary['total_clients']}")
    print(f"  Compliant Clients: {summary['compliant_clients']}")
    print(f"  Compliance Rate: {summary['compliance_rate']:.1%}")

    return True


if __name__ == "__main__":
    asyncio.run(initialize_risk_disclosure())