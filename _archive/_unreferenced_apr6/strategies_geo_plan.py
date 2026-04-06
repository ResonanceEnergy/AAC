"""
Geo Plan — Operational Base Tracker
=====================================
Tracks the Panama (primary) and Paraguay (secondary) relocation tasks,
residency timelines, banking setup, and milestone deadlines anchored
to the moon cycle schedule.

Primary Base:   PANAMA CITY, PANAMA
  Residency:    Friendly Nations Visa or Qualified Investor Visa
  Banking:      Towerbank / Banistmo (USD accounts)
  Tax:          Territorial — no tax on foreign-source income
  Crypto:       No capital gains tax on crypto (as of 2025)
  Lawyers:      Kraemer & Kraemer PA  |  Agroup Panama
  Trip:         June 10–20, 2026 (around Moon 4 new moon = June 25)

Secondary Base: ASUNCIÓN, PARAGUAY
  Residency:    Permanent residency via investment (~$70K USD)
  Banking:      Ueno Bank / Itaú Paraguay (USD accounts)
  Tax:          Territorial — 10% flat on local income, no foreign tax
  Crypto:       No capital gains tax on crypto (as of 2025)
  Lawyers:      NTL Trust / Estudio Jurídico Perezleo
  Trip:         July–August 2026 (Moon 5–6)

UAE Backup:     DUBAI, UAE (Moon 28+ / 2028+)
  Residency:    Golden Visa via DTA or real estate ($205K USD)
  Banking:      Emirates NBD / Mashreq Bank
  Trip:         Moon 28 milestone (~June 2028)

Singapore Peek: SINGAPORE (if UAE political risk emerges)
  Residency:    EntrePass or One Pass visa
  Banking:      DBS / OCBC
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class TaskStatus(str, Enum):
    NOT_STARTED   = "NOT_STARTED"
    IN_PROGRESS   = "IN_PROGRESS"
    DONE          = "DONE"
    BLOCKED       = "BLOCKED"

class TaskCategory(str, Enum):
    DOCUMENT      = "DOCUMENT"
    LEGAL         = "LEGAL"
    BANKING       = "BANKING"
    PROPERTY      = "PROPERTY"
    CRYPTO        = "CRYPTO"
    TRAVEL        = "TRAVEL"
    INTEL         = "INTEL"
    HEALTH        = "HEALTH"
    COMMS         = "COMMS"


# ═══════════════════════════════════════════════════════════════════════════
# TASK DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GeoTask:
    """One action item in the geo relocation plan."""
    id: str                           # e.g. "PAN-DOC-01"
    title: str
    category: TaskCategory
    base: str                          # "PANAMA" | "PARAGUAY" | "UAE" | "CANADA"
    status: TaskStatus = TaskStatus.NOT_STARTED
    target_moon: Optional[int] = None  # Which moon cycle to complete by
    target_date: Optional[date] = None
    notes: str = ""
    depends_on: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.status == TaskStatus.DONE

    @property
    def emoji(self) -> str:
        return {
            TaskStatus.NOT_STARTED: "○",
            TaskStatus.IN_PROGRESS: "◑",
            TaskStatus.DONE:        "●",
            TaskStatus.BLOCKED:     "✗",
        }[self.status]


# ═══════════════════════════════════════════════════════════════════════════
# CANADA — PRE-DEPARTURE TASKS
# All must be done BEFORE Moon 4 (June 2026 Panama trip)
# ═══════════════════════════════════════════════════════════════════════════

CANADA_PREDEPARTURE: List[GeoTask] = [
    GeoTask(
        id="CAN-DOC-01",
        title="Obtain RCMP Criminal Record Check (apostille ready)",
        category=TaskCategory.DOCUMENT,
        base="CANADA",
        target_moon=3,
        notes="Order via RCMP online. Takes 3-8 weeks. Apostille via Global Affairs Canada.",
    ),
    GeoTask(
        id="CAN-DOC-02",
        title="Obtain long-form birth certificate (apostille ready)",
        category=TaskCategory.DOCUMENT,
        base="CANADA",
        target_moon=3,
        notes="Order from BC/AB Vital Statistics. Apostille required for Panama FNV.",
    ),
    GeoTask(
        id="CAN-DOC-03",
        title="Obtain passport — 2 blank pages minimum, 2+ years validity",
        category=TaskCategory.DOCUMENT,
        base="CANADA",
        target_moon=2,
        notes="If renewal needed, allow 8-12 weeks. URGENT passport available in 3 days.",
    ),
    GeoTask(
        id="CAN-DOC-04",
        title="Obtain proof of income / pension / bank statement (apostille)",
        category=TaskCategory.DOCUMENT,
        base="CANADA",
        target_moon=3,
        notes=(
            "Panama FNV requires proof of $2,000/month stable income "
            "or $200K bank deposit. Show AAC trading income or IBKR statements."
        ),
    ),
    GeoTask(
        id="CAN-DOC-05",
        title="Translate all docs to Spanish (certified translator)",
        category=TaskCategory.DOCUMENT,
        base="CANADA",
        target_moon=3,
        notes="Use in-country certified translator in Panama, or pre-certified in Canada.",
        depends_on=["CAN-DOC-01", "CAN-DOC-02", "CAN-DOC-04"],
    ),
    GeoTask(
        id="CAN-FIN-01",
        title="Set up USD wire transfer capability from Canadian bank",
        category=TaskCategory.BANKING,
        base="CANADA",
        target_moon=2,
        notes="Need SWIFT wire ability to fund Panama/Paraguay accounts. Test small $500 wire.",
    ),
    GeoTask(
        id="CAN-FIN-02",
        title="Open Wise account (multi-currency, Panama CAD→USD bridge)",
        category=TaskCategory.BANKING,
        base="CANADA",
        target_moon=2,
        notes="Wise supports USD routing number. Use for Panama rent/invoices.",
    ),
    GeoTask(
        id="CAN-FIN-03",
        title="Open Binance.ca or Coinbase account (CAD fiat on-ramp exit)",
        category=TaskCategory.CRYPTO,
        base="CANADA",
        target_moon=2,
        notes="Need a clean CAD→stablecoin path before departing Canada.",
    ),
    GeoTask(
        id="CAN-HEALTH-01",
        title="Obtain 6-month supply of any prescription medications",
        category=TaskCategory.HEALTH,
        base="CANADA",
        target_moon=3,
        notes="Panama/Paraguay pharmacies carry most generics but verify ahead.",
    ),
    GeoTask(
        id="CAN-COMMS-01",
        title="Set up international phone plan or get unlocked phone + local SIM plan",
        category=TaskCategory.COMMS,
        base="CANADA",
        target_moon=3,
        notes="Cable & Wireless or Claro Panama SIM. Tigo for Paraguay.",
    ),
    GeoTask(
        id="CAN-INTEL-01",
        title="Research Panama City neighborhoods: Punta Pacifica, Marbella, Casco Viejo",
        category=TaskCategory.INTEL,
        base="CANADA",
        target_moon=2,
        notes="Punta Pacifica = expat hub, walking distance to hospitals. $1,500-2,500/mo rent.",
    ),
    GeoTask(
        id="CAN-INTEL-02",
        title="Research Paraguay: Asunción (Villa Morra), Encarnación (alt)",
        category=TaskCategory.INTEL,
        base="CANADA",
        target_moon=3,
        notes="Villa Morra is financial district. Very cheap: $600-900/mo for full apartment.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# PANAMA — JUNE 2026 TRIP (Moon 4 window, June 10-20)
# ═══════════════════════════════════════════════════════════════════════════

PANAMA_TASKS: List[GeoTask] = [
    GeoTask(
        id="PAN-TRAVEL-01",
        title="Book flights: Canada → Panama City (PTY) for June 10-20, 2026",
        category=TaskCategory.TRAVEL,
        base="PANAMA",
        target_moon=3,
        notes="Direct flights: Air Canada (YYC/YVR→PTY), Copa (PTY hub). $600-900 return.",
    ),
    GeoTask(
        id="PAN-TRAVEL-02",
        title="Book Airbnb/hotel in Punta Pacifica or Marbella for June 10-20",
        category=TaskCategory.TRAVEL,
        base="PANAMA",
        target_moon=3,
        notes="Budget $100-150/night. Check Airbnb, Booking.com. Avoid Calidonia area.",
    ),
    GeoTask(
        id="PAN-LEGAL-01",
        title="Book consultation with Kraemer & Kraemer PA (Friendly Nations Visa)",
        category=TaskCategory.LEGAL,
        base="PANAMA",
        target_moon=3,
        notes=(
            "Website: kraemer.com.pa  |  Friendly Nations Visa = ~$1,800 USD legal fee. "
            "Canada qualifies. Requires: passport, police check, birth cert, bank statement."
        ),
    ),
    GeoTask(
        id="PAN-LEGAL-02",
        title="Alternative: consult Agroup Panama (Qualified Investor Visa)",
        category=TaskCategory.LEGAL,
        base="PANAMA",
        target_moon=3,
        notes=(
            "Qualified Investor: $160K USD in bank deposit OR real estate OR securities. "
            "2-year process. Get quote and compare to FNV."
        ),
    ),
    GeoTask(
        id="PAN-BANK-01",
        title="Visit Towerbank International — open USD savings account",
        category=TaskCategory.BANKING,
        base="PANAMA",
        target_moon=4,
        notes=(
            "Towerbank: known for expat-friendly USD accounts. "
            "Requirement: ~$10K min deposit, proof of income, passport, references. "
            "Address: Punta Pacifica financial center."
        ),
        depends_on=["CAN-DOC-04"],
    ),
    GeoTask(
        id="PAN-BANK-02",
        title="Visit Banistmo (HSBC Panama) — backup USD account",
        category=TaskCategory.BANKING,
        base="PANAMA",
        target_moon=4,
        notes="Banistmo is HSBC affiliate. More mainstream. Min deposit $2,500 USD.",
    ),
    GeoTask(
        id="PAN-CRYPTO-01",
        title="Identify local crypto OTC desk or broker in Panama City",
        category=TaskCategory.CRYPTO,
        base="PANAMA",
        target_moon=4,
        notes=(
            "Panama has no crypto law yet — both legal grey zone and tax-free. "
            "Find local USDT/USDC OTC desk. Ask lawyer for referrals."
        ),
    ),
    GeoTask(
        id="PAN-INTEL-01",
        title="Scout 2-3 neighborhoods on foot — assess walkability + security",
        category=TaskCategory.INTEL,
        base="PANAMA",
        target_moon=4,
        notes="Punta Pacifica, Costa del Este, Marbella. Note grocery/pharmacy proximity.",
    ),
    GeoTask(
        id="PAN-INTEL-02",
        title="Get referral to English-speaking doctor + dentist in Panama City",
        category=TaskCategory.HEALTH,
        base="PANAMA",
        target_moon=4,
        notes="Punta Pacifica Hospital is world-class. Johns Hopkins affiliate.",
    ),
    GeoTask(
        id="PAN-INTEL-03",
        title="Price 1BR / 2BR furnished apartments for monthly rent",
        category=TaskCategory.PROPERTY,
        base="PANAMA",
        target_moon=4,
        notes=(
            "Target: $1,200-2,000/mo furnished with AC + WiFi. "
            "Check Facebook expat groups, Encuentra24.com, MLS Panama."
        ),
    ),
    GeoTask(
        id="PAN-LEGAL-03",
        title="Open Panama corporation (S.A.) for holding crypto + operating AAC",
        category=TaskCategory.LEGAL,
        base="PANAMA",
        target_moon=5,
        notes=(
            "Panama S.A. formation: ~$1,200 USD. Holdings in corp name. "
            "Required for corporate bank account + Binance business KYC."
        ),
    ),
    GeoTask(
        id="PAN-BANK-03",
        title="Open Towerbank CORPORATE USD account in Panama S.A. name",
        category=TaskCategory.BANKING,
        base="PANAMA",
        target_moon=6,
        notes="Once corp is formed (Moon 5), open corporate account. Wire AAC funds here.",
        depends_on=["PAN-LEGAL-03"],
    ),
    GeoTask(
        id="PAN-LEGAL-04",
        title="File Friendly Nations Visa application (residency)",
        category=TaskCategory.LEGAL,
        base="PANAMA",
        target_moon=6,
        notes=(
            "Timeline: docs submitted, then ~6-8 months processing. "
            "Lawyer submits on behalf. In-country residence card issued after approval."
        ),
        depends_on=["PAN-LEGAL-01", "CAN-DOC-01", "CAN-DOC-02", "CAN-DOC-04"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# PARAGUAY — JULY-AUGUST 2026 TRIP (Moon 5-6)
# ═══════════════════════════════════════════════════════════════════════════

PARAGUAY_TASKS: List[GeoTask] = [
    GeoTask(
        id="PRY-TRAVEL-01",
        title="Book flights: Panama → Asunción (ASU) or Canada → ASU, July 2026",
        category=TaskCategory.TRAVEL,
        base="PARAGUAY",
        target_moon=4,
        notes=(
            "Copa Airlines (PTY→ASU) is cleanest route if already in Panama. "
            "Alternatively: direct Air Canada YYZ→GRU→ASU via LATAM."
        ),
    ),
    GeoTask(
        id="PRY-TRAVEL-02",
        title="Book accommodation in Villa Morra, Asunción for 10-14 days",
        category=TaskCategory.TRAVEL,
        base="PARAGUAY",
        target_moon=4,
        notes="Budget: $40-80/night. Airbnb sparse — use Booking.com. Villa Morra is safe.",
    ),
    GeoTask(
        id="PRY-LEGAL-01",
        title="Book consultation with NTL Trust (Paraguay residency + corp)",
        category=TaskCategory.LEGAL,
        base="PARAGUAY",
        target_moon=4,
        notes=(
            "NTL Trust is top Paraguay expat legal firm. "
            "Permanent Residency: ~$70K USD investment requirement (real estate or deposit). "
            "Or 'simple residency' via proof of income $1,500/mo no investment required."
        ),
    ),
    GeoTask(
        id="PRY-BANK-01",
        title="Open Ueno Bank USD/PYG account (digital-friendly)",
        category=TaskCategory.BANKING,
        base="PARAGUAY",
        target_moon=5,
        notes=(
            "Ueno is Paraguay's modern digital bank (similar to Wise). "
            "Opens with passport + ID only. No minimum deposit. "
            "SWIFT USD transfers supported."
        ),
    ),
    GeoTask(
        id="PRY-BANK-02",
        title="Open Itaú Paraguay account (reliable mainstream banking)",
        category=TaskCategory.BANKING,
        base="PARAGUAY",
        target_moon=5,
        notes="Itaú has branches in all major cities. Standard KYC. Min deposit ~$500 USD.",
    ),
    GeoTask(
        id="PRY-CRYPTO-01",
        title="Register with Binance Paraguay or local P2P exchange",
        category=TaskCategory.CRYPTO,
        base="PARAGUAY",
        target_moon=5,
        notes=(
            "Paraguay has cheap hydro electricity — major mining hub. "
            "P2P OTC desks widely available. USDT ↔ PYG/USD conversion easy."
        ),
    ),
    GeoTask(
        id="PRY-INTEL-01",
        title="Scout Villa Morra + Carmelitas neighborhoods on foot",
        category=TaskCategory.INTEL,
        base="PARAGUAY",
        target_moon=5,
        notes="Carmelitas = expat favourite. Multiple co-working spaces, restaurants, safe.",
    ),
    GeoTask(
        id="PRY-INTEL-02",
        title="Price 1BR / 2BR furnished apartments for monthly rent in Asunción",
        category=TaskCategory.PROPERTY,
        base="PARAGUAY",
        target_moon=5,
        notes=(
            "VERY cheap: $400-800/mo furnished with AC + WiFi. "
            "Check Facebook groups 'Expats in Asuncion Paraguay', InfoCasas.com.py."
        ),
    ),
    GeoTask(
        id="PRY-LEGAL-02",
        title="File Paraguay simple residency (income-based, no investment)",
        category=TaskCategory.LEGAL,
        base="PARAGUAY",
        target_moon=6,
        notes=(
            "Requires: passport, police check, birth cert, proof of income $1,500+/mo. "
            "Processing: 3-6 months. Lawyer submits. No minimum stay requirement."
        ),
        depends_on=["PRY-LEGAL-01", "CAN-DOC-01", "CAN-DOC-02"],
    ),
    GeoTask(
        id="PRY-CRYPTO-02",
        title="Install solar + battery UPS on Paraguay apartment for server uptime",
        category=TaskCategory.CRYPTO,
        base="PARAGUAY",
        target_moon=7,
        notes=(
            "Paraguay hydro is cheapest electricity in S. America. "
            "Run AAC trading server locally. Low grid outage risk post-solar."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# UAE — MOON 28 MILESTONE (~JUNE 2028)
# ═══════════════════════════════════════════════════════════════════════════

UAE_TASKS: List[GeoTask] = [
    GeoTask(
        id="UAE-INTEL-01",
        title="Research Dubai Golden Visa eligibility criteria (Moon 20 check-in)",
        category=TaskCategory.INTEL,
        base="UAE",
        target_moon=20,
        notes=(
            "Golden Visa: $205K USD real estate property OR employer sponsorship OR "
            "entrepreneur/investor path. 10-year renewable. No tax on personal income."
        ),
    ),
    GeoTask(
        id="UAE-INTEL-02",
        title="Research UAE crypto regulatory environment (Moon 24 review)",
        category=TaskCategory.INTEL,
        base="UAE",
        target_moon=24,
        notes=(
            "VARA = Virtual Assets Regulatory Authority. "
            "Bybit, OKX, Binance, Kraken — all licensed or in process. "
            "DIFC: financial free zone with common law jurisdiction."
        ),
    ),
    GeoTask(
        id="UAE-TRAVEL-01",
        title="Scout trip: Dubai + Abu Dhabi, Moon 28 window (~June 2028)",
        category=TaskCategory.TRAVEL,
        base="UAE",
        target_moon=28,
        notes=(
            "Assess: Marina, Downtown Dubai, DIFC area for long-term base. "
            "Visit VARA and DIFC offices. Meet 2 licensed crypto brokers."
        ),
    ),
    GeoTask(
        id="UAE-LEGAL-01",
        title="Engage Golden Visa agent / immigration lawyer in Dubai",
        category=TaskCategory.LEGAL,
        base="UAE",
        target_moon=28,
        notes="DTA (Dubai Tenancy Authority) accredited agents. Budget $3,000-5,000 fees.",
    ),
    GeoTask(
        id="UAE-BANK-01",
        title="Open Emirates NBD or Mashreq Bank account in Dubai",
        category=TaskCategory.BANKING,
        base="UAE",
        target_moon=29,
        notes=(
            "UAE banking requires in-person visit. "
            "Emirates NBD: min $10K deposit. Mashreq: more flexible for new residents."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# MOON MILESTONE MAP
# Maps moon number → key geo milestone
# ═══════════════════════════════════════════════════════════════════════════

MOON_MILESTONE_MAP: Dict[int, str] = {
    1:  "Start: Canada pre-departure checklist",
    2:  "RCMP check + passport + Wise account",
    3:  "All apostille docs ready. Panama flights booked.",
    4:  "★ PANAMA TRIP (Jun 10-20, 2026) — Scout + legal consult + bank visit",
    5:  "Paraguay trip (July-August 2026) — Scout + Ueno Bank + lawyer",
    6:  "Paraguay residency filed. Panama corp filed.",
    7:  "Panama FNV application in. Paraguay address established.",
    8:  "Server relocation research (Paraguay co-lo vs home setup)",
    9:  "Panama residency card expected. 1st quarterly review.",
    10: "Evaluate: Panama vs Paraguay as primary base.",
    11: "Confirm primary base city. Sign 12-month lease.",
    12: "Life Boat complete. Pre-ignition readiness check.",
    13: "★ DEFAULT IGNITION — Rocket Ship phase begins",
    14: "First rocket allocation deployed. Flare bridge tested.",
    15: "Morpho FXRP vault position opened.",
    16: "Solana Marinade staking active. LP pools initiated.",
    17: "First yield harvest and reinvestment.",
    18: "6-month trajectory review. Adjust geo plan.",
    19: "Paraguay permanent residency target.",
    20: "UAE Golden Visa intel review (18-month ahead check).",
    21: "Annual portfolio review — rebalance Rocket targets.",
    22: "Check: BRICS Unit tokenized proxy availability.",
    23: "Check: mBridge volume milestones.",
    24: "UAE crypto regulatory review.",
    25: "Mid-Rocket trajectory review.",
    26: "Evaluate Singapore EntrePass as alternate path.",
    27: "Panama corp 1-year filing renewal.",
    28: "★ UAE SCOUT TRIP (~June 2028). Golden Visa meeting.",
    29: "UAE bank account opened.",
    30: "Re-evaluate: primary HQ for Rocket Orbit phase.",
    31: "ETH staking compounding review.",
    32: "XRP/Flare DeFi yield optimization.",
    33: "Assess SOL ecosystem maturity for Rocket Orbit.",
    34: "BRICS Unit direct exposure evaluation.",
    35: "2-year portfolio total performance review.",
    36: "Begin ORBIT phase preparation.",
    37: "Consider: Singapore One Pass (if UAE not progressing).",
    38: "Final Rocket Ship phase allocation review.",
    39: "★ ORBIT — Rocket Ship phase closes. Begin Orbit.",
}


# ═══════════════════════════════════════════════════════════════════════════
# GEO PLAN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class GeoPlanEngine:
    """
    Tracks all geo tasks and produces milestone dashboards.
    """

    ALL_TASKS: List[GeoTask] = (
        CANADA_PREDEPARTURE + PANAMA_TASKS + PARAGUAY_TASKS + UAE_TASKS
    )

    STATE_FILE = Path("data/rocket_ship_geo_state.json")

    def tasks_for_base(self, base: str) -> List[GeoTask]:
        return [t for t in self.ALL_TASKS if t.base == base]

    def tasks_for_moon(self, moon: int) -> List[GeoTask]:
        return [t for t in self.ALL_TASKS if t.target_moon == moon]

    def incomplete_tasks(self) -> List[GeoTask]:
        return [t for t in self.ALL_TASKS if not t.is_complete]

    def complete_tasks(self) -> List[GeoTask]:
        return [t for t in self.ALL_TASKS if t.is_complete]

    def mark_done(self, task_id: str) -> bool:
        for t in self.ALL_TASKS:
            if t.id == task_id:
                t.status = TaskStatus.DONE
                logger.info("Task marked DONE: %s", task_id)
                return True
        return False

    def save_state(self) -> None:
        """Persist task states to JSON."""
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        states = {t.id: t.status.value for t in self.ALL_TASKS}
        self.STATE_FILE.write_text(json.dumps(states, indent=2), encoding="utf-8")
        logger.info("Geo state saved: %s", self.STATE_FILE)

    def load_state(self) -> None:
        """Load previously saved task states from JSON."""
        if not self.STATE_FILE.exists():
            return
        states: Dict[str, str] = json.loads(self.STATE_FILE.read_text(encoding="utf-8"))
        task_map = {t.id: t for t in self.ALL_TASKS}
        for tid, status_val in states.items():
            if tid in task_map:
                task_map[tid].status = TaskStatus(status_val)
        logger.info("Geo state loaded: %s tasks", len(states))

    def format_dashboard(self, current_moon: int = 1, base_filter: Optional[str] = None) -> str:
        """ASCII dashboard of geo tasks relevant to current moon."""
        milestone = MOON_MILESTONE_MAP.get(current_moon, "—")
        upcoming = [t for t in self.ALL_TASKS
                    if t.target_moon is not None and current_moon <= t.target_moon <= current_moon + 2
                    and not t.is_complete]
        if base_filter:
            upcoming = [t for t in upcoming if t.base == base_filter]
        upcoming.sort(key=lambda t: (t.target_moon or 99, t.id))

        total = len(self.ALL_TASKS)
        done  = len(self.complete_tasks())

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════════════╗",
            f"║  GEO PLAN TRACKER   Moon {current_moon:<3}   Progress: {done}/{total} tasks complete  ║",
            f"║  Milestone:  {milestone[:58]:<58}  ║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
        ]

        bases_shown: List[str] = (
            [base_filter] if base_filter else ["CANADA", "PANAMA", "PARAGUAY", "UAE"]
        )
        for base in bases_shown:
            base_tasks = self.tasks_for_base(base)
            done_b = sum(1 for t in base_tasks if t.is_complete)
            lines.append(f"║  {base:<10}  [{done_b}/{len(base_tasks)} done]" + " " * 48 + "  ║")
            for t in base_tasks:
                if t.target_moon and t.target_moon <= current_moon + 2:
                    moon_str = f"M{t.target_moon}" if t.target_moon else "  "
                    lines.append(
                        f"║    {t.emoji}  [{moon_str:>3}]  [{t.category.value[:4]}]  {t.title[:47]:<47}  ║"
                    )

        if upcoming:
            lines.append("╠══════════════════════════════════════════════════════════════════════════╣")
            lines.append("║  UPCOMING (next 3 moons):                                               ║")
            for t in upcoming[:6]:
                lines.append(
                    f"║    {t.emoji} M{t.target_moon:<2}  {t.id:<14}  {t.title[:44]:<44}  ║"
                )

        lines.append("╚══════════════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)
