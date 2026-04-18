"""
Rocket Ship — 15 Key Indicators + Gulf Yuan Ignition Trigger
=============================================================
Tracks the structural de-dollarization thesis and signals when the
Life Boat phase ends and the Rocket Ship ignites.

The 15 indicators:
    1.  USD global reserve share (IMF COFER quarterly)
    2.  Central bank gold purchases (WGC monthly)
    3.  BRICS Unit pilot scale (IRIAS/BRICS summit updates)
    4.  mBridge transaction volume (Atlantic Council / BIS)
    5.  Yuan global trade finance share (SWIFT RMB Tracker monthly)
    6.  XRP ODL / Ripple utility volume (Ripple quarterly)
    7.  Flare TVL / FXRP minted (DefiLlama / Flare dashboard)
    8.  Solana stablecoin transfer volume (Messari / Solana Explorer)
    9.  Global stablecoin market cap (DefiLlama / CoinGecko)
    10. Bitcoin ETF inflows / nation-state holdings (ETF trackers / Treasury)
    11. Ethereum staking / L2 DeFi activity (L2Beat / Beacon Chain)
    12. e-CNY transactions / international use (PBOC monthly)
    13. Gold price USD/oz (Kitco / LBMA)
    14. BRICS local-currency trade % (BRICS reports)
    15. ISO-20022 / CBDC interoperability progress (BIS / central bank pilots)

Gulf Yuan Oil Trigger (bonus ignition signal #16):
    Saudi Arabia or UAE confirms yuan oil contract settled via mBridge.
    Iran Hormuz yuan mandate already live (March 2026).
    Escalation path: WATCHING → EMERGING → CONFIRMED (= IGNITION)

Ignition criteria:
    ≥10 of 15 indicators GREEN  AND  trigger ≠ WATCHING  → IGNITE
    OR trigger == CONFIRMED (regardless of green count)  → IGNITE

Each indicator stores:
    - Current reading (str)
    - Status (GREEN / YELLOW / RED)
    - Trend direction
    - Last update date
    - Threshold logic and data sources
    - Manual override flag (for when API data is unavailable)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from strategies.rocket_ship.core import (
    INDICATORS_REQUIRED_FOR_IGNITION,
    IndicatorStatus,
    TriggerStatus,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IndicatorDefinition:
    """Full specification of one monitored indicator."""
    id: int
    name: str
    code: str                          # Short identifier
    description: str
    current_reading: str               # Latest known value
    baseline_reading: str              # Value when thesis began (March 2026)
    green_threshold: str               # What makes this GREEN
    yellow_range: str                  # What is inconclusive
    red_threshold: str                 # What is against thesis
    trend: str                         # "↑" | "↓" | "→"
    data_sources: List[str]            # Free public tracking sources
    update_frequency: str              # "daily" | "weekly" | "monthly" | "quarterly"
    last_updated: date = field(default_factory=date.today)
    status: IndicatorStatus = IndicatorStatus.YELLOW
    manual_override: bool = False      # If True, status set manually


@dataclass
class IndicatorState:
    """Runtime state of an indicator (mutable)."""
    code: str
    status: IndicatorStatus
    current_reading: str
    last_updated: datetime
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# THE 15 INDICATORS (research as of March 2026)
# ═══════════════════════════════════════════════════════════════════════════

INDICATORS: List[IndicatorDefinition] = [
    # ── 1 ── USD Reserve Share ──────────────────────────────────────────────
    IndicatorDefinition(
        id=1,
        name="USD Global Reserve Share",
        code="USD_RESERVE",
        description=(
            "IMF COFER data tracks what % of global foreign exchange reserves are held"
            " in USD. Peak was 72% in 2000. Structural decline confirms de-dollarization."
        ),
        current_reading="~56-58%",
        baseline_reading="~65% (2015)",
        green_threshold="Below 55% OR declining quarter-over-quarter for 3+ quarters",
        yellow_range="55-62%, trend unclear",
        red_threshold="Rising above 62%+ (flight to safety, dollar dominance resurgence)",
        trend="↓",
        data_sources=["IMF COFER (imf.org/external/np/sta/cofer)", "Fed FRBNY"],
        update_frequency="quarterly",
        status=IndicatorStatus.GREEN,
    ),
    # ── 2 ── Central Bank Gold Purchases ──────────────────────────────────
    IndicatorDefinition(
        id=2,
        name="Central Bank Gold Purchases",
        code="CB_GOLD",
        description=(
            "Net tonnes of gold purchased by central banks globally per year."
            " Record levels (>1,000t/yr since 2022) signal sovereign reserve"
            " de-dollarization. Physical gold = the BRICS Unit anchor."
        ),
        current_reading="Net 5t (Jan 2026); ~1,000t+ annual pace",
        baseline_reading="~500t/yr (pre-2022)",
        green_threshold="Annual rate >800t OR single month net buy >50t",
        yellow_range="200-800t/yr",
        red_threshold="Net selling or <200t/yr (reversal of trend)",
        trend="↑",
        data_sources=["World Gold Council (gold.org/research)", "WGC Gold Demand Trends"],
        update_frequency="monthly",
        status=IndicatorStatus.GREEN,
    ),
    # ── 3 ── BRICS Unit Pilot Scale ──────────────────────────────────────
    IndicatorDefinition(
        id=3,
        name="BRICS Unit Pilot Scale",
        code="BRICS_UNIT",
        description=(
            "Tracks operationalization of the BRICS Unit (40% gold + 60% basket)."
            " Pilot launched Oct 31 2025 via IRIAS/Russian Academy — 100 units issued"
            " (~0.98g gold each). 2026 India BRICS chairship pushing CBDC bridge links."
        ),
        current_reading="Pilot: 100 units issued (~0.98g gold; $95 USD)",
        baseline_reading="Pre-pilot (Sep 2025)",
        green_threshold=(
            "Official multi-country adoption OR second pilot expansion"
            " OR India BRICS CBDC link confirmed"
        ),
        yellow_range="Pilot ongoing, no new expansion news",
        red_threshold="Pilot officially cancelled or BRICS summit rejects Unit",
        trend="↑",
        data_sources=["IRIAS (iras.ru)", "BRICS 2026 Summit communiqués", "Reuters BRICS"],
        update_frequency="monthly",
        status=IndicatorStatus.YELLOW,   # Pilot only, not scaled yet
    ),
    # ── 4 ── mBridge Transaction Volume ────────────────────────────────────
    IndicatorDefinition(
        id=4,
        name="mBridge Transaction Volume",
        code="MBRIDGE_VOL",
        description=(
            "Cumulative value processed on Project mBridge (multi-CBDC wholesale"
            " settlement platform). $55.5B+ processed (4,000+ txns), 95% e-CNY."
            " Saudi/UAE full members since 2024. UAE govt txns live (Nov 2025)."
            " THIS IS THE PRIMARY IGNITION SIGNAL — Gulf energy settlement spike here"
            " = Rocket Ship launch."
        ),
        current_reading="$55.5B+ cumulative (Nov 2025 data)",
        baseline_reading="$22M (2022 pilot)",
        green_threshold="Monthly new volume >$5B OR Gulf energy txn confirmed",
        yellow_range="$1-5B new monthly volume, no energy-specific news",
        red_threshold="Volume stagnation or major country withdrawal from platform",
        trend="↑",
        data_sources=[
            "Atlantic Council CBDC Tracker (atlanticcouncil.org/cbdctracker)",
            "BIS Innovation Hub (bis.org)",
            "mBridge official reports",
        ],
        update_frequency="monthly",
        status=IndicatorStatus.GREEN,
    ),
    # ── 5 ── Yuan Global Trade Finance Share ─────────────────────────────
    IndicatorDefinition(
        id=5,
        name="Yuan Global Trade Finance Share",
        code="YUAN_TRADE",
        description=(
            "SWIFT RMB Tracker: yuan's share of global trade finance and payments."
            " Currently 8%+ (4x since 2020). ~30-39% of China's own cross-border trade."
            " Growing in BRICS invoicing; digital rails accelerate."
        ),
        current_reading="~8%+ global trade finance",
        baseline_reading="~2% (2020)",
        green_threshold=">10% global trade finance OR >50% China bilateral BRICS trade",
        yellow_range="6-10%",
        red_threshold="Declining below 5% (reversal)",
        trend="↑",
        data_sources=[
            "SWIFT RMB Tracker (swift.com/our-solutions/compliance-and-shared-services)",
            "PBOC cross-border payment data",
        ],
        update_frequency="monthly",
        status=IndicatorStatus.GREEN,
    ),
    # ── 6 ── XRP ODL / Ripple Utility Volume ─────────────────────────────
    IndicatorDefinition(
        id=6,
        name="XRP ODL / Ripple Utility Volume",
        code="XRP_ODL",
        description=(
            "Ripple On-Demand Liquidity (ODL) transaction volume. Trillions cumulative."
            " Conditional OCC national trust bank charter (Dec 2025). Active in Asia,"
            " Middle East, Africa, LatAm — perfect for mBridge corridor bridging."
            " ISO-20022 native. No speculative BRICS deal confirmed."
        ),
        current_reading="Trillions cumulative ODL; OCC charter conditional",
        baseline_reading="Millions (2020)",
        green_threshold="OCC charter fully granted OR new bank/CBDC ODL integration OR Ripple HQ geo-expansion",
        yellow_range="Steady ODL growth, no major institutional news",
        red_threshold="SEC enforcement revival OR major bank exits RippleNet",
        trend="↑",
        data_sources=[
            "Ripple quarterly insights (ripple.com/insights)",
            "RippleNet partner announcements",
            "ODL corridor data (indirect via XRP Ledger analytics)",
        ],
        update_frequency="monthly",
        status=IndicatorStatus.GREEN,
    ),
    # ── 7 ── Flare TVL / FXRP Minted ──────────────────────────────────────
    IndicatorDefinition(
        id=7,
        name="Flare TVL / FXRP Minted",
        code="FLARE_TVL",
        description=(
            "Total Value Locked on Flare Network and FXRP minting activity."
            " TVL ~$200M (+400% YoY). 100M+ FXRP minted. Institutional minting"
            " via Hex Trust. SparkDEX, Kinetic, Enosys, Morpho vaults live."
            " Flare's 5B XRP ecosystem TVL target is the thesis compass."
        ),
        current_reading="TVL ~$200M; 100M+ FXRP minted",
        baseline_reading="TVL ~$40M (early 2025)",
        green_threshold="TVL >$500M OR FXRP minted >500M OR major institutional integration",
        yellow_range="TVL $100-500M",
        red_threshold="TVL collapse >50% OR Flare exploit / hack",
        trend="↑",
        data_sources=[
            "DefiLlama (defillama.com/chain/flare)",
            "Flare Dashboard (flare.network)",
            "Dune Analytics (Flare FXRP dashboard)",
        ],
        update_frequency="weekly",
        status=IndicatorStatus.GREEN,
    ),
    # ── 8 ── Solana Stablecoin Transfer Volume ────────────────────────────
    IndicatorDefinition(
        id=8,
        name="Solana Stablecoin Transfer Volume",
        code="SOL_STABLE_VOL",
        description=(
            "Monthly stablecoin transaction volume on Solana. Currently leads all"
            " blockchain networks at $650B+/mo. Institutional adoption (JPM commercial"
            " paper). USDC supply ~$15B on Solana. The Web3 payments layer."
        ),
        current_reading="$650B+/month (leads all chains)",
        baseline_reading="~$100B/month (2024)",
        green_threshold=">$1T/month OR Solana IPO proceeds OR major CBDC pilot on Solana",
        yellow_range="$300B-$1T/month",
        red_threshold="Volume drops below $100B/month OR systemic exploit",
        trend="↑",
        data_sources=[
            "Messari (messari.io/asset/solana/metrics)",
            "Solana Explorer stablecoin stats",
            "Visa Crypto Insights Dashboard",
        ],
        update_frequency="monthly",
        status=IndicatorStatus.GREEN,
    ),
    # ── 9 ── Global Stablecoin Market Cap ────────────────────────────────
    IndicatorDefinition(
        id=9,
        name="Global Stablecoin Market Cap",
        code="STABLE_MCAP",
        description=(
            "Total market cap of all stablecoins globally. ~$300B+ (March 2026)."
            " USDT ~$187B, USDC ~$75B. GENIUS Act (US 2025) + MiCA (EU) = regulated"
            " rails. Signals institutional DeFi adoption and new-system utility demand."
        ),
        current_reading="~$300B+ total",
        baseline_reading="~$150B (2024)",
        green_threshold=">$500B total OR regulated gold-pegged stable >$10B",
        yellow_range="$200-500B",
        red_threshold="Collapse below $100B OR USDT de-peg event",
        trend="↑",
        data_sources=[
            "DefiLlama stablecoins (defillama.com/stablecoins)",
            "CoinGecko stablecoins category",
            "Circle/Tether transparency reports",
        ],
        update_frequency="weekly",
        status=IndicatorStatus.GREEN,
    ),
    # ── 10 ── Bitcoin ETF Inflows / Nation-State Holdings ─────────────────
    IndicatorDefinition(
        id=10,
        name="Bitcoin ETF Inflows & Nation-State Reserve",
        code="BTC_ETF",
        description=(
            "Bitcoin ETF AUM (currently $100B+) and US Strategic Bitcoin Reserve"
            " holdings. Nation-state adoption as reserve asset confirms BTC = digital"
            " gold in the multipolar era. El Salvador + US Strategic Reserve = template."
        ),
        current_reading="$100B+ AUM; US Strategic Bitcoin Reserve active",
        baseline_reading="$0 ETF AUM (pre-Jan 2024 approval)",
        green_threshold=">$200B ETF AUM OR 2nd nation-state reserve announcement",
        yellow_range="$50B-$200B ETF AUM",
        red_threshold="US Strategic Reserve sold or ETF regulatory reversal",
        trend="↑",
        data_sources=[
            "BitcoinTreasuries.net",
            "Bloomberg ETF tracker",
            "US Treasury announcements",
        ],
        update_frequency="weekly",
        status=IndicatorStatus.GREEN,
    ),
    # ── 11 ── Ethereum Staking / L2 Activity ──────────────────────────────
    IndicatorDefinition(
        id=11,
        name="Ethereum Staking / L2 DeFi Activity",
        code="ETH_L2",
        description=(
            "ETH validator staking volume (% of supply staked) and L2 DeFi TVL."
            " Growing staking ETF approvals (2026 possible). Fusaka upgrade improves"
            " scalability. L2 (Arbitrum, Base, Optimism) dominate DeFi TVL."
            " RWA tokenization on-chain is the 2100-proof use case."
        ),
        current_reading="~28% of ETH supply staked; L2 TVL growing",
        baseline_reading="~25% staked (2025)",
        green_threshold=">35% staked OR ETH staking ETF approved OR RWA TVL >$50B on-chain",
        yellow_range="20-35% staked, steady growth",
        red_threshold="Staking regulatory crackdown OR major L2 exploit",
        trend="↑",
        data_sources=[
            "Beacon Chain stats (beaconcha.in)",
            "L2Beat (l2beat.com)",
            "DefiLlama ETH/L2",
        ],
        update_frequency="weekly",
        status=IndicatorStatus.GREEN,
    ),
    # ── 12 ── e-CNY Transactions / International Use ─────────────────────
    IndicatorDefinition(
        id=12,
        name="e-CNY Transactions & International Use",
        code="ECNY_VOL",
        description=(
            "Digital yuan (e-CNY) transaction volume (domestic + international)."
            " $2.4T+ domestic. mBridge is 95% e-CNY. Jan 1 2026: e-CNY becomes"
            " interest-bearing — major boost to international holdings appeal."
            " Saudi/UAE government use on mBridge live."
        ),
        current_reading="$2.4T+ domestic; mBridge 95%; interest-bearing since Jan 2026",
        baseline_reading="$1T domestic (2024)",
        green_threshold=(
            "$5T+ domestic OR 3+ countries outside BRICS using e-CNY for trade"
        ),
        yellow_range="$2T-$5T domestic, limited international",
        red_threshold="Major countries reject e-CNY OR PBOC rolls back interest-bearing feature",
        trend="↑",
        data_sources=[
            "PBOC press releases (pbc.gov.cn)",
            "BIS mBridge reports",
            "Reuters/Bloomberg e-CNY coverage",
        ],
        update_frequency="monthly",
        status=IndicatorStatus.YELLOW,  # International use still limited
    ),
    # ── 13 ── Gold Price (USD/oz) ─────────────────────────────────────────
    IndicatorDefinition(
        id=13,
        name="Gold Price (USD/oz)",
        code="GOLD_PRICE",
        description=(
            "Gold price in USD. All-time high territory (~$3,000-$5,000+ per oz in 2026)."
            " Confirms BRICS Unit's 40% gold anchor is appreciating vs USD."
            " Structural gold bull market = structural USD weakness = thesis confirmation."
        ),
        current_reading="~$3,000+ (record territory)",
        baseline_reading="~$1,800 (2022)",
        green_threshold=">$3,000/oz (record highs, confirming de-dollarization)",
        yellow_range="$2,000-$3,000/oz",
        red_threshold="Strong sustained move below $1,800 (risk-off, dollar flight)",
        trend="↑",
        data_sources=[
            "Kitco Gold Charts (kitco.com)",
            "LBMA Gold Price (lbma.org.uk)",
            "TradingView XAUUSD",
        ],
        update_frequency="daily",
        status=IndicatorStatus.GREEN,
    ),
    # ── 14 ── BRICS Local-Currency Trade Percentage ───────────────────────
    IndicatorDefinition(
        id=14,
        name="BRICS Local-Currency Trade %",
        code="BRICS_LC",
        description=(
            "Percentage of trade between BRICS nations conducted in local currencies"
            " (not USD). Russia-China: 90%+ in rubles/yuan. India-Russia: ~70% rupee."
            " Confirms structural shift away from USD invoicing in commodity trade."
        ),
        current_reading="Russia-China: 90%+; India-Russia: ~70%; BRICS overall: rising",
        baseline_reading="Russia-China: ~50% (2020)",
        green_threshold=(
            "Russia-China >95% OR 3rd major bilateral non-USD trade deal (e.g., Brazil-China)"
        ),
        yellow_range="Russia-China 75-95%",
        red_threshold="Reversal — major BRICS member returns to USD invoicing",
        trend="↑",
        data_sources=[
            "BRICS summit communiqués",
            "Russian Central Bank reports",
            "Bloomberg bilateral trade data",
        ],
        update_frequency="monthly",
        status=IndicatorStatus.GREEN,
    ),
    # ── 15 ── ISO-20022 / CBDC Interoperability Progress ─────────────────
    IndicatorDefinition(
        id=15,
        name="ISO-20022 / CBDC Interoperability Progress",
        code="ISO_CBDC",
        description=(
            "Advancement of ISO-20022 financial messaging standard adoption and"
            " cross-CBDC interoperability pilots. XRP and Flare are ISO-20022 native."
            " mBridge uses ISO-20022 metadata. BIS Agora project tests bank tokenization."
            " More ISO-20022 adopters = more institutions able to use new rails."
        ),
        current_reading=(
            "XRP/Flare ISO-20022 native; mBridge ISO-20022 metadata live;"
            " BIS Agora active; SWIFT ISO-20022 migration mostly complete"
        ),
        baseline_reading="SWIFT beginning ISO-20022 migration (2020)",
        green_threshold=(
            "3+ major central banks go live with ISO-20022 CBDC interop"
            " OR BIS Agora successfully settles cross-border transactions"
        ),
        yellow_range="Pilots ongoing, no production go-live",
        red_threshold="BIS Agora cancelled OR major CBDC project abandoned",
        trend="↑",
        data_sources=[
            "BIS Innovation Hub (bis.org/innovation)",
            "SWIFT ISO-20022 migration tracker",
            "ISO-20022 Registry (iso20022.org)",
        ],
        update_frequency="monthly",
        status=IndicatorStatus.GREEN,
    ),
]

# Quick lookup by code
INDICATOR_MAP: Dict[str, IndicatorDefinition] = {ind.code: ind for ind in INDICATORS}


# ═══════════════════════════════════════════════════════════════════════════
# GULF YUAN OIL TRIGGER (Ignition Signal #16)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GulfYuanTrigger:
    """
    Tracks the Gulf Yuan Oil ignition trigger.

    Escalation path:
        WATCHING  → No confirmed news, Iran Hormuz mandate only (live since Q1 2026)
        EMERGING  → Swaps/partial flows, press statements, mBridge energy spike
        CONFIRMED → Official Saudi or UAE yuan oil contract via mBridge (= IGNITION)

    On CONFIRMED: Rocket Ship launches immediately regardless of moon cycle.
    """
    status: TriggerStatus = TriggerStatus.WATCHING
    last_updated: datetime = field(default_factory=datetime.utcnow)
    description: str = (
        "Iran Hormuz yuan mandate live (Q1 2026). Saudi Arabia (SAMA on mBridge since 2024)"
        " and UAE (govt txns Nov 2025) partially engaged. No full yuan oil contract announced."
    )
    evidence: List[str] = field(default_factory=list)
    confirmed_date: Optional[date] = None

    # Escalation indicators to watch
    watch_list: List[str] = field(default_factory=lambda: [
        "Saudi Aramco yuan-priced crude news (Reuters/S&P Global)",
        "UAE ADNOC mBridge oil volume spike",
        "Shanghai International Energy Exchange (INE) non-USD crude contracts",
        "mBridge monthly report showing energy settlement category",
        "Saudi Ministry of Energy press statements",
        "PBOC/SAMA joint announcement",
        "Bloomberg Gulf yuan crude reporting",
    ])

    def is_ignited(self) -> bool:
        return self.status == TriggerStatus.CONFIRMED

    def update_status(self, new_status: TriggerStatus, note: str = "") -> None:
        self.status = new_status
        self.last_updated = datetime.utcnow()
        if note:
            self.evidence.append(f"[{date.today()}] {note}")
        if new_status == TriggerStatus.CONFIRMED and self.confirmed_date is None:
            self.confirmed_date = date.today()
        logger.info("Gulf Yuan Trigger updated: %s — %s", new_status.value, note)


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class IndicatorEngine:
    """
    Manages all 15 indicators + Gulf trigger.
    Computes green count and determines if ignition criteria are met.
    """

    def __init__(self) -> None:
        self.indicators: List[IndicatorDefinition] = [
            IndicatorDefinition(**{
                k: v for k, v in ind.__dict__.items()
            }) for ind in INDICATORS
        ]
        self.gulf_trigger = GulfYuanTrigger()
        self._state_file = Path("rocket_ship_indicators.json")

    def count_by_status(self) -> Dict[str, int]:
        """Return count of GREEN / YELLOW / RED indicators."""
        counts: Dict[str, int] = {
            IndicatorStatus.GREEN.value: 0,
            IndicatorStatus.YELLOW.value: 0,
            IndicatorStatus.RED.value: 0,
        }
        for ind in self.indicators:
            counts[ind.status.value] += 1
        return counts

    def green_count(self) -> int:
        return sum(1 for ind in self.indicators if ind.status == IndicatorStatus.GREEN)

    def check_ignition(self) -> Tuple[bool, str]:
        """
        Returns (should_ignite: bool, reason: str).

        Ignition fires if:
          a) Gulf Trigger CONFIRMED (override), OR
          b) ≥ 10 of 15 indicators GREEN

        Returns immediately with first matching condition.
        """
        if self.gulf_trigger.is_ignited():
            return True, "GULF YUAN OIL TRIGGER CONFIRMED — Rocket Ship launches NOW"
        greens = self.green_count()
        if greens >= INDICATORS_REQUIRED_FOR_IGNITION:
            return True, f"{greens}/15 indicators GREEN — ignition threshold reached"
        return False, f"{greens}/15 GREEN (need {INDICATORS_REQUIRED_FOR_IGNITION})"

    def update_indicator(
        self,
        code: str,
        new_status: IndicatorStatus,
        new_reading: Optional[str] = None,
        notes: str = "",
    ) -> None:
        """Manually update an indicator's status (for weekly data entry)."""
        ind = INDICATOR_MAP.get(code)
        if ind is None:
            logger.warning("Unknown indicator code: %s", code)
            return
        # Update the live copy in self.indicators
        for live_ind in self.indicators:
            if live_ind.code == code:
                live_ind.status = new_status
                live_ind.last_updated = date.today()
                live_ind.manual_override = True
                if new_reading:
                    live_ind.current_reading = new_reading
                logger.info("Indicator %s updated: %s — %s", code, new_status.value, notes)
                break

    def format_dashboard(self) -> str:
        """Return a formatted ASCII dashboard of all 15 indicators."""
        STATUS_SYMBOLS = {
            IndicatorStatus.GREEN:  "● GREEN ",
            IndicatorStatus.YELLOW: "◌ YELLOW",
            IndicatorStatus.RED:    "✗ RED   ",
        }
        TREND_LABELS = {"↑": "RISING", "↓": "FALLING", "→": "FLAT"}

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════════════╗",
            "║         ROCKET SHIP — 15 INDICATOR DASHBOARD                           ║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
        ]
        for ind in self.indicators:
            sym = STATUS_SYMBOLS[ind.status]
            trend = ind.trend
            line = f"║  {ind.id:>2}. {sym}  {trend}  {ind.name:<42} ║"
            lines.append(line)

        counts = self.count_by_status()
        greens = counts[IndicatorStatus.GREEN.value]
        ignite, reason = self.check_ignition()

        lines += [
            "╠══════════════════════════════════════════════════════════════════════════╣",
            f"║  GREEN: {greens:>2}/15  │  Need: {INDICATORS_REQUIRED_FOR_IGNITION}/15  │"
            f"  {'IGNITION READY' if ignite else 'LIFE BOAT HOLD':^30}  ║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
        ]

        # Gulf trigger
        gt = self.gulf_trigger
        gt_line = f"  GULF YUAN TRIGGER: [{gt.status.value.upper():^12}]"
        lines.append(f"║{gt_line:<74}║")
        lines.append("╚══════════════════════════════════════════════════════════════════════════╝")

        return "\n".join(lines)

    def save_state(self, path: Optional[Path] = None) -> None:
        """Persist indicator states to JSON."""
        out_path = path or self._state_file
        data: Dict[str, Any] = {
            "saved_at": datetime.utcnow().isoformat(),
            "indicators": [
                {
                    "code": ind.code,
                    "status": ind.status.value,
                    "current_reading": ind.current_reading,
                    "last_updated": ind.last_updated.isoformat(),
                    "manual_override": ind.manual_override,
                }
                for ind in self.indicators
            ],
            "gulf_trigger": {
                "status": self.gulf_trigger.status.value,
                "last_updated": self.gulf_trigger.last_updated.isoformat(),
                "description": self.gulf_trigger.description,
                "evidence": self.gulf_trigger.evidence,
                "confirmed_date": (
                    self.gulf_trigger.confirmed_date.isoformat()
                    if self.gulf_trigger.confirmed_date else None
                ),
            },
        }
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Indicator state saved to %s", out_path)

    def load_state(self, path: Optional[Path] = None) -> bool:
        """Load previous indicator states from JSON. Returns True if loaded."""
        load_path = path or self._state_file
        if not load_path.exists():
            return False
        try:
            data = json.loads(load_path.read_text(encoding="utf-8"))
            live_map = {ind.code: ind for ind in self.indicators}
            for saved in data.get("indicators", []):
                code = saved["code"]
                if code in live_map:
                    live_map[code].status = IndicatorStatus(saved["status"])
                    live_map[code].current_reading = saved.get("current_reading", "")
                    live_map[code].last_updated = date.fromisoformat(saved["last_updated"])
                    live_map[code].manual_override = saved.get("manual_override", False)
            gt = data.get("gulf_trigger", {})
            if gt:
                self.gulf_trigger.status = TriggerStatus(gt.get("status", "watching"))
                self.gulf_trigger.evidence = gt.get("evidence", [])
                cd = gt.get("confirmed_date")
                if cd:
                    self.gulf_trigger.confirmed_date = date.fromisoformat(cd)
            logger.info("Indicator state loaded from %s", load_path)
            return True
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load indicator state: %s", exc)
            return False


# Type hint fix for Tuple usage
from typing import Tuple  # noqa: E402
