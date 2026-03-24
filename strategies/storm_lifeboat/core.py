"""
Storm Lifeboat Matrix — Core Data Structures & Constants
==========================================================
All shared types, enums, asset definitions, scenario taxonomy,
and configuration for the Storm Lifeboat simulation engine.

43-scenario geopolitical/financial/systemic risk model.
20 tracked assets across equities, commodities, crypto, and fixed income.
4 volatility regimes: CALM → ELEVATED → CRISIS → PANIC.

Starting capital: $45,120 CAD (~$32,486 USD).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# FINANCIAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

STARTING_CAPITAL_CAD = 45_120.0
CAD_TO_USD = 0.72
STARTING_CAPITAL_USD = STARTING_CAPITAL_CAD * CAD_TO_USD
RISK_FREE_RATE = 0.045
TRADING_DAYS_PER_YEAR = 252
MC_DEFAULT_PATHS = 100_000
MC_DEFAULT_HORIZON = 90  # days


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class Asset(Enum):
    """Full asset universe — 20 instruments."""
    # Commodities / Havens
    OIL = "OIL"        # WTI Crude (via USO or /CL futures)
    GOLD = "GOLD"      # Gold (via GLD or /GC)
    SILVER = "SILVER"  # Silver (via SLV or /SI)
    GDX = "GDX"        # Gold Miners ETF
    # Equities
    SPY = "SPY"        # S&P 500
    QQQ = "QQQ"        # Nasdaq-100
    XLF = "XLF"        # Financials
    XLRE = "XLRE"      # Real Estate
    KRE = "KRE"        # Regional Banks
    JETS = "JETS"      # Airlines
    XLY = "XLY"        # Consumer Discretionary
    XLE = "XLE"        # Energy
    TSLA = "TSLA"      # Tesla — full ecosystem moat (EV, solar, Optimus, energy)
    SMR = "SMR"        # NuScale Power — small modular reactor proxy
    # Fixed Income
    TLT = "TLT"        # 20+ Year Treasury
    HYG = "HYG"        # High Yield Corporate
    # Crypto
    BTC = "BTC"        # Bitcoin
    ETH = "ETH"        # Ethereum
    XRP = "XRP"        # Ripple
    BITO = "BITO"      # Bitcoin ETF proxy


class VolRegime(Enum):
    """Volatility regime classification."""
    CALM = "calm"           # VIX < 15
    ELEVATED = "elevated"   # VIX 15-25
    CRISIS = "crisis"       # VIX 25-40
    PANIC = "panic"         # VIX > 40


class MandateLevel(Enum):
    """Trading mandate escalation levels."""
    OBSERVE = "observe"               # No action, monitor
    DEFENSIVE = "defensive"           # Hedge existing positions
    STANDARD = "standard"             # Normal bearish operations
    AGGRESSIVE = "aggressive"         # High conviction entries
    MAX_CONVICTION = "max_conviction" # Full tilt — tail risk confirmed


class MoonPhase(Enum):
    """13-moon cycle phases."""
    NEW = "new"             # Reset, reassess
    WAXING = "waxing"       # Accumulate positions
    FULL = "full"           # Execute decisions
    WANING = "waning"       # Hedge, trim, protect


class ScenarioStatus(Enum):
    """Tracking status for each scenario."""
    DORMANT = "dormant"         # Not yet showing signals
    EMERGING = "emerging"       # Early indicators firing
    ACTIVE = "active"           # Scenario in progress
    ESCALATING = "escalating"   # Intensity increasing
    PEAK = "peak"               # Maximum intensity
    RECEDING = "receding"       # Winding down


# ═══════════════════════════════════════════════════════════════════════════
# 43-SCENARIO TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioDefinition:
    """Full specification of a crisis scenario."""
    id: int
    name: str
    code: str                          # Short identifier
    description: str
    trigger_indicators: List[str]      # What to watch
    probability: float                 # Current base probability (0-1)
    impact_severity: float             # 0-1 scale, how bad if realized
    beneficiary_assets: List[Asset]    # Go UP if scenario fires
    victim_assets: List[Asset]         # Go DOWN if scenario fires
    oil_sensitivity: float             # -1 to +1, how this scenario affects oil
    status: ScenarioStatus = ScenarioStatus.DORMANT


# Master scenario definitions
SCENARIOS: List[ScenarioDefinition] = [
    ScenarioDefinition(
        id=1, name="Hormuz Strait Closure", code="HORMUZ",
        description="Iran closes Strait of Hormuz — 20% of global oil transits blocked",
        trigger_indicators=["Oil > $120", "USN carrier deployment", "Iran naval exercises",
                            "Insurance rates Persian Gulf shipping",
                            "Kharg Island military activity", "Yuan-for-passage offers"],
        probability=0.45, impact_severity=0.95,
        beneficiary_assets=[Asset.OIL, Asset.GOLD, Asset.GDX, Asset.XLE, Asset.SMR],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.JETS, Asset.KRE, Asset.XLRE,
                       Asset.BTC, Asset.ETH],
        oil_sensitivity=1.0,
    ),
    ScenarioDefinition(
        id=2, name="US Sovereign Debt Crisis", code="DEBT_CRISIS",
        description="10Y yield spikes above 6%, debt-to-GDP triggers rating downgrade",
        trigger_indicators=["10Y yield > 5.5%", "CDS spreads widening", "Failed Treasury auction",
                            "Dollar index collapse"],
        probability=0.20, impact_severity=0.90,
        beneficiary_assets=[Asset.GOLD, Asset.SILVER, Asset.GDX, Asset.BTC],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.XLF, Asset.KRE, Asset.TLT, Asset.HYG],
        oil_sensitivity=0.2,
    ),
    ScenarioDefinition(
        id=3, name="China-Taiwan Escalation", code="TAIWAN",
        description="PLA military action against Taiwan — semiconductor supply chain severed",
        trigger_indicators=["PLA exercises near Taiwan", "US carrier groups Pacific",
                            "Semiconductor stockpiling", "TSMC share price crash"],
        probability=0.15, impact_severity=0.98,
        beneficiary_assets=[Asset.GOLD, Asset.OIL, Asset.GDX],
        victim_assets=[Asset.QQQ, Asset.SPY, Asset.XLY, Asset.ETH, Asset.BTC],
        oil_sensitivity=0.6,
    ),
    ScenarioDefinition(
        id=4, name="European Banking Contagion", code="EU_BANKS",
        description="Major European bank failure triggers systemic contagion",
        trigger_indicators=["Deutsche Bank CDS > 200bps", "ECB emergency liquidity",
                            "European bank index -20%+", "Interbank lending freeze"],
        probability=0.20, impact_severity=0.85,
        beneficiary_assets=[Asset.GOLD, Asset.TLT, Asset.GDX],
        victim_assets=[Asset.SPY, Asset.XLF, Asset.KRE, Asset.HYG, Asset.ETH],
        oil_sensitivity=-0.2,
    ),
    ScenarioDefinition(
        id=5, name="DeFi Cascade Collapse", code="DEFI_CASCADE",
        description="Major DeFi protocol failure triggers crypto-wide liquidation cascade",
        trigger_indicators=["Stablecoin depeg", "TVL collapse > 40%", "CEX withdrawal freeze",
                            "BTC -30% in 48hrs"],
        probability=0.25, impact_severity=0.70,
        beneficiary_assets=[Asset.GOLD, Asset.TLT],
        victim_assets=[Asset.BTC, Asset.ETH, Asset.XRP, Asset.BITO],
        oil_sensitivity=-0.1,
    ),
    ScenarioDefinition(
        id=6, name="Commodity Supercycle Peak", code="SUPERCYCLE",
        description="Gold > $5000, silver > $100, commodities enter blow-off top",
        trigger_indicators=["Gold > $4500", "Silver > $80", "DXY < 95",
                            "Central bank gold buying record"],
        probability=0.30, impact_severity=0.60,
        beneficiary_assets=[Asset.GOLD, Asset.SILVER, Asset.GDX, Asset.OIL, Asset.XLE,
                            Asset.SMR],
        victim_assets=[Asset.QQQ, Asset.XLY, Asset.BITO],
        oil_sensitivity=0.4,
    ),
    ScenarioDefinition(
        id=7, name="US Commercial Real Estate Collapse", code="CRE_COLLAPSE",
        description="Office vacancy > 25%, CMBS defaults cascade, regional banks exposed",
        trigger_indicators=["CMBS delinquency > 10%", "Office REIT -40%",
                            "Regional bank failures", "FDIC intervention"],
        probability=0.35, impact_severity=0.75,
        beneficiary_assets=[Asset.GOLD, Asset.TLT],
        victim_assets=[Asset.XLRE, Asset.KRE, Asset.XLF, Asset.HYG],
        oil_sensitivity=-0.1,
    ),
    ScenarioDefinition(
        id=8, name="AI Bubble Burst", code="AI_BUBBLE",
        description="AI hype cycle collapses as monetization fails — QQQ -30%+",
        trigger_indicators=["NVDA -40%", "AI startup failures", "Tech layoffs spike",
                            "QQQ breaks 200-day MA decisively"],
        probability=0.20, impact_severity=0.70,
        beneficiary_assets=[Asset.GOLD, Asset.TLT, Asset.XLE, Asset.TSLA],
        victim_assets=[Asset.QQQ, Asset.SPY, Asset.BITO],
        oil_sensitivity=-0.15,
    ),
    ScenarioDefinition(
        id=9, name="Emerging Market Currency Crisis", code="EM_FX_CRISIS",
        description="Dollar squeeze triggers EM debt defaults — contagion to DM",
        trigger_indicators=["DXY > 115", "EM sovereign defaults", "Capital flight acceleration",
                            "IMF emergency lending"],
        probability=0.15, impact_severity=0.65,
        beneficiary_assets=[Asset.GOLD, Asset.TLT],
        victim_assets=[Asset.SPY, Asset.HYG, Asset.ETH, Asset.XRP],
        oil_sensitivity=0.1,
    ),
    ScenarioDefinition(
        id=10, name="Global Food Crisis", code="FOOD_CRISIS",
        description="Multi-region crop failure — food inflation triggers social unrest",
        trigger_indicators=["Wheat/corn/rice prices +50%", "Export bans",
                            "UN food security alert", "Fertilizer shortage"],
        probability=0.15, impact_severity=0.60,
        beneficiary_assets=[Asset.GOLD, Asset.OIL],
        victim_assets=[Asset.SPY, Asset.XLY, Asset.JETS, Asset.ETH],
        oil_sensitivity=0.3,
    ),
    ScenarioDefinition(
        id=11, name="Climate Catastrophe Trigger", code="CLIMATE_SHOCK",
        description="Major climate event (ice sheet, gulf stream) causes rapid policy shift",
        trigger_indicators=["Extreme weather events cluster", "Insurance market crisis",
                            "Government emergency climate policy", "RE infrastructure demand spike"],
        probability=0.10, impact_severity=0.80,
        beneficiary_assets=[Asset.GOLD, Asset.XLE],
        victim_assets=[Asset.XLRE, Asset.JETS, Asset.XLY, Asset.SPY],
        oil_sensitivity=0.5,
    ),
    ScenarioDefinition(
        id=12, name="Monetary System Reset", code="MONETARY_RESET",
        description="BRICS gold-backed settlement, petrodollar displacement accelerates",
        trigger_indicators=["BRICS settlement system launch", "Saudi accepts non-USD oil payment",
                            "Central bank gold accumulation", "DXY < 90",
                            "mBridge volume surge"],
        probability=0.15, impact_severity=0.95,
        beneficiary_assets=[Asset.GOLD, Asset.SILVER, Asset.GDX, Asset.BTC, Asset.OIL,
                            Asset.XRP],
        victim_assets=[Asset.TLT, Asset.SPY, Asset.XLF, Asset.HYG],
        oil_sensitivity=0.3,
    ),
    ScenarioDefinition(
        id=13, name="Japanese Financial Crisis", code="JAPAN_CRISIS",
        description="Yen collapse triggers BOJ intervention, JGB sell-off, global carry trade unwind",
        trigger_indicators=["USD/JPY > 170", "JGB 10Y > 2%", "BOJ emergency meeting",
                            "Yen carry trade unwinding"],
        probability=0.20, impact_severity=0.80,
        beneficiary_assets=[Asset.GOLD, Asset.TLT],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.XLF, Asset.ETH, Asset.BTC],
        oil_sensitivity=-0.1,
    ),
    ScenarioDefinition(
        id=14, name="US Election Chaos", code="ELECTION_CHAOS",
        description="Contested election, constitutional crisis, policy paralysis",
        trigger_indicators=["Polling uncertainty extreme", "Legal challenges filed",
                            "Market volatility spike pre-election", "VIX > 35 pre-Nov"],
        probability=0.15, impact_severity=0.55,
        beneficiary_assets=[Asset.GOLD, Asset.TLT, Asset.BTC],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.XLF, Asset.KRE],
        oil_sensitivity=0.1,
    ),
    ScenarioDefinition(
        id=15, name="Pandemic Resurgence", code="PANDEMIC_V2",
        description="Novel pathogen or resistant variant triggers lockdowns and supply disruption",
        trigger_indicators=["WHO emergency declaration", "Hospital capacity warnings",
                            "Travel restrictions reimposed", "Pharma stockpiling"],
        probability=0.10, impact_severity=0.75,
        beneficiary_assets=[Asset.GOLD, Asset.TLT, Asset.QQQ],
        victim_assets=[Asset.JETS, Asset.XLY, Asset.XLRE, Asset.OIL, Asset.KRE],
        oil_sensitivity=-0.4,
    ),
    # ── Scenarios 16-20: Added from Storm Lifeboat thesis (March 2026) ──
    ScenarioDefinition(
        id=16, name="US Middle East Troop Withdrawal", code="US_WITHDRAWAL",
        description="Full or partial US troop withdrawal from Middle East — power vacuum, "
                    "regional realignment, Iran/China/Russia influence expansion",
        trigger_indicators=["US base closures announced", "Troop drawdown orders",
                            "Congressional war-fatigue resolutions",
                            "GCC bilateral defense deals with China/Russia"],
        probability=0.20, impact_severity=0.70,
        beneficiary_assets=[Asset.OIL, Asset.GOLD, Asset.GDX, Asset.XLE],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.XLF, Asset.JETS],
        oil_sensitivity=0.6,
    ),
    ScenarioDefinition(
        id=17, name="Iran Reparations / Grand Bargain", code="IRAN_DEAL",
        description="US agrees to large-scale reparations or sanctions relief as part of "
                    "a grand bargain to reopen Strait of Hormuz — massive fiscal impact",
        trigger_indicators=["Diplomatic back-channel leaks", "UN mediation talks",
                            "Oil-for-peace framework proposals",
                            "Congressional appropriations debate"],
        probability=0.05, impact_severity=0.65,
        beneficiary_assets=[Asset.SPY, Asset.JETS, Asset.XLY, Asset.HYG],
        victim_assets=[Asset.OIL, Asset.GOLD, Asset.GDX, Asset.XLE],
        oil_sensitivity=-0.7,
    ),
    ScenarioDefinition(
        id=18, name="Petrodollar Death Spiral", code="PETRODOLLAR_SPIRAL",
        description="Accelerating de-dollarization of oil trade — yuan/ruble settlement "
                    "gains critical mass, Gulf states diversify reserves, dollar index collapses",
        trigger_indicators=["Saudi accepts yuan for oil", "BRICS settlement volume surge",
                            "DXY < 95", "Central bank USD reserve share < 55%",
                            "mBridge volume doubling"],
        probability=0.25, impact_severity=0.95,
        beneficiary_assets=[Asset.GOLD, Asset.SILVER, Asset.GDX, Asset.OIL,
                            Asset.BTC, Asset.XRP],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.XLF, Asset.TLT, Asset.HYG],
        oil_sensitivity=0.5,
    ),
    ScenarioDefinition(
        id=19, name="Iran Nuclear Breakout + GCC Realignment", code="IRAN_NUCLEAR",
        description="Iran demonstrates nuclear capability or makes credible nuclear threat — "
                    "GCC states accept Iranian/Chinese protection umbrella, yuan oil enforcement",
        trigger_indicators=["IAEA enrichment alert >90%", "Underground test indicators",
                            "GCC emergency summit", "Yuan protection-for-passage offers",
                            "Israeli preemptive strike talk"],
        probability=0.15, impact_severity=0.98,
        beneficiary_assets=[Asset.GOLD, Asset.OIL, Asset.GDX, Asset.SILVER, Asset.XLE],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.XLF, Asset.JETS, Asset.KRE,
                       Asset.XLRE, Asset.HYG],
        oil_sensitivity=0.9,
    ),
    ScenarioDefinition(
        id=20, name="Elite Financial Network Exposure", code="ELITE_EXPOSURE",
        description="Ongoing release of Epstein-adjacent financial records and congressional "
                    "probes into bank facilitation erode institutional trust — accelerates "
                    "capital flight from traditional finance to hard assets and crypto",
        trigger_indicators=["New DOJ file tranche release", "Congressional subpoena of bank records",
                            "Major bank settlement announcement",
                            "Viral social media amplification >100k engagements",
                            "Institutional trust index decline"],
        probability=0.30, impact_severity=0.60,
        beneficiary_assets=[Asset.GOLD, Asset.BTC, Asset.XRP, Asset.ETH],
        victim_assets=[Asset.XLF, Asset.KRE, Asset.HYG, Asset.SPY],
        oil_sensitivity=0.0,
    ),
    # ══════════════════════════════════════════════════════════════════════
    # Scenarios 21-43: US Western Hemisphere Pivot & Fortress America
    # ══════════════════════════════════════════════════════════════════════
    ScenarioDefinition(
        id=21, name="US Pivot to Western Hemisphere", code="HEMISPHERE_PIVOT",
        description="Monroe Doctrine 2.0 — US consolidates economic and security control "
                    "over the Western Hemisphere, redirecting resources from Middle East/Europe",
        trigger_indicators=["Monroe Doctrine executive order", "Hemisphere trade deal signing",
                            "US-LATAM summit announcements", "OAS military framework reboot",
                            "Western Hemisphere energy corridor proposals"],
        probability=0.35, impact_severity=0.75,
        beneficiary_assets=[Asset.OIL, Asset.XLE, Asset.GOLD, Asset.TSLA, Asset.SMR],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.JETS, Asset.XLF],
        oil_sensitivity=0.3,
    ),
    ScenarioDefinition(
        id=22, name="Trump NATO Withdrawal", code="NATO_EXIT",
        description="US formally withdraws from or de-funds NATO — European defense vacuum, "
                    "dollar confidence shaken, gold/commodities spike",
        trigger_indicators=["NATO withdrawal announcement", "US troop repatriation orders",
                            "European defense spending surge", "NATO Article 5 doubt",
                            "Congressional NATO funding debate"],
        probability=0.25, impact_severity=0.85,
        beneficiary_assets=[Asset.GOLD, Asset.GDX, Asset.OIL, Asset.XLE, Asset.BTC],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.XLF, Asset.JETS, Asset.HYG],
        oil_sensitivity=0.4,
    ),
    ScenarioDefinition(
        id=23, name="US Abandonment of Europe", code="EUROPE_ABANDON",
        description="Full US strategic pivot away from Europe — Russian sphere consolidation, "
                    "European equities crash, flight to hard assets",
        trigger_indicators=["US ambassador recalls", "European defense pact without US",
                            "Russian influence expansion", "EU emergency summit",
                            "Transatlantic trade breakdown"],
        probability=0.20, impact_severity=0.80,
        beneficiary_assets=[Asset.GOLD, Asset.GDX, Asset.OIL, Asset.BTC, Asset.SILVER],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.XLF, Asset.JETS, Asset.HYG],
        oil_sensitivity=0.3,
    ),
    ScenarioDefinition(
        id=24, name="Canada Managed Decline", code="CANADA_DECLINE",
        description="US economic pressure on Canada — resource extraction leverage, "
                    "CAD weakness, energy corridor dominance",
        trigger_indicators=["US-Canada tariff escalation", "CAD/USD below 0.65",
                            "Canadian resource sector selloff", "Pipeline renegotiations",
                            "Cross-border trade restrictions"],
        probability=0.30, impact_severity=0.55,
        beneficiary_assets=[Asset.OIL, Asset.XLE, Asset.GOLD],
        victim_assets=[Asset.SPY, Asset.KRE, Asset.HYG],
        oil_sensitivity=0.2,
    ),
    ScenarioDefinition(
        id=25, name="Greenland Acquisition", code="GREENLAND_ACQ",
        description="US pursues purchase or strategic control of Greenland — Arctic resource "
                    "access, rare earth deposits, military positioning",
        trigger_indicators=["Greenland purchase offer", "Arctic military buildup",
                            "Rare earth extraction bids", "Danish sovereignty dispute",
                            "Arctic shipping route development"],
        probability=0.20, impact_severity=0.50,
        beneficiary_assets=[Asset.OIL, Asset.GOLD, Asset.GDX, Asset.XLE, Asset.SMR],
        victim_assets=[Asset.SPY, Asset.JETS],
        oil_sensitivity=0.15,
    ),
    ScenarioDefinition(
        id=26, name="Panama Canal Reclamation", code="PANAMA_RECLAIM",
        description="US exerts direct control or sovereignty over Panama Canal — trade "
                    "chokepoint leverage, China confrontation at canal",
        trigger_indicators=["Panama Canal sovereignty claims", "US military deployment Panama",
                            "Chinese shipping rerouting", "Canal toll restructuring",
                            "Central American security framework"],
        probability=0.15, impact_severity=0.60,
        beneficiary_assets=[Asset.OIL, Asset.XLE, Asset.GOLD],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.JETS, Asset.XLY],
        oil_sensitivity=0.25,
    ),
    ScenarioDefinition(
        id=27, name="Latin American Resource Lock-in", code="LATAM_LOCKIN",
        description="US secures exclusive resource agreements across Latin America — lithium,"
                    " copper, oil, agriculture locked into hemisphere trade bloc",
        trigger_indicators=["Bilateral resource treaties", "LATAM trade bloc formation",
                            "Chinese investment expulsion", "Critical mineral agreements",
                            "Agricultural export deals"],
        probability=0.30, impact_severity=0.55,
        beneficiary_assets=[Asset.OIL, Asset.XLE, Asset.GOLD, Asset.TSLA, Asset.SMR],
        victim_assets=[Asset.QQQ, Asset.XLY],
        oil_sensitivity=0.2,
    ),
    ScenarioDefinition(
        id=28, name="Arctic Command Expansion", code="ARCTIC_EXPAND",
        description="US expands Arctic military presence — new bases, icebreaker fleet, "
                    "resource claim enforcement along Northern Sea Route",
        trigger_indicators=["Arctic base construction", "Icebreaker fleet expansion",
                            "Arctic sovereignty claims", "Northern Sea Route patrols",
                            "Arctic resource extraction permits"],
        probability=0.20, impact_severity=0.45,
        beneficiary_assets=[Asset.OIL, Asset.GOLD, Asset.XLE, Asset.SMR],
        victim_assets=[Asset.SPY, Asset.JETS],
        oil_sensitivity=0.15,
    ),
    ScenarioDefinition(
        id=29, name="Southern Border Militarization", code="BORDER_MILITARY",
        description="Full militarization of US southern border — economic disruption, "
                    "labor market shock, trade friction with Mexico",
        trigger_indicators=["Border military deployment", "Mexico trade sanctions",
                            "Labor shortage indicators", "Border closure events",
                            "USMCA renegotiation threats"],
        probability=0.35, impact_severity=0.50,
        beneficiary_assets=[Asset.GOLD, Asset.OIL],
        victim_assets=[Asset.SPY, Asset.XLY, Asset.JETS, Asset.KRE],
        oil_sensitivity=0.1,
    ),
    ScenarioDefinition(
        id=30, name="Venezuela Regime Change", code="VENEZUELA_REGIME",
        description="US-backed regime change in Venezuela — restores oil output, "
                    "hemisphere supply chain reshoring of heavy crude",
        trigger_indicators=["Venezuelan opposition support", "US sanctions adjustment",
                            "Military intervention signals", "Oil sector restructuring",
                            "PDVSA privatization offers"],
        probability=0.15, impact_severity=0.55,
        beneficiary_assets=[Asset.OIL, Asset.XLE, Asset.SPY],
        victim_assets=[Asset.GOLD, Asset.BTC],
        oil_sensitivity=-0.3,
    ),
    ScenarioDefinition(
        id=31, name="Lithium Triangle Security", code="LITHIUM_TRIANGLE",
        description="US secures lithium supply from Bolivia-Argentina-Chile triangle — "
                    "EV supply chain fortress, Tesla strategic advantage",
        trigger_indicators=["Lithium bilateral agreements", "Mining security deployments",
                            "Chinese lithium company restrictions", "EV battery supply deals",
                            "Critical mineral stockpiling"],
        probability=0.25, impact_severity=0.45,
        beneficiary_assets=[Asset.TSLA, Asset.SMR, Asset.XLE],
        victim_assets=[Asset.QQQ],
        oil_sensitivity=0.05,
    ),
    ScenarioDefinition(
        id=32, name="Cuba Embargo Tightening", code="CUBA_EMBARGO",
        description="Intensified Cuba sanctions and naval blockade — hemisphere control "
                    "signal, Russian/Chinese base denial",
        trigger_indicators=["Cuba sanctions expansion", "Naval blockade enforcement",
                            "Russian base eviction demands", "Cuban humanitarian crisis",
                            "Caribbean security pact"],
        probability=0.20, impact_severity=0.35,
        beneficiary_assets=[Asset.GOLD, Asset.OIL],
        victim_assets=[Asset.JETS, Asset.XLY],
        oil_sensitivity=0.05,
    ),
    ScenarioDefinition(
        id=33, name="Brazil & Argentina Economic Leverage", code="BRAZIL_ARGENTINA",
        description="US economic pressure on South American giants — trade concessions, "
                    "agricultural leverage, currency manipulation countermeasures",
        trigger_indicators=["BRICS exit pressure", "Agricultural trade deals",
                            "Currency swap agreements", "Mercosur restructuring",
                            "IMF conditionality alignment"],
        probability=0.25, impact_severity=0.50,
        beneficiary_assets=[Asset.OIL, Asset.GOLD, Asset.XLE],
        victim_assets=[Asset.SPY, Asset.HYG, Asset.XLY],
        oil_sensitivity=0.1,
    ),
    ScenarioDefinition(
        id=34, name="Northern Border Resource Integration", code="NORTH_BORDER",
        description="US-Canada deep resource integration — energy corridor unification, "
                    "rare earth sharing, cross-border infrastructure",
        trigger_indicators=["US-Canada energy pact", "Pipeline fast-track approvals",
                            "Cross-border resource agreements", "Joint Arctic development",
                            "Integrated power grid proposals"],
        probability=0.25, impact_severity=0.40,
        beneficiary_assets=[Asset.OIL, Asset.XLE, Asset.SMR, Asset.TSLA],
        victim_assets=[Asset.QQQ],
        oil_sensitivity=0.15,
    ),
    ScenarioDefinition(
        id=35, name="Military Redeployment from Middle East", code="MIDEAST_REDEPLOY",
        description="US redeployment of military assets from Middle East to Western "
                    "Hemisphere — power vacuum in Gulf, hemisphere fortress buildup",
        trigger_indicators=["Base closure announcements", "Troop repatriation schedules",
                            "Hemisphere base expansion", "Gulf allies defense deals",
                            "CENTCOM restructuring"],
        probability=0.25, impact_severity=0.70,
        beneficiary_assets=[Asset.OIL, Asset.GOLD, Asset.GDX, Asset.XLE],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.JETS, Asset.XLF],
        oil_sensitivity=0.5,
    ),
    ScenarioDefinition(
        id=36, name="Energy Independence via Hemisphere", code="ENERGY_HEMISPHERE",
        description="US achieves full energy independence through hemisphere resources — "
                    "Canadian oil sands, Gulf of Mexico, Venezuela heavy crude, shale",
        trigger_indicators=["Energy independence declaration", "Hemisphere energy summit",
                            "Oil import restrictions from outside hemisphere",
                            "Refinery capacity expansion", "Strategic reserve policy shift"],
        probability=0.30, impact_severity=0.55,
        beneficiary_assets=[Asset.OIL, Asset.XLE, Asset.SMR, Asset.TSLA],
        victim_assets=[Asset.JETS, Asset.QQQ],
        oil_sensitivity=-0.2,
    ),
    ScenarioDefinition(
        id=37, name="SpaceX/Starlink Hemispheric Dominance", code="STARLINK_DOMINANCE",
        description="SpaceX/Starlink achieves communication dominance over Western Hemisphere"
                    " — digital infrastructure moat, military comms backbone",
        trigger_indicators=["Starlink government contracts", "Hemisphere-wide coverage",
                            "Military communication deals", "Competitor satellite failures",
                            "Internet sovereignty policies"],
        probability=0.30, impact_severity=0.45,
        beneficiary_assets=[Asset.TSLA, Asset.QQQ, Asset.SPY],
        victim_assets=[Asset.GOLD],
        oil_sensitivity=0.0,
    ),
    ScenarioDefinition(
        id=38, name="Fusion & Micro-Reactor Rollout", code="FUSION_ROLLOUT",
        description="Accelerated deployment of small modular reactors and fusion prototypes"
                    " — energy revolution, grid independence, commodity demand shift",
        trigger_indicators=["SMR deployment announcements", "Fusion breakthrough news",
                            "Grid modernization contracts", "Nuclear regulatory fast-track",
                            "Energy infrastructure investment surge"],
        probability=0.20, impact_severity=0.60,
        beneficiary_assets=[Asset.SMR, Asset.TSLA, Asset.GOLD, Asset.SILVER],
        victim_assets=[Asset.OIL, Asset.XLE],
        oil_sensitivity=-0.4,
    ),
    ScenarioDefinition(
        id=39, name="Rare Earth Supply Chain Fortress", code="RARE_EARTH_FORTRESS",
        description="US secures hemispheric rare earth supply chain — China dependency "
                    "eliminated, domestic/allied processing capacity",
        trigger_indicators=["Rare earth mine openings", "Processing plant construction",
                            "China export ban triggers", "Hemisphere mineral stockpile",
                            "Defense Production Act invocation"],
        probability=0.25, impact_severity=0.50,
        beneficiary_assets=[Asset.TSLA, Asset.SMR, Asset.GDX, Asset.XLE],
        victim_assets=[Asset.QQQ],
        oil_sensitivity=0.05,
    ),
    ScenarioDefinition(
        id=40, name="Migration Control as National Security", code="MIGRATION_SECURITY",
        description="Migration reframed as national security — labor market disruption, "
                    "border spending surge, social services strain",
        trigger_indicators=["National emergency declarations", "Border deployment surge",
                            "Immigration reform legislation", "Labor market tightening",
                            "Social spending increase"],
        probability=0.40, impact_severity=0.40,
        beneficiary_assets=[Asset.GOLD, Asset.OIL],
        victim_assets=[Asset.SPY, Asset.XLY, Asset.KRE, Asset.XLRE],
        oil_sensitivity=0.05,
    ),
    ScenarioDefinition(
        id=41, name="2100 Fortress State Consolidation", code="FORTRESS_2100",
        description="Long-term US fortress state emergence — self-sufficient hemisphere, "
                    "de-globalization accelerates, new world order crystallization",
        trigger_indicators=["Long-term infrastructure plans", "Autarky policy signals",
                            "Global trade volume decline", "Hemisphere GDP share growth",
                            "De-globalization research trending"],
        probability=0.15, impact_severity=0.85,
        beneficiary_assets=[Asset.GOLD, Asset.OIL, Asset.XLE, Asset.TSLA, Asset.SMR,
                            Asset.BTC],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.XLF, Asset.JETS, Asset.HYG],
        oil_sensitivity=0.3,
    ),
    ScenarioDefinition(
        id=42, name="Elite Capital Redirection", code="ELITE_CAPITAL",
        description="Institutional and elite capital flows redirect from global to "
                    "hemisphere assets — real estate, energy, commodities, crypto",
        trigger_indicators=["Institutional rebalancing filings", "Real estate foreign investment",
                            "Commodity fund inflows", "Crypto institutional adoption",
                            "Capital flight from Europe/Asia"],
        probability=0.25, impact_severity=0.50,
        beneficiary_assets=[Asset.GOLD, Asset.OIL, Asset.BTC, Asset.XLE, Asset.XLRE,
                            Asset.TSLA],
        victim_assets=[Asset.TLT, Asset.HYG],
        oil_sensitivity=0.1,
    ),
    ScenarioDefinition(
        id=43, name="Nuclear Umbrella Americas Only", code="NUCLEAR_AMERICAS",
        description="US nuclear deterrence explicitly limited to Americas — NATO allies "
                    "lose umbrella, triggers global arms race and safe haven surge",
        trigger_indicators=["Nuclear posture review changes", "Deterrence doctrine shift",
                            "European nuclear program acceleration",
                            "Arms race indicators", "Defense spending surge global"],
        probability=0.10, impact_severity=0.90,
        beneficiary_assets=[Asset.GOLD, Asset.GDX, Asset.SILVER, Asset.BTC, Asset.OIL],
        victim_assets=[Asset.SPY, Asset.QQQ, Asset.XLF, Asset.JETS, Asset.HYG,
                       Asset.TLT],
        oil_sensitivity=0.4,
    ),
]

SCENARIO_MAP: Dict[str, ScenarioDefinition] = {s.code: s for s in SCENARIOS}


# ═══════════════════════════════════════════════════════════════════════════
# ASSET PRICING — March 2026 baseline
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_PRICES: Dict[Asset, float] = {
    Asset.OIL: 115.0,
    Asset.GOLD: 4861.0,
    Asset.SILVER: 78.0,
    Asset.GDX: 95.0,
    Asset.SPY: 665.0,
    Asset.QQQ: 450.0,
    Asset.XLF: 35.0,
    Asset.XLRE: 22.0,
    Asset.KRE: 58.0,
    Asset.JETS: 18.0,
    Asset.XLY: 180.0,
    Asset.XLE: 92.0,
    Asset.TSLA: 280.0,
    Asset.SMR: 28.0,
    Asset.TLT: 88.0,
    Asset.HYG: 75.0,
    Asset.BTC: 68000.0,
    Asset.ETH: 3800.0,
    Asset.XRP: 2.50,
    Asset.BITO: 22.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# VOLATILITY & DRIFT PROFILES PER REGIME
# ═══════════════════════════════════════════════════════════════════════════

# Annualized volatilities by regime
REGIME_VOLATILITIES: Dict[VolRegime, Dict[Asset, float]] = {
    VolRegime.CALM: {
        Asset.OIL: 0.30, Asset.GOLD: 0.15, Asset.SILVER: 0.22, Asset.GDX: 0.30,
        Asset.SPY: 0.14, Asset.QQQ: 0.18, Asset.XLF: 0.18, Asset.XLRE: 0.20,
        Asset.KRE: 0.22, Asset.JETS: 0.25, Asset.XLY: 0.18, Asset.XLE: 0.25,
        Asset.TSLA: 0.45, Asset.SMR: 0.50,
        Asset.TLT: 0.15, Asset.HYG: 0.08, Asset.BTC: 0.60, Asset.ETH: 0.70,
        Asset.XRP: 0.80, Asset.BITO: 0.55,
    },
    VolRegime.ELEVATED: {
        Asset.OIL: 0.50, Asset.GOLD: 0.30, Asset.SILVER: 0.40, Asset.GDX: 0.50,
        Asset.SPY: 0.25, Asset.QQQ: 0.30, Asset.XLF: 0.30, Asset.XLRE: 0.32,
        Asset.KRE: 0.35, Asset.JETS: 0.38, Asset.XLY: 0.28, Asset.XLE: 0.35,
        Asset.TSLA: 0.60, Asset.SMR: 0.65,
        Asset.TLT: 0.20, Asset.HYG: 0.12, Asset.BTC: 0.80, Asset.ETH: 0.90,
        Asset.XRP: 1.00, Asset.BITO: 0.75,
    },
    VolRegime.CRISIS: {
        Asset.OIL: 0.90, Asset.GOLD: 0.72, Asset.SILVER: 0.82, Asset.GDX: 1.02,
        Asset.SPY: 0.48, Asset.QQQ: 0.52, Asset.XLF: 0.58, Asset.XLRE: 0.50,
        Asset.KRE: 0.60, Asset.JETS: 0.55, Asset.XLY: 0.45, Asset.XLE: 0.45,
        Asset.TSLA: 0.75, Asset.SMR: 0.80,
        Asset.TLT: 0.25, Asset.HYG: 0.18, Asset.BTC: 1.12, Asset.ETH: 1.20,
        Asset.XRP: 1.28, Asset.BITO: 1.05,
    },
    VolRegime.PANIC: {
        Asset.OIL: 1.20, Asset.GOLD: 0.95, Asset.SILVER: 1.10, Asset.GDX: 1.40,
        Asset.SPY: 0.75, Asset.QQQ: 0.85, Asset.XLF: 0.90, Asset.XLRE: 0.80,
        Asset.KRE: 0.95, Asset.JETS: 0.85, Asset.XLY: 0.70, Asset.XLE: 0.65,
        Asset.TSLA: 1.00, Asset.SMR: 1.10,
        Asset.TLT: 0.35, Asset.HYG: 0.30, Asset.BTC: 1.50, Asset.ETH: 1.60,
        Asset.XRP: 1.70, Asset.BITO: 1.40,
    },
}

# Crisis-regime annualized drifts (current default — Hormuz/debt stress era)
CRISIS_DRIFTS: Dict[Asset, float] = {
    Asset.OIL: 0.60, Asset.GOLD: 0.40, Asset.SILVER: 0.40, Asset.GDX: 0.55,
    Asset.SPY: -0.40, Asset.QQQ: -0.45, Asset.XLF: -0.40, Asset.XLRE: -0.20,
    Asset.KRE: -0.45, Asset.JETS: -0.50, Asset.XLY: -0.35, Asset.XLE: 0.30,
    Asset.TSLA: 0.15, Asset.SMR: 0.25,
    Asset.TLT: 0.10, Asset.HYG: -0.15,
    Asset.BTC: -0.50, Asset.ETH: -0.55, Asset.XRP: -0.45, Asset.BITO: -0.48,
}

# Calm-regime drifts (risk-on / bull market)
CALM_DRIFTS: Dict[Asset, float] = {
    Asset.OIL: 0.05, Asset.GOLD: 0.08, Asset.SILVER: 0.10, Asset.GDX: 0.12,
    Asset.SPY: 0.12, Asset.QQQ: 0.15, Asset.XLF: 0.10, Asset.XLRE: 0.08,
    Asset.KRE: 0.10, Asset.JETS: 0.12, Asset.XLY: 0.14, Asset.XLE: 0.08,
    Asset.TSLA: 0.25, Asset.SMR: 0.20,
    Asset.TLT: 0.03, Asset.HYG: 0.05,
    Asset.BTC: 0.30, Asset.ETH: 0.35, Asset.XRP: 0.25, Asset.BITO: 0.28,
}


# ═══════════════════════════════════════════════════════════════════════════
# CORRELATION MATRIX — 18×18 crisis-weighted
# ═══════════════════════════════════════════════════════════════════════════

# Asset order for correlation matrix
ASSET_ORDER: List[Asset] = list(Asset)

# Pairwise correlations (crisis regime)
_RAW_CORRELATIONS: Dict[Tuple[Asset, Asset], float] = {
    # Oil correlations
    (Asset.OIL, Asset.GOLD): 0.55, (Asset.OIL, Asset.SILVER): 0.50,
    (Asset.OIL, Asset.GDX): 0.60, (Asset.OIL, Asset.SPY): -0.30,
    (Asset.OIL, Asset.QQQ): -0.25, (Asset.OIL, Asset.XLF): -0.35,
    (Asset.OIL, Asset.XLRE): -0.40, (Asset.OIL, Asset.KRE): -0.38,
    (Asset.OIL, Asset.JETS): -0.55, (Asset.OIL, Asset.XLY): -0.30,
    (Asset.OIL, Asset.XLE): 0.80, (Asset.OIL, Asset.TLT): 0.05,
    (Asset.OIL, Asset.HYG): -0.25, (Asset.OIL, Asset.BTC): -0.40,
    (Asset.OIL, Asset.ETH): -0.35, (Asset.OIL, Asset.XRP): -0.30,
    (Asset.OIL, Asset.BITO): -0.38,
    # Gold correlations
    (Asset.GOLD, Asset.SILVER): 0.85, (Asset.GOLD, Asset.GDX): 0.70,
    (Asset.GOLD, Asset.SPY): -0.25, (Asset.GOLD, Asset.QQQ): -0.20,
    (Asset.GOLD, Asset.XLF): -0.30, (Asset.GOLD, Asset.XLRE): -0.35,
    (Asset.GOLD, Asset.KRE): -0.32, (Asset.GOLD, Asset.JETS): -0.20,
    (Asset.GOLD, Asset.XLY): -0.22, (Asset.GOLD, Asset.XLE): 0.35,
    (Asset.GOLD, Asset.TLT): 0.30, (Asset.GOLD, Asset.HYG): -0.20,
    (Asset.GOLD, Asset.BTC): -0.10, (Asset.GOLD, Asset.ETH): -0.15,
    (Asset.GOLD, Asset.XRP): -0.12, (Asset.GOLD, Asset.BITO): -0.10,
    # Silver
    (Asset.SILVER, Asset.GDX): 0.75, (Asset.SILVER, Asset.SPY): -0.20,
    (Asset.SILVER, Asset.QQQ): -0.15, (Asset.SILVER, Asset.XLF): -0.25,
    (Asset.SILVER, Asset.XLRE): -0.30, (Asset.SILVER, Asset.KRE): -0.28,
    (Asset.SILVER, Asset.JETS): -0.18, (Asset.SILVER, Asset.XLY): -0.20,
    (Asset.SILVER, Asset.XLE): 0.30, (Asset.SILVER, Asset.TLT): 0.20,
    (Asset.SILVER, Asset.HYG): -0.15, (Asset.SILVER, Asset.BTC): -0.10,
    (Asset.SILVER, Asset.ETH): -0.12, (Asset.SILVER, Asset.XRP): -0.10,
    (Asset.SILVER, Asset.BITO): -0.10,
    # GDX
    (Asset.GDX, Asset.SPY): -0.35, (Asset.GDX, Asset.QQQ): -0.30,
    (Asset.GDX, Asset.XLF): -0.40, (Asset.GDX, Asset.XLRE): -0.45,
    (Asset.GDX, Asset.KRE): -0.42, (Asset.GDX, Asset.JETS): -0.25,
    (Asset.GDX, Asset.XLY): -0.28, (Asset.GDX, Asset.XLE): 0.40,
    (Asset.GDX, Asset.TLT): 0.25, (Asset.GDX, Asset.HYG): -0.25,
    (Asset.GDX, Asset.BTC): -0.15, (Asset.GDX, Asset.ETH): -0.18,
    (Asset.GDX, Asset.XRP): -0.15, (Asset.GDX, Asset.BITO): -0.15,
    # SPY
    (Asset.SPY, Asset.QQQ): 0.92, (Asset.SPY, Asset.XLF): 0.75,
    (Asset.SPY, Asset.XLRE): 0.65, (Asset.SPY, Asset.KRE): 0.70,
    (Asset.SPY, Asset.JETS): 0.60, (Asset.SPY, Asset.XLY): 0.85,
    (Asset.SPY, Asset.XLE): -0.10, (Asset.SPY, Asset.TLT): -0.30,
    (Asset.SPY, Asset.HYG): 0.50, (Asset.SPY, Asset.BTC): 0.50,
    (Asset.SPY, Asset.ETH): 0.45, (Asset.SPY, Asset.XRP): 0.40,
    (Asset.SPY, Asset.BITO): 0.48,
    # QQQ
    (Asset.QQQ, Asset.XLF): 0.70, (Asset.QQQ, Asset.XLRE): 0.55,
    (Asset.QQQ, Asset.KRE): 0.62, (Asset.QQQ, Asset.JETS): 0.50,
    (Asset.QQQ, Asset.XLY): 0.80, (Asset.QQQ, Asset.XLE): -0.15,
    (Asset.QQQ, Asset.TLT): -0.35, (Asset.QQQ, Asset.HYG): 0.45,
    (Asset.QQQ, Asset.BTC): 0.55, (Asset.QQQ, Asset.ETH): 0.50,
    (Asset.QQQ, Asset.XRP): 0.45, (Asset.QQQ, Asset.BITO): 0.52,
    # XLF
    (Asset.XLF, Asset.XLRE): 0.60, (Asset.XLF, Asset.KRE): 0.82,
    (Asset.XLF, Asset.JETS): 0.45, (Asset.XLF, Asset.XLY): 0.55,
    (Asset.XLF, Asset.XLE): -0.05, (Asset.XLF, Asset.TLT): -0.25,
    (Asset.XLF, Asset.HYG): 0.55, (Asset.XLF, Asset.BTC): 0.35,
    (Asset.XLF, Asset.ETH): 0.30, (Asset.XLF, Asset.XRP): 0.28,
    (Asset.XLF, Asset.BITO): 0.33,
    # XLRE
    (Asset.XLRE, Asset.KRE): 0.72, (Asset.XLRE, Asset.JETS): 0.35,
    (Asset.XLRE, Asset.XLY): 0.50, (Asset.XLRE, Asset.XLE): -0.15,
    (Asset.XLRE, Asset.TLT): 0.10, (Asset.XLRE, Asset.HYG): 0.45,
    (Asset.XLRE, Asset.BTC): 0.30, (Asset.XLRE, Asset.ETH): 0.25,
    (Asset.XLRE, Asset.XRP): 0.22, (Asset.XLRE, Asset.BITO): 0.28,
    # KRE
    (Asset.KRE, Asset.JETS): 0.42, (Asset.KRE, Asset.XLY): 0.55,
    (Asset.KRE, Asset.XLE): -0.08, (Asset.KRE, Asset.TLT): -0.20,
    (Asset.KRE, Asset.HYG): 0.55, (Asset.KRE, Asset.BTC): 0.32,
    (Asset.KRE, Asset.ETH): 0.28, (Asset.KRE, Asset.XRP): 0.25,
    (Asset.KRE, Asset.BITO): 0.30,
    # JETS
    (Asset.JETS, Asset.XLY): 0.55, (Asset.JETS, Asset.XLE): -0.20,
    (Asset.JETS, Asset.TLT): -0.15, (Asset.JETS, Asset.HYG): 0.35,
    (Asset.JETS, Asset.BTC): 0.25, (Asset.JETS, Asset.ETH): 0.22,
    (Asset.JETS, Asset.XRP): 0.20, (Asset.JETS, Asset.BITO): 0.23,
    # XLY
    (Asset.XLY, Asset.XLE): -0.10, (Asset.XLY, Asset.TLT): -0.28,
    (Asset.XLY, Asset.HYG): 0.50, (Asset.XLY, Asset.BTC): 0.40,
    (Asset.XLY, Asset.ETH): 0.35, (Asset.XLY, Asset.XRP): 0.30,
    (Asset.XLY, Asset.BITO): 0.38,
    # XLE
    (Asset.XLE, Asset.TLT): 0.05, (Asset.XLE, Asset.HYG): -0.10,
    (Asset.XLE, Asset.BTC): -0.15, (Asset.XLE, Asset.ETH): -0.18,
    (Asset.XLE, Asset.XRP): -0.12, (Asset.XLE, Asset.BITO): -0.15,
    (Asset.XLE, Asset.TSLA): 0.10, (Asset.XLE, Asset.SMR): 0.25,
    # TSLA — low correlation to broad equities in crisis (moat thesis), positive to energy/SMR
    (Asset.TSLA, Asset.SMR): 0.35,
    (Asset.TSLA, Asset.SPY): 0.40, (Asset.TSLA, Asset.QQQ): 0.50,
    (Asset.TSLA, Asset.XLF): 0.15, (Asset.TSLA, Asset.XLRE): 0.10,
    (Asset.TSLA, Asset.KRE): 0.12, (Asset.TSLA, Asset.JETS): 0.20,
    (Asset.TSLA, Asset.XLY): 0.45, (Asset.TSLA, Asset.TLT): -0.15,
    (Asset.TSLA, Asset.HYG): 0.20, (Asset.TSLA, Asset.BTC): 0.40,
    (Asset.TSLA, Asset.ETH): 0.38, (Asset.TSLA, Asset.XRP): 0.30,
    (Asset.TSLA, Asset.BITO): 0.38, (Asset.TSLA, Asset.OIL): -0.10,
    (Asset.TSLA, Asset.GOLD): -0.05, (Asset.TSLA, Asset.SILVER): -0.05,
    (Asset.TSLA, Asset.GDX): -0.08,
    # SMR — nuclear/energy-adjacent, positive to commodities/energy, modest equity correlation
    (Asset.SMR, Asset.SPY): 0.20, (Asset.SMR, Asset.QQQ): 0.25,
    (Asset.SMR, Asset.XLF): 0.10, (Asset.SMR, Asset.XLRE): 0.08,
    (Asset.SMR, Asset.KRE): 0.10, (Asset.SMR, Asset.JETS): 0.12,
    (Asset.SMR, Asset.XLY): 0.15, (Asset.SMR, Asset.TLT): 0.05,
    (Asset.SMR, Asset.HYG): 0.08, (Asset.SMR, Asset.BTC): 0.10,
    (Asset.SMR, Asset.ETH): 0.08, (Asset.SMR, Asset.XRP): 0.06,
    (Asset.SMR, Asset.BITO): 0.10, (Asset.SMR, Asset.OIL): 0.30,
    (Asset.SMR, Asset.GOLD): 0.20, (Asset.SMR, Asset.SILVER): 0.18,
    (Asset.SMR, Asset.GDX): 0.22,
    # TLT
    (Asset.TLT, Asset.HYG): -0.10, (Asset.TLT, Asset.BTC): -0.20,
    (Asset.TLT, Asset.ETH): -0.22, (Asset.TLT, Asset.XRP): -0.18,
    (Asset.TLT, Asset.BITO): -0.20,
    # HYG
    (Asset.HYG, Asset.BTC): 0.35, (Asset.HYG, Asset.ETH): 0.30,
    (Asset.HYG, Asset.XRP): 0.28, (Asset.HYG, Asset.BITO): 0.33,
    # Crypto
    (Asset.BTC, Asset.ETH): 0.85, (Asset.BTC, Asset.XRP): 0.75,
    (Asset.BTC, Asset.BITO): 0.95,
    (Asset.ETH, Asset.XRP): 0.80, (Asset.ETH, Asset.BITO): 0.82,
    (Asset.XRP, Asset.BITO): 0.72,
}


def build_correlation_matrix() -> "np.ndarray":
    """Build symmetric 18x18 correlation matrix from pairwise dict."""
    import numpy as np
    n = len(ASSET_ORDER)
    corr = np.eye(n)
    for i, a in enumerate(ASSET_ORDER):
        for j, b in enumerate(ASSET_ORDER):
            if i == j:
                continue
            key = (a, b) if (a, b) in _RAW_CORRELATIONS else (b, a)
            if key in _RAW_CORRELATIONS:
                corr[i, j] = _RAW_CORRELATIONS[key]
    return corr


# ═══════════════════════════════════════════════════════════════════════════
# RESULT DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AssetForecast:
    """Monte Carlo forecast for a single asset."""
    asset: Asset
    current_price: float
    mean_price: float
    median_price: float
    pct_5: float
    pct_25: float
    pct_75: float
    pct_95: float
    expected_return_pct: float
    prob_down_10: float
    prob_down_20: float
    prob_down_30: float
    prob_up_10: float
    prob_up_20: float
    var_95: float           # 95% VaR over horizon
    cvar_95: float          # Conditional VaR
    n_paths: int = MC_DEFAULT_PATHS


@dataclass
class PortfolioForecast:
    """Full Monte Carlo portfolio forecast."""
    timestamp: datetime
    regime: VolRegime
    asset_forecasts: Dict[Asset, AssetForecast]
    portfolio_var_95: float
    portfolio_cvar_95: float
    weighted_return_pct: float
    mandate: MandateLevel
    active_scenarios: List[str]    # codes of active scenarios
    coherence_score: float         # 0-1, PlanckPhire coherence
    moon_phase: MoonPhase
    horizon_days: int
    n_simulations: int


@dataclass
class ScenarioState:
    """Tracked state of a single scenario at a point in time."""
    code: str
    status: ScenarioStatus
    probability: float
    indicators_firing: List[str]
    last_updated: datetime


@dataclass
class HelixBriefing:
    """Daily Helix News briefing output."""
    date: date
    headline: str
    regime: VolRegime
    mandate: MandateLevel
    moon_phase: MoonPhase
    coherence_score: float
    active_scenarios: List[ScenarioState]
    top_trades: List[str]         # Recommended actions
    portfolio_summary: str
    risk_alert: Optional[str]


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StormConfig:
    """Master configuration for the Storm Lifeboat engine."""
    n_simulations: int = MC_DEFAULT_PATHS
    horizon_days: int = MC_DEFAULT_HORIZON
    starting_capital_cad: float = STARTING_CAPITAL_CAD
    cad_to_usd: float = CAD_TO_USD
    risk_free_rate: float = RISK_FREE_RATE
    regime: VolRegime = VolRegime.CRISIS
    seed: Optional[int] = 42
    # Mandate thresholds
    aggressive_prob_threshold: float = 0.40   # P(SPY -10%) to go aggressive
    max_conviction_prob: float = 0.60         # P(SPY -15%) to go max conviction
    # Data output
    data_dir: str = "data/storm_lifeboat"
