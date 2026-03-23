"""
Rocket Ship — Core Data Structures & Constants
================================================
All shared types, enums, asset definitions, and configuration for the
post-petrodollar transition engine.

Asset universe spans:
- Crypto rails: XRP, Flare (FXRP), SOL, ETH, BTC
- New reserve assets: tokenized gold, gold-pegged stablecoins, BRICS Unit proxies
- Legacy safe havens retained during Life Boat phase
- DeFi/yield instruments and CBDC-bridge exposure

Phase system:
    LIFE_BOAT  →  Neutral survival (Moons 1-12, BTC/gold/stables/XRP)
    IGNITING   →  Trigger confirmed, deploying Rocket allocations
    ROCKET     →  Full yield deployment (Moons 13-39)
    ORBIT      →  Stabilized multi-country 2030 final state
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# Thesis inception — Life Boat activated
LIFEBOAT_INCEPTION: date = date(2026, 3, 22)

# Moon cycle timing (synodic month = 29.53 days for astronomical accuracy)
SYNODIC_MONTH_DAYS: float = 29.530589    # Real new-moon period
REGULARIZED_MOON_DAYS: int = 28          # Used for phi-window math

# Cycle boundaries (absolute moon count from inception)
LIFEBOAT_MOON_START: int = 1
LIFEBOAT_MOON_END: int = 12              # Life Boat runs moons 1-12
ROCKET_MOON_START: int = 13             # Default Rocket ignition (or earlier on Gulf trigger)
ROCKET_MOON_END: int = 39              # Rocket rebalance window closes ~June 2029
ORBIT_MOON_START: int = 40             # Full 2030 diversification state

# Indicator thresholds for phase transition
INDICATORS_REQUIRED_FOR_IGNITION: int = 10  # of 15 must be GREEN
INDICATORS_TOTAL: int = 15

# Financial constants
STARTING_CAPITAL_CAD: float = 45_120.0
CAD_TO_USD: float = 0.72
STARTING_CAPITAL_USD: float = STARTING_CAPITAL_CAD * CAD_TO_USD

# Target allocations (Rocket Ship)
ROCKET_ALLOC_XRP_FLARE: float = 0.25
ROCKET_ALLOC_SOLANA:    float = 0.20
ROCKET_ALLOC_ETHEREUM:  float = 0.15
ROCKET_ALLOC_BITCOIN:   float = 0.15
ROCKET_ALLOC_BRICS:     float = 0.15    # Unit proxies + gold-pegged stables
ROCKET_ALLOC_DEFI_TOOLS: float = 0.10  # Self-custody oracles + multi-chain wallets

# Target allocations (Life Boat)
LIFEBOAT_ALLOC_BTC_GOLD: float = 0.40
LIFEBOAT_ALLOC_STABLES:  float = 0.30
LIFEBOAT_ALLOC_XRP:      float = 0.20
LIFEBOAT_ALLOC_CASH:     float = 0.10


# ═══════════════════════════════════════════════════════════════════════════
# PHASE ENUM
# ═══════════════════════════════════════════════════════════════════════════

class SystemPhase(Enum):
    """Current macro-strategic phase of the thesis."""
    LIFE_BOAT  = "life_boat"   # Survival: BTC/gold/stables/XRP
    IGNITING   = "igniting"    # Gulf trigger confirmed, deploying
    ROCKET     = "rocket"      # Full Rocket allocation in yield
    ORBIT      = "orbit"       # 2030 stable state — multi-country diversified


class MoonPhase(Enum):
    """New-moon cycle phase for rebalance timing."""
    NEW    = "new"     # Days 1-7: reset, reassess thesis
    WAXING = "waxing"  # Days 8-14: accumulate — phi window day 10-11
    FULL   = "full"    # Days 15-21: execute max conviction — phi window day 17-18
    WANING = "waning"  # Days 22-28: hedge, trim, protect


class IndicatorStatus(Enum):
    """Traffic-light status for each of the 15 indicators."""
    GREEN  = "green"   # Confirms new system thesis
    YELLOW = "yellow"  # Inconclusive / neutral
    RED    = "red"     # Against thesis / warning


class TriggerStatus(Enum):
    """State of the Gulf Yuan Oil ignition trigger."""
    WATCHING   = "watching"   # No confirmed news yet
    EMERGING   = "emerging"   # Early signals (swaps/partial flows)
    CONFIRMED  = "confirmed"  # Official Saudi/UAE yuan oil via mBridge


class AssetTier(Enum):
    """Where an asset sits in the new financial system."""
    DIGITAL_GOLD   = "digital_gold"    # BTC — neutral reserve
    PHYSICAL_GOLD  = "physical_gold"   # Gold / gold-pegged stables (PAXG)
    FX_BRIDGE      = "fx_bridge"       # XRP — neutral cross-currency settlement
    DEFI_LAYER     = "defi_layer"      # Flare/FXRP, SOL DeFi — yield generation
    SMART_CONTRACT = "smart_contract"  # ETH — RWA tokenization backbone
    BRICS_EXPOSURE = "brics_exposure"  # BRICS Unit proxies, e-CNY rails, gold-DLT
    STABLECOIN     = "stablecoin"      # USDC/USDT/PAXG — daily utility + LP
    LEGACY_EQUITY  = "legacy_equity"   # Transitional equities during Life Boat


class GeoBase(Enum):
    """Physical base of operations locations."""
    PANAMA    = "panama"    # Primary — dollarized, #1 expat, Tocumen hub
    PARAGUAY  = "paraguay"  # Secondary — 0% territorial, cheap, fast residency
    UAE       = "uae"       # Phase 2 (2027+) — mBridge hub, 0% tax
    SINGAPORE = "singapore" # Phase 3 (2028+) — Asia yuan bridge
    MALAYSIA  = "malaysia"  # Phase 3 (2028+) — MM2H, territorial
    THAILAND  = "thailand"  # Phase 3 — mBridge participant, Elite/LTR visa
    SWITZERLAND = "switzerland"  # Phase 3 (2028+) — Zug, RWA tokenization


# ═══════════════════════════════════════════════════════════════════════════
# ASSET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

class Asset(Enum):
    """Full Rocket Ship asset universe."""
    # ── Crypto Rails (Core Thesis) ──
    BTC   = "BTC"   # Bitcoin — digital gold, ETFs, US Strategic Reserve
    ETH   = "ETH"   # Ethereum — smart-contract backbone, staking, L2/RWA
    XRP   = "XRP"   # Ripple — neutral FX bridge, ODL, ISO-20022, OCC charter
    FLR   = "FLR"   # Flare Network — XRPFi DeFi layer, oracles, FXRP minting
    FXRP  = "FXRP"  # Flare-wrapped XRP — lend/stake on Morpho/SparkDEX
    SOL   = "SOL"   # Solana — $650B+/mo stablecoin volume, RWA, payments
    WXRP  = "WXRP"  # wXRP on Solana (Hex Trust/LayerZero 1:1) — DeFi on SOL
    # ── Stablecoins & Gold-Pegged ──
    USDC  = "USDC"  # USD Coin — regulated, GENIUS Act aligned
    USDT  = "USDT"  # Tether — largest market cap ($187B+)
    PAXG  = "PAXG"  # PAX Gold — tokenized 1oz gold bar, gold-pegged stable
    # ── BRICS / New System Exposure ──
    BRICS_UNIT = "BRICS_UNIT"  # Wholesale digital settlement unit (40% gold + 60% basket)
    ECNY       = "ECNY"        # Digital yuan (e-CNY) via mBridge/compliant fintech
    # ── Physical / Legacy Havens (Life Boat) ──
    GOLD_PHYS  = "GOLD_PHYS"   # Physical gold (bullion/bars)
    SILVER     = "SILVER"      # Physical silver (secondary haven)
    CASH_USD   = "CASH_USD"    # USD cash / bank deposits (liquidity buffer)
    # ── DeFi Yield Positions ──
    MORPHO_FXRP = "MORPHO_FXRP"  # FXRP deposited in Morpho vault (lending yield)
    SOL_LP       = "SOL_LP"       # Solana DEX LP position (stables/WXRP)
    ETH_STAKE    = "ETH_STAKE"    # ETH validator staking (~3-5% APY)


# ═══════════════════════════════════════════════════════════════════════════
# ASSET METADATA
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AssetProfile:
    """Complete profile of each Rocket Ship asset."""
    asset: Asset
    tier: AssetTier
    name: str
    role_in_new_system: str
    rocket_allocation_pct: float       # Target % in Rocket phase
    lifeboat_allocation_pct: float     # Target % in Life Boat phase
    how_to_bridge: str                 # Practical bridging/yield action
    why_2100_proof: str                # Long-horizon thesis
    risk_notes: str
    custody: str                       # "self" | "exchange" | "bank" | "protocol"


ASSET_PROFILES: Dict[str, AssetProfile] = {
    Asset.XRP.value: AssetProfile(
        asset=Asset.XRP, tier=AssetTier.FX_BRIDGE,
        name="XRP / Ripple",
        role_in_new_system=(
            "Neutral instant FX bridge. Settles in seconds for <$0.01. Bridges any"
            " currency pair without pre-funding. RippleNet ODL live globally (Asia/ME/Africa/LatAm)."
            " ISO-20022 native. OCC national trust bank charter (conditional, Dec 2025)."
        ),
        rocket_allocation_pct=0.25,
        lifeboat_allocation_pct=0.20,
        how_to_bridge="Hold XRP in self-custody. Bridge native XRP → FXRP on Flare app.",
        why_2100_proof="ISO-20022 standard; banks & CBDCs already test ODL. Bridge to mBridge/Unit.",
        risk_notes="No official BRICS deal (speculation). Regulation risk. Custody critical.",
        custody="self",
    ),
    Asset.FLR.value: AssetProfile(
        asset=Asset.FLR, tier=AssetTier.DEFI_LAYER,
        name="Flare Network (FLR + FXRP)",
        role_in_new_system=(
            "Transforms XRP into programmable DeFi asset. FXRP = wrapped XRP for"
            " lending, borrowing, DEX pools, stablecoin issuance. Flare oracles feed"
            " real-world data into DeFi. TVL ~$200M (+400% YoY). 100M+ FXRP minted."
            " Institutional minting via Hex Trust."
        ),
        rocket_allocation_pct=0.25,  # Combined with XRP
        lifeboat_allocation_pct=0.0,
        how_to_bridge="Bridge XRP → FXRP on Flare native bridge → Morpho/SparkDEX for yield.",
        why_2100_proof="Unlocks XRPFi lending/perps/RWA. Multichain oracles. Targets 5B XRP ecosystem TVL.",
        risk_notes="Smart contract risk — use audited Morpho/Kinetic. Start with 10% of position.",
        custody="protocol",
    ),
    Asset.SOL.value: AssetProfile(
        asset=Asset.SOL, tier=AssetTier.DEFI_LAYER,
        name="Solana",
        role_in_new_system=(
            "High-TPS (thousands/sec) for Web3/DeFi apps, payments, NFTs. Cheap, fast"
            " consumer/institutional layer. $650B+/mo stablecoin volume — leads all chains."
            " JPM commercial paper tokenized live. USDC supply ~$15B on Solana."
        ),
        rocket_allocation_pct=0.20,
        lifeboat_allocation_pct=0.0,
        how_to_bridge="Bridge wXRP via Hex Trust/LayerZero → Solana LPs or lending protocols.",
        why_2100_proof="Scalability king. Institutional adoption (payments, RWAs). Volume leads Web3.",
        risk_notes="Network outages historically (mitigation: Firedancer upgrade). MEV/LP IL risk.",
        custody="self",
    ),
    Asset.ETH.value: AssetProfile(
        asset=Asset.ETH, tier=AssetTier.SMART_CONTRACT,
        name="Ethereum",
        role_in_new_system=(
            "Smart-contract backbone for complex DeFi, tokenized real-world assets (RWAs),"
            " and CBDC interoperability. Mature ecosystem. Staking ETFs (possible 2026+)."
            " Fusaka upgrade improves scalability. L2 activity dominates DeFi TVL."
        ),
        rocket_allocation_pct=0.15,
        lifeboat_allocation_pct=0.0,
        how_to_bridge="Stake ETH (native or via Lido/stETH). Deploy to audited L2 DeFi.",
        why_2100_proof="Upgrades ensure longevity. RWA tokenization backbone long-term.",
        risk_notes="Regulatory risk on staking classification. L2 bridge smart-contract risk.",
        custody="self",
    ),
    Asset.BTC.value: AssetProfile(
        asset=Asset.BTC, tier=AssetTier.DIGITAL_GOLD,
        name="Bitcoin",
        role_in_new_system=(
            "Ultimate neutral reserve — digital gold in multipolar world. US Strategic"
            " Bitcoin Reserve confirmed. ETFs: $100B+ AUM. Nation-state adoption growing."
            " Censorship-resistant store of value as fiat baskets fluctuate."
        ),
        rocket_allocation_pct=0.15,
        lifeboat_allocation_pct=0.40,  # Combined with physical gold
        how_to_bridge="Long-term cold storage. Add on dips during Life Boat. No yield — pure reserve.",
        why_2100_proof="Nation-state + ETF + reserve adoption = digital gold standard. No counterparty risk.",
        risk_notes="Volatility during transition. Custody/key mgmt critical. No yield in cold storage.",
        custody="self",
    ),
    Asset.PAXG.value: AssetProfile(
        asset=Asset.PAXG, tier=AssetTier.PHYSICAL_GOLD,
        name="PAX Gold / Gold-Pegged Stablecoins",
        role_in_new_system=(
            "BRICS Unit is 40% physical gold-backed — gold-pegged tokens are the"
            " retail/DeFi access layer to Unit exposure. PAXG = 1 oz fine gold on-chain."
            " Also: XAUT (Tether Gold). Growing demand as Unit/mBridge scales."
        ),
        rocket_allocation_pct=0.15,  # Part of BRICS/Unit allocation
        lifeboat_allocation_pct=0.40,  # Part of BTC+gold Life Boat
        how_to_bridge="Hold PAXG in self-custody or LP in gold-stable pools for yield.",
        why_2100_proof="Gold is 5,000yr store of value. Tokenized form = 2100-proof interoperability.",
        risk_notes="Custodian risk (Paxos). Verify audits quarterly. Not truly decentralized.",
        custody="self",
    ),
    Asset.USDC.value: AssetProfile(
        asset=Asset.USDC, tier=AssetTier.STABLECOIN,
        name="USDC / Regulated Stablecoins",
        role_in_new_system=(
            "Daily medium of exchange + yield in LP pools. Most regulated stable (GENIUS"
            " Act 2025, Circle compliance). On-ramp/off-ramp at Panama/Paraguay banks."
            " Bridge asset between fiat and DeFi yield strategies."
        ),
        rocket_allocation_pct=0.10,  # Within DeFi tools / LP
        lifeboat_allocation_pct=0.30,
        how_to_bridge="Park in money-market DeFi (Aave/Morpho). Use as LP pair on Solana/Flare.",
        why_2100_proof="Regulated stables become CBDC on-ramps. Utility survives any regulatory regime.",
        risk_notes="Circle/USDC de-peg risk (historical: SVB). Diversify across USDT/USDC/PAXG.",
        custody="exchange + self",
    ),
    Asset.BRICS_UNIT.value: AssetProfile(
        asset=Asset.BRICS_UNIT, tier=AssetTier.BRICS_EXPOSURE,
        name="BRICS Unit Proxies",
        role_in_new_system=(
            "Wholesale digital settlement instrument (not retail currency). 40% physical"
            " gold + 60% BRICS basket. Cardano-based prototype. Pilot: 100 units issued"
            " (~0.98g gold each, Oct 2025). 2026-2027 expansion phase. India pushing"
            " CBDC links at 2026 summit. Tokenized access via compliant fintech."
        ),
        rocket_allocation_pct=0.15,
        lifeboat_allocation_pct=0.0,
        how_to_bridge=(
            "Exposure via: gold-pegged stables (PAXG) + tokenized Unit products"
            " (via exchanges as they launch) + BRICS ETFs/commodity funds."
        ),
        why_2100_proof="The new global settlement layer. Gold anchor + multipolar currency basket.",
        risk_notes="Wholesale only. No retail access yet. Geopolitics can delay. Speculation.",
        custody="exchange",
    ),
    Asset.ECNY.value: AssetProfile(
        asset=Asset.ECNY, tier=AssetTier.BRICS_EXPOSURE,
        name="e-CNY / Digital Yuan",
        role_in_new_system=(
            "China's CBDC. 95% of mBridge volume. $2.4T+ domestic transactions."
            " Interest-bearing (Jan 1 2026 rollout) to boost international appeal."
            " Saudi/UAE using via mBridge for energy settlements. The backbone currency"
            " of the de-dollarized trade system."
        ),
        rocket_allocation_pct=0.05,  # Within BRICS allocation
        lifeboat_allocation_pct=0.0,
        how_to_bridge="Via compliant fintech (Alipay International, mBridge participating banks).",
        why_2100_proof="China = largest trading nation. e-CNY is the settlement medium for >30% world trade.",
        risk_notes="Geopolitical risk. Capital controls. Access limited for non-Chinese nationals.",
        custody="bank",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS (post-petrodollar transition events)
# ═══════════════════════════════════════════════════════════════════════════

class TransitionScenario(Enum):
    """Key scenarios in the post-petrodollar transition."""
    GULF_YUAN_OIL       = "gulf_yuan_oil"       # IGNITION: Saudi/UAE yuan oil via mBridge
    BRICS_UNIT_LAUNCH   = "brics_unit_launch"   # Full BRICS Unit operationalization
    MBRIDGE_SCALE       = "mbridge_scale"        # mBridge reaches $1T+ volume milestone
    DOLLAR_CRASH        = "dollar_crash"         # USD reserve share drops below 45%
    CRYPTO_REGULATION   = "crypto_regulation"    # US/EU favorable crypto framework
    PETRODOLLAR_PACT    = "petrodollar_pact"     # Saudi formally ends USD oil pacts
    RIPPLE_BRICS        = "ripple_brics"         # XRP/Ripple official BRICS integration
    DEFI_EXPLOSION      = "defi_explosion"       # Global stablecoin cap >$1T
    CBDC_INTEROP        = "cbdc_interop"         # Multiple CBDCs interoperate on public chains
    USD_SANCTIONS_FAIL  = "usd_sanctions_fail"   # Major sanctions bypass via new rails


@dataclass
class ScenarioOutcome:
    """What happens to the portfolio when a scenario fires."""
    scenario: TransitionScenario
    description: str
    probability_2027: float          # Estimated probability by 2027
    beneficiary_assets: List[Asset]  # Assets that rally
    lagging_assets: List[Asset]      # Assets left behind
    action_on_fire: str              # What to do when confirmed


SCENARIO_OUTCOMES: List[ScenarioOutcome] = [
    ScenarioOutcome(
        scenario=TransitionScenario.GULF_YUAN_OIL,
        description=(
            "Saudi Arabia or UAE confirms yuan-denominated crude oil settled via mBridge."
            " Iran Hormuz yuan mandate already live (Q1 2026). This is the IGNITION event"
            " for the Rocket Ship thesis — timeline jumps forward from Moon 13."
        ),
        probability_2027=0.65,
        beneficiary_assets=[Asset.XRP, Asset.FLR, Asset.SOL, Asset.PAXG,
                            Asset.BRICS_UNIT, Asset.GOLD_PHYS],
        lagging_assets=[Asset.CASH_USD],
        action_on_fire=(
            "IGNITE: Bridge XRP → FXRP immediately. Deploy yield tactics."
            " Begin Panama base setup. Add BRICS/Unit exposure."
        ),
    ),
    ScenarioOutcome(
        scenario=TransitionScenario.BRICS_UNIT_LAUNCH,
        description=(
            "BRICS Unit moves from pilot (100 units, Oct 2025) to full operational scale."
            " Becomes accepted wholesale settlement across BRICS+ members."
        ),
        probability_2027=0.50,
        beneficiary_assets=[Asset.PAXG, Asset.GOLD_PHYS, Asset.BRICS_UNIT,
                            Asset.XRP, Asset.ECNY],
        lagging_assets=[Asset.CASH_USD, Asset.USDT],
        action_on_fire="Maximize gold-pegged stable allocation. Add tokenized Unit products.",
    ),
    ScenarioOutcome(
        scenario=TransitionScenario.MBRIDGE_SCALE,
        description=(
            "mBridge processes $1 trillion in transactions (currently $55.5B)."
            " Energy and commodity settlements dominate new volume."
        ),
        probability_2027=0.40,
        beneficiary_assets=[Asset.XRP, Asset.ECNY, Asset.BRICS_UNIT, Asset.FLR],
        lagging_assets=[Asset.CASH_USD],
        action_on_fire="Scale UAE allocation. Bridge yield through mBridge-compatible fintech.",
    ),
    ScenarioOutcome(
        scenario=TransitionScenario.CRYPTO_REGULATION,
        description=(
            "US GENIUS Act fully enacted + MiCA (EU) creates clear, favorable framework."
            " Institutional floodgates open for DeFi yield strategies globally."
        ),
        probability_2027=0.70,
        beneficiary_assets=[Asset.BTC, Asset.ETH, Asset.SOL, Asset.USDC, Asset.FXRP],
        lagging_assets=[Asset.USDT],   # Tether compliance pressure
        action_on_fire="Scale ETH staking + Solana LPs. Add regulated DeFi yield products.",
    ),
    ScenarioOutcome(
        scenario=TransitionScenario.DEFI_EXPLOSION,
        description=(
            "Global stablecoin market cap crosses $1T (currently ~$300B). DeFi TVL"
            " surpasses $500B. Mainstream adoption of yield strategies accelerates."
        ),
        probability_2027=0.45,
        beneficiary_assets=[Asset.SOL, Asset.ETH, Asset.FLR, Asset.FXRP,
                            Asset.USDC, Asset.MORPHO_FXRP, Asset.SOL_LP],
        lagging_assets=[Asset.GOLD_PHYS],
        action_on_fire="Scale DeFi yield positions. Maximize Morpho/SparkDEX deployment.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# SHARED DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioSnapshot:
    """Current state of the Rocket Ship portfolio."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    phase: SystemPhase = SystemPhase.LIFE_BOAT
    total_usd: float = 0.0
    total_cad: float = 0.0
    allocations: Dict[str, float] = field(default_factory=dict)   # asset -> USD value
    green_indicators: int = 0
    active_moon: int = 1                  # Current absolute moon number
    days_to_ignition_default: int = 0    # Days to Moon 13 new moon
    trigger_status: str = TriggerStatus.WATCHING.value
    notes: str = ""


@dataclass
class RocketConfig:
    """Runtime configuration for the Rocket Ship engine."""
    starting_capital_cad: float = STARTING_CAPITAL_CAD
    cad_to_usd: float = CAD_TO_USD
    lifeboat_inception: date = field(default_factory=lambda: LIFEBOAT_INCEPTION)
    indicators_required: int = INDICATORS_REQUIRED_FOR_IGNITION
    verbose: bool = True
    save_state: bool = True
    state_file: str = "rocket_ship_state.json"
