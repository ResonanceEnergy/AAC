"""
DeFi Yield Analyzer — BARREN WUFFET v2.7.0
============================================
Comprehensive DeFi yield farming analysis, impermanent loss
calculation, protocol risk scoring, and sustainable yield detection.

From BARREN WUFFET Insights (641-685):
  - APY >500% usually means dilution via inflationary token emissions
  - Real yield = fees earned / TVL, without token emissions
  - Impermanent loss at 2x price change = 5.7%, at 5x = 25.5%
  - Concentrated liquidity (Uniswap V3) amplifies both fees AND IL
  - Protocol revenue > token emissions = sustainable yield
  - TVL decline + high APY = yield farming exodus (rug risk)
  - Auto-compounding adds 10-30% annually to base yield
  - Stablecoin farming has IL risk from depegs (UST=99% loss)
  - Delta-neutral strategies reduce IL but cap upside
  - Yield layering (base + bribe + boost) creates complexity risk
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging
import math
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class YieldSource(Enum):
    TRADING_FEES = "trading_fees"         # DEX swap fees
    LENDING_INTEREST = "lending_interest"   # Borrow/lend interest
    TOKEN_EMISSIONS = "token_emissions"     # Inflationary rewards
    STAKING_REWARDS = "staking_rewards"     # PoS staking yield
    BRIBES = "bribes"                       # Vote-escrowed bribes
    POINTS = "points"                       # Point systems (pre-token)
    REAL_YIELD = "real_yield"              # Protocol revenue distribution
    LIQUIDATION_FEES = "liquidation_fees"   # Keeper fees

class ProtocolRisk(Enum):
    MINIMAL = "minimal"       # Top-tier, audited, battle-tested
    LOW = "low"               # Multiple audits, >$1B TVL, 1yr+ track record
    MEDIUM = "medium"         # Audited, moderate TVL, <1yr
    HIGH = "high"             # New, single audit, low TVL
    EXTREME = "extreme"       # Unaudited, unknown team, suspicious patterns

class YieldSustainability(Enum):
    SUSTAINABLE = "sustainable"     # Fees > emissions
    MOSTLY = "mostly_sustainable"   # Fees cover >50% of yield
    MIXED = "mixed"                 # Fees cover 20-50%
    UNSUSTAINABLE = "unsustainable" # <20% from fees, mostly emissions
    PONZI_RISK = "ponzi_risk"       # Pure emissions, no revenue


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ImpermanentLossResult:
    """Result of impermanent loss calculation."""
    price_ratio: float       # Final price / initial price
    il_percentage: float     # IL as percentage (always negative)
    lp_value: float          # Value if held in LP
    hodl_value: float        # Value if just held tokens
    loss_usd: float          # Dollar loss from IL
    notes: List[str] = field(default_factory=list)


@dataclass
class YieldBreakdown:
    """Detailed breakdown of yield sources."""
    protocol: str
    pool: str
    total_apy: float
    fee_apy: float           # From trading fees
    emission_apy: float      # From token rewards
    bribe_apy: float         # From bribes
    boost_apy: float         # From veToken boosts
    real_yield_apy: float    # Fees + protocol revenue only
    sustainability: YieldSustainability
    notes: List[str] = field(default_factory=list)


@dataclass
class ProtocolRiskAssessment:
    """Risk assessment for a DeFi protocol."""
    protocol: str
    chain: str
    risk_level: ProtocolRisk
    risk_score: float  # 0-100 (higher = riskier)
    tvl_usd: float
    audit_count: int
    age_months: int
    has_bug_bounty: bool
    is_upgradeable: bool
    admin_keys: str  # "multisig", "dao", "eoa", "renounced"
    risk_factors: List[str]
    safety_factors: List[str]


@dataclass
class FarmingOpportunity:
    """Evaluated farming opportunity."""
    protocol: str
    chain: str
    pool: str
    tvl: float
    yield_breakdown: YieldBreakdown
    protocol_risk: ProtocolRisk
    il_risk: str  # "none", "low", "moderate", "high", "extreme"
    net_expected_apy: float  # After risk adjustment
    conviction: float  # 0-100
    entry_size_pct: float  # Recommended % of portfolio
    notes: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# IMPERMANENT LOSS CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════

class ImpermanentLossCalculator:
    """
    Calculate impermanent loss for various LP configurations.
    
    IL Formula (50/50 pool):
      IL = 2*sqrt(r) / (1+r) - 1
      where r = price_final / price_initial
    
    IL at key ratios:
      1.25x =  0.6% loss
      1.50x =  2.0% loss
      2.00x =  5.7% loss
      3.00x = 13.4% loss
      5.00x = 25.5% loss
      10.0x = 42.5% loss
    
    Concentrated liquidity (CLMMs) AMPLIFY IL by the concentration factor!
    """

    @classmethod
    def calculate_standard(
        cls, initial_price: float, final_price: float,
        initial_deposit: float,
    ) -> ImpermanentLossResult:
        """Calculate IL for a standard 50/50 AMM pool."""
        r = final_price / initial_price if initial_price > 0 else 1

        # IL formula: 2*sqrt(r)/(1+r) - 1
        il_factor = 2 * math.sqrt(r) / (1 + r) - 1
        il_pct = il_factor * 100

        hodl_value = initial_deposit * (1 + r) / 2  # 50% in each asset
        lp_value = initial_deposit * (1 + il_factor) * (1 + r) / 2
        # Simplified: lp_value ≈ initial_deposit * sqrt(r)
        lp_value_precise = initial_deposit * math.sqrt(r)
        hodl_value_precise = initial_deposit * (1 + r) / 2

        loss_usd = hodl_value_precise - lp_value_precise

        notes = []
        if abs(il_pct) < 1:
            notes.append(f"IL is minimal ({il_pct:.2f}%). Price stayed close to entry.")
        elif abs(il_pct) < 5:
            notes.append(f"Moderate IL ({il_pct:.2f}%). Fees should cover this in most pools.")
        elif abs(il_pct) < 15:
            notes.append(f"Significant IL ({il_pct:.2f}%). Need high fee APY to compensate.")
        else:
            notes.append(f"SEVERE IL ({il_pct:.2f}%). LP is substantially underperforming HODL.")

        return ImpermanentLossResult(
            price_ratio=round(r, 4),
            il_percentage=round(il_pct, 3),
            lp_value=round(lp_value_precise, 2),
            hodl_value=round(hodl_value_precise, 2),
            loss_usd=round(loss_usd, 2),
            notes=notes,
        )

    @classmethod
    def calculate_concentrated(
        cls, initial_price: float, final_price: float,
        initial_deposit: float,
        lower_tick: float, upper_tick: float,
    ) -> ImpermanentLossResult:
        """
        Calculate IL for concentrated liquidity (Uniswap V3 style).
        
        Concentrated IL is amplified by the concentration factor:
          concentration = sqrt(upper/lower) / (sqrt(upper/lower) - 1)
        """
        # Standard IL first
        standard = cls.calculate_standard(initial_price, final_price, initial_deposit)

        # Concentration factor
        if upper_tick <= lower_tick:
            return standard

        tick_ratio = upper_tick / lower_tick
        concentration = math.sqrt(tick_ratio) / (math.sqrt(tick_ratio) - 1)

        # Amplified IL
        amplified_il = standard.il_percentage * concentration

        # Check if price went out of range
        out_of_range = final_price < lower_tick or final_price > upper_tick

        notes = list(standard.notes)
        notes.append(f"Concentration factor: {concentration:.1f}x")
        notes.append(f"Amplified IL: {amplified_il:.2f}% (vs {standard.il_percentage:.2f}% standard)")

        if out_of_range:
            notes.append("⚠️ Price moved OUT OF RANGE — full IL realized, zero fee income!")

        return ImpermanentLossResult(
            price_ratio=standard.price_ratio,
            il_percentage=round(amplified_il, 3),
            lp_value=round(standard.lp_value * (1 + (amplified_il - standard.il_percentage) / 100), 2),
            hodl_value=standard.hodl_value,
            loss_usd=round(standard.loss_usd * concentration, 2),
            notes=notes,
        )

    @classmethod
    def breakeven_fee_apy(
        cls, il_percentage: float, time_in_pool_days: int,
    ) -> float:
        """Calculate fee APY needed to break even on IL."""
        if time_in_pool_days <= 0:
            return float("inf")
        annualized_il = abs(il_percentage) * 365 / time_in_pool_days
        return round(annualized_il, 2)


# ═══════════════════════════════════════════════════════════════════════════
# YIELD SUSTAINABILITY ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class YieldSustainabilityAnalyzer:
    """
    Determine if yield is sustainable or inflationary ponzinomics.
    
    Red flags for unsustainable yield:
      1. APY >500% with no clear revenue source
      2. Token emissions > protocol revenue
      3. TVL declining while APY increasing (desperate retention)
      4. Token price declining despite high emissions
      5. No veToken / governance mechanism to align incentives
      6. Anonymous team with no track record
    
    Signs of sustainable yield:
      1. Revenue from actual usage (trading fees, lending interest)
      2. Growing TVL with stable/declining APY (organic growth)
      3. Token buyback from revenue (fee switch)
      4. Real yield > 50% of total yield
    """

    @classmethod
    def analyze(
        cls,
        protocol: str, pool: str,
        total_apy: float,
        fee_apy: float,
        emission_apy: float,
        bribe_apy: float = 0,
        boost_apy: float = 0,
        tvl_30d_change_pct: float = 0,
        emission_token_price_30d_change: float = 0,
    ) -> YieldBreakdown:
        """Analyze yield sustainability."""
        real_yield = fee_apy  # Only fees count as real
        notes = []

        # Sustainability classification
        if total_apy <= 0:
            sustainability = YieldSustainability.UNSUSTAINABLE
            notes.append("No yield detected.")
        elif fee_apy / total_apy > 0.80 if total_apy > 0 else False:
            sustainability = YieldSustainability.SUSTAINABLE
            notes.append(f"✅ {fee_apy/total_apy*100:.0f}% of yield from fees — genuinely sustainable.")
        elif fee_apy / total_apy > 0.50 if total_apy > 0 else False:
            sustainability = YieldSustainability.MOSTLY
            notes.append(f"Mostly sustainable: {fee_apy/total_apy*100:.0f}% from fees.")
        elif fee_apy / total_apy > 0.20 if total_apy > 0 else False:
            sustainability = YieldSustainability.MIXED
            notes.append(f"Mixed: only {fee_apy/total_apy*100:.0f}% from fees. Dependent on emissions.")
        elif emission_apy > 500:
            sustainability = YieldSustainability.PONZI_RISK
            notes.append(f"⚠️ {emission_apy:.0f}% emission APY with negligible fee yield — PONZI RISK!")
        else:
            sustainability = YieldSustainability.UNSUSTAINABLE
            notes.append("Yield primarily from token emissions — unsustainable long-term.")

        # Red flag checks
        if total_apy > 500 and fee_apy < 50:
            notes.append(f"🚩 APY {total_apy:.0f}% but only {fee_apy:.1f}% from fees!")

        if tvl_30d_change_pct < -30 and emission_apy > 100:
            notes.append("🚩 TVL dropping sharply despite high emissions — yield farming exodus!")

        if emission_token_price_30d_change < -50:
            notes.append("🚩 Emission token down >50% in 30d — yield is effectively halved!")

        return YieldBreakdown(
            protocol=protocol,
            pool=pool,
            total_apy=round(total_apy, 2),
            fee_apy=round(fee_apy, 2),
            emission_apy=round(emission_apy, 2),
            bribe_apy=round(bribe_apy, 2),
            boost_apy=round(boost_apy, 2),
            real_yield_apy=round(real_yield, 2),
            sustainability=sustainability,
            notes=notes,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOL RISK SCORER
# ═══════════════════════════════════════════════════════════════════════════

class ProtocolRiskScorer:
    """
    Score DeFi protocols on safety dimensions.
    
    Risk areas:
      - Smart contract risk (audits, track record, complexity)
      - Admin/governance risk (upgradeable, multisig, EOA control)
      - Economic risk (tokenomics, TVL sustainability)
      - Oracle risk (dependency on price feeds)
      - Bridge risk (cross-chain dependencies)
    """

    @classmethod
    def score(
        cls,
        protocol: str, chain: str,
        tvl_usd: float, audit_count: int, age_months: int,
        has_bug_bounty: bool, is_upgradeable: bool,
        admin_keys: str,  # "multisig", "dao", "eoa", "renounced"
        has_oracle_dependency: bool = False,
        has_bridge_dependency: bool = False,
        had_exploit: bool = False,
    ) -> ProtocolRiskAssessment:
        """Score protocol risk from 0 (safe) to 100 (dangerous)."""
        risk_score = 0
        risk_factors = []
        safety_factors = []

        # TVL factor (higher TVL = more skin in the game / battle-tested)
        if tvl_usd > 1_000_000_000:
            safety_factors.append(f"TVL ${tvl_usd/1e9:.1f}B — top-tier size")
        elif tvl_usd > 100_000_000:
            safety_factors.append(f"TVL ${tvl_usd/1e6:.0f}M — substantial")
        elif tvl_usd > 10_000_000:
            risk_score += 10
            risk_factors.append(f"TVL ${tvl_usd/1e6:.0f}M — moderate")
        else:
            risk_score += 25
            risk_factors.append(f"TVL ${tvl_usd/1e6:.1f}M — LOW, liquidity risk")

        # Audit factor
        if audit_count >= 3:
            safety_factors.append(f"{audit_count} audits completed")
        elif audit_count >= 1:
            risk_score += 10
            risk_factors.append(f"Only {audit_count} audit(s)")
        else:
            risk_score += 30
            risk_factors.append("NO AUDITS — high smart contract risk")

        # Age factor
        if age_months >= 24:
            safety_factors.append(f"{age_months} months live — battle-tested")
        elif age_months >= 12:
            risk_score += 5
        elif age_months >= 6:
            risk_score += 15
            risk_factors.append(f"Only {age_months} months old")
        else:
            risk_score += 25
            risk_factors.append(f"Very new ({age_months} months) — high unknown risk")

        # Bug bounty
        if has_bug_bounty:
            safety_factors.append("Active bug bounty program")
        else:
            risk_score += 5

        # Admin keys
        if admin_keys == "renounced":
            safety_factors.append("Admin keys renounced — immutable")
        elif admin_keys == "dao":
            safety_factors.append("DAO-governed — decentralized control")
        elif admin_keys == "multisig":
            risk_score += 10
            risk_factors.append("Multisig admin — centralization risk")
        elif admin_keys == "eoa":
            risk_score += 25
            risk_factors.append("EOA admin keys — SINGLE POINT OF FAILURE")

        # Upgradeable
        if is_upgradeable:
            risk_score += 10
            risk_factors.append("Upgradeable contracts — rug risk if admin compromised")

        # Oracle
        if has_oracle_dependency:
            risk_score += 5
            risk_factors.append("Oracle dependency — manipulation/failure risk")

        # Bridge
        if has_bridge_dependency:
            risk_score += 10
            risk_factors.append("Cross-chain bridge dependency — bridge hack risk")

        # Exploit history
        if had_exploit:
            risk_score += 15
            risk_factors.append("Previous exploit on record")

        risk_score = min(100, risk_score)

        # Risk level classification
        if risk_score <= 15:
            risk_level = ProtocolRisk.MINIMAL
        elif risk_score <= 30:
            risk_level = ProtocolRisk.LOW
        elif risk_score <= 50:
            risk_level = ProtocolRisk.MEDIUM
        elif risk_score <= 70:
            risk_level = ProtocolRisk.HIGH
        else:
            risk_level = ProtocolRisk.EXTREME

        return ProtocolRiskAssessment(
            protocol=protocol,
            chain=chain,
            risk_level=risk_level,
            risk_score=risk_score,
            tvl_usd=tvl_usd,
            audit_count=audit_count,
            age_months=age_months,
            has_bug_bounty=has_bug_bounty,
            is_upgradeable=is_upgradeable,
            admin_keys=admin_keys,
            risk_factors=risk_factors,
            safety_factors=safety_factors,
        )


# ═══════════════════════════════════════════════════════════════════════════
# OPPORTUNITY SCREENER
# ═══════════════════════════════════════════════════════════════════════════

class DeFiOpportunityScreener:
    """
    Screen and rank DeFi farming opportunities.
    
    Ranking factors:
      1. Risk-adjusted yield (higher = better)
      2. Sustainability score (sustainable > emissions)
      3. Protocol safety (lower risk = better)
      4. IL exposure (lower = better)
      5. TVL stability (growing = better)
    """

    IL_RISK_MAP = {
        "stablecoin_pair": "none",
        "correlated_pair": "low",        # ETH/stETH, BTC/WBTC
        "major_pair": "moderate",        # ETH/USDC
        "volatile_pair": "high",         # ALT/ETH
        "exotic_pair": "extreme",        # SHIB/DOGE
    }

    @classmethod
    def risk_adjust_yield(
        cls, total_apy: float, real_yield_apy: float,
        protocol_risk: ProtocolRisk, il_risk: str,
    ) -> float:
        """Apply risk haircut to yield."""
        risk_haircuts = {
            ProtocolRisk.MINIMAL: 0.95,
            ProtocolRisk.LOW: 0.85,
            ProtocolRisk.MEDIUM: 0.65,
            ProtocolRisk.HIGH: 0.40,
            ProtocolRisk.EXTREME: 0.15,
        }

        il_haircuts = {
            "none": 1.0,
            "low": 0.95,
            "moderate": 0.85,
            "high": 0.70,
            "extreme": 0.50,
        }

        # Use real yield as base, add portion of emission yield
        base = real_yield_apy + (total_apy - real_yield_apy) * 0.3  # Discount emissions 70%
        adjusted = base * risk_haircuts.get(protocol_risk, 0.5)
        adjusted *= il_haircuts.get(il_risk, 0.7)

        return round(adjusted, 2)

    @classmethod
    def recommended_allocation(
        cls, risk_adjusted_apy: float, protocol_risk: ProtocolRisk,
    ) -> float:
        """Recommend allocation as % of DeFi portfolio."""
        max_alloc = {
            ProtocolRisk.MINIMAL: 25.0,
            ProtocolRisk.LOW: 15.0,
            ProtocolRisk.MEDIUM: 8.0,
            ProtocolRisk.HIGH: 3.0,
            ProtocolRisk.EXTREME: 0.0,
        }
        cap = max_alloc.get(protocol_risk, 5.0)
        # Scale by yield attractiveness
        alloc = min(cap, risk_adjusted_apy / 5)
        return round(alloc, 1)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🐺 BARREN WUFFET — DeFi Yield Analyzer v2.7.0")
    print("=" * 55)

    # Impermanent Loss: standard
    il = ImpermanentLossCalculator.calculate_standard(
        initial_price=2000, final_price=4000, initial_deposit=10000,
    )
    print(f"\nImpermanent Loss (2x price move):")
    print(f"  IL: {il.il_percentage:.2f}%")
    print(f"  LP Value: ${il.lp_value:,.2f}")
    print(f"  HODL Value: ${il.hodl_value:,.2f}")
    print(f"  Loss: ${il.loss_usd:,.2f}")

    # Concentrated IL
    il_conc = ImpermanentLossCalculator.calculate_concentrated(
        initial_price=2000, final_price=4000, initial_deposit=10000,
        lower_tick=1500, upper_tick=3000,
    )
    print(f"\nConcentrated IL (same move, tight range):")
    print(f"  IL: {il_conc.il_percentage:.2f}%")
    for n in il_conc.notes: print(f"  → {n}")

    # Yield Sustainability
    yield_info = YieldSustainabilityAnalyzer.analyze(
        protocol="Aave V3", pool="USDC Lending",
        total_apy=4.5, fee_apy=4.2, emission_apy=0.3,
    )
    print(f"\nYield Analysis: {yield_info.protocol} — {yield_info.pool}")
    print(f"  Total APY: {yield_info.total_apy}% | Real: {yield_info.real_yield_apy}%")
    print(f"  Sustainability: {yield_info.sustainability.value}")
    for n in yield_info.notes: print(f"  → {n}")

    # Ponzi yield
    ponzi = YieldSustainabilityAnalyzer.analyze(
        protocol="ShadyFarm", pool="SHADY/USDC",
        total_apy=2000, fee_apy=5, emission_apy=1995,
        tvl_30d_change_pct=-45, emission_token_price_30d_change=-70,
    )
    print(f"\n⚠️  {ponzi.protocol} — {ponzi.pool}:")
    print(f"  Total APY: {ponzi.total_apy}% | Real: {ponzi.real_yield_apy}%")
    print(f"  Sustainability: {ponzi.sustainability.value}")
    for n in ponzi.notes: print(f"  → {n}")

    # Protocol Risk
    risk = ProtocolRiskScorer.score(
        protocol="Uniswap V3", chain="ethereum",
        tvl_usd=5_000_000_000, audit_count=5, age_months=36,
        has_bug_bounty=True, is_upgradeable=False, admin_keys="dao",
    )
    print(f"\nProtocol Risk: {risk.protocol}")
    print(f"  Risk: {risk.risk_level.value} (score: {risk.risk_score})")
    print(f"  Safety: {', '.join(risk.safety_factors)}")
