"""
MEV Protection & Detection System — BARREN WUFFET v2.7.0
=========================================================
Detect, analyze, and protect against Maximal Extractable Value (MEV)
attacks including sandwich attacks, frontrunning, and JIT liquidity.

From BARREN WUFFET Insights (686-720):
  - MEV bots extract $200M+ annually from DeFi users
  - Sandwich attacks: bot frontrun + backrun a victim swap
  - Flashbots Protect provides private mempool to avoid frontrunning
  - Private RPCs reduce MEV exposure by 85-95%
  - JIT liquidity: LPs add/remove liquidity within same block
  - Proposer-Builder Separation (PBS) changes MEV distribution
  - MEV-Share redistributes extracted value to users
  - Block builders concentrate: top 3 build 90%+ of blocks
  - Slippage tolerance >1% on AMMs invites sandwich attacks
  - Time-weighted average pricing (TWAP) reduces manipulation
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class MEVAttackType(Enum):
    """MEVAttackType class."""
    SANDWICH = "sandwich"               # Frontrun + backrun victim swap
    FRONTRUN = "frontrun"               # Copy victim tx with higher gas
    BACKRUN = "backrun"                 # Arbitrage after victim tx
    JIT_LIQUIDITY = "jit_liquidity"     # Add/remove LP in same block
    LIQUIDATION = "liquidation"         # Race to liquidate undercollateralized
    TIME_BANDIT = "time_bandit"         # Reorg blocks for MEV
    CENSORSHIP = "censorship"           # Builder excludes txs
    LONG_TAIL = "long_tail"             # Niche/unique MEV opportunities

class ProtectionLevel(Enum):
    """ProtectionLevel class."""
    NONE = "none"                       # Raw mempool, fully exposed
    BASIC = "basic"                     # Reasonable slippage settings
    PRIVATE_RPC = "private_rpc"         # Private tx submission
    FLASHBOTS = "flashbots"             # Flashbots Protect / MEV Blocker
    MEV_SHARE = "mev_share"             # MEV-Share for kickbacks
    MAXIMUM = "maximum"                 # All protections + TWAP

class RiskLevel(Enum):
    """RiskLevel class."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MEVAlert:
    """Individual MEV attack detection alert."""
    attack_type: MEVAttackType
    risk_level: RiskLevel
    tx_hash: str
    victim_address: Optional[str]
    attacker_address: Optional[str]
    estimated_loss_usd: float
    block_number: int
    chain: str
    details: Dict
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SandwichDetection:
    """Detailed sandwich attack detection result."""
    is_sandwich: bool
    frontrun_tx: Optional[str]
    victim_tx: Optional[str]
    backrun_tx: Optional[str]
    attacker_profit_usd: float
    victim_loss_usd: float
    pool_address: str
    token_pair: str
    slippage_exploited: float  # Actual slippage as %
    block_number: int
    notes: List[str] = field(default_factory=list)


@dataclass
class TransactionProtectionPlan:
    """Protection plan for a pending transaction."""
    original_tx: Dict
    risk_level: RiskLevel
    vulnerability_score: float  # 0-100
    recommended_protection: ProtectionLevel
    max_slippage: float
    recommended_slippage: float
    use_private_rpc: bool
    split_trades: bool
    num_splits: int
    use_twap: bool
    twap_duration_minutes: int
    estimated_savings_usd: float
    notes: List[str] = field(default_factory=list)


@dataclass
class MEVDashboard:
    """Overview of MEV activity and protection status."""
    chain: str
    total_mev_24h_usd: float
    sandwich_count_24h: int
    frontrun_count_24h: int
    jit_count_24h: int
    top_extracted_tokens: List[Dict]
    top_builders: List[Dict]
    avg_extraction_per_block: float
    protection_effectiveness: float  # % of attacks blocked
    user_savings_30d: float


# ═══════════════════════════════════════════════════════════════════════════
# SANDWICH ATTACK DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class SandwichDetector:
    """
    Detect sandwich attacks in DEX transactions.

    A sandwich attack works:
      1. Bot sees victim's pending swap in mempool
      2. Bot frontrun: buy the token (pushes price up)
      3. Victim's swap executes at worse price
      4. Bot backrun: sell the token at higher price
      5. Bot profits from price impact

    Detection heuristics:
      - Same token pair, 3 txs in tight block sequence
      - Frontrun+backrun from same address (or linked addresses)
      - Price impact anomaly: victim gets worse fill than expected
      - Gas patterns: frontrun has higher gas than victim
    """

    @classmethod
    def detect_sandwich(
        cls,
        txs_in_block: List[Dict],
        pool_address: str,
        expected_price: float,
    ) -> Optional[SandwichDetection]:
        """
        Analyze transactions in a block for sandwich patterns.

        Each tx in txs_in_block should have:
          - hash, from_addr, to_addr, token_in, token_out,
          - amount_in, amount_out, gas_price, tx_index
        """
        if len(txs_in_block) < 3:
            return None

        # Sort by transaction index
        sorted_txs = sorted(txs_in_block, key=lambda x: x.get("tx_index", 0))

        # Look for sandwich pattern: buy → victim → sell
        for i in range(len(sorted_txs) - 2):
            front = sorted_txs[i]
            victim = sorted_txs[i + 1]
            back = sorted_txs[i + 2]

            # Check: front and back from same address
            if front.get("from_addr") != back.get("from_addr"):
                continue

            # Check: front buys what victim is buying (same direction)
            if front.get("token_out") != victim.get("token_out"):
                continue

            # Check: back sells what front bought
            if back.get("token_in") != front.get("token_out"):
                continue

            # Check: front has higher gas
            if front.get("gas_price", 0) <= victim.get("gas_price", 0):
                continue

            # Calculate losses
            victim_expected_out = victim.get("amount_in", 0) * expected_price
            victim_actual_out = victim.get("amount_out", 0)
            victim_loss = max(0, victim_expected_out - victim_actual_out)

            attacker_bought = front.get("amount_out", 0)
            attacker_sold = back.get("amount_in", 0)
            attacker_profit = attacker_sold - attacker_bought

            slippage = (victim_loss / victim_expected_out * 100) if victim_expected_out > 0 else 0

            return SandwichDetection(
                is_sandwich=True,
                frontrun_tx=front.get("hash"),
                victim_tx=victim.get("hash"),
                backrun_tx=back.get("hash"),
                attacker_profit_usd=round(attacker_profit, 2),
                victim_loss_usd=round(victim_loss, 2),
                pool_address=pool_address,
                token_pair=f"{front.get('token_in', '?')}/{front.get('token_out', '?')}",
                slippage_exploited=round(slippage, 3),
                block_number=front.get("block", 0),
                notes=[
                    "SANDWICH DETECTED",
                    f"Attacker: {front.get('from_addr', 'unknown')}",
                    f"Victim loss: ~${victim_loss:,.2f}",
                    f"Attacker profit: ~${attacker_profit:,.2f}",
                    f"Slippage exploited: {slippage:.2f}%",
                ],
            )

        return None


# ═══════════════════════════════════════════════════════════════════════════
# TRANSACTION PROTECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class TransactionProtectionEngine:
    """
    Analyze pending transactions and generate protection plans.

    Protection strategies ranked by effectiveness:
      1. Private RPC (Flashbots Protect) — removes from public mempool
      2. Low slippage tolerance (0.3-0.5%) — limits extraction
      3. Trade splitting — reduces per-trade price impact
      4. TWAP execution — spreads trades over time
      5. MEV-Share — redirects MEV back to user
      6. Limit orders — avoid market orders entirely
    """

    # Minimum trade sizes that attract MEV bots (by chain)
    MEV_ATTENTION_THRESHOLDS = {
        "ethereum": 5_000,     # $5K+ gets attention on mainnet
        "arbitrum": 2_000,     # Lower threshold on L2
        "polygon": 1_000,
        "bsc": 3_000,
        "optimism": 2_000,
        "base": 1_500,
    }

    @classmethod
    def assess_risk(
        cls, trade_value_usd: float, slippage_tolerance: float,
        chain: str, pool_liquidity: float, is_mempool_public: bool,
    ) -> TransactionProtectionPlan:
        """Assess MEV risk and generate protection plan."""
        threshold = cls.MEV_ATTENTION_THRESHOLDS.get(chain, 5000)
        notes = []

        # Base vulnerability score (0-100)
        vuln = 0

        # Trade size factor
        if trade_value_usd > threshold * 10:
            vuln += 40
            notes.append(f"LARGE trade (${trade_value_usd:,.0f}) — major MEV target.")
        elif trade_value_usd > threshold:
            vuln += 20
            notes.append(f"Medium trade above ${threshold:,} threshold for {chain}.")

        # Slippage factor — the biggest risk
        if slippage_tolerance > 3.0:
            vuln += 30
            notes.append(f"CRITICAL: {slippage_tolerance}% slippage is a gift to sandwich bots!")
        elif slippage_tolerance > 1.0:
            vuln += 20
            notes.append(f"HIGH: {slippage_tolerance}% slippage allows profitable sandwiches.")
        elif slippage_tolerance > 0.5:
            vuln += 10

        # Pool liquidity factor
        if pool_liquidity > 0:
            impact = trade_value_usd / pool_liquidity * 100
            if impact > 1.0:
                vuln += 20
                notes.append(f"Trade is {impact:.1f}% of pool liquidity — high impact risk.")
            elif impact > 0.3:
                vuln += 10

        # Public mempool
        if is_mempool_public:
            vuln += 15
            notes.append("Transaction visible in public mempool — exposed to frontrunning.")

        # Clamp
        vuln = min(100, vuln)

        # Risk level
        if vuln >= 70:
            risk = RiskLevel.CRITICAL
        elif vuln >= 50:
            risk = RiskLevel.HIGH
        elif vuln >= 30:
            risk = RiskLevel.MEDIUM
        elif vuln >= 15:
            risk = RiskLevel.LOW
        else:
            risk = RiskLevel.SAFE

        # Protection plan
        use_private = vuln >= 30
        use_twap = trade_value_usd > threshold * 5
        split = trade_value_usd > threshold * 3
        num_splits = max(1, int(trade_value_usd / threshold)) if split else 1
        num_splits = min(num_splits, 10)

        recommended_slippage = min(0.5, slippage_tolerance) if vuln > 30 else min(1.0, slippage_tolerance)
        twap_minutes = max(5, int(trade_value_usd / 10_000)) if use_twap else 0

        # Determine protection level
        if vuln >= 70:
            protection = ProtectionLevel.MAXIMUM
        elif vuln >= 50:
            protection = ProtectionLevel.FLASHBOTS
        elif vuln >= 30:
            protection = ProtectionLevel.PRIVATE_RPC
        elif vuln >= 15:
            protection = ProtectionLevel.BASIC
        else:
            protection = ProtectionLevel.NONE

        # Estimated savings
        potential_loss = trade_value_usd * slippage_tolerance / 100 * 0.5
        savings = potential_loss * 0.85 if use_private else potential_loss * 0.3

        if protection == ProtectionLevel.NONE:
            notes.append("Low risk — standard transaction should be fine.")
        else:
            notes.append(f"Recommended: {protection.value} protection.")

        return TransactionProtectionPlan(
            original_tx={"value_usd": trade_value_usd, "chain": chain},
            risk_level=risk,
            vulnerability_score=vuln,
            recommended_protection=protection,
            max_slippage=slippage_tolerance,
            recommended_slippage=recommended_slippage,
            use_private_rpc=use_private,
            split_trades=split,
            num_splits=num_splits,
            use_twap=use_twap,
            twap_duration_minutes=twap_minutes,
            estimated_savings_usd=round(savings, 2),
            notes=notes,
        )


# ═══════════════════════════════════════════════════════════════════════════
# FLASHBOTS INTEGRATION GUIDE
# ═══════════════════════════════════════════════════════════════════════════

class FlashbotsGuide:
    """
    Reference guide for Flashbots integration patterns.

    Flashbots Protect: Private transaction submission
      - Transactions go directly to block builders, skipping public mempool
      - Free to use — just change RPC endpoint
      - Supports MEV-Share for kickback redistribution

    Integration methods:
      1. RPC endpoint: https://rpc.flashbots.net (simple)
      2. eth_sendPrivateTransaction API call
      3. mev-share bundles for advanced users
    """

    FLASHBOTS_RPC = "https://rpc.flashbots.net"
    MEV_SHARE_RPC = "https://rpc.flashbots.net/fast"

    RPC_CONFIGS = {
        "ethereum": {
            "flashbots": "https://rpc.flashbots.net",
            "mev_blocker": "https://rpc.mevblocker.io",
            "mev_share": "https://rpc.flashbots.net/fast",
            "notes": "Flashbots Protect is the gold standard. Free.",
        },
        "polygon": {
            "flashbots": None,
            "private_rpc": "Use Polygon private mempool or aggregator",
            "notes": "Flashbots not available on Polygon. Use aggregators.",
        },
        "arbitrum": {
            "flashbots": None,
            "private_rpc": "Arbitrum sequencer provides some ordering protection",
            "notes": "Sequencer provides partial protection. Still use low slippage.",
        },
    }

    @classmethod
    def get_protection_config(cls, chain: str) -> Dict:
        """Get MEV protection configuration for a chain."""
        config = cls.RPC_CONFIGS.get(chain, {})
        if not config:
            return {
                "chain": chain,
                "available": False,
                "recommendation": "Use low slippage + limit orders. No private RPC available.",
            }
        return {
            "chain": chain,
            "available": True,
            **config,
        }

    @classmethod
    def generate_web3_snippet(cls, chain: str = "ethereum") -> str:
        """Generate Python snippet for Flashbots-protected transaction."""
        return f"""
# Flashbots Protect RPC — BARREN WUFFET MEV Protection
# Simply change your RPC endpoint to avoid frontrunning

from web3 import Web3

# Instead of: w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/...'))
# Use Flashbots Protect (transactions skip public mempool):
w3 = Web3(Web3.HTTPProvider('{cls.FLASHBOTS_RPC}'))

# Or MEV-Share for faster inclusion + MEV kickbacks:
# w3 = Web3(Web3.HTTPProvider('{cls.MEV_SHARE_RPC}'))

# Your transaction code works exactly the same — zero code changes needed!
# The only difference: your tx is submitted privately to block builders.
"""


# ═══════════════════════════════════════════════════════════════════════════
# MEV-AWARE DEX AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

class MEVAwareDEXRouter:
    """
    Smart order routing that accounts for MEV exposure.

    Principles:
      - Split large orders across pools to reduce impact
      - Route through pools with deepest liquidity
      - Use private RPCs for trades above threshold
      - Prefer limit orders over market orders
      - Apply dynamic slippage based on pool depth
    """

    @classmethod
    def compute_optimal_route(
        cls,
        amount_usd: float,
        pools: List[Dict],  # Each pool: {name, liquidity, fee_tier, dex}
        max_splits: int = 5,
    ) -> Dict:
        """Compute MEV-optimized routing across pools."""
        if not pools:
            return {"error": "No pools available"}

        total_liquidity = sum(p.get("liquidity", 0) for p in pools)
        if total_liquidity <= 0:
            return {"error": "No liquidity available"}

        # Sort by liquidity (deepest first)
        sorted_pools = sorted(pools, key=lambda p: p.get("liquidity", 0), reverse=True)

        # Allocate pro-rata to liquidity, capped at max_splits pools
        top_pools = sorted_pools[:max_splits]
        top_liquidity = sum(p.get("liquidity", 0) for p in top_pools)

        splits = []
        for pool in top_pools:
            liq = pool.get("liquidity", 0)
            allocation = amount_usd * (liq / top_liquidity) if top_liquidity > 0 else 0
            price_impact = allocation / liq * 100 if liq > 0 else 100

            splits.append({
                "pool": pool.get("name", "unknown"),
                "dex": pool.get("dex", "unknown"),
                "allocation_usd": round(allocation, 2),
                "allocation_pct": round(allocation / amount_usd * 100, 1),
                "estimated_impact": round(price_impact, 4),
                "fee_tier": pool.get("fee_tier", "0.3%"),
            })

        total_impact = sum(s["estimated_impact"] * s["allocation_pct"] / 100 for s in splits)

        return {
            "total_amount": amount_usd,
            "num_splits": len(splits),
            "splits": splits,
            "total_estimated_impact": round(total_impact, 4),
            "recommended_slippage": round(total_impact * 2 + 0.3, 2),  # 2x impact + base
            "use_private_rpc": amount_usd > 5000,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("🐺 BARREN WUFFET — MEV Protection & Detection System v2.7.0")
    logger.info("=" * 60)

    # Transaction Protection Assessment
    plan = TransactionProtectionEngine.assess_risk(
        trade_value_usd=50_000,
        slippage_tolerance=2.0,
        chain="ethereum",
        pool_liquidity=5_000_000,
        is_mempool_public=True,
    )
    logger.info(f"\nMEV Risk Assessment:")
    logger.info(f"  Risk Level: {plan.risk_level.value}")
    logger.info(f"  Vulnerability: {plan.vulnerability_score}/100")
    logger.info(f"  Protection: {plan.recommended_protection.value}")
    logger.info(f"  Recommended Slippage: {plan.recommended_slippage}%")
    logger.info(f"  Split Trades: {plan.split_trades} ({plan.num_splits} splits)")
    logger.info(f"  Use TWAP: {plan.use_twap} ({plan.twap_duration_minutes} min)")
    logger.info(f"  Estimated Savings: ${plan.estimated_savings_usd:,.2f}")
    for note in plan.notes:
        logger.info(f"  → {note}")

    # Flashbots config
    config = FlashbotsGuide.get_protection_config("ethereum")
    logger.info(f"\nFlashbots Config:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    # DEX Routing
    pools = [
        {"name": "WETH/USDC 0.3%", "liquidity": 50_000_000, "fee_tier": "0.3%", "dex": "Uniswap V3"},
        {"name": "WETH/USDC 0.05%", "liquidity": 30_000_000, "fee_tier": "0.05%", "dex": "Uniswap V3"},
        {"name": "WETH/USDC", "liquidity": 20_000_000, "fee_tier": "0.3%", "dex": "SushiSwap"},
        {"name": "WETH/USDC Stable", "liquidity": 15_000_000, "fee_tier": "0.01%", "dex": "Curve"},
    ]
    route = MEVAwareDEXRouter.compute_optimal_route(50_000, pools)
    logger.info(f"\nOptimal Route ({route['num_splits']} splits):")
    for split in route["splits"]:
        print(f"  → {split['dex']} {split['pool']}: ${split['allocation_usd']:,.0f} "
              f"({split['allocation_pct']}%) impact: {split['estimated_impact']:.3f}%")
    logger.info(f"  Total Impact: {route['total_estimated_impact']:.3f}%")
    logger.info(f"  Recommended Slippage: {route['recommended_slippage']}%")
