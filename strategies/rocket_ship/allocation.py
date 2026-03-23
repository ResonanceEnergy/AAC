"""
Rocket Ship — Allocation Engine
=================================
Manages Life Boat and Rocket Ship portfolio allocation models.

Life Boat allocation (Moons 1-12):
    40%  BTC + physical gold       — neutral digital/physical reserve
    30%  stablecoins (USDC/USDT)   — fiat liquidity + DeFi bridge
    20%  XRP                       — FX bridge liquidity
    10%  cash (USD)                — bank on-ramp buffer

Rocket Ship allocation (Moons 13-39):
    25%  XRP + Flare (FXRP)        — FX bridge + DeFi yield
    20%  Solana + wXRP/stables     — Web3/payments/LP yield
    15%  Ethereum                  — Smart-contract staking + L2 DeFi
    15%  Bitcoin                   — Digital gold long-term
    15%  BRICS/Unit proxies        — New system direct exposure (PAXG, tokenized Unit)
    10%  Self-custody DeFi tools   — Oracles, multi-chain wallets, yield infra

Yield tactics (Rocket phase):
    XRP → Flare bridge → FXRP deposit Morpho lending vault (target 5-8% APY)
    FXRP + stables → SparkDEX LP (target 8-15% APY variable)
    ETH → Native staking or Lido/stETH (3-5% APY, MEV boost possible)
    SOL → Marinade/native staking + DEX LPs  (5-8% APY)
    USDC → Morpho/Aave markets  (4-7% APY)
    PAXG → Gold-stable LP pools  (2-4% APY + gold appreciation)

Risk management:
    Use ISOLATED Morpho markets (limited contagion per vault)
    Never borrow > 60% of supplied collateral
    Start 10% of position in any new protocol
    Monitor audited status: Morpho (Cantina), Kinetic (audited), SparkDEX (ongoing)
    Hardware wallet self-custody for all non-yield holdings
    Multi-sig for > $50K positions
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from strategies.rocket_ship.core import (
    Asset,
    AssetTier,
    LIFEBOAT_ALLOC_BTC_GOLD,
    LIFEBOAT_ALLOC_CASH,
    LIFEBOAT_ALLOC_STABLES,
    LIFEBOAT_ALLOC_XRP,
    ROCKET_ALLOC_BRICS,
    ROCKET_ALLOC_BITCOIN,
    ROCKET_ALLOC_DEFI_TOOLS,
    ROCKET_ALLOC_ETHEREUM,
    ROCKET_ALLOC_SOLANA,
    ROCKET_ALLOC_XRP_FLARE,
    STARTING_CAPITAL_USD,
    SystemPhase,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ALLOCATION TARGETS
# ═══════════════════════════════════════════════════════════════════════════

# Life Boat targets
LIFEBOAT_TARGETS: Dict[str, float] = {
    "BTC + Physical Gold":      LIFEBOAT_ALLOC_BTC_GOLD,     # 0.40
    "Stablecoins (USDC/USDT)":  LIFEBOAT_ALLOC_STABLES,     # 0.30
    "XRP":                      LIFEBOAT_ALLOC_XRP,          # 0.20
    "Cash (USD)":               LIFEBOAT_ALLOC_CASH,         # 0.10
}

# Rocket Ship targets
ROCKET_TARGETS: Dict[str, float] = {
    "XRP + Flare (FXRP)":              ROCKET_ALLOC_XRP_FLARE,    # 0.25
    "Solana + wXRP/Stables":           ROCKET_ALLOC_SOLANA,       # 0.20
    "Ethereum (stake + L2 DeFi)":      ROCKET_ALLOC_ETHEREUM,     # 0.15
    "Bitcoin (digital gold hold)":     ROCKET_ALLOC_BITCOIN,      # 0.15
    "BRICS/Unit + Gold-pegged stables": ROCKET_ALLOC_BRICS,       # 0.15
    "DeFi Tools + Self-custody Infra": ROCKET_ALLOC_DEFI_TOOLS,   # 0.10
}


# ═══════════════════════════════════════════════════════════════════════════
# YIELD PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class YieldTactic:
    """Specification of one yield-generating strategy."""
    name: str
    assets_needed: List[str]
    protocol: str
    chain: str
    target_apy_low: float     # Conservative estimate
    target_apy_high: float    # Optimistic estimate
    how_to_execute: str
    risk_level: str           # "low" | "medium" | "high"
    audit_status: str
    max_allocation_pct: float   # Never deploy more than this % to single tactic
    notes: str


YIELD_TACTICS: List[YieldTactic] = [
    YieldTactic(
        name="FXRP Morpho Lending",
        assets_needed=["XRP → FXRP (Flare bridge)"],
        protocol="Morpho (Flare deployment)",
        chain="Flare Network",
        target_apy_low=5.0,
        target_apy_high=8.0,
        how_to_execute=(
            "1. Bridge native XRP → Flare native bridge → receive FXRP 1:1\n"
            "2. Visit Morpho Flare vault (morpho.org/flare)\n"
            "3. Deposit FXRP into isolated lending market\n"
            "4. Earn lending interest — no smart contract leverage needed\n"
            "5. Optionally borrow USDC (≤60% LTV) → redeploy for delta-neutral"
        ),
        risk_level="medium",
        audit_status="Morpho audited by Cantina (2024). Flare vaults ongoing.",
        max_allocation_pct=0.20,
        notes="Use isolated markets only. Start 10% of XRP allocation to test.",
    ),
    YieldTactic(
        name="SparkDEX FXRP/USDC LP",
        assets_needed=["FXRP", "USDC"],
        protocol="SparkDEX (Flare DEX)",
        chain="Flare Network",
        target_apy_low=8.0,
        target_apy_high=15.0,
        how_to_execute=(
            "1. Bridge FXRP + USDC to Flare\n"
            "2. Visit SparkDEX (sparkdex.io)\n"
            "3. Add liquidity to FXRP/USDC concentrated pool\n"
            "4. Collect trading fees + FLR incentive rewards\n"
            "5. Rebalance range monthly on waning moon"
        ),
        risk_level="medium",
        audit_status="SparkDEX DEX audit ongoing (2025). Core AMM standard.",
        max_allocation_pct=0.15,
        notes="Impermanent loss risk if XRP/USDC ratio moves >30%. Monitor weekly.",
    ),
    YieldTactic(
        name="Solana USDC Money Market",
        assets_needed=["USDC or wXRP on Solana"],
        protocol="Kamino Finance or Marginfi",
        chain="Solana",
        target_apy_low=5.0,
        target_apy_high=10.0,
        how_to_execute=(
            "1. Bridge wXRP to Solana via Hex Trust / LayerZero\n"
            "2. Swap 50% → USDC on Orca/Raydium\n"
            "3. Deposit USDC into Kamino money market\n"
            "4. Earn lending interest (variable)\n"
            "5. Add wXRP/USDC LP on Orca for additional fees"
        ),
        risk_level="medium",
        audit_status="Kamino: multiple audits (Ottersec, Halborn). Production-grade.",
        max_allocation_pct=0.15,
        notes="Solana network risk. Firedancer upgrade reduces outage probability.",
    ),
    YieldTactic(
        name="ETH Native Staking",
        assets_needed=["ETH"],
        protocol="Native validator or Lido (stETH)",
        chain="Ethereum",
        target_apy_low=3.0,
        target_apy_high=5.0,
        how_to_execute=(
            "1. For 32+ ETH: run own validator (Lighthouse/Prysm client)\n"
            "2. For <32 ETH: use Lido → receive stETH (liquid staking token)\n"
            "3. stETH auto-compounds daily\n"
            "4. Can deploy stETH as collateral in Aave/Morpho for additional yield"
        ),
        risk_level="low",
        audit_status="Lido: extensively audited. Largest ETH liquid staking protocol.",
        max_allocation_pct=0.15,
        notes="Slashing risk (negligible with Lido diversification). Regulatory risk on staking.",
    ),
    YieldTactic(
        name="Solana Native Staking (Marinade)",
        assets_needed=["SOL"],
        protocol="Marinade Finance (mSOL)",
        chain="Solana",
        target_apy_low=6.0,
        target_apy_high=8.0,
        how_to_execute=(
            "1. Stake SOL via Marinade → receive mSOL (liquid staking token)\n"
            "2. mSOL auto-compounds; can be used in DeFi\n"
            "3. Deploy mSOL as collateral or in LP pools for stacking yield"
        ),
        risk_level="low",
        audit_status="Marinade: widely audited, major Solana DeFi protocol.",
        max_allocation_pct=0.12,
        notes="Smart contract risk. Solana slashing risk minimal (no slashing currently).",
    ),
    YieldTactic(
        name="PAXG / Gold-Stable LP",
        assets_needed=["PAXG", "USDC or USDT"],
        protocol="Curve Finance (gold-stable pools) or Balancer",
        chain="Ethereum / Polygon",
        target_apy_low=2.0,
        target_apy_high=4.0,
        how_to_execute=(
            "1. Purchase PAXG on Kraken or Gemini\n"
            "2. Bridge to Ethereum or Polygon\n"
            "3. Add to Curve PAXG/USDC pool\n"
            "4. Earn trading fees + CRV/BAL rewards\n"
            "5. Gauge boost possible with CRV locking"
        ),
        risk_level="low",
        audit_status="Curve: battle-tested, billions TVL. Paxos PAXG: audited custodian.",
        max_allocation_pct=0.10,
        notes="Gold price appreciation is the primary thesis — LP yield is bonus.",
    ),
    YieldTactic(
        name="USDC/USDT Stablecoin Farm",
        assets_needed=["USDC or USDT"],
        protocol="Aave v3 or Morpho Blue",
        chain="Ethereum / Polygon / Base",
        target_apy_low=4.0,
        target_apy_high=7.0,
        how_to_execute=(
            "1. Keep 50% of stablecoin allocation in Aave/Morpho market\n"
            "2. Supply USDC → earn variable rate\n"
            "3. Remaining 50% as dry powder for bridge/rebalance\n"
            "4. Review rate vs Morpho isolated market weekly"
        ),
        risk_level="low",
        audit_status="Aave: industry standard, $15B+ TVL, multiple audits.",
        max_allocation_pct=0.20,
        notes="De-peg risk minimal with USDC (Circle regulated). Monitor bank reserves.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# ALLOCATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AllocationPlan:
    """Computed allocation breakdown for a given total portfolio value."""
    phase: SystemPhase
    total_usd: float
    buckets: Dict[str, Tuple[float, float]]   # label -> (pct, usd_value)
    yield_tactics: List[YieldTactic]
    estimated_portfolio_yield_low: float      # % APY low estimate
    estimated_portfolio_yield_high: float     # % APY high estimate

    @property
    def annual_yield_usd_low(self) -> float:
        return self.total_usd * self.estimated_portfolio_yield_low / 100

    @property
    def annual_yield_usd_high(self) -> float:
        return self.total_usd * self.estimated_portfolio_yield_high / 100


class AllocationEngine:
    """
    Generates portfolio allocation plans for Life Boat and Rocket phases.
    """

    def compute(self, phase: SystemPhase, total_usd: float = STARTING_CAPITAL_USD) -> AllocationPlan:
        """Return a full allocation plan for the given phase and portfolio size."""
        if phase == SystemPhase.LIFE_BOAT:
            targets = LIFEBOAT_TARGETS
            active_tactics = [t for t in YIELD_TACTICS
                              if t.name in ("USDC/USDT Stablecoin Farm",)]
            yield_low  = 1.5   # Stables only in Life Boat
            yield_high = 3.0
        else:
            # IGNITING / ROCKET / ORBIT all use rocket targets
            targets = ROCKET_TARGETS
            active_tactics = YIELD_TACTICS
            # Weighted yield estimate across Rocket allocation
            # 25% XRP/FLR at 5-8%, 20% SOL at 5-8%, 15% ETH at 3-5%, 10% stables at 4-7%
            # 15% BTC (0% yield), 15% BRICS/gold (2-4%)
            yield_low  = (0.25 * 5 + 0.20 * 5 + 0.15 * 3 + 0.10 * 4 + 0.15 * 0 + 0.15 * 2)
            yield_high = (0.25 * 8 + 0.20 * 8 + 0.15 * 5 + 0.10 * 7 + 0.15 * 0 + 0.15 * 4)

        buckets = {
            label: (pct, round(pct * total_usd, 2))
            for label, pct in targets.items()
        }

        return AllocationPlan(
            phase=phase,
            total_usd=total_usd,
            buckets=buckets,
            yield_tactics=active_tactics,
            estimated_portfolio_yield_low=round(yield_low, 1),
            estimated_portfolio_yield_high=round(yield_high, 1),
        )

    def format_dashboard(self, phase: SystemPhase, total_usd: float = STARTING_CAPITAL_USD) -> str:
        """Return formatted ASCII allocation dashboard."""
        plan = self.compute(phase, total_usd)

        phase_labels = {
            SystemPhase.LIFE_BOAT: "LIFE BOAT",
            SystemPhase.IGNITING:  "IGNITING — Deploying",
            SystemPhase.ROCKET:    "ROCKET SHIP",
            SystemPhase.ORBIT:     "ORBIT",
        }
        p_label = phase_labels.get(phase, phase.value)

        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════════════╗",
            f"║  ALLOCATION: {p_label:<20}  Total: ${plan.total_usd:>10,.0f} USD          ║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
        ]
        for label, (pct, usd) in plan.buckets.items():
            bar_len = int(pct * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(
                f"║  {pct*100:>4.0f}%  [{bar}]  ${usd:>10,.0f}  {label[:19]:<19}  ║"
            )

        lines += [
            "╠══════════════════════════════════════════════════════════════════════════╣",
            f"║  Estimated Annual Yield:  {plan.estimated_portfolio_yield_low:.1f}% – {plan.estimated_portfolio_yield_high:.1f}%"
            f"  (~${plan.annual_yield_usd_low:,.0f} – ${plan.annual_yield_usd_high:,.0f} USD/yr)  ║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
            "║  ACTIVE YIELD TACTICS:                                                  ║",
        ]
        for t in plan.yield_tactics[:4]:
            lines.append(
                f"║  • {t.name:<28}  {t.target_apy_low:.0f}-{t.target_apy_high:.0f}% APY"
                f"  [{t.risk_level.upper():>6}]  {t.chain[:12]:<12}  ║"
            )

        lines.append("╚══════════════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)
