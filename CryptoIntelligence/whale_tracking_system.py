"""
Whale Tracking & Accumulation System — BARREN WUFFET v2.7.0
=============================================================
Monitor large wallet activity, detect accumulation/distribution
patterns, and track smart money flows across chains.

From BARREN WUFFET Insights (601-640, 811-850):
  - Top 100 BTC wallets hold ~15% of supply — their moves matter
  - Whale accumulation during fear = strongest bull signal
  - Exchange deposits from known whales signal sell pressure
  - Dormant wallet awakening (>3yr) creates market anxiety
  - CEX-to-DEX flows often precede DeFi yield farming trends
  - Smart money rotation: BTC → ETH → Large cap → Mid → Micro → back to BTC
  - Wallet clustering can identify entity-level flows
  - Token unlocks (VC/team vesting) create predictable sell pressure
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class WalletType(Enum):
    """WalletType class."""
    WHALE = "whale"                   # >1000 BTC or >10K ETH
    INSTITUTION = "institution"        # Known institutional wallet
    EXCHANGE_HOT = "exchange_hot"      # Exchange hot wallet
    EXCHANGE_COLD = "exchange_cold"    # Exchange cold storage
    MINER = "miner"                   # Mining pool / miner
    SMART_MONEY = "smart_money"        # Historically profitable trader
    VC_FUND = "vc_fund"               # Venture capital fund
    FOUNDATION = "foundation"          # Project foundation/treasury
    UNKNOWN_LARGE = "unknown_large"    # Large but unidentified
    DORMANT = "dormant"               # No activity in 1yr+

class WalletActivity(Enum):
    """WalletActivity class."""
    ACCUMULATING = "accumulating"      # Net inflows
    DISTRIBUTING = "distributing"      # Net outflows
    DORMANT = "dormant"               # No activity
    CHURNING = "churning"             # High activity, no clear trend
    AWAKENED = "awakened"             # First activity after long dormancy

class FlowDirection(Enum):
    """FlowDirection class."""
    TO_EXCHANGE = "to_exchange"        # Selling pressure
    FROM_EXCHANGE = "from_exchange"    # Accumulation
    TO_DEFI = "to_defi"              # DeFi participation
    FROM_DEFI = "from_defi"          # DeFi exit
    TO_BRIDGE = "to_bridge"           # Cross-chain move
    PEER_TO_PEER = "peer_to_peer"     # OTC or transfer
    TO_MIXER = "to_mixer"            # Privacy tool
    UNKNOWN = "unknown"

class AlertPriority(Enum):
    """AlertPriority class."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedWallet:
    """A whale wallet being tracked."""
    address: str
    label: str
    wallet_type: WalletType
    chain: str
    balance: float
    balance_usd: float
    first_seen: str
    last_active: str
    activity: WalletActivity
    pnl_estimated_usd: float = 0
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class WhaleTransaction:
    """A significant whale transaction."""
    tx_hash: str
    from_address: str
    to_address: str
    from_label: str
    to_label: str
    token: str
    amount: float
    value_usd: float
    flow_direction: FlowDirection
    chain: str
    block_number: int
    timestamp: str
    significance: AlertPriority
    notes: List[str] = field(default_factory=list)


@dataclass
class AccumulationSignal:
    """Accumulation/distribution detection signal."""
    token: str
    signal_type: str  # "accumulation" or "distribution"
    whale_count: int
    total_volume_usd: float
    timeframe_hours: int
    confidence: float  # 0-100
    price_at_signal: float
    avg_whale_buy_price: float
    notes: List[str] = field(default_factory=list)


@dataclass
class SmartMoneyFlow:
    """Aggregate smart money flow analysis."""
    token: str
    net_flow_usd_24h: float  # Positive = inflow (buying)
    net_flow_usd_7d: float
    whale_buy_count: int
    whale_sell_count: int
    top_buyers: List[Dict]
    top_sellers: List[Dict]
    flow_trend: str  # "strong_buy", "buy", "neutral", "sell", "strong_sell"
    conviction: float


# ═══════════════════════════════════════════════════════════════════════════
# WHALE CLASSIFICATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class WhaleClassifier:
    """
    Classify wallets by type and behavior.

    Classification heuristics:
      - Balance size → whale vs fish
      - Known labels → exchange, institution, fund
      - Transaction patterns → trader vs HODLer
      - DeFi interactions → smart money indicator
      - Age of holdings → conviction level
    """

    # Whale thresholds by token
    WHALE_THRESHOLDS = {
        "BTC": {"whale": 100, "mega_whale": 1000, "shark": 10},
        "ETH": {"whale": 1000, "mega_whale": 10000, "shark": 100},
        "SOL": {"whale": 10000, "mega_whale": 100000, "shark": 1000},
    }

    # Known exchange hot wallet patterns (simplified)
    KNOWN_EXCHANGES = {
        "binance", "coinbase", "kraken", "bitfinex", "huobi",
        "okx", "bybit", "kucoin", "gate", "gemini",
    }

    @classmethod
    def classify_wallet(
        cls,
        address: str,
        balance: float,
        token: str,
        label: Optional[str],
        tx_count_30d: int,
        defi_interactions: int,
        last_active_days_ago: int,
        avg_hold_time_days: float,
    ) -> Tuple[WalletType, WalletActivity, List[str]]:
        """Classify a wallet."""
        tags = []

        # Check known labels
        if label:
            label_lower = label.lower()
            if any(ex in label_lower for ex in cls.KNOWN_EXCHANGES):
                wallet_type = WalletType.EXCHANGE_HOT
                tags.append("exchange")
            elif "cold" in label_lower or "storage" in label_lower:
                wallet_type = WalletType.EXCHANGE_COLD
                tags.append("cold_storage")
            elif any(vc in label_lower for vc in ["a16z", "paradigm", "sequoia", "pantera",
                                                     "polychain", "multicoin", "dragonfly"]):
                wallet_type = WalletType.VC_FUND
                tags.append("vc_fund")
            elif "foundation" in label_lower or "treasury" in label_lower:
                wallet_type = WalletType.FOUNDATION
                tags.append("foundation")
            elif "miner" in label_lower or "pool" in label_lower:
                wallet_type = WalletType.MINER
                tags.append("miner")
            else:
                wallet_type = WalletType.UNKNOWN_LARGE
        else:
            # Classify by balance
            thresholds = cls.WHALE_THRESHOLDS.get(token, {"whale": 100, "mega_whale": 1000, "shark": 10})
            if balance >= thresholds["mega_whale"]:
                wallet_type = WalletType.WHALE
                tags.append("mega_whale")
            elif balance >= thresholds["whale"]:
                wallet_type = WalletType.WHALE
                tags.append("whale")
            elif balance >= thresholds["shark"]:
                wallet_type = WalletType.UNKNOWN_LARGE
                tags.append("shark")
            else:
                wallet_type = WalletType.UNKNOWN_LARGE

        # Smart money detection
        if defi_interactions > 50 and avg_hold_time_days < 30:
            wallet_type = WalletType.SMART_MONEY
            tags.append("active_defi_trader")

        # Activity classification
        if last_active_days_ago > 365:
            activity = WalletActivity.DORMANT
            tags.append("dormant_1yr+")
        elif last_active_days_ago > 90:
            activity = WalletActivity.DORMANT
            tags.append("inactive_3mo")
        elif tx_count_30d > 50:
            activity = WalletActivity.CHURNING
            tags.append("high_frequency")
        elif tx_count_30d > 5:
            activity = WalletActivity.ACCUMULATING  # Default for active
        else:
            activity = WalletActivity.DORMANT

        return wallet_type, activity, tags


# ═══════════════════════════════════════════════════════════════════════════
# FLOW DIRECTION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class FlowDirectionDetector:
    """Detect the direction and significance of flows."""

    EXCHANGE_ADDRESSES: Set[str] = set()  # Would be populated from labels DB

    @classmethod
    def detect_flow(
        cls,
        from_label: str, to_label: str,
        amount_usd: float,
    ) -> Tuple[FlowDirection, AlertPriority, List[str]]:
        """Detect flow direction and significance."""
        from_l = from_label.lower() if from_label else ""
        to_l = to_label.lower() if to_label else ""
        notes = []

        # Exchange flows
        if any(ex in to_l for ex in ["binance", "coinbase", "kraken", "exchange"]):
            direction = FlowDirection.TO_EXCHANGE
            notes.append("Whale depositing to exchange — potential sell pressure.")
        elif any(ex in from_l for ex in ["binance", "coinbase", "kraken", "exchange"]):
            direction = FlowDirection.FROM_EXCHANGE
            notes.append("Whale withdrawing from exchange — likely accumulation.")
        elif any(d in to_l for d in ["uniswap", "aave", "compound", "curve", "defi"]):
            direction = FlowDirection.TO_DEFI
            notes.append("Whale entering DeFi — yield farming or trading.")
        elif any(d in from_l for d in ["uniswap", "aave", "compound", "curve", "defi"]):
            direction = FlowDirection.FROM_DEFI
            notes.append("Whale exiting DeFi position.")
        elif any(b in to_l for b in ["bridge", "wormhole", "stargate", "across"]):
            direction = FlowDirection.TO_BRIDGE
            notes.append("Cross-chain transfer — watch destination chain.")
        elif any(m in to_l for m in ["tornado", "mixer", "wasabi"]):
            direction = FlowDirection.TO_MIXER
            notes.append("⚠️ Whale using mixer — privacy or laundering?")
        else:
            direction = FlowDirection.PEER_TO_PEER
            notes.append("Direct transfer — possible OTC deal.")

        # Priority
        if amount_usd > 50_000_000:
            priority = AlertPriority.CRITICAL
            notes.append(f"MASSIVE: ${amount_usd/1e6:.0f}M transfer!")
        elif amount_usd > 10_000_000:
            priority = AlertPriority.HIGH
            notes.append(f"Large: ${amount_usd/1e6:.0f}M transfer.")
        elif amount_usd > 1_000_000:
            priority = AlertPriority.MEDIUM
        else:
            priority = AlertPriority.LOW

        return direction, priority, notes


# ═══════════════════════════════════════════════════════════════════════════
# ACCUMULATION / DISTRIBUTION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class AccumulationDetector:
    """
    Detect accumulation or distribution patterns from whale activity.

    Accumulation signals:
      - Multiple whales buying over 24-72h
      - Exchange outflows sustained
      - Buy concentration at specific price levels
      - Dormant wallets receiving new deposits

    Distribution signals:
      - Multiple whales sending to exchanges
      - Large OTC block sales
      - VC/team wallet unlocks + transfers
      - Coordinated multi-wallet selling
    """

    @classmethod
    def detect(
        cls,
        transactions: List[WhaleTransaction],
        current_price: float,
        timeframe_hours: int = 72,
    ) -> Optional[AccumulationSignal]:
        """Detect accumulation or distribution from recent whale txs."""
        if not transactions:
            return None

        buys = [t for t in transactions if t.flow_direction == FlowDirection.FROM_EXCHANGE]
        sells = [t for t in transactions if t.flow_direction == FlowDirection.TO_EXCHANGE]

        buy_volume = sum(t.value_usd for t in buys)
        sell_volume = sum(t.value_usd for t in sells)
        net_volume = buy_volume - sell_volume

        unique_buyers = len(set(t.to_address for t in buys))
        unique_sellers = len(set(t.from_address for t in sells))

        notes = []

        if net_volume > 0 and unique_buyers >= 3:
            signal_type = "accumulation"
            confidence = min(95, 30 + unique_buyers * 10 + (net_volume / 1_000_000) * 5)
            notes.append(f"{unique_buyers} whales net buying ${net_volume/1e6:.1f}M in {timeframe_hours}h.")
            if unique_buyers >= 5:
                notes.append("Strong multi-whale convergence — high conviction signal.")
        elif net_volume < 0 and unique_sellers >= 3:
            signal_type = "distribution"
            confidence = min(95, 30 + unique_sellers * 10 + (abs(net_volume) / 1_000_000) * 5)
            notes.append(f"{unique_sellers} whales net selling ${abs(net_volume)/1e6:.1f}M in {timeframe_hours}h.")
            if unique_sellers >= 5:
                notes.append("Heavy coordinated distribution — caution!")
        else:
            return None  # No clear signal

        # Average buy/sell price estimate
        avg_price = current_price  # Would use actual tx prices in production

        return AccumulationSignal(
            token=transactions[0].token if transactions else "UNKNOWN",
            signal_type=signal_type,
            whale_count=unique_buyers if signal_type == "accumulation" else unique_sellers,
            total_volume_usd=abs(net_volume),
            timeframe_hours=timeframe_hours,
            confidence=round(confidence, 1),
            price_at_signal=current_price,
            avg_whale_buy_price=avg_price,
            notes=notes,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TOKEN UNLOCK TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class TokenUnlockTracker:
    """
    Track upcoming token unlocks (vesting schedules).

    VC/team token unlocks create predictable sell pressure:
      - Cliff unlocks: sudden large supply increase
      - Linear vesting: gradual but consistent pressure
      - Impact: unlocks >2% of circulating supply are high-impact
    """

    @classmethod
    def assess_unlock_impact(
        cls,
        token: str,
        unlock_amount: float,
        circulating_supply: float,
        current_price: float,
        unlock_type: str,  # "cliff", "linear_monthly", "linear_daily"
        recipient: str,  # "team", "vc", "foundation", "community"
    ) -> Dict:
        """Assess the impact of an upcoming token unlock."""
        supply_pct = (unlock_amount / circulating_supply * 100) if circulating_supply > 0 else 0
        value_usd = unlock_amount * current_price

        notes = []
        if supply_pct > 5:
            impact = "CRITICAL"
            notes.append(f"🚨 {supply_pct:.1f}% of supply unlocking — massive sell pressure risk!")
        elif supply_pct > 2:
            impact = "HIGH"
            notes.append(f"⚠️ {supply_pct:.1f}% unlock — significant dilution event.")
        elif supply_pct > 0.5:
            impact = "MEDIUM"
            notes.append(f"{supply_pct:.1f}% unlock — moderate impact expected.")
        else:
            impact = "LOW"
            notes.append(f"{supply_pct:.2f}% unlock — minimal impact.")

        # Recipient matters
        if recipient in ("vc", "team"):
            notes.append(f"Recipient: {recipient} — likely to sell (at least partially).")
        elif recipient == "foundation":
            notes.append("Recipient: foundation — may hold or use for ecosystem.")
        elif recipient == "community":
            notes.append("Recipient: community — distributed sell pressure.")

        # Historical pattern
        if unlock_type == "cliff":
            notes.append("Cliff unlock — ALL tokens available immediately.")
        elif unlock_type == "linear_monthly":
            notes.append("Linear monthly — pressure spread over vesting period.")

        return {
            "token": token,
            "unlock_amount": unlock_amount,
            "supply_percentage": round(supply_pct, 2),
            "value_usd": round(value_usd, 2),
            "impact": impact,
            "recipient": recipient,
            "unlock_type": unlock_type,
            "notes": notes,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("🐺 BARREN WUFFET — Whale Tracking System v2.7.0")
    logger.info("=" * 55)

    # Classify a wallet
    w_type, w_activity, w_tags = WhaleClassifier.classify_wallet(
        address="0xabc123...",
        balance=5000,
        token="ETH",
        label=None,
        tx_count_30d=3,
        defi_interactions=5,
        last_active_days_ago=2,
        avg_hold_time_days=180,
    )
    logger.info(f"\nWallet Classification:")
    logger.info(f"  Type: {w_type.value}")
    logger.info(f"  Activity: {w_activity.value}")
    logger.info(f"  Tags: {w_tags}")

    # Flow detection
    direction, priority, notes = FlowDirectionDetector.detect_flow(
        from_label="Unknown Whale",
        to_label="Binance Hot Wallet",
        amount_usd=15_000_000,
    )
    logger.info(f"\nFlow Detection:")
    logger.info(f"  Direction: {direction.value}")
    logger.info(f"  Priority: {priority.value}")
    for n in notes:
        print(f"  \u2192 {n}")

    # Token unlock
    unlock = TokenUnlockTracker.assess_unlock_impact(
        token="ARB",
        unlock_amount=100_000_000,
        circulating_supply=3_000_000_000,
        current_price=1.20,
        unlock_type="cliff",
        recipient="vc",
    )
    logger.info(f"\nToken Unlock:")
    logger.info(f"  Impact: {unlock['impact']}")
    logger.info(f"  Supply %: {unlock['supply_percentage']}%")
    logger.info(f"  Value: ${unlock['value_usd']:,.0f}")
    for n in unlock["notes"]:
        print(f"  → {n}")
