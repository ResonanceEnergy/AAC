"""
Crypto Technical Patterns & Market Microstructure — BARREN WUFFET v2.7.0
=========================================================================
Crypto-specific technical analysis patterns, funding rate analysis,
liquidation cascade detection, and order flow dynamics.

From BARREN WUFFET Insights (721-810):
  - Funding rates >0.1% per 8h = extremely bullish sentiment (contrarian sell)
  - Negative funding + rising price = healthy rally (shorts paying longs)
  - Open interest divergence vs price = potential reversal setup
  - Liquidation cascades amplify moves: $100M+ liq = likely reversal
  - Bitcoin dominance >60% = alt season not yet started
  - BTC.D declining + ETH/BTC rising = rotation to alts beginning
  - Volume profile gaps in crypto filled 85%+ of the time
  - Weekend gaps in CME futures filled within 1 week 78% of time
  - 200-week MA has been the generational BTC bottom indicator
  - Hash ribbon buy signal has 100% historical accuracy after miner capitulation
  - Crypto market hours matter: Asian session vs US session behavior differs
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

class FundingRegime(Enum):
    EXTREME_LONG = "extreme_long"     # >0.1% / 8h — euphoric leveraged longs
    ELEVATED_LONG = "elevated_long"   # 0.03-0.1% / 8h
    NORMAL = "normal"                 # 0.005-0.03% / 8h
    NEUTRAL = "neutral"              # ~0.01% / 8h
    NEGATIVE = "negative"             # <0% — shorts paying longs
    EXTREME_SHORT = "extreme_short"   # <-0.05% — massive short pressure

class LiquidationType(Enum):
    LONG_LIQUIDATION = "long_liquidation"
    SHORT_LIQUIDATION = "short_liquidation"
    CASCADE_LONG = "cascade_long"     # Chain reaction of long liqs
    CASCADE_SHORT = "cascade_short"   # Chain reaction of short liqs

class TrendContext(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

class DominancePhase(Enum):
    BTC_DOMINANCE = "btc_dominance"         # BTC.D rising, alt season far
    ETH_ROTATION = "eth_rotation"           # ETH/BTC rising, early alt
    LARGE_CAP_ALT = "large_cap_alt_season"  # Top 20 alts outperforming
    MID_CAP_ALT = "mid_cap_alt_season"      # Mid caps running
    MICRO_CAP_MANIA = "micro_cap_mania"     # Meme coins, shitcoins pumping
    BTC_RETURN = "btc_return_flight"        # Risk-off back to BTC


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FundingRateAnalysis:
    """Funding rate analysis result."""
    symbol: str
    current_rate: float        # Per 8h
    annualized_rate: float
    regime: FundingRegime
    avg_7d: float
    avg_30d: float
    z_score: float             # Current vs 30d distribution
    signal: str                # "contrarian_sell", "contrarian_buy", "neutral"
    carry_trade_viable: bool   # Can profit from funding
    notes: List[str] = field(default_factory=list)


@dataclass
class OpenInterestAnalysis:
    """Open Interest analysis."""
    symbol: str
    total_oi_usd: float
    oi_change_24h_pct: float
    oi_change_7d_pct: float
    long_short_ratio: float        # >1 = more longs
    price_change_24h_pct: float
    divergence_detected: bool       # OI vs price divergence
    divergence_type: Optional[str]  # "bearish" or "bullish"
    notes: List[str] = field(default_factory=list)


@dataclass
class LiquidationEvent:
    """Significant liquidation event."""
    symbol: str
    liq_type: LiquidationType
    total_volume_usd: float
    count: int
    largest_single_usd: float
    price_at_liq: float
    price_move_pct: float      # Price move that triggered
    is_cascade: bool
    recovery_probability: float  # Likelihood of reversal after liq
    notes: List[str] = field(default_factory=list)


@dataclass
class DominanceAnalysis:
    """Bitcoin dominance and rotation analysis."""
    btc_dominance: float
    btc_d_30d_change: float
    eth_btc_ratio: float
    eth_btc_30d_change: float
    alt_season_index: float  # 0-100
    phase: DominancePhase
    rotation_targets: List[str]  # Sectors likely to outperform
    notes: List[str] = field(default_factory=list)


@dataclass
class CryptoTASignal:
    """Crypto-specific technical signal."""
    symbol: str
    signal_name: str
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-100
    timeframe: str
    entry_zone: Optional[Tuple[float, float]]
    target: Optional[float]
    stop: Optional[float]
    notes: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# FUNDING RATE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class FundingRateAnalyzer:
    """
    Analyze perpetual futures funding rates.
    
    Funding Rate Mechanics:
      - Perpetual futures have no expiry, so funding keeps price near spot
      - Positive funding: longs pay shorts (bullish consensus)
      - Negative funding: shorts pay longs (bearish consensus)
      - Rates settle every 8 hours (Binance) or 1 hour (some DEXs)
    
    Trading signals:
      - Extreme positive funding → contrarian sell (too many longs)
      - Extreme negative funding → contrarian buy (too many shorts)
      - Positive funding + rising price = healthy trend
      - Negative funding + rising price = strong rally (shorts squeezed)
    """

    @classmethod
    def analyze(
        cls,
        symbol: str,
        current_rate: float,  # Per 8h
        rates_7d: List[float],
        rates_30d: List[float],
        price_change_24h: float = 0,
    ) -> FundingRateAnalysis:
        """Analyze funding rate regime and generate signals."""
        annualized = current_rate * 3 * 365  # 3 periods/day * 365 days

        avg_7d = sum(rates_7d) / len(rates_7d) if rates_7d else 0
        avg_30d = sum(rates_30d) / len(rates_30d) if rates_30d else 0
        std_30d = (sum((r - avg_30d)**2 for r in rates_30d) / len(rates_30d))**0.5 if len(rates_30d) > 1 else 0.01
        z_score = (current_rate - avg_30d) / std_30d if std_30d > 0 else 0

        notes = []

        # Regime classification
        if current_rate > 0.001:  # >0.1% per 8h
            regime = FundingRegime.EXTREME_LONG
            signal = "contrarian_sell"
            notes.append(f"🔴 Funding {current_rate*100:.3f}% — EXTREME long crowding!")
            notes.append("Historically, rates this high precede 5-15% corrections.")
        elif current_rate > 0.0003:
            regime = FundingRegime.ELEVATED_LONG
            signal = "lean_contrarian_sell"
            notes.append(f"Funding elevated at {current_rate*100:.3f}% — longs building.")
        elif current_rate > 0.00005:
            regime = FundingRegime.NORMAL
            signal = "neutral"
            notes.append(f"Funding normal at {current_rate*100:.4f}%.")
        elif current_rate > -0.0001:
            regime = FundingRegime.NEUTRAL
            signal = "neutral"
        elif current_rate > -0.0005:
            regime = FundingRegime.NEGATIVE
            signal = "contrarian_buy"
            notes.append(f"Negative funding {current_rate*100:.3f}% — shorts paying.")
            if price_change_24h > 0:
                notes.append("Negative funding + rising price = VERY bullish (short squeeze setup).")
                signal = "strong_contrarian_buy"
        else:
            regime = FundingRegime.EXTREME_SHORT
            signal = "contrarian_buy"
            notes.append(f"🟢 Funding {current_rate*100:.3f}% — EXTREME short crowding!")
            notes.append("Historically, rates this negative precede sharp bounces.")

        # Carry trade analysis
        carry_viable = abs(annualized) > 20  # >20% annualized funding = profitable carry
        if carry_viable:
            direction = "short" if current_rate > 0 else "long"
            notes.append(f"Carry trade: go {direction} perp + hedge spot = {abs(annualized):.0f}% APR.")

        return FundingRateAnalysis(
            symbol=symbol,
            current_rate=round(current_rate, 6),
            annualized_rate=round(annualized, 2),
            regime=regime,
            avg_7d=round(avg_7d, 6),
            avg_30d=round(avg_30d, 6),
            z_score=round(z_score, 2),
            signal=signal,
            carry_trade_viable=carry_viable,
            notes=notes,
        )


# ═══════════════════════════════════════════════════════════════════════════
# OPEN INTEREST ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class OpenInterestAnalyzer:
    """
    Track open interest dynamics and divergences.
    
    Key patterns:
      - Rising OI + rising price = new money entering longs (bullish confirmation)
      - Rising OI + falling price = new shorts opening (bearish)
      - Falling OI + rising price = short covering rally (weak rally)
      - Falling OI + falling price = long liquidation / capitulation (potential bottom)
    """

    @classmethod
    def analyze(
        cls,
        symbol: str,
        oi_current: float,
        oi_24h_ago: float,
        oi_7d_ago: float,
        long_short_ratio: float,
        price_current: float,
        price_24h_ago: float,
    ) -> OpenInterestAnalysis:
        """Analyze OI dynamics."""
        oi_change_24h = ((oi_current / oi_24h_ago) - 1) * 100 if oi_24h_ago > 0 else 0
        oi_change_7d = ((oi_current / oi_7d_ago) - 1) * 100 if oi_7d_ago > 0 else 0
        price_change = ((price_current / price_24h_ago) - 1) * 100 if price_24h_ago > 0 else 0

        notes = []
        divergence = False
        div_type = None

        # Pattern detection
        if oi_change_24h > 5 and price_change > 2:
            notes.append("Rising OI + rising price — bullish confirmation. New longs entering.")
        elif oi_change_24h > 5 and price_change < -2:
            notes.append("Rising OI + falling price — new shorts opening. Bearish pressure.")
            divergence = True
            div_type = "bearish"
        elif oi_change_24h < -5 and price_change > 2:
            notes.append("Falling OI + rising price — SHORT COVERING rally. May be weak.")
            divergence = True
            div_type = "bearish"  # Weak rally
        elif oi_change_24h < -5 and price_change < -2:
            notes.append("Falling OI + falling price — long liquidation / capitulation.")
            divergence = True
            div_type = "bullish"  # Potential bottom

        # Long/short ratio
        if long_short_ratio > 2.0:
            notes.append(f"L/S ratio {long_short_ratio:.1f} — heavily long-skewed. Squeeze risk ↓")
        elif long_short_ratio < 0.7:
            notes.append(f"L/S ratio {long_short_ratio:.1f} — heavily short-skewed. Squeeze risk ↑")

        return OpenInterestAnalysis(
            symbol=symbol,
            total_oi_usd=oi_current,
            oi_change_24h_pct=round(oi_change_24h, 2),
            oi_change_7d_pct=round(oi_change_7d, 2),
            long_short_ratio=round(long_short_ratio, 2),
            price_change_24h_pct=round(price_change, 2),
            divergence_detected=divergence,
            divergence_type=div_type,
            notes=notes,
        )


# ═══════════════════════════════════════════════════════════════════════════
# LIQUIDATION CASCADE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class LiquidationCascadeDetector:
    """
    Detect and analyze liquidation cascades.
    
    Liquidation cascade mechanics:
      1. Price moves toward cluster of leveraged positions
      2. Positions get liquidated → forced market sells
      3. Forced selling pushes price further → more liquidations
      4. Cascade continues until liquidity absorbs the flow
      5. Often creates V-shaped reversal (liquidation wick)
    
    Key levels to watch:
      - Binance liquidation heatmap levels
      - Aggregate OI * leverage * position size at each level
      - Cascades >$100M in 1h often produce strong reversals
    """

    @classmethod
    def analyze_event(
        cls,
        symbol: str,
        total_liq_usd: float,
        liq_count: int,
        largest_single: float,
        is_longs: bool,
        price_at_start: float,
        price_at_end: float,
        duration_minutes: int,
    ) -> LiquidationEvent:
        """Analyze a liquidation event."""
        price_move = ((price_at_end / price_at_start) - 1) * 100 if price_at_start > 0 else 0
        notes = []

        # Cascade detection
        is_cascade = total_liq_usd > 50_000_000 and liq_count > 100

        liq_type = (LiquidationType.CASCADE_LONG if is_cascade else LiquidationType.LONG_LIQUIDATION) \
            if is_longs else \
            (LiquidationType.CASCADE_SHORT if is_cascade else LiquidationType.SHORT_LIQUIDATION)

        # Recovery probability
        # Historical: large cascades tend to reverse 60-80% of the time
        if is_cascade:
            recovery_prob = 70 + min(20, total_liq_usd / 50_000_000 * 5)
            notes.append(f"🌊 LIQUIDATION CASCADE: ${total_liq_usd/1e6:.0f}M liquidated!")
            notes.append(f"Price moved {abs(price_move):.1f}% in {duration_minutes} min.")
            notes.append(f"Historically, cascades this large reverse ~{recovery_prob:.0f}% of the time.")
            direction = "bounce" if is_longs else "pullback"
            notes.append(f"Likely {direction} incoming as forced selling exhausts.")
        else:
            recovery_prob = 40
            notes.append(f"Liquidation event: ${total_liq_usd/1e6:.1f}M, {liq_count} positions.")

        if largest_single > 10_000_000:
            notes.append(f"⚠️ Single liquidation of ${largest_single/1e6:.0f}M — likely institutional.")

        return LiquidationEvent(
            symbol=symbol,
            liq_type=liq_type,
            total_volume_usd=total_liq_usd,
            count=liq_count,
            largest_single_usd=largest_single,
            price_at_liq=price_at_start,
            price_move_pct=round(price_move, 2),
            is_cascade=is_cascade,
            recovery_probability=round(min(95, recovery_prob), 1),
            notes=notes,
        )


# ═══════════════════════════════════════════════════════════════════════════
# DOMINANCE & ROTATION ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class DominanceRotationAnalyzer:
    """
    Track BTC dominance and sector rotation patterns.
    
    Crypto rotation cycle (typical):
      BTC pumps → ETH catches up → Large cap alts run →
      Mid caps explode → Micro cap mania → Everything crashes →
      Capital returns to BTC → Repeat
    
    BTC Dominance levels:
      >65% = BTC season, not alt season yet
      55-65% = Transition zone
      45-55% = Alt season underway
      <45%  = Peak alt season / euphoria
    """

    @classmethod
    def analyze(
        cls,
        btc_dominance: float,
        btc_d_30d_ago: float,
        eth_btc: float,
        eth_btc_30d_ago: float,
        top10_avg_30d_return: float = 0,
        defi_tvl_change_30d: float = 0,
    ) -> DominanceAnalysis:
        """Analyze dominance and rotation."""
        btc_d_change = btc_dominance - btc_d_30d_ago
        eth_btc_change = ((eth_btc / eth_btc_30d_ago) - 1) * 100 if eth_btc_30d_ago > 0 else 0

        # Alt season index: inverse of BTC dominance, scaled
        alt_season_index = max(0, min(100, (65 - btc_dominance) * 100 / 25))

        notes = []
        rotation_targets = []

        # Phase detection
        if btc_dominance > 65:
            phase = DominancePhase.BTC_DOMINANCE
            notes.append(f"BTC.D {btc_dominance:.1f}% — BTC dominant. Alt season NOT started.")
            rotation_targets = ["BTC"]
        elif btc_dominance > 55:
            if eth_btc_change > 5:
                phase = DominancePhase.ETH_ROTATION
                notes.append("ETH/BTC rising + BTC.D declining — early alt rotation underway.")
                rotation_targets = ["ETH", "SOL", "Large Cap L1s"]
            elif btc_d_change > 0:
                phase = DominancePhase.BTC_RETURN
                notes.append("BTC.D rising — risk-off rotation back to BTC.")
                rotation_targets = ["BTC"]
            else:
                phase = DominancePhase.LARGE_CAP_ALT
                notes.append("Transition zone — large cap alts starting to move.")
                rotation_targets = ["ETH", "SOL", "AVAX", "LINK"]
        elif btc_dominance > 45:
            if top10_avg_30d_return > 30:
                phase = DominancePhase.MID_CAP_ALT
                notes.append("Full alt season — mid caps outperforming.")
                rotation_targets = ["Mid cap DeFi", "Gaming", "AI tokens"]
            else:
                phase = DominancePhase.LARGE_CAP_ALT
                notes.append("Alt season building. Large caps leading.")
                rotation_targets = ["ETH", "Layer 2s", "DeFi blue chips"]
        else:
            phase = DominancePhase.MICRO_CAP_MANIA
            notes.append(f"⚠️ BTC.D {btc_dominance:.1f}% — MICRO CAP MANIA. Peak euphoria zone!")
            notes.append("Historically, BTC.D <45% marks cycle tops. Be extremely cautious.")
            rotation_targets = ["NOTHING — reduce risk", "Take profits"]

        # Alt season index commentary
        if alt_season_index > 80:
            notes.append(f"Alt Season Index: {alt_season_index:.0f}/100 — PEAK ALT SEASON!")
        elif alt_season_index > 50:
            notes.append(f"Alt Season Index: {alt_season_index:.0f}/100 — alts favored.")
        else:
            notes.append(f"Alt Season Index: {alt_season_index:.0f}/100 — BTC still king.")

        return DominanceAnalysis(
            btc_dominance=round(btc_dominance, 2),
            btc_d_30d_change=round(btc_d_change, 2),
            eth_btc_ratio=round(eth_btc, 6),
            eth_btc_30d_change=round(eth_btc_change, 2),
            alt_season_index=round(alt_season_index, 1),
            phase=phase,
            rotation_targets=rotation_targets,
            notes=notes,
        )


# ═══════════════════════════════════════════════════════════════════════════
# CRYPTO-SPECIFIC TA PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

class CryptoPatternDetector:
    """
    Crypto-specific patterns that differ from traditional markets.
    
    Unique patterns:
      - CME gap fills (78% fill rate within 1 week)
      - Weekend gap fills 
      - Hash ribbon buy signal (100% historical accuracy for BTC)
      - VPVR gap fills (85%+ fill rate in crypto)
      - Liquidation wicks / V-reversals
      - Wyckoff accumulation (institutional accumulation)
    """

    @classmethod
    def check_cme_gap(
        cls, friday_close: float, monday_open: float, current_price: float,
    ) -> Optional[CryptoTASignal]:
        """Check for CME gap fill opportunity."""
        gap = monday_open - friday_close
        gap_pct = abs(gap) / friday_close * 100

        if gap_pct < 0.5:
            return None  # No significant gap

        # Is the gap filled?
        if gap > 0:
            # Gap up — price needs to come down to friday_close
            gap_filled = current_price <= friday_close
            direction = "bearish"
            target = friday_close
            entry = (current_price, current_price * 1.005)
        else:
            # Gap down — price needs to come up to friday_close
            gap_filled = current_price >= friday_close
            direction = "bullish"
            target = friday_close
            entry = (current_price * 0.995, current_price)

        if gap_filled:
            return None  # Already filled

        notes = [
            f"CME gap: ${friday_close:.0f} → ${monday_open:.0f} ({gap_pct:.1f}%)",
            "Historical fill rate: ~78% within 1 week.",
            f"Target: ${target:.0f} (gap fill level).",
        ]

        return CryptoTASignal(
            symbol="BTC",
            signal_name="cme_gap_fill",
            direction=direction,
            strength=70,
            timeframe="1D-1W",
            entry_zone=entry,
            target=target,
            stop=None,
            notes=notes,
        )

    @classmethod
    def check_200_week_ma(
        cls, current_price: float, ma_200w: float,
    ) -> Optional[CryptoTASignal]:
        """Check 200-week MA (generational bottom indicator for BTC)."""
        distance_pct = ((current_price / ma_200w) - 1) * 100

        if distance_pct > 50:
            return None  # Too far above — not relevant

        notes = []
        if distance_pct < 0:
            direction = "bullish"
            strength = 95
            notes.append(f"🟢 Price BELOW 200W MA — BTC has NEVER closed below it permanently!")
            notes.append(f"Distance: {abs(distance_pct):.1f}% below. Generational buying zone.")
        elif distance_pct < 20:
            direction = "bullish"
            strength = 75
            notes.append(f"Price near 200W MA ({distance_pct:.1f}% above). Strong support zone.")
        else:
            return None

        return CryptoTASignal(
            symbol="BTC",
            signal_name="200_week_ma",
            direction=direction,
            strength=strength,
            timeframe="1W",
            entry_zone=(ma_200w * 0.95, ma_200w * 1.10),
            target=ma_200w * 2.0,
            stop=ma_200w * 0.80,
            notes=notes,
        )


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🐺 BARREN WUFFET — Crypto Technical Patterns Engine v2.7.0")
    print("=" * 60)

    # Funding Rate
    rates_30d = [0.0001 * (1 + i * 0.1) for i in range(90)]  # Simulated
    fr = FundingRateAnalyzer.analyze(
        symbol="BTC-PERP",
        current_rate=0.0012,  # 0.12% — extreme
        rates_7d=rates_30d[-21:],
        rates_30d=rates_30d,
        price_change_24h=2.5,
    )
    print(f"\nFunding Rate: {fr.symbol}")
    print(f"  Rate: {fr.current_rate*100:.4f}% per 8h ({fr.annualized_rate:.0f}% annualized)")
    print(f"  Regime: {fr.regime.value}")
    print(f"  Signal: {fr.signal}")
    print(f"  Carry Trade: {fr.carry_trade_viable}")
    for n in fr.notes: print(f"  → {n}")

    # Open Interest
    oi = OpenInterestAnalyzer.analyze(
        symbol="BTC-PERP",
        oi_current=15_000_000_000,
        oi_24h_ago=14_000_000_000,
        oi_7d_ago=12_500_000_000,
        long_short_ratio=2.3,
        price_current=68000,
        price_24h_ago=66000,
    )
    print(f"\nOpen Interest: {oi.symbol}")
    print(f"  OI: ${oi.total_oi_usd/1e9:.1f}B | 24h: {oi.oi_change_24h_pct:+.1f}%")
    print(f"  L/S Ratio: {oi.long_short_ratio}")
    print(f"  Divergence: {oi.divergence_type}")
    for n in oi.notes: print(f"  → {n}")

    # Liquidation Cascade
    liq = LiquidationCascadeDetector.analyze_event(
        symbol="BTC",
        total_liq_usd=250_000_000,
        liq_count=5000,
        largest_single=25_000_000,
        is_longs=True,
        price_at_start=70000,
        price_at_end=65000,
        duration_minutes=45,
    )
    print(f"\nLiquidation Event: {liq.symbol}")
    print(f"  Type: {liq.liq_type.value}")
    print(f"  Volume: ${liq.total_volume_usd/1e6:.0f}M")
    print(f"  Cascade: {liq.is_cascade}")
    print(f"  Recovery Prob: {liq.recovery_probability}%")
    for n in liq.notes: print(f"  → {n}")

    # Dominance
    dom = DominanceRotationAnalyzer.analyze(
        btc_dominance=52.3, btc_d_30d_ago=55.1,
        eth_btc=0.052, eth_btc_30d_ago=0.048,
        top10_avg_30d_return=15,
    )
    print(f"\nDominance Analysis:")
    print(f"  BTC.D: {dom.btc_dominance}% (30d: {dom.btc_d_30d_change:+.1f}%)")
    print(f"  Phase: {dom.phase.value}")
    print(f"  Alt Season Index: {dom.alt_season_index}/100")
    print(f"  Rotation: {dom.rotation_targets}")
    for n in dom.notes: print(f"  → {n}")
