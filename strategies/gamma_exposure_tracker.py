"""
Gamma Exposure (GEX) & Dealer Hedging Flow Tracker — BARREN WUFFET v2.7.0
==========================================================================
Tracks aggregate dealer gamma exposure across the options market to
predict volatility regimes, support/resistance levels, and hedging flows.

Core concepts:
  - When dealers are NET LONG gamma → they sell rallies, buy dips → SUPPRESSES volatility
  - When dealers are NET SHORT gamma → they chase price → AMPLIFIES volatility
  - GEX "flip" levels mark the boundary between these regimes
  - Large GEX concentrations at strikes create "sticky" price magnets
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

class VolatilityRegime(Enum):
    """VolatilityRegime class."""
    SUPPRESSED = "suppressed"      # Dealers long gamma → stabilizing
    AMPLIFIED = "amplified"        # Dealers short gamma → destabilizing
    TRANSITIONAL = "transitional"  # Near GEX flip point


@dataclass
class OptionOI:
    """Open interest data for a single strike/expiry."""
    strike: float
    expiry: str
    call_oi: int = 0
    put_oi: int = 0
    call_volume: int = 0
    put_volume: int = 0
    call_iv: float = 0.0
    put_iv: float = 0.0
    call_delta: float = 0.0
    put_delta: float = 0.0
    call_gamma: float = 0.0
    put_gamma: float = 0.0

    @property
    def total_oi(self) -> int:
        """Total oi."""
        return self.call_oi + self.put_oi

    @property
    def put_call_ratio(self) -> float:
        """Put call ratio."""
        return self.put_oi / self.call_oi if self.call_oi > 0 else float("inf")


@dataclass
class GEXLevel:
    """Gamma exposure at a single strike."""
    strike: float
    call_gex: float = 0.0
    put_gex: float = 0.0
    net_gex: float = 0.0

    @property
    def is_positive(self) -> bool:
        """Is positive."""
        return self.net_gex > 0


@dataclass
class GEXProfile:
    """Complete GEX profile across all strikes."""
    symbol: str
    underlying_price: float
    timestamp: str
    levels: List[GEXLevel] = field(default_factory=list)
    total_gex: float = 0.0
    flip_level: float = 0.0
    max_gamma_strike: float = 0.0
    regime: VolatilityRegime = VolatilityRegime.TRANSITIONAL
    put_wall: float = 0.0
    call_wall: float = 0.0

    def to_dict(self) -> Dict:
        """To dict."""
        return {
            "symbol": self.symbol,
            "underlying_price": self.underlying_price,
            "total_gex": self.total_gex,
            "flip_level": self.flip_level,
            "max_gamma_strike": self.max_gamma_strike,
            "regime": self.regime.value,
            "put_wall": self.put_wall,
            "call_wall": self.call_wall,
            "num_levels": len(self.levels),
            "timestamp": self.timestamp,
        }


@dataclass
class DealerExposure:
    """Dealer hedging exposure summary."""
    net_delta: float = 0.0       # Net delta exposure (shares equivalent)
    net_gamma: float = 0.0       # Net gamma exposure ($ per 1% move)
    net_vanna: float = 0.0       # Delta change per IV change
    net_charm: float = 0.0       # Delta decay per day
    hedging_pressure: str = ""   # "buying" or "selling"
    estimated_shares: int = 0    # Estimated shares dealers must trade


# ═══════════════════════════════════════════════════════════════════════════
# GEX CALCULATOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class GammaExposureEngine:
    """
    Core engine for computing gamma exposure (GEX) profiles.

    Assumptions:
    - Dealers are generally SHORT calls to customers (customer buys calls)
    - Dealers are generally LONG puts to customers (customer buys puts)
    - Net dealer gamma = -call_gamma * call_OI + put_gamma * put_OI
    - Positive GEX → dealers hedge by selling rallies / buying dips (stabilizing)
    - Negative GEX → dealers hedge by buying rallies / selling dips (amplifying)
    """

    MULTIPLIER = 100  # Options multiplier (100 shares per contract)

    def __init__(self, symbol: str, underlying_price: float):
        self.symbol = symbol
        self.S = underlying_price
        self.oi_data: List[OptionOI] = []

    def add_oi_data(self, data: List[OptionOI]) -> None:
        """Load open interest data."""
        self.oi_data = data
        logger.info(f"Loaded {len(data)} OI records for {self.symbol}")

    def compute_gex_at_strike(self, oi: OptionOI) -> GEXLevel:
        """Compute GEX at a single strike."""
        # Call GEX: dealers SHORT calls → negative gamma contribution
        # Convention: multiply by -1 because dealers sold calls
        call_gex = -(oi.call_gamma * oi.call_oi * self.MULTIPLIER * self.S)

        # Put GEX: dealers LONG puts → positive gamma contribution
        # But put gamma adds to selling pressure, so sign is flipped
        put_gex = oi.put_gamma * oi.put_oi * self.MULTIPLIER * self.S * (-1)

        net_gex = call_gex + put_gex

        return GEXLevel(
            strike=oi.strike,
            call_gex=call_gex,
            put_gex=put_gex,
            net_gex=net_gex,
        )

    def compute_gex_profile(self) -> GEXProfile:
        """Compute full GEX profile across all strikes."""
        levels = [self.compute_gex_at_strike(oi) for oi in self.oi_data]
        levels.sort(key=lambda lvl: lvl.strike)

        total_gex = sum(lvl.net_gex for lvl in levels)

        # Find flip level (where GEX crosses zero)
        flip_level = self._find_flip_level(levels)

        # Find max gamma strike
        max_gamma_level = max(levels, key=lambda lvl: abs(lvl.net_gex)) if levels else None
        max_gamma_strike = max_gamma_level.strike if max_gamma_level else 0

        # Find put wall (largest negative put GEX below spot)
        put_wall = self._find_put_wall(levels)
        call_wall = self._find_call_wall(levels)

        # Determine regime
        regime = self._determine_regime(total_gex, flip_level)

        return GEXProfile(
            symbol=self.symbol,
            underlying_price=self.S,
            timestamp=datetime.utcnow().isoformat(),
            levels=levels,
            total_gex=round(total_gex, 2),
            flip_level=round(flip_level, 2),
            max_gamma_strike=max_gamma_strike,
            regime=regime,
            put_wall=put_wall,
            call_wall=call_wall,
        )

    def _find_flip_level(self, levels: List[GEXLevel]) -> float:
        """Find the price where GEX flips from positive to negative."""
        for i in range(len(levels) - 1):
            if (levels[i].net_gex > 0 and levels[i + 1].net_gex < 0) or \
               (levels[i].net_gex < 0 and levels[i + 1].net_gex > 0):
                # Interpolate
                if levels[i + 1].net_gex != levels[i].net_gex:
                    ratio = -levels[i].net_gex / (levels[i + 1].net_gex - levels[i].net_gex)
                    return levels[i].strike + ratio * (levels[i + 1].strike - levels[i].strike)
        return self.S  # Default to current price if no flip found

    def _find_put_wall(self, levels: List[GEXLevel]) -> float:
        """Find the strike with largest put gamma below spot."""
        below = [lvl for lvl in levels if lvl.strike < self.S and lvl.put_gex < 0]
        if below:
            return min(below, key=lambda lvl: lvl.put_gex).strike
        return 0.0

    def _find_call_wall(self, levels: List[GEXLevel]) -> float:
        """Find the strike with largest call gamma above spot."""
        above = [lvl for lvl in levels if lvl.strike > self.S and lvl.call_gex < 0]
        if above:
            return min(above, key=lambda lvl: lvl.call_gex).strike
        return 0.0

    def _determine_regime(
        self, total_gex: float, flip_level: float
    ) -> VolatilityRegime:
        """Determine the current volatility regime."""
        proximity = abs(self.S - flip_level) / self.S
        if proximity < 0.005:  # Within 0.5% of flip
            return VolatilityRegime.TRANSITIONAL
        if total_gex > 0 or self.S < flip_level:
            return VolatilityRegime.SUPPRESSED
        return VolatilityRegime.AMPLIFIED


# ═══════════════════════════════════════════════════════════════════════════
# DEALER HEDGING FLOW TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class DealerHedgingTracker:
    """
    Estimates dealer hedging flows based on GEX profile changes.

    Key insights from 250 options insights (BARREN WUFFET doctrine):
      - GEX regime determines whether dealers are WITH or AGAINST price
      - Large OI concentrations at strikes → pinning effect near OPEX
      - OPEX week: gamma decay accelerates → dealers unwind hedges
      - 0DTE options: massive intraday gamma → dealers must hedge aggressively
    """

    def __init__(self, profile: GEXProfile):
        self.profile = profile

    def estimate_hedging_flow(self, price_change_pct: float) -> DealerExposure:
        """
        Estimate required dealer hedging for a given price change.

        Args:
            price_change_pct: Expected price change in percent (e.g., 1.0 = +1%)

        Returns:
            DealerExposure with estimated hedging requirements
        """
        total_gamma = self.profile.total_gex
        price_change = self.profile.underlying_price * price_change_pct / 100

        # Net delta change = gamma * price_change
        delta_change = total_gamma * price_change / self.profile.underlying_price

        # Estimated shares to hedge
        shares = int(abs(delta_change))

        # Determine hedging direction
        if total_gamma > 0:
            # Positive gamma: dealers sell into rallies, buy into dips
            if price_change > 0:
                pressure = "selling"
            else:
                pressure = "buying"
        else:
            # Negative gamma: dealers buy into rallies, sell into dips
            if price_change > 0:
                pressure = "buying"
            else:
                pressure = "selling"

        return DealerExposure(
            net_delta=round(delta_change, 2),
            net_gamma=round(total_gamma, 2),
            hedging_pressure=pressure,
            estimated_shares=shares,
        )

    def get_support_resistance(self) -> Dict[str, List[float]]:
        """
        Identify support/resistance levels from GEX profile.

        High positive GEX = resistance (dealers sell rallies)
        High negative GEX = support (dealers buy dips)
        """
        support = []
        resistance = []

        for level in self.profile.levels:
            if level.net_gex > 0 and level.strike > self.profile.underlying_price:
                resistance.append(level.strike)
            elif level.net_gex < 0 and level.strike < self.profile.underlying_price:
                support.append(level.strike)

        # Sort and take top 5
        support.sort(reverse=True)
        resistance.sort()

        return {
            "support": support[:5],
            "resistance": resistance[:5],
            "put_wall": self.profile.put_wall,
            "call_wall": self.profile.call_wall,
            "flip_level": self.profile.flip_level,
        }

    def opex_week_analysis(self, days_to_opex: int) -> Dict:
        """
        Analyze expected behavior as OPEX approaches.

        As expiration nears:
          - Gamma increases dramatically for ATM options
          - Theta decay accelerates (T-3 to T-0 is critical)
          - Pinning effect intensifies at max pain / max OI strikes
          - 0DTE gamma can dominate intraday moves
        """
        urgency = "LOW"
        if days_to_opex <= 2:
            urgency = "CRITICAL"
        elif days_to_opex <= 5:
            urgency = "HIGH"
        elif days_to_opex <= 10:
            urgency = "MODERATE"

        # Gamma amplification factor (rough approximation)
        gamma_amp = 1.0
        if days_to_opex > 0:
            gamma_amp = math.sqrt(30 / days_to_opex)  # Gamma ~ 1/sqrt(T)

        return {
            "days_to_opex": days_to_opex,
            "urgency": urgency,
            "gamma_amplification": round(gamma_amp, 2),
            "expected_pinning": days_to_opex <= 2,
            "max_gamma_strike": self.profile.max_gamma_strike,
            "regime": self.profile.regime.value,
            "recommendation": self._opex_recommendation(days_to_opex),
        }

    def _opex_recommendation(self, days_to_opex: int) -> str:
        """Generate recommendation based on OPEX proximity."""
        if days_to_opex <= 1:
            return ("0-1 DTE: Extreme gamma. Expect pinning near max OI strikes. "
                    "Avoid new positions. Close tested spreads.")
        elif days_to_opex <= 3:
            return ("2-3 DTE: Theta accelerating. Roll or close credit spreads. "
                    "Watch for gamma squeeze near close.")
        elif days_to_opex <= 7:
            return ("4-7 DTE: Monitor GEX profile for shifts. "
                    "Consider rolling to next cycle. Reduce notional exposure.")
        return "7+ DTE: Standard management. Monitor GEX for regime changes."


# ═══════════════════════════════════════════════════════════════════════════
# VOLATILITY SURFACE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class VolatilitySurfaceAnalyzer:
    """
    Analyze the IV surface for skew, term structure, and anomalies.

    From BARREN WUFFET insights:
      - IV skew reveals market fear/greed for a stock
      - Term structure inversions signal short-term event risk
      - IV crush after earnings is systematic and tradable
      - Put skew > call skew = market pricing tail risk to downside
    """

    @staticmethod
    def compute_skew(oi_chain: List[OptionOI], atm_strike: float) -> Dict:
        """Compute IV skew metrics."""
        calls = {oi.strike: oi.call_iv for oi in oi_chain if oi.call_iv > 0}
        puts = {oi.strike: oi.put_iv for oi in oi_chain if oi.put_iv > 0}

        atm_call_iv = calls.get(atm_strike, 0)
        atm_put_iv = puts.get(atm_strike, 0)
        atm_iv = (atm_call_iv + atm_put_iv) / 2 if (atm_call_iv + atm_put_iv) > 0 else 0

        # 25-delta put IV vs 25-delta call IV
        otm_puts = {k: v for k, v in puts.items() if k < atm_strike}
        otm_calls = {k: v for k, v in calls.items() if k > atm_strike}

        # Risk reversal = 25d call IV - 25d put IV
        # Negative = market pricing more put premium (bearish skew)
        put_25d_iv = max(otm_puts.values()) if otm_puts else 0
        call_25d_iv = min(otm_calls.values()) if otm_calls else 0
        risk_reversal = call_25d_iv - put_25d_iv

        # Butterfly spread = (25d put IV + 25d call IV) / 2 - ATM IV
        # Positive = wings are expensive relative to ATM (fat tails priced in)
        butterfly = ((put_25d_iv + call_25d_iv) / 2 - atm_iv) if atm_iv > 0 else 0

        return {
            "atm_iv": round(atm_iv, 4),
            "risk_reversal": round(risk_reversal, 4),
            "butterfly": round(butterfly, 4),
            "put_skew": "steep" if risk_reversal < -0.02 else "flat",
            "tail_risk_priced": butterfly > 0.01,
        }

    @staticmethod
    def detect_iv_crush_opportunity(
        current_iv: float, historical_iv_mean: float,
        event_date_dte: int
    ) -> Dict:
        """
        Detect IV crush setup (e.g., pre-earnings).

        IV typically:
          - Rises 2-3 weeks before earnings
          - Peaks day before announcement
          - Crashes 30-70% day after (IV crush)
          - Provides systematic short vol edge
        """
        iv_percentile = (current_iv / historical_iv_mean - 1) * 100
        crush_expected = current_iv > historical_iv_mean * 1.3

        if event_date_dte <= 0:
            return {"opportunity": False, "reason": "Event has passed"}

        sizing = "FULL"
        if iv_percentile > 80:
            sizing = "OVERSIZED — extreme IV"
        elif iv_percentile < 30:
            sizing = "UNDERSIZED — IV not elevated"

        return {
            "current_iv": round(current_iv, 4),
            "historical_mean": round(historical_iv_mean, 4),
            "iv_premium_pct": round(iv_percentile, 2),
            "crush_expected": crush_expected,
            "days_to_event": event_date_dte,
            "recommended_strategy": (
                "Short Iron Condor or Short Straddle (collect high premium)"
                if crush_expected else
                "Calendar Spread (buy post-event, sell pre-event expiry)"
            ),
            "sizing": sizing,
            "expected_crush_range": f"{30 + int(iv_percentile * 0.3)}%-{50 + int(iv_percentile * 0.2)}%",
        }


# ═══════════════════════════════════════════════════════════════════════════
# PUT/CALL RATIO ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class PutCallAnalyzer:
    """
    Track put/call ratios for sentiment signals.

    BARREN WUFFET doctrine:
      - Equity P/C > 0.7 = elevated fear → contrarian bullish
      - Equity P/C < 0.4 = complacency → contrarian warning
      - Index P/C matters less (hedging noise)
      - 0DTE P/C is most noise; weekly is more signal
      - Volume P/C vs OI P/C divergence = institutional positioning shift
    """

    @staticmethod
    def analyze(
        put_volume: int, call_volume: int,
        put_oi: int, call_oi: int,
        historical_avg_pc: float = 0.55,
    ) -> Dict:
        """Compute put/call ratio metrics and sentiment."""
        volume_pc = put_volume / call_volume if call_volume > 0 else 0
        oi_pc = put_oi / call_oi if call_oi > 0 else 0

        # Sentiment classification
        if volume_pc > 0.8:
            sentiment = "EXTREME_FEAR"
            contrarian = "BULLISH"
        elif volume_pc > 0.65:
            sentiment = "FEAR"
            contrarian = "MILDLY_BULLISH"
        elif volume_pc < 0.35:
            sentiment = "EXTREME_GREED"
            contrarian = "BEARISH"
        elif volume_pc < 0.5:
            sentiment = "GREED"
            contrarian = "MILDLY_BEARISH"
        else:
            sentiment = "NEUTRAL"
            contrarian = "NEUTRAL"

        # Volume vs OI divergence
        divergence = abs(volume_pc - oi_pc) > 0.15

        return {
            "volume_pc_ratio": round(volume_pc, 3),
            "oi_pc_ratio": round(oi_pc, 3),
            "historical_avg": historical_avg_pc,
            "sentiment": sentiment,
            "contrarian_signal": contrarian,
            "volume_oi_divergence": divergence,
            "divergence_note": (
                "Volume P/C diverging from OI P/C — watch for institutional "
                "repositioning" if divergence else "No significant divergence"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🐺 BARREN WUFFET — Gamma Exposure & Dealer Flow Tracker v2.7.0")
    print("=" * 65)

    # Demo: Build a synthetic GEX profile
    engine = GammaExposureEngine("SPY", 520.0)

    # Simulated OI data
    demo_oi = []
    for strike in range(490, 560, 5):
        # Higher OI near ATM
        dist = abs(strike - 520)
        oi_factor = max(1, 30 - dist) * 1000
        gamma_val = 0.003 * math.exp(-dist**2 / 200)

        demo_oi.append(OptionOI(
            strike=strike, expiry="2025-07-18",
            call_oi=int(oi_factor * 0.7), put_oi=int(oi_factor * 1.1),
            call_gamma=gamma_val, put_gamma=gamma_val * 0.9,
        ))

    engine.add_oi_data(demo_oi)
    profile = engine.compute_gex_profile()

    print(f"\nGEX Profile: {profile.symbol}")
    print(f"  Total GEX: ${profile.total_gex:,.0f}")
    print(f"  Flip Level: ${profile.flip_level:.2f}")
    print(f"  Max Gamma: ${profile.max_gamma_strike}")
    print(f"  Regime: {profile.regime.value}")
    print(f"  Put Wall: ${profile.put_wall}")
    print(f"  Call Wall: ${profile.call_wall}")

    # Dealer hedging
    tracker = DealerHedgingTracker(profile)
    flow = tracker.estimate_hedging_flow(price_change_pct=1.0)
    print(f"\n  For +1% move:")
    print(f"    Dealer pressure: {flow.hedging_pressure}")
    print(f"    Est. shares: {flow.estimated_shares:,}")

    sr = tracker.get_support_resistance()
    print(f"\n  Support levels: {sr['support']}")
    print(f"  Resistance levels: {sr['resistance']}")

    opex = tracker.opex_week_analysis(days_to_opex=3)
    print(f"\n  OPEX Analysis ({opex['days_to_opex']} DTE):")
    print(f"    Urgency: {opex['urgency']}")
    print(f"    Gamma Amp: {opex['gamma_amplification']}x")
    print(f"    Recommendation: {opex['recommendation']}")

    # P/C Ratio
    pc = PutCallAnalyzer.analyze(
        put_volume=1_200_000, call_volume=1_500_000,
        put_oi=5_000_000, call_oi=4_500_000,
    )
    print(f"\n  Put/Call: {pc['volume_pc_ratio']} | Sentiment: {pc['sentiment']}")
    print(f"  Contrarian Signal: {pc['contrarian_signal']}")
