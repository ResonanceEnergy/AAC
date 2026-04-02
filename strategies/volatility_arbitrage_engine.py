"""
Volatility Arbitrage & Term Structure Engine — BARREN WUFFET v2.7.0
====================================================================
Advanced volatility trading: IV vs RV convergence, term structure trades,
vol surface arbitrage, variance swap replication, and volatility regime detection.

From BARREN WUFFET Insights 456-535:
  - IV consistently overprices RV by 2-4 vol points (variance risk premium)
  - Term structure inversions signal short-term stress and are mean-reverting
  - Skew extremes (risk reversals) revert faster than ATM vol
  - Vol-of-vol (VVIX) predicts VIX regime changes 1-3 days early
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS & TYPES
# ═══════════════════════════════════════════════════════════════════════════

class VolRegime(Enum):
    """VolRegime class."""
    LOW = "low_vol"                    # VIX < 13
    NORMAL = "normal_vol"              # VIX 13-20
    ELEVATED = "elevated_vol"          # VIX 20-30
    HIGH = "high_vol"                  # VIX 30-40
    EXTREME = "extreme_vol"            # VIX > 40

class TermStructure(Enum):
    """TermStructure class."""
    CONTANGO = "contango"              # Front < back (normal)
    FLAT = "flat"                      # ~equal
    BACKWARDATION = "backwardation"    # Front > back (fear)

class SkewRegime(Enum):
    """SkewRegime class."""
    NORMAL = "normal"                  # Moderate put skew
    STEEP = "steep"                    # Elevated put skew (fear)
    FLAT = "flat"                      # Low skew (complacency)
    INVERTED = "inverted"              # Call skew > put skew (squeeze risk)


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VolSurface:
    """IV surface snapshot."""
    symbol: str
    underlying_price: float
    timestamp: str = ""
    # ATM IV by expiry (DTE -> IV)
    atm_term_structure: Dict[int, float] = field(default_factory=dict)
    # Skew by expiry (DTE -> {delta: IV})
    skew_by_expiry: Dict[int, Dict[float, float]] = field(default_factory=dict)
    # Realized vol windows
    rv_5d: float = 0.0
    rv_10d: float = 0.0
    rv_20d: float = 0.0
    rv_30d: float = 0.0
    rv_60d: float = 0.0

    @property
    def front_iv(self) -> float:
        """Nearest expiry ATM IV."""
        if not self.atm_term_structure:
            return 0
        min_dte = min(self.atm_term_structure.keys())
        return self.atm_term_structure[min_dte]

    @property
    def back_iv(self) -> float:
        """Furthest expiry ATM IV."""
        if not self.atm_term_structure:
            return 0
        max_dte = max(self.atm_term_structure.keys())
        return self.atm_term_structure[max_dte]


@dataclass
class VRPSignal:
    """Variance Risk Premium signal."""
    iv_current: float
    rv_current: float
    vrp: float                     # IV - RV (positive = premium to sell)
    vrp_percentile: float          # Historical percentile
    z_score: float                 # Standardized VRP
    signal: str                    # "SELL_VOL", "BUY_VOL", "NEUTRAL"
    confidence: float              # 0-100
    recommended_strategy: str


@dataclass
class TermStructureSignal:
    """Term structure trade signal."""
    front_dte: int
    front_iv: float
    back_dte: int
    back_iv: float
    spread: float                  # back - front
    spread_pct: float              # Relative spread
    structure: TermStructure
    signal: str                    # "SELL_FRONT_BUY_BACK", etc.
    recommended_strategy: str


# ═══════════════════════════════════════════════════════════════════════════
# VARIANCE RISK PREMIUM ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class VarianceRiskPremiumEngine:
    """
    Compute and trade the Variance Risk Premium (VRP).

    The VRP is one of the most robust and well-documented risk premia:
      - IV systematically overstates actual RV (insurance premium)
      - Average VRP = 2-5 vol points (200-500bps annualized)
      - VRP is regime-dependent: larger in high-vol, sometimes negative in low-vol
      - Best harvested via short straddles, iron condors, or variance swaps

    Key BARREN WUFFET insights:
      - VRP > 3 vol points + IVR > 50% = high-confidence sell
      - VRP < 0 (RV > IV) = STOP selling vol
      - Use 20-day RV as core comparison to 30-day IV
      - VVIX > 100 = vol regime shifting, reduce positions
    """

    def __init__(self, vrp_history: Optional[List[float]] = None):
        self.vrp_history = vrp_history or []

    def compute_vrp(
        self, iv_30d: float, rv_20d: float, rv_5d: float
    ) -> VRPSignal:
        """
        Compute VRP signal.

        Args:
            iv_30d: 30-day implied volatility
            rv_20d: 20-day realized volatility
            rv_5d: 5-day realized volatility (short-term trend)
        """
        vrp = iv_30d - rv_20d
        vrp_recent = iv_30d - rv_5d

        # Percentile vs history
        if self.vrp_history:
            below = sum(1 for v in self.vrp_history if v < vrp)
            percentile = below / len(self.vrp_history) * 100
            mean = sum(self.vrp_history) / len(self.vrp_history)
            std = math.sqrt(sum((v - mean)**2 for v in self.vrp_history) / len(self.vrp_history))
            z = (vrp - mean) / std if std > 0 else 0
        else:
            percentile = 50
            z = 0

        # Signal logic
        if vrp > 4 and percentile > 70:
            signal = "SELL_VOL"
            confidence = min(95, 60 + percentile * 0.3)
            strategy = "Short Straddle or Iron Condor (high VRP = edge in selling)"
        elif vrp > 2 and percentile > 50:
            signal = "SELL_VOL"
            confidence = min(80, 50 + percentile * 0.2)
            strategy = "Iron Condor or Short Strangle (moderate VRP)"
        elif vrp < 0:
            signal = "BUY_VOL"
            confidence = min(85, 60 + abs(vrp) * 5)
            strategy = "Long Straddle or Calendar Spread (RV exceeding IV)"
        elif vrp < 1:
            signal = "NEUTRAL"
            confidence = 40
            strategy = "No edge — stand aside or use calendars"
        else:
            signal = "NEUTRAL"
            confidence = 50
            strategy = "Small position Iron Condor (marginal VRP)"

        # Override if short-term RV is spiking
        if rv_5d > iv_30d * 1.2:
            signal = "BUY_VOL"
            confidence = max(confidence, 70)
            strategy = "RV spiking above IV — buy vol or stand aside"

        return VRPSignal(
            iv_current=round(iv_30d, 4),
            rv_current=round(rv_20d, 4),
            vrp=round(vrp, 4),
            vrp_percentile=round(percentile, 2),
            z_score=round(z, 2),
            signal=signal,
            confidence=round(confidence, 2),
            recommended_strategy=strategy,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TERM STRUCTURE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class TermStructureAnalyzer:
    """
    Analyze and trade volatility term structure.

    Term structure concepts:
      - Contango (normal): front-month IV < back-month IV
        → Time value of uncertainty increases with time
      - Backwardation (stressed): front-month IV > back-month IV
        → Market pricing immediate risk (catalyst, event, crash)
      - Flat: similar IV across terms → transitional state

    Trade signals:
      - Deep backwardation → sell front/buy back calendar (mean reversion)
      - Steep contango → buy front/sell back (if event expected)
      - Term structure slope change → regime shift signal
    """

    def __init__(self, surface: VolSurface):
        self.surface = surface

    def analyze_structure(self) -> Dict:
        """Analyze the full term structure."""
        ts = self.surface.atm_term_structure
        if len(ts) < 2:
            return {"structure": "insufficient_data"}

        sorted_dtes = sorted(ts.keys())
        front = ts[sorted_dtes[0]]
        back = ts[sorted_dtes[-1]]
        spread = back - front
        spread_pct = spread / front * 100 if front > 0 else 0

        # Classify
        if spread_pct > 5:
            structure = TermStructure.CONTANGO
        elif spread_pct < -5:
            structure = TermStructure.BACKWARDATION
        else:
            structure = TermStructure.FLAT

        # Check for hump (belly higher than front and back)
        has_hump = False
        if len(sorted_dtes) >= 3:
            mid_dte = sorted_dtes[len(sorted_dtes) // 2]
            mid_iv = ts[mid_dte]
            if mid_iv > front and mid_iv > back:
                has_hump = True

        return {
            "structure": structure.value,
            "front_dte": sorted_dtes[0],
            "front_iv": round(front, 4),
            "back_dte": sorted_dtes[-1],
            "back_iv": round(back, 4),
            "spread": round(spread, 4),
            "spread_pct": round(spread_pct, 2),
            "has_hump": has_hump,
            "num_tenors": len(sorted_dtes),
        }

    def generate_signal(self) -> TermStructureSignal:
        """Generate a term structure trade signal."""
        analysis = self.analyze_structure()
        if analysis.get("structure") == "insufficient_data":
            return TermStructureSignal(
                front_dte=0, front_iv=0, back_dte=0, back_iv=0,
                spread=0, spread_pct=0,
                structure=TermStructure.FLAT,
                signal="NO_SIGNAL",
                recommended_strategy="Insufficient data",
            )

        structure = TermStructure(analysis["structure"])
        spread_pct = analysis["spread_pct"]

        if structure == TermStructure.BACKWARDATION and spread_pct < -10:
            signal = "SELL_FRONT_BUY_BACK"
            strategy = (
                "Calendar Spread: Sell front-month straddle/strangle, "
                "buy back-month. Backwardation mean-reverts → profit."
            )
        elif structure == TermStructure.BACKWARDATION:
            signal = "MILD_SELL_FRONT"
            strategy = "Mild backwardation — small calendar or stand aside"
        elif structure == TermStructure.CONTANGO and spread_pct > 15:
            signal = "BUY_FRONT_SELL_BACK"
            strategy = (
                "Reverse Calendar: buy front-month, sell back-month. "
                "Only if expecting vol event in front month."
            )
        else:
            signal = "NEUTRAL"
            strategy = "Normal contango — standard strategies apply"

        return TermStructureSignal(
            front_dte=analysis["front_dte"],
            front_iv=analysis["front_iv"],
            back_dte=analysis["back_dte"],
            back_iv=analysis["back_iv"],
            spread=analysis["spread"],
            spread_pct=spread_pct,
            structure=structure,
            signal=signal,
            recommended_strategy=strategy,
        )


# ═══════════════════════════════════════════════════════════════════════════
# SKEW ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class SkewAnalyzer:
    """
    Analyze IV skew for trade signals and market sentiment.

    Skew metrics:
      - 25Δ Risk Reversal = 25Δ call IV - 25Δ put IV
        → Negative = puts are expensive (bearish skew)
        → Very negative = extreme fear → contrarian buy signal
      - 25Δ Butterfly = (25Δ put IV + 25Δ call IV) / 2 - ATM IV
        → Positive = wings expensive (tail risk priced in)
      - Skew Z-score vs history → trade mean reversion
    """

    def __init__(self, surface: VolSurface, skew_history: Optional[List[float]] = None):
        self.surface = surface
        self.skew_history = skew_history or []

    def compute_risk_reversal(self, dte: int) -> Dict:
        """Compute 25-delta risk reversal for a given expiry."""
        skew = self.surface.skew_by_expiry.get(dte, {})
        if not skew:
            return {"risk_reversal": 0, "regime": "no_data"}

        # Find 25-delta points (or closest)
        put_25d = skew.get(-0.25, skew.get(-0.20, 0))
        call_25d = skew.get(0.25, skew.get(0.20, 0))
        atm = skew.get(0.50, skew.get(0.45, 0))

        risk_reversal = call_25d - put_25d
        butterfly = (put_25d + call_25d) / 2 - atm if atm > 0 else 0

        # Classify skew regime
        if risk_reversal < -0.05:
            regime = SkewRegime.STEEP
        elif risk_reversal > 0.02:
            regime = SkewRegime.INVERTED
        elif abs(risk_reversal) < 0.01:
            regime = SkewRegime.FLAT
        else:
            regime = SkewRegime.NORMAL

        # Z-score vs history
        z = 0
        if self.skew_history:
            mean = sum(self.skew_history) / len(self.skew_history)
            std = math.sqrt(sum((v - mean)**2 for v in self.skew_history) / len(self.skew_history))
            z = (risk_reversal - mean) / std if std > 0 else 0

        return {
            "risk_reversal": round(risk_reversal, 4),
            "butterfly": round(butterfly, 4),
            "regime": regime.value,
            "z_score": round(z, 2),
            "put_25d_iv": round(put_25d, 4),
            "call_25d_iv": round(call_25d, 4),
            "atm_iv": round(atm, 4),
        }

    def skew_trade_signal(self, dte: int) -> Dict:
        """Generate a skew-based trade signal."""
        rr = self.compute_risk_reversal(dte)
        z = rr.get("z_score", 0)
        regime = rr.get("regime", "normal")

        if z < -2:
            signal = "SELL_PUT_SKEW"
            strategy = ("Extreme put skew — sell put spreads or put ratio spreads. "
                        "Skew typically mean-reverts from Z < -2.")
            confidence = 80
        elif z > 2:
            signal = "BUY_PUT_SKEW"
            strategy = ("Flat/inverted skew — buy protective puts cheaply. "
                        "Complacency signal; skew will steepen.")
            confidence = 75
        elif regime == "steep" and z < -1:
            signal = "MILD_SELL_SKEW"
            strategy = "Elevated skew — consider selling put spreads or risk reversals"
            confidence = 60
        else:
            signal = "NEUTRAL"
            strategy = "Skew within normal bounds"
            confidence = 40

        return {
            **rr,
            "signal": signal,
            "strategy": strategy,
            "confidence": confidence,
        }


# ═══════════════════════════════════════════════════════════════════════════
# VIX / VVIX REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class VolRegimeDetector:
    """
    Detect and classify volatility regimes using VIX/VVIX.

    Regime hierarchy:
      VIX < 13: Low vol → sell far OTM, small premium, high POP
      VIX 13-20: Normal → standard strategies apply
      VIX 20-30: Elevated → increase position sizes for premium
      VIX 30-40: High → widen wings, reduce exposure, expect mean reversion
      VIX > 40: Extreme → buy vol for crash continuation, or wait for signal

    VVIX (vol of VIX):
      VVIX < 80: Low VIX volatility → VIX likely range-bound
      VVIX 80-100: Normal
      VVIX 100-120: Elevated → VIX may spike or collapse
      VVIX > 120: Extreme → regime change underway
    """

    @staticmethod
    def classify(vix: float, vvix: float = 0) -> Dict:
        """Classify current vol regime."""
        # VIX regime
        if vix < 13:
            regime = VolRegime.LOW
            vix_note = "Low vol: tight ranges, small premiums, high POP"
        elif vix < 20:
            regime = VolRegime.NORMAL
            vix_note = "Normal vol: standard strategies, balanced risk/reward"
        elif vix < 30:
            regime = VolRegime.ELEVATED
            vix_note = "Elevated: rich premiums, increase size, widen strikes"
        elif vix < 40:
            regime = VolRegime.HIGH
            vix_note = "High vol: extreme premiums, reduce notional, expect mean reversion"
        else:
            regime = VolRegime.EXTREME
            vix_note = "Extreme: crisis mode, buy vol or stand aside"

        # Strategy adjustments
        strategies = {
            VolRegime.LOW: [
                "Sell narrow iron condors (10-15 delta)",
                "Calendar spreads (benefit from low IV expanding)",
                "Butterfly spreads (cheap entry, pin plays)",
            ],
            VolRegime.NORMAL: [
                "Standard iron condors (15-20 delta)",
                "Wheel strategy on quality stocks",
                "Covered calls / bull put spreads",
            ],
            VolRegime.ELEVATED: [
                "Wider iron condors (20-25 delta, bigger credit)",
                "Aggressive CSPs on crashed quality names",
                "Short strangles on liquid underlyings",
            ],
            VolRegime.HIGH: [
                "Wide iron condors with reduced size",
                "Jade Lizards for no-upside-risk plays",
                "Put ratio spreads (sell fear at extremes)",
            ],
            VolRegime.EXTREME: [
                "Cash / minimal exposure",
                "Long vol via VIX calls or put debit spreads",
                "Wait for VIX to peak, then sell aggressively",
            ],
        }

        # VVIX analysis
        vvix_note = ""
        if vvix > 0:
            if vvix < 80:
                vvix_note = "VVIX low: VIX likely stable, range-bound"
            elif vvix < 100:
                vvix_note = "VVIX normal: typical VIX movement expected"
            elif vvix < 120:
                vvix_note = "VVIX elevated: watch for VIX spike or crash"
            else:
                vvix_note = "VVIX extreme: VIX regime change likely imminent"

        # VIX/VVIX ratio (when available)
        vix_vvix_ratio = vix / vvix if vvix > 0 else None

        return {
            "vix": vix,
            "vvix": vvix if vvix > 0 else None,
            "regime": regime.value,
            "note": vix_note,
            "vvix_note": vvix_note,
            "vix_vvix_ratio": round(vix_vvix_ratio, 3) if vix_vvix_ratio else None,
            "recommended_strategies": strategies.get(regime, []),
            "sizing_adjustment": {
                VolRegime.LOW: "standard_or_reduced",
                VolRegime.NORMAL: "standard",
                VolRegime.ELEVATED: "increase_25pct",
                VolRegime.HIGH: "reduce_50pct",
                VolRegime.EXTREME: "minimal_or_zero",
            }.get(regime, "standard"),
        }


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🐺 BARREN WUFFET — Volatility Arbitrage Engine v2.7.0")
    print("=" * 60)

    # Demo: VRP
    vrp_engine = VarianceRiskPremiumEngine(
        vrp_history=[2.1, 3.5, 1.8, 4.2, 2.9, 3.1, 0.5, 5.0, 3.8, 2.4]
    )
    vrp = vrp_engine.compute_vrp(iv_30d=0.22, rv_20d=0.17, rv_5d=0.15)
    print(f"\nVRP Signal:")
    print(f"  IV: {vrp.iv_current:.2%} | RV: {vrp.rv_current:.2%}")
    print(f"  VRP: {vrp.vrp:.4f} ({vrp.vrp_percentile:.0f}th %ile, Z={vrp.z_score})")
    print(f"  Signal: {vrp.signal} ({vrp.confidence:.0f}% confidence)")
    print(f"  Strategy: {vrp.recommended_strategy}")

    # Demo: Term Structure
    surface = VolSurface(
        symbol="SPY", underlying_price=520,
        atm_term_structure={7: 0.18, 14: 0.19, 30: 0.20, 60: 0.21, 90: 0.215},
        rv_20d=0.16,
    )
    ts_analyzer = TermStructureAnalyzer(surface)
    ts = ts_analyzer.analyze_structure()
    print(f"\nTerm Structure:")
    for k, v in ts.items():
        print(f"  {k}: {v}")

    ts_signal = ts_analyzer.generate_signal()
    print(f"  Signal: {ts_signal.signal}")
    print(f"  Strategy: {ts_signal.recommended_strategy}")

    # Demo: Regime
    regime = VolRegimeDetector.classify(vix=22.5, vvix=105)
    print(f"\nVol Regime:")
    print(f"  VIX: {regime['vix']} | VVIX: {regime['vvix']}")
    print(f"  Regime: {regime['regime']}")
    print(f"  Note: {regime['note']}")
    print(f"  VVIX: {regime['vvix_note']}")
    print(f"  Sizing: {regime['sizing_adjustment']}")
    print(f"  Strategies:")
    for s in regime['recommended_strategies']:
        print(f"    → {s}")
