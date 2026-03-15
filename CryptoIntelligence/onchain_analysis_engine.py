"""
On-Chain Analysis Engine — BARREN WUFFET v2.7.0
=================================================
Deep on-chain metrics analysis for BTC, ETH, and major L1s.
Provides macro cycle positioning, HODLer behavior analysis,
exchange flow monitoring, and supply dynamics tracking.

From BARREN WUFFET Insights (601-640):
  - MVRV Z-Score >7 = historically overbought territory
  - SOPR <1 during uptrend = strong accumulation signal
  - Exchange outflows > inflows for 30d+ = bullish supply squeeze
  - NUPL >0.75 = euphoria zone, historically precedes 40%+ drawdowns
  - Long-term holder supply ratio >68% = deep accumulation phase
  - Realized price acts as dynamic bull/bear dividing line
  - NVT Signal (smoothed) crossing above 150 = overvaluation warning
  - Coin days destroyed spikes = long-dormant whales moving
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

class MarketCyclePhase(Enum):
    """MarketCyclePhase class."""
    ACCUMULATION = "accumulation"       # Bottom forming, smart money buys
    EARLY_BULL = "early_bull"           # Breakout confirmed, momentum building
    MID_BULL = "mid_bull"               # Mainstream adoption, strong trend
    LATE_BULL = "late_bull"             # Euphoria building, divergences
    EUPHORIA = "euphoria"               # Blow-off top zone
    DISTRIBUTION = "distribution"       # Top forming, smart money exits
    EARLY_BEAR = "early_bear"           # Sell-off beginning
    CAPITULATION = "capitulation"       # Panic selling, max fear

class OnChainSignalStrength(Enum):
    """OnChainSignalStrength class."""
    SCREAMING_BUY = "screaming_buy"
    BUY = "buy"
    LEAN_BUY = "lean_buy"
    NEUTRAL = "neutral"
    LEAN_SELL = "lean_sell"
    SELL = "sell"
    SCREAMING_SELL = "screaming_sell"


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MVRVData:
    """Market Value to Realized Value metrics."""
    market_cap: float
    realized_cap: float
    mvrv_ratio: float
    mvrv_z_score: float
    signal: OnChainSignalStrength
    percentile: float  # historical percentile
    notes: List[str] = field(default_factory=list)


@dataclass
class SOPRData:
    """Spent Output Profit Ratio."""
    sopr: float
    adjusted_sopr: float  # Filters coinbase txns
    entity_adjusted_sopr: float  # Filters change outputs
    sth_sopr: float  # Short-term holder
    lth_sopr: float  # Long-term holder
    signal: OnChainSignalStrength
    notes: List[str] = field(default_factory=list)


@dataclass
class NUPLData:
    """Net Unrealized Profit/Loss."""
    nupl: float
    phase: str  # "capitulation", "hope/fear", "optimism", "belief", "euphoria"
    signal: OnChainSignalStrength
    notes: List[str] = field(default_factory=list)


@dataclass
class ExchangeFlowData:
    """Exchange inflow/outflow metrics."""
    net_flow_24h: float  # Negative = outflows (bullish)
    net_flow_7d: float
    net_flow_30d: float
    exchange_balance: float
    exchange_balance_pct: float  # % of total supply on exchanges
    trend: str  # "accumulating", "distributing", "neutral"
    signal: OnChainSignalStrength
    notes: List[str] = field(default_factory=list)


@dataclass
class SupplyDynamics:
    """Supply age and distribution metrics."""
    total_supply: float
    circulating_supply: float
    illiquid_supply_pct: float
    lth_supply_pct: float  # Long-term holder %
    sth_supply_pct: float  # Short-term holder %
    supply_shock_ratio: float  # Illiquid / liquid supply
    last_active_1yr_plus_pct: float  # % dormant >1yr
    signal: OnChainSignalStrength
    notes: List[str] = field(default_factory=list)


@dataclass
class OnChainDashboard:
    """Combined on-chain analysis dashboard."""
    timestamp: str
    asset: str
    price: float
    realized_price: float
    mvrv: MVRVData
    sopr: SOPRData
    nupl: NUPLData
    exchange_flows: ExchangeFlowData
    supply: SupplyDynamics
    cycle_phase: MarketCyclePhase
    composite_score: float  # -100 to +100
    composite_signal: OnChainSignalStrength


# ═══════════════════════════════════════════════════════════════════════════
# MVRV ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class MVRVAnalyzer:
    """
    Market Value to Realized Value analysis.
    
    MVRV = Market Cap / Realized Cap
    Realized Cap = sum of each UTXO valued at its last-moved price
    
    Key thresholds (BTC historical):
      MVRV > 3.7 → extreme overvalue (sell zone)
      MVRV Z > 7  → blow-off top territory
      MVRV 1-2.4  → fair value range
      MVRV < 1    → undervalue (buy zone, market below avg cost basis)
    """

    THRESHOLDS = {
        "extreme_overvalue": 3.7,
        "overvalue": 2.4,
        "fair_high": 1.8,
        "fair_low": 1.0,
        "undervalue": 0.8,
        "extreme_undervalue": 0.5,
    }

    Z_THRESHOLDS = {
        "blow_off_top": 7.0,
        "overheated": 5.0,
        "elevated": 3.0,
        "neutral_high": 1.5,
        "neutral_low": -0.5,
        "undervalued": -1.0,
        "capitulation": -2.0,
    }

    @classmethod
    def analyze(
        cls, market_cap: float, realized_cap: float,
        historical_mvrv_mean: float = 1.8,
        historical_mvrv_std: float = 0.9,
    ) -> MVRVData:
        """Compute MVRV ratio and Z-score with signal."""
        mvrv = market_cap / realized_cap if realized_cap > 0 else 0
        z_score = (mvrv - historical_mvrv_mean) / historical_mvrv_std if historical_mvrv_std > 0 else 0

        notes = []
        if mvrv > cls.THRESHOLDS["extreme_overvalue"]:
            signal = OnChainSignalStrength.SCREAMING_SELL
            notes.append(f"MVRV {mvrv:.2f} — EXTREME overvaluation. Top probable.")
        elif mvrv > cls.THRESHOLDS["overvalue"]:
            signal = OnChainSignalStrength.SELL
            notes.append(f"MVRV {mvrv:.2f} — Market significantly above realized value.")
        elif mvrv > cls.THRESHOLDS["fair_high"]:
            signal = OnChainSignalStrength.LEAN_SELL
            notes.append(f"MVRV {mvrv:.2f} — Upper fair value. Begin scaling out.")
        elif mvrv > cls.THRESHOLDS["fair_low"]:
            signal = OnChainSignalStrength.NEUTRAL
            notes.append(f"MVRV {mvrv:.2f} — Fair value range.")
        elif mvrv > cls.THRESHOLDS["undervalue"]:
            signal = OnChainSignalStrength.BUY
            notes.append(f"MVRV {mvrv:.2f} — Below aggregate cost basis. Accumulate.")
        else:
            signal = OnChainSignalStrength.SCREAMING_BUY
            notes.append(f"MVRV {mvrv:.2f} — Deep undervalue. Generational buying opportunity.")

        if z_score > cls.Z_THRESHOLDS["blow_off_top"]:
            notes.append(f"Z-Score {z_score:.1f} — BLOW-OFF TOP territory!")

        # Crude percentile estimate from Z-score (normal dist approximation)
        percentile = min(100, max(0, 50 + z_score * 15))

        return MVRVData(
            market_cap=market_cap,
            realized_cap=realized_cap,
            mvrv_ratio=round(mvrv, 3),
            mvrv_z_score=round(z_score, 2),
            signal=signal,
            percentile=round(percentile, 1),
            notes=notes,
        )


# ═══════════════════════════════════════════════════════════════════════════
# SOPR ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class SOPRAnalyzer:
    """
    Spent Output Profit Ratio analysis.
    
    SOPR = Realized Value / Value at Creation for spent outputs
      SOPR > 1 → coins moved at a profit
      SOPR = 1 → coins moved at breakeven (key level!)
      SOPR < 1 → coins moved at a loss
    
    Bull market pattern: SOPR bounces off 1.0 (support)
    Bear market pattern: SOPR rejected at 1.0 (resistance)
    """

    @classmethod
    def analyze(
        cls, sopr: float, adjusted_sopr: float,
        entity_sopr: float, sth_sopr: float, lth_sopr: float,
        market_trend: str = "unknown",
    ) -> SOPRData:
        """Analyze SOPR with context."""
        notes = []

        # Overall signal from adjusted SOPR
        if adjusted_sopr > 1.05:
            signal = OnChainSignalStrength.LEAN_SELL
            notes.append("Profit-taking in progress — holders selling at profit.")
        elif adjusted_sopr > 1.0:
            if market_trend == "bull":
                signal = OnChainSignalStrength.NEUTRAL
                notes.append("SOPR >1 in uptrend — healthy profit realization.")
            else:
                signal = OnChainSignalStrength.LEAN_SELL
                notes.append("SOPR >1 in downtrend — relief rally, likely to fail at 1.0.")
        elif adjusted_sopr >= 0.97:
            if market_trend == "bull":
                signal = OnChainSignalStrength.BUY
                notes.append("SOPR testing 1.0 in uptrend — classic accumulation zone!")
            else:
                signal = OnChainSignalStrength.NEUTRAL
                notes.append("SOPR testing 1.0 in downtrend — possible support but watch for break.")
        else:
            signal = OnChainSignalStrength.SCREAMING_BUY if market_trend == "bull" else OnChainSignalStrength.BUY
            notes.append("SOPR <0.97 — coins moving at loss. Potential capitulation or accumulation.")

        # STH vs LTH divergence
        if sth_sopr < 0.95 and lth_sopr > 1.1:
            notes.append("STH underwater, LTH profitable — classic late accumulation.")
        elif sth_sopr > 1.1 and lth_sopr > 1.5:
            notes.append("Both groups in heavy profit — distribution risk elevated.")

        return SOPRData(
            sopr=round(sopr, 4),
            adjusted_sopr=round(adjusted_sopr, 4),
            entity_adjusted_sopr=round(entity_sopr, 4),
            sth_sopr=round(sth_sopr, 4),
            lth_sopr=round(lth_sopr, 4),
            signal=signal,
            notes=notes,
        )


# ═══════════════════════════════════════════════════════════════════════════
# NUPL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class NUPLAnalyzer:
    """
    Net Unrealized Profit/Loss analysis.
    
    NUPL = (Market Cap - Realized Cap) / Market Cap
    Ranges:
      < 0       → Capitulation (red)
      0 - 0.25  → Hope/Fear (orange)
      0.25-0.50 → Optimism/Anxiety (yellow)
      0.50-0.75 → Belief/Denial (green)
      > 0.75    → Euphoria/Greed (blue) — sell zone
    """

    BANDS = [
        (-999, 0.0, "capitulation", OnChainSignalStrength.SCREAMING_BUY),
        (0.0, 0.25, "hope_fear", OnChainSignalStrength.BUY),
        (0.25, 0.50, "optimism_anxiety", OnChainSignalStrength.LEAN_BUY),
        (0.50, 0.75, "belief_denial", OnChainSignalStrength.NEUTRAL),
        (0.75, 999, "euphoria_greed", OnChainSignalStrength.SCREAMING_SELL),
    ]

    @classmethod
    def analyze(cls, market_cap: float, realized_cap: float) -> NUPLData:
        """Compute NUPL and classify."""
        nupl = (market_cap - realized_cap) / market_cap if market_cap > 0 else 0

        phase = "unknown"
        signal = OnChainSignalStrength.NEUTRAL
        notes = []

        for low, high, name, sig in cls.BANDS:
            if low <= nupl < high:
                phase = name
                signal = sig
                break

        if nupl > 0.75:
            notes.append(f"NUPL {nupl:.2f} — EUPHORIA. Historically precedes 40%+ drawdowns.")
        elif nupl < 0:
            notes.append(f"NUPL {nupl:.2f} — CAPITULATION. Market below aggregate cost basis.")
        else:
            notes.append(f"NUPL {nupl:.2f} — Phase: {phase}.")

        return NUPLData(nupl=round(nupl, 3), phase=phase, signal=signal, notes=notes)


# ═══════════════════════════════════════════════════════════════════════════
# EXCHANGE FLOW ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class ExchangeFlowAnalyzer:
    """
    Monitor exchange inflows/outflows for supply dynamics.
    
    Key patterns:
      - Sustained outflows = accumulation (coins moving to cold storage)
      - Spike inflows = selling pressure incoming
      - Exchange balance declining = long-term bullish supply squeeze
      - Exchange balance % < 12% for BTC = critical supply squeeze zone
    """

    @classmethod
    def analyze(
        cls,
        inflows_24h: float, outflows_24h: float,
        inflows_7d: float, outflows_7d: float,
        inflows_30d: float, outflows_30d: float,
        exchange_balance: float, total_supply: float,
    ) -> ExchangeFlowData:
        """Analyze exchange flow dynamics."""
        net_24h = inflows_24h - outflows_24h
        net_7d = inflows_7d - outflows_7d
        net_30d = inflows_30d - outflows_30d
        balance_pct = (exchange_balance / total_supply * 100) if total_supply > 0 else 0

        notes = []

        # Determine trend
        if net_7d < 0 and net_30d < 0:
            trend = "accumulating"
            signal = OnChainSignalStrength.BUY
            notes.append("Sustained exchange outflows — coins moving to cold storage.")
        elif net_7d > 0 and net_30d > 0:
            trend = "distributing"
            signal = OnChainSignalStrength.SELL
            notes.append("Sustained exchange inflows — selling pressure building.")
        else:
            trend = "neutral"
            signal = OnChainSignalStrength.NEUTRAL
            notes.append("Mixed exchange flows — no clear trend.")

        # Supply squeeze check
        if balance_pct < 12:
            notes.append(f"Exchange balance {balance_pct:.1f}% — critical supply squeeze zone!")
            if signal == OnChainSignalStrength.BUY:
                signal = OnChainSignalStrength.SCREAMING_BUY

        # Spike detection (24h vs 7d average)
        avg_daily_7d = abs(net_7d) / 7
        if abs(net_24h) > avg_daily_7d * 3 and avg_daily_7d > 0:
            notes.append(f"SPIKE: 24h flow {net_24h:,.0f} is >3x daily average!")

        return ExchangeFlowData(
            net_flow_24h=round(net_24h, 2),
            net_flow_7d=round(net_7d, 2),
            net_flow_30d=round(net_30d, 2),
            exchange_balance=round(exchange_balance, 2),
            exchange_balance_pct=round(balance_pct, 2),
            trend=trend,
            signal=signal,
            notes=notes,
        )


# ═══════════════════════════════════════════════════════════════════════════
# SUPPLY DYNAMICS ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class SupplyDynamicsAnalyzer:
    """
    Analyze supply distribution and HODLer behavior.
    
    Key concepts:
      - Illiquid supply: coins that haven't moved in 6+ months
      - Supply shock ratio: illiquid / liquid supply
      - LTH (>155 days) vs STH (<155 days) behavior
      - HODL waves: age bands of UTXO set
    """

    @classmethod
    def analyze(
        cls,
        total_supply: float, circulating_supply: float,
        illiquid_supply: float, lth_supply: float, sth_supply: float,
        dormant_1yr_plus: float,
    ) -> SupplyDynamics:
        """Analyze supply dynamics."""
        illiquid_pct = (illiquid_supply / circulating_supply * 100) if circulating_supply > 0 else 0
        lth_pct = (lth_supply / circulating_supply * 100) if circulating_supply > 0 else 0
        sth_pct = (sth_supply / circulating_supply * 100) if circulating_supply > 0 else 0
        liquid_supply = circulating_supply - illiquid_supply
        shock_ratio = illiquid_supply / liquid_supply if liquid_supply > 0 else 0
        dormant_pct = (dormant_1yr_plus / circulating_supply * 100) if circulating_supply > 0 else 0

        notes = []

        # Signal logic
        if lth_pct > 68:
            signal = OnChainSignalStrength.SCREAMING_BUY
            notes.append(f"LTH supply {lth_pct:.1f}% — deep accumulation phase (>68%).")
        elif lth_pct > 60:
            signal = OnChainSignalStrength.BUY
            notes.append(f"LTH supply {lth_pct:.1f}% — strong hands dominating.")
        elif lth_pct < 50:
            signal = OnChainSignalStrength.SELL
            notes.append(f"LTH supply {lth_pct:.1f}% — distribution underway (<50%).")
        else:
            signal = OnChainSignalStrength.NEUTRAL
            notes.append(f"LTH supply {lth_pct:.1f}% — transition zone.")

        if shock_ratio > 3.0:
            notes.append(f"Supply Shock Ratio {shock_ratio:.1f} — extreme scarcity!")

        if dormant_pct > 65:
            notes.append(f"{dormant_pct:.0f}% of supply dormant >1yr — conviction HODLing.")

        return SupplyDynamics(
            total_supply=total_supply,
            circulating_supply=circulating_supply,
            illiquid_supply_pct=round(illiquid_pct, 2),
            lth_supply_pct=round(lth_pct, 2),
            sth_supply_pct=round(sth_pct, 2),
            supply_shock_ratio=round(shock_ratio, 2),
            last_active_1yr_plus_pct=round(dormant_pct, 2),
            signal=signal,
            notes=notes,
        )


# ═══════════════════════════════════════════════════════════════════════════
# CYCLE PHASE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class CyclePhaseDetector:
    """
    Determine market cycle phase from composite on-chain signals.
    
    Combines MVRV, SOPR, NUPL, exchange flows, and supply dynamics
    into a unified cycle assessment.
    """

    SIGNAL_SCORES = {
        OnChainSignalStrength.SCREAMING_BUY: 100,
        OnChainSignalStrength.BUY: 60,
        OnChainSignalStrength.LEAN_BUY: 30,
        OnChainSignalStrength.NEUTRAL: 0,
        OnChainSignalStrength.LEAN_SELL: -30,
        OnChainSignalStrength.SELL: -60,
        OnChainSignalStrength.SCREAMING_SELL: -100,
    }

    @classmethod
    def detect(
        cls,
        mvrv_signal: OnChainSignalStrength,
        sopr_signal: OnChainSignalStrength,
        nupl_signal: OnChainSignalStrength,
        flow_signal: OnChainSignalStrength,
        supply_signal: OnChainSignalStrength,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[MarketCyclePhase, float, OnChainSignalStrength]:
        """
        Detect cycle phase from component signals.
        Returns (phase, composite_score, composite_signal).
        """
        if weights is None:
            weights = {
                "mvrv": 0.25,
                "sopr": 0.15,
                "nupl": 0.25,
                "flows": 0.20,
                "supply": 0.15,
            }

        scores = {
            "mvrv": cls.SIGNAL_SCORES[mvrv_signal],
            "sopr": cls.SIGNAL_SCORES[sopr_signal],
            "nupl": cls.SIGNAL_SCORES[nupl_signal],
            "flows": cls.SIGNAL_SCORES[flow_signal],
            "supply": cls.SIGNAL_SCORES[supply_signal],
        }

        composite = sum(scores[k] * weights[k] for k in scores)

        # Map composite score to cycle phase
        if composite > 75:
            phase = MarketCyclePhase.CAPITULATION  # Max buy signal = bottom
        elif composite > 50:
            phase = MarketCyclePhase.ACCUMULATION
        elif composite > 25:
            phase = MarketCyclePhase.EARLY_BULL
        elif composite > 0:
            phase = MarketCyclePhase.MID_BULL
        elif composite > -25:
            phase = MarketCyclePhase.LATE_BULL
        elif composite > -50:
            phase = MarketCyclePhase.DISTRIBUTION
        elif composite > -75:
            phase = MarketCyclePhase.EARLY_BEAR
        else:
            phase = MarketCyclePhase.EUPHORIA  # Max sell signal = top

        # Map to composite signal
        if composite > 60:
            comp_signal = OnChainSignalStrength.SCREAMING_BUY
        elif composite > 30:
            comp_signal = OnChainSignalStrength.BUY
        elif composite > 10:
            comp_signal = OnChainSignalStrength.LEAN_BUY
        elif composite > -10:
            comp_signal = OnChainSignalStrength.NEUTRAL
        elif composite > -30:
            comp_signal = OnChainSignalStrength.LEAN_SELL
        elif composite > -60:
            comp_signal = OnChainSignalStrength.SELL
        else:
            comp_signal = OnChainSignalStrength.SCREAMING_SELL

        return phase, round(composite, 1), comp_signal


# ═══════════════════════════════════════════════════════════════════════════
# NVT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class NVTAnalyzer:
    """
    Network Value to Transactions analysis.
    
    NVT Ratio = Market Cap / Daily Transaction Volume (USD)
    Think of it like crypto's P/E ratio.
    
    NVT Signal = Market Cap / 90-day MA of Transaction Volume
    
    Thresholds:
      NVT Signal > 150 → overvalued (price outpacing usage)
      NVT Signal 65-150 → fair value
      NVT Signal < 65   → undervalued (usage outpacing price)
    """

    @classmethod
    def analyze(
        cls, market_cap: float, daily_tx_volume: float,
        tx_volume_90d_ma: float,
    ) -> Dict:
        """Compute NVT ratio and signal."""
        nvt_ratio = market_cap / daily_tx_volume if daily_tx_volume > 0 else float("inf")
        nvt_signal = market_cap / tx_volume_90d_ma if tx_volume_90d_ma > 0 else float("inf")

        if nvt_signal > 150:
            signal = OnChainSignalStrength.SELL
            note = f"NVT Signal {nvt_signal:.0f} — OVERVALUED. Price outpacing on-chain usage."
        elif nvt_signal > 65:
            signal = OnChainSignalStrength.NEUTRAL
            note = f"NVT Signal {nvt_signal:.0f} — fair value range."
        else:
            signal = OnChainSignalStrength.BUY
            note = f"NVT Signal {nvt_signal:.0f} — UNDERVALUED. On-chain usage exceeds price."

        return {
            "nvt_ratio": round(nvt_ratio, 1),
            "nvt_signal": round(nvt_signal, 1),
            "signal": signal,
            "note": note,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("🐺 BARREN WUFFET — On-Chain Analysis Engine v2.7.0")
    logger.info("=" * 55)

    # MVRV Analysis
    mvrv = MVRVAnalyzer.analyze(market_cap=900e9, realized_cap=400e9)
    logger.info(f"\nMVRV Ratio: {mvrv.mvrv_ratio}")
    logger.info(f"  Z-Score: {mvrv.mvrv_z_score}")
    logger.info(f"  Signal: {mvrv.signal.value}")
    for n in mvrv.notes: print(f"  → {n}")

    # SOPR Analysis
    sopr = SOPRAnalyzer.analyze(
        sopr=1.02, adjusted_sopr=1.01, entity_sopr=1.005,
        sth_sopr=0.98, lth_sopr=1.35, market_trend="bull",
    )
    logger.info(f"\nSOPR: {sopr.adjusted_sopr}")
    logger.info(f"  Signal: {sopr.signal.value}")
    for n in sopr.notes: print(f"  → {n}")

    # NUPL
    nupl = NUPLAnalyzer.analyze(market_cap=900e9, realized_cap=400e9)
    logger.info(f"\nNUPL: {nupl.nupl}")
    logger.info(f"  Phase: {nupl.phase} | Signal: {nupl.signal.value}")

    # Exchange Flows
    flows = ExchangeFlowAnalyzer.analyze(
        inflows_24h=5000, outflows_24h=8000,
        inflows_7d=40000, outflows_7d=55000,
        inflows_30d=160000, outflows_30d=200000,
        exchange_balance=2_400_000, total_supply=19_500_000,
    )
    logger.info(f"\nExchange Flows:")
    logger.info(f"  Net 24h: {flows.net_flow_24h:,.0f} | 7d: {flows.net_flow_7d:,.0f}")
    logger.info(f"  Balance: {flows.exchange_balance_pct:.1f}% | Trend: {flows.trend}")
    logger.info(f"  Signal: {flows.signal.value}")

    # Cycle Detection
    phase, score, sig = CyclePhaseDetector.detect(
        mvrv.signal, sopr.signal, nupl.signal, flows.signal,
        OnChainSignalStrength.BUY,
    )
    logger.info(f"\nCycle Phase: {phase.value}")
    logger.info(f"  Composite Score: {score}")
    logger.info(f"  Composite Signal: {sig.value}")
