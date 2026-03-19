"""
Crypto Forecaster — AAC Forecaster Layer
==========================================
Crypto obeys DIFFERENT PHYSICS than equities. Core differences:

  Liquidity dominance — crypto is a liquidity amplifier
  Leverage reflexivity — funding rates + OI drive price, not the reverse
  No valuation anchor — pure narrative + momentum + reflexivity
  24/7 liquidation risk — no circuit breakers, gaps are violent
  Correlation is dynamic — sometimes correlated with SPX, sometimes inverse

Crypto-specific formulas:
  C1  Liquidity Reflexive Melt       — Global liquidity expanding + neutral funding = long
  C2  Leverage Fragility Flush       — Extreme funding + OI spike + weak spot = short cascade
  C3  Decoupling Signal              — Crypto underperforms equity + unstable funding = contagion
  C4  Vol Compression Trap           — IV crushed → violent directional move incoming
  C5  Exchange Inflow Spike          — Coins flowing to exchanges = sell pressure building
  C6  BTC Dominance Expansion        — Risk-off within crypto = alt liquidation incoming
  C7  Funding Rate Mean Reversion    — Extreme positive funding → mean revert (short-term)
  C8  Correlation Breakout           — BTC/SPX correlation changes regime

Exchanges with live data:
  - NDAX (CAD, ETH/XRP/BTC) — live credentials in .env
  - Coinbase Advanced (BTC/ETH)
  - Alternative.me Fear & Greed (free)
  - CoinGecko Pro (COINGECKO_PRO_API_KEY in .env)
  - Binance public endpoints (no auth for market data)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CRYPTO-SPECIFIC ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class CryptoRegime(Enum):
    REFLEXIVE_MELT_UP = "reflexive_melt_up"          # Liquidity + neutral funding → explosive long
    LEVERAGE_FRAGILITY = "leverage_fragility"          # Overcrowded longs, OI extreme → flush risk
    RISK_OFF_CONTAGION = "risk_off_contagion"          # Crypto sells off with TradFi
    VOL_COMPRESSION = "vol_compression"                # IV crushed, move incoming (direction TBD)
    EXCHANGE_INFLOW_PRESSURE = "exchange_inflow"       # Sell pressure building
    ALT_LIQUIDATION = "alt_liquidation"                # BTC dominance rising, alts flush
    MEAN_REVERSION = "mean_reversion"                  # Funding extreme, snap back due
    CORRELATED_BREAKDOWN = "correlated_breakdown"      # BTC/SPX correlation spiking
    ACCUMULATION = "accumulation"                      # Extreme fear + OI low + spot bid
    UNCERTAIN = "uncertain"


class CryptoDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    REDUCE_ALTS = "reduce_alts"     # BTC ok, sell ETH/alts


class CryptoExpressionType(Enum):
    SPOT_LONG = "spot_long"
    SPOT_SHORT = "spot_short"
    PERP_LONG = "perp_long"
    PERP_SHORT = "perp_short"
    OPTIONS_CALL = "crypto_call"
    OPTIONS_PUT = "crypto_put"
    REDUCE_LEVERAGE = "reduce_leverage"
    DELTA_NEUTRAL = "delta_neutral"     # sell funding, own spot
    AVOID = "avoid"


class CryptoFormula(Enum):
    C1_LIQUIDITY_MELT = "C1_liquidity_reflexive_melt"
    C2_LEVERAGE_FLUSH = "C2_leverage_fragility_flush"
    C3_DECOUPLING = "C3_decoupling_signal"
    C4_VOL_TRAP = "C4_vol_compression_trap"
    C5_EXCHANGE_INFLOW = "C5_exchange_inflow_spike"
    C6_BTC_DOMINANCE = "C6_btc_dominance_expansion"
    C7_FUNDING_REVERSION = "C7_funding_rate_mean_reversion"
    C8_CORRELATION_BREAK = "C8_correlation_breakout"


# ═══════════════════════════════════════════════════════════════════════════
# CRYPTO INPUT SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CryptoSnapshot:
    """
    Current-state reading of all crypto-relevant signals.

    Live API mapping:
        btc_price           ← CoinGecko / NDAX / Coinbase
        btc_return_1d       ← CoinGecko (% change 24h)
        eth_price           ← CoinGecko
        eth_return_1d       ← CoinGecko
        btc_dominance_pct   ← CoinGecko /global
        funding_rate_btc    ← Binance futures API (8h rate, %)
        funding_rate_eth    ← Binance futures API
        open_interest_btc   ← Binance/Coinglass (USD)
        open_interest_change_pct ← vs 24h ago
        exchange_inflow_btc ← exchange inflows (CryptoQuant / Glassnode)
        iv_btc_1w           ← 1-week BTC IV (options)
        iv_btc_1m           ← 1-month BTC IV
        fear_greed          ← alternative.me (0-100)
        spx_return_1d       ← SPX % return (TradFi correlation check)
        dxy_change_1d       ← Dollar index change (global liquidity proxy)
        m2_global_growth    ← Global M2 growth rate (quarterly)
        realized_vol_7d     ← BTC 7-day realized vol
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Price
    btc_price: Optional[float] = None
    btc_return_1d: Optional[float] = None        # %
    btc_return_7d: Optional[float] = None        # %
    eth_price: Optional[float] = None
    eth_return_1d: Optional[float] = None        # %
    eth_btc_ratio: Optional[float] = None        # ETH/BTC (declining = alt risk-off)

    # Market structure
    btc_dominance_pct: Optional[float] = None    # % BTC dominance
    btc_dominance_change_1d: Optional[float] = None

    # Leverage
    funding_rate_btc: Optional[float] = None     # 8h % (>0.01% = crowded longs)
    funding_rate_eth: Optional[float] = None
    open_interest_btc_usd: Optional[float] = None
    oi_change_pct_24h: Optional[float] = None    # % change in OI

    # Spot activity
    exchange_inflow_btc: Optional[float] = None  # coins flowing to exchanges
    exchange_inflow_change_pct: Optional[float] = None

    # Volatility
    iv_btc_1w: Optional[float] = None            # annualised IV %
    iv_btc_1m: Optional[float] = None
    realized_vol_7d: Optional[float] = None

    # Sentiment
    fear_greed: Optional[float] = None           # 0=extreme fear, 100=extreme greed

    # Macro correlation inputs
    spx_return_1d: Optional[float] = None        # for correlation check
    dxy_change_1d: Optional[float] = None        # dollar strength = headwind
    m2_global_growth: Optional[float] = None     # % — expansion = tailwind
    global_liquidity_trend: Optional[str] = None  # "expanding" | "contracting" | "neutral"


# ═══════════════════════════════════════════════════════════════════════════
# CRYPTO FORMULA RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CryptoFormulaResult:
    formula: CryptoFormula
    fired: bool
    confidence: float
    direction: CryptoDirection
    conditions_met: List[str]
    conditions_missing: List[str]
    expected_outcome: str
    timeframe_str: str
    expression: CryptoExpressionType
    risk_level: str       # "low" | "medium" | "high" | "extreme"
    note: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# CRYPTO REGIME STATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CryptoRegimeState:
    timestamp: datetime
    primary_regime: CryptoRegime
    secondary_regime: Optional[CryptoRegime]
    regime_confidence: float
    formula_results: List[CryptoFormulaResult]
    armed_formulas: List[CryptoFormula]
    long_signals: int
    short_signals: int
    net_bias: CryptoDirection

    def print_plan(self) -> str:
        lines = [
            "=" * 70,
            "CRYPTO FORECASTER",
            f"Regime: {self.primary_regime.value.upper().replace('_', ' ')}",
            f"Bias: {self.net_bias.value.upper()} | Confidence: {self.regime_confidence:.0%}",
            f"Long signals: {self.long_signals} | Short signals: {self.short_signals}",
            "=" * 70,
        ]
        for r in self.formula_results:
            if r.fired:
                star = "✅" if r.direction in (CryptoDirection.LONG,) else "🔴"
                lines.append(f"{star} [{r.formula.value}] conf={r.confidence:.0%}")
                lines.append(f"   → {r.expected_outcome}")
                lines.append(f"   → Expression: {r.expression.value} | {r.timeframe_str}")
                lines.append(f"   → Risk: {r.risk_level}")
                if r.conditions_met:
                    lines.append(f"   ✓ {' | '.join(r.conditions_met)}")
                lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CRYPTO FORECASTER ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class CryptoForecaster:
    """
    Evaluates a CryptoSnapshot against all C1-C8 formulas and
    classifies the current crypto regime.

    Usage:
        from strategies.crypto_forecaster import CryptoForecaster, CryptoSnapshot
        snap = CryptoSnapshot(
            btc_price=65000, btc_return_1d=-2.5,
            funding_rate_btc=0.018, oi_change_pct_24h=8.5,
            fear_greed=72, spx_return_1d=-1.2,
        )
        forecaster = CryptoForecaster()
        state = forecaster.evaluate(snap)
        print(state.print_plan())
    """

    # Thresholds
    _FUNDING_EXTREME_LONG = 0.010      # 8h rate > 0.01% = overcrowded longs
    _FUNDING_EXTREME_SHORT = -0.005    # negative = aggressive shorts (contrarian buy signal)
    _OI_SPIKE = 10.0                   # OI up >10% in 24h = leverage building
    _INFLOW_SPIKE_PCT = 20.0           # exchange inflows up >20% = sell pressure
    _BTC_DOM_RISING = 1.0              # dominance up >1% in 24h = alts dumping
    _FEAR_EXTREME = 25                 # extreme fear (contrarian buy)
    _GREED_EXTREME = 75                # extreme greed (contrarian sell signal)
    _IV_COMPRESSED = 40.0              # BTC IV below 40% annualised = compressed
    _IV_NORMAL = 70.0

    def evaluate(self, snap: CryptoSnapshot) -> CryptoRegimeState:
        results = [
            self._c1_liquidity_melt(snap),
            self._c2_leverage_flush(snap),
            self._c3_decoupling(snap),
            self._c4_vol_compression(snap),
            self._c5_exchange_inflow(snap),
            self._c6_btc_dominance(snap),
            self._c7_funding_reversion(snap),
            self._c8_correlation_break(snap),
        ]

        armed = [r.formula for r in results if r.fired]
        longs = sum(1 for r in results if r.fired and r.direction in (CryptoDirection.LONG,))
        shorts = sum(1 for r in results if r.fired and r.direction in (
            CryptoDirection.SHORT, CryptoDirection.REDUCE_ALTS
        ))
        net_bias = (CryptoDirection.LONG if longs > shorts
                    else CryptoDirection.SHORT if shorts > longs
                    else CryptoDirection.NEUTRAL)

        primary, secondary, confidence = self._classify_regime(results, snap)

        return CryptoRegimeState(
            timestamp=snap.timestamp,
            primary_regime=primary,
            secondary_regime=secondary,
            regime_confidence=confidence,
            formula_results=results,
            armed_formulas=armed,
            long_signals=longs,
            short_signals=shorts,
            net_bias=net_bias,
        )

    # ──────────────────────────────────────────────────────────
    # C1: Liquidity Reflexive Melt  (LONG bias, NEAR-GUARANTEE-tier)
    # ──────────────────────────────────────────────────────────

    def _c1_liquidity_melt(self, snap: CryptoSnapshot) -> CryptoFormulaResult:
        met, missing = [], []
        cp: List[float] = []

        # Global liquidity expanding
        if snap.global_liquidity_trend is not None:
            if snap.global_liquidity_trend == "expanding":
                met.append("Global liquidity expanding")
                cp.append(0.8)
            elif snap.global_liquidity_trend == "contracting":
                missing.append("Liquidity contracting (headwind)")
        else:
            missing.append("global_liquidity_trend")

        # DXY weakening (dollar down = crypto up)
        if snap.dxy_change_1d is not None:
            if snap.dxy_change_1d < -0.3:
                met.append(f"DXY weakening ({snap.dxy_change_1d:.2f}%)")
                cp.append(0.5)
        else:
            missing.append("dxy_change_1d")

        # Funding neutral or negative (not overcrowded)
        if snap.funding_rate_btc is not None:
            if snap.funding_rate_btc < self._FUNDING_EXTREME_LONG:
                met.append(f"Funding rate neutral/negative ({snap.funding_rate_btc:.4f}%)")
                cp.append(0.6)
        else:
            missing.append("funding_rate_btc")

        # Vol suppressed (cheap calls)
        if snap.iv_btc_1m is not None:
            if snap.iv_btc_1m < self._IV_NORMAL:
                met.append(f"IV {snap.iv_btc_1m:.0f}% (not expensive)")
                cp.append(0.3)
        else:
            missing.append("iv_btc_1m")

        fired = len(met) >= 3
        confidence = sum(cp) / max(len(cp), 1) if fired else 0.0

        return CryptoFormulaResult(
            formula=CryptoFormula.C1_LIQUIDITY_MELT,
            fired=fired,
            confidence=round(confidence, 3),
            direction=CryptoDirection.LONG,
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Violent upside expansion. BTC leads, ETH follows. Alts 2-4× BTC move.",
            timeframe_str="Days to weeks",
            expression=CryptoExpressionType.SPOT_LONG,
            risk_level="medium",
            note="Best entry: low IV + neutral funding + liquidity just started expanding",
        )

    # ──────────────────────────────────────────────────────────
    # C2: Leverage Fragility Flush  (SHORT bias, CONVEX)
    # ──────────────────────────────────────────────────────────

    def _c2_leverage_flush(self, snap: CryptoSnapshot) -> CryptoFormulaResult:
        met, missing = [], []
        cp: List[float] = []

        # Funding rate extreme positive
        if snap.funding_rate_btc is not None:
            if snap.funding_rate_btc > self._FUNDING_EXTREME_LONG:
                met.append(f"Funding extreme: {snap.funding_rate_btc:.4f}%/8h (overcrowded longs)")
                cp.append(min(1.0, snap.funding_rate_btc / 0.03))
        else:
            missing.append("funding_rate_btc")

        # OI spiking
        if snap.oi_change_pct_24h is not None:
            if snap.oi_change_pct_24h > self._OI_SPIKE:
                met.append(f"OI +{snap.oi_change_pct_24h:.1f}% in 24h (leverage building)")
                cp.append(min(1.0, snap.oi_change_pct_24h / 25))
        else:
            missing.append("oi_change_pct_24h")

        # Spot bid weakening (price not confirming leverage)
        if snap.btc_return_1d is not None and snap.oi_change_pct_24h is not None:
            if snap.btc_return_1d < 0 and snap.oi_change_pct_24h > 5:
                met.append(f"OI rising while price falling (distribution)")
                cp.append(0.8)

        # Greed extreme
        if snap.fear_greed is not None:
            if snap.fear_greed > self._GREED_EXTREME:
                met.append(f"Extreme greed ({snap.fear_greed})")
                cp.append(0.4)
        else:
            missing.append("fear_greed")

        fired = len(met) >= 2
        confidence = sum(cp) / max(len(cp), 1) if fired else 0.0

        return CryptoFormulaResult(
            formula=CryptoFormula.C2_LEVERAGE_FLUSH,
            fired=fired,
            confidence=round(confidence, 3),
            direction=CryptoDirection.SHORT,
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Cascading liquidation event. Long liquidations create negative feedback. Move is fast (hours).",
            timeframe_str="Hours to 2 days",
            expression=CryptoExpressionType.PERP_SHORT,
            risk_level="high",
            note="Exit before funding normalises. Don't hold short through funding flip to negative.",
        )

    # ──────────────────────────────────────────────────────────
    # C3: Decoupling Signal  (SHORT bias, INSTITUTIONAL)
    # ──────────────────────────────────────────────────────────

    def _c3_decoupling(self, snap: CryptoSnapshot) -> CryptoFormulaResult:
        met, missing = [], []
        cp: List[float] = []

        # Crypto underperforming equities
        if snap.btc_return_1d is not None and snap.spx_return_1d is not None:
            if snap.btc_return_1d < snap.spx_return_1d - 2.0:
                met.append(f"BTC {snap.btc_return_1d:.1f}% vs SPX {snap.spx_return_1d:.1f}% (underperforming)")
                cp.append(min(1.0, (snap.spx_return_1d - snap.btc_return_1d) / 10))
        else:
            missing.append("btc_return_1d / spx_return_1d")

        # Funding unstable (positive when market falling = dangerous)
        if snap.funding_rate_btc is not None and snap.btc_return_1d is not None:
            if snap.btc_return_1d < -2 and snap.funding_rate_btc > 0:
                met.append(f"Positive funding + price falling (longs will be squeezed)")
                cp.append(0.7)

        # TradFi stress present
        if snap.spx_return_1d is not None:
            if snap.spx_return_1d < -1.5:
                met.append(f"TradFi stress: SPX {snap.spx_return_1d:.1f}%")
                cp.append(0.3)

        fired = len(met) >= 2
        confidence = sum(cp) / max(len(cp), 1) if fired else 0.0

        return CryptoFormulaResult(
            formula=CryptoFormula.C3_DECOUPLING,
            fired=fired,
            confidence=round(confidence, 3),
            direction=CryptoDirection.SHORT,
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Risk-off spillover. Crypto crashes faster than stocks when decoupling breaks down.",
            timeframe_str="1-5 days",
            expression=CryptoExpressionType.REDUCE_LEVERAGE,
            risk_level="medium",
            note="Reduce spot + eliminate perp longs. Short only if funding flips.",
        )

    # ──────────────────────────────────────────────────────────
    # C4: Vol Compression Trap  (NEUTRAL → violent move)
    # ──────────────────────────────────────────────────────────

    def _c4_vol_compression(self, snap: CryptoSnapshot) -> CryptoFormulaResult:
        met, missing = [], []
        cp: List[float] = []

        if snap.iv_btc_1m is not None:
            if snap.iv_btc_1m < self._IV_COMPRESSED:
                met.append(f"IV {snap.iv_btc_1m:.0f}% — compressed below 40%")
                cp.append(min(1.0, (self._IV_COMPRESSED - snap.iv_btc_1m) / 20))
        else:
            missing.append("iv_btc_1m")

        if snap.realized_vol_7d is not None and snap.iv_btc_1m is not None:
            if snap.iv_btc_1m < snap.realized_vol_7d * 0.8:
                met.append(f"IV ({snap.iv_btc_1m:.0f}%) < realized ({snap.realized_vol_7d:.0f}%) — unusual compression")
                cp.append(0.7)
        else:
            missing.append("realized_vol_7d")

        # Macro context: if TradFi shocks accumulating, vol break likely downside
        if snap.spx_return_1d is not None and snap.spx_return_1d < -1:
            met.append("TradFi shocks accumulating — vol break likely downside")
            cp.append(0.4)

        fired = len(met) >= 2
        confidence = sum(cp) / max(len(cp), 1) if fired else 0.0
        direction = CryptoDirection.SHORT if (snap.spx_return_1d and snap.spx_return_1d < -0.5) else CryptoDirection.NEUTRAL

        return CryptoFormulaResult(
            formula=CryptoFormula.C4_VOL_TRAP,
            fired=fired,
            confidence=round(confidence, 3),
            direction=direction,
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Violent directional move (often both sides sequentially). Options cheapest now.",
            timeframe_str="Days to 2 weeks",
            expression=(CryptoExpressionType.OPTIONS_PUT if direction == CryptoDirection.SHORT
                        else CryptoExpressionType.OPTIONS_CALL),
            risk_level="high",
            note="Straddles or directional options. Direction follows the macro context.",
        )

    # ──────────────────────────────────────────────────────────
    # C5: Exchange Inflow Spike  (SHORT pressure)
    # ──────────────────────────────────────────────────────────

    def _c5_exchange_inflow(self, snap: CryptoSnapshot) -> CryptoFormulaResult:
        met, missing = [], []
        cp: List[float] = []

        if snap.exchange_inflow_change_pct is not None:
            if snap.exchange_inflow_change_pct > self._INFLOW_SPIKE_PCT:
                met.append(f"Exchange inflows +{snap.exchange_inflow_change_pct:.0f}% (coins moving to sell)")
                cp.append(min(1.0, snap.exchange_inflow_change_pct / 50))
        else:
            missing.append("exchange_inflow_change_pct")

        # Inflows + price already up = distribution
        if snap.btc_return_7d is not None:
            if snap.btc_return_7d > 10 and snap.exchange_inflow_change_pct and snap.exchange_inflow_change_pct > 15:
                met.append(f"Price up {snap.btc_return_7d:.0f}% 7d + inflows rising = distribution")
                cp.append(0.7)

        fired = len(met) >= 1 and len(cp) > 0
        confidence = sum(cp) / max(len(cp), 1) if fired else 0.0

        return CryptoFormulaResult(
            formula=CryptoFormula.C5_EXCHANGE_INFLOW,
            fired=fired,
            confidence=round(confidence, 3),
            direction=CryptoDirection.SHORT,
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Sell pressure building. Good to reduce spot / take profits. Not an immediate crash signal.",
            timeframe_str="Days to 1 week",
            expression=CryptoExpressionType.REDUCE_LEVERAGE,
            risk_level="low",
            note="On-chain sell signal — slower but reliable for positioning.",
        )

    # ──────────────────────────────────────────────────────────
    # C6: BTC Dominance Expansion  (REDUCE ALTS)
    # ──────────────────────────────────────────────────────────

    def _c6_btc_dominance(self, snap: CryptoSnapshot) -> CryptoFormulaResult:
        met, missing = [], []
        cp: List[float] = []

        if snap.btc_dominance_change_1d is not None:
            if snap.btc_dominance_change_1d > self._BTC_DOM_RISING:
                met.append(f"BTC dominance +{snap.btc_dominance_change_1d:.1f}% (alts being dumped)")
                cp.append(min(1.0, snap.btc_dominance_change_1d / 3))
        else:
            missing.append("btc_dominance_change_1d")

        if snap.eth_btc_ratio is not None and snap.btc_dominance_change_1d is not None:
            if snap.btc_dominance_change_1d > 0.5 and snap.eth_return_1d and snap.eth_return_1d < snap.btc_return_1d if snap.btc_return_1d else False:
                met.append("ETH underperforming BTC (risk-off within crypto)")
                cp.append(0.5)

        fired = len(met) >= 1 and len(cp) > 0
        confidence = sum(cp) / max(len(cp), 1) if fired else 0.0

        return CryptoFormulaResult(
            formula=CryptoFormula.C6_BTC_DOMINANCE,
            fired=fired,
            confidence=round(confidence, 3),
            direction=CryptoDirection.REDUCE_ALTS,
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Risk-off move within crypto. Alts flush 2-5× BTC move down. BTC may hold or drop less.",
            timeframe_str="Hours to days",
            expression=CryptoExpressionType.REDUCE_LEVERAGE,
            risk_level="medium",
            note="Exit ETH/alts, hold BTC or cash. Don't short BTC if dominance is rising.",
        )

    # ──────────────────────────────────────────────────────────
    # C7: Funding Rate Mean Reversion  (NEAR-GUARANTEE, small)
    # ──────────────────────────────────────────────────────────

    def _c7_funding_reversion(self, snap: CryptoSnapshot) -> CryptoFormulaResult:
        met, missing = [], []
        cp: List[float] = []

        if snap.funding_rate_btc is not None:
            if snap.funding_rate_btc > 0.02:  # very extreme
                met.append(f"Funding very extreme +{snap.funding_rate_btc:.4f}% — statistical reversion due")
                cp.append(min(1.0, snap.funding_rate_btc / 0.05))
                direction = CryptoDirection.SHORT
            elif snap.funding_rate_btc < self._FUNDING_EXTREME_SHORT:
                met.append(f"Funding extreme negative {snap.funding_rate_btc:.4f}% — contrarian long signal")
                cp.append(min(1.0, abs(snap.funding_rate_btc) / 0.02))
                direction = CryptoDirection.LONG
            else:
                direction = CryptoDirection.NEUTRAL
        else:
            missing.append("funding_rate_btc")
            direction = CryptoDirection.NEUTRAL

        fired = len(met) >= 1
        confidence = sum(cp) / max(len(cp), 1) if fired else 0.0

        return CryptoFormulaResult(
            formula=CryptoFormula.C7_FUNDING_REVERSION,
            fired=fired,
            confidence=round(confidence, 3),
            direction=direction,
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Short-term mean reversion in funding → price snap. Small edge, high probability.",
            timeframe_str="8-24 hours",
            expression=(CryptoExpressionType.DELTA_NEUTRAL if direction == CryptoDirection.SHORT
                        else CryptoExpressionType.SPOT_LONG),
            risk_level="low",
            note="Near-guarantee class trade. Small size. Used by market makers constantly.",
        )

    # ──────────────────────────────────────────────────────────
    # C8: Correlation Breakout  (REGIME CHANGE signal)
    # ──────────────────────────────────────────────────────────

    def _c8_correlation_break(self, snap: CryptoSnapshot) -> CryptoFormulaResult:
        met, missing = [], []
        cp: List[float] = []

        if snap.btc_return_1d is not None and snap.spx_return_1d is not None:
            btc_dir = 1 if snap.btc_return_1d > 0 else -1
            spx_dir = 1 if snap.spx_return_1d > 0 else -1
            if btc_dir != spx_dir and abs(snap.btc_return_1d) > 2 and abs(snap.spx_return_1d) > 0.5:
                met.append(
                    f"BTC {snap.btc_return_1d:+.1f}% vs SPX {snap.spx_return_1d:+.1f}% "
                    f"(decoupling — correlation breaking)"
                )
                cp.append(min(1.0, (abs(snap.btc_return_1d) + abs(snap.spx_return_1d)) / 15))
        else:
            missing.append("btc/spx return for correlation check")

        fired = len(met) >= 1 and len(cp) > 0
        confidence = sum(cp) / max(len(cp), 1) if fired else 0.0
        direction = CryptoDirection.LONG if (snap.btc_return_1d and snap.btc_return_1d > 0 and snap.spx_return_1d and snap.spx_return_1d < 0) else CryptoDirection.SHORT

        return CryptoFormulaResult(
            formula=CryptoFormula.C8_CORRELATION_BREAK,
            fired=fired,
            confidence=round(confidence, 3),
            direction=direction,
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Regime change signal. Correlation breakdown = crypto finding own narrative or accelerating TradFi stress.",
            timeframe_str="Days to weeks",
            expression=(CryptoExpressionType.SPOT_LONG if direction == CryptoDirection.LONG
                        else CryptoExpressionType.PERP_SHORT),
            risk_level="medium",
            note="Watch for 3-5 day confirmation. Cross-asset narrative often flips sharply.",
        )

    # ──────────────────────────────────────────────────────────
    # Regime classification
    # ──────────────────────────────────────────────────────────

    def _classify_regime(
        self,
        results: List[CryptoFormulaResult],
        snap: CryptoSnapshot,
    ) -> Tuple[CryptoRegime, Optional[CryptoRegime], float]:
        armed = {r.formula for r in results if r.fired}
        scores: Dict[CryptoRegime, float] = {r: 0.0 for r in CryptoRegime}

        if CryptoFormula.C1_LIQUIDITY_MELT in armed:
            scores[CryptoRegime.REFLEXIVE_MELT_UP] += 0.7
        if CryptoFormula.C2_LEVERAGE_FLUSH in armed:
            scores[CryptoRegime.LEVERAGE_FRAGILITY] += 0.8
        if CryptoFormula.C3_DECOUPLING in armed:
            scores[CryptoRegime.RISK_OFF_CONTAGION] += 0.7
        if CryptoFormula.C4_VOL_TRAP in armed:
            scores[CryptoRegime.VOL_COMPRESSION] += 0.7
        if CryptoFormula.C5_EXCHANGE_INFLOW in armed:
            scores[CryptoRegime.EXCHANGE_INFLOW_PRESSURE] += 0.6
        if CryptoFormula.C6_BTC_DOMINANCE in armed:
            scores[CryptoRegime.ALT_LIQUIDATION] += 0.7
        if CryptoFormula.C7_FUNDING_REVERSION in armed:
            scores[CryptoRegime.MEAN_REVERSION] += 0.5
        if CryptoFormula.C8_CORRELATION_BREAK in armed:
            scores[CryptoRegime.CORRELATED_BREAKDOWN] += 0.5

        # Fear extreme → accumulation zone
        if snap.fear_greed is not None and snap.fear_greed < 20:
            scores[CryptoRegime.ACCUMULATION] += 0.6

        sorted_regimes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_regimes[0][0] if sorted_regimes[0][1] > 0 else CryptoRegime.UNCERTAIN
        primary_score = sorted_regimes[0][1]
        secondary = sorted_regimes[1][0] if sorted_regimes[1][1] > 0.2 else None
        confidence = min(1.0, primary_score)

        return primary, secondary, confidence


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: build snapshot from CoinGecko/NDAX dicts
# ═══════════════════════════════════════════════════════════════════════════

def snapshot_from_coingecko(cg_data: Dict[str, Any], **kwargs: Any) -> CryptoSnapshot:
    """
    Build a CryptoSnapshot from CoinGecko API response data.

    cg_data expected keys:
        btc_price, btc_change_24h, eth_price, eth_change_24h,
        btc_dominance, fear_greed_value
    """
    snap = CryptoSnapshot(
        btc_price=cg_data.get("btc_price"),
        btc_return_1d=cg_data.get("btc_change_24h"),
        eth_price=cg_data.get("eth_price"),
        eth_return_1d=cg_data.get("eth_change_24h"),
        btc_dominance_pct=cg_data.get("btc_dominance"),
        fear_greed=cg_data.get("fear_greed_value"),
    )
    for k, v in kwargs.items():
        if hasattr(snap, k):
            setattr(snap, k, v)
    return snap
