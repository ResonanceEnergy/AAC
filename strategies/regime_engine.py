"""
Macro Regime Engine — AAC Forecaster Layer
==========================================
Core regime-detection brain. Implements the "IF X + Y → EXPECT Z" formula library.

Regime taxonomy:
  RISK_ON          — Liquidity abundant, spreads tight, vol suppressed
  RISK_OFF         — Broad de-risking, spreads widen, equities fall
  STAGFLATION      — Rising inflation + slowing growth simultaneously
  CREDIT_STRESS    — Credit leads equity down, funding tightens, HY/IG diverge
  LIQUIDITY_CRUNCH — Funding dries up, correlations spike, air-pocket risk
  VOL_SHOCK_ARMED  — Vol compressed while risks accumulate (spring-loading phase)
  VOL_SHOCK_ACTIVE — VIX erupting, cascades in progress
  POLICY_DELAY_TRAP — Stress visible but policy absent/vague → pre-shock convexity zone

Formula library (from the playbook conversations):
  F1  Credit-Led Breakdown
  F2  Stagflation Compression
  F3  Liquidity Mirage
  F4  Policy Delay Trap
  F5  Failed Safe Haven
  F6  Correlation Spike Predictor
  F7  Volatility Compression Bomb
  F8  Narrative Break
  F9  Leverage Reveal

Live-API inputs accepted:
  - FRED: T10Y2Y (yield curve), BAMLH0A0HYM2 (HY spread), VIXCLS, DCOILWTICO, GOLDAMGBD228NLBM
  - Finnhub: real-time prices, sector performance
  - Fear & Greed Index: sentiment
  - Any dict of {signal_key: float} can be injected manually
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# REGIME TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════

class Regime(Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    STAGFLATION = "stagflation"
    CREDIT_STRESS = "credit_stress"
    LIQUIDITY_CRUNCH = "liquidity_crunch"
    VOL_SHOCK_ARMED = "vol_shock_armed"
    VOL_SHOCK_ACTIVE = "vol_shock_active"
    POLICY_DELAY_TRAP = "policy_delay_trap"
    UNCERTAIN = "uncertain"


class FormulaTag(Enum):
    F1_CREDIT_LED_BREAKDOWN = "F1_credit_led_breakdown"
    F2_STAGFLATION_COMPRESSION = "F2_stagflation_compression"
    F3_LIQUIDITY_MIRAGE = "F3_liquidity_mirage"
    F4_POLICY_DELAY_TRAP = "F4_policy_delay_trap"
    F5_FAILED_SAFE_HAVEN = "F5_failed_safe_haven"
    F6_CORRELATION_SPIKE = "F6_correlation_spike"
    F7_VOL_COMPRESSION_BOMB = "F7_vol_compression_bomb"
    F8_NARRATIVE_BREAK = "F8_narrative_break"
    F9_LEVERAGE_REVEAL = "F9_leverage_reveal"


class SignalRiskClass(Enum):
    NEAR_GUARANTEE = "near_guarantee"    # High probability, small ROI (carry, vol mean-reversion)
    INSTITUTIONAL = "institutional"      # Widely used, crowded, confirmed
    FRINGE = "fringe"                    # Under-the-hood, alpha edge
    CONVEX = "convex"                    # Risky, big payoff, requires patience + precision


# ═══════════════════════════════════════════════════════════════════════════
# INPUT SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MacroSnapshot:
    """Current-state reading of all live macro inputs.

    All numeric fields accept float. Fields left as None are treated
    as "unknown" and excluded from formula evaluation gracefully.

    Live API mapping:
        vix               ← FRED VIXCLS or Finnhub ^VIX
        hy_spread_bps     ← FRED BAMLH0A0HYM2  (x100 = bps)
        yield_curve_10_2  ← FRED T10Y2Y         (negative = inverted)
        oil_price         ← FRED DCOILWTICO
        gold_price        ← FRED GOLDAMGBD228NLBM
        core_pce          ← FRED PCEPILFE (annualised %)
        gdp_growth        ← FRED GDPC1 or A191RL1Q225SBEA (%)
        hyg_return_1d     ← HYG daily % return (Finnhub / IBKR)
        spy_return_1d     ← SPY daily % return
        kre_return_1d     ← KRE daily % return
        fear_greed        ← alternative.me (0-100)
        dollar_index      ← DXY (Finnhub or FRED DTWEXBGS)
        credit_vol_skew   ← put/call OI ratio (Unusual Whales)
        volume_ratio      ← today vol / 20d avg vol
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Volatility
    vix: Optional[float] = None
    vix_change_1d: Optional[float] = None      # % change
    realized_vol_20d: Optional[float] = None   # 20-day realized vol SPY

    # Credit
    hy_spread_bps: Optional[float] = None      # HY OAS in bps (>400 = stress)
    ig_spread_bps: Optional[float] = None      # IG OAS in bps
    loan_spread_bps: Optional[float] = None    # Leveraged loan spread

    # Rates & Curve
    yield_curve_10_2: Optional[float] = None   # 10Y - 2Y (negative = inverted)
    yield_10y: Optional[float] = None
    real_yield_10y: Optional[float] = None     # 10Y TIPS
    breakeven_inflation: Optional[float] = None

    # Macro
    core_pce: Optional[float] = None           # %
    gdp_growth: Optional[float] = None         # %
    oil_price: Optional[float] = None          # WTI $/bbl
    gold_price: Optional[float] = None         # $/oz
    dollar_index: Optional[float] = None

    # Market internals
    spy_return_1d: Optional[float] = None      # %
    hyg_return_1d: Optional[float] = None      # %
    kre_return_1d: Optional[float] = None      # %
    qqq_return_1d: Optional[float] = None      # %
    airlines_return_1d: Optional[float] = None # JETS % return
    shipping_return_1d: Optional[float] = None # ZIM/GOGL proxy
    breadth_adv_dec: Optional[float] = None    # advance/decline ratio
    new_highs_52w: Optional[int] = None
    new_lows_52w: Optional[int] = None

    # Sentiment
    fear_greed: Optional[float] = None         # 0-100
    volume_ratio: Optional[float] = None       # today/20d avg
    safe_haven_bid: Optional[bool] = None      # True if gold/bonds rallying on risk-off day

    # Private credit / idiosyncratic
    private_credit_redemption_pct: Optional[float] = None  # % of AUM
    war_active: bool = False
    hormuz_blocked: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# FORMULA RESULTS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FormulaResult:
    """Output of a single IF X+Y → Z formula evaluation."""
    tag: FormulaTag
    fired: bool
    confidence: float            # 0.0 – 1.0
    conditions_met: List[str]    # which IF legs triggered
    conditions_missing: List[str]   # IF legs we couldn't evaluate (data absent)
    expected_outcome: str        # Z — what to expect
    risk_class: SignalRiskClass
    timeframe_days: Tuple[int, int]   # (min_days, max_days) for outcome
    expression_hint: str         # e.g. "put spreads", "VIX call spread"


# ═══════════════════════════════════════════════════════════════════════════
# REGIME STATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeState:
    """Output of the regime engine for a given MacroSnapshot."""
    timestamp: datetime
    primary_regime: Regime
    secondary_regime: Optional[Regime]
    regime_confidence: float          # 0.0 – 1.0
    formula_results: List[FormulaResult]
    armed_formulas: List[FormulaTag]  # formulas that fired
    vol_shock_readiness: float        # 0-100 checklist score (from playbook)
    bear_signals: int                 # count of bearish signals firing
    bull_signals: int
    summary: str

    @property
    def is_bearish(self) -> bool:
        return self.bear_signals > self.bull_signals

    @property
    def shock_imminent(self) -> bool:
        return self.vol_shock_readiness >= 60.0

    @property
    def top_formulas(self) -> List[FormulaResult]:
        return sorted(
            [f for f in self.formula_results if f.fired],
            key=lambda x: x.confidence,
            reverse=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# REGIME ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class RegimeEngine:
    """
    Evaluates a MacroSnapshot against all IF X+Y→Z formulas and
    classifies the current market regime.

    Usage:
        engine = RegimeEngine()
        snapshot = MacroSnapshot(vix=21.5, hy_spread_bps=380, ...)
        state = engine.evaluate(snapshot)
        print(state.primary_regime, state.armed_formulas)
    """

    # Thresholds (adjustable at init)
    _HY_STRESS_BPS = 350          # HY spread above this = stress
    _HY_SEVERE_BPS = 500
    _VIX_SUPPRESSED = 20          # VIX below this while risks accumulate
    _VIX_SHOCK = 30
    _VOL_COMPRESSION_THRESHOLD = 18
    _YIELD_INVERSION = -0.10      # 10-2Y below this = inverted
    _OIL_SHOCK = 120              # WTI above this = oil shock
    _STAGFLATION_PCE = 2.5        # core PCE above this
    _STAGFLATION_GDP = 1.5        # GDP below this
    _FEAR_EXTREME = 25            # fear/greed below this = extreme fear
    _GREED_EXTREME = 75

    def __init__(
        self,
        hy_stress_bps: float = 350,
        vix_suppressed: float = 20,
        oil_shock_threshold: float = 120,
    ):
        self._HY_STRESS_BPS = hy_stress_bps
        self._VIX_SUPPRESSED = vix_suppressed
        self._OIL_SHOCK = oil_shock_threshold

    # ──────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────

    def evaluate(self, snap: MacroSnapshot) -> RegimeState:
        """Evaluate all formulas and classify the regime."""
        results: List[FormulaResult] = [
            self._f1_credit_led_breakdown(snap),
            self._f2_stagflation_compression(snap),
            self._f3_liquidity_mirage(snap),
            self._f4_policy_delay_trap(snap),
            self._f5_failed_safe_haven(snap),
            self._f6_correlation_spike(snap),
            self._f7_vol_compression_bomb(snap),
            self._f8_narrative_break(snap),
            self._f9_leverage_reveal(snap),
        ]

        armed = [r.tag for r in results if r.fired]
        vol_score = self._vol_shock_checklist(snap)
        primary, secondary, confidence = self._classify_regime(results, snap, vol_score)
        bear, bull = self._count_signals(results, snap)
        summary = self._build_summary(primary, secondary, armed, vol_score, bear, bull)

        return RegimeState(
            timestamp=snap.timestamp,
            primary_regime=primary,
            secondary_regime=secondary,
            regime_confidence=confidence,
            formula_results=results,
            armed_formulas=armed,
            vol_shock_readiness=vol_score,
            bear_signals=bear,
            bull_signals=bull,
            summary=summary,
        )

    # ──────────────────────────────────────────────────────────
    # Vol shock readiness checklist (5-point, 20pts each)
    # ──────────────────────────────────────────────────────────

    def _vol_shock_checklist(self, snap: MacroSnapshot) -> float:
        """Returns 0-100 score. 60+ = shock window open. 80+ = imminent."""
        score = 0.0

        # 1. Credit leads down (HYG red while SPY flat/green)
        if snap.hyg_return_1d is not None and snap.spy_return_1d is not None:
            if snap.hyg_return_1d < -0.3 and snap.spy_return_1d >= -0.5:
                score += 20.0
            elif snap.hyg_return_1d < snap.spy_return_1d - 0.5:
                score += 12.0

        # 2. Banks fail on green tape
        if snap.kre_return_1d is not None and snap.spy_return_1d is not None:
            if snap.kre_return_1d < -0.5 and snap.spy_return_1d >= -0.3:
                score += 20.0
            elif snap.kre_return_1d < snap.spy_return_1d - 0.8:
                score += 10.0

        # 3. Oil up + growth stalling
        if snap.oil_price is not None and snap.airlines_return_1d is not None:
            if snap.oil_price > 90 and snap.airlines_return_1d is not None and snap.airlines_return_1d < 0:
                score += 20.0
            elif snap.oil_price > 80 and snap.airlines_return_1d is not None and snap.airlines_return_1d < -1.0:
                score += 10.0

        # 4. VIX suppressed despite macro deterioration
        if snap.vix is not None and snap.hy_spread_bps is not None:
            if snap.vix < self._VIX_SUPPRESSED and snap.hy_spread_bps > self._HY_STRESS_BPS:
                score += 20.0
            elif snap.vix < self._VIX_SUPPRESSED and snap.hy_spread_bps > 300:
                score += 10.0

        # 5. Tape thinning (vol below avg, breadth deteriorating)
        if snap.volume_ratio is not None and snap.breadth_adv_dec is not None:
            if snap.volume_ratio < 0.8 and snap.breadth_adv_dec < 0.9:
                score += 20.0
            elif snap.volume_ratio < 0.85:
                score += 8.0

        return min(score, 100.0)

    # ──────────────────────────────────────────────────────────
    # Formula F1: Credit-Led Breakdown  (INSTITUTIONAL)
    # ──────────────────────────────────────────────────────────

    def _f1_credit_led_breakdown(self, snap: MacroSnapshot) -> FormulaResult:
        met, missing = [], []
        confidence_parts: List[float] = []

        if snap.hy_spread_bps is not None:
            if snap.hy_spread_bps > self._HY_STRESS_BPS:
                met.append(f"HY spread {snap.hy_spread_bps:.0f}bps > {self._HY_STRESS_BPS}")
                confidence_parts.append(min(1.0, (snap.hy_spread_bps - self._HY_STRESS_BPS) / 150))
        else:
            missing.append("hy_spread_bps")

        if snap.hyg_return_1d is not None and snap.spy_return_1d is not None:
            divergence = snap.spy_return_1d - snap.hyg_return_1d
            if divergence > 0.5:
                met.append(f"HYG {snap.hyg_return_1d:.2f}% vs SPY {snap.spy_return_1d:.2f}%")
                confidence_parts.append(min(1.0, divergence / 2.0))
        else:
            missing.append("hyg_return_1d / spy_return_1d")

        if snap.yield_curve_10_2 is not None:
            if snap.yield_curve_10_2 < self._YIELD_INVERSION:
                met.append(f"Yield curve inverted {snap.yield_curve_10_2:.2f}%")
                confidence_parts.append(min(1.0, abs(snap.yield_curve_10_2) / 1.0))
        else:
            missing.append("yield_curve_10_2")

        fired = len(met) >= 2
        confidence = (sum(confidence_parts) / max(len(confidence_parts), 1)) if fired else 0.0

        return FormulaResult(
            tag=FormulaTag.F1_CREDIT_LED_BREAKDOWN,
            fired=fired,
            confidence=round(confidence, 3),
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Equity volatility expansion within 1-5 sessions; banks and cyclicals lead lower",
            risk_class=SignalRiskClass.INSTITUTIONAL,
            timeframe_days=(1, 15),
            expression_hint="ATM/OTM put spreads on HYG, KRE (2-5 weeks)",
        )

    # ──────────────────────────────────────────────────────────
    # Formula F2: Stagflation Compression  (INSTITUTIONAL)
    # ──────────────────────────────────────────────────────────

    def _f2_stagflation_compression(self, snap: MacroSnapshot) -> FormulaResult:
        met, missing = [], []
        confidence_parts: List[float] = []

        if snap.oil_price is not None:
            if snap.oil_price > 90:
                met.append(f"Oil ${snap.oil_price:.0f}/bbl (elevated)")
                confidence_parts.append(min(1.0, (snap.oil_price - 90) / 110))
        else:
            missing.append("oil_price")

        if snap.core_pce is not None:
            if snap.core_pce > self._STAGFLATION_PCE:
                met.append(f"Core PCE {snap.core_pce:.1f}% > {self._STAGFLATION_PCE}")
                confidence_parts.append(min(1.0, (snap.core_pce - self._STAGFLATION_PCE) / 2.0))
        else:
            missing.append("core_pce")

        if snap.gdp_growth is not None:
            if snap.gdp_growth < self._STAGFLATION_GDP:
                met.append(f"GDP growth {snap.gdp_growth:.1f}% < {self._STAGFLATION_GDP}")
                confidence_parts.append(min(1.0, (self._STAGFLATION_GDP - snap.gdp_growth) / 3.0))
        else:
            missing.append("gdp_growth")

        if snap.airlines_return_1d is not None and snap.oil_price is not None:
            if snap.oil_price > 80 and snap.airlines_return_1d < 0:
                met.append(f"Airlines red ({snap.airlines_return_1d:.2f}%) while oil elevated")
                confidence_parts.append(0.3)
        else:
            missing.append("airlines_return_1d (oil shock check)")

        fired = len(met) >= 2
        confidence = (sum(confidence_parts) / max(len(confidence_parts), 1)) if fired else 0.0

        return FormulaResult(
            tag=FormulaTag.F2_STAGFLATION_COMPRESSION,
            fired=fired,
            confidence=round(confidence, 3),
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="P/E multiple compression; airlines/transport/discretionary underperform; VIX grinds higher",
            risk_class=SignalRiskClass.INSTITUTIONAL,
            timeframe_days=(5, 90),
            expression_hint="Puts on JETS, XLY, sector ETFs. Medium-dated (4-12 weeks)",
        )

    # ──────────────────────────────────────────────────────────
    # Formula F3: Liquidity Mirage  (FRINGE)
    # ──────────────────────────────────────────────────────────

    def _f3_liquidity_mirage(self, snap: MacroSnapshot) -> FormulaResult:
        met, missing = [], []
        confidence_parts: List[float] = []

        if snap.spy_return_1d is not None:
            if snap.spy_return_1d > 0:
                met.append(f"SPY up {snap.spy_return_1d:.2f}% (index rising)")
                confidence_parts.append(0.2)
        else:
            missing.append("spy_return_1d")

        if snap.volume_ratio is not None:
            if snap.volume_ratio < 0.85:
                met.append(f"Volume ratio {snap.volume_ratio:.2f} (thin tape)")
                confidence_parts.append(min(1.0, (0.85 - snap.volume_ratio) / 0.4))
        else:
            missing.append("volume_ratio")

        if snap.breadth_adv_dec is not None:
            if snap.breadth_adv_dec < 0.95:
                met.append(f"A/D ratio {snap.breadth_adv_dec:.2f} (narrow)")
                confidence_parts.append(min(1.0, (0.95 - snap.breadth_adv_dec) / 0.4))
        else:
            missing.append("breadth_adv_dec")

        if snap.hyg_return_1d is not None and snap.spy_return_1d is not None:
            if snap.spy_return_1d > 0 and snap.hyg_return_1d < snap.spy_return_1d - 0.5:
                met.append("Credit not confirming equity rally")
                confidence_parts.append(0.7)
        else:
            missing.append("credit vs equity confirmation")

        fired = len(met) >= 3
        confidence = (sum(confidence_parts) / max(len(confidence_parts), 1)) if fired else 0.0

        return FormulaResult(
            tag=FormulaTag.F3_LIQUIDITY_MIRAGE,
            fired=fired,
            confidence=round(confidence, 3),
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Air-pocket drop; direction follows credit (likely down). Move can be sudden.",
            risk_class=SignalRiskClass.FRINGE,
            timeframe_days=(1, 5),
            expression_hint="Short-dated index puts or VIX call spread. Fast exit required.",
        )

    # ──────────────────────────────────────────────────────────
    # Formula F4: Policy Delay Trap  (CONVEX)
    # ──────────────────────────────────────────────────────────

    def _f4_policy_delay_trap(self, snap: MacroSnapshot) -> FormulaResult:
        met, missing = [], []
        confidence_parts: List[float] = []

        if snap.hy_spread_bps is not None:
            if snap.hy_spread_bps > self._HY_STRESS_BPS:
                met.append(f"Stress visible: HY {snap.hy_spread_bps:.0f}bps")
                confidence_parts.append(0.5)
        else:
            missing.append("hy_spread_bps")

        if snap.vix is not None:
            if snap.vix < self._VIX_SUPPRESSED:
                met.append(f"VIX {snap.vix:.1f} (policy not responding yet)")
                confidence_parts.append(min(1.0, (self._VIX_SUPPRESSED - snap.vix) / 10))
        else:
            missing.append("vix")

        if snap.war_active or snap.hormuz_blocked:
            met.append("Geopolitical stress active")
            confidence_parts.append(0.4)

        if snap.private_credit_redemption_pct is not None:
            if snap.private_credit_redemption_pct > 5:
                met.append(f"Private credit redemptions {snap.private_credit_redemption_pct:.0f}%")
                confidence_parts.append(min(1.0, snap.private_credit_redemption_pct / 15))
        else:
            missing.append("private_credit_redemption_pct")

        fired = len(met) >= 2
        confidence = (sum(confidence_parts) / max(len(confidence_parts), 1)) if fired else 0.0

        return FormulaResult(
            tag=FormulaTag.F4_POLICY_DELAY_TRAP,
            fired=fired,
            confidence=round(confidence, 3),
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Shock move before policy response. Highest convexity zone — options cheapest here.",
            risk_class=SignalRiskClass.CONVEX,
            timeframe_days=(1, 10),
            expression_hint="VIX call spreads + ATM puts on credit/bank ETFs. Size small, defined risk.",
        )

    # ──────────────────────────────────────────────────────────
    # Formula F5: Failed Safe Haven  (FRINGE)
    # ──────────────────────────────────────────────────────────

    def _f5_failed_safe_haven(self, snap: MacroSnapshot) -> FormulaResult:
        met, missing = [], []
        confidence_parts: List[float] = []

        if snap.spy_return_1d is not None:
            if snap.spy_return_1d < -0.5:
                met.append(f"Risk-off day: SPY {snap.spy_return_1d:.2f}%")
                confidence_parts.append(0.3)
        else:
            missing.append("spy_return_1d")

        # If gold is NOT rallying on a risk-off day, that's the signal
        if snap.gold_price is not None and snap.spy_return_1d is not None:
            if snap.spy_return_1d < -0.5 and snap.safe_haven_bid is False:
                met.append("Safe havens not bidding on risk-off day (balance sheet stress)")
                confidence_parts.append(0.9)
        else:
            missing.append("gold_price / safe_haven_bid")

        if snap.hy_spread_bps is not None:
            if snap.hy_spread_bps > self._HY_SEVERE_BPS:
                met.append(f"HY spreads severe: {snap.hy_spread_bps:.0f}bps")
                confidence_parts.append(0.6)
        else:
            missing.append("hy_spread_bps (severe level)")

        fired = len(met) >= 2
        confidence = (sum(confidence_parts) / max(len(confidence_parts), 1)) if fired else 0.0

        return FormulaResult(
            tag=FormulaTag.F5_FAILED_SAFE_HAVEN,
            fired=fired,
            confidence=round(confidence, 3),
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Systemic deleveraging event. Forced selling across asset classes. Rare but decisive.",
            risk_class=SignalRiskClass.FRINGE,
            timeframe_days=(1, 7),
            expression_hint="Crash puts (deep OTM index) + VIX calls. Very short window.",
        )

    # ──────────────────────────────────────────────────────────
    # Formula F6: Correlation Spike Predictor  (FRINGE)
    # ──────────────────────────────────────────────────────────

    def _f6_correlation_spike(self, snap: MacroSnapshot) -> FormulaResult:
        met, missing = [], []
        confidence_parts: List[float] = []

        returns = [
            snap.spy_return_1d, snap.hyg_return_1d,
            snap.kre_return_1d, snap.airlines_return_1d,
            snap.shipping_return_1d,
        ]
        known = [r for r in returns if r is not None]
        if len(known) >= 3:
            all_neg = sum(1 for r in known if r < -0.3)
            if all_neg >= int(len(known) * 0.7):
                met.append(f"{all_neg}/{len(known)} sectors red simultaneously (correlation spiking)")
                confidence_parts.append(min(1.0, all_neg / len(known)))
        else:
            missing.append("multi-sector return data")

        if snap.vix_change_1d is not None:
            if snap.vix_change_1d > 10:
                met.append(f"VIX surging {snap.vix_change_1d:.1f}% (correlation cascades live)")
                confidence_parts.append(min(1.0, snap.vix_change_1d / 30))
        else:
            missing.append("vix_change_1d")

        fired = len(met) >= 1 and len(confidence_parts) > 0
        confidence = (sum(confidence_parts) / max(len(confidence_parts), 1)) if fired else 0.0

        return FormulaResult(
            tag=FormulaTag.F6_CORRELATION_SPIKE,
            fired=fired,
            confidence=round(confidence, 3),
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Deleveraging event. Everything moves together. Risk-parity rebalancing.",
            risk_class=SignalRiskClass.FRINGE,
            timeframe_days=(1, 5),
            expression_hint="Index puts + VIX convexity. Exit into panic, not through it.",
        )

    # ──────────────────────────────────────────────────────────
    # Formula F7: Volatility Compression Bomb  (CONVEX)
    # ──────────────────────────────────────────────────────────

    def _f7_vol_compression_bomb(self, snap: MacroSnapshot) -> FormulaResult:
        met, missing = [], []
        confidence_parts: List[float] = []

        if snap.vix is not None:
            if snap.vix < self._VOL_COMPRESSION_THRESHOLD:
                met.append(f"VIX {snap.vix:.1f} — compressed (spring-loading)")
                confidence_parts.append(min(1.0, (self._VOL_COMPRESSION_THRESHOLD - snap.vix) / 10))
        else:
            missing.append("vix")

        if snap.hy_spread_bps is not None and snap.vix is not None:
            # Divergence: credit stress rising but vol not responding
            if snap.hy_spread_bps > 300 and snap.vix < 22:
                met.append(f"Credit/vol divergence: HY {snap.hy_spread_bps:.0f}bps vs VIX {snap.vix:.1f}")
                confidence_parts.append(0.7)
        else:
            missing.append("credit/vol divergence check")

        if snap.breakeven_inflation is not None and snap.yield_curve_10_2 is not None:
            if snap.breakeven_inflation > 2.5 and snap.yield_curve_10_2 < 0:
                met.append("Inflation high, curve inverted — stagflation vol trap")
                confidence_parts.append(0.5)
        else:
            missing.append("breakeven_inflation / yield_curve")

        fired = len(met) >= 2
        confidence = (sum(confidence_parts) / max(len(confidence_parts), 1)) if fired else 0.0

        return FormulaResult(
            tag=FormulaTag.F7_VOL_COMPRESSION_BOMB,
            fired=fired,
            confidence=round(confidence, 3),
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Violent vol spike when catalyst hits. Options cheapest now. Shock usually 1-5 sessions after confirmation.",
            risk_class=SignalRiskClass.CONVEX,
            timeframe_days=(1, 21),
            expression_hint="VIX call spreads (1-3M). Small size. Expect decay before explosion.",
        )

    # ──────────────────────────────────────────────────────────
    # Formula F8: Narrative Break  (CONVEX)
    # ──────────────────────────────────────────────────────────

    def _f8_narrative_break(self, snap: MacroSnapshot) -> FormulaResult:
        met, missing = [], []
        confidence_parts: List[float] = []

        # Proxy: private credit stress + market initially ignoring it
        if snap.private_credit_redemption_pct is not None:
            if snap.private_credit_redemption_pct > 8:
                met.append(f"Private credit redemptions {snap.private_credit_redemption_pct:.0f}% (narrative fracture risk)")
                confidence_parts.append(min(1.0, snap.private_credit_redemption_pct / 15))
        else:
            missing.append("private_credit_redemption_pct")

        if snap.spy_return_1d is not None and snap.fear_greed is not None:
            if snap.spy_return_1d > 0 and snap.fear_greed > 50:
                met.append("Market complacent (spy up, greed > 50) while structural stress builds")
                confidence_parts.append(0.6)
        else:
            missing.append("spy_return_1d / fear_greed")

        if snap.oil_price is not None and snap.war_active:
            if snap.oil_price > self._OIL_SHOCK:
                met.append(f"Oil ${snap.oil_price:.0f}/bbl during active war — narrative anchor breaking")
                confidence_parts.append(0.7)
        else:
            missing.append("oil shock + war check")

        fired = len(met) >= 2
        confidence = (sum(confidence_parts) / max(len(confidence_parts), 1)) if fired else 0.0

        return FormulaResult(
            tag=FormulaTag.F8_NARRATIVE_BREAK,
            fired=fired,
            confidence=round(confidence, 3),
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Trend acceleration once disbelief flips. Second move larger than first. Historical examples: 'this bank is fine', 'inflation is transitory'.",
            risk_class=SignalRiskClass.CONVEX,
            timeframe_days=(3, 60),
            expression_hint="Long-dated puts (BDC, private credit proxies). Small size, long time.",
        )

    # ──────────────────────────────────────────────────────────
    # Formula F9: Leverage Reveal  (CONVEX)
    # ──────────────────────────────────────────────────────────

    def _f9_leverage_reveal(self, snap: MacroSnapshot) -> FormulaResult:
        met, missing = [], []
        confidence_parts: List[float] = []

        # Small shock producing outsized reaction = leverage in the system
        if snap.vix_change_1d is not None and snap.spy_return_1d is not None:
            # If market moved <1% but VIX jumped >15%
            if abs(snap.spy_return_1d) < 1.5 and snap.vix_change_1d > 15:
                met.append(f"Outsized VIX reaction ({snap.vix_change_1d:.0f}%) to small move ({snap.spy_return_1d:.2f}%)")
                confidence_parts.append(min(1.0, snap.vix_change_1d / 40))
        else:
            missing.append("vix_change_1d / spy_return_1d")

        if snap.hy_spread_bps is not None:
            if snap.hy_spread_bps > self._HY_SEVERE_BPS:
                met.append(f"HY spreads revealing leverage: {snap.hy_spread_bps:.0f}bps")
                confidence_parts.append(0.6)
        else:
            missing.append("hy_spread_bps (leverage signal)")

        if snap.private_credit_redemption_pct is not None:
            if snap.private_credit_redemption_pct > 10:
                met.append("Forced redemptions = leverage unwind underway")
                confidence_parts.append(0.8)
        else:
            missing.append("private_credit_redemption_pct")

        fired = len(met) >= 2
        confidence = (sum(confidence_parts) / max(len(confidence_parts), 1)) if fired else 0.0

        return FormulaResult(
            tag=FormulaTag.F9_LEVERAGE_REVEAL,
            fired=fired,
            confidence=round(confidence, 3),
            conditions_met=met,
            conditions_missing=missing,
            expected_outcome="Forced unwinds. Liquidity evaporates. Career-making when caught. Risk: timing is everything.",
            risk_class=SignalRiskClass.CONVEX,
            timeframe_days=(1, 14),
            expression_hint="VIX calls + crash puts (index deep OTM). Treat as lottery ticket, size accordingly.",
        )

    # ──────────────────────────────────────────────────────────
    # Regime classification logic
    # ──────────────────────────────────────────────────────────

    def _classify_regime(
        self,
        results: List[FormulaResult],
        snap: MacroSnapshot,
        vol_score: float,
    ) -> Tuple[Regime, Optional[Regime], float]:
        """Map formula results → primary + secondary regime + confidence."""
        armed = {r.tag for r in results if r.fired}
        scores: Dict[Regime, float] = {r: 0.0 for r in Regime}

        if FormulaTag.F7_VOL_COMPRESSION_BOMB in armed or FormulaTag.F4_POLICY_DELAY_TRAP in armed:
            scores[Regime.VOL_SHOCK_ARMED] += 0.6
        if vol_score >= 80:
            scores[Regime.VOL_SHOCK_ACTIVE] += 0.8
        elif vol_score >= 60:
            scores[Regime.VOL_SHOCK_ARMED] += 0.4

        if FormulaTag.F1_CREDIT_LED_BREAKDOWN in armed:
            scores[Regime.CREDIT_STRESS] += 0.7
        if FormulaTag.F9_LEVERAGE_REVEAL in armed or FormulaTag.F5_FAILED_SAFE_HAVEN in armed:
            scores[Regime.LIQUIDITY_CRUNCH] += 0.7

        if FormulaTag.F2_STAGFLATION_COMPRESSION in armed:
            scores[Regime.STAGFLATION] += 0.7

        if FormulaTag.F3_LIQUIDITY_MIRAGE in armed or FormulaTag.F6_CORRELATION_SPIKE in armed:
            scores[Regime.RISK_OFF] += 0.5

        if FormulaTag.F4_POLICY_DELAY_TRAP in armed:
            scores[Regime.POLICY_DELAY_TRAP] += 0.5

        # Macro boosters
        if snap.hy_spread_bps and snap.hy_spread_bps < 200:
            scores[Regime.RISK_ON] += 0.5
        if snap.vix and snap.vix < 15 and not armed:
            scores[Regime.RISK_ON] += 0.4

        sorted_regimes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_regimes[0][0]
        primary_score = sorted_regimes[0][1]

        secondary = sorted_regimes[1][0] if sorted_regimes[1][1] > 0.2 else None
        confidence = min(1.0, primary_score)

        if primary_score == 0.0:
            primary = Regime.UNCERTAIN
            confidence = 0.1

        return primary, secondary, confidence

    def _count_signals(self, results: List[FormulaResult], snap: MacroSnapshot) -> Tuple[int, int]:
        bear = sum(1 for r in results if r.fired and r.risk_class in (
            SignalRiskClass.INSTITUTIONAL, SignalRiskClass.FRINGE, SignalRiskClass.CONVEX
        ))
        bull = 0
        if snap.hy_spread_bps and snap.hy_spread_bps < 200:
            bull += 1
        if snap.vix and snap.vix < 15:
            bull += 1
        if snap.fear_greed and snap.fear_greed < 25:
            bull += 1  # contrarian buy
        return bear, bull

    def _build_summary(
        self,
        primary: Regime,
        secondary: Optional[Regime],
        armed: List[FormulaTag],
        vol_score: float,
        bear: int,
        bull: int,
    ) -> str:
        lines = [
            f"REGIME: {primary.value.upper().replace('_', ' ')}",
        ]
        if secondary:
            lines.append(f"SECONDARY: {secondary.value.upper().replace('_', ' ')}")
        if armed:
            names = ", ".join(t.value for t in armed)
            lines.append(f"FIRED FORMULAS: {names}")
        lines.append(f"VOL SHOCK READINESS: {vol_score:.0f}/100")
        if vol_score >= 80:
            lines.append("⚠️  IMMINENT — shock window OPEN")
        elif vol_score >= 60:
            lines.append("⚡ ARMED — convexity accumulation phase")
        lines.append(f"SIGNALS: {bear} bearish / {bull} bullish")
        return " | ".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: build snapshot from FRED + common API dicts
# ═══════════════════════════════════════════════════════════════════════════

def snapshot_from_fred(fred_data: Dict[str, float], **kwargs: Any) -> MacroSnapshot:
    """
    Build a MacroSnapshot from FRED series values.

    fred_data keys (FRED series IDs):
        VIXCLS, BAMLH0A0HYM2, T10Y2Y, DCOILWTICO, GOLDAMGBD228NLBM,
        T10YIE (breakeven), DGS10, PCEPILFE (core PCE proxy), DEXUSEU

    Additional kwargs can override any MacroSnapshot field.

    Example:
        from integrations.fred_client import FredClient
        fc = FredClient()
        data = fc.latest_values(['VIXCLS', 'BAMLH0A0HYM2', 'T10Y2Y', 'DCOILWTICO'])
        snap = snapshot_from_fred(data, war_active=True)
    """
    snap = MacroSnapshot(
        vix=fred_data.get("VIXCLS"),
        hy_spread_bps=fred_data.get("BAMLH0A0HYM2") and fred_data["BAMLH0A0HYM2"] * 100,
        yield_curve_10_2=fred_data.get("T10Y2Y"),
        oil_price=fred_data.get("DCOILWTICO"),
        gold_price=fred_data.get("GOLDAMGBD228NLBM"),
        breakeven_inflation=fred_data.get("T10YIE"),
        yield_10y=fred_data.get("DGS10"),
        core_pce=fred_data.get("PCEPILFE"),
        dollar_index=fred_data.get("DTWEXBGS"),
    )
    for k, v in kwargs.items():
        if hasattr(snap, k):
            setattr(snap, k, v)
    return snap
