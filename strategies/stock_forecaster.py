"""
Multi-Industry Stock Forecaster — AAC Forecaster Layer
=======================================================
Maps a RegimeState from the regime engine onto industry/sector forecasts.
Generates ranked trade expressions for both short (0-15 days) and medium (1-6 months)
timeframes with defined-risk option structures.

Industry taxonomy:
  CREDIT          — HYG, JNK, BKLN, LQD (credit ETFs as regime proxies)
  BANKS           — KRE, IAT, XLF (regional & broad financials)
  PRIVATE_CREDIT  — ARCC, OBDC, OWL, MAIN, BXSL (BDC/private credit proxies)
  SHIPPING        — ZIM, GNK, GOGL, DAC, MATX (container + dry bulk)
  AIRLINES        — JETS, DAL, AAL, UAL (oil shock victims)
  INSURANCE       — KIE, AIG, ALL, PRU (asset + underwriting stress)
  ENERGY          — XLE, CVX, XOM (oil shock WINNERS — do NOT short these)
  TECH            — QQQ, SMH (multiple compression in stagflation)
  CONSUMER        — XLY, XRT (demand destruction)
  INDEX           — SPY, QQQ, IWM (broad expression)

Ranking methodology:
  Each opportunity is scored on 3 dimensions (0-100 each):
    roi_score    — expected payoff vs premium paid if thesis correct
    risk_score   — survivability (lower = more dangerous)
    speed_score  — how fast gains materialise
  composite = roi_score * 0.40 + speed_score * 0.35 + (100 - risk_score) * 0.25

Live API feeds accepted:
  - Price data from Finnhub or IBKR via prices dict
  - Regime state from RegimeEngine.evaluate()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from strategies.regime_engine import FormulaTag, Regime, RegimeState, SignalRiskClass

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# INDUSTRY + EXPRESSION TYPES
# ═══════════════════════════════════════════════════════════════════════════

class Industry(Enum):
    CREDIT = "credit"
    BANKS = "banks"
    PRIVATE_CREDIT = "private_credit"
    SHIPPING = "shipping"
    AIRLINES = "airlines"
    INSURANCE = "insurance"
    ENERGY = "energy"
    TECH = "tech"
    CONSUMER = "consumer"
    INDEX = "index"


class Direction(Enum):
    BEARISH = "bearish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"


class ExpressionType(Enum):
    ATM_PUT = "atm_put"
    OTM_PUT = "otm_put"
    PUT_SPREAD = "put_spread"
    NAKED_PUT = "naked_put"        # high risk — presented as info only
    CRASH_PUT = "crash_put"        # deep OTM
    VIX_CALL_SPREAD = "vix_call_spread"
    AVOID = "avoid"


class Horizon(Enum):
    INTRADAY = "intraday"           # same day
    SHORT = "short"                 # 0-15 trading days
    MEDIUM = "medium"               # 1-6 months


class FailureMode(Enum):
    POLICY_BACKSTOP = "policy_backstop"
    BAILOUT_HEADLINE = "bailout_headline"
    SUDDEN_OIL_REVERSAL = "sudden_oil_reversal"
    CREDIT_RECOVERY = "credit_recovery"
    EARNINGS_BEAT = "earnings_beat"
    DIVIDEND_HELD = "dividend_held"


# ═══════════════════════════════════════════════════════════════════════════
# TRADE OPPORTUNITY
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradeOpportunity:
    """A single ranked short-side trade opportunity."""
    rank: int
    industry: Industry
    direction: Direction
    tickers: List[str]               # ordered: best proxy first
    primary_ticker: str
    horizon: Horizon
    expression: ExpressionType

    # Brief
    thesis: str                      # one-sentence "what breaks"
    catalyst: str                    # what triggers the move
    failure_modes: List[FailureMode]

    # Scoring (0-100 each)
    roi_score: float
    risk_score: float                # higher = more dangerous
    speed_score: float
    composite_score: float

    # Structure hint
    structure_hint: str              # e.g. "ATM put, 2-5 weeks, put spread preferred"
    expiry_range_days: Tuple[int, int]   # (min_dte, max_dte) for options
    otm_pct: float                   # how far OTM (0 = ATM, 0.05 = 5% OTM)

    # Risk class
    risk_class: SignalRiskClass
    formula_sources: List[FormulaTag]  # which formulas generated this

    # Context
    timestamp: datetime = field(default_factory=datetime.utcnow)
    note: str = ""


@dataclass
class IndustryForecast:
    """All forecasts + ranked opportunities for a given regime state."""
    regime_state: RegimeState
    horizon: Horizon
    opportunities: List[TradeOpportunity]
    top_3: List[TradeOpportunity]
    industry_map: Dict[Industry, List[TradeOpportunity]]

    @property
    def best(self) -> Optional[TradeOpportunity]:
        return self.opportunities[0] if self.opportunities else None

    def print_plan(self, max_rows: int = 10) -> str:
        lines = [
            "=" * 70,
            f"STOCK FORECASTER — {self.horizon.value.upper()} HORIZON",
            f"Regime: {self.regime_state.primary_regime.value.upper().replace('_', ' ')}",
            f"Vol Shock Readiness: {self.regime_state.vol_shock_readiness:.0f}/100",
            "=" * 70,
        ]
        for i, opp in enumerate(self.opportunities[:max_rows], 1):
            stars = "⭐" * min(5, round(opp.composite_score / 20))
            lines.append(
                f"#{i:02d} [{opp.industry.value.upper():15s}] "
                f"{opp.primary_ticker:6s} | "
                f"{opp.expression.value.replace('_', ' '):15s} | "
                f"ROI:{opp.roi_score:.0f} SPD:{opp.speed_score:.0f} RSK:{opp.risk_score:.0f} "
                f"| {stars}"
            )
            lines.append(f"     Thesis: {opp.thesis}")
            lines.append(f"     Structure: {opp.structure_hint}")
            lines.append(f"     Failure: {', '.join(f.value for f in opp.failure_modes)}")
            lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# INDUSTRY PLAYBOOK (static definitions, regime-independent)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IndustrySpec:
    """Static definition of how an industry behaves in each regime."""
    industry: Industry
    tickers: List[str]
    primary_ticker: str
    base_roi_score: float
    base_risk_score: float
    base_speed_score: float
    thesis: str
    catalyst: str
    failure_modes: List[FailureMode]
    expression: ExpressionType
    short_expiry: Tuple[int, int]     # (min_dte, max_dte) for short horizon
    medium_expiry: Tuple[int, int]    # for medium horizon
    otm_pct: float
    crisis_vectors: List[str]         # which regimes activate this
    warn_do_not_short: bool = False   # e.g. energy in oil shock = DON'T SHORT


INDUSTRY_PLAYBOOK: List[IndustrySpec] = [

    # ── TIER 1: CREDIT (fastest signal)
    IndustrySpec(
        industry=Industry.CREDIT,
        tickers=["HYG", "JNK", "BKLN", "LQD"],
        primary_ticker="HYG",
        base_roi_score=92,
        base_risk_score=30,
        base_speed_score=95,
        thesis="High-yield credit reprices before equities when defaults/refinancing stress surfaces. Institutions hedge here first.",
        catalyst="HY spread widening, leveraged loan downgrades, private credit mark-downs",
        failure_modes=[FailureMode.POLICY_BACKSTOP, FailureMode.CREDIT_RECOVERY],
        expression=ExpressionType.PUT_SPREAD,
        short_expiry=(14, 42),
        medium_expiry=(60, 120),
        otm_pct=0.03,
        crisis_vectors=["credit_stress", "liquidity_crunch", "vol_shock_armed"],
    ),

    # ── TIER 1: REGIONAL BANKS (contagion transmission)
    IndustrySpec(
        industry=Industry.BANKS,
        tickers=["KRE", "IAT", "XLF"],
        primary_ticker="KRE",
        base_roi_score=88,
        base_risk_score=40,
        base_speed_score=88,
        thesis="Banks are the transmission vector of credit stress. Regional banks gap, not grind, when funding tightens.",
        catalyst="Credit event, deposit flight, CRE markdowns, capital adequacy concerns",
        failure_modes=[FailureMode.BAILOUT_HEADLINE, FailureMode.POLICY_BACKSTOP],
        expression=ExpressionType.PUT_SPREAD,
        short_expiry=(14, 42),
        medium_expiry=(60, 180),
        otm_pct=0.05,
        crisis_vectors=["credit_stress", "liquidity_crunch", "vol_shock_armed", "vol_shock_active"],
    ),

    # ── TIER 2: SHIPPING (pure operating leverage on volumes)
    IndustrySpec(
        industry=Industry.SHIPPING,
        tickers=["ZIM", "GNK", "GOGL", "DAC"],
        primary_ticker="ZIM",
        base_roi_score=82,
        base_risk_score=50,
        base_speed_score=82,
        thesis="Extreme operating leverage on trade volumes/rates. Earnings collapse fast when volumes stall.",
        catalyst="Freight rate collapse, global trade slowdown, demand destruction from stagflation",
        failure_modes=[FailureMode.SUDDEN_OIL_REVERSAL],  # rate spikes can squeeze
        expression=ExpressionType.ATM_PUT,
        short_expiry=(7, 21),
        medium_expiry=(30, 90),
        otm_pct=0.0,
        crisis_vectors=["stagflation", "risk_off", "credit_stress"],
    ),

    # ── TIER 2: AIRLINES (oil shock direct victim)
    IndustrySpec(
        industry=Industry.AIRLINES,
        tickers=["JETS", "DAL", "AAL", "UAL"],
        primary_ticker="JETS",
        base_roi_score=78,
        base_risk_score=55,
        base_speed_score=78,
        thesis="Oil at $120+ = fuel death + demand destruction. Airlines can't hedge effectively. Margins evaporate fast.",
        catalyst="Oil spike, demand guidance cut, recurring loss quarters",
        failure_modes=[FailureMode.BAILOUT_HEADLINE, FailureMode.SUDDEN_OIL_REVERSAL],
        expression=ExpressionType.ATM_PUT,
        short_expiry=(7, 21),
        medium_expiry=(30, 60),
        otm_pct=0.0,
        crisis_vectors=["stagflation", "risk_off"],
    ),

    # ── TIER 3: INSURANCE (slower, but real)
    IndustrySpec(
        industry=Industry.INSURANCE,
        tickers=["KIE", "AIG", "ALL", "PRU"],
        primary_ticker="KIE",
        base_roi_score=72,
        base_risk_score=45,
        base_speed_score=55,
        thesis="Asset-side losses from credit shock + underwriting stress. Capital ratios compress. Market reprices 'safety' assumptions.",
        catalyst="Credit portfolio markdowns, reserve releases reverse, cat event on top of financial stress",
        failure_modes=[FailureMode.EARNINGS_BEAT, FailureMode.POLICY_BACKSTOP],
        expression=ExpressionType.PUT_SPREAD,
        short_expiry=(28, 60),
        medium_expiry=(90, 180),
        otm_pct=0.05,
        crisis_vectors=["credit_stress", "liquidity_crunch", "stagflation"],
    ),

    # ── TIER 3 (DELAYED BOMB): PRIVATE CREDIT PROXIES
    IndustrySpec(
        industry=Industry.PRIVATE_CREDIT,
        tickers=["ARCC", "OBDC", "OWL", "MAIN", "BXSL"],
        primary_ticker="ARCC",
        base_roi_score=85,
        base_risk_score=60,
        base_speed_score=40,
        thesis="Dividend illusion hides insolvency until it doesn't. NAV marks delayed (mark-to-myth). Once dividend cut announced, cliff drop.",
        catalyst="Non-accrual rise, dividend cut, NAV markdown, redemption gate wider",
        failure_modes=[FailureMode.DIVIDEND_HELD, FailureMode.POLICY_BACKSTOP],
        expression=ExpressionType.OTM_PUT,
        short_expiry=(56, 90),
        medium_expiry=(120, 270),
        otm_pct=0.08,
        crisis_vectors=["credit_stress", "policy_delay_trap"],
    ),

    # ── TECH (multiple compression in stagflation)
    IndustrySpec(
        industry=Industry.TECH,
        tickers=["QQQ", "SMH", "ARKK"],
        primary_ticker="QQQ",
        base_roi_score=65,
        base_risk_score=50,
        base_speed_score=60,
        thesis="P/E compression in stagflation + rate spike. Long-duration growth assets re-rate lower as real yields rise.",
        catalyst="Rate spike, PCE print above expectations, Fed hawkish surprise",
        failure_modes=[FailureMode.POLICY_BACKSTOP, FailureMode.EARNINGS_BEAT],
        expression=ExpressionType.PUT_SPREAD,
        short_expiry=(14, 42),
        medium_expiry=(60, 180),
        otm_pct=0.05,
        crisis_vectors=["stagflation", "vol_shock_armed"],
    ),

    # ── CONSUMER DISCRETIONARY
    IndustrySpec(
        industry=Industry.CONSUMER,
        tickers=["XLY", "XRT", "AMZN"],
        primary_ticker="XLY",
        base_roi_score=62,
        base_risk_score=48,
        base_speed_score=58,
        thesis="Consumer squeezed by oil shock inflation + wage stagnation. Discretionary demand elastic → first cut.",
        catalyst="Consumer sentiment collapse, retail sales miss, personal savings rate drop",
        failure_modes=[FailureMode.POLICY_BACKSTOP, FailureMode.EARNINGS_BEAT],
        expression=ExpressionType.PUT_SPREAD,
        short_expiry=(14, 42),
        medium_expiry=(60, 120),
        otm_pct=0.05,
        crisis_vectors=["stagflation", "risk_off"],
    ),

    # ── ENERGY: DO NOT SHORT IN THIS SCENARIO
    IndustrySpec(
        industry=Industry.ENERGY,
        tickers=["XLE", "CVX", "XOM", "OXY"],
        primary_ticker="XLE",
        base_roi_score=0,
        base_risk_score=90,
        base_speed_score=0,
        thesis="ENERGY IS A WINNER IN OIL SHOCK. Do NOT short producers. They can rip even if market is down.",
        catalyst="N/A — avoid shorting in this regime",
        failure_modes=[],
        expression=ExpressionType.AVOID,
        short_expiry=(0, 0),
        medium_expiry=(0, 0),
        otm_pct=0.0,
        crisis_vectors=["stagflation"],
        warn_do_not_short=True,
    ),

    # ── BROAD INDEX (support role only)
    IndustrySpec(
        industry=Industry.INDEX,
        tickers=["SPY", "QQQ", "IWM"],
        primary_ticker="SPY",
        base_roi_score=55,
        base_risk_score=50,
        base_speed_score=60,
        thesis="Index follows credit, doesn't lead. Used to monetize volatility spikes and as catch-all hedge.",
        catalyst="Correlation spike, cascading credit events",
        failure_modes=[FailureMode.POLICY_BACKSTOP],
        expression=ExpressionType.PUT_SPREAD,
        short_expiry=(14, 30),
        medium_expiry=(60, 120),
        otm_pct=0.05,
        crisis_vectors=["vol_shock_active", "liquidity_crunch"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# STOCK FORECASTER
# ═══════════════════════════════════════════════════════════════════════════

class StockForecaster:
    """
    Maps a RegimeState → ranked TradeOpportunity list.

    Usage:
        from strategies.regime_engine import RegimeEngine, MacroSnapshot
        from strategies.stock_forecaster import StockForecaster, Horizon

        engine = RegimeEngine()
        snap = MacroSnapshot(vix=21.5, hy_spread_bps=380, ...)
        state = engine.evaluate(snap)

        forecaster = StockForecaster()
        forecast = forecaster.forecast(state, horizon=Horizon.SHORT)
        print(forecast.print_plan())
    """

    # Regime → which crisis_vectors are active
    _REGIME_VECTORS = {
        Regime.CREDIT_STRESS: {"credit_stress"},
        Regime.STAGFLATION: {"stagflation"},
        Regime.LIQUIDITY_CRUNCH: {"liquidity_crunch"},
        Regime.VOL_SHOCK_ARMED: {"vol_shock_armed", "credit_stress"},
        Regime.VOL_SHOCK_ACTIVE: {"vol_shock_active", "credit_stress", "liquidity_crunch"},
        Regime.POLICY_DELAY_TRAP: {"policy_delay_trap", "credit_stress"},
        Regime.RISK_OFF: {"risk_off"},
        Regime.UNCERTAIN: set(),
        Regime.RISK_ON: set(),
    }

    # Formula → ROI boost applied to matching opportunities
    _FORMULA_BOOSTS: Dict[FormulaTag, Dict[Industry, float]] = {
        FormulaTag.F1_CREDIT_LED_BREAKDOWN: {
            Industry.CREDIT: 15, Industry.BANKS: 12, Industry.PRIVATE_CREDIT: 8
        },
        FormulaTag.F2_STAGFLATION_COMPRESSION: {
            Industry.AIRLINES: 15, Industry.SHIPPING: 10, Industry.CONSUMER: 8, Industry.TECH: 8
        },
        FormulaTag.F3_LIQUIDITY_MIRAGE: {
            Industry.INDEX: 20, Industry.BANKS: 10
        },
        FormulaTag.F4_POLICY_DELAY_TRAP: {
            Industry.CREDIT: 10, Industry.BANKS: 8, Industry.PRIVATE_CREDIT: 12
        },
        FormulaTag.F5_FAILED_SAFE_HAVEN: {
            Industry.CREDIT: 12, Industry.BANKS: 15, Industry.INDEX: 18
        },
        FormulaTag.F7_VOL_COMPRESSION_BOMB: {
            Industry.CREDIT: 8, Industry.BANKS: 8, Industry.INDEX: 15
        },
        FormulaTag.F8_NARRATIVE_BREAK: {
            Industry.PRIVATE_CREDIT: 20, Industry.BANKS: 10
        },
        FormulaTag.F9_LEVERAGE_REVEAL: {
            Industry.BANKS: 15, Industry.CREDIT: 12, Industry.INDEX: 10
        },
    }

    def forecast(
        self,
        state: RegimeState,
        horizon: Horizon = Horizon.SHORT,
        prices: Optional[Dict[str, float]] = None,
        top_n: int = 10,
    ) -> IndustryForecast:
        """Generate ranked trade opportunities for the given regime state."""
        active_vectors = self._REGIME_VECTORS.get(state.primary_regime, set())
        if state.secondary_regime:
            active_vectors |= self._REGIME_VECTORS.get(state.secondary_regime, set())

        opportunities: List[TradeOpportunity] = []
        for i, spec in enumerate(INDUSTRY_PLAYBOOK):
            if spec.warn_do_not_short or spec.expression == ExpressionType.AVOID:
                continue

            # Check if this industry is activated by current regime
            relevance = len(set(spec.crisis_vectors) & active_vectors)
            if relevance == 0 and state.primary_regime not in (Regime.RISK_OFF,):
                # Still include if regime is generically bearish, but with lower score
                if state.bear_signals < 2:
                    continue

            roi = spec.base_roi_score
            risk = spec.base_risk_score
            speed = spec.base_speed_score

            # Apply formula boosts
            formula_sources = []
            for formula_tag in state.armed_formulas:
                boosts = self._FORMULA_BOOSTS.get(formula_tag, {})
                if spec.industry in boosts:
                    roi = min(100, roi + boosts[spec.industry] * 0.7)
                    speed = min(100, speed + boosts[spec.industry] * 0.3)
                    formula_sources.append(formula_tag)

            # Scale by regime confidence
            roi *= state.regime_confidence
            speed *= state.regime_confidence

            # Adjust risk: policy_delay_trap = cheapest options = lower risk for defined-risk
            if state.primary_regime == Regime.POLICY_DELAY_TRAP and spec.expression in (
                ExpressionType.PUT_SPREAD, ExpressionType.ATM_PUT
            ):
                risk = max(10, risk - 15)

            # Horizon adjustments
            if horizon == Horizon.SHORT:
                expiry = spec.short_expiry
                speed = min(100, speed * 1.1)
            else:
                expiry = spec.medium_expiry
                # Medium term: private credit + insurance shine
                if spec.industry in (Industry.PRIVATE_CREDIT, Industry.INSURANCE):
                    roi = min(100, roi + 10)

            composite = roi * 0.40 + speed * 0.35 + (100 - risk) * 0.25
            composite = round(composite, 1)

            opp = TradeOpportunity(
                rank=0,  # assigned after sort
                industry=spec.industry,
                direction=Direction.BEARISH,
                tickers=spec.tickers,
                primary_ticker=spec.primary_ticker,
                horizon=horizon,
                expression=spec.expression,
                thesis=spec.thesis,
                catalyst=spec.catalyst,
                failure_modes=spec.failure_modes,
                roi_score=round(roi, 1),
                risk_score=round(risk, 1),
                speed_score=round(speed, 1),
                composite_score=composite,
                structure_hint=self._build_structure_hint(spec, horizon, expiry),
                expiry_range_days=expiry,
                otm_pct=spec.otm_pct,
                risk_class=(
                    SignalRiskClass.INSTITUTIONAL if relevance > 0
                    else SignalRiskClass.FRINGE
                ),
                formula_sources=formula_sources,
            )
            opportunities.append(opp)

        # Sort by composite score descending
        opportunities.sort(key=lambda x: x.composite_score, reverse=True)
        for i, opp in enumerate(opportunities, 1):
            opp.rank = i

        top_3 = opportunities[:3]
        industry_map: Dict[Industry, List[TradeOpportunity]] = {}
        for opp in opportunities:
            industry_map.setdefault(opp.industry, []).append(opp)

        return IndustryForecast(
            regime_state=state,
            horizon=horizon,
            opportunities=opportunities[:top_n],
            top_3=top_3,
            industry_map=industry_map,
        )

    def _build_structure_hint(
        self,
        spec: IndustrySpec,
        horizon: Horizon,
        expiry: Tuple[int, int],
    ) -> str:
        min_dte, max_dte = expiry
        expr = spec.expression.value.replace("_", " ").title()
        otm_str = f" ({spec.otm_pct*100:.0f}% OTM)" if spec.otm_pct > 0 else " (ATM)"
        horizon_str = "Short-term" if horizon == Horizon.SHORT else "Medium-term"
        return (
            f"{horizon_str} | {expr}{otm_str} | "
            f"DTE: {min_dte}-{max_dte} days | "
            f"Ticker: {spec.primary_ticker} (or {', '.join(spec.tickers[1:3])})"
        )

    # ── Convenience: two-trade stack (best credit + best banks)

    def two_trade_stack(self, state: RegimeState) -> Tuple[Optional[TradeOpportunity], Optional[TradeOpportunity]]:
        """
        Returns the classic 2-trade stack:
          Trade 1: Best credit expression (anchor)
          Trade 2: Best financials/banks expression (contagion accelerator)
        """
        forecast = self.forecast(state, Horizon.SHORT)
        credit = next((o for o in forecast.opportunities if o.industry == Industry.CREDIT), None)
        banks = next((o for o in forecast.opportunities if o.industry == Industry.BANKS), None)
        return credit, banks

    def top_n_stack(self, state: RegimeState, n: int = 3) -> List[TradeOpportunity]:
        """Returns top N ranked opportunities for today."""
        forecast = self.forecast(state, Horizon.SHORT)
        return forecast.opportunities[:n]


# ═══════════════════════════════════════════════════════════════════════════
# INDUSTRY × REGIME SCORECARD (terminal display)
# ═══════════════════════════════════════════════════════════════════════════

def print_industry_regime_matrix() -> str:
    """Print the full regime × industry outcome matrix (reference table)."""
    MATRIX = {
        "CREDIT_STRESS": {
            "Credit (HYG/JNK)": "⭐⭐⭐⭐⭐ FIRST MOVER",
            "Banks (KRE/IAT)":  "⭐⭐⭐⭐  CONTAGION",
            "Private Credit":    "⭐⭐⭐   DELAYED BOMB",
            "Shipping":          "⭐⭐⭐   SECONDARY",
            "Airlines":          "⭐⭐     INDIRECT",
            "Insurance":         "⭐⭐⭐   MEDIUM TERM",
            "Energy":            "🚫 DO NOT SHORT",
        },
        "STAGFLATION": {
            "Airlines (JETS)":   "⭐⭐⭐⭐⭐ OIL SHOCK DIRECT",
            "Consumer (XLY)":    "⭐⭐⭐⭐  DEMAND CRUSH",
            "Shipping":          "⭐⭐⭐⭐  VOLUME COLLAPSE",
            "Tech (QQQ)":        "⭐⭐⭐   PE COMPRESSION",
            "Banks (KRE)":       "⭐⭐⭐   FUNDING STRESS",
            "Insurance":         "⭐⭐     ASSET SIDE",
            "Energy":            "🚫 DO NOT SHORT",
        },
        "VOL_SHOCK_ARMED": {
            "Credit (HYG)":      "⭐⭐⭐⭐⭐ CHEAPEST ENTRY NOW",
            "VIX CALLS":         "⭐⭐⭐⭐⭐ CONVEXITY OVERLAY",
            "Banks (KRE)":       "⭐⭐⭐⭐  ACCUMULATE PUTS",
            "Index (SPY)":       "⭐⭐⭐   DIRECTIONAL HEDGE",
        },
    }
    lines = ["", "INDUSTRY × REGIME MATRIX", "─" * 60]
    for regime, industries in MATRIX.items():
        lines.append(f"\n  ▶ {regime}")
        for industry, signal in industries.items():
            lines.append(f"    {industry:25s} {signal}")
    return "\n".join(lines)
