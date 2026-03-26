"""
Storm Lifeboat Capital Engine — Gold–Oil–Silver See-Saw
=========================================================
Hourly engine that scrapes live data, analyzes the gold-oil-silver
see-saw mechanics, and generates time-sensitive trading signals with
stop-loss protection.

Core thesis:
    Oil spike (Hormuz/supply shock) → inflation → stocks drop
    → Gold lags then surges (inflation hedge)
    → Silver amplifies 2-3× (monetary + industrial)
    → Rotation: oil → gold → silver as the cycle progresses

Integrates with:
    - LiveFeedEngine   — real-time prices from Polygon, FRED, Finnhub
    - ScenarioEngine   — 43 crisis scenarios with contagion
    - LunarPhiEngine   — phi-cycle timing for entries/exits
    - CoherenceEngine  — PlanckPhire alignment scoring
    - CentralAccounting — SQLite P&L persistence

Run standalone:
    python -m strategies.storm_lifeboat.capital_engine

Or hook into the AAC pipeline:
    from strategies.storm_lifeboat.capital_engine import LifeboatCapitalEngine
    engine = LifeboatCapitalEngine()
    await engine.run_hourly()  # single cycle
    await engine.run_forever() # blocking loop
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from strategies.storm_lifeboat.core import (
    Asset,
    DEFAULT_PRICES,
    MoonPhase,
    ScenarioStatus,
    VolRegime,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# See-saw assets we rotate between
SEESAW_LONG_ASSETS = [Asset.GOLD, Asset.SILVER, Asset.GDX, Asset.XLE, Asset.OIL]
SEESAW_SHORT_ASSETS = [Asset.QQQ, Asset.XLF, Asset.XLRE, Asset.KRE]

# ETF ticker mapping for options (strikes reference these)
ASSET_TO_ETF = {
    Asset.GOLD: "GLD",
    Asset.SILVER: "SLV",
    Asset.OIL: "USO",
    Asset.XLE: "XLE",
    Asset.GDX: "GDX",
    Asset.QQQ: "QQQ",
    Asset.XLF: "XLF",
    Asset.XLRE: "XLRE",
    Asset.KRE: "KRE",
    Asset.SPY: "SPY",
    Asset.BITO: "BITO",
}

# Default allocation weights by see-saw phase
DEFAULT_WEIGHTS = {
    "oil_spike": {"OIL": 0.10, "XLE": 0.20, "GOLD": 0.25, "SILVER": 0.30, "CASH": 0.15},
    "inflation_rotation": {"GOLD": 0.30, "SILVER": 0.35, "GDX": 0.15, "CASH": 0.20},
    "recovery": {"SILVER": 0.25, "GOLD": 0.20, "XLE": 0.15, "QQQ": 0.15, "CASH": 0.25},
    "neutral": {"GOLD": 0.15, "SILVER": 0.10, "XLE": 0.10, "CASH": 0.65},
}


class SeeSawPhase(Enum):
    """Current phase of the gold-oil-silver rotation."""
    NEUTRAL = "neutral"                      # No clear trigger
    OIL_SPIKE = "oil_spike"                  # Oil surging on supply shock
    INFLATION_ROTATION = "inflation_rotation" # Oil embedded, rotating to gold/silver
    GOLD_BREAKOUT = "gold_breakout"          # Gold leading, silver catching up
    SILVER_AMPLIFIER = "silver_amplifier"    # Silver high-beta outperformance
    RECOVERY = "recovery"                    # Stocks bottoming, trim havens


class StopType(Enum):
    FIXED = "fixed"
    TRAILING = "trailing"


# ═══════════════════════════════════════════════════════════════════════════
# POSITION & STOP-LOSS TRACKING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedPosition:
    """A position being tracked by the engine with stop-loss state."""
    asset: Asset
    etf: str                        # Tradeable ticker (GLD, SLV, etc.)
    entry_price: float
    current_price: float
    quantity: float                  # Notional in USD
    entry_time: datetime
    high_water_mark: float          # Highest price since entry (for trailing)
    # Stop-loss configuration
    fixed_stop_pct: float = 0.08    # 8% fixed stop-loss
    trailing_stop_pct: float = 0.05 # 5% trailing stop-loss
    # State
    stopped_out: bool = False
    stop_trigger: str = ""          # Reason for stop if triggered
    pnl_pct: float = 0.0

    def update_price(self, price: float) -> None:
        """Update current price, high water mark, and P&L."""
        self.current_price = price
        if price > self.high_water_mark:
            self.high_water_mark = price
        if self.entry_price > 0:
            self.pnl_pct = (price - self.entry_price) / self.entry_price

    def check_stop_loss(self) -> bool:
        """Check if either stop-loss has been triggered."""
        if self.stopped_out:
            return True

        # Fixed stop: price dropped X% from entry
        if self.pnl_pct <= -self.fixed_stop_pct:
            self.stopped_out = True
            self.stop_trigger = f"FIXED_STOP ({self.fixed_stop_pct:.0%} from entry)"
            return True

        # Trailing stop: price dropped X% from high water mark
        if self.high_water_mark > 0:
            drawdown_from_high = (self.current_price - self.high_water_mark) / self.high_water_mark
            if drawdown_from_high <= -self.trailing_stop_pct:
                self.stopped_out = True
                self.stop_trigger = (
                    f"TRAILING_STOP ({self.trailing_stop_pct:.0%} from "
                    f"high ${self.high_water_mark:.2f})"
                )
                return True

        return False


@dataclass
class SeeSawSignal:
    """A recommended trade from the see-saw analysis."""
    asset: Asset
    etf: str
    direction: str          # "LONG" or "SHORT" or "CLOSE"
    weight: float           # Target portfolio weight (0.0 - 1.0)
    confidence: float       # 0.0 - 1.0
    reason: str
    # Options recommendation (educational)
    option_type: str = ""   # "call", "put", "bull_call_spread", etc.
    strike_hint: str = ""   # e.g. "GLD Jun 2026 420/430"
    off_ramp: str = ""      # e.g. "Moon 3 Blue Moon"


@dataclass
class CycleReport:
    """Output of one hourly analysis cycle."""
    timestamp: datetime
    phase: SeeSawPhase
    vix: float
    regime: VolRegime
    # Prices
    gold_price: float
    silver_price: float
    oil_price: float
    gold_oil_ratio: float
    gold_silver_ratio: float
    # Crypto × Commodity ratios (cross-asset see-saw)
    btc_gold_ratio: float = 0.0
    eth_btc_ratio: float = 0.0
    btc_silver_ratio: float = 0.0
    cross_amplifier_index: float = 0.0
    fear_greed: float = 50.0
    dxy: float = 0.0
    btc_price: float = 0.0
    eth_price: float = 0.0
    # Polymarket prediction market context
    poly_oil_shock_prob: float = 0.0
    poly_gold_reprice_prob: float = 0.0
    poly_recession_prob: float = 0.0
    poly_geopolitical_prob: float = 0.0
    poly_thesis_edge: float = 0.0
    poly_top_opportunities: List[str] = field(default_factory=list)
    # Scenario context
    active_scenarios: List[str] = field(default_factory=list)
    top_firing_scenario: str = ""
    # Lunar
    moon_number: int = 0
    moon_name: str = ""
    moon_phase: str = ""
    phi_window: bool = False
    phi_coherence: float = 0.0
    position_multiplier: float = 1.0
    # Signals
    signals: List[SeeSawSignal] = field(default_factory=list)
    # Positions
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stops_triggered: List[str] = field(default_factory=list)
    # Portfolio
    portfolio_value: float = 0.0
    cash_pct: float = 1.0
    # Data source health
    sources_ok: List[str] = field(default_factory=list)
    sources_failed: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class LifeboatCapitalEngine:
    """
    Hourly engine that trades the gold-oil-silver see-saw rotation.

    Scrapes → Analyzes → Generates signals → Applies stop-losses.

    The engine does NOT submit orders directly. It produces CycleReports
    with SeeSawSignals that the AAC pipeline can route to the execution
    engine (paper or live depending on AAC_ENV).
    """

    def __init__(
        self,
        starting_capital: float = 48_100.0,
        fixed_stop_pct: float = 0.08,
        trailing_stop_pct: float = 0.05,
        cycle_interval_seconds: int = 3600,
        data_dir: Optional[str] = None,
    ):
        self.starting_capital = starting_capital
        self.portfolio_value = starting_capital
        self.cash = starting_capital
        self.fixed_stop_pct = fixed_stop_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.cycle_interval = cycle_interval_seconds

        # Position tracking
        self.positions: Dict[Asset, TrackedPosition] = {}

        # Phase tracking
        self.current_phase = SeeSawPhase.NEUTRAL
        self._prev_oil_price = 0.0
        self._prev_gold_price = 0.0
        self._oil_spike_start: Optional[datetime] = None

        # Data persistence
        self._data_dir = Path(data_dir or "data/storm_lifeboat")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._report_path = self._data_dir / "capital_engine_reports.jsonl"
        self._state_path = self._data_dir / "capital_engine_state.json"

        # Cycle history (in-memory, last 168 = 7 days of hourly)
        self.cycle_history: List[CycleReport] = []
        self._max_history = 168

        # Shutdown flag
        self._shutdown = False

        # Lazy-loaded sub-engines (avoid import-time side effects)
        self._live_feed = None
        self._scenario_engine = None
        self._lunar_engine = None
        self._coherence_engine = None
        self._polymarket_scanner = None

        logger.info(
            "LifeboatCapitalEngine initialized: capital=$%.0f, "
            "stops=%.0f%%/%.0f%%, interval=%ds",
            starting_capital, fixed_stop_pct * 100, trailing_stop_pct * 100,
            cycle_interval_seconds,
        )

    # ───── lazy initialization ─────

    def _get_live_feed(self):
        if self._live_feed is None:
            from strategies.storm_lifeboat.live_feed import LiveFeedEngine
            self._live_feed = LiveFeedEngine(cache_ttl_seconds=90)
        return self._live_feed

    def _get_scenario_engine(self):
        if self._scenario_engine is None:
            from strategies.storm_lifeboat.scenario_engine import ScenarioEngine
            self._scenario_engine = ScenarioEngine()
        return self._scenario_engine

    def _get_lunar_engine(self):
        if self._lunar_engine is None:
            from strategies.storm_lifeboat.lunar_phi import LunarPhiEngine
            self._lunar_engine = LunarPhiEngine()
        return self._lunar_engine

    def _get_coherence_engine(self):
        if self._coherence_engine is None:
            from strategies.storm_lifeboat.coherence import CoherenceEngine
            self._coherence_engine = CoherenceEngine()
        return self._coherence_engine

    def _get_polymarket_scanner(self):
        if self._polymarket_scanner is None:
            from strategies.polymarket_blackswan_scanner import PolymarketBlackSwanScanner
            self._polymarket_scanner = PolymarketBlackSwanScanner()
        return self._polymarket_scanner

    # ═══════════════════════════════════════════════════════════════════════
    # SCRAPE PHASE — Get live data
    # ═══════════════════════════════════════════════════════════════════════

    async def _scrape(self) -> Dict[str, Any]:
        """Fetch live prices, VIX, scenarios, lunar position."""
        feed = self._get_live_feed()
        snap = await feed.fetch_async()

        scenario_eng = self._get_scenario_engine()
        # Fire indicators from live news
        if snap.firing_indicators:
            for code, indicators in snap.firing_indicators.items():
                scenario_eng.update_indicators(code, indicators)

        lunar = self._get_lunar_engine()
        pos = lunar.get_position()

        coherence_eng = self._get_coherence_engine()
        active = scenario_eng.get_active_scenarios()
        coherence = coherence_eng.analyze(
            active_scenarios=active,
            moon_phase=pos.phase,
            lunar_phi_coherence=pos.phi_coherence,
            current_regime=snap.regime,
        )

        return {
            "snapshot": snap,
            "lunar": pos,
            "coherence": coherence,
            "active_scenarios": active,
            "heatmap": scenario_eng.get_risk_heatmap(),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # POLYMARKET SCRAPE — Prediction market probabilities
    # ═══════════════════════════════════════════════════════════════════════

    # Map Polymarket thesis categories → SeeSawPhase implications
    _CATEGORY_TO_PHASE = {
        "iran_war": SeeSawPhase.OIL_SPIKE,
        "us_withdrawal": SeeSawPhase.OIL_SPIKE,
        "israel_conflict": SeeSawPhase.OIL_SPIKE,
        "oil_shock": SeeSawPhase.OIL_SPIKE,
        "gold_reprice": SeeSawPhase.GOLD_BREAKOUT,
        "usd_collapse": SeeSawPhase.INFLATION_ROTATION,
        "yuan_rise": SeeSawPhase.INFLATION_ROTATION,
        "gulf_shift": SeeSawPhase.INFLATION_ROTATION,
        "inflation": SeeSawPhase.INFLATION_ROTATION,
        "fed_trapped": SeeSawPhase.NEUTRAL,
        "recession": SeeSawPhase.NEUTRAL,
        "credit_crisis": SeeSawPhase.NEUTRAL,
        "crypto_crisis": SeeSawPhase.SILVER_AMPLIFIER,
        "geopolitical": SeeSawPhase.OIL_SPIKE,
    }

    async def _scrape_polymarket(self) -> Dict[str, Any]:
        """Fetch Polymarket black swan opportunities, grouped by category.

        Returns a dict with per-category avg crowd probabilities, the top
        opportunities (by edge), and an aggregate thesis-edge score.
        Gracefully returns empty context on failure so the engine never blocks.
        """
        try:
            scanner = self._get_polymarket_scanner()
            opps = await scanner.scan(max_pages=2)
        except Exception as e:
            logger.warning("Polymarket scan failed (engine continues): %s", e)
            return {}

        if not opps:
            return {}

        # Bucket opportunities by category
        by_cat: Dict[str, List[Any]] = {}
        for opp in opps:
            by_cat.setdefault(opp.category, []).append(opp)

        # Aggregate per-phase probability: average market price across opportunities
        phase_probs: Dict[str, float] = {}
        for cat, cat_opps in by_cat.items():
            avg_price = sum(o.market_price for o in cat_opps) / len(cat_opps)
            phase_probs[cat] = avg_price

        # Top 5 by edge
        top5 = opps[:5]
        avg_edge = sum(o.edge for o in opps) / len(opps) if opps else 0.0

        return {
            "phase_probs": phase_probs,         # {category: avg_market_price}
            "top_opportunities": top5,           # List[BlackSwanOpportunity]
            "avg_thesis_edge": avg_edge,
            "total_count": len(opps),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYZE PHASE — Determine see-saw phase and signals
    # ═══════════════════════════════════════════════════════════════════════

    def _detect_phase(self, prices: Dict[Asset, float], vix: float,
                      heatmap: List[Dict],
                      crypto_ctx: Optional[Dict[str, float]] = None,
                      poly_ctx: Optional[Dict[str, Any]] = None,
                      ) -> SeeSawPhase:
        """Classify the current see-saw phase from price dynamics.

        ``crypto_ctx`` supplies cross-asset ratios computed in run_hourly():
            btc_gold_ratio, eth_btc_ratio, btc_silver_ratio,
            cross_amplifier_index, fear_greed, dxy

        ``poly_ctx`` supplies Polymarket prediction market probabilities
        from _scrape_polymarket():
            phase_probs   — {category: avg_crowd_price}
            avg_thesis_edge — average edge across all thesis-aligned opps
        """
        oil = prices.get(Asset.OIL, DEFAULT_PRICES[Asset.OIL])
        gold = prices.get(Asset.GOLD, DEFAULT_PRICES[Asset.GOLD])
        silver = prices.get(Asset.SILVER, DEFAULT_PRICES[Asset.SILVER])

        # Oil spike detection: oil up >5% from previous reading AND VIX elevated
        oil_pct_change = 0.0
        if self._prev_oil_price > 0:
            oil_pct_change = (oil - self._prev_oil_price) / self._prev_oil_price

        gold_pct_change = 0.0
        if self._prev_gold_price > 0:
            gold_pct_change = (gold - self._prev_gold_price) / self._prev_gold_price

        # Count active high-severity scenarios
        active_severe = sum(
            1 for s in heatmap
            if s.get("status") in ("active", "escalating", "peak")
            and s.get("severity", 0) >= 0.70
        )

        # ── Cross-asset crypto context ──
        ctx = crypto_ctx or {}
        btc_gold = ctx.get("btc_gold_ratio", 0.0)
        fear_greed = ctx.get("fear_greed", 50.0)
        cross_amp = ctx.get("cross_amplifier_index", 1.0)
        dxy = ctx.get("dxy", 0.0)

        # ── Polymarket prediction-market context ──
        pctx = poly_ctx or {}
        pp = pctx.get("phase_probs", {})
        # Crowd-implied probabilities for key categories
        poly_oil = max(pp.get("oil_shock", 0), pp.get("iran_war", 0),
                       pp.get("israel_conflict", 0))
        poly_gold = pp.get("gold_reprice", 0)
        poly_recession = max(pp.get("recession", 0), pp.get("credit_crisis", 0))
        poly_inflation = max(pp.get("inflation", 0), pp.get("usd_collapse", 0),
                             pp.get("yuan_rise", 0))
        poly_crypto = pp.get("crypto_crisis", 0)

        # Phase classification (original logic + crypto reinforcement + Polymarket)
        # Polymarket adjustments: lower detection thresholds when prediction
        # markets confirm crowd sees elevated risk.
        oil_spike_threshold = 0.03
        if poly_oil > 0.08:
            # Crowd gives >8% to oil-shock events — lower the bar
            oil_spike_threshold = 0.02

        if oil_pct_change > oil_spike_threshold and vix > 25:
            # Oil spiking with high vol — entry phase
            if self._oil_spike_start is None:
                self._oil_spike_start = datetime.now()
            return SeeSawPhase.OIL_SPIKE

        if self._oil_spike_start and gold_pct_change > 0.02:
            # Gold catching up after oil spike — rotation
            spike_age = datetime.now() - self._oil_spike_start
            if spike_age > timedelta(days=3):
                return SeeSawPhase.INFLATION_ROTATION

        if gold_pct_change > 0.03 and vix > 20:
            return SeeSawPhase.GOLD_BREAKOUT

        # Polymarket-driven GOLD_BREAKOUT: crowd sees >12% gold reprice
        # probability even without a large price move yet
        if poly_gold > 0.12 and gold_pct_change > 0.01 and vix > 18:
            return SeeSawPhase.GOLD_BREAKOUT

        gold_silver_ratio = gold / silver if silver > 0 else 80.0

        # SILVER_AMPLIFIER: original condition OR crypto-confirmed
        #   Original: G/S < 65 + VIX > 20
        #   Crypto boost: cross_amplifier_index > 5× AND BTC/Gold < 20
        #     means silver is crushing both gold AND crypto — supercycle
        silver_amplifier_classic = gold_silver_ratio < 65 and vix > 20
        silver_amplifier_crypto = (
            cross_amp > 5.0
            and btc_gold < 20.0
            and btc_gold > 0
        )
        if silver_amplifier_classic or silver_amplifier_crypto:
            return SeeSawPhase.SILVER_AMPLIFIER

        # Polymarket-driven INFLATION_ROTATION: crowd sees dollar collapse /
        # yuan rise / inflation without a clear oil trigger
        if poly_inflation > 0.10 and vix > 20:
            return SeeSawPhase.INFLATION_ROTATION

        # Extreme crypto fear (Fear&Greed < 15) with high VIX → treat as
        # stress even without oil spike — flight-to-safety amplifier
        if fear_greed < 15 and vix > 25 and active_severe >= 1:
            return SeeSawPhase.GOLD_BREAKOUT

        if vix < 20 and active_severe == 0:
            return SeeSawPhase.RECOVERY

        # Default: check if we have geopolitical stress
        if active_severe >= 2 or vix > 30:
            return SeeSawPhase.OIL_SPIKE  # Treat elevated stress as oil-spike readiness

        return SeeSawPhase.NEUTRAL

    def _generate_signals(
        self,
        phase: SeeSawPhase,
        prices: Dict[Asset, float],
        lunar_pos: Any,
        coherence_score: float,
        active_scenarios: List[Any],
        crypto_ctx: Optional[Dict[str, float]] = None,
        poly_ctx: Optional[Dict[str, Any]] = None,
    ) -> List[SeeSawSignal]:
        """Generate see-saw trading signals based on current phase, timing, crypto context, and Polymarket."""
        signals: List[SeeSawSignal] = []
        weights = DEFAULT_WEIGHTS.get(phase.value, DEFAULT_WEIGHTS["neutral"])

        # Phi-window multiplier: scale up during phi windows
        phi_mult = lunar_pos.position_multiplier if hasattr(lunar_pos, "position_multiplier") else 1.0
        moon_phase_str = lunar_pos.phase.value if hasattr(lunar_pos, "phase") else "waxing"
        moon_name = getattr(lunar_pos, "moon_name", "Unknown")

        # Base confidence from coherence + phase clarity
        base_conf = min(0.95, coherence_score * 0.6 + 0.3)

        # ── Cross-asset crypto context ──
        ctx = crypto_ctx or {}
        btc_gold = ctx.get("btc_gold_ratio", 0.0)
        fear_greed = ctx.get("fear_greed", 50.0)
        cross_amp = ctx.get("cross_amplifier_index", 1.0)
        dxy = ctx.get("dxy", 0.0)

        # DXY context string (added to reasons when relevant)
        dxy_note = ""
        if dxy > 110:
            dxy_note = f" [DXY={dxy:.0f} strong $ → headwind for gold/crypto]"
        elif dxy < 95:
            dxy_note = f" [DXY={dxy:.0f} weak $ → tailwind for gold/crypto]"

        # ── Polymarket prediction-market context ──
        pctx = poly_ctx or {}
        pp = pctx.get("phase_probs", {})
        poly_edge = pctx.get("avg_thesis_edge", 0.0)

        # Build Polymarket confirmation note for signal reasons
        poly_note = ""
        if pp:
            poly_oil = max(pp.get("oil_shock", 0), pp.get("iran_war", 0),
                           pp.get("israel_conflict", 0))
            poly_gold = pp.get("gold_reprice", 0)
            if phase == SeeSawPhase.OIL_SPIKE and poly_oil > 0.05:
                poly_note = f" [Poly: {poly_oil:.0%} oil-shock crowd prob]"
            elif phase == SeeSawPhase.GOLD_BREAKOUT and poly_gold > 0.05:
                poly_note = f" [Poly: {poly_gold:.0%} gold-reprice crowd prob]"
            elif phase == SeeSawPhase.INFLATION_ROTATION:
                poly_inf = max(pp.get("inflation", 0), pp.get("usd_collapse", 0))
                if poly_inf > 0.05:
                    poly_note = f" [Poly: {poly_inf:.0%} inflation/USD-collapse prob]"

        # Polymarket edge conviction: >3% avg edge → boost confidence 10%
        poly_conf_mult = 1.0
        if poly_edge > 0.03:
            poly_conf_mult = 1.10

        # Fear&Greed conviction modifier
        fg_mult = 1.0
        if fear_greed < 20:
            fg_mult = 1.15  # extreme fear = higher conviction for haven longs
        elif fear_greed > 75:
            fg_mult = 0.85  # greed = trim conviction

        # ── Phase-specific signal generation ──

        if phase == SeeSawPhase.OIL_SPIKE:
            signals.append(SeeSawSignal(
                asset=Asset.XLE, etf="XLE", direction="LONG",
                weight=weights.get("XLE", 0.20) * phi_mult,
                confidence=base_conf * 0.9,
                reason=f"Oil spike — energy sector direct beneficiary{dxy_note}",
                option_type="bull_call_spread",
                strike_hint="XLE Jun 2026 110/115 bull call spread",
                off_ramp=f"Moon 3 Blue Moon",
            ))
            signals.append(SeeSawSignal(
                asset=Asset.GOLD, etf="GLD", direction="LONG",
                weight=weights.get("GOLD", 0.25) * phi_mult,
                confidence=base_conf * 0.85 * fg_mult,
                reason=f"Oil spike → inflation hedge rotation (1-3 week lag){dxy_note}",
                option_type="bull_call_spread",
                strike_hint="GLD Jun 2026 420/430 bull call spread",
                off_ramp=f"Moon 4 Summer Solstice",
            ))
            signals.append(SeeSawSignal(
                asset=Asset.SILVER, etf="SLV", direction="LONG",
                weight=weights.get("SILVER", 0.30) * phi_mult,
                confidence=base_conf * 0.80 * fg_mult,
                reason=f"Silver high-beta amplifier — lags gold 1-3 weeks then 2-3× outperformance{dxy_note}",
                option_type="bull_call_spread",
                strike_hint="SLV Jun 2026 30/32 bull call spread",
                off_ramp=f"Moon 4 Summer Solstice",
            ))
            # Short the victims
            signals.append(SeeSawSignal(
                asset=Asset.QQQ, etf="QQQ", direction="SHORT",
                weight=0.10 * phi_mult,
                confidence=base_conf * 0.75,
                reason="Oil spike → growth fears → tech sells off",
                option_type="bear_put_spread",
                strike_hint="QQQ Jun 2026 410/400 bear put spread",
                off_ramp=f"Moon 3 Blue Moon",
            ))
            # ── Crypto cross-signal: oil spike + DXY stress ──
            if fear_greed < 25 and btc_gold > 0:
                signals.append(SeeSawSignal(
                    asset=Asset.BITO, etf="BITO", direction="SHORT",
                    weight=0.05 * phi_mult,
                    confidence=base_conf * 0.65,
                    reason=f"Oil spike + F&G={fear_greed:.0f} — crypto selling off with risk assets",
                    option_type="put",
                    strike_hint="BITO Jun 2026 ATM put",
                    off_ramp="Moon 3 Blue Moon",
                ))

        elif phase == SeeSawPhase.INFLATION_ROTATION:
            signals.append(SeeSawSignal(
                asset=Asset.GOLD, etf="GLD", direction="LONG",
                weight=weights.get("GOLD", 0.30) * phi_mult,
                confidence=base_conf * 0.92,
                reason="Inflation embedded — gold is the pure monetary hedge",
                option_type="bull_call_spread",
                strike_hint="GLD Jun 2026 430/440 bull call spread",
                off_ramp=f"Moon 5 G storm window",
            ))
            signals.append(SeeSawSignal(
                asset=Asset.SILVER, etf="SLV", direction="LONG",
                weight=weights.get("SILVER", 0.35) * phi_mult,
                confidence=base_conf * 0.90,
                reason="Silver double-whammy: monetary + industrial (solar, EVs, micro-reactors)",
                option_type="bull_call_spread",
                strike_hint="SLV Jun 2026 32/34 bull call spread",
                off_ramp=f"Moon 6 Fire Peak",
            ))
            signals.append(SeeSawSignal(
                asset=Asset.GDX, etf="GDX", direction="LONG",
                weight=weights.get("GDX", 0.15) * phi_mult,
                confidence=base_conf * 0.85,
                reason="Gold miners leverage gold price moves 2-3× with operating leverage",
                option_type="call",
                strike_hint="GDX Jun 2026 95 call",
                off_ramp=f"Moon 5 G storm window",
            ))

        elif phase == SeeSawPhase.GOLD_BREAKOUT:
            signals.append(SeeSawSignal(
                asset=Asset.GOLD, etf="GLD", direction="LONG",
                weight=0.30 * phi_mult,
                confidence=base_conf * 0.90,
                reason="Gold breakout confirmed — momentum + safe-haven convergence",
                option_type="bull_call_spread",
                strike_hint="GLD Jul 2026 440/450 bull call spread",
                off_ramp=f"Moon 6 Fire Peak",
            ))
            signals.append(SeeSawSignal(
                asset=Asset.SILVER, etf="SLV", direction="LONG",
                weight=0.35 * phi_mult,
                confidence=base_conf * 0.88,
                reason="Silver catching fire — historical pattern: lags gold then outperforms 30-100%",
                option_type="bull_call_spread",
                strike_hint="SLV Jul 2026 34/36 bull call spread",
                off_ramp=f"Moon 6 Fire Peak",
            ))

        elif phase == SeeSawPhase.SILVER_AMPLIFIER:
            signals.append(SeeSawSignal(
                asset=Asset.SILVER, etf="SLV", direction="LONG",
                weight=0.40 * phi_mult,
                confidence=base_conf * 0.92 * fg_mult,
                reason=f"Silver amplifier phase — gold/silver ratio compressing, max beta{dxy_note}",
                option_type="bull_call_spread",
                strike_hint="SLV Jul 2026 36/40 bull call spread",
                off_ramp=f"Moon 7 Autumnal Equinox",
            ))
            # Trim gold, keep silver
            signals.append(SeeSawSignal(
                asset=Asset.GOLD, etf="GLD", direction="LONG",
                weight=0.20 * phi_mult,
                confidence=base_conf * 0.80 * fg_mult,
                reason=f"Reduce gold weight — silver now the better risk/reward{dxy_note}",
                option_type="call",
                strike_hint="GLD Jul 2026 450 call",
                off_ramp=f"Moon 7 Autumnal Equinox",
            ))
            # ── Crypto cross-signals for silver amplifier ──
            if btc_gold > 0 and btc_gold < 20:
                # BTC/Gold < 20 means gold is crushing BTC → physical metals dominate
                signals.append(SeeSawSignal(
                    asset=Asset.BITO, etf="BITO", direction="SHORT",
                    weight=0.08 * phi_mult,
                    confidence=base_conf * 0.70 * fg_mult,
                    reason=f"BTC/Gold={btc_gold:.1f} (<20) — gold crushing crypto, short BITO",
                    option_type="put",
                    strike_hint="BITO Jul 2026 ATM put",
                    off_ramp="Moon 6 Fire Peak",
                ))
            if cross_amp > 5.0:
                # Cross amplifier confirms silver supercycle
                signals.append(SeeSawSignal(
                    asset=Asset.GDX, etf="GDX", direction="LONG",
                    weight=0.10 * phi_mult,
                    confidence=base_conf * 0.78 * fg_mult,
                    reason=f"Cross amplifier {cross_amp:.1f}× — commodity supercycle, add miners",
                    option_type="call",
                    strike_hint="GDX Jul 2026 95 call",
                    off_ramp="Moon 7 Autumnal Equinox",
                ))

        elif phase == SeeSawPhase.RECOVERY:
            signals.append(SeeSawSignal(
                asset=Asset.SILVER, etf="SLV", direction="LONG",
                weight=0.20 * phi_mult,
                confidence=base_conf * 0.70,
                reason="Recovery — silver industrial demand rebounds with green-energy cycle",
                option_type="call",
                strike_hint="SLV Sep 2026 40 call",
                off_ramp=f"Moon 9 Solar",
            ))
            # Re-enter equities cautiously
            signals.append(SeeSawSignal(
                asset=Asset.QQQ, etf="QQQ", direction="LONG",
                weight=0.15 * phi_mult,
                confidence=base_conf * 0.60,
                reason="Stocks bottoming — cautious re-entry",
                option_type="call",
                strike_hint="QQQ Sep 2026 420 call",
                off_ramp=f"Moon 10 Planetary",
            ))
            # ── Crypto recovery signal ──
            if fear_greed < 20:
                signals.append(SeeSawSignal(
                    asset=Asset.BITO, etf="BITO", direction="LONG",
                    weight=0.08 * phi_mult,
                    confidence=base_conf * 0.60 * fg_mult,
                    reason=f"Fear&Greed={fear_greed:.0f} extreme fear + recovery → crypto bounce play",
                    option_type="call",
                    strike_hint="BITO Sep 2026 ATM call",
                    off_ramp="Moon 10 Planetary",
                ))

        else:
            # NEUTRAL — hold cash, light positions
            signals.append(SeeSawSignal(
                asset=Asset.GOLD, etf="GLD", direction="LONG",
                weight=0.15,
                confidence=base_conf * 0.50,
                reason="Neutral — base gold allocation as portfolio insurance",
                option_type="call",
                strike_hint="GLD Dec 2026 450 call",
                off_ramp="Next phi window",
            ))

        # Adjust all signals by moon phase
        if moon_phase_str in ("new",):
            # NEW moon = reassess, don't add new positions
            for s in signals:
                s.confidence *= 0.5
                s.reason += " [NEW MOON — reduced conviction, reassess]"
        elif moon_phase_str in ("waning",):
            # WANING = protect, trim
            for s in signals:
                if s.direction == "LONG":
                    s.weight *= 0.7
                    s.reason += " [WANING — trimmed weight, protect gains]"

        # ── Apply Polymarket confirmation to all signals ──
        if poly_note or poly_conf_mult != 1.0:
            for s in signals:
                s.confidence = min(0.99, s.confidence * poly_conf_mult)
                if poly_note:
                    s.reason += poly_note

        return signals

    # ═══════════════════════════════════════════════════════════════════════
    # STOP-LOSS MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════

    def _update_positions(self, prices: Dict[Asset, float]) -> List[str]:
        """Update all tracked positions with new prices and check stops."""
        triggered: List[str] = []

        for asset, pos in list(self.positions.items()):
            price = prices.get(asset, pos.current_price)
            pos.update_price(price)

            if pos.check_stop_loss():
                triggered.append(
                    f"{pos.etf}: {pos.stop_trigger} "
                    f"(entry=${pos.entry_price:.2f}, current=${pos.current_price:.2f}, "
                    f"P&L={pos.pnl_pct:+.1%})"
                )
                # Return capital to cash
                self.cash += pos.quantity * (1 + pos.pnl_pct)
                logger.warning("STOP-LOSS: %s %s", pos.etf, pos.stop_trigger)

        # Remove stopped positions
        self.positions = {
            a: p for a, p in self.positions.items() if not p.stopped_out
        }

        return triggered

    def _apply_signals(self, signals: List[SeeSawSignal],
                       prices: Dict[Asset, float]) -> None:
        """
        Apply signal weights to portfolio — rebalance positions.
        Only enters positions when confidence > 0.5.
        """
        # Calculate total invested
        invested = sum(p.quantity for p in self.positions.values())
        self.portfolio_value = self.cash + invested

        for sig in signals:
            if sig.confidence < 0.50:
                continue
            if sig.direction == "CLOSE":
                # Close the position
                if sig.asset in self.positions:
                    pos = self.positions.pop(sig.asset)
                    self.cash += pos.quantity * (1 + pos.pnl_pct)
                continue

            target_notional = self.portfolio_value * sig.weight
            price = prices.get(sig.asset, 0)
            if price <= 0:
                continue

            if sig.asset in self.positions:
                # Adjust existing position toward target
                pos = self.positions[sig.asset]
                diff = target_notional - pos.quantity
                if abs(diff) > self.portfolio_value * 0.02:  # >2% change threshold
                    if diff > 0 and self.cash >= diff:
                        pos.quantity += diff
                        self.cash -= diff
                    elif diff < 0:
                        pos.quantity += diff  # reduce
                        self.cash -= diff     # add back to cash
            elif sig.direction == "LONG" and self.cash >= target_notional:
                # Open new position
                self.positions[sig.asset] = TrackedPosition(
                    asset=sig.asset,
                    etf=sig.etf,
                    entry_price=price,
                    current_price=price,
                    quantity=target_notional,
                    entry_time=datetime.now(),
                    high_water_mark=price,
                    fixed_stop_pct=self.fixed_stop_pct,
                    trailing_stop_pct=self.trailing_stop_pct,
                )
                self.cash -= target_notional

        # Recalculate
        invested = sum(p.quantity for p in self.positions.values())
        self.portfolio_value = self.cash + invested

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN HOURLY CYCLE
    # ═══════════════════════════════════════════════════════════════════════

    async def run_hourly(self) -> CycleReport:
        """Run a single hourly scrape → analyze → signal → stop-loss cycle."""
        t0 = time.perf_counter()

        # 1. SCRAPE
        try:
            data = await self._scrape()
            snap = data["snapshot"]
            lunar = data["lunar"]
            coherence = data["coherence"]
            active_scenarios = data["active_scenarios"]
            heatmap = data["heatmap"]
        except Exception as e:
            logger.error("Scrape failed, using defaults: %s", e)
            # Graceful degradation — use defaults
            snap = None
            lunar = None
            coherence = None
            active_scenarios = []
            heatmap = []

        # Extract prices
        if snap:
            prices = snap.prices
            vix = snap.vix
            sources_ok = snap.sources_ok
            sources_failed = snap.sources_failed
        else:
            prices = dict(DEFAULT_PRICES)
            vix = 25.0
            sources_ok = []
            sources_failed = ["all_sources"]

        gold_price = prices.get(Asset.GOLD, DEFAULT_PRICES[Asset.GOLD])
        silver_price = prices.get(Asset.SILVER, DEFAULT_PRICES[Asset.SILVER])
        oil_price = prices.get(Asset.OIL, DEFAULT_PRICES[Asset.OIL])

        # ── Build cross-asset crypto context from existing snapshot ──
        btc_price = prices.get(Asset.BTC, DEFAULT_PRICES.get(Asset.BTC, 68000.0))
        eth_price = prices.get(Asset.ETH, DEFAULT_PRICES.get(Asset.ETH, 3800.0))
        xrp_price = prices.get(Asset.XRP, DEFAULT_PRICES.get(Asset.XRP, 2.50))
        fg_value = float(snap.fear_greed) if snap else 50.0
        dxy_value = float(snap.macro.get("dollar_index", 0)) if snap else 0.0

        # Compute crypto × commodity ratios
        btc_gold_ratio = btc_price / gold_price if gold_price > 0 else 0.0
        eth_btc_ratio = eth_price / btc_price if btc_price > 0 else 0.0
        btc_silver_ratio = btc_price / silver_price if silver_price > 0 else 0.0
        # Cross amplifier: (silver/gold) × (BTC/ETH) normalised to median
        gold_silver_raw = gold_price / silver_price if silver_price > 0 else 80.0
        btc_eth_raw = btc_price / eth_price if eth_price > 0 else 18.0
        cross_amplifier_index = (1.0 / gold_silver_raw) * btc_eth_raw if gold_silver_raw > 0 else 1.0

        crypto_ctx: Dict[str, float] = {
            "btc_gold_ratio": btc_gold_ratio,
            "eth_btc_ratio": eth_btc_ratio,
            "btc_silver_ratio": btc_silver_ratio,
            "cross_amplifier_index": cross_amplifier_index,
            "fear_greed": fg_value,
            "dxy": dxy_value,
        }

        # ── Polymarket prediction-market scrape ──
        poly_ctx: Dict[str, Any] = {}
        try:
            poly_ctx = await self._scrape_polymarket()
            if poly_ctx:
                sources_ok.append("polymarket")
            else:
                sources_failed.append("polymarket")
        except Exception as e:
            logger.warning("Polymarket scrape failed (continuing): %s", e)
            sources_failed.append("polymarket")

        # 2. ANALYZE — detect see-saw phase
        phase = self._detect_phase(prices, vix, heatmap, crypto_ctx, poly_ctx)
        self.current_phase = phase

        # 3. GENERATE SIGNALS
        coherence_score = coherence.overall_score if coherence else 0.5
        signals = self._generate_signals(
            phase, prices, lunar, coherence_score, active_scenarios,
            crypto_ctx, poly_ctx,
        )

        # 4. UPDATE POSITIONS & CHECK STOPS
        stops_triggered = self._update_positions(prices)

        # 5. APPLY SIGNALS (rebalance)
        self._apply_signals(signals, prices)

        # Update price memory
        self._prev_oil_price = oil_price
        self._prev_gold_price = gold_price

        # Build report
        gold_oil_ratio = gold_price / oil_price if oil_price > 0 else 0
        gold_silver_ratio = gold_price / silver_price if silver_price > 0 else 0

        active_codes = [s.code if hasattr(s, "code") else str(s) for s in active_scenarios]
        top_firing = active_codes[0] if active_codes else "NONE"

        report = CycleReport(
            timestamp=datetime.now(),
            phase=phase,
            vix=vix,
            regime=snap.regime if snap else VolRegime.CRISIS,
            gold_price=gold_price,
            silver_price=silver_price,
            oil_price=oil_price,
            gold_oil_ratio=gold_oil_ratio,
            gold_silver_ratio=gold_silver_ratio,
            btc_gold_ratio=btc_gold_ratio,
            eth_btc_ratio=eth_btc_ratio,
            btc_silver_ratio=btc_silver_ratio,
            cross_amplifier_index=cross_amplifier_index,
            fear_greed=fg_value,
            dxy=dxy_value,
            btc_price=btc_price,
            eth_price=eth_price,
            # Polymarket fields
            poly_oil_shock_prob=max(
                poly_ctx.get("phase_probs", {}).get("oil_shock", 0),
                poly_ctx.get("phase_probs", {}).get("iran_war", 0),
            ) if poly_ctx else 0.0,
            poly_gold_reprice_prob=poly_ctx.get("phase_probs", {}).get("gold_reprice", 0) if poly_ctx else 0.0,
            poly_recession_prob=max(
                poly_ctx.get("phase_probs", {}).get("recession", 0),
                poly_ctx.get("phase_probs", {}).get("credit_crisis", 0),
            ) if poly_ctx else 0.0,
            poly_geopolitical_prob=poly_ctx.get("phase_probs", {}).get("geopolitical", 0) if poly_ctx else 0.0,
            poly_thesis_edge=poly_ctx.get("avg_thesis_edge", 0.0) if poly_ctx else 0.0,
            poly_top_opportunities=[
                f"[{o.category}] {o.outcome}@{o.market_price:.0%} edge={o.edge:.0%}: {o.market_question[:60]}"
                for o in poly_ctx.get("top_opportunities", [])[:3]
            ] if poly_ctx else [],
            active_scenarios=active_codes,
            top_firing_scenario=top_firing,
            moon_number=getattr(lunar, "moon_number", 0),
            moon_name=getattr(lunar, "moon_name", "Unknown"),
            moon_phase=lunar.phase.value if lunar and hasattr(lunar, "phase") else "unknown",
            phi_window=getattr(lunar, "in_phi_window", False),
            phi_coherence=getattr(lunar, "phi_coherence", 0.0),
            position_multiplier=getattr(lunar, "position_multiplier", 1.0),
            signals=signals,
            positions={
                a.value: {
                    "etf": p.etf,
                    "entry": p.entry_price,
                    "current": p.current_price,
                    "qty": p.quantity,
                    "pnl_pct": p.pnl_pct,
                    "hwm": p.high_water_mark,
                }
                for a, p in self.positions.items()
            },
            stops_triggered=stops_triggered,
            portfolio_value=self.portfolio_value,
            cash_pct=self.cash / self.portfolio_value if self.portfolio_value > 0 else 1.0,
            sources_ok=sources_ok,
            sources_failed=sources_failed,
        )

        # Persist
        self._save_report(report)
        self.cycle_history.append(report)
        if len(self.cycle_history) > self._max_history:
            self.cycle_history = self.cycle_history[-self._max_history:]

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Cycle complete in %.0fms: phase=%s, gold=$%.0f, oil=$%.0f, "
            "silver=$%.0f, G/O=%.1f, BTC/Gold=%.1f, F&G=%.0f, "
            "portfolio=$%.0f, %d signals, %d stops",
            elapsed, phase.value, gold_price, oil_price, silver_price,
            gold_oil_ratio, btc_gold_ratio, fg_value,
            self.portfolio_value, len(signals), len(stops_triggered),
        )

        return report

    # ═══════════════════════════════════════════════════════════════════════
    # RUN FOREVER — Hourly loop
    # ═══════════════════════════════════════════════════════════════════════

    async def run_forever(self) -> None:
        """Run the capital engine in a continuous hourly loop."""
        logger.info(
            "LifeboatCapitalEngine starting hourly loop (interval=%ds)",
            self.cycle_interval,
        )
        while not self._shutdown:
            try:
                report = await self.run_hourly()
                self._print_cycle_summary(report)
            except Exception as e:
                logger.error("Hourly cycle failed: %s", e, exc_info=True)

            # Sleep until next cycle (interruptible)
            try:
                await asyncio.wait_for(
                    self._shutdown_event_wait(),
                    timeout=self.cycle_interval,
                )
                break  # shutdown was requested
            except asyncio.TimeoutError:
                pass  # normal — time for next cycle

    async def _shutdown_event_wait(self) -> None:
        """Wait until shutdown is requested."""
        while not self._shutdown:
            await asyncio.sleep(1)

    def shutdown(self) -> None:
        """Signal the engine to stop and clean up resources."""
        logger.info("LifeboatCapitalEngine shutdown requested")
        self._shutdown = True
        # Close Polymarket scanner session if open
        if self._polymarket_scanner is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._polymarket_scanner.close())
                else:
                    loop.run_until_complete(self._polymarket_scanner.close())
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════════
    # PERSISTENCE & REPORTING
    # ═══════════════════════════════════════════════════════════════════════

    def _save_report(self, report: CycleReport) -> None:
        """Append report as JSON line."""
        try:
            record = {
                "timestamp": report.timestamp.isoformat(),
                "phase": report.phase.value,
                "vix": report.vix,
                "regime": report.regime.value,
                "gold": report.gold_price,
                "silver": report.silver_price,
                "oil": report.oil_price,
                "gold_oil_ratio": round(report.gold_oil_ratio, 2),
                "gold_silver_ratio": round(report.gold_silver_ratio, 2),
                "btc_gold_ratio": round(report.btc_gold_ratio, 2),
                "eth_btc_ratio": round(report.eth_btc_ratio, 5),
                "btc_silver_ratio": round(report.btc_silver_ratio, 1),
                "cross_amplifier": round(report.cross_amplifier_index, 2),
                "fear_greed": round(report.fear_greed, 0),
                "dxy": round(report.dxy, 1),
                "btc": round(report.btc_price, 0),
                "eth": round(report.eth_price, 0),
                "poly_oil_shock": round(report.poly_oil_shock_prob, 4),
                "poly_gold_reprice": round(report.poly_gold_reprice_prob, 4),
                "poly_recession": round(report.poly_recession_prob, 4),
                "poly_geopolitical": round(report.poly_geopolitical_prob, 4),
                "poly_thesis_edge": round(report.poly_thesis_edge, 4),
                "poly_top": report.poly_top_opportunities[:3],
                "active_scenarios": report.active_scenarios[:5],
                "moon": report.moon_number,
                "moon_phase": report.moon_phase,
                "phi_window": report.phi_window,
                "signals": [
                    {
                        "asset": s.asset.value,
                        "dir": s.direction,
                        "weight": round(s.weight, 3),
                        "conf": round(s.confidence, 2),
                        "reason": s.reason[:80],
                    }
                    for s in report.signals
                ],
                "stops": report.stops_triggered,
                "portfolio": round(report.portfolio_value, 2),
                "cash_pct": round(report.cash_pct, 3),
                "positions": {k: round(v.get("pnl_pct", 0), 4) for k, v in report.positions.items()},
            }
            with open(self._report_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug("Failed to save report: %s", e)

    def _save_state(self) -> None:
        """Save current engine state for recovery."""
        try:
            state = {
                "portfolio_value": self.portfolio_value,
                "cash": self.cash,
                "phase": self.current_phase.value,
                "positions": {
                    a.value: {
                        "etf": p.etf,
                        "entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "quantity": p.quantity,
                        "high_water_mark": p.high_water_mark,
                        "entry_time": p.entry_time.isoformat(),
                    }
                    for a, p in self.positions.items()
                },
                "saved_at": datetime.now().isoformat(),
            }
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(self._state_path)
        except Exception as e:
            logger.debug("Failed to save state: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Return current engine status (for health endpoint integration)."""
        return {
            "engine": "LifeboatCapitalEngine",
            "phase": self.current_phase.value,
            "portfolio_value": round(self.portfolio_value, 2),
            "cash_pct": round(self.cash / self.portfolio_value, 3) if self.portfolio_value else 1.0,
            "positions": {
                a.value: {
                    "etf": p.etf,
                    "pnl_pct": round(p.pnl_pct, 4),
                    "qty": round(p.quantity, 2),
                }
                for a, p in self.positions.items()
            },
            "cycles_run": len(self.cycle_history),
            "last_cycle": self.cycle_history[-1].timestamp.isoformat() if self.cycle_history else None,
        }

    def _print_cycle_summary(self, report: CycleReport) -> None:
        """Print a human-readable cycle summary to stdout."""
        print()
        print("=" * 78)
        print(f"  LIFEBOAT CAPITAL ENGINE — {report.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print("=" * 78)
        print(f"  Phase:     {report.phase.value.upper():30s}  Regime: {report.regime.value.upper()}")
        print(f"  Gold:      ${report.gold_price:>10,.0f}  Oil:    ${report.oil_price:>10,.2f}  "
              f"Silver: ${report.silver_price:>10,.2f}")
        print(f"  G/O Ratio: {report.gold_oil_ratio:>10.1f}  G/S Ratio: {report.gold_silver_ratio:>8.1f}  "
              f"VIX: {report.vix:.1f}")
        print(f"  BTC:       ${report.btc_price:>10,.0f}  ETH:    ${report.eth_price:>10,.0f}")
        print(f"  BTC/Gold:  {report.btc_gold_ratio:>10.1f}  ETH/BTC: {report.eth_btc_ratio:>8.4f}  "
              f"BTC/SLV: {report.btc_silver_ratio:.0f}")
        print(f"  CrossAmp:  {report.cross_amplifier_index:>10.2f}  F&G:    {report.fear_greed:>8.0f}  "
              f"DXY: {report.dxy:.1f}")
        print(f"  Moon:      {report.moon_number}/13 {report.moon_name} ({report.moon_phase})"
              f"{'  [PHI WINDOW]' if report.phi_window else ''}")
        print(f"  Coherence: {report.phi_coherence:.4f}  Sizing: {report.position_multiplier:.2f}x")
        # ── Polymarket prediction market overlay ──
        has_poly = (report.poly_oil_shock_prob > 0 or report.poly_gold_reprice_prob > 0
                    or report.poly_recession_prob > 0 or report.poly_thesis_edge > 0)
        if has_poly:
            print(f"  Poly Oil:  {report.poly_oil_shock_prob:>9.1%}  "
                  f"Gold: {report.poly_gold_reprice_prob:>8.1%}  "
                  f"Recess: {report.poly_recession_prob:.1%}  "
                  f"Geo: {report.poly_geopolitical_prob:.1%}")
            print(f"  Poly Edge: {report.poly_thesis_edge:>9.1%}")
            if report.poly_top_opportunities:
                print(f"  Top Opps:")
                for opp_str in report.poly_top_opportunities[:3]:
                    print(f"    {opp_str[:72]}")
        else:
            print(f"  Polymarket: no data (scanner returned empty)")
        if report.active_scenarios:
            print(f"  Scenarios: {', '.join(report.active_scenarios[:5])}")

        if report.signals:
            print()
            print(f"  {'Signal':>10}  {'Dir':>5}  {'Weight':>7}  {'Conf':>5}  {'Reason'}")
            print("  " + "-" * 70)
            for s in report.signals:
                print(f"  {s.etf:>10}  {s.direction:>5}  {s.weight:>6.0%}  "
                      f"{s.confidence:>4.0%}  {s.reason[:50]}")
                if s.strike_hint:
                    print(f"{'':>26} → {s.strike_hint}  (off-ramp: {s.off_ramp})")

        if report.stops_triggered:
            print()
            for stop in report.stops_triggered:
                print(f"  STOP-LOSS: {stop}")

        print()
        invested = sum(v.get("qty", 0) for v in report.positions.values())
        print(f"  Portfolio: ${report.portfolio_value:>12,.2f} CAD  "
              f"(Cash: {report.cash_pct:.0%}  Invested: ${invested:,.2f})")
        if report.positions:
            for asset_name, pos in report.positions.items():
                print(f"    {pos.get('etf', asset_name):>6}: "
                      f"${pos.get('qty', 0):>10,.2f}  "
                      f"P&L: {pos.get('pnl_pct', 0):>+6.1%}  "
                      f"(entry=${pos.get('entry', 0):.2f} → ${pos.get('current', 0):.2f})")

        print(f"\n  Data: {len(report.sources_ok)} OK"
              + (f", {len(report.sources_failed)} FAILED" if report.sources_failed else ""))
        print("=" * 78)


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the capital engine from command line."""
    import argparse

    # Windows cp1252 stdout fix
    if hasattr(sys.stdout, "buffer"):
        encoding = getattr(sys.stdout, "encoding", "") or ""
        if encoding.lower() != "utf-8":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Lifeboat Capital Engine — Gold-Oil-Silver See-Saw")
    parser.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    parser.add_argument("--interval", type=int, default=3600, help="Cycle interval in seconds (default: 3600)")
    parser.add_argument("--capital", type=float, default=48100.0, help="Starting capital in CAD")
    parser.add_argument("--fixed-stop", type=float, default=8.0, help="Fixed stop-loss %% (default: 8)")
    parser.add_argument("--trailing-stop", type=float, default=5.0, help="Trailing stop-loss %% (default: 5)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine = LifeboatCapitalEngine(
        starting_capital=args.capital,
        fixed_stop_pct=args.fixed_stop / 100.0,
        trailing_stop_pct=args.trailing_stop / 100.0,
        cycle_interval_seconds=args.interval,
    )

    if args.once:
        async def _run_once():
            report = await engine.run_hourly()
            engine._print_cycle_summary(report)
            engine.shutdown()
        asyncio.run(_run_once())
    else:
        try:
            asyncio.run(engine.run_forever())
        except KeyboardInterrupt:
            engine.shutdown()
            print("\nEngine stopped.")


if __name__ == "__main__":
    main()
