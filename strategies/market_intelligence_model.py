"""
Market Intelligence Model — AAC 24/7 Evolving Sentiment Engine
==============================================================
Answers: WHAT is happening? HOW STRONG is the signal? WHEN do I get in/out? HOW MUCH?

Architecture:
    Events  (UW flow, FRED, news, NCL signals)
        ↓
    EventIngestor  — normalises to MarketEvent(weight, decay)
        ↓
    SentimentEngine — applies events to per-sector scores, decays over time
        ↓
    RegimeEngine  — macro formula layer (F1-F9)
        ↓
    PositionAdvisor — regime × sentiment → PositionRecommendation(entry + exit params)
        ↓
    NCLBridge  — push intelligence to NCL, pull NCL signals back in

Timeline cadences:
    TICK      every 5 s  — vol/spread check, heartbeat update
    INTRADAY  every 1 h  — full event ingest (UW flow, dark pool, headlines)
    DAILY     every 24 h — FRED data, regime re-evaluate, recommendation refresh
    WEEKLY    every 7 d  — macro thesis update, deep NCL sync
    CYCLE     every 30 d — FFD halving phase, long-range positioning review

Standalone usage:
    model = MarketIntelligenceModel()
    asyncio.run(model.run())

One-shot query:
    recs = model.get_recommendations()        # List[PositionRecommendation]
    state = model.get_sentiment_state()       # SentimentState as dict
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# TIMELINE CADENCES
# ═══════════════════════════════════════════════════════════════════════════

class TimeHorizon(Enum):
    """Evaluation cadence layers — from tick-level to macro cycle."""
    TICK      = 5          # seconds — real-time vol/spread heartbeat
    INTRADAY  = 3_600      # 1 hour  — UW flow ingest, sentiment pulse
    DAILY     = 86_400     # 24 hr   — FRED data, regime re-eval, recommendation refresh
    WEEKLY    = 604_800    # 7 days  — macro thesis update, NCL deep sync
    CYCLE     = 2_592_000  # 30 days — FFD halving phase, long-range review


# ═══════════════════════════════════════════════════════════════════════════
# EVENT MODEL
# ═══════════════════════════════════════════════════════════════════════════

class EventSource(Enum):
    # ── Unusual Whales (live options flow intelligence) ────────────────
    UNUSUAL_WHALES_FLOW     = "uw_flow"
    UNUSUAL_WHALES_DARKPOOL = "uw_darkpool"
    UNUSUAL_WHALES_CONGRESS = "uw_congress"
    # ── CBOE / Derivatives intelligence ───────────────────────────────
    CBOE_PUT_CALL           = "cboe_put_call"   # CBOE equity put/call ratio (free)
    CBOE_GEX                = "cboe_gex"        # Gamma exposure (derived)
    # ── Sentiment indices ─────────────────────────────────────────────
    FEAR_GREED_CRYPTO       = "fg_crypto"       # Alternative.me (free, no key)
    FEAR_GREED_STOCK        = "fg_stock"        # CNN/Alt.me stock FGI (free)
    # ── Social / Retail intelligence ──────────────────────────────────
    STOCKTWITS_SENTIMENT    = "stocktwits"      # Public API, no key
    REDDIT_SENTIMENT        = "reddit"          # Tradestie free API
    GOOGLE_TRENDS           = "google_trends"   # pytrends, no key
    # ── Finnhub (live key) ─────────────────────────────────────────────
    FINNHUB_INSIDER         = "finnhub_insider" # Cluster insider selling
    FINNHUB_ANALYST         = "finnhub_analyst" # Recommendation trend shifts
    FINNHUB_CALENDAR        = "finnhub_calendar"# FOMC/CPI/NFP pre-event
    FINNHUB_NEWS            = "finnhub_news"    # Sentiment-scored company news
    # ── Polygon.io (live key) ──────────────────────────────────────────
    POLYGON_BREADTH         = "polygon_breadth" # % tickers advancing/declining
    # ── Crypto on-chain / derivatives ────────────────────────────────
    COINGLASS               = "coinglass"       # Funding rates, liquidations, OI (free)
    CRYPTO_ONCHAIN          = "crypto_onchain"  # Exchange inflows, MVRV (CoinGecko)
    WHALE_ALERT             = "whale_alert"     # Large on-chain wallet movements
    # ── Macro / FRED ──────────────────────────────────────────────────
    NEWS_HEADLINE           = "news"
    FRED_DATA               = "fred"
    # ── System ────────────────────────────────────────────────────────
    IBKR_ORDER              = "ibkr_order"
    REGIME_SHIFT            = "regime_shift"
    NCL_SIGNAL              = "ncl"
    MANUAL                  = "manual"


@dataclass
class MarketEvent:
    """A single signal contributing to sector sentiment."""
    source: EventSource
    sector: str               # matches SECTOR_KEYS below
    event_type: str           # e.g. 'bearish_flow', 'macro_shock', 'data_release'
    weight: float             # -1.0 (max bear) to +1.0 (max bull)
    decay_halflife_hours: float
    description: str
    raw: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ticker: Optional[str] = None


# All sector keys used throughout the model
SECTOR_KEYS = [
    "credit", "banks", "shipping", "airlines", "private_credit",
    "tech", "consumer", "energy", "insurance", "index", "crypto",
]

# Primary ticker per sector (for event ↔ sector lookup)
SECTOR_TICKER_MAP: Dict[str, List[str]] = {
    "credit":         ["HYG", "JNK", "BKLN", "LQD"],
    "banks":          ["KRE", "IAT", "XLF"],
    "shipping":       ["ZIM", "GNK", "GOGL", "DAC", "MATX"],
    "airlines":       ["JETS", "DAL", "AAL", "UAL"],
    "private_credit": ["ARCC", "OBDC", "OWL", "MAIN", "BXSL"],
    "tech":           ["QQQ", "SMH", "NVDA", "AAPL"],
    "consumer":       ["XLY", "XRT", "AMZN"],
    "energy":         ["XLE", "CVX", "XOM", "OIH"],
    "insurance":      ["KIE", "AIG", "ALL", "PRU"],
    "index":          ["SPY", "IWM", "DIA"],
    "crypto":         ["BTC", "ETH", "XRP", "SOL", "ADA"],
}

# Reverse lookup: ticker → sector
TICKER_TO_SECTOR: Dict[str, str] = {}
for _sector, _tickers in SECTOR_TICKER_MAP.items():
    for _t in _tickers:
        TICKER_TO_SECTOR[_t] = _sector


def ticker_to_sector(ticker: str) -> str:
    """Map a ticker symbol to its sector key. Returns 'index' as fallback."""
    return TICKER_TO_SECTOR.get(ticker.upper(), "index")


# ═══════════════════════════════════════════════════════════════════════════
# SENTIMENT STATE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SectorSentiment:
    """Live sentiment for a single sector — score decays toward 0 over time."""
    sector: str
    score: float = 0.0                   # -100 (max bear) to +100 (max bull)
    event_count: int = 0                 # total events ever processed
    events: List[MarketEvent] = field(default_factory=list)  # rolling last-20
    trend: float = 0.0                   # score change vs previous snapshot
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sector": self.sector,
            "score": round(self.score, 2),
            "trend": round(self.trend, 2),
            "event_count": self.event_count,
            "last_updated": self.last_updated.isoformat(),
            "recent_events": [
                {"source": e.source.value, "weight": e.weight,
                 "desc": e.description, "ts": e.timestamp.isoformat()}
                for e in self.events[-5:]
            ],
        }


@dataclass
class SentimentState:
    """Full market sentiment state — persisted, updated every intraday cycle."""
    sectors: Dict[str, SectorSentiment] = field(default_factory=dict)
    global_score: float = 0.0            # weighted average across sectors
    regime_overlay: str = "uncertain"    # last known primary regime
    vol_shock_readiness: float = 0.0
    dominant_theme: str = "neutral"      # computed narrative
    event_count_24h: int = 0
    version: int = 0                     # increments on every state change
    last_full_refresh: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_ncl_sync: Optional[datetime] = None

    def get_sector(self, key: str) -> SectorSentiment:
        if key not in self.sectors:
            self.sectors[key] = SectorSentiment(sector=key)
        return self.sectors[key]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_score": round(self.global_score, 2),
            "regime_overlay": self.regime_overlay,
            "vol_shock_readiness": self.vol_shock_readiness,
            "dominant_theme": self.dominant_theme,
            "event_count_24h": self.event_count_24h,
            "version": self.version,
            "last_full_refresh": self.last_full_refresh.isoformat(),
            "last_ncl_sync": self.last_ncl_sync.isoformat() if self.last_ncl_sync else None,
            "sectors": {k: v.to_dict() for k, v in self.sectors.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════
# POSITION RECOMMENDATION (entry + exit fully parameterised)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PositionRecommendation:
    """
    A complete, actionable position recommendation.
    Tells you not just WHAT, but WHEN/HOW MUCH to enter and WHEN/HOW MUCH to exit.
    """
    ticker: str
    sector: str
    direction: str           # 'bearish' | 'bullish'
    expression: str          # 'put_spread' | 'atm_put' | 'otm_put' | 'vix_call_spread'

    # ── ENTRY parameters ─────────────────────────────────────────────────
    entry_urgency: str       # 'IMMEDIATE' | 'NEXT_OPEN' | 'SCALE_IN' | 'HOLD'
    entry_size_pct: float    # fraction of available capital (0.02 – 0.15)
    entry_max_usd: float     # hard cap in USD regardless of capital
    entry_conditions: List[str]   # human-readable triggers that fired
    structure_hint: str      # e.g. "ATM put, 14-42 DTE, spread preferred"
    expiry_min_dte: int
    expiry_max_dte: int
    otm_pct: float           # 0 = ATM, 0.05 = 5% OTM

    # ── EXIT parameters ──────────────────────────────────────────────────
    target_gain_pct: float   # close when P&L reaches this (e.g. 0.80 = +80%)
    stop_loss_pct: float     # close when loss reaches this (e.g. 0.50 = -50%)
    time_stop_days: int      # close if no movement after N days
    exit_on_regime: List[str]         # close if regime shifts to any of these
    exit_on_sentiment_reversal: float  # close if sector_score gains this many points
    exit_scaling: str        # 'FULL' | 'HALF_AT_TARGET' | 'SCALE_OUT_THIRDS'

    # ── Scoring & context ────────────────────────────────────────────────
    composite_score: float
    conviction_tier: str     # 'HIGH' | 'MEDIUM' | 'LOW'
    thesis: str
    formulas_fired: List[str]
    regime: str
    regime_confidence: float
    sector_sentiment: float
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["generated_at"] = self.generated_at.isoformat()
        d["exit_on_regime"] = self.exit_on_regime
        d["entry_conditions"] = self.entry_conditions
        d["formulas_fired"] = self.formulas_fired
        return d

    def print_rec(self) -> str:
        lines = [
            f"{'═'*60}",
            f"  {self.ticker} | {self.expression.upper().replace('_',' ')} | {self.conviction_tier} conviction",
            f"  Regime: {self.regime.upper()} ({self.regime_confidence:.0%})  |  Sector sentiment: {self.sector_sentiment:+.0f}/100",
            f"{'─'*60}",
            f"  ENTRY",
            f"    Urgency : {self.entry_urgency}",
            f"    Size    : {self.entry_size_pct:.0%} of capital (max ${self.entry_max_usd:,.0f})",
            f"    DTE     : {self.expiry_min_dte}–{self.expiry_max_dte} days",
            f"    OTM     : {self.otm_pct:.0%}  |  {self.structure_hint}",
            f"    Triggers: {'; '.join(self.entry_conditions[:3])}",
            f"  EXIT",
            f"    Take profit  : +{self.target_gain_pct:.0%}  →  {self.exit_scaling}",
            f"    Stop loss    : -{self.stop_loss_pct:.0%}",
            f"    Time stop    : {self.time_stop_days}d",
            f"    Close if regime → {', '.join(self.exit_on_regime)}",
            f"    Sentiment floor: close half if sector score +{self.exit_on_sentiment_reversal:.0f}pts",
            f"  Thesis: {self.thesis[:90]}",
            f"{'═'*60}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# EVENT INGESTOR — normalises external data to MarketEvent stream
# ═══════════════════════════════════════════════════════════════════════════

class EventIngestor:
    """
    Aggregates 12+ live signal sources into MarketEvent stream.

    Weight convention:
        Strong bearish  : -0.8 to -1.0  (large UW put flow, macro shock, CBOE P/C >1.2)
        Moderate bearish: -0.4 to -0.7  (dark pool, negative headline, FGI < 25)
        Mild bearish    : -0.1 to -0.3  (insider selling, analyst downgrade)
        Neutral         :  0.0
        Mild bullish    : +0.1 to +0.3  (FGI > 75 contrarian → exit signal; retail euphoria)
        Moderate bullish: +0.4 to +0.7  (breadth expansion, policy pivot signal)

    Decay half-lives (signal persistence):
        UW options flow       :  48h  — options expire, signal fades quickly
        Dark pool print       :  24h  — inst. positioning resets daily
        News headline         :  12h  — fast-moving, quickly priced-in
        FRED macro data       :  72h  — slow macro, longer signal life
        Congress / insider    : 168h  — persistent; legal disclosure lag
        Fear & Greed index    :  36h  — daily reading, moderate persistence
        CBOE put/call ratio   :  24h  — resets daily
        Google Trends spike   :  48h  — search fear lingers 2 days
        Finnhub analyst trend :  96h  — institutional consensus slow
        Polygon breadth       :  12h  — intraday breadth resets each session
        CoinGlass funding rate:   8h  — futures funding resets every 8h
        CoinGlass liquidation :   6h  — liquidation cascade front-loaded
        Crypto on-chain       :  48h  — exchange inflow signals last 2 days
        NCL signal            :  24h  — NCL flags active intel
        Regime shift          :  96h  — regime changes persist for days
    """

    # Tracked tickers for per-instrument analysis
    _WATCH_TICKERS = [
        "HYG", "KRE", "SPY", "QQQ", "IWM", "XLF",
        "JETS", "ZIM", "XLE", "SLV",
    ]
    _CRYPTO_SLUGS = ["bitcoin", "ethereum", "ripple"]

    def __init__(self, api_key_uw: str = "", api_key_fred: str = ""):
        self._uw_key  = api_key_uw  or os.environ.get("UNUSUAL_WHALES_API_KEY", "")
        self._fred_key = api_key_fred or os.environ.get("FRED_API_KEY", "")
        self._finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
        self._polygon_key = os.environ.get("POLYGON_API_KEY", "")

    # ── Unusual Whales ────────────────────────────────────────────────────

    async def ingest_uw_flow(self, limit: int = 50) -> List[MarketEvent]:
        """Map large UW options flow alerts → MarketEvents."""
        events: List[MarketEvent] = []
        if not self._uw_key:
            return events
        try:
            from integrations.unusual_whales_client import UnusualWhalesClient
            async with UnusualWhalesClient(self._uw_key) as client:
                alerts = await client.get_flow_alerts(limit=limit)
            for alert in alerts:
                ticker = alert.get("ticker", "").upper()
                sector = ticker_to_sector(ticker)
                opt_type = alert.get("option_type", "").lower()  # 'call' or 'put'
                total_val = float(alert.get("total_value", 0) or 0)
                bearish_bearish = opt_type == "put"
                # Ignore tiny prints
                if total_val < 100_000:
                    continue
                # Weight by size and direction
                size_factor = min(1.0, total_val / 5_000_000)  # cap at $5M
                base_weight = -0.6 if bearish_bearish else +0.4
                weight = base_weight * (0.5 + 0.5 * size_factor)
                events.append(MarketEvent(
                    source=EventSource.UNUSUAL_WHALES_FLOW,
                    sector=sector,
                    event_type="bearish_flow" if bearish_bearish else "bullish_flow",
                    weight=round(weight, 3),
                    decay_halflife_hours=48.0,
                    description=f"UW {opt_type.upper()} flow {ticker} ${total_val/1e6:.1f}M",
                    ticker=ticker,
                    raw=alert,
                ))
        except Exception as exc:
            logger.debug("UW flow ingest error: %s", exc)
        return events

    async def ingest_uw_darkpool(self, limit: int = 30) -> List[MarketEvent]:
        """Map dark pool prints → MarketEvents."""
        events: List[MarketEvent] = []
        if not self._uw_key:
            return events
        try:
            from integrations.unusual_whales_client import UnusualWhalesClient
            async with UnusualWhalesClient(self._uw_key) as client:
                prints = await client.get_dark_pool(limit=limit)
        except Exception:
            return events
        for p in prints:
            ticker = p.get("ticker", "").upper()
            sector = ticker_to_sector(ticker)
            size = float(p.get("size", 0) or 0)
            price = float(p.get("price", 0) or 0)
            notional = size * price
            if notional < 500_000:
                continue
            # Dark pool = institutional; assume bearish lean in current regime
            weight = -0.5 * min(1.0, notional / 10_000_000)
            events.append(MarketEvent(
                source=EventSource.UNUSUAL_WHALES_DARKPOOL,
                sector=sector,
                event_type="darkpool_print",
                weight=round(weight, 3),
                decay_halflife_hours=24.0,
                description=f"Dark pool {ticker} {size:,.0f} @ ${price:.2f} = ${notional/1e6:.1f}M",
                ticker=ticker,
                raw=p,
            ))
        return events

    async def ingest_uw_headlines(self, limit: int = 30) -> List[MarketEvent]:
        """Map news headlines → MarketEvents using keyword weighting."""
        events: List[MarketEvent] = []
        if not self._uw_key:
            return events
        BEARISH_WORDS = [
            "crash", "collapse", "default", "downgrade", "recession",
            "layoff", "bankruptcy", "stress", "crisis", "decline",
            "plunge", "warning", "rate hike", "inflation", "stagflation",
        ]
        BULLISH_WORDS = [
            "rally", "recovery", "rate cut", "stimulus", "buyback",
            "upgrade", "beat", "surge", "approved", "deal",
        ]
        try:
            from integrations.unusual_whales_client import UnusualWhalesClient
            async with UnusualWhalesClient(self._uw_key) as client:
                headlines = await client.get_news_headlines(limit=limit)
        except Exception:
            return events
        for h in headlines:
            title = (h.get("title") or h.get("headline") or "").lower()
            ticker = (h.get("ticker") or "").upper()
            sector = ticker_to_sector(ticker) if ticker else "index"
            bear_score = sum(1 for w in BEARISH_WORDS if w in title)
            bull_score = sum(1 for w in BULLISH_WORDS if w in title)
            net = bear_score - bull_score
            if net == 0:
                continue
            weight = max(-0.7, min(0.5, -net * 0.15))
            events.append(MarketEvent(
                source=EventSource.NEWS_HEADLINE,
                sector=sector,
                event_type="bearish_headline" if net > 0 else "bullish_headline",
                weight=round(weight, 3),
                decay_halflife_hours=12.0,
                description=title[:80],
                ticker=ticker or None,
                raw=h,
            ))
        return events

    async def ingest_fred_data(self) -> List[MarketEvent]:
        """Read latest FRED macro data → MarketEvents."""
        events: List[MarketEvent] = []
        if not self._fred_key:
            return events
        try:
            from integrations.fred_client import FREDClient
            client = FREDClient()
            dashboard = await client.get_macro_dashboard()
            # HY spread
            hy = dashboard.get("BAMLH0A0HYM2")
            if hy and hy.value is not None:
                spread = float(hy.value)
                if spread > 500:
                    events.append(MarketEvent(
                        source=EventSource.FRED_DATA, sector="credit",
                        event_type="macro_shock",
                        weight=-0.9, decay_halflife_hours=72.0,
                        description=f"HY spread SEVERE: {spread:.0f}bps",
                        raw={"series": "BAMLH0A0HYM2", "value": spread},
                    ))
                elif spread > 400:
                    events.append(MarketEvent(
                        source=EventSource.FRED_DATA, sector="credit",
                        event_type="credit_stress",
                        weight=-0.7, decay_halflife_hours=72.0,
                        description=f"HY spread elevated: {spread:.0f}bps",
                        raw={"series": "BAMLH0A0HYM2", "value": spread},
                    ))
            # VIX
            vix = dashboard.get("VIXCLS")
            if vix and vix.value is not None:
                v = float(vix.value)
                if v > 30:
                    events.append(MarketEvent(
                        source=EventSource.FRED_DATA, sector="index",
                        event_type="vol_spike",
                        weight=-0.8, decay_halflife_hours=48.0,
                        description=f"VIX SPIKE: {v:.1f}",
                        raw={"series": "VIXCLS", "value": v},
                    ))
            # Oil
            oil = dashboard.get("DCOILWTICO")
            if oil and oil.value is not None:
                o = float(oil.value)
                if o > 100:
                    for sector in ("airlines", "consumer"):
                        events.append(MarketEvent(
                            source=EventSource.FRED_DATA, sector=sector,
                            event_type="oil_shock",
                            weight=-0.75, decay_halflife_hours=72.0,
                            description=f"Oil shock ${o:.1f}/bbl → {sector} pressure",
                            raw={"series": "DCOILWTICO", "value": o},
                        ))
        except Exception as exc:
            logger.debug("FRED ingest error: %s", exc)
        return events

    # ════════════════════════════════════════════════════════════════
    # FEAR & GREED — Alternative.me (free, no key)
    # ════════════════════════════════════════════════════════════════

    async def ingest_fear_greed(self) -> List[MarketEvent]:
        """
        Crypto Fear & Greed Index + Stock FGI from Alternative.me.

        Strategy logic:
            Extreme Fear (0-25)  → bearish continuation in our vol-shock regime;
                                    put premiums expensive = exit signal for longs;
                                    confirms F1-F4 formulas
            Fear (25-45)         → moderate bearish confirmation
            Neutral (45-55)      → no signal
            Greed (55-75)        → mild bearish warning (retail complacency)
            Extreme Greed (75+)  → strong complacency = contrarian SELL trigger

        Weight formula:
            score = (50 - value) / 50   →  extreme fear = +1.0 bearish,
                                            extreme greed = -1.0 (risk of reversal)
        """
        events: List[MarketEvent] = []
        try:
            from integrations.fear_greed_client import FearGreedClient
            client = FearGreedClient()
            reading = await client.get_current()
            if reading is None:
                return events
            value = reading.value          # 0-100
            classification = reading.classification
            # Direct fear signal (our model = short bias; fear confirms trend,
            # extreme greed warns of regime change)
            if value <= 25:
                weight = -0.75  # extreme fear → strong bearish continuation
            elif value <= 40:
                weight = -0.45  # fear → moderate bearish
            elif value <= 55:
                return events   # neutral → no signal
            elif value <= 70:
                weight = +0.2   # greed → mild warning (retail complacency)
            else:
                weight = +0.45  # extreme greed → contrarian exit signal
            events.append(MarketEvent(
                source=EventSource.FEAR_GREED_CRYPTO,
                sector="crypto",
                event_type=f"fg_{classification.lower().replace(' ', '_')}",
                weight=round(weight, 3),
                decay_halflife_hours=36.0,
                description=f"Crypto FGI: {value}/100 ({classification})",
                raw={"value": value, "classification": classification},
            ))
            # Also emit index-sector signal (correlated)
            events.append(MarketEvent(
                source=EventSource.FEAR_GREED_STOCK,
                sector="index",
                event_type=f"fg_stock_{classification.lower().replace(' ', '_')}",
                weight=round(weight * 0.6, 3),  # index less sensitive to crypto FGI
                decay_halflife_hours=36.0,
                description=f"Stock FGI proxy: {value}/100 ({classification})",
                raw={"value": value, "classification": classification},
            ))
        except Exception as exc:
            logger.debug("FearGreed ingest error: %s", exc)
        return events

    # ════════════════════════════════════════════════════════════════
    # CBOE PUT/CALL RATIO — free public endpoint
    # ════════════════════════════════════════════════════════════════

    async def ingest_cboe_put_call(self) -> List[MarketEvent]:
        """
        CBOE equity put/call ratio from public CBOE data API.

        Interpretation:
            > 1.20  → panic hedging; extreme fear, bearish continuation (vol spike)
            0.80-1.20 → elevated put demand; bearish bias
            0.60-0.80 → neutral
            < 0.60  → complacent; risk of reversal (contrarian sell signal)
            < 0.50  → extreme greed / complacency — dangerous

        Source: https://cdn.cboe.com/api/global/us_indices_analytics/
        """
        events: List[MarketEvent] = []
        try:
            import urllib.request
            url = ("https://cdn.cboe.com/api/global/us_indices_analytics/"
                   "EQUITY_PC_RATIO.json")
            req = urllib.request.Request(url, headers={"User-Agent": "AAC/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read())
            # Response shape: {"data": [["date", ratio], ...]}
            data = raw.get("data", [])
            if not data:
                return events
            latest = data[-1]  # [date_str, ratio_float]
            ratio = float(latest[1]) if len(latest) > 1 else None
            if ratio is None:
                return events
            if ratio > 1.2:
                weight, label = -0.85, "extreme_put_buying"
            elif ratio > 0.85:
                weight, label = -0.55, "elevated_put_demand"
            elif ratio > 0.65:
                weight, label = 0.0, "neutral"
            elif ratio > 0.50:
                weight, label = +0.20, "low_hedging_complacent"
            else:
                weight, label = +0.45, "extreme_complacency_reversal_risk"
            if weight == 0.0:
                return events
            events.append(MarketEvent(
                source=EventSource.CBOE_PUT_CALL,
                sector="index",
                event_type=label,
                weight=round(weight, 3),
                decay_halflife_hours=24.0,
                description=f"CBOE equity P/C ratio: {ratio:.3f} ({label})",
                raw={"ratio": ratio, "date": str(latest[0])},
            ))
            # High put buying also signals credit sector stress
            if ratio > 1.0:
                events.append(MarketEvent(
                    source=EventSource.CBOE_PUT_CALL,
                    sector="credit",
                    event_type="put_demand_spilling_to_credit",
                    weight=round(weight * 0.6, 3),
                    decay_halflife_hours=24.0,
                    description=f"High P/C ratio ({ratio:.3f}) → credit stress proxy",
                    raw={"ratio": ratio},
                ))
        except Exception as exc:
            logger.debug("CBOE P/C ingest error: %s", exc)
        return events

    # ════════════════════════════════════════════════════════════════
    # FINNHUB — Insider transactions + Analyst trends + Calendar
    # ════════════════════════════════════════════════════════════════

    async def ingest_finnhub_insiders(self) -> List[MarketEvent]:
        """
        Finnhub insider transactions for watched tickers.
        Cluster selling (3+ insiders selling in 30d) = bearish regime confirmation.
        """
        events: List[MarketEvent] = []
        if not self._finnhub_key:
            return events
        try:
            from integrations.finnhub_client import FinnhubClient
            client = FinnhubClient()
            for ticker in ["KRE", "HYG", "SPY", "QQQ", "XLF"]:
                txns = await client.get_insider_transactions(ticker)
                if not txns:
                    continue
                sells = [t for t in txns
                         if hasattr(t, "transaction_type") and
                         "s" in str(t.transaction_type).lower()[:1]]
                if len(sells) >= 3:
                    sector = ticker_to_sector(ticker)
                    weight = -0.35 * min(1.0, len(sells) / 8.0)
                    events.append(MarketEvent(
                        source=EventSource.FINNHUB_INSIDER,
                        sector=sector,
                        event_type="cluster_insider_selling",
                        weight=round(weight, 3),
                        decay_halflife_hours=168.0,
                        description=f"{ticker}: {len(sells)} insiders selling (30d)",
                        ticker=ticker,
                        raw={"ticker": ticker, "sell_count": len(sells)},
                    ))
        except Exception as exc:
            logger.debug("Finnhub insider ingest error: %s", exc)
        return events

    async def ingest_finnhub_analyst_trends(self) -> List[MarketEvent]:
        """
        Finnhub analyst recommendation trends for sector ETFs.
        Rising sell/strongSell consensus → bearish weight.
        """
        events: List[MarketEvent] = []
        if not self._finnhub_key:
            return events
        try:
            from integrations.finnhub_client import FinnhubClient
            client = FinnhubClient()
            for ticker in ["KRE", "HYG", "JETS", "ZIM"]:
                trends = await client.get_recommendation_trends(ticker)
                if not trends or len(trends) < 2:
                    continue
                latest = trends[0]
                prior = trends[1]
                buy_now = (latest.get("buy", 0) or 0) + (latest.get("strongBuy", 0) or 0)
                sell_now = (latest.get("sell", 0) or 0) + (latest.get("strongSell", 0) or 0)
                buy_prev = (prior.get("buy", 0) or 0) + (prior.get("strongBuy", 0) or 0)
                sell_prev = (prior.get("sell", 0) or 0) + (prior.get("strongSell", 0) or 0)
                # Net change in bearish calls
                delta_sell = (sell_now - sell_prev)
                delta_buy  = (buy_now  - buy_prev)
                if delta_sell > 2 or (buy_now == 0 and sell_now > 0):
                    sector = ticker_to_sector(ticker)
                    weight = -0.3 * min(1.0, delta_sell / 5.0) if delta_sell > 0 else -0.25
                    events.append(MarketEvent(
                        source=EventSource.FINNHUB_ANALYST,
                        sector=sector,
                        event_type="analyst_downgrade_trend",
                        weight=round(weight, 3),
                        decay_halflife_hours=96.0,
                        description=(
                            f"{ticker}: analysts shifting bearish "
                            f"(sell+{delta_sell}, buy{delta_buy:+})"
                        ),
                        ticker=ticker,
                        raw=latest,
                    ))
        except Exception as exc:
            logger.debug("Finnhub analyst ingest error: %s", exc)
        return events

    async def ingest_finnhub_economic_calendar(self) -> List[MarketEvent]:
        """
        Finnhub economic calendar — pre-position before FOMC, CPI, NFP.
        Events within 2 days raise vol-bid; within event day = heightened weight.
        """
        events: List[MarketEvent] = []
        if not self._finnhub_key:
            return events
        HIGH_IMPACT = ["fomc", "fed", "cpi", "nfp", "nonfarm", "gdp", "pce",
                        "inflation", "jobs", "unemployment", "rate decision"]
        try:
            from integrations.finnhub_client import FinnhubClient
            from datetime import date, timedelta
            client = FinnhubClient()
            today = date.today()
            events_raw = await client.get_economic_calendar(
                from_date=today.isoformat(),
                to_date=(today + timedelta(days=3)).isoformat(),
            )
            for ev in (events_raw or []):
                name = (ev.get("event") or "").lower()
                impact = (ev.get("impact") or "").lower()
                if not any(k in name for k in HIGH_IMPACT) and impact != "high":
                    continue
                ev_date_str = ev.get("date", "")
                try:
                    ev_date = date.fromisoformat(str(ev_date_str)[:10])
                except Exception:
                    continue
                days_away = (ev_date - today).days
                if days_away > 2:
                    continue
                # Closer to event = higher vol anticipation weight
                weight = -0.35 if days_away == 0 else -0.20
                events.append(MarketEvent(
                    source=EventSource.FINNHUB_CALENDAR,
                    sector="index",
                    event_type=f"macro_event_{'today' if days_away==0 else f'{days_away}d'}",
                    weight=weight,
                    decay_halflife_hours=48.0,
                    description=f"High-impact event: {ev.get('event', name)} ({days_away}d away)",
                    raw=ev,
                ))
        except Exception as exc:
            logger.debug("Finnhub calendar ingest error: %s", exc)
        return events

    # ════════════════════════════════════════════════════════════════
    # POLYGON — Market breadth (% advancing/declining tickers)
    # ════════════════════════════════════════════════════════════════

    async def ingest_polygon_breadth(self) -> List[MarketEvent]:
        """
        Polygon grouped daily bars → compute market breadth.

        Breadth signals:
            > 70% declining → strong negative breadth → weight -0.7
            50-70% declining → moderate negative breadth → weight -0.4
            30-50% declining → neutral
            < 30% declining  → positive breadth → weight +0.3

        Also detects NDX/SPX divergence: tech declining while credit holding.
        """
        events: List[MarketEvent] = []
        if not self._polygon_key:
            return events
        try:
            from integrations.polygon_client import PolygonClient
            client = PolygonClient()
            bars = await client.get_grouped_daily()
            if not bars:
                return events
            total = len(bars)
            declining = sum(1 for b in bars if hasattr(b, 'close') and
                            hasattr(b, 'open') and b.close < b.open)
            if total < 100:  # too few tickers = incomplete session
                return events
            pct_declining = declining / total
            if pct_declining > 0.70:
                weight, label = -0.75, "broad_market_selloff"
            elif pct_declining > 0.55:
                weight, label = -0.40, "negative_breadth"
            elif pct_declining < 0.30:
                weight, label = +0.30, "positive_breadth_expansion"
            else:
                return events  # neutral zone
            events.append(MarketEvent(
                source=EventSource.POLYGON_BREADTH,
                sector="index",
                event_type=label,
                weight=round(weight, 3),
                decay_halflife_hours=12.0,
                description=(
                    f"Market breadth: {pct_declining:.0%} of {total:,} "
                    f"tickers declining ({label})"
                ),
                raw={"pct_declining": pct_declining, "total": total, "declining": declining},
            ))
        except Exception as exc:
            logger.debug("Polygon breadth ingest error: %s", exc)
        return events

    # ════════════════════════════════════════════════════════════════
    # COINGLASS — Crypto funding rates, liquidations, open interest
    # ════════════════════════════════════════════════════════════════

    async def ingest_coinglass(self) -> List[MarketEvent]:
        """
        CoinGlass free public API — crypto derivatives intelligence.

        Funding rate signals:
            < -0.05% per 8h  → shorts dominant, bearish trend confirmation
            > +0.10% per 8h  → longs overextended, crowded → reversal risk

        Liquidation signals:
            Long liquidation spike → price crash accelerated
            Short liquidation spike → short squeeze, momentum shift

        Open interest vs price:
            OI rising + price falling → new shorts entering (bearish)
            OI falling + price falling → deleveraging (bearish exhaustion)
        """
        events: List[MarketEvent] = []
        try:
            import urllib.request
            # CoinGlass free API — BTC funding rate
            url = "https://open-api.coinglass.com/public/v2/funding"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AAC/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read())
            data = raw.get("data", [])
            if not data:
                raise ValueError("empty funding response")
            # Find BTC funding across major exchanges
            btc_rates = []
            for item in data:
                symbol = str(item.get("symbol", "")).upper()
                if "BTC" not in symbol:
                    continue
                rate = item.get("fundingRate")
                if rate is not None:
                    try:
                        btc_rates.append(float(rate))
                    except (TypeError, ValueError):
                        pass
            if btc_rates:
                avg_rate = sum(btc_rates) / len(btc_rates)
                if avg_rate < -0.04:
                    weight, label = -0.55, "funding_rate_bearish_extreme"
                elif avg_rate < -0.01:
                    weight, label = -0.30, "funding_rate_bearish"
                elif avg_rate > 0.10:
                    weight, label = +0.40, "funding_rate_longs_crowded"
                elif avg_rate > 0.05:
                    weight, label = +0.20, "funding_rate_longs_elevated"
                else:
                    weight, label = 0.0, "neutral"
                if weight != 0.0:
                    events.append(MarketEvent(
                        source=EventSource.COINGLASS,
                        sector="crypto",
                        event_type=f"coinglass_{label}",
                        weight=round(weight, 3),
                        decay_halflife_hours=8.0,
                        description=(
                            f"BTC avg funding rate: {avg_rate*100:.4f}%/8h ({label})"
                        ),
                        raw={"avg_funding_rate": avg_rate, "exchange_count": len(btc_rates)},
                    ))
        except Exception as exc:
            logger.debug("CoinGlass ingest error: %s", exc)
        return events

    # ════════════════════════════════════════════════════════════════
    # STOCKTWITS — Retail social sentiment (no key required)
    # ════════════════════════════════════════════════════════════════

    async def ingest_stocktwits_sentiment(self) -> List[MarketEvent]:
        """
        Stocktwits public stream API — retail sentiment for tracked tickers.
        No API key required.

        Signals:
            Bullish messages > 70%: retail crowd long → contrarian warning
            Bearish messages > 60%: retail panic → can confirm or mean-revert
            Large message volume: topic heating up
        """
        events: List[MarketEvent] = []
        import urllib.request
        for ticker in ["HYG", "SPY", "QQQ", "BTC.X", "ETH.X"]:
            try:
                url = (f"https://api.stocktwits.com/api/2/streams/symbol/"
                       f"{ticker}.json?limit=30")
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (AAC Trading Bot)"},
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    raw = json.loads(resp.read())
                messages = raw.get("messages", [])
                if not messages:
                    continue
                bull = sum(1 for m in messages
                           if m.get("entities", {}).get("sentiment", {}) and
                           m["entities"]["sentiment"].get("basic") == "Bullish")
                bear = sum(1 for m in messages
                           if m.get("entities", {}).get("sentiment", {}) and
                           m["entities"]["sentiment"].get("basic") == "Bearish")
                total_labeled = bull + bear
                if total_labeled < 5:
                    continue
                bull_pct = bull / total_labeled
                # Contrarian logic: retail > 70% bull = complacency warning
                if bull_pct > 0.75:
                    weight = +0.25  # too bullish = danger zone
                    label = "retail_euphoria"
                elif bull_pct < 0.30:
                    weight = -0.20  # retail panic selling — not necessarily contrarian yet
                    label = "retail_panic"
                else:
                    continue
                sector = ticker_to_sector(ticker.replace(".X", ""))
                if sector == "index" and ".X" in ticker:
                    sector = "crypto"
                events.append(MarketEvent(
                    source=EventSource.STOCKTWITS_SENTIMENT,
                    sector=sector,
                    event_type=f"stocktwits_{label}",
                    weight=round(weight, 3),
                    decay_halflife_hours=12.0,
                    description=(
                        f"Stocktwits {ticker}: {bull_pct:.0%} bullish "
                        f"({bull}B/{bear}Be of {total_labeled})"
                    ),
                    ticker=ticker.replace(".X", ""),
                    raw={"bull": bull, "bear": bear, "bull_pct": bull_pct},
                ))
            except Exception as exc:
                logger.debug("Stocktwits ingest %s error: %s", ticker, exc)
        return events

    # ════════════════════════════════════════════════════════════════
    # REDDIT / TRADESTIE — WSB/Options sentiment (no key required)
    # ════════════════════════════════════════════════════════════════

    async def ingest_reddit_sentiment(self) -> List[MarketEvent]:
        """
        Tradestie free API — Reddit WSB options sentiment & top tickers.
        No API key required: https://tradestie.com/api/v1/apps/reddit

        Signals:
            Sentiment < -0.3  → WSB bearish lean → mild confirmation
            Sentiment > +0.5  → WSB euphoria → contrarian exit warning
            High mention spike for a ticker with bearish sentiment → focus
        """
        events: List[MarketEvent] = []
        try:
            import urllib.request
            url = "https://tradestie.com/api/v1/apps/reddit?date=latest"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AAC/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read())
            # API returns list of {ticker, no_of_comments, sentiment, sentiment_score}
            top = raw[:20] if isinstance(raw, list) else raw.get("data", [])[:20]
            for item in top:
                ticker = str(item.get("ticker", "")).upper()
                sentiment = item.get("sentiment", "")
                score = float(item.get("sentiment_score", 0) or 0)
                mentions = int(item.get("no_of_comments", 0) or 0)
                if mentions < 10:
                    continue
                sector = ticker_to_sector(ticker)
                if sentiment == "Bearish" and score < -0.1:
                    events.append(MarketEvent(
                        source=EventSource.REDDIT_SENTIMENT,
                        sector=sector,
                        event_type="wsb_bearish_focus",
                        weight=round(-0.25 * min(1.0, abs(score)), 3),
                        decay_halflife_hours=24.0,
                        description=f"WSB bearish on {ticker} ({mentions} mentions, score={score:.2f})",
                        ticker=ticker,
                        raw=item,
                    ))
                elif sentiment == "Bullish" and score > 0.5 and mentions > 50:
                    # WSB euphoria = contrarian sell signal
                    events.append(MarketEvent(
                        source=EventSource.REDDIT_SENTIMENT,
                        sector=sector,
                        event_type="wsb_euphoria_contrarian",
                        weight=round(+0.30, 3),
                        decay_halflife_hours=24.0,
                        description=f"WSB euphoria on {ticker} ({mentions} mentions) — reversal risk",
                        ticker=ticker,
                        raw=item,
                    ))
        except Exception as exc:
            logger.debug("Reddit/Tradestie ingest error: %s", exc)
        return events

    # ════════════════════════════════════════════════════════════════
    # GOOGLE TRENDS — Search sentiment spike detection
    # ════════════════════════════════════════════════════════════════

    async def ingest_google_trends(self) -> List[MarketEvent]:
        """
        Google Trends search interest for macro fear keywords.
        Rising search volume = retail panic detection BEFORE price moves.

        Tracked terms:
            Recession / market crash / bank run / hyperinflation / short selling
        Strategy: spike in fear keywords → weight -0.4 for broad market
        """
        events: List[MarketEvent] = []
        try:
            from integrations.google_trends_client import GoogleTrendsClient
            client = GoogleTrendsClient()
            # Run in thread pool — pytrends is sync
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: client.get_ticker_sentiment_score(
                    "recession", timeframe="now 7-d"
                )
            )
            if result and isinstance(result, dict):
                trend_score = result.get("score", 0)  # 0-100
                recent = result.get("recent_avg", 0)
                baseline = result.get("baseline_avg", 0)
                if baseline > 0 and recent > baseline * 1.5:
                    spike_factor = min(1.0, (recent - baseline) / baseline)
                    weight = -0.35 * (0.5 + 0.5 * spike_factor)
                    events.append(MarketEvent(
                        source=EventSource.GOOGLE_TRENDS,
                        sector="index",
                        event_type="recession_search_spike",
                        weight=round(weight, 3),
                        decay_halflife_hours=48.0,
                        description=(
                            f"Google Trends 'recession' spike: "
                            f"{recent:.0f} vs baseline {baseline:.0f} "
                            f"(+{spike_factor:.0%})"
                        ),
                        raw={"recent": recent, "baseline": baseline, "score": trend_score},
                    ))
        except Exception as exc:
            logger.debug("Google Trends ingest error: %s", exc)
        return events

    # ════════════════════════════════════════════════════════════════
    # CRYPTO ON-CHAIN — CoinGecko-derived signals
    # ════════════════════════════════════════════════════════════════

    async def ingest_crypto_onchain(self) -> List[MarketEvent]:
        """
        Crypto on-chain derived signals via CoinGecko:
        - Large price/volume divergence (distribution top pattern)
        - Dominance shift between BTC/ETH/alts
        - 24h anomalous volume spikes for BTC/ETH
        """
        events: List[MarketEvent] = []
        try:
            import urllib.request
            cg_key = os.environ.get("COINGECKO_API_KEY", "")
            headers = {"User-Agent": "AAC/1.0", "Accept": "application/json"}
            if cg_key:
                headers["x-cg-demo-api-key"] = cg_key
            # Global market data: BTC dominance, market cap, 24h volume
            url = "https://api.coingecko.com/api/v3/global"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read())
            data = raw.get("data", {})
            # BTC dominance signal
            btc_dom = data.get("market_cap_percentage", {}).get("btc", 0)
            mktcap_change_24h = data.get("market_cap_change_percentage_24h_usd", 0)
            if mktcap_change_24h < -5:
                events.append(MarketEvent(
                    source=EventSource.CRYPTO_ONCHAIN,
                    sector="crypto",
                    event_type="global_crypto_selloff",
                    weight=round(-0.65 * min(1.0, abs(mktcap_change_24h) / 10), 3),
                    decay_halflife_hours=48.0,
                    description=(
                        f"Crypto global market cap fell {mktcap_change_24h:.1f}% in 24h "
                        f"(BTC dom: {btc_dom:.1f}%)"
                    ),
                    raw={"mktcap_change_24h": mktcap_change_24h, "btc_dominance": btc_dom},
                ))
            # BTC dominance spike (flight to safety within crypto)
            if btc_dom > 58:
                events.append(MarketEvent(
                    source=EventSource.CRYPTO_ONCHAIN,
                    sector="crypto",
                    event_type="btc_dominance_spike_altcoin_risk_off",
                    weight=-0.30,
                    decay_halflife_hours=24.0,
                    description=f"BTC dominance: {btc_dom:.1f}% — altcoins risk-off",
                    raw={"btc_dominance": btc_dom},
                ))
            # Rapid price check for BTC
            url2 = ("https://api.coingecko.com/api/v3/simple/price"
                    "?ids=bitcoin,ethereum&vs_currencies=usd"
                    "&include_24hr_change=true&include_24hr_vol=true")
            req2 = urllib.request.Request(url2, headers=headers)
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                prices = json.loads(resp2.read())
            btc_chg = prices.get("bitcoin", {}).get("usd_24h_change", 0) or 0
            eth_chg = prices.get("ethereum", {}).get("usd_24h_change", 0) or 0
            for coin, chg in [("BTC", btc_chg), ("ETH", eth_chg)]:
                if chg < -8:
                    events.append(MarketEvent(
                        source=EventSource.CRYPTO_ONCHAIN,
                        sector="crypto",
                        event_type="crypto_large_drawdown",
                        weight=round(-0.7 * min(1.0, abs(chg) / 15), 3),
                        decay_halflife_hours=24.0,
                        description=f"{coin} fell {chg:.1f}% in 24h — large drawdown",
                        ticker=coin,
                        raw={"coin": coin, "change_24h": chg},
                    ))
        except Exception as exc:
            logger.debug("Crypto on-chain ingest error: %s", exc)
        return events

    # ════════════════════════════════════════════════════════════════
    # CoinGecko deep crypto intelligence  (5 new ingest methods)
    # ════════════════════════════════════════════════════════════════

    async def ingest_coingecko_trending(self) -> List[MarketEvent]:
        """
        /search/trending — top-15 trending coins + top-6 categories.
        Detects retail FOMO bubbles, meme mania, and speculative cycle peaks.
        Update frequency: every 10 minutes on CoinGecko side.
        """
        events: List[MarketEvent] = []
        try:
            import urllib.request
            cg_key = os.environ.get("COINGECKO_API_KEY", "")
            headers: Dict[str, str] = {"User-Agent": "AAC/1.0", "Accept": "application/json"}
            if cg_key:
                headers["x-cg-demo-api-key"] = cg_key

            url = "https://api.coingecko.com/api/v3/search/trending"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read())

            coins = raw.get("coins", [])
            categories = raw.get("categories", [])

            # ── Trending coin analysis ───────────────────────────────
            low_cap_fomo_count = 0
            for item in coins[:15]:
                coin = item.get("item", {})
                mcap_rank = coin.get("market_cap_rank") or 9999
                data = coin.get("data", {})
                pct_change = 0.0
                try:
                    pct_change = float(
                        data.get("price_change_percentage_24h", {}).get("usd", 0) or 0
                    )
                except (TypeError, ValueError):
                    pct_change = 0.0
                name = coin.get("name", "unknown")
                score = coin.get("score", 0)

                # Low-cap coin trending with high gain = retail FOMO
                if mcap_rank > 200 and pct_change > 30:
                    low_cap_fomo_count += 1
                    events.append(MarketEvent(
                        source=EventSource.COINGECKO_TRENDING,
                        sector="crypto",
                        event_type="low_cap_trending_fomo",
                        weight=round(min(0.40, 0.25 + pct_change / 200), 3),
                        decay_halflife_hours=12.0,
                        description=(
                            f"{name} (rank #{mcap_rank}) trending: "
                            f"+{pct_change:.0f}% 24h — retail FOMO signal"
                        ),
                        raw={"name": name, "mcap_rank": mcap_rank,
                             "pct_change_24h": pct_change, "score": score},
                    ))

            # Multiple low-cap FOMO coins = speculative mania → contrarian sell signal
            if low_cap_fomo_count >= 3:
                events.append(MarketEvent(
                    source=EventSource.COINGECKO_TRENDING,
                    sector="crypto",
                    event_type="speculative_mania_peak",
                    weight=0.50,
                    decay_halflife_hours=8.0,
                    description=(
                        f"{low_cap_fomo_count} low-cap coins trending with >30% gain — "
                        "speculative mania, cycle peak risk"
                    ),
                    raw={"low_cap_fomo_count": low_cap_fomo_count},
                ))

            # ── Trending category analysis ───────────────────────────
            for cat in categories[:6]:
                cat_name = cat.get("name", "")
                coin_count = cat.get("coins_count", 0)
                data = cat.get("data", {})
                mktcap_1h_change = 0.0
                try:
                    mktcap_1h_change = float(
                        data.get("market_cap_1h_change", 0) or 0
                    )
                except (TypeError, ValueError):
                    mktcap_1h_change = 0.0

                # Category with massive 1h gain trending = sector euphoria
                is_meme = any(kw in cat_name.lower() for kw in
                              ["meme", "dog", "shib", "pepe", "gaming", "nft"])
                if mktcap_1h_change > 15 or (is_meme and mktcap_1h_change > 5):
                    events.append(MarketEvent(
                        source=EventSource.COINGECKO_TRENDING,
                        sector="crypto",
                        event_type="category_euphoria",
                        weight=round(min(0.45, 0.20 + mktcap_1h_change / 80), 3),
                        decay_halflife_hours=8.0,
                        description=(
                            f"Category '{cat_name}' trending +{mktcap_1h_change:.1f}% 1h "
                            f"({coin_count} coins) — speculative euphoria"
                        ),
                        raw={"category": cat_name, "mktcap_1h_change": mktcap_1h_change,
                             "coins_count": coin_count},
                    ))

        except Exception as exc:
            logger.debug("CoinGecko trending ingest error: %s", exc)
        return events

    async def ingest_coingecko_ohlc(self) -> List[MarketEvent]:
        """
        /coins/{id}/ohlc — BTC + ETH 30-day OHLC candles.
        Computes RSI(14), doji detection, bearish engulfing pattern.
        Update frequency: every 15 minutes on CoinGecko side.
        """
        events: List[MarketEvent] = []
        try:
            import urllib.request
            cg_key = os.environ.get("COINGECKO_API_KEY", "")
            headers: Dict[str, str] = {"User-Agent": "AAC/1.0", "Accept": "application/json"}
            if cg_key:
                headers["x-cg-demo-api-key"] = cg_key

            def _fetch_ohlc(coin_id: str) -> List[List[float]]:
                url = (
                    f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
                    "?vs_currency=usd&days=30"
                )
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=12) as resp:
                    return json.loads(resp.read())  # type: ignore[return-value]

            def _rsi(closes: List[float], period: int = 14) -> float:
                """Wilder RSI from a list of close prices."""
                if len(closes) < period + 1:
                    return 50.0
                gains, losses = [], []
                for i in range(1, len(closes)):
                    diff = closes[i] - closes[i - 1]
                    gains.append(max(diff, 0.0))
                    losses.append(max(-diff, 0.0))
                avg_gain = sum(gains[-period:]) / period
                avg_loss = sum(losses[-period:]) / period
                if avg_loss == 0:
                    return 100.0
                rs = avg_gain / avg_loss
                return round(100.0 - 100.0 / (1.0 + rs), 2)

            for coin_id, label in [("bitcoin", "BTC"), ("ethereum", "ETH")]:
                try:
                    candles = _fetch_ohlc(coin_id)
                except Exception as fetch_exc:
                    logger.debug("OHLC fetch %s failed: %s", coin_id, fetch_exc)
                    continue

                if len(candles) < 20:
                    continue

                # candles = [[ts, open, high, low, close], ...]
                closes = [c[4] for c in candles]
                rsi = _rsi(closes)

                # RSI signals
                if rsi > 70:
                    events.append(MarketEvent(
                        source=EventSource.COINGECKO_OHLC,
                        sector="crypto",
                        event_type="rsi_overbought",
                        weight=round(min(0.65, 0.35 + (rsi - 70) / 60), 3),
                        decay_halflife_hours=24.0,
                        description=f"{label} RSI={rsi:.1f} — overbought, reversal risk",
                        ticker=label,
                        raw={"coin": coin_id, "rsi": rsi},
                    ))
                elif rsi < 30:
                    events.append(MarketEvent(
                        source=EventSource.COINGECKO_OHLC,
                        sector="crypto",
                        event_type="rsi_oversold",
                        weight=round(max(-0.25, -0.15 - (30 - rsi) / 80), 3),
                        decay_halflife_hours=12.0,
                        description=f"{label} RSI={rsi:.1f} — oversold, potential bounce",
                        ticker=label,
                        raw={"coin": coin_id, "rsi": rsi},
                    ))

                # Doji detection on last candle (indecision at key level)
                last = candles[-1]
                o, h, l_price, c = last[1], last[2], last[3], last[4]
                candle_range = h - l_price
                body = abs(c - o)
                if candle_range > 0 and body / candle_range < 0.15:
                    events.append(MarketEvent(
                        source=EventSource.COINGECKO_OHLC,
                        sector="crypto",
                        event_type="doji_indecision",
                        weight=0.20,
                        decay_halflife_hours=16.0,
                        description=(
                            f"{label} doji candle detected — indecision at "
                            f"${c:,.0f} (body {body/c*100:.2f}% of range)"
                        ),
                        ticker=label,
                        raw={"coin": coin_id, "open": o, "close": c, "high": h, "low": l_price},
                    ))

                # Bearish engulfing: last candle's body engulfs previous candle's body
                if len(candles) >= 2:
                    prev = candles[-2]
                    p_o, p_c = prev[1], prev[4]
                    # Previous candle bullish (green), current bearish (red) and larger
                    if (p_c > p_o and c < o
                            and o > p_c and c < p_o):
                        events.append(MarketEvent(
                            source=EventSource.COINGECKO_OHLC,
                            sector="crypto",
                            event_type="bearish_engulfing",
                            weight=0.45,
                            decay_halflife_hours=24.0,
                            description=(
                                f"{label} bearish engulfing candle — "
                                f"strong reversal signal at ${c:,.0f}"
                            ),
                            ticker=label,
                            raw={"coin": coin_id, "prev_open": p_o, "prev_close": p_c,
                                 "open": o, "close": c},
                        ))

                # SMA10 / SMA30 death-cross check
                if len(closes) >= 30:
                    sma10 = sum(closes[-10:]) / 10
                    sma30 = sum(closes[-30:]) / 30
                    if sma10 < sma30 * 0.98:  # 2% gap below = confirmed death cross
                        events.append(MarketEvent(
                            source=EventSource.COINGECKO_OHLC,
                            sector="crypto",
                            event_type="death_cross",
                            weight=0.40,
                            decay_halflife_hours=48.0,
                            description=(
                                f"{label} SMA10={sma10:,.0f} < SMA30={sma30:,.0f} — "
                                "death cross bearish"
                            ),
                            ticker=label,
                            raw={"coin": coin_id, "sma10": sma10, "sma30": sma30, "rsi": rsi},
                        ))

        except Exception as exc:
            logger.debug("CoinGecko OHLC ingest error: %s", exc)
        return events

    async def ingest_coingecko_markets(self) -> List[MarketEvent]:
        """
        /coins/markets — top 100 coins by volume.
        Breadth analysis, ATH resistance zones, volume anomalies, FDV inflation risk.
        Update frequency: every 30 seconds on CoinGecko side.
        """
        events: List[MarketEvent] = []
        try:
            import urllib.request
            cg_key = os.environ.get("COINGECKO_API_KEY", "")
            headers: Dict[str, str] = {"User-Agent": "AAC/1.0", "Accept": "application/json"}
            if cg_key:
                headers["x-cg-demo-api-key"] = cg_key

            url = (
                "https://api.coingecko.com/api/v3/coins/markets"
                "?vs_currency=usd&order=volume_desc&per_page=100&page=1"
                "&price_change_percentage=1h,24h,7d"
            )
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                coins: List[Dict[str, Any]] = json.loads(resp.read())

            if not coins:
                return events

            # ── Breadth: % of top-100 declining in 24h ───────────────
            declining = sum(
                1 for c in coins
                if (c.get("price_change_percentage_24h") or 0) < 0
            )
            breadth_pct = declining / len(coins)
            if breadth_pct > 0.65:
                events.append(MarketEvent(
                    source=EventSource.COINGECKO_MARKETS,
                    sector="crypto",
                    event_type="alt_market_breadth_selloff",
                    weight=round(-0.30 - breadth_pct * 0.35, 3),
                    decay_halflife_hours=24.0,
                    description=(
                        f"{declining}/100 top coins declining 24h ({breadth_pct:.0%}) — "
                        "broad altcoin selloff"
                    ),
                    raw={"declining": declining, "total": len(coins), "breadth_pct": breadth_pct},
                ))

            # ── Volume spike detection (vol/mcap ratio > 0.20) ───────
            high_vol_count = 0
            for coin in coins:
                mcap = coin.get("market_cap") or 0
                vol = coin.get("total_volume") or 0
                if mcap > 0 and vol / mcap > 0.20:
                    high_vol_count += 1
            if high_vol_count >= 5:
                events.append(MarketEvent(
                    source=EventSource.COINGECKO_MARKETS,
                    sector="crypto",
                    event_type="anomalous_volume_spike",
                    weight=round(min(0.45, 0.20 + high_vol_count * 0.03), 3),
                    decay_halflife_hours=12.0,
                    description=(
                        f"{high_vol_count} of top-100 coins with vol/mcap > 20% — "
                        "panic or euphoria volume"
                    ),
                    raw={"high_vol_coins": high_vol_count},
                ))

            # ── ATH resistance zone: coins within 3% of all-time high ─
            ath_proximity_count = sum(
                1 for c in coins
                if -3 <= (c.get("ath_change_percentage") or -100) <= 0
            )
            if ath_proximity_count >= 8:
                events.append(MarketEvent(
                    source=EventSource.COINGECKO_MARKETS,
                    sector="crypto",
                    event_type="ath_resistance_zone",
                    weight=round(min(0.40, 0.15 + ath_proximity_count * 0.025), 3),
                    decay_halflife_hours=36.0,
                    description=(
                        f"{ath_proximity_count} top-100 coins within 3% of ATH — "
                        "major resistance zone, distribution risk"
                    ),
                    raw={"ath_proximity_count": ath_proximity_count},
                ))

            # ── FDV ratio: systemic inflation risk ────────────────────
            fdv_ratios = [
                c.get("fully_diluted_valuation", 0) or 0
                for c in coins[:20]
            ]
            mcaps = [c.get("market_cap", 0) or 0 for c in coins[:20]]
            valid_pairs = [
                (m, f) for m, f in zip(mcaps, fdv_ratios)
                if f > 0 and m > 0
            ]
            if valid_pairs:
                avg_fdv_ratio = sum(m / f for m, f in valid_pairs) / len(valid_pairs)
                if avg_fdv_ratio < 0.35:
                    events.append(MarketEvent(
                        source=EventSource.COINGECKO_MARKETS,
                        sector="crypto",
                        event_type="systemic_fdv_inflation_risk",
                        weight=-0.30,
                        decay_halflife_hours=96.0,
                        description=(
                            f"Avg mcap/FDV ratio for top-20: {avg_fdv_ratio:.2f} — "
                            "large token unlock schedule, latent sell pressure"
                        ),
                        raw={"avg_fdv_ratio": avg_fdv_ratio, "sample": len(valid_pairs)},
                    ))

        except Exception as exc:
            logger.debug("CoinGecko markets ingest error: %s", exc)
        return events

    async def ingest_coingecko_coin_sentiment(self) -> List[MarketEvent]:
        """
        /coins/{id} — community sentiment votes, watchlist users, developer activity.
        Tracks BTC, ETH, SOL for contrarian extremes and fundamental decay signals.
        Update frequency: every 60 seconds on CoinGecko side.
        """
        events: List[MarketEvent] = []
        try:
            import urllib.request
            cg_key = os.environ.get("COINGECKO_API_KEY", "")
            headers: Dict[str, str] = {"User-Agent": "AAC/1.0", "Accept": "application/json"}
            if cg_key:
                headers["x-cg-demo-api-key"] = cg_key

            for coin_id, label in [
                ("bitcoin", "BTC"), ("ethereum", "ETH"), ("solana", "SOL")
            ]:
                try:
                    url = (
                        f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                        "?localization=false&tickers=false&market_data=true"
                        "&community_data=true&developer_data=true"
                    )
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        data: Dict[str, Any] = json.loads(resp.read())
                except Exception as fetch_exc:
                    logger.debug("CoinGecko coin data %s error: %s", coin_id, fetch_exc)
                    continue

                votes_up = data.get("sentiment_votes_up_percentage") or 0.0
                votes_down = data.get("sentiment_votes_down_percentage") or 0.0
                watchlist = data.get("watchlist_portfolio_users") or 0
                dev_data = data.get("developer_data", {})
                commits_4w = dev_data.get("commit_count_4_weeks") or 0
                market_data = data.get("market_data", {})
                fdv_ratio = market_data.get("market_cap_fdv_ratio") or 1.0

                # Extreme bullish sentiment → contrarian bearish
                if votes_up > 80:
                    events.append(MarketEvent(
                        source=EventSource.COINGECKO_SENTIMENT,
                        sector="crypto",
                        event_type="extreme_bullish_sentiment",
                        weight=round(min(0.40, 0.20 + (votes_up - 80) / 50), 3),
                        decay_halflife_hours=48.0,
                        description=(
                            f"{label} community {votes_up:.0f}% bullish — "
                            "extreme euphoria, contrarian warning"
                        ),
                        ticker=label,
                        raw={"coin": coin_id, "votes_up": votes_up, "votes_down": votes_down},
                    ))

                # Extreme bearish sentiment → confirms short bias
                elif votes_up < 30:
                    events.append(MarketEvent(
                        source=EventSource.COINGECKO_SENTIMENT,
                        sector="crypto",
                        event_type="extreme_bearish_sentiment",
                        weight=round(max(-0.30, -0.15 - (30 - votes_up) / 100), 3),
                        decay_halflife_hours=48.0,
                        description=(
                            f"{label} community only {votes_up:.0f}% bullish — "
                            "extreme fear, confirms bearish positioning"
                        ),
                        ticker=label,
                        raw={"coin": coin_id, "votes_up": votes_up, "votes_down": votes_down},
                    ))

                # Low developer activity → fundamental decay
                if commits_4w < 5 and coin_id != "bitcoin":  # BTC intentionally low commit rate
                    events.append(MarketEvent(
                        source=EventSource.COINGECKO_SENTIMENT,
                        sector="crypto",
                        event_type="developer_activity_decay",
                        weight=-0.25,
                        decay_halflife_hours=96.0,
                        description=(
                            f"{label} only {commits_4w} commits in 4 weeks — "
                            "developer abandonment signal"
                        ),
                        ticker=label,
                        raw={"coin": coin_id, "commits_4w": commits_4w},
                    ))

                # Low FDV ratio → heavy future inflation / unlock pressure
                if 0 < fdv_ratio < 0.30:
                    events.append(MarketEvent(
                        source=EventSource.COINGECKO_SENTIMENT,
                        sector="crypto",
                        event_type="heavy_token_unlock_pressure",
                        weight=-0.35,
                        decay_halflife_hours=120.0,
                        description=(
                            f"{label} mcap/FDV = {fdv_ratio:.2f} — "
                            "<30% circulating, large unlock schedule ahead"
                        ),
                        ticker=label,
                        raw={"coin": coin_id, "fdv_ratio": fdv_ratio,
                             "watchlist_users": watchlist},
                    ))

        except Exception as exc:
            logger.debug("CoinGecko coin sentiment ingest error: %s", exc)
        return events

    async def ingest_coingecko_categories(self) -> List[MarketEvent]:
        """
        /coins/categories — all categories with 24h market cap change.
        Tracks DeFi (credit stress proxy), L1 infrastructure, meme bubble.
        Update frequency: every 5 minutes on CoinGecko side.
        """
        events: List[MarketEvent] = []
        try:
            import urllib.request
            cg_key = os.environ.get("COINGECKO_API_KEY", "")
            headers: Dict[str, str] = {"User-Agent": "AAC/1.0", "Accept": "application/json"}
            if cg_key:
                headers["x-cg-demo-api-key"] = cg_key

            url = (
                "https://api.coingecko.com/api/v3/coins/categories"
                "?order=market_cap_change_24h_desc"
            )
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                categories: List[Dict[str, Any]] = json.loads(resp.read())

            # Build a lookup: category id/name → 24h change
            cat_changes: Dict[str, float] = {}
            for cat in categories:
                cat_id = cat.get("id", "").lower()
                cat_name = cat.get("name", "").lower()
                change = cat.get("market_cap_change_24h") or 0.0
                cat_changes[cat_id] = change
                cat_changes[cat_name] = change

            # ── DeFi: credit stress proxy ─────────────────────────────
            defi_change = cat_changes.get("decentralized-finance-defi",
                          cat_changes.get("decentralized finance (defi)", None))
            if defi_change is not None and defi_change < -8:
                events.append(MarketEvent(
                    source=EventSource.COINGECKO_CATEGORIES,
                    sector="credit",
                    event_type="defi_credit_stress",
                    weight=round(max(-0.55, -0.30 - abs(defi_change) / 40), 3),
                    decay_halflife_hours=36.0,
                    description=(
                        f"DeFi sector down {defi_change:.1f}% in 24h — "
                        "credit stress / liquidity shock proxy"
                    ),
                    raw={"defi_change_24h": defi_change},
                ))

            # ── Layer-1: infrastructure selloff ────────────────────────
            l1_change = cat_changes.get("layer-1",
                        cat_changes.get("layer 1 (l1)", None))
            if l1_change is not None and l1_change < -10:
                events.append(MarketEvent(
                    source=EventSource.COINGECKO_CATEGORIES,
                    sector="crypto",
                    event_type="l1_infrastructure_selloff",
                    weight=round(max(-0.65, -0.35 - abs(l1_change) / 40), 3),
                    decay_halflife_hours=36.0,
                    description=(
                        f"Layer-1 category down {l1_change:.1f}% in 24h — "
                        "infrastructure selloff, crypto index bearish"
                    ),
                    raw={"l1_change_24h": l1_change},
                ))

            # ── Meme coins pumping while DeFi/L1 flat/negative ────────
            meme_change: Optional[float] = None
            for key in ["meme", "meme-token", "dog-themed-coins"]:
                if key in cat_changes:
                    meme_change = cat_changes[key]
                    break
            if (meme_change is not None and meme_change > 20
                    and (defi_change or 0) < 2):
                events.append(MarketEvent(
                    source=EventSource.COINGECKO_CATEGORIES,
                    sector="crypto",
                    event_type="meme_vs_fundamentals_divergence",
                    weight=0.45,
                    decay_halflife_hours=12.0,
                    description=(
                        f"Meme coins +{meme_change:.0f}% while DeFi~flat — "
                        "unhealthy speculative cycle, distribution likely"
                    ),
                    raw={"meme_change_24h": meme_change, "defi_change_24h": defi_change},
                ))

            # ── RWA (Real World Assets): institutional interest ────────
            rwa_change = cat_changes.get("real-world-assets",
                         cat_changes.get("real world assets (rwa)", None))
            if rwa_change is not None and rwa_change > 5:
                events.append(MarketEvent(
                    source=EventSource.COINGECKO_CATEGORIES,
                    sector="crypto",
                    event_type="rwa_institutional_inflow",
                    weight=-0.20,
                    decay_halflife_hours=48.0,
                    description=(
                        f"RWA category +{rwa_change:.1f}% in 24h — "
                        "institutional crypto adoption signal (mild bullish)"
                    ),
                    raw={"rwa_change_24h": rwa_change},
                ))

            # ── Layer-2: scaling narrative ─────────────────────────────
            l2_change = cat_changes.get("layer-2",
                        cat_changes.get("layer 2 (l2)", None))
            if l2_change is not None and l2_change < -12:
                events.append(MarketEvent(
                    source=EventSource.COINGECKO_CATEGORIES,
                    sector="crypto",
                    event_type="l2_scaling_selloff",
                    weight=round(max(-0.45, -0.25 - abs(l2_change) / 60), 3),
                    decay_halflife_hours=24.0,
                    description=(
                        f"Layer-2 category down {l2_change:.1f}% in 24h — "
                        "scaling ecosystem under pressure"
                    ),
                    raw={"l2_change_24h": l2_change},
                ))

        except Exception as exc:
            logger.debug("CoinGecko categories ingest error: %s", exc)
        return events

    # ════════════════════════════════════════════════════════════════
    # NCL signals + existing UW / FRED methods below
    # ════════════════════════════════════════════════════════════════

    async def ingest_ncl_signals(self, ncl_path: Path) -> List[MarketEvent]:
        """Read NCL market_intelligence.json → MarketEvents."""
        events: List[MarketEvent] = []
        intel_file = ncl_path / "data" / "market_intelligence.json"
        if not intel_file.exists():
            return events
        try:
            data = json.loads(intel_file.read_text())
            signals = data.get("signals", [])
            for sig in signals:
                sector = sig.get("sector", "index")
                direction = sig.get("direction", "neutral")
                confidence = float(sig.get("confidence", 0.5))
                weight = -confidence if direction == "bearish" else +confidence
                events.append(MarketEvent(
                    source=EventSource.NCL_SIGNAL,
                    sector=sector,
                    event_type=f"ncl_{direction}",
                    weight=round(weight * 0.8, 3),  # slight discount vs direct market data
                    decay_halflife_hours=24.0,
                    description=sig.get("description", f"NCL {direction} signal: {sector}"),
                    raw=sig,
                ))
        except Exception as exc:
            logger.debug("NCL signal ingest error: %s", exc)
        return events


# ═══════════════════════════════════════════════════════════════════════════
# SENTIMENT ENGINE — state updates + exponential decay
# ═══════════════════════════════════════════════════════════════════════════

class SentimentEngine:
    """
    Maintains the SentimentState.

    The score for each sector evolves as:
        score(t+dt) = score(t) × decay(dt) + new_event.weight × 100

    Decay uses the per-event half-life:
        decay(dt) = 0.5 ^ (dt_hours / halflife_hours)

    The state's global score is a weighted average of sector scores,
    weighted by the number of recent events per sector.
    """

    EVENT_WINDOW = 20          # rolling events kept per sector
    EVENT_TTL_HOURS = 72.0     # purge events older than this
    GLOBAL_DECAY_HALFLIFE_H = 24.0  # if no new events, global score fades in 24h

    def apply_events(self, events: List[MarketEvent], state: SentimentState) -> int:
        """Apply a batch of events to the state. Returns count applied."""
        applied = 0
        for event in events:
            sent = state.get_sector(event.sector)
            prev_score = sent.score
            # Apply event contribution (weight × 100 → same scale as score)
            sent.score = max(-100.0, min(100.0, sent.score + event.weight * 100.0))
            sent.trend = sent.score - prev_score
            sent.event_count += 1
            sent.events.append(event)
            # Keep rolling window
            if len(sent.events) > self.EVENT_WINDOW:
                sent.events = sent.events[-self.EVENT_WINDOW:]
            sent.last_updated = datetime.now(timezone.utc)
            state.event_count_24h += 1
            applied += 1
        state.version += 1
        return applied

    def decay_all(self, state: SentimentState) -> None:
        """Apply time-based exponential decay to all sector scores."""
        now = datetime.now(timezone.utc)
        for sent in state.sectors.values():
            elapsed_hours = (now - sent.last_updated).total_seconds() / 3600.0
            # Use compound decay of all current events' half-lives (simplified: global constant)
            decay = 0.5 ** (elapsed_hours / self.GLOBAL_DECAY_HALFLIFE_H)
            sent.score *= decay
            # Purge stale events
            cutoff = now - timedelta(hours=self.EVENT_TTL_HOURS)
            sent.events = [e for e in sent.events if e.timestamp > cutoff]
            sent.last_updated = now

    def refresh_global(self, state: SentimentState) -> None:
        """Recompute global score and dominant theme from sector scores."""
        if not state.sectors:
            state.global_score = 0.0
            state.dominant_theme = "neutral"
            return
        # Weight each sector by its recent event count (more activity = more weight)
        total_weight = 0.0
        weighted_sum = 0.0
        for sent in state.sectors.values():
            w = max(1, len(sent.events))
            weighted_sum += sent.score * w
            total_weight += w
        state.global_score = round(weighted_sum / total_weight, 2) if total_weight else 0.0
        # Compute dominant theme
        state.dominant_theme = self._compute_theme(state)

    def _compute_theme(self, state: SentimentState) -> str:
        scores = {k: v.score for k, v in state.sectors.items()}
        if not scores:
            return "neutral"
        most_negative_sector = min(scores, key=scores.get)
        most_negative_score = scores[most_negative_sector]
        regime = state.regime_overlay

        if state.global_score < -60:
            return f"broad_capitulation"
        if state.global_score < -30:
            return f"risk_off_broadening"
        if most_negative_score < -50:
            return f"{most_negative_sector}_stress_acute"
        if most_negative_score < -25:
            return f"{most_negative_sector}_stress_building"
        if "stagflation" in regime:
            return "stagflation_compression"
        if "vol_shock" in regime:
            return "vol_shock_loading"
        if "credit" in regime:
            return "credit_led_breakdown"
        if state.global_score > 30:
            return "risk_on_momentum"
        return "neutral_drift"


# ═══════════════════════════════════════════════════════════════════════════
# POSITION ADVISOR — regime × sentiment → entry/exit params
# ═══════════════════════════════════════════════════════════════════════════

class PositionAdvisor:
    """
    Converts a RegimeState + SentimentState into fully parameterised
    PositionRecommendations with precise entry and exit rules.

    Entry sizing formula:
        base_risk_pct = 5% of available capital
        conviction_mult = {HIGH: 1.0, MEDIUM: 0.7, LOW: 0.4}
        sentiment_mult = abs(sector_score / 100), floored at 0.30
        doctrine_mult = from CrossPillarHub (NORMAL=1.0, CAUTION=0.5, HALT=0.0)
        size_pct = base_risk_pct × conviction_mult × sentiment_mult × doctrine_mult
        clamped to [2%, 15%]
    """

    BASE_RISK_PCT = 0.05

    # Entry urgency thresholds
    _IMMEDIATE_THRESHOLDS = dict(regime_conf=0.70, sentiment=-50, formulas=3, vol_shock=60)
    _NEXT_OPEN_THRESHOLDS = dict(regime_conf=0.50, sentiment=-30, formulas=2, vol_shock=0)
    _SCALE_IN_THRESHOLDS  = dict(regime_conf=0.35, sentiment=-15, formulas=1, vol_shock=0)

    # Conviction tiers
    _CONVICTION_MULT = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}

    # Exit rules -> mapped by conviction
    _TARGET_GAIN = {"HIGH": 0.80, "MEDIUM": 0.60, "LOW": 0.40}
    _TIME_STOP   = {"HIGH": 20,   "MEDIUM": 30,    "LOW": 45}   # days

    def get_recommendations(
        self,
        regime_state: Any,           # RegimeState from regime_engine
        sentiment_state: SentimentState,
        available_capital: float = 920.0,
        doctrine_risk_mult: float = 1.0,
        top_n: int = 5,
    ) -> List[PositionRecommendation]:
        """Generate top-N position recommendations with full entry/exit params."""
        try:
            from strategies.stock_forecaster import StockForecaster, Horizon
            forecaster = StockForecaster()
            forecast = forecaster.forecast(regime_state, Horizon.SHORT, top_n=top_n)
            opportunities = forecast.opportunities
        except Exception as exc:
            logger.warning("StockForecaster unavailable: %s", exc)
            return []

        recs: List[PositionRecommendation] = []
        for opp in opportunities:
            sector = opp.industry.value
            sent = sentiment_state.sectors.get(sector, SectorSentiment(sector=sector))
            conviction = self._conviction_tier(opp.composite_score, abs(sent.score))
            urgency = self._entry_urgency(regime_state, sent.score, conviction)
            size_pct = self._entry_size(conviction, sent.score, doctrine_risk_mult)
            max_usd = round(available_capital * size_pct, 2)
            target_gain = self._TARGET_GAIN[conviction]
            time_stop = self._TIME_STOP[conviction]

            conditions = self._entry_conditions(regime_state, sent, opp)

            recs.append(PositionRecommendation(
                ticker=opp.primary_ticker,
                sector=sector,
                direction="bearish",
                expression=opp.expression.value,
                # Entry
                entry_urgency=urgency,
                entry_size_pct=round(size_pct, 3),
                entry_max_usd=max_usd,
                entry_conditions=conditions,
                structure_hint=opp.structure_hint,
                expiry_min_dte=opp.expiry_range_days[0],
                expiry_max_dte=opp.expiry_range_days[1],
                otm_pct=opp.otm_pct,
                # Exit
                target_gain_pct=target_gain,
                stop_loss_pct=0.50,
                time_stop_days=time_stop,
                exit_on_regime=["risk_on", "uncertain"],
                exit_on_sentiment_reversal=40.0,
                exit_scaling="FULL" if opp.expiry_range_days[1] <= 42 else "HALF_AT_TARGET",
                # Context
                composite_score=opp.composite_score,
                conviction_tier=conviction,
                thesis=opp.thesis,
                formulas_fired=[t.value for t in opp.formula_sources],
                regime=regime_state.primary_regime.value,
                regime_confidence=regime_state.regime_confidence,
                sector_sentiment=round(sent.score, 1),
            ))
        return recs

    def _conviction_tier(self, composite_score: float, sentiment_strength: float) -> str:
        combined = composite_score * 0.6 + sentiment_strength * 0.4
        if combined >= 70:
            return "HIGH"
        if combined >= 45:
            return "MEDIUM"
        return "LOW"

    def _entry_urgency(self, regime_state: Any, sentiment_score: float, conviction: str) -> str:
        t = self._IMMEDIATE_THRESHOLDS
        fired = len(regime_state.armed_formulas)
        vol = regime_state.vol_shock_readiness
        rc = regime_state.regime_confidence
        if (rc >= t["regime_conf"] and sentiment_score <= t["sentiment"]
                and fired >= t["formulas"] and vol >= t["vol_shock"]):
            return "IMMEDIATE"
        t2 = self._NEXT_OPEN_THRESHOLDS
        if rc >= t2["regime_conf"] and sentiment_score <= t2["sentiment"] and fired >= t2["formulas"]:
            return "NEXT_OPEN"
        t3 = self._SCALE_IN_THRESHOLDS
        if rc >= t3["regime_conf"] and sentiment_score <= t3["sentiment"] and fired >= t3["formulas"]:
            return "SCALE_IN"
        return "HOLD"

    def _entry_size(self, conviction: str, sentiment_score: float, doctrine_mult: float) -> float:
        s_mult = max(0.30, abs(sentiment_score) / 100.0)
        c_mult = self._CONVICTION_MULT[conviction]
        raw = self.BASE_RISK_PCT * c_mult * s_mult * doctrine_mult
        return round(max(0.02, min(0.15, raw)), 3)

    def _entry_conditions(self, rs: Any, sent: SectorSentiment, opp: Any) -> List[str]:
        conds = [
            f"Regime: {rs.primary_regime.value.upper()} ({rs.regime_confidence:.0%})",
            f"Vol shock readiness: {rs.vol_shock_readiness:.0f}/100",
            f"Sector sentiment: {sent.score:+.0f}/100",
            f"Formulas fired: {', '.join(f.value for f in rs.armed_formulas[:3])}",
            f"Composite score: {opp.composite_score:.0f}/100",
        ]
        if sent.events:
            latest = sent.events[-1]
            conds.append(f"Latest signal: {latest.description[:60]}")
        return conds


# ═══════════════════════════════════════════════════════════════════════════
# NCL BRIDGE — bidirectional push/pull
# ═══════════════════════════════════════════════════════════════════════════

class NCLBridge:
    """
    Writes AAC intelligence to NCL's data directory (push).
    Reads NCL market intelligence back in (pull).
    Uses file-based sync — works without NCL server running.
    """

    def __init__(self, ncl_data_path: str = ""):
        self._path = Path(ncl_data_path or os.environ.get("NCL_DATA_PATH", ""))

    def push(
        self,
        sentiment_state: SentimentState,
        recommendations: List[PositionRecommendation],
        regime_summary: str = "",
    ) -> bool:
        """Write AAC intelligence JSON to NCL data dir."""
        if not self._path.exists():
            return False
        try:
            out_dir = self._path / "data"
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "source": "AAC",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "regime_summary": regime_summary,
                "global_sentiment": sentiment_state.global_score,
                "dominant_theme": sentiment_state.dominant_theme,
                "vol_shock_readiness": sentiment_state.vol_shock_readiness,
                "top_recommendations": [
                    {
                        "ticker": r.ticker,
                        "sector": r.sector,
                        "urgency": r.entry_urgency,
                        "conviction": r.conviction_tier,
                        "target_gain_pct": r.target_gain_pct,
                        "entry_size_pct": r.entry_size_pct,
                        "thesis": r.thesis[:120],
                    }
                    for r in recommendations[:3]
                ],
                "sector_scores": {
                    k: round(v.score, 1)
                    for k, v in sentiment_state.sectors.items()
                },
            }
            (out_dir / "aac_intelligence.json").write_text(
                json.dumps(payload, indent=2)
            )
            return True
        except Exception as exc:
            logger.warning("NCLBridge push failed: %s", exc)
            return False

    async def pull_events(self, ingestor: EventIngestor) -> List[MarketEvent]:
        """Pull NCL market_intelligence.json → list of MarketEvents."""
        return await ingestor.ingest_ncl_signals(self._path)


# ═══════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

_STATE_FILE = Path("data/intelligence/sentiment_state.json")


def _save_state(state: SentimentState) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state.to_dict(), indent=2))
    except Exception as exc:
        logger.debug("State save failed: %s", exc)


def _load_state() -> SentimentState:
    state = SentimentState()
    for key in SECTOR_KEYS:
        state.sectors[key] = SectorSentiment(sector=key)
    if _STATE_FILE.exists():
        try:
            raw = json.loads(_STATE_FILE.read_text())
            state.global_score = raw.get("global_score", 0.0)
            state.regime_overlay = raw.get("regime_overlay", "uncertain")
            state.vol_shock_readiness = raw.get("vol_shock_readiness", 0.0)
            state.dominant_theme = raw.get("dominant_theme", "neutral")
            state.event_count_24h = raw.get("event_count_24h", 0)
            state.version = raw.get("version", 0)
            logger.info("Loaded sentiment state v%d from disk", state.version)
        except Exception as exc:
            logger.warning("State load failed (using fresh): %s", exc)
    return state


# ═══════════════════════════════════════════════════════════════════════════
# MARKET INTELLIGENCE MODEL — main 24/7 orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class MarketIntelligenceModel:
    """
    24/7 autonomous model that maintains evolving market sentiment by ingesting
    current events and tying them to macro regime state for position recommendations.

    Cadenced loops:
        TICK      (5s)   — vol shock readiness check, heartbeat emit
        INTRADAY  (1h)   — full event ingest → sentiment update → NCL push/pull
        DAILY     (24h)  — FRED macro refresh, regime re-evaluate, recs refresh
        WEEKLY    (7d)   — macro thesis / long-range review

    Usage:
        model = MarketIntelligenceModel()
        asyncio.run(model.run())          # 24/7
        # or one-shot from existing loop:
        await model.intraday_cycle()
        recs = model.get_recommendations()
    """

    def __init__(self, available_capital: float = 920.0, doctrine_risk_mult: float = 1.0):
        self._capital = available_capital
        self._doctrine_mult = doctrine_risk_mult
        self._running = False

        self._ingestor = EventIngestor()
        self._sentiment_engine = SentimentEngine()
        self._advisor = PositionAdvisor()
        self._ncl_bridge = NCLBridge()
        self._state = _load_state()

        # Cross-pillar hub: NCC governance + NCL BRAIN + relay
        try:
            from integrations.cross_pillar_hub import get_cross_pillar_hub
            self._pillar_hub: Any = get_cross_pillar_hub()
        except Exception:
            self._pillar_hub = None

        self._current_regime: Optional[Any] = None      # RegimeState
        self._recommendations: List[PositionRecommendation] = []

        # Cadence tracking
        self._last_intraday = datetime.min.replace(tzinfo=timezone.utc)
        self._last_daily    = datetime.min.replace(tzinfo=timezone.utc)
        self._last_weekly   = datetime.min.replace(tzinfo=timezone.utc)

        logger.info("MarketIntelligenceModel initialised — capital $%.0f", available_capital)

    # ── Public API ────────────────────────────────────────────────────────

    def get_recommendations(self) -> List[PositionRecommendation]:
        return list(self._recommendations)

    def get_sentiment_state(self) -> Dict[str, Any]:
        return self._state.to_dict()

    def get_current_regime(self) -> Optional[Any]:
        return self._current_regime

    def update_capital(self, capital: float) -> None:
        self._capital = capital

    def update_doctrine_mult(self, mult: float) -> None:
        self._doctrine_mult = mult

    # ── Main loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Run the 24/7 intelligence loop. Call from asyncio.run() or as a task."""
        self._running = True
        logger.info("MarketIntelligenceModel loop started")
        # Immediate first pass
        await self.intraday_cycle()
        await self.daily_cycle()

        while self._running:
            now = datetime.now(timezone.utc)
            try:
                # ── TICK: every 5 seconds ──────────────────────────────
                await self._tick_cycle()

                # ── INTRADAY: every 1 hour ─────────────────────────────
                if (now - self._last_intraday).total_seconds() >= TimeHorizon.INTRADAY.value:
                    await self.intraday_cycle()
                    self._last_intraday = now

                # ── DAILY: every 24 hours ──────────────────────────────
                if (now - self._last_daily).total_seconds() >= TimeHorizon.DAILY.value:
                    await self.daily_cycle()
                    self._last_daily = now

                # ── WEEKLY: every 7 days ───────────────────────────────
                if (now - self._last_weekly).total_seconds() >= TimeHorizon.WEEKLY.value:
                    await self._weekly_cycle()
                    self._last_weekly = now

            except Exception as exc:
                logger.error("Intelligence loop error: %s", exc)

            await asyncio.sleep(TimeHorizon.TICK.value)

    async def stop(self) -> None:
        self._running = False

    # ── Cadenced cycles ───────────────────────────────────────────────────

    async def _tick_cycle(self) -> None:
        """Every 5 seconds — fast vol/spread heartbeat only (no I/O)."""
        if self._current_regime:
            vol = self._current_regime.vol_shock_readiness
            self._state.vol_shock_readiness = vol
            if vol >= 80:
                logger.warning("⚡ VOL SHOCK IMMINENT — readiness=%d/100", vol)
            elif vol >= 60:
                logger.debug("⚡ Vol shock ARMED — readiness=%d/100", vol)

    async def intraday_cycle(self) -> None:
        """
        Every 1 hour — ingest all event sources, update sentiment,
        push to NCL, pull NCL signals back in.

        Sources run in parallel (14 concurrent):
            Unusual Whales (flow, dark pool, headlines)
            CBOE put/call ratio    — derivatives intelligence
            Fear & Greed index     — crypto + stock sentiment
            CoinGlass              — crypto funding rates / liquidations
            Crypto on-chain        — CoinGecko global market data
            CoinGecko trending     — FOMO / meme bubble detection
            Stocktwits sentiment   — retail social momentum
            Reddit WSB             — Tradestie options sentiment
            Finnhub insiders       — cluster insider selling
            Finnhub analyst trends — recommendation drift
            Finnhub calendar       — FOMC / CPI / macro event pre-position
            NCL BRAIN bridge       — bidirectional cross-pillar signals

        After ingest: pushes full AAC intelligence (sector scores, crypto
        signals, recommendations, regime) to NCL BRAIN and NCC relay.
        """
        logger.info("[INTRADAY] Starting intelligence cycle — 14 sources")
        try:
            # 1. Decay existing scores before applying new events
            self._sentiment_engine.decay_all(self._state)

            # 2. Ingest all available sources in parallel
            results = await asyncio.gather(
                # ── Unusual Whales ──────────────────────────────────────────
                self._ingestor.ingest_uw_flow(),
                self._ingestor.ingest_uw_darkpool(),
                self._ingestor.ingest_uw_headlines(),
                # ── Derivatives / Fear ───────────────────────────────────────
                self._ingestor.ingest_cboe_put_call(),
                self._ingestor.ingest_fear_greed(),
                # ── Crypto derivatives + on-chain ────────────────────────────
                self._ingestor.ingest_coinglass(),
                self._ingestor.ingest_crypto_onchain(),
                self._ingestor.ingest_coingecko_trending(),
                # ── Social / retail sentiment ────────────────────────────────
                self._ingestor.ingest_stocktwits_sentiment(),
                self._ingestor.ingest_reddit_sentiment(),
                # ── Finnhub institutional intelligence ───────────────────────
                self._ingestor.ingest_finnhub_insiders(),
                self._ingestor.ingest_finnhub_analyst_trends(),
                self._ingestor.ingest_finnhub_economic_calendar(),
                # ── NCL cross-pillar ─────────────────────────────────────────
                self._ncl_bridge.pull_events(self._ingestor),
                return_exceptions=True,
            )
            all_events: List[MarketEvent] = []
            failed_sources = 0
            for r in results:
                if isinstance(r, list):
                    all_events.extend(r)
                elif isinstance(r, Exception):
                    failed_sources += 1
                    logger.debug("[INTRADAY] Source error: %s", r)

            # 3. Apply events to sentiment state
            n = self._sentiment_engine.apply_events(all_events, self._state)
            self._sentiment_engine.refresh_global(self._state)

            logger.info(
                "[INTRADAY] Applied %d events from %d sources (%d failed) | "
                "global_score=%.1f | theme=%s",
                n,
                len(results) - failed_sources,
                failed_sources,
                self._state.global_score,
                self._state.dominant_theme,
            )

            # 4. Refresh recommendations if we have a valid regime
            if self._current_regime:
                self._recommendations = self._advisor.get_recommendations(
                    self._current_regime,
                    self._state,
                    available_capital=self._capital,
                    doctrine_risk_mult=self._doctrine_mult,
                )
                # 5. Push to NCL via legacy file bridge
                self._ncl_bridge.push(
                    self._state,
                    self._recommendations,
                    regime_summary=self._current_regime.summary if hasattr(self._current_regime, "summary") else "",
                )
                self._state.last_ncl_sync = datetime.now(timezone.utc)

            # 6. Cross-pillar: push full AAC topics to NCL BRAIN + NCC relay
            intel_payload = self._build_aac_intelligence_payload()
            if self._pillar_hub is not None:
                await self._pillar_hub.push_intelligence_to_ncl(intel_payload)
                # Check NCC governance → may update doctrine_risk_mult
                await self._sync_ncc_governance()
            await self._push_to_ncc_relay(intel_payload)

            # 7. Persist state
            self._state.last_full_refresh = datetime.now(timezone.utc)
            _save_state(self._state)

        except Exception as exc:
            logger.error("[INTRADAY] Cycle error: %s", exc)

    async def daily_cycle(self) -> None:
        """
        Every 24 hours — full FRED macro refresh + market breadth +
        Google Trends + regime re-evaluate + recommendations rebuild.

        Additional daily sources (slower / rate-limited):
            FRED macro data        — HY spread, VIX, oil
            Polygon market breadth — % advancing / declining
            Google Trends spike    — fear keyword search interest
            CoinGecko OHLC         — RSI, doji, bearish engulfing, SMA crossover
            CoinGecko markets      — top-100 breadth, ATH zones, volume anomalies
            CoinGecko sentiment    — community votes, dev activity, FDV ratio
            CoinGecko categories   — DeFi/L1/meme sector rotation
        """
        logger.info("[DAILY] Starting macro + regime refresh")
        try:
            # 0. Check NCC governance before heavy work
            if self._pillar_hub is not None:
                await self._sync_ncc_governance()

            # 1. Macro + daily sources in parallel
            daily_results = await asyncio.gather(
                self._ingestor.ingest_fred_data(),
                self._ingestor.ingest_polygon_breadth(),
                self._ingestor.ingest_google_trends(),
                self._ingestor.ingest_coingecko_ohlc(),
                self._ingestor.ingest_coingecko_markets(),
                self._ingestor.ingest_coingecko_coin_sentiment(),
                self._ingestor.ingest_coingecko_categories(),
                return_exceptions=True,
            )
            daily_events: List[MarketEvent] = []
            for r in daily_results:
                if isinstance(r, list):
                    daily_events.extend(r)
            if daily_events:
                self._sentiment_engine.apply_events(daily_events, self._state)
                self._sentiment_engine.refresh_global(self._state)

            # 2. Re-evaluate regime from env vars / latest FRED data
            await self._refresh_regime()

            # 3. Full recommendation rebuild
            if self._current_regime:
                self._recommendations = self._advisor.get_recommendations(
                    self._current_regime,
                    self._state,
                    available_capital=self._capital,
                    doctrine_risk_mult=self._doctrine_mult,
                )
                logger.info(
                    "[DAILY] Regime=%s | recs=%d | theme=%s | vol_shock=%.0f",
                    self._current_regime.primary_regime.value,
                    len(self._recommendations),
                    self._state.dominant_theme,
                    self._state.vol_shock_readiness,
                )

            # 4. Persist
            _save_state(self._state)

        except Exception as exc:
            logger.error("[DAILY] Cycle error: %s", exc)

    async def _weekly_cycle(self) -> None:
        """
        Every 7 days — macro thesis review, full NCL deep sync,
        long-range horizon recommendations.
        """
        logger.info("[WEEKLY] Starting macro thesis + deep NCL sync")
        try:
            from strategies.stock_forecaster import StockForecaster, Horizon
            if self._current_regime:
                fc = StockForecaster()
                medium_plan = fc.forecast(self._current_regime, Horizon.MEDIUM, top_n=5)
                logger.info(
                    "[WEEKLY] Medium horizon top-3: %s",
                    [o.primary_ticker for o in medium_plan.top_3],
                )
        except Exception as exc:
            logger.debug("[WEEKLY] StockForecaster unavailable: %s", exc)

    # ── NCC + NCL BRAIN integration ───────────────────────────────────────

    def _build_aac_intelligence_payload(self) -> Dict[str, Any]:
        """
        Build the full AAC intelligence topic payload sent to NCL BRAIN
        and NCC relay.  Contains all AAC-specific context:
            - Crypto sector score + recent CoinGecko signal events
            - Regime overlay + vol shock readiness
            - All 11 sector sentiment scores
            - Top position recommendations (up to 5)
            - Capital available + current doctrine risk multiplier
            - Pillar identity (AAC / BANK)
        """
        regime_val = "unknown"
        regime_conf = 0.0
        if self._current_regime:
            try:
                regime_val = self._current_regime.primary_regime.value
                regime_conf = float(self._current_regime.regime_confidence)
            except Exception:
                pass

        # Collect latest crypto-specific event signals for NCL context
        crypto_signals: List[Dict[str, Any]] = []
        crypto_sector = self._state.sectors.get("crypto")
        if crypto_sector:
            for evt in crypto_sector.events[-10:]:
                crypto_signals.append({
                    "source": evt.source.value,
                    "type": evt.event_type,
                    "weight": evt.weight,
                    "description": evt.description,
                    "ticker": evt.ticker,
                    "ts": evt.timestamp.isoformat(),
                })

        return {
            # ── AAC pillar identity ──────────────────────────────────
            "pillar": "AAC",
            "pillar_role": "BANK",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # ── Core market sentiment ────────────────────────────────
            "global_sentiment": round(self._state.global_score, 2),
            "dominant_theme": self._state.dominant_theme,
            "regime_overlay": regime_val,
            "regime_confidence": round(regime_conf, 3),
            "vol_shock_readiness": round(self._state.vol_shock_readiness, 1),
            "event_count_24h": self._state.event_count_24h,
            # ── Sector scores (all 11) ───────────────────────────────
            "sector_scores": {
                k: round(v.score, 1)
                for k, v in self._state.sectors.items()
            },
            # ── Crypto deep intelligence (CoinGecko-sourced) ─────────
            "crypto_sector_score": round(
                (crypto_sector.score if crypto_sector else 0.0), 1
            ),
            "crypto_recent_signals": crypto_signals,
            # ── Capital & governance ─────────────────────────────────
            "capital_available_usd": round(self._capital, 2),
            "doctrine_risk_mult": self._doctrine_mult,
            "pillar_doctrine_mode": (
                self._pillar_hub.state.doctrine_mode
                if self._pillar_hub is not None else "UNKNOWN"
            ),
            # ── Top recommendations (NCL/NCC can act on these) ────────
            "top_recommendations": [
                {
                    "ticker": r.ticker,
                    "sector": r.sector,
                    "direction": r.direction,
                    "urgency": r.entry_urgency,
                    "conviction": r.conviction_tier,
                    "entry_size_pct": r.entry_size_pct,
                    "target_gain_pct": r.target_gain_pct,
                    "stop_loss_pct": r.stop_loss_pct,
                    "thesis": r.thesis[:160],
                }
                for r in self._recommendations[:5]
            ],
        }

    async def _push_to_ncc_relay(self, payload: Dict[str, Any]) -> bool:
        """
        POST AAC intelligence to the NCC relay server (default :8787).
        Topic: aac/intelligence — consumed by NCC to update BANK state.
        Falls back silently if relay is offline.
        """
        relay_url = os.environ.get("NCC_RELAY_URL", "http://127.0.0.1:8787")
        ncc_token = os.environ.get("NCC_AUTH_TOKEN", "")
        try:
            import urllib.request
            body = json.dumps({**payload, "relay_topic": "aac/intelligence"}).encode()
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "AAC/1.0",
            }
            if ncc_token:
                headers["Authorization"] = f"Bearer {ncc_token}"
            req = urllib.request.Request(
                f"{relay_url}/relay/publish",
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                ok = resp.status < 300
            logger.debug("[RELAY] AAC→NCC push %s (%d recs)",
                         "OK" if ok else "FAILED", len(self._recommendations))
            return ok
        except Exception as exc:
            logger.debug("[RELAY] NCC relay offline or unreachable: %s", exc)
            return False

    async def _sync_ncc_governance(self) -> None:
        """
        Pull NCC governance directive and update doctrine_risk_mult.
        Modes: NORMAL→1.0, CAUTION→0.5, SAFE_MODE→0.0, HALT→0.0
        A HALT directive stops recommendation generation entirely.
        """
        if self._pillar_hub is None:
            return
        try:
            await self._pillar_hub.check_ncc_governance()
            new_mult = self._pillar_hub.get_risk_multiplier()
            mode = self._pillar_hub.state.doctrine_mode
            if abs(new_mult - self._doctrine_mult) > 0.01:
                logger.warning(
                    "[NCC] Doctrine mode change: mult %.1f→%.1f  mode=%s",
                    self._doctrine_mult, new_mult, mode,
                )
                self._doctrine_mult = new_mult
                self._state.regime_overlay = (
                    f"{self._state.regime_overlay}|NCC:{mode}"
                    if mode != "NORMAL" else self._state.regime_overlay
                )
        except Exception as exc:
            logger.debug("[NCC] Governance sync failed: %s", exc)

    async def _refresh_regime(self) -> None:
        """Build a MacroSnapshot from env vars and re-run RegimeEngine."""
        try:
            from strategies.regime_engine import RegimeEngine, MacroSnapshot

            def _f(key: str, default: Optional[float] = None) -> Optional[float]:
                val = os.environ.get(key)
                try:
                    return float(val) if val else default
                except ValueError:
                    return default

            def _b(key: str) -> bool:
                return os.environ.get(key, "").lower() in ("1", "true", "yes")

            snap = MacroSnapshot(
                vix=_f("MONITOR_VIX"),
                hy_spread_bps=_f("MONITOR_HY_SPREAD"),
                oil_price=_f("MONITOR_OIL"),
                core_pce=_f("MONITOR_PCE"),
                gdp_growth=_f("MONITOR_GDP"),
                yield_curve_10_2=_f("MONITOR_YIELD_CURVE"),
                private_credit_redemption_pct=_f("MONITOR_PRIV_CREDIT"),
                hyg_return_1d=_f("MONITOR_HYG_RET"),
                spy_return_1d=_f("MONITOR_SPY_RET"),
                kre_return_1d=_f("MONITOR_KRE_RET"),
                airlines_return_1d=_f("MONITOR_JETS_RET"),
                war_active=_b("MONITOR_WAR"),
                hormuz_blocked=_b("MONITOR_HORMUZ"),
            )
            self._current_regime = RegimeEngine().evaluate(snap)
            self._state.regime_overlay = self._current_regime.primary_regime.value
            self._state.vol_shock_readiness = self._current_regime.vol_shock_readiness
            logger.debug(
                "_refresh_regime: %s (%.0f%%) vol_shock=%.0f",
                self._current_regime.primary_regime.value,
                self._current_regime.regime_confidence * 100,
                self._current_regime.vol_shock_readiness,
            )
        except Exception as exc:
            logger.error("_refresh_regime failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ap = argparse.ArgumentParser(description="Market Intelligence Model — 24/7 sentiment engine")
    ap.add_argument("--balance", type=float, default=920.0, help="Available capital in USD")
    ap.add_argument("--once", action="store_true", help="Run one intraday+daily cycle then print and exit")
    ap.add_argument("--loop", action="store_true", help="Run continuous 24/7 loop (Ctrl+C to stop)")
    args = ap.parse_args()

    async def _main():
        model = MarketIntelligenceModel(available_capital=args.balance)
        if args.once or not args.loop:
            await model.daily_cycle()
            await model.intraday_cycle()
            print("\n" + "═" * 60)
            print("  MARKET INTELLIGENCE MODEL — CURRENT STATE")
            print("═" * 60)
            st = model.get_sentiment_state()
            print(f"  Global sentiment : {st['global_score']:+.1f}/100")
            print(f"  Dominant theme   : {st['dominant_theme']}")
            print(f"  Regime           : {st['regime_overlay'].upper()}")
            print(f"  Vol shock        : {st['vol_shock_readiness']:.0f}/100")
            print(f"  Events (24h)     : {st['event_count_24h']}")
            print()
            print("  TOP SECTOR SCORES:")
            for sector, data in sorted(st["sectors"].items(), key=lambda x: x[1]["score"]):
                bar = "█" * max(0, int(abs(data["score"]) / 10))
                sign = "▼" if data["score"] < 0 else "▲"
                print(f"    {sector:16s}  {sign} {abs(data['score']):5.1f}  {bar}")
            print()
            recs = model.get_recommendations()
            if recs:
                print(f"  POSITION RECOMMENDATIONS ({len(recs)} generated):")
                for r in recs:
                    print(r.print_rec())
            else:
                print("  No recommendations (insufficient regime/sentiment data)")
        else:
            await model.run()

    asyncio.run(_main())
