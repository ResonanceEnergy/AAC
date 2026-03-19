"""
MATRIX MAXIMIZER — Intelligence Feeds
========================================
Aggregates intelligence from multiple AAC pillars:
  - NCL (BRAIN): sector rotation, sentiment shifts
  - StockForecaster: ranked trade opportunities
  - RegimeEngine: formula results, shock readiness
  - News & Sentiment: geopolitical escalation scoring
  - Earnings Calendar: avoid/target earnings
  - Unusual Whales: dark pool, congress, squeeze signals

All intelligence is consumed by the scanner to enrich put scoring.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IntelSignal:
    """Single intelligence signal from any source."""
    source: str          # "ncl", "stock_forecaster", "regime_engine", "news", "unusual_whales"
    signal_type: str     # "sector_rotation", "bearish_thesis", "dark_pool_block", etc.
    ticker: str
    direction: str       # "bearish", "bullish", "neutral"
    strength: float      # 0.0 – 1.0
    thesis: str
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntelBrief:
    """Aggregated intelligence for scanner consumption."""
    signals: List[IntelSignal] = field(default_factory=list)
    ticker_sentiment: Dict[str, float] = field(default_factory=dict)  # -1.0 to 1.0
    sector_rotation: Dict[str, str] = field(default_factory=dict)     # sector → "into"/"out_of"
    earnings_blackout: List[str] = field(default_factory=list)        # tickers near earnings
    dark_pool_activity: Dict[str, float] = field(default_factory=dict)  # ticker → net $
    congress_buys: List[str] = field(default_factory=list)
    congress_sells: List[str] = field(default_factory=list)
    regime_formulas_armed: List[str] = field(default_factory=list)
    vol_shock_readiness: float = 0.0
    timestamp: str = ""

    @property
    def bearish_signals(self) -> List[IntelSignal]:
        return [s for s in self.signals if s.direction == "bearish"]

    @property
    def bullish_signals(self) -> List[IntelSignal]:
        return [s for s in self.signals if s.direction == "bullish"]

    def ticker_bias(self, ticker: str) -> float:
        """Net sentiment for a ticker: -1 (bearish) to +1 (bullish)."""
        return self.ticker_sentiment.get(ticker, 0.0)

    def print_brief(self) -> str:
        lines = [
            f"═══ INTELLIGENCE BRIEF ({self.timestamp}) ═══",
            f"  Signals: {len(self.signals)} total, "
            f"{len(self.bearish_signals)} bear, {len(self.bullish_signals)} bull",
            f"  Vol Shock Readiness: {self.vol_shock_readiness:.0f}%",
            f"  Armed Formulas: {', '.join(self.regime_formulas_armed) or 'none'}",
            f"  Earnings Blackout: {', '.join(self.earnings_blackout) or 'none'}",
        ]
        if self.dark_pool_activity:
            top = sorted(self.dark_pool_activity.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            lines.append(f"  Dark Pool: {', '.join(f'{t}=${v/1e6:.1f}M' for t, v in top)}")
        if self.congress_sells:
            lines.append(f"  Congress Selling: {', '.join(self.congress_sells[:5])}")
        return "\n".join(lines)


class IntelligenceEngine:
    """Central intelligence aggregator for MATRIX MAXIMIZER.

    Pulls from:
        1. NCL intelligence files (data/pillar_state/ncl_intelligence.json)
        2. StockForecaster output (data/stock_forecaster_output.json)
        3. RegimeEngine state (data/regime_state.json)
        4. Unusual Whales flow/dark pool (via DataFeedManager)
        5. Finnhub earnings calendar (via DataFeedManager)
        6. News sentiment (via DataFeedManager)
    """

    def __init__(self, data_feed_manager: Optional[Any] = None) -> None:
        self._feeds = data_feed_manager
        self._ncl_path = Path("data/pillar_state/ncl_intelligence.json")
        self._forecaster_path = Path("data/stock_forecaster_output.json")
        self._regime_path = Path("data/regime_state.json")

    def gather_intel(self, tickers: List[str]) -> IntelBrief:
        """Aggregate intelligence from all sources into a single brief."""
        brief = IntelBrief(timestamp=datetime.utcnow().isoformat())

        # 1. NCL signals
        ncl_signals = self._read_ncl_signals()
        brief.signals.extend(ncl_signals)

        # 2. StockForecaster rankings
        forecaster_signals = self._read_forecaster(tickers)
        brief.signals.extend(forecaster_signals)

        # 3. Regime engine formulas
        regime_signals = self._read_regime()
        brief.signals.extend(regime_signals)
        brief.regime_formulas_armed = [
            s.signal_type for s in regime_signals
            if s.signal_type.startswith("F") and s.strength > 0.5
        ]
        vol_sigs = [s for s in regime_signals if s.signal_type == "vol_shock_readiness"]
        if vol_sigs:
            brief.vol_shock_readiness = vol_sigs[0].strength * 100

        # 4. Unusual Whales flow + dark pool
        if self._feeds:
            self._enrich_from_feeds(brief, tickers)

        # 5. Compute ticker sentiment
        for ticker in tickers:
            ticker_sigs = [s for s in brief.signals if s.ticker == ticker]
            if ticker_sigs:
                bearish = sum(s.strength for s in ticker_sigs if s.direction == "bearish")
                bullish = sum(s.strength for s in ticker_sigs if s.direction == "bullish")
                total = bearish + bullish
                brief.ticker_sentiment[ticker] = (bullish - bearish) / total if total > 0 else 0.0

        return brief

    # ── NCL Intelligence ──────────────────────────────────────────────────

    def _read_ncl_signals(self) -> List[IntelSignal]:
        """Read NCL intelligence feed."""
        signals: List[IntelSignal] = []
        if not self._ncl_path.exists():
            return signals

        try:
            data = json.loads(self._ncl_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return signals

        # Sector rotation
        for sector, direction in data.get("sector_rotation", {}).items():
            signals.append(IntelSignal(
                source="ncl",
                signal_type="sector_rotation",
                ticker=sector,
                direction="bearish" if direction == "out_of" else "bullish",
                strength=0.6,
                thesis=f"NCL: sector rotation {direction} {sector}",
            ))

        # Sentiment shifts
        for entry in data.get("sentiment_shifts", []):
            signals.append(IntelSignal(
                source="ncl",
                signal_type="sentiment_shift",
                ticker=entry.get("ticker", ""),
                direction=entry.get("direction", "neutral"),
                strength=float(entry.get("magnitude", 0.5)),
                thesis=entry.get("reason", ""),
            ))

        # ML predictions
        for entry in data.get("predictions", []):
            signals.append(IntelSignal(
                source="ncl",
                signal_type="ml_prediction",
                ticker=entry.get("ticker", ""),
                direction=entry.get("direction", "neutral"),
                strength=float(entry.get("confidence", 0.5)),
                thesis=entry.get("model", "NCL ML model"),
            ))

        return signals

    # ── StockForecaster ───────────────────────────────────────────────────

    def _read_forecaster(self, tickers: List[str]) -> List[IntelSignal]:
        """Read StockForecaster output file."""
        signals: List[IntelSignal] = []
        if not self._forecaster_path.exists():
            return signals

        try:
            data = json.loads(self._forecaster_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return signals

        for opp in data.get("opportunities", []):
            opp_tickers = opp.get("tickers", [])
            matching = [t for t in opp_tickers if t in tickers]
            if not matching:
                continue

            direction = opp.get("direction", "neutral").lower()
            for ticker in matching:
                signals.append(IntelSignal(
                    source="stock_forecaster",
                    signal_type="trade_opportunity",
                    ticker=ticker,
                    direction=direction,
                    strength=float(opp.get("composite_score", 50)) / 100,
                    thesis=opp.get("thesis", ""),
                    metadata={
                        "expression": opp.get("expression", ""),
                        "horizon": opp.get("horizon", ""),
                        "catalyst": opp.get("catalyst", ""),
                        "rank": opp.get("rank", 0),
                    },
                ))

        return signals

    # ── Regime Engine ─────────────────────────────────────────────────────

    def _read_regime(self) -> List[IntelSignal]:
        """Read regime engine state."""
        signals: List[IntelSignal] = []
        if not self._regime_path.exists():
            return signals

        try:
            data = json.loads(self._regime_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return signals

        # Armed formulas
        for f in data.get("armed_formulas", []):
            signals.append(IntelSignal(
                source="regime_engine",
                signal_type=f,
                ticker="MARKET",
                direction="bearish",
                strength=0.8,
                thesis=f"Regime formula {f} is ARMED",
            ))

        # Vol shock readiness
        vsr = data.get("vol_shock_readiness", 0)
        if vsr > 0:
            signals.append(IntelSignal(
                source="regime_engine",
                signal_type="vol_shock_readiness",
                ticker="MARKET",
                direction="bearish" if vsr > 60 else "neutral",
                strength=vsr / 100,
                thesis=f"Vol shock readiness at {vsr:.0f}%",
            ))

        # Primary regime
        regime = data.get("primary_regime", "")
        if regime:
            bearish_regimes = {"credit_stress", "vol_shock_active", "vol_shock_armed",
                               "liquidity_crunch", "stagflation", "risk_off"}
            signals.append(IntelSignal(
                source="regime_engine",
                signal_type="primary_regime",
                ticker="MARKET",
                direction="bearish" if regime in bearish_regimes else "bullish",
                strength=float(data.get("regime_confidence", 0.5)),
                thesis=f"Primary regime: {regime}",
            ))

        return signals

    # ── Unusual Whales Feed ───────────────────────────────────────────────

    def _enrich_from_feeds(self, brief: IntelBrief, tickers: List[str]) -> None:
        """Enrich brief with Unusual Whales + earnings data."""
        feeds = self._feeds

        # Options flow
        try:
            flow = feeds.get_unusual_flow(min_premium=100_000, limit=100)
            for f in flow:
                if f.ticker in tickers:
                    brief.signals.append(IntelSignal(
                        source="unusual_whales",
                        signal_type="options_flow",
                        ticker=f.ticker,
                        direction=f.sentiment,
                        strength=min(1.0, f.premium / 1_000_000),
                        thesis=f"${f.premium/1000:.0f}K {f.option_type} flow, {f.sentiment}",
                        metadata={"strike": f.strike, "expiry": f.expiry},
                    ))
        except Exception as exc:
            logger.warning("Unusual Whales flow error: %s", exc)

        # Dark pool
        try:
            for ticker in tickers:
                dp = feeds.get_dark_pool(ticker=ticker, limit=20)
                if dp:
                    total = sum(d.notional for d in dp)
                    brief.dark_pool_activity[ticker] = total
        except Exception as exc:
            logger.warning("Dark pool error: %s", exc)

        # Congress trades
        try:
            congress = feeds.get_congress_trades(limit=100)
            for t in congress:
                if t.get("ticker") in tickers:
                    if "purchase" in t.get("type", "").lower():
                        brief.congress_buys.append(t["ticker"])
                    elif "sale" in t.get("type", "").lower():
                        brief.congress_sells.append(t["ticker"])
        except Exception as exc:
            logger.warning("Congress trades error: %s", exc)

        # Earnings blackout
        try:
            brief.earnings_blackout = feeds.get_tickers_near_earnings(tickers, days=5)
        except Exception as exc:
            logger.warning("Earnings calendar error: %s", exc)
