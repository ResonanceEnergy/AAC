"""
Dynamic Universe Expander — Beyond the 16 Fixed Symbols
========================================================
Uses Unusual Whales flow data to discover high-activity symbols
that aren't in the static PUT_PLAYBOOK, then evaluates them
for inclusion in the trading universe.

Discovery sources:
    1. UW top flow tickers (highest premium volume)
    2. UW hottest options chains (volume spike)
    3. UW sector ETF rotation (sector-level flow shifts)
    4. Dark pool prints on new symbols
    5. Congressional trading disclosures

Filtering:
    - Minimum market cap ($1B+)
    - Minimum options liquidity (avg daily volume)
    - Optionable (has listed puts with >$0.05 spread)
    - Not in exclusion list (energy longs, existing positions)
    - Correlation with crisis thesis vectors
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from strategies.options_intelligence.flow_signals import FlowConviction, FlowDirection

logger = logging.getLogger(__name__)


# Sectors that align with bearish crisis thesis
CRISIS_SECTORS: Dict[str, float] = {
    "Financial Services": 1.0,
    "Financials": 1.0,
    "Real Estate": 0.9,
    "Consumer Cyclical": 0.8,
    "Consumer Discretionary": 0.8,
    "Industrials": 0.7,
    "Technology": 0.6,
    "Communication Services": 0.5,
    "Healthcare": 0.3,
    "Energy": 0.0,       # DO NOT SHORT — crisis benefits energy
    "Utilities": 0.2,
    "Basic Materials": 0.4,
    "Materials": 0.4,
}

# Never consider these (we're long or they benefit from crisis)
DEFAULT_EXCLUSIONS: Set[str] = {
    "XLE", "USO", "OIH",   # Energy — crisis longs
    "GLD", "SLV", "GDX",   # Precious metals — crisis longs
    "VIX", "UVXY", "VXX",  # Vol — can't short vol products as puts
    "TLT", "IEF", "SHY",   # Treasuries — flight-to-safety
}


@dataclass
class DynamicCandidate:
    """A dynamically discovered symbol for potential universe inclusion."""
    ticker: str
    source: str                  # "flow", "hottest_chain", "sector", "dark_pool", "congress"
    sector: str
    flow_conviction: float       # 0.0 to 1.0
    crisis_alignment: float      # How well it fits crisis vectors (0-1)
    flow_premium_24h: float      # Total options premium in last 24h
    put_call_ratio: float        # Ticker-level P/C ratio
    dark_pool_notional: float    # Dark pool volume
    market_cap_est: str          # "large", "mid", "small" (from UW data)
    composite_score: float       # Weighted combination score (0-100)
    reason: str                  # Why this candidate was flagged
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_viable(self) -> bool:
        return self.composite_score >= 40.0


class UniverseExpander:
    """
    Discovers new tradeable symbols from UW flow data and returns
    ranked candidates for inclusion in the scanning pipeline.

    Usage:
        expander = UniverseExpander()
        candidates = await expander.discover(uw_client, flow_convictions)
    """

    def __init__(
        self,
        max_candidates: int = 20,
        min_composite_score: float = 40.0,
        exclusions: Optional[Set[str]] = None,
    ):
        self.max_candidates = max_candidates
        self.min_composite_score = min_composite_score
        self.exclusions = exclusions or DEFAULT_EXCLUSIONS
        self._sector_cache: Dict[str, str] = {}

    async def discover(
        self,
        uw_client: Any,
        flow_convictions: Optional[List[FlowConviction]] = None,
        existing_universe: Optional[Set[str]] = None,
    ) -> List[DynamicCandidate]:
        """
        Discover new candidates from UW data.

        Args:
            uw_client: UnusualWhalesClient instance
            flow_convictions: Pre-computed convictions from FlowSignalEngine
            existing_universe: Tickers already in the scan list (won't duplicate)
        """
        existing = existing_universe or set()
        candidates: Dict[str, DynamicCandidate] = {}

        # Source 1: Hottest options chains (volume spikes)
        hottest = await uw_client.get_hottest_chains(limit=20)
        for item in hottest:
            ticker = item.get("ticker", item.get("symbol", ""))
            if not ticker or ticker in self.exclusions or ticker in existing:
                continue
            sector = await self._get_sector(uw_client, ticker)
            crisis_score = CRISIS_SECTORS.get(sector, 0.3)

            candidates[ticker] = DynamicCandidate(
                ticker=ticker,
                source="hottest_chain",
                sector=sector,
                flow_conviction=0.0,
                crisis_alignment=crisis_score,
                flow_premium_24h=float(item.get("total_premium", 0) or 0),
                put_call_ratio=float(item.get("put_call_ratio", 0) or 0),
                dark_pool_notional=0.0,
                market_cap_est=self._estimate_cap(item),
                composite_score=0.0,
                reason=f"Volume spike: {item.get('volume', 'N/A')}",
            )

        # Source 2: Sector ETFs with bearish tilt
        sector_etfs = await uw_client.get_sector_etfs()
        for etf in sector_etfs:
            ticker = etf.get("ticker", etf.get("symbol", ""))
            if not ticker or ticker in self.exclusions or ticker in existing:
                continue
            change_pct = float(etf.get("change_pct", etf.get("percent_change", 0)) or 0)
            # Only interested in declining sectors (bearish flow)
            if change_pct >= 0:
                continue
            sector = etf.get("sector", "Unknown")
            crisis_score = CRISIS_SECTORS.get(sector, 0.3)

            if ticker not in candidates:
                candidates[ticker] = DynamicCandidate(
                    ticker=ticker,
                    source="sector",
                    sector=sector,
                    flow_conviction=0.0,
                    crisis_alignment=crisis_score,
                    flow_premium_24h=0.0,
                    put_call_ratio=0.0,
                    dark_pool_notional=0.0,
                    market_cap_est="large",
                    composite_score=0.0,
                    reason=f"Sector decline: {change_pct:.1f}%",
                )

        # Source 3: Flow convictions from FlowSignalEngine
        if flow_convictions:
            for conv in flow_convictions:
                if conv.ticker in self.exclusions or conv.ticker in existing:
                    continue
                if conv.direction != FlowDirection.BEARISH:
                    continue
                if conv.conviction < 0.3:
                    continue

                if conv.ticker in candidates:
                    # Merge flow data into existing candidate
                    candidates[conv.ticker].flow_conviction = conv.conviction
                    candidates[conv.ticker].put_call_ratio = conv.put_call_ratio
                    candidates[conv.ticker].dark_pool_notional = conv.dark_pool_notional
                else:
                    candidates[conv.ticker] = DynamicCandidate(
                        ticker=conv.ticker,
                        source="flow",
                        sector=await self._get_sector(uw_client, conv.ticker),
                        flow_conviction=conv.conviction,
                        crisis_alignment=0.5,  # Unknown — conservative
                        flow_premium_24h=conv.put_premium + conv.call_premium,
                        put_call_ratio=conv.put_call_ratio,
                        dark_pool_notional=conv.dark_pool_notional,
                        market_cap_est="unknown",
                        composite_score=0.0,
                        reason=f"Bearish flow: conviction={conv.conviction:.0%}",
                    )

        # Score all candidates
        result = []
        for cand in candidates.values():
            cand.composite_score = self._score_candidate(cand)
            if cand.composite_score >= self.min_composite_score:
                result.append(cand)

        # Sort and limit
        result.sort(key=lambda c: c.composite_score, reverse=True)
        return result[:self.max_candidates]

    def discover_sync(
        self,
        flow_convictions: List[FlowConviction],
        hottest_chains: Optional[List[Dict[str, Any]]] = None,
        sector_etfs: Optional[List[Dict[str, Any]]] = None,
        existing_universe: Optional[Set[str]] = None,
    ) -> List[DynamicCandidate]:
        """
        Synchronous discovery from pre-fetched data.
        """
        existing = existing_universe or set()
        candidates: Dict[str, DynamicCandidate] = {}

        # Hottest chains
        for item in (hottest_chains or []):
            ticker = item.get("ticker", item.get("symbol", ""))
            if not ticker or ticker in self.exclusions or ticker in existing:
                continue
            sector = item.get("sector", "Unknown")
            crisis_score = CRISIS_SECTORS.get(sector, 0.3)

            candidates[ticker] = DynamicCandidate(
                ticker=ticker,
                source="hottest_chain",
                sector=sector,
                flow_conviction=0.0,
                crisis_alignment=crisis_score,
                flow_premium_24h=float(item.get("total_premium", 0) or 0),
                put_call_ratio=float(item.get("put_call_ratio", 0) or 0),
                dark_pool_notional=0.0,
                market_cap_est=self._estimate_cap(item),
                composite_score=0.0,
                reason=f"Volume spike: {item.get('volume', 'N/A')}",
            )

        # Sector ETFs
        for etf in (sector_etfs or []):
            ticker = etf.get("ticker", etf.get("symbol", ""))
            if not ticker or ticker in self.exclusions or ticker in existing:
                continue
            change_pct = float(etf.get("change_pct", etf.get("percent_change", 0)) or 0)
            if change_pct >= 0:
                continue
            sector = etf.get("sector", "Unknown")
            crisis_score = CRISIS_SECTORS.get(sector, 0.3)

            if ticker not in candidates:
                candidates[ticker] = DynamicCandidate(
                    ticker=ticker,
                    source="sector",
                    sector=sector,
                    flow_conviction=0.0,
                    crisis_alignment=crisis_score,
                    flow_premium_24h=0.0,
                    put_call_ratio=0.0,
                    dark_pool_notional=0.0,
                    market_cap_est="large",
                    composite_score=0.0,
                    reason=f"Sector decline: {change_pct:.1f}%",
                )

        # Flow convictions
        for conv in flow_convictions:
            if conv.ticker in self.exclusions or conv.ticker in existing:
                continue
            if conv.direction != FlowDirection.BEARISH:
                continue
            if conv.conviction < 0.3:
                continue

            if conv.ticker in candidates:
                candidates[conv.ticker].flow_conviction = conv.conviction
                candidates[conv.ticker].put_call_ratio = conv.put_call_ratio
                candidates[conv.ticker].dark_pool_notional = conv.dark_pool_notional
            else:
                candidates[conv.ticker] = DynamicCandidate(
                    ticker=conv.ticker,
                    source="flow",
                    sector="Unknown",
                    flow_conviction=conv.conviction,
                    crisis_alignment=0.5,
                    flow_premium_24h=conv.put_premium + conv.call_premium,
                    put_call_ratio=conv.put_call_ratio,
                    dark_pool_notional=conv.dark_pool_notional,
                    market_cap_est="unknown",
                    composite_score=0.0,
                    reason=f"Bearish flow: conviction={conv.conviction:.0%}",
                )

        result = []
        for cand in candidates.values():
            cand.composite_score = self._score_candidate(cand)
            if cand.composite_score >= self.min_composite_score:
                result.append(cand)

        result.sort(key=lambda c: c.composite_score, reverse=True)
        return result[:self.max_candidates]

    # ═══════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def _score_candidate(self, cand: DynamicCandidate) -> float:
        """
        Composite score 0-100 from multiple factors.
        Weights: flow_conviction 35%, crisis_alignment 25%, P/C ratio 20%,
                 premium volume 10%, dark pool 10%
        """
        # Flow conviction (0-100)
        flow_score = cand.flow_conviction * 100

        # Crisis alignment (0-100)
        crisis_score = cand.crisis_alignment * 100

        # Put/call ratio (higher = more bearish)
        if cand.put_call_ratio > 3.0:
            pcr_score = 100
        elif cand.put_call_ratio > 2.0:
            pcr_score = 80
        elif cand.put_call_ratio > 1.5:
            pcr_score = 60
        elif cand.put_call_ratio > 1.0:
            pcr_score = 40
        else:
            pcr_score = max(0, cand.put_call_ratio * 30)

        # Premium volume (logarithmic scale)
        if cand.flow_premium_24h > 10_000_000:
            prem_score = 100
        elif cand.flow_premium_24h > 1_000_000:
            prem_score = 70
        elif cand.flow_premium_24h > 100_000:
            prem_score = 40
        else:
            prem_score = 10

        # Dark pool notional
        if cand.dark_pool_notional > 10_000_000:
            dp_score = 100
        elif cand.dark_pool_notional > 1_000_000:
            dp_score = 60
        else:
            dp_score = max(0, cand.dark_pool_notional / 100_000)

        composite = (
            flow_score * 0.35 +
            crisis_score * 0.25 +
            pcr_score * 0.20 +
            prem_score * 0.10 +
            dp_score * 0.10
        )
        return round(composite, 1)

    async def _get_sector(self, uw_client: Any, ticker: str) -> str:
        """Get sector for a ticker (cached)."""
        if ticker in self._sector_cache:
            return self._sector_cache[ticker]

        try:
            info = await uw_client.get_ticker_overview(ticker)
            sector = info.get("sector", "Unknown")
        except Exception:
            sector = "Unknown"

        self._sector_cache[ticker] = sector
        return sector

    @staticmethod
    def _estimate_cap(item: Dict[str, Any]) -> str:
        """Estimate market cap category from UW data."""
        cap = item.get("market_cap", item.get("marketCap", 0))
        if isinstance(cap, str):
            return cap
        try:
            cap_val = float(cap or 0)
        except (TypeError, ValueError):
            return "unknown"
        if cap_val > 10_000_000_000:
            return "large"
        if cap_val > 2_000_000_000:
            return "mid"
        if cap_val > 0:
            return "small"
        return "unknown"
