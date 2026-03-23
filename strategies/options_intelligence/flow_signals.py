"""
Flow Signal Engine — Unusual Whales Flow → Entry Signals
=========================================================
Converts raw UW options flow data into conviction multipliers and
entry triggers for the put strategy pipeline.

Signal taxonomy:
    SWEEP     — Large aggressive sweep across exchanges (highest conviction)
    BLOCK     — Block trade at single venue ($500K+ premium)
    UNUSUAL   — Volume >> open interest (accumulation signal)
    DARK_POOL — Off-exchange print with directional bias
    CONGRESS  — Congressional trade disclosure (insider signal)

Conviction flow:
    1. Ingest raw UW flow + dark pool + congress
    2. Classify each entry by signal type & directional bias
    3. Aggregate per-ticker: flow velocity, net premium, put/call tilt
    4. Produce FlowConviction (0.0-1.0) per ticker
    5. Emit FlowEntry triggers when conviction crosses threshold
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

class FlowType(Enum):
    SWEEP = "sweep"
    BLOCK = "block"
    UNUSUAL_VOLUME = "unusual_volume"
    DARK_POOL = "dark_pool"
    CONGRESS = "congress"


class FlowDirection(Enum):
    BEARISH = "bearish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"


@dataclass
class FlowConviction:
    """Per-ticker aggregated flow conviction."""
    ticker: str
    conviction: float           # 0.0 to 1.0
    direction: FlowDirection
    put_premium: float          # Total bearish premium
    call_premium: float         # Total bullish premium
    put_call_ratio: float       # Ticker-level P/C ratio
    sweep_count: int            # Number of aggressive sweeps
    block_count: int            # Number of blocks
    dark_pool_notional: float   # Dark pool $ volume
    congress_bearish: bool      # Congressional sell detected
    signal_count: int           # Total flow entries contributing
    flow_velocity: float        # Signals per hour (recency-weighted)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_actionable(self) -> bool:
        """Conviction high enough to influence trade decisions."""
        return self.conviction >= 0.4

    @property
    def is_strong(self) -> bool:
        """Strong enough to trigger a new entry."""
        return self.conviction >= 0.7


@dataclass
class FlowEntry:
    """Entry trigger emitted when flow conviction crosses threshold."""
    ticker: str
    direction: FlowDirection
    conviction: float
    reason: str
    suggested_delta: float      # Recommended delta based on flow
    urgency: str                # "immediate", "next_session", "watch"
    premium_context: str        # Description of flow activity
    timestamp: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════════════════
# FLOW SIGNAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class FlowSignalEngine:
    """
    Converts raw Unusual Whales data into actionable flow signals.

    Usage:
        engine = FlowSignalEngine()
        convictions = await engine.analyze_flow(uw_service)
        entries = engine.check_entry_triggers(convictions)
    """

    # Weight each signal type for conviction scoring
    SIGNAL_WEIGHTS: Dict[FlowType, float] = {
        FlowType.SWEEP: 1.0,           # Highest — aggressive multi-exchange
        FlowType.BLOCK: 0.8,           # Large single-venue
        FlowType.UNUSUAL_VOLUME: 0.6,  # Volume >> OI
        FlowType.DARK_POOL: 0.5,       # Off-exchange (may be hedging)
        FlowType.CONGRESS: 0.7,        # Insider knowledge edge
    }

    # Minimum flow thresholds
    MIN_PREMIUM_SWEEP = 200_000       # $200K for sweep classification
    MIN_PREMIUM_BLOCK = 500_000       # $500K for block classification
    MIN_VOL_OI_RATIO = 3.0            # Volume/OI >= 3 for unusual volume
    MIN_DARK_POOL_NOTIONAL = 1_000_000  # $1M dark pool print

    # Conviction thresholds
    CONVICTION_ENTRY_THRESHOLD = 0.6   # Minimum to suggest entry
    CONVICTION_STRONG_THRESHOLD = 0.8  # Strong enough for immediate action

    # Time decay for flow signals (older signals matter less)
    FLOW_HALFLIFE_HOURS = 4.0

    def __init__(
        self,
        entry_threshold: float = 0.6,
        lookback_hours: float = 24.0,
    ):
        self.entry_threshold = entry_threshold
        self.lookback_hours = lookback_hours
        self._flow_cache: Dict[str, List[Dict[str, Any]]] = {}

    async def analyze_flow(
        self,
        uw_client: Any,
        tickers: Optional[List[str]] = None,
    ) -> List[FlowConviction]:
        """
        Pull fresh flow data from UW and produce per-ticker convictions.

        Args:
            uw_client: UnusualWhalesClient instance
            tickers: If provided, only analyze these tickers. Otherwise scan all.

        Returns:
            List of FlowConviction sorted by conviction descending.
        """
        # Fetch flow data
        flow_data = await uw_client.get_flow(limit=100, min_premium=0)
        dark_pool_data = await uw_client.get_dark_pool(limit=50)
        congress_data = await uw_client.get_congress_trades(limit=25)

        # Bucket by ticker
        ticker_flow: Dict[str, List[Dict[str, Any]]] = {}
        ticker_dark: Dict[str, List[Any]] = {}
        ticker_congress: Dict[str, List[Dict[str, Any]]] = {}

        for entry in flow_data:
            t = entry.ticker
            if tickers and t not in tickers:
                continue
            ticker_flow.setdefault(t, []).append({
                "type": self._classify_flow_type(entry),
                "direction": self._classify_direction(entry),
                "premium": entry.premium,
                "volume": entry.volume,
                "open_interest": entry.open_interest,
                "option_type": entry.option_type,
                "strike": entry.strike,
                "expiry": entry.expiry,
                "timestamp": entry.timestamp,
            })

        for dp in dark_pool_data:
            t = dp.ticker
            if tickers and t not in tickers:
                continue
            ticker_dark.setdefault(t, []).append(dp)

        for cong in congress_data:
            t = cong.get("ticker", cong.get("symbol", ""))
            if not t:
                continue
            if tickers and t not in tickers:
                continue
            ticker_congress.setdefault(t, []).append(cong)

        # Build convictions
        all_tickers = set(ticker_flow) | set(ticker_dark) | set(ticker_congress)
        convictions = []

        for ticker in all_tickers:
            flows = ticker_flow.get(ticker, [])
            darks = ticker_dark.get(ticker, [])
            congress = ticker_congress.get(ticker, [])
            conv = self._build_conviction(ticker, flows, darks, congress)
            convictions.append(conv)

        # Sort by conviction descending
        convictions.sort(key=lambda c: c.conviction, reverse=True)
        self._flow_cache = ticker_flow
        return convictions

    def analyze_flow_sync(
        self,
        flow_data: List[Any],
        dark_pool_data: Optional[List[Any]] = None,
        congress_data: Optional[List[Dict[str, Any]]] = None,
        tickers: Optional[List[str]] = None,
    ) -> List[FlowConviction]:
        """
        Synchronous version for use with pre-fetched data.

        Args:
            flow_data: List of OptionsFlow objects
            dark_pool_data: List of DarkPoolTrade objects
            congress_data: List of congress trade dicts
            tickers: Filter to these tickers only
        """
        dark_pool_data = dark_pool_data or []
        congress_data = congress_data or []

        ticker_flow: Dict[str, List[Dict[str, Any]]] = {}
        ticker_dark: Dict[str, List[Any]] = {}
        ticker_congress: Dict[str, List[Dict[str, Any]]] = {}

        for entry in flow_data:
            t = entry.ticker if hasattr(entry, "ticker") else entry.get("ticker", "")
            if tickers and t not in tickers:
                continue
            if hasattr(entry, "premium"):
                ticker_flow.setdefault(t, []).append({
                    "type": self._classify_flow_type(entry),
                    "direction": self._classify_direction(entry),
                    "premium": entry.premium,
                    "volume": entry.volume,
                    "open_interest": entry.open_interest,
                    "option_type": entry.option_type,
                    "strike": entry.strike,
                    "expiry": entry.expiry,
                    "timestamp": getattr(entry, "timestamp", datetime.now()),
                })
            else:
                # Raw dict — classify and normalize to internal format
                ticker_flow.setdefault(t, []).append(
                    self._normalize_dict_entry(entry)
                )

        for dp in dark_pool_data:
            t = dp.ticker if hasattr(dp, "ticker") else dp.get("ticker", "")
            if tickers and t not in tickers:
                continue
            ticker_dark.setdefault(t, []).append(dp)

        for cong in congress_data:
            t = cong.get("ticker", cong.get("symbol", ""))
            if not t:
                continue
            if tickers and t not in tickers:
                continue
            ticker_congress.setdefault(t, []).append(cong)

        all_tickers = set(ticker_flow) | set(ticker_dark) | set(ticker_congress)
        convictions = []

        for ticker in all_tickers:
            flows = ticker_flow.get(ticker, [])
            darks = ticker_dark.get(ticker, [])
            congress = ticker_congress.get(ticker, [])
            conv = self._build_conviction(ticker, flows, darks, congress)
            convictions.append(conv)

        convictions.sort(key=lambda c: c.conviction, reverse=True)
        return convictions

    def check_entry_triggers(
        self,
        convictions: List[FlowConviction],
        existing_positions: Optional[List[str]] = None,
    ) -> List[FlowEntry]:
        """
        Check convictions against thresholds and produce entry triggers.

        Args:
            convictions: List of FlowConviction from analyze_flow()
            existing_positions: Tickers already held (won't double up)
        """
        existing = set(existing_positions or [])
        entries = []

        for conv in convictions:
            if conv.conviction < self.entry_threshold:
                continue

            # Skip if already holding (unless conviction is extreme)
            if conv.ticker in existing and conv.conviction < 0.9:
                continue

            urgency = "watch"
            if conv.conviction >= self.CONVICTION_STRONG_THRESHOLD:
                urgency = "immediate"
            elif conv.conviction >= self.CONVICTION_ENTRY_THRESHOLD:
                urgency = "next_session"

            # Suggested delta: higher conviction → closer to ATM
            if conv.conviction >= 0.85:
                suggested_delta = -0.40  # Aggressive
            elif conv.conviction >= 0.70:
                suggested_delta = -0.35  # Moderate
            else:
                suggested_delta = -0.30  # Conservative

            # Build reason string
            reasons = []
            if conv.sweep_count > 0:
                reasons.append(f"{conv.sweep_count} sweep(s)")
            if conv.block_count > 0:
                reasons.append(f"{conv.block_count} block(s)")
            if conv.put_call_ratio > 2.0:
                reasons.append(f"P/C={conv.put_call_ratio:.1f}")
            if conv.dark_pool_notional > self.MIN_DARK_POOL_NOTIONAL:
                reasons.append(f"DP ${conv.dark_pool_notional / 1e6:.1f}M")
            if conv.congress_bearish:
                reasons.append("Congress sell")

            premium_ctx = (
                f"Put premium ${conv.put_premium / 1000:.0f}K vs "
                f"call ${conv.call_premium / 1000:.0f}K | "
                f"{conv.signal_count} signals | "
                f"velocity {conv.flow_velocity:.1f}/hr"
            )

            entries.append(FlowEntry(
                ticker=conv.ticker,
                direction=conv.direction,
                conviction=conv.conviction,
                reason=" + ".join(reasons) if reasons else "Elevated flow",
                suggested_delta=suggested_delta,
                urgency=urgency,
                premium_context=premium_ctx,
            ))

        return entries

    def get_conviction_multiplier(
        self,
        ticker: str,
        convictions: List[FlowConviction],
    ) -> float:
        """
        Get conviction multiplier for use in position sizing.
        Returns 1.0 (no adjustment) to 2.0 (max boost).
        """
        for conv in convictions:
            if conv.ticker == ticker and conv.is_actionable:
                if conv.direction == FlowDirection.BEARISH:
                    # Bearish flow boosts put conviction
                    return 1.0 + min(1.0, conv.conviction)
                elif conv.direction == FlowDirection.BULLISH:
                    # Bullish flow reduces put conviction
                    return max(0.5, 1.0 - conv.conviction * 0.5)
        return 1.0  # No flow data — neutral

    # ═══════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_timestamp(ts: Any) -> datetime:
        """Parse a timestamp value that may be a string, datetime, or missing."""
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                return datetime.now()
        return datetime.now()

    def _normalize_dict_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Classify and normalize a raw dict flow entry to internal format."""
        premium = entry.get("total_premium", entry.get("premium", 0)) or 0
        sentiment = str(entry.get("sentiment", "")).lower()
        option_type = str(entry.get("option_type", "")).lower()
        volume = entry.get("volume", 0) or 0
        oi = entry.get("open_interest", 1) or 1
        is_sweep = entry.get("is_sweep", False)
        is_block = entry.get("is_block", False)

        # Classify type
        if is_sweep or (premium >= self.MIN_PREMIUM_SWEEP and "sweep" in sentiment):
            flow_type = FlowType.SWEEP
        elif is_block or premium >= self.MIN_PREMIUM_BLOCK:
            flow_type = FlowType.BLOCK
        elif oi > 0 and volume / oi >= self.MIN_VOL_OI_RATIO:
            flow_type = FlowType.UNUSUAL_VOLUME
        else:
            flow_type = FlowType.UNUSUAL_VOLUME

        # Classify direction
        if "bearish" in sentiment:
            direction = FlowDirection.BEARISH
        elif "bullish" in sentiment:
            direction = FlowDirection.BULLISH
        elif option_type == "put":
            direction = FlowDirection.BEARISH
        elif option_type == "call":
            direction = FlowDirection.BULLISH
        else:
            direction = FlowDirection.NEUTRAL

        return {
            "type": flow_type,
            "direction": direction,
            "premium": premium,
            "volume": volume,
            "open_interest": oi,
            "option_type": option_type,
            "timestamp": self._parse_timestamp(entry.get("timestamp")),
        }

    def _classify_flow_type(self, entry: Any) -> FlowType:
        """Classify a flow entry by type."""
        premium = getattr(entry, "premium", 0) or 0
        volume = getattr(entry, "volume", 0) or 0
        oi = getattr(entry, "open_interest", 1) or 1
        sentiment = getattr(entry, "sentiment", "")

        if premium >= self.MIN_PREMIUM_SWEEP and "sweep" in str(sentiment).lower():
            return FlowType.SWEEP
        if premium >= self.MIN_PREMIUM_BLOCK:
            return FlowType.BLOCK
        if oi > 0 and volume / oi >= self.MIN_VOL_OI_RATIO:
            return FlowType.UNUSUAL_VOLUME
        return FlowType.UNUSUAL_VOLUME  # Default

    def _classify_direction(self, entry: Any) -> FlowDirection:
        """Classify directional bias from a flow entry."""
        sentiment = str(getattr(entry, "sentiment", "")).lower()
        option_type = str(getattr(entry, "option_type", "")).lower()

        if "bearish" in sentiment:
            return FlowDirection.BEARISH
        if "bullish" in sentiment:
            return FlowDirection.BULLISH

        # Infer from option type if no sentiment tag
        if option_type == "put":
            return FlowDirection.BEARISH
        if option_type == "call":
            return FlowDirection.BULLISH

        return FlowDirection.NEUTRAL

    def _build_conviction(
        self,
        ticker: str,
        flows: List[Dict[str, Any]],
        dark_pools: List[Any],
        congress: List[Dict[str, Any]],
    ) -> FlowConviction:
        """Build a FlowConviction for one ticker from all signal types."""
        now = datetime.now()

        # Aggregate flow signals with time decay
        put_premium = 0.0
        call_premium = 0.0
        sweep_count = 0
        block_count = 0
        weighted_score = 0.0
        max_weight = 0.0

        for f in flows:
            flow_type = f.get("type", FlowType.UNUSUAL_VOLUME)
            direction = f.get("direction", FlowDirection.NEUTRAL)
            premium = f.get("premium", 0) or 0
            ts = f.get("timestamp", now)
            if isinstance(ts, str):
                ts = self._parse_timestamp(ts)

            # Time decay
            age_hours = max(0.01, (now - ts).total_seconds() / 3600)
            if age_hours > self.lookback_hours:
                continue
            decay = 0.5 ** (age_hours / self.FLOW_HALFLIFE_HOURS)

            weight = self.SIGNAL_WEIGHTS.get(flow_type, 0.3) * decay

            if direction == FlowDirection.BEARISH:
                weighted_score += weight
            elif direction == FlowDirection.BULLISH:
                weighted_score -= weight * 0.5  # Bullish reduces bearish conviction

            max_weight += self.SIGNAL_WEIGHTS.get(flow_type, 0.3)

            option_type = f.get("option_type", "")
            if option_type == "put":
                put_premium += premium
            elif option_type == "call":
                call_premium += premium

            if flow_type == FlowType.SWEEP:
                sweep_count += 1
            elif flow_type == FlowType.BLOCK:
                block_count += 1

        # Dark pool contribution
        dp_notional = 0.0
        for dp in dark_pools:
            notional = getattr(dp, "notional", 0) or dp.get("notional", 0) if isinstance(dp, dict) else 0
            dp_notional += notional
        if dp_notional > self.MIN_DARK_POOL_NOTIONAL:
            weighted_score += 0.3
            max_weight += 0.3

        # Congress contribution
        congress_bearish = False
        for cong in congress:
            tx_type = str(cong.get("transaction_type", cong.get("type", ""))).lower()
            if "sale" in tx_type or "sell" in tx_type:
                congress_bearish = True
                weighted_score += 0.5
                max_weight += 0.5
                break

        # Normalize conviction to 0-1
        if max_weight > 0:
            raw_conviction = max(0.0, weighted_score / max_weight)
        else:
            raw_conviction = 0.0
        conviction = min(1.0, raw_conviction)

        # Direction
        if put_premium > call_premium * 1.5:
            direction = FlowDirection.BEARISH
        elif call_premium > put_premium * 1.5:
            direction = FlowDirection.BULLISH
        else:
            direction = FlowDirection.NEUTRAL

        # Put/call ratio
        pcr = put_premium / call_premium if call_premium > 0 else float(bool(put_premium))

        # Flow velocity (signals per hour)
        if flows:
            span_hours = max(1.0, self.lookback_hours)
            velocity = len(flows) / span_hours
        else:
            velocity = 0.0

        return FlowConviction(
            ticker=ticker,
            conviction=conviction,
            direction=direction,
            put_premium=put_premium,
            call_premium=call_premium,
            put_call_ratio=pcr,
            sweep_count=sweep_count,
            block_count=block_count,
            dark_pool_notional=dp_notional,
            congress_bearish=congress_bearish,
            signal_count=len(flows) + len(dark_pools) + len(congress),
            flow_velocity=velocity,
        )
