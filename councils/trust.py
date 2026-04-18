from __future__ import annotations

"""
Trust Score System — quantified reliability for council intelligence.
=====================================================================

Every council signal carries a TrustScore that measures:
  - source_reliability : how trustworthy is the raw data source?
  - data_freshness     : how recent is the data?
  - evidence_volume    : how much evidence backs the signal?
  - cross_source_agreement : do multiple sources agree? (filled downstream)
  - overall            : weighted composite (auto-computed)

Usage:
    from councils.trust import TrustScore, compute_source_trust

    ts = TrustScore(source_reliability=0.7, data_freshness=0.9, evidence_volume=0.5)
    print(ts.overall)  # weighted composite
"""

import time
from dataclasses import dataclass, field
from typing import Any


# ── Weights for composite score ────────────────────────────────────────────
_WEIGHTS = {
    "source_reliability": 0.35,
    "data_freshness": 0.25,
    "evidence_volume": 0.20,
    "cross_source_agreement": 0.20,
}


@dataclass
class TrustScore:
    """Quantified reliability of a council signal.  All values 0.0 – 1.0."""

    source_reliability: float = 0.5
    data_freshness: float = 1.0
    evidence_volume: float = 0.5
    cross_source_agreement: float = 0.5
    overall: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.overall = self._compute_overall()

    def _compute_overall(self) -> float:
        raw = (
            self.source_reliability * _WEIGHTS["source_reliability"]
            + self.data_freshness * _WEIGHTS["data_freshness"]
            + self.evidence_volume * _WEIGHTS["evidence_volume"]
            + self.cross_source_agreement * _WEIGHTS["cross_source_agreement"]
        )
        return round(max(0.0, min(1.0, raw)), 3)

    def recalculate(self) -> None:
        """Recalculate overall after external mutation of component scores."""
        self.overall = self._compute_overall()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_reliability": self.source_reliability,
            "data_freshness": self.data_freshness,
            "evidence_volume": self.evidence_volume,
            "cross_source_agreement": self.cross_source_agreement,
            "overall": self.overall,
            "details": self.details,
        }


# ── Freshness helper ──────────────────────────────────────────────────────

def freshness_score(age_seconds: float, half_life: float = 3600.0) -> float:
    """Exponential decay: 1.0 when fresh, 0.5 at half_life seconds old.

    half_life defaults to 1 hour (3600s).
    """
    if age_seconds <= 0:
        return 1.0
    decay = 0.5 ** (age_seconds / half_life)
    return round(max(0.0, min(1.0, decay)), 3)


# ── Evidence volume helper ────────────────────────────────────────────────

def evidence_score(count: int, target: int = 20) -> float:
    """Saturating curve: approaches 1.0 as count → target.

    At target items: 0.95.  At 1 item: ~0.15.
    """
    if count <= 0:
        return 0.0
    ratio = count / target
    score = 1.0 - (0.5 ** ratio)
    return round(max(0.0, min(1.0, score)), 3)


# ── Source reliability helpers per council ─────────────────────────────────

def youtube_source_trust(
    view_count: int = 0,
    like_count: int = 0,
    has_transcript: bool = False,
    duration_seconds: int = 0,
    channel: str = "",
) -> float:
    """Score 0-1 for a YouTube video's source reliability.

    Factors:
    - view_count: high views → more vetted content
    - like_count: engagement signal
    - has_transcript: transcribed content is verifiable vs not
    - duration: <2min or >3hr are lower value for intel
    - channel: known finance channels get a bonus
    """
    score = 0.3  # base for any real video

    # Views (log scale: 1K=+0.1, 10K=+0.15, 100K=+0.2, 1M=+0.25)
    if view_count > 0:
        import math
        log_views = math.log10(max(view_count, 1))
        score += min(log_views / 24, 0.25)  # cap at 0.25

    # Engagement ratio
    if view_count > 100 and like_count > 0:
        ratio = like_count / view_count
        score += min(ratio * 2, 0.1)

    # Transcript availability — verifiable content
    if has_transcript:
        score += 0.15

    # Duration sweet spot (5min-90min is ideal for analysis)
    if 300 <= duration_seconds <= 5400:
        score += 0.1
    elif duration_seconds < 120 or duration_seconds > 10800:
        score -= 0.05

    # Known finance channels (configurable)
    _KNOWN_FINANCE = {
        "ThePlainBagel", "BenFelixCSI", "PatrickBoyleOnFinance",
        "TheMoneyGPS", "PeterSchiff", "GeorgeGammon",
        "SvenCarlin", "MichaelBurry", "GregoryMannarino",
    }
    if any(name.lower() in channel.lower() for name in _KNOWN_FINANCE):
        score += 0.1

    return round(max(0.0, min(1.0, score)), 3)


def xai_source_trust(
    post_count: int = 0,
    provider: str = "xai",
    is_llm_synthesized: bool = True,
) -> float:
    """Score 0-1 for X/Grok retriever reliability.

    The X council uses Grok to *generate* post summaries since the direct
    X API is broken (HTTP 402).  This means the data is LLM-synthesized,
    not real tweets.  Trust ceiling is capped accordingly.
    """
    if is_llm_synthesized:
        # Hard ceiling — these aren't real posts
        ceiling = 0.45
    else:
        ceiling = 0.85

    score = 0.2  # base

    # More posts = slightly more evidence to synthesize from
    if post_count > 0:
        score += min(post_count / 50, 0.2)

    # Provider quality
    if provider == "xai":
        score += 0.1  # Grok has native X search
    elif provider == "openai":
        score += 0.05  # no native X access

    return round(max(0.0, min(ceiling, score)), 3)


def crypto_source_trust(
    coin_count: int = 0,
    has_global_data: bool = False,
    api_tier: str = "free",
) -> float:
    """Score 0-1 for CoinGecko data reliability.

    Crypto data is the most trustworthy council — it's real market data
    from a known API, not synthesized text.
    """
    score = 0.6  # high base — real market data

    if api_tier == "pro":
        score += 0.15
    elif api_tier == "free":
        score += 0.05  # still real data, just rate-limited

    if coin_count >= 10:
        score += 0.1
    elif coin_count >= 5:
        score += 0.05

    if has_global_data:
        score += 0.1

    return round(max(0.0, min(1.0, score)), 3)


def polymarket_source_trust(
    market_count: int = 0,
    avg_liquidity: float = 0.0,
    avg_volume: float = 0.0,
) -> float:
    """Score 0-1 for Polymarket data reliability.

    Higher liquidity and volume = more trustworthy price signals.
    Low-liquidity markets are noise.
    """
    score = 0.5  # decent base — real market data

    # Liquidity (>$50K is solid, >$500K is excellent)
    if avg_liquidity > 500_000:
        score += 0.2
    elif avg_liquidity > 50_000:
        score += 0.15
    elif avg_liquidity > 5_000:
        score += 0.05

    # Volume
    if avg_volume > 1_000_000:
        score += 0.15
    elif avg_volume > 100_000:
        score += 0.1
    elif avg_volume > 10_000:
        score += 0.05

    # Market count
    if market_count >= 20:
        score += 0.05

    return round(max(0.0, min(1.0, score)), 3)


# ── Cross-source agreement ────────────────────────────────────────────────

def compute_agreement(sentiments: list[tuple[str, float]]) -> float:
    """Given (source_name, sentiment_score) pairs, measure agreement.

    Returns 1.0 if all sources agree on direction, 0.0 if maximally
    conflicting.  Requires at least 2 sources.
    """
    if len(sentiments) < 2:
        return 0.5  # no cross-validation possible

    scores = [s for _, s in sentiments]
    # Direction: above 0.5 = bullish, below 0.5 = bearish
    directions = [s > 0.5 for s in scores]
    agreement_ratio = max(
        sum(directions) / len(directions),
        1 - sum(directions) / len(directions),
    )

    # Also factor in magnitude similarity
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    coherence = max(0.0, 1.0 - variance * 4)  # 4x scaling: var=0.25 → 0.0

    combined = agreement_ratio * 0.6 + coherence * 0.4
    return round(max(0.0, min(1.0, combined)), 3)
