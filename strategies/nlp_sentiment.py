"""NLP Sentiment Scoring — VADER-based news headline scoring.

Scores Finnhub/NewsAPI headlines with VADER sentiment and aggregates
into a composite sentiment signal for the war room.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScoredHeadline:
    """A news headline with sentiment scores."""

    source: str
    headline: str
    published: str
    ticker: str
    compound: float    # VADER compound score [-1, +1]
    positive: float
    negative: float
    neutral: float
    relevance: float = 1.0  # ticker relevance weight


@dataclass
class SentimentSnapshot:
    """Aggregated sentiment for a ticker or the market."""

    ticker: str
    timestamp: str
    composite_score: float     # weighted average compound [-1, +1]
    n_headlines: int
    bullish_pct: float         # % of headlines with compound > 0.05
    bearish_pct: float         # % of headlines with compound < -0.05
    neutral_pct: float
    strongest_bull: str = ""   # most positive headline
    strongest_bear: str = ""   # most negative headline
    headlines: list[ScoredHeadline] = field(default_factory=list)

    @property
    def signal(self) -> str:
        if self.composite_score > 0.15:
            return "bullish"
        if self.composite_score < -0.15:
            return "bearish"
        return "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "composite_score": round(self.composite_score, 4),
            "signal": self.signal,
            "n_headlines": self.n_headlines,
            "bullish_pct": round(self.bullish_pct, 2),
            "bearish_pct": round(self.bearish_pct, 2),
        }


# ---------------------------------------------------------------------------
# Sentiment Analyzer
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """VADER-based news sentiment scorer.

    Parameters
    ----------
    decay_hours : float
        Exponential decay half-life for recency weighting (hours).
    min_compound : float
        Minimum absolute compound score to count as bullish/bearish.
    """

    def __init__(
        self,
        decay_hours: float = 24.0,
        min_compound: float = 0.05,
    ) -> None:
        self.decay_hours = decay_hours
        self.min_compound = min_compound
        self._vader: Any = None

    def _get_vader(self) -> Any:
        if self._vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        return self._vader

    # ── Core scoring ──────────────────────────────────────────────────────

    def score_headline(
        self,
        headline: str,
        source: str = "unknown",
        published: str = "",
        ticker: str = "SPY",
    ) -> ScoredHeadline:
        """Score a single headline with VADER."""
        vader = self._get_vader()
        scores = vader.polarity_scores(headline)

        return ScoredHeadline(
            source=source,
            headline=headline,
            published=published,
            ticker=ticker,
            compound=scores["compound"],
            positive=scores["pos"],
            negative=scores["neg"],
            neutral=scores["neu"],
        )

    def score_headlines(
        self,
        headlines: list[dict[str, Any]],
        ticker: str = "SPY",
    ) -> list[ScoredHeadline]:
        """Score a batch of headlines.

        Parameters
        ----------
        headlines : list[dict]
            Each dict should have 'title' or 'headline', optionally
            'source', 'publishedAt' or 'datetime'.
        """
        scored: list[ScoredHeadline] = []
        for h in headlines:
            text = h.get("title") or h.get("headline") or ""
            if not text:
                continue
            source = h.get("source", {})
            if isinstance(source, dict):
                source = source.get("name", "unknown")
            published = h.get("publishedAt") or h.get("datetime") or ""

            scored.append(self.score_headline(
                headline=text,
                source=str(source),
                published=str(published),
                ticker=ticker,
            ))

        scored.sort(key=lambda s: s.compound, reverse=True)
        return scored

    # ── Aggregation ───────────────────────────────────────────────────────

    def aggregate(
        self,
        scored: list[ScoredHeadline],
        ticker: str = "SPY",
        now: Optional[datetime] = None,
    ) -> SentimentSnapshot:
        """Aggregate scored headlines into a composite sentiment signal."""
        if not scored:
            return SentimentSnapshot(
                ticker=ticker,
                timestamp=datetime.now(timezone.utc).isoformat(),
                composite_score=0.0,
                n_headlines=0,
                bullish_pct=0.0,
                bearish_pct=0.0,
                neutral_pct=1.0,
            )

        now = now or datetime.now(timezone.utc)

        # Recency weighting
        weights: list[float] = []
        for s in scored:
            age_hours = self._age_hours(s.published, now)
            weight = 2 ** (-age_hours / self.decay_hours) if age_hours >= 0 else 1.0
            weights.append(weight)

        total_weight = sum(weights)
        if total_weight < 1e-10:
            total_weight = 1.0

        # Weighted composite
        composite = sum(s.compound * w for s, w in zip(scored, weights)) / total_weight

        # Sentiment distribution
        n = len(scored)
        bullish = sum(1 for s in scored if s.compound > self.min_compound)
        bearish = sum(1 for s in scored if s.compound < -self.min_compound)
        neutral = n - bullish - bearish

        strongest_bull = max(scored, key=lambda s: s.compound).headline if scored else ""
        strongest_bear = min(scored, key=lambda s: s.compound).headline if scored else ""

        return SentimentSnapshot(
            ticker=ticker,
            timestamp=now.isoformat(),
            composite_score=composite,
            n_headlines=n,
            bullish_pct=bullish / n if n > 0 else 0.0,
            bearish_pct=bearish / n if n > 0 else 0.0,
            neutral_pct=neutral / n if n > 0 else 0.0,
            strongest_bull=strongest_bull,
            strongest_bear=strongest_bear,
            headlines=scored,
        )

    # ── Finnhub integration ───────────────────────────────────────────────

    def score_finnhub_news(
        self,
        news_items: list[dict[str, Any]],
        ticker: str = "SPY",
    ) -> SentimentSnapshot:
        """Score Finnhub news response and return aggregated sentiment.

        Parameters
        ----------
        news_items : list[dict]
            Finnhub news items with 'headline', 'source', 'datetime'.
        """
        scored = self.score_headlines(news_items, ticker=ticker)
        return self.aggregate(scored, ticker=ticker)

    # ── NewsAPI integration ───────────────────────────────────────────────

    def score_newsapi_articles(
        self,
        articles: list[dict[str, Any]],
        ticker: str = "SPY",
    ) -> SentimentSnapshot:
        """Score NewsAPI articles and return aggregated sentiment.

        Parameters
        ----------
        articles : list[dict]
            NewsAPI articles with 'title', 'source', 'publishedAt'.
        """
        scored = self.score_headlines(articles, ticker=ticker)
        return self.aggregate(scored, ticker=ticker)

    # ── Multi-ticker ──────────────────────────────────────────────────────

    def score_multi_ticker(
        self,
        headlines_by_ticker: dict[str, list[dict[str, Any]]],
    ) -> dict[str, SentimentSnapshot]:
        """Score headlines for multiple tickers."""
        results: dict[str, SentimentSnapshot] = {}
        for ticker, headlines in headlines_by_ticker.items():
            scored = self.score_headlines(headlines, ticker=ticker)
            results[ticker] = self.aggregate(scored, ticker=ticker)
        return results

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _age_hours(published: str, now: datetime) -> float:
        """Parse published timestamp and return age in hours."""
        if not published:
            return 0.0
        try:
            # Try ISO format
            pub = datetime.fromisoformat(published.replace("Z", "+00:00"))
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            diff = now - pub
            return diff.total_seconds() / 3600
        except (ValueError, TypeError):
            pass
        try:
            # Try Unix timestamp
            pub = datetime.fromtimestamp(float(published), tz=timezone.utc)
            diff = now - pub
            return diff.total_seconds() / 3600
        except (ValueError, TypeError):
            return 0.0
