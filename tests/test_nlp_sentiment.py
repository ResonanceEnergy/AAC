"""Tests for strategies/nlp_sentiment.py — VADER Sentiment Scoring."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest


@pytest.fixture
def headlines():
    """Sample headlines as dicts (NewsAPI style)."""
    return [
        {"title": "Apple stock soars to all-time high on strong earnings",
         "source": {"name": "CNBC"}, "publishedAt": "2026-04-08T10:00:00Z"},
        {"title": "Markets rally as Fed signals rate cuts ahead",
         "source": {"name": "Reuters"}, "publishedAt": "2026-04-08T09:00:00Z"},
        {"title": "Inflation data comes in mixed, markets flat",
         "source": {"name": "Bloomberg"}, "publishedAt": "2026-04-08T08:00:00Z"},
        {"title": "Bank failures spark fears of financial crisis",
         "source": {"name": "FT"}, "publishedAt": "2026-04-07T12:00:00Z"},
        {"title": "Unemployment rises sharply, recession worries grow",
         "source": {"name": "WSJ"}, "publishedAt": "2026-04-07T08:00:00Z"},
    ]


@pytest.fixture
def finnhub_headlines():
    """Finnhub-style news items."""
    return [
        {"headline": "Apple beats estimates with record revenue",
         "source": "MarketWatch", "datetime": "2026-04-08T10:00:00Z"},
        {"headline": "Tech sector faces devastating regulatory crackdown",
         "source": "Reuters", "datetime": "2026-04-08T08:00:00Z"},
    ]


class TestScoredHeadline:
    def test_dataclass(self):
        from strategies.nlp_sentiment import ScoredHeadline
        s = ScoredHeadline(
            source="CNBC",
            headline="Test headline",
            published="2026-01-01",
            ticker="SPY",
            compound=0.5,
            positive=0.3,
            negative=0.0,
            neutral=0.7,
        )
        assert s.compound == 0.5
        assert s.ticker == "SPY"


class TestSentimentSnapshot:
    def test_signal_bullish(self):
        from strategies.nlp_sentiment import SentimentSnapshot
        snap = SentimentSnapshot(
            ticker="SPY",
            timestamp="2026-01-01",
            composite_score=0.25,
            n_headlines=3,
            bullish_pct=0.67,
            bearish_pct=0.0,
            neutral_pct=0.33,
        )
        assert snap.signal == "bullish"

    def test_signal_bearish(self):
        from strategies.nlp_sentiment import SentimentSnapshot
        snap = SentimentSnapshot(
            ticker="SPY",
            timestamp="2026-01-01",
            composite_score=-0.30,
            n_headlines=3,
            bullish_pct=0.0,
            bearish_pct=0.67,
            neutral_pct=0.33,
        )
        assert snap.signal == "bearish"

    def test_signal_neutral(self):
        from strategies.nlp_sentiment import SentimentSnapshot
        snap = SentimentSnapshot(
            ticker="SPY",
            timestamp="2026-01-01",
            composite_score=0.0,
            n_headlines=3,
            bullish_pct=0.33,
            bearish_pct=0.33,
            neutral_pct=0.33,
        )
        assert snap.signal == "neutral"

    def test_to_dict(self):
        from strategies.nlp_sentiment import SentimentSnapshot
        snap = SentimentSnapshot(
            ticker="SPY",
            timestamp="2026-01-01",
            composite_score=0.25,
            n_headlines=3,
            bullish_pct=0.67,
            bearish_pct=0.0,
            neutral_pct=0.33,
        )
        d = snap.to_dict()
        assert d["ticker"] == "SPY"
        assert d["signal"] == "bullish"
        assert d["n_headlines"] == 3


class TestSentimentAnalyzer:
    def test_score_positive_headline(self):
        from strategies.nlp_sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        # Use a headline VADER clearly scores positively
        scored = analyzer.score_headline("This is excellent, amazing, and wonderful news!")
        assert scored.compound > 0
        assert scored.positive > 0

    def test_score_negative_headline(self):
        from strategies.nlp_sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        scored = analyzer.score_headline("Market crash devastating losses terrible")
        assert scored.compound < 0

    def test_score_headlines_batch(self, headlines):
        from strategies.nlp_sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        scored = analyzer.score_headlines(headlines, ticker="SPY")
        assert len(scored) == 5
        # Should be sorted by compound descending
        for i in range(len(scored) - 1):
            assert scored[i].compound >= scored[i + 1].compound

    def test_aggregate(self, headlines):
        from strategies.nlp_sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        scored = analyzer.score_headlines(headlines, ticker="SPY")
        snap = analyzer.aggregate(scored, ticker="SPY")
        assert snap.n_headlines == 5
        assert snap.bullish_pct + snap.bearish_pct + snap.neutral_pct == pytest.approx(1.0)
        assert snap.strongest_bull != ""
        assert snap.strongest_bear != ""

    def test_aggregate_empty(self):
        from strategies.nlp_sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        snap = analyzer.aggregate([], ticker="SPY")
        assert snap.n_headlines == 0
        assert snap.composite_score == 0.0
        assert snap.neutral_pct == 1.0

    def test_recency_weighting(self, headlines):
        from strategies.nlp_sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer(decay_hours=12.0)
        scored = analyzer.score_headlines(headlines, ticker="SPY")
        now = datetime(2026, 4, 8, 12, 0, 0, tzinfo=timezone.utc)
        snap = analyzer.aggregate(scored, ticker="SPY", now=now)
        assert snap.n_headlines == 5

    def test_finnhub_news(self, finnhub_headlines):
        from strategies.nlp_sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        snap = analyzer.score_finnhub_news(finnhub_headlines, ticker="AAPL")
        assert snap.n_headlines == 2
        assert snap.ticker == "AAPL"

    def test_newsapi_articles(self, headlines):
        from strategies.nlp_sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        snap = analyzer.score_newsapi_articles(headlines, ticker="SPY")
        assert snap.n_headlines == 5

    def test_multi_ticker(self, headlines, finnhub_headlines):
        from strategies.nlp_sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        results = analyzer.score_multi_ticker({
            "SPY": headlines,
            "AAPL": finnhub_headlines,
        })
        assert "SPY" in results
        assert "AAPL" in results
        assert results["SPY"].n_headlines == 5
        assert results["AAPL"].n_headlines == 2

    def test_empty_headline_skipped(self):
        from strategies.nlp_sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        items = [{"title": ""}, {"title": "Good news today!"}]
        scored = analyzer.score_headlines(items, ticker="SPY")
        assert len(scored) == 1
