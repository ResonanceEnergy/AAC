from __future__ import annotations

"""Tests for the Living Doctrine Engine.

All HTTP calls are mocked — no real YouTube / Polymarket / yfinance access.
"""

import json
import math
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from councils.youtube.models import CouncilEntry, TranscriptSegment, VideoInsight, VideoMeta
from strategies.living_doctrine_engine import (
    DoctrineRule,
    DoctrineStore,
    IngestLog,
    IngestRecord,
    LivingDoctrineEngine,
    Sandbox,
    SandboxEntry,
    _extract_video_id,
    _now_iso,
    _sentiment_signal,
    insights_to_doctrine_rules,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory for LDE persistence."""
    d = tmp_path / "living_doctrine"
    d.mkdir()
    return d


@pytest.fixture()
def doctrine_store(tmp_data_dir: Path) -> DoctrineStore:
    return DoctrineStore(tmp_data_dir / "doctrine.json")


@pytest.fixture()
def sandbox(tmp_data_dir: Path) -> Sandbox:
    return Sandbox(tmp_data_dir / "sandbox.json")


@pytest.fixture()
def ingest_log(tmp_data_dir: Path) -> IngestLog:
    return IngestLog(tmp_data_dir / "ingest_log.json")


@pytest.fixture()
def sample_video_meta() -> VideoMeta:
    return VideoMeta(
        video_id="abc123",
        title="Market Analysis Q2 2026",
        channel="FinanceGuru",
        upload_date="20260401",
        duration_seconds=1800,
        description="Deep dive on Q2 outlook",
        url="https://www.youtube.com/watch?v=abc123",
        view_count=50000,
        like_count=2000,
    )


@pytest.fixture()
def sample_council_entry(sample_video_meta: VideoMeta) -> CouncilEntry:
    return CouncilEntry(
        meta=sample_video_meta,
        transcript=[
            TranscriptSegment(start=0.0, end=5.0, text="Markets are bullish"),
            TranscriptSegment(start=5.0, end=10.0, text="Buy the dip strategy"),
        ],
        insights=VideoInsight(
            title="Market Analysis Q2 2026",
            key_topics=["markets", "stocks", "bonds"],
            quotes=["Markets are bullish", "Buy the dip"],
            summary="Markets showing bullish momentum with strong buy signals",
            actionable_items=[
                "Should buy growth stocks for upside potential",
                "Must avoid overvalued tech names with downside risk",
            ],
            sentiment="positive",
            trust_score={"overall": 0.85, "source_reliability": 0.9},
        ),
        markdown_path="output/test.md",
        processed_at="2026-04-01T12:00:00+00:00",
    )


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    cfg = tmp_path / "lde_channels.json"
    cfg.write_text(json.dumps({
        "youtube_channels": [
            {"url": "https://www.youtube.com/@TestChannel", "category": "test",
             "priority": "high"},
        ],
        "ingest_settings": {
            "max_videos_per_channel": 3,
            "max_duration_hours": 1.0,
            "schedule_cron": "0 8 * * *",
        },
        "doctrine_settings": {
            "signal_decay_days": 30,
            "min_conviction_threshold": 0.6,
        },
    }), encoding="utf-8")
    return cfg


@pytest.fixture()
def engine(config_path: Path, tmp_data_dir: Path) -> LivingDoctrineEngine:
    return LivingDoctrineEngine(config_path=config_path, data_dir=tmp_data_dir)


# ============================================================================
# DoctrineRule tests
# ============================================================================

class TestDoctrineRule:
    def test_create_rule(self) -> None:
        rule = DoctrineRule(
            rule_id="test_1", text="Buy low sell high",
            source_video="v1", source_channel="ch1",
            created_at=_now_iso(), last_reinforced=_now_iso(),
            conviction=0.8, signal_value=0.5,
        )
        assert rule.conviction == 0.8
        assert rule.signal_value == 0.5

    def test_reinforce(self) -> None:
        rule = DoctrineRule(
            rule_id="r1", text="test", source_video="v1",
            source_channel="ch1", created_at=_now_iso(),
            last_reinforced=_now_iso(), conviction=0.7,
        )
        rule.reinforce(0.2)
        assert rule.conviction == pytest.approx(0.9, abs=0.01)
        assert rule.reinforcement_count == 2

    def test_reinforce_caps_at_1(self) -> None:
        rule = DoctrineRule(
            rule_id="r1", text="test", source_video="v1",
            source_channel="ch1", created_at=_now_iso(),
            last_reinforced=_now_iso(), conviction=0.95,
        )
        rule.reinforce(0.2)
        assert rule.conviction == 1.0

    def test_decay(self) -> None:
        past = "2026-01-01T00:00:00+00:00"
        rule = DoctrineRule(
            rule_id="r1", text="test", source_video="v1",
            source_channel="ch1", created_at=past,
            last_reinforced=past, conviction=1.0,
        )
        rule.decay(decay_half_life_days=30.0)
        assert rule.conviction < 1.0
        assert rule.conviction >= 0.0


# ============================================================================
# DoctrineStore tests
# ============================================================================

class TestDoctrineStore:
    def test_add_and_get_rule(self, doctrine_store: DoctrineStore) -> None:
        rule = DoctrineRule(
            rule_id="r1", text="Buy growth", source_video="v1",
            source_channel="ch1", created_at=_now_iso(),
            last_reinforced=_now_iso(), conviction=0.8,
        )
        doctrine_store.add_rule(rule)
        assert doctrine_store.get_rule("r1") is not None
        assert len(doctrine_store.rules) == 1

    def test_remove_rule(self, doctrine_store: DoctrineStore) -> None:
        rule = DoctrineRule(
            rule_id="r1", text="test", source_video="v1",
            source_channel="ch1", created_at=_now_iso(),
            last_reinforced=_now_iso(), conviction=0.8,
        )
        doctrine_store.add_rule(rule)
        assert doctrine_store.remove_rule("r1")
        assert doctrine_store.get_rule("r1") is None

    def test_remove_nonexistent(self, doctrine_store: DoctrineStore) -> None:
        assert not doctrine_store.remove_rule("nope")

    def test_persistence(self, tmp_data_dir: Path) -> None:
        path = tmp_data_dir / "persist_test.json"
        store1 = DoctrineStore(path)
        store1.add_rule(DoctrineRule(
            rule_id="r1", text="persistent rule", source_video="v1",
            source_channel="ch1", created_at=_now_iso(),
            last_reinforced=_now_iso(), conviction=0.9,
        ))
        store1.save()

        store2 = DoctrineStore(path)
        assert store2.get_rule("r1") is not None
        assert store2.get_rule("r1").text == "persistent rule"

    def test_active_rules(self, doctrine_store: DoctrineStore) -> None:
        now = _now_iso()
        doctrine_store.add_rule(DoctrineRule(
            rule_id="active", text="active rule", source_video="v1",
            source_channel="ch1", created_at=now, last_reinforced=now,
            conviction=0.8,
        ))
        doctrine_store.add_rule(DoctrineRule(
            rule_id="inactive", text="inactive rule", source_video="v1",
            source_channel="ch1", created_at=now, last_reinforced=now,
            conviction=0.1,
        ))
        active = doctrine_store.active_rules
        assert len(active) == 1
        assert active[0].rule_id == "active"

    def test_find_matching(self, doctrine_store: DoctrineStore) -> None:
        now = _now_iso()
        doctrine_store.add_rule(DoctrineRule(
            rule_id="r1",
            text="buy growth stocks for long term upside potential gains",
            source_video="v1", source_channel="ch1",
            created_at=now, last_reinforced=now, conviction=0.8,
        ))
        # Similar text should match
        match = doctrine_store.find_matching(
            "buy growth stocks for long term upside potential"
        )
        assert match is not None
        assert match.rule_id == "r1"

    def test_find_no_match(self, doctrine_store: DoctrineStore) -> None:
        now = _now_iso()
        doctrine_store.add_rule(DoctrineRule(
            rule_id="r1", text="buy growth stocks",
            source_video="v1", source_channel="ch1",
            created_at=now, last_reinforced=now, conviction=0.8,
        ))
        # Very different text should not match
        match = doctrine_store.find_matching("sell all crypto immediately")
        assert match is None

    def test_doctrine_signal_neutral(self, doctrine_store: DoctrineStore) -> None:
        assert doctrine_store.doctrine_signal() == 0.0

    def test_doctrine_signal_bullish(self, doctrine_store: DoctrineStore) -> None:
        now = _now_iso()
        doctrine_store.add_rule(DoctrineRule(
            rule_id="r1", text="bullish", source_video="v1",
            source_channel="ch1", created_at=now, last_reinforced=now,
            conviction=0.8, signal_value=0.7,
        ))
        sig = doctrine_store.doctrine_signal()
        assert sig > 0

    def test_apply_decay_prunes(self, doctrine_store: DoctrineStore) -> None:
        old = "2020-01-01T00:00:00+00:00"
        doctrine_store.add_rule(DoctrineRule(
            rule_id="old", text="ancient rule", source_video="v1",
            source_channel="ch1", created_at=old, last_reinforced=old,
            conviction=0.1,
        ))
        pruned = doctrine_store.apply_decay(half_life_days=1.0)
        assert pruned >= 1
        assert doctrine_store.get_rule("old") is None


# ============================================================================
# Sandbox tests
# ============================================================================

class TestSandbox:
    def test_add_entry(self, sandbox: Sandbox) -> None:
        entry = SandboxEntry(
            entry_id="e1", video_id="v1", channel="ch1",
            title="Test", key_topics=["test"], quotes=[],
            summary="summary", actionable_items=[], sentiment="neutral",
            trust_score={}, ingested_at=_now_iso(),
        )
        sandbox.add(entry)
        assert len(sandbox.entries) == 1

    def test_has_video(self, sandbox: Sandbox) -> None:
        sandbox.add(SandboxEntry(
            entry_id="e1", video_id="v1", channel="ch1",
            title="T", key_topics=[], quotes=[], summary="",
            actionable_items=[], sentiment="neutral",
            trust_score={}, ingested_at=_now_iso(),
        ))
        assert sandbox.has_video("v1")
        assert not sandbox.has_video("v2")

    def test_persistence(self, tmp_data_dir: Path) -> None:
        path = tmp_data_dir / "sandbox_test.json"
        s1 = Sandbox(path)
        s1.add(SandboxEntry(
            entry_id="e1", video_id="v1", channel="ch1",
            title="Persistent", key_topics=[], quotes=[], summary="",
            actionable_items=[], sentiment="neutral",
            trust_score={}, ingested_at=_now_iso(),
        ))
        s1.save()

        s2 = Sandbox(path)
        assert len(s2.entries) == 1

    def test_recent(self, sandbox: Sandbox) -> None:
        sandbox.add(SandboxEntry(
            entry_id="e1", video_id="v1", channel="ch1",
            title="Recent", key_topics=[], quotes=[], summary="",
            actionable_items=[], sentiment="neutral",
            trust_score={}, ingested_at=_now_iso(),
        ))
        recent = sandbox.recent(days=1)
        assert len(recent) == 1


# ============================================================================
# IngestLog tests
# ============================================================================

class TestIngestLog:
    def test_record_dedup(self, ingest_log: IngestLog) -> None:
        rec = IngestRecord(
            video_id="v1", channel="ch1", title="T",
            ingested_at=_now_iso(),
        )
        ingest_log.record(rec)
        assert ingest_log.is_processed("v1")
        assert not ingest_log.is_processed("v2")

    def test_persistence(self, tmp_data_dir: Path) -> None:
        path = tmp_data_dir / "ingest_test.json"
        log1 = IngestLog(path)
        log1.record(IngestRecord(
            video_id="v1", channel="ch1", title="T",
            ingested_at=_now_iso(),
        ))
        log1.save()

        log2 = IngestLog(path)
        assert log2.is_processed("v1")


# ============================================================================
# Sentiment analysis tests
# ============================================================================

class TestSentiment:
    def test_bullish_text(self) -> None:
        sentiment, signal = _sentiment_signal(
            "buy bullish growth rally breakout upside opportunity strong"
        )
        assert sentiment == "positive"
        assert signal > 0

    def test_bearish_text(self) -> None:
        sentiment, signal = _sentiment_signal(
            "sell bearish crash risk correction downgrade recession weak"
        )
        assert sentiment == "negative"
        assert signal < 0

    def test_neutral_text(self) -> None:
        sentiment, signal = _sentiment_signal(
            "the weather today is quite pleasant and sunny"
        )
        assert sentiment == "neutral"
        assert signal == 0.0

    def test_mixed_text(self) -> None:
        sentiment, signal = _sentiment_signal(
            "buy opportunity but also sell risk and crash potential"
        )
        assert sentiment in ("mixed", "positive", "negative")


# ============================================================================
# Insight to doctrine rule conversion
# ============================================================================

class TestInsightConversion:
    def test_actionable_items_become_rules(
        self, sample_council_entry: CouncilEntry,
    ) -> None:
        rules = insights_to_doctrine_rules(sample_council_entry)
        assert len(rules) >= 2  # at least 2 actionable items

    def test_rule_has_signal(
        self, sample_council_entry: CouncilEntry,
    ) -> None:
        rules = insights_to_doctrine_rules(sample_council_entry)
        # At least one rule should have a non-zero signal
        signals = [r.signal_value for r in rules]
        assert any(s != 0.0 for s in signals)

    def test_rule_tags(
        self, sample_council_entry: CouncilEntry,
    ) -> None:
        rules = insights_to_doctrine_rules(sample_council_entry)
        for rule in rules:
            assert isinstance(rule.tags, list)


# ============================================================================
# Video ID extraction
# ============================================================================

class TestVideoIdExtraction:
    def test_standard_url(self) -> None:
        assert _extract_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_short_url(self) -> None:
        assert _extract_video_id(
            "https://youtu.be/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_embed_url(self) -> None:
        assert _extract_video_id(
            "https://www.youtube.com/embed/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_invalid_url(self) -> None:
        assert _extract_video_id("https://example.com") is None


# ============================================================================
# LivingDoctrineEngine integration tests
# ============================================================================

class TestLivingDoctrineEngine:
    def test_init(self, engine: LivingDoctrineEngine) -> None:
        assert len(engine.channels) == 1
        assert engine.get_doctrine_signal() == 0.0

    def test_status(self, engine: LivingDoctrineEngine) -> None:
        status = engine.status()
        assert "total_rules" in status
        assert "active_rules" in status
        assert "doctrine_signal" in status
        assert "videos_processed" in status
        assert "channels_configured" in status
        assert status["channels_configured"] == 1

    def test_absorb_entry(
        self,
        engine: LivingDoctrineEngine,
        sample_council_entry: CouncilEntry,
    ) -> None:
        n_rules = engine._absorb_entry(sample_council_entry)
        assert n_rules >= 1
        assert len(engine.doctrine.rules) >= 1
        assert engine.sandbox.has_video("abc123")

    def test_absorb_duplicate_reinforces(
        self,
        engine: LivingDoctrineEngine,
        sample_council_entry: CouncilEntry,
    ) -> None:
        engine._absorb_entry(sample_council_entry)
        initial_rules = len(engine.doctrine.rules)

        # Absorb same entry → should reinforce existing rules
        engine._absorb_entry(sample_council_entry)
        assert len(engine.doctrine.rules) == initial_rules

    def test_feed_alpha_engine(
        self,
        engine: LivingDoctrineEngine,
        sample_council_entry: CouncilEntry,
    ) -> None:
        engine._absorb_entry(sample_council_entry)

        from strategies.alpha_engine import clear_signal_history, get_signal_history
        clear_signal_history()

        engine.feed_alpha_engine()
        history = get_signal_history("lde_doctrine")
        assert len(history) == 1

        clear_signal_history()

    @patch("councils.youtube.scraper.list_channel_videos")
    @patch("strategies.living_doctrine_engine.LivingDoctrineEngine._process_video")
    def test_ingest_channel(
        self,
        mock_process: MagicMock,
        mock_list: MagicMock,
        engine: LivingDoctrineEngine,
        sample_video_meta: VideoMeta,
        sample_council_entry: CouncilEntry,
    ) -> None:
        mock_list.return_value = [sample_video_meta]
        mock_process.return_value = sample_council_entry

        entries = engine.ingest_channel("https://youtube.com/@Test")
        assert len(entries) == 1
        assert engine.ingest_log.is_processed("abc123")

    @patch("councils.youtube.scraper.list_channel_videos")
    @patch("strategies.living_doctrine_engine.LivingDoctrineEngine._process_video")
    def test_ingest_channel_skip_processed(
        self,
        mock_process: MagicMock,
        mock_list: MagicMock,
        engine: LivingDoctrineEngine,
        sample_video_meta: VideoMeta,
        sample_council_entry: CouncilEntry,
    ) -> None:
        # Pre-mark as processed
        engine.ingest_log.record(IngestRecord(
            video_id="abc123", channel="ch1", title="T",
            ingested_at=_now_iso(),
        ))

        mock_list.return_value = [sample_video_meta]
        mock_process.return_value = sample_council_entry

        entries = engine.ingest_channel("https://youtube.com/@Test")
        assert len(entries) == 0
        mock_process.assert_not_called()

    @patch("councils.youtube.scraper.list_channel_videos")
    @patch("strategies.living_doctrine_engine.LivingDoctrineEngine._process_video")
    def test_ingest_all_channels(
        self,
        mock_process: MagicMock,
        mock_list: MagicMock,
        engine: LivingDoctrineEngine,
        sample_video_meta: VideoMeta,
        sample_council_entry: CouncilEntry,
    ) -> None:
        mock_list.return_value = [sample_video_meta]
        mock_process.return_value = sample_council_entry

        entries = engine.ingest_all_channels()
        assert len(entries) == 1

    def test_save_and_reload(
        self,
        config_path: Path,
        tmp_data_dir: Path,
        sample_council_entry: CouncilEntry,
    ) -> None:
        eng1 = LivingDoctrineEngine(config_path=config_path, data_dir=tmp_data_dir)
        eng1._absorb_entry(sample_council_entry)
        eng1._save_all()

        # Reload from disk
        eng2 = LivingDoctrineEngine(config_path=config_path, data_dir=tmp_data_dir)
        assert len(eng2.doctrine.rules) >= 1
        assert eng2.sandbox.has_video("abc123")


# ============================================================================
# Backtest module tests
# ============================================================================

class TestBacktest:
    def test_basic_backtest(self) -> None:
        from strategies.living_doctrine_backtest import backtest_doctrine_signal

        signals = [0.0, 0.5, 0.5, 0.5, -0.5, -0.5, 0.0, 0.5, 0.5, 0.5]
        dates = [f"2026-01-{i+1:02d}" for i in range(10)]
        prices = [100, 101, 102, 103, 102, 101, 100, 101, 103, 105]

        result = backtest_doctrine_signal(signals, dates, prices, ticker="TEST")
        assert result.ticker == "TEST"
        assert result.total_trades >= 0
        assert result.period != "N/A"

    def test_empty_data(self) -> None:
        from strategies.living_doctrine_backtest import backtest_doctrine_signal

        result = backtest_doctrine_signal([], [], [])
        assert "error" in result.details

    def test_sharpe_calculation(self) -> None:
        from strategies.living_doctrine_backtest import _sharpe

        # Positive returns → positive Sharpe
        returns = [0.01, 0.02, 0.01, 0.015, 0.005, 0.01, 0.02]
        assert _sharpe(returns) > 0

    def test_sortino_calculation(self) -> None:
        from strategies.living_doctrine_backtest import _sortino

        returns = [0.01, -0.005, 0.02, -0.001, 0.015]
        s = _sortino(returns)
        assert isinstance(s, float)

    def test_max_drawdown(self) -> None:
        from strategies.living_doctrine_backtest import _max_drawdown

        equity = [100, 110, 105, 95, 100, 115]
        mdd = _max_drawdown(equity)
        assert mdd > 0
        assert mdd == pytest.approx((110 - 95) / 110, abs=0.01)

    def test_stop_loss_trigger(self) -> None:
        from strategies.living_doctrine_backtest import backtest_doctrine_signal

        # Strong buy signal followed by a crash
        signals = [0.0, 0.8, 0.8, 0.8, 0.8, 0.8]
        dates = [f"2026-01-{i+1:02d}" for i in range(6)]
        prices = [100, 100, 95, 90, 85, 80]  # 20% drop

        result = backtest_doctrine_signal(
            signals, dates, prices, signal_threshold=0.3,
            stop_loss_pct=0.05,
        )
        # Should have triggered stop loss
        if result.trade_log:
            exits = [t["exit_reason"] for t in result.trade_log]
            assert "stop_loss" in exits

    @patch("yfinance.Ticker")
    def test_backtest_against_market(self, mock_ticker_cls: MagicMock) -> None:
        from strategies.living_doctrine_backtest import backtest_against_market

        import pandas as pd

        # Mock yfinance
        dates = pd.date_range("2026-01-01", periods=30)
        mock_hist = pd.DataFrame(
            {"Close": [100 + i * 0.5 for i in range(30)]},
            index=dates,
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_hist
        mock_ticker_cls.return_value = mock_ticker

        signals = [0.5] * 30
        result = backtest_against_market(signals, ticker="SPY", lookback_days=30)
        assert result.ticker == "SPY"
        assert result.period != "N/A"
