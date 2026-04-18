from __future__ import annotations

"""Living Doctrine Engine (LDE) — self-improving trading intelligence.

Ingests YouTube videos, extracts actionable insights, builds evolving
doctrine rules, and feeds signals into the alpha engine.  Persists
state to ``data/living_doctrine/`` as JSON.

Reuses existing AAC council infrastructure:
- ``councils.youtube.scraper`` — channel listing, subtitle extraction
- ``councils.youtube.analyzer`` — extractive transcript analysis
- ``councils.youtube.pipeline`` — end-to-end video processing
- ``councils.polymarket.scraper`` — prediction-market consensus
- ``councils.trust`` — trust scoring
- ``strategies.alpha_engine`` — signal combination (12 equations)
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from councils.youtube.models import CouncilEntry, TranscriptSegment, VideoMeta

_log = structlog.get_logger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data" / "living_doctrine"
_DOCTRINE_PATH = _DATA_DIR / "doctrine.json"
_SANDBOX_PATH = _DATA_DIR / "sandbox.json"
_INGEST_LOG_PATH = _DATA_DIR / "ingest_log.json"
_CONFIG_PATH = _ROOT / "config" / "lde_channels.json"

# Ensure data directory exists on import
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Defaults ───────────────────────────────────────────────────────────────────

_DEFAULT_SIGNAL_DECAY_DAYS = 30
_DEFAULT_MIN_CONVICTION = 0.6
_DEFAULT_MAX_RULES = 200
_DEFAULT_BACKTEST_LOOKBACK = 90


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class DoctrineRule:
    """A single doctrine rule — an evolving trading insight."""

    rule_id: str
    text: str
    source_video: str
    source_channel: str
    created_at: str
    last_reinforced: str
    conviction: float  # 0.0–1.0
    reinforcement_count: int = 1
    tags: list[str] = field(default_factory=list)
    sentiment: str = "neutral"
    signal_value: float = 0.0  # contribution to alpha signal
    backtest_win_rate: float | None = None

    def decay(self, decay_half_life_days: float = 30.0) -> None:
        """Apply time-based decay to conviction."""
        now = time.time()
        last = _parse_iso(self.last_reinforced) or now
        age_days = (now - last) / 86400.0
        factor = math.exp(-0.693 * age_days / max(decay_half_life_days, 1.0))
        self.conviction = max(0.0, self.conviction * factor)

    def reinforce(self, amount: float = 0.1) -> None:
        """Reinforce this rule (matching insight seen again)."""
        self.conviction = min(1.0, self.conviction + amount)
        self.reinforcement_count += 1
        self.last_reinforced = _now_iso()


@dataclass
class SandboxEntry:
    """A persistent insight stored in the sandbox knowledge base."""

    entry_id: str
    video_id: str
    channel: str
    title: str
    key_topics: list[str]
    quotes: list[str]
    summary: str
    actionable_items: list[str]
    sentiment: str
    trust_score: dict[str, Any]
    ingested_at: str
    signal_value: float = 0.0


@dataclass
class IngestRecord:
    """Record of a processed video (deduplication)."""

    video_id: str
    channel: str
    title: str
    ingested_at: str
    doctrine_rules_created: int = 0


# ============================================================================
# HELPERS
# ============================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(s: str) -> float | None:
    """Parse ISO timestamp to epoch seconds, or None."""
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


def _load_json(path: Path) -> Any:
    """Load JSON from path, returning empty dict on failure."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("lde.load_json_failed", path=str(path), error=str(exc))
        return {}


def _save_json(path: Path, data: Any) -> None:
    """Atomically write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# ============================================================================
# DOCTRINE STORE — persistent, evolving rule set
# ============================================================================

class DoctrineStore:
    """Persistent store for doctrine rules.

    Backed by a JSON file at ``data/living_doctrine/doctrine.json``.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DOCTRINE_PATH
        self._rules: dict[str, DoctrineRule] = {}
        self._load()

    # ── persistence ────────────────────────────────────────────────

    def _load(self) -> None:
        raw = _load_json(self._path)
        rules = raw.get("rules", {})
        for rid, rdata in rules.items():
            self._rules[rid] = DoctrineRule(**rdata)

    def save(self) -> None:
        data = {"rules": {k: asdict(v) for k, v in self._rules.items()},
                "updated_at": _now_iso()}
        _save_json(self._path, data)

    # ── CRUD ───────────────────────────────────────────────────────

    def add_rule(self, rule: DoctrineRule) -> None:
        self._rules[rule.rule_id] = rule

    def get_rule(self, rule_id: str) -> DoctrineRule | None:
        return self._rules.get(rule_id)

    def remove_rule(self, rule_id: str) -> bool:
        return self._rules.pop(rule_id, None) is not None

    @property
    def rules(self) -> list[DoctrineRule]:
        return list(self._rules.values())

    @property
    def active_rules(self) -> list[DoctrineRule]:
        """Return rules above the minimum conviction threshold."""
        return [r for r in self._rules.values()
                if r.conviction >= _DEFAULT_MIN_CONVICTION]

    def find_matching(self, text: str) -> DoctrineRule | None:
        """Find a rule whose core text substantially overlaps with *text*.

        Uses Jaccard similarity on word sets.
        """
        words_new = set(text.lower().split())
        if len(words_new) < 3:
            return None

        best: DoctrineRule | None = None
        best_score = 0.0
        for rule in self._rules.values():
            words_r = set(rule.text.lower().split())
            intersection = len(words_new & words_r)
            union = len(words_new | words_r)
            if union == 0:
                continue
            jaccard = intersection / union
            if jaccard > 0.5 and jaccard > best_score:
                best = rule
                best_score = jaccard
        return best

    # ── decay / prune ──────────────────────────────────────────────

    def apply_decay(self, half_life_days: float = 30.0) -> int:
        """Decay all rules and prune dead ones (conviction < 0.05)."""
        pruned = 0
        dead: list[str] = []
        for rid, rule in self._rules.items():
            rule.decay(half_life_days)
            if rule.conviction < 0.05:
                dead.append(rid)
        for rid in dead:
            del self._rules[rid]
            pruned += 1

        # Enforce max rules — drop lowest conviction
        max_rules = _DEFAULT_MAX_RULES
        if len(self._rules) > max_rules:
            sorted_rules = sorted(self._rules.items(),
                                  key=lambda x: x[1].conviction)
            to_drop = len(self._rules) - max_rules
            for rid, _ in sorted_rules[:to_drop]:
                del self._rules[rid]
                pruned += 1

        return pruned

    # ── signal generation ──────────────────────────────────────────

    def doctrine_signal(self) -> float:
        """Aggregate signal from all active rules.

        Returns a value in [-1, 1] representing the doctrine's net bias.
        """
        active = self.active_rules
        if not active:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0
        for rule in active:
            w = rule.conviction
            weighted_sum += rule.signal_value * w
            weight_total += w

        if weight_total < 1e-9:
            return 0.0

        raw = weighted_sum / weight_total
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, raw))


# ============================================================================
# SANDBOX — persistent insight accumulation
# ============================================================================

class Sandbox:
    """Accumulates processed insights from all ingested videos.

    Backed by ``data/living_doctrine/sandbox.json``.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _SANDBOX_PATH
        self._entries: dict[str, SandboxEntry] = {}
        self._load()

    def _load(self) -> None:
        raw = _load_json(self._path)
        entries = raw.get("entries", {})
        for eid, edata in entries.items():
            self._entries[eid] = SandboxEntry(**edata)

    def save(self) -> None:
        data = {"entries": {k: asdict(v) for k, v in self._entries.items()},
                "updated_at": _now_iso()}
        _save_json(self._path, data)

    def add(self, entry: SandboxEntry) -> None:
        self._entries[entry.entry_id] = entry

    def has_video(self, video_id: str) -> bool:
        return any(e.video_id == video_id for e in self._entries.values())

    @property
    def entries(self) -> list[SandboxEntry]:
        return list(self._entries.values())

    def recent(self, days: int = 7) -> list[SandboxEntry]:
        """Return entries ingested within the last *days*."""
        cutoff = time.time() - days * 86400
        result: list[SandboxEntry] = []
        for e in self._entries.values():
            ts = _parse_iso(e.ingested_at)
            if ts and ts >= cutoff:
                result.append(e)
        return result


# ============================================================================
# INGEST LOG — deduplication
# ============================================================================

class IngestLog:
    """Tracks which videos have been processed to avoid reprocessing."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _INGEST_LOG_PATH
        self._records: dict[str, IngestRecord] = {}
        self._load()

    def _load(self) -> None:
        raw = _load_json(self._path)
        records = raw.get("records", {})
        for vid, rdata in records.items():
            self._records[vid] = IngestRecord(**rdata)

    def save(self) -> None:
        data = {"records": {k: asdict(v) for k, v in self._records.items()},
                "updated_at": _now_iso()}
        _save_json(self._path, data)

    def is_processed(self, video_id: str) -> bool:
        return video_id in self._records

    def record(self, rec: IngestRecord) -> None:
        self._records[rec.video_id] = rec

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[IngestRecord]:
        return list(self._records.values())


# ============================================================================
# INSIGHT → DOCTRINE CONVERSION
# ============================================================================

_BULLISH_WORDS = frozenset({
    "buy", "bullish", "upside", "growth", "rally", "breakout", "opportunity",
    "undervalued", "upgrade", "strong", "beat", "outperform", "expansion",
    "recovery", "momentum", "accumulate", "higher", "surge", "profit",
    "dividend", "yield", "income", "oversold", "bottom",
})

_BEARISH_WORDS = frozenset({
    "sell", "bearish", "downside", "risk", "crash", "breakdown", "overvalued",
    "downgrade", "weak", "miss", "underperform", "contraction", "recession",
    "correction", "debt", "default", "inflation", "bubble", "cut", "short",
    "overbought", "top", "decline", "loss",
})


def _sentiment_signal(text: str) -> tuple[str, float]:
    """Compute sentiment and signal value from text.

    Returns (sentiment_label, signal_value) where signal is in [-1, 1].
    """
    words = text.lower().split()
    bull = sum(1 for w in words if w in _BULLISH_WORDS)
    bear = sum(1 for w in words if w in _BEARISH_WORDS)
    total = bull + bear
    if total == 0:
        return "neutral", 0.0
    if bull > bear * 2:
        return "positive", min(1.0, bull / max(total, 1))
    if bear > bull * 2:
        return "negative", -min(1.0, bear / max(total, 1))
    return "mixed", (bull - bear) / max(total, 1)


def insights_to_doctrine_rules(
    entry: CouncilEntry,
) -> list[DoctrineRule]:
    """Convert a processed YouTube council entry into doctrine rules.

    Each actionable item becomes a candidate rule; the summary also
    produces a rule if it contains actionable language.
    """
    rules: list[DoctrineRule] = []
    now = _now_iso()
    base_id = f"{entry.meta.video_id}_{entry.meta.channel}"

    # Each actionable item → a rule
    for idx, item in enumerate(entry.insights.actionable_items):
        sentiment, signal = _sentiment_signal(item)
        rule = DoctrineRule(
            rule_id=f"{base_id}_action_{idx}",
            text=item,
            source_video=entry.meta.video_id,
            source_channel=entry.meta.channel,
            created_at=now,
            last_reinforced=now,
            conviction=0.7,
            tags=entry.insights.key_topics[:5],
            sentiment=sentiment,
            signal_value=signal,
        )
        rules.append(rule)

    # Summary → rule if it has signal content
    summary = entry.insights.summary
    sentiment, signal = _sentiment_signal(summary)
    if abs(signal) > 0.1:
        rules.append(DoctrineRule(
            rule_id=f"{base_id}_summary",
            text=summary[:500],
            source_video=entry.meta.video_id,
            source_channel=entry.meta.channel,
            created_at=now,
            last_reinforced=now,
            conviction=0.6,
            tags=entry.insights.key_topics[:5],
            sentiment=sentiment,
            signal_value=signal,
        ))

    return rules


# ============================================================================
# CORE ENGINE
# ============================================================================

class LivingDoctrineEngine:
    """Orchestrates the full LDE pipeline.

    - Ingests YouTube videos from a channel subscription list
    - Extracts insights via the existing council analyzer
    - Converts insights to evolving doctrine rules
    - Feeds doctrine signals into the alpha engine
    - Persists all state to ``data/living_doctrine/``
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        data_dir: Path | str | None = None,
    ) -> None:
        cfg_path = Path(config_path) if config_path else _CONFIG_PATH
        d_dir = Path(data_dir) if data_dir else _DATA_DIR
        d_dir.mkdir(parents=True, exist_ok=True)

        self._config = self._load_config(cfg_path)
        self.doctrine = DoctrineStore(d_dir / "doctrine.json")
        self.sandbox = Sandbox(d_dir / "sandbox.json")
        self.ingest_log = IngestLog(d_dir / "ingest_log.json")
        self._data_dir = d_dir

        _log.info("lde.init", channels=len(self.channels),
                  rules=len(self.doctrine.rules),
                  sandbox_size=len(self.sandbox.entries))

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                _log.warning("lde.config_load_failed", error=str(exc))
        return {"youtube_channels": [], "ingest_settings": {},
                "doctrine_settings": {}}

    # ── config accessors ───────────────────────────────────────────

    @property
    def channels(self) -> list[dict[str, str]]:
        return self._config.get("youtube_channels", [])

    @property
    def ingest_settings(self) -> dict[str, Any]:
        return self._config.get("ingest_settings", {})

    @property
    def doctrine_settings(self) -> dict[str, Any]:
        return self._config.get("doctrine_settings", {})

    # ── ingest pipeline ────────────────────────────────────────────

    def ingest_channel(
        self,
        channel_url: str,
        *,
        max_videos: int | None = None,
        use_whisper: bool = False,
    ) -> list[CouncilEntry]:
        """Ingest latest videos from a single channel.

        1. List recent videos
        2. Skip already-processed (via ingest log)
        3. Process each video through the YouTube council pipeline
        4. Convert insights → doctrine rules
        5. Persist everything
        """
        from councils.youtube.scraper import list_channel_videos

        limit = max_videos or self.ingest_settings.get("max_videos_per_channel", 5)
        max_hours = self.ingest_settings.get("max_duration_hours", 2.0)

        videos = list_channel_videos(channel_url, limit=limit * 2,
                                     max_duration_hours=max_hours)
        _log.info("lde.channel_listed", url=channel_url, videos=len(videos))

        entries: list[CouncilEntry] = []
        processed = 0

        for video in videos:
            if processed >= limit:
                break
            if self.ingest_log.is_processed(video.video_id):
                _log.debug("lde.skip_already_processed", video_id=video.video_id)
                continue

            entry = self._process_video(video, use_whisper=use_whisper)
            if entry:
                entries.append(entry)
                n_rules = self._absorb_entry(entry)
                self.ingest_log.record(IngestRecord(
                    video_id=video.video_id,
                    channel=video.channel,
                    title=video.title,
                    ingested_at=_now_iso(),
                    doctrine_rules_created=n_rules,
                ))
                processed += 1

        self._save_all()
        _log.info("lde.channel_ingested", url=channel_url, processed=processed,
                  total_rules=len(self.doctrine.rules))
        return entries

    def ingest_all_channels(
        self,
        *,
        use_whisper: bool = False,
    ) -> list[CouncilEntry]:
        """Run ingest pipeline across all configured channels."""
        all_entries: list[CouncilEntry] = []
        for ch in self.channels:
            url = ch.get("url", "")
            if not url:
                continue
            entries = self.ingest_channel(url, use_whisper=use_whisper)
            all_entries.extend(entries)

        # Decay old rules after full cycle
        decay_days = self.doctrine_settings.get(
            "signal_decay_days", _DEFAULT_SIGNAL_DECAY_DAYS)
        pruned = self.doctrine.apply_decay(decay_days)
        if pruned:
            _log.info("lde.decay_pruned", count=pruned)
            self.doctrine.save()

        return all_entries

    def ingest_url(
        self,
        video_url: str,
        *,
        use_whisper: bool = False,
    ) -> CouncilEntry | None:
        """Ingest a single YouTube video by URL."""
        from councils.youtube.scraper import list_channel_videos

        # Extract video ID from URL
        video_id = _extract_video_id(video_url)
        if not video_id:
            _log.warning("lde.invalid_url", url=video_url)
            return None

        if self.ingest_log.is_processed(video_id):
            _log.info("lde.already_processed", video_id=video_id)
            return None

        # Build minimal VideoMeta from yt-dlp
        from councils.youtube.scraper import get_youtube_subtitles

        meta = _fetch_video_meta(video_url)
        if not meta:
            return None

        entry = self._process_video(meta, use_whisper=use_whisper)
        if entry:
            n_rules = self._absorb_entry(entry)
            self.ingest_log.record(IngestRecord(
                video_id=meta.video_id,
                channel=meta.channel,
                title=meta.title,
                ingested_at=_now_iso(),
                doctrine_rules_created=n_rules,
            ))
            self._save_all()
        return entry

    # ── processing ─────────────────────────────────────────────────

    @staticmethod
    def _process_video(
        video: VideoMeta,
        *,
        use_whisper: bool = False,
    ) -> CouncilEntry | None:
        """Run a video through the YouTube council pipeline."""
        from councils.youtube.pipeline import process_video
        try:
            return process_video(video, use_whisper=use_whisper)
        except Exception as exc:
            _log.warning("lde.process_video_failed",
                         video_id=video.video_id, error=str(exc))
            return None

    def _absorb_entry(self, entry: CouncilEntry) -> int:
        """Absorb a council entry: store in sandbox + generate doctrine rules."""
        # Store in sandbox
        sandbox_entry = SandboxEntry(
            entry_id=entry.meta.video_id,
            video_id=entry.meta.video_id,
            channel=entry.meta.channel,
            title=entry.meta.title,
            key_topics=entry.insights.key_topics,
            quotes=entry.insights.quotes,
            summary=entry.insights.summary,
            actionable_items=entry.insights.actionable_items,
            sentiment=entry.insights.sentiment,
            trust_score=entry.insights.trust_score,
            ingested_at=_now_iso(),
        )
        self.sandbox.add(sandbox_entry)

        # Convert insights → doctrine rules
        new_rules = insights_to_doctrine_rules(entry)
        added = 0
        for rule in new_rules:
            # Check if a similar rule already exists → reinforce
            existing = self.doctrine.find_matching(rule.text)
            if existing:
                existing.reinforce()
                _log.debug("lde.rule_reinforced", rule_id=existing.rule_id)
            else:
                self.doctrine.add_rule(rule)
                added += 1
                _log.debug("lde.rule_added", rule_id=rule.rule_id)

        return added

    # ── signal interface ───────────────────────────────────────────

    def get_doctrine_signal(self) -> float:
        """Return the current aggregate doctrine signal [-1, 1]."""
        return self.doctrine.doctrine_signal()

    def feed_alpha_engine(self) -> None:
        """Push the current doctrine signal into the alpha engine history."""
        from strategies.alpha_engine import record_council_observation
        signal = self.get_doctrine_signal()
        record_council_observation("lde_doctrine", signal)
        _log.info("lde.alpha_fed", signal=round(signal, 4))

    def get_polymarket_consensus(self) -> dict[str, Any]:
        """Fetch Polymarket trending markets and check for doctrine alignment.

        Returns a dict with market consensus and alignment stats.
        """
        import asyncio
        from councils.polymarket.scraper import scrape_trending_markets

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    markets = pool.submit(
                        asyncio.run, scrape_trending_markets(limit=20)
                    ).result(timeout=30)
            else:
                markets = asyncio.run(scrape_trending_markets(limit=20))
        except Exception as exc:
            _log.warning("lde.polymarket_failed", error=str(exc))
            return {"error": str(exc), "markets": []}

        # Cross-reference with active doctrine rules
        active = self.doctrine.active_rules
        rule_topics = set()
        for r in active:
            rule_topics.update(t.lower() for t in r.tags)

        aligned = []
        for m in markets:
            q_lower = m.question.lower()
            overlap = [t for t in rule_topics if t in q_lower]
            if overlap:
                aligned.append({
                    "question": m.question,
                    "yes_price": m.yes_price,
                    "volume": m.volume,
                    "matching_topics": overlap,
                })

        return {
            "total_markets": len(markets),
            "aligned_markets": len(aligned),
            "alignments": aligned[:10],
            "doctrine_signal": self.get_doctrine_signal(),
        }

    # ── persistence ────────────────────────────────────────────────

    def _save_all(self) -> None:
        self.doctrine.save()
        self.sandbox.save()
        self.ingest_log.save()

    # ── status / reporting ─────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return a comprehensive status summary."""
        active = self.doctrine.active_rules
        return {
            "total_rules": len(self.doctrine.rules),
            "active_rules": len(active),
            "sandbox_entries": len(self.sandbox.entries),
            "videos_processed": self.ingest_log.count,
            "doctrine_signal": round(self.get_doctrine_signal(), 4),
            "channels_configured": len(self.channels),
            "top_rules": [
                {"text": r.text[:100], "conviction": round(r.conviction, 3),
                 "signal": round(r.signal_value, 3)}
                for r in sorted(active, key=lambda r: r.conviction,
                                reverse=True)[:5]
            ],
        }


# ============================================================================
# VIDEO URL PARSING
# ============================================================================

def _extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    import re
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _fetch_video_meta(video_url: str) -> VideoMeta | None:
    """Fetch basic video metadata via yt-dlp."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        _log.error("yt-dlp not installed")
        return None

    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            if not info:
                return None
            return VideoMeta(
                video_id=info.get("id", ""),
                title=info.get("title", "Untitled"),
                channel=info.get("channel", info.get("uploader", "Unknown")),
                upload_date=info.get("upload_date", ""),
                duration_seconds=info.get("duration") or 0,
                description=info.get("description", ""),
                url=video_url,
                view_count=info.get("view_count") or 0,
                like_count=info.get("like_count") or 0,
                tags=info.get("tags") or [],
            )
    except Exception as exc:
        _log.warning("lde.fetch_meta_failed", url=video_url, error=str(exc))
        return None


# ============================================================================
# SCHEDULER INTEGRATION
# ============================================================================

def start_scheduler(engine: LivingDoctrineEngine | None = None) -> Any:
    """Start APScheduler for daily auto-ingest.

    Returns the scheduler instance.
    """
    from apscheduler.schedulers.background import BackgroundScheduler

    eng = engine or LivingDoctrineEngine()
    cron = eng.ingest_settings.get("schedule_cron", "0 8 * * *")
    parts = cron.split()

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        eng.ingest_all_channels,
        "cron",
        hour=int(parts[1]) if len(parts) > 1 else 8,
        minute=int(parts[0]) if parts else 0,
        id="lde_daily_ingest",
        replace_existing=True,
    )
    scheduler.start()
    _log.info("lde.scheduler_started", cron=cron)
    return scheduler
