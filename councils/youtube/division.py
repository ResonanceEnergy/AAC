from __future__ import annotations

"""YouTube Council Division — autonomous YouTube scanning as a DivisionProtocol.

Periodically scans configured channels for new videos, transcribes and
analyzes them, and publishes INTEL_UPDATE signals with the insights.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from divisions.division_protocol import DivisionProtocol, Signal, SignalType

from councils.youtube.models import CouncilEntry
from councils.youtube.pipeline import process_video
from councils.youtube.scraper import list_channel_videos

_log = structlog.get_logger(__name__)

# Track which videos have already been processed across cycles
_SEEN_DB = Path(__file__).parent / "output" / "youtube_seen.json"


class YouTubeCouncilDivision(DivisionProtocol):
    """Autonomous YouTube channel monitor.

    Each scan cycle:
    1. Scrapes configured channels for recent videos
    2. Skips already-processed videos (persistent seen-DB)
    3. Transcribes + analyzes new videos
    4. Publishes INTEL_UPDATE signals with key insights
    """

    def __init__(
        self,
        channels: list[str] | None = None,
        limit_per_channel: int = 5,
        max_hours: float = 4.0,
        use_llm: str | None = None,
    ) -> None:
        super().__init__(division_name="youtube_council")
        self._channels = channels or []
        self._limit = limit_per_channel
        self._max_hours = max_hours
        self._use_llm = use_llm
        self._seen: set[str] = set()
        self._total_processed: int = 0
        self._last_entries: list[CouncilEntry] = []

        self._load_seen()

    # ── Seen-DB persistence ────────────────────────────────────────────────

    def _load_seen(self) -> None:
        """Load previously processed video IDs from disk."""
        if _SEEN_DB.exists():
            try:
                data = json.loads(_SEEN_DB.read_text(encoding="utf-8"))
                self._seen = set(data.get("seen_ids", []))
            except (json.JSONDecodeError, KeyError):
                self._seen = set()

    def _save_seen(self) -> None:
        """Persist seen video IDs to disk."""
        _SEEN_DB.parent.mkdir(parents=True, exist_ok=True)
        _SEEN_DB.write_text(
            json.dumps({"seen_ids": sorted(self._seen), "updated": datetime.now(timezone.utc).isoformat()}),
            encoding="utf-8",
        )

    # ── Channel management ─────────────────────────────────────────────────

    def add_channel(self, channel_url: str) -> None:
        """Add a channel to the watch list."""
        if channel_url not in self._channels:
            self._channels.append(channel_url)
            _log.info("youtube.channel_added", channel=channel_url)

    def remove_channel(self, channel_url: str) -> None:
        """Remove a channel from the watch list."""
        if channel_url in self._channels:
            self._channels.remove(channel_url)

    # ── DivisionProtocol implementation ────────────────────────────────────

    async def scan(self) -> list[Signal]:
        """Scan all channels for new videos, process them, return signals."""
        if not self._channels:
            _log.debug("youtube.no_channels_configured")
            return []

        signals: list[Signal] = []
        new_entries: list[CouncilEntry] = []

        for channel in self._channels:
            try:
                videos = list_channel_videos(
                    channel, limit=self._limit, max_duration_hours=self._max_hours,
                )
            except Exception as exc:
                _log.warning("youtube.scrape_failed", channel=channel, error=str(exc))
                continue

            # Filter to unseen videos
            new_videos = [v for v in videos if v.video_id not in self._seen]
            if not new_videos:
                _log.debug("youtube.no_new_videos", channel=channel)
                continue

            _log.info("youtube.new_videos", channel=channel, count=len(new_videos))

            for video in new_videos:
                try:
                    entry = process_video(video, use_llm=self._use_llm)
                except Exception as exc:
                    _log.warning("youtube.process_failed", video_id=video.video_id, error=str(exc))
                    continue

                if not entry:
                    continue

                self._seen.add(video.video_id)
                self._total_processed += 1
                new_entries.append(entry)

                # Emit an INTEL_UPDATE signal per video
                trust_overall = entry.insights.trust_score.get("overall", 0.6)
                signals.append(Signal(
                    signal_type=SignalType.INTEL_UPDATE,
                    source_division=self.division_name,
                    timestamp=datetime.now(timezone.utc),
                    data={
                        "type": "youtube_insight",
                        "title": entry.meta.title,
                        "channel": entry.meta.channel,
                        "video_id": entry.meta.video_id,
                        "sentiment": entry.insights.sentiment,
                        "key_topics": entry.insights.key_topics[:5],
                        "summary": entry.insights.summary[:300],
                        "actionable_items": entry.insights.actionable_items[:3],
                        "markdown_path": entry.markdown_path,
                        "trust_score": entry.insights.trust_score,
                    },
                    confidence=trust_overall,
                    urgency=0,
                ))

        self._last_entries = new_entries
        if new_entries:
            self._save_seen()

        return signals

    async def report(self) -> dict[str, Any]:
        """Status report for dashboards."""
        return {
            "division": self.division_name,
            "channels_watched": len(self._channels),
            "channels": self._channels,
            "total_processed": self._total_processed,
            "seen_videos": len(self._seen),
            "last_batch_size": len(self._last_entries),
            "last_batch_titles": [e.meta.title for e in self._last_entries[:5]],
            "llm_provider": self._use_llm or "extractive",
        }
