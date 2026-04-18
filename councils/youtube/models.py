from __future__ import annotations

"""Data models for YouTube Council."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VideoMeta:
    """Metadata for a single YouTube video."""

    video_id: str
    title: str
    channel: str
    upload_date: str
    duration_seconds: int
    description: str
    url: str
    view_count: int = 0
    like_count: int = 0
    tags: list[str] = field(default_factory=list)


@dataclass
class TranscriptSegment:
    """A timestamped segment of transcript."""

    start: float
    end: float
    text: str


@dataclass
class VideoInsight:
    """Extracted insights from a video."""

    title: str
    key_topics: list[str]
    quotes: list[str]
    summary: str
    actionable_items: list[str]
    sentiment: str  # "positive", "negative", "neutral", "mixed"
    trust_score: dict[str, Any] = field(default_factory=dict)


@dataclass
class CouncilEntry:
    """Complete processed entry for a single video."""

    meta: VideoMeta
    transcript: list[TranscriptSegment]
    insights: VideoInsight
    markdown_path: str
    processed_at: str
