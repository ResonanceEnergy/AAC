from __future__ import annotations

"""Data models for Xai Council."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class XPost:
    """A single X/Twitter post."""

    post_id: str
    author: str
    text: str
    created_at: str
    url: str
    likes: int = 0
    reposts: int = 0
    replies: int = 0
    views: int = 0
    media_urls: list[str] = field(default_factory=list)
    is_reply: bool = False
    is_repost: bool = False


@dataclass
class XInsight:
    """Analyzed insight from X posts."""

    source: str
    key_themes: list[str]
    notable_posts: list[str]
    sentiment_summary: str
    engagement_highlights: list[str]
    emerging_topics: list[str]
    actionable_items: list[str]
    trust_score: dict[str, Any] = field(default_factory=dict)


@dataclass
class CouncilEntry:
    """Complete processed entry for an X analysis session."""

    posts: list[XPost]
    insights: XInsight
    markdown_path: str
    processed_at: str
