from __future__ import annotations

"""Xai Council Division — autonomous X/Twitter monitoring as a DivisionProtocol.

Periodically runs configured search queries and user monitors through
the Grok-powered X retrieval pipeline, analyzes posts, and publishes
INTEL_UPDATE signals with market-relevant insights.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from divisions.division_protocol import DivisionProtocol, Signal, SignalType

from councils.xai.pipeline import run_xai_council
from councils.xai.models import CouncilEntry

_log = structlog.get_logger(__name__)

_SEEN_DB = Path(__file__).parent / "output" / "xai_seen.json"


class XaiCouncilDivision(DivisionProtocol):
    """Autonomous X/Twitter intelligence monitor.

    Each scan cycle:
    1. Runs configured search queries and user monitors through Grok API
    2. Tracks previously-seen post IDs to avoid duplicate alerts
    3. Publishes INTEL_UPDATE signals with themes and actionable items
    """

    def __init__(
        self,
        search_queries: list[str] | None = None,
        users_to_follow: list[str] | None = None,
        max_posts: int = 10,
        provider: str = "xai",
    ) -> None:
        super().__init__(division_name="xai_council")
        self._queries = search_queries or []
        self._users = users_to_follow or []
        self._max_posts = max_posts
        self._provider = provider
        self._seen_post_ids: set[str] = set()
        self._total_scans: int = 0
        self._total_posts: int = 0
        self._last_entries: list[CouncilEntry] = []

        self._load_seen()

    # ── Seen-DB persistence ────────────────────────────────────────────────

    def _load_seen(self) -> None:
        if _SEEN_DB.exists():
            try:
                data = json.loads(_SEEN_DB.read_text(encoding="utf-8"))
                self._seen_post_ids = set(data.get("seen_ids", []))
            except (json.JSONDecodeError, KeyError):
                self._seen_post_ids = set()

    def _save_seen(self) -> None:
        _SEEN_DB.parent.mkdir(parents=True, exist_ok=True)
        _SEEN_DB.write_text(
            json.dumps({
                "seen_ids": sorted(self._seen_post_ids),
                "updated": datetime.now(timezone.utc).isoformat(),
            }),
            encoding="utf-8",
        )

    # ── Watch-list management ──────────────────────────────────────────────

    def add_query(self, query: str) -> None:
        if query not in self._queries:
            self._queries.append(query)
            _log.info("xai.query_added", query=query)

    def remove_query(self, query: str) -> None:
        if query in self._queries:
            self._queries.remove(query)

    def add_user(self, user: str) -> None:
        if user not in self._users:
            self._users.append(user)
            _log.info("xai.user_added", user=user)

    def remove_user(self, user: str) -> None:
        if user in self._users:
            self._users.remove(user)

    # ── DivisionProtocol implementation ────────────────────────────────────

    async def scan(self) -> list[Signal]:
        """Run all queries and user monitors, return signals for new insights."""
        if not self._queries and not self._users:
            _log.debug("xai.no_monitors_configured")
            return []

        signals: list[Signal] = []
        new_entries: list[CouncilEntry] = []

        # Process search queries
        for query in self._queries:
            try:
                entry = run_xai_council(
                    search_query=query,
                    provider=self._provider,
                    max_posts=self._max_posts,
                )
            except Exception as exc:
                _log.warning("xai.query_failed", query=query, error=str(exc))
                continue

            if not entry:
                continue

            self._total_scans += 1
            new_post_ids = {p.post_id for p in entry.posts if p.post_id} - self._seen_post_ids
            if not new_post_ids:
                _log.debug("xai.no_new_posts", query=query)
                continue

            self._seen_post_ids.update(new_post_ids)
            self._total_posts += len(new_post_ids)
            new_entries.append(entry)

            signals.append(self._entry_to_signal(entry, source_label=f"query:{query}"))

        # Process user monitors
        for user in self._users:
            try:
                entry = run_xai_council(
                    user=user,
                    provider=self._provider,
                    max_posts=self._max_posts,
                )
            except Exception as exc:
                _log.warning("xai.user_failed", user=user, error=str(exc))
                continue

            if not entry:
                continue

            self._total_scans += 1
            new_post_ids = {p.post_id for p in entry.posts if p.post_id} - self._seen_post_ids
            if not new_post_ids:
                continue

            self._seen_post_ids.update(new_post_ids)
            self._total_posts += len(new_post_ids)
            new_entries.append(entry)

            signals.append(self._entry_to_signal(entry, source_label=f"user:{user}"))

        self._last_entries = new_entries
        if new_entries:
            self._save_seen()

        return signals

    def _entry_to_signal(self, entry: CouncilEntry, source_label: str) -> Signal:
        """Convert a council entry to an INTEL_UPDATE signal."""
        trust_overall = entry.insights.trust_score.get("overall", 0.5)
        return Signal(
            signal_type=SignalType.INTEL_UPDATE,
            source_division=self.division_name,
            timestamp=datetime.now(timezone.utc),
            data={
                "type": "xai_insight",
                "source": source_label,
                "post_count": len(entry.posts),
                "themes": entry.insights.key_themes[:5] if entry.insights.key_themes else [],
                "sentiment": entry.insights.sentiment_summary,
                "emerging_topics": entry.insights.emerging_topics[:3] if entry.insights.emerging_topics else [],
                "actionable_items": entry.insights.actionable_items[:3] if entry.insights.actionable_items else [],
                "markdown_path": entry.markdown_path,
                "trust_score": entry.insights.trust_score,
            },
            confidence=trust_overall,
            urgency=0,
        )

    async def report(self) -> dict[str, Any]:
        """Status report for dashboards."""
        return {
            "division": self.division_name,
            "queries": self._queries,
            "users": self._users,
            "total_scans": self._total_scans,
            "total_posts_seen": self._total_posts,
            "seen_post_ids": len(self._seen_post_ids),
            "last_batch_size": len(self._last_entries),
            "provider": self._provider,
        }
