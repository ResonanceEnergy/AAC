from __future__ import annotations

"""Markdown formatter for YouTube Council output."""

from datetime import datetime, timezone

from councils.youtube.models import TranscriptSegment, VideoInsight, VideoMeta


def _fmt_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_to_markdown(
    meta: VideoMeta,
    segments: list[TranscriptSegment],
    insights: VideoInsight,
) -> str:
    """Format a video + transcript + insights into clean Markdown."""
    duration_m = meta.duration_seconds // 60
    duration_s = meta.duration_seconds % 60

    lines = [
        f"# {meta.title}",
        "",
        f"**Channel**: {meta.channel}",
        f"**Date**: {meta.upload_date}",
        f"**Duration**: {duration_m}m {duration_s}s",
        f"**URL**: {meta.url}",
        f"**Views**: {meta.view_count:,}  |  **Likes**: {meta.like_count:,}",
        "",
        "---",
        "",
        "## Insights",
        "",
        f"**Sentiment**: {insights.sentiment}",
        "",
        "### Key Topics",
        "",
    ]
    for topic in insights.key_topics:
        lines.append(f"- {topic}")

    lines.extend(["", "### Summary", "", insights.summary, ""])

    if insights.quotes:
        lines.extend(["### Notable Quotes", ""])
        for q in insights.quotes:
            lines.append(f"> {q}")
            lines.append("")

    if insights.actionable_items:
        lines.extend(["### Actionable Items", ""])
        for item in insights.actionable_items:
            lines.append(f"- [ ] {item}")
        lines.append("")

    lines.extend(["---", "", "## Full Transcript", ""])

    for seg in segments:
        ts = _fmt_timestamp(seg.start)
        lines.append(f"**[{ts}]** {seg.text}")
        lines.append("")

    lines.extend([
        "---",
        f"*Processed by YouTube Council -- {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
    ])

    return "\n".join(lines)
