from __future__ import annotations

"""Markdown formatter for Xai Council output."""

from datetime import datetime, timezone

from councils.xai.models import XInsight, XPost


def format_to_markdown(posts: list[XPost], insights: XInsight) -> str:
    """Format X posts + insights into clean Markdown."""
    lines = [
        f"# Xai Council Report -- {insights.source}",
        "",
        f"**Processed**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Posts Analyzed**: {len(posts)}",
        f"**Sentiment**: {insights.sentiment_summary}",
        "",
        "---",
        "",
        "## Key Themes",
        "",
    ]
    for theme in insights.key_themes:
        lines.append(f"- {theme}")

    lines.extend(["", "## Emerging Topics", ""])
    for topic in insights.emerging_topics:
        lines.append(f"- {topic}")

    lines.extend(["", "## Engagement Highlights", ""])
    for h in insights.engagement_highlights:
        lines.append(f"- {h}")

    if insights.actionable_items:
        lines.extend(["", "## Actionable Items", ""])
        for item in insights.actionable_items:
            lines.append(f"- [ ] {item}")

    lines.extend(["", "## Notable Posts", ""])
    for post_text in insights.notable_posts:
        lines.append(f"> {post_text}")
        lines.append("")

    lines.extend(["---", "", "## All Posts", ""])
    for p in posts:
        lines.append(f"### @{p.author} -- {p.created_at}")
        lines.append(f"Likes: {p.likes} | Reposts: {p.reposts} | Replies: {p.replies}")
        lines.append("")
        lines.append(p.text)
        if p.url:
            lines.append(f"[Link]({p.url})")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(f"*Processed by Xai Council -- {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    return "\n".join(lines)
