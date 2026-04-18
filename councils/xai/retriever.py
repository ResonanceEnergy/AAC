from __future__ import annotations

"""X post retrieval via Grok API (workaround for missing X_BEARER_TOKEN)."""

import json
import os
import re
from textwrap import dedent

import httpx
import structlog

from councils.xai.models import XPost

_log = structlog.get_logger(__name__)


def _get_api_config(provider: str) -> tuple[str, str, str] | None:
    """Return (base_url, api_key, model) for the given provider."""
    if provider == "xai":
        key = os.environ.get("XAI_API_KEY")
        if key:
            return ("https://api.x.ai/v1", key, "grok-3-mini")
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return ("https://api.openai.com/v1", key, "gpt-4o-mini")
    return None


def search_x_via_grok(
    query: str,
    provider: str = "xai",
    max_posts: int = 30,
) -> list[XPost]:
    """Ask Grok to retrieve and summarize recent X posts matching a query.

    Since direct X API requires paid tier (broken per API inventory),
    we use Grok's built-in web/X search capability to find posts.
    """
    config = _get_api_config(provider)
    if not config:
        _log.warning("api_key_missing", provider=provider)
        return []

    base_url, api_key, model = config

    prompt = dedent(f"""\
        Search X/Twitter for recent posts about: {query}
        Return up to {max_posts} posts as a JSON array. Each post should have:
        - "post_id": string (tweet ID or "unknown")
        - "author": string (handle without @)
        - "text": string (full post text)
        - "created_at": string (date if known, else "recent")
        - "likes": number (estimate if unknown, use 0)
        - "reposts": number
        - "replies": number
        - "is_reply": boolean
        - "is_repost": boolean

        Return ONLY the JSON array, no markdown fences or extra text.
        Focus on substantive posts, skip spam/ads.
    """)

    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        content = re.sub(r'^```json\s*', '', content.strip())
        content = re.sub(r'\s*```$', '', content.strip())
        raw_posts = json.loads(content)

        posts = []
        for p in raw_posts:
            author = str(p.get("author", "unknown")).lstrip("@")
            posts.append(XPost(
                post_id=str(p.get("post_id", "unknown")),
                author=author,
                text=str(p.get("text", "")),
                created_at=str(p.get("created_at", "recent")),
                url=f"https://x.com/{author}/status/{p.get('post_id', '')}" if p.get("post_id") != "unknown" else "",
                likes=int(p.get("likes", 0)),
                reposts=int(p.get("reposts", 0)),
                replies=int(p.get("replies", 0)),
                is_reply=bool(p.get("is_reply", False)),
                is_repost=bool(p.get("is_repost", False)),
            ))
        _log.info("grok_search_done", posts=len(posts), query=query)
        return posts

    except Exception as exc:
        _log.warning("grok_search_failed", error=str(exc))
        return []


def get_user_posts_via_grok(
    username: str,
    provider: str = "xai",
    max_posts: int = 30,
) -> list[XPost]:
    """Retrieve recent posts from a specific X user via Grok."""
    return search_x_via_grok(f"from:@{username}", provider=provider, max_posts=max_posts)
