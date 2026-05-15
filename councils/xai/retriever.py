from __future__ import annotations

"""X post retrieval via Grok API (workaround for missing X_BEARER_TOKEN)."""

import json
import os
import re
import time
from textwrap import dedent

import httpx
import structlog

from councils.xai.models import XPost

_log = structlog.get_logger(__name__)

# Module-level circuit breaker: when Grok consistently rejects calls (HTTP
# 400/401/403 — bad model, bad key, bad schema) the dashboard otherwise spams
# one warning per query per minute. Trip after a few consecutive failures and
# stay quiet for an hour.
_FAILURE_THRESHOLD = 3
_COOLDOWN_SECONDS = 3600
_consecutive_failures: dict[str, int] = {}
_suppressed_until: dict[str, float] = {}
_suppression_logged: dict[str, bool] = {}


def _is_suppressed(provider: str) -> bool:
    until = _suppressed_until.get(provider, 0.0)
    if until and time.time() < until:
        if not _suppression_logged.get(provider, False):
            _log.error(
                "grok_circuit_open",
                provider=provider,
                cooldown_s=_COOLDOWN_SECONDS,
                hint="check XAI_API_KEY / model name; suppressing further calls",
            )
            _suppression_logged[provider] = True
        return True
    if until and time.time() >= until:
        _suppressed_until.pop(provider, None)
        _suppression_logged.pop(provider, None)
        _consecutive_failures[provider] = 0
    return False


def _record_failure(provider: str) -> None:
    n = _consecutive_failures.get(provider, 0) + 1
    _consecutive_failures[provider] = n
    if n >= _FAILURE_THRESHOLD:
        _suppressed_until[provider] = time.time() + _COOLDOWN_SECONDS
        _suppression_logged[provider] = False


def _record_success(provider: str) -> None:
    _consecutive_failures[provider] = 0
    _suppressed_until.pop(provider, None)
    _suppression_logged.pop(provider, None)


def _env_key(name: str) -> str:
    """Read an env var, preferring repo `.env` (process env may carry stale keys)."""
    try:
        from pathlib import Path

        from dotenv import dotenv_values

        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            v = dotenv_values(str(env_path))
            k = (v.get(name) or "").strip()
            if k:
                return k
    except ImportError:
        pass
    return (os.environ.get(name) or "").strip()


def _get_api_config(provider: str) -> tuple[str, str, str] | None:
    """Return (base_url, api_key, model) for the given provider.

    For xai we use grok-4 because the Responses API + x_search/web_search
    tools require grok-4 (grok-3-mini does not support built-in search tools).
    """
    if provider == "xai":
        key = _env_key("XAI_API_KEY") or _env_key("GROK_API_KEY")
        if key:
            return ("https://api.x.ai/v1", key, "grok-4")
    elif provider == "openai":
        key = _env_key("OPENAI_API_KEY")
        if key:
            return ("https://api.openai.com/v1", key, "gpt-4o-mini")
    return None


def search_x_via_grok(
    query: str,
    provider: str = "xai",
    max_posts: int = 30,
) -> list[XPost]:
    """Retrieve recent X posts matching a query, backed by real search.

    For provider=xai this calls the xAI Responses API with the built-in
    `x_search` + `web_search` tools (grok-4). The legacy Chat Completions
    `search_parameters` field was deprecated mid-2026 (HTTP 410), and asking
    grok-3-mini to "search X" via plain chat completions was pure
    hallucination — there was no actual retrieval happening.

    For provider=openai we fall back to a plain chat-completions call which
    has no real X search; it is left in place only so the existing council
    plumbing keeps returning a list (typically empty).
    """
    config = _get_api_config(provider)
    if not config:
        _log.warning("api_key_missing", provider=provider)
        return []

    if _is_suppressed(provider):
        return []

    base_url, api_key, model = config

    if provider == "xai":
        return _search_x_via_responses(query, base_url, api_key, model, max_posts)
    return _search_x_via_chat(query, base_url, api_key, model, max_posts, provider)


_X_SEARCH_SYSTEM = (
    "You are a retrieval agent. Use the x_search and web_search tools to find "
    "recent, substantive posts on X/Twitter matching the user's query. Return "
    "ONLY a JSON array — no prose, no markdown fences."
)


def _x_search_prompt(query: str, max_posts: int) -> str:
    return dedent(f"""\
        Query: {query}
        Find up to {max_posts} recent X/Twitter posts. Skip ads/spam.
        Return a JSON array; each element MUST have these keys:
          "post_id" (string, tweet id if known else "unknown"),
          "author"  (handle without @),
          "text"    (full post text),
          "created_at" (ISO date if known, else "recent"),
          "likes" (int, 0 if unknown),
          "reposts" (int),
          "replies" (int),
          "is_reply" (bool),
          "is_repost" (bool).
        Return ONLY the JSON array.
    """)


def _parse_posts(content: str) -> list[XPost]:
    content = re.sub(r'^```json\s*', '', content.strip())
    content = re.sub(r'\s*```$', '', content.strip())
    # Tolerate text-before-JSON: grab the first [...] block
    if not content.startswith("["):
        m = re.search(r"\[.*\]", content, re.S)
        if m:
            content = m.group(0)
    raw_posts = json.loads(content)
    posts: list[XPost] = []
    for p in raw_posts:
        author = str(p.get("author", "unknown")).lstrip("@")
        pid = str(p.get("post_id", "unknown"))
        posts.append(XPost(
            post_id=pid,
            author=author,
            text=str(p.get("text", "")),
            created_at=str(p.get("created_at", "recent")),
            url=f"https://x.com/{author}/status/{pid}" if pid != "unknown" else "",
            likes=int(p.get("likes", 0) or 0),
            reposts=int(p.get("reposts", 0) or 0),
            replies=int(p.get("replies", 0) or 0),
            is_reply=bool(p.get("is_reply", False)),
            is_repost=bool(p.get("is_repost", False)),
        ))
    return posts


def _search_x_via_responses(
    query: str,
    base_url: str,
    api_key: str,
    model: str,
    max_posts: int,
) -> list[XPost]:
    """xAI Responses API path with real x_search + web_search tools."""
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": _X_SEARCH_SYSTEM},
            {"role": "user", "content": _x_search_prompt(query, max_posts)},
        ],
        "tools": [{"type": "x_search"}, {"type": "web_search"}],
        "max_tool_calls": 6,
    }
    try:
        resp = httpx.post(
            f"{base_url}/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("output_text") or "").strip()
        if not text:
            for item in data.get("output", []) or []:
                if item.get("type") == "message":
                    for c in item.get("content", []) or []:
                        if c.get("type") in ("output_text", "text"):
                            text = (text + "\n" + (c.get("text") or "")).strip()
        posts = _parse_posts(text) if text else []
        _log.info("grok_search_done", posts=len(posts), query=query, via="responses")
        _record_success("xai")
        return posts
    except (httpx.HTTPError, json.JSONDecodeError, ValueError, KeyError) as exc:
        _log.warning("grok_search_failed", error=str(exc), via="responses")
        _record_failure("xai")
        return []


def _search_x_via_chat(
    query: str,
    base_url: str,
    api_key: str,
    model: str,
    max_posts: int,
    provider: str,
) -> list[XPost]:
    """Fallback Chat Completions path (no real search; openai only)."""
    prompt = _x_search_prompt(query, max_posts)
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
        posts = _parse_posts(content)
        _log.info("grok_search_done", posts=len(posts), query=query, via="chat")
        _record_success(provider)
        return posts
    except (httpx.HTTPError, json.JSONDecodeError, ValueError, KeyError) as exc:
        _log.warning("grok_search_failed", error=str(exc), via="chat")
        _record_failure(provider)
        return []


def get_user_posts_via_grok(
    username: str,
    provider: str = "xai",
    max_posts: int = 30,
) -> list[XPost]:
    """Retrieve recent posts from a specific X user via Grok."""
    return search_x_via_grok(f"from:@{username}", provider=provider, max_posts=max_posts)
