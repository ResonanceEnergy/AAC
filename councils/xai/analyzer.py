from __future__ import annotations

"""Post analysis -- extractive (no LLM) and LLM-enhanced."""

import json
import os
import re
from textwrap import dedent

import httpx
import structlog

from councils.xai.models import XInsight, XPost

_log = structlog.get_logger(__name__)


# ── Trust scoring ──────────────────────────────────────────────────────────────

def _compute_xai_trust(
    posts: list[XPost],
    provider: str,
    is_llm_synthesized: bool = True,
) -> dict[str, float]:
    """Compute trust score for X/Grok analysis."""
    from councils.trust import TrustScore, xai_source_trust, evidence_score

    src = xai_source_trust(
        post_count=len(posts),
        provider=provider,
        is_llm_synthesized=is_llm_synthesized,
    )
    ev = evidence_score(len(posts), target=30)

    ts = TrustScore(
        source_reliability=src,
        data_freshness=0.5 if is_llm_synthesized else 0.9,
        evidence_volume=ev,
    )
    return ts.to_dict()


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


def analyze_posts_extractive(posts: list[XPost], source: str) -> XInsight:
    """Extractive analysis of X posts (no LLM required)."""
    all_text = " ".join(p.text for p in posts)

    stop_words = {"the", "a", "an", "is", "to", "and", "of", "in", "it", "for",
                  "that", "this", "with", "on", "are", "was", "be", "have", "but",
                  "not", "you", "we", "they", "or", "at", "by", "so", "if", "do",
                  "my", "your", "just", "like", "can", "i", "me", "rt", "https",
                  "http", "co", "t", "s", "re", "ve"}
    word_freq: dict[str, int] = {}
    for word in re.findall(r'\b[a-z]{4,}\b', all_text.lower()):
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    themes = [w for w, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]]

    sorted_posts = sorted(posts, key=lambda p: p.likes + p.reposts * 2, reverse=True)
    notable = [p.text[:200] for p in sorted_posts[:5]]

    total_likes = sum(p.likes for p in posts)
    total_reposts = sum(p.reposts for p in posts)
    engagement = [
        f"Total posts analyzed: {len(posts)}",
        f"Total likes: {total_likes:,}",
        f"Total reposts: {total_reposts:,}",
        f"Avg likes/post: {total_likes // max(len(posts), 1)}",
    ]

    pos_words = {"good", "great", "love", "best", "amazing", "bullish", "winning", "fire"}
    neg_words = {"bad", "terrible", "hate", "worst", "bearish", "crash", "scam", "dump"}
    words_l = all_text.lower().split()
    pos_c = sum(1 for w in words_l if w in pos_words)
    neg_c = sum(1 for w in words_l if w in neg_words)
    sentiment = "positive" if pos_c > neg_c * 2 else "negative" if neg_c > pos_c * 2 else "mixed" if pos_c and neg_c else "neutral"

    trust = _compute_xai_trust(posts, provider="xai", is_llm_synthesized=True)

    return XInsight(
        source=source,
        key_themes=themes,
        notable_posts=notable,
        sentiment_summary=sentiment,
        engagement_highlights=engagement,
        emerging_topics=themes[:5],
        actionable_items=[],
        trust_score=trust,
    )


def analyze_posts_with_llm(
    posts: list[XPost],
    source: str,
    provider: str = "xai",
) -> XInsight | None:
    """Deep analysis of X posts using xAI Grok or OpenAI."""
    config = _get_api_config(provider)
    if not config:
        return None

    base_url, api_key, model = config

    posts_text = "\n---\n".join(
        f"@{p.author} ({p.created_at}) [Likes:{p.likes}]: {p.text}"
        for p in posts[:40]
    )

    prompt = dedent(f"""\
        Analyze these X/Twitter posts and return a JSON object with:
        - "key_themes": list of 10 dominant themes/topics
        - "notable_posts": list of 5 most significant post excerpts
        - "sentiment_summary": overall sentiment (positive/negative/neutral/mixed)
        - "engagement_highlights": list of 3 engagement observations
        - "emerging_topics": list of 5 emerging/trending topics
        - "actionable_items": list of actionable insights or things to track

        Source: {source}
        Posts:
        {posts_text[:8000]}

        Return ONLY valid JSON, no markdown fences.
    """)

    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        content = re.sub(r'^```json\s*', '', content.strip())
        content = re.sub(r'\s*```$', '', content.strip())
        data = json.loads(content)

        trust = _compute_xai_trust(posts, provider=provider, is_llm_synthesized=True)

        return XInsight(
            source=source,
            key_themes=data.get("key_themes", []),
            notable_posts=data.get("notable_posts", []),
            sentiment_summary=data.get("sentiment_summary", "neutral"),
            engagement_highlights=data.get("engagement_highlights", []),
            emerging_topics=data.get("emerging_topics", []),
            actionable_items=data.get("actionable_items", []),
            trust_score=trust,
        )
    except Exception as exc:
        _log.warning("llm_analysis_failed", error=str(exc), provider=provider)
        return None
