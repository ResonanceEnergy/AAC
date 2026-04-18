from __future__ import annotations

"""Transcript analysis — extractive (no LLM) and LLM-enhanced."""

import json
import os
import re
from textwrap import dedent

import structlog

from councils.youtube.models import TranscriptSegment, VideoInsight, VideoMeta

_log = structlog.get_logger(__name__)


# ── Trust scoring ──────────────────────────────────────────────────────────────

def _compute_video_trust(
    meta: VideoMeta,
    segments: list[TranscriptSegment],
) -> dict[str, float]:
    """Compute trust score for a YouTube video analysis."""
    from councils.trust import (
        TrustScore, youtube_source_trust, evidence_score, freshness_score,
    )
    import time
    from datetime import datetime

    src = youtube_source_trust(
        view_count=meta.view_count,
        like_count=meta.like_count,
        has_transcript=len(segments) > 0,
        duration_seconds=int(meta.duration_seconds),
        channel=meta.channel,
    )

    # Evidence: based on transcript segments
    ev = evidence_score(len(segments), target=50)

    # Freshness from upload_date
    fresh = 1.0
    if meta.upload_date and len(meta.upload_date) == 8:
        try:
            dt = datetime.strptime(meta.upload_date, "%Y%m%d")
            age = time.time() - dt.timestamp()
            fresh = freshness_score(age, half_life=86400 * 3)  # 3-day half-life
        except ValueError:
            pass

    ts = TrustScore(
        source_reliability=src,
        data_freshness=fresh,
        evidence_volume=ev,
    )
    return ts.to_dict()


# ── Extractive helpers ─────────────────────────────────────────────────────────

def _extract_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 15]


def _score_sentence(sentence: str, title_words: set[str]) -> float:
    """Score a sentence for importance."""
    words = set(sentence.lower().split())
    score = 0.0
    score += len(words & title_words) * 2.0
    word_count = len(words)
    if 10 <= word_count <= 40:
        score += 1.0
    action_words = {"should", "must", "need", "important", "key", "critical",
                    "recommend", "strategy", "because", "therefore", "however"}
    score += len(words & action_words) * 1.5
    return score


def analyze_transcript(
    segments: list[TranscriptSegment],
    meta: VideoMeta,
) -> VideoInsight:
    """Extract insights from transcript segments without requiring an LLM."""
    full_text = " ".join(s.text for s in segments)
    sentences = _extract_sentences(full_text)
    title_words = set(meta.title.lower().split()) - {"the", "a", "an", "is", "to", "and", "of", "in"}

    scored = [(s, _score_sentence(s, title_words)) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Key topics via word frequency
    stop_words = {
        "the", "a", "an", "is", "to", "and", "of", "in", "it", "for",
        "that", "this", "with", "on", "are", "was", "be", "have", "has",
        "but", "not", "you", "we", "they", "he", "she", "or", "at", "by",
        "so", "if", "do", "my", "your", "its", "just", "like", "can",
        "what", "when", "how", "all", "will", "from", "up", "out", "no",
        "there", "about", "which", "one", "would", "been", "them", "then",
        "could", "more", "also", "into", "some", "than", "other", "very",
        "i", "me", "um", "uh", "yeah", "okay", "right", "well", "know",
        "think", "going", "really", "thing", "things", "get", "got", "lot",
        "much", "kind", "way", "actually", "here", "now",
    }
    word_freq: dict[str, int] = {}
    for word in full_text.lower().split():
        clean = re.sub(r'[^a-z]', '', word)
        if clean and len(clean) > 3 and clean not in stop_words:
            word_freq[clean] = word_freq.get(clean, 0) + 1

    top_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    key_topics = [t[0] for t in top_topics]

    quotes = [s[0] for s in scored[:5]]

    summary = ". ".join(s[0] for s in scored[:3]) + "." if scored else "No transcript available."

    action_kw = {"should", "must", "need", "recommend", "try", "consider", "make sure"}
    actionable = [
        s for s in sentences
        if any(kw in s.lower() for kw in action_kw)
    ][:5]

    # Sentiment
    positive_words = {"good", "great", "excellent", "amazing", "love", "best", "perfect",
                      "awesome", "fantastic", "wonderful", "impressive"}
    negative_words = {"bad", "terrible", "worst", "hate", "awful", "horrible", "poor",
                      "disappointing", "failure", "problem", "wrong"}
    words_lower = full_text.lower().split()
    pos_count = sum(1 for w in words_lower if w in positive_words)
    neg_count = sum(1 for w in words_lower if w in negative_words)

    if pos_count > neg_count * 2:
        sentiment = "positive"
    elif neg_count > pos_count * 2:
        sentiment = "negative"
    elif pos_count > 0 and neg_count > 0:
        sentiment = "mixed"
    else:
        sentiment = "neutral"

    trust = _compute_video_trust(meta, segments)

    return VideoInsight(
        title=meta.title,
        key_topics=key_topics,
        quotes=quotes,
        summary=summary,
        actionable_items=actionable,
        sentiment=sentiment,
        trust_score=trust,
    )


# ── LLM-enhanced analysis ─────────────────────────────────────────────────────

def analyze_with_llm(
    transcript_text: str,
    meta: VideoMeta,
    provider: str = "xai",
) -> VideoInsight | None:
    """Use xAI Grok or OpenAI for deeper analysis. Returns None if unavailable."""
    import httpx

    if provider == "xai":
        api_key = os.environ.get("XAI_API_KEY")
        base_url = "https://api.x.ai/v1"
        model = "grok-3-mini"
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = "https://api.openai.com/v1"
        model = "gpt-4o-mini"
    else:
        return None

    if not api_key:
        _log.warning("llm_key_missing", provider=provider)
        return None

    prompt = dedent(f"""\
        Analyze this YouTube video transcript and return a JSON object with:
        - "key_topics": list of 10 key topics (single words or short phrases)
        - "quotes": list of 5 most important/interesting quotes
        - "summary": 3-sentence summary
        - "actionable_items": list of actionable takeaways
        - "sentiment": one of "positive", "negative", "neutral", "mixed"

        Video: {meta.title}
        Channel: {meta.channel}

        Transcript (first 6000 chars):
        {transcript_text[:6000]}

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

        # Build pseudo-segments for trust computation
        pseudo_segs = [TranscriptSegment(start=0.0, duration=0.0, text=transcript_text[:6000])]
        trust = _compute_video_trust(meta, pseudo_segs)
        # LLM analysis gets a small boost for deeper reasoning
        trust["source_reliability"] = min(trust.get("source_reliability", 0.5) + 0.05, 1.0)

        return VideoInsight(
            title=meta.title,
            key_topics=data.get("key_topics", []),
            quotes=data.get("quotes", []),
            summary=data.get("summary", ""),
            actionable_items=data.get("actionable_items", []),
            sentiment=data.get("sentiment", "neutral"),
            trust_score=trust,
        )
    except Exception as exc:
        _log.warning("llm_analysis_failed", error=str(exc), provider=provider)
        return None
