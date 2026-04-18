from __future__ import annotations

"""
War Room Council Feeds — YouTube + X → Indicators + Scenarios
=============================================================

Bridge between the council scrapers (councils/youtube, councils/xai) and the
War Room engine.  Converts VideoInsight / XInsight outputs into quantitative
signals that update IndicatorState, evaluate scenario trigger_indicators, and
bump milestone tracking.

Usage (async):
    from strategies.war_room_council_feeds import (
        fetch_youtube_intel, fetch_x_intel, apply_council_to_indicators,
    )

    yt_result = await fetch_youtube_intel()
    x_result  = await fetch_x_intel()
    apply_council_to_indicators(yt_result, x_result, live_feed_result)

Wired automatically when war_room_live_feeds.fetch_all_live_data() runs.
"""

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from strategies.alpha_engine import (
    alpha_combine_councils,
    record_council_observation,
)

logger = logging.getLogger("WarRoomCouncilFeeds")

STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "war_engine"

# ============================================================================
# CRISIS KEYWORD TAXONOMY — maps themes/topics to scenario & indicator domains
# ============================================================================

# Keywords → scenario codes (from storm_lifeboat/core.py SCENARIO_MAP)
_SCENARIO_KEYWORDS: dict[str, list[str]] = {
    # Geopolitical / Oil
    "hormuz":        ["hormuz_blockade"],
    "strait":        ["hormuz_blockade"],
    "iran":          ["hormuz_blockade", "iran_deal", "iran_nuclear", "petrodollar_spiral"],
    "oil":           ["hormuz_blockade", "petrodollar_spiral", "energy_hemisphere"],
    "opec":          ["hormuz_blockade", "petrodollar_spiral"],
    "petrodollar":   ["petrodollar_spiral"],
    "sanctions":     ["iran_nuclear", "petrodollar_spiral", "cuba_embargo"],
    "nuclear":       ["iran_nuclear", "nuclear_americas"],
    # Credit / Financial
    "debt":          ["us_debt_crisis", "credit_cascade"],
    "treasury":      ["us_debt_crisis"],
    "default":       ["us_debt_crisis", "credit_cascade"],
    "credit":        ["us_debt_crisis", "credit_cascade"],
    "banking":       ["eu_banking_crisis"],
    "bank":          ["eu_banking_crisis", "japan_crisis"],
    "recession":     ["credit_cascade", "soft_landing"],
    "inflation":     ["monetary_reset", "us_debt_crisis"],
    "deflation":     ["japan_crisis", "soft_landing"],
    "fed":           ["monetary_reset", "soft_landing"],
    "rates":         ["monetary_reset", "soft_landing"],
    "yield":         ["us_debt_crisis", "japan_crisis"],
    "yen":           ["japan_crisis"],
    "boj":           ["japan_crisis"],
    # Tech / AI
    "bubble":        ["ai_bubble_burst"],
    "nvidia":        ["ai_bubble_burst"],
    "semiconductor": ["ai_bubble_burst", "rare_earth_fortress"],
    "chip":          ["ai_bubble_burst", "rare_earth_fortress"],
    "starlink":      ["starlink_dominance"],
    "fusion":        ["fusion_rollout"],
    # Geopolitical — other
    "taiwan":        ["taiwan_strait_crisis"],
    "china":         ["taiwan_strait_crisis", "rare_earth_fortress"],
    "nato":          ["nato_exit"],
    "europe":        ["europe_abandon", "eu_banking_crisis"],
    "canada":        ["canada_decline", "north_border"],
    "greenland":     ["greenland_acquisition"],
    "panama":        ["panama_canal"],
    "arctic":        ["arctic_expand"],
    "venezuela":     ["venezuela_regime"],
    "lithium":       ["lithium_triangle"],
    "cuba":          ["cuba_embargo"],
    "brazil":        ["brazil_argentina"],
    "argentina":     ["brazil_argentina"],
    "migration":     ["migration_security"],
    "border":        ["border_military", "north_border", "migration_security"],
    "military":      ["border_military", "military_redeploy"],
    "latam":         ["latam_lockin", "hemisphere_pivot"],
    "hemisphere":    ["hemisphere_pivot", "energy_hemisphere"],
    "monroe":        ["hemisphere_pivot", "latam_lockin"],
    # Crypto / DeFi
    "bitcoin":       ["defi_liquidation_cascade"],
    "crypto":        ["defi_liquidation_cascade"],
    "defi":          ["defi_liquidation_cascade"],
    "stablecoin":    ["defi_liquidation_cascade"],
    "tether":        ["defi_liquidation_cascade"],
    "ethereum":      ["defi_liquidation_cascade"],
    # Commodities
    "gold":          ["commodity_supercycle"],
    "silver":        ["commodity_supercycle"],
    "commodity":     ["commodity_supercycle"],
    "rare earth":    ["rare_earth_fortress"],
    # Real estate / Other
    "commercial real estate": ["cre_meltdown"],
    "real estate":   ["cre_meltdown"],
    "office":        ["cre_meltdown"],
    "pandemic":      ["pandemic_v2"],
    "virus":         ["pandemic_v2"],
    "election":      ["election_chaos"],
    "tariff":        ["us_debt_crisis", "hemisphere_pivot"],
    "trade war":     ["hemisphere_pivot", "rare_earth_fortress"],
    # Macro
    "food":          ["food_crisis"],
    "famine":        ["food_crisis"],
    "climate":       ["climate_shock"],
    "weather":       ["climate_shock", "food_crisis"],
    "drought":       ["food_crisis", "climate_shock"],
    # Catch-all crisis
    "crash":         ["black_swan"],
    "collapse":      ["black_swan", "credit_cascade"],
    "black swan":    ["black_swan"],
    "war":           ["hormuz_blockade", "taiwan_strait_crisis"],
}

# Sentiment words for quantitative scoring of council text
_BEARISH_WORDS = frozenset({
    "crash", "collapse", "crisis", "recession", "bearish", "panic",
    "fear", "sell", "dump", "plunge", "freefall", "meltdown", "default",
    "bankrupt", "liquidation", "contagion", "stagflation", "downgrade",
    "warning", "risk", "threat", "escalation", "sanctions", "war",
    "tariff", "retaliation", "shutdown", "layoff", "unemployment",
    "overvalued", "bubble", "correction", "reversal", "negative",
    "disaster", "catastrophe", "devastation", "turmoil",
})

_BULLISH_WORDS = frozenset({
    "rally", "boom", "bullish", "optimistic", "recovery", "growth",
    "buy", "surge", "breakout", "record", "high", "upgrade",
    "expansion", "hiring", "dovish", "easing", "peace", "deal",
    "agreement", "partnership", "innovation", "breakthrough",
    "positive", "strong", "momentum", "opportunity",
})


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class CouncilFeedResult:
    """Aggregated intelligence from YouTube + X councils."""
    timestamp: str = ""
    # YouTube
    yt_videos_processed: int = 0
    yt_sentiment_score: float = 0.5   # 0=bearish, 1=bullish
    yt_severity_score: float = 0.0    # 0=calm, 1=crisis
    yt_key_topics: list[str] = field(default_factory=list)
    yt_actionable_items: list[str] = field(default_factory=list)
    yt_scenario_hits: dict[str, int] = field(default_factory=dict)
    yt_trust_score: float = 0.0
    # X/Twitter
    x_posts_analyzed: int = 0
    x_sentiment_score: float = 0.5
    x_severity_score: float = 0.0
    x_key_themes: list[str] = field(default_factory=list)
    x_emerging_topics: list[str] = field(default_factory=list)
    x_actionable_items: list[str] = field(default_factory=list)
    x_scenario_hits: dict[str, int] = field(default_factory=dict)
    x_trust_score: float = 0.0
    # Combined
    combined_sentiment: float = 0.5
    combined_severity: float = 0.0
    combined_trust: float = 0.0
    scenario_signals: dict[str, float] = field(default_factory=dict)
    alpha_signal: float | None = None
    alpha_weights: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.yt_videos_processed:
            parts.append(f"YT:{self.yt_videos_processed}vid sent={self.yt_sentiment_score:.2f} trust={self.yt_trust_score:.2f}")
        if self.x_posts_analyzed:
            parts.append(f"X:{self.x_posts_analyzed}posts sent={self.x_sentiment_score:.2f} trust={self.x_trust_score:.2f}")
        if self.scenario_signals:
            top = sorted(self.scenario_signals.items(), key=lambda x: -x[1])[:3]
            parts.append("scenarios=" + ",".join(f"{k}:{v:.0%}" for k, v in top))
        if self.combined_trust > 0:
            parts.append(f"trust={self.combined_trust:.2f}")
        if self.alpha_signal is not None:
            parts.append(f"alpha={self.alpha_signal:.3f}")
        return " | ".join(parts) if parts else "no council data"


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def _sentiment_from_text(text: str) -> float:
    """Score text 0 (bearish/fear) to 1 (bullish/greed)."""
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    bear = len(words & _BEARISH_WORDS)
    bull = len(words & _BULLISH_WORDS)
    total = bear + bull
    if total == 0:
        return 0.5
    return round(bull / total, 3)


def _severity_from_text(text: str) -> float:
    """Score text 0 (calm) to 1 (crisis-level) based on crisis keyword density."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return 0.0
    crisis_words = _BEARISH_WORDS | frozenset({"crisis", "war", "nuclear", "pandemic", "famine"})
    hits = sum(1 for w in words if w in crisis_words)
    density = hits / len(words)
    return round(min(density * 10, 1.0), 3)  # 10% crisis words = max severity


def _map_sentiment_label(label: str) -> float:
    """Convert sentiment labels to 0-1 numeric score."""
    mapping = {
        "positive": 0.75,
        "bullish": 0.85,
        "negative": 0.25,
        "bearish": 0.15,
        "mixed": 0.45,
        "neutral": 0.50,
    }
    return mapping.get(label.lower().strip(), 0.5)


def _extract_scenario_hits(topics: list[str], text: str = "") -> dict[str, int]:
    """Map topics/themes to scenario codes using keyword taxonomy."""
    hits: dict[str, int] = {}
    combined = " ".join(topics) + " " + text
    combined_lower = combined.lower()
    for keyword, scenarios in _SCENARIO_KEYWORDS.items():
        if keyword in combined_lower:
            for sc in scenarios:
                hits[sc] = hits.get(sc, 0) + 1
    return hits


# ============================================================================
# FETCH FUNCTIONS — called by war_room_live_feeds
# ============================================================================

# Default channels/queries for market intelligence
_DEFAULT_YT_CHANNELS: list[str] = []  # User adds via config or CLI
_DEFAULT_X_QUERIES: list[str] = [
    "stock market crash OR recession 2026 OR financial crisis",
    "oil price OR OPEC OR Iran sanctions",
    "bitcoin crash OR crypto regulation OR DeFi liquidation",
    "tariff trade war OR geopolitical risk",
    "VIX spike OR market volatility OR black swan",
]
_DEFAULT_X_USERS: list[str] = []


async def fetch_youtube_intel(
    channels: list[str] | None = None,
    limit_per_channel: int = 3,
    use_llm: str | None = None,
) -> CouncilFeedResult:
    """Fetch and analyze recent YouTube videos for war room intelligence.

    Runs synchronously in an executor since yt-dlp is blocking.
    """
    result = CouncilFeedResult(timestamp=datetime.now(timezone.utc).isoformat())
    target_channels = channels if channels is not None else _DEFAULT_YT_CHANNELS

    if not target_channels:
        logger.debug("No YouTube channels configured for war room")
        return result

    loop = asyncio.get_event_loop()

    try:
        def _run_yt() -> list[Any]:
            from councils.youtube.scraper import list_channel_videos
            from councils.youtube.pipeline import process_video

            entries = []
            for channel in target_channels:
                try:
                    videos = list_channel_videos(channel, limit=limit_per_channel)
                    for video in videos:
                        try:
                            entry = process_video(video, use_llm=use_llm)
                            if entry:
                                entries.append(entry)
                        except Exception as exc:
                            logger.warning("YouTube process_video failed: %s", exc)
                except Exception as exc:
                    logger.warning("YouTube list_channel failed: %s", exc)
            return entries

        entries = await loop.run_in_executor(None, _run_yt)

        if not entries:
            return result

        result.yt_videos_processed = len(entries)

        # Aggregate sentiment and topics
        all_topics: list[str] = []
        all_actions: list[str] = []
        sentiment_scores: list[float] = []
        severity_scores: list[float] = []
        trust_scores: list[float] = []
        all_scenario_hits: dict[str, int] = {}

        for entry in entries:
            ins = entry.insights
            all_topics.extend(ins.key_topics)
            all_actions.extend(ins.actionable_items)

            # Sentiment from label + text
            label_score = _map_sentiment_label(ins.sentiment)
            text_score = _sentiment_from_text(ins.summary)
            sentiment_scores.append((label_score + text_score) / 2)

            # Severity from summary
            severity_scores.append(_severity_from_text(ins.summary))

            # Trust score
            trust_scores.append(ins.trust_score.get("overall", 0.5) if ins.trust_score else 0.5)

            # Scenario mapping
            hits = _extract_scenario_hits(ins.key_topics, ins.summary)
            for sc, count in hits.items():
                all_scenario_hits[sc] = all_scenario_hits.get(sc, 0) + count

        # Trust-weighted sentiment averaging
        total_weight = sum(trust_scores)
        if total_weight > 0:
            result.yt_sentiment_score = round(
                sum(s * w for s, w in zip(sentiment_scores, trust_scores)) / total_weight, 3)
            result.yt_severity_score = round(
                sum(s * w for s, w in zip(severity_scores, trust_scores)) / total_weight, 3)
        else:
            result.yt_sentiment_score = round(sum(sentiment_scores) / len(sentiment_scores), 3)
            result.yt_severity_score = round(sum(severity_scores) / len(severity_scores), 3)
        result.yt_trust_score = round(sum(trust_scores) / len(trust_scores), 3)
        result.yt_key_topics = list(dict.fromkeys(all_topics))[:20]  # dedupe, keep order
        result.yt_actionable_items = all_actions[:10]
        result.yt_scenario_hits = all_scenario_hits

        logger.info("YouTube intel: %d videos, sentiment=%.2f, severity=%.2f, scenarios=%s",
                     result.yt_videos_processed, result.yt_sentiment_score,
                     result.yt_severity_score, list(all_scenario_hits.keys())[:5])

    except Exception as exc:
        result.errors.append(f"YouTube council error: {exc}")
        logger.warning("YouTube council feed failed: %s", exc)

    return result


async def fetch_x_intel(
    queries: list[str] | None = None,
    users: list[str] | None = None,
    provider: str = "xai",
    max_posts: int = 20,
) -> CouncilFeedResult:
    """Fetch and analyze X/Twitter posts for war room intelligence via Grok.

    Uses the xai council (Grok-powered search) since direct Twitter API
    is broken (HTTP 402). Falls back to extractive analysis if LLM fails.
    """
    result = CouncilFeedResult(timestamp=datetime.now(timezone.utc).isoformat())
    target_queries = queries if queries is not None else _DEFAULT_X_QUERIES
    target_users = users if users is not None else _DEFAULT_X_USERS

    if not target_queries and not target_users:
        logger.debug("No X queries/users configured for war room")
        return result

    loop = asyncio.get_event_loop()

    try:
        def _run_x() -> list[Any]:
            from councils.xai.pipeline import run_xai_council
            entries = []
            for query in target_queries:
                try:
                    entry = run_xai_council(
                        search_query=query, provider=provider, max_posts=max_posts,
                    )
                    if entry:
                        entries.append(entry)
                except Exception as exc:
                    logger.warning("X council query failed '%s': %s", query, exc)
            for user in target_users:
                try:
                    entry = run_xai_council(
                        user=user, provider=provider, max_posts=max_posts,
                    )
                    if entry:
                        entries.append(entry)
                except Exception as exc:
                    logger.warning("X council user failed '@%s': %s", user, exc)
            return entries

        entries = await loop.run_in_executor(None, _run_x)

        if not entries:
            return result

        # Aggregate
        all_themes: list[str] = []
        all_emerging: list[str] = []
        all_actions: list[str] = []
        sentiment_scores: list[float] = []
        severity_scores: list[float] = []
        trust_scores: list[float] = []
        total_posts = 0
        all_scenario_hits: dict[str, int] = {}

        for entry in entries:
            ins = entry.insights
            total_posts += len(entry.posts)
            all_themes.extend(ins.key_themes)
            all_emerging.extend(ins.emerging_topics)
            all_actions.extend(ins.actionable_items)

            # Sentiment from label + post text
            label_score = _map_sentiment_label(ins.sentiment_summary)
            post_text = " ".join(p.text for p in entry.posts[:30])
            text_score = _sentiment_from_text(post_text)
            sentiment_scores.append((label_score + text_score) / 2)

            # Severity from posts
            severity_scores.append(_severity_from_text(post_text))

            # Trust score
            trust_scores.append(ins.trust_score.get("overall", 0.4) if ins.trust_score else 0.4)

            # Scenario mapping
            hits = _extract_scenario_hits(ins.key_themes + ins.emerging_topics, post_text)
            for sc, count in hits.items():
                all_scenario_hits[sc] = all_scenario_hits.get(sc, 0) + count

        result.x_posts_analyzed = total_posts
        # Trust-weighted sentiment averaging
        total_weight = sum(trust_scores)
        if total_weight > 0:
            result.x_sentiment_score = round(
                sum(s * w for s, w in zip(sentiment_scores, trust_scores)) / total_weight, 3)
            result.x_severity_score = round(
                sum(s * w for s, w in zip(severity_scores, trust_scores)) / total_weight, 3)
        else:
            result.x_sentiment_score = round(sum(sentiment_scores) / len(sentiment_scores), 3)
            result.x_severity_score = round(sum(severity_scores) / len(severity_scores), 3)
        result.x_trust_score = round(sum(trust_scores) / len(trust_scores), 3)
        result.x_key_themes = list(dict.fromkeys(all_themes))[:20]
        result.x_emerging_topics = list(dict.fromkeys(all_emerging))[:10]
        result.x_actionable_items = all_actions[:10]
        result.x_scenario_hits = all_scenario_hits

        logger.info("X intel: %d posts, sentiment=%.2f, severity=%.2f, scenarios=%s",
                     result.x_posts_analyzed, result.x_sentiment_score,
                     result.x_severity_score, list(all_scenario_hits.keys())[:5])

    except Exception as exc:
        result.errors.append(f"X council error: {exc}")
        logger.warning("X council feed failed: %s", exc)

    return result


# ============================================================================
# APPLY TO INDICATORS + SCENARIOS
# ============================================================================

def apply_council_to_indicators(
    yt: CouncilFeedResult | None,
    x: CouncilFeedResult | None,
    live_result: Any = None,
) -> dict[str, Any]:
    """Apply council intelligence to update IndicatorState and scenario signals.

    Updates:
    - x_sentiment (blended with council X data — replaces broken Twitter API)
    - news_severity (blended with council severity signals)
    - scenario_signals dict with hit counts per scenario code

    Returns dict of changes applied for logging.
    """
    changes: dict[str, Any] = {}

    # ── Combine sentiment (trust-weighted) ─────────────────────────────
    sentiments: list[tuple[float, float]] = []  # (score, trust_weight)
    severities: list[tuple[float, float]] = []

    if yt and yt.yt_videos_processed > 0:
        sentiments.append((yt.yt_sentiment_score, max(yt.yt_trust_score, 0.1)))
        severities.append((yt.yt_severity_score, max(yt.yt_trust_score, 0.1)))

    if x and x.x_posts_analyzed > 0:
        sentiments.append((x.x_sentiment_score, max(x.x_trust_score, 0.1)))
        severities.append((x.x_severity_score, max(x.x_trust_score, 0.1)))

    if sentiments:
        total_w = sum(w for _, w in sentiments)
        combined_sentiment = round(sum(s * w for s, w in sentiments) / total_w, 3)
    else:
        combined_sentiment = 0.5

    if severities:
        total_w = sum(w for _, w in severities)
        combined_severity = round(sum(s * w for s, w in severities) / total_w, 3)
    else:
        combined_severity = 0.0

    combined_trust = 0.0
    trust_parts: list[float] = []
    if yt and yt.yt_trust_score > 0:
        trust_parts.append(yt.yt_trust_score)
    if x and x.x_trust_score > 0:
        trust_parts.append(x.x_trust_score)
    if trust_parts:
        combined_trust = round(sum(trust_parts) / len(trust_parts), 3)

    # ── Merge scenario hits ──────────────────────────────────────────
    merged_scenarios: dict[str, int] = {}
    if yt:
        for sc, count in yt.yt_scenario_hits.items():
            merged_scenarios[sc] = merged_scenarios.get(sc, 0) + count
    if x:
        for sc, count in x.x_scenario_hits.items():
            merged_scenarios[sc] = merged_scenarios.get(sc, 0) + count

    # Normalize to 0-1 signal strength
    max_hits = max(merged_scenarios.values()) if merged_scenarios else 1
    scenario_signals = {sc: round(min(count / max(max_hits, 1), 1.0), 3)
                        for sc, count in merged_scenarios.items()}

    # ── Patch LiveFeedResult if provided ─────────────────────────────
    if live_result is not None:
        # Council X sentiment replaces broken Twitter API (was HTTP 402)
        if x and x.x_posts_analyzed > 0:
            if live_result.x_sentiment_score is None or live_result.x_sentiment_score == 0.5:
                live_result.x_sentiment_score = combined_sentiment
                changes["x_sentiment_patched"] = combined_sentiment

        # Blend council severity into news severity
        if combined_severity > 0:
            existing = live_result.news_severity_score or 0.0
            blended = round(existing * 0.6 + combined_severity * 0.4, 3)
            live_result.news_severity_score = blended
            changes["news_severity_blended"] = blended

        # Store council metadata on the result for downstream
        live_result.council_scenario_signals = scenario_signals
        live_result.council_topics = (
            (yt.yt_key_topics[:10] if yt else []) +
            (x.x_key_themes[:10] if x else [])
        )
        live_result.council_emerging = x.x_emerging_topics[:5] if x else []

    changes["combined_sentiment"] = combined_sentiment
    changes["combined_severity"] = combined_severity
    changes["combined_trust"] = combined_trust
    changes["scenario_signals"] = scenario_signals

    # ── Alpha engine: record observations & combine ──────────────────
    if yt and yt.yt_videos_processed > 0:
        record_council_observation("yt_sentiment", yt.yt_sentiment_score)
        record_council_observation("yt_severity", yt.yt_severity_score)
    if x and x.x_posts_analyzed > 0:
        record_council_observation("x_sentiment", x.x_sentiment_score)
        record_council_observation("x_severity", x.x_severity_score)

    alpha_result = alpha_combine_councils()
    if alpha_result is not None:
        changes["alpha_signal"] = alpha_result.combined_signal
        changes["alpha_weights"] = alpha_result.weights
        logger.info(
            "Alpha engine: signal=%.4f weights=%s",
            alpha_result.combined_signal,
            {k: round(v, 3) for k, v in alpha_result.weights.items()},
        )

    return changes


def update_scenario_statuses(
    scenario_signals: dict[str, float],
    threshold_emerging: float = 0.2,
    threshold_active: float = 0.5,
) -> list[dict[str, str]]:
    """Update scenario statuses in storm_lifeboat based on council signals.

    Returns list of status transitions for logging.
    """
    transitions: list[dict[str, str]] = []

    try:
        from strategies.storm_lifeboat.core import SCENARIOS as SL_SCENARIOS, ScenarioStatus

        by_code = {sc.code.lower(): sc for sc in SL_SCENARIOS}

        for code, signal_strength in scenario_signals.items():
            sc = by_code.get(code)
            if not sc:
                continue

            old_status = sc.status.value if hasattr(sc.status, 'value') else str(sc.status)

            if signal_strength >= threshold_active and sc.status == ScenarioStatus.DORMANT:
                sc.status = ScenarioStatus.EMERGING
                transitions.append({
                    "scenario": sc.name, "code": code,
                    "from": old_status, "to": "EMERGING",
                    "signal": f"{signal_strength:.0%}",
                })
            elif signal_strength >= threshold_active and sc.status == ScenarioStatus.EMERGING:
                sc.status = ScenarioStatus.ACTIVE
                transitions.append({
                    "scenario": sc.name, "code": code,
                    "from": old_status, "to": "ACTIVE",
                    "signal": f"{signal_strength:.0%}",
                })
            elif signal_strength >= threshold_emerging and sc.status == ScenarioStatus.DORMANT:
                sc.status = ScenarioStatus.EMERGING
                transitions.append({
                    "scenario": sc.name, "code": code,
                    "from": old_status, "to": "EMERGING",
                    "signal": f"{signal_strength:.0%}",
                })

    except ImportError:
        logger.warning("storm_lifeboat.core not importable — skipping scenario updates")

    if transitions:
        logger.info("Scenario transitions: %s", transitions)
        _persist_scenario_transitions(transitions)

    return transitions


def _persist_scenario_transitions(transitions: list[dict[str, str]]) -> None:
    """Append scenario transitions to JSONL log."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / "scenario_transitions.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        for t in transitions:
            t["timestamp"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(t) + "\n")


def persist_council_snapshot(
    yt: CouncilFeedResult | None,
    x: CouncilFeedResult | None,
    changes: dict[str, Any],
) -> None:
    """Persist council feed snapshot to JSONL for trend analysis."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / "council_snapshots.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "yt_videos": yt.yt_videos_processed if yt else 0,
        "yt_sentiment": yt.yt_sentiment_score if yt else None,
        "yt_severity": yt.yt_severity_score if yt else None,
        "yt_trust": yt.yt_trust_score if yt else None,
        "yt_topics": yt.yt_key_topics[:10] if yt else [],
        "x_posts": x.x_posts_analyzed if x else 0,
        "x_sentiment": x.x_sentiment_score if x else None,
        "x_severity": x.x_severity_score if x else None,
        "x_trust": x.x_trust_score if x else None,
        "x_themes": x.x_key_themes[:10] if x else [],
        "x_emerging": x.x_emerging_topics[:5] if x else [],
        "combined_trust": changes.get("combined_trust", 0.0),
        "scenario_signals": changes.get("scenario_signals", {}),
        "alpha_signal": changes.get("alpha_signal"),
        "alpha_weights": changes.get("alpha_weights"),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ============================================================================
# MASTER FUNCTION — called from war_room_live_feeds
# ============================================================================

async def fetch_and_apply_council_intel(
    live_result: Any = None,
    yt_channels: list[str] | None = None,
    x_queries: list[str] | None = None,
    x_users: list[str] | None = None,
    x_provider: str = "xai",
) -> CouncilFeedResult:
    """Master function: fetch YouTube + X intel, apply to indicators + scenarios.

    Called from war_room_live_feeds.fetch_all_live_data() as part of the
    full feed pipeline.
    """
    # Fetch YouTube and X in parallel
    yt_result, x_result = await asyncio.gather(
        fetch_youtube_intel(channels=yt_channels),
        fetch_x_intel(queries=x_queries, users=x_users, provider=x_provider),
        return_exceptions=True,
    )

    # Handle exceptions from gather
    if isinstance(yt_result, BaseException):
        logger.warning("YouTube council gather failed: %s", yt_result)
        yt_result = CouncilFeedResult()
    if isinstance(x_result, BaseException):
        logger.warning("X council gather failed: %s", x_result)
        x_result = CouncilFeedResult()

    # Apply to indicators
    changes = apply_council_to_indicators(yt_result, x_result, live_result)

    # Feed Living Doctrine Engine signal into alpha engine
    try:
        from strategies.living_doctrine_engine import LivingDoctrineEngine
        lde = LivingDoctrineEngine()
        lde.feed_alpha_engine()
    except (ImportError, ValueError):
        pass  # LDE not available or no rules loaded — non-fatal

    # Update scenario statuses
    scenario_signals = changes.get("scenario_signals", {})
    if scenario_signals:
        transitions = update_scenario_statuses(scenario_signals)
        changes["scenario_transitions"] = transitions

    # Persist snapshot
    persist_council_snapshot(yt_result, x_result, changes)

    # Build combined result
    combined = CouncilFeedResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        yt_videos_processed=yt_result.yt_videos_processed,
        yt_sentiment_score=yt_result.yt_sentiment_score,
        yt_severity_score=yt_result.yt_severity_score,
        yt_key_topics=yt_result.yt_key_topics,
        yt_actionable_items=yt_result.yt_actionable_items,
        yt_scenario_hits=yt_result.yt_scenario_hits,
        yt_trust_score=yt_result.yt_trust_score,
        x_posts_analyzed=x_result.x_posts_analyzed,
        x_sentiment_score=x_result.x_sentiment_score,
        x_severity_score=x_result.x_severity_score,
        x_key_themes=x_result.x_key_themes,
        x_emerging_topics=x_result.x_emerging_topics,
        x_actionable_items=x_result.x_actionable_items,
        x_scenario_hits=x_result.x_scenario_hits,
        x_trust_score=x_result.x_trust_score,
        combined_sentiment=changes.get("combined_sentiment", 0.5),
        combined_severity=changes.get("combined_severity", 0.0),
        combined_trust=changes.get("combined_trust", 0.0),
        scenario_signals=scenario_signals,
        alpha_signal=changes.get("alpha_signal"),
        alpha_weights=changes.get("alpha_weights", {}),
        errors=yt_result.errors + x_result.errors,
    )

    logger.info("Council intel applied: %s", combined.summary())

    # Propagate alpha signal to LiveFeedResult for downstream IndicatorState patching
    if live_result is not None and combined.alpha_signal is not None:
        try:
            live_result.alpha_signal = combined.alpha_signal
            live_result.alpha_weights = combined.alpha_weights
        except AttributeError:
            pass  # live_result may not have alpha fields yet

    return combined


# ============================================================================
# CLI — standalone test
# ============================================================================

def main() -> None:
    """Run council feeds standalone and print results."""
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 72)
    print("WAR ROOM COUNCIL FEEDS — YouTube + X Intelligence")
    print("=" * 72)

    result = asyncio.run(fetch_and_apply_council_intel())

    print(f"\nTimestamp: {result.timestamp}")
    print(f"\n--- YouTube Intelligence ---")
    print(f"  Videos processed: {result.yt_videos_processed}")
    print(f"  Sentiment: {result.yt_sentiment_score:.2f}")
    print(f"  Severity: {result.yt_severity_score:.2f}")
    print(f"  Topics: {result.yt_key_topics[:10]}")
    print(f"  Scenario hits: {result.yt_scenario_hits}")

    print(f"\n--- X/Twitter Intelligence ---")
    print(f"  Posts analyzed: {result.x_posts_analyzed}")
    print(f"  Sentiment: {result.x_sentiment_score:.2f}")
    print(f"  Severity: {result.x_severity_score:.2f}")
    print(f"  Themes: {result.x_key_themes[:10]}")
    print(f"  Emerging: {result.x_emerging_topics[:5]}")
    print(f"  Scenario hits: {result.x_scenario_hits}")

    print(f"\n--- Combined ---")
    print(f"  Sentiment: {result.combined_sentiment:.2f}")
    print(f"  Severity: {result.combined_severity:.2f}")
    print(f"  Scenario signals: {result.scenario_signals}")

    if result.errors:
        print(f"\n  Errors: {result.errors}")


if __name__ == "__main__":
    main()
