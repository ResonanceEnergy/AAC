#!/usr/bin/env python3
"""
Black Swan Authority Monitor — Expert Intelligence Feed
========================================================
Monitors 4 thesis-aligned authorities across YouTube and news platforms,
classifies their content against the thesis chain, and feeds authority
consensus signals into the pressure cooker.

Authorities:
    1. Jeff Snider (Eurodollar University) — Dollar system, credit crisis, global macro
    2. Tom Bilyeu (Impact Theory)          — Financial system shift, Iran money trail
    3. Andrei Jikh                          — Markets, crypto, retail investor perspective
    4. Jiang Xueqin ("China's Nostradamus") — China geopolitics, Iran war predictions

Usage:
    python -m strategies.blackswan_authority_monitor              # full scan
    python -m strategies.blackswan_authority_monitor --quick      # youtube only
    python -m strategies.blackswan_authority_monitor --consensus  # consensus report
    python -m strategies.blackswan_authority_monitor --json       # machine output
"""

import json
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# AUTHORITY DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Authority:
    """A thesis-aligned expert we track across platforms."""
    id: str
    name: str
    expertise: str
    youtube_channel_id: Optional[str]
    twitter_handle: Optional[str]
    newsapi_queries: List[str]
    thesis_weight: float = 1.0  # How central to thesis (1.0 = standard, 2.0 = core)


AUTHORITIES: Dict[str, Authority] = {
    "jeff_snider": Authority(
        id="jeff_snider",
        name="Jeff Snider (Eurodollar University)",
        expertise="Dollar system mechanics, credit crisis, global monetary plumbing",
        youtube_channel_id="UCrXNkk4IESnqU-8GMad2vyA",
        twitter_handle="JeffSnider_EDU",
        newsapi_queries=[
            '"Jeff Snider"',
            '"Eurodollar University"',
            '"eurodollar" AND ("crisis" OR "dollar" OR "credit")',
        ],
        thesis_weight=2.0,  # Core — directly tracks dollar/credit collapse chain
    ),
    "tom_bilyeu": Authority(
        id="tom_bilyeu",
        name="Tom Bilyeu (Impact Theory)",
        expertise="Financial system shift, Iran money trail, geopolitical macro",
        youtube_channel_id="UCnYMOamNKLGVlJgRUbamveA",
        twitter_handle="TomBilyeu",
        newsapi_queries=[
            '"Tom Bilyeu"',
            '"Impact Theory" AND ("Iran" OR "war" OR "economy" OR "financial")',
        ],
        thesis_weight=1.5,  # High — directly covering Iran/money/system
    ),
    "andrei_jikh": Authority(
        id="andrei_jikh",
        name="Andrei Jikh",
        expertise="Markets, crypto, personal finance, retail investor lens",
        youtube_channel_id="UCGy7SkBjcIAgTiwkXEtPnYg",
        twitter_handle="AndreiJikh",
        newsapi_queries=[
            '"Andrei Jikh"',
        ],
        thesis_weight=1.0,
    ),
    "jiang_xueqin": Authority(
        id="jiang_xueqin",
        name='Jiang Xueqin ("China\'s Nostradamus")',
        expertise="China geopolitics, Iran war prediction, Trump foreign policy, petrodollar collapse, Strait of Hormuz",
        youtube_channel_id="UCngT5c9vjxHqBEW5kdb7g8w",  # "Professor Jiang Thoughts"
        twitter_handle=None,
        newsapi_queries=[
            '"Jiang Xueqin"',
            '"China Nostradamus" AND ("Iran" OR "war" OR "Trump")',
            '"Professor Jiang" AND ("Iran" OR "war" OR "prediction")',
        ],
        thesis_weight=2.0,  # Core — called the Iran war before anyone
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# THESIS CATEGORY MATCHING — same categories as pressure cooker
# ═══════════════════════════════════════════════════════════════════════

THESIS_CATEGORIES = {
    "IRAN_WAR": [
        "iran", "iranian", "irgc", "tehran", "hormuz", "persian gulf",
        "strike", "military operation", "war with iran", "attack iran",
        "middle east war", "us iran", "bombing iran", "middle east",
        "w@r", "asymmetric war", "proxy war", "operation epic fury",
        "operation midnight hammer", "cannot win", "cannot finish",
    ],
    "OIL_SHOCK": [
        "oil price", "oil spike", "oil shock", "crude oil", "brent",
        "wti", "barrel", "energy crisis", "gasoline", "diesel",
        "opec", "oil supply", "refinery", "petrol", "$200 oil",
        "gas price", "energy war", "oil crisis",
    ],
    "DOLLAR_COLLAPSE": [
        "dollar crash", "dollar collapse", "dollar decline", "usd",
        "de-dollarization", "dollar index", "dxy", "reserve currency",
        "dollar hegemony", "dollar dominance", "dollar strength",
        "dollar explod", "dollar scramble", "petrodollar",
        "dedollarization", "end of dollar",
    ],
    "GOLD_REPRICE": [
        "gold price", "gold record", "gold rally", "gold surge",
        "gold high", "safe haven", "bullion", "precious metal",
        "gold demand", "central bank gold",
    ],
    "YUAN_RISE": [
        "yuan", "renminbi", "petroyuan", "cips", "brics currency",
        "chinese currency", "rmb settlement", "yuan reserve",
    ],
    "CREDIT_CRISIS": [
        "credit crisis", "private credit", "shadow bank", "blackrock",
        "blackstone", "blue owl", "morgan stanley", "jp morgan",
        "collateral", "redemption", "fund gate", "liquidity",
        "2008", "subprime", "credit spread", "high yield",
        "underwriting", "private debt", "credit default",
    ],
    "RECESSION": [
        "recession", "job loss", "unemployment", "layoff",
        "payroll", "negative payroll", "gdp contrac", "economic downturn",
        "hiring freeze", "consumer spending", "retail sales",
        "mcdonalds value", "dollar tree", "consumer stress",
    ],
    "HOUSING_BUST": [
        "housing crash", "mortgage delinquen", "foreclosure",
        "housing bubble", "home price", "housing market warn",
        "mortgage rate", "real estate",
    ],
    "AI_BUBBLE": [
        "ai bubble", "ai bust", "ai layoff", "ai cost",
        "meta layoff", "tech layoff", "ai debt", "ai spending",
        "nvidia crash", "data center",
    ],
    "MARKET_CRASH": [
        "market crash", "stock crash", "selloff", "correction",
        "bear market", "meltdown", "panic", "capitulation",
        "vix", "volatility spike",
    ],
    "FED_TRAPPED": [
        "fed rate", "federal reserve", "stagflation", "inflation",
        "rate cut", "dilemma", "central bank warn", "monetary policy",
        "quantitative", "yield curve",
    ],
    "US_RETREAT": [
        "us pullback", "us withdrawal", "troops home", "military retreat",
        "base closure", "exit strategy", "war fatigue",
    ],
    "GULF_PIVOT": [
        "saudi", "uae", "qatar", "gulf state", "gcc",
        "saudi china", "gulf yuan", "opec plus", "mbs",
        "saudi aramco", "gulf pivot", "china profit",
        "china secret", "china role", "china wild card",
    ],
    "GLOBAL_CONTAGION": [
        "asia crisis", "emerging market", "currency crisis",
        "swiss franc", "snb", "capital controls", "contagion",
        "global bust", "systemic risk", "sovereign debt",
    ],
}

# Sentiment words for bullish (thesis-confirming) vs bearish (thesis-denying)
BULLISH_WORDS = [
    "crash", "collapse", "crisis", "bust", "shock", "panic", "meltdown",
    "breaking", "warning", "danger", "terrifying", "historic", "emergency",
    "skyrocket", "surge", "explod", "record high", "unprecedented",
    "worst since", "never seen", "plunge", "freefall", "contagion",
    "default", "bankrupt", "failed", "gate", "block", "restrict",
    "massive", "sweeping", "brutal", "alarm", "bad",
]

BEARISH_WORDS = [
    "recovery", "rally", "stabilize", "ceasefire", "peace deal",
    "negotiate", "ease", "de-escalat", "improve", "bullish",
    "soft landing", "goldilocks", "rebound", "upturn",
]


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AuthoritySignal:
    """A single piece of content from an authority."""
    authority_id: str
    authority_name: str
    title: str
    url: str
    published: str
    source: str             # "youtube", "newsapi", "twitter"
    categories: List[str]
    sentiment: str          # "BULLISH" (thesis-confirming), "BEARISH", "NEUTRAL"
    confidence: float       # 0.0 - 1.0
    views: Optional[int] = None
    description: str = ""


@dataclass
class AuthorityProfile:
    """Aggregated view of an authority's recent stance."""
    authority_id: str
    authority_name: str
    expertise: str
    total_signals: int = 0
    bullish_signals: int = 0
    bearish_signals: int = 0
    neutral_signals: int = 0
    top_categories: List[str] = field(default_factory=list)
    latest_signal: Optional[str] = None
    consensus: str = "UNKNOWN"   # STRONGLY_BULLISH, BULLISH, NEUTRAL, BEARISH
    confidence: float = 0.0
    total_views: int = 0


@dataclass
class AuthorityConsensus:
    """Overall consensus across all authorities."""
    timestamp: str
    total_authorities: int = 0
    total_signals: int = 0
    consensus: str = "UNKNOWN"
    consensus_score: float = 0.0  # -1.0 to +1.0
    agreement_pct: float = 0.0
    profiles: Dict[str, dict] = field(default_factory=dict)
    top_categories: List[Tuple[str, int]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# YOUTUBE RSS SCRAPER
# ═══════════════════════════════════════════════════════════════════════

YT_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}


def _fetch_youtube_feed(channel_id: str, timeout: int = 15) -> str:
    """Fetch raw XML from YouTube RSS feed."""
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (AAC Authority Monitor)"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _parse_youtube_feed(xml_text: str, authority: Authority,
                        since_days: int = 90) -> List[AuthoritySignal]:
    """Parse YouTube RSS XML into AuthoritySignals."""
    signals = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)

    root = ET.fromstring(xml_text)

    for entry in root.findall("atom:entry", YT_NS):
        title_el = entry.find("atom:title", YT_NS)
        link_el = entry.find("atom:link", YT_NS)
        pub_el = entry.find("atom:published", YT_NS)
        vid_el = entry.find("yt:videoId", YT_NS)

        if title_el is None or pub_el is None:
            continue

        title = title_el.text or ""
        published_str = pub_el.text or ""
        video_id = vid_el.text if vid_el is not None else ""
        url = link_el.get("href", "") if link_el is not None else ""
        if not url and video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"

        # Parse date
        try:
            pub_dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            if pub_dt < cutoff:
                continue
        except ValueError:
            pass

        # Extract description + views from media:group
        description = ""
        views = 0
        media_group = entry.find("media:group", YT_NS)
        if media_group is not None:
            desc_el = media_group.find("media:description", YT_NS)
            if desc_el is not None and desc_el.text:
                description = desc_el.text[:1500]
            community = media_group.find("media:community", YT_NS)
            if community is not None:
                stats = community.find("media:statistics", YT_NS)
                if stats is not None:
                    views = int(stats.get("views", 0))

        # Classify — use more description for better keyword matching
        text = f"{title} {description}".lower()
        # Also decode YouTube censorship tricks (w@r -> war)
        text = text.replace("@", "a")
        categories = _classify_text(text)
        sentiment, confidence = _score_sentiment(text)

        signals.append(AuthoritySignal(
            authority_id=authority.id,
            authority_name=authority.name,
            title=title,
            url=url,
            published=published_str,
            source="youtube",
            categories=categories,
            sentiment=sentiment,
            confidence=confidence,
            views=views,
            description=description[:200],
        ))

    return signals


def scrape_youtube_authorities(since_days: int = 90) -> List[AuthoritySignal]:
    """Scrape YouTube RSS feeds for all authorities with channels."""
    all_signals = []

    for auth_id, auth in AUTHORITIES.items():
        if not auth.youtube_channel_id:
            logger.info("  %s: no YouTube channel, skipping", auth.name)
            continue

        try:
            xml = _fetch_youtube_feed(auth.youtube_channel_id)
            signals = _parse_youtube_feed(xml, auth, since_days)
            all_signals.extend(signals)
            logger.info("  %s: %d videos (last %d days)", auth.name, len(signals), since_days)
        except (URLError, ET.ParseError) as e:
            logger.warning("  %s: YouTube fetch failed: %s", auth.name, e)

    return all_signals


# ═══════════════════════════════════════════════════════════════════════
# NEWSAPI SCANNER — authority mentions + articles
# ═══════════════════════════════════════════════════════════════════════

def _fetch_newsapi(query: str, api_key: str, days_back: int = 90,
                   page_size: int = 20) -> List[dict]:
    """Fetch articles from NewsAPI 'everything' endpoint."""
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    import urllib.parse
    params = urllib.parse.urlencode({
        "q": query,
        "from": from_date,
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": api_key,
    })
    url = f"https://newsapi.org/v2/everything?{params}"
    req = Request(url, headers={"User-Agent": "AAC-AuthorityMonitor/1.0"})

    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("articles", [])
    except (URLError, json.JSONDecodeError) as e:
        logger.warning("  NewsAPI query failed [%s]: %s", query[:40], e)
        return []


def scan_newsapi_authorities(days_back: int = 90) -> List[AuthoritySignal]:
    """Search NewsAPI for articles about/by our authorities."""
    cfg = get_config()
    api_key = getattr(cfg, "news_api_key", "") or ""
    if not api_key:
        logger.warning("  NEWSAPI_KEY not found, skipping news scan")
        return []

    all_signals = []
    seen_urls = set()

    for auth_id, auth in AUTHORITIES.items():
        for query in auth.newsapi_queries:
            articles = _fetch_newsapi(query, api_key, days_back)
            # Rate-limit: NewsAPI free tier = 100 req/day
            time.sleep(1.0)

            for art in articles:
                url = art.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title = art.get("title", "") or ""
                desc = art.get("description", "") or ""
                pub = art.get("publishedAt", "")
                source_name = art.get("source", {}).get("name", "")

                text = f"{title} {desc}".lower()
                categories = _classify_text(text)
                sentiment, confidence = _score_sentiment(text)

                all_signals.append(AuthoritySignal(
                    authority_id=auth.id,
                    authority_name=auth.name,
                    title=title,
                    url=url,
                    published=pub,
                    source="newsapi",
                    categories=categories,
                    sentiment=sentiment,
                    confidence=confidence,
                    description=desc[:200],
                ))

        logger.info("  %s: %d news articles",
                     auth.name,
                     sum(1 for s in all_signals if s.authority_id == auth_id))

    return all_signals


# ═══════════════════════════════════════════════════════════════════════
# TEXT CLASSIFICATION + SENTIMENT
# ═══════════════════════════════════════════════════════════════════════

def _classify_text(text: str) -> List[str]:
    """Match text against thesis categories. Returns list of matching category names."""
    text_lower = text.lower()
    matched = []
    for category, keywords in THESIS_CATEGORIES.items():
        for kw in keywords:
            if kw in text_lower:
                matched.append(category)
                break
    return matched


def _score_sentiment(text: str) -> Tuple[str, float]:
    """Score sentiment as BULLISH (thesis-confirming), BEARISH, or NEUTRAL.
    Returns (sentiment, confidence 0-1)."""
    text_lower = text.lower()
    bull_hits = sum(1 for w in BULLISH_WORDS if w in text_lower)
    bear_hits = sum(1 for w in BEARISH_WORDS if w in text_lower)

    total = bull_hits + bear_hits
    if total == 0:
        return "NEUTRAL", 0.3

    ratio = bull_hits / total
    confidence = min(1.0, total / 8.0)  # More keyword hits = higher confidence

    if ratio >= 0.7:
        return "BULLISH", confidence
    elif ratio <= 0.3:
        return "BEARISH", confidence
    else:
        return "NEUTRAL", confidence * 0.5


# ═══════════════════════════════════════════════════════════════════════
# AUTHORITY PROFILING + CONSENSUS
# ═══════════════════════════════════════════════════════════════════════

def build_profiles(signals: List[AuthoritySignal]) -> Dict[str, AuthorityProfile]:
    """Build per-authority profiles from collected signals."""
    profiles = {}

    for auth_id, auth in AUTHORITIES.items():
        auth_signals = [s for s in signals if s.authority_id == auth_id]

        bull = sum(1 for s in auth_signals if s.sentiment == "BULLISH")
        bear = sum(1 for s in auth_signals if s.sentiment == "BEARISH")
        neut = sum(1 for s in auth_signals if s.sentiment == "NEUTRAL")
        total = len(auth_signals)
        total_views = sum(s.views or 0 for s in auth_signals)

        # Category frequency
        cat_counts: Dict[str, int] = {}
        for s in auth_signals:
            for c in s.categories:
                cat_counts[c] = cat_counts.get(c, 0) + 1
        top_cats = sorted(cat_counts, key=cat_counts.get, reverse=True)[:5]

        # Latest signal
        latest = None
        if auth_signals:
            sorted_sigs = sorted(auth_signals, key=lambda s: s.published, reverse=True)
            latest = sorted_sigs[0].title

        # Consensus for this authority
        if total == 0:
            consensus = "NO_DATA"
            conf = 0.0
        elif bull / max(total, 1) >= 0.8:
            consensus = "STRONGLY_BULLISH"
            conf = bull / total
        elif bull / max(total, 1) >= 0.6:
            consensus = "BULLISH"
            conf = bull / total
        elif bear / max(total, 1) >= 0.6:
            consensus = "BEARISH"
            conf = bear / total
        else:
            consensus = "MIXED"
            conf = max(bull, bear, neut) / max(total, 1)

        profiles[auth_id] = AuthorityProfile(
            authority_id=auth_id,
            authority_name=auth.name,
            expertise=auth.expertise,
            total_signals=total,
            bullish_signals=bull,
            bearish_signals=bear,
            neutral_signals=neut,
            top_categories=top_cats,
            latest_signal=latest,
            consensus=consensus,
            confidence=round(conf, 3),
            total_views=total_views,
        )

    return profiles


def build_consensus(signals: List[AuthoritySignal],
                    profiles: Dict[str, AuthorityProfile]) -> AuthorityConsensus:
    """Build overall authority consensus."""
    # Weighted consensus score: -1.0 (all bearish) to +1.0 (all bullish)
    weighted_sum = 0.0
    weight_total = 0.0

    for auth_id, profile in profiles.items():
        auth = AUTHORITIES[auth_id]
        w = auth.thesis_weight

        if profile.total_signals == 0:
            continue

        # Score: bull proportion minus bear proportion
        score = (profile.bullish_signals - profile.bearish_signals) / profile.total_signals
        weighted_sum += score * w
        weight_total += w

    consensus_score = weighted_sum / weight_total if weight_total > 0 else 0.0

    # Agreement: what % of authorities lean the same way
    leans = [p.consensus for p in profiles.values() if p.consensus not in ("NO_DATA", "UNKNOWN")]
    bull_count = sum(1 for l in leans if "BULLISH" in l)
    bear_count = sum(1 for l in leans if l == "BEARISH")
    total_with_data = len(leans)
    agreement = max(bull_count, bear_count) / max(total_with_data, 1)

    if consensus_score >= 0.6:
        overall = "STRONGLY_BULLISH"
    elif consensus_score >= 0.3:
        overall = "BULLISH"
    elif consensus_score <= -0.3:
        overall = "BEARISH"
    elif consensus_score <= -0.1:
        overall = "LEANING_BEARISH"
    else:
        overall = "MIXED"

    # Global category ranking
    cat_counts: Dict[str, int] = {}
    for s in signals:
        for c in s.categories:
            cat_counts[c] = cat_counts.get(c, 0) + 1
    top_global = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return AuthorityConsensus(
        timestamp=datetime.now().isoformat(),
        total_authorities=sum(1 for p in profiles.values() if p.total_signals > 0),
        total_signals=len(signals),
        consensus=overall,
        consensus_score=round(consensus_score, 4),
        agreement_pct=round(agreement * 100, 1),
        profiles={k: asdict(v) for k, v in profiles.items()},
        top_categories=top_global,
    )


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API — for pressure cooker + crisis center
# ═══════════════════════════════════════════════════════════════════════

_CACHE_FILE = PROJECT_ROOT / "data" / "blackswan_authority_scan.json"
_CACHE_TTL = 3600  # 1 hour


def get_authority_consensus() -> Dict[str, Any]:
    """Quick API for crisis center. Returns cached if fresh, else runs scan."""
    if _CACHE_FILE.exists():
        try:
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            ts = data.get("timestamp", "")
            if ts:
                cached_dt = datetime.fromisoformat(ts)
                if (datetime.now() - cached_dt).total_seconds() < _CACHE_TTL:
                    return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Run fresh scan
    return run_full_scan(save=True)


def run_full_scan(youtube_days: int = 90, news_days: int = 30,
                  save: bool = True) -> Dict[str, Any]:
    """Run complete authority scan across all sources."""
    print("\n" + "=" * 78)
    print("  BLACK SWAN AUTHORITY MONITOR — Expert Intelligence Feed")
    print("=" * 78)

    all_signals: List[AuthoritySignal] = []

    # ── YouTube RSS ──
    print("\n[YOUTUBE] Scanning authority channels...")
    yt_signals = scrape_youtube_authorities(since_days=youtube_days)
    all_signals.extend(yt_signals)
    print(f"  => {len(yt_signals)} videos from {sum(1 for a in AUTHORITIES.values() if a.youtube_channel_id)} channels")

    # ── NewsAPI ──
    print("\n[NEWSAPI] Scanning authority mentions...")
    news_signals = scan_newsapi_authorities(days_back=news_days)
    all_signals.extend(news_signals)
    print(f"  => {len(news_signals)} news articles")

    # ── Build profiles + consensus ──
    print("\n[ANALYSIS] Building authority profiles...")
    profiles = build_profiles(all_signals)
    consensus = build_consensus(all_signals, profiles)

    print(f"\n  Total signals: {consensus.total_signals}")
    print(f"  Active authorities: {consensus.total_authorities}/4")
    print(f"  Consensus: {consensus.consensus} (score: {consensus.consensus_score:+.2f})")
    print(f"  Agreement: {consensus.agreement_pct:.0f}%")

    # Serialize
    result = {
        "timestamp": consensus.timestamp,
        "total_signals": consensus.total_signals,
        "total_authorities": consensus.total_authorities,
        "consensus": consensus.consensus,
        "consensus_score": consensus.consensus_score,
        "agreement_pct": consensus.agreement_pct,
        "top_categories": consensus.top_categories,
        "profiles": consensus.profiles,
        "signals": [asdict(s) for s in all_signals],
    }

    if save:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(result, indent=2, default=str),
                               encoding="utf-8")
        print(f"\n  Saved to {_CACHE_FILE}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════

def render_consensus_report(data: Dict[str, Any]) -> str:
    """Render a text report of authority consensus."""
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("  [AUTHORITIES] BLACK SWAN THESIS — EXPERT CONSENSUS REPORT")
    lines.append("=" * 78)
    lines.append(f"  Scan time: {data.get('timestamp', 'N/A')}")
    lines.append(f"  Signals:   {data.get('total_signals', 0)} from "
                 f"{data.get('total_authorities', 0)} authorities")

    score = data.get("consensus_score", 0)
    consensus = data.get("consensus", "UNKNOWN")
    agreement = data.get("agreement_pct", 0)

    # Consensus bar
    bar_len = 40
    filled = int(((score + 1) / 2) * bar_len)
    bar = "█" * max(filled, 0) + "░" * max(bar_len - filled, 0)
    lines.append(f"\n  CONSENSUS: {consensus}")
    lines.append(f"  Score: [{bar}] {score:+.2f}")
    lines.append(f"  Agreement: {agreement:.0f}%")

    # Per-authority breakdown
    lines.append("\n  ─── AUTHORITY BREAKDOWN ───────────────────────────────────")
    profiles = data.get("profiles", {})
    for auth_id, profile in profiles.items():
        name = profile.get("authority_name", auth_id)
        total = profile.get("total_signals", 0)
        bull = profile.get("bullish_signals", 0)
        bear = profile.get("bearish_signals", 0)
        neut = profile.get("neutral_signals", 0)
        cons = profile.get("consensus", "?")
        latest = profile.get("latest_signal", "—")
        views = profile.get("total_views", 0)
        top_cats = profile.get("top_categories", [])

        if cons == "STRONGLY_BULLISH":
            icon = "🔴"
        elif cons == "BULLISH":
            icon = "🟠"
        elif cons == "BEARISH":
            icon = "🟢"
        else:
            icon = "⚪"

        lines.append(f"\n  {icon} {name}")
        lines.append(f"     Stance: {cons}  |  Signals: {total}  "
                     f"({bull}B / {neut}N / {bear}A)")
        if views:
            lines.append(f"     Views: {views:,}")
        if top_cats:
            lines.append(f"     Focus: {', '.join(top_cats[:4])}")
        if latest:
            lines.append(f"     Latest: {latest[:70]}")

    # Top categories across all authorities
    top_cats = data.get("top_categories", [])
    if top_cats:
        lines.append("\n  ─── TOP THESIS CATEGORIES (ALL AUTHORITIES) ─────────────")
        for cat, count in top_cats[:8]:
            bar = "█" * min(count, 30)
            lines.append(f"     {cat:20s} {bar} ({count})")

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)


def render_authority_feed(data: Dict[str, Any], limit: int = 30) -> str:
    """Render recent authority content feed."""
    lines = []
    lines.append("\n  ─── LATEST AUTHORITY CONTENT ──────────────────────────────")

    signals = data.get("signals", [])
    # Sort by published date, newest first
    sorted_sigs = sorted(signals, key=lambda s: s.get("published", ""), reverse=True)

    for i, sig in enumerate(sorted_sigs[:limit]):
        title = sig.get("title", "?")[:65]
        auth = sig.get("authority_name", "?").split("(")[0].strip()[:20]
        src = sig.get("source", "?")[:7]
        sent = sig.get("sentiment", "?")
        cats = sig.get("categories", [])
        views = sig.get("views")
        pub = sig.get("published", "")[:10]

        if sent == "BULLISH":
            marker = "▲"
        elif sent == "BEARISH":
            marker = "▼"
        else:
            marker = "─"

        cat_str = ",".join(cats[:2]) if cats else "uncategorized"
        view_str = f" ({views:,}v)" if views else ""

        lines.append(f"  {marker} [{pub}] {auth:>18s} | {title}")
        lines.append(f"    {src:>7s} | {cat_str}{view_str}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Black Swan Authority Monitor")
    parser.add_argument("--quick", action="store_true",
                        help="YouTube-only scan (skip NewsAPI)")
    parser.add_argument("--consensus", action="store_true",
                        help="Show consensus report from cache")
    parser.add_argument("--feed", action="store_true",
                        help="Show latest content feed")
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable JSON output")
    parser.add_argument("--days", type=int, default=90,
                        help="How many days of YouTube history (default: 90)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.consensus:
        # Show from cache
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        else:
            data = run_full_scan(youtube_days=args.days, save=True)
        print(render_consensus_report(data))
        return

    if args.quick:
        # YouTube only
        print("\n[YOUTUBE-ONLY SCAN]")
        signals = scrape_youtube_authorities(since_days=args.days)
        profiles = build_profiles(signals)
        consensus = build_consensus(signals, profiles)
        data = {
            "timestamp": consensus.timestamp,
            "total_signals": len(signals),
            "total_authorities": consensus.total_authorities,
            "consensus": consensus.consensus,
            "consensus_score": consensus.consensus_score,
            "agreement_pct": consensus.agreement_pct,
            "top_categories": consensus.top_categories,
            "profiles": {k: asdict(v) for k, v in profiles.items()},
            "signals": [asdict(s) for s in signals],
        }
    else:
        data = run_full_scan(youtube_days=args.days, save=True)

    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(render_consensus_report(data))
        if args.feed or not args.quick:
            print(render_authority_feed(data))


if __name__ == "__main__":
    main()
