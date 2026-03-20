#!/usr/bin/env python3
"""
Black Swan News Scanner — Mainstream Site Intelligence
======================================================
Scans NewsAPI (70k+ sources), Finnhub market news, and GDELT
(global event database) for headlines aligned with our black swan
thesis chain: Iran war → US withdrawal → Gulf yuan shift → gold
reprice → USD collapse.

Feeds results back into the pressure cooker as real-time signal
intelligence for the crisis center.

Usage:
    python -m strategies.blackswan_news_scanner           # full scan
    python -m strategies.blackswan_news_scanner --quick    # newsapi only
    python -m strategies.blackswan_news_scanner --json     # machine output
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# THESIS SEARCH QUERIES — crafted to hit mainstream headlines
# ═══════════════════════════════════════════════════════════════════════

# NewsAPI "everything" endpoint — max 20 queries per plan
NEWSAPI_QUERIES = [
    # Core thesis chain
    '("Iran" AND ("war" OR "attack" OR "strike" OR "military"))',
    '("Strait of Hormuz" OR "Persian Gulf" OR "IRGC")',
    '("Israel" AND ("war" OR "Hezbollah" OR "Hamas" OR "Lebanon" OR "Gaza"))',
    # Oil shock / energy
    '("oil price" AND ("surge" OR "spike" OR "shock" OR "crisis"))',
    '("OPEC" AND ("cut" OR "supply" OR "production" OR "emergency"))',
    # Dollar / gold / yuan
    '("US dollar" AND ("crash" OR "decline" OR "collapse" OR "weakening"))',
    '("gold price" AND ("record" OR "high" OR "surge" OR "rally"))',
    '("yuan" AND ("reserve currency" OR "BRICS" OR "petro-yuan" OR "de-dollarization"))',
    # Fed trapped / recession
    '("Federal Reserve" AND ("rate cut" OR "trapped" OR "dilemma" OR "stagflation"))',
    '("recession" AND ("US" OR "global" OR "warning" OR "signal"))',
    # Credit / bonds
    '("debt ceiling" OR "credit default" OR "bond market crash" OR "yield curve")',
    # Crypto risk-off
    '("bitcoin" AND ("crash" OR "sell-off" OR "risk" OR "below"))',
    # Geopolitical escalation
    '("NATO" AND ("military" OR "conflict" OR "Russia" OR "escalation"))',
    '("sanctions" AND ("Iran" OR "Russia" OR "China" OR "oil"))',
    '("nuclear" AND ("weapon" OR "threat" OR "Iran" OR "escalation"))',
]

# Finnhub category: "general", "forex", "crypto", "merger"
FINNHUB_CATEGORIES = ["general", "forex", "crypto"]

# GDELT GKG (Global Knowledge Graph) — free, no API key, JSON output
# Queries the GDELT DOC API for article counts and tone analysis
GDELT_QUERIES = [
    "Iran war attack military",
    "oil price shock crisis",
    "gold record high surge",
    "dollar collapse decline crash",
    "Federal Reserve recession stagflation",
    "BRICS de-dollarization yuan",
    "credit crisis default debt",
    "nuclear threat escalation",
]

# Thesis category mapping — which query covers which category
QUERY_CATEGORIES = {
    "iran_war": [0, 1],
    "israel_conflict": [2],
    "oil_shock": [3, 4],
    "usd_collapse": [5],
    "gold_reprice": [6],
    "yuan_rise": [7],
    "fed_trapped": [8, 9],
    "credit_crisis": [10],
    "crypto_crisis": [11],
    "geopolitical": [12, 13, 14],
}

# Severity keywords for scoring headline urgency
SEVERITY_WORDS = {
    "critical": ["attack", "strike", "war", "crash", "collapse", "emergency",
                  "nuclear", "invasion", "default", "panic"],
    "high": ["surge", "spike", "record", "escalation", "threat", "crisis",
             "sell-off", "plunge", "warning", "sanction"],
    "moderate": ["tension", "risk", "concern", "volatile", "decline", "weaken",
                 "uncertainty", "cut", "pressure", "fear"],
}


@dataclass
class NewsSignal:
    """A single news article scored against our thesis."""
    headline: str
    source: str
    url: str
    published: str
    category: str  # thesis category
    severity: str  # critical / high / moderate / low
    score: float   # 0-1 relevance score
    origin: str    # newsapi / finnhub / gdelt
    snippet: str = ""


@dataclass
class NewsScanResult:
    """Aggregated result of a mainstream news scan."""
    timestamp: datetime = field(default_factory=datetime.now)
    total_articles: int = 0
    thesis_aligned: int = 0
    signals: List[NewsSignal] = field(default_factory=list)
    category_counts: Dict[str, int] = field(default_factory=dict)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    hottest_category: str = ""
    average_severity: float = 0.0
    sources_scanned: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class BlackSwanNewsScanner:
    """Scans mainstream news for black swan thesis signals."""

    def __init__(self):
        self.config = get_config()
        self.result = NewsScanResult()
        self._seen_urls: set = set()

    # ─── NewsAPI ──────────────────────────────────────────────────────

    async def _scan_newsapi(self) -> List[NewsSignal]:
        """Scan NewsAPI for thesis-aligned articles."""
        if not self.config.news_api_key:
            self.result.errors.append("NewsAPI: no API key configured")
            return []

        from integrations.api_integration_hub import NewsAPIClient
        signals = []

        async with NewsAPIClient(self.config) as client:
            for i, query in enumerate(NEWSAPI_QUERIES):
                try:
                    resp = await client.get_financial_news(
                        query=query, language="en", page_size=10
                    )
                    if not resp.success:
                        logger.debug(f"NewsAPI query {i} failed: {resp.error}")
                        continue

                    articles = resp.data.get("articles", []) if isinstance(resp.data, dict) else []
                    for article in articles:
                        url = article.get("url", "")
                        if url in self._seen_urls:
                            continue
                        self._seen_urls.add(url)

                        headline = article.get("title", "") or ""
                        source_name = article.get("source", {}).get("name", "Unknown")
                        published = article.get("publishedAt", "")
                        description = article.get("description", "") or ""

                        category = self._classify_category(i, headline, description)
                        severity = self._score_severity(headline + " " + description)

                        signals.append(NewsSignal(
                            headline=headline,
                            source=source_name,
                            url=url,
                            published=published,
                            category=category,
                            severity=severity,
                            score=self._score_relevance(headline, description, category),
                            origin="newsapi",
                            snippet=description[:200] if description else "",
                        ))

                except Exception as e:
                    logger.warning(f"NewsAPI query {i} error: {e}")

        self.result.sources_scanned.append("newsapi")
        logger.info(f"NewsAPI: {len(signals)} thesis-aligned articles found")
        return signals

    # ─── Finnhub ──────────────────────────────────────────────────────

    async def _scan_finnhub(self) -> List[NewsSignal]:
        """Scan Finnhub market news for thesis signals."""
        if not self.config.finnhub_key:
            self.result.errors.append("Finnhub: no API key configured")
            return []

        from integrations.api_integration_hub import FinnhubClient
        signals = []

        async with FinnhubClient(self.config) as client:
            for cat in FINNHUB_CATEGORIES:
                try:
                    resp = await client.get_market_news(category=cat)
                    if not resp.success:
                        continue

                    articles = resp.data if isinstance(resp.data, list) else []
                    for article in articles:
                        url = article.get("url", "")
                        if url in self._seen_urls:
                            continue
                        self._seen_urls.add(url)

                        headline = article.get("headline", "") or ""
                        source_name = article.get("source", "Finnhub")
                        published = article.get("datetime", "")
                        if isinstance(published, (int, float)):
                            published = datetime.fromtimestamp(published).isoformat()
                        summary = article.get("summary", "") or ""

                        category = self._classify_by_text(headline + " " + summary)
                        if not category:
                            continue  # not thesis-aligned

                        severity = self._score_severity(headline + " " + summary)
                        signals.append(NewsSignal(
                            headline=headline,
                            source=source_name,
                            url=url,
                            published=str(published),
                            category=category,
                            severity=severity,
                            score=self._score_relevance(headline, summary, category),
                            origin="finnhub",
                            snippet=summary[:200] if summary else "",
                        ))

                except Exception as e:
                    logger.warning(f"Finnhub {cat} error: {e}")

        self.result.sources_scanned.append("finnhub")
        logger.info(f"Finnhub: {len(signals)} thesis-aligned articles found")
        return signals

    # ─── GDELT ────────────────────────────────────────────────────────

    async def _scan_gdelt(self) -> List[NewsSignal]:
        """Scan GDELT DOC API for thesis articles. Free, no key needed."""
        signals = []
        base = "https://api.gdeltproject.org/api/v2/doc/doc"
        loop = asyncio.get_event_loop()

        for query_text in GDELT_QUERIES:
            try:
                params = (
                    f"?query={query_text.replace(' ', '%20')}"
                    "&mode=ArtList&maxrecords=10&format=json"
                    "&timespan=24h&sort=DateDesc"
                )
                url = base + params
                req = Request(url, headers={"User-Agent": "AAC-BlackSwan/1.0"})

                # Run blocking urllib in executor to avoid blocking event loop
                resp_text = await loop.run_in_executor(
                    None, lambda u=url, r=req: urlopen(r, timeout=15).read().decode("utf-8")
                )
                data = json.loads(resp_text)
                articles = data.get("articles", [])

                for article in articles:
                    art_url = article.get("url", "")
                    if art_url in self._seen_urls:
                        continue
                    self._seen_urls.add(art_url)

                    headline = article.get("title", "") or ""
                    source_name = article.get("domain", "GDELT")
                    published = article.get("seendate", "")
                    language = article.get("language", "")
                    if language and language != "English":
                        continue  # English only

                    category = self._classify_by_text(headline)
                    if not category:
                        category = self._classify_by_text(query_text)

                    severity = self._score_severity(headline)
                    signals.append(NewsSignal(
                        headline=headline,
                        source=source_name,
                        url=art_url,
                        published=str(published),
                        category=category or "geopolitical",
                        severity=severity,
                        score=self._score_relevance(headline, "", category or "geopolitical"),
                        origin="gdelt",
                        snippet="",
                    ))

            except (URLError, json.JSONDecodeError) as e:
                logger.debug(f"GDELT query '{query_text}' error: {e}")
            except Exception as e:
                logger.warning(f"GDELT error: {e}")

        self.result.sources_scanned.append("gdelt")
        logger.info(f"GDELT: {len(signals)} thesis-aligned articles found")
        return signals

    # ─── Classification helpers ───────────────────────────────────────

    def _classify_category(self, query_index: int, headline: str, description: str) -> str:
        """Map a NewsAPI query index to a thesis category."""
        for cat, indices in QUERY_CATEGORIES.items():
            if query_index in indices:
                return cat
        # Fallback: text classification
        return self._classify_by_text(headline + " " + description) or "geopolitical"

    def _classify_by_text(self, text: str) -> str:
        """Classify text into thesis category based on keyword matching."""
        text_lower = text.lower()
        # Simple keyword → category mapping
        mapping = {
            "iran_war": ["iran", "irgc", "hormuz", "persian gulf", "tehran"],
            "israel_conflict": ["israel", "gaza", "hezbollah", "hamas", "netanyahu", "idf"],
            "oil_shock": ["oil price", "crude oil", "opec", "barrel", "brent"],
            "gold_reprice": ["gold price", "gold record", "gold surge", "xau"],
            "usd_collapse": ["us dollar", "dollar index", "dxy", "de-dollarization"],
            "yuan_rise": ["yuan", "renminbi", "brics currency", "petro-yuan"],
            "fed_trapped": ["federal reserve", "fed rate", "powell", "fomc", "stagflation"],
            "recession": ["recession", "gdp decline", "economic contraction"],
            "credit_crisis": ["debt ceiling", "credit default", "bond market", "yield curve"],
            "crypto_crisis": ["bitcoin", "crypto crash", "btc"],
            "geopolitical": ["nato", "nuclear", "sanction", "military", "war"],
            "inflation": ["inflation", "cpi", "consumer price", "stagflation"],
        }
        best_cat = ""
        best_count = 0
        for cat, kws in mapping.items():
            count = sum(1 for kw in kws if kw in text_lower)
            if count > best_count:
                best_count = count
                best_cat = cat
        return best_cat if best_count > 0 else ""

    def _score_severity(self, text: str) -> str:
        """Score headline severity based on trigger words."""
        text_lower = text.lower()
        for level in ("critical", "high", "moderate"):
            words = SEVERITY_WORDS[level]
            if any(w in text_lower for w in words):
                return level
        return "low"

    def _score_relevance(self, headline: str, description: str, category: str) -> float:
        """Score 0-1 relevance of article to the thesis."""
        text = (headline + " " + description).lower()
        score = 0.0

        # Category-specific boosters (core thesis chain scores higher)
        core_chain = {"iran_war", "oil_shock", "gold_reprice", "usd_collapse", "yuan_rise"}
        if category in core_chain:
            score += 0.3
        else:
            score += 0.15

        # Severity boost
        severity = self._score_severity(text)
        sev_map = {"critical": 0.35, "high": 0.25, "moderate": 0.15, "low": 0.05}
        score += sev_map.get(severity, 0.05)

        # Multiple thesis keyword hits in same article = higher relevance
        all_keywords = [
            "iran", "oil", "gold", "dollar", "yuan", "fed", "recession",
            "war", "nuclear", "sanctions", "brics", "hormuz", "collapse"
        ]
        hits = sum(1 for kw in all_keywords if kw in text)
        score += min(hits * 0.05, 0.30)  # max 0.30 from multi-hit

        return min(score, 1.0)

    # ─── Main scan ────────────────────────────────────────────────────

    async def scan(self, quick: bool = False) -> NewsScanResult:
        """Run the full mainstream news scan.

        Args:
            quick: If True, only scan NewsAPI (faster).
        """
        logger.info("Black Swan News Scanner: scanning mainstream sites...")
        all_signals: List[NewsSignal] = []

        # Always scan NewsAPI
        newsapi_signals = await self._scan_newsapi()
        all_signals.extend(newsapi_signals)

        if not quick:
            # Scan Finnhub + GDELT in parallel
            finnhub_task = self._scan_finnhub()
            gdelt_task = self._scan_gdelt()
            finnhub_signals, gdelt_signals = await asyncio.gather(
                finnhub_task, gdelt_task, return_exceptions=True
            )
            if isinstance(finnhub_signals, list):
                all_signals.extend(finnhub_signals)
            else:
                self.result.errors.append(f"Finnhub error: {finnhub_signals}")
            if isinstance(gdelt_signals, list):
                all_signals.extend(gdelt_signals)
            else:
                self.result.errors.append(f"GDELT error: {gdelt_signals}")

        # Deduplicate + sort by score
        all_signals.sort(key=lambda s: s.score, reverse=True)

        # Build result
        self.result.signals = all_signals
        self.result.total_articles = len(self._seen_urls)
        self.result.thesis_aligned = len(all_signals)

        # Category counts
        for sig in all_signals:
            self.result.category_counts[sig.category] = (
                self.result.category_counts.get(sig.category, 0) + 1
            )
        # Severity counts
        for sig in all_signals:
            self.result.severity_counts[sig.severity] = (
                self.result.severity_counts.get(sig.severity, 0) + 1
            )

        # Hottest category
        if self.result.category_counts:
            self.result.hottest_category = max(
                self.result.category_counts, key=self.result.category_counts.get  # type: ignore[arg-type]
            )

        # Average severity
        sev_vals = {"critical": 1.0, "high": 0.75, "moderate": 0.5, "low": 0.25}
        if all_signals:
            self.result.average_severity = sum(
                sev_vals.get(s.severity, 0.25) for s in all_signals
            ) / len(all_signals)

        logger.info(
            f"News scan complete: {self.result.thesis_aligned} signals from "
            f"{', '.join(self.result.sources_scanned)}"
        )
        return self.result

    # ─── Output formatters ────────────────────────────────────────────

    def generate_report(self, max_items: int = 30) -> str:
        """Generate a formatted text report of scan results."""
        r = self.result
        lines = [
            "=" * 70,
            "  BLACK SWAN NEWS SCANNER — MAINSTREAM INTELLIGENCE REPORT",
            "=" * 70,
            f"  Scan Time: {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Sources: {', '.join(r.sources_scanned) or 'none'}",
            f"  Total Articles Scanned: {r.total_articles}",
            f"  Thesis-Aligned Signals: {r.thesis_aligned}",
            f"  Hottest Category: {r.hottest_category.upper().replace('_', ' ') if r.hottest_category else 'N/A'}",
            f"  Average Severity: {r.average_severity:.2f}",
            "",
        ]

        # Category breakdown
        lines.append("  CATEGORY BREAKDOWN:")
        for cat, count in sorted(r.category_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(count, 30)
            lines.append(f"    {cat:<20s} {count:>3d}  {bar}")

        # Severity breakdown
        lines.append("")
        lines.append("  SEVERITY BREAKDOWN:")
        for sev in ("critical", "high", "moderate", "low"):
            count = r.severity_counts.get(sev, 0)
            marker = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}.get(sev, "⚪")
            lines.append(f"    {marker} {sev:<12s} {count:>3d}")

        # Top signals
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"  TOP {min(max_items, len(r.signals))} SIGNALS (by relevance)")
        lines.append("-" * 70)

        for i, sig in enumerate(r.signals[:max_items], 1):
            sev_badge = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}.get(
                sig.severity, "⚪"
            )
            lines.append(f"")
            lines.append(
                f"  #{i:<3d} {sev_badge} [{sig.category.upper()}] "
                f"Score: {sig.score:.2f} | {sig.origin}"
            )
            # Truncate headline to 80 chars for display
            hl = sig.headline[:80] + ("..." if len(sig.headline) > 80 else "")
            lines.append(f"       {hl}")
            lines.append(f"       Source: {sig.source} | {sig.published[:19] if sig.published else 'N/A'}")
            if sig.snippet:
                snip = sig.snippet[:120] + ("..." if len(sig.snippet) > 120 else "")
                lines.append(f"       {snip}")

        # Errors
        if r.errors:
            lines.append("")
            lines.append("  ⚠ ERRORS:")
            for err in r.errors:
                lines.append(f"    - {err}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("  Feed critical signals → Pressure Cooker → Crisis Center")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        """Return scan results as JSON-serializable dict."""
        return {
            "timestamp": self.result.timestamp.isoformat(),
            "total_articles": self.result.total_articles,
            "thesis_aligned": self.result.thesis_aligned,
            "hottest_category": self.result.hottest_category,
            "average_severity": round(self.result.average_severity, 3),
            "category_counts": self.result.category_counts,
            "severity_counts": self.result.severity_counts,
            "sources_scanned": self.result.sources_scanned,
            "errors": self.result.errors,
            "signals": [
                {
                    "headline": s.headline,
                    "source": s.source,
                    "url": s.url,
                    "published": s.published,
                    "category": s.category,
                    "severity": s.severity,
                    "score": round(s.score, 3),
                    "origin": s.origin,
                }
                for s in self.result.signals
            ],
        }

    def save_results(self, path: Optional[str] = None) -> str:
        """Save scan results to JSON file."""
        if path is None:
            data_dir = PROJECT_ROOT / "data"
            data_dir.mkdir(exist_ok=True)
            path = str(data_dir / "blackswan_news_scan.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)
        return path


# ═══════════════════════════════════════════════════════════════════════
# Module-level convenience functions
# ═══════════════════════════════════════════════════════════════════════

async def run_news_scan(quick: bool = False) -> NewsScanResult:
    """Run a full news scan and return results."""
    scanner = BlackSwanNewsScanner()
    return await scanner.scan(quick=quick)


def get_news_scan_report(quick: bool = False) -> str:
    """Run scan and return formatted report (sync wrapper)."""
    scanner = BlackSwanNewsScanner()
    asyncio.run(scanner.scan(quick=quick))
    return scanner.generate_report()


def get_news_crisis_data(quick: bool = False) -> Dict[str, Any]:
    """Run scan and return JSON data (sync wrapper)."""
    scanner = BlackSwanNewsScanner()
    asyncio.run(scanner.scan(quick=quick))
    return scanner.to_json()


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

async def _main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    quick = "--quick" in sys.argv
    json_out = "--json" in sys.argv

    scanner = BlackSwanNewsScanner()
    await scanner.scan(quick=quick)

    if json_out:
        data = scanner.to_json()
        print(json.dumps(data, indent=2))
    else:
        report = scanner.generate_report()
        print(report)

    # Save results
    path = scanner.save_results()
    logger.info(f"Results saved to {path}")


if __name__ == "__main__":
    asyncio.run(_main())
