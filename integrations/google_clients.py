from __future__ import annotations

"""Google API clients for DFV.

Four clients, all gated on env keys (return graceful empty payloads when keys
are absent so DFV routines never crash):

* ``CustomSearchClient`` — Google Custom Search JSON API. Used for news
  breadth across Reuters / Bloomberg / WSJ / FT / SEC / etc. Needs
  ``GOOGLE_API_KEY`` and ``GOOGLE_CSE_ID``.
* ``YouTubeClient`` — YouTube Data API v3. Used for earnings calls,
  Fed pressers, retail-sentiment proxies. Needs ``GOOGLE_API_KEY``.
* ``GoogleTrendsClient`` — pytrends wrapper, no key required. Retail
  interest signal for held tickers + macro keywords.
* ``GeminiClient`` — Gemini 2.x via ``google-genai``. Second-opinion LLM
  beside Grok for thesis review (DFV G1 gate). Needs ``GOOGLE_AI_API_KEY``.

All clients are intentionally synchronous. DFV routines are batch and
called from the daemon's scheduler, not a hot loop, so async would add
complexity without latency benefit here.
"""

import os
import time
from datetime import datetime, timezone
from typing import Any

import structlog

_log = structlog.get_logger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _env(name: str) -> str | None:
    val = os.environ.get(name)
    if val and val.strip() and not val.strip().startswith("#"):
        return val.strip()
    return None


# ── Custom Search (news + web) ──────────────────────────────────────────────

# Curated finance news domains for site-restricted queries.
FINANCE_NEWS_SITES: tuple[str, ...] = (
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "marketwatch.com",
    "barrons.com",
    "sec.gov",
    "federalreserve.gov",
    "treasury.gov",
)


class CustomSearchClient:
    """Google Programmable Search Engine (Custom Search JSON API).

    Setup:
        1. Create a CSE at https://programmablesearchengine.google.com/
           with "Search the entire web" enabled.
        2. Copy the cx ID into ``GOOGLE_CSE_ID``.
        3. Enable the Custom Search API in your Google Cloud project and
           paste the API key into ``GOOGLE_API_KEY``.

    Free tier: 100 queries/day, then $5 per 1000 (cap configurable in console).
    """

    def __init__(self) -> None:
        self.api_key = _env("GOOGLE_API_KEY")
        self.cse_id = _env("GOOGLE_CSE_ID")
        self._service = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.cse_id)

    def _client(self):
        if self._service is None:
            from googleapiclient.discovery import build  # local import: optional dep
            self._service = build("customsearch", "v1", developerKey=self.api_key,
                                  cache_discovery=False)
        return self._service

    def search(self, query: str, num: int = 10, date_restrict: str | None = "d7",
               sites: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        """Run a single search. Returns a list of normalised result dicts.

        ``date_restrict`` examples: ``d1`` (last day), ``w1``, ``m1``, ``y1``.
        ``sites`` restricts to a tuple of domains via OR'd ``site:`` operators.
        """
        if not self.configured:
            return []
        q = query
        if sites:
            q = f"({query}) " + " OR ".join(f"site:{s}" for s in sites)
        try:
            req = self._client().cse().list(
                q=q, cx=self.cse_id, num=min(max(num, 1), 10),
                dateRestrict=date_restrict,
            )
            resp = req.execute()
        except Exception as exc:  # noqa: BLE001 — googleapiclient raises many shapes
            _log.warning("google_cse_search_failed", query=query, error=str(exc))
            return []
        items = resp.get("items") or []
        out: list[dict[str, Any]] = []
        for it in items:
            out.append({
                "datetime": (it.get("pagemap", {}).get("metatags", [{}])[0]
                             .get("article:published_time", "")),
                "headline": it.get("title", ""),
                "snippet": it.get("snippet", ""),
                "source": (it.get("displayLink") or "").replace("www.", ""),
                "url": it.get("link", ""),
            })
        return out

    def news(self, query: str, num: int = 10, hours: int = 24) -> list[dict[str, Any]]:
        """Convenience: financial news search restricted to curated sites."""
        date_restrict = f"d{max(1, hours // 24)}" if hours >= 24 else "d1"
        return self.search(query, num=num, date_restrict=date_restrict,
                           sites=FINANCE_NEWS_SITES)


# ── YouTube Data API v3 ─────────────────────────────────────────────────────


class YouTubeClient:
    """YouTube Data API v3 — search and video metadata.

    Useful for: earnings calls, Fed pressers, CEO interviews, sentiment
    proxies (view counts on retail-trader videos for a ticker).

    Free quota: 10,000 units/day. ``search.list`` costs 100 units, so the
    practical cap is ~100 searches/day; budget accordingly.
    """

    def __init__(self) -> None:
        self.api_key = _env("GOOGLE_API_KEY")
        self._service = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _client(self):
        if self._service is None:
            from googleapiclient.discovery import build
            self._service = build("youtube", "v3", developerKey=self.api_key,
                                  cache_discovery=False)
        return self._service

    def search(self, query: str, max_results: int = 10, days: int = 7,
               order: str = "relevance") -> list[dict[str, Any]]:
        """Search videos. Returns normalised dicts (id, title, channel, published, url)."""
        if not self.configured:
            return []
        published_after = None
        if days:
            ts = time.time() - days * 86400
            published_after = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
        try:
            resp = self._client().search().list(
                q=query, part="snippet", type="video",
                maxResults=min(max(max_results, 1), 50),
                order=order,
                publishedAfter=published_after,
            ).execute()
        except Exception as exc:  # noqa: BLE001
            _log.warning("youtube_search_failed", query=query, error=str(exc))
            return []
        out: list[dict[str, Any]] = []
        for it in resp.get("items", []):
            sn = it.get("snippet", {})
            vid = (it.get("id") or {}).get("videoId", "")
            out.append({
                "video_id": vid,
                "title": sn.get("title", ""),
                "channel": sn.get("channelTitle", ""),
                "published_at": sn.get("publishedAt", ""),
                "description": sn.get("description", ""),
                "url": f"https://www.youtube.com/watch?v={vid}" if vid else "",
            })
        return out

    def video_stats(self, video_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Bulk video statistics (views, likes, comments). Costs 1 unit per call."""
        if not self.configured or not video_ids:
            return {}
        try:
            resp = self._client().videos().list(
                id=",".join(video_ids[:50]), part="statistics,contentDetails",
            ).execute()
        except Exception as exc:  # noqa: BLE001
            _log.warning("youtube_video_stats_failed", error=str(exc))
            return {}
        out: dict[str, dict[str, Any]] = {}
        for it in resp.get("items", []):
            stats = it.get("statistics", {})
            out[it.get("id", "")] = {
                "views": int(stats.get("viewCount", 0) or 0),
                "likes": int(stats.get("likeCount", 0) or 0),
                "comments": int(stats.get("commentCount", 0) or 0),
                "duration": (it.get("contentDetails") or {}).get("duration", ""),
            }
        return out


# ── Google Trends (pytrends) ────────────────────────────────────────────────


class GoogleTrendsClient:
    """pytrends wrapper. No key needed.

    ``retail_interest`` returns last-7d normalised search interest (0-100)
    for a list of keywords. Use for: ticker symbols, macro fears
    ("recession", "stock market crash"), DFV-style retail-sentiment edge.
    """

    def __init__(self, hl: str = "en-US", tz: int = 360) -> None:
        from pytrends.request import TrendReq  # local import: heavy
        self._client = TrendReq(hl=hl, tz=tz)

    def retail_interest(self, keywords: list[str], timeframe: str = "now 7-d",
                        geo: str = "US") -> dict[str, Any]:
        """Returns {keyword: latest_value, _series: {keyword: [(ts, val), ...]}}."""
        if not keywords:
            return {"ok": False, "ts": _utc_now(), "values": {}}
        # pytrends caps at 5 keywords per payload
        out_vals: dict[str, float | None] = {}
        out_series: dict[str, list[tuple[str, int]]] = {}
        for chunk_start in range(0, len(keywords), 5):
            chunk = keywords[chunk_start:chunk_start + 5]
            try:
                self._client.build_payload(chunk, timeframe=timeframe, geo=geo)
                df = self._client.interest_over_time()
            except Exception as exc:  # noqa: BLE001 — pytrends raises many shapes
                _log.warning("pytrends_failed", keywords=chunk, error=str(exc))
                for kw in chunk:
                    out_vals[kw] = None
                continue
            if df is None or df.empty:
                for kw in chunk:
                    out_vals[kw] = None
                continue
            for kw in chunk:
                if kw in df.columns:
                    series = [(ts.isoformat(), int(v)) for ts, v in df[kw].items()]
                    out_series[kw] = series
                    out_vals[kw] = float(df[kw].iloc[-1])
                else:
                    out_vals[kw] = None
        return {
            "ok": any(v is not None for v in out_vals.values()),
            "ts": _utc_now(),
            "timeframe": timeframe,
            "geo": geo,
            "values": out_vals,
            "series": out_series,
        }

    def trending_searches(self, geo: str = "united_states") -> list[str]:
        try:
            df = self._client.trending_searches(pn=geo)
        except Exception as exc:  # noqa: BLE001
            _log.warning("pytrends_trending_failed", error=str(exc))
            return []
        if df is None or df.empty:
            return []
        return [str(x) for x in df.iloc[:, 0].tolist()]


# ── Gemini (google-genai SDK) ───────────────────────────────────────────────


class GeminiClient:
    """Gemini chat completion. Second-opinion LLM beside Grok.

    Free tier on flash models: 1500 req/day, 15 req/min, 1M tokens/day.
    """

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self.api_key = _env("GOOGLE_AI_API_KEY")
        self.model = model
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from google import genai  # local import: optional dep
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def ask(self, prompt: str, system: str | None = None,
            temperature: float = 0.2) -> dict[str, Any]:
        """Single-turn ask. Returns {ok, text, model, error?}."""
        if not self.configured:
            return {"ok": False, "text": "", "error": "GOOGLE_AI_API_KEY not set"}
        from google.genai import types  # local import
        try:
            cfg_kwargs: dict[str, Any] = {"temperature": temperature}
            if system:
                cfg_kwargs["system_instruction"] = system
            resp = self._get_client().models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            return {"ok": True, "text": (resp.text or "").strip(), "model": self.model}
        except Exception as exc:  # noqa: BLE001 — SDK raises many shapes
            _log.warning("gemini_ask_failed", error=str(exc))
            return {"ok": False, "text": "", "error": str(exc), "model": self.model}
