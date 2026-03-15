"""
ClawHub API Client for AAC
===========================

HTTP client for the ClawHub skill marketplace (https://clawhub.ai).
Enables AAC to search, browse, and install skills from the 22,990+ skill registry.

Usage:
    from integrations.clawhub_client import get_clawhub_client

    client = get_clawhub_client()
    results = await client.search_skills("trading arbitrage")
    skill = await client.get_skill("self-improving-agent")
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# ClawHub base URL
CLAWHUB_API_BASE = "https://clawhub.ai/api"

# AAC-relevant skill categories
AAC_SKILL_CATEGORIES = [
    "self-improving-agent",
    "proactive-agent",
    "ontology",
    "github",
    "brave-search",
    "multi-search-engine",
    "summarize",
    "agent-browser",
    "weather",
    "skill-vetter",
    "api-gateway",
    "find-skills",
]


# Curated AAC-relevant skills for offline fallback
_CURATED_SKILLS: List[Dict[str, Any]] = [
    {"name": "self-improving-agent", "description": "Agent that improves itself by analyzing errors and adjusting behavior", "downloads": 208000, "tags": ["agent", "self-improving"]},
    {"name": "find-skills", "description": "Search and discover skills from ClawHub registry", "downloads": 203000, "tags": ["meta", "discovery"]},
    {"name": "summarize", "description": "Summarize text, articles, and documents", "downloads": 154000, "tags": ["text", "nlp"]},
    {"name": "agent-browser", "description": "Browser automation agent for web research", "downloads": 126000, "tags": ["browser", "automation"]},
    {"name": "gog", "description": "Graph of Graphs — knowledge graph agent", "downloads": 112000, "tags": ["knowledge", "graph"]},
    {"name": "github", "description": "GitHub integration — PRs, issues, repos", "downloads": 107000, "tags": ["github", "dev"]},
    {"name": "ontology", "description": "Knowledge graph and ontology management", "downloads": 103000, "tags": ["knowledge", "ontology"]},
    {"name": "proactive-agent", "description": "Agent that proactively monitors and acts on triggers", "downloads": 97000, "tags": ["agent", "proactive"]},
    {"name": "weather", "description": "Weather data and forecasts", "downloads": 92000, "tags": ["weather", "data"]},
    {"name": "skill-vetter", "description": "Security vetting for ClawHub skills", "downloads": 92000, "tags": ["security", "vetting"]},
    {"name": "brave-search", "description": "Web search via Brave Search API", "downloads": 39000, "tags": ["search", "web"]},
    {"name": "multi-search-engine", "description": "Aggregate results from multiple search engines", "downloads": 51000, "tags": ["search", "aggregation"]},
    {"name": "api-gateway", "description": "API gateway for routing and rate limiting", "downloads": 43000, "tags": ["api", "gateway"]},
]


@dataclass
class ClawHubSkillInfo:
    """Parsed metadata for a ClawHub skill"""
    name: str
    description: str = ""
    author: str = ""
    downloads: int = 0
    version: str = ""
    tags: List[str] = field(default_factory=list)
    install_cmd: str = ""


@dataclass
class ClawHubClient:
    """
    Client for the ClawHub skill marketplace.

    Provides search, browse, and install-command generation.
    Actual HTTP calls require an async HTTP library (httpx or aiohttp).
    Falls back to offline mode with curated AAC skill list when unavailable.
    """

    api_key: str = field(default="", repr=False)
    base_url: str = CLAWHUB_API_BASE
    skills_dir: str = ""
    daily_spend_limit: float = 10.0
    _http_client: Any = field(default=None, repr=False)

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def _get_http_client(self):
        """Lazily initialize HTTP client"""
        if self._http_client is not None:
            return self._http_client
        try:
            import httpx
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "AAC-Trading-Bot/2.0",
                },
                timeout=15.0,
            )
            return self._http_client
        except ImportError:
            logger.debug("httpx not installed — ClawHub client in offline mode")
            return None

    async def search_skills(
        self, query: str, limit: int = 20
    ) -> List[ClawHubSkillInfo]:
        """Search ClawHub for skills matching a query."""
        client = await self._get_http_client()
        if client is None:
            return self._offline_search(query)

        try:
            resp = await client.get(
                "/skills/search",
                params={"q": query, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
            return [self._parse_skill(s) for s in data.get("skills", [])]
        except Exception as e:
            logger.warning(f"ClawHub search failed: {e}")
            return self._offline_search(query)

    async def get_skill(self, name: str) -> Optional[ClawHubSkillInfo]:
        """Get details for a specific skill by name."""
        client = await self._get_http_client()
        if client is None:
            return self._offline_get_skill(name)

        try:
            resp = await client.get(f"/skills/{quote_plus(name)}")
            resp.raise_for_status()
            return self._parse_skill(resp.json())
        except Exception as e:
            logger.warning(f"ClawHub get_skill({name}) failed: {e}")
            return self._offline_get_skill(name)

    async def list_popular(self, limit: int = 20) -> List[ClawHubSkillInfo]:
        """List popular skills sorted by downloads."""
        client = await self._get_http_client()
        if client is None:
            return self._offline_popular()

        try:
            resp = await client.get(
                "/skills/popular", params={"limit": limit}
            )
            resp.raise_for_status()
            data = resp.json()
            return [self._parse_skill(s) for s in data.get("skills", [])]
        except Exception as e:
            logger.warning(f"ClawHub list_popular failed: {e}")
            return self._offline_popular()

    def get_install_command(self, skill_name: str) -> str:
        """Return the npx install command for a skill."""
        return f"npx clawhub@latest install {skill_name}"

    async def close(self):
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    # ── Parsing ──

    @staticmethod
    def _parse_skill(data: Dict[str, Any]) -> ClawHubSkillInfo:
        return ClawHubSkillInfo(
            name=data.get("name", ""),
            description=data.get("description", ""),
            author=data.get("author", ""),
            downloads=data.get("downloads", 0),
            version=data.get("version", ""),
            tags=data.get("tags", []),
            install_cmd=f"npx clawhub@latest install {data.get('name', '')}",
        )

    # ── Offline fallback (curated AAC-relevant skills) ──

    def _offline_search(self, query: str) -> List[ClawHubSkillInfo]:
        q = query.lower()
        return [
            self._parse_skill(s)
            for s in _CURATED_SKILLS
            if q in s["name"] or q in s.get("description", "").lower()
            or any(q in t for t in s.get("tags", []))
        ]

    def _offline_get_skill(self, name: str) -> Optional[ClawHubSkillInfo]:
        for s in _CURATED_SKILLS:
            if s["name"] == name:
                return self._parse_skill(s)
        return None

    def _offline_popular(self) -> List[ClawHubSkillInfo]:
        return [self._parse_skill(s) for s in sorted(
            _CURATED_SKILLS, key=lambda x: x["downloads"], reverse=True
        )]

    def get_status(self) -> Dict[str, Any]:
        """Return client status for health checks."""
        return {
            "configured": self.is_configured(),
            "base_url": self.base_url,
            "skills_dir": self.skills_dir,
            "daily_spend_limit": self.daily_spend_limit,
            "curated_skills": len(_CURATED_SKILLS),
        }


# ── Singleton ──

_client_instance: Optional[ClawHubClient] = None


def get_clawhub_client() -> ClawHubClient:
    """Get or create the singleton ClawHub client."""
    global _client_instance
    if _client_instance is None:
        _client_instance = ClawHubClient(
            api_key=os.environ.get("CLAWHUB_API_KEY", ""),
            skills_dir=os.environ.get("OPENCLAW_SKILLS_DIR", ""),
            daily_spend_limit=float(os.environ.get("OPENCLAW_DAILY_SPEND_LIMIT", "10.0")),
        )
    return _client_instance
