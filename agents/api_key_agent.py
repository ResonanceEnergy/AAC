#!/usr/bin/env python3
"""
API Key Provisioning & Validation Agent
========================================
Autonomous agent that manages the lifecycle of free-tier API keys:
  1. Detects which keys are missing from .env
  2. Validates configured keys with live health checks
  3. Reports status, rate-limit headroom, and data quality
  4. Provides sign-up URLs for missing keys

Covers 7 free-tier APIs:
  - FRED (Federal Reserve Economic Data)
  - Polygon.io (stocks, options, crypto)
  - Finnhub (stocks, earnings, news)
  - Alpha Vantage (fundamentals, forex)
  - NewsAPI (global news headlines)
  - Etherscan (Ethereum on-chain data)
  - Tradier (options chains, Greeks)

Usage:
    from agents.api_key_agent import APIKeyAgent
    agent = APIKeyAgent()
    report = await agent.run()
    print(report.summary())

CLI:
    python agents/api_key_agent.py
    python agents/api_key_agent.py --validate
    python agents/api_key_agent.py --missing
    python agents/api_key_agent.py --add FRED_API_KEY=your_key_here
"""

import asyncio
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, load_env_file

logger = logging.getLogger("AAC.APIKeyAgent")

# ─── API Definitions ───────────────────────────────────────────

@dataclass
class APIDefinition:
    """Definition of a managed API"""
    name: str
    env_var: str
    signup_url: str
    base_url: str
    free_tier: str
    test_endpoint: str
    test_params: Dict[str, str] = field(default_factory=dict)
    auth_style: str = "query"       # "query", "header", "bearer"
    auth_param: str = "apikey"      # query param or header name
    expected_field: str = ""        # field to check in response


MANAGED_APIS: List[APIDefinition] = [
    APIDefinition(
        name="FRED",
        env_var="FRED_API_KEY",
        signup_url="https://fred.stlouisfed.org/docs/api/api_key.html",
        base_url="https://api.stlouisfed.org/fred",
        free_tier="Unlimited requests (macro/economic data)",
        test_endpoint="/series/observations",
        test_params={"series_id": "DFF", "limit": "1", "sort_order": "desc", "file_type": "json"},
        auth_style="query",
        auth_param="api_key",
        expected_field="observations",
    ),
    APIDefinition(
        name="Polygon.io",
        env_var="POLYGON_API_KEY",
        signup_url="https://polygon.io/dashboard/signup",
        base_url="https://api.polygon.io",
        free_tier="5 calls/min (stocks, options, crypto)",
        test_endpoint="/v2/aggs/ticker/SPY/prev",
        test_params={},
        auth_style="query",
        auth_param="apiKey",
        expected_field="results",
    ),
    APIDefinition(
        name="Finnhub",
        env_var="FINNHUB_API_KEY",
        signup_url="https://finnhub.io/register",
        base_url="https://finnhub.io/api/v1",
        free_tier="60 calls/min (stocks, news, sentiment)",
        test_endpoint="/quote",
        test_params={"symbol": "AAPL"},
        auth_style="header",
        auth_param="X-Finnhub-Token",
        expected_field="c",
    ),
    APIDefinition(
        name="Alpha Vantage",
        env_var="ALPHAVANTAGE_API_KEY",
        signup_url="https://www.alphavantage.co/support/#api-key",
        base_url="https://www.alphavantage.co",
        free_tier="25 calls/day (fundamentals, forex)",
        test_endpoint="/query",
        test_params={"function": "GLOBAL_QUOTE", "symbol": "IBM"},
        auth_style="query",
        auth_param="apikey",
        expected_field="Global Quote",
    ),
    APIDefinition(
        name="NewsAPI",
        env_var="NEWS_API_KEY",
        signup_url="https://newsapi.org/register",
        base_url="https://newsapi.org/v2",
        free_tier="100 calls/day (global news)",
        test_endpoint="/top-headlines",
        test_params={"country": "us", "pageSize": "1"},
        auth_style="header",
        auth_param="X-Api-Key",
        expected_field="articles",
    ),
    APIDefinition(
        name="Etherscan",
        env_var="ETHERSCAN_API_KEY",
        signup_url="https://etherscan.io/register",
        base_url="https://api.etherscan.io",
        free_tier="5 calls/sec (Ethereum on-chain)",
        test_endpoint="/api",
        test_params={"module": "stats", "action": "ethprice"},
        auth_style="query",
        auth_param="apikey",
        expected_field="result",
    ),
    APIDefinition(
        name="Tradier",
        env_var="TRADIER_API_KEY",
        signup_url="https://developer.tradier.com/user/sign_up",
        base_url="https://sandbox.tradier.com/v1",
        free_tier="Free sandbox (options chains, Greeks)",
        test_endpoint="/markets/quotes",
        test_params={"symbols": "SPY"},
        auth_style="bearer",
        auth_param="Authorization",
        expected_field="quotes",
    ),
]

# ─── Validation Results ────────────────────────────────────────

@dataclass
class APIValidationResult:
    """Result of validating a single API"""
    api_name: str
    env_var: str
    key_present: bool
    key_valid: bool = False
    response_time_ms: float = 0.0
    sample_data: str = ""
    error: str = ""
    signup_url: str = ""
    free_tier: str = ""


@dataclass
class AgentReport:
    """Full agent report"""
    timestamp: datetime = field(default_factory=datetime.now)
    results: List[APIValidationResult] = field(default_factory=list)
    total: int = 0
    configured: int = 0
    validated: int = 0
    missing: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  API KEY AGENT — STATUS REPORT",
            f"  {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            f"  Total APIs managed:  {self.total}",
            f"  Configured (key set): {self.configured}",
            f"  Validated (live OK):  {self.validated}",
            f"  Missing keys:        {self.missing}",
            "",
        ]

        # Validated APIs
        ok = [r for r in self.results if r.key_valid]
        if ok:
            lines.append("  LIVE & VALIDATED:")
            for r in ok:
                lines.append(f"    [OK] {r.api_name:<16} {r.response_time_ms:>6.0f}ms  {r.sample_data}")

        # Configured but failed
        bad = [r for r in self.results if r.key_present and not r.key_valid]
        if bad:
            lines.append("")
            lines.append("  CONFIGURED BUT FAILED:")
            for r in bad:
                lines.append(f"    [!!] {r.api_name:<16} {r.error}")

        # Missing
        miss = [r for r in self.results if not r.key_present]
        if miss:
            lines.append("")
            lines.append("  MISSING KEYS (sign up for free):")
            for r in miss:
                lines.append(f"    [ ] {r.api_name:<16} {r.free_tier}")
                lines.append(f"        {r.signup_url}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Agent ──────────────────────────────────────────────────────

class APIKeyAgent:
    """
    Autonomous agent for API key provisioning and validation.

    Lifecycle:
      1. Scan .env for configured keys
      2. Hit each API with a lightweight health-check call
      3. Report which are live, broken, or missing
      4. Optionally add new keys to .env
    """

    def __init__(self, env_path: Optional[Path] = None):
        self.env_path = env_path or (PROJECT_ROOT / ".env")
        self.apis = MANAGED_APIS
        load_env_file()

    async def run(self, validate: bool = True) -> AgentReport:
        """Execute the full agent cycle."""
        report = AgentReport(total=len(self.apis))

        for api_def in self.apis:
            result = await self._check_api(api_def, validate=validate)
            report.results.append(result)
            if result.key_present:
                report.configured += 1
            else:
                report.missing += 1
            if result.key_valid:
                report.validated += 1

        return report

    async def _check_api(self, api_def: APIDefinition, validate: bool = True) -> APIValidationResult:
        """Check a single API: key presence + optional live validation."""
        key = os.environ.get(api_def.env_var, "").strip()
        result = APIValidationResult(
            api_name=api_def.name,
            env_var=api_def.env_var,
            key_present=bool(key),
            signup_url=api_def.signup_url,
            free_tier=api_def.free_tier,
        )

        if not key:
            return result

        if not validate:
            result.key_valid = True
            return result

        # Live validation
        import aiohttp
        import time

        url = api_def.base_url + api_def.test_endpoint
        params = dict(api_def.test_params)
        headers: Dict[str, str] = {}

        if api_def.auth_style == "query":
            params[api_def.auth_param] = key
        elif api_def.auth_style == "header":
            headers[api_def.auth_param] = key
        elif api_def.auth_style == "bearer":
            headers["Authorization"] = f"Bearer {key}"
            headers["Accept"] = "application/json"

        start = time.time()
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                async with session.get(url, params=params, headers=headers) as resp:
                    elapsed = (time.time() - start) * 1000
                    result.response_time_ms = elapsed

                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        if api_def.expected_field and api_def.expected_field in (data if isinstance(data, dict) else {}):
                            result.key_valid = True
                            # Extract a meaningful preview
                            result.sample_data = self._extract_preview(api_def.name, data)
                        elif not api_def.expected_field:
                            result.key_valid = True
                            result.sample_data = "HTTP 200"
                        else:
                            result.error = f"HTTP 200 but missing '{api_def.expected_field}' in response"
                    elif resp.status == 401:
                        result.error = "Invalid API key (401 Unauthorized)"
                    elif resp.status == 403:
                        result.error = "Access denied (403 Forbidden)"
                    elif resp.status == 429:
                        # Rate limited means the key IS valid
                        result.key_valid = True
                        result.sample_data = "Rate limited (key valid)"
                    else:
                        body = await resp.text()
                        result.error = f"HTTP {resp.status}: {body[:100]}"

        except asyncio.TimeoutError:
            result.error = "Request timed out (15s)"
        except Exception as e:
            result.error = str(e)[:120]

        return result

    def _extract_preview(self, api_name: str, data: dict) -> str:
        """Extract a human-readable preview from API response."""
        try:
            if api_name == "FRED":
                obs = data.get("observations", [{}])[0]
                return f"Fed Funds Rate: {obs.get('value', '?')}% ({obs.get('date', '?')})"
            elif api_name == "Polygon.io":
                bar = data.get("results", [{}])[0]
                return f"SPY prev close: ${bar.get('c', '?')}"
            elif api_name == "Finnhub":
                return f"AAPL: ${data.get('c', '?')} (change: {data.get('dp', '?')}%)"
            elif api_name == "Alpha Vantage":
                gq = data.get("Global Quote", {})
                return f"IBM: ${gq.get('05. price', '?')} ({gq.get('10. change percent', '?')})"
            elif api_name == "NewsAPI":
                articles = data.get("articles", [])
                if articles:
                    return f"Top: {articles[0].get('title', '?')[:50]}..."
                return f"{data.get('totalResults', 0)} articles"
            elif api_name == "Etherscan":
                res = data.get("result", {})
                if isinstance(res, dict):
                    return f"ETH: ${res.get('ethusd', '?')}"
                return str(res)[:60]
            elif api_name == "Tradier":
                quotes = data.get("quotes", {})
                q = quotes.get("quote", {}) if isinstance(quotes, dict) else {}
                return f"SPY: ${q.get('last', '?')}"
        except (IndexError, KeyError, TypeError):
            pass
        return "OK"

    def add_key(self, env_var: str, key_value: str) -> bool:
        """
        Add or update an API key in the .env file.
        Returns True if successful.
        """
        if not self.env_path.exists():
            logger.error(f".env file not found at {self.env_path}")
            return False

        # Validate env_var is one we manage
        valid_vars = {api.env_var for api in self.apis}
        if env_var not in valid_vars:
            logger.error(f"{env_var} is not a managed API key. Valid: {valid_vars}")
            return False

        # Sanitize: key should be alphanumeric/dashes/underscores only
        if not re.match(r'^[\w\-]+$', key_value):
            logger.error("API key contains invalid characters")
            return False

        content = self.env_path.read_text(encoding="utf-8")
        pattern = rf'^({re.escape(env_var)}=)(.*)$'
        match = re.search(pattern, content, re.MULTILINE)

        if match:
            content = re.sub(pattern, rf'\g<1>{key_value}', content, count=1, flags=re.MULTILINE)
        else:
            content += f"\n{env_var}={key_value}\n"

        self.env_path.write_text(content, encoding="utf-8")
        os.environ[env_var] = key_value
        logger.info(f"Added {env_var} to .env")
        return True

    def get_missing(self) -> List[APIDefinition]:
        """Return list of APIs without configured keys."""
        return [api for api in self.apis if not os.environ.get(api.env_var, "").strip()]

    def get_configured(self) -> List[APIDefinition]:
        """Return list of APIs with configured keys."""
        return [api for api in self.apis if os.environ.get(api.env_var, "").strip()]


# ─── CLI ────────────────────────────────────────────────────────

async def _main():
    import argparse
    import io

    # Windows UTF-8 fix
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="API Key Provisioning Agent")
    parser.add_argument("--validate", action="store_true", help="Run live validation (default)")
    parser.add_argument("--missing", action="store_true", help="Show only missing keys")
    parser.add_argument("--add", type=str, help="Add key: --add FRED_API_KEY=abc123")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    agent = APIKeyAgent()

    if args.add:
        if "=" not in args.add:
            print("Usage: --add ENV_VAR=key_value")
            sys.exit(1)
        var, val = args.add.split("=", 1)
        if agent.add_key(var, val):
            print(f"Added {var}")
        else:
            print(f"Failed to add {var}")
            sys.exit(1)
        return

    if args.missing:
        missing = agent.get_missing()
        if not missing:
            print("All 7 API keys are configured!")
        else:
            print(f"\n{len(missing)} missing API key(s):\n")
            for api in missing:
                print(f"  {api.env_var:<24} {api.name}")
                print(f"  {'':<24} {api.signup_url}")
                print(f"  {'':<24} {api.free_tier}")
                print()
        return

    report = await agent.run(validate=True)

    if args.json:
        import json as _json
        out = {
            "timestamp": report.timestamp.isoformat(),
            "total": report.total,
            "configured": report.configured,
            "validated": report.validated,
            "missing": report.missing,
            "results": [
                {
                    "api": r.api_name,
                    "env_var": r.env_var,
                    "key_present": r.key_present,
                    "key_valid": r.key_valid,
                    "response_time_ms": r.response_time_ms,
                    "sample_data": r.sample_data,
                    "error": r.error,
                }
                for r in report.results
            ],
        }
        print(_json.dumps(out, indent=2))
    else:
        print(report.summary())


if __name__ == "__main__":
    asyncio.run(_main())
