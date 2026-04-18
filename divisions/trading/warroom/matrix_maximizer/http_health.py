"""
MATRIX MAXIMIZER — HTTP Health Check
=======================================
Tests every API endpoint the system connects to.

Run:
    python -m strategies.matrix_maximizer.http_health
    python -m strategies.matrix_maximizer.http_health --verbose
    python -m strategies.matrix_maximizer.http_health --fix
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Ensure .env is loaded ─────────────────────────────────────────────────
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        # Manual .env parse (no dotenv dependency required)
        with open(_env_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and val and key not in os.environ:
                    os.environ[key] = val


@dataclass
class EndpointResult:
    name: str
    url: str
    status: str           # "OK", "FAIL", "SKIP", "WARN"
    latency_ms: float = 0
    http_code: int = 0
    detail: str = ""
    data_preview: str = ""


class HTTPHealthCheck:
    """Test all MATRIX MAXIMIZER API endpoints."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.results: List[EndpointResult] = []

        # Load API keys
        self._polygon = os.getenv("POLYGON_API_KEY", "")
        self._fred = os.getenv("FRED_API_KEY", "")
        self._finnhub = os.getenv("FINNHUB_API_KEY", "")
        self._uw = os.getenv("UNUSUAL_WHALES_API_KEY", "")
        self._smtp_host = os.getenv("SMTP_HOST", "")
        self._smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self._smtp_user = os.getenv("SMTP_USER", "")
        self._smtp_pass = os.getenv("SMTP_PASSWORD", "")
        self._telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._telegram_chat = os.getenv("TELEGRAM_CHAT_ID", "")

    def _http_get(self, url: str, headers: Optional[Dict[str, str]] = None,
                  timeout: int = 15) -> tuple:
        """GET request returning (status_code, json_data, latency_ms)."""
        t0 = time.time()
        try:
            req = urllib.request.Request(url)
            # Default User-Agent; can be overridden via headers dict
            if not headers or "User-Agent" not in headers:
                req.add_header("User-Agent", "AAC-MatrixMaximizer/1.0-HealthCheck")
            if headers:
                for k, v in headers.items():
                    req.add_header(k, v)
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latency = (time.time() - t0) * 1000
                return resp.status, data, latency
        except urllib.error.HTTPError as e:
            latency = (time.time() - t0) * 1000
            return e.code, None, latency
        except Exception as e:
            latency = (time.time() - t0) * 1000
            return 0, str(e), latency

    def _test(self, name: str, url: str, headers: Optional[Dict[str, str]] = None,
              validator: Optional[callable] = None, skip_reason: str = "") -> EndpointResult:
        """Test a single endpoint."""
        if skip_reason:
            r = EndpointResult(name=name, url=url[:80], status="SKIP", detail=skip_reason)
            self.results.append(r)
            return r

        code, data, latency = self._http_get(url, headers)
        preview = ""
        if isinstance(data, dict):
            preview = json.dumps(data, default=str)[:200]

        if code == 200:
            detail = "HTTP 200 OK"
            status = "OK"
            if validator and data:
                try:
                    extra = validator(data)
                    if extra:
                        detail += f" — {extra}"
                except Exception as e:
                    status = "WARN"
                    detail = f"HTTP 200 but validation failed: {e}"
        elif code == 403:
            status = "FAIL"
            detail = "HTTP 403 Forbidden — invalid/expired API key or subscription"
        elif code == 404:
            status = "FAIL"
            detail = "HTTP 404 Not Found — endpoint retired or URL changed"
        elif code == 429:
            status = "WARN"
            detail = "HTTP 429 Rate limited — API responding but throttled"
        elif code == 0:
            status = "FAIL"
            detail = f"Connection failed: {data}"
        else:
            status = "FAIL"
            detail = f"HTTP {code}"

        r = EndpointResult(
            name=name, url=url[:80], status=status,
            latency_ms=round(latency, 1), http_code=code,
            detail=detail, data_preview=preview if self.verbose else "",
        )
        self.results.append(r)
        return r

    # ══════════════════════════════════════════════════════════════════════
    # API KEY INVENTORY
    # ══════════════════════════════════════════════════════════════════════

    def check_keys(self) -> Dict[str, str]:
        """Report which API keys are configured."""
        keys = {
            "POLYGON_API_KEY": self._polygon,
            "FRED_API_KEY": self._fred,
            "FINNHUB_API_KEY": self._finnhub,
            "UNUSUAL_WHALES_API_KEY": self._uw,
            "SMTP_HOST": self._smtp_host,
            "SMTP_USER": self._smtp_user,
            "SMTP_PASSWORD": self._smtp_pass,
            "TELEGRAM_BOT_TOKEN": self._telegram_token,
            "TELEGRAM_CHAT_ID": self._telegram_chat,
        }
        report: Dict[str, str] = {}
        for k, v in keys.items():
            if not v:
                report[k] = "MISSING"
            elif k.endswith("PASSWORD") or k.endswith("TOKEN"):
                report[k] = f"SET ({v[:4]}...)"
            else:
                report[k] = f"SET ({v[:8]}...)"
        return report

    # ══════════════════════════════════════════════════════════════════════
    # POLYGON ENDPOINTS
    # ══════════════════════════════════════════════════════════════════════

    def test_polygon_snapshot(self) -> EndpointResult:
        url = (f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/SPY"
               f"?apiKey={self._polygon}")
        return self._test(
            "Polygon Stock Snapshot (SPY)", url,
            skip_reason="" if self._polygon else "No POLYGON_API_KEY",
            validator=lambda d: f"price=${d['ticker']['day'].get('c', 'N/A')}" if "ticker" in d else "no ticker data",
        )

    def test_polygon_prev_close(self) -> EndpointResult:
        url = (f"https://api.polygon.io/v2/aggs/ticker/SPY/prev"
               f"?apiKey={self._polygon}")
        return self._test(
            "Polygon Prev Close (SPY)", url,
            skip_reason="" if self._polygon else "No POLYGON_API_KEY",
            validator=lambda d: f"close=${d['results'][0]['c']}" if d.get("results") else "no results",
        )

    def test_polygon_options_chain(self) -> EndpointResult:
        today = datetime.utcnow().date().isoformat()
        url = (
            f"https://api.polygon.io/v3/reference/options/contracts"
            f"?underlying_ticker=SPY&contract_type=put"
            f"&expiration_date.gte={today}&limit=5"
            f"&apiKey={self._polygon}"
        )
        return self._test(
            "Polygon Options Chain (SPY puts)", url,
            skip_reason="" if self._polygon else "No POLYGON_API_KEY",
            validator=lambda d: f"{len(d.get('results', []))} contracts" if "results" in d else "no results",
        )

    # ══════════════════════════════════════════════════════════════════════
    # FRED ENDPOINTS
    # ══════════════════════════════════════════════════════════════════════

    def test_fred_vix(self) -> EndpointResult:
        url = (f"https://api.stlouisfed.org/fred/series/observations"
               f"?series_id=VIXCLS&api_key={self._fred}"
               f"&file_type=json&sort_order=desc&limit=1")
        return self._test(
            "FRED VIX (VIXCLS)", url,
            skip_reason="" if self._fred else "No FRED_API_KEY",
            validator=lambda d: f"VIX={d['observations'][0]['value']}" if d.get("observations") else "no obs",
        )

    def test_fred_oil(self) -> EndpointResult:
        url = (f"https://api.stlouisfed.org/fred/series/observations"
               f"?series_id=DCOILWTICO&api_key={self._fred}"
               f"&file_type=json&sort_order=desc&limit=1")
        return self._test(
            "FRED Oil (DCOILWTICO)", url,
            skip_reason="" if self._fred else "No FRED_API_KEY",
            validator=lambda d: f"WTI=${d['observations'][0]['value']}" if d.get("observations") else "no obs",
        )

    def test_fred_yield_curve(self) -> EndpointResult:
        url = (f"https://api.stlouisfed.org/fred/series/observations"
               f"?series_id=T10Y2Y&api_key={self._fred}"
               f"&file_type=json&sort_order=desc&limit=1")
        return self._test(
            "FRED Yield Curve (T10Y2Y)", url,
            skip_reason="" if self._fred else "No FRED_API_KEY",
            validator=lambda d: f"spread={d['observations'][0]['value']}%" if d.get("observations") else "no obs",
        )

    def test_fred_hy_spread(self) -> EndpointResult:
        url = (f"https://api.stlouisfed.org/fred/series/observations"
               f"?series_id=BAMLH0A0HYM2&api_key={self._fred}"
               f"&file_type=json&sort_order=desc&limit=1")
        return self._test(
            "FRED HY Spread (BAMLH0A0HYM2)", url,
            skip_reason="" if self._fred else "No FRED_API_KEY",
            validator=lambda d: f"spread={d['observations'][0]['value']}%" if d.get("observations") else "no obs",
        )

    # ══════════════════════════════════════════════════════════════════════
    # FINNHUB ENDPOINTS
    # ══════════════════════════════════════════════════════════════════════

    def test_finnhub_earnings(self) -> EndpointResult:
        today = datetime.utcnow().date().isoformat()
        url = (f"https://finnhub.io/api/v1/calendar/earnings"
               f"?from={today}&to={today}&token={self._finnhub}")
        return self._test(
            "Finnhub Earnings Calendar", url,
            skip_reason="" if self._finnhub else "No FINNHUB_API_KEY",
            validator=lambda d: f"{len(d.get('earningsCalendar', []))} events" if "earningsCalendar" in d else "OK",
        )

    def test_finnhub_news(self) -> EndpointResult:
        today = datetime.utcnow().date().isoformat()
        url = (f"https://finnhub.io/api/v1/company-news"
               f"?symbol=SPY&from={today}&to={today}&token={self._finnhub}")
        return self._test(
            "Finnhub Company News (SPY)", url,
            skip_reason="" if self._finnhub else "No FINNHUB_API_KEY",
            validator=lambda d: f"{len(d)} articles" if isinstance(d, list) else "OK",
        )

    def test_finnhub_insider(self) -> EndpointResult:
        url = (f"https://finnhub.io/api/v1/stock/insider-transactions"
               f"?symbol=AAPL&token={self._finnhub}")
        return self._test(
            "Finnhub Insider Trades (AAPL)", url,
            skip_reason="" if self._finnhub else "No FINNHUB_API_KEY",
            validator=lambda d: f"{len(d.get('data', []))} trades" if "data" in d else "OK",
        )

    # ══════════════════════════════════════════════════════════════════════
    # CNN FEAR & GREED
    # ══════════════════════════════════════════════════════════════════════

    def test_fear_greed(self) -> EndpointResult:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        return self._test(
            "CNN Fear & Greed Index", url,
            headers={
                "Accept": "application/json",
                "Referer": "https://www.cnn.com/markets/fear-and-greed",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
            validator=lambda d: f"score={d['fear_and_greed']['score']:.0f}" if "fear_and_greed" in d else "no data",
        )

    # ══════════════════════════════════════════════════════════════════════
    # UNUSUAL WHALES ENDPOINTS
    # ══════════════════════════════════════════════════════════════════════

    def test_uw_flow(self) -> EndpointResult:
        url = "https://api.unusualwhales.com/api/option-trades/flow-alerts?limit=5"
        headers = {
            "Authorization": f"Bearer {self._uw}",
            "User-Agent": "AAC/3.6.0 UnusualWhalesClient",
            "UW-CLIENT-API-ID": "100001",
        }
        return self._test(
            "Unusual Whales Options Flow", url, headers=headers,
            skip_reason="" if self._uw else "No UNUSUAL_WHALES_API_KEY",
            validator=lambda d: f"{len(d.get('data', []))} flows" if "data" in d else "unexpected response",
        )

    def test_uw_dark_pool(self) -> EndpointResult:
        url = "https://api.unusualwhales.com/api/darkpool/recent?limit=5"
        headers = {
            "Authorization": f"Bearer {self._uw}",
            "User-Agent": "AAC/3.6.0 UnusualWhalesClient",
            "UW-CLIENT-API-ID": "100001",
        }
        return self._test(
            "Unusual Whales Dark Pool", url, headers=headers,
            skip_reason="" if self._uw else "No UNUSUAL_WHALES_API_KEY",
            validator=lambda d: f"{len(d.get('data', []))} prints" if "data" in d else "unexpected response",
        )

    def test_uw_congress(self) -> EndpointResult:
        url = "https://api.unusualwhales.com/api/congress/recent-trades?limit=5"
        headers = {
            "Authorization": f"Bearer {self._uw}",
            "User-Agent": "AAC/3.6.0 UnusualWhalesClient",
            "UW-CLIENT-API-ID": "100001",
        }
        return self._test(
            "Unusual Whales Congress Trades", url, headers=headers,
            skip_reason="" if self._uw else "No UNUSUAL_WHALES_API_KEY",
            validator=lambda d: f"{len(d.get('data', []))} trades" if "data" in d else "unexpected response",
        )

    # ══════════════════════════════════════════════════════════════════════
    # TELEGRAM
    # ══════════════════════════════════════════════════════════════════════

    def test_telegram(self) -> EndpointResult:
        if not self._telegram_token:
            r = EndpointResult(name="Telegram Bot API", url="api.telegram.org",
                               status="SKIP", detail="No TELEGRAM_BOT_TOKEN")
            self.results.append(r)
            return r

        url = f"https://api.telegram.org/bot{self._telegram_token}/getMe"
        return self._test(
            "Telegram Bot API", url,
            validator=lambda d: f"bot=@{d['result']['username']}" if d.get("ok") else "auth failed",
        )

    # ══════════════════════════════════════════════════════════════════════
    # SMTP EMAIL
    # ══════════════════════════════════════════════════════════════════════

    def test_smtp(self) -> EndpointResult:
        if not all([self._smtp_host, self._smtp_user, self._smtp_pass]):
            r = EndpointResult(name="SMTP Email (Zoho)", url=self._smtp_host or "N/A",
                               status="SKIP", detail="SMTP not configured")
            self.results.append(r)
            return r

        t0 = time.time()
        try:
            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(self._smtp_user, self._smtp_pass)
                latency = (time.time() - t0) * 1000
                r = EndpointResult(
                    name="SMTP Email (Zoho)", url=f"{self._smtp_host}:{self._smtp_port}",
                    status="OK", latency_ms=round(latency, 1),
                    detail=f"STARTTLS login OK as {self._smtp_user}",
                )
        except smtplib.SMTPAuthenticationError:
            latency = (time.time() - t0) * 1000
            r = EndpointResult(
                name="SMTP Email (Zoho)", url=f"{self._smtp_host}:{self._smtp_port}",
                status="FAIL", latency_ms=round(latency, 1),
                detail="SMTP auth failed — check SMTP_PASSWORD",
            )
        except Exception as e:
            latency = (time.time() - t0) * 1000
            r = EndpointResult(
                name="SMTP Email (Zoho)", url=f"{self._smtp_host}:{self._smtp_port}",
                status="FAIL", latency_ms=round(latency, 1),
                detail=f"SMTP error: {e}",
            )

        self.results.append(r)
        return r

    # ══════════════════════════════════════════════════════════════════════
    # RUN ALL
    # ══════════════════════════════════════════════════════════════════════

    def run_all(self) -> List[EndpointResult]:
        """Run all HTTP health checks."""
        self.results = []

        # Polygon (3 endpoints)
        self.test_polygon_snapshot()
        self.test_polygon_prev_close()
        self.test_polygon_options_chain()

        # FRED (4 endpoints)
        self.test_fred_vix()
        self.test_fred_oil()
        self.test_fred_yield_curve()
        self.test_fred_hy_spread()

        # Finnhub (3 endpoints)
        self.test_finnhub_earnings()
        self.test_finnhub_news()
        self.test_finnhub_insider()

        # CNN (1 endpoint)
        self.test_fear_greed()

        # Unusual Whales (3 endpoints)
        self.test_uw_flow()
        self.test_uw_dark_pool()
        self.test_uw_congress()

        # Messaging (2 endpoints)
        self.test_telegram()
        self.test_smtp()

        return self.results

    def print_report(self) -> str:
        """Format results as a readable report."""
        lines = [
            "",
            "═" * 70,
            "  MATRIX MAXIMIZER — HTTP HEALTH CHECK",
            f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "═" * 70,
            "",
        ]

        # API key status
        keys = self.check_keys()
        lines.append("  API KEYS:")
        for k, v in keys.items():
            icon = "✓" if v.startswith("SET") else "✗"
            lines.append(f"    {icon}  {k}: {v}")
        lines.append("")

        # Endpoint results
        ok = sum(1 for r in self.results if r.status == "OK")
        fail = sum(1 for r in self.results if r.status == "FAIL")
        warn = sum(1 for r in self.results if r.status == "WARN")
        skip = sum(1 for r in self.results if r.status == "SKIP")
        total = len(self.results)

        lines.append(f"  ENDPOINTS: {ok}/{total} OK, {fail} FAIL, {warn} WARN, {skip} SKIP")
        lines.append("─" * 70)

        status_icon = {"OK": "✓", "FAIL": "✗", "WARN": "!", "SKIP": "○"}
        for r in self.results:
            icon = status_icon.get(r.status, "?")
            lat = f" ({r.latency_ms:.0f}ms)" if r.latency_ms > 0 else ""
            lines.append(f"  {icon}  {r.name}{lat}")
            lines.append(f"     {r.detail}")
            if r.data_preview:
                lines.append(f"     DATA: {r.data_preview}")

        lines.append("")
        lines.append("─" * 70)

        if fail == 0 and skip == 0:
            lines.append("  ALL ENDPOINTS OPERATIONAL ✓")
        elif fail == 0:
            lines.append(f"  OPERATIONAL ({skip} skipped — missing keys)")
        else:
            lines.append(f"  {fail} ENDPOINT(S) FAILING — see details above")

        lines.append("═" * 70)
        lines.append("")

        report = "\n".join(lines)
        return report

    def save_report(self, report: str) -> Path:
        """Save report to data/matrix_maximizer/http_health.txt."""
        out_dir = Path("data/matrix_maximizer")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "http_health.txt"
        path.write_text(report, encoding="utf-8")
        return path


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="MATRIX MAXIMIZER HTTP Health Check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show data previews")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    checker = HTTPHealthCheck(verbose=args.verbose)
    checker.run_all()
    report = checker.print_report()
    print(report)

    path = checker.save_report(report)
    print(f"  Report saved to: {path}")

    # Exit code based on results
    fails = sum(1 for r in checker.results if r.status == "FAIL")
    sys.exit(1 if fails > 0 else 0)


if __name__ == "__main__":
    main()
