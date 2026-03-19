"""
MATRIX MAXIMIZER — Live Data Feeds
====================================
Real-time market data from multiple sources:
  - Polygon: live equity/ETF prices + options chains
  - FRED: VIX, WTI oil, HY spreads, yield curve, macro
  - Finnhub: quotes, earnings calendar, insider trades, company news
  - Fear & Greed: CNN sentiment index
  - Unusual Whales: dark pool flow, congress trades, sweeps

All methods are sync wrappers around async AAC clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Cache path ────────────────────────────────────────────────────────────
_CACHE_DIR = Path("data/matrix_maximizer/cache")


@dataclass
class LiveQuote:
    """Snapshot of a single ticker."""
    symbol: str
    price: float
    change: float = 0.0
    change_pct: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: int = 0
    timestamp: str = ""
    source: str = "polygon"


@dataclass
class MacroSnapshot:
    """All macro feeds in one shot."""
    vix: float = 22.0
    oil_wti: float = 96.5
    gold: float = 2050.0
    hy_spread_bps: float = 350.0
    ig_spread_bps: float = 100.0
    yield_10y: float = 4.2
    yield_2y: float = 4.5
    yield_curve_10_2: float = -0.3
    fed_funds: float = 5.25
    fear_greed: float = 35.0
    dollar_index: float = 104.0
    timestamp: str = ""


@dataclass
class EarningsEvent:
    """Upcoming earnings announcement."""
    symbol: str
    date: str
    eps_estimate: Optional[float] = None
    revenue_estimate: Optional[float] = None
    hour: str = ""  # "bmo" (before market open) or "amc" (after market close)


@dataclass
class FlowAlert:
    """Unusual options flow detection."""
    ticker: str
    strike: float
    expiry: str
    option_type: str  # "call" or "put"
    sentiment: str    # "bullish" or "bearish"
    premium: float
    volume: int
    open_interest: int
    source: str = "unusual_whales"


@dataclass
class DarkPoolTrade:
    """Dark pool print."""
    ticker: str
    price: float
    size: int
    notional: float
    timestamp: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# LIVE DATA FEED MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class DataFeedManager:
    """Central hub for all live data feeds.

    Aggregates data from Polygon, FRED, Finnhub, Unusual Whales, and
    Fear & Greed Index into a single interface.

    Usage:
        feeds = DataFeedManager()
        prices = feeds.get_live_prices(["SPY", "QQQ", "USO"])
        macro  = feeds.get_macro_snapshot()
        earn   = feeds.get_earnings_calendar(["SPY", "AAPL"], days=14)
        flow   = feeds.get_unusual_flow(min_premium=100000)
    """

    def __init__(self) -> None:
        self._polygon_key = os.getenv("POLYGON_API_KEY", "")
        self._finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        self._fred_key = os.getenv("FRED_API_KEY", "")
        self._uw_key = os.getenv("UNUSUAL_WHALES_API_KEY", "")
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, float] = {}

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── HTTP helper ───────────────────────────────────────────────────────

    def _http_get(self, url: str, headers: Optional[Dict[str, str]] = None,
                  timeout: int = 10) -> Optional[Dict]:
        """Safe HTTP GET returning parsed JSON or None."""
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "AAC-MatrixMaximizer/1.0")
            if headers:
                for k, v in headers.items():
                    req.add_header(k, v)
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("HTTP GET failed for %s: %s", url[:80], exc)
            return None

    def _cached(self, key: str, ttl_seconds: int = 60) -> Optional[Any]:
        """Return cached value if still fresh."""
        if key in self._cache and key in self._cache_ttl:
            if (datetime.utcnow().timestamp() - self._cache_ttl[key]) < ttl_seconds:
                return self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._cache_ttl[key] = datetime.utcnow().timestamp()

    # ═══════════════════════════════════════════════════════════════════════
    # POLYGON — Live Equity Prices
    # ═══════════════════════════════════════════════════════════════════════

    def get_live_prices(self, tickers: List[str]) -> Dict[str, LiveQuote]:
        """Fetch latest prices from Polygon snapshots.

        Falls back to previous-day close if snapshot unavailable.
        """
        result: Dict[str, LiveQuote] = {}

        if not self._polygon_key:
            logger.warning("No POLYGON_API_KEY — using defaults")
            return result

        for ticker in tickers:
            cached = self._cached(f"price_{ticker}", ttl_seconds=30)
            if cached:
                result[ticker] = cached
                continue

            url = (
                f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
                f"?apiKey={self._polygon_key}"
            )
            data = self._http_get(url)
            if data and data.get("status") == "OK" and "ticker" in data:
                snap = data["ticker"]
                day = snap.get("day", {})
                prev = snap.get("prevDay", {})
                price = day.get("c") or prev.get("c", 0)
                quote = LiveQuote(
                    symbol=ticker,
                    price=price,
                    change=snap.get("todaysChange", 0),
                    change_pct=snap.get("todaysChangePerc", 0),
                    high=day.get("h", 0),
                    low=day.get("l", 0),
                    volume=int(day.get("v", 0)),
                    timestamp=datetime.utcnow().isoformat(),
                    source="polygon",
                )
                result[ticker] = quote
                self._set_cache(f"price_{ticker}", quote)
            else:
                # Fallback: previous close from aggs
                url2 = (
                    f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
                    f"?apiKey={self._polygon_key}"
                )
                data2 = self._http_get(url2)
                if data2 and data2.get("results"):
                    bar = data2["results"][0]
                    quote = LiveQuote(
                        symbol=ticker,
                        price=bar.get("c", 0),
                        high=bar.get("h", 0),
                        low=bar.get("l", 0),
                        volume=int(bar.get("v", 0)),
                        timestamp=datetime.utcnow().isoformat(),
                        source="polygon_prev",
                    )
                    result[ticker] = quote
                    self._set_cache(f"price_{ticker}", quote)

        return result

    def get_live_vix(self) -> float:
        """Fetch VIX from Polygon or FRED."""
        cached = self._cached("vix", ttl_seconds=60)
        if cached is not None:
            return cached

        # Try Polygon first (VIXY as proxy, or VIX index)
        if self._polygon_key:
            url = (
                f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/VIXY"
                f"?apiKey={self._polygon_key}"
            )
            data = self._http_get(url)
            if data and data.get("status") == "OK" and "ticker" in data:
                price = data["ticker"].get("day", {}).get("c", 0)
                if price > 0:
                    self._set_cache("vix", price)
                    return price

        # Fallback: FRED VIXCLS
        vix = self._fred_series_latest("VIXCLS")
        if vix is not None:
            self._set_cache("vix", vix)
            return vix

        return 22.0  # Default

    def get_live_oil(self) -> float:
        """Fetch WTI crude oil price (FRED DCOILWTICO or USO proxy)."""
        cached = self._cached("oil", ttl_seconds=300)
        if cached is not None:
            return cached

        # FRED WTI
        oil = self._fred_series_latest("DCOILWTICO")
        if oil is not None:
            self._set_cache("oil", oil)
            return oil

        # Fallback: USO ETF as proxy (multiply by ~0.72 for WTI approximation)
        if self._polygon_key:
            prices = self.get_live_prices(["USO"])
            if "USO" in prices:
                wti_est = prices["USO"].price  # USO tracks WTI futures
                self._set_cache("oil", wti_est)
                return wti_est

        return 96.5  # Default

    # ═══════════════════════════════════════════════════════════════════════
    # FRED — Macro Economic Data
    # ═══════════════════════════════════════════════════════════════════════

    def _fred_series_latest(self, series_id: str) -> Optional[float]:
        """Fetch latest observation from FRED."""
        if not self._fred_key:
            return None

        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={self._fred_key}"
            f"&file_type=json&sort_order=desc&limit=5"
        )
        data = self._http_get(url)
        if data and "observations" in data:
            for obs in data["observations"]:
                val = obs.get("value", ".")
                if val != ".":
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass
        return None

    def get_macro_snapshot(self) -> MacroSnapshot:
        """Full macro data from FRED + Fear & Greed."""
        cached = self._cached("macro", ttl_seconds=600)
        if cached is not None:
            return cached

        snap = MacroSnapshot(timestamp=datetime.utcnow().isoformat())

        # FRED series
        series_map = {
            "VIXCLS": "vix",
            "DCOILWTICO": "oil_wti",
            "GOLDAMGBD228NLBM": "gold",
            "BAMLH0A0HYM2": "hy_spread_bps",
            "T10Y2Y": "yield_curve_10_2",
            "DGS10": "yield_10y",
            "DGS2": "yield_2y",
            "FEDFUNDS": "fed_funds",
            "DTWEXBGS": "dollar_index",
        }

        for series_id, attr in series_map.items():
            val = self._fred_series_latest(series_id)
            if val is not None:
                # HY spread is in percentage, convert to bps
                if attr == "hy_spread_bps":
                    val *= 100
                setattr(snap, attr, val)

        # Fear & Greed Index
        fg = self._fetch_fear_greed()
        if fg is not None:
            snap.fear_greed = fg

        self._set_cache("macro", snap)
        return snap

    def _fetch_fear_greed(self) -> Optional[float]:
        """CNN Fear & Greed Index."""
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        data = self._http_get(url, headers={"Accept": "application/json"})
        if data and "fear_and_greed" in data:
            return data["fear_and_greed"].get("score")
        return None

    def get_yield_curve(self) -> Dict[str, float]:
        """Full yield curve from FRED."""
        maturities = {
            "DGS1MO": "1m", "DGS3MO": "3m", "DGS6MO": "6m",
            "DGS1": "1y", "DGS2": "2y", "DGS5": "5y",
            "DGS10": "10y", "DGS20": "20y", "DGS30": "30y",
        }
        curve: Dict[str, float] = {}
        for series_id, label in maturities.items():
            val = self._fred_series_latest(series_id)
            if val is not None:
                curve[label] = val
        return curve

    # ═══════════════════════════════════════════════════════════════════════
    # FINNHUB — Earnings, News, Insider
    # ═══════════════════════════════════════════════════════════════════════

    def get_earnings_calendar(self, tickers: Optional[List[str]] = None,
                              days_ahead: int = 14) -> List[EarningsEvent]:
        """Upcoming earnings from Finnhub."""
        if not self._finnhub_key:
            return []

        today = datetime.utcnow().date()
        end = today + timedelta(days=days_ahead)
        url = (
            f"https://finnhub.io/api/v1/calendar/earnings"
            f"?from={today.isoformat()}&to={end.isoformat()}"
            f"&token={self._finnhub_key}"
        )
        data = self._http_get(url)
        events: List[EarningsEvent] = []
        if data and "earningsCalendar" in data:
            for e in data["earningsCalendar"]:
                sym = e.get("symbol", "")
                if tickers and sym not in tickers:
                    continue
                events.append(EarningsEvent(
                    symbol=sym,
                    date=e.get("date", ""),
                    eps_estimate=e.get("epsEstimate"),
                    revenue_estimate=e.get("revenueEstimate"),
                    hour=e.get("hour", ""),
                ))
        return events

    def get_company_news(self, ticker: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """Recent news for a ticker from Finnhub."""
        if not self._finnhub_key:
            return []

        today = datetime.utcnow().date()
        start = today - timedelta(days=days_back)
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={start.isoformat()}&to={today.isoformat()}"
            f"&token={self._finnhub_key}"
        )
        data = self._http_get(url)
        if isinstance(data, list):
            return [
                {
                    "headline": a.get("headline", ""),
                    "source": a.get("source", ""),
                    "summary": a.get("summary", "")[:200],
                    "datetime": a.get("datetime", 0),
                    "sentiment": a.get("sentiment", "neutral"),
                }
                for a in data[:20]
            ]
        return []

    def get_insider_trades(self, ticker: str) -> List[Dict[str, Any]]:
        """Recent insider transactions from Finnhub."""
        if not self._finnhub_key:
            return []

        url = (
            f"https://finnhub.io/api/v1/stock/insider-transactions"
            f"?symbol={ticker}&token={self._finnhub_key}"
        )
        data = self._http_get(url)
        trades: List[Dict[str, Any]] = []
        if data and "data" in data:
            for t in data["data"][:20]:
                trades.append({
                    "name": t.get("name", ""),
                    "share": t.get("share", 0),
                    "change": t.get("change", 0),
                    "transaction_type": t.get("transactionType", ""),
                    "filing_date": t.get("filingDate", ""),
                })
        return trades

    # ═══════════════════════════════════════════════════════════════════════
    # UNUSUAL WHALES — Flow & Dark Pool
    # ═══════════════════════════════════════════════════════════════════════

    def get_unusual_flow(self, ticker: Optional[str] = None,
                         min_premium: float = 100_000,
                         limit: int = 50) -> List[FlowAlert]:
        """Fetch unusual options flow from Unusual Whales API."""
        if not self._uw_key:
            return []

        cached = self._cached(f"flow_{ticker or 'all'}", ttl_seconds=120)
        if cached is not None:
            return cached

        url = "https://api.unusualwhales.com/api/stock/flow"
        if ticker:
            url += f"?ticker={ticker}&limit={limit}"
        else:
            url += f"?limit={limit}"

        headers = {
            "Authorization": f"Bearer {self._uw_key}",
            "User-Agent": "AAC-MatrixMaximizer/1.0",
        }
        data = self._http_get(url, headers=headers)
        alerts: List[FlowAlert] = []
        if data and "data" in data:
            for f in data["data"]:
                premium = float(f.get("premium", 0))
                if premium < min_premium:
                    continue
                alerts.append(FlowAlert(
                    ticker=f.get("ticker", ""),
                    strike=float(f.get("strike", 0)),
                    expiry=f.get("expiry", ""),
                    option_type=f.get("put_call", "").lower(),
                    sentiment=f.get("sentiment", "neutral").lower(),
                    premium=premium,
                    volume=int(f.get("volume", 0)),
                    open_interest=int(f.get("open_interest", 0)),
                ))

        self._set_cache(f"flow_{ticker or 'all'}", alerts)
        return alerts

    def get_dark_pool(self, ticker: Optional[str] = None,
                      limit: int = 50) -> List[DarkPoolTrade]:
        """Fetch dark pool prints from Unusual Whales."""
        if not self._uw_key:
            return []

        url = "https://api.unusualwhales.com/api/darkpool"
        if ticker:
            url += f"?ticker={ticker}&limit={limit}"
        else:
            url += f"?limit={limit}"

        headers = {
            "Authorization": f"Bearer {self._uw_key}",
            "User-Agent": "AAC-MatrixMaximizer/1.0",
        }
        data = self._http_get(url, headers=headers)
        trades: List[DarkPoolTrade] = []
        if data and "data" in data:
            for d in data["data"]:
                trades.append(DarkPoolTrade(
                    ticker=d.get("ticker", ""),
                    price=float(d.get("price", 0)),
                    size=int(d.get("size", 0)),
                    notional=float(d.get("notional", 0)),
                    timestamp=d.get("tracking_timestamp", ""),
                ))
        return trades

    def get_congress_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch congressional trading activity from Unusual Whales."""
        if not self._uw_key:
            return []

        url = f"https://api.unusualwhales.com/api/congress/trading?limit={limit}"
        headers = {
            "Authorization": f"Bearer {self._uw_key}",
            "User-Agent": "AAC-MatrixMaximizer/1.0",
        }
        data = self._http_get(url, headers=headers)
        if data and "data" in data:
            return [
                {
                    "politician": t.get("politician", ""),
                    "ticker": t.get("ticker", ""),
                    "type": t.get("type", ""),
                    "amount": t.get("amount", ""),
                    "date": t.get("transaction_date", ""),
                }
                for t in data["data"][:limit]
            ]
        return []

    # ═══════════════════════════════════════════════════════════════════════
    # AGGREGATED PRICE + OVERRIDE RESOLUTION
    # ═══════════════════════════════════════════════════════════════════════

    def resolve_prices(self, config_tickers: List[str],
                       overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Build complete price dict: live data → overrides → defaults.

        Priority:
            1. CLI/manual overrides
            2. Live Polygon data
            3. Hardcoded defaults from core.py
        """
        from strategies.matrix_maximizer.core import DEFAULT_PRICES, Asset

        # Start with defaults
        prices: Dict[str, float] = {a.value: p for a, p in DEFAULT_PRICES.items()}

        # Layer live data
        live = self.get_live_prices(config_tickers)
        for ticker, quote in live.items():
            if quote.price > 0:
                prices[ticker] = quote.price

        # Layer overrides
        if overrides:
            prices.update(overrides)

        # Add VIX and oil as special keys
        prices["vix"] = self.get_live_vix()
        prices["oil"] = self.get_live_oil()

        return prices

    def get_tickers_near_earnings(self, tickers: List[str],
                                   days: int = 5) -> List[str]:
        """Return tickers that have earnings within N days.

        Used by scanner to avoid selling puts into earnings or to
        target earnings for premium.
        """
        events = self.get_earnings_calendar(tickers=tickers, days_ahead=days)
        return list({e.symbol for e in events})

    # ═══════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════

    def save_snapshot(self, macro: MacroSnapshot, prices: Dict[str, float]) -> None:
        """Persist latest data snapshot for offline analysis."""
        out = {
            "timestamp": datetime.utcnow().isoformat(),
            "macro": {
                "vix": macro.vix, "oil_wti": macro.oil_wti,
                "hy_spread_bps": macro.hy_spread_bps,
                "yield_curve_10_2": macro.yield_curve_10_2,
                "fear_greed": macro.fear_greed,
                "gold": macro.gold,
            },
            "prices": prices,
        }
        path = _CACHE_DIR / "latest_snapshot.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
