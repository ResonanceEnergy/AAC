from __future__ import annotations

"""DFV data helpers — IBKR / mission_control only.

History: this module briefly wrapped CoinGecko, FRED and Finnhub for the
Sprint 1 cadence routines. Those three keys are empty in this repo's
.env (and the surrounding integrations clients have known issues), so
the dependency was orphaned on 2026-05-15. DFV now relies exclusively
on what mission_control already aggregates from IBKR / Moomoo / war
room. If you need an external feed, wire it through mission_control or
shared/data_sources first — never call an external API directly from a
DFV routine.
"""

from datetime import datetime, timezone
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

# Universe constants kept as documentation; routines reference them in
# headlines but no longer fan out external quote calls.
ASIA_ADRS = ("BABA", "JD", "PDD", "TSM", "BIDU", "SE", "TM", "NIO")
US_FUTURES_PROXIES = ("SPY", "QQQ", "IWM", "DIA")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty(kind: str) -> dict[str, Any]:
    """Sentinel payload returned when an upstream feed is unavailable."""
    return {"ok": False, "ts": _utc_now(), "kind": kind, "note": "feed orphaned"}


# ── yfinance fallbacks (free, no key) ───────────────────────────────────────
# Used when FRED / Finnhub / CoinGecko keys are absent (default repo state).
# yfinance is already a primary dependency for options chains, so this adds
# no new transitive deps.

_FRED_TO_YF = {
    "VIXCLS": "^VIX",       # CBOE VIX
    "DGS10": "^TNX",        # 10-year Treasury yield (yfinance returns % * 1, e.g. 4.595)
    "DTWEXBGS": "DX-Y.NYB", # DXY (broader trade-weighted dollar proxy)
    "SP500": "^GSPC",
    "DJIA": "^DJI",
    "NASDAQ": "^IXIC",
}


def yf_macro_snapshot(series: tuple[str, ...] = ("VIXCLS", "DGS10", "DTWEXBGS")) -> dict[str, Any]:
    """Snapshot of macro proxies via yfinance. Replaces FRED when key absent.

    Keys in the output match the requested FRED series IDs so downstream
    consumers (briefs, war room) don't have to know about the swap.
    """
    out: dict[str, Any] = {"ok": False, "ts": _utc_now(), "series": {}, "source": "yfinance"}
    try:
        import yfinance as yf
    except ImportError as exc:
        out["error"] = f"yfinance not installed: {exc}"
        return out

    for fred_id in series:
        ticker = _FRED_TO_YF.get(fred_id)
        if not ticker:
            out["series"][fred_id] = None
            continue
        try:
            fi = yf.Ticker(ticker).fast_info
            val = float(fi.last_price) if fi.last_price is not None else None
            out["series"][fred_id] = val
        except (ValueError, KeyError, AttributeError, ConnectionError, OSError) as exc:
            out["series"][fred_id] = None
            out.setdefault("errors", {})[fred_id] = str(exc)

    out["ok"] = any(v is not None for v in out["series"].values())
    return out


def yf_crypto_snapshot(coins: tuple[str, ...] = ("BTC-USD", "ETH-USD")) -> dict[str, Any]:
    """Crypto spot via yfinance. Replaces CoinGecko when key/feed absent."""
    out: dict[str, Any] = {"ok": False, "ts": _utc_now(), "prices": {}, "source": "yfinance"}
    try:
        import yfinance as yf
    except ImportError as exc:
        out["error"] = f"yfinance not installed: {exc}"
        return out

    for coin in coins:
        try:
            t = yf.Ticker(coin)
            fi = t.fast_info
            last = float(fi.last_price) if fi.last_price is not None else None
            prev = float(fi.previous_close) if fi.previous_close is not None else None
            chg = ((last - prev) / prev * 100.0) if (last and prev) else None
            label = coin.replace("-USD", "/USD").upper()
            out["prices"][label] = {"price": last, "change_24h_pct": chg}
        except (ValueError, KeyError, AttributeError, ConnectionError, OSError) as exc:
            out.setdefault("errors", {})[coin] = str(exc)

    out["ok"] = bool(out["prices"])
    return out


def yf_news(symbol: str, limit: int = 10) -> list[dict[str, Any]]:
    """Recent news for a single ticker via yfinance. Replaces Finnhub.

    yfinance v0.2.4x returns either flat dicts or wrapped {content: {...}}
    depending on version; this normalises both.
    """
    try:
        import yfinance as yf
    except ImportError:
        return []

    try:
        raw = yf.Ticker(symbol).news or []
    except (ValueError, KeyError, AttributeError, ConnectionError, OSError):
        return []

    out: list[dict[str, Any]] = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        # Newer yfinance wraps under "content"
        c = item.get("content") if isinstance(item.get("content"), dict) else item
        title = c.get("title") or c.get("headline") or ""
        url = (
            c.get("canonicalUrl", {}).get("url")
            if isinstance(c.get("canonicalUrl"), dict)
            else c.get("clickThroughUrl", {}).get("url")
            if isinstance(c.get("clickThroughUrl"), dict)
            else c.get("link") or c.get("url") or ""
        )
        provider = (
            c.get("provider", {}).get("displayName")
            if isinstance(c.get("provider"), dict)
            else c.get("publisher") or c.get("source") or ""
        )
        pub = c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime") or ""
        out.append({
            "datetime": str(pub),
            "headline": title,
            "source": provider,
            "url": url,
        })
    return out
