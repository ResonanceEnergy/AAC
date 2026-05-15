from __future__ import annotations

"""DFV data helpers — sync wrappers around the async clients.

Routines and CLI commands run synchronously, but every market-data client
in this repo is async. These helpers are the bridge: each one runs the
async call via asyncio.run(), swallows errors into a structured dict so
the daemon never dies, and returns plain JSON-serialisable data.

Hard rule: NEVER use Barchart / web scraping / yahoo HTML. Only the
internal clients (CoinGecko, FRED, Finnhub, etc).
"""

import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import structlog
from dotenv import load_dotenv

# Load .env so FRED_API_KEY / FINNHUB_API_KEY / COINGECKO_API_KEY are visible
# to the underlying clients (they read os.environ at __init__).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_log = structlog.get_logger(__name__)

# ── tickers DFV cares about by slot ────────────────────────────────────
ASIA_ADRS = ("BABA", "JD", "PDD", "TSM", "BIDU", "SE", "TM", "NIO")
US_FUTURES_PROXIES = ("SPY", "QQQ", "IWM", "DIA")  # ETFs as overnight ES/NQ proxy
VIX_FRED = "VIXCLS"            # FRED daily VIX close
DXY_FRED = "DTWEXBGS"          # Broad trade-weighted dollar index (daily)
US10Y_FRED = "DGS10"           # 10y constant-maturity treasury


def _run(coro: Any) -> Any:
    """Run an async coroutine from sync code, swallowing the result on error."""
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        # If a loop is already running (shouldn't be in routines, but defensive),
        # punt and return None — the caller wraps everything in {ok: bool}.
        _log.warning("dfv.data._run.event_loop_conflict", error=str(e))
        return None
    except Exception as e:  # noqa: BLE001 — must not kill caller routine
        _log.warning("dfv.data._run.error", error=str(e))
        return None


# ── crypto ─────────────────────────────────────────────────────────────
def crypto_snapshot(coin_ids: Iterable[str] = ("bitcoin", "ethereum")) -> dict[str, Any]:
    """Return BTC/ETH (or any CoinGecko coin_ids) prices + 24h change in USD."""
    from shared.data_sources import CoinGeckoClient

    async def _go() -> list[Any]:
        client = CoinGeckoClient()
        try:
            await client.connect()
            return await client.get_prices_batch(list(coin_ids), vs_currency="usd")
        finally:
            await client.disconnect()

    ticks = _run(_go()) or []
    out: dict[str, Any] = {"ok": bool(ticks), "ts": _utc_now(), "prices": {}}
    for t in ticks:
        out["prices"][t.symbol] = {
            "price": t.price,
            "change_24h_pct": t.change_24h,
            "volume_24h": t.volume_24h,
        }
    return out


# ── macro (FRED) ───────────────────────────────────────────────────────
def macro_snapshot(series: Iterable[str] = (DXY_FRED, US10Y_FRED, VIX_FRED)) -> dict[str, Any]:
    """Latest values for FRED series. DXY / 10Y / VIX by default."""
    from integrations.fred_client import FredClient

    async def _go() -> dict[str, Any]:
        client = FredClient()
        try:
            results: dict[str, Any] = {}
            for sid in series:
                obs = await client.get_latest_value(sid)
                results[sid] = (
                    {"date": obs.date, "value": obs.value} if obs else None
                )
            return results
        finally:
            await client.close()

    data = _run(_go()) or {}
    return {"ok": bool(data), "ts": _utc_now(), "series": data}


# ── equity quotes (Finnhub) ────────────────────────────────────────────
def quotes(symbols: Iterable[str]) -> dict[str, Any]:
    """Real-time quotes for a list of symbols. Returns {sym: {current, change_pct, ...}}"""
    syms = [s.upper() for s in symbols if s]
    if not syms:
        return {"ok": True, "ts": _utc_now(), "quotes": {}}

    from integrations.finnhub_client import FinnhubClient

    async def _go() -> dict[str, Any]:
        client = FinnhubClient()
        try:
            results: dict[str, Any] = {}
            for sym in syms:
                q = await client.get_quote(sym)
                if q:
                    results[sym] = {
                        "current": q["current"],
                        "change_pct": q["change_pct"],
                        "high": q["high"],
                        "low": q["low"],
                        "prev_close": q["prev_close"],
                    }
            return results
        finally:
            await client.close()

    data = _run(_go()) or {}
    return {"ok": bool(data), "ts": _utc_now(), "quotes": data}


# ── news ───────────────────────────────────────────────────────────────
def market_news(category: str = "general", limit: int = 10) -> dict[str, Any]:
    """Top market news headlines."""
    from integrations.finnhub_client import FinnhubClient

    async def _go() -> list[Any]:
        client = FinnhubClient()
        try:
            return await client.get_news(category=category)
        finally:
            await client.close()

    items = _run(_go()) or []
    headlines = [
        {
            "headline": n.headline,
            "source": n.source,
            "url": n.url,
            "category": n.category,
            "datetime": n.datetime.isoformat() if hasattr(n.datetime, "isoformat") else str(n.datetime),
        }
        for n in items[:limit]
    ]
    return {"ok": bool(headlines), "ts": _utc_now(), "category": category, "headlines": headlines}


def earnings_window(days_ahead: int = 1) -> dict[str, Any]:
    """Earnings calendar for the next N days."""
    from integrations.finnhub_client import FinnhubClient

    async def _go() -> list[Any]:
        client = FinnhubClient()
        try:
            return await client.get_earnings_calendar(
                from_date=date.today().isoformat(),
                to_date=(date.today() + timedelta(days=days_ahead)).isoformat(),
            )
        finally:
            await client.close()

    items = _run(_go()) or []
    rows = [
        {
            "symbol": e.symbol,
            "date": e.date,
            "eps_est": e.eps_estimate,
            "rev_est": e.revenue_estimate,
            "quarter": e.quarter,
        }
        for e in items
    ]
    return {"ok": True, "ts": _utc_now(), "days_ahead": days_ahead, "earnings": rows}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
