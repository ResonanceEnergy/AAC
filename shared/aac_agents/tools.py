"""Tool registry exposed to the agents.

Each tool is a plain function that returns a JSON-serializable result. The
runtime translates these into Ollama tool-call schemas and dispatches calls.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any, Callable

import structlog

_log = structlog.get_logger().bind(component="aac_agents.tools")


# Cap how much tool output we feed back to the model — prevents context blow-up.
# Surfaces a truncation marker so the LLM knows the result was cut.
_TOOL_RESULT_CAP = 12000


# ── RAG tools ────────────────────────────────────────────────────────────────


def rag_search(query: str, k: int = 6, kind: str | None = None) -> dict[str, Any]:
    """Search the AAC code/doc index. Returns top-k chunks with file paths.

    Args:
        query: natural language search query
        k: number of chunks to return (default 6, max 20)
        kind: optional filter — "code", "doc", or "config"
    """
    from shared.aac_rag import query as rag_query  # noqa: PLC0415

    k = max(1, min(int(k), 20))
    hits = rag_query(query, k=k, kind=kind)
    return {
        "count": len(hits),
        "results": [
            {
                "path": h["path"],
                "kind": h["kind"],
                "score": round(float(h["score"]), 4),
                "preview": h["text"][:600],
            }
            for h in hits
        ],
    }


def rag_ask(question: str, k: int = 8) -> dict[str, Any]:
    """Run a full RAG ask: retrieve + LLM answer with citations.

    Use this for synthesized answers when raw chunks aren't enough. More
    expensive than rag_search.
    """
    from shared.aac_rag import ask  # noqa: PLC0415

    res = ask(question, k=int(k))
    return {
        "answer": res["answer"],
        "source_paths": [s["path"] for s in res.get("sources", [])],
    }


# ── Calendar tools ───────────────────────────────────────────────────────────


def calendar_upcoming(days: int = 14, watchlist_only: bool = False) -> dict[str, Any]:
    """Upcoming calendar events (earnings, FOMC, CPI, OPEX, etc.).

    Args:
        days: lookahead window in days (default 14, max 180)
        watchlist_only: if True, filter to events touching watchlist symbols
                        (macro events are always included)
    """
    from shared.aac_calendar import upcoming  # noqa: PLC0415

    days = max(1, min(int(days), 180))
    events = upcoming(days=days, watchlist_only=bool(watchlist_only))
    return {
        "count": len(events),
        "window_days": days,
        "events": [_compact_event(e) for e in events],
    }


def calendar_by_symbol(symbol: str, days: int = 30) -> dict[str, Any]:
    """Calendar events touching a specific ticker (plus macro events)."""
    from shared.aac_calendar import by_symbol  # noqa: PLC0415

    days = max(1, min(int(days), 365))
    events = by_symbol(symbol, days=days)
    return {
        "symbol": symbol.upper(),
        "count": len(events),
        "events": [_compact_event(e) for e in events],
    }


def calendar_by_kind(kind: str, days: int = 30) -> dict[str, Any]:
    """Calendar events of a specific kind: earnings | fed | economic | options | policy | ipo."""
    from shared.aac_calendar import by_kind  # noqa: PLC0415

    days = max(1, min(int(days), 365))
    events = by_kind(kind, days=days)
    return {
        "kind": kind,
        "count": len(events),
        "events": [_compact_event(e) for e in events],
    }


def _compact_event(e: Any) -> dict[str, Any]:
    return {
        "date": e.date.isoformat(),
        "days_away": e.days_away,
        "title": e.title,
        "kind": e.kind,
        "symbols": e.symbols,
        "importance": e.importance,
        "notes": e.notes[:240],
    }


# ── Watchlist tools ──────────────────────────────────────────────────────────


def get_watchlist() -> dict[str, Any]:
    """Return the AAC ticker universe (vol_premium + war_room regimes)."""
    from shared.watchlist import (  # noqa: PLC0415
        get_vol_premium_tickers,
        get_war_room_rules,
    )

    out: dict[str, Any] = {"vol_premium": get_vol_premium_tickers(), "war_room": {}}
    for regime in ("CRISIS", "ELEVATED", "WATCH", "CALM"):
        rules = get_war_room_rules(regime)
        out["war_room"][regime] = [
            {"symbol": str(r[0]), "direction": str(r[1]), "asset_class": str(r[2]),
             "size": float(r[3])}
            for r in rules if r and len(r) >= 4
        ]
    return out


# ── Portfolio / P&L tools (Phase 4 fix — Gap 1) ──────────────────────────────


def get_positions() -> dict[str, Any]:
    """Return current live positions from IBKR (paper or live based on env).

    Connects to IBKR via PositionTracker, refreshes once, returns flat list.
    Returns `{count: 0, positions: [], error: ...}` on any failure.
    """
    import asyncio  # noqa: PLC0415

    try:
        from TradingExecution.position_tracker import PositionTracker  # noqa: PLC0415

        tracker = PositionTracker()

        async def _go() -> list[Any]:
            await tracker.connect()
            try:
                return await tracker.refresh()
            finally:
                await tracker.disconnect()

        positions = asyncio.run(_go())
        return {
            "count": len(positions),
            "total_exposure_usd": round(tracker.total_exposure(), 2),
            "total_unrealized_pnl": round(tracker.total_unrealized_pnl(), 2),
            "paper": tracker.paper,
            "positions": [p.to_dict() for p in positions],
        }
    except Exception as exc:
        _log.warning("get_positions_failed", error=str(exc))
        return {"count": 0, "positions": [], "error": str(exc)}


def get_pnl_today() -> dict[str, Any]:
    """Return today's P&L snapshot from CentralAccounting."""
    try:
        from CentralAccounting.pnl_tracker import PnLTracker  # noqa: PLC0415

        tracker = PnLTracker()
        try:
            report = tracker.today_report()
            delta = tracker.pnl_delta(days=2)
            history = tracker.historical_summary(days=7)
            return {
                "date": report.get("date"),
                "daily_pnl": report.get("daily_pnl"),
                "position_count": len(report.get("positions") or []),
                "trade_count": len(report.get("today_trades") or []),
                "pnl_delta_2d": delta,
                "history_7d": history,
            }
        finally:
            tracker.close()
    except Exception as exc:
        _log.warning("get_pnl_today_failed", error=str(exc))
        return {"error": str(exc)}


def get_account_value() -> dict[str, Any]:
    """Return live account equity in USD with source attribution."""
    try:
        from shared.account_value_feed import AccountValueFeed  # noqa: PLC0415

        feed = AccountValueFeed()
        return {"value_usd": feed.get(), "source": feed.get_source()}
    except Exception as exc:
        _log.warning("get_account_value_failed", error=str(exc))
        return {"error": str(exc)}


# ── News tool (Gap 2) ────────────────────────────────────────────────────────


def get_news(symbol: str | None = None, days: int = 3, max_items: int = 10) -> dict[str, Any]:
    """Fetch recent financial news from Finnhub.

    Args:
        symbol: ticker for company-specific news; None for general market news
        days: lookback window in days (1-30)
        max_items: cap on returned articles (1-25)
    """
    import asyncio  # noqa: PLC0415

    days = max(1, min(int(days), 30))
    max_items = max(1, min(int(max_items), 25))

    try:
        from integrations.finnhub_client import FinnhubClient  # noqa: PLC0415

        async def _go() -> list[Any]:
            async with FinnhubClient() as client:
                if symbol:
                    from_date = (date.today() - timedelta(days=days)).isoformat()
                    to_date = date.today().isoformat()
                    return await client.get_company_news(
                        symbol=symbol.upper(), from_date=from_date, to_date=to_date,
                    )
                return await client.get_news(category="general")

        articles = asyncio.run(_go())
    except Exception as exc:
        _log.warning("get_news_failed", error=str(exc))
        return {"count": 0, "articles": [], "error": str(exc)}

    items = []
    for a in articles[:max_items]:
        items.append({
            "headline": getattr(a, "headline", ""),
            "source": getattr(a, "source", ""),
            "datetime": str(getattr(a, "datetime", "")),
            "summary": (getattr(a, "summary", "") or "")[:300],
            "url": getattr(a, "url", ""),
        })
    return {"symbol": symbol, "count": len(items), "articles": items}


# ── Option chain tool (Gap 3) ────────────────────────────────────────────────


def get_option_chain(
    symbol: str,
    expiry: str | None = None,
    right: str = "P",
    n_strikes: int = 10,
) -> dict[str, Any]:
    """Fetch an option chain via yfinance.

    Args:
        symbol: underlying ticker
        expiry: YYYY-MM-DD; if None, uses the nearest expiration
        right: "P" puts or "C" calls (default puts — AAC's bread and butter)
        n_strikes: number of strikes around the money to return (1-30)
    """
    n_strikes = max(1, min(int(n_strikes), 30))
    right = right.upper()
    if right not in ("P", "C"):
        return {"error": f"right must be 'P' or 'C', got {right!r}"}

    try:
        import yfinance as yf  # noqa: PLC0415

        ticker = yf.Ticker(symbol.upper())
        expirations = list(ticker.options or [])
        if not expirations:
            return {"error": f"no options listed for {symbol}"}
        chosen = expiry if expiry in expirations else expirations[0]

        chain = ticker.option_chain(chosen)
        df = chain.puts if right == "P" else chain.calls

        # Spot to center the strike window
        try:
            spot = float(ticker.fast_info.get("last_price") or ticker.fast_info.get("lastPrice") or 0)
        except Exception:
            spot = 0.0

        if spot > 0:
            df = df.assign(_dist=(df["strike"] - spot).abs()).sort_values("_dist").head(n_strikes)
        else:
            df = df.head(n_strikes)

        rows = []
        for _, r in df.iterrows():
            rows.append({
                "strike": float(r.get("strike", 0)),
                "lastPrice": float(r.get("lastPrice", 0) or 0),
                "bid": float(r.get("bid", 0) or 0),
                "ask": float(r.get("ask", 0) or 0),
                "volume": int(r.get("volume", 0) or 0),
                "openInterest": int(r.get("openInterest", 0) or 0),
                "impliedVolatility": round(float(r.get("impliedVolatility", 0) or 0), 4),
                "inTheMoney": bool(r.get("inTheMoney", False)),
            })
        rows.sort(key=lambda x: x["strike"])
        return {
            "symbol": symbol.upper(),
            "expiry": chosen,
            "right": right,
            "spot": spot,
            "available_expiries": expirations[:12],
            "contracts": rows,
        }
    except Exception as exc:
        _log.warning("get_option_chain_failed", symbol=symbol, error=str(exc))
        return {"error": str(exc)}


# ── Tool registry + JSON schemas (Ollama tool-call format) ───────────────────


TOOLS: dict[str, Callable[..., Any]] = {
    "rag_search": rag_search,
    "rag_ask": rag_ask,
    "calendar_upcoming": calendar_upcoming,
    "calendar_by_symbol": calendar_by_symbol,
    "calendar_by_kind": calendar_by_kind,
    "get_watchlist": get_watchlist,
    "get_positions": get_positions,
    "get_pnl_today": get_pnl_today,
    "get_account_value": get_account_value,
    "get_news": get_news,
    "get_option_chain": get_option_chain,
}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the AAC codebase/documentation index. Returns top-k chunks "
                "with file paths and previews. Use this to find where things are "
                "implemented or documented."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query"},
                    "k": {"type": "integer", "description": "Number of chunks (1-20)", "default": 6},
                    "kind": {
                        "type": "string",
                        "description": "Optional filter: code, doc, or config",
                        "enum": ["code", "doc", "config"],
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_ask",
            "description": (
                "Run a full RAG ask: retrieves context AND has the local LLM "
                "synthesize an answer with citations. More expensive than rag_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "k": {"type": "integer", "default": 8},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_upcoming",
            "description": (
                "Get upcoming financial calendar events (earnings, FOMC, CPI, "
                "NFP, OPEX, etc.) sorted by date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Lookahead window in days", "default": 14},
                    "watchlist_only": {
                        "type": "boolean",
                        "description": "Filter to events touching watchlist symbols",
                        "default": False,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_by_symbol",
            "description": "Get calendar events for a specific ticker (plus macro events).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "days": {"type": "integer", "default": 30},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_by_kind",
            "description": "Get calendar events of a specific kind.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["earnings", "fed", "economic", "options", "policy", "ipo", "other"],
                    },
                    "days": {"type": "integer", "default": 30},
                },
                "required": ["kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_watchlist",
            "description": (
                "Return the AAC ticker universe: vol_premium tickers and war_room "
                "regime allocations (CRISIS/ELEVATED/WATCH/CALM)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_positions",
            "description": (
                "Return current live positions from IBKR with quantity, avg cost, "
                "market value, and unrealized P&L for each. Use to ground answers "
                "in the actual portfolio rather than guessing."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pnl_today",
            "description": (
                "Today's P&L snapshot from CentralAccounting: daily_pnl row, "
                "position/trade counts, 2-day delta, and 7-day history."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_account_value",
            "description": (
                "Live account equity in USD with source attribution "
                "(ibkr | env | default)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": (
                "Recent financial news from Finnhub. Pass a symbol for "
                "company-specific news, or omit it for general market news."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker (optional)"},
                    "days": {"type": "integer", "default": 3},
                    "max_items": {"type": "integer", "default": 10},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_option_chain",
            "description": (
                "Option chain via yfinance. Returns N strikes around the money "
                "with bid/ask/IV/volume/OI. Default is puts (right=P)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "expiry": {"type": "string", "description": "YYYY-MM-DD; nearest if omitted"},
                    "right": {"type": "string", "enum": ["P", "C"], "default": "P"},
                    "n_strikes": {"type": "integer", "default": 10},
                },
                "required": ["symbol"],
            },
        },
    },
]


def dispatch(name: str, arguments: dict[str, Any] | str | None) -> str:
    """Invoke a tool by name with a JSON-serializable arg dict. Returns a JSON string."""
    if name not in TOOLS:
        return json.dumps({"error": f"unknown tool: {name}"})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return json.dumps({"error": f"invalid JSON arguments for {name}"})
    arguments = arguments or {}
    try:
        result = TOOLS[name](**arguments)
        payload = json.dumps(result, default=str)
        if len(payload) > _TOOL_RESULT_CAP:
            # Surface truncation explicitly so the model knows it's incomplete
            truncated = payload[:_TOOL_RESULT_CAP]
            return json.dumps({
                "_truncated": True,
                "_original_bytes": len(payload),
                "_cap_bytes": _TOOL_RESULT_CAP,
                "_hint": "Result was truncated. Re-call the tool with a smaller window (lower k, fewer days, fewer strikes) to see full data.",
                "data_preview": truncated,
            })
        return payload
    except TypeError as e:
        return json.dumps({"error": f"bad arguments for {name}: {e}"})
    except Exception as e:
        _log.warning("tool_failed", tool=name, error=str(e))
        return json.dumps({"error": f"{name} raised {type(e).__name__}: {e}"})
