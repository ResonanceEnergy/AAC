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


# ── Pillar collector tools (slim summaries for agent consumption) ────────────


def get_call_options_pillar() -> dict[str, Any]:
    """Pillar A summary: vol-premium readings, rich/cheap IV counts, covered-call
    candidates, own call positions. Strips raw chain data — agents only see
    decision-grade fields.
    """
    try:
        from monitoring import dashboard_pillars as dp  # noqa: PLC0415

        payload = dp.collect_call_options(own_call_positions=None)
    except Exception as exc:
        _log.warning("get_call_options_pillar_failed", error=str(exc))
        return {"error": str(exc)}

    readings = payload.get("vol_premium_readings") or []
    rich = sorted(
        [r for r in readings if isinstance(r.get("iv_hv_ratio"), (int, float)) and r["iv_hv_ratio"] >= 1.20],
        key=lambda r: r["iv_hv_ratio"], reverse=True,
    )[:8]
    cheap = sorted(
        [r for r in readings if isinstance(r.get("iv_hv_ratio"), (int, float)) and r["iv_hv_ratio"] <= 0.85],
        key=lambda r: r["iv_hv_ratio"],
    )[:8]
    return {
        "universe_size": payload.get("universe_size", 0),
        "rich_premium_count": payload.get("rich_premium_count", 0),
        "cheap_premium_count": payload.get("cheap_premium_count", 0),
        "rich_top": [{"ticker": r.get("ticker"), "iv_hv": round(r["iv_hv_ratio"], 2),
                      "iv": round(r.get("iv", 0), 3), "hv": round(r.get("hv", 0), 3),
                      "source": r.get("source", "?")} for r in rich],
        "cheap_top": [{"ticker": r.get("ticker"), "iv_hv": round(r["iv_hv_ratio"], 2),
                       "iv": round(r.get("iv", 0), 3), "hv": round(r.get("hv", 0), 3),
                       "source": r.get("source", "?")} for r in cheap],
        "covered_call_candidates": (payload.get("covered_call_candidates") or [])[:10],
        "own_call_summary": payload.get("own_call_summary") or {},
        "errors": payload.get("errors") or [],
    }


def get_index_flow_pillar() -> dict[str, Any]:
    """Pillar B summary: market tone, P/C ratio, net premium, dark pool,
    breadth (TICK/TRIN/AD-DC), top flow tickers."""
    try:
        from monitoring import dashboard_pillars as dp  # noqa: PLC0415

        payload = dp.collect_index_flow(uw_payload=None)
    except Exception as exc:
        _log.warning("get_index_flow_pillar_failed", error=str(exc))
        return {"error": str(exc)}

    of = payload.get("options_flow") if isinstance(payload.get("options_flow"), dict) else {}
    breadth = payload.get("breadth") if isinstance(payload.get("breadth"), dict) else {}
    return {
        "market_tone": of.get("market_tone"),
        "put_call_ratio": of.get("put_call_ratio"),
        "net_call_premium": of.get("net_call_premium"),
        "net_put_premium": of.get("net_put_premium"),
        "dark_pool_notional": of.get("dark_pool_notional"),
        "dark_pool_trade_count": of.get("dark_pool_trade_count"),
        "options_flow_signal_count": of.get("options_flow_signal_count"),
        "top_flow_tickers": (of.get("top_flow_tickers") or [])[:10],
        "breadth": {
            "tick": breadth.get("tick"),
            "trin": breadth.get("trin"),
            "ad_line": breadth.get("ad_line"),
            "dc_line": breadth.get("dc_line"),
            "regime": breadth.get("regime"),
            "source": breadth.get("source"),
        },
        "etf_flows": payload.get("etf_flows") or {},
        "errors": payload.get("errors") or [],
    }


def get_quant_research_pillar(walk_forward: bool = False) -> dict[str, Any]:
    """Pillar C summary: vol-premium signals, simple backtest hit rates,
    historical signal hit rates by source."""
    try:
        from monitoring import dashboard_pillars as dp  # noqa: PLC0415

        payload = dp.collect_quant_research(run_walk_forward=bool(walk_forward))
    except Exception as exc:
        _log.warning("get_quant_research_pillar_failed", error=str(exc))
        return {"error": str(exc)}

    bt = payload.get("simple_backtest") if isinstance(payload.get("simple_backtest"), dict) else {}
    return {
        "vol_premium_signals": (payload.get("vol_premium_signals") or [])[:10],
        "hit_rates": (payload.get("hit_rates") or [])[:10],
        "simple_backtest": {
            "ticker": bt.get("ticker"),
            "lookback_days": bt.get("lookback_days"),
            "strategies": (bt.get("strategies") or [])[:8],
        },
        "walk_forward": payload.get("walk_forward") if walk_forward else None,
        "errors": payload.get("errors") or [],
    }


def get_ibkr_breadth() -> dict[str, Any]:
    """Live IBKR-native NYSE breadth: TICK, TRIN, AD-line, DC-line + regime tag.
    Returns `{available: false}` if TWS is not reachable."""
    try:
        from integrations import ibkr_market_data_client as ibmd  # noqa: PLC0415

        if not ibmd.is_available():
            return {"available": False, "reason": "ibkr_disabled_or_module_missing"}
        snap = ibmd.get_breadth_snapshot()
        if snap is None:
            return {"available": False, "reason": "ibkr_unreachable_or_no_data"}
        return {"available": True, **snap}
    except Exception as exc:
        _log.warning("get_ibkr_breadth_failed", error=str(exc))
        return {"available": False, "error": str(exc)}


def get_ibkr_iv_hv(tickers: list[str]) -> dict[str, Any]:
    """Live IBKR-native IV30 and HV30 for a list of tickers (max 12).
    Uses generic ticks 104,106. Falls back gracefully if TWS down."""
    try:
        from integrations import ibkr_market_data_client as ibmd  # noqa: PLC0415

        if not ibmd.is_available():
            return {"available": False, "reason": "ibkr_disabled_or_module_missing"}
        capped = [str(t).upper() for t in (tickers or [])[:12]]
        if not capped:
            return {"available": False, "reason": "no_tickers"}
        snap = ibmd.get_iv_hv_snapshot(capped)
        if snap is None:
            return {"available": False, "reason": "ibkr_unreachable_or_no_data"}
        return {"available": True, "tickers": snap}
    except Exception as exc:
        _log.warning("get_ibkr_iv_hv_failed", error=str(exc))
        return {"available": False, "error": str(exc)}


def get_recent_decisions(n: int = 5) -> dict[str, Any]:
    """Read the last N decisions from data/memory/decisions.md (TradingAgents-style
    reflection log). Used by bull/bear/PM agents to see prior calls and outcomes."""
    try:
        from shared.aac_agents.memory import recent_decisions  # noqa: PLC0415

        return {"count": int(n), "decisions": recent_decisions(n=int(n))}
    except Exception as exc:
        _log.warning("get_recent_decisions_failed", error=str(exc))
        return {"count": 0, "decisions": [], "error": str(exc)}


# ── Risk snapshot tools (for portfolio_manager agent) ────────────────────────


def get_drawdown_state() -> dict[str, Any]:
    """Return current state of the multi-day DrawdownCircuitBreaker.

    Reports peak equity, current equity, drawdown_pct, and `tripped` flag.
    Read-only — does NOT record a new account value.
    """
    try:
        from strategies.drawdown_circuit_breaker import DrawdownCircuitBreaker  # noqa: PLC0415

        cb = DrawdownCircuitBreaker()
        state = cb.current_state()
        d = state.to_dict()
        d["max_drawdown_pct"] = cb.max_drawdown_pct
        return d
    except Exception as exc:
        _log.warning("get_drawdown_state_failed", error=str(exc))
        return {"available": False, "error": str(exc)}


def get_daily_loss_status(account_value_usd: float = 0.0) -> dict[str, Any]:
    """Return whether today's P&L has breached the DailyLossGuard ceiling.

    Reports {tripped, reason, max_loss_pct, account_value_usd}.
    Read-only — fails open (tripped=False) on any error.
    """
    try:
        from strategies.daily_loss_guard import DailyLossGuard  # noqa: PLC0415

        guard = DailyLossGuard(account_value_usd=account_value_usd)
        tripped, reason = guard.is_limit_reached(account_value_usd=account_value_usd)
        return {
            "tripped": bool(tripped),
            "reason": reason or "",
            "max_loss_pct": guard.max_loss_pct,
            "account_value_usd": guard.account_value_usd,
        }
    except Exception as exc:
        _log.warning("get_daily_loss_status_failed", error=str(exc))
        return {"tripped": False, "available": False, "error": str(exc)}


def get_position_exposure() -> dict[str, Any]:
    """Return aggregate exposure summary: total $, long/short counts, top-5
    by abs(market_value). Slim version of `get_positions` for the PM agent."""
    snap = get_positions()
    if snap.get("error"):
        return {"available": False, "error": snap["error"]}
    positions = snap.get("positions") or []
    longs = [p for p in positions if (p.get("quantity") or 0) > 0]
    shorts = [p for p in positions if (p.get("quantity") or 0) < 0]
    top5 = sorted(
        positions,
        key=lambda p: abs(float(p.get("market_value") or 0)),
        reverse=True,
    )[:5]
    return {
        "count": snap.get("count", 0),
        "long_count": len(longs),
        "short_count": len(shorts),
        "total_exposure_usd": snap.get("total_exposure_usd"),
        "total_unrealized_pnl": snap.get("total_unrealized_pnl"),
        "paper": snap.get("paper"),
        "top5": [
            {
                "symbol": p.get("symbol"),
                "sec_type": p.get("sec_type"),
                "quantity": p.get("quantity"),
                "market_value": p.get("market_value"),
                "unrealized_pnl": p.get("unrealized_pnl"),
                "pnl_pct": p.get("pnl_pct"),
            }
            for p in top5
        ],
    }


def get_correlation_regime() -> dict[str, Any]:
    """Return the cached correlation regime from CorrelationTracker.

    Reports {regime, absorption_ratio, effective_n_assets, n_alerts, timestamp}
    or {available: False} if no snapshot has been computed yet this session.
    """
    try:
        from strategies.correlation_tracker import CorrelationTracker  # noqa: PLC0415

        tracker = CorrelationTracker()
        snap = tracker.last_snapshot
        if snap is None:
            return {
                "available": False,
                "reason": "no correlation snapshot computed yet (tracker is per-process)",
            }
        return {
            "available": True,
            **snap.to_dict(),
        }
    except Exception as exc:
        _log.warning("get_correlation_regime_failed", error=str(exc))
        return {"available": False, "error": str(exc)}


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
    "get_call_options_pillar": get_call_options_pillar,
    "get_index_flow_pillar": get_index_flow_pillar,
    "get_quant_research_pillar": get_quant_research_pillar,
    "get_ibkr_breadth": get_ibkr_breadth,
    "get_ibkr_iv_hv": get_ibkr_iv_hv,
    "get_recent_decisions": get_recent_decisions,
    "get_drawdown_state": get_drawdown_state,
    "get_daily_loss_status": get_daily_loss_status,
    "get_position_exposure": get_position_exposure,
    "get_correlation_regime": get_correlation_regime,
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
    {
        "type": "function",
        "function": {
            "name": "get_call_options_pillar",
            "description": (
                "AAC Pillar A summary (call options): vol-premium readings (IV/HV), "
                "rich-IV (≥1.20) and cheap-IV (≤0.85) lists, covered-call candidates, "
                "own call positions. The primary tool for the options strategist."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_index_flow_pillar",
            "description": (
                "AAC Pillar B summary (index & flow): market tone, P/C ratio, net "
                "call/put premium, dark pool notional, breadth (TICK/TRIN/AD/DC), "
                "top UW flow tickers, ETF flows. Primary tool for the flow analyst."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_quant_research_pillar",
            "description": (
                "AAC Pillar C summary (quant research): vol-premium signals, simple "
                "backtest hit rates by strategy, historical signal hit rates by source. "
                "Set walk_forward=true for the slow backtest. Primary tool for the "
                "quant analyst."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "walk_forward": {"type": "boolean", "default": False},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ibkr_breadth",
            "description": (
                "Live IBKR-native NYSE breadth snapshot (TICK-NYSE, TRIN-NYSE, "
                "AD-NYSE, DC-NYSE) plus a regime tag. Returns {available: false} "
                "when TWS is not reachable."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ibkr_iv_hv",
            "description": (
                "Live IBKR-native IV30 and HV30 for a list of tickers (max 12). "
                "Use to validate a vol-premium thesis with broker-native data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of underlying tickers (max 12)",
                    },
                },
                "required": ["tickers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_decisions",
            "description": (
                "Read the last N entries from the decisions memory log "
                "(data/memory/decisions.md). Each entry has timestamp, thesis, "
                "verdict, and (when known) realised P&L. Use to learn from past calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "default": 5},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_drawdown_state",
            "description": (
                "Snapshot of the multi-day DrawdownCircuitBreaker: peak equity, "
                "current equity, drawdown_pct, and tripped flag. Read-only."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_loss_status",
            "description": (
                "Check the DailyLossGuard ceiling vs today's P&L. Returns "
                "{tripped, reason, max_loss_pct, account_value_usd}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "account_value_usd": {
                        "type": "number",
                        "description": "Override account size (0 = use default).",
                        "default": 0,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_position_exposure",
            "description": (
                "Aggregate exposure summary: total $, long/short counts, top-5 "
                "positions by abs market value. Use this instead of get_positions "
                "when you only need a sizing overview."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_correlation_regime",
            "description": (
                "Cached snapshot from CorrelationTracker: regime "
                "(normal/decorrelating/contagion), absorption_ratio, "
                "effective_n_assets, n_alerts. Returns available=False if no "
                "snapshot has been computed yet this process."
            ),
            "parameters": {"type": "object", "properties": {}},
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
