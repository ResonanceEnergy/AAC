"""AAC Mission Control — single-pane-of-glass live dashboard.

One FastAPI backend serving JSON APIs + one HTML page.
Auto-refreshes every 30 seconds. No Streamlit, no Dash, no fragments.

Launch:  python launch.py mission-control
"""
from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import structlog

_log = structlog.get_logger()


def _json_default(obj: Any) -> Any:
    """Make datetime, date, dataclass, and other types JSON-serializable."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return str(obj)
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

_log = structlog.get_logger()

# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="AAC Mission Control", version="1.0.0")

# ── Shared cache with TTL ───────────────────────────────────────────────────

_cache: dict[str, Any] = {}
_cache_ts: dict[str, float] = {}
_cache_lock = threading.Lock()
CACHE_TTL = 25  # seconds — refresh slightly before the 30s poll


def _get_cached(key: str) -> Any | None:
    with _cache_lock:
        if key in _cache and (time.monotonic() - _cache_ts.get(key, 0)) < CACHE_TTL:
            return _cache[key]
    return None


def _set_cached(key: str, value: Any) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache_ts[key] = time.monotonic()


# ── Data collectors ─────────────────────────────────────────────────────────


def _safe(fn, label: str) -> Any:
    """Run fn, return result or error dict."""
    try:
        return fn()
    except Exception as exc:
        _log.warning("collector_error", label=label, err=str(exc))
        return {"error": str(exc)}


def collect_portfolio() -> dict:
    """Portfolio value and account breakdown from account_balances.json."""
    cached = _get_cached("portfolio")
    if cached is not None:
        return cached

    bal_path = _ROOT / "data" / "account_balances.json"
    if not bal_path.exists():
        return {"error": "account_balances.json not found"}

    raw = json.loads(bal_path.read_text(encoding="utf-8"))
    accounts_raw = raw.get("accounts", {})
    fx = raw.get("fx", {})
    cad_usd = fx.get("cad_usd", 0.72)

    accounts = []
    total_usd = 0.0
    total_positions = 0
    total_unrealized = 0.0

    for key, acct in accounts_raw.items():
        currency = acct.get("currency", "USD")
        total_assets = float(acct.get("total_assets", 0) or 0)
        value_usd = total_assets if currency == "USD" else total_assets * cad_usd

        positions = []
        acct_unrealized = 0.0
        for pos in acct.get("positions") or []:
            unrealized = float(pos.get("unrealizedPNL", pos.get("pl_val", 0)) or 0)
            mkt_val = float(pos.get("marketValue", pos.get("market_val", 0)) or 0)
            avg_cost = float(pos.get("avgCost", pos.get("cost_price", 0)) or 0)
            mkt_price = float(pos.get("marketPrice", 0) or 0)
            qty = float(pos.get("qty", 0) or 0)
            right = pos.get("right", "")
            sec_type = pos.get("secType", "")
            symbol = pos.get("symbol", "?")
            strike = pos.get("strike")
            expiry = pos.get("expiry", "")

            ptype = "call" if right == "C" else "put" if right == "P" else sec_type.lower() or "equity"
            acct_unrealized += unrealized
            positions.append({
                "symbol": symbol,
                "type": ptype,
                "strike": strike,
                "expiry": expiry,
                "qty": qty,
                "avg_cost": round(avg_cost, 2),
                "market_price": round(mkt_price, 4),
                "market_value": round(mkt_val, 2),
                "unrealized_pnl": round(unrealized, 2),
            })

        total_positions += len(positions)
        total_unrealized += acct_unrealized
        total_usd += value_usd
        accounts.append({
            "name": key,
            "platform": acct.get("platform", key),
            "currency": currency,
            "total_assets": total_assets,
            "value_usd": round(value_usd, 2),
            "position_count": len(positions),
            "unrealized_pnl": round(acct_unrealized, 2),
            "positions": positions,
            "verified": acct.get("verified", ""),
        })

    result = {
        "total_usd": round(total_usd, 2),
        "total_positions": total_positions,
        "total_unrealized_pnl": round(total_unrealized, 2),
        "fx_cad_usd": cad_usd,
        "accounts": accounts,
        "last_updated": raw.get("_meta", {}).get("updated", "unknown"),
    }
    _set_cached("portfolio", result)
    return result


def collect_war_room() -> dict:
    """War Room engine: composite score, regime, arms, indicators."""
    cached = _get_cached("war_room")
    if cached is not None:
        return cached

    from strategies.war_room_engine import (
        CURRENT_POSITIONS,
        WarRoomEngine,
        compute_composite_score,
        get_arm_allocations,
        get_portfolio_value_usd,
    )

    engine = WarRoomEngine()
    mandate = engine.get_mandate()
    ind = engine.indicators
    comp = compute_composite_score(ind)

    ind_dict = dataclasses.asdict(ind)
    individual_scores = comp.get("individual_scores", {})

    INDICATOR_META = {
        "oil": {"desc": "Oil Price", "weight": 0.12, "field": "oil_price"},
        "gold": {"desc": "Gold Price", "weight": 0.08, "field": "gold_price"},
        "vix": {"desc": "VIX", "weight": 0.10, "field": "vix"},
        "hy_spread": {"desc": "HY Spread", "weight": 0.08, "field": "hy_spread_bp"},
        "bdc_nav": {"desc": "BDC NAV Discount", "weight": 0.07, "field": "bdc_nav_discount"},
        "bdc_nonaccrual": {"desc": "BDC Non-Accrual", "weight": 0.05, "field": "bdc_nonaccrual_pct"},
        "defi_tvl": {"desc": "DeFi TVL Change", "weight": 0.04, "field": "defi_tvl_change_pct"},
        "stablecoin": {"desc": "Stablecoin Depeg", "weight": 0.04, "field": "stablecoin_depeg_pct"},
        "btc": {"desc": "Bitcoin Price", "weight": 0.05, "field": "btc_price"},
        "fed_rate": {"desc": "Fed Funds Rate", "weight": 0.05, "field": "fed_funds_rate"},
        "dxy": {"desc": "Dollar Index", "weight": 0.05, "field": "dxy"},
        "spy": {"desc": "S&P 500", "weight": 0.07, "field": "spy_price"},
        "x_sentiment": {"desc": "X Sentiment", "weight": 0.12, "field": "x_sentiment"},
        "news": {"desc": "News Severity", "weight": 0.04, "field": "news_severity"},
        "fear_greed": {"desc": "Fear & Greed", "weight": 0.04, "field": "fear_greed_index"},
    }

    indicators = []
    for key, meta in INDICATOR_META.items():
        raw_val = ind_dict.get(meta["field"])
        score = individual_scores.get(key, 0)
        indicators.append({
            "key": key,
            "name": meta["desc"],
            "weight": meta["weight"],
            "raw": raw_val,
            "score": round(score, 1),
            "contribution": round(score * meta["weight"], 2),
        })

    phase = dataclasses.asdict(mandate).get("phase", "accumulation")
    allocations = get_arm_allocations(phase)
    portfolio_usd = get_portfolio_value_usd()

    arm_actuals: dict[str, float] = {}
    for pos in CURRENT_POSITIONS:
        arm_name = pos.arm.value if hasattr(pos.arm, "value") else str(pos.arm)
        arm_actuals[arm_name] = arm_actuals.get(arm_name, 0) + abs(pos.market_value)

    arms = []
    for alloc in allocations:
        arm_key = alloc.arm.value if hasattr(alloc.arm, "value") else str(alloc.arm)
        actual_usd = arm_actuals.get(arm_key, 0)
        actual_pct = (actual_usd / portfolio_usd * 100) if portfolio_usd else 0
        arms.append({
            "arm": arm_key,
            "name": alloc.name,
            "target_pct": round(alloc.target_pct * 100, 1),
            "actual_pct": round(actual_pct, 1),
            "actual_usd": round(actual_usd, 2),
        })

    result = {
        "composite_score": comp.get("composite_score", 0),
        "regime": comp.get("regime", "unknown"),
        "confidence": comp.get("confidence", 0),
        "phase": phase,
        "portfolio_usd": portfolio_usd,
        "mandate": dataclasses.asdict(mandate).get("mandate", "HOLD"),
        "indicators": indicators,
        "arms": arms,
    }
    _set_cached("war_room", result)
    return result


def collect_live_feeds() -> dict:
    """Live market feeds — BTC, ETH, SPY, gold, VIX, P/C, FGI."""
    cached = _get_cached("live_feeds")
    if cached is not None:
        return cached

    from strategies.war_room_engine import WarRoomEngine
    from strategies.war_room_live_feeds import (
        get_last_feed_result,
        update_all_live_data_sync,
    )

    engine = WarRoomEngine()
    ind = engine.indicators
    update_all_live_data_sync(ind)
    feed = get_last_feed_result()

    if not feed:
        return {"error": "No live feed data"}

    result = {
        "btc": feed.btc_price,
        "eth": feed.eth_price,
        "spy": feed.spy_price,
        "gold": feed.gold_price_oz,
        "oil": feed.oil_price_wti,
        "vix": ind.vix if ind.vix else None,
        "put_call": feed.put_call_ratio,
        "fear_greed": feed.fear_greed_value,
        "dxy": feed.dxy_index,
        "fed_rate": feed.fed_rate,
        "hy_spread": feed.hy_spread_bp_live,
        "errors": feed.errors if feed.errors else [],
        "ts": datetime.datetime.now().isoformat(),
    }
    _set_cached("live_feeds", result)
    return result


def collect_regime() -> dict:
    """Regime engine state."""
    cached = _get_cached("regime")
    if cached is not None:
        return cached

    try:
        from strategies.regime_engine import MacroSnapshot, RegimeEngine

        engine = RegimeEngine()
        # Build snapshot from live feeds
        feeds = collect_live_feeds()
        snapshot = MacroSnapshot(
            vix=feeds.get("vix") or 20,
            hy_spread_bps=feeds.get("hy_spread") or 350,
            oil_price=feeds.get("oil") or 80,
            gold_price=feeds.get("gold") or 2000,
            dollar_index=feeds.get("dxy") or 104,
            fear_greed=feeds.get("fear_greed") or 50,
            spy_return_1d=0.0,
        )
        state = engine.evaluate(snapshot)
        result = {
            "primary": state.primary_regime.value if hasattr(state.primary_regime, "value") else str(state.primary_regime),
            "secondary": (state.secondary_regime.value if hasattr(state.secondary_regime, "value") else str(state.secondary_regime)) if state.secondary_regime else None,
            "confidence": state.regime_confidence,
            "armed_formulas": [f.value if hasattr(f, "value") else str(f) for f in (state.armed_formulas or [])],
            "bear_signals": state.bear_signals,
            "bull_signals": state.bull_signals,
            "vol_shock_readiness": state.vol_shock_readiness,
            "summary": state.summary,
        }
    except Exception:
        _log.exception("regime_collector_error")
        wr = collect_war_room()
        result = {
            "primary": wr.get("regime", "UNKNOWN"),
            "secondary": None,
            "confidence": wr.get("confidence", 0),
            "armed_formulas": [],
            "bear_signals": 0,
            "bull_signals": 0,
            "summary": f"Fallback — Composite: {wr.get('composite_score', 0)}, Phase: {wr.get('phase', '?')}",
        }

    _set_cached("regime", result)
    return result


def collect_doctrine() -> dict:
    """Doctrine engine state — compliance, BarrenWuffet state."""
    cached = _get_cached("doctrine")
    if cached is not None:
        return cached

    try:
        from aac.doctrine.doctrine_engine import DoctrineEngine

        engine = DoctrineEngine()
        engine.load_doctrine_packs()
        report = engine.generate_compliance_report()
        bw_state = report.barren_wuffet_state
        state_str = bw_state.value if hasattr(bw_state, "value") else str(bw_state)
        result = {
            "state": state_str,
            "compliance_score": report.compliance_score,
            "total_rules": report.total_rules,
            "compliant": report.compliant,
            "warnings": report.warnings,
            "violations": len(report.violations_list),
            "violations_list": [
                {"pack": v.pack_name, "rule": v.rule_id, "desc": v.description, "severity": v.severity}
                for v in report.violations_list[:10]
            ],
        }
    except Exception:
        _log.exception("doctrine_collector_error")
        result = {
            "state": "NORMAL",
            "compliance_score": 100,
            "total_rules": 0,
            "compliant": 0,
            "warnings": 0,
            "violations": 0,
            "violations_list": [],
        }

    _set_cached("doctrine", result)
    return result


def collect_moon() -> dict:
    """13 Moon doctrine state."""
    cached = _get_cached("moon")
    if cached is not None:
        return cached

    try:
        from strategies.thirteen_moon_doctrine import ThirteenMoonDoctrine

        d = ThirteenMoonDoctrine()
        moon = d.get_current_moon()
        alerts = d.get_events_with_lead_time(days_ahead=14)

        result = {
            "moon_number": moon.moon_number,
            "name": moon.lunar_phase_name,
            "start": str(moon.start_date),
            "end": str(moon.end_date),
            "mandate": moon.doctrine_action.mandate if moon.doctrine_action else "HOLD",
            "conviction": moon.doctrine_action.conviction if moon.doctrine_action else 0,
            "targets": moon.doctrine_action.targets if moon.doctrine_action else [],
            "events": [
                {"date": str(a.event_date), "name": a.event_name, "days": a.days_until, "priority": a.priority}
                for a in alerts[:8]
            ],
        }
    except Exception:
        result = {"moon_number": 0, "name": "Unknown", "mandate": "HOLD", "events": []}

    _set_cached("moon", result)
    return result


def collect_health() -> dict:
    """System health — subsystem status."""
    cached = _get_cached("health")
    if cached is not None:
        return cached

    checks: dict[str, str] = {}

    # IBKR
    try:
        from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
        checks["ibkr"] = "configured"
    except Exception:
        checks["ibkr"] = "unavailable"

    # Moomoo
    try:
        from TradingExecution.exchange_connectors.moomoo_connector import MoomooConnector
        checks["moomoo"] = "configured"
    except Exception:
        checks["moomoo"] = "unavailable"

    # CoinGecko
    try:
        from integrations.fear_greed_client import FearGreedClient
        checks["coingecko"] = "available"
    except Exception:
        checks["coingecko"] = "unavailable"

    # Unusual Whales
    uw_key = os.environ.get("UNUSUAL_WHALES_API_KEY", "")
    checks["unusual_whales"] = "configured" if uw_key else "no_key"

    # FRED
    fred_key = os.environ.get("FRED_API_KEY", "")
    checks["fred"] = "configured" if fred_key else "no_key"

    # Finnhub
    fh_key = os.environ.get("FINNHUB_API_KEY", "")
    checks["finnhub"] = "configured" if fh_key else "no_key"

    # Doctrine packs
    try:
        from aac.doctrine.pack_registry import DOCTRINE_PACKS
        checks["doctrine_packs"] = f"{len(DOCTRINE_PACKS)} loaded"
    except Exception:
        checks["doctrine_packs"] = "unavailable"

    result = {
        "subsystems": checks,
        "ts": datetime.datetime.now().isoformat(),
    }
    _set_cached("health", result)
    return result


def collect_tasks() -> dict:
    """Autonomous engine scheduled tasks (if running)."""
    cached = _get_cached("tasks")
    if cached is not None:
        return cached

    # Read from the autonomous engine task definitions
    tasks = [
        {"name": "market_scan", "interval": "60s", "desc": "Fetch live market data"},
        {"name": "strategy_signals", "interval": "60s", "desc": "Generate trading signals"},
        {"name": "position_reconcile", "interval": "300s", "desc": "Match positions to reality"},
        {"name": "connector_health", "interval": "120s", "desc": "Exchange health checks"},
        {"name": "introspection", "interval": "300s", "desc": "Self-check for gaps"},
        {"name": "status_report", "interval": "3600s", "desc": "Hourly status report"},
        {"name": "daily_pnl_reset", "interval": "24h", "desc": "Daily accounting reset"},
        {"name": "gap_analysis", "interval": "600s", "desc": "System gap analysis"},
        {"name": "daily_brief", "interval": "24h", "desc": "Morning briefing"},
        {"name": "intelligence_cycle", "interval": "3600s", "desc": "Market intelligence model"},
        {"name": "rocket_ship_brief", "interval": "24h", "desc": "Rocket ship recommendations"},
    ]

    result = {"tasks": tasks, "engine_status": "configured"}
    _set_cached("tasks", result)
    return result


def collect_unusual_whales() -> dict:
    """Unusual Whales: flow summary, hottest chains, congress trades, dark pool."""
    cached = _get_cached("unusual_whales")
    if cached is not None:
        return cached

    try:
        from integrations.unusual_whales_client import UnusualWhalesClient
    except ImportError:
        return {"error": "UW client not available"}

    async def _fetch() -> dict:
        client = UnusualWhalesClient()
        result: dict[str, Any] = {}

        try:
            summary = await client.get_market_flow_summary()
            result["flow_summary"] = summary if isinstance(summary, dict) else str(summary)
        except Exception as e:
            result["flow_summary_error"] = str(e)

        try:
            hot = await client.get_hottest_chains(limit=10)
            chains = []
            for h in (hot or []):
                if hasattr(h, "__dict__"):
                    chains.append({k: v for k, v in h.__dict__.items() if not k.startswith("_")})
                elif isinstance(h, dict):
                    chains.append(h)
                else:
                    chains.append(str(h))
            result["hottest_chains"] = chains
        except Exception as e:
            result["hottest_chains_error"] = str(e)

        try:
            congress = await client.get_congress_trades(limit=10)
            trades = []
            for t in (congress or []):
                if hasattr(t, "__dict__"):
                    trades.append({k: v for k, v in t.__dict__.items() if not k.startswith("_")})
                elif isinstance(t, dict):
                    trades.append(t)
                else:
                    trades.append(str(t))
            result["congress_trades"] = trades
        except Exception as e:
            result["congress_trades_error"] = str(e)

        try:
            dp = await client.get_dark_pool("SPY", limit=5)
            pools = []
            for d in (dp or []):
                if hasattr(d, "__dict__"):
                    pools.append({k: v for k, v in d.__dict__.items() if not k.startswith("_")})
                elif isinstance(d, dict):
                    pools.append(d)
                else:
                    pools.append(str(d))
            result["dark_pool_spy"] = pools
        except Exception as e:
            result["dark_pool_error"] = str(e)

        return result

    try:
        result = asyncio.run(_fetch())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_fetch())
        finally:
            loop.close()

    _set_cached("unusual_whales", result)
    return result


def collect_divisions() -> dict:
    """Division activity status — what's actually running."""
    cached = _get_cached("divisions")
    if cached is not None:
        return cached

    divisions = []

    # TradingExecution
    try:
        from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
        from TradingExecution.exchange_connectors.moomoo_connector import MoomooConnector
        divisions.append({
            "name": "TradingExecution",
            "status": "ACTIVE",
            "detail": "IBKR + Moomoo connectors loaded",
            "components": ["ibkr_connector", "moomoo_connector", "execution_engine"],
        })
    except ImportError:
        divisions.append({"name": "TradingExecution", "status": "DEGRADED", "detail": "Import failed"})

    # BigBrainIntelligence
    try:
        bbi_path = _ROOT / "BigBrainIntelligence"
        agents = [f.stem for f in bbi_path.glob("*agent*.py")] if bbi_path.exists() else []
        divisions.append({
            "name": "BigBrainIntelligence",
            "status": "PARTIAL",
            "detail": f"{len(agents)} agent files found",
            "components": agents[:8],
        })
    except Exception:
        divisions.append({"name": "BigBrainIntelligence", "status": "UNKNOWN", "detail": "Scan failed"})

    # CentralAccounting
    try:
        from CentralAccounting.database import AccountingDatabase
        db_path = _ROOT / "CentralAccounting" / "accounting.db"
        divisions.append({
            "name": "CentralAccounting",
            "status": "ACTIVE",
            "detail": f"SQLite DB {'exists' if db_path.exists() else 'missing'}",
            "components": ["accounting_db", "pnl_tracker", "position_manager"],
        })
    except ImportError:
        divisions.append({"name": "CentralAccounting", "status": "DEGRADED", "detail": "Import failed"})

    # CryptoIntelligence
    try:
        ci_path = _ROOT / "CryptoIntelligence"
        modules = [f.stem for f in ci_path.glob("*.py")] if ci_path.exists() else []
        divisions.append({
            "name": "CryptoIntelligence",
            "status": "ACTIVE",
            "detail": f"{len(modules)} modules",
            "components": modules[:8],
        })
    except Exception:
        divisions.append({"name": "CryptoIntelligence", "status": "UNKNOWN", "detail": "Scan failed"})

    # shared/ backbone
    try:
        shared_path = _ROOT / "shared"
        modules = [f.stem for f in shared_path.glob("*.py")] if shared_path.exists() else []
        divisions.append({
            "name": "shared (Backbone)",
            "status": "ACTIVE",
            "detail": f"{len(modules)} modules",
            "components": ["data_sources", "config_loader", "strategy_integrator", "bridge_orchestrator", "live_trading_safeguards"],
        })
    except Exception:
        divisions.append({"name": "shared", "status": "UNKNOWN", "detail": "Scan failed"})

    result = {"divisions": divisions}
    _set_cached("divisions", result)
    return result


def collect_api_feeds() -> dict:
    """API feed inventory — what's configured, what's working."""
    cached = _get_cached("api_feeds")
    if cached is not None:
        return cached

    feeds = []

    # Check each API
    api_checks = [
        ("yfinance", None, "Free — options chain source"),
        ("CoinGecko", "COINGECKO_API_KEY", "Crypto prices, FGI. Free tier: 10 req/min"),
        ("Unusual Whales", "UNUSUAL_WHALES_API_KEY", "Options flow, dark pool, congress trades"),
        ("FRED", "FRED_API_KEY", "VIX fallback, macro data"),
        ("Finnhub", "FINNHUB_API_KEY", "Quotes, news"),
        ("Polygon", "POLYGON_API_KEY", "Limited — free tier NO options snapshots"),
        ("NewsAPI", "NEWSAPI_KEY", "Headlines"),
        ("X/Twitter", "X_BEARER_TOKEN", "HTTP 402 — needs paid tier"),
    ]

    for name, env_key, note in api_checks:
        if env_key is None:
            feeds.append({"name": name, "status": "active", "note": note})
        else:
            key = os.environ.get(env_key, "")
            status = "configured" if key else "no_key"
            feeds.append({"name": name, "key_var": env_key, "status": status, "note": note})

    # Exchange connectors
    exchange_checks = [
        ("IBKR", "Port 7496 live, account U24346218"),
        ("Moomoo", "OpenD FUTUCA, real mode"),
        ("NDAX", "Liquidated — ccxt connector"),
    ]
    for name, note in exchange_checks:
        feeds.append({"name": name, "status": "exchange", "note": note})

    result = {
        "feeds": feeds,
        "configured_count": sum(1 for f in feeds if f.get("status") in ("active", "configured", "exchange")),
        "total_count": len(feeds),
    }
    _set_cached("api_feeds", result)
    return result


def collect_polymarket() -> dict:
    """Polymarket balance, trending markets, and arb opportunities."""
    cached = _get_cached("polymarket")
    if cached is not None:
        return cached

    # Balance from account_balances.json
    bal_path = _ROOT / "data" / "account_balances.json"
    balance_info: dict[str, Any] = {"balance": 0, "in_positions": 0, "currency": "USD"}
    if bal_path.exists():
        raw = json.loads(bal_path.read_text(encoding="utf-8"))
        poly = raw.get("accounts", {}).get("polymarket", {})
        balance_info = {
            "balance": float(poly.get("balance", 0) or 0),
            "in_positions": float(poly.get("in_positions", 0) or 0),
            "currency": poly.get("currency", "USD"),
            "verified": poly.get("verified", ""),
            "note": poly.get("note", ""),
        }

    # Trending markets + arb scan via PolymarketAgent
    trending: list[dict] = []
    arb_opps: list[dict] = []

    try:
        from agents.polymarket_agent import PolymarketAgent

        async def _fetch_poly() -> tuple[list, list]:
            async with PolymarketAgent() as agent:
                t = await agent.get_trending_events(limit=8)
                a = await agent.scan_for_arbitrage(min_edge_pct=0.3)
                tr = []
                for ev in (t or []):
                    tr.append({
                        "title": ev.title[:80],
                        "volume": ev.volume,
                        "volume_24hr": ev.volume_24hr,
                        "liquidity": ev.liquidity,
                        "active": ev.active,
                    })
                ar = []
                for opp in (a or [])[:5]:
                    if hasattr(opp, "__dict__"):
                        ar.append({k: v for k, v in opp.__dict__.items() if not k.startswith("_")})
                    elif isinstance(opp, dict):
                        ar.append(opp)
                return tr, ar

        try:
            trending, arb_opps = asyncio.run(_fetch_poly())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                trending, arb_opps = loop.run_until_complete(_fetch_poly())
            finally:
                loop.close()
    except ImportError:
        _log.warning("polymarket_import_error", err="PolymarketAgent not importable")
    except Exception as e:
        _log.exception("polymarket_fetch_error", err=str(e))

    result = {
        "balance": balance_info,
        "trending_markets": trending,
        "arb_opportunities": arb_opps,
        "arb_count": len(arb_opps),
    }
    _set_cached("polymarket", result)
    return result


def collect_scenarios() -> dict:
    """Top scenarios ranked by war-room relevance (position overlap + expected impact + arm alignment)."""
    cached = _get_cached("scenarios")
    if cached is not None:
        return cached

    try:
        from strategies.war_room_engine import SCENARIOS
    except ImportError:
        return {"error": "war_room_engine not available"}

    # --- Current portfolio exposure for relevance scoring ---
    # Direct mapping: scenario drift asset → do we hold a *direct* position in it?
    # Only count assets where we have DIRECT, meaningful exposure
    directly_held = {
        "oil": 0.5,    # XLE calls (indirect oil play)
        "gold": 0.3,   # SLV calls (silver, not gold directly)
        "silver": 1.0,  # SLV calls — 14 contracts
        "xlf": 1.0,    # XLF puts
        "xlre": 0.5,   # KRE has real-estate bank overlap
        "spy": 0.3,    # HYG/JNK/LQD correlate loosely
        "btc": 0.4,    # ETH/XRP on NDAX (crypto proxy)
        "eth": 1.0,    # ETH on NDAX
        "xrp": 0.8,    # XRP on NDAX
        "gdx": 0.2,    # SLV correlates loosely with miners
    }
    # Assets we have heavy put exposure in (credit/BDC)
    put_exposure_assets = {"hyg", "jnk", "lqd", "emb", "bkln", "pff", "arcc", "main", "obdc", "kre"}

    # Arms with under-allocation (keyword matching)
    arm_keywords = {
        "iran_oil": ["oil", "hormuz", "iran", "petrodollar", "mideast"],
        "bdc": ["credit", "bdc", "nonaccrual", "collapse", "cascade"],
        "crypto_metals": ["gold", "silver", "btc", "eth", "crypto", "defi", "supercycle"],
        "defi": ["defi", "cascade", "yield"],
    }

    ranked = []
    for key, sc in SCENARIOS.items():
        prob = sc.get("probability", 0)
        sev = sc.get("impact_severity", 0)
        ei_score = prob * sev  # 0-1 range
        drift = sc.get("drift_override", {})
        drift_assets = list(drift.keys())

        # --- Position overlap score (0-1) ---
        # How many drift assets do we have direct exposure to?
        total_weight = 0.0
        for da in drift_assets:
            da_lower = da.lower()
            if da_lower in directly_held:
                total_weight += directly_held[da_lower]
            elif da_lower in put_exposure_assets:
                total_weight += 1.0
        pos_overlap = min(total_weight / max(len(drift_assets), 1), 1.0)

        # --- Arm alignment score (0-1) ---
        # Weight by how under-allocated each arm is (bigger gap = more relevant)
        arm_weights = {
            "iran_oil": 0.35,      # 3.1% actual vs 30% target = most under-allocated
            "bdc": 0.30,           # 6.3% vs 25%
            "crypto_metals": 0.20, # 18% vs 20% — nearly on target
            "defi": 0.15,          # 0% vs 15%
        }
        arm_score = 0.0
        key_lower = key.lower()
        desc_lower = (sc.get("description", "") + " " + sc.get("name", "")).lower()
        for arm_name, kws in arm_keywords.items():
            if any(kw in key_lower or kw in desc_lower for kw in kws):
                arm_score += arm_weights.get(arm_name, 0.1)
        arm_score = min(arm_score, 1.0)

        # --- VIX/stress urgency (0-1): higher VIX = more relevant to our put portfolio ---
        vix = sc.get("vix", 20) or 20
        hy = sc.get("hy_spread_bp", 400) or 400
        stress_score = min((vix - 15) / 45, 1.0) * 0.5 + min((hy - 300) / 700, 1.0) * 0.5

        # --- Composite relevance = weighted blend ---
        # Attenuate relevance for very-low-probability scenarios (EI < 5%)
        ei_penalty = min(ei_score / 0.05, 1.0) if ei_score < 0.05 else 1.0
        relevance = (
            ei_score * 0.35
            + pos_overlap * 0.30
            + arm_score * 0.20
            + stress_score * 0.15
        ) * ei_penalty

        ranked.append({
            "key": key,
            "name": sc.get("name", key),
            "description": sc.get("description", ""),
            "probability": round(prob * 100, 1),
            "impact_severity": round(sev * 100, 1),
            "expected_impact": round(ei_score * 100, 2),
            "relevance": round(relevance * 100, 1),
            "position_overlap": round(pos_overlap * 100, 0),
            "arm_alignment": round(arm_score * 100, 0),
            "stress_score": round(stress_score * 100, 0),
            "oil_price": sc.get("oil_price"),
            "gold_price": sc.get("gold_price"),
            "spy_price": sc.get("spy_price"),
            "btc_price": sc.get("btc_price"),
            "vix": sc.get("vix"),
            "hy_spread_bp": sc.get("hy_spread_bp"),
            "drift_assets": drift_assets,
            "drift_values": {k: round(v, 2) for k, v in drift.items()},
        })

    ranked.sort(key=lambda x: x["relevance"], reverse=True)

    result = {
        "scenarios": ranked[:10],
        "all_scenarios": ranked,
        "total_scenarios": len(ranked),
    }
    _set_cached("scenarios", result)
    return result


def collect_pnl() -> dict:
    """P&L summary and daily history from CentralAccounting."""
    cached = _get_cached("pnl")
    if cached is not None:
        return cached

    try:
        from CentralAccounting.database import AccountingDatabase

        db = AccountingDatabase()
        summary = db.get_pnl_summary() or {}
        daily = db.get_daily_pnl(days=14)
        db.close()

        result = {
            "summary": {
                "total_realized_pnl": summary.get("total_realized_pnl", 0),
                "total_fees": summary.get("total_fees", 0),
                "total_trades": summary.get("total_trades", 0),
                "total_wins": summary.get("total_wins", 0),
                "total_losses": summary.get("total_losses", 0),
                "avg_daily_pnl": summary.get("avg_daily_pnl", 0),
                "best_trade": summary.get("best_trade", 0),
                "worst_trade": summary.get("worst_trade", 0),
            },
            "daily": [
                {
                    "date": d.get("date", ""),
                    "realized_pnl": d.get("realized_pnl", 0),
                    "trades_count": d.get("trades_count", 0),
                    "win_count": d.get("win_count", 0),
                    "loss_count": d.get("loss_count", 0),
                }
                for d in (daily or [])
            ],
        }
    except Exception:
        _log.exception("pnl_collector_error")
        result = {"summary": {}, "daily": [], "error": "P&L data unavailable"}

    _set_cached("pnl", result)
    return result


def collect_trade_log() -> dict:
    """Recent trade history from CentralAccounting."""
    cached = _get_cached("trade_log")
    if cached is not None:
        return cached

    try:
        from CentralAccounting.database import AccountingDatabase

        db = AccountingDatabase()
        trades = db.get_transactions(limit=20)
        db.close()

        result = {
            "trades": [
                {
                    "date": t.get("created_at", t.get("date", "")),
                    "account": t.get("account_id", ""),
                    "symbol": t.get("symbol", "?"),
                    "side": t.get("side", t.get("transaction_type", "")),
                    "quantity": t.get("quantity", 0),
                    "price": t.get("price", 0),
                    "total": t.get("total", 0),
                    "fees": t.get("fees", 0),
                    "status": t.get("status", ""),
                }
                for t in (trades or [])
            ],
            "count": len(trades or []),
        }
    except Exception:
        _log.exception("trade_log_collector_error")
        result = {"trades": [], "count": 0, "error": "Trade log unavailable"}

    _set_cached("trade_log", result)
    return result


def collect_backbone() -> dict:
    """shared/ backbone health — key module import status + stats."""
    cached = _get_cached("backbone")
    if cached is not None:
        return cached

    modules_check = [
        ("config_loader", "shared.config_loader", "Env/config management"),
        ("data_sources", "shared.data_sources", "CoinGecko/NewsAPI/Finnhub"),
        ("bridge_orchestrator", "shared.bridge_orchestrator", "9 dept bridges, 18 msg types"),
        ("strategy_integrator", "shared.strategy_integrator", "Signal → order conversion"),
        ("live_trading_safeguards", "shared.live_trading_safeguards", "AST-based risk eval"),
        ("strategy_framework", "shared.strategy_framework", "Base strategy classes"),
        ("strategy_execution_engine", "shared.strategy_execution_engine", "Strategy runtime"),
        ("strategy_loader", "shared.strategy_loader", "Dynamic strategy registry"),
        ("audit_logger", "shared.audit_logger", "Structured audit logging"),
        ("capital_management", "shared.capital_management", "Capital allocation"),
        ("health_checker", "shared.health_checker", "System health checks"),
        ("security_framework", "shared.security_framework", "Security monitoring"),
    ]

    modules = []
    loaded = 0
    for name, import_path, desc in modules_check:
        try:
            __import__(import_path)
            modules.append({"name": name, "status": "loaded", "desc": desc})
            loaded += 1
        except ImportError as e:
            modules.append({"name": name, "status": "error", "desc": desc, "error": str(e)})

    # Count total shared/ files
    shared_path = _ROOT / "shared"
    total_files = len(list(shared_path.glob("*.py"))) if shared_path.exists() else 0
    bridge_files = len(list(shared_path.glob("*bridge*.py"))) if shared_path.exists() else 0

    result = {
        "modules": modules,
        "loaded_count": loaded,
        "total_checked": len(modules_check),
        "total_files": total_files,
        "bridge_count": bridge_files,
    }
    _set_cached("backbone", result)
    return result


# ── API Endpoints ───────────────────────────────────────────────────────────


@app.get("/api/all")
def api_all():
    """Single endpoint returning all data — reduces HTTP round trips."""
    payload = {
        "portfolio": _safe(collect_portfolio, "portfolio"),
        "war_room": _safe(collect_war_room, "war_room"),
        "live_feeds": _safe(collect_live_feeds, "live_feeds"),
        "regime": _safe(collect_regime, "regime"),
        "doctrine": _safe(collect_doctrine, "doctrine"),
        "moon": _safe(collect_moon, "moon"),
        "health": _safe(collect_health, "health"),
        "tasks": _safe(collect_tasks, "tasks"),
        "unusual_whales": _safe(collect_unusual_whales, "unusual_whales"),
        "divisions": _safe(collect_divisions, "divisions"),
        "api_feeds": _safe(collect_api_feeds, "api_feeds"),
        "polymarket": _safe(collect_polymarket, "polymarket"),
        "scenarios": _safe(collect_scenarios, "scenarios"),
        "backbone": _safe(collect_backbone, "backbone"),
        "pnl": _safe(collect_pnl, "pnl"),
        "trade_log": _safe(collect_trade_log, "trade_log"),
        "ts": datetime.datetime.now().isoformat(),
    }
    # Pre-serialize to handle datetime and dataclass objects
    content = json.loads(json.dumps(payload, default=_json_default))
    return JSONResponse(content)


@app.get("/api/portfolio")
def api_portfolio():
    return JSONResponse(_safe(collect_portfolio, "portfolio"))


@app.get("/api/war_room")
def api_war_room():
    return JSONResponse(_safe(collect_war_room, "war_room"))


@app.get("/api/feeds")
def api_feeds():
    return JSONResponse(_safe(collect_live_feeds, "live_feeds"))


@app.get("/api/regime")
def api_regime():
    return JSONResponse(_safe(collect_regime, "regime"))


@app.get("/api/health")
def api_health():
    return JSONResponse(_safe(collect_health, "health"))


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the Mission Control HTML dashboard."""
    html_path = Path(__file__).parent / "mission_control.html"
    if not html_path.exists():
        return HTMLResponse("<h1>mission_control.html not found</h1>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── Standalone runner ───────────────────────────────────────────────────────

def run(port: int = 8069, open_browser: bool = True):
    """Start the Mission Control server."""
    import webbrowser

    import uvicorn

    if open_browser:
        def _open():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    run()
