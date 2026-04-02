"""Generate a combined War Room + 13 Moon briefing and save to data/briefings/.

Enhanced V2 — includes:
  - Composite score with full indicator breakdown (raw, scored, weight, description)
  - 5-arm allocation: actual $ per arm vs target %
  - ALL account positions (IBKR, Moomoo, WealthSimple, NDAX, Polymarket, EQ Bank)
  - Unusual Whales section (hot chains, congress trades, dark pool)
  - Polymarket balance
  - 13 Moon doctrine
"""
from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BRIEFINGS_DIR = ROOT / "data" / "briefings"
DASHBOARD_PATH = ROOT / "data" / "briefings" / "dashboard.html"

# Indicator metadata — descriptions and weights matching compute_composite_score()
INDICATOR_META: dict[str, dict[str, Any]] = {
    "oil": {"desc": "Crude oil price — higher = supply disruption / geopolitical crisis", "weight": 0.12},
    "gold": {"desc": "Gold safe-haven demand — higher = flight to safety", "weight": 0.08},
    "vix": {"desc": "CBOE Volatility Index — higher = market fear", "weight": 0.10},
    "hy_spread": {"desc": "High-yield bond spread (bp) — wider = credit stress", "weight": 0.08},
    "bdc_nav": {"desc": "BDC NAV discount % — higher = private credit stress", "weight": 0.07},
    "bdc_nonaccrual": {"desc": "BDC non-accrual rate % — higher = loan defaults rising", "weight": 0.05},
    "defi_tvl": {"desc": "DeFi TVL change % — more negative = crypto deleveraging", "weight": 0.04},
    "stablecoin": {"desc": "Stablecoin depeg % — higher = crypto plumbing breaking", "weight": 0.04},
    "btc": {"desc": "Bitcoin price — lower = crypto risk-off", "weight": 0.05},
    "fed_rate": {"desc": "Fed Funds rate — higher = tighter monetary policy", "weight": 0.05},
    "dxy": {"desc": "US Dollar Index — higher = EM / commodity stress", "weight": 0.05},
    "spy": {"desc": "S&P 500 price — lower = equity stress", "weight": 0.07},
    "x_sentiment": {"desc": "X/Twitter sentiment — lower = social fear (HEAVY BIAS leading indicator)", "weight": 0.12},
    "news": {"desc": "News severity — higher = black swan risk", "weight": 0.04},
    "fear_greed": {"desc": "Crypto Fear & Greed Index — inverted: lower FGI = higher crisis", "weight": 0.04},
}

# Map IndicatorState field names to composite_score keys
_IND_RAW_KEYS: dict[str, str] = {
    "oil": "oil_price",
    "gold": "gold_price",
    "vix": "vix",
    "hy_spread": "hy_spread_bp",
    "bdc_nav": "bdc_nav_discount",
    "bdc_nonaccrual": "bdc_nonaccrual_pct",
    "defi_tvl": "defi_tvl_change_pct",
    "stablecoin": "stablecoin_depeg_pct",
    "btc": "btc_price",
    "fed_rate": "fed_funds_rate",
    "dxy": "dxy",
    "spy": "spy_price",
    "x_sentiment": "x_sentiment",
    "news": "news_severity",
    "fear_greed": "fear_greed_index",
}


def _collect_war_room() -> dict:
    """Run War Room mandate + indicators and return enhanced dict with full details."""
    from strategies.war_room_engine import (
        CURRENT_POSITIONS,
        WarRoomEngine,
        compute_composite_score,
        get_arm_allocations,
        get_portfolio_value_usd,
    )

    engine = WarRoomEngine()
    mandate_obj = engine.get_mandate()
    mandate = dataclasses.asdict(mandate_obj)
    ind = engine.indicators
    comp = compute_composite_score(ind)

    # Build detailed indicator list
    ind_dict = dataclasses.asdict(ind)
    individual_scores = comp.get("individual_scores", {})
    indicators_detailed = []
    for key, meta in INDICATOR_META.items():
        raw_field = _IND_RAW_KEYS.get(key, key)
        raw_val = ind_dict.get(raw_field, None)
        score = individual_scores.get(key, 0)
        indicators_detailed.append({
            "key": key,
            "description": meta["desc"],
            "weight": meta["weight"],
            "raw_value": raw_val,
            "score": round(score, 1),
            "weighted_contribution": round(score * meta["weight"], 2),
        })

    # Arm allocations with actual $ invested per arm
    phase = mandate.get("phase", "accumulation")
    allocations = get_arm_allocations(phase)
    portfolio_usd = get_portfolio_value_usd()

    # Sum actual $ per arm from positions
    arm_actuals: dict[str, float] = {}
    for pos in CURRENT_POSITIONS:
        arm_name = pos.arm.value if hasattr(pos.arm, "value") else str(pos.arm)
        arm_actuals[arm_name] = arm_actuals.get(arm_name, 0) + abs(pos.market_value)

    arms_detailed = []
    for alloc in allocations:
        arm_key = alloc.arm.value if hasattr(alloc.arm, "value") else str(alloc.arm)
        actual_usd = arm_actuals.get(arm_key, 0)
        target_usd = portfolio_usd * alloc.target_pct
        max_usd = portfolio_usd * alloc.max_pct
        actual_pct = (actual_usd / portfolio_usd * 100) if portfolio_usd else 0
        arms_detailed.append({
            "arm": arm_key,
            "name": alloc.name,
            "target_pct": round(alloc.target_pct * 100, 1),
            "max_pct": round(alloc.max_pct * 100, 1),
            "actual_pct": round(actual_pct, 1),
            "target_usd": round(target_usd, 2),
            "actual_usd": round(actual_usd, 2),
            "max_usd": round(max_usd, 2),
            "instruments": alloc.instruments,
            "entry_conditions": alloc.entry_conditions,
            "exit_conditions": alloc.exit_conditions,
        })

    # Live feeds
    live_feed: dict[str, Any] = {}
    try:
        from strategies.war_room_live_feeds import (
            get_last_feed_result,
            update_all_live_data_sync,
        )
        update_all_live_data_sync(ind)
        result = get_last_feed_result()
        if result:
            live_feed = {
                "btc": result.btc_price,
                "eth": result.eth_price,
                "spy": result.spy_price,
                "gold": result.gold_price_oz,
                "vix": ind.vix if ind.vix else None,
                "put_call": result.put_call_ratio,
                "fear_greed": result.fear_greed_value,
            }
    except Exception:
        pass

    return {
        "mandate": mandate,
        "composite_score": comp.get("composite_score", 0),
        "regime": comp.get("regime", "unknown"),
        "confidence": comp.get("confidence", 0),
        "phase": phase,
        "portfolio_value_usd": portfolio_usd,
        "indicators_detailed": indicators_detailed,
        "arms_detailed": arms_detailed,
        "live_feed": {
            "btc": live_feed.get("btc"),
            "eth": live_feed.get("eth"),
            "spy": live_feed.get("spy"),
            "gold": live_feed.get("gold"),
            "vix": live_feed.get("vix"),
            "put_call": live_feed.get("put_call"),
            "fear_greed": live_feed.get("fear_greed"),
        },
    }


def _collect_positions() -> dict:
    """Load all account positions from account_balances.json."""
    bal_path = ROOT / "data" / "account_balances.json"
    if not bal_path.exists():
        return {"accounts": [], "total_usd": 0}

    raw = json.loads(bal_path.read_text(encoding="utf-8"))
    meta = raw.get("_meta", {})
    accounts_raw = raw.get("accounts", {})
    fx = raw.get("fx", {})
    cad_usd = fx.get("cad_usd", 0.72)

    accounts = []
    total_usd = 0.0
    for key, acct in accounts_raw.items():
        currency = acct.get("currency", "USD")
        balance = float(acct.get("balance", 0) or 0)
        total_assets = float(acct.get("total_assets", 0) or 0)
        in_positions_val = float(acct.get("in_positions", 0) or 0)

        value_usd = total_assets if currency == "USD" else total_assets * cad_usd
        total_usd += value_usd

        positions = []
        for pos in (acct.get("positions") or []):
            unrealized = float(pos.get("unrealizedPNL", 0) or 0)
            mkt_val = float(pos.get("marketValue", pos.get("market_val", 0)) or 0)
            avg_cost = float(pos.get("avgCost", pos.get("cost_price", 0)) or 0)
            mkt_price = float(pos.get("marketPrice", 0) or 0)
            qty = float(pos.get("qty", 0) or 0)
            right = pos.get("right", "")
            sec_type = pos.get("secType", "")
            ptype = "call" if right == "C" else "put" if right == "P" else sec_type.lower() or "position"
            positions.append({
                "symbol": pos.get("symbol", "?"),
                "type": ptype,
                "strike": pos.get("strike"),
                "expiry": pos.get("expiry", ""),
                "qty": qty,
                "avg_cost": round(avg_cost, 2),
                "market_price": round(mkt_price, 4),
                "market_value": round(mkt_val, 2),
                "unrealized_pnl": round(unrealized, 2),
            })

        accounts.append({
            "name": key,
            "platform": acct.get("platform", ""),
            "currency": currency,
            "balance": balance,
            "total_assets": total_assets,
            "in_positions": in_positions_val,
            "value_usd": round(value_usd, 2),
            "note": acct.get("note", ""),
            "verified": acct.get("verified", ""),
            "positions": positions,
        })

    return {
        "accounts": accounts,
        "total_usd": round(total_usd, 2),
        "fx_cad_usd": cad_usd,
        "last_updated": meta.get("updated", "unknown"),
        "injection": raw.get("injection", {}),
    }


def _collect_unusual_whales() -> dict:
    """Fetch Unusual Whales data — hot chains, congress trades, dark pool."""
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
        return asyncio.run(_fetch())
    except RuntimeError:
        # Already running event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_fetch())
        finally:
            loop.close()


def _collect_thirteen_moon() -> dict:
    """Query 13 Moon doctrine for current state and upcoming events."""
    from strategies.thirteen_moon_doctrine import ThirteenMoonDoctrine

    d = ThirteenMoonDoctrine()
    moon = d.get_current_moon()
    alerts = d.get_events_with_lead_time(days_ahead=30)

    events = []
    for a in alerts:
        events.append({
            "date": str(a.event_date),
            "name": a.event_name,
            "type": a.event_type,
            "days_until": a.days_until,
            "priority": a.priority,
            "action": a.lead_time_action,
        })

    return {
        "moon_number": moon.moon_number,
        "moon_name": moon.lunar_phase_name,
        "start_date": str(moon.start_date),
        "end_date": str(moon.end_date),
        "fire_peak": str(moon.fire_peak_date) if moon.fire_peak_date else None,
        "mandate": moon.doctrine_action.mandate if moon.doctrine_action else None,
        "mandate_description": moon.doctrine_action.description if moon.doctrine_action else None,
        "conviction": moon.doctrine_action.conviction if moon.doctrine_action else None,
        "targets": moon.doctrine_action.targets if moon.doctrine_action else [],
        "events": events,
    }


def generate_briefing() -> Path:
    """Generate enhanced briefing JSON and return its path."""
    BRIEFINGS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    session = "morning" if now.hour < 14 else "evening"
    stamp = now.strftime("%Y-%m-%d_%H%M")

    print(f"Generating enhanced briefing {stamp} ({session})...")

    print("  Collecting War Room data...")
    war_room = _collect_war_room()
    print("  Collecting positions...")
    positions = _collect_positions()
    print("  Collecting Unusual Whales data...")
    uw = _collect_unusual_whales()
    print("  Collecting 13 Moon data...")
    moon = _collect_thirteen_moon()

    briefing = {
        "id": stamp,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "session": session,
        "war_room": war_room,
        "positions": positions,
        "unusual_whales": uw,
        "thirteen_moon": moon,
    }

    path = BRIEFINGS_DIR / f"briefing_{stamp}.json"
    path.write_text(json.dumps(briefing, indent=2, default=str), encoding="utf-8")
    print(f"  Saved: {path}")

    # Also import any old mandate files from data/war_engine/
    _import_legacy_mandates()

    # Rebuild dashboard
    _build_dashboard()

    return path


def _import_legacy_mandates():
    """Convert old mandate_*.json files into briefing format (once)."""
    war_engine_dir = ROOT / "data" / "war_engine"
    if not war_engine_dir.exists():
        return
    for f in sorted(war_engine_dir.glob("mandate_*.json")):
        stem = f.stem.replace("mandate_", "briefing_")
        target = BRIEFINGS_DIR / f"{stem}.json"
        if target.exists():
            continue
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            ts = raw.get("timestamp", "")
            date_part = ts.split(" ")[0] if " " in ts else f.stem.split("_")[1]
            time_part = ts.split(" ")[1] if " " in ts else "00:00"
            briefing = {
                "id": stem.replace("briefing_", ""),
                "date": date_part,
                "time": time_part,
                "session": raw.get("session", "unknown"),
                "war_room": {
                    "mandate": raw,
                    "indicators_detailed": [],
                    "arms_detailed": [],
                    "composite_score": raw.get("composite_score"),
                    "regime": raw.get("regime", "unknown"),
                    "live_feed": {},
                },
                "positions": None,
                "unusual_whales": None,
                "thirteen_moon": None,
            }
            target.write_text(json.dumps(briefing, indent=2, default=str), encoding="utf-8")
            print(f"  Imported legacy: {f.name} -> {target.name}")
        except Exception as e:
            print(f"  Skip legacy {f.name}: {e}")


def _build_dashboard():
    """Build the static HTML dashboard from all briefing JSON files."""
    briefings = []
    for f in sorted(BRIEFINGS_DIR.glob("briefing_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            briefings.append(data)
        except Exception:
            continue

    briefings_json = json.dumps(briefings, indent=None, default=str)
    html = _DASHBOARD_HTML.replace("__BRIEFINGS_DATA__", briefings_json)
    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    print(f"  Dashboard: {DASHBOARD_PATH}")


_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AAC Briefing Dashboard v2</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --surface2: #1c2129; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149; --orange: #db6d28;
    --gold: #e3b341; --purple: #a371f7; --cyan: #39d3f5;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', -apple-system, sans-serif; line-height: 1.6; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

  .header { display: flex; align-items: center; justify-content: space-between; padding: 16px 0; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
  .header h1 { font-size: 1.5rem; color: var(--gold); letter-spacing: 1px; }
  .header h1 span { color: var(--muted); font-weight: 400; font-size: 0.9rem; margin-left: 8px; }
  .selector { display: flex; align-items: center; gap: 12px; }
  .selector label { color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
  .selector select { background: var(--surface); color: var(--text); border: 1px solid var(--border); padding: 8px 12px; border-radius: 6px; font-size: 0.95rem; cursor: pointer; min-width: 280px; }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .full-width { grid-column: 1 / -1; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }
  .card h2 { font-size: 1.1rem; color: var(--accent); margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
  .card h2 .icon { font-size: 1.3rem; }
  .card h3 { font-size: 0.95rem; color: var(--muted); margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 1px; }

  /* Ticker */
  .ticker { display: flex; flex-wrap: wrap; gap: 16px; }
  .tick { text-align: center; min-width: 90px; }
  .tick .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
  .tick .value { font-size: 1.3rem; font-weight: 700; }

  /* Composite */
  .gauge-container { display: flex; align-items: center; gap: 20px; margin-bottom: 12px; }
  .gauge-bar { flex: 1; height: 28px; background: var(--bg); border-radius: 14px; overflow: hidden; position: relative; }
  .gauge-fill { height: 100%; border-radius: 14px; transition: width 0.6s ease; }
  .gauge-label { font-size: 2.2rem; font-weight: 700; min-width: 80px; text-align: right; }
  .regime-badge { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px; }
  .composite-explain { background: var(--bg); border-radius: 8px; padding: 12px; font-size: 0.85rem; color: var(--muted); margin-top: 10px; }

  /* Indicators table */
  .ind-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .ind-table th { text-align: left; padding: 8px 6px; color: var(--muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid var(--border); }
  .ind-table td { padding: 7px 6px; border-bottom: 1px solid var(--border); }
  .ind-table tr:last-child td { border-bottom: none; }
  .ind-bar-bg { width: 100%; height: 12px; background: var(--bg); border-radius: 6px; overflow: hidden; }
  .ind-bar-fill { height: 100%; border-radius: 6px; }
  .ind-desc { color: var(--muted); font-size: 0.8rem; }

  /* Arms table */
  .arm-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .arm-table th { text-align: left; padding: 8px 6px; color: var(--muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid var(--border); }
  .arm-table td { padding: 8px 6px; border-bottom: 1px solid var(--border); vertical-align: top; }
  .arm-table tr:last-child td { border-bottom: none; }
  .arm-pct-bar { position: relative; height: 20px; background: var(--bg); border-radius: 4px; overflow: hidden; }
  .arm-pct-target { height: 100%; opacity: 0.3; border-radius: 4px; position: absolute; top: 0; left: 0; }
  .arm-pct-actual { height: 100%; border-radius: 4px; position: absolute; top: 0; left: 0; }
  .arm-instruments { font-size: 0.8rem; color: var(--muted); }
  .arm-conditions { font-size: 0.75rem; color: var(--muted); margin-top: 4px; }

  /* Positions */
  .acct-header { display: flex; justify-content: space-between; align-items: baseline; padding: 10px 0; border-bottom: 2px solid var(--border); margin-top: 16px; }
  .acct-header:first-child { margin-top: 0; }
  .acct-name { font-size: 1rem; font-weight: 700; color: var(--accent); }
  .acct-bal { font-size: 0.9rem; color: var(--gold); }
  .pos-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .pos-table th { text-align: left; padding: 6px 5px; color: var(--muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); }
  .pos-table td { padding: 5px; border-bottom: 1px solid var(--border); font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 0.82rem; }
  .pos-table tr:last-child td { border-bottom: none; }
  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red); }

  /* UW section */
  .uw-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .uw-table th { text-align: left; padding: 6px 5px; color: var(--muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; border-bottom: 1px solid var(--border); }
  .uw-table td { padding: 5px; border-bottom: 1px solid var(--border); }
  .uw-table tr:last-child td { border-bottom: none; }

  /* Moon */
  .moon-header { display: flex; align-items: center; gap: 16px; margin-bottom: 16px; }
  .moon-number { font-size: 3rem; font-weight: 800; color: var(--gold); line-height: 1; }
  .moon-meta .moon-name { font-size: 1.3rem; font-weight: 600; }
  .moon-meta .moon-dates { font-size: 0.85rem; color: var(--muted); }
  .mandate-box { background: var(--bg); border: 1px solid var(--gold); border-radius: 8px; padding: 14px; margin-bottom: 16px; }
  .mandate-label { font-size: 0.75rem; color: var(--gold); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px; }
  .conviction-bar { display: inline-block; height: 6px; border-radius: 3px; background: var(--gold); }

  /* Events */
  .event-list { max-height: 500px; overflow-y: auto; }
  .event { display: flex; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--border); }
  .event:last-child { border-bottom: none; }
  .event-date { min-width: 90px; font-size: 0.8rem; color: var(--muted); font-family: monospace; }
  .event-days { min-width: 50px; font-size: 0.8rem; text-align: right; }
  .event-body { flex: 1; }
  .event-name { font-size: 0.9rem; font-weight: 600; }
  .event-action { font-size: 0.8rem; color: var(--muted); }
  .event-type { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; }
  .type-financial { background: rgba(88,166,255,0.15); color: var(--accent); }
  .type-astrology { background: rgba(227,179,65,0.15); color: var(--gold); }
  .type-phi { background: rgba(163,113,247,0.15); color: var(--purple); }
  .type-world { background: rgba(219,109,40,0.15); color: var(--orange); }
  .type-aac { background: rgba(63,185,80,0.15); color: var(--green); }
  .priority-CRITICAL { color: var(--red); font-weight: 700; }
  .priority-HIGH { color: var(--orange); }
  .priority-MEDIUM { color: var(--yellow); }

  /* Alerts & checks */
  .alert { background: rgba(248, 81, 73, 0.1); border-left: 3px solid var(--red); padding: 10px 14px; margin-bottom: 8px; border-radius: 0 6px 6px 0; font-size: 0.9rem; }
  .alert.warning { background: rgba(210, 153, 34, 0.1); border-left-color: var(--yellow); }
  .check-item { padding: 4px 0; font-size: 0.9rem; color: var(--muted); }
  .check-item::before { content: "[ ] "; color: var(--accent); font-family: monospace; }

  /* MC */
  .mc-stats { display: flex; flex-wrap: wrap; gap: 20px; }
  .mc-stat { text-align: center; }
  .mc-stat .val { font-size: 1.4rem; font-weight: 700; }
  .mc-stat .lbl { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }

  .no-data { text-align: center; padding: 40px; color: var(--muted); font-style: italic; }
  .legacy-note { color: var(--muted); font-style: italic; font-size: 0.85rem; padding: 10px 0; }

  /* Portfolio summary bar */
  .portfolio-bar { display: flex; flex-wrap: wrap; gap: 20px; align-items: baseline; }
  .port-total { font-size: 1.8rem; font-weight: 800; color: var(--gold); }
  .port-item { text-align: center; min-width: 100px; }
  .port-item .pval { font-size: 1.1rem; font-weight: 700; }
  .port-item .plbl { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .tab-bar { display: flex; gap: 4px; margin-bottom: 12px; }
  .tab-btn { background: var(--bg); color: var(--muted); border: 1px solid var(--border); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.82rem; }
  .tab-btn.active { background: var(--accent); color: #000; border-color: var(--accent); font-weight: 600; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>AAC BRIEFING DASHBOARD <span>v2 &mdash; Accelerated Arbitrage Corp</span></h1>
    <div class="selector">
      <label>Briefing:</label>
      <select id="briefingSelect" onchange="loadBriefing()"></select>
    </div>
  </div>
  <div id="content"></div>
</div>

<script>
const ALL_BRIEFINGS = __BRIEFINGS_DATA__;

function init() {
  const sel = document.getElementById('briefingSelect');
  ALL_BRIEFINGS.forEach((b, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    const ico = b.session === 'morning' ? '\u2600' : '\uD83C\uDF19';
    opt.textContent = `${b.date} ${b.time} ${ico} ${b.session.toUpperCase()}${i === 0 ? ' (LATEST)' : ''}`;
    sel.appendChild(opt);
  });
  if (ALL_BRIEFINGS.length > 0) loadBriefing();
  else document.getElementById('content').innerHTML = '<div class="no-data">No briefings yet. Run: python scripts/generate_briefing.py</div>';
}

function loadBriefing() {
  const b = ALL_BRIEFINGS[document.getElementById('briefingSelect').value];
  document.getElementById('content').innerHTML = renderBriefing(b);
}

function sc(score) { return score >= 70 ? 'var(--red)' : score >= 50 ? 'var(--orange)' : score >= 30 ? 'var(--yellow)' : 'var(--green)'; }
function rc(r) { const u = (r||'').toUpperCase(); return u === 'CRISIS' ? 'var(--red)' : u === 'ELEVATED' ? 'var(--orange)' : u === 'WATCH' ? 'var(--yellow)' : 'var(--green)'; }
function fmt$(v) { return '$' + Number(v||0).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2}); }
function fmtPct(v) { return Number(v||0).toFixed(1) + '%'; }
function pnlClass(v) { return v >= 0 ? 'pnl-pos' : 'pnl-neg'; }
function pnlSign(v) { return v >= 0 ? '+' + fmt$(v) : '-' + fmt$(Math.abs(v)); }

function regimeExplain(score, regime) {
  if (score >= 70) return 'CRISIS MODE: Full deployment. Maximum conviction on all arms. Market dislocation in progress.';
  if (score >= 50) return 'ELEVATED: Significant stress detected. Scaling into positions. Monitor for regime shift to CRISIS.';
  if (score >= 30) return 'WATCH: Early warning signals active. Prepare positions but wait for confirmation.';
  return 'CALM: Markets stable. Harvest income, minimal new deployment. Preserve capital.';
}

function renderBriefing(b) {
  const wr = b.war_room || {};
  const mandate = wr.mandate || {};
  const lf = wr.live_feed || {};
  const cs = wr.composite_score != null ? wr.composite_score : (wr.composite || mandate.composite_score || 0);
  const regime = wr.regime || mandate.regime || (cs >= 70 ? 'CRISIS' : cs >= 50 ? 'ELEVATED' : cs >= 30 ? 'WATCH' : 'CALM');
  const phase = wr.phase || mandate.phase || 'accumulation';
  const indDetailed = wr.indicators_detailed || [];
  const armsDetailed = wr.arms_detailed || [];
  const positions = b.positions || {};
  const uw = b.unusual_whales || {};
  const moon = b.thirteen_moon;
  const portUsd = wr.portfolio_value_usd || 0;

  let html = '<div class="grid">';

  // ======== LIVE TICKER ========
  const ticks = [
    {l:'BTC',v:lf.btc},{l:'ETH',v:lf.eth},{l:'SPY',v:lf.spy},
    {l:'Gold',v:lf.gold},{l:'VIX',v:lf.vix},{l:'P/C',v:lf.put_call},{l:'FGI',v:lf.fear_greed}
  ];
  if (ticks.some(t => t.v != null)) {
    html += '<div class="card full-width"><h2><span class="icon">&#128225;</span> Live Market Feed</h2><div class="ticker">';
    ticks.forEach(t => {
      if (t.v != null) {
        let v = typeof t.v === 'number' ? (['VIX','P/C','FGI'].includes(t.l) ? t.v.toFixed(1) : fmt$(t.v)) : t.v;
        html += `<div class="tick"><div class="label">${t.l}</div><div class="value">${v}</div></div>`;
      }
    });
    html += '</div></div>';
  }

  // ======== PORTFOLIO BALANCE (header) ========
  const accts = positions.accounts || [];
  const totalUsd = positions.total_usd || portUsd;
  if (accts.length) {
    html += '<div class="card full-width" style="background:linear-gradient(135deg,var(--surface) 60%,rgba(97,175,239,0.06))">';
    html += '<h2 style="font-size:1.4rem;margin-bottom:12px"><span class="icon">&#128176;</span> Portfolio Balance</h2><div class="portfolio-bar">';
    html += `<div class="port-total" style="font-size:2.2rem">${fmt$(totalUsd)}</div>`;
    accts.forEach(a => {
      const v = a.value_usd || 0;
      if (v > 0) html += `<div class="port-item"><div class="pval">${fmt$(v)}</div><div class="plbl">${a.name} (${a.currency})</div></div>`;
    });
    if (positions.fx_cad_usd) html += `<div class="port-item"><div class="pval">${positions.fx_cad_usd}</div><div class="plbl">CAD/USD FX</div></div>`;
    html += '</div>';
    const inj = positions.injection || {};
    if (inj.total) html += `<div style="margin-top:10px;font-size:0.85rem;color:var(--muted)">Capital Injected: ${fmt$(inj.total)} | Last updated: ${positions.last_updated || 'unknown'}</div>`;
    html += '</div>';
  }

  // Indicator display names, canonical order, and fixed colors
  const indName = {oil:'Crude Oil',gold:'Gold',vix:'VIX Volatility',hy_spread:'HY Bond Spread',bdc_nav:'BDC NAV Discount',bdc_nonaccrual:'BDC Non-Accrual Rate',defi_tvl:'DeFi TVL Change',stablecoin:'Stablecoin Depeg',btc:'Bitcoin',fed_rate:'Fed Funds Rate',dxy:'US Dollar (DXY)',spy:'S&P 500',x_sentiment:'X/Twitter Sentiment',news:'News Severity',fear_greed:'Fear & Greed Index'};
  const indOrder = ['fed_rate','dxy','oil','gold','spy','vix','hy_spread','bdc_nav','bdc_nonaccrual','btc','defi_tvl','stablecoin','fear_greed','x_sentiment','news'];
  const indColors = {fed_rate:'#e06c75',dxy:'#d19a66',oil:'var(--red)',gold:'var(--yellow)',spy:'var(--green)',vix:'var(--purple)',hy_spread:'var(--cyan)',bdc_nav:'#c678dd',bdc_nonaccrual:'#be5046',btc:'var(--orange)',defi_tvl:'#98c379',stablecoin:'#56b6c2',fear_greed:'#e5c07b',x_sentiment:'var(--accent)',news:'#61afef'};

  // ======== COMPOSITE SCORE (full-width with factor breakdown) ========
  html += `<div class="card full-width">
    <h2><span class="icon">&#127919;</span> Composite Crisis Score</h2>
    <div style="display:flex;gap:30px;align-items:flex-start;flex-wrap:wrap">
      <div style="flex:0 0 280px">
        <div class="gauge-container" style="flex-direction:column;align-items:center;gap:8px">
          <div class="gauge-label" style="font-size:3rem;color:${sc(cs)}">${Number(cs).toFixed(1)}<span style="font-size:1rem;color:var(--muted)"> / 100</span></div>
          <div class="gauge-bar" style="width:100%"><div class="gauge-fill" style="width:${cs}%;background:${sc(cs)}"></div></div>
        </div>
        <div style="text-align:center;margin:10px 0">
          <span class="regime-badge" style="background:${rc(regime)};color:#000">${regime}</span>
          <span style="margin-left:8px;color:var(--muted);font-size:0.85rem">Phase: ${phase.toUpperCase()}</span>
          ${wr.confidence ? `<br><span style="color:var(--muted);font-size:0.8rem">Confidence: ${(wr.confidence*100).toFixed(0)}%</span>` : ''}
        </div>
        <div class="composite-explain">${regimeExplain(cs, regime)}</div>
        <div style="margin-top:12px;font-size:0.78rem;color:var(--muted)">
          <strong>How it works:</strong> 15 indicators scored 0&ndash;100 (higher = more crisis). Each multiplied by its weight. Sum = composite. Regime thresholds: &lt;30 CALM, 30&ndash;50 WATCH, 50&ndash;70 ELEVATED, &ge;70 CRISIS.
        </div>
      </div>
      <div style="flex:1;min-width:300px">`;

  // Stacked contribution bar + factor rows
  if (indDetailed.length) {
    const orderedInds = indOrder.map(k => indDetailed.find(i => i.key === k)).filter(Boolean);
    const totalContrib = orderedInds.reduce((s,i) => s + (i.weighted_contribution||0), 0);
    const bySizeDesc = [...orderedInds].sort((a,b) => (b.weighted_contribution||0) - (a.weighted_contribution||0));

    html += '<div style="margin-bottom:12px;font-size:0.8rem;color:var(--muted)">CONTRIBUTION BREAKDOWN &mdash; what is driving the score</div>';
    html += '<div style="display:flex;height:22px;border-radius:6px;overflow:hidden;background:var(--bg);margin-bottom:14px">';
    bySizeDesc.forEach(ind => {
      const pct = totalContrib > 0 ? (ind.weighted_contribution / totalContrib * 100) : 0;
      if (pct > 1) {
        html += `<div title="${indName[ind.key]||ind.key}: ${ind.weighted_contribution.toFixed(2)} (${ind.score.toFixed(0)} x ${((ind.weight||0)*100).toFixed(0)}%)" style="width:${pct}%;background:${indColors[ind.key]||'var(--muted)'};transition:width 0.4s"></div>`;
      }
    });
    html += '</div>';

    // Factor rows grouped by category
    const catLabels = {fed_rate:'MACRO',spy:'EQUITIES',hy_spread:'CREDIT',btc:'CRYPTO',x_sentiment:'SENTIMENT'};
    html += `<div style="display:flex;align-items:center;gap:8px;padding:0 0 6px;font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;border-bottom:1px solid var(--border)">
      <div style="width:10px"></div>
      <div style="width:140px">Indicator</div>
      <div style="flex:1;min-width:80px">Score</div>
      <div style="width:35px;text-align:right">Pts</div>
      <div style="width:28px;text-align:right">Wt</div>
      <div style="width:42px;text-align:right">Ctb</div>
      <div style="width:40px;text-align:right">%</div>
      <div style="width:65px;text-align:right">Raw</div>
    </div>`;
    orderedInds.forEach(ind => {
      if (catLabels[ind.key]) {
        html += `<div style="padding:8px 0 4px;font-size:0.7rem;font-weight:700;color:var(--muted);letter-spacing:1px">${catLabels[ind.key]}</div>`;
      }
      const s = ind.score || 0;
      const w = ((ind.weight||0)*100).toFixed(0);
      const rawDisp = ind.raw_value != null ? (typeof ind.raw_value === 'number' ? ind.raw_value.toLocaleString(undefined,{maximumFractionDigits:2}) : ind.raw_value) : '-';
      const contribPct = totalContrib > 0 ? (ind.weighted_contribution / totalContrib * 100).toFixed(1) : '0';
      const barW = Math.min(s, 100);
      html += `<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid var(--border);font-size:0.82rem">
        <div style="width:10px;height:10px;border-radius:2px;background:${indColors[ind.key]||'var(--muted)'};flex-shrink:0"></div>
        <div style="width:140px;font-weight:600;flex-shrink:0" title="${ind.description||''}">${indName[ind.key]||ind.key}</div>
        <div style="flex:1;min-width:80px">
          <div style="height:10px;background:var(--bg);border-radius:5px;overflow:hidden"><div style="height:100%;width:${barW}%;background:${sc(s)};border-radius:5px"></div></div>
        </div>
        <div style="width:35px;text-align:right;color:${sc(s)};font-weight:700">${s.toFixed(0)}</div>
        <div style="width:28px;text-align:right;color:var(--muted);font-size:0.75rem">${w}%</div>
        <div style="width:42px;text-align:right;font-weight:600;color:${sc(s)}">${(ind.weighted_contribution||0).toFixed(2)}</div>
        <div style="width:40px;text-align:right;color:var(--muted);font-size:0.75rem">${contribPct}%</div>
        <div style="width:65px;text-align:right;font-family:monospace;font-size:0.78rem;color:var(--muted)">${rawDisp}</div>
      </div>`;
    });
    html += `<div style="display:flex;align-items:center;gap:8px;padding:6px 0;font-size:0.82rem;font-weight:700;border-top:2px solid var(--border)">
      <div style="width:10px"></div>
      <div style="width:140px">TOTAL</div>
      <div style="flex:1"></div>
      <div style="width:35px"></div>
      <div style="width:28px;text-align:right;color:var(--muted)">100%</div>
      <div style="width:42px;text-align:right;color:${sc(cs)}">${totalContrib.toFixed(2)}</div>
      <div style="width:40px"></div>
      <div style="width:65px"></div>
    </div>`;
  }
  html += '</div></div></div>';


  // ======== MONTE CARLO ========
  const mcs = mandate.mc_summary || '';
  if (mcs) {
    const mc = {};
    const re = /(\w+)\s+\$([\d,]+)/g; let m;
    while ((m = re.exec(mcs)) !== null) mc[m[1].toLowerCase()] = m[2];
    if (Object.keys(mc).length) {
      html += '<div class="card"><h2><span class="icon">&#127922;</span> Monte Carlo (90d)</h2><div class="mc-stats">';
      if (mc.mean) html += `<div class="mc-stat"><div class="val" style="color:var(--accent)">$${mc.mean}</div><div class="lbl">Mean</div></div>`;
      if (mc.p5) html += `<div class="mc-stat"><div class="val" style="color:var(--red)">$${mc.p5}</div><div class="lbl">P5 Bear</div></div>`;
      if (mc.p95) html += `<div class="mc-stat"><div class="val" style="color:var(--green)">$${mc.p95}</div><div class="lbl">P95 Bull</div></div>`;
      if (mc.var95) html += `<div class="mc-stat"><div class="val" style="color:var(--yellow)">$${mc.var95}</div><div class="lbl">VaR 95</div></div>`;
      html += '</div></div>';
    }
  }

  // ======== 5-ARM ALLOCATION ========
  if (armsDetailed.length) {
    html += '<div class="card full-width"><h2><span class="icon">&#129425;</span> 5-Arm Allocation &mdash; Actual vs Target</h2>';
    html += '<table class="arm-table"><thead><tr><th>Arm</th><th>Target</th><th>Actual</th><th>Max</th><th style="min-width:200px">Allocation</th><th>$ Actual</th><th>$ Target</th><th>Instruments</th></tr></thead><tbody>';
    armsDetailed.forEach(a => {
      const tPct = a.target_pct || 0;
      const aPct = a.actual_pct || 0;
      const mPct = a.max_pct || 0;
      const overMax = aPct > mPct;
      const barColor = overMax ? 'var(--red)' : aPct >= tPct * 0.8 ? 'var(--green)' : 'var(--yellow)';
      html += `<tr>
        <td><strong>${a.name}</strong><div class="arm-conditions">IN: ${a.entry_conditions || '-'}<br>OUT: ${a.exit_conditions || '-'}</div></td>
        <td>${fmtPct(tPct)}</td>
        <td style="color:${barColor};font-weight:700">${fmtPct(aPct)}</td>
        <td>${fmtPct(mPct)}</td>
        <td><div class="arm-pct-bar">
          <div class="arm-pct-target" style="width:${mPct}%;background:var(--muted)"></div>
          <div class="arm-pct-actual" style="width:${Math.min(aPct, 100)}%;background:${barColor}"></div>
        </div></td>
        <td style="font-weight:600">${fmt$(a.actual_usd)}</td>
        <td style="color:var(--muted)">${fmt$(a.target_usd)}</td>
        <td class="arm-instruments">${(a.instruments||[]).join(', ')}</td>
      </tr>`;
    });
    html += '</tbody></table></div>';
  }

  // ======== 15-INDICATOR MODEL (detailed table) ========
  if (indDetailed.length) {
    const orderedForTable = indOrder.map(k => indDetailed.find(i => i.key === k)).filter(Boolean);
    const catLabelsTable = {fed_rate:'MACRO',spy:'EQUITIES',hy_spread:'CREDIT',btc:'CRYPTO',x_sentiment:'SENTIMENT'};
    html += '<div class="card full-width"><h2><span class="icon">&#128202;</span> 15-Indicator Crisis Model &mdash; Detail</h2>';
    html += '<table class="ind-table"><thead><tr><th>Indicator</th><th>Description</th><th>Raw</th><th style="min-width:140px">Score (0-100)</th><th>Score</th><th>Weight</th><th>Contrib</th></tr></thead><tbody>';
    orderedForTable.forEach(ind => {
      const s = ind.score || 0;
      const w = ((ind.weight||0)*100).toFixed(0);
      const rawDisp = ind.raw_value != null ? (typeof ind.raw_value === 'number' ? ind.raw_value.toLocaleString(undefined,{maximumFractionDigits:2}) : ind.raw_value) : '-';
      const label = indName[ind.key] || ind.key;
      const catMark = catLabelsTable[ind.key] ? `<tr><td colspan="7" style="padding:10px 0 4px;font-size:0.7rem;font-weight:700;color:var(--muted);letter-spacing:1px;border-bottom:none">${catLabelsTable[ind.key]}</td></tr>` : '';
      html += catMark;
      html += `<tr>
        <td><strong>${label}</strong></td>
        <td class="ind-desc">${ind.description || ''}</td>
        <td style="font-family:monospace">${rawDisp}</td>
        <td><div class="ind-bar-bg"><div class="ind-bar-fill" style="width:${Math.min(s,100)}%;background:${sc(s)}"></div></div></td>
        <td style="color:${sc(s)};font-weight:700;text-align:right">${s.toFixed(1)}</td>
        <td style="text-align:right">${w}%</td>
        <td style="text-align:right;font-weight:600;color:${sc(s)}">${(ind.weighted_contribution||0).toFixed(2)}</td>
      </tr>`;
    });
    html += '</tbody></table></div>';
  }

  // ======== RISK ALERTS ========
  const alerts = mandate.risk_alerts || [];
  if (alerts.length) {
    html += '<div class="card"><h2><span class="icon">&#9888;</span> Risk Alerts</h2>';
    alerts.forEach(a => {
      const cls = (a.includes('DRAWDOWN') || a.includes('CRISIS')) ? 'alert' : 'alert warning';
      html += `<div class="${cls}">${a}</div>`;
    });
    html += '</div>';
  }

  // ======== CHECKLIST ========
  const checks = mandate.checklist || [];
  if (checks.length) {
    const sessLabel = (mandate.session||"").charAt(0).toUpperCase() + (mandate.session||"").slice(1);
    html += `<div class="card"><h2><span class="icon">&#9989;</span> ${sessLabel} Checklist</h2>`;
    checks.forEach(c => { html += `<div class="check-item">${c.replace(/^\[ \] /, '')}</div>`; });
    html += '</div>';
  }

  // ======== POSITIONS BY ACCOUNT ========
  if (accts.length) {
    html += '<div class="card full-width"><h2><span class="icon">&#128188;</span> All Positions by Account</h2>';
    accts.forEach(a => {
      const ps = a.positions || [];
      html += `<div class="acct-header"><span class="acct-name">${a.name.toUpperCase()} &mdash; ${a.platform || ''}</span>`;
      html += `<span class="acct-bal">${a.currency} ${fmt$(a.total_assets || a.balance)} (${fmt$(a.value_usd)} USD)</span></div>`;
      if (a.note) html += `<div style="font-size:0.8rem;color:var(--muted);padding:4px 0">${a.note}</div>`;
      if (ps.length) {
        html += '<table class="pos-table"><thead><tr><th>Symbol</th><th>Type</th><th>Strike</th><th>Expiry</th><th>Qty</th><th>Avg Cost</th><th>Mkt Price</th><th>Mkt Value</th><th>P&L</th></tr></thead><tbody>';
        ps.forEach(p => {
          const pnl = p.unrealized_pnl || 0;
          html += `<tr>
            <td><strong>${p.symbol}</strong></td>
            <td>${p.type || '-'}</td>
            <td>${p.strike ? '$'+Number(p.strike).toFixed(1) : '-'}</td>
            <td>${p.expiry || '-'}</td>
            <td>${p.qty}</td>
            <td>${fmt$(p.avg_cost)}</td>
            <td>${fmt$(p.market_price)}</td>
            <td>${fmt$(p.market_value)}</td>
            <td class="${pnlClass(pnl)}">${pnlSign(pnl)}</td>
          </tr>`;
        });
        html += '</tbody></table>';
      } else {
        html += '<div style="padding:8px 0;color:var(--muted);font-size:0.85rem">No option positions &mdash; balance only</div>';
      }
    });
    html += '</div>';
  }

  // ======== UNUSUAL WHALES ========
  if (uw && !uw.error) {
    html += '<div class="card full-width"><h2><span class="icon">&#128011;</span> Unusual Whales Intelligence</h2>';
    html += '<div class="tab-bar">';
    html += '<button class="tab-btn active" onclick="switchTab(event,\'uw-hot\')">Hot Chains</button>';
    html += '<button class="tab-btn" onclick="switchTab(event,\'uw-congress\')">Congress</button>';
    html += '<button class="tab-btn" onclick="switchTab(event,\'uw-darkpool\')">Dark Pool</button>';
    html += '<button class="tab-btn" onclick="switchTab(event,\'uw-flow\')">Flow Summary</button>';
    html += '</div>';

    // Hot chains
    const hot = uw.hottest_chains || [];
    html += '<div class="tab-content active" id="uw-hot">';
    if (hot.length) {
      html += '<table class="uw-table"><thead><tr><th>Ticker</th><th>Stock Price</th><th>Volume</th><th>OI</th><th>Premium</th><th>Sweeps</th><th>Bid/Ask Side</th><th>Sector</th></tr></thead><tbody>';
      hot.forEach(h => {
        const tkr = h.ticker_symbol || h.ticker || h.symbol || '?';
        const prem = h.premium ? fmt$(h.premium) : '-';
        const bidPct = h.bid_side_perc_7_day ? Number(h.bid_side_perc_7_day).toFixed(0)+'%' : '-';
        const askPct = h.ask_side_perc_7_day ? Number(h.ask_side_perc_7_day).toFixed(0)+'%' : '-';
        const sweeps = h.sweep_volume ? Number(h.sweep_volume).toLocaleString() : '-';
        html += `<tr>
          <td><strong>${tkr}</strong></td>
          <td>${h.stock_price ? fmt$(h.stock_price) : '-'}</td>
          <td>${Number(h.volume||0).toLocaleString()}</td>
          <td>${Number(h.open_interest||0).toLocaleString()}</td>
          <td>${prem}</td>
          <td>${sweeps}</td>
          <td style="font-size:0.78rem">Bid ${bidPct} / Ask ${askPct}</td>
          <td style="font-size:0.78rem">${h.sector || '-'}</td>
        </tr>`;
      });
      html += '</tbody></table>';
    } else { html += '<div class="no-data">No hot chains data</div>'; }
    html += '</div>';

    // Congress
    const cong = uw.congress_trades || [];
    html += '<div class="tab-content" id="uw-congress">';
    if (cong.length) {
      html += '<table class="uw-table"><thead><tr><th>Politician</th><th>Ticker</th><th>Type</th><th>Amount</th><th>Filed</th><th>Chamber</th></tr></thead><tbody>';
      cong.forEach(c => {
        const txType = c.txn_type || c.transaction_type || c.type || '-';
        const amt = c.amounts || c.amount || c.value || '-';
        const chamber = c.member_type ? c.member_type.charAt(0).toUpperCase() + c.member_type.slice(1) : '-';
        html += `<tr>
          <td><strong>${c.name || c.politician || c.representative || '?'}</strong></td>
          <td>${c.ticker || c.symbol || '-'}</td>
          <td style="color:${txType.toLowerCase().includes('sell') || txType.toLowerCase().includes('sale') ? 'var(--red)' : 'var(--green)'}">${txType}</td>
          <td>${amt}</td>
          <td>${c.filed_at_date || c.transaction_date || c.date || '-'}</td>
          <td>${chamber}</td>
        </tr>`;
      });
      html += '</tbody></table>';
    } else { html += '<div class="no-data">No congress trades data</div>'; }
    html += '</div>';

    // Dark Pool
    const dp = uw.dark_pool_spy || [];
    html += '<div class="tab-content" id="uw-darkpool">';
    if (dp.length) {
      html += '<table class="uw-table"><thead><tr><th>Ticker</th><th>Price</th><th>Size</th><th>Notional</th><th>Exchange</th><th>Time</th></tr></thead><tbody>';
      dp.forEach(d => {
        html += `<tr>
          <td><strong>${d.ticker || d.symbol || 'SPY'}</strong></td>
          <td>${d.price ? fmt$(d.price) : '-'}</td>
          <td>${Number(d.size||d.volume||0).toLocaleString()}</td>
          <td>${d.notional_value ? fmt$(d.notional_value) : (d.notional ? fmt$(d.notional) : '-')}</td>
          <td>${d.exchange || d.market_center || '-'}</td>
          <td>${d.executed_at || d.timestamp || d.time || '-'}</td>
        </tr>`;
      });
      html += '</tbody></table>';
    } else { html += '<div class="no-data">No dark pool data</div>'; }
    html += '</div>';

    // Flow summary — table view
    html += '<div class="tab-content" id="uw-flow">';
    const fs = uw.flow_summary;
    if (fs && typeof fs === 'object' && fs.data && fs.data.length) {
      html += '<table class="uw-table"><thead><tr><th>Ticker</th><th>Strike</th><th>Expiry</th><th>Type</th><th>Premium</th><th>Volume</th><th>OI</th><th>V/OI</th><th>Alert</th></tr></thead><tbody>';
      fs.data.slice(0, 25).forEach(f => {
        const prem = f.total_premium ? fmt$(f.total_premium) : '-';
        const voi = f.volume_oi_ratio ? Number(f.volume_oi_ratio).toFixed(1)+'x' : '-';
        html += `<tr>
          <td><strong>${f.ticker || '?'}</strong></td>
          <td>${f.strike || '-'}</td>
          <td>${f.expiry || '-'}</td>
          <td>${f.type || '-'}</td>
          <td>${prem}</td>
          <td>${Number(f.volume||0).toLocaleString()}</td>
          <td>${Number(f.open_interest||0).toLocaleString()}</td>
          <td>${voi}</td>
          <td style="font-size:0.75rem">${f.alert_rule || '-'}</td>
        </tr>`;
      });
      html += '</tbody></table>';
      if (fs.data.length > 25) html += `<div style="padding:8px 0;font-size:0.8rem;color:var(--muted)">Showing 25 of ${fs.data.length} flow alerts</div>`;
    } else if (fs) {
      html += '<pre style="background:var(--bg);padding:12px;border-radius:6px;font-size:0.82rem;overflow-x:auto;color:var(--text)">' + JSON.stringify(fs, null, 2) + '</pre>';
    } else {
      html += '<div class="no-data">No flow summary</div>';
    }
    html += '</div>';

    if (uw.flow_summary_error) html += `<div class="alert warning">Flow summary error: ${uw.flow_summary_error}</div>`;
    if (uw.hottest_chains_error) html += `<div class="alert warning">Hot chains error: ${uw.hottest_chains_error}</div>`;
    if (uw.congress_trades_error) html += `<div class="alert warning">Congress error: ${uw.congress_trades_error}</div>`;
    if (uw.dark_pool_error) html += `<div class="alert warning">Dark pool error: ${uw.dark_pool_error}</div>`;
    html += '</div>';
  }

  // ======== 13 MOON ========
  if (moon) {
    html += `<div class="card full-width">
      <h2><span class="icon">&#127769;</span> Thirteen Moon Doctrine</h2>
      <div class="moon-header">
        <div class="moon-number">${moon.moon_number}</div>
        <div class="moon-meta">
          <div class="moon-name">${moon.moon_name}</div>
          <div class="moon-dates">${moon.start_date} &#8594; ${moon.end_date}${moon.fire_peak ? ' | Fire Peak: ' + moon.fire_peak : ''}</div>
        </div>
      </div>`;
    if (moon.mandate) {
      html += `<div class="mandate-box">
        <div class="mandate-label">Mandate: ${moon.mandate}</div>
        <div style="margin-top:6px;font-size:0.9rem">${moon.mandate_description || ''}</div>
        ${moon.conviction ? `<div style="margin-top:8px"><span style="color:var(--muted);font-size:0.75rem">Conviction: ${(moon.conviction * 100).toFixed(0)}%</span> <div class="conviction-bar" style="width:${moon.conviction * 100}px"></div></div>` : ''}
        ${moon.targets && moon.targets.length ? `<div style="margin-top:6px;font-size:0.8rem;color:var(--muted)">Targets: ${moon.targets.join(', ')}</div>` : ''}
      </div>`;
    }
    const events = moon.events || [];
    if (events.length) {
      html += '<h3>Upcoming Events (30d)</h3><div class="event-list">';
      events.forEach(e => {
        const tc = 'type-' + (e.type || 'aac');
        const pc = 'priority-' + (e.priority || 'MEDIUM');
        const dl = e.days_until === 0 ? '<span style="color:var(--red);font-weight:700">TODAY</span>' : e.days_until + 'd';
        html += `<div class="event">
          <div class="event-date">${e.date}</div>
          <div class="event-days">${dl}</div>
          <div class="event-body">
            <div class="event-name ${pc}">${e.name} <span class="event-type ${tc}">${e.type}</span></div>
            <div class="event-action">${e.action || ''}</div>
          </div>
        </div>`;
      });
      html += '</div>';
    }
    html += '</div>';
  } else {
    html += '<div class="card full-width"><div class="legacy-note">13 Moon data not available for this briefing (legacy mandate).</div></div>';
  }

  html += '</div>';
  return html;
}

function switchTab(evt, tabId) {
  const card = evt.target.closest('.card');
  card.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  card.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  evt.target.classList.add('active');
  card.querySelector('#' + tabId).classList.add('active');
}

init();
</script>
</body>
</html>
""".lstrip()


if __name__ == "__main__":
    path = generate_briefing()
    print(f"\nDone. Open dashboard: {DASHBOARD_PATH}")
