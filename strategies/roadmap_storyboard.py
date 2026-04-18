"""
AAC Command Roadmap — Unified Daily/Weekly/Moon Dashboard
==========================================================
Self-contained HTML export that merges:
  - Daily tasks (time-slot organized, from DailyTaskAggregator)
  - Weekly calendar (7-day horizon with events)
  - 13-Moon roadmap (all 14 cycles, current position, mandates)
  - War Room status (indicators, scenarios, milestones)

Generates a single HTML file with no external dependencies.
Dark theme, localStorage task completion, tab navigation.

Usage:
    from strategies.roadmap_storyboard import export_roadmap
    path = export_roadmap()  # -> data/storyboard/aac_roadmap.html

    python -m strategies.roadmap_storyboard       # CLI export + open
    python launch.py roadmap                       # via launcher
"""
from __future__ import annotations

import glob
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = "data/storyboard/aac_roadmap.html"

# ── Live ticker configuration ────────────────────────────────────────────
_TICKER_MAP: dict[str, str] = {
    "SPY": "SPY",
    "BTC": "BTC-USD",
    "GOLD": "GLD",
    "OIL": "CL=F",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
}
_TICKER_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "storyboard" / "ticker_cache.json"
_TICKER_CACHE_TTL = 300  # seconds


def _load_ticker_cache() -> dict[str, Any] | None:
    """Return cached ticker data if still fresh, else None."""
    try:
        if _TICKER_CACHE_PATH.exists():
            with open(_TICKER_CACHE_PATH, "r", encoding="utf-8") as fh:
                cache = json.load(fh)
            if time.time() - cache.get("ts", 0) < _TICKER_CACHE_TTL:
                return cache.get("tickers")
    except (json.JSONDecodeError, OSError, ValueError):
        pass
    return None


def _save_ticker_cache(tickers: dict[str, Any]) -> None:
    """Persist ticker snapshot to disk."""
    try:
        _TICKER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_TICKER_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump({"ts": time.time(), "tickers": tickers}, fh)
    except OSError:
        pass


def _fetch_live_tickers() -> dict[str, dict[str, Any]]:
    """Fetch live prices + 5-day sparkline from yfinance. Returns per-ticker dicts."""
    try:
        import yfinance as yf  # noqa: E402
    except ImportError:
        logger.debug("yfinance not installed — skipping live tickers")
        return {}

    tickers: dict[str, dict[str, Any]] = {}
    for display, symbol in _TICKER_MAP.items():
        try:
            t = yf.Ticker(symbol)
            fi = t.fast_info
            price = float(getattr(fi, "last_price", 0) or 0)
            prev = float(getattr(fi, "previous_close", 0) or 0)
            change = round(price - prev, 2) if price and prev else 0.0
            change_pct = round(change / prev * 100, 2) if prev else 0.0
            # 5-day close for sparkline
            hist = t.history(period="5d")
            sparkline = (
                [round(float(v), 2) for v in hist["Close"].dropna().tolist()]
                if not hist.empty
                else []
            )
            tickers[display] = {
                "price": round(price, 2),
                "prev_close": round(prev, 2),
                "change": change,
                "change_pct": change_pct,
                "sparkline": sparkline[-10:],
            }
        except Exception as exc:
            logger.debug("ticker_fetch_fail: %s — %s", display, exc)
            tickers[display] = {
                "price": 0, "prev_close": 0, "change": 0,
                "change_pct": 0, "sparkline": [],
            }
    return tickers


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def export_roadmap(output_path: str = DEFAULT_OUTPUT) -> str:
    """Export the unified roadmap dashboard as a self-contained HTML file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = _collect_all_data()
    data_json = json.dumps(data, indent=2, default=str, ensure_ascii=True)

    html = _TEMPLATE.replace("/*__ROADMAP_DATA__*/", f"const DATA = {data_json};")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Roadmap dashboard exported to %s", output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

def _collect_all_data() -> dict[str, Any]:
    """Gather data from all AAC subsystems for the roadmap."""
    today = date.today()
    result: dict[str, Any] = {
        "generated": today.isoformat(),
        "today": today.isoformat(),
        "day_name": today.strftime("%A"),
    }

    # ── Portfolio / Accounts ────────────────────────────────────────
    result["portfolio"] = _collect_portfolio()

    # ── 13-Moon Doctrine ────────────────────────────────────────────
    result["moon"] = _collect_moon_data(today)

    # ── Daily Tasks ─────────────────────────────────────────────────
    result["daily"] = _collect_daily_tasks(today)

    # ── Weekly Calendar ─────────────────────────────────────────────
    result["weekly"] = _collect_weekly_calendar(today)

    # ── War Room ────────────────────────────────────────────────────
    result["war_room"] = _collect_war_room(today)

    return result


def _collect_portfolio() -> dict[str, Any]:
    """Collect paper-trading account balances, positions, and key market tickers."""
    portfolio: dict[str, Any] = {
        "accounts": [],
        "total_balance": 0.0,
        "total_equity": 0.0,
        "total_pnl": 0.0,
        "total_positions": 0,
        "tickers": {},
    }

    # ── Load paper-trading state files ───────────────────────────────
    data_dir = Path(__file__).resolve().parent.parent / "data" / "paper_trading"
    for fpath in sorted(glob.glob(str(data_dir / "*.json"))):
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            acct_id = raw.get("account_id", Path(fpath).stem)
            balance = float(raw.get("balance", 0))
            equity = float(raw.get("equity", 0))
            pnl = float(raw.get("total_pnl", 0))
            positions = raw.get("positions", {})
            portfolio["accounts"].append({
                "id": acct_id,
                "balance": round(balance, 2),
                "equity": round(equity, 2),
                "pnl": round(pnl, 2),
                "positions": len(positions),
                "top_holdings": list(positions.keys())[:5],
            })
            portfolio["total_balance"] += balance
            portfolio["total_equity"] += equity
            portfolio["total_pnl"] += pnl
            portfolio["total_positions"] += len(positions)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.debug("portfolio_read_skip: %s – %s", fpath, exc)

    # If no state files, show defaults from constants
    if not portfolio["accounts"]:
        try:
            from shared.constants import PAPER_INITIAL_BALANCE
            default_bal = float(PAPER_INITIAL_BALANCE)
        except ImportError:
            default_bal = 10_000.0
        portfolio["accounts"].append({
            "id": "default",
            "balance": default_bal,
            "equity": default_bal,
            "pnl": 0.0,
            "positions": 0,
            "top_holdings": [],
        })
        portfolio["total_balance"] = default_bal
        portfolio["total_equity"] = default_bal

    portfolio["total_balance"] = round(portfolio["total_balance"], 2)
    portfolio["total_equity"] = round(portfolio["total_equity"], 2)
    portfolio["total_pnl"] = round(portfolio["total_pnl"], 2)

    # ── Live market tickers (yfinance, cached 5 min) ────────────────
    cached = _load_ticker_cache()
    if cached:
        portfolio["tickers"] = cached
    else:
        live = _fetch_live_tickers()
        if live:
            portfolio["tickers"] = live
            _save_ticker_cache(live)
        else:
            # Ultimate fallback — zeros
            portfolio["tickers"] = {
                sym: {"price": 0, "prev_close": 0, "change": 0,
                      "change_pct": 0, "sparkline": []}
                for sym in _TICKER_MAP
            }

    return portfolio


def _collect_moon_data(today: date) -> dict[str, Any]:
    """Collect 13-Moon doctrine state."""
    try:
        from strategies.thirteen_moon_doctrine import (
            MOON_BRIEFINGS,
            SACRED_GEOMETRY_OVERLAY,
            ThirteenMoonDoctrine,
        )

        doctrine = ThirteenMoonDoctrine()
        current = doctrine.get_current_moon(today)

        # All moon cycles for the roadmap timeline
        cycles = []
        for c in doctrine.moon_cycles:
            cycle: dict[str, Any] = {
                "moon": c.moon_number,
                "name": c.lunar_phase_name,
                "start": c.start_date.isoformat(),
                "end": c.end_date.isoformat(),
                "fire_peak": c.fire_peak_date.isoformat() if c.fire_peak_date else None,
                "new_moon": c.new_moon_date.isoformat() if c.new_moon_date else None,
                "is_current": current is not None and c.moon_number == current.moon_number,
                "is_past": c.end_date < today,
                "mandate": None,
                "conviction": 0,
                "event_counts": {
                    "astrology": len(c.astrology_events),
                    "phi": len(c.phi_markers),
                    "financial": len(c.financial_events),
                    "world": len(c.world_events),
                    "aac": len(c.aac_events),
                },
            }
            if c.doctrine_action:
                cycle["mandate"] = c.doctrine_action.mandate
                cycle["conviction"] = c.doctrine_action.conviction
                cycle["mandate_desc"] = c.doctrine_action.description
                cycle["targets"] = c.doctrine_action.targets
            briefing = MOON_BRIEFINGS.get(c.moon_number, {})
            cycle["theme"] = briefing.get("theme", "")
            cycle["market_implication"] = briefing.get("market_implication", "")
            geo = SACRED_GEOMETRY_OVERLAY.get(c.moon_number, {})
            cycle["geometry"] = geo.get("geometry", "")
            cycle["frequency_hz"] = geo.get("frequency_hz", "")
            cycles.append(cycle)

        # Current moon details
        current_info: dict[str, Any] = {}
        if current:
            days_in = (today - current.start_date).days
            total = max((current.end_date - current.start_date).days, 1)
            current_info = {
                "moon_number": current.moon_number,
                "name": current.lunar_phase_name,
                "start": current.start_date.isoformat(),
                "end": current.end_date.isoformat(),
                "days_in": days_in,
                "total_days": total,
                "days_left": (current.end_date - today).days,
                "progress_pct": round(days_in / total * 100, 1),
                "mandate": current.doctrine_action.mandate if current.doctrine_action else "",
                "conviction": current.doctrine_action.conviction if current.doctrine_action else 0,
                "theme": MOON_BRIEFINGS.get(current.moon_number, {}).get("theme", ""),
            }

        # Upcoming alerts (14-day horizon)
        alerts = doctrine.get_events_with_lead_time(days_ahead=14, target=today)
        alert_list = []
        for a in alerts:
            alert_list.append({
                "date": a.event_date.isoformat(),
                "name": a.event_name,
                "type": a.event_type,
                "days_until": a.days_until,
                "priority": a.priority,
                "action": a.lead_time_action,
                "moon": a.moon_number,
            })

        return {
            "cycles": cycles,
            "current": current_info,
            "alerts": alert_list,
            "total_cycles": len(cycles),
        }
    except Exception as e:
        logger.warning("roadmap_moon_error: %s", e)
        return {"cycles": [], "current": {}, "alerts": [], "total_cycles": 0}


def _collect_daily_tasks(today: date) -> dict[str, Any]:
    """Collect daily tasks from the aggregator."""
    try:
        from monitoring.daily_tasks import DailyTaskAggregator

        agg = DailyTaskAggregator(target_date=today, horizon_days=7)
        return agg.collect_all()
    except Exception as e:
        logger.warning("roadmap_daily_error: %s", e)
        return {
            "date": today.isoformat(),
            "day_name": today.strftime("%A"),
            "total_tasks": 0,
            "completed": 0,
            "remaining": 0,
            "by_priority": {},
            "by_source": {},
            "by_slot": {},
            "slots": {},
            "today_tasks": [],
            "upcoming_tasks": [],
            "all_tasks": [],
        }


def _collect_weekly_calendar(today: date) -> dict[str, Any]:
    """Build a 7-day calendar with events from all sources."""
    try:
        from strategies.thirteen_moon_doctrine import ThirteenMoonDoctrine

        doctrine = ThirteenMoonDoctrine()

        days = []
        for offset in range(7):
            d = today + timedelta(days=offset)
            day_info: dict[str, Any] = {
                "date": d.isoformat(),
                "day_name": d.strftime("%a"),
                "day_num": d.day,
                "is_today": offset == 0,
                "events": [],
            }

            # Collect events for this specific day from the doctrine
            alerts = doctrine.get_events_with_lead_time(days_ahead=0, target=d)
            # Also look for events happening ON this day
            for cycle in doctrine.moon_cycles:
                if cycle.start_date <= d <= cycle.end_date:
                    for evt in cycle.astrology_events:
                        if evt.date == d:
                            day_info["events"].append({
                                "name": evt.name, "type": "astrology",
                                "impact": evt.impact, "desc": evt.description,
                            })
                    for fin in cycle.financial_events:
                        if fin.date == d:
                            day_info["events"].append({
                                "name": fin.name, "type": "financial",
                                "impact": fin.impact, "desc": fin.description,
                            })
                    for world in cycle.world_events:
                        if world.date == d:
                            day_info["events"].append({
                                "name": world.name, "type": "world",
                                "impact": world.impact, "desc": world.description,
                            })
                    for aac in cycle.aac_events:
                        if aac.date == d:
                            day_info["events"].append({
                                "name": aac.name, "type": "aac",
                                "impact": aac.impact, "desc": aac.description,
                            })
                    for phi in cycle.phi_markers:
                        if phi.date == d:
                            day_info["events"].append({
                                "name": phi.label, "type": "phi",
                                "impact": "HIGH" if phi.resonance_strength > 0.5 else "MEDIUM",
                                "desc": f"Phi^{phi.phi_power} resonance ({phi.resonance_strength:.2f})",
                            })
                    # Moon transitions
                    if cycle.start_date == d:
                        day_info["events"].append({
                            "name": f"Moon {cycle.moon_number} begins: {cycle.lunar_phase_name}",
                            "type": "moon_transition", "impact": "HIGH",
                            "desc": MOON_BRIEFINGS.get(cycle.moon_number, {}).get("theme", ""),
                        })
                    if cycle.fire_peak_date == d:
                        day_info["events"].append({
                            "name": f"Fire Peak: {cycle.lunar_phase_name}",
                            "type": "fire_peak", "impact": "HIGH",
                            "desc": "Peak lunar energy. Deploy per doctrine mandate.",
                        })
                    break  # Only one cycle can contain a date

            days.append(day_info)

        return {"days": days, "start": today.isoformat(), "end": (today + timedelta(days=6)).isoformat()}
    except Exception as e:
        logger.warning("roadmap_weekly_error: %s", e)
        days = []
        for offset in range(7):
            d = today + timedelta(days=offset)
            days.append({
                "date": d.isoformat(), "day_name": d.strftime("%a"),
                "day_num": d.day, "is_today": offset == 0, "events": [],
            })
        return {"days": days, "start": today.isoformat(), "end": (today + timedelta(days=6)).isoformat()}


# Import MOON_BRIEFINGS at module level for weekly calendar
try:
    from strategies.thirteen_moon_doctrine import MOON_BRIEFINGS
except ImportError:
    MOON_BRIEFINGS = {}


def _collect_war_room(today: date) -> dict[str, Any]:
    """Collect war room state: indicators, scenarios, milestones."""
    war: dict[str, Any] = {
        "composite_score": 0,
        "regime": "UNKNOWN",
        "indicators": [],
        "scenarios": [],
        "milestones": {"total": 50, "achieved": 0, "categories": {}},
        "phase": "accumulation",
    }

    try:
        from strategies.war_room_engine import (
            MILESTONES,
            SCENARIOS,
            IndicatorState,
            compute_composite_score,
            get_current_phase,
            load_milestone_state,
        )

        # Indicators
        state = IndicatorState()
        result = compute_composite_score(state)
        score = result.get("composite_score", 0)
        war["composite_score"] = round(score, 1)

        if score > 70:
            war["regime"] = "CRISIS"
        elif score > 50:
            war["regime"] = "ELEVATED"
        elif score > 30:
            war["regime"] = "WATCH"
        else:
            war["regime"] = "CALM"

        # Indicator details
        individual = result.get("individual_scores", {})
        weights = {
            "oil": 0.12, "gold": 0.08, "vix": 0.10, "hy_spread": 0.08,
            "bdc_nav": 0.07, "bdc_nonaccrual": 0.05, "defi_tvl": 0.04,
            "stablecoin": 0.04, "btc": 0.05, "fed_rate": 0.05,
            "dxy": 0.05, "spy": 0.07, "x_sentiment": 0.12,
            "news": 0.04, "fear_greed": 0.04,
        }
        for name, val in individual.items():
            war["indicators"].append({
                "name": name,
                "value": val,
                "score": round(val * weights.get(name, 0), 1),
                "weight": weights.get(name, 0),
            })

        # Scenarios summary
        for key, sc in SCENARIOS.items():
            war["scenarios"].append({
                "name": sc.get("name", key),
                "probability": sc.get("probability", sc.get("prob", 0)),
                "status": sc.get("status", "WATCH"),
            })

        # Milestones
        try:
            ms_state = load_milestone_state()
            achieved = sum(1 for v in ms_state.values() if v)
            war["milestones"]["achieved"] = achieved
        except Exception:
            pass

        # Phase
        try:
            war["phase"] = get_current_phase()
        except Exception:
            pass

    except ImportError:
        logger.debug("roadmap_war_room_import_unavailable")
    except Exception as e:
        logger.warning("roadmap_war_room_error: %s", e)

    return war


# ═══════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AAC Command Roadmap | Resonance Energy</title>
<style>
:root {
  --bg: #0a0a14;
  --bg2: #12101f;
  --bg3: #1e1a2e;
  --bg4: #2a2444;
  --gold: #c084fc;
  --gold-dim: #7c3aed;
  --silver: #94a3b8;
  --green: #34d399;
  --red: #ef4444;
  --orange: #f97316;
  --blue: #3b82f6;
  --cyan: #06b6d4;
  --pink: #ec4899;
  --yellow: #eab308;
  --text: #e5e7eb;
  --text-dim: #9ca3af;
  --border: #2e2545;
  --critical: #ef4444;
  --high: #f97316;
  --medium: #3b82f6;
  --low: #6b7280;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
  min-height: 100vh;
}

/* ── Hero Banner ─────────────────────────────────── */
.hero {
  background: linear-gradient(135deg, #0a0a14, #1e1040 40%, #2d1b69 60%, #1e1040 80%, #0a0a14);
  padding: 1.5rem 2rem;
  border-bottom: 2px solid var(--gold);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
}
.hero-left { flex: 1; min-width: 280px; }
.hero-left h1 {
  font-size: 1.4rem;
  color: var(--gold);
  letter-spacing: 2px;
  margin-bottom: 0.3rem;
}
.hero-left .date-line {
  color: var(--text-dim);
  font-size: 0.85rem;
}
.hero-center {
  flex: 2;
  text-align: center;
  min-width: 300px;
}
.moon-banner {
  display: inline-block;
  padding: 0.8rem 1.5rem;
  background: var(--bg2);
  border: 1px solid var(--gold-dim);
  border-radius: 10px;
  text-align: center;
}
.moon-banner .moon-name { color: var(--gold); font-size: 1.1rem; font-weight: bold; }
.moon-banner .moon-meta { color: var(--text-dim); font-size: 0.75rem; margin-top: 0.2rem; }
.moon-banner .mandate { font-size: 1rem; font-weight: bold; margin-top: 0.3rem; }
.progress-bar {
  width: 100%;
  height: 6px;
  background: var(--bg3);
  border-radius: 3px;
  margin-top: 0.4rem;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--gold-dim), var(--gold));
  border-radius: 3px;
  transition: width 0.5s;
}
.hero-right {
  flex: 1;
  text-align: right;
  min-width: 200px;
}
.regime-badge {
  display: inline-block;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: bold;
}
.regime-CALM { background: #16a34a22; border: 1px solid #16a34a; color: #4ade80; }
.regime-WATCH { background: #2563eb22; border: 1px solid #2563eb; color: #60a5fa; }
.regime-ELEVATED { background: #d9770622; border: 1px solid #d97706; color: #fbbf24; }
.regime-CRISIS { background: #dc262622; border: 1px solid #dc2626; color: #f87171; }
.score-line { color: var(--text-dim); font-size: 0.8rem; margin-top: 0.3rem; }

/* ── Portfolio Ticker Strip ──────────────────────── */
.ticker-strip {
  display: flex;
  align-items: center;
  gap: 1.2rem;
  padding: 0.5rem 2rem;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  overflow-x: auto;
  flex-wrap: wrap;
}
.ticker-item {
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
  white-space: nowrap;
  font-size: 0.8rem;
}
.ticker-label { color: var(--text-dim); font-weight: 600; }
.ticker-val { color: var(--gold); font-weight: bold; }
.ticker-sep { color: var(--border); font-size: 0.6rem; }
.acct-strip {
  display: flex;
  gap: 0.8rem;
  align-items: center;
}
.acct-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.2rem 0.6rem;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 0.75rem;
}
.acct-chip .acct-id { color: var(--text-dim); }
.acct-chip .acct-bal { color: var(--green); font-weight: bold; }
.acct-chip .acct-pnl-pos { color: var(--green); }
.acct-chip .acct-pnl-neg { color: var(--red); }
.acct-chip .acct-pos { color: var(--text-dim); font-size: 0.7rem; }
.portfolio-total {
  font-size: 0.8rem;
  color: var(--gold);
  font-weight: bold;
  margin-right: 0.5rem;
}
.ticker-change { font-size: 0.72rem; font-weight: 600; margin-left: 0.15rem; }
.change-pos { color: var(--green); }
.change-neg { color: var(--red); }
.sparkline { width: 50px; height: 16px; margin-left: 0.25rem; vertical-align: middle; }

/* ── Tab Navigation ──────────────────────────────── */
.tabs {
  display: flex;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 0 1rem;
}
.tab {
  padding: 0.8rem 1.5rem;
  cursor: pointer;
  color: var(--text-dim);
  border-bottom: 3px solid transparent;
  font-size: 0.85rem;
  font-family: inherit;
  background: none;
  border-top: none;
  border-left: none;
  border-right: none;
  transition: all 0.2s;
  white-space: nowrap;
}
.tab:hover { color: var(--text); background: var(--bg3); }
.tab.active {
  color: var(--gold);
  border-bottom-color: var(--gold);
  background: var(--bg3);
}
.tab .badge {
  display: inline-block;
  background: var(--critical);
  color: #fff;
  font-size: 0.65rem;
  padding: 0.1rem 0.4rem;
  border-radius: 8px;
  margin-left: 0.4rem;
  vertical-align: middle;
}
.tab-content { display: none; padding: 1.5rem; }
.tab-content.active { display: block; }

/* ── TODAY Tab ────────────────────────────────────── */
.time-slot {
  margin-bottom: 1.5rem;
}
.slot-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.5rem;
}
.slot-emoji { font-size: 1.2rem; }
.slot-label { font-weight: bold; color: var(--gold); font-size: 0.9rem; }
.slot-time { color: var(--text-dim); font-size: 0.75rem; }
.task-list { list-style: none; }
.task-item {
  display: flex;
  align-items: flex-start;
  gap: 0.6rem;
  padding: 0.6rem 0.8rem;
  margin: 0.3rem 0;
  background: var(--bg2);
  border-radius: 6px;
  border-left: 3px solid var(--border);
  transition: all 0.2s;
}
.task-item:hover { background: var(--bg3); }
.task-item.completed { opacity: 0.5; }
.task-item.completed .task-name { text-decoration: line-through; }
.task-item.priority-CRITICAL { border-left-color: var(--critical); }
.task-item.priority-HIGH { border-left-color: var(--high); }
.task-item.priority-MEDIUM { border-left-color: var(--medium); }
.task-item.priority-LOW { border-left-color: var(--low); }
.task-check {
  width: 18px;
  height: 18px;
  margin-top: 2px;
  cursor: pointer;
  accent-color: var(--gold);
}
.task-body { flex: 1; }
.task-name { font-size: 0.85rem; color: var(--text); }
.task-desc { font-size: 0.72rem; color: var(--text-dim); margin-top: 0.15rem; }
.task-meta {
  display: flex;
  gap: 0.5rem;
  margin-top: 0.2rem;
  flex-wrap: wrap;
}
.task-tag {
  font-size: 0.65rem;
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
  background: var(--bg3);
  color: var(--text-dim);
}
.tag-source { border: 1px solid var(--gold-dim); color: var(--gold); }
.tag-assets { border: 1px solid var(--cyan); color: var(--cyan); }
.priority-badge {
  font-size: 0.65rem;
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
  font-weight: bold;
}
.pri-CRITICAL { background: var(--critical); color: #fff; }
.pri-HIGH { background: var(--high); color: #fff; }
.pri-MEDIUM { background: var(--medium); color: #fff; }
.pri-LOW { background: var(--low); color: #fff; }

/* Summary cards */
.summary-row {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
}
.summary-card {
  flex: 1;
  min-width: 140px;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
}
.summary-card .value { font-size: 1.6rem; font-weight: bold; color: var(--gold); }
.summary-card .label { font-size: 0.75rem; color: var(--text-dim); margin-top: 0.2rem; }

/* ── THIS WEEK Tab ───────────────────────────────── */
.week-grid {
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}
.day-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.8rem;
  min-height: 120px;
  transition: all 0.2s;
}
.day-card:hover { border-color: var(--gold-dim); }
.day-card.today {
  border-color: var(--gold);
  box-shadow: 0 0 10px rgba(192,132,252,0.2);
}
.day-header {
  text-align: center;
  margin-bottom: 0.5rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
}
.day-name { font-size: 0.75rem; color: var(--text-dim); }
.day-num { font-size: 1.2rem; font-weight: bold; color: var(--text); }
.day-card.today .day-num { color: var(--gold); }
.day-events { }
.day-event {
  font-size: 0.68rem;
  padding: 0.25rem 0.4rem;
  margin: 0.2rem 0;
  border-radius: 4px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  cursor: default;
}
.day-event.type-astrology { background: #7c3aed22; border-left: 2px solid #a78bfa; color: #c4b5fd; }
.day-event.type-financial { background: #05966922; border-left: 2px solid #34d399; color: #6ee7b7; }
.day-event.type-world { background: #ea580c22; border-left: 2px solid #f97316; color: #fdba74; }
.day-event.type-aac { background: #0891b222; border-left: 2px solid #06b6d4; color: #67e8f9; }
.day-event.type-phi { background: #7c3aed22; border-left: 2px solid #818cf8; color: #a5b4fc; }
.day-event.type-moon_transition { background: #c084fc22; border-left: 2px solid #c084fc; color: #e9d5ff; }
.day-event.type-fire_peak { background: #dc262622; border-left: 2px solid #ef4444; color: #fca5a5; }
.day-event-count {
  text-align: center;
  font-size: 0.7rem;
  color: var(--text-dim);
  margin-top: 0.3rem;
}

/* ── Upcoming alerts table ──────────────────────── */
.alert-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 1rem;
}
.alert-table th {
  text-align: left;
  padding: 0.5rem 0.8rem;
  font-size: 0.75rem;
  color: var(--gold);
  border-bottom: 2px solid var(--border);
}
.alert-table td {
  padding: 0.5rem 0.8rem;
  font-size: 0.78rem;
  border-bottom: 1px solid #1e1a2e;
}
.alert-table tr:hover { background: var(--bg3); }
.days-badge {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 10px;
  font-size: 0.7rem;
  font-weight: bold;
  min-width: 40px;
  text-align: center;
}
.days-0 { background: var(--critical); color: #fff; }
.days-1 { background: #dc262699; color: #fca5a5; }
.days-2-3 { background: #d9770699; color: #fde68a; }
.days-4-7 { background: #2563eb66; color: #93c5fd; }
.days-8plus { background: var(--bg3); color: var(--text-dim); }

/* ── ROADMAP Tab ─────────────────────────────────── */
.roadmap-scroll {
  overflow-x: auto;
  padding: 1rem 0 2rem;
}
.roadmap-track {
  display: flex;
  min-width: max-content;
  position: relative;
  padding: 2rem 1rem 1rem;
  align-items: flex-start;
}
.roadmap-track::before {
  content: '';
  position: absolute;
  top: 3.5rem;
  left: 1rem;
  right: 1rem;
  height: 4px;
  background: linear-gradient(90deg, var(--gold-dim) 0%, var(--gold) 50%, var(--gold-dim) 100%);
  border-radius: 2px;
}
.moon-node {
  position: relative;
  width: 180px;
  flex-shrink: 0;
  text-align: center;
  cursor: pointer;
}
.moon-dot {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  margin: 1.6rem auto 0.5rem;
  position: relative;
  z-index: 2;
  border: 2px solid var(--border);
  background: var(--bg2);
  transition: all 0.3s;
}
.moon-node.past .moon-dot { background: var(--gold-dim); border-color: var(--gold-dim); }
.moon-node.current .moon-dot {
  background: var(--gold);
  border-color: var(--gold);
  box-shadow: 0 0 12px rgba(192,132,252,0.5);
  width: 26px;
  height: 26px;
  margin-top: calc(1.6rem - 3px);
}
.moon-node.future .moon-dot { background: var(--bg3); border-color: var(--border); }
.moon-label { font-size: 0.72rem; color: var(--text-dim); margin-top: 0.3rem; }
.moon-node.current .moon-label { color: var(--gold); font-weight: bold; }
.moon-mandate-tag {
  font-size: 0.65rem;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  display: inline-block;
  margin-top: 0.2rem;
}
.mandate-PURIFY { background: #9b59b622; color: #c084fc; }
.mandate-DEPLOY { background: #27ae6022; color: #34d399; }
.mandate-HOLD { background: #3498db22; color: #60a5fa; }
.mandate-EXIT { background: #e74c3c22; color: #f87171; }
.mandate-ROTATE { background: #f39c1222; color: #fbbf24; }
.mandate-REBALANCE { background: #1abc9c22; color: #2dd4bf; }
.mandate-ACCUMULATE { background: #2ecc7122; color: #4ade80; }
.moon-events-count {
  font-size: 0.65rem;
  color: var(--text-dim);
  margin-top: 0.15rem;
}

/* Moon detail panel */
.moon-detail {
  background: var(--bg2);
  border: 1px solid var(--gold-dim);
  border-radius: 8px;
  padding: 1.2rem;
  margin-top: 1rem;
  display: none;
}
.moon-detail.active { display: block; }
.moon-detail h3 { color: var(--gold); font-size: 1rem; margin-bottom: 0.5rem; }
.detail-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}
.detail-section h4 {
  font-size: 0.8rem;
  color: var(--gold);
  margin-bottom: 0.4rem;
}
.detail-section p {
  font-size: 0.78rem;
  color: var(--text-dim);
  line-height: 1.5;
}

/* ── WAR ROOM Tab ────────────────────────────────── */
.indicator-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}
.indicator-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.7rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.ind-name { font-size: 0.78rem; color: var(--text-dim); }
.ind-score { font-size: 0.85rem; font-weight: bold; }

.scenario-list { margin-top: 1rem; }
.scenario-item {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  padding: 0.6rem 0.8rem;
  margin: 0.3rem 0;
  background: var(--bg2);
  border-radius: 6px;
  border-left: 3px solid var(--border);
}
.scenario-prob {
  font-size: 0.9rem;
  font-weight: bold;
  color: var(--gold);
  min-width: 45px;
  text-align: right;
}
.scenario-name { font-size: 0.82rem; flex: 1; }
.scenario-status {
  font-size: 0.7rem;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
}
.status-ACTIVE { background: #dc262633; color: #f87171; }
.status-WATCH { background: #d9770633; color: #fbbf24; }
.status-MONITORING { background: #2563eb33; color: #60a5fa; }

/* Milestones gauge */
.milestone-gauge {
  text-align: center;
  margin: 1.5rem 0;
}
.gauge-ring {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  background: conic-gradient(var(--gold) var(--pct), var(--bg3) var(--pct));
  display: inline-flex;
  align-items: center;
  justify-content: center;
  position: relative;
}
.gauge-inner {
  width: 90px;
  height: 90px;
  border-radius: 50%;
  background: var(--bg);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
}
.gauge-value { font-size: 1.4rem; font-weight: bold; color: var(--gold); }
.gauge-label { font-size: 0.65rem; color: var(--text-dim); }

/* ── Footer ──────────────────────────────────────── */
.footer {
  text-align: center;
  padding: 1rem;
  color: var(--text-dim);
  font-size: 0.7rem;
  border-top: 1px solid var(--border);
}

/* ── Responsive ──────────────────────────────────── */
@media (max-width: 768px) {
  .hero { flex-direction: column; text-align: center; }
  .hero-right { text-align: center; }
  .week-grid { grid-template-columns: repeat(3, 1fr); }
  .detail-grid { grid-template-columns: 1fr; }
  .indicator-grid { grid-template-columns: 1fr 1fr; }
  .tab { padding: 0.6rem 0.8rem; font-size: 0.75rem; }
}
</style>
</head>
<body>

<script>
/*__ROADMAP_DATA__*/
</script>

<script>
// ── Helpers ──────────────────────────────────────────────────
const LS_KEY = 'aac_roadmap_completed';

function getCompleted() {
  try { return JSON.parse(localStorage.getItem(LS_KEY) || '{}'); } catch { return {}; }
}
function setCompleted(id, done) {
  const c = getCompleted();
  if (done) { c[id] = new Date().toISOString(); } else { delete c[id]; }
  localStorage.setItem(LS_KEY, JSON.stringify(c));
}

function priClass(p) { return 'pri-' + (p || 'LOW'); }
function daysBadgeClass(d) {
  if (d === 0) return 'days-0';
  if (d === 1) return 'days-1';
  if (d <= 3) return 'days-2-3';
  if (d <= 7) return 'days-4-7';
  return 'days-8plus';
}

function mandateClass(m) {
  if (!m) return '';
  const word = m.includes('/') ? m.split('/')[0] : m;
  return 'mandate-' + word;
}

function mandateColor(m) {
  const colors = {PURIFY:'#c084fc',DEPLOY:'#34d399',HOLD:'#60a5fa',EXIT:'#f87171',
                  ROTATE:'#fbbf24',REBALANCE:'#2dd4bf',ACCUMULATE:'#4ade80'};
  const word = (m||'').split('/')[0];
  return colors[word] || '#94a3b8';
}

// ── Tab Switching ─────────────────────────────────────────────
function switchTab(tabId) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
  document.getElementById(tabId).classList.add('active');
}

// ── Build UI ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  buildHero();
  buildToday();
  buildWeek();
  buildRoadmap();
  buildWarRoom();
});

function buildHero() {
  const moon = DATA.moon.current || {};
  const wr = DATA.war_room || {};
  const regime = wr.regime || 'UNKNOWN';

  // Left
  document.getElementById('hero-title').textContent = 'AAC COMMAND ROADMAP';
  document.getElementById('hero-date').textContent = `${DATA.day_name}, ${DATA.today}`;

  // Center — Moon banner
  const mb = document.getElementById('moon-banner');
  if (moon.moon_number !== undefined) {
    mb.querySelector('.moon-name').textContent =
      `Moon ${moon.moon_number}: ${moon.name || ''}`;
    mb.querySelector('.moon-meta').textContent =
      `Day ${moon.days_in || 0} of ${moon.total_days || 0} | ${moon.days_left || 0} days left`;
    const mandate = moon.mandate || 'N/A';
    const mandateEl = mb.querySelector('.mandate');
    mandateEl.textContent = mandate + (moon.conviction ? ` (${Math.round(moon.conviction*100)}%)` : '');
    mandateEl.style.color = mandateColor(mandate);
    const fill = document.getElementById('progress-fill');
    fill.style.width = (moon.progress_pct || 0) + '%';
  }

  // Right — Regime
  const rb = document.getElementById('regime-badge');
  rb.textContent = regime;
  rb.className = `regime-badge regime-${regime}`;
  document.getElementById('score-line').textContent =
    `Composite: ${wr.composite_score || 0}/100 | Phase: ${(wr.phase||'').toUpperCase()}`;

  // ── Ticker Strip ──────────────────────────────────────────────
  const pf = DATA.portfolio || {};
  const strip = document.getElementById('ticker-strip');
  let html = '';

  // Account chips
  html += `<span class="portfolio-total">$${fmt(pf.total_balance||0)}</span>`;
  html += '<div class="acct-strip">';
  for (const a of (pf.accounts || [])) {
    const pnlCls = a.pnl >= 0 ? 'acct-pnl-pos' : 'acct-pnl-neg';
    const pnlSign = a.pnl >= 0 ? '+' : '';
    html += `<div class="acct-chip">
      <span class="acct-id">${a.id}</span>
      <span class="acct-bal">$${fmt(a.balance)}</span>
      <span class="${pnlCls}">${pnlSign}${fmt(a.pnl)}</span>
      <span class="acct-pos">${a.positions} pos</span>
    </div>`;
  }
  html += '</div>';

  // Ticker values
  const tickers = pf.tickers || {};
  const tickerOrder = ['SPY','BTC','GOLD','OIL','VIX','DXY'];
  for (const sym of tickerOrder) {
    const t = tickers[sym];
    if (!t) continue;
    const price = typeof t === 'object' ? t.price : t;
    const chg = typeof t === 'object' ? (t.change_pct || 0) : 0;
    const sparkData = typeof t === 'object' ? (t.sparkline || []) : [];
    const isUp = chg >= 0;
    const arrow = chg === 0 ? '' : (isUp ? '\u25B2' : '\u25BC');
    const changeCls = chg === 0 ? '' : (isUp ? 'change-pos' : 'change-neg');
    html += `<span class="ticker-sep">\u2502</span>`;
    html += `<span class="ticker-item">`;
    html += `<span class="ticker-label">${sym}</span> `;
    html += `<span class="ticker-val">${fmtTicker(sym, price)}</span>`;
    if (chg !== 0) {
      html += `<span class="ticker-change ${changeCls}">${arrow}${Math.abs(chg).toFixed(2)}%</span>`;
    }
    html += buildSparkline(sparkData);
    html += `</span>`;
  }
  strip.innerHTML = html;
}

function buildSparkline(data) {
  if (!data || data.length < 2) return '';
  const w = 50, h = 16;
  const mn = Math.min(...data), mx = Math.max(...data);
  const rng = mx - mn || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - mn) / rng) * (h - 2) - 1;
    return x.toFixed(1) + ',' + y.toFixed(1);
  }).join(' ');
  const up = data[data.length - 1] >= data[0];
  const clr = up ? 'var(--green)' : 'var(--red)';
  return `<svg class="sparkline" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline points="${pts}" fill="none" stroke="${clr}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}

function fmt(n) { return Number(n).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2}); }
function fmtTicker(sym, v) {
  if (!v) return '--';
  if (sym === 'BTC') return '$' + Number(v).toLocaleString(undefined, {maximumFractionDigits:0});
  if (sym === 'VIX' || sym === 'DXY') return Number(v).toFixed(1);
  return '$' + Number(v).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2});
}

function buildToday() {
  const daily = DATA.daily || {};
  const slots = daily.slots || {};
  const bySlot = daily.by_slot || {};
  const completed = getCompleted();

  // Summary cards
  const critCount = (daily.by_priority || {}).CRITICAL || 0;
  const highCount = (daily.by_priority || {}).HIGH || 0;
  document.getElementById('sum-total').textContent = daily.total_tasks || 0;
  document.getElementById('sum-critical').textContent = critCount;
  document.getElementById('sum-high').textContent = highCount;
  document.getElementById('sum-done').textContent = daily.completed || 0;

  // Update tab badge
  if (critCount > 0) {
    document.getElementById('today-badge').textContent = critCount;
    document.getElementById('today-badge').style.display = 'inline-block';
  }

  const container = document.getElementById('today-slots');
  container.innerHTML = '';

  const slotOrder = ['pre_market','market_open','mid_day','power_hour','after_hours','overnight'];
  for (const key of slotOrder) {
    const info = slots[key];
    const tasks = bySlot[key] || [];
    if (!info && tasks.length === 0) continue;

    const slotDiv = document.createElement('div');
    slotDiv.className = 'time-slot';

    const emoji = info ? info.emoji : '';
    const label = info ? info.label : key;
    const time = info ? `${info.start} - ${info.end}` : '';

    slotDiv.innerHTML = `
      <div class="slot-header">
        <span class="slot-emoji">${emoji}</span>
        <span class="slot-label">${label}</span>
        <span class="slot-time">${time}</span>
        <span class="slot-time" style="margin-left:auto">${tasks.length} task${tasks.length!==1?'s':''}</span>
      </div>
      <ul class="task-list" id="slot-${key}"></ul>
    `;
    container.appendChild(slotDiv);

    const ul = slotDiv.querySelector(`#slot-${key}`);
    for (const task of tasks) {
      const isDone = completed[task.task_id] || task.completed;
      const li = document.createElement('li');
      li.className = `task-item priority-${task.priority} ${isDone?'completed':''}`;
      li.innerHTML = `
        <input type="checkbox" class="task-check" data-id="${task.task_id}"
               ${isDone?'checked':''} onchange="toggleTask(this)">
        <div class="task-body">
          <div class="task-name">
            <span class="priority-badge ${priClass(task.priority)}">${task.priority}</span>
            ${task.name || ''}
          </div>
          <div class="task-desc">${task.description || ''}</div>
          ${task.action_required ? `<div class="task-desc" style="color:var(--gold)">▸ ${task.action_required}</div>` : ''}
          <div class="task-meta">
            <span class="task-tag tag-source">${task.source || ''}</span>
            ${task.days_until > 0 ? `<span class="task-tag">${task.days_until}d away</span>` : ''}
            ${(task.assets||[]).length ? `<span class="task-tag tag-assets">${task.assets.join(', ')}</span>` : ''}
          </div>
        </div>
      `;
      ul.appendChild(li);
    }
  }

  // If no tasks at all
  if (container.children.length === 0) {
    container.innerHTML = '<p style="color:var(--text-dim);text-align:center;padding:2rem">No tasks for today. Check the WEEK or ROADMAP tabs.</p>';
  }
}

function toggleTask(el) {
  const id = el.dataset.id;
  setCompleted(id, el.checked);
  const item = el.closest('.task-item');
  if (el.checked) { item.classList.add('completed'); } else { item.classList.remove('completed'); }
}

function buildWeek() {
  const weekly = DATA.weekly || {};
  const days = weekly.days || [];
  const grid = document.getElementById('week-grid');
  grid.innerHTML = '';

  for (const day of days) {
    const card = document.createElement('div');
    card.className = `day-card ${day.is_today ? 'today' : ''}`;

    let eventsHtml = '';
    const maxShow = 4;
    const events = day.events || [];
    for (let i = 0; i < Math.min(events.length, maxShow); i++) {
      const e = events[i];
      eventsHtml += `<div class="day-event type-${e.type}" title="${(e.desc||'').replace(/"/g,'&quot;')}">${e.name}</div>`;
    }
    if (events.length > maxShow) {
      eventsHtml += `<div class="day-event-count">+${events.length - maxShow} more</div>`;
    }
    if (events.length === 0) {
      eventsHtml = '<div class="day-event-count" style="margin-top:1rem">No events</div>';
    }

    card.innerHTML = `
      <div class="day-header">
        <div class="day-name">${day.day_name}</div>
        <div class="day-num">${day.day_num}</div>
      </div>
      <div class="day-events">${eventsHtml}</div>
    `;
    grid.appendChild(card);
  }

  // Upcoming alerts table
  buildUpcomingAlerts();
}

function buildUpcomingAlerts() {
  const alerts = (DATA.moon || {}).alerts || [];
  const tbody = document.getElementById('alerts-tbody');
  tbody.innerHTML = '';

  if (alerts.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-dim)">No upcoming alerts</td></tr>';
    return;
  }

  for (const a of alerts) {
    const tr = document.createElement('tr');
    const dClass = daysBadgeClass(a.days_until);
    tr.innerHTML = `
      <td><span class="days-badge ${dClass}">${a.days_until === 0 ? 'TODAY' : a.days_until + 'd'}</span></td>
      <td>${a.date}</td>
      <td><span class="priority-badge ${priClass(a.priority)}">${a.priority}</span> ${a.name}</td>
      <td><span class="task-tag">${a.type}</span></td>
      <td style="font-size:0.72rem;color:var(--text-dim)">${a.action || ''}</td>
    `;
    tbody.appendChild(tr);
  }
}

function buildRoadmap() {
  const moon = DATA.moon || {};
  const cycles = moon.cycles || [];
  const track = document.getElementById('roadmap-track');
  track.innerHTML = '';

  for (const c of cycles) {
    const node = document.createElement('div');
    let cls = 'moon-node';
    if (c.is_current) cls += ' current';
    else if (c.is_past) cls += ' past';
    else cls += ' future';
    node.className = cls;
    node.dataset.moon = c.moon;

    const mandate = c.mandate || '';
    const mandateCls = mandateClass(mandate);
    const totalEvents = Object.values(c.event_counts || {}).reduce((a,b) => a+b, 0);

    node.innerHTML = `
      <div class="moon-dot"></div>
      <div class="moon-label">Moon ${c.moon}</div>
      <div class="moon-label" style="font-size:0.65rem">${c.name || ''}</div>
      ${mandate ? `<span class="moon-mandate-tag ${mandateCls}">${mandate}</span>` : ''}
      <div class="moon-events-count">${totalEvents} events</div>
    `;

    node.addEventListener('click', () => showMoonDetail(c));
    track.appendChild(node);
  }

  // Auto-scroll to current moon
  setTimeout(() => {
    const curr = track.querySelector('.moon-node.current');
    if (curr) {
      curr.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
  }, 200);
}

function showMoonDetail(c) {
  const panel = document.getElementById('moon-detail-panel');
  panel.classList.add('active');

  const mandate = c.mandate || 'N/A';
  const conviction = c.conviction ? Math.round(c.conviction * 100) + '%' : 'N/A';
  const targets = (c.targets || []).join(', ') || 'N/A';

  document.getElementById('md-title').textContent = `Moon ${c.moon}: ${c.name || ''}`;
  document.getElementById('md-title').style.color = mandateColor(mandate);

  document.getElementById('md-dates').textContent = `${c.start} to ${c.end}`;
  document.getElementById('md-mandate').textContent = `${mandate} (${conviction})`;
  document.getElementById('md-mandate').style.color = mandateColor(mandate);
  document.getElementById('md-desc').textContent = c.mandate_desc || '';
  document.getElementById('md-targets').textContent = targets;
  document.getElementById('md-theme').textContent = c.theme || 'N/A';
  document.getElementById('md-market').textContent = c.market_implication || 'N/A';
  document.getElementById('md-geometry').textContent = c.geometry || 'N/A';
  document.getElementById('md-frequency').textContent = c.frequency_hz ? c.frequency_hz + ' Hz' : 'N/A';

  const ec = c.event_counts || {};
  document.getElementById('md-events').textContent =
    `Astrology: ${ec.astrology||0} | Phi: ${ec.phi||0} | Financial: ${ec.financial||0} | World: ${ec.world||0} | AAC: ${ec.aac||0}`;
}

function buildWarRoom() {
  const wr = DATA.war_room || {};

  // Indicators
  const indGrid = document.getElementById('ind-grid');
  indGrid.innerHTML = '';
  const indicators = wr.indicators || [];
  for (const ind of indicators) {
    const scoreColor = ind.score > 7 ? 'var(--critical)' : ind.score > 5 ? 'var(--orange)' : ind.score > 3 ? 'var(--yellow)' : 'var(--green)';
    const card = document.createElement('div');
    card.className = 'indicator-card';
    card.innerHTML = `
      <span class="ind-name">${ind.name}</span>
      <span class="ind-score" style="color:${scoreColor}">${ind.score}</span>
    `;
    indGrid.appendChild(card);
  }
  if (indicators.length === 0) {
    indGrid.innerHTML = '<p style="color:var(--text-dim);padding:1rem">Indicator data unavailable — run live feeds first.</p>';
  }

  // Scenarios
  const scList = document.getElementById('scenario-list');
  scList.innerHTML = '';
  const scenarios = wr.scenarios || [];
  const topScenarios = scenarios.slice(0, 10);
  for (const sc of topScenarios) {
    const item = document.createElement('div');
    item.className = 'scenario-item';
    const status = sc.status || 'WATCH';
    const statusCls = 'status-' + status;
    const prob = typeof sc.probability === 'number' ? sc.probability + '%' : sc.probability || '?';
    item.innerHTML = `
      <span class="scenario-prob">${prob}</span>
      <span class="scenario-name">${sc.name}</span>
      <span class="scenario-status ${statusCls}">${status}</span>
    `;
    scList.appendChild(item);
  }
  if (scenarios.length === 0) {
    scList.innerHTML = '<p style="color:var(--text-dim);padding:1rem">No scenario data available.</p>';
  }

  // Milestones gauge
  const ms = wr.milestones || {};
  const achieved = ms.achieved || 0;
  const total = ms.total || 50;
  const pct = Math.round(achieved / total * 100);
  const gauge = document.getElementById('milestone-gauge');
  gauge.style.setProperty('--pct', pct + '%');
  document.getElementById('ms-value').textContent = `${achieved}/${total}`;
}
</script>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- HERO BANNER                                                      -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<div class="hero">
  <div class="hero-left">
    <h1 id="hero-title">AAC COMMAND ROADMAP</h1>
    <div class="date-line" id="hero-date"></div>
  </div>
  <div class="hero-center">
    <div class="moon-banner" id="moon-banner">
      <div class="moon-name"></div>
      <div class="moon-meta"></div>
      <div class="mandate"></div>
      <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    </div>
  </div>
  <div class="hero-right">
    <div class="regime-badge regime-UNKNOWN" id="regime-badge">UNKNOWN</div>
    <div class="score-line" id="score-line"></div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- PORTFOLIO TICKER STRIP                                            -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<div class="ticker-strip" id="ticker-strip"></div>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- TAB NAVIGATION                                                    -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<div class="tabs">
  <button class="tab active" data-tab="tab-today" onclick="switchTab('tab-today')">
    TODAY <span class="badge" id="today-badge" style="display:none">0</span>
  </button>
  <button class="tab" data-tab="tab-week" onclick="switchTab('tab-week')">THIS WEEK</button>
  <button class="tab" data-tab="tab-roadmap" onclick="switchTab('tab-roadmap')">ROADMAP</button>
  <button class="tab" data-tab="tab-warroom" onclick="switchTab('tab-warroom')">WAR ROOM</button>
</div>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- TAB: TODAY                                                        -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<div id="tab-today" class="tab-content active">
  <div class="summary-row">
    <div class="summary-card">
      <div class="value" id="sum-total">0</div>
      <div class="label">Total Tasks</div>
    </div>
    <div class="summary-card">
      <div class="value" id="sum-critical" style="color:var(--critical)">0</div>
      <div class="label">Critical</div>
    </div>
    <div class="summary-card">
      <div class="value" id="sum-high" style="color:var(--orange)">0</div>
      <div class="label">High Priority</div>
    </div>
    <div class="summary-card">
      <div class="value" id="sum-done" style="color:var(--green)">0</div>
      <div class="label">Completed</div>
    </div>
  </div>
  <div id="today-slots"></div>
</div>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- TAB: THIS WEEK                                                    -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<div id="tab-week" class="tab-content">
  <h2 style="color:var(--gold);margin-bottom:1rem;font-size:1.1rem">7-Day Calendar</h2>
  <div class="week-grid" id="week-grid"></div>

  <h2 style="color:var(--gold);margin-top:2rem;margin-bottom:0.5rem;font-size:1.1rem">Upcoming Alerts (14 days)</h2>
  <table class="alert-table">
    <thead>
      <tr>
        <th>In</th>
        <th>Date</th>
        <th>Event</th>
        <th>Type</th>
        <th>Action</th>
      </tr>
    </thead>
    <tbody id="alerts-tbody"></tbody>
  </table>
</div>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- TAB: ROADMAP                                                      -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<div id="tab-roadmap" class="tab-content">
  <h2 style="color:var(--gold);margin-bottom:0.5rem;font-size:1.1rem">13-Moon Progression</h2>
  <p style="color:var(--text-dim);font-size:0.78rem;margin-bottom:1rem">
    Click any moon to see details. Current position highlighted.
    March 2026 &rarr; April 2027
  </p>
  <div class="roadmap-scroll">
    <div class="roadmap-track" id="roadmap-track"></div>
  </div>

  <div class="moon-detail" id="moon-detail-panel">
    <h3 id="md-title"></h3>
    <div class="detail-grid">
      <div class="detail-section">
        <h4>Dates</h4>
        <p id="md-dates"></p>
        <h4 style="margin-top:0.6rem">Mandate</h4>
        <p id="md-mandate"></p>
        <h4 style="margin-top:0.6rem">Description</h4>
        <p id="md-desc"></p>
        <h4 style="margin-top:0.6rem">Targets</h4>
        <p id="md-targets"></p>
      </div>
      <div class="detail-section">
        <h4>Theme</h4>
        <p id="md-theme"></p>
        <h4 style="margin-top:0.6rem">Market Implication</h4>
        <p id="md-market"></p>
        <h4 style="margin-top:0.6rem">Sacred Geometry</h4>
        <p id="md-geometry"></p>
        <h4 style="margin-top:0.6rem">Frequency</h4>
        <p id="md-frequency"></p>
        <h4 style="margin-top:0.6rem">Events</h4>
        <p id="md-events"></p>
      </div>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════════════════════════════ -->
<!-- TAB: WAR ROOM                                                     -->
<!-- ══════════════════════════════════════════════════════════════════ -->
<div id="tab-warroom" class="tab-content">
  <h2 style="color:var(--gold);margin-bottom:1rem;font-size:1.1rem">15-Indicator Composite</h2>
  <div class="indicator-grid" id="ind-grid"></div>

  <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-top:1.5rem">
    <div style="flex:2;min-width:300px">
      <h2 style="color:var(--gold);margin-bottom:0.5rem;font-size:1.1rem">Scenario Watch</h2>
      <div class="scenario-list" id="scenario-list"></div>
    </div>
    <div style="flex:1;min-width:200px">
      <h2 style="color:var(--gold);margin-bottom:0.5rem;font-size:1.1rem;text-align:center">Milestones</h2>
      <div class="milestone-gauge" id="milestone-gauge" style="--pct:0%">
        <div class="gauge-ring">
          <div class="gauge-inner">
            <div class="gauge-value" id="ms-value">0/50</div>
            <div class="gauge-label">achieved</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="footer">
  AAC Command Roadmap | Resonance Energy | Generated <span id="gen-date"></span>
  <script>document.getElementById('gen-date').textContent = DATA.generated;</script>
</div>

</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import io
    import sys
    import webbrowser

    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

    path = export_roadmap()
    abs_path = os.path.abspath(path)
    print(f"Roadmap exported to: {abs_path}")
    webbrowser.open(f"file:///{abs_path}")
