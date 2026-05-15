from __future__ import annotations

"""DFV daily routines — pre-market brief, midday, EOD, weekend DD.

Each routine is a pure function: takes nothing, returns a dict report.
The daemon calls them on schedule; the CLI calls them on demand.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from agents.dfv import data as dfv_data
from agents.dfv.decision_engine import DFV

_log = structlog.get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
BRIEF_DIR = REPO_ROOT / "agents" / "dfv" / "memory" / "briefs"


def _save_brief(name: str, payload: dict[str, Any]) -> Path:
    BRIEF_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = BRIEF_DIR / f"{ts}_{name}.json"
    import json
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # Best-effort: index into DFV semantic memory so future `ask` calls can recall it.
    try:
        from agents.dfv import rag as dfv_rag  # noqa: PLC0415
        dfv_rag.index_brief(path, payload)
    except Exception as e:  # noqa: BLE001 — daemon must not die on RAG failures
        _log.warning("dfv.routines.index_brief_failed", error=str(e))
    return path


def _safe_collect_payload() -> dict[str, Any]:
    """Pull current dashboard payload; tolerate failures."""
    try:
        from monitoring.mission_control import collect_payload  # type: ignore
        return collect_payload() or {}
    except Exception as e:  # noqa: BLE001 — daemon must not die on collector errors
        _log.warning("dfv.collect_failed", error=str(e))
        return {}


def brief() -> dict[str, Any]:
    """Pre-market / on-demand brief. The first thing DFV produces every morning."""
    dfv = DFV()
    payload = _safe_collect_payload()
    portfolio = payload.get("portfolio", {})
    war_room = payload.get("war_room", {})

    held_symbols = sorted({
        (p.get("symbol") or p.get("underlying") or "").upper()
        for acct in portfolio.get("accounts", {}).values()
        for p in acct.get("positions", [])
        if (p.get("symbol") or p.get("underlying"))
    })
    held_symbols = [s for s in held_symbols if s]

    theses = dfv.thesis.all()
    missing_theses = [s for s in held_symbols if s not in theses]
    stale_theses = dfv.thesis.needs_review(max_age_days=30)
    watchlist = dfv.watchlist.all()

    report = {
        "type": "brief",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "voice": "DFV / Roaring Kitty",
        "portfolio_summary": {
            "total_equity_usd": portfolio.get("total_assets_usd")
            or portfolio.get("total_equity")
            or portfolio.get("total_value"),
            "cash_usd": portfolio.get("cash_usd"),
            "buying_power_usd": portfolio.get("buying_power_usd"),
            "open_positions": len([
                p for acct in portfolio.get("accounts", {}).values()
                for p in acct.get("positions", [])
            ]),
        },
        "war_room": {
            "composite": war_room.get("composite_score"),
            "regime": war_room.get("regime"),
            "phase": war_room.get("phase"),
            "mandate": war_room.get("mandate"),
        },
        "discipline": {
            "held_symbols": held_symbols,
            "missing_thesis": missing_theses,
            "stale_thesis": stale_theses,
            "watchlist_size": len(watchlist),
        },
        "headline": _headline(missing_theses, stale_theses, war_room),
    }
    path = _save_brief("brief", report)
    report["saved_to"] = str(path.relative_to(REPO_ROOT))
    _log.info("dfv.brief", missing=len(missing_theses), stale=len(stale_theses))
    return report


def midday() -> dict[str, Any]:
    """Midday position drift + flow check on holdings."""
    payload = _safe_collect_payload()
    report = {
        "type": "midday",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "war_room_now": payload.get("war_room", {}),
        "alerts": payload.get("alerts", []),
        "note": "Drift check, flow check on held names. No trades — observe.",
    }
    _save_brief("midday", report)
    return report


def eod() -> dict[str, Any]:
    """End-of-day debrief: P&L attribution, conviction nudges, tomorrow's catalysts."""
    dfv = DFV()
    payload = _safe_collect_payload()
    pnl = payload.get("pnl", {}) or {}
    report = {
        "type": "eod",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pnl": {
            "today_realized": pnl.get("today_realized"),
            "mtd_realized": pnl.get("mtd_realized"),
            "ytd_realized": pnl.get("ytd_realized"),
        },
        "thesis_review_needed": dfv.thesis.needs_review(max_age_days=30),
        "decisions_today": len(dfv.decisions.tail(500)),
        "note": "Update theses. Nudge conviction. Write the lesson down.",
    }
    _save_brief("eod", report)
    return report


def weekend_dd() -> dict[str, Any]:
    """Weekend deep DD slot — read 10-Qs, refresh screens, post-mortems."""
    dfv = DFV()
    report = {
        "type": "weekend_dd",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "open_theses": list(dfv.thesis.all().keys()),
        "to_review_this_weekend": dfv.thesis.needs_review(max_age_days=14),
        "note": "Read the filings. Refresh the deep-value and squeeze screens. "
                "Write up any closed positions from the week.",
    }
    _save_brief("weekend_dd", report)
    return report


def _headline(missing: list[str], stale: list[str], war_room: dict[str, Any]) -> str:
    bits: list[str] = []
    regime = (war_room.get("regime") or "").upper() or "?"
    phase = (war_room.get("phase") or "").lower() or "?"
    bits.append(f"Regime {regime} · phase {phase}.")
    if missing:
        bits.append(f"{len(missing)} held name(s) without a thesis: {', '.join(missing[:5])}"
                    + ("…" if len(missing) > 5 else "") + ". Hard rule #1.")
    if stale:
        bits.append(f"{len(stale)} thesis review(s) overdue.")
    if not missing and not stale:
        bits.append("Discipline clean. I like the book.")
    return " ".join(bits)


# ── Sprint 1: distinct cadence routines ────────────────────────────────
# Replace the four stale aliases (asia_digest, open_bell_prep,
# close_debrief, asia_watch) with their own substantive scope.

def _held_symbols() -> list[str]:
    """Pull held symbols from mission_control payload (defensive)."""
    portfolio = _safe_collect_payload().get("portfolio", {})
    syms = sorted({
        (p.get("symbol") or p.get("underlying") or "").upper()
        for acct in portfolio.get("accounts", {}).values()
        for p in acct.get("positions", [])
        if (p.get("symbol") or p.get("underlying"))
    })
    return [s for s in syms if s]


def asia_digest() -> dict[str, Any]:
    """04:00 ET — Asia overnight tape: crypto, dollar, rates, ADR pre-market."""
    crypto = dfv_data.crypto_snapshot()
    macro = dfv_data.macro_snapshot()
    adrs = dfv_data.quotes(dfv_data.ASIA_ADRS)
    news = dfv_data.market_news(category="general", limit=5)

    # Lightweight regime tilt: BTC 24h sign + ADR breadth
    btc = crypto.get("prices", {}).get("BITCOIN/USD", {}).get("change_24h_pct")
    adr_q = adrs.get("quotes", {})
    adr_up = sum(1 for q in adr_q.values() if (q.get("change_pct") or 0) > 0)
    adr_dn = sum(1 for q in adr_q.values() if (q.get("change_pct") or 0) < 0)
    tilt = "risk-on" if (btc and btc > 0 and adr_up > adr_dn) else (
        "risk-off" if (btc is not None and btc < 0 and adr_dn > adr_up) else "mixed"
    )

    report = {
        "type": "asia_digest",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tilt": tilt,
        "crypto": crypto,
        "macro": macro,
        "asia_adrs": adrs,
        "headlines": news.get("headlines", []),
        "headline": (
            f"Asia tape: {tilt}. BTC 24h {btc:+.2f}% · ADR breadth {adr_up}/{adr_up+adr_dn}."
            if btc is not None else
            f"Asia tape: {tilt} (no crypto signal). ADR breadth {adr_up}/{adr_up+adr_dn}."
        ),
    }
    _save_brief("asia_digest", report)
    _log.info("dfv.asia_digest", tilt=tilt, adr_up=adr_up, adr_dn=adr_dn)
    return report


def open_bell_prep() -> dict[str, Any]:
    """09:25 ET — last 5 minutes before the open. Held names, top alerts, war room."""
    payload = _safe_collect_payload()
    held = _held_symbols()
    pre = dfv_data.quotes(held) if held else {"ok": True, "quotes": {}}

    # Largest pre-market movers among held names
    movers = sorted(
        [
            {"symbol": s, "change_pct": q.get("change_pct"), "current": q.get("current")}
            for s, q in pre.get("quotes", {}).items()
            if q.get("change_pct") is not None
        ],
        key=lambda x: abs(x["change_pct"] or 0),
        reverse=True,
    )[:5]

    war_room = payload.get("war_room", {}) or {}
    alerts = (payload.get("alerts", []) or [])[:3]

    report = {
        "type": "open_bell_prep",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "minutes_to_open": "T-5",
        "held_premarket": pre,
        "top_movers_held": movers,
        "war_room": {
            "composite": war_room.get("composite_score"),
            "regime": war_room.get("regime"),
            "phase": war_room.get("phase"),
        },
        "alerts": alerts,
        "headline": (
            f"Open in 5. {len(held)} held names. "
            + (f"Biggest mover: {movers[0]['symbol']} {movers[0]['change_pct']:+.2f}%."
               if movers else "No pre-market movement on the book.")
        ),
    }
    _save_brief("open_bell_prep", report)
    _log.info("dfv.open_bell_prep", held=len(held), alerts=len(alerts))
    return report


def close_debrief() -> dict[str, Any]:
    """17:00 ET — close prints, watchlist movers, stale theses, tomorrow's catalysts."""
    dfv = DFV()
    held = _held_symbols()
    closes = dfv_data.quotes(held) if held else {"ok": True, "quotes": {}}

    watch = list(dfv.watchlist.all().keys())
    watch_quotes = dfv_data.quotes(watch[:25]) if watch else {"ok": True, "quotes": {}}
    watch_movers = sorted(
        [
            {"symbol": s, "change_pct": q.get("change_pct"), "current": q.get("current")}
            for s, q in watch_quotes.get("quotes", {}).items()
            if (q.get("change_pct") is not None)
        ],
        key=lambda x: abs(x["change_pct"] or 0),
        reverse=True,
    )[:5]

    stale = dfv.thesis.needs_review(max_age_days=30)
    next_actions = [
        {"symbol": s, "action": "re-state thesis or close position"} for s in stale
    ]

    earnings = dfv_data.earnings_window(days_ahead=1).get("earnings", [])
    held_set = set(held)
    earnings_for_book = [e for e in earnings if e["symbol"] in held_set]

    report = {
        "type": "close_debrief",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "held_closes": closes,
        "watchlist_movers": watch_movers,
        "stale_theses": stale,
        "next_actions": next_actions,
        "tomorrow_earnings_on_book": earnings_for_book,
        "tomorrow_earnings_universe_count": len(earnings),
        "headline": (
            f"Close: {len(held)} held. "
            + (f"{len(stale)} stale thesis(es) — write the lesson tonight. "
               if stale else "Discipline clean. ")
            + (f"{len(earnings_for_book)} earnings on book tomorrow."
               if earnings_for_book else "No earnings on book tomorrow.")
        ),
    }
    _save_brief("close_debrief", report)
    _log.info("dfv.close_debrief", held=len(held), stale=len(stale))
    return report


def asia_watch() -> dict[str, Any]:
    """22:00 ET — going to bed. Crypto, VIX/dollar, US futures-proxy ETFs."""
    crypto = dfv_data.crypto_snapshot()
    macro = dfv_data.macro_snapshot(series=(dfv_data.VIX_FRED, dfv_data.DXY_FRED))
    futures_proxy = dfv_data.quotes(dfv_data.US_FUTURES_PROXIES)

    btc_chg = crypto.get("prices", {}).get("BITCOIN/USD", {}).get("change_24h_pct")
    spy_chg = futures_proxy.get("quotes", {}).get("SPY", {}).get("change_pct")
    vix_obs = (macro.get("series", {}) or {}).get(dfv_data.VIX_FRED)
    vix_val = vix_obs["value"] if isinstance(vix_obs, dict) else None

    risk_flag = "calm"
    if vix_val is not None and vix_val > 25:
        risk_flag = "elevated"
    if (btc_chg is not None and btc_chg < -3) or (spy_chg is not None and spy_chg < -1):
        risk_flag = "stressed"

    report = {
        "type": "asia_watch",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "risk_flag": risk_flag,
        "crypto": crypto,
        "macro": macro,
        "us_close_proxies": futures_proxy,
        "headline": (
            f"Overnight watch: {risk_flag}. "
            + (f"VIX {vix_val:.2f}. " if vix_val is not None else "")
            + (f"BTC 24h {btc_chg:+.2f}%. " if btc_chg is not None else "")
            + (f"SPY {spy_chg:+.2f}%." if spy_chg is not None else "")
        ).strip(),
    }
    _save_brief("asia_watch", report)
    _log.info("dfv.asia_watch", risk=risk_flag)
    return report
