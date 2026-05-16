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
    invalidation_breaches = _detect_invalidation_breaches(dfv, payload)

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
            "invalidation_breaches": invalidation_breaches,
            "watchlist_size": len(watchlist),
        },
        "headline": _headline(missing_theses, stale_theses, war_room, invalidation_breaches),
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
    """End-of-day debrief: P&L, expiry sweep, auto post-mortems, conviction nudges."""
    dfv = DFV()
    payload = _safe_collect_payload()
    pnl = payload.get("pnl", {}) or {}

    expired = _detect_expirations(dfv)
    losers = _detect_realized_losses(dfv, payload)
    nudges = _apply_conviction_nudges(dfv, expired, losers)

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
        "expirations_processed": expired,
        "losses_processed": losers,
        "conviction_nudges": nudges,
        "note": "Theses for closed names auto-postmortemed; conviction nudged where loss > $0.",
    }
    _save_brief("eod", report)
    _log.info("dfv.eod", expired=len(expired), losers=len(losers), nudges=len(nudges))
    return report


def _detect_expirations(dfv: DFV) -> list[dict[str, Any]]:
    """Find theses whose target.expiry has passed and write a post-mortem (idempotent)."""
    today = datetime.now(timezone.utc).date()
    out: list[dict[str, Any]] = []
    for sym, rec in dfv.thesis.all().items():
        target = rec.get("target") or {}
        expiry_str = str(target.get("expiry") or "")
        if not expiry_str:
            continue
        try:
            exp_date = datetime.fromisoformat(expiry_str).date()
        except (ValueError, TypeError):
            continue
        if exp_date > today:
            continue
        if dfv.postmortems.has(sym, expiry_str):
            continue  # already written
        realized = float(target.get("realized_pnl") or 0.0)
        status = target.get("status") or ("expired_worthless" if realized <= 0 else "closed_profit")
        pm = dfv.postmortems.append({
            "symbol": sym,
            "status": status,
            "realized_pnl_usd": realized,
            "expiry": expiry_str,
            "thesis_at_open": rec.get("thesis", ""),
            "invalidation_at_open": rec.get("invalidation", ""),
            "what_happened": f"Position reached expiry {expiry_str}. Realized {realized:+.2f}.",
            "lesson": "Auto-generated by eod(). Operator should expand with the actual lesson.",
            "tags": ["auto_eod", status],
            "author": "DFV-eod-auto",
        })
        out.append({"symbol": sym, "expiry": expiry_str, "realized_pnl_usd": realized})
        _log.info("dfv.eod.postmortem", symbol=sym, expiry=expiry_str, realized=realized)
    return out


def _detect_realized_losses(dfv: DFV, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Surface theses where target.realized_pnl < 0 but no post-mortem yet."""
    out: list[dict[str, Any]] = []
    for sym, rec in dfv.thesis.all().items():
        target = rec.get("target") or {}
        realized = float(target.get("realized_pnl") or 0.0)
        expiry_str = str(target.get("expiry") or "")
        if realized >= 0:
            continue
        if expiry_str and dfv.postmortems.has(sym, expiry_str):
            continue
        out.append({"symbol": sym, "realized_pnl_usd": realized})
    return out


def _apply_conviction_nudges(
    dfv: DFV, expired: list[dict[str, Any]], losers: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Drop conviction one tier on any name that closed at a loss (floor 1)."""
    nudged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in [*expired, *losers]:
        sym = entry.get("symbol", "").upper()
        if not sym or sym in seen:
            continue
        if float(entry.get("realized_pnl_usd") or 0.0) >= 0:
            continue
        seen.add(sym)
        prior = dfv.conviction.get(sym)
        if prior <= 1:
            continue
        new_tier = prior - 1
        dfv.conviction.set(
            sym, new_tier,
            reason=f"eod auto-nudge: realized loss {entry.get('realized_pnl_usd'):+.2f}",
        )
        nudged.append({"symbol": sym, "from": prior, "to": new_tier})
        _log.info("dfv.eod.conviction_nudge", symbol=sym, **{"from": prior, "to": new_tier})
    return nudged



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


def _headline(missing: list[str], stale: list[str], war_room: dict[str, Any],
              breaches: list[dict[str, Any]] | None = None) -> str:
    bits: list[str] = []
    regime = (war_room.get("regime") or "").upper() or "?"
    phase = (war_room.get("phase") or "").lower() or "?"
    bits.append(f"Regime {regime} · phase {phase}.")
    if breaches:
        names = ", ".join(b["symbol"] for b in breaches[:5])
        bits.append(f"INVALIDATION BREACH: {names}. Review immediately.")
    if missing:
        bits.append(f"{len(missing)} held name(s) without a thesis: {', '.join(missing[:5])}"
                    + ("…" if len(missing) > 5 else "") + ". Hard rule #1.")
    if stale:
        bits.append(f"{len(stale)} thesis review(s) overdue.")
    if not missing and not stale and not breaches:
        bits.append("Discipline clean. I like the book.")
    return " ".join(bits)


def _detect_invalidation_breaches(dfv: DFV, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Surface theses where current price has crossed the documented invalidation level.

    Best-effort: parses the first dollar amount out of the invalidation string and
    compares to the latest price found in the mission_control payload. If either
    side is missing, the thesis is silently skipped (no false alarm).
    """
    import re
    portfolio = payload.get("portfolio", {}) or {}
    # Build symbol → last_price map from any account/position
    prices: dict[str, float] = {}
    for acct in portfolio.get("accounts", {}).values():
        for p in acct.get("positions", []):
            sym = (p.get("symbol") or p.get("underlying") or "").upper()
            for k in ("last_price", "mark_price", "price", "current_price"):
                v = p.get(k)
                if isinstance(v, (int, float)) and v > 0:
                    prices[sym] = float(v)
                    break

    out: list[dict[str, Any]] = []
    for sym, rec in dfv.thesis.all().items():
        invalidation = str(rec.get("invalidation") or "")
        if not invalidation:
            continue
        m = re.search(r"\$?(\d+(?:\.\d+)?)", invalidation)
        if not m:
            continue
        try:
            level = float(m.group(1))
        except ValueError:
            continue
        last = prices.get(sym)
        if last is None or last <= 0:
            continue
        # Direction inferred from invalidation language
        triggered = False
        text = invalidation.lower()
        if any(kw in text for kw in (">", "above", "exceeds", "sustained")):
            triggered = last > level
        elif any(kw in text for kw in ("<", "below", "breaches", "breach")):
            triggered = last < level
        else:
            # Default: long thesis breaks if price falls below level
            triggered = last < level
        if triggered:
            out.append({
                "symbol": sym,
                "level": level,
                "last_price": last,
                "invalidation": invalidation[:160],
            })
    return out



# ── Sprint 1: distinct cadence routines ────────────────────────────────
# Replace the four stale aliases (asia_digest, open_bell_prep,
# close_debrief, asia_watch) with their own substantive scope.
#
# 2026-05-15: external feeds (CoinGecko / FRED / Finnhub) were orphaned
# from these routines because their keys are empty in .env. Routines
# now operate exclusively on what mission_control already aggregates
# (IBKR / Moomoo / war room) plus DFV's own thesis & watchlist memory.

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
    """04:00 ET — first read of the day. Held names + war-room snapshot."""
    payload = _safe_collect_payload()
    war_room = payload.get("war_room", {}) or {}
    held = _held_symbols()
    overnight_alerts = (payload.get("alerts", []) or [])[:5]

    composite = war_room.get("composite_score")
    regime = (war_room.get("regime") or "").upper() or "?"

    # yfinance fallbacks for macro + crypto (no key required)
    macro = dfv_data.yf_macro_snapshot()
    crypto = dfv_data.yf_crypto_snapshot()

    btc_chg = (
        ((crypto.get("prices") or {}).get("BTC/USD") or {}).get("change_24h_pct")
        if crypto.get("ok") else None
    )
    btc_str = f"BTC 24h {btc_chg:+.2f}%" if isinstance(btc_chg, (int, float)) else "BTC n/a"

    vix = (macro.get("series") or {}).get("VIXCLS")
    vix_str = f"VIX {vix:.2f}" if isinstance(vix, (int, float)) else "VIX n/a"

    # Headlines via Google Custom Search (curated finance domains)
    headlines: list[dict[str, Any]] = []
    try:
        from integrations.google_clients import CustomSearchClient
        cse = CustomSearchClient()
        if cse.configured:
            for query in ("market open", "Federal Reserve", "earnings"):
                headlines.extend(cse.news(query, num=5, hours=24))
            # de-dupe by url, cap at 15
            seen: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for h in headlines:
                u = h.get("url", "")
                if u and u not in seen:
                    seen.add(u)
                    deduped.append(h)
            headlines = deduped[:15]
    except ImportError:
        pass

    report = {
        "type": "asia_digest",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "held_symbols": held,
        "war_room": {
            "composite": composite,
            "regime": regime,
            "phase": war_room.get("phase"),
        },
        "overnight_alerts": overnight_alerts,
        "asia_adr_universe": list(dfv_data.ASIA_ADRS),
        "macro": macro,
        "crypto": crypto,
        "headlines": headlines,
        "headline": (
            f"Asia open. Regime {regime}. {len(held)} held names. "
            f"{len(overnight_alerts)} alert(s). {vix_str} · {btc_str}. "
            f"{len(headlines)} news item(s)."
        ),
        "note": "Macro/crypto via yfinance fallback; news via Google CSE when configured.",
    }
    _save_brief("asia_digest", report)
    _log.info("dfv.asia_digest", held=len(held), alerts=len(overnight_alerts),
              vix=vix, btc_chg=btc_chg, headlines=len(headlines))
    return report


# Macro / sentiment keywords for the retail-pulse routine. Classic
# DFV thesis: when retail Googles "stock market crash" and "recession",
# we are closer to a bottom than the financial press admits.
_RETAIL_MACRO_KEYWORDS: tuple[str, ...] = (
    "stock market crash",
    "recession",
    "buy the dip",
    "VIX",
    "bear market",
)


def retail_pulse() -> dict[str, Any]:
    """Google Trends + YouTube snapshot of retail interest in held names + macro.

    Runs cheaply (pytrends needs no key, YouTube optional). Output feeds
    DFV's contrarian compass — retail euphoria → trim, retail panic → buy.
    """
    held = _held_symbols()
    keywords = list(_RETAIL_MACRO_KEYWORDS) + held[:5]  # cap to keep pytrends quotas tame

    trends_data: dict[str, Any] = {"ok": False, "values": {}}
    trending: list[str] = []
    try:
        from integrations.google_clients import GoogleTrendsClient
        gt = GoogleTrendsClient()
        trends_data = gt.retail_interest(keywords)
        trending = gt.trending_searches()
    except Exception as exc:  # noqa: BLE001 — pytrends + import errors
        _log.warning("dfv.retail_pulse.trends_failed", error=str(exc))

    youtube_hits: dict[str, list[dict[str, Any]]] = {}
    try:
        from integrations.google_clients import YouTubeClient
        yt = YouTubeClient()
        if yt.configured:
            for sym in held[:3]:  # quota: 3 searches × 100 units = 300/day
                youtube_hits[sym] = yt.search(
                    f"{sym} earnings call OR analyst", max_results=5, days=14)
    except Exception as exc:  # noqa: BLE001
        _log.warning("dfv.retail_pulse.youtube_failed", error=str(exc))

    # Contrarian compass: panic-keyword level vs. ticker-keyword level
    panic = [trends_data.get("values", {}).get(k) for k in
             ("stock market crash", "recession", "bear market")]
    panic_avg = (sum(p for p in panic if isinstance(p, (int, float))) /
                 max(1, sum(1 for p in panic if isinstance(p, (int, float))))) \
                 if any(isinstance(p, (int, float)) for p in panic) else None

    if isinstance(panic_avg, (int, float)):
        if panic_avg >= 70:
            sentiment = "extreme_fear"
        elif panic_avg >= 40:
            sentiment = "fear"
        elif panic_avg <= 15:
            sentiment = "complacent"
        else:
            sentiment = "neutral"
    else:
        sentiment = "unknown"

    report = {
        "type": "retail_pulse",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trends": trends_data,
        "trending_searches_us": trending[:10],
        "youtube_per_symbol": youtube_hits,
        "panic_index": panic_avg,
        "sentiment": sentiment,
        "headline": (f"Retail pulse: panic={panic_avg:.0f}/100 ({sentiment}). "
                     f"Tracked {len(keywords)} keywords, {len(youtube_hits)} symbol(s) on YT.")
                     if isinstance(panic_avg, (int, float))
                     else f"Retail pulse: keywords={len(keywords)}, YT symbols={len(youtube_hits)}.",
    }
    _save_brief("retail_pulse", report)
    _log.info("dfv.retail_pulse", panic=panic_avg, sentiment=sentiment,
              kw=len(keywords), yt_syms=len(youtube_hits))
    return report


def open_bell_prep() -> dict[str, Any]:
    """09:25 ET — last 5 minutes before the open. Held names, top alerts, war room."""
    payload = _safe_collect_payload()
    held = _held_symbols()
    war_room = payload.get("war_room", {}) or {}
    alerts = (payload.get("alerts", []) or [])[:3]
    portfolio = payload.get("portfolio", {}) or {}

    report = {
        "type": "open_bell_prep",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "minutes_to_open": "T-5",
        "portfolio": {
            "total_equity_usd": portfolio.get("total_assets_usd")
                or portfolio.get("total_equity")
                or portfolio.get("total_value"),
            "buying_power_usd": portfolio.get("buying_power_usd"),
            "open_positions": len([
                p for acct in portfolio.get("accounts", {}).values()
                for p in acct.get("positions", [])
            ]),
        },
        "held_symbols": held,
        "war_room": {
            "composite": war_room.get("composite_score"),
            "regime": war_room.get("regime"),
            "phase": war_room.get("phase"),
        },
        "alerts": alerts,
        "headline": (
            f"Open in 5. {len(held)} held name(s). "
            f"{len(alerts)} alert(s) on the wire."
        ),
    }
    _save_brief("open_bell_prep", report)
    _log.info("dfv.open_bell_prep", held=len(held), alerts=len(alerts))
    return report


def close_debrief() -> dict[str, Any]:
    """17:00 ET — close summary, stale theses + next required actions."""
    dfv = DFV()
    payload = _safe_collect_payload()
    held = _held_symbols()
    pnl = payload.get("pnl", {}) or {}

    stale = dfv.thesis.needs_review(max_age_days=30)
    next_actions = [
        {"symbol": s, "action": "re-state thesis or close position"} for s in stale
    ]
    watch = list(dfv.watchlist.all().keys())

    report = {
        "type": "close_debrief",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "held_symbols": held,
        "pnl": {
            "today_realized": pnl.get("today_realized"),
            "mtd_realized": pnl.get("mtd_realized"),
            "ytd_realized": pnl.get("ytd_realized"),
        },
        "stale_theses": stale,
        "next_actions": next_actions,
        "watchlist_size": len(watch),
        "headline": (
            f"Close: {len(held)} held. "
            + (f"{len(stale)} stale thesis(es) — write the lesson tonight."
               if stale else "Discipline clean.")
        ),
    }
    _save_brief("close_debrief", report)
    _log.info("dfv.close_debrief", held=len(held), stale=len(stale))
    return report


def asia_watch() -> dict[str, Any]:
    """22:00 ET — going to bed. Snapshot the book + war-room risk state."""
    payload = _safe_collect_payload()
    war_room = payload.get("war_room", {}) or {}
    held = _held_symbols()
    alerts = (payload.get("alerts", []) or [])[:5]

    composite = war_room.get("composite_score")
    risk_flag = "calm"
    if isinstance(composite, (int, float)):
        if composite < 30:
            risk_flag = "stressed"
        elif composite < 50:
            risk_flag = "elevated"

    report = {
        "type": "asia_watch",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "risk_flag": risk_flag,
        "war_room": {
            "composite": composite,
            "regime": war_room.get("regime"),
            "phase": war_room.get("phase"),
        },
        "held_symbols": held,
        "alerts": alerts,
        "headline": (
            f"Overnight watch: {risk_flag}. {len(held)} held name(s). "
            f"Composite {composite if composite is not None else 'n/a'}."
        ),
    }
    _save_brief("asia_watch", report)
    _log.info("dfv.asia_watch", risk=risk_flag, held=len(held))
    return report
