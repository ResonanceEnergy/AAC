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
    # Best-effort: incrementally reindex into DFV semantic memory (rag_lite/SQLite FTS5).
    try:
        from agents.dfv import rag_lite  # noqa: PLC0415
        rag_lite.reindex()
    except Exception as e:  # noqa: BLE001 — daemon must not die on RAG failures
        _log.warning("dfv.routines.reindex_failed", error=str(e))
    return path


def _safe_collect_payload() -> dict[str, Any]:
    """Pull current dashboard payload; tolerate failures."""
    try:
        from monitoring.mission_control import collect_payload  # type: ignore
        return collect_payload() or {}
    except Exception as e:  # noqa: BLE001 — daemon must not die on collector errors
        _log.warning("dfv.collect_failed", error=str(e))
        return {}


def _iter_accounts(portfolio: dict[str, Any]) -> list[dict[str, Any]]:
    """`accounts` can be dict-of-dict or list-of-dict depending on collector
    version. Always yield a list of account dicts."""
    accounts = portfolio.get("accounts") or {}
    if isinstance(accounts, dict):
        return [a for a in accounts.values() if isinstance(a, dict)]
    if isinstance(accounts, list):
        return [a for a in accounts if isinstance(a, dict)]
    return []


def _portfolio_summary(portfolio: dict[str, Any]) -> dict[str, Any]:
    """Single source of truth for portfolio numbers used by briefs.

    `monitoring.mission_control.collect_portfolio()` emits ``total_usd``,
    ``total_cash_usd``, ``total_buying_power_usd``. Older callers tried
    ``total_assets_usd`` / ``cash_usd`` / ``buying_power_usd`` which were
    never written, so every brief showed null. Read both spellings here
    once so the brief and open_bell_prep agree with midday.
    """
    return {
        "total_equity_usd": (
            portfolio.get("total_usd")
            or portfolio.get("total_assets_usd")
            or portfolio.get("total_equity")
            or portfolio.get("total_value")
        ),
        "cash_usd": (
            portfolio.get("total_cash_usd")
            or portfolio.get("cash_usd")
        ),
        "buying_power_usd": (
            portfolio.get("total_buying_power_usd")
            or portfolio.get("buying_power_usd")
        ),
        "open_positions": (
            portfolio.get("total_positions")
            or len([
                p for acct in _iter_accounts(portfolio)
                for p in acct.get("positions", [])
            ])
        ),
    }


# Tag written by orphan_guard for auto-stubbed theses.
_SKELETON_TAG = "auto_skeleton"


def _is_missing_real_thesis(theses: dict[str, Any], symbol: str) -> bool:
    """A held name lacks a real thesis if there's no entry at all OR the entry
    is a TODO skeleton (orphan_guard wrote it with conviction=0 + skeleton tag).

    The skeleton is *recognition*, not *cover*. The discipline section must
    keep flagging skeletons until the operator writes a real thesis.
    """
    rec = theses.get(symbol)
    if not rec:
        return True
    tags = rec.get("tags") or []
    if isinstance(tags, list) and _SKELETON_TAG in tags:
        return True
    if int(rec.get("conviction") or 0) == 0:
        return True
    return False


def _gemini_headline(kind: str, report: dict[str, Any]) -> str | None:
    """One-paragraph AI summary of a brief. Returns None if Gemini not configured."""
    try:
        from integrations.google_clients import GeminiClient  # noqa: PLC0415
    except ImportError:
        return None
    gem = GeminiClient()
    if not gem.configured:
        return None
    import json as _json
    snippet = _json.dumps(report, default=str)[:6000]
    prompt = (
        f"DFV {kind} report below. In 3 short lines:\n"
        f"  1) headline (one sentence)\n"
        f"  2) biggest risk right now\n"
        f"  3) next action (autonomous or needs human OK)\n\n"
        f"REPORT:\n{snippet}"
    )
    system = (
        "You are DFV (Roaring Kitty). Be terse, factual, no fluff. "
        "Numbers > adjectives. Surface only what changed or matters today."
    )
    r = gem.ask(prompt, system=system, temperature=0.2)
    if not r.get("ok"):
        _log.warning("dfv.gemini_headline_failed", kind=kind, error=r.get("error"))
        return None
    return (r.get("text") or "").strip() or None


def brief() -> dict[str, Any]:
    """Pre-market / on-demand brief. The first thing DFV produces every morning."""
    dfv = DFV()
    payload = _safe_collect_payload()
    portfolio = payload.get("portfolio", {})
    war_room = payload.get("war_room", {})

    held_symbols = sorted({
        (p.get("symbol") or p.get("underlying") or "").upper()
        for acct in _iter_accounts(portfolio)
        for p in acct.get("positions", [])
        if (p.get("symbol") or p.get("underlying"))
    })
    held_symbols = [s for s in held_symbols if s]

    theses = dfv.thesis.all()
    # A held name has a *real* thesis only when it's not a TODO skeleton:
    # orphan_guard writes skeletons with conviction=0 + tag 'auto_skeleton'.
    # We treat skeletons as still-missing so the operator can't tolerate
    # an orphan by virtue of the auto-writer pretending it's covered.
    missing_theses = [s for s in held_symbols if _is_missing_real_thesis(theses, s)]
    stale_theses = dfv.thesis.needs_review(max_age_days=30)
    watchlist = dfv.watchlist.all()
    invalidation_breaches = _detect_invalidation_breaches(dfv, payload)

    # Auto-skeleton: writes TODO theses for orphans + emits notifications.
    # (Re-fetch theses + missing list after so the brief reflects the new state.)
    try:
        from agents.dfv import orphan_guard  # noqa: PLC0415
        orphan_result = orphan_guard.scan_and_stub(payload, dfv=dfv)
    except Exception as exc:  # noqa: BLE001
        _log.warning("dfv.brief.orphan_guard_failed", error=str(exc))
        orphan_result = {"written": [], "orphans": []}
    if orphan_result.get("written"):
        # Refresh in-memory thesis view so downstream "missing_thesis" is accurate
        theses = dfv.thesis.all()
        missing_theses = [s for s in held_symbols if _is_missing_real_thesis(theses, s)]

    # Push invalidation-breach notifications (idempotent via dedupe_key)
    try:
        from agents.dfv.notifications import notify  # noqa: PLC0415
        for breach in invalidation_breaches:
            sym = (breach.get("symbol") or "").upper()
            notify(
                dfv=dfv,
                kind="invalidation_breach",
                symbol=sym,
                title=f"🚨 Invalidation breached — {sym}",
                body=str(breach.get("rule") or breach.get("note") or ""),
                severity="critical",
                dedupe_key=f"breach:{sym}:{breach.get('rule', '')[:40]}",
                extra=breach,
            )
    except Exception as exc:  # noqa: BLE001
        _log.warning("dfv.brief.breach_notify_failed", error=str(exc))

    report = {
        "type": "brief",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "voice": "DFV / Roaring Kitty",
        "portfolio_summary": _portfolio_summary(portfolio),
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
            "roll_triggers": _detect_roll_triggers(dfv, payload),
            "watchlist_size": len(watchlist),
        },
        "headline": _headline(missing_theses, stale_theses, war_room, invalidation_breaches),
    }
    report["ai_headline"] = _gemini_headline("brief", report)
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
    report["ai_headline"] = _gemini_headline("midday", report)
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
    screener_summary = _refresh_watchlist_from_screeners(dfv)

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
        "watchlist_refresh": screener_summary,
        "journal_prompt_pending": not dfv.journal.has_today(),
        "note": "Theses for closed names auto-postmortemed; conviction nudged where loss > $0.",
    }
    # If no journal entry yet today, fire a prompt notification.
    if not dfv.journal.has_today():
        try:
            from agents.dfv.notifications import notify  # noqa: PLC0415
            today_str = datetime.now(timezone.utc).date().isoformat()
            notify(
                dfv=dfv,
                kind="journal_prompt",
                title="📓 EOD journal — what did you learn?",
                body="Open the dashboard and write one sentence. No entry = no compounding insight.",
                severity="info",
                dedupe_key=f"journal_prompt:{today_str}",
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("dfv.eod.journal_prompt_failed", error=str(exc))
    report["ai_headline"] = _gemini_headline("eod", report)
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
    nudge_log_path = REPO_ROOT / "agents" / "dfv" / "memory" / "conviction_nudges.jsonl"
    nudge_log_path.parent.mkdir(parents=True, exist_ok=True)
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
        reason = f"eod auto-nudge: realized loss {entry.get('realized_pnl_usd'):+.2f}"
        dfv.conviction.set(sym, new_tier, reason=reason)
        nudged.append({"symbol": sym, "from": prior, "to": new_tier})
        # Audit-trail JSONL so the dashboard can show recent nudges.
        try:
            import json as _json
            with nudge_log_path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "symbol": sym, "from": prior, "to": new_tier,
                    "reason": reason, "source": "eod_auto",
                }) + "\n")
        except OSError as e:
            _log.warning("dfv.eod.nudge_log_failed", error=str(e))
        _log.info("dfv.eod.conviction_nudge", symbol=sym, **{"from": prior, "to": new_tier})
    return nudged


def _detect_roll_triggers(dfv: DFV, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Find option positions inside their roll-trigger DTE window.

    Per the $659 macro-hedge postmortem cluster: long options that drift past
    21 DTE without a plan bleed theta and expire worthless. Surface every name
    whose nearest expiry sits at-or-inside its thesis.roll_trigger_dte.
    """
    out: list[dict[str, Any]] = []
    today = datetime.now(timezone.utc).date()
    doctrine_default = 21
    try:
        doctrine_default = int(
            (dfv.doctrine.get("roll_policy", {}) or {}).get("default_trigger_dte", 21)
        )
    except (ValueError, TypeError):
        pass

    portfolio = payload.get("portfolio", {}) or {}
    # Build symbol -> [expiries] map across all venues
    positions_by_sym: dict[str, list[dict[str, Any]]] = {}
    for acct in _iter_accounts(portfolio):
        for p in acct.get("positions", []) or []:
            sym = (p.get("symbol") or p.get("underlying") or "").upper()
            if not sym:
                continue
            positions_by_sym.setdefault(sym, []).append(p)

    for sym, rec in dfv.thesis.all().items():
        positions = positions_by_sym.get(sym, [])
        if not positions:
            continue
        trigger_dte = int(rec.get("roll_trigger_dte") or doctrine_default)
        # Collect expiries (option positions only).
        expiries: list[tuple[str, int]] = []
        for p in positions:
            exp_str = str(p.get("expiry") or p.get("expiration") or "")
            if not exp_str:
                continue
            try:
                exp_date = datetime.fromisoformat(exp_str[:10]).date()
            except (ValueError, TypeError):
                continue
            dte = (exp_date - today).days
            expiries.append((exp_str, dte))
        if not expiries:
            continue
        expiries.sort(key=lambda x: x[1])
        nearest_expiry, nearest_dte = expiries[0]
        if nearest_dte <= trigger_dte:
            out.append({
                "symbol": sym,
                "expiry": nearest_expiry,
                "dte": nearest_dte,
                "trigger_dte": trigger_dte,
                "status": "expired" if nearest_dte < 0 else "roll_or_kill",
            })
    return out



def weekend_dd() -> dict[str, Any]:
    """Weekend DD slot — write a real markdown DD per held name with conviction >= N.

    Per operator directive 2026-05-16: weekend slot should not just list
    names, it should *force* a 60-second whiteboard restatement of each
    high-conviction thesis. Markdown files land in
    ``agents/dfv/memory/dd/{SYMBOL}_dd.md`` and are idempotently
    overwritten every Saturday.
    """
    dfv = DFV()
    doctrine = (dfv.doctrine.get("weekend_dd") or {}) if isinstance(dfv.doctrine, dict) else {}
    min_conviction = int(doctrine.get("min_conviction", 3))
    out_dir_rel = str(doctrine.get("out_dir", "agents/dfv/memory/dd"))
    out_dir = REPO_ROOT / out_dir_rel
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _safe_collect_payload()
    portfolio = payload.get("portfolio", {}) or {}
    # Build a quick (symbol -> price) lookup for the DD writer.
    last_price: dict[str, float] = {}
    held_qty: dict[str, float] = {}
    held_unreal: dict[str, float] = {}
    for acct in _iter_accounts(portfolio):
        for p in acct.get("positions", []) or []:
            sym = (p.get("symbol") or p.get("underlying") or "").upper()
            if not sym:
                continue
            for k in ("market_price", "last_price", "price", "mark_price"):
                v = p.get(k)
                if isinstance(v, (int, float)) and v > 0:
                    last_price[sym] = float(v)
                    break
            try:
                held_qty[sym] = held_qty.get(sym, 0.0) + float(p.get("qty") or 0.0)
            except (TypeError, ValueError):
                pass
            try:
                held_unreal[sym] = held_unreal.get(sym, 0.0) + float(p.get("unrealized_pnl") or 0.0)
            except (TypeError, ValueError):
                pass

    written: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    convictions = dfv.conviction.all()
    for sym, rec in dfv.thesis.all().items():
        conv = int(rec.get("conviction") or (convictions.get(sym, {}) or {}).get("tier") or 0)
        if conv < min_conviction:
            skipped.append({"symbol": sym, "conviction": conv})
            continue
        md = _render_dd_markdown(
            sym, rec,
            current_price=last_price.get(sym),
            qty=held_qty.get(sym),
            unrealized=held_unreal.get(sym),
        )
        path = out_dir / f"{sym}_dd.md"
        path.write_text(md, encoding="utf-8")
        written.append({"symbol": sym, "conviction": conv, "path": str(path.relative_to(REPO_ROOT))})

    report = {
        "type": "weekend_dd",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_conviction": min_conviction,
        "dd_files_written": written,
        "skipped_below_conviction": skipped,
        "to_review_this_weekend": dfv.thesis.needs_review(max_age_days=14),
        "note": "Each DD is a 60-second whiteboard restatement. Read the filings; "
                "if you can't restate it in plain English, drop the position.",
    }
    _save_brief("weekend_dd", report)
    _log.info("dfv.weekend_dd", written=len(written), skipped=len(skipped))
    return report


def _render_dd_markdown(
    symbol: str,
    rec: dict[str, Any],
    *,
    current_price: float | None,
    qty: float | None,
    unrealized: float | None,
) -> str:
    """Render one weekly DD page. Pure formatting — no I/O."""
    target = rec.get("target") or {}
    catalysts = rec.get("catalysts") or []
    sizing = rec.get("sizing") or {}
    exit_ladder = target.get("exit_ladder") if isinstance(target, dict) else None
    lines: list[str] = []
    lines.append(f"# {symbol} — Weekend DD")
    lines.append("")
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='minutes')} by DFV.weekend_dd_")
    lines.append("")
    lines.append(f"**Conviction:** {rec.get('conviction', '?')} | "
                 f"**Author:** {rec.get('author', '?')} | "
                 f"**Updated:** {rec.get('updated', '?')}")
    lines.append("")
    lines.append("## Thesis snapshot")
    lines.append("")
    lines.append(f"> {rec.get('thesis', '_(no thesis)_')}")
    lines.append("")
    lines.append("## Catalysts and invalidation")
    lines.append("")
    if catalysts:
        for c in catalysts:
            lines.append(f"- {c}")
    else:
        lines.append("- _(no catalysts logged)_")
    lines.append("")
    lines.append(f"**Invalidation:** {rec.get('invalidation', '_(none)_ ')}")
    lines.append(f"**Horizon:** {rec.get('horizon', '_(none)_')}")
    lines.append("")
    lines.append("## Position and P&L")
    lines.append("")
    lines.append(f"- Qty held (across venues): **{qty if qty is not None else 'n/a'}**")
    lines.append(f"- Current price: **{current_price if current_price is not None else 'n/a'}**")
    lines.append(f"- Unrealized P&L (USD): **{unrealized if unrealized is not None else 'n/a'}**")
    if sizing:
        lines.append(f"- Doctrinal max % of book: **{sizing.get('max_pct_book', 'n/a')}**")
    lines.append("")
    lines.append("## Exit ladder")
    lines.append("")
    if isinstance(exit_ladder, dict) and exit_ladder:
        for k, v in exit_ladder.items():
            lines.append(f"- **{k}:** {v}")
    else:
        lines.append("- _(no exit ladder defined — write one before next session)_")
    lines.append("")
    lines.append("## What would change my mind")
    lines.append("")
    lines.append("- If invalidation triggers above are hit, close per ladder.")
    lines.append("- If catalyst window closes without the move, downgrade conviction one tier.")
    lines.append("- If I cannot restate this thesis in 60 seconds on a whiteboard, drop the name.")
    lines.append("")
    return "\n".join(lines)


def _refresh_watchlist_from_screeners(dfv: DFV) -> dict[str, Any]:
    """Run the EOD screeners and swap the watchlist atomically.

    Returns a summary dict for the EOD brief; never raises.
    """
    cfg = (dfv.doctrine.get("screeners") or {}) if isinstance(dfv.doctrine, dict) else {}
    if not cfg.get("enabled", True):
        return {"status": "disabled"}
    active = list(cfg.get("active") or [])
    top_n = int(cfg.get("top_n_per_screen", 5))
    cap = int(cfg.get("total_cap", 15))
    try:
        from agents.dfv import screeners as _screeners  # noqa: PLC0415
        entries = _screeners.run_all(
            active=active or None,
            top_n_per_screen=top_n,
            total_cap=cap,
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("dfv.eod.screener_failed", error=str(exc))
        return {"status": "error", "error": str(exc)}
    try:
        dfv.watchlist.replace_all(entries)
    except Exception as exc:  # noqa: BLE001
        _log.warning("dfv.eod.watchlist_swap_failed", error=str(exc))
        return {"status": "error", "error": str(exc), "would_have_written": len(entries)}
    return {
        "status": "ok",
        "count": len(entries),
        "active_screeners": active,
        "top_5": list(entries.keys())[:5],
    }


def drift_monitor() -> dict[str, Any]:
    """Compare war-room arm allocations to doctrine targets; alert on persistent drift.

    Reads ``mission_control.collect_payload().war_room.arms`` (a list of
    ``{arm, target_pct, actual_pct, actual_usd, name}``), computes
    ``drift = actual_pct - target_pct`` for each, and bumps a per-arm
    counter at ``agents/dfv/memory/drift_state.json``. When ``|drift| >=
    threshold_pct`` for ``sessions_required`` consecutive snapshots, a
    ``notify(kind='arm_drift')`` fires. Counter resets when the arm
    returns inside the band.
    """
    dfv = DFV()
    cfg = (dfv.doctrine.get("drift_monitor") or {}) if isinstance(dfv.doctrine, dict) else {}
    if not cfg.get("enabled", True):
        return {"type": "drift_monitor", "status": "disabled"}
    threshold = float(cfg.get("threshold_pct", 5.0))
    required = int(cfg.get("sessions_required", 3))
    severity = str(cfg.get("notify_severity", "warn"))

    payload = _safe_collect_payload()
    war_room = payload.get("war_room") or {}
    arms = war_room.get("arms") or []
    if not isinstance(arms, list) or not arms:
        return {"type": "drift_monitor", "status": "no_arms", "war_room_keys": list(war_room.keys())}

    state_path = REPO_ROOT / "agents" / "dfv" / "memory" / "drift_state.json"
    import json as _json
    try:
        state = _json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    except Exception:  # noqa: BLE001 — corrupt state file, start fresh
        state = {}
    if not isinstance(state, dict):
        state = {}

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    for arm in arms:
        if not isinstance(arm, dict):
            continue
        key = str(arm.get("arm") or "").strip()
        if not key:
            continue
        target = float(arm.get("target_pct") or 0.0)
        actual = float(arm.get("actual_pct") or 0.0)
        drift = actual - target
        abs_drift = abs(drift)
        prev = state.get(key, {}) if isinstance(state.get(key), dict) else {}
        prev_count = int(prev.get("off_sessions") or 0)
        if abs_drift >= threshold:
            count = prev_count + 1
        else:
            count = 0
        state[key] = {
            "last_seen": now_iso,
            "target_pct": target,
            "actual_pct": actual,
            "drift_pct": round(drift, 3),
            "off_sessions": count,
        }
        row = {
            "arm": key,
            "name": arm.get("name", key),
            "target_pct": target,
            "actual_pct": actual,
            "drift_pct": round(drift, 3),
            "off_sessions": count,
            "off_target": abs_drift >= threshold,
        }
        rows.append(row)
        if count >= required:
            alerts.append(row)
            try:
                from agents.dfv.notifications import notify  # noqa: PLC0415
                notify(
                    dfv=dfv,
                    kind="arm_drift",
                    title=f"⚖️ Arm drift — {key}",
                    body=(
                        f"{arm.get('name', key)}: actual {actual:.1f}% vs "
                        f"target {target:.1f}% (drift {drift:+.1f}pp) for "
                        f"{count} consecutive sessions. Re-balance or accept and re-doctrine."
                    ),
                    severity=severity,
                    dedupe_key=f"arm_drift:{key}:{count // required}",
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning("dfv.drift_monitor.notify_failed", arm=key, error=str(exc))

    try:
        state_path.write_text(_json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _log.warning("dfv.drift_monitor.state_write_failed", error=str(exc))

    report = {
        "type": "drift_monitor",
        "generated_at": now_iso,
        "threshold_pct": threshold,
        "sessions_required": required,
        "arms": rows,
        "alerts_fired": alerts,
    }
    _save_brief("drift_monitor", report)
    _log.info("dfv.drift_monitor", arms=len(rows), alerts=len(alerts))
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
    for acct in _iter_accounts(portfolio):
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
        for acct in _iter_accounts(portfolio)
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
    # Defensive: trends_data may not be a dict if pytrends returns an unexpected shape.
    if not isinstance(trends_data, dict):
        _log.warning("dfv.retail_pulse.trends_unexpected_shape",
                     type=type(trends_data).__name__)
        trends_data = {"ok": False, "values": {}, "raw": trends_data}
    trends_values = trends_data.get("values") or {}
    if not isinstance(trends_values, dict):
        trends_values = {}
    panic = [trends_values.get(k) for k in
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
        "portfolio": _portfolio_summary(portfolio),
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
