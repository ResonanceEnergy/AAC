"""🐱 DFV Dashboard — for DFV, by DFV.

The Roaring Kitty operator console. Single screen. His priorities, in his order.

Run:    python launch.py dfv-dash
Direct: streamlit run monitoring/dfv_dashboard.py --server.port 8503

Hard rule: this file is a MIRROR of doctrine, not a fork. Numbers come from
collect_payload() and the agents/dfv/memory/ files. No collectors live here.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from agents.dfv import routines as dfv_routines
from agents.dfv.daemon import heartbeat_status
from agents.dfv.decision_engine import DFV

ROOT = Path(__file__).resolve().parent.parent

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐱 DFV — Roaring Kitty",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
)

DFV_RED = "#ff4d4d"
DFV_GREEN = "#33cc66"
DFV_YELLOW = "#ffcc00"
DFV_BLUE = "#5599ff"


# ── Data loaders (cached) ────────────────────────────────────────────────────
@st.cache_data(ttl=20)
def _load_payload() -> dict[str, Any]:
    """Pull the unified mission_control payload. Cached 20s — DFV reads, doesn't poll."""
    try:
        from monitoring.mission_control import collect_payload
        return collect_payload() or {}
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)}


@st.cache_data(ttl=10)
def _load_dfv_state() -> dict[str, Any]:
    """All DFV memory in one shot."""
    dfv = DFV()
    try:
        recon = dfv.reconciliation.read() or {}
    except Exception:  # noqa: BLE001
        recon = {}
    try:
        notifs = dfv.notifications.tail(100)
    except Exception:  # noqa: BLE001
        notifs = []
    return {
        "theses": dfv.thesis.all(),
        "conviction": dfv.conviction.all(),
        "watchlist": dfv.watchlist.all(),
        "decisions": dfv.decisions.tail(50),
        "postmortems": dfv.postmortems.all()[-10:],
        "heartbeat": _safe_heartbeat(),
        "doctrine": dfv.doctrine,
        "reconciliation": recon,
        "notifications": notifs,
    }


def _safe_heartbeat() -> dict[str, Any]:
    try:
        return heartbeat_status() or {}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "alive": False}


def _load_latest_brief() -> dict[str, Any] | None:
    bdir = ROOT / "agents" / "dfv" / "memory" / "briefs"
    if not bdir.exists():
        return None
    files = sorted(bdir.glob("*.json"), reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ── Position model ───────────────────────────────────────────────────────────
def _flatten_positions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    portfolio = payload.get("portfolio", {}) or {}
    accounts = portfolio.get("accounts") or []
    # accounts can be either list[dict] (current schema) or dict[str, dict]
    if isinstance(accounts, dict):
        iterable = accounts.items()
    else:
        iterable = ((a.get("name") or a.get("platform") or "?", a) for a in accounts)
    for acct_name, acct in iterable:
        for p in acct.get("positions", []) or []:
            sym = (p.get("symbol") or p.get("underlying") or "").upper()
            if not sym:
                continue
            greeks = p.get("greeks") or {}
            qty = p.get("quantity") or p.get("qty") or 0
            try:
                qty_f = float(qty)
            except (ValueError, TypeError):
                qty_f = 0.0
            # Options are quoted per-share; multiplier 100 unless overridden.
            mult = 100 if (p.get("asset_type") or p.get("type") or "").lower() in ("option", "put", "call") else 1
            rows.append({
                "symbol": sym,
                "account": acct_name,
                "qty": qty,
                "side": p.get("side") or p.get("direction") or "-",
                "last_price": p.get("last_price") or p.get("mark_price") or p.get("price") or 0,
                "avg_cost": p.get("avg_cost") or p.get("cost_basis") or 0,
                "unrealized": p.get("unrealized_pnl") or p.get("pnl_unrealized") or 0,
                "expiry": p.get("expiry") or p.get("expiration") or "",
                "asset_type": p.get("asset_type") or p.get("type") or "stock",
                "delta": greeks.get("delta") if greeks else p.get("delta"),
                "theta": greeks.get("theta") if greeks else p.get("theta"),
                "vega": greeks.get("vega") if greeks else p.get("vega"),
                "_qty_signed": qty_f,
                "_multiplier": mult,
            })
    return rows


def _book_greeks(positions: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate signed book-level Greeks. Theta is dollars-per-day."""
    total = {"delta": 0.0, "theta": 0.0, "vega": 0.0}
    have_any = False
    for p in positions:
        for k in ("delta", "theta", "vega"):
            v = p.get(k)
            if isinstance(v, (int, float)):
                total[k] += float(v) * p["_qty_signed"] * p["_multiplier"]
                have_any = True
    total["_have_any"] = have_any  # type: ignore[assignment]
    return total


def _parse_invalidation_level(text: str) -> tuple[float | None, str]:
    """Return (level, direction) where direction is 'below' or 'above'."""
    if not text:
        return None, "below"
    m = re.search(r"\$?(\d+(?:\.\d+)?)", text)
    if not m:
        return None, "below"
    try:
        level = float(m.group(1))
    except ValueError:
        return None, "below"
    low = text.lower()
    if any(kw in low for kw in (">", "above", "exceeds", "sustained")):
        return level, "above"
    return level, "below"


# ── "AM I OK?" traffic light ─────────────────────────────────────────────────
def _render_traffic_light(payload: dict[str, Any], dfv_state: dict[str, Any]) -> None:
    """One-glance health check. Hard rule #4: explain in 60 seconds.

    GREEN  = all checks pass
    AMBER  = 1–2 soft fails
    RED    = any hard fail OR 3+ soft fails
    """
    portfolio = payload.get("portfolio", {}) or {}
    equity = float(
        portfolio.get("total_usd")
        or portfolio.get("total_assets_usd")
        or portfolio.get("total_equity")
        or portfolio.get("total_value")
        or 0.0
    )
    cash = float(portfolio.get("total_cash_usd") or portfolio.get("cash_usd") or 0.0)
    cash_pct = (cash / equity * 100.0) if equity > 0 else 100.0  # no equity = ignore

    theses = dfv_state.get("theses") or {}
    orphans = [
        s for s, t in theses.items()
        if "auto_skeleton" in (t.get("tags") or [])
    ]

    recon = dfv_state.get("reconciliation") or {}
    mismatches = int(recon.get("mismatch_count") or 0)

    positions = _flatten_positions(payload)
    book_g = _book_greeks(positions)
    theta_abs = abs(float(book_g.get("theta") or 0.0)) if book_g.get("_have_any") else 0.0
    theta_pct = (theta_abs / equity) if equity > 0 else 0.0

    today = datetime.now(timezone.utc).date().isoformat()
    breaches_today = [
        n for n in (dfv_state.get("notifications") or [])
        if n.get("kind") == "invalidation_breach" and str(n.get("ts", "")).startswith(today)
    ]

    soft_fails: list[str] = []
    hard_fails: list[str] = []

    if cash_pct < 5:
        hard_fails.append(f"💀 Cash {cash_pct:.1f}% — below hard floor (rule #6: cash is a position)")
    elif cash_pct < 10:
        soft_fails.append(f"⚠️ Cash {cash_pct:.1f}% — below the 10% rule")

    if theta_pct > 0.01:
        hard_fails.append(f"💀 Theta ${theta_abs:,.0f}/day = {theta_pct*100:.2f}% of book — bleeding")
    elif theta_pct > 0.005:
        soft_fails.append(f"⚠️ Theta ${theta_abs:,.0f}/day = {theta_pct*100:.2f}% of book — elevated")

    if breaches_today:
        hard_fails.append(f"🚨 {len(breaches_today)} invalidation breach(es) today — rule #3 in play")

    if orphans:
        soft_fails.append(
            f"📝 {len(orphans)} TODO thesis stub(s): {', '.join(orphans[:3])}"
            + (" …" if len(orphans) > 3 else "")
        )

    if mismatches:
        soft_fails.append(f"⚖️ {mismatches} position drift(s) across venues")

    fail_count = len(hard_fails) + len(soft_fails)
    if hard_fails or fail_count >= 3:
        color, status, tagline = "#7f1d1d", "🔴 RED", "Stand down. Fix the hard fails before any new trade."
    elif fail_count >= 1:
        color, status, tagline = "#854d0e", "🟡 AMBER", "Tighten up. Soft fails accumulating."
    else:
        color, status, tagline = "#14532d", "🟢 GREEN", (
            "Cash &gt; 10%, theta benign, no breaches, theses written, books reconciled. I like the book."
        )

    items = "".join(
        f"<li style='margin:2px 0;'>{f}</li>" for f in (hard_fails + soft_fails)
    )
    body_html = (
        f"<div style='background:{color};color:white;padding:14px 18px;"
        f"border-radius:10px;margin-bottom:12px;'>"
        f"<div style='display:flex;align-items:baseline;gap:12px;'>"
        f"<h2 style='margin:0;font-size:1.4em;'>AM I OK? {status}</h2>"
        f"<span style='opacity:0.85;'>— {tagline}</span></div>"
        + (f"<ul style='margin:8px 0 0 18px;padding:0;'>{items}</ul>" if items else "")
        + "</div>"
    )
    st.markdown(body_html, unsafe_allow_html=True)


# ── Catalyst countdown strip (next 7 days) ───────────────────────────────────
_ISO_DATE_RE = __import__("re").compile(r"(\d{4}-\d{2}-\d{2})")


def _render_catalyst_countdown(payload: dict[str, Any], dfv_state: dict[str, Any]) -> None:
    """Horizontal strip: every dated catalyst in the next 7 days, sorted by proximity.

    Pulls from two sources:
      1. payload["alerts"] (collector-driven: earnings, FOMC, etc.)
      2. theses[sym]["catalysts"] (DFV-written strings — parse ISO dates if any)
    """
    today = datetime.now(timezone.utc).date()
    horizon = 7
    items: list[dict[str, Any]] = []

    # Source 1 — collector alerts
    for a in payload.get("alerts") or []:
        days = a.get("days_until") if a.get("days_until") is not None else a.get("days_ahead")
        if days is None:
            continue
        try:
            d_int = int(days)
        except (ValueError, TypeError):
            continue
        if 0 <= d_int <= horizon:
            items.append({
                "days": d_int,
                "symbol": (a.get("symbol") or a.get("ticker") or "·").upper(),
                "what": a.get("description") or a.get("title") or a.get("event") or "event",
                "src": "alert",
            })

    # Source 2 — thesis-embedded catalysts
    for sym, t in (dfv_state.get("theses") or {}).items():
        for cat in (t.get("catalysts") or []):
            if not isinstance(cat, str):
                continue
            m = _ISO_DATE_RE.search(cat)
            if not m:
                continue
            try:
                cat_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            except ValueError:
                continue
            delta = (cat_date - today).days
            if 0 <= delta <= horizon:
                items.append({
                    "days": delta,
                    "symbol": sym.upper(),
                    "what": cat.strip(),
                    "src": "thesis",
                })

    if not items:
        return  # silent — keep the page calm when nothing's pending

    items.sort(key=lambda r: (r["days"], r["symbol"]))

    chips = []
    for it in items[:12]:  # cap visual noise
        d = it["days"]
        if d == 0:
            badge = "🔥 TODAY"
            bg = "#7f1d1d"
        elif d <= 2:
            badge = f"🟠 T-{d}d"
            bg = "#854d0e"
        else:
            badge = f"🟡 T-{d}d"
            bg = "#3f3f46"
        what = it["what"]
        if len(what) > 60:
            what = what[:57] + "…"
        chips.append(
            f"<span style='display:inline-block;background:{bg};color:white;"
            f"padding:4px 10px;border-radius:14px;margin:3px 6px 3px 0;"
            f"font-size:0.9em;'>"
            f"<b>{badge}</b> &nbsp;<b>{it['symbol']}</b>&nbsp;·&nbsp;{what}</span>"
        )
    st.markdown(
        "<div style='margin:0 0 10px 0;'>"
        "<span style='opacity:0.7;font-size:0.85em;'>📅 Catalyst countdown (≤7d)</span><br/>"
        + "".join(chips)
        + "</div>",
        unsafe_allow_html=True,
    )


# ── Headline strip ───────────────────────────────────────────────────────────
def _render_headline(payload: dict[str, Any], dfv_state: dict[str, Any]) -> None:
    brief = _load_latest_brief() or {}
    headline = (
        brief.get("headline")
        or brief.get("note")
        or "No brief on file. Run `python -m agents.dfv brief`."
    )
    when = brief.get("generated_at") or "—"
    btype = (brief.get("type") or "brief").upper()

    col_h, col_hb = st.columns([6, 1])
    with col_h:
        st.markdown(
            f"<div style='font-size:1.5rem;line-height:1.4;color:#eee;"
            f"border-left:4px solid {DFV_BLUE};padding-left:14px;'>"
            f"<b>🐱 {btype}</b> &middot; <span style='color:#888'>{when[:19]}</span><br>"
            f"<span style='color:#fff'>{headline}</span></div>",
            unsafe_allow_html=True,
        )
    with col_hb:
        hb = dfv_state["heartbeat"]
        if hb.get("error"):
            status, color = "ERROR", DFV_RED
        elif hb.get("alive"):
            status, color = "ALIVE", DFV_GREEN
        elif "alive" in hb:
            status, color = "STALE", DFV_RED
        else:
            status, color = "UNKNOWN", DFV_YELLOW
        last_ts = hb.get("last_ts") or hb.get("last_routine_ts") or ""
        last = str(last_ts)[11:19] if last_ts else "—"
        age = hb.get("age_seconds")
        age_str = f"{int(age)}s ago" if isinstance(age, (int, float)) else ""
        last_routine = hb.get("last_routine") or "—"
        st.markdown(
            f"<div style='text-align:center;background:{color}22;padding:10px;border-radius:6px;'>"
            f"<div style='font-size:0.75rem;color:#888'>DAEMON</div>"
            f"<div style='font-size:1.1rem;color:{color};font-weight:bold'>{status}</div>"
            f"<div style='font-size:0.7rem;color:#aaa'>last beat {last} {age_str}</div>"
            f"<div style='font-size:0.65rem;color:#888'>last routine: {last_routine}</div></div>",
            unsafe_allow_html=True,
        )


# ── Top numbers strip ────────────────────────────────────────────────────────
def _render_metrics_strip(payload: dict[str, Any], dfv_state: dict[str, Any]) -> dict[str, Any]:
    portfolio = payload.get("portfolio", {}) or {}
    pnl = payload.get("pnl", {}) or {}
    equity = float(
        portfolio.get("total_usd")
        or portfolio.get("total_assets_usd")
        or portfolio.get("total_equity")
        or portfolio.get("total_value")
        or 0.0
    )
    cash = float(portfolio.get("total_cash_usd") or portfolio.get("cash_usd") or 0.0)
    bp = float(portfolio.get("total_buying_power_usd") or portfolio.get("buying_power_usd") or 0.0)
    cash_pct = (cash / equity * 100) if equity > 0 else 0.0

    positions = _flatten_positions(payload)
    open_count = len(positions)
    theses = dfv_state["theses"]
    held_syms = {p["symbol"] for p in positions}
    missing = [s for s in held_syms if s not in theses]
    today_realized = float(pnl.get("today_realized") or pnl.get("realized_today") or portfolio.get("total_realized_pnl_usd") or 0.0)

    # FOMO vetoes today
    today = datetime.now(timezone.utc).date().isoformat()
    fomo_vetoes = 0
    for d in dfv_state["decisions"]:
        if not str(d.get("ts", "")).startswith(today):
            continue
        dec = d.get("decision", {}) or {}
        blob = (dec.get("summary") or "") + " " + " ".join(dec.get("notes") or [])
        if "FOMO" in blob.upper():
            fomo_vetoes += 1

    cols = st.columns(8)
    cols[0].metric("Equity", f"${equity:,.0f}")
    cash_help = "🚨 BELOW 10% — dry powder is sacred" if 0 < cash_pct < 10 else ""
    cols[1].metric("Cash %", f"{cash_pct:.1f}%", delta=cash_help, delta_color="inverse")
    cols[2].metric("Dry Powder", f"${cash:,.0f}")
    cols[3].metric("Buying Power", f"${bp:,.0f}")
    cols[4].metric(
        "Today P&L",
        f"${today_realized:+,.0f}",
        delta_color="normal" if today_realized >= 0 else "inverse",
    )
    cols[5].metric(
        "Positions / Theses",
        f"{open_count} / {len(theses)}",
        delta=f"{len(missing)} missing" if missing else "all theses ✓",
        delta_color="inverse" if missing else "normal",
    )
    cols[6].metric("FOMO vetoes (today)", str(fomo_vetoes))
    book_g = _book_greeks(positions)
    if book_g.get("_have_any"):
        # Theta is typically negative for long options → dollars/day bleeding off.
        cols[7].metric(
            "Theta / day", f"${book_g['theta']:+,.0f}",
            delta=f"vega ${book_g['vega']:+,.0f}", delta_color="off",
        )
    else:
        cols[7].metric("Theta / day", "n/a", delta="no greeks", delta_color="off")

    return {
        "equity": equity,
        "cash": cash,
        "cash_pct": cash_pct,
        "positions": positions,
        "missing_thesis": missing,
        "held_syms": held_syms,
        "book_greeks": book_g,
    }


# ── The Book (positions × theses × invalidation) ─────────────────────────────
def _render_the_book(metrics: dict[str, Any], dfv_state: dict[str, Any]) -> None:
    st.markdown("### 📖 The Book")
    st.caption("Every position. Every thesis. Every invalidation level. Hard rule #1.")

    positions = metrics["positions"]
    theses = dfv_state["theses"]
    conviction_map = {k.upper(): v.get("tier", 0) for k, v in dfv_state["conviction"].items()}

    if not positions:
        st.info("No open positions. Cash is a position. ✊")
        return

    rows: list[dict[str, Any]] = []
    today = datetime.now(timezone.utc).date()
    doctrine_default_dte = int(
        ((dfv_state["doctrine"].get("roll_policy") or {}).get("default_trigger_dte") or 21)
    )
    for p in positions:
        sym = p["symbol"]
        thesis = theses.get(sym) or {}
        invalidation = thesis.get("invalidation") or ""
        level, direction = _parse_invalidation_level(invalidation)
        last = float(p.get("last_price") or 0)
        breach = ""
        if level is not None and last > 0:
            if direction == "above" and last > level:
                breach = "🚨"
            elif direction == "below" and last < level:
                breach = "🚨"
        roll_dte = int(thesis.get("roll_trigger_dte") or doctrine_default_dte)
        dte: int | str = ""
        roll_flag = ""
        if p.get("expiry"):
            try:
                exp = datetime.fromisoformat(str(p["expiry"])[:10]).date()
                d = (exp - today).days
                dte = max(0, d)
                if d <= 0:
                    dte = "TODAY 🔔"
                    roll_flag = "🔥"
                elif d <= roll_dte:
                    roll_flag = "⏰"
            except (ValueError, TypeError):
                pass

        thesis_status = "✅" if sym in theses else "❌ MISSING"
        conviction = conviction_map.get(sym, thesis.get("conviction", 0))

        def _fmt_greek(v: Any) -> str:
            return f"{float(v):+.2f}" if isinstance(v, (int, float)) else "-"

        rows.append({
            "🚨": breach,
            "⏰": roll_flag,
            "symbol": sym,
            "side": p.get("side", "-"),
            "qty": p.get("qty"),
            "last": f"{last:.2f}" if last else "-",
            "avg": f"{float(p.get('avg_cost') or 0):.2f}",
            "unrealized": f"{float(p.get('unrealized') or 0):+,.0f}",
            "DTE": dte if dte != "" else "-",
            "roll@DTE": roll_dte,
            "Δ": _fmt_greek(p.get("delta")),
            "Θ": _fmt_greek(p.get("theta")),
            "ν": _fmt_greek(p.get("vega")),
            "thesis": thesis_status,
            "T": conviction or "-",
            "invalidation": (invalidation[:50] + "…") if len(invalidation) > 50 else (invalidation or "—"),
            "account": p.get("account", "-"),
        })

    # Sort: breaches first, then roll-flags, then missing thesis, then by symbol
    rows.sort(key=lambda r: (r["🚨"] != "🚨", r["⏰"] == "", "MISSING" not in r["thesis"], r["symbol"]))
    st.dataframe(rows, hide_index=True, use_container_width=True)

    if metrics["missing_thesis"]:
        st.error(
            f"❌ **HARD RULE #1 VIOLATED** — held without a thesis: "
            f"{', '.join(metrics['missing_thesis'])}. Write one or close it."
        )

    with st.expander("📄 Whiteboard export (60-second card per position)", expanded=False):
        for sym in sorted({p["symbol"] for p in positions}):
            thesis = theses.get(sym)
            if not thesis:
                st.markdown(f"#### {sym} — *no thesis*")
                continue
            tier = conviction_map.get(sym, thesis.get("conviction", "—"))
            target = thesis.get("target") or {}
            sizing = thesis.get("sizing") or {}
            cats = thesis.get("catalysts") or []
            wb = (
                f"### {sym} — conviction T{tier} — horizon: {thesis.get('horizon', '—')}\n\n"
                f"**Thesis (60s):** {thesis.get('thesis', '—')}\n\n"
                f"**Invalidation:** {thesis.get('invalidation') or '—'}\n\n"
                f"**Catalysts:** {', '.join(cats) if cats else '—'}\n\n"
                f"**Target:** {target}\n\n"
                f"**Sizing:** {sizing}\n\n"
                f"**Roll trigger:** {thesis.get('roll_trigger_dte', doctrine_default_dte)} DTE\n\n"
                f"**Author / rev / updated:** {thesis.get('author', '—')} / r{thesis.get('revision', '?')} / {str(thesis.get('updated', ''))[:19]}\n"
            )
            st.markdown(wb)
            st.download_button(
                f"Download {sym} whiteboard.md",
                wb,
                file_name=f"{sym}_whiteboard.md",
                mime="text/markdown",
                key=f"wb_{sym}",
            )
            st.divider()


# ── Discipline Ledger ────────────────────────────────────────────────────────
def _render_discipline(metrics: dict[str, Any], dfv_state: dict[str, Any]) -> None:
    st.markdown("### ✊ Discipline Ledger")
    rules = (dfv_state["doctrine"].get("hard_rules") or [])
    theses = dfv_state["theses"]
    held = metrics["held_syms"]

    # Compute pass/fail per rule
    missing = metrics["missing_thesis"]
    stale = []
    for sym, rec in theses.items():
        try:
            updated = datetime.fromisoformat(rec.get("updated", ""))
            if (datetime.now(timezone.utc) - updated).days > 30:
                stale.append(sym)
        except (ValueError, TypeError):
            stale.append(sym)
    no_invalidation = [sym for sym in held if sym in theses and not (theses[sym].get("invalidation") or "").strip()]
    cash_ok = metrics["cash_pct"] >= 10

    checks = [
        (not missing, f"Every position has a thesis ({len(missing)} missing)"),
        (not no_invalidation, f"Every thesis has an invalidation ({len(no_invalidation)} blank)"),
        (not stale, f"No thesis older than 30 days ({len(stale)} stale)"),
        (cash_ok, f"Cash ≥ 10% of equity ({metrics['cash_pct']:.1f}%)"),
    ]
    for ok, label in checks:
        icon = "✅" if ok else "❌"
        color = DFV_GREEN if ok else DFV_RED
        st.markdown(
            f"<div style='color:{color};font-family:monospace'>{icon} {label}</div>",
            unsafe_allow_html=True,
        )

    if rules:
        with st.expander("🔒 Hard rules (never overridden)", expanded=False):
            for r in rules:
                st.markdown(f"- {r}")


# ── Seven Gates Scratchpad ───────────────────────────────────────────────────
def _render_seven_gates() -> None:
    st.markdown("### 🚦 Seven Gates Scratchpad")
    st.caption("Test any proposal. 60-second whiteboard rule.")

    with st.form("seven_gates_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        symbol = c1.text_input("Symbol", value="GME").upper()
        side = c2.selectbox("Side", ["long", "short"])
        size_pct = c3.number_input("Size % of book", 0.0, 1.0, 0.03, 0.005, format="%.3f")

        c4, c5, c6 = st.columns(3)
        portfolio_value = c4.number_input("Portfolio value ($)", 0.0, 1e9, 100_000.0, 1000.0)
        cash_after = c5.number_input("Cash AFTER trade ($)", 0.0, 1e9, 25_000.0, 500.0)
        slippage = c6.number_input("Expected slippage %", 0.0, 0.10, 0.005, 0.001, format="%.3f")

        c7, c8, c9 = st.columns(3)
        catalyst_days = c7.number_input("Catalyst within (days)", 0, 999, 7)
        catalyst_ack = c8.checkbox("Catalyst acknowledged", value=False)
        factor_pct = c9.number_input("Factor concentration after", 0.0, 1.0, 0.20, 0.05)

        submit = st.form_submit_button("🚦 Run Gates", type="primary")

    if not submit:
        return

    proposal = {
        "symbol": symbol,
        "action": "buy" if side == "long" else "sell",
        "side": side,
        "size_pct": float(size_pct),
        "expected_slippage_pct": float(slippage),
        "cash_after_trade": float(cash_after),
        "portfolio_value": float(portfolio_value),
        "factor_concentration_after": float(factor_pct),
        "catalyst_within_days": int(catalyst_days),
        "catalyst_acknowledged": bool(catalyst_ack),
    }
    try:
        decision = DFV().evaluate(proposal)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gate evaluation failed: {exc}")
        return

    verdict = decision.verdict
    color_map = {
        "approved": DFV_GREEN,
        "approved_with_notes": DFV_BLUE,
        "returned": DFV_YELLOW,
        "vetoed": DFV_RED,
    }
    color = color_map.get(verdict, DFV_YELLOW)
    st.markdown(
        f"<div style='font-size:1.6rem;color:{color};font-weight:bold;text-align:center;"
        f"border:2px solid {color};border-radius:8px;padding:10px'>"
        f"VERDICT: {verdict.upper().replace('_', ' ')}</div>",
        unsafe_allow_html=True,
    )
    if decision.summary:
        st.markdown(f"**60-second whiteboard:** {decision.summary}")
    for n in decision.notes:
        st.caption(f"• {n}")
    for fx in decision.fixes_required:
        st.warning(f"Fix required: {fx}")

    gate_rows = []
    for g in decision.gates:
        icon = "✅" if g.outcome == "pass" else ("❌" if g.severity == "hard" else "⚠️")
        gate_rows.append({
            "": icon,
            "gate": g.gate_id,
            "name": g.name,
            "outcome": g.outcome,
            "severity": g.severity,
            "note": g.note,
        })
    st.dataframe(gate_rows, hide_index=True, use_container_width=True)


# ── Decisions tail ───────────────────────────────────────────────────────────
def _render_recent_decisions(dfv_state: dict[str, Any]) -> None:
    st.markdown("### 📋 Recent Decisions")
    decisions = dfv_state["decisions"][-15:]
    if not decisions:
        st.caption("No decisions logged.")
        return
    rows = []
    so_icon_map = {"AGREE": "🟢", "DISAGREE": "🔴", "CONCERN": "🟡", "UNCLEAR": "⚪"}
    for d in reversed(decisions):
        dec = d.get("decision", {}) or {}
        prop = d.get("proposal", {}) or {}
        sym = prop.get("symbol") or dec.get("symbol") or "?"
        verdict = dec.get("verdict", "?")
        failed_gates = [g.get("gate_id") for g in (dec.get("gates") or []) if g.get("outcome") == "fail"]
        notes = dec.get("notes") or []
        summary = dec.get("summary") or (notes[0] if notes else "")
        icon_map = {"approved": "✅", "approved_with_notes": "🔵", "returned": "⚠️", "vetoed": "❌"}
        icon = icon_map.get(verdict, "❔")
        so = dec.get("second_opinion") or {}
        if so.get("ok"):
            # New jury shape: {majority, dissent_count, votes: {AGREE:..,DISAGREE:..}, panelists:[..]}
            if "majority" in so and isinstance(so.get("votes"), dict):
                maj = (so.get("majority") or "").upper()
                votes = so["votes"]
                tally = "-".join(str(votes.get(k, 0)) for k in ("AGREE", "CONCERN", "DISAGREE"))
                dissent = int(so.get("dissent_count") or 0)
                dissent_mark = f" ⚡{dissent}" if dissent > 0 else ""
                so_cell = f"{so_icon_map.get(maj, '⚪')} {maj} ({tally}){dissent_mark}"
            else:
                so_verdict = (so.get("verdict") or "").upper()
                so_cell = (f"{so_icon_map.get(so_verdict, '')} {so_verdict}".strip()) or "-"
        elif so:
            so_cell = "ERR"
        else:
            so_cell = "-"
        rows.append({
            "": icon,
            "ts": str(d.get("ts", ""))[:19],
            "symbol": sym,
            "size": f"{float(prop.get('size_pct') or 0)*100:.1f}%",
            "verdict": verdict,
            "killed by": ",".join(failed_gates) if failed_gates else "-",
            "2nd op": so_cell,
            "summary": summary[:80],
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)


# ── Post-mortems ─────────────────────────────────────────────────────────────
def _render_postmortems(dfv_state: dict[str, Any]) -> None:
    st.markdown("### 🪦 Post-mortems — what closed, what's the lesson")
    pms = dfv_state["postmortems"]
    if not pms:
        st.caption("No post-mortems on file. Every closed name should have one.")
        return
    for pm in reversed(pms):
        sym = pm.get("symbol", "?")
        status = pm.get("status", "?")
        rpnl = float(pm.get("realized_pnl_usd") or 0)
        color = DFV_GREEN if rpnl >= 0 else DFV_RED
        with st.expander(
            f"{sym} — {status} — ${rpnl:+,.0f} — {pm.get('expiry', '')}",
            expanded=False,
        ):
            st.markdown(f"**Thesis at open:** {pm.get('thesis_at_open', '—')}")
            st.markdown(f"**Invalidation:** {pm.get('invalidation_at_open', '—')}")
            st.markdown(f"**What happened:** {pm.get('what_happened', '—')}")
            st.markdown(
                f"<div style='color:{color};font-weight:bold'>Lesson: {pm.get('lesson', '—')}</div>",
                unsafe_allow_html=True,
            )


# ── Catalysts ≤ 5d ───────────────────────────────────────────────────────────
def _render_catalysts(payload: dict[str, Any], dfv_state: dict[str, Any]) -> None:
    st.markdown("### 📅 Catalyst calendar (30d)")
    alerts = payload.get("alerts") or []
    near: list[tuple[int, dict[str, Any]]] = []
    for a in alerts:
        days = a.get("days_until") if a.get("days_until") is not None else a.get("days_ahead")
        if days is None:
            continue
        try:
            d_int = int(days)
        except (ValueError, TypeError):
            continue
        if d_int <= 30:
            near.append((d_int, a))
    if not near:
        st.caption("No catalysts in the 30-day window. Quiet means quiet — don't manufacture a trade.")
        return
    near.sort(key=lambda x: x[0])
    for d, a in near:
        if d <= 5:
            bucket = f"🔴 T-{d}d"
        elif d <= 14:
            bucket = f"🟡 T-{d}d"
        else:
            bucket = f"⚪ T-{d}d"
        sym = a.get("symbol") or a.get("event") or "?"
        desc = a.get("description") or a.get("title") or ""
        st.markdown(f"- **{bucket}** · **{sym}**: {desc}")


# ── Roll alerts ──────────────────────────────────────────────────────────────
def _render_roll_alerts(metrics: dict[str, Any], dfv_state: dict[str, Any]) -> None:
    """Long options inside their roll-trigger DTE window. $659 postmortem says: act."""
    st.markdown("### ⏰ Roll / kill lane")
    st.caption("Long options inside roll_trigger_dte. Theta beats realized vol from here on.")
    today = datetime.now(timezone.utc).date()
    doctrine_default = int(
        ((dfv_state["doctrine"].get("roll_policy") or {}).get("default_trigger_dte") or 21)
    )
    theses = dfv_state["theses"]
    rows: list[dict[str, Any]] = []
    for p in metrics["positions"]:
        exp = str(p.get("expiry") or "")
        if not exp:
            continue
        try:
            exp_d = datetime.fromisoformat(exp[:10]).date()
        except (ValueError, TypeError):
            continue
        dte = (exp_d - today).days
        sym = p["symbol"]
        trigger = int((theses.get(sym) or {}).get("roll_trigger_dte") or doctrine_default)
        if dte > trigger:
            continue
        rows.append({
            "symbol": sym,
            "account": p["account"],
            "qty": p["qty"],
            "expiry": exp[:10],
            "DTE": dte,
            "trigger_DTE": trigger,
            "Θ": (f"{float(p['theta']):+.2f}" if isinstance(p.get("theta"), (int, float)) else "-"),
            "unrealized": f"{float(p.get('unrealized') or 0):+,.0f}",
            "status": "EXPIRED" if dte < 0 else ("TODAY" if dte == 0 else "ROLL/KILL"),
        })
    if not rows:
        st.caption("Clean. No options inside the roll window.")
        return
    rows.sort(key=lambda r: (r["DTE"], r["symbol"]))
    st.dataframe(rows, hide_index=True, use_container_width=True)


# ── Reconciliation panel ─────────────────────────────────────────────────────
def _render_reconciliation(metrics: dict[str, Any], dfv_state: dict[str, Any]) -> None:
    st.markdown("### 🔄 Reconciliation — theses ↔ positions")
    st.caption("Per venue. Every thesis should map to a position; every position should map to a thesis.")
    theses = set(dfv_state["theses"].keys())
    held = metrics["held_syms"]
    by_venue: dict[str, set[str]] = {}
    for p in metrics["positions"]:
        by_venue.setdefault(p["account"], set()).add(p["symbol"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Theses without a position**")
        orphans = sorted(theses - held)
        if not orphans:
            st.caption("None. Every thesis is in the book.")
        else:
            for s in orphans:
                st.markdown(f"- `{s}`")
    with c2:
        st.markdown("**Positions without a thesis (by venue)**")
        any_missing = False
        for venue, syms in sorted(by_venue.items()):
            missing_at_venue = sorted(s for s in syms if s not in theses)
            if not missing_at_venue:
                continue
            any_missing = True
            st.markdown(f"- **{venue}**: {', '.join(missing_at_venue)}")
        if not any_missing:
            st.caption("None. Every position has a thesis.")

    with st.expander("Per-venue position counts", expanded=False):
        for venue, syms in sorted(by_venue.items()):
            st.markdown(f"- **{venue}** · {len(syms)} symbol(s): {', '.join(sorted(syms))}")


# ── Postmortem clusters ──────────────────────────────────────────────────────
def _render_postmortem_clusters(dfv_state: dict[str, Any]) -> None:
    """Group postmortems by tag. Surface the $659 macro_hedge cluster."""
    st.markdown("### 🧱 Lesson clusters")
    pms = dfv_state["postmortems"]
    if not pms:
        st.caption("No postmortems yet.")
        return
    by_tag: dict[str, list[tuple[str, float]]] = {}
    for pm in pms:
        tags = pm.get("tags") or ["untagged"]
        rpnl = float(pm.get("realized_pnl_usd") or 0)
        sym = pm.get("symbol", "?")
        for t in tags:
            by_tag.setdefault(t, []).append((sym, rpnl))
    clusters = sorted(by_tag.items(), key=lambda kv: sum(p for _, p in kv[1]))
    surfaced = False
    for tag, items in clusters:
        n = len(items)
        if n < 2:
            continue
        total = sum(p for _, p in items)
        color = DFV_RED if total < 0 else DFV_GREEN
        st.markdown(
            f"<div style='color:{color};font-family:monospace'>"
            f"<b>{tag}</b> — {n} trade(s), realized <b>${total:+,.0f}</b> · "
            f"{', '.join(s for s, _ in items)}</div>",
            unsafe_allow_html=True,
        )
        surfaced = True
    if not surfaced:
        st.caption("No clusters yet (need ≥2 postmortems sharing a tag).")


# ── Conviction nudges ────────────────────────────────────────────────────────
def _render_conviction_nudges() -> None:
    st.markdown("### 📉 Recent conviction nudges")
    path = ROOT / "agents" / "dfv" / "memory" / "conviction_nudges.jsonl"
    if not path.exists():
        st.caption("No nudges logged. EOD will write here when it drops a tier.")
        return
    lines = path.read_text(encoding="utf-8").splitlines()[-20:]
    rows: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        st.caption("No nudges logged.")
        return
    table = [{
        "ts": str(r.get("ts", ""))[:19],
        "symbol": r.get("symbol", "?"),
        "from": r.get("from"),
        "to": r.get("to"),
        "reason": (r.get("reason") or "")[:80],
        "source": r.get("source", "—"),
    } for r in rows]
    st.dataframe(table, hide_index=True, use_container_width=True)


# ── Discipline streak ────────────────────────────────────────────────────────
def _discipline_streak_days(dfv_state: dict[str, Any]) -> int:
    """Consecutive days (back from today UTC) with zero vetoed/returned decisions."""
    from datetime import timedelta
    decisions = dfv_state["decisions"]
    bad_by_day: dict[str, int] = {}
    for d in decisions:
        ts = str(d.get("ts", ""))
        if len(ts) < 10:
            continue
        day = ts[:10]
        verdict = ((d.get("decision") or {}).get("verdict") or "").lower()
        if verdict in ("vetoed", "returned"):
            bad_by_day[day] = bad_by_day.get(day, 0) + 1
    today = datetime.now(timezone.utc).date()
    streak = 0
    while True:
        day_key = (today - timedelta(days=streak)).isoformat()
        if bad_by_day.get(day_key, 0) > 0:
            break
        streak += 1
        if streak > 365:
            break
    return streak


# ── Reconciler banner (live drift detection from agents.dfv.reconciler) ──────
def _render_reconciler_banner() -> None:
    try:
        snap = DFV().reconciliation.read()
    except Exception:  # noqa: BLE001
        return
    if not snap:
        return
    mm = int(snap.get("mismatch_count") or 0)
    if mm <= 0:
        return
    diffs = snap.get("mismatches") or []
    detail = "; ".join(
        f"{d.get('kind', '?')}: {d.get('symbol', '')}" for d in diffs[:4]
    )
    st.markdown(
        f"<div style='background:#3a0d0d;border:2px solid {DFV_RED};border-radius:6px;"
        f"padding:10px;color:{DFV_RED};font-weight:bold'>"
        f"🚨 RECONCILIATION DRIFT — {mm} mismatch(es). {detail}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Thesis write-back editor ─────────────────────────────────────────────────
def _render_thesis_editor(dfv_state: dict[str, Any]) -> None:
    theses = dfv_state["theses"]
    if not theses:
        st.caption("No theses on file yet.")
        return
    st.markdown("#### ✏️ Edit thesis (write-back)")
    sym = st.selectbox(
        "Symbol",
        sorted(theses.keys()),
        key="dfv_thesis_edit_sym",
    )
    if not sym:
        return
    rec = theses[sym] or {}
    with st.form(f"thesis_edit_{sym}"):
        thesis_txt = st.text_area("Thesis", value=rec.get("thesis", ""), height=80)
        invalidation = st.text_area("Invalidation", value=rec.get("invalidation", ""), height=60)
        c1, c2 = st.columns(2)
        conviction = c1.number_input(
            "Conviction (0-3)",
            min_value=0, max_value=3,
            value=int(rec.get("conviction", 0) or 0),
        )
        max_pct = c2.number_input(
            "Max % of book",
            min_value=0.0, max_value=1.0,
            value=float((rec.get("sizing") or {}).get("max_pct_book", 0.0) or 0.0),
            step=0.005, format="%.3f",
        )
        catalysts = st.text_input(
            "Catalysts (comma-separated)",
            value=", ".join(rec.get("catalysts") or []),
        )
        tags = st.text_input(
            "Tags (comma-separated)",
            value=", ".join(rec.get("tags") or []),
        )
        submit = st.form_submit_button("💾 Save")
    if not submit:
        return
    try:
        DFV().thesis.set(
            symbol=sym,
            thesis=thesis_txt.strip(),
            invalidation=invalidation.strip(),
            conviction=int(conviction),
            horizon=rec.get("horizon", "TBD"),
            catalysts=[c.strip() for c in catalysts.split(",") if c.strip()],
            target=rec.get("target") or {},
            sizing={**(rec.get("sizing") or {}), "max_pct_book": float(max_pct)},
            tags=[t.strip() for t in tags.split(",") if t.strip()],
        )
        st.cache_data.clear()
        st.success(f"Thesis updated for {sym}.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Save failed: {exc}")


# ── Journal panel ────────────────────────────────────────────────────────────
def _render_journal() -> None:
    st.markdown("### 📓 Journal")
    try:
        dfv = DFV()
        recent = dfv.journal.tail(5)
        has_today = dfv.journal.has_today()
    except Exception as exc:  # noqa: BLE001
        st.error(f"journal unavailable: {exc}")
        return
    if not has_today:
        st.warning("No entry today. Compounding insight requires writing it down.")
    with st.form("dfv_journal_form", clear_on_submit=True):
        entry = st.text_area("What did you learn today?", height=100)
        c1, c2 = st.columns(2)
        mood = c1.selectbox("Mood", ["", "calm", "tilted", "patient", "greedy", "fearful"])
        tags = c2.text_input("Tags (comma-separated)")
        submit = st.form_submit_button("📝 Save entry")
    if submit and entry.strip():
        try:
            DFV().journal.append(
                entry=entry.strip(),
                mood=mood,
                tags=[t.strip() for t in tags.split(",") if t.strip()],
            )
            st.cache_data.clear()
            st.success("Entry saved.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Save failed: {exc}")
    if recent:
        for e in reversed(recent):
            st.markdown(
                f"<div style='font-family:monospace;font-size:0.85rem;color:{DFV_BLUE}'>"
                f"<b>{str(e.get('ts', ''))[:16]}</b> · "
                f"<i>{e.get('mood') or '-'}</i> · "
                f"{', '.join(e.get('tags') or []) or '—'}<br>"
                f"{e.get('entry', '')}</div>",
                unsafe_allow_html=True,
            )


# ── P&L attribution panel ────────────────────────────────────────────────────
def _render_pnl_attribution() -> None:
    st.markdown("### 💵 P&L attribution — by tag · tier · verdict")
    try:
        from agents.dfv.pnl_attribution import attribute  # noqa: PLC0415
        data = attribute()
    except Exception as exc:  # noqa: BLE001
        st.error(f"P&L attribution failed: {exc}")
        return
    st.caption(f"Total realized across {data.get('total', {}).get('n', 0)} postmortems: "
               f"${data.get('total', {}).get('realized', 0):+,.0f}")
    cols = st.columns(3)
    for col, key, label in zip(cols, ("by_tag", "by_tier", "by_verdict"),
                                ("Tag", "Conviction tier", "Verdict at open")):
        with col:
            st.markdown(f"**By {label}**")
            buckets = data.get(key) or {}
            if not buckets:
                st.caption("—")
                continue
            rows = []
            for k, v in sorted(buckets.items(), key=lambda kv: kv[1].get("realized", 0)):
                rows.append({
                    "bucket": str(k),
                    "n": v.get("n", 0),
                    "realized": f"{v.get('realized', 0):+,.0f}",
                    "hit_rate": f"{v.get('hit_rate', 0):.0%}",
                })
            st.dataframe(rows, hide_index=True, use_container_width=True)


# ── War-room composite 30d sparkline ─────────────────────────────────────────
def _render_war_room_trend() -> None:
    st.markdown("### 📈 War-room composite (30d)")
    briefs_dir = ROOT / "agents" / "dfv" / "memory" / "briefs"
    if not briefs_dir.exists():
        st.caption("No briefs yet.")
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    series: list[dict[str, Any]] = []
    for p in sorted(briefs_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ts = d.get("generated_at") or ""
        try:
            ts_dt = datetime.fromisoformat(ts)
        except ValueError:
            continue
        if ts_dt < cutoff:
            continue
        comp = ((d.get("war_room") or {}).get("composite"))
        if comp is None:
            continue
        try:
            series.append({"ts": ts_dt, "composite": float(comp)})
        except (TypeError, ValueError):
            continue
    if not series:
        st.caption("Briefs lack a composite_score — sparkline empty.")
        return
    series.sort(key=lambda r: r["ts"])
    chart_data = {r["ts"].isoformat(): r["composite"] for r in series}
    try:
        st.line_chart(chart_data, use_container_width=True)
    except Exception:  # noqa: BLE001
        # Fallback: render as table
        st.dataframe(
            [{"ts": k[:16], "composite": v} for k, v in chart_data.items()],
            hide_index=True, use_container_width=True,
        )


# ── Roll-or-kill action button (proposes only, never executes) ───────────────
def _render_roll_or_kill_buttons(metrics: dict[str, Any]) -> None:
    st.markdown("#### 🎯 Quote roll / kill (propose-only)")
    st.caption("Click to draft a roll ticket. The order is NEVER sent. Human-in-loop required.")
    today = datetime.now(timezone.utc).date()
    candidates: list[dict[str, Any]] = []
    for p in metrics["positions"]:
        exp_s = str(p.get("expiry") or "")
        if not exp_s:
            continue
        try:
            exp_d = datetime.fromisoformat(exp_s[:10]).date()
        except (ValueError, TypeError):
            continue
        dte = (exp_d - today).days
        if dte > 25:
            continue
        candidates.append({**p, "_dte": dte})
    if not candidates:
        st.caption("No eligible positions in roll window.")
        return
    for p in candidates[:10]:
        sym = p["symbol"]
        key = f"roll_btn_{sym}_{p.get('expiry')}_{p.get('strike')}"
        cols = st.columns([3, 1])
        cols[0].markdown(
            f"`{sym}` · {p.get('account')} · {p.get('expiry', '')[:10]} · "
            f"strike ${p.get('strike') or '?'} · DTE {p['_dte']}"
        )
        if cols[1].button("Quote roll", key=key):
            try:
                from agents.dfv.roll_engine import quote_and_review  # noqa: PLC0415
                ticket = quote_and_review({
                    "symbol": sym,
                    "strike": p.get("strike"),
                    "expiry": p.get("expiry"),
                    "side": p.get("right") or p.get("side") or "put",
                    "quantity": p.get("qty"),
                })
                gd = ticket.get("gate_decision") or {}
                verdict = gd.get("verdict", "?")
                st.success(
                    f"Ticket drafted for {sym}: {ticket['rationale']} "
                    f"· Verdict: {verdict} · Status: {ticket['status']}"
                )
                st.caption("Pending operator OK — autonomy.trade_execution=human_in_loop")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Roll proposal failed: {exc}")


# ── /ask box ─────────────────────────────────────────────────────────────────
def _render_ask_box() -> None:
    st.markdown("**💬 /ask DFV**")
    q = st.text_input("Ask DFV (uses LLM + RAG over briefs/theses)", key="dfv_ask")
    if not q:
        return
    if not st.button("Ask", key="dfv_ask_btn"):
        return
    with st.spinner("DFV thinking…"):
        try:
            from agents.dfv import llm as dfv_llm
            answer = dfv_llm.ask(q)  # type: ignore[attr-defined]
            st.markdown(answer if isinstance(answer, str) else json.dumps(answer, default=str, indent=2))
        except AttributeError:
            try:
                dec = DFV().review_prompt(q)
                st.markdown(f"**Verdict:** {dec.verdict}\n\n{dec.summary}")
                for n in dec.notes:
                    st.caption(f"• {n}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"/ask failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"/ask failed: {exc}")


# ── Watchlist ────────────────────────────────────────────────────────────────
# ── GME / meme YOLO row ──────────────────────────────────────────────────────
_MEME_UNIVERSE = {"GME", "AMC", "BB", "BBBY", "KOSS", "NOK", "PLTR", "DJT", "TSLA"}
_MEME_TAGS = {"meme", "yolo", "wsb", "retail"}


def _meme_symbols(dfv_state: dict[str, Any]) -> list[str]:
    """Union of tagged-meme theses + classic meme universe present in any DFV memory."""
    out: set[str] = set()
    for sym, t in (dfv_state.get("theses") or {}).items():
        tags = {str(x).lower() for x in (t.get("tags") or [])}
        if tags & _MEME_TAGS or sym.upper() in _MEME_UNIVERSE:
            out.add(sym.upper())
    for sym in (dfv_state.get("watchlist") or {}):
        if sym.upper() in _MEME_UNIVERSE:
            out.add(sym.upper())
    return sorted(out)


def _render_meme_row(payload: dict[str, Any], dfv_state: dict[str, Any]) -> None:
    """The DFV special: meme tickers with UW flow + X velocity if available.

    Pulls live signal from payload['uw'] / payload['indicators'] if the collector
    populated them; otherwise shows '—' (no fake data — rule #4: explain in 60s).
    """
    symbols = _meme_symbols(dfv_state)
    if not symbols:
        return  # silent — don't manufacture meme content

    st.markdown("### 🚀 Meme / YOLO row — where the retail flow is")

    uw_overview = (payload.get("uw") or {}).get("overview") or {}
    indicators = payload.get("indicators") or {}
    x_block = indicators.get("x_sentiment") or indicators.get("retail_pulse") or {}
    x_by_sym: dict[str, Any] = {}
    if isinstance(x_block, dict):
        x_by_sym = x_block.get("by_symbol") or {}

    theses = dfv_state.get("theses") or {}
    conviction = dfv_state.get("conviction") or {}

    rows: list[dict[str, Any]] = []
    for sym in symbols:
        t = theses.get(sym) or {}
        conv = conviction.get(sym) or {}
        uw_sym = uw_overview.get(sym) or {}
        x_sym = x_by_sym.get(sym) or {}
        flow = uw_sym.get("net_premium") or uw_sym.get("flow_score")
        flow_cell = f"${float(flow):,.0f}" if isinstance(flow, (int, float)) else "—"
        oi_change = uw_sym.get("oi_change_pct") or uw_sym.get("hottest_chain")
        oi_cell = (
            f"{float(oi_change):+.1f}%"
            if isinstance(oi_change, (int, float))
            else (str(oi_change)[:24] if oi_change else "—")
        )
        x_vel = x_sym.get("velocity") or x_sym.get("mentions_24h")
        x_cell = f"{int(x_vel)}/h" if isinstance(x_vel, (int, float)) else "—"
        tier = (conv.get("tier") or t.get("conviction") or "—")
        thesis_summary = (t.get("summary") or t.get("thesis") or "")[:60]
        rows.append({
            "symbol": sym,
            "tier": tier,
            "UW flow": flow_cell,
            "OI Δ": oi_cell,
            "X velocity": x_cell,
            "thesis": thesis_summary or "(no thesis on file)",
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption(
        "Flow & velocity sourced from collector payload. '—' = collector didn't populate it; "
        "size accordingly. No FOMO entries from this row — rule #5."
    )


def _render_watchlist(dfv_state: dict[str, Any]) -> None:
    st.markdown("### 👁 Watchlist")
    wl = dfv_state["watchlist"]
    if not wl:
        st.caption("Empty. Add via `dfv watch add SYM --reason '…'`")
        return
    rows = [
        {
            "symbol": s,
            "added": str(rec.get("added", ""))[:10],
            "reason": rec.get("reason", "")[:80],
            "source": rec.get("source", "—"),
        }
        for s, rec in sorted(wl.items())
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
def _render_sidebar(dfv_state: dict[str, Any]) -> None:
    with st.sidebar:
        st.markdown("# 🐱 DFV")
        st.caption("Roaring Kitty operator. Cash is a position.")
        st.divider()

        if st.button("🔄 Refresh now"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.markdown("**Run a routine**")
        c1, c2 = st.columns(2)
        if c1.button("Brief"):
            with st.spinner("Briefing…"):
                try:
                    dfv_routines.brief()
                    st.cache_data.clear()
                    st.success("Brief written.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"brief() failed: {exc}")
        if c2.button("EOD"):
            with st.spinner("EOD…"):
                try:
                    dfv_routines.eod()
                    st.cache_data.clear()
                    st.success("EOD written.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"eod() failed: {exc}")

        st.divider()
        autonomy = (dfv_state["doctrine"].get("autonomy") or {})
        trade_exec = autonomy.get("trade_execution", "?")
        color = DFV_GREEN if trade_exec == "human_in_loop" else DFV_RED
        st.markdown(
            f"**Autonomy** &middot; trade_execution<br>"
            f"<span style='color:{color};font-weight:bold'>{trade_exec}</span>",
            unsafe_allow_html=True,
        )
        st.caption("Switch in `config/doctrine/dfv_doctrine.yaml`. Requires DFV-AUTONOMY-CHANGE marker.")

        st.divider()
        st.markdown("**Conviction tiers**")
        conv = dfv_state["doctrine"].get("conviction") or {}
        for tier in sorted(conv.keys(), reverse=True):
            spec = conv[tier]
            st.caption(f"T{tier} — {spec.get('label', '')} (max {spec.get('max_pct_book', 0)*100:.0f}%)")

        st.divider()
        streak = _discipline_streak_days(dfv_state)
        streak_color = DFV_GREEN if streak >= 7 else (DFV_YELLOW if streak >= 1 else DFV_RED)
        st.markdown(
            f"**Discipline streak**<br>"
            f"<span style='font-size:1.4rem;color:{streak_color};font-weight:bold'>{streak} day(s)</span>",
            unsafe_allow_html=True,
        )
        st.caption("Consecutive days with zero vetoed / returned decisions.")

        st.divider()
        _render_ask_box()


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    payload = _load_payload()
    if "_error" in payload:
        st.error(f"mission_control.collect_payload failed: {payload['_error']}")
        st.caption("DFV runs blind without payload. Fix collectors before trusting any number on this page.")
    dfv_state = _load_dfv_state()

    _render_sidebar(dfv_state)
    _render_reconciler_banner()
    _render_traffic_light(payload, dfv_state)
    _render_catalyst_countdown(payload, dfv_state)
    _render_headline(payload, dfv_state)
    st.divider()

    metrics = _render_metrics_strip(payload, dfv_state)
    st.divider()

    _render_the_book(metrics, dfv_state)
    _render_thesis_editor(dfv_state)
    st.divider()

    # Roll lane + reconciliation — the two failure modes that cost real money.
    col_r1, col_r2 = st.columns([3, 2])
    with col_r1:
        _render_roll_alerts(metrics, dfv_state)
        _render_roll_or_kill_buttons(metrics)
    with col_r2:
        _render_discipline(metrics, dfv_state)
    st.divider()

    _render_reconciliation(metrics, dfv_state)
    st.divider()

    _render_pnl_attribution()
    st.divider()

    col_left, col_right = st.columns([3, 2])
    with col_left:
        _render_recent_decisions(dfv_state)
        with st.expander("🚦 Seven-Gates Scratchpad (manual proposal test)", expanded=False):
            _render_seven_gates()
    with col_right:
        _render_catalysts(payload, dfv_state)
        st.markdown("")
        _render_conviction_nudges()
        st.markdown("")
        _render_journal()

    st.divider()
    col_p, col_c = st.columns(2)
    with col_p:
        _render_postmortems(dfv_state)
    with col_c:
        _render_postmortem_clusters(dfv_state)

    st.divider()
    _render_war_room_trend()

    st.divider()
    _render_watchlist(dfv_state)

    st.divider()
    _render_meme_row(payload, dfv_state)

    st.caption(
        f"🐱 DFV Dashboard · payload ts {payload.get('ts', '—')} · "
        "doctrine: config/doctrine/dfv_doctrine.yaml · "
        "memory: agents/dfv/memory/"
    )


if __name__ == "__main__":
    main()
