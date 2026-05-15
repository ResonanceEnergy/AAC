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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.dfv import routines as dfv_routines  # noqa: E402
from agents.dfv.daemon import heartbeat_status  # noqa: E402
from agents.dfv.decision_engine import DFV  # noqa: E402

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
    return {
        "theses": dfv.thesis.all(),
        "conviction": dfv.conviction.all(),
        "watchlist": dfv.watchlist.all(),
        "decisions": dfv.decisions.tail(50),
        "postmortems": dfv.postmortems.all()[-10:],
        "heartbeat": _safe_heartbeat(),
        "doctrine": dfv.doctrine,
    }


def _safe_heartbeat() -> dict[str, Any]:
    try:
        return heartbeat_status() or {}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "status": "unknown"}


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
            rows.append({
                "symbol": sym,
                "account": acct_name,
                "qty": p.get("quantity") or p.get("qty") or 0,
                "side": p.get("side") or p.get("direction") or "-",
                "last_price": p.get("last_price") or p.get("mark_price") or p.get("price") or 0,
                "avg_cost": p.get("avg_cost") or p.get("cost_basis") or 0,
                "unrealized": p.get("unrealized_pnl") or p.get("pnl_unrealized") or 0,
                "expiry": p.get("expiry") or p.get("expiration") or "",
                "asset_type": p.get("asset_type") or p.get("type") or "stock",
            })
    return rows


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
        status = (hb.get("status") or "unknown").lower()
        color = DFV_GREEN if status in ("alive", "running", "ok") else DFV_RED if status == "dead" else DFV_YELLOW
        last = str(hb.get("last_beat", "—"))[-8:] if hb.get("last_beat") else "—"
        st.markdown(
            f"<div style='text-align:center;background:{color}22;padding:10px;border-radius:6px;'>"
            f"<div style='font-size:0.75rem;color:#888'>DAEMON</div>"
            f"<div style='font-size:1.1rem;color:{color};font-weight:bold'>{status.upper()}</div>"
            f"<div style='font-size:0.7rem;color:#aaa'>last beat {last}</div></div>",
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

    cols = st.columns(7)
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

    return {
        "equity": equity,
        "cash": cash,
        "cash_pct": cash_pct,
        "positions": positions,
        "missing_thesis": missing,
        "held_syms": held_syms,
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
        dte = ""
        if p.get("expiry"):
            try:
                exp = datetime.fromisoformat(str(p["expiry"])[:10]).date()
                dte = max(0, (exp - today).days)
                if dte == 0:
                    dte = "TODAY 🔔"
            except (ValueError, TypeError):
                pass

        thesis_status = "✅" if sym in theses else "❌ MISSING"
        conviction = conviction_map.get(sym, thesis.get("conviction", 0))

        rows.append({
            "🚨": breach,
            "symbol": sym,
            "side": p.get("side", "-"),
            "qty": p.get("qty"),
            "last": f"{last:.2f}" if last else "-",
            "avg": f"{float(p.get('avg_cost') or 0):.2f}",
            "unrealized": f"{float(p.get('unrealized') or 0):+,.0f}",
            "DTE": dte if dte != "" else "-",
            "thesis": thesis_status,
            "conviction": conviction or "-",
            "invalidation": (invalidation[:50] + "…") if len(invalidation) > 50 else (invalidation or "—"),
            "account": p.get("account", "-"),
        })

    # Sort: breaches first, then missing thesis, then by symbol
    rows.sort(key=lambda r: (r["🚨"] != "🚨", "MISSING" not in r["thesis"], r["symbol"]))
    st.dataframe(rows, hide_index=True, use_container_width=True)

    if metrics["missing_thesis"]:
        st.error(
            f"❌ **HARD RULE #1 VIOLATED** — held without a thesis: "
            f"{', '.join(metrics['missing_thesis'])}. Write one or close it."
        )


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
        rows.append({
            "": icon,
            "ts": str(d.get("ts", ""))[:19],
            "symbol": sym,
            "size": f"{float(prop.get('size_pct') or 0)*100:.1f}%",
            "verdict": verdict,
            "killed by": ",".join(failed_gates) if failed_gates else "-",
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
    st.markdown("### 📅 Catalysts ≤ 5 days")
    alerts = payload.get("alerts") or []
    near = []
    for a in alerts:
        days = a.get("days_until") or a.get("days_ahead")
        if days is None:
            continue
        try:
            if int(days) <= 5:
                near.append(a)
        except (ValueError, TypeError):
            continue
    if not near:
        st.caption("No catalysts in the 5-day window. Quiet means quiet — don't manufacture a trade.")
        return
    for a in sorted(near, key=lambda x: int(x.get("days_until") or x.get("days_ahead") or 999)):
        st.markdown(
            f"- **T-{a.get('days_until', a.get('days_ahead'))}d** · "
            f"{a.get('symbol', a.get('event', '?'))}: {a.get('description', a.get('title', ''))}"
        )


# ── Watchlist ────────────────────────────────────────────────────────────────
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


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    payload = _load_payload()
    if "_error" in payload:
        st.error(f"mission_control.collect_payload failed: {payload['_error']}")
        st.caption("DFV runs blind without payload. Fix collectors before trusting any number on this page.")
    dfv_state = _load_dfv_state()

    _render_sidebar(dfv_state)
    _render_headline(payload, dfv_state)
    st.divider()

    metrics = _render_metrics_strip(payload, dfv_state)
    st.divider()

    _render_the_book(metrics, dfv_state)
    st.divider()

    col_left, col_right = st.columns([3, 2])
    with col_left:
        _render_seven_gates()
    with col_right:
        _render_discipline(metrics, dfv_state)
        st.markdown("")
        _render_catalysts(payload, dfv_state)

    st.divider()
    col_d, col_p = st.columns(2)
    with col_d:
        _render_recent_decisions(dfv_state)
    with col_p:
        _render_postmortems(dfv_state)

    st.divider()
    _render_watchlist(dfv_state)

    st.caption(
        f"🐱 DFV Dashboard · payload ts {payload.get('ts', '—')} · "
        "doctrine: config/doctrine/dfv_doctrine.yaml · "
        "memory: agents/dfv/memory/"
    )


if __name__ == "__main__":
    main()
