from __future__ import annotations

import asyncio
import datetime as dt
import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import structlog

_log = structlog.get_logger()
_THIS_FILE = Path(__file__).resolve()

_COLLECTOR_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    ImportError,
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    return str(obj)


# ── Cached collectors ───────────────────────────────────────────────────────


_CACHED_COLLECT: Callable[[str], dict[str, Any]] | None = None
_CACHED_PILLAR: Callable[..., dict[str, Any]] | None = None


def _get_cached_pillar() -> Callable[..., dict[str, Any]]:
    """Cache wrapper for the three pillar collectors. Longer TTL (5 min)
    because each pillar makes multiple yfinance calls."""
    global _CACHED_PILLAR
    if _CACHED_PILLAR is not None:
        return _CACHED_PILLAR
    import streamlit as st

    @st.cache_data(ttl=300, show_spinner=False)
    def _cached_pillar(name: str, key: str = "") -> dict[str, Any]:  # noqa: ARG001
        try:
            from monitoring import dashboard_pillars as dp
        except ImportError as exc:
            return {"error": f"pillar_bootstrap_failed: {exc}"}
        try:
            if name == "call_options":
                portfolio = st.session_state.get("__portfolio_payload__", {})
                own_calls = [
                    p for p in _flatten_positions(portfolio)
                    if str(p.get("type", "")).lower() == "call"
                ]
                return dp.collect_call_options(own_call_positions=own_calls)
            if name == "index_flow":
                uw = st.session_state.get("__uw_payload__", {})
                return dp.collect_index_flow(uw_payload=uw)
            if name == "quant_research":
                return dp.collect_quant_research(run_walk_forward=False)
            if name == "quant_walk_forward":
                ticker = st.session_state.get("__wf_ticker__", "SPY")
                return dp.collect_quant_research(backtest_ticker=ticker, run_walk_forward=True)
        except _COLLECTOR_ERRORS as exc:
            _log.warning("pillar_collector_failed", pillar=name, error=str(exc))
            return {"error": str(exc)}
        return {"error": f"unknown_pillar:{name}"}

    _CACHED_PILLAR = _cached_pillar
    return _cached_pillar


def _get_cached_collect() -> Callable[[str], dict[str, Any]]:
    """Return a single ``@st.cache_data`` function keyed by collector name.

    A single function with ``name: str`` as an argument means streamlit's
    cache key includes the name, so each collector gets its own slot
    (avoids the 17-closures-share-one-cache-slot bug).
    """
    global _CACHED_COLLECT
    if _CACHED_COLLECT is not None:
        return _CACHED_COLLECT
    import streamlit as st

    @st.cache_data(ttl=30, show_spinner=False)
    def _cached_collect(name: str) -> dict[str, Any]:
        try:
            from monitoring import mission_control as mc
        except ImportError as exc:
            _log.warning("collector_bootstrap_failed", collector=name, error=str(exc))
            return {"error": f"bootstrap_failed: {exc}"}
        fn = getattr(mc, f"collect_{name}", None)
        if fn is None:
            return {"error": f"collector_missing:collect_{name}"}
        try:
            result = fn()
        except _COLLECTOR_ERRORS as exc:
            _log.warning("collector_failed", collector=name, error=str(exc))
            return {"error": str(exc)}
        return result if isinstance(result, dict) else {"value": result}

    _CACHED_COLLECT = _cached_collect
    return _cached_collect


# ── Helpers ─────────────────────────────────────────────────────────────────


def _safe_get(payload: dict[str, Any], key: str) -> dict[str, Any]:
    val = payload.get(key)
    return val if isinstance(val, dict) else {}


def _fmt_usd(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"${v:,.2f}"
    return "-"


def _fmt_num(v: Any, digits: int = 2) -> str:
    if isinstance(v, (int, float)):
        return f"{v:,.{digits}f}"
    return "-"


def _fmt_pct(v: Any, digits: int = 1) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.{digits}f}%"
    return "-"


def _derive_health_status(health: dict[str, Any]) -> tuple[str, int, int]:
    subs = health.get("subsystems") or {}
    if not isinstance(subs, dict) or not subs:
        return "unknown", 0, 0
    bad = {"unavailable", "no_key", "error"}
    total = len(subs)
    ok = sum(1 for v in subs.values() if str(v).split(" ")[0] not in bad)
    if ok == total:
        return "healthy", ok, total
    if ok >= total * 0.6:
        return "degraded", ok, total
    return "down", ok, total


def _flatten_positions(portfolio: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for acct in portfolio.get("accounts") or []:
        name = acct.get("name", "?")
        for pos in acct.get("positions") or []:
            rows.append(
                {
                    "account": name,
                    "symbol": pos.get("symbol", "?"),
                    "type": pos.get("type", "?"),
                    "strike": pos.get("strike"),
                    "expiry": pos.get("expiry", ""),
                    "qty": pos.get("qty", 0),
                    "avg_cost": pos.get("avg_cost", 0),
                    "market_price": pos.get("market_price", 0),
                    "market_value": pos.get("market_value", 0),
                    "unrealized_pnl": pos.get("unrealized_pnl", 0),
                }
            )
    return rows


def _accounts_summary(portfolio: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for acct in portfolio.get("accounts") or []:
        rows.append(
            {
                "account": acct.get("name", "?"),
                "platform": acct.get("platform", ""),
                "currency": acct.get("currency", ""),
                "total_assets": acct.get("total_assets", 0),
                "value_usd": acct.get("value_usd", 0),
                "positions": acct.get("position_count", 0),
                "unrealized_pnl": acct.get("unrealized_pnl", 0),
                "verified": acct.get("verified", ""),
                "days_stale": acct.get("days_stale", 0),
            }
        )
    return rows


# ── Tab renderers ───────────────────────────────────────────────────────────


def _render_overview(payload: dict[str, Any]) -> None:
    import streamlit as st

    portfolio = _safe_get(payload, "portfolio")
    health = _safe_get(payload, "health")
    pnl = _safe_get(payload, "pnl")
    calls = _safe_get(payload, "call_options")
    flow = _safe_get(payload, "index_flow")
    quant = _safe_get(payload, "quant_research")

    pnl_summary = pnl.get("summary") if isinstance(pnl.get("summary"), dict) else {}
    status, ok, total = _derive_health_status(health)

    # ---------- Top KPI strip ----------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Portfolio (USD)", _fmt_usd(portfolio.get("total_usd")))
    k2.metric("Unrealized P&L", _fmt_usd(portfolio.get("total_unrealized_pnl")))
    k3.metric("Realized P&L", _fmt_usd(pnl_summary.get("total_realized_pnl")))
    k4.metric("Positions", str(portfolio.get("total_positions", 0)))
    k5.metric("Health", f"{status}", delta=f"{ok}/{total} checks")

    st.divider()

    # ---------- Three pillar status cards ----------
    own = calls.get("own_call_summary") if isinstance(calls.get("own_call_summary"), dict) else {}
    of = flow.get("options_flow") if isinstance(flow.get("options_flow"), dict) else {}
    vp_signals = quant.get("vol_premium_signals") if isinstance(quant.get("vol_premium_signals"), list) else []
    bt = quant.get("simple_backtest") if isinstance(quant.get("simple_backtest"), dict) else {}
    bt_strategies = bt.get("strategies") if isinstance(bt.get("strategies"), list) else []
    best_bt: dict[str, Any] | None = None
    best_wr = -1.0
    for sval in bt_strategies:
        if isinstance(sval, dict) and isinstance(sval.get("win_rate"), (int, float)):
            wr = float(sval["win_rate"])
            if wr > best_wr:
                best_wr = wr
                best_bt = sval

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("### 🟢 Call Options")
        st.metric("Open call positions", str(own.get("position_count", 0)))
        st.caption(
            f"Rich-IV: **{calls.get('rich_premium_count', 0)}**  ·  "
            f"Cheap-IV: **{calls.get('cheap_premium_count', 0)}**  ·  "
            f"Universe: {calls.get('universe_size', 0)}"
        )
        cc = calls.get("covered_call_candidates") or []
        if cc:
            st.caption(f"Covered-call candidates: {len(cc)} (top: {', '.join(str(c.get('ticker', '?')) for c in cc[:5])})")
    with p2:
        st.markdown("### 📊 Index & Flow")
        st.metric("Market tone", str(of.get("market_tone", "?")))
        st.caption(
            f"P/C: **{_fmt_num(of.get('put_call_ratio'))}**  ·  "
            f"Net call prem: **{_fmt_usd(of.get('net_call_premium'))}**  ·  "
            f"Net put prem: **{_fmt_usd(of.get('net_put_premium'))}**"
        )
        st.caption(
            f"Dark pool: **{_fmt_usd(of.get('dark_pool_notional'))}** "
            f"({of.get('dark_pool_trade_count', 0)} trades)  ·  "
            f"Flow signals: **{of.get('options_flow_signal_count', 0)}**"
        )
    with p3:
        st.markdown("### 🔬 Quant Research")
        st.metric("Vol-premium signals", str(len(vp_signals)))
        if best_bt:
            st.caption(
                f"Best 90d win-rate: **{best_bt.get('win_rate', 0) * 100:.1f}%** "
                f"({best_bt.get('strategy', '?')}, n={best_bt.get('n_signals', 0)})"
            )
        else:
            st.caption("No backtest results yet.")
        st.caption(f"Tracked sources: {len(quant.get('hit_rates') or [])}")

    st.divider()

    # ---------- Today's action items ----------
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("**🔥 Hot vol-premium (IV/HV ≥ 1.20)**")
        readings = calls.get("vol_premium_readings") or []
        hot = [r for r in readings if isinstance(r.get("iv_hv_ratio"), (int, float)) and r["iv_hv_ratio"] >= 1.20]
        if hot:
            st.dataframe(
                [
                    {"ticker": r.get("ticker"), "IV/HV": round(r["iv_hv_ratio"], 2), "IV": round(r.get("iv", 0), 3)}
                    for r in hot[:8]
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("None right now.")
    with a2:
        st.markdown("**📈 Top UW flow tickers**")
        top_flow = of.get("top_flow_tickers") or []
        if top_flow:
            st.dataframe(top_flow[:8], width="stretch", hide_index=True)
        else:
            st.caption("No flow data.")
    with a3:
        st.markdown("**🎯 Top quant signals**")
        if vp_signals:
            st.dataframe(vp_signals[:8], width="stretch", hide_index=True)
        else:
            st.caption("No signals fired.")

    st.divider()

    # ---------- Accounts ----------
    st.subheader("Accounts")
    rows = _accounts_summary(portfolio)
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No account data.")

    # ---------- Pillar errors (collapsed) ----------
    pillar_errs: list[str] = []
    for label, p in (("call_options", calls), ("index_flow", flow), ("quant_research", quant)):
        for e in p.get("errors") or []:
            pillar_errs.append(f"[{label}] {e}")
    if pillar_errs:
        with st.expander(f"⚠ Pillar collector warnings ({len(pillar_errs)})", expanded=False):
            for e in pillar_errs:
                st.caption(e)

    st.caption(
        f"FX CAD→USD {portfolio.get('fx_cad_usd', '?')} · "
        f"Portfolio updated {portfolio.get('last_updated', '?')} · "
        f"Payload ts {payload.get('ts', '-')}"
    )


def _render_positions(payload: dict[str, Any]) -> None:
    import streamlit as st

    portfolio = _safe_get(payload, "portfolio")
    rows = _flatten_positions(portfolio)
    if not rows:
        st.info("No positions across any account.")
        return

    accounts = sorted({r["account"] for r in rows})
    types = sorted({r["type"] for r in rows})
    c1, c2 = st.columns(2)
    sel_accounts = c1.multiselect("Account", accounts, default=accounts)
    sel_types = c2.multiselect("Type", types, default=types)

    filtered = [r for r in rows if r["account"] in sel_accounts and r["type"] in sel_types]
    total_mv = sum(r["market_value"] for r in filtered if isinstance(r["market_value"], (int, float)))
    total_pnl = sum(r["unrealized_pnl"] for r in filtered if isinstance(r["unrealized_pnl"], (int, float)))

    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", str(len(filtered)))
    m2.metric("Total Market Value", _fmt_usd(total_mv))
    m3.metric("Total Unrealized PnL", _fmt_usd(total_pnl))

    st.dataframe(filtered, width="stretch", hide_index=True)


def _render_war_room(payload: dict[str, Any]) -> None:
    import streamlit as st

    war = _safe_get(payload, "war_room")
    if not war:
        st.info("War Room collector empty.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite Score", _fmt_num(war.get("composite_score")))
    c2.metric("Regime", str(war.get("regime", "?")))
    c3.metric("Phase", str(war.get("phase", "?")))
    c4.metric("Mandate", str(war.get("mandate", "?")))

    st.subheader("Indicators (weighted composite)")
    ind = war.get("indicators") or []
    if ind:
        st.dataframe(ind, width="stretch", hide_index=True)
    else:
        st.info("No indicator data.")

    st.subheader("Arms — target vs actual allocation")
    arms = war.get("arms") or []
    if arms:
        st.dataframe(arms, width="stretch", hide_index=True)
    else:
        st.info("No arm allocations.")


def _render_pnl(payload: dict[str, Any]) -> None:
    import streamlit as st

    pnl = _safe_get(payload, "pnl")
    if pnl.get("error"):
        st.warning(f"P&L: {pnl['error']}")
    summary = pnl.get("summary") if isinstance(pnl.get("summary"), dict) else {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Realized PnL", _fmt_usd(summary.get("total_realized_pnl")))
    c2.metric("Total Trades", str(summary.get("total_trades", 0)))
    c3.metric("Wins / Losses", f"{summary.get('total_wins', 0)} / {summary.get('total_losses', 0)}")
    c4.metric("Avg Daily PnL", _fmt_usd(summary.get("avg_daily_pnl")))

    c5, c6, c7 = st.columns(3)
    c5.metric("Best Trade", _fmt_usd(summary.get("best_trade")))
    c6.metric("Worst Trade", _fmt_usd(summary.get("worst_trade")))
    c7.metric("Fees Paid", _fmt_usd(summary.get("total_fees")))

    daily = pnl.get("daily") or []
    if daily:
        st.subheader("Daily realised PnL (last 14 days)")
        try:
            sorted_daily = sorted(
                (d for d in daily if isinstance(d, dict)),
                key=lambda d: str(d.get("date", "")),
            )
            chart_data = {d.get("date", ""): d.get("realized_pnl", 0) for d in sorted_daily}
            st.line_chart(chart_data)
        except (TypeError, ValueError, KeyError):
            pass
        st.dataframe(daily, width="stretch", hide_index=True)
    else:
        st.info("No daily PnL history yet.")


def _render_trades(payload: dict[str, Any]) -> None:
    import streamlit as st

    tl = _safe_get(payload, "trade_log")
    st.metric("Recorded trades", str(tl.get("count", 0)))
    if tl.get("error"):
        st.warning(f"Trade log: {tl['error']}")
    trades = tl.get("trades") or []
    if trades:
        st.dataframe(trades, width="stretch", hide_index=True)
    else:
        st.info("No trades recorded.")


def _render_divisions(payload: dict[str, Any]) -> None:
    import streamlit as st

    div = _safe_get(payload, "divisions")
    rows = div.get("divisions") or []
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No division data.")


def _render_tasks(payload: dict[str, Any]) -> None:
    import streamlit as st

    dt_ = _safe_get(payload, "daily_tasks")
    if not dt_:
        st.info("Daily tasks collector empty.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Day", str(dt_.get("day_name", dt_.get("date", "?"))))
    c2.metric("Total", str(dt_.get("total_tasks", 0)))
    c3.metric("Completed", str(dt_.get("completed", 0)))
    c4.metric("Remaining", str(dt_.get("remaining", 0)))

    by_priority = dt_.get("by_priority") or {}
    by_slot = dt_.get("by_slot") or {}
    if by_priority or by_slot:
        col_p, col_s = st.columns(2)
        if by_priority:
            col_p.subheader("By priority")
            col_p.dataframe(
                [{"priority": k, "count": v} for k, v in by_priority.items()],
                width="stretch",
                hide_index=True,
            )
        if by_slot:
            col_s.subheader("By slot")
            col_s.dataframe(
                [
                    {"slot": k, "count": len(v) if isinstance(v, list) else v}
                    for k, v in by_slot.items()
                ],
                width="stretch",
                hide_index=True,
            )

    today = dt_.get("today_tasks") or []
    if today:
        st.subheader(f"Today ({len(today)})")
        st.dataframe(today, width="stretch", hide_index=True)
    upcoming = dt_.get("upcoming_tasks") or []
    if upcoming:
        with st.expander(f"Upcoming ({len(upcoming)})"):
            st.dataframe(upcoming, width="stretch", hide_index=True)


def _render_polymarket(payload: dict[str, Any]) -> None:
    import streamlit as st

    poly = _safe_get(payload, "polymarket")
    c1, c2 = st.columns(2)
    c1.metric("Balance", str(poly.get("balance", "-")))
    c2.metric("Arb opportunities", str(poly.get("arb_count", 0)))

    trending = poly.get("trending_markets") or []
    if trending:
        st.subheader("Trending markets")
        st.dataframe(trending, width="stretch", hide_index=True)
    else:
        st.info("No trending markets.")

    arbs = poly.get("arb_opportunities") or []
    if arbs:
        st.subheader("Arb opportunities")
        st.dataframe(arbs, width="stretch", hide_index=True)


def _render_scenarios(payload: dict[str, Any]) -> None:
    import streamlit as st

    sc = _safe_get(payload, "scenarios")
    st.metric("Total scenarios", str(sc.get("total_scenarios", 0)))
    rows = sc.get("scenarios") or sc.get("all_scenarios") or []
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No scenarios.")


def _render_uw(payload: dict[str, Any]) -> None:
    import streamlit as st

    uw = _safe_get(payload, "unusual_whales")
    if uw.get("error"):
        st.error(f"Unusual Whales error: {uw['error']}")

    flow = uw.get("flow_summary") or {}
    if isinstance(flow, dict) and flow.get("error"):
        st.error(f"Flow summary: {flow['error']}")

    detected_auth_error = False
    for v in (uw.get("flow_summary"), uw.get("dark_pool_spy")):
        if isinstance(v, dict) and "401" in str(v.get("error", "")):
            detected_auth_error = True
            break
    if detected_auth_error:
        st.warning(
            "Unusual Whales returned HTTP 401 — UNUSUAL_WHALES_API_KEY is invalid or expired. "
            "Verify the key in `.env`."
        )

    c1, c2 = st.columns(2)
    c1.subheader("Flow summary")
    c1.json(flow, expanded=False)
    c2.subheader("Dark pool — SPY")
    c2.json(uw.get("dark_pool_spy", {}), expanded=False)

    hot = uw.get("hottest_chains") or []
    if hot:
        st.subheader("Hottest option chains")
        st.dataframe(hot, width="stretch", hide_index=True)

    congress = uw.get("congress_trades") or []
    if congress:
        st.subheader("Congress trades")
        st.dataframe(congress, width="stretch", hide_index=True)


def _render_regime(payload: dict[str, Any]) -> None:
    import streamlit as st

    regime = _safe_get(payload, "regime")
    doctrine = _safe_get(payload, "doctrine")
    moon = _safe_get(payload, "moon")

    st.subheader("Regime")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Primary", str(regime.get("primary", "?")))
    c2.metric("Secondary", str(regime.get("secondary") or "—"))
    conf = regime.get("confidence", 0)
    c3.metric("Confidence", _fmt_pct(conf * 100 if isinstance(conf, float) and conf <= 1 else conf))
    c4.metric("Vol shock ready", str(regime.get("vol_shock_readiness", "?")))

    c5, c6 = st.columns(2)
    c5.metric("Bear signals", str(regime.get("bear_signals", 0)))
    c6.metric("Bull signals", str(regime.get("bull_signals", 0)))

    armed = regime.get("armed_formulas") or []
    if armed:
        st.write("**Armed formulas:** " + ", ".join(str(a) for a in armed))
    summary = regime.get("summary")
    if summary:
        st.info(summary)

    st.divider()
    st.subheader("Doctrine")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("BarrenWuffet state", str(doctrine.get("state", "?")))
    d2.metric("Compliance score", _fmt_num(doctrine.get("compliance_score")))
    d3.metric("Compliant rules", f"{doctrine.get('compliant', 0)} / {doctrine.get('total_rules', 0)}")
    d4.metric("Violations", str(doctrine.get("violations", 0)))

    violations = doctrine.get("violations_list") or []
    if violations:
        st.subheader("Violations")
        st.dataframe(violations, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Moon phase")
    m1, m2, m3 = st.columns(3)
    m1.metric("Moon", f"#{moon.get('moon_number', '?')} · {moon.get('name', '?')}")
    m2.metric("Mandate", str(moon.get("mandate", "?")))
    m3.metric("Conviction", str(moon.get("conviction", "?")))
    st.caption(f"{moon.get('start', '')} → {moon.get('end', '')}")
    events = moon.get("events") or []
    if events:
        st.dataframe(events, width="stretch", hide_index=True)


def _render_health(payload: dict[str, Any]) -> None:
    import streamlit as st

    health = _safe_get(payload, "health")
    backbone = _safe_get(payload, "backbone")

    status, ok, total = _derive_health_status(health)
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall status", status)
    c2.metric("Subsystems OK", f"{ok} / {total}")
    c3.metric("Last check", str(health.get("ts", "-"))[:19])

    subs = health.get("subsystems") or {}
    if isinstance(subs, dict) and subs:
        st.subheader("Subsystems")
        st.dataframe(
            [{"subsystem": k, "status": v} for k, v in subs.items()],
            width="stretch",
            hide_index=True,
        )

    st.subheader("Backbone modules")
    b1, b2, b3 = st.columns(3)
    b1.metric("Loaded", str(backbone.get("loaded_count", 0)))
    b2.metric("Checked", str(backbone.get("total_checked", 0)))
    b3.metric("Bridges", str(backbone.get("bridge_count", 0)))
    modules = backbone.get("modules") or []
    if modules:
        st.dataframe(modules, width="stretch", hide_index=True)


def _render_feeds(payload: dict[str, Any]) -> None:
    import streamlit as st

    feeds = _safe_get(payload, "api_feeds")
    live = _safe_get(payload, "live_feeds")

    c1, c2 = st.columns(2)
    c1.metric("API feeds configured", str(feeds.get("configured_count", 0)))
    c2.metric("API feeds total", str(feeds.get("total_count", 0)))

    feed_rows = feeds.get("feeds") or []
    if feed_rows:
        st.subheader("API feeds")
        st.dataframe(feed_rows, width="stretch", hide_index=True)

    st.subheader("Live market feed")
    if live.get("error"):
        st.warning(f"Live feed: {live['error']}")
    cols = st.columns(5)
    fields = [
        ("BTC", "btc"),
        ("ETH", "eth"),
        ("SPY", "spy"),
        ("Gold", "gold"),
        ("Oil", "oil"),
    ]
    for col, (label, key) in zip(cols, fields):
        col.metric(label, _fmt_num(live.get(key)))
    cols2 = st.columns(5)
    fields2 = [
        ("VIX", "vix"),
        ("Put/Call", "put_call"),
        ("Fear/Greed", "fear_greed"),
        ("DXY", "dxy"),
        ("HY spread (bp)", "hy_spread"),
    ]
    for col, (label, key) in zip(cols2, fields2):
        col.metric(label, _fmt_num(live.get(key)))
    errors = live.get("errors") or []
    if errors:
        with st.expander(f"Feed errors ({len(errors)})"):
            for e in errors:
                st.code(str(e))


# ── Pillar A — Call Options ─────────────────────────────────────────────────


def _render_call_options(payload: dict[str, Any]) -> None:
    import streamlit as st

    st.subheader("🟢 Call Options — Strategies · Research · Flow")
    pillar = _safe_get(payload, "call_options")
    if pillar.get("error"):
        st.error(pillar["error"])
        return
    st.caption(
        f"Universe: {pillar.get('universe_size', 0)} tickers  ·  "
        f"As of {pillar.get('as_of', '-')}  ·  "
        f"Took {pillar.get('duration_s', 0)}s  ·  cache 5m"
    )

    own = pillar.get("own_call_summary") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open call positions", own.get("position_count", 0))
    c2.metric("Call mkt value", _fmt_usd(own.get("total_market_value")))
    c3.metric("Call unrealized P&L", _fmt_usd(own.get("total_unrealized_pnl")))
    c4.metric("Rich-IV names", pillar.get("rich_premium_count", 0))

    st.markdown("**Own call positions by underlying**")
    by_und = own.get("by_underlying") or []
    if by_und:
        st.dataframe(by_und, hide_index=True, width="stretch")
    else:
        st.info("No open long-call positions detected across accounts.")

    st.markdown("**IV / HV vol-premium readings** (sorted by IV/HV ratio desc)")
    readings = pillar.get("vol_premium_readings") or []
    if readings:
        # Display IV/HV nicely.
        view = []
        for r in readings:
            view.append({
                "ticker": r.get("ticker"),
                "spot": r.get("spot"),
                "HV (30d)": f"{r.get('realized_hv', 0)*100:.1f}%",
                "IV (ATM)": f"{r.get('implied_vol', 0)*100:.1f}%" if r.get("option_available") else "-",
                "IV/HV": f"{r.get('iv_hv_ratio', 0):.2f}" if r.get("option_available") else "-",
                "rich": "🔥" if r.get("iv_hv_ratio", 0) >= 1.20 else ("❄️" if 0 < r.get("iv_hv_ratio", 0) <= 0.85 else ""),
            })
        st.dataframe(view, hide_index=True, width="stretch")
    else:
        st.info("No vol-premium readings available (yfinance throttled?).")

    st.markdown("**Covered-call income screen** (own underlyings, ~5% OTM, ~30 DTE)")
    cc = pillar.get("covered_call_candidates") or []
    if cc:
        st.dataframe(cc, hide_index=True, width="stretch")
    else:
        st.caption("_No covered-call candidates found — provide own-underlying shares to enable._")

    errs = pillar.get("errors") or []
    if errs:
        with st.expander(f"Pillar A errors ({len(errs)})"):
            for e in errs:
                st.code(str(e))


# ── Pillar B — Index Strategy / Order Flow / Options Flow ───────────────────


def _render_index_flow(payload: dict[str, Any]) -> None:
    import streamlit as st

    st.subheader("📊 Index Strategy · Order Flow · Options Flow · Research")
    pillar = _safe_get(payload, "index_flow")
    if pillar.get("error"):
        st.error(pillar["error"])
        return
    st.caption(
        f"As of {pillar.get('as_of', '-')}  ·  "
        f"Took {pillar.get('duration_s', 0)}s  ·  cache 5m"
    )

    of = pillar.get("options_flow") or {}
    uw_dead = (
        of.get("market_tone") in ("unknown", None)
        and not of.get("top_flow_tickers")
        and (of.get("net_call_premium", 0) or 0) == 0
    )
    if uw_dead:
        st.warning(
            "🔴 Unusual Whales returned no flow data — likely 401 (invalid `UNUSUAL_WHALES_API_KEY`). "
            "Options-flow metrics below will read as zeros until the key is refreshed."
        )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market tone", str(of.get("market_tone", "?")).upper())
    c2.metric("Put/Call", _fmt_num(of.get("put_call_ratio"), 3))
    c3.metric("Net call premium", _fmt_usd(of.get("net_call_premium")))
    c4.metric("Net put premium", _fmt_usd(of.get("net_put_premium")))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Dark pool notional", _fmt_usd(of.get("dark_pool_notional")))
    c6.metric("Dark pool trades", of.get("dark_pool_trade_count", 0))
    c7.metric("Flow signals", of.get("options_flow_signal_count", 0))
    c8.metric("Top flow tickers", ", ".join((of.get("top_flow_tickers") or [])[:5]) or "-")

    st.markdown("**Index ETF flows** (Δshares × NAV proxy)")
    etf = pillar.get("etf_flows") or []
    if etf:
        view = []
        for e in etf:
            flow_val = e.get("daily_flow_usd")
            if flow_val is None:
                flow_str = "first sample (need 2+ days)" if e.get("shares_outstanding") is None else "-"
            else:
                flow_str = _fmt_usd(flow_val)
            view.append({
                "symbol": e.get("symbol"),
                "date": e.get("date"),
                "price": e.get("price"),
                "shares_outstanding": e.get("shares_outstanding") or "—",
                "AUM": _fmt_usd(e.get("total_assets")) if e.get("total_assets") else "—",
                "daily flow": flow_str,
                "error": e.get("error", ""),
            })
        st.dataframe(view, hide_index=True, width="stretch")
        st.caption(
            "_Daily flow requires shares-outstanding history; ETFFlowClient builds the baseline on first run._"
        )
    else:
        st.info("No ETF flow data.")

    st.markdown("**GEX walls** (Unusual Whales — top dealer-positioned strikes)")
    walls = pillar.get("gex_walls") or {}
    if walls:
        for sym, items in walls.items():
            with st.expander(f"{sym} — {len(items)} walls"):
                if items:
                    st.dataframe(items, hide_index=True, width="stretch")
    else:
        st.caption("_No GEX wall data (UW key may need refresh)._")

    st.markdown("**NYSE Breadth**")
    breadth = pillar.get("breadth") or {}
    if breadth:
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("TRIN", _fmt_num(breadth.get("trin"), 3))
        b2.metric("TICK", _fmt_num(breadth.get("tick"), 0))
        b3.metric("ADV-DECL", _fmt_num(breadth.get("adv_minus_decl"), 0))
        b4.metric("McClellan", _fmt_num(breadth.get("mcclellan_oscillator"), 1))
        b5.metric("Regime", str(breadth.get("regime", "?")).upper())
        if breadth.get("notes"):
            st.caption("  ·  ".join(breadth["notes"]))
    else:
        st.caption("_Breadth unavailable._")

    st.markdown("**COT positioning** (E-mini index futures)")
    cot = pillar.get("cot_positioning") or []
    if cot:
        view = []
        for r in cot:
            sig = r.get("extreme_signal") or {}
            view.append({
                "market": r.get("market", "?"),
                "report_date": r.get("report_date", ""),
                "leveraged_net": r.get("leveraged_net"),
                "asset_mgr_net": r.get("asset_mgr_net"),
                "dealer_net": r.get("dealer_net"),
                "extreme": sig.get("signal", "") if sig else "",
                "z_score": round(sig.get("z_score", 0), 2) if sig else "",
            })
        st.dataframe(view, hide_index=True, width="stretch")
    else:
        st.caption("_COT unavailable (first call downloads ~7 MB ZIP, retry next cycle)._")

    errs = pillar.get("errors") or []
    if errs:
        with st.expander(f"Pillar B errors ({len(errs)})"):
            for e in errs:
                st.code(str(e))


# ── Pillar C — Quant Trading Analysis / Research / Strategies ───────────────


def _render_quant_research(payload: dict[str, Any]) -> None:
    import streamlit as st

    st.subheader("🔬 Quant Trading — Analysis · Research · Strategies")
    pillar = _safe_get(payload, "quant_research")
    if pillar.get("error"):
        st.error(pillar["error"])
        return
    st.caption(
        f"As of {pillar.get('as_of', '-')}  ·  Took {pillar.get('duration_s', 0)}s  ·  cache 5m"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Vol-premium signals", pillar.get("vol_premium_signal_count", 0))
    bt = pillar.get("simple_backtest") or {}
    strategies = (bt.get("strategies") or []) if isinstance(bt, dict) else []
    if not isinstance(strategies, list):
        strategies = []
    best_wr_frac = max(
        (s.get("win_rate", 0) for s in strategies if isinstance(s, dict) and isinstance(s.get("win_rate"), (int, float))),
        default=0,
    )
    c2.metric("Best 90d win-rate", f"{best_wr_frac * 100:.1f}%")
    hr = pillar.get("hit_rates") or {}
    c3.metric("Tracked signal sources", len(hr) if hasattr(hr, "__len__") else 0)

    st.markdown("**Vol-premium signals** (LONG_PUT when IV/HV ≥ 1.20)")
    sigs = pillar.get("vol_premium_signals") or []
    if sigs:
        st.dataframe(sigs, hide_index=True, width="stretch")
    else:
        st.info("No vol-premium signals firing right now.")

    st.markdown("**90-day proxy backtest** (war_room / vol_premium / combined)")
    if strategies:
        view = []
        for s in strategies:
            if not isinstance(s, dict):
                continue
            view.append({
                "strategy": s.get("strategy", "?"),
                "signals": s.get("n_signals"),
                "wins": s.get("n_wins"),
                "win_rate %": round(float(s.get("win_rate", 0)) * 100, 1),
                "avg_drawdown %": round(float(s.get("avg_drawdown_pct", 0)), 2),
                "notes": s.get("notes", ""),
            })
        st.dataframe(view, hide_index=True, width="stretch")
        if bt.get("start_date"):
            st.caption(f"Window: {bt.get('start_date')} → {bt.get('end_date')}")
    else:
        st.caption("_Backtest unavailable._")

    st.markdown("**Walk-forward backtest** (mean-reversion vs momentum)")
    wf_col1, wf_col2 = st.columns([1, 3])
    ticker = wf_col1.text_input("Ticker", value="SPY", key="__wf_ticker_input__")
    if wf_col2.button("Run walk-forward (5–15s)", key="__wf_run__"):
        st.session_state["__wf_ticker__"] = ticker
        with st.spinner(f"Running walk-forward on {ticker} ..."):
            wf_payload = _get_cached_pillar()("quant_walk_forward", key=ticker)
        st.session_state["__wf_result__"] = wf_payload
    wf_payload = st.session_state.get("__wf_result__", {})
    wf_data = wf_payload.get("walk_forward") or {} if isinstance(wf_payload, dict) else {}
    if wf_data:
        if wf_data.get("error"):
            st.warning(wf_data["error"])
        else:
            mr = wf_data.get("mean_reversion") or {}
            mom = wf_data.get("momentum") or {}
            view = []
            for label, r in (("mean_reversion", mr), ("momentum", mom)):
                if r:
                    view.append({
                        "strategy": label,
                        "folds": r.get("total_folds"),
                        "OOS Sharpe": round(r.get("oos_sharpe", 0), 3),
                        "OOS return": round(r.get("oos_return", 0), 4),
                        "OOS hit-rate": round(r.get("oos_hit_rate", 0), 3),
                        "max DD": round(r.get("max_drawdown", 0), 3),
                    })
            if view:
                st.dataframe(view, hide_index=True, width="stretch")

    st.markdown("**Signal journal hit rates** (logged signals → resolved outcomes)")
    if hr:
        view = []
        for src, stats in hr.items():
            if isinstance(stats, dict):
                view.append({
                    "source": src,
                    "wins": stats.get("wins"),
                    "losses": stats.get("losses"),
                    "rate %": round(stats.get("rate", 0) * 100, 1) if isinstance(stats.get("rate"), (int, float)) else "-",
                })
        if view:
            st.dataframe(view, hide_index=True, width="stretch")
    else:
        st.caption("_No resolved signals tracked yet._")

    errs = pillar.get("errors") or []
    if errs:
        with st.expander(f"Pillar C errors ({len(errs)})"):
            for e in errs:
                st.code(str(e))


def _data_source_status(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Return [(label, status_emoji)] for the top status strip.
    Status emoji: 🟢 healthy, 🟡 degraded, 🔴 down, ⚪ unknown.
    """
    out: list[tuple[str, str]] = []

    # Portfolio (any account verified today/yesterday)
    pf = _safe_get(payload, "portfolio")
    accounts = pf.get("accounts") or []
    fresh = sum(1 for a in accounts if str(a.get("verified", "")).startswith(dt.date.today().isoformat()[:7]))
    out.append(("Portfolio", "🟢" if fresh else ("🟡" if accounts else "⚪")))

    # IBKR (live_feeds errors mention it when down)
    live = _safe_get(payload, "live_feeds")
    ib_down = any("IBKR" in str(e) or "TWS" in str(e) for e in (live.get("errors") or []))
    out.append(("IBKR", "🔴" if ib_down else "🟢"))

    # Unusual Whales
    uw = _safe_get(payload, "unusual_whales")
    fs = uw.get("flow_summary") if isinstance(uw.get("flow_summary"), dict) else {}
    uw_bad = bool(uw.get("error")) or bool(fs.get("error")) or (isinstance(fs, dict) and not fs)
    out.append(("Unusual Whales", "🔴" if uw_bad else "🟢"))

    # CFTC COT (gated off by default \u2014 zip 404 for current years)
    flow = _safe_get(payload, "index_flow")
    cftc_enabled = os.environ.get("AAC_DASHBOARD_ENABLE_CFTC", "0") == "1"
    if not cftc_enabled:
        out.append(("CFTC COT", "\u26aa"))
    else:
        cot_ok = bool(flow.get("cot_positioning"))
        out.append(("CFTC COT", "\ud83d\udfe2" if cot_ok else "\ud83d\udd34"))

    # Breadth (gated off by default \u2014 ^TRIN delisted)
    breadth_enabled = os.environ.get("AAC_DASHBOARD_ENABLE_BREADTH", "0") == "1"
    if not breadth_enabled:
        out.append(("NYSE Breadth", "\u26aa"))
    else:
        breadth = flow.get("breadth") or {}
        out.append(("NYSE Breadth", "\ud83d\udfe2" if breadth else "\ud83d\udd34"))

    # yfinance / vol-premium  (now: IBKR primary, yfinance fallback)
    calls = _safe_get(payload, "call_options")
    readings = calls.get("vol_premium_readings") or []
    src = "ibkr" if readings and any(r.get("source") == "ibkr" for r in readings) else "yf"
    out.append((f"IV/HV ({src})", "🟢" if readings else "🔴"))

    return out


def _render(payload: dict[str, Any]) -> None:
    import streamlit as st

    st.title("AAC Trading Intelligence Dashboard")
    st.caption(
        f"Last updated: {payload.get('ts', '-')}  ·  ops cache 30s  ·  pillar cache 5m  ·  auto-refresh 30s"
    )

    # Data-source status strip
    statuses = _data_source_status(payload)
    cols = st.columns(len(statuses))
    for col, (label, emoji) in zip(cols, statuses):
        col.markdown(f"<div style='text-align:center;font-size:0.85rem'>{emoji}<br>{label}</div>", unsafe_allow_html=True)
    st.divider()

    # Sidebar cache controls
    with st.sidebar:
        st.subheader("Controls")
        if st.button("🔄 Refresh now (clear cache)"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Payload ts: {payload.get('ts', '-')}")

    # Make portfolio + UW available to pillar collectors via session state.
    st.session_state["__portfolio_payload__"] = payload.get("portfolio") or {}
    st.session_state["__uw_payload__"] = payload.get("unusual_whales") or {}

    pillar_tabs = st.tabs(
        [
            "📈 Overview",
            "🟢 Call Options",
            "📊 Index & Flow",
            "🔬 Quant Research",
            "💼 Positions",
            "💰 PnL & Trades",
            "🩺 Health & War Room",
            "📰 Other",
        ]
    )

    with pillar_tabs[0]:
        _render_overview(payload)

    with pillar_tabs[1]:
        _render_call_options(payload)

    with pillar_tabs[2]:
        _render_index_flow(payload)

    with pillar_tabs[3]:
        _render_quant_research(payload)

    with pillar_tabs[4]:
        _render_positions(payload)

    with pillar_tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            _render_pnl(payload)
        with c2:
            _render_trades(payload)

    with pillar_tabs[6]:
        c1, c2 = st.columns(2)
        with c1:
            _render_health(payload)
        with c2:
            _render_war_room(payload)
        _render_feeds(payload)

    with pillar_tabs[7]:
        sub = st.tabs(["UW", "Regime / Doctrine", "Tasks", "Divisions", "Polymarket", "Scenarios", "Raw"])
        with sub[0]:
            _render_uw(payload)
        with sub[1]:
            _render_regime(payload)
        with sub[2]:
            _render_tasks(payload)
        with sub[3]:
            _render_divisions(payload)
        with sub[4]:
            _render_polymarket(payload)
        with sub[5]:
            _render_scenarios(payload)
        with sub[6]:
            with st.expander("Raw payload", expanded=False):
                st.code(json.dumps(payload, default=_json_default, indent=2), language="json")


# ── Entry point ─────────────────────────────────────────────────────────────


_COLLECTOR_NAMES = (
    "portfolio",
    "war_room",
    "live_feeds",
    "regime",
    "doctrine",
    "moon",
    "health",
    "tasks",
    "daily_tasks",
    "unusual_whales",
    "divisions",
    "api_feeds",
    "polymarket",
    "scenarios",
    "backbone",
    "pnl",
    "trade_log",
)


def main() -> None:
    try:
        import streamlit as st
    except ImportError:
        _log.error("streamlit_not_installed")
        return

    from streamlit.errors import StreamlitAPIException

    try:
        st.set_page_config(
            page_title="AAC Streamlit Dashboard",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
    except StreamlitAPIException:
        pass

    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=30_000, key="aac_dashboard_autorefresh")
    except ImportError:
        pass

    placeholder = st.empty()
    placeholder.caption("Loading collectors ...")

    with st.spinner("Collecting portfolio, war room, health, and feed data ..."):
        payload: dict[str, Any] = {"ts": dt.datetime.now().isoformat()}
        collect = _get_cached_collect()
        for name in _COLLECTOR_NAMES:
            payload[name] = collect(name)

        # Pillar collectors run after the operational ones so they can read
        # the freshly-collected portfolio + UW snapshot via session state.
        st.session_state["__portfolio_payload__"] = payload.get("portfolio") or {}
        st.session_state["__uw_payload__"] = payload.get("unusual_whales") or {}
        pillar = _get_cached_pillar()
        for pillar_name in ("call_options", "index_flow", "quant_research"):
            payload[pillar_name] = pillar(pillar_name)

    placeholder.empty()
    _render(payload)


# ── Backward-compatible helpers ─────────────────────────────────────────────


def run_streamlit_dashboard(port: int = 8501) -> int:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_THIS_FILE),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    _log.info("starting_streamlit_dashboard", port=port)
    return subprocess.call(cmd, cwd=str(_THIS_FILE.parent.parent))


class AACStreamlitDashboard:
    """Adapter used by ``monitoring.aac_master_monitoring_dashboard`` web mode."""

    async def run_dashboard(self, port: int = 8501) -> int:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(_THIS_FILE),
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ]
        proc = subprocess.Popen(cmd, cwd=str(_THIS_FILE.parent.parent))
        _log.info("streamlit_dashboard_running", pid=proc.pid, port=port)
        try:
            while proc.poll() is None:
                await asyncio.sleep(1.0)
            return proc.returncode or 0
        except asyncio.CancelledError:
            proc.terminate()
            raise
        except KeyboardInterrupt:
            proc.terminate()
            return 0
        finally:
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()

