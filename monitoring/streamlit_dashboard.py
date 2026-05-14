from __future__ import annotations

import asyncio
import datetime as dt
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
    war = _safe_get(payload, "war_room")
    regime = _safe_get(payload, "regime")

    status, ok, total = _derive_health_status(health)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio (USD)", _fmt_usd(portfolio.get("total_usd")))
    c2.metric("Unrealized PnL", _fmt_usd(portfolio.get("total_unrealized_pnl")))
    c3.metric("Positions", str(portfolio.get("total_positions", 0)))
    c4.metric("Regime", str(regime.get("primary", war.get("regime", "?"))))
    c5.metric("Health", f"{status} ({ok}/{total})")

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Composite", _fmt_num(war.get("composite_score")))
    c7.metric("Confidence", _fmt_pct(war.get("confidence", 0) * 100 if isinstance(war.get("confidence"), float) and war.get("confidence", 0) <= 1 else war.get("confidence")))
    c8.metric("Phase", str(war.get("phase", "?")))
    c9.metric("Mandate", str(war.get("mandate", "?")))

    st.subheader("Accounts")
    rows = _accounts_summary(portfolio)
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No account data.")

    st.caption(
        f"FX CAD→USD {portfolio.get('fx_cad_usd', '?')} · "
        f"Updated {portfolio.get('last_updated', '?')} · "
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
            chart_data = {d.get("date", ""): d.get("realized_pnl", 0) for d in daily}
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


def _render(payload: dict[str, Any]) -> None:
    import streamlit as st

    st.title("AAC Monitoring Dashboard")
    st.caption(f"Last updated: {payload.get('ts', '-')}  ·  cache TTL 30s  ·  auto-refresh 30s")

    tabs = st.tabs(
        [
            "Overview",
            "Positions",
            "War Room",
            "PnL",
            "Trades",
            "Tasks",
            "Divisions",
            "Polymarket",
            "Scenarios",
            "Unusual Whales",
            "Regime / Doctrine",
            "Health",
            "Feeds",
            "Raw",
        ]
    )
    renderers = [
        _render_overview,
        _render_positions,
        _render_war_room,
        _render_pnl,
        _render_trades,
        _render_tasks,
        _render_divisions,
        _render_polymarket,
        _render_scenarios,
        _render_uw,
        _render_regime,
        _render_health,
        _render_feeds,
    ]
    for tab, fn in zip(tabs[:-1], renderers):
        with tab:
            fn(payload)
    with tabs[-1]:
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
    placeholder.title("AAC Monitoring Dashboard")
    placeholder.caption("Loading collectors ...")

    with st.spinner("Collecting portfolio, war room, health, and feed data ..."):
        payload: dict[str, Any] = {"ts": dt.datetime.now().isoformat()}
        collect = _get_cached_collect()
        for name in _COLLECTOR_NAMES:
            payload[name] = collect(name)

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
