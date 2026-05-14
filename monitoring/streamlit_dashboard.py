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

# Collector errors we expect from mission_control collectors. Anything else
# should propagate so we can find real bugs.
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
# Each collector wrapped in @st.cache_data(ttl=30) so reruns are cheap.
# We import streamlit lazily so this module is importable outside streamlit.


def _make_collector(name: str) -> Callable[[], dict[str, Any]]:
    """Return a cached wrapper around mission_control.collect_<name>."""
    import streamlit as st

    @st.cache_data(ttl=30, show_spinner=False)
    def _inner() -> dict[str, Any]:
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

    _inner.__name__ = f"collect_{name}_cached"
    return _inner


# ── Render helpers ──────────────────────────────────────────────────────────


def _safe_get(payload: dict[str, Any], key: str) -> dict[str, Any]:
    val = payload.get(key)
    return val if isinstance(val, dict) else {}


def _render_overview(payload: dict[str, Any]) -> None:
    import streamlit as st

    portfolio = _safe_get(payload, "portfolio")
    health = _safe_get(payload, "health")
    pnl = _safe_get(payload, "pnl")
    pnl_summary = pnl.get("summary", {}) if isinstance(pnl.get("summary"), dict) else {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Timestamp", payload.get("ts", "-"))
    total_usd = portfolio.get("total_usd")
    c2.metric(
        "Portfolio (USD)",
        f"${total_usd:,.2f}" if isinstance(total_usd, (int, float)) else "-",
    )
    unreal = portfolio.get("total_unrealized_pnl")
    c3.metric(
        "Unrealized PnL",
        f"${unreal:,.2f}" if isinstance(unreal, (int, float)) else "-",
    )
    c4.metric("Health", str(health.get("status", "unknown")))

    accounts = portfolio.get("accounts") or []
    if accounts:
        st.subheader("Accounts")
        st.dataframe(accounts, width="stretch")

    realized = pnl_summary.get("realized")
    if realized is not None:
        c5, c6, c7 = st.columns(3)
        c5.metric(
            "Realized PnL",
            f"${realized:,.2f}" if isinstance(realized, (int, float)) else str(realized),
        )
        c6.metric("Trades", str(pnl_summary.get("trade_count", "-")))
        c7.metric("Win Rate", str(pnl_summary.get("win_rate", "-")))


def _render_table(payload: dict[str, Any], key: str, list_key: str, empty_msg: str) -> None:
    import streamlit as st

    section = _safe_get(payload, key)
    rows = section.get(list_key) or []
    if rows:
        st.dataframe(rows, width="stretch")
    else:
        st.info(empty_msg)
    if "error" in section:
        st.warning(f"collector error: {section['error']}")


def _render_json(payload: dict[str, Any], key: str) -> None:
    import streamlit as st

    st.json(payload.get(key, {}), expanded=False)


def _render(payload: dict[str, Any]) -> None:
    import streamlit as st

    st.title("AAC Monitoring Dashboard")
    st.caption(f"Last updated: {payload.get('ts', '-')}  ·  cache TTL: 30s")

    tabs = st.tabs(
        [
            "Overview",
            "Positions",
            "PnL",
            "Trades",
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

    with tabs[0]:
        _render_overview(payload)

    with tabs[1]:
        st.subheader("War Room Positions")
        war = _safe_get(payload, "war_room")
        positions = war.get("positions") or war.get("open_positions") or []
        if positions:
            st.dataframe(positions, width="stretch")
        else:
            _render_json(payload, "war_room")

    with tabs[2]:
        pnl = _safe_get(payload, "pnl")
        st.subheader("PnL Summary")
        st.json(pnl.get("summary", {}), expanded=True)
        daily = pnl.get("daily") or []
        if daily:
            st.subheader("Daily PnL")
            try:
                st.line_chart(daily)
            except (TypeError, ValueError):
                st.dataframe(daily, width="stretch")

    with tabs[3]:
        st.subheader(f"Trade Log ({_safe_get(payload, 'trade_log').get('count', 0)} trades)")
        _render_table(payload, "trade_log", "trades", "No trades recorded.")

    with tabs[4]:
        st.subheader("Divisions")
        _render_table(payload, "divisions", "divisions", "No division data.")

    with tabs[5]:
        poly = _safe_get(payload, "polymarket")
        c1, c2 = st.columns(2)
        c1.metric("Balance", str(poly.get("balance", "-")))
        c2.metric("Arb opportunities", str(poly.get("arb_count", 0)))
        st.subheader("Trending markets")
        trending = poly.get("trending_markets") or []
        if trending:
            st.dataframe(trending, width="stretch")
        else:
            st.info("No trending markets.")
        arbs = poly.get("arb_opportunities") or []
        if arbs:
            st.subheader("Arb opportunities")
            st.dataframe(arbs, width="stretch")

    with tabs[6]:
        scenarios = _safe_get(payload, "scenarios")
        st.metric("Total scenarios", str(scenarios.get("total_scenarios", 0)))
        rows = scenarios.get("scenarios") or scenarios.get("all_scenarios") or []
        if rows:
            st.dataframe(rows, width="stretch")
        else:
            _render_json(payload, "scenarios")

    with tabs[7]:
        uw = _safe_get(payload, "unusual_whales")
        st.subheader("Flow summary")
        st.json(uw.get("flow_summary", {}), expanded=False)
        st.subheader("Hottest chains")
        hot = uw.get("hottest_chains") or []
        if hot:
            st.dataframe(hot, width="stretch")
        st.subheader("Congress trades")
        congress = uw.get("congress_trades") or []
        if congress:
            st.dataframe(congress, width="stretch")
        st.subheader("Dark pool — SPY")
        st.json(uw.get("dark_pool_spy", {}), expanded=False)

    with tabs[8]:
        c1, c2, c3 = st.columns(3)
        c1.subheader("Regime")
        c1.json(payload.get("regime", {}), expanded=False)
        c2.subheader("Doctrine")
        c2.json(payload.get("doctrine", {}), expanded=False)
        c3.subheader("Moon phase")
        c3.json(payload.get("moon", {}), expanded=False)

    with tabs[9]:
        st.subheader("System Health")
        st.json(payload.get("health", {}), expanded=True)
        st.subheader("Backbone")
        backbone = _safe_get(payload, "backbone")
        c1, c2, c3 = st.columns(3)
        c1.metric("Modules loaded", str(backbone.get("loaded_count", 0)))
        c2.metric("Modules checked", str(backbone.get("total_checked", 0)))
        c3.metric("Bridges", str(backbone.get("bridge_count", 0)))

    with tabs[10]:
        st.subheader("API Feeds")
        feeds = _safe_get(payload, "api_feeds")
        c1, c2 = st.columns(2)
        c1.metric("Configured", str(feeds.get("configured_count", 0)))
        c2.metric("Total", str(feeds.get("total_count", 0)))
        feed_rows = feeds.get("feeds") or []
        if feed_rows:
            st.dataframe(feed_rows, width="stretch")
        st.subheader("Live Feeds")
        _render_json(payload, "live_feeds")

    with tabs[11]:
        with st.expander("Raw payload", expanded=False):
            st.code(json.dumps(payload, default=_json_default, indent=2), language="json")


# ── Entry point (run by `streamlit run`) ────────────────────────────────────


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
    """Streamlit entry point.

    Strategy:
      1. Paint page-config + title FIRST so the user sees something instantly.
      2. Kick off cached collectors under a spinner.
      3. Auto-refresh every 30s.
    """
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
        # set_page_config can only run once per session — harmless on rerun.
        pass

    # Auto-refresh every 30s. Optional dependency.
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
        for name in _COLLECTOR_NAMES:
            collector = _make_collector(name)
            payload[name] = collector()

    placeholder.empty()
    _render(payload)


# ── Backward-compatible helpers (do not remove without updating callers) ────


def run_streamlit_dashboard(port: int = 8501) -> int:
    """Run this module via the streamlit CLI in a subprocess.

    Called from ``core/aac_master_launcher.py``.
    """
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
