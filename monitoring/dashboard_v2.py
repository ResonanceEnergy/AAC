"""AAC dashboard v2 — sidebar navigation, dark theme, redesigned pages.

Reuses data collectors and agent panels from ``streamlit_dashboard.py``;
replaces only the presentation layer. Launch via ``python launch.py dashboard2``.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from typing import Any

import structlog

from monitoring import dashboard_ui as ui
from monitoring.streamlit_dashboard import (
    _COLLECTOR_NAMES,
    _FUNCTIONS_REGISTRY,
    _accounts_summary,
    _data_source_status,
    _derive_health_status,
    _flatten_positions,
    _get_cached_collect,
    _get_cached_pillar,
    _json_default,
    _render_agent_panel,
    _render_briefings_tab,
    _render_chat_tab,
    _render_decisions_panel,
    _render_functions_panel,
    _run_function,
    _safe_get,
)

_log = structlog.get_logger()


# ── Status & header ─────────────────────────────────────────────────────────


_EMOJI_TO_STATE: dict[str, str] = {
    "\U0001f7e2": "ok",      # green
    "\U0001f7e1": "warn",    # yellow
    "\U0001f534": "bad",     # red
    "\u26aa": "idle",        # white (gated/disabled)
}


def _render_status_strip(payload: dict[str, Any]) -> None:
    import streamlit as st

    sources = _data_source_status(payload)
    if not sources:
        return
    pills_html: list[str] = []
    for name, status in sources:
        s = _EMOJI_TO_STATE.get(str(status), None)
        if s is None:
            s = "ok" if status == "ok" else ("warn" if status in ("degraded", "stale") else "bad")
        pills_html.append(ui.pill(name, s))
    st.markdown(" ".join(pills_html), unsafe_allow_html=True)


def _render_header(payload: dict[str, Any]) -> None:
    import streamlit as st

    ts = payload.get("ts") or dt.datetime.now().isoformat()
    cols = st.columns([3, 2])
    with cols[0]:
        st.markdown(
            "<div style='display:flex;align-items:baseline;gap:14px;'>"
            "<span style='font-size:22px;font-weight:700;color:#e2e8f0;'>AAC Command Center</span>"
            f"<span style='color:#94a3b8;font-size:12px;'>updated {ui.fmt_relative_time(ts)}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            "<div style='text-align:right;'>"
            f"<span style='color:#94a3b8;font-size:11px;'>{dt.datetime.now().strftime('%a %b %d %H:%M:%S')}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    _render_status_strip(payload)


# ── Page: Overview ──────────────────────────────────────────────────────────


def _page_overview(payload: dict[str, Any]) -> None:
    import streamlit as st

    portfolio = _safe_get(payload, "portfolio")
    war = _safe_get(payload, "war_room")
    health = _safe_get(payload, "health")
    pnl = _safe_get(payload, "pnl")

    # Aggregate totals across accounts
    accounts = _accounts_summary(portfolio)
    total_equity = sum(a.get("equity", 0) or 0 for a in accounts)
    total_cash = sum(a.get("cash", 0) or 0 for a in accounts)
    total_buying = sum(a.get("buying_power", 0) or 0 for a in accounts)

    rows = _flatten_positions(portfolio)
    open_positions = len(rows)
    total_unrl = sum(r.get("unrealized_pnl", 0) or 0 for r in rows if isinstance(r.get("unrealized_pnl"), (int, float)))

    pnl_today = pnl.get("today_realized") if isinstance(pnl, dict) else None
    pnl_mtd = pnl.get("mtd_realized") if isinstance(pnl, dict) else None

    status_label, ok_count, total_count = _derive_health_status(health)

    ui.kpi_row([
        ui.kpi("Total Equity", ui.fmt_usd(total_equity), sub=f"{len(accounts)} accounts"),
        ui.kpi("Cash", ui.fmt_usd(total_cash), sub=f"BP: {ui.fmt_usd(total_buying)}"),
        ui.kpi(
            "Unrealized",
            ui.fmt_usd(total_unrl, signed=True),
            delta_dir=ui.delta_dir_from(total_unrl),
            sub=f"{open_positions} open",
        ),
    ])
    ui.kpi_row([
        ui.kpi(
            "Today",
            ui.fmt_usd(pnl_today, signed=True) if pnl_today is not None else "—",
            delta_dir=ui.delta_dir_from(pnl_today),
            sub="realized",
        ),
        ui.kpi(
            "MTD",
            ui.fmt_usd(pnl_mtd, signed=True) if pnl_mtd is not None else "—",
            delta_dir=ui.delta_dir_from(pnl_mtd),
            sub="realized",
        ),
        ui.kpi(
            "Subsystems",
            f"{ok_count}/{total_count}",
            sub=status_label,
            delta_dir="up" if ok_count == total_count else ("down" if ok_count < total_count // 2 else "flat"),
        ),
    ])

    # War room summary
    if war:
        ui.section("War Room", sub=f"regime: {war.get('regime', '?')} · phase: {war.get('phase', '?')}")
        ui.kpi_row([
            ui.kpi("Composite Score", ui.fmt_num(war.get("composite_score"))),
            ui.kpi("Regime", str(war.get("regime", "?")).upper()),
            ui.kpi("Phase", str(war.get("phase", "?")).upper()),
            ui.kpi("Mandate", str(war.get("mandate", "?"))),
        ])

    # Accounts breakdown
    ui.section("Accounts")
    if accounts:
        # Staleness chip per row + top-level banner if anything is wrong.
        stale_accounts: list[str] = []
        for a in accounts:
            days = int(a.get("days_stale") or 0)
            if days >= 14:
                a["fresh"] = f"🔴 {days}d"
                stale_accounts.append(f"{a.get('account', '?')} ({days}d)")
            elif days >= 3:
                a["fresh"] = f"⚠️ {days}d"
                stale_accounts.append(f"{a.get('account', '?')} ({days}d)")
            else:
                a["fresh"] = f"✅ {days}d"

        if stale_accounts:
            st.warning(
                f"⚠️ STALE LIVE DATA — last verified > 3 days ago: "
                + ", ".join(stale_accounts)
                + ". Run `python -m monitoring.live_portfolio_refresh` and check `data/account_balances.json::_meta.live_refresh` for credential errors."
            )

        ui.smart_table(
            accounts,
            column_config={
                "account": ui.col_text("Account", width="medium"),
                "platform": ui.col_text("Broker"),
                "equity": ui.col_usd("Equity", width="medium"),
                "cash": ui.col_usd("Cash", width="medium"),
                "buying_power": ui.col_usd("Buying Power", width="medium"),
                "unrealized_pnl": ui.col_usd("Unrl P&L"),
                "realized_pnl": ui.col_usd("Rlzd P&L"),
                "positions": ui.col_int("Pos"),
                "currency": ui.col_text("Ccy"),
                "verified": ui.col_text("Verified"),
                "fresh": ui.col_text("Fresh"),
            },
            column_order=["account", "platform", "equity", "cash", "buying_power",
                          "unrealized_pnl", "realized_pnl", "positions", "currency",
                          "verified", "fresh"],
        )
    else:
        st.caption("No account data.")

    # Alerts tail
    _render_alerts_strip()

    # Functions panel moved to its own page ("Functions")


# ── Page: Functions & Data Flows ───────────────────────────────


def _summarise_result(result: Any) -> tuple[str, int, list[str]]:
    """Return (kind, size, sample_keys) for header chip line."""
    if isinstance(result, dict):
        keys = [k for k in result.keys() if not k.startswith("_")]
        return "dict", len(keys), keys[:8]
    if isinstance(result, list):
        if result and isinstance(result[0], dict):
            return "list[dict]", len(result), list(result[0].keys())[:8]
        return "list", len(result), []
    return type(result).__name__, 1, []


def _render_result_body(result: Any) -> None:
    import streamlit as st

    # Tabular preview when possible
    if isinstance(result, dict):
        # Surface tabular sub-keys as smart_tables
        rendered_any = False
        for key, val in list(result.items())[:30]:
            if isinstance(val, list) and val and isinstance(val[0], dict):
                ui.section(key)
                ui.smart_table(val[:200])
                rendered_any = True
        if not rendered_any:
            # Flat dict — show as 2-col table
            flat_rows = [
                {"key": k, "value": v if isinstance(v, (str, int, float, bool)) or v is None
                 else json.dumps(v, default=_json_default)[:200]}
                for k, v in result.items()
            ]
            if flat_rows:
                ui.smart_table(flat_rows, column_config={
                    "key": ui.col_text("Key", width="medium"),
                    "value": ui.col_text("Value", width="large"),
                }, hide_uuids=False)
    elif isinstance(result, list) and result and isinstance(result[0], dict):
        ui.smart_table(result[:200])
    else:
        st.code(json.dumps(result, default=_json_default, indent=2)[:20_000], language="json")

    with st.expander("Raw JSON", expanded=False):
        st.code(json.dumps(result, default=_json_default, indent=2)[:200_000], language="json")


def _page_functions(payload: dict[str, Any]) -> None:
    """Run any collector / pillar / probe on demand and inspect result."""
    import time

    import streamlit as st

    st.caption(
        "Each button runs the underlying function fresh (bypassing the 30s/300s cache) "
        "and renders the result as a table when possible. Use this to verify a data source "
        "is alive, time it, and drill into specific collectors without flipping pages."
    )

    last = st.session_state.get("__fn_panel_last__")  # (label, kind, target, result, took_ms, ts)
    history: list[dict[str, Any]] = st.session_state.get("__fn_panel_history__", [])

    # ---- Top control bar ----
    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 1])
    if cc1.button("▶︎ Run all collectors", use_container_width=True):
        for name in _COLLECTOR_NAMES:
            t0 = time.perf_counter()
            res = _run_function("collector", name)
            took = int((time.perf_counter() - t0) * 1000)
            history.append({"label": f"collector:{name}", "ok": not (isinstance(res, dict) and res.get("error")),
                            "took_ms": took, "ts": dt.datetime.now().strftime("%H:%M:%S"),
                            "keys": len(res) if isinstance(res, (dict, list)) else 1})
        st.session_state["__fn_panel_history__"] = history[-200:]
        st.rerun()
    if cc2.button("▶︎ Run all 3 pillars", use_container_width=True):
        for tgt in ("call_options", "index_flow", "quant_research"):
            t0 = time.perf_counter()
            res = _run_function("pillar", tgt)
            took = int((time.perf_counter() - t0) * 1000)
            history.append({"label": f"pillar:{tgt}", "ok": not (isinstance(res, dict) and res.get("error")),
                            "took_ms": took, "ts": dt.datetime.now().strftime("%H:%M:%S"),
                            "keys": len(res) if isinstance(res, (dict, list)) else 1})
        st.session_state["__fn_panel_history__"] = history[-200:]
        st.rerun()
    if cc3.button("🧹 Clear history", use_container_width=True):
        st.session_state.pop("__fn_panel_history__", None)
        st.session_state.pop("__fn_panel_last__", None)
        st.rerun()
    cc4.caption(f"{len(history)} runs")

    # ---- Run history KPIs ----
    if history:
        ok_n = sum(1 for h in history if h.get("ok"))
        bad_n = len(history) - ok_n
        avg_ms = sum(h.get("took_ms", 0) for h in history) / len(history)
        slowest = max(history, key=lambda h: h.get("took_ms", 0))
        ui.kpi_row([
            ui.kpi("Runs", str(len(history))),
            ui.kpi("OK", str(ok_n), delta_dir="up" if not bad_n else "flat"),
            ui.kpi("Errors", str(bad_n), delta_dir="down" if bad_n else "flat"),
            ui.kpi("Avg latency", f"{int(avg_ms)} ms"),
            ui.kpi("Slowest", slowest.get("label", "—"), sub=f"{slowest.get('took_ms', 0)} ms"),
        ])

    # ---- Button grid by group ----
    for group, items in _FUNCTIONS_REGISTRY.items():
        ui.section(group)
        cols = st.columns(min(4, len(items)))
        for idx, (label, kind, target) in enumerate(items):
            col = cols[idx % len(cols)]
            if kind == "url":
                col.link_button(label, target, use_container_width=True)
                continue
            if col.button(label, key=f"v2_fn_{kind}_{target}", use_container_width=True):
                t0 = time.perf_counter()
                with st.spinner(f"Running {label}…"):
                    result = _run_function(kind, target)
                took = int((time.perf_counter() - t0) * 1000)
                ok = not (isinstance(result, dict) and result.get("error"))
                size = len(result) if isinstance(result, (dict, list)) else 1
                st.session_state["__fn_panel_last__"] = {
                    "label": label, "kind": kind, "target": target,
                    "result": result, "took_ms": took,
                    "ts": dt.datetime.now().strftime("%H:%M:%S"), "ok": ok,
                }
                history.append({"label": f"{kind}:{target}", "ok": ok, "took_ms": took,
                                "ts": dt.datetime.now().strftime("%H:%M:%S"), "keys": size})
                st.session_state["__fn_panel_history__"] = history[-200:]
                st.rerun()

    # ---- Last result panel ----
    if last:
        ui.section(f"Last result · {last['label']}")
        result = last["result"]
        kind, size, sample = _summarise_result(result)
        chips = [
            ui.pill(f"{last['kind']}:{last['target']}", "ok" if last["ok"] else "bad"),
            ui.pill(f"{last['took_ms']} ms", "ok" if last["took_ms"] < 1000 else "warn"),
            ui.pill(f"{kind} · {size}", "ok"),
            ui.pill(last["ts"], "ok"),
        ]
        import streamlit as st  # noqa: PLC0415
        st.markdown(" ".join(chips), unsafe_allow_html=True)
        if isinstance(result, dict) and result.get("error"):
            st.error(f"❌ {result['error']}")
        if sample:
            st.caption("Sample fields: " + ", ".join(f"`{k}`" for k in sample))
        _render_result_body(result)

    # ---- Run history table ----
    if history:
        ui.section("Run history")
        ui.smart_table(
            list(reversed(history)),
            column_config={
                "ts": ui.col_text("Time"),
                "label": ui.col_text("Function", width="medium"),
                "ok": ui.col_text("OK"),
                "keys": ui.col_int("Size"),
                "took_ms": ui.col_int("ms"),
            },
            column_order=["ts", "label", "ok", "keys", "took_ms"],
            height=300,
        )


# ── Page: Positions ─────────────────────────────────────────────────────────


def _page_positions(payload: dict[str, Any]) -> None:
    import streamlit as st

    portfolio = _safe_get(payload, "portfolio")
    rows = _flatten_positions(portfolio)
    if not rows:
        st.info("No positions across any account.")
        return

    accounts = sorted({r["account"] for r in rows})
    types = sorted({r["type"] for r in rows})

    ui.section("Filters")
    fc1, fc2 = st.columns(2)
    sel_accounts = fc1.multiselect("Account", accounts, default=accounts, label_visibility="collapsed")
    sel_types = fc2.multiselect("Type", types, default=types, label_visibility="collapsed")
    filtered = [r for r in rows if r["account"] in sel_accounts and r["type"] in sel_types]

    total_mv = sum(r.get("market_value", 0) or 0 for r in filtered if isinstance(r.get("market_value"), (int, float)))
    total_pnl = sum(r.get("unrealized_pnl", 0) or 0 for r in filtered if isinstance(r.get("unrealized_pnl"), (int, float)))
    long_n = sum(1 for r in filtered if (r.get("qty") or r.get("quantity") or 0) > 0)
    short_n = sum(1 for r in filtered if (r.get("qty") or r.get("quantity") or 0) < 0)

    # Aggregate portfolio Greeks (if any positions report them)
    def _g(key: str) -> float:
        return sum(
            (r.get(key) or 0) * (r.get("qty") or 0)
            for r in filtered if isinstance(r.get(key), (int, float))
        )
    port_delta = _g("delta")
    port_theta = _g("theta")
    port_vega = _g("vega")

    ui.kpi_row([
        ui.kpi("Positions", str(len(filtered)), sub=f"{long_n} long · {short_n} short"),
        ui.kpi("Market Value", ui.fmt_usd(total_mv)),
        ui.kpi(
            "Unrealized P&L",
            ui.fmt_usd(total_pnl, signed=True),
            delta_dir=ui.delta_dir_from(total_pnl),
            sub=f"{(total_pnl/total_mv*100 if total_mv else 0):+.2f}% of MV" if total_mv else None,
        ),
        ui.kpi("Δ Delta", ui.fmt_num(port_delta, 1) if port_delta else "—",
               sub=f"Θ {ui.fmt_num(port_theta, 1)} · ν {ui.fmt_num(port_vega, 1)}" if (port_theta or port_vega) else None),
    ])

    # Expiry Wall — group option positions by DTE bucket
    opt_rows = [r for r in filtered if (r.get("type") or "").lower() in {"call", "put", "option"}]
    if opt_rows:
        buckets = {"≤7d": 0, "8–30d": 0, "31–60d": 0, "61–120d": 0, "120d+": 0, "unknown": 0}
        for r in opt_rows:
            dte = r.get("dte")
            if dte is None:
                exp = str(r.get("expiry") or "")
                try:
                    if exp:
                        dte = (dt.date.fromisoformat(exp[:10]) - dt.date.today()).days
                except (ValueError, TypeError):
                    dte = None
            if dte is None:
                buckets["unknown"] += abs(r.get("qty") or 0)
            elif dte <= 7:
                buckets["≤7d"] += abs(r.get("qty") or 0)
            elif dte <= 30:
                buckets["8–30d"] += abs(r.get("qty") or 0)
            elif dte <= 60:
                buckets["31–60d"] += abs(r.get("qty") or 0)
            elif dte <= 120:
                buckets["61–120d"] += abs(r.get("qty") or 0)
            else:
                buckets["120d+"] += abs(r.get("qty") or 0)
        ui.section("Expiry Wall", sub="option contracts by days-to-expiry")
        try:
            import pandas as pd
            df = pd.DataFrame([{"bucket": k, "contracts": v} for k, v in buckets.items() if v])
            if not df.empty:
                st.bar_chart(df, x="bucket", y="contracts", height=200)
        except ImportError:
            ui.smart_table([{"bucket": k, "contracts": v} for k, v in buckets.items()])

    ui.section(f"Holdings · {len(filtered)} rows")
    ui.smart_table(
        filtered,
        column_config={
            "account": ui.col_text("Account", width="medium"),
            "type": ui.col_text("Type"),
            "symbol": ui.col_text("Symbol"),
            "strike": ui.col_usd("Strike"),
            "expiry": ui.col_text("Expiry"),
            "dte": ui.col_int("DTE"),
            "qty": ui.col_int("Qty"),
            "avg_cost": ui.col_usd("Avg Cost"),
            "market_price": ui.col_usd("Mkt Px"),
            "market_value": ui.col_usd("Mkt Value", width="medium"),
            "unrealized_pnl": ui.col_usd("Unrl P&L", width="medium"),
            "delta": ui.col_num("Δ", digits=2) if hasattr(ui, "col_num") else ui.col_text("Δ"),
            "theta": ui.col_num("Θ", digits=2) if hasattr(ui, "col_num") else ui.col_text("Θ"),
            "iv": ui.col_num("IV", digits=3) if hasattr(ui, "col_num") else ui.col_text("IV"),
            "currency": ui.col_text("Ccy"),
        },
        column_order=["account", "type", "symbol", "strike", "expiry", "dte",
                      "qty", "avg_cost", "market_price", "market_value",
                      "unrealized_pnl", "delta", "theta", "iv", "currency"],
        height=560,
    )


# ── Page: P&L and Trades ────────────────────────────────────────────────────


def _page_pnl_trades(payload: dict[str, Any]) -> None:
    import streamlit as st

    pnl = _safe_get(payload, "pnl")
    trades = _safe_get(payload, "trade_log")

    daily = pnl.get("daily") if isinstance(pnl, dict) else None
    today = pnl.get("today_realized") if isinstance(pnl, dict) else None
    week = pnl.get("week_realized") if isinstance(pnl, dict) else None
    mtd = pnl.get("mtd_realized") if isinstance(pnl, dict) else None
    ytd = pnl.get("ytd_realized") if isinstance(pnl, dict) else None

    ui.kpi_row([
        ui.kpi("Today", ui.fmt_usd(today, signed=True) if today is not None else "—",
               delta_dir=ui.delta_dir_from(today)),
        ui.kpi("Week", ui.fmt_usd(week, signed=True) if week is not None else "—",
               delta_dir=ui.delta_dir_from(week)),
        ui.kpi("MTD", ui.fmt_usd(mtd, signed=True) if mtd is not None else "—",
               delta_dir=ui.delta_dir_from(mtd)),
        ui.kpi("YTD", ui.fmt_usd(ytd, signed=True) if ytd is not None else "—",
               delta_dir=ui.delta_dir_from(ytd)),
    ])

    ui.section("Daily realised P&L (last 14 days)")
    if isinstance(daily, list) and daily:
        try:
            import pandas as pd

            df = pd.DataFrame(daily)
            if "date" in df.columns:
                df = df.sort_values("date")
            st.bar_chart(df, x="date", y="realized" if "realized" in df.columns else df.columns[1], height=240)
        except (ImportError, ValueError, KeyError):
            ui.smart_table(daily)
    else:
        st.caption("No daily P&L history.")

    ui.section("Recent trades")
    rows = trades.get("trades") if isinstance(trades, dict) else (trades if isinstance(trades, list) else [])
    if rows:
        ui.smart_table(
            rows[:200],
            column_config={
                "ts": ui.col_text("Time", width="medium"),
                "symbol": ui.col_text("Symbol"),
                "side": ui.col_text("Side"),
                "qty": ui.col_int("Qty"),
                "price": ui.col_usd("Price"),
                "notional": ui.col_usd("Notional", width="medium"),
                "realized_pnl": ui.col_usd("Realized P&L", width="medium"),
            },
            column_order=["ts", "symbol", "side", "qty", "price", "notional", "realized_pnl"],
            height=520,
        )
    else:
        st.caption("No trade log entries.")


# ── Page: Options Flow (UW) ─────────────────────────────────────────────────


def _page_options_flow(payload: dict[str, Any]) -> None:
    import streamlit as st

    uw = _safe_get(payload, "unusual_whales")
    if uw.get("error"):
        st.error(f"Unusual Whales: {uw['error']}")

    # KPIs derived from snapshot merge
    tone = str(uw.get("market_tone") or "?").upper()
    pcr = uw.get("put_call_ratio")
    flow_n = uw.get("options_flow_signal_count") or 0
    total_prem = uw.get("total_options_premium") or 0
    dp_n = uw.get("dark_pool_trade_count") or 0
    dp_not = uw.get("dark_pool_notional") or 0
    cong_n = uw.get("congress_trade_count") or 0
    top_tickers = uw.get("top_flow_tickers") or []

    tone_dir = "up" if tone == "BULLISH" else ("down" if tone == "BEARISH" else "flat")
    ui.kpi_row([
        ui.kpi("Market Tone", tone, delta_dir=tone_dir),
        ui.kpi("Put/Call Ratio", ui.fmt_num(pcr, decimals=3) if pcr is not None else "—",
               delta_dir="down" if (pcr or 0) > 1 else "up"),
        ui.kpi("Flow Signals", ui.fmt_num(flow_n)),
        ui.kpi("Premium (today)", ui.fmt_usd(total_prem)),
        ui.kpi("Dark Pool Notional", ui.fmt_usd(dp_not), sub=f"{dp_n:,} trades"),
        ui.kpi("Congress Trades", ui.fmt_num(cong_n)),
    ])

    if top_tickers:
        ui.section("Top flow tickers")
        st.markdown(
            " ".join(f"<span class='pill ok'>{t}</span>" for t in top_tickers[:10]),
            unsafe_allow_html=True,
        )

    # Market Tide
    tide = uw.get("market_tide_latest") or {}
    if tide:
        ui.section("Market Tide (latest)")
        ui.kpi_row([
            ui.kpi("Net Call Premium", ui.fmt_usd(uw.get("market_tide_net_call_premium"), signed=True),
                   delta_dir=ui.delta_dir_from(uw.get("market_tide_net_call_premium"))),
            ui.kpi("Net Put Premium", ui.fmt_usd(uw.get("market_tide_net_put_premium"), signed=True),
                   delta_dir=ui.delta_dir_from(uw.get("market_tide_net_put_premium"), inverse=True)),
            ui.kpi("Net Volume", ui.fmt_num(tide.get("net_volume"))),
            ui.kpi("As of", ui.fmt_relative_time(tide.get("timestamp"))),
        ])

    # IV ranks
    iv = uw.get("iv_ranks") or {}
    if iv:
        ui.section("IV Rank")
        iv_rows = [{"ticker": k, "iv_rank": v.get("iv_rank") if isinstance(v, dict) else v} for k, v in iv.items()]
        ui.smart_table(
            iv_rows,
            column_config={
                "ticker": ui.col_text("Ticker"),
                "iv_rank": ui.col_progress("IV Rank", min_value=0, max_value=100, width="medium"),
            },
        )

    # GEX walls
    walls = uw.get("gex_walls") or {}
    if walls:
        ui.section("GEX walls (dealer-positioned strikes)")
        for sym, items in walls.items():
            if not items:
                continue
            with st.expander(f"{sym} — {len(items)} walls", expanded=False):
                ui.smart_table(
                    items,
                    column_config={
                        "strike": ui.col_usd("Strike"),
                        "gamma": ui.col_usd("Gamma"),
                        "exposure": ui.col_usd("Exposure"),
                    },
                )

    # Hottest chains
    hot = uw.get("hottest_chains") or []
    if hot:
        ui.section("Hottest option chains")
        ui.smart_table(
            hot,
            column_config={
                "ticker": ui.col_text("Ticker"),
                "expiry": ui.col_text("Expiry", width="medium"),
                "strike": ui.col_usd("Strike"),
                "type": ui.col_text("Type"),
                "volume": ui.col_int("Vol"),
                "open_interest": ui.col_int("OI"),
                "premium": ui.col_usd("Premium", width="medium"),
                "iv": ui.col_pct("IV", decimals=1),
            },
            height=420,
        )

    # Congress trades — the worst offender from screenshot
    congress = uw.get("congress_trades") or []
    if congress:
        ui.section(f"Congress trades · {len(congress)} filings")
        # Normalise the messy shape: drop UUIDs, format dates, surface the
        # parties involved cleanly.
        cleaned = []
        for c in congress:
            if not isinstance(c, dict):
                continue
            cleaned.append({
                "reported": c.get("reported_date") or c.get("filed_date"),
                "transaction": c.get("transaction_date") or c.get("trade_date"),
                "politician": c.get("politician_name") or c.get("reporter") or c.get("name") or "—",
                "party": c.get("party") or "",
                "chamber": c.get("chamber") or c.get("member_type") or "",
                "ticker": c.get("ticker") or c.get("symbol") or "—",
                "issuer": c.get("issuer_name") or c.get("issuer") or "",
                "side": (c.get("transaction_type") or c.get("type") or "").upper(),
                "amount_low": c.get("amount_low") or c.get("amount_min"),
                "amount_high": c.get("amount_high") or c.get("amount_max"),
            })
        ui.smart_table(
            cleaned,
            column_config={
                "reported": ui.col_text("Reported", width="small"),
                "transaction": ui.col_text("Tx Date", width="small"),
                "politician": ui.col_text("Politician", width="medium"),
                "party": ui.col_text("Party"),
                "chamber": ui.col_text("Chamber"),
                "ticker": ui.col_text("Ticker"),
                "issuer": ui.col_text("Issuer", width="large"),
                "side": ui.col_text("Side"),
                "amount_low": ui.col_usd("$ Low"),
                "amount_high": ui.col_usd("$ High"),
            },
            column_order=["reported", "transaction", "politician", "party", "chamber",
                          "ticker", "issuer", "side", "amount_low", "amount_high"],
            height=460,
        )


# ── Page: Index & Flow ──────────────────────────────────────────────────────


def _page_index_flow(payload: dict[str, Any]) -> None:
    import streamlit as st

    pillar = _safe_get(payload, "index_flow")
    if pillar.get("error"):
        st.error(pillar["error"])
        return
    st.caption(f"as of {pillar.get('as_of', '-')} · took {pillar.get('duration_s', 0)}s · 5m cache")

    of = pillar.get("options_flow") or {}
    tone = str(of.get("market_tone") or "?").upper()
    tone_dir = "up" if tone == "BULLISH" else ("down" if tone == "BEARISH" else "flat")
    ui.kpi_row([
        ui.kpi("Market Tone", tone, delta_dir=tone_dir),
        ui.kpi("Put/Call", ui.fmt_num(of.get("put_call_ratio"), decimals=3)),
        ui.kpi("Net Call $", ui.fmt_usd(of.get("net_call_premium"), signed=True),
               delta_dir=ui.delta_dir_from(of.get("net_call_premium"))),
        ui.kpi("Net Put $", ui.fmt_usd(of.get("net_put_premium"), signed=True),
               delta_dir=ui.delta_dir_from(of.get("net_put_premium"), inverse=True)),
        ui.kpi("Dark Pool $", ui.fmt_usd(of.get("dark_pool_notional"))),
        ui.kpi("Flow Signals", ui.fmt_num(of.get("options_flow_signal_count"))),
    ])

    top = of.get("top_flow_tickers") or []
    if top:
        st.markdown(
            "**Top flow tickers:** " + " ".join(f"<span class='pill ok'>{t}</span>" for t in top[:10]),
            unsafe_allow_html=True,
        )

    # ETF flows
    ui.section("Index ETF flows", sub="Δshares × NAV proxy")
    etf = pillar.get("etf_flows") or []
    if etf:
        view = []
        for e in etf:
            view.append({
                "symbol": e.get("symbol"),
                "date": e.get("date"),
                "price": e.get("price"),
                "shares_outstanding": e.get("shares_outstanding"),
                "AUM": e.get("total_assets"),
                "daily_flow": e.get("daily_flow_usd"),
                "note": e.get("error") or ("first sample (need 2+ days)" if e.get("daily_flow_usd") is None and e.get("shares_outstanding") is None else ""),
            })
        ui.smart_table(
            view,
            column_config={
                "symbol": ui.col_text("Symbol"),
                "date": ui.col_text("Date"),
                "price": ui.col_usd("Price"),
                "shares_outstanding": ui.col_int("Shares Out"),
                "AUM": ui.col_usd("AUM"),
                "daily_flow": ui.col_usd("Daily Flow"),
                "note": ui.col_text("Note", width="medium"),
            },
        )
    else:
        st.caption("No ETF flow data.")

    # Breadth
    ui.section("NYSE Breadth")
    breadth = pillar.get("breadth") or {}
    if breadth:
        ui.kpi_row([
            ui.kpi("TRIN", ui.fmt_num(breadth.get("trin"), decimals=3)),
            ui.kpi("TICK", ui.fmt_num(breadth.get("tick"))),
            ui.kpi("ADV-DECL", ui.fmt_num(breadth.get("adv_minus_decl"))),
            ui.kpi("McClellan", ui.fmt_num(breadth.get("mcclellan_oscillator"))),
            ui.kpi("Regime", str(breadth.get("regime", "?")).upper()),
        ])
        if breadth.get("notes"):
            st.caption("  ·  ".join(breadth["notes"]))
    else:
        st.caption("Breadth unavailable.")

    # COT
    ui.section("COT positioning", sub="E-mini index futures")
    cot = pillar.get("cot_positioning") or []
    if cot:
        view = []
        for r in cot:
            sig = r.get("extreme_signal") or {}
            view.append({
                "market": r.get("market"),
                "report_date": r.get("report_date"),
                "leveraged_net": r.get("leveraged_net"),
                "asset_mgr_net": r.get("asset_mgr_net"),
                "dealer_net": r.get("dealer_net"),
                "extreme": sig.get("signal", "") if sig else "",
                "z_score": round(sig.get("z_score", 0), 2) if sig else None,
            })
        ui.smart_table(
            view,
            column_config={
                "market": ui.col_text("Market", width="medium"),
                "report_date": ui.col_text("Report"),
                "leveraged_net": ui.col_int("Levered Net"),
                "asset_mgr_net": ui.col_int("Asset Mgr Net"),
                "dealer_net": ui.col_int("Dealer Net"),
                "extreme": ui.col_text("Extreme"),
                "z_score": ui.col_pct("Z-score", decimals=2),
            },
        )
    else:
        st.caption("COT unavailable (first call downloads ~7 MB ZIP, retry next cycle).")

    errs = pillar.get("errors") or []
    if errs:
        with st.expander(f"Pillar B errors ({len(errs)})"):
            for e in errs:
                st.code(str(e))

    _render_agent_panel(
        agent="flow_analyst",
        title="Ask the Flow Analyst",
        placeholder="e.g. What's the tape saying right now? Any smart-money tickers?",
        state_key="__pillar_b_agent",
        default_prompt="Read today's flow + breadth and call market tone.",
    )


# ── Page: Call Options ──────────────────────────────────────────────────────


def _page_call_options(payload: dict[str, Any]) -> None:
    import streamlit as st

    pillar = _safe_get(payload, "call_options")
    if pillar.get("error"):
        st.error(pillar["error"])
        return
    st.caption(f"as of {pillar.get('as_of', '-')} · took {pillar.get('duration_s', 0)}s")

    summary = pillar.get("summary") or {}
    if summary:
        ui.kpi_row([
            ui.kpi("Candidates", ui.fmt_num(summary.get("candidate_count"))),
            ui.kpi("Top Score", ui.fmt_num(summary.get("top_score"), decimals=2)),
            ui.kpi("Avg Premium", ui.fmt_usd(summary.get("avg_premium"))),
            ui.kpi("Best Symbol", str(summary.get("best_symbol") or "—")),
        ])

    candidates = pillar.get("candidates") or []
    if candidates:
        ui.section(f"Covered call candidates · {len(candidates)}")
        ui.smart_table(
            candidates,
            column_config={
                "symbol": ui.col_text("Symbol"),
                "score": ui.col_pct("Score", decimals=2),
                "price": ui.col_usd("Spot"),
                "call_premium": ui.col_usd("Premium"),
                "call_delta": ui.col_pct("Δ", decimals=3),
                "iv": ui.col_pct("IV", decimals=2),
                "dte": ui.col_int("DTE"),
            },
            height=420,
        )
    else:
        st.caption("No covered call candidates surfaced this cycle.")

    errs = pillar.get("errors") or []
    if errs:
        with st.expander(f"Pillar A errors ({len(errs)})"):
            for e in errs:
                st.code(str(e))

    _render_agent_panel(
        agent="options_analyst",
        title="Ask the Options Analyst",
        placeholder="e.g. Best premium income for next 30 days?",
        state_key="__pillar_a_agent",
        default_prompt="Rank today's covered-call candidates by risk-adjusted yield.",
    )


# ── Page: Quant Research ────────────────────────────────────────────────────


def _page_quant(payload: dict[str, Any]) -> None:
    import streamlit as st

    pillar = _safe_get(payload, "quant_research")
    if pillar.get("error"):
        st.error(pillar["error"])
        return
    st.caption(f"as of {pillar.get('as_of', '-')} · took {pillar.get('duration_s', 0)}s")

    # Display whatever sections the pillar publishes
    for key, val in pillar.items():
        if key in {"as_of", "duration_s", "errors", "error"}:
            continue
        if isinstance(val, list) and val and isinstance(val[0], dict):
            ui.section(key.replace("_", " ").title())
            ui.smart_table(val[:200])
        elif isinstance(val, dict) and val:
            with st.expander(key.replace("_", " ").title(), expanded=False):
                st.json(val, expanded=False)

    errs = pillar.get("errors") or []
    if errs:
        with st.expander(f"Pillar C errors ({len(errs)})"):
            for e in errs:
                st.code(str(e))

    _render_agent_panel(
        agent="quant_analyst",
        title="Ask the Quant",
        placeholder="e.g. Any mean-reversion setups in the universe?",
        state_key="__pillar_c_agent",
        default_prompt="Find me the top 3 quant signals and explain.",
    )


# ── Page: Health & Feeds ────────────────────────────────────────────────────


def _page_health(payload: dict[str, Any]) -> None:
    import streamlit as st

    health = _safe_get(payload, "health")
    backbone = _safe_get(payload, "backbone")
    feeds = _safe_get(payload, "live_feeds")
    api_feeds = _safe_get(payload, "api_feeds")

    label, ok_n, total_n = _derive_health_status(health)
    ui.kpi_row([
        ui.kpi("Subsystems", f"{ok_n}/{total_n}", sub=label,
               delta_dir="up" if ok_n == total_n else "down"),
        ui.kpi("Backbone", str(backbone.get("status", "?")).upper() if backbone else "—"),
        ui.kpi("API Feeds", str(len(api_feeds) if isinstance(api_feeds, (list, dict)) else 0)),
        ui.kpi("Live Feeds", str(len(feeds) if isinstance(feeds, (list, dict)) else 0)),
    ])

    ui.section("Subsystems", sub="status · key · note · latency")
    subs = health.get("subsystems") if isinstance(health, dict) else None
    if isinstance(subs, list) and subs:
        ui.smart_table(subs)
    elif isinstance(subs, dict) and subs:
        rows = []
        for name, v in subs.items():
            if isinstance(v, dict):
                rows.append({
                    "subsystem": name,
                    "status": v.get("status", "?"),
                    "kind": v.get("kind", ""),
                    "key": "yes" if v.get("key_present") else "no",
                    "note": v.get("note", ""),
                    "latency_ms": v.get("latency_ms"),
                    "last_success": v.get("last_success", ""),
                    "last_error": (v.get("last_error") or "")[:80],
                })
            else:
                rows.append({"subsystem": name, "status": str(v)})
        ui.smart_table(
            rows,
            column_config={
                "subsystem": ui.col_text("Subsystem", width="medium"),
                "status": ui.col_text("Status"),
                "kind": ui.col_text("Kind"),
                "key": ui.col_text("Key"),
                "note": ui.col_text("Note", width="medium"),
                "latency_ms": ui.col_int("ms"),
                "last_success": ui.col_text("Last OK"),
                "last_error": ui.col_text("Last Error", width="medium"),
            },
            column_order=["subsystem", "status", "kind", "key", "note",
                          "latency_ms", "last_success", "last_error"],
            height=480,
        )
    else:
        st.caption("No subsystem data.")

    ui.section("Backbone modules")
    if isinstance(backbone, dict):
        modules = backbone.get("modules") or []
        if modules:
            ui.smart_table(modules)
        else:
            st.json(backbone, expanded=False)
    else:
        st.caption("No backbone data.")

    ui.section("API feeds")
    if isinstance(api_feeds, list) and api_feeds:
        ui.smart_table(api_feeds)
    elif isinstance(api_feeds, dict):
        rows = [{"feed": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in api_feeds.items()]
        ui.smart_table(rows)
    else:
        st.caption("No API feed data.")

    ui.section("Live market feed")
    if feeds:
        if isinstance(feeds, list):
            ui.smart_table(feeds)
        else:
            st.json(feeds, expanded=False)
    else:
        st.caption("No live feed data.")


# ── Page: War Room ──────────────────────────────────────────────────────────


def _page_war_room(payload: dict[str, Any]) -> None:
    import streamlit as st

    war = _safe_get(payload, "war_room")
    if not war:
        st.info("War Room collector empty.")
        return

    ui.kpi_row([
        ui.kpi("Composite Score", ui.fmt_num(war.get("composite_score"))),
        ui.kpi("Regime", str(war.get("regime", "?")).upper()),
        ui.kpi("Phase", str(war.get("phase", "?")).upper()),
        ui.kpi("Mandate", str(war.get("mandate", "?"))),
    ])

    ui.section("Indicators", sub="weighted composite")
    ind = war.get("indicators") or []
    if ind:
        ui.smart_table(ind)

    ui.section("Arms", sub="target vs actual allocation")
    arms = war.get("arms") or []
    if arms:
        ui.smart_table(arms)

    regime = _safe_get(payload, "regime")
    if regime:
        ui.section("Regime detail")
        st.json(regime, expanded=False)

    moon = _safe_get(payload, "moon")
    if moon:
        ui.section("Moon phase")
        st.json(moon, expanded=False)


# ── Page: Tasks & Briefings ─────────────────────────────────────────────────


def _page_tasks(payload: dict[str, Any]) -> None:
    import streamlit as st

    tasks = _safe_get(payload, "tasks")
    daily = _safe_get(payload, "daily_tasks")

    today = (daily.get("today") if isinstance(daily, dict) else None) or []
    overdue = (daily.get("overdue") if isinstance(daily, dict) else None) or []
    upcoming = (daily.get("upcoming") if isinstance(daily, dict) else None) or []

    ui.kpi_row([
        ui.kpi("Today", str(len(today))),
        ui.kpi("Overdue", str(len(overdue)),
               delta_dir="down" if overdue else "flat"),
        ui.kpi("Upcoming", str(len(upcoming))),
        ui.kpi("All Tasks", str(len(tasks) if isinstance(tasks, list) else
                                 len(tasks.get("tasks", [])) if isinstance(tasks, dict) else 0)),
    ])

    if today:
        ui.section(f"Today · {len(today)}")
        ui.smart_table(today)
    if overdue:
        ui.section(f"Overdue · {len(overdue)}")
        ui.smart_table(overdue)
    if upcoming:
        ui.section(f"Upcoming · {len(upcoming)}")
        ui.smart_table(upcoming)

    if not (today or overdue or upcoming):
        st.caption("No task data.")


# ── Page: Other (Polymarket, Scenarios, Divisions, Raw) ─────────────────────


def _page_other(payload: dict[str, Any]) -> None:
    import streamlit as st

    sub = st.tabs(["Polymarket", "Scenarios", "Divisions", "Decisions", "Raw payload"])

    with sub[0]:
        poly = _safe_get(payload, "polymarket")
        ui.section("Trending markets")
        trending = poly.get("trending") if isinstance(poly, dict) else None
        if trending:
            ui.smart_table(trending)
        else:
            st.caption("No trending data.")
        ui.section("Arbitrage opportunities")
        arbs = poly.get("arbs") if isinstance(poly, dict) else None
        if arbs:
            ui.smart_table(arbs)
        else:
            st.caption("None right now.")

    with sub[1]:
        scen = _safe_get(payload, "scenarios")
        if scen:
            if isinstance(scen, list):
                ui.smart_table(scen)
            else:
                st.json(scen, expanded=False)
        else:
            st.caption("No scenarios.")

    with sub[2]:
        div = _safe_get(payload, "divisions")
        if div:
            if isinstance(div, list):
                ui.smart_table(div)
            elif isinstance(div, dict):
                rows = [{"division": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in div.items()]
                ui.smart_table(rows)
        else:
            st.caption("No division data.")

    with sub[3]:
        _render_decisions_panel()

    with sub[4]:
        st.code(json.dumps(payload, default=_json_default, indent=2)[:200_000], language="json")


# ── Layout & nav ────────────────────────────────────────────────────────────


def _render_alerts_strip(limit: int = 10) -> None:
    """Tail recent alert/log lines if available."""
    import streamlit as st
    from pathlib import Path

    candidates = [
        Path("logs/alerts.log"),
        Path("logs/alerts.jsonl"),
        Path("logs/dashboard.log"),
    ]
    src = next((p for p in candidates if p.exists() and p.stat().st_size > 0), None)
    if src is None:
        return
    try:
        with src.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()[-limit:]
    except OSError:
        return
    if not lines:
        return
    ui.section("Recent Alerts", sub=str(src))
    st.code("".join(lines), language="log")


# ── Page: Risk ──────────────────────────────────────────────────────────────


def _page_risk(payload: dict[str, Any]) -> None:
    import streamlit as st

    portfolio = _safe_get(payload, "portfolio")
    rows = _flatten_positions(portfolio)
    if not rows:
        st.info("No positions — no risk to render.")
        return

    total_mv = sum(r.get("market_value", 0) or 0 for r in rows if isinstance(r.get("market_value"), (int, float)))

    def _g(key: str) -> float:
        return sum((r.get(key) or 0) * (r.get("qty") or 0)
                   for r in rows if isinstance(r.get(key), (int, float)))
    delta = _g("delta")
    gamma = _g("gamma")
    theta = _g("theta")
    vega = _g("vega")

    ui.kpi_row([
        ui.kpi("Δ Delta", ui.fmt_num(delta, decimals=1) if delta else "—",
               sub="$ exposure per +$1 underlying"),
        ui.kpi("Γ Gamma", ui.fmt_num(gamma, decimals=2) if gamma else "—"),
        ui.kpi("Θ Theta", ui.fmt_num(theta, decimals=2) if theta else "—",
               sub="daily decay $", delta_dir=ui.delta_dir_from(theta)),
        ui.kpi("ν Vega", ui.fmt_num(vega, decimals=2) if vega else "—",
               sub="per +1 vol pt"),
    ])

    # Simple parametric stress scenarios
    ui.section("Stress Scenarios", sub="parametric, 1st-order Greeks")
    scenarios = [
        ("SPY +5%",   {"underlying_pct": +5}),
        ("SPY -5%",   {"underlying_pct": -5}),
        ("SPY -10%",  {"underlying_pct": -10}),
        ("Vol +10pt", {"vol_pt": +10}),
        ("Vol -10pt", {"vol_pt": -10}),
        ("1 day decay", {"theta_days": 1}),
        ("5 day decay", {"theta_days": 5}),
    ]
    table = []
    for name, shock in scenarios:
        u_pct = shock.get("underlying_pct", 0) / 100.0
        v_pt = shock.get("vol_pt", 0)
        d_days = shock.get("theta_days", 0)
        # Approximate: assume avg underlying $100 for delta·%·100 conversion if no spot
        pnl_delta = delta * u_pct * 100
        pnl_gamma = 0.5 * gamma * (u_pct * 100) ** 2
        pnl_vega = vega * v_pt
        pnl_theta = theta * d_days
        total = pnl_delta + pnl_gamma + pnl_vega + pnl_theta
        table.append({
            "scenario": name,
            "Δ pnl": round(pnl_delta, 2),
            "Γ pnl": round(pnl_gamma, 2),
            "ν pnl": round(pnl_vega, 2),
            "Θ pnl": round(pnl_theta, 2),
            "total": round(total, 2),
            "% MV": round(total / total_mv * 100, 2) if total_mv else 0,
        })
    ui.smart_table(table, column_config={
        "scenario": ui.col_text("Scenario", width="medium"),
        "Δ pnl": ui.col_usd("Δ"),
        "Γ pnl": ui.col_usd("Γ"),
        "ν pnl": ui.col_usd("ν"),
        "Θ pnl": ui.col_usd("Θ"),
        "total": ui.col_usd("Total"),
        "% MV": ui.col_pct("% MV"),
    })

    # Concentration
    ui.section("Concentration")
    by_sym: dict[str, float] = {}
    for r in rows:
        sym = r.get("symbol", "?")
        by_sym[sym] = by_sym.get(sym, 0) + (r.get("market_value") or 0)
    conc = sorted(({"symbol": s, "market_value": v,
                    "weight_pct": (v / total_mv * 100) if total_mv else 0}
                   for s, v in by_sym.items()),
                  key=lambda x: -abs(x["market_value"]))[:20]
    ui.smart_table(conc, column_config={
        "symbol": ui.col_text("Symbol"),
        "market_value": ui.col_usd("Mkt Value"),
        "weight_pct": ui.col_pct("Weight"),
    })


# ── Page: Blotter ───────────────────────────────────────────────────────────


def _page_blotter(payload: dict[str, Any]) -> None:
    import streamlit as st

    trades = _safe_get(payload, "trade_log")
    rows: list[dict[str, Any]] = []
    if isinstance(trades, list):
        rows = trades
    elif isinstance(trades, dict):
        rows = trades.get("trades") or trades.get("rows") or []

    ui.kpi_row([
        ui.kpi("Trades (window)", str(len(rows))),
        ui.kpi("Buys", str(sum(1 for r in rows if str(r.get("side", "")).lower() in ("buy", "b")))),
        ui.kpi("Sells", str(sum(1 for r in rows if str(r.get("side", "")).lower() in ("sell", "s")))),
        ui.kpi("Notional",
               ui.fmt_usd(sum((r.get("notional") or r.get("amount") or 0) for r in rows
                              if isinstance(r.get("notional", r.get("amount", 0)), (int, float))))),
    ])

    if not rows:
        st.info("No trades in the trade_log payload. Wire `collect_trade_log` in mission_control or"
                " ensure executions are streaming into CentralAccounting.")
        return

    ui.section(f"Executions · {len(rows)} rows")
    ui.smart_table(rows, height=500)


# ── Page: News ──────────────────────────────────────────────────────────────


def _page_news(payload: dict[str, Any]) -> None:
    import streamlit as st

    portfolio = _safe_get(payload, "portfolio")
    symbols = sorted({(p.get("symbol") or "").upper()
                      for a in portfolio.get("accounts", [])
                      for p in (a.get("positions") or [])
                      if p.get("symbol")})

    st.caption(f"Held tickers: {', '.join(symbols) if symbols else '(none)'}")

    try:
        from integrations.finnhub_client import FinnhubClient  # type: ignore
    except ImportError:
        st.warning("integrations.finnhub_client not importable.")
        return

    if not symbols:
        st.info("No symbols held — News page has nothing to fetch.")
        return

    try:
        client = FinnhubClient()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Finnhub client init failed: {exc}")
        return

    chosen = st.multiselect("Tickers", symbols, default=symbols[:5])
    if not chosen:
        return

    for sym in chosen:
        ui.section(sym)
        try:
            items = client.company_news(sym, days=3) if hasattr(client, "company_news") else []
        except Exception as exc:  # noqa: BLE001
            st.caption(f"  {sym}: {exc}")
            continue
        if not items:
            st.caption("No recent headlines.")
            continue
        rows = [{
            "datetime": it.get("datetime") or it.get("date") or "",
            "headline": it.get("headline") or it.get("title") or "",
            "source": it.get("source", ""),
            "url": it.get("url", ""),
        } for it in items[:20]]
        ui.smart_table(rows, column_config={
            "datetime": ui.col_text("When"),
            "headline": ui.col_text("Headline", width="medium"),
            "source": ui.col_text("Source"),
            "url": ui.col_link("Link"),
        }, column_order=["datetime", "headline", "source", "url"], height=320)


PAGES = [
    ("Overview",        "📈", _page_overview),
    ("Positions",       "💼", _page_positions),
    ("Risk",            "🛡️", _page_risk),
    ("P&L · Trades",    "💰", _page_pnl_trades),
    ("Blotter",         "📒", _page_blotter),
    ("Options Flow",    "🌊", _page_options_flow),
    ("Index & Flow",    "📊", _page_index_flow),
    ("Call Options",    "🟢", _page_call_options),
    ("Quant",           "🔬", _page_quant),
    ("War Room",        "⚔️", _page_war_room),
    ("Health",          "🩺", _page_health),
    ("News",            "📰", _page_news),
    ("Tasks",           "✅", _page_tasks),
    ("Functions",       "⚡", _page_functions),
    ("Chat",            "🤖", None),       # uses _render_chat_tab
    ("Briefings",       "📝", None),       # uses _render_briefings_tab
    ("Other",           "📦", _page_other),
]


_PAYLOAD_TTL_SECONDS = 30


def _load_payload(force: bool = False) -> dict[str, Any]:
    """Build (or reuse) the unified payload, cached in session_state for TTL."""
    import streamlit as st

    cache = st.session_state.get("__payload_cache__")
    now = dt.datetime.now()
    if cache and not force:
        ts = cache.get("__built_at__")
        if isinstance(ts, dt.datetime) and (now - ts).total_seconds() < _PAYLOAD_TTL_SECONDS:
            return cache

    placeholder = st.empty()
    placeholder.caption("Collecting data ...")
    with st.spinner("Collecting portfolio, war room, health, and feeds ..."):
        payload: dict[str, Any] = {"ts": now.isoformat()}
        collect = _get_cached_collect()
        for name in _COLLECTOR_NAMES:
            try:
                payload[name] = collect(name)
            except Exception as exc:  # noqa: BLE001
                _log.warning("collector_failed", name=name, error=str(exc))
                payload[name] = {"error": str(exc)}

        st.session_state["__portfolio_payload__"] = payload.get("portfolio") or {}
        st.session_state["__uw_payload__"] = payload.get("unusual_whales") or {}
        pillar = _get_cached_pillar()
        for pn in ("call_options", "index_flow", "quant_research"):
            try:
                payload[pn] = pillar(pn)
            except Exception as exc:  # noqa: BLE001
                _log.warning("pillar_failed", name=pn, error=str(exc))
                payload[pn] = {"error": str(exc)}

    payload["__built_at__"] = now
    st.session_state["__payload_cache__"] = payload
    placeholder.empty()
    return payload


def _render(payload: dict[str, Any]) -> None:
    import streamlit as st

    ui.inject_theme()
    _render_header(payload)

    labels = [f"{icon}  {name}" for name, icon, _ in PAGES]
    with st.sidebar:
        st.markdown("### AAC v2")
        choice = st.radio("Page", labels, label_visibility="collapsed", key="__nav_v2")
        st.markdown("---")
        # Countdown to next refresh
        built = payload.get("__built_at__")
        if isinstance(built, dt.datetime):
            age = (dt.datetime.now() - built).total_seconds()
            remaining = max(0, _PAYLOAD_TTL_SECONDS - int(age))
            st.caption(f"Next refresh in {remaining}s · built {built.strftime('%H:%M:%S')}")
        else:
            st.caption(f"Auto-refresh: {_PAYLOAD_TTL_SECONDS}s · {dt.datetime.now().strftime('%H:%M:%S')}")
        if st.button("Force reload", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("__payload_cache__", None)
            st.rerun()

    idx = labels.index(choice) if choice in labels else 0
    name, _, fn = PAGES[idx]
    st.markdown(f"<div class='section-h'>{name}</div>", unsafe_allow_html=True)
    if fn is not None:
        fn(payload)
    elif name == "Chat":
        _render_chat_tab()
    elif name == "Briefings":
        _render_briefings_tab()


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    try:
        import streamlit as st
    except ImportError:
        _log.error("streamlit_not_installed")
        return

    from streamlit.errors import StreamlitAPIException

    try:
        st.set_page_config(
            page_title="AAC Command Center v2",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except StreamlitAPIException:
        pass

    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=_PAYLOAD_TTL_SECONDS * 1000,
                       key="aac_dashboard_v2_autorefresh")
    except ImportError:
        pass

    payload = _load_payload(force=False)
    _render(payload)


def run_streamlit_dashboard_v2(port: int = 8502) -> int:
    import subprocess

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        __file__, "--server.port", str(port),
        "--server.headless", "true",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    main()
