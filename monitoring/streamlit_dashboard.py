#!/usr/bin/env python3
"""
AAC Unified Matrix Monitor — Streamlit Dashboard v2.0
======================================================
Modernized dashboard consolidating ALL AAC subsystems into a single view.
Reads cached JSON snapshots + optional live connections.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

try:
    import streamlit as st
except ImportError:
    raise SystemExit("streamlit is required: pip install streamlit")

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data loaders — all guarded, return dicts or empty
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    """Load a JSON file, return empty dict on failure."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("JSON load error %s: %s", path, e)
    return {}


def load_balance_snapshot() -> dict:
    return _load_json(PROJECT_ROOT / "balance_snapshot.json")


def load_matrix_maximizer() -> dict:
    return _load_json(PROJECT_ROOT / "data" / "matrix_maximizer_latest.json")


def load_latest_helix_briefing() -> dict:
    import glob
    briefing_dir = PROJECT_ROOT / "data" / "storm_lifeboat"
    files = sorted(glob.glob(str(briefing_dir / "helix_briefing_*.json")))
    if files:
        return _load_json(Path(files[-1]))
    return {}


def load_system_registry() -> dict:
    try:
        from monitoring.aac_system_registry import SystemRegistry
        return SystemRegistry().collect_full_snapshot()
    except Exception as e:
        logger.debug("Registry error: %s", e)
        return {}


def load_lunar_position() -> dict:
    try:
        from strategies.storm_lifeboat.lunar_phi import LunarPhiEngine
        lp = LunarPhiEngine()
        pos = lp.get_position()
        return {
            "moon_number": pos.moon_number,
            "moon_name": pos.moon_name,
            "phase": pos.phase.value if hasattr(pos.phase, "value") else str(pos.phase),
            "day_in_moon": pos.day_in_moon,
            "in_phi_window": pos.in_phi_window,
            "phi_coherence": pos.phi_coherence,
            "position_multiplier": pos.position_multiplier,
        }
    except Exception as e:
        logger.debug("Lunar position error: %s", e)
        return {}


def load_scenario_heatmap() -> list:
    try:
        from strategies.storm_lifeboat.scenario_engine import ScenarioEngine
        return ScenarioEngine().get_risk_heatmap()
    except Exception as e:
        logger.debug("Scenario heatmap error: %s", e)
        return []


def load_turboquant_report() -> dict:
    """Load TurboQuant compression stats and similar states."""
    try:
        from strategies.turboquant_engine import get_compression_report, get_market_index
        report = get_compression_report()
        idx = get_market_index()
        report["recent_entries"] = [
            {"timestamp": e.timestamp, "regime": e.regime}
            for e in idx.entries[-20:]
        ]
        return report
    except Exception as e:
        logger.debug("TurboQuant error: %s", e)
        return {}


def load_ibkr_positions() -> dict:
    """Connect to IBKR (port 7496 LIVE) and fetch positions + account data."""
    try:
        from ib_insync import IB
        port = int(os.environ.get("IBKR_PORT", "7496"))
        ib = IB()
        ib.connect("127.0.0.1", port, clientId=99, timeout=8, readonly=True)
        ib.sleep(1)

        acct_values = ib.accountValues()
        positions = ib.positions()
        orders = ib.openOrders()

        # Parse account summary
        acct = {}
        for av in acct_values:
            if av.currency in ("CAD", "USD", "BASE"):
                acct[f"{av.tag}_{av.currency}"] = av.value

        net_liq = float(acct.get("NetLiquidation_CAD", 0))
        cash = float(acct.get("TotalCashValue_CAD", 0))
        unrealized = float(acct.get("UnrealizedPnL_USD", 0))

        pos_list = []
        for p in positions:
            c = p.contract
            pos_list.append({
                "Symbol": c.localSymbol or c.symbol,
                "Type": c.right if hasattr(c, "right") and c.right else "STK",
                "Strike": getattr(c, "strike", ""),
                "Expiry": getattr(c, "lastTradeDateOrContractMonth", ""),
                "Qty": p.position,
                "Avg Cost": round(p.avgCost, 2),
                "Market Val": round(p.position * p.avgCost, 2),
            })

        ib.disconnect()
        return {
            "status": "ok",
            "net_liquidation": net_liq,
            "cash": cash,
            "unrealized_pnl": unrealized,
            "positions": pos_list,
            "open_orders": len(orders),
            "account": acct.get("AccountCode_BASE", ""),
        }
    except Exception as e:
        logger.debug("IBKR connection error: %s", e)
        return {"status": "error", "error": str(e), "positions": []}



# ---------------------------------------------------------------------------
# Streamlit Dashboard — main entry
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AAC Matrix Monitor",
        page_icon="\U0001f9ec",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        "<h1 style='text-align:center;'>\U0001f9ec AAC Matrix Monitor</h1>"
        "<p style='text-align:center;color:#888;'>"
        "Autonomous Algorithmic Command -- Unified Dashboard</p>",
        unsafe_allow_html=True,
    )

    # Auto-refresh slider
    refresh_secs = st.sidebar.slider("Auto-refresh (sec)", 30, 600, 120)

    # --- Load all data (cached in session state) --------------------
    if "last_refresh" not in st.session_state or (
        datetime.now(timezone.utc) - st.session_state.last_refresh
    ).total_seconds() > refresh_secs:
        with st.spinner("Loading system data..."):
            st.session_state.balances = load_balance_snapshot()
            st.session_state.mm = load_matrix_maximizer()
            st.session_state.helix = load_latest_helix_briefing()
            st.session_state.registry = load_system_registry()
            st.session_state.lunar = load_lunar_position()
            st.session_state.heatmap = load_scenario_heatmap()
            st.session_state.ibkr = load_ibkr_positions()
            st.session_state.tq = load_turboquant_report()
            st.session_state.last_refresh = datetime.now(timezone.utc)

    balances = st.session_state.get("balances", {})
    mm = st.session_state.get("mm", {})
    helix = st.session_state.get("helix", {})
    registry = st.session_state.get("registry", {})
    lunar = st.session_state.get("lunar", {})
    heatmap = st.session_state.get("heatmap", [])
    ibkr = st.session_state.get("ibkr", {})
    tq = st.session_state.get("tq", {})

    # --- Sidebar ----------------------------------------------------
    st.sidebar.markdown("### Controls")
    if st.sidebar.button("\U0001f504 Force Refresh"):
        st.session_state.pop("last_refresh", None)
        st.rerun()

    last_ref = st.session_state.get("last_refresh")
    if last_ref:
        st.sidebar.caption(f"Last refresh: {last_ref.strftime('%H:%M:%S UTC')}")

    # 13-Moon storyboard sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### \U0001f319 13-Moon Doctrine")
    storyboard_path = PROJECT_ROOT / "data" / "storyboard" / "thirteen_moon_storyboard.html"
    if storyboard_path.exists():
        if st.sidebar.button("Open Storyboard"):
            import webbrowser
            webbrowser.open(f"file:///{str(storyboard_path).replace(os.sep, '/')}")
            st.sidebar.success("Opened in browser")

    # ================================================================
    # TOP ROW -- Portfolio Totals
    # ================================================================
    ibkr_nl = ibkr.get("net_liquidation", 0)
    moo_data = balances.get("moomoo", {})
    poly_data = balances.get("polymarket", {})
    moo_nl = moo_data.get("net_liquidation", 0)
    poly_nl = poly_data.get("net_liquidation", 0)
    summary = balances.get("_summary", {})
    total_usd = summary.get("total_usd_approx", ibkr_nl * 0.72 + moo_nl + poly_nl)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("\U0001f4b0 Total (est. USD)", f"${total_usd:,.0f}")
    c2.metric("\U0001f3db\ufe0f IBKR Net Liq", f"${ibkr_nl:,.2f} CAD" if ibkr_nl else "--")
    c3.metric("\U0001f4f1 Moomoo", f"${moo_nl:,.2f}" if moo_nl else "--")
    c4.metric("\U0001f3b2 Polymarket", f"${poly_nl:,.2f} USDC" if poly_nl else "--")
    c5.metric("\U0001f4ca IBKR Unreal P&L", f"${ibkr.get('unrealized_pnl', 0):,.2f} USD")

    st.markdown("---")

    # ================================================================
    # TABS
    # ================================================================
    tab_positions, tab_strategies, tab_regime, tab_health, tab_tq = st.tabs([
        "\U0001f4cb Positions & P/L",
        "\U0001f3af Strategy Matrix",
        "\U0001f300 Regime & Intel",
        "\U0001f3d7\ufe0f System Health",
        "\u26a1 TurboQuant",
    ])

    # -- TAB 1: Positions & P/L -------------------------------------
    with tab_positions:
        st.subheader("\U0001f3db\ufe0f IBKR Portfolio")
        if ibkr.get("status") == "ok":
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("Account", ibkr.get("account", "?"))
            ic2.metric("Net Liquidation", f"${ibkr['net_liquidation']:,.2f} CAD")
            ic3.metric("Cash", f"${ibkr.get('cash', 0):,.2f} CAD")
            ic4.metric("Open Orders", ibkr.get("open_orders", 0))

            positions = ibkr.get("positions", [])
            if positions and pd is not None:
                st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
            elif positions:
                for p in positions:
                    st.write(f"- **{p['Symbol']}** {p['Type']} ${p.get('Strike','')} "
                             f"exp {p.get('Expiry','')} | qty {p['Qty']} | avg ${p['Avg Cost']}")
            else:
                st.info("No IBKR positions")
        else:
            st.warning(f"IBKR offline: {ibkr.get('error', 'not connected')}")
            st.caption("Start TWS/Gateway on port 7496 to see live positions")

        st.markdown("---")

        st.subheader("\U0001f4f1 Moomoo Portfolio")
        if moo_data.get("status") == "ok":
            moo_positions = moo_data.get("positions", [])
            if moo_positions and pd is not None:
                st.dataframe(pd.DataFrame(moo_positions), use_container_width=True, hide_index=True)
            elif moo_positions:
                for p in moo_positions:
                    st.write(f"- **{p.get('symbol','')}** qty={p.get('qty',0)} "
                             f"mkt=${p.get('market_val',0):,.2f} P&L=${p.get('pl_val',0):,.2f}")
            else:
                st.info("No Moomoo positions")
        else:
            st.info("Moomoo data from last balance scan (run _check_all_balances.py to refresh)")

        st.markdown("---")

        st.subheader("\U0001f3b2 Polymarket")
        if poly_data.get("status") == "ok":
            poly_bal = poly_data.get("balances", {})
            pc1, pc2 = st.columns(2)
            pc1.metric("CLOB USDC", f"${poly_bal.get('CLOB_USDC', 0):,.2f}")
            pc2.metric("Proxy Balance", f"${poly_bal.get('CLOB_POLY_PROXY', 0):,.2f}")
        else:
            st.info("Polymarket not connected")

        scan_time = summary.get("scan_time", "")
        if scan_time:
            st.caption(f"Balance snapshot from: {scan_time}")

    # -- TAB 2: Strategy Matrix --------------------------------------
    with tab_strategies:
        col_mm, col_sl = st.columns(2)

        with col_mm:
            st.subheader("\U0001f3af Matrix Maximizer")
            if mm:
                m1, m2, m3 = st.columns(3)
                m1.metric("Mandate", str(mm.get("mandate", "?")).upper())
                m2.metric("Circuit Breaker", mm.get("circuit_breaker", "?"))
                m3.metric("Timestamp", mm.get("timestamp", "?")[:16])

                picks = mm.get("top_picks", mm.get("picks", []))
                if picks and pd is not None:
                    rows = []
                    for p in picks[:8]:
                        rows.append({
                            "Ticker": p.get("ticker"),
                            "Strike": p.get("strike"),
                            "Expiry": p.get("expiry"),
                            "Score": f"{p.get('score', 0):.1f}",
                            "Contracts": p.get("contracts"),
                            "Cost": f"${p.get('cost', 0):,.0f}",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                elif picks:
                    for p in picks[:5]:
                        st.write(f"- {p.get('ticker')} ${p.get('strike')} "
                                 f"exp {p.get('expiry')} score={p.get('score',0):.0f}")
                else:
                    st.info("No picks -- run Matrix Maximizer")
            else:
                st.info("Matrix Maximizer has not run yet")

        with col_sl:
            st.subheader("\U0001f30a Storm Lifeboat")
            if lunar:
                l1, l2, l3 = st.columns(3)
                l1.metric("Moon", f"#{lunar.get('moon_number','')} {lunar.get('moon_name','')}")
                l2.metric("Phase", str(lunar.get('phase', '')).upper())
                l3.metric("Phi Coherence", f"{lunar.get('phi_coherence', 0):.3f}")
                l4, l5 = st.columns(2)
                l4.metric("Day", f"{lunar.get('day_in_moon', '?')}/28")
                l5.metric("Pos. Multiplier", f"{lunar.get('position_multiplier', 1):.2f}x")
            else:
                st.info("Lunar engine not available")

            if helix:
                st.markdown(f"**Helix Briefing** ({helix.get('date', '?')})")
                if helix.get("headline"):
                    st.info(helix["headline"])
                bh1, bh2, bh3 = st.columns(3)
                bh1.metric("Mandate", str(helix.get("mandate", "?")).upper())
                bh2.metric("Regime", str(helix.get("regime", "?")).upper())
                bh3.metric("Coherence", helix.get("coherence_score", "?"))
                if helix.get("risk_alert"):
                    st.warning(f"\u26a0\ufe0f {helix['risk_alert']}")
                trades = helix.get("top_trades", [])
                if trades:
                    with st.expander(f"Top Trades ({len(trades)})"):
                        for t in trades[:10]:
                            st.write(f"- {t}")

        st.markdown("---")
        st.subheader("\U0001f525 Scenario Heatmap")
        if heatmap:
            icons = {"dormant": "\u26aa", "emerging": "\U0001f7e1", "active": "\U0001f7e0",
                     "escalating": "\U0001f534", "peak": "\U0001f525", "receding": "\U0001f7e2"}
            if pd is not None:
                rows = []
                default_icon = "\u26aa"
                for sc in heatmap:
                    status_str = sc.get("status", "")
                    ico = icons.get(status_str, default_icon)
                    rows.append({
                        "Status": f"{ico} {status_str.upper()}",
                        "Scenario": sc.get("name", "?"),
                        "Risk": f"{sc.get('risk_score', 0):.2f}",
                        "Prob": f"{sc.get('probability', 0):.0%}",
                        "Indicators": f"{sc.get('indicators_firing', 0)}/{sc.get('indicators_total', 0)}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                for sc in heatmap:
                    ico = icons.get(sc.get("status", ""), "\u26aa")
                    st.write(f"{ico} **{sc.get('name','?')}** -- Risk: {sc.get('risk_score',0):.2f} "
                             f"P: {sc.get('probability',0):.0%}")
        else:
            st.info("Scenario engine not available")

    # -- TAB 3: Regime & Intel ---------------------------------------
    with tab_regime:
        st.subheader("\U0001f300 Regime Engine")
        regime_str = mm.get("regime", helix.get("regime", "unknown"))
        if regime_str and regime_str != "unknown":
            st.metric("Current Regime", str(regime_str).upper())
        else:
            st.info("Regime data not available -- run Matrix Maximizer or Storm Lifeboat")

        st.markdown("---")
        st.subheader("\U0001f4ca Market Context")
        mc1, mc2, mc3, mc4 = st.columns(4)
        vix = mm.get("vix", helix.get("vix", "--"))
        oil = mm.get("oil_price", helix.get("oil_price", "--"))
        war = mm.get("war_active", helix.get("war_active", None))
        hormuz = mm.get("hormuz_blocked", helix.get("hormuz_blocked", None))
        mc1.metric("VIX", vix if vix != "--" else "--")
        mc2.metric("Oil", f"${oil}" if oil != "--" else "--")
        mc3.metric("War", "YES" if war else ("NO" if war is not None else "--"))
        mc4.metric("Hormuz", "BLOCKED" if hormuz else ("OPEN" if hormuz is not None else "--"))

        st.markdown("---")
        st.subheader("\U0001f504 Capital Rotation Matrix")
        strategies_display = [
            {"Strategy": "WAR ROOM -- IBKR Options", "Broker": "IBKR", "Type": "PRIMARY",
             "Status": "LIVE" if ibkr.get("status") == "ok" else "OFFLINE",
             "Balance": f"${ibkr.get('net_liquidation', 0):,.2f} CAD"},
            {"Strategy": "MOOMOO LEAPS", "Broker": "Moomoo", "Type": "SECONDARY",
             "Status": moo_data.get("status", "offline").upper(),
             "Balance": f"${moo_nl:,.2f} USD"},
            {"Strategy": "POLYMARKET DIVISION", "Broker": "Polymarket", "Type": "SATELLITE",
             "Status": poly_data.get("status", "offline").upper(),
             "Balance": f"${poly_nl:,.2f} USDC"},
            {"Strategy": "CRYPTO HEDGE", "Broker": "NDAX", "Type": "LEGACY",
             "Status": "LIQUIDATED", "Balance": "$0"},
            {"Strategy": "WEALTHSIMPLE TFSA", "Broker": "Wealthsimple", "Type": "SATELLITE",
             "Status": balances.get("wealthsimple", {}).get("status", "not_connected").upper(),
             "Balance": "--"},
        ]
        if pd is not None:
            st.dataframe(pd.DataFrame(strategies_display), use_container_width=True, hide_index=True)
        else:
            for s_row in strategies_display:
                st.write(f"- **{s_row['Strategy']}** [{s_row['Type']}] -- {s_row['Status']} | {s_row['Balance']}")

    # -- TAB 4: System Health ----------------------------------------
    with tab_health:
        if registry:
            s = registry.get("summary", {})
            h1, h2, h3, h4, h5 = st.columns(5)
            h1.metric("APIs", f"{s.get('apis_configured', 0)}/{s.get('total_apis', 0)}")
            h2.metric("Exchanges", f"{s.get('exchanges_online', 0)}/{s.get('exchanges_total', 0)}")
            h3.metric("Strategies", f"{s.get('strategies_ok', 0)}/{s.get('strategies_total', 0)}")
            h4.metric("Departments", f"{s.get('departments_ok', 0)}/{s.get('departments_total', 0)}")
            h5.metric("Infrastructure", f"{s.get('infra_ok', 0)}/{s.get('infra_total', 0)}")

            with st.expander("\U0001f50c Exchange Gateways"):
                for ex in registry.get("exchanges", []):
                    icon = {"green": "\U0001f7e2", "yellow": "\U0001f7e1", "red": "\U0001f534"}.get(ex.get("health"), "\u26aa")
                    lat = f" ({ex['latency_ms']}ms)" if ex.get("latency_ms") else ""
                    st.write(f"{icon} **{ex['name']}** -- {ex.get('detail', '')}{lat}")

            with st.expander("\U0001f3d7\ufe0f Infrastructure Services"):
                for svc in registry.get("infrastructure", []):
                    icon = {"green": "\U0001f7e2", "yellow": "\U0001f7e1", "red": "\U0001f534"}.get(svc.get("health"), "\u26aa")
                    st.write(f"{icon} **{svc['name']}** -- {svc.get('detail', '')}")

            with st.expander("\U0001f9e0 Strategy Engines"):
                for item in registry.get("strategies", []):
                    icon = {"green": "\U0001f7e2", "yellow": "\U0001f7e1", "red": "\U0001f534"}.get(item.get("health"), "\u26aa")
                    st.write(f"{icon} **{item['name']}** -- {item.get('detail', '')}")

            with st.expander(f"\U0001f511 API Inventory ({s.get('total_apis', 0)} APIs)"):
                if pd is not None:
                    api_rows = []
                    for a in registry.get("apis", []):
                        api_rows.append({
                            "Status": "\u2705" if a.get("configured") else "\u274c",
                            "Name": a["name"],
                            "Category": a.get("category", ""),
                            "Priority": a.get("priority", ""),
                        })
                    st.dataframe(pd.DataFrame(api_rows), use_container_width=True, hide_index=True)

            orphans = registry.get("orphans", [])
            if orphans:
                with st.expander(f"\U0001f4c4 Orphan Scripts ({len(orphans)})"):
                    for o in orphans:
                        st.write(f"\U0001f4c4 **{o['script']}** -- {o.get('description', '')[:80]}")
        else:
            st.warning("System Registry not available")

    # -- TAB 5: TurboQuant ------------------------------------------
    with tab_tq:
        st.subheader("\u26a1 TurboQuant Integration Hub")
        st.caption("arXiv:2504.19874 -- 12 indices across all AAC data pipelines")

        # Load full integration report
        tq_hub_report = None
        try:
            from strategies.turboquant_integrations import IntegrationHub
            _hub = IntegrationHub()
            tq_hub_report = _hub.full_report()
        except Exception:
            pass

        if tq_hub_report:
            # Top-level metrics
            tq1, tq2, tq3, tq4 = st.columns(4)
            tq1.metric("Total Entries", tq_hub_report.get("total_entries", 0))
            tq2.metric("Active Indices", f"{tq_hub_report.get('num_active_indices', 0)}/12")
            comp_b = tq_hub_report.get("total_compressed_bytes", 0)
            orig_b = tq_hub_report.get("total_original_bytes", 0)
            tq3.metric("Compressed", f"{comp_b:,} B" if comp_b < 1_000_000 else f"{comp_b/1_000_000:.1f} MB")
            tq4.metric("Compression", f"{tq_hub_report.get('overall_compression_ratio', 0):.1f}x")

            st.markdown("---")

            # Per-index table
            st.subheader("Index Status")
            if pd is not None:
                idx_rows = []
                idx_labels = {
                    "options_scan": "\U0001f4ca Options Scan",
                    "mc_summary": "\U0001f3b2 Monte Carlo",
                    "correlation": "\U0001f517 Correlation",
                    "market_state": "\U0001f30d Market State",
                    "scenario": "\u26a0\ufe0f Scenarios",
                    "greeks": "\U0001f4c9 Greeks",
                    "ml_features": "\U0001f916 ML Features",
                    "price_pattern": "\U0001f4c8 Price Patterns",
                    "sentiment": "\U0001f4ac Sentiment",
                    "polymarket": "\U0001f3b0 Polymarket",
                    "strategy": "\u265f\ufe0f Strategy",
                    "portfolio": "\U0001f4b0 Portfolio",
                }
                for name, info in tq_hub_report.get("indices", {}).items():
                    idx_rows.append({
                        "Index": idx_labels.get(name, name),
                        "Dim": info.get("dimension", 0),
                        "Entries": info.get("entries", 0),
                        "Bits": info.get("bit_width", 3),
                        "Ratio": f"{info.get('compression_ratio', 0):.1f}x",
                        "Status": "\u2705" if info.get("entries", 0) > 0 else "\u23f3",
                    })
                st.dataframe(pd.DataFrame(idx_rows), use_container_width=True, hide_index=True)

            st.markdown("---")

            # Original single-index stats (market state)
            if tq:
                idx_stats = tq.get("index_stats", {})
                bounds = tq.get("theoretical_bounds", {})

                st.subheader("Theoretical MSE Bounds (Theorem 1)")
                if pd is not None:
                    bound_rows = []
                    for b_val in [1, 2, 3, 4]:
                        key = f"{b_val}_bit_mse"
                        mse_val = bounds.get(key, 0)
                        active = "\u2705" if b_val == idx_stats.get("bit_width", 3) else ""
                        ratio = 32 / b_val
                        bound_rows.append({
                            "Bits": b_val,
                            "MSE Bound": f"{mse_val:.3f}",
                            "Compression": f"{ratio:.0f}x",
                            "Active": active,
                        })
                    st.dataframe(pd.DataFrame(bound_rows), use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("Recent Market State Entries")
                recent = tq.get("recent_entries", [])
                if recent and pd is not None:
                    st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)
                elif recent:
                    for r in recent:
                        st.write(f"- {r['timestamp']} [{r['regime']}]")
                else:
                    st.info("No market states indexed yet. Run regime engine to populate.")

            with st.expander("Algorithm & Integration Details"):
                st.markdown("""
**TurboQuant** compresses high-dimensional vectors while preserving:
- **MSE distortion**: within 2.7x of Shannon lower bound
- **Inner-product similarity**: unbiased estimator via QJL residual

**12 AAC Integration Points:**
| # | Index | Dim | Source |
|---|-------|-----|--------|
| 1 | Options Scan | 64 | Matrix Maximizer put recommendations |
| 2 | Monte Carlo | 32 | War Room MC simulation fingerprints |
| 3 | Correlation | 66 | 11x11 cross-asset correlation regimes |
| 4 | Market State | 32 | MacroSnapshot + RegimeState vectors |
| 5 | Scenarios | 43 | Storm Lifeboat 43 crisis probabilities |
| 6 | Greeks | 16 | Portfolio-level Greeks aggregation |
| 7 | ML Features | 32 | Training feature vectors |
| 8 | Price Patterns | 64 | Multi-asset return patterns |
| 9 | Sentiment | 32 | Reddit sentiment vectors |
| 10 | Polymarket | 32 | Thesis-market edge vectors |
| 11 | Strategy | 32 | Options strategy Greeks |
| 12 | Portfolio | 16 | Cross-platform balance snapshots |
""")

        else:
            st.info("TurboQuant Integration Hub not loaded. Import: strategies.turboquant_integrations")

    # -- Footer ------------------------------------------------------
    st.markdown("---")
    fc1, fc2, fc3 = st.columns(3)
    fc1.caption("AAC Matrix Monitor v2.0")
    fc2.caption(f"Refreshed: {st.session_state.get('last_refresh', datetime.now(timezone.utc)).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    fc3.caption("\U0001f9ec Resonance Energy -- Autonomous Algorithmic Command")

    # Auto-refresh via rerun
    import time as _time
    _time.sleep(refresh_secs)
    st.rerun()


# Legacy entry point for backward compatibility
def run_streamlit_dashboard():
    main()


if __name__ == "__main__":
    main()

