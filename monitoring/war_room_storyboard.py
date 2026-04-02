#!/usr/bin/env python3
"""
AAC War Room Storyboard -- Interactive browser dashboard.

Visual narrative of the War Room's Monte Carlo simulations, 50-milestone
spiderweb, 5-arm allocations, 7 scenarios, and 12-indicator composite.

Usage:
    launch.bat war-room                           # recommended on Windows
    python monitoring/war_room_storyboard.py          # launch on port 8502
    python monitoring/war_room_storyboard.py --port N  # custom port
"""
from __future__ import annotations

import io
import os
import sys

# pythonw.exe guard
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import streamlit as st

# Page config must be first Streamlit call
st.set_page_config(
    page_title="AAC War Room Storyboard",
    page_icon="\u2694\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

import numpy as np

import strategies.war_room_engine as wre
from config.account_balances import Balances

# ---------------------------------------------------------------------------
# Imports from war_room_engine
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date, timedelta

from strategies.thirteen_moon_doctrine import (
    MOON_BRIEFINGS,
    SACRED_GEOMETRY_OVERLAY,
    ThirteenMoonDoctrine,
)
from strategies.war_room_engine import (
    ACCOUNTS,
    ASSETS,
    CAD_TO_USD,
    CORRELATION_MATRIX,
    CRISIS_DRIFTS,
    CRISIS_VOLS,
    CURRENT_POSITIONS,
    MILESTONES,
    PHASE_THRESHOLDS,
    SCENARIOS,
    SPOT_PRICES,
    STARTING_CAPITAL_CAD,
    ArmAllocation,
    ArmType,
    GreeksResult,
    IndicatorState,
    MCResult,
    MilestoneCategory,
    bs_put,
    check_milestones,
    compute_composite_score,
    generate_mandate,
    get_arm_allocations,
    get_current_phase,
    get_portfolio_value_usd,
    get_spiderweb_chains,
    load_milestone_state,
    run_monte_carlo,
    run_scenario_mc,
)


# ---------------------------------------------------------------------------
# Helper: color coding
# ---------------------------------------------------------------------------
def _regime_color(regime: str) -> str:
    return {"CRISIS": "#ff4444", "ELEVATED": "#ff8c00",
            "WATCH": "#ffd700", "CALM": "#44ff44"}.get(regime, "#888888")


def _pnl_color(val: float) -> str:
    return "#44ff44" if val >= 0 else "#ff4444"


def _prob_bar(p: float, label: str = "") -> str:
    pct = int(p * 100)
    bar = "\u2588" * (pct // 5) + "\u2591" * (20 - pct // 5)
    txt = f"{label} {bar} {pct}%" if label else f"{bar} {pct}%"
    return txt


def _refresh_live_war_room_state() -> dict:
    """Refresh in-memory War Room balances/positions from central account store."""
    global CURRENT_POSITIONS, ACCOUNTS, CAD_TO_USD

    try:
        fx = float(Balances.cad_usd() or CAD_TO_USD)
        wre.CAD_TO_USD = fx

        live_accounts = wre._load_accounts()
        live_positions = wre._load_live_positions()

        wre.ACCOUNTS = live_accounts
        wre.CURRENT_POSITIONS = live_positions
        ACCOUNTS = live_accounts
        CURRENT_POSITIONS = live_positions
        CAD_TO_USD = fx

        return {
            "ok": True,
            "fx": fx,
            "account_count": len(live_accounts),
            "position_count": len(live_positions),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ============================================================================
# SIDEBAR -- Controls
# ============================================================================
st.sidebar.title("\u2694\ufe0f War Room Controls")

# MC paths selector
mc_paths = st.sidebar.select_slider(
    "MC Simulation Paths",
    options=[1_000, 5_000, 10_000, 50_000, 100_000],
    value=10_000,
)
mc_horizon = st.sidebar.slider("Horizon (days)", 30, 365, 90)
mc_seed = st.sidebar.number_input("Random Seed (0=random)", 0, 99999, 0)

st.sidebar.markdown("---")

# Scenario selector
scenario_choice = st.sidebar.selectbox(
    "Scenario Override",
    ["-- Base Case --"] + list(SCENARIOS.keys()),
    format_func=lambda x: SCENARIOS[x]["name"] if x in SCENARIOS else x,
)

st.sidebar.markdown("---")

# Indicator overrides
st.sidebar.subheader("12-Indicator Overrides")
ind_oil = st.sidebar.number_input("Oil ($/bbl)", 50.0, 300.0, float(SPOT_PRICES["oil"]), 1.0)
ind_gold = st.sidebar.number_input("Gold ($/oz)", 1500.0, 10000.0, float(SPOT_PRICES["gold"]), 50.0)
ind_vix = st.sidebar.number_input("VIX", 10.0, 80.0, 25.0, 1.0)
ind_spy = st.sidebar.number_input("SPY", 300.0, 800.0, float(SPOT_PRICES["spy"]), 5.0)
ind_btc = st.sidebar.number_input("BTC", 10000.0, 200000.0, float(SPOT_PRICES["btc"]), 1000.0)
ind_hy = st.sidebar.number_input("HY Spread (bp)", 100.0, 1200.0, 550.0, 25.0)

run_btn = st.sidebar.button("\u26a1 Run Simulation", type="primary", use_container_width=True)

# Keep War Room wired to latest account_balances.json on every rerun
refresh_meta = _refresh_live_war_room_state()

# ============================================================================
# HEADER
# ============================================================================
st.title("\u2694\ufe0f AAC War Room Storyboard")
st.caption("Forward Monte Carlo -- 50-Milestone Spiderweb -- 5-Arm Allocation -- 7 Scenarios -- 12 Indicators -- 13 Moon Doctrine")

# ============================================================================
# ACT 1: THE SITUATION (Current State)
# ============================================================================
st.header("Act 1: The Situation")

col_phase, col_score, col_val, col_regime = st.columns(4)

indicators = IndicatorState(
    oil_price=ind_oil, gold_price=ind_gold, vix=ind_vix,
    spy_price=ind_spy, btc_price=ind_btc, hy_spread_bp=ind_hy,
)
composite = compute_composite_score(indicators)
phase = get_current_phase()
pf_val = get_portfolio_value_usd()

col_phase.metric("Phase", phase.upper())
col_score.metric("Crisis Score", f"{composite['composite_score']:.0f}/100")
col_val.metric("Portfolio", f"${pf_val:,.0f} USD")
regime = composite["regime"]
col_regime.markdown(
    f"<div style='text-align:center;padding:12px;background:{_regime_color(regime)};"
    f"border-radius:8px;color:#000;font-weight:bold;font-size:1.4em'>"
    f"{regime}</div>",
    unsafe_allow_html=True,
)

# 12-indicator heatmap
st.subheader("12-Indicator Heatmap")
ind_scores = composite["individual_scores"]
ind_cols = st.columns(6)
for i, (name, score) in enumerate(ind_scores.items()):
    c = ind_cols[i % 6]
    bg = "#ff4444" if score > 70 else "#ff8c00" if score > 45 else "#ffd700" if score > 25 else "#44ff44"
    c.markdown(
        f"<div style='text-align:center;padding:6px;background:{bg};"
        f"border-radius:6px;color:#000;margin:2px;font-size:0.85em'>"
        f"<b>{name}</b><br/>{score:.0f}</div>",
        unsafe_allow_html=True,
    )

# ============================================================================
# ACT 2: THE POSITIONS (Current Portfolio)
# ============================================================================
st.header("Act 2: The Positions")

pos_col1, pos_col2 = st.columns([3, 2])

with pos_col1:
    st.subheader("IBKR Live Positions")
    pos_rows = []
    for p in CURRENT_POSITIONS:
        pnl = p.pnl
        pnl_pct = p.pnl_pct
        pos_rows.append({
            "Symbol": p.symbol,
            "Type": p.position_type.upper(),
            "Qty": p.quantity,
            "Strike": f"${p.strike:,.1f}" if p.strike else "--",
            "Expiry": p.expiry or "--",
            "Entry": f"${p.entry_price:.2f}",
            "Current": f"${p.current_price:.2f}",
            "P&L": f"${pnl:+,.0f}",
            "P&L%": f"{pnl_pct:+.0f}%",
            "Arm": p.arm.value,
        })
    st.dataframe(pos_rows, use_container_width=True, hide_index=True)

with pos_col2:
    st.subheader("Account Balances")
    fx = float(Balances.cad_usd() or CAD_TO_USD)
    for name, acct in ACCOUNTS.items():
        usd_val = float(acct.get("balance_usd", 0.0) or 0.0)
        cad_val = float(acct.get("balance_cad", 0.0) or 0.0)

        if usd_val <= 0 and cad_val > 0:
            usd_val = cad_val * fx
        if cad_val <= 0 and usd_val > 0:
            cad_val = usd_val / fx if fx else 0.0

        note = acct.get("note", "")
        st.metric(
            name,
            f"${usd_val:,.2f} USD",
            delta=f"C${cad_val:,.2f} CAD | {note[:18]}",
        )

# 5-arm allocation
st.subheader("5-Arm Allocation")
arms = get_arm_allocations(phase)
arm_cols = st.columns(5)
arm_icons = {
    "Iran/Oil Crisis": "\U0001f6e2\ufe0f",
    "Private Credit/BDC": "\U0001f4b3",
    "Crypto & Metals": "\u26cf\ufe0f",
    "DeFi Yield Farm": "\U0001f4ca",
    "TradFi Rotate": "\U0001f3e6",
}
for i, alloc in enumerate(arms):
    with arm_cols[i]:
        icon = arm_icons.get(alloc.name, "\U0001f4bc")
        st.markdown(f"**{icon} {alloc.name}**")
        st.progress(alloc.target_pct)
        st.caption(f"Target: {alloc.target_pct*100:.0f}% | Max: {alloc.max_pct*100:.0f}%")
        st.caption(alloc.entry_conditions[:60])

# ============================================================================
# ACT 3: THE SIMULATION (Monte Carlo)
# ============================================================================
st.header("Act 3: The Simulation")

# Cache MC results in session state
if run_btn or "mc_result" not in st.session_state:
    with st.spinner(f"Running {mc_paths:,} path Monte Carlo over {mc_horizon} days..."):
        seed = mc_seed if mc_seed > 0 else None
        if scenario_choice != "-- Base Case --":
            mc = run_scenario_mc(scenario_choice, n_paths=mc_paths)
            st.info(f"Scenario: **{SCENARIOS[scenario_choice]['name']}** -- {SCENARIOS[scenario_choice]['description']}")
        else:
            mc = run_monte_carlo(n_paths=mc_paths, horizon_days=mc_horizon, seed=seed)
        st.session_state["mc_result"] = mc

mc: MCResult = st.session_state["mc_result"]

# Top metrics
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Mean Portfolio", f"${mc.portfolio_mean:,.0f}")
mc2.metric("Median Portfolio", f"${mc.portfolio_median:,.0f}")
mc3.metric("Bear (P5)", f"${mc.portfolio_p5:,.0f}")
mc4.metric("Bull (P95)", f"${mc.portfolio_p95:,.0f}")
mc5.metric("Runtime", f"{mc.runtime_ms:.0f}ms")

mc_a, mc_b = st.columns(2)

with mc_a:
    st.subheader("VaR / CVaR")
    st.metric("Value at Risk (95%)", f"${mc.var_95:,.0f}")
    st.metric("Conditional VaR (95%)", f"${mc.cvar_95:,.0f}")

with mc_b:
    st.subheader("Key Probabilities")
    probs = [
        ("Oil > $120", mc.prob_oil_above_120),
        ("Gold > $5500", mc.prob_gold_above_3500),
        ("SPY < $600", mc.prob_spy_below_500),
        ("BTC < $55K", mc.prob_btc_below_60k),
        ("Portfolio > $150K", mc.prob_portfolio_above_150k),
        ("Portfolio > $1M", mc.prob_portfolio_above_1m),
    ]
    for label, prob in probs:
        pct = prob * 100
        color = "#ff4444" if pct > 50 else "#ffd700" if pct > 20 else "#44ff44"
        st.markdown(
            f"<span style='color:{color};font-weight:bold'>{label}: {pct:.1f}%</span>"
            f" {'|' * int(pct // 2.5)}",
            unsafe_allow_html=True,
        )

# Per-asset price distributions
st.subheader("Asset Price Distributions (90-Day Forward)")
asset_rows = []
for asset in ASSETS:
    spot = SPOT_PRICES[asset]
    mean = mc.asset_means[asset]
    chg = ((mean - spot) / spot) * 100
    asset_rows.append({
        "Asset": asset.upper(),
        "Spot": f"${spot:,.2f}",
        "Mean": f"${mean:,.2f}",
        "P5 (Bear)": f"${mc.asset_p5[asset]:,.2f}",
        "P25": f"${mc.asset_p25[asset]:,.2f}",
        "P75": f"${mc.asset_p75[asset]:,.2f}",
        "P95 (Bull)": f"${mc.asset_p95[asset]:,.2f}",
        "Fwd Change": f"{chg:+.1f}%",
    })
st.dataframe(asset_rows, use_container_width=True, hide_index=True)

# ============================================================================
# ACT 4: THE SCENARIOS (7 Worlds)
# ============================================================================
st.header("Act 4: The Scenarios")
st.caption("7 possible futures -- probability-weighted with MC validation")

scen_cols = st.columns(4)
for i, (key, sc) in enumerate(SCENARIOS.items()):
    with scen_cols[i % 4]:
        prob_pct = sc["probability"] * 100
        bg = "#ff4444" if prob_pct >= 20 else "#ff8c00" if prob_pct >= 10 else "#555"
        st.markdown(
            f"<div style='background:{bg};padding:10px;border-radius:8px;"
            f"margin:4px;color:#fff'>"
            f"<b>{sc['name']}</b><br/>"
            f"Prob: {prob_pct:.0f}%<br/>"
            f"Oil: ${sc['oil_price']:,.0f} | Gold: ${sc['gold_price']:,.0f}<br/>"
            f"SPY: ${sc['spy_price']:,.0f} | BTC: ${sc['btc_price']:,.0f}<br/>"
            f"VIX: {sc['vix']:.0f} | HY: {sc['hy_spread_bp']:.0f}bp"
            f"</div>",
            unsafe_allow_html=True,
        )

# ============================================================================
# ACT 5: THE SPIDERWEB (50 Milestones)
# ============================================================================
st.header("Act 5: The Spiderweb")
st.caption("50 milestones across 10 categories -- causal chains link triggers to strategy pivots")

# Group by category
categories = {}
for m in MILESTONES:
    cat = m.category.value
    categories.setdefault(cat, []).append(m)

cat_icons = {
    "dollar": "\U0001f4b0", "oil": "\U0001f6e2\ufe0f", "gold": "\u2728",
    "credit": "\U0001f4b3", "crypto": "\U0001f4b1", "macro": "\U0001f4c8",
    "geopolitical": "\U0001f30d", "defi": "\U0001f916", "equity": "\U0001f4c9",
    "phase": "\U0001f3af",
}

# Summary metrics
triggered = [m for m in MILESTONES if m.triggered]
st.metric("Milestones Triggered", f"{len(triggered)} / {len(MILESTONES)}")

# Category tabs
cat_tabs = st.tabs([f"{cat_icons.get(c, '')} {c.upper()}" for c in categories])
for tab, (cat, ms_list) in zip(cat_tabs, categories.items()):
    with tab:
        for m in ms_list:
            status = "\u2705" if m.triggered else "\u23f3"
            conf_bar = "\u2588" * int(m.confidence * 10) + "\u2591" * (10 - int(m.confidence * 10))
            with st.expander(f"{status} #{m.id} {m.name} -- Confidence: {conf_bar} {m.confidence*100:.0f}%"):
                st.markdown(f"**Trigger:** {m.trigger_condition}")
                st.markdown(f"**Action:** {m.strategy_action}")
                st.markdown(f"**Phase:** {m.phase} | **Threshold:** {m.threshold_op} {m.threshold_value:,.1f}")
                if m.leads_to:
                    chain_names = []
                    by_id = {ms.id: ms for ms in MILESTONES}
                    for lid in m.leads_to:
                        linked = by_id.get(lid)
                        if linked:
                            chain_names.append(f"#{lid} {linked.name}")
                    st.markdown(f"**Triggers:** {' -> '.join(chain_names)}")
                if m.triggered_date:
                    st.success(f"Triggered: {m.triggered_date}")

# ============================================================================
# ACT 6: THE MANDATE (Today's Orders)
# ============================================================================
st.header("Act 6: The Mandate")

mandate_btn = st.button("\U0001f4dc Generate Today's Mandate", use_container_width=True)
if mandate_btn:
    with st.spinner("Generating mandate (running quick MC)..."):
        mandate = generate_mandate(indicators=indicators, run_mc=True)

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.subheader(f"{mandate.session.upper()} Session -- {mandate.timestamp}")
        st.metric("Regime", mandate.regime)
        st.metric("Composite Score", f"{mandate.composite_score:.0f}")
        st.metric("Portfolio", f"${mandate.portfolio_value_usd:,.0f}")
        st.code(mandate.mc_summary)

    with m_col2:
        st.subheader("Arm Actions")
        for arm, action in mandate.arm_actions.items():
            st.markdown(f"- **{arm}**: {action}")

        if mandate.risk_alerts:
            st.subheader("\u26a0\ufe0f Risk Alerts")
            for alert in mandate.risk_alerts:
                st.warning(alert)

        if mandate.new_milestones:
            st.subheader("New Milestones Triggered")
            for ms_name in mandate.new_milestones:
                st.success(ms_name)

    st.subheader("Checklist")
    for item in mandate.checklist:
        st.markdown(item)

# ============================================================================
# ACT 7: THE GREEKS (Position Risk)
# ============================================================================
st.header("Act 7: The Greeks")
st.caption("Black-Scholes Greeks for all put positions")

greeks_rows = []
for p in CURRENT_POSITIONS:
    if p.strike and p.expiry and "put" in p.position_type:
        # Calculate DTE
        try:
            exp_date = time.strptime(p.expiry, "%Y-%m-%d")
            dte = max((time.mktime(exp_date) - time.time()) / 86400, 0.01)
        except Exception:
            dte = 30
        T = dte / 365.0
        spot = SPOT_PRICES.get(p.symbol.lower(), p.current_price * 100)
        # Use ATM vol approximation
        vol = CRISIS_VOLS.get(p.symbol.lower(), 0.50)
        g = bs_put(spot, p.strike, T, 0.045, vol)
        greeks_rows.append({
            "Symbol": p.symbol,
            "Strike": f"${p.strike:.1f}",
            "DTE": f"{dte:.0f}",
            "Price": f"${g.price:.4f}",
            "Delta": f"{g.delta:.4f}",
            "Gamma": f"{g.gamma:.6f}",
            "Vega": f"${g.vega:.4f}",
            "Theta": f"${g.theta:.4f}",
            "Score": f"{g.greek_score():.0f}",
            "Money": g.moneyness,
        })

if greeks_rows:
    st.dataframe(greeks_rows, use_container_width=True, hide_index=True)
else:
    st.info("No put positions with strike/expiry data.")

# ============================================================================
# ACT 8: THE 13 MOON DOCTRINE
# ============================================================================
st.header("Act 8: The 13 Moon Doctrine")
st.caption("Lunar cycles, fire peaks, doctrine mandates, sacred geometry, and upcoming events")

doctrine = ThirteenMoonDoctrine()
today = date.today()
current_moon = doctrine.get_current_moon(today)

# ── Current Moon Banner ──
if current_moon:
    moon_num = current_moon.moon_number
    briefing = MOON_BRIEFINGS.get(moon_num, {})
    geometry = SACRED_GEOMETRY_OVERLAY.get(moon_num, {})
    days_in = (today - current_moon.start_date).days
    days_left = (current_moon.end_date - today).days
    pct_through = days_in / max((current_moon.end_date - current_moon.start_date).days, 1)

    mandate_text = ""
    conviction = 0.0
    if current_moon.doctrine_action:
        mandate_text = current_moon.doctrine_action.mandate
        conviction = current_moon.doctrine_action.conviction

    mandate_colors = {
        "PURIFY": "#9b59b6", "DEPLOY": "#27ae60", "HOLD": "#3498db",
        "EXIT": "#e74c3c", "ROTATE": "#f39c12", "REBALANCE": "#1abc9c",
        "ACCUMULATE": "#2ecc71",
    }
    mandate_color = mandate_colors.get(mandate_text.split("/")[0] if "/" in mandate_text else mandate_text, "#95a5a6")

    st.markdown(
        f"<div style='background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px;border-radius:12px;"
        f"border-left:5px solid {mandate_color};margin-bottom:16px'>"
        f"<h2 style='margin:0;color:#e0e0e0'>"
        f"\U0001f319 Moon {moon_num}: {current_moon.lunar_phase_name}</h2>"
        f"<p style='color:#aaa;margin:4px 0'>"
        f"{current_moon.start_date.strftime('%b %d')} \u2014 {current_moon.end_date.strftime('%b %d, %Y')}"
        f" &nbsp;|&nbsp; Day {days_in} of {(current_moon.end_date - current_moon.start_date).days}"
        f" &nbsp;|&nbsp; {days_left} days remaining</p>"
        f"<p style='color:{mandate_color};font-size:1.3em;font-weight:bold;margin:8px 0'>"
        f"\u2694\ufe0f {mandate_text} &nbsp; (conviction: {conviction:.0%})</p>"
        f"<p style='color:#ccc'>{briefing.get('theme', '')}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Progress bar
    st.progress(pct_through, text=f"Moon {moon_num} progress: {pct_through:.0%}")

    # Key dates row
    col_fire, col_new, col_geo, col_freq = st.columns(4)
    with col_fire:
        fp = current_moon.fire_peak_date
        if fp:
            fp_delta = (fp - today).days
            fp_label = f"{fp.strftime('%b %d')}" + (f" ({fp_delta}d)" if fp_delta > 0 else " (PAST)" if fp_delta < 0 else " **TODAY**")
        else:
            fp_label = "N/A"
        st.metric("\U0001f525 Fire Peak", fp_label)
    with col_new:
        nm = current_moon.new_moon_date
        if nm:
            nm_delta = (nm - today).days
            nm_label = f"{nm.strftime('%b %d')}" + (f" ({nm_delta}d)" if nm_delta > 0 else " (PAST)" if nm_delta < 0 else " **TODAY**")
        else:
            nm_label = "N/A"
        st.metric("\U0001f311 New Moon", nm_label)
    with col_geo:
        st.metric("\U0001f4d0 Geometry", geometry.get("geometry", "N/A"))
    with col_freq:
        freq = geometry.get("frequency_hz", "")
        fname = geometry.get("frequency_name", "")
        st.metric("\U0001f3b5 Frequency", f"{freq} Hz" if freq else "N/A", delta=fname if fname else None)

    # Briefing expander
    with st.expander(f"\U0001f4dc Moon {moon_num} Briefing & Sacred Geometry", expanded=False):
        b_cols = st.columns(2)
        with b_cols[0]:
            st.markdown("**Lunar Phase**")
            st.write(briefing.get("lunar", ""))
            st.markdown("**Astro Highlights**")
            st.write(briefing.get("astro_highlights", ""))
            st.markdown("**Market Implication**")
            st.write(briefing.get("market_implication", ""))
            st.markdown("**Empirical**")
            st.write(briefing.get("empirical", ""))
        with b_cols[1]:
            st.markdown("**Sacred Geometry**")
            st.write(f"Shape: {geometry.get('geometry', 'N/A')}")
            st.write(geometry.get("description", ""))
            if "platonic_solid" in geometry:
                st.write(f"Platonic Solid: {geometry['platonic_solid']}")
            if "phi_link" in geometry:
                st.write(f"Phi Link: {geometry['phi_link']}")
            if "correlation" in geometry:
                st.markdown("**Portfolio Correlation**")
                st.write(geometry["correlation"])
            if current_moon.doctrine_action:
                st.markdown("**Doctrine Mandate**")
                st.write(current_moon.doctrine_action.description)
                if current_moon.doctrine_action.targets:
                    st.write(f"Targets: {', '.join(current_moon.doctrine_action.targets)}")

    # Events in this moon cycle
    evt_count = (len(current_moon.astrology_events) + len(current_moon.phi_markers) +
                 len(current_moon.financial_events) + len(current_moon.world_events) +
                 len(current_moon.aac_events))
    with st.expander(f"\U0001f4c5 Events This Moon Cycle ({evt_count} total)", expanded=False):
        evt_rows = []
        for e in current_moon.astrology_events:
            delta = (e.date - today).days
            evt_rows.append({"Date": e.date.strftime("%b %d"), "Days": delta, "Type": "\u2b50 Astro",
                             "Event": e.name, "Impact": e.impact})
        for e in current_moon.phi_markers:
            delta = (e.date - today).days
            evt_rows.append({"Date": e.date.strftime("%b %d"), "Days": delta, "Type": "\u03c6 Phi",
                             "Event": e.label, "Impact": f"{e.resonance_strength:.0%}"})
        for e in current_moon.financial_events:
            delta = (e.date - today).days
            evt_rows.append({"Date": e.date.strftime("%b %d"), "Days": delta, "Type": "\U0001f4b0 Finance",
                             "Event": e.name, "Impact": e.impact})
        for e in current_moon.world_events:
            delta = (e.date - today).days
            evt_rows.append({"Date": e.date.strftime("%b %d"), "Days": delta, "Type": "\U0001f30d World",
                             "Event": e.name, "Impact": e.impact})
        for e in current_moon.aac_events:
            delta = (e.date - today).days
            evt_rows.append({"Date": e.date.strftime("%b %d"), "Days": delta, "Type": "\U0001f916 AAC",
                             "Event": e.name, "Impact": e.impact})
        evt_rows.sort(key=lambda r: r["Days"])
        if evt_rows:
            st.dataframe(evt_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No events catalogued for this cycle.")

# ── Upcoming Alerts (next 14 days) ──
alerts = doctrine.get_events_with_lead_time(days_ahead=14, target=today)
if alerts:
    with st.expander(f"\u26a0\ufe0f Upcoming Alerts \u2014 Next 14 Days ({len(alerts)})", expanded=True):
        alert_rows = []
        for a in sorted(alerts, key=lambda x: x.days_until):
            pri_colors = {"CRITICAL": "\U0001f534", "HIGH": "\U0001f7e0", "MEDIUM": "\U0001f7e1", "LOW": "\u26aa"}
            alert_rows.append({
                "": pri_colors.get(a.priority, "\u26aa"),
                "Date": a.event_date.strftime("%b %d"),
                "In": f"{a.days_until}d",
                "Type": a.event_type.title(),
                "Event": a.event_name,
                "Action": a.lead_time_action,
                "Priority": a.priority,
            })
        st.dataframe(alert_rows, use_container_width=True, hide_index=True)

# ── Full 14-Moon Timeline ──
with st.expander("\U0001f30d Full 14-Moon Timeline (Moon 0 \u2192 Moon 13)", expanded=False):
    timeline_rows = []
    for cycle in doctrine.moon_cycles:
        mn = cycle.moon_number
        b = MOON_BRIEFINGS.get(mn, {})
        g = SACRED_GEOMETRY_OVERLAY.get(mn, {})
        is_current = current_moon and mn == current_moon.moon_number
        mandate_str = cycle.doctrine_action.mandate if cycle.doctrine_action else ""
        conv = f"{cycle.doctrine_action.conviction:.0%}" if cycle.doctrine_action else ""
        fp_str = cycle.fire_peak_date.strftime("%b %d") if cycle.fire_peak_date else ""
        timeline_rows.append({
            "Moon": f"{'>>> ' if is_current else ''}{mn}",
            "Name": cycle.lunar_phase_name,
            "Dates": f"{cycle.start_date.strftime('%b %d')} - {cycle.end_date.strftime('%b %d')}",
            "Fire Peak": fp_str,
            "Mandate": mandate_str,
            "Conv.": conv,
            "Geometry": g.get("geometry", ""),
            "Hz": g.get("frequency_hz", ""),
            "Theme": b.get("theme", ""),
        })
    st.dataframe(timeline_rows, use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;font-size:0.8em'>"
    "AAC War Room Storyboard v1.1 -- "
    f"{len(ASSETS)} assets -- {len(MILESTONES)} milestones -- "
    f"{len(SCENARIOS)} scenarios -- {len(CURRENT_POSITIONS)} positions -- "
    f"14 moon cycles"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================================
# CLI entry point
# ============================================================================
if __name__ == "__main__":
    # This file is run by: streamlit run monitoring/war_room_storyboard.py
    pass
