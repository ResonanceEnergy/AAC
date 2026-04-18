#!/usr/bin/env python3
"""
AAC War Room Storyboard v2.0 -- Interactive browser dashboard.

Visual narrative of the War Room's Monte Carlo simulations, 50-milestone
spiderweb, 5-arm allocations, 7 scenarios, 15-indicator composite,
13-Moon Doctrine, Polymarket Division, Goal-Mandate Roadmap, Paper Trading
Division bakeoff, and Position Calendar with Roll Discipline.

12 Acts:
  1. The Situation (15-indicator heatmap + regime)
  2. The Positions (portfolio by account)
  3. The Simulation (Monte Carlo forward paths)
  4. The Scenarios (43+ crisis worlds)
  5. The Spiderweb (50 milestones)
  6. The Mandate (today's orders)
  7. The Greeks (position risk)
  8. The 13 Moon Doctrine (lunar cycles + events)
  9. Polymarket Division (thesis matching + scanner)
  10. Goal-Mandate Roadmap (5 missions, 8 sprints)
  11. Paper Trading Division (9 strategies, gate progression)
  12. Position Calendar & Roll Discipline

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
from strategies.war_room_live_feeds import (
    get_last_feed_result,
    update_all_live_data_sync,
)

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

# Polymarket Division — safe import
try:
    from strategies.polymarket_division import get_division_status
    from strategies.polymarket_division.active_scanner import ActiveScanner
    from strategies.polymarket_division.polymc_agent import PolyMCAgent
    from strategies.polymarket_division.polymc_monitor import PolyMCMonitor
    from strategies.polymarket_division.war_room_poly import WarRoomPoly

    POLYMARKET_AVAILABLE = True
except ImportError:
    POLYMARKET_AVAILABLE = False

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

# -- Live Feeds --
st.sidebar.subheader("\U0001f4e1 Live Data Feeds")
if st.sidebar.button("\u26a1 Fetch Live Data", use_container_width=True):
    with st.spinner("Fetching live data from 11 sources..."):
        try:
            live_ind = update_all_live_data_sync()
            st.session_state["live_indicators"] = live_ind
            st.session_state["live_fetch_ts"] = time.time()
            feed = get_last_feed_result()
            if feed:
                st.session_state["last_feed_result"] = feed
            st.sidebar.success("Live data fetched!")
        except Exception as exc:
            st.sidebar.error(f"Live fetch failed: {exc}")

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

# -- Probability Threshold Sliders (Fix 6) --
st.sidebar.subheader("\U0001f3af Probability Thresholds")
thresh_oil = st.sidebar.number_input("Oil Threshold ($)", 50.0, 300.0, 120.0, 5.0)
thresh_gold = st.sidebar.number_input("Gold Threshold ($)", 2000.0, 10000.0, 5500.0, 100.0)
thresh_spy = st.sidebar.number_input("SPY Threshold ($)", 300.0, 800.0, 560.0, 10.0)
thresh_btc = st.sidebar.number_input("BTC Threshold ($)", 10000.0, 200000.0, 50000.0, 5000.0)
thresh_pf1 = st.sidebar.number_input("Portfolio Tier 1 ($)", 50000.0, 1000000.0, 150000.0, 10000.0)
thresh_pf2 = st.sidebar.number_input("Portfolio Tier 2 ($)", 500000.0, 10000000.0, 1000000.0, 100000.0)

st.sidebar.markdown("---")

# -- 15-Indicator Overrides (Fix 5) --
st.sidebar.subheader("\U0001f4ca 15-Indicator Overrides")

# Get defaults from live data if available, else engine defaults
_live_ind = st.session_state.get("live_indicators")
_defaults = _live_ind if _live_ind else IndicatorState()

st.sidebar.caption("**Financial (12)**")
ind_oil = st.sidebar.number_input("Oil ($/bbl)", 50.0, 300.0, float(_defaults.oil_price), 1.0)
ind_gold = st.sidebar.number_input("Gold ($/oz)", 1500.0, 10000.0, float(_defaults.gold_price), 50.0)
ind_vix = st.sidebar.number_input("VIX", 10.0, 80.0, float(_defaults.vix), 1.0)
ind_spy = st.sidebar.number_input("SPY", 300.0, 800.0, float(_defaults.spy_price), 5.0)
ind_btc = st.sidebar.number_input("BTC", 10000.0, 200000.0, float(_defaults.btc_price), 1000.0)
ind_hy = st.sidebar.number_input("HY Spread (bp)", 100.0, 1200.0, float(_defaults.hy_spread_bp), 25.0)
ind_bdc_nav = st.sidebar.number_input("BDC NAV Discount (%)", 0.0, 50.0, float(_defaults.bdc_nav_discount), 1.0)
ind_bdc_nonaccrual = st.sidebar.number_input("BDC Non-Accrual (%)", 0.0, 20.0, float(_defaults.bdc_nonaccrual_pct), 0.5)
ind_defi_tvl = st.sidebar.number_input("DeFi TVL Change (%)", -100.0, 200.0, float(_defaults.defi_tvl_change_pct), 5.0)
ind_stablecoin = st.sidebar.number_input("Stablecoin Depeg (%)", 0.0, 10.0, float(_defaults.stablecoin_depeg_pct), 0.1)
ind_fed = st.sidebar.number_input("Fed Funds Rate (%)", 0.0, 10.0, float(_defaults.fed_funds_rate), 0.25)
ind_dxy = st.sidebar.number_input("DXY", 80.0, 130.0, float(_defaults.dxy), 0.5)

st.sidebar.caption("**Sentiment (3)**")
ind_x_sent = st.sidebar.slider("X/Twitter Sentiment", 0.0, 1.0, float(_defaults.x_sentiment), 0.05)
ind_news = st.sidebar.slider("News Severity", 0.0, 1.0, float(_defaults.news_severity), 0.05)
ind_fg = st.sidebar.slider("Fear & Greed Index", 0.0, 100.0, float(_defaults.fear_greed_index), 1.0)

run_btn = st.sidebar.button("\u26a1 Run Simulation", type="primary", use_container_width=True)

# Keep War Room wired to latest account_balances.json on every rerun
refresh_meta = _refresh_live_war_room_state()

# ============================================================================
# HEADER
# ============================================================================
st.title("\u2694\ufe0f AAC War Room Storyboard")
st.caption("Forward Monte Carlo -- 50-Milestone Spiderweb -- 5-Arm Allocation -- 7 Scenarios -- 15 Indicators -- 13 Moon Doctrine -- Goal-Mandate Roadmap -- Paper Trading -- Position Calendar")

# -- Data Freshness Banner (Fix 4) --
_live_ts = st.session_state.get("live_fetch_ts")
if _live_ts:
    _age_min = (time.time() - _live_ts) / 60
    if _age_min < 60:
        st.success(f"\U0001f4e1 Live data: {_age_min:.0f} min ago")
    else:
        st.warning(f"\u26a0\ufe0f Live data is {_age_min / 60:.1f} hours stale -- click 'Fetch Live Data' to refresh")
else:
    st.warning("\u26a0\ufe0f Using hardcoded spot prices (Apr 9). Click '\u26a1 Fetch Live Data' in sidebar to get real-time data.")

# ============================================================================
# ACT 1: THE SITUATION (Current State)
# ============================================================================
st.header("Act 1: The Situation")

col_phase, col_score, col_val, col_regime = st.columns(4)

indicators = IndicatorState(
    oil_price=ind_oil, gold_price=ind_gold, vix=ind_vix,
    spy_price=ind_spy, btc_price=ind_btc, hy_spread_bp=ind_hy,
    bdc_nav_discount=ind_bdc_nav, bdc_nonaccrual_pct=ind_bdc_nonaccrual,
    defi_tvl_change_pct=ind_defi_tvl, stablecoin_depeg_pct=ind_stablecoin,
    fed_funds_rate=ind_fed, dxy=ind_dxy,
    x_sentiment=ind_x_sent, news_severity=ind_news,
    fear_greed_index=ind_fg,
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

# 15-indicator heatmap
st.subheader("15-Indicator Heatmap")
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
    st.subheader("All Positions by Account")
    # Group by account
    _pos_by_account: dict[str, list] = {}
    for p in CURRENT_POSITIONS:
        _pos_by_account.setdefault(p.account, []).append(p)

    for acct_name, acct_positions in _pos_by_account.items():
        st.markdown(f"**{acct_name}** ({len(acct_positions)} positions)")
        pos_rows = []
        for p in acct_positions:
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
        _mc_kwargs = dict(
            n_paths=mc_paths,
            horizon_days=mc_horizon,
            seed=seed,
            portfolio_value=pf_val if pf_val > 0 else STARTING_CAPITAL_CAD * CAD_TO_USD,
            oil_threshold=thresh_oil,
            gold_threshold=thresh_gold,
            spy_threshold=thresh_spy,
            btc_threshold=thresh_btc,
            portfolio_tier1=thresh_pf1,
            portfolio_tier2=thresh_pf2,
        )
        if scenario_choice != "-- Base Case --":
            mc = run_scenario_mc(scenario_choice, n_paths=mc_paths)
            st.info(f"Scenario: **{SCENARIOS[scenario_choice]['name']}** -- {SCENARIOS[scenario_choice]['description']}")
        else:
            mc = run_monte_carlo(**_mc_kwargs)
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
        (f"Oil > ${mc.oil_threshold:,.0f}", mc.prob_oil_above),
        (f"Gold > ${mc.gold_threshold:,.0f}", mc.prob_gold_above),
        (f"SPY < ${mc.spy_threshold:,.0f}", mc.prob_spy_below),
        (f"BTC < ${mc.btc_threshold:,.0f}", mc.prob_btc_below),
        (f"Portfolio > ${mc.portfolio_tier1/1000:,.0f}K", mc.prob_portfolio_above_tier1),
        (f"Portfolio > ${mc.portfolio_tier2/1000000:,.0f}M", mc.prob_portfolio_above_tier2),
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
# ACT 9: POLYMARKET DIVISION
# ============================================================================
st.header("Act 9: Polymarket Division")

if POLYMARKET_AVAILABLE:
    try:
        div_status = get_division_status()
        loaded = sum(1 for v in div_status.values() if v.get("status") == "loaded")
        total = len(div_status)

        # War Room Poly — thesis matching (geopolitical scenarios)
        try:
            wrp = WarRoomPoly()
            thesis = wrp.get_thesis_chain()
            pressure = thesis.get("pressure_level", "unknown")
            matches = thesis.get("matches", [])
            stages = thesis.get("escalation_stages", [])

            p_col1, p_col2, p_col3 = st.columns(3)
            p_col1.metric("Strategies Loaded", f"{loaded}/{total}")
            p_col2.metric("Pressure Level", pressure.upper())
            p_col3.metric("Thesis Matches", len(matches))

            if stages:
                st.subheader("⚔️ Escalation Stages")
                for stage in stages[:5]:
                    st.markdown(f"- **{stage.get('name', '')}**: {stage.get('description', '')}")

            if matches:
                st.subheader("🎯 Thesis Matches")
                match_rows = []
                for m in matches[:10]:
                    match_rows.append({
                        "Market": m.get("question", m.get("market", "")),
                        "Alignment": f"{m.get('alignment', 0):.0%}",
                        "Thesis": m.get("thesis", ""),
                    })
                if match_rows:
                    st.dataframe(match_rows, use_container_width=True, hide_index=True)
        except Exception:
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("Strategies Loaded", f"{loaded}/{total}")
            p_col2.metric("Division Status", "ACTIVE" if loaded > 0 else "OFFLINE")

        # PolyMC — Monte Carlo portfolio
        try:
            agent = PolyMCAgent()
            mc_result = agent.run_portfolio_monte_carlo()
            portfolio = agent.TARGET_PORTFOLIO

            st.subheader("🎲 PolyMC Portfolio Monte Carlo")
            mc_cols = st.columns(5)
            mc_cols[0].metric("Total Cost", f"${mc_result.get('total_cost', 0):.2f}")
            mc_cols[1].metric("Mean Return", f"${mc_result.get('mean_return', 0):.2f}")
            mc_cols[2].metric("EV %", f"{mc_result.get('ev_pct', 0):.1f}%")
            mc_cols[3].metric("P(Profit)", f"{mc_result.get('prob_profit', 0):.0%}")
            mc_cols[4].metric("Sharpe", f"{mc_result.get('sharpe', 0):.2f}")

            port_rows = []
            for bet in portfolio:
                port_rows.append({
                    "Market": bet.market_name,
                    "Side": bet.side,
                    "Entry": f"${bet.entry_price:.2f}",
                    "Our Prob": f"{bet.our_probability:.0%}",
                    "Size": f"${bet.size_usd:.0f}",
                })
            if port_rows:
                st.dataframe(port_rows, use_container_width=True, hide_index=True)
        except Exception:
            st.info("PolyMC Agent data unavailable.")

        # Monitor — exit signals
        try:
            monitor = PolyMCMonitor()
            signals = monitor.check_exit_signals()
            if signals:
                st.subheader("📡 Exit Signals")
                sig_rows = []
                for sig in signals:
                    sig_rows.append({
                        "Bet": sig.bet_name,
                        "Signal": sig.signal_type,
                        "Price": f"${sig.current_price:.2f}",
                        "Action": sig.action,
                        "Urgency": sig.urgency,
                    })
                st.dataframe(sig_rows, use_container_width=True, hide_index=True)
            else:
                st.success("All positions stable — no exit signals.")
        except Exception:
            st.info("Monitor data unavailable.")

        # Active Scanner status
        try:
            scanner = ActiveScanner(dry_run=True)
            mode = "DRY RUN" if scanner.dry_run else "LIVE"
            sc_cols = st.columns(4)
            sc_cols[0].metric("Scanner Mode", mode)
            sc_cols[1].metric("Bets Today", f"{scanner.daily_bet_count}/{scanner.MAX_DAILY_BETS}")
            sc_cols[2].metric("Max Position", f"${scanner.MAX_POSITION_SIZE_USD}")
            sc_cols[3].metric("Min Edge", f"{scanner.MIN_EDGE_THRESHOLD:.0%}")
        except Exception:
            st.info("Active Scanner data unavailable.")

        # Strategy registry
        st.subheader("📋 Strategy Registry")
        strat_rows = []
        for key, info in div_status.items():
            strat_rows.append({
                "Strategy": info.get("name", key),
                "Status": info.get("status", "unknown"),
                "Description": info.get("description", ""),
            })
        if strat_rows:
            st.dataframe(strat_rows, use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"Polymarket Division data unavailable: {e}")
else:
    st.info("Polymarket Division not installed — enable strategies/polymarket_division to see metrics.")

# ============================================================================
# ACT 10: GOAL-MANDATE ROADMAP
# ============================================================================
st.header("Act 10: Goal-Mandate Roadmap")
st.caption("5 core missions, 8 sprints, honest assessment of what works vs what's broken")

# Core Missions status
_CORE_MISSIONS = [
    {"name": "Find Trades", "icon": "\U0001f50d", "status": "WORKING",
     "desc": "yfinance options chain, UW flow data, War Room 15-indicator composite, Polymarket scanner"},
    {"name": "Execute Trades", "icon": "\u26a1", "status": "WORKING",
     "desc": "IBKR live (15 positions), Moomoo OpenD (degraded), Polymarket CLOB, WealthSimple manual"},
    {"name": "Manage Risk", "icon": "\U0001f6e1\ufe0f", "status": "IN PROGRESS",
     "desc": "ROLL_DISCIPLINE codified, 21-DTE triggers, max 20 contracts. Dead-put gate. Position calendar active."},
    {"name": "Track P&L", "icon": "\U0001f4b0", "status": "WORKING",
     "desc": "account_balances.json central store, per-account positions, FX conversion, MV tracking"},
    {"name": "Monitor Health", "icon": "\U0001f4e1", "status": "WORKING",
     "desc": "Matrix Monitor (24/30 collectors), War Room feeds (10/12), mission_control.py, degradation panel"},
]

miss_cols = st.columns(5)
for i, m in enumerate(_CORE_MISSIONS):
    with miss_cols[i]:
        bg = "#27ae60" if m["status"] == "WORKING" else "#f39c12" if m["status"] == "IN PROGRESS" else "#e74c3c"
        st.markdown(
            f"<div style='text-align:center;padding:12px;background:{bg};"
            f"border-radius:8px;color:#fff;margin:2px'>"
            f"<div style='font-size:1.8em'>{m['icon']}</div>"
            f"<b>{m['name']}</b><br/>"
            f"<small>{m['status']}</small></div>",
            unsafe_allow_html=True,
        )

# Sprint Roadmap
_SPRINTS = [
    {"id": 0, "name": "Cleanup & Foundation", "status": "DONE", "desc": "Archive 37 strategies, fix crashes, lint zero"},
    {"id": 1, "name": "Signal Pipeline", "status": "DONE", "desc": "Reliable market data -> signal path (yfinance + UW + FRED)"},
    {"id": 2, "name": "Execution", "status": "DONE", "desc": "Signal -> order -> confirmation (IBKR live, 23 trades executed)"},
    {"id": 3, "name": "Risk & Roll Discipline", "status": "DONE", "desc": "ROLL_DISCIPLINE codified, 21-DTE rules, dead-put gate"},
    {"id": 4, "name": "P&L Tracking", "status": "DONE", "desc": "account_balances.json, position snapshots, trade logs"},
    {"id": 5, "name": "Monitoring", "status": "DONE", "desc": "Matrix Monitor, Mission Control, War Room Storyboard, 1928+ tests"},
    {"id": 6, "name": "Second Strategy", "status": "IN PROGRESS", "desc": "Polymarket Division active, Paper Trading bakeoff running"},
    {"id": 7, "name": "Automation & Scheduling", "status": "NOT STARTED", "desc": "Unattended running, scheduled scans, auto-roll execution"},
]

with st.expander("\U0001f4cb Sprint Roadmap (0-7)", expanded=True):
    for sp in _SPRINTS:
        icon = "\u2705" if sp["status"] == "DONE" else "\U0001f7e1" if sp["status"] == "IN PROGRESS" else "\u23f3"
        st.markdown(f"{icon} **Sprint {sp['id']}: {sp['name']}** -- _{sp['status']}_ -- {sp['desc']}")

# ============================================================================
# ACT 11: PAPER TRADING DIVISION
# ============================================================================
st.header("Act 11: Paper Trading Division")
st.caption("Strategy bakeoff, gate progression, zero real capital at risk")

# Paper Trading status
_PT_STRATEGIES = [
    {"name": "poly_grid", "division": "Polymarket", "status": "ACTIVE", "phase": "BOOTSTRAP"},
    {"name": "poly_dca", "division": "Polymarket", "status": "ACTIVE", "phase": "BOOTSTRAP"},
    {"name": "poly_momentum", "division": "Polymarket", "status": "ACTIVE", "phase": "BOOTSTRAP"},
    {"name": "poly_mean_rev", "division": "Polymarket", "status": "ACTIVE", "phase": "BOOTSTRAP"},
    {"name": "poly_arb", "division": "Polymarket", "status": "ACTIVE", "phase": "BOOTSTRAP"},
    {"name": "crypto_grid", "division": "Crypto", "status": "ACTIVE", "phase": "BOOTSTRAP"},
    {"name": "crypto_dca", "division": "Crypto", "status": "ACTIVE", "phase": "BOOTSTRAP"},
    {"name": "crypto_momentum", "division": "Crypto", "status": "ACTIVE", "phase": "BOOTSTRAP"},
    {"name": "crypto_mean_rev", "division": "Crypto", "status": "ACTIVE", "phase": "BOOTSTRAP"},
]

pt_col1, pt_col2, pt_col3 = st.columns(3)
pt_col1.metric("Strategies Active", f"{len(_PT_STRATEGIES)}")
pt_col2.metric("Divisions", "2 (Polymarket + Crypto)")
pt_col3.metric("Virtual Capital", "$20K ($10K/div)")

# 7 Non-Negotiable Mandates
with st.expander("\u2694\ufe0f 7 Non-Negotiable Mandates", expanded=False):
    _MANDATES = [
        "M1: Zero real capital at risk (JSON-only persistence)",
        "M2: Deterministic execution (fixed slippage/fees, no randomness)",
        "M3: Continuous scoring (StrategyOptimizer ranks strategies)",
        "M4: Data from councils only (consume INTEL_UPDATE signals)",
        "M5: Doctrine-gated execution (HALT/SAFE_MODE stops activity)",
        "M6: Full audit trail (all fills/scores persisted)",
        "M7: Strategy isolation (independent P&L tracking)",
    ]
    for m in _MANDATES:
        st.markdown(f"\u2705 {m}")

# Gate Promotion Criteria
with st.expander("\U0001f3af Gate Promotion Criteria", expanded=False):
    gate_cols = st.columns(4)
    gate_cols[0].markdown("**BOOTSTRAP -> CALIBRATE**\n\n- 50+ trades\n- System online 48h\n- No crashes")
    gate_cols[1].markdown("**CALIBRATE -> COMPETE**\n\n- 100+ trades\n- Win rate > 50%\n- Max DD < 20%")
    gate_cols[2].markdown("**COMPETE -> VALIDATE**\n\n- 500+ trades\n- Win rate > 55%\n- Sharpe > 1.0\n- Max DD < 15%")
    gate_cols[3].markdown("**VALIDATE -> PILOT**\n\n- 1000+ trades\n- Consistent 4 weeks\n- Sharpe > 1.2\n- Deploy $50-$100 real")

# Strategy registry table
st.subheader("Strategy Registry")
st.dataframe(_PT_STRATEGIES, use_container_width=True, hide_index=True)

# ============================================================================
# ACT 12: POSITION CALENDAR & ROLL DISCIPLINE
# ============================================================================
st.header("Act 12: Position Calendar & Roll Discipline")
st.caption("Upcoming expirations, roll triggers, and codified discipline from Apr 6 post-mortem")

# Roll Discipline rules
rd_col1, rd_col2 = st.columns(2)

with rd_col1:
    st.subheader("\U0001f4dc Roll Discipline Rules")
    rd_rules = wre.ROLL_DISCIPLINE
    st.markdown(f"- **Max contracts/position:** {rd_rules['max_contracts_per_position']}")
    st.markdown(f"- **Roll trigger DTE:** {rd_rules['roll_trigger_dte']} days")
    st.markdown(f"- **Max OTM % (short-dated):** {rd_rules['max_otm_pct_short_dated']:.0%}")
    st.markdown(f"- **Dead-put gate:** {'YES' if rd_rules['dead_put_gate'] else 'NO'} (if bid=$0, do NOT roll)")
    leaps_pct, puts_pct = rd_rules['leaps_vs_puts_allocation']
    st.markdown(f"- **Allocation:** {leaps_pct:.0%} LEAPS / {puts_pct:.0%} directional puts")

with rd_col2:
    st.subheader("\u26a0\ufe0f Post-Mortem Lessons")
    st.warning("Apr 6: All Apr 17 puts expired worthless ($0 bid at 11 DTE). OBDC x65 was untradeable.")
    st.info("Encoded as ROLL_DISCIPLINE hard rules for all future entries.")

# Position Calendar
st.subheader("\U0001f4c5 Position Calendar")

import datetime as _dt

_today = _dt.date.today()
_calendar_events = [
    {"date": "2026-04-10", "event": "XLF 21-DTE roll trigger", "action": "Evaluate XLF $46P May 1. If bid > $0.10 roll to Jun. If $0 -> dead-put gate.", "priority": "HIGH"},
    {"date": "2026-04-10", "event": "March CPI Release", "action": "CPI war-month inflation. IBKR/WS final week positioning.", "priority": "HIGH"},
    {"date": "2026-04-11", "event": "Q1 Bank Earnings Begin", "action": "IV rank assessment for XLF, HYG.", "priority": "MEDIUM"},
    {"date": "2026-04-17", "event": "Apr OPEX + ECB Rate Decision", "action": "IBKR: ARCC/PFF/MAIN/JNK expire. WS: ARCC/JNK/KRE/OBDC expire. ~$1,645 total loss.", "priority": "CRITICAL"},
    {"date": "2026-04-20", "event": "Iran Nuclear Talks", "action": "XLF May 1 roll decision (25 DTE).", "priority": "HIGH"},
    {"date": "2026-04-22", "event": "FAANG Earnings Begin", "action": "TSLA/AMZN/MSFT/META/AAPL/GOOG. Earnings IV plays.", "priority": "MEDIUM"},
    {"date": "2026-04-24", "event": "LQD/EMB 21-DTE roll trigger", "action": "Roll decision for May 15 puts per ROLL_DISCIPLINE.", "priority": "HIGH"},
    {"date": "2026-04-30", "event": "March PCE + Q1 GDP", "action": "Recession signal watch.", "priority": "HIGH"},
    {"date": "2026-05-01", "event": "XLF $46P expiry", "action": "If not rolled Apr 10, expires.", "priority": "CRITICAL"},
    {"date": "2026-05-06", "event": "FOMC May Meeting", "action": "Emergency cut watch + DTE 45 Jun puts.", "priority": "HIGH"},
    {"date": "2026-05-15", "event": "LQD/EMB expiry", "action": "If not rolled Apr 24, expires.", "priority": "CRITICAL"},
    {"date": "2026-05-28", "event": "Jun 18 positions: 21-DTE trigger", "action": "IBKR: SLV/XLE calls, BKLN/HYG puts. WS: OWL puts.", "priority": "HIGH"},
    {"date": "2026-06-18", "event": "Jun OPEX (8 IBKR + 1 WS)", "action": "Major expiry cluster.", "priority": "CRITICAL"},
    {"date": "2026-06-26", "event": "OBDC 21-DTE roll trigger", "action": "OBDC $7.5P x11 Jul 17 roll decision.", "priority": "HIGH"},
    {"date": "2026-07-17", "event": "OBDC $7.5P x11 expiry", "action": "Jul OPEX.", "priority": "CRITICAL"},
]

cal_rows = []
for evt in _calendar_events:
    evt_date = _dt.date.fromisoformat(evt["date"])
    days_until = (evt_date - _today).days
    if days_until < -7:
        continue  # skip events more than a week past
    pri_icon = {
        "CRITICAL": "\U0001f534", "HIGH": "\U0001f7e0", "MEDIUM": "\U0001f7e1",
    }.get(evt["priority"], "\u26aa")
    status = "PAST" if days_until < 0 else "TODAY" if days_until == 0 else f"{days_until}d"
    cal_rows.append({
        "": pri_icon,
        "Date": evt["date"],
        "In": status,
        "Event": evt["event"],
        "Action": evt["action"],
        "Priority": evt["priority"],
    })

if cal_rows:
    st.dataframe(cal_rows, use_container_width=True, hide_index=True)

# Active vs Expiring positions summary
st.subheader("Position Expiry Clusters")
from collections import Counter as _Counter
expiry_counts = _Counter()
for p in CURRENT_POSITIONS:
    if p.expiry:
        expiry_counts[p.expiry] += 1

if expiry_counts:
    exp_rows = []
    for exp_date, count in sorted(expiry_counts.items()):
        try:
            ed = _dt.date.fromisoformat(exp_date)
            days_to = (ed - _today).days
        except ValueError:
            days_to = 999
        exp_rows.append({
            "Expiry": exp_date,
            "Days": days_to,
            "Positions": count,
            "Status": "EXPIRED" if days_to < 0 else "EXPIRING" if days_to <= 7 else "ACTIVE",
        })
    st.dataframe(exp_rows, use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;font-size:0.8em'>"
    "AAC War Room Storyboard v2.0 -- "
    f"{len(ASSETS)} assets -- {len(MILESTONES)} milestones -- "
    f"{len(SCENARIOS)} scenarios -- {len(CURRENT_POSITIONS)} positions -- "
    f"15 indicators -- 14 moon cycles -- "
    f"5 core missions -- 9 strategies -- 8 sprints"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================================
# CLI entry point
# ============================================================================
if __name__ == "__main__":
    # This file is run by: streamlit run monitoring/war_room_storyboard.py
    pass
