#!/usr/bin/env python3
"""
AAC 13-Moon Doctrine — Live Dynamic Dashboard
==============================================
Standalone Streamlit dashboard for the 13-Moon compounding timeline.
Auto-refreshes every 60 seconds. Runs on port 8503 by default.

Usage:
    python launch.py 13-moon
    streamlit run monitoring/thirteen_moon_dashboard.py --server.port 8503
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
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="13-Moon Doctrine | AAC",
    page_icon="\U0001f319",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.thirteen_moon_doctrine import (
    AGE_OF_AQUARIUS,
    CRYPTO_DOCTRINE,
    DALIO_BIG_CYCLE,
    LEAPS_PLAYBOOK,
    MOON_BRIEFINGS,
    SACRED_GEOMETRY_OVERLAY,
    SATURN_NEPTUNE_DEEPDIVE,
    WAR_ROOM_DOCTRINE,
    ThirteenMoonDoctrine,
)

# Polymarket Division — safe import
try:
    from strategies.polymarket_division import get_division_status
    from strategies.polymarket_division.active_scanner import ActiveScanner
    from strategies.polymarket_division.polymc_agent import PolyMCAgent
    from strategies.polymarket_division.polymc_monitor import PolyMCMonitor

    POLYMARKET_AVAILABLE = True
except ImportError:
    POLYMARKET_AVAILABLE = False

# ── Auto-refresh every 60 seconds ──────────────────────────────────────────
REFRESH_SECONDS = 60

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;color:#c084fc'>"
    "\U0001f319 13-Moon Doctrine Timeline \U0001f319</h1>"
    "<p style='text-align:center;color:#888'>"
    "Live compounding calendar -- March 2026 to April 2027 -- "
    "Lunar cycles, phi coherence, sacred geometry, doctrine mandates"
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# BUILD DOCTRINE (fresh on each rerun)
# ---------------------------------------------------------------------------
doctrine = ThirteenMoonDoctrine()
today = date.today()
current_moon = doctrine.get_current_moon(today)

# ---------------------------------------------------------------------------
# SIDEBAR — Navigation & Controls
# ---------------------------------------------------------------------------
st.sidebar.header("\U0001f319 13-Moon Controls")
st.sidebar.caption(f"Today: {today.strftime('%A, %B %d, %Y')}")

auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=True)
if auto_refresh:
    from streamlit_autorefresh import st_autorefresh  # type: ignore[import-untyped]
    st_autorefresh(interval=REFRESH_SECONDS * 1000, limit=None, key="moon_autorefresh")

alert_days = st.sidebar.slider("Alert horizon (days)", 7, 60, 14)
show_deep_dives = st.sidebar.checkbox("Show deep-dive sections", value=False)

# Jump-to-moon selector
moon_names = {c.moon_number: f"Moon {c.moon_number}: {c.lunar_phase_name}" for c in doctrine.moon_cycles}
selected_moon_num = st.sidebar.selectbox(
    "Jump to Moon",
    options=list(moon_names.keys()),
    index=current_moon.moon_number if current_moon else 0,
    format_func=lambda n: moon_names[n],
)

# ============================================================================
# SECTION 1: CURRENT MOON BANNER
# ============================================================================
st.header("Current Moon")

MANDATE_COLORS = {
    "PURIFY": "#9b59b6", "DEPLOY": "#27ae60", "HOLD": "#3498db",
    "EXIT": "#e74c3c", "ROTATE": "#f39c12", "REBALANCE": "#1abc9c",
    "ACCUMULATE": "#2ecc71",
}

if current_moon:
    mn = current_moon.moon_number
    briefing = MOON_BRIEFINGS.get(mn, {})
    geometry = SACRED_GEOMETRY_OVERLAY.get(mn, {})
    days_in = (today - current_moon.start_date).days
    total_days = max((current_moon.end_date - current_moon.start_date).days, 1)
    days_left = (current_moon.end_date - today).days
    pct_through = days_in / total_days

    mandate_text = ""
    conviction = 0.0
    if current_moon.doctrine_action:
        mandate_text = current_moon.doctrine_action.mandate
        conviction = current_moon.doctrine_action.conviction

    mandate_word = mandate_text.split("/")[0] if "/" in mandate_text else mandate_text
    mandate_color = MANDATE_COLORS.get(mandate_word, "#95a5a6")

    st.markdown(
        f"<div style='background:linear-gradient(135deg,#1a1a2e,#16213e);padding:24px;border-radius:14px;"
        f"border-left:6px solid {mandate_color};margin-bottom:16px'>"
        f"<h2 style='margin:0;color:#e0e0e0'>"
        f"\U0001f319 Moon {mn}: {current_moon.lunar_phase_name}</h2>"
        f"<p style='color:#aaa;margin:6px 0'>"
        f"{current_moon.start_date.strftime('%b %d')} \u2014 {current_moon.end_date.strftime('%b %d, %Y')}"
        f" &nbsp;|&nbsp; Day {days_in} of {total_days}"
        f" &nbsp;|&nbsp; {days_left} days remaining</p>"
        f"<p style='color:{mandate_color};font-size:1.4em;font-weight:bold;margin:10px 0'>"
        f"\u2694\ufe0f {mandate_text} &nbsp; (conviction: {conviction:.0%})</p>"
        f"<p style='color:#ccc;font-size:1.1em'>{briefing.get('theme', '')}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.progress(pct_through, text=f"Moon {mn} progress: {pct_through:.0%}")

    # Key dates row
    col_fire, col_new, col_geo, col_freq = st.columns(4)
    with col_fire:
        fp = current_moon.fire_peak_date
        if fp:
            fp_delta = (fp - today).days
            fp_label = f"{fp.strftime('%b %d')}" + (
                f" ({fp_delta}d)" if fp_delta > 0 else " (PAST)" if fp_delta < 0 else " **TODAY**"
            )
        else:
            fp_label = "N/A"
        st.metric("\U0001f525 Fire Peak", fp_label)
    with col_new:
        nm = current_moon.new_moon_date
        if nm:
            nm_delta = (nm - today).days
            nm_label = f"{nm.strftime('%b %d')}" + (
                f" ({nm_delta}d)" if nm_delta > 0 else " (PAST)" if nm_delta < 0 else " **TODAY**"
            )
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
    with st.expander(f"\U0001f4dc Moon {mn} Briefing & Sacred Geometry", expanded=True):
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
            if geometry.get("platonic_solid"):
                st.write(f"Platonic Solid: {geometry['platonic_solid']}")
            if geometry.get("phi_link"):
                st.write(f"Phi Link: {geometry['phi_link']}")
            if geometry.get("correlation"):
                st.markdown("**Portfolio Correlation**")
                st.write(geometry["correlation"])
            if current_moon.doctrine_action:
                st.markdown("**Doctrine Mandate**")
                st.write(current_moon.doctrine_action.description)
                if current_moon.doctrine_action.targets:
                    st.write(f"Targets: {', '.join(current_moon.doctrine_action.targets)}")
else:
    st.warning("No active moon cycle found for today.")


# ============================================================================
# SECTION 2: UPCOMING ALERTS
# ============================================================================
st.header(f"\u26a0\ufe0f Upcoming Alerts -- Next {alert_days} Days")

alerts = doctrine.get_events_with_lead_time(days_ahead=alert_days, target=today)
if alerts:
    alert_rows = []
    pri_icons = {"CRITICAL": "\U0001f534", "HIGH": "\U0001f7e0", "MEDIUM": "\U0001f7e1", "LOW": "\u26aa"}
    for a in sorted(alerts, key=lambda x: x.days_until):
        alert_rows.append({
            "": pri_icons.get(a.priority, "\u26aa"),
            "Date": a.event_date.strftime("%b %d"),
            "In": f"{a.days_until}d",
            "Moon": a.moon_number,
            "Type": a.event_type.title(),
            "Event": a.event_name,
            "Action": a.lead_time_action,
            "Priority": a.priority,
        })
    st.dataframe(alert_rows, use_container_width=True, hide_index=True)
else:
    st.success(f"No alerts in the next {alert_days} days.")


# ============================================================================
# SECTION 3: EVENTS IN CURRENT MOON
# ============================================================================
if current_moon:
    evt_count = (
        len(current_moon.astrology_events) + len(current_moon.phi_markers)
        + len(current_moon.financial_events) + len(current_moon.world_events)
        + len(current_moon.aac_events)
    )
    st.header(f"\U0001f4c5 Events This Moon Cycle ({evt_count} total)")

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


# ============================================================================
# SECTION 3B: SPACE WEATHER (NOAA SWPC — free, no key)
# ============================================================================

@st.cache_data(ttl=300)  # cache 5 minutes
def _fetch_space_weather() -> dict:
    """Pull current space weather from NOAA SWPC public JSON endpoints."""
    result: dict = {}
    timeout = 6

    # ── Planetary K-index (Kp) — geomagnetic activity ──
    try:
        req = urllib.request.Request(
            "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
            headers={"User-Agent": "AAC-13Moon/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rows = json.loads(resp.read().decode())
        if rows:
            latest = rows[-1]
            result["kp_time"] = latest.get("time_tag", "")
            result["kp_value"] = float(latest.get("Kp", 0))
    except Exception:
        pass

    # ── Solar wind speed (DSCOVR real-time) ──
    try:
        req = urllib.request.Request(
            "https://services.swpc.noaa.gov/products/summary/solar-wind-speed.json",
            headers={"User-Agent": "AAC-13Moon/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        if isinstance(data, list) and data:
            result["wind_speed"] = data[0].get("proton_speed")
        elif isinstance(data, dict):
            result["wind_speed"] = data.get("WindSpeed") or data.get("proton_speed")
    except Exception:
        pass

    # ── Solar flux (10.7cm) ──
    try:
        req = urllib.request.Request(
            "https://services.swpc.noaa.gov/products/summary/10cm-flux.json",
            headers={"User-Agent": "AAC-13Moon/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        if isinstance(data, list) and data:
            result["solar_flux"] = data[0].get("flux")
            result["flux_time"] = data[0].get("time_tag")
        elif isinstance(data, dict):
            result["solar_flux"] = data.get("Flux") or data.get("flux")
            result["flux_time"] = data.get("TimeStamp") or data.get("time_tag")
    except Exception:
        pass

    # ── Geomagnetic storm / alerts ──
    try:
        req = urllib.request.Request(
            "https://services.swpc.noaa.gov/products/noaa-scales.json",
            headers={"User-Agent": "AAC-13Moon/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        # data["0"] = current, data["-1"] = previous
        current = data.get("0", {})
        result["geo_storm"] = current.get("G", {})   # Geomagnetic storm scale
        result["solar_rad"] = current.get("S", {})    # Solar radiation scale
        result["radio_blackout"] = current.get("R", {})  # Radio blackout scale
    except Exception:
        pass

    # ── Sunspot number ──
    try:
        req = urllib.request.Request(
            "https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json",
            headers={"User-Agent": "AAC-13Moon/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rows = json.loads(resp.read().decode())
        # Find the entry closest to today
        today_str = date.today().isoformat()[:7]  # "YYYY-MM"
        for row in rows:
            ts = row.get("time-tag", "")
            if ts.startswith(today_str):
                result["ssn_predicted"] = row.get("predicted_ssn")
                result["ssn_high"] = row.get("high_ssn")
                result["ssn_low"] = row.get("low_ssn")
                break
    except Exception:
        pass

    return result


st.header("\u2600\ufe0f Space Weather")
st.caption(
    "Live solar & geomagnetic data from NOAA Space Weather Prediction Center. "
    "Solar cycle 25 peak (2024-2026) amplifies geomagnetic volatility -- correlated with market sentiment shifts."
)

sw = _fetch_space_weather()
if sw:
    col_kp, col_wind, col_flux, col_ssn = st.columns(4)

    # Kp index
    with col_kp:
        kp = sw.get("kp_value")
        if kp is not None:
            kp_label = (
                "Quiet" if kp < 4 else
                "Active" if kp < 5 else
                "Minor Storm" if kp < 6 else
                "Moderate Storm" if kp < 7 else
                "Strong Storm" if kp < 8 else
                "Severe Storm"
            )
            st.metric("\U0001f9f2 Kp Index", f"{kp:.1f}", delta=kp_label)
        else:
            st.metric("\U0001f9f2 Kp Index", "N/A")

    # Solar wind
    with col_wind:
        wind = sw.get("wind_speed")
        if wind:
            st.metric("\U0001f4a8 Solar Wind", f"{wind} km/s")
        else:
            st.metric("\U0001f4a8 Solar Wind", "N/A")

    # Solar flux (10.7cm)
    with col_flux:
        flux = sw.get("solar_flux")
        if flux:
            st.metric("\u2600\ufe0f Solar Flux", f"{flux} sfu")
        else:
            st.metric("\u2600\ufe0f Solar Flux", "N/A")

    # Sunspot number
    with col_ssn:
        ssn = sw.get("ssn_predicted")
        if ssn is not None:
            st.metric("\u2b50 Sunspot #", f"{ssn:.0f}")
        else:
            st.metric("\u2b50 Sunspot #", "N/A")

    # NOAA Scales row
    geo = sw.get("geo_storm", {})
    rad = sw.get("solar_rad", {})
    rbo = sw.get("radio_blackout", {})

    if geo or rad or rbo:
        col_g, col_s, col_r = st.columns(3)
        scale_colors = {"0": "\U0001f7e2", "1": "\U0001f7e1", "2": "\U0001f7e0", "3": "\U0001f534", "4": "\U0001f534", "5": "\u26d4"}
        with col_g:
            g_scale = str(geo.get("Scale", "0") or "0")
            g_icon = scale_colors.get(g_scale, "\u26aa")
            st.markdown(f"**{g_icon} Geomagnetic Storm:** G{g_scale}")
            if geo.get("Text"):
                st.caption(geo["Text"])
        with col_s:
            s_scale = str(rad.get("Scale", "0") or "0")
            s_icon = scale_colors.get(s_scale, "\u26aa")
            st.markdown(f"**{s_icon} Solar Radiation:** S{s_scale}")
            if rad.get("Text"):
                st.caption(rad["Text"])
        with col_r:
            r_scale = str(rbo.get("Scale", "0") or "0")
            r_icon = scale_colors.get(r_scale, "\u26aa")
            st.markdown(f"**{r_icon} Radio Blackout:** R{r_scale}")
            if rbo.get("Text"):
                st.caption(rbo["Text"])

    # Kp interpretation for doctrine
    kp = sw.get("kp_value")
    if kp is not None and kp >= 5:
        st.warning(
            f"\u26a0\ufe0f **Geomagnetic Storm Active (Kp={kp:.0f})** -- "
            f"Elevated geomagnetic activity historically correlates with increased market volatility "
            f"and sentiment shifts. Solar cycle 25 peak amplifies fire peak resonance."
        )
    elif kp is not None and kp >= 4:
        st.info(
            f"\U0001f9f2 **Geomagnetic Activity Elevated (Kp={kp:.0f})** -- "
            f"Approaching storm threshold. Monitor for CME impacts."
        )
else:
    st.info("Space weather data unavailable -- NOAA SWPC endpoints may be down.")


# ============================================================================
# SECTION 4: PHI COHERENCE WAVE
# ============================================================================
st.header("\u03c6 Phi Coherence Wave")
st.caption(
    "Dan Winter PlanckPhire principle: successive phi powers applied to the synodic "
    "lunar interval create fractal resonance nodes. High resonance = higher volatility amplification."
)

phi_rows = []
for cycle in doctrine.moon_cycles:
    for p in cycle.phi_markers:
        delta = (p.date - today).days
        status = "\U0001f7e2" if delta > 0 else ("\u26a1" if delta == 0 else "\u26aa")
        phi_rows.append({
            "": status,
            "Date": p.date.strftime("%b %d, %Y"),
            "Days": delta,
            "Moon": cycle.moon_number,
            "Power": f"\u03c6^{p.phi_power}",
            "Value": f"{p.phi_value:.4f}",
            "Days From Anchor": p.days_from_anchor,
            "Resonance": f"{p.resonance_strength:.0%}",
            "Label": p.label,
        })
phi_rows.sort(key=lambda r: r["Days"])
if phi_rows:
    st.dataframe(phi_rows, use_container_width=True, hide_index=True)


# ============================================================================
# SECTION 5: FULL 14-MOON TIMELINE
# ============================================================================
st.header("\U0001f30d Full 14-Moon Timeline")

timeline_rows = []
for cycle in doctrine.moon_cycles:
    mn = cycle.moon_number
    b = MOON_BRIEFINGS.get(mn, {})
    g = SACRED_GEOMETRY_OVERLAY.get(mn, {})
    is_current = current_moon and mn == current_moon.moon_number
    mandate_str = cycle.doctrine_action.mandate if cycle.doctrine_action else ""
    conv = f"{cycle.doctrine_action.conviction:.0%}" if cycle.doctrine_action else ""
    fp_str = cycle.fire_peak_date.strftime("%b %d") if cycle.fire_peak_date else ""
    evt_total = (
        len(cycle.astrology_events) + len(cycle.phi_markers)
        + len(cycle.financial_events) + len(cycle.world_events)
        + len(cycle.aac_events)
    )
    timeline_rows.append({
        "Moon": f"{'>>> ' if is_current else ''}{mn}",
        "Name": cycle.lunar_phase_name,
        "Dates": f"{cycle.start_date.strftime('%b %d')} - {cycle.end_date.strftime('%b %d')}",
        "Fire Peak": fp_str,
        "Mandate": mandate_str,
        "Conv.": conv,
        "Geometry": g.get("geometry", ""),
        "Hz": g.get("frequency_hz", ""),
        "Events": evt_total,
        "Theme": b.get("theme", ""),
    })
st.dataframe(timeline_rows, use_container_width=True, hide_index=True)


# ============================================================================
# SECTION 6: SELECTED MOON DETAIL
# ============================================================================
selected_cycle = None
for c in doctrine.moon_cycles:
    if c.moon_number == selected_moon_num:
        selected_cycle = c
        break

if selected_cycle and (not current_moon or selected_cycle.moon_number != current_moon.moon_number):
    st.header(f"\U0001f50d Moon {selected_cycle.moon_number}: {selected_cycle.lunar_phase_name}")

    sel_brief = MOON_BRIEFINGS.get(selected_cycle.moon_number, {})
    sel_geo = SACRED_GEOMETRY_OVERLAY.get(selected_cycle.moon_number, {})

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Dates:** {selected_cycle.start_date.strftime('%b %d')} -- "
                    f"{selected_cycle.end_date.strftime('%b %d, %Y')}")
        if selected_cycle.fire_peak_date:
            st.markdown(f"**Fire Peak:** {selected_cycle.fire_peak_date.strftime('%b %d')}")
        if selected_cycle.new_moon_date:
            st.markdown(f"**New Moon:** {selected_cycle.new_moon_date.strftime('%b %d')}")
        if selected_cycle.doctrine_action:
            st.markdown(f"**Mandate:** {selected_cycle.doctrine_action.mandate} "
                        f"(conviction: {selected_cycle.doctrine_action.conviction:.0%})")
            st.write(selected_cycle.doctrine_action.description)
    with col_b:
        st.markdown(f"**Theme:** {sel_brief.get('theme', 'N/A')}")
        st.markdown(f"**Geometry:** {sel_geo.get('geometry', 'N/A')} -- "
                    f"{sel_geo.get('description', '')}")
        if sel_brief.get("market_implication"):
            st.markdown(f"**Market Implication:** {sel_brief['market_implication']}")

    # Events in selected moon
    sel_evts = []
    for e in selected_cycle.astrology_events:
        sel_evts.append({"Date": e.date.strftime("%b %d"), "Type": "\u2b50 Astro",
                         "Event": e.name, "Impact": e.impact})
    for e in selected_cycle.phi_markers:
        sel_evts.append({"Date": e.date.strftime("%b %d"), "Type": "\u03c6 Phi",
                         "Event": e.label, "Impact": f"{e.resonance_strength:.0%}"})
    for e in selected_cycle.financial_events:
        sel_evts.append({"Date": e.date.strftime("%b %d"), "Type": "\U0001f4b0 Finance",
                         "Event": e.name, "Impact": e.impact})
    for e in selected_cycle.world_events:
        sel_evts.append({"Date": e.date.strftime("%b %d"), "Type": "\U0001f30d World",
                         "Event": e.name, "Impact": e.impact})
    for e in selected_cycle.aac_events:
        sel_evts.append({"Date": e.date.strftime("%b %d"), "Type": "\U0001f916 AAC",
                         "Event": e.name, "Impact": e.impact})
    if sel_evts:
        st.dataframe(sel_evts, use_container_width=True, hide_index=True)


# ============================================================================
# SECTION 7: DEEP DIVES (toggled from sidebar)
# ============================================================================
if show_deep_dives:
    st.header("\U0001f9ed Deep Dives")

    # ── Saturn-Neptune Conjunction ──
    with st.expander("\U0001fa90 Saturn-Neptune Conjunction (Feb 2027)", expanded=True):
        sn = SATURN_NEPTUNE_DEEPDIVE
        st.markdown(f"**Date:** {sn['date']} &nbsp; **Degree:** {sn['degree']} &nbsp; "
                    f"**Cycle:** {sn['cycle_years']} years")
        st.info(sn["significance"])
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Saturn themes:** {sn['saturn_themes']}")
            st.markdown(f"**Neptune themes:** {sn['neptune_themes']}")
            st.markdown(f"**Blend:** {sn['blend']}")
        with col2:
            st.markdown("**Historical Parallels**")
            for h in sn["historical_parallels"]:
                st.markdown(f"- **{h['year']}** ({h['sign']}): {h['events'][:120]}...")

        st.markdown("**2027 Predictions**")
        for k, v in sn["2027_predictions"].items():
            st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")

        st.markdown(f"**Doctrine Action:** {sn['doctrine_action']}")

    # ── Age of Aquarius ──
    with st.expander("\u2652 Age of Aquarius Transition", expanded=False):
        aq = AGE_OF_AQUARIUS
        st.markdown(f"**Phenomenon:** {aq['phenomenon']}")
        st.markdown(f"**Great Year:** {aq['great_year']}")
        st.markdown(f"**Current Transition:** {aq['current_transition']}")
        for k, v in aq.items():
            if k not in ("phenomenon", "great_year", "current_transition") and isinstance(v, str):
                st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")

    # ── Dalio Big Cycle ──
    with st.expander("\U0001f4ca Dalio Big Cycle", expanded=False):
        dc = DALIO_BIG_CYCLE
        st.markdown(f"**Framework:** {dc['framework']}")
        st.markdown("**Five Forces:**")
        for f in dc["five_forces"]:
            st.markdown(f"- {f}")
        st.markdown("**Stages:**")
        for stage, desc in dc["stages"].items():
            st.markdown(f"- **{stage}:** {desc}")
        st.markdown(f"**Current Position (2026):** {dc['current_position_2026']}")

        if "bridgewater_portfolio_2026" in dc:
            bw = dc["bridgewater_portfolio_2026"]
            st.markdown(f"**Bridgewater:** AUM {bw['aum']}, {bw['holdings']} holdings, "
                        f"Turnover {bw['turnover']}")
            st.caption(bw["note"])

        st.markdown("**Doctrine Alignment:**")
        for k, v in dc["doctrine_alignment"].items():
            st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")

    # ── LEAPS Playbook ──
    with st.expander("\U0001f4b5 LEAPS Playbook", expanded=False):
        lp = LEAPS_PLAYBOOK
        st.markdown(f"**Total Capital:** ${lp['total_capital']:,}")
        st.markdown(f"**Strategy:** {lp['strategy']}")
        st.markdown(f"**Entry Window:** {lp['entry_window']}")
        st.markdown(f"**Existing Book:** {lp.get('existing_book', 'N/A')}")
        st.markdown("**Positions:**")
        for name, pos in lp.get("positions", {}).items():
            st.markdown(f"- **{name}** ({pos.get('ticker', '')}): "
                        f"{pos.get('allocation_pct', 0)}% = ${pos.get('amount', 0):,}, "
                        f"Strike: {pos.get('strike', 'N/A')}, "
                        f"Contracts: {pos.get('contracts', 'N/A')}")
            st.caption(pos.get("thesis", ""))

    # ── Crypto Doctrine ──
    with st.expander("\U0001fa99 Crypto Doctrine", expanded=False):
        cd = CRYPTO_DOCTRINE
        st.markdown(f"**Thesis:** {cd['thesis']}")
        st.markdown(f"**Current Regime:** {cd['regime_current']}")

        if "positions" in cd:
            st.markdown("**Positions:**")
            for name, pos in cd["positions"].items():
                status = pos.get("status", "UNKNOWN")
                st.markdown(f"- **{name}**: {status}")
                if pos.get("reason"):
                    st.caption(pos["reason"])

    # ── War Room Doctrine ──
    if isinstance(WAR_ROOM_DOCTRINE, dict):
        with st.expander("\u2694\ufe0f War Room Doctrine", expanded=False):
            for k, v in WAR_ROOM_DOCTRINE.items():
                if isinstance(v, str):
                    st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
                elif isinstance(v, dict):
                    st.markdown(f"**{k.replace('_', ' ').title()}:**")
                    for sk, sv in v.items():
                        st.markdown(f"- {sk}: {sv}")


# ============================================================================
# SECTION 8: ALL EVENTS CHRONOLOGICAL
# ============================================================================
with st.expander("\U0001f4cb All Events Chronological (all layers)", expanded=False):
    all_events = doctrine.get_all_events_sorted()
    if all_events:
        all_rows = []
        for evt in all_events:
            evt_date = evt.get("date", "")
            delta = (date.fromisoformat(evt_date) - today).days if evt_date else 0
            type_icons = {
                "astrology": "\u2b50", "phi": "\u03c6", "financial": "\U0001f4b0",
                "world": "\U0001f30d", "aac": "\U0001f916",
            }
            icon = type_icons.get(evt.get("type", ""), "\u2022")
            all_rows.append({
                "Date": evt_date,
                "Days": delta,
                "Type": f"{icon} {evt.get('type', '').title()}",
                "Event": evt.get("name", ""),
                "Impact": evt.get("impact", ""),
                "Category": evt.get("category", ""),
            })
        st.dataframe(all_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No events in doctrine.")


# ============================================================================
# SECTION: POLYMARKET DIVISION
# ============================================================================
if POLYMARKET_AVAILABLE:
    st.header("🎯 Polymarket Division")

    # Division status
    try:
        div_status = get_division_status()
        loaded = sum(1 for v in div_status.values() if v.get("status") == "loaded")
        total = len(div_status)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategies Loaded", f"{loaded}/{total}")
        col2.metric("Division Status", "ACTIVE" if loaded > 0 else "OFFLINE")

        # PolyMC Agent — portfolio & MC results
        try:
            agent = PolyMCAgent()
            portfolio = agent.TARGET_PORTFOLIO
            col3.metric("Target Bets", len(portfolio))

            mc_result = agent.run_portfolio_monte_carlo()
            col4.metric("Portfolio EV", f"{mc_result.get('ev_pct', 0):.1f}%")

            st.subheader("📊 PolyMC Portfolio")
            port_rows = []
            for bet in portfolio:
                port_rows.append({
                    "Market": bet.market_name,
                    "Entry": f"${bet.entry_price:.2f}",
                    "Our Prob": f"{bet.our_probability:.0%}",
                    "Size": f"${bet.size_usd:.0f}",
                    "Side": bet.side,
                })
            if port_rows:
                st.dataframe(port_rows, use_container_width=True, hide_index=True)

            # MC summary
            mc_cols = st.columns(5)
            mc_cols[0].metric("Mean Return", f"${mc_result.get('mean_return', 0):.2f}")
            mc_cols[1].metric("P(Profit)", f"{mc_result.get('prob_profit', 0):.0%}")
            mc_cols[2].metric("Sharpe", f"{mc_result.get('sharpe', 0):.2f}")
            mc_cols[3].metric("VaR 95%", f"${mc_result.get('var_95', 0):.2f}")
            mc_cols[4].metric("Max Payout", f"${mc_result.get('max_payout', 0):.2f}")
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
                        "Threshold": f"${sig.threshold:.2f}",
                        "Action": sig.action,
                        "Urgency": sig.urgency,
                    })
                st.dataframe(sig_rows, use_container_width=True, hide_index=True)
            else:
                st.success("No exit signals — all positions stable.")
        except Exception:
            st.info("PolyMC Monitor data unavailable.")

        # Active Scanner status
        try:
            scanner = ActiveScanner(dry_run=True)
            sc_cols = st.columns(4)
            mode = "DRY RUN" if scanner.dry_run else "LIVE"
            sc_cols[0].metric("Scanner Mode", mode)
            sc_cols[1].metric("Bets Today", f"{scanner.daily_bet_count}/{scanner.MAX_DAILY_BETS}")
            sc_cols[2].metric("Max Position", f"${scanner.MAX_POSITION_SIZE_USD}")
            sc_cols[3].metric("Min Edge", f"{scanner.MIN_EDGE_THRESHOLD:.0%}")
        except Exception:
            st.info("Active Scanner data unavailable.")

        # Per-strategy status
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
    st.header("🎯 Polymarket Division")
    st.info("Polymarket Division not installed — install strategies/polymarket_division to enable.")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#666;font-size:0.8em'>"
    f"13-Moon Doctrine Dashboard v1.0 -- "
    f"{len(doctrine.moon_cycles)} moon cycles -- "
    f"{sum(len(c.astrology_events) + len(c.phi_markers) + len(c.financial_events) + len(c.world_events) + len(c.aac_events) for c in doctrine.moon_cycles)} total events -- "
    f"Auto-refresh: {'ON' if auto_refresh else 'OFF'} -- "
    f"Last update: {today.isoformat()}"
    f"</div>",
    unsafe_allow_html=True,
)
