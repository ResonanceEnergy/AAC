#!/usr/bin/env python3
"""
AAC Matrix Monitor - Streamlit Dashboard
=========================================
Lightweight Streamlit frontend that pulls live data from
AACMasterMonitoringDashboard.collect_monitoring_data().

Launch:
    streamlit run src/aac/monitoring/dashboard_monitor.py
"""

import asyncio
import sys
import os

import streamlit as st
from datetime import datetime

# Ensure project root is importable
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

st.set_page_config(
    page_title="AAC Matrix Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 AAC Matrix Monitor")
st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


@st.cache_resource
def _get_dashboard():
    """Instantiate the master dashboard once per Streamlit session."""
    try:
        from monitoring.aac_master_monitoring_dashboard import AACMasterMonitoringDashboard
        return AACMasterMonitoringDashboard()
    except Exception as exc:
        st.error(f"Failed to load master dashboard: {exc}")
        return None


def _run_async(coro):
    """Run an async coroutine from sync Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result(timeout=30)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


dashboard = _get_dashboard()

if dashboard is None:
    st.stop()

# Collect live data
try:
    data = _run_async(dashboard.collect_monitoring_data())
except Exception as exc:
    st.error(f"Data collection error: {exc}")
    data = {}

# ── Health Overview ──
st.markdown("## System Health")
health = data.get('health', {})
col1, col2, col3, col4 = st.columns(4)

with col1:
    cpu = health.get('system', {}).get('cpu_percent', '—')
    st.metric("CPU", f"{cpu}%")
with col2:
    mem = health.get('system', {}).get('memory_percent', '—')
    st.metric("Memory", f"{mem}%")
with col3:
    disk = health.get('system', {}).get('disk_percent', '—')
    st.metric("Disk", f"{disk}%")
with col4:
    net_status = health.get('network', {}).get('status', '—')
    st.metric("Network", net_status)

# ── PnL ──
st.markdown("## P&L")
pnl = data.get('pnl', {})
pcol1, pcol2, pcol3 = st.columns(3)
with pcol1:
    st.metric("Total P&L", f"${pnl.get('total_pnl', 0):,.2f}")
with pcol2:
    st.metric("Win Rate", f"{pnl.get('win_rate', 0):.1%}")
with pcol3:
    st.metric("Active Positions", pnl.get('active_positions', 0))

# ── Risk ──
st.markdown("## Risk")
risk = data.get('risk', {})
rcol1, rcol2 = st.columns(2)
with rcol1:
    st.metric("Portfolio Risk Score", risk.get('portfolio_risk_score', '—'))
with rcol2:
    breakers = risk.get('circuit_breakers_triggered', 0)
    st.metric("Circuit Breakers", breakers, delta_color="inverse")

# ── Alerts ──
alerts = data.get('alerts', [])
if alerts:
    st.markdown("## Active Alerts")
    for alert in alerts[:10]:
        severity = alert.get('severity', 'info')
        msg = alert.get('message', str(alert))
        if severity == 'critical':
            st.error(msg)
        elif severity == 'warning':
            st.warning(msg)
        else:
            st.info(msg)

# ── Auto-refresh ──
st.markdown("---")
if st.button("Refresh"):
    st.rerun()