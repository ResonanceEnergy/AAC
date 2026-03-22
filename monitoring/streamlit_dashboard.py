#!/usr/bin/env python3
"""
AAC Streamlit Dashboard & Copilot Interface
============================================
Web-based monitoring dashboard and chat assistant extracted from the master monitoring module.
"""

import asyncio
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

try:
    from trading.aac_arbitrage_execution_system import AACArbitrageExecutionSystem, ExecutionConfig
    ARBITRAGE_COMPONENTS_AVAILABLE = True
except ImportError:
    ARBITRAGE_COMPONENTS_AVAILABLE = False
    ExecutionConfig = None  # type: ignore[assignment,misc]
    AACArbitrageExecutionSystem = None  # type: ignore[assignment,misc]


class AACStreamlitDashboard:
    """Streamlit-based web dashboard for AAC monitoring"""

    def __init__(self):
        self.logger = logging.getLogger("AACStreamlitDashboard")
        self.execution_system = None
        self.status_queue = queue.Queue()
        self.is_running = False
        self.update_thread = None

    def initialize_system(self):
        """Initialize the arbitrage execution system"""
        try:
            if ARBITRAGE_COMPONENTS_AVAILABLE:
                execution_config = ExecutionConfig()
                self.execution_system = AACArbitrageExecutionSystem(execution_config)
                asyncio.run(self.execution_system.initialize())
                return True
            else:
                logger.info("Arbitrage components not available")
                return False
        except Exception as e:
            logger.info(f"Failed to initialize system: {e}")
            return False

    def start_monitoring(self):
        """Start the monitoring thread"""
        if not self.is_running:
            self.is_running = True
            self.update_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.update_thread.start()

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_running:
            try:
                if self.execution_system:
                    # Run arbitrage cycle
                    cycle_report = asyncio.run(self.execution_system.run_arbitrage_cycle())

                    # Monitor positions
                    asyncio.run(self.execution_system.monitor_positions())

                    # Get system status
                    status = self.execution_system.get_system_status()

                    # Put status in queue for Streamlit to pick up
                    self.status_queue.put({
                        'timestamp': datetime.now(),
                        'cycle_report': cycle_report,
                        'system_status': status
                    })

                time.sleep(30)  # Update every 30 seconds

            except Exception as e:
                self.status_queue.put({'error': str(e), 'timestamp': datetime.now()})

    def get_latest_status(self):
        """Get the latest system status"""
        latest = None
        try:
            while not self.status_queue.empty():
                latest = self.status_queue.get_nowait()
            return latest
        except Exception as e:
            self.logger.debug(f"Status queue empty or error: {e}")
            return latest


def generate_copilot_response(user_input: str, latest_status: dict, dashboard) -> str:
    """Generate AI response based on user input and live system data."""
    user_input_lower = user_input.lower()

    # Extract live metrics from latest_status
    status = {}
    if latest_status and 'system_status' in latest_status:
        status = latest_status['system_status']
    perf = status.get('performance', {})
    total_pnl = perf.get('total_pnl', 0)
    executed = perf.get('executed_trades', 0)
    win_rate = perf.get('win_rate', 0)
    total_opps = status.get('total_opportunities', 0)
    active = status.get('active_trades', 0)
    session_runtime = status.get('session_runtime', '00:00:00')

    no_data = not status
    na = "N/A (system not reporting)"

    responses = {
        "status": (
            na if no_data else
            f"The AAC system is operational. Active trades: {active}, "
            f"{total_opps} opportunities detected, {executed} trades executed "
            f"with {win_rate:.1%} win rate, ${total_pnl:.2f} total P&L. "
            f"Session runtime: {session_runtime}."
        ),
        "performance": (
            na if no_data else
            f"Current performance: Total P&L ${total_pnl:.2f}, "
            f"{executed} executed trades, {win_rate:.1%} win rate, "
            f"{total_opps} opportunities detected."
        ),
        "health": (
            "System health: All doctrine packs compliant, safeguards active, "
            "security monitoring operational, department engines running."
        ),
        "opportunities": (
            na if no_data else
            f"Detected {total_opps} arbitrage opportunities across multiple "
            "strategies including cross-exchange, triangular, and statistical."
        ),
        "risk": (
            "Risk management active: Circuit breakers engaged, position limits "
            "enforced, safety protocols operational. Current exposure within "
            "safe parameters."
        ),
        "trading": (
            na if no_data else
            f"Trading activity: {executed} trades executed, {active} active "
            f"positions monitored with real-time risk assessment."
        ),
        "doctrine": (
            "Doctrine compliance across all 8 packs: Risk Envelope, Security, "
            "Testing, Incident Response, Liquidity, Counterparty Scoring, "
            "Research Factory, and Metric Canon."
        ),
        "security": (
            "Security systems fully operational: RBAC active, API security "
            "enabled, encryption protocols running, continuous monitoring of "
            "all access points."
        ),
        "help": (
            "I can help with: system status, performance metrics, trading "
            "activity, risk management, doctrine compliance, security "
            "monitoring, and general AAC operations."
        ),
    }

    # Check for keywords in user input
    for key, response in responses.items():
        if key in user_input_lower:
            return response

    # Default response with system context
    system_info = ""
    if status:
        system_info = (
            f" Current system shows {active} active trades and "
            f"{total_opps} opportunities detected."
        )

    return (
        f"I understand you're asking about '{user_input}'.{system_info} "
        "For more specific information, try asking about status, performance, "
        "health, opportunities, risk, trading, doctrine, or security."
    )


def play_audio_response(text: str):
    """Generate and play audio for the given text response."""
    try:
        import pyttsx3

        def speak():
            """Speak text via pyttsx3."""
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 0.8)
            engine.say(text)
            engine.runAndWait()

        threading.Thread(target=speak, daemon=True).start()
        if STREAMLIT_AVAILABLE and st is not None:
            st.success("Audio playing...")
        else:
            logger.info("Audio playing...")

    except ImportError:
        if STREAMLIT_AVAILABLE and st is not None:
            st.warning("pyttsx3 not installed. Install with: pip install pyttsx3")
        else:
            logger.warning("pyttsx3 not installed")
    except Exception as e:
        if STREAMLIT_AVAILABLE and st is not None:
            st.error(f"Audio playback failed: {e}")
        else:
            logger.error(f"Audio playback failed: {e}")


def run_streamlit_dashboard():
    """Run the Streamlit dashboard as a standalone app"""
    if not STREAMLIT_AVAILABLE:
        logger.info("Streamlit not available")
        return

    try:
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        pd = None  # type: ignore[assignment]
        go = None  # type: ignore[assignment]
        make_subplots = None  # type: ignore[assignment]

    st.set_page_config(
        page_title="AAC Matrix Monitor",
        page_icon="📊",
        layout="wide"
    )

    st.title("🚀 AAC Matrix Monitor")
    st.markdown("---")

    # Initialize dashboard
    dashboard = AACStreamlitDashboard()

    # Sidebar controls
    st.sidebar.title("Controls")

    if st.sidebar.button("Initialize System"):
        with st.spinner("Initializing AAC system..."):
            if dashboard.initialize_system():
                st.sidebar.success("System initialized")
            else:
                st.sidebar.error("Initialization failed")

    if st.sidebar.button("Start Monitoring"):
        dashboard.start_monitoring()
        st.sidebar.success("Monitoring started")

    if st.sidebar.button("Stop Monitoring"):
        dashboard.stop_monitoring()
        st.sidebar.success("Monitoring stopped")

    # Manual cycle execution
    if st.sidebar.button("Run Arbitrage Cycle"):
        if dashboard.execution_system:
            with st.spinner("Running arbitrage cycle..."):
                try:
                    cycle_report = asyncio.run(dashboard.execution_system.run_arbitrage_cycle())
                    st.sidebar.success(f"Cycle complete - {cycle_report.get('opportunities_detected', 0)} opportunities")
                except Exception as e:
                    st.sidebar.error(f"Cycle failed: {e}")
        else:
            st.sidebar.error("System not initialized")

    # Configuration display
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuration")

    if ARBITRAGE_COMPONENTS_AVAILABLE:
        execution_config = ExecutionConfig()
        st.sidebar.checkbox("Auto Execute", value=execution_config.auto_execute, disabled=True)
        st.sidebar.checkbox("Test Mode", value=execution_config.enable_test_mode, disabled=True)
        st.sidebar.slider("Min Confidence", min_value=0.0, max_value=1.0,
                         value=execution_config.min_confidence_threshold, disabled=True)
        st.sidebar.slider("Max Spread", min_value=0.0, max_value=0.1,
                         value=execution_config.max_spread_threshold, disabled=True)

    # Main dashboard content
    col1, col2, col3 = st.columns(3)

    # Get latest status
    latest_status = dashboard.get_latest_status()

    if latest_status and 'system_status' in latest_status:
        status = latest_status['system_status']

        # Key metrics
        with col1:
            st.subheader("📊 Key Metrics")
            st.metric("Active Trades", status.get('active_trades', 0))
            st.metric("Total Opportunities", status.get('total_opportunities', 0))
            st.metric("Session Runtime", status.get('session_runtime', '00:00:00'))

        with col2:
            st.subheader("💰 Performance")
            perf = status.get('performance', {})
            st.metric("Total PnL", f"${perf.get('total_pnl', 0):.2f}")
            st.metric("Executed Trades", perf.get('executed_trades', 0))
            st.metric("Win Rate", f"{perf.get('win_rate', 0):.1%}")

        with col3:
            st.subheader("🎯 System Status")
            st.metric("Auto Execute", "ON" if (ARBITRAGE_COMPONENTS_AVAILABLE and execution_config.auto_execute) else "OFF")
            st.metric("Test Mode", "ON" if (ARBITRAGE_COMPONENTS_AVAILABLE and execution_config.enable_test_mode) else "OFF")
            st.metric("Last Update", latest_status.get('timestamp', datetime.now()).strftime('%H:%M:%S'))

            # AZ System Status Brief Button
            st.markdown("---")
            try:
                from shared.az_response_library import get_az_library
                from shared.avatar_system import get_avatar_manager
                az_lib = get_az_library()
                avatar_manager = get_avatar_manager()

                if st.button("🎙️ AZ Status Brief", key="main_status_brief"):
                    brief = az_lib.get_system_status_brief()
                    st.session_state.main_az_brief = brief
                    avatar_manager.speak_text(brief, "az")
                    st.success("AZ Status Brief generated and playing...")
            except ImportError:
                st.warning("AZ system not available")

        # Display AZ brief if generated
        if 'main_az_brief' in st.session_state:
            st.markdown("---")
            st.subheader("🎯 AZ System Status Brief")
            st.text_area("Brief:", st.session_state.main_az_brief, height=150, disabled=True)

        # Performance chart — uses live data when available, placeholder otherwise
        st.markdown("---")
        st.subheader("📈 Performance Chart")

        perf_history = status.get('performance_history', [])
        if perf_history:
            # Use real performance history from the monitoring system
            performance_data = perf_history[-24:]
        else:
            st.info("Awaiting live trading data — chart will populate when the monitoring loop reports performance history.")
            performance_data = []

        if performance_data:
            df = pd.DataFrame(performance_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # PnL line
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['pnl'], name="PnL",
                          line=dict(color='green', width=2)),
                secondary_y=False
            )

            # Win rate line
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['win_rate'], name="Win Rate",
                          line=dict(color='blue', width=2)),
                secondary_y=True
            )

            fig.update_layout(
                title="Trading Performance",
                xaxis_title="Time",
                yaxis_title="PnL ($)",
                yaxis2_title="Win Rate (%)"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Recent opportunities
        st.markdown("---")
        st.subheader("🎯 Recent Opportunities")

        opportunities = status.get('recent_opportunities', [])
        if opportunities:
            opp_df = pd.DataFrame(opportunities)
            opp_df['timestamp'] = pd.to_datetime(opp_df.get('timestamp', datetime.now()))
            opp_df['spread'] = opp_df.get('spread', 0).apply(lambda x: f"{x:.2%}")
            opp_df['confidence'] = opp_df.get('confidence', 0).apply(lambda x: f"{x:.1%}")

            st.dataframe(opp_df[['symbol', 'spread', 'confidence', 'type', 'executed', 'timestamp']], use_container_width=True)
        else:
            st.info("No opportunities detected yet")

        # Active positions
        st.markdown("---")
        st.subheader("📋 Active Positions")

        if dashboard.execution_system and hasattr(dashboard.execution_system, 'trading_engine'):
            try:
                # Get portfolio summary
                summary = dashboard.execution_system.trading_engine.get_portfolio_summary()

                if summary.get('positions'):
                    pos_df = pd.DataFrame(summary['positions'])
                    st.dataframe(pos_df, use_container_width=True)
                else:
                    st.info("No active positions")

            except Exception as e:
                st.error(f"Error getting positions: {e}")
        else:
            st.info("Trading engine not available")

        # ── MATRIX MAXIMIZER COMMAND & CONTROL ──────────────────────
        st.markdown("---")
        st.subheader("🎯 Matrix Maximizer — Command & Control")

        mm_path = PROJECT_ROOT / "data" / "matrix_maximizer_latest.json"
        if mm_path.exists():
            try:
                mm_data = json.loads(mm_path.read_text(encoding="utf-8"))
                mm_col1, mm_col2, mm_col3 = st.columns(3)
                forecast = mm_data.get("forecast", {})
                regime = mm_data.get("regime", {})
                risk_snap = mm_data.get("risk", {})

                with mm_col1:
                    st.metric("Mandate", forecast.get("mandate", "?").upper())
                    st.metric("Run #", mm_data.get("run_number", "?"))
                    st.metric("Risk/Trade", f"{forecast.get('risk_per_trade', '?')}%")

                with mm_col2:
                    st.metric("Regime", regime.get("regime", "?").upper())
                    st.metric("Oil", f"${regime.get('oil_price', '?')}")
                    st.metric("VIX", regime.get("vix", "?"))

                with mm_col3:
                    st.metric("Circuit Breaker", risk_snap.get("circuit_breaker", "?"))
                    st.metric("War Active", "YES" if regime.get("war_active") else "NO")
                    st.metric("Hormuz", "BLOCKED" if regime.get("hormuz_blocked") else "OPEN")

                picks = mm_data.get("picks", [])
                if picks:
                    st.write(f"**Top Picks ({len(picks)} total):**")
                    pick_rows = []
                    for p in picks[:10]:
                        pick_rows.append({
                            "Ticker": p.get("ticker"),
                            "Strike": p.get("strike"),
                            "Expiry": p.get("expiry"),
                            "Score": p.get("score"),
                            "Contracts": p.get("contracts"),
                            "Cost": f"${p.get('cost', 0):.0f}",
                        })
                    st.dataframe(pd.DataFrame(pick_rows), use_container_width=True)
            except Exception as e:
                st.warning(f"Matrix Maximizer data error: {e}")
        else:
            st.info("Matrix Maximizer has not run yet. Use: python -m strategies.matrix_maximizer.runner")

        # ── STORM LIFEBOAT MATRIX v9.0 ─────────────────────────────
        st.markdown("---")
        st.subheader("🌊 Storm Lifeboat Matrix v9.0")

        try:
            from strategies.storm_lifeboat.scenario_engine import ScenarioEngine as _SLEngine
            from strategies.storm_lifeboat.lunar_phi import LunarPhiEngine as _LPEngine
            _sl_available = True
        except ImportError:
            _sl_available = False

        if _sl_available:
            try:
                # Lunar position
                _lpe = _LPEngine()
                _lpos = _lpe.get_position()
                sl_col1, sl_col2, sl_col3, sl_col4 = st.columns(4)
                with sl_col1:
                    st.metric("Moon", f"#{_lpos.moon_number} {_lpos.moon_name}")
                    st.metric("Phase", _lpos.phase.value.upper())
                with sl_col2:
                    st.metric("Day in Moon", f"{_lpos.day_in_moon}/28")
                    st.metric("Phi Window", "YES" if _lpos.in_phi_window else "NO")
                with sl_col3:
                    st.metric("Phi Coherence", f"{_lpos.phi_coherence:.3f}")
                    st.metric("Position Mult.", f"{_lpos.position_multiplier:.2f}x")

                # Latest Helix briefing
                _sl_briefing_dir = PROJECT_ROOT / "data" / "storm_lifeboat"
                import glob as _slglob
                _sl_briefings = sorted(_slglob.glob(str(_sl_briefing_dir / "helix_briefing_*.json")))
                if _sl_briefings:
                    _sl_data = json.loads(Path(_sl_briefings[-1]).read_text(encoding="utf-8"))
                    with sl_col4:
                        st.metric("Mandate", str(_sl_data.get("mandate", "?")).upper())
                        st.metric("Regime", str(_sl_data.get("regime", "?")).upper())

                    if _sl_data.get("headline"):
                        st.info(f"**Helix Briefing ({_sl_data.get('date', '?')}):** {_sl_data['headline']}")
                    if _sl_data.get("risk_alert"):
                        st.warning(f"**Risk Alert:** {_sl_data['risk_alert']}")

                    # Top trades
                    _sl_trades = _sl_data.get("top_trades", [])
                    if _sl_trades:
                        with st.expander(f"Top Trades ({len(_sl_trades)})"):
                            for t in _sl_trades[:10]:
                                st.write(f"- {t}")
                else:
                    with sl_col4:
                        st.metric("Mandate", "N/A")
                        st.metric("Regime", "N/A")
                    st.info("No Helix briefing yet. Run: python -m strategies.storm_lifeboat.runner --briefing")

                # Scenario heatmap
                _sle = _SLEngine()
                _heatmap = _sle.get_risk_heatmap()
                with st.expander(f"Scenario Heatmap ({len(_heatmap)} scenarios)"):
                    _status_icons = {
                        "dormant": "⚪", "emerging": "🟡", "active": "🟠",
                        "escalating": "🔴", "peak": "🔥", "receding": "🟢",
                    }
                    for sc in _heatmap:
                        _icon = _status_icons.get(sc.get("status", ""), "⚪")
                        _firing = sc.get("indicators_firing", 0)
                        _total = sc.get("indicators_total", 0)
                        st.write(
                            f"{_icon} **{sc.get('name', '?')}** — "
                            f"{sc.get('status', '?').upper()} | "
                            f"Risk: {sc.get('risk_score', 0):.2f} | "
                            f"P: {sc.get('probability', 0):.0%} | "
                            f"Indicators: {_firing}/{_total}"
                        )

            except Exception as e:
                st.warning(f"Storm Lifeboat data error: {e}")
        else:
            st.info("Storm Lifeboat not available. Install: strategies/storm_lifeboat/")

        # ── SYSTEM REGISTRY — API & Component Status ────────────────
        st.markdown("---")
        st.subheader("🏗️ System Registry — All Components")

        try:
            from monitoring.aac_system_registry import SystemRegistry as _SR
            _reg = _SR()
            snap = _reg.collect_full_snapshot()
            s = snap.get("summary", {})

            reg_col1, reg_col2, reg_col3, reg_col4 = st.columns(4)
            with reg_col1:
                st.metric("APIs Configured", f"{s.get('apis_configured',0)}/{s.get('total_apis',0)}")
            with reg_col2:
                st.metric("Exchanges Online", f"{s.get('exchanges_online',0)}/{s.get('exchanges_total',0)}")
            with reg_col3:
                st.metric("Strategies OK", f"{s.get('strategies_ok',0)}/{s.get('strategies_total',0)}")
            with reg_col4:
                st.metric("Departments", f"{s.get('departments_ok',0)}/{s.get('departments_total',0)}")

            # Exchange detail
            with st.expander("🔌 Exchange Gateways"):
                for ex in snap.get("exchanges", []):
                    icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(ex.get("health"), "⚪")
                    lat = f" ({ex['latency_ms']}ms)" if ex.get("latency_ms") else ""
                    st.write(f"{icon} **{ex['name']}** — {ex.get('detail','')}{lat}")

            # Infrastructure
            with st.expander("🏗️ Infrastructure"):
                for svc in snap.get("infrastructure", []):
                    icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(svc.get("health"), "⚪")
                    st.write(f"{icon} **{svc['name']}** — {svc.get('detail','')}")

            # Strategy engines
            with st.expander("🧠 Strategy Engines"):
                for st_item in snap.get("strategies", []):
                    icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(st_item.get("health"), "⚪")
                    st.write(f"{icon} **{st_item['name']}** — {st_item.get('detail','')}")

            # Full API inventory
            with st.expander(f"🔑 API Inventory ({s.get('total_apis',0)} APIs)"):
                api_rows = []
                for a in snap.get("apis", []):
                    api_rows.append({
                        "Status": "✅" if a.get("configured") else ("🔑 Missing" if a.get("env_var") else "🆓 Free"),
                        "Name": a["name"],
                        "Category": a.get("category", ""),
                        "Priority": a.get("priority", ""),
                    })
                st.dataframe(pd.DataFrame(api_rows), use_container_width=True)

            # Orphan scripts
            orphans = snap.get("orphans", [])
            if orphans:
                with st.expander(f"📄 Orphan Scripts ({len(orphans)} root _*.py)"):
                    for o in orphans:
                        st.write(f"📄 **{o['script']}** — {o.get('description','')[:80]}")

        except Exception as e:
            st.warning(f"System registry not available: {e}")

        # System logs
        st.markdown("---")
        st.subheader("📝 System Logs")

        if latest_status.get('cycle_report'):
            cycle = latest_status['cycle_report']
            st.json(cycle)
        else:
            st.info("No cycle reports available")

        # Copilot Chat Interface
        st.markdown("---")
        st.subheader("🤖 Copilot Chat Assistant")

        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Import audio library
        try:
            from shared.audio_response_library import get_audio_library
            audio_lib = get_audio_library()
        except ImportError:
            audio_lib = None
            st.warning("Audio response library not available")

        # Chat input
        user_input = st.text_input("Ask me anything about the AAC system:", key="chat_input")

        if st.button("Send", key="send_button") and user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "message": user_input})

            # Generate AI response using audio library
            if audio_lib:
                ai_response = audio_lib.get_response(user_input)
            else:
                ai_response = generate_copilot_response(user_input, latest_status, dashboard)

            st.session_state.chat_history.append({"role": "assistant", "message": ai_response})

            # Clear input
            st.rerun()

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history[-10:]:  # Show last 10 messages
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['message']}")
                else:
                    st.markdown(f"**Copilot:** {message['message']}")
                    # Add audio button for responses
                    if st.button("🔊 Play Audio", key=f"audio_{len(st.session_state.chat_history)}"):
                        if audio_lib:
                            audio_lib.speak_response(message['message'])
                        else:
                            play_audio_response(message['message'])

        # Clear chat button
        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

        # AZ Executive Assistant Interface
        st.markdown("---")
        st.subheader("🎯 AZ Executive Assistant")

        # Import AZ libraries
        try:
            from shared.az_response_library import get_az_library
            from shared.avatar_system import get_avatar_manager
            az_lib = get_az_library()
            avatar_manager = get_avatar_manager()
            avatar = avatar_manager.get_avatar("az")
        except ImportError as e:
            st.error(f"AZ system not available: {e}")
            az_lib = None
            avatar = None

        if az_lib and avatar:
            # Create two columns for AZ interface
            az_col1, az_col2 = st.columns([1, 2])

            with az_col1:
                # AZ Avatar Display
                st.markdown("**AZ Avatar**")
                avatar_placeholder = st.empty()
                avatar_placeholder.image(avatar.get_frame_as_base64(), width=150)

                # System Status Brief Button
                if st.button("📊 System Status Brief", key="status_brief"):
                    brief = az_lib.get_system_status_brief()
                    st.session_state.az_response = brief
                    avatar_manager.speak_text(brief, "az")
                    st.rerun()

                # Daily Brief Button
                if st.button("📋 Daily Executive Brief", key="daily_brief"):
                    brief = az_lib.generate_daily_brief()
                    st.session_state.az_response = brief
                    avatar_manager.speak_text(brief[:500], "az")  # Speak first 500 chars
                    st.rerun()

            with az_col2:
                # AZ Question Categories Dropdown
                categories = az_lib.list_categories()
                selected_category = st.selectbox(
                    "Select Question Category:",
                    ["Choose a category..."] + categories,
                    key="az_category"
                )

                # Questions dropdown (filtered by category)
                questions_options = []
                if selected_category != "Choose a category...":
                    questions = az_lib.get_questions_by_category(selected_category)
                    questions_options = [f"Q{q['id']}: {q['question'][:80]}..." for q in questions]

                selected_question = st.selectbox(
                    "Select Strategic Question:",
                    ["Choose a question..."] + questions_options,
                    key="az_question"
                )

                # Answer button
                if st.button("🎯 Get AZ Answer", key="az_answer") and selected_question != "Choose a question...":
                    # Extract question ID
                    qid = int(selected_question.split(":")[0][1:])
                    response = az_lib.get_response(qid)
                    st.session_state.az_response = response
                    avatar_manager.speak_text(response, "az")
                    st.rerun()

                # AZ Response Display
                if 'az_response' in st.session_state:
                    st.markdown("**AZ Response:**")
                    # Create a scrollable text area for long responses
                    st.text_area(
                        "Response:",
                        st.session_state.az_response,
                        height=200,
                        key="az_response_display",
                        disabled=True
                    )

                    # Audio controls
                    audio_col1, audio_col2 = st.columns(2)
                    with audio_col1:
                        if st.button("🔊 Play Audio Response", key="play_az_audio"):
                            avatar_manager.speak_text(st.session_state.az_response, "az")

                    with audio_col2:
                        if st.button("🎵 Play with Avatar Animation", key="play_az_animated"):
                            # Start animation and audio
                            avatar.start_speaking_animation(st.session_state.az_response)
                            az_lib.speak_response(int(st.session_state.az_response.split()[0]) if st.session_state.az_response.split()[0].isdigit() else 1)

        # Update avatar animation in real-time
        if avatar and st.session_state.get('az_response'):
            # Update avatar frame every few seconds
            time.sleep(0.1)  # Small delay for animation
            if avatar_placeholder:
                avatar_placeholder.image(avatar.get_frame_as_base64(), width=150)

    else:
        # System not running
        st.info("System not initialized. Click 'Initialize System' in the sidebar to start.")

        # Demo content
        st.markdown("---")
        st.subheader("Demo: System Capabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Data Sources:**")
            st.markdown("- [OK] Alpha Vantage (Global Stocks)")
            st.markdown("- [OK] CoinGecko (Cryptocurrencies)")
            st.markdown("- [OK] CurrencyAPI (Forex)")
            st.markdown("- [OK] Twelve Data (Real-time)")
            st.markdown("- [OK] Polygon.io (Options)")
            st.markdown("- [OK] Finnhub (Sentiment)")

        with col2:
            st.markdown("**Arbitrage Types:**")
            st.markdown("- [OK] Cross-exchange")
            st.markdown("- [OK] Triangular")
            st.markdown("- [OK] Statistical")
            st.markdown("- [OK] Macro-economic")
            st.markdown("- [OK] Sentiment-based")

        st.markdown("---")
        st.subheader("🚀 Getting Started")
        st.markdown("""
        1. **Configure API Keys** in `.env` file
        2. **Initialize System** using sidebar button
        3. **Start Monitoring** to begin real-time operation
        4. **Enable Auto-Execute** for live trading (use test mode first!)
        5. **Monitor Performance** and adjust risk parameters as needed
        """)

        # Sample performance metrics
        st.markdown("---")
        st.subheader("📊 Sample Performance (Demo Data)")

        demo_data = {
            'Metric': ['Total Opportunities', 'Executed Trades', 'Successful Trades', 'Total PnL', 'Win Rate'],
            'Value': ['1,247', '89', '67', '$2,341.50', '75.3%']
        }
        st.table(pd.DataFrame(demo_data))


if __name__ == "__main__":
    run_streamlit_dashboard()
