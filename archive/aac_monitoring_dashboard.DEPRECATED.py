#!/usr/bin/env python3
"""
AAC Arbitrage Monitoring Dashboard
==================================

⚠️  DEPRECATED: This file is deprecated and will be removed.
   Use 'python aac_master_launcher.py --dashboard-only --display-mode web' instead.

New unified launcher:
    python aac_master_launcher.py --dashboard-only --display-mode web

Real-time monitoring and control dashboard for AAC arbitrage system.
Provides comprehensive oversight of trading activities, performance metrics,
and system health.

Features:
- Real-time position monitoring
- Performance analytics
- Risk management dashboard
- System health indicators
- Trade execution logs
- Interactive controls

Web-based dashboard using Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
import threading
import queue

# Import AAC components
from aac_arbitrage_execution_system import AACArbitrageExecutionSystem, ExecutionConfig
from binance_trading_engine import TradingConfig
from binance_arbitrage_integration import BinanceConfig

class AACMonitoringDashboard:
    """Real-time AAC monitoring dashboard"""

    def __init__(self):
        self.execution_system: Optional[AACArbitrageExecutionSystem] = None
        self.status_queue = queue.Queue()
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None

    def initialize_system(self):
        """Initialize the arbitrage execution system"""
        try:
            execution_config = ExecutionConfig()
            self.execution_system = AACArbitrageExecutionSystem(execution_config)
            asyncio.run(self.execution_system.initialize())
            return True
        except Exception as e:
            st.error(f"Failed to initialize system: {e}")
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

    def get_latest_status(self) -> Optional[Dict]:
        """Get the latest system status"""
        try:
            while not self.status_queue.empty():
                latest = self.status_queue.get_nowait()
            return latest
        except:
            return None

def create_performance_chart(data: List[Dict]) -> go.Figure:
    """Create performance chart"""
    if not data:
        return go.Figure()

    df = pd.DataFrame(data)
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

    return fig

def create_opportunities_table(opportunities: List[Dict]) -> pd.DataFrame:
    """Create opportunities DataFrame"""
    if not opportunities:
        return pd.DataFrame()

    df = pd.DataFrame(opportunities)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['spread'] = df['spread'].apply(lambda x: f"{x:.2%}")
    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")

    return df[['symbol', 'spread', 'confidence', 'type', 'executed', 'timestamp']]

def main():
    """Main dashboard function"""
    print("⚠️  DEPRECATED: aac_monitoring_dashboard.py is deprecated!")
    print("   Use: python aac_master_launcher.py --dashboard-only --display-mode web")
    print()

    st.set_page_config(
        page_title="AAC Arbitrage Dashboard",
        page_icon="📊",
        layout="wide"
    )

    st.title("🚀 AAC Arbitrage Monitoring Dashboard")
    st.markdown("---")

    # Initialize dashboard
    dashboard = AACMonitoringDashboard()

    # Sidebar controls
    st.sidebar.title("🎛️ Controls")

    if st.sidebar.button("🔄 Initialize System"):
        with st.spinner("Initializing AAC system..."):
            if dashboard.initialize_system():
                st.sidebar.success("✅ System initialized")
            else:
                st.sidebar.error("❌ Initialization failed")

    if st.sidebar.button("▶️ Start Monitoring"):
        dashboard.start_monitoring()
        st.sidebar.success("✅ Monitoring started")

    if st.sidebar.button("⏹️ Stop Monitoring"):
        dashboard.stop_monitoring()
        st.sidebar.success("✅ Monitoring stopped")

    # Manual cycle execution
    if st.sidebar.button("🔄 Run Arbitrage Cycle"):
        if dashboard.execution_system:
            with st.spinner("Running arbitrage cycle..."):
                try:
                    cycle_report = asyncio.run(dashboard.execution_system.run_arbitrage_cycle())
                    st.sidebar.success(f"✅ Cycle complete - {cycle_report.get('opportunities_detected', 0)} opportunities")
                except Exception as e:
                    st.sidebar.error(f"❌ Cycle failed: {e}")
        else:
            st.sidebar.error("❌ System not initialized")

    # Configuration display
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuration")

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
            st.metric("Auto Execute", "ON" if execution_config.auto_execute else "OFF")
            st.metric("Test Mode", "ON" if execution_config.enable_test_mode else "OFF")
            st.metric("Last Update", latest_status.get('timestamp', datetime.now()).strftime('%H:%M:%S'))

        # Performance chart
        st.markdown("---")
        st.subheader("📈 Performance Chart")

        # Mock performance data for demo (would be real data in production)
        performance_data = [
            {'timestamp': datetime.now() - timedelta(hours=i),
             'pnl': 100 * (i % 10 - 5),
             'win_rate': 0.6 + 0.1 * (i % 3)}
            for i in range(24)
        ]

        perf_chart = create_performance_chart(performance_data)
        st.plotly_chart(perf_chart, use_container_width=True)

        # Recent opportunities
        st.markdown("---")
        st.subheader("🎯 Recent Opportunities")

        opportunities = status.get('recent_opportunities', [])
        if opportunities:
            opp_df = create_opportunities_table(opportunities)
            st.dataframe(opp_df, use_container_width=True)
        else:
            st.info("No opportunities detected yet")

        # Active positions
        st.markdown("---")
        st.subheader("📋 Active Positions")

        if dashboard.execution_system and hasattr(dashboard.execution_system, 'trading_engine'):
            try:
                # Get portfolio summary
                summary = dashboard.execution_system.trading_engine.get_portfolio_summary()

                if summary['positions']:
                    pos_df = pd.DataFrame(summary['positions'])
                    st.dataframe(pos_df, use_container_width=True)
                else:
                    st.info("No active positions")

            except Exception as e:
                st.error(f"Error getting positions: {e}")
        else:
            st.info("Trading engine not available")

        # System logs
        st.markdown("---")
        st.subheader("📝 System Logs")

        if latest_status.get('cycle_report'):
            cycle = latest_status['cycle_report']
            st.json(cycle)
        else:
            st.info("No cycle reports available")

    else:
        # System not running
        st.info("🔄 System not initialized. Click 'Initialize System' in the sidebar to start.")

        # Demo content
        st.markdown("---")
        st.subheader("🎯 Demo: System Capabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Data Sources:**")
            st.markdown("- ✅ Alpha Vantage (Global Stocks)")
            st.markdown("- ✅ CoinGecko (Cryptocurrencies)")
            st.markdown("- ✅ CurrencyAPI (Forex)")
            st.markdown("- ✅ Twelve Data (Real-time)")
            st.markdown("- ✅ Polygon.io (Options)")
            st.markdown("- ✅ Finnhub (Sentiment)")

        with col2:
            st.markdown("**Arbitrage Types:**")
            st.markdown("- ✅ Cross-exchange")
            st.markdown("- ✅ Triangular")
            st.markdown("- ✅ Statistical")
            st.markdown("- ✅ Macro-economic")
            st.markdown("- ✅ Sentiment-based")

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
    main()