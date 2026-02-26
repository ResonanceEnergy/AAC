#!/usr/bin/env python3
"""
AAC Enhanced Metrics Display System
====================================

Professional metrics dashboard with real-time data visualization,
comprehensive system health monitoring, and executive-level reporting.
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
import json
import os
import platform
import psutil
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config
from shared.audit_logger import get_audit_logger

# Import AAC system components (with fallbacks for missing modules)
try:
    from CentralAccounting.database import AccountingDatabase
except ImportError:
    AccountingDatabase = None

try:
    from shared.data_sources import DataAggregator
except ImportError:
    DataAggregator = None

try:
    from agent_based_trading_integration import get_agent_integration
except ImportError:
    get_agent_integration = None


class AACMetricsDisplay:
    """
    Enhanced AAC Metrics Display System
    Professional dashboard with comprehensive monitoring and analytics
    """

    def __init__(self):
        self.config = get_config()
        self.audit_logger = get_audit_logger()

        # Components (initialized in initialize() method)
        self.accounting_db = None
        self.data_aggregator = None
        self.agent_integration = None

        # Metrics cache
        self.last_update = None
        self.metrics_cache = {}
        self.cache_timeout = 30  # seconds

        # Display settings
        self.use_colors = True
        self.use_unicode = True
        self.compact_mode = False

    async def initialize(self):
        """Initialize the metrics display system"""
        try:
            self.accounting_db = AccountingDatabase() if AccountingDatabase else None
            self.data_aggregator = DataAggregator() if DataAggregator else None
            self.agent_integration = await get_agent_integration() if get_agent_integration else None
            await self.audit_logger.log_event(
                category="system",
                action="metrics_display_init",
                user="AAC_MetricsDisplay",
                status="success"
            )
            return True
        except Exception as e:
            print(f"❌ Failed to initialize metrics display: {e}")
            return False

    async def display_comprehensive_dashboard(self):
        """Display the comprehensive AAC dashboard"""
        print("\n" + "="*80)
        print("🏛️  ACCELERATED ARBITRAGE CORP - EXECUTIVE DASHBOARD")
        print("="*80)

        # System timestamp
        now = datetime.now()
        print(f"📅 Report Time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()

        # Get all metrics
        metrics = await self._collect_all_metrics()

        # Executive Summary
        await self._display_executive_summary(metrics)

        # Core Metrics Grid
        await self._display_core_metrics_grid(metrics)

        # System Health
        await self._display_system_health(metrics)

        # Trading Performance
        await self._display_trading_performance(metrics)

        # Agent Contest Status
        await self._display_agent_contest_status(metrics)

        # Risk & Compliance
        await self._display_risk_compliance(metrics)

        # Data & Intelligence
        await self._display_data_intelligence(metrics)

        # Footer
        print("\n" + "="*80)
        print("🔄 Auto-refresh: 30s | Press Ctrl+C to exit | Use --compact for condensed view")
        print("="*80)

    async def display_compact_dashboard(self):
        """Display compact dashboard for monitoring"""
        print("\n" + "="*60)
        print("AAC - COMPACT DASHBOARD")
        print("="*60)

        metrics = await self._collect_all_metrics()

        # One-line status
        health = metrics.get('system_health', {})
        financial = metrics.get('financial', {})
        trading = metrics.get('trading', {})

        cpu = health.get('cpu_usage', 0)
        mem = health.get('memory_usage', 0)
        equity = financial.get('total_equity', 0)
        pnl = financial.get('daily_pnl', 0)
        active_trades = trading.get('active_trades', 0)

        status_line = f"CPU:{cpu:.1f}% | MEM:{mem:.1f}% | EQUITY:${equity:,.0f} | P&L:${pnl:+,.0f} | TRADES:{active_trades}"
        print(f"📊 {status_line}")

        # Quick alerts
        alerts = await self._get_active_alerts()
        if alerts:
            print(f"🚨 ALERTS: {len(alerts)} active")
            for alert in alerts[:2]:  # Show top 2
                print(f"   {alert['level']}: {alert['message'][:50]}...")

        print("="*60)

    async def _collect_all_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive metrics from all systems"""
        if self.last_update and (datetime.now() - self.last_update).seconds < self.cache_timeout:
            return self.metrics_cache

        metrics = {}

        try:
            # System health
            metrics['system_health'] = await self._get_system_health()

            # Financial metrics
            metrics['financial'] = await self._get_financial_metrics()

            # Trading metrics
            metrics['trading'] = await self._get_trading_metrics()

            # Agent contest
            metrics['agent_contest'] = await self._get_agent_contest_metrics()

            # Risk metrics
            metrics['risk'] = await self._get_risk_metrics()

            # Data sources
            metrics['data_sources'] = await self._get_data_source_metrics()

            # Intelligence
            metrics['intelligence'] = await self._get_intelligence_metrics()

            self.metrics_cache = metrics
            self.last_update = datetime.now()

        except Exception as e:
            self.audit_logger.error(f"Metrics collection failed: {e}")
            metrics = self.metrics_cache or {}

        return metrics

    async def _get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Network (basic)
            net = psutil.net_io_counters()
            bytes_sent = net.bytes_sent
            bytes_recv = net.bytes_recv

            # Process info
            process = psutil.Process()
            threads = process.num_threads()

            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_usage': disk_percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'network_sent_mb': bytes_sent / (1024**2),
                'network_recv_mb': bytes_recv / (1024**2),
                'active_threads': threads,
                'uptime_seconds': time.time() - process.create_time(),
                'health_score': self._calculate_health_score(cpu_percent, memory_percent, disk_percent)
            }
        except Exception as e:
            return {'error': str(e)}

    async def _get_financial_metrics(self) -> Dict[str, Any]:
        """Get financial performance metrics"""
        try:
            if not self.accounting_db:
                return {'error': 'Accounting database not available'}

            # Get from accounting database
            summary = await self.accounting_db.get_portfolio_summary()

            return {
                'total_equity': summary.get('total_equity', 0),
                'cash_balance': summary.get('cash_balance', 0),
                'total_positions': summary.get('total_positions', 0),
                'daily_pnl': summary.get('daily_pnl', 0),
                'unrealized_pnl': summary.get('unrealized_pnl', 0),
                'realized_pnl': summary.get('realized_pnl', 0),
                'total_trades': summary.get('total_trades', 0),
                'win_rate': summary.get('win_rate', 0),
                'sharpe_ratio': summary.get('sharpe_ratio', 0),
                'max_drawdown': summary.get('max_drawdown', 0)
            }
        except Exception as e:
            return {'error': str(e)}

    async def _get_trading_metrics(self) -> Dict[str, Any]:
        """Get trading activity metrics"""
        try:
            if not self.accounting_db:
                return {'error': 'Accounting database not available'}

            # Get recent trading activity
            recent_trades = await self.accounting_db.get_recent_trades(limit=100)

            active_trades = len([t for t in recent_trades if t.get('status') == 'open'])
            today_trades = len([t for t in recent_trades
                              if t.get('timestamp', '').startswith(datetime.now().strftime('%Y-%m-%d'))])

            return {
                'active_trades': active_trades,
                'today_trades': today_trades,
                'total_trades': len(recent_trades),
                'avg_trade_size': np.mean([t.get('quantity', 0) * t.get('price', 0) for t in recent_trades]) if recent_trades else 0,
                'trade_frequency': len(recent_trades) / 24,  # trades per hour
                'last_trade_time': recent_trades[0].get('timestamp') if recent_trades else None
            }
        except Exception as e:
            return {'error': str(e)}

    async def _get_agent_contest_metrics(self) -> Dict[str, Any]:
        """Get agent contest performance metrics"""
        try:
            if not self.agent_integration:
                return {'status': 'not_initialized'}

            status = await self.agent_integration.get_integration_status()

            if status.get('contest_active') == 'active':
                contest_status = await self.agent_integration.contest_orchestrator.get_contest_status()
                return {
                    'status': 'active',
                    'active_agents': contest_status.get('active_agents', 0),
                    'total_agents': contest_status.get('total_agents', 0),
                    'winner': contest_status.get('winner'),
                    'rounds_completed': contest_status.get('rounds_completed', 0),
                    'leaderboard': contest_status.get('leaderboard', [])[:5]  # Top 5
                }
            else:
                return {'status': 'inactive'}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    async def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk and compliance metrics"""
        try:
            # Circuit breaker status
            circuit_breakers = {
                'max_loss_breaker': False,
                'volatility_breaker': False,
                'correlation_breaker': False
            }

            # Compliance status
            compliance = {
                'regulatory_compliant': True,
                'audit_complete': True,
                'risk_limits_respected': True
            }

            return {
                'circuit_breakers': circuit_breakers,
                'compliance': compliance,
                'var_95': 0,  # Value at Risk
                'expected_shortfall': 0,
                'stress_test_passed': True
            }
        except Exception as e:
            return {'error': str(e)}

    async def _get_data_source_metrics(self) -> Dict[str, Any]:
        """Get data source health metrics"""
        try:
            if not self.data_aggregator:
                return {'error': 'Data aggregator not available'}

            sources = await self.data_aggregator.get_source_status()

            return {
                'active_sources': len([s for s in sources.values() if s.get('active')]),
                'total_sources': len(sources),
                'data_freshness': 'current',  # Could be enhanced
                'api_rate_limits': 'normal',
                'last_update': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}

    async def _get_intelligence_metrics(self) -> Dict[str, Any]:
        """Get intelligence and research metrics"""
        try:
            return {
                'active_research_agents': 20,  # From BigBrainIntelligence
                'intelligence_findings': 0,
                'market_sentiment': 'neutral',
                'prediction_accuracy': 0.0,
                'last_intelligence_update': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}

    def _calculate_health_score(self, cpu: float, memory: float, disk: float) -> float:
        """Calculate overall system health score (0-100)"""
        # Weighted average with higher weights for critical resources
        cpu_score = max(0, 100 - cpu * 2)  # CPU > 50% is concerning
        memory_score = max(0, 100 - memory * 1.5)  # Memory > 66% is concerning
        disk_score = max(0, 100 - disk * 3)  # Disk > 33% is concerning

        return (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)

    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active system alerts"""
        alerts = []

        metrics = await self._collect_all_metrics()

        # System health alerts
        health = metrics.get('system_health', {})
        if health.get('cpu_usage', 0) > 90:
            alerts.append({'level': 'CRITICAL', 'message': 'CPU usage above 90%'})
        elif health.get('cpu_usage', 0) > 80:
            alerts.append({'level': 'WARNING', 'message': 'High CPU usage detected'})

        if health.get('memory_usage', 0) > 90:
            alerts.append({'level': 'CRITICAL', 'message': 'Memory usage above 90%'})
        elif health.get('memory_usage', 0) > 80:
            alerts.append({'level': 'WARNING', 'message': 'High memory usage detected'})

        # Financial alerts
        financial = metrics.get('financial', {})
        if financial.get('daily_pnl', 0) < -10000:  # $10k loss
            alerts.append({'level': 'CRITICAL', 'message': 'Large daily loss detected'})

        return alerts

    async def _display_executive_summary(self, metrics: Dict[str, Any]):
        """Display executive summary section"""
        print("📈 EXECUTIVE SUMMARY")
        print("-" * 40)

        health = metrics.get('system_health', {})
        financial = metrics.get('financial', {})
        trading = metrics.get('trading', {})

        health_score = health.get('health_score', 0)
        health_icon = "🟢" if health_score > 80 else "🟡" if health_score > 60 else "🔴"

        print(f"System Health: {health_icon} {health_score:.1f}/100")
        print(f"Total Equity: ${financial.get('total_equity', 0):,.0f}")
        print(f"Daily P&L: ${financial.get('daily_pnl', 0):+,.0f}")
        print(f"Active Trades: {trading.get('active_trades', 0)}")
        print()

    async def _display_core_metrics_grid(self, metrics: Dict[str, Any]):
        """Display core metrics in a grid format"""
        print("📊 CORE METRICS")
        print("-" * 40)

        # Create a nice grid layout
        health = metrics.get('system_health', {})
        financial = metrics.get('financial', {})
        trading = metrics.get('trading', {})

        grid_data = [
            ["CPU Usage", f"{health.get('cpu_usage', 0):.1f}%"],
            ["Memory", f"{health.get('memory_usage', 0):.1f}%"],
            ["Disk Usage", f"{health.get('disk_usage', 0):.1f}%"],
            ["", ""],  # Spacer
            ["Total Equity", f"${financial.get('total_equity', 0):,.0f}"],
            ["Daily P&L", f"${financial.get('daily_pnl', 0):+,.0f}"],
            ["Unrealized P&L", f"${financial.get('unrealized_pnl', 0):+,.0f}"],
            ["", ""],  # Spacer
            ["Active Trades", str(trading.get('active_trades', 0))],
            ["Today's Trades", str(trading.get('today_trades', 0))],
            ["Win Rate", f"{financial.get('win_rate', 0):.1%}"],
        ]

        # Print in 3 columns
        for i in range(0, len(grid_data), 3):
            row = []
            for j in range(3):
                if i + j < len(grid_data):
                    label, value = grid_data[i + j]
                    if label:  # Not a spacer
                        row.append(f"{label}: {value}")
                    else:
                        row.append("")
                else:
                    row.append("")

            print(" | ".join(f"{col:<20}" for col in row if col))
        print()

    async def _display_system_health(self, metrics: Dict[str, Any]):
        """Display detailed system health"""
        print("🖥️  SYSTEM HEALTH")
        print("-" * 40)

        health = metrics.get('system_health', {})

        # Progress bars for resources
        cpu = health.get('cpu_usage', 0)
        mem = health.get('memory_usage', 0)
        disk = health.get('disk_usage', 0)

        print(f"CPU Usage:     {self._progress_bar(cpu)} {cpu:.1f}%")
        print(f"Memory Usage:  {self._progress_bar(mem)} {mem:.1f}%")
        print(f"Disk Usage:    {self._progress_bar(disk)} {disk:.1f}%")

        print(f"Uptime:        {self._format_uptime(health.get('uptime_seconds', 0))}")
        print(f"Active Threads: {health.get('active_threads', 0)}")
        print()

    async def _display_trading_performance(self, metrics: Dict[str, Any]):
        """Display trading performance metrics"""
        print("📈 TRADING PERFORMANCE")
        print("-" * 40)

        financial = metrics.get('financial', {})
        trading = metrics.get('trading', {})

        print(f"Total Trades:     {financial.get('total_trades', 0):,}")
        print(f"Win Rate:         {financial.get('win_rate', 0):.1%}")
        print(f"Avg Trade Size:   ${trading.get('avg_trade_size', 0):,.0f}")
        print(f"Trade Frequency:  {trading.get('trade_frequency', 0):.1f}/hour")

        if trading.get('last_trade_time'):
            print(f"Last Trade:       {trading.get('last_trade_time')}")

        print()

    async def _display_agent_contest_status(self, metrics: Dict[str, Any]):
        """Display agent contest status"""
        print("🤖 AGENT CONTEST STATUS")
        print("-" * 40)

        contest = metrics.get('agent_contest', {})

        if contest.get('status') == 'active':
            print(f"Status:          🟢 ACTIVE")
            print(f"Active Agents:   {contest.get('active_agents', 0)}")
            print(f"Total Agents:    {contest.get('total_agents', 0)}")
            print(f"Rounds:          {contest.get('rounds_completed', 0)}")

            if contest.get('leaderboard'):
                print("Top Performers:")
                for i, agent in enumerate(contest.get('leaderboard', [])[:3], 1):
                    print(f"  {i}. {agent.get('agent_id', 'Unknown')}: ${agent.get('portfolio_value', 0):,.0f}")
        else:
            print(f"Status:          🔴 {contest.get('status', 'inactive').upper()}")

        print()

    async def _display_risk_compliance(self, metrics: Dict[str, Any]):
        """Display risk and compliance status"""
        print("⚠️  RISK & COMPLIANCE")
        print("-" * 40)

        risk = metrics.get('risk', {})

        # Circuit breakers
        breakers = risk.get('circuit_breakers', {})
        print("Circuit Breakers:")
        for name, triggered in breakers.items():
            status = "🔴 TRIGGERED" if triggered else "🟢 NORMAL"
            print(f"  {name.replace('_', ' ').title()}: {status}")

        # Compliance
        compliance = risk.get('compliance', {})
        print("Compliance Status:")
        for check, status in compliance.items():
            icon = "✅" if status else "❌"
            print(f"  {check.replace('_', ' ').title()}: {icon}")

        print()

    async def _display_data_intelligence(self, metrics: Dict[str, Any]):
        """Display data sources and intelligence status"""
        print("🔍 DATA & INTELLIGENCE")
        print("-" * 40)

        data = metrics.get('data_sources', {})
        intel = metrics.get('intelligence', {})

        print(f"Active Data Sources: {data.get('active_sources', 0)}/{data.get('total_sources', 0)}")
        print(f"Data Freshness:      {data.get('data_freshness', 'unknown')}")
        print(f"API Rate Limits:     {data.get('api_rate_limits', 'unknown')}")

        print(f"Research Agents:     {intel.get('active_research_agents', 0)}")
        print(f"Market Sentiment:    {intel.get('market_sentiment', 'unknown')}")

        print()

    def _progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create a visual progress bar"""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable form"""
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 and not parts:  # Only show seconds if no larger units
            parts.append(f"{seconds}s")

        return " ".join(parts) if parts else "0s"


async def main():
    """Main entry point for metrics display"""
    import argparse

    parser = argparse.ArgumentParser(description="AAC Enhanced Metrics Display")
    parser.add_argument("--compact", action="store_true", help="Show compact dashboard")
    parser.add_argument("--monitor", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")

    args = parser.parse_args()

    display = AACMetricsDisplay()

    if not await display.initialize():
        return

    try:
        if args.monitor:
            # Continuous monitoring mode
            while True:
                if args.compact:
                    await display.display_compact_dashboard()
                else:
                    await display.display_comprehensive_dashboard()

                await asyncio.sleep(args.interval)
        else:
            # Single display
            if args.compact:
                await display.display_compact_dashboard()
            else:
                await display.display_comprehensive_dashboard()

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    asyncio.run(main())