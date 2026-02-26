#!/usr/bin/env python3
"""
AAC 2100 Real-Time Monitoring Dashboard
========================================
Comprehensive real-time monitoring and display system for live trading operations.

Features:
- Real-time P&L tracking
- System health monitoring
- Circuit breaker status
- Risk metrics dashboard
- Trading activity feed
- Alert management
- Performance analytics
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

# Try to import curses, fallback for Windows
try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False
    print("Warning: curses not available, using text-based dashboard")

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config
from shared.monitoring import get_monitoring_manager
from shared.production_safeguards import get_production_safeguards, get_safeguards_health
from CentralAccounting.financial_analysis_engine import FinancialAnalysisEngine
from CryptoIntelligence.crypto_intelligence_engine import CryptoIntelligenceEngine
from aac.doctrine.doctrine_integration import get_doctrine_integration


class AACMonitoringDashboard:
    """Real-time AAC 2100 monitoring dashboard"""

    def __init__(self):
        self.config = get_config()
        self.monitoring = get_monitoring_manager()
        self.safeguards = get_production_safeguards()
        self.financial_engine = FinancialAnalysisEngine()
        self.crypto_engine = CryptoIntelligenceEngine()
        self.doctrine_integration = None

        # Dashboard state
        self.running = False
        self.last_update = datetime.now()
        self.refresh_rate = 1.0  # seconds

        # Data caches
        self.pnl_data = {}
        self.health_data = {}
        self.risk_data = {}
        self.trading_data = {}
        self.doctrine_data = {}
        self.alerts = []

    async def initialize(self):
        """Initialize monitoring components"""
        print("🔄 Initializing AAC 2100 Monitoring Dashboard...")

        # Initialize safeguards
        from shared.production_safeguards import initialize_production_safeguards
        await initialize_production_safeguards()

        # Initialize doctrine integration
        self.doctrine_integration = get_doctrine_integration()
        await self.doctrine_integration.initialize()

        # Monitoring service is lazy-loaded via get_monitoring_manager()
        # No explicit initialization needed

        print("✅ Monitoring dashboard initialized")

    async def collect_monitoring_data(self) -> Dict[str, Any]:
        """Collect all monitoring data"""
        try:
            # System health
            health_data = await self._get_system_health()

            # P&L data
            pnl_data = await self._get_pnl_data()

            # Risk metrics
            risk_data = await self._get_risk_metrics()

            # Trading activity
            trading_data = await self._get_trading_activity()

            # Doctrine compliance
            doctrine_data = await self._get_doctrine_compliance()

            # Safeguards status
            safeguards_data = get_safeguards_health()

            # Alerts
            alerts_data = await self._get_alerts()

            return {
                'timestamp': datetime.now(),
                'health': health_data,
                'pnl': pnl_data,
                'risk': risk_data,
                'trading': trading_data,
                'doctrine': doctrine_data,
                'safeguards': safeguards_data,
                'alerts': alerts_data
            }

        except Exception as e:
            return {
                'timestamp': datetime.now(),
                'error': str(e),
                'health': {},
                'pnl': {},
                'risk': {},
                'trading': {},
                'doctrine': {},
                'safeguards': {},
                'alerts': []
            }

    async def _get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health data"""
        health = {
            'overall_status': 'healthy',
            'departments': {},
            'infrastructure': {},
            'performance': {}
        }

        try:
            # Department health checks
            departments = ['BigBrainIntelligence', 'CentralAccounting', 'CryptoIntelligence', 'TradingExecution']
            for dept in departments:
                health['departments'][dept] = await self._check_department_health(dept)

            # Infrastructure health
            health['infrastructure'] = {
                'database': await self._check_database_health(),
                'network': await self._check_network_health(),
                'memory': await self._check_memory_usage(),
                'cpu': await self._check_cpu_usage()
            }

            # Performance metrics
            health['performance'] = {
                'latency_p99': 0.0,  # TODO: implement
                'throughput': 0,     # TODO: implement
                'error_rate': 0.0    # TODO: implement
            }

            # Overall status
            dept_statuses = [d.get('status', 'unknown') for d in health['departments'].values()]
            if 'critical' in dept_statuses:
                health['overall_status'] = 'critical'
            elif 'warning' in dept_statuses:
                health['overall_status'] = 'warning'

        except Exception as e:
            health['overall_status'] = 'error'
            health['error'] = str(e)

        return health

    async def _check_department_health(self, department: str) -> Dict[str, Any]:
        """Check health of a specific department"""
        try:
            if department == 'CentralAccounting':
                # Check database connectivity and recent transactions
                db_status = await self.financial_engine.health_check()
                return {
                    'status': 'healthy' if db_status.get('database_connected') else 'critical',
                    'last_transaction': db_status.get('last_transaction_time'),
                    'pending_reconciliations': db_status.get('pending_count', 0)
                }
            elif department == 'CryptoIntelligence':
                # Check venue monitoring and data feeds
                venue_status = await self.crypto_engine.get_venue_health()
                return {
                    'status': 'healthy',
                    'venues_monitored': len(venue_status),
                    'average_health_score': sum(v['health_score'] for v in venue_status.values()) / len(venue_status)
                }
            elif department == 'BigBrainIntelligence':
                # Check agent activity and predictions
                return {
                    'status': 'healthy',
                    'active_agents': 11,  # TODO: implement actual check
                    'predictions_today': 0  # TODO: implement actual check
                }
            elif department == 'TradingExecution':
                # Check trading status and positions
                return {
                    'status': 'healthy',
                    'active_positions': 0,  # TODO: implement actual check
                    'orders_pending': 0     # TODO: implement actual check
                }
            else:
                return {'status': 'unknown'}

        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            # Simple connectivity check
            from CentralAccounting.database import DatabaseManager
            db = DatabaseManager()
            connected = await db.health_check()
            return {
                'status': 'healthy' if connected else 'critical',
                'connection_pool_size': getattr(db, 'pool_size', 1),
                'active_connections': getattr(db, 'active_connections', 1)
            }
        except:
            return {'status': 'critical'}

    async def _check_network_health(self) -> Dict[str, Any]:
        """Check network connectivity"""
        return {
            'status': 'healthy',
            'latency_ms': 15,  # TODO: implement actual network check
            'packet_loss': 0.0
        }

    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        import psutil
        memory = psutil.virtual_memory()
        return {
            'used_percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'status': 'warning' if memory.percent > 85 else 'healthy'
        }

    async def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage"""
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        return {
            'used_percent': cpu_percent,
            'status': 'warning' if cpu_percent > 90 else 'healthy'
        }

    async def _get_pnl_data(self) -> Dict[str, Any]:
        """Get P&L data"""
        try:
            risk_metrics = await self.financial_engine.update_risk_metrics()
            return {
                'daily_pnl': risk_metrics.daily_pnl,
                'total_equity': risk_metrics.total_equity,
                'unrealized_pnl': risk_metrics.unrealized_pnl,
                'realized_pnl': risk_metrics.realized_pnl,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'max_drawdown': risk_metrics.max_drawdown_pct
            }
        except:
            return {
                'daily_pnl': 0.0,
                'total_equity': 100000.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }

    async def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics"""
        try:
            risk_metrics = await self.financial_engine.update_risk_metrics()
            return {
                'var_99': risk_metrics.var_99,
                'expected_shortfall': risk_metrics.expected_shortfall,
                'beta': risk_metrics.beta,
                'correlation_matrix': risk_metrics.correlation_matrix,
                'stress_test_results': risk_metrics.stress_test_results
            }
        except:
            return {}

    async def _get_trading_activity(self) -> Dict[str, Any]:
        """Get trading activity data"""
        return {
            'orders_today': 0,      # TODO: implement
            'fills_today': 0,       # TODO: implement
            'active_strategies': 0, # TODO: implement
            'venue_utilization': {} # TODO: implement
        }

    async def _get_doctrine_compliance(self) -> Dict[str, Any]:
        """Get doctrine compliance data"""
        try:
            if self.doctrine_integration:
                compliance_report = await self.doctrine_integration.run_compliance_check()
                health_status = await self.doctrine_integration.get_health_status()

                return {
                    'compliance_score': compliance_report.get('compliance_score', 0),
                    'az_prime_state': compliance_report.get('az_prime_state', 'unknown'),
                    'compliant': compliance_report.get('compliant', 0),
                    'warnings': compliance_report.get('warnings', 0),
                    'violations': compliance_report.get('violations', 0),
                    'monitoring_active': health_status.get('monitoring_active', False),
                    'departments_connected': health_status.get('departments_connected', 0),
                    'last_check': compliance_report.get('generated_at')
                }
            else:
                return {
                    'compliance_score': 0,
                    'az_prime_state': 'not_initialized',
                    'compliant': 0,
                    'warnings': 0,
                    'violations': 0,
                    'monitoring_active': False,
                    'departments_connected': 0,
                    'last_check': None
                }
        except Exception as e:
            return {
                'compliance_score': 0,
                'az_prime_state': 'error',
                'compliant': 0,
                'warnings': 0,
                'violations': 0,
                'monitoring_active': False,
                'departments_connected': 0,
                'last_check': None,
                'error': str(e)
            }

    async def _get_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        alerts = []

        # Check safeguards for alerts
        safeguards_status = get_safeguards_health()
        if safeguards_status.get('overall_health') != 'healthy':
            alerts.append({
                'level': 'warning',
                'message': f"Safeguards status: {safeguards_status['overall_health']}",
                'timestamp': datetime.now()
            })

        # Check circuit breakers
        for exchange, status in safeguards_status.get('exchanges', {}).items():
            if status.get('circuit_breaker_state') == 'open':
                alerts.append({
                    'level': 'critical',
                    'message': f"Circuit breaker open for {exchange}",
                    'timestamp': datetime.now()
                })

        # Check doctrine compliance
        if self.doctrine_integration:
            try:
                compliance_report = await self.doctrine_integration.run_compliance_check()
                violations = compliance_report.get('violations', 0)
                warnings = compliance_report.get('warnings', 0)
                az_prime_state = compliance_report.get('az_prime_state', 'normal')

                if violations > 0:
                    alerts.append({
                        'level': 'critical',
                        'message': f"Doctrine violations: {violations} active",
                        'timestamp': datetime.now()
                    })

                if warnings > 0:
                    alerts.append({
                        'level': 'warning',
                        'message': f"Doctrine warnings: {warnings} active",
                        'timestamp': datetime.now()
                    })

                if az_prime_state in ['CAUTION', 'CRITICAL']:
                    alerts.append({
                        'level': 'warning' if az_prime_state == 'CAUTION' else 'critical',
                        'message': f"AZ Prime state: {az_prime_state}",
                        'timestamp': datetime.now()
                    })

            except Exception as e:
                alerts.append({
                    'level': 'warning',
                    'message': f"Doctrine monitoring error: {str(e)}",
                    'timestamp': datetime.now()
                })

        return alerts

    def display_dashboard(self, stdscr, data: Dict[str, Any]):
        """Display the monitoring dashboard"""
        if not data or 'error' in data:
            stdscr.clear()
            stdscr.addstr(0, 0, "🔄 Loading AAC 2100 Monitoring Dashboard...")
            if 'error' in data:
                stdscr.addstr(2, 0, f"Error: {data['error']}")
            stdscr.refresh()
            return

        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Header
        header = f"[DEPLOY] AAC 2100 LIVE TRADING DASHBOARD - {data['timestamp'].strftime('%H:%M:%S')}"
        stdscr.addstr(0, 0, header[:width-1], curses.A_BOLD)

        # System Health
        health = data.get('health', {})
        y_pos = 2
        stdscr.addstr(y_pos, 0, "🏥 SYSTEM HEALTH", curses.A_BOLD)
        y_pos += 1

        status_colors = {
            'healthy': curses.COLOR_GREEN,
            'warning': curses.COLOR_YELLOW,
            'critical': curses.COLOR_RED,
            'error': curses.COLOR_RED
        }

        overall_status = health.get('overall_status', 'unknown')
        color = status_colors.get(overall_status, curses.COLOR_WHITE)
        curses.init_pair(1, color, curses.COLOR_BLACK)
        stdscr.addstr(y_pos, 0, f"Overall: {overall_status.upper()}", curses.color_pair(1))
        y_pos += 2

        # Department status
        departments = health.get('departments', {})
        for dept, status in departments.items():
            dept_status = status.get('status', 'unknown')
            color = status_colors.get(dept_status, curses.COLOR_WHITE)
            curses.init_pair(2, color, curses.COLOR_BLACK)
            stdscr.addstr(y_pos, 0, f"{dept}: {dept_status.upper()}", curses.color_pair(2))
            y_pos += 1

        # P&L Section
        y_pos += 1
        stdscr.addstr(y_pos, 0, "[MONEY] P&L SUMMARY", curses.A_BOLD)
        y_pos += 1

        pnl = data.get('pnl', {})
        stdscr.addstr(y_pos, 0, f"Daily P&L: ${pnl.get('daily_pnl', 0):,.2f}")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Total Equity: ${pnl.get('total_equity', 0):,.2f}")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Unrealized P&L: ${pnl.get('unrealized_pnl', 0):,.2f}")
        y_pos += 1
        stdscr.addstr(y_pos, 0, f"Max Drawdown: {pnl.get('max_drawdown', 0):.2%}")
        y_pos += 2

        # Safeguards Status
        stdscr.addstr(y_pos, 0, "[SHIELD]️ SAFEGUARDS STATUS", curses.A_BOLD)
        y_pos += 1

        safeguards = data.get('safeguards', {})
        overall_health = safeguards.get('overall_health', 'unknown')
        color = status_colors.get(overall_health, curses.COLOR_WHITE)
        curses.init_pair(3, color, curses.COLOR_BLACK)
        stdscr.addstr(y_pos, 0, f"Overall: {overall_health.upper()}", curses.color_pair(3))
        y_pos += 1

        exchanges = safeguards.get('exchanges', {})
        for exchange, status in exchanges.items():
            cb_state = status.get('circuit_breaker_state', 'unknown')
            cb_color = curses.COLOR_RED if cb_state == 'open' else curses.COLOR_GREEN
            curses.init_pair(4, cb_color, curses.COLOR_BLACK)
            stdscr.addstr(y_pos, 0, f"{exchange}: CB={cb_state.upper()}", curses.color_pair(4))
            y_pos += 1

        # Doctrine Compliance Section
        y_pos += 1
        stdscr.addstr(y_pos, 0, "[DOCTRINE] COMPLIANCE MATRIX", curses.A_BOLD)
        y_pos += 1

        doctrine = data.get('doctrine', {})
        compliance_score = doctrine.get('compliance_score', 0)
        az_prime_state = doctrine.get('az_prime_state', 'unknown')

        # Compliance score with color coding
        score_color = curses.COLOR_GREEN if compliance_score >= 90 else curses.COLOR_YELLOW if compliance_score >= 70 else curses.COLOR_RED
        curses.init_pair(6, score_color, curses.COLOR_BLACK)
        stdscr.addstr(y_pos, 0, f"Compliance Score: {compliance_score}%", curses.color_pair(6))
        y_pos += 1

        # AZ Prime state
        az_color = curses.COLOR_RED if az_prime_state in ['CRITICAL', 'error'] else curses.COLOR_YELLOW if az_prime_state == 'CAUTION' else curses.COLOR_GREEN
        curses.init_pair(7, az_color, curses.COLOR_BLACK)
        stdscr.addstr(y_pos, 0, f"AZ Prime State: {az_prime_state.upper()}", curses.color_pair(7))
        y_pos += 1

        # Compliance metrics
        compliant = doctrine.get('compliant', 0)
        warnings = doctrine.get('warnings', 0)
        violations = doctrine.get('violations', 0)
        stdscr.addstr(y_pos, 0, f"✅ Compliant: {compliant} | ⚠️ Warnings: {warnings} | ❌ Violations: {violations}")
        y_pos += 1

        # Monitoring status
        monitoring_active = doctrine.get('monitoring_active', False)
        monitor_color = curses.COLOR_GREEN if monitoring_active else curses.COLOR_RED
        curses.init_pair(8, monitor_color, curses.COLOR_BLACK)
        stdscr.addstr(y_pos, 0, f"Monitoring: {'ACTIVE' if monitoring_active else 'INACTIVE'}", curses.color_pair(8))
        y_pos += 1

        # Alerts
        alerts = data.get('alerts', [])
        if alerts:
            y_pos += 1
            stdscr.addstr(y_pos, 0, "[ALERT] ACTIVE ALERTS", curses.A_BOLD | curses.A_BLINK)
            y_pos += 1

            for alert in alerts[:3]:  # Show top 3 alerts
                alert_color = curses.COLOR_RED if alert['level'] == 'critical' else curses.COLOR_YELLOW
                curses.init_pair(5, alert_color, curses.COLOR_BLACK)
                stdscr.addstr(y_pos, 0, f"• {alert['message']}", curses.color_pair(5))
                y_pos += 1

        # Footer
        footer_y = height - 2
        stdscr.addstr(footer_y, 0, "Press 'q' to quit | 'r' to refresh | Auto-refresh: 1s")
        stdscr.addstr(footer_y + 1, 0, f"Last update: {data['timestamp'].strftime('%H:%M:%S')}")

        stdscr.refresh()

    async def run_dashboard(self):
        """Run the monitoring dashboard"""
        if CURSES_AVAILABLE and platform.system() != 'Windows':
            # Use curses on Unix-like systems
            def dashboard_thread():
                curses.wrapper(self._run_curses_dashboard)

            # Start dashboard in separate thread
            dashboard_thread = threading.Thread(target=dashboard_thread, daemon=True)
            dashboard_thread.start()
        else:
            # Use text-based dashboard on Windows or when curses unavailable
            def dashboard_thread():
                self._run_text_dashboard()

            # Start dashboard in separate thread
            dashboard_thread = threading.Thread(target=dashboard_thread, daemon=True)
            dashboard_thread.start()

        # Main monitoring loop
        self.running = True
        while self.running:
            try:
                # Collect data
                data = await self.collect_monitoring_data()
                self.last_update = datetime.now()

                # Update shared data for dashboard
                self._latest_data = data

                # Wait before next update
                await asyncio.sleep(self.refresh_rate)

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(5)

    def _run_curses_dashboard(self, stdscr):
        """Run the curses-based dashboard"""
        # Setup curses
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.use_default_colors()
        stdscr.timeout(1000)  # Refresh every second

        while self.running:
            try:
                # Get latest data
                data = getattr(self, '_latest_data', {})

                # Display dashboard
                self.display_dashboard(stdscr, data)

                # Handle input
                key = stdscr.getch()
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('r'):
                    # Force refresh
                    pass

            except Exception as e:
                stdscr.clear()
                stdscr.addstr(0, 0, f"Dashboard error: {e}")
                stdscr.refresh()
                time.sleep(2)

    def _run_text_dashboard(self):
        """Run the text-based dashboard for Windows compatibility"""
        print("\n" + "="*80)
        print("AAC 2100 Real-Time Monitoring Dashboard (Text Mode)")
        print("="*80)
        print("Press Ctrl+C to quit | Auto-refresh: 1s")
        print()

        while self.running:
            try:
                # Get latest data
                data = getattr(self, '_latest_data', {})

                if data:
                    # Clear screen (simple approach)
                    print("\033[2J\033[H", end="")  # ANSI clear screen

                    # Display dashboard
                    self._display_text_dashboard(data)

                # Wait before next update
                time.sleep(1)

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"Dashboard error: {e}")
                time.sleep(2)

    def _display_text_dashboard(self, data: Dict[str, Any]):
        """Display dashboard in text mode"""
        print("="*80)
        print("AAC 2100 Real-Time Monitoring Dashboard")
        print("="*80)
        print(f"Last update: {data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # System Health
        health = data.get('health', {})
        print("🔍 SYSTEM HEALTH")
        print("-" * 20)
        status = health.get('overall_status', 'unknown')
        status_emoji = {'healthy': '🟢', 'warning': '🟡', 'critical': '🔴', 'unknown': '⚪'}.get(status, '⚪')
        print(f"Overall Status: {status_emoji} {status.upper()}")

        # Department status
        departments = health.get('departments', {})
        if departments:
            print("\nDepartments:")
            for dept, info in departments.items():
                dept_status = info.get('status', 'unknown')
                emoji = {'healthy': '🟢', 'warning': '🟡', 'critical': '🔴', 'unknown': '⚪'}.get(dept_status, '⚪')
                print(f"  {emoji} {dept}: {dept_status}")

        # Infrastructure
        infra = health.get('infrastructure', {})
        if infra:
            print("\nInfrastructure:")
            for component, status in infra.items():
                if isinstance(status, dict):
                    comp_status = status.get('status', 'unknown')
                else:
                    comp_status = 'healthy' if status else 'critical'
                emoji = {'healthy': '🟢', 'warning': '🟡', 'critical': '🔴', 'unknown': '⚪'}.get(comp_status, '⚪')
                print(f"  {emoji} {component}: {comp_status}")

        # P&L Data
        pnl = data.get('pnl', {})
        if pnl:
            print("\n[MONEY] P&L SUMMARY")
            print("-" * 15)
            daily_pnl = pnl.get('daily_pnl', 0)
            total_pnl = pnl.get('total_pnl', 0)
            print(f"Daily P&L: ${daily_pnl:,.2f}")
            print(f"Total P&L: ${total_pnl:,.2f}")

        # Risk Metrics
        risk = data.get('risk', {})
        if risk:
            print("\n[WARN]️  RISK METRICS")
            print("-" * 15)
            var_95 = risk.get('var_95', 0)
            max_drawdown = risk.get('max_drawdown', 0)
            print(f"VaR (95%): ${var_95:,.2f}")
            print(f"Max Drawdown: ${max_drawdown:,.2f}")

        # Trading Activity
        trading = data.get('trading', {})
        if trading:
            print("\n📈 TRADING ACTIVITY")
            print("-" * 20)
            active_positions = trading.get('active_positions', 0)
            pending_orders = trading.get('pending_orders', 0)
            print(f"Active Positions: {active_positions}")
            print(f"Pending Orders: {pending_orders}")

        # Safeguards
        safeguards = data.get('safeguards', {})
        if safeguards:
            print("\n[SHIELD]️  PRODUCTION SAFEGUARDS")
            print("-" * 25)
            circuit_breaker = safeguards.get('circuit_breaker_active', False)
            rate_limited = safeguards.get('rate_limited', False)
            print(f"Circuit Breaker: {'🔴 ACTIVE' if circuit_breaker else '🟢 NORMAL'}")
            print(f"Rate Limiting: {'🟡 ACTIVE' if rate_limited else '🟢 NORMAL'}")

        # Doctrine Compliance
        doctrine = data.get('doctrine', {})
        if doctrine:
            print("\n[DOCTRINE] COMPLIANCE MATRIX")
            print("-" * 25)
            compliance_score = doctrine.get('compliance_score', 0)
            az_prime_state = doctrine.get('az_prime_state', 'unknown')
            
            # Compliance score with color emoji
            if compliance_score >= 90:
                score_emoji = '🟢'
            elif compliance_score >= 70:
                score_emoji = '🟡'
            else:
                score_emoji = '🔴'
            print(f"Compliance Score: {score_emoji} {compliance_score}%")
            
            # AZ Prime state
            if az_prime_state in ['CRITICAL', 'error']:
                az_emoji = '🔴'
            elif az_prime_state == 'CAUTION':
                az_emoji = '🟡'
            else:
                az_emoji = '🟢'
            print(f"AZ Prime State: {az_emoji} {az_prime_state.upper()}")
            
            # Compliance metrics
            compliant = doctrine.get('compliant', 0)
            warnings = doctrine.get('warnings', 0)
            violations = doctrine.get('violations', 0)
            print(f"✅ Compliant: {compliant} | ⚠️ Warnings: {warnings} | ❌ Violations: {violations}")
            
            # Monitoring status
            monitoring_active = doctrine.get('monitoring_active', False)
            monitor_emoji = '🟢' if monitoring_active else '🔴'
            print(f"Monitoring: {monitor_emoji} {'ACTIVE' if monitoring_active else 'INACTIVE'}")

        # Alerts
        alerts = data.get('alerts', [])
        if alerts:
            print("\n[ALERT] ACTIVE ALERTS")
            print("-" * 15)
            for alert in alerts[:5]:  # Show first 5 alerts
                severity = alert.get('severity', 'info')
                emoji = {'critical': '🔴', 'warning': '🟡', 'info': 'ℹ️'}.get(severity, 'ℹ️')
                print(f"  {emoji} {alert.get('title', 'Unknown alert')}")

        print("\n" + "="*80)
        print("Press Ctrl+C to quit | Auto-refresh: 1s")
        print("="*80)

    async def start_continuous_monitoring(self):
        """Start continuous monitoring system"""
        print("🔄 Starting AAC 2100 Continuous Monitoring...")

        await self.initialize()
        await self.run_dashboard()

        print("✅ Continuous monitoring started")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        print("🛑 Monitoring stopped")


async def main():
    """Main entry point"""
    dashboard = AACMonitoringDashboard()

    try:
        await dashboard.start_continuous_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down monitoring dashboard...")
        dashboard.stop_monitoring()
    except Exception as e:
        print(f"[CROSS] Monitoring dashboard failed: {e}")
        dashboard.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())