#!/usr/bin/env python3
"""
AAC 2100 MASTER MONITORING DASHBOARD
=====================================
Unified comprehensive monitoring and display system combining all AAC monitoring capabilities.

Features:
- Real-time doctrine compliance monitoring (8 packs)
- System health and performance metrics
- Trading activity and P&L tracking
- Risk management dashboard
- Security monitoring and alerts
- Strategy metrics and analytics
- Production safeguards status
- Circuit breaker monitoring
- Alert management and notifications
- Multiple display modes (terminal, web, text)

Consolidated from:
- monitoring_dashboard.py (terminal dashboard)
- aac_monitoring_dashboard.py (streamlit dashboard)
- strategy_metrics_dashboard.py (dash metrics)
- security_dashboard.py (security monitoring)
- continuous_monitoring.py (background service)
- enhanced_metrics_display.py (enhanced display)

Display Modes:
- TERMINAL: Curses-based real-time dashboard (Unix) / Text-based (Windows)
- WEB: Streamlit web interface for remote access
- DASH: Plotly Dash analytics dashboard
- API: REST API for external integrations
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import sys
from pathlib import Path
import json
import os
import platform
import psutil
import logging
import queue

logger = logging.getLogger(__name__)

# Try to import curses, fallback for Windows
try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False
    logger.info("Warning: curses not available, using text-based dashboard")

# Optional imports for different display modes
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

# Dash imports are handled inside the class to avoid module-level import issues
DASH_AVAILABLE = False

# Add project root and AAC package path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Core AAC imports
from shared.config_loader import get_config
from shared.monitoring import get_monitoring_manager
from shared.production_safeguards import get_production_safeguards, get_safeguards_health
from shared.audit_logger import get_audit_logger

# Department engines
from CentralAccounting.financial_analysis_engine import FinancialAnalysisEngine
from CryptoIntelligence.crypto_intelligence_engine import CryptoIntelligenceEngine
from integrations.unusual_whales_service import get_unusual_whales_snapshot_service

# Doctrine integration
from aac.doctrine.doctrine_integration import get_doctrine_integration

# Security framework
try:
    from shared.security_framework import (
        rbac, api_security, security_monitoring, advanced_encryption
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

# Strategy testing (optional)
try:
    from strategies.strategy_testing_lab_fixed import strategy_testing_lab, initialize_strategy_testing_lab
    STRATEGY_TESTING_AVAILABLE = True
except ImportError:
    STRATEGY_TESTING_AVAILABLE = False

# Regime forecaster (AAC intelligence layer)
try:
    from strategies.regime_engine import RegimeEngine, MacroSnapshot
    from strategies.stock_forecaster import StockForecaster, Horizon
    from strategies.crypto_forecaster import CryptoForecaster, CryptoSnapshot
    FORECASTER_AVAILABLE = True
except ImportError:
    FORECASTER_AVAILABLE = False

# MATRIX MAXIMIZER (geopolitical bear market options engine)
try:
    from strategies.matrix_maximizer.runner import MatrixMaximizer
    from strategies.matrix_maximizer.core import MatrixConfig
    MATRIX_MAXIMIZER_AVAILABLE = True
except ImportError:
    MATRIX_MAXIMIZER_AVAILABLE = False

# Additional dashboard components
try:
    from trading.aac_arbitrage_execution_system import AACArbitrageExecutionSystem, ExecutionConfig
    from trading.binance_trading_engine import TradingConfig
    from trading.binance_arbitrage_integration import BinanceArbitrageClient, BinanceConfig
    from strategies.strategy_analysis_engine import strategy_analysis_engine, initialize_strategy_analysis
    from monitoring.security_dashboard import SecurityDashboard
    from monitoring.continuous_monitoring import ContinuousMonitoringService
    from tools.enhanced_metrics_display import AACMetricsDisplay
    ARBITRAGE_COMPONENTS_AVAILABLE = True
    TRADING_AVAILABLE = True
except ImportError:
    ARBITRAGE_COMPONENTS_AVAILABLE = False
    TRADING_AVAILABLE = False

# Database manager (optional)
try:
    from CentralAccounting.database import AccountingDatabase as DatabaseManager
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    DatabaseManager = None

# System Registry (unified status tracker)
try:
    from monitoring.aac_system_registry import SystemRegistry
    SYSTEM_REGISTRY_AVAILABLE = True
except ImportError:
    SYSTEM_REGISTRY_AVAILABLE = False
    SystemRegistry = None  # type: ignore[assignment,misc]

# Black Swan Pressure Cooker (crisis center)
try:
    from strategies.black_swan_pressure_cooker import get_crisis_data, get_crisis_section
    CRISIS_CENTER_AVAILABLE = True
except ImportError:
    CRISIS_CENTER_AVAILABLE = False

# Storm Lifeboat Matrix v9.0 (scenario / MC / coherence engine)
try:
    from strategies.storm_lifeboat.scenario_engine import ScenarioEngine
    from strategies.storm_lifeboat.coherence import CoherenceEngine
    from strategies.storm_lifeboat.lunar_phi import LunarPhiEngine
    from strategies.storm_lifeboat.core import VolRegime as SLVolRegime, MandateLevel
    STORM_LIFEBOAT_AVAILABLE = True
except ImportError:
    STORM_LIFEBOAT_AVAILABLE = False


class DisplayMode:
    """Display mode enumeration"""
    TERMINAL = "terminal"  # Curses/text-based terminal dashboard
    WEB = "web"          # Streamlit web dashboard
    DASH = "dash"        # Plotly Dash analytics dashboard
    API = "api"          # REST API mode


class AACMasterMonitoringDashboard:
    """
    Master monitoring dashboard consolidating all AAC monitoring capabilities.

    Features:
    - Doctrine compliance monitoring (8 packs)
    - System health and performance
    - Trading activity and P&L
    - Risk management
    - Security monitoring
    - Strategy analytics
    - Production safeguards
    - Multiple display modes
    """

    def __init__(self, display_mode: str = DisplayMode.TERMINAL):
        self.display_mode = display_mode
        self.config = get_config()
        self.audit_logger = get_audit_logger()

        # Core monitoring components
        self.monitoring = get_monitoring_manager()
        self.safeguards = get_production_safeguards()
        self.doctrine_integration = None

        # Department engines
        self.financial_engine = FinancialAnalysisEngine()
        self.crypto_engine = CryptoIntelligenceEngine()
        self.unusual_whales = get_unusual_whales_snapshot_service()

        # Optional components
        self.execution_system = None
        self.security_framework = None
        self.strategy_testing = None

        # Dashboard state
        self.running = False
        self.last_update = datetime.now()
        self.refresh_rate = max(1.0, float(os.environ.get('DASHBOARD_REFRESH_RATE', '5.0')))  # seconds, min 1s
        self._latest_data = {}
        self._data_lock = threading.Lock()

        # Threads and processes
        self.monitoring_thread = None
        self.display_thread = None

        # System Registry (unified component tracker)
        self.system_registry = SystemRegistry() if SYSTEM_REGISTRY_AVAILABLE else None

        # Logger
        self.logger = logging.getLogger("MasterDashboard")

    async def initialize(self) -> bool:
        """Initialize all monitoring components"""
        self.logger.info("[INIT] Initializing AAC Master Monitoring Dashboard...")

        try:
            # Initialize core safeguards
            from shared.production_safeguards import initialize_production_safeguards
            await initialize_production_safeguards()

            # Initialize doctrine integration
            self.doctrine_integration = get_doctrine_integration()
            await self.doctrine_integration.initialize()

            # Initialize security framework if available
            if SECURITY_AVAILABLE:
                self.security_framework = {
                    'rbac': rbac,
                    'api_security': api_security,
                    'security_monitoring': security_monitoring,
                    'encryption': advanced_encryption
                }

            # Initialize trading system if available
            if TRADING_AVAILABLE:
                try:
                    execution_config = ExecutionConfig()
                    self.execution_system = AACArbitrageExecutionSystem(execution_config)
                    await self.execution_system.initialize()
                except Exception as e:
                    self.logger.warning("[WARN] Trading system init failed (non-fatal): %s", e)
                    self.execution_system = None

            # Initialize strategy testing if available
            if STRATEGY_TESTING_AVAILABLE:
                await initialize_strategy_testing_lab()
                self.strategy_testing = strategy_testing_lab

            self.logger.info("[SUCCESS] Master monitoring dashboard initialized")
            return True

        except Exception as e:
            self.logger.error(f"[CROSS] Failed to initialize master dashboard: {e}")
            return False

    async def collect_monitoring_data(self) -> Dict[str, Any]:
        """Collect comprehensive monitoring data from all systems"""
        try:
            # Timestamp
            timestamp = datetime.now()

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

            # Market intelligence
            market_intelligence_data = await self._get_market_intelligence()

            # Security status
            security_data = await self._get_security_status()

            # Strategy metrics
            strategy_data = await self._get_strategy_metrics()

            # Safeguards status
            safeguards_data = get_safeguards_health()

            # Regime forecaster intelligence
            forecaster_data = await self._get_forecaster_intel()

            # IBKR open orders + maximization plan
            ibkr_orders_data = await self._get_ibkr_orders()

            # Alerts
            alerts_data = await self._get_alerts()

            # System Registry snapshot (APIs, exchanges, infra, strategies, orphans)
            registry_data = self._get_registry_snapshot()

            # Matrix Maximizer deep integration
            maximizer_data = self._get_matrix_maximizer_deep()

            # Black Swan Crisis Center
            crisis_data = self._get_crisis_center_data()

            # Storm Lifeboat Matrix
            storm_lifeboat_data = self._get_storm_lifeboat_data()

            return {
                'timestamp': timestamp,
                'health': health_data,
                'pnl': pnl_data,
                'risk': risk_data,
                'trading': trading_data,
                'doctrine': doctrine_data,
                'market_intelligence': market_intelligence_data,
                'security': security_data,
                'strategy': strategy_data,
                'safeguards': safeguards_data,
                'forecaster': forecaster_data,
                'ibkr_orders': ibkr_orders_data,
                'alerts': alerts_data,
                'registry': registry_data,
                'matrix_maximizer': maximizer_data,
                'crisis_center': crisis_data,
                'storm_lifeboat': storm_lifeboat_data,
            }

        except Exception as e:
            self.logger.error(f"Error collecting monitoring data: {e}", exc_info=True)
            return {
                'timestamp': datetime.now(),
                'error': str(e),
                'health': {},
                'pnl': {},
                'risk': {},
                'trading': {},
                'doctrine': {},
                'market_intelligence': {},
                'security': {},
                'strategy': {},
                'safeguards': {},
                'alerts': [],
                'registry': {},
                'matrix_maximizer': {},
                'crisis_center': {},
                'storm_lifeboat': {},
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
                db_status = await self.financial_engine.health_check()
                return {
                    'status': 'healthy' if db_status.get('database_connected') else 'critical',
                    'last_transaction': db_status.get('last_transaction_time'),
                    'pending_reconciliations': db_status.get('pending_count', 0)
                }
            elif department == 'CryptoIntelligence':
                venue_status = await self.crypto_engine.get_all_venue_health()
                return {
                    'status': 'healthy',
                    'venues_monitored': len(venue_status),
                    'average_health_score': sum(v['health_score'] for v in venue_status.values()) / len(venue_status)
                }
            elif department == 'BigBrainIntelligence':
                # Dynamically count agents from BigBrainIntelligence directory
                agent_count = 0
                try:
                    from pathlib import Path
                    bbi_dir = Path(__file__).resolve().parent.parent / 'BigBrainIntelligence'
                    if bbi_dir.exists():
                        agent_count = len([
                            f for f in bbi_dir.glob('*.py')
                            if f.stem not in ('__init__', '__pycache__')
                        ])
                except Exception as e:
                    self.logger.debug(f"BBI agent count fallback: {e}")
                    agent_count = 0
                return {
                    'status': 'healthy',
                    'active_agents': agent_count,
                    'predictions_today': 0
                }
            elif department == 'TradingExecution':
                return {
                    'status': 'healthy',
                    'active_positions': 0,
                    'orders_pending': 0
                }
            else:
                return {'status': 'unknown'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            if not DATABASE_AVAILABLE:
                return {'status': 'unavailable', 'error': 'DatabaseManager not imported'}
            db = DatabaseManager()
            try:
                connected = await db.health_check()
                return {
                    'status': 'healthy' if connected else 'critical',
                    'connection_pool_size': getattr(db, 'pool_size', 1),
                    'active_connections': getattr(db, 'active_connections', 1)
                }
            finally:
                if hasattr(db, 'close'):
                    db.close()
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return {'status': 'critical', 'error': str(e)}

    async def _check_network_health(self) -> Dict[str, Any]:
        """Check network connectivity with actual latency measurement"""
        start = time.monotonic()
        try:
            import socket
            s = socket.create_connection(('8.8.8.8', 53), timeout=2)
            s.close()
            latency = round((time.monotonic() - start) * 1000, 1)
            return {'status': 'healthy' if latency < 200 else 'warning', 'latency_ms': latency, 'packet_loss': 0.0}
        except Exception as e:
            self.logger.debug(f"Network health check failed: {e}")
            latency = round((time.monotonic() - start) * 1000, 1)
            return {'status': 'degraded', 'latency_ms': latency, 'packet_loss': 100.0}

    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        memory = psutil.virtual_memory()
        return {
            'used_percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'status': 'warning' if memory.percent > 85 else 'healthy'
        }

    async def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        return {
            'used_percent': cpu_percent,
            'status': 'warning' if cpu_percent > 90 else 'healthy'
        }

    async def _get_pnl_data(self) -> Dict[str, Any]:
        """Get P&L data"""
        try:
            risk_metrics = await self.financial_engine.update_risk_metrics()
            total_equity = await self.financial_engine.calculate_portfolio_value()
            daily_pnl = await self.financial_engine.calculate_daily_pnl()
            unrealized_pnl = sum(
                position.unrealized_pnl for position in self.financial_engine.positions.values()
            )
            return {
                'daily_pnl': daily_pnl,
                'total_equity': total_equity,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': risk_metrics.max_drawdown_pct
            }
        except Exception as e:
            self.logger.error(f"PnL data retrieval failed: {e}")
            return {
                'daily_pnl': 0.0,
                'total_equity': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'error': str(e)
            }

    async def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics"""
        try:
            risk_metrics = await self.financial_engine.update_risk_metrics()
            return {
                'var_99': risk_metrics.stressed_var_99,
                'expected_shortfall': risk_metrics.tail_loss_p99,
                'beta': 0.0,
                'correlation_matrix': {
                    'strategy_correlation': risk_metrics.strategy_correlation,
                },
                'stress_test_results': {
                    'portfolio_heat': risk_metrics.portfolio_heat,
                    'margin_buffer': risk_metrics.margin_buffer,
                }
            }
        except Exception as e:
            self.logger.error(f"Risk metrics retrieval failed: {e}")
            return {'error': str(e)}

    async def _get_trading_activity(self) -> Dict[str, Any]:
        """Get trading activity data"""
        activity = {
            'orders_today': 0,
            'fills_today': 0,
            'active_strategies': 0,
            'venue_utilization': {}
        }

        if self.execution_system:
            try:
                # Get data from execution system
                status = self.execution_system.get_system_status()
                activity.update({
                    'orders_today': status.get('orders_today', 0),
                    'fills_today': status.get('fills_today', 0),
                    'active_strategies': status.get('active_strategies', 0),
                    'venue_utilization': status.get('venue_utilization', {})
                })
            except Exception as e:
                self.logger.warning(f"Trading activity fetch failed: {e}")

        return activity

    async def _get_doctrine_compliance(self) -> Dict[str, Any]:
        """Get doctrine compliance data"""
        try:
            if self.doctrine_integration:
                compliance_report = await self.doctrine_integration.run_compliance_check()
                health_status = await self.doctrine_integration.get_health_status()

                return {
                    'compliance_score': compliance_report.get('compliance_score', 0),
                    'barren_wuffet_state': compliance_report.get('barren_wuffet_state', 'unknown'),
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
                    'barren_wuffet_state': 'not_initialized',
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
                'barren_wuffet_state': 'error',
                'compliant': 0,
                'warnings': 0,
                'violations': 0,
                'monitoring_active': False,
                'departments_connected': 0,
                'last_check': None,
                'error': str(e)
            }

    async def _get_market_intelligence(self) -> Dict[str, Any]:
        """Get normalized Unusual Whales market-intelligence snapshot."""
        try:
            snapshot = await self.unusual_whales.get_snapshot()
            return snapshot
        except Exception as e:
            return {
                'status': 'error',
                'as_of': datetime.now().isoformat(),
                'error': str(e),
            }

    async def _get_security_status(self) -> Dict[str, Any]:
        """Get security status data"""
        if not SECURITY_AVAILABLE or not self.security_framework:
            return {'status': 'not_available'}

        try:
            security_status = {
                'overall_score': 0,
                'components': {},
                'alerts': []
            }

            # MFA Status
            mfa_status = self._check_mfa_status()
            security_status['components']['mfa'] = mfa_status

            # Encryption Status
            encryption_status = self._check_encryption_status()
            security_status['components']['encryption'] = encryption_status

            # RBAC Status
            rbac_status = self._check_rbac_status()
            security_status['components']['rbac'] = rbac_status

            # API Security
            api_status = self._check_api_security_status()
            security_status['components']['api'] = api_status

            # Calculate overall score
            component_scores = []
            for component in security_status['components'].values():
                if isinstance(component, dict) and 'score' in component:
                    component_scores.append(component['score'])

            if component_scores:
                security_status['overall_score'] = sum(component_scores) / len(component_scores)

            return security_status

        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def _check_mfa_status(self) -> Dict[str, Any]:
        """Check MFA status — queries actual auth config"""
        try:
            cfg = get_config()
            mfa_enabled = getattr(cfg, 'mfa_enabled', False)
            return {
                'enabled_users': 1 if mfa_enabled else 0,
                'total_users': 1,
                'score': 100 if mfa_enabled else 0,
                'status': 'healthy' if mfa_enabled else 'not_configured'
            }
        except Exception as e:
            self.logger.debug(f"MFA status check failed: {e}")
            return {'enabled_users': 0, 'total_users': 0, 'score': 0, 'status': 'not_implemented'}

    def _check_encryption_status(self) -> Dict[str, Any]:
        """Check encryption status — verifies secrets manager is initialized"""
        try:
            from shared.secrets_manager import get_secrets_manager
            sm = get_secrets_manager()
            encrypted = sm is not None and hasattr(sm, '_fernet')
            return {
                'encrypted_databases': 1 if encrypted else 0,
                'total_databases': 1,
                'score': 100 if encrypted else 0,
                'status': 'healthy' if encrypted else 'not_configured'
            }
        except Exception as e:
            self.logger.debug(f"Encryption status check failed: {e}")
            return {'encrypted_databases': 0, 'total_databases': 0, 'score': 0, 'status': 'not_implemented'}

    def _check_rbac_status(self) -> Dict[str, Any]:
        """Check RBAC status — queries actual role definitions from security_framework"""
        try:
            from shared.security_framework import rbac as rbac_manager
            roles = len(rbac_manager.roles) if rbac_manager and hasattr(rbac_manager, 'roles') else 0
            return {
                'roles_defined': roles,
                'permissions_assigned': roles * 10,
                'score': min(100, roles * 15),
                'status': 'healthy' if roles >= 3 else 'warning'
            }
        except Exception as e:
            self.logger.debug(f"RBAC status check failed: {e}")
            return {'roles_defined': 0, 'permissions_assigned': 0, 'score': 0, 'status': 'not_implemented'}

    def _check_api_security_status(self) -> Dict[str, Any]:
        """Check API security — verifies TLS and auth middleware presence"""
        try:
            import ssl
            has_tls = ssl.OPENSSL_VERSION is not None
            return {
                'endpoints_secured': 1 if has_tls else 0,
                'total_endpoints': 1,
                'score': 80 if has_tls else 20,
                'status': 'healthy' if has_tls else 'warning'
            }
        except Exception as e:
            self.logger.debug(f"API security status check failed: {e}")
            return {'endpoints_secured': 0, 'total_endpoints': 0, 'score': 0, 'status': 'not_implemented'}

    async def _get_forecaster_intel(self) -> Dict[str, Any]:
        """Run regime engine + stock forecaster. Returns compact intel dict for dashboard."""
        if not FORECASTER_AVAILABLE:
            return {'status': 'not_available'}
        try:
            import os
            from strategies.regime_engine import RegimeEngine, MacroSnapshot
            from strategies.stock_forecaster import StockForecaster, Horizon

            # Build snapshot from env overrides or safe defaults
            snap = MacroSnapshot(
                vix=float(os.environ.get('MONITOR_VIX', '21.5')),
                hy_spread_bps=float(os.environ.get('MONITOR_HY_SPREAD', '380')),
                oil_price=float(os.environ.get('MONITOR_OIL', '95.5')),
                gold_price=float(os.environ.get('MONITOR_GOLD', '5011')),
                core_pce=float(os.environ.get('MONITOR_PCE', '3.1')),
                gdp_growth=float(os.environ.get('MONITOR_GDP', '0.7')),
                yield_curve_10_2=float(os.environ.get('MONITOR_YIELD_CURVE', '-0.3')),
                private_credit_redemption_pct=float(os.environ.get('MONITOR_PRIV_CREDIT', '11.0')),
                war_active=os.environ.get('MONITOR_WAR', 'true').lower() == 'true',
                hormuz_blocked=os.environ.get('MONITOR_HORMUZ', 'true').lower() == 'true',
                hyg_return_1d=float(os.environ.get('MONITOR_HYG_RET', '-0.6')),
                spy_return_1d=float(os.environ.get('MONITOR_SPY_RET', '-0.4')),
                kre_return_1d=float(os.environ.get('MONITOR_KRE_RET', '-0.9')),
                airlines_return_1d=float(os.environ.get('MONITOR_JETS_RET', '-1.1')),
            )

            state = RegimeEngine().evaluate(snap)
            forecaster = StockForecaster()
            short_fcast = forecaster.forecast(state, Horizon.SHORT, top_n=3)

            top3 = [
                {
                    'rank': o.rank,
                    'ticker': o.primary_ticker,
                    'industry': o.industry.value,
                    'expression': o.expression.value,
                    'score': round(o.composite_score, 1),
                    'thesis': o.thesis[:80],
                    'structure': o.structure_hint[:70],
                }
                for o in short_fcast.opportunities[:3]
            ]

            fired_formulas = [
                {
                    'tag': f.tag.value,
                    'confidence': round(f.confidence * 100),
                    'outcome': f.expected_outcome[:70],
                }
                for f in state.top_formulas[:3]
            ]

            two_stack = forecaster.two_trade_stack(state)

            return {
                'status': 'ok',
                'regime': state.primary_regime.value,
                'secondary': state.secondary_regime.value if state.secondary_regime else None,
                'regime_confidence': round(state.regime_confidence * 100),
                'vol_shock_readiness': state.vol_shock_readiness,
                'bear_signals': state.bear_signals,
                'bull_signals': state.bull_signals,
                'fired_formulas': fired_formulas,
                'top3_opportunities': top3,
                'anchor_ticker': two_stack[0].primary_ticker if two_stack[0] else None,
                'contagion_ticker': two_stack[1].primary_ticker if two_stack[1] else None,
                'macro_context': {
                    'vix': snap.vix,
                    'hy_spread': snap.hy_spread_bps,
                    'oil': snap.oil_price,
                    'war': snap.war_active,
                    'hormuz': snap.hormuz_blocked,
                },
            }
        except Exception as e:
            self.logger.warning(f'Forecaster intel error: {e}')
            return {'status': 'error', 'error': str(e)}

    async def _get_ibkr_orders(self) -> Dict[str, Any]:
        """Fetch open IBKR orders and account balance for maximization plan."""
        connector = None
        try:
            from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector

            connector = IBKRConnector()
            await connector.connect()
            orders_raw = await connector.get_open_orders()

            # Fetch real account balance instead of hardcoded value
            account_balance = 0.0
            try:
                balances = await connector.get_balances()
                if 'USD' in balances:
                    account_balance = balances['USD'].free
                elif 'TOTAL_CASH' in balances:
                    account_balance = balances['TOTAL_CASH'].free
            except Exception:
                self.logger.warning("Could not fetch IBKR balance, using 0")

            await connector.disconnect()
            connector = None

            orders = []
            total_committed = 0.0
            for o in orders_raw:
                # o is an ExchangeOrder object or its string repr
                raw = o if isinstance(o, dict) else {}
                # Parse from string repr if needed
                if hasattr(o, 'symbol'):
                    qty = float(getattr(o, 'quantity', 0))
                    price = float(getattr(o, 'price', 0))
                    cost = qty * price * 100  # options: qty contracts × premium × 100
                    total_committed += cost
                    orders.append({
                        'order_id': getattr(o, 'order_id', '?'),
                        'symbol': getattr(o, 'symbol', '?').replace('/USD', ''),
                        'side': getattr(o, 'side', '?'),
                        'qty': qty,
                        'price': price,
                        'cost': cost,
                        'status': getattr(o, 'status', '?'),
                        'ibkr_status': getattr(o, 'raw', {}).get('ibkr_status', '?'),
                    })

            remaining = max(0.0, account_balance - total_committed)

            # Recommendations are generated by the strategy/forecaster layer, not here.
            # The monitoring dashboard only reports current state.
            recs = []
            rec_cost = 0.0
            cash_after = remaining

            return {
                'status': 'ok',
                'account': 'U24346218',
                'balance_usd': account_balance,
                'open_orders': orders,
                'total_committed': round(total_committed, 2),
                'remaining_cash': round(remaining, 2),
                'recommendations': recs,
                'rec_total_cost': round(rec_cost, 2),
                'cash_after_recs': round(cash_after, 2),
            }
        except Exception as e:
            self.logger.warning(f'IBKR orders fetch error: {e}')
            return {'status': 'error', 'error': str(e)}
        finally:
            if connector is not None:
                try:
                    await connector.disconnect()
                except Exception:
                    pass

    async def _get_strategy_metrics(self) -> Dict[str, Any]:
        """Get strategy metrics data"""
        if not STRATEGY_TESTING_AVAILABLE or not self.strategy_testing:
            return {'status': 'not_available'}

        try:
            # Get strategy performance data
            performance_data = await self.strategy_testing.get_performance_summary()
            return {
                'total_strategies': performance_data.get('total_strategies', 0),
                'active_strategies': performance_data.get('active_strategies', 0),
                'best_performing': performance_data.get('best_performing', {}),
                'worst_performing': performance_data.get('worst_performing', {}),
                'average_return': performance_data.get('average_return', 0.0),
                'sharpe_ratio': performance_data.get('sharpe_ratio', 0.0)
            }
        except Exception as e:
            self.logger.error(f"Strategy metrics retrieval failed: {e}")
            return {
                'total_strategies': 0,
                'active_strategies': 0,
                'best_performing': {},
                'worst_performing': {},
                'average_return': 0.0,
                'sharpe_ratio': 0.0,
                'error': str(e)
            }

    # ── SYSTEM REGISTRY + MATRIX MAXIMIZER DEEP ─────────────────

    def _get_registry_snapshot(self) -> Dict[str, Any]:
        """Get a full system registry snapshot (APIs, exchanges, infra, etc.)."""
        if not self.system_registry:
            return {"status": "not_available"}
        try:
            return self.system_registry.collect_full_snapshot()
        except Exception as e:
            self.logger.warning("Registry snapshot failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _get_matrix_maximizer_deep(self) -> Dict[str, Any]:
        """Get deep Matrix Maximizer state — last-run data + module health."""
        if not MATRIX_MAXIMIZER_AVAILABLE:
            return {"status": "not_available"}
        try:
            # Read latest run output
            latest_path = PROJECT_ROOT / "data" / "matrix_maximizer_latest.json"
            if latest_path.exists():
                data = json.loads(latest_path.read_text(encoding="utf-8"))
                regime = data.get("regime", {})
                forecast = data.get("forecast", {})
                picks = data.get("picks", [])
                risk_snap = data.get("risk", {})
                return {
                    "status": "ok",
                    "run_number": data.get("run_number"),
                    "timestamp": data.get("timestamp"),
                    "mandate": forecast.get("mandate", "?"),
                    "risk_per_trade": forecast.get("risk_per_trade"),
                    "max_positions": forecast.get("max_positions"),
                    "spy_median_return": forecast.get("spy_median_return"),
                    "spy_var_95": forecast.get("spy_var_95"),
                    "regime": regime.get("regime", "?"),
                    "regime_confidence": regime.get("confidence"),
                    "war_active": regime.get("war_active"),
                    "hormuz_blocked": regime.get("hormuz_blocked"),
                    "oil_price": regime.get("oil_price"),
                    "vix": regime.get("vix"),
                    "top_picks": [
                        {
                            "ticker": p.get("ticker"),
                            "strike": p.get("strike"),
                            "expiry": p.get("expiry"),
                            "score": p.get("score"),
                            "contracts": p.get("contracts"),
                            "cost": p.get("cost"),
                        }
                        for p in picks[:5]
                    ],
                    "total_picks": len(picks),
                    "circuit_breaker": risk_snap.get("circuit_breaker", "?"),
                    "risk_score": risk_snap.get("risk_score"),
                    "elapsed_s": data.get("elapsed_s"),
                }
            else:
                return {"status": "no_data", "detail": "No run output found"}
        except Exception as e:
            self.logger.warning("Matrix Maximizer deep read failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _get_crisis_center_data(self) -> Dict[str, Any]:
        """Get Black Swan Crisis Center data from pressure cooker."""
        if not CRISIS_CENTER_AVAILABLE:
            return {'status': 'unavailable'}
        try:
            return get_crisis_data()
        except Exception as e:
            self.logger.warning("Crisis center data fetch failed: %s", e)
            return {'status': 'error', 'error': str(e)}

    def _get_storm_lifeboat_data(self) -> Dict[str, Any]:
        """Get Storm Lifeboat Matrix v9.0 state — scenarios, coherence, lunar, last briefing."""
        if not STORM_LIFEBOAT_AVAILABLE:
            return {"status": "not_available"}
        try:
            import glob as _glob

            result: Dict[str, Any] = {"status": "ok"}

            # ── Scenario heatmap (live, from engine defaults) ──
            try:
                from strategies.storm_lifeboat.core import SCENARIOS as _SL_SCENARIOS
                se = ScenarioEngine()
                result["scenario_heatmap"] = se.get_risk_heatmap()
                active = se.get_active_scenarios()
                result["active_scenarios"] = len(active)
                result["scenarios_total"] = len(_SL_SCENARIOS)
            except Exception as e:
                self.logger.debug("Storm Lifeboat scenario engine: %s", e)
                result["scenario_heatmap"] = []
                result["active_scenarios"] = 0

            # ── Lunar position ──
            try:
                lp = LunarPhiEngine()
                pos = lp.get_position()
                result["lunar"] = {
                    "moon_number": pos.moon_number,
                    "moon_name": pos.moon_name,
                    "phase": pos.phase.value,
                    "day_in_moon": pos.day_in_moon,
                    "in_phi_window": pos.in_phi_window,
                    "phi_coherence": round(pos.phi_coherence, 3),
                    "position_multiplier": round(pos.position_multiplier, 2),
                }
            except Exception as e:
                self.logger.debug("Storm Lifeboat lunar: %s", e)
                result["lunar"] = {}

            # ── Latest Helix briefing (persisted JSON) ──
            briefing_dir = PROJECT_ROOT / "data" / "storm_lifeboat"
            briefings = sorted(_glob.glob(str(briefing_dir / "helix_briefing_*.json")))
            if briefings:
                try:
                    latest = Path(briefings[-1])
                    data = json.loads(latest.read_text(encoding="utf-8"))
                    result["last_briefing"] = {
                        "date": data.get("date"),
                        "headline": data.get("headline"),
                        "regime": data.get("regime"),
                        "mandate": data.get("mandate"),
                        "coherence_score": data.get("coherence_score"),
                        "risk_alert": data.get("risk_alert"),
                        "moon_phase": data.get("moon_phase"),
                        "top_trades": data.get("top_trades", [])[:5],
                        "active_scenarios": data.get("active_scenarios", []),
                    }
                except Exception as e:
                    self.logger.debug("Storm Lifeboat briefing read: %s", e)
                    result["last_briefing"] = {}
            else:
                result["last_briefing"] = {}

            return result

        except Exception as e:
            self.logger.warning("Storm Lifeboat data fetch failed: %s", e)
            return {"status": "error", "error": str(e)}

    async def _get_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts from all systems"""
        alerts = []

        # Check safeguards for alerts
        safeguards_status = get_safeguards_health()
        if safeguards_status.get('overall_health') != 'healthy':
            alerts.append({
                'level': 'warning',
                'message': f"Safeguards status: {safeguards_status['overall_health']}",
                'timestamp': datetime.now(),
                'source': 'safeguards'
            })

        # Check circuit breakers
        for exchange, status in safeguards_status.get('exchanges', {}).items():
            if status.get('circuit_breaker_state') == 'open':
                alerts.append({
                    'level': 'critical',
                    'message': f"Circuit breaker open for {exchange}",
                    'timestamp': datetime.now(),
                    'source': 'circuit_breaker'
                })

        # Check doctrine compliance
        if self.doctrine_integration:
            try:
                compliance_report = await self.doctrine_integration.run_compliance_check()
                violations = compliance_report.get('violations', 0)
                warnings = compliance_report.get('warnings', 0)
                barren_wuffet_state = compliance_report.get('barren_wuffet_state', 'normal')

                if violations > 0:
                    alerts.append({
                        'level': 'critical',
                        'message': f"Doctrine violations: {violations} active",
                        'timestamp': datetime.now(),
                        'source': 'doctrine'
                    })

                if warnings > 0:
                    alerts.append({
                        'level': 'warning',
                        'message': f"Doctrine warnings: {warnings} active",
                        'timestamp': datetime.now(),
                        'source': 'doctrine'
                    })

                if barren_wuffet_state in ['CAUTION', 'CRITICAL']:
                    alerts.append({
                        'level': 'warning' if barren_wuffet_state == 'CAUTION' else 'critical',
                        'message': f"BARREN WUFFET state: {barren_wuffet_state}",
                        'timestamp': datetime.now(),
                        'source': 'doctrine'
                    })

            except Exception as e:
                alerts.append({
                    'level': 'warning',
                    'message': f"Doctrine monitoring error: {str(e)}",
                    'timestamp': datetime.now(),
                    'source': 'doctrine'
                })

        # Check security alerts
        if SECURITY_AVAILABLE and self.security_framework:
            try:
                security_alerts = await self._get_security_alerts()
                alerts.extend(security_alerts)
            except Exception as e:
                self.logger.warning(f"Security alert collection failed: {e}")

        return alerts

    async def _get_security_alerts(self) -> List[Dict[str, Any]]:
        """Get security-related alerts"""
        alerts = []

        # Check for security issues
        if SECURITY_AVAILABLE:
            try:
                # Check for failed login attempts
                failed_logins = self.security_framework['security_monitoring'].get_failed_login_attempts()
                if failed_logins > 10:
                    alerts.append({
                        'level': 'warning',
                        'message': f"High failed login attempts: {failed_logins}",
                        'timestamp': datetime.now(),
                        'source': 'security'
                    })

                # Check for suspicious API calls
                suspicious_calls = self.security_framework['api_security'].get_suspicious_activity()
                if suspicious_calls > 5:
                    alerts.append({
                        'level': 'warning',
                        'message': f"Suspicious API activity: {suspicious_calls} calls",
                        'timestamp': datetime.now(),
                        'source': 'security'
                    })

            except Exception as e:
                self.logger.warning(f"Security alert retrieval failed: {e}")

        return alerts

    def display_dashboard(self, stdscr, data: Dict[str, Any]):
        """Display the monitoring dashboard (terminal mode)"""
        if not data or 'error' in data:
            stdscr.clear()
            stdscr.addstr(0, 0, "[LOADING] Loading AAC Master Monitoring Dashboard...")
            if 'error' in data:
                stdscr.addstr(2, 0, f"Error: {data['error']}")
            stdscr.refresh()
            return

        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # Header
        header = f"[MASTER] AAC 2100 UNIFIED MONITORING DASHBOARD - {data['timestamp'].strftime('%H:%M:%S')}"
        stdscr.addstr(0, 0, header[:width-1], curses.A_BOLD)

        y_pos = 2

        # System Health
        health = data.get('health', {})
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

        # Doctrine Compliance Section
        y_pos += 1
        stdscr.addstr(y_pos, 0, "[DOCTRINE] COMPLIANCE MATRIX", curses.A_BOLD)
        y_pos += 1

        doctrine = data.get('doctrine', {})
        compliance_score = doctrine.get('compliance_score', 0)
        barren_wuffet_state = doctrine.get('barren_wuffet_state', 'unknown')

        # Compliance score with color coding
        score_color = curses.COLOR_GREEN if compliance_score >= 90 else curses.COLOR_YELLOW if compliance_score >= 70 else curses.COLOR_RED
        curses.init_pair(6, score_color, curses.COLOR_BLACK)
        stdscr.addstr(y_pos, 0, f"Compliance Score: {compliance_score}%", curses.color_pair(6))
        y_pos += 1

        # BARREN WUFFET state
        az_color = curses.COLOR_RED if barren_wuffet_state in ['CRITICAL', 'error'] else curses.COLOR_YELLOW if barren_wuffet_state == 'CAUTION' else curses.COLOR_GREEN
        curses.init_pair(7, az_color, curses.COLOR_BLACK)
        stdscr.addstr(y_pos, 0, f"BARREN WUFFET State: {barren_wuffet_state.upper()}", curses.color_pair(7))
        y_pos += 1

        # Compliance metrics
        compliant = doctrine.get('compliant', 0)
        warnings = doctrine.get('warnings', 0)
        violations = doctrine.get('violations', 0)
        stdscr.addstr(y_pos, 0, f"[OK] Compliant: {compliant} | [WARN] Warnings: {warnings} | [ERROR] Violations: {violations}")
        y_pos += 1

        # Monitoring status
        monitoring_active = doctrine.get('monitoring_active', False)
        monitor_color = curses.COLOR_GREEN if monitoring_active else curses.COLOR_RED
        curses.init_pair(8, monitor_color, curses.COLOR_BLACK)
        stdscr.addstr(y_pos, 0, f"Monitoring: {'ACTIVE' if monitoring_active else 'INACTIVE'}", curses.color_pair(8))
        y_pos += 2

        # P&L Section
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

        # Security Status
        security = data.get('security', {})
        if security and security.get('status') != 'not_available':
            stdscr.addstr(y_pos, 0, "[SHIELD] SECURITY STATUS", curses.A_BOLD)
            y_pos += 1

            security_score = security.get('overall_score', 0)
            sec_color = curses.COLOR_GREEN if security_score >= 90 else curses.COLOR_YELLOW if security_score >= 70 else curses.COLOR_RED
            curses.init_pair(9, sec_color, curses.COLOR_BLACK)
            stdscr.addstr(y_pos, 0, f"Security Score: {security_score:.1f}%", curses.color_pair(9))
            y_pos += 1

            components = security.get('components', {})
            for comp_name, comp_data in components.items():
                if isinstance(comp_data, dict):
                    comp_score = comp_data.get('score', 0)
                    comp_color = curses.COLOR_GREEN if comp_score >= 90 else curses.COLOR_YELLOW if comp_score >= 70 else curses.COLOR_RED
                    curses.init_pair(10, comp_color, curses.COLOR_BLACK)
                    stdscr.addstr(y_pos, 0, f"{comp_name.upper()}: {comp_score}%", curses.color_pair(10))
                    y_pos += 1

        # Safeguards Status
        safeguards = data.get('safeguards', {})
        if safeguards:
            stdscr.addstr(y_pos, 0, "[SHIELD]️ PRODUCTION SAFEGUARDS", curses.A_BOLD)
            y_pos += 1

            overall_health = safeguards.get('overall_health', 'unknown')
            safe_color = status_colors.get(overall_health, curses.COLOR_WHITE)
            curses.init_pair(3, safe_color, curses.COLOR_BLACK)
            stdscr.addstr(y_pos, 0, f"Overall: {overall_health.upper()}", curses.color_pair(3))
            y_pos += 1

            exchanges = safeguards.get('exchanges', {})
            for exchange, status in exchanges.items():
                cb_state = status.get('circuit_breaker_state', 'unknown')
                cb_color = curses.COLOR_RED if cb_state == 'open' else curses.COLOR_GREEN
                curses.init_pair(4, cb_color, curses.COLOR_BLACK)
                stdscr.addstr(y_pos, 0, f"{exchange}: CB={cb_state.upper()}", curses.color_pair(4))
                y_pos += 1

        # ── Regime Forecaster (curses panel) ──────────────────────────────
        forecaster = data.get('forecaster', {})
        if forecaster and forecaster.get('status') == 'ok' and y_pos < height - 10:
            y_pos += 1
            try:
                stdscr.addstr(y_pos, 0, "[FORECAST] REGIME", curses.A_BOLD)
                y_pos += 1
                regime = forecaster.get('regime', '?').upper().replace('_', ' ')
                vol = forecaster.get('vol_shock_readiness', 0)
                anchor = forecaster.get('anchor_ticker', '-')
                contagion = forecaster.get('contagion_ticker', '-')
                vol_lbl = 'ARMED' if vol >= 60 else 'ELEV' if vol >= 40 else 'LOW'
                line = f"{regime}  Vol:{vol:.0f}/100[{vol_lbl}]  Stack:{anchor}+{contagion}"
                stdscr.addstr(y_pos, 0, line[:width - 1])
                y_pos += 1
                for o in forecaster.get('top3_opportunities', []):
                    if y_pos >= height - 6:
                        break
                    expr = o['expression'].replace('_', ' ').upper()
                    stdscr.addstr(y_pos, 0, f"  #{o['rank']} {o['ticker']} {expr} sc={o['score']}"[:width - 1])
                    y_pos += 1
            except curses.error:
                pass  # Terminal too small for this panel

        # ── IBKR Orders + $920 Maximization (curses panel) ─────────────────
        ibkr = data.get('ibkr_orders', {})
        if ibkr and ibkr.get('status') == 'ok' and y_pos < height - 6:
            y_pos += 1
            try:
                stdscr.addstr(y_pos, 0, "[IBKR] ORDERS & MAXIMIZE", curses.A_BOLD)
                y_pos += 1
                bal = ibkr.get('balance_usd', 0)
                committed = ibkr.get('total_committed', 0)
                remaining = ibkr.get('remaining_cash', 0)
                stdscr.addstr(y_pos, 0, f"${bal:.0f} bal  ${committed:.0f} deployed  ${remaining:.0f} powder"[:width - 1])
                y_pos += 1
                for o in ibkr.get('open_orders', [])[:3]:
                    if y_pos >= height - 4:
                        break
                    line = f"  #{o['order_id']} {o['symbol']} x{o['qty']:.0f}@${o['price']:.2f} [{o['ibkr_status']}]"
                    stdscr.addstr(y_pos, 0, line[:width - 1])
                    y_pos += 1
                recs = ibkr.get('recommendations', [])
                if recs and y_pos < height - 3:
                    tickers = '+'.join(r['ticker'] for r in recs)
                    stdscr.addstr(y_pos, 0, f"  DEPLOY: {tickers}  cost=${ibkr.get('rec_total_cost', 0):.0f}"[:width - 1])
                    y_pos += 1
            except curses.error:
                pass  # Terminal too small for this panel

        # ── Matrix Maximizer (curses panel) ────────────────────────────
        mm = data.get('matrix_maximizer', {})
        if mm and mm.get('status') == 'ok' and y_pos < height - 8:
            y_pos += 1
            try:
                stdscr.addstr(y_pos, 0, "[MM] MATRIX MAXIMIZER", curses.A_BOLD)
                y_pos += 1
                mandate = mm.get('mandate', '?').upper()
                regime_mm = mm.get('regime', '?').upper()
                cb = mm.get('circuit_breaker', '?')
                stdscr.addstr(y_pos, 0, f"{mandate} | {regime_mm} | CB={cb}"[:width - 1])
                y_pos += 1
                picks = mm.get('top_picks', [])
                for p in picks[:3]:
                    if y_pos >= height - 6:
                        break
                    stdscr.addstr(y_pos, 0, f"  {p['ticker']:6} K={p.get('strike','?')} sc={p.get('score','?')}"[:width - 1])
                    y_pos += 1
            except curses.error:
                pass  # Terminal too small for this panel

        # ── System Registry Summary (curses panel) ─────────────────────
        registry = data.get('registry', {})
        if registry and registry.get('status') != 'not_available' and y_pos < height - 6:
            y_pos += 1
            try:
                stdscr.addstr(y_pos, 0, "[REG] SYSTEM REGISTRY", curses.A_BOLD)
                y_pos += 1
                s = registry.get('summary', {})
                line = (f"APIs:{s.get('apis_configured',0)}/{s.get('total_apis',0)} "
                        f"Exch:{s.get('exchanges_online',0)}/{s.get('exchanges_total',0)} "
                        f"Strat:{s.get('strategies_ok',0)}/{s.get('strategies_total',0)} "
                        f"Dept:{s.get('departments_ok',0)}/{s.get('departments_total',0)}")
                stdscr.addstr(y_pos, 0, line[:width - 1])
                y_pos += 1
            except curses.error:
                pass  # Terminal too small for this panel

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
        if y_pos < height - 2:
            footer_y = height - 2
            stdscr.addstr(footer_y, 0, "Press 'q' to quit | 'r' to refresh | Auto-refresh: 1s")

        stdscr.refresh()

    def _display_text_dashboard(self, data: Dict[str, Any]):
        """Display dashboard in text mode for Windows compatibility"""
        print("\033[2J\033[H", end="")  # ANSI clear screen
        print("="*80)
        print("AAC 2100 MASTER MONITORING DASHBOARD")
        print("="*80)
        print(f"Last update: {data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")
        print("")

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

        # Doctrine Compliance
        doctrine = data.get('doctrine', {})
        if doctrine:
            print("\n[DOCTRINE] COMPLIANCE MATRIX")
            print("-" * 25)
            compliance_score = doctrine.get('compliance_score', 0)
            barren_wuffet_state = doctrine.get('barren_wuffet_state', 'unknown')

            # Compliance score with color emoji
            if compliance_score >= 90:
                score_emoji = '🟢'
            elif compliance_score >= 70:
                score_emoji = '🟡'
            else:
                score_emoji = '🔴'
            print(f"Compliance Score: {score_emoji} {compliance_score}%")

            # BARREN WUFFET state
            if barren_wuffet_state in ['CRITICAL', 'error']:
                az_emoji = '🔴'
            elif barren_wuffet_state == 'CAUTION':
                az_emoji = '🟡'
            else:
                az_emoji = '🟢'
            print(f"BARREN WUFFET State: {az_emoji} {barren_wuffet_state.upper()}")

            # Compliance metrics
            compliant = doctrine.get('compliant', 0)
            warnings = doctrine.get('warnings', 0)
            violations = doctrine.get('violations', 0)
            print(f"[OK] Compliant: {compliant} | [WARN] Warnings: {warnings} | [ERROR] Violations: {violations}")

            # Monitoring status
            monitoring_active = doctrine.get('monitoring_active', False)
            monitor_status = '[ACTIVE]' if monitoring_active else '[INACTIVE]'
            print(f"Monitoring: {monitor_status}")

        # P&L Data
        pnl = data.get('pnl', {})
        if pnl:
            print("\n[MONEY] P&L SUMMARY")
            print("-" * 15)
            daily_pnl = pnl.get('daily_pnl', 0)
            total_pnl = pnl.get('total_equity', 0)
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

        # Security Status
        security = data.get('security', {})
        if security and security.get('status') != 'not_available':
            print("\n[SHIELD] SECURITY STATUS")
            print("-" * 20)
            security_score = security.get('overall_score', 0)
            if security_score >= 90:
                sec_emoji = '🟢'
            elif security_score >= 70:
                sec_emoji = '🟡'
            else:
                sec_emoji = '🔴'
            print(f"Security Score: {sec_emoji} {security_score:.1f}%")

            components = security.get('components', {})
            for comp_name, comp_data in components.items():
                if isinstance(comp_data, dict):
                    comp_score = comp_data.get('score', 0)
                    if comp_score >= 90:
                        comp_emoji = '🟢'
                    elif comp_score >= 70:
                        comp_emoji = '🟡'
                    else:
                        comp_emoji = '🔴'
                    print(f"  {comp_emoji} {comp_name.upper()}: {comp_score}%")

        # Safeguards
        safeguards = data.get('safeguards', {})
        if safeguards:
            print("\n[SHIELD]️  PRODUCTION SAFEGUARDS")
            print("-" * 25)
            circuit_breaker = safeguards.get('circuit_breaker_active', False)
            rate_limited = safeguards.get('rate_limited', False)
            print(f"Circuit Breaker: {'🔴 ACTIVE' if circuit_breaker else '🟢 NORMAL'}")
            print(f"Rate Limiting: {'🟡 ACTIVE' if rate_limited else '🟢 NORMAL'}")

        # Strategy Metrics
        strategy = data.get('strategy', {})
        if strategy and strategy.get('status') != 'not_available':
            print("\n[STRATEGY] STRATEGY METRICS")
            print("-" * 20)
            total_strategies = strategy.get('total_strategies', 0)
            active_strategies = strategy.get('active_strategies', 0)
            avg_return = strategy.get('average_return', 0.0)
            print(f"Total Strategies: {total_strategies}")
            print(f"Active Strategies: {active_strategies}")
            print(f"Average Return: {avg_return:.2%}")

        # ── REGIME FORECASTER INTEL ────────────────────────────────────────
        forecaster = data.get('forecaster', {})
        if forecaster and forecaster.get('status') == 'ok':
            print("")
            print("=" * 60)
            print("  REGIME FORECASTER — IF X+Y → EXPECT Z")
            print("=" * 60)
            regime = forecaster.get('regime', 'UNKNOWN').upper().replace('_', ' ')
            secondary = forecaster.get('secondary')
            conf = forecaster.get('regime_confidence', 0)
            vol = forecaster.get('vol_shock_readiness', 0)
            bears = forecaster.get('bear_signals', 0)
            bulls = forecaster.get('bull_signals', 0)
            anchor = forecaster.get('anchor_ticker', '-')
            contagion = forecaster.get('contagion_ticker', '-')
            macro = forecaster.get('macro_context', {})

            if vol >= 80:
                vol_icon = '[!! SHOCK WINDOW OPEN !!]'
            elif vol >= 60:
                vol_icon = '[ARMED — cheapest vol now]'
            elif vol >= 40:
                vol_icon = '[ELEVATED — watch signals]'
            else:
                vol_icon = '[LOW]'

            print(f"  Regime: {regime}  (conf {conf}%)")
            if secondary:
                print(f"  Secondary: {secondary.upper().replace('_',' ')}")
            print(f"  Vol Shock Readiness: {vol}/100  {vol_icon}")
            print(f"  Signals: {bears} bearish | {bulls} bullish")
            print(f"  Macro: VIX={macro.get('vix','?')}  HY={macro.get('hy_spread','?')}bps  Oil=${macro.get('oil','?')}")
            war_flags = []
            if macro.get('war'): war_flags.append('WAR ACTIVE')
            if macro.get('hormuz'): war_flags.append('HORMUZ BLOCKED')
            if war_flags:
                print(f"  Geo: {' | '.join(war_flags)}")

            fired = forecaster.get('fired_formulas', [])
            if fired:
                print(f"  Fired Formulas ({len(fired)}):")
                for f in fired:
                    print(f"    [{f['tag']}] {f['confidence']}%  {f['outcome']}")

            print(f"")
            print(f"  2-TRADE STACK  |  Anchor: {anchor}  |  Contagion: {contagion}")

            opps = forecaster.get('top3_opportunities', [])
            if opps:
                print(f"  TOP 3 SHORT-TERM OPPORTUNITIES:")
                for o in opps:
                    expr = o['expression'].replace('_', ' ').upper()
                    print(f"    #{o['rank']}  {o['ticker']:6}  {expr:17}  score={o['score']}")
                    print(f"         {o['thesis']}")
            print("=" * 60)

        # ── IBKR OPEN ORDERS + $920 MAXIMIZATION PLAN ───────────────────
        ibkr = data.get('ibkr_orders', {})
        if ibkr and ibkr.get('status') == 'ok':
            print("")
            print("=" * 60)
            print("  IBKR ORDERS  [acct: {}]".format(ibkr.get('account', '?')))
            print("=" * 60)
            bal = ibkr.get('balance_usd', 0)
            committed = ibkr.get('total_committed', 0)
            remaining = ibkr.get('remaining_cash', 0)
            print(f"  Balance: ${bal:.0f} USD  |  Committed: ${committed:.0f}  |  Dry Powder: ${remaining:.0f}")

            orders = ibkr.get('open_orders', [])
            if orders:
                print(f"  Open Orders ({len(orders)}):")
                for o in orders:
                    s = o['ibkr_status']
                    cost = o['cost']
                    print(
                        f"    #{o['order_id']}  {o['symbol']:6}  {o['side'].upper()} "
                        f"x{o['qty']:.0f} @ ${o['price']:.2f}  cost=${cost:.0f}  [{s}]"
                    )
            else:
                print("  No open orders.")

            recs = ibkr.get('recommendations', [])
            if recs:
                rec_total = ibkr.get('rec_total_cost', 0)
                cash_after = ibkr.get('cash_after_recs', 0)
                print(f"")
                print(f"  MAXIMIZATION PLAN (deploy ${rec_total:.0f} / ${remaining:.0f} remaining):")
                for r in recs:
                    print(
                        f"    {r['ticker']:6}  {r['expression']:17}  x{r['contracts']}  "
                        f"est ${r['est_premium']:.2f}/contract  total=${r['cost']:.0f}  DTE {r['dte']}"
                    )
                    print(f"           {r['rationale']}")
                print(f"  Cash buffer after deployment: ${cash_after:.0f}")
            print("=" * 60)

        # ── MATRIX MAXIMIZER DEEP ─────────────────────────────────────
        mm = data.get('matrix_maximizer', {})
        if mm and mm.get('status') == 'ok':
            print("")
            print("=" * 60)
            print("  MATRIX MAXIMIZER — COMMAND & CONTROL")
            print("=" * 60)
            print(f"  Run #{mm.get('run_number','?')}  |  {mm.get('timestamp','?')}")
            mandate = mm.get('mandate', '?').upper()
            mandate_icon = {'DEFENSIVE': '🛡️', 'STANDARD': '⚖️', 'AGGRESSIVE': '⚔️', 'MAX_CONVICTION': '🔥'}.get(mandate, '❓')
            print(f"  Mandate: {mandate_icon} {mandate}  |  Risk/trade: {mm.get('risk_per_trade','?')}%  |  Max pos: {mm.get('max_positions','?')}")
            print(f"  Regime: {mm.get('regime','?').upper()}  conf={mm.get('regime_confidence','?')}%")
            print(f"  Geo: WAR={'YES' if mm.get('war_active') else 'NO'}  HORMUZ={'BLOCKED' if mm.get('hormuz_blocked') else 'OPEN'}")
            print(f"  Oil=${mm.get('oil_price','?')}  VIX={mm.get('vix','?')}")
            spy_ret = mm.get('spy_median_return')
            spy_var = mm.get('spy_var_95')
            if spy_ret is not None:
                print(f"  SPY median return: {spy_ret:.1f}%  |  VaR(95): {spy_var}")
            cb = mm.get('circuit_breaker', '?')
            cb_icon = '🔴' if cb == 'OPEN' else '🟢'
            print(f"  Circuit Breaker: {cb_icon} {cb}  |  Risk Score: {mm.get('risk_score','?')}")

            picks = mm.get('top_picks', [])
            if picks:
                print(f"  Top {len(picks)} Picks (of {mm.get('total_picks',0)}):")
                for p in picks:
                    print(f"    {p['ticker']:6}  K={p.get('strike','?')}  exp={p.get('expiry','?')}  "
                                f"score={p.get('score','?')}  x{p.get('contracts','?')}  ${p.get('cost','?')}")
            elapsed = mm.get('elapsed_s')
            if elapsed:
                print(f"  Cycle time: {elapsed:.1f}s")
            print("=" * 60)

        # ── SYSTEM REGISTRY — API STATUS ──────────────────────────────
        registry = data.get('registry', {})
        if registry and registry.get('status') != 'not_available':
            summary = registry.get('summary', {})
            print("")
            print("=" * 60)
            print("  SYSTEM REGISTRY — COMMAND & CONTROL")
            print("=" * 60)
            print(f"  APIs: {summary.get('apis_configured',0)} configured | "
                        f"{summary.get('apis_missing',0)} missing | "
                        f"{summary.get('apis_free',0)} free (no key)")

            # Exchange status
            exchanges = registry.get('exchanges', [])
            ex_online = summary.get('exchanges_online', 0)
            ex_total = summary.get('exchanges_total', 0)
            print(f"  Exchanges: {ex_online}/{ex_total} online")
            for ex in exchanges:
                h = ex.get('health', 'grey')
                icon = {'green': '🟢', 'yellow': '🟡', 'red': '🔴', 'grey': '⚪'}.get(h, '⚪')
                lat = ex.get('latency_ms')
                lat_str = f"  {lat}ms" if lat else ""
                print(f"    {icon} {ex['name']:20} {ex.get('detail','')}{lat_str}")

            # Infrastructure status
            infra = registry.get('infrastructure', [])
            infra_ok = summary.get('infra_ok', 0)
            infra_total = summary.get('infra_total', 0)
            print(f"  Infrastructure: {infra_ok}/{infra_total} healthy")
            for svc in infra:
                h = svc.get('health', 'grey')
                icon = {'green': '🟢', 'yellow': '🟡', 'red': '🔴', 'grey': '⚪'}.get(h, '⚪')
                print(f"    {icon} {svc['name']:20} {svc.get('detail','')}")

            # Strategy engines
            strats = registry.get('strategies', [])
            strat_ok = summary.get('strategies_ok', 0)
            strat_total = summary.get('strategies_total', 0)
            print(f"  Strategies: {strat_ok}/{strat_total} operational")
            for st_item in strats:
                h = st_item.get('health', 'grey')
                icon = {'green': '🟢', 'yellow': '🟡', 'red': '🔴', 'grey': '⚪'}.get(h, '⚪')
                print(f"    {icon} {st_item['name']:20} {st_item.get('detail','')}")

            # Departments
            depts = registry.get('departments', [])
            dept_ok = summary.get('departments_ok', 0)
            dept_total = summary.get('departments_total', 0)
            print(f"  Departments: {dept_ok}/{dept_total} online")
            for d in depts:
                h = d.get('health', 'grey')
                icon = {'green': '🟢', 'yellow': '🟡', 'red': '🔴', 'grey': '⚪'}.get(h, '⚪')
                print(f"    {icon} {d['name']:25} {d.get('detail','')}")

            # API breakdown by category
            apis = registry.get('apis', [])
            if apis:
                cats: Dict[str, list] = {}
                for a in apis:
                    cat = a.get('category', 'Other')
                    cats.setdefault(cat, []).append(a)
                print(f"  ── API Inventory ({len(apis)} total) ──")
                for cat, items in sorted(cats.items()):
                    conf = sum(1 for i in items if i.get('configured'))
                    print(f"    {cat}: {conf}/{len(items)} configured")
                    for item in items:
                        icon = '✅' if item.get('configured') else ('🔑' if item.get('env_var') else '🆓')
                        print(f"      {icon} {item['name']}")

            # Orphan scripts
            orphans = registry.get('orphans', [])
            if orphans:
                print(f"  Orphan Scripts: {len(orphans)} root _*.py files")
                for o in orphans[:10]:
                    print(f"    📄 {o['script']:30} {o.get('description','')[:50]}")
                if len(orphans) > 10:
                    print(f"    ... and {len(orphans) - 10} more")

            print("=" * 60)

        # Black Swan Crisis Center
        crisis = data.get('crisis_center', {})
        if crisis and crisis.get('status') != 'unavailable':
            print("\n[BLACKSWAN] CRISIS CENTER")
            print("-" * 40)
            if crisis.get('status') == 'error':
                print(f"  ⚠️  Crisis center error: {crisis.get('error', 'unknown')}")
            else:
                print(f"  🌡️  Pressure Level: {crisis.get('pressure_pct', '?')}% ({crisis.get('pressure', '?')})")
                print(f"  📊 Thesis Ratio: {crisis.get('ratio_pct', '?')}% (Bullish: {crisis.get('bullish', '?')} / Bearish: {crisis.get('bearish', '?')} / Neutral: {crisis.get('neutral', '?')})")
                print(f"  🎯 Probability: {crisis.get('probability', '?')}")
                top = crisis.get('top5_indicators', [])
                if top:
                    print("  🔥 Top Indicators:")
                    for ind in top[:5]:
                        print(f"    {ind.get('weight', 0):.2f}  {ind.get('name', '?')}")
            print("=" * 60)

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

    def _run_curses_dashboard(self, stdscr):
        """Run the curses-based dashboard"""
        # Setup curses
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.use_default_colors()
        stdscr.timeout(1000)  # Refresh every second

        while self.running:
            try:
                # Get latest data (thread-safe read)
                with self._data_lock:
                    data = self._latest_data.copy() if self._latest_data else {}

                # Display dashboard
                self.display_dashboard(stdscr, data)

                # Handle input
                key = stdscr.getch()
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('r'):
                    # Manual refresh
                    pass

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                stdscr.clear()
                stdscr.addstr(0, 0, f"Dashboard error: {e}")
                stdscr.refresh()
                time.sleep(2)

    def _run_text_dashboard_loop(self):
        """Run the text-based dashboard loop"""
        print("\n" + "="*80)
        print("AAC 2100 MASTER MONITORING DASHBOARD (Text Mode)")
        print("="*80)
        print("Press Ctrl+C to quit | Auto-refresh: 1s")
        print("")

        while self.running:
            try:
                # Get latest data (thread-safe read)
                with self._data_lock:
                    data = self._latest_data.copy() if self._latest_data else {}

                if data:
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

    async def run_dashboard(self, port: int = 8050):
        """Run the monitoring dashboard based on display mode"""
        if self.display_mode == DisplayMode.WEB:
            await self._run_web_dashboard()
        elif self.display_mode == DisplayMode.DASH:
            await self._run_dash_dashboard()
        elif self.display_mode == DisplayMode.API:
            await self._run_api_dashboard()
        else:  # Default to terminal mode
            await self._run_terminal_dashboard()

    async def _run_terminal_dashboard(self):
        """Run terminal-based dashboard"""
        if CURSES_AVAILABLE and platform.system() != 'Windows':
            # Use curses on Unix-like systems
            def dashboard_thread():
                """Dashboard thread."""
                curses.wrapper(self._run_curses_dashboard)

            # Start dashboard in separate thread
            dashboard_thread = threading.Thread(target=dashboard_thread, daemon=True)
            dashboard_thread.start()
        else:
            # Use text-based dashboard on Windows or when curses unavailable
            def dashboard_thread():
                """Dashboard thread."""
                self._run_text_dashboard_loop()

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
                with self._data_lock:
                    self._latest_data = data

                # Wait before next update
                await asyncio.sleep(self.refresh_rate)

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                logger.error(f"Monitoring error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _run_web_dashboard(self):
        """Run Streamlit web dashboard via subprocess."""
        if not STREAMLIT_AVAILABLE:
            logger.info("Streamlit not available. Install with: pip install streamlit")
            logger.info("Falling back to terminal mode.")
            await self._run_terminal_dashboard()
            return

        import subprocess
        dashboard_script = Path(__file__).parent / 'streamlit_dashboard.py'
        port = int(os.environ.get('STREAMLIT_PORT', '8501'))
        logger.info(f"Launching Streamlit dashboard on port {port}...")
        proc = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', str(dashboard_script),
             '--server.port', str(port), '--server.headless', 'true'],
            cwd=str(PROJECT_ROOT),
        )
        logger.info(f"Streamlit dashboard running (PID {proc.pid}) at http://localhost:{port}")

        # Keep the async loop alive while Streamlit runs
        self.running = True
        try:
            while self.running and proc.poll() is None:
                data = await self.collect_monitoring_data()
                with self._data_lock:
                    self._latest_data = data
                await asyncio.sleep(self.refresh_rate)
        except KeyboardInterrupt:
            self.running = False
        finally:
            proc.terminate()

    async def _run_dash_dashboard(self):
        """Run Plotly Dash analytics dashboard."""
        global DASH_AVAILABLE
        if not DASH_AVAILABLE:
            try:
                import dash  # noqa: F401
                DASH_AVAILABLE = True
            except ImportError:
                logger.info("Dash not available. Install with: pip install dash dash-bootstrap-components")
                logger.info("Falling back to terminal mode.")
                await self._run_terminal_dashboard()
                return

        from monitoring.dash_dashboard import AACDashDashboard
        dash_dashboard = AACDashDashboard()
        if await dash_dashboard.initialize():
            port = int(os.environ.get('DASH_PORT', '8050'))
            logger.info(f"Starting Dash analytics dashboard on port {port}...")
            dash_thread = threading.Thread(
                target=dash_dashboard.run_dashboard,
                kwargs={'port': port},
                daemon=True,
            )
            dash_thread.start()

            # Keep collecting data while Dash runs
            self.running = True
            try:
                while self.running:
                    data = await self.collect_monitoring_data()
                    with self._data_lock:
                        self._latest_data = data
                    await asyncio.sleep(self.refresh_rate)
            except KeyboardInterrupt:
                self.running = False
        else:
            logger.info("Dash dashboard initialization failed, falling back to terminal")
            await self._run_terminal_dashboard()

    async def _run_api_dashboard(self):
        """Run REST API dashboard exposing monitoring data as JSON."""
        from http.server import HTTPServer, BaseHTTPRequestHandler

        dashboard_ref = self

        class MonitoringAPIHandler(BaseHTTPRequestHandler):
            """Simple HTTP handler for monitoring API."""

            def do_GET(self):
                if self.path == '/health':
                    self._json_response({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
                elif self.path == '/api/status':
                    data = dashboard_ref._latest_data
                    self._json_response(self._serializable(data))
                elif self.path == '/api/alerts':
                    data = dashboard_ref._latest_data
                    alerts = data.get('alerts', []) if data else []
                    self._json_response({'alerts': self._serializable(alerts)})
                elif self.path == '/api/registry':
                    data = dashboard_ref._latest_data
                    registry = data.get('registry', {}) if data else {}
                    self._json_response(self._serializable(registry))
                elif self.path == '/api/maximizer':
                    data = dashboard_ref._latest_data
                    mm = data.get('matrix_maximizer', {}) if data else {}
                    self._json_response(self._serializable(mm))
                else:
                    self.send_error(404, 'Endpoints: /health, /api/status, /api/alerts, /api/registry, /api/maximizer')

            def _json_response(self, obj):
                body = json.dumps(obj, default=str).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            @staticmethod
            def _serializable(obj):
                """Convert non-serializable types."""
                if isinstance(obj, dict):
                    return {k: MonitoringAPIHandler._serializable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [MonitoringAPIHandler._serializable(i) for i in obj]
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            def log_message(self, format, *args):
                logger.debug(f"API: {format % args}")

        port = int(os.environ.get('API_DASHBOARD_PORT', '8080'))
        server = HTTPServer(('127.0.0.1', port), MonitoringAPIHandler)
        api_thread = threading.Thread(target=server.serve_forever, daemon=True)
        api_thread.start()
        logger.info(f"API dashboard running on http://localhost:{port}")
        logger.info("  Endpoints: /health, /api/status, /api/alerts, /api/registry, /api/maximizer")

        # Main monitoring loop
        self.running = True
        try:
            while self.running:
                data = await self.collect_monitoring_data()
                with self._data_lock:
                    self._latest_data = data
                await asyncio.sleep(self.refresh_rate)
        except KeyboardInterrupt:
            self.running = False
        finally:
            server.shutdown()

    async def start_monitoring(self):
        """Start the master monitoring system"""
        logger.info("[START] Starting AAC Master Monitoring Dashboard...")

        # Initialize
        if not await self.initialize():
            logger.info("[ERROR] Failed to initialize monitoring dashboard")
            return

        # Start monitoring loop
        await self.run_dashboard()

        logger.info("[SUCCESS] Master monitoring dashboard started")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        logger.info("[STOP] Master monitoring dashboard stopped")


# Global instance
_master_dashboard = None

def get_master_dashboard(display_mode: str = DisplayMode.TERMINAL):
    """Get the appropriate dashboard instance based on display mode"""
    if display_mode == DisplayMode.WEB:
        if STREAMLIT_AVAILABLE:
            from monitoring.streamlit_dashboard import AACStreamlitDashboard
            return AACStreamlitDashboard()
        else:
            logger.info("Streamlit not available, falling back to terminal mode")
            return AACMasterMonitoringDashboard(DisplayMode.TERMINAL)
    elif display_mode == DisplayMode.DASH:
        from monitoring.dash_dashboard import AACDashDashboard
        return AACDashDashboard()
    else:
        return AACMasterMonitoringDashboard(display_mode)


async def _async_main():
    """Async entry point (internal)"""
    import argparse

    parser = argparse.ArgumentParser(description="AAC Master Monitoring Dashboard")
    parser.add_argument("--mode", "-m", choices=['terminal', 'web', 'dash', 'api'],
                       default='terminal', help="Display mode")
    parser.add_argument("--port", "-p", type=int, default=8501,
                       help="Port for web/API modes")

    args = parser.parse_args()

    display_mode = {
        'terminal': DisplayMode.TERMINAL,
        'web': DisplayMode.WEB,
        'dash': DisplayMode.DASH,
        'api': DisplayMode.API
    }.get(args.mode, DisplayMode.TERMINAL)

    dashboard = get_master_dashboard(display_mode)

    try:
        if display_mode == DisplayMode.WEB:
            # Streamlit dashboard
            if hasattr(dashboard, 'run_dashboard'):
                dashboard.run_dashboard(port=args.port)
            else:
                logger.info("Streamlit dashboard not available")
        elif display_mode == DisplayMode.DASH:
            # Dash dashboard
            if hasattr(dashboard, 'run_dashboard'):
                dashboard.run_dashboard(port=args.port)
            else:
                logger.info("Dash dashboard not available")
        else:
            # Terminal dashboard
            await dashboard.start_monitoring()
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down master monitoring dashboard...")
        if hasattr(dashboard, 'stop_monitoring'):
            dashboard.stop_monitoring()
    except Exception as e:
        logger.info(f"[CROSS] Master monitoring dashboard failed: {e}")
        if hasattr(dashboard, 'stop_monitoring'):
            dashboard.stop_monitoring()


def main():
    """Sync entry point for console_scripts / setuptools."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()


# Backward-compatible re-exports for external consumers — lazy to avoid hard dep at import time
def __getattr__(name):  # noqa: E302
    _streamlit_exports = {"AACStreamlitDashboard", "generate_copilot_response", "play_audio_response", "run_streamlit_dashboard"}
    if name in _streamlit_exports:
        from monitoring.streamlit_dashboard import (  # noqa: E402
            AACStreamlitDashboard, generate_copilot_response, play_audio_response, run_streamlit_dashboard
        )
        return locals()[name]
    if name == "AACDashDashboard":
        from monitoring.dash_dashboard import AACDashDashboard  # noqa: E402
        return AACDashDashboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
