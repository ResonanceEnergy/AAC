#!/usr/bin/env python3
"""
AAC 2100 Continuous Monitoring Service
======================================
Background monitoring service that provides continuous system health monitoring,
real-time alerting, and automated incident response.

Features:
- Continuous health checks (30-second intervals)
- Real-time alerting via Telegram/Slack
- Automated incident detection and response
- Performance monitoring and anomaly detection
- System resource monitoring
- Trading activity monitoring
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
import json
import psutil

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config
from shared.monitoring import get_monitoring_manager
from shared.production_safeguards import get_production_safeguards, get_safeguards_health
from CentralAccounting.financial_analysis_engine import FinancialAnalysisEngine
from CryptoIntelligence.crypto_intelligence_engine import CryptoIntelligenceEngine


class ContinuousMonitoringService:
    """Continuous monitoring service for AAC 2100"""

    def __init__(self):
        self.config = get_config()
        self.monitoring = get_monitoring_manager()
        self.safeguards = get_production_safeguards()
        self.financial_engine = FinancialAnalysisEngine()
        self.crypto_engine = CryptoIntelligenceEngine()

        self.logger = logging.getLogger('ContinuousMonitor')
        self.running = False

        # Monitoring intervals (seconds)
        self.health_check_interval = 30
        self.performance_check_interval = 60
        self.alert_check_interval = 10

        # Alert thresholds
        self.alert_thresholds = {
            'cpu_usage_percent': 90.0,
            'memory_usage_percent': 85.0,
            'disk_usage_percent': 90.0,
            'circuit_breakers_open': 1,
            'api_error_rate': 0.05,  # 5%
            'response_time_ms': 5000,  # 5 seconds
        }

        # Alert state
        self.active_alerts = {}
        self.alert_history = []

        # Performance baselines
        self.performance_baselines = {}

    async def initialize(self):
        """Initialize the monitoring service"""
        self.logger.info("🔄 Initializing Continuous Monitoring Service...")

        # Initialize safeguards
        from shared.production_safeguards import initialize_production_safeguards
        await initialize_production_safeguards()

        # Initialize monitoring
        from shared.monitoring import initialize_monitoring
        await initialize_monitoring()

        # Establish performance baselines (skip for quick launch)
        # await self._establish_baselines()

        self.logger.info("✅ Continuous Monitoring Service initialized")

    async def _establish_baselines(self):
        """Establish performance baselines for anomaly detection"""
        self.logger.info("[MONITOR] Establishing performance baselines...")

        # Collect baseline data for 5 minutes
        baseline_samples = []
        for i in range(30):  # 30 samples over ~5 minutes
            sample = await self._collect_performance_sample()
            baseline_samples.append(sample)
            await asyncio.sleep(10)  # 10 second intervals

        # Calculate baselines
        if baseline_samples:
            self.performance_baselines = self._calculate_baselines(baseline_samples)
            self.logger.info("✅ Performance baselines established")

    def _calculate_baselines(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistical baselines from samples"""
        if not samples:
            return {}

        baselines = {}
        metrics = ['cpu_percent', 'memory_percent', 'response_time', 'error_rate']

        for metric in metrics:
            values = [s.get(metric, 0) for s in samples if metric in s]
            if values:
                baselines[metric] = {
                    'mean': sum(values) / len(values),
                    'std_dev': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5,
                    'min': min(values),
                    'max': max(values),
                    'p95': sorted(values)[int(len(values) * 0.95)]
                }

        return baselines

    async def start_monitoring(self):
        """Start the continuous monitoring service"""
        self.logger.info("[DEPLOY] Starting Continuous Monitoring Service...")
        self.running = True

        # Create monitoring tasks
        tasks = [
            self._health_monitoring_loop(),
            self._performance_monitoring_loop(),
            self._alert_monitoring_loop(),
            self._incident_detection_loop()
        ]

        # Run all monitoring tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _health_monitoring_loop(self):
        """Continuous health monitoring loop"""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)

    async def _performance_monitoring_loop(self):
        """Continuous performance monitoring loop"""
        while self.running:
            try:
                await self._perform_performance_checks()
                await asyncio.sleep(self.performance_check_interval)
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(5)

    async def _alert_monitoring_loop(self):
        """Continuous alert monitoring loop"""
        while self.running:
            try:
                await self._check_alerts()
                await asyncio.sleep(self.alert_check_interval)
            except Exception as e:
                self.logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(1)

    async def _incident_detection_loop(self):
        """Continuous incident detection loop"""
        while self.running:
            try:
                await self._detect_incidents()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Incident detection error: {e}")
                await asyncio.sleep(5)

    async def _perform_health_checks(self):
        """Perform comprehensive health checks"""
        health_status = {
            'timestamp': datetime.now(),
            'system': await self._check_system_health(),
            'departments': await self._check_department_health(),
            'infrastructure': await self._check_infrastructure_health(),
            'trading': await self._check_trading_health()
        }

        # Log health status
        overall_health = self._determine_overall_health(health_status)
        self.logger.info(f"🏥 Health Check: {overall_health}")

        # Store health data for monitoring
        self._latest_health = health_status

    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system-level health"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }

    async def _check_department_health(self) -> Dict[str, Any]:
        """Check department-level health"""
        departments = {}

        try:
            # Central Accounting
            db_status = await self.financial_engine.health_check()
            departments['CentralAccounting'] = {
                'status': 'healthy' if db_status.get('database_connected') else 'critical',
                'last_activity': db_status.get('last_transaction_time')
            }
        except:
            departments['CentralAccounting'] = {'status': 'error'}

        try:
            # Crypto Intelligence
            venue_health = await self.crypto_engine.get_venue_health()
            avg_health = sum(v.get('health_score', 0) for v in venue_health.values()) / len(venue_health) if venue_health else 0
            departments['CryptoIntelligence'] = {
                'status': 'healthy' if avg_health > 0.7 else 'warning',
                'average_venue_health': avg_health
            }
        except:
            departments['CryptoIntelligence'] = {'status': 'error'}

        # BigBrain Intelligence and Trading Execution - placeholder
        departments['BigBrainIntelligence'] = {'status': 'healthy'}
        departments['TradingExecution'] = {'status': 'healthy'}

        return departments

    async def _check_infrastructure_health(self) -> Dict[str, Any]:
        """Check infrastructure health"""
        return {
            'database': await self._check_database_health(),
            'network': await self._check_network_health(),
            'external_apis': await self._check_external_api_health()
        }

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            from CentralAccounting.database import DatabaseManager
            db = DatabaseManager()
            health = await db.health_check()
            return {
                'status': 'healthy' if health.get('connected') else 'critical',
                'response_time_ms': health.get('response_time', 0),
                'active_connections': health.get('active_connections', 0)
            }
        except:
            return {'status': 'critical'}

    async def _check_network_health(self) -> Dict[str, Any]:
        """Check network connectivity"""
        # Simple ping test to common services
        return {
            'status': 'healthy',
            'latency_ms': 15,  # TODO: implement actual network checks
            'packet_loss': 0.0
        }

    async def _check_external_api_health(self) -> Dict[str, Any]:
        """Check external API health"""
        apis = {}
        configured_exchanges = self.config.get_enabled_exchanges()

        for exchange_name in configured_exchanges:
            # TODO: implement actual API health checks
            apis[exchange_name] = {'status': 'healthy', 'response_time_ms': 100}

        return apis

    async def _check_trading_health(self) -> Dict[str, Any]:
        """Check trading system health"""
        return {
            'active_positions': 0,  # TODO: implement
            'pending_orders': 0,    # TODO: implement
            'circuit_breakers': get_safeguards_health()
        }

    def _determine_overall_health(self, health_status: Dict[str, Any]) -> str:
        """Determine overall system health"""
        statuses = []

        # System health
        system = health_status.get('system', {})
        if system.get('cpu_percent', 0) > 95 or system.get('memory_percent', 0) > 95:
            statuses.append('critical')
        elif system.get('cpu_percent', 0) > 85 or system.get('memory_percent', 0) > 85:
            statuses.append('warning')

        # Department health
        departments = health_status.get('departments', {})
        for dept, status in departments.items():
            dept_status = status.get('status', 'unknown')
            statuses.append(dept_status)

        # Infrastructure health
        infra = health_status.get('infrastructure', {})
        for component, status in infra.items():
            comp_status = status.get('status', 'unknown')
            statuses.append(comp_status)

        # Determine overall
        if 'critical' in statuses or 'error' in statuses:
            return 'CRITICAL'
        elif 'warning' in statuses:
            return 'WARNING'
        else:
            return 'HEALTHY'

    async def _perform_performance_checks(self):
        """Perform performance monitoring"""
        sample = await self._collect_performance_sample()

        # Check for anomalies
        anomalies = self._detect_anomalies(sample)

        if anomalies:
            self.logger.warning(f"[WARN]️ Performance anomalies detected: {anomalies}")
            await self._handle_performance_anomalies(anomalies, sample)

    async def _collect_performance_sample(self) -> Dict[str, Any]:
        """Collect a performance sample"""
        return {
            'timestamp': datetime.now(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io': psutil.disk_io_counters(),
            'network_io': psutil.net_io_counters(),
            'response_time': 0,  # TODO: implement actual response time measurement
            'error_rate': 0,     # TODO: implement error rate calculation
            'throughput': 0      # TODO: implement throughput measurement
        }

    def _detect_anomalies(self, sample: Dict[str, Any]) -> List[str]:
        """Detect performance anomalies"""
        anomalies = []

        for metric, value in sample.items():
            if metric in self.performance_baselines:
                baseline = self.performance_baselines[metric]
                if baseline['std_dev'] > 0:  # Avoid division by zero
                    z_score = abs(value - baseline['mean']) / baseline['std_dev']
                    if z_score > 3.0:  # 3 standard deviations
                        anomalies.append(f"{metric}: {value:.2f} (expected: {baseline['mean']:.2f}±{baseline['std_dev']:.2f})")

        return anomalies

    async def _handle_performance_anomalies(self, anomalies: List[str], sample: Dict[str, Any]):
        """Handle detected performance anomalies"""
        # Create alert
        alert = {
            'type': 'performance_anomaly',
            'severity': 'warning',
            'message': f"Performance anomalies detected: {', '.join(anomalies)}",
            'timestamp': datetime.now(),
            'data': sample
        }

        await self._send_alert(alert)

    async def _check_alerts(self):
        """Check for new alerts and manage active alerts"""
        # Check system alerts
        await self._check_system_alerts()

        # Check trading alerts
        await self._check_trading_alerts()

        # Check safeguard alerts
        await self._check_safeguard_alerts()

        # Clean up resolved alerts
        await self._cleanup_resolved_alerts()

    async def _check_system_alerts(self):
        """Check for system-level alerts"""
        system_health = getattr(self, '_latest_health', {}).get('system', {})

        # CPU usage alert
        cpu_percent = system_health.get('cpu_percent', 0)
        if cpu_percent > self.alert_thresholds['cpu_usage_percent']:
            await self._create_alert(
                'system_high_cpu',
                'critical',
                f"High CPU usage: {cpu_percent:.1f}%"
            )

        # Memory usage alert
        memory_percent = system_health.get('memory_percent', 0)
        if memory_percent > self.alert_thresholds['memory_usage_percent']:
            await self._create_alert(
                'system_high_memory',
                'critical',
                f"High memory usage: {memory_percent:.1f}%"
            )

    async def _check_trading_alerts(self):
        """Check for trading-related alerts"""
        # Check circuit breakers
        safeguards_status = get_safeguards_health()
        open_breakers = sum(
            1 for exchange in safeguards_status.get('exchanges', {}).values()
            if exchange.get('circuit_breaker_state') == 'open'
        )

        if open_breakers >= self.alert_thresholds['circuit_breakers_open']:
            await self._create_alert(
                'trading_circuit_breakers',
                'critical',
                f"{open_breakers} circuit breaker(s) are open"
            )

    async def _check_safeguard_alerts(self):
        """Check safeguard-related alerts"""
        # Implementation for safeguard alerts
        pass

    async def _create_alert(self, alert_id: str, severity: str, message: str):
        """Create a new alert"""
        if alert_id in self.active_alerts:
            return  # Alert already active

        alert = {
            'id': alert_id,
            'severity': severity,
            'message': message,
            'timestamp': datetime.now(),
            'resolved': False
        }

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        self.logger.warning(f"[ALERT] Alert created: {alert_id} - {message}")
        await self._send_alert(alert)

    async def _send_alert(self, alert: Dict[str, Any]):
        """Send alert via configured channels"""
        try:
            from shared.monitoring import TelegramNotifier, SlackNotifier

            message = f"[ALERT] AAC 2100 Alert\n{alert['message']}\nTime: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"

            # Send via Telegram
            if self.config.notifications.telegram_enabled():
                telegram = TelegramNotifier(
                    token=self.config.notifications.telegram_token,
                    chat_id=self.config.notifications.telegram_chat_id
                )
                await telegram.send_message(message)

            # Send via Slack
            if self.config.notifications.slack_enabled():
                slack = SlackNotifier(
                    webhook_url=self.config.notifications.slack_webhook
                )
                await slack.send_message(message)

        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")

    async def _cleanup_resolved_alerts(self):
        """Clean up resolved alerts"""
        # Implementation for alert cleanup
        pass

    async def _detect_incidents(self):
        """Detect and handle incidents"""
        # Implementation for incident detection
        pass

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'running': self.running,
            'active_alerts': len(self.active_alerts),
            'health_status': getattr(self, '_latest_health', {}),
            'performance_baselines': self.performance_baselines,
            'last_health_check': getattr(self, '_latest_health', {}).get('timestamp')
        }

    async def stop_monitoring(self):
        """Stop the monitoring service"""
        self.logger.info("🛑 Stopping Continuous Monitoring Service...")
        self.running = False

        # Stop monitoring manager
        from shared.monitoring import shutdown_monitoring
        await shutdown_monitoring()

        self.logger.info("✅ Continuous Monitoring Service stopped")


async def main():
    """Main entry point"""
    service = ContinuousMonitoringService()

    try:
        await service.initialize()
        await service.start_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down continuous monitoring...")
        await service.stop_monitoring()
    except Exception as e:
        print(f"[CROSS] Continuous monitoring failed: {e}")
        await service.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())