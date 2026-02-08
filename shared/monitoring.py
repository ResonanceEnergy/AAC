#!/usr/bin/env python3
"""
System Monitoring & Health Checks
=================================
Production monitoring, health checks, and alerting for ACC system.
"""

import asyncio
import json
import logging
import os
import psutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import sys

# For async HTTP requests
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import get_config, get_project_path

logger = logging.getLogger('monitoring')


# ============================================
# NOTIFICATION PROVIDERS
# ============================================

class TelegramNotifier:
    """
    Sends notifications via Telegram Bot API.
    
    Setup:
    1. Create a bot with @BotFather and get the token
    2. Get your chat_id by messaging @userinfobot
    3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID', '')
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.logger = logging.getLogger('TelegramNotifier')
        self._enabled = bool(self.bot_token and self.chat_id)
        
        if not self._enabled:
            self.logger.warning("Telegram notifications disabled - missing bot_token or chat_id")
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram"""
        if not self._enabled:
            return False
        
        if not AIOHTTP_AVAILABLE:
            self.logger.error("aiohttp not available for Telegram notifications")
            return False
        
        url = f"{self.api_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        self.logger.debug("Telegram message sent successfully")
                        return True
                    else:
                        error = await resp.text()
                        self.logger.error(f"Telegram API error: {resp.status} - {error}")
                        return False
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    async def send_alert(self, alert: 'Alert') -> bool:
        """Send an alert as a formatted Telegram message"""
        emoji = {
            'critical': '[ALERT]',
            'warning': '[WARN]️',
            'info': 'ℹ️',
        }.get(alert.severity, '📢')
        
        message = f"""
{emoji} <b>ACC Alert - {alert.severity.upper()}</b>

<b>{alert.title}</b>
{alert.message}

<i>Category:</i> {alert.category}
<i>Time:</i> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return await self.send_message(message.strip())


class SlackNotifier:
    """
    Sends notifications via Slack Webhooks.
    
    Setup:
    1. Create a Slack App and enable Incoming Webhooks
    2. Set SLACK_WEBHOOK_URL env var
    """
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv('SLACK_WEBHOOK_URL', '')
        self.logger = logging.getLogger('SlackNotifier')
        self._enabled = bool(self.webhook_url)
        
        if not self._enabled:
            self.logger.warning("Slack notifications disabled - missing webhook_url")
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    async def send_message(self, text: str, blocks: Optional[List] = None) -> bool:
        """Send a message via Slack"""
        if not self._enabled:
            return False
        
        if not AIOHTTP_AVAILABLE:
            self.logger.error("aiohttp not available for Slack notifications")
            return False
        
        payload = {"text": text}
        if blocks:
            payload["blocks"] = blocks
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        self.logger.debug("Slack message sent successfully")
                        return True
                    else:
                        error = await resp.text()
                        self.logger.error(f"Slack API error: {resp.status} - {error}")
                        return False
        except Exception as e:
            self.logger.error(f"Failed to send Slack message: {e}")
            return False
    
    async def send_alert(self, alert: 'Alert') -> bool:
        """Send an alert as a formatted Slack message"""
        color = {
            'critical': '#FF0000',
            'warning': '#FFA500',
            'info': '#0000FF',
        }.get(alert.severity, '#808080')
        
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*ACC Alert - {alert.severity.upper()}*\n*{alert.title}*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Message:*\n{alert.message}"},
                    {"type": "mrkdwn", "text": f"*Category:*\n{alert.category}"},
                    {"type": "mrkdwn", "text": f"*Time:*\n{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"},
                ]
            }
        ]
        
        return await self.send_message(alert.title, blocks)


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'latency_ms': self.latency_ms,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_percent: float
    disk_used_gb: float
    open_files: int
    threads: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    severity: str  # 'info', 'warning', 'critical'
    category: str  # 'system', 'trading', 'risk', 'connection'
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'severity': self.severity,
            'category': self.category,
            'title': self.title,
            'message': self.message,
            'data': self.data,
            'acknowledged': self.acknowledged,
            'created_at': self.created_at.isoformat(),
        }


class HealthChecker:
    """
    Performs health checks on system components.
    """
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.logger = logging.getLogger('HealthChecker')
        
        # Register default checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.register_check('system_resources', self._check_system_resources)
        self.register_check('database', self._check_database)
        self.register_check('config', self._check_config)
        self.register_check('disk_space', self._check_disk_space)
    
    def register_check(self, name: str, check_fn: Callable):
        """Register a health check function"""
        self.checks[name] = check_fn
        self.logger.debug(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a single health check"""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Unknown health check: {name}"
            )
        
        start = time.time()
        try:
            check_fn = self.checks[name]
            if asyncio.iscoroutinefunction(check_fn):
                result = await check_fn()
            else:
                result = check_fn()
            
            result.latency_ms = (time.time() - start) * 1000
            self.results[name] = result
            return result
            
        except Exception as e:
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                latency_ms=(time.time() - start) * 1000
            )
            self.results[name] = result
            return result
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        tasks = [self.run_check(name) for name in self.checks]
        await asyncio.gather(*tasks)
        return self.results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.results:
            return HealthStatus.UNKNOWN
        
        statuses = [r.status for r in self.results.values()]
        
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system CPU and memory"""
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        status = HealthStatus.HEALTHY
        message = "System resources OK"
        
        if cpu > 90 or memory.percent > 90:
            status = HealthStatus.UNHEALTHY
            message = f"Critical resource usage - CPU: {cpu}%, Memory: {memory.percent}%"
        elif cpu > 70 or memory.percent > 80:
            status = HealthStatus.DEGRADED
            message = f"High resource usage - CPU: {cpu}%, Memory: {memory.percent}%"
        
        return HealthCheckResult(
            name='system_resources',
            status=status,
            message=message,
            details={
                'cpu_percent': cpu,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
            }
        )
    
    def _check_database(self) -> HealthCheckResult:
        """Check database connectivity"""
        try:
            from CentralAccounting.database import AccountingDatabase
            db = AccountingDatabase()
            conn = db.connect()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            db.close()
            
            return HealthCheckResult(
                name='database',
                status=HealthStatus.HEALTHY,
                message="Database connection OK"
            )
        except Exception as e:
            return HealthCheckResult(
                name='database',
                status=HealthStatus.UNHEALTHY,
                message=f"Database error: {str(e)}"
            )
    
    def _check_config(self) -> HealthCheckResult:
        """Check configuration is loaded"""
        try:
            config = get_config()
            
            # Verify critical config sections exist
            checks = []
            if hasattr(config, 'binance'):
                checks.append('binance')
            if hasattr(config, 'risk'):
                checks.append('risk')
            
            if len(checks) >= 2:
                return HealthCheckResult(
                    name='config',
                    status=HealthStatus.HEALTHY,
                    message="Configuration loaded",
                    details={'loaded_sections': checks}
                )
            else:
                return HealthCheckResult(
                    name='config',
                    status=HealthStatus.DEGRADED,
                    message="Some config sections missing",
                    details={'loaded_sections': checks}
                )
        except Exception as e:
            return HealthCheckResult(
                name='config',
                status=HealthStatus.UNHEALTHY,
                message=f"Config error: {str(e)}"
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space"""
        disk = psutil.disk_usage('/')
        
        status = HealthStatus.HEALTHY
        message = f"Disk space OK ({100 - disk.percent:.1f}% free)"
        
        if disk.percent > 95:
            status = HealthStatus.UNHEALTHY
            message = f"Critical: Only {100 - disk.percent:.1f}% disk space free"
        elif disk.percent > 85:
            status = HealthStatus.DEGRADED
            message = f"Warning: Only {100 - disk.percent:.1f}% disk space free"
        
        return HealthCheckResult(
            name='disk_space',
            status=status,
            message=message,
            details={
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent_used': disk.percent,
            }
        )


class MetricsCollector:
    """
    Collects and stores system metrics over time.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: List[SystemMetrics] = []
        self.logger = logging.getLogger('MetricsCollector')
    
    def collect(self) -> SystemMetrics:
        """Collect current system metrics"""
        process = psutil.Process()
        disk = psutil.disk_usage('/')
        memory = psutil.virtual_memory()
        
        metrics = SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            disk_percent=disk.percent,
            disk_used_gb=disk.used / (1024**3),
            open_files=len(process.open_files()) if hasattr(process, 'open_files') else 0,
            threads=process.num_threads(),
        )
        
        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        return metrics
    
    def get_averages(self, minutes: int = 5) -> Dict[str, float]:
        """Get average metrics over specified time window"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent = [m for m in self.metrics_history if m.timestamp > cutoff]
        
        if not recent:
            return {}
        
        return {
            'avg_cpu': sum(m.cpu_percent for m in recent) / len(recent),
            'avg_memory': sum(m.memory_percent for m in recent) / len(recent),
            'max_cpu': max(m.cpu_percent for m in recent),
            'max_memory': max(m.memory_percent for m in recent),
            'samples': len(recent),
        }


class AlertManager:
    """
    Manages system alerts and notifications.
    Supports Telegram and Slack notifications.
    """
    
    def __init__(
        self,
        enable_telegram: bool = True,
        enable_slack: bool = True,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        slack_webhook: Optional[str] = None,
    ):
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        self._alert_counter = 0
        self.logger = logging.getLogger('AlertManager')
        
        # Notification providers
        self.telegram = TelegramNotifier(telegram_token, telegram_chat_id) if enable_telegram else None
        self.slack = SlackNotifier(slack_webhook) if enable_slack else None
        
        # Alert thresholds
        self.thresholds = {
            'cpu_warning': 70,
            'cpu_critical': 90,
            'memory_warning': 80,
            'memory_critical': 95,
            'disk_warning': 85,
            'disk_critical': 95,
        }
        
        # Track sent alerts to avoid spam (cooldown period)
        self._sent_alerts: Dict[str, datetime] = {}
        self.cooldown_seconds = 300  # 5 minutes between duplicate alerts
    
    def add_callback(self, callback: Callable):
        """Add alert notification callback"""
        self.alert_callbacks.append(callback)
    
    def _should_send_notification(self, alert_key: str) -> bool:
        """Check if we should send notification (cooldown check)"""
        if alert_key in self._sent_alerts:
            elapsed = (datetime.now() - self._sent_alerts[alert_key]).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False
        return True
    
    async def create_alert(
        self,
        severity: str,
        category: str,
        title: str,
        message: str,
        data: Optional[Dict] = None,
        send_notification: bool = True,
    ) -> Alert:
        """Create and store a new alert"""
        self._alert_counter += 1
        alert = Alert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._alert_counter:04d}",
            severity=severity,
            category=category,
            title=title,
            message=message,
            data=data or {},
        )
        
        self.alerts.append(alert)
        self.logger.log(
            logging.CRITICAL if severity == 'critical' else 
            logging.WARNING if severity == 'warning' else logging.INFO,
            f"[{severity.upper()}] {title}: {message}"
        )
        
        # Send external notifications for critical/warning alerts
        alert_key = f"{category}:{title}"
        if send_notification and severity in ('critical', 'warning'):
            if self._should_send_notification(alert_key):
                await self._send_notifications(alert)
                self._sent_alerts[alert_key] = datetime.now()
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
        
        return alert
    
    async def _send_notifications(self, alert: Alert):
        """Send alert to all enabled notification channels"""
        tasks = []
        
        if self.telegram and self.telegram.enabled:
            tasks.append(self.telegram.send_alert(alert))
        
        if self.slack and self.slack.enabled:
            tasks.append(self.slack.send_alert(alert))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Notification error: {result}")
    
    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_unacknowledged(self, severity: Optional[str] = None) -> List[Alert]:
        """Get unacknowledged alerts"""
        alerts = [a for a in self.alerts if not a.acknowledged]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts
    
    async def check_metrics_and_alert(self, metrics: SystemMetrics):
        """Check metrics and create alerts if thresholds exceeded"""
        if metrics.cpu_percent >= self.thresholds['cpu_critical']:
            await self.create_alert(
                'critical', 'system', 'Critical CPU Usage',
                f"CPU usage at {metrics.cpu_percent}%",
                {'cpu_percent': metrics.cpu_percent}
            )
        elif metrics.cpu_percent >= self.thresholds['cpu_warning']:
            await self.create_alert(
                'warning', 'system', 'High CPU Usage',
                f"CPU usage at {metrics.cpu_percent}%",
                {'cpu_percent': metrics.cpu_percent}
            )
        
        if metrics.memory_percent >= self.thresholds['memory_critical']:
            await self.create_alert(
                'critical', 'system', 'Critical Memory Usage',
                f"Memory usage at {metrics.memory_percent}%",
                {'memory_percent': metrics.memory_percent}
            )
        elif metrics.memory_percent >= self.thresholds['memory_warning']:
            await self.create_alert(
                'warning', 'system', 'High Memory Usage',
                f"Memory usage at {metrics.memory_percent}%",
                {'memory_percent': metrics.memory_percent}
            )


class MonitoringService:
    """
    Main monitoring service that coordinates all monitoring components.
    """
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.health_checker = HealthChecker()
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self._running = False
        self.logger = logging.getLogger('MonitoringService')
    
    async def start(self):
        """Start the monitoring service"""
        self._running = True
        self.logger.info("Monitoring service started")
        
        while self._running:
            try:
                # Collect metrics
                metrics = self.metrics_collector.collect()
                
                # Check for alerts
                await self.alert_manager.check_metrics_and_alert(metrics)
                
                # Run health checks
                await self.health_checker.run_all_checks()
                
                # Log status
                status = self.health_checker.get_overall_status()
                if status != HealthStatus.HEALTHY:
                    self.logger.warning(f"System health: {status.value}")
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    def stop(self):
        """Stop the monitoring service"""
        self._running = False
        self.logger.info("Monitoring service stopped")
    
    def get_health_report(self) -> Dict:
        """Get comprehensive health report"""
        return {
            'overall_status': self.health_checker.get_overall_status().value,
            'checks': {name: r.to_dict() for name, r in self.health_checker.results.items()},
            'metrics': self.metrics_collector.metrics_history[-1].to_dict() if self.metrics_collector.metrics_history else {},
            'averages_5min': self.metrics_collector.get_averages(5),
            'unacknowledged_alerts': len(self.alert_manager.get_unacknowledged()),
            'critical_alerts': len(self.alert_manager.get_unacknowledged('critical')),
            'timestamp': datetime.now().isoformat(),
        }
    
    def get_health_endpoint(self) -> Dict:
        """Get health status for API endpoint"""
        status = self.health_checker.get_overall_status()
        return {
            'status': status.value,
            'healthy': status == HealthStatus.HEALTHY,
            'timestamp': datetime.now().isoformat(),
        }


# Convenience functions
_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """Get or create the global monitoring service"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service


def get_monitoring_manager() -> MonitoringService:
    """Get or create the global monitoring manager (alias for get_monitoring_service)"""
    return get_monitoring_service()


async def health_check() -> Dict:
    """Quick health check endpoint"""
    service = get_monitoring_service()
    await service.health_checker.run_all_checks()
    return service.get_health_endpoint()


async def initialize_monitoring():
    """Initialize the monitoring system"""
    service = get_monitoring_service()
    # Any initialization logic here
    pass


async def shutdown_monitoring():
    """Shutdown the monitoring system"""
    service = get_monitoring_service()
    # Any shutdown logic here
    pass


# CLI for testing
if __name__ == '__main__':
    async def test():
        print("=== ACC Monitoring System Test ===\n")
        
        # Create service
        service = MonitoringService(check_interval=5)
        
        # Collect metrics
        metrics = service.metrics_collector.collect()
        print(f"CPU: {metrics.cpu_percent}%")
        print(f"Memory: {metrics.memory_percent}%")
        print(f"Disk: {metrics.disk_percent}%")
        
        # Run health checks
        print("\n--- Health Checks ---")
        results = await service.health_checker.run_all_checks()
        for name, result in results.items():
            print(f"  {name}: {result.status.value} - {result.message}")
        
        print(f"\nOverall Status: {service.health_checker.get_overall_status().value}")
        
        # Get full report
        print("\n--- Health Report ---")
        report = service.get_health_report()
        print(json.dumps(report, indent=2, default=str))
    
    asyncio.run(test())
