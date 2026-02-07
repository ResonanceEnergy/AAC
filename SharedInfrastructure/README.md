# Shared Infrastructure

This directory contains the core shared infrastructure components for the Accelerated Arbitrage Corp (ACC) system. These components provide essential services across all departments and ensure system reliability, security, and monitoring.

## Components

### 🔐 Security Monitor (`security_monitor.py`)
- **Purpose**: Monitors security events and threats across the entire ACC system
- **Features**:
  - Real-time security event detection
  - Automated threat response
  - Integration with audit logging
  - Configurable security policies
- **Key Functions**:
  - `monitor_security_events()`: Main monitoring loop
  - `process_security_event()`: Handle detected security events
  - `get_security_status()`: Current security status

### 📊 Audit Logger (`audit_logger.py`)
- **Purpose**: Comprehensive audit logging system for compliance and security monitoring
- **Features**:
  - Structured audit entries with timestamps
  - Compliance-ready logging (7-year retention)
  - Search and filtering capabilities
  - Automatic log rotation and cleanup
- **Key Functions**:
  - `log_event()`: Log audit events
  - `get_recent_entries()`: Retrieve recent audit entries
  - `get_compliance_report()`: Generate compliance reports

### 🏥 System Monitor (`system_monitor.py`)
- **Purpose**: Monitors overall system health and performance across all departments
- **Features**:
  - Component health checking
  - System resource monitoring (CPU, memory, disk)
  - Automated health reporting
  - Alert generation for issues
- **Key Functions**:
  - `start_monitoring()`: Begin monitoring loop
  - `perform_health_checks()`: Check all components
  - `get_health_status()`: Current system health

### 🔍 Health Checker (`health_checker.py`)
- **Purpose**: Provides detailed health checking capabilities for system components
- **Features**:
  - Individual component health checks
  - Custom health check registration
  - Health history tracking
  - Comprehensive health summaries
- **Key Functions**:
  - `run_health_check()`: Check specific component
  - `run_all_checks()`: Check all components
  - `get_health_summary()`: Overall health summary

### 📈 Metrics Collector (`metrics_collector.py`)
- **Purpose**: Collects and aggregates system metrics from all departments
- **Features**:
  - Real-time metrics collection
  - Multiple metric types (gauge, count, histogram)
  - Metric aggregation and querying
  - Export capabilities (JSON, Prometheus)
- **Key Functions**:
  - `start_collection()`: Begin metrics collection
  - `record_metric()`: Record metric values
  - `get_metric_value()`: Retrieve metric values

### 🚨 Alert Manager (`alert_manager.py`)
- **Purpose**: Manages system alerts and notifications across all departments
- **Features**:
  - Alert creation and management
  - Multiple notification channels (email, log, console)
  - Alert escalation policies
  - Rule-based alert generation
- **Key Functions**:
  - `create_alert()`: Create new alerts
  - `acknowledge_alert()`: Acknowledge alerts
  - `check_alert_rules()`: Evaluate alert rules

## Usage

### Basic Usage

```python
from SharedInfrastructure.security_monitor import get_security_monitor
from SharedInfrastructure.audit_logger import get_audit_logger
from SharedInfrastructure.system_monitor import get_system_monitor

# Get instances
security = await get_security_monitor()
audit = get_audit_logger()
system = await get_system_monitor()

# Log an audit event
audit.log_event("component", "action", "details", "info")

# Check system health
health = await system.get_health_status()
print(f"System status: {health['overall_status']}")
```

### Starting Infrastructure Services

```python
import asyncio
from SharedInfrastructure.system_monitor import get_system_monitor
from SharedInfrastructure.metrics_collector import get_metrics_collector

async def start_infrastructure():
    # Start monitoring services
    system_monitor = await get_system_monitor()
    metrics_collector = await get_metrics_collector()

    # Start monitoring loops
    await asyncio.gather(
        system_monitor.start_monitoring(),
        metrics_collector.start_collection()
    )

# Run infrastructure
asyncio.run(start_infrastructure())
```

## Configuration

Infrastructure components can be configured through:

1. **Environment Variables**: For runtime configuration
2. **Config Files**: Located in the `config/` directory
3. **Direct API**: Programmatic configuration

Example configuration:

```python
# Configure alert manager
from SharedInfrastructure.alert_manager import get_alert_manager

alert_mgr = await get_alert_manager()
alert_mgr.configure_email(
    smtp_server="smtp.company.com",
    smtp_port=587,
    sender_email="alerts@acc-system.local",
    recipient_emails=["ops@company.com", "security@company.com"]
)
```

## Integration Points

### Department Integration

Each infrastructure component integrates with department-specific engines:

- **CentralAccounting**: Financial metrics and health checks
- **CryptoIntelligence**: Venue monitoring and security events
- **BigBrainIntelligence**: Research agent status and metrics
- **TradingExecution**: Trading metrics and performance monitoring

### Cross-Department Communication

Infrastructure components enable cross-department communication through:

- **Bridge Services**: Inter-departmental message passing
- **Shared Metrics**: Unified metrics collection
- **Centralized Alerts**: System-wide alert management
- **Audit Logging**: Comprehensive activity tracking

## Monitoring and Maintenance

### Health Checks

Run comprehensive health checks:

```python
from SharedInfrastructure.health_checker import get_health_checker

checker = await get_health_checker()
summary = await checker.get_health_summary()
print(f"Overall health: {summary['overall_status']}")
```

### Metrics Export

Export metrics for analysis:

```python
from SharedInfrastructure.metrics_collector import get_metrics_collector

collector = await get_metrics_collector()
metrics_json = collector.export_metrics("json")
prometheus_data = collector.export_metrics("prometheus")
```

### Alert Management

Manage system alerts:

```python
from SharedInfrastructure.alert_manager import get_alert_manager

alert_mgr = await get_alert_manager()
active_alerts = alert_mgr.get_active_alerts()
summary = alert_mgr.get_alert_summary()
```

## Security Considerations

- All components implement proper access controls
- Audit logging captures all security-relevant events
- Encryption is used for sensitive data
- Components validate inputs and handle errors gracefully

## Performance

- Asynchronous operations for non-blocking performance
- Efficient metric storage with automatic cleanup
- Configurable monitoring intervals
- Resource usage monitoring to prevent system overload

## Dependencies

- `psutil`: System resource monitoring
- `asyncio`: Asynchronous operations
- `smtplib`: Email notifications
- `statistics`: Metric calculations
- `json`: Data serialization

## Future Enhancements

- Distributed monitoring across multiple nodes
- Advanced alerting with PagerDuty integration
- Metrics visualization dashboard
- Automated remediation actions
- Machine learning-based anomaly detection