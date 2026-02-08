# AAC 2100 Monitoring Systems
## Real-Time Monitoring and Alerting for Live Trading Operations

The AAC 2100 monitoring systems provide comprehensive real-time monitoring, alerting, and incident management for live trading operations.

## Components

### 1. Continuous Monitoring Service (`continuous_monitoring.py`)
Background service that provides continuous system health monitoring with automated alerting.

**Features:**
- Continuous health checks (30-second intervals)
- Real-time alerting via Telegram/Slack
- Automated incident detection and response
- Performance monitoring and anomaly detection
- System resource monitoring
- Trading activity monitoring

### 2. Real-Time Monitoring Dashboard (`monitoring_dashboard.py`)
Terminal-based real-time dashboard displaying system health, P&L, risk metrics, and alerts.

**Features:**
- Real-time system health monitoring
- Live P&L tracking and visualization
- Risk metrics display (VaR, drawdown, Sharpe ratio)
- Active alerts and notifications
- Safeguards status monitoring
- Department health overview

### 3. Monitoring Launcher (`monitoring_launcher.py`)
Unified launcher for starting monitoring systems with various configuration options.

### 4. Monitoring Configuration (`config/monitoring_config.yaml`)
Comprehensive configuration file defining all monitoring settings, thresholds, and notification channels.

## Quick Start

### Start Full System with Monitoring
```bash
python main.py --monitor
```

### Start Monitoring Systems Only
```bash
python main.py --monitoring-only
```

### Launch Monitoring Separately
```bash
# Launch both service and dashboard
python monitoring_launcher.py

# Launch service only
python monitoring_launcher.py --service-only

# Launch dashboard only
python monitoring_launcher.py --dashboard-only
```

## Configuration

### Alert Thresholds
Configure monitoring thresholds in `config/monitoring_config.yaml`:

```yaml
thresholds:
  system:
    cpu_usage_percent: 90.0
    memory_usage_percent: 85.0
    disk_usage_percent: 90.0
  trading:
    circuit_breakers_open: 1
    api_error_rate: 0.05  # 5%
    response_time_ms: 5000
```

### Notification Channels
Configure notification channels:

```yaml
notifications:
  telegram:
    enabled: true
    token: "${TELEGRAM_BOT_TOKEN}"
    chat_id: "${TELEGRAM_CHAT_ID}"
  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"
```

## Dashboard Features

### Real-Time Metrics
- **System Health**: CPU, memory, disk usage
- **P&L Tracking**: Total, daily, hourly P&L
- **Risk Metrics**: VaR (95%), max drawdown, Sharpe ratio
- **Active Alerts**: Current system alerts and warnings
- **Safeguards Status**: Circuit breaker states and rate limits

### Navigation
- `q` or `Ctrl+C`: Quit dashboard
- `r`: Refresh data
- `h`: Show help
- Arrow keys: Navigate panels

## Alert System

### Alert Levels
- **INFO**: Informational messages
- **WARNING**: Potential issues requiring attention
- **CRITICAL**: Immediate action required

### Automated Responses
The system can automatically respond to certain incidents:
- Open circuit breakers trigger trading pauses
- High memory usage triggers service restarts
- Database connection loss triggers failover

## Health Checks

### System Health
- CPU and memory usage monitoring
- Disk space monitoring
- Network connectivity checks

### Department Health
- Central Accounting database connectivity
- Crypto Intelligence venue health
- Trading Execution engine status
- BigBrain Intelligence processing status

### Infrastructure Health
- Database performance and connectivity
- Redis cache status
- External API health (Binance, Coinbase, Kraken)

## Performance Monitoring

### Baselines
The system establishes performance baselines during startup and detects anomalies using statistical analysis (3-sigma rule).

### Metrics Collected
- Response times (P95, mean, max)
- Throughput (operations per second)
- Error rates (percentage)
- Resource utilization (CPU, memory, disk)

## Incident Management

### Automated Detection
- Performance degradation
- System resource exhaustion
- Trading anomalies
- External service failures

### Response Actions
- Alert generation and escalation
- Automatic service restarts
- Trading system pauses
- Failover activation

## Logging

All monitoring activities are logged to:
- Console output
- `logs/monitoring.log` file
- Structured JSON format for analysis

## Integration with Existing Systems

The monitoring systems integrate seamlessly with existing AAC 2100 components:

- **Production Safeguards**: Monitors circuit breaker states and rate limits
- **Health Server**: Uses existing HTTP health endpoints
- **Notification Manager**: Extends existing Telegram/Slack integration
- **Financial Analysis Engine**: Pulls P&L and risk metrics
- **Crypto Intelligence Engine**: Monitors venue health and connectivity

## Environment Variables

Required environment variables for notifications:

```bash
# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Email (optional)
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

## Troubleshooting

### Common Issues

1. **Dashboard not displaying correctly**
   - Ensure terminal supports curses library
   - Check Python version (requires 3.7+)
   - Verify all dependencies are installed

2. **Alerts not being sent**
   - Check environment variables are set
   - Verify API tokens/webhooks are valid
   - Check network connectivity

3. **High CPU usage from monitoring**
   - Adjust monitoring intervals in config
   - Reduce number of health checks
   - Consider running monitoring on separate instance

### Debug Mode
Enable debug logging:
```bash
export MONITORING_LOG_LEVEL=DEBUG
python monitoring_launcher.py
```

## Production Deployment

For production deployment:

1. **Separate monitoring instance** for high availability
2. **Load balancer** for dashboard access
3. **External alerting** for critical notifications
4. **Metrics storage** (InfluxDB/Prometheus) for historical data
5. **Automated failover** for monitoring system itself

## API Reference

### Monitoring Service API
```python
from continuous_monitoring import ContinuousMonitoringService

service = ContinuousMonitoringService()
await service.initialize()
await service.start_monitoring()

# Get monitoring status
status = service.get_monitoring_status()
```

### Dashboard API
```python
from monitoring_dashboard import AACMonitoringDashboard

dashboard = AACMonitoringDashboard()
await dashboard.initialize()
await dashboard.run_dashboard()
```

## Contributing

When adding new monitoring features:

1. Update `config/monitoring_config.yaml` with new settings
2. Add health checks to `continuous_monitoring.py`
3. Update dashboard display in `monitoring_dashboard.py`
4. Add tests for new functionality
5. Update this documentation

## License

This monitoring system is part of the AAC 2100 trading platform.