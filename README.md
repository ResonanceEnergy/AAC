# Accelerated Arbitrage Corp (ACC)

A sophisticated cryptocurrency arbitrage and trading system with multi-theater research agents, real-time execution, and comprehensive risk management.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                │
│              Central coordination & signal aggregation              │
├────────────────┬────────────────┬───────────────────────────────────┤
│   THEATER B    │   THEATER C    │   THEATER D                       │
│   Attention    │ Infrastructure │   Information Asymmetry           │
│   & Narrative  │   & Latency    │   & Alpha                         │
├────────────────┴────────────────┴───────────────────────────────────┤
│                    BIGBRAIN INTELLIGENCE                            │
│         11 Research Agents + CryptoIntelligence Integration         │
├─────────────────────────────────────────────────────────────────────┤
│                    TRADING EXECUTION                                │
│      Binance • Coinbase • Kraken | Risk Manager | Order Manager     │
├─────────────────────────────────────────────────────────────────────┤
│                    CENTRAL ACCOUNTING                               │
│              SQLite Database | Transaction Ledger                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Redis (optional, for caching)
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/accelerated-arbitrage-corp/acc.git
cd acc

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Copy environment template:**
```bash
cp .env.example .env
```

2. **Edit `.env` with your credentials:**
```env
# Exchange API Keys
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_secret
BINANCE_TESTNET=true

COINBASE_API_KEY=your_api_key
COINBASE_API_SECRET=your_secret
COINBASE_PASSPHRASE=your_passphrase

KRAKEN_API_KEY=your_api_key
KRAKEN_API_SECRET=your_secret

# Notifications (optional)
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
SLACK_WEBHOOK_URL=https://hooks.slack.com/...

# Trading Mode
PAPER_TRADING=true
DRY_RUN=false
```

3. **Review config files:**
- `config/config.yaml` - Main configuration
- `config/trading_config.yaml` - Trading parameters
- `model_risk_caps.json` - Risk limits per exchange

### Running the System

```bash
# Start with paper trading (recommended for testing)
python main.py

# Or use PowerShell launcher
.\Start-ACC.ps1
```

## 📁 Project Structure

```
ACC/
├── orchestrator.py          # Central coordinator
├── main.py                  # Entry point
├── BigBrainIntelligence/    # Research agents
│   ├── agents.py            # 11 specialized agents
│   └── research_agent.py    # Agent base classes
├── TradingExecution/        # Trading engine
│   ├── execution_engine.py  # Order execution
│   ├── trading_engine.py    # Trade management
│   ├── risk_manager.py      # Risk controls
│   ├── order_manager.py     # Order persistence
│   └── exchange_connectors/ # Exchange APIs
│       ├── binance_connector.py
│       ├── coinbase_connector.py
│       └── kraken_connector.py
├── CentralAccounting/       # Financial tracking
│   └── database.py          # SQLite ledger
├── CryptoIntelligence/      # Crypto analysis
│   └── crypto_bigbrain_integration.py
├── shared/                  # Common utilities
│   ├── config_loader.py     # Configuration
│   ├── data_sources.py      # Market data
│   ├── utils.py             # CircuitBreaker, RateLimiter
│   ├── monitoring.py        # Health checks & alerts
│   ├── secrets_manager.py   # API key encryption
│   ├── audit_logger.py      # Compliance logging
│   └── health_server.py     # HTTP health endpoints
├── config/                  # Configuration files
├── data/                    # Persistent data
├── logs/                    # Log files
└── tests/                   # Test suite
```

## 🎭 Theater System

### Theater B - Attention & Narrative
Monitors social sentiment, news flow, and market narratives.
- `narrative_analyzer` - News and social media analysis
- `social_sentiment` - Twitter/Reddit sentiment
- `influencer_tracker` - Key opinion leader monitoring

### Theater C - Infrastructure & Latency
Tracks exchange health, network conditions, and execution quality.
- `latency_monitor` - Exchange response times
- `liquidity_scanner` - Order book depth analysis
- `fee_optimizer` - Trading cost optimization

### Theater D - Information Asymmetry
Detects alpha opportunities and on-chain signals.
- `whale_tracker` - Large wallet movements
- `mempool_analyzer` - Pending transaction analysis
- `orderflow_analyzer` - Market microstructure

## ⚙️ Configuration Guide

### Risk Management (`model_risk_caps.json`)
```json
{
  "binance": {
    "max_position_size": 10000,
    "max_daily_loss": 500,
    "max_positions": 5
  }
}
```

### Trading Parameters (`config/trading_config.yaml`)
```yaml
execution:
  default_slippage_tolerance: 0.002  # 0.2%
  order_timeout_seconds: 30
  
risk:
  max_portfolio_risk: 0.02  # 2% max risk per trade
  stop_loss_pct: 0.05       # 5% stop loss
```

## 🐳 Docker Deployment

```bash
# Start core services
docker-compose up -d

# Start with monitoring stack
docker-compose --profile monitoring up -d

# View logs
docker-compose logs -f acc
```

### Services
| Service | Port | Description |
|---------|------|-------------|
| acc | 8080 | Main application + health server |
| redis | 6379 | Caching & message queue |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Dashboards (admin/admin) |

## 📊 Health & Monitoring

### Health Endpoints
- `GET /health` - Overall system health
- `GET /health/live` - Liveness probe (K8s)
- `GET /health/ready` - Readiness probe (K8s)
- `GET /health/detailed` - Component-level status
- `GET /metrics` - Prometheus metrics

### Key Metrics
- `acc_signals_total` - Signals generated by theater
- `acc_orders_total` - Orders by status/exchange
- `acc_positions_active` - Open position count
- `acc_pnl_total` - Realized P&L
- `acc_circuit_breaker_state` - Circuit breaker status

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test class
python -m pytest tests/test_suite.py::TestExecutionEngine -v
```

## 🔒 Security

- **API Key Encryption**: Keys encrypted at rest using Fernet (AES-128)
- **Input Validation**: All order parameters validated before submission
- **Audit Logging**: All API calls and orders logged for compliance
- **Circuit Breakers**: Automatic protection against exchange failures
- **Rate Limiting**: Per-exchange rate limit enforcement

## ⚠️ Risk Warnings

1. **Paper Trading First**: Always test with `PAPER_TRADING=true` before live trading
2. **Start Small**: Begin with minimal position sizes
3. **Monitor Actively**: Watch for unexpected behavior
4. **Understand Fees**: Exchange fees can eliminate arbitrage profits
5. **Network Latency**: Arbitrage opportunities may disappear before execution

## 📜 License

Proprietary - All rights reserved

## 🤝 Support

For issues and questions, please open a GitHub issue or contact the development team.
