# 🚀 Accelerated Arbitrage Corp (AAC) - Complete Arbitrage Trading System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-orange.svg)]()

AAC is a comprehensive, production-ready arbitrage trading system that identifies and executes profitable arbitrage opportunities across global financial markets including stocks, cryptocurrencies, forex, commodities, and derivatives.

## 🏗️ Architecture Overview

### Original ACC Architecture
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
│         20 Research Agents + 6 Super Agents + CryptoIntelligence Integration         │
├─────────────────────────────────────────────────────────────────────┤
│                    TRADING EXECUTION                                │
│      Binance • Coinbase • Kraken | Risk Manager | Order Manager     │
├─────────────────────────────────────────────────────────────────────┤
│                    CENTRAL ACCOUNTING                               │
│              SQLite Database | Transaction Ledger                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Enhanced AAC Arbitrage System
```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-SOURCE ARBITRAGE DETECTOR                  │
│         Alpha Vantage • CoinGecko • CurrencyAPI • Twelve Data       │
│         Polygon.io • Finnhub • ECB • World Bank • More...           │
├─────────────────────────────────────────────────────────────────────┤
│                    ARBITRAGE STRATEGIES                             │
│   Cross-Exchange • Triangular • Statistical • Macro • Sentiment     │
├─────────────────────────────────────────────────────────────────────┤
│                    BINANCE TRADING ENGINE                           │
│         Order Management • Risk Controls • Position Tracking        │
├─────────────────────────────────────────────────────────────────────┤
│                    EXECUTION SYSTEM                                 │
│         Real-time Monitoring • Automated Trading • Performance      │
├─────────────────────────────────────────────────────────────────────┤
│                    MONITORING DASHBOARD                             │
│         Streamlit Web UI • Real-time Charts • System Health         │
└─────────────────────────────────────────────────────────────────────┘
```

## 🎯 Key Features

### Multi-Source Data Integration
- **Alpha Vantage**: Global stock market data (25 calls/day)
- **CoinGecko**: Cryptocurrency data (unlimited calls)
- **CurrencyAPI**: Forex rates (300 calls/month)
- **Twelve Data**: Real-time market data (800 calls/day)
- **Polygon.io**: US market and options data (5M calls/month)
- **Finnhub**: Real-time quotes and sentiment (150 calls/day)
- **ECB**: European economic data (free)
- **World Bank**: Macroeconomic indicators (free)

### Arbitrage Strategies
- ✅ **Cross-Exchange Arbitrage**: Price differences between exchanges
- ✅ **Triangular Arbitrage**: Currency triangle opportunities
- ✅ **Statistical Arbitrage**: Mean-reversion strategies
- ✅ **Macro Arbitrage**: Economic indicator-based
- ✅ **Sentiment-Based Arbitrage**: News and social sentiment

### Trading & Risk Management
- **Binance Integration**: Spot and futures trading
- **Position Sizing**: Risk-based position calculation
- **Stop Loss**: Automatic loss protection
- **Performance Tracking**: Real-time P&L monitoring
- **Risk Controls**: Daily loss limits and position caps

### Monitoring & Control
- **Real-Time Dashboard**: Web-based monitoring interface
- **Performance Analytics**: Charts and metrics
- **System Health**: Automated health checks
- **Trade Logging**: Complete execution history

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Internet connection for API access
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Accelerated-Arbitrage-Corp

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Copy environment template:**
```bash
cp .env.example .env
```

2. **Edit `.env` with your API credentials:**
```env
# Alpha Vantage (Global Stocks)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# CoinGecko (Cryptocurrencies)
COINGECKO_API_KEY=your_coingecko_key

# CurrencyAPI (Forex)
CURRENCY_API_KEY=your_currency_api_key

# Twelve Data (Real-time Data)
TWELVE_DATA_API_KEY=your_twelve_data_key

# Polygon.io (US Market & Options)
POLYGON_API_KEY=your_polygon_key

# Finnhub (Real-time Quotes & Sentiment)
FINNHUB_API_KEY=your_finnhub_key

# Binance (Exchange Trading)
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret
BINANCE_TESTNET=true

# Trading Configuration
AUTO_EXECUTE=false
ENABLE_TEST_MODE=true
MAX_POSITION_SIZE_USD=10000
MAX_DAILY_LOSS_USD=1000
MIN_CONFIDENCE_THRESHOLD=0.7
```

### Running the Complete AAC System

```bash
# 🚀 UNIFIED SYSTEM LAUNCH (Recommended)
# Launch complete system: doctrine + agents + trading + monitoring
python core/aac_master_launcher.py --mode paper    # Paper trading (default)
python core/aac_master_launcher.py --mode live     # Live trading (CAUTION!)
python core/aac_master_launcher.py --mode dry-run  # Dry run mode

# 🔍 COMPONENT-SPECIFIC LAUNCH
python core/aac_master_launcher.py --doctrine-only  # Doctrine compliance only
python core/aac_master_launcher.py --agents-only    # Department agents only
python core/aac_master_launcher.py --trading-only   # Trading systems only

# 📊 MONITORING ONLY
python core/aac_master_launcher.py --monitoring-only   # Full monitoring system
python core/aac_master_launcher.py --dashboard-only    # Dashboard only (terminal)
python core/aac_master_launcher.py --service-only      # Background service only
python core/aac_master_launcher.py --dashboard-only --display-mode web  # Web dashboard
```

### ⚠️ DEPRECATED Startup Methods

The following startup methods are **deprecated** and will be removed:

```bash
# ❌ DEPRECATED - Use aac_master_launcher.py instead
python core/main.py                           # → python core/aac_master_launcher.py
python integrations/run_integrated_system.py          # → python core/aac_master_launcher.py --doctrine-only
python deployment/deploy_aac_system.py              # → python core/aac_master_launcher.py
python monitoring/monitoring_launcher.py            # → python core/aac_master_launcher.py --monitoring-only
python monitoring/aac_monitoring_dashboard.py       # → python core/aac_master_launcher.py --dashboard-only
.\scripts\Start-ACC.ps1                         # → python core/aac_master_launcher.py
```

### Legacy Testing Commands

```bash
# Test individual components
python multi_source_arbitrage_demo.py
streamlit run aac_monitoring_dashboard.py
```

## 📁 Project Structure

### New Organized Directory Structure
```
aac-main/
├── core/                          # Core application files
│   ├── aac_master_launcher.py     # Main system launcher
│   ├── main.py                    # Legacy entry point (deprecated)
│   ├── orchestrator.py            # System orchestrator
│   └── command_center.py          # Command center interface
├── agents/                        # Agent-related files
│   ├── agent_based_trading.py     # Trading agent contest system
│   ├── aac_agent_consolidation.py # Agent consolidation system
│   └── avatar_system.py           # Avatar agent system
├── strategies/                    # Strategy implementation
│   ├── strategy_agent_master_mapping.py # Agent-strategy mapping
│   ├── etf_nav_dislocation.py      # Individual strategy files
│   └── ...                        # 49 strategy implementations
├── trading/                       # Trading systems
│   ├── aac_arbitrage_execution_system.py
│   ├── binance_trading_engine.py
│   └── live_trading_environment.py
├── integrations/                  # API integrations
│   ├── api_integration_hub.py
│   ├── market_data_aggregator.py
│   └── coinbase_api_async.py
├── monitoring/                    # Monitoring and dashboards
│   ├── aac_master_monitoring_dashboard.py
│   ├── continuous_monitoring.py
│   └── security_dashboard.py
├── deployment/                    # Deployment and production
│   ├── aac_deployment_engine.py
│   ├── deploy_aac_system.py
│   └── production_readiness_integration.py
├── reddit/                        # Reddit integration
│   ├── aac_reddit_integration.py
│   ├── aac_reddit_web_scraper.py
│   └── reddit_sentiment_integration.py
├── docs/                          # Documentation
│   ├── AAC_2100_DOCTRINE_PACKS_COMPLETE.md
│   ├── AAC_2100_IMPLEMENTATION_ROADMAP.md
│   └── business_continuity/
├── tools/                         # Utility tools
│   ├── deep_dive_analysis.py
│   ├── validate_strategies.py
│   └── fix_json.py
├── demos/                         # Demonstration files
├── scripts/                       # Setup and utility scripts
├── tests/                         # Test files
├── data/                          # Data files and samples
├── logs/                          # Log files
├── reports/                       # Report files and metrics
├── temp/                          # Temporary files
├── archive/                       # Deprecated/orphaned files
├── config/                        # Configuration files
├── shared/                        # Shared utilities
├── models/                        # ML models
├── assets/                        # Static assets
└── divisions/                     # Department-specific code
    ├── BigBrainIntelligence/
    ├── CentralAccounting/
    ├── ComplianceArbitrageDivision/
    └── ...
```
├── deploy_production.py                 # Production deployment script
├── additional_data_sources.py           # Extended data source catalog
├── polygon_arbitrage_integration.py     # Polygon.io integration
├── finnhub_arbitrage_integration.py     # Finnhub integration
├── advanced_arbitrage_integration.py    # Multi-source analysis engine
├── worldwide_arbitrage_demo.py          # Global arbitrage demo
├── aac_enhanced_arbitrage_roadmap.py    # Implementation roadmap
├── test_*.py                            # API and component tests
└── .env                                 # API key configuration
```

### Original ACC System
```
ACC/
├── orchestrator.py                      # Central coordinator
├── main.py                              # Entry point
├── BigBrainIntelligence/                # Research agents
│   ├── agents.py                        # 20 specialized research agents
│   └── research_agent.py                # Agent base classes
├── TradingExecution/                    # Trading engine
│   ├── execution_engine.py              # Order execution
│   ├── trading_engine.py                # Trade management
│   ├── risk_manager.py                  # Risk controls
│   ├── order_manager.py                 # Order persistence
│   └── exchange_connectors/             # Exchange APIs
│       ├── binance_connector.py
│       ├── coinbase_connector.py
│       └── kraken_connector.py
├── CentralAccounting/                   # Financial tracking
│   └── database.py                      # SQLite ledger
├── CryptoIntelligence/                  # Crypto analysis
│   └── crypto_bigbrain_integration.py
├── shared/                              # Common utilities
│   ├── config_loader.py                 # Configuration
│   ├── data_sources.py                  # Market data
│   ├── utils.py                         # CircuitBreaker, RateLimiter
│   ├── monitoring.py                    # Health checks & alerts
│   ├── secrets_manager.py               # API key encryption
│   ├── audit_logger.py                  # Compliance logging
│   └── health_server.py                # HTTP health endpoints
├── config/                              # Configuration files
├── data/                                # Persistent data
├── logs/                                # Log files
└── tests/                               # Test suite
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

## 🔗 AAC Arbitrage System

### Data Sources Integration

The enhanced AAC system integrates multiple financial data APIs for comprehensive market coverage:

| API | Purpose | Rate Limit | Status |
|-----|---------|------------|--------|
| Alpha Vantage | Global Stocks | 25/day | ✅ Configured |
| CoinGecko | Crypto | Unlimited | ✅ Configured |
| CurrencyAPI | Forex | 300/month | ✅ Configured |
| Twelve Data | Real-time | 800/day | ✅ Configured |
| Polygon.io | US Market | 5M/month | ✅ Configured |
| Finnhub | Quotes/Sentiment | 150/day | ✅ Configured |
| ECB | Economic Data | Unlimited | ✅ Configured |
| World Bank | Macro Data | Unlimited | ✅ Configured |
| Binance | Trading | Varies | ✅ Integrated |

### Arbitrage Strategies

#### Cross-Exchange Arbitrage
```python
from multi_source_arbitrage_demo import MultiSourceArbitrageDetector

detector = MultiSourceArbitrageDetector()
opportunities = await detector.detect_opportunities()

for opp in opportunities:
    if opp['type'] == 'cross_exchange':
        print(f"Arbitrage: {opp['symbol']} - Spread: {opp['spread']:.2%}")
```

#### Triangular Arbitrage
Exploits inefficiencies in currency triangles (BTC → ETH → USDT → BTC).

#### Statistical Arbitrage
Uses statistical models for mean-reversion opportunities.

### Trading Engine

#### Binance Integration
```python
from binance_trading_engine import BinanceTradingEngine, TradingConfig

config = TradingConfig(max_position_size_usd=10000)
engine = BinanceTradingEngine(binance_config, config)

# Place limit order
await engine.place_limit_order('BTCUSDT', 'BUY', 0.001, 45000)

# Check positions
positions = await engine.check_positions()
```

#### Risk Management
- **Position Sizing**: Risk-based calculation
- **Stop Loss**: Automatic 5% stop loss protection
- **Daily Loss Limits**: Maximum $1000 daily loss
- **Position Caps**: Maximum 10 concurrent positions

### Monitoring Dashboard

Start the real-time monitoring dashboard:

```bash
streamlit run aac_monitoring_dashboard.py
```

Features:
- Real-time position monitoring
- Performance charts and analytics
- System health indicators
- Trade execution logs
- Interactive controls

### Production Deployment

#### Test Mode
```bash
python deploy_production.py --mode test
```

#### Live Trading
```bash
# WARNING: This will execute real trades!
python deploy_production.py --mode live
```

#### System Status
```bash
python deploy_production.py --status
python deploy_production.py --health-check
```

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

## 🔑 Automated Exchange API Credential Loading

AAC automatically loads your Binance, Coinbase, and Kraken API credentials from your `.env` file using the config loader. To enable live trading:

1. Copy `.env.example` to `.env` and fill in your real API keys:
   ```env
   BINANCE_API_KEY=your_key
   BINANCE_API_SECRET=your_secret
   COINBASE_API_KEY=your_key
   COINBASE_API_SECRET=your_secret
   COINBASE_PASSPHRASE=your_passphrase
   KRAKEN_API_KEY=your_key
   KRAKEN_API_SECRET=your_secret
   ```
2. The config loader will automatically detect and securely inject these credentials into the trading engines at runtime.
3. **Never commit your real `.env` file to version control.**

For more details, see `shared/config_loader.py` and `.env.example`.
