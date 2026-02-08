# 🚀 AAC Matrix Monitor - Advanced Arbitrage Corporation

[![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Proprietary-orange.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-green.svg)]()

**AAC Matrix Monitor** is a comprehensive enterprise financial intelligence platform featuring real-time monitoring, AI-powered analytics, multi-department orchestration, and the revolutionary AZ Executive Assistant. Built for institutional-grade trading operations with advanced security frameworks and compliance monitoring.

## 🏗️ Architecture Overview

### AAC Matrix Monitor System Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                    AAC MATRIX MONITOR                               │
│              Real-time Enterprise Intelligence Platform             │
├────────────────┬────────────────┬────────────────┬─────────────────┤
│   🤖 AZ EXEC   │   📊 MONITOR   │   🏛️ DEPART    │   🔒 SECURITY     │
│   ASSISTANT    │   DASHBOARD    │   DIVISIONS    │   FRAMEWORK      │
│   45 Strategic │   Streamlit    │   15 Divisions │   RBAC + MFA     │
│   Questions    │   Web UI       │   Orchestrated  │   Encryption     │
├────────────────┴────────────────┴────────────────┴─────────────────┤
│                    BIGBRAIN INTELLIGENCE                            │
│         20 Research Agents + 6 Super Agents + Avatar Animation      │
├─────────────────────────────────────────────────────────────────────┤
│                    ARBITRAGE TRADING ENGINE                         │
│      Multi-Source Detection • Risk Management • Position Tracking   │
├─────────────────────────────────────────────────────────────────────┤
│                    CENTRAL ACCOUNTING                               │
│              SQLite Database | Transaction Ledger | Analytics       │
└─────────────────────────────────────────────────────────────────────┘
```

### Department Divisions Architecture
```
AAC Divisions/
├── CentralAccounting/              # Financial Analysis Engine
├── ComplianceArbitrageDivision/    # Regulatory Compliance
├── CorporateBankingDivision/       # Institutional Banking
├── CryptoIntelligence/             # Cryptocurrency Analysis
├── HR_Division/                    # Personnel Management
├── InternationalInsuranceDivision/ # Global Insurance
├── LudwigLawDivision/              # Legal Compliance
├── OptionsArbitrageDivision/       # Options Strategies
├── PaperTradingDivision/           # Risk-Free Testing
├── PortfolioManagementDivision/    # Asset Allocation
├── QuantitativeArbitrageDivision/  # Statistical Models
├── QuantitativeResearchDivision/   # Research & Modeling
├── RiskManagementDivision/         # Risk Assessment
├── StatisticalArbitrageDivision/   # Statistical Trading
├── StructuralArbitrageDivision/    # Cross-Market Arbitrage
├── TechnologyArbitrageDivision/    # Tech Sector Opportunities
└── TechnologyInfrastructureDivision/ # System Administration
```

## 🎯 Key Features

### 🤖 AZ Executive Assistant
- **45 Strategic Questions**: Comprehensive framework across 8 categories
- **Avatar Animation**: Real-time facial expressions with OpenCV
- **Audio Responses**: Text-to-speech integration with pyttsx3
- **Interactive Interface**: Streamlit-powered strategic guidance
- **Categories**: Market Analysis, Risk Assessment, Strategy Optimization, Technology Integration, Compliance & Regulation, Performance Metrics, Innovation & Research, Crisis Management

### 📊 Matrix Monitor Dashboard
- **Browser Auto-Open**: Automatic dashboard launch in default browser
- **Real-Time Monitoring**: Live system health and performance metrics
- **Multi-Department View**: Unified monitoring across all divisions
- **Security Dashboard**: Authentication, API security, and compliance monitoring
- **Performance Analytics**: Interactive charts and risk visualizations

### 🔀 Multi-Source Arbitrage Engine
- **Alpha Vantage**: Global stock market data (25 calls/day)
- **CoinGecko**: Cryptocurrency data (unlimited calls)
- **CurrencyAPI**: Forex rates (300 calls/month)
- **Twelve Data**: Real-time market data (800 calls/day)
- **Polygon.io**: US market and options data (5M calls/month)
- **Finnhub**: Real-time quotes and sentiment (150 calls/day)
- **ECB**: European economic data (free)
- **World Bank**: Macroeconomic indicators (free)

### 🏛️ Department Divisions
- **Central Accounting & Finance** - Financial analysis and reporting
- **Crypto Intelligence** - Cryptocurrency market analysis
- **Corporate Banking** - Institutional banking operations
- **Human Resources** - Personnel management
- **International Insurance** - Global insurance products
- **Ludwig Law Division** - Legal compliance and contracts
- **Options Arbitrage** - Options trading strategies
- **Paper Trading** - Risk-free strategy testing
- **Portfolio Management** - Asset allocation and optimization
- **Quantitative Research** - Statistical modeling
- **Risk Management** - Risk assessment and mitigation
- **Technology Infrastructure** - System administration
- **Statistical Arbitrage** - Statistical trading models
- **Structural Arbitrage** - Cross-market arbitrage
- **Technology Arbitrage** - Tech sector opportunities

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

### Security & Compliance
- **Role-Based Access Control (RBAC)**
- **Multi-Factor Authentication (MFA)**
- **End-to-End Encryption**
- **Audit Logging & Compliance Monitoring**
- **Circuit Breaker Protection**
- **Production Safeguards**

## 🚀 Quick Start

### Prerequisites
- **Python 3.14+**
- **Git**
- **Internet connection for API access**
- **Windows/Linux/macOS**

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd aac-matrix-monitor

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### ⚡ One-Click Launch (Recommended)

#### Full AAC Matrix Monitor System
```bash
# Windows Batch Files
LFGCC!.bat              # Launch complete system (doctrine + agents + trading + monitoring)
LFGCC_DASHBOARD!.bat    # Launch Matrix Monitor dashboard only (auto-opens browser)

# Python Direct Launch
python core/aac_master_launcher.py --mode paper    # Paper trading (default)
python core/aac_master_launcher.py --mode live     # Live trading (CAUTION!)
```

#### Component-Specific Launch
```bash
# Launch individual components
python core/aac_master_launcher.py --dashboard-only    # Matrix Monitor only
python core/aac_master_launcher.py --az-assistant      # AZ Executive Assistant only
python core/aac_master_launcher.py --agents-only       # Department agents only
python core/aac_master_launcher.py --trading-only      # Trading systems only
```

### 🤖 AZ Executive Assistant

Launch the strategic guidance system:

```bash
# Launch AZ Assistant (opens in browser automatically)
python core/aac_master_launcher.py --az-assistant

# Or use the dashboard launcher
LFGCC_DASHBOARD!.bat
```

Features:
- 45 strategic questions across 8 categories
- Real-time avatar animation
- Audio responses with pyttsx3
- Interactive Streamlit interface

### 📊 Matrix Monitor Dashboard

Launch the real-time monitoring dashboard:

```bash
# Auto-opens browser to localhost:8080
LFGCC_DASHBOARD!.bat

# Or manual launch
python monitoring/aac_master_monitoring_dashboard.py
```

Features:
- Real-time system health monitoring
- Multi-department performance metrics
- Security status dashboard
- Trading activity visualization
- Interactive charts and analytics

## 🤖 AZ Executive Assistant

The AZ Executive Assistant is AAC's revolutionary AI-powered strategic guidance system featuring 45 carefully crafted questions across 8 critical business categories.

### Strategic Question Categories

1. **📈 Market Analysis** (6 questions)
   - Market trend assessment and forecasting
   - Competitive landscape analysis
   - Customer behavior insights
   - Industry disruption identification
   - Regulatory impact evaluation
   - Technology adoption trends

2. **⚠️ Risk Assessment** (6 questions)
   - Operational risk evaluation
   - Financial risk modeling
   - Cybersecurity threat analysis
   - Compliance risk identification
   - Strategic risk assessment
   - Reputation risk management

3. **🎯 Strategy Optimization** (6 questions)
   - Business model innovation
   - Competitive positioning
   - Resource allocation optimization
   - Growth strategy development
   - Market expansion planning
   - Partnership and alliance strategy

4. **💻 Technology Integration** (5 questions)
   - Digital transformation roadmap
   - AI/ML implementation strategy
   - Cloud migration planning
   - Cybersecurity framework
   - Data analytics and BI strategy

5. **⚖️ Compliance & Regulation** (6 questions)
   - Regulatory compliance framework
   - Industry standards adherence
   - Data privacy and protection
   - Ethical business practices
   - Governance and oversight
   - Audit and reporting requirements

6. **📊 Performance Metrics** (5 questions)
   - KPI development and tracking
   - Performance measurement systems
   - Benchmarking and comparison
   - ROI and value creation metrics
   - Continuous improvement frameworks

7. **🚀 Innovation & Research** (6 questions)
   - Innovation pipeline management
   - R&D investment strategy
   - Technology scouting and evaluation
   - Intellectual property strategy
   - Market research and insights
   - Future trends and forecasting

8. **🛡️ Crisis Management** (5 questions)
   - Crisis preparedness planning
   - Business continuity strategy
   - Emergency response protocols
   - Stakeholder communication
   - Recovery and resilience planning

### Technical Features

- **Avatar Animation**: Real-time facial expressions using OpenCV
- **Audio Integration**: Text-to-speech responses with pyttsx3
- **Interactive Interface**: Streamlit-powered navigation
- **Comprehensive Framework**: Institutional-grade strategic guidance
- **Real-time Processing**: Live avatar animation and audio feedback

### Launch Commands

```bash
# Launch AZ Assistant with full system
LFGCC!.bat

# Launch AZ Assistant only
python core/aac_master_launcher.py --az-assistant

# Access via Matrix Monitor dashboard
LFGCC_DASHBOARD!.bat
```
# 🚀 UNIFIED SYSTEM LAUNCH (Recommended)
# Launch complete AAC Matrix Monitor system
python core/aac_master_launcher.py --mode paper    # Paper trading (default)
python core/aac_master_launcher.py --mode live     # Live trading (CAUTION!)
python core/aac_master_launcher.py --mode dry-run  # Dry run mode

# 🔍 COMPONENT-SPECIFIC LAUNCH
python core/aac_master_launcher.py --az-assistant     # AZ Executive Assistant only
python core/aac_master_launcher.py --dashboard-only   # Matrix Monitor dashboard only
python core/aac_master_launcher.py --agents-only      # Department agents only
python core/aac_master_launcher.py --trading-only     # Trading systems only

# 📊 MONITORING ONLY
python core/aac_master_launcher.py --monitoring-only   # Full monitoring system
python core/aac_master_launcher.py --service-only      # Background service only
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

## 📁 Project Structure

### AAC Matrix Monitor Directory Structure
```
aac-matrix-monitor/
├── core/                          # Core orchestration system
│   ├── aac_master_launcher.py     # Master system launcher with browser auto-open
│   ├── main.py                    # Legacy entry point (deprecated)
│   ├── orchestrator.py            # System orchestrator
│   └── command_center.py          # Command center interface
├── monitoring/                    # Matrix Monitor dashboard system
│   ├── aac_master_monitoring_dashboard.py  # Streamlit dashboard (auto-opens browser)
│   ├── continuous_monitoring.py   # Background monitoring service
│   └── security_dashboard.py      # Security monitoring interface
├── agents/                        # AI agent systems
│   ├── aac_agent_consolidation.py # Agent consolidation system
│   ├── avatar_system.py           # AZ Executive Assistant avatar system
│   ├── aac_az_questions_100.json  # AZ strategic questions database
│   └── master_agent_file.py       # Master agent orchestration
├── BigBrainIntelligence/          # Advanced AI research agents
│   ├── agents.py                  # 20 specialized research agents
│   ├── research_agent.py          # Agent base classes
│   └── requirements.txt           # AI-specific dependencies
├── CentralAccounting/             # Financial analysis engine
│   ├── database.py                # SQLite financial database
│   └── financial_analysis_engine.py # Financial analytics
├── ComplianceArbitrageDivision/   # Regulatory compliance
├── CorporateBankingDivision/      # Institutional banking
├── CryptoIntelligence/            # Cryptocurrency analysis
├── HR_Division/                   # Human resources management
├── InternationalInsuranceDivision/ # Global insurance operations
├── LudwigLawDivision/             # Legal compliance division
├── OptionsArbitrageDivision/      # Options trading strategies
├── PaperTradingDivision/          # Risk-free strategy testing
├── PortfolioManagementDivision/   # Asset allocation optimization
├── QuantitativeArbitrageDivision/ # Statistical arbitrage models
├── QuantitativeResearchDivision/  # Research and modeling
├── RiskManagementDivision/        # Risk assessment and mitigation
├── StatisticalArbitrageDivision/  # Statistical trading strategies
├── StructuralArbitrageDivision/   # Cross-market arbitrage
├── TechnologyArbitrageDivision/   # Technology sector opportunities
├── TechnologyInfrastructureDivision/ # System administration
├── strategies/                    # Trading strategy implementations
│   ├── strategy_agent_master_mapping.py
│   └── 50+ individual strategy files
├── trading/                       # Trading execution systems
│   ├── aac_arbitrage_execution_system.py
│   ├── binance_trading_engine.py
│   └── live_trading_environment.py
├── integrations/                  # External API integrations
│   ├── api_integration_hub.py
│   ├── market_data_aggregator.py
│   └── coinbase_api_async.py
├── shared/                        # Shared utilities and libraries
│   ├── config_loader.py           # Configuration management
│   ├── data_sources.py            # Market data sources
│   ├── utils.py                   # CircuitBreaker, RateLimiter
│   ├── monitoring.py              # Health checks & alerts
│   ├── secrets_manager.py         # API key encryption
│   ├── audit_logger.py            # Compliance logging
│   └── health_server.py          # HTTP health endpoints
├── config/                        # Configuration files
│   ├── alert_rules.yml            # Monitoring alert rules
│   └── ab_test_example.json       # A/B testing configuration
├── tools/                         # Utility tools
├── tests/                         # Test suite
├── docs/                          # Documentation
├── scripts/                       # Automation scripts
├── data/                          # Data files and samples
├── logs/                          # Log files
├── reports/                       # Report files and metrics
├── temp/                          # Temporary files
├── archive/                       # Deprecated/orphaned files
├── assets/                        # Static assets
├── models/                        # ML models
├── demos/                         # Demonstration files
├── reddit/                        # Reddit integration
├── deployment/                    # Deployment configurations
├── k8s/                          # Kubernetes manifests
└── automation/                    # GitHub automation scripts
    ├── aac_github_setup.bat       # Windows GitHub setup automation
    └── aac_github_setup.ps1       # PowerShell GitHub setup automation
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

## � GitHub Automation Setup

AAC Matrix Monitor includes automated GitHub repository setup and deployment scripts for seamless version control and collaboration.

### Automated Setup Scripts

#### Windows Batch Script (`aac_github_setup.bat`)
```batch
# Automated GitHub setup and commit
aac_github_setup.bat
```

#### PowerShell Script (`aac_github_setup.ps1`)
```powershell
# Enhanced PowerShell automation with GitHub integration
.\aac_github_setup.ps1 -GitHubUsername your_username

# Skip remote setup if needed
.\aac_github_setup.ps1 -SkipRemoteSetup
```

### Setup Process

1. **Create GitHub Repository**:
   - Go to https://github.com/new
   - Repository name: `aac-matrix-monitor`
   - Make it **PRIVATE** (recommended for financial systems)
   - **DO NOT** initialize with README, .gitignore, or license

2. **Run Automation Script**:
   ```batch
   # Windows
   aac_github_setup.bat

   # Or PowerShell with username
   .\aac_github_setup.ps1 -GitHubUsername your_github_username
   ```

3. **Complete Remote Setup** (if not using PowerShell script):
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/aac-matrix-monitor.git
   git push -u origin main
   ```

### What the Automation Does

- ✅ **Git Initialization**: Ensures repository is properly initialized
- ✅ **File Staging**: Adds all AAC system files to git
- ✅ **Comprehensive Commit**: Creates detailed commit with all system features
- ✅ **Remote Setup**: Configures GitHub remote (PowerShell script)
- ✅ **Push to GitHub**: Deploys complete system to repository
- ✅ **Documentation**: Includes professional commit messages and setup guidance

### Commit Message Includes

- 🚀 Major features (Matrix Monitor, AZ Assistant, Department Architecture)
- 📊 Dashboard capabilities and real-time monitoring
- 🤖 AI components and avatar animation system
- 🏛️ All 15 department divisions
- 🔧 Technical improvements and security features
- 📈 Performance and reliability metrics

## �📜 License

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
