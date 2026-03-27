# Accelerated Arbitrage Corp (AAC) — BARREN WUFFET Trading System

[![CI](https://github.com/ResonanceEnergy/AAC/actions/workflows/ci.yml/badge.svg)](https://github.com/ResonanceEnergy/AAC/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: Proprietary](https://img.shields.io/badge/license-proprietary-red.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.7.0-green.svg)](CHANGELOG.md)

**BARREN WUFFET** (codename: AZ SUPREME) — Enterprise-grade AI-powered algorithmic trading platform with 50+ arbitrage strategies, 35 OpenClaw skills, ML/AI signal generation, multi-exchange execution, Telegram bot integration, and full audit trail.

> *"Every byte of information is an edge. Store it. Search it. Trade on it."* — BARREN WUFFET

### Key Features
- **AAC Matrix Monitor** — Central command-and-control hub: hybrid dashboard / console / C2 center with 4 display modes (Terminal, Web, Dash, API), 20+ panels, 5 pillar endpoints, 9 elite trading desk components
- **35 OpenClaw Skills** — Market intelligence, trading, crypto/DeFi, banking, wealth building, advanced analysis
- **Telegram Bot** — `@barrenwuffet069bot` with NL intent detection and 7-department routing
- **Doctrine Engine** — 12 packs, 4-state machine (NORMAL → CAUTION → SAFE_MODE → HALT)
- **7 Departments** — Executive, BigBrainIntelligence, TradingExecution, CryptoIntelligence, CentralAccounting, SharedInfrastructure, Jonny Bravo Division
- **Scam Detection** — FrankenClaw patterns, token legitimacy checks
- **Deep Dive Research** — 1000+ lines of curated OpenClaw intelligence in doctrine memory

---

## Quick Start

```bash
# 1. Clone & enter repo
git clone https://github.com/ResonanceEnergy/AAC.git && cd AAC

# 2. Create virtual environment (Python 3.9+)
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.template .env
# Edit .env — add API keys, DB URL, Kafka/Redis hosts
# For higher-risk credentials, prefer *_FILE entries that point at ignored
# local files under secrets/ instead of placing raw values directly in .env

# 5. Launch
chmod +x launch.sh
./launch.sh paper          # paper trading mode (safe default)
```

---

## Architecture

```
AAC/
├── src/aac/                  # Core platform
│   ├── strategies/           # 69 arbitrage strategy modules
│   ├── trading/              # Order execution engine
│   ├── models/               # ML model training pipeline
│   ├── integrations/         # Exchange + data connectors
│   ├── monitoring/           # AAC Matrix Monitor — Central C2 Hub
│   ├── shared/               # Canonical shared infrastructure
│   └── divisions/            # 12 business unit stubs
├── TradingExecution/         # Exchange connector layer
├── CentralAccounting/        # P&L, position tracking
├── BigBrainIntelligence/     # AI signal aggregation
├── CryptoIntelligence/       # Crypto market analytics
├── agent_jonny_bravo_division/ # Agent-based execution
├── aac/NCC/NCC-Doctrine/     # Operational doctrine & config
├── docs/                     # Documentation
├── tests/                    # Pytest test suite
├── launch.sh                 # macOS/Linux launcher
├── .env.template             # Environment variable template
└── requirements.txt          # Python dependencies
```

---

## Exchange Support

| Exchange     | Connector | Spot | Futures | Paper |
|--------------|-----------|------|---------|-------|
| Binance      | ccxt      | ✅   | ✅      | ✅    |
| Coinbase Pro | ccxt      | ✅   | —       | ✅    |
| Kraken       | ccxt      | ✅   | ✅      | ✅    |
| IBKR         | TWS API   | ✅   | ✅      | ✅    |

---

## Strategy Library (50+ strategies)

Strategy categories under `src/aac/strategies/`:

- **Statistical Arbitrage** — pairs trading, cointegration, mean reversion
- **Cross-Exchange Arbitrage** — latency, triangular, funding rate
- **Dispersion & Correlation** — active dispersion, correlation risk premium
- **Flow & Macro** — flow pressure, real economy feedback, credit-equity
- **Structural** — basis trading, roll yield, term structure
- **ML-Driven** — ML training pipeline, signal aggregation (planned: LSTM, PPO, NLP)

---

## ML / AI Stack

| Component          | Framework               | Status      | Purpose                    |
|--------------------|-------------------------|-------------|----------------------------|
| Price Forecasting  | PyTorch LSTM            | Planned     | 15-min ahead price signals |
| RL Execution       | Stable-Baselines3 PPO   | Planned     | Order routing optimization |
| Sentiment NLP      | HuggingFace Transformers| Planned     | News/social signal alpha   |
| Big Brain          | Ensemble (XGBoost+DL)   | Implemented | Multi-strategy arbitration |
| Grok Agent         | BigBrainIntelligence    | Implemented | Meta-learning signal fusion|
| ML Pipeline        | scikit-learn            | Implemented | Model training & evaluation|

---

## Safety Controls

- `LIVE_TRADING_ENABLED=false` by default — must be explicitly set to `true`
- `PAPER_TRADING=true` in all CI and dev environments
- Pre-commit hooks block commits with secrets (`detect-private-key`)
- `.env` permanently gitignored — never tracked
- CI security scan: `pip-audit` + `trufflehog` on every push
- Audit trail: tamper-evident log via `shared/audit_logger.py`

---

## Testing

```bash
# All tests (paper mode, no live exchange calls)
./launch.sh test

# Or directly:
.venv/bin/python -m pytest tests/ -q --tb=short -m "not live and not exchange"

# With coverage:
.venv/bin/python -m pytest tests/ --cov=src/aac --cov-report=html
```

---

## Configuration

All configuration via environment variables. See `.env.template` for the full list.

Key variables:

| Variable               | Default        | Description                    |
|------------------------|----------------|--------------------------------|
| `PAPER_TRADING`        | `true`         | Enables paper trading mode     |
| `LIVE_TRADING_ENABLED` | `false`        | Must be `true` for live orders |
| `AAC_ENV`              | `development`  | Runtime environment            |
| `BINANCE_API_KEY`      | —              | Binance API key                |
| `DATABASE_URL`         | `sqlite://`    | Database connection string     |
| `REDIS_URL`            | `redis://localhost:6379` | Redis broker        |

---

## Development

```bash
# Install pre-commit hooks (run once after clone)
pip install pre-commit
pre-commit install

# Lint manually
black src/ tests/
isort src/ tests/
flake8 src/ tests/

# Type check
mypy src/aac/
```

---

## Launch Modes

```bash
./launch.sh dashboard   # Dash web UI on :8050
./launch.sh monitor     # System health monitor
./launch.sh paper       # Paper trading (no real orders)
./launch.sh core        # Core engine only
./launch.sh full        # All services
./launch.sh test        # Run test suite
```

---

## Risk Disclosure

> **IMPORTANT**: This software is for educational and research purposes.
> Trading cryptocurrencies, equities, options, and prediction markets involves
> substantial risk of loss and is not suitable for every investor.
>
> - Past performance does not guarantee future results.
> - Paper trading results may not reflect real market conditions.
> - Never trade with funds you cannot afford to lose.
> - AI-generated signals are not financial advice.
> - Conduct your own due diligence before making any investment decisions.
>
> The authors and contributors accept no liability for financial losses
> incurred through the use of this software.

---

*AAC / BARREN WUFFET is a proprietary system. Unauthorized use or distribution is prohibited.*
