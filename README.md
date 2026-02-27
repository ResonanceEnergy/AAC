# Accelerated Arbitrage Corp (AAC) — NCC Trading System

[![CI](https://github.com/ResonanceEnergy/AAC/actions/workflows/ci.yml/badge.svg)](https://github.com/ResonanceEnergy/AAC/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: Proprietary](https://img.shields.io/badge/license-proprietary-red.svg)](LICENSE)

Enterprise-grade algorithmic trading platform with 69 arbitrage strategies, ML/AI signal generation, multi-exchange execution, and full audit trail.

---

## Quick Start

```bash
# 1. Clone & enter repo
git clone https://github.com/ResonanceEnergy/AAC.git && cd AAC

# 2. Create virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.template .env
# Edit .env — add API keys, DB URL, Kafka/Redis hosts

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
│   ├── models/               # ML models (LSTM, PPO, NLP)
│   ├── integrations/         # Exchange + data connectors
│   ├── monitoring/           # Health, metrics, alerts
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
| IBKR         | ib_insync | ✅   | ✅      | ✅    |

---

## Strategy Library (69 strategies)

Strategy categories under `src/aac/strategies/`:

- **Statistical Arbitrage** — pairs trading, cointegration, mean reversion
- **Cross-Exchange Arbitrage** — latency, triangular, funding rate
- **Dispersion & Correlation** — active dispersion, correlation risk premium
- **Flow & Macro** — flow pressure, real economy feedback, credit-equity
- **Structural** — basis trading, roll yield, term structure
- **ML-Driven** — LSTM forecasting, PPO reinforcement learning, NLP sentiment

---

## ML / AI Stack

| Component          | Framework               | Purpose                    |
|--------------------|-------------------------|----------------------------|
| Price Forecasting  | PyTorch LSTM            | 15-min ahead price signals |
| RL Execution       | Stable-Baselines3 PPO   | Order routing optimization |
| Sentiment NLP      | HuggingFace Transformers| News/social signal alpha   |
| Big Brain          | Ensemble (XGBoost+DL)   | Multi-strategy arbitration |
| Grok Agent         | BigBrainIntelligence    | Meta-learning signal fusion|

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

*AAC is a proprietary system. Unauthorized use or distribution is prohibited.*
