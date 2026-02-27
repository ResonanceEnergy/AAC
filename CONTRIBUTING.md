# Contributing to AAC

## Quick Start

```bash
# Clone and set up
git clone https://github.com/ResonanceEnergy/AAC.git
cd AAC
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"

# Copy env template
cp .env.template .env

# Run tests
make test          # fast tests (no network)
make test-all      # all tests including slow
```

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and add tests
3. Run linting: `make lint`
4. Run tests: `make test`
5. Commit with conventional commits: `fix:`, `feat:`, `chore:`, `refactor:`
6. Push and open a PR against `main`

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `shared/` | Core infrastructure (audit, monitoring, connectors) |
| `strategies/` | 70+ trading strategy implementations |
| `TradingExecution/` | Order execution engine |
| `BigBrainIntelligence/` | AI/ML agent framework |
| `CentralAccounting/` | Financial accounting + P&L |
| `CryptoIntelligence/` | Crypto-specific intelligence |
| `core/` | Orchestrator, risk engine, launcher |
| `monitoring/` | Dashboard and health checks |
| `tests/` | Test suite (pytest) |
| `src/aac/divisions/` | Business unit modules |
| `aac/NCC/Doctrine/` | Governance doctrine docs |

## Testing

```bash
# Run fast tests only
pytest tests/ -m "not live and not exchange and not slow" --timeout=15

# Run specific test file
pytest tests/test_suite.py -q --tb=short

# Run with coverage
make test-cov
```

### Test Markers

| Marker | Meaning |
|--------|---------|
| `@pytest.mark.live` | Requires live API keys |
| `@pytest.mark.exchange` | Requires exchange connection |
| `@pytest.mark.slow` | Takes >5 seconds (network calls) |
| `@pytest.mark.integration` | Full integration test |
| `@pytest.mark.paper` | Paper-trading mode test |

## Code Style

- **Formatter**: Black (line-length 100)
- **Import sort**: isort (black profile)
- **Linter**: Ruff + Flake8
- **Type checking**: mypy (ignore_missing_imports)

Pre-commit hooks enforce these automatically:
```bash
pip install pre-commit
pre-commit install
```

## Environment Variables

See `.env.template` for all supported variables. Key ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `AAC_ENV` | `production` | Environment mode |
| `PAPER_TRADING` | `true` | Enable paper trading |
| `LIVE_TRADING_ENABLED` | `false` | Enable live trading |

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `chore:` — maintenance
- `refactor:` — code restructure
- `docs:` — documentation only
- `test:` — test changes only
