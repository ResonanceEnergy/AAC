# AAC — Copilot Context Guardrails

> **Trust these instructions first. Only search the codebase if something here is incomplete or wrong.**

## Identity

**Accelerated Arbitrage Corp (AAC)** — codename BARREN WUFFET / AZ SUPREME.
AI-powered algorithmic trading platform. Part of Resonance Energy's NCC quad (NCL=Brain, SuperAgency=Agency, AAC=Bank, DigitalLabour=Labour).

- Repo: `C:\dev\AAC_fresh` (branch: main)
- Python: 3.12 via `.venv\Scripts\python.exe` — requires `>=3.9,<3.14`
- Version: 3.6.0 (pyproject.toml is source of truth)
- Owner: ResonanceEnergy org on GitHub

## Directory Map — Where Things Go

| Directory | Purpose | Touch? |
|---|---|---|
| `aac/` | Core platform package (doctrine, NCC, divisions) | Careful |
| `core/` | Command center, orchestration, state mgmt | Careful |
| `shared/` | Cross-cutting infra, data sources, monitoring frameworks | Careful |
| `strategies/` | Strategy algorithms, execution engines, war room | Careful |
| `TradingExecution/` | Exchange connectors (IBKR, Moomoo, NDAX) | Careful |
| `integrations/` | External API clients (UW, CoinGecko, Finnhub, etc.) | Normal |
| `BigBrainIntelligence/` | Research & analysis agents | Normal |
| `CryptoIntelligence/` | Crypto analytics, venue health, on-chain | Normal |
| `CentralAccounting/` | P&L, position tracking, database | Normal |
| `monitoring/` | Dashboards, continuous monitoring loops | Normal |
| `agents/` | Agent workflows and simulations | Normal |
| `SharedInfrastructure/` | Infra utilities | Normal |
| `config/` | YAML configs, doctrine packs | Normal |
| `data/` | Storyboard, static data, local DBs | Read-mostly |
| `tests/` | Pytest suite | Normal |
| `tools/` | Validation scripts, exporters | Normal |
| `scripts/` | Operational scripts | Normal |
| `_scratch/` | Temporary diagnostic scripts — disposable | Free |
| `_archive/` | Archived/dead code | Don't touch |
| `.context/` | Durable project context (10-folder system) | Update when relevant |
| `.github/` | CI, copilot instructions, guardrails | Update carefully |
| `docs/` | Documentation | Normal |

### File Placement Rules
- **New diagnostic/one-off scripts** → `_scratch/` (prefixed with underscore)
- **New strategies** → `strategies/`
- **New API clients** → `integrations/`
- **Operational docs** → `.context/` appropriate subfolder
- **NEVER put temp scripts at project root** — the root is cluttered enough

## API Inventory — What Works Right Now

| API | Status | Key Location | Notes |
|---|---|---|---|
| **yfinance** | WORKING (Free) | No key needed | Primary options chain source |
| **CoinGecko** | WORKING (Free tier) | `.env` COINGECKO_API_KEY | Pro key expired → auto-downgrades. 10 req/min |
| **Unusual Whales** | PARTIAL | `.env` UNUSUAL_WHALES_API_KEY | Connection works. Field parsing BROKEN (strike/premium/$0). Schema changed. |
| **FRED** | WORKING | `.env` FRED_API_KEY | VIX fallback, macro data |
| **Finnhub** | WORKING | `.env` FINNHUB_API_KEY | Quotes, news |
| **Polygon** | LIMITED | `.env` POLYGON_API_KEY | Free tier: NO options snapshots (403). Needs $79/mo |
| **IBKR** | WORKING | Port 7496 live, TWS | Account U24346218. 8 live put trades executed |
| **Moomoo** | WORKING | OpenD, FUTUCA | Real mode, $365.15 USD, trade PIN=069420 |
| **NDAX** | LIQUIDATED | `.env` | Sold all crypto. Connector uses ccxt (login+password+uid) |
| **Tradier** | NOT CONFIGURED | — | No key |
| **NewsAPI** | WORKING | `.env` NEWSAPI_KEY | Headlines |
| **X/Twitter** | BROKEN | `.env` X_BEARER_TOKEN | HTTP 402 — needs paid tier |

### API Rules
- **ALWAYS use internal API clients** from `integrations/` and `shared/data_sources.py` — NEVER use Barchart, external websites, or manual scraping when we have API access
- **ALWAYS check this table** before telling the user an API doesn't work
- When an API returns errors, check if the key is in `.env` first

## Build & Test

```powershell
# Activate venv
cd C:\dev\AAC_fresh
.venv\Scripts\activate

# Run tests (primary command)
.venv\Scripts\python.exe -m pytest --timeout=30 -q --ignore=tests/security_integration_test.py --ignore=tests/test_bridge_integration.py --ignore=tests/test_ecb_api.py --ignore=tests/test_market_data_quick.py --tb=short

# Lint
.venv\Scripts\python.exe -m ruff check .

# Single file test
.venv\Scripts\python.exe -m pytest tests/test_specific.py -v --timeout=30
```

### Known Test Issues
- `test_autonomous.py` — times out (>180s), pre-existing, skip it
- Tests making real HTTP calls cause env-dependent failures — always mock `urllib.request.urlopen`
- `conftest.py` at root handles shared fixtures

## Coding Conventions

- Async-first design with `asyncio` and `aiohttp`
- `structlog` for logging (use `_log` convention for logger instances)
- `from __future__ import annotations` MUST be first import in any file
- Type hints on public functions; use `dict[str, Any]` for mixed-type dicts
- Line length: configured in pyproject.toml (ruff)
- Narrowed exceptions — never `except Exception: pass`
- Guard clauses over deep nesting
- `python-dotenv` for env vars; keys live in `.env`, NEVER hardcoded

## Forbidden Patterns — Things That Keep Breaking

1. **Swallowing exceptions silently** — `except Exception: pass` hides real bugs. Always log or re-raise.
2. **sys.path.insert hacks** — already 113 of these. Don't add more. Use proper package imports.
3. **Using Barchart/external sites** when internal APIs exist — check the API inventory above.
4. **Creating files at project root** — use `_scratch/`, `scripts/`, or appropriate directory.
5. **pythonw.exe stdout crash** — if writing a service, add devnull guard: `if sys.stdout is None: sys.stdout = open(os.devnull, "w")`
6. **aiohttp c-ares DNS failure** — use `aiohttp.resolver.ThreadedResolver()` + `TCPConnector(resolver=...)` to force system DNS
7. **Forgetting to update .context/** — after significant work, update `.context/01_overview/current-state.md`
8. **Random behavior in operational code** — use deterministic slippage/scoring in non-simulation paths
9. **Committing .env or API keys** — GitHub Push Protection will block. Check `git diff` before commit.

## Context System

The durable context lives in `.context/` (10-folder system):
- `01_overview/current-state.md` — START HERE for project orientation
- `02_architecture/system-map.md` — subsystem relationships
- `03_inventory/` — directory map, external data sources
- `04_workstreams/active-workstreams.md` — what's being worked on
- `05_decisions/` — engineering decisions with rationale
- `07_gaps/` — gap scans and remediation status
- `08_runbooks/` — repeatable workflows (UW integration, developer workflow)

Also check:
- `AGENTS.md` (root) — behavioral rules for AI agents
- `.context/STATUS.md` — living status dashboard (what works/broken/active)
- `/memories/repo/` — concise high-signal facts

## Active Positions & Trading State

- IBKR: 8 live put positions (ARCC, PFF, LQD, EMB, MAIN, JNK, KRE, IWM) — Apr/Jul expiry
- WealthSimple TFSA: Roll-down plan in `APR10_ROLL_EXECUTION_PLAN.md`
- Moomoo: Real mode, options approval pending
- Crypto: NDAX liquidated ($4,492 CAD), no active crypto positions
- DRY_RUN=false, PAPER_TRADING=false, LIVE_TRADING_ENABLED=true

## Launch

```powershell
python launch.py dashboard   # Web monitoring dashboard
python launch.py monitor     # Terminal system monitor
python launch.py paper       # Paper trading engine
python launch.py live        # LIVE trading (caution!)
python launch.py git-sync    # Sync with remote
```

Single launcher rule: `launch.py` is THE launcher. See `.github/SINGLE_LAUNCHER_RULE.md`.
