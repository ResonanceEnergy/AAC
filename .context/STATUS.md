# AAC Living Status Dashboard

> **Last updated:** 2026-06-07
> **Updated by:** Full codebase audit execution (Phases 0-7)
> **Update this file** after every significant change. This is the single source of truth for what works.

---

## System Health — What Works

| Component | Status | Notes |
|---|---|---|
| **IBKR Connector** | LIVE | Port 7496, account U24346218, 8 live put positions |
| **Moomoo Connector** | LIVE | OpenD, FUTUCA, real mode, $365.15 USD |
| **yfinance** | WORKING | Free, primary options chain source |
| **CoinGecko** | DEGRADED | Pro key expired → free tier (10 req/min). Prices work. |
| **Unusual Whales** | WORKING | Key valid, connection works. Field parsing FIXED (2026-03-31). Headers, URLs, field mappings updated. |
| **FRED** | WORKING | VIX fallback, macro data |
| **Finnhub** | WORKING | Quotes, news |
| **NewsAPI** | WORKING | Headlines |
| **Doctrine Engine** | WORKING | 12 packs, 4-state machine |
| **Matrix Monitor** | WORKING | 4 display modes, 20+ panels |
| **War Room Engine** | WORKING | Wired to integrator |
| **Paper Trading** | WORKING | `launch.py paper` |
| **Web Dashboard** | WORKING | `launch.py dashboard` |
| **CI Pipeline** | WORKING | `.github/workflows/ci.yml` |
| **Pytest Suite** | WORKING | 1671 passed, 23 skipped, 0 failures (after 7-phase audit) |

## What's Broken

| Component | Problem | Priority | Notes |
|---|---|---|---|
| ~~UW Field Parsing~~ | ~~FIXED 2026-03-31~~ | DONE | Headers, URLs, version, field mappings all updated. 31 tests passing. |
| **CoinGecko Pro** | Key expired, returns 403 | LOW | Auto-downgrades to free tier. Works fine for now. |
| **Polygon Options** | Free tier: 403 on snapshots | LOW | Needs $79/mo upgrade. Not blocking. |
| **X/Twitter API** | HTTP 402 | LOW | Needs paid tier. Graceful fallback to 0.5. |
| **NDAX** | LIQUIDATED | NONE | All crypto sold. Connector exists but unused. |

## Active Positions (Real Money)

| Venue | Positions | Expiry | Notes |
|---|---|---|---|
| IBKR | 8 puts (ARCC, PFF, LQD, EMB, MAIN, JNK, KRE, IWM) | Apr/Jul 2026 | $910 total invested |
| WealthSimple TFSA | Roll-down planned | Apr→Jul | See `APR10_ROLL_EXECUTION_PLAN.md` |
| Moomoo | No options yet | — | Options approval pending |

## Active Workstreams

| Workstream | Status | Owner | Notes |
|---|---|---|---|
| Context Guardrails | DONE | — | copilot-instructions.md, AGENTS.md, STATUS.md, 3 path-specific instructions |
| UW Field Parsing Fix | DONE | — | Fixed 2026-03-31: headers, URLs, version, field mappings. 31 tests. |
| Root Directory Cleanup | DONE | — | 170+ → 31 root files. 99→`_scratch/`, 22→`docs/archive/` |
| Architecture Rework v3.3 | Phase 1-2 DONE | — | Phase 3-7 planned (see `AAC_ARCHITECTURE_REWORK_PLAN.md`) |
| Full Codebase Audit | COMPLETE | — | 593 files audited, 225K LOC. 7-phase remediation executed. 86 files archived, 3 runtime bombs fixed, 10 silent exception handlers fixed, test suite hardened. 1671 tests passing. |
| WS TFSA Roll-Down | PLANNED | — | Plan in `APR10_ROLL_EXECUTION_PLAN.md`, ~C$614 budget |

## Known Test Issues

- `test_autonomous.py` — times out (>180s), pre-existing, always skip
- Tests marked `@pytest.mark.api` — auto-skipped by default (real HTTP calls). Run with `pytest -m api` to validate API keys.
- `conftest.py` at root handles shared fixtures + auto-skip hook

## Environment

| Setting | Value |
|---|---|
| DRY_RUN | false |
| PAPER_TRADING | false |
| LIVE_TRADING_ENABLED | true |
| AAC_ENV | production |
| Python | 3.12 via `.venv\Scripts\python.exe` |
| Linter | ruff (configured in pyproject.toml) |

## Key Files Quick Reference

| Need to... | Look at... |
|---|---|
| Understand the project | `.github/copilot-instructions.md`, then `.context/01_overview/current-state.md` |
| Check agent behavior rules | `AGENTS.md` (root) |
| See system architecture | `.context/02_architecture/system-map.md` |
| Find API clients | `integrations/`, `shared/data_sources.py` |
| Check active work | `.context/04_workstreams/active-workstreams.md` |
| Run UW operations | `.context/08_runbooks/unusual-whales-integration.md` |
| Developer workflow | `.context/08_runbooks/developer-workflow.md` |
| Run tests | `.context/09_tests/test-commands.md` |
| Read gap analysis | `.context/07_gaps/gap-scan-2026-03-16.md` |
| Launch the system | `python launch.py <mode>` |
