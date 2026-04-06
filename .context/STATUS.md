# AAC Living Status Dashboard

> **Last updated:** 2026-04-06 08:45 ET
> **Updated by:** Full alignment — test suite validated (1712p), lint clean, position/date audit, context sync
> **Update this file** after every significant change. This is the single source of truth for what works.

---

## System Health — What Works

| Component | Status | Notes |
|---|---|---|
| **IBKR Connector** | LIVE | Port 7496, account U24346218. KRE/IWM expired Apr 4. 7 active positions. |
| **Moomoo Connector** | LIVE | OpenD, FUTUCA, real mode, $365.15 USD. Options approval still pending. |
| **yfinance** | WORKING | Free, primary options chain source |
| **CoinGecko** | DEGRADED | Pro key expired → free tier (10 req/min). Prices work. |
| **Unusual Whales** | WORKING | Key valid, connection works. Field parsing FIXED (2026-03-31). |
| **FRED** | WORKING | VIX fallback, macro data |
| **Finnhub** | WORKING | Quotes, news |
| **NewsAPI** | WORKING | Headlines |
| **Doctrine Engine** | WORKING | 11 packs, 4-state machine |
| **Matrix Monitor** | WORKING | Parallel collection (5s timeout/collector), degradation panel, ASCII banner. Confirmed live 2026-04-02: 24/30 collectors OK, 6 timeout (expected — IBKR/NCC/NCL offline). |
| **War Room Engine** | WORKING | Wired to integrator. Regime: STAGFLATION (70%), Vol Shock 40/100. |
| **Polymarket Division** | ACTIVE | py-clob-client v0.34.6, wallet live ($535.73 USDC), active_scanner.py (450+ lines), `launch.py polymarket` |
| **Paper Trading** | WORKING | `launch.py paper` |
| **Web Dashboard** | WORKING | `launch.py dashboard` |
| **CI Pipeline** | WORKING | `.github/workflows/ci.yml` |
| **Pytest Suite** | WORKING | **1712 passed**, 23 skipped, 1 xfailed (2026-04-06) |

## What's Broken

| Component | Problem | Priority | Notes |
|---|---|---|---|
| **CoinGecko Pro** | Key expired, returns 403 | LOW | Auto-downgrades to free tier. Works fine for now. |
| **Polygon Options** | Free tier: 403 on snapshots | LOW | Needs $79/mo upgrade. Not blocking. |
| **X/Twitter API** | HTTP 402 | LOW | Needs paid tier. Graceful fallback to 0.5. |
| **NDAX** | LIQUIDATED | NONE | All crypto sold. Connector exists but unused. |

## Active Positions (Real Money)

### IBKR (Account U24346218)

| Ticker | Strike | Qty | Entry | Expiry | Status |
|--------|--------|-----|-------|--------|--------|
| ARCC | $17P | 1 | $0.25 | Apr 17 | **ROLLING DECISION Apr 10** |
| PFF | $29P | 1 | $0.40 | Apr 17 | Let expire (down -92%) |
| MAIN | $50P | 1 | $0.85 | Apr 17 | Roll or exit decision due |
| JNK | $92P | 1 | $0.80 | Apr 17 | Hold/roll decision |
| XLF | $46P | 1 | $0.75 | May 1 | Monitor |
| LQD | $106P | 1 | $0.64 | May 15 | Performing (+71%) |
| EMB | $90P | 1 | $0.75 | May 15 | Performing |
| BKLN | $20P | 3 | $0.40 | Jun 18 | Hold |
| HYG | $77P | 1 | $0.80 | Jun 18 | Hold |
| ~~KRE~~ | ~~$58P~~ | ~~1~~ | ~~$1.45~~ | ~~Apr 4~~ | **EXPIRED** |
| ~~IWM~~ | ~~$230P~~ | ~~1~~ | ~~$3.96~~ | ~~Apr~~ | **EXPIRED** |

### WealthSimple TFSA ($18,637.76 CAD as of Mar 29)

| Ticker | Strike | Qty | Expiry | Status |
|--------|--------|-----|--------|--------|
| ARCC | $16P | 10 | Apr 17 | **Roll to Jun $15P on Apr 10** |
| OBDC | $10P | 65 | Apr 17 | **Roll to Jul $7.5P on Apr 10** (+17%) |
| JNK | $94P | 5 | Apr 17 | **Roll 2, let 3 expire on Apr 10** |
| KRE | $60P | 1 | Apr 17 | **Close for ~$94 credit on Apr 10** |
| GLD | $515C | 1 | Mar 19, 2027 | LEAPS hold (+25%) |
| OWL | $8P | 5 | Jun 18 | Hold |
| XLE | $85C | 26 | Jan 15, 2027 | LEAPS hold (+162%) |

### Other Venues

| Venue | Value | Notes |
|-------|-------|-------|
| Moomoo | $2,609.26 USD | SQQQ x172, SPXS x106, GLD $298C, SLV $33.5C x7 |
| Polymarket | $535.73 USDC | ACTIVE — 3 unified strategies (War Room, PolyMC, PlanktonXD), active_scanner.py, 2 open orders, 124 trades |
| NDAX | LIQUIDATED | $4,492 CAD withdrawn |

## Critical Dates

| Date | Event | Action |
|------|-------|--------|
| **Apr 4** | KRE $58P / IWM $230P expiry | **EXPIRED** (confirmed) |
| **Apr 10** | **WS TFSA 7-DTE roll window** | **4 DAYS AWAY** — Execute rolls per `docs/APR10_ROLL_EXECUTION_PLAN.md` |
| **Apr 17** | **IBKR + WS Apr OPEX** | Let PFF expire, manage ARCC/MAIN/JNK |
| May 1 | XLF $46P expiry | Monitor |
| May 15 | LQD/EMB expiry | Monitor (performing) |
| Jun 18 | BKLN x3, HYG, OWL expiry | Monitor |

## Active Workstreams

| Workstream | Status | Notes |
|---|---|---|
| Monitoring Overhaul | **DONE** | Parallel collectors, degradation panel, ASCII banner, 5s timeouts. Live-tested 2026-04-02. |
| 3,297 Quality Fixes | **DONE** | ruff clean, lint zero, F821/F402/F601/E741/E701 all fixed. |
| Context Guardrails | DONE | copilot-instructions.md, AGENTS.md, STATUS.md, 3 path-specific |
| Root Cleanup + Archives | DONE | 86 files archived, root 170+ → 30 |
| **Apr 10 Roll Execution** | **UPCOMING** | WS TFSA roll-down, ~C$614 budget. See `docs/APR10_ROLL_EXECUTION_PLAN.md` |
| **Polymarket Division Activation** | **DONE** | active_scanner.py (450+ lines), 3 strategies unified, launch.py mode, DRY_RUN=true default |
| Architecture Rework v3.3 | Phase 1-2 DONE | Phase 3-7 planned |
| Moomoo Options Approval | WAITING | Applied ~Mar 15, still pending |

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
