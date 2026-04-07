# AAC Living Status Dashboard

> **Last updated:** 2026-04-07 (Monday, late)
> **Updated by:** Paper trading fill fix (market orders always fill — deterministic). Architecture v3.3 Phase 3-7 assessed. 1715 tests green, 0 failed.
> **Update this file** after every significant change. This is the single source of truth for what works.

---

## System Health — What Works

| Component | Status | Notes |
|---|---|---|
| **IBKR Connector** | LIVE | Port 7496, account U24346218. 5 active puts (all OTM). 4 Apr puts expired worthless. |
| **Moomoo Connector** | LIVE | OpenD, FUTUCA, real mode, $365.15 USD. Options approval still pending. |
| **yfinance** | WORKING | Free, primary options chain source |
| **CoinGecko** | DEGRADED | Pro key expired → free tier (10 req/min). Prices work. |
| **Unusual Whales** | WORKING | Key valid, connection works. Field parsing FIXED (2026-03-31). |
| **FRED** | WORKING | VIX fallback, macro data |
| **Finnhub** | WORKING | Quotes, news |
| **NewsAPI** | WORKING | Headlines |
| **Doctrine Engine** | WORKING | 11 packs, 4-state machine |
| **Matrix Monitor** | WORKING | Parallel collection (5s timeout/collector), degradation panel, ASCII banner. Confirmed live 2026-04-02: 24/30 collectors OK, 6 timeout (expected — IBKR/NCC/NCL offline). |
| **War Room Engine** | WORKING | Wired to integrator. Regime: STAGFLATION (70%), Vol Shock 40/100. ROLL_DISCIPLINE rules added Apr 6. |
| **Polymarket Division** | ACTIVE | py-clob-client v0.34.6, wallet live ($535.73 USDC), active_scanner.py (450+ lines), `launch.py polymarket` |
| **Paper Trading** | WORKING | `launch.py paper` |
| **Web Dashboard** | WORKING | `launch.py dashboard` |
| **CI Pipeline** | WORKING | `.github/workflows/ci.yml` |
| **Pytest Suite** | WORKING | **1715 passed**, 0 failed, 23 skipped, 1 xfailed (2026-04-07 — paper fill fix) |

## What's Broken

| Component | Problem | Priority | Notes |
|---|---|---|---|
| **CoinGecko Pro** | Key expired, returns 403 | LOW | Auto-downgrades to free tier. Works fine for now. |
| **Polygon Options** | Free tier: 403 on snapshots | LOW | Needs $79/mo upgrade. Not blocking. |
| **X/Twitter API** | HTTP 402 | LOW | Needs paid tier. Graceful fallback to 0.5. |
| **NDAX** | LIQUIDATED | NONE | All crypto sold. Connector exists but unused. |

## Active Positions (Real Money) — Updated Apr 7

### IBKR (Account U24346218) — ~$920 equity

| Ticker | Strike | Qty | Entry | Expiry | Spot | Status |
|--------|--------|-----|-------|--------|------|--------|
| XLF | $46P | 1 | $0.75 | May 1 | $49.90 | **21-DTE roll decision Apr 10** |
| LQD | $106P | 1 | $0.63 | May 15 | $109.11 | OTM, monitor |
| EMB | $90P | 1 | $0.48 | May 15 | $93.96 | OTM, monitor |
| BKLN | $20P | 3 | $0.40 | Jun 18 | $20.49 | Near ATM, hold |
| HYG | $77P | 1 | $0.80 | Jun 18 | $79.61 | OTM, hold |
| ~~ARCC~~ | ~~$17P~~ | ~~1~~ | ~~$0.25~~ | ~~Apr 17~~ | | **EXPIRING WORTHLESS** ($0 bid) |
| ~~PFF~~ | ~~$29P~~ | ~~1~~ | ~~$0.17~~ | ~~Apr 17~~ | | **EXPIRING WORTHLESS** ($0 bid) |
| ~~MAIN~~ | ~~$50P~~ | ~~1~~ | ~~$0.73~~ | ~~Apr 17~~ | | **EXPIRING WORTHLESS** ($0 bid) |
| ~~JNK~~ | ~~$92P~~ | ~~1~~ | ~~$0.35~~ | ~~Apr 17~~ | | **EXPIRING WORTHLESS** ($0 bid) |
| ~~KRE~~ | ~~$58P~~ | ~~1~~ | ~~$1.45~~ | ~~Apr 4~~ | | EXPIRED |
| ~~IWM~~ | ~~$230P~~ | ~~1~~ | ~~$3.96~~ | ~~Apr 4~~ | | EXPIRED |

**IBKR Apr expired premium lost: -$150** (ARCC $25 + PFF $17 + MAIN $73 + JNK $35)

### WealthSimple TFSA (~$18,638 CAD / ~$13,398 USD)

| Ticker | Strike | Qty | Entry | Expiry | Spot | Status |
|--------|--------|-----|-------|--------|------|--------|
| GLD | $515C | 1 | $19.40 | Mar 2027 | $428.66 | LEAPS — OTM, hold (time value) |
| XLE | $85C | 26 | $0.37 | Jan 2027 | $59.37 | LEAPS — deep OTM, hold |
| OWL | $8P | 5 | $0.75 | Jun 18 | $8.55 | Near ATM, hold |
| ~~ARCC~~ | ~~$16P~~ | ~~10~~ | ~~$0.13~~ | ~~Apr 17~~ | | **EXPIRING WORTHLESS** ($0 bid) |
| ~~JNK~~ | ~~$94P~~ | ~~5~~ | ~~$0.57~~ | ~~Apr 17~~ | | **EXPIRING WORTHLESS** ($0 bid) |
| ~~KRE~~ | ~~$60P~~ | ~~1~~ | ~~$1.05~~ | ~~Apr 17~~ | | **EXPIRING WORTHLESS** ($0 bid) |
| ~~OBDC~~ | ~~$10P~~ | ~~65~~ | ~~$0.15~~ | ~~Apr 17~~ | | **EXPIRING WORTHLESS** ($0 bid, $975 loss) |

**WS Apr expired premium lost: -$1,495** (ARCC $130 + JNK $285 + KRE $105 + OBDC $975)

### Moomoo (FUTUCA) — ~$17,684 USD

| Ticker | Type | Qty | Spot | MV | Notes |
|--------|------|-----|------|-----|-------|
| SQQQ | shares | 172 | $76.54 | $13,164 | Inverse QQQ — performing in VIX 24 |
| SPXS | shares | 106 | $39.19 | $4,155 | Inverse SPX — performing |
| GLD | $298C | 1 | $428.66 | ? | No entry data |
| SLV | $33.5C | 7 | $65.79 | ? | No entry data |
| Cash | | | | $365 | |

### Other Venues

| Venue | Value | Notes |
|-------|-------|-------|
| Polymarket | $535.73 USDC | ACTIVE — 3 strategies, active_scanner.py |
| NDAX | LIQUIDATED | $4,492 CAD withdrawn |

### Grand Total: ~$32,538 USD | VIX: 24.05 | CAD/USD: 0.7189

**TOTAL EXPIRED PREMIUM LOST: -$1,645** (all Apr puts)

## Critical Dates

| Date | Event | Action |
|------|-------|--------|
| ~~Apr 4~~ | ~~KRE/IWM expiry~~ | EXPIRED |
| **Apr 10** | **XLF 21-DTE roll trigger** | Evaluate XLF $46P May 1. If bid > $0.10, roll to Jun. If $0 → dead-put gate. |
| **Apr 17** | **IBKR + WS Apr OPEX** | All Apr puts expire worthless. No action needed — $0 bid confirmed. |
| **Apr 24** | **LQD/EMB 21-DTE roll trigger** | Roll decision for May 15 puts per ROLL_DISCIPLINE. |
| May 1 | XLF $46P expiry | If not rolled Apr 10, expires. |
| May 15 | LQD/EMB expiry | If not rolled Apr 24, expires. |
| **May 28** | **BKLN/HYG/OWL 21-DTE roll trigger** | Jun 18 puts roll decision. First real test of new discipline. |
| Jun 18 | BKLN x3, HYG, OWL, XLE Jun call, SLV Jun call | Jun OPEX — multiple positions. |

## Active Workstreams

| Workstream | Status | Notes |
|---|---|---|
| Monitoring Overhaul | **DONE** | Parallel collectors, degradation panel, ASCII banner, 5s timeouts. Live-tested 2026-04-02. |
| 3,297 Quality Fixes | **DONE** | ruff clean, lint zero, F821/F402/F601/E741/E701 all fixed. |
| Context Guardrails | DONE | copilot-instructions.md, AGENTS.md, STATUS.md, 3 path-specific |
| Root Cleanup + Archives | DONE | 86→100+ files archived, root 170+ → 30 |
| ~~Apr 10 Roll Execution~~ | **CANCELLED** | Apr puts all $0 bid — no credit to recover. Roll plan dead. See post-mortem in war_room_engine.ROLL_DISCIPLINE. |
| **Polymarket Division Activation** | **DONE** | active_scanner.py (450+ lines), 3 strategies unified, launch.py mode, DRY_RUN=true default |
| **Codebase Consolidation** | **DONE** | 100+ files archived/deleted. SharedInfrastructure removed. _scratch 87→29. strategies 99→86. Empty stub dirs purged (api-gateway, market-data). Balance scanner relocated to scripts/. |
| Architecture Rework v3.3 | **ALL PHASES COMPLETE (1-7)** | Strategy Advisor loop, NCL Relay heartbeat, Doctrine Terrain routing, Monitor panels — all wired into orchestrator |
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
