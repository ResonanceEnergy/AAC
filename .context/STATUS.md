# AAC Living Status Dashboard

> **Last updated:** 2026-04-09 (Wednesday)
> **Updated by:** Live IBKR data pull — 15 positions confirmed (10 new: SLV/TSLA/XLE calls, OBDC puts). Net liq CAD $20,079.57. VIX 20.88. Port confirmed 7497.
> **Update this file** after every significant change. This is the single source of truth for what works.

---

## System Health — What Works

| Component | Status | Notes |
|---|---|---|
| **IBKR Connector** | LIVE | Port 7497, account U24346218. 15 active positions (5 calls + 10 puts). Net liq CAD $20,079.57. |
| **Moomoo Connector** | DEGRADED | OpenD running but API port 11111 NOT listening. Dummy credentials in OpenD.xml. Needs auth fix + restart. |
| **yfinance** | WORKING | Free, primary options chain source |
| **CoinGecko** | DEGRADED | Pro key expired → free tier (10 req/min). Prices work. |
| **Unusual Whales** | WORKING | Key valid, connection works. Field parsing FIXED (2026-03-31). |
| **FRED** | WORKING | VIX fallback, macro data |
| **Finnhub** | WORKING | Quotes, news |
| **NewsAPI** | WORKING | Headlines |
| **Doctrine Engine** | WORKING | 11 packs, 4-state machine |
| **Matrix Monitor** | WORKING | Parallel collection (5s timeout/collector), degradation panel, ASCII banner. Confirmed live 2026-04-02: 24/30 collectors OK, 6 timeout (expected — IBKR/NCC/NCL offline). |
| **War Room Engine** | WORKING | Wired to integrator. Regime: WATCH (39.6), 15-indicator live feeds. ROLL_DISCIPLINE rules added Apr 6. |
| **13-Moon Doctrine** | WORKING | Moon 1 (Pink Moon) — DEPLOY mandate. 184 events across 6 overlay layers. Lead-time alerts active. Storyboard HTML + JSON export. |
| **Polymarket Division** | ACTIVE | py-clob-client v0.34.6, wallet live ($535.73 USDC), active_scanner.py (450+ lines), `launch.py polymarket` |
| **Paper Trading** | WORKING | `launch.py paper`. Paper trading divisions: polymarket_paper (5 strategies), crypto_paper (4 strategies). Optimizer, bakeoff gate system, YAML configs all operational. |
| **Bakeoff Engine** | WORKING | Gate progression (SPEC→SCALE), composite scoring, YAML configs (metric_canon, policy, checklists) created. |
| **Web Dashboard** | WORKING | `launch.py dashboard` |
| **CI Pipeline** | WORKING | `.github/workflows/ci.yml` |
| **Pytest Suite** | WORKING | **1928+ passed**, 0 failed (2026-04-08 — live feeds + paper trading tests) |

## What's Broken

| Component | Problem | Priority | Notes |
|---|---|---|---|
| **CoinGecko Pro** | Key expired, returns 403 | LOW | Auto-downgrades to free tier. Works fine for now. |
| **Polygon Options** | Free tier: 403 on snapshots | LOW | Needs $79/mo upgrade. Not blocking. |
| **X/Twitter API** | HTTP 402 | LOW | Needs paid tier. Graceful fallback to 0.5. |
| **NDAX** | LIQUIDATED | NONE | All crypto sold. Connector exists but unused. |

## Active Positions (Real Money) — Updated Apr 9 (IBKR live-verified)

### IBKR (Account U24346218) — Net Liq CAD $20,079.57 (~USD $14,520) | Cash CAD $2,700.83

**Calls (5 positions — $12,143 MV)**

| Ticker | Strike | Qty | Expiry | MV | PnL | Status |
|--------|--------|-----|--------|------|------|--------|
| SLV | $66C | 8 | Jun 18 | $6,560 | +$282 | ITM, hold |
| SLV | $75C | 2 | Jan 2027 | $2,201 | +$139 | LEAPS, OTM |
| TSLA | $500C | 1 | Jan 2027 | $1,887 | -$154 | LEAPS, deep OTM |
| SLV | $70C | 2 | Jun 18 | $1,270 | +$128 | Near ATM, hold |
| XLE | $65C | 3 | Jun 18 | $225 | -$581 | OTM, underwater |

**Puts (10 positions — $431 MV)**

| Ticker | Strike | Qty | Expiry | MV | PnL | Status |
|--------|--------|-----|--------|------|------|--------|
| OBDC | $7.5P | 11 | Jul 17 | $178 | -$16 | OTM, hold |
| BKLN | $20P | 3 | Jun 18 | $118 | -$1 | Near ATM, hold |
| HYG | $77P | 1 | Jun 18 | $36 | -$44 | OTM, hold |
| LQD | $106P | 1 | May 15 | $28 | -$35 | OTM, monitor |
| EMB | $90P | 1 | May 15 | $26 | -$22 | OTM, monitor |
| MAIN | $49.7P | 1 | Apr 17 | $20 | -$53 | Near-worthless, expiring |
| XLF | $46P | 1 | May 1 | $13 | -$62 | OTM, 21-DTE roll window |
| ARCC | $17P | 1 | Apr 17 | $10 | -$15 | Near-worthless, expiring |
| JNK | $92P | 1 | Apr 17 | $2 | -$33 | Near-worthless, expiring |
| PFF | $29P | 1 | Apr 17 | $0 | -$17 | Worthless, expiring |

**IBKR Total PnL: -$484** (Calls -$186, Puts -$298). KRE/IWM expired Apr 4 (-$541).

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

### Grand Total: ~$46,220 USD | VIX: 20.88 | CAD/USD: 0.7231

**Note:** IBKR live-verified Apr 9. WS TFSA/Moomoo balances from last manual check — Moomoo OpenD disconnected.
**EXPIRED PREMIUM (KRE/IWM Apr 4): -$541.** Apr 17 puts (~$150 IBKR + ~$1,495 WS) still pending.

## Critical Dates

### 13-Moon Doctrine — Next 30 Days (Moon 1: Pink Moon)

| Date | Layer | Event | Action |
|------|-------|-------|--------|
| **Apr 8** | Financial | FOMC March Minutes Released | Pre-FOMC positioning |
| **Apr 10** | Financial+AAC | March CPI + Dispersion Strategy + 7-DTE puts | CPI war-month inflation; IBKR/WS final week |
| **Apr 11** | Financial | Q1 Bank Earnings Begin | IV rank assessment |
| **Apr 13** | World | IMF/World Bank Spring Meetings | Monitor communique leaks |
| **Apr 14** | AAC | War Day 45: Major Inflection | Scenario analysis |
| **Apr 15** | AAC | Seesaw: Inflation Rotation + US Debt P=20% + BTC $100K watch | Monitor triggers |
| **Apr 16** | AAC | IBKR/WS PUTS: EXPIRY EVE | Close all Apr 17 positions |
| **Apr 17** | Financial | Apr OPEX + ECB Rate Decision | All Apr puts expire worthless |
| **Apr 19** | Phi | phi^1 resonance (Apr 17-21) | Monitor |
| **Apr 20** | World+AAC | Iran Nuclear Talks + XLF May 1 Roll (25 DTE) | Roll decision |
| **Apr 22-30** | Financial | FAANG Earnings (TSLA/AMZN/MSFT/META/AAPL/GOOG) | Earnings IV plays |
| **Apr 25** | Astrology | Uranus enters Gemini | Paradigm shift marker |
| **Apr 30** | Financial+AAC | March PCE + Q1 GDP Advance | Recession signal watch |
| **May 1** | AAC+Phi | VIX 25 regime shift + Composite 70 trigger + Flower Moon Scorpio | Conviction trigger |
| **May 6** | Financial+AAC | FOMC May Meeting + DTE 45 Jun puts | Emergency cut watch + theta acceleration |

### Position Calendar

| Date | Event | Action |
|------|-------|--------|
| ~~Apr 4~~ | ~~KRE/IWM expiry~~ | EXPIRED |
| **Apr 10** | **XLF 21-DTE roll trigger** | Evaluate XLF $46P May 1. If bid > $0.10, roll to Jun. If $0 → dead-put gate. |
| **Apr 17** | **IBKR + WS Apr OPEX** | IBKR: ARCC/PFF/MAIN/JNK puts expire. WS: ARCC/JNK/KRE/OBDC puts expire. |
| **Apr 24** | **LQD/EMB 21-DTE roll trigger** | Roll decision for May 15 puts per ROLL_DISCIPLINE. |
| May 1 | XLF $46P expiry | If not rolled Apr 10, expires. |
| May 15 | LQD/EMB expiry | If not rolled Apr 24, expires. |
| **May 28** | **Jun 18 positions: 21-DTE trigger** | IBKR: SLV $66C x8, SLV $70C x2, XLE $65C x3, BKLN x3, HYG. WS: OWL $8P x5. |
| Jun 18 | Jun OPEX (8 IBKR + 1 WS) | IBKR: SLV $66C x8, SLV $70C x2, XLE $65C x3, BKLN $20P x3, HYG $77P. WS: OWL $8P x5. |
| **Jun 26** | **OBDC 21-DTE roll trigger** | OBDC $7.5P x11 Jul 17 roll decision. |
| Jul 17 | OBDC $7.5P x11 expiry | Jul OPEX. |
| Jan 2027 | SLV $75C x2, TSLA $500C x1 LEAPS | Long-dated — monitor quarterly. |

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
| **Moomoo OpenD Auth Fix** | BLOCKED | OpenD running but API port 11111 not listening. Dummy credentials in OpenD.xml. User must update + restart. |

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
| 13-Moon timeline | `python -m strategies.thirteen_moon_doctrine --upcoming 30` |
| 13-Moon storyboard | `data/storyboard/thirteen_moon_storyboard.html` |
| Launch the system | `python launch.py <mode>` |
