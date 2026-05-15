# AAC Living Status Dashboard

> **Last updated:** 2026-05-15 (PM)
> **Updated by:** 250-gap audit + first remediation pass. Audit doc: `.context/07_gaps/gap-audit-2026-05-15-250.md`. Fixed: 39 root-clutter logs deleted, 36 duplicate modules collapsed to shims, 43 silent excepts rewritten with debug-logging, deterministic 1.5 bps slippage in `TradingExecution/execution_engine.py`, pre-commit hook + `tools/check_forbidden_patterns.py` now block new silent-excepts / sys.path.insert / root pytest logs. New tests: `tests/test_portfolio_manager_agent.py` (13 cases). Pytest **4981 passed, 25 skipped, 1 xfailed, 2 pre-existing MagicMock failures** in `test_breadth_client.py` (unrelated).
> **Update this file** after every significant change. This is the single source of truth for what works.

---

## System Health — What Works

| Component | Status | Notes |
|---|---|---|
| **IBKR Connector** | LIVE | Port 7497, account U24346218. 15 active positions (5 calls + 10 puts). Net liq CAD $20,079.57. |
| **Moomoo Connector** | DEGRADED | OpenD relaunched 2026-05-14 (PID 19612) but API port 11111 still requires real login in OpenD.xml. Dummy credentials persist. |
| **yfinance** | WORKING | Free, primary options chain source |
| **CoinGecko** | DEGRADED | Pro key expired → free tier (10 req/min). Prices work. |
| **Unusual Whales** | FIXED (parsing) | Field parsing repaired 2026-05-15: `get_flow()` now decodes OCC `option_chain` (e.g. `SPY260418C00530000`) when top-level strike/type/expiry missing — no more $0 strikes / "unknown" types. `_top_gex_walls()` reads correct UW fields (`call_gamma_oi`/`put_gamma_oi`/`total_gamma_per_one_pct_move_oi`) — GEX no longer silently zero. Tests: `tests/test_unusual_whales_occ_parsing.py` (5 cases). Token still needs valid replacement. |
| **NYSE Breadth Client** | NEW | `integrations/breadth_client.py` — yfinance `^TRIN`/`^TICK`/`^ADD`/`^DECL`/`^ADV`, McClellan Oscillator (EMA19−EMA39), votes-based regime classifier. No API key. |
| **CFTC COT Client** | NEW | `integrations/cftc_cot_client.py` — Traders in Financial Futures yearly zips (ES/NQ/RTY/YM/VX), positioning + 52-week z-score extreme signals. Custom UA required (cftc.gov 403s default urllib). |
| **ETF Flow Client** | NEW | `integrations/etf_flow_client.py` — yfinance `fast_info`/`info`, daily flow estimated as Δshares × NAV, persists snapshots to `data/etf_flow_history.json` (400/symbol cap). 27-ticker default universe (SPY/QQQ/sectors/credit/commodities). |
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
| **Pytest Suite** | WORKING | **4908 passed**, 25 skipped, 1 xfailed, **3 failed** (2026-05-14). Failures: `tests/test_account_equity_feed.py::TestPnlSnapshotWiring` (3 cases) — `_account_value_feed.get_source()` returns `"default"` when TWS offline, triggering drawdown skip. Mock fix needed. |
| **WSB Sentiment Feed** | WORKING | TradeStie API wired into `war_room_live_feeds.fetch_all_live_data()` — free, no auth, top-50 WSB stocks, 15-min updates |
| **Circuit Breaker** | NEW | `shared/circuit_breaker.py` — clean CLOSED/OPEN/HALF_OPEN state machine extracted from dead quantum wrapper |
| **Codebase Audit** | COMPLETE | 398 archive/scratch files categorised (2026-04-20). 4 critical missing imports restored. 10 sci-fi dead files identified. |
| **Vol Premium Strategy** | NEW | `strategies/vol_premium_signals.py` — VRP thesis, LONG_PUT when IV/HV > 1.20, confidence 0.50–0.90, size 3–8% |
| **Signal Aggregator** | NEW | `strategies/signal_aggregator.py` — weighted merge (0.60/0.40), agreement boost +0.05, max confidence 0.95 |
| **Simple Backtest** | NEW | `strategies/simple_backtest.py` — 90-day win-rate comparison, 3 strategy proxies, fully offline |
| **Dual-Strategy Scan** | NEW | `core/orchestrator.py` `war_room_scan()` runs both strategies concurrently, aggregates signals |
| **Market Scheduler** | NEW (wired) | `core/market_scheduler.py` orchestrates auto_trader, eod_reporter, execution_throttle, order_monitor, position_reconciler. Imported by `launch.py`. Confirmed 2026-05-14. |
| **AAC RAG** | NEW | `shared/aac_rag/` (LanceDB + Ollama qwen2.5-coder:7b). `python -m shared.aac_rag {reindex|query|ask}`. Smoke test: `tests/test_aac_rag_smoke.py` (3/3 pass). |
| **AAC Calendar** | NEW | `shared/aac_calendar/` aggregator package |
| **AAC Agents** | NEW | `shared/aac_agents/` runtime + tools + history |

## What's Broken

| Component | Problem | Priority | Notes |
|---|---|---|---|
| **CoinGecko Pro** | Key expired, returns 403 | LOW | Auto-downgrades to free tier. Works fine for now. |
| **Polygon Options** | Free tier: 403 on snapshots | LOW | Needs $79/mo upgrade. Not blocking. |
| **X/Twitter API** | HTTP 402 | LOW | Needs paid tier. Graceful fallback to 0.5. |
| **NDAX** | ACTIVE (zero positions, account open) | LOW | XRP+ETH sold Mar 2026 ($4,492 CAD). Account open, NDAX_API_KEY/SECRET/USER_ID in .env, TESTNET=false. Connector ready. Wired back into docs 2026-04-20. Re-enter BTC/CAD on next doctrine signal. |

## Active Positions (Real Money) — Updated 2026-05-15 (post Apr-17 / May-1 expiries)

### IBKR (Account U24346218) — Net Liq CAD $20,079.57 (~USD $14,520) | Cash CAD $2,700.83

**Calls (5 positions — $12,143 MV)**

| Ticker | Strike | Qty | Expiry | MV | PnL | Status |
|--------|--------|-----|--------|------|------|--------|
| SLV | $66C | 8 | Jun 18 | $6,560 | +$282 | ITM, hold |
| SLV | $75C | 2 | Jan 2027 | $2,201 | +$139 | LEAPS, OTM |
| TSLA | $500C | 1 | Jan 2027 | $1,887 | -$154 | LEAPS, deep OTM |
| SLV | $70C | 2 | Jun 18 | $1,270 | +$128 | Near ATM, hold |
| XLE | $65C | 3 | Jun 18 | $225 | -$581 | OTM, underwater |

**Puts (5 live positions — Apr 17 / May 1 expired worthless)**

| Ticker | Strike | Qty | Expiry | Status | DFV thesis |
|--------|--------|-----|--------|--------|------------|
| OBDC | $7.5P | 11 | Jul 17 | OTM, hold | conv 2 — BDC credit-stress lottery |
| BKLN | $20P | 3 | Jun 18 | Near ATM, hold | conv 2 — leveraged-loan dislocation |
| HYG | $77P | 1 | Jun 18 | OTM, hold | conv 2 — HY spread widening |
| LQD | $106P | 1 | **May 15 — TODAY** | 0-DTE, let die | conv 1 — IG credit hedge |
| EMB | $90P | 1 | **May 15 — TODAY** | 0-DTE, let die | conv 1 — EM USD-debt hedge |

**Expired worthless (closed)**

| Ticker | Strike | Expiry | Realized PnL |
|--------|--------|--------|--------------|
| MAIN | $49.7P | 2026-04-17 | -$53 |
| ARCC | $17P | 2026-04-17 | -$15 |
| JNK | $92P | 2026-04-17 | -$33 |
| PFF | $29P | 2026-04-17 | -$17 |
| XLF | $46P | 2026-05-01 | -$62 |
| KRE | $58P | 2026-04-04 | -$271 |
| IWM | $230P | 2026-04-04 | -$270 |

**Realized put losses since Apr 4: -$721.** Theses (incl. post-mortems) in `agents/dfv/memory/thesis_log.json`.

### WealthSimple TFSA (~$18,638 CAD / ~$13,398 USD)

| Ticker | Strike | Qty | Entry | Expiry | Spot | Status |
|--------|--------|-----|-------|--------|------|--------|
| GLD | $515C | 1 | $19.40 | Mar 2027 | $428.66 | LEAPS — OTM, hold (time value) |
| XLE | $85C | 26 | $0.37 | Jan 2027 | $59.37 | LEAPS — deep OTM, hold |
| OWL | $8P | 5 | $0.75 | Jun 18 | $8.55 | Near ATM, hold |

**WS Apr 17 expired worthless (closed)**: ARCC $16P (-$130), JNK $94P (-$285), KRE $60P (-$105), OBDC $10P (-$975). Total realized: **-$1,495**.

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
| **Moomoo OpenD Auth Fix** | BLOCKED | OpenD relaunched 2026-05-14 (PID 19612). Port 11111 still down — dummy creds in OpenD.xml. User must update + restart. |
| **GitHub Cleanup (May 14)** | DONE | 488 stale changes → 9 thematic commits pushed: gitignore noise, `__future__` sweep (239 files), substantive rollup (91 files), repo hygiene, configs, core/trading/shared infra, strategies, planktonxd, monitoring+tests. |
| **`test_account_equity_feed` Fix** | TODO | 3 failing tests need `_account_value_feed.get_source()` mocked to `"env"` or `"ibkr"` so drawdown update isn't skipped when TWS offline. |

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
