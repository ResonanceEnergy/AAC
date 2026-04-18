# AAC Goal-Mandate Roadmap

> **Created:** 2026-04-08
> **Purpose:** Replace aspirational STATUS claims with a brutally honest assessment and a concrete, sequenced task list.
> **Rule:** Nothing is "DONE" until it runs end-to-end with a passing test. No more aspirational checkmarks.

---

## Part 1 — Honest State of the Codebase

### What ACTUALLY Works (Verified)

| Component | Evidence | Confidence |
|---|---|---|
| IBKR connector — order execution | 8 real trades executed, account U24346218 | HIGH |
| Moomoo connector — account queries | Real mode, FUTUCA, $365 USD | HIGH |
| yfinance options chains | Free, no key needed, used in War Room | HIGH |
| launch.py mode dispatch | Starts processes, routes to correct modules | HIGH |
| Pytest suite — 1715 passing | CI green, verified | HIGH |
| CoinGecko free tier prices | 10 req/min, auto-downgrade works | MEDIUM |
| Unusual Whales field parsing | Fixed 2026-03-31, key valid | MEDIUM |
| FRED / Finnhub / NewsAPI | Keys in .env, basic queries work | MEDIUM |

### What Exists But Doesn't Actually Work End-to-End

| Component | Problem | Reality |
|---|---|---|
| **Orchestrator run loop** | Starts 10+ background tasks, most are stubs or log-and-return | ~30% real logic |
| **Unified Integrator** | Own docstring admits "50 strategies loaded but NEVER wired" | Facade |
| **Strategy Advisor Engine** | Class exists, orchestrator never instantiates it | Dead code |
| **NCL Relay** | Sends heartbeat, never receives directives. Fallback to defaults. | One-way stub |
| **Doctrine Terrain Routing** | Name in comments only. No real routing logic. | Not implemented |
| **Matrix Monitor collectors** | 6-7 real, rest hardcoded/stubbed. Display handlers are `pass` | ~25% real |
| **Doctrine Engine state actions** | 4 states defined, transition handlers are `pass` | States work, actions don't |
| **War Room regime detection** | Drifts hardcoded (oil +65%, spy -25%). Not reading live data. | Simulation only |
| **ROLL_DISCIPLINE** | Defined in war room, never enforced anywhere | Unenforced |
| **Paper trading validation** | References missing `PaperTradingDivision.order_simulator` | Will crash |
| **44 strategy files** | 5-7 wired, ~37 orphaned scaffolds | 85% dead code |
| **~200+ Python files** | Orphaned, never imported by running system | Dead weight |

### What's Broken / Non-Functional

| Component | Status |
|---|---|
| CoinGecko Pro | Key expired, 403 |
| Polygon options snapshots | Free tier, 403 |
| X/Twitter API | HTTP 402 |
| NDAX | Liquidated, connector unused |
| Kraken / Coinbase / SnapTrade / MetalX / NoXi Rise connectors | Stubs or partial |
| Cross-pillar governance (NCC/NCL) | Those systems don't exist as running services |

---

## Part 2 — Goal Mandate

### MISSION
AAC exists to **make money through algorithmic trading**. Every line of code must serve one of these goals:

1. **Find trades** — Scan markets, detect edges, generate signals
2. **Execute trades** — Place orders on real exchanges with real money
3. **Manage risk** — Position sizing, stop losses, roll discipline, exposure limits
4. **Track P&L** — Know what we own, what it's worth, what we've made/lost
5. **Monitor health** — Know when systems are up/down, APIs working/broken

Everything else is overhead. If it doesn't serve one of these five, it's drift.

### PRINCIPLES
- **No aspirational code.** If it doesn't run, delete it or finish it.
- **No `pass` in production paths.** Either implement or raise `NotImplementedError`.
- **Test it or it doesn't exist.** Every feature needs a test that proves it works.
- **One thing at a time.** Finish Sprint N before starting Sprint N+1.
- **Honest status tracking.** STATUS.md reflects reality, not ambition.

---

## Part 3 — Sprint Roadmap

### Sprint 0: Cleanup & Foundation (PREREQUISITE)
**Goal:** Remove dead code, fix crashes, establish honest baseline.

- [ ] **0.1** Archive 37+ orphaned strategy files to `_archive/strategies/`
- [ ] **0.2** Replace all `pass` statements in core paths (orchestrator, doctrine, monitor) with `raise NotImplementedError("TODO")` — make failures visible
- [ ] **0.3** Fix paper trading ImportError (`PaperTradingDivision.order_simulator`)
- [ ] **0.4** Delete dead connectors (MetalX, NoXi Rise) — move stubs to `_archive/`
- [ ] **0.5** Remove "quantum", "cross-temporal", "AI autonomy" feature flags from orchestrator — they're science fiction
- [ ] **0.6** Update STATUS.md to reflect REAL state (this roadmap)
- [ ] **0.7** Run full test suite — must stay at 1715+ passing
- **Exit criteria:** `ruff check .` clean, `pytest` green, no silent `pass` in core paths

### Sprint 1: Signal Pipeline (FIND TRADES)
**Goal:** One reliable path from market data → actionable signal.

- [ ] **1.1** Pick THE top strategy (War Room or Storm Lifeboat) as primary signal source
- [ ] **1.2** Wire it to live market data (yfinance + Finnhub) — no hardcoded drifts
- [ ] **1.3** War Room regime detection reads VIX, SPY, oil prices from APIs (not constants)
- [ ] **1.4** Signal output format: `{ticker, direction, confidence, entry, stop, target, size}`
- [ ] **1.5** End-to-end test: market data in → signal out (mocked data, deterministic)
- **Exit criteria:** `python launch.py core` generates real signals from live data

### Sprint 2: Execution Path (EXECUTE TRADES)
**Goal:** Signal → order on IBKR, with confirmation.

- [ ] **2.1** Clean execution path: signal → risk check → order → confirmation
- [ ] **2.2** IBKR connector: `place_order()` → `get_order_status()` → `confirm_fill()`
- [ ] **2.3** Paper mode that mirrors real execution but doesn't send to exchange
- [ ] **2.4** Position tracker: what do we own right now? (read from IBKR, not in-memory)
- [ ] **2.5** End-to-end test: signal → mock IBKR → fill confirmation
- **Exit criteria:** Can place a real trade from a system-generated signal

### Sprint 3: Risk & Roll Discipline (MANAGE RISK)
**Goal:** Never hold a position without knowing our exposure and exit plan.

- [ ] **3.1** Position exposure calculator: total $ at risk, per-position risk
- [ ] **3.2** ROLL_DISCIPLINE enforced: 21-DTE trigger → evaluate → roll or close
- [ ] **3.3** Max position size rules (% of account, absolute $)
- [ ] **3.4** Stop loss / take profit automation or alerts
- [ ] **3.5** End-to-end test: position hits 21-DTE → roll decision fires
- **Exit criteria:** No position can be opened without risk check passing

### Sprint 4: P&L Tracking (TRACK P&L)
**Goal:** Know what we've made and lost, live and historically.

- [ ] **4.1** Reconcile positions across IBKR + Moomoo + WealthSimple
- [ ] **4.2** Daily P&L snapshot to SQLite (what we own, mark-to-market)
- [ ] **4.3** Historical trade log: entry, exit, P&L per trade
- [ ] **4.4** Simple CLI report: `python launch.py pnl` → current positions + today's P&L
- [ ] **4.5** Test: snapshot → next day snapshot → P&L delta calculated correctly
- **Exit criteria:** Can answer "how much did we make/lose today?" with one command

### Sprint 5: Monitoring (MONITOR HEALTH)
**Goal:** Single dashboard that shows real system state, not aspirational panels.

- [ ] **5.1** Strip monitor down to working collectors only (6-7 real ones)
- [ ] **5.2** Dashboard shows: API health, open positions, P&L, last signal, last trade
- [ ] **5.3** Alerting: notify when API goes down, position hits stop, system error
- [ ] **5.4** Remove fake panels and `pass`-statement display handlers
- [ ] **5.5** Test: dashboard loads, all panels render with real or mocked data
- **Exit criteria:** `python launch.py dashboard` shows useful, accurate information

### Sprint 6: Second Strategy & Diversification
**Goal:** Add a second independent signal source.

- [ ] **6.1** Activate second strategy (e.g., Storm Lifeboat or Polymarket edge)
- [ ] **6.2** Signal aggregator: combine two signals with weights
- [ ] **6.3** Backtest both strategies against last 90 days
- [ ] **6.4** Compare: which strategy would have made/lost money?
- [ ] **6.5** End-to-end: combined signal → execution → P&L tracking
- **Exit criteria:** Two strategies producing independent signals that get aggregated

### Sprint 7: Automation & Scheduling
**Goal:** System runs unattended during market hours.

- [ ] **7.1** Scheduled scan: every 15 min during market hours
- [ ] **7.2** Scheduled roll check: daily at market open
- [ ] **7.3** Daily P&L snapshot at market close
- [ ] **7.4** Health check: every 5 min, alert on failure
- [ ] **7.5** Graceful restart on crash
- **Exit criteria:** Leave it running for a full trading day, comes back healthy

---

## Part 4 — What Gets Deleted / Archived

These are **not coming back** unless explicitly revived in a future sprint:

| Component | Action | Reason |
|---|---|---|
| Quantum Arbitrage Engine | DELETE | Science fiction |
| Cross-Temporal Processor | DELETE | Science fiction |
| AI Incident Predictor | DELETE | No model, no data |
| Predictive Maintenance | DELETE | Over-engineering |
| 37+ orphaned strategy files | ARCHIVE | Never wired |
| MetalX / NoXi Rise connectors | ARCHIVE | No exchange access |
| Kraken / Coinbase stubs | ARCHIVE | No accounts, no keys |
| SnapTrade stub | ARCHIVE | Not configured |
| Pillar federation readers | DELETE | NCC/NCL don't exist as services |
| 13-Moon Doctrine dashboard | ARCHIVE | Derivative, not core |
| Mission Control dashboard | ARCHIVE | Duplicate of monitor |

---

## Part 5 — Status Tracking Rules

1. **STATUS.md** gets updated at the END of each sprint, not during
2. Only three status values: `NOT STARTED`, `IN PROGRESS`, `DONE`
3. `DONE` means: code works, test passes, user has verified
4. No more "Phase 1-7 COMPLETE" unless each phase has a passing test
5. This roadmap is the source of truth — STATUS.md mirrors it

---

## Current Sprint: **0 — Cleanup & Foundation**
## Next Sprint: **1 — Signal Pipeline**
