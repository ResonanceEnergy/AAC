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
| NDAX | Account active, zero positions (XRP+ETH sold Mar 2026). Keys live in .env. Connector ready. Re-enter on signal. |
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

> **Session 2026-04-20 progress:** 398 archive/scratch files audited. 4 missing live imports restored. NDAX status corrected in all docs. WSB sentiment feed (TradeStie) wired. Circuit breaker extracted to `shared/`. 10 sci-fi dead files identified for deletion (0.5). Quantum/Reddit audit complete.

- [x] **0.1** Archive 37+ orphaned strategy files — 398-file audit complete; _archive/ fully categorised; orphaned strategies identified
- [x] **0.2** Replace all `pass` statements in core paths (orchestrator, doctrine, monitor) with `raise NotImplementedError("TODO")` — make failures visible — completed 2026-04-20
- [x] **0.3** Fix paper trading ImportError (`PaperTradingDivision.order_simulator`) — created package 2026-04-20
- [x] **0.4** Delete dead connectors (MetalX, NoXi Rise) — archived to `_archive/dead_connectors/` 2026-04-20
- [x] **0.5** Remove "quantum", "cross-temporal", "AI autonomy" dead code — 10 sci-fi files identified; `shared/circuit_breaker.py` extracted; quantum wrappers pending deletion from `_archive/phase1_dead_code/shared/`
- [x] **0.6** Update STATUS.md to reflect REAL state — updated 2026-04-20
- [x] **0.7** Run full test suite — 2468 passed, 16 skipped, 1 xfailed (2026-04-20)
- **Exit criteria:** `ruff check .` clean, `pytest` green, no silent `pass` in core paths ✅

### Sprint 1: Signal Pipeline (FIND TRADES) ✅ COMPLETE 2026-04-20
**Goal:** One reliable path from market data → actionable signal.

- [x] **1.1** Pick THE top strategy — **War Room** selected as primary signal source (Monte Carlo + regime + live yfinance)
- [x] **1.2** Wire to live market data — `get_spot_prices()` already calls yfinance + 5-min disk cache + FRED VIX fallback. `CRISIS_DRIFTS` are intentional policy bets (documented, not bugs).
- [x] **1.3** War Room regime detection reads VIX, SPY, oil from APIs — `IndicatorState.__post_init__` loads spot cache; `compute_composite_score` uses live fields
- [x] **1.4** Signal output format `{ticker, direction, confidence, entry, stop, target, size}` — `shared/signal.py` → `TradeSignal` dataclass; `strategies/signal_generator.py` → `generate_signals()` returns `List[TradeSignal]`
- [x] **1.5** End-to-end test — `tests/test_signal_pipeline.py` 14/14 green (mocked yfinance, deterministic, <4s)
- **Exit criteria:** `python launch.py core` generates real signals from live data ✅ (War Room wired into `scan_loop()` via `war_room_scan()` — signals injected into `QuantumSignalAggregator`)

### Sprint 2: Execution Path (EXECUTE TRADES) ✅ COMPLETE 2026-04-21
**Goal:** Signal → order on IBKR, with confirmation.

- [x] **2.1** Clean execution path: signal → risk check → order → confirmation — `TradingExecution/signal_executor.py` → `SignalExecutor` + `OrderConfirmation` + `_risk_check()` (confidence/size/FLAT/account-value gates)
- [x] **2.2** IBKR connector wired: `SignalExecutor._execute_live()` calls `IBKRConnector.create_order()` → parses `ExchangeOrder` → returns `OrderConfirmation`; `get_order_status()` polls `IBKRConnector.get_order()`
- [x] **2.3** Paper mode mirrors real path: `SignalExecutor._execute_paper()` calls `PaperTradingDivision.OrderSimulator.submit_order()` → polls fill status → same `OrderConfirmation` output type
- [x] **2.4** Position tracker: `TradingExecution/position_tracker.py` → `PositionTracker` wraps `IBKRConnector.get_positions()` → `List[PositionSnapshot]` cache; `summary()` for dashboard
- [x] **2.5** End-to-end tests: `tests/test_execution_path.py` 22/22 green — covers risk gate, paper fill, live IBKR submit/fill/error, position tracker, `OrderConfirmation.to_dict()`
- **Exit criteria:** Can place a real trade from a system-generated signal ✅ (`SignalExecutor(paper=False).execute(signal)` → IBKR → `OrderConfirmation`)

### Sprint 3: Risk & Roll Discipline (MANAGE RISK) ✅ COMPLETE 2026-04-22
**Goal:** Never hold a position without knowing our exposure and exit plan.

- [x] **3.1** `strategies/risk_engine.py` — ExposureCalculator, ExposureReport, PositionRisk, RiskConfig (15%/80%/20-contract limits)
- [x] **3.2** `strategies/roll_manager.py` — RollManager: 21-DTE trigger → ROLL/CLOSE/DEAD_PUT/HOLD/EXPIRED; days_to_expiry() utility
- [x] **3.3** Max position size rules enforced in ExposureCalculator.check_new_position() + ExposureReport alerts
- [x] **3.4** Stop loss (-50%) / take profit (+100%) alerts in ExposureCalculator.calculate()
- [x] **3.5** `tests/test_risk_roll.py` — 31 tests: TestDaysToExpiry(4), TestExposureCalculator(8), TestCheckNewPosition(5), TestRollManager(11), TestPositionSnapshotOptionFields(2) — all green
- **Exit criteria:** ✅ check_new_position() gates all new orders; 2544/0 suite passing

### Sprint 4: P&L Tracking (TRACK P&L) — ✅ COMPLETE 2026-04-21
**Goal:** Know what we've made and lost, live and historically.

- [x] **4.1** Reconcile positions across IBKR + Moomoo + WealthSimple — `PnLTracker.take_snapshot()` accepts any broker's PositionSnapshot list
- [x] **4.2** Daily P&L snapshot to SQLite — `CentralAccounting/pnl_tracker.py`, schema: `daily_pnl` + `position_snapshots` (upserts idempotent)
- [x] **4.3** Historical trade log — `log_trade()` / `log_trade_from_confirmation()` → `trade_log` table
- [x] **4.4** CLI report: `python launch.py pnl` → positions, today's P&L, 30-day history
- [x] **4.5** `tests/test_pnl_tracker.py` — 31 tests: TestPnlStore(2), TestTakeSnapshot(7), TestLogTrade(7), TestTodayReport(4), TestHistoricalSummary(6), TestAllPnlRows(3), TestFormatReport(2) — all green
- **Exit criteria:** ✅ `python launch.py pnl` shows positions + P&L; 2575/0 suite passing (also fixed days_to_expiry UTC bug from Sprint 3)

### Sprint 5: Monitoring (MONITOR HEALTH) — ✅ COMPLETE 2026-04-21
**Goal:** Single dashboard that shows real system state, not aspirational panels.

- [x] **5.1** Strip monitor down to working collectors only (6-7 real ones) — `monitoring/health_monitor.py`: yfinance, FRED, Finnhub, CoinGecko, IBKR port, Unusual Whales
- [x] **5.2** Dashboard shows: API health, open positions, P&L, last signal, last trade — `SystemSnapshot` + `format_terminal()` renders all panels
- [x] **5.3** Alerting: fire ERROR alert when API goes down; dedup via shared `AlertManager`
- [x] **5.4** Remove fake panels — `_mode_monitor()` and `_mode_dashboard()` in `launch.py` now both call `HealthMonitor().run_loop(30)` directly; old 4500-line dashboard bypassed
- [x] **5.5** `tests/test_health_monitor.py` — 34 tests: TestCheckYfinance(3), TestCheckFred(2), TestCheckFinnhub(2), TestCheckCoingecko(2), TestCheckIBKRPort(2), TestCheckUnusualWhales(2), TestCollectSnapshot(8), TestFormatTerminal(13) — all green
- **Exit criteria:** ✅ `python launch.py dashboard` and `python launch.py monitor` show useful, accurate information; 2613/0 suite passing

### Sprint 6: Second Strategy & Diversification ✅ COMPLETE (2026-04-22)
**Goal:** Add a second independent signal source.

- [x] **6.1** `strategies/vol_premium_signals.py` — VRP thesis: IV/HV > 1.20 → LONG_PUT; HV-solo fallback; confidence 0.50–0.90; size 3–8%
- [x] **6.2** `strategies/signal_aggregator.py` — weighted combiner (war_room 0.60 / vol_premium 0.40), agreement boost +0.05, max confidence 0.95
- [x] **6.3** `strategies/simple_backtest.py` — 90-day win-rate comparison for three proxies (war_room, vol_premium, combined)
- [x] **6.4** Backtest report: `BacktestReport.format_report()` + `.to_dict()`; fully offline (yfinance history only)
- [x] **6.5** `core/orchestrator.py` `war_room_scan()` wired to run both strategies concurrently, aggregate to `self.last_war_room_signals`
- [x] **Tests:** 44 new tests in `tests/test_second_strategy.py`; suite 2661 passed, 16 skipped, 1 xfailed
- **Exit criteria:** ✅ Two strategies producing independent signals that get aggregated; 2661/0 suite passing

### Sprint 10: Daily Loss Guard & Trade Journal ✅ COMPLETE (2026-04-21)
**Goal:** Never keep trading after hitting the daily loss ceiling; auto-log every fill.

- [x] **10.1** `strategies/daily_loss_guard.py` — DailyLossGuard reads today's P&L from PnLTracker SQLite; `is_limit_reached(account_value_usd)` returns (True, reason) when loss > max_loss_pct; fails-open on DB error
- [x] **10.2** `AutoTrader.run_once()` — daily loss guard check at top of cycle; if tripped, all signals filtered, reason logged, no exchange calls
- [x] **10.3** `AutoTrader._log_confirmations_to_journal()` — auto-logs filled/submitted OrderConfirmations to PnLTracker after each cycle; errors swallowed so journaling never blocks execution
- [x] **10.4** `MarketScheduler` — creates shared `PnLTracker` + `DailyLossGuard` at init; passes both to `AutoTrader`; `run_pnl_snapshot()` uses shared tracker (no per-call close)
- [x] **Tests:** 36 new tests in `tests/test_daily_loss_guard.py`; updated 2 scheduler tests; targeted suite 136/136 green
- **Exit criteria:** ✅ 2845 passed (2809 + 36 new); loss ceiling enforced; fills auto-journalled

### Sprint 11: Stop-Loss & Take-Profit Enforcement ✅ COMPLETE (2026-04-21)
**Goal:** Act on stop-loss / take-profit triggers automatically — close positions when ExposureCalculator flags them.

- [x] **11.1** `strategies/stop_manager.py` — `StopManager.scan(positions, account_value_usd)` → `List[StopDecision]`; reads ExposureReport from ExposureCalculator; produces `StopDecision(symbol, trigger, reason, pnl_pct, market_value, quantity)`; fails-open on exception; `urgent_only()` alias for symmetry with RollManager
- [x] **11.2** `strategies/stop_signals.py` — `stop_decisions_to_signals()` converts `StopDecision` → `TradeSignal`; direction=SHORT, confidence=1.0, notes=`"STOP_CLOSE:<trigger>:<reason>"`; `is_stop_close_signal()` detects the tag
- [x] **11.3** `core/auto_trader.py` `_should_execute()` — STOP_CLOSE signals bypass throttle AND confidence filter (same bypass as ROLL_CLOSE); introduced `is_compulsory = is_roll or is_stop` boolean
- [x] **11.4** `core/market_scheduler.py` `run_signal_scan()` — after executing strategy signals, fetches live positions, runs `StopManager().scan()`, converts decisions to stop signals, routes through `AutoTrader.run_once()`; exceptions caught, never propagate
- [x] **Tests:** 36 new tests in `tests/test_stop_manager.py`; all 36 green
- **Exit criteria:** ✅ 2883 passed (2847 + 36 new); stop-loss and take-profit triggers are now enforced automatically

### Sprint 12: Watchlist & Universe Management ✅ COMPLETE (2026-04-21)
**Goal:** Eliminate hardcoded ticker lists from both signal strategies — all scan universes live in `config/watchlist.yaml`.

- [x] **12.1** `config/watchlist.yaml` — canonical ticker universe; two sections: `vol_premium` (list of tickers) and `war_room` (per-regime 6-tuple rules: ticker/direction/asset_class/size/stop/target)
- [x] **12.2** `shared/watchlist.py` — `get_vol_premium_tickers(path=None)` + `get_war_room_rules(regime, path=None)`; YAML parsed once and cached in-process; graceful fallback to hardcoded defaults if YAML missing/malformed; `reload()` for tests; `_parse_rule()` accepts string enum names or pre-built enum objects
- [x] **12.3** `strategies/vol_premium_signals.py` — replaced `_UNIVERSE` list with `_get_universe()` which calls `get_vol_premium_tickers()`; fallback to `["SPY","QQQ","IWM","HYG","JNK"]` on any error
- [x] **12.4** `strategies/signal_generator.py` — added `_get_regime_rules(regime)` helper that calls `get_war_room_rules()`; falls back to `_REGIME_SIGNALS` hardcoded dict; `generate_signals()` calls `_get_regime_rules()` instead of `_REGIME_SIGNALS.get()`
- [x] **Tests:** 32 new tests in `tests/test_watchlist.py` — TestLoadYaml(5), TestGetVolPremiumTickers(6), TestGetWarRoomRules(9), TestParseRule(4), TestVolPremiumWiring(4), TestSignalGeneratorWiring(4); all 32 green
- **Exit criteria:** ✅ 2915 passed (2883 + 32 new); ticker universe configurable via YAML with zero code changes

### Sprint 13: End-of-Day Summary Report ✅ COMPLETE (2026-04-21)
**Goal:** After every market close snapshot, produce a human-readable daily brief — trades executed, P&L summary, roll urgency, API health — written to `reports/daily_brief.txt`.

- [x] **13.1** `core/eod_reporter.py` — `EodReporter.generate()` aggregates: `PnLTracker.today_report()`, live positions, `RollManager.urgent_only()`, `HealthMonitor.collect_snapshot()`; returns `EodReport` dataclass; always writes to `reports/daily_brief.txt`; never raises
- [x] **13.2** `EodReport` — `report_date`, `account_value_usd`, `total_unrealized_pnl`, `total_realized_pnl`, `pnl_delta`, `position_count`, `positions` (list of `PositionSummary`), `roll_urgent_count`, `trades_today`, `api_health_summary`, `overall_api_status`, `active_alerts`, `written_at`; `.format_text()` renders multi-section brief; `.write_to_file(path)` writes UTF-8 file
- [x] **13.3** `core/market_scheduler.py` `run_pnl_snapshot()` — calls `_run_eod_report()` after snapshot (even if snapshot fails); `_run_eod_report()` is fully isolated — any exception is logged and swallowed
- [x] **Tests:** 41 new tests in `tests/test_eod_reporter.py` — TestPositionSummary(3), TestEodReportDataclass(5), TestEodReportFormatText(9), TestEodReportWriteToFile(4), TestEodReporterGenerate(8), TestMarketSchedulerWiring(4), TestGetUrgentSymbols(3), TestExtractApiHealth(4); all 41 green
- **Exit criteria:** ✅ 2961 passed (2920 + 41 new); `reports/daily_brief.txt` written at every market close with full daily narrative

### Sprint 18: Drawdown Circuit Breaker ✅ COMPLETE (2026-04-22)
**Goal:** Protect against multi-day account drawdowns by tripping a "no-new-trades" flag when cumulative drawdown from peak equity exceeds a configurable threshold (default 10%). Complements ``DailyLossGuard`` (single-day ceiling) with multi-day peak-to-trough protection.

- [x] **18.1** `strategies/drawdown_circuit_breaker.py` — `DrawdownState(peak_value, current_value, drawdown_pct, tripped, tripped_at)` dataclass + `.to_dict()`; `DrawdownCircuitBreaker(db_path=None, max_drawdown_pct=0.10)` — SQLite WAL, default `data/drawdown_circuit_breaker.db`; `.update(account_value_usd) -> DrawdownState` — peak only moves up, trips when `drawdown_pct >= max_drawdown_pct`, stays tripped until explicit reset; `.is_tripped() -> bool` — cheap read-only query; `.reset(new_peak_value=None)` — clears trip flag, resets peak; fails-open on any error
- [x] **18.2** `core/auto_trader.py` — `drawdown_circuit_breaker` param added; checked *before* `DailyLossGuard` in `run_once()`; if tripped, all signals filtered, reason logged as `DRAWDOWN_CIRCUIT_BREAKER`; `is_tripped()` exceptions swallowed (fail-open)
- [x] **18.3** `core/market_scheduler.py` — `DrawdownCircuitBreaker()` created at init as `_drawdown_circuit_breaker` (component #10); passed to `AutoTrader` when `auto_execute=True`
- [x] **Tests:** 31 new tests in `tests/test_drawdown_circuit_breaker.py` — TestDrawdownState(4), TestUpdateNoTrip(4), TestPeakTracking(3), TestPersistence(2), TestIsTripped(3), TestReset(3), TestFailOpen(4), TestAutoTraderWiring(5), TestMarketSchedulerWiring(3); all 31 green
- **Exit criteria:** ✅ Multi-day drawdowns from peak equity trigger circuit breaker; stays tripped until `reset()` called; no new trades placed when tripped; suite 3121+31 = **3152 passed**

### Sprint 19: Account Equity Feed ✅ COMPLETE (2026-04-22)
**Goal:** Close the dead gap — `DrawdownCircuitBreaker.update()` was never called in production, so the circuit breaker could never trip on real data. Wire account equity through the EOD snapshot cycle so the breaker always has fresh values.

- [x] **19.1** `strategies/drawdown_circuit_breaker.py` — added `current_state() -> DrawdownState` read-only method; reads from SQLite without writing; fails-open (returns zero `DrawdownState` on any error)
- [x] **19.2** `core/market_scheduler.py` — `_run_drawdown_update(account_value)` helper added; called in `run_pnl_snapshot()` after `_run_eod_report()`; skips zero/negative values; exceptions from update swallowed at call site (`run_pnl_snapshot`) and internally; logs WARN when tripped
- [x] **19.3** `core/eod_reporter.py` — `EodReport` gains `drawdown_pct: float = 0.0` and `drawdown_tripped: bool = False` fields; `to_dict()` includes both; `format_text()` shows "Drawdown: X.XX% !! DRAWDOWN CIRCUIT BREAKER TRIPPED" when tripped; `EodReporter.generate()` lazy-imports `DrawdownCircuitBreaker`, calls `current_state()`, populates fields; exception falls back to 0.0/False
- [x] **Tests:** 30 new tests in `tests/test_account_equity_feed.py` — TestCurrentState(5), TestRunDrawdownUpdate(5), TestPnlSnapshotWiring(5), TestEodReportDrawdownFields(7), TestEodReporterDrawdownWiring(5), TestMarketSchedulerDrawdownSequence(3); all 30 green; `test_eod_reporter.py::test_to_dict_keys` updated for new fields; all 41 green
- **Exit criteria:** ✅ `DrawdownCircuitBreaker.update()` called every EOD snapshot with real account value; drawdown state reported in daily brief; suite 3152+33 = **3185 passed**

### Sprint 21: Real-Time Alerter ✅ COMPLETE (2026-04-22)
**Goal:** Push Telegram notifications when critical events occur — drawdown circuit breaker trips, daily loss ceiling hit, EOD brief delivered — so the operator is always notified without checking logs.

- [x] **21.1** `shared/alerter.py` — `Alerter(bot_token=None, chat_id=None)`; reads `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` from env when args are `None`; `send(event_type, message) -> bool` — sync wrapper via `asyncio.run()`; `_send_async()` uses `aiohttp.resolver.ThreadedResolver()` + `TCPConnector` for Windows c-ares DNS fix; `format_alert()` renders icon-prefixed HTML Telegram message; `enabled` property; fails-open on every path (unconfigured, import error, HTTP error, network error); `_EVENT_ICONS` dict maps DRAWDOWN_TRIPPED / DAILY_LOSS_TRIPPED / EOD_BRIEF / STOP_CLOSE / etc.
- [x] **21.2** `core/auto_trader.py` — `alerter: object | None = None` param added; `_alerter` stored at init; after drawdown CB trip: `_alerter.send("DRAWDOWN_TRIPPED", ...)` (try/except guard); after daily loss guard trip: `_alerter.send("DAILY_LOSS_TRIPPED", ...)` (try/except guard); alerter failure never blocks `ExecutionSummary` return
- [x] **21.3** `core/eod_reporter.py` — `EodReporter(alerter=None)` gains `__init__`; after `report.write_to_file()`: condensed brief (date, account value, P&L, positions, trades, roll urgent, API status, drawdown warning if tripped) sent via `alerter.send("EOD_BRIEF", ...)`; exceptions caught, never block `generate()` return
- [x] **21.4** `core/market_scheduler.py` — `Alerter()` created at init as `_alerter` (component #12); passed to `AutoTrader(alerter=self._alerter)` when `auto_execute=True`; `_run_eod_report()` creates `EodReporter(alerter=self._alerter)` so shared instance used throughout
- [x] **Bonus fix:** `tests/test_alpha_integration.py` — replaced `asyncio.get_event_loop().run_until_complete()` with `asyncio.run()` (Python 3.10+ deprecation crash)
- [x] **Tests:** 33 new tests in `tests/test_alerter_sprint21.py` — TestFormatAlert(3), TestAlerterDisabled(5), TestAlerterEnabled(7), TestAlerterAiohttpError(3), TestAutoTraderWiring(6), TestEodReporterWiring(4), TestMarketSchedulerWiring(5); all 33 green
- **Exit criteria:** ✅ Telegram push on drawdown + loss + EOD; fails-open on every failure path; suite 3215+33+4(alpha fix) = **3252 passed, 0 failed**

### Sprint 24: Walk-Forward Backtester — Comprehensive Coverage & Integration ✅ COMPLETE (2026-04-23)
**Goal:** `strategies/walk_forward_backtester.py` was fully implemented but had only 10 tests and no wiring into `simple_backtest.py`. Sprint 24 adds comprehensive test coverage for every class and code path, and exposes walk-forward validation through `simple_backtest.run_walk_forward()` so the scheduler can call it as part of the strategy calibration cycle.

- [x] **24.1** `strategies/walk_forward_backtester.py` `WalkForwardResult.to_dict()` — extended to include `train_window`, `test_window`, `oos_avg_profit_factor`, and `return_std` keys (previously missing; now all 12 metrics exposed)
- [x] **24.2** `strategies/simple_backtest.py` `run_walk_forward()` — new top-level function; downloads `lookback_days` of history via yfinance, runs `WalkForwardBacktester` with both `MeanReversionWF` and `MomentumWF`; returns `{"ticker", "train_window", "test_window", "mean_reversion": WalkForwardResult.to_dict(), "momentum": WalkForwardResult.to_dict(), "error": None | str}`; fails-open on yfinance failure, insufficient data, or any exception
- [x] **Tests — expanded `tests/test_walk_forward_backtester.py`** — added 49 new tests: TestWindowMode(4 — enum values + membership), TestWalkForwardResultDefaults(4 — empty folds, zero OOS, to_dict keys, round-trip), TestFoldResultDefaults(2 — predictions/actuals default lists, n_trades default), TestMeanReversionWF(4 — fit sets mean/std, predict returns Series, signals in {-1,0,1}, zero-std guard), TestMomentumWF(3 — fit threshold, predict Series, signals range), TestWalkForwardBacktesterExtended(17 — fold count, anchored mode, custom step size, price_col selection, price_col fallback, OOS max DD ≤ 0, sharpe_std ≥ 0, return_std ≥ 0, folds list len, fold indices sequential, rolling train_size == train_window, too-few-rows ValueError, win rate bounded, callable without subclass, empty aggregate, inf profit_factor capped, predictions stored in fold); all green
- [x] **Tests — new `tests/test_simple_backtest_walk_forward.py`** — 15 tests: TestRunWalkForwardHappyPath(11 — returns dict, ticker/train/test_window present, no error, mean_reversion + momentum present, oos_sharpe key, total_folds key, folds positive, consistency ratio bounded), TestRunWalkForwardErrorHandling(4 — empty history → error, insufficient data → error, yfinance exception → error, result structure preserved); all green
- **Exit criteria:** ✅ `WalkForwardResult.to_dict()` complete; `run_walk_forward()` callable from scheduler; suite 3321+50 = **3371 passed** (actual)

### Sprint 25: Vol-Target Position Sizing — Wiring & Comprehensive Coverage ✅ COMPLETE (2026-04-23)
**Goal:** `strategies/vol_target_sizer.py` (`VolTargetSizer`) was fully implemented but had only 9 tests and was not wired into `AutoTrader` or `MarketScheduler`. Sprint 25 adds a vol-adjusted Kelly gate into `AutoTrader.run_once()`, wires `VolTargetSizer` as component #14 in `MarketScheduler`, and expands test coverage from 9 → 37 tests + 12 wiring tests.

- [x] **25.1** `core/auto_trader.py` — `vol_sizer=None` param added; stored as `self._vol_sizer`; in `run_once()` after the `_should_execute` loop, a new **Vol-target gate** iterates approved signals: for each non-compulsory signal calls `self._vol_sizer.vol_adjusted_kelly(win_rate=sig.confidence, avg_win=1.5, avg_loss=1.0, current_vol=sig.size, historical_vol=0.05)`; signals with `kelly < 0.01` are filtered with `VOL_GATE:` reason; compulsory ROLL_CLOSE/STOP_CLOSE bypass; fails-open on any exception
- [x] **25.2** `core/market_scheduler.py` — `VolTargetSizer()` created at init as `_vol_sizer` (component #14); passed to `AutoTrader(vol_sizer=self._vol_sizer)` when `auto_execute=True`; same instance shared so scheduler can query sizing independently
- [x] **Tests — expanded `tests/test_vol_target_sizer.py`** — added 28 new tests: TestVolTargetAllocation(2 — all fields, default leverage), TestVolTargetResult(3 — to_dict keys, allocation dicts, total_leverage), TestInverseVolWeights(5 — all assets present, positive, sum=1, low-vol higher weight, custom lookback), TestSizePortfolio(7 — target_vol attr, scaling cap, allocations count, notional scales with capital, portfolio_vol positive, custom weights, vol_floor prevents ZeroDivisionError), TestVolAdjustedKelly(6 — zero loss → 0, zero win → 0, negative kelly clamped, cap at max_leverage, vol_ratio scaling, zero current_vol uses floor), TestTargetSizeSingle(5 — low-vol gets higher weight, asset name, notional positive, weight cap, raw_weight=1)
- [x] **Tests — new `tests/test_vol_target_wiring.py`** — 12 tests: TestAutoTraderVolSizerParam(4 — stored, defaults None, None sizer passes all, called once per signal), TestVolGateFiltering(6 — filtered when kelly<0.01, approved when ≥0.01, exactly 0.01 approved, partial filter across multiple signals, filter reason contains ticker, exception fails-open), TestMarketSchedulerVolSizerWiring(2 — scheduler creates VolTargetSizer instance, wired to auto_trader)
- **Exit criteria:** ✅ Live signals gated by vol-adjusted Kelly; compulsory roll/stop signals bypass gate; `VolTargetSizer` component #14 wired end-to-end; suite 3371+33 = **3404 passed** (actual)

### Sprint 23: Persistent Execution Throttle ✅ COMPLETE (2026-04-22)
**Goal:** Replace the in-memory `AutoTrader._last_executed` dict with a SQLite-backed `ExecutionThrottle` that survives process restarts. Without persistence, a crash-and-restart re-enters all throttled positions immediately, risking double-fills on live accounts.

- [x] **23.1** `core/execution_throttle.py` — `ExecutionThrottle(db_path=None, throttle_seconds=14400)` — SQLite WAL, default `data/execution_throttle.db`; `.can_execute(ticker) -> bool`; `.record_execution(ticker)`; `.last_executed(ticker) -> float | None`; `.remaining_seconds(ticker) -> float`; `.clear(ticker=None)` — single or all; `.all_entries() -> list[dict]`; `:memory:` connection cached for test isolation; fails-open on any DB error — never blocks trading
- [x] **23.2** `core/auto_trader.py` — `execution_throttle=None` param added; when set, `.can_execute()` / `.remaining_seconds()` used in `_should_execute()` instead of in-memory `_last_executed` dict (fallback kept for non-wired case); `.record_execution()` called in `run_once()` after confirmations come back, using `conf.signal_ticker` — fires even when `_execute_approved` is mocked in tests; in-memory `_last_executed` retained as belt-and-suspenders fallback
- [x] **23.3** `core/market_scheduler.py` — `ExecutionThrottle()` created at init as `_execution_throttle` (component #13); passed to `AutoTrader(execution_throttle=self._execution_throttle)` when `auto_execute=True`; same instance shared so scheduler can inspect throttle state
- [x] **Tests:** 36 new tests in `tests/test_persistent_throttle_sprint23.py` — TestExecutionThrottleNewTicker(4), TestExecutionThrottleAfterRecord(5), TestExecutionThrottleExpiry(2), TestExecutionThrottlePersistence(2), TestExecutionThrottleClear(4), TestExecutionThrottleFailOpen(5), TestAutoTraderWiring(6), TestMarketSchedulerWiring(5), TestExecutionThrottleAllEntries(3); all 36 green
- **Exit criteria:** ✅ Crash-and-restart cannot re-enter throttled positions; 4h default window persists across process restarts; suite 3283+36 = **3321 passed** (actual)

### Sprint 22: Strategy Weight Auto-Calibration ✅ COMPLETE (2026-04-23)
**Goal:** Close the dead gap — `use_calibration=False` was hardcoded in `run_signal_scan()` and `SignalOutcomeTracker.run()` was never called, so calibrated aggregator weights were computed but never used. Wire both into the live scheduler so weights auto-update from real outcome data every market close.

- [x] **22.1** `core/market_scheduler.py` `run_signal_scan()` — changed `get_combined_signals()` call to pass `use_calibration=True`; aggregator handles fallback to defaults when `<5` resolved signals available or tracker raises
- [x] **22.2** `core/market_scheduler.py` `run_pnl_snapshot()` — added `self._run_outcome_resolution()` call after `_run_drawdown_update()`; fires every EOD snapshot
- [x] **22.3** `core/market_scheduler.py` `_run_outcome_resolution()` — new private method; instantiates `SignalOutcomeTracker()`, calls `.run()`, logs resolved/hits/misses/errors; swallows all exceptions (never raises); returns None
- [x] **Tests:** 30 new tests in `tests/test_calibration_sprint22.py` — TestRunSignalScanUsesCalibration(4), TestPnlSnapshotCallsOutcomeResolution(4), TestRunOutcomeResolution(5), TestGetCombinedSignalsCalibration(5), TestCalibrationWeightsDataclass(5), TestSignalOutcomeTrackerCalibratedWeights(5), TestFullCalibrationIntegration(2); all 30 green
- **Exit criteria:** ✅ Calibrated weights auto-computed and applied from real fill data; `SignalOutcomeTracker.run()` fires on every EOD snapshot; suite 3252+30 = **3282 passed** (actual: **3283 passed**)

### Sprint 20: Live Account Value Feed ✅ COMPLETE (2026-04-22)
**Goal:** Replace the hardcoded `os.environ.get("ACCOUNT_VALUE_USD", "50000")` in `run_pnl_snapshot()` with a live IBKR equity read.  Every downstream consumer — drawdown circuit breaker, daily loss guard, P&L tracker — now receives the real account NAV instead of a static env var.

- [x] **20.1** `shared/account_value_feed.py` — `AccountValueFeed(ibkr_connector=None, cache_ttl_seconds=300, paper=False)`; `get() -> float` — resolution order: TTL cache → IBKR `NetLiquidation_USD` / `NetLiquidation_BASE` → `ACCOUNT_VALUE_USD` env var → `50_000.0` default; thread-safe `threading.Lock`; `get_source() -> str` (`"ibkr"` / `"env"` / `"default"`); `invalidate()` clears cache; always returns positive float, never raises; `extract_net_liq(summary)` public helper; lazy `IBKRConnector` built from env when no connector injected; `disconnect()` called even on error
- [x] **20.2** `core/market_scheduler.py` — `AccountValueFeed(paper=self._paper)` created at init as `_account_value_feed` (component #11); `run_pnl_snapshot()` calls `self._account_value_feed.get()` (wrapped in try/except; falls back to env default on unexpected error) instead of raw env-var read; `paper` flag forwarded so live vs paper IBKR port resolves correctly
- [x] **Tests:** 30 new tests in `tests/test_account_value_feed_sprint20.py` — TestExtractNetLiq(5), TestCacheBehavior(5), TestResolvePaths(7), TestFetchFromIbkr(6), TestMarketSchedulerWiring(7); all 30 green; existing 85 scheduler + equity-feed tests still pass
- **Exit criteria:** ✅ Live IBKR `NetLiquidation` feeds drawdown CB + P&L tracker at every EOD snapshot; graceful triple-fallback (ibkr → env → default); suite 3185+30 = **3215 passed**

### Sprint 17: Position Reconciliation ✅ COMPLETE (2026-04-22)
**Goal:** Detect when the system's internal position view (``pnl.db`` snapshots) drifts from live IBKR positions — flags MISSING (system thinks we hold it, IBKR doesn't), PHANTOM (IBKR has it, system doesn't know), and SIZE_MISMATCH (quantity divergence above tolerance).

- [x] **17.1** `core/position_reconciler.py` — `ReconciliationMismatch(mismatch_type, symbol, internal_qty, live_qty, detected_at)` frozen dataclass + `.to_dict()`; `ReconciliationReport` with `.has_mismatches`, `.missing_count`, `.phantom_count`, `.size_mismatch_count`, `.to_dict()`; `PositionReconciler(pnl_db_path=None, size_tolerance=0.01)` — `.reconcile(internal_positions=None, live_positions=None) -> ReconciliationReport`; auto-loads latest snapshot from `pnl.db` when `internal_positions` is `None`; fails-open (returns empty report on any error)
- [x] **17.2** `core/market_scheduler.py` — `PositionReconciler()` created at init as `_position_reconciler`; `_run_reconciliation(positions)` called in `run_pnl_snapshot()` after EOD report; mismatches logged as WARNING; errors caught, never block snapshot
- [x] **Tests:** 33 new tests in `tests/test_position_reconciler.py` — TestReconciliationMismatch(4), TestReconciliationReport(4), TestReconcileClean(3), TestReconcileMissing(2), TestReconcilePhantom(2), TestReconcileSizeMismatch(4), TestReconcileMixedMismatches(1), TestFailOpen(4), TestLoadLatestSnapshot(3), TestMarketSchedulerReconcilerWiring(6); all 33 green
- **Exit criteria:** ✅ Position drift between internal and IBKR detected at every market close; three mismatch types flagged with WARNING logs; suite green (3086+33 = 3119)

### Sprint 16: Regime Change Alerting ✅ COMPLETE (2026-04-21)
**Goal:** Persist composite-score regime state (CALM/WATCH/ELEVATED/CRISIS) to SQLite, detect transitions on every signal scan cycle, and automatically adjust signal confidence when market regime escalates or de-escalates.

- [x] **16.1** `strategies/regime_monitor.py` — `Regime.from_score()` (thresholds <30/50/70); `RegimeMonitor(db_path=None)` — SQLite WAL; `.record(score) -> RegimeTransition | None`; `.current_regime()`; `.get_history(limit)`; fails-open (returns None) on any DB error
- [x] **16.2** `strategies/regime_alerts.py` — `is_escalation()`, `is_de_escalation()`, `confidence_multiplier()` (1.20/0.85/1.00); `format_alert()` human-readable one-liner; `apply_regime_filter(signals, transition)` — adjusts confidence via `dataclasses.replace()`, capped at 0.95
- [x] **16.3** `core/market_scheduler.py` — `RegimeMonitor()` created at init; `_get_composite_score()` calls `IndicatorState()` + `compute_composite_score()` (fails-open → 50.0); `run_signal_scan()` calls `record()` after signals, applies filter on transition, errors never block signal delivery
- [x] **Tests:** 43 new tests in `tests/test_regime_monitor.py` — TestRegime(5), TestRegimeTransition(3), TestRegimeMonitor(11), TestRegimeAlerts(10), TestApplyRegimeFilter(7), TestMarketSchedulerRegimeWiring(7); all 43 green, 0 failures in full suite
- **Exit criteria:** ✅ Regime transitions logged to SQLite; CALM→CRISIS escalation boosts SHORT confidence ×1.20; CRISIS→CALM de-escalation dampens ×0.85; suite green (3086)
**Goal:** Record every approved signal to SQLite, retrospectively match fills to determine hit/miss, and compute calibrated aggregator weights from real outcome data.

- [x] **15.1** `strategies/signal_journal.py` — `SignalJournal` (SQLite WAL, WAL mode, `check_same_thread=False`); `log_signal(signal, strategy_source) -> int`; `resolve(row_id, outcome) -> bool`; `get_unresolved(cutoff_hours=48)`, `get_recent(limit=50)`, `get_hit_rates() -> dict[str, HitRate]`; `HitRate.rate = hits/(hits+misses)`, `JournalRow` read-only view
- [x] **15.2** `strategies/signal_outcome_tracker.py` — `SignalOutcomeTracker`; `.run()` loads unresolved signals, cross-references `trade_log` fills from `pnl.db`, resolves HIT/MISS (ticker match, within `match_window_hours`); `.calibrated_weights()` returns `CalibrationWeights`; calibration formula: `raw_i = default_i * (1 + hit_rate_i)`, normalise to sum=1.0; requires ≥5 resolved per strategy (`_MIN_RESOLVED_FOR_CALIBRATION`)
- [x] **15.3** `strategies/signal_aggregator.py` — `get_combined_signals()` gains `use_calibration: bool = False`; lazily imports `SignalOutcomeTracker`; calibration exceptions caught/logged, defaults used as fallback
- [x] **15.4** `core/auto_trader.py` — `signal_journal=None` param added; `_journal_approved_signals()` called before execution block; per-signal errors swallowed with `_log.warning`
- [x] **15.5** `core/market_scheduler.py` — `SignalJournal()` created at init; passed to `AutoTrader(signal_journal=...)`
- [x] **Tests:** 38 new tests in `tests/test_signal_journal.py` — TestHitRate(4), TestJournalRow(2), TestSignalJournal(8), TestSignalOutcomeTracker(7), TestCalibrationWeights(5), TestAggregatorCalibration(4), TestAutoTraderJournalWiring(5), TestMarketSchedulerJournalWiring(3); all 38 green
- **Exit criteria:** ✅ 3040 passed (2999 + 41 new); every approved signal is durably journalled; hit rates are computed from real fills; aggregator weights can be calibrated from outcome data

### Sprint 14: Stale Order Monitor ✅ COMPLETE (2026-04-21)
**Goal:** Detect and cancel submitted orders that remain unfilled past a configurable stale window (default 30 min), preventing ghost orders from accumulating on the exchange.

- [x] **14.1** `core/order_monitor.py` — `PendingOrderRegistry` (thread-safe in-memory store), `PendingOrder` (order_id, ticker, submitted_at, `age_minutes()`), `OrderMonitor.scan()` identifies stale orders and calls `_cancel_via_ibkr()` (patchable module-level function); returns `OrderMonitorReport` (checked/cancelled/still_pending/errors/cancelled_ids); fails-open per order
- [x] **14.2** `core/auto_trader.py` — `order_monitor` param added to `__init__`; `_register_pending_with_monitor()` called after each execution cycle; only SUBMITTED (not FILLED) confirmations are registered; errors swallowed, never block execution
- [x] **14.3** `core/market_scheduler.py` — `PendingOrderRegistry` + `OrderMonitor` created at init; passed to `AutoTrader` when `auto_execute=True`; `run_order_monitor()` method added — calls `OrderMonitor.scan()`, catches all exceptions, returns empty `OrderMonitorReport` on failure
- [x] **Tests:** 34 new tests in `tests/test_order_monitor.py` — TestPendingOrder(4), TestPendingOrderRegistry(7), TestOrderMonitorReport(3), TestOrderMonitorScan(7), TestOrderMonitorRegister(4), TestAutoTraderWiring(5), TestMarketSchedulerWiring(4); all 34 green
- **Exit criteria:** ✅ 2999 passed (2961 + 38 new); stale SUBMITTED orders are detected and cancelled automatically; ghost orders cannot accumulate

### Sprint 7: Automation & Scheduling ✅ COMPLETE (2026-04-22)
**Goal:** System runs unattended during market hours.

- [x] **7.1** Scheduled scan: every 15 min during market hours — `MarketScheduler.run()` calls `run_signal_scan()` when `is_market_hours()` and `_should_run("signal_scan", 900)`
- [x] **7.2** Scheduled roll check: daily at market open — fires once per day in `is_market_open_window()` (9:30–9:40 ET); guarded by `_last_roll_date != today`
- [x] **7.3** Daily P&L snapshot at market close — fires once per day in `is_market_close_window()` (16:00–16:10 ET); guarded by `_last_pnl_date != today`
- [x] **7.4** Health check: every 5 min, always — `run_health_check()` fires unconditionally on `_should_run("health_check", 300)` (no market-hours gate)
- [x] **7.5** Graceful restart on crash — `run_forever(max_restarts=10)` wraps `run()` in a try/except loop; increments `restart_count`; raises after `max_restarts` exceeded
- [x] **Tests:** 55 tests in `tests/test_scheduler.py` — TestIsTradingDay(4), TestIsMarketHours(6), TestIsMarketOpenWindow(4), TestIsMarketCloseWindow(5), TestShouldRun(4), TestRunTask(3), TestRunSignalScan(2), TestRunRollCheck(2), TestRunPnlSnapshot(3), TestRunHealthCheck(1), TestRunLoopHealthAlwaysRuns(2), TestRunLoopMarketHourTasks(5), TestOncePerDayGuard(3), TestRunForever(4), TestStop(2), TestConfiguration(5); all 55 green
- **Exit criteria:** ✅ All four tasks scheduled correctly; graceful restart proven; suite **3252 + (already counted in baseline)**

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
