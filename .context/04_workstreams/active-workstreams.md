# Active Workstreams

> Last updated: 2026-04-09

1. **Context Guardrails (DONE)**
   - `.github/copilot-instructions.md` — AUTO-READ by Copilot, master context
   - `AGENTS.md` (root) — AI agent behavioral rules
   - `.context/STATUS.md` — living status dashboard
   - `.github/instructions/` — path-specific instructions (trading, integrations, tests)
   - Root cleanup: 170+ → 30 files (99 to `_scratch/`, 22 reports to `docs/archive/`)

2. **UW Field Parsing Fix (DONE)**
   - Fixed 2026-03-31 — field mapping updated in `integrations/unusual_whales_client.py`
   - API schema change resolved: strike_price, premium, put_call fields now parsing correctly

3. Gap Reduction
   - Use local scanners first.
   - Treat abstract base methods separately from real implementation gaps.
   - Capture each verified gap batch in `07_gaps/`.

4. Monitoring Reliability
   - Dashboard callbacks should degrade gracefully when caches are empty.
   - Monitoring helpers should auto-return initialized singleton components.
   - Matrix Monitor: 24/30 collectors OK (6 timeout expected — IBKR/NCC/NCL offline).

5. Execution Determinism
   - Simulation helpers should use deterministic slippage and scoring where possible.
   - Avoid random behavior in operational health and risk paths.

6. Context Consolidation
   - Keep durable session summaries in `.context/06_sessions/`.
   - Keep repo-memory notes short and high-signal.
   - **UPDATE `.context/STATUS.md` after every significant work session.**

7. **Architecture Rework v3.3 (DONE)**
   - Phase 1: Wire 7 strategies — DONE
   - Phase 2: Fix MI gap — DONE
   - Phase 3: Strategy Advisor Engine — DONE (`strategies/strategy_advisor_engine.py`, 30-min orchestrator loop)
   - Phase 4: Doctrine rework — DONE (`aac/doctrine/strategic_doctrine.py`, terrain loop in orchestrator)
   - Phase 5: NCL relay — DONE (`shared/ncc_relay_client.py`, heartbeat loop in orchestrator)
   - Phase 6: Monitor display — DONE (33 collectors in master dashboard, paper divisions added)
   - Phase 7: Connector alignment — DONE

8. **Polymarket Division Activation (DONE)**
   - `strategies/polymarket_division/active_scanner.py` — Unified live trading engine (450+ lines)
     - 3 strategies: War Room Poly, PolyMC Agent, PlanktonXD Harvester
     - Modes: scan (one-shot), monitor (continuous dry-run), live (continuous + execute)
     - Edge detection: min 3% threshold, MC simulation, deep OTM micro-arbitrage
     - Execution: DRY_RUN=true default, 50 daily bet limit, $25 max position
     - Report generation: human-readable + JSON

9. **Paper Trading Divisions (ACTIVE)**
   - `divisions/trading/polymarket_paper/` — 5 strategies (Grid, DCA, Momentum, MeanRev, Arbitrage), $10K virtual
   - `divisions/trading/crypto_paper/` — 4 strategies (Grid, DCA, Momentum, MeanRev), $10K virtual
   - Enterprise wired: 11 divisions total, council→paper→warroom signal flow
   - Bakeoff YAML configs created: `aac/bakeoff/metrics/metric_canon.yaml`, `policy/bakeoff_policy.yaml`, `checklists/gate_checklists.yaml`
   - Doctrine: `.context/04_workstreams/paper-trading-doctrine.md` — 13-week roadmap, 8 mandates, gate promotion criteria
   - Tests: 47 tests in `tests/test_paper_trading_divisions.py` — all green
   - **Phase:** RUNNING — smoke test 16/16 pass, monitor collectors wired, persistence verified
   - `strategies/polymarket_division/__init__.py` — Division registry (4 entries + get_division_status())
   - `launch.py polymarket` — CLI mode added (scan/monitor/live + --interval/--max-cycles/--json)
   - SDK: py-clob-client v0.34.6 verified, CLOB API authenticated
   - Wallet: LIVE — 2 open orders, 124 trades, $535.73 USDC
   - Internet research: Official Polymarket docs (market makers, trading, inventory, builder program)
   - `_scratch/_test_poly_connectivity.py` — diagnostic script confirming wallet connectivity

10. **13-Moon Doctrine (ACTIVE)**
    - Engine: `strategies/thirteen_moon_doctrine.py` (~1400 lines) — 14 moon cycles, 184 events, 6 overlay layers
    - Storyboard: `strategies/thirteen_moon_storyboard.py` (~950 lines) — interactive HTML timeline
    - Output: `data/storyboard/thirteen_moon_storyboard.html` + JSON export
    - **Current:** Moon 1 (Pink Moon) — DEPLOY mandate
    - **Timeline:** Moon 0 (Mar 3, 2026 Total Lunar Eclipse) → Moon 13 (Apr 19, 2027)
    - **Layers:** Astrology (41), Phi Coherence (14), Financial (33), World News (20), Doctrine Actions (14), AAC System (~62)
    - AAC sub-layers: trades, options lifecycle, war room, milestones, seesaw, scenarios, strategies, automation
    - Lead-time alert system: 14 event categories with escalating day-threshold actions
    - Saturn-Neptune conjunction deepdive (Feb 20, 2027 at 0° Aries — 36-year cycle)
    - Integrated into Matrix Monitor sidebar (open/regenerate storyboard)
    - CLI: `python -m strategies.thirteen_moon_doctrine --upcoming 30`

11. **War Room Live Feeds (ACTIVE)**
    - `strategies/war_room_live_feeds.py` — 12 async fetchers, 15 indicators wired to live APIs
    - `strategies/war_room_engine.py` — Composite scoring (0-100), regime detection (CALM/WATCH/ELEVATED/CRISIS)
    - **Current:** Composite 39.6 | REGIME: WATCH | 10/12 feeds returning live data
    - Feeds working: CoinGecko, FRED, Finnhub, yfinance (BDC), Unusual Whales, Fear & Greed, NewsAPI, MetaMask, NDAX, **IBKR (TWS on port 7497, LIVE)**
    - Feeds offline: X/Twitter (HTTP 402)
    - **Moomoo OpenD**: Process running but API port 11111 NOT listening. Dummy credentials in OpenD.xml — user must update + restart.
    - CoinGecko global data bug FIXED — DeFi cap, BTC dominance now extracting
    - New BDC fetcher: yfinance basket (ARCC/MAIN/FSK/OBDC) → NAV discount + non-accrual proxy
