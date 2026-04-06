# Active Workstreams

> Last updated: 2026-04-06

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

7. Architecture Rework v3.3
   - Phase 1: Wire 7 strategies — DONE
   - Phase 2: Fix MI gap — DONE
   - Phase 3: Strategy Advisor Engine — PLANNED
   - Phase 4: Doctrine rework — PLANNED
   - Phase 5: NCL relay — PLANNED
   - Phase 6: Monitor display — PLANNED
   - Phase 7: Connector alignment — FUTURE

8. **Polymarket Division Activation (DONE)**
   - `strategies/polymarket_division/active_scanner.py` — Unified live trading engine (450+ lines)
     - 3 strategies: War Room Poly, PolyMC Agent, PlanktonXD Harvester
     - Modes: scan (one-shot), monitor (continuous dry-run), live (continuous + execute)
     - Edge detection: min 3% threshold, MC simulation, deep OTM micro-arbitrage
     - Execution: DRY_RUN=true default, 50 daily bet limit, $25 max position
     - Report generation: human-readable + JSON
   - `strategies/polymarket_division/__init__.py` — Division registry (4 entries + get_division_status())
   - `launch.py polymarket` — CLI mode added (scan/monitor/live + --interval/--max-cycles/--json)
   - SDK: py-clob-client v0.34.6 verified, CLOB API authenticated
   - Wallet: LIVE — 2 open orders, 124 trades, $535.73 USDC
   - Internet research: Official Polymarket docs (market makers, trading, inventory, builder program)
   - `_scratch/_test_poly_connectivity.py` — diagnostic script confirming wallet connectivity
