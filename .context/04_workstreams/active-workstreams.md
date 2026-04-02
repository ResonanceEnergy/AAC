# Active Workstreams

> Last updated: 2026-04-10

1. **Context Guardrails (DONE)**
   - `.github/copilot-instructions.md` — AUTO-READ by Copilot, master context
   - `AGENTS.md` (root) — AI agent behavioral rules
   - `.context/STATUS.md` — living status dashboard
   - `.github/instructions/` — path-specific instructions (trading, integrations, tests)
   - Root cleanup: 170+ → 30 files (99 to `_scratch/`, 22 reports to `docs/archive/`)

2. **UW Field Parsing Fix (PLANNED)**
   - `integrations/unusual_whales_client.py` needs field mapping update
   - API schema changed: strike_price, premium, put_call, sentiment return $0/blank
   - Need to inspect actual API response JSON to find new field names

3. Gap Reduction
   - Use local scanners first.
   - Treat abstract base methods separately from real implementation gaps.
   - Capture each verified gap batch in `07_gaps/`.

4. Monitoring Reliability
   - Dashboard callbacks should degrade gracefully when caches are empty.
   - Monitoring helpers should auto-return initialized singleton components.

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
