# Current State

Date: 2026-04-10
Repo: AAC
Branch: `main`
Workspace: `c:\dev\AAC_fresh`
Python: 3.12 via `.venv\Scripts\python.exe`
Version: 3.6.0 (pyproject.toml)

## Operational Summary

AAC is in **LIVE TRADING** mode with real money positions on IBKR.

### What's Running
- IBKR: 8 live put positions ($910 invested) — ARCC, PFF, LQD, EMB, MAIN, JNK, KRE, IWM
- Moomoo: Real mode, FUTUCA, $365.15 USD — options approval pending
- WealthSimple TFSA: Roll-down plan created (C$614 budget, Apr→Jul)
- Matrix Monitor: 4 display modes, 20+ panels
- Doctrine Engine: 12 packs, 4-state machine
- 7 strategies wired to unified integrator (War Room, Storm Lifeboat, Matrix Maximizer, Exploitation Matrix, Polymarket BlackSwan, BlackSwan Authority, PolyMC)

### What's Broken
- Unusual Whales: Key works but field parsing broken (API schema changed — strike/premium/$0)
- CoinGecko: Pro key expired → auto-downgrades to free tier (10 req/min)
- Polygon: Free tier can't do options snapshots (403)
- X/Twitter: HTTP 402 (needs paid tier)

### Context System (NEW as of 2026-04-10)
- `.github/copilot-instructions.md` — AUTO-READ by Copilot, master guardrails
- `AGENTS.md` (root) — behavioral rules for AI agents
- `.context/STATUS.md` — living status dashboard (single source of truth)
- `.context/` 10-folder system — durable project context
- Root directory has 170+ items — cleanup to `_scratch/` planned

### Architecture Rework v3.3
- Phase 1: Wire 7 strategies — DONE
- Phase 2: Fix MI gap — DONE
- Phase 3-7: Strategy Advisor, Doctrine rework, NCL relay, Monitor display, Connector alignment — PLANNED

## Entry Points
- **Start here:** `.github/copilot-instructions.md` → then this file → then `../STATUS.md`
- **What's broken:** `../STATUS.md`
- **Active work:** `../04_workstreams/active-workstreams.md`
- **Gap analysis:** `../07_gaps/gap-scan-2026-03-16.md`
- **Developer workflow:** `../08_runbooks/developer-workflow.md`
- **Test commands:** `../09_tests/test-commands.md`
