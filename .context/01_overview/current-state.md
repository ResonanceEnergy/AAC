# Current State

Date: 2026-04-06
Repo: AAC
Branch: `main`
Workspace: `c:\dev\AAC_fresh`
Python: 3.12 via `.venv\Scripts\python.exe`
Version: 3.6.0 (pyproject.toml)

## Operational Summary

AAC is in **LIVE TRADING** mode with real money positions on IBKR and an active Polymarket division.

### What's Running
- IBKR: 9 live put positions ($910+ invested) — ARCC, PFF, LQD, EMB, MAIN, JNK, XLF, BKLN x3, HYG
- Moomoo: Real mode, FUTUCA, $2,609.26 USD — options approval pending
- WealthSimple TFSA: Roll-down plan created (C$614 budget, Apr→Jul)
- Matrix Monitor: 4 display modes, 20+ panels, 24/30 collectors OK
- Doctrine Engine: 12 packs, 4-state machine
- 8 strategies wired to unified integrator (War Room, Storm Lifeboat, Matrix Maximizer, Exploitation Matrix, Polymarket BlackSwan, BlackSwan Authority, PolyMC, Polymarket Active Scanner)
- **Polymarket Division**: ACTIVE — `active_scanner.py` (450+ lines), 3 strategies unified (War Room Poly, PolyMC, PlanktonXD), py-clob-client v0.34.6, wallet live ($535.73 USDC, 2 open orders, 124 trades)
- `launch.py polymarket` mode — scan/monitor/live CLI with `--mode`, `--interval`, `--max-cycles`, `--json`
- Pytest: **1712 passed**, 23 skipped, 1 xfailed

### What's Broken
- CoinGecko: Pro key expired → auto-downgrades to free tier (10 req/min)
- Polygon: Free tier can't do options snapshots (403)
- X/Twitter: HTTP 402 (needs paid tier)

### What Was Fixed
- Unusual Whales: Field parsing FIXED (2026-03-31) — was returning $0 for strike/premium
- KRE $58P / IWM $230P: EXPIRED Apr 4 (removed from active positions)

### Context System
- `.github/copilot-instructions.md` — AUTO-READ by Copilot, master guardrails
- `AGENTS.md` (root) — behavioral rules for AI agents
- `.context/STATUS.md` — living status dashboard (single source of truth)
- `.context/` 10-folder system — durable project context

### Architecture Rework v3.3
- Phase 1: Wire 7 strategies — DONE
- Phase 2: Fix MI gap — DONE
- Phase 3-7: Strategy Advisor, Doctrine rework, NCL relay, Monitor display, Connector alignment — PLANNED

### Polymarket Division (NEW)
- `strategies/polymarket_division/active_scanner.py` — Unified live trading engine (scan/monitor/live modes)
  - 3 concurrent strategy scanners: War Room Poly, PolyMC Agent, PlanktonXD Harvester
  - Edge detection: min 3% threshold, Monte Carlo simulation, deep OTM micro-arbitrage
  - Execution: DRY_RUN=true default, daily bet limit (50), $25 max position
  - Monitor loop: continuous scan→execute→sleep with configurable intervals
  - Report generation: human-readable + JSON output
- `strategies/polymarket_division/__init__.py` — Division registry with 4 entries (war_room, planktonxd, polymc, active_scanner)
- `launch.py polymarket` — CLI mode for Active Scanner
- SDK: py-clob-client v0.34.6 on Polygon (chain 137)
- Wallet: 2 open orders, 124 historical trades, $535.73 USDC
- Market Maker knowledge: GTC/GTD/FOK/FAK order types, batch orders (up to 15), makers never pay fees, inventory split/merge via CTF

## Entry Points
- **Start here:** `.github/copilot-instructions.md` → then this file → then `../STATUS.md`
- **What's broken:** `../STATUS.md`
- **Active work:** `../04_workstreams/active-workstreams.md`
- **Gap analysis:** `../07_gaps/gap-scan-2026-03-16.md`
- **Developer workflow:** `../08_runbooks/developer-workflow.md`
- **Test commands:** `../09_tests/test-commands.md`
