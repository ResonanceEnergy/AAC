# Current State

Date: 2026-04-07
Repo: AAC
Branch: `main`
Workspace: `c:\dev\AAC_fresh`
Python: 3.12 via `.venv\Scripts\python.exe`
Version: 3.6.0 (pyproject.toml)

## Operational Summary

AAC is in **LIVE TRADING** mode. Apr 6 post-mortem completed — 8 puts expired worthless, ROLL_DISCIPLINE rules encoded.

### What's Running
- IBKR: 5 active puts (XLF May 1, LQD/EMB May 15, BKLN x3/HYG Jun 18) — all OTM
- IBKR: 4 Apr 17 puts expiring worthless (ARCC/PFF/MAIN/JNK) — $150 premium lost
- Moomoo: Real mode, FUTUCA, ~$17,684 USD — SQQQ x172 + SPXS x106 performing in VIX 24
- WealthSimple TFSA: ~$18,638 CAD — 3 active (GLD/XLE LEAPS + OWL put), 4 Apr puts expiring worthless ($1,495 lost)
- Matrix Monitor: 4 display modes, 20+ panels, 24/30 collectors OK
- Doctrine Engine: 12 packs, 4-state machine
- 8 strategies wired to unified integrator
- **Polymarket Division**: ACTIVE — $535.73 USDC, active_scanner.py
- Grand total: ~$32,538 USD | VIX 24.05 | CAD/USD 0.7189
- Pytest: **1714 passed**, 23 skipped, 1 xfailed

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
