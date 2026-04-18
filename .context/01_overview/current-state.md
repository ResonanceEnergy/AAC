# Current State

Date: 2026-04-09
Repo: AAC
Branch: `main`
Workspace: `c:\dev\AAC_fresh`
Python: 3.12 via `.venv\Scripts\python.exe`
Version: 3.6.0 (pyproject.toml)

## Operational Summary

AAC is in **LIVE TRADING** mode. Live IBKR data pull Apr 9 confirmed 15 positions (10 new). Port 7497 verified.

### What's Running
- IBKR: 15 active positions (5 calls + 10 puts) — Net liq CAD $20,079.57 (~USD $14,520), Cash CAD $2,700.83
  - Calls: SLV $66C x8, SLV $75C x2 LEAPS, TSLA $500C x1 LEAPS, SLV $70C x2, XLE $65C x3 ($12,143 MV)
  - Puts: OBDC $7.5P x11, BKLN $20P x3, HYG $77P, LQD $106P, EMB $90P, XLF $46P + 4 Apr 17 expiring ($431 MV)
- Moomoo: OpenD running but API port 11111 NOT listening — dummy credentials in OpenD.xml. Last known: ~$17,684 USD
- WealthSimple TFSA: ~$18,638 CAD — 3 active (GLD/XLE LEAPS + OWL put), 4 Apr puts expiring worthless ($1,495 lost)
- **War Room**: Composite 39.6 | REGIME: WATCH | 15 live API feeds (10/12 sources returning data)
- **13-Moon Doctrine**: Moon 1 (Pink Moon) — DEPLOY mandate. 184 events, 6 overlay layers, lead-time alerts active
- Matrix Monitor: 4 display modes, 20+ panels, 24/30 collectors OK
- Doctrine Engine: 12 packs, 4-state machine
- 8 strategies wired to unified integrator
- **Polymarket Division**: ACTIVE — $535.73 USDC, active_scanner.py
- Grand total: ~$46,220 USD | VIX 20.88 | CAD/USD 0.7231
- Pytest: **1928 passed**, 0 failed, 23 skipped, 1 xfailed
- Architecture Rework v3.3: **ALL PHASES COMPLETE (1-7)** — Strategy Advisor, NCL Relay, Doctrine Terrain, Monitor panels wired into orchestrator

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
- **13-Moon Doctrine** — temporal overlay layer:
  - Engine: `strategies/thirteen_moon_doctrine.py` (~1400 lines)
  - Storyboard: `strategies/thirteen_moon_storyboard.py` (~950 lines)
  - HTML: `data/storyboard/thirteen_moon_storyboard.html`
  - CLI: `python -m strategies.thirteen_moon_doctrine --upcoming 30`
  - 14 moon cycles (Mar 3, 2026 → Apr 19, 2027), 184 events, 6 overlay layers
  - Lead-time alerts: 14 event categories with escalating day-threshold actions
  - Integrated into Matrix Monitor sidebar

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
