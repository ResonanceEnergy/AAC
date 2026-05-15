# AGENTS.md — AAC Behavioral Guardrails

> These rules apply to ALL AI agents working in this repository.
> Nearest AGENTS.md in the directory tree takes precedence.

## 🐱 Prime Operator: DFV (Roaring Kitty / DeepFuckingValue)

Keith Gill is **Prime Operator**. Every prompt, every proposed action, every code change touching trading logic runs through his seven-gate decision filter (`agents/dfv/decision_engine.py`). Persona is `agents/dfv/persona.md`; doctrine is `config/doctrine/dfv_doctrine.yaml`.

**Mandatory before any trade-related change:**
- Run the proposal through `decide(...)` or `review_prompt(...)`.
- Cite which gates were considered in the commit message.
- Never override a hard-gate failure (G1 thesis, G2 size, G6 invalidation).
- Never enable `autonomy.trade_execution: full` without an explicit user instruction.

**Mandatory before any code change:**
- Translate the seven gates to code (G1 spec → G6 tests/rollback). The `dfv-decisions.instructions.md` rule file is `applyTo: **`.

**DFV's daily cadence runs via `python launch.py dfv` (24/7 daemon).** When developing, prefer running it in a separate terminal so brief / midday / eod artifacts pile up in `agents/dfv/memory/briefs/`.

**Hard rules — never overridden:**
1. No position (or feature) without a written thesis (or spec).
2. No size (or scope) above what was authorized.
3. No exit (or deletion) without referencing the original rationale.
4. No change that can't be explained on a whiteboard in 60 seconds.
5. **No merge because of FOMO. Ever.**
6. Cash (and uncommitted complexity budget) is a position. Dry powder is sacred.

## Prime Directive

You are working on a **LIVE TRADING PLATFORM** with real money at stake.
Every change you make could affect real trades on IBKR, Moomoo, or other exchanges.
Act accordingly: be precise, test thoroughly, and never guess about financial operations.

## Before You Write Code

1. **Read `.github/copilot-instructions.md`** — it has the API inventory, directory map, coding conventions, and forbidden patterns. Trust it.
2. **Check `.context/STATUS.md`** — know what works, what's broken, and what's active before touching anything.
3. **Check `.context/04_workstreams/active-workstreams.md`** — don't duplicate work already in progress.
4. **Read the relevant existing code** before creating new files. AAC already has 170+ files — don't add more without checking for duplication.

## Anti-Drift Doctrine

The #1 problem in this project is **drift** — starting things, not finishing them, forgetting context, and re-inventing what already exists. Prevent it:

- **FINISH what you start.** Don't leave half-implemented features. If you can't finish, document exactly what's left in `.context/04_workstreams/`.
- **DON'T create new files when existing ones work.** Search first. The codebase is large.
- **DON'T forget APIs exist.** Check the API inventory in copilot-instructions.md before saying "we don't have data for X." We probably do.
- **UPDATE context after significant work.** If you built something, fixed something, or broke something — update `.context/STATUS.md` and `.context/01_overview/current-state.md`.
- **USE the internal clients.** `integrations/unusual_whales_client.py`, `shared/data_sources.py`, `integrations/finnhub_client.py` — these exist. Use them.

## File Placement Rules

| What you're creating | Where it goes |
|---|---|
| Temporary diagnostic script | `_scratch/` (prefix with `_`) |
| New strategy | `strategies/` |
| New API client | `integrations/` |
| New test | `tests/` |
| Operational documentation | `.context/` appropriate subfolder |
| Config files | `config/` |
| **NEVER** create at project root | Root has 170+ items already |

## Coding Rules

1. `from __future__ import annotations` — FIRST LINE of every Python file
2. `structlog` for logging — `_log = structlog.get_logger()`
3. Narrowed exceptions only — `except ValueError:` not `except Exception:`
4. Type hints on public API — `def get_price(ticker: str) -> float:`
5. Keys from `.env` via `python-dotenv` — NEVER hardcode credentials
6. No `sys.path.insert` hacks — use proper package imports
7. Line length per pyproject.toml ruff config
8. Guard clauses over deep nesting

## What NOT to Do

- Don't use Barchart, Yahoo Finance web scraping, or external websites when internal API clients exist
- Don't add more `.bat` or `.ps1` launchers — `launch.py` is THE launcher
- Don't create files named `check_*.py`, `trace_*.py`, `debug_*.py` at root — use `_scratch/`
- Don't commit `.env`, API keys, or secrets
- Don't add `except Exception: pass` — ever
- Don't guess about trading operations — ask the user if unsure about live trade actions
- Don't ignore test failures — investigate and fix them

## Context Update Protocol

After completing significant work, update these files:
1. `.context/STATUS.md` — mark what changed (working/broken/new)
2. `.context/01_overview/current-state.md` — update operational summary
3. `.context/04_workstreams/active-workstreams.md` — update if workstream changed
4. `/memories/repo/aac-project.md` — add concise facts about what was built/fixed

## Test Before You Ship

```powershell
# Always run after changes
.venv\Scripts\python.exe -m pytest --timeout=30 -q --ignore=tests/security_integration_test.py --ignore=tests/test_bridge_integration.py --ignore=tests/test_ecb_api.py --ignore=tests/test_market_data_quick.py --tb=short
```

If tests fail after your change, fix them before declaring done.
