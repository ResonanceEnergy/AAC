# PR Summary

<!-- One sentence: what changed and why. -->

## Scope

- [ ] Code change
- [ ] Test only
- [ ] Docs / `.context/` only
- [ ] CI / tooling
- [ ] Trading-path change (requires extra scrutiny — see below)

## Trading-Path Checklist (required if you touched `TradingExecution/`, `strategies/`, `core/`, or `trading/`)

- [ ] No change to `DRY_RUN`, `PAPER_TRADING`, or `LIVE_TRADING_ENABLED` defaults
- [ ] Existing safety checks (position limits, kill switches, order validation) preserved
- [ ] Tested with `DRY_RUN=true`
- [ ] No new `random.random()` in non-simulation paths
- [ ] Decision documented in `.context/05_decisions/` (if behavior change)

## Test Evidence

```
.venv\Scripts\python.exe -m pytest --timeout=30 -q --tb=short
```

<!-- Paste pass/fail summary -->

## Context Updates

- [ ] `.context/STATUS.md` updated (if user-visible behavior changed)
- [ ] `/memories/repo/` updated (if a new pattern or fix worth remembering)

## Forbidden Patterns Check

- [ ] No new `except Exception: pass`
- [ ] No new `sys.path.insert` hacks
- [ ] No new files at project root (use `_scratch/`, `scripts/`, or appropriate dir)
- [ ] No hardcoded API keys / `.env` not committed
- [ ] No use of Barchart / external scraping when internal API client exists
