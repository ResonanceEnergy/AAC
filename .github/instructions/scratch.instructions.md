---
applyTo: "_scratch/**"
---

# `_scratch/` — Disposable Diagnostic Scripts

Files here are throwaway. Most rules are relaxed.

## Allowed

- One-off scripts to investigate bugs, probe APIs, validate hypotheses
- Hardcoded values, no error handling, no tests
- Files prefixed with `_` (e.g. `_check_uw_fields.py`)
- Quick PowerShell helpers (`.ps1`)

## Still Forbidden

- Real API keys / secrets in source (use `.env` even here)
- Live trade execution (use `DRY_RUN=true` always; if you need to test live, use `tests/` with proper mocks)
- `sys.path.insert` hacks — use `python -m _scratch.script_name` from repo root instead
- Anything that writes to `data/`, `logs/`, or repo state without obvious cleanup

## Lifecycle

- If a `_scratch/` script proves useful → promote it to `scripts/` or `tools/` (rename, add tests, remove `_` prefix)
- If unused for 30+ days → safe to delete
- If it documented a finding → move the finding to `.context/` and delete the script
