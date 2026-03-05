# SINGLE LAUNCHER RULE

> **One launcher to rule them all.**

## Rule

This repository uses **exactly one** launcher implementation: **`launch.py`** (project root).

| Allowed files | Purpose |
|---|---|
| `launch.py` | **THE launcher** — all modes, all platforms, single source of truth |
| `launch.bat` | Thin Windows CMD wrapper → calls `python launch.py %*` |
| `launch.sh` | Thin Unix/macOS wrapper → calls `python3 launch.py "$@"` |

No other launcher, startup, boot, or auto-run scripts may exist in the repository root.

## Prohibited

The following patterns are **banned** and must not be re-introduced:

- `*.bat` files that contain `python` commands (except `launch.bat`)
- `*.ps1` launcher scripts
- Any file named `*LAUNCH*`, `*launch*`, `*startup*`, `*boot*` (except the three above)
- Duplicate mode logic split across shell scripts and Python files
- Hardcoded absolute paths in any launch file

## Why

| Problem | Before | After |
|---|---|---|
| 5 files, 3 languages | `launch.ps1`, `launch.sh`, `AAC_AUTO_LAUNCH.bat`, `LFGCC_DASHBOARD!.bat`, `LFGCC!.bat` | `launch.py` + 2 thin wrappers |
| Modes scattered | Some modes in PS1, others in BAT, different subset in SH | All 8 modes in one place |
| Stale paths | Two BAT files had wrong directory paths | Single `PROJECT_ROOT = Path(__file__).parent` |
| No git-sync | Only in one BAT file | `launch.py git-sync` available everywhere |

## Adding a New Mode

1. Open `launch.py`
2. Add the mode name to `MODES` list
3. Add a description to `MODE_DESCRIPTIONS`
4. Write a `_mode_<name>()` handler function
5. Register it in `MODE_DISPATCH`
6. Done — works on Windows, macOS, and Linux immediately

## Available Modes

```
python launch.py dashboard   # Web monitoring dashboard
python launch.py monitor     # Terminal system monitor
python launch.py paper       # Paper trading engine
python launch.py core        # Core orchestrator
python launch.py full        # Full system launch
python launch.py test        # Pytest suite
python launch.py health      # Health check
python launch.py git-sync    # Git commit+push, then dashboard
```

## Enforcement

- **Code review**: Reject any PR that adds a new launcher file outside `launch.py`
- **CI check** (optional): Add a pre-commit hook or CI step:
  ```bash
  # Fail if any extra launch/boot files appear
  EXTRA=$(find . -maxdepth 1 \( -name '*launch*' -o -name '*LAUNCH*' -o -name '*startup*' \) \
          ! -name 'launch.py' ! -name 'launch.bat' ! -name 'launch.sh' | head -1)
  [ -n "$EXTRA" ] && echo "RULE VIOLATION: $EXTRA — see .github/SINGLE_LAUNCHER_RULE.md" && exit 1
  ```
