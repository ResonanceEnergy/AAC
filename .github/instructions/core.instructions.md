---
applyTo: "core/**,aac/**"
---

# Core & Doctrine — Orchestration Layer

This is the brain of AAC. Changes here ripple across every division.

## Key Modules

- `core/command_center.py` — top-level orchestration; agents subscribe here
- `core/aac_master_launcher.py` — referenced by `aac-launch` script entry
- `aac/` — doctrine packs, NCC integration, division wiring
- `aac/bakeoff/` — strategy bakeoff policy & metrics canon

## Rules

1. NEVER add a new launcher in `core/` — `launch.py` (root) is THE launcher (see `.github/SINGLE_LAUNCHER_RULE.md`)
2. Orchestration is async-first — use `asyncio.gather`, never thread pools, for fan-out
3. State must be observable — emit via `structlog` so the monitoring dashboards can pick it up
4. No silent failure — if a subsystem fails to start, raise; the launcher decides whether to continue
5. Doctrine packs (`config/doctrine_packs.yaml`, `aac/bakeoff/policy/*.yaml`) are config, not code — change YAML before changing logic
6. Cross-division calls go through `command_center` — divisions should not import each other directly

## Before Modifying

- Read `.context/02_architecture/system-map.md` to understand the subsystem graph
- If touching the launcher: also update `.github/SINGLE_LAUNCHER_RULE.md` if rules change
- If adding a new division: register it in `command_center` and add an entry to the directory map in `.github/copilot-instructions.md`
