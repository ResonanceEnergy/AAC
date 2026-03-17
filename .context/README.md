# AAC Context System

This directory is the durable project context layer for AAC.

Purpose:
- Keep architectural, operational, and session context in the repo.
- Reduce dependence on chat history and rate-limited research agents.
- Give future work a stable entry point.

Structure:
- `01_overview/` — current project state and orientation
- `02_architecture/` — system map and major subsystems
- `03_inventory/` — directory and component inventory
- `04_workstreams/` — active initiatives and ownership notes
- `05_decisions/` — engineering decisions and rationale
- `06_sessions/` — dated session summaries
- `07_gaps/` — gap scans and remediation status
- `08_runbooks/` — repeatable operational workflows
- `09_tests/` — canonical test commands and validation scope
- `10_memory/` — repo-local memory policy and conventions

Usage:
- Start work in `01_overview/current-state.md`.
- Check `07_gaps/` before starting gap filling.
- Record material changes in `05_decisions/` and `06_sessions/`.
- Keep files concise and current; this is a working index, not an archive dump.
