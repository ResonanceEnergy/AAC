---
applyTo: "docs/**,.context/**"
---

# Documentation & Context

Two distinct doc systems — don't confuse them.

## `.context/` — Durable Project Context (10-folder system)

Lives at repo root. Agents read this to orient. Update after significant work.

| Folder | Purpose |
|---|---|
| `01_overview/current-state.md` | START HERE for orientation |
| `02_architecture/system-map.md` | Subsystem relationships |
| `03_inventory/` | Directory map, external data sources |
| `04_workstreams/active-workstreams.md` | What's being worked on |
| `05_decisions/` | Engineering decisions with rationale |
| `07_gaps/` | Gap scans and remediation status |
| `08_runbooks/` | Repeatable workflows |
| `STATUS.md` | Living status dashboard (working/broken/active) |

## `docs/` — User-Facing Documentation

API references, setup guides, user tutorials. Stable, polished prose.

## Rules

1. `.context/` is for AAC operators and AI agents — terse, factual, dated
2. `docs/` is for users (including future you) — prose, examples, screenshots OK
3. NEVER duplicate content between them — link instead
4. After completing significant work, update `.context/STATUS.md` AND `.context/01_overview/current-state.md`
5. Decisions that change behavior → ADR-style entry in `.context/05_decisions/` (date, context, decision, rationale, consequences)
6. Don't create `*_FINAL.md`, `*_v2.md`, `*_NEW.md` — edit the canonical file in place
7. Markdown files use `.md` not `.MD`; lowercase filenames except for `README.md`, `AGENTS.md`, `CHANGELOG.md`
