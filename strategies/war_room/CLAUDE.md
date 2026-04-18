# War Room — MWP/ICM Workspace

> **Read this first. Every time.**

## Identity

You are operating inside the **AAC War Room** — a crisis-thesis trading system
that runs 100K-path Monte Carlo simulations, tracks 50 milestones in a spiderweb,
scores 15 indicators into a composite regime signal, and manages positions across
5 strategic arms on IBKR, Moomoo, and WealthSimple.

## Routing

1. Read `CONTEXT.md` in this directory for the stage index
2. Determine which stage you are in from the user's request
3. Navigate to `stages/<NN>-<name>/CONTEXT.md` for that stage's contract
4. Load only the references listed in that contract
5. Write outputs only to that stage's `output/` directory

## Rules

- **One stage, one job.** Scanning doesn't evaluate. Planning doesn't execute.
- **Plain text interface.** Markdown files only. No binary, no DB.
- **Every output is an edit surface.** Human can edit any output before the next stage.
- **One-way references.** Stage N reads Stage N-1 output. Never the reverse.
- **Canonical sources.** Config lives in `_config/`. Don't duplicate it in outputs.

## Runtime Code

The Python engine files live one level up (not in this workspace):

| File | Purpose |
|---|---|
| `../war_room_engine.py` | MC simulation, Greeks, milestones, indicators, 5-arm model |
| `../war_room_live_feeds.py` | 11 async API fetchers → LiveFeedResult |
| `../war_room_auto.py` | Auto-update (7 tasks) + auto-evolve (4 tasks) |

This MWP workspace governs the **decision workflow**. The Python code is the **execution runtime**.
Do not confuse the two. The workspace tells agents what to do. The code does it.

## Safety

This is a **LIVE TRADING PLATFORM**. Real money on IBKR, Moomoo, WealthSimple.
- Stage 04 (Execute) requires human confirmation before any trade action
- Never bypass DRY_RUN / kill switch / position limit checks
- When in doubt, stop and ask
