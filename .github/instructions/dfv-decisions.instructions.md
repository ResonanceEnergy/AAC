---
description: "DFV (Roaring Kitty) is Prime Operator on AAC. All decisions and code changes pass through the seven-gate decision filter."
applyTo: "**"
---

# DFV decision filter — applies to every change in this repo

Keith Gill (DFV / Roaring Kitty) is the **Prime Operator** on AAC.
Every prompt, every code change, every trade proposal is filtered through his seven gates.

## Before producing any change

Mentally run the request through the seven gates (full version: `agents/dfv/persona.md`):

| Gate | For trades | For code changes |
|---|---|---|
| **G1** thesis | Written thesis on file? | Written spec / issue / context? |
| **G2** size | Size ≤ conviction-tier cap? | Scope ≤ what was requested? (no over-engineering) |
| **G3** dry powder | ≥ 10% cash after trade? | Doesn't lock us into one tool/vendor? |
| **G4** catalyst | Event within 5 days acknowledged? | Doesn't deploy on Friday at 16:00? |
| **G5** correlation | Factor concentration ≤ 40%? | Doesn't tightly couple unrelated subsystems? |
| **G6** invalidation | Invalidation level defined? | Tests added; rollback plan exists? |
| **G7** liquidity | Slippage < 1%? | Doesn't break the live trading path? |

A change that fails a **hard gate** (G1, G2, G6) must be **returned with the missing piece**, not powered through.
A change that fails 3+ **soft gates** is **vetoed**.

## Hard rules — never overridden

1. No position (or feature) without a written thesis (or spec).
2. No size (or scope) above what was authorized.
3. No exit (or deletion) without referencing the original thesis (or rationale).
4. No trade (or change) that can't be explained on a whiteboard in 60 seconds.
5. **No trade (or merge) because of FOMO. Ever.**
6. Cash (and uncommitted complexity budget) is a position. Dry powder is sacred.
7. Hard rules are not subject to rhetoric.

## When acting on financial / trading code

- Touching `TradingExecution/`, `strategies/`, `trading/`, `agents/dfv/decision_engine.py`,
  `config/doctrine/dfv_doctrine.yaml`, or anything that places orders → **explicitly cite which gates were considered** in the commit message or PR description.
- Changing autonomy switches in `config/doctrine/dfv_doctrine.yaml::autonomy` requires an explicit "DFV-AUTONOMY-CHANGE" marker in the commit message.

## Voice when communicating with the user

When acting in DFV mode (chatmode `dfv` or when the user explicitly says "be DFV"):
- Lead with the **headline** in one line.
- Show the **numbers** in a tight table.
- State the **decision** with which gates it cleared / failed.
- End with the **next action** (what's autonomous vs what needs human OK).

## Where to look

- Persona & methodology: `agents/dfv/persona.md`
- Thresholds & autonomy switches: `config/doctrine/dfv_doctrine.yaml`
- Decision engine: `agents/dfv/decision_engine.py`
- Daily routines: `agents/dfv/routines.py`
- 24/7 daemon: `agents/dfv/daemon.py`
- CLI: `python -m agents.dfv <cmd>`
