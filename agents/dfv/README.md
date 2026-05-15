# DFV — Roaring Kitty agent

> **Codename:** DFV  ·  **Operator:** Keith Patrick Gill (Roaring Kitty / DeepFuckingValue)
> **Role:** Prime Operator on AAC. Every prompt and every proposed action is filtered through this lens.

## What it is

A concentrated, opinionated, deep-value operator persona implemented as:

1. **Persona** — `persona.md` defines voice, methodology, the seven decision gates, hard rules.
2. **Doctrine** — `config/doctrine/dfv_doctrine.yaml` holds the thresholds, autonomy switches, conviction tiers, schedules.
3. **Decision engine** — `decision_engine.py` runs every proposal through the seven gates and returns an explicit verdict.
4. **Memory** — `memory_store.py` + `memory/*.json` track theses, convictions, watchlist, every decision.
5. **Routines** — `routines.py` produce pre-market brief, midday, EOD, weekend DD reports.
6. **Daemon** — `daemon.py` schedules routines around the clock per the doctrine cadence.
7. **CLI** — `python -m agents.dfv <cmd>` for one-shot use.

## Quickstart

```powershell
cd C:\dev\AAC_fresh
.venv\Scripts\activate

# One-shot pre-market brief
.venv\Scripts\python.exe -m agents.dfv brief

# Run the seven-gate review on a prompt
.venv\Scripts\python.exe -m agents.dfv review "Should I YOLO into NVDA calls?"

# Status snapshot
.venv\Scripts\python.exe -m agents.dfv status

# Add a thesis
.venv\Scripts\python.exe -m agents.dfv thesis set GME `
    --thesis "Console refresh + tariff tailwind, $4.6B cash, no debt, brand still owns gaming retail." `
    --conviction 5 --horizon "12-24 months" `
    --catalysts "Holiday season|Q4 earnings|Buyback announcement" `
    --invalidation "Cash burn > $200M/qtr OR loss of net cash position" `
    --target "$45 base / $80 bull (DCF + comps)" --max-pct 0.20

# Run the 24/7 daemon
.venv\Scripts\python.exe -m agents.dfv daemon
# or
.venv\Scripts\python.exe launch.py dfv
```

## The seven gates

Every proposal goes through these. A doctrine-defined verdict is returned.

| Gate | Question | Severity |
|---|---|---|
| G1 | Is there a written thesis on this ticker? | hard |
| G2 | Is the proposed size ≤ the conviction-tier cap? | hard |
| G3 | Does this leave ≥ 10% dry powder? | soft |
| G4 | Is there a catalyst in the next 5 days that's been acknowledged? | soft |
| G5 | Does this push factor concentration > 40%? | soft |
| G6 | Is the invalidation level defined? | hard |
| G7 | Expected slippage < 1.0%? | soft |

- 1+ hard fail → **vetoed**
- 3+ soft fails → **vetoed**
- 2 soft fails → **returned**
- 1 soft fail → **approved with notes**
- 0 fails → **approved**

## Autonomy

Defined in `config/doctrine/dfv_doctrine.yaml::autonomy`.

| Domain | Default |
|---|---|
| Research, screening, briefings, alerts, memory | `full` |
| Watchlist add/remove | `full` |
| Conviction tier nudges | `full` |
| **Order placement** | `human_in_loop` ← DFV proposes; human signs |
| Margin / leverage changes | `human_in_loop` |
| New ticker entering active book | `human_in_loop` |

To enable full trade autonomy, change `trade_execution: human_in_loop` to `full`.
**Do this with intent.** AAC is wired to live brokers (IBKR U24346218, Moomoo FUTUCA, NDAX).

## VS Code / Copilot integration

- `.github/chatmodes/dfv.chatmode.md` — puts Copilot in DFV voice.
- `.github/instructions/dfv-decisions.instructions.md` — applyTo: `**` so every change is reviewed.
- `.github/copilot-instructions.md` — section "DFV is Prime Operator".
- `AGENTS.md` — DFV anti-drift directives.
