# Stage 05: Report

> Generate daily mandate, P&L summary, and storyboard data.

## Inputs

| Source | File/Location | Section/Scope | Why |
|---|---|---|---|
| Latest eval | `../02-evaluate/output/latest-eval.md` | Regime + score | Current state |
| Execution log | `../04-execute/output/executions.md` | Full file | What was traded today |
| Identity config | `../../_config/identity.md` | Mandate Format | Report structure |
| Risk config | `../../_config/risk.md` | Full file | Safety status |
| Current positions | Runtime: `CURRENT_POSITIONS` | Full list | Portfolio snapshot |
| Account balances | Runtime: `ACCOUNTS` | Full dict | Account totals |

## Process

1. Compute portfolio value from account balances
2. Calculate P&L:
   - Per-position: entry price vs current price
   - Per-arm: sum of position P&L by arm type
   - Per-account: sum by broker
   - Total: USD equivalent of all accounts
3. Generate mandate JSON (format in `_config/identity.md`)
4. Identify milestones approaching (within 10% of threshold)
5. Flag positions needing attention (expiring within 21 DTE, large drawdown)
6. Write all reports to `output/`
7. Persist mandate to `data/war_engine/mandate_YYYY-MM-DD.json`
8. Regenerate storyboard HTML if storyboard_regen interval has passed

## Outputs

| Artifact | Location | Format |
|---|---|---|
| Daily mandate | `output/mandate.md` | Human-readable mandate summary |
| Mandate JSON | `output/mandate.json` | Machine-readable mandate (see identity.md) |
| P&L summary | `output/pnl.md` | Per-position, per-arm, per-account, total |
| Portfolio snapshot | `output/portfolio.md` | All positions with Greeks, arm, account |
| Risk status | `output/risk-status.md` | Kill switch state, drawdown, daily loss |

## Runtime

- **Auto**: `WarRoomAutoEngine` runs `mandate_gen` every 24 hours
- **Auto**: `storyboard_regen` every 1 hour
- **Manual**: `python strategies/war_room_engine.py --mandate`
- **Manual**: `python strategies/war_room_engine.py --positions`
- **Manual**: `python strategies/war_room_engine.py --arms`

## CLI Quick Reference

```
python strategies/war_room_engine.py                  # full dashboard
python strategies/war_room_engine.py --monte-carlo    # 100K MC simulation
python strategies/war_room_engine.py --milestones     # spiderweb status
python strategies/war_room_engine.py --greeks         # position Greeks
python strategies/war_room_engine.py --mandate        # daily mandate
python strategies/war_room_engine.py --positions      # all positions
python strategies/war_room_engine.py --arms           # 5-arm breakdown
python strategies/war_room_engine.py --indicators     # 15-indicator model
python strategies/war_room_engine.py --scenario NAME  # run scenario
python strategies/war_room_engine.py --phase          # phase status
python strategies/war_room_engine.py --json           # JSON output
```
