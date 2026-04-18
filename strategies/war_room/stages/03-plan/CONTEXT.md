# Stage 03: Plan

> Analyze scenarios. Propose arm rebalancing. Check roll discipline. Size positions.

## Inputs

| Source | File/Location | Section/Scope | Why |
|---|---|---|---|
| Latest eval | `../02-evaluate/output/latest-eval.md` | Regime + score | Current crisis level |
| Triggered milestones | `../02-evaluate/output/triggered.md` | Full file | Actions to take |
| Risk config | `../../_config/risk.md` | All sections | Position limits, roll rules |
| Arm config | `../../_config/arms.md` | Current phase section | Target allocations |
| Current positions | Runtime: `CURRENT_POSITIONS` | Full list | What we hold now |
| Scenarios | `../../shared/scenarios.md` | Full file | MC scenario definitions |

## Process

1. Read regime and triggered milestones from Stage 02
2. For each triggered milestone, look up its `strategy_action`
3. Check current arm allocations vs. targets (from `_config/arms.md`)
4. Run Monte Carlo if needed: `python strategies/war_room_engine.py --monte-carlo`
5. Apply roll discipline checks:
   - Any position within 21 DTE? → Flag for roll
   - Any position with $0 bid? → Flag as dead (do NOT roll)
   - Any position > 20 contracts? → Flag for reduction
   - OTM > 5% on short-dated? → Flag for adjustment
6. Check position limits against risk config
7. Generate ranked action list: what to buy, sell, roll, close
8. Write plan to `output/plan.md`

## Outputs

| Artifact | Location | Format |
|---|---|---|
| Action plan | `output/plan.md` | Ranked actions with sizing, rationale, priority |
| Roll alerts | `output/rolls.md` | Positions needing roll with recommended strikes/dates |
| Arm rebalance | `output/rebalance.md` | Current vs. target arm % with proposed trades |
| MC results | `output/monte-carlo.md` | Scenario probabilities, VaR/CVaR, percentile dist |

## Constraints

- No single action can exceed 15% of portfolio
- Max 5 evolution steps per day (from AutoEvolveParams)
- Scenario probability adjustments capped at 3% per day
- Arm shifts capped at 2% per day per arm
- All probabilities must sum to 1.0, all arm %s must sum to 100%

## This Stage Does NOT Execute

Planning proposes. Stage 04 executes. Human reviews between.
