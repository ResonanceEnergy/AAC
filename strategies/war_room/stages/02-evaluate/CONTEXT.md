# Stage 02: Evaluate

> Compute composite crisis score. Classify regime. Check milestone triggers.

## Inputs

| Source | File/Location | Section/Scope | Why |
|---|---|---|---|
| Latest scan | `../01-scan/output/latest-scan.md` | Full file | Current indicator values |
| Indicator config | `../../_config/indicators.md` | Weights + Thresholds | Scoring rules |
| Milestones | `../../shared/milestones.md` | Full file | 50-gate trigger definitions |
| Previous eval | `output/latest-eval.md` | Regime + score | Hysteresis comparison |

## Process

1. Read the 15 indicator values from Stage 01 output
2. Compute individual indicator scores (0-100 each) using piecewise linear functions
3. Apply weights (sum = 1.00) to get composite crisis score (0-100)
4. Classify regime: CALM (<=30), WATCH (30-50), ELEVATED (50-70), CRISIS (>70)
5. Apply hysteresis: regime only flips if score crosses threshold by 3+ pts AND holds 30+ min
6. Check all 50 milestones against current state → list newly triggered
7. Trace spiderweb chains from any newly triggered milestone (leads_to graph)
8. Write evaluation to `output/latest-eval.md`

## Outputs

| Artifact | Location | Format |
|---|---|---|
| Evaluation | `output/latest-eval.md` | Composite score, regime, individual scores, delta from previous |
| Triggered milestones | `output/triggered.md` | List of newly triggered milestones + their strategy actions |
| Approaching milestones | `output/approaching.md` | Milestones within 10% of trigger threshold |

## Regime Actions

| Regime | What Changes |
|---|---|
| CALM → WATCH | Tighten stops, review positions weekly instead of daily |
| WATCH → ELEVATED | Full crisis positioning, all 5 arms active, daily review |
| ELEVATED → CRISIS | Max vega on puts, gamma scalping enabled, take profit above score 85 |
| Any → CALM | Reduce options 50%, rotate to income/treasury |

## Runtime

- **Auto**: `WarRoomAutoEngine` runs `composite_trend` every 5 minutes
- **Auto**: `milestone_check` every 5 minutes
- **Manual**: `python strategies/war_room_engine.py --indicators`
