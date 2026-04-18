# War Room — Stage Index

## What This Workspace Does

Orchestrates the War Room's twice-daily decision cycle: scan markets, evaluate regime,
plan position changes, execute trades (with human gate), and report results.

Each stage reads the previous stage's output. Human reviews between stages.

## Stages

| Stage | Directory | Job | Cadence |
|---|---|---|---|
| 01 | `stages/01-scan/` | Pull 11 live feeds, snapshot 15 indicators | Every 5 min (auto) |
| 02 | `stages/02-evaluate/` | Composite score, regime, milestone triggers | Every 5 min (auto) |
| 03 | `stages/03-plan/` | Scenario analysis, arm rebalance, roll checks | Twice daily / on demand |
| 04 | `stages/04-execute/` | Trade decisions — **HUMAN GATE** | On demand only |
| 05 | `stages/05-report/` | Mandate, P&L summary, storyboard refresh | Daily / on demand |

## Current State (April 2026)

- **Phase**: Accumulation ($45K → $150K target)
- **Regime**: WATCH (composite ~39.6) | VIX: 20.88
- **Thesis**: Stagflation (70%), Vol Shock 40/100
- **Active Arms**: Iran/Oil, BDC/NonAccrual, Crypto & Metals, TradFi Rotate
- **Accounts**: IBKR (live, 15 positions, net liq CAD $20,079), Moomoo (OpenD degraded), WealthSimple TFSA
- **IBKR Portfolio**: 5 calls ($12,143 MV) + 10 puts ($431 MV). Silver-heavy call book (SLV $66C x8, $70C x2, $75C x2 LEAPS). TSLA $500C LEAPS. XLE $65C x3.

## Config

Workspace-wide configuration lives in `_config/`:

| File | Contents |
|---|---|
| `_config/risk.md` | Roll discipline, position limits, drawdown caps |
| `_config/arms.md` | 5-arm allocation rules by phase |
| `_config/indicators.md` | 15 indicators, weights, regime thresholds |
| `_config/identity.md` | War Room voice, thesis, mandate format |

## Shared Resources

Cross-stage reference material in `shared/`:

| File | Contents |
|---|---|
| `shared/milestones.md` | 50-milestone spiderweb (canonical source) |
| `shared/scenarios.md` | Scenario definitions + MC overrides |
| `shared/accounts.md` | Account inventory + constraints |
