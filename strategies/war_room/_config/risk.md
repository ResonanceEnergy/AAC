# Risk Configuration

> Canonical source for all position limits, drawdown caps, and roll discipline.
> Referenced by: stages/03-plan, stages/04-execute

## Roll Discipline (Post-Mortem: Apr 6 2026)

All Apr 17 puts expired worthless ($0 bid at 11 DTE). 65-contract OBDC position
had $0 recovery. These rules are hard-coded from that lesson.

| Rule | Value | Rationale |
|---|---|---|
| Max contracts per position | 20 | OBDC x65 was untradeable |
| Roll trigger DTE | 21 | Theta kills value below 21 DTE |
| Max OTM % (short-dated) | 5% | Puts <=3 months must be near the money |
| Dead put gate | ON | If STC bid = $0, do NOT roll. Re-evaluate thesis |
| LEAPS vs Puts allocation | 70/30 | 70% LEAPS, 30% directional puts |

## Position Limits

| Limit | Value |
|---|---|
| Max drawdown (account-level) | 25% |
| Daily loss limit | 8% |
| Max open positions | 30 |
| Max single position % | 15% of portfolio |
| Max strategy allocation % | 40% of portfolio |
| Trailing stop (crypto arm) | 10% |

## Account Constraints

| Account | Platform | Constraint |
|---|---|---|
| IBKR U24346218 | IBKR | CAD-denominated, limited buying power |
| Moomoo FUTUCA | Moomoo | USD, options approval pending, PIN=069420 |
| WealthSimple TFSA | WealthSimple | Tax-sheltered, no margin, LEAPS preferred |

## Kill Switch

- Composite score > 90 → halt all new entries, protect capital
- Any account drawdown > 30% → close 50% of positions immediately
- Stablecoin depeg > 2% → close all crypto positions
- If in doubt: **do nothing**. No trade is better than a bad trade.
