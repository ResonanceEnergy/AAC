# 5-Arm Allocation Rules

> Canonical source for arm targets by phase.
> Referenced by: stages/03-plan, stages/04-execute

## Accumulation Phase ($0 — $150K) ← CURRENT

| Arm | Code | Target % | Max % | Instruments | Entry Condition | Exit Condition |
|---|---|---|---|---|---|---|
| Iran/Oil Crisis | `IRAN_OIL` | 30% | 40% | SPY puts, QQQ puts, XLE calls, USO calls | Oil >$100, VIX >22, Hormuz tension | Oil <$90 or VIX <18 |
| Private Credit/BDC | `BDC_NONACCRUAL` | 25% | 35% | FSK puts, TCPC puts, HYG puts, BX puts | HY spread >400bp, NAV discount >10% | HY spread <300bp |
| Crypto & Metals | `CRYPTO_METALS` | 20% | 30% | GLD calls, SLV calls, BITO puts, GDX calls | Gold >$4500, BTC <$70K | Gold <$4000, BTC >$90K |
| DeFi Yield Farm | `DEFI_YIELD` | 15% | 25% | AAVE puts, UNI puts, DeFi shorts | DeFi TVL drop >15%, yield compression | DeFi TVL recovery |
| TradFi Rotate | `TRADFI_ROTATE` | 10% | 20% | Treasury ETFs, Dividend ETFs, Income plays | Composite <40, recovery signals | Crisis re-escalation |

## Growth Phase ($150K — $1M)

| Arm | Target % | Max % | Shift |
|---|---|---|---|
| Iran/Oil (Reduced) | 20% | 25% | Confirmed escalation only |
| Private Credit | 15% | 20% | Non-accrual spike only |
| Metals (Dominant) | 30% | 40% | Gold/silver trend dominant |
| DeFi (Reduced) | 10% | 15% | Major stress only |
| TradFi Income | 25% | 35% | Yield opportunity focus |

## Rotation Phase ($1M — $5M)

| Arm | Target % | Max % | Shift |
|---|---|---|---|
| Oil (Minimal) | 5% | 10% | Black swan only |
| Credit (Minimal) | 5% | 10% | Systemic risk only |
| Metals Preservation | 25% | 30% | Inflation hedge |
| DeFi (Zero) | 0% | 5% | Never in preservation |
| TradFi Income (Core) | 65% | 80% | Always active |

## Phase Transitions

| Gate | Threshold | Action |
|---|---|---|
| Accumulation → Growth | $150K USD | Close 30% puts, rotate to gold/miners, 10% treasury |
| Growth → Rotation | $1M USD | Income mode, max 15% options, 40% fixed income |
| Rotation → Preservation | $5M USD | Max 5% options, 60% yield, endowment mode |
