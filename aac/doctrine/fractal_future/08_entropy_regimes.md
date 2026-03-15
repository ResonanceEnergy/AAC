# AAC — Entropy Regimes (v1.0)

## Market Entropy (Operational)
In AAC, entropy manifests as:
- **Volatility** (energy dispersion)
- **Correlation coupling** (diversification failure)
- **Liquidity fragility** (gaps, slippage)
- **Regime drift** (non-stationarity)
- **Tail risk** (rare shocks)

## Entropy Regime Scale
| Level | Name | Description |
|-------|------|-------------|
| E1 | Calm / Compression | Low vol, tight ranges, stable conditions |
| E2 | Normal / Structured | Moderate vol, clear structure, tradeable |
| E3 | Elevated / Unstable | Rising vol, unclear direction, wicks/gaps increase |
| E4 | Shock / Contagion | High vol, correlation spike, liquidation risk |
| E5 | Disorder / Liquidation | Extreme vol, market-wide stress, fills unreliable |

## Regime Detection Inputs (Starter)
- Realized vol / ATR percentile
- Vol-of-vol
- Correlation cluster score
- Wick/gap frequency (liquidity proxy)
- Trend structure stability

## Rule
> Strategy selection and sizing must be entropy-regime aware.
> If entropy regime rises, leverage and exposure must fall automatically.
