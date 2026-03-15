# Signal Research & Regime Detection (v1.0)

## Objective
Detect market regimes using multi-scale features so signals are evaluated in-context.

## Regime Types (Starter Set — Mixed Universe)

### 1) TREND / EXPANSION
- Directional structure: HH/HL (bull) or LH/LL (bear)
- Breakouts follow through more often than fail
- Best in: momentum continuation, breakout retests

### 2) RANGE / MEAN-REVERSION
- Oscillation within defined band; breakouts fail
- Mean reversion dominates at edges
- Best in: range scalps, vol-selling (bounded)

### 3) VOLATILITY EXPANSION (RISK-OFF / PANIC)
- Realized vol jumps; large candles; frequent gaps/wicks
- Correlation rises ("everything moves together")
- Best in: defensive hedges, short-duration mean reversion (tight risk only)

### 4) VOLATILITY COMPRESSION (CALM / COIL)
- Vol drops; ranges tighten; coiling structures
- Breakout probability rises but direction uncertain
- Best in: breakout-prep, straddle logic, level mapping

### 5) LIQUIDITY SHOCK / GAP RISK
- Discontinuous moves: gaps (equities) / liquidation cascades (crypto)
- Slippage rises; fills unreliable
- Best in: reduced sizing, longer confirmation, hedging

### 6) CORRELATION SPIKE / CONTAGION
- Cross-asset correlation rises; diversification fails
- Portfolio behaves like one bet
- Best in: hedged structures, reduced gross exposure

## Feature Families (Multi-Scale)
- Price structure: HH/HL vs range compression
- Volatility: realized vol, ATR bands, vol-of-vol
- Liquidity proxies: gaps, wick frequency, spreads
- Correlation: cross-asset correlation index / cluster similarity
- Flow proxies: options flow / volume anomalies (when available)
- Time-of-day / seasonality

## Regime Decision Rule
Use a "Regime Vote":
- Each feature family votes a regime state with confidence
- Final regime = weighted consensus
- Track regime stability and transitions

## Outputs
- Weekly Regime Map (primary)
- Daily Signal Brief (signals scored by regime)
- Monthly "regime drift" report
