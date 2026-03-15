# Multi-Scale Validation Standard (v1.0)

## Why
Most strategies "work" only in one slice:
- one asset
- one timeframe
- one volatility band
- one market cycle

FRACTAL FUTURE standard prevents brittle wins.

## Required Tests (Minimum)

### A) Timeframe Coherence
- Evaluate at 3 timeframes (e.g., 15m / 1h / 1d) OR clearly define scope.

### B) Regime Robustness
- Evaluate in ≥ 3 regimes:
  - trend
  - range
  - vol expansion

### C) Parameter Stability
- Signal must not require hyper-tuned parameters
- Evaluate performance under parameter perturbation (±10–25%)

### D) Out-of-Sample Discipline
- Reserve out-of-sample periods
- Walk-forward if possible

## Pass/Fail Outcomes
- **PASS:** stable performance across tests
- **BOUNDED:** works only in specified regime/timeframe → label and gate risk
- **FAIL:** unstable / non-stationary without robust filter

## Artifact
Every strategy gets a Backtest Packet + Risk Packet.
