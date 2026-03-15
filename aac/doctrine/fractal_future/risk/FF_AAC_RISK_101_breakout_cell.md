# Cell Entropy Budget — Breakout Cell (FF_AAC_RISK_101)

## Cell
- **Cell Name:** Breakout / Expansion Cell
- **Allocation Cap:** __%
- **Max Cell Drawdown:** __%
- **Max Loss Streak Before Throttle:** __
- **Max Allowed Entropy Regime:** E__ (typically E1–E3; throttle above)
- **Kill Switch Triggers:**
  - 2 consecutive failed breakouts in E3+
  - Liquidity shock proxy high
  - Correlation spike + failed continuation

## Valves
- Regime mismatch throttle: reduce size by __% or pause
- Cooldown after kill switch: __ sessions
- Require retest confirmation when entropy ≥ E3

## Recovery
- Re-run multi-scale validation checks for the signal variant in current regime
