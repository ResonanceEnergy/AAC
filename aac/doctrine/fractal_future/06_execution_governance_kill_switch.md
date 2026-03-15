# Execution Governance & Kill Switch (v1.0)

## Governance Gates
- **G1:** Definition clarity (signal is operational)
- **G2:** Evidence label ≥ E3 before paper trading, ≥ E4 before live size
- **G3:** Multi-scale validation passed or explicitly bounded
- **G4:** Risk packet complete (fail-local enforced)
- **G5:** Regime filter integrated

## Kill Switch Triggers (Starter)
- Daily loss exceeds threshold
- Drawdown exceeds threshold
- Regime mismatch detected + performance decay
- Volatility spike beyond model assumption
- Correlation spike across active cells

## Audit
Every trigger produces:
- Timestamp
- State snapshot
- Explanation notes
- Required remediation tasks
