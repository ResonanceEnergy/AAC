# Fail-Local Risk OS (v1.0)

## Doctrine
No single strategy is allowed to threaten portfolio survivability.
Failure must be local, informative, and containable.

## Risk Packets (Mandatory Per Strategy Cell)
- Max position size
- Max daily loss
- Max weekly loss
- Max drawdown before throttle
- Exposure caps by asset/class
- Volatility-based sizing rule
- Kill switch conditions

## Blast Radius Containment
- **Strategy isolation:** each cell has capital allocation cap
- **Correlation caps:** reduce simultaneous exposure to correlated bets
- **Regime mismatch throttle:** if regime changes, reduce size or pause
- **Circuit breakers:** automatic downshift after loss streaks

## Recovery Protocol
After a kill switch triggers:
1) Pause
2) Diagnose (regime mismatch? execution? signal drift?)
3) Re-validate (multi-scale standard)
4) Only then re-enable
