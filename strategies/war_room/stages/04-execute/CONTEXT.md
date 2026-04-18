# Stage 04: Execute

> **HUMAN GATE.** Execute planned trades only after human confirmation.

## Inputs

| Source | File/Location | Section/Scope | Why |
|---|---|---|---|
| Action plan | `../03-plan/output/plan.md` | Full file | What to do |
| Roll alerts | `../03-plan/output/rolls.md` | Full file | Rolls to execute |
| Risk config | `../../_config/risk.md` | Kill Switch section | Safety checks |
| Account constraints | `../../_config/risk.md` | Account Constraints | Per-account limits |

## Process

1. Present the action plan to the human for review
2. **WAIT FOR EXPLICIT CONFIRMATION** before any trade
3. For each confirmed action:
   a. Validate against kill switch conditions
   b. Validate against position limits
   c. Check DRY_RUN flag (if true, log only)
   d. Route to correct connector:
      - IBKR: `TradingExecution/ibkr_connector.py`
      - Moomoo: `TradingExecution/moomoo_connector.py`
      - WealthSimple: Manual (no API, human places order)
   e. Log execution result
4. Write execution record to `output/executions.md`

## Outputs

| Artifact | Location | Format |
|---|---|---|
| Execution log | `output/executions.md` | Order ID, symbol, qty, price, status, timestamp |
| Rejections | `output/rejections.md` | Actions rejected by risk checks + reason |

## Safety Rules

These are **NON-NEGOTIABLE**:

1. **Never auto-execute.** Every trade requires human "yes"
2. **Never bypass DRY_RUN.** If DRY_RUN=true, log the order but don't send it
3. **Never exceed position limits.** Check `_config/risk.md` before every order
4. **Never ignore kill switch.** If composite > 90 or drawdown > 30%, halt
5. **WealthSimple has no API.** Flag WS actions as "manual execution required"
6. **Moomoo needs PIN.** Moomoo orders require trade PIN (069420)
7. **Check buying power.** IBKR has limited CAD buying power — verify before placing

## Rollback

If an execution fails or is placed in error:
- Log the error immediately in `output/executions.md`
- Do NOT retry automatically
- Flag for human review
