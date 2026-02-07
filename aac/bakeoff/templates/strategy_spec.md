# Strategy Specification Template

**Strategy ID:** `STRAT_XXX`  
**Owner:**  
**Version:** 1.0.0  
**Date:** YYYY-MM-DD  
**Current Gate:** SPEC

---

## 1. Hypothesis

### Core Thesis
_What is the market inefficiency this strategy exploits?_

### Edge Source
_Why does this edge exist? Why hasn't it been arbitraged away?_

### Expected Alpha
_What is the expected return above benchmark?_

### Time Horizon
_Expected holding period and trade frequency_

---

## 2. Assumptions

### Market Structure
- [ ] _Assumption 1_
- [ ] _Assumption 2_

### Liquidity
- [ ] _Expected bid-ask spread_
- [ ] _Expected market depth_
- [ ] _Minimum volume requirements_

### Volatility Regime
- [ ] _Works in low vol: YES/NO_
- [ ] _Works in high vol: YES/NO_
- [ ] _Regime detection method:_

### Correlation
- [ ] _Expected correlation to BTC:_
- [ ] _Expected correlation to ETH:_
- [ ] _Hedging requirements:_

---

## 3. Failure Modes & Signatures

| ID | Failure Mode | Signature | Mitigation | Kill Threshold |
|----|--------------|-----------|------------|----------------|
| FM-001 | _Description_ | _How to detect_ | _Response_ | _When to halt_ |
| FM-002 | | | | |
| FM-003 | | | | |
| FM-004 | | | | |
| FM-005 | | | | |

---

## 4. Data Contracts

### Required Data Sources

| Source | Data Type | Freshness Req | Fallback |
|--------|-----------|---------------|----------|
| _Exchange_ | _Order book_ | _<100ms_ | _Use cached_ |
| | | | |

### Schema Definitions
```yaml
# Define expected schema for each data source
```

### Data Quality Requirements
- Freshness p95: _X ms_
- Null tolerance: _X%_
- Schema version: _X_

---

## 5. Execution Model

### Order Types Used
- [ ] Market orders
- [ ] Limit orders
- [ ] Stop-loss orders
- [ ] Other: _____

### Execution Assumptions
- Expected slippage (bps): _X_
- Expected fill rate: _X%_
- Max acceptable time-to-fill: _X ms_

### Position Sizing
- Method: _Fixed / Volatility-adjusted / Kelly_
- Base size: _$X_
- Max size: _$X_

---

## 6. Risk Envelope

### Position Limits
| Limit | Value | Hard/Soft |
|-------|-------|-----------|
| Max single position | $X | Hard |
| Max total exposure | $X | Hard |
| Max positions | X | Soft |

### Loss Limits
| Limit | Value | Action |
|-------|-------|--------|
| Max daily loss | X% | Halt trading |
| Max weekly loss | X% | Reduce size 50% |
| Max drawdown | X% | Kill switch |

### Concentration Limits
- Max single asset: X%
- Max correlated assets: X%

---

## 7. Metric Canon References

### Performance Metrics
- Primary: `reconciled_net_return`
- Secondary: `profit_factor`, `sharpe_ratio`

### Risk Metrics
- `max_drawdown_pct`
- `daily_loss_pct`
- `tail_loss_p95`

### Execution Metrics
- `slippage_bps_p50`
- `fill_rate`
- `time_to_fill_p95`

---

## 8. Kill Switch Plan

### Automatic Triggers
| Condition | Action |
|-----------|--------|
| daily_loss >= 3% | HALT all trading |
| drawdown >= 15% | HALT + close positions |
| recon_backlog >= 60m | HALT |

### Manual Kill Switch
1. Command: `python -m acc.kill_switch --strategy STRAT_XXX`
2. Dashboard: _Link to kill switch button_
3. Escalation: _Contact info_

### Position Unwinding
- Method: _Market / TWAP / VWAP_
- Max time to unwind: _X minutes_
- Slippage budget: _X bps_

---

## 9. Logging / Audit / Retention

### Logging Requirements
- [ ] All orders logged with timestamps
- [ ] All fills logged with execution details
- [ ] All signals logged with confidence scores
- [ ] All risk checks logged with pass/fail

### Audit Trail
- Retention period: _X days_
- Storage location: _____
- Access controls: _____

---

## 10. Gate Plan

### SIM Plan
- Backtest period: _Start - End_
- Walk-forward windows: _X_
- Success criteria: _Sharpe >= 1.5, DD < 15%_
- Expected duration: _2 weeks_

### PAPER Plan
- Start date: _____
- Success criteria: _Profit factor >= 1.2, no Sev1_
- Expected duration: _2 weeks_

### PILOT Plan
- Capital: _$1,000_
- Position size: _$200_
- Success criteria: _Sharpe >= 1.0, DD < 10%_
- Expected duration: _4 weeks_

---

## Approvals

| Gate | Reviewer | Date | Status |
|------|----------|------|--------|
| SPEC | | | ⬜ Pending |
| SIM | | | ⬜ Pending |
| PAPER | | | ⬜ Pending |
| PILOT | | | ⬜ Pending |
| POST_ANALYSIS | | | ⬜ Pending |
| SCALE | | | ⬜ Pending |
