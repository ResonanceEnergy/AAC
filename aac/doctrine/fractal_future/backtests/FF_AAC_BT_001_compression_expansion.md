# E3 Backtest — Compression→Expansion Breakout Trigger (FF_AAC_BT_001)

- **Strategy / Signal:** Compression-to-Expansion Breakout Trigger
- **Universe:** Mixed
- **Assets:**
  - Equities: SPY, QQQ, TSLA (starter)
  - Crypto: BTC, ETH (starter)
- **Period:** Last 5–10 years equities; last 5–8 years crypto (or available)
- **Execution Assumptions:**
  - Enter on breakout close or breakout+retest variant
  - Exit rules: fixed R multiple or trailing stop; document both
- **Fees/Slippage Assumptions:** (define per asset class)

## Multi-Scale Tests

### Timeframes Tested
- Equities: 15m / 1h / 1d
- Crypto: 1h / 4h / 1d

### Regimes Tested
- Vol compression → expansion transitions
- Trend continuation
- Range false breakout periods

### Parameter Sensitivity (±10–25%)
- ATR percentile thresholds
- Compression window length
- Breakout expansion threshold

### Out-of-Sample
- Walk-forward by year or rolling 6–12 month holdouts

## Results (Fill After Running)
- Key metrics:
- Failure modes:
- PASS / BOUNDED / FAIL:
- Evidence label: target E3
- Next action: forward test rules + risk packet enforcement
