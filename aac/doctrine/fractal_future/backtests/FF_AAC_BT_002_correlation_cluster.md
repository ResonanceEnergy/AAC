# E3 Backtest — Correlation Cluster Score (FF_AAC_BT_002)

- **Strategy / Signal:** Correlation Cluster Score (risk posture / throttle)
- **Universe:** Mixed
- **Assets Basket:**
  - Equities: SPY, QQQ, TSLA, NVDA (starter)
  - Crypto: BTC, ETH, XRP (starter)
- **Period:** As available per asset
- **Purpose:** Measure whether applying throttles when correlation cluster is high reduces drawdowns.

## Multi-Scale Tests

### Timeframes Tested
- Daily primary; intraday optional

### Regimes Tested
- Normal (E2)
- Elevated/unstable (E3)
- Contagion/shock (E4)

### Parameter Sensitivity (±10–25%)
- Correlation window length
- Cluster threshold

### Out-of-Sample
- Holdout periods around known stress events (define them in your dataset)

## Results (Fill After Running)
- Baseline portfolio: no throttle
- Throttled portfolio: reduce gross exposure and cap correlated cells when cluster high
- Compare: drawdown, recovery time, volatility, win-rate stability
- PASS if drawdowns reduce materially without destroying returns
- Otherwise BOUNDED (only helps in E4) or FAIL
- Evidence label: target E3
