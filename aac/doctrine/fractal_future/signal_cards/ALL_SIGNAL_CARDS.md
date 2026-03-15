# AAC Signal Cards — FRACTAL FUTURE (Mixed Universe)
## 10 Starter Signals (E0–E2 Research Grade)

---

## Signal 1 — MTF Trend Structure Score

- **Signal Name:** MTF Trend Structure Score
- **Hypothesis:** When trend structure aligns across timeframes, continuation probability increases.
- **Operational Definition:** Compute structure score on each timeframe (HH/HL vs LH/LL + slope proxy). Signal = weighted sum across 3 timeframes.
- **Feature Inputs:** Structure classification, slope proxy, breakout/failed-break rate
- **Timeframes:** Equities: 15m/1h/1d | Crypto: 1h/4h/1d
- **Regime Dependency:** Best in TREND/EXPANSION; degrades in RANGE.
- **Expected Behavior:** Higher score → higher continuation expectancy.
- **Failure Modes:** Whipsaw during correlation spike or liquidity shock.
- **Evidence Level:** E1 → candidate for E3 backtest
- **Notes:** Use as regime filter + sizing dial.

---

## Signal 2 — Compression-to-Expansion Breakout Trigger

- **Signal Name:** Compression-to-Expansion Breakout Trigger
- **Hypothesis:** Prolonged volatility compression increases probability of breakout expansion move.
- **Operational Definition:** Identify compression (ATR percentile low + range narrowing for N periods). Trigger when range expands above threshold with directional close.
- **Feature Inputs:** ATR percentile, range contraction, breakout range expansion
- **Timeframes:** 1h and 1d (or 4h and 1d crypto)
- **Regime Dependency:** VOL COMPRESSION transitioning to TREND or VOL EXPANSION.
- **Expected Behavior:** Expansion after compression yields asymmetric move.
- **Failure Modes:** False breakouts in RANGE regimes; gap risk.
- **Evidence Level:** E1–E2 (needs regime filter)
- **Notes:** Pair with retest confirmation for equities; wick filter for crypto.

---

## Signal 3 — Range Edge Mean Reversion

- **Signal Name:** Range Edge Mean Reversion
- **Hypothesis:** In stable ranges, edges provide positive expectancy mean reversion entries.
- **Operational Definition:** Define range using recent swing highs/lows or volatility bands. Entry when price tags edge + rejection signature (close back inside).
- **Feature Inputs:** Range bounds, rejection candle pattern, midline target
- **Timeframes:** 15m/1h with 1d context
- **Regime Dependency:** RANGE/MEAN-REVERSION only. Must be throttled in TREND/VOL EXPANSION.
- **Expected Behavior:** Small repeated gains with tight risk.
- **Failure Modes:** Trend emergence (range breaks and runs).
- **Evidence Level:** E1
- **Notes:** Requires strict fail-local stops + regime kill switch.

---

## Signal 4 — Risk-Off Vol Expansion Trigger

- **Signal Name:** Risk-Off Vol Expansion Trigger
- **Hypothesis:** A volatility regime shift predicts drawdown risk and changes optimal strategy set.
- **Operational Definition:** Trigger when realized vol / ATR rises above percentile threshold AND correlation increases.
- **Feature Inputs:** Realized vol, ATR, correlation cluster score
- **Timeframes:** 1h + 1d (4h + 1d crypto)
- **Regime Dependency:** VOL EXPANSION / CONTAGION
- **Expected Behavior:** Signals "reduce risk / hedge / switch cells."
- **Failure Modes:** Over-triggering in noisy periods; delayed detection.
- **Evidence Level:** E1–E2
- **Notes:** Primarily a *portfolio posture* signal, not an entry signal.

---

## Signal 5 — Correlation Cluster Score

- **Signal Name:** Correlation Cluster Score
- **Hypothesis:** When correlations spike, portfolio behaves like one trade and risk must be reduced.
- **Operational Definition:** Compute rolling correlations across basket (indices/mega-caps/crypto majors). Cluster score = mean corr + dispersion metric.
- **Feature Inputs:** Rolling corr matrix, cluster measure
- **Timeframes:** Daily primary; intraday optional
- **Regime Dependency:** CORRELATION SPIKE / CONTAGION
- **Expected Behavior:** Triggers exposure caps and hedging posture.
- **Failure Modes:** False positives in slow markets; regime lag.
- **Evidence Level:** E2 candidate
- **Notes:** Connect directly to Fail-Local Risk OS.

---

## Signal 6 — Liquidity Shock Proxy

- **Signal Name:** Liquidity Shock Proxy
- **Hypothesis:** Abnormal wick/gap frequency predicts poor fills + elevated stop-out risk.
- **Operational Definition:** Wickiness ratio (wick length / body) + gap frequency (equities) over rolling window. Trigger when above baseline threshold.
- **Feature Inputs:** Candle wick stats, gap stats, range stats
- **Timeframes:** 15m/1h + daily context
- **Regime Dependency:** LIQUIDITY SHOCK
- **Expected Behavior:** Throttles tight-stop strategies and reduces size.
- **Failure Modes:** Over-throttling during healthy volatility.
- **Evidence Level:** E1–E2
- **Notes:** Critical for crypto liquidation wicks.

---

## Signal 7 — Breakout Retest Confirmation

- **Signal Name:** Breakout Retest Confirmation
- **Hypothesis:** Breakouts that retest and hold are more reliable than impulse-only breakouts.
- **Operational Definition:** Identify breakout above key level, then require retest within M bars holding above.
- **Feature Inputs:** Level detection, retest behavior, close location
- **Timeframes:** 15m/1h (equities) and 1h/4h (crypto)
- **Regime Dependency:** TREND/EXPANSION and post-COMPRESSION transitions
- **Expected Behavior:** Higher win rate; slightly fewer trades.
- **Failure Modes:** V-shaped continuation without retest (missed moves).
- **Evidence Level:** E1
- **Notes:** Combine with MTF Trend Structure Score for sizing.

---

## Signal 8 — Momentum Exhaustion Reversal Candidate

- **Signal Name:** Momentum Exhaustion Reversal Candidate
- **Hypothesis:** After extreme expansion, probability of mean reversion increases (bounded).
- **Operational Definition:** Identify extended move + volatility spike + failure to continue (e.g., lower high after blow-off).
- **Feature Inputs:** Extension measure, vol spike, continuation failure
- **Timeframes:** 1h/4h with daily context
- **Regime Dependency:** VOL EXPANSION transitions and end-of-trend events
- **Expected Behavior:** High reward but lower hit rate; requires strict risk.
- **Failure Modes:** "Shorting strength" in true trend continuation (gets run over).
- **Evidence Level:** E0–E1 (must be bounded + heavily gated)
- **Notes:** Only valid with strong regime confirmation + kill switch.

---

## Signal 9 — Flow/Volume Anomaly Trigger

- **Signal Name:** Flow/Volume Anomaly Trigger (Options/Volume)
- **Hypothesis:** Abnormal flow/volume indicates information arrival or positioning shift.
- **Operational Definition:** Trigger when flow/volume z-score exceeds threshold and aligns with regime filter.
- **Feature Inputs:** Options flow anomalies (if available), volume spikes, relative volume
- **Timeframes:** Intraday + daily confirmation
- **Regime Dependency:** Strongest during COMPRESSION→EXPANSION and TREND
- **Expected Behavior:** Helps pick direction for breakout moves.
- **Failure Modes:** Noise spikes; misinterpreted hedging flows.
- **Evidence Level:** E1–E2 (data dependent)
- **Notes:** Plug into TSLA/SPY/QQQ + high beta names.

---

## Signal 10 — Regime-Filtered Strategy Selector (Meta-Signal)

- **Signal Name:** Regime-Filtered Strategy Selector (Meta)
- **Hypothesis:** The best "signal" is choosing the correct strategy cell for the current regime.
- **Operational Definition:**
  - If regime = TREND → enable trend/breakout cells
  - If regime = RANGE → enable mean reversion cells
  - If regime = VOL EXPANSION/CONTAGION → reduce exposure + hedge + shorten duration
  - If regime = LIQ SHOCK → reduce size + avoid tight stops
- **Feature Inputs:** Regime classifier outputs, confidence, transition risk
- **Timeframes:** Daily posture; intraday execution
- **Regime Dependency:** All regimes (it *is* the regime layer)
- **Expected Behavior:** Raises portfolio-level expectancy and reduces blowups.
- **Failure Modes:** Misclassification; regime lag.
- **Evidence Level:** E1 → target E3 through backtests comparing with/without selector.
- **Notes:** This is the AAC expression of FRACTAL FUTURE: coherence across scale + fail-local.
