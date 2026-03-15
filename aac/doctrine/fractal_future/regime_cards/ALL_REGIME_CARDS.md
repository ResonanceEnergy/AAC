# AAC Regime Card — TREND / EXPANSION

- **Regime Name:** TREND / EXPANSION
- **Definition (operational):**
  - Directional structure dominates: HH/HL (bull) or LH/LL (bear)
  - Breakouts follow through more often than they fail
  - Pullbacks are bought/sold with continuation
- **Feature Votes:**
  - Structure: strong trend structure present on 2+ timeframes
  - Volatility: moderate or rising, but not chaotic
  - Correlation: moderate; leaders emerge
  - Liquidity proxies: stable gaps; controlled wicks
- **Confidence:**
  - High when trend is consistent across 15m/1h/1d (or 1h/4h/1d for crypto)
- **Transition Indicators:**
  - Compression after trend (range building), divergence, or vol spike + failed continuation
- **Strategy Cells Favored:**
  - Trend cell, momentum continuation, breakout retests
- **Strategy Cells Throttled:**
  - Pure mean reversion fades

---

# AAC Regime Card — RANGE / MEAN-REVERSION

- **Regime Name:** RANGE / MEAN-REVERSION
- **Definition (operational):**
  - Price oscillates within a defined band; breakouts fail more often than follow through
  - Mean reversion dominates at edges; midline acts as gravity
- **Feature Votes:**
  - Structure: repeated rejections at range boundaries
  - Volatility: stable to compressing
  - Correlation: mixed
  - Liquidity proxies: wickier than trend; fewer sustained runs
- **Confidence:**
  - High when range boundaries persist across 1h and 1d (or 4h and 1d in crypto)
- **Transition Indicators:**
  - Volatility expansion, clean breakout + retest, correlation spike
- **Strategy Cells Favored:**
  - Mean reversion cell, range scalps, vol-selling structures (bounded)
- **Strategy Cells Throttled:**
  - Trend continuation breakouts

---

# AAC Regime Card — VOLATILITY EXPANSION (RISK-OFF / PANIC)

- **Regime Name:** VOLATILITY EXPANSION (RISK-OFF / PANIC)
- **Definition (operational):**
  - Realized volatility and intrabar range jump materially above baseline
  - Large candles, frequent gaps (equities) / impulsive liquidation moves (crypto)
  - Correlation rises sharply ("everything moves together")
- **Feature Votes:**
  - Volatility: high and rising
  - Correlation: high / clustering
  - Liquidity proxies: gappy, large wicks, slippage risk
  - Structure: breakdowns accelerate; support fails quickly
- **Confidence:**
  - High when vol > threshold for 2–3 consecutive sessions and correlation spikes
- **Transition Indicators:**
  - Stabilization + compression, exhaustion reversal, vol-of-vol decline
- **Strategy Cells Favored:**
  - Volatility cell, defensive hedges, short-duration mean reversion ONLY with tight fail-local risk
- **Strategy Cells Throttled:**
  - Leverage-heavy trend chasing, martingale mean reversion

---

# AAC Regime Card — VOLATILITY COMPRESSION (CALM / COIL)

- **Regime Name:** VOLATILITY COMPRESSION (CALM / COIL)
- **Definition (operational):**
  - Realized volatility drops; ranges tighten; coiling structures appear
  - Breakout probability increases but direction is uncertain pre-break
- **Feature Votes:**
  - Volatility: low and decreasing
  - Structure: compression patterns, lower ATR bands
  - Correlation: often stable
  - Liquidity proxies: fewer gaps; smoother candles
- **Confidence:**
  - High when compression persists across 1h + 1d (or 4h + 1d crypto)
- **Transition Indicators:**
  - Sudden range expansion, breakout volume/flow spike
- **Strategy Cells Favored:**
  - Breakout-prep, straddle/strangle logic (bounded), level mapping
- **Strategy Cells Throttled:**
  - Volatility-chasing systems

---

# AAC Regime Card — LIQUIDITY SHOCK / GAP RISK

- **Regime Name:** LIQUIDITY SHOCK / GAP RISK
- **Definition (operational):**
  - Discontinuous price moves: gaps (equities) / wicks and liquidation cascades (crypto)
  - Liquidity thins; slippage rises; spreads effectively widen
- **Feature Votes:**
  - Liquidity proxies: abnormal gaps/wicks frequency
  - Volatility: episodic spikes (not necessarily persistent)
  - Structure: levels break without tradeable retests
- **Confidence:**
  - High when wick/gap frequency exceeds baseline and fills are inconsistent
- **Transition Indicators:**
  - Liquidity normalization, reduced gaps/wicks, stable order behavior
- **Strategy Cells Favored:**
  - Risk-control cell, hedging, reduced sizing, longer confirmation
- **Strategy Cells Throttled:**
  - Tight stop strategies, high-frequency scalps, strategies needing precise fills

---

# AAC Regime Card — CORRELATION SPIKE / CONTAGION

- **Regime Name:** CORRELATION SPIKE / CONTAGION
- **Definition (operational):**
  - Cross-asset correlation rises materially (indices/mega-caps/crypto move together)
  - Diversification fails; portfolio behaves like one bet
- **Feature Votes:**
  - Correlation: high clustering across watched universe
  - Volatility: often rising or unstable
  - Structure: macro dominates micro; idiosyncratic signals degrade
- **Confidence:**
  - High when correlation cluster score is elevated across multiple sessions
- **Transition Indicators:**
  - Correlation mean reverts; leadership dispersion returns
- **Strategy Cells Favored:**
  - Hedged structures, reduced gross exposure, regime-aware trend
- **Strategy Cells Throttled:**
  - Multi-name diversification assumptions, unhedged broad risk
