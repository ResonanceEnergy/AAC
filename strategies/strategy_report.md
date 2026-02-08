# ACC Arbitrage Strategies

## Summary

- Total Strategies: 50
- Valid Strategies: 50
.1%

## Strategies by Category

### Etf Arbitrage

- [PASS] **ETF–NAV Dislocation Harvesting** (ID: 1)
  - Exploit ETF price vs. fair value gaps using creation/redemption, hedged with index futures.
  - Sources: turn6search94, turn6search122

- [PASS] **Flow-Pressure Contrarian (ETF/Funds)** (ID: 16)
  - Short names with self-inflated returns from flow-chasing; expect reversion.
  - Sources: turn9search168, turn9search170

- [PASS] **ETF Primary-Market Routing** (ID: 19)
  - Use create/redeem to reduce impact on large ETF orders and capture NAV edges.
  - Sources: turn9search169

- [PASS] **ETF Create/Redeem Latency Edge** (ID: 33)
  - Arb stale NAV during big underlying moves by tapping primary liquidity.
  - Sources: turn9search169

- [PASS] **Bubble‑Watch Flow Contrarian (ETFs)** (ID: 34)
  - Short/underweight names where self-inflated flows flag impending unwind.
  - Sources: turn9search168

- [PASS] **Option‑Trading ETF Rollover Signal** (ID: 36)
  - Front‑run/absorb systematic call‑writing ETF rolls that shape IV surface.
  - Sources: turn9search172

### Index Arbitrage

- [PASS] **Index Reconstitution & Closing-Auction Liquidity** (ID: 2)
  - Predict adds/deletes; provide liquidity into the close to monetize execution frictions.
  - Sources: turn9search150, turn9search144, turn9search147

- [PASS] **Active Dispersion (Correlation Risk Premium)** (ID: 8)
  - Long single-name vol vs short index vol only when IC >> RC; de-risk on compression.
  - Sources: turn9search127, turn9search126

- [PASS] **Conditional Correlation Carry** (ID: 9)
  - Trade negative down-correlation premium using sector/index options.
  - Sources: turn9search160

- [PASS] **Index Cash-and-Carry** (ID: 15)
  - Exploit spot–futures fair-value drifts around dividends/financing with basket hedges.
  - Sources: turn9search124

- [PASS] **Be the Patient Counterparty on Rebalance Days** (ID: 17)
  - Provide liquidity to trackers demanding immediacy at reconstitution.
  - Sources: turn9search154

- [PASS] **Index Inclusion Fade** (ID: 30)
  - Fade S&P 500 inclusion pops over 2–8 weeks given structural decline of index effect.
  - Sources: turn9search150, turn9search151, turn9search153

- [PASS] **Reconstitution Close Microstructure** (ID: 31)
  - Target Russell day: volumes +220% in last 30 minutes; specialize per venue rules.
  - Sources: turn9search144

### Volatility Arbitrage

- [PASS] **FOMC Cycle & Pre-Announcement Drift** (ID: 5)
  - Harvest pre-FOMC equity premium with regime filter and tight risk.
  - Sources: turn9search133, turn9search134, turn9search132, turn9search137

- [PASS] **Variance Risk Premium (Cross-Asset)** (ID: 6)
  - Systematically sell variance where IV > RV across assets and tenors.
  - Sources: turn9search156, turn9search157

- [PASS] **Session-Split VRP** (ID: 7)
  - Short variance overnight and neutral/long intraday per VRP decomposition.
  - Sources: turn9search158

- [PASS] **Earnings IV Run-Up / Crush** (ID: 13)
  - Short defined-risk vega into earnings where implied move exceeds realized distribution.
  - Sources: turn9search178, turn9search179

- [PASS] **IV–RV Alignment Trades** (ID: 14)
  - Exploit IV–RV gap with tenor/strike-matched RV; avoid catalyst traps.
  - Sources: turn9search174, turn9search175

- [PASS] **VRP Term/Moneyness Tilt** (ID: 21)
  - Bias short-vol to maturities/moneyness with better variance pricing compensation.
  - Sources: turn9search157

- [PASS] **Flow Pressure & Real-Economy Feedback (Credit-Equity)** (ID: 24)
  - Exploit liquidity-driven selling that raises financing costs; contrarian entries.
  - Sources: turn9search171

- [PASS] **Pre-FOMC VIX/Equity Pair** (ID: 28)
  - Short VIX / long equities into announcements, flatten after; effect is short-lived now.
  - Sources: turn9search132, turn9search133

- [PASS] **Euronext Imbalance Capture** (ID: 32)
  - Harvest higher imbalance notional on EU rebalance days via auction tools.
  - Sources: turn9search149

- [PASS] **Cross‑Asset VRP Basket** (ID: 37)
  - Diversify variance selling into commodities/FX/bonds where VRP is significant.
  - Sources: turn9search156

- [PASS] **VRP Term‑Slope Timing** (ID: 38)
  - Lean shorter maturities when the variance risk premium slope steepens in stress.
  - Sources: turn9search157

- [PASS] **Robust VRP via Synthetic Variance Swaps** (ID: 39)
  - Replicate variance swaps from options for cleaner VRP capture.
  - Sources: turn9search161

- [PASS] **Overnight vs Intraday Variance Skew** (ID: 40)
  - Short overnight variance, long intraday variance where decomposition supports.
  - Sources: turn9search158

- [PASS] **Event Vega Calendars** (ID: 47)
  - Use calendars/diagonals to monetize earnings IV crush while capping tail risk.
  - Sources: turn9search178, turn9search179

- [PASS] **Tenor‑Matched IV–RV** (ID: 48)
  - Align RV horizon to option tenor; sidestep common IV–RV pitfalls.
  - Sources: turn9search174, turn9search175

### Event Driven

- [PASS] **Turn-of-the-Month Overlay** (ID: 10)
  - Own equities from last trading day to +3 days; implement via futures for capital efficiency.
  - Sources: turn9search180, turn9search182

- [PASS] **Post-Earnings/Accruals Subset Alpha** (ID: 12)
  - Target extreme surprises with clean accruals/low coverage; avoid naïve PEAD.
  - Sources: turn9search166, turn9search167

- [PASS] **Monetary Momentum Window** (ID: 20)
  - Position over multi-week drift around expansionary vs contractionary FOMC decisions.
  - Sources: turn9search135, turn9search132

- [PASS] **Pre-FOMC Regime Switch Filter** (ID: 29)
  - Disable pre-FOMC equity drift trade post-2015 unless uncertainty is elevated.
  - Sources: turn9search137

- [PASS] **Contextual Accruals** (ID: 45)
  - Use accruals in micro‑caps/low‑institutional settings where persistence remains.
  - Sources: turn9search165, turn9search163

- [PASS] **PEAD Disaggregation** (ID: 46)
  - Within‑firm signal construction to avoid aggregation bias in post‑earnings drift.
  - Sources: turn9search166

### Seasonality

- [PASS] **Overnight vs. Intraday Split (News-Guided)** (ID: 4)
  - Go long overnight, neutral intraday with NLP topic filters to pick names.
  - Sources: turn9search138, turn9search140

- [PASS] **Overnight Jump Reversion** (ID: 11)
  - Fade large overnight jumps with short-horizon mean reversion controls.
  - Sources: turn9search139

- [PASS] **NLP-Guided Overnight Selector** (ID: 22)
  - Use topic prevalence models to pick overnight longs/shorts.
  - Sources: turn9search138

- [PASS] **Attention-Weighted TOM Overlay** (ID: 25)
  - Combine TOM calendar with attention/flow filters to enhance hit rate.
  - Sources: turn9search182, turn9search140

- [PASS] **Weekly Overnight Seasonality Timing** (ID: 26)
  - Go long Mon–Tue overnight and avoid/short Fri→Mon per documented weekly pattern.
  - Sources: turn9search143, turn9search140

- [PASS] **Clientele Split Allocator** (ID: 27)
  - Allocate to overnight vs intraday based on household/intermediary sentiment regime.
  - Sources: turn9search142

- [PASS] **Overnight Jump Fade (Stock‑Specific)** (ID: 44)
  - Systematically fade prior‑night jump portfolios with short‑term horizons.
  - Sources: turn9search139

- [PASS] **TOM Futures‑Only Overlay** (ID: 49)
  - Exploit TOM via S&P futures; strong persistence documented in futures.
  - Sources: turn9search184, turn9search181

- [PASS] **Overnight Drift in Attention Stocks** (ID: 50)
  - Focus overnight exposure on meme/attention names where effect is strongest.
  - Sources: turn9search141, turn9search140

### Flow Based

- [PASS] **Muni Fund Outflow Liquidity Provision** (ID: 35)
  - Buy pressured muni bonds during mutual fund outflow waves; spreads mean-revert.
  - Sources: turn9search173

### Market Making

- [PASS] **Closing-Auction Imbalance Micro-Alpha** (ID: 3)
  - Use real-time imbalance/paired-quantity to forecast final prints in the last minutes.
  - Sources: turn9search146, turn9search145

- [PASS] **EU Closing-Auction Imbalance Unlock** (ID: 18)
  - Target Euronext's untapped imbalance notional with venue-specific tools.
  - Sources: turn9search149

- [PASS] **Auction-Aware MM with RL** (ID: 23)
  - RL market-making that anticipates closing auctions to optimize inventory transfer.
  - Sources: turn9search148

### Correlation

- [PASS] **Conditional Dependence Trades** (ID: 41)
  - Trade up/down correlation skews inferred from joint option surfaces.
  - Sources: turn9search160

- [PASS] **IC–RC Gate for Dispersion** (ID: 42)
  - Engage dispersion only when implied minus realized correlation is wide.
  - Sources: turn9search127, turn9search131

- [PASS] **Concentration‑Aware Dispersion** (ID: 43)
  - Deploy dispersion when mega‑cap concentration risk is elevated.
  - Sources: turn9search130

