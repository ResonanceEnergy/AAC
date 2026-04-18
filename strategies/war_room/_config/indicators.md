# 15-Indicator Composite Model

> Canonical source for indicator definitions, weights, and regime thresholds.
> Referenced by: stages/02-evaluate

## Indicators

| # | Name | Field | Live Source | Weight | Crisis Direction |
|---|---|---|---|---|---|
| 1 | Oil Price | `oil_price` | FRED `DCOILWTICO` | 0.12 | Higher = more crisis |
| 2 | Gold Price | `gold_price` | Finnhub GLD x10.9 | 0.08 | Higher = flight to safety |
| 3 | VIX | `vix` | IBKR live -> FRED `VIXCLS` fallback | 0.10 | Higher = more fear |
| 4 | HY Spread | `hy_spread_bp` | FRED `BAMLH0A0HYM2` (x100 to bp) | 0.08 | Wider = more stress |
| 5 | BDC NAV Discount | `bdc_nav_discount` | yfinance BDC basket P/B (ARCC/MAIN/FSK/OBDC) | 0.07 | Higher = more stress |
| 6 | BDC Non-Accrual | `bdc_nonaccrual_pct` | yfinance BDC P/B proxy (estimated) | 0.05 | Higher = more stress |
| 7 | DeFi TVL Change | `defi_tvl_change_pct` | CoinGecko `/global` defi_market_cap vs baseline | 0.04 | More negative = stress |
| 8 | Stablecoin Depeg | `stablecoin_depeg_pct` | CoinGecko USDT/USDC deviation from $1 | 0.04 | Higher = crisis |
| 9 | BTC Price | `btc_price` | CoinGecko | 0.05 | Lower = crypto stress |
| 10 | Fed Funds Rate | `fed_funds_rate` | FRED `DFF` | 0.05 | Higher = tightening |
| 11 | DXY | `dxy` | FRED `DTWEXBGS` x0.82 | 0.05 | Higher = EM stress |
| 12 | SPY Price | `spy_price` | IBKR live -> Finnhub fallback | 0.07 | Lower = equity stress |
| 13 | **X/Twitter Sentiment** | `x_sentiment` | Twitter v2 (HTTP 402 — degraded to 0.5) | **0.12** | Lower = fear (HEAVY BIAS) |
| 14 | News Severity | `news_severity` | NewsAPI + UW put/call ratio blend | 0.04 | Higher = black swan |
| 15 | Fear & Greed Index | `fear_greed_index` | alternative.me (free, no key) | 0.04 | Lower = fear |

**Total weight: 1.00** (Financial: 0.80, Sentiment: 0.20)

All 15 indicators now have live API feeds via `war_room_live_feeds.py`.
X/Twitter (indicator 13) degrades to 0.5 (neutral) when API returns HTTP 402.

### Data Sources Summary

| Source | Indicators Fed | Key Required | Status |
|---|---|---|---|
| IBKR TWS | VIX, SPY, account data, positions | Port 7496 | LIVE |
| CoinGecko | BTC, stablecoin depeg, DeFi TVL change | COINGECKO_API_KEY (free OK) | WORKING |
| FRED | Oil, Fed rate, DXY, HY spread, VIX fallback | FRED_API_KEY | WORKING |
| Finnhub | SPY, Gold (GLD) | FINNHUB_API_KEY | WORKING |
| yfinance | BDC NAV discount, BDC non-accrual proxy | None (free) | WORKING |
| Unusual Whales | Put/call ratio -> news blend | UNUSUAL_WHALES_API_KEY | WORKING |
| alternative.me | Fear & Greed Index | None (free) | WORKING |
| NewsAPI | News severity | NEWS_API_KEY | WORKING |
| Twitter v2 | X sentiment | X_BEARER_TOKEN | BROKEN (402) |

## Regime Thresholds

| Regime | Composite Score | Positioning |
|---|---|---|
| CALM | 0 — 30 | Reduce options 50%, rotate to income |
| WATCH | 30 — 50 | Maintain positions, tighten stops |
| ELEVATED | 50 — 70 | Full crisis positioning, all arms active |
| CRISIS | 70 — 100 | Max vega puts, gamma scalping, profit-taking above 85 |

## Hysteresis (Anti-Flapping)

| Parameter | Value |
|---|---|
| Threshold crossing buffer | +/- 3 points |
| Minimum hold time | 30 minutes |

A regime flip requires the composite to cross the threshold by 3+ points
AND stay there for 30+ minutes. Prevents noisy oscillation at boundaries.
