# AAC Gap Analysis & Missing Techniques — 2026-04-08

> Combined internal 10-category infrastructure audit + external research from
> AQR, VectorBT, PyPortfolioOpt, Riskfolio-Lib, ML4T (Stefan Jansen),
> Wikipedia (Greeks, Risk Parity), tastylive strategies, Investopedia.

---

## Executive Summary

AAC has **strong foundations** — 50+ strategies, 8 exchange connectors, a 16-indicator
war room, 12-equation alpha engine, regime detection, Greeks calculation, Monte Carlo
simulation, and live trading on IBKR with real positions. However, there are **critical
gaps** between "code that exists" and "production-grade quant infrastructure." The
biggest missing pieces are:

1. **Portfolio optimization** — no mean-variance, risk parity, Black-Litterman, or HRP
2. **Dynamic correlation** — static 11×11 Cholesky only, no rolling/DCC-GARCH
3. **Walk-forward backtesting** — no out-of-sample validation framework
4. **ML pipeline for production** — sklearn exists but only for ex-post analysis
5. **Smart order execution** — no TWAP/VWAP/SOR algorithms
6. **Hedge auto-execution** — recommendations generated but never acted on

---

## TIER 1 — CRITICAL GAPS (Missing entirely, high impact)

### 1. Portfolio Optimization Engine
**Status**: ❌ Missing  
**What exists**: `strategies/paper_trading/optimizer.py` does composite scoring (not optimization). `strategies/alpha_engine.py` combines signals but doesn't optimize allocations.  
**What's needed**:
- **Mean-Variance Optimization** (Markowitz efficient frontier)
- **Black-Litterman Model** — combine market equilibrium with your alpha views
- **Risk Parity** — equalize risk contribution across assets (Bridgewater All Weather style)
- **Hierarchical Risk Parity (HRP)** — clustering-based, no covariance inversion needed
- **Kelly Criterion at portfolio level** — exists for single positions only
- **CVaR optimization** — minimize tail risk instead of variance

**Recommended library**: `PyPortfolioOpt` (5.6k stars, MIT, covers all above) or `Riskfolio-Lib` (24 risk measures, HRP, NCO, Black-Litterman Bayesian)  
**Where to build**: `strategies/portfolio_optimizer.py`  
**Integrates with**: Alpha engine outputs → optimizer → execution engine  
**Estimated LOC**: 400-600  

### 2. Dynamic Correlation Tracking
**Status**: ❌ Missing  
**What exists**: Static 11×11 correlation matrix in `strategies/matrix_maximizer/monte_carlo.py` using Cholesky decomposition. `strategies/regime_engine.py` has F6 correlation spike (rule-based, not computed). `strategies/paper_trading/risk_manager.py` defines `max_correlated_exposure_pct` but NEVER enforces it.  
**What's needed**:
- **Exponentially Weighted Moving (EWM) correlation** — simple, fast, recent-biased
- **DCC-GARCH** — Dynamic Conditional Correlation, industry standard for time-varying correlations
- **Rolling window correlation** with configurable lookback (21d, 63d, 252d)
- **Contagion detection** — alert when cross-asset correlations spike (crisis regime)
- **Correlation breakdown monitoring** — detect when historical correlations fail

**Where to build**: `strategies/correlation_tracker.py`  
**Integrates with**: Risk manager, regime engine, war room  
**Estimated LOC**: 300-400  
**Dependencies**: numpy, scipy (already installed)

### 3. Walk-Forward Backtesting Framework  
**Status**: ❌ Missing  
**What exists**: `strategies/matrix_maximizer/backtester.py` (MatrixBacktester with attribution) and `strategies/living_doctrine_backtest.py` (signal threshold entries). Both are in-sample only — no walk-forward, no rolling window, no train/test split.  
**What's needed**:
- **Walk-forward optimization** — train on window N, test on window N+1, slide forward
- **Anchored walk-forward** — expanding training window
- **Combinatorial purged cross-validation** (CPCV) — proper financial cross-validation
- **Transaction cost modeling** — commissions, slippage, market impact
- **Performance attribution** — by signal, by regime, by asset class
- **Benchmark comparison** — vs SPY, vs 60/40, vs risk parity

**Recommended approach**: Build on existing backtester, add walk-forward wrapper  
**Where to build**: `strategies/walk_forward_backtester.py`  
**Estimated LOC**: 500-700  
**Key insight from ML4T**: Never use standard k-fold CV on time series — information leakage corrupts results

### 4. ML Production Pipeline
**Status**: ❌ Missing for production use  
**What exists**: `strategies/strategy_analysis_engine.py` uses GradientBoostingClassifier and RandomForestRegressor for **ex-post analysis only** — no trained models making live predictions.  
**What's needed**:
- **Feature engineering pipeline** — technical indicators, fundamental ratios, sentiment scores, regime features → unified feature matrix
- **Model training with walk-forward** — XGBoost/LightGBM for alpha signal generation
- **Online learning** — update models incrementally with new data
- **Feature importance tracking** — SHAP values to explain model decisions
- **Model registry** — version trained models, track performance degradation
- **Alpha factor library** — WorldQuant 101 Formulaic Alphas as features

**Phase 1 (quick win)**: Train XGBoost on existing alpha engine features to predict next-day option premium direction  
**Phase 2**: Add NLP sentiment from Finnhub/NewsAPI as features  
**Phase 3**: Ensemble with war room signals for final score  

**Where to build**: `strategies/ml_pipeline/` (new package)  
**Estimated LOC**: 800-1200  
**Dependencies**: xgboost, lightgbm, shap (new installs needed)

---

## TIER 2 — HIGH PRIORITY GAPS (Partial implementation, significant impact)

### 5. Smart Order Execution
**Status**: ⚠️ Partial  
**What exists**: `TradingExecution/execution_engine.py` sends market/limit orders. 8 connectors work. `TradingExecution/order_manager.py` tracks order state.  
**What's missing**:
- **TWAP** (Time-Weighted Average Price) — split large orders evenly over time
- **VWAP** (Volume-Weighted Average Price) — weight order slices by volume profile
- **Smart Order Routing (SOR)** — route to best venue across IBKR/Moomoo
- **Iceberg orders** — show only partial size, replenish as filled
- **Implementation shortfall tracking** — measure execution quality vs decision price

**Where to build**: `TradingExecution/smart_execution.py`  
**Estimated LOC**: 400-500  
**Priority**: HIGH for accounts >$50K where execution quality matters

### 6. Hedge Auto-Execution
**Status**: ⚠️ Partial  
**What exists**: `strategies/greeks_portfolio_risk.py` generates `HedgeRecommendation` objects with delta/gamma/vega aggregation. `strategies/matrix_maximizer/risk.py` generates `HedgeAlert` for energy/TLT/VIX. `strategies/variance_risk_premium.py` has delta hedging concept with `hedge_frequency=300`.  
**What's missing**: Recommendations are generated but **NEVER automatically executed**. No tail-risk hedging strategy. No hedge ratio optimization.  
**What's needed**:
- **Hedge executor** — take HedgeRecommendation → place order via execution engine
- **Delta-neutral rebalancer** — auto-adjust hedge when portfolio delta exceeds threshold
- **Gamma scalping logic** — profit from gamma by delta-hedging frequently
- **Tail risk hedging** — systematic OTM put buying / VIX call spreading
- **Hedge cost tracking** — P&L attribution between hedge and core positions

**Where to build**: `strategies/hedge_executor.py`  
**Estimated LOC**: 300-400

### 7. Real-Time P&L (Replace Mock)
**Status**: ⚠️ MOCK in production  
**What exists**: `shared/internal_money_monitor.py` is a MOCK with $1M simulated accounts. `CentralAccounting/financial_analysis_engine.py` has daily P&L and drawdown tracking. `CentralAccounting/database.py` uses SQLite.  
**What's needed**:
- **Replace InternalMoneyMonitor** with IBKR account query (already have connector!)
- **Streaming P&L** — subscribe to IBKR real-time P&L updates
- **Multi-account aggregation** — IBKR + Moomoo + WealthSimple in one view
- **Intraday mark-to-market** — for options positions using live greeks
- **P&L attribution** — by strategy, by asset class, by time period

**Where to build**: Replace `shared/internal_money_monitor.py` with real implementation  
**Estimated LOC**: 300-400  
**Quick win**: The IBKR connector already supports `reqAccountSummary` — wire it up

### 8. NLP Sentiment Analysis
**Status**: ❌ Missing  
**What exists**: NewsAPI and Finnhub both working (confirmed in API inventory). News data is fetched but only displayed, never scored.  
**What's needed**:
- **Sentiment scoring** — FinBERT or VADER on Finnhub/NewsAPI headlines
- **Entity extraction** — link sentiment to specific tickers
- **Aggregated sentiment signal** — rolling 24h/7d sentiment score per watched ticker
- **Earnings call analysis** — detect tone shifts in quarterly reports
- **Integration with alpha engine** — sentiment as an alpha factor

**Where to build**: `integrations/sentiment_analyzer.py`  
**Estimated LOC**: 200-300  
**Dependencies**: `transformers` + `finbert-tone` model (or simpler: `vaderSentiment`)  
**Quick win**: VADER is lightweight, no GPU needed, works offline

---

## TIER 3 — MEDIUM PRIORITY (Enhancement, meaningful edge)

### 9. Volatility Surface & Term Structure
**Status**: ⚠️ Partial  
**What exists**: BSM Greeks calculation, implied vol per option.  
**What's missing**: No volatility surface (strike × expiry), no term structure analysis, no vol smile/skew modeling, no vol surface arbitrage detection.  
**What's needed**:
- **Vol surface construction** — from yfinance options chains
- **Skew metrics** — 25-delta risk reversal, butterfly
- **Term structure analysis** — contango/backwardation in VIX futures
- **Vol regime classification** — low/normal/elevated/crisis based on surface shape

**Where to build**: `strategies/vol_surface.py`  
**Estimated LOC**: 300-400

### 10. Cointegration / Pairs Trading
**Status**: ❌ Missing  
**What exists**: Cross-asset seesaw analysis (conceptual), but no formal cointegration testing.  
**What's needed**:
- **Engle-Granger cointegration test** — find cointegrated pairs
- **Johansen test** — for multi-asset cointegration
- **Half-life of mean reversion** — estimate trade duration
- **Z-score entry/exit signals** — standard pairs trading framework
- **Rolling cointegration** — adapt as relationships change

**Where to build**: `strategies/pairs_trading.py`  
**Estimated LOC**: 300-400  
**Dependencies**: `statsmodels` (likely already installed)

### 11. Gradient Boosting Alpha Signals
**Status**: ❌ Missing for production  
**What's needed**:
- **XGBoost/LightGBM/CatBoost** models trained on alpha features
- **SHAP explainability** — understand why model predicts direction
- **Feature selection** — mutual information, boruta, recursive elimination
- **Hyperparameter tuning** — Optuna-based walk-forward tuning

**Part of**: ML Pipeline (Tier 1, #4)

### 12. Volatility-Targeting Position Sizing
**Status**: ❌ Missing  
**What exists**: Kelly criterion and fixed fractional in `strategies/matrix_maximizer/execution.py`. Regime multipliers in `strategies/paper_trading/regime_detector.py`.  
**What's missing**: No target-volatility framework. Position sizes don't adapt to current realized vol.  
**What's needed**:
- **Target vol framework** — size positions so portfolio vol stays at target (e.g., 15%)
- **Inverse-volatility weighting** — allocate more to low-vol assets
- **Vol-adjusted Kelly** — scale Kelly fraction by current vs historical vol

**Where to build**: Extend `strategies/matrix_maximizer/execution.py`  
**Estimated LOC**: 100-150

### 13. Unified Risk Aggregation
**Status**: ⚠️ Fragmented  
**What exists**: 6+ risk files, 4 RiskManager classes across different subsystems. Each manages risk independently.  
**What's needed**:
- **Central risk dashboard** — aggregate risk metrics from ALL subsystems
- **Cross-strategy risk limits** — total portfolio Greeks, total drawdown, total VaR
- **Risk budget allocation** — each strategy gets a risk budget, not a capital budget

**Where to build**: `core/unified_risk.py`  
**Estimated LOC**: 300-400

---

## TIER 4 — LOWER PRIORITY (Nice-to-have, future edge)

### 14. Advanced Options Strategies
**Status**: Partially implemented  
**What exists**: Put selling strategy (live), basic spreads concept.  
**What's missing from tastylive/industry**:
- **Jade Lizard** — short put + short call spread, no upside risk
- **Broken Wing Butterfly** — directional bias with limited risk
- **Iron Condor management** — rolling, adjusting, profit-taking rules
- **Ratio spreads** — leveraged directional plays with defined risk
- **Calendar/diagonal spreads** — exploit term structure

### 15. PCA Eigenportfolios
**Status**: ❌ Missing  
**Use case**: Decompose returns into principal components → identify hidden risk factors → build factor-neutral portfolios.

### 16. Bayesian Dynamic Sharpe Tracking
**Status**: ❌ Missing  
**Use case**: Real-time Bayesian updating of Sharpe ratio estimates with uncertainty bands.

### 17. Deep Reinforcement Learning Agent
**Status**: ❌ Missing  
**From ML4T research**: Deep RL trading agents (DQN, PPO) that learn optimal execution/allocation policies. Very cutting-edge, high complexity.

### 18. GAN Synthetic Data
**Status**: ❌ Missing  
**Use case**: Generate synthetic price paths for stress testing and model training augmentation. Interesting but complex.

### 19. Incremental De-Risking & Recovery Plans
**Status**: ⚠️ Partial  
**What exists**: Circuit breakers (3 files), but they're binary ON/OFF.  
**What's needed**: Gradual position reduction (e.g., reduce 20% at -5% DD, another 20% at -8% DD) and structured recovery plans (re-enter positions systematically after DD recovery).

---

## Implementation Priority Matrix

| Priority | Gap | Effort | Impact | Dependencies |
|----------|-----|--------|--------|-------------|
| 🔴 P1 | Portfolio Optimization | Medium | Very High | PyPortfolioOpt install |
| 🔴 P1 | Dynamic Correlation | Medium | High | numpy only |
| 🔴 P1 | Walk-Forward Backtest | High | Very High | Existing backtester |
| 🔴 P1 | Replace Mock P&L | Low | High | IBKR connector exists |
| 🟠 P2 | ML Pipeline | High | Very High | xgboost, shap installs |
| 🟠 P2 | Hedge Auto-Execution | Medium | High | Greeks module exists |
| 🟠 P2 | Smart Order Execution | Medium | Medium | For >$50K portfolios |
| 🟠 P2 | NLP Sentiment | Low | Medium | vaderSentiment install |
| 🟡 P3 | Vol Surface | Medium | Medium | yfinance exists |
| 🟡 P3 | Pairs Trading | Medium | Medium | statsmodels exists |
| 🟡 P3 | Vol-Target Sizing | Low | Medium | Extend existing |
| 🟡 P3 | Unified Risk | Medium | High | Cross-cutting |
| ⚪ P4 | Advanced Options | Medium | Medium | IBKR connector exists |
| ⚪ P4 | PCA / RL / GAN | High | Speculative | Heavy dependencies |

---

## Recommended Implementation Order

### Sprint 1: Quick Wins (1-2 days each)
1. **Replace Mock P&L** — wire `InternalMoneyMonitor` to real IBKR `reqAccountSummary`
2. **NLP Sentiment** — add VADER scoring to Finnhub/NewsAPI headlines, pipe into alpha engine
3. **Vol-Target Sizing** — extend PositionSizer with inverse-vol weighting

### Sprint 2: Core Infrastructure (3-5 days each)
4. **Dynamic Correlation** — EWM rolling correlation + contagion alerts
5. **Portfolio Optimizer** — integrate PyPortfolioOpt for mean-variance + HRP
6. **Hedge Auto-Execution** — wire HedgeRecommendation → execution engine

### Sprint 3: Advanced Capabilities (5-10 days each)
7. **Walk-Forward Backtester** — proper out-of-sample validation
8. **ML Pipeline Phase 1** — XGBoost on alpha features, walk-forward trained
9. **Vol Surface** — construct from yfinance, add skew signals

### Sprint 4: Edge Generation (ongoing)
10. **Pairs Trading** — cointegration scanner + z-score signals
11. **Smart Order Execution** — TWAP/VWAP for larger positions
12. **Unified Risk** — aggregate all subsystem risks

---

## Python Libraries to Install

```
# Portfolio optimization
pip install pyportfolioopt riskfolio-lib

# ML pipeline
pip install xgboost lightgbm shap optuna

# NLP sentiment
pip install vaderSentiment
# Optional: pip install transformers (for FinBERT, needs GPU or patience)

# Statistical tests (likely already present)
pip install statsmodels arch  # arch for GARCH models
```

---

## Key Research Sources Referenced

| Source | Key Takeaway |
|--------|-------------|
| **PyPortfolioOpt** (5.6k GitHub stars) | Mean-variance, Black-Litterman, HRP in one library |
| **Riskfolio-Lib** (4k GitHub stars) | 24 risk measures, NCO, risk factor models |
| **ML for Trading** (Stefan Jansen, 17k stars) | 24-chapter ML4T: alpha factors, XGBoost, NLP, deep RL |
| **VectorBT** | Vectorized backtesting, parameter optimization at scale |
| **AQR Research** | Active extension, risk parity, diversifying alternatives |
| **Wikipedia Greeks** | Vanna, Charm, Vomma — higher-order Greeks AAC doesn't track |
| **Wikipedia Risk Parity** | Equal risk contribution math, Bridgewater All Weather |
| **tastylive** | Jade Lizard, broken wing butterfly, practical options mgmt |
| **Investopedia** | TWAP, VWAP, implementation shortfall definitions |

---

*Generated 2026-04-08 by comprehensive project audit + internet research.*
