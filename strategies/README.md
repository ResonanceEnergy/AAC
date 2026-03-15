# strategies/

Trading strategy implementations and the execution engine that runs them.

## Strategies

| Strategy | File | Description |
|----------|------|-------------|
| ETF-NAV Dislocation | `etf_nav_dislocation.py` | ETF vs. NAV arbitrage |
| Overnight Drift | `overnight_drift_attention_stocks.py` | Attention-based overnight gap trading |
| Worldwide Arbitrage | `worldwide_arbitrage_strategy.py` | Cross-exchange global arb |
| Pairs Trading | `pairs_trading_strategy.py` | Statistical pairs/mean-reversion |
| Momentum | `momentum_strategy.py` | Trend-following signals |
| Mean Reversion | `mean_reversion_strategy.py` | Reversion-to-mean signals |

## Execution Engine

`strategy_execution_engine.py` loads, validates, and runs all active strategies via the shared strategy loader and doctrine system.

## Adding a Strategy

1. Create a new file inheriting from the base strategy class.
2. Register it in `shared/strategy_loader.py` config.
3. The execution engine discovers and activates it automatically.
