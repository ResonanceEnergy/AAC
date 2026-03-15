# TradingExecution/

Core trade execution layer: order management, risk controls, and exchange connectivity.

## Key Modules

| Module | Purpose |
|--------|---------|
| `execution_engine.py` | Real order execution with pre-trade risk checks |
| `trading_engine.py` | Trade orchestration across multiple exchanges |
| `risk_manager.py` | Pre-trade and portfolio-level risk management |
| `order_manager.py` | Order lifecycle, tracking, and persistence |

## Exchange Connectors (`exchange_connectors/`)

| Connector | Exchange |
|-----------|----------|
| `ibkr_connector.py` | Interactive Brokers (via ib_insync) |
| `binance_connector.py` | Binance spot + futures |
| `coinbase_connector.py` | Coinbase Advanced Trade |
| `kraken_connector.py` | Kraken |
| `metalx_connector.py` | Metal X DEX |
