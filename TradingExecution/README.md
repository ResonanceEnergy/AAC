# TradingExecution Module

## Overview

The TradingExecution module is the core component responsible for order execution, position management, and risk control in the Accelerated Arbitrage Corp system.

## Components

### trading_engine.py
Main orchestrator for trade execution:
- Order creation and execution
- Position tracking
- Exchange connectivity
- Dry run and paper trading modes

### order_manager.py
Order lifecycle management:
- Order tracking and persistence
- State transitions
- Query/filtering capabilities
- Order history

### risk_manager.py
Pre-trade and portfolio risk management:
- Order validation against limits
- Position sizing
- Daily loss limits
- Concentration monitoring

## Usage

```python
import asyncio
from TradingExecution import TradingEngine, OrderManager, RiskManager
from TradingExecution.trading_engine import OrderSide, OrderType

async def main():
    # Initialize
    engine = TradingEngine()
    await engine.start()
    
    # Create an order
    order = await engine.create_order(
        exchange='binance',
        symbol='BTC/USDT',
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100.0,  # USD value
    )
    
    print(f"Order: {order.order_id}, Status: {order.status.value}")
    
    # Check risk status
    risk_mgr = RiskManager()
    report = risk_mgr.get_risk_report()
    print(f"Daily P&L: ${report['daily_pnl']:.2f}")
    
    await engine.stop()

asyncio.run(main())
```

## Configuration

Copy `.env.example` to `.env` in the project root and configure:
- Exchange API keys
- Risk limits
- Paper trading settings

See `trading_config.yaml` for module-specific settings.

## Safety Features

1. **Dry Run Mode**: Default enabled - logs trades without execution
2. **Paper Trading**: Simulates trades with mock balances
3. **Risk Limits**: Pre-trade validation against configured limits
4. **Daily Loss Limits**: Automatic halt when limits breached

## Status

⚠️ **Phase 1 Implementation** - Core structure in place:
- [x] Order data models
- [x] Position tracking
- [x] Risk validation
- [x] Dry run mode
- [ ] Exchange API connections (Phase 3)
- [ ] Real order execution (Phase 3)
- [ ] WebSocket price feeds (Phase 3)
