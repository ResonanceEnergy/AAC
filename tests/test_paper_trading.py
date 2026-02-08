#!/usr/bin/env python3
"""
Paper Trading Integration Tests
===============================
Comprehensive testing for the paper trading environment.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.paper_trading import (
    paper_trading_engine,
    OrderType,
    OrderSide,
    OrderStatus,
    initialize_paper_trading
)
from shared.audit_logger import get_audit_logger


async def test_basic_market_orders():
    """Test basic market order execution"""
    print("\n=== Testing Basic Market Orders ===")

    # Test buy order
    order_id = await paper_trading_engine.submit_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET
    )
    print(f"[OK] Submitted buy order: {order_id}")

    await asyncio.sleep(0.1)  # Allow execution

    # Check order status
    orders = paper_trading_engine.get_orders()
    buy_order = next((o for o in orders if o['order_id'] == order_id), None)
    assert buy_order['status'] == 'filled', f"Order not filled: {buy_order}"
    print(f"[OK] Buy order filled: {buy_order['filled_quantity']} @ ${buy_order['average_fill_price']:.2f}")

    # Test sell order
    sell_order_id = await paper_trading_engine.submit_order(
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=50,
        order_type=OrderType.MARKET
    )
    print(f"[OK] Submitted sell order: {sell_order_id}")

    await asyncio.sleep(0.1)

    # Check positions
    positions = paper_trading_engine.get_positions()
    aapl_position = next((p for p in positions if p['symbol'] == 'AAPL'), None)
    assert aapl_position is not None, "Position not found"
    assert aapl_position['quantity'] == 50, f"Expected quantity 50, got {aapl_position['quantity']}"
    print(f"[OK] Position updated: {aapl_position['quantity']} AAPL @ ${aapl_position['average_price']:.2f}")


async def test_limit_orders():
    """Test limit order execution"""
    print("\n=== Testing Limit Orders ===")

    # Get current price
    current_price = await paper_trading_engine._get_current_price("MSFT")
    limit_price = current_price * 0.98  # 2% below current price

    # Submit limit buy order
    order_id = await paper_trading_engine.submit_order(
        symbol="MSFT",
        side=OrderSide.BUY,
        quantity=200,
        order_type=OrderType.LIMIT,
        limit_price=limit_price
    )
    print(f"[OK] Submitted limit buy order: {order_id} @ ${limit_price:.2f}")

    # Wait for potential execution (limit orders may not fill immediately)
    await asyncio.sleep(2)

    # Check if order was filled or still pending
    orders = paper_trading_engine.get_orders()
    order = next((o for o in orders if o['order_id'] == order_id), None)
    print(f"[OK] Limit order status: {order['status']}")

    # Cancel if still pending
    if order['status'] == 'pending':
        cancelled = paper_trading_engine.cancel_order(order_id)
        assert cancelled, "Failed to cancel order"
        print("[OK] Cancelled pending limit order")


async def test_position_management():
    """Test position tracking and P&L calculation"""
    print("\n=== Testing Position Management ===")

    # Reset account for clean test
    await paper_trading_engine.reset_account()

    # Execute multiple trades
    trades = [
        ("SPY", OrderSide.BUY, 100),
        ("QQQ", OrderSide.BUY, 50),
        ("SPY", OrderSide.SELL, 50),  # Partial close
    ]

    for symbol, side, quantity in trades:
        order_id = await paper_trading_engine.submit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET
        )
        await asyncio.sleep(0.1)

    # Check positions
    positions = paper_trading_engine.get_positions()
    print(f"[OK] Current positions: {len(positions)}")

    for position in positions:
        print(f"  {position['symbol']}: {position['quantity']} @ ${position['average_price']:.2f} "
              f"(P&L: ${position['unrealized_pnl']:.2f})")

    # Check account summary
    summary = paper_trading_engine.get_account_summary()
    print(f"[OK] Account equity: ${summary['equity']:,.2f}")
    print(f"[OK] Total P&L: ${summary['total_pnl']:.2f}")


async def test_order_validation():
    """Test order validation and rejection"""
    print("\n=== Testing Order Validation ===")

    # Test insufficient balance
    try:
        await paper_trading_engine.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1000000,  # Very large quantity
            order_type=OrderType.MARKET
        )
        assert False, "Should have rejected large order"
    except ValueError:
        print("[OK] Correctly rejected order due to insufficient balance")

    # Test position size limit
    try:
        await paper_trading_engine.submit_order(
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=100000,  # Very large quantity
            order_type=OrderType.MARKET
        )
        assert False, "Should have rejected order due to position size limit"
    except ValueError:
        print("[OK] Correctly rejected order due to position size limit")


async def test_trade_history():
    """Test trade history recording"""
    print("\n=== Testing Trade History ===")

    # Execute a few trades
    for i in range(3):
        await paper_trading_engine.submit_order(
            symbol=f"TEST{i}",
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            quantity=10 * (i + 1),
            order_type=OrderType.MARKET
        )
        await asyncio.sleep(0.1)

    # Check trade history
    history = paper_trading_engine.get_trade_history(limit=10)
    print(f"[OK] Recorded {len(history)} trades")

    for trade in history[-3:]:  # Show last 3 trades
        print(f"  {trade['symbol']} {trade['side']} {trade['quantity']} @ ${trade['price']:.2f}")


async def test_persistence():
    """Test account state persistence"""
    print("\n=== Testing Persistence ===")

    # Make some changes
    initial_equity = paper_trading_engine.get_account_summary()['equity']

    await paper_trading_engine.submit_order(
        symbol="PERSIST",
        side=OrderSide.BUY,
        quantity=25,
        order_type=OrderType.MARKET
    )
    await asyncio.sleep(0.1)

    # Force save
    await paper_trading_engine._save_account_state()

    # Create new engine instance (simulating restart)
    from shared.paper_trading import PaperTradingEngine
    new_engine = PaperTradingEngine(account_id="paper_account_1")
    await new_engine.initialize()

    # Check if state was preserved
    new_summary = new_engine.get_account_summary()
    new_positions = new_engine.get_positions()

    assert len(new_positions) > 0, "Positions not persisted"
    persist_position = next((p for p in new_positions if p['symbol'] == 'PERSIST'), None)
    assert persist_position is not None, "PERSIST position not found"

    print("[OK] Account state persisted correctly")
    print(f"  Positions loaded: {len(new_positions)}")
    print(f"  PERSIST position: {persist_position['quantity']} shares")


async def run_performance_test():
    """Test performance with multiple concurrent orders"""
    print("\n=== Performance Test ===")

    import time
    start_time = time.time()

    # Submit 50 concurrent orders
    tasks = []
    for i in range(50):
        task = paper_trading_engine.submit_order(
            symbol=f"PERF{i % 10}",  # 10 different symbols
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            quantity=10 + (i % 20),
            order_type=OrderType.MARKET
        )
        tasks.append(task)

    # Wait for all orders to complete
    order_ids = await asyncio.gather(*tasks)
    await asyncio.sleep(1)  # Allow processing

    end_time = time.time()
    duration = end_time - start_time

    # Check results
    summary = paper_trading_engine.get_account_summary()
    positions = paper_trading_engine.get_positions()

    print(f"[OK] Processed {len(order_ids)} orders in {duration:.2f} seconds")
    print(f"  Orders per second: {len(order_ids) / duration:.1f}")
    print(f"  Final positions: {len(positions)}")
    print(f"  Account equity: ${summary['equity']:,.2f}")


async def main():
    """Run all paper trading tests"""
    print("[DEPLOY] Starting Paper Trading Integration Tests")
    print("=" * 50)

    # Initialize
    await initialize_paper_trading()

    # Run tests
    tests = [
        test_basic_market_orders,
        test_limit_orders,
        test_position_management,
        test_order_validation,
        test_trade_history,
        test_persistence,
        run_performance_test,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
            print(f"✅ {test.__name__} PASSED")
        except Exception as e:
            failed += 1
            print(f"[CROSS] {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"[MONITOR] Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("[CELEBRATION] All tests passed!")
    else:
        print("[WARN]️  Some tests failed. Check output above.")

    # Final account summary
    print("\n📈 Final Account Summary:")
    summary = paper_trading_engine.get_account_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())