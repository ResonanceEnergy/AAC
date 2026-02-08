#!/usr/bin/env python3
"""
Test Partial Fill Models Implementation
Tests for Models A, B, C, D in the AAC Bake-Off Operating Standard
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from TradingExecution.execution_engine import ExecutionEngine


async def test_partial_fill_models():
    """Test all partial fill models"""
    print("=== Testing AAC Partial Fill Models ===\n")

    engine = ExecutionEngine()

    # Test data
    order_quantity = 0.001  # BTC
    market_conditions = {
        'liquidity_ratio': 2.0,  # Good liquidity
        'volatility': 0.02,     # 2% daily volatility
        'spread_bps': 3.0,      # 3 bps spread
        'queue_position': 2,    # 2nd in queue
        'market_pressure': 0.1, # Slight buy pressure
        'queue_ahead_qty': 0.0005,  # 0.0005 BTC ahead
        'liquidity_depth': 50.0,    # 50 BTC available
        'order_size_pct': 0.0005,   # 0.05% of ADV
        'fill_speed': 1.2,      # 20% faster than normal
        'l2_data': {'bids': [], 'asks': []}  # Mock L2 data
    }

    print("Market Conditions:")
    for k, v in market_conditions.items():
        print(f"  {k}: {v}")
    print()

    # Test Model A (Fill Fraction)
    print("1. Testing Model A (Fill Fraction):")
    fill_qty_a = engine.model_a_fill_fraction(order_quantity, market_conditions)
    print(f"   Expected fill quantity: {fill_qty_a:.6f} BTC")
    print(f"   Fill percentage: {fill_qty_a/order_quantity*100:.1f}%")
    print()

    # Test Model B (Hazard/Intensity)
    print("2. Testing Model B (Hazard/Intensity):")
    fill_qty_b, fill_time_b = engine.model_b_hazard_intensity(order_quantity, market_conditions)
    print(f"   Expected fill quantity: {fill_qty_b:.6f} BTC")
    print(f"   Expected fill time: {fill_time_b:.0f} ms")
    print()

    # Test Model C (Queue-Ahead)
    print("3. Testing Model C (Queue-Ahead):")
    fill_qty_c, fill_time_c = engine.model_c_queue_ahead(order_quantity, market_conditions)
    print(f"   Expected fill quantity: {fill_qty_c:.6f} BTC")
    print(f"   Expected fill time: {fill_time_c:.0f} ms")
    print()

    # Test Model D (Adverse Selection)
    print("4. Testing Model D (Adverse Selection):")
    fill_qty_d, slippage_d = engine.model_d_adverse_selection(order_quantity, market_conditions)
    print(f"   Expected fill quantity: {fill_qty_d:.6f} BTC")
    print(f"   Expected slippage: {slippage_d:.1f} bps")
    print()

    # Test Model Selection
    print("5. Testing Optimal Model Selection:")
    model_name, selected_fill_qty, selected_metric = engine.select_optimal_partial_fill_model(
        order_quantity, market_conditions
    )
    print(f"   Selected Model: {model_name}")
    print(f"   Expected fill quantity: {selected_fill_qty:.6f} BTC")
    if model_name == 'D':
        print(f"   Expected slippage: {selected_metric:.1f} bps")
    elif model_name == 'B' or model_name == 'C':
        print(f"   Expected fill time: {selected_metric:.0f} ms")
    print()

    # Test with poor liquidity conditions
    print("6. Testing with Poor Liquidity Conditions:")
    poor_conditions = market_conditions.copy()
    poor_conditions.update({
        'liquidity_ratio': 0.2,  # Poor liquidity
        'spread_bps': 15.0,     # Wide spread
        'volatility': 0.08,     # High volatility
    })

    model_name_poor, fill_qty_poor, metric_poor = engine.select_optimal_partial_fill_model(
        order_quantity, poor_conditions
    )
    print(f"   Selected Model: {model_name_poor}")
    print(f"   Expected fill quantity: {fill_qty_poor:.6f} BTC")
    print(f"   Fill percentage: {fill_qty_poor/order_quantity*100:.1f}%")
    print()

    print("=== Partial Fill Models Test Complete ===")


async def test_integration_with_execution():
    """Test integration with order execution"""
    print("=== Testing Integration with Order Execution ===\n")

    # Enable logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
    
    engine = ExecutionEngine()
    
    # Override the execution mode for testing
    engine.paper_trading = True
    engine.dry_run = False
    
    print(f"Engine dry_run: {engine.dry_run}")
    print(f"Engine paper_trading: {engine.paper_trading}")

    # Create a test order
    from TradingExecution.execution_engine import Order, OrderSide, OrderType, OrderStatus
    from datetime import datetime

    order = Order(
        order_id="TEST_PARTIAL_FILL_001",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.001,
        price=None,  # Market order
        status=OrderStatus.PENDING,
        created_at=datetime.now(),
        exchange="binance"
    )
    
    # Add market price metadata for paper trading
    order.metadata['market_price'] = 45000.0

    # Submit order (should use partial fill models in paper trading)
    print("Submitting test order with partial fill modeling...")
    submitted_order = await engine.submit_order(order)

    print(f"Order Status: {submitted_order.status}")
    print(f"Filled Quantity: {submitted_order.filled_quantity:.6f}")
    print(f"Remaining Quantity: {submitted_order.remaining_quantity:.6f}")

    if 'partial_fill_model' in submitted_order.metadata:
        print(f"Partial Fill Model Used: {submitted_order.metadata['partial_fill_model']}")
        print(f"Model Metric: {submitted_order.metadata.get('model_metric', 'N/A')}")
        print(f"Slippage (bps): {submitted_order.metadata.get('slippage_bps', 'N/A')}")

    print()

    # Get doctrine metrics
    print("Doctrine Metrics with Partial Fill Models:")
    metrics = await engine.get_doctrine_metrics()
    partial_fill_metrics = {
        k: v for k, v in metrics.items()
        if 'partial' in k.lower() or 'model' in k.lower() or k in ['adverse_selection_cost']
    }

    for metric, value in partial_fill_metrics.items():
        print(f"  {metric}: {value}")

    print("\n=== Integration Test Complete ===")


if __name__ == "__main__":
    # Run model tests
    asyncio.run(test_partial_fill_models())

    print("\n" + "="*50 + "\n")

    # Run integration tests
    asyncio.run(test_integration_with_execution())