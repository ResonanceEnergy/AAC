#!/usr/bin/env python3
"""
BUY 10x OWL Jan 15 2027 $5 PUT @ $0.50 limit on Moomoo.
Contract: US.OWL270115P5000
"""
import sys, io
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from moomoo import (
    OpenSecTradeContext,
    OpenQuoteContext,
    SecurityFirm,
    TrdMarket,
    TrdSide,
    TrdEnv,
    OrderType,
)

CONTRACT = "US.OWL270115P5000"
SIDE = TrdSide.BUY
QTY = 10
LIMIT_PRICE = 0.50
ORDER_TYPE = OrderType.NORMAL  # Limit order

print("=" * 60)
print("  OWL PUT ORDER — LIVE EXECUTION")
print("=" * 60)
print(f"  Contract:  {CONTRACT}")
print(f"  Action:    BUY {QTY} contracts")
print(f"  Price:     ${LIMIT_PRICE:.2f} limit")
print(f"  Total:     ${QTY * LIMIT_PRICE * 100:.2f}")
print(f"  Type:      Limit (NORMAL)")
print("=" * 60)

# 1) Get fresh quote
print("\n[1] Fetching live quote...")
qctx = OpenQuoteContext(host="127.0.0.1", port=11111)
ret, snap = qctx.get_market_snapshot([CONTRACT])
if ret == 0:
    bid = snap["bid_price"].iloc[0]
    ask = snap["ask_price"].iloc[0]
    last = snap["last_price"].iloc[0]
    vol = snap["volume"].iloc[0]
    oi = snap["option_open_interest"].iloc[0]
    print(f"  Bid: ${bid}  Ask: ${ask}  Last: ${last}")
    print(f"  Volume: {vol}  OI: {oi}")
    if LIMIT_PRICE < bid:
        print(f"  WARNING: Limit ${LIMIT_PRICE} is BELOW bid ${bid} — may not fill")
    elif LIMIT_PRICE > ask:
        print(f"  NOTE: Limit ${LIMIT_PRICE} is ABOVE ask ${ask} — instant fill likely")
    else:
        print(f"  Limit ${LIMIT_PRICE} is within spread — good")
else:
    print(f"  Quote failed: {snap}")
    print("  Proceeding anyway (quote rights may be limited)")
qctx.close()

# 2) Connect trade context
print("\n[2] Connecting trade context...")
tctx = OpenSecTradeContext(
    host="127.0.0.1",
    port=11111,
    security_firm=SecurityFirm.FUTUCA,
    filter_trdmarket=TrdMarket.US,
)

# 3) Unlock trading
print("[3] Unlocking trade...")
ret, data = tctx.unlock_trade("069069")
if ret != 0:
    print(f"  UNLOCK FAILED: {data}")
    tctx.close()
    sys.exit(1)
print("  Trade unlocked OK")

# 4) Place the order
print(f"\n[4] PLACING ORDER: BUY {QTY}x {CONTRACT} @ ${LIMIT_PRICE:.2f}...")
ret, data = tctx.place_order(
    price=LIMIT_PRICE,
    qty=QTY,
    code=CONTRACT,
    trd_side=SIDE,
    order_type=ORDER_TYPE,
    trd_env=TrdEnv.REAL,
)

if ret == 0:
    print("\n  *** ORDER PLACED SUCCESSFULLY ***")
    print(data.to_string())
    order_id = data["order_id"].iloc[0] if "order_id" in data.columns else "N/A"
    print(f"\n  Order ID: {order_id}")
else:
    print(f"\n  ORDER FAILED: {data}")

# 5) Check order list
print("\n[5] Checking open orders...")
ret2, orders = tctx.order_list_query()
if ret2 == 0:
    if orders.empty:
        print("  No open orders")
    else:
        cols = [c for c in ["order_id", "code", "trd_side", "qty", "price", "order_status", "create_time"] if c in orders.columns]
        print(orders[cols].to_string())
else:
    print(f"  Order query failed: {orders}")

tctx.close()
print("\nDone.")
