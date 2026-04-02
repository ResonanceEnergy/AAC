#!/usr/bin/env python3
"""Check OWL order status."""
import io
import sys

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from moomoo import OpenSecTradeContext, SecurityFirm, TrdEnv, TrdMarket

tctx = OpenSecTradeContext(
    host="127.0.0.1", port=11111,
    security_firm=SecurityFirm.FUTUCA,
    filter_trdmarket=TrdMarket.US,
)
tctx.unlock_trade("069069")

ret, orders = tctx.order_list_query(trd_env=TrdEnv.REAL)
if ret == 0 and not orders.empty:
    for _, row in orders.iterrows():
        oid = row.get("order_id", "")
        code = row.get("code", "")
        side = row.get("trd_side", "")
        qty = row.get("qty", "")
        price = row.get("price", "")
        status = row.get("order_status", "")
        dealt = row.get("dealt_qty", 0)
        avg = row.get("dealt_avg_price", 0)
        err = row.get("last_err_msg", "")
        print(f"Order {oid}")
        print(f"  {side} {qty}x {code} @ {price}")
        print(f"  Status: {status}")
        print(f"  Filled: {dealt} @ avg {avg}")
        if err:
            print(f"  Error: {err}")
else:
    print(f"No orders or query failed (ret={ret})")

ret2, pos = tctx.position_list_query(trd_env=TrdEnv.REAL)
if ret2 == 0 and not pos.empty:
    print("\nPOSITIONS:")
    for _, p in pos.iterrows():
        code = p.get("code", "")
        qty = p.get("qty", "")
        cost = p.get("cost_price", "")
        mkt = p.get("market_val", "")
        pnl = p.get("pl_val", "")
        print(f"  {code}  qty={qty}  cost={cost}  mktVal={mkt}  pnl={pnl}")
else:
    print("\nNo positions yet")

tctx.close()
