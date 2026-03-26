#!/usr/bin/env python3
"""Quick Moomoo connectivity + OWL options check."""
import sys, io
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from moomoo import (OpenQuoteContext, OpenSecTradeContext,
                     TrdEnv, SecurityFirm, TrdMarket, Currency)

HOST = "127.0.0.1"
PORT = 11111

# 1. Quote context
print("=== QUOTE CONTEXT ===")
qctx = OpenQuoteContext(host=HOST, port=PORT)
ret, data = qctx.get_market_snapshot(["US.OWL"])
if ret == 0:
    print("OWL snapshot:")
    for col in ["code", "name", "last_price", "open_price", "high_price", "low_price", "volume"]:
        if col in data.columns:
            print(f"  {col}: {data[col].iloc[0]}")
else:
    print(f"OWL snapshot FAILED: {data}")

# 2. Trade context
print("\n=== TRADE CONTEXT ===")
tctx = OpenSecTradeContext(host=HOST, port=PORT,
                           security_firm=SecurityFirm.FUTUCA,
                           filter_trdmarket=TrdMarket.US)
ret2, data2 = tctx.unlock_trade(password_md5="069069")
if ret2 == 0:
    print("Trade unlock: OK")
else:
    print(f"Trade unlock FAILED: {data2}")

# 3. Account info
print("\n=== ACCOUNT INFO (USD) ===")
ret3, data3 = tctx.accinfo_query(trd_env=TrdEnv.REAL, currency=Currency.USD)
if ret3 == 0:
    for col in data3.columns:
        print(f"  {col}: {data3[col].iloc[0]}")
else:
    print(f"Account info FAILED: {data3}")

# 4. Positions
print("\n=== POSITIONS ===")
ret4, data4 = tctx.position_list_query(trd_env=TrdEnv.REAL)
if ret4 == 0:
    if data4.empty:
        print("  No positions")
    else:
        print(data4.to_string())
else:
    print(f"Positions FAILED: {data4}")

# 5. Option chain for OWL
print("\n=== OWL OPTION CHAIN ===")
ret5, data5 = qctx.get_option_chain(code="US.OWL")
if ret5 == 0:
    print(f"Expiry dates found: {len(data5)}")
    print(data5.head(20).to_string())
else:
    print(f"Option chain FAILED: {data5}")

# 6. If chain worked, get option list for Jan 2027
if ret5 == 0 and not data5.empty:
    # Look for Jan 2027 expiry
    print("\n=== OWL JAN 2027 PUTS ===")
    jan27_rows = data5[data5["option_expiry_date_distance"].notna()]
    print("All expiries:")
    if "strike_time" in data5.columns:
        for _, row in data5.iterrows():
            print(f"  {row.get('strike_time', 'N/A')}")
    elif "option_expiry_date_distance" in data5.columns:
        print(data5[["option_expiry_date_distance"]].to_string())

    # Try to get the actual option list
    ret6, data6 = qctx.get_option_list(code="US.OWL", index_option_type=None)
    if ret6 == 0:
        print(f"\nOption list entries: {len(data6)}")
        # Filter for puts near $5 strike, Jan 2027
        if not data6.empty:
            cols_show = [c for c in ["code", "name", "option_type", "strike_price",
                                      "strike_time", "option_expiry_date_distance"] if c in data6.columns]
            puts = data6[data6.get("option_type", "") == "PUT"] if "option_type" in data6.columns else data6
            near5 = puts  # show all for now
            print(near5[cols_show].head(40).to_string())
    else:
        print(f"Option list FAILED: {data6}")

qctx.close()
tctx.close()
print("\nDone.")
