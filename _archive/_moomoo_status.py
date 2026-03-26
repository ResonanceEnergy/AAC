#!/usr/bin/env python3
"""Check Moomoo account status, balance, positions, and options capability."""
import sys, io, os
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, r"c:\dev\AAC_fresh")
os.chdir(r"c:\dev\AAC_fresh")

from moomoo import (OpenQuoteContext, OpenSecTradeContext,
                     TrdEnv, SecurityFirm, TrdMarket, Currency, RET_OK)

HOST = "127.0.0.1"
PORT = 11111

qctx = None
tctx = None

# 1. Quote context test
print("=== CONNECTIVITY TEST ===")
try:
    qctx = OpenQuoteContext(host=HOST, port=PORT)
    ret, data = qctx.get_market_snapshot(["US.SPY"])
    if ret == RET_OK:
        row = data.iloc[0]
        print(f"SPY Last: ${row['last_price']}")
        print("Quote context: OK")
    else:
        print(f"Quote FAILED: {data}")
except Exception as e:
    print(f"Quote context ERROR: {e}")
    print("Is Moomoo OpenD running?")

# 2. Trade context + account
print()
print("=== ACCOUNT INFO (USD) ===")
try:
    tctx = OpenSecTradeContext(host=HOST, port=PORT,
                               security_firm=SecurityFirm.FUTUCA,
                               filter_trdmarket=TrdMarket.US)
    ret2, _ = tctx.unlock_trade(password_md5="069069")
    print(f"Trade unlock: {'OK' if ret2 == 0 else 'FAILED'}")

    ret3, data3 = tctx.accinfo_query(trd_env=TrdEnv.REAL, currency=Currency.USD)
    if ret3 == RET_OK:
        for col in ["total_assets", "cash", "market_val", "frozen_cash",
                     "available_funds", "buying_power", "risk_level"]:
            if col in data3.columns:
                val = data3[col].iloc[0]
                if isinstance(val, float):
                    print(f"  {col}: ${val:,.2f}")
                else:
                    print(f"  {col}: {val}")
    else:
        print(f"Account FAILED: {data3}")
except Exception as e:
    print(f"Trade context ERROR: {e}")

# 3. Positions
print()
print("=== POSITIONS ===")
try:
    if tctx:
        ret4, data4 = tctx.position_list_query(trd_env=TrdEnv.REAL)
        if ret4 == RET_OK:
            if data4.empty:
                print("  No positions")
            else:
                for _, row in data4.iterrows():
                    code = row.get("code", "")
                    qty = row.get("qty", 0)
                    cost = row.get("cost_price", 0)
                    mval = row.get("market_val", 0)
                    pl = row.get("pl_val", 0)
                    print(f"  {code}: {qty} @ ${cost} | MV: ${mval} | P&L: ${pl}")
        else:
            print(f"Positions FAILED: {data4}")
except Exception as e:
    print(f"Positions ERROR: {e}")

# 4. Open orders
print()
print("=== OPEN ORDERS ===")
try:
    if tctx:
        ret5, data5 = tctx.order_list_query(trd_env=TrdEnv.REAL)
        if ret5 == RET_OK:
            if data5.empty:
                print("  No open orders")
            else:
                for _, row in data5.iterrows():
                    oid = row.get("order_id", "")
                    code = row.get("code", "")
                    side = row.get("trd_side", "")
                    qty = row.get("qty", 0)
                    price = row.get("price", 0)
                    status = row.get("order_status", "")
                    print(f"  [{oid}] {side} {qty}x {code} @ ${price} - {status}")
        else:
            print(f"Orders FAILED: {data5}")
except Exception as e:
    print(f"Orders ERROR: {e}")

# 5. Test options capability — check if we can get option chains
print()
print("=== OPTIONS CAPABILITY TEST ===")
try:
    if qctx:
        # Try to get option chain for SPY
        ret6, data6 = qctx.get_option_expiration_date(code="US.SPY")
        if ret6 == RET_OK:
            print(f"  SPY option expiry dates available: {len(data6)}")
            if not data6.empty:
                # Show next 5 expiries
                for i, (_, row) in enumerate(data6.head(5).iterrows()):
                    d = row.to_dict()
                    print(f"    {d}")
        else:
            print(f"  Option chain FAILED: {data6}")

        # Check cheap tickers for options
        for ticker in ["OWL", "XLF", "KRE", "UVXY", "HYG"]:
            ret_t, data_t = qctx.get_market_snapshot([f"US.{ticker}"])
            if ret_t == RET_OK and not data_t.empty:
                last = data_t["last_price"].iloc[0]
                print(f"  {ticker}: ${last}")
except Exception as e:
    print(f"Options test ERROR: {e}")

# Cleanup
if qctx:
    qctx.close()
if tctx:
    tctx.close()
print()
print("Done.")
