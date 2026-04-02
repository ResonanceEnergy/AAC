#!/usr/bin/env python3
"""Check ALL available OWL option expiration dates on Moomoo."""
import io
import sys

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from moomoo import OpenQuoteContext

qctx = OpenQuoteContext(host="127.0.0.1", port=11111)

# 1) Get all available expiration dates for OWL
print("=== get_option_expiration_date ===")
ret, data = qctx.get_option_expiration_date(code="US.OWL")
if ret == 0:
    print(f"Expiry dates ({len(data)}):")
    for _, row in data.iterrows():
        d = row.to_dict()
        print(f"  {d}")
else:
    print(f"FAILED: {data}")

# 2) Query Jan 2027 chain (30-day window limit)
print()
print("=== get_option_chain start=2027-01-01 end=2027-01-31 ===")
ret2, data2 = qctx.get_option_chain(code="US.OWL", start="2027-01-01", end="2027-01-31")
if ret2 == 0:
    print(f"Contracts found: {len(data2)}")
    if not data2.empty:
        puts5 = data2[(data2["strike_price"] == 5.0) & (data2["option_type"] == "PUT")]
        print(f"$5 Puts: {len(puts5)}")
        for _, row in puts5.iterrows():
            code = row["code"]
            name = row["name"]
            exp = row["strike_time"]
            strike = row["strike_price"]
            lot = row["lot_size"]
            sid = row["stock_id"]
            print(f"  CODE: {code}")
            print(f"  Name: {name}")
            print(f"  Expiry: {exp}")
            print(f"  Strike: ${strike:.2f}")
            print(f"  Lot Size: {lot}")
            print(f"  Stock ID: {sid}")

        # Show all strikes for Jan 2027 puts
        print()
        print("All Jan 2027 PUT strikes:")
        allp = data2[data2["option_type"] == "PUT"].sort_values("strike_price")
        for _, row in allp.iterrows():
            code = row["code"]
            strike = row["strike_price"]
            print(f"  {code:35s}  strike=${strike:.1f}")

        # Try to get quote for $5 put
        if not puts5.empty:
            target = puts5.iloc[0]["code"]
            print(f"\n=== Quote for {target} ===")
            ret_q, data_q = qctx.get_market_snapshot([target])
            if ret_q == 0:
                for col in data_q.columns:
                    val = data_q[col].iloc[0]
                    print(f"  {col}: {val}")
            else:
                print(f"  Quote FAILED: {data_q}")
    else:
        print("  (empty)")
else:
    print(f"FAILED: {data2}")

qctx.close()
print("\nDone.")
