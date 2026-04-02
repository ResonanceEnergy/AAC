#!/usr/bin/env python3
"""Find OWL Jan 2027 $5 put option code on Moomoo."""
import io
import sys

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from moomoo import OpenQuoteContext

qctx = OpenQuoteContext(host="127.0.0.1", port=11111)

ret, data = qctx.get_option_chain(code="US.OWL")
if ret != 0:
    print(f"FAILED: {data}")
    qctx.close()
    sys.exit(1)

# Filter for Jan 2027 puts at $5 strike
puts = data[(data["option_type"] == "PUT") & (data["strike_price"] == 5.0)]
print(f"All OWL $5 Puts ({len(puts)} total):")
print()
for _, row in puts.iterrows():
    print(f"  {row['code']:30s}  expiry={row['strike_time']}  strike=${row['strike_price']:.1f}  id={row['stock_id']}")

# Look specifically for Jan 2027
jan27 = puts[puts["strike_time"].str.startswith("2027-01")]
print(f"\nJan 2027 $5 Puts: {len(jan27)}")
if not jan27.empty:
    for _, row in jan27.iterrows():
        print(f"  CODE: {row['code']}")
        print(f"  Name: {row['name']}")
        print(f"  Expiry: {row['strike_time']}")
        print(f"  Strike: ${row['strike_price']:.2f}")
        print(f"  Stock ID: {row['stock_id']}")
        print(f"  Lot Size: {row['lot_size']}")

        # Get quote for this option
        ret2, data2 = qctx.get_market_snapshot([row["code"]])
        if ret2 == 0:
            print(f"  Bid: ${data2['bid_price'].iloc[0]}")
            print(f"  Ask: ${data2['ask_price'].iloc[0]}")
            print(f"  Last: ${data2['last_price'].iloc[0]}")
            print(f"  Volume: {data2['volume'].iloc[0]}")
            print(f"  Open Interest: {data2.get('open_interest', ['N/A']).iloc[0] if 'open_interest' in data2.columns else 'N/A'}")
        else:
            print(f"  Quote FAILED: {data2}")
else:
    # Show all available expiries with $5 puts
    print("\nNo Jan 2027 found. Available $5 put expiry dates:")
    for _, row in puts.sort_values("strike_time").iterrows():
        print(f"  {row['strike_time']}  ->  {row['code']}")

qctx.close()
print("\nDone.")
