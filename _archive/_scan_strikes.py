#!/usr/bin/env python3
"""Get current prices and available strikes for capital engine underlyings."""
import io
import sys

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from moomoo import RET_OK, OpenQuoteContext

qctx = OpenQuoteContext(host="127.0.0.1", port=11111)

for ticker in ["XLE", "SLV", "GLD", "XRT"]:
    code = f"US.{ticker}"
    ret, data = qctx.get_market_snapshot([code])
    if ret == RET_OK and not data.empty:
        last = data["last_price"].iloc[0]
        print(f"{ticker}: Last=${last}")
    else:
        print(f"{ticker}: No quote rights - inferring from chain")

    # Get Jun 2026 chain to see available strikes
    ret2, data2 = qctx.get_option_chain(code=code, start="2026-06-01", end="2026-06-30")
    if ret2 == RET_OK and not data2.empty:
        calls = data2[data2["option_type"] == "CALL"].sort_values("strike_price")
        puts = data2[data2["option_type"] == "PUT"].sort_values("strike_price")
        c_strikes = sorted(calls["strike_price"].unique())
        p_strikes = sorted(puts["strike_price"].unique())
        print(f"  CALL strikes ({len(c_strikes)}): {c_strikes[:5]} ... {c_strikes[-5:]}")
        print(f"  PUT  strikes ({len(p_strikes)}): {p_strikes[:5]} ... {p_strikes[-5:]}")
    else:
        print(f"  Chain failed: {data2 if ret2 != RET_OK else 'empty'}")
    print()

qctx.close()
print("Done.")
