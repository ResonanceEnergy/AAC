#!/usr/bin/env python3
"""
Scan realistic option pricing for capital engine thesis on Moomoo.
Adjusted strikes to match current market + $900 budget.
"""
import sys, io
if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from moomoo import OpenQuoteContext, RET_OK

qctx = OpenQuoteContext(host="127.0.0.1", port=11111)

JUN_START = "2026-06-01"
JUN_END = "2026-06-30"

# Scan candidates for each leg
SCANS = [
    # OIL LEG — XLE slightly OTM calls (oil upside thesis)
    ("XLE CALL (Oil Upside)", "US.XLE", "CALL", [50.0, 52.5, 55.0, 57.5, 60.0]),
    # SILVER LEG — SLV OTM bull call spread candidates
    ("SLV CALL buy leg (Silver)", "US.SLV", "CALL", [65.0, 66.0, 67.0, 68.0, 70.0]),
    ("SLV CALL sell leg (Silver)", "US.SLV", "CALL", [72.0, 74.0, 75.0, 78.0, 80.0]),
    # GOLD LEG — GLD OTM calls (cheaper strikes)
    ("GLD CALL (Gold Upside)", "US.GLD", "CALL", [475.0, 480.0, 485.0, 490.0, 500.0]),
    # CONSUMER SHORT — XRT bear put spread
    ("XRT PUT buy leg (Retail Bear)", "US.XRT", "PUT", [70.0, 75.0, 80.0]),
    ("XRT PUT sell leg (Retail Bear)", "US.XRT", "PUT", [60.0, 65.0, 70.0]),
]

print("=" * 75)
print("  CAPITAL ENGINE THESIS — REALISTIC STRIKE SCAN")
print("  Budget: ~$900 | Jun 2026 expiry | Moomoo FUTUCA")
print("=" * 75)

for label, underlying, right, strikes in SCANS:
    print(f"\n  {label}")
    print(f"  {'-' * 65}")

    ret, chain = qctx.get_option_chain(code=underlying, start=JUN_START, end=JUN_END)
    if ret != RET_OK or chain.empty:
        print(f"    Chain failed: {chain if ret != RET_OK else 'empty'}")
        continue

    for strike in strikes:
        matches = chain[(chain["strike_price"] == strike) & (chain["option_type"] == right)]
        if matches.empty:
            print(f"    ${strike:>6.0f}: N/A")
            continue

        code = matches.iloc[0]["code"]
        expiry = matches.iloc[0].get("strike_time", "?")

        # Get quote
        ret_q, data_q = qctx.get_market_snapshot([code])
        if ret_q == RET_OK and not data_q.empty:
            row = data_q.iloc[0]
            bid = float(row.get("bid_price", 0) or 0)
            ask = float(row.get("ask_price", 0) or 0)
            last = float(row.get("last_price", 0) or 0)
            vol = int(row.get("volume", 0) or 0)
            oi = int(row.get("open_interest", 0) or 0)
            mid = round((bid + ask) / 2, 2) if bid > 0 and ask > 0 else last
            cost = mid * 100
            print(f"    ${strike:>6.0f}: Bid ${bid:>6.2f} | Ask ${ask:>6.2f} | "
                  f"Mid ${mid:>6.2f} | Last ${last:>6.2f} | "
                  f"Cost/ct ${cost:>7.0f} | Vol {vol:>5} | OI {oi:>5}  [{code}]")
        else:
            print(f"    ${strike:>6.0f}: Quote failed")

print("\n" + "=" * 75)
print("  RECOMMENDED DEPLOYMENT (fits $900 budget):")
print("  1. SLV Jun $67/$75 Bull Call Spread x2  (Silver 53%)")
print("  2. XLE Jun $55 Call x1-2                (Oil 30%)")
print("  3. GLD Jun $490-500 Call x1             (Gold 28%)")
print("  4. XRT Jun $75/$65 Bear Put Spread x1   (Consumer short)")
print("=" * 75)

qctx.close()
print("\nDone.")
