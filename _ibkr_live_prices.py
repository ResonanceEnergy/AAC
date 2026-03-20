#!/usr/bin/env python3
"""Pull REAL live prices from IBKR TWS for all 8 put positions."""
import sys
import io
import time

if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from ib_insync import IB, Option, Stock

ib = IB()
try:
    ib.connect("127.0.0.1", 7496, clientId=99, timeout=10)
    print(f"Connected: {ib.isConnected()}")
    print(f"Account: {ib.managedAccounts()}")
except Exception as e:
    print(f"Connection FAILED: {e}")
    print("Is TWS running on port 7496?")
    sys.exit(1)

# Our 8 real positions
positions_spec = [
    ("ARCC", 17, "20260417", "P"),
    ("PFF",  29, "20260417", "P"),
    ("LQD", 106, "20260620", "P"),
    ("EMB",  90, "20260620", "P"),
    ("MAIN", 50, "20260620", "P"),
    ("JNK",  92, "20260620", "P"),
    ("KRE",  58, "20260620", "P"),
    ("IWM", 230, "20260620", "P"),
]

entry_prices = {
    "ARCC": 0.25, "PFF": 0.40, "LQD": 0.64, "EMB": 0.75,
    "MAIN": 0.85, "JNK": 0.80, "KRE": 1.45, "IWM": 3.96,
}

# ---- STEP 1: Get IBKR portfolio positions (the real truth) ----
print()
print("=" * 100)
print("  IBKR PORTFOLIO -- ACTUAL POSITIONS FROM TWS")
print("=" * 100)
portfolio = ib.portfolio()
if portfolio:
    print()
    print("  {:<6s} {:4s} {:>7s} {:>10s} {:>5s} {:>10s} {:>12s} {:>12s}".format(
        "Sym", "Type", "Strike", "Expiry", "Qty", "AvgCost", "MktValue", "uPnL"))
    print("  " + "-" * 90)
    for pos in portfolio:
        c = pos.contract
        print("  {:<6s} {:4s} {:>7.1f} {:>10s} {:>5.0f} ${:>9.2f} ${:>11.2f} ${:>+11.2f}".format(
            c.symbol, c.secType, c.strike,
            c.lastTradeDateOrContractMonth,
            pos.position, pos.averageCost,
            pos.marketValue, pos.unrealizedPNL))
else:
    print("  No positions found in portfolio!")

# ---- STEP 2: Get underlying stock prices ----
stock_syms = list(set(s[0] for s in positions_spec))
stock_contracts = {}
for sym in stock_syms:
    stk = Stock(sym, "SMART", "USD")
    ib.qualifyContracts(stk)
    ib.reqMktData(stk, "", False, False)
    stock_contracts[sym] = stk

# ---- STEP 3: Get option market data ----
opt_contracts = []
for sym, strike, exp, right in positions_spec:
    opt = Option(sym, exp, strike, right, "SMART")
    ib.qualifyContracts(opt)
    ib.reqMktData(opt, "", False, False)
    opt_contracts.append((sym, strike, exp, right, opt))

# Wait for data to stream in
print()
print("  Waiting for market data (5 seconds)...")
ib.sleep(5)

# ---- STEP 4: Print option prices ----
print()
print("=" * 100)
print("  REAL OPTION PRICES -- LIVE FROM TWS")
print("=" * 100)
print()
print("  {:<6s} {:>6s} {:>10s} {:>8s} {:>7s} {:>7s} {:>7s} {:>7s} {:>6s} {:>8s} {:>7s}".format(
    "Sym", "Strike", "Expiry", "Spot", "Bid", "Ask", "Last", "Mid", "Entry", "P&L", "ROI"))
print("  " + "-" * 92)

total_entry = 0.0
total_current = 0.0

for sym, strike, exp, right, opt in opt_contracts:
    t_opt = ib.ticker(opt)
    t_stk = ib.ticker(stock_contracts[sym])

    spot = t_stk.marketPrice()
    bid = t_opt.bid if t_opt.bid is not None and t_opt.bid > 0 else 0
    ask = t_opt.ask if t_opt.ask is not None and t_opt.ask > 0 else 0
    last = t_opt.last if t_opt.last is not None and t_opt.last > 0 else 0
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last

    entry = entry_prices[sym]
    val = mid if mid > 0 else last
    if val <= 0 and spot and spot > 0:
        val = max(strike - spot, 0)  # intrinsic as last resort

    pnl = (val - entry) * 100
    roi = (val / entry - 1) * 100 if entry > 0 else 0

    total_entry += entry * 100
    total_current += val * 100

    def fmt(v):
        return "${:.2f}".format(v) if v > 0 else "  --"

    print("  {:<6s} ${:>5.0f} {:>10s} {:>8s} {:>7s} {:>7s} {:>7s} {:>7s} ${:>5.2f} ${:>+7.0f} {:>+6.0f}%".format(
        sym, strike, exp,
        "${:.2f}".format(spot) if spot and spot > 0 else "N/A",
        fmt(bid), fmt(ask), fmt(last), fmt(mid),
        entry, pnl, roi))

total_pnl = total_current - total_entry
total_roi = (total_current / total_entry - 1) * 100 if total_entry > 0 else 0
print("  " + "-" * 92)
print("  TOTAL entry: ${:.0f}  |  current: ${:.0f}  |  unrealized P&L: ${:+.0f}  |  ROI: {:+.0f}%".format(
    total_entry, total_current, total_pnl, total_roi))

# ---- STEP 5: Account summary ----
print()
print("=" * 100)
print("  ACCOUNT SUMMARY")
print("=" * 100)
tags = ("NetLiquidation", "TotalCashValue", "BuyingPower",
        "GrossPositionValue", "UnrealizedPnL", "RealizedPnL",
        "AvailableFunds", "MaintMarginReq")
acct = ib.accountSummary()
for item in acct:
    if item.tag in tags:
        try:
            print("  {:<30s} ${:>12,.2f} {}".format(
                item.tag, float(item.value), item.currency))
        except (ValueError, TypeError):
            pass

ib.disconnect()
print()
print("Done.")
