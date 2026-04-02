"""Test IBKR integration — run all 11 data feeds and display results."""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

from strategies.war_room_live_feeds import apply_live_data_to_indicators, fetch_all_live_data


async def test():
    print("=" * 72)
    print("IBKR INTEGRATION TEST -- Fetching all 11 data feeds...")
    print("=" * 72)
    result = await fetch_all_live_data()

    # IBKR status
    print("\n--- IBKR ---")
    print(f"  Connected: {result.ibkr_connected}")
    if result.ibkr_connected:
        nl = result.ibkr_net_liquidation
        bp = result.ibkr_buying_power
        tc = result.ibkr_total_cash
        upnl = result.ibkr_unrealized_pnl
        rpnl = result.ibkr_realized_pnl
        mm = result.ibkr_maint_margin
        print(f"  Net Liquidation: ${nl:,.2f}" if nl else "  Net Liquidation: N/A")
        print(f"  Buying Power:    ${bp:,.2f}" if bp else "  Buying Power: N/A")
        print(f"  Total Cash:      ${tc:,.2f}" if tc else "  Total Cash: N/A")
        print(f"  Unrealized P&L:  ${upnl:,.2f}" if upnl else "  Unrealized P&L: N/A")
        print(f"  Realized P&L:    ${rpnl:,.2f}" if rpnl else "  Realized P&L: N/A")
        print(f"  Maint Margin:    ${mm:,.2f}" if mm else "  Maint Margin: N/A")
        print(f"  Positions: {len(result.ibkr_positions)}")
        for p in result.ibkr_positions:
            print(f"    {p['symbol']:6s} {p['sec_type']:4s} qty={p['quantity']:>4} "
                  f"avgCost={p['avg_cost']:>8.2f} mktVal={p['market_value']:>10.2f}")
        if result.ibkr_spy_price:
            print(f"  IBKR SPY: ${result.ibkr_spy_price:.2f}")
        else:
            print("  IBKR SPY: N/A")
    else:
        print("  (TWS/Gateway not running — IBKR data skipped)")

    # VIX
    print("\n--- VIX ---")
    if result.ibkr_vix:
        print(f"  VIX: {result.ibkr_vix:.2f}")
    else:
        print("  VIX: N/A (no IBKR or FRED VIXCLS)")

    # Apply to indicators
    ind = apply_live_data_to_indicators(result)
    from strategies.war_room_engine import compute_composite_score
    score = compute_composite_score(ind)

    print("\n--- INDICATOR STATE (all 15, post-IBKR) ---")
    print(f"  Oil:          ${ind.oil_price:.2f}")
    print(f"  Gold:         ${ind.gold_price:.2f}")
    print(f"  VIX:          {ind.vix:.2f}")
    print(f"  HY Spread:    {ind.hy_spread_bp:.0f}bp")
    print(f"  BDC NAV Disc: {ind.bdc_nav_discount:.1f}%")
    print(f"  BDC Nonaccr:  {ind.bdc_nonaccrual_pct:.1f}%")
    print(f"  DeFi TVL:     {ind.defi_tvl_change_pct:.1f}%")
    print(f"  Stablecoin:   {ind.stablecoin_depeg_pct:.3f}%")
    print(f"  BTC:          ${ind.btc_price:,.0f}")
    print(f"  Fed Rate:     {ind.fed_funds_rate:.2f}%")
    print(f"  DXY:          {ind.dxy:.2f}")
    print(f"  SPY:          ${ind.spy_price:.2f}")
    print(f"  X Sentiment:  {ind.x_sentiment}")
    print(f"  News Sev:     {ind.news_severity}")
    print(f"  Fear&Greed:   {ind.fear_greed_index}")
    print(f"\n  COMPOSITE: {score['composite_score']:.1f}/100 [{score['regime']}]")

    # Portfolio state
    from strategies.war_room_engine import ACCOUNTS, CURRENT_POSITIONS
    ibkr_pos = [p for p in CURRENT_POSITIONS if p.account == "IBKR"]
    other_pos = [p for p in CURRENT_POSITIONS if p.account != "IBKR"]
    print(f"\n--- PORTFOLIO ---")
    print(f"  IBKR positions: {len(ibkr_pos)}")
    for p in ibkr_pos:
        print(f"    {p.symbol:6s} {p.position_type:5s} qty={p.quantity:>3} "
              f"entry={p.entry_price:.2f} curr={p.current_price:.2f} "
              f"PnL=${p.pnl:+.2f}")
    print(f"  Other positions: {len(other_pos)}")
    print(f"  IBKR account: {ACCOUNTS.get('IBKR', {})}")

    # Summary
    print(f"\nSummary: {result.summary()}")
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for e in result.errors:
            print(f"  - {e}")
    else:
        print("\nNo errors -- all feeds healthy!")


if __name__ == "__main__":
    asyncio.run(test())
