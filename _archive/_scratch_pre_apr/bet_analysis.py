"""$500 bankroll analysis across Polymarket Black Swan opportunities."""
import asyncio
import logging
import sys

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
import strategies.polymarket_blackswan_scanner as bs


async def run():
    s = bs.PolymarketBlackSwanScanner()
    try:
        opps = await s.scan(max_pages=5)
    finally:
        await s.close()

    BANKROLL = 500.0
    total_opps = len(opps)

    print("=" * 70)
    print(f"  POLYMARKET BET ANALYSIS -- $500 USD BANKROLL")
    print(f"  {total_opps} opportunities at Standard tier (25c cap)")
    print("=" * 70)

    # --- STRATEGY 1: Equal $25 on Top 20 ---
    print()
    print("-" * 70)
    print("  STRATEGY 1: Equal $25 on Top 20 (highest edge)")
    print("-" * 70)
    top20 = opps[:20]
    bet_per = BANKROLL / 20
    total_cost = 0
    total_if_win = 0
    print(f"  {'#':>3} {'Price':>7} {'Cost':>6} {'Win$':>8} {'Mult':>6}  Market")
    for i, o in enumerate(top20):
        shares = bet_per / o.market_price
        win_payout = shares * 1.0
        total_cost += bet_per
        total_if_win += win_payout
        q = o.market_question[:48] if hasattr(o, "question") else "?"
        print(f"  {i+1:>3} {o.market_price*100:>5.1f}c ${bet_per:>5.0f} ${win_payout:>7.0f} {win_payout/bet_per:>5.1f}x  {q}")

    avg_payout = total_if_win / 20
    print()
    print(f"  TOTAL COST:          ${total_cost:.0f}")
    print(f"  IF ALL 20 HIT:       ${total_if_win:,.0f} ({total_if_win/total_cost:.0f}x return)")
    print(f"  AVG PAYOUT/WINNER:   ${avg_payout:,.0f}")
    print(f"  BREAK-EVEN:          {total_cost/avg_payout:.1f} winners out of 20")

    # --- STRATEGY 2: Top 10 concentrated ---
    print()
    print("-" * 70)
    print("  STRATEGY 2: $50 each on Top 10 (concentrated)")
    print("-" * 70)
    top10 = opps[:10]
    bet_per = BANKROLL / 10
    total_cost = 0
    total_if_win = 0
    print(f"  {'#':>3} {'Price':>7} {'Cost':>6} {'Win$':>8} {'Mult':>6}  Market")
    for i, o in enumerate(top10):
        shares = bet_per / o.market_price
        win_payout = shares * 1.0
        total_cost += bet_per
        total_if_win += win_payout
        q = o.market_question[:48] if hasattr(o, "question") else "?"
        print(f"  {i+1:>3} {o.market_price*100:>5.1f}c ${bet_per:>5.0f} ${win_payout:>7.0f} {win_payout/bet_per:>5.1f}x  {q}")

    avg_payout = total_if_win / 10
    print()
    print(f"  TOTAL COST:          ${total_cost:.0f}")
    print(f"  IF ALL 10 HIT:       ${total_if_win:,.0f} ({total_if_win/total_cost:.0f}x return)")
    print(f"  AVG PAYOUT/WINNER:   ${avg_payout:,.0f}")
    print(f"  BREAK-EVEN:          {total_cost/avg_payout:.1f} winners out of 10")

    # --- STRATEGY 3: Max spread ---
    print()
    print("-" * 70)
    print("  STRATEGY 3: Spread across ALL opportunities")
    print("-" * 70)
    available = min(50, total_opps)
    bet_per = BANKROLL / available
    total_cost = 0
    total_if_win = 0
    payouts = []
    for o in opps[:available]:
        shares = bet_per / o.market_price
        win_payout = shares * 1.0
        total_cost += bet_per
        total_if_win += win_payout
        payouts.append(win_payout)
    avg_payout = total_if_win / available
    print(f"  Bets: {available} x ${bet_per:.2f} each = ${total_cost:.0f}")
    print(f"  IF ALL HIT:          ${total_if_win:,.0f}")
    print(f"  Smallest single win: ${min(payouts):,.0f}")
    print(f"  Largest single win:  ${max(payouts):,.0f}")
    print(f"  AVG PAYOUT/WINNER:   ${avg_payout:,.0f}")
    print(f"  BREAK-EVEN:          {total_cost/avg_payout:.1f} winners out of {available}")

    # --- CATEGORY BREAKDOWN ---
    print()
    print("-" * 70)
    print("  BY CATEGORY (if $500 allocated per category)")
    print("-" * 70)
    cats = {}
    for o in opps:
        c = o.category.upper()
        if c not in cats:
            cats[c] = []
        cats[c].append(o)
    print(f"  {'Category':>18} {'#Bets':>6} {'AvgPrice':>9} {'CheapestBet':>12} {'If1Wins':>10}")
    for cat, items in sorted(cats.items(), key=lambda x: -len(x[1])):
        n = len(items)
        avg_price = sum(o.market_price for o in items) / n
        cheapest = min(items, key=lambda o: o.market_price)
        # if you put $25 on the cheapest, what do you get?
        cheap_win = 25.0 / cheapest.market_price
        print(f"  {cat:>18} {n:>6} {avg_price*100:>7.1f}c {cheapest.market_price*100:>10.2f}c ${cheap_win:>9,.0f}")

    # --- EXPECTED VALUE ---
    print()
    print("-" * 70)
    print("  EXPECTED VALUE (using thesis probabilities)")
    print("-" * 70)
    for label, subset, n in [("Top 10", opps[:10], 10), ("Top 20", opps[:20], 20), ("Top 50", opps[:50], 50)]:
        bp = BANKROLL / n
        ev_total = 0
        for o in subset:
            shares = bp / o.market_price
            win_payout = shares * 1.0
            ev = o.thesis_probability * win_payout - bp
            ev_total += ev
        exp_return = ev_total + BANKROLL
        print(f"  {label} (${bp:.0f}/bet x {n}):")
        print(f"    Expected Value:  ${ev_total:>+,.0f}")
        print(f"    Expected Return: ${exp_return:>,.0f} ({exp_return/BANKROLL*100:.0f}% of $500)")
        print()

    # --- REALISTIC SCENARIOS ---
    print("-" * 70)
    print("  REALISTIC SCENARIOS (Top 20, $25/bet)")
    print("-" * 70)
    bet_per = 25.0
    payouts_20 = []
    for o in opps[:20]:
        payouts_20.append(bet_per / o.market_price)
    payouts_20.sort(reverse=True)
    scenarios = [
        ("0 of 20 hit (total loss)", 0, 0),
        ("1 of 20 hit (worst winner)", 1, -1),
        ("1 of 20 hit (best winner)", 1, 0),
        ("1 of 20 hit (median winner)", 1, 9),
        ("2 of 20 hit (2 best)", 2, 0),
        ("3 of 20 hit", 3, 0),
        ("5 of 20 hit", 5, 0),
    ]
    for desc, wins, start_idx in scenarios:
        if wins == 0:
            total_return = 0
        else:
            if start_idx == -1:
                # worst winner
                total_return = payouts_20[-1]
            elif start_idx == 9:
                # median
                total_return = payouts_20[len(payouts_20)//2]
            else:
                total_return = sum(payouts_20[start_idx:start_idx+wins])
        profit = total_return - BANKROLL
        print(f"  {desc:40s} -> ${total_return:>8,.0f} return (${profit:>+8,.0f} P/L)")


asyncio.run(run())
