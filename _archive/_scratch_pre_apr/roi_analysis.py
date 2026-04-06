"""Analyze ROI potential of all Black Swan scanner opportunities."""
import asyncio
import sys

sys.path.insert(0, '.')

from strategies.polymarket_blackswan_scanner import PolymarketBlackSwanScanner


async def analyze():
    scanner = PolymarketBlackSwanScanner()
    try:
        opps = await scanner.scan()

        scored = []
        for o in opps:
            payout = 1.0 / max(o.market_price, 0.01)
            ev_per_dollar = o.thesis_probability * payout  # Expected return per $1
            # Liquidity premium: penalize illiquid
            liq_score = min(o.liquidity / 50000, 1.0) if o.liquidity > 0 else 0.1
            vol_score = min(o.volume_24h / 1000000, 1.0) if o.volume_24h > 0 else 0.1
            # Composite: EV * sqrt(liquidity) * cube-root(volume)
            composite = ev_per_dollar * (liq_score ** 0.5) * (vol_score ** 0.3)

            scored.append({
                'cat': o.category,
                'outcome': o.outcome,
                'price': o.market_price,
                'thesis': o.thesis_probability,
                'edge': o.edge,
                'ev_per_dollar': ev_per_dollar,
                'payout': payout,
                'kelly': o.kelly_fraction,
                'vol': o.volume_24h,
                'liq': o.liquidity,
                'composite': composite,
                'q': o.market_question[:80],
            })

        scored.sort(key=lambda x: x['composite'], reverse=True)

        hdr = "  {:>3} {:14} {:>8} {:>6} {:>7} {:>6} {:>6} {:>7} {:>6} {:>12} {:>10}"
        print("=" * 115)
        print("  HIGHEST ROI -- Ranked by EV x Liquidity x Volume")
        print("=" * 115)
        print(hdr.format("#", "Category", "BET", "Price", "Thesis", "Edge", "EV/$", "Payout", "Kelly", "Volume", "Liq"))
        print("-" * 115)

        for i, s in enumerate(scored[:30], 1):
            bet = f"BUY {s['outcome']}"
            row = "  {:3} {:14} {:>8} {:6.4f} {:6.1%} {:6.1%} {:6.2f} {:5.1f}x {:5.1%} {:>11,.0f} {:>9,.0f}"
            print(row.format(
                i, s['cat'], bet, s['price'], s['thesis'],
                s['edge'], s['ev_per_dollar'], s['payout'], s['kelly'],
                s['vol'], s['liq'],
            ))
            print("      " + s['q'])

        print()
        print("=" * 115)
        print("  TIER BREAKDOWN")
        print("-" * 115)
        for label, lo, hi in [("TOP 10", 0, 10), ("TOP 30", 0, 30), ("ALL", 0, len(scored))]:
            subset = scored[lo:hi]
            n = len(subset)
            if n == 0:
                continue
            avg_ev = sum(s['ev_per_dollar'] for s in subset) / n
            avg_kelly = sum(s['kelly'] for s in subset) / n
            avg_price = sum(s['price'] for s in subset) / n
            total_kelly_spend = sum(s['kelly'] * 556.39 for s in subset)
            print("  {:8} | {:3} bets | Avg EV/$: {:.2f} | Avg Kelly: {:.1%} | Avg Price: {:.3f} | Kelly-sum spend: ${:,.0f}".format(
                label, n, avg_ev, avg_kelly, avg_price, total_kelly_spend))

        print()
        n = len(scored)
        print("  Balance: $556.39 USDC.e")
        print("  TOP 10 @ ~$55/bet  |  TOP 30 @ ~$18.50/bet  |  ALL {} @ ~${:.2f}/bet".format(n, 556.39 / max(n, 1)))

        # Show best 5 by PURE EV (highest expected return per dollar ignoring liquidity)
        print()
        print("=" * 115)
        print("  TOP 5 PURE EV (highest expected return per $1, ignoring liquidity)")
        print("-" * 115)
        by_ev = sorted(scored, key=lambda x: x['ev_per_dollar'], reverse=True)
        for i, s in enumerate(by_ev[:5], 1):
            bet = f"BUY {s['outcome']}"
            print("  {} | {} {:>8} @ {:.4f} | EV: ${:.2f} per $1 bet | {:.0f}x payout | Kelly: {:.1%}".format(
                i, s['cat'], bet, s['price'], s['ev_per_dollar'],
                s['payout'], s['kelly']))
            print("      " + s['q'])

        # Show best 5 by KELLY (optimal bet fraction)
        print()
        print("=" * 115)
        print("  TOP 5 KELLY (largest optimal bet fraction)")
        print("-" * 115)
        by_kelly = sorted(scored, key=lambda x: x['kelly'], reverse=True)
        for i, s in enumerate(by_kelly[:5], 1):
            kelly_bet = s['kelly'] * 556.39
            bet = f"BUY {s['outcome']}"
            print("  {} | {} {:>8} @ {:.4f} | Kelly: {:.1%} = ${:.2f} bet | EV: ${:.2f}/$ | Liq: ${:,.0f}".format(
                i, s['cat'], bet, s['price'], s['kelly'], kelly_bet,
                s['ev_per_dollar'], s['liq']))
            print("      " + s['q'])

        # Show best 5 LIQUID (highest volume > $5M with decent edge)
        print()
        print("=" * 115)
        print("  TOP 5 LIQUID (deepest pools you can actually fill)")
        print("-" * 115)
        by_liq = sorted([s for s in scored if s['vol'] > 500000], key=lambda x: x['vol'], reverse=True)
        for i, s in enumerate(by_liq[:5], 1):
            bet = f"BUY {s['outcome']}"
            print("  {} | {} {:>8} @ {:.4f} | Vol: ${:,.0f} | Liq: ${:,.0f} | EV: ${:.2f}/$ | Kelly: {:.1%}".format(
                i, s['cat'], bet, s['price'], s['vol'], s['liq'],
                s['ev_per_dollar'], s['kelly']))
            print("      " + s['q'])

        print()
        print("=" * 115)

    finally:
        await scanner.close()


if __name__ == "__main__":
    asyncio.run(analyze())
