#!/usr/bin/env python3
"""Debug PlanktonXD signal generation."""
import asyncio

from shared.audit_logger import AuditLogger
from shared.communication_framework import CommunicationFramework
from strategies.planktonxd_prediction_harvester import create_planktonxd_strategy


async def debug():
    comm = CommunicationFramework()
    audit = AuditLogger()
    h = create_planktonxd_strategy(comm, audit, bankroll=50.0)
    markets = await h.fetch_polymarket_markets(limit=300)
    signals = h.scan_for_opportunities(markets)
    print(f"Signals: {len(signals)}")
    for m, bt, outcome, edge in signals[:5]:
        print(f"  Market: {m.question[:60]}")
        print(f"  Outcome: '{outcome}', Edge: {edge:.4f}, BetType: {bt}")
        print(f"  Prices: {m.prices}")
        yt = m.yes_token_id[:20] if m.yes_token_id else "EMPTY"
        nt = m.no_token_id[:20] if m.no_token_id else "EMPTY"
        print(f"  yes_token: {yt}, no_token: {nt}")
        cost = h.calculate_bet_size(bt, edge, m)
        entry_price = m.prices.get(outcome, 0.0)
        print(f"  Bet size: {cost}, Entry price: {entry_price}")
        min_bet = min(h.MIN_BET_USD, h.bankroll * 0.04)
        print(f"  Min bet: {min_bet}, bankroll: {h.bankroll}")
        # Also try place_bet manually
        bet = h.place_bet(m, bt, outcome, edge)
        print(f"  place_bet result: {bet}")
        print()

asyncio.run(debug())
