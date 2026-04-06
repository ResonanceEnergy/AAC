"""Quick display of all Polymarket scenario bets."""
import json
from collections import defaultdict

with open("polymarket_scenario_bets.json", "r") as f:
    data = json.load(f)

print(f"Timestamp: {data['timestamp']}")
print(f"Bankroll: ${data['bankroll']}")
print(f"Max per bet: ${data['max_per_bet']}")
print(f"Total bets: {data['total_bets']}")
print(f"Active bets: {data['active_bets']}")
print()

by_scenario = defaultdict(list)
for b in data["bets"]:
    by_scenario[f"{b['scenario_code']} - {b['scenario_name']}"].append(b)

grand_total = 0
for sc, bets in sorted(by_scenario.items()):
    subtotal = sum(b["recommended_bet"] for b in bets)
    grand_total += subtotal
    print(f"=== {sc} ({len(bets)} bets, ${subtotal:.2f} recommended) ===")
    for i, b in enumerate(bets, 1):
        q = b["market_question"][:90]
        print(f"  {i:2d}. Mkt:{b['market_price']:.3f} Thesis:{b['thesis_probability']:.3f} "
              f"Edge:{b['edge']:.3f} ({b['edge_multiple']:.1f}x) "
              f"Kelly:{b['kelly_fraction']:.3f} Rec:${b['recommended_bet']:.2f}")
        print(f"      Q: {q}")
    print()

print(f"GRAND TOTAL recommended: ${grand_total:.2f}")
print(f"Scenarios: {len(by_scenario)} | Bets: {len(data['bets'])}")
