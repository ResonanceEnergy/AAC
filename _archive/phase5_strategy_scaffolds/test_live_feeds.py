"""Quick test for war_room_live_feeds integration."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

async def test():
    from strategies.war_room_engine import SPOT_PRICES, IndicatorState
    from strategies.war_room_live_feeds import apply_live_data_to_indicators, fetch_all_live_data

    print("=== BEFORE ===")
    btc_before = SPOT_PRICES["btc"]
    eth_before = SPOT_PRICES["eth"]
    print(f"  SPOT_PRICES btc={btc_before} eth={eth_before}")
    ind_before = IndicatorState()
    print(f"  IndicatorState btc={ind_before.btc_price}")
    print()

    print("=== FETCHING LIVE DATA ===")
    result = await fetch_all_live_data()
    print(f"  Summary: {result.summary()}")
    print()

    if result.errors:
        print("=== ERRORS (non-fatal) ===")
        for e in result.errors:
            print(f"  [!] {e}")
        print()

    print("=== RAW DATA ===")
    print(f"  CoinGecko: BTC={result.btc_price} ETH={result.eth_price} XRP={result.xrp_price}")
    print(f"  CoinGecko Global: MCap={result.total_market_cap_usd} BTC_dom={result.btc_dominance}")
    print(f"  UW: P/C={result.put_call_ratio} Tone={result.market_tone} Flow={result.options_flow_signals} DP={result.dark_pool_trades}")
    print(f"  MetaMask: MATIC={result.metamask_matic} USDC_Poly={result.metamask_usdc_polygon} ETH={result.metamask_eth}")
    print(f"  NDAX: {result.ndax_balances} net_cad={result.ndax_net_cad}")
    print()

    # Apply to indicators
    ind = apply_live_data_to_indicators(result)
    print("=== AFTER APPLY ===")
    print(f"  IndicatorState btc={ind.btc_price} news_severity={ind.news_severity}")
    print(f"  SPOT_PRICES btc={SPOT_PRICES['btc']} eth={SPOT_PRICES['eth']}")
    btc_changed = SPOT_PRICES["btc"] != btc_before
    eth_changed = SPOT_PRICES["eth"] != eth_before
    print(f"  btc_changed={btc_changed} eth_changed={eth_changed}")
    print()
    print("DONE")

if __name__ == "__main__":
    asyncio.run(test())
