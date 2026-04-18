# Stage 01: Scan

> Pull live market data from 11 API feeds. Snapshot the 15-indicator state.

## Inputs

| Source | File/Location | Section/Scope | Why |
|---|---|---|---|
| API inventory | `../../CLAUDE.md` | Runtime Code table | Know which feeds exist |
| Feed config | `../../_config/indicators.md` | Indicators table | Know which fields to populate |
| Previous scan | `output/latest-scan.md` | Full file | Compare for staleness/delta |

## Process

1. Call `war_room_live_feeds.update_all_live_data()` (async) or `update_all_live_data_sync()` (sync)
2. The 11 feeds populate a `LiveFeedResult` dataclass:
   - **CoinGecko**: BTC, ETH, XRP prices + global market cap + BTC dominance + stablecoin depeg
   - **Unusual Whales**: put/call ratio, market tone, options flow, dark pool, congress trades
   - **MetaMask**: on-chain wallet balances (MATIC, USDC, ETH)
   - **NDAX**: exchange balances (CAD) — liquidated, expect zeros
   - **Finnhub**: SPY quote
   - **FRED**: gold, oil WTI, fed rate, DXY, HY spread
   - **Fear & Greed**: alternative.me crypto index
   - **NewsAPI**: headline count + severity score
   - **X/Twitter**: sentiment score (0=fear, 1=greed)
   - **IBKR**: net liquidation, positions, VIX, buying power, unrealized P&L
   - **Moomoo**: (via IBKR overlay or separate OpenD call)
3. Map `LiveFeedResult` fields into `IndicatorState` (the 15-indicator model)
4. Write snapshot to `output/latest-scan.md`
5. Append to `data/war_engine/indicator_snapshots.jsonl` (persistence)

## Outputs

| Artifact | Location | Format |
|---|---|---|
| Latest scan | `output/latest-scan.md` | Markdown table of all 15 indicators + raw feed data |
| Feed health | `output/feed-health.md` | Per-feed status (OK/stale/error), last success time |

## Error Handling

- Partial data is useful. If 3 of 11 feeds fail, still write the other 8.
- Log errors in `LiveFeedResult.errors` list and in `output/feed-health.md`.
- Mark stale feeds (no update in 15 min) explicitly.

## Runtime

- **Auto**: `WarRoomAutoEngine` runs this every 5 minutes via `live_feeds_full` task
- **Manual**: `python strategies/war_room_live_feeds.py` or import + call
- **Indicator snapshot**: persisted every 60 seconds to JSONL
