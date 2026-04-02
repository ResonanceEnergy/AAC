---
applyTo: "integrations/**,shared/data_sources.py"
---

# API Integrations — Context Guardrails

## Available Clients (USE THESE — don't build new ones)

| Client | File | Import |
|---|---|---|
| Unusual Whales | `integrations/unusual_whales_client.py` | `from integrations import get_unusual_whales_client` |
| UW Snapshot Service | `integrations/unusual_whales_service.py` | Cached snapshot with 300s TTL |
| CoinGecko | `shared/data_sources.py` | `CoinGeckoClient` class |
| Finnhub | `integrations/finnhub_client.py` | Check for existing client |
| yfinance | Direct import | `import yfinance as yf` |

## Known Issues

- **CoinGecko**: Pro key expired. Client auto-downgrades to free tier. 10 req/min limit.
- **Unusual Whales**: Connection works but field parsing broken. `strike_price`, `premium`, `put_call`, `sentiment` fields return $0/blank. The API schema changed field names — client needs field mapping update.
- **Polygon**: Free tier returns 403 on options snapshots. Use yfinance as options chain source.

## Rules

- Check `.env` for key existence before declaring an API broken
- Use `os.environ.get()` or `python-dotenv` for key loading
- ALL API clients must degrade gracefully when key is missing (return empty/default, don't crash)
- Rate limit awareness: CoinGecko free=10/min, UW varies, Finnhub=60/min
- aiohttp: use `ThreadedResolver()` + `TCPConnector(resolver=...)` to avoid c-ares DNS issues on Windows
