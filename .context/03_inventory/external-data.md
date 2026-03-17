# External Data Inventory

Active or planned external data sources relevant to AAC:

- CoinGecko: crypto pricing and market data
- Finnhub: quotes, earnings, news sentiment
- Polygon: equities/options market data
- WallStreetOdds: odds and sentiment overlays
- Unusual Whales: options flow, dark pool, congress trades, filings, market intelligence

Unusual Whales integration points:
- Client: `integrations/unusual_whales_client.py`
- Package export: `integrations.get_unusual_whales_client()`
- Env var: `UNUSUAL_WHALES_API_KEY`
- Validator: `tools/validate_unusual_whales.py`
- Runbook: `.context/08_runbooks/unusual-whales-integration.md`