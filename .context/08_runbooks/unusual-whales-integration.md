# Unusual Whales Integration

Purpose:
- Integrate Unusual Whales market-intelligence data into AAC.
- Support options flow, dark pool, congress trades, and related market context.

Source announcement:
- Unusual Whales announced an API/MCP offering for stock, options, Polymarket, and broader financial data.
- Article thesis: connect an AI assistant to live market data so it stops guessing.

What the article claims is available:
- Real-time options flow
- Sweeps and block trades
- Dark pool prints with NBBO context
- Congressional trading activity
- Public-company financial statements
- 13F and insider disclosures
- Technical indicators such as RSI, MACD, VWAP, Bollinger Bands
- Screening across IV rank, put/call ratio, short interest, and similar parameters
- Polymarket data access

Current AAC status:
- Existing client: `integrations/unusual_whales_client.py`
- Existing config support: `shared/config_loader.py`
- Env var: `UNUSUAL_WHALES_API_KEY`
- Validator: `tools/validate_unusual_whales.py`

How to enable:
1. Add `UNUSUAL_WHALES_API_KEY` to `.env`
2. Run `.venv\Scripts\python.exe tools/validate_unusual_whales.py`
3. If validation succeeds, wire the client into strategy or monitoring workflows

Recommended first usage inside AAC:
- Feed options-flow and dark-pool signals into doctrine and monitoring layers
- Use Congress trades and market-overview data for BigBrainIntelligence research workflows
- Keep execution logic separated from data ingestion; the Unusual Whales client is intelligence, not routing/execution

Open items:
- MCP-specific configuration is not implemented here because the article references a hosted MCP path, but not a stable server config contract inside this repo
- If you want direct MCP usage in addition to REST, add the vendor's official server configuration once auth and transport details are confirmed from their docs