# Account Inventory

> Canonical source for account details. Do not duplicate.
> Balances update at runtime from `config/account_balances.py`.

## Active Trading Accounts

| Account | Platform | Currency | Status | Constraints |
|---|---|---|---|---|
| U24346218 | IBKR | CAD | LIVE | 15 positions (5 calls + 10 puts), Net liq CAD $20,079.57, Cash CAD $2,700.83 |
| FUTUCA | Moomoo | USD | DEGRADED | OpenD running but API port 11111 NOT listening. Dummy credentials in OpenD.xml. Last known $365.15 USD. |
| TFSA | WealthSimple | CAD | LIVE | Tax-sheltered, no margin, no API (manual trades) |

## Inactive/Liquidated

| Account | Platform | Status | Notes |
|---|---|---|---|
| NDAX | NDAX | LIQUIDATED | Sold all XRP+ETH, $4,492.04 CAD withdrawn |

## Savings/Cash

| Account | Platform | Purpose |
|---|---|---|
| EQ Bank | EQ Bank | High-interest savings, dry powder |

## Account Rules

- **IBKR**: Can trade options, futures. Port 7497 (live). CAD-denominated.
  Verify `managedAccounts()[0]` prefix: `U` = live, `DU` = paper.
  Live-verified Apr 9: 15 positions (SLV/TSLA/XLE calls + OBDC/BKLN/HYG/LQD/EMB/XLF puts + 4 Apr 17 expiring).
- **Moomoo**: Use `OpenUSTradeContext`. Must pass trade PIN for orders.
  FUTUCA market (not FUTUINC). Currency param required for accinfo_query.
  **Status:** OpenD process running but API port 11111 not listening — dummy credentials in OpenD.xml need real account details + restart.
- **WealthSimple**: No API. All trades placed manually by human.
  TFSA rules: no margin, no short selling, contribution room limited.
  SnapTrade connector exists but NOT CONFIGURED (needs dashboard.snaptrade.com setup).
- **NDAX**: Connector exists (`ccxt`, login+password+uid) but no positions.
