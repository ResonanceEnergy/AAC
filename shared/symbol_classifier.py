"""
Symbol Classifier — Single Source of Truth
===========================================
Determines whether a symbol is crypto or equity and which exchange routes it.
All other modules import from here instead of maintaining their own sets.
"""

# ── Crypto base assets (traded on NDAX / Binance / CoinGecko) ──
CRYPTO_SYMBOLS = frozenset({
    "BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "LINK", "DOT", "DOGE",
    "MATIC", "UNI", "AAVE", "ATOM", "NEAR", "FTM", "ALGO", "LTC", "BCH",
    "SHIB", "APT", "ARB", "OP", "SUI", "SEI", "INJ", "TIA",
})

# ── Equity / ETF symbols (priced via IBKR) ──
EQUITY_SYMBOLS = frozenset({
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
    "ARCC", "PFF", "LQD", "EMB", "MAIN", "JNK", "KRE",
    "GLD", "SLV", "TLT", "HYG", "XLF", "XLE", "XLK",
})


def base_symbol(symbol: str) -> str:
    """Extract the base asset from 'BTC/USDT', 'AAPL/USD', or bare 'ETH'."""
    return symbol.split("/")[0].upper()


def is_crypto(symbol: str) -> bool:
    return base_symbol(symbol) in CRYPTO_SYMBOLS


def is_equity(symbol: str) -> bool:
    return base_symbol(symbol) in EQUITY_SYMBOLS


def asset_class(symbol: str) -> str:
    """Return 'crypto', 'equity', or 'unknown'."""
    b = base_symbol(symbol)
    if b in CRYPTO_SYMBOLS:
        return "crypto"
    if b in EQUITY_SYMBOLS:
        return "equity"
    return "unknown"


def route_exchange(symbol: str) -> str:
    """Return the canonical exchange for a symbol."""
    if is_crypto(symbol):
        return "ndax"
    return "ibkr"


def normalize_pair(symbol: str) -> str:
    """Normalize a bare symbol into a proper trading pair.

    Crypto  → BTC/USDT
    Equity  → AAPL/USD
    Already paired → unchanged
    """
    if "/" in symbol:
        return symbol.upper()
    b = symbol.upper()
    if b in CRYPTO_SYMBOLS:
        return f"{b}/USDT"
    if b in EQUITY_SYMBOLS:
        return f"{b}/USD"
    # Unknown — return bare upper-cased
    return b
