from __future__ import annotations

import pytest

from shared import symbol_classifier as sc
from shared.symbol_classifier import (
    CRYPTO_SYMBOLS,
    EQUITY_SYMBOLS,
    asset_class,
    base_symbol,
    is_crypto,
    is_equity,
    normalize_pair,
    route_exchange,
)


class TestUniverses:
    def test_crypto_is_frozenset(self):
        assert isinstance(CRYPTO_SYMBOLS, frozenset)
        assert "BTC" in CRYPTO_SYMBOLS
        assert "ETH" in CRYPTO_SYMBOLS

    def test_equity_is_frozenset(self):
        assert isinstance(EQUITY_SYMBOLS, frozenset)
        assert "SPY" in EQUITY_SYMBOLS
        assert "AAPL" in EQUITY_SYMBOLS

    def test_no_overlap_between_universes(self):
        assert CRYPTO_SYMBOLS.isdisjoint(EQUITY_SYMBOLS)

    def test_crypto_known_members(self):
        for s in ("BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "LINK", "DOGE", "SHIB", "TIA"):
            assert s in CRYPTO_SYMBOLS

    def test_equity_known_members(self):
        for s in ("SPY", "QQQ", "IWM", "ARCC", "PFF", "HYG", "JNK", "TLT", "GLD"):
            assert s in EQUITY_SYMBOLS


class TestBaseSymbol:
    def test_strips_pair_separator(self):
        assert base_symbol("BTC/USDT") == "BTC"
        assert base_symbol("AAPL/USD") == "AAPL"

    def test_uppercases_bare(self):
        assert base_symbol("btc") == "BTC"
        assert base_symbol("aapl") == "AAPL"

    def test_uppercases_paired(self):
        assert base_symbol("eth/usdt") == "ETH"

    def test_unknown_passthrough(self):
        assert base_symbol("foo") == "FOO"
        assert base_symbol("foo/bar") == "FOO"


class TestIsCrypto:
    def test_known_crypto(self):
        assert is_crypto("BTC") is True
        assert is_crypto("ETH/USDT") is True
        assert is_crypto("doge") is True

    def test_equity_returns_false(self):
        assert is_crypto("SPY") is False
        assert is_crypto("AAPL/USD") is False

    def test_unknown_returns_false(self):
        assert is_crypto("FOO") is False


class TestIsEquity:
    def test_known_equity(self):
        assert is_equity("SPY") is True
        assert is_equity("aapl") is True
        assert is_equity("HYG/USD") is True

    def test_crypto_returns_false(self):
        assert is_equity("BTC") is False
        assert is_equity("eth/usdt") is False

    def test_unknown_returns_false(self):
        assert is_equity("FOO") is False


class TestAssetClass:
    def test_crypto(self):
        assert asset_class("BTC") == "crypto"
        assert asset_class("eth/usdt") == "crypto"

    def test_equity(self):
        assert asset_class("SPY") == "equity"
        assert asset_class("aapl/usd") == "equity"

    def test_unknown(self):
        assert asset_class("FOO") == "unknown"
        assert asset_class("zzz/usd") == "unknown"


class TestRouteExchange:
    def test_crypto_routes_ndax(self):
        assert route_exchange("BTC") == "ndax"
        assert route_exchange("eth/usdt") == "ndax"

    def test_equity_routes_ibkr(self):
        assert route_exchange("SPY") == "ibkr"
        assert route_exchange("aapl") == "ibkr"

    def test_unknown_routes_ibkr_default(self):
        # Default fall-through is ibkr (only crypto routes to ndax)
        assert route_exchange("FOO") == "ibkr"


class TestNormalizePair:
    def test_already_paired_uppercased(self):
        assert normalize_pair("btc/usdt") == "BTC/USDT"
        assert normalize_pair("AAPL/USD") == "AAPL/USD"

    def test_crypto_gets_usdt_quote(self):
        assert normalize_pair("BTC") == "BTC/USDT"
        assert normalize_pair("eth") == "ETH/USDT"

    def test_equity_gets_usd_quote(self):
        assert normalize_pair("SPY") == "SPY/USD"
        assert normalize_pair("aapl") == "AAPL/USD"

    def test_unknown_returns_bare_upper(self):
        assert normalize_pair("foo") == "FOO"
        assert normalize_pair("ZZZ") == "ZZZ"
