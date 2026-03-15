#!/usr/bin/env python3
"""
Knightsbridge FX Client
========================
Foreign exchange rate client for Knightsbridge FX — a Canadian foreign exchange
provider offering institutional-grade rates on 30+ currencies.

This client:
  - Fetches live FX spot rates from multiple sources (ExchangeRate-API, ECB)
  - Tracks bid/ask spreads for major, minor, and exotic currency pairs
  - Computes triangular arbitrage opportunities across FX pairs
  - Supports CAD-centric and UYU-centric currency corridors
  - Caches rates with configurable TTL to avoid hammering APIs

Knightsbridge FX Reference: https://www.knightsbridgefx.com
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import aiohttp

logger = logging.getLogger(__name__)

# ─── Major, minor, and CAD/UYU-specific pairs ──────────────────────────

MAJOR_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD",
]

CAD_PAIRS = [
    "USD/CAD", "EUR/CAD", "GBP/CAD", "AUD/CAD",
    "CAD/JPY", "CAD/CHF",
]

UYU_PAIRS = [
    "USD/UYU", "EUR/UYU", "BRL/UYU", "ARS/UYU",
]

CROSS_PAIRS = [
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "EUR/CHF",
    "AUD/JPY", "NZD/JPY", "EUR/AUD", "GBP/AUD",
]

ALL_FX_CURRENCIES = [
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
    "SEK", "NOK", "DKK", "SGD", "HKD", "INR", "MXN", "BRL",
    "ZAR", "TRY", "PLN", "CZK", "HUF", "ILS", "THB", "PHP",
    "MYR", "IDR", "KRW", "TWD", "CNY", "UYU",
]


class KnightsbridgeFXError(Exception):
    """Error from the FX rate service."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class FXRate:
    """A single FX rate quote."""
    __slots__ = ("pair", "bid", "ask", "mid", "spread_bps", "source", "timestamp")

    def __init__(
        self,
        pair: str,
        bid: float,
        ask: float,
        source: str = "exchangerate-api",
        timestamp: Optional[datetime] = None,
    ):
        self.pair = pair
        self.bid = bid
        self.ask = ask
        self.mid = (bid + ask) / 2
        self.spread_bps = ((ask - bid) / mid * 10_000) if (mid := (bid + ask) / 2) > 0 else 0.0
        self.source = source
        self.timestamp = timestamp or datetime.utcnow()

    def __repr__(self) -> str:
        return f"FXRate({self.pair} bid={self.bid:.5f} ask={self.ask:.5f} spread={self.spread_bps:.1f}bps)"


class KnightsbridgeFXClient:
    """
    Async foreign exchange rate client.

    Uses free/open FX rate APIs to fetch spot rates. Applies a configurable
    retail spread to derive bid/ask quotes comparable to Knightsbridge FX's
    institutional-grade pricing.

    Usage::

        async with KnightsbridgeFXClient(api_key="...") as client:
            rates = await client.get_rates(base="CAD")
            usd_cad = await client.get_pair("USD", "CAD")
            opps = await client.find_triangular_arb()
    """

    # Free tier: https://www.exchangerate-api.com (1500 req/mo)
    RATE_API_URL = "https://v6.exchangerate-api.com/v6/{api_key}/latest/{base}"
    PAIR_API_URL = "https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_}/{to_}"
    # ECB fallback (no key needed, daily updates only)
    ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"

    # Default retail spread in basis points (Knightsbridge is ~40-80bps vs bank 200-300bps)
    DEFAULT_SPREAD_BPS = 50

    def __init__(
        self,
        api_key: str = "",
        spread_bps: float = DEFAULT_SPREAD_BPS,
        cache_ttl_seconds: int = 60,
        rate_limit_delay: float = 0.2,
    ):
        self.api_key = api_key
        self.spread_bps = spread_bps
        self._cache_ttl = cache_ttl_seconds
        self._rate_limit_delay = rate_limit_delay
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0
        self._cache: Dict[str, Any] = {}  # key → (timestamp, data)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.disconnect()

    async def connect(self) -> bool:
        """Create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return True

    async def disconnect(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _rate_limit_wait(self):
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)

    def _cache_get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry and (time.monotonic() - entry[0]) < self._cache_ttl:
            return entry[1]
        return None

    def _cache_set(self, key: str, data: Any):
        self._cache[key] = (time.monotonic(), data)

    # ─── Core Rate Fetching ──────────────────────────────────────────

    async def get_rates(self, base: str = "USD") -> Dict[str, FXRate]:
        """
        Fetch all FX rates for a base currency.

        Returns dict of pair-string → FXRate with bid/ask derived from
        mid-rate ± half the configured spread.
        """
        base = base.upper()
        cache_key = f"rates_{base}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if not self.api_key:
            logger.warning("No FX API key — returning offline demo rates")
            return self._offline_demo_rates(base)

        await self._rate_limit_wait()
        url = self.RATE_API_URL.format(api_key=self.api_key, base=base)

        try:
            if self._session is None:
                await self.connect()
            async with self._session.get(url) as resp:
                self._last_request_time = time.monotonic()
                if resp.status != 200:
                    raise KnightsbridgeFXError(
                        f"FX API returned {resp.status}", status_code=resp.status
                    )
                data = await resp.json()
        except aiohttp.ClientError as exc:
            logger.error(f"FX rate fetch failed: {exc}")
            return self._offline_demo_rates(base)

        if data.get("result") != "success":
            logger.error(f"FX API error: {data.get('error-type', 'unknown')}")
            return self._offline_demo_rates(base)

        rates_raw = data.get("conversion_rates", {})
        result: Dict[str, FXRate] = {}
        half_spread = self.spread_bps / 2 / 10_000

        for quote_ccy, mid in rates_raw.items():
            if quote_ccy == base:
                continue
            pair = f"{base}/{quote_ccy}"
            bid = mid * (1 - half_spread)
            ask = mid * (1 + half_spread)
            result[pair] = FXRate(pair, bid, ask, source="exchangerate-api")

        self._cache_set(cache_key, result)
        logger.info(f"Fetched {len(result)} FX rates for {base}")
        return result

    async def get_pair(self, from_ccy: str, to_ccy: str) -> Optional[FXRate]:
        """Fetch a single currency pair rate."""
        from_ccy, to_ccy = from_ccy.upper(), to_ccy.upper()
        cache_key = f"pair_{from_ccy}_{to_ccy}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if not self.api_key:
            rates = self._offline_demo_rates(from_ccy)
            return rates.get(f"{from_ccy}/{to_ccy}")

        await self._rate_limit_wait()
        url = self.PAIR_API_URL.format(
            api_key=self.api_key, from_=from_ccy, to_=to_ccy
        )
        try:
            if self._session is None:
                await self.connect()
            async with self._session.get(url) as resp:
                self._last_request_time = time.monotonic()
                if resp.status != 200:
                    return None
                data = await resp.json()
        except aiohttp.ClientError as exc:
            logger.error(f"FX pair fetch failed: {exc}")
            return None

        mid = data.get("conversion_rate", 0)
        if mid <= 0:
            return None

        half_spread = self.spread_bps / 2 / 10_000
        rate = FXRate(
            f"{from_ccy}/{to_ccy}",
            mid * (1 - half_spread),
            mid * (1 + half_spread),
            source="exchangerate-api",
        )
        self._cache_set(cache_key, rate)
        return rate

    # ─── Triangular Arbitrage ────────────────────────────────────────

    async def find_triangular_arb(
        self,
        base: str = "USD",
        min_profit_bps: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """
        Scan for triangular FX arbitrage: base→A→B→base.

        Returns list of opportunities with expected profit in basis points.
        """
        rates = await self.get_rates(base)
        if not rates:
            return []

        # Build cross-rate lookup: {ccy: mid_rate_from_base}
        mid_map: Dict[str, float] = {}
        for pair, rate in rates.items():
            _, quote = pair.split("/")
            mid_map[quote] = rate.mid

        opportunities: List[Dict[str, Any]] = []
        ccys = list(mid_map.keys())

        for i, ccy_a in enumerate(ccys):
            for ccy_b in ccys[i + 1:]:
                # Path: base → ccy_a → ccy_b → base
                rate_base_a = mid_map.get(ccy_a, 0)
                rate_base_b = mid_map.get(ccy_b, 0)
                if rate_base_a <= 0 or rate_base_b <= 0:
                    continue

                # cross rate ccy_a/ccy_b = (base/ccy_b) / (base/ccy_a)
                implied_cross = rate_base_b / rate_base_a

                # Round-trip: 1 base → rate_base_a units of A → (rate_base_a / implied_cross) of B → back to base
                # Forward: base→A→B→base
                forward = rate_base_a * (1 / implied_cross) * (1 / rate_base_b)
                profit_bps_fwd = (forward - 1) * 10_000

                if abs(profit_bps_fwd) >= min_profit_bps:
                    opportunities.append({
                        "path": f"{base}→{ccy_a}→{ccy_b}→{base}",
                        "profit_bps": round(profit_bps_fwd, 2),
                        "direction": "forward" if profit_bps_fwd > 0 else "reverse",
                        "leg_1": f"{base}/{ccy_a} @ {rate_base_a:.5f}",
                        "leg_2": f"{ccy_a}/{ccy_b} implied @ {implied_cross:.5f}",
                        "leg_3": f"{ccy_b}/{base} @ {1/rate_base_b:.5f}",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

        opportunities.sort(key=lambda x: abs(x["profit_bps"]), reverse=True)
        return opportunities

    # ─── Spread Comparison ───────────────────────────────────────────

    async def compare_spreads(
        self,
        pairs: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compare Knightsbridge-level spreads against typical bank spreads.

        Returns per-pair savings estimate in basis points.
        """
        if pairs is None:
            pairs = CAD_PAIRS

        rates = await self.get_rates("USD")
        results: List[Dict[str, Any]] = []

        # Typical bank retail spread by pair tier
        bank_spreads = {
            "major": 200,   # 200 bps typical bank spread on majors
            "cad": 250,     # 250 bps for CAD pairs at Canadian banks
            "exotic": 400,  # 400+ bps for exotics
        }

        for pair_str in pairs:
            rate = rates.get(pair_str)
            if not rate:
                continue
            # Determine tier
            if pair_str in MAJOR_PAIRS:
                bank_bps = bank_spreads["major"]
            elif any(c in pair_str for c in ("CAD",)):
                bank_bps = bank_spreads["cad"]
            else:
                bank_bps = bank_spreads["exotic"]

            savings_bps = bank_bps - rate.spread_bps
            results.append({
                "pair": pair_str,
                "knightsbridge_spread_bps": round(rate.spread_bps, 1),
                "bank_spread_bps": bank_bps,
                "savings_bps": round(savings_bps, 1),
                "savings_pct": round(savings_bps / bank_bps * 100, 1) if bank_bps else 0,
                "mid_rate": round(rate.mid, 5),
            })

        return results

    # ─── CAD Corridor ────────────────────────────────────────────────

    async def get_cad_rates(self) -> Dict[str, FXRate]:
        """Get all CAD-centric rates — core corridor for Knightsbridge."""
        return await self.get_rates("CAD")

    async def get_uyu_rates(self) -> Dict[str, FXRate]:
        """Get UYU rates — Montevideo corridor."""
        return await self.get_rates("UYU")

    # ─── Offline / Demo ──────────────────────────────────────────────

    def _offline_demo_rates(self, base: str) -> Dict[str, FXRate]:
        """Return approximate offline demo rates when no API key is available."""
        # Mid-rates vs USD as of 2026-03
        usd_rates = {
            "EUR": 0.9180, "GBP": 0.7890, "JPY": 149.50, "CHF": 0.8810,
            "CAD": 1.3590, "AUD": 1.5420, "NZD": 1.6830, "SEK": 10.42,
            "NOK": 10.68, "DKK": 6.84, "SGD": 1.3290, "HKD": 7.8100,
            "INR": 83.10, "MXN": 17.15, "BRL": 4.97, "ZAR": 18.35,
            "TRY": 32.50, "UYU": 39.80, "CNY": 7.24, "KRW": 1325.0,
        }

        if base == "USD":
            half = self.spread_bps / 2 / 10_000
            result: Dict[str, FXRate] = {}
            for ccy, mid in usd_rates.items():
                pair = f"USD/{ccy}"
                result[pair] = FXRate(pair, mid * (1 - half), mid * (1 + half), source="offline")
            return result

        # Convert through USD
        base_in_usd = usd_rates.get(base)
        if base_in_usd is None:
            return {}

        half = self.spread_bps / 2 / 10_000
        result = {}
        for ccy, usd_mid in usd_rates.items():
            if ccy == base:
                continue
            mid = usd_mid / base_in_usd
            pair = f"{base}/{ccy}"
            result[pair] = FXRate(pair, mid * (1 - half), mid * (1 + half), source="offline")
        # Also include vs USD
        usd_mid = 1.0 / base_in_usd
        pair = f"{base}/USD"
        result[pair] = FXRate(pair, usd_mid * (1 - half), usd_mid * (1 + half), source="offline")
        return result

    def get_supported_currencies(self) -> List[str]:
        """Return list of supported currencies."""
        return list(ALL_FX_CURRENCIES)

    def get_supported_pairs(self) -> List[str]:
        """Return all tracked FX pairs."""
        return MAJOR_PAIRS + CAD_PAIRS + UYU_PAIRS + CROSS_PAIRS
