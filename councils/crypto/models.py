from __future__ import annotations

"""Crypto Council data models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CoinSnapshot:
    """Price snapshot of a single cryptocurrency."""

    coin_id: str
    symbol: str = ""
    price: float = 0.0
    volume_24h: float = 0.0
    change_24h: float = 0.0
    market_cap: float = 0.0


@dataclass
class TrendingCoin:
    """A coin from CoinGecko's trending endpoint."""

    coin_id: str
    name: str = ""
    symbol: str = ""
    market_cap_rank: int = 0
    thumb: str = ""
    score: int = 0


@dataclass
class GlobalData:
    """Global crypto market overview from CoinGecko."""

    total_market_cap_usd: float = 0.0
    total_volume_24h_usd: float = 0.0
    btc_dominance: float = 0.0
    eth_dominance: float = 0.0
    active_cryptocurrencies: int = 0
    market_cap_change_24h_pct: float = 0.0


@dataclass
class CryptoInsight:
    """Aggregated analysis from a crypto scan cycle."""

    top_coins: list[dict[str, Any]] = field(default_factory=list)
    trending: list[dict[str, Any]] = field(default_factory=list)
    global_data: GlobalData = field(default_factory=GlobalData)
    gainers: list[dict[str, Any]] = field(default_factory=list)
    losers: list[dict[str, Any]] = field(default_factory=list)
    sentiment: str = "neutral"
    summary: str = ""
    btc_price: float = 0.0
    eth_price: float = 0.0
    trust_score: dict[str, Any] = field(default_factory=dict)


@dataclass
class CouncilEntry:
    """Result of a single Crypto Council scan cycle."""

    coins: list[CoinSnapshot]
    trending_coins: list[TrendingCoin]
    insights: CryptoInsight
    markdown_path: str = ""
    processed_at: str = ""
