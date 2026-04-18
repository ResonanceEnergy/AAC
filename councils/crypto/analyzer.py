from __future__ import annotations

"""Crypto Council analyzer — extract insights from scraped market data."""

from typing import Any

import structlog

from councils.crypto.models import CoinSnapshot, CryptoInsight, GlobalData, TrendingCoin

_log = structlog.get_logger(__name__)


def _compute_crypto_trust(
    coins: list[CoinSnapshot],
    global_data: GlobalData,
) -> dict[str, float]:
    """Compute trust score for crypto market analysis."""
    import os
    from councils.trust import TrustScore, crypto_source_trust, evidence_score

    api_tier = "pro" if os.environ.get("COINGECKO_API_KEY") else "free"
    src = crypto_source_trust(
        coin_count=len(coins),
        has_global_data=global_data.total_market_cap_usd > 0,
        api_tier=api_tier,
    )
    ev = evidence_score(len(coins), target=50)

    ts = TrustScore(
        source_reliability=src,
        data_freshness=0.95,  # CoinGecko data is near-real-time
        evidence_volume=ev,
    )
    return ts.to_dict()


def analyze_crypto_market(
    coins: list[CoinSnapshot],
    trending: list[TrendingCoin],
    global_data: GlobalData,
) -> CryptoInsight:
    """Produce aggregated crypto market insights."""
    if not coins:
        return CryptoInsight(summary="No coin data to analyze.")

    # Top coins by volume
    by_vol = sorted(coins, key=lambda c: c.volume_24h, reverse=True)
    top_coins = [
        {"coin": c.coin_id, "price": c.price, "volume_24h": c.volume_24h, "change_24h": c.change_24h}
        for c in by_vol[:10]
    ]

    # Trending
    trending_list = [
        {"coin": t.coin_id, "name": t.name, "symbol": t.symbol, "rank": t.market_cap_rank}
        for t in trending[:10]
    ]

    # Gainers (top positive 24h change)
    gainers = sorted(
        [c for c in coins if c.change_24h > 0],
        key=lambda c: c.change_24h,
        reverse=True,
    )
    gainers_list = [
        {"coin": c.coin_id, "change_24h": round(c.change_24h, 2), "price": c.price}
        for c in gainers[:5]
    ]

    # Losers (biggest negative 24h change)
    losers = sorted(
        [c for c in coins if c.change_24h < 0],
        key=lambda c: c.change_24h,
    )
    losers_list = [
        {"coin": c.coin_id, "change_24h": round(c.change_24h, 2), "price": c.price}
        for c in losers[:5]
    ]

    # Sentiment
    up_count = sum(1 for c in coins if c.change_24h > 0)
    down_count = sum(1 for c in coins if c.change_24h < 0)
    avg_change = sum(c.change_24h for c in coins) / len(coins) if coins else 0

    if avg_change > 3:
        sentiment = "strongly_bullish"
    elif avg_change > 1:
        sentiment = "bullish"
    elif avg_change < -3:
        sentiment = "strongly_bearish"
    elif avg_change < -1:
        sentiment = "bearish"
    elif up_count > down_count * 1.5:
        sentiment = "lean_bullish"
    elif down_count > up_count * 1.5:
        sentiment = "lean_bearish"
    else:
        sentiment = "neutral"

    # BTC/ETH highlight
    btc = next((c for c in coins if c.coin_id == "bitcoin"), None)
    eth = next((c for c in coins if c.coin_id == "ethereum"), None)

    # Summary
    parts = [
        f"Scanned {len(coins)} coins.",
        f"BTC ${btc.price:,.0f} ({btc.change_24h:+.1f}%)" if btc else "",
        f"ETH ${eth.price:,.0f} ({eth.change_24h:+.1f}%)" if eth else "",
        f"Market cap ${global_data.total_market_cap_usd / 1e12:.2f}T" if global_data.total_market_cap_usd else "",
        f"({global_data.market_cap_change_24h_pct:+.1f}% 24h)" if global_data.market_cap_change_24h_pct else "",
        f"BTC dominance {global_data.btc_dominance:.1f}%." if global_data.btc_dominance else "",
        f"Sentiment: {sentiment}.",
        f"{up_count} up, {down_count} down.",
    ]
    summary = " ".join(p for p in parts if p)

    trust = _compute_crypto_trust(coins, global_data)

    return CryptoInsight(
        top_coins=top_coins,
        trending=trending_list,
        global_data=global_data,
        gainers=gainers_list,
        losers=losers_list,
        sentiment=sentiment,
        summary=summary,
        btc_price=btc.price if btc else 0,
        eth_price=eth.price if eth else 0,
        trust_score=trust,
    )
