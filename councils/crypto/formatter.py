from __future__ import annotations

"""Crypto Council formatter — markdown report from market insights."""

from councils.crypto.models import CoinSnapshot, CryptoInsight


def format_to_markdown(coins: list[CoinSnapshot], insights: CryptoInsight) -> str:
    """Render a Markdown report of the crypto council scan."""
    lines: list[str] = []
    lines.append("# Crypto Council Report\n")

    # Global overview
    gd = insights.global_data
    if gd and gd.total_market_cap_usd:
        lines.append("## Market Overview\n")
        lines.append(f"- **Total Market Cap:** ${gd.total_market_cap_usd / 1e9:,.1f}B")
        lines.append(f"- **24h Volume:** ${gd.total_volume_24h_usd / 1e9:,.1f}B")
        lines.append(f"- **BTC Dominance:** {gd.btc_dominance:.1f}%")
        lines.append(f"- **ETH Dominance:** {gd.eth_dominance:.1f}%")
        lines.append(f"- **Active Coins:** {gd.active_cryptocurrencies:,}")
        lines.append(f"- **Market Cap Change (24h):** {gd.market_cap_change_24h_pct:+.2f}%")
        lines.append("")

    # Sentiment
    lines.append(f"**Sentiment:** {insights.sentiment}\n")
    lines.append(f"> {insights.summary}\n")

    # Gainers
    if insights.gainers:
        lines.append("## Top Gainers (24h)\n")
        lines.append("| Coin | Price | Change |")
        lines.append("|------|------:|-------:|")
        for g in insights.gainers:
            lines.append(f"| {g['coin']} | ${g['price']:,.2f} | {g['change_24h']:+.1f}% |")
        lines.append("")

    # Losers
    if insights.losers:
        lines.append("## Top Losers (24h)\n")
        lines.append("| Coin | Price | Change |")
        lines.append("|------|------:|-------:|")
        for lo in insights.losers:
            lines.append(f"| {lo['coin']} | ${lo['price']:,.2f} | {lo['change_24h']:+.1f}% |")
        lines.append("")

    # Trending
    if insights.trending:
        lines.append("## Trending Coins\n")
        lines.append("| Coin | Symbol | Rank |")
        lines.append("|------|--------|-----:|")
        for t in insights.trending:
            rank = t.get("rank", "—")
            lines.append(f"| {t['name']} | {t['symbol']} | {rank} |")
        lines.append("")

    # Top coins by volume
    if insights.top_coins:
        lines.append("## Top by Volume (24h)\n")
        lines.append("| Coin | Price | Volume (24h) | Change |")
        lines.append("|------|------:|-------------:|-------:|")
        for c in insights.top_coins:
            lines.append(
                f"| {c['coin']} | ${c['price']:,.2f} | ${c['volume_24h'] / 1e6:,.1f}M | {c['change_24h']:+.1f}% |"
            )
        lines.append("")

    # All coins table
    if coins:
        lines.append("## All Scanned Coins\n")
        lines.append("| Coin | Symbol | Price | Vol (24h) | Change | Mkt Cap |")
        lines.append("|------|--------|------:|----------:|-------:|--------:|")
        for c in sorted(coins, key=lambda x: x.market_cap, reverse=True)[:30]:
            lines.append(
                f"| {c.coin_id} | {c.symbol.upper()} | ${c.price:,.2f} "
                f"| ${c.volume_24h / 1e6:,.1f}M | {c.change_24h:+.1f}% "
                f"| ${c.market_cap / 1e9:,.2f}B |"
            )
        lines.append("")

    return "\n".join(lines)
