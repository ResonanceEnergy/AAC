from __future__ import annotations

"""Crypto Council Division — autonomous crypto market scanner.

Periodically scrapes CoinGecko for price, trending, and global data,
analyzes the market, and publishes INTEL_UPDATE signals.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from divisions.division_protocol import DivisionProtocol, Signal, SignalType

from councils.crypto.models import CouncilEntry
from councils.crypto.pipeline import run_crypto_council
from councils.crypto.scraper import DEFAULT_COINS

_log = structlog.get_logger(__name__)

_SEEN_DB = Path(__file__).parent / "output" / "crypto_seen.json"


class CryptoCouncilDivision(DivisionProtocol):
    """Autonomous crypto intelligence scraper.

    Each scan cycle:
    1. Fetches prices for a coin watchlist via CoinGecko
    2. Fetches trending coins
    3. Analyses gainers/losers/sentiment
    4. Publishes INTEL_UPDATE signals with crypto market insights
    """

    def __init__(
        self,
        coin_ids: list[str] | None = None,
        vs_currency: str = "usd",
    ) -> None:
        super().__init__(division_name="crypto_council")
        self._coin_ids = coin_ids or list(DEFAULT_COINS)
        self._vs_currency = vs_currency
        self._total_scans: int = 0
        self._last_entry: CouncilEntry | None = None
        self._seen_coins: set[str] = set()
        self._load_seen()

    def _load_seen(self) -> None:
        if _SEEN_DB.exists():
            try:
                data = json.loads(_SEEN_DB.read_text(encoding="utf-8"))
                self._seen_coins = set(data.get("seen_ids", []))
            except (json.JSONDecodeError, KeyError):
                self._seen_coins = set()

    def _save_seen(self) -> None:
        _SEEN_DB.parent.mkdir(parents=True, exist_ok=True)
        trimmed = sorted(self._seen_coins)[-500:]
        _SEEN_DB.write_text(
            json.dumps({
                "seen_ids": trimmed,
                "updated": datetime.now(timezone.utc).isoformat(),
            }),
            encoding="utf-8",
        )

    async def scan(self) -> list[Signal]:
        """Scrape CoinGecko, analyze, return intel signals."""
        signals: list[Signal] = []

        try:
            entry = await run_crypto_council(
                coin_ids=self._coin_ids,
                vs_currency=self._vs_currency,
            )
        except Exception as exc:
            _log.warning("crypto_council.scan_failed", error=str(exc))
            return []

        if not entry:
            return []

        self._total_scans += 1
        self._last_entry = entry
        insights = entry.insights

        # Track coins seen
        new_coins = {c.coin_id for c in entry.coins} - self._seen_coins
        self._seen_coins.update(c.coin_id for c in entry.coins)
        self._save_seen()

        # Primary intel signal
        trust_overall = insights.trust_score.get("overall", 0.7)
        signals.append(
            Signal(
                signal_type=SignalType.INTEL_UPDATE,
                source_division=self.division_name,
                timestamp=datetime.now(timezone.utc),
                data={
                    "type": "crypto_scan",
                    "total_coins": len(entry.coins),
                    "trending_count": len(entry.trending_coins),
                    "btc_price": insights.btc_price,
                    "eth_price": insights.eth_price,
                    "sentiment": insights.sentiment,
                    "new_coins": len(new_coins),
                    "total_scans": self._total_scans,
                    "gainers": insights.gainers[:3],
                    "losers": insights.losers[:3],
                    "summary": insights.summary,
                    "report_path": entry.markdown_path,
                    "trust_score": insights.trust_score,
                },
                confidence=trust_overall,
                urgency=0,
            ),
        )

        # Alert on big movers (>10% change either way)
        big_movers = [
            c for c in entry.coins if abs(c.change_24h) > 10
        ]
        if big_movers:
            signals.append(
                Signal(
                    signal_type=SignalType.INTEL_UPDATE,
                    source_division=self.division_name,
                    timestamp=datetime.now(timezone.utc),
                    data={
                        "type": "crypto_big_movers",
                        "movers": [
                            {
                                "coin": c.coin_id,
                                "price": c.price,
                                "change_24h": round(c.change_24h, 2),
                            }
                            for c in big_movers
                        ],
                    },
                    confidence=min(trust_overall + 0.1, 1.0),
                    urgency=1,
                ),
            )

        _log.info(
            "crypto_council.scan_complete",
            signals=len(signals),
            coins=len(entry.coins),
            sentiment=insights.sentiment,
        )
        return signals

    def report(self) -> dict[str, Any]:
        """Return current division state."""
        base = super().report()
        base.update({
            "total_scans": self._total_scans,
            "watchlist_size": len(self._coin_ids),
            "coins_seen": len(self._seen_coins),
            "vs_currency": self._vs_currency,
            "last_btc": self._last_entry.insights.btc_price if self._last_entry else None,
            "last_sentiment": self._last_entry.insights.sentiment if self._last_entry else None,
        })
        return base
