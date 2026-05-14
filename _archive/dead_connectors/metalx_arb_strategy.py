#!/usr/bin/env python3
"""
Metal X ↔ CEX Arbitrage Strategy (Vector 2)
=============================================
Cross-exchange arbitrage between Metal X DEX and centralized exchanges.

Edge: Metal X has ZERO gas fees + ZERO trading fees on XBTC/XMD,
creating a persistent fee asymmetry against CEX maker/taker fees.

Pairs monitored:
  - XBTC/XMD vs BTC/USD on Coinbase, Kraken, IBKR
  - XETH/XMD vs ETH/USD on Coinbase, Kraken
  - XPR/XMD vs XPR/USDT on any CEX listing
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from shared.audit_logger import AuditLogger
from shared.communication import CommunicationFramework
from shared.strategy_framework import (
    BaseArbitrageStrategy,
    SignalType,
    StrategyConfig,
    TradingSignal,
)

logger = logging.getLogger(__name__)


class MetalXCEXArbitrageStrategy(BaseArbitrageStrategy):
    """
    Detects price spreads between Metal X DEX and centralized exchanges.

    Strategy logic:
      1. Poll ticker on Metal X (zero-fee side)
      2. Compare against CEX best bid/ask
      3. If spread > min_spread_bps, generate buy+sell signals
      4. Factor in XMD ↔ USD peg risk as additional spread cushion

    Config risk_envelope keys:
      - min_spread_bps: Minimum spread in basis points to trigger (default 15)
      - max_position_usd: Max notional per leg (default $5,000)
      - xmd_peg_buffer_bps: Extra buffer for XMD depeg risk (default 5)
      - max_open_arb_positions: Concurrent arb legs (default 3)
    """

    # Which CEXs to compare against, by priority
    CEX_PRIORITY = ["ibkr", "coinbase", "kraken", "ndax"]

    # Metal X pairs and their CEX equivalents
    PAIR_MAP = {
        "BTC/XMD": ["BTC/USD", "BTC/USDT"],
        "ETH/XMD": ["ETH/USD", "ETH/USDT"],
        "XPR/XMD": ["XPR/USDT"],
        "MTL/XMD": ["MTL/USDT"],
    }

    def __init__(
        self,
        config: StrategyConfig,
        communication: CommunicationFramework,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, communication, audit_logger)
        self.last_signal_time: Optional[datetime] = None
        self._risk = config.risk_envelope or {}
        self.min_spread_bps = self._risk.get("min_spread_bps", 15)
        self.max_position_usd = self._risk.get("max_position_usd", 5000)
        self.xmd_buffer_bps = self._risk.get("xmd_peg_buffer_bps", 5)
        self.max_open = self._risk.get("max_open_arb_positions", 3)
        self._open_arb_count = 0
        self._last_spreads: Dict[str, float] = {}

    async def _initialize_strategy(self):
        """Initialize Metal X arb state and subscribe to data feeds."""
        self._open_arb_count = 0
        self._last_spreads = {}
        self._venue_status = {venue: 'unknown' for venue in self.CEX_PRIORITY}
        self._pair_tracking = {pair: {'last_price': None, 'last_cex_price': None} for pair in self.PAIR_MAP}
        logger.info(
            f"MetalXCEXArb initialized — min spread {self.min_spread_bps} bps, "
            f"max position ${self.max_position_usd}, "
            f"tracking {len(self.PAIR_MAP)} pairs across {len(self.CEX_PRIORITY)} venues"
        )

    def _should_generate_signal(self) -> bool:
        """Generate signals if we haven't hit the open position cap."""
        if self._open_arb_count >= self.max_open:
            return False
        # Cooldown: at most 1 signal per 10 seconds
        if self.last_signal_time:
            if datetime.now() - self.last_signal_time < timedelta(seconds=10):
                return False
        return True

    async def _generate_signals(self) -> List[TradingSignal]:
        """
        Compare Metal X prices vs CEX prices.

        market_data is expected to contain:
          - metalx_tickers: {symbol: {bid, ask, last}}
          - cex_tickers: {exchange: {symbol: {bid, ask, last}}}
        """
        signals: List[TradingSignal] = []

        mx_tickers = self.market_data.get("metalx_tickers", {})
        cex_tickers = self.market_data.get("cex_tickers", {})

        if not mx_tickers or not cex_tickers:
            return signals

        for mx_pair, cex_equivalents in self.PAIR_MAP.items():
            mx_data = mx_tickers.get(mx_pair)
            if not mx_data:
                continue

            mx_ask = mx_data.get("ask", 0)
            mx_bid = mx_data.get("bid", 0)
            if mx_ask <= 0 or mx_bid <= 0:
                continue

            # Find best CEX price for the equivalent pair
            for cex_name in self.CEX_PRIORITY:
                cex_markets = cex_tickers.get(cex_name, {})
                for cex_pair in cex_equivalents:
                    cex_data = cex_markets.get(cex_pair)
                    if not cex_data:
                        continue

                    cex_bid = cex_data.get("bid", 0)
                    cex_ask = cex_data.get("ask", 0)
                    if cex_bid <= 0 or cex_ask <= 0:
                        continue

                    # Opportunity 1: Buy on Metal X (ask), sell on CEX (bid)
                    spread_buy_mx = (cex_bid - mx_ask) / mx_ask * 10000  # bps

                    # Opportunity 2: Buy on CEX (ask), sell on Metal X (bid)
                    spread_buy_cex = (mx_bid - cex_ask) / cex_ask * 10000

                    effective_min = self.min_spread_bps + self.xmd_buffer_bps

                    if spread_buy_mx > effective_min:
                        qty = min(
                            self.max_position_usd / mx_ask,
                            self.max_position_usd / cex_bid,
                        )
                        confidence = min(spread_buy_mx / 100, 1.0)

                        signals.append(
                            TradingSignal(
                                strategy_id=self.config.strategy_id,
                                signal_type=SignalType.LONG,
                                symbol=mx_pair,
                                quantity=qty,
                                price=mx_ask,
                                confidence=confidence,
                                metadata={
                                    "arb_type": "buy_metalx_sell_cex",
                                    "cex": cex_name,
                                    "cex_pair": cex_pair,
                                    "spread_bps": round(spread_buy_mx, 2),
                                    "mx_ask": mx_ask,
                                    "cex_bid": cex_bid,
                                    "zero_fee": mx_pair == "BTC/XMD",
                                },
                            )
                        )
                        self._last_spreads[mx_pair] = spread_buy_mx

                    elif spread_buy_cex > effective_min:
                        qty = min(
                            self.max_position_usd / cex_ask,
                            self.max_position_usd / mx_bid,
                        )
                        confidence = min(spread_buy_cex / 100, 1.0)

                        signals.append(
                            TradingSignal(
                                strategy_id=self.config.strategy_id,
                                signal_type=SignalType.LONG,
                                symbol=cex_pair,
                                quantity=qty,
                                price=cex_ask,
                                confidence=confidence,
                                metadata={
                                    "arb_type": "buy_cex_sell_metalx",
                                    "cex": cex_name,
                                    "mx_pair": mx_pair,
                                    "spread_bps": round(spread_buy_cex, 2),
                                    "cex_ask": cex_ask,
                                    "mx_bid": mx_bid,
                                },
                            )
                        )
                        self._last_spreads[mx_pair] = spread_buy_cex

        if signals:
            self._open_arb_count += len(signals)
            self.last_signal_time = datetime.now()

        return signals
