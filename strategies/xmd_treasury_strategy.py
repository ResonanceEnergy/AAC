#!/usr/bin/env python3
"""
XMD Stablecoin Treasury Strategy (Vector 3)
=============================================
Monitors and trades the Metal Dollar (XMD) peg to maximize
yield while managing depeg risk.

XMD is a USD-backed stablecoin on XPR Network, NMLS-licensed.
The strategy exploits premium/discount oscillations vs fiat USD.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

from shared.strategy_framework import (
    BaseArbitrageStrategy,
    StrategyConfig,
    TradingSignal,
    SignalType,
)
from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class XMDTreasuryStrategy(BaseArbitrageStrategy):
    """
    XMD stablecoin peg arbitrage and treasury management.

    Opportunities:
      1. XMD trades at slight premium → sell XMD, buy USD elsewhere
      2. XMD trades at slight discount → buy XMD, redeem at par
      3. XMD lending yield > risk-free rate → park treasury in XMD lending
      4. XMD pair as settlement: use XMD as zero-fee intermediary

    Config risk_envelope keys:
      - peg_threshold_bps: Deviation from $1.00 to trigger (default 10)
      - max_xmd_exposure: Max XMD holding in USD (default $25,000)
      - lending_min_apy: Minimum lending APY to deploy (default 3.0%)
      - depeg_stop_loss_pct: Hard stop if XMD depegs > this % (default 2.0%)
    """

    def __init__(
        self,
        config: StrategyConfig,
        communication: CommunicationFramework,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, communication, audit_logger)
        self._risk = config.risk_envelope or {}
        self.peg_threshold_bps = self._risk.get("peg_threshold_bps", 10)
        self.max_xmd_exposure = self._risk.get("max_xmd_exposure", 25000)
        self.lending_min_apy = self._risk.get("lending_min_apy", 3.0)
        self.depeg_stop_loss_pct = self._risk.get("depeg_stop_loss_pct", 2.0)
        self._xmd_price_history: List[float] = []

    async def _initialize_strategy(self):
        logger.info(
            f"XMD Treasury initialized — peg threshold {self.peg_threshold_bps} bps, "
            f"max exposure ${self.max_xmd_exposure}"
        )

    def _should_generate_signal(self) -> bool:
        if self.last_signal_time:
            if datetime.now() - self.last_signal_time < timedelta(seconds=30):
                return False
        return True

    async def _generate_signals(self) -> List[TradingSignal]:
        """
        Monitor XMD peg and generate treasury signals.

        market_data expected:
          - xmd_price: float (current XMD/USD price)
          - xmd_lending_apy: float (current lending rate %)
          - xmd_volume_24h: float
        """
        signals: List[TradingSignal] = []

        xmd_price = self.market_data.get("xmd_price", 1.0)
        lending_apy = self.market_data.get("xmd_lending_apy", 0.0)
        volume_24h = self.market_data.get("xmd_volume_24h", 0.0)

        self._xmd_price_history.append(xmd_price)
        if len(self._xmd_price_history) > 1000:
            self._xmd_price_history = self._xmd_price_history[-500:]

        # Calculate peg deviation
        peg_deviation_bps = abs(xmd_price - 1.0) * 10000
        peg_direction = "premium" if xmd_price > 1.0 else "discount"

        # Safety: hard stop on major depeg
        if abs(xmd_price - 1.0) * 100 > self.depeg_stop_loss_pct:
            signals.append(
                TradingSignal(
                    strategy_id=self.config.strategy_id,
                    signal_type=SignalType.CLOSE,
                    symbol="XMD/USD",
                    quantity=0,  # Close all
                    confidence=1.0,
                    metadata={
                        "reason": "depeg_stop_loss",
                        "xmd_price": xmd_price,
                        "deviation_pct": abs(xmd_price - 1.0) * 100,
                    },
                )
            )
            self.last_signal_time = datetime.now()
            return signals

        # Peg arbitrage opportunity
        if peg_deviation_bps > self.peg_threshold_bps:
            qty = min(self.max_xmd_exposure, volume_24h * 0.01)  # Max 1% of volume

            if peg_direction == "premium":
                # XMD overvalued → sell XMD
                signals.append(
                    TradingSignal(
                        strategy_id=self.config.strategy_id,
                        signal_type=SignalType.SHORT,
                        symbol="XMD/USD",
                        quantity=qty,
                        price=xmd_price,
                        confidence=min(peg_deviation_bps / 50, 0.95),
                        metadata={
                            "arb_type": "peg_premium_sell",
                            "deviation_bps": peg_deviation_bps,
                            "xmd_price": xmd_price,
                        },
                    )
                )
            else:
                # XMD undervalued → buy XMD at discount
                signals.append(
                    TradingSignal(
                        strategy_id=self.config.strategy_id,
                        signal_type=SignalType.LONG,
                        symbol="XMD/USD",
                        quantity=qty,
                        price=xmd_price,
                        confidence=min(peg_deviation_bps / 50, 0.95),
                        metadata={
                            "arb_type": "peg_discount_buy",
                            "deviation_bps": peg_deviation_bps,
                            "xmd_price": xmd_price,
                        },
                    )
                )

        # Lending yield opportunity
        if lending_apy > self.lending_min_apy and peg_deviation_bps < self.peg_threshold_bps:
            signals.append(
                TradingSignal(
                    strategy_id=self.config.strategy_id,
                    signal_type=SignalType.LONG,
                    symbol="XMD/LEND",
                    quantity=self.max_xmd_exposure * 0.5,
                    confidence=min(lending_apy / 10, 0.9),
                    metadata={
                        "arb_type": "lending_yield",
                        "lending_apy": lending_apy,
                        "xmd_price": xmd_price,
                    },
                )
            )

        if signals:
            self.last_signal_time = datetime.now()

        return signals
