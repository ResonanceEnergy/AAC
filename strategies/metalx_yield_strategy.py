#!/usr/bin/env python3
"""
Metal X Yield Strategy (Vector 8)
==================================
Automated yield farming, lending, and borrowing on Metal X DEX.

Metal X offers:
  - Lending pools with variable APY
  - Borrowing against crypto collateral
  - Liquidity provision for swap pools
  - Yield farming rewards in MTL/XPR
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from shared.strategy_framework import (
    BaseArbitrageStrategy,
    StrategyConfig,
    TradingSignal,
    SignalType,
)
from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class MetalXYieldStrategy(BaseArbitrageStrategy):
    """
    Yield optimization across Metal X DeFi products.

    Sub-strategies:
      1. Lending rate arbitrage: move capital to highest-yielding pool
      2. Borrow-lend spread: borrow low, lend high across assets
      3. LP yield farming: provide liquidity to pools with best rewards
      4. Yield compounding: auto-compound farming rewards

    Config risk_envelope keys:
      - min_lending_apy: Minimum APY to deploy lending (default 3.0)
      - max_lending_usd: Max capital in lending (default $10,000)
      - max_borrow_ltv: Max loan-to-value ratio (default 0.65)
      - min_lp_apy: Minimum LP APY to enter pool (default 8.0)
      - max_lp_usd: Max capital in LP (default $5,000)
      - impermanent_loss_limit_pct: Exit LP if IL > this (default 5.0)
    """

    def __init__(
        self,
        config: StrategyConfig,
        communication: CommunicationFramework,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, communication, audit_logger)
        self.last_signal_time: Optional[datetime] = None
        self._risk = config.risk_envelope or {}
        self.min_lending_apy = self._risk.get("min_lending_apy", 3.0)
        self.max_lending_usd = self._risk.get("max_lending_usd", 10000)
        self.max_borrow_ltv = self._risk.get("max_borrow_ltv", 0.65)
        self.min_lp_apy = self._risk.get("min_lp_apy", 8.0)
        self.max_lp_usd = self._risk.get("max_lp_usd", 5000)
        self.il_limit_pct = self._risk.get("impermanent_loss_limit_pct", 5.0)
        self._current_allocations: Dict[str, float] = {}

    async def _initialize_strategy(self):
        """Initialize yield strategy state and allocation tracking."""
        self._current_allocations = {}
        self._pool_history = []
        self._total_yield_earned = 0.0
        self._active_lending_positions = {}
        self._active_lp_positions = {}
        logger.info(
            f"MetalXYield initialized — lending min APY {self.min_lending_apy}%, "
            f"LP min APY {self.min_lp_apy}%, "
            f"max lending ${self.max_lending_usd:,.0f}, max LP ${self.max_lp_usd:,.0f}"
        )

    def _should_generate_signal(self) -> bool:
        # Yield strategies check less frequently (every 60s)
        if self.last_signal_time:
            if datetime.now() - self.last_signal_time < timedelta(seconds=60):
                return False
        return True

    async def _generate_signals(self) -> List[TradingSignal]:
        """
        Generate yield signals from Metal X DeFi data.

        market_data expected:
          - lending_pools: [{asset, supply_apy, borrow_apy, utilization, total_supply}]
          - lp_pools: [{pair, apy, tvl, volume_24h, rewards_token}]
          - farming_rewards: [{pool, reward_rate, reward_token, apy_boost}]
        """
        signals: List[TradingSignal] = []

        lending_pools = self.market_data.get("lending_pools", [])
        lp_pools = self.market_data.get("lp_pools", [])

        # 1. Lending rate signals — deploy to best pools
        if lending_pools:
            best_pools = sorted(
                [p for p in lending_pools if p.get("supply_apy", 0) >= self.min_lending_apy],
                key=lambda p: p.get("supply_apy", 0),
                reverse=True,
            )

            for pool in best_pools[:3]:
                asset = pool.get("asset", "")
                apy = pool.get("supply_apy", 0)
                deploy_amount = min(
                    self.max_lending_usd,
                    pool.get("total_supply", 0) * 0.05,  # Max 5% of pool
                )

                if deploy_amount > 0:
                    signals.append(
                        TradingSignal(
                            strategy_id=self.config.strategy_id,
                            signal_type=SignalType.LONG,
                            symbol=f"{asset}/LEND",
                            quantity=deploy_amount,
                            confidence=min(apy / 15, 0.9),
                            metadata={
                                "yield_type": "lending",
                                "apy": apy,
                                "pool_utilization": pool.get("utilization", 0),
                                "asset": asset,
                            },
                        )
                    )

        # 2. Borrow-lend spread — borrow cheap asset, lend expensive one
        if len(lending_pools) >= 2:
            cheapest_borrow = min(
                [p for p in lending_pools if p.get("borrow_apy", 999) < 20],
                key=lambda p: p.get("borrow_apy", 999),
                default=None,
            )
            best_supply = max(
                lending_pools, key=lambda p: p.get("supply_apy", 0), default=None
            )

            if cheapest_borrow and best_supply:
                borrow_rate = cheapest_borrow.get("borrow_apy", 0)
                supply_rate = best_supply.get("supply_apy", 0)
                spread = supply_rate - borrow_rate

                if spread > 1.5:  # At least 1.5% spread
                    borrow_usd = self.max_lending_usd * self.max_borrow_ltv
                    signals.append(
                        TradingSignal(
                            strategy_id=self.config.strategy_id,
                            signal_type=SignalType.LONG,
                            symbol=f"{best_supply.get('asset', '')}/YIELD",
                            quantity=borrow_usd,
                            confidence=min(spread / 5, 0.85),
                            metadata={
                                "yield_type": "borrow_lend_spread",
                                "borrow_asset": cheapest_borrow.get("asset", ""),
                                "borrow_rate": borrow_rate,
                                "supply_asset": best_supply.get("asset", ""),
                                "supply_rate": supply_rate,
                                "net_spread": spread,
                            },
                        )
                    )

        # 3. LP pool signals
        if lp_pools:
            attractive_lps = sorted(
                [p for p in lp_pools if p.get("apy", 0) >= self.min_lp_apy],
                key=lambda p: p.get("apy", 0),
                reverse=True,
            )

            for pool in attractive_lps[:2]:
                pair = pool.get("pair", "")
                apy = pool.get("apy", 0)
                tvl = pool.get("tvl", 0)
                deposit = min(self.max_lp_usd, tvl * 0.02)  # Max 2% of TVL

                if deposit > 0:
                    signals.append(
                        TradingSignal(
                            strategy_id=self.config.strategy_id,
                            signal_type=SignalType.LONG,
                            symbol=f"{pair}/LP",
                            quantity=deposit,
                            confidence=min(apy / 20, 0.85),
                            metadata={
                                "yield_type": "liquidity_provision",
                                "pair": pair,
                                "apy": apy,
                                "tvl": tvl,
                                "rewards_token": pool.get("rewards_token", ""),
                            },
                        )
                    )

        if signals:
            self.last_signal_time = datetime.now()

        return signals
