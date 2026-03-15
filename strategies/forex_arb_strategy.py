#!/usr/bin/env python3
"""
Forex Arbitrage Strategy (Strategy #53)
=========================================
Cross-provider foreign exchange arbitrage.

Edge: Knightsbridge FX offers 40-80 bps spreads on CAD pairs vs
200-300 bps at Canadian banks. This strategy:
  1. Polls live FX rates from multiple providers
  2. Detects triangular FX arbitrage across 30+ currencies
  3. Identifies cross-provider spread dislocations
  4. Generates signals for CAD-corridor and UYU-corridor opportunities

Pairs monitored:
  - Majors: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD
  - CAD corridor: USD/CAD, EUR/CAD, GBP/CAD, AUD/CAD, CAD/JPY
  - UYU corridor: USD/UYU, EUR/UYU, BRL/UYU
  - Crosses: EUR/GBP, EUR/JPY, GBP/JPY, EUR/CHF
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


class ForexArbitrageStrategy(BaseArbitrageStrategy):
    """
    Detects FX arbitrage across providers and triangular paths.

    Strategy logic:
      1. Poll Knightsbridge-grade FX rates
      2. Run triangular arb scanner (base→A→B→base)
      3. Compare spreads vs bank retail
      4. Generate signals when profit exceeds min threshold

    Config risk_envelope keys:
      - min_profit_bps: Minimum triangular arb profit to trigger (default 5)
      - max_position_usd: Max notional per FX trade (default $25,000)
      - max_open_fx_positions: Concurrent FX legs (default 5)
      - cad_corridor_weight: Extra weight for CAD pairs (default 1.5)
    """

    CORRIDORS = {
        "cad": ["USD/CAD", "EUR/CAD", "GBP/CAD", "AUD/CAD", "CAD/JPY", "CAD/CHF"],
        "uyu": ["USD/UYU", "EUR/UYU", "BRL/UYU", "ARS/UYU"],
        "major": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "NZD/USD"],
    }

    def __init__(
        self,
        config: StrategyConfig,
        communication: CommunicationFramework,
        audit_logger: AuditLogger,
    ):
        super().__init__(config, communication, audit_logger)
        self._risk = config.risk_envelope or {}
        self.min_profit_bps = self._risk.get("min_profit_bps", 5)
        self.max_position_usd = self._risk.get("max_position_usd", 25000)
        self.max_open = self._risk.get("max_open_fx_positions", 5)
        self.cad_weight = self._risk.get("cad_corridor_weight", 1.5)
        self._open_fx_count = 0
        self._last_arb_scan: Optional[datetime] = None

    async def _initialize_strategy(self):
        """Initialize FX arbitrage state and corridor tracking."""
        self._corridor_spreads = {corridor: {} for corridor in self.CORRIDORS}
        self._last_arb_scan = None
        self._open_fx_count = 0
        self._trade_history = []
        logger.info(
            f"ForexArb initialized — min profit {self.min_profit_bps} bps, "
            f"max position ${self.max_position_usd:,.0f}, "
            f"CAD weight {self.cad_weight}x, "
            f"corridors: {list(self.CORRIDORS.keys())}"
        )

    def _should_generate_signal(self) -> bool:
        """Rate-limit signal generation."""
        if self._open_fx_count >= self.max_open:
            return False
        if self.last_signal_time:
            if datetime.now() - self.last_signal_time < timedelta(seconds=30):
                return False
        return True

    async def _generate_signals(self) -> List[TradingSignal]:
        """
        Scan FX rates for arbitrage opportunities.

        Expects market_data to contain:
          - fx_rates: {pair_str: {bid, ask, mid}} from ForexDataSource
          - fx_arb_opportunities: list of triangular arb results
        """
        signals: List[TradingSignal] = []

        # ── Triangular arb signals ─────────────────────────────────
        arb_opps = self.market_data.get("fx_arb_opportunities", [])
        for opp in arb_opps:
            profit = abs(opp.get("profit_bps", 0))
            if profit < self.min_profit_bps:
                continue

            path = opp.get("path", "unknown")
            # Boost priority for CAD-corridor arbs
            is_cad = "CAD" in path
            adjusted_profit = profit * (self.cad_weight if is_cad else 1.0)

            signal = TradingSignal(
                strategy_id="53_forex_tri_arb",
                signal_type=SignalType.LONG,
                symbol=path,
                quantity=0.0,
                confidence=min(profit / 20, 1.0),
                metadata={
                    "strategy": "forex_triangular_arb",
                    "signal_tag": f"fx_tri_arb_{path.replace('→', '_')}_{datetime.now().strftime('%H%M%S')}",
                    "path": path,
                    "raw_profit_bps": profit,
                    "adjusted_profit_bps": round(adjusted_profit, 2),
                    "direction": opp.get("direction", "forward"),
                    "cad_corridor": is_cad,
                    "strength": min(adjusted_profit / 50, 1.0),
                    "legs": {
                        "leg_1": opp.get("leg_1"),
                        "leg_2": opp.get("leg_2"),
                        "leg_3": opp.get("leg_3"),
                    },
                },
            )
            signals.append(signal)
            logger.info(f"FX arb signal: {path} → {adjusted_profit:.1f} bps")

        # ── Spread dislocation signals ─────────────────────────────
        fx_rates = self.market_data.get("fx_rates", {})
        for pair_str, tick_data in fx_rates.items():
            bid = tick_data.get("bid", 0)
            ask = tick_data.get("ask", 0)
            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2
            spread_bps = (ask - bid) / mid * 10_000

            # If spread is unusually wide (>100 bps on a major), it's a dislocation
            is_major = pair_str in self.CORRIDORS["major"]
            threshold = 100 if is_major else 200

            if spread_bps > threshold:
                signal = TradingSignal(
                    strategy_id="53_forex_dislocation",
                    signal_type=SignalType.LONG,
                    symbol=pair_str,
                    quantity=0.0,
                    confidence=0.6,
                    metadata={
                        "strategy": "forex_spread_dislocation",
                        "signal_tag": f"fx_dislocation_{pair_str.replace('/', '_')}_{datetime.now().strftime('%H%M%S')}",
                        "pair": pair_str,
                        "spread_bps": round(spread_bps, 1),
                        "threshold_bps": threshold,
                        "is_major": is_major,
                        "strength": min(spread_bps / 500, 1.0),
                    },
                )
                signals.append(signal)

        if signals:
            self._open_fx_count += len(signals)
            self.last_signal_time = datetime.now()

        return signals

    async def _on_signal_executed(self, signal: TradingSignal, result: Any):
        """Called after an FX signal is executed."""
        self._open_fx_count = max(0, self._open_fx_count - 1)
        self.audit_logger.log_event(
            "fx_signal_executed",
            {
                "signal_tag": signal.metadata.get("signal_tag", ""),
                "symbol": signal.symbol,
                "result": str(result),
            },
        )
