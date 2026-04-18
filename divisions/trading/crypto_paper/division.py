from __future__ import annotations

"""Crypto Paper Trading Division — simulated cryptocurrency trading.

Consumes intel from CryptoCouncilDivision and runs paper trades
using Grid, DCA, Momentum, and MeanReversion strategies on top
crypto assets.  Continuously scores and optimises strategies.

No real money or orders are placed.  All trades are simulated through
the PaperTradingEngine with deterministic slippage and fees.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from divisions.division_protocol import (
    DivisionHealth,
    DivisionProtocol,
    Signal,
    SignalType,
)
from strategies.paper_trading.engine import PaperTradingEngine
from strategies.paper_trading.metrics import MetricsTracker
from strategies.paper_trading.optimizer import StrategyOptimizer
from strategies.paper_trading.regime_detector import RegimeDetector
from strategies.paper_trading.risk_manager import RiskConfig, RiskManager
from strategies.paper_trading.strategies import (
    DCAStrategy,
    GridStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    DCAConfig,
    GridConfig,
    MomentumConfig,
    MeanReversionConfig,
)

_log = structlog.get_logger(__name__)

_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "paper_trading" / "crypto"


class CryptoPaperDivision(DivisionProtocol):
    """Paper-trade crypto markets using multiple bot strategies.

    Data flow:
        CryptoCouncilDivision → (INTEL_UPDATE) → this division
        This division → (TRADE_SIGNAL) → War Room / dashboards

    Strategies deployed:
        - Grid: range-bound accumulation on BTC/ETH in consolidation
        - DCA: systematic accumulation on dips across watchlist
        - Momentum: ride trending altcoins showing sustained moves
        - MeanReversion: fade extreme moves in volatile alts
    """

    def __init__(
        self,
        starting_balance: float = 10_000.0,
        persist: bool = True,
    ) -> None:
        super().__init__(division_name="crypto_paper")

        persist_dir = _PERSIST_DIR if persist else None

        # Risk management, regime detection, metrics
        self._risk_manager = RiskManager(RiskConfig(
            max_drawdown_pct=20.0,
            daily_loss_limit_pct=5.0,
            max_open_positions=25,
            max_position_pct=0.12,
            max_strategy_allocation_pct=35.0,
            trailing_stop_pct=0.10,            # 10% trailing stop for crypto
        ))
        self._regime_detector = RegimeDetector()
        self._metrics = MetricsTracker()

        self._engine = PaperTradingEngine(
            account_id="crypto_paper",
            starting_balance=starting_balance,
            fee_rate=0.001,        # ~0.1% typical crypto spot fee
            slippage_bps=8,        # crypto markets are reasonably liquid
            persist_dir=persist_dir,
            risk_manager=self._risk_manager,
        )

        # -- Strategy instances -----------------------------------------------
        self._grid = GridStrategy(
            GridConfig(
                name="crypto_grid",
                grid_levels=12,
                grid_range_pct=0.08,       # ±8% grid around BTC/ETH
                max_position_pct=0.10,
                stop_loss_pct=0.10,
                take_profit_pct=0.15,
            ),
            self._engine,
        )

        self._dca = DCAStrategy(
            DCAConfig(
                name="crypto_dca",
                buy_dip_pct=0.04,          # buy on 4%+ dips
                max_buys=15,
                sell_after_pct=0.12,       # sell when up 12%
                max_position_pct=0.12,
                stop_loss_pct=0.15,
                take_profit_pct=0.25,
            ),
            self._engine,
        )

        self._momentum = MomentumStrategy(
            MomentumConfig(
                name="crypto_momentum",
                lookback_periods=6,
                entry_threshold_pct=0.04,  # enter on 4%+ sustained move
                exit_threshold_pct=-0.02,  # exit on 2%+ reversal
                max_position_pct=0.08,
                stop_loss_pct=0.08,
                take_profit_pct=0.20,
            ),
            self._engine,
        )

        self._mean_rev = MeanReversionStrategy(
            MeanReversionConfig(
                name="crypto_mean_rev",
                window=15,
                entry_z=2.0,               # buy at 2σ below mean
                exit_z=0.5,
                max_position_pct=0.08,
                stop_loss_pct=0.12,
                take_profit_pct=0.18,
            ),
            self._engine,
        )

        strategies = [self._grid, self._dca, self._momentum, self._mean_rev]

        self._optimizer = StrategyOptimizer(
            engine=self._engine,
            strategies=strategies,
            min_trades_to_score=3,
            persist_dir=persist_dir,
        )

        # Latest intel snapshot received from council
        self._latest_intel: dict[str, Any] = {}
        self._total_cycles: int = 0

    # -- DivisionProtocol overrides ------------------------------------------

    async def scan(self) -> list[Signal]:
        """Run one optimisation cycle using latest intel."""
        signals: list[Signal] = []

        market_data = self._build_market_data()
        if not market_data.get("prices"):
            _log.debug("crypto_paper.no_prices")
            return signals

        # Update regime detection
        prices = market_data.get("prices", {})
        if prices:
            self._regime_detector.update_prices(prices)

        try:
            cycle = self._optimizer.run_cycle(market_data)
        except Exception as exc:
            _log.warning("crypto_paper.cycle_failed", error=str(exc))
            self.health = DivisionHealth.DEGRADED
            return signals

        self._total_cycles += 1
        self.health = DivisionHealth.HEALTHY

        # Record completed trades for metrics
        for trade in self._engine.account.trade_history[-(self._engine.account.total_trades or 0):]:
            self._metrics.record_trade(trade.pnl)
        self._metrics.update_equity(self._engine.account.equity)

        # Emit trade signal with cycle summary
        perf = self._engine.get_performance()
        rankings = self._optimizer.get_rankings()
        best = self._optimizer.get_best_strategy()

        signals.append(Signal(
            signal_type=SignalType.TRADE_SIGNAL,
            source_division=self.division_name,
            timestamp=datetime.now(timezone.utc),
            data={
                "type": "crypto_paper_cycle",
                "cycle": self._total_cycles,
                "equity": perf.get("equity", 0),
                "total_pnl": perf.get("total_pnl", 0),
                "total_pnl_pct": perf.get("total_pnl_pct", 0),
                "total_trades": perf.get("total_trades", 0),
                "win_rate": perf.get("win_rate", 0),
                "best_strategy": best,
                "rankings": rankings[:3],
                "open_positions": len(self._engine.account.positions),
                "regime": self._regime_detector.current_regime.value,
                "risk_halted": self._risk_manager.state.is_halted,
            },
            confidence=0.6,
            urgency=0,
        ))

        # Alert on drawdown
        dd = perf.get("max_drawdown_pct", 0)
        if dd > 10:
            signals.append(Signal(
                signal_type=SignalType.RISK_ALERT,
                source_division=self.division_name,
                timestamp=datetime.now(timezone.utc),
                data={
                    "type": "crypto_paper_drawdown",
                    "drawdown_pct": dd,
                    "equity": perf.get("equity", 0),
                },
                confidence=0.9,
                urgency=2,
            ))

        return signals

    async def report(self) -> dict[str, Any]:
        perf = self._engine.get_performance()
        rankings = self._optimizer.get_rankings()
        return {
            "division": self.division_name,
            "cycles": self._total_cycles,
            "account": {
                "equity": perf.get("equity", 0),
                "cash": perf.get("cash_balance", 0),
                "total_pnl": perf.get("total_pnl", 0),
                "total_pnl_pct": perf.get("total_pnl_pct", 0),
                "total_trades": perf.get("total_trades", 0),
                "win_rate": perf.get("win_rate", 0),
                "max_drawdown_pct": perf.get("max_drawdown_pct", 0),
            },
            "strategies": rankings,
            "best_strategy": self._optimizer.get_best_strategy(),
            "open_positions": self._engine.get_positions_summary(),
            "regime": self._regime_detector.get_status(),
            "risk": self._risk_manager.get_status(),
            "metrics": self._metrics.get_status(),
        }

    async def consume_signal(self, signal: Signal) -> None:
        """Accept INTEL_UPDATE from Crypto council."""
        await super().consume_signal(signal)

        if signal.signal_type != SignalType.INTEL_UPDATE:
            return

        data = signal.data
        if data.get("type") != "crypto_scan":
            return

        self._latest_intel = data
        _log.info(
            "crypto_paper.intel_received",
            coins=data.get("total_coins", 0),
            sentiment=data.get("sentiment", ""),
        )

    # -- Internal helpers ----------------------------------------------------

    def _build_market_data(self) -> dict[str, Any]:
        """Build market_data dict from latest intel for strategy consumption."""
        md: dict[str, Any] = {
            "prices": {},
            "volumes": {},
            "changes": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        intel = self._latest_intel
        if not intel:
            return md

        # Extract prices from coin data relayed in intel
        # The CryptoCouncilDivision sends btc_price, eth_price directly
        btc = intel.get("btc_price")
        eth = intel.get("eth_price")
        if btc:
            md["prices"]["bitcoin"] = float(btc)
        if eth:
            md["prices"]["ethereum"] = float(eth)

        # Gainers/losers contain coin snapshots
        for mover in intel.get("gainers", []) + intel.get("losers", []):
            if isinstance(mover, dict):
                cid = mover.get("coin_id", mover.get("symbol", ""))
                price = mover.get("price", 0)
                change = mover.get("change_24h", 0)
                vol = mover.get("volume_24h", 0)
                if cid and price:
                    md["prices"][cid] = float(price)
                    md["changes"][cid] = float(change)
                    md["volumes"][cid] = float(vol)

        return md
