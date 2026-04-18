from __future__ import annotations

"""Polymarket Paper Trading Division — simulated prediction market trading.

Consumes intel from PolymarketCouncilDivision and runs paper trades
using Grid, DCA, Momentum, and Arbitrage strategies on trending
prediction markets.  Continuously scores and optimises strategies.

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
    ArbitrageStrategy,
    DCAStrategy,
    GridStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    ArbitrageConfig,
    DCAConfig,
    GridConfig,
    MomentumConfig,
    MeanReversionConfig,
)

_log = structlog.get_logger(__name__)

_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "paper_trading" / "polymarket"


class PolymarketPaperDivision(DivisionProtocol):
    """Paper-trade prediction markets using multiple bot strategies.

    Data flow:
        PolymarketCouncilDivision → (INTEL_UPDATE) → this division
        This division → (TRADE_SIGNAL) → War Room / dashboards

    Strategies deployed:
        - Grid: range-bound oscillation on stable markets
        - DCA: accumulate conviction positions on dips
        - Momentum: ride trending markets
        - MeanReversion: fade overextended prices
        - Arbitrage: exploit YES+NO < $1.00 mispricing
    """

    def __init__(
        self,
        starting_balance: float = 10_000.0,
        persist: bool = True,
    ) -> None:
        super().__init__(division_name="polymarket_paper")

        persist_dir = _PERSIST_DIR if persist else None

        # Risk management, regime detection, metrics
        self._risk_manager = RiskManager(RiskConfig(
            max_drawdown_pct=25.0,             # wider for prediction markets
            daily_loss_limit_pct=8.0,
            max_open_positions=30,
            max_position_pct=0.15,
            max_strategy_allocation_pct=40.0,
            trailing_stop_pct=0.0,             # disabled — binary outcomes
        ))
        self._regime_detector = RegimeDetector()
        self._metrics = MetricsTracker()

        self._engine = PaperTradingEngine(
            account_id="polymarket_paper",
            starting_balance=starting_balance,
            fee_rate=0.002,        # Polymarket ~0.2% fee
            slippage_bps=10,       # prediction markets are thinner
            persist_dir=persist_dir,
            risk_manager=self._risk_manager,
        )

        # -- Strategy instances -----------------------------------------------
        self._grid = GridStrategy(
            GridConfig(
                name="poly_grid",
                grid_levels=8,
                grid_range_pct=0.15,       # wider range for prediction markets
                max_position_pct=0.08,
                stop_loss_pct=0.20,        # wider stops — binary payoffs
                take_profit_pct=0.40,
            ),
            self._engine,
        )

        self._dca = DCAStrategy(
            DCAConfig(
                name="poly_dca",
                buy_dip_pct=0.05,
                max_buys=10,
                sell_after_pct=0.15,
                max_position_pct=0.10,
                stop_loss_pct=0.25,
                take_profit_pct=0.50,
            ),
            self._engine,
        )

        self._momentum = MomentumStrategy(
            MomentumConfig(
                name="poly_momentum",
                lookback_periods=4,
                entry_threshold_pct=0.05,
                exit_threshold_pct=-0.03,
                max_position_pct=0.08,
                stop_loss_pct=0.15,
                take_profit_pct=0.30,
            ),
            self._engine,
        )

        self._mean_rev = MeanReversionStrategy(
            MeanReversionConfig(
                name="poly_mean_rev",
                window=10,
                entry_z=1.5,
                exit_z=0.3,
                max_position_pct=0.08,
                stop_loss_pct=0.20,
                take_profit_pct=0.25,
            ),
            self._engine,
        )

        self._arb = ArbitrageStrategy(
            ArbitrageConfig(
                name="poly_arb",
                min_edge_pct=0.5,
                max_bet_size=50.0,
                max_position_pct=0.12,
                stop_loss_pct=0.10,
                take_profit_pct=0.20,
            ),
            self._engine,
        )

        strategies = [self._grid, self._dca, self._momentum, self._mean_rev, self._arb]

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
            _log.debug("polymarket_paper.no_prices")
            return signals

        # Update regime detection
        prices = market_data.get("prices", {})
        if prices:
            self._regime_detector.update_prices(prices)

        try:
            cycle = self._optimizer.run_cycle(market_data)
        except Exception as exc:
            _log.warning("polymarket_paper.cycle_failed", error=str(exc))
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
                "type": "polymarket_paper_cycle",
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
                    "type": "polymarket_paper_drawdown",
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
        """Accept INTEL_UPDATE from Polymarket council."""
        await super().consume_signal(signal)

        if signal.signal_type != SignalType.INTEL_UPDATE:
            return

        data = signal.data
        if data.get("type") != "polymarket_scan":
            return

        self._latest_intel = data
        _log.info(
            "polymarket_paper.intel_received",
            markets=data.get("total_markets", 0),
            volume=data.get("total_volume", 0),
        )

    # -- Internal helpers ----------------------------------------------------

    def _build_market_data(self) -> dict[str, Any]:
        """Build market_data dict from latest intel for strategy consumption."""
        md: dict[str, Any] = {
            "prices": {},
            "volumes": {},
            "changes": {},
            "arb_opportunities": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        intel = self._latest_intel
        if not intel:
            return md

        # Extract prices from top volume markets
        top_vol = intel.get("top_volume", [])
        for mkt in top_vol:
            if isinstance(mkt, dict):
                cid = mkt.get("condition_id") or mkt.get("question", "")[:30]
                price = mkt.get("price") or mkt.get("best_yes", 0)
                vol = mkt.get("volume", 0)
                if cid and price:
                    md["prices"][cid] = float(price)
                    md["volumes"][cid] = float(vol)

        # Arbitrage opportunities (YES+NO < 1.0)
        md["arb_opportunities"] = intel.get("arb_opportunities", [])

        return md
