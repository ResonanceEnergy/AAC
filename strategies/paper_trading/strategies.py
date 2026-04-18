from __future__ import annotations

"""Trading strategy algorithms for paper trading.

Implements the most successful retail bot strategies:
- Grid: buy/sell at fixed price levels in a range
- DCA: dollar-cost average on dips or at intervals
- Momentum/Trend-following: ride sustained price moves
- Mean-Reversion: bet on snap-back to average
- Arbitrage: exploit bid/ask or cross-market inefficiencies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

from strategies.paper_trading.engine import OrderSide, PaperTradingEngine

_log = structlog.get_logger(__name__)


@dataclass
class StrategyConfig:
    """Common strategy parameters."""

    name: str
    max_position_pct: float = 0.10  # max % of portfolio per position
    stop_loss_pct: float = 0.05  # 5% stop-loss
    take_profit_pct: float = 0.10  # 10% take-profit
    cooldown_seconds: float = 60.0  # min time between trades
    enabled: bool = True


class StrategyBase(ABC):
    """Base class for all trading strategies.

    Subclasses implement `evaluate()` which returns trade signals
    given current market data, and the engine executes them.
    """

    def __init__(self, config: StrategyConfig, engine: PaperTradingEngine) -> None:
        self.config = config
        self.engine = engine
        self._trade_count: int = 0
        self._last_trade_ts: float = 0.0

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def evaluate(self, market_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Evaluate market data and return trade signals.

        Parameters
        ----------
        market_data : dict
            Keys vary by market type. Common keys:
            - "prices": dict[symbol, float]
            - "volumes": dict[symbol, float]
            - "changes": dict[symbol, float] (24h %)
            - "timestamp": str

        Returns
        -------
        list of dicts with keys:
            - "symbol": str
            - "side": "buy" | "sell"
            - "quantity": float
            - "reason": str
        """

    def execute_signals(self, signals: list[dict[str, Any]], prices: dict[str, float]) -> int:
        """Execute trade signals through the paper engine. Returns fill count."""
        filled = 0
        for sig in signals:
            symbol = sig["symbol"]
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue

            side = OrderSide.BUY if sig["side"] == "buy" else OrderSide.SELL
            qty = sig["quantity"]

            # Position size guard
            max_value = self.engine.account.equity * self.config.max_position_pct
            if side == OrderSide.BUY and (price * qty) > max_value:
                qty = max_value / price

            if qty <= 0:
                continue

            order = self.engine.place_order(
                symbol=symbol,
                side=side,
                quantity=qty,
                price=price,
                strategy=self.name,
            )
            if order and order.status.value == "filled":
                filled += 1
                self._trade_count += 1
                _log.info(
                    "strategy.trade_executed",
                    strategy=self.name,
                    symbol=symbol,
                    side=sig["side"],
                    qty=round(qty, 4),
                    price=price,
                    reason=sig.get("reason", ""),
                )
        return filled

    def check_stops(self, prices: dict[str, float]) -> int:
        """Check stop-loss and take-profit on open positions. Returns exits."""
        exits = 0
        positions = list(self.engine.account.positions.values())
        for pos in positions:
            if pos.strategy != self.name:
                continue
            price = prices.get(pos.symbol)
            if price is None:
                continue

            pnl_pct = 0.0
            if pos.avg_entry_price > 0:
                if pos.side == OrderSide.BUY:
                    pnl_pct = (price - pos.avg_entry_price) / pos.avg_entry_price
                else:
                    pnl_pct = (pos.avg_entry_price - price) / pos.avg_entry_price

            reason = ""
            should_exit = False
            if pnl_pct <= -self.config.stop_loss_pct:
                reason = f"stop_loss ({pnl_pct:.1%})"
                should_exit = True
            elif pnl_pct >= self.config.take_profit_pct:
                reason = f"take_profit ({pnl_pct:.1%})"
                should_exit = True

            if should_exit:
                sell_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
                self.engine.place_order(
                    symbol=pos.symbol,
                    side=sell_side,
                    quantity=pos.quantity,
                    price=price,
                    strategy=self.name,
                )
                exits += 1
                _log.info("strategy.stop_exit", strategy=self.name, symbol=pos.symbol, reason=reason)
        return exits

    def report(self) -> dict[str, Any]:
        """Strategy-level performance metrics."""
        trades = [t for t in self.engine.account.trade_history if t.strategy == self.name]
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = sum(1 for t in trades if t.pnl < 0)
        total_pnl = sum(t.pnl for t in trades)
        return {
            "strategy": self.name,
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": round((wins / len(trades) * 100) if trades else 0, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(trades), 2) if trades else 0,
        }


# =============================================================================
# GRID STRATEGY — buy/sell at fixed price intervals within a range
# =============================================================================


@dataclass
class GridConfig(StrategyConfig):
    """Grid strategy parameters."""

    grid_levels: int = 10  # number of grid lines
    grid_range_pct: float = 0.10  # ±10% from midpoint
    qty_per_level: float = 0.0  # auto-calculated if 0


class GridStrategy(StrategyBase):
    """Grid trading — places buy/sell orders at fixed price levels.

    Works best in sideways/range-bound markets. Buys at lower grid
    levels, sells at upper grid levels. Profits from price oscillation.
    """

    def __init__(self, config: GridConfig, engine: PaperTradingEngine) -> None:
        super().__init__(config, engine)
        self._grids: dict[str, list[float]] = {}  # symbol → grid levels
        self._filled_levels: dict[str, set[int]] = {}

    def _build_grid(self, symbol: str, mid_price: float) -> list[float]:
        cfg: GridConfig = self.config  # type: ignore[assignment]
        low = mid_price * (1 - cfg.grid_range_pct)
        high = mid_price * (1 + cfg.grid_range_pct)
        step = (high - low) / (cfg.grid_levels - 1) if cfg.grid_levels > 1 else (high - low)
        levels = [low + step * i for i in range(cfg.grid_levels)]
        self._grids[symbol] = levels
        self._filled_levels[symbol] = set()
        return levels

    def evaluate(self, market_data: dict[str, Any]) -> list[dict[str, Any]]:
        prices: dict[str, float] = market_data.get("prices", {})
        signals: list[dict[str, Any]] = []
        cfg: GridConfig = self.config  # type: ignore[assignment]

        for symbol, price in prices.items():
            if symbol not in self._grids:
                self._build_grid(symbol, price)

            levels = self._grids[symbol]
            filled = self._filled_levels[symbol]
            mid_idx = len(levels) // 2

            # Determine quantity per level
            qty = cfg.qty_per_level
            if qty <= 0:
                max_val = self.engine.account.equity * cfg.max_position_pct
                qty = max_val / price / cfg.grid_levels if price > 0 else 0

            for i, level in enumerate(levels):
                if i in filled:
                    continue
                if i < mid_idx and price <= level:
                    signals.append({
                        "symbol": symbol, "side": "buy", "quantity": qty,
                        "reason": f"grid_buy_L{i} @ {level:.4f}",
                    })
                    filled.add(i)
                elif i > mid_idx and price >= level:
                    pos = self.engine.account.positions.get(symbol)
                    if pos and pos.quantity > 0:
                        sell_qty = min(qty, pos.quantity)
                        signals.append({
                            "symbol": symbol, "side": "sell", "quantity": sell_qty,
                            "reason": f"grid_sell_L{i} @ {level:.4f}",
                        })
                        filled.add(i)

        return signals


# =============================================================================
# DCA STRATEGY — dollar-cost average into positions on dips
# =============================================================================


@dataclass
class DCAConfig(StrategyConfig):
    """DCA strategy parameters."""

    buy_dip_pct: float = 0.03  # buy when price dips 3%+
    amount_per_buy: float = 0.0  # fixed $ amount per buy (0 = auto)
    max_buys: int = 20  # max DCA buys per symbol
    sell_after_pct: float = 0.08  # sell when up 8% from avg entry


class DCAStrategy(StrategyBase):
    """Dollar-cost averaging — buy on dips, sell on recovery.

    Tracks a reference price per symbol, buys when price dips by
    `buy_dip_pct` from last buy, sells when up `sell_after_pct` from
    average entry.
    """

    def __init__(self, config: DCAConfig, engine: PaperTradingEngine) -> None:
        super().__init__(config, engine)
        self._ref_prices: dict[str, float] = {}
        self._buy_counts: dict[str, int] = {}

    def evaluate(self, market_data: dict[str, Any]) -> list[dict[str, Any]]:
        prices: dict[str, float] = market_data.get("prices", {})
        signals: list[dict[str, Any]] = []
        cfg: DCAConfig = self.config  # type: ignore[assignment]

        for symbol, price in prices.items():
            if price <= 0:
                continue

            ref = self._ref_prices.get(symbol)
            buys = self._buy_counts.get(symbol, 0)

            # Determine buy amount
            amount = cfg.amount_per_buy
            if amount <= 0:
                amount = self.engine.account.equity * cfg.max_position_pct / cfg.max_buys

            qty = amount / price

            # Check for dip buy
            if ref is None:
                # First sighting — set reference, do initial buy
                self._ref_prices[symbol] = price
                signals.append({
                    "symbol": symbol, "side": "buy", "quantity": qty,
                    "reason": "dca_initial_buy",
                })
                self._buy_counts[symbol] = 1
            elif price <= ref * (1 - cfg.buy_dip_pct) and buys < cfg.max_buys:
                signals.append({
                    "symbol": symbol, "side": "buy", "quantity": qty,
                    "reason": f"dca_dip_buy ({((ref - price) / ref) * 100:.1f}% down from ref)",
                })
                self._ref_prices[symbol] = price
                self._buy_counts[symbol] = buys + 1

            # Check for profit take
            pos = self.engine.account.positions.get(symbol)
            if pos and pos.avg_entry_price > 0:
                gain = (price - pos.avg_entry_price) / pos.avg_entry_price
                if gain >= cfg.sell_after_pct:
                    signals.append({
                        "symbol": symbol, "side": "sell", "quantity": pos.quantity,
                        "reason": f"dca_take_profit ({gain:.1%} gain)",
                    })
                    self._ref_prices[symbol] = price
                    self._buy_counts[symbol] = 0

        return signals


# =============================================================================
# MOMENTUM STRATEGY — ride sustained price moves
# =============================================================================


@dataclass
class MomentumConfig(StrategyConfig):
    """Momentum / trend-following parameters."""

    lookback_periods: int = 5  # how many price ticks to track
    entry_threshold_pct: float = 0.03  # enter when cumulative move > 3%
    exit_threshold_pct: float = -0.015  # exit when reversal > 1.5%


class MomentumStrategy(StrategyBase):
    """Trend-following momentum — buy when price shows sustained upward
    movement, sell when trend reverses.
    """

    def __init__(self, config: MomentumConfig, engine: PaperTradingEngine) -> None:
        super().__init__(config, engine)
        self._price_history: dict[str, list[float]] = {}

    def evaluate(self, market_data: dict[str, Any]) -> list[dict[str, Any]]:
        prices: dict[str, float] = market_data.get("prices", {})
        signals: list[dict[str, Any]] = []
        cfg: MomentumConfig = self.config  # type: ignore[assignment]

        for symbol, price in prices.items():
            if price <= 0:
                continue

            history = self._price_history.setdefault(symbol, [])
            history.append(price)
            if len(history) > cfg.lookback_periods * 2:
                history[:] = history[-(cfg.lookback_periods * 2):]

            if len(history) < cfg.lookback_periods:
                continue

            # Calculate momentum
            old_price = history[-cfg.lookback_periods]
            momentum = (price - old_price) / old_price if old_price > 0 else 0

            pos = self.engine.account.positions.get(symbol)
            max_val = self.engine.account.equity * cfg.max_position_pct
            qty = max_val / price if price > 0 else 0

            if momentum >= cfg.entry_threshold_pct and not pos:
                signals.append({
                    "symbol": symbol, "side": "buy", "quantity": qty,
                    "reason": f"momentum_entry ({momentum:.2%} over {cfg.lookback_periods} ticks)",
                })
            elif momentum <= cfg.exit_threshold_pct and pos and pos.side == OrderSide.BUY:
                signals.append({
                    "symbol": symbol, "side": "sell", "quantity": pos.quantity,
                    "reason": f"momentum_exit ({momentum:.2%} reversal)",
                })

        return signals


# =============================================================================
# MEAN-REVERSION STRATEGY — bet on snap-back to average
# =============================================================================


@dataclass
class MeanReversionConfig(StrategyConfig):
    """Mean-reversion parameters."""

    window: int = 20  # price ticks for moving average
    entry_z: float = 2.0  # Z-score threshold to enter
    exit_z: float = 0.5  # Z-score to exit (back near mean)


class MeanReversionStrategy(StrategyBase):
    """Mean-reversion — buy when price is significantly below its
    moving average (oversold), sell when it returns to the mean.
    """

    def __init__(self, config: MeanReversionConfig, engine: PaperTradingEngine) -> None:
        super().__init__(config, engine)
        self._price_history: dict[str, list[float]] = {}

    def evaluate(self, market_data: dict[str, Any]) -> list[dict[str, Any]]:
        prices: dict[str, float] = market_data.get("prices", {})
        signals: list[dict[str, Any]] = []
        cfg: MeanReversionConfig = self.config  # type: ignore[assignment]

        for symbol, price in prices.items():
            if price <= 0:
                continue

            history = self._price_history.setdefault(symbol, [])
            history.append(price)
            if len(history) > cfg.window * 3:
                history[:] = history[-(cfg.window * 3):]

            if len(history) < cfg.window:
                continue

            window = history[-cfg.window:]
            mean = sum(window) / len(window)
            variance = sum((p - mean) ** 2 for p in window) / len(window)
            std = variance ** 0.5 if variance > 0 else 1e-9

            z_score = (price - mean) / std
            pos = self.engine.account.positions.get(symbol)
            max_val = self.engine.account.equity * cfg.max_position_pct
            qty = max_val / price if price > 0 else 0

            if z_score <= -cfg.entry_z and not pos:
                signals.append({
                    "symbol": symbol, "side": "buy", "quantity": qty,
                    "reason": f"mean_rev_buy (z={z_score:.2f}, mean={mean:.4f})",
                })
            elif pos and pos.side == OrderSide.BUY and z_score >= cfg.exit_z:
                signals.append({
                    "symbol": symbol, "side": "sell", "quantity": pos.quantity,
                    "reason": f"mean_rev_exit (z={z_score:.2f}, back to mean)",
                })

        return signals


# =============================================================================
# ARBITRAGE STRATEGY — exploit cross-market price differences
# =============================================================================


@dataclass
class ArbitrageConfig(StrategyConfig):
    """Arbitrage parameters (for prediction markets like Polymarket)."""

    min_edge_pct: float = 0.5  # minimum edge to trade
    max_bet_size: float = 50.0  # max $ per arb bet


class ArbitrageStrategy(StrategyBase):
    """Prediction market arbitrage — exploit YES+NO < 1.0 mispricing.

    For Polymarket: if YES + NO prices sum to less than 1.0, buy the
    underpriced side. Also handles cross-market arbitrage when the
    same event appears with different probabilities.
    """

    def evaluate(self, market_data: dict[str, Any]) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        cfg: ArbitrageConfig = self.config  # type: ignore[assignment]
        arb_opps = market_data.get("arb_opportunities", [])

        for opp in arb_opps:
            edge = opp.get("edge_pct", 0)
            if edge < cfg.min_edge_pct:
                continue

            symbol = opp.get("condition_id", opp.get("symbol", ""))
            side_str = opp.get("side", "YES").lower()
            price = opp.get("yes_price", 0.5) if side_str == "yes" else opp.get("no_price", 0.5)

            if price <= 0 or price >= 1.0:
                continue

            qty = min(cfg.max_bet_size, self.engine.account.equity * cfg.max_position_pct) / price

            signals.append({
                "symbol": symbol,
                "side": "buy",
                "quantity": qty,
                "reason": f"arb_edge ({edge:.2f}% on {side_str.upper()}, price={price:.3f})",
            })

        return signals
