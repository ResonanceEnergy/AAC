from __future__ import annotations

"""Paper Trading Engine — simulated portfolio, order book, and P&L tracking.

Provides a realistic paper trading environment with:
- Virtual balance management
- Order placement (market and limit)
- Position tracking with live P&L
- Fill simulation with configurable slippage
- Trade history and performance metrics
- Integration with CentralAccounting for persistent recording
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class PaperOrder:
    """Simulated order."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float = 0.0  # limit price (0 for market)
    filled_price: float = 0.0
    filled_qty: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    strategy: str = ""
    created_at: str = ""
    filled_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["side"] = self.side.value
        d["order_type"] = self.order_type.value
        d["status"] = self.status.value
        return d


@dataclass
class PaperPosition:
    """Open position in the paper account."""

    symbol: str
    side: OrderSide
    quantity: float
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    strategy: str = ""
    opened_at: str = ""

    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.current_price

    def update_price(self, price: float) -> None:
        self.current_price = price
        if self.side == OrderSide.BUY:
            self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.avg_entry_price - price) * self.quantity


@dataclass
class TradeRecord:
    """Completed trade for history."""

    trade_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    fees: float
    strategy: str
    opened_at: str
    closed_at: str
    hold_seconds: float = 0.0


@dataclass
class PaperAccount:
    """Virtual trading account state."""

    account_id: str
    starting_balance: float
    cash_balance: float
    positions: dict[str, PaperPosition] = field(default_factory=dict)
    open_orders: list[PaperOrder] = field(default_factory=list)
    trade_history: list[TradeRecord] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_fees: float = 0.0
    peak_equity: float = 0.0
    created_at: str = ""

    @property
    def equity(self) -> float:
        pos_value = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash_balance + pos_value

    @property
    def total_pnl(self) -> float:
        return self.equity - self.starting_balance

    @property
    def total_pnl_pct(self) -> float:
        if self.starting_balance == 0:
            return 0.0
        return (self.total_pnl / self.starting_balance) * 100

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def max_drawdown_pct(self) -> float:
        if self.peak_equity == 0:
            return 0.0
        dd = (self.peak_equity - self.equity) / self.peak_equity * 100
        return max(dd, 0.0)


class PaperTradingEngine:
    """Core paper trading engine — simulates order execution and portfolio management.

    Parameters
    ----------
    account_id : str
        Unique account identifier.
    starting_balance : float
        Initial virtual cash balance.
    fee_rate : float
        Trading fee as a fraction (e.g. 0.001 = 0.1%).
    slippage_bps : float
        Simulated slippage in basis points (e.g. 5 = 0.05%).
    persist_dir : Path | None
        Directory to persist account state (JSON). None = no persistence.
    risk_manager : RiskManager | None
        Optional account-level risk manager. When set, all orders pass
        through pre-trade risk checks before execution.
    """

    def __init__(
        self,
        account_id: str = "paper_default",
        starting_balance: float = 100_000.0,
        fee_rate: float = 0.001,
        slippage_bps: float = 5.0,
        persist_dir: Path | None = None,
        risk_manager: Any | None = None,
    ) -> None:
        self._fee_rate = fee_rate
        self._slippage_bps = slippage_bps
        self._persist_dir = persist_dir
        self._risk_manager = risk_manager

        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            state_file = persist_dir / f"{account_id}.json"
            if state_file.exists():
                self.account = self._load_account(state_file)
                _log.info("paper_engine.loaded", account=account_id, equity=self.account.equity)
            else:
                self.account = self._new_account(account_id, starting_balance)
        else:
            self.account = self._new_account(account_id, starting_balance)

    def _new_account(self, account_id: str, balance: float) -> PaperAccount:
        return PaperAccount(
            account_id=account_id,
            starting_balance=balance,
            cash_balance=balance,
            peak_equity=balance,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # -- Order execution ------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        order_type: OrderType = OrderType.MARKET,
        strategy: str = "",
    ) -> PaperOrder | None:
        """Place and immediately fill a market order, or queue a limit order."""
        # Risk manager pre-trade gate
        if self._risk_manager is not None:
            allowed, reason = self._risk_manager.pre_trade_check(
                account_equity=self.account.equity,
                account_positions=self.account.positions,
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                price=price,
                strategy=strategy,
            )
            if not allowed:
                _log.warning(
                    "paper_engine.risk_blocked",
                    symbol=symbol,
                    side=side.value,
                    reason=reason,
                )
                return None

        order = PaperOrder(
            order_id=uuid.uuid4().hex[:12],
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            strategy=strategy,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        if order_type == OrderType.MARKET:
            return self._fill_order(order, price)
        else:
            # Limit order — store for later evaluation
            self.account.open_orders.append(order)
            return order

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply deterministic slippage."""
        slip = price * (self._slippage_bps / 10_000)
        if side == OrderSide.BUY:
            return price + slip
        return price - slip

    def _fill_order(self, order: PaperOrder, market_price: float) -> PaperOrder | None:
        """Execute fill at market_price with slippage and fees."""
        fill_price = self._apply_slippage(market_price, order.side)
        cost = fill_price * order.quantity
        fee = cost * self._fee_rate

        if order.side == OrderSide.BUY:
            total_cost = cost + fee
            if total_cost > self.account.cash_balance:
                _log.warning("paper_engine.insufficient_funds", needed=total_cost, available=self.account.cash_balance)
                order.status = OrderStatus.CANCELLED
                return order
            self.account.cash_balance -= total_cost
            self._add_to_position(order.symbol, OrderSide.BUY, order.quantity, fill_price, order.strategy)
        else:
            # Selling — check position exists
            pos = self.account.positions.get(order.symbol)
            if not pos or pos.quantity < order.quantity:
                _log.warning("paper_engine.insufficient_position", symbol=order.symbol)
                order.status = OrderStatus.CANCELLED
                return order
            proceeds = cost - fee
            self.account.cash_balance += proceeds
            self._reduce_position(order.symbol, order.quantity, fill_price, order.strategy)

        order.filled_price = fill_price
        order.filled_qty = order.quantity
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now(timezone.utc).isoformat()
        self.account.total_fees += fee

        # Update peak equity
        eq = self.account.equity
        if eq > self.account.peak_equity:
            self.account.peak_equity = eq

        self._persist()
        return order

    def _add_to_position(
        self, symbol: str, side: OrderSide, qty: float, price: float, strategy: str,
    ) -> None:
        existing = self.account.positions.get(symbol)
        if existing and existing.side == side:
            # Average into existing
            total_qty = existing.quantity + qty
            existing.avg_entry_price = (
                (existing.avg_entry_price * existing.quantity) + (price * qty)
            ) / total_qty
            existing.quantity = total_qty
            existing.update_price(price)
        elif existing and existing.side != side:
            # Opposite side — close part of position
            if qty >= existing.quantity:
                self._close_position(symbol, price, strategy)
                remaining = qty - existing.quantity
                if remaining > 0:
                    self.account.positions[symbol] = PaperPosition(
                        symbol=symbol,
                        side=side,
                        quantity=remaining,
                        avg_entry_price=price,
                        current_price=price,
                        strategy=strategy,
                        opened_at=datetime.now(timezone.utc).isoformat(),
                    )
            else:
                self._reduce_position(symbol, qty, price, strategy)
        else:
            self.account.positions[symbol] = PaperPosition(
                symbol=symbol,
                side=side,
                quantity=qty,
                avg_entry_price=price,
                current_price=price,
                strategy=strategy,
                opened_at=datetime.now(timezone.utc).isoformat(),
            )

    def _reduce_position(self, symbol: str, qty: float, price: float, strategy: str) -> None:
        pos = self.account.positions.get(symbol)
        if not pos:
            return
        close_qty = min(qty, pos.quantity)
        if pos.side == OrderSide.BUY:
            pnl = (price - pos.avg_entry_price) * close_qty
        else:
            pnl = (pos.avg_entry_price - price) * close_qty

        pos.realized_pnl += pnl
        pos.quantity -= close_qty

        # Record trade
        self._record_trade(symbol, pos.side.value, close_qty, pos.avg_entry_price, price, pnl, strategy, pos.opened_at)

        if pos.quantity <= 1e-9:
            del self.account.positions[symbol]

    def _close_position(self, symbol: str, price: float, strategy: str) -> None:
        pos = self.account.positions.get(symbol)
        if not pos:
            return
        self._reduce_position(symbol, pos.quantity, price, strategy)

    def _record_trade(
        self, symbol: str, side: str, qty: float,
        entry: float, exit_price: float, pnl: float,
        strategy: str, opened_at: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        trade = TradeRecord(
            trade_id=uuid.uuid4().hex[:12],
            symbol=symbol,
            side=side,
            quantity=qty,
            entry_price=entry,
            exit_price=exit_price,
            pnl=pnl,
            fees=abs(pnl) * self._fee_rate,
            strategy=strategy,
            opened_at=opened_at,
            closed_at=now,
        )
        self.account.trade_history.append(trade)
        self.account.total_trades += 1
        if pnl > 0:
            self.account.winning_trades += 1
        elif pnl < 0:
            self.account.losing_trades += 1

    # -- Price updates --------------------------------------------------------

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all held positions."""
        for symbol, price in prices.items():
            pos = self.account.positions.get(symbol)
            if pos:
                pos.update_price(price)
        eq = self.account.equity
        if eq > self.account.peak_equity:
            self.account.peak_equity = eq

        # Post-update risk checks (circuit breaker, daily loss, trailing stops)
        if self._risk_manager is not None:
            alerts = self._risk_manager.post_update_check(
                account_equity=eq,
                peak_equity=self.account.peak_equity,
                positions=self.account.positions,
            )
            for alert in alerts:
                _log.warning("paper_engine.risk_alert", **alert)

        self._persist()

    def check_limit_orders(self, prices: dict[str, float]) -> list[PaperOrder]:
        """Check pending limit orders against current prices and fill any that match."""
        filled: list[PaperOrder] = []
        remaining: list[PaperOrder] = []

        for order in self.account.open_orders:
            price = prices.get(order.symbol)
            if price is None:
                remaining.append(order)
                continue

            should_fill = False
            if order.side == OrderSide.BUY and price <= order.price:
                should_fill = True
            elif order.side == OrderSide.SELL and price >= order.price:
                should_fill = True

            if should_fill:
                result = self._fill_order(order, price)
                if result and result.status == OrderStatus.FILLED:
                    filled.append(result)
                else:
                    remaining.append(order)
            else:
                remaining.append(order)

        self.account.open_orders = remaining
        return filled

    # -- Reporting ------------------------------------------------------------

    def get_performance(self) -> dict[str, Any]:
        """Return account performance summary."""
        a = self.account
        return {
            "account_id": a.account_id,
            "starting_balance": a.starting_balance,
            "cash_balance": round(a.cash_balance, 2),
            "equity": round(a.equity, 2),
            "total_pnl": round(a.total_pnl, 2),
            "total_pnl_pct": round(a.total_pnl_pct, 2),
            "total_trades": a.total_trades,
            "winning_trades": a.winning_trades,
            "losing_trades": a.losing_trades,
            "win_rate": round(a.win_rate, 1),
            "max_drawdown_pct": round(a.max_drawdown_pct, 2),
            "total_fees": round(a.total_fees, 2),
            "open_positions": len(a.positions),
            "peak_equity": round(a.peak_equity, 2),
        }

    def get_positions_summary(self) -> list[dict[str, Any]]:
        """Return all open positions."""
        return [
            {
                "symbol": p.symbol,
                "side": p.side.value,
                "quantity": p.quantity,
                "avg_entry": round(p.avg_entry_price, 4),
                "current_price": round(p.current_price, 4),
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "market_value": round(p.market_value, 2),
                "strategy": p.strategy,
            }
            for p in self.account.positions.values()
        ]

    def get_trade_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent trade history."""
        trades = self.account.trade_history[-limit:]
        return [
            {
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "side": t.side,
                "qty": t.quantity,
                "entry": round(t.entry_price, 4),
                "exit": round(t.exit_price, 4),
                "pnl": round(t.pnl, 2),
                "strategy": t.strategy,
                "closed_at": t.closed_at,
            }
            for t in trades
        ]

    # -- Persistence ----------------------------------------------------------

    def _persist(self) -> None:
        if not self._persist_dir:
            return
        path = self._persist_dir / f"{self.account.account_id}.json"
        state = {
            "account_id": self.account.account_id,
            "starting_balance": self.account.starting_balance,
            "cash_balance": self.account.cash_balance,
            "total_trades": self.account.total_trades,
            "winning_trades": self.account.winning_trades,
            "losing_trades": self.account.losing_trades,
            "total_fees": self.account.total_fees,
            "peak_equity": self.account.peak_equity,
            "created_at": self.account.created_at,
            "positions": {
                sym: {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "quantity": p.quantity,
                    "avg_entry_price": p.avg_entry_price,
                    "current_price": p.current_price,
                    "realized_pnl": p.realized_pnl,
                    "strategy": p.strategy,
                    "opened_at": p.opened_at,
                }
                for sym, p in self.account.positions.items()
            },
            "trade_history": [asdict(t) for t in self.account.trade_history[-500:]],
        }
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _load_account(self, path: Path) -> PaperAccount:
        data = json.loads(path.read_text(encoding="utf-8"))
        positions: dict[str, PaperPosition] = {}
        for sym, pd_raw in data.get("positions", {}).items():
            positions[sym] = PaperPosition(
                symbol=pd_raw["symbol"],
                side=OrderSide(pd_raw["side"]),
                quantity=pd_raw["quantity"],
                avg_entry_price=pd_raw["avg_entry_price"],
                current_price=pd_raw.get("current_price", 0),
                realized_pnl=pd_raw.get("realized_pnl", 0),
                strategy=pd_raw.get("strategy", ""),
                opened_at=pd_raw.get("opened_at", ""),
            )
        history = [
            TradeRecord(**t) for t in data.get("trade_history", [])
        ]
        return PaperAccount(
            account_id=data["account_id"],
            starting_balance=data["starting_balance"],
            cash_balance=data["cash_balance"],
            positions=positions,
            trade_history=history,
            total_trades=data.get("total_trades", 0),
            winning_trades=data.get("winning_trades", 0),
            losing_trades=data.get("losing_trades", 0),
            total_fees=data.get("total_fees", 0),
            peak_equity=data.get("peak_equity", data["starting_balance"]),
            created_at=data.get("created_at", ""),
        )
