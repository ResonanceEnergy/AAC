"""
MATRIX MAXIMIZER — IBKR Execution Engine
==========================================
Wires MATRIX MAXIMIZER picks directly into IBKR TWS:
  - Paper mode (port 7497) for validation
  - Live mode (port 7496) for real execution
  - Position tracking with P&L
  - Kelly criterion + fixed-fractional sizing
  - Order state machine (PENDING → FILLED → CLOSED)

All methods are sync wrappers — IBKR connector is async underneath.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_STATE_DIR = Path("data/matrix_maximizer/positions")


class OrderState(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    CLOSED = "closed"


class ExecutionMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    DRY_RUN = "dry_run"


@dataclass
class TrackedPosition:
    """Managed put position with full lifecycle tracking."""
    position_id: str
    ticker: str
    strike: float
    expiry: str
    contracts: int
    entry_premium: float
    entry_date: str
    entry_delta: float
    cost_basis: float          # contracts × premium × 100
    current_premium: float = 0.0
    current_delta: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    pnl_pct: float = 0.0
    days_held: int = 0
    state: str = "open"
    ibkr_order_id: str = ""
    ibkr_con_id: int = 0
    notes: str = ""

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    def to_scanner_position(self) -> Dict[str, Any]:
        """Convert to scanner.Position format for roll checking."""
        from strategies.matrix_maximizer.scanner import Position
        return Position(
            ticker=self.ticker,
            strike=self.strike,
            expiry=self.expiry,
            entry_date=self.entry_date,
            entry_premium=self.entry_premium,
            entry_delta=self.entry_delta,
            contracts=self.contracts,
            cost_basis=self.cost_basis,
            current_premium=self.current_premium,
            current_delta=self.current_delta,
            days_held=self.days_held,
            pnl_pct=self.pnl_pct,
        )


@dataclass
class OrderResult:
    """Result of an order submission."""
    success: bool
    order_id: str = ""
    state: str = ""
    fill_price: float = 0.0
    filled_qty: int = 0
    message: str = ""
    timestamp: str = ""


@dataclass
class AccountSnapshot:
    """IBKR account summary."""
    total_value: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    maintenance_margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    put_exposure: float = 0.0
    positions_count: int = 0
    mode: str = "unknown"


class PositionSizer:
    """Order sizing engine — Kelly criterion + fixed-fractional.

    Methods:
        kelly_size() — optimal fraction from win rate + payoff ratio
        fixed_fractional() — simple % of account per trade
        max_contracts() — account-aware contract count
    """

    def kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly criterion: f* = (p*b - q) / b

        Args:
            win_rate: probability of winning (0-1)
            avg_win: average win amount
            avg_loss: average loss amount (positive number)

        Returns:
            Optimal fraction of capital to risk (0-1)
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0

        b = avg_win / avg_loss  # payoff ratio
        p = win_rate
        q = 1 - p

        kelly = (p * b - q) / b
        # Half-Kelly for safety
        return max(0.0, min(0.25, kelly * 0.5))

    def fixed_fractional(self, account_size: float, risk_pct: float,
                         premium: float, contracts_multiplier: int = 100) -> int:
        """Fixed-fractional position sizing.

        Args:
            account_size: total account value
            risk_pct: max % of account to risk (e.g. 0.01 = 1%)
            premium: per-contract premium
            contracts_multiplier: options multiplier (100 for equity options)

        Returns:
            Number of contracts to buy
        """
        max_risk = account_size * risk_pct
        cost_per_contract = premium * contracts_multiplier
        if cost_per_contract <= 0:
            return 0
        return max(1, int(max_risk / cost_per_contract))

    def max_contracts(self, account_size: float, risk_pct: float,
                      premium: float, max_portfolio_pct: float = 0.20,
                      existing_exposure: float = 0.0) -> int:
        """Account-aware sizing with portfolio-level constraints.

        Args:
            account_size: total account value
            risk_pct: per-trade risk %
            premium: per-contract premium
            max_portfolio_pct: max total put exposure %
            existing_exposure: current put exposure $

        Returns:
            Capped contract count
        """
        # Per-trade limit
        per_trade = self.fixed_fractional(account_size, risk_pct, premium)

        # Portfolio-level cap
        remaining_budget = (account_size * max_portfolio_pct) - existing_exposure
        if remaining_budget <= 0:
            return 0
        portfolio_cap = int(remaining_budget / (premium * 100)) if premium > 0 else 0

        return max(0, min(per_trade, portfolio_cap))


class ExecutionEngine:
    """IBKR execution engine for MATRIX MAXIMIZER.

    Modes:
        DRY_RUN  — Log orders, no submission (default)
        PAPER    — Submit to IBKR paper account (port 7497)
        LIVE     — Submit to IBKR live account (port 7496)

    Usage:
        engine = ExecutionEngine(mode=ExecutionMode.PAPER)
        result = engine.buy_put("SPY", 560, "2026-04-18", 2, 3.50)
        positions = engine.get_positions()
        engine.close_position(position_id)
    """

    def __init__(self, mode: ExecutionMode = ExecutionMode.DRY_RUN,
                 account_size: float = 920.0) -> None:
        self.mode = mode
        self.account_size = account_size
        self.sizer = PositionSizer()
        self._positions: Dict[str, TrackedPosition] = {}
        self._order_history: List[Dict[str, Any]] = []
        self._ibkr = None

        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._load_positions()

        if mode != ExecutionMode.DRY_RUN:
            self._init_ibkr()

    def _init_ibkr(self) -> None:
        """Initialize IBKR connector."""
        try:
            from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
            env_port = int(os.getenv("IBKR_PORT", "7497"))
            port = 7497 if self.mode == ExecutionMode.PAPER else env_port
            self._ibkr = IBKRConnector(
                host=os.getenv("IBKR_HOST", "127.0.0.1"),
                port=port,
                client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
                account=os.getenv("IBKR_ACCOUNT", ""),
            )
            logger.info("IBKR connector initialized — port %d (%s)", port, self.mode.value)
        except Exception as exc:
            logger.error("Failed to init IBKR: %s", exc)
            self._ibkr = None

    def _run_async(self, coro):
        """Run async coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, coro).result(timeout=30)
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # ═══════════════════════════════════════════════════════════════════════
    # ORDER OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════

    def buy_put(self, ticker: str, strike: float, expiry: str,
                contracts: int, premium: float,
                delta: float = 0.0, limit_price: Optional[float] = None) -> OrderResult:
        """Buy put option(s).

        Args:
            ticker: underlying symbol
            strike: put strike price
            expiry: expiry date (YYYY-MM-DD)
            contracts: number of contracts
            premium: expected premium per share
            delta: entry delta for tracking
            limit_price: limit order price (default: premium)

        Returns:
            OrderResult with success status
        """
        cost = contracts * premium * 100

        # Validate against account
        if cost > self.account_size * 0.20:
            return OrderResult(
                success=False,
                message=f"Order cost ${cost:.0f} exceeds 20% of account ${self.account_size:.0f}",
            )

        # Generate position ID
        pos_id = f"MM_{ticker}_{strike}P_{expiry}_{datetime.utcnow().strftime('%H%M%S')}"

        if self.mode == ExecutionMode.DRY_RUN:
            # Simulate fill
            pos = TrackedPosition(
                position_id=pos_id,
                ticker=ticker,
                strike=strike,
                expiry=expiry,
                contracts=contracts,
                entry_premium=premium,
                entry_date=datetime.utcnow().strftime("%Y-%m-%d"),
                entry_delta=delta,
                cost_basis=cost,
                current_premium=premium,
                current_delta=delta,
            )
            self._positions[pos_id] = pos
            self._save_positions()

            logger.info("DRY RUN: BUY %d %s $%.0fP %s @ $%.2f = $%.0f",
                        contracts, ticker, strike, expiry, premium, cost)

            return OrderResult(
                success=True,
                order_id=pos_id,
                state="filled",
                fill_price=premium,
                filled_qty=contracts,
                message="DRY RUN — simulated fill",
                timestamp=datetime.utcnow().isoformat(),
            )

        # IBKR submission
        if not self._ibkr:
            return OrderResult(success=False, message="IBKR not connected")

        try:
            order_price = limit_price or premium
            symbol = f"{ticker}{expiry.replace('-', '')}{int(strike * 1000):08d}P"

            result = self._run_async(
                self._ibkr.create_order(
                    symbol=symbol,
                    side="buy",
                    order_type="limit",
                    quantity=contracts,
                    price=order_price,
                )
            )

            pos = TrackedPosition(
                position_id=pos_id,
                ticker=ticker,
                strike=strike,
                expiry=expiry,
                contracts=contracts,
                entry_premium=order_price,
                entry_date=datetime.utcnow().strftime("%Y-%m-%d"),
                entry_delta=delta,
                cost_basis=contracts * order_price * 100,
                ibkr_order_id=str(getattr(result, "order_id", "")),
            )
            self._positions[pos_id] = pos
            self._save_positions()

            logger.info("%s: BUY %d %s $%.0fP %s @ $%.2f",
                        self.mode.value.upper(), contracts, ticker, strike, expiry, order_price)

            return OrderResult(
                success=True,
                order_id=pos_id,
                state=str(getattr(result, "status", "submitted")),
                fill_price=order_price,
                filled_qty=contracts,
                timestamp=datetime.utcnow().isoformat(),
            )

        except Exception as exc:
            logger.error("IBKR order failed: %s", exc)
            return OrderResult(success=False, message=str(exc))

    def close_position(self, position_id: str,
                       limit_price: Optional[float] = None) -> OrderResult:
        """Close an open position by selling the puts."""
        if position_id not in self._positions:
            return OrderResult(success=False, message=f"Position {position_id} not found")

        pos = self._positions[position_id]
        if not pos.is_open:
            return OrderResult(success=False, message=f"Position already {pos.state}")

        if self.mode == ExecutionMode.DRY_RUN:
            sell_price = limit_price or pos.current_premium
            pos.realized_pnl = (sell_price - pos.entry_premium) * pos.contracts * 100
            pos.state = "closed"
            self._save_positions()

            logger.info("DRY RUN: CLOSE %s @ $%.2f — P&L: $%.2f",
                        position_id, sell_price, pos.realized_pnl)

            return OrderResult(
                success=True,
                order_id=position_id,
                state="closed",
                fill_price=sell_price,
                filled_qty=pos.contracts,
                message=f"DRY RUN — P&L: ${pos.realized_pnl:.2f}",
                timestamp=datetime.utcnow().isoformat(),
            )

        if not self._ibkr:
            return OrderResult(success=False, message="IBKR not connected")

        try:
            sell_price = limit_price or pos.current_premium
            symbol = f"{pos.ticker}{pos.expiry.replace('-', '')}{int(pos.strike * 1000):08d}P"

            result = self._run_async(
                self._ibkr.create_order(
                    symbol=symbol,
                    side="sell",
                    order_type="limit",
                    quantity=pos.contracts,
                    price=sell_price,
                )
            )

            pos.realized_pnl = (sell_price - pos.entry_premium) * pos.contracts * 100
            pos.state = "closed"
            self._save_positions()

            return OrderResult(
                success=True,
                order_id=position_id,
                state="closed",
                fill_price=sell_price,
                filled_qty=pos.contracts,
                timestamp=datetime.utcnow().isoformat(),
            )

        except Exception as exc:
            logger.error("IBKR close failed: %s", exc)
            return OrderResult(success=False, message=str(exc))

    def execute_picks(self, picks: List[Dict[str, Any]],
                      mandate_risk_pct: float = 0.01,
                      max_positions: int = 5) -> List[OrderResult]:
        """Execute top picks from scanner output.

        Uses PositionSizer to determine contract count per pick.
        Respects portfolio-level exposure limits.
        """
        results: List[OrderResult] = []
        open_count = sum(1 for p in self._positions.values() if p.is_open)
        remaining_slots = max_positions - open_count

        if remaining_slots <= 0:
            logger.info("Max positions reached (%d) — no new orders", max_positions)
            return results

        existing_exposure = sum(
            p.cost_basis for p in self._positions.values() if p.is_open
        )

        for pick in picks[:remaining_slots]:
            ticker = pick.get("ticker", "")
            strike = pick.get("strike", 0)
            expiry = pick.get("expiry", "")
            premium = pick.get("premium", 0)
            delta = pick.get("delta", 0)

            contracts = self.sizer.max_contracts(
                account_size=self.account_size,
                risk_pct=mandate_risk_pct,
                premium=premium,
                existing_exposure=existing_exposure,
            )

            if contracts <= 0:
                logger.info("Sizing returned 0 contracts for %s — skipping", ticker)
                continue

            result = self.buy_put(ticker, strike, expiry, contracts, premium, delta)
            results.append(result)

            if result.success:
                existing_exposure += contracts * premium * 100

        return results

    # ═══════════════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════

    def get_positions(self, include_closed: bool = False) -> List[TrackedPosition]:
        """Get all tracked positions."""
        if include_closed:
            return list(self._positions.values())
        return [p for p in self._positions.values() if p.is_open]

    def get_account_snapshot(self) -> AccountSnapshot:
        """Current account status."""
        open_positions = self.get_positions()
        put_exposure = sum(p.cost_basis for p in open_positions)
        unrealized = sum(p.unrealized_pnl for p in open_positions)
        realized = sum(p.realized_pnl for p in self._positions.values() if not p.is_open)

        snap = AccountSnapshot(
            total_value=self.account_size + unrealized + realized,
            cash=self.account_size - put_exposure,
            buying_power=self.account_size - put_exposure,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            put_exposure=put_exposure,
            positions_count=len(open_positions),
            mode=self.mode.value,
        )

        # Try to get real IBKR data
        if self._ibkr and self.mode != ExecutionMode.DRY_RUN:
            try:
                ibkr_summary = self._run_async(self._ibkr.get_account_summary())
                if ibkr_summary:
                    snap.total_value = float(ibkr_summary.get("NetLiquidation", snap.total_value))
                    snap.cash = float(ibkr_summary.get("TotalCashValue", snap.cash))
                    snap.buying_power = float(ibkr_summary.get("BuyingPower", snap.buying_power))
                    snap.maintenance_margin = float(ibkr_summary.get("MaintMarginReq", 0))
            except Exception as exc:
                logger.warning("IBKR account summary failed: %s", exc)

        return snap

    def update_positions(self, prices: Dict[str, float]) -> None:
        """Update all open positions with current prices via BS repricing."""
        from strategies.matrix_maximizer.greeks import BlackScholesEngine
        from strategies.matrix_maximizer.core import ASSET_VOLATILITIES, Asset

        bs = BlackScholesEngine()
        today = datetime.utcnow().date()

        for pos in self._positions.values():
            if not pos.is_open:
                continue

            spot = prices.get(pos.ticker, 0)
            if spot <= 0:
                continue

            # Calculate DTE
            try:
                exp_date = datetime.strptime(pos.expiry, "%Y-%m-%d").date()
                dte = (exp_date - today).days
            except (ValueError, TypeError):
                dte = 30

            if dte <= 0:
                pos.state = "expired"
                intrinsic = max(0, pos.strike - spot)
                pos.realized_pnl = (intrinsic - pos.entry_premium) * pos.contracts * 100
                continue

            sigma = ASSET_VOLATILITIES.get(Asset(pos.ticker), 0.25)
            greeks = bs.price_put(spot, pos.strike, dte, sigma)

            pos.current_premium = greeks.price
            pos.current_delta = greeks.delta
            pos.market_value = greeks.price * pos.contracts * 100
            pos.unrealized_pnl = (greeks.price - pos.entry_premium) * pos.contracts * 100
            pos.pnl_pct = (greeks.price - pos.entry_premium) / pos.entry_premium if pos.entry_premium > 0 else 0
            pos.days_held = (today - datetime.strptime(pos.entry_date, "%Y-%m-%d").date()).days

        self._save_positions()

    # ═══════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════

    def _save_positions(self) -> None:
        """Persist positions to JSON."""
        data = {}
        for pid, pos in self._positions.items():
            data[pid] = {
                "position_id": pos.position_id,
                "ticker": pos.ticker,
                "strike": pos.strike,
                "expiry": pos.expiry,
                "contracts": pos.contracts,
                "entry_premium": pos.entry_premium,
                "entry_date": pos.entry_date,
                "entry_delta": pos.entry_delta,
                "cost_basis": pos.cost_basis,
                "current_premium": pos.current_premium,
                "current_delta": pos.current_delta,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "realized_pnl": pos.realized_pnl,
                "pnl_pct": pos.pnl_pct,
                "days_held": pos.days_held,
                "state": pos.state,
                "ibkr_order_id": pos.ibkr_order_id,
                "notes": pos.notes,
            }

        path = _STATE_DIR / "positions.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_positions(self) -> None:
        """Load positions from JSON."""
        path = _STATE_DIR / "positions.json"
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for pid, pdata in data.items():
                self._positions[pid] = TrackedPosition(**pdata)
            logger.info("Loaded %d positions from disk", len(self._positions))
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.warning("Failed to load positions: %s", exc)

    def print_positions(self) -> str:
        """Human-readable position summary."""
        positions = self.get_positions(include_closed=True)
        if not positions:
            return "  No positions tracked"

        lines = [
            "  Ticker  Strike  Expiry      Contracts  Entry    Current  P&L      Status",
            "  " + "-" * 76,
        ]
        for p in positions:
            pnl_str = f"${p.unrealized_pnl:+.0f}" if p.is_open else f"${p.realized_pnl:+.0f}"
            lines.append(
                f"  {p.ticker:<7s} ${p.strike:<7.0f} {p.expiry}  "
                f"{p.contracts:<10d} ${p.entry_premium:<7.2f} "
                f"${p.current_premium:<7.2f} {pnl_str:<8s} {p.state}"
            )
        return "\n".join(lines)
