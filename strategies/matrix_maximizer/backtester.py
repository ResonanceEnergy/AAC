"""
MATRIX MAXIMIZER — Backtester
================================
Historical simulation and performance attribution:
  - Run Monte Carlo cycles against past regimes
  - Track hypothetical P&L over date ranges
  - Attribute returns to oil betas, deltas, timing
  - Compare strategies (single puts vs spreads)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Single simulated trade in backtesting."""
    date: str
    ticker: str
    strike: float
    entry_premium: float
    exit_premium: float
    contracts: int
    pnl: float
    pnl_pct: float
    dte_at_entry: int
    exit_reason: str            # "expiry", "stop_loss", "take_profit", "roll"
    regime_at_entry: str
    oil_at_entry: float
    vix_at_entry: float


@dataclass
class BacktestResult:
    """Complete backtest output."""
    start_date: str
    end_date: str
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    avg_pnl_per_trade: float
    avg_winner: float
    avg_loser: float
    profit_factor: float        # gross profit / gross loss
    sharpe_ratio: float
    max_consecutive_losses: int
    best_trade: float
    worst_trade: float
    equity_curve: List[float] = field(default_factory=list)
    trades: List[BacktestTrade] = field(default_factory=list)
    attribution: Optional[Dict[str, float]] = None

    def print_card(self) -> str:
        lines = [
            f"  BACKTEST: {self.start_date} → {self.end_date}",
            f"  Trades: {self.total_trades} | W/L: {self.winners}/{self.losers} "
            f"({self.win_rate:.0%})",
            f"  Total P&L: ${self.total_pnl:+,.0f} | Avg/Trade: ${self.avg_pnl_per_trade:+.0f}",
            f"  Avg Win: ${self.avg_winner:+.0f} | Avg Loss: ${self.avg_loser:+.0f}",
            f"  Profit Factor: {self.profit_factor:.2f}x | Sharpe: {self.sharpe_ratio:.2f}",
            f"  Max DD: ${self.max_drawdown:+,.0f} | Max Consec. Losses: {self.max_consecutive_losses}",
            f"  Best: ${self.best_trade:+,.0f} | Worst: ${self.worst_trade:+,.0f}",
        ]
        if self.attribution:
            lines.append("  ATTRIBUTION:")
            for k, v in self.attribution.items():
                lines.append(f"    {k}: ${v:+,.0f}")
        return "\n".join(lines)


class MatrixBacktester:
    """Historical backtester for MATRIX MAXIMIZER strategies.

    Usage:
        bt = MatrixBacktester()
        result = bt.backtest(
            scenarios=scenarios,     # List of historical regime snapshots
            strategy="single_put",   # or "bear_spread"
            initial_capital=920.0,
        )
        print(result.print_card())
    """

    def __init__(self) -> None:
        from strategies.matrix_maximizer.core import (
            ASSET_OIL_BETAS,
            ASSET_VOLATILITIES,
            DEFAULT_PRICES,
            Asset,
            ScenarioWeights,
        )
        from strategies.matrix_maximizer.greeks import BlackScholesEngine
        self.bs = BlackScholesEngine()
        self._asset_vols = ASSET_VOLATILITIES
        self._oil_betas = ASSET_OIL_BETAS

    def backtest(self, scenarios: List[Dict[str, Any]],
                 strategy: str = "single_put",
                 initial_capital: float = 920.0,
                 risk_per_trade: float = 0.01,
                 target_delta: float = -0.35,
                 dte: int = 30,
                 stop_loss_pct: float = -0.50,
                 take_profit_pct: float = 1.0) -> BacktestResult:
        """Run a backtest across historical scenarios.

        Each scenario dict should have:
            date, vix, oil_price, regime, prices (dict of ticker→price)

        Args:
            scenarios: List of daily market snapshots
            strategy: "single_put" or "bear_spread"
            initial_capital: starting capital
            risk_per_trade: fraction of capital per trade
            target_delta: put delta target
            dte: days to expiry for new puts
            stop_loss_pct: exit if loss exceeds this %
            take_profit_pct: exit if gain exceeds this %
        """
        from strategies.matrix_maximizer.core import ASSET_VOLATILITIES, Asset

        trades: List[BacktestTrade] = []
        equity = initial_capital
        equity_curve = [equity]
        peak_equity = equity
        max_drawdown = 0.0

        open_trades: List[Dict[str, Any]] = []

        tickers = ["SPY", "QQQ", "JETS", "KRE", "HYG"]

        for i, snap in enumerate(scenarios):
            date = snap.get("date", "")
            vix = snap.get("vix", 22)
            oil = snap.get("oil_price", 80)
            regime = snap.get("regime", "normal")
            prices = snap.get("prices", {})

            # Update open trades
            closed_this_step: List[int] = []
            for j, ot in enumerate(open_trades):
                spot = prices.get(ot["ticker"], ot["entry_spot"])
                sigma = ASSET_VOLATILITIES.get(Asset(ot["ticker"]), 0.25)
                remaining_dte = max(1, ot["dte"] - (i - ot["entry_idx"]))

                greeks = self.bs.price_put(spot, ot["strike"], remaining_dte, sigma)
                pnl_pct = (greeks.price - ot["entry_premium"]) / ot["entry_premium"] if ot["entry_premium"] > 0 else 0

                exit_reason = None
                exit_premium = greeks.price

                if remaining_dte <= 1:
                    exit_reason = "expiry"
                    exit_premium = max(0, ot["strike"] - spot)
                elif pnl_pct <= stop_loss_pct:
                    exit_reason = "stop_loss"
                elif pnl_pct >= take_profit_pct:
                    exit_reason = "take_profit"

                if exit_reason:
                    pnl = (exit_premium - ot["entry_premium"]) * ot["contracts"] * 100
                    trades.append(BacktestTrade(
                        date=date,
                        ticker=ot["ticker"],
                        strike=ot["strike"],
                        entry_premium=ot["entry_premium"],
                        exit_premium=exit_premium,
                        contracts=ot["contracts"],
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        dte_at_entry=ot["dte"],
                        exit_reason=exit_reason,
                        regime_at_entry=ot["regime"],
                        oil_at_entry=ot["oil"],
                        vix_at_entry=ot["vix"],
                    ))
                    equity += pnl
                    closed_this_step.append(j)

            # Remove closed trades (reverse order to preserve indices)
            for j in sorted(closed_this_step, reverse=True):
                open_trades.pop(j)

            # Open new trades if no open positions and enough capital
            if not open_trades and equity > 50:
                for ticker in tickers[:2]:  # Top 2 tickers per day
                    spot = prices.get(ticker, 0)
                    if spot <= 0:
                        continue

                    sigma = ASSET_VOLATILITIES.get(Asset(ticker), 0.25)

                    # Adjust sigma for VIX
                    if vix > 30:
                        sigma *= 1.3
                    elif vix < 18:
                        sigma *= 0.85

                    strike = self.bs.find_strike_for_delta(spot, target_delta, dte, sigma)
                    greeks = self.bs.price_put(spot, strike, dte, sigma)

                    if greeks.price <= 0:
                        continue

                    max_cost = equity * risk_per_trade
                    contracts = max(1, int(max_cost / (greeks.price * 100)))
                    cost = contracts * greeks.price * 100

                    if cost > equity * 0.20:
                        contracts = max(1, int(equity * 0.20 / (greeks.price * 100)))

                    open_trades.append({
                        "ticker": ticker,
                        "strike": strike,
                        "entry_premium": greeks.price,
                        "entry_spot": spot,
                        "contracts": contracts,
                        "dte": dte,
                        "entry_idx": i,
                        "regime": regime,
                        "oil": oil,
                        "vix": vix,
                    })

            equity_curve.append(equity)
            peak_equity = max(peak_equity, equity)
            dd = equity - peak_equity
            max_drawdown = min(max_drawdown, dd)

        # Compute stats
        return self._compute_stats(trades, equity_curve, max_drawdown,
                                    scenarios[0].get("date", "") if scenarios else "",
                                    scenarios[-1].get("date", "") if scenarios else "")

    def generate_historical_scenarios(self, days: int = 90,
                                       base_vix: float = 22,
                                       base_oil: float = 80,
                                       crisis_at: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate synthetic historical scenarios for backtesting.

        Creates day-by-day snapshots with random walks + optional crisis event.
        """
        import random

        from strategies.matrix_maximizer.core import DEFAULT_PRICES

        scenarios = []
        vix = base_vix
        oil = base_oil
        prices = dict(DEFAULT_PRICES)

        for day in range(days):
            date = (datetime.utcnow() - timedelta(days=days - day)).strftime("%Y-%m-%d")

            # Random walk
            vix_change = random.gauss(0, 1.5)
            oil_change = random.gauss(0, 1.0)
            vix = max(10, vix + vix_change)
            oil = max(30, oil + oil_change)

            # Crisis injection
            if crisis_at and day == crisis_at:
                vix += 15
                oil += 20
                for t in prices:
                    prices[t] *= 0.93  # 7% crash

            # Normal daily moves
            for t in prices:
                daily_return = random.gauss(0, 0.01)  # 1% daily vol
                prices[t] *= (1 + daily_return)

            regime = "normal"
            if vix > 35:
                regime = "vol_shock_active"
            elif vix > 28:
                regime = "vol_shock_armed"
            elif oil > 100:
                regime = "stagflation"

            scenarios.append({
                "date": date,
                "vix": round(vix, 2),
                "oil_price": round(oil, 2),
                "regime": regime,
                "prices": {k: round(v, 2) for k, v in prices.items()},
            })

        return scenarios

    def _compute_stats(self, trades: List[BacktestTrade],
                       equity_curve: List[float],
                       max_drawdown: float,
                       start_date: str, end_date: str) -> BacktestResult:
        """Compute performance statistics from trades."""
        if not trades:
            return BacktestResult(
                start_date=start_date, end_date=end_date,
                total_trades=0, winners=0, losers=0, win_rate=0,
                total_pnl=0, max_drawdown=0, avg_pnl_per_trade=0,
                avg_winner=0, avg_loser=0, profit_factor=0,
                sharpe_ratio=0, max_consecutive_losses=0,
                best_trade=0, worst_trade=0,
                equity_curve=equity_curve, trades=trades,
            )

        pnls = [t.pnl for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        # Max consecutive losses
        max_consec = 0
        current_consec = 0
        for p in pnls:
            if p <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0.01

        # Sharpe ratio (annualized, assuming ~252 trading days)
        import math
        avg_daily = sum(pnls) / len(pnls) if pnls else 0
        if len(pnls) > 1:
            std = (sum((p - avg_daily) ** 2 for p in pnls) / (len(pnls) - 1)) ** 0.5
            sharpe = (avg_daily / std) * math.sqrt(252) if std > 0 else 0
        else:
            sharpe = 0

        # Attribution
        attribution = self._compute_attribution(trades)

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_trades=len(trades),
            winners=len(winners),
            losers=len(losers),
            win_rate=len(winners) / len(trades),
            total_pnl=sum(pnls),
            max_drawdown=max_drawdown,
            avg_pnl_per_trade=sum(pnls) / len(pnls),
            avg_winner=sum(winners) / len(winners) if winners else 0,
            avg_loser=sum(losers) / len(losers) if losers else 0,
            profit_factor=gross_profit / gross_loss,
            sharpe_ratio=sharpe,
            max_consecutive_losses=max_consec,
            best_trade=max(pnls),
            worst_trade=min(pnls),
            equity_curve=equity_curve,
            trades=trades,
            attribution=attribution,
        )

    def _compute_attribution(self, trades: List[BacktestTrade]) -> Dict[str, float]:
        """Attribute P&L to market factors."""
        from strategies.matrix_maximizer.core import ASSET_OIL_BETAS, Asset

        attr: Dict[str, float] = {
            "high_vix_trades": 0.0,
            "high_oil_trades": 0.0,
            "crisis_regime_trades": 0.0,
            "normal_regime_trades": 0.0,
            "by_exit_reason": {},
        }

        exit_reasons: Dict[str, float] = {}

        for t in trades:
            if t.vix_at_entry > 28:
                attr["high_vix_trades"] += t.pnl
            if t.oil_at_entry > 95:
                attr["high_oil_trades"] += t.pnl
            if t.regime_at_entry in ("vol_shock_active", "vol_shock_armed", "credit_stress"):
                attr["crisis_regime_trades"] += t.pnl
            else:
                attr["normal_regime_trades"] += t.pnl

            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + t.pnl

        attr["by_exit_reason"] = exit_reasons
        return attr
