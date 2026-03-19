"""
MATRIX MAXIMIZER — Advanced Multi-Leg Strategies
===================================================
Beyond single-leg puts:
  - Bear put spreads (defined risk verticals)
  - Collars (protective puts + covered calls)
  - Straddles/strangles (volatility plays)
  - Iron condors/butterflies (range-bound)
  - Rolling calendar (weekly put rotation)
  - Sector rotation (dynamic ticker selection)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StrategyLeg:
    """Single leg of a multi-leg options strategy."""
    ticker: str
    strike: float
    expiry: str
    option_type: str   # "put" or "call"
    side: str          # "buy" or "sell"
    contracts: int
    premium: float
    delta: float = 0.0
    greeks: Optional[Dict[str, float]] = None


@dataclass
class MultiLegStrategy:
    """Complete multi-leg strategy with risk/reward profile."""
    name: str
    strategy_type: str  # "bear_put_spread", "collar", "straddle", etc.
    ticker: str
    legs: List[StrategyLeg]
    net_debit: float           # positive = debit, negative = credit
    max_profit: float
    max_loss: float
    breakeven: float
    reward_risk_ratio: float
    net_delta: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    thesis: str = ""
    score: float = 0.0

    @property
    def total_contracts(self) -> int:
        return sum(l.contracts for l in self.legs)

    @property
    def total_cost(self) -> float:
        return abs(self.net_debit) * self.legs[0].contracts * 100 if self.legs else 0

    def print_card(self) -> str:
        lines = [
            f"  {self.name} — {self.ticker} ({self.strategy_type})",
            f"    Legs: {len(self.legs)} | Net: ${self.net_debit:+.2f}",
            f"    Max Profit: ${self.max_profit:.0f} | Max Loss: ${self.max_loss:.0f}",
            f"    Breakeven: ${self.breakeven:.2f} | R:R = {self.reward_risk_ratio:.1f}x",
            f"    Net Greeks: Δ={self.net_delta:.3f} Θ={self.net_theta:.3f} V={self.net_vega:.3f}",
        ]
        for i, leg in enumerate(self.legs, 1):
            lines.append(
                f"    Leg {i}: {leg.side.upper()} {leg.contracts}x "
                f"{leg.ticker} ${leg.strike:.0f}{leg.option_type[0].upper()} "
                f"{leg.expiry} @ ${leg.premium:.2f}"
            )
        if self.thesis:
            lines.append(f"    Thesis: {self.thesis}")
        return "\n".join(lines)


class AdvancedStrategyEngine:
    """Generates multi-leg strategies from MATRIX MAXIMIZER context.

    Usage:
        engine = AdvancedStrategyEngine(bs_engine)
        spreads = engine.bear_put_spreads("SPY", 560, 0.25, 30, mandate)
        collars = engine.collars("QQQ", 480, 100, 0.22, 30)
        straddles = engine.straddles("SPY", 560, 0.25, 7)
    """

    def __init__(self, bs_engine=None) -> None:
        if bs_engine is None:
            from strategies.matrix_maximizer.greeks import BlackScholesEngine
            bs_engine = BlackScholesEngine()
        self.bs = bs_engine

    # ═══════════════════════════════════════════════════════════════════════
    # BEAR PUT SPREADS
    # ═══════════════════════════════════════════════════════════════════════

    def bear_put_spreads(self, ticker: str, spot: float, sigma: float,
                         dte: int, mandate=None,
                         width_pcts: Optional[List[float]] = None) -> List[MultiLegStrategy]:
        """Generate bear put spread candidates.

        Bear put spread = BUY higher-strike put + SELL lower-strike put.
        Defined risk: max loss = net debit, max profit = width - debit.
        """
        if width_pcts is None:
            width_pcts = [0.02, 0.03, 0.05]  # 2%, 3%, 5% strike width

        strategies: List[MultiLegStrategy] = []

        # Generate strikes at OTM levels
        otm_levels = [0.03, 0.05, 0.07, 0.10]

        for otm in otm_levels:
            long_strike = round(spot * (1 - otm), 0)
            long_greeks = self.bs.price_put(spot, long_strike, dte, sigma)

            for width_pct in width_pcts:
                short_strike = round(long_strike - spot * width_pct, 0)
                if short_strike <= 0:
                    continue

                short_greeks = self.bs.price_put(spot, short_strike, dte, sigma)

                net_debit = long_greeks.price - short_greeks.price
                if net_debit <= 0:
                    continue

                width = long_strike - short_strike
                max_profit = (width - net_debit) * 100
                max_loss = net_debit * 100
                breakeven = long_strike - net_debit

                if max_loss <= 0:
                    continue

                rr = max_profit / max_loss

                strategies.append(MultiLegStrategy(
                    name=f"{ticker} {long_strike}/{short_strike}P spread",
                    strategy_type="bear_put_spread",
                    ticker=ticker,
                    legs=[
                        StrategyLeg(
                            ticker=ticker, strike=long_strike, expiry="",
                            option_type="put", side="buy", contracts=1,
                            premium=long_greeks.price, delta=long_greeks.delta,
                        ),
                        StrategyLeg(
                            ticker=ticker, strike=short_strike, expiry="",
                            option_type="put", side="sell", contracts=1,
                            premium=short_greeks.price, delta=short_greeks.delta,
                        ),
                    ],
                    net_debit=net_debit,
                    max_profit=max_profit,
                    max_loss=max_loss,
                    breakeven=breakeven,
                    reward_risk_ratio=rr,
                    net_delta=long_greeks.delta - short_greeks.delta,
                    net_theta=long_greeks.theta - short_greeks.theta,
                    net_vega=long_greeks.vega - short_greeks.vega,
                    thesis=f"Bear view: {ticker} below ${breakeven:.0f} by expiry",
                    score=rr * 25 + (1 - otm) * 50,  # Higher score for tighter, better R:R
                ))

        strategies.sort(key=lambda s: s.score, reverse=True)
        return strategies[:10]

    # ═══════════════════════════════════════════════════════════════════════
    # COLLARS
    # ═══════════════════════════════════════════════════════════════════════

    def collars(self, ticker: str, spot: float, shares: int,
                sigma: float, dte: int) -> List[MultiLegStrategy]:
        """Generate collar strategies (own shares + buy put + sell call).

        Zero-cost collar: call credit offsets put debit.
        """
        strategies: List[MultiLegStrategy] = []
        contracts = shares // 100

        if contracts <= 0:
            return strategies

        put_otm_levels = [0.03, 0.05, 0.07]
        call_otm_levels = [0.03, 0.05, 0.07]

        for put_otm in put_otm_levels:
            put_strike = round(spot * (1 - put_otm), 0)
            put_greeks = self.bs.price_put(spot, put_strike, dte, sigma)

            for call_otm in call_otm_levels:
                call_strike = round(spot * (1 + call_otm), 0)
                # Price call using put-call parity approximation
                call_price = max(0.01, put_greeks.price * 0.8 * (1 + call_otm - put_otm))

                net_debit = put_greeks.price - call_price
                max_loss_per_share = spot - put_strike + net_debit
                max_profit_per_share = call_strike - spot - net_debit

                if max_loss_per_share <= 0:
                    continue

                strategies.append(MultiLegStrategy(
                    name=f"{ticker} {put_strike}P/{call_strike}C collar",
                    strategy_type="collar",
                    ticker=ticker,
                    legs=[
                        StrategyLeg(
                            ticker=ticker, strike=put_strike, expiry="",
                            option_type="put", side="buy", contracts=contracts,
                            premium=put_greeks.price, delta=put_greeks.delta,
                        ),
                        StrategyLeg(
                            ticker=ticker, strike=call_strike, expiry="",
                            option_type="call", side="sell", contracts=contracts,
                            premium=call_price, delta=0.3,
                        ),
                    ],
                    net_debit=net_debit,
                    max_profit=max_profit_per_share * contracts * 100,
                    max_loss=max_loss_per_share * contracts * 100,
                    breakeven=spot + net_debit,
                    reward_risk_ratio=max_profit_per_share / max_loss_per_share if max_loss_per_share > 0 else 0,
                    net_delta=put_greeks.delta + 0.3,  # approximate
                    thesis=f"Protection at ${put_strike}, capped at ${call_strike}",
                    score=50 + (1 / max(0.01, abs(net_debit))) * 10,
                ))

        strategies.sort(key=lambda s: s.score, reverse=True)
        return strategies[:6]

    # ═══════════════════════════════════════════════════════════════════════
    # STRADDLES & STRANGLES
    # ═══════════════════════════════════════════════════════════════════════

    def straddles(self, ticker: str, spot: float, sigma: float,
                  dte: int) -> List[MultiLegStrategy]:
        """Generate straddle (ATM) and strangle (OTM) strategies.

        Used for high-VIX events: earnings, FOMC, geopolitical escalation.
        """
        strategies: List[MultiLegStrategy] = []

        # ATM Straddle
        atm_strike = round(spot, 0)
        put_greeks = self.bs.price_put(spot, atm_strike, dte, sigma)
        call_price = put_greeks.price * 1.05  # Approximate ATM call

        total_premium = put_greeks.price + call_price
        breakeven_down = atm_strike - total_premium
        breakeven_up = atm_strike + total_premium

        strategies.append(MultiLegStrategy(
            name=f"{ticker} ${atm_strike} straddle",
            strategy_type="straddle",
            ticker=ticker,
            legs=[
                StrategyLeg(
                    ticker=ticker, strike=atm_strike, expiry="",
                    option_type="put", side="buy", contracts=1,
                    premium=put_greeks.price, delta=put_greeks.delta,
                ),
                StrategyLeg(
                    ticker=ticker, strike=atm_strike, expiry="",
                    option_type="call", side="buy", contracts=1,
                    premium=call_price, delta=0.50,
                ),
            ],
            net_debit=total_premium,
            max_profit=spot * 100,  # Theoretically unlimited on call side
            max_loss=total_premium * 100,
            breakeven=breakeven_down,  # Lower breakeven
            reward_risk_ratio=spot / total_premium if total_premium > 0 else 0,
            net_delta=put_greeks.delta + 0.50,
            thesis=f"Expects big move: breakevens ${breakeven_down:.0f}↓ / ${breakeven_up:.0f}↑",
        ))

        # OTM Strangles at various widths
        for width_pct in [0.03, 0.05, 0.07]:
            put_strike = round(spot * (1 - width_pct), 0)
            call_strike = round(spot * (1 + width_pct), 0)

            p_greeks = self.bs.price_put(spot, put_strike, dte, sigma)
            c_price = max(0.01, p_greeks.price * 0.9)

            total = p_greeks.price + c_price
            be_down = put_strike - total
            be_up = call_strike + total

            strategies.append(MultiLegStrategy(
                name=f"{ticker} ${put_strike}P/${call_strike}C strangle",
                strategy_type="strangle",
                ticker=ticker,
                legs=[
                    StrategyLeg(
                        ticker=ticker, strike=put_strike, expiry="",
                        option_type="put", side="buy", contracts=1,
                        premium=p_greeks.price, delta=p_greeks.delta,
                    ),
                    StrategyLeg(
                        ticker=ticker, strike=call_strike, expiry="",
                        option_type="call", side="buy", contracts=1,
                        premium=c_price, delta=0.30,
                    ),
                ],
                net_debit=total,
                max_profit=spot * 100,
                max_loss=total * 100,
                breakeven=be_down,
                reward_risk_ratio=spot / total if total > 0 else 0,
                net_delta=p_greeks.delta + 0.30,
                thesis=f"Wider move: breakevens ${be_down:.0f}↓ / ${be_up:.0f}↑",
                score=100 - width_pct * 500,
            ))

        return strategies

    # ═══════════════════════════════════════════════════════════════════════
    # IRON CONDORS & BUTTERFLIES
    # ═══════════════════════════════════════════════════════════════════════

    def iron_condors(self, ticker: str, spot: float, sigma: float,
                     dte: int) -> List[MultiLegStrategy]:
        """Generate iron condor candidates for range-bound markets.

        Iron Condor = bear call spread + bull put spread (credit strategy).
        Profit if price stays in range.
        """
        strategies: List[MultiLegStrategy] = []

        for wing_width in [0.03, 0.05]:
            for spread_width_pct in [0.02, 0.03]:
                # Put side (bull put spread = sell higher put, buy lower put)
                put_sell_strike = round(spot * (1 - wing_width), 0)
                put_buy_strike = round(put_sell_strike - spot * spread_width_pct, 0)

                # Call side (bear call spread = sell lower call, buy higher call)
                call_sell_strike = round(spot * (1 + wing_width), 0)
                call_buy_strike = round(call_sell_strike + spot * spread_width_pct, 0)

                # Price all legs
                ps_greeks = self.bs.price_put(spot, put_sell_strike, dte, sigma)
                pb_greeks = self.bs.price_put(spot, put_buy_strike, dte, sigma)
                # Approximate call pricing
                cs_price = max(0.01, ps_greeks.price * 0.85)
                cb_price = max(0.005, pb_greeks.price * 0.85)

                put_credit = ps_greeks.price - pb_greeks.price
                call_credit = cs_price - cb_price
                total_credit = put_credit + call_credit

                if total_credit <= 0:
                    continue

                spread_width = spot * spread_width_pct
                max_loss = (spread_width - total_credit) * 100
                max_profit = total_credit * 100

                if max_loss <= 0:
                    continue

                strategies.append(MultiLegStrategy(
                    name=f"{ticker} IC {put_buy_strike}/{put_sell_strike}P - {call_sell_strike}/{call_buy_strike}C",
                    strategy_type="iron_condor",
                    ticker=ticker,
                    legs=[
                        StrategyLeg(ticker=ticker, strike=put_buy_strike, expiry="",
                                    option_type="put", side="buy", contracts=1,
                                    premium=pb_greeks.price, delta=pb_greeks.delta),
                        StrategyLeg(ticker=ticker, strike=put_sell_strike, expiry="",
                                    option_type="put", side="sell", contracts=1,
                                    premium=ps_greeks.price, delta=ps_greeks.delta),
                        StrategyLeg(ticker=ticker, strike=call_sell_strike, expiry="",
                                    option_type="call", side="sell", contracts=1,
                                    premium=cs_price, delta=-0.30),
                        StrategyLeg(ticker=ticker, strike=call_buy_strike, expiry="",
                                    option_type="call", side="buy", contracts=1,
                                    premium=cb_price, delta=0.20),
                    ],
                    net_debit=-total_credit,  # Negative = credit received
                    max_profit=max_profit,
                    max_loss=max_loss,
                    breakeven=put_sell_strike - total_credit,
                    reward_risk_ratio=max_profit / max_loss if max_loss > 0 else 0,
                    thesis=f"Range-bound: profit if {ticker} stays ${put_sell_strike}-${call_sell_strike}",
                    score=max_profit / max_loss * 50 if max_loss > 0 else 0,
                ))

        strategies.sort(key=lambda s: s.score, reverse=True)
        return strategies[:6]

    def butterflies(self, ticker: str, spot: float, sigma: float,
                    dte: int) -> List[MultiLegStrategy]:
        """Generate put butterfly spreads for directional bearish bets.

        Put butterfly = buy 1 ITM put + sell 2 ATM puts + buy 1 OTM put.
        """
        strategies: List[MultiLegStrategy] = []

        for target_pct in [0.03, 0.05, 0.07]:
            center = round(spot * (1 - target_pct), 0)
            wing = round(spot * 0.03, 0)  # 3% wing width

            upper = center + wing
            lower = center - wing

            upper_greeks = self.bs.price_put(spot, upper, dte, sigma)
            center_greeks = self.bs.price_put(spot, center, dte, sigma)
            lower_greeks = self.bs.price_put(spot, lower, dte, sigma)

            net_debit = upper_greeks.price - 2 * center_greeks.price + lower_greeks.price
            if net_debit <= 0:
                continue

            max_profit = (wing - net_debit) * 100
            max_loss = net_debit * 100

            if max_loss <= 0:
                continue

            strategies.append(MultiLegStrategy(
                name=f"{ticker} {upper}/{center}/{lower} put butterfly",
                strategy_type="butterfly",
                ticker=ticker,
                legs=[
                    StrategyLeg(ticker=ticker, strike=upper, expiry="",
                                option_type="put", side="buy", contracts=1,
                                premium=upper_greeks.price, delta=upper_greeks.delta),
                    StrategyLeg(ticker=ticker, strike=center, expiry="",
                                option_type="put", side="sell", contracts=2,
                                premium=center_greeks.price, delta=center_greeks.delta),
                    StrategyLeg(ticker=ticker, strike=lower, expiry="",
                                option_type="put", side="buy", contracts=1,
                                premium=lower_greeks.price, delta=lower_greeks.delta),
                ],
                net_debit=net_debit,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven=upper - net_debit,
                reward_risk_ratio=max_profit / max_loss if max_loss > 0 else 0,
                thesis=f"Max profit if {ticker} at ${center} at expiry",
                score=max_profit / max_loss * 40 if max_loss > 0 else 0,
            ))

        return strategies

    # ═══════════════════════════════════════════════════════════════════════
    # ROLLING CALENDAR
    # ═══════════════════════════════════════════════════════════════════════

    def weekly_roll_plan(self, ticker: str, spot: float, sigma: float,
                         weeks: int = 4, target_delta: float = -0.35) -> List[Dict[str, Any]]:
        """Generate a weekly put rotation plan.

        Buy weekly puts, rolling each Friday to the next week.
        """
        plan: List[Dict[str, Any]] = []
        today = datetime.utcnow().date()

        for w in range(weeks):
            dte = 7 * (w + 1)
            strike = self.bs.find_strike_for_delta(spot, target_delta, dte, sigma)
            greeks = self.bs.price_put(spot, strike, dte, sigma)

            entry_date = today + timedelta(days=7 * w)
            expiry_date = entry_date + timedelta(days=7)

            plan.append({
                "week": w + 1,
                "entry": entry_date.isoformat(),
                "expiry": expiry_date.isoformat(),
                "ticker": ticker,
                "strike": strike,
                "premium": greeks.price,
                "delta": greeks.delta,
                "theta": greeks.theta,
                "cost_1x": greeks.price * 100,
                "action": "BUY" if w == 0 else "ROLL",
            })

        return plan

    # ═══════════════════════════════════════════════════════════════════════
    # SECTOR ROTATION
    # ═══════════════════════════════════════════════════════════════════════

    def rotate_scan_tickers(self, regime: str, oil_price: float,
                            vix: float) -> List[str]:
        """Dynamically select scan tickers based on market regime.

        Different regimes favor different sectors for put buying.
        """
        # Base tickers always included
        base = ["SPY", "QQQ"]

        if regime in ("vol_shock_active", "vol_shock_armed", "credit_stress"):
            # Crisis: hit financials, consumer, high-beta
            return base + ["KRE", "HYG", "XLY", "JETS", "BITO"]

        if regime == "stagflation":
            # Stagflation: consumer discretionary hurts most, energy benefits
            return base + ["XLY", "KRE", "HYG", "JETS", "ZIM"]

        if oil_price > 100:
            # Oil shock: airlines, transport, consumer hit hardest
            return base + ["JETS", "XLY", "KRE", "ZIM", "HYG"]

        if vix > 30:
            # High vol: everything under pressure, focus on largest impact
            return base + ["KRE", "HYG", "JETS", "BITO", "XLY"]

        if regime == "risk_on":
            # Risk-on: fewer bearish targets, maybe hedges only
            return base + ["BITO", "ZIM"]

        # Default — full scan
        return base + ["JETS", "KRE", "HYG", "XLY", "ZIM", "BITO"]

    # ═══════════════════════════════════════════════════════════════════════
    # STRATEGY SELECTOR
    # ═══════════════════════════════════════════════════════════════════════

    def recommend_strategies(self, ticker: str, spot: float, sigma: float,
                              dte: int, mandate=None,
                              vix: float = 22.0) -> List[MultiLegStrategy]:
        """Auto-select best strategy type based on market conditions.

        High VIX (>30): straddles/strangles (sell vol)
        Low VIX (<18): butterflies (cheap directional bets)
        Crisis regime: bear put spreads (defined risk)
        Range-bound: iron condors
        """
        all_strategies: List[MultiLegStrategy] = []

        if vix > 30:
            # Volatility regime — straddles/strangles and spreads
            all_strategies.extend(self.straddles(ticker, spot, sigma, dte))
            all_strategies.extend(self.bear_put_spreads(ticker, spot, sigma, dte, mandate))

        elif vix < 18:
            # Low vol — cheap butterflies and spreads
            all_strategies.extend(self.butterflies(ticker, spot, sigma, dte))
            all_strategies.extend(self.bear_put_spreads(ticker, spot, sigma, dte, mandate))

        else:
            # Normal — spreads and condors
            all_strategies.extend(self.bear_put_spreads(ticker, spot, sigma, dte, mandate))
            all_strategies.extend(self.iron_condors(ticker, spot, sigma, dte))

        all_strategies.sort(key=lambda s: s.reward_risk_ratio, reverse=True)
        return all_strategies[:10]
