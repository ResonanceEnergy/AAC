"""
MATRIX MAXIMIZER — Options Scanner & Auto-Roll Engine
=======================================================
Scrapes options chains, filters by delta/volume/OI, prices with BS,
detects delta decay, and generates roll/pyramid signals.

Data sources (existing AAC infrastructure):
    - Polygon REST API (POLYGON_API_KEY in .env)
    - Unusual Whales (UNUSUAL_WHALES_API_KEY in .env)
    - IBKR connector for live fills
    - Fallback: synthetic chain from BS model when API unavailable

Auto-roll pattern (from the original playbook):
    1. Entry: Buy liquid 5-15% OTM puts (delta -0.25 to -0.45)
    2. Hold 3-7 days
    3. If profitable: Close 50%, roll remainder + capital into next weekly
    4. Pyramid: each successful roll increases notional 1.5x
    5. Cut losers at -40% premium
    6. Repeat weekly
"""

from __future__ import annotations

import json
import logging
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from strategies.matrix_maximizer.core import (
    Asset,
    MatrixConfig,
    MandateLevel,
    RollAction,
    SystemMandate,
)
from strategies.matrix_maximizer.greeks import BlackScholesEngine, GreeksResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OptionContract:
    """Raw option contract from chain data."""
    ticker: str
    strike: float
    expiry: str                  # YYYY-MM-DD
    dte: int                     # Days to expiry
    bid: float
    ask: float
    mid: float                   # (bid + ask) / 2
    volume: int
    open_interest: int
    iv: float                    # Implied volatility
    source: str = "polygon"      # polygon | unusual_whales | synthetic


@dataclass
class PutRecommendation:
    """Scored put recommendation with full Greeks and sizing."""
    rank: int
    ticker: str
    contract: OptionContract
    greeks: GreeksResult

    # Sizing
    contracts: int               # Number of contracts to buy
    total_cost: float            # contracts × mid × 100
    risk_pct: float              # % of account at risk

    # Scoring
    delta_score: float           # How close to target delta (0-100)
    liquidity_score: float       # Volume + OI score (0-100)
    edge_score: float            # BS theo vs market premium (0-100)
    composite_score: float       # Weighted composite

    # Context
    mandate: MandateLevel
    thesis: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def print_card(self) -> str:
        """Compact recommendation card."""
        return (
            f"#{self.rank} {self.ticker} PUT ${self.contract.strike:.0f} "
            f"exp {self.contract.expiry} ({self.contract.dte}d)\n"
            f"  Premium: ${self.contract.mid:.2f} | IV: {self.contract.iv:.0%} | "
            f"Delta: {self.greeks.delta:.3f}\n"
            f"  Contracts: {self.contracts} | Cost: ${self.total_cost:.0f} | "
            f"Risk: {self.risk_pct:.1%}\n"
            f"  Score: {self.composite_score:.0f}/100 | {self.thesis}"
        )


@dataclass
class Position:
    """Tracked open position for auto-roll monitoring."""
    ticker: str
    strike: float
    expiry: str
    entry_date: str
    entry_premium: float
    entry_delta: float
    contracts: int
    cost_basis: float            # Total cost when opened
    current_premium: float = 0.0
    current_delta: float = 0.0
    days_held: int = 0
    pnl_pct: float = 0.0


@dataclass
class RollSignal:
    """Auto-roll recommendation for an existing position."""
    position: Position
    action: RollAction
    reason: str

    # New position details (if rolling)
    new_strike: Optional[float] = None
    new_expiry: Optional[str] = None
    new_contracts: Optional[int] = None
    new_premium: Optional[float] = None
    new_delta: Optional[float] = None

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def print_signal(self) -> str:
        """Human-readable roll signal."""
        lines = [
            f"ROLL SIGNAL: {self.action.value.upper()} — {self.position.ticker} "
            f"${self.position.strike:.0f}P",
            f"  Reason: {self.reason}",
        ]
        if self.new_strike is not None:
            lines.append(
                f"  NEW: ${self.new_strike:.0f}P exp {self.new_expiry} "
                f"x{self.new_contracts} @ ${self.new_premium:.2f}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# OPTIONS SCANNER
# ═══════════════════════════════════════════════════════════════════════════

class OptionsScanner:
    """Options chain scanner with BS pricing and auto-roll logic.

    Usage:
        scanner = OptionsScanner(config, bs_engine)
        recs = scanner.scan_all(mandate, prices)
        rolls = scanner.check_rolls(positions, prices)
    """

    def __init__(
        self,
        config: MatrixConfig,
        bs_engine: BlackScholesEngine,
    ) -> None:
        self.config = config
        self.bs = bs_engine
        self._polygon_key = ""
        self._uw_key = ""
        self._load_keys()

    def _load_keys(self) -> None:
        """Load API keys from environment (via AAC config_loader)."""
        try:
            from shared.config_loader import get_env
            self._polygon_key = get_env("POLYGON_API_KEY", "")
            self._uw_key = get_env("UNUSUAL_WHALES_API_KEY", "")
        except ImportError:
            import os
            self._polygon_key = os.environ.get("POLYGON_API_KEY", "")
            self._uw_key = os.environ.get("UNUSUAL_WHALES_API_KEY", "")

    def _http_get(self, url: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """Safe HTTP GET returning JSON or None."""
        try:
            ctx = ssl.create_default_context()
            req = urllib.request.Request(url, headers={"User-Agent": "AAC-MatrixMaximizer/1.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            logger.debug("HTTP GET failed for %s: %s", url[:80], exc)
            return None

    def _fetch_polygon_chain(
        self,
        ticker: str,
        max_dte: int = 45,
    ) -> List[OptionContract]:
        """Fetch put options chain from Polygon.io REST API."""
        if not self._polygon_key:
            logger.debug("No Polygon key, using synthetic chain for %s", ticker)
            return []

        today = datetime.utcnow().date()
        exp_max = today + timedelta(days=max_dte)

        url = (
            f"https://api.polygon.io/v3/reference/options/contracts?"
            f"underlying_ticker={ticker}&contract_type=put"
            f"&expiration_date.gte={today.isoformat()}"
            f"&expiration_date.lte={exp_max.isoformat()}"
            f"&limit=100&order=asc&sort=strike_price"
            f"&apiKey={self._polygon_key}"
        )
        data = self._http_get(url, timeout=15)
        if not data or "results" not in data:
            return []

        contracts = []
        for item in data["results"]:
            exp_str = item.get("expiration_date", "")
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
            except (ValueError, TypeError):
                continue

            # Fetch snapshot for bid/ask/volume
            snap_url = (
                f"https://api.polygon.io/v3/snapshot/options/{ticker}/"
                f"{item.get('ticker', '')}?apiKey={self._polygon_key}"
            )
            snap = self._http_get(snap_url, timeout=10)
            snap_results = snap.get("results", {}) if snap else {}
            day_data = snap_results.get("day", {})
            greeks_data = snap_results.get("greeks", {})

            bid = snap_results.get("details", {}).get("last_quote", {}).get("bid", 0.0)
            ask = snap_results.get("details", {}).get("last_quote", {}).get("ask", 0.0)
            if not bid and not ask:
                # Fallback to day close
                bid = day_data.get("close", 0.0) * 0.95
                ask = day_data.get("close", 0.0) * 1.05

            contracts.append(OptionContract(
                ticker=ticker,
                strike=item.get("strike_price", 0.0),
                expiry=exp_str,
                dte=dte,
                bid=bid,
                ask=ask,
                mid=(bid + ask) / 2 if bid and ask else day_data.get("close", 0.0),
                volume=day_data.get("volume", 0),
                open_interest=day_data.get("open_interest", 0),
                iv=greeks_data.get("implied_volatility", 0.25),
                source="polygon",
            ))

        return contracts

    def _generate_synthetic_chain(
        self,
        ticker: str,
        spot: float,
        sigma: float = 0.25,
    ) -> List[OptionContract]:
        """Generate synthetic options chain using BS model when API unavailable."""
        today = datetime.utcnow().date()
        contracts = []

        seen = set()  # Deduplicate (strike, expiry) for low-priced tickers
        for weeks_out in [1, 2, 3, 4, 5, 6]:
            dte = weeks_out * 7
            expiry = (today + timedelta(days=dte)).isoformat()

            # Generate strikes from 20% OTM to ATM in 2.5% steps
            for otm_pct in [0.20, 0.175, 0.15, 0.125, 0.10, 0.075, 0.05, 0.025, 0.0]:
                K = round(spot * (1.0 - otm_pct), 0)
                key = (K, expiry)
                if key in seen:
                    continue
                seen.add(key)
                greeks = self.bs.price_put(spot, K, dte, sigma)

                contracts.append(OptionContract(
                    ticker=ticker,
                    strike=K,
                    expiry=expiry,
                    dte=dte,
                    bid=max(0.01, greeks.price * 0.95),
                    ask=greeks.price * 1.05,
                    mid=greeks.price,
                    volume=max(100, int(5000 * (1.0 - otm_pct))),
                    open_interest=max(500, int(20000 * (1.0 - otm_pct))),
                    iv=sigma,
                    source="synthetic",
                ))

        return contracts

    def scan_ticker(
        self,
        ticker: str,
        spot: float,
        mandate: SystemMandate,
        sigma: float = 0.25,
    ) -> List[PutRecommendation]:
        """Scan a single ticker for put recommendations.

        Fetches chain (Polygon or synthetic), filters by delta/volume/OI,
        prices with BS, scores, and returns ranked recommendations.
        """
        # Fetch chain
        contracts = self._fetch_polygon_chain(ticker, self.config.max_expiry_days)
        if not contracts:
            contracts = self._generate_synthetic_chain(ticker, spot, sigma)

        if not contracts:
            return []

        recs = []
        for contract in contracts:
            # Price with BS
            greeks = self.bs.price_put(
                S=spot,
                K=contract.strike,
                T_days=contract.dte,
                sigma=contract.iv or sigma,
            )

            # Filter by delta range
            if not (self.config.target_delta_min <= greeks.delta <= self.config.target_delta_max):
                continue

            # Filter by liquidity
            if contract.volume < self.config.min_volume and contract.source != "synthetic":
                continue
            if contract.open_interest < self.config.min_open_interest and contract.source != "synthetic":
                continue

            # Scoring
            target_delta = (self.config.target_delta_min + self.config.target_delta_max) / 2
            delta_score = max(0, 100 - abs(greeks.delta - target_delta) * 500)

            vol_oi = contract.volume + contract.open_interest
            liquidity_score = min(100, vol_oi / 100)

            # Edge: BS theo vs market mid
            if contract.mid > 0:
                edge = (greeks.price - contract.mid) / contract.mid * 100
                edge_score = min(100, max(0, 50 + edge * 10))
            else:
                edge_score = 50

            composite = delta_score * 0.40 + liquidity_score * 0.30 + edge_score * 0.30

            # Position sizing
            risk_budget = self.config.account_size * mandate.risk_per_trade_pct
            cost_per_contract = contract.mid * 100
            if cost_per_contract <= 0:
                continue
            contracts_count = max(1, min(
                mandate.max_contracts_per_name,
                int(risk_budget / cost_per_contract),
            ))

            recs.append(PutRecommendation(
                rank=0,
                ticker=ticker,
                contract=contract,
                greeks=greeks,
                contracts=contracts_count,
                total_cost=contracts_count * cost_per_contract,
                risk_pct=contracts_count * cost_per_contract / self.config.account_size,
                delta_score=delta_score,
                liquidity_score=liquidity_score,
                edge_score=edge_score,
                composite_score=composite,
                mandate=mandate.level,
                thesis=f"Bear put on {ticker} — oil-shock regime",
            ))

        # Sort by composite score descending
        recs.sort(key=lambda r: r.composite_score, reverse=True)
        for i, rec in enumerate(recs):
            rec.rank = i + 1

        return recs[:10]  # Top 10 per ticker

    def scan_all(
        self,
        mandate: SystemMandate,
        prices: Optional[Dict[str, float]] = None,
        sigmas: Optional[Dict[str, float]] = None,
    ) -> List[PutRecommendation]:
        """Scan all configured tickers and return unified ranked recommendations."""
        from strategies.matrix_maximizer.core import ASSET_VOLATILITIES, DEFAULT_PRICES, Asset

        all_recs: List[PutRecommendation] = []

        for ticker in self.config.scan_tickers:
            spot = (prices or {}).get(ticker) or DEFAULT_PRICES.get(Asset(ticker), 100.0)
            sigma = (sigmas or {}).get(ticker) or ASSET_VOLATILITIES.get(Asset(ticker), 0.25)

            logger.info("Scanning %s @ $%.2f (IV=%.0f%%)", ticker, spot, sigma * 100)
            recs = self.scan_ticker(ticker, spot, mandate, sigma)
            all_recs.extend(recs)

        # Re-rank globally
        all_recs.sort(key=lambda r: r.composite_score, reverse=True)
        for i, rec in enumerate(all_recs):
            rec.rank = i + 1

        logger.info("Scanner found %d put recommendations across %d tickers",
                     len(all_recs), len(self.config.scan_tickers))
        return all_recs

    # ═══════════════════════════════════════════════════════════════════
    # AUTO-ROLL ENGINE
    # ═══════════════════════════════════════════════════════════════════

    def check_rolls(
        self,
        positions: List[Position],
        prices: Dict[str, float],
        mandate: SystemMandate,
    ) -> List[RollSignal]:
        """Check all open positions for auto-roll signals.

        Roll triggers:
            1. Delta decay ≥ 20% from entry → roll to fresh put
            2. Premium down ≥ 40% → stop loss
            3. Profitable + 3-7 days held → close 50%, pyramid rest
            4. DTE ≤ 3 → close or roll to avoid pin risk
        """
        from strategies.matrix_maximizer.core import ASSET_VOLATILITIES, Asset

        signals: List[RollSignal] = []

        for pos in positions:
            spot = prices.get(pos.ticker, 0.0)
            if spot <= 0:
                continue

            sigma = ASSET_VOLATILITIES.get(Asset(pos.ticker), 0.25)
            try:
                exp_date = datetime.strptime(pos.expiry, "%Y-%m-%d").date()
                dte = (exp_date - datetime.utcnow().date()).days
            except (ValueError, TypeError):
                dte = 30

            if dte <= 0:
                signals.append(RollSignal(
                    position=pos,
                    action=RollAction.CLOSE,
                    reason="Expired — close position",
                ))
                continue

            # Re-price current position
            current_greeks = self.bs.price_put(spot, pos.strike, dte, sigma)
            pos.current_premium = current_greeks.price
            pos.current_delta = current_greeks.delta
            pos.pnl_pct = (current_greeks.price - pos.entry_premium) / pos.entry_premium if pos.entry_premium > 0 else 0.0

            # Check 1: DTE ≤ 3 → close to avoid pin risk
            if dte <= 3:
                signals.append(RollSignal(
                    position=pos,
                    action=RollAction.CLOSE,
                    reason=f"DTE={dte}, avoiding pin risk",
                ))
                continue

            # Check 2: Premium stop loss
            if pos.pnl_pct <= -self.config.premium_stop_loss_pct:
                signals.append(RollSignal(
                    position=pos,
                    action=RollAction.CLOSE,
                    reason=f"Stop loss: premium down {pos.pnl_pct:.0%} (limit {-self.config.premium_stop_loss_pct:.0%})",
                ))
                continue

            # Check 3: Delta decay
            if self.bs.delta_decay_check(pos.entry_delta, current_greeks, self.config.delta_decay_threshold):
                # Find new strike at target delta
                target_delta = (self.config.target_delta_min + self.config.target_delta_max) / 2
                new_strike = self.bs.find_strike_for_delta(spot, target_delta, 14, sigma)
                new_greeks = self.bs.price_put(spot, new_strike, 14, sigma)
                new_expiry = (datetime.utcnow().date() + timedelta(days=14)).isoformat()

                signals.append(RollSignal(
                    position=pos,
                    action=RollAction.ROLL_DEEPER,
                    reason=(
                        f"Delta decay: entry {pos.entry_delta:.3f} -> current {current_greeks.delta:.3f} "
                        f"({abs(1 - abs(current_greeks.delta)/abs(pos.entry_delta)):.0%} decay)"
                    ),
                    new_strike=new_strike,
                    new_expiry=new_expiry,
                    new_contracts=pos.contracts,
                    new_premium=new_greeks.price,
                    new_delta=new_greeks.delta,
                ))
                continue

            # Check 4: Profitable + held 3-7 days → pyramid
            if pos.pnl_pct > 0.10 and pos.days_held >= 3 and mandate.pyramid_allowed:
                new_contracts = int(pos.contracts * self.config.pyramid_multiplier)
                target_delta = (self.config.target_delta_min + self.config.target_delta_max) / 2
                new_strike = self.bs.find_strike_for_delta(spot, target_delta, 14, sigma)
                new_greeks = self.bs.price_put(spot, new_strike, 14, sigma)
                new_expiry = (datetime.utcnow().date() + timedelta(days=14)).isoformat()

                signals.append(RollSignal(
                    position=pos,
                    action=RollAction.PYRAMID,
                    reason=(
                        f"Profitable ({pos.pnl_pct:+.0%}), {pos.days_held}d held — "
                        f"closing 50%, pyramiding {new_contracts}x into next weekly"
                    ),
                    new_strike=new_strike,
                    new_expiry=new_expiry,
                    new_contracts=new_contracts,
                    new_premium=new_greeks.price,
                    new_delta=new_greeks.delta,
                ))
                continue

            # No action needed
            signals.append(RollSignal(
                position=pos,
                action=RollAction.HOLD,
                reason=(
                    f"Holding: PnL {pos.pnl_pct:+.0%}, delta {current_greeks.delta:.3f}, "
                    f"{dte}d remaining"
                ),
            ))

        return signals
