"""
PlanktonXD Prediction Market Harvester — BARREN WUFFET v2.7.0
==============================================================
Emulates the planktonXD strategy: high-frequency prediction market
micro-arbitrage that turned ~$1,000 → $106,000 in one year on Polymarket.

planktonXD (0x4ffe49ba2a4cae123536a8af4fda48faeb609f71):
  - 61,000+ predictions, Feb 2025–Feb 2026
  - ~170 trades/day via automated bot
  - Biggest single win: $2,527.40 (only 2% of total profit)
  - Perfect 45° profit curve, near-zero drawdowns

Strategy Pillars:
  1. DEEP OTM HARVESTING — Buy "impossible" outcomes priced 0.1¢–3¢ that the
     crowd systematically underprices. When one hits, 500x–23,750x payoff.
  2. SPREAD MARKET-MAKING — Place orders on both sides of thin order books
     in niche/long-tail markets to earn bid-ask spread.
  3. MULTI-MARKET DIVERSIFICATION — Sports, weather, crypto prices, politics,
     esports — scan thousands of markets 24/7 for pricing inefficiencies.
  4. ANTIFRAGILE POSITION SIZING — $5–$25 per bet, never all-in.
     Compounding > moonshots. 0.5% daily via volume.
  5. LIQUIDITY DESERT SNIPING — In thin order books (esports sub-leagues,
     extreme price ranges), pick up cheap shares from panic/misoperation.

Core Insight: "Buying lottery tickets where the math is on your side."
  — The probability of an event is underestimated by the market.
  — Cost: a few dollars. Payoff: thousands on rare tail events.
  — Certainty > Odds: high-probability small profits PLUS cheap tail hedges.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from shared.strategy_framework import (
    BaseArbitrageStrategy,
    StrategyConfig,
    TradingSignal,
    SignalType,
)
from shared.communication import CommunicationFramework
from shared.audit_logger import AuditLogger
from agents.polymarket_agent import PolymarketAgent

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS & TYPES
# ═══════════════════════════════════════════════════════════════════════════

class MarketCategory(Enum):
    """Prediction-market verticals to scan (planktonXD bets on ALL of these)."""
    CRYPTO_PRICE = "crypto_price"
    POLITICS = "politics"
    SPORTS = "sports"
    ESPORTS = "esports"
    WEATHER = "weather"
    ECONOMICS = "economics"
    ENTERTAINMENT = "entertainment"
    SCIENCE = "science"


class BetType(Enum):
    """Types of prediction-market bets."""
    DEEP_OTM_TAIL = "deep_otm_tail"         # 0.1¢–1¢ — pure tail-risk lottery
    CHEAP_CONTRARIAN = "cheap_contrarian"    # 1¢–3¢ — crowd underprices
    SPREAD_HARVEST = "spread_harvest"        # Market-making bid/ask
    LIQUIDITY_SNIPE = "liquidity_snipe"      # Thin book panic pickups


class PositionState(Enum):
    OPEN = "open"
    CLOSED_WIN = "closed_win"
    CLOSED_LOSS = "closed_loss"
    EXPIRED_WORTHLESS = "expired_worthless"


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PredictionMarket:
    """A single prediction-market event slot."""
    market_id: str
    category: MarketCategory
    question: str
    outcomes: List[str]                     # e.g. ["Yes", "No"]
    prices: Dict[str, float]               # outcome → price (0.0–1.0)
    volume_24h: float = 0.0
    liquidity: float = 0.0
    resolution_time: Optional[datetime] = None
    source: str = "polymarket"

    @property
    def cheapest_outcome_price(self) -> float:
        return min(self.prices.values()) if self.prices else 1.0

    @property
    def cheapest_outcome(self) -> str:
        if not self.prices:
            return ""
        return min(self.prices, key=lambda k: self.prices[k])

    @property
    def spread(self) -> float:
        """Bid-ask implied spread (sum of all outcome prices minus 1.0)."""
        return sum(self.prices.values()) - 1.0 if self.prices else 0.0

    @property
    def is_thin_book(self) -> bool:
        return self.liquidity < 500.0

    @property
    def hours_to_resolution(self) -> Optional[float]:
        if self.resolution_time is None:
            return None
        delta = self.resolution_time - datetime.now()
        return max(delta.total_seconds() / 3600.0, 0.0)


@dataclass
class PlanktonBet:
    """A single bet in the planktonXD style."""
    bet_id: str
    market: PredictionMarket
    bet_type: BetType
    outcome: str
    entry_price: float                      # Cost per share (0.0–1.0)
    shares: float                           # Number of shares
    cost: float                             # Total dollars spent
    potential_payout: float                  # If outcome resolves YES
    implied_probability: float              # = entry_price
    estimated_true_probability: float       # Our model's estimate
    edge: float                             # true_prob - implied_prob
    state: PositionState = PositionState.OPEN
    pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None

    @property
    def potential_roi(self) -> float:
        if self.cost <= 0:
            return 0.0
        return (self.potential_payout - self.cost) / self.cost

    @property
    def risk_reward_ratio(self) -> float:
        if self.cost <= 0:
            return 0.0
        return self.potential_payout / self.cost


@dataclass
class HarvesterStats:
    """Running stats emulating planktonXD's public profile."""
    total_bets: int = 0
    winning_bets: int = 0
    losing_bets: int = 0
    total_invested: float = 0.0
    total_returned: float = 0.0
    biggest_win: float = 0.0
    biggest_loss: float = 0.0
    current_streak: int = 0                 # Positive = wins, negative = losses
    daily_bet_count: int = 0
    categories_traded: Dict[str, int] = field(default_factory=dict)

    @property
    def net_profit(self) -> float:
        return self.total_returned - self.total_invested

    @property
    def win_rate(self) -> float:
        if self.total_bets == 0:
            return 0.0
        return self.winning_bets / self.total_bets

    @property
    def roi(self) -> float:
        if self.total_invested == 0:
            return 0.0
        return self.net_profit / self.total_invested

    @property
    def avg_bet_size(self) -> float:
        if self.total_bets == 0:
            return 0.0
        return self.total_invested / self.total_bets

    @property
    def profit_factor(self) -> float:
        """Gross wins / gross losses."""
        gross_wins = self.total_returned - (self.losing_bets * self.avg_bet_size)
        gross_losses = max(self.total_invested - gross_wins, 0.001)
        return gross_wins / gross_losses


# ═══════════════════════════════════════════════════════════════════════════
# CORE STRATEGY
# ═══════════════════════════════════════════════════════════════════════════

class PlanktonXDPredictionHarvester(BaseArbitrageStrategy):
    """
    Emulates planktonXD's high-frequency prediction-market harvesting strategy.

    Key parameters calibrated from planktonXD's on-chain data:
      - ~170 bets/day across ALL market categories
      - $5–$25 per bet (never exceeds $50)
      - Targets outcomes priced at 0.1¢–3¢ (deep OTM tail bets)
      - Also market-makes on spread when bid-ask > 2%
      - Win rate is LOW (~5-15%) but winners pay 100x–23,750x
      - Net result: smooth 45° profit curve via massive diversification
    """

    # ─── planktonXD-calibrated constants ────────────────────────────────
    MIN_BET_USD = 5.0
    MAX_BET_USD = 25.0
    ABSOLUTE_MAX_BET_USD = 50.0             # Hard ceiling
    MAX_DAILY_BETS = 200                    # planktonXD averages ~170
    MAX_OPEN_POSITIONS = 500                # Diversification is the edge

    # Deep OTM thresholds (the core of the strategy)
    DEEP_OTM_MAX_PRICE = 0.03              # Buy outcomes priced ≤ 3¢
    DEEP_OTM_MIN_PRICE = 0.001             # Floor price (0.1¢)
    TAIL_BET_MAX_PRICE = 0.01              # Pure tail: ≤ 1¢

    # Edge thresholds
    MIN_EDGE_DEEP_OTM = 0.005              # 0.5% edge on deep OTM (tiny edge, huge payoff)
    MIN_EDGE_SPREAD = 0.015                # 1.5% edge on spread harvesting
    MIN_EDGE_CONTRARIAN = 0.02             # 2% edge on contrarian plays

    # Spread / market-making
    MIN_SPREAD_FOR_MM = 0.02               # Market-make when spread > 2%
    MM_POSITION_SIZE_USD = 10.0            # Market-making size

    # Risk controls (planktonXD's biggest win was only 2% of total profit)
    MAX_SINGLE_MARKET_EXPOSURE_PCT = 2.0   # Never >2% of bankroll in one market
    MAX_CATEGORY_EXPOSURE_PCT = 25.0       # Cap per category
    BANKROLL_PROTECTION_FLOOR = 0.10       # Stop if bankroll drops to 10% of peak

    def __init__(self, config: StrategyConfig, communication: CommunicationFramework,
                 audit_logger: AuditLogger):
        super().__init__(config, communication, audit_logger)

        # Portfolio state
        self.bankroll = 1000.0              # planktonXD started with ~$1,000
        self.peak_bankroll = 1000.0
        self.open_bets: Dict[str, PlanktonBet] = {}
        self.closed_bets: List[PlanktonBet] = []
        self.stats = HarvesterStats()

        # Market scanner state
        self.scanned_markets: Dict[str, PredictionMarket] = {}
        self.market_scan_interval = timedelta(minutes=5)
        self.last_scan_time: Optional[datetime] = None

        # Category exposure tracking
        self.category_exposure: Dict[MarketCategory, float] = {
            cat: 0.0 for cat in MarketCategory
        }

        # Daily counters (reset at midnight)
        self._daily_bet_count = 0
        self._daily_invested = 0.0
        self._last_reset_date: Optional[datetime] = None

    # ─── Initialization ────────────────────────────────────────────────

    async def _initialize_strategy(self):
        """Initialize the planktonXD harvester."""
        await self.communication.subscribe_to_messages(
            self.config.strategy_id,
            ['prediction_market_prices', 'prediction_market_resolution',
             'market_liquidity_update']
        )
        logger.info(
            f"PlanktonXD Harvester initialized — bankroll=${self.bankroll:.2f}, "
            f"targeting {self.MAX_DAILY_BETS} bets/day across "
            f"{len(MarketCategory)} categories"
        )

    def _should_generate_signal(self) -> bool:
        """Determine if conditions are right to generate signals."""
        if self._daily_bet_count >= self.MAX_DAILY_BETS:
            return False
        if self.bankroll <= self.peak_bankroll * self.BANKROLL_PROTECTION_FLOOR:
            return False
        return True

    async def _subscribe_market_data(self):
        """Subscribe to prediction market data feeds."""
        data_types = [
            'prediction_market_prices',
            'prediction_market_resolution',
            'prediction_market_orderbook',
            'crypto_spot_price',         # For crypto-price prediction markets
            'sports_odds',               # For sports-related markets
        ]
        await self.communication.subscribe_to_messages(
            self.config.strategy_id, data_types
        )

    # ─── Live Polymarket Feed ─────────────────────────────────────────

    async def fetch_polymarket_markets(self, limit: int = 200) -> List['PredictionMarket']:
        """
        Pull active markets from Polymarket via PolymarketAgent and
        convert them into PredictionMarket objects that the scanner
        understands.  Updates self.scanned_markets in-place.
        """
        agent = PolymarketAgent()
        try:
            raw_markets = await agent.get_active_markets(limit=limit)
        finally:
            await agent.close()

        TAG_CATEGORY_MAP = {
            "crypto": MarketCategory.CRYPTO_PRICE,
            "bitcoin": MarketCategory.CRYPTO_PRICE,
            "ethereum": MarketCategory.CRYPTO_PRICE,
            "politics": MarketCategory.POLITICS,
            "elections": MarketCategory.POLITICS,
            "sports": MarketCategory.SPORTS,
            "nba": MarketCategory.SPORTS,
            "nfl": MarketCategory.SPORTS,
            "mlb": MarketCategory.SPORTS,
            "esports": MarketCategory.ESPORTS,
            "weather": MarketCategory.WEATHER,
            "economics": MarketCategory.ECONOMICS,
            "fed": MarketCategory.ECONOMICS,
            "entertainment": MarketCategory.ENTERTAINMENT,
            "science": MarketCategory.SCIENCE,
        }

        converted: List[PredictionMarket] = []
        for pm in raw_markets:
            # Infer category from question keywords
            q_lower = pm.question.lower()
            category = MarketCategory.CRYPTO_PRICE  # default
            for keyword, cat in TAG_CATEGORY_MAP.items():
                if keyword in q_lower:
                    category = cat
                    break

            mkt = PredictionMarket(
                market_id=pm.condition_id or pm.slug,
                category=category,
                question=pm.question,
                outcomes=["Yes", "No"],
                prices={"Yes": pm.yes_price, "No": pm.no_price},
                volume_24h=pm.volume,
                liquidity=pm.liquidity,
                source="polymarket",
            )
            self.scanned_markets[mkt.market_id] = mkt
            converted.append(mkt)

        self.last_scan_time = datetime.now()
        logger.info(
            f"Polymarket feed: ingested {len(converted)} live markets "
            f"into PlanktonXD scanner"
        )
        return converted

    # ─── Market Scanning (planktonXD scans 24/7) ──────────────────────

    def scan_for_opportunities(
        self, markets: List[PredictionMarket]
    ) -> List[Tuple[PredictionMarket, BetType, str, float]]:
        """
        Scan a batch of prediction markets for planktonXD-style opportunities.

        Returns list of (market, bet_type, outcome, edge) tuples.
        """
        opportunities = []

        for market in markets:
            # Skip markets resolving too soon (need time for price to move)
            hours_left = market.hours_to_resolution
            if hours_left is not None and hours_left < 1.0:
                continue

            # ── 1. DEEP OTM TAIL BETS (the planktonXD signature move) ──
            for outcome, price in market.prices.items():
                if self.DEEP_OTM_MIN_PRICE <= price <= self.DEEP_OTM_MAX_PRICE:
                    true_prob = self._estimate_true_probability(market, outcome, price)
                    edge = true_prob - price

                    if price <= self.TAIL_BET_MAX_PRICE:
                        # Ultra-cheap tail: 0.1¢–1¢ range (planktonXD's bread & butter)
                        if edge > self.MIN_EDGE_DEEP_OTM:
                            opportunities.append(
                                (market, BetType.DEEP_OTM_TAIL, outcome, edge)
                            )
                    else:
                        # Cheap contrarian: 1¢–3¢ range
                        if edge > self.MIN_EDGE_CONTRARIAN:
                            opportunities.append(
                                (market, BetType.CHEAP_CONTRARIAN, outcome, edge)
                            )

            # ── 2. SPREAD HARVESTING (market-making on thin books) ──
            if market.spread > self.MIN_SPREAD_FOR_MM and market.is_thin_book:
                opportunities.append(
                    (market, BetType.SPREAD_HARVEST, "", market.spread)
                )

            # ── 3. LIQUIDITY DESERT SNIPING ──
            if market.is_thin_book and market.volume_24h < 100:
                for outcome, price in market.prices.items():
                    if price <= self.DEEP_OTM_MAX_PRICE:
                        true_prob = self._estimate_true_probability(
                            market, outcome, price
                        )
                        if true_prob > price * 2:  # Price is <50% of true prob
                            opportunities.append(
                                (market, BetType.LIQUIDITY_SNIPE, outcome,
                                 true_prob - price)
                            )

        # Sort by edge descending
        opportunities.sort(key=lambda x: x[3], reverse=True)
        return opportunities

    def _estimate_true_probability(
        self, market: PredictionMarket, outcome: str, market_price: float
    ) -> float:
        """
        Estimate true probability of an outcome.

        planktonXD's edge: the market systematically underprices tail events.
        A 0.1¢ price implies 0.1% probability, but many events have true
        probability closer to 1-5%.

        In practice this would use:
        - Historical base rates for similar events
        - Current market conditions (crypto volatility, etc.)
        - Information asymmetry signals
        - Liquidity-adjusted fair value models
        """
        # Base: market price is the implied probability
        implied_prob = market_price

        # Category-specific adjustments
        adjustments = {
            MarketCategory.CRYPTO_PRICE: self._crypto_tail_adjustment,
            MarketCategory.ESPORTS: self._esports_adjustment,
            MarketCategory.SPORTS: self._sports_adjustment,
            MarketCategory.WEATHER: self._weather_adjustment,
            MarketCategory.POLITICS: self._politics_adjustment,
        }

        adjust_fn = adjustments.get(market.category)
        if adjust_fn:
            return adjust_fn(market, outcome, implied_prob)

        # Default: assume market underprices tail by 2-5x for very cheap outcomes
        if implied_prob < 0.01:
            return implied_prob * 3.0  # Markets underestimate tails 3x
        elif implied_prob < 0.03:
            return implied_prob * 1.5
        return implied_prob

    def _crypto_tail_adjustment(
        self, market: PredictionMarket, outcome: str, implied_prob: float
    ) -> float:
        """
        Crypto price markets: extreme moves are underpriced during trending markets.
        planktonXD bought SOL<$130 at 0.7¢ → it hit, returning 9,285%.
        During "mainstream bullish" periods, bearish tails get crushed to near-zero.
        """
        # Crypto has fat tails — market underestimates crash probability
        if implied_prob < 0.01:
            return implied_prob * 5.0   # 5x underpricing in extreme scenarios
        elif implied_prob < 0.03:
            return implied_prob * 2.5
        return implied_prob * 1.3

    def _esports_adjustment(
        self, market: PredictionMarket, outcome: str, implied_prob: float
    ) -> float:
        """
        Esports sub-leagues: high information asymmetry, low liquidity.
        planktonXD bought Fuego (VALORANT) at 0.1¢ → $874.09 (23,750% ROI).
        These markets are "arbitrage paradises" due to tiny audience.
        """
        if implied_prob < 0.005:
            return implied_prob * 8.0   # Massive underpricing in niche esports
        elif implied_prob < 0.02:
            return implied_prob * 3.0
        return implied_prob * 1.5

    def _sports_adjustment(
        self, market: PredictionMarket, outcome: str, implied_prob: float
    ) -> float:
        """Sports: lower info asymmetry but still mispriced at extremes."""
        if implied_prob < 0.01:
            return implied_prob * 2.5
        return implied_prob * 1.2

    def _weather_adjustment(
        self, market: PredictionMarket, outcome: str, implied_prob: float
    ) -> float:
        """
        Weather: "Zero earthquakes worldwide" → $15.05 cost, $1,330 return.
        Highly predictable base rates that the crowd mis-estimates.
        """
        if implied_prob < 0.01:
            return implied_prob * 4.0
        return implied_prob * 1.5

    def _politics_adjustment(
        self, market: PredictionMarket, outcome: str, implied_prob: float
    ) -> float:
        """Politics: mainstream narrative bias creates tail mispricing."""
        if implied_prob < 0.01:
            return implied_prob * 3.0
        return implied_prob * 1.3

    # ─── Position Sizing (planktonXD's discipline) ────────────────────

    def calculate_bet_size(
        self, bet_type: BetType, edge: float, market: PredictionMarket
    ) -> float:
        """
        Calculate bet size using planktonXD's position-sizing rules:
        - $5–$25 per bet
        - Never exceeds 2% of bankroll on a single market
        - Scale with edge magnitude
        """
        # Hard floor / ceiling
        base_bet = self.MIN_BET_USD

        # Scale with edge (higher edge → larger bet, up to max)
        if bet_type == BetType.DEEP_OTM_TAIL:
            # Tail bets: small and diversified
            bet = base_bet + (edge * 200)  # Edge of 0.05 → $15
        elif bet_type == BetType.CHEAP_CONTRARIAN:
            bet = base_bet + (edge * 300)
        elif bet_type == BetType.SPREAD_HARVEST:
            bet = self.MM_POSITION_SIZE_USD
        elif bet_type == BetType.LIQUIDITY_SNIPE:
            bet = base_bet + (edge * 150)
        else:
            bet = base_bet

        # Cap at configured maximum
        bet = min(bet, self.MAX_BET_USD)
        bet = min(bet, self.ABSOLUTE_MAX_BET_USD)

        # Never exceed 2% of bankroll per market
        max_market_bet = self.bankroll * (self.MAX_SINGLE_MARKET_EXPOSURE_PCT / 100)
        existing_exposure = sum(
            b.cost for b in self.open_bets.values()
            if b.market.market_id == market.market_id
        )
        available = max_market_bet - existing_exposure
        bet = min(bet, max(available, 0))

        # Category cap: 25% of bankroll per category
        max_cat = self.bankroll * (self.MAX_CATEGORY_EXPOSURE_PCT / 100)
        cat_exposure = self.category_exposure.get(market.category, 0.0)
        bet = min(bet, max(max_cat - cat_exposure, 0))

        # Never bet more than we can afford
        bet = min(bet, self.bankroll * 0.05)  # Hard 5% single-bet cap

        return max(round(bet, 2), 0.0)

    # ─── Bet Execution ────────────────────────────────────────────────

    def place_bet(
        self,
        market: PredictionMarket,
        bet_type: BetType,
        outcome: str,
        edge: float,
    ) -> Optional[PlanktonBet]:
        """
        Place a planktonXD-style bet.

        Returns PlanktonBet if placed, None if rejected by risk checks.
        """
        # ── Pre-flight checks ──
        self._maybe_reset_daily_counters()

        if self._daily_bet_count >= self.MAX_DAILY_BETS:
            return None
        if len(self.open_bets) >= self.MAX_OPEN_POSITIONS:
            return None
        if self.bankroll <= self.peak_bankroll * self.BANKROLL_PROTECTION_FLOOR:
            logger.warning("Bankroll below protection floor — halting bets")
            return None

        cost = self.calculate_bet_size(bet_type, edge, market)
        if cost < self.MIN_BET_USD:
            return None

        entry_price = market.prices.get(outcome, 0.0)
        if entry_price <= 0:
            return None

        shares = cost / entry_price
        potential_payout = shares * 1.0  # Pays $1 per share if outcome = YES
        true_prob = self._estimate_true_probability(market, outcome, entry_price)

        bet = PlanktonBet(
            bet_id=f"pxd_{self.stats.total_bets + 1:06d}",
            market=market,
            bet_type=bet_type,
            outcome=outcome,
            entry_price=entry_price,
            shares=shares,
            cost=cost,
            potential_payout=potential_payout,
            implied_probability=entry_price,
            estimated_true_probability=true_prob,
            edge=edge,
        )

        # Deduct from bankroll
        self.bankroll -= cost
        self.open_bets[bet.bet_id] = bet

        # Update tracking
        self._daily_bet_count += 1
        self._daily_invested += cost
        self.stats.total_bets += 1
        self.stats.total_invested += cost
        cat_name = market.category.value
        self.stats.categories_traded[cat_name] = (
            self.stats.categories_traded.get(cat_name, 0) + 1
        )
        self.category_exposure[market.category] = (
            self.category_exposure.get(market.category, 0.0) + cost
        )

        logger.info(
            f"BET PLACED [{bet.bet_id}] {bet_type.value} | "
            f"{market.category.value}: '{market.question[:60]}' | "
            f"Outcome='{outcome}' @ {entry_price:.4f} | "
            f"Cost=${cost:.2f} → Potential ${potential_payout:.2f} "
            f"({bet.potential_roi:.0%} ROI) | Edge={edge:.4f}"
        )
        return bet

    # ─── Resolution Handling ────────────────────────────────────────────

    def resolve_bet(self, bet_id: str, winning_outcome: str) -> Optional[PlanktonBet]:
        """Resolve a bet when the prediction market settles."""
        bet = self.open_bets.pop(bet_id, None)
        if bet is None:
            return None

        bet.closed_at = datetime.now()

        if bet.outcome == winning_outcome:
            # WINNER — shares pay $1 each
            payout = bet.shares
            bet.pnl = payout - bet.cost
            bet.state = PositionState.CLOSED_WIN
            self.bankroll += payout
            self.stats.winning_bets += 1
            self.stats.total_returned += payout
            self.stats.biggest_win = max(self.stats.biggest_win, bet.pnl)
            self.stats.current_streak = max(self.stats.current_streak, 0) + 1

            logger.info(
                f"WIN [{bet.bet_id}] PnL=${bet.pnl:.2f} "
                f"({bet.potential_roi:.0%} ROI) | "
                f"Bankroll=${self.bankroll:.2f}"
            )
        else:
            # LOSS — shares expire worthless
            bet.pnl = -bet.cost
            bet.state = PositionState.EXPIRED_WORTHLESS
            self.stats.losing_bets += 1
            self.stats.biggest_loss = min(self.stats.biggest_loss, bet.pnl)
            self.stats.current_streak = min(self.stats.current_streak, 0) - 1

        # Update peak
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)

        # Release category exposure
        self.category_exposure[bet.market.category] = max(
            self.category_exposure.get(bet.market.category, 0.0) - bet.cost, 0.0
        )

        self.closed_bets.append(bet)
        return bet

    # ─── Signal Generation (AAC integration) ────────────────────────

    async def _generate_signals(self) -> List[TradingSignal]:
        """Generate AAC trading signals from prediction-market scan."""
        signals = []

        markets = list(self.scanned_markets.values())
        opportunities = self.scan_for_opportunities(markets)

        for market, bet_type, outcome, edge in opportunities:
            bet = self.place_bet(market, bet_type, outcome, edge)
            if bet is None:
                continue

            signal = TradingSignal(
                strategy_id=self.config.strategy_id,
                signal_type=SignalType.LONG,
                symbol=f"PRED:{market.market_id}:{outcome}",
                quantity=bet.shares,
                price=bet.entry_price,
                confidence=min(edge * 10, 1.0),
                metadata={
                    'bet_id': bet.bet_id,
                    'bet_type': bet_type.value,
                    'category': market.category.value,
                    'question': market.question,
                    'outcome': outcome,
                    'cost': bet.cost,
                    'potential_payout': bet.potential_payout,
                    'potential_roi': bet.potential_roi,
                    'edge': edge,
                    'implied_probability': bet.implied_probability,
                    'estimated_true_probability': bet.estimated_true_probability,
                },
            )
            signals.append(signal)

        return signals

    def _update_market_data(self, data: Dict[str, Any]):
        """Ingest prediction market data updates."""
        data_type = data.get('type')

        if data_type == 'prediction_market_prices':
            market_id = data.get('market_id', '')
            if market_id in self.scanned_markets:
                mkt = self.scanned_markets[market_id]
                mkt.prices = data.get('prices', mkt.prices)
                mkt.volume_24h = data.get('volume_24h', mkt.volume_24h)
                mkt.liquidity = data.get('liquidity', mkt.liquidity)
            else:
                self.scanned_markets[market_id] = PredictionMarket(
                    market_id=market_id,
                    category=MarketCategory(data.get('category', 'crypto_price')),
                    question=data.get('question', ''),
                    outcomes=data.get('outcomes', []),
                    prices=data.get('prices', {}),
                    volume_24h=data.get('volume_24h', 0.0),
                    liquidity=data.get('liquidity', 0.0),
                    resolution_time=data.get('resolution_time'),
                    source=data.get('source', 'polymarket'),
                )

        elif data_type == 'prediction_market_resolution':
            market_id = data.get('market_id', '')
            winning_outcome = data.get('winning_outcome', '')
            # Resolve all open bets on this market
            bets_to_resolve = [
                bid for bid, b in self.open_bets.items()
                if b.market.market_id == market_id
            ]
            for bid in bets_to_resolve:
                self.resolve_bet(bid, winning_outcome)

    # ─── Portfolio Analytics ──────────────────────────────────────────

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get planktonXD-style portfolio summary."""
        return {
            'strategy': 'PlanktonXD Prediction Harvester',
            'bankroll': round(self.bankroll, 2),
            'peak_bankroll': round(self.peak_bankroll, 2),
            'net_profit': round(self.stats.net_profit, 2),
            'total_roi': f"{self.stats.roi:.1%}",
            'total_bets': self.stats.total_bets,
            'win_rate': f"{self.stats.win_rate:.1%}",
            'winning_bets': self.stats.winning_bets,
            'losing_bets': self.stats.losing_bets,
            'biggest_win': round(self.stats.biggest_win, 2),
            'biggest_loss': round(self.stats.biggest_loss, 2),
            'avg_bet_size': round(self.stats.avg_bet_size, 2),
            'open_positions': len(self.open_bets),
            'daily_bets_today': self._daily_bet_count,
            'categories_traded': self.stats.categories_traded,
            'category_exposure': {
                k.value: round(v, 2) for k, v in self.category_exposure.items()
            },
        }

    def get_expected_value_analysis(self) -> Dict[str, Any]:
        """
        EV analysis explaining why the math works.

        planktonXD's math:
          - 100 bets at $15 each, priced at 1¢ (implied 1% prob)
          - Total cost: $1,500
          - If true probability is 3% (market underprices 3x):
            - Expected winners: 3
            - Each winner pays: $15 / 0.01 = $1,500
            - Expected return: 3 × $1,500 = $4,500
            - Expected profit: $4,500 - $1,500 = $3,000 (200% ROI)
        """
        avg_price = 0.01          # Average entry price
        true_prob_multiple = 3.0  # Market underprices by this factor
        bets_per_batch = 100
        bet_size = 15.0

        total_cost = bets_per_batch * bet_size
        payout_per_win = bet_size / avg_price
        expected_winners = bets_per_batch * avg_price * true_prob_multiple
        expected_return = expected_winners * payout_per_win
        expected_profit = expected_return - total_cost

        return {
            'model': 'planktonXD Tail Harvesting EV',
            'avg_entry_price': avg_price,
            'true_prob_multiple': true_prob_multiple,
            'bets_per_batch': bets_per_batch,
            'bet_size_usd': bet_size,
            'total_cost': total_cost,
            'payout_per_winner': payout_per_win,
            'expected_winners': expected_winners,
            'expected_return': expected_return,
            'expected_profit': expected_profit,
            'expected_roi': f"{expected_profit / total_cost:.0%}",
            'note': (
                "Even with only 3% true probability on '1% implied' events, "
                "the strategy is massively +EV. planktonXD's edge: the crowd "
                "systematically underprices tail events across thousands of markets."
            ),
        }

    # ─── Helpers ──────────────────────────────────────────────────────

    def _maybe_reset_daily_counters(self):
        """Reset daily counters at midnight."""
        now = datetime.now()
        if self._last_reset_date is None or now.date() != self._last_reset_date.date():
            self.stats.daily_bet_count = self._daily_bet_count
            self._daily_bet_count = 0
            self._daily_invested = 0.0
            self._last_reset_date = now

    async def _close_all_positions(self):
        """Close all open positions (for shutdown)."""
        for bet_id in list(self.open_bets.keys()):
            bet = self.open_bets.pop(bet_id)
            bet.state = PositionState.EXPIRED_WORTHLESS
            bet.pnl = -bet.cost
            bet.closed_at = datetime.now()
            self.closed_bets.append(bet)
        logger.info("All open prediction market positions closed")

    async def _unsubscribe_market_data(self):
        """Unsubscribe from data feeds."""
        logger.debug("PlanktonXD: market data unsubscribe delegated to communication framework")


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

class PlanktonXDSimulator:
    """
    Monte Carlo simulator for the planktonXD strategy.

    Validates the math by simulating thousands of bets with configurable
    parameters to show expected outcomes.
    """

    def __init__(
        self,
        starting_bankroll: float = 1000.0,
        bets_per_day: int = 170,
        days: int = 365,
        avg_bet_size: float = 15.0,
        avg_entry_price: float = 0.01,
        true_prob_multiple: float = 3.0,
    ):
        self.starting_bankroll = starting_bankroll
        self.bets_per_day = bets_per_day
        self.days = days
        self.avg_bet_size = avg_bet_size
        self.avg_entry_price = avg_entry_price
        self.true_prob_multiple = true_prob_multiple

    def run_simulation(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Run a single Monte Carlo path."""
        if seed is not None:
            random.seed(seed)

        bankroll = self.starting_bankroll
        peak = bankroll
        total_bets = 0
        wins = 0
        losses = 0
        biggest_win = 0.0
        daily_pnls = []

        true_prob = self.avg_entry_price * self.true_prob_multiple

        for day in range(self.days):
            daily_pnl = 0.0

            for _ in range(self.bets_per_day):
                if bankroll < self.avg_bet_size:
                    break  # Bust

                # Size each bet (bounded by bankroll)
                bet = min(self.avg_bet_size, bankroll * 0.05)
                bankroll -= bet
                total_bets += 1

                # Simulate outcome
                if random.random() < true_prob:
                    # Winner — pays $1 per share, shares = bet / entry_price
                    payout = bet / self.avg_entry_price
                    profit = payout - bet
                    bankroll += payout
                    wins += 1
                    biggest_win = max(biggest_win, profit)
                    daily_pnl += profit
                else:
                    losses += 1
                    daily_pnl -= bet

            peak = max(peak, bankroll)
            daily_pnls.append(daily_pnl)

        return {
            'final_bankroll': round(bankroll, 2),
            'peak_bankroll': round(peak, 2),
            'net_profit': round(bankroll - self.starting_bankroll, 2),
            'total_roi': f"{(bankroll - self.starting_bankroll) / self.starting_bankroll:.0%}",
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': f"{wins / max(total_bets, 1):.1%}",
            'biggest_win': round(biggest_win, 2),
            'avg_daily_pnl': round(sum(daily_pnls) / max(len(daily_pnls), 1), 2),
            'max_drawdown': round(
                min(daily_pnls) if daily_pnls else 0, 2
            ),
            'profitable_days': sum(1 for d in daily_pnls if d > 0),
            'losing_days': sum(1 for d in daily_pnls if d < 0),
        }

    def run_monte_carlo(
        self, num_paths: int = 1000
    ) -> Dict[str, Any]:
        """Run multiple simulation paths and return statistics."""
        results = [self.run_simulation(seed=i) for i in range(num_paths)]

        finals = [r['final_bankroll'] for r in results]
        profits = [r['net_profit'] for r in results]

        return {
            'paths': num_paths,
            'starting_bankroll': self.starting_bankroll,
            'params': {
                'bets_per_day': self.bets_per_day,
                'days': self.days,
                'avg_bet_size': self.avg_bet_size,
                'avg_entry_price': self.avg_entry_price,
                'true_prob_multiple': self.true_prob_multiple,
            },
            'median_final': round(sorted(finals)[len(finals) // 2], 2),
            'mean_final': round(sum(finals) / len(finals), 2),
            'best_case': round(max(finals), 2),
            'worst_case': round(min(finals), 2),
            'pct_profitable': f"{sum(1 for p in profits if p > 0) / len(profits):.0%}",
            'mean_profit': round(sum(profits) / len(profits), 2),
            'median_profit': round(
                sorted(profits)[len(profits) // 2], 2
            ),
            'p10_profit': round(
                sorted(profits)[int(len(profits) * 0.1)], 2
            ),
            'p90_profit': round(
                sorted(profits)[int(len(profits) * 0.9)], 2
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY / REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════

def create_planktonxd_strategy(
    communication: CommunicationFramework,
    audit_logger: AuditLogger,
    bankroll: float = 1000.0,
) -> PlanktonXDPredictionHarvester:
    """Factory function to create and configure a PlanktonXD harvester."""
    config = StrategyConfig(
        strategy_id="s51_planktonxd_prediction_harvester",
        name="PlanktonXD Prediction Market Harvester",
        strategy_type="prediction_market_arbitrage",
        edge_source="tail_event_mispricing",
        time_horizon="intraday_to_weekly",
        complexity="medium",
        data_requirements=[
            "prediction_market_prices",
            "prediction_market_orderbook",
            "prediction_market_resolution",
            "crypto_spot_price",
        ],
        execution_requirements=[
            "polymarket_api",
            "automated_execution",
            "24_7_scanning",
        ],
        risk_envelope={
            "max_single_bet_usd": 25.0,
            "max_daily_bets": 200,
            "max_open_positions": 500,
            "max_category_exposure_pct": 25.0,
            "bankroll_protection_floor_pct": 10.0,
        },
        cross_department_dependencies={
            "CryptoIntelligence": ["crypto_price_feeds"],
            "BigBrainIntelligence": ["probability_estimation"],
        },
    )

    harvester = PlanktonXDPredictionHarvester(config, communication, audit_logger)
    harvester.bankroll = bankroll
    harvester.peak_bankroll = bankroll
    return harvester
