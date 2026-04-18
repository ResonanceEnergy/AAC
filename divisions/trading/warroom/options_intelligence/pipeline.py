"""
Options Intelligence Pipeline — Integration Layer
===================================================
Wires the 5 intelligence modules into the existing AAC strategy pipeline:

1. FlowSignalEngine → Conviction multipliers for MacroCrisisPutEngine
2. UniverseExpander → Dynamic candidates for OptionsScanner
3. AITradeScorer   → Score/filter before order generation
4. SkewOptimizer   → Optimal strike selection for each target
5. FeedbackLoop    → Log fills and tune parameters

Usage:
    pipeline = OptionsIntelligencePipeline()
    result = pipeline.run(assessment, underlying_prices)
    # result.orders — enhanced PutOrderSpecs
    # result.scores — AI scores for each order
    # result.dynamic_tickers — new universe additions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from strategies.macro_crisis_put_strategy import (
    PUT_PLAYBOOK,
    CrisisAssessment,
    MacroCrisisPutEngine,
    PutOrderSpec,
)
from strategies.options_intelligence.ai_scorer import (
    AITradeScorer,
    TradeScore,
    TradeSetup,
)
from strategies.options_intelligence.feedback import (
    FeedbackLoop,
    FillRecord,
)
from strategies.options_intelligence.flow_signals import (
    FlowConviction,
    FlowEntry,
    FlowSignalEngine,
)
from strategies.options_intelligence.skew_optimizer import (
    OptimalStrike,
    SkewOptimizer,
)
from strategies.options_intelligence.universe import (
    DynamicCandidate,
    UniverseExpander,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete output from the intelligence pipeline."""
    # Original orders from crisis engine (unmodified)
    base_orders: List[PutOrderSpec]

    # Enhanced with intelligence
    scored_orders: List[Dict[str, Any]]     # Orders + their AI scores
    rejected_orders: List[Dict[str, Any]]   # Orders that failed scoring

    # Flow intelligence
    flow_convictions: Dict[str, FlowConviction]
    entry_triggers: List[FlowEntry]

    # Universe expansion
    dynamic_candidates: List[DynamicCandidate]

    # Optimal strikes (if chain data available)
    optimal_strikes: Dict[str, OptimalStrike]

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    pipeline_version: str = "1.0"

    @property
    def actionable_orders(self) -> List[Dict[str, Any]]:
        """Orders that passed AI scoring (score >= 60)."""
        return [o for o in self.scored_orders if o["score"].is_actionable]

    @property
    def strong_orders(self) -> List[Dict[str, Any]]:
        """High-conviction orders (score >= 80)."""
        return [o for o in self.scored_orders if o["score"].is_strong]

    def summary(self) -> str:
        lines = [
            f"=== Options Intelligence Pipeline v{self.pipeline_version} ===",
            f"Base orders: {len(self.base_orders)}",
            f"Scored: {len(self.scored_orders)} | Rejected: {len(self.rejected_orders)}",
            f"Actionable: {len(self.actionable_orders)} | Strong: {len(self.strong_orders)}",
            f"Flow triggers: {len(self.entry_triggers)}",
            f"Dynamic candidates: {len(self.dynamic_candidates)}",
            f"Optimal strikes found: {len(self.optimal_strikes)}",
        ]
        return "\n".join(lines)


class OptionsIntelligencePipeline:
    """
    Main pipeline that orchestrates all 5 intelligence modules.

    Can run in two modes:
    1. Sync mode (score_trade_sync) — heuristic only, no LLM, no async
    2. Async mode (full pipeline) — LLM scoring + async UW API calls

    Usage:
        pipeline = OptionsIntelligencePipeline()

        # Sync mode (always available)
        result = pipeline.run_sync(assessment, prices)

        # Async mode (with UW + LLM)
        result = await pipeline.run(assessment, prices, uw_client=uw_client)
    """

    def __init__(
        self,
        account_balance: float = 8800.0,
        min_score: int = 60,
        paper_trading: bool = True,
    ):
        self.flow_engine = FlowSignalEngine()
        self.universe_expander = UniverseExpander()
        self.scorer = AITradeScorer()
        self.skew_optimizer = SkewOptimizer()
        self.feedback = FeedbackLoop()
        self.min_score = min_score
        self.account_balance = account_balance

        self._crisis_engine = MacroCrisisPutEngine(
            account_balance=account_balance,
            paper_trading=paper_trading,
        )

    def run_sync(
        self,
        assessment: CrisisAssessment,
        underlying_prices: Dict[str, float],
        flow_data: Optional[Dict[str, Any]] = None,
        chain_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        regime: str = "unknown",
        vix: float = 0.0,
    ) -> PipelineResult:
        """
        Run the full pipeline in synchronous mode (heuristic scoring).

        Args:
            assessment: CrisisAssessment from CrisisMonitor
            underlying_prices: {symbol: price}
            flow_data: Pre-fetched UW snapshot data (optional)
            chain_data: Pre-fetched options chains {ticker: [contracts]} (optional)
            regime: Active regime name from RegimeEngine
            vix: Current VIX level
        """
        # Step 1: Generate base orders from crisis engine
        base_orders = self._crisis_engine.generate_orders(assessment, underlying_prices)
        logger.info("Base orders generated: %d", len(base_orders))

        # Step 2: Flow analysis (if data available)
        convictions_list: List[FlowConviction] = []
        flow_convictions: Dict[str, FlowConviction] = {}
        entry_triggers: List[FlowEntry] = []
        if flow_data:
            convictions_list = self.flow_engine.analyze_flow_sync(
                flow_data=flow_data.get("flow", []),
                dark_pool_data=flow_data.get("dark_pool", []),
                congress_data=flow_data.get("congress", []),
            )
            flow_convictions = {c.ticker: c for c in convictions_list}
            entry_triggers = self.flow_engine.check_entry_triggers(convictions_list)
            logger.info("Flow convictions: %d tickers, %d triggers",
                        len(flow_convictions), len(entry_triggers))

        # Step 3: Universe expansion (from flow convictions)
        dynamic_candidates: List[DynamicCandidate] = []
        if convictions_list:
            existing = {t.symbol for t in PUT_PLAYBOOK}
            dynamic_candidates = self.universe_expander.discover_sync(
                flow_convictions=convictions_list,
                existing_universe=existing,
            )
            logger.info("Dynamic candidates found: %d", len(dynamic_candidates))

        # Step 4: Score each order with AI scorer
        scored_orders: List[Dict[str, Any]] = []
        rejected_orders: List[Dict[str, Any]] = []

        for order in base_orders:
            conviction = flow_convictions.get(order.symbol)
            setup = TradeSetup(
                ticker=order.symbol,
                direction="put",
                strike=order.strike,
                expiry=order.expiry,
                dte=45,  # Approximate from target
                premium=order.max_price,
                delta=order.target_delta,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                iv=0.22,  # Default estimate
                flow_conviction=conviction.conviction if conviction else 0.0,
                put_call_ratio=conviction.put_call_ratio if conviction else 1.0,
                sweep_count=conviction.sweep_count if conviction else 0,
                dark_pool_notional=conviction.dark_pool_notional if conviction else 0.0,
                regime=regime,
                vix=vix,
                hy_spread_bps=0.0,
                existing_positions=len(base_orders),
                account_balance=self.account_balance,
                risk_pct=(order.max_price * order.contracts * 100) / self.account_balance
                    if self.account_balance > 0 else 0,
            )

            score = self.scorer.score_trade_sync(setup)

            entry = {"order": order, "score": score, "setup": setup}
            if score.is_actionable:
                scored_orders.append(entry)
            else:
                rejected_orders.append(entry)

        logger.info("Scored: %d actionable, %d rejected (min_score=%d)",
                     len(scored_orders), len(rejected_orders), self.min_score)

        # Step 5: Skew optimization (if chain data available)
        optimal_strikes: Dict[str, OptimalStrike] = {}
        if chain_data:
            for order in base_orders:
                if order.symbol in chain_data:
                    spot = underlying_prices.get(order.symbol, 0)
                    if spot > 0:
                        optimal = self.skew_optimizer.find_optimal_strike(
                            chain=chain_data[order.symbol],
                            spot=spot,
                            target_delta=order.target_delta,
                            ticker=order.symbol,
                            expiry=order.expiry,
                        )
                        if optimal:
                            optimal_strikes[order.symbol] = optimal

            logger.info("Optimal strikes found: %d/%d tickers",
                        len(optimal_strikes), len(chain_data))

        return PipelineResult(
            base_orders=base_orders,
            scored_orders=scored_orders,
            rejected_orders=rejected_orders,
            flow_convictions=flow_convictions,
            entry_triggers=entry_triggers,
            dynamic_candidates=dynamic_candidates,
            optimal_strikes=optimal_strikes,
        )

    async def run(
        self,
        assessment: CrisisAssessment,
        underlying_prices: Dict[str, float],
        uw_client: Optional[Any] = None,
        chain_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        regime: str = "unknown",
        vix: float = 0.0,
    ) -> PipelineResult:
        """
        Run the full pipeline with async UW API calls and LLM scoring.

        Same as run_sync but uses:
        - Async UW client for live flow data
        - LLM scoring when API keys available
        """
        # Step 1: Base orders
        base_orders = self._crisis_engine.generate_orders(assessment, underlying_prices)

        # Step 2: Live flow analysis (async)
        convictions_list: List[FlowConviction] = []
        flow_convictions: Dict[str, FlowConviction] = {}
        entry_triggers: List[FlowEntry] = []
        if uw_client:
            convictions_list = await self.flow_engine.analyze_flow(uw_client)
            flow_convictions = {c.ticker: c for c in convictions_list}
            entry_triggers = self.flow_engine.check_entry_triggers(convictions_list)

        # Step 3: Universe expansion (async with UW client)
        dynamic_candidates: List[DynamicCandidate] = []
        if uw_client:
            existing = {t.symbol for t in PUT_PLAYBOOK}
            dynamic_candidates = await self.universe_expander.discover(
                uw_client=uw_client,
                existing_universe=existing,
                flow_convictions=convictions_list,
            )

        # Step 4: AI scoring (async with LLM)
        scored_orders: List[Dict[str, Any]] = []
        rejected_orders: List[Dict[str, Any]] = []

        for order in base_orders:
            conviction = flow_convictions.get(order.symbol)
            setup = TradeSetup(
                ticker=order.symbol,
                direction="put",
                strike=order.strike,
                expiry=order.expiry,
                dte=45,
                premium=order.max_price,
                delta=order.target_delta,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                iv=0.22,
                flow_conviction=conviction.conviction if conviction else 0.0,
                put_call_ratio=conviction.put_call_ratio if conviction else 1.0,
                sweep_count=conviction.sweep_count if conviction else 0,
                dark_pool_notional=conviction.dark_pool_notional if conviction else 0.0,
                regime=regime,
                vix=vix,
                hy_spread_bps=0.0,
                existing_positions=len(base_orders),
                account_balance=self.account_balance,
                risk_pct=(order.max_price * order.contracts * 100) / self.account_balance
                    if self.account_balance > 0 else 0,
            )

            score = await self.scorer.score_trade(setup)

            entry = {"order": order, "score": score, "setup": setup}
            if score.is_actionable:
                scored_orders.append(entry)
            else:
                rejected_orders.append(entry)

        # Step 5: Skew optimization
        optimal_strikes: Dict[str, OptimalStrike] = {}
        if chain_data:
            for order in base_orders:
                if order.symbol in chain_data:
                    spot = underlying_prices.get(order.symbol, 0)
                    if spot > 0:
                        optimal = self.skew_optimizer.find_optimal_strike(
                            chain=chain_data[order.symbol],
                            spot=spot,
                            target_delta=order.target_delta,
                            ticker=order.symbol,
                            expiry=order.expiry,
                        )
                        if optimal:
                            optimal_strikes[order.symbol] = optimal

        return PipelineResult(
            base_orders=base_orders,
            scored_orders=scored_orders,
            rejected_orders=rejected_orders,
            flow_convictions=flow_convictions,
            entry_triggers=entry_triggers,
            dynamic_candidates=dynamic_candidates,
            optimal_strikes=optimal_strikes,
        )

    def log_fill(
        self,
        order: PutOrderSpec,
        fill_price: float,
        score: TradeScore,
        flow_conviction: float = 0.0,
        skew_value_score: float = 0.0,
        regime: str = "unknown",
        vix: float = 0.0,
    ) -> FillRecord:
        """Log a fill to the feedback loop."""
        fill = FillRecord(
            fill_id=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{order.symbol}",
            timestamp=datetime.now().isoformat(),
            ticker=order.symbol,
            direction="put",
            strike=order.strike,
            expiry=order.expiry,
            dte_at_entry=45,
            quantity=order.contracts,
            fill_price=fill_price,
            total_cost=fill_price * order.contracts * 100,
            delta=order.target_delta,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            iv=0.22,
            flow_conviction=flow_conviction,
            ai_score=score.score,
            skew_value_score=skew_value_score,
            regime=regime,
            vix_at_entry=vix,
        )
        self.feedback.log_fill(fill)
        return fill
