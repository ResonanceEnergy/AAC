"""
IBKR Executor — Options Intelligence → IBKR Bridge
=====================================================
Connects the OptionsIntelligencePipeline to IBKR for live execution:

1. Fetches live prices, option chains, and quotes from IBKR
2. Runs the intelligence pipeline with real market data
3. Executes actionable orders through IBKR's options API
4. Logs fills to the feedback loop for performance tracking

Usage:
    executor = IBKRExecutor()
    await executor.connect()
    result = await executor.run_pipeline(assessment, regime="CRISIS", vix=28.5)

    # Review before executing
    for order in result.actionable_orders:
        print(order["order"].symbol, order["score"].score)

    # Execute all actionable orders
    fills = await executor.execute_orders(result)

    # Or one at a time
    fill = await executor.execute_single(result.actionable_orders[0])
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from TradingExecution.exchange_connectors.ibkr_connector import IBKRConnector
from TradingExecution.exchange_connectors.base_connector import ExchangeOrder
from strategies.options_intelligence.pipeline import (
    OptionsIntelligencePipeline,
    PipelineResult,
)
from strategies.options_intelligence.feedback import FillRecord
from strategies.options_intelligence.skew_optimizer import OptimalStrike
from strategies.macro_crisis_put_strategy import CrisisAssessment, PutOrderSpec

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a single order through IBKR."""
    order: PutOrderSpec
    exchange_order: Optional[ExchangeOrder]
    fill_record: Optional[FillRecord]
    success: bool
    error: Optional[str] = None
    used_optimal_strike: bool = False


@dataclass
class ExecutionSummary:
    """Summary of a batch execution run."""
    total_attempted: int
    successful: int
    failed: int
    results: List[ExecutionResult]
    total_premium_committed: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def report(self) -> str:
        lines = [
            f"=== IBKR Execution Summary ===",
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Attempted: {self.total_attempted}",
            f"Successful: {self.successful} | Failed: {self.failed}",
            f"Total premium committed: ${self.total_premium_committed:,.2f}",
            "",
        ]
        for r in self.results:
            status = "OK" if r.success else f"FAIL: {r.error}"
            oid = r.exchange_order.order_id if r.exchange_order else "N/A"
            lines.append(
                f"  {r.order.symbol} {r.order.expiry} ${r.order.strike} "
                f"x{r.order.contracts} @ ${r.order.max_price:.2f} — {status} (id={oid})"
            )
        return "\n".join(lines)


class IBKRExecutor:
    """
    Bridge between the Options Intelligence Pipeline and IBKR execution.

    Handles the full lifecycle:
        connect → fetch data → run pipeline → execute → log fills
    """

    def __init__(
        self,
        host: str = '',
        port: int = 0,
        client_id: int = 0,
        account: str = '',
        paper: bool = True,
        account_balance: float = 8800.0,
        min_score: int = 60,
        dry_run: bool = True,
    ):
        self._connector = IBKRConnector(
            host=host,
            port=port,
            client_id=client_id,
            account=account,
            paper=paper,
        )
        self._pipeline = OptionsIntelligencePipeline(
            account_balance=account_balance,
            min_score=min_score,
            paper_trading=paper,
        )
        self._dry_run = dry_run
        self._connected = False

    @property
    def connector(self) -> IBKRConnector:
        """Direct access to the IBKR connector."""
        return self._connector

    @property
    def pipeline(self) -> OptionsIntelligencePipeline:
        """Direct access to the pipeline."""
        return self._pipeline

    # ── Connection ──────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway."""
        self._connected = await self._connector.connect()
        if self._connected:
            logger.info("IBKRExecutor connected to IBKR")
        else:
            logger.error("IBKRExecutor failed to connect to IBKR")
        return self._connected

    async def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._connected:
            await self._connector.disconnect()
            self._connected = False
            logger.info("IBKRExecutor disconnected")

    # ── Data Fetching ───────────────────────────────────────────────────

    async def fetch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch live prices for all symbols from IBKR."""
        prices: Dict[str, float] = {}
        for symbol in symbols:
            try:
                ticker = await self._connector.get_ticker(symbol)
                if ticker and ticker.last_price and ticker.last_price > 0:
                    prices[symbol] = ticker.last_price
                elif ticker and ticker.bid and ticker.bid > 0:
                    prices[symbol] = (ticker.bid + ticker.ask) / 2
            except Exception as e:
                logger.warning("Failed to fetch price for %s: %s", symbol, e)
        return prices

    async def fetch_chains(
        self,
        symbols: List[str],
        expiry: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch option chains from IBKR for pipeline skew analysis.

        Converts IBKR chain data into the format expected by SkewOptimizer.
        """
        chains: Dict[str, List[Dict[str, Any]]] = {}
        for symbol in symbols:
            try:
                chain_data = await self._connector.get_option_chain(symbol, expiry)
                if chain_data:
                    # Fetch quotes for each strike to build the full chain
                    # the skew optimizer needs
                    contracts = await self._build_chain_contracts(
                        symbol, chain_data, expiry
                    )
                    if contracts:
                        chains[symbol] = contracts
            except Exception as e:
                logger.warning("Failed to fetch chain for %s: %s", symbol, e)
        return chains

    async def _build_chain_contracts(
        self,
        symbol: str,
        chain_data: List[Dict[str, Any]],
        target_expiry: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build detailed contract data from IBKR chain for the skew optimizer.

        Fetches quotes for puts at each strike near the money.
        """
        contracts: List[Dict[str, Any]] = []

        # Pick the first matching expiry
        if not chain_data:
            return contracts

        expiry_data = chain_data[0]
        expiry = target_expiry or expiry_data["expiry"]
        strikes = expiry_data["strikes"]

        # Get underlying price to filter strikes near the money
        try:
            ticker = await self._connector.get_ticker(symbol)
            spot = ticker.last_price if ticker and ticker.last_price > 0 else 0
        except Exception:
            spot = 0

        if spot <= 0:
            return contracts

        # Filter to strikes within 20% of spot (OTM puts)
        lower = spot * 0.80
        upper = spot * 1.05
        relevant_strikes = [s for s in strikes if lower <= s <= upper]

        # Limit to avoid excessive API calls
        if len(relevant_strikes) > 15:
            step = len(relevant_strikes) // 15
            relevant_strikes = relevant_strikes[::step]

        for strike in relevant_strikes:
            try:
                quote = await self._connector.get_option_quote(
                    symbol, expiry, strike, "P"
                )
                if quote:
                    contracts.append({
                        "strike": strike,
                        "expiry": expiry,
                        "right": "P",
                        "bid": quote.get("bid") or 0.0,
                        "ask": quote.get("ask") or 0.0,
                        "iv": quote.get("iv") or 0.0,
                        "volume": quote.get("volume") or 0,
                        "open_interest": quote.get("open_interest") or 0,
                        "delta": -(0.5 * strike / spot) if spot > 0 else 0,
                    })
            except Exception as e:
                logger.debug("Failed quote for %s %s %.1f P: %s", symbol, expiry, strike, e)

        return contracts

    async def fetch_positions(self) -> List[Dict[str, Any]]:
        """Get current IBKR positions."""
        return await self._connector.get_positions()

    async def fetch_account_summary(self) -> Dict[str, Any]:
        """Get IBKR account summary."""
        return await self._connector.get_account_summary()

    # ── Pipeline Execution ──────────────────────────────────────────────

    async def run_pipeline(
        self,
        assessment: CrisisAssessment,
        symbols: Optional[List[str]] = None,
        flow_data: Optional[Dict[str, Any]] = None,
        regime: str = "unknown",
        vix: float = 0.0,
        fetch_chains: bool = True,
        target_expiry: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run the full intelligence pipeline with live IBKR data.

        Steps:
            1. Fetch live prices from IBKR for all symbols
            2. Optionally fetch option chains for skew analysis
            3. Run the pipeline with real data
            4. Return scored, filtered, optimized orders

        Args:
            assessment: CrisisAssessment from the crisis monitor
            symbols: Override symbols to fetch (default: from PUT_PLAYBOOK)
            flow_data: Pre-fetched UW flow data (optional)
            regime: Current regime from RegimeEngine
            vix: Current VIX level
            fetch_chains: Whether to fetch live option chains
            target_expiry: Target expiration for chain data
        """
        if not self._connected:
            raise RuntimeError("Not connected to IBKR. Call connect() first.")

        from strategies.macro_crisis_put_strategy import PUT_PLAYBOOK
        if symbols is None:
            symbols = list({t.symbol for t in PUT_PLAYBOOK})

        logger.info("Fetching live IBKR data for %d symbols...", len(symbols))

        # Fetch live prices
        prices = await self.fetch_prices(symbols)
        logger.info("Got prices for %d/%d symbols", len(prices), len(symbols))

        # Fetch chains if requested
        chain_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
        if fetch_chains:
            chain_data = await self.fetch_chains(symbols, target_expiry)
            logger.info("Got chains for %d symbols", len(chain_data) if chain_data else 0)

        # Run the pipeline
        result = self._pipeline.run_sync(
            assessment=assessment,
            underlying_prices=prices,
            flow_data=flow_data,
            chain_data=chain_data,
            regime=regime,
            vix=vix,
        )

        logger.info(
            "Pipeline complete — %d actionable, %d strong, %d rejected",
            len(result.actionable_orders),
            len(result.strong_orders),
            len(result.rejected_orders),
        )

        return result

    # ── Order Execution ─────────────────────────────────────────────────

    async def execute_orders(
        self,
        result: PipelineResult,
        strong_only: bool = False,
    ) -> ExecutionSummary:
        """
        Execute actionable orders from a pipeline result through IBKR.

        Args:
            result: PipelineResult from run_pipeline()
            strong_only: If True, only execute score>=80 orders
        """
        orders = result.strong_orders if strong_only else result.actionable_orders

        if not orders:
            logger.info("No orders to execute")
            return ExecutionSummary(
                total_attempted=0, successful=0, failed=0, results=[]
            )

        logger.info(
            "Executing %d %sorders (dry_run=%s)",
            len(orders),
            "strong " if strong_only else "",
            self._dry_run,
        )

        results: List[ExecutionResult] = []
        total_premium = 0.0

        for entry in orders:
            order: PutOrderSpec = entry["order"]
            score = entry["score"]

            # Use optimal strike if available
            optimal = result.optimal_strikes.get(order.symbol)
            actual_strike = order.strike
            actual_price = order.max_price
            used_optimal = False

            if optimal and optimal.strike != order.strike:
                logger.info(
                    "%s: Using optimal strike $%.1f (score=%d) instead of $%.1f",
                    order.symbol, optimal.strike, optimal.value_score, order.strike,
                )
                actual_strike = optimal.strike
                used_optimal = True

            exec_result = await self.execute_single_order(
                order=order,
                score=score,
                strike_override=actual_strike if used_optimal else None,
                flow_conviction=result.flow_convictions.get(order.symbol),
                regime=entry["setup"].regime if "setup" in entry else "unknown",
                vix=entry["setup"].vix if "setup" in entry else 0.0,
            )
            exec_result.used_optimal_strike = used_optimal
            results.append(exec_result)

            if exec_result.success:
                total_premium += actual_price * order.contracts * 100

        successful = sum(1 for r in results if r.success)
        summary = ExecutionSummary(
            total_attempted=len(results),
            successful=successful,
            failed=len(results) - successful,
            results=results,
            total_premium_committed=total_premium,
        )

        logger.info(summary.report())
        return summary

    async def execute_single_order(
        self,
        order: PutOrderSpec,
        score: Any,
        strike_override: Optional[float] = None,
        flow_conviction: Any = None,
        regime: str = "unknown",
        vix: float = 0.0,
    ) -> ExecutionResult:
        """
        Execute a single order through IBKR and log the fill.

        Args:
            order: PutOrderSpec to execute
            score: TradeScore from the AI scorer
            strike_override: Use this strike instead of order.strike
            flow_conviction: FlowConviction for this ticker
            regime: Current regime
            vix: Current VIX
        """
        if not self._connected:
            return ExecutionResult(
                order=order, exchange_order=None, fill_record=None,
                success=False, error="Not connected to IBKR",
            )

        strike = strike_override or order.strike
        right = "P" if order.side == "buy" else "C"  # Protective puts

        if self._dry_run:
            logger.info(
                "[DRY RUN] Would execute: %s %s %s $%.1f x%d @ $%.2f (score=%d)",
                order.side, order.symbol, order.expiry, strike,
                order.contracts, order.max_price,
                score.score if hasattr(score, "score") else 0,
            )
            return ExecutionResult(
                order=order, exchange_order=None, fill_record=None,
                success=True, error=None,
            )

        try:
            # Get a fresh quote to validate price
            quote = await self._connector.get_option_quote(
                order.symbol, order.expiry, strike, right
            )
            ask = quote.get("ask")
            if ask and ask > order.max_price * 1.5:
                return ExecutionResult(
                    order=order, exchange_order=None, fill_record=None,
                    success=False,
                    error=f"Ask ${ask:.2f} exceeds 150% of max ${order.max_price:.2f}",
                )

            # Place the order
            exchange_order = await self._connector.create_option_order(
                symbol=order.symbol,
                expiry=order.expiry,
                strike=strike,
                right=right,
                side=order.side,
                quantity=order.contracts,
                order_type=order.order_type,
                price=order.max_price,
            )

            # Log fill to feedback loop
            fill_record = self._pipeline.log_fill(
                order=order,
                fill_price=order.max_price,
                score=score,
                flow_conviction=(
                    flow_conviction.conviction if flow_conviction else 0.0
                ),
                skew_value_score=0.0,
                regime=regime,
                vix=vix,
            )

            logger.info(
                "Executed: %s %s $%.1f x%d → order_id=%s",
                order.symbol, order.expiry, strike, order.contracts,
                exchange_order.order_id,
            )

            return ExecutionResult(
                order=order,
                exchange_order=exchange_order,
                fill_record=fill_record,
                success=True,
            )

        except Exception as e:
            logger.error("Failed to execute %s: %s", order.symbol, e)
            return ExecutionResult(
                order=order, exchange_order=None, fill_record=None,
                success=False, error=str(e),
            )

    # ── Position Monitoring ─────────────────────────────────────────────

    async def get_option_positions(self) -> List[Dict[str, Any]]:
        """Get only option positions from IBKR."""
        positions = await self.fetch_positions()
        return [p for p in positions if p.get("sec_type") == "OPT"]

    async def check_exit_conditions(
        self,
        profit_target_pct: float = 50.0,
        stop_loss_pct: float = -80.0,
    ) -> List[Dict[str, Any]]:
        """
        Check option positions for exit conditions.

        Returns positions that hit profit target or stop loss.
        """
        positions = await self.get_option_positions()
        exits: List[Dict[str, Any]] = []

        for pos in positions:
            avg_cost = pos.get("avg_cost", 0)
            market_value = pos.get("market_value", 0)
            if avg_cost <= 0:
                continue

            pnl_pct = ((market_value - avg_cost) / avg_cost) * 100

            if pnl_pct >= profit_target_pct:
                exits.append({
                    **pos,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "profit_target",
                })
            elif pnl_pct <= stop_loss_pct:
                exits.append({
                    **pos,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "stop_loss",
                })

        return exits
