"""
MATRIX MAXIMIZER — Main Orchestrator
=======================================
The supreme engine. Chains every subsystem into a single execution cycle:

    NCC Governance Check
        ↓
    Live Data Feed Resolution
        ↓
    Regime Context Load
        ↓
    Intelligence Gathering (NCL, StockForecaster, UW)
        ↓
    Scenario Weight Adjustment (Bridge)
        ↓
    Monte Carlo Simulation (10,000 paths)
        ↓
    System Mandate Generation
        ↓
    Options Chain Scan + Score
        ↓
    Advanced Strategy Recommendations
        ↓
    Auto-Roll Check (existing positions)
        ↓
    7-Layer Risk Evaluation + Enhanced Risk
        ↓
    Execution (paper/live/dry-run)
        ↓
    Dashboard Recording + Alerts
        ↓
    Result Push (NCC/NCL/Dashboard)
        ↓
    JSON Output + CLI Summary

Scheduling:
    Designed for semi-daily execution (pre-market + mid-session).
    Can be triggered manually via CLI or by AAC's PipelineRunner.

CLI Usage:
    python -m strategies.matrix_maximizer.runner
    python -m strategies.matrix_maximizer.runner --account 5000 --oil 110 --vix 30
    python -m strategies.matrix_maximizer.runner --chatbot
    python -m strategies.matrix_maximizer.runner --backtest 90
    python -m strategies.matrix_maximizer.runner --schedule
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from strategies.matrix_maximizer.bridge import PillarBridge, RegimeContext
from strategies.matrix_maximizer.core import (
    DEFAULT_PRICES,
    Asset,
    MatrixConfig,
    PortfolioForecast,
    ScenarioWeights,
    SystemMandate,
)
from strategies.matrix_maximizer.greeks import BlackScholesEngine, GreeksResult
from strategies.matrix_maximizer.monte_carlo import MonteCarloEngine
from strategies.matrix_maximizer.risk import (
    CircuitBreaker,
    RiskManager,
    RiskSnapshot,
)
from strategies.matrix_maximizer.scanner import OptionsScanner, Position, PutRecommendation

logger = logging.getLogger(__name__)


class MatrixMaximizer:
    """Supreme orchestrator for the geopolitical bear market options engine.

    Wires all subsystems: data feeds, intelligence, execution, alerts,
    advanced strategies, backtester, dashboard, chatbot, scheduler.

    Usage:
        mm = MatrixMaximizer(config)
        result = mm.run_full_cycle(positions=[...])
    """

    VERSION = "2.0.0"

    def __init__(self, config: Optional[MatrixConfig] = None) -> None:
        self.config = config or MatrixConfig()

        # ── Core engines (original) ──
        self.mc = MonteCarloEngine(self.config)
        self.bs = BlackScholesEngine()
        self.scanner = OptionsScanner(self.config, self.bs)
        self.risk = RiskManager(self.config)
        self.bridge = PillarBridge(self.config)

        # ── Extended subsystems (lazy-initialized for fault tolerance) ──
        self.data_feeds = self._init_optional("data_feeds")
        self.intelligence = self._init_intelligence(self.data_feeds)
        self.execution = self._init_optional("execution")
        self.alerts = self._init_optional("alerts")
        self.advanced = self._init_optional("advanced_strategies")
        self.backtester = self._init_optional("backtester")
        self.dashboard = self._init_optional("dashboard")
        self.chatbot_engine: Any = None  # Initialized on demand via get_chatbot()

        self._run_count = 0
        self._last_result: Optional[Dict[str, Any]] = None
        logger.info(
            "MATRIX MAXIMIZER v%s initialized — account=$%s, tickers=%d",
            self.VERSION, f"{self.config.account_size:,.0f}", len(self.config.scan_tickers),
        )

    @staticmethod
    def _init_optional(module_name: str) -> Any:
        """Initialize an extended subsystem; return None on failure."""
        try:
            if module_name == "data_feeds":
                from strategies.matrix_maximizer.data_feeds import DataFeedManager
                return DataFeedManager()
            elif module_name == "intelligence":
                from strategies.matrix_maximizer.intelligence import IntelligenceEngine
                return IntelligenceEngine()
            elif module_name == "execution":
                from strategies.matrix_maximizer.execution import ExecutionEngine
                return ExecutionEngine()
            elif module_name == "alerts":
                from strategies.matrix_maximizer.alerts import AlertManager
                return AlertManager()
            elif module_name == "advanced_strategies":
                from strategies.matrix_maximizer.advanced_strategies import AdvancedStrategyEngine
                return AdvancedStrategyEngine()
            elif module_name == "backtester":
                from strategies.matrix_maximizer.backtester import MatrixBacktester
                return MatrixBacktester()
            elif module_name == "dashboard":
                from strategies.matrix_maximizer.dashboard import MatrixDashboard
                return MatrixDashboard()
        except Exception as exc:
            logger.debug("Optional module '%s' not available: %s", module_name, exc)
        return None

    @staticmethod
    def _init_intelligence(data_feeds: Any) -> Any:
        """Initialize IntelligenceEngine with data feed reference."""
        try:
            from strategies.matrix_maximizer.intelligence import IntelligenceEngine
            return IntelligenceEngine(data_feed_manager=data_feeds)
        except Exception as exc:
            logger.debug("Intelligence module not available: %s", exc)
        return None

    def get_chatbot(self) -> Any:
        """Get or create the chatbot instance (needs references to other subsystems)."""
        if self.chatbot_engine is not None:
            return self.chatbot_engine
        try:
            from strategies.matrix_maximizer.chatbot import ChatContext, MatrixChatbot
            ctx = ChatContext(
                runner=self,
                execution=self.execution,
                dashboard=self.dashboard,
                alerts=self.alerts,
                scheduler=None,
                backtester=self.backtester,
                data_feeds=self.data_feeds,
                intelligence=self.intelligence,
                advanced=self.advanced,
            )
            self.chatbot_engine = MatrixChatbot(ctx)
        except Exception as exc:
            logger.debug("Chatbot not available: %s", exc)
        return self.chatbot_engine

    def run_full_cycle(
        self,
        positions: Optional[List[Position]] = None,
        prices: Optional[Dict[str, float]] = None,
        daily_pnl: float = 0.0,
        cumulative_pnl: float = 0.0,
        war_active: Optional[bool] = None,
        hormuz_blocked: Optional[bool] = None,
        auto_execute: bool = False,
    ) -> Dict[str, Any]:
        """Execute one complete MATRIX MAXIMIZER cycle.

        Args:
            positions: Currently open positions (for risk + roll checks)
            prices: Override prices (ticker → price). Falls back to live → DEFAULT.
            daily_pnl: Today's P&L in dollars
            cumulative_pnl: Cumulative P&L in dollars
            war_active: Override war_active flag
            hormuz_blocked: Override hormuz_blocked flag
            auto_execute: If True, send picks to ExecutionEngine

        Returns:
            Complete cycle result dict with all subsystem outputs.
        """
        t0 = time.time()
        self._run_count += 1
        positions = positions or []
        prices = prices or {}

        result: Dict[str, Any] = {
            "version": self.VERSION,
            "run_number": self._run_count,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "running",
        }

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: NCC GOVERNANCE CHECK
        # ═══════════════════════════════════════════════════════════════
        logger.info("=" * 60)
        logger.info("  MATRIX MAXIMIZER — CYCLE %d", self._run_count)
        logger.info("=" * 60)

        if not self.bridge.should_trade():
            doctrine = self.bridge.get_doctrine_mode()
            logger.warning("NCC GOVERNANCE BLOCK: doctrine=%s — cycle aborted", doctrine)
            result["status"] = "blocked"
            result["reason"] = f"NCC doctrine: {doctrine}"
            self._last_result = result
            return result

        ncc_multiplier = self.bridge.get_risk_multiplier()
        logger.info("Step 1: NCC governance OK — multiplier=%.1f", ncc_multiplier)

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: LIVE DATA FEEDS
        # ═══════════════════════════════════════════════════════════════
        live_prices: Dict[str, float] = dict(DEFAULT_PRICES)
        feed_snapshot: Dict[str, Any] = {}

        if self.data_feeds and not prices:
            try:
                resolved = self.data_feeds.resolve_prices(
                    [a.value for a in Asset]
                )
                for ticker, price in resolved.items():
                    for asset in Asset:
                        if asset.value == ticker:
                            live_prices[asset] = price
                            break
                feed_snapshot["live_prices_resolved"] = len(resolved)

                live_vix = self.data_feeds.get_live_vix()
                if live_vix:
                    prices["vix"] = live_vix
                    feed_snapshot["live_vix"] = live_vix

                live_oil = self.data_feeds.get_live_oil()
                if live_oil:
                    prices["oil"] = live_oil
                    feed_snapshot["live_oil"] = live_oil

                logger.info("Step 2: Live data — %d prices, VIX=%.1f, Oil=$%.1f",
                            len(resolved), prices.get("vix", 0), prices.get("oil", 0))
            except Exception as exc:
                logger.warning("Step 2: Live data feed error: %s — using defaults", exc)
        else:
            logger.info("Step 2: Using provided/default prices")

        # Apply any explicit overrides on top
        for ticker, price in prices.items():
            for asset in Asset:
                if asset.value == ticker:
                    live_prices[asset] = price
                    break

        result["data_feeds"] = feed_snapshot

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: REGIME CONTEXT
        # ═══════════════════════════════════════════════════════════════
        regime = self.bridge.get_regime_context()

        if "oil" in prices:
            regime.oil_price = prices["oil"]
        if "vix" in prices:
            regime.vix = prices["vix"]
        if war_active is not None:
            regime.war_active = war_active
        if hormuz_blocked is not None:
            regime.hormuz_blocked = hormuz_blocked

        logger.info(
            "Step 3: Regime=%s (conf=%.0f) war=%s hormuz=%s oil=$%.1f vix=%.1f",
            regime.regime, regime.confidence, regime.war_active,
            regime.hormuz_blocked, regime.oil_price, regime.vix,
        )
        result["regime"] = {
            "regime": regime.regime,
            "confidence": regime.confidence,
            "war_active": regime.war_active,
            "hormuz_blocked": regime.hormuz_blocked,
            "oil_price": regime.oil_price,
            "vix": regime.vix,
        }

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: INTELLIGENCE GATHERING
        # ═══════════════════════════════════════════════════════════════
        intel_brief = None
        if self.intelligence:
            try:
                intel_brief = self.intelligence.gather_intel(
                    tickers=[a.value for a in self.config.scan_tickers],
                )
                result["intelligence"] = {
                    "signals": len(intel_brief.signals) if intel_brief else 0,
                    "regime_formulas": intel_brief.regime_formulas_armed if intel_brief else [],
                    "earnings_blackout": intel_brief.earnings_blackout if intel_brief else [],
                }
                logger.info("Step 4: Intel — %d signals, %d formulas",
                            len(intel_brief.signals) if intel_brief else 0,
                            len(intel_brief.regime_formulas_armed) if intel_brief else 0)
            except Exception as exc:
                logger.warning("Step 4: Intelligence error: %s", exc)
                result["intelligence"] = {"error": str(exc)}
        else:
            logger.info("Step 4: Intelligence module not available")

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: SCENARIO WEIGHT ADJUSTMENT
        # ═══════════════════════════════════════════════════════════════
        base_weights = ScenarioWeights()
        adjusted_weights = self.bridge.adjust_scenario_weights(base_weights, regime)
        logger.info(
            "Step 5: Scenario weights — BASE=%.0f%% BEAR=%.0f%% BULL=%.0f%%",
            adjusted_weights.base * 100, adjusted_weights.bear * 100, adjusted_weights.bull * 100,
        )
        result["weights"] = {
            "base": round(adjusted_weights.base, 3),
            "bear": round(adjusted_weights.bear, 3),
            "bull": round(adjusted_weights.bull, 3),
        }

        # ═══════════════════════════════════════════════════════════════
        # STEP 6: MONTE CARLO SIMULATION
        # ═══════════════════════════════════════════════════════════════
        forecast = self.mc.simulate(
            prices=live_prices,
            oil_price_override=regime.oil_price,
            vix_override=regime.vix,
            scenario_weights=adjusted_weights,
        )

        logger.info("Step 6: Monte Carlo complete — mandate=%s", forecast.mandate.level.value)

        spy = forecast.asset_forecasts.get(Asset.SPY)
        result["forecast"] = {
            "mandate": forecast.mandate.level.value,
            "risk_per_trade": forecast.mandate.risk_per_trade_pct,
            "max_positions": forecast.mandate.max_contracts_per_name,
            "spy_median_return": spy.expected_return_pct if spy else None,
            "spy_prob_down_10": spy.prob_down_10 if spy else None,
            "spy_var_95": spy.var_95_1d if spy else None,
        }

        # ═══════════════════════════════════════════════════════════════
        # STEP 7: OPTIONS CHAIN SCAN
        # ═══════════════════════════════════════════════════════════════
        picks = self.scanner.scan_all(forecast.mandate, prices=live_prices)
        logger.info("Step 7: Scanner found %d put recommendations", len(picks))

        for pick in picks:
            pick.contracts = min(
                pick.contracts,
                forecast.mandate.max_contracts_per_name,
            )
            if pick.contracts < 1:
                pick.contracts = 1

        result["picks"] = [
            {
                "ticker": p.ticker,
                "strike": p.contract.strike,
                "expiry": p.contract.expiry,
                "premium": p.contract.mid,
                "delta": p.greeks.delta if p.greeks else None,
                "score": p.composite_score,
                "contracts": p.contracts,
                "cost": p.total_cost,
            }
            for p in picks[:15]
        ]

        # ═══════════════════════════════════════════════════════════════
        # STEP 8: ADVANCED STRATEGY RECOMMENDATIONS
        # ═══════════════════════════════════════════════════════════════
        if self.advanced and picks:
            try:
                top_pick = picks[0]
                spot = live_prices.get(
                    next((a for a in Asset if a.value == top_pick.ticker), Asset.SPY),
                    top_pick.contract.strike,
                )
                from strategies.matrix_maximizer.core import ASSET_VOLATILITIES
                sigma = ASSET_VOLATILITIES.get(
                    next((a for a in Asset if a.value == top_pick.ticker), Asset.SPY),
                    0.30,
                )
                try:
                    exp_dt = datetime.strptime(top_pick.contract.expiry, "%Y-%m-%d")
                    dte = max(1, (exp_dt - datetime.utcnow()).days)
                except (ValueError, TypeError):
                    dte = 45
                recs = self.advanced.recommend_strategies(
                    top_pick.ticker, spot, sigma, dte, forecast.mandate, regime.vix,
                )
                result["advanced_strategies"] = {
                    "ticker": top_pick.ticker,
                    "count": len(recs),
                    "top": recs[0].print_card() if recs else "none",
                }
                logger.info("Step 8: Advanced strategies — %d for %s",
                            len(recs), top_pick.ticker)
            except Exception as exc:
                logger.warning("Step 8: Advanced strategy error: %s", exc)
        else:
            logger.info("Step 8: Advanced strategies skipped")

        # ═══════════════════════════════════════════════════════════════
        # STEP 9: AUTO-ROLL CHECK
        # ═══════════════════════════════════════════════════════════════
        rolls = self.scanner.check_rolls(positions, live_prices, forecast.mandate)
        logger.info("Step 9: Roll check — %d signals", len(rolls))

        result["rolls"] = [
            {
                "ticker": r.position.ticker,
                "trigger": r.reason,
                "action": r.action.value,
                "new_strike": r.new_strike,
                "new_expiry": r.new_expiry,
                "new_contracts": r.new_contracts,
            }
            for r in rolls
        ]

        # ═══════════════════════════════════════════════════════════════
        # STEP 10: RISK EVALUATION
        # ═══════════════════════════════════════════════════════════════
        greeks_map: Dict[str, GreeksResult] = {}
        for pos in positions:
            for asset in Asset:
                if asset.value == pos.ticker:
                    spot = live_prices.get(asset, pos.strike)
                    try:
                        exp_dt = datetime.strptime(pos.expiry, "%Y-%m-%d")
                        dte = max(1, (exp_dt - datetime.utcnow()).days)
                    except (ValueError, TypeError):
                        dte = 30
                    gr = self.bs.price_put(
                        spot=spot,
                        strike=pos.strike,
                        time_years=dte / 365.0,
                        sigma=0.30,
                    )
                    greeks_map[pos.ticker] = gr
                    break

        snapshot = self.risk.evaluate(
            positions=positions,
            forecast=forecast,
            greeks_map=greeks_map,
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            oil_price=regime.oil_price,
            vix=regime.vix,
            ncc_risk_multiplier=ncc_multiplier,
        )

        logger.info(
            "Step 10: Risk — %s | %d/%d checks passed",
            snapshot.circuit_breaker.value, snapshot.passed,
            snapshot.passed + snapshot.failed,
        )

        result["risk"] = {
            "circuit_breaker": snapshot.circuit_breaker.value,
            "passed": snapshot.passed,
            "failed": snapshot.failed,
            "exposure_pct": snapshot.exposure_pct,
            "portfolio_delta": snapshot.portfolio_delta,
            "var_95_1d": snapshot.var_95_1d,
            "cvar_95_1d": snapshot.cvar_95_1d,
            "hedge_alerts": [
                {"type": h.hedge_type, "ticker": h.ticker, "urgency": h.urgency}
                for h in snapshot.hedge_alerts
            ],
        }

        # Enhanced risk: tail risk + correlation stress
        if positions:
            price_map = {pos.ticker: live_prices.get(
                next((a for a in Asset if a.value == pos.ticker), Asset.SPY), pos.strike
            ) for pos in positions}

            stress = self.risk.correlation_stress_test(positions, price_map)
            result["risk"]["correlation_stress"] = stress

        tail = self.risk.tail_risk_analysis(forecast)
        result["risk"]["tail_risk"] = tail

        # Liquidity risk on picks
        if picks:
            liq = self.risk.liquidity_risk(picks[:10])
            result["risk"]["liquidity"] = liq

        # Margin estimate
        if positions:
            margin = self.risk.margin_estimate(positions, price_map)
            result["risk"]["margin"] = margin

        self.risk.save_state(snapshot)

        # ═══════════════════════════════════════════════════════════════
        # STEP 11: EXECUTION
        # ═══════════════════════════════════════════════════════════════
        if auto_execute and self.execution and picks:
            if snapshot.circuit_breaker in (CircuitBreaker.GREEN, CircuitBreaker.YELLOW):
                try:
                    pick_dicts_for_exec = result.get("picks", [])[:3]
                    exec_results = self.execution.execute_picks(
                        pick_dicts_for_exec,
                        mandate_risk_pct=forecast.mandate.risk_per_trade_pct,
                        max_positions=forecast.mandate.max_contracts_per_name,
                    )
                    result["execution"] = {
                        "attempted": len(exec_results),
                        "filled": sum(1 for r in exec_results if r.success),
                    }
                    logger.info("Step 11: Execution — %d attempted, %d filled",
                                len(exec_results),
                                sum(1 for r in exec_results if r.success))
                except Exception as exc:
                    logger.warning("Step 11: Execution error: %s", exc)
                    result["execution"] = {"error": str(exc)}
            else:
                result["execution"] = {"skipped": f"circuit_breaker={snapshot.circuit_breaker.value}"}
                logger.info("Step 11: Execution blocked — %s", snapshot.circuit_breaker.value)
        else:
            logger.info("Step 11: Execution skipped (auto_execute=%s)", auto_execute)

        # ═══════════════════════════════════════════════════════════════
        # STEP 12: DASHBOARD + ALERTS
        # ═══════════════════════════════════════════════════════════════
        # Record daily snapshot
        if self.dashboard:
            try:
                from strategies.matrix_maximizer.dashboard import DailySnapshot
                snap = DailySnapshot(
                    date=datetime.utcnow().strftime("%Y-%m-%d"),
                    equity=self.config.account_size + cumulative_pnl,
                    unrealized_pnl=daily_pnl,
                    realized_pnl=cumulative_pnl,
                    open_positions=len(positions),
                    trades_today=0,
                    vix=regime.vix,
                    oil=regime.oil_price,
                    regime=regime.regime,
                    circuit_breaker=snapshot.circuit_breaker.value,
                    mandate=forecast.mandate.level.value,
                    top_picks=[p.ticker for p in picks[:5]],
                )
                self.dashboard.record_snapshot(snap)
                logger.info("Step 12a: Dashboard snapshot recorded")
            except Exception as exc:
                logger.warning("Step 12a: Dashboard error: %s", exc)

        # Send alerts
        if self.alerts:
            try:
                # New picks alert
                for p in picks[:3]:
                    self.alerts.send_trade_alert(
                        p.ticker, p.contract.strike, p.contract.expiry,
                        p.contract.mid, p.contracts, action="PICK",
                    )

                # Circuit breaker changes
                if snapshot.circuit_breaker != CircuitBreaker.GREEN:
                    self.alerts.send_circuit_breaker_alert(
                        "GREEN", snapshot.circuit_breaker.value,
                        f"Risk checks: {snapshot.passed}/{snapshot.passed + snapshot.failed} passed",
                    )

                # Roll alerts
                for r in rolls:
                    self.alerts.send_roll_alert(
                        r.position.ticker, r.action.value, r.reason,
                    )
                logger.info("Step 12b: Alerts dispatched")
            except Exception as exc:
                logger.warning("Step 12b: Alert error: %s", exc)

        # ═══════════════════════════════════════════════════════════════
        # STEP 13: PUSH RESULTS TO PILLARS
        # ═══════════════════════════════════════════════════════════════
        pick_dicts = result.get("picks", [])
        self.bridge.push_results(snapshot, pick_dicts, regime)
        result["integration"] = self.bridge.get_integration_status()

        # ═══════════════════════════════════════════════════════════════
        # FINALIZE
        # ═══════════════════════════════════════════════════════════════
        elapsed = time.time() - t0
        result["status"] = "complete"
        result["elapsed_seconds"] = round(elapsed, 2)

        self._save_output(result)
        self._last_result = result

        logger.info("=" * 60)
        logger.info("  CYCLE %d COMPLETE in %.1fs — %s",
                     self._run_count, elapsed, snapshot.circuit_breaker.value.upper())
        logger.info("  Mandate: %s | Picks: %d | Rolls: %d",
                     forecast.mandate.level.value, len(picks), len(rolls))
        logger.info("=" * 60)

        return result

    def _save_output(self, result: Dict[str, Any]) -> None:
        """Save cycle output to JSON file."""
        try:
            out_dir = Path("data/matrix_maximizer")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"cycle_{self._run_count}_{ts}.json"
            (out_dir / filename).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            # Also write latest
            (out_dir / "latest.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save output: %s", exc)

    def print_summary(self, result: Optional[Dict[str, Any]] = None) -> str:
        """Print a human-readable cycle summary."""
        r = result or self._last_result
        if not r:
            return "No cycle results available."

        lines = [
            "",
            "=" * 72,
            "  MATRIX MAXIMIZER — CYCLE SUMMARY",
            f"  Run #{r.get('run_number', '?')} | {r.get('timestamp', '')}",
            f"  Status: {r.get('status', 'unknown').upper()} | "
            f"Elapsed: {r.get('elapsed_seconds', 0):.1f}s",
            "=" * 72,
        ]

        # Regime
        regime = r.get("regime", {})
        lines.append(f"\n  REGIME: {regime.get('regime', '?').upper()} "
                     f"(conf={regime.get('confidence', 0):.0f})")
        lines.append(f"  Oil: ${regime.get('oil_price', 0):.1f} | "
                     f"VIX: {regime.get('vix', 0):.1f} | "
                     f"War: {regime.get('war_active', False)} | "
                     f"Hormuz: {regime.get('hormuz_blocked', False)}")

        # Weights
        w = r.get("weights", {})
        lines.append(f"\n  SCENARIO WEIGHTS: "
                     f"Base={w.get('base', 0):.0%} "
                     f"Bear={w.get('bear', 0):.0%} "
                     f"Bull={w.get('bull', 0):.0%}")

        # Forecast
        fc = r.get("forecast", {})
        lines.append(f"\n  MANDATE: {fc.get('mandate', '?').upper()} "
                     f"(risk/trade={fc.get('risk_per_trade', 0):.1%}, "
                     f"max_pos={fc.get('max_positions', 0)})")
        lines.append(f"  SPY: median={fc.get('spy_median_return', 0):.1f}% | "
                     f"P(10%down)={fc.get('spy_prob_down_10', 0):.0%} | "
                     f"VaR95={fc.get('spy_var_95', 0):.1%}")

        # Top picks
        picks = r.get("picks", [])
        if picks:
            lines.append(f"\n  TOP PUTS ({len(picks)} found):")
            lines.append(f"  {'Ticker':<8}{'Strike':<10}{'Expiry':<14}"
                         f"{'Premium':<10}{'Delta':<10}{'Score':<8}{'Cost':<10}")
            lines.append("  " + "-" * 68)
            for p in picks[:10]:
                delta_str = f"{p.get('delta', 0):.3f}" if p.get('delta') else "n/a"
                lines.append(
                    f"  {p['ticker']:<8}${p['strike']:<9.0f}{p.get('expiry', '?'):<14}"
                    f"${p.get('premium', 0):<9.2f}{delta_str:<10}"
                    f"{p.get('score', 0):<8.1f}${p.get('cost', 0):<9.0f}"
                )

        # Rolls
        rolls = r.get("rolls", [])
        if rolls:
            lines.append(f"\n  ROLL SIGNALS ({len(rolls)}):")
            for rl in rolls:
                lines.append(f"    {rl['ticker']}: {rl['trigger']} → "
                             f"new K=${rl.get('new_strike', '?')} "
                             f"exp={rl.get('new_expiry', '?')}")

        # Risk
        risk = r.get("risk", {})
        lines.append(f"\n  RISK: {risk.get('circuit_breaker', '?').upper()} | "
                     f"Checks: {risk.get('passed', 0)}/{risk.get('passed', 0) + risk.get('failed', 0)} "
                     f"| Exposure: {risk.get('exposure_pct', 0):.1%}")
        lines.append(f"  Delta: {risk.get('portfolio_delta', 0):.3f} | "
                     f"VaR95: {risk.get('var_95_1d', 0):.1%} | "
                     f"CVaR95: {risk.get('cvar_95_1d', 0):.1%}")
        for ha in risk.get("hedge_alerts", []):
            lines.append(f"    [{ha['urgency'].upper()}] {ha['type']}: {ha['ticker']}")

        lines.append("\n" + "=" * 72)
        return "\n".join(lines)


def main() -> None:
    """CLI entry point for MATRIX MAXIMIZER."""
    parser = argparse.ArgumentParser(
        description="MATRIX MAXIMIZER — Geopolitical Bear Market Options Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m strategies.matrix_maximizer.runner
  python -m strategies.matrix_maximizer.runner --account 5000 --oil 110 --vix 30
  python -m strategies.matrix_maximizer.runner --war --hormuz --paths 50000
  python -m strategies.matrix_maximizer.runner --chatbot
  python -m strategies.matrix_maximizer.runner --backtest 90
  python -m strategies.matrix_maximizer.runner --schedule
        """,
    )
    parser.add_argument("--account", type=float, default=920.0,
                        help="Account size in USD (default: 920)")
    parser.add_argument("--oil", type=float, default=None,
                        help="Override oil price (WTI)")
    parser.add_argument("--vix", type=float, default=None,
                        help="Override VIX level")
    parser.add_argument("--paths", type=int, default=10000,
                        help="MC simulation paths (default: 10000)")
    parser.add_argument("--horizon", type=float, default=0.25,
                        help="Forecast horizon in years (default: 0.25 = 3 months)")
    parser.add_argument("--war", action="store_true",
                        help="Force war_active flag")
    parser.add_argument("--hormuz", action="store_true",
                        help="Force hormuz_blocked flag")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of summary")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress logger output")
    parser.add_argument("--execute", action="store_true",
                        help="Auto-execute top picks (paper or live)")
    parser.add_argument("--chatbot", action="store_true",
                        help="Launch interactive chatbot mode")
    parser.add_argument("--backtest", type=int, default=None, metavar="DAYS",
                        help="Run backtest for N days instead of live cycle")
    parser.add_argument("--schedule", action="store_true",
                        help="Start scheduler daemon for automated trading")
    parser.add_argument("--report", action="store_true",
                        help="Generate daily report without running cycle")

    args = parser.parse_args()

    # Logging setup
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Config
    config = MatrixConfig(
        account_size=args.account,
        n_simulations=args.paths,
        horizon_days=int(args.horizon * 365),
    )

    mm = MatrixMaximizer(config)

    # ── Mode: Chatbot ──
    if args.chatbot:
        chatbot = mm.get_chatbot()
        if chatbot:
            print("\n  MATRIX MAXIMIZER CHATBOT v2.0")
            print("  Type 'help' for commands, 'quit' to exit.\n")
            while True:
                try:
                    user_input = input("  matrix> ")
                except (EOFError, KeyboardInterrupt):
                    print("\n  Goodbye.")
                    break
                if user_input.strip().lower() in ("quit", "exit", "q"):
                    print("  Goodbye.")
                    break
                response = chatbot.handle(user_input)
                print(f"\n{response}\n")
        else:
            print("Chatbot module not available.")
        return

    # ── Mode: Backtest ──
    if args.backtest:
        if mm.backtester:
            print(f"\n  Running {args.backtest}-day backtest...")
            scenarios = mm.backtester.generate_historical_scenarios(
                days=args.backtest,
            )
            bt_result = mm.backtester.backtest(
                scenarios,
                initial_capital=config.account_size,
            )
            print(bt_result.print_card())
        else:
            print("Backtester module not available.")
        return

    # ── Mode: Report ──
    if args.report:
        if mm.dashboard:
            report = mm.dashboard.daily_report()
            print(report)
        else:
            print("Dashboard module not available.")
        return

    # ── Mode: Schedule ──
    if args.schedule:
        try:
            from strategies.matrix_maximizer.scheduler import MatrixScheduler, ScheduleSlot
            sched = MatrixScheduler()
            sched.add_task("full_cycle", ScheduleSlot.PRE_MARKET,
                           lambda: mm.run_full_cycle())
            sched.add_task("mid_update", ScheduleSlot.MID_SESSION,
                           lambda: mm.run_full_cycle())
            sched.add_task("close_report", ScheduleSlot.AFTER_HOURS,
                           lambda: print(mm.dashboard.daily_report(config.account_size))
                           if mm.dashboard else None)
            print("\n  MATRIX MAXIMIZER SCHEDULER STARTED")
            print("  Press Ctrl+C to stop.\n")
            sched.print_schedule()
            sched.start()
            # Block until interrupted
            import threading
            threading.Event().wait()
        except KeyboardInterrupt:
            print("\n  Scheduler stopped.")
        return

    # ── Mode: Standard cycle ──
    prices: Dict[str, float] = {}
    if args.oil is not None:
        prices["oil"] = args.oil
    if args.vix is not None:
        prices["vix"] = args.vix

    result = mm.run_full_cycle(
        prices=prices,
        war_active=True if args.war else None,
        hormuz_blocked=True if args.hormuz else None,
        auto_execute=args.execute,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(mm.print_summary(result))


if __name__ == "__main__":
    main()
