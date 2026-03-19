"""
MATRIX MAXIMIZER — Main Orchestrator
=======================================
The supreme engine. Chains every subsystem into a single execution cycle:

    NCC Governance Check
        ↓
    Regime Context Load
        ↓
    Scenario Weight Adjustment (Bridge)
        ↓
    Monte Carlo Simulation (10,000 paths)
        ↓
    System Mandate Generation
        ↓
    Options Chain Scan + Score
        ↓
    Auto-Roll Check (existing positions)
        ↓
    7-Layer Risk Evaluation
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

from strategies.matrix_maximizer.core import (
    Asset,
    MatrixConfig,
    PortfolioForecast,
    ScenarioWeights,
    SystemMandate,
    DEFAULT_PRICES,
)
from strategies.matrix_maximizer.monte_carlo import MonteCarloEngine
from strategies.matrix_maximizer.greeks import BlackScholesEngine, GreeksResult
from strategies.matrix_maximizer.scanner import OptionsScanner, Position, PutRecommendation
from strategies.matrix_maximizer.risk import (
    CircuitBreaker,
    RiskManager,
    RiskSnapshot,
)
from strategies.matrix_maximizer.bridge import PillarBridge, RegimeContext

logger = logging.getLogger(__name__)


class MatrixMaximizer:
    """Supreme orchestrator for the geopolitical bear market options engine.

    Usage:
        mm = MatrixMaximizer(config)
        result = mm.run_full_cycle(positions=[...])
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[MatrixConfig] = None) -> None:
        self.config = config or MatrixConfig()
        self.mc = MonteCarloEngine(self.config)
        self.bs = BlackScholesEngine()
        self.scanner = OptionsScanner(self.config, self.bs)
        self.risk = RiskManager(self.config)
        self.bridge = PillarBridge(self.config)

        self._run_count = 0
        self._last_result: Optional[Dict[str, Any]] = None
        logger.info(
            "MATRIX MAXIMIZER v%s initialized — account=$%s, tickers=%d",
            self.VERSION, f"{self.config.account_size:,.0f}", len(self.config.scan_tickers),
        )

    def run_full_cycle(
        self,
        positions: Optional[List[Position]] = None,
        prices: Optional[Dict[str, float]] = None,
        daily_pnl: float = 0.0,
        cumulative_pnl: float = 0.0,
        war_active: Optional[bool] = None,
        hormuz_blocked: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Execute one complete MATRIX MAXIMIZER cycle.

        Args:
            positions: Currently open positions (for risk + roll checks)
            prices: Override prices (ticker → price). Falls back to DEFAULT_PRICES.
            daily_pnl: Today's P&L in dollars
            cumulative_pnl: Cumulative P&L in dollars
            war_active: Override war_active flag
            hormuz_blocked: Override hormuz_blocked flag

        Returns:
            Complete cycle result dict with:
                - forecast: Monte Carlo portfolio forecast
                - mandate: System mandate (defensive/standard/aggressive/max_conviction)
                - picks: Ranked put recommendations
                - rolls: Roll signals for existing positions
                - risk: Risk snapshot with circuit breaker state
                - regime: Current regime context
                - integration: Pillar connection status
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
        # STEP 2: REGIME CONTEXT
        # ═══════════════════════════════════════════════════════════════
        regime = self.bridge.get_regime_context()

        # Override with live prices if provided
        if "oil" in prices:
            regime.oil_price = prices["oil"]
        if "vix" in prices:
            regime.vix = prices["vix"]
        if war_active is not None:
            regime.war_active = war_active
        if hormuz_blocked is not None:
            regime.hormuz_blocked = hormuz_blocked

        logger.info(
            "Step 2: Regime=%s (conf=%.0f) war=%s hormuz=%s oil=$%.1f vix=%.1f",
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
        # STEP 3: SCENARIO WEIGHT ADJUSTMENT
        # ═══════════════════════════════════════════════════════════════
        base_weights = ScenarioWeights()
        adjusted_weights = self.bridge.adjust_scenario_weights(base_weights, regime)
        logger.info(
            "Step 3: Scenario weights — BASE=%.0f%% BEAR=%.0f%% BULL=%.0f%%",
            adjusted_weights.base * 100, adjusted_weights.bear * 100, adjusted_weights.bull * 100,
        )
        result["weights"] = {
            "base": round(adjusted_weights.base, 3),
            "bear": round(adjusted_weights.bear, 3),
            "bull": round(adjusted_weights.bull, 3),
        }

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: MONTE CARLO SIMULATION
        # ═══════════════════════════════════════════════════════════════
        current_prices = dict(DEFAULT_PRICES)
        for ticker, price in prices.items():
            for asset in Asset:
                if asset.value == ticker:
                    current_prices[asset] = price
                    break

        forecast = self.mc.simulate(
            prices=current_prices,
            oil_price_override=regime.oil_price,
            vix_override=regime.vix,
            scenario_weights=adjusted_weights,
        )

        logger.info("Step 4: Monte Carlo complete — mandate=%s", forecast.mandate.level.value)

        # Serialize forecast
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
        # STEP 5: OPTIONS CHAIN SCAN
        # ═══════════════════════════════════════════════════════════════
        picks = self.scanner.scan_all(forecast.mandate, prices=current_prices)
        logger.info("Step 5: Scanner found %d put recommendations", len(picks))

        # Apply mandate sizing to picks
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
            for p in picks[:15]  # Top 15
        ]

        # ═══════════════════════════════════════════════════════════════
        # STEP 6: AUTO-ROLL CHECK
        # ═══════════════════════════════════════════════════════════════
        rolls = self.scanner.check_rolls(positions, current_prices, forecast.mandate)
        logger.info("Step 6: Roll check — %d signals", len(rolls))

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
        # STEP 7: RISK EVALUATION
        # ═══════════════════════════════════════════════════════════════
        # Build greeks map for risk manager
        greeks_map: Dict[str, GreeksResult] = {}
        for pos in positions:
            for asset in Asset:
                if asset.value == pos.ticker:
                    spot = current_prices.get(asset, pos.strike)
                    # Compute remaining time from expiry string
                    try:
                        exp_dt = datetime.strptime(pos.expiry, "%Y-%m-%d")
                        dte = max(1, (exp_dt - datetime.utcnow()).days)
                    except (ValueError, TypeError):
                        dte = 30  # Fallback
                    gr = self.bs.price_put(
                        spot=spot,
                        strike=pos.strike,
                        time_years=dte / 365.0,
                        sigma=0.30,  # Use implied vol from position if available
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
            "Step 7: Risk — %s | %d/%d checks passed",
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

        # Persist risk state
        self.risk.save_state(snapshot)

        # ═══════════════════════════════════════════════════════════════
        # STEP 8: PUSH RESULTS TO PILLARS
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

        # Save full output
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

    # Build price overrides
    prices: Dict[str, float] = {}
    if args.oil is not None:
        prices["oil"] = args.oil
    if args.vix is not None:
        prices["vix"] = args.vix

    # Run
    mm = MatrixMaximizer(config)

    result = mm.run_full_cycle(
        prices=prices,
        war_active=True if args.war else None,
        hormuz_blocked=True if args.hormuz else None,
    )

    # Output
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(mm.print_summary(result))


if __name__ == "__main__":
    main()
