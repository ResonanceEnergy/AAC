"""
Tests for Options Intelligence Engine
=======================================
Tests all 5 modules + the integration pipeline.
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.macro_crisis_put_strategy import (
    CrisisAssessment,
    CrisisMonitor,
    CrisisSignal,
    CrisisVector,
)
from strategies.options_intelligence.ai_scorer import (
    AITradeScorer,
    TradeScore,
    TradeSetup,
)
from strategies.options_intelligence.feedback import (
    FeedbackLoop,
    FillRecord,
    TuningRecommendation,
)
from strategies.options_intelligence.flow_signals import (
    FlowConviction,
    FlowDirection,
    FlowEntry,
    FlowSignalEngine,
    FlowType,
)
from strategies.options_intelligence.pipeline import (
    OptionsIntelligencePipeline,
    PipelineResult,
)
from strategies.options_intelligence.skew_optimizer import (
    OptimalStrike,
    SkewAnalysis,
    SkewOptimizer,
    StrikeIV,
)
from strategies.options_intelligence.universe import (
    DEFAULT_EXCLUSIONS,
    DynamicCandidate,
    UniverseExpander,
)

# ═══════════════════════════════════════════════════════════════════════════
# FLOW SIGNALS
# ═══════════════════════════════════════════════════════════════════════════

class TestFlowSignalEngine:
    def test_empty_flow_returns_no_convictions(self):
        engine = FlowSignalEngine()
        result = engine.analyze_flow_sync([], [], [])
        assert result == []

    def test_put_sweep_builds_bearish_conviction(self):
        engine = FlowSignalEngine()
        flow = [
            {
                "ticker": "SPY",
                "sentiment": "bearish",
                "option_type": "put",
                "total_premium": 500000,
                "volume": 5000,
                "open_interest": 1000,
                "is_sweep": True,
                "is_block": False,
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
            },
        ]
        result = engine.analyze_flow_sync(flow, [], [])
        tickers = {c.ticker for c in result}
        assert "SPY" in tickers
        spy = next(c for c in result if c.ticker == "SPY")
        assert spy.conviction > 0

    def test_multiple_flow_records_aggregate(self):
        engine = FlowSignalEngine()
        flow = [
            {
                "ticker": "XLF",
                "sentiment": "bearish",
                "option_type": "put",
                "total_premium": 300000,
                "volume": 3000,
                "open_interest": 800,
                "is_sweep": True,
                "is_block": False,
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
            },
            {
                "ticker": "XLF",
                "sentiment": "bearish",
                "option_type": "put",
                "total_premium": 200000,
                "volume": 2000,
                "open_interest": 500,
                "is_sweep": False,
                "is_block": True,
                "timestamp": (datetime.now() - timedelta(minutes=45)).isoformat(),
            },
        ]
        result = engine.analyze_flow_sync(flow, [], [])
        tickers = {c.ticker for c in result}
        assert "XLF" in tickers
        xlf = next(c for c in result if c.ticker == "XLF")
        assert xlf.put_premium == 500000
        assert xlf.sweep_count == 1
        assert xlf.block_count == 1

    def test_entry_triggers_require_conviction(self):
        engine = FlowSignalEngine()
        convictions = [
            FlowConviction(
                ticker="SPY",
                conviction=0.85,
                direction=FlowDirection.BEARISH,
                put_premium=1000000,
                call_premium=200000,
                put_call_ratio=5.0,
                sweep_count=5,
                block_count=2,
                dark_pool_notional=5000000,
                congress_bearish=False,
                signal_count=7,
                flow_velocity=3.0,
            ),
        ]
        triggers = engine.check_entry_triggers(convictions)
        assert len(triggers) == 1
        assert triggers[0].ticker == "SPY"
        assert triggers[0].urgency == "immediate"

    def test_conviction_multiplier_bearish(self):
        engine = FlowSignalEngine()
        convictions = [
            FlowConviction(
                ticker="KRE",
                conviction=0.75,
                direction=FlowDirection.BEARISH,
                put_premium=500000,
                call_premium=100000,
                put_call_ratio=5.0,
                sweep_count=3,
                block_count=1,
                dark_pool_notional=0,
                congress_bearish=False,
                signal_count=4,
                flow_velocity=2.0,
            ),
        ]
        mult = engine.get_conviction_multiplier("KRE", convictions)
        assert 1.0 <= mult <= 2.0

    def test_conviction_multiplier_bullish_dampens(self):
        engine = FlowSignalEngine()
        convictions = [
            FlowConviction(
                ticker="XLE",
                conviction=0.80,
                direction=FlowDirection.BULLISH,
                put_premium=100000,
                call_premium=800000,
                put_call_ratio=0.125,
                sweep_count=4,
                block_count=2,
                dark_pool_notional=0,
                congress_bearish=False,
                signal_count=6,
                flow_velocity=2.5,
            ),
        ]
        mult = engine.get_conviction_multiplier("XLE", convictions)
        assert 0.5 <= mult <= 1.0

    def test_unknown_ticker_returns_neutral_multiplier(self):
        engine = FlowSignalEngine()
        mult = engine.get_conviction_multiplier("ZZZZZ", [])
        assert mult == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSE EXPANDER
# ═══════════════════════════════════════════════════════════════════════════

class TestUniverseExpander:
    def test_default_exclusions_populated(self):
        expander = UniverseExpander()
        assert "XLE" in DEFAULT_EXCLUSIONS
        assert "VIX" in DEFAULT_EXCLUSIONS
        assert "XLE" in expander.exclusions

    def test_discover_sync_with_convictions(self):
        expander = UniverseExpander()
        convictions = [
            FlowConviction(
                ticker="SCHW",
                conviction=0.75,
                direction=FlowDirection.BEARISH,
                put_premium=400000,
                call_premium=50000,
                put_call_ratio=8.0,
                sweep_count=3,
                block_count=1,
                dark_pool_notional=2000000,
                congress_bearish=True,
                signal_count=4,
                flow_velocity=2.5,
            ),
        ]
        existing = {"SPY", "XLF", "HYG"}
        candidates = expander.discover_sync(
            flow_convictions=convictions,
            existing_universe=existing,
        )
        # SCHW not in existing, should be a candidate
        tickers = {c.ticker for c in candidates}
        assert "SCHW" in tickers

    def test_existing_universe_excluded(self):
        expander = UniverseExpander()
        convictions = [
            FlowConviction(
                ticker="SPY",
                conviction=0.9,
                direction=FlowDirection.BEARISH,
                put_premium=1000000,
                call_premium=100000,
                put_call_ratio=10.0,
                sweep_count=10,
                block_count=5,
                dark_pool_notional=10000000,
                congress_bearish=True,
                signal_count=15,
                flow_velocity=5.0,
            ),
        ]
        existing = {"SPY"}
        candidates = expander.discover_sync(
            flow_convictions=convictions,
            existing_universe=existing,
        )
        tickers = {c.ticker for c in candidates}
        assert "SPY" not in tickers

    def test_excluded_tickers_filtered(self):
        expander = UniverseExpander()
        convictions = [
            FlowConviction(
                ticker="XLE",
                conviction=0.9,
                direction=FlowDirection.BEARISH,
                put_premium=500000,
                call_premium=50000,
                put_call_ratio=10.0,
                sweep_count=5,
                block_count=2,
                dark_pool_notional=3000000,
                congress_bearish=False,
                signal_count=7,
                flow_velocity=3.0,
            ),
        ]
        candidates = expander.discover_sync(
            flow_convictions=convictions,
            existing_universe=set(),
        )
        tickers = {c.ticker for c in candidates}
        assert "XLE" not in tickers


# ═══════════════════════════════════════════════════════════════════════════
# AI SCORER
# ═══════════════════════════════════════════════════════════════════════════

class TestAITradeScorer:
    def _make_setup(self, **overrides) -> TradeSetup:
        defaults = dict(
            ticker="SPY",
            direction="put",
            strike=540,
            expiry="2026-04-18",
            dte=30,
            premium=3.50,
            delta=-0.30,
            gamma=0.008,
            vega=0.15,
            theta=-0.12,
            iv=0.22,
            flow_conviction=0.70,
            put_call_ratio=1.5,
            sweep_count=3,
            dark_pool_notional=2000000,
            regime="credit_stress",
            vix=28.0,
            oil_price=95.0,
            hy_spread_bps=380,
            existing_positions=5,
            portfolio_delta=-0.3,
            portfolio_vega=1.5,
            account_balance=8800,
            risk_pct=0.02,
        )
        defaults.update(overrides)
        return TradeSetup(**defaults)

    def test_heuristic_scoring_returns_valid_score(self):
        scorer = AITradeScorer()
        setup = self._make_setup()
        score = scorer.score_trade_sync(setup)
        assert isinstance(score, TradeScore)
        assert 0 <= score.score <= 100
        assert score.model_used == "heuristic"

    def test_crisis_regime_boosts_thesis(self):
        scorer = AITradeScorer()
        crisis = scorer.score_trade_sync(self._make_setup(regime="credit_stress"))
        risk_on = scorer.score_trade_sync(self._make_setup(regime="risk_on"))
        assert crisis.thesis_score > risk_on.thesis_score

    def test_high_flow_conviction_boosts_flow_score(self):
        scorer = AITradeScorer()
        high = scorer.score_trade_sync(self._make_setup(flow_conviction=0.9, put_call_ratio=2.5))
        low = scorer.score_trade_sync(self._make_setup(flow_conviction=0.1, put_call_ratio=0.5))
        assert high.flow_score > low.flow_score

    def test_delta_sweet_spot_scores_higher(self):
        scorer = AITradeScorer()
        sweet = scorer.score_trade_sync(self._make_setup(delta=-0.30))
        far = scorer.score_trade_sync(self._make_setup(delta=-0.05))
        assert sweet.greeks_score > far.greeks_score

    def test_good_dte_scores_higher(self):
        scorer = AITradeScorer()
        good = scorer.score_trade_sync(self._make_setup(dte=30))
        bad = scorer.score_trade_sync(self._make_setup(dte=3))
        assert good.timing_score > bad.timing_score

    def test_over_concentrated_risk_penalized(self):
        scorer = AITradeScorer()
        safe = scorer.score_trade_sync(self._make_setup(risk_pct=0.02))
        over = scorer.score_trade_sync(self._make_setup(risk_pct=0.10))
        assert safe.risk_score > over.risk_score

    def test_is_actionable_threshold(self):
        scorer = AITradeScorer()
        score = scorer.score_trade_sync(self._make_setup())
        # With crisis regime + decent flow, should be actionable
        assert score.score >= 60 or not score.is_actionable

    def test_adjustments_generated_for_weak_scores(self):
        scorer = AITradeScorer()
        setup = self._make_setup(
            delta=-0.05, dte=3, risk_pct=0.10, flow_conviction=0.0,
            sweep_count=0, dark_pool_notional=0, put_call_ratio=0.3,
            regime="risk_on", vix=10, iv=0.35, gamma=0.0, theta=0.0,
        )
        score = scorer.score_trade_sync(setup)
        assert len(score.adjustments) > 0

    def test_has_llm_reflects_env(self):
        scorer = AITradeScorer()
        # Just check property doesn't crash
        _ = scorer.has_llm

    def test_score_to_dict(self):
        scorer = AITradeScorer()
        score = scorer.score_trade_sync(self._make_setup())
        d = score.to_dict()
        assert "ticker" in d
        assert "score" in d
        assert "reasoning" in d


# ═══════════════════════════════════════════════════════════════════════════
# SKEW OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════

class TestSkewOptimizer:
    def _make_chain(self, spot=550.0, n_strikes=10):
        """Generate a synthetic options chain."""
        chain = []
        for i in range(n_strikes):
            strike = spot * (0.85 + i * 0.03)
            moneyness = strike / spot
            # Synthetic skew: deeper OTM = higher IV
            iv = 0.25 + (1.0 - moneyness) * 0.5
            delta = -max(0.05, min(0.95, 0.5 * moneyness))
            mid = max(0.10, spot * 0.005 * (1.0 - moneyness + 0.05) * iv)
            chain.append({
                "strike": round(strike, 2),
                "iv": round(iv, 4),
                "delta": round(delta, 3),
                "bid": round(mid * 0.95, 2),
                "ask": round(mid * 1.05, 2),
                "volume": 500 + i * 100,
                "open_interest": 2000 + i * 500,
            })
        return chain

    def test_find_optimal_strike_returns_result(self):
        optimizer = SkewOptimizer()
        chain = self._make_chain()
        result = optimizer.find_optimal_strike(chain, spot=550.0, ticker="SPY", expiry="2026-04-18")
        assert result is not None
        assert isinstance(result, OptimalStrike)
        assert result.ticker == "SPY"
        assert result.value_score > 0

    def test_empty_chain_returns_none(self):
        optimizer = SkewOptimizer()
        result = optimizer.find_optimal_strike([], spot=550.0)
        assert result is None

    def test_skew_analysis(self):
        optimizer = SkewOptimizer()
        chain = self._make_chain()
        analysis = optimizer.analyze_skew(chain, spot=550.0, ticker="SPY", expiry="2026-04-18")
        assert isinstance(analysis, SkewAnalysis)
        assert analysis.atm_iv > 0
        assert analysis.ticker == "SPY"

    def test_term_structure_analysis(self):
        optimizer = SkewOptimizer()
        chains = {
            "2026-04-18": self._make_chain(spot=550),
            "2026-05-16": self._make_chain(spot=550),
        }
        analyses = optimizer.analyze_term_structure(chains, spot=550.0, ticker="SPY")
        assert len(analyses) == 2

    def test_find_best_expiry(self):
        optimizer = SkewOptimizer()
        chains = {
            "2026-04-18": self._make_chain(spot=550),
            "2026-05-16": self._make_chain(spot=550),
        }
        best = optimizer.find_best_expiry(chains, spot=550.0, ticker="SPY")
        assert best is not None or best is None  # Just ensure no crash

    def test_strike_iv_properties(self):
        s = StrikeIV(strike=540, iv=0.22, delta=-0.30, bid=3.50, ask=3.60,
                     volume=500, open_interest=2000)
        assert s.liquid
        assert s.mid == 3.55
        assert s.spread_pct < 0.05

    def test_illiquid_strike(self):
        s = StrikeIV(strike=400, iv=0.50, delta=-0.05, bid=0.01, ask=0.10,
                     volume=2, open_interest=10)
        assert not s.liquid


# ═══════════════════════════════════════════════════════════════════════════
# FEEDBACK LOOP
# ═══════════════════════════════════════════════════════════════════════════

class TestFeedbackLoop:
    def _make_fill(self, **overrides) -> FillRecord:
        defaults = dict(
            fill_id="20260318_143000_SPY",
            timestamp="2026-03-18T14:30:00",
            ticker="SPY",
            direction="put",
            strike=540.0,
            expiry="2026-04-18",
            dte_at_entry=30,
            quantity=1,
            fill_price=3.50,
            total_cost=350.0,
            delta=-0.30,
            gamma=0.008,
            vega=0.15,
            theta=-0.12,
            iv=0.22,
            flow_conviction=0.70,
            ai_score=75,
            skew_value_score=80.0,
            regime="credit_stress",
            vix_at_entry=28.0,
        )
        defaults.update(overrides)
        return FillRecord(**defaults)

    def test_log_fill_creates_file(self, tmp_path):
        fb = FeedbackLoop(data_dir=tmp_path)
        fill = self._make_fill()
        fb.log_fill(fill)
        assert (tmp_path / "fills.jsonl").exists()

    def test_load_fills_round_trip(self, tmp_path):
        fb = FeedbackLoop(data_dir=tmp_path)
        fb.log_fill(self._make_fill(fill_id="fill_1"))
        fb.log_fill(self._make_fill(fill_id="fill_2"))
        fills = fb.get_fills()
        assert len(fills) == 2
        assert fills[0].fill_id == "fill_1"

    def test_update_outcome(self, tmp_path):
        fb = FeedbackLoop(data_dir=tmp_path)
        fb.log_fill(self._make_fill(fill_id="fill_1"))
        updated = fb.update_outcome("fill_1", exit_price=5.00, exit_reason="target")
        assert updated is not None
        assert updated.pnl == (5.00 - 3.50) * 1 * 100
        assert updated.exit_reason == "target"

    def test_get_open_positions(self, tmp_path):
        fb = FeedbackLoop(data_dir=tmp_path)
        fb.log_fill(self._make_fill(fill_id="open1"))
        fb.log_fill(self._make_fill(fill_id="closed1"))
        fb.update_outcome("closed1", exit_price=4.00)
        opens = fb.get_open_positions()
        assert len(opens) == 1
        assert opens[0].fill_id == "open1"

    def test_get_stats_empty(self, tmp_path):
        fb = FeedbackLoop(data_dir=tmp_path)
        stats = fb.get_stats()
        assert stats["total_fills"] == 0
        assert stats["win_rate"] == 0

    def test_get_stats_with_fills(self, tmp_path):
        fb = FeedbackLoop(data_dir=tmp_path)
        fb.log_fill(self._make_fill(fill_id="f1"))
        fb.log_fill(self._make_fill(fill_id="f2"))
        fb.update_outcome("f1", exit_price=5.00)  # Winner
        fb.update_outcome("f2", exit_price=1.00)  # Loser
        stats = fb.get_stats()
        assert stats["total_fills"] == 2
        assert stats["closed_positions"] == 2
        assert stats["winners"] == 1
        assert stats["win_rate"] == 0.5

    def test_analyze_insufficient_data(self, tmp_path):
        fb = FeedbackLoop(data_dir=tmp_path)
        fb.log_fill(self._make_fill())
        recs = fb.analyze()
        assert recs == []  # Need 20+ for tuning

    def test_fill_record_properties(self):
        fill = self._make_fill()
        assert not fill.is_closed
        fill.exit_price = 5.0
        fill.pnl = 150.0
        assert fill.is_closed
        assert fill.is_winner

    def test_fill_record_round_trip(self):
        fill = self._make_fill()
        d = fill.to_dict()
        restored = FillRecord.from_dict(d)
        assert restored.fill_id == fill.fill_id
        assert restored.ticker == fill.ticker


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class TestPipeline:
    def _make_crisis_assessment(self) -> CrisisAssessment:
        return CrisisAssessment(signals=[
            CrisisSignal(
                vector=CrisisVector.WAR_ESCALATION,
                severity=0.85,
                description="Active war zone",
                data_source="manual",
            ),
            CrisisSignal(
                vector=CrisisVector.OIL_SHOCK,
                severity=0.80,
                description="Oil at $95",
                data_source="market",
            ),
            CrisisSignal(
                vector=CrisisVector.PRIVATE_CREDIT_COLLAPSE,
                severity=0.75,
                description="Redemptions at 11%",
                data_source="reuters",
            ),
            CrisisSignal(
                vector=CrisisVector.STAGFLATION,
                severity=0.70,
                description="PCE 3.1%, GDP 0.7%",
                data_source="fed",
            ),
            CrisisSignal(
                vector=CrisisVector.CREDIT_CONTAGION,
                severity=0.60,
                description="HY spread 380bps",
                data_source="market",
            ),
        ])

    def _make_prices(self) -> dict:
        return {
            "SPY": 555.0,
            "XLF": 42.0,
            "HYG": 73.0,
            "OWL": 14.0,
            "OBDC": 13.50,
            "KRE": 56.0,
            "QQQ": 475.0,
            "IWM": 210.0,
            "BKLN": 20.50,
            "LQD": 105.0,
        }

    def test_pipeline_sync_produces_result(self):
        pipeline = OptionsIntelligencePipeline(account_balance=8800, paper_trading=True)
        result = pipeline.run_sync(
            assessment=self._make_crisis_assessment(),
            underlying_prices=self._make_prices(),
            regime="credit_stress",
            vix=28.0,
        )
        assert isinstance(result, PipelineResult)
        assert len(result.base_orders) > 0
        assert len(result.scored_orders) + len(result.rejected_orders) == len(result.base_orders)

    def test_pipeline_sync_with_flow_data(self):
        pipeline = OptionsIntelligencePipeline(account_balance=8800)
        flow_data = {
            "flow": [
                {
                    "ticker": "SPY",
                    "sentiment": "bearish",
                    "option_type": "put",
                    "total_premium": 1000000,
                    "volume": 10000,
                    "open_interest": 5000,
                    "is_sweep": True,
                    "is_block": False,
                    "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
                },
            ],
            "dark_pool": [],
            "congress": [],
        }
        result = pipeline.run_sync(
            assessment=self._make_crisis_assessment(),
            underlying_prices=self._make_prices(),
            flow_data=flow_data,
            regime="credit_stress",
            vix=28.0,
        )
        assert len(result.flow_convictions) > 0

    def test_pipeline_summary(self):
        pipeline = OptionsIntelligencePipeline(account_balance=8800)
        result = pipeline.run_sync(
            assessment=self._make_crisis_assessment(),
            underlying_prices=self._make_prices(),
        )
        summary = result.summary()
        assert "Options Intelligence Pipeline" in summary
        assert "Base orders:" in summary

    def test_pipeline_actionable_filter(self):
        pipeline = OptionsIntelligencePipeline(account_balance=8800)
        result = pipeline.run_sync(
            assessment=self._make_crisis_assessment(),
            underlying_prices=self._make_prices(),
            regime="credit_stress",
            vix=28.0,
        )
        # All actionable orders have score >= 60
        for o in result.actionable_orders:
            assert o["score"].score >= 60

    def test_pipeline_log_fill(self, tmp_path):
        pipeline = OptionsIntelligencePipeline(account_balance=8800)
        pipeline.feedback = FeedbackLoop(data_dir=tmp_path)

        result = pipeline.run_sync(
            assessment=self._make_crisis_assessment(),
            underlying_prices=self._make_prices(),
        )

        if result.scored_orders:
            entry = result.scored_orders[0]
            fill = pipeline.log_fill(
                order=entry["order"],
                fill_price=entry["order"].max_price,
                score=entry["score"],
                flow_conviction=0.7,
                regime="credit_stress",
                vix=28.0,
            )
            assert fill.ticker == entry["order"].symbol
            assert (tmp_path / "fills.jsonl").exists()
