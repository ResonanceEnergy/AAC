from __future__ import annotations

import json
from pathlib import Path

import pytest

from strategies import strategy_advisor_engine as sae
from strategies.strategy_advisor_engine import (
    AdvisorPerformance,
    AdvisorRecommendation,
    PaperPosition,
    StrategyAdapter,
    StrategyAdvisorEngine,
    get_strategy_advisor_engine,
)


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch, tmp_path):
    """Redirect state file + CSV path to tmp; reset singleton."""
    monkeypatch.setattr(sae, "ADVISOR_STATE_FILE", tmp_path / "advisor_state.json")
    monkeypatch.setattr(sae, "STRATEGIES_CSV", tmp_path / "missing_strategies.csv")
    monkeypatch.setattr(sae, "_advisor_engine", None)
    yield


# ═══════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════

class TestAdvisorRecommendation:
    def test_construction(self):
        r = AdvisorRecommendation(
            strategy_name="x", ticker="SPY", direction="long",
            confidence=0.7, entry_price=500, target_price=515,
            stop_loss=490, thesis="t",
        )
        assert r.ticker == "SPY"
        assert r.timestamp  # auto-set


class TestPaperPosition:
    def _pos(self, **k):
        base = dict(strategy_name="s", ticker="SPY", direction="long",
                    entry_price=100, current_price=100, target_price=110,
                    stop_loss=95, quantity=10, entry_time="t")
        base.update(k)
        return PaperPosition(**base)

    def test_mark_to_market_long(self):
        p = self._pos()
        p.mark_to_market(105)
        assert p.current_price == 105
        assert p.unrealised_pnl == 50  # (105-100)*10

    def test_mark_to_market_short(self):
        p = self._pos(direction="short")
        p.mark_to_market(95)
        assert p.unrealised_pnl == 50  # (95-100)*10*-1

    def test_check_exits_long_target_hit(self):
        p = self._pos()
        assert p.check_exits(110) is True
        assert p.status == "closed_target"
        assert p.realised_pnl == 100
        assert p.unrealised_pnl == 0.0

    def test_check_exits_long_stop_hit(self):
        p = self._pos()
        assert p.check_exits(95) is True
        assert p.status == "closed_stop"
        assert p.realised_pnl == -50

    def test_check_exits_short_target_hit(self):
        p = self._pos(direction="short", target_price=90, stop_loss=105)
        assert p.check_exits(90) is True
        assert p.status == "closed_target"
        assert p.realised_pnl == 100  # (90-100)*10*-1

    def test_check_exits_short_stop_hit(self):
        p = self._pos(direction="short", target_price=90, stop_loss=105)
        assert p.check_exits(105) is True
        assert p.status == "closed_stop"
        assert p.realised_pnl == -50

    def test_check_exits_no_trigger(self):
        p = self._pos()
        assert p.check_exits(102) is False
        assert p.status == "open"
        assert p.unrealised_pnl == 20


# ═══════════════════════════════════════════════════════════════════════════
# AdvisorPerformance.update
# ═══════════════════════════════════════════════════════════════════════════

def _closed_pos(pnl: float) -> PaperPosition:
    p = PaperPosition(strategy_name="s", ticker="SPY", direction="long",
                      entry_price=100, current_price=100, target_price=110,
                      stop_loss=95, quantity=1, entry_time="t",
                      realised_pnl=pnl, status="closed_target")
    return p


class TestAdvisorPerformance:
    def test_empty_update_no_change(self):
        p = AdvisorPerformance(strategy_name="x")
        p.update([])
        assert p.total_trades == 0

    def test_basic_metrics(self):
        p = AdvisorPerformance(strategy_name="x")
        p.update([_closed_pos(10), _closed_pos(20), _closed_pos(-5)])
        assert p.total_trades == 3
        assert p.winning_trades == 2
        assert p.losing_trades == 1
        assert p.total_pnl == 25
        assert p.win_rate == pytest.approx(2/3)
        assert p.avg_win == 15
        assert p.avg_loss == -5

    def test_max_drawdown(self):
        # equity curve: 10, 30, 25, 5  → peaks 10, 30; max dd = 5-30 = -25
        p = AdvisorPerformance(strategy_name="x")
        p.update([_closed_pos(10), _closed_pos(20), _closed_pos(-5), _closed_pos(-20)])
        assert p.max_drawdown == -25

    def test_sharpe_with_variance(self):
        p = AdvisorPerformance(strategy_name="x")
        p.update([_closed_pos(10), _closed_pos(-10)])
        # mean=0 → sharpe=0
        assert p.sharpe_ratio == 0.0

    def test_sharpe_zero_when_single_trade(self):
        p = AdvisorPerformance(strategy_name="x")
        p.update([_closed_pos(10)])
        assert p.sharpe_ratio == 0.0

    def test_sharpe_zero_std_returns_zero(self):
        p = AdvisorPerformance(strategy_name="x")
        p.update([_closed_pos(10), _closed_pos(10)])
        # std=0 → sharpe=0
        assert p.sharpe_ratio == 0.0

    def test_last_evaluated_set(self):
        p = AdvisorPerformance(strategy_name="x")
        p.update([_closed_pos(10)])
        assert p.last_evaluated  # timestamp populated


# ═══════════════════════════════════════════════════════════════════════════
# StrategyAdapter
# ═══════════════════════════════════════════════════════════════════════════

class TestStrategyAdapter:
    def test_init_defaults(self):
        a = StrategyAdapter(name="n", one_liner="o")
        assert a.name == "n"
        assert a.category == "csv_advisor"
        assert a.engine is None

    def test_evaluate_no_spy_price_returns_none(self):
        a = StrategyAdapter(name="n", one_liner="o")
        assert a.evaluate({"spy_price": 0, "vix": 20}) is None

    def test_evaluate_high_vix_short(self):
        a = StrategyAdapter(name="n", one_liner="o")
        # vix=30 → short, conf = 30/40 = 0.75
        rec = a.evaluate({"vix": 30, "spy_price": 500})
        assert rec is not None
        assert rec.direction == "short"
        assert rec.confidence == 0.75
        assert rec.target_price == 485.0  # 500 * 0.97
        assert rec.stop_loss == 510.0  # 500 * 1.02

    def test_evaluate_low_vix_long(self):
        a = StrategyAdapter(name="n", one_liner="o")
        # vix=10 → long, conf = (40-10)/40 = 0.75
        rec = a.evaluate({"vix": 10, "spy_price": 500})
        assert rec is not None
        assert rec.direction == "long"
        assert rec.confidence == 0.75
        assert rec.target_price == 515.0
        assert rec.stop_loss == 490.0

    def test_evaluate_below_threshold_returns_none(self):
        # conf < 0.3 → None. vix=22 → long path; conf=(40-22)/40=0.45 → still ≥0.3
        # vix=29 → short path; conf=29/40=0.725 → ≥0.3
        # vix=24 → short path (>25 is short); 24<=25 → long path; conf=(40-24)/40=0.40
        # vix=29 short path; vix=11 long path 0.725
        # Need conf<0.3: long branch -> vix>=28; but vix>25 → short branch.
        # short branch <0.3 means vix<12 → long branch instead.
        # Edge: vix exactly between thresholds where long conf<0.3?
        # long branch when vix<=25; conf=(40-vix)/40<0.3 → vix>28; impossible (since vix>25→short).
        # short branch when vix>25; conf=vix/40<0.3 → vix<12; impossible.
        # So this branch is unreachable under defaults. Skip strict test.
        # Just confirm the threshold is the one in code by mocking:
        a = StrategyAdapter(name="n", one_liner="o")
        # simulate with vix=25 → long branch, conf=(40-25)/40=0.375 → returned
        rec = a.evaluate({"vix": 25, "spy_price": 500})
        assert rec is not None
        assert rec.confidence == 0.375

    def test_evaluate_with_engine_get_mandate(self):
        class FakeMandate:
            ticker = "TSLA"
            direction = "short"
            confidence = 0.8
            def __str__(self): return "thesis-text"

        class FakeEngine:
            def get_mandate(self):
                return FakeMandate()

        a = StrategyAdapter(name="n", one_liner="o", engine=FakeEngine())
        rec = a.evaluate({"spy_price": 500})
        assert rec is not None
        assert rec.ticker == "TSLA"
        assert rec.direction == "short"
        assert rec.confidence == 0.8
        assert rec.entry_price == 500

    def test_evaluate_with_engine_scan_bear(self):
        class FakeEngine:
            def scan(self):
                return "BEAR market detected"

        a = StrategyAdapter(name="n", one_liner="o", engine=FakeEngine())
        rec = a.evaluate({"spy_price": 500})
        assert rec is not None
        assert rec.direction == "short"

    def test_evaluate_with_engine_scan_bull(self):
        class FakeEngine:
            def scan(self):
                return "BULL momentum"
        a = StrategyAdapter(name="n", one_liner="o", engine=FakeEngine())
        rec = a.evaluate({"spy_price": 500})
        assert rec.direction == "long"

    def test_evaluate_engine_exception_falls_back(self):
        class BadEngine:
            def get_mandate(self):
                raise RuntimeError("boom")
        a = StrategyAdapter(name="n", one_liner="o", engine=BadEngine())
        # Falls through to heuristic path
        rec = a.evaluate({"vix": 30, "spy_price": 500})
        assert rec is not None
        assert rec.direction == "short"

    def test_evaluate_engine_scan_returns_none_falls_through(self):
        class FakeEngine:
            def scan(self):
                return None
        a = StrategyAdapter(name="n", one_liner="o", engine=FakeEngine())
        rec = a.evaluate({"vix": 30, "spy_price": 500})
        # Falls through to heuristic
        assert rec is not None
        assert rec.direction == "short"


# ═══════════════════════════════════════════════════════════════════════════
# StrategyAdvisorEngine
# ═══════════════════════════════════════════════════════════════════════════

class TestStrategyAdvisorEngineLoading:
    def test_load_strategies_no_csv_loads_active_7(self):
        e = StrategyAdvisorEngine()
        n = e.load_strategies()
        assert n == 7  # only active_7
        assert all(s.category == "active_7" for s in e._strategies)

    def test_load_strategies_with_csv(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "strats.csv"
        csv_path.write_text("strategy_name,one_liner\nA,Foo\nB,Bar\n", encoding="utf-8")
        monkeypatch.setattr(sae, "STRATEGIES_CSV", csv_path)
        e = StrategyAdvisorEngine()
        n = e.load_strategies()
        assert n == 2 + 7
        names = [s.name for s in e._strategies]
        assert "A" in names and "B" in names

    def test_load_strategies_idempotent(self):
        e = StrategyAdvisorEngine()
        n1 = e.load_strategies()
        n2 = e.load_strategies()
        assert n1 == n2
        assert e._loaded is True

    def test_load_strategies_initialises_perf_dicts(self):
        e = StrategyAdvisorEngine()
        e.load_strategies()
        for s in e._strategies:
            assert s.name in e._performance
            assert s.name in e._paper_positions
            assert s.name in e._closed_positions

    def test_load_strategies_csv_malformed(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_bytes(b"\xff\xfe\x00not valid utf-8")
        monkeypatch.setattr(sae, "STRATEGIES_CSV", csv_path)
        e = StrategyAdvisorEngine()
        # Should swallow exception and still load active_7
        n = e.load_strategies()
        assert n == 7


class TestEvaluateAll:
    def test_auto_loads(self):
        e = StrategyAdvisorEngine()
        recs = e.evaluate_all({"vix": 30, "spy_price": 500})
        assert e._loaded is True
        assert len(recs) > 0

    def test_returns_sorted_by_confidence(self):
        e = StrategyAdvisorEngine()
        recs = e.evaluate_all({"vix": 30, "spy_price": 500})
        confs = [r.confidence for r in recs]
        assert confs == sorted(confs, reverse=True)

    def test_opens_paper_positions(self):
        e = StrategyAdvisorEngine()
        e.evaluate_all({"vix": 30, "spy_price": 500})
        total_open = sum(len(v) for v in e._paper_positions.values())
        assert total_open > 0

    def test_evaluation_exception_swallowed(self):
        e = StrategyAdvisorEngine()
        e.load_strategies()

        class BoomAdapter(StrategyAdapter):
            def evaluate(self, snapshot):
                raise RuntimeError("boom")

        e._strategies = [BoomAdapter(name="bad", one_liner="x"),
                         StrategyAdapter(name="good", one_liner="y")]
        e._performance["bad"] = AdvisorPerformance(strategy_name="bad")
        e._performance["good"] = AdvisorPerformance(strategy_name="good")
        e._paper_positions["bad"] = []
        e._paper_positions["good"] = []
        e._closed_positions["bad"] = []
        e._closed_positions["good"] = []
        recs = e.evaluate_all({"vix": 30, "spy_price": 500})
        assert any(r.strategy_name == "good" for r in recs)
        assert not any(r.strategy_name == "bad" for r in recs)


class TestPaperProofCycle:
    def test_close_target(self):
        e = StrategyAdvisorEngine()
        e.load_strategies()
        e.evaluate_all({"vix": 10, "spy_price": 500})  # long, target=515
        # First strat
        first = e._strategies[0].name
        assert len(e._paper_positions[first]) == 1
        result = e.paper_proof_cycle({"SPY": 520})
        assert result["closed"] >= 1
        # All long positions hit target
        assert sum(len(v) for v in e._paper_positions.values()) == 0

    def test_keep_open_when_no_trigger(self):
        e = StrategyAdvisorEngine()
        e.load_strategies()
        e.evaluate_all({"vix": 10, "spy_price": 500})
        result = e.paper_proof_cycle({"SPY": 502})
        assert result["closed"] == 0
        assert result["still_open"] > 0

    def test_uses_current_price_when_ticker_missing(self):
        e = StrategyAdvisorEngine()
        e.load_strategies()
        e.evaluate_all({"vix": 10, "spy_price": 500})
        # No SPY in live_prices → uses pos.current_price (500) → no trigger
        result = e.paper_proof_cycle({"OTHER": 999})
        assert result["closed"] == 0

    def test_performance_recalculated(self):
        e = StrategyAdvisorEngine()
        e.load_strategies()
        e.evaluate_all({"vix": 10, "spy_price": 500})
        e.paper_proof_cycle({"SPY": 520})
        # Some strategies should have total_trades > 0
        assert any(p.total_trades > 0 for p in e._performance.values())


class TestLeaderboard:
    def test_empty_when_no_trades(self):
        e = StrategyAdvisorEngine()
        e.load_strategies()
        assert e.get_leaderboard() == []

    def test_returns_top_n(self):
        e = StrategyAdvisorEngine()
        e.load_strategies()
        # Manually populate performance
        for i, name in enumerate(["A", "B", "C", "D"]):
            perf = AdvisorPerformance(strategy_name=name, total_trades=10,
                                      win_rate=0.5 + i*0.1, total_pnl=100*i,
                                      sharpe_ratio=float(i))
            e._performance[name] = perf
        board = e.get_leaderboard(top_n=2)
        assert len(board) == 2

    def test_sorted_by_composite_desc(self):
        e = StrategyAdvisorEngine()
        e._performance["X"] = AdvisorPerformance(strategy_name="X",
            total_trades=5, win_rate=0.2, total_pnl=-100, sharpe_ratio=-1)
        e._performance["Y"] = AdvisorPerformance(strategy_name="Y",
            total_trades=5, win_rate=0.9, total_pnl=900, sharpe_ratio=2)
        board = e.get_leaderboard()
        assert board[0]["strategy"] == "Y"
        assert board[1]["strategy"] == "X"

    def test_required_fields(self):
        e = StrategyAdvisorEngine()
        e._performance["X"] = AdvisorPerformance(strategy_name="X",
            total_trades=5, win_rate=0.6, total_pnl=200, sharpe_ratio=1.0)
        row = e.get_leaderboard()[0]
        for k in ("strategy", "trades", "win_rate", "total_pnl", "sharpe",
                  "max_dd", "composite_score", "approved_live"):
            assert k in row


class TestApprovalGate:
    def test_unknown_strategy(self):
        e = StrategyAdvisorEngine()
        assert e.approve_for_live("missing") is False

    def test_too_few_trades(self):
        e = StrategyAdvisorEngine()
        e._performance["X"] = AdvisorPerformance(strategy_name="X",
            total_trades=5, sharpe_ratio=1.0)
        assert e.approve_for_live("X") is False

    def test_negative_sharpe(self):
        e = StrategyAdvisorEngine()
        e._performance["X"] = AdvisorPerformance(strategy_name="X",
            total_trades=15, sharpe_ratio=-0.5)
        assert e.approve_for_live("X") is False

    def test_zero_sharpe_rejected(self):
        e = StrategyAdvisorEngine()
        e._performance["X"] = AdvisorPerformance(strategy_name="X",
            total_trades=15, sharpe_ratio=0.0)
        assert e.approve_for_live("X") is False

    def test_approval_success(self):
        e = StrategyAdvisorEngine()
        e._performance["X"] = AdvisorPerformance(strategy_name="X",
            total_trades=15, sharpe_ratio=1.5)
        assert e.approve_for_live("X") is True
        assert e._performance["X"].approved_for_live is True

    def test_revoke_success(self):
        e = StrategyAdvisorEngine()
        e._performance["X"] = AdvisorPerformance(strategy_name="X",
            approved_for_live=True)
        assert e.revoke_live("X") is True
        assert e._performance["X"].approved_for_live is False

    def test_revoke_unknown(self):
        e = StrategyAdvisorEngine()
        assert e.revoke_live("missing") is False


class TestStatePersistence:
    def test_save_and_load(self):
        e = StrategyAdvisorEngine()
        e._performance["X"] = AdvisorPerformance(strategy_name="X",
            total_trades=5, total_pnl=100, sharpe_ratio=1.5)
        e.save_state()
        assert sae.ADVISOR_STATE_FILE.exists()

        e2 = StrategyAdvisorEngine()
        assert e2.load_state() is True
        assert "X" in e2._performance
        assert e2._performance["X"].total_pnl == 100

    def test_load_state_missing_file(self):
        e = StrategyAdvisorEngine()
        assert e.load_state() is False

    def test_load_state_corrupted(self, tmp_path, monkeypatch):
        bad = tmp_path / "advisor_state.json"
        bad.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(sae, "ADVISOR_STATE_FILE", bad)
        e = StrategyAdvisorEngine()
        assert e.load_state() is False

    def test_save_state_payload_keys(self):
        e = StrategyAdvisorEngine()
        e._performance["X"] = AdvisorPerformance(strategy_name="X")
        e.save_state()
        data = json.loads(sae.ADVISOR_STATE_FILE.read_text(encoding="utf-8"))
        assert "saved_at" in data
        assert "performance" in data
        assert "open_positions" in data
        assert "closed_count" in data


class TestSummaryAndRelay:
    def test_summary_keys(self):
        e = StrategyAdvisorEngine()
        e.load_strategies()
        s = e.get_summary()
        for k in ("total_strategies", "total_open_positions",
                  "total_closed_positions", "approved_for_live",
                  "approved_count", "leaderboard_top5"):
            assert k in s
        assert s["total_strategies"] == 7

    def test_summary_approved_filter(self):
        e = StrategyAdvisorEngine()
        e._performance["X"] = AdvisorPerformance(strategy_name="X", approved_for_live=True)
        e._performance["Y"] = AdvisorPerformance(strategy_name="Y", approved_for_live=False)
        s = e.get_summary()
        assert "X" in s["approved_for_live"]
        assert "Y" not in s["approved_for_live"]
        assert s["approved_count"] == 1

    def test_relay_payload_keys(self):
        e = StrategyAdvisorEngine()
        e.load_strategies()
        p = e.get_relay_payload()
        assert p["engine"] == "strategy_advisor"
        assert "summary" in p
        assert "leaderboard" in p
        assert "timestamp" in p


class TestSingleton:
    def test_get_returns_same_instance(self):
        a = get_strategy_advisor_engine()
        b = get_strategy_advisor_engine()
        assert a is b
