from __future__ import annotations
"""tests/test_second_strategy.py — Sprint 6: Vol Premium + Aggregator + Backtest.

All tests are deterministic and offline — yfinance is mocked throughout.
"""

import math
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from shared.signal import AssetClass, Direction, TradeSignal
from strategies.vol_premium_signals import (
    VolPremiumReading,
    _compute_realized_hv,
    _get_atm_iv,
    _get_closes_and_spot,
    generate_vol_premium_signals,
    get_vol_premium_readings,
)
from strategies.signal_aggregator import (
    AggregatedSignal,
    aggregate,
    get_combined_signals,
)
from strategies.simple_backtest import (
    BacktestReport,
    StrategyPerformance,
    _evaluate_trades,
    _rolling_hv,
    run_backtest,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_signal(
    ticker: str = "SPY",
    direction: Direction = Direction.LONG_PUT,
    confidence: float = 0.70,
    entry: float = 500.0,
    strategy: str = "war_room_engine",
) -> TradeSignal:
    return TradeSignal(
        ticker=ticker,
        direction=direction,
        confidence=confidence,
        entry=entry,
        stop=0.0,
        target=0.0,
        size=0.05,
        strategy=strategy,
        regime="TEST",
        asset_class=AssetClass.OPTION,
    )


def _mock_yf_ticker(closes: list[float], option_exp: list[str], atm_put_iv: float):
    """Build a mock yfinance Ticker that returns controlled data."""
    import pandas as pd

    ticker_mock = MagicMock()

    # history() returns a DataFrame with 'Close' column
    idx = pd.date_range(end="2026-04-21", periods=len(closes), freq="B")
    ticker_mock.history.return_value = pd.DataFrame({"Close": closes}, index=idx)

    # options
    ticker_mock.options = option_exp

    # option_chain() returns an object with .puts DataFrame
    chain_mock = MagicMock()
    puts_df = pd.DataFrame({
        "strike": [490.0, 500.0, 510.0],
        "impliedVolatility": [atm_put_iv * 1.05, atm_put_iv, atm_put_iv * 0.95],
        "lastPrice": [2.50, 3.00, 1.80],
    })
    chain_mock.puts = puts_df
    ticker_mock.option_chain.return_value = chain_mock

    return ticker_mock


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: _compute_realized_hv
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeRealizedHV:
    def test_returns_zero_for_fewer_than_5_prices(self):
        assert _compute_realized_hv([100.0, 101.0, 99.0]) == 0.0

    def test_constant_prices_return_zero(self):
        closes = [100.0] * 35
        hv = _compute_realized_hv(closes)
        assert hv == pytest.approx(0.0, abs=1e-9)

    def test_volatile_prices_return_positive(self):
        import numpy as np
        rng = np.random.default_rng(42)
        prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.015, 35))
        hv = _compute_realized_hv(list(prices))
        assert hv > 0.10  # annualised vol should be comfortably above 10 %

    def test_annual_scaling(self):
        # Alternating ±1 % moves → annualised HV ≈ 0.01 * sqrt(252) ≈ 0.1587
        import math
        closes = [100.0 * (math.exp(0.01 * (1 if i % 2 == 0 else -1)) ** i) for i in range(35)]
        # Use synthetic log-return series with known std instead
        # std(log_ret) ≈ 0.01, annualised = 0.01 * sqrt(252) ≈ 0.1587
        import numpy as np
        rng = np.random.default_rng(123)
        prices = 100.0 * np.cumprod(1 + rng.choice([-0.01, 0.01], size=35))
        hv = _compute_realized_hv(list(prices))
        assert 0.12 < hv < 0.22


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: _get_closes_and_spot / _get_atm_iv
# ══════════════════════════════════════════════════════════════════════════════

class TestYFinanceFetchers:
    def test_get_closes_returns_values_from_yfinance(self):
        closes_data = [100.0 + i for i in range(35)]
        mock_ticker = _mock_yf_ticker(closes_data, ["2026-05-16"], 0.20)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            closes, spot = _get_closes_and_spot("SPY")

        assert len(closes) == 35
        assert spot == pytest.approx(134.0)

    def test_get_closes_returns_empty_on_failure(self):
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = RuntimeError("network error")

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            closes, spot = _get_closes_and_spot("SPY")

        assert closes == []
        assert spot == 0.0

    def test_get_atm_iv_finds_nearest_strike(self):
        closes_data = [490.0 + i for i in range(35)]
        mock_ticker = _mock_yf_ticker(closes_data, ["2026-05-16"], 0.22)
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            iv = _get_atm_iv("SPY", 500.0)

        assert iv == pytest.approx(0.22)

    def test_get_atm_iv_returns_zero_on_empty_chain(self):
        import pandas as pd
        mock_ticker = MagicMock()
        mock_ticker.options = ["2026-05-16"]
        chain_mock = MagicMock()
        chain_mock.puts = pd.DataFrame()
        mock_ticker.option_chain.return_value = chain_mock
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            iv = _get_atm_iv("SPY", 500.0)

        assert iv == 0.0

    def test_get_atm_iv_returns_zero_on_exception(self):
        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = RuntimeError("timeout")

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            iv = _get_atm_iv("SPY", 500.0)

        assert iv == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: get_vol_premium_readings
# ══════════════════════════════════════════════════════════════════════════════

class TestGetVolPremiumReadings:
    # Patch at function level — avoids the HV=0 guard for flat prices.
    _CLOSES = [490.0 + i * 0.5 for i in range(35)]  # slight uptrend → HV > 0
    _SPOT = 500.0

    def test_returns_reading_for_each_ticker(self):
        with (
            patch("strategies.vol_premium_signals._get_closes_and_spot", return_value=(self._CLOSES, self._SPOT)),
            patch("strategies.vol_premium_signals._get_atm_iv", return_value=0.25),
        ):
            readings = get_vol_premium_readings(tickers=["SPY", "QQQ"], fetch_iv=True)
        assert len(readings) == 2

    def test_reading_fields_populated(self):
        with (
            patch("strategies.vol_premium_signals._get_closes_and_spot", return_value=(self._CLOSES, self._SPOT)),
            patch("strategies.vol_premium_signals._get_atm_iv", return_value=0.25),
        ):
            readings = get_vol_premium_readings(tickers=["SPY"], fetch_iv=True)
        r = readings[0]
        assert r.ticker == "SPY"
        assert r.realized_hv > 0
        assert r.implied_vol == pytest.approx(0.25)
        assert r.iv_hv_ratio > 0
        assert r.option_available is True

    def test_fetch_iv_false_skips_options_chain(self):
        with (
            patch("strategies.vol_premium_signals._get_closes_and_spot", return_value=(self._CLOSES, self._SPOT)),
        ):
            readings = get_vol_premium_readings(tickers=["SPY"], fetch_iv=False)
        r = readings[0]
        assert r.implied_vol == 0.0
        assert r.option_available is False


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: generate_vol_premium_signals
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateVolPremiumSignals:
    def _patch_readings(self, readings: list[VolPremiumReading]):
        return patch(
            "strategies.vol_premium_signals.get_vol_premium_readings",
            return_value=readings,
        )

    def test_emits_signal_when_iv_hv_above_threshold(self):
        readings = [
            VolPremiumReading("SPY", 0.15, 0.22, 1.47, 500.0, option_available=True),
        ]
        with self._patch_readings(readings):
            signals = generate_vol_premium_signals(fetch_iv=True)

        assert len(signals) == 1
        s = signals[0]
        assert s.ticker == "SPY"
        assert s.direction == Direction.LONG_PUT
        assert s.asset_class == AssetClass.OPTION
        assert 0.50 <= s.confidence <= 0.90
        assert s.strategy == "vol_premium"

    def test_no_signal_when_iv_hv_below_threshold(self):
        readings = [
            VolPremiumReading("SPY", 0.15, 0.16, 1.07, 500.0, option_available=True),
        ]
        with self._patch_readings(readings):
            signals = generate_vol_premium_signals(fetch_iv=True)
        assert signals == []

    def test_hv_only_signal_when_iv_unavailable_and_hv_high(self):
        readings = [
            VolPremiumReading("HYG", 0.30, 0.0, 0.0, 75.0, option_available=False),
        ]
        with self._patch_readings(readings):
            signals = generate_vol_premium_signals(fetch_iv=False)

        assert len(signals) == 1
        assert signals[0].confidence <= 0.60

    def test_no_hv_only_signal_when_hv_low(self):
        readings = [
            VolPremiumReading("HYG", 0.12, 0.0, 0.0, 75.0, option_available=False),
        ]
        with self._patch_readings(readings):
            signals = generate_vol_premium_signals(fetch_iv=False)
        assert signals == []

    def test_signals_sorted_by_confidence_descending(self):
        readings = [
            VolPremiumReading("JNK", 0.18, 0.28, 1.56, 95.0, option_available=True),
            VolPremiumReading("SPY", 0.15, 0.26, 1.73, 500.0, option_available=True),
        ]
        with self._patch_readings(readings):
            signals = generate_vol_premium_signals(fetch_iv=True)

        confidences = [s.confidence for s in signals]
        assert confidences == sorted(confidences, reverse=True)

    def test_returns_empty_on_readings_exception(self):
        with patch(
            "strategies.vol_premium_signals.get_vol_premium_readings",
            side_effect=RuntimeError("unexpected"),
        ):
            signals = generate_vol_premium_signals()
        assert signals == []

    def test_size_capped_at_max(self):
        # Very high IV/HV ratio → size should not exceed _MAX_SIZE
        readings = [
            VolPremiumReading("SPY", 0.10, 0.50, 5.0, 500.0, option_available=True),
        ]
        with self._patch_readings(readings):
            signals = generate_vol_premium_signals(fetch_iv=True)

        assert len(signals) == 1
        assert signals[0].size <= 0.08


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: aggregate (signal_aggregator)
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregate:
    def test_single_source_passes_through(self):
        s = _make_signal("SPY", confidence=0.70, strategy="war_room")
        result = aggregate([([s], 1.0)])
        assert len(result) == 1
        assert result[0].signal.ticker == "SPY"
        assert result[0].signal.confidence == pytest.approx(0.70)
        assert result[0].boosted is False

    def test_agreement_boosts_confidence(self):
        s1 = _make_signal("SPY", confidence=0.65, strategy="war_room")
        s2 = _make_signal("SPY", confidence=0.55, strategy="vol_premium")
        result = aggregate([([s1], 0.6), ([s2], 0.4)])
        assert len(result) == 1
        r = result[0]
        assert r.boosted is True
        # weighted avg ≈ (0.65*0.6 + 0.55*0.4) / 1.0 = 0.61, boosted by 0.05 → 0.66
        assert r.signal.confidence > r.raw_confidence

    def test_different_tickers_stay_separate(self):
        s1 = _make_signal("SPY", confidence=0.70)
        s2 = _make_signal("QQQ", confidence=0.60)
        result = aggregate([([s1, s2], 1.0)])
        assert len(result) == 2

    def test_different_directions_stay_separate(self):
        s1 = _make_signal("SPY", direction=Direction.LONG_PUT, confidence=0.70)
        s2 = _make_signal("SPY", direction=Direction.LONG_CALL, confidence=0.60)
        result = aggregate([([s1, s2], 1.0)])
        assert len(result) == 2

    def test_output_sorted_by_confidence_descending(self):
        signals = [
            _make_signal("A", confidence=0.50),
            _make_signal("B", confidence=0.80),
            _make_signal("C", confidence=0.65),
        ]
        result = aggregate([(signals, 1.0)])
        confs = [r.signal.confidence for r in result]
        assert confs == sorted(confs, reverse=True)

    def test_empty_lists_return_empty(self):
        result = aggregate([([], 1.0), ([], 0.5)])
        assert result == []

    def test_best_signal_used_as_base(self):
        s1 = _make_signal("SPY", confidence=0.80, entry=500.0, strategy="war_room")
        s2 = _make_signal("SPY", confidence=0.60, entry=498.0, strategy="vol_premium")
        result = aggregate([([s1], 1.0), ([s2], 0.5)])
        # s1 has higher confidence → used as base → entry = 500
        assert result[0].signal.entry == pytest.approx(500.0)

    def test_max_confidence_cap(self):
        # Even many sources agreeing should not exceed _MAX_CONFIDENCE
        signals = [_make_signal("SPY", confidence=0.95, strategy=f"s{i}") for i in range(10)]
        result = aggregate([(signals, 1.0)])
        assert result[0].signal.confidence <= 0.95

    def test_sources_listed_in_aggregated_signal(self):
        s1 = _make_signal("SPY", strategy="war_room")
        s2 = _make_signal("SPY", strategy="vol_premium")
        result = aggregate([([s1], 0.6), ([s2], 0.4)])
        assert "war_room" in result[0].sources
        assert "vol_premium" in result[0].sources


# ══════════════════════════════════════════════════════════════════════════════
# Section 6: get_combined_signals
# ══════════════════════════════════════════════════════════════════════════════

class TestGetCombinedSignals:
    def test_combines_both_strategies(self):
        wr = [_make_signal("SPY", confidence=0.70, strategy="war_room_engine")]
        vp = [_make_signal("QQQ", confidence=0.60, strategy="vol_premium")]

        with (
            patch("strategies.signal_generator.generate_signals", return_value=wr),
            patch("strategies.vol_premium_signals.generate_vol_premium_signals", return_value=vp),
        ):
            combined = get_combined_signals()

        tickers = [s.ticker for s in combined]
        assert "SPY" in tickers
        assert "QQQ" in tickers

    def test_agreement_signal_gets_boosted(self):
        wr = [_make_signal("SPY", confidence=0.70, strategy="war_room_engine")]
        vp = [_make_signal("SPY", confidence=0.60, strategy="vol_premium")]

        with (
            patch("strategies.signal_generator.generate_signals", return_value=wr),
            patch("strategies.vol_premium_signals.generate_vol_premium_signals", return_value=vp),
        ):
            combined = get_combined_signals()

        spy = next(s for s in combined if s.ticker == "SPY")
        # Boosted confidence: weighted avg = (0.70*0.6 + 0.60*0.4)/1.0 = 0.66, +0.05 = 0.71
        assert spy.confidence > 0.65

    def test_empty_strategies_return_empty(self):
        with (
            patch("strategies.signal_generator.generate_signals", return_value=[]),
            patch("strategies.vol_premium_signals.generate_vol_premium_signals", return_value=[]),
        ):
            combined = get_combined_signals()
        assert combined == []


# ══════════════════════════════════════════════════════════════════════════════
# Section 7: simple_backtest helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestRollingHV:
    def test_returns_same_length_as_input(self):
        closes = [100.0 + i * 0.5 for i in range(40)]
        hvs = _rolling_hv(closes, window=10)
        assert len(hvs) == len(closes)

    def test_first_window_entries_are_zero(self):
        closes = [100.0 + i for i in range(40)]
        hvs = _rolling_hv(closes, window=10)
        assert all(h == 0.0 for h in hvs[:10])

    def test_values_after_window_are_positive_for_volatile_series(self):
        import numpy as np
        rng = np.random.default_rng(99)
        prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, 50))
        hvs = _rolling_hv(list(prices), window=10)
        assert all(h >= 0 for h in hvs)
        assert any(h > 0 for h in hvs[10:])

    def test_constant_prices_give_zero_hv(self):
        closes = [100.0] * 40
        hvs = _rolling_hv(closes, window=10)
        assert all(h == pytest.approx(0.0, abs=1e-9) for h in hvs[10:])


class TestEvaluateTrades:
    def _make_dates_closes(self, n: int = 30) -> tuple[list[date], list[float]]:
        import datetime
        start = date(2026, 1, 2)
        dates = [start + timedelta(days=i) for i in range(n)]
        # Decreasing prices: each day -0.5%
        closes = [500.0 * (0.995 ** i) for i in range(n)]
        return dates, closes

    def test_win_when_price_falls_enough(self):
        dates, closes = self._make_dates_closes(30)
        # Signal on day 0; by day 10, price has fallen ~5% → win
        mask = [True] + [False] * 29
        trades = _evaluate_trades("SPY", "test", dates, closes, mask, hold_days=10, win_threshold=0.02)
        assert len(trades) == 1
        assert trades[0].win is True

    def test_no_signal_no_trades(self):
        dates, closes = self._make_dates_closes(30)
        mask = [False] * 30
        trades = _evaluate_trades("SPY", "test", dates, closes, mask)
        assert trades == []

    def test_signal_at_end_with_no_future_data_skipped(self):
        dates, closes = self._make_dates_closes(15)
        mask = [False] * 14 + [True]   # signal on last day, no future data
        trades = _evaluate_trades("SPY", "test", dates, closes, mask, hold_days=10)
        assert trades == []

    def test_no_win_when_price_rises(self):
        dates = [date(2026, 1, 2) + timedelta(days=i) for i in range(30)]
        # Rising prices: +1% per day
        closes = [500.0 * (1.01 ** i) for i in range(30)]
        mask = [True] + [False] * 29
        trades = _evaluate_trades("SPY", "test", dates, closes, mask, hold_days=10, win_threshold=0.02)
        assert len(trades) == 1
        assert trades[0].win is False


# ══════════════════════════════════════════════════════════════════════════════
# Section 8: run_backtest (integration)
# ══════════════════════════════════════════════════════════════════════════════

class TestRunBacktest:
    def _make_mock_yf(self):
        """Return a mock yfinance module where all tickers have declining prices."""
        import numpy as np
        import pandas as pd

        # 130 business days of 0.3% daily decline
        closes_spy = [500.0 * (0.997 ** i) for i in range(130)]
        idx = pd.bdate_range(end="2026-04-21", periods=130)

        def make_ticker(closes):
            m = MagicMock()
            m.history.return_value = pd.DataFrame({"Close": closes}, index=idx)
            m.options = ["2026-05-16"]
            chain = MagicMock()
            puts = pd.DataFrame({
                "strike": [490.0, 500.0, 510.0],
                "impliedVolatility": [0.22, 0.20, 0.18],
            })
            chain.puts = puts
            m.option_chain.return_value = chain
            return m

        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = make_ticker(closes_spy)
        return mock_yf

    def test_returns_backtest_report(self):
        mock_yf = self._make_mock_yf()
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            report = run_backtest(tickers=["SPY"], lookback_days=130)
        assert isinstance(report, BacktestReport)

    def test_report_has_three_strategies(self):
        mock_yf = self._make_mock_yf()
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            report = run_backtest(tickers=["SPY"], lookback_days=130)
        strategy_names = {r.strategy for r in report.results}
        assert strategy_names == {"war_room_proxy", "vol_premium_proxy", "combined_proxy"}

    def test_win_rates_between_0_and_1(self):
        mock_yf = self._make_mock_yf()
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            report = run_backtest(tickers=["SPY"], lookback_days=130)
        for r in report.results:
            assert 0.0 <= r.win_rate <= 1.0

    def test_format_report_returns_string(self):
        mock_yf = self._make_mock_yf()
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            report = run_backtest(tickers=["SPY"], lookback_days=130)
        text = report.format_report()
        assert "Strategy" in text
        assert "war_room_proxy" in text

    def test_to_dict_contains_strategies_key(self):
        mock_yf = self._make_mock_yf()
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            report = run_backtest(tickers=["SPY"], lookback_days=130)
        d = report.to_dict()
        assert "strategies" in d
        assert len(d["strategies"]) == 3
