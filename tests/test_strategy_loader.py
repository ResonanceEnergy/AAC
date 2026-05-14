from __future__ import annotations

import csv
from pathlib import Path

import pytest

from shared import strategy_loader as sl_module
from shared.strategy_loader import (
    StrategyCategory,
    StrategyConfig,
    StrategyLoader,
    StrategyStatus,
    get_strategy_loader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FIELDS = ["id", "strategy_name", "one_liner", "sources"]


def _write_csv(path: Path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _row(
    sid=1,
    name="ETF NAV Arbitrage Strategy Detector",
    one_liner="Exploit ETF NAV vs market price discrepancies on creation/redemption events",
    sources="turn1;turn2",
):
    return {"id": str(sid), "strategy_name": name, "one_liner": one_liner, "sources": sources}


def _make_loader(tmp_path: Path, rows) -> StrategyLoader:
    """Build a loader pointing at a fresh CSV under tmp_path."""
    csv_name = "test_strategies.csv"
    # StrategyLoader resolves Path(__file__).parent.parent / csv_path
    # so we need to give it an absolute path workaround: create the file at
    # the resolved location, OR monkey-patch csv_path post-init.
    csv_file = tmp_path / csv_name
    _write_csv(csv_file, rows)
    loader = StrategyLoader(csv_path=csv_name)
    loader.csv_path = csv_file  # override resolved path
    return loader


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestStrategyCategory:
    def test_values(self):
        assert StrategyCategory.ETF_ARBITRAGE.value == "etf_arbitrage"
        assert StrategyCategory.INDEX_ARBITRAGE.value == "index_arbitrage"
        assert StrategyCategory.VOLATILITY_ARBITRAGE.value == "volatility_arbitrage"
        assert StrategyCategory.EVENT_DRIVEN.value == "event_driven"
        assert StrategyCategory.SEASONALITY.value == "seasonality"
        assert StrategyCategory.FLOW_BASED.value == "flow_based"
        assert StrategyCategory.MARKET_MAKING.value == "market_making"
        assert StrategyCategory.CORRELATION.value == "correlation"

    def test_count(self):
        assert len(list(StrategyCategory)) == 8


class TestStrategyStatus:
    def test_values(self):
        assert StrategyStatus.VALID.value == "valid"
        assert StrategyStatus.INVALID.value == "invalid"
        assert StrategyStatus.REQUIRES_REVIEW.value == "requires_review"
        assert StrategyStatus.NOT_IMPLEMENTED.value == "not_implemented"

    def test_count(self):
        assert len(list(StrategyStatus)) == 4


# ---------------------------------------------------------------------------
# StrategyConfig dataclass
# ---------------------------------------------------------------------------


class TestStrategyConfigDataclass:
    def test_required_fields(self):
        cfg = StrategyConfig(
            id=1,
            name="Test",
            description="desc",
            sources=["turn1"],
            category=StrategyCategory.EVENT_DRIVEN,
            status=StrategyStatus.VALID,
            validation_errors=[],
        )
        assert cfg.id == 1
        assert cfg.implementation_notes is None

    def test_is_valid_true_when_status_valid(self):
        cfg = StrategyConfig(1, "n", "d", ["turn1"], StrategyCategory.EVENT_DRIVEN, StrategyStatus.VALID, [])
        assert cfg.is_valid is True

    def test_is_valid_false_when_invalid(self):
        cfg = StrategyConfig(1, "n", "d", ["turn1"], StrategyCategory.EVENT_DRIVEN, StrategyStatus.INVALID, ["x"])
        assert cfg.is_valid is False


# ---------------------------------------------------------------------------
# Loader init
# ---------------------------------------------------------------------------


class TestStrategyLoaderInit:
    def test_default_csv_path(self):
        loader = StrategyLoader()
        assert loader.csv_path.name == "50_arbitrage_strategies.csv"
        assert loader.strategies == []
        assert loader._loaded is False

    def test_custom_csv_path(self):
        loader = StrategyLoader(csv_path="custom.csv")
        assert loader.csv_path.name == "custom.csv"


# ---------------------------------------------------------------------------
# load_strategies
# ---------------------------------------------------------------------------


class TestLoadStrategies:
    @pytest.mark.asyncio
    async def test_loads_rows(self, tmp_path):
        loader = _make_loader(
            tmp_path,
            [_row(1), _row(2, name="Volatility Premium Capture", one_liner="VRP IV vs RV gap exploit on SPX index")],
        )
        out = await loader.load_strategies()
        assert len(out) == 2
        assert out[0].id == 1
        assert out[1].id == 2

    @pytest.mark.asyncio
    async def test_idempotent_when_already_loaded(self, tmp_path):
        loader = _make_loader(tmp_path, [_row(1)])
        first = await loader.load_strategies()
        # mutate file; should not reload
        _write_csv(loader.csv_path, [_row(1), _row(2)])
        second = await loader.load_strategies()
        assert first is second
        assert len(second) == 1

    @pytest.mark.asyncio
    async def test_missing_file_raises(self, tmp_path):
        loader = StrategyLoader(csv_path="nope.csv")
        loader.csv_path = tmp_path / "absent.csv"
        with pytest.raises(FileNotFoundError):
            await loader.load_strategies()

    @pytest.mark.asyncio
    async def test_skips_unparseable_row(self, tmp_path):
        # Row with non-int id should be skipped (caught in _parse_strategy_row)
        loader = _make_loader(
            tmp_path,
            [
                _row(1),
                {"id": "not_an_int", "strategy_name": "x", "one_liner": "y" * 30, "sources": "turn1"},
            ],
        )
        out = await loader.load_strategies()
        assert len(out) == 1
        assert out[0].id == 1


# ---------------------------------------------------------------------------
# Unicode sanitisation
# ---------------------------------------------------------------------------


class TestUnicodeSanitization:
    @pytest.mark.asyncio
    async def test_em_dash_replaced(self, tmp_path):
        loader = _make_loader(
            tmp_path,
            [_row(1, name="ETF \u2014 NAV Arbitrage Bot")],
        )
        out = await loader.load_strategies()
        assert "\u2014" not in out[0].name
        assert "ETF - NAV Arbitrage Bot" == out[0].name

    @pytest.mark.asyncio
    async def test_non_ascii_stripped(self, tmp_path):
        loader = _make_loader(
            tmp_path,
            [_row(1, name="ETF\u00e9 NAV Arbitrage Detector")],
        )
        out = await loader.load_strategies()
        # \u00e9 is non-ASCII and not in the explicit replacement table → stripped
        assert "\u00e9" not in out[0].name


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------


class TestCategorize:
    def test_etf_arbitrage(self):
        loader = StrategyLoader()
        assert loader._categorize_strategy("ETF NAV", "creation/redemption") == StrategyCategory.ETF_ARBITRAGE

    def test_index_arbitrage(self):
        loader = StrategyLoader()
        assert loader._categorize_strategy("Index Reconstitution", "inclusion event") == StrategyCategory.INDEX_ARBITRAGE

    def test_volatility(self):
        loader = StrategyLoader()
        assert (
            loader._categorize_strategy("Volatility Premium", "IV vs RV variance gap")
            == StrategyCategory.VOLATILITY_ARBITRAGE
        )

    def test_event_driven(self):
        loader = StrategyLoader()
        assert loader._categorize_strategy("Earnings drift", "post fomc move") == StrategyCategory.EVENT_DRIVEN

    def test_seasonality(self):
        loader = StrategyLoader()
        assert loader._categorize_strategy("Overnight TOM", "weekly seasonality") == StrategyCategory.SEASONALITY

    def test_flow(self):
        loader = StrategyLoader()
        assert loader._categorize_strategy("Liquidity flow", "pressure") == StrategyCategory.FLOW_BASED

    def test_market_making(self):
        loader = StrategyLoader()
        assert loader._categorize_strategy("Auction MM", "market making") == StrategyCategory.MARKET_MAKING

    def test_correlation(self):
        loader = StrategyLoader()
        assert loader._categorize_strategy("Dispersion", "correlation trade") == StrategyCategory.CORRELATION

    def test_default_event_driven(self):
        loader = StrategyLoader()
        assert loader._categorize_strategy("Random", "no keyword match") == StrategyCategory.EVENT_DRIVEN


# ---------------------------------------------------------------------------
# _validate_strategy
# ---------------------------------------------------------------------------


class TestValidateStrategy:
    @pytest.mark.asyncio
    async def test_valid_when_all_fields_good(self):
        loader = StrategyLoader()
        status, errors = await loader._validate_strategy(
            1, "Name", "A reasonably long description here", ["turn1"]
        )
        assert status == StrategyStatus.VALID
        assert errors == []

    @pytest.mark.asyncio
    async def test_missing_name(self):
        loader = StrategyLoader()
        status, errors = await loader._validate_strategy(1, "", "desc that is long enough now", ["turn1"])
        assert status == StrategyStatus.INVALID
        assert any("name is required" in e for e in errors)

    @pytest.mark.asyncio
    async def test_missing_description(self):
        loader = StrategyLoader()
        status, errors = await loader._validate_strategy(1, "n", "", ["turn1"])
        assert status == StrategyStatus.INVALID
        assert any("description is required" in e for e in errors)

    @pytest.mark.asyncio
    async def test_missing_sources(self):
        loader = StrategyLoader()
        status, errors = await loader._validate_strategy(1, "n", "long enough description here", [])
        assert status == StrategyStatus.INVALID
        assert any("sources are required" in e for e in errors)

    @pytest.mark.asyncio
    async def test_invalid_source_format(self):
        loader = StrategyLoader()
        status, errors = await loader._validate_strategy(
            1, "n", "long enough description here", ["foo"]
        )
        assert status == StrategyStatus.INVALID
        assert any("Invalid source format" in e for e in errors)

    @pytest.mark.asyncio
    async def test_description_too_short(self):
        loader = StrategyLoader()
        status, errors = await loader._validate_strategy(1, "n", "short", ["turn1"])
        assert status == StrategyStatus.INVALID
        assert any("too short" in e for e in errors)

    @pytest.mark.asyncio
    async def test_description_too_long(self):
        loader = StrategyLoader()
        status, errors = await loader._validate_strategy(1, "n", "x" * 250, ["turn1"])
        assert status == StrategyStatus.INVALID
        assert any("too long" in e for e in errors)


# ---------------------------------------------------------------------------
# get_strategies_by_category / get_valid_strategies
# ---------------------------------------------------------------------------


class TestGetters:
    @pytest.mark.asyncio
    async def test_get_strategies_by_category(self, tmp_path):
        loader = _make_loader(
            tmp_path,
            [
                _row(1, name="ETF NAV Arbitrage Bot", one_liner="ETF creation redemption discrepancy bot"),
                _row(2, name="Volatility VRP", one_liner="IV vs RV variance premium harvest tactic"),
            ],
        )
        out = await loader.get_strategies_by_category(StrategyCategory.ETF_ARBITRAGE)
        assert len(out) == 1
        assert out[0].id == 1

    @pytest.mark.asyncio
    async def test_get_strategies_by_category_empty(self, tmp_path):
        loader = _make_loader(tmp_path, [_row(1)])
        out = await loader.get_strategies_by_category(StrategyCategory.MARKET_MAKING)
        assert out == []

    @pytest.mark.asyncio
    async def test_get_valid_strategies(self, tmp_path):
        loader = _make_loader(
            tmp_path,
            [
                _row(1),
                _row(2, name="Bad Source Strategy", sources="badsrc"),  # invalid source format
            ],
        )
        out = await loader.get_valid_strategies()
        assert len(out) == 1
        assert out[0].id == 1


# ---------------------------------------------------------------------------
# get_strategy_summary
# ---------------------------------------------------------------------------


class TestGetStrategySummary:
    @pytest.mark.asyncio
    async def test_summary_counts(self, tmp_path):
        loader = _make_loader(
            tmp_path,
            [
                _row(1),
                _row(2, name="Volatility VRP", one_liner="IV vs RV variance premium harvest tactic"),
                _row(3, name="Bad Strategy", sources="badsrc"),
            ],
        )
        summary = await loader.get_strategy_summary()
        assert summary["total_strategies"] == 3
        assert summary["valid_strategies"] == 2
        assert summary["invalid_strategies"] == 1
        assert summary["validation_rate"] == pytest.approx(2 / 3)
        assert sum(summary["categories"].values()) == 3

    @pytest.mark.asyncio
    async def test_summary_empty(self, tmp_path):
        loader = _make_loader(tmp_path, [])
        summary = await loader.get_strategy_summary()
        assert summary["total_strategies"] == 0
        assert summary["validation_rate"] == 0


# ---------------------------------------------------------------------------
# validate_all_strategies
# ---------------------------------------------------------------------------


class TestValidateAllStrategies:
    @pytest.mark.asyncio
    async def test_segregates_passed_and_failed(self, tmp_path):
        loader = _make_loader(
            tmp_path,
            [
                _row(1),
                _row(2, name="Bad", sources="bad"),
            ],
        )
        results = await loader.validate_all_strategies()
        assert len(results["passed"]) == 1
        assert len(results["failed"]) == 1
        assert results["passed"][0]["id"] == 1
        assert "category" in results["passed"][0]
        assert results["failed"][0]["id"] == 2
        assert results["failed"][0]["errors"]

    @pytest.mark.asyncio
    async def test_warnings_key_present(self, tmp_path):
        loader = _make_loader(tmp_path, [_row(1)])
        results = await loader.validate_all_strategies()
        assert "warnings" in results
        assert results["warnings"] == []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


class TestGetStrategyLoader:
    def test_returns_singleton(self):
        # reset module global
        sl_module._strategy_loader = None
        a = get_strategy_loader()
        b = get_strategy_loader()
        assert a is b
        assert isinstance(a, StrategyLoader)

    def test_returns_existing_instance(self):
        sentinel = StrategyLoader()
        sl_module._strategy_loader = sentinel
        assert get_strategy_loader() is sentinel
        sl_module._strategy_loader = None  # cleanup
