"""Unit tests for strategies.strategy_implementation_factory — StrategyImplementationFactory."""
import pytest
from unittest.mock import MagicMock


class TestStrategyImplementationFactory:
    """Tests for StrategyImplementationFactory strategy mappings and helpers."""

    @pytest.fixture
    def factory(self):
        """Create factory with mocked dependencies."""
        mock_data_agg = MagicMock()
        mock_comm = MagicMock()
        mock_audit = MagicMock()
        from strategies.strategy_implementation_factory import StrategyImplementationFactory
        return StrategyImplementationFactory(mock_data_agg, mock_comm, mock_audit)

    def test_strategy_mappings_populated(self, factory):
        assert len(factory.strategy_mappings) > 0

    def test_fifty_strategies_mapped(self, factory):
        # CSV defines ~50 strategies, mapping should have ~50 entries
        assert len(factory.strategy_mappings) >= 48

    def test_etf_strategies_present(self, factory):
        etf_keys = [k for k in factory.strategy_mappings if k.startswith("ETF")]
        assert len(etf_keys) >= 3

    def test_volatility_strategies_present(self, factory):
        vol_keys = [k for k in factory.strategy_mappings if "Variance" in k or "VRP" in k or "Dispersion" in k]
        assert len(vol_keys) >= 5

    def test_mapping_values_are_strings(self, factory):
        for name, cls_name in factory.strategy_mappings.items():
            assert isinstance(cls_name, str), f"Mapping for {name} should be a string class name"

    def test_mapping_values_end_with_strategy(self, factory):
        for name, cls_name in factory.strategy_mappings.items():
            assert cls_name.endswith("Strategy"), f"Class name for {name} should end with 'Strategy'"

    def test_strategy_name_to_module(self, factory):
        module = factory._strategy_name_to_module("ETF-NAV Dislocation Harvesting")
        assert isinstance(module, str)
        assert len(module) > 0

    def test_implemented_strategies_initially_empty(self, factory):
        assert factory.implemented_strategies == {}

    def test_strategy_loader_initialized(self, factory):
        assert factory.strategy_loader is not None
