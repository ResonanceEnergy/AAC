"""Sprint 54 — assert mock ML models + mock simulation are gone.

Source-level + behavioral assertions.  Tests that:
- ``strategy_testing_lab.run_strategy_simulation`` raises NotImplementedError.
- ``_run_mock_simulation`` no longer exists on the lab.
- ``StrategyAnalysisEngine._initialize_prediction_models`` no longer creates
  fabricated model dicts (prediction_models is empty after init).
- ``_create_mock_model`` method is gone from the analysis engine.
- ``_analyze_market_regime`` raises NotImplementedError instead of returning
  hardcoded 1.2/0.8/0.9/1.1/1.0 regime multipliers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_LAB = _REPO / "strategies" / "strategy_testing_lab_fixed.py"
_ENG = _REPO / "strategies" / "strategy_analysis_engine.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level assertions
# ---------------------------------------------------------------------------

def test_lab_no_run_mock_simulation_method():
    src = _read(_LAB)
    assert "async def _run_mock_simulation" not in src
    assert "Run mock simulation with realistic ARB returns" not in src
    # The fake-returns lookup table is also gone.
    assert "'mean': 0.12, 'std': 0.08, 'win_rate': 0.65" not in src


def test_lab_no_using_mock_simulation_warning():
    src = _read(_LAB)
    assert "Strategy {strategy_id} not implemented, using mock simulation" not in src
    assert "using mock simulation" not in src


def test_engine_no_create_mock_model_method():
    src = _read(_ENG)
    assert "def _create_mock_model" not in src
    assert "self._create_mock_model" not in src


def test_engine_no_mock_regime_modifiers():
    src = _read(_ENG)
    # The hardcoded regime multipliers (1.2/0.8/0.9/1.1/1.0) were the
    # signature of the fake regime analysis.
    assert "'bull_market': 1.2" not in src
    assert "'bear_market': 0.8" not in src
    assert "'current_regime_prediction': 'neutral'" not in src
    assert "# Mock regime analysis" not in src
    assert "# Mock current regime" not in src


# ---------------------------------------------------------------------------
# Behavioral assertions (no network)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_strategy_simulation_raises():
    from strategies.strategy_testing_lab_fixed import strategy_testing_lab

    # Inject a synthetic config so we can reach the NotImplementedError path
    # without needing the optional CSV to be present.
    strategy_testing_lab.strategy_configs.setdefault(
        "__sprint54_probe__",
        {"id": "__sprint54_probe__", "name": "probe", "implemented": True},
    )
    with pytest.raises(NotImplementedError, match="Sprint 54"):
        await strategy_testing_lab.run_strategy_simulation(
            "__sprint54_probe__", "1M", 10
        )


@pytest.mark.asyncio
async def test_run_strategy_simulation_unknown_id_raises_value_error():
    from strategies.strategy_testing_lab_fixed import strategy_testing_lab

    with pytest.raises(ValueError, match="not found"):
        await strategy_testing_lab.run_strategy_simulation(
            "__nonexistent_strategy_id__", "1M", 10
        )


@pytest.mark.asyncio
async def test_initialize_prediction_models_leaves_dict_empty():
    from strategies.strategy_analysis_engine import StrategyAnalysisEngine

    eng = StrategyAnalysisEngine()
    await eng._initialize_prediction_models()
    assert eng.prediction_models == {}
    # And _create_mock_model is gone.
    assert not hasattr(eng, "_create_mock_model")


@pytest.mark.asyncio
async def test_analyze_market_regime_raises():
    from strategies.strategy_analysis_engine import StrategyAnalysisEngine

    eng = StrategyAnalysisEngine()
    with pytest.raises(NotImplementedError, match="Sprint 54"):
        await eng._analyze_market_regime({"total_return_pct": 5.0, "volatility": 0.15})


def test_lab_module_attribute_absent():
    import importlib

    import strategies.strategy_testing_lab_fixed as mod

    importlib.reload(mod)
    assert not hasattr(mod.strategy_testing_lab, "_run_mock_simulation")


def test_engine_module_attribute_absent():
    import importlib

    import strategies.strategy_analysis_engine as mod

    importlib.reload(mod)
    assert not hasattr(mod.strategy_analysis_engine, "_create_mock_model")
