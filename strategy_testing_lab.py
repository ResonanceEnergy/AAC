"""
strategy_testing_lab — Re-export shim.
Delegates to strategies.strategy_testing_lab_fixed which contains the real
StrategyTestingLab implementation.
"""

from strategies.strategy_testing_lab_fixed import (  # noqa: F401
    StrategyTestingLab,
    initialize_strategy_testing_lab,
    strategy_testing_lab,
)
