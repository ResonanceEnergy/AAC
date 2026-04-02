"""AAC Integration Module - Cross-Department Coordination."""
from .cross_department_engine import (
    BakeoffIntegration,
    BigBrainIntelligenceAdapter,
    CentralAccountingAdapter,
    CrossDepartmentEvent,
    CrossDepartmentIntegrationEngine,
    CryptoIntelligenceAdapter,
    Department,
    DepartmentMetric,
    TradingExecutionAdapter,
)

__all__ = [
    "CrossDepartmentIntegrationEngine",
    "BakeoffIntegration",
    "Department",
    "DepartmentMetric",
    "CrossDepartmentEvent",
    "TradingExecutionAdapter",
    "BigBrainIntelligenceAdapter",
    "CentralAccountingAdapter",
    "CryptoIntelligenceAdapter",
]
