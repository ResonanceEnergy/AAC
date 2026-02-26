"""AAC Doctrine Package - Doctrine analysis and application engine."""

from .doctrine_engine import (
    DoctrineEngine,
    DoctrineApplicationService,
    Department,
    ComplianceState,
    AZPrimeState,
    ActionType,
    DoctrineViolation,
    DoctrineComplianceReport,
    DOCTRINE_PACKS,
)

from .doctrine_integration import (
    DoctrineOrchestrator,
    TradingExecutionDoctrineAdapter,
    BigBrainDoctrineAdapter,
    CentralAccountingDoctrineAdapter,
    CryptoIntelligenceDoctrineAdapter,
    SharedInfraDoctrineAdapter,
)

__all__ = [
    # Engine
    "DoctrineEngine",
    "DoctrineApplicationService",
    "Department",
    "ComplianceState",
    "AZPrimeState",
    "ActionType",
    "DoctrineViolation",
    "DoctrineComplianceReport",
    "DOCTRINE_PACKS",
    # Integration
    "DoctrineOrchestrator",
    "TradingExecutionDoctrineAdapter",
    "BigBrainDoctrineAdapter",
    "CentralAccountingDoctrineAdapter",
    "CryptoIntelligenceDoctrineAdapter",
    "SharedInfraDoctrineAdapter",
]
