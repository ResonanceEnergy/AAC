"""AAC Doctrine Package - Doctrine analysis and application engine."""

from .doctrine_engine import (
    DoctrineEngine,
    DoctrineApplicationService,
    Department,
    ComplianceState,
    BarrenWuffetState,
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
    StrategicDoctrineAdapter,
    FFDDoctrineAdapter,
)

from .strategic_doctrine import (
    StrategicDoctrineEngine,
    MarketTerrain,
    StrategicPosture,
    PowerLaw,
    TerrainAssessment,
    ForceAssessment,
    PowerAssessment,
    StrategicDirective,
    get_strategic_doctrine_engine,
)

__all__ = [
    # Engine
    "DoctrineEngine",
    "DoctrineApplicationService",
    "Department",
    "ComplianceState",
    "BarrenWuffetState",
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
    "StrategicDoctrineAdapter",
    # Strategic Doctrine (Art of War + 48 Laws)
    "StrategicDoctrineEngine",
    "MarketTerrain",
    "StrategicPosture",
    "PowerLaw",
    "TerrainAssessment",
    "ForceAssessment",
    "PowerAssessment",
    "StrategicDirective",
    "get_strategic_doctrine_engine",
]
