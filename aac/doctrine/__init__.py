"""AAC Doctrine Package - Doctrine analysis and application engine."""

from .doctrine_engine import (
    DOCTRINE_PACKS,
    ActionType,
    BarrenWuffetState,
    ComplianceState,
    Department,
    DoctrineApplicationService,
    DoctrineComplianceReport,
    DoctrineEngine,
    DoctrineViolation,
)
from .doctrine_integration import (
    BigBrainDoctrineAdapter,
    CentralAccountingDoctrineAdapter,
    CryptoIntelligenceDoctrineAdapter,
    DoctrineOrchestrator,
    FFDDoctrineAdapter,
    SharedInfraDoctrineAdapter,
    StrategicDoctrineAdapter,
    TradingExecutionDoctrineAdapter,
)
from .strategic_doctrine import (
    ForceAssessment,
    MarketTerrain,
    PowerAssessment,
    PowerLaw,
    StrategicDirective,
    StrategicDoctrineEngine,
    StrategicPosture,
    TerrainAssessment,
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
