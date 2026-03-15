# shared/

Common utilities, configuration, and infrastructure used across all AAC 2100 modules.

## Key Modules

| Module | Purpose |
|--------|---------|
| `config_loader.py` | YAML configuration loading with `ConfigSchema` validation |
| `audit_logger.py` | SQLite-based compliance audit trail |
| `secrets_manager.py` | Fernet-encrypted secrets at rest |
| `security_framework.py` | MFA, RBAC, and API security |
| `api_key_manager.py` | Key acquisition, rotation, validation for 100+ data feeds |
| `production_safeguards.py` | Rate limiting and circuit breakers for live trading |
| `capital_management.py` | Real-time capital tracking and regulatory compliance |
| `health_checker.py` | Component health checks and status aggregation |
| `market_data_feeds.py` | Unified market data feed abstraction |
| `paper_trading.py` | Paper trading engine for strategy validation |
| `strategy_loader.py` | Dynamic strategy discovery and loading |

## Usage

```python
from shared.config_loader import get_config
from shared.audit_logger import get_audit_logger
from shared.production_safeguards import get_production_safeguards
```
