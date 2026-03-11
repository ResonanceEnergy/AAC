# AAC Comprehensive Gap Analysis Report

**Generated:** 2025-01-XX  
**Scope:** Tests, Configuration, Core Infrastructure, Integration Layer, Trading Execution, Data Pipeline  
**Codebase:** `c:\Users\gripa\OneDrive\Desktop\AAC_fresh`

---

## Summary

| Area | Critical | High | Medium | Low | Total |
|------|----------|------|--------|-----|-------|
| 1. Tests | 5 | 6 | 4 | 2 | 17 |
| 2. Configuration | 3 | 3 | 2 | 1 | 9 |
| 3. Core Infrastructure | 1 | 3 | 3 | 1 | 8 |
| 4. Integration Layer | 1 | 3 | 2 | 1 | 7 |
| 5. Trading Execution | 2 | 3 | 3 | 1 | 9 |
| 6. Data Pipeline | 2 | 2 | 1 | 1 | 6 |
| **Total** | **14** | **20** | **15** | **7** | **56** |

---

## 1. Tests Directory

### CRITICAL

**GAP-T01: `tests/test.py` — Empty test file**  
- **File:** `tests/test.py`, Line 1  
- **Description:** Entire file is `print('test')`. Contains zero `test_` functions, zero assertions, zero imports of the system under test. Pytest discovers this file but finds nothing to run. Creates false confidence in test counts.

**GAP-T02: `tests/test_live.py` — No actual tests**  
- **File:** `tests/test_live.py`, Lines 1–8  
- **Description:** File contains only `async def main(): print("Hello")` and `asyncio.run(main())`. No test functions, no assertions, no imports of any system module. The name implies live integration testing but delivers nothing.

**GAP-T03: `tests/test_imports.py` — Not pytest-compatible**  
- **File:** `tests/test_imports.py`, Lines 1–90+  
- **Description:** Uses `print()` statements for output and try/except blocks to check importability. No `test_` prefixed functions discoverable by pytest. No assertions. This is a manual script, not a test suite. Pytest will discover the file but find zero tests.

**GAP-T04: `tests/test_code_quality.py` — No real assertions; `pass` in except blocks**  
- **File:** `tests/test_code_quality.py`, Lines 21, 24  
- **Description:** Uses `try/except: pass` pattern, silently swallowing import failures. Uses `print()` instead of assertions. Not a proper pytest test — will show as "collected 0 items" or pass vacuously.

**GAP-T05: Zero test coverage for critical production modules**  
- **Description:** The following production-critical modules have NO corresponding test files:
  - `core/orchestrator.py` (1,727 lines — the main orchestrator)
  - `core/aac_master_launcher.py` (655 lines — system launch sequence)
  - `shared/production_safeguards.py` (275 lines — circuit breakers, rate limiters)
  - `shared/bridge_orchestrator.py` (560 lines — inter-department messaging)
  - `TradingExecution/execution_engine.py` (2,092 lines — order execution)
  - `TradingExecution/risk_manager.py` (230 lines — pre-trade risk checks)
  - `TradingExecution/order_manager.py` (191 lines — order lifecycle)
  - `TradingExecution/exchange_connectors/binance_connector.py` (613 lines)
  - `TradingExecution/exchange_connectors/coinbase_connector.py` (473 lines)
  - `TradingExecution/exchange_connectors/kraken_connector.py` (488 lines)
  - `CentralAccounting/financial_analysis_engine.py`
  - `CentralAccounting/database.py`

### HIGH

**GAP-T06: `tests/simple_test.py` — Not pytest-compatible**  
- **File:** `tests/simple_test.py`, Lines 1–30+  
- **Description:** Uses `asyncio.run()` and `print()` statements. No `test_` functions, no assertions. Manual script structure — not discoverable by pytest.

**GAP-T07: `tests/test_barren_wuffet.py` — Not pytest-compatible**  
- **File:** `tests/test_barren_wuffet.py`, Lines 1–50+  
- **Description:** Uses `asyncio.run()` entry point and print-based validation. No pytest assertions. Functions not named with `test_` prefix in a pytest-discoverable way.

**GAP-T08: `tests/test_integration.py` — Not pytest-compatible**  
- **File:** `tests/test_integration.py`, Lines 45–46  
- **Description:** Uses `asyncio.run(test_integration())` under `if __name__` guard. The test function is not in a class or properly decorated for pytest auto-discovery. Print-based result reporting.

**GAP-T09: `tests/test_bridge_integration.py` — Not pytest-compatible**  
- **File:** `tests/test_bridge_integration.py`  
- **Description:** Manual asyncio flow with print-based validation. Not structured for pytest collection.

**GAP-T10: `tests/test_partial_fill_models.py` — Script, not pytest test**  
- **File:** `tests/test_partial_fill_models.py`, Lines 170–177  
- **Description:** Uses `asyncio.run()` under `if __name__`. Only `print()` statements, zero assertions. Tests critical execution engine partial fill models but validates nothing programmatically.

**GAP-T11: `tests/test_paper_trading.py` — Hybrid script/test (not auto-discoverable)**  
- **File:** `tests/test_paper_trading.py`, Lines 310–311  
- **Description:** Contains assertions (good), but wrapped in `asyncio.run(main())` under `if __name__`. Functions like `test_basic_market_orders()` are not decorated with `@pytest.mark.asyncio`, so pytest will NOT auto-discover them. Must be run manually.

### MEDIUM

**GAP-T12: `tests/test_live_trading_safeguards.py` — Hybrid script/test**  
- **File:** `tests/test_live_trading_safeguards.py`, Lines 394–395  
- **Description:** Contains actual assertions (good), but uses `asyncio.run(main())` pattern. Functions named `test_*` but not decorated for pytest async. Similar hybrid issue as test_paper_trading.py. Has 395 lines of substantive assertions but won't be collected by `pytest`.

**GAP-T13: `tests/test_market_data_integration.py` — Script-only execution**  
- **File:** `tests/test_market_data_integration.py`, Lines 440–448  
- **Description:** Uses `asyncio.run(run_market_data_tests())` entry. Not pytest-compatible structure.

**GAP-T14: `tests/test_market_data_quick.py` — Script-only execution**  
- **File:** `tests/test_market_data_quick.py`, Lines 56–57  
- **Description:** Uses `asyncio.run(test_market_data())` entry. Not pytest-compatible.

**GAP-T15: `conftest.py` — Missing critical fixtures**  
- **File:** `conftest.py`, Lines 1–60  
- **Description:** Root conftest provides `project_root` and `paper_trading_env` fixtures but is missing:
  - Mock exchange connector fixture
  - Mock trading engine fixture  
  - Test database (SQLite in-memory) fixture
  - Mock API client fixture
  - Sample Order/Position factory fixtures
  - Event loop fixture configuration for pytest-asyncio

### LOW

**GAP-T16: `tests/test_health_status.py` — Print-only, no assertions**  
- **File:** `tests/test_health_status.py`  
- **Description:** Has proper pytest markers but relies on print output rather than assert statements. Tests pass by not throwing exceptions rather than by validating behavior.

**GAP-T17: API key tests run against live services**  
- **Files:** `tests/test_alpha_vantage.py`, `tests/test_finnhub_key.py`, `tests/test_polygon_key.py`, `tests/test_twelve_data.py`, `tests/test_ecb_api.py`, `tests/test_premium_apis.py`  
- **Description:** These tests hit live external APIs. No mocking, no retry, no rate-limit handling. Will fail in CI/CD without credentials. Should be marked `@pytest.mark.live` or use VCR/responses cassettes.

---

## 2. Configuration Files

### CRITICAL

**GAP-C01: Version mismatch between `pyproject.toml` and `setup.cfg`**  
- **Files:** `pyproject.toml` Line 7 (`version = "2.7.0"`), `setup.cfg` Line 3 (`version = 2.2.0`)  
- **Description:** Two build configuration files declare different versions. This causes ambiguity about the actual package version. `pip install -e .` will use pyproject.toml (2.7.0) but any tool reading setup.cfg gets 2.2.0.

**GAP-C02: `websockets` package used but not declared as dependency**  
- **Files affected:** `integrations/market_data_aggregator.py` L13, `shared/market_data_connector.py` L19, `trading/live_trading_infrastructure.py` L24, `integrations/openclaw_gateway_bridge.py` L281  
- **Missing from:** `pyproject.toml`, `requirements.txt`  
- **Description:** Four production files `import websockets` at module level, but it is not listed in any dependency specification. Fresh installs will crash with `ModuleNotFoundError`.

**GAP-C03: `pytest-asyncio` version conflict between lock file and spec**  
- **Files:** `requirements-lock.txt` (pins `pytest-asyncio==1.2.0`), `pyproject.toml` (requires `>=0.21.0`), `setup.cfg` (requires `>=0.21.0`)  
- **Description:** The lock file pins an ancient version (1.2.0, from 2021) that predates the `asyncio_mode = auto` feature used in `pytest.ini`. Running `pip install -r requirements-lock.txt` will install a broken testing setup. Version 0.21.0+ is needed for auto mode.

### HIGH

**GAP-C04: `selenium` and `webdriver-manager` in `requirements.txt` but not in `pyproject.toml`**  
- **File:** `requirements.txt` (has `selenium>=4.15.0`, `webdriver-manager>=4.0.0`)  
- **File:** `pyproject.toml` (missing both)  
- **Description:** Dependency split: pyproject.toml is the canonical build file but is missing browser automation deps that requirements.txt declares. Installing via `pip install .` won't get selenium.

**GAP-C05: `requirements-lock.txt` contains undeclared packages**  
- **File:** `requirements-lock.txt`  
- **Description:** Lock file includes `beautifulsoup4`, `nltk`, `peewee`, `Flask`, `gunicorn` and others that appear in neither `requirements.txt` nor `pyproject.toml`. The lock file is out of sync with the actual dependency declarations and may be from a different environment snapshot.

**GAP-C06: All dependencies use `>=` with no upper bound**  
- **File:** `requirements.txt`, `pyproject.toml`  
- **Description:** Every dependency uses `>=` without upper bounds (e.g., `ccxt>=4.0.0`, `fastapi>=0.100.0`). A breaking change in any dependency will silently break the system. Should use `>=X.Y,<Z.0` ranges or a proper lock file.

### MEDIUM

**GAP-C07: Duplicate dev dependencies in `setup.cfg`**  
- **File:** `setup.cfg`, Lines 10–18  
- **Description:** `setup.cfg [options.extras_require] dev` duplicates the same dev dependencies already in `pyproject.toml [project.optional-dependencies] dev`. Two places to maintain the same list — they will drift.

**GAP-C08: `pytest.ini` `asyncio_mode = auto` requires pytest-asyncio ≥0.18**  
- **File:** `pytest.ini`, Line 4  
- **Description:** The `asyncio_mode = auto` setting was introduced in pytest-asyncio 0.18. Combined with GAP-C03 (lock file pins 1.2.0), this configuration is broken for anyone using the lock file.

### LOW

**GAP-C09: No `.python-version` or `runtime.txt`**  
- **Description:** `pyproject.toml` specifies `requires-python = ">=3.9"` but there's no `.python-version` file for pyenv or `runtime.txt` for deployment platforms. CI/CD and deployment platforms must guess the Python version.

---

## 3. Core Infrastructure

### CRITICAL

**GAP-I01: `production_safeguards.py` — Circuit breaker not wired into `safe_call()`**  
- **File:** `shared/production_safeguards.py`, Lines 152–162  
- **Description:** `ExchangeSafeguards.safe_call()` awaits the rate limiter token but **never checks or updates the circuit breaker**. The circuit breaker (`CircuitBreaker`) is instantiated and stored but its `call()` or state-check methods are never invoked in the actual call path. The `try/yield/except` block just re-raises without recording failures. This means:
  - Circuit will never trip to OPEN state
  - Failed calls won't increment the failure counter
  - The circuit breaker is effectively dead code

### HIGH

**GAP-I02: `orchestrator.py` — Duplicate GTA import block**  
- **File:** `core/orchestrator.py`, Lines 89–94 and Lines 96–101  
- **Description:** The Global Talent Acquisition import block is duplicated verbatim:
  ```python
  # Import Global Talent Acquisition integration  (appears twice)
  try:
      from shared.global_talent_integration import get_gta_integration, initialize_gta_integration
      GTA_AVAILABLE = True
  except ImportError:
      GTA_AVAILABLE = False
  ```
  Harmless but indicates copy-paste error and sloppy code. The duplicate overwrites `GTA_AVAILABLE` with the same value.

**GAP-I03: `bridge_orchestrator.py` — `_process_generic_message()` is a stub**  
- **File:** `shared/bridge_orchestrator.py`, Lines 463–473  
- **Description:** The generic message handler only logs the message. Contains an explicit TODO comment: *"In a full implementation, this would: Validate message content, Apply business logic rules, Forward to appropriate internal systems, Generate responses if needed."* Any message without a specific handler is silently dropped after logging.

**GAP-I04: `aac_master_launcher.py` — `_shutdown_monitoring()` is empty**  
- **File:** `core/aac_master_launcher.py`, Lines 395–397  
- **Description:** The monitoring shutdown method contains only a log message — no actual shutdown logic:
  ```python
  async def _shutdown_monitoring(self):
      """Shutdown monitoring systems"""
      logger.info("Shutting down monitoring systems")
  ```
  Called from the main `_shutdown_systems()` method at line 384, so monitoring resources (if any) are never cleaned up.

### MEDIUM

**GAP-I05: `orchestrator.py` — No graceful handling of division initialization failures**  
- **File:** `core/orchestrator.py`, Lines 500–790  
- **Description:** Each division (15+ total) is initialized in sequence. If any `await get_*_division()` call raises a non-ImportError exception (e.g., timeout, config error), the entire `initialize()` method fails via the outer `try/except`. There's no partial-initialization recovery or division-level error isolation.

**GAP-I06: `orchestrator.py` — Log messages claim divisions are "active" before verification**  
- **File:** `core/orchestrator.py`, Lines 786–790  
- **Description:** After the initialization loop, log messages unconditionally state all divisions are active: *"Legal Division active | Insurance Division active | Banking Division active"* — regardless of which `*_AVAILABLE` flags are actually True. Misleading in production logs.

**GAP-I07: `sub_agent_spawner.py` — No health check or heartbeat for spawned tasks**  
- **File:** `core/sub_agent_spawner.py`  
- **Description:** Tasks are `run_all()`-ed with semaphore concurrency and retries, but there's no mechanism to check on long-running tasks, no heartbeat, and no way to cancel individual tasks after submission. The `BatchResult` aggregation only happens after all tasks complete or timeout.

### LOW

**GAP-I08: `orchestrator.py` — `_checkpoint_path` not configurable**  
- **File:** `core/orchestrator.py`  
- **Description:** Checkpoint path is hardcoded via `get_project_path()`. Not configurable through constructor or config. Makes testing checkpoint save/load difficult without filesystem side effects.

---

## 4. Integration Layer

### CRITICAL

**GAP-N01: `coinbase_api_async.py` — Missing `import json`**  
- **File:** `integrations/coinbase_api_async.py`, Line 49  
- **Description:** The `request()` method calls `json.dumps(data)` at line 49 but `json` is never imported. File imports: `os, time, hmac, hashlib, base64, aiohttp`. Any POST request with a body will crash with `NameError: name 'json' is not defined`. This is a production-breaking bug.

### HIGH

**GAP-N02: `coinbase_api_async.py` — No error handling, retry, or rate limiting**  
- **File:** `integrations/coinbase_api_async.py`, Lines 47–52  
- **Description:** The `request()` method does `resp.raise_for_status()` with no retry logic, no exponential backoff, no rate-limit header parsing, and no connection timeout. Coinbase's API has strict rate limits (3-6 req/sec). Compare with `api_integration_hub.py` which has comprehensive retry/rate-limit logic. A single 429/503 response will crash the caller.

**GAP-N03: `market_data_aggregator.py` — Exchanges initialized with null API keys**  
- **File:** `integrations/market_data_aggregator.py`, Line ~50  
- **Description:** All 9 exchanges (binance, coinbase, kraken, etc.) are initialized with `apiKey=None, secret=None` in the constructor. While some public endpoints work without keys, authenticated endpoints (order placement, balance queries) will fail silently or raise cryptic errors. No warning is logged about unauthenticated mode.

**GAP-N04: `market_data_aggregator.py` — Uses `websockets` (undeclared dependency)**  
- **File:** `integrations/market_data_aggregator.py`, Line 13  
- **Description:** `import websockets` at module level with no try/except guard. Combined with GAP-C02 (not in any dependency file), this module will crash on import in a clean environment.

### MEDIUM

**GAP-N05: `openclaw_gateway_bridge.py` — `_handle_webhook()` has no authentication**  
- **File:** `integrations/openclaw_gateway_bridge.py`, Line 575  
- **Description:** The webhook handler accepts and processes incoming data with no signature verification, no API key validation, and no IP allowlisting. Any external caller can inject webhook payloads.

**GAP-N06: `api_integration_hub.py` — API keys stored in plain text in config**  
- **File:** `integrations/api_integration_hub.py`  
- **Description:** API client reads keys from environment variables or config dict, but there's no validation that secrets aren't accidentally logged or serialized. The `to_dict()` or `__repr__` methods could expose credentials.

### LOW

**GAP-N07: No integration health dashboard endpoint**  
- **Description:** The integration layer has individual health checks per connector but no unified health endpoint that reports the status of all integrations simultaneously. Monitoring must poll each connector individually.

---

## 5. Trading Execution

### CRITICAL

**GAP-E01: Dual `Order` dataclass definitions with incompatible field names**  
- **Files:**
  - `TradingExecution/execution_engine.py`, Line 77: `average_fill_price: float = 0.0`
  - `TradingExecution/trading_engine.py`, Line 68: `average_price: float = 0.0`
- **Description:** Two different `Order` dataclasses exist in the TradingExecution package with different field names for the same concept. `order_manager.py` (Line 73, 110) uses `average_price` (matching `trading_engine.py`). The exchange connectors (binance L580, coinbase L446, kraken L461) also use `average_price`. But `execution_engine.py` uses `average_fill_price`. Any code path that passes an Order between these modules will get `AttributeError` or silent 0.0 defaults. Additional differences:
  - `execution_engine.Order` has `stop_price`, `remaining_quantity`, `filled_at` — absent from `trading_engine.Order`
  - `trading_engine.Order` field order differs (`exchange` before `symbol`)

**GAP-E02: SQL column name mismatch in order persistence**  
- **File:** `TradingExecution/execution_engine.py`, Lines 870–886  
- **Description:** The `_persist_order()` SQL INSERT uses column name `average_price` in the schema, but passes `order.average_fill_price` as the value. The `getattr(order, 'average_fill_price', order.price)` fallback at line 885 suggests awareness of the mismatch but doesn't fix the root cause. If the SQLite table was created with `average_price` column, the column/value alignment in the INSERT may be positionally correct but is semantically confusing and fragile.

### HIGH

**GAP-E03: Stop-loss and OCO orders not implemented in any connector**  
- **File:** `TradingExecution/exchange_connectors/base_connector.py`, Lines 350–390  
- **Description:** `create_stop_loss_order()` and `create_oco_order()` both `raise NotImplementedError("Subclass must implement")` in the base class. None of the three concrete connectors (Binance, Coinbase, Kraken) override these methods. Any strategy using stop-loss orders will crash at runtime.

**GAP-E04: `risk_manager.py` — Daily loss tracking resets on restart**  
- **File:** `TradingExecution/risk_manager.py`  
- **Description:** `RiskManager` tracks daily loss in memory (`self.daily_pnl`). There is no persistence — restarting the process resets the daily loss counter to 0. A trader could exceed daily loss limits by restarting the system mid-day.

**GAP-E05: Dual `Position` dataclass definitions with incompatible fields**  
- **Files:**
  - `TradingExecution/execution_engine.py`, Line 113: Position with `stop_loss`, `take_profit`, `status`, `closed_at`
  - `TradingExecution/trading_engine.py`, Line 76: Position with `unrealized_pnl` as stored field (not property), no `stop_loss`/`take_profit`
- **Description:** Same structural issue as GAP-E01 but for Position objects. The execution_engine Position has stop-loss/take-profit tracking; the trading_engine Position does not. Passing positions between systems will lose critical risk parameters.

### MEDIUM

**GAP-E06: No partial fill reconciliation across engines**  
- **File:** `TradingExecution/execution_engine.py`, `TradingExecution/trading_engine.py`  
- **Description:** Both engines handle partial fills internally but there's no reconciliation mechanism if both are active simultaneously. The `execution_engine` updates `filled_quantity` and `remaining_quantity` while `trading_engine` tracks `filled_quantity` independently. No shared state or event bus connects them.

**GAP-E07: `order_manager.py` — JSON persistence with no file locking**  
- **File:** `TradingExecution/order_manager.py`, Lines 50–80  
- **Description:** Orders are persisted to JSON files. No file locking mechanism is used. Concurrent writes from multiple coroutines could corrupt the order file. The save/load cycle is not atomic.

**GAP-E08: Kraken connector — No testnet support with safety warning**  
- **File:** `TradingExecution/exchange_connectors/kraken_connector.py`  
- **Description:** Connector includes comments warning that Kraken has no public testnet. Unlike Binance and Coinbase connectors, there's no sandbox mode. Paper trading must be simulated at a higher layer, but there's no enforcement preventing accidental live order placement.

### LOW

**GAP-E09: Exchange connectors don't share a common order ID format**  
- **Description:** Each connector returns exchange-native order IDs. No mapping layer translates between internal AAC order IDs and exchange order IDs, making cross-exchange order tracking and reconciliation difficult.

---

## 6. Data Pipeline

### CRITICAL

**GAP-D01: No data pipeline exists**  
- **Directory:** `data/`  
- **Description:** The `data/` directory contains only:
  - `__init__.py` (defines `DATA_DIR` and `PAPER_TRADING_DIR` paths)
  - `paper_trading/polymarket_schema.py` (SQLite schema for prediction markets)
  - `paper_trading/paper_account_1.json` (sample account data)
  
  There is no ETL pipeline, no data ingestion framework, no data validation layer, no data transformation code, no historical data storage, no data versioning. The system generates market data signals in the integration layer but has no structured pipeline to store, transform, or replay them.

**GAP-D02: No historical data storage or replay capability**  
- **Description:** The orchestrator processes real-time signals but there's no mechanism to:
  - Store historical market data for backtesting
  - Replay historical signals through strategies
  - Archive trade history beyond JSON files
  - Generate performance reports from historical data
  
  The `execution_engine.py` has SQLite persistence for orders/positions but no time-series data store for market data.

### HIGH

**GAP-D03: `polymarket_schema.py` — Schema defined but no migration system**  
- **File:** `data/paper_trading/polymarket_schema.py`, Lines 1–322  
- **Description:** Comprehensive SQLite schema with 8 tables defined as SQL string constants. No migration system (Alembic, custom versioning, etc.). Schema changes require manual `DROP TABLE`/`CREATE TABLE` cycles, losing all existing data.

**GAP-D04: No data validation layer**  
- **Description:** Market data flows from integration connectors directly to the orchestrator and execution engine with no intermediate validation:
  - No schema validation for incoming market ticks
  - No range checking (e.g., negative prices, zero volumes)
  - No deduplication of duplicate ticks
  - No gap detection in time series data
  - No staleness detection for cached prices

### MEDIUM

**GAP-D05: `paper_account_1.json` — No schema enforcement**  
- **File:** `data/paper_trading/paper_account_1.json`  
- **Description:** Paper trading account data stored as unvalidated JSON. No Pydantic model or JSON Schema validation. Corrupt or malformed account files will cause runtime errors rather than clean validation failures.

### LOW

**GAP-D06: No data export or reporting pipeline**  
- **Description:** The system can export final metrics (orchestrator `_export_final_metrics()`) but has no structured reporting pipeline for:
  - Daily P&L reports
  - Trade execution quality reports
  - Strategy performance attribution
  - Regulatory reporting (if applicable)

---

## Cross-Cutting Concerns

### HIGH

**GAP-X01: No CI/CD configuration**  
- **Description:** No `.github/workflows/`, `Jenkinsfile`, `azure-pipelines.yml`, or any CI/CD configuration found. Tests (even the working ones) are never run automatically. Code quality checks (black, isort, flake8, ruff, mypy — all declared as dev deps) have no automated enforcement.

**GAP-X02: No type checking enforcement**  
- **File:** `core/py.typed` exists (marker file for PEP 561)  
- **Description:** The `py.typed` marker declares this package supports type checking, but `mypy` is only a dev dependency with no configuration in `pyproject.toml` or `setup.cfg`. No `mypy.ini` or `[tool.mypy]` section exists. The type annotations throughout the codebase are never validated.

### MEDIUM

**GAP-X03: `__pycache__` directories committed to repository**  
- **Description:** Multiple `__pycache__/` directories exist in the workspace tree (root, aac/, agents/, etc.). These should be in `.gitignore` and not tracked. Indicates either missing `.gitignore` entries or lack of `git clean`.

---

## Recommendations (Priority Order)

1. **Immediate (Critical):** Fix `coinbase_api_async.py` missing `json` import (GAP-N01) — production crash on any POST request
2. **Immediate (Critical):** Wire circuit breaker into `safe_call()` (GAP-I01) — safety system is non-functional
3. **Immediate (Critical):** Add `websockets` to `pyproject.toml` dependencies (GAP-C02) — 4 modules crash on import
4. **Immediate (Critical):** Resolve version mismatch pyproject.toml vs setup.cfg (GAP-C01)
5. **Immediate (Critical):** Unify `Order`/`Position` dataclass definitions (GAP-E01, GAP-E05) — use single source of truth
6. **Short-term (High):** Convert script-style tests to pytest-compatible format (GAP-T06–T14)
7. **Short-term (High):** Add test coverage for core modules (GAP-T05)
8. **Short-term (High):** Fix `pytest-asyncio` lock file version (GAP-C03)
9. **Short-term (High):** Implement stop-loss orders in at least one connector (GAP-E03)
10. **Medium-term:** Build data pipeline infrastructure (GAP-D01, D02)
11. **Medium-term:** Add CI/CD pipeline (GAP-X01)
12. **Medium-term:** Add data validation layer (GAP-D04)
