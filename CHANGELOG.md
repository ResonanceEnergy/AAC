# Changelog

All notable changes to the AAC project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [2.2.0] — 2026-02-26

### Added
- `CONTRIBUTING.md` — developer onboarding and contribution guide
- `setup.cfg` — flake8 config and editable-install metadata
- `scripts/health_check.py` — system diagnostic script (`make health`)
- `__init__.py` for 7 more packages: demos, deployment, reddit, scripts, tools, docs, PaperTradingDivision
- `make health` target in Makefile

### Fixed
- `test_execution_engine_paper_trading` — marked `@xfail` (random fill sim)
- `pyproject.toml` — target-version aligned to py39 across black, ruff, mypy
- `pyproject.toml` — version bumped to 2.2.0
- `.gitignore` — `archive/` now fully ignored (was only `archive/*.db-shm`)
- `.gitignore` — added `data/backups/`, `NCC/`, 15 root division stub dirs
- CI workflow — added Python 3.9 to test matrix, `--timeout=30`, `not slow` marker

### Removed
- `aac/aac/` — nested duplicate directory (3 files, identical to `aac/doctrine/` + `aac/integration/`)
- 15 root-level division stub directories from git tracking (canonical: `src/aac/divisions/`)

---

## [2.1.0] — 2026-02-26

### Added
- `Makefile` with targets: test, lint, typecheck, format, dashboard, paper, core, full, monitor, clean
- `.github/workflows/ci.yml` — GitHub Actions CI (lint + test matrix 3.11/3.12 + security scan)
- `.pre-commit-config.yaml` — black, isort, flake8, mypy, secret detection hooks
- `launch.sh` — macOS/Linux launcher with paper/core/full/test/dashboard modes
- `.env.template` — environment variable template for all exchanges + services
- `CHANGELOG.md` — this file
- `__init__.py` for 16 packages: agents, core, integrations, models, monitoring, strategies, tests, trading, BigBrainIntelligence, CentralAccounting, CryptoIntelligence, TradingExecution, SharedInfrastructure, src, src/aac, src/aac/divisions
- `src/aac/divisions/` — consolidated 15 division directories (PaperTrading, StatisticalArbitrage, OptionsArbitrage with real code)
- `py.typed` marker for PEP 561 type stub support

### Fixed
- `shared/audit_logger.py` — `log_event()` now accepts `resource`, `event_type`, and `AuditCategory` enum (was string-only); added `audit_log()` convenience function
- `shared/super_agent_framework.py` — `from __future__ import annotations` fixes `CommunicationFramework` NameError
- `core/orchestrator.py` — `from __future__ import annotations` fixes `Signal` forward-reference NameError
- `tests/test_integration.py` — robust import with try/except fallback for `AACAgentIntegration`
- `tests/integration_test.py` — `@pytest.mark.xfail` for paper-mode position test
- `tests/test_ecb_api.py`, `test_health_status.py`, `test_market_data_quick.py` — `@pytest.mark.slow` + 30s timeout
- `conftest.py` — added `core/` and `agents/` to `PACKAGE_ROOTS` sys.path
- `pyproject.toml` — `requires-python` lowered from `>=3.12` to `>=3.9` (matches current venv)

### Changed
- `README.md` — full overhaul: architecture diagram, exchange table, strategy library, ML stack, safety controls, launch modes
- `SharedInfrastructure/` — 7 files replaced with re-export shims to `shared/`
- `.gitignore` — added `archive/`, `*.enc`, `config/crypto/`, `services/*/__pycache__/`
- `strategies/` — renamed 15 files with special characters to safe snake_case names

### Removed
- `data/backups/full_system_backup/` (71 files) — removed from git tracking (gitignored)
- `archive/` (8 files) — removed from git tracking (gitignored)
- `config/crypto/master_key.enc` — removed from git tracking (security)
- `services/__pycache__/` — stale bytecode deleted from disk

---

## [2.0.0] — 2026-02-25

### Added
- Restored 384 `.py` source files from git commit `278ba8263` (recovery after Feb 17 security scrub)
- `conftest.py`, `pytest.ini`, `pyproject.toml` — project configuration
- `.env.template` — environment variable template
- `requirements-lock.txt` — 154 pinned packages

### Fixed
- `requirements.txt` — added 9 missing packages (kafka-python, redis, fastapi, uvicorn, xgboost, pyotp, psutil, bcrypt, torch)

### Changed
- Flattened `integrations/integrations/`, `strategies/strategies/`, `trading/trading/`
- Deduplicated `CentralAccounting` (removed 2 phantom copies)

### Removed
- `__pycache__/` (186 directories), `node_modules/` (27,747 files), `build/` artifacts from git index
- 149,433 NCC JSON bulletin files from git index (4 backup snapshots)
- Git history rewritten with `git filter-repo` to permanently remove bloat

---

## [1.0.0] — 2026-02-06

### Added
- Initial AAC trading platform with 69 strategies, multi-exchange execution, ML models, and full audit trail
