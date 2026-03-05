# AAC System Audit Report — 2026-03-04

**Project:** Accelerated Arbitrage Corp (AAC) — BARREN WUFFET Trading System  
**Version:** 2.7.0  
**Python Runtime:** 3.14.2 (local) | CI targets 3.9 / 3.11 / 3.12  
**Codebase Size:** 391 Python files, ~143,700 lines of code  
**License:** Proprietary  

---

## 1. EXECUTIVE SUMMARY

| Area | Rating | Key Issues |
|---|---|---|
| **Security** | 🔴 HIGH RISK | 6 critical vulnerabilities (eval, debug mode, weak crypto) |
| **Tests** | 🟡 MODERATE | 13 of 180 tests failing; 3 tests hang indefinitely |
| **Architecture** | 🟡 MODERATE | Duplicate dirs, abandoned src/ layout, shared/ overloaded |
| **Dependencies** | 🟢 GOOD | Well-managed; lock file present; optional deps separated |
| **Imports** | 🟢 GOOD | All 12 top-level packages import cleanly |
| **CI/CD** | 🟡 MODERATE | CI exists but runs on Python 3.11 (not 3.14); coverage partial |
| **Code Quality** | 🟡 MODERATE | 345+ linter warnings; many f-string logging issues |

**Overall System Health: 5.5 / 10 — Needs remediation before production**

---

## 2. TEST SUITE RESULTS

### 2.1 Summary

| Metric | Value |
|---|---|
| Tests discovered | 180 |
| Passed | 143 |
| Failed | 13 |
| Xpassed (unexpected pass) | 3 |
| Hanging (timeout) | 3+ (bridge_integration, ecb_api, others) |

### 2.2 Failing Tests (13)

| # | Test | Error | Root Cause |
|---|---|---|---|
| 1 | `test_gateway_bridge::test_intent_values` | `MessageIntent` has no `QUERY` attribute | Source code changed, test not updated |
| 2 | `test_gateway_bridge::test_message_creation` | `OpenClawMessage.__init__()` unexpected kwarg `text` | API signature changed |
| 3 | `test_gateway_bridge::test_bridge_has_classifier` | Bridge missing `classifier` attribute | Source refactored |
| 4 | `test_gateway_bridge::test_session_creation` | `OpenClawSession()` unexpected kwarg `user_id` | API signature changed |
| 5 | `test_openclaw_skills::test_skill_count` | Expected 35 skills, found 93 | Skills grew 2.7x, test has hardcoded count |
| 6 | `test_openclaw_skills::test_all_skills_have_required_fields` | Missing `category` field | Skill schema changed |
| 7 | `test_openclaw_skills::test_skill_names_list` | Count mismatch (93 vs 35) | Same as #5 |
| 8 | `test_openclaw_skills::test_categories_present` | `KeyError: 'category'` | Skill schema removed `category` |
| 9 | `test_openclaw_skills::test_get_skills_by_category` | 0 results vs expected ≥5 | Category system missing |
| 10 | `test_openclaw_skills::test_research_intel_domains` | Domain set mismatch | Domains expanded |
| 11 | `test_telegram_bot::test_bot_has_skill_handlers` | Expected 35 handlers, found 93 | Same as #5 |
| 12 | `test_telegram_bot::test_memory_store_and_recall` | `BarrenWuffetMemory` no `store` attr | API changed |
| 13 | `test_telegram_bot::test_memory_recall_not_found` | `BarrenWuffetMemory` no `recall` attr | API changed |

### 2.3 Hanging Tests

- `test_bridge_integration.py` — hangs indefinitely (likely async await without timeout)
- `test_ecb_api.py` — hangs on live HTTP call to ECB API
- Several API tests (alpha_vantage, finnhub, twelve_data, world_bank) — require live API keys

### 2.4 Missing Dev Dependency

`pytest-timeout` is listed in `pyproject.toml[dev]` and used in CI, but **not installed locally** and not in `requirements.txt`. This means:
- No test timeout protection locally
- Hanging tests block the entire suite

---

## 3. SECURITY AUDIT

### 3.1 Critical Vulnerabilities (6)

| # | Vulnerability | File | Severity |
|---|---|---|---|
| 1 | **`eval()` on subprocess output** | `setup_machine.py:129` | 🔴 CRITICAL |
| 2 | **`debug=True` default in production config** | `shared/config_loader.py:190, 248` | 🔴 CRITICAL |
| 3 | **MD5 hash for integrity** | `tools/code_quality_improvement_system.py:214` | 🔴 CRITICAL |
| 4 | **`os.system("")` call** | `scripts/health_check.py:175` | 🔴 CRITICAL |
| 5 | **Encryption key written to disk unencrypted** | `shared/api_key_manager.py:174-195` | 🔴 CRITICAL |
| 6 | **Plaintext encryption key on disk** | `shared/secrets_manager.py` | 🔴 CRITICAL |

### 3.2 High Vulnerabilities (3)

| # | Vulnerability | File | Severity |
|---|---|---|---|
| 7 | SSRF risk in HTTP clients | `integrations/coinbase_api_async.py` | 🟠 HIGH |
| 8 | Dash debug mode in production | `monitoring/aac_master_monitoring_dashboard.py:1474` | 🟠 HIGH |
| 9 | Weak password derivation | `scripts/setup_production.py` | 🟠 HIGH |

### 3.3 Medium Vulnerabilities (4)

| # | Vulnerability | File |
|---|---|---|
| 10 | Partial API key logging | `demos/worldwide_arbitrage_demo*.py` |
| 11 | Verbose error messages in exceptions | `CentralAccounting/database.py:752` |
| 12 | No constant-time credential comparison | `shared/security_framework.py` |
| 13 | Config validation leaks structure | `shared/config_loader.py` |

### 3.4 Positive Security Findings

- ✅ No hardcoded API keys in source (environment variables used)
- ✅ No committed `.env` file (properly gitignored)
- ✅ Parameterized SQL queries (no SQL injection)
- ✅ No unsafe pickle / YAML deserialization
- ✅ RBAC system implemented with default roles
- ✅ Paper trading mode enforced in test environment

---

## 4. ARCHITECTURE & CODE QUALITY

### 4.1 Structural Issues

| # | Issue | Impact |
|---|---|---|
| 1 | **Abandoned `src/aac/` layout** — `src/aac/__init__.py` is empty, `src/aac/divisions/` is empty | Confusion about canonical import paths |
| 2 | **`SharedInfrastructure/` duplicates `shared/`** — 7 files duplicated (alert_manager, audit_logger, health_checker, incident_manager, metrics_collector, security_monitor, system_monitor) | Code drift, bug fixes may not propagate |
| 3 | **`shared/` is overloaded** — 76 files mixing utilities, bridges, agents, trading logic | Difficult to navigate, unclear boundaries |
| 4 | **`strategies/` is a flat dump** — 78 files with inconsistent naming (camelCase vs snake_case) | No category organization |
| 5 | **Version mismatch** — README badge says 2.4.0, pyproject.toml says 2.7.0 | Confusing for users/contributors |
| 6 | **Duplicate pytest config** — Both `pytest.ini` and `pyproject.toml` contain pytest settings; pytest warns about ignoring pyproject.toml | Config ambiguity |

### 4.2 Code Quality Metrics (Linter Findings)

| Category | Count | Priority |
|---|---|---|
| f-string logging (`logger.info(f"...")` instead of `logger.info("...", ...)`) | ~50+ | Low |
| Catching bare `Exception` | ~15 | Medium |
| Missing file encoding in `open()` | ~5 | Low |
| Redefining `signal` from outer scope | 4 | Low |
| Missing class members (`stop_execution`, `start_execution`, `get_account_balances`) | 3 | High |
| Unable to import (`command_center`) | 1 | High |
| Missing constructor arguments (`quantum_advantage`, `cross_temporal_score`) | 2 | High |
| **Total** | **345+** |  |

### 4.3 Positive Architecture Findings

- ✅ No circular import chains detected
- ✅ All 12 top-level packages import cleanly
- ✅ Core modules use try/except import guards for optional deps
- ✅ Integrations use lazy loading pattern
- ✅ Proper entry points defined in pyproject.toml
- ✅ CI pipeline covers lint + test + coverage

---

## 5. DEPENDENCIES & PACKAGING

### 5.1 Dependency Health

| Aspect | Status |
|---|---|
| `pyproject.toml` dependencies | ✅ Well-organized with core/ml/dev/prod groups |
| `requirements.txt` | ✅ Aligned with pyproject.toml |
| `requirements-lock.txt` | ✅ Pinned lock file present |
| Optional heavy deps (torch, transformers) | ✅ Properly separated into `[ml]` extra |
| Dev tools (pytest, black, ruff, mypy) | ✅ In `[dev]` extra |

### 5.2 Dependency Risks

| # | Issue | Risk |
|---|---|---|
| 1 | `requirements.txt` includes `selenium` + `webdriver-manager` but pyproject.toml does not | Inconsistency between install methods |
| 2 | `confluent-kafka` in both `requirements.txt` (always) and pyproject.toml `[prod]` (optional) | Dual listing |
| 3 | `hiredis` in both `requirements.txt` (always) and pyproject.toml `[prod]` (optional) | Dual listing |
| 4 | `pytest-timeout` in pyproject.toml `[dev]` but missing from `requirements.txt` | Breaks local test timeouts |
| 5 | Python 3.14.2 used locally, CI tests 3.9/3.11/3.12 — **3.14 is not tested in CI** | Potential compatibility gaps |

---

## 6. CI/CD PIPELINE

### 6.1 Current Pipeline (`.github/workflows/ci.yml`)

- **Lint job:** black, isort, flake8
- **Test job:** Matrix across 3.9, 3.11, 3.12; coverage on shared/strategies/TradingExecution
- **Coverage upload:** Codecov integration

### 6.2 CI Gaps

| # | Gap | Recommendation |
|---|---|---|
| 1 | Python 3.14 not in test matrix | Add 3.14 to matrix |
| 2 | No security scanning (SAST/DAST) | Add `bandit` or `safety` check |
| 3 | No type checking in CI | Add `mypy` step |
| 4 | Coverage only on 3 of 12+ packages | Expand `--cov` to include core, integrations, etc. |
| 5 | No dependency vulnerability scanning | Add `pip-audit` or `safety check` |
| 6 | Ruff listed in dev deps but not used in CI | Replace flake8 with ruff (faster) |

---

## 7. PRIORITIZED REMEDIATION PLAN

### Immediate (P0 — This Week)

1. **Fix `eval()` vulnerability** in `setup_machine.py` → use `ast.literal_eval()`
2. **Set `debug=False` as default** in `shared/config_loader.py`
3. **Remove `os.system("")`** from `scripts/health_check.py`
4. **Fix 13 failing tests** — update test assertions to match current source APIs
5. **Install `pytest-timeout`** and add to `requirements.txt`
6. **Fix hanging `test_bridge_integration.py`** — add async timeout or mark as `@pytest.mark.integration`

### Short-Term (P1 — Within 2 Weeks)

7. **Fix key management** in `api_key_manager.py` and `secrets_manager.py`
8. **Replace MD5** with SHA-256 in all non-checksum uses
9. **Update README version badge** from 2.4.0 to 2.7.0
10. **Delete or complete the `src/` migration** — pick one canonical layout
11. **Merge `SharedInfrastructure/`** into `shared/` (or delete if truly duplicate)
12. **Fix 3 missing member errors** in `orchestrator.py` (stop_execution, start_execution, get_account_balances)
13. **Fix missing constructor args** (quantum_advantage, cross_temporal_score) in Signal class
14. **Disable Dash `debug=True`** in production dashboard

### Medium-Term (P2 — Within 1 Month)

15. **Reorganize `shared/`** into subdirectories (core, bridges, agents, trading)
16. **Organize `strategies/`** by asset class (equity, crypto, etf, index)
17. **Normalize strategy file naming** to consistent snake_case
18. **Add `bandit` security scan** to CI pipeline
19. **Add Python 3.14** to CI test matrix
20. **Expand test coverage** to core/, integrations/, trading/
21. **Add `mypy` type checking** to CI
22. **Add URL whitelist validation** for exchange API connectors
23. **Implement `hmac.compare_digest()`** for credential comparison
24. **Consolidate pytest config** — remove either `pytest.ini` or `[tool.pytest]` from pyproject.toml

### Long-Term (P3)

25. **Add dependency vulnerability scanning** (`pip-audit`) to CI
26. **Implement proper secrets management** (system keyring, Vault, or cloud KMS)
27. **Add integration test suite** with mocked external APIs
28. **Conduct penetration testing** before production deployment
29. **Implement structured logging** across all modules (replace f-string logging)
30. **Add pre-commit hooks** for automated quality checks

---

## 8. FILE ORGANIZATION MAP

```
AAC/                          391 .py files, ~143,700 lines
├── core/                     8 files  — ✅ Clean
├── shared/                   76 files — 🔴 Overloaded
├── strategies/               78 files — 🟡 Flat, needs categorization
├── trading/                  18 files — ✅ Well-organized
├── integrations/             11 files — ✅ Good lazy-loading
├── monitoring/               4 files  — ✅ Compact
├── config/                   3 files  — ✅ Clean
├── agents/                   6 files  — ✅ OK
├── CentralAccounting/        3 files  — ✅ Clean
├── BigBrainIntelligence/     4 files  — ✅ Clean
├── CryptoIntelligence/       varies   — ✅ OK
├── tests/                    25 files — 🟡 Needs timeout protection
├── SharedInfrastructure/     8 files  — 🔴 Duplicate of shared/
├── src/aac/                  empty    — 🔴 Abandoned, should delete
└── scripts/tools/demos/      varies   — 🟡 Review needed
```

---

*Report generated 2026-03-04 by system audit.*
