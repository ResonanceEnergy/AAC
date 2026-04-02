# Unusual Whales Integration — Full Audit Report
> Generated: 2026-03-31 | Source: Complete file-by-file code read of every UW-related module

---

## FILE INVENTORY (22 files touch UW)

### Core Modules
| # | File | Role | Health |
|---|------|------|--------|
| 1 | `integrations/unusual_whales_client.py` | Core async API client (274 lines) | BROKEN — wrong URLs, missing header, wrong fields |
| 2 | `integrations/unusual_whales_service.py` | Cached snapshot service (165 lines) | WORKS — but depends on broken client |
| 3 | `integrations/__init__.py` | Lazy import exports | OK |

### Consumer Modules
| # | File | Role | Health |
|---|------|------|--------|
| 4 | `strategies/matrix_maximizer/data_feeds.py` | DUPLICATE sync UW client (570 lines) | BROKEN — wrong URLs, wrong field names |
| 5 | `strategies/matrix_maximizer/http_health.py` | Health check endpoints | BROKEN — tests wrong URLs |
| 6 | `strategies/matrix_maximizer/intelligence.py` | Signal enrichment from feeds | BROKEN — calls broken data_feeds.py |
| 7 | `strategies/matrix_maximizer/scanner.py` | Options scanner | OK — just stores UW key, no UW calls |
| 8 | `core/daily_recommendation_engine.py` | UW intelligence layer | BROKEN — calls nonexistent method |
| 9 | `strategies/market_intelligence_model.py` | 14-source ingestor | PARTIAL — URLs correct but field names wrong |
| 10 | `monitoring/aac_master_monitoring_dashboard.py` | Dashboard display | OK — uses service layer correctly |
| 11 | `strategies/storm_lifeboat/live_feed.py` | Live feed engine | OK — uses service layer correctly |
| 12 | `aac/doctrine/ffd/ffd_engine.py` | FFD doctrine engine | OK — uses service layer correctly |
| 13 | `core/unified_component_integrator.py` | Component registry | OK — just registers class |
| 14 | `aac/doctrine/pack_registry.py` | Elite desk components list | OK — just a string reference |

### Config / Docs / Tests
| # | File | Role | Health |
|---|------|------|--------|
| 15 | `shared/config_loader.py` | Config: `unusual_whales_key` field | OK |
| 16 | `tools/validate_unusual_whales.py` | Validation script | WORKS — but will get wrong data from broken client |
| 17 | `.context/08_runbooks/unusual-whales-integration.md` | Runbook | STALE — doesn't mention field issues |
| 18 | `tests/test_unusual_whales_service.py` | Service + FFD tests | OK — but only tests service layer, not client parsing |
| 19 | `tests/test_matrix_maximizer_http.py` | HTTP health check tests | OK — mocked, so doesn't catch real API issues |
| 20 | `tests/test_monitoring.py` | Dashboard test | OK — mocked |
| 21 | `tests/test_storm_lifeboat_live_feed.py` | Live feed test | OK — mocked |
| 22 | `tests/test_api_registry.py` + `test_config_loader.py` | Key loading tests | OK |

---

## CRITICAL FINDINGS

### FLAG 1: MISSING REQUIRED HEADER (ALL API CALLS WILL FAIL)

**Source:** UW official SKILL.md at `unusualwhales.com/skill.md`
> "All requests MUST include the header: `UW-CLIENT-API-ID: 100001`"

**Reality:** ZERO files in the codebase send this header. Not `unusual_whales_client.py`, not `data_feeds.py`, not `http_health.py`. Every single API call is missing this mandatory header.

**Affected files:** ALL 5 modules that make HTTP calls to UW API.

---

### FLAG 2: HALLUCINATED / WRONG ENDPOINTS

Per UW's own hallucination blacklist and valid endpoint reference:

| File | URL Used | Correct URL | Status |
|------|----------|-------------|--------|
| `data_feeds.py` L465 | `/api/stock/flow` | `/api/option-trades/flow-alerts` | **HALLUCINATED** (on their blacklist) |
| `data_feeds.py` L500 | `/api/darkpool` (no path suffix) | `/api/darkpool/recent` | WRONG |
| `data_feeds.py` L523 | `/api/congress/trading` | `/api/congress/recent-trades` | WRONG |
| `http_health.py` L313 | `/api/stock/flow?limit=5` | `/api/option-trades/flow-alerts?limit=5` | **HALLUCINATED** |
| `http_health.py` L325 | `/api/darkpool?limit=5` | `/api/darkpool/recent?limit=5` | WRONG |
| `http_health.py` L337 | `/api/congress/trading?limit=5` | `/api/congress/recent-trades?limit=5` | WRONG |
| `unusual_whales_client.py` L180 | `/api/stock/{ticker}/info` | NOT A VALID ENDPOINT | WRONG |
| `unusual_whales_client.py` L196 | `/api/market/spike` | `/api/screener/option-contracts` | WRONG |
| `unusual_whales_client.py` L213 | `/api/market/sector-etfs` | NOT A VALID ENDPOINT | WRONG |
| `unusual_whales_client.py` L247 | `/api/stock/{ticker}/max-pain` | NOT A VALID ENDPOINT | WRONG |

**6 endpoints are completely wrong. 2 are on the official hallucination blacklist.**

---

### FLAG 3: WRONG RESPONSE FIELD NAMES

The client parses fields that may not match the actual API response schema. Per UW API examples:

| What we parse | What API likely returns | Where |
|---------------|----------------------|-------|
| `item.get('premium', 0)` | `total_premium` | client.py L115 |
| `item.get('strike_price', 0)` | `strike` or `strike_price` | client.py L113 — uncertain |
| `item.get('put_call', ...)` | `type` (per screener example) | client.py L114 |
| `item.get('volume', 0)` | `total_size` or `volume` | client.py L116 — uncertain |
| `item.get('expires_date', ...)` | unknown field name | client.py L113 |
| `item.get('sentiment', ...)` | may not exist in flow-alerts | client.py L114 |
| `d.get("notional", 0)` | `notional_value` or calculated | data_feeds.py L514 |
| `d.get("tracking_timestamp")` | `executed_at` (per examples) | data_feeds.py L515 |

**Without a real API call to inspect actual JSON, we can't be 100% sure — but multiple fields are almost certainly wrong, explaining the $0/blank results.**

---

### FLAG 4: NONEXISTENT METHOD CALL (WILL CRASH AT RUNTIME)

**File:** `core/daily_recommendation_engine.py` line ~400
```python
trades = await client.get_darkpool_trades(limit=limit)
```
**Problem:** `UnusualWhalesClient` has NO method named `get_darkpool_trades()`. The correct method is `get_dark_pool()`. This will raise `AttributeError` at runtime.

---

### FLAG 5: DUPLICATE IMPLEMENTATION (ARCHITECTURAL CONFLICT)

There are **TWO completely separate UW client implementations**:

1. **`integrations/unusual_whales_client.py`** — async, uses `aiohttp` via `APIClient` base, 13 methods
2. **`strategies/matrix_maximizer/data_feeds.py`** — sync, uses `urllib.request`, 3 methods

They use **different URLs**, **different field mappings**, **different auth patterns**, and will return **different results** (both wrong). The Matrix Maximizer modules (`intelligence.py`, `scanner.py`) use the sync duplicate, while everything else uses the async original.

**This is a maintenance nightmare. One fix won't fix both.**

---

### FLAG 6: PHANTOM FILE REFERENCE

**File:** `monitoring/aac_master_monitoring_dashboard.py` line ~1656
```python
uw_path = PROJECT_ROOT / "data" / "unusual_whales_paper_balance.json"
```
**Problem:** This file does not exist. The code silently falls back to a hardcoded `$1000.0` balance. This "UNUSUAL WHALES" strategy in the dashboard is phantom — it has no actual paper trading engine behind it.

---

### FLAG 7: STALE USER-AGENT VERSION

**File:** `unusual_whales_client.py` line 67
```python
_USER_AGENT = "AAC/2.7.0 UnusualWhalesClient"
```
**Problem:** Project is at version 3.6.0. Minor but indicates the client hasn't been maintained since v2.7.

---

### FLAG 8: `from __future__ import annotations` MISSING

**File:** `integrations/unusual_whales_client.py` — does NOT have the required `from __future__ import annotations` as the first import. Violates project coding convention (AGENTS.md rule #1).

---

## WHAT DOESN'T MAKE SENSE

1. **Why two separate UW client implementations?** The Matrix Maximizer has its own sync HTTP client in `data_feeds.py` that duplicates the async client in `integrations/`. Nobody wired them together. They use different endpoints. This makes no engineering sense.

2. **"UNUSUAL WHALES" paper strategy in dashboard** — The dashboard shows an "UNUSUAL WHALES" strategy with a $1,000 paper balance, but there is no actual strategy engine that trades based on UW signals. The `data/unusual_whales_paper_balance.json` file was never created. This is a placeholder that was never implemented.

3. **Tests all pass but nothing works** — Every test that touches UW data uses `MagicMock` or `AsyncMock`. The mocks return perfectly shaped data, so tests pass, but the real API calls would fail due to wrong URLs, missing headers, and wrong field names. There are ZERO integration tests that hit the real API.

4. **`get_flow()` vs `get_flow_alerts()`** — The client has both `get_flow()` and `get_flow_alerts()` that hit the SAME endpoint (`/api/option-trades/flow-alerts`), but `get_flow()` parses into `OptionsFlow` dataclasses while `get_flow_alerts()` returns raw dicts. Consumers use different ones inconsistently.

5. **`get_etf_flow()` just calls `get_ticker_overview()`** — The method name suggests ETF-specific flow data, but it just calls the ticker info endpoint (which doesn't exist anyway). Misleading API.

---

## WHAT IS NOT WORKABLE

1. **ANY live UW API call will fail** — Missing `UW-CLIENT-API-ID: 100001` header means all requests get rejected or return empty/error responses. This is NOT a field parsing issue — it's a total connectivity failure masked by graceful fallbacks that return empty lists.

2. **`get_ticker_overview()`** — Endpoint `/api/stock/{ticker}/info` doesn't exist. This method will always return `{}`.

3. **`get_hottest_chains()`** — Endpoint `/api/market/spike` doesn't exist. Always returns `[]`.

4. **`get_sector_etfs()`** — Endpoint `/api/market/sector-etfs` doesn't exist. Always returns `[]`.

5. **`get_max_pain()`** — Endpoint `/api/stock/{ticker}/max-pain` doesn't exist. Always returns `{}`.

6. **`get_etf_flow()`** — Calls broken `get_ticker_overview()`. Always returns `{}`.

7. **Dark pool signals in `daily_recommendation_engine.py`** — Calls `get_darkpool_trades()` which doesn't exist as a method. Will crash with `AttributeError`.

---

## IMPLEMENTATION STEPS

### Phase 1: Make the API Actually Work (CRITICAL)
**Goal:** Fix the core client so any UW API call returns real data.

1. **Add `UW-CLIENT-API-ID: 100001` header** to `_get_auth_headers()` in `unusual_whales_client.py`
2. **Add `from __future__ import annotations`** as first import
3. **Update `_USER_AGENT`** to `"AAC/3.6.0 UnusualWhalesClient"`
4. **Fix all endpoint URLs** in `unusual_whales_client.py`:
   - Remove `get_ticker_overview()` (endpoint doesn't exist) — or remap to `/api/stock/{ticker}/option-contracts`
   - Fix `get_hottest_chains()` → `/api/screener/option-contracts`
   - Remove `get_sector_etfs()` (endpoint doesn't exist)
   - Remove `get_max_pain()` (endpoint doesn't exist)
   - Fix `get_etf_flow()` to call something that exists
5. **Run `tools/validate_unusual_whales.py`** to confirm real data returns
6. **Inspect actual JSON response** from each working endpoint to map field names correctly
7. **Fix field mappings** in `get_flow()` and `get_dark_pool()` based on real response

### Phase 2: Fix Consumer Modules (HIGH)
**Goal:** All consumers call correct methods with correct field names.

8. **Fix `daily_recommendation_engine.py`** — Change `get_darkpool_trades()` → `get_dark_pool()`
9. **Fix field name references** in `daily_recommendation_engine.py` based on actual response
10. **Fix `market_intelligence_model.py`** field names based on actual response

### Phase 3: Eliminate Duplicate Client (MEDIUM)
**Goal:** Single source of truth for UW API calls.

11. **Delete the sync UW implementation** from `data_feeds.py` (lines ~447-540)
12. **Create thin sync wrapper** in `data_feeds.py` that delegates to the async client via `asyncio.run_coroutine_threadsafe()` or a sync adapter
13. **Fix `http_health.py`** endpoint URLs to match the corrected client
14. **Update `intelligence.py`** to use corrected data_feeds methods

### Phase 4: Clean Up Phantoms (LOW)
**Goal:** Remove misleading/dead code.

15. **Remove phantom UW paper strategy** from dashboard, or create the actual `data/unusual_whales_paper_balance.json` file
16. **Remove or fix broken methods** (`get_ticker_overview`, `get_sector_etfs`, `get_max_pain`, `get_etf_flow`)
17. **Consolidate `get_flow()` and `get_flow_alerts()`** — pick one pattern (structured vs raw) and standardize

### Phase 5: Add Real Tests (LOW)
**Goal:** Tests that would have caught these issues.

18. **Add integration test** in `tools/validate_unusual_whales.py` that inspects actual field names
19. **Add unit tests** that validate URL construction and header inclusion against known-good values
20. **Add field mapping tests** that assert parsed dataclass fields match expected API response shape

---

## DEPENDENCY GRAPH

```
.env (UNUSUAL_WHALES_API_KEY)
    │
    ├── shared/config_loader.py (loads key)
    │       │
    │       ├── integrations/unusual_whales_client.py (BROKEN — wrong URLs, missing header)
    │       │       │
    │       │       ├── integrations/unusual_whales_service.py (caches snapshots)
    │       │       │       │
    │       │       │       ├── monitoring/aac_master_monitoring_dashboard.py (displays)
    │       │       │       ├── strategies/storm_lifeboat/live_feed.py (feeds)
    │       │       │       └── aac/doctrine/ffd/ffd_engine.py (applies to doctrine)
    │       │       │
    │       │       ├── core/daily_recommendation_engine.py (BROKEN — wrong method name)
    │       │       ├── strategies/market_intelligence_model.py (wrong field names)
    │       │       └── tools/validate_unusual_whales.py (validator)
    │       │
    │       └── strategies/matrix_maximizer/data_feeds.py (DUPLICATE sync client — wrong URLs)
    │               │
    │               ├── strategies/matrix_maximizer/intelligence.py (enrichment)
    │               └── strategies/matrix_maximizer/scanner.py (key only, no calls)
    │
    └── strategies/matrix_maximizer/http_health.py (BROKEN — wrong URLs)
```

**Fix order:** Client (root) → Service layer (already OK) → Consumers → Duplicates → Phantoms
