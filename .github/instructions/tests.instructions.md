---
applyTo: "tests/**"
---

# Test Code — Conventions

## Run Command

```powershell
.venv\Scripts\python.exe -m pytest --timeout=30 -q --ignore=tests/security_integration_test.py --ignore=tests/test_bridge_integration.py --ignore=tests/test_ecb_api.py --ignore=tests/test_market_data_quick.py --tb=short
```

## Rules

- Mock ALL HTTP calls — `urllib.request.urlopen`, `httpx`, `aiohttp` sessions
- Use `pytest.mark.timeout(30)` for any test that could hang
- Tests must be deterministic — no random data, no real API calls
- Use `conftest.py` (root) for shared fixtures
- Always skip `test_autonomous.py` (>180s timeout, pre-existing)

## Patterns

- Mock API clients from `integrations/` using `unittest.mock.patch`
- For IBKR tests: mock `ib_insync.IB` connection
- For async tests: use `pytest-asyncio` with `@pytest.mark.asyncio`
