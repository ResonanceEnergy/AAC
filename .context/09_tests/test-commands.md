# Test Commands

Primary command:
- `.venv\Scripts\python.exe -m pytest --timeout=30 -q --ignore=tests/security_integration_test.py --ignore=tests/test_bridge_integration.py --ignore=tests/test_ecb_api.py --ignore=tests/test_market_data_quick.py --tb=short`

Targeted validation for this session:
- Run `get_errors` on edited files first.
- Run targeted pytest on monitoring, crypto intelligence, and strategy execution areas if tests exist.

Notes:
- Some legacy tests are slow or environment-dependent.
- Local HTTP tests should be mocked where possible.
