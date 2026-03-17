# Developer Workflow

Preferred workflow for AAC:
1. Use `.venv\Scripts\python.exe` for all Python commands.
2. Run targeted scans before editing broad areas.
3. Distinguish abstract framework code from broken implementation code.
4. Apply minimal fixes, then validate with targeted error checks or tests.
5. Record durable conclusions in `.context/` and repo memory.

When rate limited:
- Avoid subagent dependence for core repo discovery.
- Use local scanners, targeted file reads, and repo-local context files.

Useful commands:
- `.venv\Scripts\python.exe _gap_scan_v2.py`
- `.venv\Scripts\python.exe _find_stub_returns.py`
- `.venv\Scripts\python.exe -m pytest --timeout=30 -q --tb=short`
