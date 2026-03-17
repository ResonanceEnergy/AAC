# Active Workstreams

1. Gap Reduction
- Use local scanners first.
- Treat abstract base methods separately from real implementation gaps.
- Capture each verified gap batch in `07_gaps/`.

2. Monitoring Reliability
- Dashboard callbacks should degrade gracefully when caches are empty.
- Monitoring helpers should auto-return initialized singleton components.

3. Execution Determinism
- Simulation helpers should use deterministic slippage and scoring where possible.
- Avoid random behavior in operational health and risk paths.

4. Context Consolidation
- Keep durable session summaries in `.context/06_sessions/`.
- Keep repo-memory notes short and high-signal.
