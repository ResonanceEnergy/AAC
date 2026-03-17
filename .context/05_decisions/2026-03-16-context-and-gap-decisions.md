# 2026-03-16 Decisions

1. Subagent rate limits are treated as an external platform constraint, not a repo defect.
2. Local scanning is the fallback and default for gap discovery during rate-limited periods.
3. `.context/` is introduced as the durable repo-local context layer to reduce fragmented work across sessions.
4. Monitoring singleton accessors should not raise `NotImplementedError` when the service constructor already initializes dependencies.
5. Placeholder UI paths should return real empty-state content instead of "coming soon" stubs.
6. Deterministic models are preferred over random scoring in venue-health and simulated execution paths when the code is part of operational flow.
