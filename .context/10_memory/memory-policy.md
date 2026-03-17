# Memory Policy

AAC uses three layers of memory:
- Chat/session memory for temporary reasoning
- Repo memory for concise long-lived facts
- `.context/` for durable, human-readable project context

Rules:
- Put stable operational context in `.context/`.
- Put concise high-signal repo facts in repo memory.
- Do not treat scanner output as final without file verification.
- Prefer updating existing context files over creating throwaway notes.
