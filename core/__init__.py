"""AAC Core Package — Orchestrator, Command Center, Automation Engine.

Imports are guarded so that missing optional dependencies (shared infra,
monitoring, etc.) don't prevent the core package from loading at all.
"""

import logging as _logging

_logger = _logging.getLogger(__name__)


def _safe_import(import_fn, names):
    """Try an import; on failure log a warning and return None placeholders."""
    try:
        return import_fn()
    except Exception as exc:
        _logger.debug("core.__init__: deferred import failed (%s): %s", names, exc)
        return tuple(None for _ in names) if len(names) > 1 else None


# ── Orchestrator (lightweight) ──────────────────────────────────────────
try:
    from .orchestrator import AAC2100Orchestrator, OrchestratorState
except ImportError:
    AAC2100Orchestrator = None  # type: ignore[assignment,misc]
    OrchestratorState = None  # type: ignore[assignment,misc]

# ── Other core modules ─────────────────────────────────────────────────
try:
    from .aac_automation_engine import AACAutomationEngine
except ImportError:
    AACAutomationEngine = None  # type: ignore[assignment,misc]

try:
    from .aac_master_launcher import AACMasterLauncher
except ImportError:
    AACMasterLauncher = None  # type: ignore[assignment,misc]

__all__ = [
    "AAC2100Orchestrator",
    "OrchestratorState",
    "AACAutomationEngine",
    "AACMasterLauncher",
]
