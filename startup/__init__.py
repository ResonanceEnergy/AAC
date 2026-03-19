"""
startup — Consolidated AAC startup package.

One package to boot the entire BARREN WUFFET / AZ SUPREME system.
Replaces the scattered startup logic previously spread across:
  - start_autonomous.bat (gateways + paper)
  - aac_automation.py (git + dashboard)
  - full_activation.py (all divisions, all agents)
  - automate.py (pre-flight validation)
  - pipeline_runner.py (single Fib pipeline cycle)

Everything is still accessible through ``python launch.py <mode>``.

New unified modes added by this package:
  - ``matrix``  — Matrix Monitor dashboard (terminal / web / dash)
  - ``all``     — Full startup: gateways → pre-flight → paper engine → matrix monitor
  - ``preflight`` — Quick pre-flight validation (env, imports, config)
"""

__all__ = [
    "gateways",
    "preflight",
    "matrix_monitor",
    "phases",
]
