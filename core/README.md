# core/

Central orchestration and automation layer for AAC 2100.

## Key Modules

| Module | Purpose |
|--------|---------|
| `orchestrator.py` | Main sense → decide → act → reconcile control loop |
| `command_center.py` | AI avatars, real-time metrics, executive oversight |
| `aac_automation_engine.py` | Automated task scheduling and pipeline execution |
| `aac_master_launcher.py` | System bootstrap and service initialization |
| `sub_agent_spawner.py` | Parallel data collection with concurrency limits |
| `acc_advanced_state.py` | Advanced state machine implementation |

## Architecture

The orchestrator drives the main loop, delegating to strategies via `shared/strategy_loader.py` and executing trades through `TradingExecution/`.
