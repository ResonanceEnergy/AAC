# Doctrine + FFD Integration Blueprint

Date: 2026-03-16
Scope: Preserve original doctrine behavior, elevate FFD into the doctrine system, and make the doctrine architecture extensible beyond Pack 11.

## What Exists Today

### Original doctrine core
- `aac/doctrine/doctrine_engine.py` is the real compliance engine.
- It owns threshold evaluation, violation creation, BARREN WUFFET state changes, and automated action routing.
- The live built-in pack registry only defines Packs 1-8.
- The engine expects an external YAML file at `config/doctrine_packs.yaml`, but that file does not exist in this repo, so the system falls back to the hardcoded registry.

### Original doctrine orchestration
- `aac/doctrine/doctrine_integration.py` collects metrics from department adapters and runs compliance checks through `DoctrineApplicationService`.
- It already instantiates:
  - departmental adapters for Packs 1-8,
  - `StrategicDoctrineAdapter` for Packs 9-10,
  - `FFDDoctrineAdapter` for Pack 11.
- It therefore *collects* strategic and FFD metrics, but the core engine has no Pack 9, 10, or 11 definitions to evaluate against.

### Strategic doctrine
- `aac/doctrine/strategic_doctrine.py` is a separate strategy overlay engine.
- It exposes metrics for Packs 9 and 10, but those packs are not first-class in `DOCTRINE_PACKS`.

### Future Financial Doctrine
- `aac/doctrine/ffd/ffd_engine.py` is a separate macro / monetary-transition intelligence engine.
- It contains a rich Pack 11 metric model and a standalone `FFD_DOCTRINE_PACK` definition.
- `FFDDoctrineAdapter` exposes FFD metrics to the orchestration layer, but the core doctrine engine does not register or enforce Pack 11.

## Critical Findings

1. Packs 9-11 are not first-class doctrine packs in the live compliance engine.
2. The doctrine engine's source of truth is hardcoded, not YAML-configured, because `config/doctrine_packs.yaml` is missing.
3. `DoctrineOrchestrator` claims broader doctrine coverage than the core engine actually enforces.
4. FFD currently behaves as an attached intelligence source, not as a native doctrine pack.
5. The future-proofing problem is architectural, not just additive: new doctrine domains cannot be integrated cleanly while pack registration, metric definitions, and adapters are split across separate systems.

## Integration Principles

1. Keep Packs 1-8 behavior unchanged.
2. Preserve Strategic Doctrine as Packs 9-10, not as an informal overlay.
3. Preserve FFD as its own engine, but register it as Pack 11 inside the same compliance model as the rest of doctrine.
4. Separate doctrine pack registration from domain engine implementation.
5. Make Pack 12+ additions possible without editing core engine logic each time.

## Target Architecture

### 1. Canonical doctrine registry
Create one canonical registry layer that the doctrine engine always consumes.

Responsibilities:
- hold pack definitions for Packs 1-11,
- expose pack metadata, thresholds, failure modes, and BARREN WUFFET hooks,
- support both built-in defaults and optional external config override.

Preferred shape:
- new module such as `aac/doctrine/pack_registry.py`
- `BUILTIN_DOCTRINE_PACKS` becomes the baseline source of truth
- `DoctrineEngine` reads from registry instead of embedding all definitions directly

### 2. Provider-based doctrine domains
Each doctrine domain should have two pieces:
- a domain engine that computes metrics and state
- a provider/adapter that exposes doctrine pack definitions plus current metrics

This means:
- Packs 1-8 remain backed by department adapters
- Packs 9-10 remain backed by `StrategicDoctrineEngine`
- Pack 11 remains backed by `FFDEngine`

But all packs register through the same registry and are evaluated by the same engine.

### 3. Native Packs 9-11 in the engine
Move Strategic Doctrine and FFD from “extra metrics collected by orchestrator” to “registered doctrine packs recognized by `DoctrineEngine`”.

Result:
- `_find_metric_definition()` can resolve Pack 9-11 metrics
- compliance score includes Pack 9-11
- doctrine summaries and violations become accurate
- BARREN WUFFET state changes can be driven by FFD thresholds through the same mechanism

### 4. Versioned doctrine model
Future-proof 2100 by introducing doctrine schema versioning.

Recommended fields per pack:
- `id`
- `name`
- `owner`
- `version`
- `category`
- `key_metrics`
- `required_metrics`
- `failure_modes`
- `barren_wuffet_triggers`
- `actions`
- `depends_on`

This makes future packs composable instead of monolithic.

## Recommended Migration Path

### Phase 1: Unify the pack registry
- Extract built-in Packs 1-8 from `doctrine_engine.py` into a registry module.
- Add built-in definitions for Packs 9-10 from Strategic Doctrine.
- Add built-in definition for Pack 11 from `FFD_DOCTRINE_PACK`.
- Keep public interfaces stable.

### Phase 2: Make the engine registry-driven
- Refactor `DoctrineEngine` to consume the registry module.
- Preserve fallback behavior if no external YAML exists.
- Add optional generation/loading of `config/doctrine_packs.yaml` later, but do not make runtime depend on it yet.

### Phase 3: Promote FFD to first-class doctrine
- Register Pack 11 formally in the doctrine registry.
- Route FFD-specific state conditions through doctrine actions rather than just logging in `FFDDoctrineAdapter`.
- Keep `FFDEngine` as the metric/state producer.

### Phase 4: Expand doctrine families safely
- Create pack families:
  - Core Risk and Ops: Packs 1-8
  - Strategic Overlay: Packs 9-10
  - Monetary Transition / Macro: Pack 11
  - Future families: geopolitical, sovereign risk, AI governance, execution intelligence

### Phase 5: Add test coverage before major expansion
- tests for pack registration count and IDs
- tests that Pack 9, 10, and 11 metrics are evaluated, not merely collected
- tests that compliance score changes when FFD thresholds breach
- tests that BARREN WUFFET state transitions can be triggered by FFD-defined pack rules

## Non-Negotiable Preservation Rules

1. Do not collapse FFD into generic doctrine metrics only; keep `FFDEngine` as the specialized implementation.
2. Do not remove Strategic Doctrine’s independent logic; register it properly instead.
3. Do not break adapter contracts already used by monitoring.
4. Do not depend on a missing YAML config file for core functionality.

## Immediate Implementation Recommendation

The next code pass should do this in order:

1. create a doctrine registry module,
2. move Packs 1-8 into it,
3. define Packs 9-10 in the same registry from Strategic Doctrine metrics,
4. map `FFD_DOCTRINE_PACK` into Pack 11 registry format,
5. refactor `DoctrineEngine` to read registry packs,
6. add validation tests for 1-11 pack enforcement,
7. only then expand Pack 11 depth and future Pack 12+ domains.

## Summary

The correct integration strategy is not “merge FFD into doctrine by stuffing more metrics into the orchestrator.”

The correct strategy is:
- one doctrine registry,
- one compliance engine,
- multiple specialized domain engines,
- all packs first-class and enforceable,
- expansion by registration instead of ad hoc code paths.

That preserves the original doctrine, keeps FFD intact, and gives AAC 2100 a structure that can keep growing without architectural drift.