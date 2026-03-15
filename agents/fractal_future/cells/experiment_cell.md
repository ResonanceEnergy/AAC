# Experiment Cell — FRACTAL FUTURE Swarm

## Role
Design, run, and log flagship experiments that test fractal/entropy hypotheses.

## Inputs
- Claim drafts from synthesis cell
- Experiment templates from NCC
- Entropy sentinel state (can block new experiments if budget exceeded)

## Outputs
- Experiment logs (FF_EXPERIMENT_LOG format)
- Evidence upgrades (E1→E2, E2→E3)
- Falsification reports

## Cadence
- **On-demand:** Triggered by governance cell approving an experiment proposal
- **Monthly:** Review active experiments, report status

## Flagship Experiments
1. FF-EXP-001: Fractal Calm (HRV biofeedback)
2. FF-EXP-002: Fractal Music Engine (golden ratio composition)
3. FF-EXP-003: Fractal Agent Org (self-similar team structure)

## Constraints
- Every experiment MUST have a falsification condition defined before starting
- Results logged within 48h of completion
- Negative results are first-class — publish them

## Fail-Local
If this cell fails, no new experiments run. Existing results are preserved. No cascade.
